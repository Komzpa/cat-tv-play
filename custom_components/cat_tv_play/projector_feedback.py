"""Meow-to-projector feedback labels for Cat TV active learning."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FEEDBACK_SCHEMA = "cat_projector_meow_feedback.v1"
SIDECAR_SCHEMA = "cat_projector_feedback.v1"
PLAY_LABEL = "meow: play_projector"

ACTIVE_PLAY_OUTCOMES = {"active_play"}
NO_ACTIVE_PLAY_OUTCOME = "projector_started_no_active_play"
NOT_PLAY_OUTCOMES = {"not_near_projector", NO_ACTIVE_PLAY_OUTCOME}
NEUTRAL_OUTCOMES = {"engaged_watch", "near_projector_no_play", "watching_only"}
TRAINING_TARGETS = (PLAY_LABEL,)


@dataclass(frozen=True)
class RecordingEvidence:
    recording_dir: Path | None
    outcome: str
    confidence: float
    source: str
    evidence_refs: dict[str, str]
    notes: str


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl_line(path: Path, line_number: int) -> dict[str, Any] | None:
    if line_number <= 0 or not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if index == line_number:
                return json.loads(line)
    return None


def meow_event_key(row: dict[str, Any]) -> str:
    source_path = str(row.get("representative_source_path") or "")
    source_line = str(row.get("representative_source_line") or "")
    event_at = str(row.get("event_at") or "")
    sidecar_path = str(row.get("sidecar_path") or "")
    raw_events = row.get("raw_events")
    if isinstance(raw_events, list):
        for raw in raw_events:
            if isinstance(raw, dict) and raw.get("sidecar_path"):
                sidecar_path = str(raw["sidecar_path"])
                break
    key = "|".join(part for part in (sidecar_path, source_path, source_line, event_at) if part)
    if key:
        return key
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def feedback_id(row: dict[str, Any], recording_dir: Path | None) -> str:
    raw = "|".join(
        [
            meow_event_key(row),
            str(row.get("matched_session_start_at") or ""),
            str(recording_dir or ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def resolve_sidecar_path(row: dict[str, Any]) -> Path | None:
    direct = row.get("sidecar_path")
    if direct:
        return Path(str(direct)).expanduser()

    raw_events = row.get("raw_events")
    if isinstance(raw_events, list):
        for raw in raw_events:
            if isinstance(raw, dict) and raw.get("sidecar_path"):
                return Path(str(raw["sidecar_path"])).expanduser()

    source_path = row.get("representative_source_path")
    source_line = row.get("representative_source_line")
    if not source_path or source_line in (None, ""):
        return None
    try:
        line_number = int(source_line)
    except (TypeError, ValueError):
        return None
    event = read_jsonl_line(Path(str(source_path)).expanduser(), line_number)
    if not event:
        return None
    sidecar_path = event.get("sidecar_path")
    return Path(str(sidecar_path)).expanduser() if sidecar_path else None


def _recording_created_at(recording_dir: Path) -> datetime | None:
    manifest_path = recording_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    return parse_timestamp(manifest.get("created_at"))


def index_recordings(recordings_root: Path) -> list[tuple[Path, datetime]]:
    if not recordings_root.exists():
        return []
    rows: list[tuple[Path, datetime]] = []
    for manifest_path in sorted(recordings_root.glob("*/manifest.json")):
        created_at = _recording_created_at(manifest_path.parent)
        if created_at is not None:
            rows.append((manifest_path.parent, created_at))
    return rows


def match_recording_dir(
    row: dict[str, Any],
    recording_index: list[tuple[Path, datetime]],
    *,
    max_delta_seconds: float,
) -> Path | None:
    explicit = row.get("recording_dir") or row.get("source_recording_dir")
    if explicit:
        path = Path(str(explicit)).expanduser()
        if path.exists():
            return path

    session_at = parse_timestamp(row.get("matched_session_start_at")) or parse_timestamp(row.get("event_at"))
    if session_at is None:
        return None

    best: tuple[float, Path] | None = None
    for recording_dir, created_at in recording_index:
        delta = abs((created_at - session_at).total_seconds())
        if delta <= max_delta_seconds and (best is None or delta < best[0]):
            best = (delta, recording_dir)
    return best[1] if best else None


def _human_feedback(recording_dir: Path) -> dict[str, Any] | None:
    for name in ("meow_projector_feedback.json", "behavior_feedback.json", "human_feedback.json"):
        path = recording_dir / name
        if path.exists():
            payload = read_json(path)
            if isinstance(payload, dict):
                payload["_feedback_path"] = str(path)
                return payload
    return None


def _notification_evidence(recording_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    for name in ("telegram_notification.json", "telegram_live_notification.json"):
        path = recording_dir / name
        if path.exists():
            payload = read_json(path)
            if isinstance(payload, dict):
                return payload, path
    return None, None


def infer_recording_evidence(recording_dir: Path | None) -> RecordingEvidence:
    if recording_dir is None:
        return RecordingEvidence(None, "unknown", 0.0, "no_recording_match", {}, "no matched projector recording")

    human = _human_feedback(recording_dir)
    if human:
        outcome = str(human.get("behavior_outcome") or human.get("outcome") or "unknown")
        confidence = _float_or_default(human.get("confidence"), 1.0)
        path = str(human.get("_feedback_path") or "")
        return RecordingEvidence(
            recording_dir,
            outcome,
            confidence,
            "human_review",
            {"human_feedback": path} if path else {},
            str(human.get("notes") or "human-reviewed projector outcome"),
        )

    notification, notification_path = _notification_evidence(recording_dir)
    if notification:
        if notification.get("jump_highlight") or _float_or_default(notification.get("max_height_cm"), 0.0) > 0:
            return RecordingEvidence(
                recording_dir,
                "active_play",
                0.95,
                "auto_high_confidence_jump",
                {"notification": str(notification_path)} if notification_path else {},
                "high-confidence jump/paw projector highlight",
            )
        if notification.get("selection_method") == "no_jump_highlight":
            return RecordingEvidence(
                recording_dir,
                "unknown",
                0.35,
                "verified_recording_no_jump_highlight",
                {"notification": str(notification_path)} if notification_path else {},
                "no jump highlight is not enough evidence for not_play",
            )

    verified_path = recording_dir / "verified.json"
    if verified_path.exists():
        return RecordingEvidence(
            recording_dir,
            "unknown",
            0.2,
            "verified_recording_without_behavior_label",
            {"verified": str(verified_path)},
            "recording exists but behavior has not been reviewed",
        )

    return RecordingEvidence(
        recording_dir,
        "unknown",
        0.0,
        "recording_without_coverage_proof",
        {},
        "missing coverage proof",
    )


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def training_labels_for_outcome(outcome: str) -> dict[str, str]:
    if outcome in ACTIVE_PLAY_OUTCOMES:
        return {PLAY_LABEL: "positive"}
    if outcome in NOT_PLAY_OUTCOMES:
        return {PLAY_LABEL: "negative"}
    if outcome in NEUTRAL_OUTCOMES:
        return {PLAY_LABEL: "unknown"}
    return {PLAY_LABEL: "unknown"}


def projector_attempt_linked(row: dict[str, Any]) -> bool:
    if row.get("matched_session_start_at"):
        return True
    outcome = str(row.get("outcome") or "")
    return outcome in {"started_projector", "projector_started"}


def projector_started_by_meow(row: dict[str, Any]) -> bool:
    return str(row.get("outcome") or "") in {"started_projector", "projector_started"}


def training_evidence_for_attempt(row: dict[str, Any], evidence: RecordingEvidence) -> RecordingEvidence:
    if (
        projector_started_by_meow(row)
        and evidence.outcome == "unknown"
        and evidence.source == "verified_recording_no_jump_highlight"
    ):
        return RecordingEvidence(
            evidence.recording_dir,
            NO_ACTIVE_PLAY_OUTCOME,
            evidence.confidence,
            evidence.source,
            evidence.evidence_refs,
            "projector started from meow; verified projector recording had no active play highlight",
        )
    return evidence


def build_feedback_row(
    row: dict[str, Any],
    evidence: RecordingEvidence,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    sidecar_path = resolve_sidecar_path(row)
    linked_attempt = projector_attempt_linked(row)
    evidence = training_evidence_for_attempt(row, evidence)
    labels = training_labels_for_outcome(evidence.outcome) if linked_attempt else {PLAY_LABEL: "unknown"}
    raw_events = row.get("raw_events") if isinstance(row.get("raw_events"), list) else []
    intent_hint = {
        "label": row.get("intent_label"),
        "confidence": row.get("intent_confidence"),
        "play_probability": row.get("intent_play_probability"),
    }
    payload = {
        "schema": FEEDBACK_SCHEMA,
        "feedback_id": feedback_id(row, evidence.recording_dir),
        "created_at": created_at or utc_now_iso(),
        "meow": {
            "event_key": meow_event_key(row),
            "event_at": row.get("event_at"),
            "event_at_local": row.get("event_at_local"),
            "source_id": row.get("source_id"),
            "sidecar_path": str(sidecar_path) if sidecar_path else "",
            "source_path": row.get("representative_source_path"),
            "source_line": row.get("representative_source_line"),
            "meow_score": row.get("meow_score"),
            "broad_acoustic_label": row.get("broad_acoustic_label"),
            "broad_acoustic_confidence": row.get("broad_acoustic_confidence"),
            "intent_hint": intent_hint,
            "raw_event_count": row.get("raw_event_count") or len(raw_events),
        },
        "projector_attempt": {
            "outcome": row.get("outcome"),
            "policy_decision": row.get("policy_decision"),
            "linked_to_session": linked_attempt,
            "matched_session_start_at": row.get("matched_session_start_at"),
            "matched_session_start_delta_seconds": row.get("matched_session_start_delta_seconds"),
        },
        "behavior_outcome": {
            "label": evidence.outcome,
            "confidence": evidence.confidence,
            "source": evidence.source,
            "notes": evidence.notes,
            "recording_dir": str(evidence.recording_dir) if evidence.recording_dir else "",
            "evidence_refs": evidence.evidence_refs,
        },
        "training": {
            "labels": labels,
            "label_policy": "derived_projector_feedback_does_not_overwrite_sher_meow_intent",
        },
    }
    return payload


def build_feedback_rows(
    outcome_rows: list[dict[str, Any]],
    *,
    recordings_root: Path,
    recording_match_window_seconds: float = 180.0,
) -> list[dict[str, Any]]:
    recording_index = index_recordings(recordings_root)
    feedback_rows: list[dict[str, Any]] = []
    for row in outcome_rows:
        recording_dir = match_recording_dir(
            row,
            recording_index,
            max_delta_seconds=recording_match_window_seconds,
        )
        feedback_rows.append(build_feedback_row(row, infer_recording_evidence(recording_dir)))
    return feedback_rows


def export_training_rows(feedback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feedback in feedback_rows:
        meow = feedback.get("meow") if isinstance(feedback.get("meow"), dict) else {}
        labels = feedback.get("training", {}).get("labels", {})
        if not isinstance(labels, dict):
            continue
        for label_name, label_value in sorted(labels.items()):
            if label_value == "unknown":
                continue
            rows.append(
                {
                    "feedback_id": feedback.get("feedback_id"),
                    "label": label_name,
                    "value": label_value,
                    "sidecar_path": meow.get("sidecar_path", ""),
                    "event_at": meow.get("event_at", ""),
                    "source_id": meow.get("source_id", ""),
                    "behavior_outcome": feedback.get("behavior_outcome", {}).get("label", ""),
                    "evidence_source": feedback.get("behavior_outcome", {}).get("source", ""),
                }
            )
    return rows


def merge_feedback_into_sidecar(sidecar: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    updated = dict(sidecar)
    existing = updated.get("cat_projector_feedback")
    block = dict(existing) if isinstance(existing, dict) else {}
    block["schema_version"] = SIDECAR_SCHEMA
    entries = block.get("entries")
    if not isinstance(entries, list):
        entries = []

    feedback_entry = {
        "feedback_id": feedback.get("feedback_id"),
        "created_at": feedback.get("created_at"),
        "behavior_outcome": feedback.get("behavior_outcome"),
        "projector_attempt": feedback.get("projector_attempt"),
        "training": feedback.get("training"),
    }
    without_same = [
        entry
        for entry in entries
        if not isinstance(entry, dict) or entry.get("feedback_id") != feedback.get("feedback_id")
    ]
    without_same.append(feedback_entry)
    block["entries"] = without_same
    block["latest_feedback_id"] = feedback.get("feedback_id")
    block["training_labels"] = feedback.get("training", {}).get("labels", {})
    block["label_policy"] = "derived_projector_feedback_does_not_overwrite_sher_meow_intent"
    updated["cat_projector_feedback"] = block
    return updated


def write_sidecar_feedback(
    feedback_rows: list[dict[str, Any]],
    *,
    backup_root: Path | None = None,
    write_unknown: bool = False,
) -> dict[str, int]:
    stats = {
        "candidate_feedback_rows": 0,
        "sidecars_updated": 0,
        "sidecars_missing": 0,
        "unknown_skipped": 0,
    }
    for feedback in feedback_rows:
        labels = feedback.get("training", {}).get("labels", {})
        if not isinstance(labels, dict):
            continue
        if not write_unknown and all(value == "unknown" for value in labels.values()):
            stats["unknown_skipped"] += 1
            continue
        stats["candidate_feedback_rows"] += 1
        sidecar_path_raw = feedback.get("meow", {}).get("sidecar_path")
        if not sidecar_path_raw:
            stats["sidecars_missing"] += 1
            continue
        sidecar_path = Path(str(sidecar_path_raw)).expanduser()
        if not sidecar_path.exists():
            stats["sidecars_missing"] += 1
            continue
        sidecar = read_json(sidecar_path)
        updated = merge_feedback_into_sidecar(sidecar, feedback)
        if backup_root is not None:
            backup_path = (
                backup_root / hashlib.sha256(str(sidecar_path).encode("utf-8")).hexdigest()[:16] / sidecar_path.name
            )
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(sidecar_path, backup_path)
        write_json(sidecar_path, updated)
        stats["sidecars_updated"] += 1
    return stats


def summarize_feedback(feedback_rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    training: dict[str, dict[str, int]] = {PLAY_LABEL: {}}
    evidence_sources: dict[str, int] = {}
    for row in feedback_rows:
        behavior = row.get("behavior_outcome") if isinstance(row.get("behavior_outcome"), dict) else {}
        outcome = str(behavior.get("label") or "unknown")
        source = str(behavior.get("source") or "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        evidence_sources[source] = evidence_sources.get(source, 0) + 1
        labels = row.get("training", {}).get("labels", {})
        if isinstance(labels, dict):
            for label_name, label_value in labels.items():
                bucket = training.setdefault(str(label_name), {})
                summary_value = "abstained" if str(label_value) == "unknown" else str(label_value)
                bucket[summary_value] = bucket.get(summary_value, 0) + 1
    readiness = training_readiness(training)
    return {
        "schema": f"{FEEDBACK_SCHEMA}.summary",
        "created_at": utc_now_iso(),
        "row_count": len(feedback_rows),
        "outcomes": outcomes,
        "training": training,
        "training_readiness": readiness,
        "deploy_blocked": any(not row["ready"] for row in readiness.values()),
        "evidence_sources": evidence_sources,
    }


def training_readiness(training: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    readiness: dict[str, dict[str, Any]] = {}
    for target in TRAINING_TARGETS:
        counts = training.get(target, {})
        positive = int(counts.get("positive", 0))
        negative = int(counts.get("negative", 0))
        reasons: list[str] = []
        if positive <= 0:
            reasons.append("missing_positive_examples")
        if negative <= 0:
            reasons.append("missing_negative_examples")
        readiness[target] = {
            "ready": not reasons,
            "positive": positive,
            "negative": negative,
            "abstained": int(counts.get("abstained", 0)),
            "blocked_reason": ",".join(reasons),
        }
    return readiness
