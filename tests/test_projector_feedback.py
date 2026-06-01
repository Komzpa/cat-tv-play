from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "cat_tv_play" / "projector_feedback.py"
SPEC = spec_from_file_location("cat_tv_play_projector_feedback", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
projector_feedback = module_from_spec(SPEC)
sys.modules[SPEC.name] = projector_feedback
SPEC.loader.exec_module(projector_feedback)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_meow_feedback.py"
SCRIPT_SPEC = spec_from_file_location("cat_projector_meow_feedback_script", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
feedback_script = module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = feedback_script
SCRIPT_SPEC.loader.exec_module(feedback_script)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _outcome_row(event_jsonl: Path, *, line: int = 1) -> dict:
    return {
        "schema": "cat_projector_meow_outcome.v1",
        "event_at": "2026-05-18T10:00:00Z",
        "event_at_local": "2026-05-18T14:00:00+04:00",
        "matched_session_start_at": "2026-05-18T10:00:01Z",
        "matched_session_start_delta_seconds": 1.0,
        "outcome": "started_projector",
        "policy_decision": "allow_start",
        "source_id": "living_room_vigi",
        "meow_score": 0.91,
        "intent_label": "attention",
        "intent_confidence": 0.7,
        "intent_play_probability": 0.2,
        "representative_source_path": str(event_jsonl),
        "representative_source_line": line,
    }


def test_backfill_jump_highlight_marks_play_projector_without_overwriting_intent(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "audio" / "meow.json"
    _write_json(sidecar_path, {"sher_meow_intent": {"label": "attention", "source": "human"}})
    event_jsonl = tmp_path / "events.jsonl"
    event_jsonl.write_text(json.dumps({"sidecar_path": str(sidecar_path)}, ensure_ascii=False) + "\n", encoding="utf-8")

    recording_dir = tmp_path / "recordings" / "session"
    _write_json(recording_dir / "manifest.json", {"created_at": "2026-05-18T10:00:02Z"})
    _write_json(
        recording_dir / "telegram_notification.json",
        {"jump_highlight": {"height_cm": 120.0}, "max_height_cm": 120.0},
    )

    rows = projector_feedback.build_feedback_rows([_outcome_row(event_jsonl)], recordings_root=tmp_path / "recordings")

    assert rows[0]["behavior_outcome"]["label"] == "active_play"
    assert rows[0]["training"]["labels"] == {"meow: play_projector": "positive"}

    stats = projector_feedback.write_sidecar_feedback(rows, backup_root=tmp_path / "backups")
    updated = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert stats["sidecars_updated"] == 1
    assert updated["sher_meow_intent"]["label"] == "attention"
    assert updated["cat_projector_feedback"]["training_labels"]["meow: play_projector"] == "positive"
    assert updated["cat_projector_feedback"]["label_policy"] == (
        "derived_projector_feedback_does_not_overwrite_sher_meow_intent"
    )


def test_human_not_near_projector_marks_play_projector_negative(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "meow.json"
    _write_json(sidecar_path, {})
    event_jsonl = tmp_path / "events.jsonl"
    event_jsonl.write_text(json.dumps({"sidecar_path": str(sidecar_path)}, ensure_ascii=False) + "\n", encoding="utf-8")
    recording_dir = tmp_path / "recordings" / "session"
    _write_json(recording_dir / "manifest.json", {"created_at": "2026-05-18T10:00:01Z"})
    _write_json(recording_dir / "verified.json", {"verified_chunk_frames": 100})
    _write_json(recording_dir / "meow_projector_feedback.json", {"behavior_outcome": "not_near_projector"})

    rows = projector_feedback.build_feedback_rows([_outcome_row(event_jsonl)], recordings_root=tmp_path / "recordings")

    assert rows[0]["behavior_outcome"]["label"] == "not_near_projector"
    assert rows[0]["training"]["labels"] == {"meow: play_projector": "negative"}


def test_started_projector_no_jump_highlight_marks_no_active_play_negative(tmp_path: Path) -> None:
    event_jsonl = tmp_path / "events.jsonl"
    event_jsonl.write_text(json.dumps({"sidecar_path": str(tmp_path / "missing.json")}) + "\n", encoding="utf-8")
    recording_dir = tmp_path / "recordings" / "session"
    _write_json(recording_dir / "manifest.json", {"created_at": "2026-05-18T10:00:01Z"})
    _write_json(recording_dir / "verified.json", {"verified_chunk_frames": 100})
    _write_json(recording_dir / "telegram_notification.json", {"selection_method": "no_jump_highlight"})

    rows = projector_feedback.build_feedback_rows([_outcome_row(event_jsonl)], recordings_root=tmp_path / "recordings")

    assert rows[0]["behavior_outcome"]["label"] == "projector_started_no_active_play"
    assert rows[0]["training"]["labels"] == {"meow: play_projector": "negative"}


def test_unlinked_no_jump_highlight_stays_unknown(tmp_path: Path) -> None:
    event_jsonl = tmp_path / "events.jsonl"
    event_jsonl.write_text(json.dumps({"sidecar_path": str(tmp_path / "missing.json")}) + "\n", encoding="utf-8")
    recording_dir = tmp_path / "recordings" / "session"
    _write_json(recording_dir / "manifest.json", {"created_at": "2026-05-18T10:00:01Z"})
    _write_json(recording_dir / "verified.json", {"verified_chunk_frames": 100})
    _write_json(recording_dir / "telegram_notification.json", {"selection_method": "no_jump_highlight"})
    row = _outcome_row(event_jsonl)
    row["outcome"] = "no_projector_start"
    row["matched_session_start_at"] = None
    row["matched_session_start_delta_seconds"] = None

    rows = projector_feedback.build_feedback_rows([row], recordings_root=tmp_path / "recordings")

    assert rows[0]["behavior_outcome"]["label"] == "unknown"
    assert rows[0]["training"]["labels"] == {"meow: play_projector": "unknown"}


def test_watching_only_is_not_a_hard_negative(tmp_path: Path) -> None:
    event_jsonl = tmp_path / "events.jsonl"
    event_jsonl.write_text("{}\n", encoding="utf-8")
    recording_dir = tmp_path / "recordings" / "session"
    _write_json(recording_dir / "manifest.json", {"created_at": "2026-05-18T10:00:01Z"})
    _write_json(recording_dir / "meow_projector_feedback.json", {"behavior_outcome": "watching_only"})

    rows = projector_feedback.build_feedback_rows([_outcome_row(event_jsonl)], recordings_root=tmp_path / "recordings")

    assert rows[0]["training"]["labels"] == {"meow: play_projector": "unknown"}


def test_no_projector_start_near_jump_does_not_become_play_projector(tmp_path: Path) -> None:
    event_jsonl = tmp_path / "events.jsonl"
    event_jsonl.write_text("{}\n", encoding="utf-8")
    recording_dir = tmp_path / "recordings" / "session"
    _write_json(recording_dir / "manifest.json", {"created_at": "2026-05-18T10:00:01Z"})
    _write_json(recording_dir / "telegram_notification.json", {"jump_highlight": {"height_cm": 120.0}})
    row = _outcome_row(event_jsonl)
    row["outcome"] = "no_projector_start"
    row["matched_session_start_at"] = None
    row["matched_session_start_delta_seconds"] = None

    rows = projector_feedback.build_feedback_rows([row], recordings_root=tmp_path / "recordings")

    assert rows[0]["behavior_outcome"]["label"] == "active_play"
    assert rows[0]["projector_attempt"]["linked_to_session"] is False
    assert rows[0]["training"]["labels"] == {"meow: play_projector": "unknown"}


def test_positive_only_projector_feedback_blocks_training_promotion(tmp_path: Path) -> None:
    event_jsonl = tmp_path / "events.jsonl"
    event_jsonl.write_text("{}\n", encoding="utf-8")
    row = _outcome_row(event_jsonl)
    evidence = projector_feedback.RecordingEvidence(
        None,
        "active_play",
        0.95,
        "auto_high_confidence_jump",
        {},
        "test active play",
    )

    summary = projector_feedback.summarize_feedback([projector_feedback.build_feedback_row(row, evidence)])

    assert summary["deploy_blocked"] is True
    assert summary["training_readiness"]["meow: play_projector"]["ready"] is False
    assert summary["training_readiness"]["meow: play_projector"]["blocked_reason"] == "missing_negative_examples"


def test_projector_feedback_training_ready_requires_active_play_and_not_play(tmp_path: Path) -> None:
    event_jsonl = tmp_path / "events.jsonl"
    event_jsonl.write_text("{}\n", encoding="utf-8")
    active_row = _outcome_row(event_jsonl)
    not_near_row = _outcome_row(event_jsonl)
    active = projector_feedback.RecordingEvidence(None, "active_play", 0.95, "human_review", {}, "test active play")
    not_near = projector_feedback.RecordingEvidence(
        None,
        "projector_started_no_active_play",
        1.0,
        "human_review",
        {},
        "test no approach",
    )

    rows = [
        projector_feedback.build_feedback_row(active_row, active),
        projector_feedback.build_feedback_row(not_near_row, not_near),
    ]
    summary = projector_feedback.summarize_feedback(rows)

    assert summary["training_readiness"]["meow: play_projector"]["ready"] is True
    assert summary["deploy_blocked"] is False


def test_unknown_projector_rows_do_not_satisfy_training_negative() -> None:
    summary = projector_feedback.summarize_feedback(
        [
            {
                "behavior_outcome": {"label": "unknown", "source": "verified_recording_no_jump_highlight"},
                "training": {
                    "labels": {
                        "meow: play_projector": "unknown",
                    }
                },
            }
        ]
    )

    assert summary["training"]["meow: play_projector"]["abstained"] == 1
    assert "unknown" not in summary["training"]["meow: play_projector"]
    assert summary["training_readiness"]["meow: play_projector"]["negative"] == 0
    assert summary["training_readiness"]["meow: play_projector"]["abstained"] == 1
    assert summary["training_readiness"]["meow: play_projector"]["ready"] is False


def test_database_feedback_writes_normal_acoustic_classification_fact(monkeypatch, tmp_path: Path) -> None:
    event_root = tmp_path / "events"
    sidecar_path = event_root / "living_room_vigi" / "meow.json"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text("{}", encoding="utf-8")
    executed_many: list[tuple[str, list[dict[str, object]]]] = []

    class Jsonb:
        def __init__(self, value):
            self.value = value

    class Rows:
        def __init__(self, *, one=None, many=None):
            self.one = one
            self.many = many or []

        def fetchone(self):
            return self.one

        def fetchall(self):
            return self.many

        def __iter__(self):
            return iter(self.many)

    class Cursor:
        rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def executemany(self, sql, params):
            batch = list(params)
            executed_many.append((str(sql), batch))
            self.rowcount = len(batch)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def execute(self, sql, params=None):
            if "from acoustic_class_sets" in str(sql):
                return Rows(one=(8,))
            if "from acoustic_class_definitions" in str(sql):
                return Rows(many=[(value,) for value in feedback_script.PROJECTOR_CLASS_BY_LABEL.values()])
            if "from acoustic_events" in str(sql):
                return Rows(many=[("living_room_vigi/meow.json",)])
            raise AssertionError(sql)

        def cursor(self):
            return Cursor()

        def commit(self):
            pass

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda _url: Connection()))
    monkeypatch.setitem(sys.modules, "psycopg.types", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "psycopg.types.json", SimpleNamespace(Jsonb=Jsonb))

    summary = feedback_script.write_database_feedback(
        [
            {
                "meow": {"sidecar_path": str(sidecar_path)},
                "training": {"labels": {"meow: play_projector": "negative"}},
                "behavior_outcome": {"confidence": 0.83},
            }
        ],
        database_url="postgresql:///unit",
        event_root=event_root,
    )

    insert_sql, insert_rows = executed_many[0]
    assert "insert into acoustic_event_classifications" in insert_sql
    assert "cat_projector_meow_feedback" not in insert_sql
    assert insert_rows[0]["class_key"] == "cat_meow_projector_no_play"
    assert insert_rows[0]["source"] == "cat_projector_feedback"
    assert summary["classification_upserts"] == 1
    assert summary["class_counts"] == {"cat_meow_projector_no_play": 1}
