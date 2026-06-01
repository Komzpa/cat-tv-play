#!/usr/bin/env python3
"""Build meow-to-projector feedback ledgers and training rows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_FEEDBACK_SPEC = importlib.util.spec_from_file_location(
    "cat_tv_play_projector_feedback",
    REPO_ROOT / "custom_components" / "cat_tv_play" / "projector_feedback.py",
)
assert _FEEDBACK_SPEC is not None
assert _FEEDBACK_SPEC.loader is not None
feedback = importlib.util.module_from_spec(_FEEDBACK_SPEC)
sys.modules[_FEEDBACK_SPEC.name] = feedback
_FEEDBACK_SPEC.loader.exec_module(feedback)

STATE_ROOT = Path("~/.openclaw/state/cat-tv-learning").expanduser()
DEFAULT_EVENT_ROOT = Path("/srv/cold-storage/hass/audio/vigi_audio_events")


def _latest_outcome_ledger() -> Path:
    root = STATE_ROOT / "label-review" / "meow-outcomes"
    ledgers = sorted(root.glob("*_meow_projector_outcomes.jsonl"))
    if not ledgers:
        raise FileNotFoundError(f"no meow outcome ledgers found under {root}")
    return ledgers[-1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=None, help="Existing meow outcome JSONL ledger.")
    parser.add_argument("--recordings-root", type=Path, default=STATE_ROOT / "recordings")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for feedback.jsonl, training_rows.jsonl, and summary.json.",
    )
    parser.add_argument("--recording-match-window-seconds", type=float, default=180.0)
    parser.add_argument(
        "--write-sidecars",
        action="store_true",
        help=(
            "Merge derived cat_projector_feedback blocks into source meow sidecars. "
            "Without this, only files under --out-dir are written."
        ),
    )
    parser.add_argument(
        "--write-database",
        action="store_true",
        help="Write derived feedback as normal Postgres acoustic_event_classifications facts.",
    )
    parser.add_argument(
        "--database-url",
        default="",
        help="Postgres URL for --write-database, for example postgresql:///openclaw_acoustic.",
    )
    parser.add_argument(
        "--event-root",
        type=Path,
        default=DEFAULT_EVENT_ROOT,
        help="Root used to map source sidecar paths to acoustic_events.relative_path for --write-database.",
    )
    parser.add_argument(
        "--write-unknown-sidecars",
        action="store_true",
        help="Also write unknown-only feedback entries into sidecars. Default keeps ambiguous rows only in the ledger.",
    )
    parser.add_argument(
        "--require-training-ready",
        action="store_true",
        help="Exit non-zero when derived projector feedback is one-class and should not be promoted to a model.",
    )
    return parser


PROJECTOR_CLASS_BY_LABEL = {
    "positive": "cat_meow_projector_play",
    "negative": "cat_meow_projector_no_play",
    "unknown": "cat_meow_projector_unknown",
}


def write_database_feedback(
    feedback_rows: list[dict[str, object]],
    *,
    database_url: str,
    event_root: Path,
) -> dict[str, object]:
    if not database_url:
        raise ValueError("--database-url is required with --write-database")
    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError as exc:  # pragma: no cover - depends on deployment venv.
        raise RuntimeError("psycopg is required for --write-database") from exc

    event_root = event_root.expanduser().resolve()
    payloads: list[dict[str, object]] = []
    skipped_missing_sidecar_path = 0
    skipped_unknown_label = 0
    for feedback_row in feedback_rows:
        meow = feedback_row.get("meow") if isinstance(feedback_row.get("meow"), dict) else {}
        sidecar_path_raw = meow.get("sidecar_path") if isinstance(meow, dict) else ""
        if not sidecar_path_raw:
            skipped_missing_sidecar_path += 1
            continue
        sidecar_path = Path(str(sidecar_path_raw)).expanduser().resolve()
        try:
            relative_path = str(sidecar_path.relative_to(event_root))
        except ValueError as exc:
            raise ValueError(f"sidecar path is outside --event-root: {sidecar_path}") from exc
        labels = (
            feedback_row.get("training", {}).get("labels", {}) if isinstance(feedback_row.get("training"), dict) else {}
        )
        training_value = str(labels.get(feedback.PLAY_LABEL) or "").strip().lower()
        class_key = PROJECTOR_CLASS_BY_LABEL.get(training_value)
        if class_key is None:
            skipped_unknown_label += 1
            continue
        behavior_outcome = feedback_row.get("behavior_outcome") or {}
        try:
            confidence = float(behavior_outcome.get("confidence")) if isinstance(behavior_outcome, dict) else None
        except (TypeError, ValueError):
            confidence = None
        payloads.append(
            {
                "feedback": feedback_row,
                "relative_path": relative_path,
                "class_key": class_key,
                "source": "cat_projector_feedback",
                "rank": 0,
                "confidence": confidence,
            }
        )

    with psycopg.connect(database_url) as conn:
        class_set_row = conn.execute(
            """
            select id
            from acoustic_class_sets
            where taxonomy_key = 'default'
            order by created_at desc, id desc
            limit 1
            """
        ).fetchone()
        if class_set_row is None:
            raise RuntimeError("no default acoustic class set; run home-audio-mesh ensure-db-indexes first")
        class_set_id = int(class_set_row[0])
        known_classes = {
            str(row[0])
            for row in conn.execute(
                "select class_key from acoustic_class_definitions where class_set_id = %s",
                (class_set_id,),
            )
        }
        missing_classes = sorted(set(PROJECTOR_CLASS_BY_LABEL.values()) - known_classes)
        if missing_classes:
            raise RuntimeError(
                "default acoustic class set is missing projector classes; "
                f"run home-audio-mesh ensure-db-indexes first: {missing_classes}"
            )

        found_rows = conn.execute(
            "select relative_path from acoustic_events where relative_path = any(%s)",
            ([row["relative_path"] for row in payloads],),
        ).fetchall()
        existing_relative_paths = {str(row[0]) for row in found_rows}
        db_payloads = [
            {
                **row,
                "class_set_id": class_set_id,
                "payload": Jsonb(row["feedback"]),
            }
            for row in payloads
            if str(row["relative_path"]) in existing_relative_paths
        ]

        with conn.cursor() as cur:
            cur.executemany(
                """
                insert into acoustic_event_classifications (
                  relative_path, class_set_id, class_key, source, rank,
                  confidence, payload, updated_at
                )
                values (
                  %(relative_path)s, %(class_set_id)s, %(class_key)s, %(source)s,
                  %(rank)s, %(confidence)s, %(payload)s, now()
                )
                on conflict (relative_path, class_set_id, class_key, source) do update set
                  rank = excluded.rank,
                  confidence = excluded.confidence,
                  payload = excluded.payload,
                  updated_at = now()
                """,
                db_payloads,
            )
            classification_upserts = max(cur.rowcount, 0)

            stale_pairs = [
                {
                    "relative_path": row["relative_path"],
                    "class_set_id": class_set_id,
                    "source": "cat_projector_feedback",
                    "class_keys": [
                        class_key for class_key in PROJECTOR_CLASS_BY_LABEL.values() if class_key != row["class_key"]
                    ],
                }
                for row in db_payloads
            ]
            cur.executemany(
                """
                delete from acoustic_event_classifications
                where relative_path = %(relative_path)s
                  and class_set_id = %(class_set_id)s
                  and source = %(source)s
                  and class_key = any(%(class_keys)s)
                """,
                stale_pairs,
            )
            stale_classification_deletes = max(cur.rowcount, 0)
        conn.commit()

    label_counts: dict[str, int] = {}
    for row in payloads:
        label_counts[str(row["class_key"])] = label_counts.get(str(row["class_key"]), 0) + 1
    return {
        "candidate_feedback_rows": len(feedback_rows),
        "payload_rows": len(payloads),
        "skipped_missing_sidecar_path": skipped_missing_sidecar_path,
        "skipped_unmapped_label": skipped_unknown_label,
        "missing_acoustic_events": len(payloads) - len(db_payloads),
        "classification_upserts": classification_upserts,
        "stale_classification_deletes": stale_classification_deletes,
        "class_counts": label_counts,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outcomes_path = args.outcomes.expanduser() if args.outcomes else _latest_outcome_ledger()
    out_dir = (
        args.out_dir.expanduser()
        if args.out_dir
        else (STATE_ROOT / "label-review" / "meow-feedback" / outcomes_path.stem)
    )
    recordings_root = args.recordings_root.expanduser()

    outcome_rows = feedback.iter_jsonl(outcomes_path)
    feedback_rows = feedback.build_feedback_rows(
        outcome_rows,
        recordings_root=recordings_root,
        recording_match_window_seconds=args.recording_match_window_seconds,
    )
    training_rows = feedback.export_training_rows(feedback_rows)
    summary = feedback.summarize_feedback(feedback_rows)
    summary.update(
        {
            "outcomes_path": str(outcomes_path),
            "recordings_root": str(recordings_root),
            "out_dir": str(out_dir),
            "sidecar_write_requested": bool(args.write_sidecars),
        }
    )

    feedback.write_jsonl(out_dir / "feedback.jsonl", feedback_rows)
    feedback.write_jsonl(out_dir / "training_rows.jsonl", training_rows)

    if args.write_sidecars:
        summary["sidecar_write"] = feedback.write_sidecar_feedback(
            feedback_rows,
            backup_root=out_dir / "sidecar-backups",
            write_unknown=args.write_unknown_sidecars,
        )
    if args.write_database:
        summary["database_write"] = write_database_feedback(
            feedback_rows,
            database_url=args.database_url,
            event_root=args.event_root,
        )

    feedback.write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_training_ready and summary.get("deploy_blocked"):
        print("projector feedback training is not ready; see training_readiness in summary.json", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
