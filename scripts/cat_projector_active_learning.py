#!/usr/bin/env python3
"""Run one Cat Projector active-learning iteration.

The iteration is deliberately offline:

1. materialize saved review UI labels into a normal labels.csv package;
2. train the repo-owned candidate detector from all available labels.csv files;
3. rescore every reviewable input frame with the fresh model;
4. write probe_rows.json and an uncertainty-sorted queue for the next review pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import cat_projector_frame_detector as detector
from scripts import cat_projector_label_review_server as review

STATE_ROOT = Path("~/.openclaw/state/cat-tv-learning").expanduser()
RESCORES_ROOT = STATE_ROOT / "label-review" / "rescores"

_WORKER_MODEL: Any | None = None
_WORKER_METADATA: dict[str, Any] | None = None


def _utc_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _collect_labels(extra_labels: list[Path]) -> list[Path]:
    extra_resolved = {path.expanduser().resolve() for path in extra_labels if path.expanduser().exists()}
    candidates: list[Path] = []
    for root in (
        STATE_ROOT / "datasets",
        REPO_ROOT / "datasets" / "cat-tv-learning" / "detector-training",
        review.DATASET_ROOT / "detector-training",
    ):
        if root.exists():
            candidates.extend(sorted(root.glob("*/labels.csv")))
    candidates.extend(path.expanduser() for path in extra_labels)

    seen: set[Path] = set()
    labels: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        package_name = resolved.parent.name
        if resolved not in extra_resolved and (
            package_name.startswith("cat-projector-review-ui-")
            or package_name.startswith("cat-projector-ui-")
        ):
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        labels.append(resolved)
    return labels


def _materialize_review_training_package() -> dict[str, Any]:
    return review._materialize_review_labels_as_training_package(  # noqa: SLF001
        {
            "action": "retrain_model",
            "reason": "offline active-learning iteration",
        }
    )


def _train_model(labels: list[Path], *, model_path: Path, metadata_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    train_args = SimpleNamespace(
        labels=labels,
        out=model_path,
        metadata=metadata_path,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        detector.train(train_args)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _reviewable_frame_paths() -> list[Path]:
    paths: list[Path] = []
    for root in review.SCAN_ROOTS:
        if not root.exists():
            continue
        if root.name == "recordings":
            for recording_dir in root.iterdir():
                if recording_dir.is_dir():
                    paths.extend(review._recording_frame_paths(recording_dir))  # noqa: SLF001
            continue
        for group_key in review._iter_frame_group_dirs(root):  # noqa: SLF001
            if group_key.name.startswith("cat-projector-review-ui-") or group_key.name.startswith("cat-projector-ui-"):
                continue
            paths.extend(review._input_frame_paths_for_group(group_key))  # noqa: SLF001

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        try:
            resolved = review._safe_local_path(path)  # noqa: SLF001
        except ValueError:
            continue
        if resolved in seen or not resolved.exists() or review._is_review_artifact_image(resolved):  # noqa: SLF001
            continue
        seen.add(resolved)
        unique.append(resolved)
    return sorted(unique, key=lambda item: item.as_posix())


def _init_worker(model_path: str, metadata_path: str) -> None:
    global _WORKER_MODEL, _WORKER_METADATA
    _WORKER_MODEL, _WORKER_METADATA = detector.load_model(Path(model_path), Path(metadata_path))


def _score_one(path_raw: str) -> dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_METADATA is None:
        raise RuntimeError("worker model is not loaded")
    image_path = Path(path_raw)
    with detector.Image.open(image_path) as image:
        rgb = image.convert("RGB")
    predictions = sorted(
        detector.score_candidates(rgb, model=_WORKER_MODEL, metadata=_WORKER_METADATA),
        key=lambda item: item.cat_probability,
        reverse=True,
    )
    top_candidates: list[dict[str, Any]] = []
    for prediction in predictions[:12]:
        if prediction.candidate is None:
            continue
        bbox = detector._format_bbox_xywh(prediction.candidate.bbox_xywh)  # noqa: SLF001
        top_candidates.append(
            {
                "p": round(prediction.cat_probability, 4),
                "bbox": bbox,
                "source": prediction.candidate.source,
                "area": prediction.candidate.area_px,
            }
        )
    best = top_candidates[0] if top_candidates else {}
    return {
        "raw_path": str(image_path),
        "candidate_count": len(predictions),
        "best_probability": float(best.get("p", 0.0)),
        "best_bbox": str(best.get("bbox", "")),
        "best_source": str(best.get("source", "")),
        "top_candidates": top_candidates,
    }


def _rescore_frames(paths: list[Path], *, model_path: Path, metadata_path: Path, jobs: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, jobs),
        initializer=_init_worker,
        initargs=(str(model_path), str(metadata_path)),
    ) as executor:
        futures = {executor.submit(_score_one, str(path)): index for index, path in enumerate(paths)}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            row = future.result()
            row["global_frame"] = index
            rows.append(row)
    return sorted(rows, key=lambda row: int(row["global_frame"]))


def _write_rescore_outputs(
    run_dir: Path,
    *,
    rows: list[dict[str, Any]],
    model_metadata: dict[str, Any],
    training_package: dict[str, Any],
    labels: list[Path],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    model_summary = {key: value for key, value in model_metadata.items() if key != "training_rows"}
    training_summary = {key: value for key, value in training_package.items() if key != "copied"}
    for row in rows:
        row["model_path"] = model_metadata.get("model_path", "")
        row["model_created_at"] = model_metadata.get("created_at", "")
        row["uncertainty_score"] = round(1.0 - min(1.0, abs(float(row["best_probability"]) - 0.5) * 2.0), 4)
    uncertain = sorted(rows, key=lambda row: (float(row["uncertainty_score"]), float(row["best_probability"])), reverse=True)

    probe_rows_path = run_dir / "probe_rows.json"
    uncertain_path = run_dir / "uncertain_queue.json"
    manifest_path = run_dir / "manifest.json"
    probe_rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    uncertain_path.write_text(
        json.dumps(uncertain[:500], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "kind": "cat_projector_active_learning_rescore_v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "frame_count": len(rows),
        "labels": [str(path) for path in labels],
        "model": model_summary,
        "probe_rows": str(probe_rows_path),
        "training_package": training_summary,
        "uncertain_queue": str(uncertain_path),
        "top_uncertain": uncertain[:20],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_path = RESCORES_ROOT / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_iteration(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or f"active-learning-{_utc_slug()}"
    run_dir = args.output_root.expanduser() / run_id
    if run_dir.exists() and not args.replace_existing:
        raise RuntimeError(f"{run_dir} already exists; pass --replace-existing to rebuild it")
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)

    training_package = _materialize_review_training_package()
    labels = _collect_labels([Path(training_package["labels_csv"]), *args.labels])
    model_metadata = _train_model(labels, model_path=args.model.expanduser(), metadata_path=args.metadata.expanduser(), args=args)
    frames = _reviewable_frame_paths()
    if not frames:
        raise RuntimeError("no reviewable frames found for rescoring")
    rows = _rescore_frames(
        frames,
        model_path=args.model.expanduser(),
        metadata_path=args.metadata.expanduser(),
        jobs=args.jobs,
    )
    return _write_rescore_outputs(
        run_dir,
        rows=rows,
        model_metadata=model_metadata,
        training_package=training_package,
        labels=labels,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, action="append", default=[])
    parser.add_argument("--model", type=Path, default=detector.DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=detector.DEFAULT_METADATA_PATH)
    parser.add_argument("--output-root", type=Path, default=RESCORES_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--min-positive", type=int, default=5)
    parser.add_argument("--min-negative", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--replace-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    manifest = run_iteration(parse_args(argv))
    print(
        json.dumps(
            {
                "kind": manifest["kind"],
                "created_at": manifest["created_at"],
                "frame_count": manifest["frame_count"],
                "model": manifest["model"],
                "probe_rows": manifest["probe_rows"],
                "training_package": manifest["training_package"],
                "uncertain_queue": manifest["uncertain_queue"],
                "top_uncertain": manifest["top_uncertain"][:5],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
