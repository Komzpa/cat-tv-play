#!/usr/bin/env python3
"""Export reviewed Cat Projector masks to YOLO segmentation datasets.

Splits are by recording/session key, not by adjacent frames, to avoid leaking
near-identical video frames between train and validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

STATE_ROOT = Path("~/.openclaw/state/cat-tv-learning").expanduser()
REVIEW_ROOT = STATE_ROOT / "label-review"
LABELS_ROOT = REVIEW_ROOT / "labels"
MASKS_ROOT = REVIEW_ROOT / "masks"


def datetime_now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ExportItem:
    label_path: Path
    image_path: Path
    split_key: str
    split_key_source: str
    label: dict[str, Any]
    polygons: tuple[tuple[tuple[float, float], ...], ...]
    mask_ids: tuple[str, ...] = ()
    mask_ref_paths: tuple[str, ...] = ()
    discarded_mask_count: int = 0


@dataclass(frozen=True)
class SkippedLabel:
    label_path: Path
    reason: str
    image_path: Path | None = None


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    path: str = ""


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:160] or "frame"


def _split_for_key(key: str, val_fraction: float) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe_float(value: Any) -> float:
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError("non-finite coordinate")
    return number


def _polygon_from_mask(mask: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
    polygon = mask.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None
    points: list[tuple[float, float]] = []
    for point in polygon:
        if not isinstance(point, dict):
            return None
        try:
            points.append((_json_safe_float(point["x"]), _json_safe_float(point["y"])))
        except (KeyError, TypeError, ValueError):
            return None
    return tuple(points) if len(points) >= 3 else None


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _split_key_for_label(label: dict[str, Any], image_path: Path) -> tuple[str, str]:
    for key in ("source_recording_dir", "source_video_path", "video_id"):
        value = str(label.get(key) or "")
        if value:
            return value, key
    parts = image_path.parts
    for marker in ("recordings", "datasets", "batch_reviews"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return "/".join(parts[index : index + 2]), f"path:{marker}"
    return str(image_path.parent), "path:parent"


def _candidate_mask_ref_paths(label_path: Path, label: dict[str, Any], ref: dict[str, Any]) -> list[Path]:
    raw_path = str(ref.get("path") or "")
    if not raw_path:
        return []
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return [path]
    candidates = [Path.cwd() / path, label_path.parent / path]
    if label_path.parent.name == "labels":
        sibling_masks_root = label_path.parent.parent / "masks"
        case_id = str(label.get("case_id") or label_path.stem)
        if case_id:
            candidates.append(sibling_masks_root / case_id / path.name)
        candidates.append(sibling_masks_root / path)
    case_id = str(label.get("case_id") or label_path.stem)
    if case_id:
        candidates.append(MASKS_ROOT / case_id / path.name)
    candidates.append(MASKS_ROOT / path)
    return candidates


def _load_mask_ref(label_path: Path, label: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    for path in _candidate_mask_ref_paths(label_path, label, ref):
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _masks_for_label(label_path: Path, label: dict[str, Any]) -> list[dict[str, Any]]:
    masks = [mask for mask in label.get("masks") or [] if isinstance(mask, dict)]
    seen_ids = {str(mask.get("mask_id") or mask.get("id") or "") for mask in masks}
    for ref in label.get("mask_refs") or []:
        if not isinstance(ref, dict):
            continue
        sidecar = _load_mask_ref(label_path, label, ref)
        if sidecar is None:
            continue
        mask_id = str(sidecar.get("mask_id") or sidecar.get("id") or ref.get("id") or "")
        if mask_id and mask_id in seen_ids:
            continue
        masks.append(sidecar)
        if mask_id:
            seen_ids.add(mask_id)
    return masks


def collect_items_with_skips(labels_root: Path) -> tuple[list[ExportItem], list[SkippedLabel]]:
    items: list[ExportItem] = []
    skipped: list[SkippedLabel] = []
    for label_path in sorted(labels_root.glob("*.json")):
        try:
            label = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped.append(SkippedLabel(label_path, "invalid_json"))
            continue
        label_kind = str(label.get("label") or "")
        if label_kind not in {"cat", "not_cat"}:
            skipped.append(SkippedLabel(label_path, f"unsupported_label:{label_kind or 'missing'}"))
            continue
        image_raw = str(label.get("image_path") or "")
        if not image_raw:
            skipped.append(SkippedLabel(label_path, "missing_image_path"))
            continue
        image_path = Path(image_raw).expanduser()
        if not image_path.exists():
            skipped.append(SkippedLabel(label_path, "missing_image_file", image_path))
            continue
        all_masks = _masks_for_label(label_path, label)
        all_valid_polygons = tuple(
            polygon for mask in all_masks for polygon in [_polygon_from_mask(mask)] if polygon is not None
        )
        polygons = all_valid_polygons if label_kind == "cat" else ()
        if label_kind == "cat" and not polygons:
            skipped.append(SkippedLabel(label_path, "cat_without_valid_polygon", image_path))
            continue
        split_key, split_key_source = _split_key_for_label(label, image_path)
        items.append(
            ExportItem(
                label_path=label_path,
                image_path=image_path,
                split_key=split_key,
                split_key_source=split_key_source,
                label=label,
                polygons=polygons,
                mask_ids=tuple(
                    str(mask.get("mask_id") or mask.get("id") or "")
                    for mask in all_masks
                    if str(mask.get("mask_id") or mask.get("id") or "")
                ),
                mask_ref_paths=tuple(
                    str(ref.get("path") or "")
                    for ref in label.get("mask_refs") or []
                    if isinstance(ref, dict) and ref.get("path")
                ),
                discarded_mask_count=0 if label_kind == "cat" else len(all_valid_polygons),
            )
        )
    return items, skipped


def collect_items(labels_root: Path) -> list[ExportItem]:
    items, _skipped = collect_items_with_skips(labels_root)
    return items


def validate_items(
    items: list[ExportItem], skipped: list[SkippedLabel], *, val_fraction: float = 0.2
) -> dict[str, Any]:
    issues: list[ValidationIssue] = []
    sessions: set[str] = set()
    split_sessions: dict[str, set[str]] = {"train": set(), "val": set()}
    split_counts: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    positives = 0
    negatives = 0
    split_key_sources: Counter[str] = Counter()

    for skipped_label in skipped:
        severity = (
            "error"
            if skipped_label.reason in {"invalid_json", "missing_image_path", "missing_image_file"}
            else "warning"
        )
        issues.append(
            ValidationIssue(
                severity=severity,
                code=skipped_label.reason,
                message=f"skipped label: {skipped_label.reason}",
                path=str(skipped_label.image_path or skipped_label.label_path),
            )
        )

    for item in items:
        sessions.add(item.split_key)
        split_key_sources[item.split_key_source] += 1
        split = _split_for_key(item.split_key, val_fraction)
        split_sessions.setdefault(split, set()).add(item.split_key)
        split_counts[split] += 1
        if item.polygons:
            positives += 1
        else:
            negatives += 1
        try:
            with Image.open(item.image_path) as image:
                width, height = image.size
        except OSError as exc:
            issues.append(
                ValidationIssue("error", "unreadable_image", f"cannot open image: {exc}", str(item.image_path))
            )
            continue
        image_sizes[f"{width}x{height}"] += 1
        if width <= 0 or height <= 0:
            issues.append(
                ValidationIssue(
                    "error", "invalid_image_size", "image has non-positive dimensions", str(item.image_path)
                )
            )
            continue
        for polygon in item.polygons:
            if _polygon_area(polygon) <= 1.0:
                issues.append(
                    ValidationIssue("error", "empty_polygon", "polygon area is <= 1 px", str(item.label_path))
                )
            for x, y in polygon:
                if not (0 <= x <= width and 0 <= y <= height):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "polygon_out_of_bounds",
                            f"point ({x:.1f}, {y:.1f}) outside {width}x{height}",
                            str(item.label_path),
                        )
                    )

    leaked_sessions = sorted(split_sessions.get("train", set()) & split_sessions.get("val", set()))
    for session in leaked_sessions:
        issues.append(ValidationIssue("error", "session_split_leak", "session appears in both train and val", session))

    issue_counts = Counter(issue.code for issue in issues)
    severity_counts = Counter(issue.severity for issue in issues)
    return {
        "kind": "cat_projector_yolo_segmentation_validation_v1",
        "session_count": len(sessions),
        "frame_count": len(items),
        "positive_count": positives,
        "negative_count": negatives,
        "split_counts": dict(sorted(split_counts.items())),
        "split_session_counts": {key: len(value) for key, value in sorted(split_sessions.items())},
        "split_key_sources": dict(sorted(split_key_sources.items())),
        "image_sizes": dict(image_sizes.most_common()),
        "bad_or_missing_label_count": sum(1 for issue in issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
        "issue_counts": dict(sorted(issue_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "issues": [issue.__dict__ for issue in issues[:200]],
    }


def _dataset_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (item["split"], item["source_image"], item["label"])):
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_yolo_label(
    path: Path, polygons: tuple[tuple[tuple[float, float], ...], ...], *, width: int, height: int
) -> None:
    lines: list[str] = []
    for polygon in polygons:
        coords: list[str] = ["0"]
        for x, y in polygon:
            coords.append(f"{max(0.0, min(1.0, x / width)):.6f}")
            coords.append(f"{max(0.0, min(1.0, y / height)):.6f}")
        lines.append(" ".join(coords))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_yolo_segmentation(
    items: list[ExportItem], output_root: Path, *, val_fraction: float, copy: bool
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"{output_root} already exists; choose a new output directory")
    manifest_rows: list[dict[str, Any]] = []
    for item in items:
        split = _split_for_key(item.split_key, val_fraction)
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        row_id = hashlib.sha256(f"{item.label_path}|{item.image_path}".encode()).hexdigest()[:10]
        image_name = (
            f"{_safe_slug(item.split_key)}__{_safe_slug(item.image_path.stem)}__"
            f"{row_id}{item.image_path.suffix.lower()}"
        )
        target_image = image_dir / image_name
        if copy:
            shutil.copy2(item.image_path, target_image)
        else:
            target_image.symlink_to(item.image_path)
        with Image.open(item.image_path) as image:
            width, height = image.size
        target_label = label_dir / f"{target_image.stem}.txt"
        _write_yolo_label(target_label, item.polygons, width=width, height=height)
        manifest_rows.append(
            {
                "case_id": item.label.get("case_id") or item.label_path.stem,
                "image": str(target_image),
                "label": str(target_label),
                "source_image": str(item.image_path),
                "source_label": str(item.label_path),
                "split": split,
                "split_key": item.split_key,
                "split_key_source": item.split_key_source,
                "calibration_id": item.label.get("calibration_id"),
                "video_id": item.label.get("video_id"),
                "source_recording_dir": item.label.get("source_recording_dir"),
                "label_kind": item.label.get("label"),
                "label_updated_at": item.label.get("updated_at") or item.label.get("reviewed_at"),
                "review_status": item.label.get("review_status"),
                "cat_present": item.label.get("cat_present"),
                "candidate_is_cat": item.label.get("candidate_is_cat"),
                "review_decision": item.label.get("review_decision"),
                "geometry_status": item.label.get("geometry_status"),
                "polygon_count": len(item.polygons),
                "polygon_point_counts": [len(polygon) for polygon in item.polygons],
                "polygon_area_px": [round(float(_polygon_area(polygon)), 2) for polygon in item.polygons],
                "image_width": width,
                "image_height": height,
                "image_sha256": _sha256_path(item.image_path),
                "mask_ids": list(item.mask_ids),
                "mask_ref_paths": list(item.mask_ref_paths),
                "had_discarded_mask": bool(item.discarded_mask_count),
                "discarded_mask_count": item.discarded_mask_count,
                "negative_reason": item.label.get("negative_reason")
                or ("hard_negative_or_empty_frame" if not item.polygons else ""),
            }
        )

    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: cat",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "kind": "cat_projector_yolo_segmentation_export_v1",
        "exported_at": datetime_now_utc(),
        "labels_root": str(LABELS_ROOT),
        "item_count": len(manifest_rows),
        "train_count": sum(1 for row in manifest_rows if row["split"] == "train"),
        "val_count": sum(1 for row in manifest_rows if row["split"] == "val"),
        "class_names": ["cat"],
        "split_policy": "recording_session_hash",
        "hard_negative_count": sum(1 for row in manifest_rows if row["polygon_count"] == 0),
        "source_mask_ref_count": sum(len(row.get("mask_ref_paths") or []) for row in manifest_rows),
        "discarded_mask_count": sum(int(row.get("discarded_mask_count") or 0) for row in manifest_rows),
        "rows": manifest_rows,
    }
    manifest["dataset_hash"] = _dataset_hash(manifest_rows)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, default=LABELS_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--symlink", action="store_true", help="symlink images instead of copying them")
    parser.add_argument(
        "--validate-only", action="store_true", help="validate reviewed labels without writing YOLO files"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items, skipped = collect_items_with_skips(args.labels_root.expanduser())
    val_fraction = max(0.0, min(0.9, args.val_fraction))
    validation = validate_items(items, skipped, val_fraction=val_fraction)
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
        return 1 if validation["bad_or_missing_label_count"] else 0
    if args.output is None:
        raise SystemExit("--output is required unless --validate-only is used")
    manifest = export_yolo_segmentation(
        items,
        args.output.expanduser(),
        val_fraction=val_fraction,
        copy=not args.symlink,
    )
    manifest["validation"] = validation
    (args.output.expanduser() / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: manifest[key]
        for key in ("kind", "item_count", "train_count", "val_count", "hard_negative_count", "dataset_hash")
    }
    summary["validation"] = {
        key: validation[key]
        for key in (
            "session_count",
            "positive_count",
            "negative_count",
            "bad_or_missing_label_count",
            "warning_count",
            "image_sizes",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
