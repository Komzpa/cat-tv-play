#!/usr/bin/env python3
"""Train, evaluate, and report on Sher YOLO segmentation models."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_STATE_ROOT = Path("~/.openclaw/state/cat-tv-learning").expanduser()
DEFAULT_CALIBRATION = DEFAULT_STATE_ROOT / "calibrations" / "living_room_wall_20260514_sher_pilot.json"


def _load_repo_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _utc_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_summary() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, check=False, text=True, capture_output=True).stdout.strip()

    return {
        "repo": str(REPO_ROOT),
        "head": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "dirty": bool(run(["git", "status", "--short"])),
        "diff_stat": run(["git", "diff", "--stat"]),
    }


def _load_manifest_for_dataset(dataset_yaml: Path) -> dict[str, Any]:
    manifest_path = dataset_yaml.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _dataset_hash(dataset_yaml: Path) -> str:
    manifest = _load_manifest_for_dataset(dataset_yaml)
    digest = hashlib.sha256()
    digest.update(dataset_yaml.read_bytes())
    if manifest:
        digest.update(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for label in sorted(dataset_yaml.parent.glob("labels/*/*.txt")):
        digest.update(str(label.relative_to(dataset_yaml.parent)).encode("utf-8"))
        digest.update(label.read_bytes())
    return digest.hexdigest()


def _provenance(kind: str, args: argparse.Namespace, dataset_yaml: Path | None = None) -> dict[str, Any]:
    dataset_manifest = _load_manifest_for_dataset(dataset_yaml) if dataset_yaml else {}
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not callable(value)
    }
    return {
        "kind": kind,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "git": _git_summary(),
        "config": config,
        "dataset": {
            "dataset_yaml": str(dataset_yaml) if dataset_yaml else None,
            "dataset_hash": _dataset_hash(dataset_yaml) if dataset_yaml else None,
            "export_manifest": str(dataset_yaml.parent / "manifest.json") if dataset_yaml else None,
            "export_dataset_hash": dataset_manifest.get("dataset_hash"),
            "item_count": dataset_manifest.get("item_count"),
            "train_count": dataset_manifest.get("train_count"),
            "val_count": dataset_manifest.get("val_count"),
            "hard_negative_count": dataset_manifest.get("hard_negative_count"),
        },
    }


def _require_local_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _looks_like_generic_yolo_segmentation_base(path: Path) -> bool:
    expanded = path.expanduser()
    name = expanded.name.lower()
    return (
        expanded.parent.name == "base"
        or (name.startswith("yolo") and name.endswith("-seg.pt"))
        or (name.startswith("yolo") and name.endswith("seg.pt"))
    )


def _training_mode_for_base_model(base_model: Path, *, allow_new_sher_lineage: bool) -> str:
    if not _looks_like_generic_yolo_segmentation_base(base_model):
        return "fine_tune_existing_sher"
    if allow_new_sher_lineage:
        return "new_sher_lineage_from_generic_yolo"
    raise ValueError(
        "base model looks like generic YOLO segmentation weights, not an existing Sher checkpoint: "
        f"{base_model}. Routine Sher refreshes must fine-tune from the latest Sher .pt. "
        "Pass --allow-new-sher-lineage only when intentionally bootstrapping a fresh lineage."
    )


def train(args: argparse.Namespace) -> int:
    dataset_yaml = _require_local_file(args.dataset, "dataset")
    base_model = args.base_model.expanduser()
    if not base_model.exists() and not args.allow_download_base:
        raise FileNotFoundError(f"base model must be a local file unless --allow-download-base is set: {base_model}")
    training_mode = _training_mode_for_base_model(
        base_model,
        allow_new_sher_lineage=args.allow_new_sher_lineage,
    )
    try:
        import ultralytics
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("ultralytics is required for training; install the segmentation extra") from exc

    run_name = args.run_name or f"sher-yolo-seg-{_utc_slug()}"
    run_root = args.run_root.expanduser()
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    provenance = _provenance("cat_projector_yolo_segmentation_train_v1", args, dataset_yaml)
    provenance["runtime"] = {"python": sys.version, "ultralytics": getattr(ultralytics, "__version__", "")}
    provenance["base_model"] = {
        "path": str(base_model),
        "sha256": _sha256_path(base_model) if base_model.exists() else None,
    }
    provenance["training_mode"] = training_mode
    provenance["parent_model"] = (
        {
            "path": str(base_model),
            "sha256": _sha256_path(base_model) if base_model.exists() else None,
        }
        if training_mode == "fine_tune_existing_sher"
        else None
    )
    (run_dir / "train_config.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model = YOLO(str(base_model))
    train_kwargs: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(run_root),
        "name": run_name,
        "exist_ok": True,
        "seed": args.seed,
        "optimizer": args.optimizer,
    }
    for key in ("lr0", "lrf", "warmup_epochs", "freeze"):
        value = getattr(args, key)
        if value is not None:
            train_kwargs[key] = value
    result = model.train(**train_kwargs)
    best = run_dir / "weights" / "best.pt"
    if args.out:
        out = args.out.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out)
    else:
        out = best
    provenance["model"] = {"path": str(out), "sha256": _sha256_path(out) if out.exists() else None}
    provenance["metrics"] = getattr(result, "results_dict", {}) or {}
    (run_dir / "train_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"kind": provenance["kind"], "run_dir": str(run_dir), "model": str(out)}, sort_keys=True))
    return 0


def _read_yolo_polygon_labels(path: Path, width: int, height: int) -> list[list[tuple[float, float]]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    polygons: list[list[tuple[float, float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = [float(value) for value in parts[1:]]
        polygons.append([(coords[index] * width, coords[index + 1] * height) for index in range(0, len(coords), 2)])
    return polygons


def _polygon_mask(points: list[tuple[float, float]], size: tuple[int, int]) -> np.ndarray:
    if len(points) < 3:
        return np.zeros((size[1], size[0]), dtype=bool)
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).polygon(points, fill=1)
    return np.asarray(image, dtype=bool)


def _polygons_mask(polygons: list[list[tuple[float, float]]], size: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), dtype=bool)
    for polygon in polygons:
        mask |= _polygon_mask(polygon, size)
    return mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if not union:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _resize_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _top_y(mask: np.ndarray) -> float | None:
    ys = np.nonzero(mask)[0]
    if len(ys) == 0:
        return None
    return float(np.quantile(ys.astype(float), 0.05))


def _top_point(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    top_y = float(np.quantile(ys.astype(float), 0.05))
    band = ys <= int(np.ceil(top_y))
    if not np.any(band):
        band = ys == ys.min()
    return (float(np.median(xs[band].astype(float))), top_y)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _load_homography(calibration_path: Path | None) -> tuple[float, ...] | None:
    if calibration_path is None:
        return None
    calibration_path = calibration_path.expanduser()
    if not calibration_path.exists():
        return None
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if "homography" in data:
        return tuple(float(value) for value in data["homography"])
    calibration = _load_repo_module(
        "cat_tv_play_calibration_for_yolo_segmentation",
        REPO_ROOT / "custom_components" / "cat_tv_play" / "calibration.py",
    )
    unit_scale = 0.1 if data.get("wall_points_mm") and not data.get("wall_points_cm") else 1.0
    wall_points = data.get("wall_points_cm") or data.get("wall_points_mm") or []
    points = [
        calibration.CalibrationPoint(
            image_x=float(image_point[0]),
            image_y=float(image_point[1]),
            wall_x_cm=float(wall_point[0]) * unit_scale,
            wall_y_cm=float(wall_point[1]) * unit_scale,
        )
        for image_point, wall_point in zip(data.get("image_points_px") or [], wall_points, strict=False)
    ]
    if len(points) < 4:
        return None
    return tuple(float(value) for value in calibration.image_to_wall_homography(points))


def _transform_image_point(
    homography: tuple[float, ...] | None, point: tuple[float, float] | None
) -> tuple[float, float] | None:
    if homography is None or point is None:
        return None
    x, y = point
    h0, h1, h2, h3, h4, h5, h6, h7 = homography
    denominator = h6 * x + h7 * y + 1.0
    if abs(denominator) < 1e-9:
        return None
    return ((h0 * x + h1 * y + h2) / denominator, (h3 * x + h4 * y + h5) / denominator)


def _load_legacy_detector(args: argparse.Namespace) -> Any | None:
    if not args.legacy_model:
        return None
    detection = _load_repo_module(
        "cat_tv_play_detection_for_yolo_eval",
        REPO_ROOT / "custom_components" / "cat_tv_play" / "detection.py",
    )
    return detection.LegacyContrastDetector(
        model_path=args.legacy_model.expanduser(),
        metadata_path=args.legacy_metadata.expanduser() if args.legacy_metadata else None,
        min_probability=0.0,
    )


def _save_overlay(
    path: Path,
    image_path: Path,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str,
    *,
    measurement_point: tuple[float, float] | None = None,
    bbox_xywh: tuple[int, int, int, int] | None = None,
) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    pixels = overlay.load()
    for y, x in zip(*np.nonzero(gt_mask), strict=False):
        pixels[int(x), int(y)] = (50, 220, 90, 95)
    for y, x in zip(*np.nonzero(pred_mask), strict=False):
        pixels[int(x), int(y)] = (255, 80, 40, 95)
    composed = Image.alpha_composite(rgb.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(composed)
    if bbox_xywh is not None:
        x, y, width, height = bbox_xywh
        draw.rectangle((x, y, x + width, y + height), outline=(255, 210, 50, 255), width=3)
    if measurement_point is not None:
        x, y = measurement_point
        draw.line((x - 10, y, x + 10, y), fill=(80, 180, 255, 255), width=3)
        draw.line((x, y - 10, x, y + 10), fill=(80, 180, 255, 255), width=3)
    draw.rectangle((0, 0, min(composed.width, 900), 24), fill=(0, 0, 0, 190))
    draw.text((8, 5), title, fill=(255, 255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(path)


def eval_model(args: argparse.Namespace) -> int:
    dataset_yaml = _require_local_file(args.dataset, "dataset")
    model_path = _require_local_file(args.model, "model")
    manifest = _load_manifest_for_dataset(dataset_yaml)
    rows = list(manifest.get("rows") or [])
    if not rows:
        raise ValueError("dataset manifest has no rows")
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("ultralytics is required for evaluation; install the segmentation extra") from exc

    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(model_path))
    eval_rows: list[dict[str, Any]] = []
    tp = fp = fn = tn = 0
    ious: list[float] = []
    height_errors_px: list[float] = []
    height_errors_cm: list[float] = []
    homography = _load_homography(args.calibration)
    legacy_detector = _load_legacy_detector(args)
    legacy_tp = legacy_fp = legacy_fn = legacy_tn = 0
    overlay_counts: dict[str, int] = {}

    for index, row in enumerate(rows):
        image_path = Path(row["image"]).expanduser()
        label_path = Path(row["label"]).expanduser()
        with Image.open(image_path) as image:
            width, height = image.size
            rgb_image = image.convert("RGB")
        gt_mask = _polygons_mask(_read_yolo_polygon_labels(label_path, width, height), (width, height))
        result = model.predict(str(image_path), conf=args.confidence_threshold, verbose=False, device=args.device)[0]
        pred_mask = np.zeros((height, width), dtype=bool)
        pred_score = 0.0
        if getattr(result, "masks", None) is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks))
            if len(masks):
                best_index = int(np.argmax(scores))
                pred_mask = _resize_mask(masks[best_index], width=width, height=height)
                pred_score = float(scores[best_index])
        gt_present = bool(gt_mask.any())
        pred_present = bool(pred_mask.any())
        if gt_present and pred_present:
            tp += 1
        elif gt_present:
            fn += 1
        elif pred_present:
            fp += 1
        else:
            tn += 1
        iou = _iou(gt_mask, pred_mask) if gt_present or pred_present else None
        if iou is not None:
            ious.append(iou)
        gt_top = _top_y(gt_mask)
        pred_top = _top_y(pred_mask)
        pred_point = _top_point(pred_mask)
        gt_point = _top_point(gt_mask)
        pred_wall = _transform_image_point(homography, pred_point)
        gt_wall = _transform_image_point(homography, gt_point)
        height_error_px = None
        height_error_cm = None
        if gt_top is not None and pred_top is not None:
            height_error_px = float(pred_top - gt_top)
            height_errors_px.append(abs(height_error_px))
        if gt_wall is not None and pred_wall is not None:
            height_error_cm = float(pred_wall[1] - gt_wall[1])
            height_errors_cm.append(abs(height_error_cm))
        buckets = ["true_negative"]
        if gt_present and not pred_present:
            buckets = ["false_negative"]
        elif pred_present and not gt_present:
            buckets = ["false_positive"]
        elif gt_present and pred_present and pred_score < args.low_confidence_threshold:
            buckets = ["low_confidence_true_positive"]
        elif gt_present and pred_present:
            buckets = ["true_positive"]
        if pred_wall is not None and pred_wall[1] >= args.high_jump_cm:
            buckets.append("high_mask_top_height_jump")
        if (
            gt_wall is not None
            and pred_wall is not None
            and abs(float(pred_wall[1] - gt_wall[1])) >= args.record_change_cm
        ):
            buckets.append("frames_that_can_change_jump_record")
        legacy_present = None
        legacy_score = None
        if legacy_detector is not None:
            legacy_detections = legacy_detector.detect(rgb_image)
            legacy_score = float(legacy_detections[0].score) if legacy_detections else 0.0
            legacy_present = legacy_score >= args.legacy_threshold
            if gt_present and legacy_present:
                legacy_tp += 1
            elif gt_present:
                legacy_fn += 1
            elif legacy_present:
                legacy_fp += 1
            else:
                legacy_tn += 1
            if legacy_present != pred_present:
                buckets.append("segmentation_legacy_disagreement")
            if legacy_present and not gt_present:
                buckets.append("legacy_false_positive_hard_negative")
        if pred_present and not gt_present:
            buckets.append("false_positive_hard_negative")
        overlay_paths: list[str] = []
        if "true_negative" not in buckets:
            pred_bbox = _bbox_from_mask(pred_mask)
            title = (
                f"score={pred_score:.3f} iou={iou if iou is not None else -1:.3f} "
                f"top_err_px={height_error_px} top_err_cm={height_error_cm} "
                f"backend=ultralytics_yolo_segmentation tracker=eval_static legacy_score={legacy_score}"
            )
            for bucket in dict.fromkeys(buckets):
                if bucket == "true_positive":
                    continue
                if overlay_counts.get(bucket, 0) >= args.overlay_limit:
                    continue
                overlay_path = out_dir / "overlays" / bucket / f"{index:05d}.jpg"
                _save_overlay(
                    overlay_path,
                    image_path,
                    gt_mask,
                    pred_mask,
                    f"{bucket} {title}",
                    measurement_point=pred_point,
                    bbox_xywh=pred_bbox,
                )
                overlay_counts[bucket] = overlay_counts.get(bucket, 0) + 1
                overlay_paths.append(str(overlay_path.relative_to(out_dir)))
        eval_rows.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "source_image": row.get("source_image"),
                "source_label": row.get("source_label"),
                "split": row.get("split"),
                "hard_negative": not gt_present,
                "gt_present": gt_present,
                "pred_present": pred_present,
                "score": pred_score,
                "legacy_present": legacy_present,
                "legacy_score": legacy_score,
                "mask_iou": iou,
                "height_top_error_px": height_error_px,
                "height_top_error_cm": height_error_cm,
                "pred_wall_height_cm": None if pred_wall is None else float(pred_wall[1]),
                "gt_wall_height_cm": None if gt_wall is None else float(gt_wall[1]),
                "measurement_point": None
                if pred_point is None
                else {"image_x": pred_point[0], "image_y": pred_point[1]},
                "bbox_xywh": None if _bbox_from_mask(pred_mask) is None else list(_bbox_from_mask(pred_mask)),
                "bucket": buckets[0],
                "buckets": buckets,
                "overlay_paths": overlay_paths,
            }
        )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    legacy_precision = legacy_tp / (legacy_tp + legacy_fp) if legacy_tp + legacy_fp else None
    legacy_recall = legacy_tp / (legacy_tp + legacy_fn) if legacy_tp + legacy_fn else None
    summary = _provenance("cat_projector_yolo_segmentation_eval_v1", args, dataset_yaml)
    summary["model"] = {"path": str(model_path), "sha256": _sha256_path(model_path)}
    summary["metrics"] = {
        "cat_presence_precision": precision,
        "cat_presence_recall": recall,
        "false_positive_hard_negative_count": fp,
        "true_positive_count": tp,
        "false_negative_count": fn,
        "true_negative_count": tn,
        "mean_mask_iou": float(np.mean(ious)) if ious else None,
        "mean_abs_height_top_error_px": float(np.mean(height_errors_px)) if height_errors_px else None,
        "mean_abs_height_top_error_cm": float(np.mean(height_errors_cm)) if height_errors_cm else None,
        "height_error_cm_reason": None if height_errors_cm else "no calibration or no matched mask tops",
        "legacy_cat_presence_precision": legacy_precision,
        "legacy_cat_presence_recall": legacy_recall,
        "legacy_false_positive_hard_negative_count": legacy_fp if legacy_detector is not None else None,
        "legacy_true_positive_count": legacy_tp if legacy_detector is not None else None,
        "legacy_false_negative_count": legacy_fn if legacy_detector is not None else None,
        "legacy_true_negative_count": legacy_tn if legacy_detector is not None else None,
        "segmentation_beats_legacy_hard_negatives": (None if legacy_detector is None else fp < legacy_fp),
    }
    summary["rows"] = eval_rows
    eval_path = out_dir / "eval.json"
    eval_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_report_from_files(manifest, summary, out_dir / "report.html", limit=args.report_limit)
    print(json.dumps({"kind": summary["kind"], "out": str(out_dir), "metrics": summary["metrics"]}, sort_keys=True))
    return 0


def render_report_from_files(manifest: dict[str, Any], eval_data: dict[str, Any], out: Path, *, limit: int) -> None:
    rows = eval_data.get("rows") or []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_buckets = row.get("buckets") if isinstance(row.get("buckets"), list) else [row.get("bucket") or "unknown"]
        for bucket in row_buckets:
            buckets.setdefault(str(bucket), []).append(row)
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Sher YOLO segmentation report</title>",
        (
            "<style>body{font-family:sans-serif;background:#111;color:#ddd}"
            " img{max-width:360px} .row{display:inline-block;margin:8px;"
            "vertical-align:top;width:380px}</style>"
        ),
        "<h1>Sher YOLO segmentation report</h1>",
        f"<pre>{html.escape(json.dumps(eval_data.get('metrics', {}), indent=2, sort_keys=True))}</pre>",
        f"<p>dataset: {html.escape(str(manifest.get('dataset_hash') or manifest.get('item_count')))}</p>",
    ]
    for bucket, bucket_rows in sorted(buckets.items()):
        parts.append(f"<h2>{html.escape(bucket)} ({len(bucket_rows)})</h2>")
        for row in bucket_rows[:limit]:
            parts.append("<div class='row'>")
            parts.append(f"<div>{html.escape(Path(str(row.get('image'))).name)}</div>")
            for overlay in row.get("overlay_paths") or (
                [] if not row.get("overlay_path") else [row.get("overlay_path")]
            ):
                parts.append(f"<a href='{html.escape(str(overlay))}'><img src='{html.escape(str(overlay))}'></a>")
            summary = {
                key: row.get(key)
                for key in (
                    "score",
                    "legacy_score",
                    "legacy_present",
                    "mask_iou",
                    "height_top_error_px",
                    "height_top_error_cm",
                    "pred_wall_height_cm",
                    "bbox_xywh",
                    "measurement_point",
                    "source_image",
                )
            }
            parts.append(
                f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre>"
            )
            parts.append("</div>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_report(args: argparse.Namespace) -> int:
    manifest = json.loads(args.dataset_manifest.expanduser().read_text(encoding="utf-8"))
    eval_data = json.loads(args.eval.expanduser().read_text(encoding="utf-8"))
    render_report_from_files(manifest, eval_data, args.out.expanduser(), limit=args.limit)
    print(
        json.dumps(
            {"kind": "cat_projector_yolo_segmentation_report_v1", "out": str(args.out.expanduser())}, sort_keys=True
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--base-model", type=Path, required=True)
    train_parser.add_argument("--out", type=Path)
    train_parser.add_argument("--run-root", type=Path, default=DEFAULT_STATE_ROOT / "yolo-runs")
    train_parser.add_argument("--run-name", default="")
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--imgsz", type=int, default=960)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--seed", type=int, default=20260523)
    train_parser.add_argument("--optimizer", default="auto")
    train_parser.add_argument("--lr0", type=float)
    train_parser.add_argument("--lrf", type=float)
    train_parser.add_argument("--warmup-epochs", type=float)
    train_parser.add_argument("--freeze", type=int)
    train_parser.add_argument("--allow-download-base", action="store_true")
    train_parser.add_argument(
        "--allow-new-sher-lineage",
        action="store_true",
        help=(
            "permit training from generic YOLO segmentation weights. Omit this for routine refreshes, "
            "which should fine-tune from the latest Sher .pt."
        ),
    )
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--dataset", type=Path, required=True)
    eval_parser.add_argument("--model", type=Path, required=True)
    eval_parser.add_argument("--out", type=Path, required=True)
    eval_parser.add_argument("--legacy-model", type=Path)
    eval_parser.add_argument("--legacy-metadata", type=Path)
    eval_parser.add_argument("--legacy-threshold", type=float, default=0.5)
    eval_parser.add_argument("--device", default=None)
    eval_parser.add_argument("--confidence-threshold", type=float, default=0.55)
    eval_parser.add_argument("--low-confidence-threshold", type=float, default=0.55)
    eval_parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    eval_parser.add_argument("--high-jump-cm", type=float, default=120.0)
    eval_parser.add_argument("--record-change-cm", type=float, default=8.0)
    eval_parser.add_argument("--overlay-limit", type=int, default=200)
    eval_parser.add_argument("--report-limit", type=int, default=200)
    eval_parser.set_defaults(func=eval_model)

    report_parser = subparsers.add_parser("render-report")
    report_parser.add_argument("--dataset-manifest", type=Path, required=True)
    report_parser.add_argument("--eval", type=Path, required=True)
    report_parser.add_argument("--out", type=Path, required=True)
    report_parser.add_argument("--limit", type=int, default=200)
    report_parser.set_defaults(func=render_report)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
