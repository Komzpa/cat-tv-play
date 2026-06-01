#!/usr/bin/env python3
"""Run one Cat Projector active-learning iteration.

The iteration is deliberately offline:

1. materialize saved review UI labels into a normal labels.csv package;
2. use the configured detector backend to rescore every reviewable input frame;
3. remeasure jump heights from the fresh per-frame best candidates;
4. write probe_rows.json and an uncertainty-sorted queue for the next review pass.

The legacy backend still retrains the old CatBoost candidate scorer first. The
segmentation backend must not depend on that legacy train step.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import importlib.util
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

from scripts import cat_projector_frame_detector as detector  # noqa: E402
from scripts import cat_projector_label_review_server as review  # noqa: E402

_CALIBRATION_SPEC = importlib.util.spec_from_file_location(
    "cat_tv_play_calibration_for_active_learning",
    REPO_ROOT / "custom_components" / "cat_tv_play" / "calibration.py",
)
assert _CALIBRATION_SPEC is not None
calibration = importlib.util.module_from_spec(_CALIBRATION_SPEC)
assert _CALIBRATION_SPEC.loader is not None
sys.modules[_CALIBRATION_SPEC.name] = calibration
_CALIBRATION_SPEC.loader.exec_module(calibration)
CalibrationPoint = calibration.CalibrationPoint
image_to_wall_homography = calibration.image_to_wall_homography
transform_image_point = calibration.transform_image_point

_DETECTION_SPEC = importlib.util.spec_from_file_location(
    "cat_tv_play_detection_for_active_learning",
    REPO_ROOT / "custom_components" / "cat_tv_play" / "detection.py",
)
assert _DETECTION_SPEC is not None
cat_detection = importlib.util.module_from_spec(_DETECTION_SPEC)
assert _DETECTION_SPEC.loader is not None
sys.modules[_DETECTION_SPEC.name] = cat_detection
_DETECTION_SPEC.loader.exec_module(cat_detection)

_MEASUREMENT_SPEC = importlib.util.spec_from_file_location(
    "cat_tv_play_measurement_for_active_learning",
    REPO_ROOT / "custom_components" / "cat_tv_play" / "measurement.py",
)
assert _MEASUREMENT_SPEC is not None
cat_measurement = importlib.util.module_from_spec(_MEASUREMENT_SPEC)
assert _MEASUREMENT_SPEC.loader is not None
sys.modules[_MEASUREMENT_SPEC.name] = cat_measurement
_MEASUREMENT_SPEC.loader.exec_module(cat_measurement)

_TRACKING_SPEC = importlib.util.spec_from_file_location(
    "cat_tv_play_tracking_for_active_learning",
    REPO_ROOT / "custom_components" / "cat_tv_play" / "tracking.py",
)
assert _TRACKING_SPEC is not None
cat_tracking = importlib.util.module_from_spec(_TRACKING_SPEC)
assert _TRACKING_SPEC.loader is not None
sys.modules[_TRACKING_SPEC.name] = cat_tracking
_TRACKING_SPEC.loader.exec_module(cat_tracking)

STATE_ROOT = Path("~/.openclaw/state/cat-tv-learning").expanduser()
RESCORES_ROOT = STATE_ROOT / "label-review" / "rescores"
DEFAULT_CALIBRATION_PATH = STATE_ROOT / "calibrations" / "living_room_wall_20260514_sher_pilot.json"
DEFAULT_SEGMENTATION_MODEL = os.environ.get("CAT_PROJECTOR_SEGMENTATION_MODEL")
DEFAULT_ALIGNMENT_REFERENCE_CANDIDATES = (
    STATE_ROOT
    / "calibration-captures"
    / "20260516T032058+0400_one_meter_cardboard_profiles"
    / "projector_camera_calibration_8s.mp4",
    STATE_ROOT
    / "calibration-captures"
    / "20260516T032241+0400_one_meter_cardboard_profiles_clean_white"
    / "projector_camera_calibration_clean_white_8s.mp4",
)

_WORKER_MODEL: Any | None = None
_WORKER_METADATA: dict[str, Any] | None = None
_WORKER_DETECTOR: Any | None = None


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
            package_name.startswith("cat-projector-review-ui-") or package_name.startswith("cat-projector-ui-")
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


def _train_model(
    labels: list[Path], *, model_path: Path, metadata_path: Path, args: argparse.Namespace
) -> dict[str, Any]:
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


def _detector_model_metadata(detector_config: dict[str, Any]) -> dict[str, Any]:
    backend = str(detector_config.get("backend") or "")
    model_path = detector_config.get("model_path")
    return {
        "kind": "cat_projector_detector_metadata_v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_path": model_path or "",
        "backend": backend,
        "training_status": "skipped_legacy_catboost_training",
        "training_reason": f"{backend}_backend_uses_configured_detector",
    }


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


def _measurement_for_detection(detection: Any) -> tuple[Any | None, str]:
    if detection.mask is not None:
        point = cat_measurement.mask_top_measurement_point(detection.mask, score=detection.score)
        if point is not None:
            return point, ""
        return (
            cat_measurement.legacy_bbox_top_measurement_point(detection.bbox_xywh, score=detection.score),
            "segmentation_mask_rejected_using_legacy_bbox_top",
        )
    return (
        cat_measurement.legacy_bbox_top_measurement_point(detection.bbox_xywh, score=detection.score),
        "missing_segmentation_mask_using_legacy_bbox_top",
    )


def _normalised_detector_config(args: argparse.Namespace) -> dict[str, Any]:
    backend = str(args.detector_backend or "auto")
    if backend == "auto" and args.segmentation_model:
        backend = "segmentation"
    elif backend == "auto":
        backend = "legacy"
    return {
        "backend": backend,
        "requested_backend": str(args.detector_backend or "auto"),
        "model_path": str(args.segmentation_model.expanduser()) if args.segmentation_model else None,
        "device": args.device,
        "confidence_threshold": args.confidence_threshold,
        "allow_fake": bool(args.allow_fake_detector),
        "warnings": [],
    }


def _init_worker(model_path: str, metadata_path: str, detector_config_raw: str) -> None:
    global _WORKER_MODEL, _WORKER_METADATA, _WORKER_DETECTOR
    detector_config = json.loads(detector_config_raw)
    backend = str(detector_config.get("backend") or "legacy")
    if backend == "legacy":
        _WORKER_MODEL, _WORKER_METADATA = detector.load_model(Path(model_path), Path(metadata_path))
        _WORKER_DETECTOR = cat_detection.LegacyContrastDetector(
            model_path=Path(model_path),
            metadata_path=Path(metadata_path),
            min_probability=0.0,
        )
    else:
        _WORKER_MODEL, _WORKER_METADATA = None, None
        _WORKER_DETECTOR = cat_detection.build_detector(
            cat_detection.DetectorConfig(
                backend=backend,
                model_path=detector_config.get("model_path"),
                device=detector_config.get("device"),
                confidence_threshold=float(detector_config.get("confidence_threshold") or 0.5),
                legacy_model_path=model_path,
                legacy_metadata_path=metadata_path,
                allow_fake=bool(detector_config.get("allow_fake")),
            )
        )


def _score_one(path_raw: str) -> dict[str, Any]:
    if _WORKER_DETECTOR is None:
        raise RuntimeError("worker detector is not loaded")
    image_path = Path(path_raw)
    with detector.Image.open(image_path) as image:
        rgb = image.convert("RGB")
    context = cat_detection.DetectorContext(source_path=str(image_path))
    detections = sorted(_WORKER_DETECTOR.detect(rgb, context), key=lambda item: item.score, reverse=True)
    top_candidates: list[dict[str, Any]] = []
    measurement_points: list[dict[str, Any] | None] = []
    for detection in detections[:12]:
        point, warning = _measurement_for_detection(detection)
        measurement_points.append(cat_measurement.measurement_point_to_dict(point))
        row = detection.to_debug_row()
        row["measurement_point"] = measurement_points[-1]
        row["measurement_warning"] = warning
        top_candidates.append(row)
    best = top_candidates[0] if top_candidates else {}
    return {
        "raw_path": str(image_path),
        "detector_backend": getattr(_WORKER_DETECTOR, "source", "unknown"),
        "detector_model_id": getattr(_WORKER_DETECTOR, "model_id", ""),
        "candidate_count": len(detections),
        "best_probability": float(best.get("p", 0.0)),
        "best_bbox": str(best.get("bbox", "")),
        "best_source": str(best.get("source", "")),
        "best_measurement_point": best.get("measurement_point"),
        "best_measurement_warning": best.get("measurement_warning") or "",
        "best_has_mask": bool(best.get("has_mask")),
        "source_size_px": {"width": rgb.width, "height": rgb.height},
        "top_candidates": top_candidates,
    }


def _rescore_frames(
    paths: list[Path],
    *,
    model_path: Path,
    metadata_path: Path,
    jobs: int,
    detector_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, jobs),
        initializer=_init_worker,
        initargs=(str(model_path), str(metadata_path), json.dumps(detector_config, sort_keys=True)),
    ) as executor:
        futures = {executor.submit(_score_one, str(path)): index for index, path in enumerate(paths)}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            row = future.result()
            row["global_frame"] = index
            rows.append(row)
    return sorted(rows, key=lambda row: int(row["global_frame"]))


def _parse_bbox_xywh(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return None
    if len(parts) != 4:
        return None
    try:
        x, y, width, height = (float(part) for part in parts)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return (x, y, width, height)


def _load_height_calibration(calibration_path: Path) -> tuple[tuple[float, ...], tuple[float, float] | None] | None:
    if not calibration_path.exists():
        return None
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    image_points = data.get("image_points_px") or []
    calibration_size = data.get("image_size_px")
    calibration_size_px: tuple[float, float] | None = None
    if isinstance(calibration_size, list | tuple) and len(calibration_size) >= 2:
        calibration_size_px = (float(calibration_size[0]), float(calibration_size[1]))
    if "homography" in data:
        return tuple(float(value) for value in data["homography"]), calibration_size_px
    wall_points = data.get("wall_points_cm") or data.get("wall_points_mm") or []
    if len(image_points) < 4 or len(wall_points) < 4:
        return None
    unit_scale = 0.1 if data.get("wall_points_mm") and not data.get("wall_points_cm") else 1.0
    points = [
        CalibrationPoint(
            image_x=float(image_point[0]),
            image_y=float(image_point[1]),
            wall_x_cm=float(wall_point[0]) * unit_scale,
            wall_y_cm=float(wall_point[1]) * unit_scale,
        )
        for image_point, wall_point in zip(image_points, wall_points, strict=False)
    ]
    return image_to_wall_homography(points), calibration_size_px


def _row_frame_size(row: dict[str, Any], raw_path: Path) -> tuple[float, float] | None:
    size = row.get("source_size_px")
    if isinstance(size, dict) and size.get("width") and size.get("height"):
        return (float(size["width"]), float(size["height"]))
    if isinstance(size, list | tuple) and len(size) >= 2:
        return (float(size[0]), float(size[1]))
    if not raw_path.exists():
        return None
    with detector.Image.open(raw_path) as image:
        width, height = image.size
    return (float(width), float(height))


def _scale_image_point_for_calibration(
    image_x: float,
    image_y: float,
    *,
    frame_size: tuple[float, float] | None,
    calibration_size: tuple[float, float] | None,
) -> tuple[float, float]:
    if frame_size is None or calibration_size is None:
        return image_x, image_y
    frame_width, frame_height = frame_size
    calibration_width, calibration_height = calibration_size
    if frame_width <= 0 or frame_height <= 0:
        return image_x, image_y
    return image_x * calibration_width / frame_width, image_y * calibration_height / frame_height


def _default_alignment_reference() -> Path | None:
    for path in DEFAULT_ALIGNMENT_REFERENCE_CANDIDATES:
        if path.exists():
            return path
    return None


def _read_alignment_image(path: Path, size: tuple[float, float]) -> Any | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    target_size = (int(round(size[0])), int(round(size[1])))
    if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
        capture = cv2.VideoCapture(str(path))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    if gray.shape[::-1] != target_size:
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    return gray


def _prepare_alignment_image(gray: Any) -> Any:
    import cv2  # type: ignore[import-not-found]

    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _detect_projection_edges(gray: Any) -> dict[str, float] | None:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 130)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=180, maxLineGap=20)
    if lines is None:
        return None
    horizontals: list[tuple[float, float]] = []
    verticals: list[tuple[float, float]] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = (float(value) for value in line)
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if length < 180:
            continue
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if angle < 8 or angle > 172:
            horizontals.append(((y1 + y2) / 2.0, length))
        elif 82 < angle < 98:
            verticals.append(((x1 + x2) / 2.0, length))
    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    def weighted_mean(values: list[tuple[float, float]]) -> float:
        weight = sum(item[1] for item in values)
        return sum(item[0] * item[1] for item in values) / max(weight, 1.0)

    def edge_cluster(values: list[tuple[float, float]], *, choose_max: bool) -> float:
        extreme = max(values, key=lambda item: item[0])[0] if choose_max else min(values, key=lambda item: item[0])[0]
        tolerance = max(8.0, min(width, height) * 0.025)
        near_extreme = [item for item in values if abs(item[0] - extreme) <= tolerance]
        return weighted_mean(near_extreme)

    # Use the outer projector-wall edges. A weighted mean over every long wall
    # line is unsafe: shelves, screen texture seams, or table edges can pull the
    # bottom edge upward and turn a low cat into a fake high jump.
    top_candidates = [item for item in horizontals if item[0] < height * 0.35]
    bottom_candidates = [item for item in horizontals if item[0] > height * 0.45]
    left_candidates = [item for item in verticals if item[0] < width * 0.45]
    right_candidates = [item for item in verticals if item[0] > width * 0.5]
    if not (top_candidates and bottom_candidates and left_candidates and right_candidates):
        return None
    return {
        "top_y": edge_cluster(top_candidates, choose_max=False),
        "bottom_y": edge_cluster(bottom_candidates, choose_max=True),
        "left_x": edge_cluster(left_candidates, choose_max=False),
        "right_x": edge_cluster(right_candidates, choose_max=True),
    }


def _edge_alignment_from_edges(
    reference_edges: dict[str, float] | None,
    current_edges: dict[str, float] | None,
) -> dict[str, Any] | None:
    if not reference_edges or not current_edges:
        return None
    current_width = current_edges["right_x"] - current_edges["left_x"]
    current_height = current_edges["bottom_y"] - current_edges["top_y"]
    reference_width = reference_edges["right_x"] - reference_edges["left_x"]
    reference_height = reference_edges["bottom_y"] - reference_edges["top_y"]
    if min(current_width, current_height, reference_width, reference_height) <= 20:
        return None
    scale_x = reference_width / current_width
    scale_y = reference_height / current_height
    tx = reference_edges["left_x"] - current_edges["left_x"] * scale_x
    ty = reference_edges["top_y"] - current_edges["top_y"] * scale_y
    if not (0.85 <= scale_x <= 1.15 and 0.85 <= scale_y <= 1.15):
        return None
    if abs(tx) > 100 or abs(ty) > 100:
        return None
    import numpy as np

    return {
        "applied": True,
        "current_to_calibration": np.array([[scale_x, 0.0, tx], [0.0, scale_y, ty]], dtype=float),
        "method": "projection_edge_alignment",
        "edge_scale_x": round(float(scale_x), 5),
        "edge_scale_y": round(float(scale_y), 5),
        "edge_tx_px": round(float(tx), 2),
        "edge_ty_px": round(float(ty), 2),
        "reference_edges": {key: round(float(value), 2) for key, value in reference_edges.items()},
        "current_edges": {key: round(float(value), 2) for key, value in current_edges.items()},
    }


def _load_alignment_reference(path: Path | None, calibration_size: tuple[float, float] | None) -> dict[str, Any] | None:
    if path is None or calibration_size is None or not path.exists():
        return None
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    gray = _read_alignment_image(path, calibration_size)
    if gray is None:
        return None
    prepared = _prepare_alignment_image(gray)
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
    keypoints, descriptors = orb.detectAndCompute(prepared, None)
    if descriptors is None or len(keypoints) < 30:
        return None
    return {
        "path": str(path),
        "size": calibration_size,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "projection_edges": _detect_projection_edges(gray),
    }


def _estimate_current_to_calibration_alignment(
    raw_path: Path,
    alignment_reference: dict[str, Any] | None,
    calibration_size: tuple[float, float] | None,
) -> dict[str, Any] | None:
    if alignment_reference is None or calibration_size is None or not raw_path.exists():
        return None
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except Exception:
        return None
    current_gray = _read_alignment_image(raw_path, calibration_size)
    if current_gray is None:
        return None
    edge_alignment = _edge_alignment_from_edges(
        alignment_reference.get("projection_edges"),
        _detect_projection_edges(current_gray),
    )
    prepared = _prepare_alignment_image(current_gray)
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
    current_keypoints, current_descriptors = orb.detectAndCompute(prepared, None)
    if current_descriptors is None or len(current_keypoints) < 30:
        return edge_alignment or {
            "applied": False,
            "reason": "few_current_features",
            "current_features": len(current_keypoints),
        }
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(alignment_reference["descriptors"], current_descriptors, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        match, neighbor = pair
        if match.distance < 0.72 * neighbor.distance:
            good.append(match)
    if len(good) < 20:
        if edge_alignment:
            edge_alignment["feature_fallback_reason"] = "few_alignment_matches"
            edge_alignment["feature_match_count"] = len(good)
            return edge_alignment
        return {"applied": False, "reason": "few_alignment_matches", "match_count": len(good)}
    reference_points = np.float32([alignment_reference["keypoints"][match.queryIdx].pt for match in good])
    current_points = np.float32([current_keypoints[match.trainIdx].pt for match in good])
    ref_to_current, inliers = cv2.estimateAffinePartial2D(
        reference_points,
        current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=3000,
        confidence=0.995,
    )
    if ref_to_current is None or inliers is None:
        return {"applied": False, "reason": "alignment_affine_failed", "match_count": len(good)}
    inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 12 or inlier_count / max(1, len(good)) < 0.35:
        if edge_alignment:
            edge_alignment["feature_fallback_reason"] = "weak_alignment"
            edge_alignment["feature_match_count"] = len(good)
            edge_alignment["feature_inlier_count"] = inlier_count
            return edge_alignment
        return {
            "applied": False,
            "reason": "weak_alignment",
            "match_count": len(good),
            "inlier_count": inlier_count,
            "inlier_fraction": round(inlier_count / max(1, len(good)), 3),
        }
    current_to_ref = cv2.invertAffineTransform(ref_to_current)
    predicted = (ref_to_current @ np.c_[reference_points, np.ones(len(reference_points))].T).T
    errors = np.linalg.norm(predicted - current_points, axis=1)[inlier_mask]
    a00, _a01, tx = ref_to_current[0]
    a10, _a11, ty = ref_to_current[1]
    scale = float((a00 * a00 + a10 * a10) ** 0.5)
    rotation = float(np.degrees(np.arctan2(a10, a00)))
    return {
        "applied": True,
        "current_to_calibration": current_to_ref,
        "method": "feature_alignment",
        "match_count": len(good),
        "inlier_count": inlier_count,
        "inlier_fraction": round(inlier_count / max(1, len(good)), 3),
        "tx_px": round(float(tx), 2),
        "ty_px": round(float(ty), 2),
        "scale": round(scale, 5),
        "rotation_deg": round(rotation, 3),
        "median_error_px": round(float(np.median(errors)), 2),
        "p90_error_px": round(float(np.percentile(errors, 90)), 2),
    }


def _apply_alignment_to_point(
    calibration_x: float,
    calibration_y: float,
    alignment: dict[str, Any] | None,
) -> tuple[float, float, dict[str, Any]]:
    if not alignment or not alignment.get("applied"):
        debug = {"alignment_applied": False}
        if alignment:
            debug.update({key: value for key, value in alignment.items() if key != "current_to_calibration"})
        else:
            debug["reason"] = "alignment_unavailable"
        return calibration_x, calibration_y, debug
    import numpy as np

    matrix = alignment["current_to_calibration"]
    mapped = matrix @ np.array([calibration_x, calibration_y, 1.0], dtype=float)
    debug = {key: value for key, value in alignment.items() if key not in {"current_to_calibration"}}
    debug["aligned_calibration_image_x"] = round(float(mapped[0]), 2)
    debug["aligned_calibration_image_y"] = round(float(mapped[1]), 2)
    return float(mapped[0]), float(mapped[1]), debug


def _with_wall_coordinates_for_frame(
    point: Any,
    *,
    homography: tuple[float, ...],
    frame_size: tuple[float, float] | None,
    calibration_size: tuple[float, float] | None,
    alignment: dict[str, Any] | None = None,
) -> Any:
    calibration_x, calibration_y = _scale_image_point_for_calibration(
        float(point.image_x),
        float(point.image_y),
        frame_size=frame_size,
        calibration_size=calibration_size,
    )
    aligned_x, aligned_y, alignment_debug = _apply_alignment_to_point(calibration_x, calibration_y, alignment)
    wall_x, wall_y = transform_image_point(homography, aligned_x, aligned_y)
    debug = dict(point.debug)
    if frame_size is not None and calibration_size is not None:
        debug["coordinate_transform"] = {
            "frame_size_px": [round(frame_size[0], 2), round(frame_size[1], 2)],
            "calibration_size_px": [round(calibration_size[0], 2), round(calibration_size[1], 2)],
            "calibration_image_x": round(calibration_x, 2),
            "calibration_image_y": round(calibration_y, 2),
            "alignment": alignment_debug,
        }
    return cat_measurement.MeasurementPoint(
        point_type=point.point_type,
        image_x=point.image_x,
        image_y=point.image_y,
        wall_x_cm=wall_x,
        wall_y_cm=wall_y,
        confidence=point.confidence,
        uncertainty_px=point.uncertainty_px,
        source=point.source,
        debug=debug,
    )


def _row_recording_key(raw_path: Path) -> tuple[str, Path | None]:
    _source_video, recording_dir = review._recording_context(raw_path)  # noqa: SLF001
    if recording_dir is not None:
        return str(recording_dir), recording_dir
    group_key = review._video_group_key(  # noqa: SLF001
        review.ReviewCase(
            id="height-index",
            image_path=raw_path,
            label=raw_path.stem,
            source="rescore",
            mtime=raw_path.stat().st_mtime if raw_path.exists() else 0.0,
        )
    )
    return str(group_key), None


def _remeasure_jump_heights(
    rows: list[dict[str, Any]],
    *,
    calibration_path: Path,
    min_probability: float,
    alignment_reference_path: Path | None = None,
) -> dict[str, Any]:
    calibration = _load_height_calibration(calibration_path)
    measured_at = datetime.now(UTC).isoformat(timespec="seconds")
    if calibration is None:
        return {
            "kind": "cat_projector_active_learning_jump_heights_v1",
            "created_at": measured_at,
            "calibration_path": str(calibration_path),
            "height_min_probability": min_probability,
            "status": "skipped_missing_calibration",
            "videos": [],
        }
    homography, calibration_size = calibration
    alignment_reference = _load_alignment_reference(alignment_reference_path, calibration_size)
    alignment_stats = {
        "enabled": alignment_reference_path is not None,
        "reference": str(alignment_reference_path) if alignment_reference_path else "",
        "reference_loaded": alignment_reference is not None,
        "applied_frame_count": 0,
        "fallback_frame_count": 0,
        "fallback_reasons": {},
    }

    videos: dict[str, dict[str, Any]] = {}
    trackers: dict[str, Any] = {}
    measured_frame_count = 0
    accepted_frame_count = 0
    for row in rows:
        raw_path = Path(str(row.get("raw_path") or ""))
        if review._saved_label_says_no_cat(raw_path):  # noqa: SLF001
            row["human_review_label"] = "not_cat"
            row["review_excluded_from_jump_height"] = True
            row["review_exclusion_reason"] = "human_review_not_cat"
            continue
        point = cat_measurement.measurement_point_from_dict(row.get("best_measurement_point"))
        bbox = _parse_bbox_xywh(row.get("best_bbox"))
        if point is None and bbox is not None:
            point = cat_measurement.legacy_bbox_top_measurement_point(
                bbox,
                score=float(row.get("best_probability") or 0.0),
            )
        if point is None:
            continue
        frame_size = _row_frame_size(row, raw_path)
        alignment = _estimate_current_to_calibration_alignment(raw_path, alignment_reference, calibration_size)
        if alignment and alignment.get("applied"):
            alignment_stats["applied_frame_count"] += 1
        else:
            alignment_stats["fallback_frame_count"] += 1
            reason = str((alignment or {}).get("reason") or "alignment_unavailable")
            reasons = alignment_stats["fallback_reasons"]
            reasons[reason] = int(reasons.get(reason) or 0) + 1
        point = _with_wall_coordinates_for_frame(
            point,
            homography=homography,
            frame_size=frame_size,
            calibration_size=calibration_size,
            alignment=alignment,
        )
        wall_x_cm = float(point.wall_x_cm or 0.0)
        top_height_cm = float(point.wall_y_cm or 0.0)
        row["best_measurement_point"] = cat_measurement.measurement_point_to_dict(point)
        if bbox is not None:
            x, y, width, _height = bbox
            legacy_image_x, legacy_image_y = _scale_image_point_for_calibration(
                x + width / 2.0,
                y,
                frame_size=frame_size,
                calibration_size=calibration_size,
            )
            legacy_image_x, legacy_image_y, legacy_alignment_debug = _apply_alignment_to_point(
                legacy_image_x,
                legacy_image_y,
                alignment,
            )
            legacy_wall_x_cm, legacy_top_height_cm = transform_image_point(homography, legacy_image_x, legacy_image_y)
            row["legacy_bbox_top_wall_x_cm"] = round(float(legacy_wall_x_cm), 1)
            row["legacy_bbox_top_height_cm"] = round(float(legacy_top_height_cm), 1)
            row["legacy_bbox_alignment"] = legacy_alignment_debug
        row["best_top_wall_x_cm"] = round(float(wall_x_cm), 1)
        row["best_top_height_cm"] = round(float(top_height_cm), 1)
        row["measurement_source"] = point.point_type
        row["measurement_confidence"] = round(float(point.confidence), 4)
        measured_frame_count += 1
        key, recording_dir = _row_recording_key(raw_path)
        tracker = trackers.setdefault(key, cat_tracking.CatWallKalmanTracker())
        top_candidates = row.get("top_candidates") if isinstance(row.get("top_candidates"), list) else []
        first_candidate = top_candidates[0] if top_candidates and isinstance(top_candidates[0], dict) else {}
        area_px = int(first_candidate.get("mask_area_px") or first_candidate.get("area") or 0)
        t = float(row.get("timestamp_seconds") or (float(row.get("global_frame") or 0.0) / 15.0))
        wall_detection = cat_tracking.wall_detection_from_measurement(point, t=t, area_px=area_px)
        track_output = tracker.step(t, [wall_detection])
        if track_output is None:
            row["tracker_status"] = "reset"
            row["tracker_reason"] = "reset_after_misses"
            row["tracker_confirmed"] = False
        else:
            row["tracker_status"] = "accepted" if track_output.accepted is not None else "rejected"
            row["tracker_reason"] = track_output.reason
            row["tracker_confirmed"] = bool(track_output.confirmed)
            row["tracker_height_cm"] = (
                None if track_output.accepted_raw_y_cm is None else round(float(track_output.accepted_raw_y_cm), 1)
            )
        probability = float(row.get("best_probability") or 0.0)
        if probability < min_probability or row.get("tracker_status") != "accepted":
            continue
        accepted_frame_count += 1
        video_id = review._video_id_for_path(recording_dir or raw_path.parent)  # noqa: SLF001
        current = videos.get(key)
        candidate = {
            "video_id": video_id,
            "recording_dir": str(recording_dir) if recording_dir else None,
            "source_key": key,
            "label": recording_dir.name if recording_dir else raw_path.parent.name,
            "max_jump_height_cm": round(float(top_height_cm), 1),
            "max_frame_path": str(raw_path),
            "max_frame_index": row.get("global_frame"),
            "max_bbox": row.get("best_bbox"),
            "max_probability": probability,
            "max_source": row.get("best_source"),
            "measurement_source": row.get("measurement_source"),
            "measurement_confidence": row.get("measurement_confidence"),
            "measurement_point": row.get("best_measurement_point"),
            "legacy_bbox_top_height_cm": row.get("legacy_bbox_top_height_cm"),
        }
        if current is None or float(candidate["max_jump_height_cm"]) > float(current["max_jump_height_cm"]):
            videos[key] = candidate

    return {
        "kind": "cat_projector_active_learning_jump_heights_v1",
        "created_at": measured_at,
        "calibration_path": str(calibration_path),
        "height_min_probability": min_probability,
        "status": "done",
        "measured_frame_count": measured_frame_count,
        "accepted_frame_count": accepted_frame_count,
        "frame_alignment": alignment_stats,
        "videos": sorted(videos.values(), key=lambda item: float(item["max_jump_height_cm"]), reverse=True),
    }


def _write_rescore_outputs(
    run_dir: Path,
    *,
    rows: list[dict[str, Any]],
    model_metadata: dict[str, Any],
    training_package: dict[str, Any],
    labels: list[Path],
    jump_heights: dict[str, Any],
    detector_config: dict[str, Any],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    model_summary = {key: value for key, value in model_metadata.items() if key != "training_rows"}
    training_summary = {key: value for key, value in training_package.items() if key != "copied"}
    for row in rows:
        row["model_path"] = model_metadata.get("model_path", "")
        row["model_created_at"] = model_metadata.get("created_at", "")
        probability_uncertainty = 1.0 - min(1.0, abs(float(row["best_probability"]) - 0.5) * 2.0)
        height = float(row.get("best_top_height_cm") or 0.0)
        confidence = float(row.get("measurement_confidence") or 0.0)
        legacy_height = row.get("legacy_bbox_top_height_cm")
        disagreement = 0.0
        if legacy_height not in {None, ""}:
            disagreement = min(1.0, abs(height - float(legacy_height)) / 40.0)
        peak_impact = min(1.0, max(0.0, height - 80.0) / 80.0)
        low_confidence_at_height = peak_impact * (1.0 - confidence)
        row["uncertainty_score"] = round(
            max(probability_uncertainty, low_confidence_at_height, disagreement),
            4,
        )
        row["review_priority_score"] = round(
            probability_uncertainty * 40.0
            + peak_impact * 45.0
            + low_confidence_at_height * 35.0
            + disagreement * 30.0
            + (15.0 if row.get("detector_backend") == "legacy_contrast_catboost" else 0.0),
            4,
        )
        reasons: list[str] = []
        if probability_uncertainty >= 0.6:
            reasons.append(f"uncertain p={float(row['best_probability']):.2f}")
        if peak_impact >= 0.5:
            reasons.append(f"high measurement {height:.1f} cm")
        if low_confidence_at_height >= 0.25:
            reasons.append("low confidence near apex")
        if disagreement >= 0.3:
            reasons.append("mask/bbox height disagreement")
        if row.get("detector_backend") == "legacy_contrast_catboost":
            reasons.append("legacy detector fallback")
        if row.get("best_measurement_warning"):
            reasons.append(str(row["best_measurement_warning"]))
        row["review_priority_reasons"] = reasons or ["low impact"]
    uncertain = sorted(
        rows,
        key=lambda row: (float(row.get("review_priority_score") or 0.0), float(row["uncertainty_score"])),
        reverse=True,
    )

    probe_rows_path = run_dir / "probe_rows.json"
    uncertain_path = run_dir / "uncertain_queue.json"
    jump_heights_path = run_dir / "jump_heights.json"
    manifest_path = run_dir / "manifest.json"
    probe_rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    uncertain_path.write_text(
        json.dumps(uncertain[:500], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    jump_heights_path.write_text(
        json.dumps(jump_heights, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "kind": "cat_projector_active_learning_rescore_v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "frame_count": len(rows),
        "labels": [str(path) for path in labels],
        "model": model_summary,
        "detector_config": detector_config,
        "warnings": detector_config.get("warnings", []),
        "probe_rows": str(probe_rows_path),
        "jump_heights": str(jump_heights_path),
        "jump_height_summary": {
            "status": jump_heights.get("status"),
            "video_count": len(jump_heights.get("videos") or []),
            "max_jump_height_cm": (jump_heights.get("videos") or [{}])[0].get("max_jump_height_cm"),
            "height_min_probability": jump_heights.get("height_min_probability"),
        },
        "training_package": training_summary,
        "uncertain_queue": str(uncertain_path),
        "top_uncertain": uncertain[:20],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    latest_path = RESCORES_ROOT / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESCORES_ROOT / "jump_heights_latest.json").write_text(
        json.dumps(jump_heights, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_iteration(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or f"active-learning-{_utc_slug()}"
    run_dir = args.output_root.expanduser() / run_id
    if run_dir.exists() and not args.replace_existing:
        raise RuntimeError(f"{run_dir} already exists; pass --replace-existing to rebuild it")
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)

    detector_config = _normalised_detector_config(args)
    training_package = _materialize_review_training_package()
    labels = _collect_labels([Path(training_package["labels_csv"]), *args.labels])
    if detector_config["backend"] == "legacy":
        model_metadata = _train_model(
            labels,
            model_path=args.model.expanduser(),
            metadata_path=args.metadata.expanduser(),
            args=args,
        )
    else:
        model_metadata = _detector_model_metadata(detector_config)
    frames = _reviewable_frame_paths()
    if not frames:
        raise RuntimeError("no reviewable frames found for rescoring")
    rows = _rescore_frames(
        frames,
        model_path=args.model.expanduser(),
        metadata_path=args.metadata.expanduser(),
        jobs=args.jobs,
        detector_config=detector_config,
    )
    disable_frame_alignment = bool(getattr(args, "disable_frame_alignment", False))
    requested_alignment_reference = getattr(args, "alignment_reference", None)
    alignment_reference = (
        None if disable_frame_alignment else (requested_alignment_reference or _default_alignment_reference())
    )
    jump_heights = _remeasure_jump_heights(
        rows,
        calibration_path=args.calibration.expanduser(),
        min_probability=args.height_min_probability,
        alignment_reference_path=alignment_reference.expanduser() if alignment_reference else None,
    )
    return _write_rescore_outputs(
        run_dir,
        rows=rows,
        model_metadata=model_metadata,
        training_package=training_package,
        labels=labels,
        jump_heights=jump_heights,
        detector_config=detector_config,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, action="append", default=[])
    parser.add_argument("--model", type=Path, default=detector.DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=detector.DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--detector-backend",
        choices=("auto", "segmentation", "legacy", "fake"),
        default="auto",
        help=(
            "auto uses segmentation when --segmentation-model or "
            "CAT_PROJECTOR_SEGMENTATION_MODEL is set; legacy is explicit fallback/debug."
        ),
    )
    parser.add_argument(
        "--segmentation-model",
        type=Path,
        default=Path(DEFAULT_SEGMENTATION_MODEL) if DEFAULT_SEGMENTATION_MODEL else None,
    )
    parser.add_argument(
        "--allow-fake-detector", action="store_true", help="Allow backend=fake for tests/dev smoke only."
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=RESCORES_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--min-positive", type=int, default=5)
    parser.add_argument("--min-negative", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument(
        "--alignment-reference",
        type=Path,
        default=None,
        help=(
            "Reference calibration image/video for per-frame current-to-calibration "
            "alignment; auto-detected when omitted."
        ),
    )
    parser.add_argument(
        "--disable-frame-alignment", action="store_true", help="Use only resolution scaling before homography."
    )
    parser.add_argument("--height-min-probability", type=float, default=0.5)
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
                "jump_heights": manifest.get("jump_heights"),
                "jump_height_summary": manifest.get("jump_height_summary"),
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
