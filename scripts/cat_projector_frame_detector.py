#!/usr/bin/env python3
"""Train and apply the local Cat Projector candidate detector.

This is deliberately candidate-level. A projector-camera frame can contain Sher
and still have a false detector box on projected prey. Training rows therefore
use `label_candidate_is_cat` for the candidate box; `label_cat_present` remains
frame-level context.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from catboost import CatBoostClassifier, Pool
except Exception:  # pragma: no cover - CLI reports this clearly.
    CatBoostClassifier = None  # type: ignore[assignment]
    Pool = None  # type: ignore[assignment]


DEFAULT_DATASETS_ROOT = Path("~/.openclaw/state/cat-tv-learning/datasets").expanduser()
DEFAULT_MODELS_ROOT = Path("~/.openclaw/state/cat-tv-learning/models").expanduser()
DEFAULT_MODEL_PATH = DEFAULT_MODELS_ROOT / "cat_projector_candidate_detector_v1.cbm"
DEFAULT_METADATA_PATH = DEFAULT_MODELS_ROOT / "cat_projector_candidate_detector_v1.metadata.json"
BUNDLED_MODELS_ROOT = REPO_ROOT / "datasets" / "cat-tv-learning" / "detector-models"
BUNDLED_MODEL_PATH = BUNDLED_MODELS_ROOT / "cat_projector_candidate_detector_v1" / "cat_projector_candidate_detector_v1.cbm"
BUNDLED_METADATA_PATH = BUNDLED_MODEL_PATH.with_name("cat_projector_candidate_detector_v1.metadata.json")
PROJECTOR_POLYGON_1280 = (
    (40.22, 57.27),
    (937.92, 101.0),
    (908.0, 599.0),
    (48.16, 680.97),
)
FEATURE_NAMES = [
    "area_ratio",
    "width_ratio",
    "height_ratio",
    "x_center_ratio",
    "y_center_ratio",
    "top_y_ratio",
    "bottom_y_ratio",
    "aspect_ratio",
    "fill_ratio",
    "mean_luma",
    "median_luma",
    "dark_pixel_ratio",
    "edge_touch_left",
    "edge_touch_right",
    "edge_touch_bottom",
    "inside_projection_center",
    "inside_projection_area_ratio",
    "outside_projection_area_ratio",
]
LABEL_FIELDNAMES = [
    "image_relpath",
    "label_cat_present",
    "label_cat_playing",
    "review_status",
    "candidate_bbox_xywh",
    "label_candidate_is_cat",
    "negative_reason",
    "bbox_xywh",
    "occlusion",
    "confidence",
    "notes",
    "source_recording_dir",
    "source_chunk",
    "source_offset_seconds",
    "ha_session_id",
    "frigate_event_id",
    "video_slug",
    "candidate_reason",
]
MIN_CANDIDATE_AREA_RATIO = 0.00015
MAX_CANDIDATE_AREA_RATIO = 0.25
MAX_CANDIDATE_WIDTH_RATIO = 0.75
MAX_CANDIDATE_HEIGHT_RATIO = 0.88
DEFAULT_POST_POSITIVE_MIN_PROBABILITY = 0.55
BOTTOM_EDGE_PROPOSAL_WINDOWS_1280 = (
    (560, 140),
    (590, 120),
    (620, 100),
)
BOTTOM_EDGE_PROPOSAL_XS_1280 = tuple(range(40, 701, 80))
BOTTOM_EDGE_PROPOSAL_WIDTHS_1280 = (120, 180)


def _projector_polygon_for_size(width: int, height: int) -> tuple[tuple[float, float], ...]:
    scale_x = width / 1280.0
    scale_y = height / 720.0
    return tuple((x * scale_x, y * scale_y) for x, y in PROJECTOR_POLYGON_1280)


@dataclass(frozen=True)
class Candidate:
    bbox_xywh: tuple[float, float, float, float]
    top_x_px: float
    top_y_px: float
    area_px: int
    source: str
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class CandidatePrediction:
    candidate: Candidate | None
    cat_probability: float
    model_path: str
    model_version: str
    features: dict[str, float]


@dataclass(frozen=True)
class TrainingSample:
    image_path: Path
    row: dict[str, str]
    label: int
    bbox_xywh: tuple[float, float, float, float]
    sample_role: str


def _candidate_area_ratio(candidate: Candidate, image_width: int, image_height: int) -> float:
    _left, _top, width, height = candidate.bbox_xywh
    return (width * height) / float(image_width * image_height)


def _is_plausible_candidate(candidate: Candidate, image_width: int, image_height: int) -> bool:
    _left, _top, width, height = candidate.bbox_xywh
    area_ratio = _candidate_area_ratio(candidate, image_width, image_height)
    if area_ratio < MIN_CANDIDATE_AREA_RATIO or area_ratio > MAX_CANDIDATE_AREA_RATIO:
        return False
    if width / float(image_width) > MAX_CANDIDATE_WIDTH_RATIO:
        return False
    if height / float(image_height) > MAX_CANDIDATE_HEIGHT_RATIO:
        return False
    return True


def _parse_bbox_xywh(raw: str) -> tuple[float, float, float, float] | None:
    parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (float(part) for part in parts)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _format_bbox_xywh(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(str(int(round(value))) for value in bbox)


def _default_font(size: int) -> Any:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_scaled_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    *,
    scale: float,
    x_offset: int,
    y_offset: int,
    color: tuple[int, int, int],
    width: int = 4,
) -> None:
    x, y, w, h = bbox
    left = x_offset + int(round(x * scale))
    top = y_offset + int(round(y * scale))
    right = x_offset + int(round((x + w) * scale))
    bottom = y_offset + int(round((y + h) * scale))
    for inset in range(width):
        draw.rectangle((left - inset, top - inset, right + inset, bottom + inset), outline=color)


def _ellipsize_text(draw: ImageDraw.ImageDraw, text: str, *, font: Any, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    while text and draw.textlength(text + suffix, font=font) > max_width:
        text = text[:-1]
    return text + suffix if text else suffix


def _frame_cat_status(row: dict[str, str]) -> tuple[str, tuple[int, int, int], tuple[float, float, float, float] | None]:
    """Return the frame-level cat answer; candidate negatives are not no-cat proof."""

    frame_label = str(row.get("label_cat_present") or "").strip().lower()
    cat_bbox = _parse_bbox_xywh(str(row.get("bbox_xywh") or ""))

    if frame_label == "yes":
        if cat_bbox is not None:
            return "CAT", (78, 220, 120), cat_bbox
        return "CAT VISIBLE - BBOX MISSING", (255, 190, 80), None
    if frame_label == "no":
        return "NO CAT IN FRAME", (235, 95, 95), None
    return "UNSURE - REVIEW FRAME", (170, 170, 170), None


def _projection_mask(width: int, height: int) -> np.ndarray:
    polygon = _projector_polygon_for_size(width, height)
    mask_img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_img).polygon(polygon, fill=255)
    return np.array(mask_img) > 0


def _bottom_edge_window_candidates(width: int, height: int) -> list[Candidate]:
    """Generate model-scored proposals for Sher sitting at the projector bottom edge."""

    scale_x = width / 1280.0
    scale_y = height / 720.0
    candidates: list[Candidate] = []
    for y_1280, h_1280 in BOTTOM_EDGE_PROPOSAL_WINDOWS_1280:
        y = y_1280 * scale_y
        h = h_1280 * scale_y
        for x_1280 in BOTTOM_EDGE_PROPOSAL_XS_1280:
            x = x_1280 * scale_x
            for w_1280 in BOTTOM_EDGE_PROPOSAL_WIDTHS_1280:
                w = w_1280 * scale_x
                if x + w > width:
                    continue
                candidates.append(
                    Candidate(
                        bbox_xywh=(float(x), float(y), float(w), float(h)),
                        top_x_px=float(x + w / 2.0),
                        top_y_px=float(y),
                        area_px=int(round(w * h)),
                        source="bottom_edge_window_proposal",
                    )
                )
    return candidates


def _robust_linear_match(source: np.ndarray, camera: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    source_values = source[mask].astype(np.float32)
    camera_values = camera[mask].astype(np.float32)
    keep = (source_values > 45) & (camera_values > 35)
    if int(keep.sum()) < 200:
        return 1.0, 0.0
    source_values = source_values[keep]
    camera_values = camera_values[keep]
    source_low, source_high = np.percentile(source_values, [10, 90])
    camera_low, camera_high = np.percentile(camera_values, [10, 90])
    source_span = max(1.0, float(source_high - source_low))
    scale = max(0.1, min(4.0, float(camera_high - camera_low) / source_span))
    offset = float(np.median(camera_values - scale * source_values))
    return scale, offset


def _match_background_luma(background: Image.Image, camera_frame: Image.Image, mask: np.ndarray) -> Image.Image:
    background_arr = np.asarray(background.convert("L").resize(camera_frame.size), dtype=np.uint8)
    camera_arr = np.asarray(camera_frame.convert("L"), dtype=np.uint8)
    fit_mask = mask & (background_arr > 45) & (camera_arr > 35)
    if int(fit_mask.sum()) >= 200:
        median_abs_diff = float(np.median(np.abs(background_arr[fit_mask].astype(np.int16) - camera_arr[fit_mask].astype(np.int16))))
        if median_abs_diff <= 3.0:
            return Image.fromarray(background_arr, mode="L")
    scale, offset = _robust_linear_match(background_arr, camera_arr, mask)
    matched = np.clip(background_arr.astype(np.float32) * scale + offset, 0, 255)
    return Image.fromarray(matched.astype(np.uint8), mode="L")


def _warp_source_to_camera(source_frame: Image.Image, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - runtime dependency is installed on projector host.
        raise RuntimeError("opencv-python is required for source-subtracted projector candidates") from exc

    source_gray = np.asarray(source_frame.convert("L").resize((1280, 720)), dtype=np.uint8)
    source_points = np.float32([[0, 0], [1279, 0], [1279, 719], [0, 719]])
    camera_points = np.float32(_projector_polygon_for_size(width, height))
    homography = cv2.getPerspectiveTransform(source_points, camera_points)
    return cv2.warpPerspective(source_gray, homography, (width, height))


def source_subtracted_residual(
    camera_frame: Image.Image,
    *,
    source_frame: Image.Image,
    room_background: Image.Image | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dark residual after explaining the projector plane by source video."""

    camera_gray = np.asarray(camera_frame.convert("L"), dtype=np.uint8)
    height, width = camera_gray.shape
    projection = _projection_mask(width, height)
    warped_source = _warp_source_to_camera(source_frame, width, height)
    if room_background is None:
        expected = camera_gray.astype(np.float32)
    else:
        matched_background = _match_background_luma(room_background, camera_frame, ~projection)
        expected = np.asarray(matched_background.convert("L").resize((width, height)), dtype=np.uint8).astype(np.float32)

    fit_mask = projection.copy()
    fit_mask[int(height * 0.69) :, :] = False
    fit_mask[:, int(width * 0.82) :] = False
    scale, offset = _robust_linear_match(warped_source, camera_gray, fit_mask)
    expected[projection] = np.clip(scale * warped_source[projection].astype(np.float32) + offset, 0, 255)
    return expected - camera_gray.astype(np.float32), warped_source


def detect_source_subtracted_candidate_components(
    camera_frame: Image.Image,
    *,
    source_frame: Image.Image,
    room_background: Image.Image | None = None,
    residual_baseline: np.ndarray | Image.Image | None = None,
    threshold: float = 26.0,
    source_bright_threshold: int = 150,
    min_area_px: int = 40,
) -> list[Candidate]:
    """Find candidates after subtracting the known projected source frame.

    The source video explains projected prey. Remaining dark residuals are
    candidate boxes for the cat/not-cat model, not final cat decisions.
    """

    try:
        from scipy import ndimage
    except Exception as exc:  # pragma: no cover - scipy is a runtime dependency here.
        raise RuntimeError("scipy is required for source-subtracted projector candidates") from exc

    width, height = camera_frame.size
    projection = _projection_mask(width, height)
    residual, warped_source = source_subtracted_residual(
        camera_frame,
        source_frame=source_frame,
        room_background=room_background,
    )
    if residual_baseline is not None:
        if isinstance(residual_baseline, Image.Image):
            baseline = np.asarray(residual_baseline.convert("F").resize((width, height)), dtype=np.float32)
        else:
            baseline = np.asarray(residual_baseline, dtype=np.float32)
            if baseline.shape != residual.shape:
                raise ValueError(
                    f"residual_baseline shape {baseline.shape} does not match frame shape {residual.shape}"
                )
        residual = residual - baseline
    mask = residual > threshold
    mask &= (~projection) | (warped_source > source_bright_threshold)
    mask[: max(2, int(height * 0.05)), :] = False
    mask[:, : max(2, int(width * 0.015))] = False
    mask[:, int(width * 0.82) :] = False
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
    mask = ndimage.binary_closing(mask, structure=np.ones((4, 4)))

    labels, count = ndimage.label(mask)
    candidates: list[Candidate] = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if len(xs) < min_area_px:
            continue
        left = int(xs.min())
        right = int(xs.max()) + 1
        top = int(ys.min())
        bottom = int(ys.max()) + 1
        candidate_width = right - left
        candidate_height = bottom - top
        if candidate_width < 8 or candidate_height < 10:
            continue
        if candidate_width > int(width * 0.28) or candidate_height > int(height * 0.36):
            continue
        top_xs = xs[ys == ys.min()]
        component_mask = labels == label
        candidates.append(
            Candidate(
                bbox_xywh=(float(left), float(top), float(candidate_width), float(candidate_height)),
                top_x_px=float(top_xs.mean()),
                top_y_px=float(top),
                area_px=int(len(xs)),
                source="source_subtracted_projector_residual",
                mask=component_mask.copy(),
            )
        )
    return sorted(candidates, key=lambda item: item.area_px, reverse=True)


def _bbox_mask(mask: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    height, width = mask.shape
    x, y, w, h = bbox
    left = max(0, min(width, int(np.floor(x))))
    top = max(0, min(height, int(np.floor(y))))
    right = max(left + 1, min(width, int(np.ceil(x + w))))
    bottom = max(top + 1, min(height, int(np.ceil(y + h))))
    out = np.zeros_like(mask, dtype=bool)
    out[top:bottom, left:right] = True
    return out


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_intersection_ratio(
    candidate: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> float:
    cx, cy, cw, ch = candidate
    rx, ry, rw, rh = reference
    left = max(cx, rx)
    top = max(cy, ry)
    right = min(cx + cw, rx + rw)
    bottom = min(cy + ch, ry + rh)
    if right <= left or bottom <= top:
        return 0.0
    candidate_area = cw * ch
    if candidate_area <= 0:
        return 0.0
    return ((right - left) * (bottom - top)) / candidate_area


def _erase_bbox_for_negative_mining(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    margin_ratio: float = 0.10,
) -> Image.Image:
    """Hide the confirmed cat box so the second detector pass finds adjacent false cats."""

    erased = image.convert("RGB").copy()
    arr = np.asarray(erased, dtype=np.uint8)
    height, width = arr.shape[:2]
    x, y, w, h = bbox
    margin = max(4, int(round(max(w, h) * margin_ratio)))
    left = max(0, int(np.floor(x)) - margin)
    top = max(0, int(np.floor(y)) - margin)
    right = min(width, int(np.ceil(x + w)) + margin)
    bottom = min(height, int(np.ceil(y + h)) + margin)
    if left >= right or top >= bottom:
        return erased
    projection = _projection_mask(width, height)
    surrounding = projection.copy()
    surrounding[top:bottom, left:right] = False
    if surrounding.any():
        fill = tuple(int(value) for value in np.median(arr[surrounding], axis=0))
    else:
        fill = tuple(int(value) for value in np.median(arr.reshape(-1, 3), axis=0))
    ImageDraw.Draw(erased).rectangle((left, top, right, bottom), fill=fill)
    return erased


def extract_candidate_features(image: Image.Image, bbox_xywh: tuple[float, float, float, float]) -> dict[str, float]:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    height, width = arr.shape
    x, y, w, h = bbox_xywh
    left = max(0, min(width - 1, int(np.floor(x))))
    top = max(0, min(height - 1, int(np.floor(y))))
    right = max(left + 1, min(width, int(np.ceil(x + w))))
    bottom = max(top + 1, min(height, int(np.ceil(y + h))))
    crop = arr[top:bottom, left:right]
    dark = crop < 110
    projection = _projection_mask(width, height)
    box_mask = _bbox_mask(projection, bbox_xywh)
    box_area = max(1, int(box_mask.sum()))
    inside_projection_area = int((box_mask & projection).sum())
    outside_projection_area = box_area - inside_projection_area
    area = max(1.0, float((right - left) * (bottom - top)))
    return {
        "area_ratio": area / float(width * height),
        "width_ratio": (right - left) / float(width),
        "height_ratio": (bottom - top) / float(height),
        "x_center_ratio": ((left + right) / 2.0) / float(width),
        "y_center_ratio": ((top + bottom) / 2.0) / float(height),
        "top_y_ratio": top / float(height),
        "bottom_y_ratio": bottom / float(height),
        "aspect_ratio": (right - left) / max(1.0, float(bottom - top)),
        "fill_ratio": float(dark.mean()) if dark.size else 0.0,
        "mean_luma": float(crop.mean()) if crop.size else 255.0,
        "median_luma": float(np.median(crop)) if crop.size else 255.0,
        "dark_pixel_ratio": float(dark.mean()) if dark.size else 0.0,
        "edge_touch_left": 1.0 if left <= 2 else 0.0,
        "edge_touch_right": 1.0 if right >= width - 2 else 0.0,
        "edge_touch_bottom": 1.0 if bottom >= height - 2 else 0.0,
        "inside_projection_center": 1.0 if projection[min(height - 1, int((top + bottom) / 2)), min(width - 1, int((left + right) / 2))] else 0.0,
        "inside_projection_area_ratio": inside_projection_area / float(box_area),
        "outside_projection_area_ratio": outside_projection_area / float(box_area),
    }


def detect_dark_candidate_components(image: Image.Image) -> list[Candidate]:
    from scipy import ndimage

    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    projection = _projection_mask(image.width, image.height)
    screen_values = arr[projection]
    median = float(np.median(screen_values)) if screen_values.size else 180.0
    interior_iterations = max(3, min(image.width, image.height) // 80)
    projection_interior = ndimage.binary_erosion(projection, iterations=interior_iterations)
    masks = {
        "dark_body_inside_projection_interior": (arr < min(140, median - 35)) & projection_interior,
        "dark_body_on_projection": (arr < min(115, median - 45)) & projection,
        "dark_motion_or_body_outside_projection": (arr < 95) & ~projection,
    }
    candidates: list[Candidate] = []
    for source, mask in masks.items():
        labels, count = ndimage.label(mask)
        for label in range(1, count + 1):
            ys, xs = np.nonzero(labels == label)
            if len(xs) < 120:
                continue
            left = int(xs.min())
            right = int(xs.max()) + 1
            top = int(ys.min())
            bottom = int(ys.max()) + 1
            width = right - left
            height = bottom - top
            if width < 8 or height < 8:
                continue
            top_xs = xs[ys == ys.min()]
            candidates.append(
                Candidate(
                    bbox_xywh=(float(left), float(top), float(width), float(height)),
                    top_x_px=float(top_xs.mean()),
                    top_y_px=float(top),
                    area_px=int(len(xs)),
                    source=source,
                )
            )
    return sorted(candidates, key=lambda item: item.area_px, reverse=True)


def _row_label(row: dict[str, str]) -> int | None:
    candidate_label = str(row.get("label_candidate_is_cat") or "").strip().lower()
    if candidate_label in {"yes", "true", "1"}:
        return 1
    if candidate_label in {"no", "false", "0"}:
        return 0
    return None


def _training_samples_from_labels(paths: Iterable[Path]) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for labels_path in paths:
        package_dir = labels_path.parent
        with labels_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_relpath = str(row.get("image_relpath") or "")
                if not image_relpath:
                    continue
                image_path = package_dir / image_relpath
                candidate_bbox = _parse_bbox_xywh(str(row.get("candidate_bbox_xywh") or ""))
                candidate_label = _row_label(row)
                if candidate_bbox is not None and candidate_label is not None:
                    samples.append(
                        TrainingSample(
                            image_path=image_path,
                            row=row,
                            label=candidate_label,
                            bbox_xywh=candidate_bbox,
                            sample_role="candidate_bbox",
                        )
                    )

                frame_label = str(row.get("label_cat_present") or "").strip().lower()
                human_bbox = _parse_bbox_xywh(str(row.get("bbox_xywh") or ""))
                if frame_label == "yes" and human_bbox is not None:
                    if candidate_label == 1 and candidate_bbox == human_bbox:
                        continue
                    samples.append(
                        TrainingSample(
                            image_path=image_path,
                            row=row,
                            label=1,
                            bbox_xywh=human_bbox,
                            sample_role="human_bbox",
                        )
                    )
    return samples


def _features_for_bbox(image_path: Path, bbox: tuple[float, float, float, float]) -> dict[str, float] | None:
    image = Image.open(image_path).convert("RGB")
    candidate = Candidate(bbox, bbox[0], bbox[1], int(bbox[2] * bbox[3]), "label")
    if not _is_plausible_candidate(candidate, image.width, image.height):
        return None
    return extract_candidate_features(image, bbox)


def train(args: argparse.Namespace) -> int:
    if CatBoostClassifier is None or Pool is None:
        raise RuntimeError("catboost is required to train the cat projector detector")
    samples = _training_samples_from_labels([path.expanduser() for path in args.labels])
    vectors: list[list[float]] = []
    labels: list[int] = []
    used_rows: list[dict[str, Any]] = []
    skipped = 0
    for sample in samples:
        if not sample.image_path.exists():
            skipped += 1
            continue
        features = _features_for_bbox(sample.image_path, sample.bbox_xywh)
        if features is None:
            skipped += 1
            continue
        vectors.append([features[name] for name in FEATURE_NAMES])
        labels.append(sample.label)
        used_rows.append(
            {
                "image": str(sample.image_path),
                "label": sample.label,
                "bbox_xywh": _format_bbox_xywh(sample.bbox_xywh),
                "sample_role": sample.sample_role,
            }
        )
    positive_count = int(sum(labels))
    negative_count = int(len(labels) - positive_count)
    if positive_count < args.min_positive or negative_count < args.min_negative:
        raise RuntimeError(
            f"need at least {args.min_positive} positive and {args.min_negative} negative candidate labels; "
            f"got positive={positive_count} negative={negative_count} skipped={skipped}"
        )

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="Logloss",
        verbose=False,
        random_seed=args.seed,
        allow_writing_files=False,
    )
    model.fit(Pool(vectors, labels, feature_names=FEATURE_NAMES))
    model_path = args.out.expanduser()
    metadata_path = args.metadata.expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    metadata = {
        "kind": "cat_projector_candidate_detector_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "feature_names": FEATURE_NAMES,
        "model_path": str(model_path),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "skipped_count": skipped,
        "training_rows": used_rows,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


def _resolve_model_paths(model_path: Path, metadata_path: Path) -> tuple[Path, Path]:
    model_path = model_path.expanduser()
    metadata_path = metadata_path.expanduser()
    if model_path.exists() and metadata_path.exists():
        return model_path, metadata_path
    if (
        model_path == DEFAULT_MODEL_PATH.expanduser()
        and metadata_path == DEFAULT_METADATA_PATH.expanduser()
        and BUNDLED_MODEL_PATH.exists()
        and BUNDLED_METADATA_PATH.exists()
    ):
        return BUNDLED_MODEL_PATH, BUNDLED_METADATA_PATH
    return model_path, metadata_path


def default_model_available() -> bool:
    return (
        (DEFAULT_MODEL_PATH.exists() and DEFAULT_METADATA_PATH.exists())
        or (BUNDLED_MODEL_PATH.exists() and BUNDLED_METADATA_PATH.exists())
    )


def load_model(model_path: Path = DEFAULT_MODEL_PATH, metadata_path: Path = DEFAULT_METADATA_PATH) -> tuple[Any, dict[str, Any]]:
    if CatBoostClassifier is None or Pool is None:
        raise RuntimeError("catboost is required to score the cat projector detector")
    model_path, metadata_path = _resolve_model_paths(model_path, metadata_path)
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"cat projector detector model is missing: {model_path} / {metadata_path}")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_path"] = str(model_path)
    metadata["metadata_path"] = str(metadata_path)
    return model, metadata


def score_candidates(
    image: Image.Image,
    *,
    model: Any,
    metadata: dict[str, Any],
    candidates: list[Candidate] | None = None,
) -> list[CandidatePrediction]:
    if Pool is None:
        raise RuntimeError("catboost Pool is unavailable")
    candidates = candidates if candidates is not None else detect_dark_candidate_components(image)
    candidates = [candidate for candidate in candidates if _is_plausible_candidate(candidate, image.width, image.height)]
    if not candidates:
        return []
    vectors = []
    feature_maps = []
    for candidate in candidates:
        features = extract_candidate_features(image, candidate.bbox_xywh)
        feature_maps.append(features)
        vectors.append([features[name] for name in FEATURE_NAMES])
    probabilities = model.predict_proba(Pool(vectors, feature_names=FEATURE_NAMES))
    return [
        CandidatePrediction(
            candidate=candidate,
            cat_probability=float(probabilities[index][1]),
            model_path=str(metadata.get("model_path") or ""),
            model_version=str(metadata.get("created_at") or ""),
            features=feature_maps[index],
        )
        for index, candidate in enumerate(candidates)
    ]


def best_cat_candidate(
    image: Image.Image,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    min_probability: float = 0.55,
) -> CandidatePrediction:
    model, metadata = load_model(model_path, metadata_path)
    return best_cat_candidate_from_model(
        image,
        model=model,
        metadata=metadata,
        fallback_model_path=model_path,
        min_probability=min_probability,
    )


def best_cat_candidate_from_model(
    image: Image.Image,
    *,
    model: Any,
    metadata: dict[str, Any],
    fallback_model_path: Path,
    min_probability: float = 0.55,
) -> CandidatePrediction:
    predictions = sorted(
        score_candidates(image, model=model, metadata=metadata),
        key=lambda item: item.cat_probability,
        reverse=True,
    )
    if not predictions:
        return CandidatePrediction(None, 0.0, str(fallback_model_path), str(metadata.get("created_at") or ""), {})
    best = predictions[0]
    if best.cat_probability < min_probability:
        return CandidatePrediction(None, best.cat_probability, best.model_path, best.model_version, best.features)
    return best


def _extract_frame(source: Path, offset_seconds: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{offset_seconds:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not output.exists() or output.stat().st_size < 1024:
        raise RuntimeError(f"failed to extract {source} at {offset_seconds:.3f}s: {result.stderr[-400:]}")


def _chunk_path(recording_dir: Path, chunk_index: int) -> Path:
    path = recording_dir / f"chunk_{chunk_index:04d}.mp4"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_candidate_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("chosen") or payload.get("records") or []
    else:
        records = payload
    return [record for record in records if isinstance(record, dict)]


def mine_hard_negatives(args: argparse.Namespace) -> int:
    recording_dir = args.recording_dir.expanduser()
    output_dir = args.output_root.expanduser() / args.package_id
    if output_dir.exists() and not args.replace_existing:
        raise RuntimeError(f"{output_dir} already exists; pass --replace-existing to rebuild it")
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((recording_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    records = _load_candidate_records(args.candidate_scores.expanduser())
    for record in records[: args.limit]:
        chunk_index = int(record["chunk"])
        offset_seconds = float(record.get("offset") if record.get("offset") is not None else record.get("offset_seconds") or 0.0)
        source = _chunk_path(recording_dir, chunk_index)
        image_name = f"{recording_dir.name}__chunk_{chunk_index:04d}__{int(round(offset_seconds * 1000)):06d}ms.jpg"
        image_path = frames_dir / image_name
        _extract_frame(source, offset_seconds, image_path)
        bbox = record.get("bbox")
        best = record.get("best") if isinstance(record.get("best"), dict) else {}
        if bbox is None and best:
            bbox = [best.get("x0"), best.get("y0"), best.get("x1"), best.get("y1")]
        candidate_boxes: list[tuple[float, float, float, float]] = []
        if isinstance(bbox, list) and len(bbox) == 4 and all(value is not None for value in bbox):
            x0, y0, x1, y1 = (float(value) for value in bbox)
            scale_x = 1280.0 / 640.0
            scale_y = 720.0 / 360.0
            candidate_boxes.append((x0 * scale_x, y0 * scale_y, (x1 - x0) * scale_x, (y1 - y0) * scale_y))
        image = Image.open(image_path).convert("RGB")
        for candidate in detect_dark_candidate_components(image)[: args.max_auto_candidates_per_frame]:
            if _is_plausible_candidate(candidate, image.width, image.height):
                candidate_boxes.append(candidate.bbox_xywh)
        seen_boxes: set[str] = set()
        for candidate_bbox in candidate_boxes:
            formatted_bbox = _format_bbox_xywh(candidate_bbox)
            if not formatted_bbox or formatted_bbox in seen_boxes:
                continue
            seen_boxes.add(formatted_bbox)
            row = {
                "image_relpath": image_path.relative_to(output_dir).as_posix(),
                "label_cat_present": "unsure",
                "label_cat_playing": "unsure",
                "review_status": "machine_prefilled_needs_human_review",
                "candidate_bbox_xywh": formatted_bbox,
                "label_candidate_is_cat": "no",
                "negative_reason": args.negative_reason,
                "bbox_xywh": "",
                "occlusion": "",
                "confidence": "low",
                "notes": f"hard negative mined from detector/scorer false-positive review; original reason={record.get('reason') or 'unknown'} score={record.get('score')}",
                "source_recording_dir": str(recording_dir),
                "source_chunk": str(source),
                "source_offset_seconds": f"{offset_seconds:.3f}",
                "ha_session_id": manifest.get("ha_session_id") or "",
                "frigate_event_id": manifest.get("frigate_event_id") or "",
                "video_slug": str(manifest.get("video_url") or "").rsplit("/", 1)[-1],
                "candidate_reason": "hard_negative_detector_false_positive",
            }
            rows.append(row)

    labels_path = output_dir / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "kind": "cat_projector_hard_negative_candidate_package_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate_scores": str(args.candidate_scores.expanduser()),
        "frame_count": len(rows),
        "labels_csv": str(labels_path),
        "package_dir": str(output_dir),
        "recording_dir": str(recording_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


def mine_post_positive_hard_negatives(args: argparse.Namespace) -> int:
    labels_path = args.labels.expanduser()
    source_package = labels_path.parent
    output_dir = args.output_root.expanduser() / args.package_id
    if output_dir.exists() and not args.replace_existing:
        raise RuntimeError(f"{output_dir} already exists; pass --replace-existing to rebuild it")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    frames_dir = output_dir / "frames"
    debug_dir = output_dir / "erased-debug"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_erased_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    model, metadata = load_model(args.model.expanduser(), args.metadata.expanduser())
    rows: list[dict[str, Any]] = []
    scanned_positive_frames = 0
    skipped_overlap = 0
    skipped_low_probability = 0
    seen_keys: set[tuple[str, str]] = set()
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for source_index, source_row in enumerate(csv.DictReader(handle)):
            if len(rows) >= args.max_rows:
                break
            frame_label = str(source_row.get("label_cat_present") or "").strip().lower()
            human_bbox = _parse_bbox_xywh(str(source_row.get("bbox_xywh") or ""))
            image_relpath = str(source_row.get("image_relpath") or "")
            if frame_label != "yes" or human_bbox is None or not image_relpath:
                continue
            source_image = source_package / image_relpath
            if not source_image.exists():
                continue

            image = Image.open(source_image).convert("RGB")
            erased = _erase_bbox_for_negative_mining(image, human_bbox, margin_ratio=args.erase_margin_ratio)
            predictions = sorted(
                score_candidates(erased, model=model, metadata=metadata),
                key=lambda item: item.cat_probability,
                reverse=True,
            )
            scanned_positive_frames += 1
            target_name = f"{Path(image_relpath).stem}__post_positive_{source_index:04d}.jpg"
            target = frames_dir / target_name
            copied_source = False
            if args.keep_erased_debug:
                erased.save(debug_dir / target_name, quality=92)

            accepted_in_frame = 0
            for prediction in predictions:
                if accepted_in_frame >= args.max_candidates_per_frame or len(rows) >= args.max_rows:
                    break
                if prediction.candidate is None:
                    continue
                if prediction.cat_probability < args.min_probability:
                    skipped_low_probability += 1
                    continue
                candidate_bbox = prediction.candidate.bbox_xywh
                if (
                    _bbox_iou(candidate_bbox, human_bbox) > args.max_cat_iou
                    or _bbox_intersection_ratio(candidate_bbox, human_bbox) > args.max_cat_candidate_overlap
                ):
                    skipped_overlap += 1
                    continue
                formatted_bbox = _format_bbox_xywh(candidate_bbox)
                key = (str(source_image), formatted_bbox)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if not copied_source:
                    shutil.copy2(source_image, target)
                    copied_source = True
                row = {field: "" for field in LABEL_FIELDNAMES}
                row.update(source_row)
                row.update(
                    {
                        "image_relpath": target.relative_to(output_dir).as_posix(),
                        "label_cat_present": "yes",
                        "label_cat_playing": str(source_row.get("label_cat_playing") or "unsure"),
                        "review_status": "machine_prefilled_needs_human_review",
                        "candidate_bbox_xywh": formatted_bbox,
                        "label_candidate_is_cat": "no",
                        "negative_reason": args.negative_reason,
                        "confidence": "low",
                        "notes": (
                            "post-positive hard negative: confirmed Sher bbox was erased, "
                            f"then detector still scored this non-overlapping candidate as cat "
                            f"p={prediction.cat_probability:.3f}; original_image={image_relpath}; "
                            f"human_bbox_xywh={_format_bbox_xywh(human_bbox)}"
                        ),
                        "candidate_reason": "post_positive_erased_cat_false_positive",
                    }
                )
                rows.append(row)
                accepted_in_frame += 1

    labels_out = output_dir / "labels.csv"
    with labels_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    output_metadata = {
        "kind": "cat_projector_post_positive_hard_negative_package_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frame_count": len(rows),
        "labels_csv": str(labels_out),
        "max_cat_iou": args.max_cat_iou,
        "max_cat_candidate_overlap": args.max_cat_candidate_overlap,
        "min_probability": args.min_probability,
        "model": str(args.model.expanduser()),
        "model_created_at": metadata.get("created_at"),
        "package_dir": str(output_dir),
        "scanned_positive_frames": scanned_positive_frames,
        "skipped_low_probability": skipped_low_probability,
        "skipped_overlap": skipped_overlap,
        "source_labels": str(labels_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(output_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output_metadata, ensure_ascii=False, sort_keys=True))
    return 0


def bootstrap_live_jump_events(args: argparse.Namespace) -> int:
    events_path = args.events.expanduser()
    source_dir = events_path.parent
    output_dir = args.output_root.expanduser() / args.package_id
    if output_dir.exists() and not args.replace_existing:
        raise RuntimeError(f"{output_dir} already exists; pass --replace-existing to rebuild it")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= args.limit:
                break
            event = json.loads(line)
            image_path = Path(str(event.get("image_path") or ""))
            if not image_path.exists():
                continue
            bbox_xyxy = event.get("component_bbox_xyxy")
            if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
                continue
            x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
            bbox_xywh = (x0, y0, x1 - x0, y1 - y0)
            image = Image.open(image_path).convert("RGB")
            candidate = Candidate(bbox_xywh, x0, y0, int(max(0.0, x1 - x0) * max(0.0, y1 - y0)), "live_jump_event")
            if not _is_plausible_candidate(candidate, image.width, image.height):
                continue
            target = frames_dir / image_path.name
            shutil.copy2(image_path, target)
            row = {field: "" for field in LABEL_FIELDNAMES}
            row.update(
                {
                    "image_relpath": target.relative_to(output_dir).as_posix(),
                    "label_cat_present": "unsure",
                    "label_cat_playing": "unsure",
                    "review_status": "machine_prefilled_needs_human_review",
                    "candidate_bbox_xywh": _format_bbox_xywh(bbox_xywh),
                    "label_candidate_is_cat": "unsure",
                    "negative_reason": "",
                    "bbox_xywh": "",
                    "occlusion": "",
                    "confidence": "low",
                    "notes": (
                        "candidate from live jump watcher event; this is a review candidate, "
                        "not a positive label until a human confirms the box is on Sher"
                    ),
                    "source_recording_dir": str(source_dir),
                    "source_chunk": "",
                    "source_offset_seconds": "",
                    "ha_session_id": "",
                    "frigate_event_id": "",
                    "video_slug": "live-jump-watch",
                    "candidate_reason": "live_jump_event_candidate",
                }
            )
            rows.append(row)

    labels_path = output_dir / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "kind": "cat_projector_live_jump_event_candidate_package_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "events": str(events_path),
        "frame_count": len(rows),
        "labels_csv": str(labels_path),
        "package_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


def bootstrap_candidates(args: argparse.Namespace) -> int:
    labels_path = args.labels.expanduser()
    source_package = labels_path.parent
    output_dir = args.output_root.expanduser() / args.package_id
    if output_dir.exists() and not args.replace_existing:
        raise RuntimeError(f"{output_dir} already exists; pass --replace-existing to rebuild it")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for source_row in csv.DictReader(handle):
            frame_label = str(source_row.get("label_cat_present") or "").strip().lower()
            if frame_label not in {"yes", "no"}:
                continue
            image_relpath = str(source_row.get("image_relpath") or "")
            source_image = source_package / image_relpath
            if not source_image.exists():
                continue
            image = Image.open(source_image).convert("RGB")
            candidates = [
                candidate
                for candidate in detect_dark_candidate_components(image)
                if _is_plausible_candidate(candidate, image.width, image.height)
            ]
            if frame_label == "yes":
                candidates = candidates[:1]
                candidate_label = "unsure"
                negative_reason = ""
            else:
                candidates = candidates[: args.max_negative_candidates_per_frame]
                candidate_label = "no"
                negative_reason = "projected_prey_or_empty_wall"
            for index, candidate in enumerate(candidates):
                target_name = f"{Path(image_relpath).stem}__candidate_{index:02d}.jpg"
                target = frames_dir / target_name
                if not target.exists():
                    shutil.copy2(source_image, target)
                row = {field: "" for field in LABEL_FIELDNAMES}
                row.update(source_row)
                row.update(
                    {
                        "image_relpath": target.relative_to(output_dir).as_posix(),
                        "review_status": "machine_prefilled_needs_human_review",
                        "candidate_bbox_xywh": _format_bbox_xywh(candidate.bbox_xywh),
                        "label_candidate_is_cat": candidate_label,
                        "negative_reason": negative_reason,
                        "confidence": "low",
                        "notes": (
                            f"candidate-level bootstrap from frame label {frame_label}; "
                            f"candidate_source={candidate.source}; original_image={image_relpath}; "
                            f"needs human review before leaderboard use"
                        ),
                    }
                )
                rows.append(row)
            if len(rows) >= args.max_rows:
                break

    labels_out = output_dir / "labels.csv"
    with labels_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "kind": "cat_projector_candidate_bootstrap_package_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frame_count": len(rows),
        "labels_csv": str(labels_out),
        "package_dir": str(output_dir),
        "source_labels": str(labels_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


def render_frame_cat_report(args: argparse.Namespace) -> int:
    labels_path = args.labels.expanduser()
    package_dir = labels_path.parent
    output_path = args.out.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"no rows in {labels_path}")

    tile_w = args.tile_width
    image_h = int(round(tile_w * 9 / 16))
    caption_h = 72
    tile_h = image_h + caption_h
    columns = max(1, args.columns)
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, sheet_rows * tile_h), (14, 17, 21))
    draw = ImageDraw.Draw(sheet)
    title_font = _default_font(20)
    small_font = _default_font(13)
    summary = {
        "kind": "cat_projector_frame_cat_report_v1",
        "labels": str(labels_path),
        "out": str(output_path),
        "rows": len(rows),
        "cat": 0,
        "no_cat": 0,
        "missing_cat_bbox": 0,
        "unsure": 0,
        "missing_images": 0,
    }

    for index, row in enumerate(rows):
        col = index % columns
        row_index = index // columns
        x0 = col * tile_w
        y0 = row_index * tile_h
        image_relpath = str(row.get("image_relpath") or "")
        image_path = package_dir / image_relpath
        status, color, cat_bbox = _frame_cat_status(row)
        if status.startswith("CAT VISIBLE"):
            summary["missing_cat_bbox"] += 1
        elif status.startswith("CAT"):
            summary["cat"] += 1
        elif status.startswith("NO CAT"):
            summary["no_cat"] += 1
        else:
            summary["unsure"] += 1

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            summary["missing_images"] += 1
            draw.rectangle((x0, y0, x0 + tile_w, y0 + image_h), fill=(35, 35, 35))
            draw.text((x0 + 8, y0 + 8), "MISSING IMAGE", fill=(255, 120, 120), font=title_font)
            continue

        scale = min(tile_w / image.width, image_h / image.height)
        resized = image.resize((int(round(image.width * scale)), int(round(image.height * scale))))
        image_x = x0 + (tile_w - resized.width) // 2
        image_y = y0 + (image_h - resized.height) // 2
        sheet.paste(resized, (image_x, image_y))

        draw.rectangle((x0, y0, x0 + tile_w, y0 + 28), fill=(0, 0, 0))
        draw.text((x0 + 6, y0 + 4), status, fill=color, font=title_font)
        if cat_bbox is not None:
            _draw_scaled_bbox(draw, cat_bbox, scale=scale, x_offset=image_x, y_offset=image_y, color=color, width=4)
        if args.draw_candidate_context:
            candidate_bbox = _parse_bbox_xywh(str(row.get("candidate_bbox_xywh") or ""))
            candidate_label = str(row.get("label_candidate_is_cat") or "").strip().lower()
            if candidate_bbox is not None and candidate_label != "yes":
                _draw_scaled_bbox(
                    draw,
                    candidate_bbox,
                    scale=scale,
                    x_offset=image_x,
                    y_offset=image_y,
                    color=(230, 190, 65),
                    width=2,
                )

        caption_y = y0 + image_h + 4
        reason = str(row.get("candidate_reason") or row.get("negative_reason") or row.get("video_slug") or "")
        detail = "human bbox" if cat_bbox is not None else str(row.get("negative_reason") or row.get("review_status") or "")
        if status.startswith("CAT VISIBLE"):
            detail = "frame says cat, but bbox_xywh is empty"
        caption_width = tile_w - 12
        draw.text((x0 + 6, caption_y), _ellipsize_text(draw, reason, font=small_font, max_width=caption_width), fill=(210, 210, 210), font=small_font)
        draw.text((x0 + 6, caption_y + 20), _ellipsize_text(draw, detail, font=small_font, max_width=caption_width), fill=(180, 180, 180), font=small_font)
        draw.text((x0 + 6, caption_y + 40), _ellipsize_text(draw, Path(image_relpath).name, font=small_font, max_width=caption_width), fill=(150, 150, 150), font=small_font)

    sheet.save(output_path, quality=92)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.fail_on_missing_cat_bbox and summary["missing_cat_bbox"]:
        raise RuntimeError(f"{summary['missing_cat_bbox']} cat-present rows have no bbox_xywh; wrote {output_path}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine_parser = subparsers.add_parser("mine-hard-negatives")
    mine_parser.add_argument("--candidate-scores", type=Path, required=True)
    mine_parser.add_argument("--recording-dir", type=Path, required=True)
    mine_parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    mine_parser.add_argument("--package-id", default="cat-projector-hard-negatives")
    mine_parser.add_argument("--limit", type=int, default=60)
    mine_parser.add_argument("--max-auto-candidates-per-frame", type=int, default=4)
    mine_parser.add_argument("--negative-reason", default="detector_false_positive_not_cat")
    mine_parser.add_argument("--replace-existing", action="store_true")
    mine_parser.set_defaults(func=mine_hard_negatives)

    post_positive_parser = subparsers.add_parser("mine-post-positive-hard-negatives")
    post_positive_parser.add_argument("--labels", type=Path, required=True)
    post_positive_parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    post_positive_parser.add_argument("--package-id", default="cat-projector-post-positive-hard-negatives")
    post_positive_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    post_positive_parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    post_positive_parser.add_argument("--min-probability", type=float, default=DEFAULT_POST_POSITIVE_MIN_PROBABILITY)
    post_positive_parser.add_argument("--max-cat-iou", type=float, default=0.08)
    post_positive_parser.add_argument("--max-cat-candidate-overlap", type=float, default=0.10)
    post_positive_parser.add_argument("--erase-margin-ratio", type=float, default=0.10)
    post_positive_parser.add_argument("--max-candidates-per-frame", type=int, default=3)
    post_positive_parser.add_argument("--max-rows", type=int, default=240)
    post_positive_parser.add_argument("--negative-reason", default="post_positive_erased_cat_false_positive")
    post_positive_parser.add_argument("--keep-erased-debug", action="store_true")
    post_positive_parser.add_argument("--replace-existing", action="store_true")
    post_positive_parser.set_defaults(func=mine_post_positive_hard_negatives)

    live_jump_parser = subparsers.add_parser("bootstrap-live-jump-events")
    live_jump_parser.add_argument("--events", type=Path, required=True)
    live_jump_parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    live_jump_parser.add_argument("--package-id", default="cat-projector-live-jump-events")
    live_jump_parser.add_argument("--limit", type=int, default=80)
    live_jump_parser.add_argument("--replace-existing", action="store_true")
    live_jump_parser.set_defaults(func=bootstrap_live_jump_events)

    bootstrap_parser = subparsers.add_parser("bootstrap-candidates")
    bootstrap_parser.add_argument("--labels", type=Path, required=True)
    bootstrap_parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    bootstrap_parser.add_argument("--package-id", default="cat-projector-candidate-bootstrap")
    bootstrap_parser.add_argument("--max-negative-candidates-per-frame", type=int, default=2)
    bootstrap_parser.add_argument("--max-rows", type=int, default=200)
    bootstrap_parser.add_argument("--replace-existing", action="store_true")
    bootstrap_parser.set_defaults(func=bootstrap_candidates)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--labels", type=Path, action="append", required=True)
    train_parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    train_parser.add_argument("--min-positive", type=int, default=5)
    train_parser.add_argument("--min-negative", type=int, default=5)
    train_parser.add_argument("--iterations", type=int, default=80)
    train_parser.add_argument("--depth", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=0.08)
    train_parser.add_argument("--seed", type=int, default=20260516)
    train_parser.set_defaults(func=train)

    report_parser = subparsers.add_parser("render-frame-cat-report")
    report_parser.add_argument("--labels", type=Path, required=True)
    report_parser.add_argument("--out", type=Path, required=True)
    report_parser.add_argument("--limit", type=int, default=0)
    report_parser.add_argument("--columns", type=int, default=4)
    report_parser.add_argument("--tile-width", type=int, default=320)
    report_parser.add_argument("--draw-candidate-context", action="store_true")
    report_parser.add_argument("--fail-on-missing-cat-bbox", action="store_true")
    report_parser.set_defaults(func=render_frame_cat_report)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
