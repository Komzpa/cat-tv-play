#!/usr/bin/env python3
"""Watch the projector camera for live calibrated cat-jump height events."""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_config as site_config  # noqa: E402
from scripts.cat_projector_frame_detector import (  # noqa: E402
    DEFAULT_METADATA_PATH as DEFAULT_DETECTOR_METADATA_PATH,
)
from scripts.cat_projector_frame_detector import DEFAULT_MODEL_PATH as DEFAULT_DETECTOR_MODEL_PATH  # noqa: E402
from scripts.cat_projector_frame_detector import (  # noqa: E402
    PROJECTOR_POLYGON_1280,
    Candidate,
    _bottom_edge_window_candidates,
    default_model_available,
    load_model,
    score_candidates,
)
from scripts.measure_cat_projector_jump_height import Homography, _read_calibration  # noqa: E402

DEFAULT_CALIBRATION = site_config.calibration_path()
DEFAULT_SNAPSHOT_URL = site_config.camera_snapshot_url()
DEFAULT_STREAM_URL = site_config.camera_stream_url()
DEFAULT_OUTPUT_DIR = site_config.learning_root() / "live-jump-watch"


@dataclass(frozen=True)
class Component:
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    top_x_px: float
    top_y_px: float


@dataclass(frozen=True)
class LiveJumpSample:
    timestamp: str
    source: str
    candidate_found: bool
    top_x_px: float | None = None
    top_y_px: float | None = None
    wall_x_mm: float | None = None
    top_height_mm: float | None = None
    top_height_cm: float | None = None
    component_area_px: int | None = None
    component_bbox_xyxy: tuple[int, int, int, int] | None = None
    triggered: bool = False
    image_path: str | None = None
    detector_cat_probability: float | None = None
    detector_model_path: str | None = None
    source_time_s: float | None = None
    coordinate_transform: dict[str, Any] | None = None


@dataclass(frozen=True)
class JumpTrackState:
    trail_points: tuple[tuple[float, float, float], ...] = ()
    highest_sample: LiveJumpSample | None = None
    missed_samples: int = 0


@dataclass(frozen=True)
class JumpPeakHold:
    height_cm: float
    image: Image.Image
    bbox_xyxy: tuple[int, int, int, int] | None = None
    top_xy: tuple[float, float] | None = None
    source: str | None = None
    source_time_s: float | None = None
    seen_count: int = 1


@dataclass(frozen=True)
class VerticalCropState:
    center_x_px: float | None = None
    center_y_px: float | None = None


@dataclass(frozen=True)
class TrackPeak:
    index: int
    x_px: float
    y_px: float
    height_cm: float


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).astimezone().isoformat(timespec="milliseconds")


def _slug_timestamp() -> str:
    return dt.datetime.now(dt.UTC).astimezone().strftime("%Y%m%dT%H%M%S%z")


def _parse_roi(raw: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--roi must be left,top,right,bottom")
    left, top, right, bottom = parts
    if right <= left or bottom <= top:
        raise ValueError("--roi right/bottom must be greater than left/top")
    return left, top, right, bottom


def _parse_optional_path(raw: str) -> Path | None:
    raw = raw.strip()
    return Path(raw).expanduser() if raw else None


def _default_font(size: int) -> Any:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _load_image(source: str, *, timeout_seconds: float) -> Image.Image:
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=timeout_seconds)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return Image.open(Path(source).expanduser()).convert("RGB")


def iter_mjpeg_frames(url: str, *, timeout_seconds: float) -> Any:
    """Yield PIL images from an MJPEG stream without reconnecting per frame."""
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        buffer = bytearray()
        for chunk in response.iter_content(chunk_size=16384):
            if not chunk:
                continue
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)
                if start < 0 or end < 0:
                    if start > 0:
                        del buffer[:start]
                    break
                jpeg = bytes(buffer[start : end + 2])
                del buffer[: end + 2]
                yield Image.open(io.BytesIO(jpeg)).convert("RGB")


def detect_dark_components(
    image: Image.Image,
    *,
    roi: tuple[int, int, int, int],
    threshold: int,
    scale: int,
    min_area_px: int,
    min_height_px: int,
    max_width_px: int,
) -> list[Component]:
    left, top, right, bottom = roi
    left = max(0, left)
    top = max(0, top)
    right = min(image.width, right)
    bottom = min(image.height, bottom)
    crop = image.crop((left, top, right, bottom)).convert("L")
    small_width = max(1, crop.width // scale)
    small_height = max(1, crop.height // scale)
    small = crop.resize((small_width, small_height), Image.Resampling.BOX)
    mask = np.asarray(small, dtype=np.uint8) < threshold
    components: list[Component] = []
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if len(xs) == 0:
            continue
        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())
        area_px = int(len(xs) * scale * scale)
        width = (max_x - min_x + 1) * scale
        height = (max_y - min_y + 1) * scale
        if area_px < min_area_px or height < min_height_px or width > max_width_px:
            continue
        top_point = robust_component_top_px(labels, label)
        if top_point is None:
            continue
        top_x_small, top_y_small = top_point
        top_x = left + (top_x_small + 0.5) * scale
        top_y = top + (top_y_small + 0.5) * scale
        components.append(
            Component(
                area_px=area_px,
                bbox_xyxy=(
                    left + min_x * scale,
                    top + min_y * scale,
                    left + (max_x + 1) * scale,
                    top + (max_y + 1) * scale,
                ),
                top_x_px=top_x,
                top_y_px=top_y,
            )
        )

    return sorted(components, key=lambda item: item.area_px, reverse=True)


def robust_component_top_px(
    labels: np.ndarray,
    label: int,
    *,
    top_percentile: float = 3.0,
    cap_height_px: int = 2,
    min_cap_pixels: int = 3,
) -> tuple[float, float] | None:
    """Return a stable top point instead of trusting a single noisy pixel."""
    ys, xs = np.nonzero(labels == label)
    if len(ys) < min_cap_pixels:
        return None
    robust_top_y = float(np.percentile(ys, top_percentile))
    cap = ys <= robust_top_y + cap_height_px
    if int(cap.sum()) < min_cap_pixels:
        return None
    return float(np.median(xs[cap])), float(np.median(ys[cap]))


def sample_image(
    image: Image.Image,
    *,
    source: str,
    homography: Any,
    floor_y_mm: float,
    roi: tuple[int, int, int, int],
    threshold: int,
    scale: int,
    min_area_px: int,
    min_height_px: int,
    max_width_px: int,
    jump_threshold_cm: float,
    detector_model: Path | None = None,
    detector_metadata: Path | None = None,
    detector_loaded_model: Any | None = None,
    detector_loaded_metadata: dict[str, Any] | None = None,
    detector_min_probability: float = 0.60,
) -> tuple[LiveJumpSample, Component | None]:
    if detector_model is not None:
        components = detect_dark_components(
            image,
            roi=roi,
            threshold=threshold,
            scale=scale,
            min_area_px=min_area_px,
            min_height_px=min_height_px,
            max_width_px=max_width_px,
        )
        component_candidates = [
            Candidate(
                bbox_xywh=(
                    float(component.bbox_xyxy[0]),
                    float(component.bbox_xyxy[1]),
                    float(component.bbox_xyxy[2] - component.bbox_xyxy[0]),
                    float(component.bbox_xyxy[3] - component.bbox_xyxy[1]),
                ),
                top_x_px=component.top_x_px,
                top_y_px=component.top_y_px,
                area_px=component.area_px,
                source="live_jump_roi_component",
            )
            for component in components
        ]
        model = detector_loaded_model
        metadata = detector_loaded_metadata
        if model is None or metadata is None:
            model, metadata = load_model(detector_model, detector_metadata or DEFAULT_DETECTOR_METADATA_PATH)
        predictions = sorted(
            score_candidates(image, model=model, metadata=metadata, candidates=component_candidates),
            key=lambda item: item.cat_probability,
            reverse=True,
        )
        if not predictions or predictions[0].cat_probability < detector_min_probability:
            bottom_predictions = sorted(
                score_candidates(
                    image,
                    model=model,
                    metadata=metadata,
                    candidates=_bottom_edge_window_candidates(image.width, image.height),
                ),
                key=lambda item: item.cat_probability,
                reverse=True,
            )
            if bottom_predictions and (
                not predictions or bottom_predictions[0].cat_probability > predictions[0].cat_probability
            ):
                predictions = bottom_predictions
        if not predictions:
            return (
                LiveJumpSample(
                    timestamp=_now_iso(),
                    source=source,
                    candidate_found=False,
                    detector_cat_probability=0.0,
                    detector_model_path=str(detector_model),
                ),
                None,
            )
        prediction = predictions[0]
        if prediction.cat_probability < detector_min_probability:
            return (
                LiveJumpSample(
                    timestamp=_now_iso(),
                    source=source,
                    candidate_found=False,
                    detector_cat_probability=round(prediction.cat_probability, 4),
                    detector_model_path=prediction.model_path,
                ),
                None,
            )
        candidate = prediction.candidate
        if candidate is None:  # pragma: no cover - score_candidates returns candidates for non-empty predictions.
            raise RuntimeError("cat detector returned a prediction without a candidate")
        component = Component(
            area_px=candidate.area_px,
            bbox_xyxy=(
                int(round(candidate.bbox_xywh[0])),
                int(round(candidate.bbox_xywh[1])),
                int(round(candidate.bbox_xywh[0] + candidate.bbox_xywh[2])),
                int(round(candidate.bbox_xywh[1] + candidate.bbox_xywh[3])),
            ),
            top_x_px=candidate.top_x_px,
            top_y_px=candidate.top_y_px,
        )
        wall_x_mm, wall_y_mm = homography.project(component.top_x_px, component.top_y_px)
        height_mm = max(0.0, wall_y_mm - floor_y_mm)
        height_cm = height_mm / 10.0
        return (
            LiveJumpSample(
                timestamp=_now_iso(),
                source=source,
                candidate_found=True,
                top_x_px=round(component.top_x_px, 2),
                top_y_px=round(component.top_y_px, 2),
                wall_x_mm=round(wall_x_mm, 1),
                top_height_mm=round(height_mm, 1),
                top_height_cm=round(height_cm, 1),
                component_area_px=component.area_px,
                component_bbox_xyxy=component.bbox_xyxy,
                triggered=height_cm >= jump_threshold_cm,
                detector_cat_probability=round(prediction.cat_probability, 4),
                detector_model_path=prediction.model_path,
            ),
            component,
        )

    components = detect_dark_components(
        image,
        roi=roi,
        threshold=threshold,
        scale=scale,
        min_area_px=min_area_px,
        min_height_px=min_height_px,
        max_width_px=max_width_px,
    )
    if not components:
        return LiveJumpSample(timestamp=_now_iso(), source=source, candidate_found=False), None

    component = components[0]
    wall_x_mm, wall_y_mm = homography.project(component.top_x_px, component.top_y_px)
    height_mm = max(0.0, wall_y_mm - floor_y_mm)
    height_cm = height_mm / 10.0
    return (
        LiveJumpSample(
            timestamp=_now_iso(),
            source=source,
            candidate_found=True,
            top_x_px=round(component.top_x_px, 2),
            top_y_px=round(component.top_y_px, 2),
            wall_x_mm=round(wall_x_mm, 1),
            top_height_mm=round(height_mm, 1),
            top_height_cm=round(height_cm, 1),
            component_area_px=component.area_px,
            component_bbox_xyxy=component.bbox_xyxy,
            triggered=height_cm >= jump_threshold_cm,
        ),
        component,
    )


def _save_jpeg_atomic(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    image.save(temporary_path, format="JPEG", quality=92)
    temporary_path.replace(output_path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def update_track_state(
    state: JumpTrackState,
    sample: LiveJumpSample,
    *,
    max_points: int,
    max_missed_samples: int = 15,
) -> JumpTrackState:
    if not sample.candidate_found or sample.top_x_px is None or sample.top_y_px is None or sample.top_height_cm is None:
        missed_samples = state.missed_samples + 1
        if missed_samples <= max_missed_samples:
            return JumpTrackState(
                trail_points=state.trail_points,
                highest_sample=state.highest_sample,
                missed_samples=missed_samples,
            )
        return JumpTrackState(highest_sample=state.highest_sample, missed_samples=missed_samples)
    trail = (*state.trail_points, (sample.top_x_px, sample.top_y_px, sample.top_height_cm))[-max_points:]
    highest = state.highest_sample
    if highest is None or (sample.top_height_cm or 0.0) > (highest.top_height_cm or 0.0):
        highest = sample
    return JumpTrackState(trail_points=trail, highest_sample=highest)


def _alpha_for_age(index: int, count: int, *, minimum: int = 35, maximum: int = 220) -> int:
    if count <= 1:
        return maximum
    age_ratio = index / float(count - 1)
    return int(round(minimum + (maximum - minimum) * age_ratio))


def _projection_polygon_for_size(size: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    scale_x = size[0] / 1280.0
    scale_y = size[1] / 720.0
    return tuple((x * scale_x, y * scale_y) for x, y in PROJECTOR_POLYGON_1280)


def _projection_polygon_for_image(image: Image.Image) -> tuple[tuple[float, float], ...]:
    return _projection_polygon_for_size((float(image.width), float(image.height)))


def _affine_matrix_2x3(raw: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    rows: list[tuple[float, float, float]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        try:
            rows.append((float(row[0]), float(row[1]), float(row[2])))
        except (TypeError, ValueError):
            return None
    return rows[0], rows[1]


def _coordinate_transform_size(raw: Any, key: str) -> tuple[float, float] | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        width = float(value[0])
        height = float(value[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _calibration_point_to_frame_point(
    point: tuple[float, float],
    coordinate_transform: dict[str, Any] | None,
    *,
    fallback_frame_size: tuple[int, int],
) -> tuple[float, float]:
    if not isinstance(coordinate_transform, dict):
        return point
    calibration_size = _coordinate_transform_size(coordinate_transform, "calibration_size_px")
    frame_size = _coordinate_transform_size(coordinate_transform, "frame_size_px") or (
        float(fallback_frame_size[0]),
        float(fallback_frame_size[1]),
    )
    if calibration_size is None:
        return point
    raw_x, raw_y = point
    alignment = coordinate_transform.get("alignment")
    if isinstance(alignment, dict):
        matrix = _affine_matrix_2x3(alignment.get("calibration_to_current"))
        if matrix is not None:
            (a00, a01, a02), (a10, a11, a12) = matrix
            raw_x = a00 * point[0] + a01 * point[1] + a02
            raw_y = a10 * point[0] + a11 * point[1] + a12
    return raw_x * frame_size[0] / calibration_size[0], raw_y * frame_size[1] / calibration_size[1]


def _wall_line_inside_projection(
    *,
    image: Image.Image,
    homography: Homography,
    wall_y_mm: float,
    coordinate_transform: dict[str, Any] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    calibration_size = _coordinate_transform_size(coordinate_transform, "calibration_size_px")
    polygon_px = (
        _projection_polygon_for_size(calibration_size)
        if calibration_size is not None
        else _projection_polygon_for_image(image)
    )
    polygon_wall = [homography.project(x, y) for x, y in polygon_px]
    intersections: list[tuple[float, float]] = []
    for index, (ax, ay) in enumerate(polygon_wall):
        bx, by = polygon_wall[(index + 1) % len(polygon_wall)]
        if abs(ay - wall_y_mm) < 1e-6:
            intersections.append((ax, wall_y_mm))
        if (ay - wall_y_mm) * (by - wall_y_mm) > 0 or abs(by - ay) < 1e-6:
            continue
        t = (wall_y_mm - ay) / (by - ay)
        if 0.0 <= t <= 1.0:
            intersections.append((ax + (bx - ax) * t, wall_y_mm))
    unique: list[tuple[float, float]] = []
    for point in intersections:
        if not any(abs(point[0] - seen[0]) < 1e-3 and abs(point[1] - seen[1]) < 1e-3 for seen in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    unique = sorted(unique, key=lambda point: point[0])
    left_px = _calibration_point_to_frame_point(
        homography.unproject(unique[0][0], unique[0][1]),
        coordinate_transform,
        fallback_frame_size=image.size,
    )
    right_px = _calibration_point_to_frame_point(
        homography.unproject(unique[-1][0], unique[-1][1]),
        coordinate_transform,
        fallback_frame_size=image.size,
    )
    return left_px, right_px


def _local_track_peaks(points: tuple[tuple[float, float, float], ...]) -> list[TrackPeak]:
    if len(points) < 3:
        return []
    raw_peaks: list[TrackPeak] = []
    for index in range(1, len(points) - 1):
        prev_height = points[index - 1][2]
        x_px, y_px, height = points[index]
        next_height = points[index + 1][2]
        if height >= prev_height and height > next_height and height >= 45:
            raw_peaks.append(TrackPeak(index=index, x_px=x_px, y_px=y_px, height_cm=height))
    peaks: list[TrackPeak] = []
    min_gap = 18
    for peak in raw_peaks:
        if peaks and peak.index - peaks[-1].index < min_gap:
            if peak.height_cm > peaks[-1].height_cm:
                peaks[-1] = peak
            continue
        peaks.append(peak)
    return peaks


def _draw_text_with_stroke(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    fill: tuple[int, int, int, int] | tuple[int, int, int],
    font: Any,
    stroke_width: int = 3,
) -> None:
    stroke_alpha = fill[3] if len(fill) == 4 else 255
    draw.text(
        xy,
        text,
        fill=fill,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, stroke_alpha),
    )


def _draw_track_state(
    base: Image.Image,
    state: JumpTrackState,
    *,
    font: Any,
    homography: Homography | None,
    floor_y_mm: float | None,
    display_max_height_cm: float | None = None,
) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = state.trail_points
    if len(points) >= 2:
        for index in range(1, len(points)):
            x0, y0, _height0 = points[index - 1]
            x1, y1, _height1 = points[index]
            alpha = _alpha_for_age(index, len(points), minimum=25, maximum=190)
            draw.line((x0, y0, x1, y1), fill=(56, 189, 248, alpha), width=2)
        for index, (x, y, _height) in enumerate(points):
            alpha = _alpha_for_age(index, len(points), minimum=30, maximum=210)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(56, 189, 248, alpha))

    highest = state.highest_sample
    global_max_height_cm = (
        display_max_height_cm
        if display_max_height_cm is not None
        else (highest.top_height_cm if highest is not None else None)
    )
    for peak in _local_track_peaks(points):
        if global_max_height_cm is not None and abs(peak.height_cm - global_max_height_cm) < 0.05:
            continue
        alpha = _alpha_for_age(peak.index, len(points), minimum=40, maximum=210)
        draw.line((peak.x_px - 12, peak.y_px, peak.x_px + 12, peak.y_px), fill=(255, 230, 88, alpha), width=2)
        draw.line((peak.x_px, peak.y_px - 10, peak.x_px, peak.y_px + 10), fill=(255, 230, 88, alpha), width=2)
        _draw_text_with_stroke(
            draw,
            (max(8, peak.x_px + 8), max(8, peak.y_px - 30)),
            f"{peak.height_cm:.0f} cm",
            fill=(255, 240, 160, alpha),
            font=font,
            stroke_width=2,
        )

    if global_max_height_cm is not None and homography is not None and floor_y_mm is not None:
        coordinate_transform = highest.coordinate_transform if highest is not None else None
        line = _wall_line_inside_projection(
            image=base,
            homography=homography,
            wall_y_mm=floor_y_mm + global_max_height_cm * 10.0,
            coordinate_transform=coordinate_transform,
        )
        if line is not None:
            left_px, right_px = line
            draw.line((*left_px, *right_px), fill=(255, 64, 190, 235), width=4)
            label_x = max(8.0, min(left_px[0], right_px[0]) + 10)
            label_y = max(8.0, min(left_px[1], right_px[1]) - 34)
            _draw_text_with_stroke(
                draw,
                (label_x, label_y),
                f"max {global_max_height_cm:.1f} cm",
                fill=(255, 230, 250, 245),
                font=font,
                stroke_width=3,
            )

    base.alpha_composite(overlay)


def crop_jump_peak_hold(
    image: Image.Image,
    sample: LiveJumpSample,
    *,
    pad_px: int = 96,
    min_width_px: int = 220,
    min_height_px: int = 300,
) -> Image.Image:
    if sample.component_bbox_xyxy is not None:
        left, top, right, bottom = sample.component_bbox_xyxy
    elif sample.top_x_px is not None and sample.top_y_px is not None:
        left = int(round(sample.top_x_px - min_width_px / 2))
        right = int(round(sample.top_x_px + min_width_px / 2))
        top = int(round(sample.top_y_px - min_height_px / 2))
        bottom = int(round(sample.top_y_px + min_height_px / 2))
    else:
        return image.copy()

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    width = max(min_width_px, right - left + pad_px * 2)
    height = max(min_height_px, bottom - top + pad_px * 2)
    if height < width * 1.25:
        height = min(float(image.height), width * 1.25)
    crop_left = int(round(center_x - width / 2.0))
    crop_top = int(round(center_y - height / 2.0))
    crop_right = int(round(center_x + width / 2.0))
    crop_bottom = int(round(center_y + height / 2.0))

    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_right > image.width:
        crop_left -= crop_right - image.width
        crop_right = image.width
    if crop_bottom > image.height:
        crop_top -= crop_bottom - image.height
        crop_bottom = image.height
    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)

    return image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")


def _source_time_seconds(source: str | None) -> float | None:
    if not source:
        return None
    match = re.search(r"chunk_(\d+)_(\d+)\.jpg", source)
    if not match:
        return None
    chunk_index = int(match.group(1))
    frame_index = int(match.group(2))
    milliseconds_match = re.search(r"chunk_\d+_(\d+)ms", source)
    if milliseconds_match:
        return chunk_index * 5.0 + int(milliseconds_match.group(1)) / 1000.0
    return chunk_index * 5.0 + max(0, frame_index - 1) / 15.0


def _holds_are_same_jump(
    old: JumpPeakHold,
    new: JumpPeakHold,
    *,
    time_window_s: float,
    horizontal_window_px: float,
) -> bool:
    if old.source_time_s is not None and new.source_time_s is not None:
        return abs(old.source_time_s - new.source_time_s) <= time_window_s
    if old.top_xy is None or new.top_xy is None:
        return False
    close_x = abs(old.top_xy[0] - new.top_xy[0]) <= horizontal_window_px
    close_y = abs(old.top_xy[1] - new.top_xy[1]) <= 260
    return close_x and close_y


def update_top_jump_holds(
    holds: tuple[JumpPeakHold, ...],
    image: Image.Image,
    sample: LiveJumpSample,
    *,
    max_holds: int = 3,
    min_height_cm: float = 45.0,
    provisional_min_height_cm: float = 5.0,
    same_jump_window_s: float = 3.0,
    horizontal_window_px: float = 180.0,
) -> tuple[JumpPeakHold, ...]:
    if (
        not sample.candidate_found
        or sample.top_height_cm is None
        or sample.top_height_cm < provisional_min_height_cm
        or sample.top_x_px is None
        or sample.top_y_px is None
    ):
        return holds
    if sample.top_height_cm >= max(min_height_cm, 80.0) and not _looks_like_reaching_cat_body(sample):
        return holds

    new_hold = JumpPeakHold(
        height_cm=float(sample.top_height_cm),
        image=crop_jump_peak_hold(image, sample),
        bbox_xyxy=sample.component_bbox_xyxy,
        top_xy=(float(sample.top_x_px), float(sample.top_y_px)),
        source=sample.source,
        source_time_s=sample.source_time_s if sample.source_time_s is not None else _source_time_seconds(sample.source),
    )
    if new_hold.height_cm < min_height_cm:
        if any(hold.height_cm >= min_height_cm for hold in holds):
            return holds
        if not holds:
            return (new_hold,)
        best_hold = max(holds, key=lambda item: item.height_cm)
        if new_hold.height_cm > best_hold.height_cm:
            return (new_hold,)
        return (best_hold,)

    holds = tuple(hold for hold in holds if hold.height_cm >= min_height_cm)
    updated = list(holds)
    for index, hold in enumerate(updated):
        if _holds_are_same_jump(
            hold,
            new_hold,
            time_window_s=same_jump_window_s,
            horizontal_window_px=horizontal_window_px,
        ):
            if new_hold.height_cm > hold.height_cm:
                updated[index] = JumpPeakHold(
                    height_cm=new_hold.height_cm,
                    image=new_hold.image,
                    bbox_xyxy=new_hold.bbox_xyxy,
                    top_xy=new_hold.top_xy,
                    source=new_hold.source,
                    source_time_s=new_hold.source_time_s,
                    seen_count=hold.seen_count + 1,
                )
            else:
                updated[index] = JumpPeakHold(
                    height_cm=hold.height_cm,
                    image=hold.image,
                    bbox_xyxy=hold.bbox_xyxy,
                    top_xy=hold.top_xy,
                    source=hold.source,
                    source_time_s=hold.source_time_s,
                    seen_count=hold.seen_count + 1,
                )
            return tuple(updated[:max_holds])

    updated.append(new_hold)
    updated.sort(key=lambda item: item.height_cm, reverse=True)
    return tuple(updated[:max_holds])


def _looks_like_reaching_cat_body(sample: LiveJumpSample) -> bool:
    if sample.component_bbox_xyxy is None:
        return True
    left, top, right, bottom = sample.component_bbox_xyxy
    width = max(1, right - left)
    height = max(1, bottom - top)
    return height >= width * 1.10


def confirmed_top_jump_holds(
    holds: tuple[JumpPeakHold, ...],
    *,
    min_seen_count: int = 2,
) -> tuple[JumpPeakHold, ...]:
    confirmed = tuple(hold for hold in holds if hold.seen_count >= min_seen_count)
    if any(hold.height_cm >= 100.0 for hold in confirmed):
        confirmed = tuple(hold for hold in confirmed if hold.height_cm >= 100.0)
    if any(hold.height_cm >= 80.0 for hold in confirmed):
        confirmed = tuple(hold for hold in confirmed if hold.height_cm >= 80.0)
    return tuple(sorted(confirmed, key=lambda item: item.height_cm, reverse=True))


def _draw_top_jump_holds(
    base: Image.Image,
    holds: tuple[JumpPeakHold, ...],
    *,
    font: Any,
) -> None:
    holds = confirmed_top_jump_holds(holds)
    if not holds:
        return
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    panel_width = max(210, min(300, base.width // 4))
    margin = max(12, base.width // 90)
    gap = max(8, base.height // 90)
    thumb_width = panel_width - margin * 2
    label_height = max(26, min(38, base.height // 18))
    thumb_height = max(82, min(124, (base.height - margin * 2 - gap * 2 - label_height * 3) // 3))
    row_height = label_height + thumb_height
    x0 = base.width - panel_width - margin
    y0 = margin
    draw.rounded_rectangle(
        (x0, y0, base.width - margin, y0 + margin + (row_height + gap) * len(holds)),
        radius=6,
        fill=(0, 0, 0, 115),
        outline=(255, 255, 255, 75),
        width=1,
    )
    for index, hold in enumerate(holds[:3], start=1):
        top = y0 + margin + (index - 1) * (row_height + gap)
        label_box = (x0 + margin, top, x0 + margin + thumb_width, top + label_height)
        draw.rectangle(label_box, fill=(0, 0, 0, 205))
        _draw_text_with_stroke(
            draw,
            (label_box[0] + 8, label_box[1] + 3),
            f"#{index} {hold.height_cm:.1f} cm",
            fill=(255, 245, 255, 245),
            font=font,
            stroke_width=3,
        )
        thumb = hold.image.copy()
        thumb.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        thumb_canvas = thumb.convert("RGBA")
        thumb_top = top + label_height
        paste_x = x0 + margin + (thumb_width - thumb.width) // 2
        paste_y = thumb_top + (thumb_height - thumb.height) // 2
        overlay.alpha_composite(thumb_canvas, (paste_x, paste_y))
        box = (paste_x, paste_y, paste_x + thumb.width, paste_y + thumb.height)
        draw.rectangle(box, outline=(255, 64, 190, 230), width=3)
    base.alpha_composite(overlay)


def _vertical_xy(
    x: float,
    y: float,
    *,
    crop_left: int,
    crop_top: int,
    scale: float,
) -> tuple[float, float]:
    return ((x - crop_left) * scale, (y - crop_top) * scale)


def _vertical_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    crop_left: int,
    crop_top: int,
    scale: float,
    output_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox_xyxy
    out_width, out_height = output_size
    return (
        max(0, min(out_width, int(round((left - crop_left) * scale)))),
        max(0, min(out_height, int(round((top - crop_top) * scale)))),
        max(0, min(out_width, int(round((right - crop_left) * scale)))),
        max(0, min(out_height, int(round((bottom - crop_top) * scale)))),
    )


def _clamped_text_xy(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: Any,
    image_size: tuple[int, int],
    margin: int,
) -> tuple[float, float]:
    x, y = xy
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=4)
        width = right - left
        height = bottom - top
    except Exception:
        width = len(text) * max(10, getattr(font, "size", 24) // 2)
        height = max(24, getattr(font, "size", 24))
    return (
        max(margin, min(float(image_size[0] - margin - width), x)),
        max(margin, min(float(image_size[1] - margin - height), y)),
    )


def _show_current_vertical_marker(
    sample: LiveJumpSample,
    *,
    confirmed_holds: tuple[JumpPeakHold, ...],
) -> bool:
    if sample.component_bbox_xyxy is None or sample.top_x_px is None or sample.top_y_px is None:
        return False
    if not confirmed_holds or confirmed_holds[0].height_cm < 80.0:
        return True
    if sample.top_height_cm is None:
        return False
    return sample.top_height_cm >= 75.0


def _line_segments_outside_rect(
    coords: tuple[float, float, float, float],
    rect: tuple[float, float, float, float] | None,
) -> list[tuple[float, float, float, float]]:
    if rect is None:
        return [coords]
    x0, y0, x1, y1 = coords
    left, top, right, bottom = rect
    if right <= left or bottom <= top:
        return [coords]
    dx = x1 - x0
    dy = y1 - y0
    t0 = 0.0
    t1 = 1.0
    for p, q in (
        (-dx, x0 - left),
        (dx, right - x0),
        (-dy, y0 - top),
        (dy, bottom - y0),
    ):
        if abs(p) < 1e-9:
            if q < 0:
                return [coords]
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return [coords]
    segments: list[tuple[float, float, float, float]] = []
    gap = 0.015
    if t0 > 0.0:
        ta = 0.0
        tb = max(0.0, t0 - gap)
        if tb > ta:
            segments.append((x0 + dx * ta, y0 + dy * ta, x0 + dx * tb, y0 + dy * tb))
    if t1 < 1.0:
        ta = min(1.0, t1 + gap)
        tb = 1.0
        if tb > ta:
            segments.append((x0 + dx * ta, y0 + dy * ta, x0 + dx * tb, y0 + dy * tb))
    return segments or []


def _draw_vertical_track_state(
    base: Image.Image,
    state: JumpTrackState,
    *,
    font: Any,
    homography: Homography | None,
    floor_y_mm: float | None,
    display_max_height_cm: float | None,
    source_image: Image.Image,
    crop_left: int,
    crop_top: int,
    scale: float,
    show_local_peaks: bool = True,
    show_trail: bool = True,
    line_coordinate_transform: dict[str, Any] | None = None,
    max_line_label: bool = True,
    max_line_width: int = 4,
    max_line_fill: tuple[int, int, int, int] = (255, 64, 190, 235),
    max_line_anchor_xy: tuple[float, float] | None = None,
) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = state.trail_points
    if show_trail and len(points) >= 2:
        for index in range(1, len(points)):
            x0, y0 = _vertical_xy(
                points[index - 1][0], points[index - 1][1], crop_left=crop_left, crop_top=crop_top, scale=scale
            )
            x1, y1 = _vertical_xy(
                points[index][0], points[index][1], crop_left=crop_left, crop_top=crop_top, scale=scale
            )
            alpha = _alpha_for_age(index, len(points), minimum=25, maximum=190)
            draw.line((x0, y0, x1, y1), fill=(56, 189, 248, alpha), width=2)
        for index, (x, y, _height) in enumerate(points):
            px, py = _vertical_xy(x, y, crop_left=crop_left, crop_top=crop_top, scale=scale)
            alpha = _alpha_for_age(index, len(points), minimum=30, maximum=210)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(56, 189, 248, alpha))

    for peak in _local_track_peaks(points) if show_local_peaks else ():
        if display_max_height_cm is not None and abs(peak.height_cm - display_max_height_cm) < 0.05:
            continue
        x, y = _vertical_xy(peak.x_px, peak.y_px, crop_left=crop_left, crop_top=crop_top, scale=scale)
        if x < 0 or x > base.width or y < 0 or y > base.height:
            continue
        alpha = _alpha_for_age(peak.index, len(points), minimum=40, maximum=210)
        draw.line((x - 14, y, x + 14, y), fill=(255, 230, 88, alpha), width=2)
        draw.line((x, y - 12, x, y + 12), fill=(255, 230, 88, alpha), width=2)
        label = f"{peak.height_cm:.0f} cm"
        label_xy = _clamped_text_xy(
            draw,
            (x + 10, y - 34),
            label,
            font=font,
            image_size=base.size,
            margin=8,
        )
        _draw_text_with_stroke(draw, label_xy, label, fill=(255, 240, 160, alpha), font=font, stroke_width=2)

    if display_max_height_cm is not None and homography is not None and floor_y_mm is not None:
        coordinate_transform = line_coordinate_transform
        if coordinate_transform is None and state.highest_sample is not None:
            coordinate_transform = state.highest_sample.coordinate_transform
        line = _wall_line_inside_projection(
            image=source_image,
            homography=homography,
            wall_y_mm=floor_y_mm + display_max_height_cm * 10.0,
            coordinate_transform=coordinate_transform,
        )
        if line is not None:
            left_px, right_px = line
            x0, y0 = _vertical_xy(left_px[0], left_px[1], crop_left=crop_left, crop_top=crop_top, scale=scale)
            x1, y1 = _vertical_xy(right_px[0], right_px[1], crop_left=crop_left, crop_top=crop_top, scale=scale)
            if max_line_anchor_xy is not None and abs(x1 - x0) > 1e-6:
                anchor_x, anchor_y = _vertical_xy(
                    max_line_anchor_xy[0],
                    max_line_anchor_xy[1],
                    crop_left=crop_left,
                    crop_top=crop_top,
                    scale=scale,
                )
                t = (anchor_x - x0) / (x1 - x0)
                projected_y = y0 + (y1 - y0) * t
                if projected_y > anchor_y:
                    delta_y = projected_y - anchor_y
                    y0 -= delta_y
                    y1 -= delta_y
            draw.line((x0, y0, x1, y1), fill=max_line_fill, width=max(1, max_line_width))
            if max_line_label:
                label = f"max {display_max_height_cm:.1f} cm"
                label_xy = _clamped_text_xy(
                    draw,
                    (min(x0, x1) + 12, min(y0, y1) - 42),
                    label,
                    font=font,
                    image_size=base.size,
                    margin=8,
                )
                _draw_text_with_stroke(draw, label_xy, label, fill=(255, 230, 250, 245), font=font, stroke_width=3)

    base.alpha_composite(overlay)


def _draw_vertical_sample_marker(
    base: Image.Image,
    sample: LiveJumpSample,
    *,
    font: Any,
    homography: Homography | None,
    floor_y_mm: float | None,
    crop_left: int,
    crop_top: int,
    scale: float,
) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    status_fill = (74, 222, 128, 255) if sample.triggered else (255, 210, 64, 255)
    if sample.component_bbox_xyxy is not None:
        bbox = _vertical_bbox(
            sample.component_bbox_xyxy, crop_left=crop_left, crop_top=crop_top, scale=scale, output_size=base.size
        )
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            draw.rectangle(bbox, outline=status_fill, width=5)
    if sample.top_x_px is None or sample.top_y_px is None:
        base.alpha_composite(overlay)
        return
    x, y = _vertical_xy(sample.top_x_px, sample.top_y_px, crop_left=crop_left, crop_top=crop_top, scale=scale)
    if homography is not None and floor_y_mm is not None and sample.wall_x_mm is not None:
        try:
            floor_x, floor_y = homography.unproject(sample.wall_x_mm, floor_y_mm)
            fx, fy = _vertical_xy(floor_x, floor_y, crop_left=crop_left, crop_top=crop_top, scale=scale)
            draw.line((x, y, fx, fy), fill=(255, 64, 64, 255), width=5)
            draw.line((fx - 24, fy, fx + 24, fy), fill=(255, 64, 64, 255), width=4)
        except ValueError:
            pass
    draw.line((x - 20, y, x + 20, y), fill=(255, 64, 64, 255), width=4)
    draw.line((x, y - 20, x, y + 20), fill=(255, 64, 64, 255), width=4)
    height_label = f"{sample.top_height_cm:.1f} cm" if sample.top_height_cm is not None else "visible top"
    label = height_label
    label_xy = _clamped_text_xy(
        draw,
        (x + 14, y - 44),
        label,
        font=font,
        image_size=base.size,
        margin=8,
    )
    _draw_text_with_stroke(draw, label_xy, label, fill=(255, 255, 255, 255), font=font, stroke_width=4)
    base.alpha_composite(overlay)


def _draw_vertical_hud(base: Image.Image) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    margin = max(26, base.width // 34)
    corner = max(54, base.width // 10)
    color = (56, 189, 248, 95)
    hot = (255, 64, 190, 120)
    for x0, y0, sx, sy in (
        (margin, margin, 1, 1),
        (base.width - margin, margin, -1, 1),
        (margin, base.height - margin, 1, -1),
        (base.width - margin, base.height - margin, -1, -1),
    ):
        draw.line((x0, y0, x0 + sx * corner, y0), fill=color, width=2)
        draw.line((x0, y0, x0, y0 + sy * corner), fill=color, width=2)
    for y in range(margin * 3, base.height - margin * 3, max(160, base.height // 7)):
        draw.line((margin, y, base.width - margin, y), fill=(56, 189, 248, 22), width=1)
    draw.line((margin, base.height // 2, base.width - margin, base.height // 2), fill=hot, width=1)
    base.alpha_composite(overlay)


def _draw_top_jump_holds_top_bar(
    base: Image.Image,
    holds: tuple[JumpPeakHold, ...],
    *,
    font: Any,
    avoid_bbox_xyxy: tuple[int, int, int, int] | None = None,
    publish_safe_zone: bool = False,
    session_title: str | None = None,
    session_subtitle: str | None = None,
) -> None:
    holds = tuple(hold for hold in confirmed_top_jump_holds(holds) if hold.height_cm >= 80.0)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    margin = max(54, base.width // 20) if publish_safe_zone else max(18, base.width // 45)
    gap = max(14, base.width // 90 if publish_safe_zone else base.width // 80)
    panel_height = max(270, min(340, base.height // 6)) if publish_safe_zone else max(270, min(360, base.height // 6))
    y0 = margin
    if avoid_bbox_xyxy is not None:
        _left, top, _right, bottom = avoid_bbox_xyxy
        if top < y0 + panel_height + margin and bottom > y0:
            mid_y0 = margin + panel_height + gap
            if bottom <= mid_y0 or top >= mid_y0 + panel_height:
                y0 = mid_y0
    draw.rounded_rectangle(
        (margin, y0, base.width - margin, y0 + panel_height),
        radius=10,
        fill=(0, 0, 0, 135),
        outline=(255, 255, 255, 80),
        width=2,
    )
    slot_width = int((base.width - margin * 2 - gap * 4) / 3)
    title_font = _default_font(max(30, base.width // 32) if publish_safe_zone else max(32, base.width // 28))
    row_font = _default_font(max(24, base.width // 42) if publish_safe_zone else max(28, base.width // 34))
    title = session_title or ("RECORDS" if publish_safe_zone else "RECORD BOARD")
    _draw_text_with_stroke(
        draw,
        (margin + gap, y0 + 10),
        title,
        fill=(255, 245, 255, 245),
        font=title_font,
        stroke_width=4,
    )
    subtitle = session_subtitle or ("top jump has not happened yet" if not holds else "live top jumps")
    _draw_text_with_stroke(
        draw,
        (margin + gap, y0 + (54 if publish_safe_zone else max(48, base.width // 16))),
        subtitle,
        fill=(190, 230, 255, 220),
        font=row_font,
        stroke_width=3,
    )
    state_text = "waiting for first jump" if not holds else "live top jumps"
    state_top = y0 + (88 if publish_safe_zone else max(82, base.width // 12))
    if session_subtitle is not None:
        _draw_text_with_stroke(
            draw,
            (margin + gap, state_top),
            state_text,
            fill=(255, 255, 255, 190),
            font=row_font,
            stroke_width=3,
        )
    thumb_height = panel_height - (
        140
        if publish_safe_zone and session_subtitle is not None
        else 112
        if publish_safe_zone
        else max(140, base.width // 7)
    )
    row_top = y0 + (
        122
        if publish_safe_zone and session_subtitle is not None
        else 96
        if publish_safe_zone
        else max(108, base.width // 8)
    )
    for index in range(1, 4):
        hold = holds[index - 1] if index - 1 < len(holds) else None
        x0 = margin + gap + (index - 1) * (slot_width + gap)
        label = f"#{index} {hold.height_cm:.1f} cm" if hold is not None else f"#{index} waiting"
        _draw_text_with_stroke(
            draw,
            (x0 + 8, row_top),
            label,
            fill=(255, 245, 255, 245) if hold is not None else (255, 255, 255, 180),
            font=font if hold is not None else row_font,
            stroke_width=4,
        )
        thumb_box = (
            x0 + 8,
            row_top + (46 if publish_safe_zone else 58),
            x0 + slot_width - 8,
            row_top + (46 if publish_safe_zone else 58) + thumb_height,
        )
        if hold is None:
            draw.rounded_rectangle(
                thumb_box,
                radius=8,
                fill=(255, 255, 255, 22),
                outline=(255, 255, 255, 58),
                width=2,
            )
            _draw_text_with_stroke(
                draw,
                (thumb_box[0] + 18, thumb_box[1] + max(18, thumb_height // 3)),
                "not yet",
                fill=(255, 255, 255, 150),
                font=row_font,
                stroke_width=3,
            )
            continue
        thumb = hold.image.copy()
        thumb.thumbnail((slot_width - 16, thumb_height), Image.Resampling.LANCZOS)
        paste_x = x0 + (slot_width - thumb.width) // 2
        paste_y = row_top + (46 if publish_safe_zone else 58) + (thumb_height - thumb.height) // 2
        overlay.alpha_composite(thumb.convert("RGBA"), (paste_x, paste_y))
        draw.rectangle(
            (paste_x, paste_y, paste_x + thumb.width, paste_y + thumb.height),
            outline=(255, 64, 190, 235),
            width=4,
        )
    base.alpha_composite(overlay)


def _fit_image_on_canvas(image: Image.Image, size: tuple[int, int], fill: tuple[int, int, int, int]) -> Image.Image:
    canvas = Image.new("RGBA", size, fill)
    fitted = image.copy().convert("RGB")
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.alpha_composite(fitted.convert("RGBA"), ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def _cover_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    scale = max(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - target_width) // 2)
    top = max(0, (resized.height - target_height) // 2)
    return resized.crop((left, top, left + target_width, top + target_height))


def render_jump_results_hold(
    image: Image.Image,
    output_path: Path,
    *,
    top_holds: tuple[JumpPeakHold, ...] = (),
    output_size: tuple[int, int] | None = None,
    publish_safe_zone: bool = False,
    session_title: str | None = None,
    session_subtitle: str | None = None,
) -> None:
    """Render a final freeze frame that makes jump results readable."""

    size = output_size or image.size
    base = _cover_resize(image.convert("RGB"), size).filter(ImageFilter.GaussianBlur(radius=3.0)).convert("RGBA")
    dim = Image.new("RGBA", size, (0, 0, 0, 150))
    base.alpha_composite(dim)
    draw = ImageDraw.Draw(base)
    holds = confirmed_top_jump_holds(top_holds)
    title_font_size = max(38, min(size) // 11)
    label_font_size = max(28, min(size) // 16)
    title_font = _default_font(title_font_size)
    label_font = _default_font(label_font_size)
    margin = max(24, min(size) // 24)
    if not holds:
        _draw_text_with_stroke(
            draw,
            (margin, margin),
            "top jumps",
            fill=(255, 245, 255, 245),
            font=title_font,
            stroke_width=5,
        )
        _draw_text_with_stroke(
            draw,
            (margin, margin + title_font_size + 20),
            "no confirmed jumps",
            fill=(255, 220, 220, 235),
            font=label_font,
            stroke_width=4,
        )
        _save_jpeg_atomic(base.convert("RGB"), output_path)
        return

    if size[1] > size[0]:
        _draw_top_jump_holds_top_bar(
            base,
            holds,
            font=_default_font(42),
            publish_safe_zone=publish_safe_zone,
            session_title=session_title,
            session_subtitle=session_subtitle,
        )
        _save_jpeg_atomic(base.convert("RGB"), output_path)
        return

    _draw_text_with_stroke(
        draw,
        (margin, margin),
        "top jumps",
        fill=(255, 245, 255, 245),
        font=title_font,
        stroke_width=5,
    )

    gap = max(20, size[0] // 70)
    y0 = margin + max(74, size[1] // 9)
    slot_width = (size[0] - margin * 2 - gap * 2) // 3
    slot_height = size[1] - y0 - margin
    layout = [(margin + index * (slot_width + gap), y0, slot_width, slot_height) for index in range(min(3, len(holds)))]

    for index, (hold, (x0, y0, slot_width, slot_height)) in enumerate(zip(holds[:3], layout, strict=False), start=1):
        draw.rounded_rectangle(
            (x0, y0, x0 + slot_width, y0 + slot_height),
            radius=10,
            fill=(0, 0, 0, 165),
            outline=(255, 64, 190, 235),
            width=max(3, min(size) // 160),
        )
        label = f"#{index} {hold.height_cm:.1f} cm"
        _draw_text_with_stroke(
            draw,
            (x0 + 18, y0 + 12),
            label,
            fill=(255, 245, 255, 250),
            font=label_font,
            stroke_width=4,
        )
        thumb_top = y0 + max(58, label_font_size + 24)
        thumb_box = (
            max(1, slot_width - 28),
            max(1, slot_height - (thumb_top - y0) - 16),
        )
        thumb = _fit_image_on_canvas(hold.image, thumb_box, (0, 0, 0, 255))
        base.alpha_composite(thumb, (x0 + 14, thumb_top))
    _save_jpeg_atomic(base.convert("RGB"), output_path)


def annotate_image(
    image: Image.Image,
    sample: LiveJumpSample,
    *,
    homography: Homography | None = None,
    floor_y_mm: float | None = None,
    track_state: JumpTrackState | None = None,
    top_holds: tuple[JumpPeakHold, ...] = (),
) -> Image.Image:
    annotated = image.convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    label_font = _default_font(max(22, image.width // 42))
    status_font = _default_font(max(30, image.width // 30))
    status_fill = (74, 222, 128) if sample.triggered else (255, 210, 64)
    confirmed_holds = confirmed_top_jump_holds(top_holds)
    display_max_height_cm = confirmed_holds[0].height_cm if confirmed_holds else None
    if track_state is not None:
        _draw_track_state(
            annotated,
            track_state,
            font=label_font,
            homography=homography,
            floor_y_mm=floor_y_mm,
            display_max_height_cm=display_max_height_cm,
        )
    _draw_top_jump_holds(annotated, top_holds, font=label_font)
    if sample.component_bbox_xyxy is not None:
        draw.rectangle(sample.component_bbox_xyxy, outline=status_fill, width=4)
    if sample.top_x_px is not None and sample.top_y_px is not None:
        x = sample.top_x_px
        y = sample.top_y_px
        if homography is not None and floor_y_mm is not None and sample.wall_x_mm is not None:
            try:
                floor_x, floor_y = homography.unproject(sample.wall_x_mm, floor_y_mm)
                draw.line((x, y, floor_x, floor_y), fill=(255, 64, 64), width=5)
                draw.line((floor_x - 20, floor_y, floor_x + 20, floor_y), fill=(255, 64, 64), width=4)
            except ValueError:
                pass
        draw.line((x - 18, y, x + 18, y), fill=(255, 64, 64), width=4)
        draw.line((x, y - 18, x, y + 18), fill=(255, 64, 64), width=4)
        height_label = f"{sample.top_height_cm:.1f} cm" if sample.top_height_cm is not None else "visible top"
        label = height_label
        _draw_text_with_stroke(
            draw,
            (max(8, x + 12), max(8, y - 36)),
            label,
            fill=(255, 255, 255),
            font=label_font,
        )
    else:
        if track_state is None or not track_state.trail_points:
            draw.text((16, 14), "NO CAT", fill=(255, 96, 96), font=status_font, stroke_width=4, stroke_fill=(0, 0, 0))
    return annotated


def render_annotation(
    image: Image.Image,
    sample: LiveJumpSample,
    output_path: Path,
    *,
    homography: Homography | None = None,
    floor_y_mm: float | None = None,
    track_state: JumpTrackState | None = None,
    top_holds: tuple[JumpPeakHold, ...] = (),
) -> None:
    annotated = annotate_image(
        image,
        sample,
        homography=homography,
        floor_y_mm=floor_y_mm,
        track_state=track_state,
        top_holds=top_holds,
    )
    _save_jpeg_atomic(annotated.convert("RGB"), output_path)


def _vertical_target_x(
    *,
    image: Image.Image,
    sample: LiveJumpSample,
    track_state: JumpTrackState | None,
) -> float:
    if sample.component_bbox_xyxy is not None:
        left, _top, right, _bottom = sample.component_bbox_xyxy
        return (left + right) / 2.0
    if sample.top_x_px is not None:
        return float(sample.top_x_px)
    if track_state is not None and track_state.trail_points:
        return float(track_state.trail_points[-1][0])
    return image.width / 2.0


def _vertical_target_y(
    *,
    image: Image.Image,
    sample: LiveJumpSample,
    track_state: JumpTrackState | None,
) -> float:
    if sample.component_bbox_xyxy is not None:
        _left, top, _right, bottom = sample.component_bbox_xyxy
        return top * 0.42 + bottom * 0.58
    if sample.top_y_px is not None:
        return float(sample.top_y_px)
    if track_state is not None and track_state.trail_points:
        return float(track_state.trail_points[-1][1])
    return image.height * 0.58


def _vertical_avoid_bbox(
    *,
    sample: LiveJumpSample,
    crop_left: int,
    crop_top: int,
    scale: float,
) -> tuple[int, int, int, int] | None:
    if sample.component_bbox_xyxy is None:
        return None
    left, top, right, bottom = sample.component_bbox_xyxy
    return (
        int(round((left - crop_left) * scale)),
        int(round((top - crop_top) * scale)),
        int(round((right - crop_left) * scale)),
        int(round((bottom - crop_top) * scale)),
    )


def _crop_with_context_padding(
    source: Image.Image,
    *,
    crop_left: float,
    crop_top: float,
    crop_width: float,
    crop_height: float,
    output_size: tuple[int, int],
) -> Image.Image:
    canvas_width = max(1, int(round(crop_width)))
    canvas_height = max(1, int(round(crop_height)))
    background = Image.new("RGB", (canvas_width, canvas_height), (10, 10, 10))
    left_i = int(math.floor(crop_left))
    top_i = int(math.floor(crop_top))
    right_i = int(math.ceil(crop_left + crop_width))
    bottom_i = int(math.ceil(crop_top + crop_height))
    src_left = max(0, left_i)
    src_top = max(0, top_i)
    src_right = min(source.width, right_i)
    src_bottom = min(source.height, bottom_i)
    if src_right > src_left and src_bottom > src_top:
        paste_x = src_left - left_i
        paste_y = src_top - top_i
        background.paste(source.crop((src_left, src_top, src_right, src_bottom)), (paste_x, paste_y))
    return background.resize(output_size, Image.Resampling.LANCZOS)


def render_vertical_annotation(
    image: Image.Image,
    sample: LiveJumpSample,
    output_path: Path,
    *,
    homography: Homography | None = None,
    floor_y_mm: float | None = None,
    track_state: JumpTrackState | None = None,
    top_holds: tuple[JumpPeakHold, ...] = (),
    crop_state: VerticalCropState | None = None,
    output_size: tuple[int, int] = (1080, 1920),
    close_crop_bounds: tuple[float, float, float, float] | None = None,
    publish_safe_zone: bool = False,
    session_title: str | None = None,
    session_subtitle: str | None = None,
) -> VerticalCropState:
    out_width, out_height = output_size
    confirmed_holds = confirmed_top_jump_holds(top_holds)
    show_current_marker = _show_current_vertical_marker(sample, confirmed_holds=confirmed_holds)
    if close_crop_bounds is not None:
        crop_left_f, crop_top_f, crop_width_f, crop_height_f = close_crop_bounds
        crop_width = max(1, int(round(crop_width_f)))
        crop_height = max(1, int(round(crop_height_f)))
        crop_left = int(round(crop_left_f))
        crop_top = int(round(crop_top_f))
        center_x = crop_left_f + crop_width_f / 2.0
        center_y = crop_top_f + crop_height_f / 2.0
        portrait = _crop_with_context_padding(
            image.convert("RGB"),
            crop_left=crop_left_f,
            crop_top=crop_top_f,
            crop_width=crop_width_f,
            crop_height=crop_height_f,
            output_size=output_size,
        )
    else:
        crop_height = min(image.height, max(500, int(round(image.height * 0.78))))
        crop_width = min(image.width, int(round(crop_height * out_width / out_height)))
        previous_x = crop_state.center_x_px if crop_state is not None else None
        previous_y = crop_state.center_y_px if crop_state is not None else None
        if show_current_marker:
            target_x = _vertical_target_x(image=image, sample=sample, track_state=track_state)
            target_y = _vertical_target_y(image=image, sample=sample, track_state=track_state)
        elif track_state is not None and track_state.trail_points:
            target_x = float(track_state.trail_points[-1][0])
            target_y = float(track_state.trail_points[-1][1])
        elif previous_x is not None:
            target_x = previous_x
            target_y = previous_y if previous_y is not None else image.height * 0.58
        else:
            target_x = image.width / 2.0
            target_y = image.height * 0.58
        if previous_x is None:
            center_x = target_x
        else:
            max_step = max(10.0, crop_width * 0.055)
            delta = max(-max_step, min(max_step, target_x - previous_x))
            center_x = previous_x + delta
        if previous_y is None:
            center_y = target_y
        else:
            max_step_y = max(14.0, crop_height * 0.060)
            delta_y = max(-max_step_y, min(max_step_y, target_y - previous_y))
            center_y = previous_y + delta_y
        if show_current_marker and sample.component_bbox_xyxy is not None:
            left, top, right, bottom = sample.component_bbox_xyxy
            margin = crop_width * 0.12
            vertical_margin = crop_height * 0.16
            crop_left = center_x - crop_width / 2.0
            crop_top = center_y - crop_height / 2.0
            if left < crop_left + margin:
                center_x -= (crop_left + margin) - left
            if right > crop_left + crop_width - margin:
                center_x += right - (crop_left + crop_width - margin)
            if top < crop_top + vertical_margin:
                center_y -= (crop_top + vertical_margin) - top
            if bottom > crop_top + crop_height - vertical_margin:
                center_y += bottom - (crop_top + crop_height - vertical_margin)
        half_width = crop_width / 2.0
        half_height = crop_height / 2.0
        center_x = max(half_width, min(float(image.width) - half_width, center_x))
        center_y = max(half_height, min(float(image.height) - half_height, center_y))
        crop_left = int(round(center_x - half_width))
        crop_top = int(round(center_y - half_height))
        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height
        if crop_right > image.width:
            crop_left = image.width - crop_width
            crop_right = image.width
        if crop_left < 0:
            crop_left = 0
            crop_right = crop_width
        if crop_bottom > image.height:
            crop_top = image.height - crop_height
            crop_bottom = image.height
        if crop_top < 0:
            crop_top = 0
            crop_bottom = crop_height
        portrait = (
            image.convert("RGB")
            .crop((crop_left, crop_top, crop_right, crop_bottom))
            .resize(output_size, Image.Resampling.LANCZOS)
        )
    portrait = ImageEnhance.Contrast(portrait).enhance(1.08)
    portrait = (
        ImageEnhance.Sharpness(portrait)
        .enhance(1.18)
        .filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=4))
    )
    label_font = _default_font(max(42, out_width // 24))
    scale = out_width / float(crop_width)
    portrait_rgba = portrait.convert("RGBA")
    _draw_vertical_hud(portrait_rgba)
    display_max_height_cm = confirmed_holds[0].height_cm if confirmed_holds else None
    if track_state is not None:
        _draw_vertical_track_state(
            portrait_rgba,
            track_state,
            font=label_font,
            homography=homography,
            floor_y_mm=floor_y_mm,
            display_max_height_cm=display_max_height_cm,
            source_image=image,
            crop_left=crop_left,
            crop_top=crop_top,
            scale=scale,
        )
    _draw_top_jump_holds_top_bar(
        portrait_rgba,
        top_holds,
        font=label_font,
        avoid_bbox_xyxy=(
            _vertical_avoid_bbox(sample=sample, crop_left=crop_left, crop_top=crop_top, scale=scale)
            if show_current_marker
            else None
        ),
        publish_safe_zone=publish_safe_zone,
        session_title=session_title,
        session_subtitle=session_subtitle,
    )
    if show_current_marker:
        _draw_vertical_sample_marker(
            portrait_rgba,
            sample,
            font=label_font,
            homography=homography,
            floor_y_mm=floor_y_mm,
            crop_left=crop_left,
            crop_top=crop_top,
            scale=scale,
        )
    _save_jpeg_atomic(portrait_rgba.convert("RGB"), output_path)
    return VerticalCropState(center_x_px=center_x, center_y_px=center_y)


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _sample_to_payload(sample: LiveJumpSample) -> dict[str, Any]:
    return {"kind": "cat_projector_live_jump_sample_v1", **asdict(sample)}


def _active_detector_model(args: argparse.Namespace) -> Path | None:
    if getattr(args, "disable_detector", False):
        return None
    if args.detector_model is not None:
        return args.detector_model
    if default_model_available():
        return DEFAULT_DETECTOR_MODEL_PATH
    return None


def run_once(args: argparse.Namespace) -> LiveJumpSample:
    calibration = args.calibration.expanduser()
    homography, floor_y_mm = _read_calibration(calibration)
    source = args.image or args.snapshot_url
    image = _load_image(source, timeout_seconds=args.timeout_seconds)
    detector_model = _active_detector_model(args)
    detector_loaded_model = None
    detector_loaded_metadata = None
    if detector_model is not None:
        detector_loaded_model, detector_loaded_metadata = load_model(
            detector_model,
            args.detector_metadata or DEFAULT_DETECTOR_METADATA_PATH,
        )
    sample, _component = sample_image(
        image,
        source=source,
        homography=homography,
        floor_y_mm=floor_y_mm,
        roi=args.roi,
        threshold=args.dark_threshold,
        scale=args.scale,
        min_area_px=args.min_area_px,
        min_height_px=args.min_height_px,
        max_width_px=args.max_width_px,
        jump_threshold_cm=args.jump_threshold_cm,
        detector_model=detector_model,
        detector_metadata=args.detector_metadata,
        detector_loaded_model=detector_loaded_model,
        detector_loaded_metadata=detector_loaded_metadata,
        detector_min_probability=args.detector_min_probability,
    )
    if args.output_image and sample.candidate_found:
        output_image = args.output_image.expanduser()
        render_annotation(image, sample, output_image)
        sample = LiveJumpSample(**(asdict(sample) | {"image_path": str(output_image)}))
    print(json.dumps(_sample_to_payload(sample), ensure_ascii=False, sort_keys=True))
    return sample


def watch(args: argparse.Namespace) -> None:
    calibration = args.calibration.expanduser()
    homography, floor_y_mm = _read_calibration(calibration)
    output_dir = args.output_dir.expanduser()
    events_path = output_dir / "events.jsonl"
    latest_image_path = args.latest_image.expanduser() if args.latest_image else output_dir / "latest_jump_overlay.jpg"
    latest_json_path = args.latest_json.expanduser() if args.latest_json else output_dir / "latest_jump_sample.json"
    interval_seconds = 1.0 / args.fps
    deadline = time.monotonic() + args.duration_seconds if args.duration_seconds else None
    last_trigger_monotonic = 0.0
    track_state = JumpTrackState()
    detector_model = _active_detector_model(args)
    detector_loaded_model = None
    detector_loaded_metadata = None

    frame_iter = iter_mjpeg_frames(args.stream_url, timeout_seconds=args.timeout_seconds) if args.stream_url else None
    while deadline is None or time.monotonic() < deadline:
        loop_started = time.monotonic()
        try:
            if detector_model is not None and (detector_loaded_model is None or detector_loaded_metadata is None):
                try:
                    detector_loaded_model, detector_loaded_metadata = load_model(
                        detector_model,
                        args.detector_metadata or DEFAULT_DETECTOR_METADATA_PATH,
                    )
                except Exception as exc:  # noqa: BLE001
                    payload = {
                        "kind": "cat_projector_live_jump_error_v1",
                        "timestamp": _now_iso(),
                        "source": args.stream_url or args.snapshot_url,
                        "error": f"detector disabled after load failure: {exc}",
                    }
                    _write_jsonl(events_path, payload)
                    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
                    detector_model = None

            source = args.stream_url or args.snapshot_url
            image = (
                next(frame_iter)
                if frame_iter is not None
                else _load_image(args.snapshot_url, timeout_seconds=args.timeout_seconds)
            )
            sample, _component = sample_image(
                image,
                source=source,
                homography=homography,
                floor_y_mm=floor_y_mm,
                roi=args.roi,
                threshold=args.dark_threshold,
                scale=args.scale,
                min_area_px=args.min_area_px,
                min_height_px=args.min_height_px,
                max_width_px=args.max_width_px,
                jump_threshold_cm=args.jump_threshold_cm,
                detector_model=detector_model,
                detector_metadata=args.detector_metadata,
                detector_loaded_model=detector_loaded_model,
                detector_loaded_metadata=detector_loaded_metadata,
                detector_min_probability=args.detector_min_probability,
            )
            track_state = update_track_state(track_state, sample, max_points=args.track_points)
            payload = _sample_to_payload(sample)
            if not args.no_latest_overlay:
                render_annotation(
                    image,
                    sample,
                    latest_image_path,
                    homography=homography,
                    floor_y_mm=floor_y_mm,
                    track_state=None if args.no_track_overlay else track_state,
                )
                latest_payload = payload | {
                    "image_path": str(latest_image_path),
                    "track_points": len(track_state.trail_points),
                    "hold_max_height_cm": track_state.highest_sample.top_height_cm
                    if track_state.highest_sample
                    else None,
                }
                _write_json_atomic(latest_json_path, latest_payload)
            if sample.triggered and loop_started - last_trigger_monotonic >= args.cooldown_seconds:
                image_path = output_dir / "frames" / f"jump_{_slug_timestamp()}.jpg"
                render_annotation(
                    image,
                    sample,
                    image_path,
                    homography=homography,
                    floor_y_mm=floor_y_mm,
                    track_state=None if args.no_track_overlay else track_state,
                )
                payload["image_path"] = str(image_path)
                last_trigger_monotonic = loop_started
                _write_jsonl(events_path, payload)
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
            elif args.print_samples:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception as exc:  # noqa: BLE001
            payload = {
                "kind": "cat_projector_live_jump_error_v1",
                "timestamp": _now_iso(),
                "source": args.stream_url or args.snapshot_url,
                "error": str(exc),
            }
            _write_jsonl(events_path, payload)
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)

        elapsed = time.monotonic() - loop_started
        time.sleep(max(0.0, interval_seconds - elapsed))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION.expanduser())
    parser.add_argument("--roi", type=_parse_roi, default=_parse_roi("0,0,1280,720"))
    parser.add_argument("--dark-threshold", type=int, default=80)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--min-area-px", type=int, default=2500)
    parser.add_argument("--min-height-px", type=int, default=45)
    parser.add_argument("--max-width-px", type=int, default=360)
    parser.add_argument("--jump-threshold-cm", type=float, default=85.0)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--detector-model", type=_parse_optional_path, default=None)
    parser.add_argument("--detector-metadata", type=_parse_optional_path, default=DEFAULT_DETECTOR_METADATA_PATH)
    parser.add_argument("--detector-min-probability", type=float, default=0.60)
    parser.add_argument("--disable-detector", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    once_parser = subparsers.add_parser("once", help="Measure one image or live snapshot and print JSON.")
    _add_common_arguments(once_parser)
    once_parser.add_argument("--snapshot-url", default=DEFAULT_SNAPSHOT_URL)
    once_parser.add_argument("--image", help="Local image path; overrides --snapshot-url.")
    once_parser.add_argument("--output-image", type=Path)

    watch_parser = subparsers.add_parser("watch", help="Poll live snapshots and write triggered jump events.")
    _add_common_arguments(watch_parser)
    watch_parser.add_argument("--snapshot-url", default=DEFAULT_SNAPSHOT_URL)
    watch_parser.add_argument(
        "--stream-url", default=DEFAULT_STREAM_URL, help="MJPEG stream URL; set empty string to poll --snapshot-url."
    )
    watch_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR.expanduser())
    watch_parser.add_argument("--fps", type=float, default=20.0)
    watch_parser.add_argument("--duration-seconds", type=float)
    watch_parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    watch_parser.add_argument("--print-samples", action="store_true")
    watch_parser.add_argument("--latest-image", type=Path, help="Continuously overwritten annotated JPEG preview.")
    watch_parser.add_argument("--latest-json", type=Path, help="Continuously overwritten latest sample JSON.")
    watch_parser.add_argument("--no-latest-overlay", action="store_true")
    watch_parser.add_argument("--track-points", type=int, default=180, help="Number of recent cat-top points to draw.")
    watch_parser.add_argument("--no-track-overlay", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "once":
        run_once(args)
    elif args.command == "watch":
        watch(args)
    else:  # pragma: no cover
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
