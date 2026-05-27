"""Eye-safety overlays for projector play sessions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

Point = tuple[float, float]
BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class PersonDetection:
    """A person detection in projector-camera pixel coordinates."""

    bbox_xyxy: BBoxXYXY
    confidence: float
    source: str = "person_detector"
    mask: np.ndarray | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyOverlayZone:
    """A black overlay polygon in source-video pixel coordinates."""

    polygon: tuple[Point, ...]
    camera_bbox_xyxy: BBoxXYXY
    camera_eye_band_xyxy: BBoxXYXY | None = None
    camera_projected_polygon: tuple[Point, ...] = ()
    camera_eye_band_coverage: float | None = None
    source: str = "projector_eye_safety_eye_band"
    confidence: float | None = None
    camera_overlap_area_px: int = 0


@dataclass(frozen=True)
class SafetyOverlayResult:
    """Computed eye-safety zones plus the decision status."""

    status: str
    zones: tuple[SafetyOverlayZone, ...] = ()
    debug: dict[str, Any] = field(default_factory=dict)


def compute_eye_safety_overlay(
    *,
    camera_size: tuple[int, int],
    source_size: tuple[int, int],
    projector_polygon: Iterable[Point],
    people: Iterable[PersonDetection],
    eye_band_top_fraction: float = 0.07,
    eye_band_bottom_fraction: float = 0.19,
    eye_band_left_fraction: float = 0.32,
    eye_band_right_fraction: float = 0.68,
    padding_px: int = 12,
    min_overlap_area_px: int = 24,
) -> SafetyOverlayResult:
    """Map projected person eye bands from camera pixels into source pixels.

    If geometry tooling is unavailable, fail open by returning
    ``safety_overlay_unavailable``. The caller can keep playback unchanged and
    surface the debug payload instead of silently claiming protection.
    """

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on optional review tooling.
        return SafetyOverlayResult("safety_overlay_unavailable", debug={"error": f"cv2 import failed: {exc}"})

    camera_width, camera_height = _positive_size(camera_size, "camera_size")
    source_width, source_height = _positive_size(source_size, "source_size")
    projector_points = tuple((float(x), float(y)) for x, y in projector_polygon)
    if len(projector_points) != 4:
        return SafetyOverlayResult(
            "safety_overlay_unavailable",
            debug={"error": "projector_polygon must contain four points", "point_count": len(projector_points)},
        )

    people = tuple(people)
    if not people:
        return SafetyOverlayResult("no_person", debug={"person_count": 0})

    projection_mask = _polygon_mask(projector_points, camera_width, camera_height)
    if not projection_mask.any():
        return SafetyOverlayResult("safety_overlay_unavailable", debug={"error": "empty projector polygon mask"})

    inverse_homography = _camera_to_source_homography(
        projector_points,
        source_width=source_width,
        source_height=source_height,
        cv2=cv2,
    )
    source_to_camera_homography = np.linalg.inv(inverse_homography).astype(np.float32)

    zones: list[SafetyOverlayZone] = []
    skipped: list[dict[str, Any]] = []
    for index, person in enumerate(people):
        eye_band = _person_eye_band_bbox(
            person,
            camera_width=camera_width,
            camera_height=camera_height,
            eye_band_top_fraction=eye_band_top_fraction,
            eye_band_bottom_fraction=eye_band_bottom_fraction,
            eye_band_left_fraction=eye_band_left_fraction,
            eye_band_right_fraction=eye_band_right_fraction,
            padding_px=padding_px,
        )
        eye_mask = _person_eye_band_mask(
            person,
            eye_band=eye_band,
            camera_width=camera_width,
            camera_height=camera_height,
        )
        overlap = eye_mask & projection_mask
        overlap_area = int(overlap.sum())
        if overlap_area < min_overlap_area_px:
            skipped.append({"index": index, "reason": "no_projection_overlap", "overlap_area_px": overlap_area})
            continue

        polygon = _mask_to_source_polygon(
            overlap,
            inverse_homography=inverse_homography,
            source_width=source_width,
            source_height=source_height,
            cv2=cv2,
        )
        if len(polygon) < 3:
            skipped.append({"index": index, "reason": "empty_source_polygon", "overlap_area_px": overlap_area})
            continue
        camera_projected_polygon = _source_polygon_to_camera(
            polygon,
            source_to_camera_homography=source_to_camera_homography,
            camera_width=camera_width,
            camera_height=camera_height,
            cv2=cv2,
        )
        camera_projected_mask = _polygon_mask(camera_projected_polygon, camera_width, camera_height)
        coverage = _mask_coverage(overlap, camera_projected_mask)
        zones.append(
            SafetyOverlayZone(
                polygon=polygon,
                camera_bbox_xyxy=_clamp_bbox(person.bbox_xyxy, camera_width, camera_height),
                camera_eye_band_xyxy=eye_band,
                camera_projected_polygon=camera_projected_polygon,
                camera_eye_band_coverage=coverage,
                confidence=float(person.confidence),
                camera_overlap_area_px=overlap_area,
            )
        )

    if not zones:
        return SafetyOverlayResult(
            "no_projection_overlap",
            debug={"person_count": len(people), "skipped": skipped},
        )

    return SafetyOverlayResult(
        "active",
        zones=tuple(zones),
        debug={"person_count": len(people), "zone_count": len(zones), "skipped": skipped},
    )


def render_eye_safety_overlay(
    source_frame: Image.Image,
    result: SafetyOverlayResult,
    *,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Draw black safety zones onto a source-video frame."""

    output = source_frame.convert("RGB")
    if result.status != "active":
        return output
    draw = ImageDraw.Draw(output)
    for zone in result.zones:
        if len(zone.polygon) >= 3:
            draw.polygon(zone.polygon, fill=fill)
    return output


def _positive_size(size: tuple[int, int], label: str) -> tuple[int, int]:
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} must be positive")
    return width, height


def _camera_to_source_homography(
    projector_polygon: tuple[Point, ...],
    *,
    source_width: int,
    source_height: int,
    cv2: Any,
) -> np.ndarray:
    source_points = np.float32(
        [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]]
    )
    camera_points = np.float32(projector_polygon)
    source_to_camera = cv2.getPerspectiveTransform(source_points, camera_points)
    inverse = np.linalg.inv(source_to_camera)
    return inverse.astype(np.float32)


def _polygon_mask(points: tuple[Point, ...], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return np.asarray(mask, dtype=np.uint8) > 0


def _clamp_bbox(bbox_xyxy: BBoxXYXY, width: int, height: int) -> BBoxXYXY:
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    left = max(0.0, min(float(width - 1), min(x0, x1)))
    top = max(0.0, min(float(height - 1), min(y0, y1)))
    right = max(0.0, min(float(width), max(x0, x1)))
    bottom = max(0.0, min(float(height), max(y0, y1)))
    return left, top, right, bottom


def _person_eye_band_bbox(
    person: PersonDetection,
    *,
    camera_width: int,
    camera_height: int,
    eye_band_top_fraction: float,
    eye_band_bottom_fraction: float,
    eye_band_left_fraction: float,
    eye_band_right_fraction: float,
    padding_px: int,
) -> BBoxXYXY:
    if not 0.0 <= eye_band_top_fraction < eye_band_bottom_fraction <= 1.0:
        raise ValueError("eye band fractions must satisfy 0 <= top < bottom <= 1")
    if not 0.0 <= eye_band_left_fraction < eye_band_right_fraction <= 1.0:
        raise ValueError("eye band horizontal fractions must satisfy 0 <= left < right <= 1")
    x0, y0, x1, y1 = _clamp_bbox(person.bbox_xyxy, camera_width, camera_height)
    person_width = x1 - x0
    person_height = y1 - y0
    eye_left = x0 + person_width * eye_band_left_fraction
    eye_right = x0 + person_width * eye_band_right_fraction
    eye_top = y0 + person_height * eye_band_top_fraction
    eye_bottom = y0 + person_height * eye_band_bottom_fraction
    left = max(0, int(np.floor(eye_left - padding_px)))
    top = max(0, int(np.floor(eye_top - padding_px)))
    right = min(camera_width, int(np.ceil(eye_right + padding_px)))
    bottom = min(camera_height, int(np.ceil(eye_bottom + padding_px)))
    eye_band = (float(left), float(top), float(right), float(bottom))
    return _apply_eye_band_prediction(
        eye_band,
        person.debug,
        camera_width=camera_width,
        camera_height=camera_height,
    )


def _apply_eye_band_prediction(
    eye_band: BBoxXYXY,
    debug: dict[str, Any],
    *,
    camera_width: int,
    camera_height: int,
) -> BBoxXYXY:
    raw_offset = debug.get("prediction_offset_px")
    if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
        return eye_band
    try:
        offset_x = float(raw_offset[0])
        offset_y = float(raw_offset[1])
    except (TypeError, ValueError):
        return eye_band
    raw_padding = debug.get("prediction_padding_px", 0.0)
    try:
        prediction_padding = max(0.0, float(raw_padding))
    except (TypeError, ValueError):
        prediction_padding = 0.0
    left, top, right, bottom = eye_band
    predicted = (left + offset_x, top + offset_y, right + offset_x, bottom + offset_y)
    union = (
        min(left, predicted[0]) - prediction_padding,
        min(top, predicted[1]) - prediction_padding,
        max(right, predicted[2]) + prediction_padding,
        max(bottom, predicted[3]) + prediction_padding,
    )
    return _clamp_bbox(union, camera_width, camera_height)


def _person_eye_band_mask(
    person: PersonDetection,
    *,
    eye_band: BBoxXYXY,
    camera_width: int,
    camera_height: int,
) -> np.ndarray:
    left, top, right, bottom = (int(round(value)) for value in eye_band)
    mask = np.zeros((camera_height, camera_width), dtype=bool)
    if right <= left or bottom <= top:
        return mask

    mask[top:bottom, left:right] = True
    if person.mask is not None:
        person_mask = np.asarray(person.mask, dtype=bool)
        if person_mask.shape == mask.shape:
            padded_person = np.zeros_like(mask)
            padded_person[top:bottom, left:right] = True
            mask &= person_mask | padded_person
    return mask


def _mask_to_source_polygon(
    mask: np.ndarray,
    *,
    inverse_homography: np.ndarray,
    source_width: int,
    source_height: int,
    cv2: Any,
) -> tuple[Point, ...]:
    mask_u8 = mask.astype(np.uint8)
    contours, _hierarchy = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ()
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.5, 0.02 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    camera_points = approx.reshape(-1, 2).astype(np.float32).reshape(-1, 1, 2)
    source_points = cv2.perspectiveTransform(camera_points, inverse_homography).reshape(-1, 2)
    polygon: list[Point] = []
    for x, y in source_points:
        polygon.append(
            (
                float(max(0.0, min(float(source_width - 1), float(x)))),
                float(max(0.0, min(float(source_height - 1), float(y)))),
            )
        )
    return tuple(polygon)


def _source_polygon_to_camera(
    polygon: tuple[Point, ...],
    *,
    source_to_camera_homography: np.ndarray,
    camera_width: int,
    camera_height: int,
    cv2: Any,
) -> tuple[Point, ...]:
    source_points = np.float32(polygon).reshape(-1, 1, 2)
    camera_points = cv2.perspectiveTransform(source_points, source_to_camera_homography).reshape(-1, 2)
    projected: list[Point] = []
    for x, y in camera_points:
        projected.append(
            (
                float(max(0.0, min(float(camera_width - 1), float(x)))),
                float(max(0.0, min(float(camera_height - 1), float(y)))),
            )
        )
    return tuple(projected)


def _mask_coverage(target: np.ndarray, cover: np.ndarray) -> float:
    target_area = int(target.sum())
    if target_area <= 0:
        return 0.0
    return float((target & cover).sum() / target_area)
