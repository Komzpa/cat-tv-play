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
    source: str = "projector_eye_safety_head_band"
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
    head_fraction: float = 0.45,
    padding_px: int = 18,
    min_overlap_area_px: int = 24,
) -> SafetyOverlayResult:
    """Map projected person head bands from camera pixels into source pixels.

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

    zones: list[SafetyOverlayZone] = []
    skipped: list[dict[str, Any]] = []
    for index, person in enumerate(people):
        head_mask = _person_head_mask(
            person,
            camera_width=camera_width,
            camera_height=camera_height,
            head_fraction=head_fraction,
            padding_px=padding_px,
        )
        overlap = head_mask & projection_mask
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
        zones.append(
            SafetyOverlayZone(
                polygon=polygon,
                camera_bbox_xyxy=_clamp_bbox(person.bbox_xyxy, camera_width, camera_height),
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


def _person_head_mask(
    person: PersonDetection,
    *,
    camera_width: int,
    camera_height: int,
    head_fraction: float,
    padding_px: int,
) -> np.ndarray:
    if not 0.0 < head_fraction <= 1.0:
        raise ValueError("head_fraction must be in the range (0, 1]")
    x0, y0, x1, y1 = _clamp_bbox(person.bbox_xyxy, camera_width, camera_height)
    head_bottom = y0 + (y1 - y0) * head_fraction
    left = max(0, int(np.floor(x0 - padding_px)))
    top = max(0, int(np.floor(y0 - padding_px)))
    right = min(camera_width, int(np.ceil(x1 + padding_px)))
    bottom = min(camera_height, int(np.ceil(head_bottom + padding_px)))

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
