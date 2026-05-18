"""Projector source subtraction helpers.

When the clip shown on the projector is known, projected prey should not be
classified as a cat candidate. These helpers warp the expected source frame into
the camera image, subtract it from the camera frame, and return remaining dark
residual components for a downstream cat/not-cat detector or human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SourceSubtractionCandidate:
    """A residual blob candidate in camera pixel coordinates."""

    bbox_xywh: tuple[float, float, float, float]
    top_x_px: float
    top_y_px: float
    area_px: int
    source: str = "source_subtracted_projector_residual"


def projector_polygon_for_size(
    projector_polygon_1280: Iterable[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[tuple[float, float], ...]:
    """Scale a 1280x720 projector polygon to a camera frame size."""

    scale_x = width / 1280.0
    scale_y = height / 720.0
    return tuple((x * scale_x, y * scale_y) for x, y in projector_polygon_1280)


def projection_mask(projector_polygon: Iterable[tuple[float, float]], width: int, height: int) -> np.ndarray:
    """Return a boolean mask for the projected wall area."""

    mask_image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_image).polygon(tuple(projector_polygon), fill=255)
    return np.asarray(mask_image) > 0


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


def warp_source_to_camera(
    source_frame: Image.Image,
    *,
    projector_polygon: Iterable[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Warp a 1280x720 source frame into the camera's projector polygon."""

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on optional review tooling.
        raise RuntimeError("opencv-python is required for projector source subtraction") from exc

    source_gray = np.asarray(source_frame.convert("L").resize((1280, 720)), dtype=np.uint8)
    source_points = np.float32([[0, 0], [1279, 0], [1279, 719], [0, 719]])
    camera_points = np.float32(tuple(projector_polygon))
    homography = cv2.getPerspectiveTransform(source_points, camera_points)
    return cv2.warpPerspective(source_gray, homography, (width, height))


def source_subtracted_residual(
    camera_frame: Image.Image,
    *,
    source_frame: Image.Image,
    projector_polygon: Iterable[tuple[float, float]],
    room_background: Image.Image | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dark residual after explaining the projector plane by source video."""

    camera_gray = np.asarray(camera_frame.convert("L"), dtype=np.uint8)
    height, width = camera_gray.shape
    projector_polygon = tuple(projector_polygon)
    projection = projection_mask(projector_polygon, width, height)
    warped_source = warp_source_to_camera(
        source_frame,
        projector_polygon=projector_polygon,
        width=width,
        height=height,
    )
    if room_background is None:
        expected = camera_gray.astype(np.float32)
    else:
        expected = np.asarray(room_background.convert("L").resize((width, height)), dtype=np.uint8).astype(np.float32)

    fit_mask = projection.copy()
    fit_mask[int(height * 0.69) :, :] = False
    fit_mask[:, int(width * 0.82) :] = False
    scale, offset = _robust_linear_match(warped_source, camera_gray, fit_mask)
    expected[projection] = np.clip(scale * warped_source[projection].astype(np.float32) + offset, 0, 255)
    return expected - camera_gray.astype(np.float32), warped_source


def detect_source_subtracted_candidates(
    camera_frame: Image.Image,
    *,
    source_frame: Image.Image,
    projector_polygon: Iterable[tuple[float, float]],
    room_background: Image.Image | None = None,
    residual_baseline: np.ndarray | Image.Image | None = None,
    threshold: float = 26.0,
    source_bright_threshold: int = 150,
    min_area_px: int = 40,
) -> list[SourceSubtractionCandidate]:
    """Find residual candidates after subtracting the known projected source.

    This is candidate generation, not a final cat decision. A downstream model
    or reviewer still decides whether each candidate is the cat.
    """

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on optional review tooling.
        raise RuntimeError("opencv-python is required for projector source subtraction") from exc

    width, height = camera_frame.size
    projector_polygon = tuple(projector_polygon)
    projection = projection_mask(projector_polygon, width, height)
    residual, warped_source = source_subtracted_residual(
        camera_frame,
        source_frame=source_frame,
        projector_polygon=projector_polygon,
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
    mask_u8 = mask.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((4, 4), dtype=np.uint8))

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    candidates: list[SourceSubtractionCandidate] = []
    for label in range(1, count):
        left, top, candidate_width, candidate_height, area = stats[label]
        if int(area) < min_area_px:
            continue
        if int(candidate_width) < 8 or int(candidate_height) < 10:
            continue
        if int(candidate_width) > int(width * 0.28) or int(candidate_height) > int(height * 0.36):
            continue
        ys, xs = np.nonzero(labels == label)
        top_xs = xs[ys == ys.min()]
        candidates.append(
            SourceSubtractionCandidate(
                bbox_xywh=(float(left), float(top), float(candidate_width), float(candidate_height)),
                top_x_px=float(top_xs.mean()),
                top_y_px=float(top),
                area_px=int(area),
            )
        )
    return sorted(candidates, key=lambda item: item.area_px, reverse=True)
