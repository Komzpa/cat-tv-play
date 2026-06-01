"""Mask-first cat jump measurement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

BBoxXYWH = tuple[float, float, float, float]


@dataclass(frozen=True)
class MeasurementPoint:
    point_type: str
    image_x: float
    image_y: float
    wall_x_cm: float | None = None
    wall_y_cm: float | None = None
    confidence: float = 0.0
    uncertainty_px: float | None = None
    source: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JumpMeasurement:
    event_id: str
    frame_index: int | None
    timestamp_seconds: float | None
    provisional_peak_cm: float | None
    source_point_type: str
    confidence: float
    trust_flags: tuple[str, ...] = ()
    debug: dict[str, Any] = field(default_factory=dict)


def mask_top_measurement_point(
    mask: np.ndarray,
    *,
    score: float,
    top_fraction: float = 0.05,
    source: str = "mask_top_p5",
    min_area_px: int = 64,
    min_width_px: int = 4,
    min_height_px: int = 4,
    max_fill_fraction: float = 1.01,
) -> MeasurementPoint | None:
    """Return a robust top-of-mask point.

    Image y grows downward, so the "top" is a low y value. Instead of the single
    minimum y pixel, use a small top band and take its median x plus a y
    quantile. This resists whisker/noise pixels and jagged masks.
    """

    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or not mask_bool.any():
        return None
    labels, component_count = ndimage.label(mask_bool)
    if component_count <= 0:
        return None
    component_ids, component_areas = np.unique(labels[labels > 0], return_counts=True)
    largest_component = int(component_ids[int(np.argmax(component_areas))])
    component = labels == largest_component
    area_px = int(component.sum())
    if area_px < min_area_px:
        return None
    component_ys, component_xs = np.nonzero(component)
    width_px = int(component_xs.max() - component_xs.min() + 1)
    height_px = int(component_ys.max() - component_ys.min() + 1)
    if width_px < min_width_px or height_px < min_height_px:
        return None
    fill_fraction = area_px / float(width_px * height_px)
    if fill_fraction > max_fill_fraction:
        return None
    ys, xs = component_ys, component_xs
    quantile = max(0.0, min(0.49, float(top_fraction)))
    top_y = float(np.quantile(ys.astype(float), quantile))
    band = ys <= int(np.ceil(top_y))
    if not np.any(band):
        band = ys == ys.min()
    band_xs = xs[band]
    band_ys = ys[band]
    return MeasurementPoint(
        point_type=source,
        image_x=float(np.median(band_xs.astype(float))),
        image_y=top_y,
        confidence=float(max(0.0, min(1.0, score))),
        uncertainty_px=float(max(1.0, np.std(band_ys.astype(float)) if len(band_ys) > 1 else 1.0)),
        source="segmentation_mask",
        debug={
            "mask_area_px": int(mask_bool.sum()),
            "selected_component_area_px": area_px,
            "selected_component_bbox_xywh": [
                int(component_xs.min()),
                int(component_ys.min()),
                width_px,
                height_px,
            ],
            "component_count": int(component_count),
            "discarded_component_area_px": int(mask_bool.sum()) - area_px,
            "fill_fraction": round(float(fill_fraction), 4),
            "top_fraction": quantile,
            "band_pixel_count": int(len(band_xs)),
        },
    )


def legacy_bbox_top_measurement_point(
    bbox_xywh: BBoxXYWH,
    *,
    score: float,
    source: str = "legacy_bbox_top",
) -> MeasurementPoint:
    x, y, width, _height = bbox_xywh
    return MeasurementPoint(
        point_type=source,
        image_x=float(x + width / 2.0),
        image_y=float(y),
        confidence=float(max(0.0, min(0.45, score * 0.5))),
        uncertainty_px=max(4.0, float(width) * 0.15),
        source="legacy_bbox",
        debug={"bbox_xywh": [float(value) for value in bbox_xywh], "low_trust_fallback": True},
    )


def transform_image_point(homography: tuple[float, ...], image_x: float, image_y: float) -> tuple[float, float]:
    if len(homography) != 8:
        raise ValueError("homography must contain exactly eight coefficients")
    h0, h1, h2, h3, h4, h5, h6, h7 = homography
    denominator = h6 * image_x + h7 * image_y + 1.0
    if abs(denominator) < 1e-9:
        raise ValueError("image point maps to infinity for this calibration")
    wall_x = (h0 * image_x + h1 * image_y + h2) / denominator
    wall_y = (h3 * image_x + h4 * image_y + h5) / denominator
    return (float(wall_x), float(wall_y))


def with_wall_coordinates(point: MeasurementPoint, homography: tuple[float, ...]) -> MeasurementPoint:
    wall_x, wall_y = transform_image_point(homography, point.image_x, point.image_y)
    return MeasurementPoint(
        point_type=point.point_type,
        image_x=point.image_x,
        image_y=point.image_y,
        wall_x_cm=wall_x,
        wall_y_cm=wall_y,
        confidence=point.confidence,
        uncertainty_px=point.uncertainty_px,
        source=point.source,
        debug=point.debug,
    )


def measurement_point_to_dict(point: MeasurementPoint | None) -> dict[str, Any] | None:
    if point is None:
        return None
    return {
        "point_type": point.point_type,
        "image_x": round(float(point.image_x), 2),
        "image_y": round(float(point.image_y), 2),
        "wall_x_cm": None if point.wall_x_cm is None else round(float(point.wall_x_cm), 2),
        "wall_y_cm": None if point.wall_y_cm is None else round(float(point.wall_y_cm), 2),
        "confidence": round(float(point.confidence), 4),
        "uncertainty_px": None if point.uncertainty_px is None else round(float(point.uncertainty_px), 2),
        "source": point.source,
        "debug": point.debug,
    }


def measurement_point_from_dict(raw: dict[str, Any] | None) -> MeasurementPoint | None:
    if not isinstance(raw, dict):
        return None
    return MeasurementPoint(
        point_type=str(raw.get("point_type") or "unknown"),
        image_x=float(raw["image_x"]),
        image_y=float(raw["image_y"]),
        wall_x_cm=None if raw.get("wall_x_cm") is None else float(raw["wall_x_cm"]),
        wall_y_cm=None if raw.get("wall_y_cm") is None else float(raw["wall_y_cm"]),
        confidence=float(raw.get("confidence") or 0.0),
        uncertainty_px=None if raw.get("uncertainty_px") is None else float(raw["uncertainty_px"]),
        source=str(raw.get("source") or ""),
        debug=raw.get("debug") if isinstance(raw.get("debug"), dict) else {},
    )
