"""Shared Cat TV publish-render framing and presentation helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECTOR_POLYGON_1280 = (
    (40.22, 57.27),
    (937.92, 101.0),
    (908.0, 599.0),
    (48.16, 680.97),
)


def bbox_union(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Return the union of two xyxy bounding boxes."""

    if first is None:
        return second
    if second is None:
        return first
    left, top, right, bottom = first
    next_left, next_top, next_right, next_bottom = second
    return (
        min(left, next_left),
        min(top, next_top),
        max(right, next_right),
        max(bottom, next_bottom),
    )


def bbox_for_points(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    """Return xyxy bounds for a point iterable."""

    point_list = list(points)
    if not point_list:
        return None
    return (
        min(point[0] for point in point_list),
        min(point[1] for point in point_list),
        max(point[0] for point in point_list),
        max(point[1] for point in point_list),
    )


def projector_screen_bbox(image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return the projector screen bounds scaled to an image size."""

    image_width, image_height = image_size
    scale_x = image_width / 1280.0
    scale_y = image_height / 720.0
    return bbox_for_points((x * scale_x, y * scale_y) for x, y in PROJECTOR_POLYGON_1280) or (
        0.0,
        0.0,
        float(image_width),
        float(image_height),
    )


def content_bbox(
    *,
    image_size: tuple[int, int],
    cat_boxes: Iterable[tuple[float, float, float, float] | None] = (),
    cat_polygons: Iterable[Iterable[tuple[float, float]]] = (),
) -> tuple[float, float, float, float]:
    """Return publish content bounds: projector screen plus all cat geometry."""

    bounds: tuple[float, float, float, float] | None = projector_screen_bbox(image_size)
    for box in cat_boxes:
        bounds = bbox_union(bounds, box)
    for polygon in cat_polygons:
        bounds = bbox_union(bounds, bbox_for_points(polygon))
    assert bounds is not None
    return bounds


def close_crop_bounds(
    *,
    image_size: tuple[int, int],
    content_bounds: tuple[float, float, float, float] | None,
    output_size: tuple[int, int] = (1080, 1920),
) -> tuple[float, float, float, float] | None:
    """Fit content bounds into a fixed 9:16 crop, allowing context padding outside the source frame."""

    if content_bounds is None:
        return None
    image_width, image_height = image_size
    left, top, right, bottom = content_bounds
    if right <= left or bottom <= top:
        return None
    pad_x = max(20.0, (right - left) * 0.025)
    pad_y = max(20.0, (bottom - top) * 0.035)
    left -= pad_x
    right += pad_x
    top -= pad_y
    bottom += pad_y
    crop_width = right - left
    crop_height = bottom - top
    aspect = output_size[0] / float(output_size[1])
    if crop_width / crop_height > aspect:
        crop_height = crop_width / aspect
    else:
        crop_width = crop_height * aspect
    crop_width = min(float(image_width), max(1.0, crop_width))
    crop_height = max(500.0, crop_height, crop_width / aspect)
    crop_width = crop_height * aspect
    if crop_width > image_width:
        crop_width = float(image_width)
        crop_height = crop_width / aspect
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_left = center_x - crop_width / 2.0
    crop_top = center_y - crop_height / 2.0
    if crop_width <= image_width:
        crop_left = max(0.0, min(float(image_width) - crop_width, crop_left))
    if crop_height <= image_height:
        crop_top = max(0.0, min(float(image_height) - crop_height, crop_top))
    return crop_left, crop_top, crop_width, crop_height


def recording_datetime(*, label: str | None = None, recording_dir: str | Path | None = None) -> datetime | None:
    """Extract the first YYYYMMDDTHHMMSS timestamp from publish labels/paths."""

    labels = [str(label or "")]
    if recording_dir is not None:
        labels.append(Path(recording_dir).name)
    for value in labels:
        match = re.search(r"(\d{8}T\d{6})", value)
        if match is None:
            continue
        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        except ValueError:
            continue
    return None


def recording_date_label(*, label: str | None = None, recording_dir: str | Path | None = None) -> str | None:
    """Return a viewer-friendly publish date label."""

    recorded_at = recording_datetime(label=label, recording_dir=recording_dir)
    if recorded_at is None:
        return None
    return f"{recorded_at:%A}, {recorded_at.day} {recorded_at:%B %Y}"


def presentation_metadata(
    *,
    subject: str = "Sher",
    label: str | None = None,
    recording_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return stable text metadata for social/publish overlays."""

    title = f"{subject.upper()} JUMPS"
    return {
        "subject": subject,
        "title": title,
        "recorded_on": recording_date_label(label=label, recording_dir=recording_dir),
    }
