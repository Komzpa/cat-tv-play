"""Shared Cat TV publish-render framing and presentation helpers."""

from __future__ import annotations

import math
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
    """Return whole-trajectory cat bounds, without forcing the projector into the close shot."""

    bounds: tuple[float, float, float, float] | None = None
    for box in cat_boxes:
        bounds = bbox_union(bounds, box)
    for polygon in cat_polygons:
        bounds = bbox_union(bounds, bbox_for_points(polygon))
    return bounds or (0.0, 0.0, float(image_size[0]), float(image_size[1]))


def close_crop_bounds(
    *,
    image_size: tuple[int, int],
    content_bounds: tuple[float, float, float, float] | None,
    output_size: tuple[int, int] = (972, 1170),
) -> tuple[float, float, float, float] | None:
    """Fit a complete trajectory into the video stage, retaining source-edge padding."""

    if content_bounds is None:
        return None
    left, top, right, bottom = content_bounds
    if right <= left or bottom <= top:
        return None
    # Leave at least 24 stage pixels around source-supported geometry.
    pad_x = max(20.0, (right - left) * 24.0 / max(1, output_size[0] - 48))
    pad_y = max(20.0, (bottom - top) * 24.0 / max(1, output_size[1] - 48))
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
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_left = center_x - crop_width / 2.0
    crop_top = center_y - crop_height / 2.0
    return crop_left, crop_top, crop_width, crop_height


def video_stage(output_size: tuple[int, int] = (1080, 1920)) -> tuple[int, int, int, int]:
    """Return the fixed x/y/width/height stage below the image cards."""

    width, height = output_size
    return (round(54 * width / 1080), round(390 * height / 1920),
            round(972 * width / 1080), round(1170 * height / 1920))


def camera_crop(
    open_bounds: tuple[float, float, float, float],
    close_bounds: tuple[float, float, float, float],
    progress: float,
) -> tuple[float, float, float, float]:
    """Interpolate one nested crop; never recenter on individual cat frames."""

    return tuple(start + (end - start) * progress for start, end in zip(open_bounds, close_bounds))


def trajectory_camera_plan(
    *,
    image_size: tuple[int, int],
    observations: list[tuple[float, tuple[float, float, float, float] | None]],
    content_bounds: tuple[float, float, float, float],
    output_size: tuple[int, int] = (1080, 1920),
) -> dict[str, Any]:
    """Plan a room introduction, one source-time ease and one fixed landing crop."""

    stage = video_stage(output_size)
    close = close_crop_bounds(image_size=image_size, content_bounds=content_bounds, output_size=stage[2:])
    if close is None:
        raise ValueError("a trajectory camera requires source-supported geometry")
    room = bbox_union((0.0, 0.0, float(image_size[0]), float(image_size[1])),
                      (close[0], close[1], close[0] + close[2], close[1] + close[3]))
    assert room is not None
    room_width, room_height = room[2] - room[0], room[3] - room[1]
    aspect = stage[2] / stage[3]
    room_width = max(room_width, room_height * aspect)
    room_height = room_width / aspect
    opened = ((room[0] + room[2] - room_width) / 2.0,
              (room[1] + room[3] - room_height) / 2.0, room_width, room_height)
    first_time = observations[0][0] if observations else 0.0
    motion_onset = None
    previous_top = None
    for timestamp, bounds in observations:
        if bounds is None:
            continue
        if previous_top is not None and bounds[1] < previous_top - 8.0:
            motion_onset = timestamp - first_time
            break
        previous_top = bounds[1]
    zoom_start = max(0.5, motion_onset) if motion_onset is not None else None
    frames = []
    progress = 0.0
    previous_crop = opened
    for timestamp, _bounds in observations:
        elapsed = timestamp - first_time
        fraction = max(0.0, min(1.0, (elapsed - zoom_start) / 1.2)) if zoom_start is not None else 0.0
        desired = fraction * fraction * (3.0 - 2.0 * fraction)
        lower, upper = progress, max(progress, desired)
        # Sparse original PTS must not turn an eased move into a one-frame cut.
        for _ in range(24):
            candidate_progress = (lower + upper) / 2.0
            candidate = camera_crop(opened, close, candidate_progress)
            log_scale_step = abs(math.log(previous_crop[2] / candidate[2]))
            center_step = math.hypot(
                candidate[0] + candidate[2] / 2.0 - previous_crop[0] - previous_crop[2] / 2.0,
                candidate[1] + candidate[3] / 2.0 - previous_crop[1] - previous_crop[3] / 2.0,
            ) * stage[2] / candidate[2]
            if log_scale_step <= 0.12 and center_step <= stage[2] * 0.05:
                lower = candidate_progress
            else:
                upper = candidate_progress
        progress = desired if upper - desired >= -1e-7 else lower
        previous_crop = camera_crop(opened, close, progress)
        frames.append({"source_pts_seconds": timestamp, "elapsed_seconds": elapsed,
                       "zoom_progress": progress, "crop_bounds": list(previous_crop)})
    return {"stage_xywh": list(stage), "open_crop_bounds": list(opened),
            "close_crop_bounds": list(close), "motion_onset_seconds": motion_onset,
            "zoom_start_seconds": zoom_start, "ease_seconds": 1.2,
            "timing_policy": "source_pts_ease_with_adjacent_frame_scale_and_center_limits",
            "frames": frames}


def match_review_frames(
    *,
    source_frames: Iterable[Path],
    review_frames: Iterable[Path],
    cat_bounds_by_review: dict[str, tuple[float, float, float, float] | None],
) -> dict[str, dict[str, Any]]:
    """Associate cached annotations by scene and cat pixels, never filename ordinals.

    JPEG re-encoding is tolerated, but unrelated poses are not reused. Missing
    associations are left absent so the renderer can report them explicitly.
    """

    from PIL import Image, ImageChops, ImageStat

    references = []
    for path in review_frames:
        if not path.is_file():
            continue
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        bounds = cat_bounds_by_review.get(str(path))
        patch = image.crop(bounds).resize((64, 64), Image.Resampling.BILINEAR) if bounds else None
        references.append((path, image.size, image.resize((128, 72), Image.Resampling.BILINEAR), bounds, patch))
    matches = {}
    for source_path in source_frames:
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        scene = source.resize((128, 72), Image.Resampling.BILINEAR)
        scene_scores = []
        for reference in references:
            if reference[1] != source.size:
                continue
            difference = ImageChops.difference(scene, reference[2])
            scene_mae = sum(ImageStat.Stat(difference).mean) / 3.0
            if scene_mae <= 2.0:
                scene_scores.append((scene_mae, reference))
        best = None
        for scene_mae, reference in sorted(scene_scores, key=lambda item: item[0])[:12]:
            path, _size, _scene, bounds, reference_patch = reference
            if bounds is not None:
                patch = source.crop(bounds).resize((64, 64), Image.Resampling.BILINEAR)
                patch_mae = sum(ImageStat.Stat(ImageChops.difference(patch, reference_patch)).mean) / 3.0
                if patch_mae > 6.0:
                    continue
            else:
                patch_mae = scene_mae
            score = scene_mae + patch_mae / 4.0
            if best is None or score < best[0]:
                best = (score, path, scene_mae, patch_mae)
        if best is not None:
            matches[str(source_path)] = {
                "review_frame": str(best[1]), "scene_mae_255": best[2],
                "cat_patch_mae_255": best[3], "match_score": best[0],
                "method": "nearest_scene_and_cat_patch_pixels",
            }
    return matches


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
    months = ("января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря")
    return f"{recorded_at.day} {months[recorded_at.month - 1]} {recorded_at.year}"


def presentation_metadata(
    *,
    subject: str = "Шер",
    label: str | None = None,
    recording_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return stable text metadata for social/publish overlays."""

    recorded_on = recording_date_label(label=label, recording_dir=recording_dir)
    title = f"{subject} · {recorded_on}" if recorded_on else subject
    return {
        "subject": subject,
        "title": title,
        "recorded_on": recorded_on,
    }
