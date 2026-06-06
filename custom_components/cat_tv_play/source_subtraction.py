"""Projector source subtraction helpers.

When the clip shown on the projector is known, projected prey should not be
classified as a cat candidate. These helpers warp the expected source frame into
the camera image, subtract it from the camera frame, and return remaining dark
residual components for a downstream cat/not-cat detector or human review.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

PROJECTOR_RGB_FEATURE_NAMES = (
    "red",
    "green",
    "blue",
    "channel_max",
    "channel_min",
    "channel_mean",
    "srgb_luma",
    "saturation",
    "sqrt_channel_max",
    "bias",
)


@dataclass(frozen=True)
class SourceSubtractionCandidate:
    """A residual blob candidate in camera pixel coordinates."""

    bbox_xywh: tuple[float, float, float, float]
    top_x_px: float
    top_y_px: float
    area_px: int
    source: str = "source_subtracted_projector_residual"
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class AdaptiveBackground:
    """A no-cat room background with freshness metadata for review pipelines."""

    image: Image.Image
    frame_count: int
    source: str = "recent_no_cat_frames"


def robust_component_top_px(
    labels: np.ndarray,
    label: int,
    *,
    top_percentile: float = 3.0,
    cap_height_px: int = 5,
    min_cap_pixels: int = 8,
) -> tuple[float, float] | None:
    """Return a stable top point for a connected component.

    A single isolated high pixel must not become a fake jump apex. Use a thin
    upper cap of the component and report its median point instead.
    """

    ys, xs = np.nonzero(labels == label)
    if len(ys) < min_cap_pixels:
        return None

    robust_top_y = float(np.percentile(ys, top_percentile))
    cap = ys <= robust_top_y + cap_height_px
    if int(cap.sum()) < min_cap_pixels:
        return None

    return float(np.median(xs[cap])), float(np.median(ys[cap]))


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


def _normalize_update_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    normalized = np.asarray(mask, dtype=bool)
    if normalized.shape != shape:
        raise ValueError(f"background update mask shape {normalized.shape} does not match frame shape {shape}")
    return normalized


def update_room_background(
    previous: Image.Image,
    frame: Image.Image,
    *,
    update_mask: np.ndarray,
    alpha: float = 0.08,
) -> Image.Image:
    """Update background only where the cat cannot plausibly be.

    `update_mask=True` means this pixel is allowed to learn from the current
    frame. Keep all pixels on the possible cat path set to False.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    previous_arr = np.asarray(previous.convert("L").resize(frame.size), dtype=np.float32)
    frame_arr = np.asarray(frame.convert("L"), dtype=np.float32)
    mask = _normalize_update_mask(update_mask, frame_arr.shape)
    updated = previous_arr.copy()
    updated[mask] = previous_arr[mask] * (1.0 - alpha) + frame_arr[mask] * alpha
    return Image.fromarray(np.clip(updated, 0, 255).astype(np.uint8), mode="L")


def build_room_background(
    frames_without_cat: Iterable[Image.Image],
    *,
    percentile: float = 50.0,
    source: str = "recent_no_cat_frames",
    update_mask: np.ndarray | None = None,
) -> AdaptiveBackground:
    """Build a fresh room background from frames known not to contain the cat.

    Do not pass arbitrary active-session frames here. A cat that sits still for
    long enough becomes part of a median/percentile background and disappears
    from foreground detection. The input frames must be selected by a no-cat
    gate, a human-reviewed no-cat segment, or a dedicated empty-room snapshot.
    Because projector-camera IR exposure changes with scene brightness, callers
    should refresh this background from recent no-cat frames or update it with a
    mask that excludes every pixel where the cat could plausibly sit or jump.
    """

    frames = [np.asarray(frame.convert("L"), dtype=np.uint8) for frame in frames_without_cat]
    if not frames:
        raise ValueError("at least one no-cat frame is required to build a room background")
    first_shape = frames[0].shape
    if any(frame.shape != first_shape for frame in frames):
        raise ValueError("all no-cat background frames must have the same size")
    if update_mask is not None:
        mask = _normalize_update_mask(update_mask, first_shape)
        if not mask.any():
            raise ValueError("background update mask excludes every pixel")
        stacked = np.stack(frames, axis=0)
        seed = frames[0].astype(np.float32)
        learned = np.percentile(stacked[:, mask], percentile, axis=0)
        seed[mask] = learned
        image = Image.fromarray(np.clip(seed, 0, 255).astype(np.uint8), mode="L")
        return AdaptiveBackground(image=image, frame_count=len(frames), source=source)
    background = np.percentile(np.stack(frames, axis=0), percentile, axis=0)
    image = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8), mode="L")
    return AdaptiveBackground(image=image, frame_count=len(frames), source=source)


def match_background_luma(background: Image.Image, camera_frame: Image.Image, mask: np.ndarray) -> Image.Image:
    """Match a no-cat background to the current IR brightness."""

    background_arr = np.asarray(background.convert("L").resize(camera_frame.size), dtype=np.uint8)
    camera_arr = np.asarray(camera_frame.convert("L"), dtype=np.uint8)
    scale, offset = _robust_linear_match(background_arr, camera_arr, mask)
    matched = np.clip(background_arr.astype(np.float32) * scale + offset, 0, 255)
    return Image.fromarray(matched.astype(np.uint8), mode="L")


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


def warp_source_rgb_to_camera(
    source_frame: Image.Image,
    *,
    projector_polygon: Iterable[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Warp a source RGB frame into camera pixels.

    The projector camera does not see the wall like a normal RGB viewer. Keep
    the RGB channels until the current camera frame can fit its own
    projector-color-to-camera-luma mapping.
    """

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on optional review tooling.
        raise RuntimeError("opencv-python is required for projector source subtraction") from exc

    source_rgb = np.asarray(source_frame.convert("RGB").resize((1280, 720)), dtype=np.uint8)
    source_points = np.float32([[0, 0], [1279, 0], [1279, 719], [0, 719]])
    camera_points = np.float32(tuple(projector_polygon))
    homography = cv2.getPerspectiveTransform(source_points, camera_points)
    return cv2.warpPerspective(source_rgb, homography, (width, height))


def _fit_projector_rgb_to_camera_luma(
    warped_rgb: np.ndarray,
    camera_gray: np.ndarray,
    fit_mask: np.ndarray,
    *,
    max_fit_pixels: int = 3000,
) -> np.ndarray | None:
    rgb = warped_rgb.astype(np.float32)
    camera = camera_gray.astype(np.float32)
    source_signal = rgb.max(axis=2)
    keep = fit_mask & (source_signal > 35) & (camera > 25)
    if int(keep.sum()) < 500:
        return None
    if int(keep.sum()) > max_fit_pixels:
        keep = _subsample_mask(keep, max_pixels=max_fit_pixels)

    features = _projector_rgb_feature_matrix(rgb, keep)
    target = camera[keep]
    coefficients = _ridge_fit_projector_luma(features, target)
    prediction = features @ coefficients
    residual = np.abs(prediction - target)
    cutoff = max(8.0, float(np.percentile(residual, 70)))
    robust_keep = residual <= cutoff
    if int(robust_keep.sum()) < 500:
        return _regularized_projector_coefficients(features, target, coefficients)
    coefficients = _ridge_fit_projector_luma(features[robust_keep], target[robust_keep])
    return _regularized_projector_coefficients(features[robust_keep], target[robust_keep], coefficients)


def _projector_rgb_features(rgb: np.ndarray) -> np.ndarray:
    """Return projector-color features used to model the camera's mono/IR view."""

    rgb_f = rgb.astype(np.float32, copy=False)
    red = rgb_f[..., 0]
    green = rgb_f[..., 1]
    blue = rgb_f[..., 2]
    channel_max = rgb_f.max(axis=2)
    channel_min = rgb_f.min(axis=2)
    channel_mean = rgb_f.mean(axis=2)
    srgb_luma = 0.299 * red + 0.587 * green + 0.114 * blue
    saturation = channel_max - channel_min
    sqrt_channel_max = np.sqrt(np.clip(channel_max, 0, 255) / 255.0) * 255.0
    bias = np.ones(red.shape, dtype=np.float32)
    return np.stack(
        [
            red,
            green,
            blue,
            channel_max,
            channel_min,
            channel_mean,
            srgb_luma,
            saturation,
            sqrt_channel_max,
            bias,
        ],
        axis=2,
    )


def _projector_rgb_feature_matrix(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return selected projector-color feature rows without allocating a full feature image."""

    rgb_f = rgb.astype(np.float32, copy=False)
    red = rgb_f[..., 0][mask]
    green = rgb_f[..., 1][mask]
    blue = rgb_f[..., 2][mask]
    channel_max = np.maximum(np.maximum(red, green), blue)
    channel_min = np.minimum(np.minimum(red, green), blue)
    channel_mean = (red + green + blue) / 3.0
    srgb_luma = 0.299 * red + 0.587 * green + 0.114 * blue
    saturation = channel_max - channel_min
    sqrt_channel_max = np.sqrt(np.clip(channel_max, 0, 255) / 255.0) * 255.0
    bias = np.ones(red.shape, dtype=np.float32)
    return np.column_stack(
        [
            red,
            green,
            blue,
            channel_max,
            channel_min,
            channel_mean,
            srgb_luma,
            saturation,
            sqrt_channel_max,
            bias,
        ]
    )


def _predict_projector_rgb_luma(rgb: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32, copy=False)
    red = rgb_f[..., 0]
    green = rgb_f[..., 1]
    blue = rgb_f[..., 2]
    channel_max = np.maximum(np.maximum(red, green), blue)
    channel_min = np.minimum(np.minimum(red, green), blue)
    channel_mean = (red + green + blue) / 3.0
    srgb_luma = 0.299 * red + 0.587 * green + 0.114 * blue
    saturation = channel_max - channel_min
    sqrt_channel_max = np.sqrt(np.clip(channel_max, 0, 255) / 255.0) * 255.0
    return (
        coefficients[0] * red
        + coefficients[1] * green
        + coefficients[2] * blue
        + coefficients[3] * channel_max
        + coefficients[4] * channel_min
        + coefficients[5] * channel_mean
        + coefficients[6] * srgb_luma
        + coefficients[7] * saturation
        + coefficients[8] * sqrt_channel_max
        + coefficients[9]
    )


def _subsample_mask(mask: np.ndarray, *, max_pixels: int) -> np.ndarray:
    selected = np.zeros(mask.shape, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(ys) <= max_pixels:
        selected[ys, xs] = True
        return selected
    step = max(1, int(np.floor(len(ys) / max_pixels)))
    selected[ys[::step][:max_pixels], xs[::step][:max_pixels]] = True
    return selected


def _ridge_fit_projector_luma(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit projector colors to camera luma without chasing correlated colors."""

    feature_count = features.shape[1] - 1
    feature_scale = np.maximum(np.std(features[:, :feature_count], axis=0), 1.0)
    centered = features.copy()
    centered[:, :feature_count] = centered[:, :feature_count] / feature_scale
    ridge_weights = np.full(features.shape[1], 10.0, dtype=np.float32)
    ridge_weights[-1] = 0.0
    ridge = np.diag(ridge_weights)
    try:
        coefficients = np.linalg.solve(centered.T @ centered + ridge, centered.T @ target)
    except np.linalg.LinAlgError:
        coefficients, *_rest = np.linalg.lstsq(centered, target, rcond=None)
    coefficients[:feature_count] = coefficients[:feature_count] / feature_scale
    return coefficients.astype(np.float32)


def _regularized_projector_coefficients(
    features: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Keep the fitted camera response bounded without breaking IR-like fits."""

    raw = coefficients.astype(np.float32, copy=True)
    raw[:-1] = np.clip(raw[:-1], -4.0, 4.0)
    raw[-1] = float(np.median(target - features[:, :-1] @ raw[:-1]))
    if float(np.abs(raw[:-1]).sum()) < 0.02:
        return _fit_grayscale_projector_coefficients(features, target)
    return raw.astype(np.float32)


def _fit_grayscale_projector_coefficients(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    luma_index = PROJECTOR_RGB_FEATURE_NAMES.index("srgb_luma")
    luma = features[:, luma_index]
    source_low, source_high = np.percentile(luma, [10, 90])
    target_low, target_high = np.percentile(target, [10, 90])
    scale = max(0.02, min(4.0, float(target_high - target_low) / max(1.0, float(source_high - source_low))))
    offset = float(np.median(target - scale * luma))
    coefficients = np.zeros(features.shape[1], dtype=np.float32)
    coefficients[luma_index] = scale
    coefficients[-1] = offset
    return coefficients


def _projector_rgb_expected_luma(
    warped_rgb: np.ndarray,
    camera_gray: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    coefficients = _fit_projector_rgb_to_camera_luma(warped_rgb, camera_gray, fit_mask)
    warped_luma = np.asarray(Image.fromarray(warped_rgb, mode="RGB").convert("L"), dtype=np.uint8)
    if coefficients is None:
        scale, offset = _robust_linear_match(warped_luma, camera_gray, fit_mask)
        return np.clip(scale * warped_luma.astype(np.float32) + offset, 0, 255)

    expected = _predict_projector_rgb_luma(warped_rgb, coefficients)
    return np.clip(expected, 0, 255)


def source_subtracted_residual(
    camera_frame: Image.Image,
    *,
    source_frame: Image.Image,
    projector_polygon: Iterable[tuple[float, float]],
    room_background: Image.Image | AdaptiveBackground | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dark residual after explaining the projector plane by source video."""

    camera_gray = np.asarray(camera_frame.convert("L"), dtype=np.uint8)
    height, width = camera_gray.shape
    projector_polygon = tuple(projector_polygon)
    projection = projection_mask(projector_polygon, width, height)
    warped_source_rgb = warp_source_rgb_to_camera(
        source_frame,
        projector_polygon=projector_polygon,
        width=width,
        height=height,
    )
    warped_source = np.asarray(Image.fromarray(warped_source_rgb, mode="RGB").convert("L"), dtype=np.uint8)
    if room_background is None:
        expected = camera_gray.astype(np.float32)
    else:
        background_image = room_background.image if isinstance(room_background, AdaptiveBackground) else room_background
        matched_background = match_background_luma(background_image, camera_frame, ~projection)
        expected = np.asarray(
            matched_background.convert("L").resize((width, height)),
            dtype=np.uint8,
        ).astype(np.float32)

    fit_mask = projection.copy()
    fit_mask[int(height * 0.69) :, :] = False
    fit_mask[:, int(width * 0.82) :] = False
    mapped_source = _projector_rgb_expected_luma(warped_source_rgb, camera_gray, fit_mask)
    expected[projection] = mapped_source[projection]
    return expected - camera_gray.astype(np.float32), warped_source


def detect_source_subtracted_candidates(
    camera_frame: Image.Image,
    *,
    source_frame: Image.Image,
    projector_polygon: Iterable[tuple[float, float]],
    room_background: Image.Image | AdaptiveBackground | None = None,
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
        top_point = robust_component_top_px(labels, label)
        if top_point is None:
            continue
        top_x_px, top_y_px = top_point
        component_mask = labels == label
        candidates.append(
            SourceSubtractionCandidate(
                bbox_xywh=(float(left), float(top), float(candidate_width), float(candidate_height)),
                top_x_px=top_x_px,
                top_y_px=top_y_px,
                area_px=int(area),
                mask=component_mask.copy(),
            )
        )
    return sorted(candidates, key=lambda item: item.area_px, reverse=True)
