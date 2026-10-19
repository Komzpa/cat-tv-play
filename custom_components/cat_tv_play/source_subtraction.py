"""Grounded projector response and candidate-local explanation.

Residuals are usable only with a native structural correspondence and an
independently validated, candidate-excluded ten-feature photometric response.
Unavailable evidence leaves ordinary detector candidates untouched.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any


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
class AdaptiveBackground:
    """A no-cat room background with freshness metadata for review pipelines."""

    image: Image.Image
    frame_count: int
    source: str = "recent_no_cat_frames"

@dataclass(frozen=True)
class PhotometricEvidence:
    """A frozen ten-feature response with disjoint, candidate-excluded proof."""

    model_id: str
    coefficients: tuple[float, ...]
    camera_media_sha256: str
    source_media_sha256: str
    geometry_sha256: str
    fit_camera_rgb_sha256s: tuple[str, ...]
    fit_source_rgb_sha256s: tuple[str, ...]
    fit_support_mask_sha256s: tuple[str, ...]
    fit_excluded_candidate_mask_sha256s: tuple[str, ...]
    validation_camera_rgb_sha256s: tuple[str, ...]
    validation_source_rgb_sha256s: tuple[str, ...]
    validation_support_mask_sha256s: tuple[str, ...]
    validation_excluded_candidate_mask_sha256s: tuple[str, ...]
    fit_candidate_overlap_pixels: tuple[int, ...]
    validation_candidate_overlap_pixels: tuple[int, ...]
    fit_rgb_bins: tuple[int, ...]
    validation_rgb_bins: tuple[int, ...]
    fit_unclipped_luma_range: tuple[float, float]
    validation_unclipped_luma_range: tuple[float, float]
    validation_max_abs_error: float

    def valid(self) -> bool:
        fit_count = len(self.fit_camera_rgb_sha256s)
        validation_count = len(self.validation_camera_rgb_sha256s)
        fit_bins = set(self.fit_rgb_bins)
        validation_bins = set(self.validation_rgb_bins)
        return (
            self.model_id == "projector-rgb10-v1"
            and len(self.coefficients) == len(PROJECTOR_RGB_FEATURE_NAMES)
            and bool(np.isfinite(self.coefficients).all())
            and all(_is_sha256(value) for value in (self.camera_media_sha256, self.source_media_sha256, self.geometry_sha256))
            and fit_count > 0
            and validation_count > 0
            and len(self.fit_source_rgb_sha256s) == fit_count
            and len(self.fit_support_mask_sha256s) == fit_count
            and len(self.fit_excluded_candidate_mask_sha256s) == fit_count
            and len(self.validation_source_rgb_sha256s) == validation_count
            and len(self.validation_support_mask_sha256s) == validation_count
            and len(self.validation_excluded_candidate_mask_sha256s) == validation_count
            and len(self.fit_candidate_overlap_pixels) == fit_count
            and len(self.validation_candidate_overlap_pixels) == validation_count
            and all(_is_sha256(value) for values in (
                self.fit_camera_rgb_sha256s,
                self.fit_source_rgb_sha256s,
                self.fit_support_mask_sha256s,
                self.fit_excluded_candidate_mask_sha256s,
                self.validation_camera_rgb_sha256s,
                self.validation_source_rgb_sha256s,
                self.validation_support_mask_sha256s,
                self.validation_excluded_candidate_mask_sha256s,
            ) for value in values)
            and set(self.fit_camera_rgb_sha256s).isdisjoint(self.validation_camera_rgb_sha256s)
            and set(self.fit_source_rgb_sha256s).isdisjoint(self.validation_source_rgb_sha256s)
            and all(value == 0 for value in (*self.fit_candidate_overlap_pixels, *self.validation_candidate_overlap_pixels))
            and bool(fit_bins.intersection(validation_bins))
            and np.isfinite((*self.fit_unclipped_luma_range, *self.validation_unclipped_luma_range, self.validation_max_abs_error)).all()
            and 0.0 <= self.fit_unclipped_luma_range[0] <= self.fit_unclipped_luma_range[1] <= 255.0
            and 0.0 <= self.validation_unclipped_luma_range[0] <= self.validation_unclipped_luma_range[1] <= 255.0
            and self.validation_max_abs_error >= 0.0
        )

    def applies_to(self, correspondence: Any) -> bool:
        return (
            self.valid()
            and correspondence.camera_identity.media_sha256 == self.camera_media_sha256
            and correspondence.geometry_sha256 == self.geometry_sha256
            and all(
                identity.media_sha256 == self.source_media_sha256
                for family in correspondence.candidate_families
                for identity in family.source_frames
            )
            and correspondence.camera_identity.rgb_sha256 not in self.fit_camera_rgb_sha256s
            and correspondence.camera_identity.rgb_sha256 not in self.validation_camera_rgb_sha256s
        )


@dataclass(frozen=True)
class ResidualView:
    family_id: str
    source_identity: Any
    residual: np.ndarray
    warped_source_rgb: np.ndarray


@dataclass(frozen=True)
class ResidualAssessment:
    structural_status: str
    photometric_valid: bool
    views: tuple[ResidualView, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SourceExplanation:
    structural_valid: bool
    photometric_valid: bool
    explained: bool
    candidate_error: float | None
    family_errors: tuple[tuple[str, tuple[float, ...]], ...]
    reasons: tuple[str, ...]


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)




def _candidate_rgb_bins(rgb: np.ndarray, mask: np.ndarray) -> set[int]:
    pixels = np.asarray(rgb, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    if not len(pixels):
        return set()
    bins = (pixels[:, 0] >> 4).astype(np.uint16) * 256
    bins += (pixels[:, 1] >> 4).astype(np.uint16) * 16
    bins += (pixels[:, 2] >> 4).astype(np.uint16)
    return set(int(value) for value in np.unique(bins))



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


def _source_correspondence_module() -> Any:
    name = "cat_tv_play_source_correspondence"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().with_name("source_correspondence.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native correspondence module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def source_subtracted_residual(
    camera_frame: Image.Image,
    *,
    correspondence: Any,
    photometry: PhotometricEvidence | None,
    candidate_mask: np.ndarray,
) -> ResidualAssessment:
    """Evaluate one candidate against every surviving native source identity."""

    reasons: list[str] = []
    if correspondence.structural_status != "supported" or not correspondence.candidate_families:
        return ResidualAssessment(correspondence.structural_status, False, (), ("structural_correspondence_unresolved",))
    if any(family.structural_status != "supported" or not family.source_frames for family in correspondence.candidate_families):
        return ResidualAssessment(correspondence.structural_status, False, (), ("surviving_source_family_incomplete",))
    if photometry is None or not photometry.applies_to(correspondence):
        return ResidualAssessment(correspondence.structural_status, False, (), ("independent_photometric_evidence_unavailable",))

    camera_rgb = np.asarray(camera_frame.convert("RGB"), dtype=np.uint8)
    camera_identity = correspondence.camera_identity
    if camera_rgb.shape != (camera_identity.height, camera_identity.width, 3):
        return ResidualAssessment(correspondence.structural_status, False, (), ("camera_geometry_mismatch",))
    if hashlib.sha256(camera_rgb.tobytes()).hexdigest() != camera_identity.rgb_sha256:
        return ResidualAssessment(correspondence.structural_status, False, (), ("camera_rgb_identity_mismatch",))

    support_path = Path(correspondence.support_mask_path) if correspondence.support_mask_path else None
    if support_path is None or not correspondence.support_mask_sha256:
        return ResidualAssessment(correspondence.structural_status, False, (), ("camera_object_support_unavailable",))
    try:
        support_bytes = support_path.read_bytes()
        if hashlib.sha256(support_bytes).hexdigest() != correspondence.support_mask_sha256:
            raise ValueError("camera object support hash mismatch")
        with Image.open(support_path) as support_image:
            support = np.asarray(support_image.convert("L"), dtype=np.uint8) > 0
    except (OSError, ValueError) as exc:
        return ResidualAssessment(correspondence.structural_status, False, (), (f"camera_object_support_invalid:{exc}",))

    candidate = np.asarray(candidate_mask, dtype=bool)
    if candidate.shape != support.shape or candidate.shape != camera_rgb.shape[:2] or not candidate.any():
        return ResidualAssessment(correspondence.structural_status, False, (), ("candidate_mask_geometry_invalid",))
    # Structural support anchors the proposal; it is not its filled boundary.
    # Keep background, holes and possible occluders in the original candidate.
    if not np.any(candidate & support):
        return ResidualAssessment(correspondence.structural_status, False, (), ("candidate_disjoint_from_correspondence_support",))

    calibrated_bins = set(photometry.fit_rgb_bins).intersection(photometry.validation_rgb_bins)
    views: list[ResidualView] = []
    camera_gray = np.asarray(Image.fromarray(camera_rgb, mode="RGB").convert("L"), dtype=np.float32)
    coefficients = np.asarray(photometry.coefficients, dtype=np.float32)
    try:
        import cv2

        source_module = _source_correspondence_module()
        homography = np.asarray(correspondence.source_to_camera, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1e-12:
            raise ValueError("source-to-camera geometry invalid")
        covered_dimensions: set[tuple[int, int]] = set()
        for family in correspondence.candidate_families:
            for identity in family.source_frames:
                if identity.media_sha256 != photometry.source_media_sha256:
                    raise ValueError("source media differs from photometric calibration")
                source_frame = source_module.load_native_rgb(identity)
                if source_frame.size not in covered_dimensions:
                    coverage = cv2.warpPerspective(
                        np.full((source_frame.height, source_frame.width), 255, dtype=np.uint8),
                        homography,
                        (camera_identity.width, camera_identity.height),
                        flags=cv2.INTER_LINEAR,
                    )
                    if np.any(coverage[candidate] != 255):
                        raise ValueError("candidate outside native source projection coverage")
                    covered_dimensions.add(source_frame.size)
                warped_rgb = cv2.warpPerspective(
                    np.asarray(source_frame, dtype=np.uint8),
                    homography,
                    (camera_identity.width, camera_identity.height),
                    flags=cv2.INTER_LINEAR,
                )
                if not _candidate_rgb_bins(warped_rgb, candidate).issubset(calibrated_bins):
                    return ResidualAssessment(
                        correspondence.structural_status, False, (),
                        ("candidate_color_outside_independent_support",),
                    )
                expected = _predict_projector_rgb_luma(warped_rgb, coefficients)
                candidate_prediction = expected[candidate]
                if not np.isfinite(candidate_prediction).all() or np.any(candidate_prediction < 0.0) or np.any(candidate_prediction > 255.0):
                    raise ValueError("unclipped candidate prediction outside physical luma range")
                residual = expected - camera_gray
                views.append(ResidualView(family.family_id, identity, residual, warped_rgb))
    except (OSError, ValueError, RuntimeError) as exc:
        reasons.append(f"native_candidate_evaluation_failed:{exc}")
        return ResidualAssessment(correspondence.structural_status, False, (), tuple(reasons))

    return ResidualAssessment(correspondence.structural_status, True, tuple(views), ())


def explain_projected_candidate(
    *,
    camera_frame: Image.Image,
    correspondence: Any,
    candidate_mask: np.ndarray,
    photometry: PhotometricEvidence | None,
) -> SourceExplanation:
    """Explain a candidate only when every surviving native alternative agrees."""

    assessment = source_subtracted_residual(
        camera_frame,
        correspondence=correspondence,
        photometry=photometry,
        candidate_mask=candidate_mask,
    )
    structural_valid = assessment.structural_status == "supported"
    if not assessment.photometric_valid or not assessment.views or photometry is None:
        return SourceExplanation(structural_valid, False, False, None, (), assessment.reasons)

    family_errors: dict[str, list[float]] = {}
    mask = np.asarray(candidate_mask, dtype=bool)
    for view in assessment.views:
        family_errors.setdefault(view.family_id, []).append(float(np.max(np.abs(view.residual[mask]))))
    all_errors = tuple(sorted((family_id, tuple(errors)) for family_id, errors in family_errors.items()))
    candidate_error = max(error for errors in family_errors.values() for error in errors)
    explained = (
        set(family_errors) == {family.family_id for family in correspondence.candidate_families}
        and all(error <= photometry.validation_max_abs_error for errors in family_errors.values() for error in errors)
    )
    reasons = () if explained else ("candidate_residual_exceeds_disjoint_validation_error", "not_explained_across_all_surviving_source_candidates")
    return SourceExplanation(structural_valid, True, explained, candidate_error, all_errors, reasons)
