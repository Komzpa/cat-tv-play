import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

MODULE_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "cat_tv_play" / "source_subtraction.py"
SPEC = spec_from_file_location("cat_tv_play_source_subtraction", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
source_subtraction = module_from_spec(SPEC)
sys.modules[SPEC.name] = source_subtraction
SPEC.loader.exec_module(source_subtraction)

PROJECTOR_POLYGON = (
    (40.22, 57.27),
    (937.92, 101.0),
    (908.0, 599.0),
    (48.16, 680.97),
)


def _camera_room_with_projection(source: Image.Image, *, room_luma: int = 95) -> Image.Image:
    warped = source_subtraction.warp_source_to_camera(
        source,
        projector_polygon=PROJECTOR_POLYGON,
        width=1280,
        height=720,
    )
    camera_arr = np.full((720, 1280), room_luma, dtype=np.uint8)
    mask = source_subtraction.projection_mask(PROJECTOR_POLYGON, 1280, 720)
    camera_arr[mask] = warped[mask]
    return Image.fromarray(camera_arr).convert("RGB")


def test_source_subtracted_candidates_ignore_projected_prey_and_keep_cat() -> None:
    cv2 = pytest.importorskip("cv2")
    source = Image.new("RGB", (1280, 720), "white")
    ImageDraw.Draw(source).ellipse((500, 220, 570, 280), fill="black")
    warped = source_subtraction.warp_source_to_camera(
        source,
        projector_polygon=PROJECTOR_POLYGON,
        width=1280,
        height=720,
    )
    camera = Image.fromarray(warped).convert("RGB")
    ImageDraw.Draw(camera).rectangle((620, 430, 700, 620), fill="black")

    candidates = source_subtraction.detect_source_subtracted_candidates(
        camera,
        source_frame=source,
        projector_polygon=PROJECTOR_POLYGON,
    )

    assert candidates
    best = candidates[0]
    x, y, width, height = best.bbox_xywh
    assert x <= 625 <= x + width
    assert y <= 435 <= y + height
    assert all(not (390 <= candidate.bbox_xywh[0] <= 470 and candidate.bbox_xywh[1] < 320) for candidate in candidates)
    assert cv2 is not None


def test_source_subtracted_candidates_accept_residual_baseline() -> None:
    pytest.importorskip("cv2")
    source = Image.new("RGB", (1280, 720), "white")
    camera = Image.fromarray(
        source_subtraction.warp_source_to_camera(
            source,
            projector_polygon=PROJECTOR_POLYGON,
            width=1280,
            height=720,
        )
    ).convert("RGB")
    ImageDraw.Draw(camera).rectangle((620, 430, 700, 620), fill="black")
    residual, _warped = source_subtraction.source_subtracted_residual(
        camera,
        source_frame=source,
        projector_polygon=PROJECTOR_POLYGON,
    )
    baseline = residual.copy()
    baseline[430:620, 620:700] = 0

    candidates = source_subtraction.detect_source_subtracted_candidates(
        camera,
        source_frame=source,
        projector_polygon=PROJECTOR_POLYGON,
        residual_baseline=baseline,
    )

    assert candidates
    x, y, width, height = candidates[0].bbox_xywh
    assert x <= 625 <= x + width
    assert y <= 435 <= y + height


def test_room_background_from_no_cat_frame_keeps_static_sitting_cat() -> None:
    pytest.importorskip("cv2")
    source = Image.new("RGB", (1280, 720), "white")
    no_cat_camera = _camera_room_with_projection(source)
    camera_with_cat = no_cat_camera.copy()
    ImageDraw.Draw(camera_with_cat).rectangle((585, 650, 675, 715), fill="black")
    good_background = source_subtraction.build_room_background([no_cat_camera])

    good_candidates = source_subtraction.detect_source_subtracted_candidates(
        camera_with_cat,
        source_frame=source,
        projector_polygon=PROJECTOR_POLYGON,
        room_background=good_background,
    )

    assert any(580 <= candidate.bbox_xywh[0] <= 620 and candidate.bbox_xywh[1] >= 640 for candidate in good_candidates)
    assert good_background.frame_count == 1


def test_room_background_matches_ir_brightness_shift() -> None:
    pytest.importorskip("cv2")
    source = Image.new("RGB", (1280, 720), "white")
    no_cat_camera = _camera_room_with_projection(source).convert("L")
    dark_current = no_cat_camera.point(lambda value: max(0, int(value * 0.55)))
    camera_with_cat = dark_current.convert("RGB")
    ImageDraw.Draw(camera_with_cat).rectangle((585, 650, 675, 715), fill="black")
    background = source_subtraction.build_room_background([no_cat_camera])

    candidates = source_subtraction.detect_source_subtracted_candidates(
        camera_with_cat,
        source_frame=source,
        projector_polygon=PROJECTOR_POLYGON,
        room_background=background,
    )

    assert any(580 <= candidate.bbox_xywh[0] <= 620 and candidate.bbox_xywh[1] >= 640 for candidate in candidates)


def test_background_update_mask_keeps_possible_cat_path_out_of_average() -> None:
    base = Image.new("L", (100, 80), 120)
    current = Image.new("L", (100, 80), 180)
    ImageDraw.Draw(current).rectangle((40, 55, 60, 75), fill=20)
    update_mask = np.ones((80, 100), dtype=bool)
    update_mask[45:80, 30:70] = False

    updated = source_subtraction.update_room_background(base, current, update_mask=update_mask, alpha=0.5)
    updated_arr = np.asarray(updated)

    assert updated_arr[10, 10] == 150
    assert updated_arr[60, 50] == 120


def test_source_subtracted_candidates_reject_bad_baseline_shape() -> None:
    pytest.importorskip("cv2")
    source = Image.new("RGB", (1280, 720), "white")
    camera = Image.fromarray(
        source_subtraction.warp_source_to_camera(
            source,
            projector_polygon=PROJECTOR_POLYGON,
            width=1280,
            height=720,
        )
    ).convert("RGB")

    with pytest.raises(ValueError, match="residual_baseline shape"):
        source_subtraction.detect_source_subtracted_candidates(
            camera,
            source_frame=source,
            projector_polygon=PROJECTOR_POLYGON,
            residual_baseline=np.zeros((10, 10), dtype=np.float32),
        )


def test_source_subtraction_fits_projector_rgb_to_camera_luma() -> None:
    pytest.importorskip("cv2")
    source_arr = np.zeros((720, 1280, 3), dtype=np.uint8)
    yy, xx = np.indices((720, 1280))
    source_arr[..., 0] = (xx % 256).astype(np.uint8)
    source_arr[..., 1] = (yy % 256).astype(np.uint8)
    source_arr[..., 2] = ((xx // 3 + yy // 2) % 256).astype(np.uint8)
    source = Image.fromarray(source_arr, mode="RGB")
    warped_rgb = source_subtraction.warp_source_rgb_to_camera(
        source,
        projector_polygon=PROJECTOR_POLYGON,
        width=1280,
        height=720,
    ).astype(np.float32)
    projection = source_subtraction.projection_mask(PROJECTOR_POLYGON, 1280, 720)
    camera_arr = np.full((720, 1280), 80, dtype=np.uint8)
    camera_arr[projection] = np.clip(
        0.15 * warped_rgb[..., 0][projection]
        + 0.70 * warped_rgb[..., 1][projection]
        + 0.05 * warped_rgb[..., 2][projection]
        + 18,
        0,
        255,
    ).astype(np.uint8)
    camera = Image.fromarray(camera_arr, mode="L").convert("RGB")

    residual, _warped = source_subtraction.source_subtracted_residual(
        camera,
        source_frame=source,
        projector_polygon=PROJECTOR_POLYGON,
    )

    assert float(np.percentile(np.abs(residual[projection]), 95)) < 3.0


def test_robust_component_top_ignores_single_high_noise_pixel() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:30, 15:25] = 1
    labels[2, 5] = 1

    top = source_subtraction.robust_component_top_px(labels, 1)

    assert top is not None
    top_x, top_y = top
    assert 15 <= top_x <= 24
    assert 10 <= top_y <= 15


def test_robust_component_top_rejects_tiny_components() -> None:
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[4:6, 4:6] = 1

    assert source_subtraction.robust_component_top_px(labels, 1) is None
