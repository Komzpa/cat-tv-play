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
