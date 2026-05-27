import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "cat_tv_play" / "projector_safety.py"
SPEC = spec_from_file_location("cat_tv_play_projector_safety", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
projector_safety = module_from_spec(SPEC)
sys.modules[SPEC.name] = projector_safety
SPEC.loader.exec_module(projector_safety)


PROJECTOR_POLYGON = ((100.0, 50.0), (500.0, 50.0), (500.0, 350.0), (100.0, 350.0))


def _zone_result(bbox: tuple[float, float, float, float]):
    return projector_safety.compute_eye_safety_overlay(
        camera_size=(640, 480),
        source_size=(1280, 720),
        projector_polygon=PROJECTOR_POLYGON,
        people=[projector_safety.PersonDetection(bbox_xyxy=bbox, confidence=0.88, source="test")],
        padding_px=0,
        min_overlap_area_px=4,
    )


def test_eye_safety_head_band_maps_camera_overlap_to_source_polygon() -> None:
    result = _zone_result((200.0, 100.0, 300.0, 300.0))

    assert result.status == "active"
    assert len(result.zones) == 1
    zone = result.zones[0]
    xs = [point[0] for point in zone.polygon]
    ys = [point[1] for point in zone.polygon]
    assert min(xs) >= 315.0
    assert max(xs) <= 645.0
    assert min(ys) >= 115.0
    assert max(ys) <= 340.0
    assert zone.camera_overlap_area_px > 0


def test_eye_safety_ignores_person_outside_projection() -> None:
    result = _zone_result((520.0, 100.0, 610.0, 280.0))

    assert result.status == "no_projection_overlap"
    assert result.zones == ()


def test_eye_safety_clips_partial_head_overlap_to_projection() -> None:
    result = _zone_result((60.0, 80.0, 180.0, 240.0))

    assert result.status == "active"
    xs = [point[0] for point in result.zones[0].polygon]
    assert min(xs) == 0.0
    assert max(xs) < 260.0


def test_eye_safety_no_person_status() -> None:
    result = projector_safety.compute_eye_safety_overlay(
        camera_size=(640, 480),
        source_size=(1280, 720),
        projector_polygon=PROJECTOR_POLYGON,
        people=[],
    )

    assert result.status == "no_person"
    assert result.zones == ()


def test_eye_safety_unavailable_for_missing_projection_points() -> None:
    result = projector_safety.compute_eye_safety_overlay(
        camera_size=(640, 480),
        source_size=(1280, 720),
        projector_polygon=((0.0, 0.0), (1.0, 1.0)),
        people=[projector_safety.PersonDetection(bbox_xyxy=(10.0, 10.0, 40.0, 80.0), confidence=0.8)],
    )

    assert result.status == "safety_overlay_unavailable"
    assert "projector_polygon" in result.debug["error"]


def test_render_eye_safety_overlay_blacks_only_active_zone() -> None:
    result = _zone_result((200.0, 100.0, 300.0, 300.0))
    frame = Image.new("RGB", (1280, 720), (255, 255, 255))

    rendered = projector_safety.render_eye_safety_overlay(frame, result)
    arr = np.asarray(rendered)

    assert tuple(arr[130, 330]) == (0, 0, 0)
    assert tuple(arr[20, 20]) == (255, 255, 255)
