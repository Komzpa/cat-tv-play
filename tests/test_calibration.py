import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "cat_tv_play" / "calibration.py"
SPEC = spec_from_file_location("cat_tv_play_calibration", CALIBRATION_PATH)
assert SPEC is not None
assert SPEC.loader is not None
calibration = module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)

CalibrationPoint = calibration.CalibrationPoint
image_to_wall_homography = calibration.image_to_wall_homography
transform_image_point = calibration.transform_image_point


def test_homography_maps_calibration_points() -> None:
    points = [
        CalibrationPoint(616, 719, 0, 0),
        CalibrationPoint(615, 425, 0, 100),
        CalibrationPoint(469, 427, -52, 100),
        CalibrationPoint(755, 416, 48, 100),
        CalibrationPoint(269, 719, -128, 0),
        CalibrationPoint(269, 442, -128, 100),
    ]

    homography = image_to_wall_homography(points)

    for point in points:
        wall_x, wall_y = transform_image_point(homography, point.image_x, point.image_y)
        assert wall_x == pytest_approx(point.wall_x_cm, abs=4)
        assert wall_y == pytest_approx(point.wall_y_cm, abs=4)


def test_sher_pilot_jump_height_regression() -> None:
    points = [
        CalibrationPoint(616, 719, 0, 0),
        CalibrationPoint(615, 425, 0, 100),
        CalibrationPoint(469, 427, -52, 100),
        CalibrationPoint(755, 416, 48, 100),
        CalibrationPoint(269, 719, -128, 0),
        CalibrationPoint(269, 442, -128, 100),
    ]

    homography = image_to_wall_homography(points)
    wall_x, wall_y = transform_image_point(homography, 253, 220)

    assert wall_x == pytest_approx(-135, abs=8)
    assert wall_y == pytest_approx(180, abs=8)


def test_requires_four_points() -> None:
    try:
        image_to_wall_homography([CalibrationPoint(0, 0, 0, 0)])
    except ValueError as exc:
        assert "at least four" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def pytest_approx(value: float, *, abs: float):  # noqa: A002
    import pytest

    return pytest.approx(value, abs=abs)
