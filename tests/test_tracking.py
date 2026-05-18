import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "cat_tv_play" / "tracking.py"
SPEC = spec_from_file_location("cat_tv_play_tracking", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
tracking = module_from_spec(SPEC)
sys.modules[SPEC.name] = tracking
SPEC.loader.exec_module(tracking)


def _detection(t: float, x_cm: float, y_cm: float, *, confidence: float = 0.9, area_px: int = 1800):
    return tracking.WallDetection(t=t, x_cm=x_cm, y_cm=y_cm, confidence=confidence, area_px=area_px)


def test_rejects_one_frame_false_spike_before_update() -> None:
    tracker = tracking.CatWallKalmanTracker()
    outputs = [
        tracker.step(0.0, [_detection(0.0, 40.0, 40.0)]),
        tracker.step(0.1, [_detection(0.1, 44.0, 82.0)]),
        tracker.step(0.2, [_detection(0.2, 44.0, 250.0, confidence=0.95, area_px=1600)]),
        tracker.step(0.3, [_detection(0.3, 48.0, 135.0)]),
    ]

    assert outputs[2] is not None
    assert outputs[2].accepted is None
    assert outputs[2].accepted_raw_y_cm is None
    accepted_heights = [output.accepted_raw_y_cm for output in outputs if output and output.accepted_raw_y_cm]
    assert 250.0 not in accepted_heights
    assert max(accepted_heights) == 135.0


def test_multiple_candidates_choose_physically_plausible_one() -> None:
    tracker = tracking.CatWallKalmanTracker()
    assert tracker.step(0.0, [_detection(0.0, 50.0, 45.0)]) is not None

    output = tracker.step(
        0.1,
        [
            _detection(0.1, 300.0, 210.0, confidence=0.99, area_px=2500),
            _detection(0.1, 56.0, 58.0, confidence=0.65, area_px=1000),
        ],
    )

    assert output is not None
    assert output.accepted is not None
    assert output.accepted.x_cm == 56.0
    assert output.accepted.y_cm == 58.0


def test_low_confidence_high_false_detection_rejected() -> None:
    tracker = tracking.CatWallKalmanTracker()
    assert tracker.step(0.0, [_detection(0.0, 20.0, 35.0)]) is not None

    output = tracker.step(0.1, [_detection(0.1, 22.0, 230.0, confidence=0.05, area_px=80)])

    assert output is not None
    assert output.accepted is None
    assert output.reason == "predicted_no_accepted_detection"


def test_missed_frames_do_not_reset_immediately() -> None:
    tracker = tracking.CatWallKalmanTracker(max_misses_before_reset=3)
    assert tracker.step(0.0, [_detection(0.0, 10.0, 20.0)]) is not None

    first_miss = tracker.step(0.1, [])
    second_miss = tracker.step(0.2, [])
    third_miss = tracker.step(0.3, [])
    reset_output = tracker.step(0.4, [])

    assert first_miss is not None
    assert second_miss is not None
    assert third_miss is not None
    assert reset_output is None


def test_confirmed_peak_uses_raw_accepted_detection_not_smoothed_state() -> None:
    peak = tracking.confirmed_peak_height_cm(
        [
            tracking.AcceptedJumpPoint(t=0.0, raw_y_cm=40.0, filtered_y_cm=40.0, confidence=0.9),
            tracking.AcceptedJumpPoint(t=0.1, raw_y_cm=180.0, filtered_y_cm=142.0, confidence=0.8),
            tracking.AcceptedJumpPoint(t=0.2, raw_y_cm=120.0, filtered_y_cm=136.0, confidence=0.9),
        ]
    )

    assert peak == 180.0
