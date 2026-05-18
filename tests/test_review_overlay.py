import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PIL import Image, ImageDraw

MODULE_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "cat_tv_play" / "review_overlay.py"
SPEC = spec_from_file_location("cat_tv_play_review_overlay", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
review_overlay = module_from_spec(SPEC)
sys.modules[SPEC.name] = review_overlay
SPEC.loader.exec_module(review_overlay)


def _frame(color: str = "gray") -> Image.Image:
    image = Image.new("RGB", (640, 360), color)
    ImageDraw.Draw(image).rectangle((280, 120, 340, 220), fill="black")
    return image


def _hold(t: float, height_cm: float):
    return review_overlay.crop_jump_peak_hold(
        _frame(),
        t=t,
        height_cm=height_cm,
        peak_x_px=310,
        peak_y_px=150,
    )


def test_update_top_jump_holds_keeps_three_highest_distinct_peaks() -> None:
    holds: list = []
    for candidate in [_hold(0.0, 80), _hold(1.0, 120), _hold(2.0, 100), _hold(3.0, 90)]:
        holds = review_overlay.update_top_jump_holds(holds, candidate)

    assert [hold.height_cm for hold in holds] == [120, 100, 90]


def test_update_top_jump_holds_merges_same_jump_window() -> None:
    holds = review_overlay.update_top_jump_holds([], _hold(1.0, 120))
    holds = review_overlay.update_top_jump_holds(holds, _hold(1.3, 140))
    holds = review_overlay.update_top_jump_holds(holds, _hold(1.5, 130))

    assert len(holds) == 1
    assert holds[0].height_cm == 140


def test_crop_jump_peak_hold_clamps_to_frame_edges() -> None:
    hold = review_overlay.crop_jump_peak_hold(
        _frame(),
        t=0.0,
        height_cm=100,
        peak_x_px=4,
        peak_y_px=4,
        crop_width_px=120,
        crop_height_px=80,
    )

    assert hold.image.size == (120, 80)


def test_render_top_jump_holds_overlay_draws_on_right_side() -> None:
    frame = _frame("white")
    rendered = review_overlay.render_top_jump_holds_overlay(
        frame,
        [_hold(0.0, 160), _hold(1.0, 140), _hold(2.0, 130)],
        panel_width_px=180,
    )

    assert rendered.size == frame.size
    assert rendered.getpixel((620, 20)) != frame.getpixel((620, 20))
    assert rendered.getpixel((20, 20)) == frame.getpixel((20, 20))
