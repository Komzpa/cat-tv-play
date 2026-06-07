import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "cat_tv_play" / "publish_render.py"
SPEC = spec_from_file_location("cat_tv_play_publish_render", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
publish_render = module_from_spec(SPEC)
sys.modules[SPEC.name] = publish_render
SPEC.loader.exec_module(publish_render)


def test_publish_content_crop_uses_screen_and_cat_geometry() -> None:
    screen = publish_render.projector_screen_bbox((1280, 720))
    content = publish_render.content_bbox(
        image_size=(1280, 720),
        cat_boxes=[(341.0, 277.0, 921.0, 720.0)],
    )
    crop = publish_render.close_crop_bounds(
        image_size=(1280, 720),
        content_bounds=content,
        output_size=(1080, 1920),
    )

    assert tuple(round(value, 2) for value in screen) == (40.22, 57.27, 937.92, 680.97)
    assert tuple(round(value, 2) for value in content) == (40.22, 57.27, 937.92, 720.0)
    assert crop is not None
    left, _top, width, _height = crop
    assert left <= screen[0]
    assert 960.0 < left + width < 980.0


def test_publish_content_bbox_includes_cat_polygons() -> None:
    content = publish_render.content_bbox(
        image_size=(1280, 720),
        cat_polygons=[
            [(900.0, 700.0), (970.0, 710.0), (940.0, 720.0)],
        ],
    )

    assert tuple(round(value, 2) for value in content) == (40.22, 57.27, 970.0, 720.0)


def test_recording_date_label_from_recording_dir() -> None:
    label = publish_render.recording_date_label(
        recording_dir="/tmp/20260518T184855_20260518T184852-cats-tv-october-compilation"
    )

    assert label == "Monday, 18 May 2026"
    assert publish_render.presentation_metadata(recording_dir="20260518T184855_demo") == {
        "recorded_on": "Monday, 18 May 2026",
        "subject": "Sher",
        "title": "SHER JUMPS",
    }
