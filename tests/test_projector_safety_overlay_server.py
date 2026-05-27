import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "cat_projector_safety_overlay_server.py"
SPEC = spec_from_file_location("cat_projector_safety_overlay_server_test", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
overlay_server = module_from_spec(SPEC)
sys.modules[SPEC.name] = overlay_server
SPEC.loader.exec_module(overlay_server)


FULL_FRAME_PROJECTOR = ((0.0, 0.0), (1279.0, 0.0), (1279.0, 719.0), (0.0, 719.0))


def test_source_filter_rejects_projected_video_content() -> None:
    source = Image.new("RGB", (1280, 720), "white")
    ImageDraw.Draw(source).ellipse((520, 160, 760, 360), fill="black")
    camera = Image.fromarray(
        overlay_server.source_subtraction.warp_source_to_camera(
            source,
            projector_polygon=FULL_FRAME_PROJECTOR,
            width=1280,
            height=720,
        )
    ).convert("RGB")
    people = [
        overlay_server.PersonDetection(
            bbox_xyxy=(500.0, 130.0, 790.0, 390.0),
            confidence=0.9,
            source="test",
        )
    ]

    accepted, skipped = overlay_server._filter_source_projected_people(
        people,
        camera_image=camera,
        source_frame=source,
        projector_polygon=FULL_FRAME_PROJECTOR,
        residual_threshold=28.0,
        min_residual_area_px=1200,
        min_residual_fraction=0.025,
    )

    assert accepted == []
    assert skipped[0]["reason"] == "matches_projected_source"


def test_source_filter_keeps_real_occluder_in_projector_beam() -> None:
    source = Image.new("RGB", (1280, 720), "white")
    camera = source.copy()
    ImageDraw.Draw(camera).rectangle((500, 130, 790, 390), fill="black")
    people = [
        overlay_server.PersonDetection(
            bbox_xyxy=(500.0, 130.0, 790.0, 390.0),
            confidence=0.9,
            source="test",
        )
    ]

    accepted, skipped = overlay_server._filter_source_projected_people(
        people,
        camera_image=camera,
        source_frame=source,
        projector_polygon=FULL_FRAME_PROJECTOR,
        residual_threshold=28.0,
        min_residual_area_px=1200,
        min_residual_fraction=0.025,
    )

    assert len(accepted) == 1
    assert skipped == []
    assert accepted[0].debug["source_subtracted_residual_area_px"] > 1200


def test_source_filter_rejects_projected_video_content_from_reference_frame() -> None:
    current_source = Image.new("RGB", (1280, 720), "white")
    cat_intro_source = Image.new("RGB", (1280, 720), "white")
    ImageDraw.Draw(cat_intro_source).ellipse((520, 160, 760, 360), fill="black")
    camera = Image.fromarray(
        overlay_server.source_subtraction.warp_source_to_camera(
            cat_intro_source,
            projector_polygon=FULL_FRAME_PROJECTOR,
            width=1280,
            height=720,
        )
    ).convert("RGB")
    people = [
        overlay_server.PersonDetection(
            bbox_xyxy=(500.0, 130.0, 790.0, 390.0),
            confidence=0.9,
            source="test",
        )
    ]

    accepted, skipped = overlay_server._filter_source_projected_people(
        people,
        camera_image=camera,
        source_frame=current_source,
        source_reference_frames=[cat_intro_source],
        projector_polygon=FULL_FRAME_PROJECTOR,
        residual_threshold=28.0,
        min_residual_area_px=1200,
        min_residual_fraction=0.025,
    )

    assert accepted == []
    assert skipped[0]["reason"] == "matches_projected_source"
    assert skipped[0]["best_reference_index"] == 1
