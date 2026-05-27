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
        min_residual_fraction=0.10,
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
        min_residual_fraction=0.10,
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
        min_residual_fraction=0.10,
    )

    assert accepted == []
    assert skipped[0]["reason"] == "matches_projected_source"
    assert skipped[0]["best_reference_index"] == 1


def test_source_filter_falls_back_to_physical_occluder_without_detector_person() -> None:
    source = Image.new("RGB", (1280, 720), "white")
    camera = source.copy()
    ImageDraw.Draw(camera).rectangle((520, 180, 760, 520), fill="black")

    accepted, skipped = overlay_server._filter_source_projected_people(
        [],
        camera_image=camera,
        source_frame=source,
        projector_polygon=FULL_FRAME_PROJECTOR,
        residual_threshold=28.0,
        min_residual_area_px=1200,
        min_residual_fraction=0.10,
    )

    assert skipped == []
    assert len(accepted) == 1
    assert accepted[0].source == "source_subtracted_human_occluder"
    assert accepted[0].bbox_xyxy == (520.0, 180.0, 761.0, 521.0)


def test_source_filter_ignores_own_fixed_black_rect_feedback() -> None:
    source = Image.new("RGB", (1280, 720), "white")
    camera = source.copy()
    ImageDraw.Draw(camera).rectangle((520, 260, 760, 420), fill="black")

    accepted, skipped = overlay_server._filter_source_projected_people(
        [],
        camera_image=camera,
        source_frame=source,
        ignored_source_polygons=[((520.0, 260.0), (760.0, 260.0), (760.0, 420.0), (520.0, 420.0))],
        projector_polygon=FULL_FRAME_PROJECTOR,
        residual_threshold=28.0,
        min_residual_area_px=1200,
        min_residual_fraction=0.10,
    )

    assert accepted == []
    assert skipped == []


def test_eye_safety_hold_keeps_last_zone_during_short_detector_gap() -> None:
    person = overlay_server.PersonDetection(
        bbox_xyxy=(500.0, 130.0, 790.0, 390.0),
        confidence=0.9,
        source="test",
    )
    zone = overlay_server.projector_safety.SafetyOverlayZone(
        polygon=((520.0, 160.0), (760.0, 160.0), (760.0, 260.0), (520.0, 260.0)),
        camera_bbox_xyxy=person.bbox_xyxy,
    )
    active = overlay_server.SafetyOverlayResult("active", zones=(zone,), debug={"zone_count": 1})

    result, held_result, held_people, held_at = overlay_server._apply_eye_safety_hold(
        active,
        current_people=[person],
        held_result=None,
        held_people=[],
        held_at=0.0,
        now=10.0,
        hold_seconds=2.0,
    )
    assert result is active

    gap = overlay_server.SafetyOverlayResult("no_person", debug={"person_count": 0})
    held, held_result, held_people, held_at = overlay_server._apply_eye_safety_hold(
        gap,
        current_people=[],
        held_result=held_result,
        held_people=held_people,
        held_at=held_at,
        now=11.2,
        hold_seconds=2.0,
    )

    assert held.status == "active"
    assert held.zones == (zone,)
    assert held.debug["held_after_last_detection"] is True
    assert held.debug["held_source_status"] == "no_person"
    assert held_people == [person]


def test_eye_safety_hold_expires_after_timeout() -> None:
    zone = overlay_server.projector_safety.SafetyOverlayZone(
        polygon=((520.0, 160.0), (760.0, 160.0), (760.0, 260.0), (520.0, 260.0)),
        camera_bbox_xyxy=(500.0, 130.0, 790.0, 390.0),
    )
    held_result = overlay_server.SafetyOverlayResult("active", zones=(zone,), debug={})
    current = overlay_server.SafetyOverlayResult("no_person", debug={"person_count": 0})

    result, next_held_result, held_people, held_at = overlay_server._apply_eye_safety_hold(
        current,
        current_people=[],
        held_result=held_result,
        held_people=[],
        held_at=10.0,
        now=12.1,
        hold_seconds=2.0,
    )

    assert result is current
    assert next_held_result is None
    assert held_people == []
    assert held_at == 0.0


def test_runtime_defaults_prioritize_low_latency_camera_updates() -> None:
    args = overlay_server.parse_args(["--source-video", "cats.mp4"])

    assert args.fps == 20
    assert args.camera_sample_interval == 0.12
    assert args.camera_snapshot_timeout == 0.35
    assert args.eye_safety_hold_seconds == 2.0
