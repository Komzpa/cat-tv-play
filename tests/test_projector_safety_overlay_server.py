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


def test_runtime_parse_args_enables_residual_occluder_fallback_by_default() -> None:
    args = overlay_server.parse_args(["--source-video", "source.mp4"])

    assert args.enable_residual_occluder_fallback is True
    assert args.camera_snapshot_timeout == 2.0


def test_runtime_parse_args_can_disable_residual_occluder_fallback() -> None:
    args = overlay_server.parse_args(["--source-video", "source.mp4", "--disable-residual-occluder-fallback"])

    assert args.enable_residual_occluder_fallback is False


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


def test_source_filter_can_run_scaled_without_changing_detector_bbox() -> None:
    source = Image.new("RGB", (1280, 720), "white")
    camera = source.copy()
    ImageDraw.Draw(camera).rectangle((500, 130, 790, 390), fill="black")
    person = overlay_server.PersonDetection(
        bbox_xyxy=(500.0, 130.0, 790.0, 390.0),
        confidence=0.9,
        source="test",
    )

    accepted, skipped = overlay_server._filter_source_projected_people(
        [person],
        camera_image=camera,
        source_frame=source,
        projector_polygon=FULL_FRAME_PROJECTOR,
        residual_threshold=28.0,
        min_residual_area_px=1200,
        min_residual_fraction=0.10,
        source_filter_scale=0.5,
    )

    assert skipped == []
    assert accepted[0].bbox_xyxy == person.bbox_xyxy
    assert accepted[0].debug["source_filter_scale"] == 0.5


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


def test_source_filter_keeps_person_when_intro_cat_only_masks_body_residual() -> None:
    current_source = Image.new("RGB", (1280, 720), "white")
    intro_source = Image.new("RGB", (1280, 720), "white")
    # A dark projected startup cat in the center can make the full person bbox
    # look source-like, but the eye band still carries the physical occluder.
    ImageDraw.Draw(intro_source).rectangle((460, 180, 820, 560), fill="black")
    camera = Image.fromarray(
        overlay_server.source_subtraction.warp_source_to_camera(
            intro_source,
            projector_polygon=FULL_FRAME_PROJECTOR,
            width=1280,
            height=720,
        )
    ).convert("RGB")
    ImageDraw.Draw(camera).rectangle((500, 90, 790, 140), fill="black")
    person = overlay_server.PersonDetection(
        bbox_xyxy=(500.0, 70.0, 790.0, 590.0),
        confidence=0.9,
        source="test",
    )

    accepted, skipped = overlay_server._filter_source_projected_people(
        [person],
        camera_image=camera,
        source_frame=current_source,
        source_reference_frames=[intro_source],
        projector_polygon=FULL_FRAME_PROJECTOR,
        residual_threshold=28.0,
        min_residual_area_px=1200,
        min_residual_fraction=0.10,
    )

    assert skipped == []
    assert len(accepted) == 1
    assert accepted[0].bbox_xyxy == person.bbox_xyxy
    assert accepted[0].debug["source_subtracted_best_reference_index"] == 1
    assert accepted[0].debug["source_subtracted_residual_fraction"] < 0.10
    assert accepted[0].debug["source_subtracted_eye_band_residual_fraction"] >= 0.10

    result = overlay_server.compute_eye_safety_overlay(
        camera_size=camera.size,
        source_size=(1280, 720),
        projector_polygon=FULL_FRAME_PROJECTOR,
        people=accepted,
        min_overlap_area_px=24,
    )
    assert result.status == "active"
    rendered = overlay_server.render_eye_safety_overlay(current_source, result)
    assert rendered.getpixel((640, 130)) == (0, 0, 0)


def test_source_filter_does_not_create_eye_zone_from_residual_without_detector_person_by_default() -> None:
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

    assert accepted == []
    assert skipped == [{"reason": "residual_occluder_fallback_disabled"}]


def test_source_filter_skips_residual_work_without_detector_people_by_default(monkeypatch) -> None:
    def fail_build_residual_views(**_kwargs):
        raise AssertionError("residual views should not be built without detector people")

    monkeypatch.setattr(overlay_server, "_build_residual_views", fail_build_residual_views)

    accepted, skipped = overlay_server._filter_source_projected_people(
        [],
        camera_image=Image.new("RGB", (1280, 720), "white"),
        source_frame=Image.new("RGB", (1280, 720), "white"),
        projector_polygon=FULL_FRAME_PROJECTOR,
        residual_threshold=28.0,
        min_residual_area_px=1200,
        min_residual_fraction=0.10,
    )

    assert accepted == []
    assert skipped == [{"reason": "residual_occluder_fallback_disabled"}]


def test_source_filter_can_fall_back_to_physical_occluder_when_enabled() -> None:
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
        enable_residual_occluder_fallback=True,
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


def test_eye_safety_trail_keeps_recent_eye_zones() -> None:
    old_zone = overlay_server.projector_safety.SafetyOverlayZone(
        polygon=((520.0, 160.0), (760.0, 160.0), (760.0, 260.0), (520.0, 260.0)),
        camera_bbox_xyxy=(500.0, 130.0, 790.0, 390.0),
    )
    new_zone = overlay_server.projector_safety.SafetyOverlayZone(
        polygon=((560.0, 160.0), (800.0, 160.0), (800.0, 260.0), (560.0, 260.0)),
        camera_bbox_xyxy=(540.0, 130.0, 830.0, 390.0),
    )
    old = overlay_server.SafetyOverlayResult("active", zones=(old_zone,), debug={})
    current = overlay_server.SafetyOverlayResult("active", zones=(new_zone,), debug={"zone_count": 1})

    trailed, recent = overlay_server._apply_eye_safety_trail(
        current,
        recent_eye_zone_results=[(9.7, old)],
        now=10.0,
        trail_seconds=0.5,
    )

    assert trailed.status == "active"
    assert trailed.zones == (old_zone, new_zone)
    assert trailed.debug["eye_safety_trail_zone_count"] == 2
    assert len(recent) == 2


def test_eye_safety_trail_drops_expired_zones() -> None:
    old_zone = overlay_server.projector_safety.SafetyOverlayZone(
        polygon=((520.0, 160.0), (760.0, 160.0), (760.0, 260.0), (520.0, 260.0)),
        camera_bbox_xyxy=(500.0, 130.0, 790.0, 390.0),
    )
    new_zone = overlay_server.projector_safety.SafetyOverlayZone(
        polygon=((560.0, 160.0), (800.0, 160.0), (800.0, 260.0), (560.0, 260.0)),
        camera_bbox_xyxy=(540.0, 130.0, 830.0, 390.0),
    )
    old = overlay_server.SafetyOverlayResult("active", zones=(old_zone,), debug={})
    current = overlay_server.SafetyOverlayResult("active", zones=(new_zone,), debug={"zone_count": 1})

    trailed, recent = overlay_server._apply_eye_safety_trail(
        current,
        recent_eye_zone_results=[(9.0, old)],
        now=10.0,
        trail_seconds=0.5,
    )

    assert trailed is current
    assert recent == [(10.0, current)]


def test_motion_prediction_annotates_people_from_previous_velocity() -> None:
    previous = overlay_server.PersonDetection(
        bbox_xyxy=(100.0, 100.0, 180.0, 260.0),
        confidence=0.9,
        source="test",
    )
    current = overlay_server.PersonDetection(
        bbox_xyxy=(130.0, 112.0, 210.0, 272.0),
        confidence=0.91,
        source="test",
    )

    annotated = overlay_server._annotate_motion_prediction(
        [current],
        previous_people=[previous],
        previous_at=10.0,
        now=10.1,
        horizon_seconds=0.25,
        padding_px=16.0,
        max_prediction_px=220.0,
    )

    assert annotated[0].bbox_xyxy == current.bbox_xyxy
    assert annotated[0].debug["prediction_offset_px"] == (75.0, 30.0)
    assert annotated[0].debug["prediction_padding_px"] == 16.0
    assert annotated[0].debug["eye_velocity_px_s"] == (300.0, 120.0)


def test_motion_prediction_clamps_large_offset() -> None:
    previous = overlay_server.PersonDetection(
        bbox_xyxy=(100.0, 100.0, 180.0, 260.0),
        confidence=0.9,
        source="test",
    )
    current = overlay_server.PersonDetection(
        bbox_xyxy=(500.0, 100.0, 580.0, 260.0),
        confidence=0.91,
        source="test",
    )

    annotated = overlay_server._annotate_motion_prediction(
        [current],
        previous_people=[previous],
        previous_at=10.0,
        now=10.1,
        horizon_seconds=0.25,
        padding_px=16.0,
        max_prediction_px=80.0,
    )

    offset_x, offset_y = annotated[0].debug["prediction_offset_px"]
    assert offset_x == 80.0
    assert offset_y == 0.0


def test_physical_track_predicts_person_through_short_detector_gap() -> None:
    first = overlay_server.PersonDetection(
        bbox_xyxy=(100.0, 100.0, 180.0, 260.0),
        confidence=0.9,
        source="test",
    )
    second = overlay_server.PersonDetection(
        bbox_xyxy=(140.0, 100.0, 220.0, 260.0),
        confidence=0.9,
        source="test",
    )

    people, tracks, debug = overlay_server._update_physical_person_tracks(
        [first],
        [],
        now=10.0,
        camera_size=(640, 480),
        max_missing_seconds=1.0,
        max_speed_px_s=1800.0,
        smoothing_alpha=1.0,
    )
    people, tracks, debug = overlay_server._update_physical_person_tracks(
        [second],
        tracks,
        now=10.2,
        camera_size=(640, 480),
        max_missing_seconds=1.0,
        max_speed_px_s=1800.0,
        smoothing_alpha=1.0,
    )
    predicted, tracks, debug = overlay_server._update_physical_person_tracks(
        [],
        tracks,
        now=10.4,
        camera_size=(640, 480),
        max_missing_seconds=1.0,
        max_speed_px_s=1800.0,
        smoothing_alpha=1.0,
    )

    assert people[0].bbox_xyxy == second.bbox_xyxy
    assert len(predicted) == 1
    assert predicted[0].source == "physics_smoothed_person_track"
    assert predicted[0].bbox_xyxy[0] > second.bbox_xyxy[0]
    assert predicted[0].debug["physics_predicted"] is True
    assert debug["physics_track_predicted_count"] == 1


def test_predicted_only_physics_track_is_not_served_as_overlay_person() -> None:
    predicted = overlay_server.PersonDetection(
        bbox_xyxy=(140.0, 100.0, 220.0, 260.0),
        confidence=0.8,
        source="physics_smoothed_person_track",
        debug={"physics_predicted": True},
    )

    assert overlay_server._people_with_current_frame_evidence([predicted]) == []

    result = overlay_server.compute_eye_safety_overlay(
        camera_size=(640, 480),
        source_size=(1280, 720),
        projector_polygon=((0.0, 0.0), (639.0, 0.0), (639.0, 479.0), (0.0, 479.0)),
        people=overlay_server._people_with_current_frame_evidence([predicted]),
        min_overlap_area_px=24,
    )

    assert result.status == "no_person"
    assert result.zones == ()


def test_current_detector_person_is_still_served_as_overlay_person() -> None:
    current = overlay_server.PersonDetection(
        bbox_xyxy=(140.0, 100.0, 220.0, 260.0),
        confidence=0.8,
        source="opencv_mobilenet_ssd",
        debug={"physics_smoothed": True, "physics_predicted": False},
    )

    assert overlay_server._people_with_current_frame_evidence([current]) == [current]


def test_physical_track_expires_after_missing_window() -> None:
    track = overlay_server.SmoothedPersonTrack(
        bbox_xyxy=(100.0, 100.0, 180.0, 260.0),
        velocity_xy_px_s=(200.0, 0.0),
        confidence=0.9,
        last_seen_at=10.0,
        last_update_at=10.0,
        source="test",
    )

    predicted, tracks, debug = overlay_server._update_physical_person_tracks(
        [],
        [track],
        now=11.2,
        camera_size=(640, 480),
        max_missing_seconds=1.0,
        max_speed_px_s=1800.0,
        smoothing_alpha=0.65,
    )

    assert predicted == []
    assert tracks == []
    assert debug["physics_track_count"] == 0


def test_stale_active_result_expires_to_clear_overlay() -> None:
    zone = overlay_server.projector_safety.SafetyOverlayZone(
        polygon=((520.0, 160.0), (760.0, 160.0), (760.0, 260.0), (520.0, 260.0)),
        camera_bbox_xyxy=(500.0, 130.0, 790.0, 390.0),
    )
    active = overlay_server.SafetyOverlayResult("active", zones=(zone,), debug={"zone_count": 1})

    fresh = overlay_server._expire_stale_active_result(
        active,
        now=10.2,
        updated_at=10.0,
        max_age_seconds=0.35,
    )
    stale = overlay_server._expire_stale_active_result(
        active,
        now=10.5,
        updated_at=10.0,
        max_age_seconds=0.35,
    )

    assert fresh is active
    assert stale.status == "no_person"
    assert stale.zones == ()
    assert stale.debug["stale_active_expired"] is True
    assert stale.debug["stale_active_age_seconds"] == 0.5


def test_runtime_defaults_prioritize_live_safety_camera_updates() -> None:
    args = overlay_server.parse_args(["--source-video", "cats.mp4"])

    assert args.fps == 20
    assert args.source_reference_frames == 3
    assert args.source_filter_scale == 0.35
    assert args.eye_band_top_fraction == 0.07
    assert args.eye_band_bottom_fraction == 0.19
    assert args.eye_band_left_fraction == 0.20
    assert args.eye_band_right_fraction == 0.92
    assert args.padding_px == 12
    assert args.camera_sample_interval == 0.06
    assert args.camera_snapshot_timeout == 2.0
    assert args.eye_safety_trail_seconds == 0.15
    assert args.eye_safety_hold_seconds == 0.0
    assert args.eye_safety_prediction_seconds == 0.25
    assert args.eye_safety_prediction_padding_px == 16.0
    assert args.eye_safety_max_prediction_px == 220.0
    assert args.max_active_overlay_age == 0.9
    assert args.person_track_max_missing_seconds == 1.0
    assert args.person_track_max_speed_px_s == 1800.0
    assert args.person_track_smoothing_alpha == 0.65
    assert args.status_only is False
    assert args.source_tracking_fps == 5
