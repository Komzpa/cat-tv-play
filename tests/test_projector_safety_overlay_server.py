import sys
import threading
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path

import pytest

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


def test_runtime_parse_args_disables_residual_occluder_fallback_by_default() -> None:
    args = overlay_server.parse_args(["--source-video", "source.mp4"])

    assert args.enable_residual_occluder_fallback is False
    assert args.camera_snapshot_timeout == 2.0
    assert args.safety_negative_review_interval == 60.0
    assert args.safety_negative_review_max_frames_per_day == 720
    assert args.ha_source_video_entity is None
    assert args.ha_source_video_poll_interval == 1.0


def test_projector_source_video_tracks_ha_entity_and_keeps_last_good_source(monkeypatch) -> None:
    values = iter(["http://ha/local/cat-tv/flies.mp4", "unavailable"])
    monkeypatch.setattr(overlay_server, "_load_ha_token", lambda path: "token")
    monkeypatch.setattr(
        overlay_server,
        "_read_ha_state",
        lambda *args, **kwargs: next(values),
    )
    source = overlay_server.ProjectorSourceVideo(
        fallback_source="http://ha/local/cat-tv/october.mp4",
        ha_url="http://ha",
        ha_config_path=Path("config.py"),
        entity_id="input_text.cat_projector_video_url",
        poll_interval_seconds=1.0,
        timeout_seconds=1.0,
    )

    assert source.source() == "http://ha/local/cat-tv/october.mp4"
    source.refresh()
    assert source.source() == "http://ha/local/cat-tv/flies.mp4"
    source.refresh()
    assert source.source() == "http://ha/local/cat-tv/flies.mp4"
    assert source.debug()["source_video_error"] is None


def test_runtime_parse_args_can_enable_residual_occluder_fallback() -> None:
    args = overlay_server.parse_args(["--source-video", "source.mp4", "--enable-residual-occluder-fallback"])

    assert args.enable_residual_occluder_fallback is True


def test_runtime_parse_args_can_disable_residual_occluder_fallback() -> None:
    args = overlay_server.parse_args(["--source-video", "source.mp4", "--disable-residual-occluder-fallback"])

    assert args.enable_residual_occluder_fallback is False


def test_safety_negative_review_rows_stop_at_daily_frame_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(overlay_server, "SAFETY_REVIEW_ROOT", tmp_path / "safety-review")
    camera = Image.new("RGB", (32, 32), "white")
    person = overlay_server.PersonDetection(
        bbox_xyxy=(1.0, 2.0, 12.0, 14.0),
        confidence=0.8,
        source="test",
    )
    result = overlay_server.SafetyOverlayResult("active")

    first = overlay_server._append_safety_negative_review_rows(
        camera_image=camera,
        raw_people=[person],
        people=[person],
        result=result,
        source_filter_skipped=[],
        performance={"safety_sample_index": 1},
        source_video="source.mp4",
        max_frames_per_day=1,
    )
    second = overlay_server._append_safety_negative_review_rows(
        camera_image=camera,
        raw_people=[person],
        people=[person],
        result=result,
        source_filter_skipped=[],
        performance={"safety_sample_index": 2},
        source_video="source.mp4",
        max_frames_per_day=1,
    )

    assert first == 1
    assert second == 0
    frames_dir = overlay_server.SAFETY_REVIEW_ROOT / overlay_server.time.strftime("%Y%m%d") / "frames"
    assert len(list(frames_dir.iterdir())) == 1


def test_projector_active_gate_idles_only_on_explicit_off_states() -> None:
    assert (
        overlay_server._projector_active_from_states(
            {
                "input_boolean.cat_projector_session": "off",
                "binary_sensor.cat_projector_display_active": "off",
            }
        )
        is False
    )
    assert (
        overlay_server._projector_active_from_states(
            {
                "input_boolean.cat_projector_session": "on",
                "binary_sensor.cat_projector_display_active": "off",
            }
        )
        is True
    )
    assert (
        overlay_server._projector_active_from_states(
            {
                "input_boolean.cat_projector_session": "unknown",
                "binary_sensor.cat_projector_display_active": "off",
            }
        )
        is True
    )


def test_safety_worker_does_not_sample_source_references_while_projector_inactive(monkeypatch) -> None:
    class IdleGate:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        def debug(self, now: float | None = None) -> dict[str, object]:
            return {"projector_active_gate_active": False}

    runtime = overlay_server.SafetyRuntime()
    args = overlay_server.parse_args(["--source-video", "source.mp4"])
    args.inactive_sample_interval = 0.01
    monkeypatch.setattr(overlay_server, "ProjectorActiveGate", IdleGate)
    monkeypatch.setattr(
        overlay_server,
        "_sample_source_reference_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source references sampled while idle")),
    )

    def stop_after_snapshot(snapshot: overlay_server.SafetyComputationSnapshot) -> None:
        runtime.stop()
        original_update(snapshot)

    original_update = runtime.update_snapshot
    monkeypatch.setattr(runtime, "update_snapshot", stop_after_snapshot)

    overlay_server._run_safety_worker(args, runtime=runtime)


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


def test_physical_track_drops_prediction_after_it_leaves_frame() -> None:
    track = overlay_server.SmoothedPersonTrack(
        bbox_xyxy=(570.0, 100.0, 630.0, 260.0),
        velocity_xy_px_s=(400.0, 0.0),
        confidence=0.9,
        last_seen_at=10.0,
        last_update_at=10.0,
        source="test",
    )

    predicted, tracks, debug = overlay_server._update_physical_person_tracks(
        [],
        [track],
        now=10.3,
        camera_size=(640, 480),
        max_missing_seconds=1.0,
        max_speed_px_s=1800.0,
        smoothing_alpha=0.65,
    )

    assert predicted == []
    assert tracks == []
    assert debug["physics_track_predicted_count"] == 0
    assert debug["physics_track_offscreen_count"] == 1


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
    assert args.ha_url is None
    assert args.ha_config_path is None
    assert args.ha_active_entity == list(overlay_server.DEFAULT_HA_ACTIVE_ENTITIES)
    assert args.ha_active_poll_interval == 1.0
    assert args.ha_active_timeout == 1.0
    assert args.inactive_sample_interval == 2.0
    assert args.inactive_status_interval == 1.0


def _camera_evidence(sequence: int = 1) -> overlay_server.CameraFrame:
    jpeg = BytesIO()
    Image.new("RGB", (64, 64), "white").save(jpeg, format="JPEG")
    return overlay_server.CameraFrame(
        jpeg=jpeg.getvalue(), received_at=10.0, received_wall_at=100.0,
        sequence=sequence, age_ms=100.0, connection=1,
    )


def test_camera_stream_selection_only_converts_known_snapshot_endpoint() -> None:
    choose = overlay_server._camera_stream_url
    assert choose("http://camera:8081/shot.jpg?fps=25", None) == "http://camera:8081/video?fps=25"
    assert choose("http://camera/other/shot.jpg", None) is None
    assert choose("http://frigate/api/camera/latest.jpg", None) is None
    assert choose("http://camera/snapshot", "http://camera/custom-mjpeg") == "http://camera/custom-mjpeg"


def test_latest_camera_slot_skips_duplicates_and_accepts_reset_only_after_reconnect() -> None:
    stream = overlay_server.LatestCameraStream("http://camera/video", timeout=2.0, max_age=0.9)
    first = _camera_evidence()
    newest = replace(first, sequence=2, received_at=10.2)
    assert stream._accept(first)
    assert stream._accept(newest)
    assert not stream._accept(replace(newest, received_at=10.8))
    assert stream.latest(10.3) == (newest, None)
    assert stream.latest(11.1)[1] == "camera stream frame stale"
    with pytest.raises(ValueError, match="regressed"):
        stream._accept(first)
    restarted = replace(first, connection=2, received_at=11.2)
    assert stream._accept(restarted)
    assert stream.latest(11.3) == (restarted, None)


def test_mjpeg_reads_complete_metadata_frames_and_rejects_partial_frame() -> None:
    class Response(BytesIO):
        headers = overlay_server.BytesHeaderParser().parsebytes(
            b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
        )

    def part(sequence: int, body: bytes) -> bytes:
        return (
            f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(body)}\r\n"
            f"X-Frame-Sequence: {sequence}\r\nX-Frame-Age-Ms: 12\r\nX-Frame-Encode-Ms: 3\r\n\r\n"
        ).encode() + body + b"\r\n"

    parts = overlay_server._iter_mjpeg_frames(
        Response(part(4, b"first") + part(5, b"newest")[:-4]), timeout=2.0, connection=7
    )
    first = next(parts)
    assert (first.jpeg, first.sequence, first.age_ms, first.encode_ms, first.connection) == (b"first", 4, 12, 3, 7)
    with pytest.raises(EOFError, match="disconnected"):
        next(parts)


@pytest.mark.parametrize("body, expected", [(b"", "disconnected"), (
    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 1\r\n\r\nx\r\n", "requires X-Frame"
)])
def test_stream_failure_is_visible_without_snapshot_fallback(monkeypatch, body, expected) -> None:
    class Response(BytesIO):
        headers = overlay_server.BytesHeaderParser().parsebytes(
            b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
        )

    urls = []
    stream = overlay_server.LatestCameraStream("http://camera/video", timeout=2.0, max_age=0.9)

    def open_stream(request, **kwargs):
        urls.append(request.full_url)
        return Response(body)

    monkeypatch.setattr(overlay_server, "urlopen", open_stream)
    monkeypatch.setattr(stream._stop, "wait", lambda timeout: stream._stop.set())
    stream.set_active(True)
    stream._run()
    frame, error = stream.latest(overlay_server.time.monotonic())
    assert frame is None
    assert expected in error
    assert urls == ["http://camera/video"]


@pytest.mark.parametrize("failure", ["stale", "disconnected"])
def test_safety_worker_does_not_renew_duplicate_frame_evidence(monkeypatch, failure) -> None:
    clock = [10.0]
    monkeypatch.setattr(overlay_server.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(overlay_server.time, "time", lambda: clock[0] + 90.0)

    def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(overlay_server.time, "sleep", advance)
    monkeypatch.setattr(overlay_server, "_sample_source_reference_frames", lambda *args, **kwargs: [])
    monkeypatch.setattr(overlay_server, "_filter_source_projected_people", lambda *args, **kwargs: ([], []))
    args = overlay_server.parse_args(["--source-video", "source.mp4", "--source-size", "64x64"])
    gate = overlay_server.ProjectorActiveGate(
        ha_url=None, ha_config_path=None, active_entities=(), poll_interval_seconds=1, timeout_seconds=1,
    )
    stream = overlay_server.LatestCameraStream("http://camera/video", timeout=2, max_age=0.9)
    stream._accept(_camera_evidence())
    runtime = overlay_server.SafetyRuntime()
    runtime.update_source_frame(Image.new("RGB", (64, 64)), 1, "source.mp4")
    snapshots = []

    class Detector:
        calls = 0

        def detect(self, image):
            self.calls += 1
            return []

    detector = Detector()

    def publish(snapshot):
        snapshots.append(snapshot)
        if snapshot.result.status == "safety_overlay_unavailable":
            runtime.stop()
        elif failure == "disconnected":
            with stream._lock:
                stream._error = "camera stream disconnected"

    monkeypatch.setattr(runtime, "update_snapshot", publish)
    overlay_server._run_safety_samples(args, runtime=runtime, detector=detector, active_gate=gate, camera_stream=stream)
    assert detector.calls == 1
    assert [snapshot.result.status for snapshot in snapshots] == ["no_person", "safety_overlay_unavailable"]
    assert snapshots[0].result.zones == ()
    assert snapshots[0].people == []
    assert snapshots[0].updated_at == pytest.approx(9.9)
    assert snapshots[0].updated_at_wall == pytest.approx(99.9)
    assert failure in snapshots[-1].camera_error


def test_status_publication_expires_clear_evidence_and_keeps_original_timestamp() -> None:
    args = overlay_server.parse_args(["--source-video", "source.mp4"])
    snapshot = overlay_server.SafetyComputationSnapshot(
        result=overlay_server.SafetyOverlayResult("no_person"), camera_image=Image.new("RGB", (64, 64)),
        updated_at=10.0, updated_at_wall=100.0,
    )
    state = overlay_server.SafetyOverlayState()
    for now in (10.1, 10.3):
        result, error = overlay_server._current_safety_result(snapshot, now=now, args=args)
        state.update(result, people=[], source_size=(64, 64), updated_at=snapshot.updated_at_wall, camera_error=error)
        assert state.snapshot()["updated_at"] == 100.0
    result, error = overlay_server._current_safety_result(snapshot, now=11.0, args=args)
    assert result.status == "safety_overlay_unavailable"
    assert result.zones == ()
    assert error == "camera evidence stale"


def test_slow_control_refresh_does_not_block_cached_safety_or_source_reads(monkeypatch) -> None:
    gate_entered, source_entered, release, readers_done = (threading.Event() for _ in range(4))
    monkeypatch.setattr(overlay_server, "_load_ha_token", lambda path: "token")

    def slow_read(ha_url, token, entity_id, **kwargs):
        (source_entered if entity_id == "source" else gate_entered).set()
        assert release.wait(2.0), "control refresh was not released"
        return "new.mp4" if entity_id == "source" else "off"

    monkeypatch.setattr(overlay_server, "_read_ha_state", slow_read)
    gate = overlay_server.ProjectorActiveGate(
        ha_url="http://ha", ha_config_path=Path("config.py"), active_entities=("active",),
        poll_interval_seconds=1, timeout_seconds=1,
    )
    source = overlay_server.ProjectorSourceVideo(
        fallback_source="old.mp4", ha_url="http://ha", ha_config_path=Path("config.py"), entity_id="source",
        poll_interval_seconds=1, timeout_seconds=1,
    )
    refreshers = [threading.Thread(target=cache.refresh) for cache in (gate, source)]
    observed = []

    def read_caches():
        observed.append((gate.active(overlay_server.time.monotonic()), source.source(), source.debug()))
        readers_done.set()

    reader = threading.Thread(target=read_caches)
    try:
        for thread in refreshers:
            thread.start()
        assert gate_entered.wait(1.0) and source_entered.wait(1.0)
        reader.start()
        assert readers_done.wait(0.5), "safety/status cache readers waited for HA HTTP"
        assert observed == [(True, "old.mp4", {"source_video_entity": "source", "source_video": "old.mp4", "source_video_error": None})]
    finally:
        release.set()
        for thread in refreshers:
            thread.join(2.0)
        if reader.ident is not None:
            reader.join(2.0)
    assert gate.active(overlay_server.time.monotonic()) is False
    assert source.source() == "new.mp4"

    def failed_read(*args, **kwargs):
        raise OSError("HA unavailable")

    monkeypatch.setattr(overlay_server, "_read_ha_state", failed_read)
    gate.refresh()
    source.refresh()
    assert gate.active(overlay_server.time.monotonic()) is True
    assert gate.debug()["projector_active_gate_error"] == "HA unavailable"
    assert source.source() == "new.mp4"
    assert source.debug()["source_video_error"] == "HA unavailable"
