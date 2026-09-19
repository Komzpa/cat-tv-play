from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, ImageDraw

from cat_tv_play_notify import DeliveryResult

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_learning_recorder.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("cat_projector_learning_recorder", SCRIPT_PATH)
recorder = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = recorder
SCRIPT_SPEC.loader.exec_module(recorder)

#: Stand-in for whatever treat dispenser an installation configures.
REWARD_ENTITY_ID = "button.example_treat_dispenser"


def _write_synthetic_wall_calibration(path: Path) -> Path:
    """Write a neutral 1280x720 wall calibration, floor line at 0 mm."""

    path.write_text(
        json.dumps(
            {
                "calibration": {
                    "image_points_px": [[60.0, 700.0], [1180.0, 660.0], [1150.0, 90.0], [80.0, 140.0]],
                    "wall_points_mm": [[0.0, 0.0], [3000.0, 0.0], [3000.0, 2200.0], [0.0, 2200.0]],
                    "floor_y_mm": 0,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class CatProjectorLearningRecorderTest(unittest.TestCase):
    def test_recordings_root_can_be_separate_from_mutable_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            state_root = tmp_root / "state"
            recordings_root = tmp_root / "cold" / "recordings"
            manifest = recordings_root / "20260712T120000_session" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"ha_session_id": "session"}\n', encoding="utf-8")

            args = recorder.parse_args(["--root", str(state_root), "--recordings-root", str(recordings_root), "status"])

            self.assertEqual(recorder._recordings_root(args, state_root), recordings_root)  # noqa: SLF001
            self.assertEqual(recorder._latest_manifest_path(recordings_root), manifest)  # noqa: SLF001

    def test_custom_state_root_keeps_legacy_recordings_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_root = Path(tmpdir) / "state"
            args = recorder.parse_args(["--root", str(state_root), "status"])

            self.assertEqual(recorder._recordings_root(args, state_root), state_root / "recordings")  # noqa: SLF001

    def test_tiktok_publish_render_requirements_pin_viewer_safe_contract(self) -> None:
        doc = (Path(__file__).resolve().parents[1] / "docs" / "tiktok-render-requirements.md").read_text(
            encoding="utf-8"
        )

        required_phrases = [
            "exact source chunk and frame window",
            "TikTok safe zones",
            "Detector internals such as model type, probability, chunk id, frame id, or",
            "Once the close crop is selected for a jump, keep that trajectory crop fixed",
            "Output FPS must match real motion cadence",
            "Neural enhancement must run when the local tools and models are installed",
            "Frame interpolation must not create ghost overlays",
            "source FPS, output FPS, and selected jump",
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, doc)

    def test_telegram_jump_roi_covers_whole_camera_frame(self) -> None:
        self.assertEqual(recorder.DEFAULT_TELEGRAM_JUMP_ROI, (0, 0, 1280, 720))

    def test_review_deep_link_targets_highlight_frame_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recording = Path(tmpdir) / "recordings" / "20260606T000431_session"
            recording.mkdir(parents=True)
            (recording / "chunk_0204.mp4").write_bytes(b"fake")

            with mock.patch.object(recorder.subprocess, "run") as run:

                def fake_extract(command, **_kwargs):
                    if command[0] == "ffprobe":
                        if "frame=best_effort_timestamp_time,pkt_pts_time" in command:
                            return mock.Mock(
                                returncode=0,
                                stdout=json.dumps(
                                    {
                                        "frames": [
                                            {"best_effort_timestamp_time": f"{index / 30:.6f}"} for index in range(107)
                                        ]
                                    }
                                ),
                            )
                        return mock.Mock(returncode=0, stdout=json.dumps({"streams": [{"avg_frame_rate": "30/1"}]}))
                    output_pattern = Path(command[-1])
                    for index in range(107):
                        output_pattern.with_name(output_pattern.name.replace("%05d", f"{index:05d}")).write_bytes(
                            b"x" * 1024
                        )
                    return mock.Mock(returncode=0)

                run.side_effect = fake_extract
                url = recorder._review_deep_link(  # noqa: SLF001
                    recording,
                    max_record={"chunk": 204, "offset_seconds": 3.5},
                    base_url="http://review.local/ui.html?mode=review",
                )
                self.assertTrue((recording / "review_frames" / "chunk_0204_00105.jpg").exists())
                self.assertTrue((recording / "review_frames" / "chunk_0204_00104.jpg").exists())
                self.assertTrue((recording / "review_frames" / "chunk_0204_00106.jpg").exists())
                ffmpeg_command = next(call.args[0] for call in run.call_args_list if call.args[0][0] == "ffmpeg")
                self.assertIn("-vsync", ffmpeg_command)
                self.assertNotIn("-vf", ffmpeg_command)

            self.assertIn("mode=review", url)
            self.assertIn("review_video_id=video.", url)
            self.assertIn("review_frame_label=chunk_0204_00105.jpg", url)

    def test_select_clip_windows_uses_start_middle_and_end(self) -> None:
        chunks = [Path(f"chunk_{index:04d}.mp4") for index in range(12)]

        windows = recorder._select_clip_windows(chunks, window_chunks=3, max_clips=3)  # noqa: SLF001

        self.assertEqual(
            windows,
            [
                [Path("chunk_0000.mp4"), Path("chunk_0001.mp4"), Path("chunk_0002.mp4")],
                [Path("chunk_0004.mp4"), Path("chunk_0005.mp4"), Path("chunk_0006.mp4")],
                [Path("chunk_0009.mp4"), Path("chunk_0010.mp4"), Path("chunk_0011.mp4")],
            ],
        )

    def test_telegram_notification_is_marked_once_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260516T120000_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-16T12:00:00+04:00",
                        "ha_session_id": "20260516T120000-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(4):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            real_datetime = recorder.datetime
            highlight = recorder.JumpHighlight(
                chunks=[recording / "chunk_0001.mp4", recording / "chunk_0002.mp4"],
                max_height_cm=111.3,
                max_record={"chunk": 1, "height_cm": 111.3, "reason": "calibrated_high_jump"},
                candidate_count=3,
            )
            with (
                mock.patch.object(
                    recorder,
                    "_build_annotated_jump_clip",
                    return_value={"max_height_cm": 111.3, "candidate_frame_count": 18, "frame_count": 60},
                ) as build_clip,  # noqa: SLF001
                mock.patch.object(recorder, "_send_telegram_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(recorder, "_select_jump_highlight_clip_window", return_value=highlight),
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    return_value=recorder.CameraHumanCheck(False, "no_camera_human_detected", checked_frames=2),
                ),
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-16T12:10:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                def fake_build_clip(_chunks, path, **_kwargs):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"annotated")
                    return {"max_height_cm": 111.3, "candidate_frame_count": 18, "frame_count": 60}

                build_clip.side_effect = fake_build_clip

                payload = recorder._maybe_send_telegram_notification(  # noqa: SLF001
                    root,
                    max_age_seconds=3600,
                    window_chunks=2,
                    max_clips=2,
                )
                second_payload = recorder._maybe_send_telegram_notification(  # noqa: SLF001
                    root,
                    max_age_seconds=3600,
                    window_chunks=2,
                    max_clips=2,
                )

        self.assertEqual(payload["telegram_action"], "sent")
        self.assertEqual(second_payload["telegram_action"], "already_sent")
        self.assertEqual(payload["selection_method"], "calibrated_jump_highlight")
        self.assertEqual(payload["jump_caption"], "Шер прыгнул на 111.3 см")
        self.assertIn("\nРазметить: ", payload["caption"])
        self.assertIn("review_video_id=video.", payload["review_url"])
        self.assertIn("review_frame_label=chunk_0001_00000.jpg", payload["review_url"])
        self.assertEqual(send_notification.call_count, 1)
        self.assertEqual(send_notification.call_args.kwargs["caption"], payload["caption"])

    def test_rescore_publish_candidate_skips_broken_raw_frame_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "clean"
            frame_dir = recording / "review_frames"
            frame_dir.mkdir(parents=True)
            frame_path = frame_dir / "chunk_0124_00023.jpg"
            Image.new("RGB", (1280, 720), "white").save(frame_path)
            (recording / "chunk_0124.mp4").write_bytes(b"fake chunk")
            broken_frame = root / "batch_reviews" / "raw" / "chunk_0158_00041.jpg"
            broken_frame.parent.mkdir(parents=True)
            broken_frame.write_bytes(b"fake frame")
            jump_heights = root / "jump_heights.json"
            jump_heights.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "label": "broken",
                                "max_jump_height_cm": 147.0,
                                "max_frame_path": str(broken_frame),
                                "recording_dir": str(recording),
                            },
                            {
                                "label": "clean",
                                "max_jump_height_cm": 142.0,
                                "max_frame_path": str(frame_path),
                                "recording_dir": str(recording),
                                "max_bbox": "556,285,82,215",
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            candidate, suspicious = recorder._select_publishable_rescore_jump(  # noqa: SLF001
                jump_heights,
                min_source_fps=None,
            )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.video["label"], "clean")
        self.assertEqual(candidate.integrity.status, "clean")
        self.assertEqual(candidate.integrity.chunk_index, 124)
        self.assertEqual(candidate.integrity.frame_index, 23)
        self.assertEqual(len(suspicious), 1)
        self.assertEqual(suspicious[0].integrity.reason, "frame_path_outside_recording_dir")

    def test_rescore_probe_row_merges_mask_polygon_into_render_candidate(self) -> None:
        video = {
            "max_bbox": "10,20,30,40",
            "max_probability": 0.51,
            "max_frame_path": "/tmp/frame.jpg",
        }
        row = {
            "raw_path": "/tmp/frame.jpg",
            "best_bbox": "11,21,31,41",
            "best_probability": 0.87,
            "detector_backend": "ultralytics_yolo_segmentation",
            "detector_model_id": "/models/sher-yolo-seg-contour.pt",
            "model_created_at": "2026-06-06T18:16:21+00:00",
            "best_top_height_cm": 63.4,
            "best_mask_polygon": [{"x": 11, "y": 21}, {"x": 42, "y": 21}, {"x": 42, "y": 62}, {"x": 11, "y": 62}],
            "best_measurement_point": {"image_x": 27.5, "image_y": 19.5, "wall_x_cm": 4.2},
        }

        merged = recorder._merge_probe_row_into_rescore_video(video, row)  # noqa: SLF001
        record = recorder._rescore_candidate_record(  # noqa: SLF001
            merged,
            recorder.SourceIntegrity(status="clean", reason="ok", chunk_index=1, frame_index=2, offset_seconds=0.5),
        )

        self.assertEqual(merged["max_bbox"], "11,21,31,41")
        self.assertEqual(merged["max_probability"], 0.87)
        self.assertEqual(merged["max_source"], "ultralytics_yolo_segmentation")
        self.assertEqual(merged["max_model_path"], "/models/sher-yolo-seg-contour.pt")
        self.assertEqual(merged["max_jump_height_cm"], row["best_top_height_cm"])
        self.assertEqual(merged["max_frame_path"], row["raw_path"])
        self.assertEqual(merged["best_measurement_point"], row["best_measurement_point"])
        sample = recorder._rescore_live_sample(merged, source="frame.jpg")  # noqa: SLF001
        self.assertEqual(sample.top_x_px, 27.5)
        self.assertEqual(sample.top_y_px, 19.5)
        self.assertEqual(sample.wall_x_mm, 42.0)
        self.assertEqual(record["top_px"], [27.5, 19.5])
        self.assertEqual(
            record["mask_polygon"],
            [{"x": 11, "y": 21}, {"x": 42, "y": 21}, {"x": 42, "y": 62}, {"x": 11, "y": 62}],
        )
        self.assertEqual(
            recorder._mask_polygon_points(record["mask_polygon"]),  # noqa: SLF001
            [(11.0, 21.0), (42.0, 21.0), (42.0, 62.0), (11.0, 62.0)],
        )

    def test_rescore_publish_render_uses_jump_heights_calibration_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration = root / "calibrations" / "wall.json"
            calibration.parent.mkdir()
            calibration.write_text("{}", encoding="utf-8")
            jump_heights = root / "jump_heights.json"
            jump_heights.write_text(
                json.dumps({"calibration_path": str(calibration), "videos": []}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(recorder._calibration_path_for_jump_heights(jump_heights), calibration)  # noqa: SLF001

    def test_rescore_source_frame_timeline_extracts_original_chunk_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recording = Path(tmpdir) / "recording"
            recording.mkdir(parents=True)
            chunk_path = recording / "chunk_0012.mp4"
            chunk_path.write_bytes(b"fake")
            integrity = recorder.SourceIntegrity(
                status="clean",
                reason="ok",
                recording_dir=recording,
                chunk_path=chunk_path,
                chunk_index=12,
            )
            calls = []

            def fake_extract(chunks, frames_dir):
                calls.append((chunks, frames_dir))
                paths = []
                for index in [2, 0, 1]:
                    path = frames_dir / f"chunk_0012_{index:05d}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (32, 32), "white").save(path)
                    paths.append(path)
                other = frames_dir / "chunk_0013_00000.jpg"
                Image.new("RGB", (32, 32), "white").save(other)
                paths.append(other)
                return sorted(paths)

            with mock.patch.object(recorder, "_extract_exact_video_frames", side_effect=fake_extract):
                frames = recorder._rescore_source_frame_timeline(integrity, Path(tmpdir) / "source")  # noqa: SLF001

        self.assertEqual(
            [path.name for path in frames], ["chunk_0012_00000.jpg", "chunk_0012_00001.jpg", "chunk_0012_00002.jpg"]
        )
        self.assertEqual(calls[0][0], [chunk_path])

    def test_probe_row_lookup_falls_back_to_recording_scoped_frame_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recording"
            review_dir = recording / "review_frames"
            other_recording = root / "other"
            review_dir.mkdir(parents=True)
            other_recording.mkdir()
            row = {"raw_path": str(review_dir / "chunk_0012_00023.jpg"), "best_bbox": "1,2,3,4"}
            other_row = {"raw_path": str(other_recording / "chunk_0012_00023.jpg"), "best_bbox": "9,9,9,9"}
            rows_by_raw_path = {
                row["raw_path"]: row,
                other_row["raw_path"]: other_row,
            }
            rows_by_frame_key = recorder._load_probe_rows_by_frame_key(  # noqa: SLF001
                rows_by_raw_path,
                recording_dir=recording,
            )
            extracted_path = root / "tmp" / "source" / "chunk_0012_00023.jpg"

            found = recorder._probe_row_for_frame_path(  # noqa: SLF001
                extracted_path,
                rows_by_raw_path,
                rows_by_frame_key,
            )

        self.assertIs(found, row)

    def test_publishable_rescore_render_does_not_slow_motion_to_reach_min_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "20260518T184855_20260518T184852-cats-tv-october-compilation"
            review_dir = recording / "review_frames"
            review_dir.mkdir(parents=True)
            for index in range(4):
                Image.new("RGB", (1280, 720), "white").save(review_dir / f"chunk_0012_{index:05d}.jpg")
            chunk_path = recording / "chunk_0012.mp4"
            chunk_path.write_bytes(b"fake chunk")
            jump_heights = root / "jump_heights.json"
            calibration = _write_synthetic_wall_calibration(root / "wall_calibration.json")
            peak_frame = review_dir / "chunk_0012_00002.jpg"
            jump_heights.write_text(
                json.dumps(
                    {
                        "calibration_path": str(calibration),
                        "videos": [
                            {
                                "label": "clean",
                                "recording_dir": str(recording),
                                "max_jump_height_cm": 110.0,
                                "max_frame_path": str(peak_frame),
                                "max_bbox": "500,300,80,200",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            probe_rows = root / "probe_rows.json"
            probe_rows.write_text(
                json.dumps(
                    [
                        {
                            "raw_path": str(review_dir / f"chunk_0012_{index:05d}.jpg"),
                            "best_bbox": f"{500 + index},300,80,200",
                            "best_top_height_cm": 40.0 + index * 20,
                            "best_probability": 0.9,
                        }
                        for index in range(4)
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "out.mp4"
            encode_calls = []
            draw_calls = []

            def fake_draw(image, output, **kwargs):
                draw_calls.append(kwargs)
                image.save(output)

            def fake_encode(**kwargs):
                encode_calls.append(kwargs)
                kwargs["output_path"].write_bytes(b"x" * 4096)

            fps_up_calls = []

            def fake_fps_up(source_path, output_path, **kwargs):
                fps_up_calls.append((source_path, output_path, kwargs))
                output_path.write_bytes(b"x" * 4096)
                return {"method": "fake_fps_up", "fps": kwargs["fps"], "path": str(output_path)}

            def fake_extract(_chunks, frames_dir):
                paths = []
                for index in range(120):
                    path = frames_dir / f"chunk_0012_{index:05d}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (1280, 720), "white").save(path)
                    paths.append(path)
                return paths

            with (
                mock.patch.object(recorder, "_ffprobe_frame_count", return_value=120),
                mock.patch.object(recorder, "_ffprobe_duration_seconds", return_value=4.8),
                mock.patch.object(recorder, "_frame_offset_seconds_for_index", return_value=2.0),
                mock.patch.object(recorder, "_chunk_frame_rate", return_value=25.0),
                mock.patch.object(recorder, "_extract_exact_video_frames", side_effect=fake_extract),
                mock.patch.object(recorder, "_draw_rescore_overlay_frame", side_effect=fake_draw),
                mock.patch.object(recorder, "_encode_frame_sequence", side_effect=fake_encode),
                mock.patch.object(recorder, "_enhance_vertical_clip_for_social", side_effect=fake_fps_up),
                mock.patch.object(recorder, "_neural_upscale_clip_with_realesrgan", return_value=None),
            ):
                metadata = recorder._render_publishable_rescore_jump(  # noqa: SLF001
                    jump_heights_path=jump_heights,
                    output_path=output_path,
                    probe_rows_path=probe_rows,
                    min_duration_seconds=8.0,
                    min_jump_count=1,
                )

        self.assertEqual(encode_calls[0]["input_fps"], 25.0)
        self.assertEqual(encode_calls[0]["fps"], 25)
        self.assertEqual(encode_calls[0]["vf"], "scale=1080:1920")
        self.assertEqual(encode_calls[0]["metadata"]["comment"], "TikTok-ready clean-source Cat Projector jump render.")
        self.assertNotIn("model", encode_calls[0]["metadata"]["comment"].lower())
        self.assertNotIn("probability", encode_calls[0]["metadata"]["comment"].lower())
        self.assertEqual(len(draw_calls), 101)
        self.assertEqual(
            {
                tuple(round(value, 2) for value in call["close_crop_bounds"])
                for call in draw_calls
                if call["close_crop_bounds"] is not None
            },
            {(17.78, -468.73, 942.58, 1675.71)},
        )
        self.assertEqual(
            [round(value, 2) for value in metadata["timeline"]["camera_plan"]["screen_bbox"]],
            [40.22, 57.27, 937.92, 680.97],
        )
        self.assertEqual(
            [round(value, 2) for value in metadata["timeline"]["camera_plan"]["content_bbox"]],
            [40.22, 57.27, 937.92, 680.97],
        )
        self.assertEqual(metadata["timeline"]["camera_plan"]["zoom_scope"], "global_screen_plus_cat_polygons")
        self.assertEqual({call["session_title"] for call in draw_calls}, {"SHER JUMPS"})
        self.assertEqual({call["session_subtitle"] for call in draw_calls}, {"Monday, 18 May 2026"})
        self.assertEqual(
            metadata["timeline"]["presentation"],
            {"subject": "Sher", "title": "SHER JUMPS", "recorded_on": "Monday, 18 May 2026"},
        )
        self.assertEqual(metadata["timeline"]["render_input_fps"], 25.0)
        self.assertEqual(metadata["timeline"]["output_fps"], 25)
        self.assertEqual(metadata["timeline"]["source"], "source_chunk_exact_frames_probe_rows_by_frame_id")
        self.assertEqual(metadata["timeline"]["jump_event_count"], 1)
        self.assertEqual(metadata["timeline"]["source_frames"], 101)
        self.assertEqual(metadata["timeline"]["motion_seconds"], 4.04)
        self.assertEqual(metadata["timeline"]["tracked_frames"], 4)
        self.assertEqual(metadata["timeline"]["missing_probe_frames"], 97)
        self.assertEqual(Path(metadata["source_integrity"]["chunk_path"]).name, "chunk_0012.mp4")
        self.assertEqual(Path(metadata["source_integrity"]["frame_path"]).name, "chunk_0012_00002.jpg")
        self.assertEqual(metadata["source_integrity"]["chunk"], 12)
        self.assertEqual(metadata["source_integrity"]["frame_index"], 2)
        self.assertEqual(metadata["timeline"]["jump_events"][0]["start_frame"], "chunk_0012_00000.jpg")
        self.assertEqual(metadata["timeline"]["jump_events"][0]["end_frame"], "chunk_0012_00100.jpg")
        self.assertEqual(metadata["timeline"]["jump_events"][0]["pre_available_seconds"], 2.0)
        self.assertEqual(metadata["timeline"]["jump_events"][0]["post_available_seconds"], 2.0)
        self.assertEqual(metadata["enhancement"]["fps_upscale"]["fps"], 40)
        self.assertEqual(metadata["enhancement"]["neural_upscale"], None)
        self.assertEqual(metadata["enhancement"]["rife_fps_up"]["enabled"], False)
        self.assertIn("avoid_ghost_contours", metadata["enhancement"]["rife_fps_up"]["reason"])
        self.assertEqual(len(fps_up_calls), 1)
        self.assertFalse(metadata["timeline"]["min_duration_met"])
        self.assertEqual(metadata["timeline"]["results_hold_frames"], 0)
        self.assertEqual(metadata["timeline"]["results_hold_seconds"], 0.0)

    def test_publish_overlay_keeps_record_board_inside_tiktok_safe_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "frame.jpg"
            image = Image.new("RGB", (1280, 720), "gray")
            sample = recorder.live_jump.LiveJumpSample(
                timestamp="2026-06-07T00:00:00+04:00",
                source="frame.jpg",
                candidate_found=True,
                top_x_px=600.0,
                top_y_px=300.0,
                wall_x_mm=0.0,
                top_height_mm=1100.0,
                top_height_cm=110.0,
                component_bbox_xyxy=(560, 300, 640, 520),
                triggered=True,
                image_path="frame.jpg",
                source_time_s=1.0,
            )
            track_state = recorder.live_jump.update_track_state(
                recorder.live_jump.JumpTrackState(), sample, max_points=90
            )
            holds = recorder.live_jump.update_top_jump_holds((), image, sample, min_height_cm=45.0)

            recorder._draw_rescore_overlay_frame(  # noqa: SLF001
                image,
                output,
                video={
                    "max_bbox": "560,300,80,220",
                    "max_jump_height_cm": 110.0,
                    "best_measurement_point": {"image_x": 600.0, "image_y": 300.0, "wall_x_cm": 0.0},
                },
                sample=sample,
                track_state=track_state,
                top_holds=holds,
                active=True,
            )

            rendered = Image.open(output).convert("RGB")
            top_strip = rendered.crop((0, 0, rendered.width, 45))
            left_strip = rendered.crop((0, 0, 45, rendered.height))

        self.assertLess(
            sum(1 for red, green, blue in top_strip.getdata() if red > 180 or green > 180 or blue > 180),
            30,
        )
        self.assertLess(
            sum(1 for red, green, blue in left_strip.getdata() if red > 180 or green > 180 or blue > 180),
            30,
        )

    def test_publishable_selection_skips_low_fps_source_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recording"
            review_dir = recording / "review_frames"
            review_dir.mkdir(parents=True)
            high_frame = review_dir / "chunk_0001_00000.jpg"
            low_frame = review_dir / "chunk_0002_00000.jpg"
            Image.new("RGB", (32, 32), "white").save(high_frame)
            Image.new("RGB", (32, 32), "white").save(low_frame)
            (recording / "chunk_0001.mp4").write_bytes(b"fake high")
            (recording / "chunk_0002.mp4").write_bytes(b"fake low")
            jump_heights = root / "jump_heights.json"
            jump_heights.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "label": "low-fps-taller",
                                "recording_dir": str(recording),
                                "max_jump_height_cm": 142.0,
                                "max_frame_path": str(low_frame),
                            },
                            {
                                "label": "high-fps-shorter",
                                "recording_dir": str(recording),
                                "max_jump_height_cm": 122.2,
                                "max_frame_path": str(high_frame),
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_duration(path):
                return 1.0 if path.name == "chunk_0002.mp4" else 10.0 / 24.0

            with (
                mock.patch.object(recorder, "_ffprobe_frame_count", return_value=10),
                mock.patch.object(recorder, "_ffprobe_duration_seconds", side_effect=fake_duration),
                mock.patch.object(recorder, "_frame_offset_seconds_for_index", return_value=0.0),
            ):
                candidate, suspicious = recorder._select_publishable_rescore_jump(  # noqa: SLF001
                    jump_heights,
                    min_source_fps=24.0,
                )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.video["label"], "high-fps-shorter")
        self.assertIn("source_fps_below_publishable_min", suspicious[0].integrity.reason)

    def test_publishable_render_requires_three_clean_jump_candidates_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recording"
            review_dir = recording / "review_frames"
            review_dir.mkdir(parents=True)
            videos = []
            for index, height in enumerate((122.0, 118.0), start=1):
                frame = review_dir / f"chunk_{index:04d}_00010.jpg"
                Image.new("RGB", (32, 32), "white").save(frame)
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"fake chunk")
                videos.append(
                    {
                        "label": f"jump-{index}",
                        "recording_dir": str(recording),
                        "max_jump_height_cm": height,
                        "max_frame_path": str(frame),
                    }
                )
            jump_heights = root / "jump_heights.json"
            jump_heights.write_text(json.dumps({"videos": videos}) + "\n", encoding="utf-8")

            with (
                mock.patch.object(recorder, "_ffprobe_frame_count", return_value=100),
                mock.patch.object(recorder, "_ffprobe_duration_seconds", return_value=4.0),
                mock.patch.object(recorder, "_frame_offset_seconds_for_index", return_value=1.0),
                self.assertRaisesRegex(RuntimeError, "need at least 3"),
            ):
                recorder._render_publishable_rescore_jump(  # noqa: SLF001
                    jump_heights_path=jump_heights,
                    output_path=root / "out.mp4",
                )

    def test_record_rejects_sparse_successful_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording_dir = root / "recording"

            def fake_run(command, stdout=None, stderr=None, check=False, **_kwargs):
                Path(command[-1]).write_bytes(b"x" * 8192)
                return SimpleNamespace(returncode=0)

            with (
                mock.patch.object(recorder.subprocess, "run", side_effect=fake_run),
                mock.patch.object(recorder, "_ffprobe_frame_count", return_value=20),
                mock.patch.object(recorder, "_ffprobe_duration_seconds", return_value=5.0),
                mock.patch.object(recorder.time, "monotonic", side_effect=[0.0, 0.0, 1.0]),
            ):
                recorder.record(
                    argparse.Namespace(
                        recording_dir=recording_dir,
                        camera_url="http://camera/video?fps=30",
                        duration_seconds=0.1,
                        segment_seconds=5,
                    )
                )

            self.assertFalse((recording_dir / "chunk_0000.mp4").exists())
            self.assertTrue((recording_dir / "bad_chunks" / "chunk_0000.bad.mp4").exists())
            rows = [
                json.loads(line) for line in (recording_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["effective_fps"], 4.0)
            self.assertFalse(rows[0]["finalized"])
            self.assertIn("low_effective_fps", rows[0]["reject_reason"])

    def test_trajectory_close_crop_bounds_cover_full_cat_path(self) -> None:
        bounds = recorder._trajectory_close_crop_bounds(  # noqa: SLF001
            image_size=(1280, 720),
            trajectory_bbox=(320.0, 250.0, 760.0, 690.0),
            output_size=(1080, 1920),
        )

        self.assertIsNotNone(bounds)
        assert bounds is not None
        left, top, width, height = bounds
        self.assertLessEqual(left, 320.0)
        self.assertLessEqual(top, 250.0)
        self.assertGreaterEqual(left + width, 760.0)
        self.assertGreaterEqual(top + height, 690.0)
        self.assertLess(width, 720.0)
        self.assertAlmostEqual(width / height, 1080 / 1920, delta=0.02)

    def test_publishable_crop_uses_screen_and_cat_content_bounds(self) -> None:
        screen = recorder._projector_screen_bbox((1280, 720))  # noqa: SLF001
        cat_path = (341.0, 277.0, 921.0, 720.0)
        content = recorder._bbox_union(screen, cat_path)  # noqa: SLF001
        bounds = recorder._trajectory_close_crop_bounds(  # noqa: SLF001
            image_size=(1280, 720),
            trajectory_bbox=content,
            output_size=(1080, 1920),
        )

        self.assertIsNotNone(bounds)
        assert bounds is not None
        left, _top, width, _height = bounds
        self.assertLessEqual(left, screen[0])
        self.assertLess(960.0, left + width)
        self.assertLess(left + width, 980.0)

    def test_telegram_vertical_render_uses_shared_publish_crop_and_presentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260518T184855_sher"
            recording.mkdir(parents=True)
            chunk = recording / "chunk_0000.mp4"
            chunk.write_bytes(b"x" * 4096)
            output = root / "jump.mp4"
            vertical = root / "jump_tiktok.mp4"

            def fake_extract(_chunks, frames_dir, **_kwargs):
                paths = []
                for index, box in enumerate([(610, 560, 690, 710), (1000, 420, 1070, 680)], start=1):
                    path = frames_dir / f"chunk_0000_{index:05d}.jpg"
                    image = Image.new("RGB", (1280, 720), "white")
                    ImageDraw.Draw(image).rectangle(box, fill="black")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path)
                    paths.append(path)
                return paths

            samples = [
                recorder.live_jump.LiveJumpSample(
                    timestamp="2026-05-18T18:48:55+04:00",
                    source="chunk_0000_00001.jpg",
                    candidate_found=True,
                    top_height_cm=92.0,
                    component_bbox_xyxy=(610, 560, 690, 710),
                ),
                recorder.live_jump.LiveJumpSample(
                    timestamp="2026-05-18T18:48:55+04:00",
                    source="chunk_0000_00002.jpg",
                    candidate_found=True,
                    top_height_cm=126.0,
                    component_bbox_xyxy=(1000, 420, 1070, 680),
                ),
            ]
            vertical_calls = []

            def fake_sample(*_args, **_kwargs):
                return samples.pop(0)

            def fake_render_vertical(image, _sample, output_path, **kwargs):
                vertical_calls.append(kwargs)
                image.save(output_path)
                return recorder.live_jump.VerticalCropState(center_x_px=640, center_y_px=360)

            def fake_render_annotation(image, _sample, output_path, **_kwargs):
                image.save(output_path)

            def fake_encode(**kwargs):
                kwargs["output_path"].write_bytes(b"x" * 4096)

            with (
                mock.patch.object(recorder, "default_model_available", return_value=False),
                mock.patch.object(recorder, "_extract_video_frames", side_effect=fake_extract),
                mock.patch.object(recorder, "_room_background_for_frames", return_value=None),
                mock.patch.object(recorder, "_projector_source_video_for_chunks", return_value=None),
                mock.patch.object(recorder, "_sample_render_frame", side_effect=fake_sample),
                mock.patch.object(recorder.live_jump, "_read_calibration", return_value=(None, None)),
                mock.patch.object(recorder.live_jump, "render_annotation", side_effect=fake_render_annotation),
                mock.patch.object(recorder.live_jump, "render_vertical_annotation", side_effect=fake_render_vertical),
                mock.patch.object(recorder, "_extract_exact_video_frames", return_value=[]),
                mock.patch.object(recorder, "_queue_continuity_hard_cases", return_value=[]),
                mock.patch.object(recorder, "_encode_frame_sequence", side_effect=fake_encode),
            ):
                stats = recorder._build_annotated_jump_clip(  # noqa: SLF001
                    [chunk],
                    output,
                    vertical_output_path=vertical,
                    fps=20,
                    min_jump_height_cm=85.0,
                )

            self.assertEqual(stats["vertical_clip"], str(vertical))
            frame_calls = vertical_calls[:2]
            self.assertEqual(len(frame_calls), 2)
            self.assertEqual({call["session_title"] for call in frame_calls}, {"SHER JUMPS"})
            self.assertEqual({call["session_subtitle"] for call in frame_calls}, {"Monday, 18 May 2026"})
            self.assertEqual({call["publish_safe_zone"] for call in frame_calls}, {True})
            self.assertEqual(len({tuple(call["close_crop_bounds"]) for call in frame_calls}), 1)
            close_left, _close_top, close_width, _close_height = frame_calls[0]["close_crop_bounds"]
            self.assertLessEqual(close_left, 40.22)
            self.assertGreaterEqual(close_left + close_width, 1070.0)

    def test_realesrgan_timeout_returns_fallback_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.mp4"
            output = root / "out.mp4"
            fake_bin = root / "realesrgan-ncnn-vulkan"
            fake_model_dir = root / "models"
            source.write_bytes(b"x" * 4096)
            fake_bin.write_bytes(b"fake")
            fake_model_dir.mkdir()

            def fake_run(command, **kwargs):
                if command[0] == "ffmpeg" and str(command[-1]).endswith("frame_%08d.png"):
                    frame_path = Path(command[-1].replace("%08d", "00000001"))
                    frame_path.parent.mkdir(parents=True, exist_ok=True)
                    frame_path.write_bytes(b"png")
                    return mock.Mock(returncode=0, stderr="", stdout="")
                if command[0] == str(fake_bin):
                    raise recorder.subprocess.TimeoutExpired(command, timeout=0.01)
                return mock.Mock(returncode=0, stderr="", stdout="")

            with (
                mock.patch.object(recorder, "DEFAULT_REALESRGAN_BIN", fake_bin),
                mock.patch.object(recorder, "DEFAULT_REALESRGAN_MODEL_DIR", fake_model_dir),
                mock.patch.object(recorder.subprocess, "run", side_effect=fake_run),
            ):
                metadata = recorder._neural_upscale_clip_with_realesrgan(  # noqa: SLF001
                    source,
                    output,
                    fps=10,
                    time_budget_seconds=0.01,
                )

        self.assertEqual(metadata["status"], "timeout_fallback")
        self.assertEqual(metadata["fallback_reason"], "realesrgan_timeout")
        self.assertEqual(metadata["stage"], "post_overlay")

    def test_yolo_seg_expected_alert_is_not_confirmed_by_legacy_catboost_sample(self) -> None:
        sample = recorder.live_jump.LiveJumpSample(
            timestamp="2026-06-06T00:00:00+04:00",
            source="/tmp/render/chunk_0124_00023.jpg",
            candidate_found=True,
            top_height_cm=142.0,
            component_bbox_xyxy=(556, 285, 638, 500),
            detector_model_path="/models/cat_projector_candidate_detector_v1.cbm",
        )
        expected = {
            "chunk": 124,
            "offset_seconds": 5.5,
            "detector_backend": "ultralytics_yolo_segmentation",
            "model_path": "/models/sher-yolo-seg-contour-gentle-20260606.pt",
            "reason": "yolo_segmentation_rescore_clean_source",
        }

        self.assertFalse(recorder._sample_matches_alert_record(sample, expected, sample_fps=4))  # noqa: SLF001

        yolo_sample = recorder.live_jump.LiveJumpSample(
            timestamp="2026-06-06T00:00:00+04:00",
            source="/tmp/render/chunk_0124_00023.jpg",
            candidate_found=True,
            top_height_cm=142.0,
            component_bbox_xyxy=(556, 285, 638, 500),
            detector_model_path="/models/sher-yolo-seg-contour-gentle-20260606.pt",
        )
        self.assertTrue(recorder._sample_matches_alert_record(yolo_sample, expected, sample_fps=4))  # noqa: SLF001

    def test_telegram_notification_sends_enhanced_vertical_clip_first_when_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260519T120000_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-19T12:00:00+04:00",
                        "ha_session_id": "20260519T120000-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(4):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            highlight = recorder.JumpHighlight(
                chunks=[recording / "chunk_0001.mp4", recording / "chunk_0002.mp4"],
                max_height_cm=126.0,
                max_record={"chunk": 1, "height_cm": 126.0, "reason": "calibrated_high_jump"},
                candidate_count=3,
            )

            real_datetime = recorder.datetime
            build_kwargs = []

            def fake_build_clip(_chunks, path, **kwargs):
                build_kwargs.append(kwargs)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"horizontal annotated")
                vertical = kwargs.get("vertical_output_path")
                if vertical is not None:
                    vertical.parent.mkdir(parents=True, exist_ok=True)
                    vertical.write_bytes(b"vertical annotated")
                return {
                    "max_height_cm": 126.0,
                    "candidate_frame_count": 20,
                    "frame_count": 60,
                    "vertical_clip": str(vertical),
                }

            def fake_enhance(source, output, **kwargs):
                output.write_bytes(b"enhanced vertical")
                return {
                    "path": str(output),
                    "source_path": str(source),
                    "fps": recorder.DEFAULT_TELEGRAM_ENHANCED_FPS,
                    "resolution": "1080x1920",
                    "method": "ffmpeg_minterpolate_zscale_unsharp",
                    "metadata": {"title": kwargs["title"]},
                }

            with (
                mock.patch.object(recorder, "_build_annotated_jump_clip", side_effect=fake_build_clip),
                mock.patch.object(recorder, "_enhance_vertical_clip_for_social", side_effect=fake_enhance),
                mock.patch.object(recorder, "_send_telegram_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(recorder, "_select_jump_highlight_clip_window", return_value=highlight),
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    return_value=recorder.CameraHumanCheck(False, "no_camera_human_detected", checked_frames=2),
                ),
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-19T12:10:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                payload = recorder._maybe_send_telegram_notification(  # noqa: SLF001
                    root,
                    max_age_seconds=3600,
                    window_chunks=2,
                    max_clips=2,
                )

        sent_clips = send_notification.call_args.kwargs["clips"]
        self.assertEqual(payload["telegram_action"], "sent")
        self.assertEqual(sent_clips[0].name, "sher_jump_highlight_tiktok.mp4")
        self.assertEqual(sent_clips[1].name, "sher_jump_highlight_vertical.mp4")
        self.assertEqual(sent_clips[2].name, "sher_jump_highlight.mp4")
        self.assertEqual(Path(payload["clips"][0]).name, "sher_jump_highlight_tiktok.mp4")
        self.assertEqual(
            payload["render_stats"]["enhanced_vertical_clip"]["method"], "ffmpeg_minterpolate_zscale_unsharp"
        )
        self.assertEqual(build_kwargs[0]["training_queue_root"], root / "detector-training-queue")

    def test_continuity_hard_case_queue_captures_in_frame_disappear_reappear(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frames = root / "frames"
            queue_root = root / "queue"
            frames.mkdir()
            frame_paths = []
            for index in range(3):
                path = frames / f"frame_{index:06d}.jpg"
                Image.new("RGB", (320, 180), "white").save(path)
                frame_paths.append(path)
            seen_before = recorder.live_jump.LiveJumpSample(
                timestamp="2026-05-19T00:00:00+04:00",
                source=str(frame_paths[0]),
                candidate_found=True,
                top_x_px=155,
                top_y_px=90,
                component_bbox_xyxy=(130, 60, 180, 150),
            )
            missed = recorder.live_jump.LiveJumpSample(
                timestamp="2026-05-19T00:00:00+04:00",
                source=str(frame_paths[1]),
                candidate_found=False,
            )
            seen_after = recorder.live_jump.LiveJumpSample(
                timestamp="2026-05-19T00:00:00+04:00",
                source=str(frame_paths[2]),
                candidate_found=True,
                top_x_px=164,
                top_y_px=92,
                component_bbox_xyxy=(138, 62, 188, 152),
            )

            queued = recorder._queue_continuity_hard_cases(  # noqa: SLF001
                frame_paths=frame_paths,
                samples=[seen_before, missed, seen_after],
                queue_root=queue_root,
            )

            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0]["label_status"], "needs_human_bbox")
            package_dirs = list(queue_root.glob("continuity-hard-cases-*"))
            self.assertEqual(len(package_dirs), 1)
            self.assertTrue((package_dirs[0] / queued[0]["frame_relpath"]).exists())

    def test_append_results_hold_frames_repeats_summary_until_video_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir)
            image = Image.new("RGB", (320, 180), "gray")
            hold_image = Image.new("RGB", (80, 140), "white")
            ImageDraw.Draw(hold_image).rectangle((25, 20, 55, 120), fill="black")
            holds = (recorder.live_jump.JumpPeakHold(height_cm=111.3, image=hold_image, seen_count=2),)

            appended = recorder._append_results_hold_frames(  # noqa: SLF001
                frame_dir=frame_dir,
                frame_index=7,
                image=image,
                top_holds=holds,
                fps=4,
                seconds=2.0,
                output_size=(180, 320),
            )

            self.assertEqual(appended, 8)
            frames = sorted(frame_dir.glob("frame_*.jpg"))
            self.assertEqual(len(frames), 8)
            self.assertEqual(frames[0].name, "frame_000007.jpg")
            self.assertEqual(frames[-1].name, "frame_000014.jpg")
            self.assertEqual(Image.open(frames[0]).size, (180, 320))

    def test_cat_food_reward_dispenses_once_per_session_and_hour(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_recording = root / "recordings" / "20260518T120000_session"
            second_recording = root / "recordings" / "20260518T120030_session_retry"
            third_recording = root / "recordings" / "20260518T120100_next_session"
            first_recording.mkdir(parents=True)
            second_recording.mkdir(parents=True)
            third_recording.mkdir(parents=True)

            with (
                mock.patch.object(recorder, "_load_ha_token", return_value="token") as load_token,  # noqa: SLF001
                mock.patch.object(recorder, "_ha_call_service", return_value={"response": []}) as call_service,  # noqa: SLF001
            ):
                first = recorder._maybe_dispense_cat_food_reward(  # noqa: SLF001
                    root,
                    recording_dir=first_recording,
                    manifest={"ha_session_id": "session-1"},
                    height_cm=151.2,
                    source="test",
                    ha_url="http://ha.local",
                    entity_id=REWARD_ENTITY_ID,
                )
                same_session = recorder._maybe_dispense_cat_food_reward(  # noqa: SLF001
                    root,
                    recording_dir=second_recording,
                    manifest={"ha_session_id": "session-1"},
                    height_cm=166.0,
                    source="test",
                    ha_url="http://ha.local",
                    entity_id=REWARD_ENTITY_ID,
                )
                cooldown = recorder._maybe_dispense_cat_food_reward(  # noqa: SLF001
                    root,
                    recording_dir=third_recording,
                    manifest={"ha_session_id": "session-2"},
                    height_cm=170.0,
                    source="test",
                    ha_url="http://ha.local",
                    entity_id=REWARD_ENTITY_ID,
                )

        self.assertEqual(first["cat_food_reward_action"], "dispensed")
        self.assertEqual(same_session["cat_food_reward_action"], "already_rewarded_session")
        self.assertEqual(cooldown["cat_food_reward_action"], "cooldown")
        self.assertEqual(load_token.call_count, 1)
        call_service.assert_called_once_with(
            "http://ha.local",
            "token",
            "button",
            "press",
            {"entity_id": REWARD_ENTITY_ID},
        )

    def test_cat_food_reward_ignores_low_jump(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260518T120000_session"
            recording.mkdir(parents=True)

            with mock.patch.object(recorder, "_ha_call_service") as call_service:  # noqa: SLF001
                payload = recorder._maybe_dispense_cat_food_reward(  # noqa: SLF001
                    root,
                    recording_dir=recording,
                    manifest={"ha_session_id": "session-1"},
                    height_cm=149.9,
                    source="test",
                    ha_url="http://ha.local",
                    entity_id=REWARD_ENTITY_ID,
                )

        self.assertEqual(payload["cat_food_reward_action"], "below_threshold")
        call_service.assert_not_called()

    def test_cat_food_reward_records_failed_dispense_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260518T120000_session"
            recording.mkdir(parents=True)

            with (
                mock.patch.object(recorder, "_load_ha_token", return_value="token"),  # noqa: SLF001
                mock.patch.object(recorder, "_ha_call_service", side_effect=RuntimeError("HA offline")),  # noqa: SLF001
            ):
                payload = recorder._maybe_dispense_cat_food_reward(  # noqa: SLF001
                    root,
                    recording_dir=recording,
                    manifest={"ha_session_id": "session-1"},
                    height_cm=151.0,
                    source="test",
                    ha_url="http://ha.local",
                    entity_id=REWARD_ENTITY_ID,
                )
                repeat = recorder._maybe_dispense_cat_food_reward(  # noqa: SLF001
                    root,
                    recording_dir=recording,
                    manifest={"ha_session_id": "session-1"},
                    height_cm=160.0,
                    source="test",
                    ha_url="http://ha.local",
                    entity_id=REWARD_ENTITY_ID,
                )

        self.assertEqual(payload["cat_food_reward_action"], "dispense_failed")
        self.assertIn("HA offline", payload["error"])
        self.assertEqual(repeat["cat_food_reward_action"], "already_rewarded_recording")

    def test_telegram_notification_skips_when_render_has_no_confirmed_jump(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260516T120000_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-16T12:00:00+04:00",
                        "ha_session_id": "20260516T120000-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(4):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            real_datetime = recorder.datetime
            highlight = recorder.JumpHighlight(
                chunks=[recording / "chunk_0001.mp4", recording / "chunk_0002.mp4"],
                max_height_cm=97.9,
                max_record={"chunk": 2, "height_cm": 97.9, "reason": "calibrated_high_jump"},
                candidate_count=1,
            )
            with (
                mock.patch.object(
                    recorder,
                    "_build_annotated_jump_clip",
                    return_value={"max_height_cm": None, "candidate_frame_count": 1, "frame_count": 60},
                ),
                mock.patch.object(recorder, "_send_telegram_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(recorder, "_select_jump_highlight_clip_window", return_value=highlight),
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    return_value=recorder.CameraHumanCheck(False, "no_camera_human_detected", checked_frames=2),
                ),
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-16T12:10:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                payload = recorder._maybe_send_telegram_notification(
                    root,
                    max_age_seconds=3600,
                    window_chunks=2,
                    max_clips=2,
                )

        self.assertEqual(payload["telegram_action"], "no_confirmed_rendered_jump")
        send_notification.assert_not_called()

    def test_telegram_notification_skips_when_rendered_alert_height_falls_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260606T031200_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-06-06T03:12:00+04:00",
                        "ha_session_id": "20260606T031200-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(4):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            high_record = {
                "chunk": 2,
                "offset_seconds": 0.25,
                "height_cm": 89.0,
                "reason": "source_subtracted_calibrated_high_jump",
            }
            highlight = recorder.JumpHighlight(
                chunks=[recording / "chunk_0001.mp4", recording / "chunk_0002.mp4"],
                max_height_cm=89.0,
                max_record=high_record,
                candidate_count=1,
            )
            real_datetime = recorder.datetime
            with (
                mock.patch.object(
                    recorder,
                    "_build_annotated_jump_clip",
                    return_value={
                        "max_height_cm": 83.3,
                        "candidate_frame_count": 1,
                        "frame_count": 60,
                        "canonical_alert_record": high_record | {"height_cm": 83.3},
                    },
                ) as build_clip,
                mock.patch.object(recorder, "_send_telegram_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(recorder, "_select_jump_highlight_clip_window", return_value=highlight),
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    return_value=recorder.CameraHumanCheck(False, "no_camera_human_detected", checked_frames=2),
                ),
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-06-06T03:15:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                payload = recorder._maybe_send_telegram_notification(
                    root,
                    max_age_seconds=3600,
                    window_chunks=2,
                    max_clips=2,
                )
                marker = json.loads((recording / "telegram_notification.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["telegram_action"], "rendered_jump_below_threshold")
        self.assertEqual(marker["telegram_action"], "rendered_jump_below_threshold")
        self.assertEqual(build_clip.call_args.kwargs["expected_alert_record"], high_record)
        send_notification.assert_not_called()

    def test_telegram_notification_skips_camera_human_after_jump_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260517T203508_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-17T20:35:08+04:00",
                        "ha_session_id": "20260517T203500-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cat-tv/cats-tv-october-compilation-fMiAfGtt9rc-seed30.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(2):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)
            highlight = recorder.JumpHighlight(
                chunks=[recording / "chunk_0000.mp4", recording / "chunk_0001.mp4"],
                max_height_cm=111.3,
                max_record={"chunk": 1, "height_cm": 111.3, "reason": "calibrated_high_jump"},
                candidate_count=3,
            )

            real_datetime = recorder.datetime
            with (
                mock.patch.object(recorder, "_build_annotated_jump_clip") as build_clip,  # noqa: SLF001
                mock.patch.object(recorder, "_send_telegram_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(
                    recorder, "_select_jump_highlight_clip_window", return_value=highlight
                ) as select_highlight,
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    return_value=recorder.CameraHumanCheck(
                        True,
                        "camera_human_detected",
                        checked_frames=4,
                        frame="chunk_0001_00001.jpg",
                        confidence=0.99,
                        bbox_xyxy=(10, 20, 200, 420),
                    ),
                ),
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-17T20:40:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                payload = recorder._maybe_send_telegram_notification(  # noqa: SLF001
                    root,
                    max_age_seconds=3600,
                    window_chunks=2,
                    max_clips=2,
                )
                marker = json.loads((recording / "telegram_notification.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["telegram_action"], "camera_human_detected")
        self.assertEqual(payload["camera_human_check"]["reason"], "camera_human_detected")
        self.assertEqual(marker["selection_method"], "camera_human_detected")
        self.assertEqual(select_highlight.call_count, 1)
        self.assertEqual(build_clip.call_count, 0)
        self.assertEqual(send_notification.call_count, 0)

    def test_render_frame_rejects_low_dark_fallback_after_detector_rejection(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        ImageDraw.Draw(image).rectangle((60, 60, 85, 95), fill="black")
        homography = mock.Mock()
        floor_y_mm = 0.0
        rejected = recorder.live_jump.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=False,
            detector_cat_probability=0.30,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            detector_path = Path(tmpdir) / "cat_projector_candidate_detector_v1.cbm"
            detector_path.write_bytes(b"fake model")
            with (
                mock.patch.object(recorder, "DEFAULT_DETECTOR_MODEL_PATH", detector_path),
                mock.patch.object(recorder, "default_model_available", return_value=True),
                mock.patch.object(
                    recorder.live_jump,
                    "sample_image",
                    return_value=(rejected, None),
                ) as sample_image,
            ):
                sample = recorder._sample_render_frame(  # noqa: SLF001
                    image,
                    source="synthetic.jpg",
                    homography=homography,
                    floor_y_mm=floor_y_mm,
                    min_jump_height_cm=85.0,
                    detector_loaded_model=object(),
                    detector_loaded_metadata={"created_at": "test"},
                )

        self.assertFalse(sample.candidate_found)
        self.assertEqual(sample_image.call_count, 1)

    def test_render_frame_rejects_high_dark_fallback_after_detector_rejection(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        homography = mock.Mock()
        rejected = recorder.live_jump.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=False,
            detector_cat_probability=0.30,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            detector_path = Path(tmpdir) / "cat_projector_candidate_detector_v1.cbm"
            detector_path.write_bytes(b"fake model")
            with (
                mock.patch.object(recorder, "DEFAULT_DETECTOR_MODEL_PATH", detector_path),
                mock.patch.object(recorder, "default_model_available", return_value=True),
                mock.patch.object(
                    recorder.live_jump,
                    "sample_image",
                    return_value=(rejected, None),
                ) as sample_image,
            ):
                sample = recorder._sample_render_frame(  # noqa: SLF001
                    image,
                    source="synthetic.jpg",
                    homography=homography,
                    floor_y_mm=0.0,
                    min_jump_height_cm=85.0,
                    detector_loaded_model=object(),
                    detector_loaded_metadata={"created_at": "test"},
                )

        self.assertFalse(sample.candidate_found)
        self.assertEqual(sample_image.call_count, 1)

    def test_render_frame_keeps_source_subtracted_alert_over_legacy_full_frame_candidate(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        homography = mock.Mock()
        source_projector_object = recorder.live_jump.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=True,
            top_height_cm=94.2,
            component_bbox_xyxy=(45, 20, 72, 62),
            detector_cat_probability=0.91,
        )
        raw_cat = recorder.live_jump.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=True,
            top_height_cm=34.7,
            component_bbox_xyxy=(58, 70, 78, 98),
            detector_cat_probability=0.91,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            detector_path = Path(tmpdir) / "cat_projector_candidate_detector_v1.cbm"
            detector_path.write_bytes(b"fake model")
            with (
                mock.patch.object(recorder, "DEFAULT_DETECTOR_MODEL_PATH", detector_path),
                mock.patch.object(recorder, "default_model_available", return_value=True),
                mock.patch.object(
                    recorder,
                    "_sample_source_subtracted_frame",
                    return_value=source_projector_object,
                ),
                mock.patch.object(
                    recorder.live_jump,
                    "sample_image",
                    return_value=(raw_cat, None),
                ),
            ):
                sample = recorder._sample_render_frame(  # noqa: SLF001
                    image,
                    source="synthetic.jpg",
                    source_frame=Image.new("RGB", (120, 100), "white"),
                    homography=homography,
                    floor_y_mm=0.0,
                    min_jump_height_cm=85.0,
                    detector_loaded_model=object(),
                    detector_loaded_metadata={"created_at": "test"},
                )

        self.assertEqual(sample.component_bbox_xyxy, (45, 20, 72, 62))
        self.assertEqual(sample.top_height_cm, 94.2)

    def test_render_frame_uses_raw_components_only_without_detector_model(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        homography = mock.Mock()
        raw_cat = recorder.live_jump.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=True,
            top_height_cm=32.2,
            component_bbox_xyxy=(50, 60, 86, 96),
            detector_cat_probability=0.42,
        )

        with (
            mock.patch.object(recorder, "default_model_available", return_value=False),
            mock.patch.object(
                recorder.live_jump,
                "sample_image",
                return_value=(raw_cat, None),
            ) as sample_image,
        ):
            sample = recorder._sample_render_frame(  # noqa: SLF001
                image,
                source="synthetic.jpg",
                homography=homography,
                floor_y_mm=0.0,
                min_jump_height_cm=85.0,
                detector_loaded_model=None,
                detector_loaded_metadata=None,
            )

        self.assertTrue(sample.candidate_found)
        self.assertEqual(sample.top_height_cm, 32.2)
        self.assertEqual(sample.component_bbox_xyxy, (50, 60, 86, 96))
        self.assertEqual(sample_image.call_count, 1)

    def test_render_sample_original_overlay_keeps_frame_specific_detector_payload(self) -> None:
        sample = recorder.live_jump.LiveJumpSample(
            timestamp="2026-06-06T03:39:55+04:00",
            source="/tmp/render/chunk_0114_00096.jpg",
            candidate_found=True,
            top_x_px=618.0,
            top_y_px=378.0,
            wall_x_mm=-74.6,
            top_height_cm=115.7,
            component_area_px=6672,
            component_bbox_xyxy=(520, 368, 672, 508),
            detector_cat_probability=0.83,
            detector_model_path="/models/cat_projector_candidate_detector_v1.cbm",
        )

        overlay = recorder._render_sample_original_overlay(sample)  # noqa: SLF001

        self.assertEqual(
            overlay,
            {
                "frame_label": "chunk_0114_00096.jpg",
                "render_frame_label": "chunk_0114_00096.jpg",
                "source_frame_label": "chunk_0114_00096.jpg",
                "chunk": 114,
                "frame_index": 96,
                "offset_seconds": 4.75,
                "bbox": [520, 368, 672, 508],
                "area": 6672,
                "top_px": [618.0, 378.0],
                "height_cm": 115.7,
                "wall_x_mm": -74.6,
                "cat_probability": 0.83,
                "model_path": "/models/cat_projector_candidate_detector_v1.cbm",
                "reason": "telegram_render_detector",
            },
        )

    def test_render_frame_rejects_low_detector_probability_instead_of_soft_fallback(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        homography = mock.Mock()
        rejected = recorder.live_jump.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=False,
            detector_cat_probability=0.30,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            detector_path = Path(tmpdir) / "cat_projector_candidate_detector_v1.cbm"
            detector_path.write_bytes(b"fake model")
            with (
                mock.patch.object(recorder, "DEFAULT_DETECTOR_MODEL_PATH", detector_path),
                mock.patch.object(recorder, "default_model_available", return_value=True),
                mock.patch.object(
                    recorder.live_jump,
                    "sample_image",
                    return_value=(rejected, None),
                ) as sample_image,
            ):
                sample = recorder._sample_render_frame(  # noqa: SLF001
                    image,
                    source="synthetic.jpg",
                    homography=homography,
                    floor_y_mm=0.0,
                    min_jump_height_cm=85.0,
                    detector_loaded_model=object(),
                    detector_loaded_metadata={"created_at": "test"},
                )

        self.assertFalse(sample.candidate_found)
        self.assertEqual(sample_image.call_count, 1)

    def test_room_background_uses_bright_quantile_on_possible_cat_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recording"
            recording.mkdir()
            startup = Image.new("L", (100, 100), 180)
            ImageDraw.Draw(startup).rectangle((40, 70, 60, 95), fill=20)
            startup.convert("RGB").save(recording / "startup_projector_camera.jpg")

            frames_dir = root / "frames"
            frames_dir.mkdir()
            frame_dark = Image.new("L", (100, 100), 180)
            ImageDraw.Draw(frame_dark).rectangle((40, 70, 60, 95), fill=20)
            frame_bright = Image.new("L", (100, 100), 180)
            ImageDraw.Draw(frame_bright).rectangle((40, 70, 60, 95), fill=220)
            dark_path = frames_dir / "chunk_0001_00001.jpg"
            bright_path = frames_dir / "chunk_0001_00002.jpg"
            frame_dark.convert("RGB").save(dark_path)
            frame_bright.convert("RGB").save(bright_path)

            background = recorder._room_background_for_frames(  # noqa: SLF001
                [recording / "chunk_0001.mp4"],
                [dark_path, bright_path],
            )

        self.assertIsNotNone(background)
        assert background is not None
        self.assertGreater(background.convert("L").getpixel((50, 80)), 190)

    def test_source_subtracted_sample_uses_model_probability_order(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        high_probability_low_candidate = recorder.Candidate(
            bbox_xywh=(10, 70, 20, 20),
            top_x_px=20,
            top_y_px=70,
            area_px=400,
            source="test",
        )
        lower_probability_high_candidate = recorder.Candidate(
            bbox_xywh=(50, 15, 20, 20),
            top_x_px=60,
            top_y_px=15,
            area_px=400,
            source="test",
        )
        homography = mock.Mock()
        homography.project.side_effect = lambda _x, y: (0.0, 1000.0 - float(y) * 10.0)

        with (
            mock.patch.object(
                recorder,
                "detect_source_subtracted_candidate_components",
                return_value=[high_probability_low_candidate, lower_probability_high_candidate],
            ),
            mock.patch.object(
                recorder,
                "score_candidates",
                return_value=[
                    SimpleNamespace(
                        candidate=high_probability_low_candidate,
                        cat_probability=0.99,
                        model_path="model.cbm",
                    ),
                    SimpleNamespace(
                        candidate=lower_probability_high_candidate,
                        cat_probability=0.65,
                        model_path="model.cbm",
                    ),
                ],
            ),
        ):
            sample = recorder._sample_source_subtracted_frame(  # noqa: SLF001
                image,
                source="synthetic.jpg",
                source_frame=Image.new("RGB", (120, 100), "white"),
                room_background=None,
                residual_baseline=None,
                homography=homography,
                floor_y_mm=0.0,
                jump_threshold_cm=85.0,
                detector_loaded_model=object(),
                detector_loaded_metadata={"created_at": "test"},
                min_probability=0.55,
            )

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.component_bbox_xyxy, (10, 70, 30, 90))
        self.assertEqual(sample.detector_cat_probability, 0.99)
        self.assertEqual(sample.top_height_cm, 30.0)

    def test_telegram_notification_sends_jump_clip_with_height_caption(self) -> None:
        clips = [Path("/tmp/preview.mp4")]
        with mock.patch.object(recorder, "deliver_notification") as deliver:
            deliver.return_value = DeliveryResult(
                delivered=True,
                audited=False,
                source="cat_projector_learning_recorder",
                delivery_class="user_facing",
                reason="telegram_sent",
            )
            recorder._send_telegram_notification(  # noqa: SLF001
                recording_dir=Path("/tmp/recording"),
                manifest={
                    "ha_session_id": "20260517T015216-cats-tv",
                    "video_url": "http://example.invalid/cats-tv.mp4",
                },
                clips=clips,
                caption="Шер прыгнул на 111.3 см",
            )

        self.assertEqual(deliver.call_args.kwargs["delivery_class"], "user_facing")
        self.assertEqual(deliver.call_args.kwargs["media"], clips)
        self.assertEqual(deliver.call_args.args[0], "Шер прыгнул на 111.3 см")

    def test_telegram_notification_skips_when_no_high_jump_instead_of_random_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260516T120000_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-16T12:00:00+04:00",
                        "ha_session_id": "20260516T120000-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(4):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            real_datetime = recorder.datetime
            with (
                mock.patch.object(recorder, "_send_telegram_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(recorder, "_select_jump_highlight_clip_window", return_value=None),
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-16T12:10:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                payload = recorder._maybe_send_telegram_notification(  # noqa: SLF001
                    root,
                    max_age_seconds=3600,
                    window_chunks=2,
                    max_clips=2,
                )
                marker = json.loads((recording / "telegram_notification.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["telegram_action"], "no_jump_highlight")
            self.assertEqual(send_notification.call_count, 0)
            self.assertEqual(marker["selection_method"], "no_jump_highlight")

    def test_final_telegram_scan_uses_bounded_recent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260518T120000_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-18T12:00:00+04:00",
                        "ha_session_id": "20260518T120000-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(100):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            seen_chunks: list[Path] = []

            def fake_select(chunks, **_kwargs):
                seen_chunks.extend(chunks)
                return None

            real_datetime = recorder.datetime
            with (
                mock.patch.object(recorder, "_select_jump_highlight_clip_window", side_effect=fake_select),
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-18T12:10:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                payload = recorder._maybe_send_telegram_notification(  # noqa: SLF001
                    root,
                    max_age_seconds=3600,
                    window_chunks=6,
                    max_clips=3,
                )
                marker = json.loads((recording / "telegram_notification.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["telegram_action"], "no_jump_highlight")
        self.assertEqual(len(seen_chunks), recorder.DEFAULT_TELEGRAM_FINAL_SCAN_MAX_CHUNKS)
        self.assertEqual(seen_chunks[0].name, "chunk_0028.mp4")
        self.assertEqual(seen_chunks[-1].name, "chunk_0099.mp4")
        self.assertEqual(marker["scan_chunk_count"], recorder.DEFAULT_TELEGRAM_FINAL_SCAN_MAX_CHUNKS)

    def test_motion_records_select_diverse_clip_windows(self) -> None:
        chunks_by_index = {index: Path(f"chunk_{index:04d}.mp4") for index in range(20)}
        records = [
            {"chunk": 10, "score": 900},
            {"chunk": 12, "score": 850},
            {"chunk": 2, "score": 700},
            {"chunk": 18, "score": 600},
        ]

        windows = recorder._motion_records_to_clip_windows(  # noqa: SLF001
            chunks_by_index,
            records,
            window_chunks=4,
            max_clips=3,
            separation_chunks=4,
        )

        self.assertEqual(
            windows,
            [
                [Path("chunk_0008.mp4"), Path("chunk_0009.mp4"), Path("chunk_0010.mp4"), Path("chunk_0011.mp4")],
                [Path("chunk_0000.mp4"), Path("chunk_0001.mp4"), Path("chunk_0002.mp4"), Path("chunk_0003.mp4")],
                [Path("chunk_0016.mp4"), Path("chunk_0017.mp4"), Path("chunk_0018.mp4"), Path("chunk_0019.mp4")],
            ],
        )

    def test_live_telegram_notification_is_sent_during_active_recording_with_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260516T120000_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-16T12:00:00+04:00",
                        "ha_session_id": "20260516T120000-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(6):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            real_datetime = recorder.datetime
            highlight = recorder.JumpHighlight(
                chunks=[recording / f"chunk_{index:04d}.mp4" for index in range(6)],
                max_height_cm=98.4,
                max_record={"chunk": 3, "height_cm": 98.4, "reason": "calibrated_high_jump"},
                candidate_count=2,
            )
            with (
                mock.patch.object(recorder, "_build_annotated_jump_clip") as build_clip,  # noqa: SLF001
                mock.patch.object(recorder, "_enhance_vertical_clip_for_social") as enhance_clip,  # noqa: SLF001
                mock.patch.object(recorder, "_send_telegram_live_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(recorder, "_select_jump_highlight_clip_window", return_value=highlight),
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    return_value=recorder.CameraHumanCheck(False, "no_camera_human_detected", checked_frames=6),
                ),
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-16T12:10:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                def fake_build_clip(_chunks, path, **_kwargs):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"annotated")
                    vertical = _kwargs.get("vertical_output_path")
                    if vertical is not None:
                        vertical.parent.mkdir(parents=True, exist_ok=True)
                        vertical.write_bytes(b"vertical annotated")
                    return {"max_height_cm": 98.4, "candidate_frame_count": 9, "frame_count": 90}

                build_clip.side_effect = fake_build_clip

                def fake_enhance(_source, output, **_kwargs):
                    output.write_bytes(b"enhanced vertical")
                    return {"path": str(output), "method": "ffmpeg_minterpolate_zscale_unsharp"}

                enhance_clip.side_effect = fake_enhance

                payload = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260516T120000-cats-tv",
                    window_chunks=6,
                    cooldown_seconds=300,
                )
                second_payload = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260516T120000-cats-tv",
                    window_chunks=6,
                    cooldown_seconds=300,
                )

        self.assertEqual(payload["telegram_live_action"], "sent")
        self.assertEqual(second_payload["telegram_live_action"], "cooldown")
        self.assertEqual(send_notification.call_count, 1)
        self.assertEqual(payload["jump_caption"], "Шер прыгнул на 98.4 см")
        self.assertIn("\nРазметить: ", payload["caption"])
        self.assertIn("review_video_id=video.", payload["review_url"])
        self.assertIn("review_frame_label=chunk_0003_00000.jpg", payload["review_url"])
        self.assertEqual(Path(send_notification.call_args.kwargs["clip"]).name, "live_20260516T121000_tiktok.mp4")
        self.assertEqual(payload["enhanced_vertical_clip"], str(send_notification.call_args.kwargs["clip"]))
        self.assertEqual(
            payload["render_stats"]["enhanced_vertical_clip"]["method"], "ffmpeg_minterpolate_zscale_unsharp"
        )
        self.assertEqual(send_notification.call_args.kwargs["caption"], payload["caption"])

    def test_live_telegram_notification_triggers_food_reward_for_high_jump(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260518T120000_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-18T12:00:00+04:00",
                        "ha_session_id": "20260518T120000-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(6):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            real_datetime = recorder.datetime
            highlight = recorder.JumpHighlight(
                chunks=[recording / f"chunk_{index:04d}.mp4" for index in range(6)],
                max_height_cm=151.0,
                max_record={"chunk": 3, "height_cm": 151.0, "reason": "calibrated_high_jump"},
                candidate_count=2,
            )
            with (
                mock.patch.object(recorder, "_build_annotated_jump_clip") as build_clip,  # noqa: SLF001
                mock.patch.object(recorder, "_send_telegram_live_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(recorder, "_select_jump_highlight_clip_window", return_value=highlight),
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    return_value=recorder.CameraHumanCheck(False, "no_camera_human_detected", checked_frames=6),
                ),
                mock.patch.object(
                    recorder,
                    "_maybe_dispense_cat_food_reward",
                    return_value={"cat_food_reward_action": "dispensed"},
                ) as reward,
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-18T12:10:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                def fake_build_clip(_chunks, path, **_kwargs):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"annotated")
                    return {"max_height_cm": 151.0, "candidate_frame_count": 9, "frame_count": 90}

                build_clip.side_effect = fake_build_clip

                payload = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260518T120000-cats-tv",
                    window_chunks=6,
                    cooldown_seconds=300,
                    ha_url="http://ha.local",
                )
                marker = json.loads((recording / "telegram_live_notification.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["telegram_live_action"], "sent")
        self.assertEqual(payload["cat_food_reward_action"], "dispensed")
        self.assertEqual(marker["cat_food_reward"]["cat_food_reward_action"], "dispensed")
        reward.assert_called_once()
        self.assertEqual(reward.call_args.kwargs["height_cm"], 151.0)
        self.assertEqual(reward.call_args.kwargs["ha_url"], "http://ha.local")
        self.assertEqual(send_notification.call_count, 1)

    def test_live_telegram_send_failure_preserves_clip_for_retry_without_second_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260605T035211_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-06-05T03:52:11+04:00",
                        "ha_session_id": "20260605T035203-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(6):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            real_datetime = recorder.datetime
            highlight = recorder.JumpHighlight(
                chunks=[recording / f"chunk_{index:04d}.mp4" for index in range(6)],
                max_height_cm=152.2,
                max_record={"chunk": 3, "height_cm": 152.2, "reason": "calibrated_high_jump"},
                candidate_count=2,
            )

            def fake_build_clip(_chunks, path, **kwargs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"annotated")
                vertical = kwargs.get("vertical_output_path")
                if vertical is not None:
                    vertical.write_bytes(b"vertical")
                return {"max_height_cm": 152.2, "candidate_frame_count": 9, "frame_count": 90}

            send_results = [
                DeliveryResult(
                    delivered=False,
                    audited=True,
                    source="cat_projector_learning_recorder",
                    delivery_class="user_facing",
                    reason="telegram_send_failed",
                    returncode=127,
                    stderr="FileNotFoundError: notify command",
                ),
                DeliveryResult(
                    delivered=True,
                    audited=False,
                    source="cat_projector_learning_recorder",
                    delivery_class="user_facing",
                    reason="telegram_sent",
                ),
            ]
            with (
                mock.patch.object(recorder, "_build_annotated_jump_clip", side_effect=fake_build_clip),
                mock.patch.object(
                    recorder, "_send_telegram_live_notification", side_effect=send_results
                ) as send_notification,  # noqa: SLF001
                mock.patch.object(
                    recorder, "_select_jump_highlight_clip_window", return_value=highlight
                ) as select_highlight,
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    return_value=recorder.CameraHumanCheck(False, "no_camera_human_detected", checked_frames=6),
                ),
                mock.patch.object(
                    recorder,
                    "_maybe_dispense_cat_food_reward",
                    return_value={"cat_food_reward_action": "dispensed"},
                ) as reward,
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-06-05T04:09:01+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                first = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260605T035203-cats-tv",
                    window_chunks=6,
                    cooldown_seconds=300,
                )
                marker_after_failure = json.loads(
                    (recording / "telegram_live_notification.json").read_text(encoding="utf-8")
                )

                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-06-05T04:09:30+04:00"
                )
                cooldown = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260605T035203-cats-tv",
                    window_chunks=6,
                    cooldown_seconds=300,
                )

                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-06-05T04:11:30+04:00"
                )
                retried = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260605T035203-cats-tv",
                    window_chunks=6,
                    cooldown_seconds=300,
                )
                marker_after_retry = json.loads(
                    (recording / "telegram_live_notification.json").read_text(encoding="utf-8")
                )

        self.assertEqual(first["telegram_live_action"], "send_failed")
        self.assertEqual(marker_after_failure["telegram_live_action"], "send_failed")
        self.assertEqual(marker_after_failure["max_height_cm"], 152.2)
        self.assertEqual(marker_after_failure["cat_food_reward"]["cat_food_reward_action"], "dispensed")
        self.assertEqual(cooldown["telegram_live_action"], "scan_cooldown")
        self.assertEqual(cooldown["previous_telegram_live_action"], "send_failed")
        self.assertEqual(retried["telegram_live_action"], "sent")
        self.assertEqual(marker_after_retry["telegram_live_action"], "sent")
        self.assertEqual(reward.call_count, 1)
        self.assertEqual(send_notification.call_count, 2)
        self.assertEqual(select_highlight.call_count, 1)

    def test_live_telegram_notification_skips_when_render_has_no_confirmed_jump(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260516T120000_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-16T12:00:00+04:00",
                        "ha_session_id": "20260516T120000-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(6):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            real_datetime = recorder.datetime
            highlight = recorder.JumpHighlight(
                chunks=[recording / f"chunk_{index:04d}.mp4" for index in range(6)],
                max_height_cm=97.9,
                max_record={"chunk": 5, "height_cm": 97.9, "reason": "calibrated_high_jump"},
                candidate_count=1,
            )
            with (
                mock.patch.object(
                    recorder,
                    "_build_annotated_jump_clip",
                    return_value={"max_height_cm": None, "candidate_frame_count": 1, "frame_count": 90},
                ),
                mock.patch.object(recorder, "_send_telegram_live_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(recorder, "_select_jump_highlight_clip_window", return_value=highlight),
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    return_value=recorder.CameraHumanCheck(False, "no_camera_human_detected", checked_frames=6),
                ),
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-16T12:10:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                payload = recorder._maybe_send_live_telegram_notification(
                    root,
                    ha_session_id="20260516T120000-cats-tv",
                    window_chunks=6,
                    cooldown_seconds=300,
                )

        self.assertEqual(payload["telegram_live_action"], "no_confirmed_rendered_jump")
        send_notification.assert_not_called()

    def test_live_telegram_notification_sends_attachment_with_height_caption(self) -> None:
        clip = Path("/tmp/live.mp4")
        with mock.patch.object(recorder, "deliver_notification") as deliver:
            deliver.return_value = DeliveryResult(
                delivered=True,
                audited=False,
                source="cat_projector_learning_recorder",
                delivery_class="user_facing",
                reason="telegram_sent",
            )
            recorder._send_telegram_live_notification(  # noqa: SLF001
                recording_dir=Path("/tmp/recording"),
                manifest={
                    "ha_session_id": "20260517T015216-cats-tv",
                    "video_url": "http://example.invalid/cats-tv.mp4",
                },
                clip=clip,
                caption="Шер прыгнул на 98.4 см",
            )

        self.assertEqual(deliver.call_args.kwargs["delivery_class"], "user_facing")
        self.assertEqual(deliver.call_args.kwargs["media"], [clip])

    def test_live_telegram_no_curated_highlight_sets_cooldown_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260516T120000_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-16T12:00:00+04:00",
                        "ha_session_id": "20260516T120000-cats-tv",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cats-tv.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(6):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)

            real_datetime = recorder.datetime
            with (
                mock.patch.object(recorder, "_send_telegram_live_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(
                    recorder, "_select_jump_highlight_clip_window", return_value=None
                ) as select_highlight,
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-16T12:10:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                payload = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260516T120000-cats-tv",
                    window_chunks=6,
                    cooldown_seconds=300,
                )
                second_payload = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260516T120000-cats-tv",
                    window_chunks=6,
                    cooldown_seconds=300,
                )
                marker = json.loads((recording / "telegram_live_notification.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["telegram_live_action"], "no_jump_highlight")
        self.assertEqual(second_payload["telegram_live_action"], "scan_cooldown")
        self.assertEqual(marker["selection_method"], "no_jump_highlight")
        self.assertEqual(send_notification.call_count, 0)
        self.assertEqual(select_highlight.call_count, 1)

    def test_live_telegram_camera_human_marker_uses_cat_tv_play_policy_for_rescan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260517T203508_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-05-17T20:35:08+04:00",
                        "ha_session_id": "20260517T203500-cats-tv-october-compilation-fMiAfGtt9rc-seed30",
                        "recording_dir": str(recording),
                        "video_url": "http://example.invalid/cat-tv/cats-tv-october-compilation-fMiAfGtt9rc-seed30.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (recording / "verified.json").write_text("{}\n", encoding="utf-8")
            for index in range(6):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)
            highlight = recorder.JumpHighlight(
                chunks=[recording / f"chunk_{index:04d}.mp4" for index in range(6)],
                max_height_cm=122.4,
                max_record={"chunk": 3, "height_cm": 122.4, "reason": "calibrated_high_jump"},
                candidate_count=2,
            )

            real_datetime = recorder.datetime

            def fake_build_clip(_chunks, path, **_kwargs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"annotated")
                return {"max_height_cm": 122.4, "candidate_frame_count": 18, "frame_count": 60}

            with (
                mock.patch.object(
                    recorder,
                    "_build_annotated_jump_clip",
                    side_effect=fake_build_clip,
                ),  # noqa: SLF001
                mock.patch.object(recorder, "_send_telegram_live_notification") as send_notification,  # noqa: SLF001
                mock.patch.object(
                    recorder, "_select_jump_highlight_clip_window", return_value=highlight
                ) as select_highlight,
                mock.patch.object(
                    recorder,
                    "_cat_tv_play_live_marker_decision",
                    side_effect=[
                        (
                            True,
                            {
                                "telegram_live_action": "scan_cooldown",
                                "retry_allowed": False,
                                "previous_telegram_live_action": "camera_human_detected",
                                "age_seconds": 30.0,
                            },
                        ),
                        (True, None),
                    ],
                ) as policy_decision,
                mock.patch.object(
                    recorder,
                    "detect_camera_human_presence",
                    side_effect=[
                        recorder.CameraHumanCheck(
                            True,
                            "camera_human_detected",
                            checked_frames=4,
                            frame="chunk_0003_00001.jpg",
                            confidence=0.99,
                            bbox_xyxy=(50, 100, 240, 420),
                        ),
                        recorder.CameraHumanCheck(False, "no_camera_human_detected", checked_frames=6),
                    ],
                ) as human_check,
                mock.patch.object(recorder, "datetime") as datetime_mock,
            ):
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-17T20:40:00+04:00"
                )
                datetime_mock.fromisoformat = real_datetime.fromisoformat
                datetime_mock.side_effect = real_datetime

                payload = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260517T203500-cats-tv-october-compilation-fMiAfGtt9rc-seed30",
                    window_chunks=6,
                    cooldown_seconds=300,
                )
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-17T20:40:30+04:00"
                )
                second_payload = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260517T203500-cats-tv-october-compilation-fMiAfGtt9rc-seed30",
                    window_chunks=6,
                    cooldown_seconds=300,
                )
                datetime_mock.now.return_value.astimezone.return_value = real_datetime.fromisoformat(
                    "2026-05-17T20:50:00+04:00"
                )
                third_payload = recorder._maybe_send_live_telegram_notification(  # noqa: SLF001
                    root,
                    ha_session_id="20260517T203500-cats-tv-october-compilation-fMiAfGtt9rc-seed30",
                    window_chunks=6,
                    cooldown_seconds=300,
                )
                marker = json.loads((recording / "telegram_live_notification.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["telegram_live_action"], "camera_human_detected")
        self.assertEqual(second_payload["telegram_live_action"], "scan_cooldown")
        self.assertEqual(second_payload["previous_telegram_live_action"], "camera_human_detected")
        self.assertEqual(third_payload["telegram_live_action"], "sent")
        self.assertEqual(marker["selection_method"], "calibrated_jump_highlight")
        self.assertEqual(policy_decision.call_count, 2)
        self.assertEqual(human_check.call_count, 2)
        self.assertEqual(send_notification.call_count, 1)
        self.assertEqual(select_highlight.call_count, 2)

    def test_ensure_active_recording_runs_live_telegram_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = argparse.Namespace(
                root=root,
                ha_url="http://example.invalid",
                snapshot_url="http://example.invalid/shot.jpg",
                camera_url="http://example.invalid/video",
                duration_seconds=60,
                segment_seconds=5,
                telegram_notify=True,
                telegram_max_age_seconds=3600,
                telegram_window_chunks=6,
                telegram_max_clips=3,
                telegram_live_cooldown_seconds=300,
            )
            ha_session = recorder.HaSession(
                session_id="20260516T120000-cats-tv",
                frigate_event_id="event",
                video_url="http://example.invalid/cats-tv.mp4",
                session_state="on",
                renderer_state="playing",
            )

            with (
                mock.patch.object(recorder, "_read_ha_session", return_value=ha_session),
                mock.patch.object(recorder, "_read_pid", return_value=1234),
                mock.patch.object(recorder, "_pid_is_running", return_value=True),
                mock.patch.object(recorder, "_recorder_matches_session", return_value=True),
                mock.patch.object(
                    recorder,
                    "_maybe_send_live_telegram_notification",
                    return_value={"telegram_live_action": "sent", "clip": "/tmp/live.mp4"},
                ) as live_notify,
                mock.patch("builtins.print") as print_mock,
            ):
                rc = recorder.ensure(args)

        self.assertEqual(rc, 0)
        live_notify.assert_called_once()
        payload = json.loads(print_mock.call_args.args[0])
        self.assertEqual(payload["action"], "already_recording")
        self.assertEqual(payload["telegram_live_action"], "sent")

    def test_ensure_off_session_notification_failure_is_nonfatal_and_cooled_down(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "20260712T221208_session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text("{}\n", encoding="utf-8")
            args = argparse.Namespace(
                root=root,
                recordings_root=root / "recordings",
                ha_url="http://example.invalid",
                snapshot_url="http://example.invalid/shot.jpg",
                camera_url="http://example.invalid/video",
                duration_seconds=60,
                segment_seconds=5,
                telegram_notify=True,
                telegram_max_age_seconds=3600,
                telegram_window_chunks=6,
                telegram_max_clips=3,
                telegram_live_cooldown_seconds=300,
            )
            ha_session = recorder.HaSession(
                session_id="",
                frigate_event_id="",
                video_url="http://example.invalid/cats-tv.mp4",
                session_state="off",
                renderer_state="unavailable",
            )

            with (
                mock.patch.object(recorder, "_read_ha_session", return_value=ha_session),
                mock.patch.object(recorder, "_read_pid", return_value=1234),
                mock.patch.object(recorder, "_pid_is_running", return_value=False),
                mock.patch.object(
                    recorder,
                    "_maybe_send_telegram_notification",
                    side_effect=RuntimeError("preview extraction failed"),
                ) as final_notify,
                mock.patch("builtins.print") as print_mock,
            ):
                first_rc = recorder.ensure(args)
                second_rc = recorder.ensure(args)

            payloads = [json.loads(call.args[0]) for call in print_mock.call_args_list]
            marker = json.loads((recording / "telegram_notification_error.json").read_text(encoding="utf-8"))
            session_rows = [
                json.loads(line) for line in (root / "sessions.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual((first_rc, second_rc), (0, 0))
        final_notify.assert_called_once()
        self.assertEqual(payloads[0]["telegram_action"], "postprocess_failed")
        self.assertEqual(payloads[1]["telegram_action"], "postprocess_retry_cooldown")
        self.assertEqual(marker["telegram_action"], "postprocess_failed")
        self.assertEqual(session_rows[-1]["kind"], "cat_projector_telegram_notification_postprocess_failed")


if __name__ == "__main__":
    unittest.main()
