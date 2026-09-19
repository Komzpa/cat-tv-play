from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from scripts.measure_cat_projector_jump_height import compute_homography

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_live_jump_watch.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("cat_projector_live_jump_watch", SCRIPT_PATH)
live_watch = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = live_watch
SCRIPT_SPEC.loader.exec_module(live_watch)


#: A neutral 1280x720 wall calibration: the camera looks slightly down and to
#: the side of a flat wall whose floor line sits at 0 mm. No real room is
#: described here, so the geometry tests stay independent of any installation.
SYNTHETIC_IMAGE_POINTS_PX = (
    (60.0, 700.0),
    (1180.0, 660.0),
    (1150.0, 90.0),
    (80.0, 140.0),
)
SYNTHETIC_WALL_POINTS_MM = (
    (0.0, 0.0),
    (3000.0, 0.0),
    (3000.0, 2200.0),
    (0.0, 2200.0),
)


def _synthetic_wall_calibration() -> tuple[object, float]:
    """Return ``(homography, floor_y_mm)`` for the neutral wall above."""

    return compute_homography(SYNTHETIC_IMAGE_POINTS_PX, SYNTHETIC_WALL_POINTS_MM), 0.0


class CatProjectorLiveJumpWatchTest(unittest.TestCase):
    def test_calibration_round_trips_through_the_wall_plane_model(self) -> None:
        homography, floor_y_mm = _synthetic_wall_calibration()
        payload = {
            "calibration": {
                "image_points_px": [list(point) for point in SYNTHETIC_IMAGE_POINTS_PX],
                "wall_points_mm": [list(point) for point in SYNTHETIC_WALL_POINTS_MM],
                "floor_y_mm": 0,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wall_calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, loaded_floor = live_watch._read_calibration(path)

        self.assertEqual(loaded_floor, floor_y_mm)
        for image_point, wall_point in zip(SYNTHETIC_IMAGE_POINTS_PX, SYNTHETIC_WALL_POINTS_MM, strict=True):
            _wall_x, wall_y = loaded.project(*image_point)
            self.assertAlmostEqual(wall_y, wall_point[1], delta=1.0)

    def test_default_calibration_comes_from_site_configuration(self) -> None:
        calibration = live_watch.DEFAULT_CALIBRATION

        self.assertEqual(calibration.suffix, ".json")
        self.assertTrue(calibration.is_absolute())

    def test_detects_dark_cat_component_and_measures_height(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((42, 20, 68, 92), fill="black")
        homography = _compute_test_homography()

        sample, component = live_watch.sample_image(
            image,
            source="synthetic.jpg",
            homography=homography,
            floor_y_mm=0,
            roi=(0, 0, 120, 100),
            threshold=100,
            scale=2,
            min_area_px=100,
            min_height_px=20,
            max_width_px=100,
            jump_threshold_cm=70,
        )

        self.assertIsNotNone(component)
        self.assertTrue(sample.candidate_found)
        self.assertTrue(sample.triggered)
        self.assertGreater(sample.component_area_px or 0, 1000)
        self.assertAlmostEqual(sample.top_height_cm or 0, 79.0, delta=4.0)

    def test_vertical_annotation_shows_record_board_before_confirmed_jump(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "vertical.jpg"
            image = Image.new("RGB", (640, 360), (30, 30, 30))
            sample = live_watch.LiveJumpSample(
                timestamp="2026-06-06T00:00:00+04:00",
                source="chunk_0001_00001.jpg",
                candidate_found=False,
            )

            live_watch.render_vertical_annotation(
                image,
                sample,
                output,
                output_size=(360, 640),
            )

            rendered = Image.open(output).convert("RGB")
            top_strip = rendered.crop((0, 0, rendered.width, 150))
            self.assertGreater(len(set(top_strip.getdata())), 1)

    def test_ignores_small_projected_prey_speck(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((50, 50, 54, 54), fill="black")
        homography = _compute_test_homography()

        sample, component = live_watch.sample_image(
            image,
            source="synthetic.jpg",
            homography=homography,
            floor_y_mm=0,
            roi=(0, 0, 120, 100),
            threshold=100,
            scale=2,
            min_area_px=100,
            min_height_px=20,
            max_width_px=100,
            jump_threshold_cm=70,
        )

        self.assertIsNone(component)
        self.assertFalse(sample.candidate_found)

    def test_component_top_uses_robust_cap_instead_of_single_speck(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((42, 40, 68, 92), fill="black")
        draw.line((55, 8, 55, 40), fill="black")

        components = live_watch.detect_dark_components(
            image,
            roi=(0, 0, 120, 100),
            threshold=100,
            scale=1,
            min_area_px=100,
            min_height_px=20,
            max_width_px=100,
        )

        self.assertEqual(len(components), 1)
        self.assertGreater(components[0].top_y_px, 35)

    def test_once_writes_annotation_path_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration = root / "calibration.json"
            source = root / "frame.jpg"
            output = root / "annotated.jpg"
            calibration.write_text(
                json.dumps(
                    {
                        "calibration": {
                            "image_points_px": [[0, 100], [100, 100], [100, 0], [0, 0]],
                            "wall_points_mm": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
                            "floor_y_mm": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            image = Image.new("RGB", (100, 100), "white")
            ImageDraw.Draw(image).rectangle((40, 20, 60, 95), fill="black")
            image.save(source)
            args = live_watch.parse_args(
                [
                    "once",
                    "--calibration",
                    str(calibration),
                    "--image",
                    str(source),
                    "--output-image",
                    str(output),
                    "--roi",
                    "0,0,100,100",
                    "--min-area-px",
                    "100",
                    "--min-height-px",
                    "20",
                    "--disable-detector",
                ]
            )

            with mock.patch("builtins.print"):
                sample = live_watch.run_once(args)

            self.assertTrue(sample.candidate_found)
            self.assertTrue(output.exists())

    def test_active_detector_model_uses_bundled_default_when_available(self) -> None:
        args = live_watch.parse_args(["once", "--image", "synthetic.jpg"])

        with mock.patch.object(live_watch, "default_model_available", return_value=True):
            self.assertEqual(live_watch._active_detector_model(args), live_watch.DEFAULT_DETECTOR_MODEL_PATH)  # noqa: SLF001

    def test_default_roi_covers_whole_camera_frame(self) -> None:
        args = live_watch.parse_args(["once", "--image", "synthetic.jpg"])

        self.assertEqual(args.roi, (0, 0, 1280, 720))

    def test_render_annotation_draws_height_line_to_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "annotated.jpg"
            image = Image.new("RGB", (100, 100), "white")
            homography = _compute_test_homography()
            sample = live_watch.LiveJumpSample(
                timestamp="2026-05-17T00:00:00+04:00",
                source="synthetic.jpg",
                candidate_found=True,
                top_x_px=50,
                top_y_px=20,
                wall_x_mm=500,
                top_height_mm=800,
                top_height_cm=80,
                component_bbox_xyxy=(40, 20, 60, 95),
                triggered=False,
            )

            live_watch.render_annotation(image, sample, output, homography=homography, floor_y_mm=0)

            annotated = Image.open(output).convert("RGB")
            self.assertNotEqual(annotated.getpixel((50, 20)), (255, 255, 255))
            self.assertNotEqual(annotated.getpixel((50, 100 - 1)), (255, 255, 255))

    def test_track_state_keeps_trail_and_highest_sample(self) -> None:
        low = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=True,
            top_x_px=40,
            top_y_px=80,
            top_height_cm=20,
        )
        high = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=True,
            top_x_px=50,
            top_y_px=20,
            top_height_cm=80,
        )
        state = live_watch.JumpTrackState()

        state = live_watch.update_track_state(state, low, max_points=2)
        state = live_watch.update_track_state(state, high, max_points=2)
        state = live_watch.update_track_state(state, low, max_points=2)

        self.assertEqual(len(state.trail_points), 2)
        self.assertEqual(state.highest_sample, high)

    def test_track_state_expires_trail_after_consecutive_misses(self) -> None:
        seen = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=True,
            top_x_px=50,
            top_y_px=20,
            top_height_cm=80,
        )
        missing = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="synthetic.jpg",
            candidate_found=False,
        )

        state = live_watch.update_track_state(live_watch.JumpTrackState(), seen, max_points=3, max_missed_samples=2)
        state = live_watch.update_track_state(state, missing, max_points=3, max_missed_samples=2)
        self.assertEqual(len(state.trail_points), 1)
        state = live_watch.update_track_state(state, missing, max_points=3, max_missed_samples=2)
        self.assertEqual(len(state.trail_points), 1)
        state = live_watch.update_track_state(state, missing, max_points=3, max_missed_samples=2)

        self.assertEqual(state.trail_points, ())
        self.assertEqual(state.highest_sample, seen)

    def test_render_annotation_does_not_stamp_no_cat_over_active_jump_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "annotated.jpg"
            image = Image.new("RGB", (240, 160), "white")
            missing_sample = live_watch.LiveJumpSample(
                timestamp="2026-05-17T00:00:00+04:00",
                source="synthetic.jpg",
                candidate_found=False,
            )
            track_state = live_watch.JumpTrackState(trail_points=((60, 120, 20), (80, 80, 60)))

            live_watch.render_annotation(image, missing_sample, output, track_state=track_state)

            annotated = Image.open(output).convert("RGB")
            top_left = annotated.crop((0, 0, 150, 55))
            pixels = top_left.tobytes()
            red_text_pixels = sum(
                1
                for red, green, blue in zip(pixels[0::3], pixels[1::3], pixels[2::3], strict=False)
                if red > 180 and green < 130 and blue < 130
            )
            self.assertLess(red_text_pixels, 20)

    def test_render_annotation_draws_right_side_top_three_holds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "annotated.jpg"
            image = Image.new("RGB", (640, 360), "gray")
            sample = live_watch.LiveJumpSample(
                timestamp="2026-05-17T00:00:00+04:00",
                source="synthetic.jpg",
                candidate_found=False,
            )
            holds = (
                live_watch.JumpPeakHold(height_cm=120.0, image=Image.new("RGB", (80, 80), "black"), seen_count=2),
                live_watch.JumpPeakHold(height_cm=105.0, image=Image.new("RGB", (80, 80), "white"), seen_count=2),
                live_watch.JumpPeakHold(height_cm=93.0, image=Image.new("RGB", (80, 80), "black"), seen_count=2),
            )

            live_watch.render_annotation(image, sample, output, top_holds=holds)

            annotated = Image.open(output).convert("RGB")
            right_panel = annotated.crop((470, 0, 640, 250))
            magenta_pixels = sum(
                1 for red, green, blue in right_panel.getdata() if red > 180 and blue > 130 and green < 110
            )
            self.assertGreater(magenta_pixels, 100)

    def test_render_vertical_annotation_tracks_right_side_cat_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "vertical.jpg"
            image = Image.new("RGB", (1280, 720), "white")
            ImageDraw.Draw(image).rectangle((1000, 420, 1070, 680), fill="black")
            sample = live_watch.LiveJumpSample(
                timestamp="2026-05-19T00:00:00+04:00",
                source="synthetic.jpg",
                candidate_found=True,
                top_x_px=1035,
                top_y_px=420,
                top_height_cm=105.0,
                component_bbox_xyxy=(1000, 420, 1070, 680),
            )

            state = live_watch.render_vertical_annotation(
                image,
                sample,
                output,
                crop_state=live_watch.VerticalCropState(center_x_px=640),
                output_size=(270, 480),
            )

            self.assertTrue(output.exists())
            self.assertGreater(state.center_x_px or 0, 900)
            self.assertGreater(state.center_y_px or 0, 430)
            vertical = Image.open(output).convert("RGB")
            dark_pixels = sum(1 for red, green, blue in vertical.getdata() if red < 40 and green < 40 and blue < 40)
            self.assertGreater(dark_pixels, 1000)

    def test_render_vertical_annotation_draws_current_marker_after_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "vertical.jpg"
            image = Image.new("RGB", (1280, 720), "white")
            sample = live_watch.LiveJumpSample(
                timestamp="2026-05-19T00:00:00+04:00",
                source="synthetic.jpg",
                candidate_found=True,
                top_x_px=1188,
                top_y_px=430,
                top_height_cm=88.0,
                component_bbox_xyxy=(1160, 430, 1230, 675),
                detector_cat_probability=0.91,
            )

            live_watch.render_vertical_annotation(
                image,
                sample,
                output,
                crop_state=live_watch.VerticalCropState(center_x_px=1180),
                output_size=(270, 480),
            )

            vertical = Image.open(output).convert("RGB")
            red_pixels = sum(1 for red, green, blue in vertical.getdata() if red > 180 and green < 110 and blue < 110)
            yellow_pixels = sum(
                1 for red, green, blue in vertical.getdata() if red > 180 and green > 150 and blue < 120
            )
            self.assertGreater(red_pixels, 50)
            self.assertGreater(yellow_pixels, 50)

    def test_render_vertical_annotation_zooms_low_cat_into_portrait_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "vertical.jpg"
            image = Image.new("RGB", (1280, 720), "white")
            ImageDraw.Draw(image).rectangle((610, 560, 690, 710), fill="black")
            sample = live_watch.LiveJumpSample(
                timestamp="2026-05-19T00:00:00+04:00",
                source="synthetic.jpg",
                candidate_found=True,
                top_x_px=650,
                top_y_px=560,
                top_height_cm=42.0,
                component_bbox_xyxy=(610, 560, 690, 710),
                detector_cat_probability=0.96,
            )

            state = live_watch.render_vertical_annotation(
                image,
                sample,
                output,
                output_size=(270, 480),
            )

            vertical = Image.open(output).convert("RGB")
            dark_ys = [
                y
                for y in range(vertical.height)
                for x in range(vertical.width)
                if all(channel < 40 for channel in vertical.getpixel((x, y)))
            ]
            self.assertTrue(dark_ys)
            self.assertLess(max(dark_ys), vertical.height - 12)
            self.assertGreater(state.center_y_px or 0, 430)

    def test_vertical_track_state_does_not_clamp_off_crop_peak_label(self) -> None:
        base = Image.new("RGBA", (270, 480), (255, 255, 255, 255))
        state = live_watch.JumpTrackState(
            trail_points=(
                (0.0, 220.0, 20.0),
                (20.0, 120.0, 120.0),
                (40.0, 220.0, 30.0),
            )
        )

        live_watch._draw_vertical_track_state(  # noqa: SLF001
            base,
            state,
            font=live_watch._default_font(42),  # noqa: SLF001
            homography=None,
            floor_y_mm=None,
            display_max_height_cm=None,
            source_image=Image.new("RGB", (1280, 720), "white"),
            crop_left=200,
            crop_top=0,
            scale=1.0,
        )

        yellow_pixels = sum(
            1 for red, green, blue, alpha in base.getdata() if alpha > 0 and red > 180 and green > 150 and blue < 120
        )
        self.assertLess(yellow_pixels, 10)

    def test_render_jump_results_hold_makes_portrait_scoreboard_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "hold.jpg"
            image = Image.new("RGB", (1280, 720), "gray")
            hold_image = Image.new("RGB", (120, 260), "white")
            ImageDraw.Draw(hold_image).rectangle((38, 35, 82, 230), fill="black")
            holds = (
                live_watch.JumpPeakHold(height_cm=126.9, image=hold_image, seen_count=3),
                live_watch.JumpPeakHold(height_cm=105.0, image=Image.new("RGB", (160, 160), "black"), seen_count=2),
            )

            live_watch.render_jump_results_hold(
                image,
                output,
                top_holds=holds,
                output_size=(270, 480),
            )

            rendered = Image.open(output).convert("RGB")
            self.assertEqual(rendered.size, (270, 480))
            magenta_pixels = sum(
                1 for red, green, blue in rendered.getdata() if red > 180 and blue > 130 and green < 110
            )
            self.assertGreater(magenta_pixels, 500)

    def test_render_jump_results_hold_does_not_show_debug_seen_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "hold.jpg"
            image = Image.new("RGB", (1280, 720), "gray")
            hold_image = Image.new("RGB", (120, 260), "white")
            holds = (live_watch.JumpPeakHold(height_cm=142.0, image=hold_image, seen_count=13),)
            texts: list[str] = []
            original_draw = live_watch._draw_text_with_stroke  # noqa: SLF001

            def capture_text(draw, xy, text, **kwargs):
                texts.append(text)
                return original_draw(draw, xy, text, **kwargs)

            with mock.patch.object(live_watch, "_draw_text_with_stroke", side_effect=capture_text):
                live_watch.render_jump_results_hold(
                    image,
                    output,
                    top_holds=holds,
                    output_size=(270, 480),
                )

            self.assertIn("#1 142.0 cm", texts)
            self.assertNotIn("x13", texts)

    def test_render_jump_results_hold_uses_publish_safe_title_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "hold.jpg"
            image = Image.new("RGB", (1280, 720), "gray")
            hold_image = Image.new("RGB", (120, 260), "white")
            holds = (live_watch.JumpPeakHold(height_cm=122.2, image=hold_image, seen_count=4),)
            texts: list[str] = []
            original_draw = live_watch._draw_text_with_stroke  # noqa: SLF001

            def capture_text(draw, xy, text, **kwargs):
                texts.append(text)
                return original_draw(draw, xy, text, **kwargs)

            with mock.patch.object(live_watch, "_draw_text_with_stroke", side_effect=capture_text):
                live_watch.render_jump_results_hold(
                    image,
                    output,
                    top_holds=holds,
                    output_size=(270, 480),
                    publish_safe_zone=True,
                )

            self.assertIn("RECORDS", texts)
            self.assertNotIn("RECORD BOARD", texts)

    def test_top_jump_bar_can_show_publish_session_context(self) -> None:
        base = Image.new("RGBA", (1080, 1920), (20, 20, 20, 255))
        texts: list[str] = []
        original_draw = live_watch._draw_text_with_stroke  # noqa: SLF001

        def capture_text(draw, xy, text, **kwargs):
            texts.append(text)
            return original_draw(draw, xy, text, **kwargs)

        with mock.patch.object(live_watch, "_draw_text_with_stroke", side_effect=capture_text):
            live_watch._draw_top_jump_holds_top_bar(  # noqa: SLF001
                base,
                (),
                font=live_watch._default_font(42),  # noqa: SLF001
                publish_safe_zone=True,
                session_title="SHER JUMPS",
                session_subtitle="Monday, 18 May 2026",
            )

        self.assertIn("SHER JUMPS", texts)
        self.assertIn("Monday, 18 May 2026", texts)
        self.assertIn("waiting for first jump", texts)

    def test_line_segments_outside_rect_skip_cat_body(self) -> None:
        segments = live_watch._line_segments_outside_rect(  # noqa: SLF001
            (0.0, 10.0, 100.0, 10.0),
            (40.0, 0.0, 60.0, 20.0),
        )

        self.assertEqual(len(segments), 2)
        self.assertLess(segments[0][2], 40.0)
        self.assertGreater(segments[1][0], 60.0)

    def test_wall_height_line_maps_calibrated_rescore_alignment_back_to_frame(self) -> None:
        homography = live_watch.Homography((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        coordinate_transform = {
            "frame_size_px": [1280.0, 720.0],
            "calibration_size_px": [1280.0, 720.0],
            "alignment": {
                "applied": True,
                "current_to_calibration": [[1.0, 0.0, -20.0], [0.0, 1.0, -30.0]],
                "calibration_to_current": [[1.0, 0.0, 20.0], [0.0, 1.0, 30.0]],
            },
        }

        line = live_watch._wall_line_inside_projection(  # noqa: SLF001
            image=Image.new("RGB", (1280, 720), "white"),
            homography=homography,
            wall_y_mm=100.0,
            coordinate_transform=coordinate_transform,
        )

        self.assertIsNotNone(line)
        assert line is not None
        self.assertAlmostEqual(line[0][1], 130.0)
        self.assertAlmostEqual(line[1][1], 130.0)

    def test_rescore_live_sample_preserves_coordinate_transform_for_renderer(self) -> None:
        recorder_path = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_learning_recorder.py"
        spec = importlib.util.spec_from_file_location(
            "cat_projector_learning_recorder_for_live_watch_test", recorder_path
        )
        assert spec is not None and spec.loader is not None
        recorder = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = recorder
        spec.loader.exec_module(recorder)
        coordinate_transform = {
            "frame_size_px": [1280.0, 720.0],
            "calibration_size_px": [1280.0, 720.0],
            "alignment": {"calibration_to_current": [[1.0, 0.0, 20.0], [0.0, 1.0, 30.0]]},
        }

        sample = recorder._rescore_live_sample(  # noqa: SLF001
            {
                "max_bbox": "10,20,30,40",
                "max_jump_height_cm": 90.0,
                "best_measurement_point": {
                    "image_x": 25.0,
                    "image_y": 20.0,
                    "wall_x_cm": 10.0,
                    "debug": {"coordinate_transform": coordinate_transform},
                },
            },
            source="frame.jpg",
        )

        self.assertEqual(sample.coordinate_transform, coordinate_transform)

    def test_render_annotation_frames_vertical_hold_with_vertical_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "annotated.jpg"
            image = Image.new("RGB", (640, 360), "gray")
            sample = live_watch.LiveJumpSample(
                timestamp="2026-05-17T00:00:00+04:00",
                source="synthetic.jpg",
                candidate_found=False,
            )
            hold_image = Image.new("RGB", (48, 160), "white")
            ImageDraw.Draw(hold_image).rectangle((14, 24, 34, 140), fill="black")
            holds = (live_watch.JumpPeakHold(height_cm=126.9, image=hold_image, seen_count=2),)

            live_watch.render_annotation(image, sample, output, top_holds=holds)

            annotated = Image.open(output).convert("RGB")
            right_panel = annotated.crop((430, 30, 640, 145))
            magenta_columns = {
                x
                for x in range(right_panel.width)
                for y in range(right_panel.height)
                if ((pixel := right_panel.getpixel((x, y)))[0] > 180 and pixel[2] > 130 and pixel[1] < 110)
            }
            self.assertGreater(len(magenta_columns), 20)
            self.assertLess(len(magenta_columns), 90)

    def test_top_jump_holds_keep_top_three_distinct_jumps_by_height(self) -> None:
        image = Image.new("RGB", (200, 160), "white")
        holds: tuple[live_watch.JumpPeakHold, ...] = ()
        for index, (height, x) in enumerate([(70.0, 40), (120.0, 80), (95.0, 120), (140.0, 160)]):
            sample = live_watch.LiveJumpSample(
                timestamp="2026-05-17T00:00:00+04:00",
                source=f"chunk_{index:04d}_00001.jpg",
                candidate_found=True,
                top_x_px=x,
                top_y_px=50,
                top_height_cm=height,
                component_bbox_xyxy=(x - 10, 45, x + 10, 110),
            )
            holds = live_watch.update_top_jump_holds(
                holds,
                image,
                sample,
                max_holds=3,
                min_height_cm=45.0,
            )

        self.assertEqual([round(hold.height_cm, 1) for hold in holds], [140.0, 120.0, 95.0])

    def test_top_jump_holds_use_explicit_source_time_instead_of_filename_fps_guess(self) -> None:
        image = Image.new("RGB", (200, 160), "white")
        first = live_watch.LiveJumpSample(
            timestamp="2026-06-06T00:00:00+04:00",
            source="chunk_0001_00001.jpg",
            candidate_found=True,
            top_x_px=80,
            top_y_px=60,
            top_height_cm=90.0,
            component_bbox_xyxy=(60, 50, 100, 140),
            source_time_s=10.0,
        )
        second = live_watch.LiveJumpSample(
            timestamp="2026-06-06T00:00:00+04:00",
            source="chunk_0001_00040.jpg",
            candidate_found=True,
            top_x_px=82,
            top_y_px=55,
            top_height_cm=120.0,
            component_bbox_xyxy=(62, 45, 102, 140),
            source_time_s=14.2,
        )

        holds: tuple[live_watch.JumpPeakHold, ...] = ()
        holds = live_watch.update_top_jump_holds(holds, image, first, min_height_cm=45.0)
        holds = live_watch.update_top_jump_holds(holds, image, second, min_height_cm=45.0)

        self.assertEqual([round(hold.height_cm, 1) for hold in holds], [120.0, 90.0])

    def test_top_jump_holds_replace_same_jump_as_it_rises(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        holds: tuple[live_watch.JumpPeakHold, ...] = ()
        for frame, height, y in [(1, 55.0, 110), (10, 82.0, 80), (20, 116.0, 42)]:
            sample = live_watch.LiveJumpSample(
                timestamp="2026-05-17T00:00:00+04:00",
                source=f"chunk_0042_{frame:05d}.jpg",
                candidate_found=True,
                top_x_px=110,
                top_y_px=y,
                top_height_cm=height,
                component_bbox_xyxy=(90, y, 140, 160),
            )
            holds = live_watch.update_top_jump_holds(holds, image, sample, max_holds=3, min_height_cm=45.0)

        self.assertEqual(len(holds), 1)
        self.assertEqual(round(holds[0].height_cm, 1), 116.0)
        self.assertEqual(holds[0].seen_count, 3)

    def test_top_jump_holds_reject_horizontal_projected_object_as_jump_hold(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        projected_object = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="chunk_0042_00048.jpg",
            candidate_found=True,
            top_x_px=110,
            top_y_px=70,
            top_height_cm=94.2,
            component_bbox_xyxy=(70, 70, 170, 135),
        )
        reaching_cat = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="chunk_0045_00001.jpg",
            candidate_found=True,
            top_x_px=130,
            top_y_px=42,
            top_height_cm=142.6,
            component_bbox_xyxy=(110, 42, 170, 165),
        )

        holds = live_watch.update_top_jump_holds((), image, projected_object, max_holds=3, min_height_cm=45.0)
        holds = live_watch.update_top_jump_holds(holds, image, reaching_cat, max_holds=3, min_height_cm=45.0)

        self.assertEqual([round(hold.height_cm, 1) for hold in holds], [142.6])

    def test_top_jump_holds_show_provisional_sitting_max_until_first_jump(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        holds: tuple[live_watch.JumpPeakHold, ...] = ()
        for frame, height in [(1, 22.0), (12, 32.0), (35, 28.0)]:
            sample = live_watch.LiveJumpSample(
                timestamp="2026-05-17T00:00:00+04:00",
                source=f"chunk_0042_{frame:05d}.jpg",
                candidate_found=True,
                top_x_px=110,
                top_y_px=120,
                top_height_cm=height,
                component_bbox_xyxy=(90, 110, 140, 160),
            )
            holds = live_watch.update_top_jump_holds(holds, image, sample, min_height_cm=45.0)

        self.assertEqual([round(hold.height_cm, 1) for hold in holds], [32.0])

    def test_top_jump_holds_drop_provisional_sitting_max_after_first_jump(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        sitting = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="chunk_0042_00001.jpg",
            candidate_found=True,
            top_x_px=110,
            top_y_px=120,
            top_height_cm=32.0,
            component_bbox_xyxy=(90, 110, 140, 160),
        )
        jump = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="chunk_0042_00045.jpg",
            candidate_found=True,
            top_x_px=130,
            top_y_px=70,
            top_height_cm=70.0,
            component_bbox_xyxy=(110, 70, 160, 160),
        )

        holds = live_watch.update_top_jump_holds((), image, sitting, min_height_cm=45.0)
        holds = live_watch.update_top_jump_holds(holds, image, jump, min_height_cm=45.0)

        self.assertEqual([round(hold.height_cm, 1) for hold in holds], [70.0])

    def test_top_jump_holds_append_new_jump_after_old_clip_record(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        old_hold = live_watch.JumpPeakHold(
            height_cm=130.0,
            image=Image.new("RGB", (80, 80), "black"),
            top_xy=(110.0, 50.0),
            source="chunk_0001_00001.jpg",
            source_time_s=5.0,
        )
        sample = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="chunk_0002_00001.jpg",
            candidate_found=True,
            top_x_px=112,
            top_y_px=55,
            top_height_cm=90.0,
            component_bbox_xyxy=(90, 55, 140, 150),
        )

        holds = live_watch.update_top_jump_holds((old_hold,), image, sample, max_holds=3)

        self.assertEqual([round(hold.height_cm, 1) for hold in holds], [130.0, 90.0])

    def test_top_jump_holds_replace_lowest_full_panel_slot_with_later_higher_jump(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        holds = (
            live_watch.JumpPeakHold(height_cm=50.0, image=Image.new("RGB", (80, 80), "black"), source_time_s=5.0),
            live_watch.JumpPeakHold(height_cm=70.0, image=Image.new("RGB", (80, 80), "black"), source_time_s=15.0),
            live_watch.JumpPeakHold(height_cm=90.0, image=Image.new("RGB", (80, 80), "black"), source_time_s=25.0),
        )
        later_high_jump = live_watch.LiveJumpSample(
            timestamp="2026-05-17T00:00:00+04:00",
            source="chunk_0010_00001.jpg",
            candidate_found=True,
            top_x_px=180,
            top_y_px=30,
            top_height_cm=140.0,
            component_bbox_xyxy=(150, 30, 210, 150),
        )

        updated = live_watch.update_top_jump_holds(holds, image, later_high_jump, max_holds=3)

        self.assertEqual([round(hold.height_cm, 1) for hold in updated], [140.0, 90.0, 70.0])

    def test_top_jump_holds_confirm_only_repeated_same_jump(self) -> None:
        one_frame_spike = live_watch.JumpPeakHold(
            height_cm=97.9,
            image=Image.new("RGB", (80, 80), "black"),
            source_time_s=10.0,
            seen_count=1,
        )
        repeated_jump = live_watch.JumpPeakHold(
            height_cm=92.0,
            image=Image.new("RGB", (80, 80), "black"),
            source_time_s=20.0,
            seen_count=2,
        )

        confirmed = live_watch.confirmed_top_jump_holds((one_frame_spike, repeated_jump))

        self.assertEqual([round(hold.height_cm, 1) for hold in confirmed], [92.0])

    def test_confirmed_top_jump_holds_keep_only_high_jumps_after_100cm_peak(self) -> None:
        lower = live_watch.JumpPeakHold(
            height_cm=94.2,
            image=Image.new("RGB", (80, 80), "black"),
            seen_count=2,
        )
        higher = live_watch.JumpPeakHold(
            height_cm=142.6,
            image=Image.new("RGB", (80, 80), "black"),
            seen_count=2,
        )

        confirmed = live_watch.confirmed_top_jump_holds((lower, higher))

        self.assertEqual([round(hold.height_cm, 1) for hold in confirmed], [142.6])

    def test_confirmed_top_jump_holds_hide_low_artifacts_after_real_jump(self) -> None:
        false_low = live_watch.JumpPeakHold(
            height_cm=53.0,
            image=Image.new("RGB", (80, 80), "black"),
            seen_count=5,
        )
        real_jump = live_watch.JumpPeakHold(
            height_cm=122.8,
            image=Image.new("RGB", (80, 80), "black"),
            seen_count=2,
        )

        confirmed = live_watch.confirmed_top_jump_holds((false_low, real_jump))

        self.assertEqual([round(hold.height_cm, 1) for hold in confirmed], [122.8])

    def test_render_max_line_uses_confirmed_hold_not_unconfirmed_track_spike(self) -> None:
        image = Image.new("RGB", (1280, 720), "white")
        homography, floor_y_mm = _synthetic_wall_calibration()
        unconfirmed_spike = live_watch.LiveJumpSample(
            timestamp="2026-05-18T00:00:00+04:00",
            source="chunk_0010_00001.jpg",
            candidate_found=True,
            top_x_px=700,
            top_y_px=250,
            top_height_mm=1505,
            top_height_cm=150.5,
        )
        state = live_watch.JumpTrackState(
            trail_points=((700.0, 250.0, 150.5),),
            highest_sample=unconfirmed_spike,
        )
        confirmed_hold = live_watch.JumpPeakHold(
            height_cm=122.2,
            image=Image.new("RGB", (80, 80), "black"),
            seen_count=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "annotated.jpg"
            live_watch.render_annotation(
                image,
                unconfirmed_spike,
                output,
                homography=homography,
                floor_y_mm=floor_y_mm,
                track_state=state,
                top_holds=(confirmed_hold,),
            )

            annotated = Image.open(output).convert("RGB")

        spike_line = live_watch._wall_line_inside_projection(
            image=image,
            homography=homography,
            wall_y_mm=floor_y_mm + 1505,
        )
        confirmed_line = live_watch._wall_line_inside_projection(
            image=image,
            homography=homography,
            wall_y_mm=floor_y_mm + 1222,
        )
        assert spike_line is not None
        assert confirmed_line is not None
        spike_y = int(round((spike_line[0][1] + spike_line[1][1]) / 2.0))
        confirmed_y = int(round((confirmed_line[0][1] + confirmed_line[1][1]) / 2.0))

        def magenta_count(y: int) -> int:
            band = annotated.crop((0, max(0, y - 3), 1100, min(annotated.height, y + 4)))
            return sum(1 for red, green, blue in band.getdata() if red > 190 and blue > 140 and green < 120)

        self.assertLess(magenta_count(spike_y), 20)
        self.assertGreater(magenta_count(confirmed_y), 250)

    def test_projection_max_line_stays_at_constant_wall_height(self) -> None:
        image = Image.new("RGB", (1280, 720), "white")
        homography, _floor_y_mm = _synthetic_wall_calibration()

        line = live_watch._wall_line_inside_projection(
            image=image,
            homography=homography,
            wall_y_mm=1000,
        )

        self.assertIsNotNone(line)
        assert line is not None
        left_px, right_px = line
        _left_x_mm, left_y_mm = homography.project(*left_px)
        _right_x_mm, right_y_mm = homography.project(*right_px)
        self.assertAlmostEqual(left_y_mm, 1000, delta=1)
        self.assertAlmostEqual(right_y_mm, 1000, delta=1)

    def test_iter_mjpeg_frames_reads_multiple_jpegs_from_one_stream(self) -> None:
        first = Image.new("RGB", (4, 4), "black")
        second = Image.new("RGB", (4, 4), "white")
        first_buffer = live_watch.io.BytesIO()
        second_buffer = live_watch.io.BytesIO()
        first.save(first_buffer, format="JPEG")
        second.save(second_buffer, format="JPEG")
        content = (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + first_buffer.getvalue()
            + b"\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + second_buffer.getvalue()
            + b"\r\n"
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                del chunk_size
                yield content[:30]
                yield content[30:]

        with mock.patch.object(live_watch.requests, "get", return_value=FakeResponse()):
            frames = list(live_watch.iter_mjpeg_frames("http://example.invalid/video", timeout_seconds=1))

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].size, (4, 4))

    def test_watch_degrades_to_motion_path_when_detector_load_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration = root / "calibration.json"
            output_dir = root / "watch"
            calibration.write_text(
                json.dumps(
                    {
                        "calibration": {
                            "image_points_px": [[0, 100], [100, 100], [100, 0], [0, 0]],
                            "wall_points_mm": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
                            "floor_y_mm": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            image = Image.new("RGB", (100, 100), "white")
            ImageDraw.Draw(image).rectangle((40, 20, 60, 95), fill="black")
            args = live_watch.parse_args(
                [
                    "watch",
                    "--calibration",
                    str(calibration),
                    "--snapshot-url",
                    "synthetic.jpg",
                    "--stream-url",
                    "",
                    "--output-dir",
                    str(output_dir),
                    "--duration-seconds",
                    "0.01",
                    "--fps",
                    "1000",
                    "--roi",
                    "0,0,100,100",
                    "--min-area-px",
                    "100",
                    "--min-height-px",
                    "20",
                    "--detector-model",
                    str(root / "broken.cbm"),
                    "--print-samples",
                ]
            )

            with (
                mock.patch.object(live_watch, "load_model", side_effect=RuntimeError("bad detector")) as load_model,
                mock.patch.object(live_watch, "_load_image", return_value=image),
                mock.patch("builtins.print") as printed,
            ):
                live_watch.watch(args)

            self.assertEqual(load_model.call_count, 1)
            events = (output_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("detector disabled after load failure", events)
            printed_payloads = [json.loads(call.args[0]) for call in printed.call_args_list]
            self.assertTrue(any(payload.get("candidate_found") for payload in printed_payloads))

    def test_watch_writes_latest_overlay_even_without_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration = root / "calibration.json"
            output_dir = root / "watch"
            calibration.write_text(
                json.dumps(
                    {
                        "calibration": {
                            "image_points_px": [[0, 100], [100, 100], [100, 0], [0, 0]],
                            "wall_points_mm": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
                            "floor_y_mm": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            image = Image.new("RGB", (100, 100), "white")
            ImageDraw.Draw(image).rectangle((40, 80, 60, 95), fill="black")
            args = live_watch.parse_args(
                [
                    "watch",
                    "--calibration",
                    str(calibration),
                    "--snapshot-url",
                    "synthetic.jpg",
                    "--stream-url",
                    "",
                    "--output-dir",
                    str(output_dir),
                    "--duration-seconds",
                    "0.01",
                    "--fps",
                    "1000",
                    "--roi",
                    "0,0,100,100",
                    "--min-area-px",
                    "100",
                    "--min-height-px",
                    "10",
                    "--jump-threshold-cm",
                    "95",
                    "--disable-detector",
                ]
            )

            with (
                mock.patch.object(live_watch, "_load_image", return_value=image),
                mock.patch("builtins.print"),
            ):
                live_watch.watch(args)

            self.assertTrue((output_dir / "latest_jump_overlay.jpg").exists())
            latest = json.loads((output_dir / "latest_jump_sample.json").read_text(encoding="utf-8"))
            self.assertTrue(latest["candidate_found"])
            self.assertFalse(latest["triggered"])


def _compute_test_homography():
    return compute_homography(
        image_points_px=((0, 100), (100, 100), (100, 0), (0, 0)),
        wall_points_mm=((0, 0), (1000, 0), (1000, 1000), (0, 1000)),
    )


if __name__ == "__main__":
    unittest.main()
