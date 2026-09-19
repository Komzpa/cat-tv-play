from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_cat_projector_jump_review.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("build_cat_projector_jump_review", SCRIPT_PATH)
jump_review = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = jump_review
SCRIPT_SPEC.loader.exec_module(jump_review)


class CatProjectorJumpReviewTest(unittest.TestCase):
    def test_parse_extract_spec(self) -> None:
        spec = jump_review._parse_extract_spec("152:1.6:highest paw")  # noqa: SLF001

        self.assertEqual(spec.chunk_index, 152)
        self.assertEqual(spec.offset_seconds, 1.6)
        self.assertEqual(spec.label, "highest_paw")

    def test_scan_writes_manifest_sheet_and_observation_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "ha_session_id": "session-id",
                        "video_url": "http://example.invalid/cat.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for index in range(3):
                (recording / f"chunk_{index:04d}.mp4").write_bytes(b"x" * 4096)
            output = root / "review"
            args = jump_review.parse_args(
                [
                    "scan",
                    "--recording-dir",
                    str(recording),
                    "--output-dir",
                    str(output),
                    "--sheet-chunk-count",
                    "2",
                    "--extract",
                    "1:1.2:jump",
                ]
            )

            def fake_sheet(*, output_path, **_kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"sheet")

            def fake_frame(_source, _offset, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"frame" * 300)

            with (
                mock.patch.object(jump_review, "_build_sheet", side_effect=fake_sheet),  # noqa: SLF001
                mock.patch.object(jump_review, "_extract_frame", side_effect=fake_frame),  # noqa: SLF001
            ):
                jump_review.scan(args)

            manifest = json.loads((output / "scan_manifest.json").read_text(encoding="utf-8"))
            template = (output / "observations_template.jsonl").read_text(encoding="utf-8")

        self.assertEqual(manifest["kind"], "cat_projector_jump_review_scan_v1")
        self.assertEqual(len(manifest["sheets"]), 2)
        self.assertIn('"top_x_px": null', template)
        self.assertIn('"measurement_role": "visible_top"', template)
        self.assertIn('"include_in_max": true', template)
        self.assertIn("chunk_0001", template)

    @unittest.skipIf(jump_review.Image is None, "Pillow is not installed")
    def test_scan_can_prefill_detector_top_point_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recording = root / "recordings" / "session"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text("{}\n", encoding="utf-8")
            (recording / "chunk_0001.mp4").write_bytes(b"x" * 4096)
            model = root / "detector.cbm"
            model.write_text("model", encoding="utf-8")
            metadata = root / "detector.json"
            metadata.write_text("{}", encoding="utf-8")
            output = root / "review"
            args = jump_review.parse_args(
                [
                    "scan",
                    "--recording-dir",
                    str(recording),
                    "--output-dir",
                    str(output),
                    "--extract",
                    "1:1.2:jump",
                    "--propose-detector-top-points",
                    "--detector-model",
                    str(model),
                    "--detector-metadata",
                    str(metadata),
                ]
            )

            def fake_sheet(*, output_path, **_kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"sheet")

            def fake_frame(_source, _offset, output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                jump_review.Image.new("RGB", (640, 360), "white").save(output_path)

            prediction = SimpleNamespace(
                candidate=SimpleNamespace(
                    top_x_px=320.25,
                    top_y_px=120.5,
                    bbox_xywh=(300.0, 120.0, 40.0, 90.0),
                ),
                cat_probability=0.72,
                model_path=str(model),
            )
            with (
                mock.patch.object(jump_review, "_build_sheet", side_effect=fake_sheet),  # noqa: SLF001
                mock.patch.object(jump_review, "_extract_frame", side_effect=fake_frame),  # noqa: SLF001
                mock.patch.object(jump_review, "load_model", return_value=(object(), {"model_path": str(model)})),
                mock.patch.object(jump_review, "best_cat_candidate_from_model", return_value=prediction),
            ):
                jump_review.scan(args)

            row = json.loads((output / "observations_template.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(row["top_x_px"], 320.25)
        self.assertEqual(row["top_y_px"], 120.5)
        self.assertEqual(row["measurement_role"], "detector_candidate_top")
        self.assertEqual(row["candidate_bbox_xywh"], "300,120,40,90")
        self.assertEqual(row["detector_cat_probability"], 0.72)

    @unittest.skipIf(jump_review.Image is None, "Pillow is not installed")
    def test_render_uses_included_max_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review = root / "review"
            frames = review / "frames"
            frames.mkdir(parents=True)
            frame = frames / "jump.jpg"
            jump_review.Image.new("RGB", (640, 360), "white").save(frame)
            measurements = review / "measurements.json"
            output = review / "annotated.jpg"
            measurements.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "frame_id": "frames/jump.jpg",
                                "cat_top_x_px": 100,
                                "cat_top_y_px": 40,
                                "cat_top_height_mm": 1230,
                                "cat_top_height_cm": 123.0,
                                "include_in_max": False,
                                "exclusion_reason": "empty wall blur",
                                "source": "chunk_0001.mp4",
                            },
                            {
                                "frame_id": "frames/jump.jpg",
                                "cat_top_x_px": 110,
                                "cat_top_y_px": 70,
                                "cat_top_height_mm": 1130,
                                "cat_top_height_cm": 113.0,
                                "include_in_max": True,
                                "source": "chunk_0001.mp4",
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = jump_review.parse_args(
                [
                    "render",
                    "--measurements",
                    str(measurements),
                    "--review-dir",
                    str(review),
                    "--output",
                    str(output),
                ]
            )

            jump_review.render(args)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1024)


if __name__ == "__main__":
    unittest.main()
