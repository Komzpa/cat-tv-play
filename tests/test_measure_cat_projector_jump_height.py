from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "measure_cat_projector_jump_height.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("measure_cat_projector_jump_height", SCRIPT_PATH)
jump_height = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = jump_height
SCRIPT_SPEC.loader.exec_module(jump_height)


class CatProjectorJumpHeightTest(unittest.TestCase):
    def test_homography_maps_image_point_to_wall_height(self) -> None:
        homography = jump_height.compute_homography(
            image_points_px=((0, 100), (100, 100), (100, 0), (0, 0)),
            wall_points_mm=((0, 0), (1000, 0), (1000, 1000), (0, 1000)),
        )

        x_mm, y_mm = homography.project(50, 25)

        self.assertAlmostEqual(x_mm, 500.0, places=6)
        self.assertAlmostEqual(y_mm, 750.0, places=6)

    def test_homography_unprojects_wall_point_to_image_point(self) -> None:
        homography = jump_height.compute_homography(
            image_points_px=((0, 100), (100, 100), (100, 0), (0, 0)),
            wall_points_mm=((0, 0), (1000, 0), (1000, 1000), (0, 1000)),
        )

        x_px, y_px = homography.unproject(500, 750)

        self.assertAlmostEqual(x_px, 50.0, places=6)
        self.assertAlmostEqual(y_px, 25.0, places=6)

    def test_homography_uses_extra_calibration_points(self) -> None:
        homography = jump_height.compute_homography(
            image_points_px=((0, 100), (100, 100), (100, 0), (0, 0), (50, 50)),
            wall_points_mm=((0, 0), (1000, 0), (1000, 1000), (0, 1000), (500, 500)),
        )

        x_mm, y_mm = homography.project(50, 25)

        self.assertAlmostEqual(x_mm, 500.0, places=3)
        self.assertAlmostEqual(y_mm, 750.0, places=3)

    def test_measurement_uses_bbox_top_center_and_skips_absent_cat(self) -> None:
        homography = jump_height.compute_homography(
            image_points_px=((0, 100), (100, 100), (100, 0), (0, 0)),
            wall_points_mm=((0, 0), (1000, 0), (1000, 1000), (0, 1000)),
        )

        observations = [
            jump_height.Observation("jump.jpg", 50, 20, "high", "chunk_0001.mp4", "visible_top", True, ""),
            jump_height.Observation("lower.jpg", 40, 80, "medium", "chunk_0002.mp4", "visible_top", True, ""),
        ]
        records = jump_height.measure_observations(observations, homography=homography, floor_y_mm=0)

        self.assertEqual(records[0]["cat_top_height_mm"], 800.0)
        self.assertEqual(records[0]["cat_top_height_cm"], 80.0)
        self.assertEqual(records[1]["cat_top_height_mm"], 200.0)

    def test_reads_labeling_csv_bbox_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_relpath", "label_cat_present", "bbox_xywh", "confidence", "source_chunk"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_relpath": "frames/jump.jpg",
                        "label_cat_present": "yes",
                        "bbox_xywh": "40,20,20,60",
                        "confidence": "high",
                        "source_chunk": "chunk_0001.mp4",
                    }
                )
                writer.writerow(
                    {
                        "image_relpath": "frames/empty.jpg",
                        "label_cat_present": "no",
                        "bbox_xywh": "10,10,20,20",
                        "confidence": "high",
                        "source_chunk": "chunk_0002.mp4",
                    }
                )

            observations = jump_height.read_observations(path)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].frame_id, "frames/jump.jpg")
        self.assertEqual(observations[0].x_px, 50.0)
        self.assertEqual(observations[0].y_px, 20.0)
        self.assertTrue(observations[0].include_in_max)

    def test_reads_positive_candidate_bbox_and_skips_negative_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_relpath",
                        "label_cat_present",
                        "label_candidate_is_cat",
                        "candidate_bbox_xywh",
                        "confidence",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_relpath": "frames/jump.jpg",
                        "label_cat_present": "yes",
                        "label_candidate_is_cat": "yes",
                        "candidate_bbox_xywh": "300,120,40,90",
                        "confidence": "detector_candidate_needs_human_review",
                    }
                )
                writer.writerow(
                    {
                        "image_relpath": "frames/prey.jpg",
                        "label_cat_present": "yes",
                        "label_candidate_is_cat": "no",
                        "candidate_bbox_xywh": "100,100,20,20",
                        "confidence": "hard negative",
                    }
                )

            observations = jump_height.read_observations(path)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].frame_id, "frames/jump.jpg")
        self.assertEqual(observations[0].x_px, 320.0)
        self.assertEqual(observations[0].y_px, 120.0)

    def test_false_candidate_row_preserves_manual_cat_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_relpath",
                        "label_cat_present",
                        "label_candidate_is_cat",
                        "candidate_bbox_xywh",
                        "bbox_xywh",
                        "confidence",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_relpath": "frames/cat-and-prey.jpg",
                        "label_cat_present": "yes",
                        "label_candidate_is_cat": "no",
                        "candidate_bbox_xywh": "100,100,20,20",
                        "bbox_xywh": "300,120,40,90",
                        "confidence": "human_reviewed_manual_cat_box",
                    }
                )

            observations = jump_height.read_observations(path)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].x_px, 320.0)
        self.assertEqual(observations[0].y_px, 120.0)

    def test_excludes_rough_blur_from_max_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration = root / "calibration.json"
            observations = root / "observations.jsonl"
            output_json = root / "jump_heights.json"
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
            observations.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "frame_id": "empty-blur",
                                "top_x_px": 50,
                                "top_y_px": 10,
                                "measurement_role": "ambiguous_blur",
                                "confidence": "rough motion blur above visible cat",
                            }
                        ),
                        json.dumps(
                            {
                                "frame_id": "visible-body",
                                "top_x_px": 50,
                                "top_y_px": 30,
                                "measurement_role": "visible_top",
                                "confidence": "medium",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            jump_height.main(
                [
                    "--calibration",
                    str(calibration),
                    "--observations",
                    str(observations),
                    "--output-json",
                    str(output_json),
                ]
            )
            payload = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(payload["max_frame_id"], "visible-body")
        self.assertEqual(payload["max_jump_height_cm"], 70.0)
        self.assertEqual(payload["included_record_count"], 1)
        self.assertEqual(payload["excluded_record_count"], 1)

    def test_reads_distance_weighted_wall_plane_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "active_calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "wall_plane_model": {
                            "coefficients": {
                                "uA": 2,
                                "uB": 0,
                                "uC": 10,
                                "a": 0,
                                "b": -3,
                                "c": 1200,
                                "g": 0,
                                "h": 0,
                            }
                        },
                        "floor_y_mm": 100,
                    }
                ),
                encoding="utf-8",
            )

            homography, floor_y_mm = jump_height._read_calibration(path)

        self.assertEqual(homography.project(20, 300), (50, 300))
        self.assertEqual(floor_y_mm, 100)

    def test_cli_writes_json_and_csv_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration = root / "calibration.json"
            labels = root / "labels.csv"
            output_json = root / "jump_heights.json"
            output_csv = root / "jump_heights.csv"
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
            labels.write_text(
                "image_relpath,label_cat_present,bbox_xywh,confidence,source_chunk\n"
                'frames/jump.jpg,yes,"40,20,20,60",high,chunk_0001.mp4\n',
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--calibration",
                    str(calibration),
                    "--observations",
                    str(labels),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ],
                check=False,
                cwd=tmpdir,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "cat_projector_jump_height_measurements_v1")
            self.assertEqual(payload["max_jump_height_cm"], 80.0)
            self.assertEqual(payload["included_record_count"], 1)
            self.assertIn("cat_top_height_cm", output_csv.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
