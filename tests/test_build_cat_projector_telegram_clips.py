from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_cat_projector_telegram_clips.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("build_cat_projector_telegram_clips", SCRIPT_PATH)
telegram_clips = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = telegram_clips
SCRIPT_SPEC.loader.exec_module(telegram_clips)


class CatProjectorTelegramClipsTest(unittest.TestCase):
    def _write_labels(self, path: Path, source_chunk: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_relpath",
                    "label_cat_present",
                    "label_cat_playing",
                    "confidence",
                    "notes",
                    "source_chunk",
                    "source_offset_seconds",
                    "ha_session_id",
                    "frigate_event_id",
                    "video_slug",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image_relpath": "frames/play.jpg",
                    "label_cat_present": "yes",
                    "label_cat_playing": "yes",
                    "confidence": "high",
                    "notes": "pawed at the wall twice",
                    "source_chunk": str(source_chunk),
                    "source_offset_seconds": "2.5",
                    "ha_session_id": "20260516T003115-cat-sher-pounce-lanes",
                    "frigate_event_id": "event-1",
                    "video_slug": "safe-stale-video-field.mp4",
                }
            )
            writer.writerow(
                {
                    "image_relpath": "frames/watch.jpg",
                    "label_cat_present": "yes",
                    "label_cat_playing": "no",
                    "confidence": "high",
                    "notes": "watched only",
                    "source_chunk": str(source_chunk),
                    "source_offset_seconds": "3.0",
                    "ha_session_id": "session-2",
                    "frigate_event_id": "event-2",
                    "video_slug": "boring.mp4",
                }
            )

    def test_loads_only_reviewed_play_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = root / "labels.csv"
            chunk = root / "chunk_0001.mp4"
            self._write_labels(labels, chunk)

            candidates = telegram_clips.load_play_candidates(labels)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].image_relpath, "frames/play.jpg")
        self.assertEqual(candidates[0].source_offset_seconds, 2.5)

    def test_build_outbox_writes_manifest_caption_and_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = root / "labels.csv"
            chunk = root / "chunk_0001.mp4"
            chunk.write_bytes(b"source")
            self._write_labels(labels, chunk)
            output_dir = root / "outbox"

            def fake_run(command, **_kwargs):
                output_path = Path(command[-1])
                output_path.write_bytes(b"x" * 2048)
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(telegram_clips.subprocess, "run", side_effect=fake_run) as run:
                manifest = telegram_clips.build_outbox(
                    labels_csv=labels,
                    output_dir=output_dir,
                    max_clips=3,
                    preroll_seconds=1.0,
                    duration_seconds=6.0,
                    include_unsure=False,
                )

            self.assertEqual(manifest["kind"], "cat_projector_telegram_clip_outbox_v1")
            self.assertEqual(manifest["clip_count"], 1)
            clip = manifest["clips"][0]
            self.assertTrue(Path(clip["clip_path"]).exists())
            caption = Path(clip["caption_path"]).read_text(encoding="utf-8")
            self.assertIn("video: cat-sher-pounce-lanes.mp4", caption)
            self.assertIn("recorded video_url field was stale", caption)
            self.assertIn("pawed at the wall twice", caption)
            command = run.call_args.args[0]
            self.assertIn("-ss", command)
            self.assertIn("1.500", command)
            self.assertIn("-t", command)
            self.assertIn("6.000", command)

    def test_refuses_unreviewed_package_without_play_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = root / "labels.csv"
            labels.write_text(
                "image_relpath,label_cat_present,label_cat_playing,source_chunk,source_offset_seconds\n"
                "frames/empty.jpg,no,no,/tmp/chunk.mp4,1.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "no reviewed cat-playing rows"):
                telegram_clips.build_outbox(
                    labels_csv=labels,
                    output_dir=root / "outbox",
                    max_clips=3,
                    preroll_seconds=1.0,
                    duration_seconds=6.0,
                    include_unsure=False,
                )


if __name__ == "__main__":
    unittest.main()
