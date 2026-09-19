import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="the meow intent model needs scikit-learn")

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
MODULE_PATH = SCRIPTS_DIR / "sher_meow_intent_model.py"
SPEC = importlib.util.spec_from_file_location("sher_meow_intent_model", MODULE_PATH)
sher_meow_intent_model = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = sher_meow_intent_model
SPEC.loader.exec_module(sher_meow_intent_model)


class SherMeowIntentModelTest(unittest.TestCase):
    def _write_sidecar(self, root: Path, index: int, *, label: str = "sher_meow") -> Path:
        room = root / "living"
        room.mkdir(parents=True, exist_ok=True)
        embedding_dir = root / "embeddings" / "panns_cnn14"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        wav_path = room / f"20260506T12000{index}.000000Z_living_room_vigi_candidate.wav"
        wav_path.write_bytes(b"RIFF")
        embedding_path = embedding_dir / f"clip{index}.npy"
        np.save(embedding_path, np.asarray([index, index + 0.5, index + 1.0], dtype=np.float32))
        sidecar = {
            "started_at": f"2026-05-06T08:00:0{index}Z",
            "source_id": "living_room_vigi",
            "wav_path": str(wav_path),
            "duration_seconds": 1.0 + index * 0.1,
            "features": {
                "rms": 100.0 + index,
                "peak": 1000.0 + index,
                "zero_crossing_rate": 0.1,
                "voice_band_ratio": 0.8,
                "meow_band_ratio": 0.7,
                "snore_low_band_ratio": 0.1,
                "snore_mid_band_ratio": 0.2,
                "high_tonal_ratio": 0.3,
            },
            "human_acoustic_labels": [label],
            "audio_tagging_models": {
                "panns_cnn14": {
                    "operational_scores": {"cat_meow": 0.8, "speech": 0.1},
                    "top_labels": [{"label": "Meow", "score": 0.8}],
                }
            },
            "embedding_refs": {"panns_cnn14": {"path": str(embedding_path), "shape": [3]}},
            "ha_context": {
                "states": {
                    "sensor.sher_s_weight": "4.2",
                    "sensor.cat_feeder": "ok",
                    "input_boolean.sleeping": "off",
                }
            },
        }
        sidecar_path = wav_path.with_suffix(".json")
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")
        return sidecar_path

    def test_build_queue_uses_embeddings_and_pet_context(self) -> None:
        # Queue building calls the acoustic tagging models, which live in the
        # audio project rather than this repository.  The runtime already
        # degrades to AudioTaggingUnavailable when they are absent, so a host
        # without that project installed should skip rather than fail.
        pytest.importorskip(
            "acoustic_event_tagging_models",
            reason="building the queue needs the audio project's tagging models",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(4):
                self._write_sidecar(root, index)
            out_dir = root / "queue"
            args = type(
                "Args",
                (),
                {
                    "sidecars": [root],
                    "root": root,
                    "out_dir": out_dir,
                    "audio_tagging_model": "panns_cnn14",
                    "min_meow_confidence": 0.2,
                    "clusters": 2,
                    "limit": 3,
                    "pca_components": 2,
                },
            )()

            sher_meow_intent_model.build_queue(args)

            rows = [json.loads(line) for line in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["kind"], "sher_meow_intent")
        self.assertIn("food_or_water", rows[0]["intent_options"])
        self.assertEqual(rows[0]["context"]["sensor.sher_s_weight"], "4.2")


if __name__ == "__main__":
    unittest.main()
