from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="the meow context scorer needs scipy")
pytest.importorskip(
    "acoustic_event_tagging_models",
    reason="the meow context scorer needs the audio project's tagging helpers",
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
MODULE_PATH = SCRIPTS_DIR / "score_cat_meow_context_model.py"
SPEC = importlib.util.spec_from_file_location("score_cat_meow_context_model", MODULE_PATH)
scorer = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = scorer
SPEC.loader.exec_module(scorer)


class FakeCatContextBackend:
    model_id = "fake_cat_context"

    def score(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        return {"brushing": 0.2, "waiting_for_food": 0.7, "isolation": 0.1}


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes((np.zeros(3200, dtype=np.int16)).tobytes())


class CatMeowContextModelTests(unittest.TestCase):
    def test_prepare_model_input_matches_ready_model_shape(self) -> None:
        audio = np.zeros(16000, dtype=np.float32)
        features = scorer.prepare_model_input(audio, 16000)

        self.assertEqual(features.shape, (1, 128, 174, 1))
        self.assertEqual(features.dtype, np.float32)

    def test_context_scores_map_to_sher_intents(self) -> None:
        mapped = scorer.intent_scores_from_context({"brushing": 0.2, "waiting_for_food": 0.7, "isolation": 0.1})

        self.assertEqual(mapped["food_or_water"], 0.7)
        self.assertEqual(mapped["lonely"], 0.1)
        self.assertEqual(mapped["complaint"], 0.2)

    def test_score_sidecar_writes_cat_context_teacher_without_audio_tagging_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav_path = root / "clip.wav"
            sidecar_path = root / "clip.json"
            _write_wav(wav_path)
            sidecar_path.write_text(
                json.dumps(
                    {
                        "wav_path": str(wav_path),
                        "cat_meow_candidate": {"event_ready": True, "score": 0.9},
                    }
                ),
                encoding="utf-8",
            )

            status = scorer.score_sidecar(
                sidecar_path,
                backend=FakeCatContextBackend(),
                min_meow_confidence=0.2,
                dry_run=False,
            )

            self.assertEqual(status, "scored")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            context_model = sidecar["cat_vocalization_context_models"][scorer.MODEL_NAME]
            self.assertEqual(context_model["context_label"], "waiting_for_food")
            teacher = sidecar["sher_meow_intent_teacher"][scorer.MODEL_NAME]
            self.assertEqual(teacher["label"], "food_or_water")
            self.assertNotIn("embedding_refs", sidecar)


if __name__ == "__main__":
    unittest.main()
