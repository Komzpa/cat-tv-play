#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import wave
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_audio_tagging as audio_tagging  # noqa: E402
import cat_tv_play_config as site_config  # noqa: E402

DEFAULT_ROOT = site_config.audio_events_root()
DEFAULT_MODEL_CACHE = site_config.model_cache_root() / "isolahgvis-cat-meow-classification"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_CACHE / "best_model_fold_2.h5"
MODEL_ID = "IsolaHGVIS/Cat-Meow-Classification@87ce8fa412a0bc5413baad1bbc29fa7a7d8e32aa"
MODEL_URL = "https://huggingface.co/IsolaHGVIS/Cat-Meow-Classification/resolve/main/best_model_fold_2.h5"
MODEL_NAME = "isolahgvis_cat_meow_classification"
CLASSES = ["brushing", "waiting_for_food", "isolation"]
SAMPLE_RATE = 16000


class CatMeowContextBackend(Protocol):
    model_id: str

    def score(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]: ...


def _iter_sidecars(root: Path, *, room: str = "") -> list[Path]:
    root = root.expanduser()
    if root.is_file() and root.suffix == ".json":
        return [root]
    if room:
        return sorted((root / room).glob("*.json"))
    return sorted(root.glob("*/*.json"))


def _read_hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    for path in (Path("~/.cache/huggingface/token").expanduser(), Path("~/.huggingface/token").expanduser()):
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return ""


def download_model(path: Path) -> Path:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = _read_hf_token()
    request = urllib.request.Request(MODEL_URL)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with urllib.request.urlopen(request, timeout=600) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp_path.replace(path)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                "IsolaHGVIS/Cat-Meow-Classification is gated on Hugging Face; "
                "accept the model conditions and set HF_TOKEN before downloading"
            ) from exc
        raise
    return path


def _read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(f"unsupported WAV sample width {sample_width}: {path}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    gcd = int(np.gcd(source_rate, target_rate))
    return signal.resample_poly(audio, target_rate // gcd, source_rate // gcd).astype(np.float32)


def _hz_to_mel(frequency: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(*, sample_rate: int, n_fft: int, n_mels: int, fmax: float) -> np.ndarray:
    max_hz = min(float(fmax), sample_rate / 2.0)
    mel_points = np.linspace(_hz_to_mel(np.asarray([0.0]))[0], _hz_to_mel(np.asarray([max_hz]))[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = int(bins[index]), int(bins[index + 1]), int(bins[index + 2])
        if center > left:
            filters[index, left:center] = (np.arange(left, center) - left) / float(center - left)
        if right > center:
            filters[index, center:right] = (right - np.arange(center, right)) / float(right - center)
    return filters


def prepare_model_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = _resample(np.asarray(audio, dtype=np.float32), sample_rate, SAMPLE_RATE)
    if audio.size == 0:
        audio = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    n_fft = 2048
    hop = 512
    padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode="reflect" if audio.size > 2 else "constant")
    _, _, stft = signal.stft(
        padded,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(stft).astype(np.float32) ** 2
    mel = _mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=n_fft, n_mels=128, fmax=8000.0) @ power
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    mel_db -= float(np.max(mel_db))
    if mel_db.shape[1] > 174:
        mel_db = mel_db[:, :174]
    elif mel_db.shape[1] < 174:
        mel_db = np.pad(mel_db, ((0, 0), (0, 174 - mel_db.shape[1])))
    return mel_db.astype(np.float32).reshape(1, 128, 174, 1)


class KerasCatMeowContextBackend:
    model_id = MODEL_ID

    def __init__(self, model_path: Path) -> None:
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as exc:
            raise RuntimeError("tensorflow is required for the cat meow context model") from exc
        self._model = tf.keras.models.load_model(model_path.expanduser(), compile=False)

    def score(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        features = prepare_model_input(audio, sample_rate)
        predictions = np.asarray(self._model.predict(features, verbose=0)[0], dtype=np.float32)
        total = float(np.sum(predictions))
        if total > 0:
            predictions = predictions / total
        return {label: round(float(score), 6) for label, score in zip(CLASSES, predictions, strict=False)}


def intent_scores_from_context(context_scores: dict[str, float]) -> dict[str, float]:
    return {
        "food_or_water": float(context_scores.get("waiting_for_food", 0.0)),
        "lonely": float(context_scores.get("isolation", 0.0)),
        "complaint": float(context_scores.get("brushing", 0.0)),
    }


def _has_meow_evidence(sidecar: dict[str, object], *, min_confidence: float) -> bool:
    cat_candidate = sidecar.get("cat_meow_candidate")
    if isinstance(cat_candidate, dict) and cat_candidate.get("event_ready") is True:
        try:
            return float(cat_candidate.get("score") or 0.0) >= min_confidence
        except (TypeError, ValueError):
            return False
    human_labels = sidecar.get("human_acoustic_labels")
    if isinstance(human_labels, list) and any(str(label) in {"sher_meow", "cat_meow"} for label in human_labels):
        return True
    for field in ("machine_classification_teacher", "machine_classification"):
        classification = sidecar.get(field)
        if not isinstance(classification, dict):
            continue
        if classification.get("acoustic_label") != "sher_meow":
            continue
        try:
            return float(classification.get("confidence") or 0.0) >= min_confidence
        except (TypeError, ValueError):
            return False
    return False


def _wav_path(sidecar: dict[str, object], sidecar_path: Path) -> Path:
    return Path(str(sidecar.get("wav_path") or sidecar_path.with_suffix(".wav"))).expanduser()


def update_sidecar_with_context_scores(
    sidecar: dict[str, object],
    *,
    wav_path: Path,
    audio_hash: str,
    scores: dict[str, float],
) -> None:
    top_labels = [
        {"label": label, "score": round(float(score), 6)}
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]
    best_label, best_score = top_labels[0]["label"], float(top_labels[0]["score"])
    intent_scores = intent_scores_from_context(scores)
    cat_models = sidecar.setdefault("cat_vocalization_context_models", {})
    if not isinstance(cat_models, dict):
        cat_models = {}
        sidecar["cat_vocalization_context_models"] = cat_models
    cat_models[MODEL_NAME] = {
        "model_id": MODEL_ID,
        "scored_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "wav_path": str(wav_path),
        "audio_sha256": audio_hash,
        "top_labels": top_labels,
        "context_label": best_label,
        "context_score": round(best_score, 6),
        "intent_scores": {label: round(score, 6) for label, score in intent_scores.items() if score > 0.0},
    }
    teacher = sidecar.setdefault("sher_meow_intent_teacher", {})
    if not isinstance(teacher, dict):
        teacher = {}
        sidecar["sher_meow_intent_teacher"] = teacher
    if intent_scores:
        intent_label, intent_score = max(intent_scores.items(), key=lambda item: item[1])
    else:
        intent_label, intent_score = "unknown_unclear", 0.0
    teacher[MODEL_NAME] = {
        "label": intent_label,
        "confidence": round(float(intent_score), 6),
        "intent_scores": {label: round(score, 6) for label, score in intent_scores.items() if score > 0.0},
        "source_context_label": best_label,
        "source_context_score": round(best_score, 6),
    }


def score_sidecar(
    path: Path,
    *,
    backend: CatMeowContextBackend,
    min_meow_confidence: float,
    dry_run: bool,
) -> str:
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if not _has_meow_evidence(sidecar, min_confidence=min_meow_confidence):
        return "not_meow"
    wav_path = _wav_path(sidecar, path)
    if not wav_path.exists():
        return "missing_wav"
    audio_hash = audio_tagging.sidecar_audio_hash(wav_path)
    models = sidecar.get("cat_vocalization_context_models")
    if isinstance(models, dict):
        existing = models.get(MODEL_NAME)
        if isinstance(existing, dict) and existing.get("audio_sha256") == audio_hash:
            return "unchanged"
    sample_rate, audio = _read_wav_mono(wav_path)
    scores = backend.score(audio, sample_rate)
    if dry_run:
        return "would_score"
    update_sidecar_with_context_scores(sidecar, wav_path=wav_path, audio_hash=audio_hash, scores=scores)
    path.write_text(audio_tagging.json_dumps(sidecar), encoding="utf-8")
    return "scored"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Sher meows with a cat vocalization context model.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--room", default="")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-meow-confidence", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path.expanduser()
    if args.download and not model_path.exists():
        try:
            download_model(model_path)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    if not model_path.exists():
        raise SystemExit(
            f"model file is missing: {model_path}. Run with --download after accepting the gated "
            "Hugging Face model conditions and setting HF_TOKEN."
        )

    backend = KerasCatMeowContextBackend(model_path)
    stats: Counter[str] = Counter()
    processed = 0
    for path in _iter_sidecars(args.root, room=args.room):
        if args.limit and processed >= args.limit:
            break
        try:
            status = score_sidecar(
                path,
                backend=backend,
                min_meow_confidence=args.min_meow_confidence,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            print(f"{path}: {type(exc).__name__}: {exc}", flush=True)
        stats[status] += 1
        processed += 1

    print(
        json.dumps(
            {
                "model": MODEL_NAME,
                "model_id": MODEL_ID,
                "root": str(args.root.expanduser()),
                "dry_run": args.dry_run,
                "processed": processed,
                "stats": dict(stats),
                "finished_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
