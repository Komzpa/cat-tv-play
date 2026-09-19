#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_audio_tagging as audio_tagging  # noqa: E402

#: Handcrafted meow features, named by the audio project that extracts them.
#: Resolved lazily so ``--help`` works without that project installed.
MEOW_FEATURE_NAMES = audio_tagging.meow_feature_names


SEED = 20260504
MEOW_POSITIVE_LABELS = {"sher_meow", "meow", "cat_meow"}
MEOW_NEGATIVE_LABELS = {"not_meow"}


def _feature_vector(sidecar: dict[str, object]) -> list[float] | None:
    features = sidecar.get("features")
    if not isinstance(features, dict):
        return None
    duration_seconds = float(sidecar.get("duration_seconds") or 0.0)
    return [
        math.log1p(float(features.get("rms", 0.0))),
        math.log1p(float(features.get("peak", 0.0))),
        float(features.get("zero_crossing_rate", 0.0)),
        float(features.get("voice_band_ratio", 0.0)),
        float(features.get("meow_band_ratio", 0.0)),
        float(features.get("snore_low_band_ratio", 0.0)),
        float(features.get("snore_mid_band_ratio", 0.0)),
        float(features.get("high_tonal_ratio", 0.0)),
        duration_seconds,
    ]


def _human_acoustic_labels(sidecar: dict[str, object]) -> set[str]:
    labels: set[str] = set()
    human_labels = sidecar.get("human_acoustic_labels")
    if isinstance(human_labels, list):
        labels.update(str(value or "").strip().lower() for value in human_labels)
    human_label = str(sidecar.get("human_acoustic_label") or "").strip().lower()
    if human_label:
        labels.add(human_label)
    return {label for label in labels if label}


def _training_label(
    sidecar: dict[str, object],
    *,
    include_legacy_event_ready: bool = False,
) -> tuple[int | None, str]:
    explicit = str(sidecar.get("cat_meow_label") or sidecar.get("meow_label") or "").strip().lower()
    if explicit in {"positive", *MEOW_POSITIVE_LABELS}:
        return 1, "explicit_meow_label"
    if explicit in {"negative", *MEOW_NEGATIVE_LABELS}:
        return 0, "explicit_meow_label"

    human_labels = _human_acoustic_labels(sidecar)
    if human_labels & MEOW_POSITIVE_LABELS:
        return 1, "human_acoustic_meow_label"
    if human_labels:
        return 0, "human_acoustic_not_meow_label"

    candidate = sidecar.get("cat_meow_candidate")
    wake_label = sidecar.get("wake_label")
    label_status = sidecar.get("label_status")
    if (
        include_legacy_event_ready
        and isinstance(candidate, dict)
        and candidate.get("event_ready")
        and wake_label != "positive"
    ):
        return 1, "legacy_event_ready_seed"

    if wake_label == "negative" or label_status in {"negative", "human_reviewed", "human_assisted_reviewed"}:
        return 0, "human_negative_audio_review"

    return None, ""


def _iter_sidecars(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            output.append(path)
        elif path.is_dir():
            output.extend(sorted(path.glob("**/*.json")))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sher meow classifier from shared VIGI sidecars.")
    parser.add_argument("sidecars", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--include-legacy-event-ready", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in _iter_sidecars(args.sidecars):
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        vector = _feature_vector(sidecar)
        if vector is None:
            continue
        label, source = _training_label(
            sidecar,
            include_legacy_event_ready=args.include_legacy_event_ready,
        )
        if label is None:
            continue
        rows.append({"path": str(path), "x": vector, "y": label, "source": source})

    if len({row["y"] for row in rows}) < 2:
        raise RuntimeError("need both positive and negative meow labels")

    X = np.asarray([row["x"] for row in rows], dtype=np.float32)
    y = np.asarray([row["y"] for row in rows], dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    model.fit(X_train_scaled, y_train)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = probabilities >= args.threshold

    source_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = source_counts.setdefault(str(row["source"]), {"positive": 0, "negative": 0})
        bucket["positive" if row["y"] else "negative"] += 1

    payload = {
        "kind": "sher_meow_logistic_v1",
        "feature_names": MEOW_FEATURE_NAMES(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "threshold": args.threshold,
        "seed": SEED,
        "training_counts": {
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "sources": source_counts,
        },
        "validation": {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(y_test, predictions, output_dict=True),
        },
        "notes": [
            "Weakly supervised from the shared acoustic-event sidecars; event emission uses this model "
            "probability, not tonal if-gates.",
            "Legacy event_ready seeds are disabled by default because they can self-reinforce false meow positives.",
            "Wake-word negatives and human-reviewed scrape/keyboard/music clips are negative acoustic examples.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["training_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["validation"], ensure_ascii=False, indent=2)[:4000])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
