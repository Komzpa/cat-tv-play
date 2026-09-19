#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_audio_tagging as audio_tagging  # noqa: E402
import cat_tv_play_config as site_config  # noqa: E402
from cat_tv_play_atomic_io import read_json_or_none  # noqa: E402

SEED = 20260506
DEFAULT_INTENTS = [
    "attention",
    "food_or_water",
    "play",
    "lonely",
    "complaint",
    "distress",
    "greeting",
    "unknown_unclear",
]
SHER_INTENT_UX_SCENARIO = "sher_meow_intent"
SHER_INTENT_LEARNING_VALUE = (
    "Adds a diverse confirmed-Sher meow intent example or checks the external cat-context teacher."
)
SHER_INTENT_BENCHMARK_TARGET = "Sher meow intent teacher-vs-human agreement and intent confusion matrix"
INTENT_TEACHER_FEATURES = [
    "teacher_intent:food_or_water",
    "teacher_intent:lonely",
    "teacher_intent:complaint",
    "cat_context:waiting_for_food",
    "cat_context:isolation",
    "cat_context:brushing",
]
HANDCRAFTED_FEATURES = [
    "log_rms",
    "log_peak",
    "zero_crossing_rate",
    "voice_band_ratio",
    "meow_band_ratio",
    "snore_low_band_ratio",
    "snore_mid_band_ratio",
    "high_tonal_ratio",
    "duration_seconds",
    "local_hour_sin",
    "local_hour_cos",
]
CONTEXT_STATES = [
    "input_boolean.sleeping",
    "input_boolean.sleeping_in_living_room",
    "fan.living_room_vent",
    "vacuum.vacuum_x20",
    "sensor.vacuum_x20_status",
    "weather.forecast_home",
]


@dataclass
class MeowRow:
    path: Path
    sidecar: dict[str, Any]
    vector: np.ndarray
    source: str
    confidence: float


def _read_object_or_none(path: Path) -> dict[str, Any] | None:
    payload = read_json_or_none(path)
    return payload if isinstance(payload, dict) else None


def _iter_sidecars(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file() and path.suffix == ".json":
            output.append(path)
        elif path.is_dir():
            output.extend(sorted(path.glob("**/*.json")))
    return output


def _started_at(sidecar: dict[str, Any], path: Path) -> datetime | None:
    value = sidecar.get("started_at")
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    stem = path.name.split("_", 1)[0]
    try:
        return datetime.strptime(stem, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def _local_hour(sidecar: dict[str, Any], path: Path) -> float:
    started_at = _started_at(sidecar, path)
    if started_at is None:
        return 0.0
    return float((started_at.astimezone(UTC).hour + 4) % 24)


def _human_labels(sidecar: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for key in ("human_acoustic_label", "cat_meow_label"):
        value = str(sidecar.get(key) or "").strip().lower()
        if value:
            labels.add(value)
    raw_labels = sidecar.get("human_acoustic_labels")
    if isinstance(raw_labels, list):
        labels.update(str(value or "").strip().lower() for value in raw_labels)
    review_label = sidecar.get("review_label")
    if isinstance(review_label, dict):
        labels.add(str(review_label.get("tag") or "").strip().lower())
        raw_tags = review_label.get("tags")
        if isinstance(raw_tags, list):
            labels.update(str(value or "").strip().lower() for value in raw_tags)
    return {label for label in labels if label}


def _classification(sidecar: dict[str, Any], field: str) -> dict[str, Any]:
    value = sidecar.get(field)
    return value if isinstance(value, dict) else {}


def _meow_evidence(sidecar: dict[str, Any]) -> tuple[str, float]:
    labels = _human_labels(sidecar)
    if "sher_meow" in labels or "cat_meow" in labels or "positive" in labels:
        return "human_sher_meow", 1.0
    if sidecar.get("cat_meow_label") == "positive":
        return "cat_meow_label", 1.0
    candidate = sidecar.get("cat_meow_candidate")
    if isinstance(candidate, dict) and candidate.get("event_ready") is True:
        return "meow_model_event", float(candidate.get("score") or 0.0)
    for field in ("machine_classification_teacher", "machine_classification"):
        classification = _classification(sidecar, field)
        if classification.get("acoustic_label") == "sher_meow":
            return field, float(classification.get("confidence") or 0.0)
    return "", 0.0


def _intent_label(sidecar: dict[str, Any]) -> str:
    payload = sidecar.get("sher_meow_intent")
    if isinstance(payload, dict):
        label = str(payload.get("label") or "").strip()
        if label:
            return label
    review_label = sidecar.get("review_label")
    if isinstance(review_label, dict):
        note = str(review_label.get("note") or "")
        for prefix in ("intent:", "meow_intent:"):
            for token in note.replace(",", " ").split():
                if token.startswith(prefix):
                    return token[len(prefix) :].strip()
    return ""


def _intent_teacher_vector(sidecar: dict[str, Any]) -> list[float]:
    teacher_payload = sidecar.get("sher_meow_intent_teacher")
    if not isinstance(teacher_payload, dict):
        teacher_payload = {}
    model_payload = teacher_payload.get("isolahgvis_cat_meow_classification")
    if not isinstance(model_payload, dict):
        model_payload = {}
    intent_scores = model_payload.get("intent_scores")
    if not isinstance(intent_scores, dict):
        intent_scores = {}

    context_payload = sidecar.get("cat_vocalization_context_models")
    if not isinstance(context_payload, dict):
        context_payload = {}
    context_model = context_payload.get("isolahgvis_cat_meow_classification")
    if not isinstance(context_model, dict):
        context_model = {}
    context_scores = {
        str(row.get("label")): float(row.get("score") or 0.0)
        for row in context_model.get("top_labels", [])
        if isinstance(row, dict)
    }

    return [
        float(intent_scores.get("food_or_water") or 0.0),
        float(intent_scores.get("lonely") or 0.0),
        float(intent_scores.get("complaint") or 0.0),
        float(context_scores.get("waiting_for_food") or 0.0),
        float(context_scores.get("isolation") or 0.0),
        float(context_scores.get("brushing") or 0.0),
    ]


def _state_value(sidecar: dict[str, Any], entity_id: str) -> str:
    context = sidecar.get("ha_context")
    if not isinstance(context, dict):
        return "unknown"
    states = context.get("states")
    if not isinstance(states, dict):
        return "unknown"
    return str(states.get(entity_id) or "unknown")


def _feature_vector(sidecar: dict[str, Any], path: Path, *, audio_tagging_model: str) -> np.ndarray | None:
    features = sidecar.get("features")
    if not isinstance(features, dict):
        return None
    hour = _local_hour(sidecar, path)
    base = [
        math.log1p(float(features.get("rms") or 0.0)),
        math.log1p(float(features.get("peak") or 0.0)),
        float(features.get("zero_crossing_rate") or 0.0),
        float(features.get("voice_band_ratio") or 0.0),
        float(features.get("meow_band_ratio") or 0.0),
        float(features.get("snore_low_band_ratio") or 0.0),
        float(features.get("snore_mid_band_ratio") or 0.0),
        float(features.get("high_tonal_ratio") or 0.0),
        float(sidecar.get("duration_seconds") or 0.0),
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
    ]
    tagging = _classification(sidecar, "audio_tagging_models")
    operational = audio_tagging.audio_tagging_operational_scores(sidecar, audio_tagging_model) if tagging else {}
    operational_vector = [float(operational.get(label) or 0.0) for label in audio_tagging.OPERATIONAL_LABELS]
    teacher_vector = _intent_teacher_vector(sidecar)
    embedding = audio_tagging.load_embedding_ref(sidecar, audio_tagging_model)
    if embedding is None:
        return None
    return np.concatenate(
        [
            np.asarray(base, dtype=np.float32),
            np.asarray(operational_vector, dtype=np.float32),
            np.asarray(teacher_vector, dtype=np.float32),
            embedding.astype(np.float32),
        ]
    )


def _collect_rows(
    paths: list[Path],
    *,
    audio_tagging_model: str,
    min_confidence: float,
) -> tuple[list[MeowRow], dict[str, int]]:
    rows: list[MeowRow] = []
    stats: dict[str, int] = {
        "seen": 0,
        "not_meow": 0,
        "missing_embedding": 0,
        "missing_features": 0,
        "kept": 0,
    }
    for path in _iter_sidecars(paths):
        stats["seen"] += 1
        sidecar = _read_object_or_none(path)
        if sidecar is None:
            continue
        source, confidence = _meow_evidence(sidecar)
        if not source or confidence < min_confidence:
            stats["not_meow"] += 1
            continue
        vector = _feature_vector(sidecar, path, audio_tagging_model=audio_tagging_model)
        if vector is None:
            if isinstance(sidecar.get("features"), dict):
                stats["missing_embedding"] += 1
            else:
                stats["missing_features"] += 1
            continue
        rows.append(MeowRow(path=path, sidecar=sidecar, vector=vector, source=source, confidence=confidence))
        stats["kept"] += 1
    return rows, stats


def _context_summary(sidecar: dict[str, Any]) -> dict[str, str]:
    summary = {entity_id: _state_value(sidecar, entity_id) for entity_id in CONTEXT_STATES}
    context = sidecar.get("ha_context")
    if isinstance(context, dict):
        states = context.get("states")
        if isinstance(states, dict):
            for entity_id, value in states.items():
                text = str(entity_id).lower()
                if any(
                    marker in text
                    for marker in ("feeder", "fountain", "waterer", "pet_food", "desiccant", "sher_s_weight")
                ):
                    summary[str(entity_id)] = str(value)
    return {key: value for key, value in summary.items() if value != "unknown"}


def _write_manifest(
    rows: list[MeowRow],
    *,
    out_dir: Path,
    clusters: np.ndarray,
    distances: np.ndarray,
    limit: int,
    root: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cluster: dict[int, list[tuple[float, MeowRow]]] = {}
    for index, row in enumerate(rows):
        by_cluster.setdefault(int(clusters[index]), []).append((float(distances[index]), row))
    for bucket in by_cluster.values():
        bucket.sort(key=lambda item: item[0])

    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for cluster_id in sorted(by_cluster):
            bucket = by_cluster[cluster_id]
            if round_index >= len(bucket):
                continue
            distance, row = bucket[round_index]
            wav_path = Path(str(row.sidecar.get("wav_path") or row.path.with_suffix(".wav"))).expanduser()
            try:
                relative_path = row.path.resolve().relative_to(root.expanduser().resolve()).as_posix()
            except ValueError:
                relative_path = str(row.path.resolve())
            selected.append(
                {
                    "review_id": f"sher_meow_intent:{len(selected) + 1:03d}",
                    "kind": "sher_meow_intent",
                    "ux_scenario": SHER_INTENT_UX_SCENARIO,
                    "review_focus": "intent_cluster_or_teacher_check",
                    "expected_learning_value": SHER_INTENT_LEARNING_VALUE,
                    "benchmark_target": SHER_INTENT_BENCHMARK_TARGET,
                    "cluster_id": cluster_id,
                    "cluster_distance": round(distance, 6),
                    "sidecar_path": str(row.path),
                    "relative_path": relative_path,
                    "wav_path": str(wav_path),
                    "started_at": row.sidecar.get("started_at"),
                    "source_id": row.sidecar.get("source_id"),
                    "duration_seconds": row.sidecar.get("duration_seconds"),
                    "meow_evidence": row.source,
                    "meow_confidence": round(row.confidence, 6),
                    "current_intent": _intent_label(row.sidecar),
                    "intent_teacher_scores": _intent_teacher_vector(row.sidecar),
                    "intent_options": DEFAULT_INTENTS,
                    "context": _context_summary(row.sidecar),
                    "note": "Label with sher_meow_intent.label, not as a broad acoustic class.",
                }
            )
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
        round_index += 1

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    playlist_path = out_dir / "playlist.m3u"
    playlist_path.write_text(
        "".join(str(item["wav_path"]) + "\n" for item in selected),
        encoding="utf-8",
    )
    return len(selected)


def build_queue(args: argparse.Namespace) -> None:
    rows, stats = _collect_rows(
        args.sidecars,
        audio_tagging_model=args.audio_tagging_model,
        min_confidence=args.min_meow_confidence,
    )
    if len(rows) < 2:
        raise RuntimeError(f"need at least two Sher meow rows with embeddings, got {len(rows)}")
    X = np.asarray([row.vector for row in rows], dtype=np.float32)
    scaled = StandardScaler().fit_transform(X)
    components = min(args.pca_components, scaled.shape[0] - 1, scaled.shape[1])
    reduced = PCA(n_components=components, random_state=SEED).fit_transform(scaled) if components >= 2 else scaled
    cluster_count = min(args.clusters, len(rows))
    model = KMeans(n_clusters=cluster_count, random_state=SEED, n_init=10)
    clusters = model.fit_predict(reduced)
    distances = np.linalg.norm(reduced - model.cluster_centers_[clusters], axis=1)
    selected_count = _write_manifest(
        rows,
        out_dir=args.out_dir,
        clusters=clusters,
        distances=distances,
        limit=args.limit,
        root=args.root,
    )
    report = {
        "kind": "sher_meow_intent_queue",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "stats": stats,
        "candidates": len(rows),
        "rows": selected_count,
        "clusters": cluster_count,
        "pca_components": int(components),
        "intent_options": DEFAULT_INTENTS,
        "manifest": str(args.out_dir / "manifest.jsonl"),
        "playlist": str(args.out_dir / "playlist.m3u"),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train(args: argparse.Namespace) -> None:
    rows, stats = _collect_rows(
        args.sidecars,
        audio_tagging_model=args.audio_tagging_model,
        min_confidence=args.min_meow_confidence,
    )
    labeled = [(row, _intent_label(row.sidecar)) for row in rows]
    labeled = [(row, label) for row, label in labeled if label and label != "unknown_unclear"]
    counts: dict[str, int] = {}
    for _row, label in labeled:
        counts[label] = counts.get(label, 0) + 1
    keep = {label for label, count in counts.items() if count >= args.min_class_count}
    labeled = [(row, label) for row, label in labeled if label in keep]
    if len(keep) < 2:
        raise RuntimeError(f"need at least two intent classes with >= {args.min_class_count} labels; counts={counts}")
    X = np.asarray([row.vector for row, _label in labeled], dtype=np.float32)
    y = np.asarray([label for _row, label in labeled])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("install catboost to train Sher meow intent model") from exc
    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test).reshape(-1).astype(str)
    model_path = args.out.with_suffix(".cbm")
    model.save_model(str(model_path))
    payload = {
        "kind": "sher_meow_intent_catboost_v1",
        "model_path": str(model_path),
        "audio_tagging_model": args.audio_tagging_model,
        "feature_names": HANDCRAFTED_FEATURES
        + [f"panns_operational:{label}" for label in audio_tagging.OPERATIONAL_LABELS]
        + INTENT_TEACHER_FEATURES
        + [
            f"embedding:{index}"
            for index in range(
                len(labeled[0][0].vector)
                - len(HANDCRAFTED_FEATURES)
                - len(audio_tagging.OPERATIONAL_LABELS)
                - len(INTENT_TEACHER_FEATURES)
            )
        ],
        "classes": [str(label) for label in model.classes_],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "training_counts": {
            "total": len(labeled),
            "classes": {label: int((y == label).sum()) for label in sorted(set(y))},
            "dropped_classes": {label: count for label, count in counts.items() if label not in keep},
            "source_stats": stats,
        },
        "validation": {
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        },
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["training_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["validation"], ensure_ascii=False, indent=2)[:4000])
    print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and train Sher meow intent models over the shared room-audio corpus."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-queue", help="cluster likely Sher meows into a diverse intent review queue")
    build.add_argument("sidecars", nargs="+", type=Path)
    build.add_argument("--root", type=Path, default=site_config.audio_events_root())
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--audio-tagging-model", default="panns_cnn14")
    build.add_argument("--min-meow-confidence", type=float, default=0.20)
    build.add_argument("--clusters", type=int, default=50)
    build.add_argument("--limit", type=int, default=100)
    build.add_argument("--pca-components", type=int, default=32)
    build.set_defaults(func=build_queue)

    train_parser = subparsers.add_parser("train", help="train a CatBoost intent head from sher_meow_intent labels")
    train_parser.add_argument("sidecars", nargs="+", type=Path)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--audio-tagging-model", default="panns_cnn14")
    train_parser.add_argument("--min-meow-confidence", type=float, default=0.0)
    train_parser.add_argument("--min-class-count", type=int, default=5)
    train_parser.add_argument("--iterations", type=int, default=200)
    train_parser.add_argument("--depth", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=0.08)
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
