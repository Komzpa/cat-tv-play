#!/usr/bin/env python3
"""Local Cat Projector label-review backend.

This server owns the durable review state for the Cat Projector calibrator UI:
it lists existing frames from the local corpus, saves cat/not-cat labels and
portable masks, provides a local click-to-contour segmentation fallback, and
runs or queues explicit local retrain/rescore actions.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "web"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_dataset_root() -> Path:
    configured = os.environ.get("CAT_TV_LEARNING_ROOT")
    if configured:
        return Path(configured).expanduser()

    repo_dataset = REPO_ROOT / "datasets" / "cat-tv-learning"
    if repo_dataset.exists():
        return repo_dataset

    tasks_loop_dataset = Path("~/Dropbox/Reemxy/tasks-loop/datasets/cat-tv-learning").expanduser()
    if tasks_loop_dataset.exists():
        return tasks_loop_dataset

    return repo_dataset


DATASET_ROOT = _default_dataset_root()
STATE_ROOT = Path("~/.openclaw/state/cat-tv-learning").expanduser()
REVIEW_ROOT = STATE_ROOT / "label-review"
LABELS_ROOT = REVIEW_ROOT / "labels"
MASKS_ROOT = REVIEW_ROOT / "masks"
QUEUE_ROOT = REVIEW_ROOT / "actions"
JOBS_ROOT = REVIEW_ROOT / "jobs"
VIDEO_STATUS_ROOT = REVIEW_ROOT / "videos"
TRAINING_DATASETS_ROOT = STATE_ROOT / "datasets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov"}
LABEL_NAMESPACE = "cat_projector_label_review"
SAM_ENDPOINT = os.environ.get("CAT_PROJECTOR_SAM_ENDPOINT", "http://127.0.0.1:8766/segment").strip()
OUTPUT_FRAME_DIR_NAMES = {
    "model-output",
    "model_output",
    "output",
    "outputs",
    "annotated",
    "annotated_frames",
    "previous-model",
    "previous_model",
}
INPUT_FRAME_DIR_NAMES = {
    "frames",
    "input",
    "inputs",
    "original",
    "original_frames",
    "raw",
    "source2",
    "source",
}
PREFERRED_INPUT_FRAME_DIR_NAMES = (
    "raw",
    "frames",
    "input",
    "inputs",
    "original",
    "original_frames",
    "source2",
    "source",
)
PREFERRED_OUTPUT_FRAME_DIR_NAMES = (
    "model-output",
    "model_output",
    "annotated_frames",
    "annotated",
    "previous-model",
    "previous_model",
    "output",
    "outputs",
)

SCAN_ROOTS = (
    DATASET_ROOT / "calibration-captures",
    DATASET_ROOT / "datasets",
    DATASET_ROOT / "detector-training",
    STATE_ROOT / "jump-review",
    STATE_ROOT / "batch_reviews",
    STATE_ROOT / "telegram-clips",
    STATE_ROOT / "recordings",
    STATE_ROOT / "datasets",
)

ALLOWED_ROOTS = (
    WEB_ROOT,
    DATASET_ROOT,
    STATE_ROOT,
    REVIEW_ROOT,
)

_REPROCESSED_OUTPUT_CACHE: tuple[Path, tuple[tuple[Path, frozenset[str], tuple[Path, ...]], ...]] | None = None
_RECORDING_CHUNK_INDEX_CACHE: dict[tuple[Path, str], dict[int, tuple[Path, ...]]] = {}
_DISCOVERY_CACHE_TTL_SECONDS = 30.0
_DISCOVER_CASES_CACHE: tuple[float, tuple[str, ...], tuple[ReviewCase, ...]] | None = None
_DISCOVER_VIDEOS_CACHE: tuple[float, tuple[str, ...], tuple[ReviewVideo, ...]] | None = None
_JUMP_HEIGHT_INDEX_CACHE: tuple[float, Path, dict[str, dict[str, Any]]] | None = None
ALLOW_LIVE_JOBS = False
_ACTIVE_JOB_ID: str | None = None
_JOB_LOCK = threading.Lock()
LIVE_JOB_COMMAND: list[str] | None = None
_CURRENT_MODEL_ON_DEMAND_CACHE: dict[Path, dict[str, Any] | None] = {}
_CURRENT_MODEL_INFO_CACHE: dict[str, Any] | None = None
_CURRENT_MODEL_DETECTOR: Any | None = None
_PROJECTOR_SOURCE_OFFSET_CACHE: dict[tuple[Path, str], tuple[float, float, tuple[float, ...]]] = {}
_SOURCE_VIDEO_CAPTURE_CACHE: dict[str, tuple[Any, float]] = {}
_CAT_DETECTION_MODULE: Any | None = None
_CAT_MEASUREMENT_MODULE: Any | None = None
DEFAULT_REVIEW_FRAME_FPS = 30.0
LEGACY_TELEGRAM_REVIEW_FRAME_FPS = 15.0


@dataclass(frozen=True)
class ReviewCase:
    id: str
    image_path: Path
    label: str
    source: str
    mtime: float
    source_video_path: Path | None = None
    source_recording_dir: Path | None = None
    detector_probability: float | None = None
    candidate_bbox_xywh: tuple[float, float, float, float] | None = None
    review_status: str = "unreviewed"
    human_label: str | None = None
    notes: str = ""

    def uncertainty_score(self) -> float:
        if self.review_status in {"saved", "reviewed"}:
            return -1000.0
        probability = self.detector_probability
        if probability is not None:
            return 1.0 - min(1.0, abs(probability - 0.5) * 2.0)
        if self.review_status == "unsure":
            return 0.85
        return 0.35

    def priority_tuple(self) -> tuple[float, float]:
        return (self.uncertainty_score(), self.mtime)


@dataclass(frozen=True)
class ReviewVideo:
    id: str
    label: str
    source: str
    mtime: float
    source_recording_dir: Path | None = None
    source_video_path: Path | None = None
    frame_count: int = 0
    output_frame_count: int = 0
    output_artifacts: tuple[Path, ...] = ()
    review_status: str = "unreviewed"
    notes: str = ""
    max_jump_height_cm: float | None = None
    max_jump_height_source: str = ""

    def priority_tuple(self) -> tuple[float, float]:
        status_penalty = -1000.0 if self.review_status in {"relabeled_ok", "reviewed", "ok"} else 0.0
        return (status_penalty + min(1.0, self.frame_count / 250.0), self.mtime)

    def height_priority_tuple(self) -> tuple[float, float, float, float]:
        height = self.max_jump_height_cm
        return (
            1.0 if height is not None else 0.0,
            float(height) if height is not None else -1.0,
            *self.priority_tuple(),
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _discovery_cache_key() -> tuple[str, ...]:
    return tuple(str(root) for root in SCAN_ROOTS)


def _clear_discovery_caches() -> None:
    global _CURRENT_MODEL_DETECTOR, _CURRENT_MODEL_INFO_CACHE, _DISCOVER_CASES_CACHE, _DISCOVER_VIDEOS_CACHE, _JUMP_HEIGHT_INDEX_CACHE
    _DISCOVER_CASES_CACHE = None
    _DISCOVER_VIDEOS_CACHE = None
    _JUMP_HEIGHT_INDEX_CACHE = None
    _CURRENT_MODEL_DETECTOR = None
    _CURRENT_MODEL_INFO_CACHE = None
    _CURRENT_MODEL_ON_DEMAND_CACHE.clear()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug[:180] or "cat_projector_label_review"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_local_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not any(_is_within(resolved, root) for root in ALLOWED_ROOTS):
        raise ValueError(f"path is outside allowed corpus roots: {path}")
    return resolved


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _ffprobe_frame_rate(path: Path) -> float | None:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0] if isinstance(streams[0], dict) else {}
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = str(stream.get(key) or "")
        if "/" in raw:
            numerator_raw, denominator_raw = raw.split("/", 1)
            try:
                numerator = float(numerator_raw)
                denominator = float(denominator_raw)
            except ValueError:
                continue
            if denominator and numerator > 0:
                return numerator / denominator
        else:
            value = _float_or_none(raw)
            if value and value > 0:
                return value
    return None


def _chunk_frame_rate(path: Path | None) -> float:
    if path is None:
        return DEFAULT_REVIEW_FRAME_FPS
    return _ffprobe_frame_rate(path) or DEFAULT_REVIEW_FRAME_FPS


def _ffprobe_frame_times(path: Path) -> list[float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time,pkt_pts_time",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []
    times: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        value = _float_or_none(frame.get("best_effort_timestamp_time"))
        if value is None:
            value = _float_or_none(frame.get("pkt_pts_time"))
        if value is not None:
            times.append(value)
    return times


def _source_frame_index_for_offset(path: Path, offset_seconds: float) -> int:
    times = _ffprobe_frame_times(path)
    if times:
        return min(range(len(times)), key=lambda index: (abs(times[index] - offset_seconds), index))
    return max(0, int(round(offset_seconds * _chunk_frame_rate(path))))


def _height_index_keys_for_path(path: Path | None) -> set[str]:
    if path is None:
        return set()
    keys = {str(path), path.name}
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser()
    keys.update({str(resolved), resolved.name})
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        keys.update(_height_index_keys_for_path(path.parent))
    return {key for key in keys if key}


def _add_height_index_entry(
    index: dict[str, dict[str, Any]],
    *,
    path: Path | None,
    video_id: str | None = None,
    height_cm: float | None,
    source: str,
) -> None:
    if height_cm is None:
        return
    keys = _height_index_keys_for_path(path)
    if video_id:
        keys.add(video_id)
    for key in keys:
        existing = index.get(key)
        if existing is None or height_cm > float(existing.get("max_jump_height_cm") or -1):
            index[key] = {
                "max_jump_height_cm": round(height_cm, 1),
                "max_jump_height_source": source,
            }


def _height_from_record(record: dict[str, Any]) -> float | None:
    for key in ("max_jump_height_cm", "max_height_cm", "cat_top_height_cm", "best_top_height_cm"):
        value = _float_or_none(record.get(key))
        if value is not None:
            return value
    for key in ("max_jump_height_mm", "max_height_mm", "cat_top_height_mm"):
        value = _float_or_none(record.get(key))
        if value is not None:
            return value / 10.0
    scan = record.get("scan") if isinstance(record.get("scan"), dict) else None
    if scan:
        return _height_from_record(scan)
    render_stats = record.get("render_stats") if isinstance(record.get("render_stats"), dict) else None
    if render_stats:
        return _height_from_record(render_stats)
    return None


def _index_height_record(index: dict[str, dict[str, Any]], record: dict[str, Any], *, source: str) -> None:
    height_cm = _height_from_record(record)
    max_frame_path = record.get("max_frame_path") or record.get("observation_path")
    if _saved_label_says_no_cat(max_frame_path):
        return
    path_values: list[Any] = [
        record.get("recording_dir"),
        record.get("recording"),
        record.get("source"),
        record.get("source_key"),
        record.get("source_video_path"),
        max_frame_path,
    ]
    max_record = record.get("max_record") if isinstance(record.get("max_record"), dict) else None
    if max_record:
        max_record_frame_path = max_record.get("max_frame_path") or max_record.get("observation_path")
        if _saved_label_says_no_cat(max_record_frame_path):
            return
        path_values.extend([max_record.get("recording_dir"), max_record.get("recording"), max_record.get("source")])
        height_cm = height_cm if height_cm is not None else _height_from_record(max_record)
    video_id = str(record.get("video_id") or "") or None
    for raw_path in path_values:
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        _add_height_index_entry(index, path=path, video_id=video_id, height_cm=height_cm, source=source)


def _index_jump_height_payload(source_path: Path, data: Any) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    records: list[Any] = []
    if isinstance(data, dict):
        if isinstance(data.get("videos"), list):
            records.extend(data["videos"])
        if isinstance(data.get("records"), list):
            records.extend(data["records"])
        records.append(data)
    elif isinstance(data, list):
        records.extend(data)
    for record in records:
        if isinstance(record, dict):
            _index_height_record(index, record, source=str(source_path))
    return index


def _build_jump_height_index() -> dict[str, dict[str, Any]]:
    global _JUMP_HEIGHT_INDEX_CACHE
    now = time.monotonic()
    if _JUMP_HEIGHT_INDEX_CACHE is not None:
        cached_at, cached_state_root, cached_index = _JUMP_HEIGHT_INDEX_CACHE
        if cached_state_root == STATE_ROOT and now - cached_at <= _DISCOVERY_CACHE_TTL_SECONDS:
            return dict(cached_index)

    latest_heights = STATE_ROOT / "label-review" / "rescores" / "jump_heights_latest.json"
    if latest_heights.exists():
        index = _index_jump_height_payload(latest_heights, _read_json(latest_heights))
        if index:
            _JUMP_HEIGHT_INDEX_CACHE = (now, STATE_ROOT, dict(index))
            return index
        rescore_root = latest_heights.parent
        previous = sorted(
            (path for path in rescore_root.glob("*/jump_heights.json") if path.exists()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in previous:
            index = _index_jump_height_payload(path, _read_json(path))
            if index:
                _JUMP_HEIGHT_INDEX_CACHE = (now, STATE_ROOT, dict(index))
                return index
        _JUMP_HEIGHT_INDEX_CACHE = (now, STATE_ROOT, {})
        return {}

    index: dict[str, dict[str, Any]] = {}
    candidate_files: list[Path] = []
    for root, patterns in (
        (STATE_ROOT / "batch_reviews", ("**/scan_results.json", "**/render_summary*.json", "**/summary.json")),
        (STATE_ROOT / "recordings", ("*/telegram*_notification.json",)),
        (STATE_ROOT / "jump-heights", ("*/measurements*.json",)),
        (STATE_ROOT / "jump-observations", ("*.measurements.json",)),
    ):
        if root.exists():
            for pattern in patterns:
                candidate_files.extend(sorted(root.glob(pattern)))

    seen: set[Path] = set()
    for path in candidate_files:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        index.update(_index_jump_height_payload(path, _read_json(path)))
    _JUMP_HEIGHT_INDEX_CACHE = (now, STATE_ROOT, dict(index))
    return index


def _jump_height_for_video(
    height_index: dict[str, dict[str, Any]],
    *,
    video_id: str,
    recording_dir: Path | None,
    group_key: Path | None,
    source_video: Path | None,
) -> tuple[float | None, str]:
    for key in (
        video_id,
        *sorted(_height_index_keys_for_path(recording_dir)),
        *sorted(_height_index_keys_for_path(source_video)),
        *sorted(_height_index_keys_for_path(group_key)),
    ):
        row = height_index.get(key)
        if not row:
            continue
        return _float_or_none(row.get("max_jump_height_cm")), str(row.get("max_jump_height_source") or "")
    return None, ""


def _encode_path(path: Path) -> str:
    raw = str(_safe_local_path(path)).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_path(token: str) -> Path:
    padding = "=" * (-len(token) % 4)
    raw = base64.urlsafe_b64decode((token + padding).encode("ascii")).decode("utf-8")
    return _safe_local_path(Path(raw))


def _case_id_for_path(path: Path) -> str:
    digest = hashlib.sha256(str(_safe_local_path(path)).encode("utf-8")).hexdigest()[:18]
    return f"case.{digest}"


def _video_id_for_path(path: Path) -> str:
    digest = hashlib.sha256(str(_safe_local_path(path)).encode("utf-8")).hexdigest()[:18]
    return f"video.{digest}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    _clear_discovery_caches()


def _label_path(case_id: str) -> Path:
    return LABELS_ROOT / f"{_safe_slug(case_id)}.json"


def _mask_path(case_id: str, mask_id: str) -> Path:
    return MASKS_ROOT / _safe_slug(case_id) / f"{_safe_slug(mask_id)}.json"


def _video_status_path(video_id: str) -> Path:
    return VIDEO_STATUS_ROOT / f"{_safe_slug(video_id)}.json"


def _job_path(job_id: str) -> Path:
    return JOBS_ROOT / f"{_safe_slug(job_id)}.json"


def _load_label_for_case(case_id: str) -> dict[str, Any]:
    return _read_json(_label_path(case_id))


def _load_status_for_video(video_id: str) -> dict[str, Any]:
    return _read_json(_video_status_path(video_id))


def _parse_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        try:
            return (float(raw["x"]), float(raw["y"]), float(raw["width"]), float(raw["height"]))
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        parts = [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (float(part) for part in parts)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label_rows_by_image(root: Path) -> dict[Path, dict[str, str]]:
    rows: dict[Path, dict[str, str]] = {}
    for csv_path in root.rglob("labels.csv"):
        for row in _read_csv_rows(csv_path):
            image_relpath = row.get("image_relpath") or row.get("image") or ""
            if not image_relpath:
                continue
            image_path = (csv_path.parent / image_relpath).resolve()
            rows[image_path] = row
    return rows


def _probe_rows_by_image(root: Path) -> dict[Path, dict[str, str]]:
    rows: dict[Path, dict[str, str]] = {}
    if not root.exists():
        return rows
    for probe_path in root.rglob("probe_rows.json"):
        try:
            payload = json.loads(probe_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            raw_path = row.get("raw_path")
            if not raw_path:
                continue
            try:
                image_path = _safe_local_path(Path(str(raw_path)))
            except ValueError:
                continue
            rows[image_path] = {
                "detector_cat_probability": str(row.get("best_probability") or ""),
                "candidate_bbox_xywh": str(row.get("best_bbox") or ""),
                "candidate_reason": str(row.get("best_source") or "probe_rows"),
                "detector_backend": str(row.get("detector_backend") or ""),
                "detector_model_id": str(row.get("detector_model_id") or ""),
                "measurement_source": str(row.get("measurement_source") or ""),
                "measurement_confidence": str(row.get("measurement_confidence") or ""),
                "best_top_height_cm": str(row.get("best_top_height_cm") or ""),
                "best_top_wall_x_cm": str(row.get("best_top_wall_x_cm") or ""),
                "legacy_bbox_top_height_cm": str(row.get("legacy_bbox_top_height_cm") or ""),
                "best_measurement_point": row.get("best_measurement_point"),
                "best_measurement_warning": str(row.get("best_measurement_warning") or ""),
                "tracker_status": str(row.get("tracker_status") or ""),
                "tracker_reason": str(row.get("tracker_reason") or ""),
                "tracker_confirmed": row.get("tracker_confirmed"),
                "tracker_height_cm": str(row.get("tracker_height_cm") or ""),
                "review_priority_score": str(row.get("review_priority_score") or ""),
                "review_priority_reasons": row.get("review_priority_reasons")
                if isinstance(row.get("review_priority_reasons"), list)
                else [],
                "notes": f"probe_rows: {row.get('candidate_count', '?')} candidates",
            }
    return rows


def _merge_missing_metadata_rows(
    rows_by_image: dict[Path, dict[str, Any]],
    fallback_rows_by_image: dict[Path, dict[str, Any]],
) -> None:
    for image_path, fallback_row in fallback_rows_by_image.items():
        row = rows_by_image.setdefault(image_path, {})
        for key, value in fallback_row.items():
            if value in (None, "", []):
                continue
            if row.get(key) in (None, "", []):
                row[key] = value


def _current_model_info() -> dict[str, Any]:
    global _CURRENT_MODEL_INFO_CACHE
    if _CURRENT_MODEL_INFO_CACHE is not None:
        return _CURRENT_MODEL_INFO_CACHE
    manifest = _read_json(REVIEW_ROOT / "rescores" / "latest.json")
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    _CURRENT_MODEL_INFO_CACHE = {
        "model_created_at": str(model.get("created_at") or ""),
        "model_path": str(model.get("model_path") or ""),
    }
    return _CURRENT_MODEL_INFO_CACHE


def _model_overlay_from_row(row: dict[str, Any], *, role: str, label: str) -> dict[str, Any] | None:
    bbox = _parse_bbox(row.get("candidate_bbox_xywh") or row.get("best_bbox"))
    measurement_point = (
        row.get("best_measurement_point") if isinstance(row.get("best_measurement_point"), dict) else None
    )
    probability = None
    for key in ("detector_cat_probability", "cat_probability", "probability", "best_probability"):
        if row.get(key) not in {None, ""}:
            probability = _float_or_none(row.get(key))
            break
    top_height_cm = _float_or_none(row.get("best_top_height_cm"))
    top_wall_x_cm = _float_or_none(row.get("best_top_wall_x_cm"))
    backend = str(row.get("detector_backend") or "")
    model_id = str(row.get("detector_model_id") or "")
    measurement_source = str(row.get("measurement_source") or "")
    if role in {"current_model", "original_model"} and bbox is None and measurement_point is None:
        return None
    if not any((bbox, measurement_point, probability is not None, top_height_cm is not None, backend, model_id)):
        return None
    model_created_at = str(row.get("model_created_at") or "")
    model_path = str(row.get("model_path") or row.get("detector_model_path") or row.get("detector_model_id") or "")
    if role == "current_model" and not model_created_at:
        current_model = _current_model_info()
        model_created_at = str(current_model.get("model_created_at") or "")
        if not model_path:
            model_path = str(current_model.get("model_path") or "")
    return {
        "role": role,
        "label": label,
        "bbox_xywh": bbox,
        "detector_probability": probability,
        "detector_backend": backend,
        "detector_model_id": model_id,
        "model_created_at": model_created_at,
        "model_path": model_path,
        "measurement_source": measurement_source,
        "measurement_confidence": _float_or_none(row.get("measurement_confidence")),
        "top_height_cm": top_height_cm,
        "top_wall_x_cm": top_wall_x_cm,
        "measurement_point": measurement_point,
        "measurement_warning": str(row.get("best_measurement_warning") or ""),
        "review_priority_score": _float_or_none(row.get("review_priority_score")),
        "review_priority_reasons": row.get("review_priority_reasons")
        if isinstance(row.get("review_priority_reasons"), list)
        else [],
    }


def _row_model_probability(row: dict[str, Any]) -> float | None:
    for key in ("detector_cat_probability", "cat_probability", "probability", "best_probability"):
        if row.get(key) not in {None, ""}:
            value = _float_or_none(row.get(key))
            if value is not None:
                return value
    return None


def _trusted_measurement_row(row: dict[str, Any]) -> bool:
    if not row or _float_or_none(row.get("best_top_height_cm")) is None:
        return False
    backend = str(row.get("detector_backend") or "")
    if backend.startswith("telegram_"):
        return True
    probability = _row_model_probability(row)
    measurement_source = str(row.get("measurement_source") or "")
    warning = str(row.get("best_measurement_warning") or "")
    legacy_bbox = measurement_source == "legacy_bbox_top" or "legacy_bbox" in warning or "missing_segmentation" in warning
    if legacy_bbox and (probability is None or probability < 0.5):
        return False
    if probability is not None and probability < 0.5:
        return False
    return True


def _display_measurement_row(
    metadata_row: dict[str, Any],
    current_row: dict[str, Any],
    original_row: dict[str, Any],
) -> dict[str, Any]:
    for row in (metadata_row, current_row, original_row):
        if _trusted_measurement_row(row):
            return row
    return {}


def _telegram_render_overlay_row(
    recording_dir: Path,
    overlay: dict[str, Any],
    *,
    frame_times_by_chunk: dict[int, list[float]] | None = None,
    frame_rate_by_chunk: dict[int, float] | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    frame_label = str(overlay.get("source_frame_label") or "")
    if not frame_label and overlay.get("chunk") not in (None, "") and overlay.get("offset_seconds") not in (None, ""):
        try:
            chunk_index_for_label = int(overlay["chunk"])
            offset_seconds = float(overlay.get("offset_seconds") or 0.0)
            chunk_path = recording_dir / f"chunk_{chunk_index_for_label:04d}.mp4"
            if chunk_path.exists():
                if frame_times_by_chunk is not None:
                    if chunk_index_for_label not in frame_times_by_chunk:
                        frame_times_by_chunk[chunk_index_for_label] = _ffprobe_frame_times(chunk_path)
                    frame_times = frame_times_by_chunk[chunk_index_for_label]
                else:
                    frame_times = _ffprobe_frame_times(chunk_path)
                if frame_times:
                    source_frame_index = min(
                        range(len(frame_times)),
                        key=lambda index: (abs(frame_times[index] - offset_seconds), index),
                    )
                else:
                    if frame_rate_by_chunk is not None:
                        if chunk_index_for_label not in frame_rate_by_chunk:
                            frame_rate_by_chunk[chunk_index_for_label] = _chunk_frame_rate(chunk_path)
                        frame_rate = frame_rate_by_chunk[chunk_index_for_label]
                    else:
                        frame_rate = _chunk_frame_rate(chunk_path)
                    source_frame_index = max(0, int(round(offset_seconds * frame_rate)))
                frame_label = f"chunk_{chunk_index_for_label:04d}_{source_frame_index:05d}.jpg"
        except (TypeError, ValueError):
            frame_label = ""
    if not frame_label:
        frame_label = str(overlay.get("frame_label") or "")
    if not frame_label:
        try:
            frame_label = f"chunk_{int(overlay['chunk']):04d}_{int(overlay['frame_index']):05d}.jpg"
        except (KeyError, TypeError, ValueError):
            return None
    if not re.match(r"chunk_\d{4}_\d{5}\.jpg\Z", frame_label):
        return None
    image_path = recording_dir / "review_frames" / frame_label
    if not image_path.exists():
        return None
    try:
        image_path = _safe_local_path(image_path)
    except ValueError:
        return None
    bbox_xywh: tuple[float, float, float, float] | None = None
    bbox = overlay.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if x2 > x1 and y2 > y1:
                bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
        except (TypeError, ValueError):
            bbox_xywh = None
    top_px = overlay.get("top_px")
    measurement_point: dict[str, Any] | None = None
    if isinstance(top_px, (list, tuple)) and len(top_px) >= 2:
        try:
            measurement_point = {
                "image_x": float(top_px[0]),
                "image_y": float(top_px[1]),
                "wall_y_cm": _float_or_none(overlay.get("height_cm")),
                "wall_x_cm": (
                    float(overlay["wall_x_mm"]) / 10.0 if overlay.get("wall_x_mm") not in (None, "") else None
                ),
                "point_type": "telegram_render_original_top",
                "source": "telegram_render_original",
                "debug": {
                    "recording_dir": str(recording_dir),
                    "frame_label": frame_label,
                    "render_frame_label": overlay.get("render_frame_label") or overlay.get("frame_label"),
                    "chunk": overlay.get("chunk"),
                    "frame_index": overlay.get("frame_index"),
                    "offset_seconds": overlay.get("offset_seconds"),
                },
            }
        except (TypeError, ValueError):
            measurement_point = None
    model_path = str(overlay.get("model_path") or "")
    model_id = Path(model_path).name if model_path else "cat_projector_telegram_render"
    return image_path, {
        "detector_cat_probability": str(overlay.get("cat_probability") or ""),
        "candidate_bbox_xywh": _bbox_dict_to_csv(bbox_xywh) if bbox_xywh else "",
        "candidate_reason": str(overlay.get("reason") or "telegram_render_original"),
        "detector_backend": "telegram_render_original",
        "detector_model_id": model_id,
        "measurement_source": "telegram_render_original",
        "measurement_confidence": "",
        "best_top_height_cm": str(overlay.get("height_cm") or ""),
        "best_top_wall_x_cm": (
            str(float(overlay["wall_x_mm"]) / 10.0) if overlay.get("wall_x_mm") not in (None, "") else ""
        ),
        "best_measurement_point": measurement_point,
        "best_measurement_warning": "",
        "review_priority_score": "8",
        "review_priority_reasons": ["telegram render original model"],
        "notes": f"telegram render original: {overlay.get('height_cm', '?')} cm",
    }


def _telegram_render_stats_for_payload(payload: dict[str, Any], recording_dir: Path) -> dict[str, Any] | None:
    render_stats = payload.get("render_stats")
    if isinstance(render_stats, dict) and isinstance(render_stats.get("original_model_overlays"), list):
        return render_stats
    for marker_name in ("telegram_live_notification.json", "telegram_notification.json"):
        marker_path = recording_dir / marker_name
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(marker, dict):
            continue
        marker_render_stats = marker.get("render_stats")
        if isinstance(marker_render_stats, dict) and isinstance(
            marker_render_stats.get("original_model_overlays"),
            list,
        ):
            return marker_render_stats
    return render_stats if isinstance(render_stats, dict) else None


def _telegram_jump_highlight_rows_by_image() -> dict[Path, dict[str, Any]]:
    rows: dict[Path, dict[str, Any]] = {}
    sessions_path = STATE_ROOT / "sessions.jsonl"
    if not sessions_path.exists():
        return rows
    try:
        lines = sessions_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if "jump_highlight" not in line or "telegram" not in line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        action = str(payload.get("telegram_live_action") or payload.get("telegram_action") or "")
        kind = str(payload.get("kind") or "")
        if action != "sent" and not kind.endswith("_notification_sent"):
            continue
        highlight = payload.get("jump_highlight")
        if not isinstance(highlight, dict):
            continue
        recording_value = payload.get("recording_dir") or highlight.get("recording_dir")
        if not recording_value:
            continue
        try:
            recording_dir = _safe_local_path(Path(str(recording_value)))
        except ValueError:
            continue
        review_frames_dir = recording_dir / "review_frames"
        if not review_frames_dir.exists():
            continue
        render_stats = _telegram_render_stats_for_payload(payload, recording_dir)
        if isinstance(render_stats, dict) and isinstance(render_stats.get("original_model_overlays"), list):
            frame_times_by_chunk: dict[int, list[float]] = {}
            frame_rate_by_chunk: dict[int, float] = {}
            for overlay in render_stats["original_model_overlays"]:
                if not isinstance(overlay, dict):
                    continue
                row = _telegram_render_overlay_row(
                    recording_dir,
                    overlay,
                    frame_times_by_chunk=frame_times_by_chunk,
                    frame_rate_by_chunk=frame_rate_by_chunk,
                )
                if row is None:
                    continue
                image_path, image_row = row
                rows[image_path] = image_row
        try:
            chunk_index = int(highlight["chunk"])
            offset_seconds = float(highlight.get("offset_seconds") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        chunk_path = recording_dir / f"chunk_{chunk_index:04d}.mp4"
        if chunk_path.exists():
            expected_frame = float(_source_frame_index_for_offset(chunk_path, offset_seconds))
        else:
            expected_frame = offset_seconds * DEFAULT_REVIEW_FRAME_FPS
        peak_frame_index = max(0, int(round(expected_frame)))
        candidate_paths = [review_frames_dir / f"chunk_{chunk_index:04d}_{peak_frame_index:05d}.jpg"]
        bbox_xywh: tuple[float, float, float, float] | None = None
        bbox = highlight.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = (float(value) for value in bbox)
                if x2 > x1 and y2 > y1:
                    bbox_xywh = (x1, y1, x2 - x1, y2 - y1)
            except (TypeError, ValueError):
                bbox_xywh = None
        top_px = highlight.get("top_px")
        measurement_point: dict[str, Any] | None = None
        if isinstance(top_px, (list, tuple)) and len(top_px) >= 2:
            try:
                measurement_point = {
                    "image_x": float(top_px[0]),
                    "image_y": float(top_px[1]),
                    "wall_y_cm": _float_or_none(highlight.get("height_cm")),
                    "wall_x_cm": (
                        float(highlight["wall_x_mm"]) / 10.0
                        if highlight.get("wall_x_mm") not in (None, "")
                        else None
                    ),
                    "point_type": "telegram_jump_highlight_top",
                    "source": "telegram_jump_highlight",
                    "debug": {
                        "recording_dir": str(recording_dir),
                        "chunk": chunk_index,
                        "offset_seconds": offset_seconds,
                    },
                }
            except (TypeError, ValueError):
                measurement_point = None
        for image_path in candidate_paths:
            if not image_path.exists():
                continue
            try:
                image_path = _safe_local_path(image_path)
            except ValueError:
                continue
            if image_path in rows:
                continue
            match = re.match(r"chunk_\d+_(\d+)\.", image_path.name)
            frame_index = int(match.group(1)) if match else peak_frame_index
            is_peak_frame = frame_index == peak_frame_index
            rows[image_path] = {
                "detector_cat_probability": str(highlight.get("cat_probability") or ""),
                "candidate_bbox_xywh": _bbox_dict_to_csv(bbox_xywh) if bbox_xywh else "",
                "candidate_reason": str(highlight.get("reason") or "telegram_jump_highlight"),
                "detector_backend": "telegram_jump_highlight",
                "detector_model_id": "cat_projector_telegram_notification",
                "measurement_source": "telegram_jump_highlight",
                "measurement_confidence": "",
                "best_top_height_cm": str(highlight.get("height_cm") or "") if is_peak_frame else "",
                "best_top_wall_x_cm": (
                    str(float(highlight["wall_x_mm"]) / 10.0)
                    if is_peak_frame and highlight.get("wall_x_mm") not in (None, "")
                    else ""
                ),
                "best_measurement_point": measurement_point,
                "best_measurement_warning": "",
                "review_priority_score": "35" if is_peak_frame else "2",
                "review_priority_reasons": ["telegram jump highlight"] if is_peak_frame else ["telegram jump highlight context"],
                "notes": (
                    f"telegram jump highlight: {highlight.get('height_cm', '?')} cm"
                    if is_peak_frame
                    else f"telegram jump highlight context; peak frame {peak_frame_index}"
                ),
            }
    return rows


def _current_model_rows_by_image() -> dict[Path, dict[str, Any]]:
    latest_rescore_root = _latest_rescore_root()
    if latest_rescore_root is None:
        return {}
    return _probe_rows_by_image(latest_rescore_root)


def _load_repo_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _cat_detection_module() -> Any:
    global _CAT_DETECTION_MODULE
    if _CAT_DETECTION_MODULE is None:
        _CAT_DETECTION_MODULE = _load_repo_module(
            "cat_tv_play_detection_for_label_review_server",
            REPO_ROOT / "custom_components" / "cat_tv_play" / "detection.py",
        )
    return _CAT_DETECTION_MODULE


def _cat_measurement_module() -> Any:
    global _CAT_MEASUREMENT_MODULE
    if _CAT_MEASUREMENT_MODULE is None:
        _CAT_MEASUREMENT_MODULE = _load_repo_module(
            "cat_tv_play_measurement_for_label_review_server",
            REPO_ROOT / "custom_components" / "cat_tv_play" / "measurement.py",
        )
    return _CAT_MEASUREMENT_MODULE


def _measurement_for_detection(detection: Any) -> tuple[dict[str, Any] | None, str]:
    cat_measurement = _cat_measurement_module()

    if getattr(detection, "mask", None) is not None:
        point = cat_measurement.mask_top_measurement_point(detection.mask, score=detection.score)
        if point is not None:
            return cat_measurement.measurement_point_to_dict(point), ""
        point = cat_measurement.legacy_bbox_top_measurement_point(
            detection.bbox_xywh,
            score=detection.score,
        )
        return cat_measurement.measurement_point_to_dict(point), "segmentation_mask_rejected_using_legacy_bbox_top"
    point = cat_measurement.legacy_bbox_top_measurement_point(detection.bbox_xywh, score=detection.score)
    return cat_measurement.measurement_point_to_dict(point), "missing_segmentation_mask_using_legacy_bbox_top"


def _current_model_detector() -> Any | None:
    global _CURRENT_MODEL_DETECTOR
    if _CURRENT_MODEL_DETECTOR is not None:
        return _CURRENT_MODEL_DETECTOR
    try:
        from scripts import cat_projector_frame_detector as detector
        cat_detection = _cat_detection_module()
    except Exception as exc:
        print(f"[cat-projector-label-review] current model detector import failed: {exc}", flush=True)
        return None

    manifest = _read_json(REVIEW_ROOT / "rescores" / "latest.json")
    detector_config = manifest.get("detector_config") if isinstance(manifest.get("detector_config"), dict) else {}
    backend = str(detector_config.get("backend") or "legacy")
    model_path = detector.DEFAULT_MODEL_PATH
    metadata_path = detector.DEFAULT_METADATA_PATH
    model_meta = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    if backend == "legacy" and model_meta.get("model_path"):
        model_path = Path(str(model_meta["model_path"])).expanduser()
        metadata_path = model_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            metadata_path = detector.DEFAULT_METADATA_PATH
    try:
        if backend == "legacy":
            _CURRENT_MODEL_DETECTOR = cat_detection.LegacyContrastDetector(
                model_path=model_path,
                metadata_path=metadata_path,
                min_probability=0.0,
            )
        else:
            _CURRENT_MODEL_DETECTOR = cat_detection.build_detector(
                cat_detection.DetectorConfig(
                    backend=backend,
                    model_path=detector_config.get("model_path"),
                    device=detector_config.get("device"),
                    confidence_threshold=float(detector_config.get("confidence_threshold") or 0.5),
                    legacy_model_path=str(detector.DEFAULT_MODEL_PATH),
                    legacy_metadata_path=str(detector.DEFAULT_METADATA_PATH),
                    allow_fake=bool(detector_config.get("allow_fake")),
                )
            )
    except Exception as exc:
        print(f"[cat-projector-label-review] current model detector init failed: {exc}", flush=True)
        _CURRENT_MODEL_DETECTOR = None
    return _CURRENT_MODEL_DETECTOR


def _source_video_for_recording(recording_dir: Path) -> str | None:
    manifest = _manifest_for_recording_dir(recording_dir)
    video_url = str(manifest.get("video_url") or "").strip()
    if not video_url:
        return None
    parsed = urlparse(video_url)
    basename = Path(parsed.path).name
    local_candidates = [
        REPO_ROOT / "tmp" / "cat-tv-youtube-trimmed" / basename,
        REPO_ROOT / "tmp" / "cat-tv-youtube-seeds" / basename,
        Path("/config/www/cat-tv") / basename,
        STATE_ROOT / "source-videos" / basename,
        STATE_ROOT / "videos" / basename,
    ]
    for candidate in local_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return str(candidate)
    if parsed.scheme in {"http", "https"}:
        return video_url
    return None


def _source_video_capture(source_video: str) -> tuple[Any, float] | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    cached = _SOURCE_VIDEO_CAPTURE_CACHE.get(source_video)
    if cached is not None:
        return cached
    capture = cv2.VideoCapture(source_video)
    if not capture.isOpened():
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    _SOURCE_VIDEO_CAPTURE_CACHE[source_video] = (capture, duration)
    return capture, duration


def _source_frame_at(source_video: str, seconds: float) -> Image.Image | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    cached = _source_video_capture(source_video)
    if cached is None:
        return None
    capture, duration = cached
    seek_seconds = max(0.0, float(seconds))
    if duration > 1.0:
        seek_seconds %= duration
    capture.set(cv2.CAP_PROP_POS_MSEC, seek_seconds * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _review_frame_elapsed_seconds(recording_dir: Path, image_path: Path) -> float | None:
    match = re.match(r"(?:chunk|source)_(\d+)_(\d+)\.(?:jpe?g|png|webp)$", image_path.name, re.IGNORECASE)
    if not match:
        return None
    chunk_index = int(match.group(1))
    frame_index = int(match.group(2))
    chunk_path = recording_dir / f"chunk_{chunk_index:04d}.mp4"
    if not chunk_path.exists():
        return None
    frame_times = _ffprobe_frame_times(chunk_path)
    if frame_times:
        offset_seconds = frame_times[min(frame_index, len(frame_times) - 1)]
    else:
        offset_seconds = frame_index / _chunk_frame_rate(chunk_path)
    manifest = _manifest_for_recording_dir(recording_dir)
    segment_seconds = _float_or_none(manifest.get("segment_seconds")) or 5.0
    return chunk_index * segment_seconds + offset_seconds


def _projector_source_match_score(camera_gray: Any, source_gray: Any, detector_module: Any) -> float:
    import cv2  # type: ignore[import-not-found]
    import numpy as local_np

    height, width = camera_gray.shape
    source_points = local_np.float32([[0, 0], [1279, 0], [1279, 719], [0, 719]])
    camera_points = local_np.float32(detector_module._projector_polygon_for_size(width, height))
    homography = cv2.getPerspectiveTransform(source_points, camera_points)
    warped = cv2.warpPerspective(cv2.resize(source_gray, (1280, 720)), homography, (width, height))
    fit_mask = detector_module._projection_mask(width, height)
    fit_mask[int(height * 0.69) :, :] = False
    fit_mask[:, int(width * 0.82) :] = False
    scale, offset = detector_module._robust_linear_match(warped, camera_gray, fit_mask)
    expected = local_np.clip(scale * warped.astype(local_np.float32) + offset, 0, 255)
    diff = local_np.abs(expected.astype(local_np.int16) - camera_gray.astype(local_np.int16))[fit_mask]
    return float(local_np.percentile(diff, 75)) + float(local_np.mean(diff)) * 0.2


def _best_projector_source_time(
    camera_frame: Image.Image,
    source_video: str,
    detector_module: Any,
) -> tuple[float, float, list[float]] | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    cached = _source_video_capture(source_video)
    if cached is None:
        return None
    capture, duration = cached
    if duration <= 1.0:
        return None
    camera_gray = np.asarray(camera_frame.convert("L"), dtype=np.uint8)

    def score_at(second: float) -> tuple[float, float] | None:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, second) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        source_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return (_projector_source_match_score(camera_gray, source_gray, detector_module), second)

    coarse: list[tuple[float, float]] = []
    step = 5.0
    second = 0.0
    while second < duration:
        scored = score_at(second)
        if scored is not None:
            coarse.append(scored)
        second += step
    if not coarse:
        return None
    best_second = min(coarse, key=lambda item: item[0])[1]
    refined: list[tuple[float, float]] = []
    for delta_index in range(-20, 21):
        candidate_second = (best_second + delta_index * 0.25) % duration
        scored = score_at(candidate_second)
        if scored is not None:
            refined.append(scored)
    best_score, best_time = min(refined or coarse, key=lambda item: item[0])
    reference_times = {best_time}
    for _score, coarse_second in sorted(coarse, key=lambda item: item[0])[:5]:
        reference_times.add(coarse_second)
    for _score, coarse_second in sorted(coarse, key=lambda item: item[0])[:3]:
        for delta in (-2.0, -1.0, 0.0, 1.0, 2.0):
            reference_times.add((coarse_second + delta) % duration)
    return best_time, best_score, sorted(reference_times)


def _source_frame_for_review_image(image_path: Path) -> tuple[Image.Image, dict[str, Any]] | None:
    _chunk_path, recording_dir = _recording_context(image_path)
    if recording_dir is None:
        return None
    source_video = _source_video_for_recording(recording_dir)
    if not source_video:
        return None
    elapsed = _review_frame_elapsed_seconds(recording_dir, image_path)
    if elapsed is None:
        return None
    detector_module = __import__("scripts.cat_projector_frame_detector", fromlist=["dummy"])
    cache_key = (recording_dir.resolve(), source_video)
    cached_offset = _PROJECTOR_SOURCE_OFFSET_CACHE.get(cache_key)
    if cached_offset is None:
        sync_path = recording_dir / "review_source_sync.json"
        sync = _read_json(sync_path)
        if sync.get("source_video") == source_video and sync.get("source_offset_seconds") not in {None, ""}:
            cached_offset = (
                float(sync["source_offset_seconds"]),
                float(sync.get("source_sync_match_score") or 0.0),
                tuple(float(value) for value in sync.get("reference_times_seconds") or []),
            )
            _PROJECTOR_SOURCE_OFFSET_CACHE[cache_key] = cached_offset
    if cached_offset is None:
        with Image.open(image_path) as image:
            match = _best_projector_source_time(image.convert("RGB"), source_video, detector_module)
        if match is None:
            return None
        best_time, match_score, reference_times = match
        capture = _source_video_capture(source_video)
        duration = capture[1] if capture is not None else 0.0
        offset = best_time - elapsed
        if duration > 1.0:
            offset %= duration
        cached_offset = (offset, match_score, tuple(reference_times))
        _PROJECTOR_SOURCE_OFFSET_CACHE[cache_key] = cached_offset
        try:
            _write_json(
                recording_dir / "review_source_sync.json",
                {
                    "kind": "cat_projector_review_source_sync_v1",
                    "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "source_video": source_video,
                    "source_offset_seconds": round(offset, 6),
                    "source_sync_match_score": round(match_score, 6),
                    "reference_times_seconds": [round(value, 3) for value in reference_times],
                    "calibration_frame": str(image_path),
                },
            )
        except Exception:
            pass
    source_offset, match_score, cached_reference_times = cached_offset
    reference_times = list(cached_reference_times)
    capture = _source_video_capture(source_video)
    duration = capture[1] if capture is not None else 0.0
    source_seconds = elapsed + source_offset
    if duration > 1.0:
        source_seconds %= duration
    source_frame = _source_frame_at(source_video, source_seconds)
    if source_frame is None:
        return None
    reference_frames = [
        frame
        for second in reference_times
        if (frame := _source_frame_at(source_video, second)) is not None
    ]
    return source_frame, {
        "recording_dir": str(recording_dir),
        "source_video": source_video,
        "recording_elapsed_seconds": round(elapsed, 3),
        "source_seconds": round(source_seconds, 3),
        "source_offset_seconds": round(source_offset, 3),
        "source_sync_match_score": round(match_score, 4),
        "projector_source_reference_frames": reference_frames,
    }


def _current_model_row_for_image(image_path: Path) -> dict[str, Any] | None:
    try:
        image_path = _safe_local_path(image_path)
    except ValueError:
        return None
    if image_path in _CURRENT_MODEL_ON_DEMAND_CACHE:
        return _CURRENT_MODEL_ON_DEMAND_CACHE[image_path]
    detector = _current_model_detector()
    if detector is None:
        _CURRENT_MODEL_ON_DEMAND_CACHE[image_path] = None
        return None
    used_source_subtraction = False
    try:
        cat_detection = _cat_detection_module()

        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        context_debug: dict[str, Any] = {}
        if getattr(detector, "source", "") == "legacy_contrast_catboost":
            source = _source_frame_for_review_image(image_path)
            if source is not None:
                source_frame, source_debug = source
                context_debug.update(source_debug)
                context_debug["projector_source_frame"] = source_frame
        used_source_subtraction = "projector_source_frame" in context_debug
        context = cat_detection.DetectorContext(source_path=str(image_path), debug=context_debug)
        detections = sorted(detector.detect(rgb, context), key=lambda item: item.score, reverse=True)
    except Exception:
        print(f"[cat-projector-label-review] current model on-demand failed: {image_path}", flush=True)
        _CURRENT_MODEL_ON_DEMAND_CACHE[image_path] = None
        return None

    top_candidates: list[dict[str, Any]] = []
    for detection in detections[:12]:
        point, warning = _measurement_for_detection(detection)
        row = detection.to_debug_row()
        row["measurement_point"] = point
        row["measurement_warning"] = warning
        top_candidates.append(row)
    best = top_candidates[0] if top_candidates else {}
    if used_source_subtraction and _float_or_none(best.get("p")) is not None and float(best["p"]) < 0.5:
        best = {}
    latest_manifest = _read_json(REVIEW_ROOT / "rescores" / "latest.json")
    latest_model = latest_manifest.get("model") if isinstance(latest_manifest.get("model"), dict) else {}
    detector_backend = getattr(detector, "source", "unknown")
    best_debug = best.get("debug") if isinstance(best.get("debug"), dict) else {}
    if best_debug.get("backend"):
        detector_backend = str(best_debug["backend"])
    elif used_source_subtraction:
        detector_backend = "legacy_contrast_catboost_source_subtracted"
    result = {
        "raw_path": str(image_path),
        "detector_backend": detector_backend,
        "detector_model_id": getattr(detector, "model_id", ""),
        "model_created_at": str(latest_model.get("created_at") or ""),
        "model_path": str(latest_model.get("model_path") or getattr(detector, "model_id", "")),
        "detector_cat_probability": str(best.get("p") or ""),
        "candidate_bbox_xywh": str(best.get("bbox") or ""),
        "candidate_reason": str(best.get("source") or "current_model_on_demand"),
        "candidate_count": str(len(detections)),
        "best_probability": str(best.get("p") or ""),
        "best_bbox": str(best.get("bbox") or ""),
        "best_source": str(best.get("source") or ""),
        "best_measurement_point": best.get("measurement_point"),
        "best_measurement_warning": str(best.get("measurement_warning") or ""),
        "measurement_source": str((best.get("measurement_point") or {}).get("point_type") or ""),
        "measurement_confidence": str((best.get("measurement_point") or {}).get("confidence") or ""),
        "notes": f"current model on-demand: {len(detections)} candidates",
        "top_candidates": top_candidates,
    }
    _CURRENT_MODEL_ON_DEMAND_CACHE[image_path] = result
    return result


def _original_model_rows_by_image() -> dict[Path, dict[str, Any]]:
    return _telegram_jump_highlight_rows_by_image()


def _latest_rescore_root() -> Path | None:
    rescores_root = STATE_ROOT / "label-review" / "rescores"
    latest_path = rescores_root / "latest.json"
    if latest_path.exists():
        manifest = _read_json(latest_path)
        probe_rows = manifest.get("probe_rows")
        if probe_rows:
            probe_path = Path(str(probe_rows)).expanduser()
            if probe_path.exists():
                return probe_path.parent
    if not rescores_root.exists():
        return None
    runs = [path for path in rescores_root.iterdir() if path.is_dir() and (path / "probe_rows.json").exists()]
    if not runs:
        return None
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _review_metadata_rows_by_image() -> dict[Path, dict[str, Any]]:
    rows_by_image: dict[Path, dict[str, Any]] = {}
    for root in (STATE_ROOT / "datasets", DATASET_ROOT / "datasets", DATASET_ROOT / "detector-training"):
        if root.exists():
            rows_by_image.update(_label_rows_by_image(root))
    for root in (
        STATE_ROOT / "batch_reviews",
        STATE_ROOT / "jump-review",
        DATASET_ROOT / "detector-training",
    ):
        if root.exists():
            rows_by_image.update(_probe_rows_by_image(root))
    rows_by_image.update(_current_model_rows_by_image())
    _merge_missing_metadata_rows(rows_by_image, _original_model_rows_by_image())
    return rows_by_image


def _is_output_frame_dir_name(name: str) -> bool:
    return (
        name in OUTPUT_FRAME_DIR_NAMES
        or name.startswith("annotated_")
        or name.endswith("_annotated_frames")
        or "candidate_thumb" in name
        or "review_sheet" in name
        or name == "sheets"
        or name == "thumbs"
    )


def _is_input_frame_dir_name(name: str) -> bool:
    return name in INPUT_FRAME_DIR_NAMES


def _is_review_artifact_image(path: Path) -> bool:
    if _is_output_frame_dir_name(path.parent.name):
        return True
    name = path.name.lower()
    return any(
        marker in name
        for marker in (
            "sheet",
            "report",
            "overlay",
            "hold_aspect",
            "candidate_thumb",
            "annotated_",
        )
    )


def _is_generated_review_training_path(path: Path) -> bool:
    return any(
        part.startswith("cat-projector-review-ui-") or part.startswith("cat-projector-ui-") for part in path.parts
    )


def _manifest_for_recording_dir(recording_dir: Path) -> dict[str, Any]:
    return _read_json(recording_dir / "manifest.json")


def _recording_dir_for_timestamp(timestamp: str) -> Path | None:
    recordings_root = STATE_ROOT / "recordings"
    if not recordings_root.exists() or not timestamp:
        return None
    matches = sorted(recordings_root.glob(f"{timestamp}_*"))
    for candidate in matches:
        if candidate.is_dir() and (candidate / "manifest.json").exists():
            return candidate.resolve()
    return None


def _batch_review_date(path: Path) -> str | None:
    for parent in path.parents:
        match = re.match(r".*?(?P<date>\d{8})$", parent.name)
        if match:
            return match.group("date")
    return None


def _chunk_index_for_frame(path: Path) -> int | None:
    match = re.match(r"(?:chunk|source)_(\d+)_\d+\.(?:jpe?g|png|webp)$", path.name, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _reprocessed_outputs() -> tuple[tuple[Path, frozenset[str], tuple[Path, ...]], ...]:
    global _REPROCESSED_OUTPUT_CACHE
    reprocessed_root = STATE_ROOT / "reprocessed"
    if _REPROCESSED_OUTPUT_CACHE is not None and _REPROCESSED_OUTPUT_CACHE[0] == reprocessed_root:
        return _REPROCESSED_OUTPUT_CACHE[1]
    outputs: list[tuple[Path, frozenset[str], tuple[Path, ...]]] = []
    if reprocessed_root.exists():
        output_dirs = sorted(
            (path for path in reprocessed_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for output_dir in output_dirs:
            manifest = _read_json(output_dir / "manifest.json")
            values = frozenset(_string_values(manifest))
            artifacts: list[Path] = []
            for artifact in sorted(output_dir.glob("*.mp4")):
                try:
                    artifacts.append(_safe_local_path(artifact))
                except ValueError:
                    continue
            outputs.append((output_dir, values, tuple(artifacts)))
    _REPROCESSED_OUTPUT_CACHE = (reprocessed_root, tuple(outputs))
    return _REPROCESSED_OUTPUT_CACHE[1]


def _recording_chunk_index_for_date(batch_date: str) -> dict[int, tuple[Path, ...]]:
    cache_key = (STATE_ROOT, batch_date)
    cached = _RECORDING_CHUNK_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    recordings_root = STATE_ROOT / "recordings"
    index: dict[int, list[Path]] = {}
    if recordings_root.exists():
        for recording_dir in recordings_root.glob(f"{batch_date}T*"):
            if not recording_dir.is_dir():
                continue
            resolved = recording_dir.resolve()
            for chunk_path in recording_dir.glob("chunk_*.mp4"):
                match = re.match(r"chunk_(\d+)\.mp4$", chunk_path.name, re.IGNORECASE)
                if not match:
                    continue
                index.setdefault(int(match.group(1)), []).append(resolved)
    frozen = {
        chunk_index: tuple(sorted(recording_dirs, key=lambda candidate: candidate.stat().st_mtime, reverse=True))
        for chunk_index, recording_dirs in index.items()
    }
    _RECORDING_CHUNK_INDEX_CACHE[cache_key] = frozen
    return frozen


def _reprocessed_manifest_matches_recording(recording_dir: Path) -> bool:
    recording_dir_raw = str(recording_dir)
    recording_dir_resolved = str(recording_dir.resolve())
    for _output_dir, values, _artifacts in _reprocessed_outputs():
        if recording_dir_raw in values or recording_dir_resolved in values:
            return True
    return False


def _recording_dir_for_batch_chunk(path: Path) -> Path | None:
    chunk_index = _chunk_index_for_frame(path)
    batch_date = _batch_review_date(path)
    if chunk_index is None or batch_date is None:
        return None
    candidates = list(_recording_chunk_index_for_date(batch_date).get(chunk_index, ()))
    if not candidates:
        return None
    artifact_matches = [candidate for candidate in candidates if _reprocessed_manifest_matches_recording(candidate)]
    if artifact_matches:
        return sorted(artifact_matches, key=lambda candidate: candidate.stat().st_mtime, reverse=True)[0]
    return sorted(candidates, key=lambda candidate: candidate.stat().st_mtime, reverse=True)[0]


def _batch_review_recording_context(path: Path) -> tuple[Path | None, Path | None]:
    if not any(parent.name == "batch_reviews" for parent in path.parents):
        return None, None
    for parent in path.parents:
        match = re.match(r"(?P<timestamp>\d{8}T\d{6})_", parent.name)
        if not match:
            continue
        recording_dir = _recording_dir_for_timestamp(match.group("timestamp"))
        if recording_dir is None:
            continue
        chunk_match = re.match(r"chunk_(\d+)_\d+\.(?:jpe?g|png|webp)$", path.name, re.IGNORECASE)
        if chunk_match:
            chunk_path = recording_dir / f"chunk_{int(chunk_match.group(1)):04d}.mp4"
            if chunk_path.exists():
                return chunk_path.resolve(), recording_dir
        chunks = sorted(
            candidate
            for candidate in recording_dir.iterdir()
            if candidate.suffix.lower() in VIDEO_EXTENSIONS and ".part." not in candidate.name
        )
        return (chunks[0] if chunks else None, recording_dir)
    recording_dir = _recording_dir_for_batch_chunk(path)
    if recording_dir is not None:
        chunk_index = _chunk_index_for_frame(path)
        chunk_path = recording_dir / f"chunk_{chunk_index:04d}.mp4" if chunk_index is not None else None
        if chunk_path is not None and chunk_path.exists():
            return chunk_path.resolve(), recording_dir
        chunks = sorted(
            candidate
            for candidate in recording_dir.iterdir()
            if candidate.suffix.lower() in VIDEO_EXTENSIONS and ".part." not in candidate.name
        )
        return (chunks[0] if chunks else None, recording_dir)
    return None, None


def _recording_context(path: Path) -> tuple[Path | None, Path | None]:
    for parent in path.parents:
        if (parent / "manifest.json").exists() and parent.parent.name == "recordings":
            chunks = sorted(
                candidate
                for candidate in parent.iterdir()
                if candidate.suffix.lower() in VIDEO_EXTENSIONS and ".part." not in candidate.name
            )
            return (chunks[0] if chunks else None, parent)
    return _batch_review_recording_context(path)


def _source_name(path: Path) -> str:
    for root in SCAN_ROOTS:
        if root.exists() and _is_within(path, root):
            return root.name
    return "corpus"


def _image_size(path: Path) -> dict[str, int] | None:
    try:
        with Image.open(path) as image:
            return {"width": int(image.width), "height": int(image.height)}
    except Exception:
        return None


def _discover_cases(limit: int) -> list[ReviewCase]:
    global _DISCOVER_CASES_CACHE
    cache_key = _discovery_cache_key()
    now = time.monotonic()
    if _DISCOVER_CASES_CACHE is not None:
        cached_at, cached_key, cached_cases = _DISCOVER_CASES_CACHE
        if cached_key == cache_key and now - cached_at <= _DISCOVERY_CACHE_TTL_SECONDS:
            return list(cached_cases[:limit])

    rows_by_image = _review_metadata_rows_by_image()

    seen: set[Path] = set()
    cases: list[ReviewCase] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for image_path in root.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if _is_generated_review_training_path(image_path):
                continue
            if _is_review_artifact_image(image_path):
                continue
            try:
                image_path = _safe_local_path(image_path)
                stat = image_path.stat()
            except (OSError, ValueError):
                continue
            if image_path in seen or stat.st_size < 512:
                continue
            seen.add(image_path)

            case_id = _case_id_for_path(image_path)
            saved = _load_label_for_case(case_id)
            row = rows_by_image.get(image_path, {})
            source_video, recording_dir = _recording_context(image_path)
            probability = None
            for key in ("detector_cat_probability", "cat_probability", "probability", "best_probability"):
                try:
                    if row.get(key) not in {None, ""}:
                        probability = float(row[key])
                        break
                except ValueError:
                    pass
            human_label = (
                saved.get("label") or row.get("label_cat_present") or row.get("label_candidate_is_cat") or None
            )
            review_status = saved.get("review_status") or row.get("review_status") or "unreviewed"
            label = str(saved.get("title") or image_path.name)
            cases.append(
                ReviewCase(
                    id=case_id,
                    image_path=image_path,
                    label=label,
                    source=_source_name(image_path),
                    mtime=stat.st_mtime,
                    source_video_path=source_video,
                    source_recording_dir=recording_dir,
                    detector_probability=probability,
                    candidate_bbox_xywh=_parse_bbox(
                        row.get("candidate_bbox_xywh") or row.get("best_bbox") or saved.get("candidate_bbox_xywh")
                    ),
                    review_status=str(review_status),
                    human_label=str(human_label) if human_label else None,
                    notes=str(saved.get("notes") or row.get("notes") or ""),
                )
            )

    cases.sort(key=lambda item: item.priority_tuple(), reverse=True)
    _DISCOVER_CASES_CACHE = (now, cache_key, tuple(cases))
    return cases[:limit]


def _case_to_payload(case: ReviewCase, include_size: bool = False) -> dict[str, Any]:
    image_token = _encode_path(case.image_path)
    video_token = _encode_path(case.source_video_path) if case.source_video_path else None
    payload: dict[str, Any] = {
        "id": case.id,
        "kind": LABEL_NAMESPACE + "_case_v1",
        "label": case.label,
        "source": case.source,
        "image_path": str(case.image_path),
        "image_url": f"/api/cat-projector-label-review/file/{image_token}",
        "source_video_path": str(case.source_video_path) if case.source_video_path else None,
        "source_video_url": f"/api/cat-projector-label-review/file/{video_token}" if video_token else None,
        "source_recording_dir": str(case.source_recording_dir) if case.source_recording_dir else None,
        "detector_probability": case.detector_probability,
        "candidate_bbox_xywh": case.candidate_bbox_xywh,
        "uncertainty_score": round(case.uncertainty_score(), 4),
        "review_status": case.review_status,
        "human_label": case.human_label,
        "notes": case.notes,
        "mtime": case.mtime,
    }
    if include_size:
        payload["source_size_px"] = _image_size(case.image_path)
    return payload


def _review_is_saved(value: Any) -> bool:
    return str(value or "").lower() in {"saved", "reviewed"}


def _frame_says_no_cat(frame: dict[str, Any]) -> bool:
    cat_present = frame.get("cat_present")
    if cat_present is True or str(cat_present).lower() in {"true", "yes", "cat"}:
        return False
    if cat_present is False or str(cat_present).lower() in {"false", "no", "not_cat"}:
        return True
    for key in ("human_label", "label_cat_present"):
        value = str(frame.get(key) or "").lower()
        if value in {"yes", "cat", "true"}:
            return False
        if value in {"no", "not_cat", "false"}:
            return True
    candidate_is_cat = frame.get("candidate_is_cat")
    if candidate_is_cat is True or str(candidate_is_cat).lower() in {"true", "yes", "cat"}:
        return False
    if candidate_is_cat is False or str(candidate_is_cat).lower() in {"false", "no", "not_cat"}:
        return True
    label_candidate = str(frame.get("label_candidate_is_cat") or "").lower()
    if label_candidate in {"yes", "cat", "true"}:
        return False
    return label_candidate in {"no", "not_cat", "false"}


def _saved_label_says_no_cat(image_path: Path | str | None) -> bool:
    if image_path is None:
        return False
    try:
        path = Path(image_path)
        saved = _load_label_for_case(_case_id_for_path(path))
    except (OSError, ValueError):
        return False
    return _review_is_saved(saved.get("review_status")) and _frame_says_no_cat(saved)


def _all_paths_saved_no_cat(paths: list[Path]) -> bool:
    return bool(paths) and all(_saved_label_says_no_cat(path) for path in paths)


def _frame_has_detector_metadata(frame: dict[str, Any]) -> bool:
    return any(
        frame.get(key) not in {None, ""}
        for key in (
            "detector_backend",
            "detector_model_id",
            "measurement_source",
            "best_top_height_cm",
            "review_priority_score",
            "tracker_status",
        )
    )


def _probability_value(frame: dict[str, Any]) -> float | None:
    try:
        value = frame.get("detector_probability")
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _suspicion_for_frame(frame: dict[str, Any]) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []
    reviewed = _review_is_saved(frame.get("review_status"))
    reviewed_no_cat = reviewed and _frame_says_no_cat(frame)
    if not reviewed:
        score += 100.0
        reasons.append("unreviewed")
    probability = _probability_value(frame)
    if probability is not None:
        uncertainty = max(0.0, 1.0 - min(1.0, abs(probability - 0.5) * 2.0))
        score += uncertainty * 50.0
        if uncertainty >= 0.5:
            reasons.append(f"uncertain p={probability:.2f}")
    explicit_uncertainty = frame.get("uncertainty_score")
    try:
        if explicit_uncertainty not in {None, ""}:
            score += max(0.0, float(explicit_uncertainty)) * 25.0
    except (TypeError, ValueError):
        pass
    if not reviewed and not frame.get("model_output_path") and not _frame_has_detector_metadata(frame):
        score += 10.0
        reasons.append("model output missing")
    if frame.get("review_decision") in {"false_positive", "missed_cat", "bad_geometry"} and not reviewed_no_cat:
        score += 20.0
        reasons.append("manual correction nearby")
    priority_score = _float_or_none(frame.get("review_priority_score"))
    if priority_score is not None:
        score += priority_score
        for reason in frame.get("review_priority_reasons") or []:
            if isinstance(reason, str) and reason not in reasons:
                reasons.append(reason)
    height = _float_or_none(frame.get("best_top_height_cm"))
    confidence = _float_or_none(frame.get("measurement_confidence"))
    if height is not None and height >= 120:
        score += min(35.0, (height - 100.0) * 0.5)
        if reviewed_no_cat:
            reasons.append(f"reviewed no-cat but still measured {height:.1f} cm")
        else:
            reasons.append(f"peak candidate {height:.1f} cm")
    if height is not None and confidence is not None and height >= 100 and confidence < 0.55:
        score += 20.0
        reasons.append("low confidence around apex")
    if frame.get("measurement_source") == "legacy_bbox_top":
        score += 15.0
        reasons.append("legacy bbox measurement")
    if frame.get("best_measurement_warning"):
        score += 15.0
        reasons.append(str(frame["best_measurement_warning"]))
    if frame.get("tracker_status") == "rejected":
        score += 25.0
        reasons.append(f"tracker rejected: {frame.get('tracker_reason') or 'physics gate'}")
    if reviewed_no_cat and not any(reason.startswith("reviewed no-cat") for reason in reasons):
        score = 0.0
        reasons = ["reviewed no-cat"]
    if not reasons:
        reasons.append("low priority")
    frame["suspicion_score"] = round(score, 4)
    frame["suspicion_reasons"] = reasons
    frame["suspicion_reason"] = reasons[0]
    frame["reviewed"] = _review_is_saved(frame.get("review_status"))
    return frame


def _timeline_for_video(
    video_id: str,
    *,
    requested_frame_label: str | None = None,
) -> tuple[ReviewVideo, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str | None, float]:
    video = next((item for item in _discover_videos(10000) if item.id == video_id), None)
    if video is None:
        raise ValueError(f"unknown review video: {video_id}")
    materialized_path, resolved_frame_label, frame_rate = _materialize_recording_review_frame(video, requested_frame_label)
    input_paths = _frame_paths_for_review_video(video)
    if materialized_path is not None and materialized_path not in input_paths:
        input_paths.append(materialized_path)
        input_paths = sorted(input_paths, key=_sort_frame_path)
    frames: list[dict[str, Any]] = []
    if input_paths:
        frames = _frame_payloads_for_review_video(video, input_paths, offset=0, limit=max(1, len(input_paths)))
    frames = [_suspicion_for_frame(frame) for frame in frames]
    suspect_queue = sorted(
        (
            {
                "frame_id": frame["id"],
                "frame_index": frame["frame_index"],
                "score": frame["suspicion_score"],
                "reason": frame["suspicion_reason"],
                "reasons": frame["suspicion_reasons"],
                "neighborhood": {
                    "start_frame": max(0, int(frame["frame_index"]) - 20),
                    "end_frame": min(max(0, len(frames) - 1), int(frame["frame_index"]) + 20),
                },
            }
            for frame in frames
            if not frame["reviewed"] and frame["suspicion_score"] > 0
        ),
        key=lambda item: (item["score"], -int(item["frame_index"])),
        reverse=True,
    )
    reviewed_suspect_queue = sorted(
        (
            {
                "frame_id": frame["id"],
                "frame_index": frame["frame_index"],
                "score": frame["suspicion_score"],
                "reason": frame["suspicion_reason"],
                "reasons": frame["suspicion_reasons"],
                "review_decision": frame.get("review_decision"),
                "human_label": frame.get("human_label"),
                "neighborhood": {
                    "start_frame": max(0, int(frame["frame_index"]) - 20),
                    "end_frame": min(max(0, len(frames) - 1), int(frame["frame_index"]) + 20),
                },
            }
            for frame in frames
            if frame["reviewed"] and frame["suspicion_score"] > 0
        ),
        key=lambda item: (item["score"], -int(item["frame_index"])),
        reverse=True,
    )
    return video, frames, suspect_queue, reviewed_suspect_queue, resolved_frame_label, frame_rate


def _image_paths_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for image_path in root.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            image_path = _safe_local_path(image_path)
            if image_path.stat().st_size >= 512:
                paths.append(image_path)
        except (OSError, ValueError):
            continue
    return sorted(paths, key=lambda path: (path.parent.name, path.name))


def _image_count_under(root: Path) -> int:
    if not root.exists():
        return 0
    count = 0
    for image_path in root.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            if _safe_local_path(image_path).stat().st_size >= 512:
                count += 1
        except (OSError, ValueError):
            continue
    return count


def _direct_image_count(root: Path) -> int:
    if not root.exists():
        return 0
    count = 0
    for image_path in root.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            if _safe_local_path(image_path).stat().st_size >= 512:
                count += 1
        except (OSError, ValueError):
            continue
    return count


def _frame_count_for_group(group_key: Path) -> int:
    for directory_name in PREFERRED_INPUT_FRAME_DIR_NAMES:
        count = _image_count_under(group_key / directory_name)
        if count:
            return count
    return sum(1 for image_path in _image_paths_under(group_key) if not _is_review_artifact_image(image_path))


def _output_frame_count_for_group(group_key: Path) -> int:
    counts = [_image_count_under(group_key / directory_name) for directory_name in PREFERRED_OUTPUT_FRAME_DIR_NAMES]
    return max(counts, default=0)


def _iter_frame_group_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    groups: set[Path] = set()
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        if _is_generated_review_training_path(directory):
            continue
        name = directory.name
        if _is_input_frame_dir_name(name) and _direct_image_count(directory):
            groups.add(directory.parent)
            continue
        if _is_output_frame_dir_name(name):
            continue
        if (directory / "labels.csv").exists():
            groups.add(directory)
    return sorted(groups, key=lambda path: path.stat().st_mtime, reverse=True)


def _candidate_output_path(input_path: Path) -> Path | None:
    candidates: list[Path] = []
    for parent in [input_path.parent, *input_path.parents]:
        if parent == parent.parent:
            break
        for directory_name in PREFERRED_OUTPUT_FRAME_DIR_NAMES:
            candidates.append(parent / directory_name / input_path.name)
            candidates.append(parent / directory_name / input_path.with_suffix(".jpg").name)
            candidates.append(parent / directory_name / input_path.with_suffix(".png").name)
        if _is_input_frame_dir_name(parent.name):
            for directory_name in PREFERRED_OUTPUT_FRAME_DIR_NAMES:
                candidates.append(parent.parent / directory_name / input_path.name)
                candidates.append(parent.parent / directory_name / input_path.with_suffix(".jpg").name)
                candidates.append(parent.parent / directory_name / input_path.with_suffix(".png").name)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return _safe_local_path(candidate)
        except ValueError:
            continue
    if _is_input_frame_dir_name(input_path.parent.name):
        input_peers = _image_paths_under(input_path.parent)
        try:
            input_index = input_peers.index(input_path)
        except ValueError:
            input_index = -1
        if input_index >= 0:
            for directory_name in PREFERRED_OUTPUT_FRAME_DIR_NAMES:
                output_dir = input_path.parent.parent / directory_name
                output_peers = _image_paths_under(output_dir)
                if input_index < len(output_peers):
                    return output_peers[input_index]
    return None


def _candidate_output_paths_for_inputs(input_paths: list[Path]) -> dict[Path, Path | None]:
    input_index_by_dir: dict[Path, dict[Path, int]] = {}
    output_peers_by_dir: dict[Path, list[Path]] = {}
    output_paths: dict[Path, Path | None] = {}
    for input_path in input_paths:
        exact_candidates: list[Path] = []
        for parent in [input_path.parent, *input_path.parents]:
            if parent == parent.parent:
                break
            for directory_name in PREFERRED_OUTPUT_FRAME_DIR_NAMES:
                exact_candidates.append(parent / directory_name / input_path.name)
                exact_candidates.append(parent / directory_name / input_path.with_suffix(".jpg").name)
                exact_candidates.append(parent / directory_name / input_path.with_suffix(".png").name)
            if _is_input_frame_dir_name(parent.name):
                for directory_name in PREFERRED_OUTPUT_FRAME_DIR_NAMES:
                    exact_candidates.append(parent.parent / directory_name / input_path.name)
                    exact_candidates.append(parent.parent / directory_name / input_path.with_suffix(".jpg").name)
                    exact_candidates.append(parent.parent / directory_name / input_path.with_suffix(".png").name)
        for candidate in exact_candidates:
            try:
                if candidate.is_file():
                    output_paths[input_path] = _safe_local_path(candidate)
                    break
            except ValueError:
                continue
        if input_path in output_paths:
            continue
        output_paths[input_path] = None
        if not _is_input_frame_dir_name(input_path.parent.name):
            continue
        index_by_path = input_index_by_dir.get(input_path.parent)
        if index_by_path is None:
            index_by_path = {path: index for index, path in enumerate(_image_paths_under(input_path.parent))}
            input_index_by_dir[input_path.parent] = index_by_path
        input_index = index_by_path.get(input_path, -1)
        if input_index < 0:
            continue
        for directory_name in PREFERRED_OUTPUT_FRAME_DIR_NAMES:
            output_dir = input_path.parent.parent / directory_name
            output_peers = output_peers_by_dir.get(output_dir)
            if output_peers is None:
                output_peers = _image_paths_under(output_dir)
                output_peers_by_dir[output_dir] = output_peers
            if input_index < len(output_peers):
                output_paths[input_path] = output_peers[input_index]
                break
    return output_paths


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_string_values(item))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_string_values(item))
        return values
    return [str(value)]


def _manifest_matches_video(
    manifest_values: set[str] | frozenset[str],
    *,
    video_id: str,
    recording_dir: Path | None,
    group_key: Path,
    source_video: Path | None,
) -> bool:
    if video_id and video_id in manifest_values:
        return True
    for candidate in (recording_dir, group_key, source_video):
        if candidate is not None and str(candidate) in manifest_values:
            return True
    return False


def _reprocessed_dir_name_matches_video(
    output_dir: Path,
    *,
    video_id: str,
    recording_dir: Path | None,
    group_key: Path,
) -> bool:
    haystack = output_dir.name
    if recording_dir and recording_dir.name[:15] in haystack:
        return True
    if group_key.name and group_key.name in haystack:
        return True
    if video_id:
        compact = re.sub(r"[^A-Za-z0-9]+", "", video_id)
        if compact and compact[:9] in re.sub(r"[^A-Za-z0-9]+", "", haystack):
            return True
    return False


def _output_artifacts_for_video(
    *,
    video_id: str,
    recording_dir: Path | None,
    group_key: Path,
    source_video: Path | None,
) -> tuple[Path, ...]:
    artifacts: list[Path] = []
    for output_dir, manifest_values, output_artifacts in _reprocessed_outputs():
        matches = _manifest_matches_video(
            manifest_values,
            video_id=video_id,
            recording_dir=recording_dir,
            group_key=group_key,
            source_video=source_video,
        )
        if not matches:
            matches = _reprocessed_dir_name_matches_video(
                output_dir,
                video_id=video_id,
                recording_dir=recording_dir,
                group_key=group_key,
            )
        if not matches:
            continue
        artifacts.extend(output_artifacts)
    seen: set[Path] = set()
    unique: list[Path] = []
    for artifact in artifacts:
        if artifact in seen:
            continue
        seen.add(artifact)
        unique.append(artifact)
    return tuple(unique)


def _artifact_to_payload(path: Path) -> dict[str, Any]:
    token = _encode_path(path)
    return {
        "path": str(path),
        "url": f"/api/cat-projector-label-review/file/{token}",
        "label": path.stem.replace("_", " "),
        "bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
    }


def _video_group_key(case: ReviewCase) -> Path:
    if case.source_recording_dir:
        return case.source_recording_dir
    for parent in case.image_path.parents:
        if _is_input_frame_dir_name(parent.name) or _is_output_frame_dir_name(parent.name):
            return parent.parent
    return case.image_path.parent


def _input_frame_paths_for_group(group_key: Path) -> list[Path]:
    for directory_name in PREFERRED_INPUT_FRAME_DIR_NAMES:
        frame_paths = [
            path for path in _image_paths_under(group_key / directory_name) if not _is_review_artifact_image(path)
        ]
        if frame_paths:
            return frame_paths
    return [image_path for image_path in _image_paths_under(group_key) if not _is_review_artifact_image(image_path)]


def _unique_case_image_paths(group: list[ReviewCase]) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for case in sorted(group, key=lambda item: (item.image_path.parent.as_posix(), item.image_path.name)):
        if case.image_path in seen:
            continue
        seen.add(case.image_path)
        paths.append(case.image_path)
    return paths


def _video_payload(video: ReviewVideo) -> dict[str, Any]:
    status = _load_status_for_video(video.id)
    video_token = _encode_path(video.source_video_path) if video.source_video_path else None
    output_artifacts = [_artifact_to_payload(path) for path in video.output_artifacts]
    return {
        "id": video.id,
        "kind": LABEL_NAMESPACE + "_video_v1",
        "label": video.label,
        "source": video.source,
        "source_recording_dir": str(video.source_recording_dir) if video.source_recording_dir else None,
        "source_video_path": str(video.source_video_path) if video.source_video_path else None,
        "source_video_url": f"/api/cat-projector-label-review/file/{video_token}" if video_token else None,
        "frame_count": video.frame_count,
        "output_frame_count": video.output_frame_count,
        "output_artifact_count": len(output_artifacts),
        "output_artifacts": output_artifacts,
        "review_status": status.get("review_status") or video.review_status,
        "notes": status.get("notes") or video.notes,
        "max_jump_height_cm": video.max_jump_height_cm,
        "max_jump_height_source": video.max_jump_height_source,
        "mtime": video.mtime,
    }


def _merge_review_video(existing: ReviewVideo | None, update: ReviewVideo) -> ReviewVideo:
    if existing is None:
        return update
    output_artifacts = tuple(dict.fromkeys([*existing.output_artifacts, *update.output_artifacts]))
    if existing.max_jump_height_cm is None:
        max_jump_height_cm = update.max_jump_height_cm
        max_jump_height_source = update.max_jump_height_source
    elif update.max_jump_height_cm is None or existing.max_jump_height_cm >= update.max_jump_height_cm:
        max_jump_height_cm = existing.max_jump_height_cm
        max_jump_height_source = existing.max_jump_height_source
    else:
        max_jump_height_cm = update.max_jump_height_cm
        max_jump_height_source = update.max_jump_height_source
    return ReviewVideo(
        id=existing.id,
        label=existing.label if existing.mtime >= update.mtime else update.label,
        source=existing.source if existing.mtime >= update.mtime else update.source,
        mtime=max(existing.mtime, update.mtime),
        source_recording_dir=existing.source_recording_dir or update.source_recording_dir,
        source_video_path=existing.source_video_path or update.source_video_path,
        frame_count=existing.frame_count + update.frame_count,
        output_frame_count=existing.output_frame_count + update.output_frame_count,
        output_artifacts=output_artifacts,
        review_status=existing.review_status if existing.mtime >= update.mtime else update.review_status,
        notes=existing.notes or update.notes,
        max_jump_height_cm=max_jump_height_cm,
        max_jump_height_source=max_jump_height_source,
    )


def _discover_videos(limit: int) -> list[ReviewVideo]:
    global _DISCOVER_VIDEOS_CACHE
    cache_key = _discovery_cache_key()
    now = time.monotonic()
    if _DISCOVER_VIDEOS_CACHE is not None:
        cached_at, cached_key, cached_videos = _DISCOVER_VIDEOS_CACHE
        if cached_key == cache_key and now - cached_at <= _DISCOVERY_CACHE_TTL_SECONDS:
            return list(cached_videos[:limit])

    videos: dict[str, ReviewVideo] = {}
    height_index = _build_jump_height_index()

    for root in SCAN_ROOTS:
        if not root.exists() or root.name == "recordings":
            continue
        for key in _iter_frame_group_dirs(root):
            try:
                key = _safe_local_path(key)
            except ValueError:
                continue
            input_paths = _input_frame_paths_for_group(key)
            sample_paths = input_paths[:20]
            if not sample_paths:
                continue
            contexts = [_recording_context(path) for path in sample_paths]
            source_video = next((source for source, _recording in contexts if source), None)
            recording_dir = next((recording for _source, recording in contexts if recording), None)
            frame_count = _frame_count_for_group(key)
            output_count = _output_frame_count_for_group(key)
            video_id = _video_id_for_path(recording_dir or key)
            output_artifacts = _output_artifacts_for_video(
                video_id=video_id,
                recording_dir=recording_dir,
                group_key=key,
                source_video=source_video,
            )
            status = _load_status_for_video(video_id)
            max_jump_height_cm, max_jump_height_source = _jump_height_for_video(
                height_index,
                video_id=video_id,
                recording_dir=recording_dir,
                group_key=key,
                source_video=source_video,
            )
            if _all_paths_saved_no_cat(input_paths):
                max_jump_height_cm = None
                max_jump_height_source = ""
            update = ReviewVideo(
                id=video_id,
                label=status.get("title") or (recording_dir.name if recording_dir else key.name),
                source=_source_name(recording_dir or key),
                mtime=key.stat().st_mtime,
                source_recording_dir=recording_dir,
                source_video_path=source_video,
                frame_count=frame_count,
                output_frame_count=output_count,
                output_artifacts=output_artifacts,
                review_status=status.get("review_status") or "unreviewed",
                notes=status.get("notes") or "",
                max_jump_height_cm=max_jump_height_cm,
                max_jump_height_source=max_jump_height_source,
            )
            videos[video_id] = _merge_review_video(videos.get(video_id), update)

    recordings_root = STATE_ROOT / "recordings"
    if recordings_root.exists():
        for recording_dir in recordings_root.iterdir():
            if not recording_dir.is_dir():
                continue
            try:
                recording_dir = _safe_local_path(recording_dir)
            except ValueError:
                continue
            video_id = _video_id_for_path(recording_dir)
            if video_id in videos:
                continue
            chunks = sorted(
                candidate
                for candidate in recording_dir.iterdir()
                if candidate.suffix.lower() in VIDEO_EXTENSIONS and ".part." not in candidate.name
            )
            frame_count = _frame_count_for_group(recording_dir)
            if not chunks and not frame_count:
                continue
            status = _load_status_for_video(video_id)
            max_jump_height_cm, max_jump_height_source = _jump_height_for_video(
                height_index,
                video_id=video_id,
                recording_dir=recording_dir,
                group_key=recording_dir,
                source_video=chunks[0] if chunks else None,
            )
            if max_jump_height_cm is not None and _all_paths_saved_no_cat(_recording_frame_paths(recording_dir)):
                max_jump_height_cm = None
                max_jump_height_source = ""
            output_artifacts = _output_artifacts_for_video(
                video_id=video_id,
                recording_dir=recording_dir,
                group_key=recording_dir,
                source_video=chunks[0] if chunks else None,
            )
            videos[video_id] = ReviewVideo(
                id=video_id,
                label=status.get("title") or recording_dir.name,
                source="recordings",
                mtime=recording_dir.stat().st_mtime,
                source_recording_dir=recording_dir,
                source_video_path=chunks[0] if chunks else None,
                frame_count=frame_count,
                output_frame_count=_output_frame_count_for_group(recording_dir),
                output_artifacts=output_artifacts,
                review_status=status.get("review_status") or "unreviewed",
                notes=status.get("notes") or "",
                max_jump_height_cm=max_jump_height_cm,
                max_jump_height_source=max_jump_height_source,
            )

    rows = sorted(videos.values(), key=lambda item: item.height_priority_tuple(), reverse=True)
    _DISCOVER_VIDEOS_CACHE = (now, cache_key, tuple(rows))
    return rows[:limit]


def _recording_frame_paths(recording_dir: Path) -> list[Path]:
    frame_dirs = [
        recording_dir / "frames",
        recording_dir / "review_frames",
        recording_dir / "label_frames",
        recording_dir / "raw",
        recording_dir / "input",
        recording_dir / "inputs",
        recording_dir / "original",
        recording_dir / "original_frames",
        recording_dir / "source2",
        recording_dir / "source",
    ]
    paths: list[Path] = []
    for frame_dir in frame_dirs:
        paths.extend(_image_paths_under(frame_dir))
    if not paths:
        for image_path in _image_paths_under(recording_dir):
            if image_path.parent.name not in OUTPUT_FRAME_DIR_NAMES:
                paths.append(image_path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen or _is_review_artifact_image(path):
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _sort_frame_path(path: Path) -> tuple[int, int, str]:
    match = re.match(r"(?:chunk|source)_(\d+)_(\d+)\.(?:jpe?g|png|webp)$", path.name, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)), path.as_posix())
    return (10**9, 10**9, path.as_posix())


def _frame_paths_for_review_video(video: ReviewVideo) -> list[Path]:
    paths: list[Path] = []
    if video.source_recording_dir:
        for root in SCAN_ROOTS:
            if not root.exists() or root.name == "recordings":
                continue
            for group_key in _iter_frame_group_dirs(root):
                input_paths = _input_frame_paths_for_group(group_key)
                if not input_paths:
                    continue
                sample_paths = input_paths[:20]
                if not any(_recording_context(path)[1] == video.source_recording_dir for path in sample_paths):
                    continue
                for path in input_paths:
                    if _recording_context(path)[1] == video.source_recording_dir:
                        paths.append(path)
        if not paths:
            paths = _recording_frame_paths(video.source_recording_dir)
    else:
        for root in SCAN_ROOTS:
            if not root.exists():
                continue
            for group_key in _iter_frame_group_dirs(root):
                if _video_id_for_path(group_key) == video.id:
                    paths.extend(_input_frame_paths_for_group(group_key))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if _is_review_artifact_image(path) or path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return sorted(unique, key=_sort_frame_path)


def _source_video_path_for_frame(video: ReviewVideo, input_path: Path) -> Path | None:
    if video.source_recording_dir:
        match = re.match(r"(?:chunk|source)_(\d+)_\d+\.(?:jpe?g|png|webp)$", input_path.name, re.IGNORECASE)
        if match:
            chunk_path = video.source_recording_dir / f"chunk_{int(match.group(1)):04d}.mp4"
            if chunk_path.exists():
                return chunk_path
    return video.source_video_path


def _recording_highlight_for_chunk(recording_dir: Path, chunk_index: int) -> dict[str, Any] | None:
    sessions_path = STATE_ROOT / "sessions.jsonl"
    if not sessions_path.exists():
        return None
    try:
        lines = sessions_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if "jump_highlight" not in line or "telegram" not in line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        recording_value = payload.get("recording_dir")
        if not recording_value:
            continue
        try:
            payload_recording_dir = _safe_local_path(Path(str(recording_value)))
        except ValueError:
            continue
        if payload_recording_dir != recording_dir:
            continue
        highlight = payload.get("jump_highlight")
        if not isinstance(highlight, dict):
            continue
        try:
            if int(highlight.get("chunk")) == chunk_index:
                return highlight
        except (TypeError, ValueError):
            continue
    return None


def _source_fps_review_frame_label(video: ReviewVideo, frame_label: str | None) -> str | None:
    if not video.source_recording_dir or not frame_label:
        return None
    match = re.match(r"(?:chunk|source)_(\d+)_(\d+)\.(?:jpe?g|png|webp)$", frame_label, re.IGNORECASE)
    if not match:
        return None
    chunk_index = int(match.group(1))
    highlight = _recording_highlight_for_chunk(video.source_recording_dir, chunk_index)
    if not highlight:
        return frame_label
    try:
        offset_seconds = float(highlight.get("offset_seconds") or 0.0)
    except (TypeError, ValueError):
        return frame_label
    requested_frame_index = int(match.group(2))
    legacy_frame_index = max(0, int(offset_seconds * LEGACY_TELEGRAM_REVIEW_FRAME_FPS))
    if requested_frame_index != legacy_frame_index:
        return frame_label
    chunk_path = video.source_recording_dir / f"chunk_{chunk_index:04d}.mp4"
    if not chunk_path.exists():
        return frame_label
    frame_index = _source_frame_index_for_offset(chunk_path, offset_seconds)
    return f"chunk_{chunk_index:04d}_{frame_index:05d}.jpg"


def _materialize_recording_review_frame(video: ReviewVideo, frame_label: str | None) -> tuple[Path | None, str | None, float]:
    if not video.source_recording_dir or not frame_label:
        return None, None, DEFAULT_REVIEW_FRAME_FPS
    frame_label = _source_fps_review_frame_label(video, frame_label) or frame_label
    match = re.match(r"(?:chunk|source)_(\d+)_(\d+)\.(?:jpe?g|png|webp)$", frame_label, re.IGNORECASE)
    if not match:
        return None, frame_label, DEFAULT_REVIEW_FRAME_FPS
    chunk_index = int(match.group(1))
    frame_index = int(match.group(2))
    chunk_path = video.source_recording_dir / f"chunk_{chunk_index:04d}.mp4"
    if not chunk_path.exists():
        return None, frame_label, DEFAULT_REVIEW_FRAME_FPS
    frame_rate = _chunk_frame_rate(chunk_path)
    output_path = video.source_recording_dir / "review_frames" / f"chunk_{chunk_index:04d}_{frame_index:05d}.jpg"
    chunk_glob = f"chunk_{chunk_index:04d}_*.jpg"
    if output_path.exists() and output_path.stat().st_size > 512 and len(list(output_path.parent.glob(chunk_glob))) > 1:
        return output_path.resolve(), frame_label, frame_rate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_pattern = output_path.parent / f"chunk_{chunk_index:04d}_%05d.jpg"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            "0",
            "-i",
            str(chunk_path),
            "-vsync",
            "0",
            "-start_number",
            "0",
            "-q:v",
            "2",
            str(output_pattern),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 512:
        return None, frame_label, frame_rate
    _clear_discovery_caches()
    return output_path.resolve(), frame_label, frame_rate


def _frame_payloads_for_review_video(
    video: ReviewVideo,
    input_paths: list[Path],
    *,
    offset: int = 0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    metadata_by_path = _review_metadata_rows_by_image()
    current_model_rows = _current_model_rows_by_image()
    original_model_rows = _original_model_rows_by_image()
    selected = input_paths[max(0, offset) : max(0, offset) + max(1, limit)]
    for input_path in selected:
        on_demand_row = _current_model_row_for_image(input_path)
        if on_demand_row:
            current_model_rows[input_path] = on_demand_row
    output_paths = _candidate_output_paths_for_inputs(selected) if video.output_frame_count else {}
    frames: list[dict[str, Any]] = []
    for frame_index, input_path in enumerate(selected, start=max(0, offset)):
        case_id = _case_id_for_path(input_path)
        saved = _load_label_for_case(case_id)
        row = metadata_by_path.get(input_path, {})
        current_model_row = current_model_rows.get(input_path, {})
        original_model_row = original_model_rows.get(input_path, {})
        display_row = _display_measurement_row(row, current_model_row, original_model_row)
        current_model_overlay = _model_overlay_from_row(
            current_model_row,
            role="current_model",
            label="current model",
        )
        original_model_overlay = _model_overlay_from_row(
            original_model_row,
            role="original_model",
            label="original capture",
        )
        probability = saved.get("detector_probability")
        for key in ("detector_cat_probability", "cat_probability", "probability", "best_probability"):
            if probability not in {None, ""}:
                break
            try:
                if display_row.get(key) not in {None, ""}:
                    probability = float(display_row[key])
                    break
            except ValueError:
                pass
        bbox = saved.get("candidate_bbox_xywh") or _parse_bbox(
            display_row.get("candidate_bbox_xywh") or display_row.get("best_bbox")
        )
        label_cat_present = saved.get("label_cat_present") or row.get("label_cat_present") or None
        label_candidate_is_cat = saved.get("label_candidate_is_cat") or row.get("label_candidate_is_cat") or None
        output_path = output_paths.get(input_path)
        source_video_path = _source_video_path_for_frame(video, input_path)
        input_token = _encode_path(input_path)
        output_token = _encode_path(output_path) if output_path else None
        frames.append(
            {
                "id": case_id,
                "kind": LABEL_NAMESPACE + "_video_frame_v1",
                "video_id": video.id,
                "frame_index": frame_index,
                "label": input_path.name,
                "image_path": str(input_path),
                "image_url": f"/api/cat-projector-label-review/file/{input_token}",
                "model_output_path": str(output_path) if output_path else None,
                "model_output_url": f"/api/cat-projector-label-review/file/{output_token}" if output_token else None,
                "source_video_path": str(source_video_path) if source_video_path else None,
                "source_recording_dir": str(video.source_recording_dir) if video.source_recording_dir else None,
                "candidate_bbox_xywh": bbox,
                "current_model_overlay": current_model_overlay,
                "original_model_overlay": original_model_overlay,
                "model_overlays": [
                    overlay for overlay in (current_model_overlay, original_model_overlay) if overlay is not None
                ],
                "detector_probability": probability,
                "detector_backend": display_row.get("detector_backend") or "",
                "detector_model_id": display_row.get("detector_model_id") or "",
                "model_created_at": display_row.get("model_created_at") or None,
                "model_path": display_row.get("model_path") or None,
                "measurement_source": display_row.get("measurement_source") or "",
                "measurement_confidence": _float_or_none(display_row.get("measurement_confidence")),
                "best_top_height_cm": _float_or_none(display_row.get("best_top_height_cm")),
                "best_top_wall_x_cm": _float_or_none(display_row.get("best_top_wall_x_cm")),
                "legacy_bbox_top_height_cm": _float_or_none(display_row.get("legacy_bbox_top_height_cm")),
                "measurement_point": display_row.get("best_measurement_point")
                if isinstance(display_row.get("best_measurement_point"), dict)
                else None,
                "best_measurement_warning": display_row.get("best_measurement_warning") or "",
                "tracker_status": display_row.get("tracker_status") or "",
                "tracker_reason": display_row.get("tracker_reason") or "",
                "tracker_confirmed": bool(display_row.get("tracker_confirmed"))
                if display_row.get("tracker_confirmed") not in {None, ""}
                else False,
                "tracker_height_cm": _float_or_none(display_row.get("tracker_height_cm")),
                "review_priority_score": _float_or_none(display_row.get("review_priority_score")),
                "review_priority_reasons": display_row.get("review_priority_reasons")
                if isinstance(display_row.get("review_priority_reasons"), list)
                else [],
                "review_status": saved.get("review_status") or row.get("review_status") or "unreviewed",
                "human_label": saved.get("label")
                or row.get("label_cat_present")
                or row.get("label_candidate_is_cat")
                or None,
                "label_cat_present": label_cat_present,
                "label_candidate_is_cat": label_candidate_is_cat,
                "review_decision": saved.get("review_decision"),
                "cat_present": saved.get("cat_present"),
                "candidate_is_cat": saved.get("candidate_is_cat"),
                "geometry_status": saved.get("geometry_status"),
                "notes": saved.get("notes") or row.get("notes") or "",
                # Do not open every image while building a video timeline. The
                # browser learns naturalWidth/naturalHeight when the selected
                # frame image loads; eager PIL probes made long recordings look
                # like the UI was hung.
                "source_size_px": saved.get("source_size_px"),
            }
        )
    return frames


def _frames_for_video(video_id: str, *, offset: int = 0, limit: int = 50) -> tuple[ReviewVideo, list[dict[str, Any]]]:
    video = next((item for item in _discover_videos(10000) if item.id == video_id), None)
    if video is None:
        raise ValueError(f"unknown review video: {video_id}")
    input_paths = _frame_paths_for_review_video(video)
    return video, _frame_payloads_for_review_video(video, input_paths, offset=offset, limit=limit)


def _save_video_status(video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    video = next((item for item in _discover_videos(10000) if item.id == video_id), None)
    if video is None:
        raise ValueError(f"unknown review video: {video_id}")
    row = {
        "kind": LABEL_NAMESPACE + "_video_status_v1",
        "video_id": video_id,
        "review_status": str(payload.get("review_status") or "relabeled_ok"),
        "notes": str(payload.get("notes") or ""),
        "source_recording_dir": str(video.source_recording_dir) if video.source_recording_dir else None,
        "source_video_path": str(video.source_video_path) if video.source_video_path else None,
        "updated_at": _utc_now(),
    }
    _write_json(_video_status_path(video_id), row)
    return row


def _normalise_point(point: Any) -> tuple[float, float] | None:
    if isinstance(point, dict):
        if "x" in point and "y" in point:
            return float(point["x"]), float(point["y"])
        if "point" in point:
            return _normalise_point(point["point"])
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def _is_local_sam_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    return (
        host.startswith("10.")
        or host.startswith("192.168.")
        or any(host.startswith(f"172.{octet}.") for octet in range(16, 32))
    )


def _segment_with_optional_sam(
    image_path: Path,
    positive_points: list[Any],
    negative_points: list[Any],
    existing_polygon: list[Any] | None = None,
    *,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    if SAM_ENDPOINT:
        if not _is_local_sam_endpoint(SAM_ENDPOINT):
            raise ValueError("CAT_PROJECTOR_SAM_ENDPOINT must point to localhost or a private LAN host")
        request = Request(
            SAM_ENDPOINT,
            data=json.dumps(
                {
                    "image_path": str(image_path),
                    "positive_points": positive_points,
                    "negative_points": negative_points,
                    "existing_polygon": existing_polygon or [],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            polygon = payload.get("polygon") or payload.get("contour")
            if isinstance(polygon, list) and len(polygon) >= 3:
                payload["kind"] = LABEL_NAMESPACE + "_mask_v1"
                payload["source"] = payload.get("source") or "local_sam_service"
                return payload
        except Exception as exc:
            if not allow_fallback:
                raise ValueError(f"SAM endpoint failed: {exc}") from exc
            fallback = _click_to_contour(image_path, positive_points, negative_points)
            fallback["sam_error"] = str(exc)
            fallback["source"] = "degraded_click_contour_after_sam_error"
            return fallback
    if not allow_fallback:
        raise ValueError("CAT_PROJECTOR_SAM_ENDPOINT is not configured; start the local SAM service")
    return _click_to_contour(image_path, positive_points, negative_points)


def _component_polygon(mask: np.ndarray, max_vertices: int = 48) -> list[dict[str, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return []
    x0, y0 = float(xs.mean()), float(ys.mean())
    boundary = []
    height, width = mask.shape
    for y, x in zip(ys, xs, strict=True):
        if x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1:
            boundary.append((x, y))
            continue
        if not mask[y - 1 : y + 2, x - 1 : x + 2].all():
            boundary.append((x, y))
    if not boundary:
        boundary = list(zip(xs, ys, strict=True))
    buckets: dict[int, tuple[float, float, float]] = {}
    for x, y in boundary:
        angle = math.atan2(y - y0, x - x0)
        bucket = int((angle + math.pi) / (2 * math.pi) * max_vertices)
        distance = (x - x0) ** 2 + (y - y0) ** 2
        if bucket not in buckets or distance > buckets[bucket][2]:
            buckets[bucket] = (float(x), float(y), float(distance))
    sorted_boundary = sorted(
        buckets.values(),
        key=lambda item: math.atan2(item[1] - y0, item[0] - x0),
    )
    points = [{"x": round(x, 2), "y": round(y, 2)} for x, y, _distance in sorted_boundary]
    return points


def _click_to_contour(image_path: Path, positive_points: list[Any], negative_points: list[Any]) -> dict[str, Any]:
    positives = [_normalise_point(point) for point in positive_points]
    positives = [point for point in positives if point is not None]
    if not positives:
        raise ValueError("positive_points is required")
    negatives = [_normalise_point(point) for point in negative_points]
    negatives = [point for point in negatives if point is not None]

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    height, width = arr.shape[:2]
    seed_x = int(max(0, min(width - 1, round(positives[-1][0]))))
    seed_y = int(max(0, min(height - 1, round(positives[-1][1]))))
    seed = arr[seed_y, seed_x]

    radius = max(12.0, min(width, height) * 0.08)
    yy, xx = np.mgrid[:height, :width]
    spatial = np.hypot(xx - seed_x, yy - seed_y)
    color_distance = np.linalg.norm(arr - seed, axis=2)
    threshold = max(26.0, float(np.percentile(color_distance[spatial <= radius], 68)) + 18.0)
    allowed = (color_distance <= threshold) & (spatial <= radius * 2.8)

    for negative in negatives:
        nx = int(max(0, min(width - 1, round(negative[0]))))
        ny = int(max(0, min(height - 1, round(negative[1]))))
        allowed[np.hypot(xx - nx, yy - ny) <= radius * 0.55] = False

    mask = np.zeros((height, width), dtype=bool)
    if not allowed[seed_y, seed_x]:
        allowed[seed_y, seed_x] = True
    queue: deque[tuple[int, int]] = deque([(seed_x, seed_y)])
    mask[seed_y, seed_x] = True
    max_area = int(width * height * 0.18)
    while queue and int(mask.sum()) < max_area:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if mask[ny, nx] or not allowed[ny, nx]:
                continue
            mask[ny, nx] = True
            queue.append((nx, ny))

    # Slightly close single-pixel holes without importing cv2/scipy.
    padded = np.pad(mask, 1, mode="constant")
    neighbours = sum(
        padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    )
    mask = mask | (neighbours >= 7)
    polygon = _component_polygon(mask)
    if len(polygon) < 3:
        raise ValueError("contour failed")
    ys, xs = np.nonzero(mask)
    bbox = {
        "x": round(float(xs.min()), 2),
        "y": round(float(ys.min()), 2),
        "width": round(float(xs.max() - xs.min() + 1), 2),
        "height": round(float(ys.max() - ys.min() + 1), 2),
    }
    return {
        "kind": LABEL_NAMESPACE + "_mask_v1",
        "source": "server_click_contour",
        "polygon": polygon,
        "bbox_xywh": bbox,
        "score": round(min(0.95, 0.35 + len(xs) / max(1.0, width * height * 0.04)), 4),
    }


def _save_label(payload: dict[str, Any]) -> dict[str, Any]:
    case_id = str(payload.get("case_id") or payload.get("id") or "")
    if not case_id:
        image_path = _path_from_payload(payload)
        case_id = _case_id_for_path(image_path)
    image_path = _path_from_payload(payload)
    review_decision = str(payload.get("review_decision") or "").strip()
    label = str(payload.get("label") or payload.get("human_label") or "unsure")
    if review_decision == "good":
        label = "cat" if label not in {"not_cat", "no"} else "not_cat"
    elif review_decision == "false_positive":
        label = "not_cat"
    elif review_decision in {"missed_cat", "bad_geometry"}:
        label = "cat"
    elif review_decision == "unsure":
        label = "unsure"
    review_status = str(payload.get("review_status") or "saved")
    masks = payload.get("masks") or payload.get("cat_annotations") or []
    if not isinstance(masks, list):
        masks = []

    mask_refs: list[dict[str, Any]] = []
    mask_payloads: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        if not isinstance(mask, dict):
            continue
        mask_id = str(mask.get("id") or f"mask.{index + 1}")
        raw_kind_label = str(mask.get("kind_label") or mask.get("kind") or label)
        if raw_kind_label.startswith(LABEL_NAMESPACE):
            raw_kind_label = label
        mask_payload = {
            "kind": LABEL_NAMESPACE + "_mask_v1",
            "case_id": case_id,
            "image_path": str(image_path),
            "mask_id": mask_id,
            "label": mask.get("label") or label,
            "kind_label": raw_kind_label,
            "polygon": mask.get("polygon") or [],
            "bbox_xywh": mask.get("bbox_xywh"),
            "positive_points": mask.get("positive_points") or [],
            "negative_points": mask.get("negative_points") or [],
            "source": mask.get("source") or "manual",
            "updated_at": _utc_now(),
        }
        path = _mask_path(case_id, mask_id)
        _write_json(path, mask_payload)
        mask_refs.append({"id": mask_id, "path": str(path), "label": mask_payload["label"]})
        mask_payloads.append(mask_payload)

    candidate_is_cat = payload.get("candidate_is_cat")
    cat_present = payload.get("cat_present")
    geometry_status = payload.get("geometry_status")
    if review_decision == "good":
        candidate_is_cat = label == "cat"
        cat_present = label == "cat"
        geometry_status = "ok" if label == "cat" else None
    elif review_decision == "false_positive":
        candidate_is_cat = False
        cat_present = bool(payload.get("cat_present")) if "cat_present" in payload else False
        geometry_status = None
    elif review_decision == "missed_cat":
        cat_present = True
        candidate_is_cat = False
        geometry_status = "corrected" if masks else "missing"
    elif review_decision == "bad_geometry":
        cat_present = True
        candidate_is_cat = True
        geometry_status = str(geometry_status or ("corrected" if masks else "bad"))
    elif review_decision == "unsure":
        cat_present = None if cat_present in {None, ""} else cat_present
        candidate_is_cat = None if candidate_is_cat in {None, ""} else candidate_is_cat
        geometry_status = geometry_status or None

    label_cat_present = payload.get("label_cat_present")
    if label_cat_present is None:
        if cat_present is True:
            label_cat_present = "yes"
        elif cat_present is False:
            label_cat_present = "no"
        else:
            label_cat_present = "yes" if label == "cat" else "no" if label == "not_cat" else ""
    label_candidate_is_cat = payload.get("label_candidate_is_cat")
    if label_candidate_is_cat is None:
        if candidate_is_cat is True:
            label_candidate_is_cat = "yes"
        elif candidate_is_cat is False:
            label_candidate_is_cat = "no"
        else:
            label_candidate_is_cat = "yes" if label == "cat" else "no" if label == "not_cat" else ""

    row = {
        "kind": LABEL_NAMESPACE + "_label_v1",
        "case_id": case_id,
        "video_id": payload.get("video_id"),
        "frame_index": payload.get("frame_index"),
        "image_path": str(image_path),
        "model_output_path": payload.get("model_output_path"),
        "source_video_path": payload.get("source_video_path"),
        "source_recording_dir": payload.get("source_recording_dir"),
        "label": label,
        "label_cat_present": label_cat_present,
        "label_candidate_is_cat": label_candidate_is_cat,
        "review_decision": review_decision or None,
        "cat_present": cat_present,
        "candidate_is_cat": candidate_is_cat,
        "geometry_status": geometry_status,
        "review_notes": payload.get("review_notes") or payload.get("notes") or "",
        "reviewed_at": _utc_now(),
        "reviewer_source": payload.get("reviewer_source") or "local_ui",
        "review_status": review_status,
        "candidate_bbox_xywh": payload.get("candidate_bbox_xywh"),
        "notes": payload.get("notes") or "",
        "detector_probability": payload.get("detector_probability"),
        "mask_refs": mask_refs,
        "masks": mask_payloads,
        "updated_at": _utc_now(),
    }
    _write_json(_label_path(case_id), row)
    return row


TRAINING_LABEL_FIELDNAMES = [
    "image_relpath",
    "label_cat_present",
    "label_cat_playing",
    "review_status",
    "candidate_bbox_xywh",
    "label_candidate_is_cat",
    "negative_reason",
    "bbox_xywh",
    "occlusion",
    "confidence",
    "notes",
    "source_recording_dir",
    "source_chunk",
    "source_offset_seconds",
    "ha_session_id",
    "frigate_event_id",
    "video_slug",
    "candidate_reason",
]


def _bbox_dict_to_csv(raw: Any) -> str:
    if isinstance(raw, dict):
        try:
            values = (raw["x"], raw["y"], raw["width"], raw["height"])
        except KeyError:
            return ""
        return ",".join(str(int(round(float(value)))) for value in values)
    if isinstance(raw, str):
        parsed = _parse_bbox(raw)
        if parsed is None:
            return ""
        return ",".join(str(int(round(value))) for value in parsed)
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            return ",".join(str(int(round(float(value)))) for value in raw)
        except (TypeError, ValueError):
            return ""
    return ""


def _mask_ref_path(label: dict[str, Any], ref: dict[str, Any]) -> Path | None:
    raw_path = str(ref.get("path") or "")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    case_id = str(label.get("case_id") or "")
    if case_id:
        return MASKS_ROOT / _safe_slug(case_id) / path.name
    return MASKS_ROOT / path


def _masks_for_training_label(label: dict[str, Any]) -> list[dict[str, Any]]:
    masks = [mask for mask in label.get("masks") or [] if isinstance(mask, dict)]
    if masks:
        return masks
    loaded: list[dict[str, Any]] = []
    for ref in label.get("mask_refs") or []:
        if not isinstance(ref, dict):
            continue
        path = _mask_ref_path(label, ref)
        if path is None:
            continue
        raw = _read_json(path)
        if raw:
            loaded.append(raw)
    return loaded


def _mask_bbox_for_label(label: dict[str, Any]) -> str:
    for mask in _masks_for_training_label(label):
        if not isinstance(mask, dict):
            continue
        bbox = _bbox_dict_to_csv(mask.get("bbox_xywh"))
        if bbox:
            return bbox
    return _bbox_dict_to_csv(label.get("candidate_bbox_xywh"))


def _corrected_mask_bbox_for_label(label: dict[str, Any]) -> str:
    for mask in _masks_for_training_label(label):
        if not isinstance(mask, dict):
            continue
        bbox = _bbox_dict_to_csv(mask.get("bbox_xywh"))
        if bbox:
            return bbox
    return ""


def _bbox_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    return max(0.0, right - left) * max(0.0, bottom - top)


def _old_candidate_negative_bbox_for_label(label: dict[str, Any], corrected_bbox: str) -> str:
    if str(label.get("review_decision") or "") not in {"bad_geometry", "missed_cat"}:
        return ""
    old_bbox = _bbox_dict_to_csv(label.get("candidate_bbox_xywh"))
    if not old_bbox or not corrected_bbox or old_bbox == corrected_bbox:
        return ""
    old_parsed = _parse_bbox(old_bbox)
    corrected_parsed = _parse_bbox(corrected_bbox)
    if old_parsed is None or corrected_parsed is None:
        return ""
    if _bbox_intersection_area(old_parsed, corrected_parsed) > 0:
        return ""
    return old_bbox


def _source_chunk_for_label(label: dict[str, Any], image_path: Path, recording_dir: Path | None) -> tuple[str, str]:
    if recording_dir is None:
        return "", ""
    chunk_match = re.match(r"chunk_(\d+)_(\d+)\.(?:jpe?g|png|webp)$", image_path.name, re.IGNORECASE)
    if not chunk_match:
        return "", ""
    chunk_index = int(chunk_match.group(1))
    frame_index = int(chunk_match.group(2))
    chunk_path = recording_dir / f"chunk_{chunk_index:04d}.mp4"
    offset = frame_index / _chunk_frame_rate(chunk_path if chunk_path.exists() else None)
    return (str(chunk_path) if chunk_path.exists() else "", f"{offset:.3f}")


def _review_labels_for_training(payload: dict[str, Any]) -> list[dict[str, Any]]:
    video_id = str(payload.get("video_id") or "")
    case_id = str(payload.get("case_id") or "")
    labels: list[dict[str, Any]] = []
    if not LABELS_ROOT.exists():
        return labels
    for path in sorted(LABELS_ROOT.glob("*.json")):
        label = _read_json(path)
        if video_id and str(label.get("video_id") or "") != video_id:
            continue
        if case_id and not video_id and str(label.get("case_id") or "") != case_id:
            continue
        if str(label.get("label") or "") not in {"cat", "not_cat"}:
            continue
        image_path_raw = str(label.get("image_path") or "")
        if not image_path_raw:
            continue
        try:
            image_path = _safe_local_path(Path(image_path_raw))
        except ValueError:
            continue
        if not image_path.exists():
            continue
        label["_label_file"] = str(path)
        label["_image_path"] = image_path
        labels.append(label)
    return labels


def _materialize_review_labels_as_training_package(payload: dict[str, Any]) -> dict[str, Any]:
    labels = _review_labels_for_training(payload)
    if not labels:
        raise ValueError("no saved cat/not-cat review labels found for retrain action")

    video_id = _safe_slug(str(payload.get("video_id") or "all-review-labels"))
    package_id = f"cat-projector-review-ui-{video_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    package_dir = TRAINING_DATASETS_ROOT / package_id
    frames_dir = package_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, str]] = []
    copied: list[dict[str, str]] = []
    for index, label in enumerate(labels):
        image_path = label["_image_path"]
        target = (
            frames_dir
            / f"{index:04d}_{_safe_slug(str(label.get('case_id') or image_path.stem))}{image_path.suffix.lower()}"
        )
        shutil.copy2(image_path, target)
        label_kind = str(label.get("label") or "")
        recording_raw = str(label.get("source_recording_dir") or "")
        recording_dir = Path(recording_raw).expanduser() if recording_raw else None
        if recording_dir is None or not recording_dir.exists():
            _source_video, inferred_recording = _recording_context(image_path)
            recording_dir = inferred_recording
        source_chunk, source_offset = _source_chunk_for_label(label, image_path, recording_dir)
        bbox = _mask_bbox_for_label(label)
        corrected_bbox = _corrected_mask_bbox_for_label(label)
        old_negative_bbox = _old_candidate_negative_bbox_for_label(label, corrected_bbox)
        row = {field: "" for field in TRAINING_LABEL_FIELDNAMES}
        row.update(
            {
                "image_relpath": target.relative_to(package_dir).as_posix(),
                "label_cat_present": "yes" if label_kind == "cat" else "no",
                "label_cat_playing": "yes" if label_kind == "cat" else "unsure",
                "review_status": "human_reviewed_from_cat_projector_label_ui",
                "candidate_bbox_xywh": bbox,
                "label_candidate_is_cat": "yes" if label_kind == "cat" else "no",
                "negative_reason": "" if label_kind == "cat" else "human_review_not_cat",
                "bbox_xywh": bbox if label_kind == "cat" else "",
                "occlusion": "unknown" if label_kind == "cat" else "",
                "confidence": "high",
                "notes": (
                    f"materialized from review label {label.get('_label_file')}; "
                    f"{label.get('notes') or ''}"
                ).strip(),
                "source_recording_dir": str(recording_dir) if recording_dir else "",
                "source_chunk": source_chunk,
                "source_offset_seconds": source_offset,
                "video_slug": Path(str(label.get("source_video_path") or "")).name,
                "candidate_reason": "human_review_ui_mask" if label_kind == "cat" else "human_review_ui_not_cat",
            }
        )
        rows.append(row)
        if label_kind == "cat" and old_negative_bbox:
            old_candidate_row = dict(row)
            old_candidate_row.update(
                {
                    "candidate_bbox_xywh": old_negative_bbox,
                    "label_candidate_is_cat": "no",
                    "negative_reason": "old_candidate_no_overlap_corrected_mask",
                    "candidate_reason": "human_review_old_candidate_hard_negative",
                    "notes": f"{row['notes']}; old model candidate did not intersect corrected mask".strip(),
                }
            )
            rows.append(old_candidate_row)
        copied.append(
            {
                "label_file": str(label.get("_label_file") or ""),
                "source_image": str(image_path),
                "copied_image": str(target),
                "label": label_kind,
                "bbox_xywh": bbox,
                "old_candidate_negative_bbox_xywh": old_negative_bbox,
            }
        )

    labels_csv = package_dir / "labels.csv"
    with labels_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    positive_count = sum(1 for row in rows if row["label_candidate_is_cat"] == "yes")
    negative_count = sum(1 for row in rows if row["label_candidate_is_cat"] == "no")
    manifest = {
        "kind": LABEL_NAMESPACE + "_training_package_v1",
        "created_at": _utc_now(),
        "package_id": package_id,
        "package_dir": str(package_dir),
        "labels_csv": str(labels_csv),
        "source_action_payload": {key: value for key, value in payload.items() if key not in {"training_package"}},
        "source_label_count": len(labels),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "copied": copied,
    }
    _write_json(package_dir / "manifest.json", manifest)
    return manifest


def _path_from_payload(payload: dict[str, Any]) -> Path:
    token = str(payload.get("image_token") or "")
    if token:
        return _decode_path(token)
    image_url = str(payload.get("image_url") or payload.get("image_src") or "")
    marker = "/api/cat-projector-label-review/file/"
    if marker in image_url:
        return _decode_path(image_url.rsplit(marker, 1)[-1].split("?", 1)[0])
    raw = str(payload.get("image_path") or payload.get("image") or "")
    if not raw:
        raise ValueError("image_path is required")
    return _safe_local_path(Path(raw))


def _queue_action(action: str, payload: dict[str, Any], *, status: str = "queued") -> dict[str, Any]:
    action_id = f"{int(time.time())}-{_safe_slug(action)}-{hashlib.sha1(os.urandom(12)).hexdigest()[:8]}"
    log_message = (
        "recorded by review UI; live jobs are disabled, so nothing will pick this up automatically"
        if status == "recorded"
        else "queued by review UI; operator must run the matching offline command"
    )
    row = {
        "kind": LABEL_NAMESPACE + "_action_v1",
        "id": action_id,
        "action": action,
        "status": status,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "payload": payload,
        "log": [log_message],
    }
    _write_json(QUEUE_ROOT / f"{action_id}.json", row)
    return row


def _list_actions(limit: int = 50) -> list[dict[str, Any]]:
    if not QUEUE_ROOT.exists():
        return []
    paths = sorted(QUEUE_ROOT.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [_read_json(path) for path in paths[:limit]]


def _update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    row = _read_json(_job_path(job_id))
    row.update(updates)
    row["updated_at"] = _utc_now()
    _write_json(_job_path(job_id), row)
    return row


def _live_job_command() -> list[str]:
    if LIVE_JOB_COMMAND is not None:
        return list(LIVE_JOB_COMMAND)
    jobs = os.environ.get("CAT_PROJECTOR_LIVE_JOB_JOBS", "16").strip() or "16"
    return [sys.executable, str(REPO_ROOT / "scripts" / "cat_projector_active_learning.py"), "--jobs", jobs]


def _summarize_active_learning_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def _run_live_job(job_id: str) -> None:
    global _ACTIVE_JOB_ID
    try:
        path = _job_path(job_id)
        if not path.exists():
            return
        row = _read_json(path)
        log = list(row.get("log") or [])
        command = _live_job_command()
        requested_action = row.get("action") or "rescore_recording"
        log.append(f"running local active-learning iteration for requested action: {requested_action}")
        _update_job(job_id, status="running", log=log, command=command)
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        log = list(_read_json(path).get("log") or [])
        if completed.returncode:
            tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-12:])
            log.append(f"failed with exit code {completed.returncode}")
            if tail:
                log.append(tail)
            _update_job(job_id, status="failed", log=log)
            return
        result = _summarize_active_learning_stdout(completed.stdout)
        frame_count = result.get("frame_count")
        if frame_count is not None:
            log.append(f"active-learning finished; rescored {frame_count} frames")
        else:
            log.append("active-learning finished")
        _update_job(job_id, status="done", log=log, result=result)
    finally:
        with _JOB_LOCK:
            if job_id == _ACTIVE_JOB_ID:
                _ACTIVE_JOB_ID = None


def _start_or_queue_job(payload: dict[str, Any]) -> dict[str, Any]:
    global _ACTIVE_JOB_ID
    action = str(payload.get("action") or "rescore_recording")
    if action not in {"retrain_model", "rescore_recording", "rerender_overlay"}:
        raise ValueError("job action must be retrain_model, rescore_recording, or rerender_overlay")
    payload = dict(payload)
    payload["job_scope"] = "full_active_learning_iteration"
    payload["job_scope_note"] = (
        "The local job runs retrain/rescore/re-measure together so jump heights and review priority stay consistent."
    )
    if action == "retrain_model" and "training_package" not in payload:
        payload["training_package"] = _materialize_review_labels_as_training_package(payload)
    if not ALLOW_LIVE_JOBS:
        action_payload = dict(payload)
        action_payload["job_mode"] = "recorded_without_live_jobs"
        return _queue_action(
            "rescore_recording" if action == "rerender_overlay" else action, action_payload, status="recorded"
        )

    with _JOB_LOCK:
        if _ACTIVE_JOB_ID:
            raise ValueError(f"live job already running: {_ACTIVE_JOB_ID}")
        job_id = f"{int(time.time())}-{_safe_slug(action)}-{hashlib.sha1(os.urandom(12)).hexdigest()[:8]}"
        _ACTIVE_JOB_ID = job_id
    row = {
        "kind": LABEL_NAMESPACE + "_job_v1",
        "id": job_id,
        "action": action,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "payload": payload,
        "log": [
            "started by review UI",
        ],
    }
    _write_json(_job_path(job_id), row)
    thread = threading.Thread(target=_run_live_job, args=(job_id,), daemon=True)
    thread.start()
    return row


def _list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    if not JOBS_ROOT.exists():
        return []
    _mark_stale_running_jobs()
    paths = sorted(JOBS_ROOT.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [_read_json(path) for path in paths[:limit]]


def _mark_stale_running_jobs(*, stale_after_seconds: float = 300.0) -> None:
    if not JOBS_ROOT.exists():
        return
    with _JOB_LOCK:
        active_job_id = _ACTIVE_JOB_ID
    now = datetime.now(UTC)
    for path in JOBS_ROOT.glob("*.json"):
        row = _read_json(path)
        if row.get("status") != "running" or row.get("id") == active_job_id:
            continue
        raw_updated = str(row.get("updated_at") or row.get("created_at") or "")
        try:
            updated_at = datetime.fromisoformat(raw_updated)
        except ValueError:
            updated_at = now
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        if (now - updated_at).total_seconds() < stale_after_seconds:
            continue
        log = list(row.get("log") or [])
        if not any("marked failed after server restart" in str(item) for item in log):
            log.append("marked failed after server restart; no active local worker owns this job")
        row["status"] = "failed"
        row["log"] = log
        row["updated_at"] = _utc_now()
        _write_json(path, row)


class CatProjectorLabelReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write(f"[cat-projector-label-review] {format % args}\n")

    def _read_body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message, "status": int(status)}, status=status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/cat-projector-label-review/cases":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["300"])[0])
            cases = [_case_to_payload(case, include_size=True) for case in _discover_cases(limit)]
            self._send_json({"kind": LABEL_NAMESPACE + "_queue_v1", "cases": cases})
            return
        if parsed.path == "/api/cat-projector-label-review/videos":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["100"])[0])
            videos = [_video_payload(video) for video in _discover_videos(limit)]
            self._send_json({"kind": LABEL_NAMESPACE + "_videos_v1", "videos": videos})
            return
        if parsed.path.startswith("/api/cat-projector-label-review/videos/"):
            parts = parsed.path.split("/")
            if len(parts) >= 6 and parts[5] == "frames":
                query = parse_qs(parsed.query)
                offset = int(query.get("offset", ["0"])[0])
                limit = int(query.get("limit", ["80"])[0])
                video, frames = _frames_for_video(unquote(parts[4]), offset=offset, limit=limit)
                self._send_json(
                    {
                        "kind": LABEL_NAMESPACE + "_video_frames_v1",
                        "video": _video_payload(video),
                        "offset": offset,
                        "limit": limit,
                        "frames": frames,
                    }
                )
                return
            if len(parts) >= 6 and parts[5] == "timeline":
                query = parse_qs(parsed.query)
                requested_frame_label = query.get("review_frame_label", query.get("frame_label", [""]))[0]
                video, frames, suspect_queue, reviewed_suspect_queue, resolved_frame_label, frame_rate = _timeline_for_video(
                    unquote(parts[4]),
                    requested_frame_label=requested_frame_label,
                )
                self._send_json(
                    {
                        "kind": LABEL_NAMESPACE + "_video_timeline_v1",
                        "video": _video_payload(video),
                        "frames": frames,
                        "suspect_queue": suspect_queue,
                        "reviewed_suspect_queue": reviewed_suspect_queue,
                        "resolved_review_frame_label": resolved_frame_label,
                        "playback": {"default_fps": frame_rate, "neighborhood_frames": 20},
                    }
                )
                return
            if len(parts) >= 6 and parts[5] == "status":
                video_id = unquote(parts[4])
                self._send_json(
                    _load_status_for_video(video_id) or {"video_id": video_id, "review_status": "unreviewed"}
                )
                return
        if parsed.path.startswith("/api/cat-projector-label-review/jobs/"):
            job_id = unquote(parsed.path.rsplit("/", 1)[-1])
            row = _read_json(_job_path(job_id))
            if not row:
                self._send_error_json(HTTPStatus.NOT_FOUND, "unknown job")
                return
            self._send_json(row)
            return
        if parsed.path.startswith("/api/cat-projector-label-review/file/"):
            self._send_api_file(parsed.path)
            return
        if parsed.path == "/api/cat-projector-label-review/label":
            query = parse_qs(parsed.query)
            case_id = query.get("case_id", [""])[0]
            if not case_id:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "case_id is required")
                return
            self._send_json(_load_label_for_case(case_id) or {"case_id": case_id, "review_status": "unreviewed"})
            return
        if parsed.path == "/api/cat-projector-label-review/actions":
            self._send_json({"kind": LABEL_NAMESPACE + "_actions_v1", "actions": _list_actions()})
            return
        if parsed.path == "/api/cat-projector-label-review/jobs":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["50"])[0])
            self._send_json(
                {
                    "kind": LABEL_NAMESPACE + "_jobs_v1",
                    "allow_live_jobs": ALLOW_LIVE_JOBS,
                    "active_job_id": _ACTIVE_JOB_ID,
                    "jobs": _list_jobs(limit),
                    "actions": _list_actions(limit),
                }
            )
            return
        if self._send_static_if_exists(WEB_ROOT, parsed.path):
            return
        if self._send_dataset_static_if_exists(parsed.path):
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/cat-projector-label-review/file/"):
            self._send_api_file(parsed.path, include_body=False)
            return
        if self._send_static_if_exists(WEB_ROOT, parsed.path, include_body=False):
            return
        if self._send_dataset_static_if_exists(parsed.path, include_body=False):
            return
        super().do_HEAD()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_body_json()
            if parsed.path == "/api/cat-projector-label-review/segment":
                image_path = _path_from_payload(payload)
                result = _segment_with_optional_sam(
                    image_path,
                    list(payload.get("positive_points") or []),
                    list(payload.get("negative_points") or []),
                    list(payload.get("existing_polygon") or []),
                    allow_fallback=bool(payload.get("allow_fallback")),
                )
                self._send_json(result)
                return
            if parsed.path == "/api/cat-projector-label-review/labels":
                self._send_json(_save_label(payload))
                return
            if parsed.path.startswith("/api/cat-projector-label-review/frames/"):
                parts = parsed.path.split("/")
                if len(parts) >= 6 and parts[5] == "review":
                    payload = dict(payload)
                    payload["case_id"] = payload.get("case_id") or unquote(parts[4])
                    self._send_json(_save_label(payload))
                    return
            if parsed.path.startswith("/api/cat-projector-label-review/videos/"):
                parts = parsed.path.split("/")
                if len(parts) >= 6 and parts[5] == "status":
                    self._send_json(_save_video_status(unquote(parts[4]), payload))
                    return
            if parsed.path == "/api/cat-projector-label-review/actions":
                action = str(payload.get("action") or "")
                if action not in {"retrain_model", "rescore_recording"}:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "action must be retrain_model or rescore_recording")
                    return
                if action == "retrain_model":
                    payload = dict(payload)
                    payload["training_package"] = _materialize_review_labels_as_training_package(payload)
                self._send_json(_queue_action(action, payload))
                return
            if parsed.path == "/api/cat-projector-label-review/jobs":
                self._send_json(_start_or_queue_job(payload))
                return
        except Exception as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def _send_api_file(self, request_path: str, *, include_body: bool = True) -> None:
        token = unquote(request_path.rsplit("/", 1)[-1])
        try:
            self._send_file(_decode_path(token), include_body=include_body)
        except (OSError, ValueError) as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))

    def _send_file(self, path: Path, *, include_body: bool = True) -> None:
        path = _safe_local_path(path)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        stat = path.stat()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Cache-Control", "private, max-age=60")
        self.end_headers()
        if not include_body:
            return
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _send_dataset_static_if_exists(self, request_path: str, *, include_body: bool = True) -> bool:
        return self._send_static_if_exists(DATASET_ROOT, request_path, include_body=include_body)

    def _send_static_if_exists(self, root: Path, request_path: str, *, include_body: bool = True) -> bool:
        relative = Path(unquote(request_path).lstrip("/"))
        if any(part in {"", ".", ".."} for part in relative.parts):
            return False
        path = root / relative
        if not path.is_file():
            return False
        self._send_file(path, include_body=include_body)
        return True


def build_fake_corpus(root: Path) -> Path:
    corpus = root / "cat-tv-learning"
    frames = corpus / "datasets" / "fake-review" / "frames"
    output_frames = corpus / "datasets" / "fake-review" / "model-output"
    frames.mkdir(parents=True, exist_ok=True)
    output_frames.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    examples = [
        ("borderline.jpg", "0.52", "unclear", "180,90,90,160"),
        ("cat.jpg", "0.93", "yes", "160,70,110,210"),
        ("not-cat.jpg", "0.08", "no", "220,120,70,80"),
    ]
    for name, probability, frame_label, bbox in examples:
        image = Image.new("RGB", (420, 260), (220, 222, 224))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 400, 230), outline=(180, 180, 180), width=2)
        if frame_label == "yes":
            draw.ellipse((160, 70, 270, 230), fill=(28, 28, 28))
            draw.polygon((220, 80, 245, 30, 260, 88), fill=(28, 28, 28))
        elif frame_label == "no":
            draw.ellipse((220, 120, 290, 190), fill=(120, 120, 120))
            draw.line((230, 120, 290, 180), fill=(80, 80, 80), width=6)
        else:
            draw.ellipse((180, 90, 270, 220), fill=(70, 70, 70))
            draw.rectangle((235, 40, 270, 120), fill=(75, 75, 75))
        image.save(frames / name, quality=92)
        output = Image.new("RGB", (420, 260), (205, 207, 210))
        output_draw = ImageDraw.Draw(output)
        output_draw.rectangle((20, 20, 400, 230), outline=(120, 160, 220), width=2)
        output_draw.text((28, 30), f"previous model: {frame_label}", fill=(20, 20, 20))
        bx, by, bw, bh = (float(value) for value in bbox.split(","))
        output_draw.rectangle((bx, by, bx + bw, by + bh), outline=(255, 210, 40), width=4)
        output.save(output_frames / name, quality=92)
        rows.append(
            {
                "image_relpath": f"frames/{name}",
                "label_cat_present": frame_label,
                "review_status": "unreviewed",
                "candidate_bbox_xywh": bbox,
                "label_candidate_is_cat": "",
                "confidence": probability,
                "detector_cat_probability": probability,
                "notes": f"fake {frame_label}",
            }
        )
    with (corpus / "datasets" / "fake-review" / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return corpus


def run_server(host: str, port: int) -> None:
    for directory in (REVIEW_ROOT, LABELS_ROOT, MASKS_ROOT, QUEUE_ROOT, JOBS_ROOT, VIDEO_STATUS_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), CatProjectorLabelReviewHandler)
    url = f"http://{host}:{port}/calibration-tools/projector-wall-calibrator.html"
    print(f"cat_projector_label_review listening on {url}", flush=True)
    server.serve_forever()


def run_fake_smoke(tmp_root: Path) -> int:
    original_dataset = globals()["DATASET_ROOT"]
    original_state = globals()["STATE_ROOT"]
    original_review = globals()["REVIEW_ROOT"]
    original_labels = globals()["LABELS_ROOT"]
    original_masks = globals()["MASKS_ROOT"]
    original_queue = globals()["QUEUE_ROOT"]
    original_jobs = globals()["JOBS_ROOT"]
    original_video_status = globals()["VIDEO_STATUS_ROOT"]
    original_training_datasets = globals()["TRAINING_DATASETS_ROOT"]
    original_scan_roots = globals()["SCAN_ROOTS"]
    original_allowed = globals()["ALLOWED_ROOTS"]
    try:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        fake_dataset = build_fake_corpus(tmp_root)
        fake_state = tmp_root / "state"
        fake_review = fake_state / "label-review"
        globals()["DATASET_ROOT"] = fake_dataset
        globals()["STATE_ROOT"] = fake_state
        globals()["REVIEW_ROOT"] = fake_review
        globals()["LABELS_ROOT"] = fake_review / "labels"
        globals()["MASKS_ROOT"] = fake_review / "masks"
        globals()["QUEUE_ROOT"] = fake_review / "actions"
        globals()["JOBS_ROOT"] = fake_review / "jobs"
        globals()["VIDEO_STATUS_ROOT"] = fake_review / "videos"
        globals()["TRAINING_DATASETS_ROOT"] = fake_state / "datasets"
        globals()["SCAN_ROOTS"] = (fake_dataset / "datasets",)
        globals()["ALLOWED_ROOTS"] = (fake_dataset, fake_state, fake_review)
        for directory in (LABELS_ROOT, MASKS_ROOT, QUEUE_ROOT, JOBS_ROOT, VIDEO_STATUS_ROOT):
            directory.mkdir(parents=True, exist_ok=True)

        server = ThreadingHTTPServer(("127.0.0.1", 0), CatProjectorLabelReviewHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(f"{base_url}/api/cat-projector-label-review/cases?limit=10", timeout=10) as response:
                cases = json.loads(response.read().decode("utf-8"))["cases"]
            with urlopen(f"{base_url}/api/cat-projector-label-review/videos?limit=10", timeout=10) as response:
                videos = json.loads(response.read().decode("utf-8"))["videos"]
            with urlopen(
                f"{base_url}/api/cat-projector-label-review/videos/{videos[0]['id']}/frames?limit=10",
                timeout=10,
            ) as response:
                video_frames = json.loads(response.read().decode("utf-8"))["frames"]
            with urlopen(
                f"{base_url}/api/cat-projector-label-review/videos/{videos[0]['id']}/timeline",
                timeout=10,
            ) as response:
                timeline = json.loads(response.read().decode("utf-8"))
            with urlopen(f"{base_url}/datasets/fake-review/frames/borderline.jpg", timeout=10) as response:
                static_asset_status = response.status
        finally:
            server.shutdown()
            thread.join(timeout=5)

        if len(cases) != 3:
            raise AssertionError(f"expected 3 fake cases, got {len(cases)}")
        if static_asset_status != HTTPStatus.OK:
            raise AssertionError(f"dataset static fallback returned {static_asset_status}")
        if "borderline" not in cases[0]["image_path"]:
            raise AssertionError(f"borderline case was not first: {cases[0]['image_path']}")
        if not videos or videos[0]["frame_count"] != 3:
            raise AssertionError(f"expected fake review video with 3 frames, got {videos}")
        if not video_frames or not video_frames[0]["model_output_url"]:
            raise AssertionError("video frames did not expose previous-model output")
        if (
            not timeline["suspect_queue"]
            or "borderline" not in timeline["frames"][timeline["suspect_queue"][0]["frame_index"]]["image_path"]
        ):
            raise AssertionError("timeline did not put the suspicious borderline frame first")

        server = ThreadingHTTPServer(("127.0.0.1", 0), CatProjectorLabelReviewHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"

        def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
            request = Request(
                f"{base_url}{path}",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=10) as response:
                    return json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                if hasattr(exc, "read"):
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise AssertionError(f"POST {path} failed: {detail}") from exc
                raise

        cat_case = next(case for case in cases if case["human_label"] == "yes")
        not_cat_case = next(case for case in cases if case["human_label"] == "no")
        try:
            contour = post(
                "/api/cat-projector-label-review/segment",
                {
                    "image_path": cat_case["image_path"],
                    "positive_points": [{"x": 210, "y": 130}],
                    "negative_points": [],
                    "allow_fallback": True,
                },
            )
            saved_missed = post(
                f"/api/cat-projector-label-review/frames/{cat_case['id']}/review",
                {
                    "case_id": cat_case["id"],
                    "video_id": videos[0]["id"],
                    "frame_index": 1,
                    "image_path": cat_case["image_path"],
                    "model_output_path": video_frames[1]["model_output_path"],
                    "review_decision": "missed_cat",
                    "review_status": "saved",
                    "masks": [{"id": "sher", "label": "Sher", "kind": "cat", **contour}],
                    "notes": "fake smoke missed cat",
                },
            )
            saved_false_positive = post(
                f"/api/cat-projector-label-review/frames/{not_cat_case['id']}/review",
                {
                    "case_id": not_cat_case["id"],
                    "video_id": videos[0]["id"],
                    "frame_index": 2,
                    "image_path": not_cat_case["image_path"],
                    "review_decision": "false_positive",
                    "review_status": "saved",
                    "masks": [],
                    "notes": "fake smoke false positive",
                },
            )
            retrain = post(
                "/api/cat-projector-label-review/actions",
                {"action": "retrain_model", "reason": "fake smoke"},
            )
            rescore = post(
                "/api/cat-projector-label-review/actions",
                {"action": "rescore_recording", "video_id": videos[0]["id"], "recording_dir": "/tmp/fake-recording"},
            )
            job = post(
                "/api/cat-projector-label-review/jobs",
                {"action": "rescore_recording", "video_id": videos[0]["id"], "reason": "fake smoke"},
            )
            video_status = post(
                f"/api/cat-projector-label-review/videos/{videos[0]['id']}/status",
                {"review_status": "relabeled_ok", "notes": "fake video ok"},
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
        if saved_missed["mask_refs"] == [] or saved_missed["review_decision"] != "missed_cat":
            raise AssertionError("missed-cat label did not write review mask")
        if (
            saved_false_positive["label_candidate_is_cat"] != "no"
            or saved_false_positive["review_decision"] != "false_positive"
        ):
            raise AssertionError("false-positive label did not save candidate no")
        if retrain["status"] != "queued" or rescore["status"] != "queued":
            raise AssertionError("actions were not queued")
        if job["status"] != "recorded":
            raise AssertionError("job facade did not record disabled live job honestly")
        if video_status["review_status"] != "relabeled_ok":
            raise AssertionError("video status was not saved")
        print(
            json.dumps(
                {
                    "cases": cases,
                    "videos": videos,
                    "video_frames": video_frames,
                    "timeline": timeline,
                    "missed_cat_label": saved_missed,
                    "false_positive_label": saved_false_positive,
                    "video_status": video_status,
                    "actions": [retrain, rescore, job],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        globals()["DATASET_ROOT"] = original_dataset
        globals()["STATE_ROOT"] = original_state
        globals()["REVIEW_ROOT"] = original_review
        globals()["LABELS_ROOT"] = original_labels
        globals()["MASKS_ROOT"] = original_masks
        globals()["QUEUE_ROOT"] = original_queue
        globals()["JOBS_ROOT"] = original_jobs
        globals()["VIDEO_STATUS_ROOT"] = original_video_status
        globals()["TRAINING_DATASETS_ROOT"] = original_training_datasets
        globals()["SCAN_ROOTS"] = original_scan_roots
        globals()["ALLOWED_ROOTS"] = original_allowed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument(
        "--allow-live-jobs", action="store_true", help="Allow one local active-learning rescore/retrain job at a time."
    )
    parser.add_argument("--fake-smoke", action="store_true", help="Run the repo-local fake-corpus smoke test and exit.")
    parser.add_argument("--tmp-root", type=Path, default=Path("tmp/cat_projector_label_review_smoke"))
    args = parser.parse_args()
    globals()["ALLOW_LIVE_JOBS"] = bool(args.allow_live_jobs)
    if args.fake_smoke:
        return run_fake_smoke(args.tmp_root)
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
