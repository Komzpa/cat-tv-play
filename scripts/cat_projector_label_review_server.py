#!/usr/bin/env python3
"""Local Cat Projector label-review backend.

This server owns the durable review state for the Cat Projector calibrator UI:
it lists existing frames from the local corpus, saves cat/not-cat labels and
portable masks, provides a local click-to-contour segmentation fallback, and
queues explicit retrain/rescore actions for an operator to run later.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
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
VIDEO_STATUS_ROOT = REVIEW_ROOT / "videos"
TRAINING_DATASETS_ROOT = STATE_ROOT / "datasets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov"}
LABEL_NAMESPACE = "cat_projector_label_review"
SAM_ENDPOINT = os.environ.get("CAT_PROJECTOR_SAM_ENDPOINT", "").strip()
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
    review_status: str = "unreviewed"
    notes: str = ""

    def priority_tuple(self) -> tuple[float, float]:
        status_penalty = -1000.0 if self.review_status in {"relabeled_ok", "reviewed", "ok"} else 0.0
        return (status_penalty + min(1.0, self.frame_count / 250.0), self.mtime)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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


def _label_path(case_id: str) -> Path:
    return LABELS_ROOT / f"{_safe_slug(case_id)}.json"


def _mask_path(case_id: str, mask_id: str) -> Path:
    return MASKS_ROOT / _safe_slug(case_id) / f"{_safe_slug(mask_id)}.json"


def _video_status_path(video_id: str) -> Path:
    return VIDEO_STATUS_ROOT / f"{_safe_slug(video_id)}.json"


def _load_label_for_case(case_id: str) -> dict[str, Any]:
    return _read_json(_label_path(case_id))


def _load_status_for_video(video_id: str) -> dict[str, Any]:
    return _read_json(_video_status_path(video_id))


def _parse_bbox(raw: str | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
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
    rows_by_image: dict[Path, dict[str, str]] = {}
    for root in (STATE_ROOT / "datasets", DATASET_ROOT / "datasets", DATASET_ROOT / "detector-training"):
        if root.exists():
            rows_by_image.update(_label_rows_by_image(root))

    seen: set[Path] = set()
    cases: list[ReviewCase] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for image_path in root.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if _is_output_frame_dir_name(image_path.parent.name):
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
            for key in ("detector_cat_probability", "cat_probability", "probability"):
                try:
                    if row.get(key) not in {None, ""}:
                        probability = float(row[key])
                        break
                except ValueError:
                    pass
            human_label = (
                saved.get("label")
                or row.get("label_cat_present")
                or row.get("label_candidate_is_cat")
                or None
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
                    candidate_bbox_xywh=_parse_bbox(row.get("candidate_bbox_xywh") or saved.get("candidate_bbox_xywh")),
                    review_status=str(review_status),
                    human_label=str(human_label) if human_label else None,
                    notes=str(saved.get("notes") or row.get("notes") or ""),
                )
            )

    cases.sort(key=lambda item: item.priority_tuple(), reverse=True)
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


def _video_group_key(case: ReviewCase) -> Path:
    if case.source_recording_dir:
        return case.source_recording_dir
    for parent in case.image_path.parents:
        if _is_input_frame_dir_name(parent.name) or _is_output_frame_dir_name(parent.name):
            return parent.parent
    return case.image_path.parent


def _input_frame_paths_for_group(group_key: Path) -> list[Path]:
    for directory_name in PREFERRED_INPUT_FRAME_DIR_NAMES:
        frame_paths = _image_paths_under(group_key / directory_name)
        if frame_paths:
            return frame_paths
    return [
        image_path
        for image_path in _image_paths_under(group_key)
        if not _is_output_frame_dir_name(image_path.parent.name)
    ]


def _video_payload(video: ReviewVideo) -> dict[str, Any]:
    status = _load_status_for_video(video.id)
    video_token = _encode_path(video.source_video_path) if video.source_video_path else None
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
        "review_status": status.get("review_status") or video.review_status,
        "notes": status.get("notes") or video.notes,
        "mtime": video.mtime,
    }


def _discover_videos(limit: int) -> list[ReviewVideo]:
    cases = _discover_cases(10000)
    grouped: dict[Path, list[ReviewCase]] = {}
    for case in cases:
        grouped.setdefault(_video_group_key(case), []).append(case)

    videos: dict[str, ReviewVideo] = {}
    for key, group in grouped.items():
        group = sorted(group, key=lambda item: item.image_path.name)
        source_video = next((item.source_video_path for item in group if item.source_video_path), None)
        recording_dir = next((item.source_recording_dir for item in group if item.source_recording_dir), None)
        input_paths = _input_frame_paths_for_group(key)
        output_count = sum(1 for path in input_paths if _candidate_output_path(path))
        video_id = _video_id_for_path(recording_dir or key)
        status = _load_status_for_video(video_id)
        videos[video_id] = ReviewVideo(
            id=video_id,
            label=status.get("title") or (recording_dir.name if recording_dir else key.name),
            source=_source_name(recording_dir or key),
            mtime=max(item.mtime for item in group),
            source_recording_dir=recording_dir,
            source_video_path=source_video,
            frame_count=len(input_paths) or len(group),
            output_frame_count=output_count,
            review_status=status.get("review_status") or "unreviewed",
            notes=status.get("notes") or "",
        )

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
            frame_paths = _recording_frame_paths(recording_dir)
            if not chunks and not frame_paths:
                continue
            status = _load_status_for_video(video_id)
            newest = max([recording_dir.stat().st_mtime, *[path.stat().st_mtime for path in frame_paths[:200]]])
            videos[video_id] = ReviewVideo(
                id=video_id,
                label=status.get("title") or recording_dir.name,
                source="recordings",
                mtime=newest,
                source_recording_dir=recording_dir,
                source_video_path=chunks[0] if chunks else None,
                frame_count=len(frame_paths),
                output_frame_count=sum(1 for path in frame_paths if _candidate_output_path(path)),
                review_status=status.get("review_status") or "unreviewed",
                notes=status.get("notes") or "",
            )

    rows = sorted(videos.values(), key=lambda item: item.priority_tuple(), reverse=True)
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
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _frames_for_video(video_id: str, *, offset: int = 0, limit: int = 50) -> tuple[ReviewVideo, list[dict[str, Any]]]:
    video = next((item for item in _discover_videos(10000) if item.id == video_id), None)
    if video is None:
        raise ValueError(f"unknown review video: {video_id}")
    cases_by_path = {case.image_path: case for case in _discover_cases(10000)}
    if video.source_recording_dir:
        input_paths = _recording_frame_paths(video.source_recording_dir)
        if not input_paths:
            input_paths = sorted(
                case.image_path for case in cases_by_path.values() if case.source_recording_dir == video.source_recording_dir
            )
    else:
        group_key = next((_video_group_key(case) for case in cases_by_path.values() if _video_id_for_path(_video_group_key(case)) == video_id), None)
        input_paths = _input_frame_paths_for_group(group_key) if group_key else []
    selected = input_paths[max(0, offset) : max(0, offset) + max(1, min(limit, 500))]
    frames: list[dict[str, Any]] = []
    for frame_index, input_path in enumerate(selected, start=max(0, offset)):
        case = cases_by_path.get(input_path)
        case_id = case.id if case else _case_id_for_path(input_path)
        saved = _load_label_for_case(case_id)
        output_path = _candidate_output_path(input_path)
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
                "source_video_path": str(video.source_video_path) if video.source_video_path else None,
                "source_recording_dir": str(video.source_recording_dir) if video.source_recording_dir else None,
                "candidate_bbox_xywh": case.candidate_bbox_xywh if case else saved.get("candidate_bbox_xywh"),
                "detector_probability": case.detector_probability if case else saved.get("detector_probability"),
                "review_status": saved.get("review_status") or (case.review_status if case else "unreviewed"),
                "human_label": saved.get("label") or (case.human_label if case else None),
                "notes": saved.get("notes") or (case.notes if case else ""),
                "source_size_px": _image_size(input_path),
            }
        )
    return video, frames


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
        padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
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
    label = str(payload.get("label") or payload.get("human_label") or "unsure")
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
        "label_cat_present": "yes" if label == "cat" else "no" if label == "not_cat" else "",
        "label_candidate_is_cat": payload.get("label_candidate_is_cat")
        or ("yes" if label == "cat" else "no" if label == "not_cat" else ""),
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


def _mask_bbox_for_label(label: dict[str, Any]) -> str:
    for mask in label.get("masks") or []:
        if not isinstance(mask, dict):
            continue
        bbox = _bbox_dict_to_csv(mask.get("bbox_xywh"))
        if bbox:
            return bbox
    return _bbox_dict_to_csv(label.get("candidate_bbox_xywh"))


def _source_chunk_for_label(label: dict[str, Any], image_path: Path, recording_dir: Path | None) -> tuple[str, str]:
    if recording_dir is None:
        return "", ""
    chunk_match = re.match(r"chunk_(\d+)_(\d+)\.(?:jpe?g|png|webp)$", image_path.name, re.IGNORECASE)
    if not chunk_match:
        return "", ""
    chunk_index = int(chunk_match.group(1))
    frame_index = int(chunk_match.group(2))
    chunk_path = recording_dir / f"chunk_{chunk_index:04d}.mp4"
    offset = frame_index / 15.0
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
        target = frames_dir / f"{index:04d}_{_safe_slug(str(label.get('case_id') or image_path.stem))}{image_path.suffix.lower()}"
        shutil.copy2(image_path, target)
        label_kind = str(label.get("label") or "")
        recording_raw = str(label.get("source_recording_dir") or "")
        recording_dir = Path(recording_raw).expanduser() if recording_raw else None
        if recording_dir is None or not recording_dir.exists():
            _source_video, inferred_recording = _recording_context(image_path)
            recording_dir = inferred_recording
        source_chunk, source_offset = _source_chunk_for_label(label, image_path, recording_dir)
        bbox = _mask_bbox_for_label(label)
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
                "notes": f"materialized from review label {label.get('_label_file')}; {label.get('notes') or ''}".strip(),
                "source_recording_dir": str(recording_dir) if recording_dir else "",
                "source_chunk": source_chunk,
                "source_offset_seconds": source_offset,
                "video_slug": Path(str(label.get("source_video_path") or "")).name,
                "candidate_reason": "human_review_ui_mask" if label_kind == "cat" else "human_review_ui_not_cat",
            }
        )
        rows.append(row)
        copied.append(
            {
                "label_file": str(label.get("_label_file") or ""),
                "source_image": str(image_path),
                "copied_image": str(target),
                "label": label_kind,
                "bbox_xywh": bbox,
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
        "source_action_payload": {
            key: value
            for key, value in payload.items()
            if key not in {"training_package"}
        },
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


def _queue_action(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    action_id = f"{int(time.time())}-{_safe_slug(action)}-{hashlib.sha1(os.urandom(12)).hexdigest()[:8]}"
    row = {
        "kind": LABEL_NAMESPACE + "_action_v1",
        "id": action_id,
        "action": action,
        "status": "queued",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "payload": payload,
        "log": [
            "queued by review UI; operator must run the matching offline command",
        ],
    }
    _write_json(QUEUE_ROOT / f"{action_id}.json", row)
    return row


def _list_actions(limit: int = 50) -> list[dict[str, Any]]:
    if not QUEUE_ROOT.exists():
        return []
    paths = sorted(QUEUE_ROOT.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [_read_json(path) for path in paths[:limit]]


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
            if len(parts) >= 6 and parts[5] == "status":
                video_id = unquote(parts[4])
                self._send_json(_load_status_for_video(video_id) or {"video_id": video_id, "review_status": "unreviewed"})
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
    for directory in (REVIEW_ROOT, LABELS_ROOT, MASKS_ROOT, QUEUE_ROOT, VIDEO_STATUS_ROOT):
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
        globals()["VIDEO_STATUS_ROOT"] = fake_review / "videos"
        globals()["TRAINING_DATASETS_ROOT"] = fake_state / "datasets"
        globals()["SCAN_ROOTS"] = (fake_dataset / "datasets",)
        globals()["ALLOWED_ROOTS"] = (fake_dataset, fake_state, fake_review)
        for directory in (LABELS_ROOT, MASKS_ROOT, QUEUE_ROOT, VIDEO_STATUS_ROOT):
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
            saved_cat = post(
                "/api/cat-projector-label-review/labels",
                {
                    "case_id": cat_case["id"],
                    "video_id": videos[0]["id"],
                    "frame_index": 1,
                    "image_path": cat_case["image_path"],
                    "model_output_path": video_frames[1]["model_output_path"],
                    "label": "cat",
                    "review_status": "saved",
                    "masks": [{"id": "sher", "label": "Sher", "kind": "cat", **contour}],
                    "notes": "fake smoke cat",
                },
            )
            saved_not_cat = post(
                "/api/cat-projector-label-review/labels",
                {
                    "case_id": not_cat_case["id"],
                    "video_id": videos[0]["id"],
                    "frame_index": 2,
                    "image_path": not_cat_case["image_path"],
                    "label": "not_cat",
                    "review_status": "saved",
                    "masks": [],
                    "notes": "fake smoke not-cat",
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
            video_status = post(
                f"/api/cat-projector-label-review/videos/{videos[0]['id']}/status",
                {"review_status": "relabeled_ok", "notes": "fake video ok"},
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
        if saved_cat["mask_refs"] == []:
            raise AssertionError("cat label did not write mask ref")
        if saved_not_cat["label_cat_present"] != "no":
            raise AssertionError("not-cat label did not save frame-level no")
        if retrain["status"] != "queued" or rescore["status"] != "queued":
            raise AssertionError("actions were not queued")
        if video_status["review_status"] != "relabeled_ok":
            raise AssertionError("video status was not saved")
        print(
            json.dumps(
                {
                    "cases": cases,
                    "videos": videos,
                    "video_frames": video_frames,
                    "cat_label": saved_cat,
                    "not_cat_label": saved_not_cat,
                    "video_status": video_status,
                    "actions": [retrain, rescore],
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
        globals()["VIDEO_STATUS_ROOT"] = original_video_status
        globals()["TRAINING_DATASETS_ROOT"] = original_training_datasets
        globals()["SCAN_ROOTS"] = original_scan_roots
        globals()["ALLOWED_ROOTS"] = original_allowed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--fake-smoke", action="store_true", help="Run the repo-local fake-corpus smoke test and exit.")
    parser.add_argument("--tmp-root", type=Path, default=Path("tmp/cat_projector_label_review_smoke"))
    args = parser.parse_args()
    if args.fake_smoke:
        return run_fake_smoke(args.tmp_root)
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
