#!/usr/bin/env python3
"""Record Cat TV play sessions from the projector camera for later learning.

This deliberately records the projector camera feed directly. Home Assistant
and Frigate session ids are useful metadata, but they are not proof that usable
training video exists.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import urlopen

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_config as site_config  # noqa: E402

DEFAULT_HA_URL = site_config.home_assistant_url()
DEFAULT_CAMERA_SNAPSHOT_URL = site_config.camera_snapshot_url()
DEFAULT_CAMERA_VIDEO_URL = site_config.camera_video_url()
DEFAULT_ROOT = site_config.learning_root()
_CONFIGURED_RECORDINGS_ROOT = os.environ.get("CAT_TV_RECORDINGS_ROOT")
DEFAULT_RECORDINGS_ROOT = Path(_CONFIGURED_RECORDINGS_ROOT).expanduser() if _CONFIGURED_RECORDINGS_ROOT else None
DEFAULT_DURATION_SECONDS = 30 * 60
DEFAULT_SEGMENT_SECONDS = 5
DEFAULT_RECORDER_MIN_CHUNK_EFFECTIVE_FPS = 20.0
DEFAULT_TELEGRAM_MAX_AGE_SECONDS = 2 * 60 * 60
DEFAULT_TELEGRAM_WINDOW_CHUNKS = 6
DEFAULT_TELEGRAM_MAX_CLIPS = 3
DEFAULT_TELEGRAM_FINAL_SCAN_MAX_CHUNKS = 72
DEFAULT_TELEGRAM_LIVE_COOLDOWN_SECONDS = 5 * 60
DEFAULT_TELEGRAM_LIVE_SCAN_COOLDOWN_SECONDS = 60
DEFAULT_TELEGRAM_FINAL_ERROR_COOLDOWN_SECONDS = 5 * 60
DEFAULT_TELEGRAM_MOTION_SCAN_FPS = 4
DEFAULT_TELEGRAM_DETECTOR_MIN_PROBABILITY = 0.60
DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM = 85.0
DEFAULT_TELEGRAM_JUMP_SCAN_FPS = 4
DEFAULT_TELEGRAM_JUMP_RENDER_FPS = 20
DEFAULT_TELEGRAM_ENHANCED_FPS = 40
DEFAULT_PUBLISHABLE_NEURAL_UPSCALE_SECONDS = 420.0
DEFAULT_TIKTOK_OUTPUT_SIZE = (1080, 1920)
DEFAULT_TIKTOK_SAFE_ZONE = {"left": 54, "top": 54, "right": 54, "bottom": 260}
DEFAULT_PUBLISHABLE_MIN_SOURCE_FPS = DEFAULT_RECORDER_MIN_CHUNK_EFFECTIVE_FPS
DEFAULT_PUBLISHABLE_MIN_JUMP_COUNT = 3
DEFAULT_REALESRGAN_BIN = Path("~/.local/bin/realesrgan-ncnn-vulkan").expanduser()
DEFAULT_REALESRGAN_MODEL_DIR = Path("~/.local/opt/models").expanduser()
DEFAULT_RIFE_BIN = Path("~/.local/bin/rife-ncnn-vulkan").expanduser()
DEFAULT_RIFE_MODEL_DIR = Path("~/.local/opt/rife-ncnn-vulkan-20221029-ubuntu/rife-v4.6").expanduser()
DEFAULT_TELEGRAM_JUMP_ROI = (0, 0, 1280, 720)
DEFAULT_TELEGRAM_JUMP_DARK_THRESHOLD = 80
DEFAULT_TELEGRAM_JUMP_SCALE = 4
DEFAULT_TELEGRAM_JUMP_MIN_AREA_PX = 2500
DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_PX = 45
DEFAULT_TELEGRAM_JUMP_MAX_WIDTH_PX = 360
DEFAULT_TELEGRAM_RENDER_MIN_AREA_PX = 650
DEFAULT_TELEGRAM_RENDER_MIN_HEIGHT_PX = 16
DEFAULT_TELEGRAM_RENDER_MAX_WIDTH_PX = 460
DEFAULT_TELEGRAM_RESULTS_HOLD_SECONDS = 3.0
DEFAULT_CONTINUITY_HARD_CASE_MAX_MISSED_FRAMES = 30
DEFAULT_CONTINUITY_HARD_CASE_MAX_REAPPEAR_DISTANCE_PX = 190.0
DEFAULT_CAT_FOOD_REWARD_MIN_HEIGHT_CM = 150.0
DEFAULT_CAT_FOOD_REWARD_COOLDOWN_SECONDS = 60 * 60
DEFAULT_CAT_FOOD_REWARD_BUTTON_ENTITY_ID = site_config.entity("treat_dispenser_button")
DEFAULT_CAT_TV_PLAY_ROOT = REPO_ROOT
DEFAULT_CAT_PROJECTOR_REVIEW_UI_URL = os.environ.get(
    "CAT_PROJECTOR_REVIEW_UI_URL",
    site_config.service_url("review_ui_url"),
)
DEFAULT_REVIEW_FRAME_FPS = 30.0
TELEGRAM_CURATED_REASONS = {"cat_candidate_detector", "outside_projection_motion"}
VERIFY_TIMEOUT_SECONDS = 25
PROJECTOR_POLYGON_1280 = (
    (40.22, 57.27),
    (937.92, 101.0),
    (908.0, 599.0),
    (48.16, 680.97),
)
_SOURCE_VIDEO_CACHE: dict[str, tuple[Any, float]] = {}
_CAT_TV_PLAY_NOTIFICATION_POLICY: Any | None = None
_CAT_TV_PLAY_PUBLISH_RENDER: Any | None = None

from cat_tv_play_notify import DeliveryResult, deliver_notification  # noqa: E402
from cat_tv_play_time import parse_iso_timestamp  # noqa: E402
from scripts import cat_projector_live_jump_watch as live_jump  # noqa: E402
from scripts.cat_projector_clip_policy import CameraHumanCheck, detect_camera_human_presence  # noqa: E402
from scripts.cat_projector_frame_detector import (  # noqa: E402
    DEFAULT_METADATA_PATH as DEFAULT_DETECTOR_METADATA_PATH,
)
from scripts.cat_projector_frame_detector import DEFAULT_MODEL_PATH as DEFAULT_DETECTOR_MODEL_PATH  # noqa: E402
from scripts.cat_projector_frame_detector import (  # noqa: E402
    Candidate,
    best_cat_candidate_from_model,
    default_model_available,
    detect_source_subtracted_candidate_components,
    load_model,
    score_candidates,
    source_subtracted_residual,
)


@dataclass(frozen=True)
class HaSession:
    session_id: str
    frigate_event_id: str
    video_url: str
    session_state: str
    renderer_state: str


@dataclass(frozen=True)
class JumpHighlight:
    chunks: list[Path]
    max_height_cm: float
    max_record: dict[str, Any]
    candidate_count: int


@dataclass(frozen=True)
class SourceIntegrity:
    status: str
    reason: str
    frame_path: Path | None = None
    recording_dir: Path | None = None
    chunk_path: Path | None = None
    chunk_index: int | None = None
    frame_index: int | None = None
    offset_seconds: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "frame_path": str(self.frame_path) if self.frame_path is not None else None,
            "recording_dir": str(self.recording_dir) if self.recording_dir is not None else None,
            "chunk_path": str(self.chunk_path) if self.chunk_path is not None else None,
            "chunk": self.chunk_index,
            "frame_index": self.frame_index,
            "offset_seconds": round(self.offset_seconds, 3) if self.offset_seconds is not None else None,
        }


@dataclass(frozen=True)
class RescoreJumpCandidate:
    video: dict[str, Any]
    integrity: SourceIntegrity


@dataclass(frozen=True)
class PublishableJumpSegment:
    candidate: RescoreJumpCandidate
    frames: tuple[Path, ...]
    active_frame_index: int
    source_fps: float
    source_duration_seconds: float
    pre_available_seconds: float
    post_available_seconds: float
    probe_raw_prefix: Path | None = None
    frame_source: str = "recording_chunk"


def _now_compact() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")


def _load_ha_token() -> str:
    return site_config.home_assistant_token()


def _ha_state(base_url: str, token: str, entity_id: str) -> dict[str, Any]:
    import requests

    response = requests.get(
        f"{base_url.rstrip('/')}/api/states/{entity_id}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=8,
    )
    response.raise_for_status()
    return dict(response.json())


def _ha_call_service(base_url: str, token: str, domain: str, service: str, data: dict[str, Any]) -> dict[str, Any]:
    import requests

    response = requests.post(
        f"{base_url.rstrip('/')}/api/services/{domain}/{service}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=data,
        timeout=8,
    )
    response.raise_for_status()
    try:
        return {"response": response.json()}
    except ValueError:
        return {"response_text": response.text}


def _read_ha_session(base_url: str) -> HaSession:
    token = _load_ha_token()
    session = _ha_state(base_url, token, site_config.entity("session_id_text"))
    event = _ha_state(base_url, token, site_config.entity("frigate_event_id_text"))
    video = _ha_state(base_url, token, site_config.entity("video_url_text"))
    latch = _ha_state(base_url, token, site_config.entity("session_boolean"))
    renderer = _ha_state(base_url, token, site_config.entity("media_player"))
    return HaSession(
        session_id=str(session.get("state") or ""),
        frigate_event_id=str(event.get("state") or ""),
        video_url=str(video.get("state") or ""),
        session_state=str(latch.get("state") or ""),
        renderer_state=str(renderer.get("state") or ""),
    )


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _capture_snapshot(snapshot_url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(snapshot_url, timeout=8) as response:
        data = response.read()
    if len(data) < 1024:
        raise RuntimeError(f"snapshot was too small to trust: {len(data)} bytes")
    output.write_bytes(data)


def _ffmpeg_chunk_command(
    *,
    camera_url: str,
    output_path: Path,
    segment_seconds: int,
) -> list[str]:
    return [
        "timeout",
        "--kill-after=3s",
        f"{segment_seconds + 10}s",
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-y",
        "-rw_timeout",
        "5000000",
        "-use_wallclock_as_timestamps",
        "1",
        "-i",
        camera_url,
        "-an",
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-t",
        str(segment_seconds),
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _ffprobe_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return 0
    raw = result.stdout.strip().splitlines()
    if not raw:
        return 0
    try:
        return int(raw[-1])
    except ValueError:
        return 0


def _ffprobe_frame_rate(path: Path) -> float | None:
    result = subprocess.run(
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
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0] if isinstance(streams[0], dict) else {}
    for key in ("avg_frame_rate", "r_frame_rate"):
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
            try:
                value = float(raw)
            except ValueError:
                continue
            if value > 0:
                return value
    return None


def _chunk_frame_rate(path: Path | None) -> float:
    if path is None:
        return DEFAULT_REVIEW_FRAME_FPS
    return _ffprobe_frame_rate(path) or DEFAULT_REVIEW_FRAME_FPS


def _chunk_effective_frame_rate(path: Path | None, *, frames: int | None = None) -> float:
    if path is None:
        return 0.0
    frame_count = _ffprobe_frame_count(path) if frames is None else frames
    duration = _ffprobe_duration_seconds(path)
    if duration is None or duration <= 0.0:
        return 0.0
    return frame_count / duration


def _ffprobe_duration_seconds(path: Path | None) -> float | None:
    if path is None:
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams")
        stream = streams[0] if isinstance(streams, list) and streams and isinstance(streams[0], dict) else {}
        duration = float(stream.get("duration") or 0.0)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return duration if duration > 0 else None


def _ffprobe_frame_times(path: Path) -> list[float]:
    result = subprocess.run(
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
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []
    times: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        raw = frame.get("best_effort_timestamp_time")
        if raw in (None, ""):
            raw = frame.get("pkt_pts_time")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            times.append(value)
    return times


def _source_frame_index_for_offset(path: Path, offset_seconds: float) -> int:
    times = _ffprobe_frame_times(path)
    if times:
        return min(range(len(times)), key=lambda index: (abs(times[index] - offset_seconds), index))
    return max(0, int(round(offset_seconds * _chunk_frame_rate(path))))


def _frame_offset_seconds_for_index(path: Path, frame_index: int) -> float:
    times = _ffprobe_frame_times(path)
    if 0 <= frame_index < len(times):
        return float(times[frame_index])
    return max(0.0, frame_index / max(_chunk_frame_rate(path), 1.0))


def _frame_label_parts(path: Path) -> tuple[int, int] | None:
    match = re.match(r"(?:chunk|source)_(\d+)_(\d+)\.(?:jpe?g|png|webp)$", path.name, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _source_integrity_for_rescore_video(video: dict[str, Any]) -> SourceIntegrity:
    frame_raw = video.get("max_frame_path")
    recording_raw = video.get("recording_dir")
    frame_path = Path(str(frame_raw)).expanduser() if frame_raw else None
    recording_dir = Path(str(recording_raw)).expanduser() if recording_raw else None
    if frame_path is None:
        return SourceIntegrity(status="missing", reason="missing_max_frame_path", recording_dir=recording_dir)
    if recording_dir is None:
        return SourceIntegrity(status="missing", reason="missing_recording_dir", frame_path=frame_path)
    try:
        frame_resolved = frame_path.resolve()
        recording_resolved = recording_dir.resolve()
    except OSError:
        return SourceIntegrity(
            status="broken", reason="path_resolve_failed", frame_path=frame_path, recording_dir=recording_dir
        )
    try:
        frame_resolved.relative_to(recording_resolved)
    except ValueError:
        return SourceIntegrity(
            status="broken",
            reason="frame_path_outside_recording_dir",
            frame_path=frame_path,
            recording_dir=recording_dir,
        )
    if not frame_resolved.exists():
        return SourceIntegrity(
            status="broken",
            reason="frame_path_missing",
            frame_path=frame_path,
            recording_dir=recording_dir,
        )
    parts = _frame_label_parts(frame_path)
    if parts is None:
        return SourceIntegrity(
            status="broken",
            reason="frame_label_not_chunk_frame",
            frame_path=frame_path,
            recording_dir=recording_dir,
        )
    chunk_index, frame_index = parts
    chunk_path = recording_dir / f"chunk_{chunk_index:04d}.mp4"
    if not chunk_path.exists():
        return SourceIntegrity(
            status="broken",
            reason="chunk_path_missing",
            frame_path=frame_path,
            recording_dir=recording_dir,
            chunk_path=chunk_path,
            chunk_index=chunk_index,
            frame_index=frame_index,
        )
    frame_count = _ffprobe_frame_count(chunk_path)
    if frame_count and frame_index >= frame_count:
        return SourceIntegrity(
            status="broken",
            reason="frame_index_outside_chunk",
            frame_path=frame_path,
            recording_dir=recording_dir,
            chunk_path=chunk_path,
            chunk_index=chunk_index,
            frame_index=frame_index,
        )
    return SourceIntegrity(
        status="clean",
        reason="frame_path_matches_recording_chunk",
        frame_path=frame_path,
        recording_dir=recording_dir,
        chunk_path=chunk_path,
        chunk_index=chunk_index,
        frame_index=frame_index,
        offset_seconds=_frame_offset_seconds_for_index(chunk_path, frame_index),
    )


def _verify_first_chunk(recording_dir: Path, *, min_frames: int = 10) -> Path:
    deadline = time.monotonic() + VERIFY_TIMEOUT_SECONDS
    latest_error = "no chunk appeared"
    while time.monotonic() < deadline:
        chunks = [path for path in sorted(recording_dir.glob("chunk_*.mp4")) if ".part." not in path.name]
        for chunk in chunks:
            if chunk.stat().st_size < 4096:
                latest_error = f"{chunk.name} is still too small"
                continue
            frames = _ffprobe_frame_count(chunk)
            effective_fps = _chunk_effective_frame_rate(chunk, frames=frames)
            if frames >= min_frames and effective_fps >= DEFAULT_RECORDER_MIN_CHUNK_EFFECTIVE_FPS:
                return chunk
            latest_error = (
                f"{chunk.name} has {frames} readable frames at {effective_fps:.2f} fps; "
                f"need >= {min_frames} frames and >= {DEFAULT_RECORDER_MIN_CHUNK_EFFECTIVE_FPS:.2f} fps"
            )
        time.sleep(1)
    raise RuntimeError(f"recorder did not produce a verified chunk within {VERIFY_TIMEOUT_SECONDS}s: {latest_error}")


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop_existing(pid_path: Path) -> None:
    pid = _read_pid(pid_path)
    if not pid or not _pid_is_running(pid):
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return
        time.sleep(0.25)
    raise RuntimeError(f"existing recorder pid {pid} did not stop")


def _recordings_root(args: argparse.Namespace, root: Path) -> Path:
    configured = getattr(args, "recordings_root", None)
    return Path(configured).expanduser() if configured is not None else root / "recordings"


def _latest_manifest(recordings_root: Path) -> dict[str, Any] | None:
    manifests = sorted(recordings_root.glob("*/manifest.json"))
    if not manifests:
        return None
    try:
        return json.loads(manifests[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_manifest_path(recordings_root: Path) -> Path | None:
    manifests = sorted(recordings_root.glob("*/manifest.json"))
    if not manifests:
        return None
    return manifests[-1]


def _recorder_matches_session(recordings_root: Path, ha_session: HaSession) -> bool:
    manifest = _latest_manifest(recordings_root)
    if not manifest:
        return False
    if str(manifest.get("ha_session_id") or "") != ha_session.session_id:
        return False
    recording_dir = Path(str(manifest.get("recording_dir") or ""))
    return (recording_dir / "verified.json").exists()


def parse_iso_datetime(value: str) -> datetime | None:
    return parse_iso_timestamp(value)


_parse_iso_datetime = parse_iso_datetime


def _cat_tv_play_notification_policy() -> Any | None:
    global _CAT_TV_PLAY_NOTIFICATION_POLICY
    if _CAT_TV_PLAY_NOTIFICATION_POLICY is not None:
        return _CAT_TV_PLAY_NOTIFICATION_POLICY
    policy_path = DEFAULT_CAT_TV_PLAY_ROOT / "custom_components" / "cat_tv_play" / "notification_policy.py"
    if not policy_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("cat_tv_play_notification_policy", policy_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _CAT_TV_PLAY_NOTIFICATION_POLICY = module
    return module


def _cat_tv_play_publish_render() -> Any | None:
    global _CAT_TV_PLAY_PUBLISH_RENDER
    if _CAT_TV_PLAY_PUBLISH_RENDER is not None:
        return _CAT_TV_PLAY_PUBLISH_RENDER
    helper_path = DEFAULT_CAT_TV_PLAY_ROOT / "custom_components" / "cat_tv_play" / "publish_render.py"
    if not helper_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("cat_tv_play_publish_render", helper_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _CAT_TV_PLAY_PUBLISH_RENDER = module
    return module


def _require_cat_tv_play_publish_render() -> Any:
    module = _cat_tv_play_publish_render()
    if module is None:
        raise RuntimeError(f"cat-tv-play publish_render helper not found under {DEFAULT_CAT_TV_PLAY_ROOT}")
    return module


def _cat_tv_play_live_marker_decision(
    marker: dict[str, Any],
    *,
    now: datetime,
    sent_cooldown_seconds: int,
    scan_cooldown_seconds: int,
) -> tuple[bool, dict[str, Any] | None]:
    policy = _cat_tv_play_notification_policy()
    if policy is None:
        return False, None
    decision = policy.decide_live_notification_from_marker(
        marker,
        now=now,
        sent_cooldown_seconds=sent_cooldown_seconds,
        scan_cooldown_seconds=scan_cooldown_seconds,
    )
    if decision is None:
        return True, None
    return True, decision.to_payload()


def _select_clip_windows(chunks: list[Path], *, window_chunks: int, max_clips: int) -> list[list[Path]]:
    if not chunks:
        return []
    span = max(1, min(window_chunks, len(chunks)))
    starts = [0, max(0, (len(chunks) - span) // 2), max(0, len(chunks) - span)]
    windows: list[list[Path]] = []
    seen: set[tuple[str, ...]] = set()
    for start in starts:
        window = chunks[start : start + span]
        key = tuple(str(path) for path in window)
        if key in seen:
            continue
        seen.add(key)
        windows.append(window)
        if len(windows) >= max_clips:
            break
    return windows


def _motion_records_to_clip_windows(
    chunks_by_index: dict[int, Path],
    records: list[dict[str, Any]],
    *,
    window_chunks: int,
    max_clips: int,
    separation_chunks: int = 8,
) -> list[list[Path]]:
    windows: list[list[Path]] = []
    chosen_chunks: list[int] = []
    span = max(1, window_chunks)
    available = sorted(chunks_by_index)
    if not available:
        return windows
    first_available = available[0]
    last_available = available[-1]

    for record in sorted(records, key=lambda item: float(item.get("score") or 0), reverse=True):
        chunk_index = int(record["chunk"])
        if chunk_index not in chunks_by_index:
            continue
        if any(abs(chunk_index - chosen) < separation_chunks for chosen in chosen_chunks):
            continue

        start = max(first_available, chunk_index - span // 2)
        end = min(last_available, start + span - 1)
        start = max(first_available, end - span + 1)
        window = [chunks_by_index[index] for index in range(start, end + 1) if index in chunks_by_index]
        if len(window) < min(span, len(available)):
            continue
        windows.append(window)
        chosen_chunks.append(chunk_index)
        if len(windows) >= max_clips:
            break
    return windows


def _chunk_index(path: Path) -> int | None:
    match = re.match(r"chunk_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _telegram_record_rank(record: dict[str, Any]) -> tuple[int, float]:
    """Prefer visible Sher over projected prey when choosing Telegram clips."""

    reason = str(record.get("reason") or "")
    score = float(record.get("score") or 0)
    if reason == "cat_candidate_detector":
        return 3, score
    if reason == "outside_projection_motion":
        return 2, score
    if reason == "dark_body_on_projection":
        return 1, score
    return 0, score


def _curated_telegram_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [record for record in records if str(record.get("reason") or "") in TELEGRAM_CURATED_REASONS],
        key=_telegram_record_rank,
        reverse=True,
    )


def _score_motion_highlight_records(
    chunks: list[Path],
    *,
    sample_fps: int,
    scale_width: int = 640,
    detector_model: Path | None = None,
    detector_metadata: Path | None = None,
    detector_min_probability: float = DEFAULT_TELEGRAM_DETECTOR_MIN_PROBABILITY,
) -> list[dict[str, Any]]:
    """Find frames where Sher moves, gated by the shared cat candidate detector when available."""

    import numpy as np
    from PIL import Image, ImageDraw
    from scipy import ndimage

    records: list[dict[str, Any]] = []
    detector_loaded_model = None
    detector_loaded_metadata = None
    if detector_model is not None:
        detector_loaded_model, detector_loaded_metadata = load_model(
            detector_model,
            detector_metadata or DEFAULT_DETECTOR_METADATA_PATH,
        )
    with tempfile.TemporaryDirectory(prefix="cat-projector-motion-") as tmpdir:
        frames_dir = Path(tmpdir)
        for chunk in chunks:
            match = re.match(r"chunk_(\d+)", chunk.stem)
            if not match:
                continue
            chunk_index = int(match.group(1))
            pattern = frames_dir / f"chunk_{chunk_index:04d}_%03d.jpg"
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(chunk),
                    "-vf",
                    f"fps={sample_fps},scale={scale_width}:-1",
                    str(pattern),
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"failed to sample {chunk}: {result.stderr[-400:]}")

        frame_paths = sorted(frames_dir.glob("chunk_*.jpg"))
        if not frame_paths:
            return records

        first = Image.open(frame_paths[0]).convert("L")
        width, height = first.size
        scale_x = width / 1280.0
        scale_y = height / 720.0
        projector_polygon = [(x * scale_x, y * scale_y) for x, y in PROJECTOR_POLYGON_1280]
        projector_mask_img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(projector_mask_img).polygon(projector_polygon, fill=255)
        projector_mask = np.array(projector_mask_img) > 0
        outside_mask = ~projector_mask

        previous: np.ndarray | None = None
        for path in frame_paths:
            image = Image.open(path).convert("L")
            arr = np.array(image)
            match = re.match(r"chunk_(\d+)_(\d+)\.jpg", path.name)
            if not match:
                continue
            chunk_index = int(match.group(1))
            frame_index = int(match.group(2))
            offset_seconds = (frame_index - 1) / sample_fps

            frame_records: list[dict[str, Any]] = []
            if (
                detector_model is not None
                and detector_loaded_model is not None
                and detector_loaded_metadata is not None
            ):
                prediction = best_cat_candidate_from_model(
                    image.convert("RGB"),
                    model=detector_loaded_model,
                    metadata=detector_loaded_metadata,
                    fallback_model_path=detector_model,
                    min_probability=detector_min_probability,
                )
                if prediction.candidate is not None:
                    candidate = prediction.candidate
                    x, y, w, h = candidate.bbox_xywh
                    frame_records.append(
                        {
                            "chunk": chunk_index,
                            "offset_seconds": round(offset_seconds, 3),
                            "score": round(prediction.cat_probability * 100000.0, 1),
                            "reason": "cat_candidate_detector",
                            "bbox": [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))],
                            "area": candidate.area_px,
                            "cat_probability": round(prediction.cat_probability, 4),
                            "model_path": prediction.model_path,
                        }
                    )

            if previous is not None:
                diff = np.abs(arr.astype(np.int16) - previous.astype(np.int16)).astype(np.uint8)
                moving = (diff > 28) & outside_mask
                moving[: max(1, int(height * 0.08)), :] = False
                labels, _count = ndimage.label(moving)
                for label_id, box in enumerate(ndimage.find_objects(labels), start=1):
                    if box is None:
                        continue
                    ys, xs = box
                    component = labels[box] == label_id
                    area = int(component.sum())
                    if area < 180 or area > width * height * 0.045:
                        continue
                    x0, x1 = xs.start, xs.stop
                    y0, y1 = ys.start, ys.stop
                    component_height = y1 - y0
                    component_width = x1 - x0
                    if component_height > height * 0.7 and component_width < width * 0.04:
                        continue
                    if y1 < height * 0.72:
                        continue
                    tall_boost = min(2.5, component_height / max(1.0, height * 0.22))
                    score = area * (1.0 + tall_boost)
                    frame_records.append(
                        {
                            "chunk": chunk_index,
                            "offset_seconds": round(offset_seconds, 3),
                            "score": round(score, 1),
                            "reason": "outside_projection_motion",
                            "bbox": [x0, y0, x1, y1],
                            "area": area,
                        }
                    )

            screen_values = arr[projector_mask]
            median = float(np.median(screen_values)) if screen_values.size else 180.0
            dark = (arr < min(115, median - 45)) & projector_mask
            labels, _count = ndimage.label(dark)
            for label_id, box in enumerate(ndimage.find_objects(labels), start=1):
                if box is None:
                    continue
                ys, xs = box
                component = labels[box] == label_id
                area = int(component.sum())
                if area < 500 or area > width * height * 0.05:
                    continue
                x0, x1 = xs.start, xs.stop
                y0, y1 = ys.start, ys.stop
                component_height = y1 - y0
                component_width = x1 - x0
                if x0 < width * 0.06:
                    continue
                if component_height < height * 0.08 or component_width < width * 0.025:
                    continue
                if component_height > height * 0.82 and x0 < width * 0.08:
                    continue
                top_boost = max(0.0, (height * 0.78 - y0) / max(1.0, height * 0.78))
                score = area * (1.0 + top_boost) * 3.0
                frame_records.append(
                    {
                        "chunk": chunk_index,
                        "offset_seconds": round(offset_seconds, 3),
                        "score": round(score, 1),
                        "reason": "dark_body_on_projection",
                        "bbox": [x0, y0, x1, y1],
                        "area": area,
                    }
                )

            if frame_records:
                records.append(max(frame_records, key=_telegram_record_rank))
            previous = arr
    return sorted(records, key=_telegram_record_rank, reverse=True)


def _select_motion_highlight_clip_windows(
    chunks: list[Path],
    *,
    window_chunks: int,
    max_clips: int,
    sample_fps: int = DEFAULT_TELEGRAM_MOTION_SCAN_FPS,
    detector_model: Path | None = None,
    detector_metadata: Path | None = None,
    detector_min_probability: float = DEFAULT_TELEGRAM_DETECTOR_MIN_PROBABILITY,
) -> tuple[list[list[Path]], list[dict[str, Any]]]:
    chunks_by_index: dict[int, Path] = {}
    for chunk in chunks:
        match = re.match(r"chunk_(\d+)", chunk.stem)
        if match:
            chunks_by_index[int(match.group(1))] = chunk
    records = _score_motion_highlight_records(
        chunks,
        sample_fps=sample_fps,
        detector_model=detector_model,
        detector_metadata=detector_metadata,
        detector_min_probability=detector_min_probability,
    )
    windows = _motion_records_to_clip_windows(
        chunks_by_index,
        _curated_telegram_records(records),
        window_chunks=window_chunks,
        max_clips=max_clips,
    )
    return windows, records


def _select_recent_live_highlight_clip_window(
    chunks: list[Path],
    *,
    window_chunks: int,
    sample_fps: int = DEFAULT_TELEGRAM_MOTION_SCAN_FPS,
    detector_model: Path | None = None,
    detector_metadata: Path | None = None,
    detector_min_probability: float = DEFAULT_TELEGRAM_DETECTOR_MIN_PROBABILITY,
) -> tuple[list[Path] | None, list[dict[str, Any]]]:
    span = max(1, window_chunks)
    scan_chunks = chunks[-max(span * 4, span) :]
    selected_windows, records = _select_motion_highlight_clip_windows(
        scan_chunks,
        window_chunks=span,
        max_clips=1,
        sample_fps=sample_fps,
        detector_model=detector_model,
        detector_metadata=detector_metadata,
        detector_min_probability=detector_min_probability,
    )
    return (selected_windows[0] if selected_windows else None), records


def _extract_video_frames(
    chunks: list[Path],
    frames_dir: Path,
    *,
    fps: int,
    scale_width: int | None = None,
) -> list[Path]:
    for chunk in chunks:
        chunk_index = _chunk_index(chunk)
        if chunk_index is None:
            continue
        pattern = frames_dir / f"chunk_{chunk_index:04d}_%05d.jpg"
        video_filter = f"fps={fps}"
        if scale_width is not None:
            video_filter = f"{video_filter},scale={scale_width}:-1"
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(chunk),
                "-vf",
                video_filter,
                str(pattern),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to extract frames from {chunk}: {result.stderr[-400:]}")
    return sorted(frames_dir.glob("chunk_*.jpg"))


def _extract_exact_video_frames(chunks: list[Path], frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk_index = _chunk_index(chunk)
        if chunk_index is None:
            continue
        pattern = frames_dir / f"chunk_{chunk_index:04d}_%05d.jpg"
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(chunk),
                "-vsync",
                "0",
                "-start_number",
                "0",
                str(pattern),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to extract exact source frames from {chunk}: {result.stderr[-400:]}")
    return sorted(frames_dir.glob("chunk_*.jpg"))


def _recording_dir_from_chunks(chunks: list[Path]) -> Path | None:
    if not chunks:
        return None
    first = chunks[0].resolve()
    if first.parent.name == "chunks":
        return first.parent.parent
    return first.parent


def _read_recording_manifest_for_chunks(chunks: list[Path]) -> dict[str, Any]:
    recording_dir = _recording_dir_from_chunks(chunks)
    if recording_dir is None:
        return {}
    manifest_path = recording_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _projector_source_video_for_chunks(chunks: list[Path]) -> Path | str | None:
    manifest = _read_recording_manifest_for_chunks(chunks)
    video_url = str(manifest.get("video_url") or "").strip()
    if not video_url:
        return None
    basename = Path(urlparse(video_url).path).name
    if not basename:
        return None
    candidates = [
        REPO_ROOT / "tmp" / "cat-tv-youtube-trimmed" / basename,
        REPO_ROOT / "tmp" / "cat-tv-youtube-seeds" / basename,
        Path("/config/www/cat-tv") / basename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    parsed = urlparse(video_url)
    if parsed.scheme in {"http", "https"}:
        return video_url
    return None


def _source_frame_at(source_video: Path | str, seconds: float) -> Any:
    try:
        import cv2
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency check.
        raise RuntimeError("opencv-python is required for projector-source subtraction") from exc

    key = str(source_video.resolve()) if isinstance(source_video, Path) else str(source_video)
    cached = _SOURCE_VIDEO_CACHE.get(key)
    if cached is None:
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open projector source video {source_video}")
        duration_ms = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS)) * 1000.0
        _SOURCE_VIDEO_CACHE[key] = (cap, duration_ms)
    else:
        cap, duration_ms = cached
    seek_ms = max(0.0, seconds * 1000.0)
    if duration_ms > 1.0:
        seek_ms %= duration_ms
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read projector source frame at {seconds:.3f}s from {source_video}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _frame_time_seconds(
    chunk_index: int,
    frame_index: int,
    *,
    sample_fps: int,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
) -> float:
    return chunk_index * segment_seconds + (frame_index - 1) / sample_fps


def _source_residual_baseline(
    frame_paths: list[Path],
    *,
    source_video: Path | None = None,
    source_frames: dict[tuple[int, int], Path] | None = None,
    sample_fps: int,
    limit: int = 80,
) -> Any | None:
    import numpy as np
    from PIL import Image

    residuals: list[np.ndarray] = []
    stride = max(1, len(frame_paths) // max(1, limit))
    for path in frame_paths[::stride]:
        match = re.match(r"chunk_(\d+)_(\d+)\.jpg", path.name)
        if not match:
            continue
        chunk_index = int(match.group(1))
        frame_index = int(match.group(2))
        source_path = source_frames.get((chunk_index, frame_index)) if source_frames else None
        if source_path is not None and source_path.exists():
            source_frame = Image.open(source_path).convert("RGB")
        elif source_video is not None:
            source_frame = _source_frame_at(
                source_video,
                _frame_time_seconds(chunk_index, frame_index, sample_fps=sample_fps),
            )
        else:
            continue
        residual, _warped = source_subtracted_residual(Image.open(path).convert("RGB"), source_frame=source_frame)
        residuals.append(residual)
    if not residuals:
        return None
    return np.percentile(np.stack(residuals, axis=0), 75, axis=0).astype(np.float32)


def _possible_cat_update_mask(width: int, height: int) -> Any:
    import numpy as np

    mask = np.ones((height, width), dtype=bool)
    # Sher can sit on the lower room/floor strip for minutes; never learn this
    # into the adaptive background from active-session frames.
    mask[int(height * 0.58) :, :] = False
    # Left entry and stool/center-bottom corridor also saw repeated cat presence.
    mask[int(height * 0.42) :, : int(width * 0.62)] = False
    # Right furniture/table region produced false cat boxes; don't let it adapt
    # from active frames either.
    mask[int(height * 0.38) :, int(width * 0.70) :] = False
    return mask


def _update_room_background(
    previous: Any,
    frame: Any,
    *,
    update_mask: Any,
    alpha: float = 0.08,
) -> Any:
    import numpy as np
    from PIL import Image

    previous_arr = np.asarray(previous.convert("L").resize(frame.size), dtype=np.float32)
    frame_arr = np.asarray(frame.convert("L"), dtype=np.float32)
    mask = np.asarray(update_mask, dtype=bool)
    if mask.shape != frame_arr.shape:
        raise ValueError(f"background update mask shape {mask.shape} does not match frame shape {frame_arr.shape}")
    updated = previous_arr.copy()
    updated[mask] = previous_arr[mask] * (1.0 - alpha) + frame_arr[mask] * alpha
    return Image.fromarray(np.clip(updated, 0, 255).astype(np.uint8), mode="L")


def _room_background_for_frames(chunks: list[Path], frame_paths: list[Path]) -> Any | None:
    import numpy as np
    from PIL import Image

    recording_dir = _recording_dir_from_chunks(chunks)
    if recording_dir is None:
        return None
    manifest = _read_recording_manifest_for_chunks(chunks)
    candidates = []
    snapshot_path = str(manifest.get("snapshot_path") or "").strip()
    if snapshot_path:
        candidates.append(Path(snapshot_path).expanduser())
    candidates.append(recording_dir / "startup_projector_camera.jpg")

    background = None
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            background = Image.open(candidate).convert("RGB")
            break
    if background is None:
        return None

    sampled_frames = frame_paths[:: max(1, len(frame_paths) // 60)]
    path_stack: list[np.ndarray] = []
    update_mask = None
    for path in sampled_frames:
        frame = Image.open(path).convert("RGB")
        if frame.size != background.size:
            background = background.resize(frame.size)
        update_mask = _possible_cat_update_mask(frame.width, frame.height)
        path_stack.append(np.asarray(frame.convert("L"), dtype=np.uint8))
        background = _update_room_background(background, frame, update_mask=update_mask)
    if path_stack and update_mask is not None:
        # The startup snapshot can already contain Sher. For the known cat path,
        # use a bright temporal quantile from the current session instead of
        # learning a dark stationary cat into the room background.
        background_arr = np.asarray(background.convert("L"), dtype=np.uint8).copy()
        path_background = np.percentile(np.stack(path_stack, axis=0), 90, axis=0).astype(np.uint8)
        keep_background = np.asarray(update_mask, dtype=bool)
        background_arr[~keep_background] = path_background[~keep_background]
        background = Image.fromarray(background_arr, mode="L")
    return background


def _extract_source_video_frames(
    source_video: Path | str,
    chunks: list[Path],
    output_dir: Path,
    *,
    fps: int,
) -> dict[tuple[int, int], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[tuple[int, int], Path] = {}
    for chunk in chunks:
        chunk_index = _chunk_index(chunk)
        if chunk_index is None:
            continue
        pattern = output_dir / f"source_{chunk_index:04d}_%05d.jpg"
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{chunk_index * DEFAULT_SEGMENT_SECONDS:.3f}",
                "-i",
                str(source_video),
                "-t",
                f"{DEFAULT_SEGMENT_SECONDS:.3f}",
                "-vf",
                f"fps={fps}",
                str(pattern),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to extract source frames from {source_video}: {result.stderr[-400:]}")
        for path in sorted(output_dir.glob(f"source_{chunk_index:04d}_*.jpg")):
            match = re.match(r"source_(\d+)_(\d+)\.jpg", path.name)
            if match:
                extracted[(int(match.group(1)), int(match.group(2)))] = path
    return extracted


def _sample_from_candidate(
    *,
    image: Any,
    source: str,
    candidate: Candidate,
    cat_probability: float,
    detector_model_path: str,
    homography: Any,
    floor_y_mm: float,
    jump_threshold_cm: float,
) -> Any:
    x, y, width, height = candidate.bbox_xywh
    bbox_xyxy = (
        int(round(x)),
        int(round(y)),
        int(round(x + width)),
        int(round(y + height)),
    )
    wall_x_mm, wall_y_mm = homography.project(candidate.top_x_px, candidate.top_y_px)
    height_mm = max(0.0, wall_y_mm - floor_y_mm)
    height_cm = height_mm / 10.0
    return live_jump.LiveJumpSample(
        timestamp=live_jump._now_iso(),  # noqa: SLF001
        source=source,
        candidate_found=True,
        top_x_px=round(candidate.top_x_px, 2),
        top_y_px=round(candidate.top_y_px, 2),
        wall_x_mm=round(wall_x_mm, 1),
        top_height_mm=round(height_mm, 1),
        top_height_cm=round(height_cm, 1),
        component_area_px=candidate.area_px,
        component_bbox_xyxy=bbox_xyxy,
        triggered=height_cm >= jump_threshold_cm,
        image_path=str(image.filename) if getattr(image, "filename", None) else None,
        detector_cat_probability=round(cat_probability, 4),
        detector_model_path=detector_model_path,
    )


def _detector_context_payload(
    *,
    detector_loaded_metadata: dict[str, Any] | None,
    backend: str,
    path: str,
    probability: float | None = None,
    threshold: float = DEFAULT_TELEGRAM_DETECTOR_MIN_PROBABILITY,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "path": path,
        "model_created_at": (detector_loaded_metadata or {}).get("created_at"),
        "model_path": (detector_loaded_metadata or {}).get("model_path"),
        "metadata_path": (detector_loaded_metadata or {}).get("metadata_path"),
        "cat_probability": round(float(probability), 4) if probability is not None else None,
        "min_probability": threshold,
    }


def _sample_source_subtracted_frame(
    image: Any,
    *,
    source: str,
    source_frame: Any,
    room_background: Any | None,
    residual_baseline: Any | None,
    homography: Any,
    floor_y_mm: float,
    jump_threshold_cm: float,
    detector_loaded_model: Any | None,
    detector_loaded_metadata: dict[str, Any] | None,
    min_probability: float,
) -> Any | None:
    if detector_loaded_model is None or detector_loaded_metadata is None:
        return None
    candidates = detect_source_subtracted_candidate_components(
        image,
        source_frame=source_frame,
        room_background=room_background,
        residual_baseline=residual_baseline,
    )
    predictions = sorted(
        score_candidates(image, model=detector_loaded_model, metadata=detector_loaded_metadata, candidates=candidates),
        key=lambda item: item.cat_probability,
        reverse=True,
    )
    if not predictions or predictions[0].candidate is None or predictions[0].cat_probability < min_probability:
        return None
    return _sample_from_candidate(
        image=image,
        source=source,
        candidate=predictions[0].candidate,
        cat_probability=predictions[0].cat_probability,
        detector_model_path=predictions[0].model_path,
        homography=homography,
        floor_y_mm=floor_y_mm,
        jump_threshold_cm=jump_threshold_cm,
    )


def _sample_chunk_offset(
    sample: Any,
    *,
    sample_fps: float,
) -> tuple[int, float] | None:
    frame_label = Path(str(getattr(sample, "source", ""))).name
    match = re.match(r"chunk_(\d+)_(\d+)\.jpg", frame_label)
    if not match:
        return None
    chunk_index = int(match.group(1))
    frame_index = int(match.group(2))
    return chunk_index, max(0.0, (frame_index - 1) / sample_fps if sample_fps else 0.0)


def _sample_matches_alert_record(
    sample: Any,
    max_record: dict[str, Any] | None,
    *,
    sample_fps: float,
) -> bool:
    if not isinstance(max_record, dict):
        return False
    located = _sample_chunk_offset(sample, sample_fps=sample_fps)
    if located is None:
        return False
    chunk_index, offset_seconds = located
    try:
        expected_chunk = int(max_record["chunk"])
    except (KeyError, TypeError, ValueError):
        return False
    if chunk_index != expected_chunk:
        return False
    try:
        expected_offset = float(max_record.get("offset_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        expected_offset = 0.0
    tolerance = max(0.08, 1.5 / max(sample_fps, 1.0))
    if abs(offset_seconds - expected_offset) > tolerance:
        return False
    expected_backend = str(max_record.get("detector_backend") or max_record.get("max_source") or "")
    expected_model = str(max_record.get("model_path") or max_record.get("detector_model_id") or "")
    expected_reason = str(max_record.get("reason") or "")
    expects_yolo_seg = (
        "ultralytics_yolo_segmentation" in expected_backend
        or "yolo_segmentation" in expected_reason
        or expected_model.endswith(".pt")
    )
    if expects_yolo_seg:
        sample_model = str(getattr(sample, "detector_model_path", "") or "")
        return sample_model.endswith(".pt") and (
            not expected_model or Path(sample_model).name == Path(expected_model).name or sample_model == expected_model
        )
    return True


def _alert_record_from_sample(sample: Any, *, sample_fps: float, reason: str) -> dict[str, Any] | None:
    located = _sample_chunk_offset(sample, sample_fps=sample_fps)
    if located is None:
        return None
    chunk_index, offset_seconds = located
    if getattr(sample, "top_height_cm", None) is None:
        return None
    return {
        "chunk": chunk_index,
        "offset_seconds": round(offset_seconds, 3),
        "score": round(float(sample.top_height_cm) * 1000.0 + float(sample.component_area_px or 0), 1),
        "reason": reason,
        "height_cm": round(float(sample.top_height_cm), 1),
        "wall_x_mm": getattr(sample, "wall_x_mm", None),
        "top_px": [getattr(sample, "top_x_px", None), getattr(sample, "top_y_px", None)],
        "bbox": list(sample.component_bbox_xyxy) if getattr(sample, "component_bbox_xyxy", None) is not None else None,
        "area": getattr(sample, "component_area_px", None),
        "cat_probability": getattr(sample, "detector_cat_probability", None),
        "model_path": getattr(sample, "detector_model_path", None),
    }


def _score_jump_highlight_records(
    chunks: list[Path],
    *,
    sample_fps: int = DEFAULT_TELEGRAM_JUMP_SCAN_FPS,
    min_jump_height_cm: float = DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
) -> list[dict[str, Any]]:
    """Find calibrated high-jump frames; no jump means no Telegram clip."""

    from PIL import Image

    homography, floor_y_mm = live_jump._read_calibration(live_jump.DEFAULT_CALIBRATION)  # noqa: SLF001
    detector_model = DEFAULT_DETECTOR_MODEL_PATH if default_model_available() else None
    detector_loaded_model = None
    detector_loaded_metadata = None
    if detector_model is not None:
        try:
            detector_loaded_model, detector_loaded_metadata = load_model(
                DEFAULT_DETECTOR_MODEL_PATH,
                DEFAULT_DETECTOR_METADATA_PATH,
            )
        except Exception:
            detector_model = None
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cat-projector-jump-scan-") as tmpdir:
        frame_paths = _extract_video_frames(chunks, Path(tmpdir), fps=sample_fps)
        room_background = _room_background_for_frames(chunks, frame_paths)
        source_video = _projector_source_video_for_chunks(chunks)
        source_frames = (
            _extract_source_video_frames(source_video, chunks, Path(tmpdir) / "source", fps=sample_fps)
            if source_video is not None
            else {}
        )
        residual_baseline = (
            _source_residual_baseline(
                frame_paths,
                source_video=source_video,
                source_frames=source_frames,
                sample_fps=sample_fps,
            )
            if source_video is not None and detector_loaded_model is not None and detector_loaded_metadata is not None
            else None
        )
        for path in frame_paths:
            match = re.match(r"chunk_(\d+)_(\d+)\.jpg", path.name)
            if not match:
                continue
            chunk_index = int(match.group(1))
            frame_index = int(match.group(2))
            offset_seconds = (frame_index - 1) / sample_fps
            image = Image.open(path).convert("RGB")
            source_sample = None
            if source_video is not None:
                source_path = source_frames.get((chunk_index, frame_index))
                source_frame = (
                    Image.open(source_path).convert("RGB")
                    if source_path is not None and source_path.exists()
                    else _source_frame_at(
                        source_video,
                        _frame_time_seconds(chunk_index, frame_index, sample_fps=sample_fps),
                    )
                )
                source_sample = _sample_source_subtracted_frame(
                    image,
                    source=str(path),
                    source_frame=source_frame,
                    room_background=room_background,
                    residual_baseline=residual_baseline,
                    homography=homography,
                    floor_y_mm=floor_y_mm,
                    jump_threshold_cm=min_jump_height_cm,
                    detector_loaded_model=detector_loaded_model,
                    detector_loaded_metadata=detector_loaded_metadata,
                    min_probability=DEFAULT_TELEGRAM_DETECTOR_MIN_PROBABILITY,
                )
            if (
                source_sample is not None
                and source_sample.candidate_found
                and source_sample.top_height_cm is not None
                and source_sample.top_height_cm >= min_jump_height_cm
            ):
                records.append(
                    {
                        "chunk": chunk_index,
                        "offset_seconds": round(offset_seconds, 3),
                        "score": round(
                            source_sample.top_height_cm * 1000.0 + float(source_sample.component_area_px or 0),
                            1,
                        ),
                        "reason": "source_subtracted_calibrated_high_jump",
                        "height_cm": source_sample.top_height_cm,
                        "wall_x_mm": source_sample.wall_x_mm,
                        "top_px": [source_sample.top_x_px, source_sample.top_y_px],
                        "bbox": list(source_sample.component_bbox_xyxy)
                        if source_sample.component_bbox_xyxy is not None
                        else None,
                        "area": source_sample.component_area_px,
                        "cat_probability": source_sample.detector_cat_probability,
                        "model_path": source_sample.detector_model_path,
                        "source_video": str(source_video),
                    }
                )
                continue
            sample, _component = live_jump.sample_image(
                image,
                source=str(path),
                homography=homography,
                floor_y_mm=floor_y_mm,
                roi=DEFAULT_TELEGRAM_JUMP_ROI,
                threshold=DEFAULT_TELEGRAM_JUMP_DARK_THRESHOLD,
                scale=DEFAULT_TELEGRAM_JUMP_SCALE,
                min_area_px=DEFAULT_TELEGRAM_JUMP_MIN_AREA_PX,
                min_height_px=DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_PX,
                max_width_px=DEFAULT_TELEGRAM_JUMP_MAX_WIDTH_PX,
                jump_threshold_cm=min_jump_height_cm,
                detector_model=detector_model,
                detector_metadata=DEFAULT_DETECTOR_METADATA_PATH,
                detector_loaded_model=detector_loaded_model,
                detector_loaded_metadata=detector_loaded_metadata,
                detector_min_probability=DEFAULT_TELEGRAM_DETECTOR_MIN_PROBABILITY,
            )
            if not sample.candidate_found or sample.top_height_cm is None:
                continue
            if sample.top_height_cm < min_jump_height_cm:
                continue
            records.append(
                {
                    "chunk": chunk_index,
                    "offset_seconds": round(offset_seconds, 3),
                    "score": round(sample.top_height_cm * 1000.0 + float(sample.component_area_px or 0), 1),
                    "reason": "calibrated_high_jump",
                    "height_cm": sample.top_height_cm,
                    "wall_x_mm": sample.wall_x_mm,
                    "top_px": [sample.top_x_px, sample.top_y_px],
                    "bbox": list(sample.component_bbox_xyxy) if sample.component_bbox_xyxy is not None else None,
                    "area": sample.component_area_px,
                    "cat_probability": sample.detector_cat_probability,
                    "model_path": sample.detector_model_path,
                }
            )
    return sorted(records, key=lambda item: float(item.get("score") or 0), reverse=True)


def _sample_render_frame(
    image: Any,
    *,
    source: str,
    source_frame: Any | None = None,
    room_background: Any | None = None,
    residual_baseline: Any | None = None,
    homography: Any,
    floor_y_mm: float,
    min_jump_height_cm: float,
    detector_loaded_model: Any | None,
    detector_loaded_metadata: dict[str, Any] | None,
) -> Any:
    detector_model = DEFAULT_DETECTOR_MODEL_PATH if default_model_available() else None
    source_sample = None
    if source_frame is not None:
        source_sample = _sample_source_subtracted_frame(
            image,
            source=source,
            source_frame=source_frame,
            room_background=room_background,
            residual_baseline=residual_baseline,
            homography=homography,
            floor_y_mm=floor_y_mm,
            jump_threshold_cm=min_jump_height_cm,
            detector_loaded_model=detector_loaded_model,
            detector_loaded_metadata=detector_loaded_metadata,
            min_probability=DEFAULT_TELEGRAM_DETECTOR_MIN_PROBABILITY,
        )
        if source_sample is not None and source_sample.candidate_found:
            return source_sample
    if detector_model is not None and detector_loaded_model is not None and detector_loaded_metadata is not None:
        sample, _component = live_jump.sample_image(
            image,
            source=source,
            homography=homography,
            floor_y_mm=floor_y_mm,
            roi=DEFAULT_TELEGRAM_JUMP_ROI,
            threshold=DEFAULT_TELEGRAM_JUMP_DARK_THRESHOLD,
            scale=DEFAULT_TELEGRAM_JUMP_SCALE,
            min_area_px=DEFAULT_TELEGRAM_RENDER_MIN_AREA_PX,
            min_height_px=DEFAULT_TELEGRAM_RENDER_MIN_HEIGHT_PX,
            max_width_px=DEFAULT_TELEGRAM_RENDER_MAX_WIDTH_PX,
            jump_threshold_cm=min_jump_height_cm,
            detector_model=detector_model,
            detector_metadata=DEFAULT_DETECTOR_METADATA_PATH,
            detector_loaded_model=detector_loaded_model,
            detector_loaded_metadata=detector_loaded_metadata,
            detector_min_probability=DEFAULT_TELEGRAM_DETECTOR_MIN_PROBABILITY,
        )
        if sample.candidate_found:
            return sample
        return live_jump.LiveJumpSample(
            timestamp=sample.timestamp,
            source=source,
            candidate_found=False,
            detector_cat_probability=None,
        )

    if source_sample is not None:
        return source_sample

    sample, _component = live_jump.sample_image(
        image,
        source=source,
        homography=homography,
        floor_y_mm=floor_y_mm,
        roi=DEFAULT_TELEGRAM_JUMP_ROI,
        threshold=DEFAULT_TELEGRAM_JUMP_DARK_THRESHOLD,
        scale=DEFAULT_TELEGRAM_JUMP_SCALE,
        min_area_px=DEFAULT_TELEGRAM_RENDER_MIN_AREA_PX,
        min_height_px=DEFAULT_TELEGRAM_RENDER_MIN_HEIGHT_PX,
        max_width_px=DEFAULT_TELEGRAM_RENDER_MAX_WIDTH_PX,
        jump_threshold_cm=min_jump_height_cm,
        detector_model=None,
    )
    return sample


def _select_jump_highlight_clip_window(
    chunks: list[Path],
    *,
    window_chunks: int,
    sample_fps: int = DEFAULT_TELEGRAM_JUMP_SCAN_FPS,
    min_jump_height_cm: float = DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
) -> JumpHighlight | None:
    chunks_by_index = {index: chunk for chunk in chunks if (index := _chunk_index(chunk)) is not None}
    records = _score_jump_highlight_records(
        chunks,
        sample_fps=sample_fps,
        min_jump_height_cm=min_jump_height_cm,
    )
    if not records:
        return None
    windows = _motion_records_to_clip_windows(
        chunks_by_index,
        records,
        window_chunks=window_chunks,
        max_clips=1,
        separation_chunks=0,
    )
    if not windows:
        return None
    best_record = records[0]
    return JumpHighlight(
        chunks=windows[0],
        max_height_cm=float(best_record["height_cm"]),
        max_record=best_record,
        candidate_count=len(records),
    )


def _resolve_jump_heights_data_path(jump_heights_path: Path) -> Path:
    path = jump_heights_path.expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "videos" not in payload and isinstance(payload.get("jump_heights"), str):
        nested_path = Path(str(payload["jump_heights"])).expanduser()
        if not nested_path.is_absolute():
            nested_path = path.parent / nested_path
        return nested_path
    return path


def _load_jump_heights_payload(jump_heights_path: Path) -> Any:
    return json.loads(_resolve_jump_heights_data_path(jump_heights_path).read_text(encoding="utf-8"))


def _load_jump_height_videos(jump_heights_path: Path) -> list[dict[str, Any]]:
    payload = _load_jump_heights_payload(jump_heights_path)
    videos = payload.get("videos") if isinstance(payload, dict) else payload
    if not isinstance(videos, list):
        raise ValueError(f"jump heights file has no videos list: {jump_heights_path}")
    return [item for item in videos if isinstance(item, dict)]


def _calibration_path_for_jump_heights(jump_heights_path: Path) -> Path:
    payload = _load_jump_heights_payload(jump_heights_path)
    raw_path = payload.get("calibration_path") if isinstance(payload, dict) else None
    if isinstance(raw_path, str) and raw_path.strip():
        calibration_path = Path(raw_path).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = jump_heights_path.expanduser().parent / calibration_path
        return calibration_path
    return live_jump.DEFAULT_CALIBRATION.expanduser()


def _source_integrity_with_reason(integrity: SourceIntegrity, *, reason: str) -> SourceIntegrity:
    return SourceIntegrity(
        status="missing",
        reason=reason,
        frame_path=integrity.frame_path,
        recording_dir=integrity.recording_dir,
        chunk_path=integrity.chunk_path,
        chunk_index=integrity.chunk_index,
        frame_index=integrity.frame_index,
        offset_seconds=integrity.offset_seconds,
    )


def _select_publishable_rescore_jump(
    jump_heights_path: Path,
    *,
    min_source_fps: float | None = DEFAULT_PUBLISHABLE_MIN_SOURCE_FPS,
) -> tuple[RescoreJumpCandidate | None, list[RescoreJumpCandidate]]:
    candidates, suspicious = _select_publishable_rescore_jumps(
        jump_heights_path,
        min_source_fps=min_source_fps,
        min_jump_count=1,
    )
    return (candidates[0] if candidates else None), suspicious


def _select_publishable_rescore_jumps(
    jump_heights_path: Path,
    *,
    min_source_fps: float | None = DEFAULT_PUBLISHABLE_MIN_SOURCE_FPS,
    min_jump_count: int = DEFAULT_PUBLISHABLE_MIN_JUMP_COUNT,
) -> tuple[list[RescoreJumpCandidate], list[RescoreJumpCandidate]]:
    videos = sorted(
        _load_jump_height_videos(jump_heights_path),
        key=lambda item: float(item.get("max_jump_height_cm") or 0.0),
        reverse=True,
    )
    selected: list[RescoreJumpCandidate] = []
    suspicious: list[RescoreJumpCandidate] = []
    selected_times: list[tuple[Path, float]] = []
    for video in videos:
        if video.get("max_jump_height_cm") in (None, ""):
            continue
        integrity = _source_integrity_for_rescore_video(video)
        candidate = RescoreJumpCandidate(video=video, integrity=integrity)
        if integrity.status == "clean" and min_source_fps is not None and min_source_fps > 0.0:
            source_fps = _chunk_effective_frame_rate(integrity.chunk_path) if integrity.chunk_path is not None else 0.0
            if source_fps < float(min_source_fps):
                suspicious.append(
                    RescoreJumpCandidate(
                        video=video,
                        integrity=_source_integrity_with_reason(
                            integrity,
                            reason=f"source_fps_below_publishable_min:{source_fps:.3f}<{float(min_source_fps):.3f}",
                        ),
                    )
                )
                continue
        if integrity.status == "clean":
            source_time = (float(integrity.chunk_index or 0) * DEFAULT_SEGMENT_SECONDS) + float(
                integrity.offset_seconds or 0.0
            )
            recording_dir = integrity.recording_dir.expanduser() if integrity.recording_dir is not None else None
            if recording_dir is not None and any(
                prior_dir == recording_dir and abs(source_time - prior_time) <= 3.0
                for prior_dir, prior_time in selected_times
            ):
                suspicious.append(
                    RescoreJumpCandidate(
                        video=video,
                        integrity=_source_integrity_with_reason(
                            integrity, reason="same_jump_as_selected_publish_event"
                        ),
                    )
                )
                continue
            selected.append(candidate)
            if recording_dir is not None:
                selected_times.append((recording_dir, source_time))
            if len(selected) >= max(1, min_jump_count):
                return selected, suspicious
            continue
        suspicious.append(candidate)
    return selected, suspicious


def _probe_raw_prefix_for_path(raw_path: str | Path | None) -> Path | None:
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    if path.parent.name == "raw":
        return path.parent
    return None


def _select_same_source_probe_rescore_jumps(
    seed_video: dict[str, Any],
    rows_by_raw_path: dict[str, dict[str, Any]],
    *,
    min_source_fps: float | None,
    min_jump_count: int,
) -> list[RescoreJumpCandidate]:
    recording_raw = seed_video.get("recording_dir")
    recording_dir = Path(str(recording_raw)).expanduser() if recording_raw else None
    probe_raw_prefix = _probe_raw_prefix_for_path(seed_video.get("max_frame_path"))
    if recording_dir is None or probe_raw_prefix is None:
        return []
    selected: list[RescoreJumpCandidate] = []
    selected_times: list[float] = []
    rows: list[tuple[float, str, dict[str, Any]]] = []
    prefix_text = str(probe_raw_prefix)
    for raw_path, row in rows_by_raw_path.items():
        path = Path(raw_path).expanduser()
        try:
            path.relative_to(probe_raw_prefix)
        except ValueError:
            continue
        height = row.get("best_top_height_cm")
        try:
            height_cm = float(height)
        except (TypeError, ValueError):
            continue
        if height_cm < 45.0:
            continue
        rows.append((height_cm, raw_path, row))
    for height_cm, raw_path, row in sorted(rows, key=lambda item: item[0], reverse=True):
        raw_frame_path = Path(raw_path).expanduser()
        parts = _frame_label_parts(raw_frame_path)
        if parts is None:
            continue
        chunk_index, frame_index = parts
        chunk_path = recording_dir / f"chunk_{chunk_index:04d}.mp4"
        if not chunk_path.exists():
            continue
        if min_source_fps is not None and min_source_fps > 0.0:
            source_fps = _chunk_effective_frame_rate(chunk_path)
            if source_fps < float(min_source_fps):
                continue
        raw_frames = _probe_raw_frame_timeline(probe_raw_prefix, chunk_index)
        raw_fps = len(raw_frames) / float(DEFAULT_SEGMENT_SECONDS) if raw_frames else 0.0
        frame_count = len(raw_frames) if raw_frames else _ffprobe_frame_count(chunk_path)
        if frame_count and frame_index > frame_count:
            continue
        offset_seconds = (
            frame_index / raw_fps if raw_fps > 0.0 else _frame_offset_seconds_for_index(chunk_path, frame_index)
        )
        source_time = float(chunk_index) * DEFAULT_SEGMENT_SECONDS + offset_seconds
        if any(abs(source_time - prior_time) <= 3.0 for prior_time in selected_times):
            continue
        video = _merge_probe_row_into_rescore_video(seed_video, row)
        video.update(
            {
                "max_frame_path": raw_path,
                "max_frame_index": frame_index,
                "max_jump_height_cm": height_cm,
                "probe_raw_prefix": prefix_text,
                "probe_row_raw_path": raw_path,
                "recording_dir": str(recording_dir),
            }
        )
        integrity = SourceIntegrity(
            status="clean",
            reason="probe_row_frame_key_matches_recording_chunk",
            frame_path=raw_frame_path,
            recording_dir=recording_dir,
            chunk_path=chunk_path,
            chunk_index=chunk_index,
            frame_index=frame_index,
            offset_seconds=offset_seconds,
        )
        selected.append(RescoreJumpCandidate(video=video, integrity=integrity))
        selected_times.append(source_time)
        if len(selected) >= max(1, min_jump_count):
            return selected
    return selected


def _load_probe_rows_by_raw_path(probe_rows_path: Path) -> dict[str, dict[str, Any]]:
    if not probe_rows_path.exists():
        return {}
    payload = json.loads(probe_rows_path.expanduser().read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    by_raw_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("raw_path")
        if isinstance(raw_path, str) and raw_path:
            by_raw_path[str(Path(raw_path).expanduser())] = row
    return by_raw_path


def _load_probe_rows_by_frame_key(
    rows_by_raw_path: dict[str, dict[str, Any]],
    *,
    recording_dir: Path | None,
    external_raw_prefix: Path | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    recording_resolved: Path | None = None
    if recording_dir is not None:
        try:
            recording_resolved = recording_dir.expanduser().resolve()
        except OSError:
            recording_resolved = recording_dir.expanduser()
    for raw_path, row in rows_by_raw_path.items():
        path = Path(raw_path).expanduser()
        include = False
        if recording_resolved is not None:
            try:
                path.resolve().relative_to(recording_resolved)
                include = True
            except (OSError, ValueError):
                include = False
        if not include and external_raw_prefix is not None:
            try:
                path.relative_to(external_raw_prefix)
                include = True
            except ValueError:
                include = False
        if not include:
            continue
        key = _frame_label_parts(path)
        if key is not None:
            by_key[key] = row
    return by_key


def _merge_probe_row_into_rescore_video(video: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(video)
    if not row:
        return merged
    field_pairs = {
        "best_bbox": "max_bbox",
        "best_probability": "max_probability",
        "detector_backend": "max_source",
        "detector_model_id": "max_model_path",
        "model_created_at": "model_created_at",
        "best_mask_polygon": "best_mask_polygon",
        "best_measurement_point": "best_measurement_point",
    }
    for source_key, target_key in field_pairs.items():
        if row.get(source_key) is not None:
            merged[target_key] = row[source_key]
    if row.get("best_top_height_cm") is not None:
        merged["max_jump_height_cm"] = row["best_top_height_cm"]
    if row.get("measurement_source") is not None:
        merged["measurement_source"] = row["measurement_source"]
    if row.get("raw_path") is not None:
        merged["max_frame_path"] = row["raw_path"]
    merged["probe_row_raw_path"] = row.get("raw_path")
    return merged


def _rescore_frame_video_from_probe(video: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    if row is not None:
        return _merge_probe_row_into_rescore_video(video, row)
    empty = dict(video)
    for key in (
        "max_bbox",
        "max_probability",
        "max_jump_height_cm",
        "best_mask_polygon",
        "best_measurement_point",
        "probe_row_raw_path",
    ):
        empty.pop(key, None)
    return empty


def _rescore_source_frame_timeline(integrity: SourceIntegrity, frames_dir: Path) -> list[Path]:
    if integrity.chunk_path is None or integrity.chunk_index is None:
        return []
    frames = _extract_exact_video_frames([integrity.chunk_path], frames_dir)
    return [
        path for path in frames if _frame_label_parts(path) and _frame_label_parts(path)[0] == integrity.chunk_index
    ]


def _probe_raw_frame_timeline(raw_prefix: Path, chunk_index: int) -> list[Path]:
    return [
        path
        for path in sorted(raw_prefix.expanduser().glob(f"chunk_{chunk_index:04d}_*.jpg"))
        if _frame_label_parts(path) is not None
    ]


def _publishable_segment_for_candidate(
    candidate: RescoreJumpCandidate,
    frames_dir: Path,
    *,
    pre_seconds: float,
    post_seconds: float,
) -> PublishableJumpSegment | None:
    integrity = candidate.integrity
    if (
        integrity.chunk_path is None
        or integrity.chunk_index is None
        or integrity.frame_index is None
        or integrity.offset_seconds is None
    ):
        return None
    probe_raw_prefix = _probe_raw_prefix_for_path(
        candidate.video.get("probe_row_raw_path")
    ) or _probe_raw_prefix_for_path(candidate.video.get("max_frame_path"))
    raw_frames = (
        _probe_raw_frame_timeline(probe_raw_prefix, integrity.chunk_index) if probe_raw_prefix is not None else []
    )
    if raw_frames:
        frames = raw_frames
        source_fps = len(raw_frames) / float(DEFAULT_SEGMENT_SECONDS)
        chunk_duration = float(DEFAULT_SEGMENT_SECONDS)
        frame_source = "probe_raw_frames"
    else:
        source_fps = _chunk_effective_frame_rate(integrity.chunk_path)
        if source_fps <= 0.0:
            return None
        chunk_duration = _ffprobe_duration_seconds(integrity.chunk_path) or 0.0
        frames = _rescore_source_frame_timeline(integrity, frames_dir)
        frame_source = "recording_chunk"
    if not frames:
        return None
    start_s = max(0.0, float(integrity.offset_seconds) - max(0.0, pre_seconds))
    end_s = min(chunk_duration, float(integrity.offset_seconds) + max(0.0, post_seconds))
    selected: list[Path] = []
    active_frame_index = 0
    for frame_path in frames:
        parts = _frame_label_parts(frame_path)
        if parts is None:
            continue
        _chunk_index, frame_index = parts
        frame_time_s = frame_index / source_fps
        if start_s <= frame_time_s <= end_s:
            if parts == (integrity.chunk_index, integrity.frame_index):
                active_frame_index = len(selected)
            selected.append(frame_path)
    if not selected:
        return None
    if not any(_frame_label_parts(path) == (integrity.chunk_index, integrity.frame_index) for path in selected):
        active_frame_index = min(
            range(len(selected)), key=lambda index: abs(_frame_label_parts(selected[index])[1] - integrity.frame_index)
        )  # type: ignore[index]
    return PublishableJumpSegment(
        candidate=candidate,
        frames=tuple(selected),
        active_frame_index=active_frame_index,
        source_fps=source_fps,
        source_duration_seconds=chunk_duration,
        pre_available_seconds=max(0.0, float(integrity.offset_seconds) - start_s),
        post_available_seconds=max(0.0, end_s - float(integrity.offset_seconds)),
        probe_raw_prefix=probe_raw_prefix,
        frame_source=frame_source,
    )


def _probe_row_for_frame_path(
    frame_path: Path,
    rows_by_raw_path: dict[str, dict[str, Any]],
    rows_by_frame_key: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    row = rows_by_raw_path.get(str(frame_path.expanduser()))
    if row is not None:
        return row
    if rows_by_frame_key:
        key = _frame_label_parts(frame_path)
        if key is not None:
            return rows_by_frame_key.get(key)
    return None


def _concat_file_line(path: Path) -> str:
    escaped = str(path).replace("'", "'\\''")
    return f"file '{escaped}'\n"


def _build_preview_clip(chunks: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = output_path.with_suffix(".concat.txt")
    concat_list.write_text("".join(_concat_file_line(path) for path in chunks), encoding="utf-8")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-an",
        "-vf",
        "fps=15,scale=960:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 4096:
        raise RuntimeError(f"failed to build Telegram preview {output_path}: {result.stderr[-400:]}")


def _sample_center(sample: Any) -> tuple[float, float] | None:
    if sample.component_bbox_xyxy is not None:
        left, top, right, bottom = sample.component_bbox_xyxy
        return ((left + right) / 2.0, (top + bottom) / 2.0)
    if sample.top_x_px is not None and sample.top_y_px is not None:
        return (float(sample.top_x_px), float(sample.top_y_px))
    return None


def _sample_is_inside_frame(sample: Any, *, width: int, height: int, margin_px: int = 18) -> bool:
    if sample.component_bbox_xyxy is not None:
        left, top, right, bottom = sample.component_bbox_xyxy
        return left > margin_px and top > margin_px and right < width - margin_px and bottom < height - margin_px
    if sample.top_x_px is None or sample.top_y_px is None:
        return False
    return margin_px < sample.top_x_px < width - margin_px and margin_px < sample.top_y_px < height - margin_px


def _queue_continuity_hard_cases(
    *,
    frame_paths: list[Path],
    samples: list[Any],
    queue_root: Path,
    max_missed_frames: int = DEFAULT_CONTINUITY_HARD_CASE_MAX_MISSED_FRAMES,
    max_reappear_distance_px: float = DEFAULT_CONTINUITY_HARD_CASE_MAX_REAPPEAR_DISTANCE_PX,
) -> list[dict[str, Any]]:
    """Save miss frames when Sher disappears and reappears in-place inside the frame."""

    if not frame_paths or len(frame_paths) != len(samples):
        return []
    try:
        with Image.open(frame_paths[0]) as first_frame:
            width, height = first_frame.size
    except Exception:
        return []
    package_dir = queue_root / f"continuity-hard-cases-{datetime.now().astimezone().strftime('%Y%m%d')}"
    frames_dir = package_dir / "frames"
    jsonl_path = package_dir / "queue.jsonl"
    queued: list[dict[str, Any]] = []
    last_seen_index: int | None = None
    last_seen_sample: Any | None = None
    missed_indices: list[int] = []

    for index, sample in enumerate(samples):
        if sample.candidate_found:
            if last_seen_index is not None and last_seen_sample is not None and missed_indices:
                previous_center = _sample_center(last_seen_sample)
                current_center = _sample_center(sample)
                missed_count = len(missed_indices)
                if (
                    previous_center is not None
                    and current_center is not None
                    and missed_count <= max_missed_frames
                    and _sample_is_inside_frame(last_seen_sample, width=width, height=height)
                    and _sample_is_inside_frame(sample, width=width, height=height)
                ):
                    distance = (
                        (current_center[0] - previous_center[0]) ** 2 + (current_center[1] - previous_center[1]) ** 2
                    ) ** 0.5
                    if distance <= max_reappear_distance_px:
                        frames_dir.mkdir(parents=True, exist_ok=True)
                        for missed_index in missed_indices:
                            source_frame = frame_paths[missed_index]
                            copied_name = (
                                f"{source_frame.stem}__between_{last_seen_index:06d}_{index:06d}"
                                f"__miss_{missed_index:06d}{source_frame.suffix}"
                            )
                            copied_path = frames_dir / copied_name
                            shutil.copy2(source_frame, copied_path)
                            payload = {
                                "kind": "cat_projector_continuity_hard_case_v1",
                                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                "frame_index": missed_index,
                                "frame_relpath": str(copied_path.relative_to(package_dir)),
                                "source_frame": str(source_frame),
                                "reason": "cat_missing_between_nearby_in-frame_detections",
                                "previous_frame_index": last_seen_index,
                                "next_frame_index": index,
                                "previous_bbox_xyxy": list(last_seen_sample.component_bbox_xyxy)
                                if last_seen_sample.component_bbox_xyxy is not None
                                else None,
                                "next_bbox_xyxy": list(sample.component_bbox_xyxy)
                                if sample.component_bbox_xyxy is not None
                                else None,
                                "previous_top_xy": [last_seen_sample.top_x_px, last_seen_sample.top_y_px],
                                "next_top_xy": [sample.top_x_px, sample.top_y_px],
                                "reappear_distance_px": round(distance, 1),
                                "label_status": "needs_human_bbox",
                            }
                            with jsonl_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                            queued.append(payload)
            last_seen_index = index
            last_seen_sample = sample
            missed_indices = []
            continue
        if last_seen_sample is not None:
            missed_indices.append(index)
            if len(missed_indices) > max_missed_frames:
                last_seen_index = None
                last_seen_sample = None
                missed_indices = []

    return queued


def _encode_frame_sequence(
    *,
    frame_dir: Path,
    output_path: Path,
    fps: int,
    vf: str,
    crf: int,
    metadata: dict[str, str] | None = None,
    input_fps: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_args: list[str] = []
    for key, value in (metadata or {}).items():
        metadata_args.extend(["-metadata", f"{key}={value}"])
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-framerate",
        f"{input_fps:.6g}" if input_fps is not None else str(fps),
        "-i",
        str(frame_dir / "frame_%06d.jpg"),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        *metadata_args,
        str(output_path),
    ]
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 4096:
        raise RuntimeError(f"failed to build annotated Telegram jump clip {output_path}: {result.stderr[-400:]}")


def _enhance_vertical_clip_for_social(
    source_path: Path,
    output_path: Path,
    *,
    height_cm: float | None,
    fps: int = DEFAULT_TELEGRAM_ENHANCED_FPS,
    title: str | None = None,
    comment: str | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_title = title or (f"Sher jump {height_cm:.1f} cm" if height_cm is not None else "Sher jump highlight")
    metadata_comment = comment or (
        "Cat projector jump highlight; source-frame measurement preserved, "
        "post-render enhanced with ffmpeg minterpolate and zscale."
    )
    vf = (
        f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
        "zscale=w=1080:h=1920:f=spline36,"
        "unsharp=5:5:0.75:3:3:0.35"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        f"title={metadata_title}",
        "-metadata",
        "artist=Sher",
        "-metadata",
        "genre=Cat projector jump",
        "-metadata",
        f"comment={metadata_comment}",
        "-metadata",
        f"description={metadata_comment}",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 4096:
        raise RuntimeError(f"failed to enhance Telegram vertical clip {output_path}: {result.stderr[-500:]}")
    return {
        "path": str(output_path),
        "source_path": str(source_path),
        "fps": fps,
        "resolution": "1080x1920",
        "video_filter": vf,
        "method": "ffmpeg_minterpolate_zscale_unsharp",
        "metadata": {
            "title": metadata_title,
            "artist": "Sher",
            "genre": "Cat projector jump",
            "comment": metadata_comment,
            "description": metadata_comment,
        },
    }


def _neural_upscale_clip_with_realesrgan(
    source_path: Path,
    output_path: Path,
    *,
    fps: int,
    width: int = 1080,
    height: int = 1920,
    time_budget_seconds: float = 45.0,
) -> dict[str, Any] | None:
    if not DEFAULT_REALESRGAN_BIN.exists() or not DEFAULT_REALESRGAN_MODEL_DIR.exists():
        return {
            "status": "unavailable",
            "backend": "realesrgan-ncnn-vulkan",
            "binary": str(DEFAULT_REALESRGAN_BIN),
            "model_dir": str(DEFAULT_REALESRGAN_MODEL_DIR),
            "fallback_reason": "binary_or_model_dir_missing",
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cat-projector-realesrgan-") as tmpdir:
        tmp_path = Path(tmpdir)
        low_dir = tmp_path / "low"
        up_dir = tmp_path / "up"
        low_dir.mkdir()
        up_dir.mkdir()
        low_w = max(2, width // 2)
        low_h = max(2, height // 2)
        extract = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source_path),
                "-vf",
                f"fps={fps},scale={low_w}:{low_h}",
                str(low_dir / "frame_%08d.png"),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if extract.returncode != 0:
            raise RuntimeError(f"failed to extract frames for Real-ESRGAN: {extract.stderr[-500:]}")
        low_frames = sorted(low_dir.glob("*.png"))
        if not low_frames:
            raise RuntimeError("Real-ESRGAN frame extraction produced no frames")
        try:
            upscale = subprocess.run(
                [
                    str(DEFAULT_REALESRGAN_BIN),
                    "-i",
                    str(low_dir),
                    "-o",
                    str(up_dir),
                    "-m",
                    str(DEFAULT_REALESRGAN_MODEL_DIR),
                    "-n",
                    "realesr-animevideov3",
                    "-s",
                    "2",
                    "-f",
                    "png",
                    "-j",
                    "2:2:2",
                ],
                check=False,
                text=True,
                capture_output=True,
                timeout=time_budget_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout_fallback",
                "backend": "realesrgan-ncnn-vulkan",
                "binary": str(DEFAULT_REALESRGAN_BIN),
                "model_dir": str(DEFAULT_REALESRGAN_MODEL_DIR),
                "model": "realesr-animevideov3",
                "scale": 2,
                "stage": "post_overlay",
                "input_frame_count": len(low_frames),
                "time_budget_seconds": time_budget_seconds,
                "fallback_reason": "realesrgan_timeout",
            }
        if upscale.returncode != 0:
            raise RuntimeError(f"Real-ESRGAN failed: {upscale.stderr[-800:] or upscale.stdout[-800:]}")
        up_frames = sorted(up_dir.glob("*.png"))
        if len(up_frames) < max(1, len(low_frames) - 1):
            raise RuntimeError(f"Real-ESRGAN returned too few frames: {len(up_frames)} of {len(low_frames)}")
        encode = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(up_dir / "frame_%08d.png"),
                "-vf",
                f"scale={width}:{height}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-metadata",
                "comment=Real-ESRGAN ncnn Vulkan x2 neural upscale applied after per-frame cat tracking render.",
                str(output_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if encode.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 4096:
            raise RuntimeError(f"failed to encode Real-ESRGAN clip {output_path}: {encode.stderr[-500:]}")
        return {
            "status": "applied",
            "backend": "realesrgan-ncnn-vulkan",
            "binary": str(DEFAULT_REALESRGAN_BIN),
            "model_dir": str(DEFAULT_REALESRGAN_MODEL_DIR),
            "model": "realesr-animevideov3",
            "scale": 2,
            "stage": "post_overlay",
            "input_resolution": f"{low_w}x{low_h}",
            "output_resolution": f"{width}x{height}",
            "input_frame_count": len(low_frames),
            "output_frame_count": len(up_frames),
            "time_budget_seconds": time_budget_seconds,
        }


def _parse_bbox_xywh(raw: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw, str):
        try:
            parts = [float(part.strip()) for part in raw.split(",")]
        except ValueError:
            return None
    elif isinstance(raw, (list, tuple)):
        try:
            parts = [float(part) for part in raw]
        except (TypeError, ValueError):
            return None
    else:
        return None
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def _rescore_candidate_record(video: dict[str, Any], integrity: SourceIntegrity) -> dict[str, Any]:
    bbox = _parse_bbox_xywh(video.get("max_bbox"))
    measurement_point = video.get("best_measurement_point") or video.get("measurement_point")
    measurement_point = measurement_point if isinstance(measurement_point, dict) else {}
    record = {
        "chunk": integrity.chunk_index,
        "offset_seconds": round(float(integrity.offset_seconds or 0.0), 3),
        "height_cm": round(float(video.get("max_jump_height_cm") or 0.0), 1),
        "reason": "yolo_segmentation_rescore_clean_source",
        "cat_probability": video.get("max_probability"),
        "model_path": video.get("model_path") or video.get("max_model_path"),
        "detector_backend": video.get("max_source") or "ultralytics_yolo_segmentation",
        "measurement_source": video.get("measurement_source"),
        "source_frame_path": str(integrity.frame_path) if integrity.frame_path is not None else None,
        "source_integrity": integrity.to_payload(),
    }
    if bbox is not None:
        x, y, width, height = bbox
        record["bbox"] = [int(round(x)), int(round(y)), int(round(x + width)), int(round(y + height))]
        record["bbox_xywh"] = [round(x, 2), round(y, 2), round(width, 2), round(height, 2)]
        record["top_px"] = [
            measurement_point.get("image_x")
            if measurement_point.get("image_x") is not None
            else round(x + width / 2.0, 2),
            measurement_point.get("image_y") if measurement_point.get("image_y") is not None else round(y, 2),
        ]
    mask_polygon = video.get("best_mask_polygon")
    if isinstance(mask_polygon, list) and mask_polygon:
        record["mask_polygon"] = mask_polygon
    return record


def _rescore_live_sample(
    video: dict[str, Any],
    *,
    source: str,
    triggered: bool = False,
    source_time_s: float | None = None,
) -> Any:
    bbox_xywh = _parse_bbox_xywh(video.get("max_bbox"))
    bbox_xyxy: tuple[int, int, int, int] | None = None
    if bbox_xywh is not None:
        x, y, width, height = bbox_xywh
        bbox_xyxy = (int(round(x)), int(round(y)), int(round(x + width)), int(round(y + height)))
    point = video.get("best_measurement_point") or video.get("measurement_point")
    point = point if isinstance(point, dict) else {}
    top_x = point.get("image_x")
    top_y = point.get("image_y")
    if (top_x is None or top_y is None) and bbox_xywh is not None:
        x, y, width, _height = bbox_xywh
        top_x = x + width / 2.0
        top_y = y
    height_cm = video.get("max_jump_height_cm")
    wall_x_cm = point.get("wall_x_cm", video.get("best_top_wall_x_cm"))
    point_debug = point.get("debug") if isinstance(point.get("debug"), dict) else {}
    coordinate_transform = (
        point_debug.get("coordinate_transform") if isinstance(point_debug.get("coordinate_transform"), dict) else None
    )
    return live_jump.LiveJumpSample(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        source=source,
        candidate_found=bbox_xyxy is not None and height_cm is not None,
        top_x_px=float(top_x) if top_x is not None else None,
        top_y_px=float(top_y) if top_y is not None else None,
        wall_x_mm=float(wall_x_cm) * 10.0 if wall_x_cm is not None else None,
        top_height_mm=float(height_cm) * 10.0 if height_cm is not None else None,
        top_height_cm=float(height_cm) if height_cm is not None else None,
        component_bbox_xyxy=bbox_xyxy,
        triggered=triggered,
        image_path=source,
        detector_cat_probability=video.get("max_probability"),
        detector_model_path=video.get("max_model_path") or video.get("model_path"),
        source_time_s=source_time_s,
        coordinate_transform=coordinate_transform,
    )


def _mask_polygon_points(raw: Any) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        return []
    points: list[tuple[float, float]] = []
    for item in raw:
        try:
            if isinstance(item, dict):
                points.append((float(item["x"]), float(item["y"])))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                points.append((float(item[0]), float(item[1])))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _cover_resize_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    scale = max(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - target_width) // 2)
    top = max(0, (resized.height - target_height) // 2)
    return resized.crop((left, top, left + target_width, top + target_height)).convert("RGB")


def _crop_with_context_padding(
    source: Image.Image,
    *,
    crop_left: float,
    crop_top: float,
    crop_width: float,
    crop_height: float,
    output_size: tuple[int, int],
) -> Image.Image:
    canvas_width = max(1, int(round(crop_width)))
    canvas_height = max(1, int(round(crop_height)))
    background = Image.new("RGB", (canvas_width, canvas_height), (10, 10, 10))
    left_i = int(math.floor(crop_left))
    top_i = int(math.floor(crop_top))
    right_i = int(math.ceil(crop_left + crop_width))
    bottom_i = int(math.ceil(crop_top + crop_height))
    src_left = max(0, left_i)
    src_top = max(0, top_i)
    src_right = min(source.width, right_i)
    src_bottom = min(source.height, bottom_i)
    if src_right > src_left and src_bottom > src_top:
        paste_x = src_left - left_i
        paste_y = src_top - top_i
        background.paste(source.crop((src_left, src_top, src_right, src_bottom)), (paste_x, paste_y))
    return background.resize(output_size, Image.Resampling.LANCZOS)


def _rescore_bbox_xyxy(video: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = _parse_bbox_xywh(video.get("max_bbox"))
    if bbox is None:
        return None
    x, y, width, height = bbox
    return x, y, x + width, y + height


def _trajectory_close_crop_bounds(
    *,
    image_size: tuple[int, int],
    trajectory_bbox: tuple[float, float, float, float] | None,
    output_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    return _require_cat_tv_play_publish_render().close_crop_bounds(
        image_size=image_size,
        content_bounds=trajectory_bbox,
        output_size=output_size,
    )


def _bbox_union(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    return _require_cat_tv_play_publish_render().bbox_union(first, second)


def _bbox_for_points(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    return _require_cat_tv_play_publish_render().bbox_for_points(points)


def _projector_screen_bbox(image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    return _require_cat_tv_play_publish_render().projector_screen_bbox(image_size)


def _publish_recording_datetime(video: dict[str, Any], integrity: SourceIntegrity | None = None) -> datetime | None:
    return _require_cat_tv_play_publish_render().recording_datetime(
        label=str(video.get("label") or ""),
        recording_dir=integrity.recording_dir if integrity is not None else None,
    )


def _publish_recording_date_label(video: dict[str, Any], integrity: SourceIntegrity | None = None) -> str | None:
    return _require_cat_tv_play_publish_render().recording_date_label(
        label=str(video.get("label") or ""),
        recording_dir=integrity.recording_dir if integrity is not None else None,
    )


def _trajectory_bbox_for_rescore_frames(
    frames: list[Path],
    *,
    rows_by_raw_path: dict[str, dict[str, Any]],
    rows_by_frame_key: dict[tuple[int, int], dict[str, Any]] | None = None,
    fallback_video: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    boxes: list[tuple[float, float, float, float]] = []
    for frame_path in frames:
        row = _probe_row_for_frame_path(frame_path, rows_by_raw_path, rows_by_frame_key)
        if row is None:
            continue
        frame_video = _merge_probe_row_into_rescore_video(fallback_video, row)
        box = _bbox_for_points(_mask_polygon_points(frame_video.get("best_mask_polygon")))
        if box is None:
            box = _rescore_bbox_xyxy(frame_video)
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _alpha_scaled(mask: Image.Image, alpha_scale: float) -> Image.Image:
    scale = max(0.0, min(1.0, alpha_scale))
    return mask.point(lambda value: int(round(value * scale)))


def _draw_soft_outer_glow(
    base: Image.Image,
    *,
    polygon: list[tuple[int, int]] | None = None,
    rect: tuple[int, int, int, int] | None = None,
    color: tuple[int, int, int] = (255, 64, 190),
    alpha: int = 82,
    buffer_px: int = 8,
    blur_px: float = 11.0,
) -> None:
    if not polygon and rect is None:
        return
    mask = Image.new("L", base.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    if polygon and len(polygon) >= 3:
        mask_draw.polygon(polygon, fill=255)
    elif rect is not None:
        mask_draw.rectangle(rect, fill=255)
    else:
        return
    kernel = max(3, buffer_px * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    expanded = mask.filter(ImageFilter.MaxFilter(kernel))
    outer_ring = ImageChops.subtract(expanded, mask)
    glow_alpha = outer_ring.filter(ImageFilter.GaussianBlur(radius=blur_px))
    glow_alpha = ImageChops.multiply(glow_alpha, ImageChops.invert(mask))
    glow_alpha = _alpha_scaled(glow_alpha, alpha / 255.0)
    glow = Image.new("RGBA", base.size, (*color, 0))
    glow.putalpha(glow_alpha)
    base.alpha_composite(glow)


def _draw_rescore_overlay_frame(
    image: Image.Image,
    output_path: Path,
    *,
    video: dict[str, Any],
    record_video: dict[str, Any] | None = None,
    crop_video: dict[str, Any] | None = None,
    sample: Any | None = None,
    track_state: Any | None = None,
    top_holds: tuple[Any, ...] = (),
    homography: Any | None = None,
    floor_y_mm: float | None = None,
    zoom_progress: float = 1.0,
    close_crop_bounds: tuple[float, float, float, float] | None = None,
    active: bool,
    output_size: tuple[int, int] = (1080, 1920),
    session_title: str | None = None,
    session_subtitle: str | None = None,
) -> None:
    record_video = record_video or video
    crop_bbox = _parse_bbox_xywh((crop_video or video).get("max_bbox"))
    bbox = _parse_bbox_xywh(video.get("max_bbox"))
    source = image.convert("RGB")
    if crop_bbox is not None:
        x, y, width, height = crop_bbox
        center_x = x + width / 2.0
        center_y = y + height * 0.52
    elif bbox is not None:
        x, y, width, height = bbox
        center_x = x + width / 2.0
        center_y = y + height * 0.52
    else:
        center_x = source.width / 2.0
        center_y = source.height / 2.0
    if close_crop_bounds is not None:
        close_left, close_top, close_crop_width, close_crop_height = close_crop_bounds
        close_center_x = close_left + close_crop_width / 2.0
        close_center_y = close_top + close_crop_height / 2.0
        open_center_x = source.width / 2.0
        open_center_y = source.height / 2.0
    else:
        close_crop_height = min(source.height, max(500, int(round(source.height * 0.78))))
        close_crop_width = min(source.width, int(round(close_crop_height * output_size[0] / output_size[1])))
        close_center_x = center_x
        close_center_y = center_y
        open_center_x = center_x
        open_center_y = center_y
    full_crop_width = float(source.width)
    full_crop_height = full_crop_width * output_size[1] / float(output_size[0])
    zoom = max(0.0, min(1.0, zoom_progress))
    smooth_zoom = zoom * zoom * (3.0 - 2.0 * zoom)
    crop_width = full_crop_width + (close_crop_width - full_crop_width) * smooth_zoom
    crop_height = full_crop_height + (close_crop_height - full_crop_height) * smooth_zoom
    crop_center_x = open_center_x + (close_center_x - open_center_x) * smooth_zoom
    crop_center_y = open_center_y + (close_center_y - open_center_y) * smooth_zoom
    crop_left = crop_center_x - crop_width / 2.0
    crop_top = crop_center_y - crop_height / 2.0
    if crop_width <= source.width:
        crop_left = max(0.0, min(float(source.width) - crop_width, crop_left))
    if crop_height <= source.height:
        crop_top = max(0.0, min(float(source.height) - crop_height, crop_top))
    scale = output_size[0] / float(crop_width)
    portrait = _crop_with_context_padding(
        source,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_width=crop_width,
        crop_height=crop_height,
        output_size=output_size,
    )
    portrait = ImageEnhance.Contrast(portrait).enhance(1.08)
    portrait = (
        ImageEnhance.Sharpness(portrait)
        .enhance(1.16)
        .filter(ImageFilter.UnsharpMask(radius=1.0, percent=75, threshold=4))
    )
    base = portrait.convert("RGBA")
    draw = ImageDraw.Draw(base)
    title_font = live_jump._default_font(50)  # noqa: SLF001
    height_cm = float(video.get("max_jump_height_cm") or 0.0)
    display_height_cm = None
    if track_state is not None and getattr(track_state, "highest_sample", None) is not None:
        display_height_cm = getattr(track_state.highest_sample, "top_height_cm", None)
    record_height_cm = float(display_height_cm if display_height_cm is not None else height_cm)
    max_line_anchor_xy = None
    if (
        sample is not None
        and getattr(sample, "top_x_px", None) is not None
        and getattr(sample, "top_y_px", None) is not None
        and getattr(sample, "top_height_cm", None) is not None
        and abs(float(sample.top_height_cm) - record_height_cm) < 0.05
    ):
        max_line_anchor_xy = (float(sample.top_x_px), float(sample.top_y_px))
    if track_state is not None and record_height_cm >= 80.0:
        live_jump._draw_vertical_track_state(  # noqa: SLF001
            base,
            track_state,
            font=title_font,
            homography=homography,
            floor_y_mm=floor_y_mm,
            display_max_height_cm=record_height_cm,
            source_image=source,
            crop_left=int(round(crop_left)),
            crop_top=int(round(crop_top)),
            scale=scale,
            show_local_peaks=False,
            show_trail=False,
            line_coordinate_transform=getattr(sample, "coordinate_transform", None),
            max_line_label=False,
            max_line_width=2,
            max_line_fill=(255, 64, 190, 135),
            max_line_anchor_xy=max_line_anchor_xy,
        )
    if bbox is not None:
        x, y, width, height = bbox
        rect = (
            int(round((x - crop_left) * scale)),
            int(round((y - crop_top) * scale)),
            int(round((x + width - crop_left) * scale)),
            int(round((y + height - crop_top) * scale)),
        )
        mask_points = _mask_polygon_points(video.get("best_mask_polygon"))
        if mask_points:
            transformed = [
                (int(round((point_x - crop_left) * scale)), int(round((point_y - crop_top) * scale)))
                for point_x, point_y in mask_points
            ]
            if len(transformed) >= 3:
                _draw_soft_outer_glow(
                    base,
                    polygon=transformed,
                    alpha=132 if active else 58,
                    buffer_px=11 if active else 6,
                    blur_px=10.0 if active else 7.0,
                )
        else:
            _draw_soft_outer_glow(
                base,
                rect=rect,
                alpha=108 if active else 46,
                buffer_px=10 if active else 6,
                blur_px=10.0 if active else 7.0,
            )
        if height_cm >= 80.0:
            safe_top = int(DEFAULT_TIKTOK_SAFE_ZONE["top"]) + 260
            safe_bottom = output_size[1] - int(DEFAULT_TIKTOK_SAFE_ZONE["bottom"])
            label_y = max(safe_top, min(safe_bottom - 68, rect[1] - 58))
            live_jump._draw_text_with_stroke(  # noqa: SLF001
                draw,
                (max(int(DEFAULT_TIKTOK_SAFE_ZONE["left"]), min(output_size[0] - 260, rect[0])), label_y),
                f"{height_cm:.1f} cm",
                fill=(255, 245, 255, 255),
                font=title_font,
                stroke_width=4,
            )
    live_jump._draw_top_jump_holds_top_bar(  # noqa: SLF001
        base,
        top_holds,
        font=live_jump._default_font(42),  # noqa: SLF001
        avoid_bbox_xyxy=rect if bbox is not None else None,
        publish_safe_zone=True,
        session_title=session_title,
        session_subtitle=session_subtitle,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    live_jump._save_jpeg_atomic(base.convert("RGB"), output_path)  # noqa: SLF001


def _render_publishable_rescore_jump(
    *,
    jump_heights_path: Path,
    output_path: Path,
    preview_path: Path | None = None,
    probe_rows_path: Path | None = None,
    fps: int = DEFAULT_TELEGRAM_ENHANCED_FPS,
    pre_seconds: float = 2.0,
    post_seconds: float = 2.0,
    min_duration_seconds: float = 8.0,
    min_source_fps: float | None = DEFAULT_PUBLISHABLE_MIN_SOURCE_FPS,
    min_jump_count: int = DEFAULT_PUBLISHABLE_MIN_JUMP_COUNT,
) -> dict[str, Any]:
    requested_jump_heights_path = jump_heights_path
    jump_heights_path = _resolve_jump_heights_data_path(jump_heights_path)
    candidates, suspicious = _select_publishable_rescore_jumps(
        jump_heights_path,
        min_source_fps=min_source_fps,
        min_jump_count=min_jump_count,
    )
    if len(candidates) < min_jump_count:
        fps_suffix = (
            f" with source fps >= {min_source_fps:.3f}" if min_source_fps is not None and min_source_fps > 0 else ""
        )
        raise RuntimeError(
            f"need at least {min_jump_count} source-integrity-clean jump candidates{fps_suffix} "
            f"in {requested_jump_heights_path}; found {len(candidates)}"
        )
    probe_rows_path = probe_rows_path or jump_heights_path.parent / "probe_rows.json"
    rows_by_raw_path = _load_probe_rows_by_raw_path(probe_rows_path)
    seed_videos = sorted(
        _load_jump_height_videos(jump_heights_path),
        key=lambda item: float(item.get("max_jump_height_cm") or 0.0),
        reverse=True,
    )
    if seed_videos:
        same_source_candidates = _select_same_source_probe_rescore_jumps(
            seed_videos[0],
            rows_by_raw_path,
            min_source_fps=None,
            min_jump_count=min_jump_count,
        )
        if len(same_source_candidates) >= min_jump_count:
            candidates = same_source_candidates
        elif min_jump_count > 1:
            recording_dirs = {
                str(candidate.integrity.recording_dir)
                for candidate in candidates
                if candidate.integrity.recording_dir is not None
            }
            if len(recording_dirs) > 1:
                raise RuntimeError(
                    f"need at least {min_jump_count} publishable jumps from one source recording; "
                    f"found {len(same_source_candidates)} for top seed"
                )
    primary = candidates[0]
    primary_video = primary.video
    calibration_path = _calibration_path_for_jump_heights(jump_heights_path)
    with tempfile.TemporaryDirectory(prefix="cat-projector-publish-render-") as tmpdir:
        tmp_path = Path(tmpdir)
        segments: list[PublishableJumpSegment] = []
        for index, candidate in enumerate(candidates):
            segment = _publishable_segment_for_candidate(
                candidate,
                tmp_path / f"source_frames_{index:02d}",
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
            )
            if segment is not None:
                segments.append(segment)
        if len(segments) < min_jump_count:
            raise RuntimeError(f"need at least {min_jump_count} publishable jump source windows; found {len(segments)}")
        annotated_dir = tmp_path / "annotated"
        annotated_dir.mkdir()
        tracked_frames = 0
        masked_frames = 0
        missing_probe_frames = 0
        homography, floor_y_mm = live_jump._read_calibration(calibration_path)  # noqa: SLF001
        top_holds: tuple[Any, ...] = ()
        output_frame_index = 0
        preview_frame_path: Path | None = None
        segment_metadata: list[dict[str, Any]] = []
        encode_fps = max(1, int(round(min(segment.source_fps for segment in segments))))
        publish_helpers = _require_cat_tv_play_publish_render()
        presentation = publish_helpers.presentation_metadata(
            label=str(primary_video.get("label") or ""),
            recording_dir=primary.integrity.recording_dir,
        )
        session_title = str(presentation.get("title") or "SHER JUMPS")
        session_subtitle = presentation.get("recorded_on")
        global_trajectory_bbox: tuple[float, float, float, float] | None = None
        global_content_bbox: tuple[float, float, float, float] | None = None
        global_screen_bbox: tuple[float, float, float, float] | None = None
        global_image_size: tuple[int, int] | None = None
        for segment in segments:
            integrity = segment.candidate.integrity
            rows_by_frame_key = _load_probe_rows_by_frame_key(
                rows_by_raw_path,
                recording_dir=integrity.recording_dir,
                external_raw_prefix=segment.probe_raw_prefix,
            )
            probe_row = (
                _probe_row_for_frame_path(integrity.frame_path, rows_by_raw_path, rows_by_frame_key)
                if integrity.frame_path is not None
                else None
            )
            segment_video = _merge_probe_row_into_rescore_video(segment.candidate.video, probe_row)
            with Image.open(segment.frames[0]) as first_image:
                if global_image_size is None:
                    global_image_size = first_image.size
            trajectory_bbox = _trajectory_bbox_for_rescore_frames(
                list(segment.frames),
                rows_by_raw_path=rows_by_raw_path,
                rows_by_frame_key=rows_by_frame_key,
                fallback_video=segment_video,
            )
            if trajectory_bbox is None:
                continue
            global_trajectory_bbox = _bbox_union(global_trajectory_bbox, trajectory_bbox)
        if global_image_size is not None:
            global_screen_bbox = publish_helpers.projector_screen_bbox(global_image_size)
            global_content_bbox = publish_helpers.content_bbox(
                image_size=global_image_size,
                cat_boxes=[global_trajectory_bbox],
            )
        global_close_crop_bounds = (
            publish_helpers.close_crop_bounds(
                image_size=global_image_size,
                content_bounds=global_content_bbox,
                output_size=(1080, 1920),
            )
            if global_image_size is not None
            else None
        )
        first_active_output_frame = segments[0].active_frame_index if segments else 0
        zoom_ramp_frames = max(1.0, min(float(first_active_output_frame), float(fps) * 2.0))
        for segment_index, segment in enumerate(segments):
            candidate = segment.candidate
            integrity = candidate.integrity
            rows_by_frame_key = _load_probe_rows_by_frame_key(
                rows_by_raw_path,
                recording_dir=integrity.recording_dir,
                external_raw_prefix=segment.probe_raw_prefix,
            )
            probe_row = (
                _probe_row_for_frame_path(integrity.frame_path, rows_by_raw_path, rows_by_frame_key)
                if integrity.frame_path is not None
                else None
            )
            segment_video = _merge_probe_row_into_rescore_video(candidate.video, probe_row)
            track_state = live_jump.JumpTrackState()
            segment_output_start = output_frame_index
            for index, frame_path in enumerate(segment.frames):
                image = Image.open(frame_path).convert("RGB")
                frame_row = _probe_row_for_frame_path(frame_path, rows_by_raw_path, rows_by_frame_key)
                frame_video = _rescore_frame_video_from_probe(segment_video, frame_row)
                if frame_row is None:
                    missing_probe_frames += 1
                else:
                    tracked_frames += 1
                    if _mask_polygon_points(frame_video.get("best_mask_polygon")):
                        masked_frames += 1
                parts = _frame_label_parts(frame_path)
                frame_index = parts[1] if parts is not None else index
                frame_sample = _rescore_live_sample(
                    frame_video,
                    source=str(frame_path),
                    triggered=index >= segment.active_frame_index,
                    source_time_s=(float(integrity.chunk_index or 0) * DEFAULT_SEGMENT_SECONDS)
                    + frame_index / max(segment.source_fps, 1.0),
                )
                track_state = live_jump.update_track_state(track_state, frame_sample, max_points=90)
                top_holds = live_jump.update_top_jump_holds(
                    top_holds,
                    image,
                    frame_sample,
                    max_holds=3,
                    min_height_cm=45.0,
                )
                output_path_frame = annotated_dir / f"frame_{output_frame_index:06d}.jpg"
                _draw_rescore_overlay_frame(
                    image,
                    output_path_frame,
                    video=frame_video,
                    record_video=segment_video,
                    crop_video=frame_video,
                    sample=frame_sample,
                    track_state=track_state,
                    top_holds=top_holds,
                    homography=homography,
                    floor_y_mm=floor_y_mm,
                    zoom_progress=output_frame_index / zoom_ramp_frames,
                    close_crop_bounds=global_close_crop_bounds,
                    active=index >= segment.active_frame_index,
                    session_title=session_title,
                    session_subtitle=session_subtitle,
                )
                if segment_index == 0 and index == segment.active_frame_index:
                    preview_frame_path = output_path_frame
                output_frame_index += 1
            segment_metadata.append(
                {
                    "active_frame_index": segment.active_frame_index,
                    "chunk": integrity.chunk_index,
                    "end_frame": segment.frames[-1].name,
                    "height_cm": candidate.video.get("max_jump_height_cm"),
                    "label": candidate.video.get("label"),
                    "frame_source": segment.frame_source,
                    "output_start_frame": segment_output_start,
                    "post_available_seconds": round(segment.post_available_seconds, 3),
                    "pre_available_seconds": round(segment.pre_available_seconds, 3),
                    "probe_raw_prefix": str(segment.probe_raw_prefix) if segment.probe_raw_prefix is not None else None,
                    "recording_dir": str(integrity.recording_dir) if integrity.recording_dir is not None else None,
                    "source_fps": round(segment.source_fps, 3),
                    "source_integrity": integrity.to_payload(),
                    "source_seconds": round(len(segment.frames) / max(segment.source_fps, 1.0), 3),
                    "start_frame": segment.frames[0].name,
                }
            )
        source_frame_count = sum(len(segment.frames) for segment in segments)
        source_motion_seconds = source_frame_count / max(encode_fps, 1)
        results_hold_frame_count = 0
        input_fps = float(encode_fps)
        base_output_path = tmp_path / "tracked_base.mp4"
        _encode_frame_sequence(
            frame_dir=annotated_dir,
            output_path=base_output_path,
            fps=encode_fps,
            vf="scale=1080:1920",
            crf=18,
            input_fps=input_fps,
            metadata={
                "title": f"Sher jumps {float(primary_video.get('max_jump_height_cm') or 0.0):.1f} cm",
                "artist": "Sher",
                "genre": "Cat projector jump",
                "comment": "TikTok-ready clean-source Cat Projector jump render.",
                "description": "Source-integrity-clean multi-jump render; measurements stay tied to source frames.",
            },
        )
        fps_upscale = None
        neural_output_path = tmp_path / "tracked_neural.mp4" if fps > encode_fps else output_path
        neural_upscale = _neural_upscale_clip_with_realesrgan(
            base_output_path,
            neural_output_path,
            fps=encode_fps,
            time_budget_seconds=DEFAULT_PUBLISHABLE_NEURAL_UPSCALE_SECONDS,
        )
        if fps > encode_fps:
            fps_source_path = (
                neural_output_path
                if isinstance(neural_upscale, dict) and neural_upscale.get("status") == "applied"
                else base_output_path
            )
            fps_upscale = _enhance_vertical_clip_for_social(
                fps_source_path,
                output_path,
                height_cm=float(primary_video.get("max_jump_height_cm") or 0.0),
                fps=fps,
                title=f"Sher jumps {float(primary_video.get('max_jump_height_cm') or 0.0):.1f} cm",
                comment="Cat projector jump highlight; exact source frames tracked before fps-up.",
            )
        elif not isinstance(neural_upscale, dict) or neural_upscale.get("status") != "applied":
            shutil.copy2(base_output_path, output_path)
        if preview_path is not None and preview_frame_path is not None:
            shutil.copy2(preview_frame_path, preview_path)
    record = _rescore_candidate_record(primary_video, primary.integrity)
    metadata = {
        "kind": "cat_projector_publishable_jump_render_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "jump_heights_path": str(jump_heights_path),
        "requested_jump_heights_path": str(requested_jump_heights_path),
        "calibration_path": str(calibration_path),
        "output_path": str(output_path),
        "preview_path": str(preview_path) if preview_path is not None else None,
        "height_cm": primary_video.get("max_jump_height_cm"),
        "video_id": primary_video.get("video_id"),
        "label": primary_video.get("label"),
        "source_integrity": primary.integrity.to_payload(),
        "probe_rows_path": str(probe_rows_path) if probe_rows_path.exists() else None,
        "probe_row_raw_path": primary_video.get("probe_row_raw_path"),
        "mask_polygon_points": len(_mask_polygon_points(primary_video.get("best_mask_polygon"))),
        "timeline": {
            "source": "source_chunk_exact_frames_probe_rows_by_frame_id",
            "jump_event_count": len(segments),
            "jump_events": segment_metadata,
            "source_frames": source_frame_count,
            "tracked_frames": tracked_frames,
            "masked_frames": masked_frames,
            "missing_probe_frames": missing_probe_frames,
            "source_input_fps": round(min(segment.source_fps for segment in segments), 3),
            "render_input_fps": round(input_fps, 3),
            "output_fps": encode_fps,
            "requested_output_fps": fps,
            "min_source_fps": min_source_fps,
            "min_jump_count": min_jump_count,
            "motion_seconds": round(source_motion_seconds, 3),
            "min_duration_met": source_motion_seconds >= float(min_duration_seconds),
            "results_hold_frames": results_hold_frame_count,
            "results_hold_seconds": round(results_hold_frame_count / max(input_fps, 1.0), 3),
            "min_duration_seconds": min_duration_seconds,
            "camera_plan": {
                "close_crop_bounds": [round(float(value), 3) for value in global_close_crop_bounds]
                if global_close_crop_bounds is not None
                else None,
                "trajectory_bbox": [round(float(value), 3) for value in global_trajectory_bbox]
                if global_trajectory_bbox is not None
                else None,
                "screen_bbox": [round(float(value), 3) for value in global_screen_bbox]
                if global_screen_bbox is not None
                else None,
                "content_bbox": [round(float(value), 3) for value in global_content_bbox]
                if global_content_bbox is not None
                else None,
                "zoom_ramp_frames": round(zoom_ramp_frames, 3),
                "zoom_scope": "global_screen_plus_cat_polygons",
            },
            "presentation": {
                "subject": presentation.get("subject"),
                "title": session_title,
                "recorded_on": session_subtitle,
            },
        },
        "top_holds": [round(float(hold.height_cm), 1) for hold in live_jump.confirmed_top_jump_holds(top_holds)],
        "jump_highlight": record,
        "suspicious_candidates": [
            {
                "height_cm": item.video.get("max_jump_height_cm"),
                "label": item.video.get("label"),
                "max_frame_path": item.video.get("max_frame_path"),
                "source_integrity": item.integrity.to_payload(),
            }
            for item in suspicious[:20]
        ],
        "enhancement": {
            "backend": "per_frame_probe_rows_ffmpeg_fps_scale",
            "fps": encode_fps,
            "requested_fps": fps,
            "resolution": "1080x1920",
            "neural_upscale": neural_upscale,
            "fps_upscale": fps_upscale,
            "rife_fps_up": {
                "installed": DEFAULT_RIFE_BIN.exists() and DEFAULT_RIFE_MODEL_DIR.exists(),
                "enabled": False,
                "reason": "disabled_until_overlays_are_drawn_after_interpolation_to_avoid_ghost_contours",
                "binary": str(DEFAULT_RIFE_BIN) if DEFAULT_RIFE_BIN.exists() else None,
                "model_dir": str(DEFAULT_RIFE_MODEL_DIR) if DEFAULT_RIFE_MODEL_DIR.exists() else None,
            },
        },
        "metadata": {
            "title": f"Sher jumps {float(primary_video.get('max_jump_height_cm') or 0.0):.1f} cm",
            "artist": "Sher",
            "genre": "Cat projector jump",
            "comment": "TikTok-ready clean-source Cat Projector jump render.",
        },
    }
    _json_dump(output_path.with_suffix(".json"), metadata)
    return metadata


def _append_results_hold_frames(
    *,
    frame_dir: Path,
    frame_index: int,
    image: Image.Image,
    top_holds: tuple[Any, ...],
    fps: int,
    seconds: float = DEFAULT_TELEGRAM_RESULTS_HOLD_SECONDS,
    output_size: tuple[int, int] | None = None,
    publish_safe_zone: bool = False,
    session_title: str | None = None,
    session_subtitle: str | None = None,
) -> int:
    frame_count = max(1, int(round(fps * seconds)))
    if frame_count <= 0:
        return 0
    hold_frame = frame_dir / f"frame_{frame_index:06d}.jpg"
    live_jump.render_jump_results_hold(
        image,
        hold_frame,
        top_holds=top_holds,
        output_size=output_size,
        publish_safe_zone=publish_safe_zone,
        session_title=session_title,
        session_subtitle=session_subtitle,
    )
    for offset in range(1, frame_count):
        shutil.copy2(hold_frame, frame_dir / f"frame_{frame_index + offset:06d}.jpg")
    return frame_count


def _render_sample_original_overlay(
    sample: Any,
    *,
    sample_fps: float = DEFAULT_TELEGRAM_JUMP_RENDER_FPS,
    source_frame_label: str | None = None,
) -> dict[str, Any] | None:
    if not getattr(sample, "candidate_found", False) or getattr(sample, "component_bbox_xyxy", None) is None:
        return None
    frame_label = Path(str(getattr(sample, "source", ""))).name
    match = re.match(r"chunk_(\d+)_(\d+)\.jpg", frame_label)
    if not match:
        return None
    chunk_index = int(match.group(1))
    frame_index = int(match.group(2))
    bbox = getattr(sample, "component_bbox_xyxy", None)
    top_x = getattr(sample, "top_x_px", None)
    top_y = getattr(sample, "top_y_px", None)
    top_px = [top_x, top_y] if top_x is not None and top_y is not None else None
    detector_model_path = getattr(sample, "detector_model_path", None)
    return {
        "frame_label": frame_label,
        "render_frame_label": frame_label,
        "source_frame_label": source_frame_label or frame_label,
        "chunk": chunk_index,
        "frame_index": frame_index,
        "offset_seconds": round(max(0, frame_index - 1) / sample_fps, 3) if sample_fps else 0.0,
        "bbox": [int(value) for value in bbox],
        "area": getattr(sample, "component_area_px", None),
        "top_px": top_px,
        "height_cm": getattr(sample, "top_height_cm", None),
        "wall_x_mm": getattr(sample, "wall_x_mm", None),
        "cat_probability": getattr(sample, "detector_cat_probability", None),
        "model_path": detector_model_path,
        "reason": "telegram_render_detector" if detector_model_path else "telegram_render_dark_component",
    }


def _build_annotated_jump_clip(
    chunks: list[Path],
    output_path: Path,
    *,
    vertical_output_path: Path | None = None,
    training_queue_root: Path | None = None,
    fps: int = DEFAULT_TELEGRAM_JUMP_RENDER_FPS,
    min_jump_height_cm: float = DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
    expected_alert_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    homography, floor_y_mm = live_jump._read_calibration(live_jump.DEFAULT_CALIBRATION)  # noqa: SLF001
    detector_loaded_model = None
    detector_loaded_metadata = None
    if default_model_available():
        try:
            detector_loaded_model, detector_loaded_metadata = load_model(
                DEFAULT_DETECTOR_MODEL_PATH,
                DEFAULT_DETECTOR_METADATA_PATH,
            )
        except Exception:
            detector_loaded_model = None
            detector_loaded_metadata = None
    with tempfile.TemporaryDirectory(prefix="cat-projector-jump-render-") as tmpdir:
        tmp_path = Path(tmpdir)
        raw_dir = tmp_path / "raw"
        annotated_dir = tmp_path / "annotated"
        vertical_dir = tmp_path / "vertical"
        raw_dir.mkdir()
        annotated_dir.mkdir()
        if vertical_output_path is not None:
            vertical_dir.mkdir()
        frame_paths = _extract_video_frames(chunks, raw_dir, fps=fps)
        if not frame_paths:
            raise RuntimeError("cannot build annotated jump clip without extracted frames")
        room_background = _room_background_for_frames(chunks, frame_paths)
        source_video = _projector_source_video_for_chunks(chunks)
        source_frames = (
            _extract_source_video_frames(source_video, chunks, tmp_path / "source", fps=fps)
            if source_video is not None
            else {}
        )
        residual_baseline = (
            _source_residual_baseline(
                frame_paths,
                source_video=source_video,
                source_frames=source_frames,
                sample_fps=fps,
            )
            if source_video is not None and detector_loaded_model is not None and detector_loaded_metadata is not None
            else None
        )
        top_holds: tuple[Any, ...] = ()
        samples: list[Any] = []
        original_model_overlays: list[dict[str, Any]] = []
        detector_context = _detector_context_payload(
            detector_loaded_metadata=detector_loaded_metadata,
            backend="source_subtracted"
            if detector_loaded_model is not None and source_video is not None
            else "legacy_full_frame",
            path="source_subtracted" if detector_loaded_model is not None and source_video is not None else "legacy",
        )
        canonical_alert_sample: Any | None = None
        canonical_alert_record: dict[str, Any] | None = None
        last_image: Image.Image | None = None
        for frame_path in frame_paths:
            source_frame = None
            if source_video is not None:
                match = re.match(r"chunk_(\d+)_(\d+)\.jpg", frame_path.name)
                if match:
                    chunk_index = int(match.group(1))
                    frame_index = int(match.group(2))
                    source_path = source_frames.get((chunk_index, frame_index))
                    source_frame = (
                        Image.open(source_path).convert("RGB")
                        if source_path is not None and source_path.exists()
                        else _source_frame_at(
                            source_video,
                            _frame_time_seconds(chunk_index, frame_index, sample_fps=fps),
                        )
                    )
            image = Image.open(frame_path).convert("RGB")
            last_image = image
            sample = _sample_render_frame(
                image,
                source=str(frame_path),
                source_frame=source_frame,
                room_background=room_background,
                residual_baseline=residual_baseline,
                homography=homography,
                floor_y_mm=floor_y_mm,
                min_jump_height_cm=min_jump_height_cm,
                detector_loaded_model=detector_loaded_model,
                detector_loaded_metadata=detector_loaded_metadata,
            )
            samples.append(sample)
            if (
                canonical_alert_sample is None
                and sample.candidate_found
                and sample.top_height_cm is not None
                and sample.top_height_cm >= min_jump_height_cm
                and (
                    expected_alert_record is None
                    or _sample_matches_alert_record(sample, expected_alert_record, sample_fps=fps)
                )
            ):
                canonical_alert_sample = sample
                canonical_alert_record = _alert_record_from_sample(
                    sample,
                    sample_fps=fps,
                    reason=str(
                        (expected_alert_record or {}).get("reason")
                        or (
                            "source_subtracted_calibrated_high_jump"
                            if source_frame is not None
                            else "calibrated_high_jump"
                        )
                    ),
                )
        publish_helpers = _require_cat_tv_play_publish_render()
        with Image.open(frame_paths[0]) as first_render_image:
            render_image_size = first_render_image.size
        telegram_content_bbox = publish_helpers.content_bbox(
            image_size=render_image_size,
            cat_boxes=[getattr(sample, "component_bbox_xyxy", None) for sample in samples],
        )
        telegram_close_crop_bounds = publish_helpers.close_crop_bounds(
            image_size=render_image_size,
            content_bounds=telegram_content_bbox,
            output_size=(1080, 1920),
        )
        presentation = publish_helpers.presentation_metadata(recording_dir=chunks[0].parent if chunks else None)
        session_title = str(presentation.get("title") or "SHER JUMPS")
        session_subtitle = presentation.get("recorded_on")
        track_state = live_jump.JumpTrackState()
        vertical_crop_state = live_jump.VerticalCropState()
        top_holds = ()
        for index, (frame_path, sample) in enumerate(zip(frame_paths, samples, strict=True)):
            image = Image.open(frame_path).convert("RGB")
            last_image = image
            track_state = live_jump.update_track_state(track_state, sample, max_points=90)
            top_holds = live_jump.update_top_jump_holds(
                top_holds,
                image,
                sample,
                max_holds=3,
                min_height_cm=45.0,
            )
            live_jump.render_annotation(
                image,
                sample,
                annotated_dir / f"frame_{index:06d}.jpg",
                homography=homography,
                floor_y_mm=floor_y_mm,
                track_state=track_state,
                top_holds=top_holds,
            )
            if vertical_output_path is not None:
                vertical_crop_state = live_jump.render_vertical_annotation(
                    image,
                    sample,
                    vertical_dir / f"frame_{index:06d}.jpg",
                    homography=homography,
                    floor_y_mm=floor_y_mm,
                    track_state=track_state,
                    top_holds=top_holds,
                    crop_state=vertical_crop_state,
                    output_size=(1080, 1920),
                    close_crop_bounds=telegram_close_crop_bounds,
                    publish_safe_zone=True,
                    session_title=session_title,
                    session_subtitle=session_subtitle,
                )

        exact_frame_paths = _extract_exact_video_frames(chunks, tmp_path / "exact-source-overlays")
        chunks_by_index = {index: chunk for chunk in chunks if (index := _chunk_index(chunk)) is not None}
        chunk_frame_rates: dict[int, float] = {}
        for frame_path in exact_frame_paths:
            match = re.match(r"chunk_(\d+)_(\d+)\.jpg", frame_path.name)
            if not match:
                continue
            chunk_index = int(match.group(1))
            if chunk_index not in chunk_frame_rates:
                chunk_frame_rates[chunk_index] = _chunk_frame_rate(chunks_by_index.get(chunk_index))
            image = Image.open(frame_path).convert("RGB")
            source_frame = None
            frame_rate = chunk_frame_rates.get(chunk_index)
            if source_video is not None and frame_rate:
                source_frame = _source_frame_at(
                    source_video,
                    chunk_index * DEFAULT_SEGMENT_SECONDS + max(0, int(match.group(2)) - 1) / frame_rate,
                )
            sample = _sample_render_frame(
                image,
                source=str(frame_path),
                source_frame=source_frame,
                room_background=room_background,
                residual_baseline=None,
                homography=homography,
                floor_y_mm=floor_y_mm,
                min_jump_height_cm=min_jump_height_cm,
                detector_loaded_model=detector_loaded_model,
                detector_loaded_metadata=detector_loaded_metadata,
            )
            overlay = _render_sample_original_overlay(
                sample,
                sample_fps=chunk_frame_rates[chunk_index],
                source_frame_label=frame_path.name,
            )
            if overlay is not None:
                original_model_overlays.append(overlay)

        results_hold_frame_count = 0
        if last_image is not None:
            results_hold_frame_count = _append_results_hold_frames(
                frame_dir=annotated_dir,
                frame_index=len(frame_paths),
                image=last_image,
                top_holds=top_holds,
                fps=fps,
            )
            if vertical_output_path is not None:
                _append_results_hold_frames(
                    frame_dir=vertical_dir,
                    frame_index=len(frame_paths),
                    image=last_image,
                    top_holds=top_holds,
                    fps=fps,
                    output_size=(1080, 1920),
                    publish_safe_zone=True,
                    session_title=session_title,
                    session_subtitle=session_subtitle,
                )

        hard_case_queue_root = training_queue_root or DEFAULT_ROOT / "detector-training-queue"
        hard_cases = _queue_continuity_hard_cases(
            frame_paths=frame_paths,
            samples=samples,
            queue_root=hard_case_queue_root,
        )
        _encode_frame_sequence(frame_dir=annotated_dir, output_path=output_path, fps=fps, vf="scale=960:-2", crf=23)
        if vertical_output_path is not None:
            _encode_frame_sequence(
                frame_dir=vertical_dir,
                output_path=vertical_output_path,
                fps=fps,
                vf="scale=1080:1920",
                crf=22,
            )

    visible_samples = [sample for sample in samples if sample.candidate_found and sample.top_height_cm is not None]
    confirmed_holds = live_jump.confirmed_top_jump_holds(top_holds)
    max_hold = max(confirmed_holds, key=lambda hold: hold.height_cm, default=None)
    if expected_alert_record is not None and canonical_alert_sample is None:
        confirmed_height = None
    elif canonical_alert_sample is not None and canonical_alert_sample.top_height_cm is not None:
        confirmed_height = round(float(canonical_alert_sample.top_height_cm), 1)
        canonical_reason = str((expected_alert_record or {}).get("reason") or "")
        if canonical_alert_record is not None and not canonical_reason:
            canonical_reason = str(canonical_alert_record.get("reason") or "")
        detector_context = _detector_context_payload(
            detector_loaded_metadata=detector_loaded_metadata,
            backend="source_subtracted"
            if str(getattr(canonical_alert_sample, "source", "")).endswith(".jpg") and source_video is not None
            else "legacy_full_frame",
            path=canonical_reason,
            probability=getattr(canonical_alert_sample, "detector_cat_probability", None),
        )
    else:
        confirmed_height = round(float(max_hold.height_cm), 1) if max_hold is not None else None
    return {
        "max_height_cm": confirmed_height,
        "candidate_frame_count": len(visible_samples),
        "frame_count": len(samples),
        "results_hold_frame_count": results_hold_frame_count,
        "results_hold_seconds": round(results_hold_frame_count / fps, 2) if fps else 0,
        "continuity_hard_case_count": len(hard_cases),
        "continuity_hard_case_queue": str(hard_case_queue_root) if hard_cases else None,
        "vertical_clip": str(vertical_output_path)
        if vertical_output_path is not None and vertical_output_path.exists()
        else None,
        "top_holds": [round(float(hold.height_cm), 1) for hold in confirmed_holds],
        "original_model_overlays": original_model_overlays,
        "canonical_alert_record": canonical_alert_record,
        "canonical_alert_source": str(getattr(canonical_alert_sample, "source", ""))
        if canonical_alert_sample is not None
        else None,
        "detector_context": detector_context,
    }


def _jump_caption(max_height_cm: float) -> str:
    return f"Шер прыгнул на {max_height_cm:.1f} см"


def _review_video_id(recording_dir: Path) -> str:
    resolved = recording_dir.expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:18]
    return f"video.{digest}"


def _review_frame_label(
    max_record: dict[str, Any] | None,
    *,
    chunk_path: Path | None = None,
    fps: float = DEFAULT_REVIEW_FRAME_FPS,
) -> str | None:
    if not isinstance(max_record, dict):
        return None
    try:
        chunk_index = int(max_record["chunk"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        offset_seconds = float(max_record.get("offset_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        offset_seconds = 0.0
    frame_index = (
        _source_frame_index_for_offset(chunk_path, offset_seconds)
        if chunk_path
        else max(
            0,
            int(round(offset_seconds * fps)),
        )
    )
    return f"chunk_{chunk_index:04d}_{frame_index:05d}.jpg"


def _ensure_review_frame_for_link(recording_dir: Path, max_record: dict[str, Any] | None) -> Path | None:
    if not isinstance(max_record, dict):
        return None
    try:
        chunk_index = int(max_record["chunk"])
    except (KeyError, TypeError, ValueError):
        return None
    chunk_path = recording_dir / f"chunk_{chunk_index:04d}.mp4"
    if not chunk_path.exists():
        return None
    frame_rate = _chunk_frame_rate(chunk_path)
    frame_label = _review_frame_label(max_record, chunk_path=chunk_path, fps=frame_rate)
    if not frame_label:
        return None
    output_path = recording_dir / "review_frames" / frame_label
    chunk_glob = f"chunk_{chunk_index:04d}_*.jpg"
    if output_path.exists() and output_path.stat().st_size > 512 and len(list(output_path.parent.glob(chunk_glob))) > 1:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_pattern = output_path.parent / f"chunk_{chunk_index:04d}_%05d.jpg"
    command = [
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
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 512:
        return None
    return output_path


def _review_deep_link(
    recording_dir: Path,
    *,
    max_record: dict[str, Any] | None = None,
    base_url: str = DEFAULT_CAT_PROJECTOR_REVIEW_UI_URL,
) -> str:
    parsed = urlparse(base_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["review_video_id"] = _review_video_id(recording_dir)
    _ensure_review_frame_for_link(recording_dir, max_record)
    frame_rate = None
    chunk_path_for_label: Path | None = None
    if isinstance(max_record, dict):
        try:
            chunk_path = recording_dir / f"chunk_{int(max_record['chunk']):04d}.mp4"
            chunk_path_for_label = chunk_path if chunk_path.exists() else None
            frame_rate = _chunk_frame_rate(chunk_path if chunk_path.exists() else None)
        except (KeyError, TypeError, ValueError):
            frame_rate = None
    frame_label = _review_frame_label(
        max_record,
        chunk_path=chunk_path_for_label,
        fps=frame_rate or DEFAULT_REVIEW_FRAME_FPS,
    )
    if frame_label:
        query["review_frame_label"] = frame_label
    return urlunparse(parsed._replace(query=urlencode(query)))


def _caption_with_review_link(caption: str, review_url: str | None) -> str:
    if not review_url:
        return caption
    return f"{caption}\nРазметить: {review_url}"


def _camera_human_skip_payload(
    *,
    recording_dir: Path,
    manifest: dict[str, Any],
    human_check: CameraHumanCheck,
    live: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    action_key = "telegram_live_action" if live else "telegram_action"
    action_value = "camera_human_detected" if human_check.detected else "human_check_unavailable"
    timestamp_key = "last_scan_at" if live else "sent_at"
    payload = {
        "kind": "cat_projector_telegram_live_notification_skipped"
        if live
        else "cat_projector_telegram_notification_skipped",
        "camera_human_check": human_check.to_json(),
        "ha_session_id": manifest.get("ha_session_id"),
        "recording_dir": str(recording_dir),
        "selection_method": action_value,
        timestamp_key: (now or datetime.now().astimezone()).isoformat(timespec="seconds"),
        action_key: action_value,
        "video_url": manifest.get("video_url"),
    }
    return payload


def _send_telegram_notification(
    *,
    recording_dir: Path,
    manifest: dict[str, Any],
    clips: list[Path],
    caption: str | None = None,
) -> DeliveryResult:
    del recording_dir, manifest
    return deliver_notification(
        caption,
        source="cat_projector_learning_recorder",
        delivery_class="user_facing",
        reason="cat_projector_verified_jump_clip",
        media=clips,
    )


def _send_telegram_live_notification(
    *,
    recording_dir: Path,
    manifest: dict[str, Any],
    clip: Path,
    caption: str | None = None,
) -> DeliveryResult:
    return deliver_notification(
        caption,
        source="cat_projector_learning_recorder",
        delivery_class="user_facing",
        reason="cat_projector_live_verified_jump_clip",
        media=[clip],
    )


def _delivery_result_payload(result: DeliveryResult) -> dict[str, Any]:
    return {
        "audited": result.audited,
        "delivered": result.delivered,
        "delivery_class": result.delivery_class,
        "reason": result.reason,
        "returncode": result.returncode,
        "source": result.source,
        "stderr": result.stderr,
        "stdout": result.stdout,
    }


def _coerce_delivery_result(result: Any) -> DeliveryResult:
    if isinstance(result, DeliveryResult):
        return result
    return DeliveryResult(
        delivered=True,
        audited=False,
        source="cat_projector_learning_recorder",
        delivery_class="user_facing",
        reason="telegram_sent",
    )


def _live_sent_payload_from_marker(
    marker: dict[str, Any],
    *,
    now: datetime,
    delivery_result: DeliveryResult,
) -> dict[str, Any]:
    payload = dict(marker)
    payload.update(
        {
            "kind": "cat_projector_telegram_live_notification_sent",
            "last_sent_at": now.isoformat(timespec="seconds"),
            "sent_at": now.isoformat(timespec="seconds"),
            "telegram_live_action": "sent",
            "delivery_result": _delivery_result_payload(delivery_result),
        }
    )
    payload.pop("last_send_attempt_at", None)
    return payload


def _retry_live_send_failed_marker(
    root: Path,
    marker_path: Path,
    marker: dict[str, Any],
    *,
    now: datetime,
    scan_cooldown_seconds: int,
    recording_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    last_attempt = _parse_iso_datetime(str(marker.get("last_send_attempt_at") or marker.get("last_scan_at") or ""))
    if last_attempt is not None:
        age_seconds = (now - last_attempt).total_seconds()
        if age_seconds < scan_cooldown_seconds:
            return {
                "telegram_live_action": "scan_cooldown",
                "previous_telegram_live_action": "send_failed",
                "retry_allowed": False,
                "age_seconds": round(age_seconds, 1),
                "recording_dir": str(recording_dir),
            }

    clip = Path(str(marker.get("clip") or ""))
    caption = str(marker.get("caption") or "")
    if not clip.exists():
        payload = dict(marker)
        payload.update(
            {
                "last_scan_at": now.isoformat(timespec="seconds"),
                "telegram_live_action": "send_failed_clip_missing",
                "delivery_result": {
                    "delivered": False,
                    "reason": "send_failed_clip_missing",
                    "stderr": f"clip does not exist: {clip}",
                },
            }
        )
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {
            "telegram_live_action": "send_failed_clip_missing",
            "clip": str(clip),
            "recording_dir": str(recording_dir),
        }

    result = _coerce_delivery_result(
        _send_telegram_live_notification(recording_dir=recording_dir, manifest=manifest, clip=clip, caption=caption)
    )
    if result.delivered:
        payload = _live_sent_payload_from_marker(marker, now=now, delivery_result=result)
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {
            "telegram_live_action": "sent",
            "recording_dir": str(recording_dir),
            "caption": caption,
            "clip": str(clip),
            "cat_food_reward_action": (payload.get("cat_food_reward") or {}).get("cat_food_reward_action")
            if isinstance(payload.get("cat_food_reward"), dict)
            else None,
            "max_height_cm": payload.get("max_height_cm"),
        }

    payload = dict(marker)
    payload.update(
        {
            "last_send_attempt_at": now.isoformat(timespec="seconds"),
            "last_scan_at": now.isoformat(timespec="seconds"),
            "telegram_live_action": "send_failed",
            "delivery_result": _delivery_result_payload(result),
        }
    )
    _json_dump(marker_path, payload)
    _append_jsonl(root / "sessions.jsonl", payload)
    return {
        "telegram_live_action": "send_failed",
        "recording_dir": str(recording_dir),
        "caption": caption,
        "clip": str(clip),
        "delivery_result": _delivery_result_payload(result),
    }


def _parse_service_target(entity_id: str) -> tuple[str, str]:
    domain, _, _name = entity_id.partition(".")
    if not domain:
        raise ValueError(f"invalid Home Assistant entity id for reward: {entity_id!r}")
    if domain == "button":
        return "button", "press"
    raise ValueError(f"unsupported cat food reward entity domain: {domain!r}")


def _maybe_dispense_cat_food_reward(
    root: Path,
    *,
    recording_dir: Path,
    manifest: dict[str, Any],
    height_cm: float,
    source: str,
    ha_url: str,
    min_height_cm: float = DEFAULT_CAT_FOOD_REWARD_MIN_HEIGHT_CM,
    cooldown_seconds: int = DEFAULT_CAT_FOOD_REWARD_COOLDOWN_SECONDS,
    entity_id: str = DEFAULT_CAT_FOOD_REWARD_BUTTON_ENTITY_ID,
) -> dict[str, Any]:
    now = datetime.now().astimezone()
    session_id = str(manifest.get("ha_session_id") or "")
    marker_path = recording_dir / "cat_food_reward.json"
    state_path = root / "cat_food_reward_state.json"
    base_payload: dict[str, Any] = {
        "kind": "cat_projector_cat_food_reward",
        "cat_food_reward_action": "not_applicable",
        "height_cm": round(float(height_cm), 1),
        "min_height_cm": min_height_cm,
        "recording_dir": str(recording_dir),
        "ha_session_id": session_id,
        "source": source,
        "entity_id": entity_id,
        "checked_at": now.isoformat(timespec="seconds"),
    }

    if not entity_id:
        return base_payload | {"cat_food_reward_action": "not_configured"}

    if height_cm < min_height_cm:
        return base_payload | {"cat_food_reward_action": "below_threshold"}

    if marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            marker = {}
        return base_payload | {
            "cat_food_reward_action": "already_rewarded_recording",
            "previous_reward": marker,
        }

    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    last_session_id = str(state.get("last_ha_session_id") or "")
    if session_id and last_session_id == session_id:
        payload = base_payload | {
            "cat_food_reward_action": "already_rewarded_session",
            "last_dispensed_at": state.get("last_dispensed_at"),
        }
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return payload

    last_dispensed_at = _parse_iso_datetime(str(state.get("last_dispensed_at") or ""))
    if last_dispensed_at is not None:
        age_seconds = (now - last_dispensed_at).total_seconds()
        if age_seconds < cooldown_seconds:
            payload = base_payload | {
                "cat_food_reward_action": "cooldown",
                "age_seconds": round(age_seconds, 1),
                "cooldown_seconds": cooldown_seconds,
                "last_dispensed_at": last_dispensed_at.isoformat(timespec="seconds"),
            }
            _json_dump(marker_path, payload)
            _append_jsonl(root / "sessions.jsonl", payload)
            return payload

    domain, service = _parse_service_target(entity_id)
    try:
        token = _load_ha_token()
        service_result = _ha_call_service(ha_url, token, domain, service, {"entity_id": entity_id})
    except Exception as exc:
        payload = base_payload | {
            "cat_food_reward_action": "dispense_failed",
            "cooldown_seconds": cooldown_seconds,
            "ha_service": f"{domain}.{service}",
            "error": str(exc),
        }
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return payload

    payload = base_payload | {
        "cat_food_reward_action": "dispensed",
        "dispensed_at": now.isoformat(timespec="seconds"),
        "cooldown_seconds": cooldown_seconds,
        "ha_service": f"{domain}.{service}",
        "ha_service_result": service_result,
    }
    state_payload = {
        "last_dispensed_at": now.isoformat(timespec="seconds"),
        "last_ha_session_id": session_id,
        "last_height_cm": round(float(height_cm), 1),
        "last_recording_dir": str(recording_dir),
        "last_source": source,
        "entity_id": entity_id,
        "cooldown_seconds": cooldown_seconds,
    }
    _json_dump(state_path, state_payload)
    _json_dump(marker_path, payload)
    _append_jsonl(root / "sessions.jsonl", payload)
    return payload


def _maybe_send_telegram_notification(
    root: Path,
    *,
    recordings_root: Path | None = None,
    max_age_seconds: int,
    window_chunks: int,
    max_clips: int,
    ha_url: str = DEFAULT_HA_URL,
) -> dict[str, Any]:
    del max_clips
    manifest_path = _latest_manifest_path(recordings_root or root / "recordings")
    if manifest_path is None:
        return {"telegram_action": "no_manifest"}
    recording_dir = manifest_path.parent
    marker_path = recording_dir / "telegram_notification.json"
    if marker_path.exists():
        return {"telegram_action": "already_sent", "recording_dir": str(recording_dir)}
    verified_path = recording_dir / "verified.json"
    if not verified_path.exists():
        return {"telegram_action": "not_verified", "recording_dir": str(recording_dir)}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    created_at = _parse_iso_datetime(str(manifest.get("created_at") or ""))
    if created_at is not None:
        age_seconds = (datetime.now().astimezone() - created_at).total_seconds()
        if age_seconds > max_age_seconds:
            return {
                "telegram_action": "too_old",
                "age_seconds": round(age_seconds, 1),
                "recording_dir": str(recording_dir),
            }

    chunks = [path for path in sorted(recording_dir.glob("chunk_*.mp4")) if ".part." not in path.name]
    if not chunks:
        return {"telegram_action": "no_chunks", "recording_dir": str(recording_dir)}

    outbox_dir = recording_dir / "telegram_notification"
    scan_chunks = chunks[-DEFAULT_TELEGRAM_FINAL_SCAN_MAX_CHUNKS:]
    highlight = _select_jump_highlight_clip_window(
        scan_chunks,
        window_chunks=window_chunks,
        min_jump_height_cm=DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
    )
    if highlight is None:
        payload = {
            "kind": "cat_projector_telegram_notification_skipped",
            "recording_dir": str(recording_dir),
            "sent_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "ha_session_id": manifest.get("ha_session_id"),
            "min_jump_height_cm": DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
            "scan_chunk_count": len(scan_chunks),
            "scan_chunk_limit": DEFAULT_TELEGRAM_FINAL_SCAN_MAX_CHUNKS,
            "selection_method": "no_jump_highlight",
            "telegram_action": "no_jump_highlight",
            "video_url": manifest.get("video_url"),
        }
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {"telegram_action": "no_jump_highlight", "recording_dir": str(recording_dir)}

    human_check = detect_camera_human_presence(highlight.chunks)
    if human_check.should_skip:
        payload = _camera_human_skip_payload(
            recording_dir=recording_dir,
            manifest=manifest,
            human_check=human_check,
            live=False,
        )
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {
            "telegram_action": payload["telegram_action"],
            "recording_dir": str(recording_dir),
            "camera_human_check": human_check.to_json(),
        }

    clip_path = outbox_dir / "sher_jump_highlight.mp4"
    vertical_clip_path = outbox_dir / "sher_jump_highlight_vertical.mp4"
    enhanced_vertical_clip_path = outbox_dir / "sher_jump_highlight_tiktok.mp4"
    render_stats = _build_annotated_jump_clip(
        highlight.chunks,
        clip_path,
        vertical_output_path=vertical_clip_path,
        training_queue_root=root / "detector-training-queue",
        min_jump_height_cm=DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
        expected_alert_record=highlight.max_record,
    )
    rendered_max_height = render_stats.get("max_height_cm")
    if rendered_max_height is None or float(rendered_max_height) < DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM:
        skip_reason = "no_confirmed_rendered_jump" if rendered_max_height is None else "rendered_jump_below_threshold"
        payload = {
            "kind": "cat_projector_telegram_notification_skipped",
            "recording_dir": str(recording_dir),
            "sent_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "ha_session_id": manifest.get("ha_session_id"),
            "jump_highlight": highlight.max_record,
            "jump_highlight_candidate_count": highlight.candidate_count,
            "min_jump_height_cm": DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
            "render_stats": render_stats,
            "selection_method": skip_reason,
            "telegram_action": skip_reason,
            "video_url": manifest.get("video_url"),
        }
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {"telegram_action": skip_reason, "recording_dir": str(recording_dir)}

    max_height_cm = float(rendered_max_height)
    canonical_alert_record = render_stats.get("canonical_alert_record") or highlight.max_record
    reward_payload = _maybe_dispense_cat_food_reward(
        root,
        recording_dir=recording_dir,
        manifest=manifest,
        height_cm=max_height_cm,
        source="final_telegram_notification",
        ha_url=ha_url,
    )
    jump_caption = _jump_caption(max_height_cm)
    review_url = _review_deep_link(recording_dir, max_record=canonical_alert_record)
    caption = _caption_with_review_link(jump_caption, review_url)
    if vertical_clip_path.exists():
        try:
            render_stats["enhanced_vertical_clip"] = _enhance_vertical_clip_for_social(
                vertical_clip_path,
                enhanced_vertical_clip_path,
                height_cm=max_height_cm,
                title=f"Sher jump {max_height_cm:.1f} cm",
                comment=(
                    f"Sher jump highlight {max_height_cm:.1f} cm. "
                    "Measured on original source frames; TikTok-ready render enhanced after detection."
                ),
            )
        except Exception as exc:
            render_stats["enhanced_vertical_clip_error"] = str(exc)
    clips = [path for path in (enhanced_vertical_clip_path, vertical_clip_path, clip_path) if path.exists()]
    delivery_result = _coerce_delivery_result(
        _send_telegram_notification(recording_dir=recording_dir, manifest=manifest, clips=clips, caption=caption)
    )
    payload = {
        "kind": "cat_projector_telegram_notification_sent",
        "recording_dir": str(recording_dir),
        "sent_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "caption": caption,
        "jump_caption": jump_caption,
        "review_url": review_url,
        "clips": [str(clip) for clip in clips],
        "ha_session_id": manifest.get("ha_session_id"),
        "jump_highlight": canonical_alert_record,
        "selected_jump_highlight": highlight.max_record,
        "jump_highlight_candidate_count": highlight.candidate_count,
        "max_height_cm": max_height_cm,
        "min_jump_height_cm": DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
        "render_stats": render_stats,
        "detector_context": render_stats.get("detector_context"),
        "cat_food_reward": reward_payload,
        "scan_chunk_count": len(scan_chunks),
        "scan_chunk_limit": DEFAULT_TELEGRAM_FINAL_SCAN_MAX_CHUNKS,
        "selection_method": "calibrated_jump_highlight",
        "source_window": [path.name for path in highlight.chunks],
        "delivery_result": _delivery_result_payload(delivery_result),
        "video_url": manifest.get("video_url"),
    }
    if not delivery_result.delivered:
        reason = str(delivery_result.reason or delivery_result.stderr or "")
        suppressed_duplicate = "duplicate" in reason
        payload.update(
            {
                "kind": "cat_projector_telegram_notification_duplicate_suppressed"
                if suppressed_duplicate
                else "cat_projector_telegram_notification_send_failed",
                "telegram_action": "duplicate_suppressed" if suppressed_duplicate else "send_failed",
            }
        )
        payload.pop("sent_at", None)
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {
            "telegram_action": payload["telegram_action"],
            "recording_dir": str(recording_dir),
            "caption": caption,
            "jump_caption": jump_caption,
            "review_url": review_url,
            "clips": [str(clip) for clip in clips],
            "cat_food_reward_action": reward_payload.get("cat_food_reward_action"),
            "delivery_result": _delivery_result_payload(delivery_result),
            "max_height_cm": max_height_cm,
            "selection_method": "calibrated_jump_highlight",
            "render_stats": render_stats,
        }
    _json_dump(marker_path, payload)
    _append_jsonl(root / "sessions.jsonl", payload)
    return {
        "telegram_action": "sent",
        "recording_dir": str(recording_dir),
        "caption": caption,
        "jump_caption": jump_caption,
        "review_url": review_url,
        "clips": [str(clip) for clip in clips],
        "cat_food_reward_action": reward_payload.get("cat_food_reward_action"),
        "max_height_cm": max_height_cm,
        "selection_method": "calibrated_jump_highlight",
        "render_stats": render_stats,
    }


def _maybe_send_telegram_notification_guarded(
    root: Path,
    *,
    recordings_root: Path,
    max_age_seconds: int,
    window_chunks: int,
    max_clips: int,
    ha_url: str,
    retry_cooldown_seconds: int = DEFAULT_TELEGRAM_FINAL_ERROR_COOLDOWN_SECONDS,
) -> dict[str, Any]:
    """Keep optional final-notification work from failing recorder reconciliation."""

    manifest_path = _latest_manifest_path(recordings_root)
    recording_dir = manifest_path.parent if manifest_path is not None else None
    error_marker_path = recording_dir / "telegram_notification_error.json" if recording_dir is not None else None
    notification_marker_path = recording_dir / "telegram_notification.json" if recording_dir is not None else None
    now = datetime.now().astimezone()

    if (
        error_marker_path is not None
        and error_marker_path.exists()
        and notification_marker_path is not None
        and not notification_marker_path.exists()
    ):
        try:
            error_marker = json.loads(error_marker_path.read_text(encoding="utf-8"))
            last_attempt = _parse_iso_datetime(str(error_marker.get("last_attempt_at") or ""))
        except Exception:
            error_marker = {}
            last_attempt = None
        if last_attempt is not None:
            age_seconds = (now - last_attempt).total_seconds()
            if 0 <= age_seconds < retry_cooldown_seconds:
                return {
                    "telegram_action": "postprocess_retry_cooldown",
                    "previous_telegram_action": str(error_marker.get("telegram_action") or "postprocess_failed"),
                    "retry_allowed": False,
                    "age_seconds": round(age_seconds, 1),
                    "retry_cooldown_seconds": retry_cooldown_seconds,
                    "recording_dir": str(recording_dir),
                }

    try:
        payload = _maybe_send_telegram_notification(
            root,
            recordings_root=recordings_root,
            max_age_seconds=max_age_seconds,
            window_chunks=window_chunks,
            max_clips=max_clips,
            ha_url=ha_url,
        )
    except Exception as exc:
        failure = {
            "kind": "cat_projector_telegram_notification_postprocess_failed",
            "telegram_action": "postprocess_failed",
            "last_attempt_at": now.isoformat(timespec="seconds"),
            "retry_cooldown_seconds": retry_cooldown_seconds,
            "recording_dir": str(recording_dir) if recording_dir is not None else None,
            "error": str(exc),
        }
        marker_write_error: str | None = None
        try:
            if error_marker_path is not None:
                _json_dump(error_marker_path, failure)
            _append_jsonl(root / "sessions.jsonl", failure)
        except OSError as marker_exc:
            marker_write_error = str(marker_exc)
        if marker_write_error:
            failure["marker_write_error"] = marker_write_error
        return failure

    if error_marker_path is not None:
        with contextlib.suppress(OSError):
            error_marker_path.unlink(missing_ok=True)
    return payload


def _maybe_send_live_telegram_notification(
    root: Path,
    *,
    recordings_root: Path | None = None,
    ha_session_id: str,
    window_chunks: int,
    cooldown_seconds: int,
    scan_cooldown_seconds: int = DEFAULT_TELEGRAM_LIVE_SCAN_COOLDOWN_SECONDS,
    ha_url: str = DEFAULT_HA_URL,
) -> dict[str, Any]:
    manifest_path = _latest_manifest_path(recordings_root or root / "recordings")
    if manifest_path is None:
        return {"telegram_live_action": "no_manifest"}
    recording_dir = manifest_path.parent
    verified_path = recording_dir / "verified.json"
    if not verified_path.exists():
        return {"telegram_live_action": "not_verified", "recording_dir": str(recording_dir)}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("ha_session_id") or "") != ha_session_id:
        return {
            "telegram_live_action": "session_mismatch",
            "ha_session_id": ha_session_id,
            "recording_dir": str(recording_dir),
        }

    marker_path = recording_dir / "telegram_live_notification.json"
    now = datetime.now().astimezone()
    if marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            action = str(marker.get("telegram_live_action") or ("sent" if marker.get("clip") else ""))
            last_sent = _parse_iso_datetime(str(marker.get("last_sent_at") or marker.get("sent_at") or ""))
            last_scan = _parse_iso_datetime(str(marker.get("last_scan_at") or ""))
        except Exception:
            action = ""
            last_sent = None
            last_scan = None
        if action == "send_failed":
            return _retry_live_send_failed_marker(
                root,
                marker_path,
                marker,
                now=now,
                scan_cooldown_seconds=scan_cooldown_seconds,
                recording_dir=recording_dir,
                manifest=manifest,
            )
        policy_available, policy_payload = _cat_tv_play_live_marker_decision(
            marker,
            now=now,
            sent_cooldown_seconds=cooldown_seconds,
            scan_cooldown_seconds=scan_cooldown_seconds,
        )
        if policy_payload is not None:
            policy_payload["recording_dir"] = str(recording_dir)
            return policy_payload
        if not policy_available and action in {"camera_human_detected", "human_check_unavailable"}:
            return {
                "telegram_live_action": action,
                "recording_dir": str(recording_dir),
                "already_marked": True,
            }
        if action == "sent" and last_sent is not None:
            age_seconds = (now - last_sent).total_seconds()
            if age_seconds < cooldown_seconds:
                return {
                    "telegram_live_action": "cooldown",
                    "age_seconds": round(age_seconds, 1),
                    "recording_dir": str(recording_dir),
                }
        if action in {"no_curated_highlight", "no_jump_highlight"} and last_scan is not None:
            age_seconds = (now - last_scan).total_seconds()
            if age_seconds < scan_cooldown_seconds:
                return {
                    "telegram_live_action": "scan_cooldown",
                    "age_seconds": round(age_seconds, 1),
                    "recording_dir": str(recording_dir),
                }

    chunks = [path for path in sorted(recording_dir.glob("chunk_*.mp4")) if ".part." not in path.name]
    span = max(1, window_chunks)
    if len(chunks) < span:
        return {
            "telegram_live_action": "not_enough_chunks",
            "chunk_count": len(chunks),
            "recording_dir": str(recording_dir),
        }

    highlight = _select_jump_highlight_clip_window(
        chunks[-max(span * 4, span) :],
        window_chunks=span,
        min_jump_height_cm=DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
    )
    if highlight is None:
        payload = {
            "kind": "cat_projector_telegram_live_notification_skipped",
            "candidate_count": 0,
            "ha_session_id": manifest.get("ha_session_id"),
            "min_jump_height_cm": DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
            "recording_dir": str(recording_dir),
            "selection_method": "no_jump_highlight",
            "last_scan_at": now.isoformat(timespec="seconds"),
            "telegram_live_action": "no_jump_highlight",
            "video_url": manifest.get("video_url"),
        }
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {
            "telegram_live_action": "no_jump_highlight",
            "candidate_count": 0,
            "recording_dir": str(recording_dir),
        }

    human_check = detect_camera_human_presence(highlight.chunks)
    if human_check.should_skip:
        payload = _camera_human_skip_payload(
            recording_dir=recording_dir,
            manifest=manifest,
            human_check=human_check,
            live=True,
            now=now,
        )
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {
            "telegram_live_action": payload["telegram_live_action"],
            "recording_dir": str(recording_dir),
            "camera_human_check": human_check.to_json(),
        }

    clip_path = recording_dir / "telegram_live_notification" / f"live_{now.strftime('%Y%m%dT%H%M%S')}.mp4"
    vertical_clip_path = (
        recording_dir / "telegram_live_notification" / f"live_{now.strftime('%Y%m%dT%H%M%S')}_vertical.mp4"
    )
    render_stats = _build_annotated_jump_clip(
        highlight.chunks,
        clip_path,
        vertical_output_path=vertical_clip_path,
        training_queue_root=root / "detector-training-queue",
        min_jump_height_cm=DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
        expected_alert_record=highlight.max_record,
    )
    rendered_max_height = render_stats.get("max_height_cm")
    if rendered_max_height is None or float(rendered_max_height) < DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM:
        skip_reason = "no_confirmed_rendered_jump" if rendered_max_height is None else "rendered_jump_below_threshold"
        payload = {
            "kind": "cat_projector_telegram_live_notification_skipped",
            "candidate_count": highlight.candidate_count,
            "ha_session_id": manifest.get("ha_session_id"),
            "jump_highlight": highlight.max_record,
            "jump_highlight_candidate_count": highlight.candidate_count,
            "min_jump_height_cm": DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
            "render_stats": render_stats,
            "recording_dir": str(recording_dir),
            "selection_method": skip_reason,
            "last_scan_at": now.isoformat(timespec="seconds"),
            "telegram_live_action": skip_reason,
            "video_url": manifest.get("video_url"),
        }
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {
            "telegram_live_action": skip_reason,
            "candidate_count": highlight.candidate_count,
            "recording_dir": str(recording_dir),
        }

    max_height_cm = float(rendered_max_height)
    canonical_alert_record = render_stats.get("canonical_alert_record") or highlight.max_record
    reward_payload = _maybe_dispense_cat_food_reward(
        root,
        recording_dir=recording_dir,
        manifest=manifest,
        height_cm=max_height_cm,
        source="live_telegram_notification",
        ha_url=ha_url,
    )
    jump_caption = _jump_caption(max_height_cm)
    review_url = _review_deep_link(recording_dir, max_record=canonical_alert_record)
    caption = _caption_with_review_link(jump_caption, review_url)
    enhanced_vertical_clip_path = (
        recording_dir / "telegram_live_notification" / f"live_{now.strftime('%Y%m%dT%H%M%S')}_tiktok.mp4"
    )
    if vertical_clip_path.exists():
        try:
            render_stats["enhanced_vertical_clip"] = _enhance_vertical_clip_for_social(
                vertical_clip_path,
                enhanced_vertical_clip_path,
                height_cm=max_height_cm,
                title=f"Sher live jump {max_height_cm:.1f} cm",
                comment=(
                    f"Sher live jump highlight {max_height_cm:.1f} cm. "
                    "Measured on original source frames; TikTok-ready render enhanced after detection."
                ),
            )
        except Exception as exc:
            render_stats["enhanced_vertical_clip_error"] = str(exc)
    live_clip = (
        enhanced_vertical_clip_path
        if enhanced_vertical_clip_path.exists()
        else vertical_clip_path
        if vertical_clip_path.exists()
        else clip_path
    )
    payload = {
        "kind": "cat_projector_telegram_live_notification_sent",
        "caption": caption,
        "jump_caption": jump_caption,
        "review_url": review_url,
        "clip": str(live_clip),
        "horizontal_clip": str(clip_path),
        "vertical_clip": str(vertical_clip_path) if vertical_clip_path.exists() else None,
        "enhanced_vertical_clip": str(enhanced_vertical_clip_path) if enhanced_vertical_clip_path.exists() else None,
        "ha_session_id": manifest.get("ha_session_id"),
        "jump_highlight": canonical_alert_record,
        "selected_jump_highlight": highlight.max_record,
        "jump_highlight_candidate_count": highlight.candidate_count,
        "max_height_cm": max_height_cm,
        "min_jump_height_cm": DEFAULT_TELEGRAM_JUMP_MIN_HEIGHT_CM,
        "render_stats": render_stats,
        "detector_context": render_stats.get("detector_context"),
        "cat_food_reward": reward_payload,
        "recording_dir": str(recording_dir),
        "sent_at": now.isoformat(timespec="seconds"),
        "last_sent_at": now.isoformat(timespec="seconds"),
        "selection_method": "calibrated_jump_highlight",
        "source_chunk_first": highlight.chunks[0].name,
        "source_chunk_last": highlight.chunks[-1].name,
        "telegram_live_action": "sent",
        "video_url": manifest.get("video_url"),
    }
    delivery_result = _coerce_delivery_result(
        _send_telegram_live_notification(
            recording_dir=recording_dir,
            manifest=manifest,
            clip=live_clip,
            caption=caption,
        )
    )
    payload["delivery_result"] = _delivery_result_payload(delivery_result)
    if not delivery_result.delivered:
        payload.update(
            {
                "kind": "cat_projector_telegram_live_notification_send_failed",
                "last_scan_at": now.isoformat(timespec="seconds"),
                "last_send_attempt_at": now.isoformat(timespec="seconds"),
                "telegram_live_action": "send_failed",
            }
        )
        payload.pop("sent_at", None)
        payload.pop("last_sent_at", None)
        _json_dump(marker_path, payload)
        _append_jsonl(root / "sessions.jsonl", payload)
        return {
            "telegram_live_action": "send_failed",
            "recording_dir": str(recording_dir),
            "caption": caption,
            "clip": str(live_clip),
            "cat_food_reward_action": reward_payload.get("cat_food_reward_action"),
            "delivery_result": _delivery_result_payload(delivery_result),
            "max_height_cm": max_height_cm,
        }
    _json_dump(marker_path, payload)
    _append_jsonl(root / "sessions.jsonl", payload)
    return {
        "telegram_live_action": "sent",
        "recording_dir": str(recording_dir),
        "caption": caption,
        "jump_caption": jump_caption,
        "review_url": review_url,
        "clip": str(live_clip),
        "enhanced_vertical_clip": str(enhanced_vertical_clip_path) if enhanced_vertical_clip_path.exists() else None,
        "cat_food_reward_action": reward_payload.get("cat_food_reward_action"),
        "render_stats": render_stats,
        "max_height_cm": max_height_cm,
    }


def record(args: argparse.Namespace) -> int:
    recording_dir = args.recording_dir.expanduser()
    recording_dir.mkdir(parents=True, exist_ok=True)
    log_path = recording_dir / "ffmpeg.log"
    deadline = time.monotonic() + args.duration_seconds
    chunk_index = 0
    while time.monotonic() < deadline:
        output_path = recording_dir / f"chunk_{chunk_index:04d}.mp4"
        partial_path = recording_dir / f"chunk_{chunk_index:04d}.part.mp4"
        partial_path.unlink(missing_ok=True)
        command = _ffmpeg_chunk_command(
            camera_url=args.camera_url,
            output_path=partial_path,
            segment_seconds=args.segment_seconds,
        )
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with log_path.open("ab") as log_handle:
            log_handle.write(f"\n[{started_at}] {' '.join(command)}\n".encode())
            result = subprocess.run(command, stdout=log_handle, stderr=log_handle, check=False)
            log_handle.write(f"[exit {result.returncode}] {partial_path}\n".encode())
        frames = _ffprobe_frame_count(partial_path)
        size_bytes = partial_path.stat().st_size if partial_path.exists() else 0
        effective_fps = _chunk_effective_frame_rate(partial_path, frames=frames)
        reject_reason = None
        if frames <= 0:
            reject_reason = "no_readable_frames"
        elif size_bytes < 4096:
            reject_reason = f"too_small:{size_bytes}<4096"
        elif effective_fps < DEFAULT_RECORDER_MIN_CHUNK_EFFECTIVE_FPS:
            reject_reason = f"low_effective_fps:{effective_fps:.3f}<{DEFAULT_RECORDER_MIN_CHUNK_EFFECTIVE_FPS:.3f}"
        finalized = reject_reason is None
        if finalized:
            partial_path.replace(output_path)
        elif partial_path.exists():
            bad_dir = recording_dir / "bad_chunks"
            bad_dir.mkdir(exist_ok=True)
            partial_path.replace(bad_dir / f"chunk_{chunk_index:04d}.bad.mp4")
        _append_jsonl(
            recording_dir / "chunks.jsonl",
            {
                "chunk_index": chunk_index,
                "effective_fps": round(effective_fps, 3),
                "finalized": finalized,
                "frames": frames,
                "path": str(output_path),
                "partial_path": str(partial_path),
                "reject_reason": reject_reason,
                "returncode": result.returncode,
                "size_bytes": output_path.stat().st_size if output_path.exists() else size_bytes,
                "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
        chunk_index += 1
    return 0


def start(args: argparse.Namespace) -> int:
    root = args.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    recordings_root = _recordings_root(args, root)
    recordings_root.mkdir(parents=True, exist_ok=True)
    pid_path = root / "current_recorder.pid"
    if args.replace_existing:
        _stop_existing(pid_path)
    elif (pid := _read_pid(pid_path)) and _pid_is_running(pid):
        raise RuntimeError(f"recorder already running as pid {pid}; pass --replace-existing to rotate it")

    ha_session = _read_ha_session(args.ha_url)
    session_id = ha_session.session_id or f"manual-{_now_compact()}"
    safe_session_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in session_id)
    recording_dir = recordings_root / f"{_now_compact()}_{safe_session_id}"
    recording_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = recording_dir / "startup_projector_camera.jpg"
    _capture_snapshot(args.snapshot_url, snapshot_path)

    log_path = recording_dir / "ffmpeg.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "record",
        "--recording-dir",
        str(recording_dir),
        "--camera-url",
        args.camera_url,
        "--duration-seconds",
        str(args.duration_seconds),
        "--segment-seconds",
        str(args.segment_seconds),
    ]
    log_handle = (recording_dir / "recorder.log").open("ab")
    process = subprocess.Popen(command, stdout=log_handle, stderr=log_handle, start_new_session=True)
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")

    manifest = {
        "camera_url": args.camera_url,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_seconds": args.duration_seconds,
        "recorder_command": command,
        "ffmpeg_log_path": str(log_path),
        "frigate_event_id": ha_session.frigate_event_id,
        "ha_renderer_state_at_start": ha_session.renderer_state,
        "ha_session_id": ha_session.session_id,
        "ha_session_state_at_start": ha_session.session_state,
        "pid": process.pid,
        "recording_dir": str(recording_dir),
        "segment_seconds": args.segment_seconds,
        "snapshot_path": str(snapshot_path),
        "video_url": ha_session.video_url,
    }
    _json_dump(recording_dir / "manifest.json", manifest)
    _append_jsonl(root / "sessions.jsonl", manifest | {"kind": "projector_camera_recording_started"})

    verified_chunk = _verify_first_chunk(recording_dir)
    verified = {
        "kind": "projector_camera_recording_verified",
        "recording_dir": str(recording_dir),
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "verified_chunk": str(verified_chunk),
        "verified_chunk_bytes": verified_chunk.stat().st_size,
        "verified_chunk_effective_fps": round(_chunk_effective_frame_rate(verified_chunk), 3),
        "verified_chunk_frames": _ffprobe_frame_count(verified_chunk),
    }
    _json_dump(recording_dir / "verified.json", verified)
    _append_jsonl(root / "sessions.jsonl", verified)
    print(json.dumps(verified, ensure_ascii=False, sort_keys=True))
    return 0


def status(args: argparse.Namespace) -> int:
    root = args.root.expanduser()
    recordings_root = _recordings_root(args, root)
    pid_path = root / "current_recorder.pid"
    pid = _read_pid(pid_path)
    payload = {
        "pid": pid,
        "running": bool(pid and _pid_is_running(pid)),
    }
    recordings = sorted(recordings_root.glob("*/manifest.json"))
    if recordings:
        latest_manifest = recordings[-1]
        payload["latest_manifest"] = str(latest_manifest)
        payload["latest_recording_dir"] = str(latest_manifest.parent)
        chunks = [path for path in sorted(latest_manifest.parent.glob("chunk_*.mp4")) if ".part." not in path.name]
        payload["latest_chunks"] = [str(path) for path in chunks[-5:]]
        payload["active_partial_chunks"] = [
            str(path) for path in sorted(latest_manifest.parent.glob("chunk_*.part.mp4"))[-2:]
        ]
        verified_path = latest_manifest.parent / "verified.json"
        payload["latest_verified"] = verified_path.exists()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def stop(args: argparse.Namespace) -> int:
    root = args.root.expanduser()
    _stop_existing(root / "current_recorder.pid")
    print(json.dumps({"stopped": True}, sort_keys=True))
    return 0


def render_publishable_jump(args: argparse.Namespace) -> int:
    metadata = _render_publishable_rescore_jump(
        jump_heights_path=args.jump_heights.expanduser(),
        output_path=args.output.expanduser(),
        preview_path=args.preview.expanduser() if args.preview else None,
        probe_rows_path=args.probe_rows.expanduser() if args.probe_rows else None,
        fps=args.fps,
        min_duration_seconds=args.min_duration_seconds,
        min_source_fps=args.min_source_fps,
        min_jump_count=args.min_jump_count,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


def ensure(args: argparse.Namespace) -> int:
    """Keep direct learning recording aligned with the HA projector session."""

    root = args.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    recordings_root = _recordings_root(args, root)
    pid_path = root / "current_recorder.pid"
    ha_session = _read_ha_session(args.ha_url)
    pid = _read_pid(pid_path)
    running = bool(pid and _pid_is_running(pid))

    if ha_session.session_state != "on":
        if running:
            _stop_existing(pid_path)
        telegram_payload: dict[str, Any] = {}
        if args.telegram_notify:
            telegram_payload = _maybe_send_telegram_notification_guarded(
                root,
                recordings_root=recordings_root,
                max_age_seconds=args.telegram_max_age_seconds,
                window_chunks=args.telegram_window_chunks,
                max_clips=args.telegram_max_clips,
                ha_url=args.ha_url,
            )
        payload = {
            "action": "stopped" if running else "already_stopped",
            "ha_session_state": ha_session.session_state,
            "running": False,
        }
        payload.update(telegram_payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    if running and _recorder_matches_session(recordings_root, ha_session):
        telegram_payload: dict[str, Any] = {}
        if args.telegram_notify:
            telegram_payload = _maybe_send_live_telegram_notification(
                root,
                recordings_root=recordings_root,
                ha_session_id=ha_session.session_id,
                window_chunks=args.telegram_window_chunks,
                cooldown_seconds=args.telegram_live_cooldown_seconds,
                ha_url=args.ha_url,
            )
        payload = {
            "action": "already_recording",
            "ha_session_id": ha_session.session_id,
            "pid": pid,
            "running": True,
        }
        payload.update(telegram_payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    if running:
        _stop_existing(pid_path)

    start_args = argparse.Namespace(
        root=root,
        recordings_root=recordings_root,
        ha_url=args.ha_url,
        snapshot_url=args.snapshot_url,
        camera_url=args.camera_url,
        duration_seconds=args.duration_seconds,
        segment_seconds=args.segment_seconds,
        replace_existing=True,
    )
    return start(start_args)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--recordings-root", type=Path, default=DEFAULT_RECORDINGS_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--ha-url", default=DEFAULT_HA_URL)
    start_parser.add_argument("--snapshot-url", default=DEFAULT_CAMERA_SNAPSHOT_URL)
    start_parser.add_argument("--camera-url", default=DEFAULT_CAMERA_VIDEO_URL)
    start_parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS)
    start_parser.add_argument("--segment-seconds", type=int, default=DEFAULT_SEGMENT_SECONDS)
    start_parser.add_argument("--replace-existing", action="store_true")
    start_parser.set_defaults(func=start)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--recording-dir", type=Path, required=True)
    record_parser.add_argument("--camera-url", default=DEFAULT_CAMERA_VIDEO_URL)
    record_parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS)
    record_parser.add_argument("--segment-seconds", type=int, default=DEFAULT_SEGMENT_SECONDS)
    record_parser.set_defaults(func=record)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=status)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.set_defaults(func=stop)

    render_parser = subparsers.add_parser("render-publishable-jump")
    render_parser.add_argument("--jump-heights", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--preview", type=Path)
    render_parser.add_argument("--probe-rows", type=Path)
    render_parser.add_argument("--fps", type=int, default=DEFAULT_TELEGRAM_ENHANCED_FPS)
    render_parser.add_argument("--min-duration-seconds", type=float, default=8.0)
    render_parser.add_argument("--min-source-fps", type=float, default=DEFAULT_PUBLISHABLE_MIN_SOURCE_FPS)
    render_parser.add_argument("--min-jump-count", type=int, default=DEFAULT_PUBLISHABLE_MIN_JUMP_COUNT)
    render_parser.set_defaults(func=render_publishable_jump)

    ensure_parser = subparsers.add_parser("ensure")
    ensure_parser.add_argument("--ha-url", default=DEFAULT_HA_URL)
    ensure_parser.add_argument("--snapshot-url", default=DEFAULT_CAMERA_SNAPSHOT_URL)
    ensure_parser.add_argument("--camera-url", default=DEFAULT_CAMERA_VIDEO_URL)
    ensure_parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS)
    ensure_parser.add_argument("--segment-seconds", type=int, default=DEFAULT_SEGMENT_SECONDS)
    ensure_parser.add_argument("--telegram-notify", action="store_true")
    ensure_parser.add_argument("--telegram-max-age-seconds", type=int, default=DEFAULT_TELEGRAM_MAX_AGE_SECONDS)
    ensure_parser.add_argument("--telegram-window-chunks", type=int, default=DEFAULT_TELEGRAM_WINDOW_CHUNKS)
    ensure_parser.add_argument("--telegram-max-clips", type=int, default=DEFAULT_TELEGRAM_MAX_CLIPS)
    ensure_parser.add_argument(
        "--telegram-live-cooldown-seconds",
        type=int,
        default=DEFAULT_TELEGRAM_LIVE_COOLDOWN_SECONDS,
    )
    ensure_parser.set_defaults(func=ensure)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
