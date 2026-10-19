#!/usr/bin/env python3
"""Serve a Cat TV source video with live eye-safety blackout overlays."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from email.parser import BytesHeaderParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_projector_safety_module() -> Any:
    path = REPO_ROOT / "custom_components" / "cat_tv_play" / "projector_safety.py"
    spec = importlib.util.spec_from_file_location("cat_tv_play_projector_safety_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load projector_safety module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module




projector_safety = _load_projector_safety_module()
PersonDetection = projector_safety.PersonDetection
SafetyOverlayResult = projector_safety.SafetyOverlayResult
compute_eye_safety_overlay = projector_safety.compute_eye_safety_overlay
render_eye_safety_overlay = projector_safety.render_eye_safety_overlay

DEFAULT_CAMERA_SNAPSHOT_URL = "http://192.168.100.39:8081/shot.jpg"
DEFAULT_PROJECTOR_POLYGON = (
    (40.22, 57.27),
    (937.92, 101.0),
    (908.0, 599.0),
    (48.16, 680.97),
)
DEFAULT_HUMAN_DETECTOR_DIR = Path("~/.openclaw/state/cat-tv-learning/models/opencv-mobilenet-ssd").expanduser()
DEFAULT_HUMAN_DETECTOR_PROTOTXT = DEFAULT_HUMAN_DETECTOR_DIR / "MobileNetSSD_deploy.prototxt"
DEFAULT_HUMAN_DETECTOR_MODEL = DEFAULT_HUMAN_DETECTOR_DIR / "MobileNetSSD_deploy.caffemodel"
DEFAULT_LEARNING_ROOT = Path("~/.openclaw/state/cat-tv-learning").expanduser()
SAFETY_REVIEW_ROOT = DEFAULT_LEARNING_ROOT / "safety-review"
PERSON_CLASS_ID = 15
DEFAULT_SAFETY_NEGATIVE_REVIEW_INTERVAL = 60.0
DEFAULT_SAFETY_NEGATIVE_REVIEW_MAX_FRAMES_PER_DAY = 720
DEFAULT_HA_ACTIVE_ENTITIES = (
    "input_boolean.cat_projector_session",
    "binary_sensor.cat_projector_display_active",
)


class MobileNetPersonDetector:
    """Small local OpenCV DNN person detector."""

    def __init__(self, *, prototxt: Path, model: Path, min_confidence: float) -> None:
        if not prototxt.exists() or not model.exists():
            raise FileNotFoundError(f"missing OpenCV MobileNet SSD files under {prototxt.parent}")
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - optional runtime dependency.
            raise RuntimeError(f"cv2 import failed: {exc}") from exc
        self._cv2 = cv2
        self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
        self.min_confidence = min_confidence

    def detect(self, image: Image.Image) -> list[PersonDetection]:
        frame = np.asarray(image.convert("RGB"))
        height, width = frame.shape[:2]
        blob = self._cv2.dnn.blobFromImage(
            self._cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5,
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        people: list[PersonDetection] = []
        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            class_id = int(detections[0, 0, index, 1])
            if class_id != PERSON_CLASS_ID or confidence < self.min_confidence:
                continue
            x0, y0, x1, y1 = (
                detections[0, 0, index, 3:7] * np.array([width, height, width, height])
            ).astype(float)
            people.append(
                PersonDetection(
                    bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                    confidence=confidence,
                    source="opencv_mobilenet_ssd",
                    debug={"class_id": class_id},
                )
            )
        return people


class UnavailablePersonDetector:
    def __init__(self, error: str) -> None:
        self.error = error

    def detect(self, image: Image.Image) -> list[PersonDetection]:
        del image
        raise RuntimeError(self.error)


class SafetyOverlayState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.payload: dict[str, Any] = {"status": "starting"}
        self.camera_jpeg: bytes | None = None
        self.debug_camera_jpeg: bytes | None = None
        self.last_active_payload: dict[str, Any] | None = None
        self.last_active_camera_jpeg: bytes | None = None
        self.last_active_debug_camera_jpeg: bytes | None = None

    def update(
        self,
        result: SafetyOverlayResult,
        *,
        people: list[PersonDetection],
        source_size: tuple[int, int],
        updated_at: float | None,
        fixed_black_rect: tuple[int, int, int, int] | None = None,
        camera_image: Image.Image | None = None,
        camera_error: str | None = None,
        performance: dict[str, Any] | None = None,
    ) -> None:
        camera_jpeg = _render_camera_jpeg(camera_image)
        debug_camera_jpeg = _render_debug_camera_jpeg(
            camera_image,
            result=result,
            people=people,
        )
        payload = {
            "status": result.status,
            "zone_count": len(result.zones),
            "zones": [asdict(zone) for zone in result.zones],
            "person_count": len(people),
            "people": [asdict(person) for person in people],
            "source_size": [int(source_size[0]), int(source_size[1])],
            "fixed_black_rect": fixed_black_rect,
            "debug": result.debug,
            "performance": performance or {},
            "camera_error": camera_error,
            "updated_at": updated_at,
        }
        with self._lock:
            self.payload = payload
            self.camera_jpeg = camera_jpeg
            self.debug_camera_jpeg = debug_camera_jpeg
            if result.status == "active":
                self.last_active_payload = dict(payload)
                self.last_active_camera_jpeg = camera_jpeg
                self.last_active_debug_camera_jpeg = debug_camera_jpeg

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.payload)

    def debug_camera_snapshot(self) -> bytes | None:
        with self._lock:
            return self.debug_camera_jpeg

    def camera_snapshot(self) -> bytes | None:
        with self._lock:
            return self.camera_jpeg

    def last_active_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self.last_active_payload) if self.last_active_payload is not None else None

    def last_active_debug_camera_snapshot(self) -> bytes | None:
        with self._lock:
            return self.last_active_debug_camera_jpeg

    def last_active_camera_snapshot(self) -> bytes | None:
        with self._lock:
            return self.last_active_camera_jpeg


class OverlayRequestHandler(SimpleHTTPRequestHandler):
    state: SafetyOverlayState

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/status.json":
            payload = json.dumps(self.state.snapshot(), ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/last-active-status.json":
            snapshot = self.state.last_active_snapshot()
            if snapshot is None:
                self.send_error(404, "No active safety overlay has been observed yet")
                return
            payload = json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/debug-camera.jpg":
            payload = self.state.debug_camera_snapshot()
            if payload is None:
                self.send_error(404, "Debug camera frame is not available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/camera.jpg":
            payload = self.state.camera_snapshot()
            if payload is None:
                self.send_error(404, "Camera frame is not available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/last-active-debug-camera.jpg":
            payload = self.state.last_active_debug_camera_snapshot()
            if payload is None:
                self.send_error(404, "No active debug camera frame has been observed yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/last-active-camera.jpg":
            payload = self.state.last_active_camera_snapshot()
            if payload is None:
                self.send_error(404, "No active camera frame has been observed yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()


@dataclass(frozen=True)
class CameraFrame:
    jpeg: bytes
    received_at: float
    received_wall_at: float
    sequence: int | None
    age_ms: float | None
    encode_ms: float | None = None
    connection: int = 0
    read_started_at: float | None = None

    @property
    def evidence_at(self) -> float:
        local_start = self.received_at if self.read_started_at is None else min(self.read_started_at, self.received_at)
        return local_start - (self.age_ms or 0.0) / 1000.0

    @property
    def evidence_wall_at(self) -> float:
        return self.received_wall_at - (self.received_at - self.evidence_at)

    @property
    def key(self) -> tuple[int, int] | None:
        return (self.connection, self.sequence) if self.sequence is not None else None


def _camera_stream_url(snapshot_url: str, explicit_stream_url: str | None) -> str | None:
    if explicit_stream_url:
        return explicit_stream_url
    parsed = urlsplit(snapshot_url)
    if parsed.scheme in {"http", "https"} and parsed.path == "/shot.jpg":
        return parsed._replace(path="/video").geturl()
    return None


def _camera_frame(
    jpeg: bytes, headers: Any, *, connection: int = 0, required: bool = False, read_started_at: float | None = None
) -> CameraFrame:
    sequence_header = headers.get("X-Frame-Sequence")
    age_header = headers.get("X-Frame-Age-Ms")
    if required or sequence_header is not None or age_header is not None:
        if sequence_header is None or age_header is None:
            raise ValueError("camera frame requires X-Frame-Sequence and X-Frame-Age-Ms")
        sequence = int(sequence_header)
        age_ms = float(age_header)
        if sequence < 0 or not math.isfinite(age_ms) or age_ms < 0:
            raise ValueError("invalid camera frame sequence/age")
    else:
        sequence = None
        age_ms = None
    encode_header = headers.get("X-Frame-Encode-Ms")
    encode_ms = float(encode_header) if encode_header is not None else None
    if encode_ms is not None and (not math.isfinite(encode_ms) or encode_ms < 0):
        raise ValueError("invalid camera frame encode age")
    return CameraFrame(jpeg, time.monotonic(), time.time(), sequence, age_ms, encode_ms, connection, read_started_at)


def _read_camera_snapshot(url: str, *, timeout: float = 2.0) -> CameraFrame:
    request = Request(url, headers={"Cache-Control": "no-cache", "Accept-Encoding": "identity"})
    started = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        jpeg = bytearray()
        while True:
            if time.monotonic() - started >= timeout:
                raise TimeoutError("camera snapshot complete-frame deadline exceeded")
            chunk = response.read1(65536)
            if not chunk:
                break
            jpeg.extend(chunk)
            if len(jpeg) > 16 * 1024 * 1024:
                raise ValueError("camera snapshot JPEG too large")
        if time.monotonic() - started >= timeout:
            raise TimeoutError("camera snapshot complete-frame deadline exceeded")
        return _camera_frame(bytes(jpeg), response.headers, read_started_at=started)


def _iter_mjpeg_frames(response: Any, *, timeout: float, connection: int) -> Iterator[CameraFrame]:
    if response.headers.get_content_type() != "multipart/x-mixed-replace":
        raise ValueError("camera stream is not multipart/x-mixed-replace")
    boundary = response.headers.get_param("boundary")
    if not boundary or len(boundary) > 200:
        raise ValueError("camera stream has no valid multipart boundary")
    marker = b"--" + boundary.encode("ascii")
    buffer = bytearray()
    deadline = time.monotonic() + timeout

    def fill() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("camera stream complete-frame deadline exceeded")
        chunk = response.read1(65536)
        if not chunk:
            raise EOFError("camera stream disconnected")
        buffer.extend(chunk)
        if time.monotonic() >= deadline:
            raise TimeoutError("camera stream complete-frame deadline exceeded")

    def line() -> bytes:
        while True:
            end = buffer.find(b"\r\n")
            if end >= 0:
                if end > 8192:
                    raise ValueError("camera stream header line too large")
                value = bytes(buffer[:end])
                del buffer[:end + 2]
                return value
            if len(buffer) > 8192:
                raise ValueError("camera stream header line too large")
            fill()

    while True:
        deadline = time.monotonic() + timeout
        boundary_line = line()
        if boundary_line == b"":
            boundary_line = line()
        if boundary_line != marker:
            raise ValueError("camera stream boundary missing or ended")
        header_lines = bytearray()
        while True:
            header_line = line()
            if not header_line:
                break
            header_lines.extend(header_line + b"\r\n")
            if len(header_lines) > 16384:
                raise ValueError("camera stream headers too large")
        headers = BytesHeaderParser().parsebytes(bytes(header_lines))
        size = int(headers.get("Content-Length", "0"))
        if headers.get_content_type() != "image/jpeg" or not 0 < size <= 16 * 1024 * 1024:
            raise ValueError("camera stream requires a bounded JPEG Content-Length")
        while len(buffer) < size:
            fill()
        jpeg = bytes(buffer[:size])
        del buffer[:size]
        yield _camera_frame(jpeg, headers, connection=connection, required=True, read_started_at=deadline - timeout)


class LatestCameraStream:
    """Drain MJPEG independently of inference; retain one complete JPEG, never a FIFO."""

    def __init__(self, url: str, *, timeout: float, max_age: float, on_error: Callable[[str | None], None] | None = None) -> None:
        self.url = url
        self.timeout = timeout
        self.max_age = max_age
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._frame: CameraFrame | None = None
        self._error: str | None = "camera stream not connected"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._on_error = on_error

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._enabled.set()

    def set_active(self, active: bool) -> None:
        if active:
            self._enabled.set()
        else:
            self._enabled.clear()
            with self._lock:
                self._frame = None
                self._set_error_locked("camera stream inactive")

    def latest(self, now: float) -> tuple[CameraFrame | None, str | None]:
        with self._lock:
            frame, error = self._frame, self._error
        if error is not None:
            return frame, error
        if frame is None:
            return None, "camera stream has no complete frame"
        if now - frame.evidence_at > self.max_age:
            return frame, "camera stream frame stale"
        return frame, None
    def _set_error_locked(self, error: str | None) -> None:
        self._error = error
        if self._on_error is not None:
            self._on_error(error)

    def _accept(self, frame: CameraFrame) -> bool:
        with self._lock:
            previous = self._frame
            if previous is not None and previous.connection == frame.connection:
                if frame.sequence == previous.sequence:
                    return False
                if frame.sequence is None or previous.sequence is None or frame.sequence < previous.sequence:
                    raise ValueError("camera stream sequence regressed without reconnect")
            self._frame = frame
            self._set_error_locked(None)
        return True

    def _run(self) -> None:
        connection = 0
        while not self._stop.is_set():
            if not self._enabled.wait(0.1) or self._stop.is_set():
                continue
            try:
                request = Request(self.url, headers={"Cache-Control": "no-cache", "Accept-Encoding": "identity"})
                with urlopen(request, timeout=self.timeout) as response:
                    connection += 1
                    for frame in _iter_mjpeg_frames(response, timeout=self.timeout, connection=connection):
                        if self._stop.is_set() or not self._enabled.is_set():
                            break
                        self._accept(frame)
            except Exception as exc:
                with self._lock:
                    self._set_error_locked(str(exc))
                self._stop.wait(0.1)


def _refresh_control_cache(refresh: Callable[[], None], interval: float, stop: threading.Event) -> None:
    while not stop.is_set():
        refresh()
        stop.wait(interval)


def _load_ha_token(config_path: Path) -> str:
    spec = importlib.util.spec_from_file_location("cat_projector_overlay_ha_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Home Assistant config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    token = (
        getattr(module, "HOME_ASSISTANT_TOKEN", None)
        or getattr(module, "HASS_TOKEN", None)
        or getattr(module, "HA_TOKEN", None)
    )
    if not token:
        raise RuntimeError(f"Home Assistant token not found in {config_path}")
    return str(token)


def _read_ha_state(ha_url: str, token: str, entity_id: str, *, timeout: float) -> str:
    request = Request(
        f"{ha_url.rstrip('/')}/api/states/{entity_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload.get("state", "unknown"))


def _projector_active_from_states(states: dict[str, str]) -> bool:
    if not states:
        return True
    return any(state != "off" for state in states.values())


class ProjectorActiveGate:
    def __init__(
        self,
        *,
        ha_url: str | None,
        ha_config_path: Path | None,
        active_entities: tuple[str, ...],
        poll_interval_seconds: float,
        timeout_seconds: float,
    ) -> None:
        self._ha_url = ha_url
        self._ha_config_path = ha_config_path
        self._active_entities = active_entities
        self._poll_interval_seconds = max(0.1, poll_interval_seconds)
        self._timeout_seconds = max(0.1, timeout_seconds)
        self._token: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._refreshed_at: float | None = None
        self._active = True
        self._states: dict[str, str] = {}
        self._error: str | None = None

    def start(self) -> None:
        threading.Thread(
            target=_refresh_control_cache,
            args=(self.refresh, self._poll_interval_seconds, self._stop),
            daemon=True,
        ).start()

    def close(self) -> None:
        self._stop.set()

    def refresh(self) -> None:
        if not self._ha_url or not self._ha_config_path or not self._active_entities:
            return
        try:
            if self._token is None:
                self._token = _load_ha_token(self._ha_config_path.expanduser())
            states = {
                entity_id: _read_ha_state(self._ha_url, self._token, entity_id, timeout=self._timeout_seconds)
                for entity_id in self._active_entities
            }
            active = _projector_active_from_states(states)
            error = None
        except Exception as exc:
            with self._lock:
                states = dict(self._states)
            active = True
            error = str(exc)
        with self._lock:
            self._states = states
            self._active = active
            self._error = error
            self._refreshed_at = time.monotonic()

    def active(self, now: float) -> bool:
        return bool(self.debug(now).get("projector_active_gate_active", True))

    def debug(self, now: float | None = None) -> dict[str, Any]:
        if not self._ha_url or not self._ha_config_path or not self._active_entities:
            return {"projector_active_gate_enabled": False}
        now = time.monotonic() if now is None else now
        with self._lock:
            active, states, error = self._active, dict(self._states), self._error
            refreshed_at = self._refreshed_at
        if refreshed_at is None:
            active, error = True, "Home Assistant active gate refresh pending"
        elif now - refreshed_at > self._poll_interval_seconds + self._timeout_seconds * len(self._active_entities):
            active, error = True, "Home Assistant active gate refresh stale"
        return {
            "projector_active_gate_enabled": True,
            "projector_active_gate_active": active,
            "projector_active_gate_states": states,
            "projector_active_gate_error": error,
        }


class ProjectorSourceVideo:
    def __init__(
        self,
        *,
        fallback_source: str,
        ha_url: str | None,
        ha_config_path: Path | None,
        entity_id: str | None,
        poll_interval_seconds: float,
        timeout_seconds: float,
    ) -> None:
        self._ha_url = ha_url
        self._ha_config_path = ha_config_path
        self._entity_id = entity_id
        self._poll_interval_seconds = max(0.1, poll_interval_seconds)
        self._timeout_seconds = max(0.1, timeout_seconds)
        self._token: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._source = fallback_source
        self._error: str | None = None

    def start(self) -> None:
        threading.Thread(
            target=_refresh_control_cache,
            args=(self.refresh, self._poll_interval_seconds, self._stop),
            daemon=True,
        ).start()

    def close(self) -> None:
        self._stop.set()

    def source(self) -> str:
        with self._lock:
            return self._source

    def refresh(self) -> None:
        if not self._ha_url or not self._ha_config_path or not self._entity_id:
            return
        try:
            if self._token is None:
                self._token = _load_ha_token(self._ha_config_path.expanduser())
            candidate = _read_ha_state(
                self._ha_url,
                self._token,
                self._entity_id,
                timeout=self._timeout_seconds,
            ).strip()
            with self._lock:
                if candidate not in {"", "unknown", "unavailable"}:
                    self._source = candidate
                self._error = None
        except Exception as exc:
            with self._lock:
                self._error = str(exc)

    def debug(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source_video_entity": self._entity_id,
                "source_video": self._source,
                "source_video_error": self._error,
            }


def _parse_projector_polygon(value: str) -> tuple[tuple[float, float], ...]:
    if not value:
        return DEFAULT_PROJECTOR_POLYGON
    points: list[tuple[float, float]] = []
    for pair in value.split(";"):
        x, y = pair.split(",", 1)
        points.append((float(x), float(y)))
    if len(points) != 4:
        raise ValueError("--projector-polygon must contain four x,y pairs separated by semicolons")
    return tuple(points)


def _parse_source_rect(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("rectangle must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("rectangle must satisfy x1 > x0 and y1 > y0")
    return x0, y0, x1, y1


def _render_fixed_black_rect(
    source_frame: Image.Image,
    rect: tuple[int, int, int, int] | None,
) -> Image.Image:
    if rect is None:
        return source_frame
    output = source_frame.convert("RGB")
    draw = ImageDraw.Draw(output)
    draw.rectangle(rect, fill=(0, 0, 0))
    return output


def _render_camera_jpeg(camera_image: Image.Image | None) -> bytes | None:
    if camera_image is None:
        return None
    output = BytesIO()
    camera_image.convert("RGB").save(output, format="JPEG", quality=88)
    return output.getvalue()


def _render_debug_camera_jpeg(
    camera_image: Image.Image | None,
    *,
    result: SafetyOverlayResult,
    people: list[PersonDetection],
) -> bytes | None:
    if camera_image is None:
        return None
    image = camera_image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for person in people:
        draw.rectangle(person.bbox_xyxy, outline=(255, 180, 0, 255), width=4)
    for zone in result.zones:
        if zone.camera_eye_band_xyxy is not None:
            draw.rectangle(zone.camera_eye_band_xyxy, fill=(0, 0, 0, 170), outline=(0, 0, 0, 255), width=3)
    output = BytesIO()
    image.save(output, format="JPEG", quality=88)
    return output.getvalue()



def _format_bbox_xywh_from_xyxy(bbox_xyxy: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox_xyxy
    return ",".join(
        str(int(round(value)))
        for value in (
            x0,
            y0,
            max(1.0, x1 - x0),
            max(1.0, y1 - y0),
        )
    )


def _append_safety_negative_review_rows(
    *,
    camera_image: Image.Image | None,
    raw_people: list[PersonDetection],
    people: list[PersonDetection],
    result: SafetyOverlayResult,
    person_detector_skipped: list[dict[str, Any]],
    performance: dict[str, Any],
    source_video: str,
    max_frames_per_day: int,
) -> int:
    if camera_image is None:
        return 0
    if not raw_people and not people and result.status != "active" and not person_detector_skipped:
        return 0
    run_dir = SAFETY_REVIEW_ROOT / time.strftime("%Y%m%d")
    frames_dir = run_dir / "frames"
    if max_frames_per_day > 0 and _safety_negative_review_frame_count(frames_dir) >= max_frames_per_day:
        return 0
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_path = run_dir / "labels.csv"
    fieldnames = [
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
    frame_name = f"safety_{int(time.time() * 1000)}_{int(performance.get('safety_sample_index') or 0):07d}.jpg"
    frame_path = frames_dir / frame_name
    camera_image.save(frame_path, quality=92)

    rows: list[dict[str, str]] = []
    candidates = raw_people or people
    if not candidates and result.zones:
        for zone in result.zones:
            bbox = zone.camera_eye_band_xyxy or zone.camera_bbox_xyxy
            candidates.append(
                PersonDetection(
                    bbox_xyxy=bbox,
                    confidence=0.0,
                    source="eye_safety_zone",
                    debug={"zone": asdict(zone)},
                )
            )
    for index, person in enumerate(candidates):
        row = {field: "" for field in fieldnames}
        row.update(
            {
                "image_relpath": frame_path.relative_to(run_dir).as_posix(),
                "label_cat_present": "no",
                "label_cat_playing": "no",
                "review_status": "machine_prefilled_needs_human_review",
                "bbox_xywh": "",
                "candidate_bbox_xywh": _format_bbox_xywh_from_xyxy(person.bbox_xyxy),
                "label_candidate_is_cat": "no",
                "negative_reason": "eye_safety_guard",
                "confidence": "machine",
                "notes": json.dumps(
                    {
                        "person_index": index,
                        "person": asdict(person),
                        "status": result.status,
                        "person_detector_skipped": person_detector_skipped,
                        "performance": performance,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "video_slug": Path(source_video).name,
                "candidate_reason": "eye_safety_guard_negative",
            }
        )
        rows.append(row)
    if not rows:
        return 0
    write_header = not labels_path.exists() or labels_path.stat().st_size == 0
    with labels_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _safety_negative_review_frame_count(frames_dir: Path) -> int:
    if not frames_dir.exists():
        return 0
    return sum(1 for _path in frames_dir.glob("safety_*.jpg"))


def _source_polygons_to_camera_mask(
    *,
    source_polygons: list[tuple[tuple[float, float], ...]],
    source_size: tuple[int, int],
    projector_polygon: tuple[tuple[float, float], ...],
    camera_size: tuple[int, int],
) -> np.ndarray:
    """Project source-space polygons into the camera frame via the frozen homography.

    Pure geometry: consumed by the realtime rect proof harness for calibration
    checks. It grants no privacy or source-explanation permission.
    """
    width, height = camera_size
    mask = Image.new("L", (width, height), 0)
    if not source_polygons:
        return np.asarray(mask, dtype=bool)
    try:
        import cv2
    except Exception:
        return np.asarray(mask, dtype=bool)
    source_width, source_height = source_size
    source_points = np.float32(
        [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]]
    )
    camera_points = np.float32(projector_polygon)
    homography = cv2.getPerspectiveTransform(source_points, camera_points)
    draw = ImageDraw.Draw(mask)
    for polygon in source_polygons:
        if len(polygon) < 3:
            continue
        points = np.float32(polygon).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
        draw.polygon([(float(x), float(y)) for x, y in projected], fill=255)
    return np.asarray(mask, dtype=bool)


def _open_video_capture(source: str) -> Any:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError(f"cv2 import failed: {exc}") from exc
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open source video: {source}")
    return capture


def _frame_from_capture(capture: Any, *, source_size: tuple[int, int]) -> Image.Image:
    ok, frame = capture.read()
    if not ok:
        capture.set(1, 0)
        ok, frame = capture.read()
    if not ok:
        raise RuntimeError("failed to read source video frame")
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"cv2 import failed: {exc}") from exc
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    if image.size != source_size:
        image = image.resize(source_size)
    return image


def _start_ffmpeg_hls(
    *,
    output_dir: Path,
    source_size: tuple[int, int],
    fps: int,
    hls_time: int,
    hls_list_size: int,
) -> subprocess.Popen[bytes]:
    width, height = source_size
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-g",
        str(max(1, fps * hls_time)),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "hls",
        "-hls_time",
        str(hls_time),
        "-hls_list_size",
        str(hls_list_size),
        "-hls_flags",
        "delete_segments+append_list+omit_endlist",
        str(output_dir / "stream.m3u8"),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


@dataclass(frozen=True)
class SafetyComputationSnapshot:
    result: SafetyOverlayResult = field(default_factory=lambda: SafetyOverlayResult("starting"))
    people: list[PersonDetection] = field(default_factory=list)
    camera_image: Image.Image | None = None
    camera_error: str | None = None
    performance: dict[str, Any] = field(default_factory=dict)
    updated_at: float | None = None
    updated_at_wall: float | None = None


class SafetyRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source_frame: Image.Image | None = None
        self._source_frame_index = 0
        self._source_updated_at: float | None = None
        self._source_video: str | None = None
        self._snapshot = SafetyComputationSnapshot()
        self._stop = False
        self._camera_stream_error: str | None = None

    def stop(self) -> None:
        with self._lock:
            self._stop = True

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def update_source_frame(self, source_frame: Image.Image, frame_index: int, source_video: str) -> None:
        with self._lock:
            self._source_frame = source_frame
            self._source_frame_index = frame_index
            self._source_updated_at = time.monotonic()
            self._source_video = source_video

    def source_frame_snapshot(self) -> tuple[Image.Image | None, int, float | None, str | None]:
        with self._lock:
            return self._source_frame, self._source_frame_index, self._source_updated_at, self._source_video

    def update_snapshot(self, snapshot: SafetyComputationSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def snapshot(self) -> SafetyComputationSnapshot:
        with self._lock:
            return self._snapshot
    def update_camera_stream_error(self, error: str | None) -> None:
        with self._lock:
            self._camera_stream_error = error

    def camera_stream_error(self) -> str | None:
        with self._lock:
            return self._camera_stream_error


@dataclass(frozen=True)
class SmoothedPersonTrack:
    bbox_xyxy: tuple[float, float, float, float]
    velocity_xy_px_s: tuple[float, float]
    confidence: float
    last_seen_at: float
    last_update_at: float
    source: str
    debug: dict[str, Any] = field(default_factory=dict)


def _bbox_center(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox_xyxy
    return (float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0


def _bbox_lerp(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    return (
        float(left[0]) * (1.0 - alpha) + float(right[0]) * alpha,
        float(left[1]) * (1.0 - alpha) + float(right[1]) * alpha,
        float(left[2]) * (1.0 - alpha) + float(right[2]) * alpha,
        float(left[3]) * (1.0 - alpha) + float(right[3]) * alpha,
    )


def _shift_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    offset_x: float,
    offset_y: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox_xyxy
    return x0 + offset_x, y0 + offset_y, x1 + offset_x, y1 + offset_y


def _clamp_server_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    camera_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    width, height = camera_size
    x0, y0, x1, y1 = bbox_xyxy
    left = max(0.0, min(float(width - 1), min(x0, x1)))
    top = max(0.0, min(float(height - 1), min(y0, y1)))
    right = max(0.0, min(float(width), max(x0, x1)))
    bottom = max(0.0, min(float(height), max(y0, y1)))
    return left, top, right, bottom


def _bbox_intersects_camera_frame(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    camera_size: tuple[int, int],
) -> bool:
    width, height = camera_size
    x0, y0, x1, y1 = bbox_xyxy
    left, right = min(x0, x1), max(x0, x1)
    top, bottom = min(y0, y1), max(y0, y1)
    return right > 0.0 and bottom > 0.0 and left < float(width) and top < float(height)


def _predict_track_bbox(
    track: SmoothedPersonTrack,
    *,
    now: float,
    camera_size: tuple[int, int],
    max_speed_px_s: float,
) -> tuple[float, float, float, float]:
    elapsed = max(0.0, now - track.last_update_at)
    velocity_x, velocity_y = _clamp_prediction_offset(
        track.velocity_xy_px_s[0],
        track.velocity_xy_px_s[1],
        max_prediction_px=max_speed_px_s,
    )
    return _shift_bbox(track.bbox_xyxy, velocity_x * elapsed, velocity_y * elapsed)


def _track_to_detection(
    track: SmoothedPersonTrack,
    *,
    now: float,
    max_missing_seconds: float,
    camera_size: tuple[int, int],
    max_speed_px_s: float,
) -> PersonDetection:
    bbox = _predict_track_bbox(
        track,
        now=now,
        camera_size=camera_size,
        max_speed_px_s=max_speed_px_s,
    )
    missing_seconds = max(0.0, now - track.last_seen_at)
    decay = 1.0 if max_missing_seconds <= 0 else max(0.2, 1.0 - missing_seconds / max_missing_seconds)
    return PersonDetection(
        bbox_xyxy=bbox,
        confidence=float(track.confidence * decay),
        source="physics_smoothed_person_track",
        debug={
            **track.debug,
            "physics_smoothed": True,
            "physics_predicted": True,
            "track_missing_seconds": round(missing_seconds, 3),
            "track_velocity_px_s": (round(track.velocity_xy_px_s[0], 2), round(track.velocity_xy_px_s[1], 2)),
        },
    )


def _update_physical_person_tracks(
    people: list[PersonDetection],
    tracks: list[SmoothedPersonTrack],
    *,
    now: float,
    camera_size: tuple[int, int],
    max_missing_seconds: float,
    max_speed_px_s: float,
    smoothing_alpha: float,
) -> tuple[list[PersonDetection], list[SmoothedPersonTrack], dict[str, Any]]:
    if max_missing_seconds <= 0:
        return people, [], {"physics_track_enabled": False}

    alpha = max(0.05, min(1.0, smoothing_alpha))
    active_tracks = [track for track in tracks if now - track.last_seen_at <= max_missing_seconds]
    unmatched_tracks = list(active_tracks)
    output_people: list[PersonDetection] = []
    next_tracks: list[SmoothedPersonTrack] = []
    matched_count = 0
    predicted_count = 0
    offscreen_count = 0

    for person in people:
        center_x, center_y = _bbox_center(person.bbox_xyxy)
        match_index: int | None = None
        match_distance: float | None = None
        for index, track in enumerate(unmatched_tracks):
            predicted_bbox = _predict_track_bbox(
                track,
                now=now,
                camera_size=camera_size,
                max_speed_px_s=max_speed_px_s,
            )
            predicted_x, predicted_y = _bbox_center(predicted_bbox)
            distance = ((center_x - predicted_x) ** 2 + (center_y - predicted_y) ** 2) ** 0.5
            elapsed = max(0.001, now - track.last_update_at)
            gate = max(120.0, max_speed_px_s * elapsed * 1.25)
            if distance <= gate and (match_distance is None or distance < match_distance):
                match_index = index
                match_distance = distance
        if match_index is None:
            bbox = _clamp_server_bbox(person.bbox_xyxy, camera_size=camera_size)
            next_tracks.append(
                SmoothedPersonTrack(
                    bbox_xyxy=bbox,
                    velocity_xy_px_s=(0.0, 0.0),
                    confidence=person.confidence,
                    last_seen_at=now,
                    last_update_at=now,
                    source=person.source,
                    debug=dict(person.debug),
                )
            )
            output_people.append(person)
            continue

        matched_count += 1
        track = unmatched_tracks.pop(match_index)
        predicted_bbox = _predict_track_bbox(
            track,
            now=now,
            camera_size=camera_size,
            max_speed_px_s=max_speed_px_s,
        )
        smoothed_bbox = _clamp_server_bbox(
            _bbox_lerp(predicted_bbox, person.bbox_xyxy, alpha),
            camera_size=camera_size,
        )
        elapsed = max(0.001, now - track.last_update_at)
        previous_x, previous_y = _bbox_center(track.bbox_xyxy)
        current_x, current_y = _bbox_center(person.bbox_xyxy)
        measured_vx, measured_vy = _clamp_prediction_offset(
            (current_x - previous_x) / elapsed,
            (current_y - previous_y) / elapsed,
            max_prediction_px=max_speed_px_s,
        )
        velocity_x = track.velocity_xy_px_s[0] * (1.0 - alpha) + measured_vx * alpha
        velocity_y = track.velocity_xy_px_s[1] * (1.0 - alpha) + measured_vy * alpha
        velocity_x, velocity_y = _clamp_prediction_offset(
            velocity_x,
            velocity_y,
            max_prediction_px=max_speed_px_s,
        )
        debug = {
            **person.debug,
            "physics_smoothed": True,
            "physics_predicted": False,
            "track_match_distance_px": round(match_distance or 0.0, 2),
            "track_velocity_px_s": (round(velocity_x, 2), round(velocity_y, 2)),
        }
        smoothed = PersonDetection(
            bbox_xyxy=smoothed_bbox,
            confidence=person.confidence,
            source=person.source,
            mask=person.mask,
            debug=debug,
        )
        next_tracks.append(
            SmoothedPersonTrack(
                bbox_xyxy=smoothed_bbox,
                velocity_xy_px_s=(velocity_x, velocity_y),
                confidence=person.confidence,
                last_seen_at=now,
                last_update_at=now,
                source=person.source,
                debug=debug,
            )
        )
        output_people.append(smoothed)

    for track in unmatched_tracks:
        missing_seconds = now - track.last_seen_at
        if missing_seconds > max_missing_seconds:
            continue
        predicted = _track_to_detection(
            track,
            now=now,
            max_missing_seconds=max_missing_seconds,
            camera_size=camera_size,
            max_speed_px_s=max_speed_px_s,
        )
        if not _bbox_intersects_camera_frame(predicted.bbox_xyxy, camera_size=camera_size):
            offscreen_count += 1
            continue
        predicted_count += 1
        output_people.append(predicted)
        next_tracks.append(
            SmoothedPersonTrack(
                bbox_xyxy=predicted.bbox_xyxy,
                velocity_xy_px_s=track.velocity_xy_px_s,
                confidence=track.confidence,
                last_seen_at=track.last_seen_at,
                last_update_at=now,
                source=track.source,
                debug=track.debug,
            )
        )

    return (
        output_people,
        next_tracks,
        {
            "physics_track_enabled": True,
            "physics_track_count": len(next_tracks),
            "physics_track_matched_count": matched_count,
            "physics_track_predicted_count": predicted_count,
            "physics_track_offscreen_count": offscreen_count,
            "physics_track_max_missing_seconds": max_missing_seconds,
        },
    )


def _clamp_prediction_offset(
    offset_x: float,
    offset_y: float,
    *,
    max_prediction_px: float,
) -> tuple[float, float]:
    distance = float((offset_x * offset_x + offset_y * offset_y) ** 0.5)
    if distance <= max_prediction_px or distance <= 0.0:
        return offset_x, offset_y
    scale = max_prediction_px / distance
    return offset_x * scale, offset_y * scale


def _prediction_horizon_seconds(
    *,
    configured_seconds: float,
    camera_sample_interval: float,
    worker_loop_ms: float | None,
    video_frame_interval: float,
) -> float:
    measured = camera_sample_interval + video_frame_interval
    if worker_loop_ms is not None:
        measured = max(measured, worker_loop_ms / 1000.0 + video_frame_interval)
    return max(0.0, max(configured_seconds, measured))


def _annotate_motion_prediction(
    people: list[PersonDetection],
    *,
    previous_people: list[PersonDetection],
    previous_at: float | None,
    now: float,
    horizon_seconds: float,
    padding_px: float,
    max_prediction_px: float,
) -> list[PersonDetection]:
    if not people:
        return []
    if not previous_people or previous_at is None or now <= previous_at:
        return [
            PersonDetection(
                bbox_xyxy=person.bbox_xyxy,
                confidence=person.confidence,
                source=person.source,
                mask=person.mask,
                debug={
                    **person.debug,
                    "prediction_horizon_seconds": round(horizon_seconds, 3),
                    "prediction_padding_px": padding_px,
                    "prediction_offset_px": (0.0, 0.0),
                    "eye_velocity_px_s": (0.0, 0.0),
                },
            )
            for person in people
        ]

    elapsed = max(1e-3, now - previous_at)
    unmatched = list(previous_people)
    annotated: list[PersonDetection] = []
    for person in people:
        center_x, center_y = _bbox_center(person.bbox_xyxy)
        match_index: int | None = None
        match_distance: float | None = None
        for index, candidate in enumerate(unmatched):
            previous_x, previous_y = _bbox_center(candidate.bbox_xyxy)
            distance = ((center_x - previous_x) ** 2 + (center_y - previous_y) ** 2) ** 0.5
            if match_distance is None or distance < match_distance:
                match_index = index
                match_distance = distance
        velocity_x = 0.0
        velocity_y = 0.0
        if match_index is not None and match_distance is not None:
            previous = unmatched.pop(match_index)
            previous_x, previous_y = _bbox_center(previous.bbox_xyxy)
            velocity_x = (center_x - previous_x) / elapsed
            velocity_y = (center_y - previous_y) / elapsed
        offset_x, offset_y = _clamp_prediction_offset(
            velocity_x * horizon_seconds,
            velocity_y * horizon_seconds,
            max_prediction_px=max_prediction_px,
        )
        annotated.append(
            PersonDetection(
                bbox_xyxy=person.bbox_xyxy,
                confidence=person.confidence,
                source=person.source,
                mask=person.mask,
                debug={
                    **person.debug,
                    "prediction_horizon_seconds": round(horizon_seconds, 3),
                    "prediction_padding_px": padding_px,
                    "prediction_offset_px": (round(offset_x, 2), round(offset_y, 2)),
                    "eye_velocity_px_s": (round(velocity_x, 2), round(velocity_y, 2)),
                },
            )
        )
    return annotated


def _people_with_current_frame_evidence(people: list[PersonDetection]) -> list[PersonDetection]:
    """Keep only detections backed by the current camera frame for served overlays."""

    current: list[PersonDetection] = []
    for person in people:
        if person.debug.get("physics_predicted") is True:
            continue
        if person.source == "physics_smoothed_person_track":
            continue
        current.append(person)
    return current


def _build_detector(args: argparse.Namespace) -> MobileNetPersonDetector | UnavailablePersonDetector:
    try:
        return MobileNetPersonDetector(
            prototxt=args.human_detector_prototxt.expanduser(),
            model=args.human_detector_model.expanduser(),
            min_confidence=args.person_min_confidence,
        )
    except Exception as exc:
        return UnavailablePersonDetector(f"person detector unavailable: {exc}")


def _run_safety_worker(args: argparse.Namespace, *, runtime: SafetyRuntime) -> None:
    detector = _build_detector(args)
    active_gate = ProjectorActiveGate(
        ha_url=args.ha_url,
        ha_config_path=args.ha_config_path,
        active_entities=tuple(args.ha_active_entity),
        poll_interval_seconds=args.ha_active_poll_interval,
        timeout_seconds=args.ha_active_timeout,
    )
    stream_url = _camera_stream_url(args.camera_snapshot_url, args.camera_stream_url)
    camera_stream = (
        LatestCameraStream(
            stream_url, timeout=args.camera_snapshot_timeout, max_age=args.camera_max_frame_age,
            on_error=runtime.update_camera_stream_error,
        )
        if stream_url is not None else None
    )
    active_gate.start()
    if camera_stream is not None:
        camera_stream.start()
    try:
        _run_safety_samples(args, runtime=runtime, detector=detector, active_gate=active_gate, camera_stream=camera_stream)
    finally:
        active_gate.close()
        if camera_stream is not None:
            camera_stream.close()


def _run_safety_samples(
    args: argparse.Namespace,
    *,
    runtime: SafetyRuntime,
    detector: Any,
    active_gate: ProjectorActiveGate,
    camera_stream: LatestCameraStream | None,
) -> None:
    last_camera_key: tuple[int, int] | None = None
    recent_eye_zone_results: list[tuple[float, SafetyOverlayResult]] = []
    held_result: SafetyOverlayResult | None = None
    held_people: list[PersonDetection] = []
    held_at = 0.0
    previous_people: list[PersonDetection] = []
    previous_people_at: float | None = None
    physics_tracks: list[SmoothedPersonTrack] = []
    last_negative_review_at = 0.0
    sample_index = 0

    while not runtime.should_stop():
        gate_checked_at = time.monotonic()
        gate_debug = active_gate.debug(gate_checked_at)
        if not gate_debug.get("projector_active_gate_active", True):
            if camera_stream is not None:
                camera_stream.set_active(False)
            last_camera_key = None
            recent_eye_zone_results = []
            held_result = None
            held_people = []
            previous_people = []
            previous_people_at = None
            physics_tracks = []
            sample_index += 1
            result = SafetyOverlayResult(
                "no_person",
                debug={
                    "person_count": 0,
                    "safety_worker_idle": True,
                    "idle_reason": "projector_inactive",
                    **gate_debug,
                },
            )
            runtime.update_snapshot(
                SafetyComputationSnapshot(
                    result=result,
                    people=[],
                    camera_image=None,
                    camera_error=None,
                    performance={
                        "safety_sample_index": sample_index,
                        "safety_worker_idle": True,
                        "idle_reason": "projector_inactive",
                        "inactive_sample_interval_ms": round(args.inactive_sample_interval * 1000.0, 1),
                        **gate_debug,
                    },
                    updated_at=gate_checked_at,
                    updated_at_wall=time.time(),
                )
            )
            time.sleep(args.inactive_sample_interval)
            continue

        if camera_stream is not None:
            camera_stream.set_active(True)
        source_frame, source_frame_index, source_updated_at, source_video = runtime.source_frame_snapshot()
        if source_frame is None or source_video is None:
            time.sleep(min(0.02, max(0.001, args.camera_sample_interval)))
            continue

        started = time.monotonic()
        sample_index += 1
        camera_image: Image.Image | None = None
        camera_error: str | None = None
        camera_frame: CameraFrame | None = None
        raw_people: list[PersonDetection] = []
        people: list[PersonDetection] = []
        camera_read_ms: float | None = None
        detector_ms: float | None = None
        result: SafetyOverlayResult

        try:
            camera_started = time.monotonic()
            if camera_stream is None:
                camera_frame = _read_camera_snapshot(args.camera_snapshot_url, timeout=args.camera_snapshot_timeout)
            else:
                camera_frame, stream_error = camera_stream.latest(camera_started)
                if stream_error is not None:
                    raise RuntimeError(stream_error)
            if camera_frame is None:
                raise RuntimeError("camera has no complete frame")
            if time.monotonic() - camera_frame.evidence_at > args.camera_max_frame_age:
                raise RuntimeError("camera frame stale")
            if camera_frame.key is not None and camera_frame.key == last_camera_key:
                time.sleep(min(0.02, max(0.001, args.camera_sample_interval)))
                continue
            last_camera_key = camera_frame.key
            camera_image = Image.open(BytesIO(camera_frame.jpeg)).convert("RGB")
            camera_read_ms = (time.monotonic() - camera_started) * 1000.0
            person_detector_skipped: list[dict[str, Any]] = []
            try:
                detector_started = time.monotonic()
                raw_people = detector.detect(camera_image)
                detector_ms = (time.monotonic() - detector_started) * 1000.0
            except Exception as exc:
                person_detector_skipped = [{"reason": "person_detector_unavailable", "error": str(exc)}]
                camera_error = f"person detector unavailable: {exc}"
            # Real-person eye safety is fail-closed: every full-frame person
            # detection reaches the overlay computation unchanged. Projected
            # video matching has no independent photometric evidence, so it
            # must never suppress a detected person here.
            people = list(raw_people)
        except Exception as exc:
            camera_error = str(exc)

        now = time.monotonic()
        if camera_frame is not None and now - camera_frame.evidence_at > args.camera_max_frame_age:
            camera_error = "camera frame stale after inference"
        worker_loop_ms_so_far = (now - started) * 1000.0
        horizon_seconds = _prediction_horizon_seconds(
            configured_seconds=args.eye_safety_prediction_seconds,
            camera_sample_interval=args.camera_sample_interval,
            worker_loop_ms=worker_loop_ms_so_far,
            video_frame_interval=1.0 / max(1, args.fps),
        )
        physics_debug: dict[str, Any] = {"physics_track_enabled": False}
        if camera_image is not None and not camera_error:
            people, physics_tracks, physics_debug = _update_physical_person_tracks(
                people,
                physics_tracks,
                now=now,
                camera_size=camera_image.size,
                max_missing_seconds=args.person_track_max_missing_seconds,
                max_speed_px_s=args.person_track_max_speed_px_s,
                smoothing_alpha=args.person_track_smoothing_alpha,
            )
        people = _annotate_motion_prediction(
            people,
            previous_people=previous_people,
            previous_at=previous_people_at,
            now=now,
            horizon_seconds=horizon_seconds,
            padding_px=args.eye_safety_prediction_padding_px,
            max_prediction_px=args.eye_safety_max_prediction_px,
        )
        if people:
            previous_people = list(people)
            previous_people_at = now
        overlay_people = _people_with_current_frame_evidence(people)

        if camera_image is None or camera_error:
            result = SafetyOverlayResult("safety_overlay_unavailable", debug={"error": camera_error})
        else:
            result = compute_eye_safety_overlay(
                camera_size=camera_image.size,
                source_size=args.source_size,
                projector_polygon=args.projector_polygon,
                people=overlay_people,
                eye_band_top_fraction=args.eye_band_top_fraction,
                eye_band_bottom_fraction=args.eye_band_bottom_fraction,
                eye_band_left_fraction=args.eye_band_left_fraction,
                eye_band_right_fraction=args.eye_band_right_fraction,
                padding_px=args.padding_px,
                min_overlap_area_px=args.min_overlap_area_px,
            )
            result = SafetyOverlayResult(
                result.status,
                zones=result.zones,
                debug={
                    **result.debug,
                    "person_detector_skipped": person_detector_skipped,
                    "raw_person_count": len(raw_people),
                    "tracked_person_count": len(people),
                    "overlay_person_count": len(overlay_people),
                    "prediction_horizon_seconds": round(horizon_seconds, 3),
                    "prediction_padding_px": args.eye_safety_prediction_padding_px,
                    "max_prediction_px": args.eye_safety_max_prediction_px,
                    **gate_debug,
                    **physics_debug,
                },
            )
            result, recent_eye_zone_results = _apply_eye_safety_trail(
                result,
                recent_eye_zone_results=recent_eye_zone_results,
                now=now,
                trail_seconds=args.eye_safety_trail_seconds,
            )
            result, held_result, held_people, held_at = _apply_eye_safety_hold(
                result,
                current_people=overlay_people,
                held_result=held_result,
                held_people=held_people,
                held_at=held_at,
                now=now,
                hold_seconds=args.eye_safety_hold_seconds,
            )

        finished = time.monotonic()
        if camera_frame is not None and finished - camera_frame.evidence_at > args.camera_max_frame_age:
            camera_error = "camera frame stale after computation"
            result = SafetyOverlayResult("safety_overlay_unavailable", debug={"error": camera_error})
        if camera_error:
            overlay_people = []
            held_result, held_people = None, []
            recent_eye_zone_results = []
            previous_people, previous_people_at = [], None
            physics_tracks = []
        performance = {
            "safety_sample_index": sample_index,
            "safety_worker_loop_ms": round((finished - started) * 1000.0, 1),
            "camera_sample_interval_ms": round(args.camera_sample_interval * 1000.0, 1),
            "camera_snapshot_timeout_ms": round(args.camera_snapshot_timeout * 1000.0, 1),
            "camera_read_ms": round(camera_read_ms, 1) if camera_read_ms is not None else None,
            "camera_transport": "mjpeg" if camera_stream is not None else "snapshot",
            "camera_frame_sequence": camera_frame.sequence if camera_frame is not None else None,
            "camera_frame_connection": camera_frame.connection if camera_frame is not None else None,
            "camera_frame_age_at_send_ms": camera_frame.age_ms if camera_frame is not None else None,
            "camera_frame_encode_ms": camera_frame.encode_ms if camera_frame is not None else None,
            "camera_frame_age_ms": (
                round((finished - camera_frame.evidence_at) * 1000.0, 1) if camera_frame is not None else None
            ),
            "detector_ms": round(detector_ms, 1) if detector_ms is not None else None,
            "source_frame_index": source_frame_index,
            "source_video": source_video,
            "source_frame_age_ms": (
                round((finished - source_updated_at) * 1000.0, 1) if source_updated_at is not None else None
            ),
            "eye_safety_trail_seconds": args.eye_safety_trail_seconds,
            "eye_safety_hold_seconds": args.eye_safety_hold_seconds,
            "eye_safety_prediction_horizon_ms": round(horizon_seconds * 1000.0, 1),
            "eye_safety_prediction_padding_px": args.eye_safety_prediction_padding_px,
            "eye_safety_max_prediction_px": args.eye_safety_max_prediction_px,
            "person_track_max_missing_seconds": args.person_track_max_missing_seconds,
            "person_track_max_speed_px_s": args.person_track_max_speed_px_s,
            "person_track_smoothing_alpha": args.person_track_smoothing_alpha,
            **gate_debug,
            **physics_debug,
        }
        if (
            camera_image is not None
            and now - last_negative_review_at >= args.safety_negative_review_interval
            and (
                raw_people
                or person_detector_skipped
                or result.status == "active"
                or int(physics_debug.get("physics_track_predicted_count") or 0) > 0
            )
        ):
            rows_written = _append_safety_negative_review_rows(
                camera_image=camera_image,
                raw_people=raw_people,
                people=people,
                result=result,
                person_detector_skipped=person_detector_skipped,
                performance=performance,
                source_video=source_video,
                max_frames_per_day=args.safety_negative_review_max_frames_per_day,
            )
            if rows_written:
                last_negative_review_at = now
                performance = {
                    **performance,
                    "safety_negative_review_rows_written": rows_written,
                    "safety_negative_review_labels": str(SAFETY_REVIEW_ROOT / time.strftime("%Y%m%d") / "labels.csv"),
                }
        runtime.update_snapshot(
            SafetyComputationSnapshot(
                result=result,
                people=held_people if result.debug.get("held_after_last_detection") else overlay_people,
                camera_image=camera_image,
                camera_error=camera_error,
                performance=performance,
                updated_at=camera_frame.evidence_at if camera_frame is not None and not camera_error else finished,
                updated_at_wall=camera_frame.evidence_wall_at if camera_frame is not None and not camera_error else time.time(),
            )
        )
        elapsed = time.monotonic() - started
        if elapsed < args.camera_sample_interval:
            time.sleep(args.camera_sample_interval - elapsed)


def _run_renderer(args: argparse.Namespace, *, output_dir: Path, state: SafetyOverlayState) -> None:
    capture = _open_video_capture(str(args.source_video))
    ffmpeg = _start_ffmpeg_hls(
        output_dir=output_dir,
        source_size=args.source_size,
        fps=args.fps,
        hls_time=args.hls_time,
        hls_list_size=args.hls_list_size,
    )
    assert ffmpeg.stdin is not None
    frame_interval = 1.0 / max(1, args.fps)
    runtime = SafetyRuntime()
    safety_worker = threading.Thread(target=_run_safety_worker, args=(args,), kwargs={"runtime": runtime})
    safety_worker.daemon = True
    safety_worker.start()
    frame_index = 0
    last_status_monotonic = time.monotonic()

    try:
        while True:
            started = time.monotonic()
            frame_index += 1
            source_frame = _frame_from_capture(capture, source_size=args.source_size)
            runtime.update_source_frame(source_frame, frame_index, str(args.source_video))
            snapshot = runtime.snapshot()
            result, camera_error = _current_safety_result(snapshot, now=time.monotonic(), args=args, stream_error=runtime.camera_stream_error())
            rendered_started = time.monotonic()
            rendered = _render_fixed_black_rect(source_frame, args.fixed_black_rect)
            rendered = render_eye_safety_overlay(rendered, result)
            render_ms = (time.monotonic() - rendered_started) * 1000.0
            write_started = time.monotonic()
            try:
                ffmpeg.stdin.write(rendered.tobytes())
                ffmpeg.stdin.flush()
            except BrokenPipeError as exc:
                raise RuntimeError("ffmpeg HLS writer exited") from exc
            write_ms = (time.monotonic() - write_started) * 1000.0

            finished = time.monotonic()
            status_interval_ms = (finished - last_status_monotonic) * 1000.0
            last_status_monotonic = finished
            safety_age_ms = (
                round((finished - snapshot.updated_at) * 1000.0, 1) if snapshot.updated_at is not None else None
            )
            result, camera_error = _current_safety_result(snapshot, now=finished, args=args, stream_error=runtime.camera_stream_error())
            performance = {
                **snapshot.performance,
                "frame_index": frame_index,
                "video_loop_ms": round((finished - started) * 1000.0, 1),
                "video_status_interval_ms": round(status_interval_ms, 1),
                "target_frame_interval_ms": round(frame_interval * 1000.0, 1),
                "video_render_ms": round(render_ms, 1),
                "video_write_ms": round(write_ms, 1),
                "safety_result_age_ms": safety_age_ms,
                "max_active_overlay_age_ms": round(args.max_active_overlay_age * 1000.0, 1),
                "video_decoupled_from_safety_worker": True,
            }
            state.update(
                result,
                people=snapshot.people if result.status == snapshot.result.status else [],
                source_size=args.source_size,
                updated_at=snapshot.updated_at_wall,
                fixed_black_rect=args.fixed_black_rect,
                camera_image=snapshot.camera_image,
                camera_error=camera_error,
                performance=performance,
            )
            elapsed = time.monotonic() - started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        runtime.stop()
        capture.release()


def _run_status_only(args: argparse.Namespace, *, state: SafetyOverlayState) -> None:
    capture: Any | None = None
    active_source_video: str | None = None
    status_interval = 1.0 / max(1, args.fps)
    source_interval = 1.0 / max(1, args.source_tracking_fps)
    runtime = SafetyRuntime()
    source_video = ProjectorSourceVideo(
        fallback_source=str(args.source_video),
        ha_url=args.ha_url,
        ha_config_path=args.ha_config_path,
        entity_id=args.ha_source_video_entity,
        poll_interval_seconds=args.ha_source_video_poll_interval,
        timeout_seconds=args.ha_active_timeout,
    )
    source_video.start()
    safety_worker = threading.Thread(target=_run_safety_worker, args=(args,), kwargs={"runtime": runtime})
    safety_worker.daemon = True
    safety_worker.start()
    frame_index = 0
    last_status_monotonic = time.monotonic()
    next_source_at = 0.0

    try:
        while True:
            started = time.monotonic()
            snapshot = runtime.snapshot()
            worker_idle = bool(snapshot.performance.get("safety_worker_idle"))
            source_debug = source_video.debug()
            if not worker_idle and started >= next_source_at:
                selected_source_video = source_debug["source_video"]
                if capture is None or selected_source_video != active_source_video:
                    if capture is not None:
                        capture.release()
                    capture = _open_video_capture(selected_source_video)
                    active_source_video = selected_source_video
                    frame_index = 0
                frame_index += 1
                source_frame = _frame_from_capture(capture, source_size=args.source_size)
                runtime.update_source_frame(source_frame, frame_index, active_source_video)
                next_source_at = started + source_interval
                snapshot = runtime.snapshot()
            elif worker_idle:
                next_source_at = started + source_interval
            finished = time.monotonic()
            status_interval_ms = (finished - last_status_monotonic) * 1000.0
            last_status_monotonic = finished
            safety_age_ms = (
                round((finished - snapshot.updated_at) * 1000.0, 1) if snapshot.updated_at is not None else None
            )
            result, camera_error = _current_safety_result(snapshot, now=finished, args=args, stream_error=runtime.camera_stream_error())
            performance = {
                **snapshot.performance,
                "frame_index": frame_index,
                "status_only": True,
                "native_video_playback": True,
                "source_tracking_fps": args.source_tracking_fps,
                "status_target_interval_ms": round(status_interval * 1000.0, 1),
                "inactive_status_interval_ms": round(args.inactive_status_interval * 1000.0, 1),
                "status_loop_ms": round((finished - started) * 1000.0, 1),
                "status_interval_ms": round(status_interval_ms, 1),
                "safety_result_age_ms": safety_age_ms,
                "max_active_overlay_age_ms": round(args.max_active_overlay_age * 1000.0, 1),
                "hls_encoder_enabled": False,
                **source_debug,
            }
            state.update(
                result,
                people=snapshot.people if result.status == snapshot.result.status else [],
                source_size=args.source_size,
                updated_at=snapshot.updated_at_wall,
                fixed_black_rect=args.fixed_black_rect,
                camera_image=snapshot.camera_image,
                camera_error=camera_error,
                performance=performance,
            )
            elapsed = time.monotonic() - started
            target_interval = args.inactive_status_interval if worker_idle else status_interval
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
    finally:
        runtime.stop()
        source_video.close()
        if capture is not None:
            capture.release()


def _current_safety_result(
    snapshot: SafetyComputationSnapshot, *, now: float, args: argparse.Namespace, stream_error: str | None = None
) -> tuple[SafetyOverlayResult, str | None]:
    # A healthy publication loop must not make old camera evidence current.
    if stream_error and snapshot.camera_image is not None:
        return SafetyOverlayResult("safety_overlay_unavailable", debug={**snapshot.result.debug, "error": stream_error}), stream_error
    if (
        snapshot.camera_image is not None
        and (snapshot.updated_at is None or now - snapshot.updated_at > args.camera_max_frame_age)
    ):
        error = snapshot.camera_error or "camera evidence stale"
        return SafetyOverlayResult("safety_overlay_unavailable", debug={**snapshot.result.debug, "error": error}), error
    return _expire_stale_active_result(
        snapshot.result, now=now, updated_at=snapshot.updated_at, max_age_seconds=args.max_active_overlay_age
    ), snapshot.camera_error


def _expire_stale_active_result(
    result: SafetyOverlayResult,
    *,
    now: float,
    updated_at: float | None,
    max_age_seconds: float,
) -> SafetyOverlayResult:
    if result.status != "active" or max_age_seconds <= 0:
        return result
    if updated_at is None:
        return SafetyOverlayResult(
            "no_person",
            debug={**result.debug, "stale_active_expired": True, "stale_active_age_seconds": None},
        )
    age = now - updated_at
    if age <= max_age_seconds:
        return result
    return SafetyOverlayResult(
        "no_person",
        debug={
            **result.debug,
            "stale_active_expired": True,
            "stale_active_age_seconds": round(age, 3),
            "max_active_overlay_age_seconds": max_age_seconds,
        },
    )


def _apply_eye_safety_trail(
    result: SafetyOverlayResult,
    *,
    recent_eye_zone_results: list[tuple[float, SafetyOverlayResult]],
    now: float,
    trail_seconds: float,
) -> tuple[SafetyOverlayResult, list[tuple[float, SafetyOverlayResult]]]:
    if trail_seconds <= 0:
        return result, []

    cutoff = now - trail_seconds
    recent = [(timestamp, previous) for timestamp, previous in recent_eye_zone_results if timestamp >= cutoff]
    if result.status == "active" and result.zones:
        recent.append((now, result))
    if result.status != "active" or not result.zones:
        return result, recent

    zones = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _timestamp, previous in recent:
        for zone in previous.zones:
            key = tuple((round(x), round(y)) for x, y in zone.polygon)
            if key in seen:
                continue
            seen.add(key)
            zones.append(zone)
    if len(zones) == len(result.zones):
        return result, recent

    debug = {
        **result.debug,
        "eye_safety_trail_seconds": trail_seconds,
        "eye_safety_trail_zone_count": len(zones),
        "eye_safety_current_zone_count": len(result.zones),
    }
    return SafetyOverlayResult(result.status, zones=tuple(zones), debug=debug), recent


def _apply_eye_safety_hold(
    result: SafetyOverlayResult,
    *,
    current_people: list[PersonDetection],
    held_result: SafetyOverlayResult | None,
    held_people: list[PersonDetection],
    held_at: float,
    now: float,
    hold_seconds: float,
) -> tuple[SafetyOverlayResult, SafetyOverlayResult | None, list[PersonDetection], float]:
    if hold_seconds <= 0:
        return result, None, [], 0.0
    if result.status == "active" and result.zones:
        return result, result, list(current_people), now
    if held_result is None or now - held_at > hold_seconds:
        return result, None, [], 0.0

    held_debug = {
        **result.debug,
        "held_after_last_detection": True,
        "held_for_seconds": round(now - held_at, 3),
        "held_source_status": result.status,
        "held_zone_count": len(held_result.zones),
    }
    return (
        SafetyOverlayResult("active", zones=held_result.zones, debug=held_debug),
        held_result,
        held_people,
        held_at,
    )


def serve(args: argparse.Namespace) -> int:
    if not args.status_only and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for the HLS safety overlay server")
    output_root = args.output_dir or Path(tempfile.mkdtemp(prefix="cat-projector-safety-hls-"))
    output_root.mkdir(parents=True, exist_ok=True)
    state = SafetyOverlayState()
    if args.status_only:
        worker = threading.Thread(target=_run_status_only, args=(args,), kwargs={"state": state})
        worker.daemon = True
        worker.start()
    else:
        worker = threading.Thread(
            target=_run_renderer,
            args=(args,),
            kwargs={"output_dir": output_root, "state": state},
        )
        worker.daemon = True
        worker.start()

    handler = lambda *handler_args, **kwargs: OverlayRequestHandler(  # noqa: E731
        *handler_args,
        directory=str(output_root),
        **kwargs,
    )
    OverlayRequestHandler.state = state
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    if args.status_only:
        print(
            f"cat_projector_safety_overlay status-only listening on http://{args.host}:{args.port}/status.json",
            flush=True,
        )
    else:
        print(f"cat_projector_safety_overlay listening on http://{args.host}:{args.port}/stream.m3u8", flush=True)
    print(f"status: http://{args.host}:{args.port}/status.json", flush=True)
    httpd.serve_forever()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True, help="Local path or URL to the Cat TV source video")
    parser.add_argument(
        "--ha-source-video-entity",
        help="Home Assistant entity whose state is the active source URL in status-only mode",
    )
    parser.add_argument("--ha-source-video-poll-interval", type=float, default=1.0)
    parser.add_argument("--camera-snapshot-url", default=DEFAULT_CAMERA_SNAPSHOT_URL)
    parser.add_argument("--camera-stream-url", help="Explicit direct MJPEG URL; /shot.jpg otherwise selects /video")
    parser.add_argument("--camera-max-frame-age", type=float, default=0.9)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--source-size",
        type=lambda value: tuple(int(part) for part in value.split("x")),
        default=(1280, 720),
    )
    parser.add_argument("--projector-polygon", type=_parse_projector_polygon, default=DEFAULT_PROJECTOR_POLYGON)
    parser.add_argument("--eye-band-top-fraction", type=float, default=0.07)
    parser.add_argument("--eye-band-bottom-fraction", type=float, default=0.19)
    parser.add_argument("--eye-band-left-fraction", type=float, default=0.20)
    parser.add_argument("--eye-band-right-fraction", type=float, default=0.92)
    parser.add_argument("--padding-px", type=int, default=12)
    parser.add_argument("--min-overlap-area-px", type=int, default=24)
    parser.add_argument("--camera-sample-interval", type=float, default=0.06)
    parser.add_argument("--camera-snapshot-timeout", type=float, default=2.0)
    parser.add_argument("--eye-safety-trail-seconds", type=float, default=0.15)
    parser.add_argument("--eye-safety-hold-seconds", type=float, default=0.0)
    parser.add_argument("--eye-safety-prediction-seconds", type=float, default=0.25)
    parser.add_argument("--eye-safety-prediction-padding-px", type=float, default=16.0)
    parser.add_argument("--eye-safety-max-prediction-px", type=float, default=220.0)
    parser.add_argument("--max-active-overlay-age", type=float, default=0.9)
    parser.add_argument(
        "--safety-negative-review-interval",
        type=float,
        default=DEFAULT_SAFETY_NEGATIVE_REVIEW_INTERVAL,
    )
    parser.add_argument(
        "--safety-negative-review-max-frames-per-day",
        type=int,
        default=DEFAULT_SAFETY_NEGATIVE_REVIEW_MAX_FRAMES_PER_DAY,
        help="Maximum generated safety-review camera frames per local day; use 0 to disable the cap.",
    )
    parser.add_argument("--person-track-max-missing-seconds", type=float, default=1.0)
    parser.add_argument("--person-track-max-speed-px-s", type=float, default=1800.0)
    parser.add_argument("--person-track-smoothing-alpha", type=float, default=0.65)
    parser.add_argument("--person-min-confidence", type=float, default=0.35)
    parser.add_argument("--human-detector-prototxt", type=Path, default=DEFAULT_HUMAN_DETECTOR_PROTOTXT)
    parser.add_argument("--human-detector-model", type=Path, default=DEFAULT_HUMAN_DETECTOR_MODEL)
    parser.add_argument("--hls-time", type=int, default=1)
    parser.add_argument("--hls-list-size", type=int, default=4)
    parser.add_argument("--fixed-black-rect", type=_parse_source_rect)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--source-tracking-fps", type=int, default=5)
    parser.add_argument("--ha-url")
    parser.add_argument("--ha-config-path", type=Path)
    parser.add_argument("--ha-active-entity", action="append", default=list(DEFAULT_HA_ACTIVE_ENTITIES))
    parser.add_argument("--ha-active-poll-interval", type=float, default=1.0)
    parser.add_argument("--ha-active-timeout", type=float, default=1.0)
    parser.add_argument("--inactive-sample-interval", type=float, default=2.0)
    parser.add_argument("--inactive-status-interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.camera_max_frame_age) or args.camera_max_frame_age <= 0:
        parser.error("--camera-max-frame-age must be positive and finite")
    if not math.isfinite(args.camera_snapshot_timeout) or args.camera_snapshot_timeout <= 0:
        parser.error("--camera-snapshot-timeout must be positive and finite")
    if len(args.source_size) != 2:
        raise SystemExit("--source-size must be WIDTHxHEIGHT")
    args.source_size = (int(args.source_size[0]), int(args.source_size[1]))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
