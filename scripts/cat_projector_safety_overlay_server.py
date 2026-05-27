#!/usr/bin/env python3
"""Serve a Cat TV source video with live eye-safety blackout overlays."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

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


def _load_source_subtraction_module() -> Any:
    path = REPO_ROOT / "custom_components" / "cat_tv_play" / "source_subtraction.py"
    spec = importlib.util.spec_from_file_location("cat_tv_play_source_subtraction_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source_subtraction module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


projector_safety = _load_projector_safety_module()
source_subtraction = _load_source_subtraction_module()
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
PERSON_CLASS_ID = 15


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
        self.debug_camera_jpeg: bytes | None = None
        self.last_active_payload: dict[str, Any] | None = None
        self.last_active_debug_camera_jpeg: bytes | None = None

    def update(
        self,
        result: SafetyOverlayResult,
        *,
        people: list[PersonDetection],
        source_size: tuple[int, int],
        fixed_black_rect: tuple[int, int, int, int] | None = None,
        camera_image: Image.Image | None = None,
        camera_error: str | None = None,
        performance: dict[str, Any] | None = None,
    ) -> None:
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
            "updated_at": time.time(),
        }
        with self._lock:
            self.payload = payload
            self.debug_camera_jpeg = debug_camera_jpeg
            if result.status == "active":
                self.last_active_payload = dict(payload)
                self.last_active_debug_camera_jpeg = debug_camera_jpeg

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.payload)

    def debug_camera_snapshot(self) -> bytes | None:
        with self._lock:
            return self.debug_camera_jpeg

    def last_active_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self.last_active_payload) if self.last_active_payload is not None else None

    def last_active_debug_camera_snapshot(self) -> bytes | None:
        with self._lock:
            return self.last_active_debug_camera_jpeg


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
        super().do_GET()


def _read_camera_snapshot(url: str, *, timeout: float = 2.0) -> Image.Image:
    with urlopen(url, timeout=timeout) as response:
        return Image.open(response).convert("RGB")


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


def _bbox_residual_stats(
    residual: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    threshold: float,
    source_bright_mask: np.ndarray,
) -> tuple[int, float]:
    height, width = residual.shape
    x0, y0, x1, y1 = bbox_xyxy
    left = max(0, min(width, int(np.floor(min(x0, x1)))))
    top = max(0, min(height, int(np.floor(min(y0, y1)))))
    right = max(0, min(width, int(np.ceil(max(x0, x1)))))
    bottom = max(0, min(height, int(np.ceil(max(y0, y1)))))
    if right <= left or bottom <= top:
        return 0, 0.0
    patch = residual[top:bottom, left:right]
    bright_patch = source_bright_mask[top:bottom, left:right]
    dark = (patch > threshold) & bright_patch
    return int(dark.sum()), float(dark.mean())


def _build_residual_views(
    *,
    camera_image: Image.Image,
    source_frame: Image.Image,
    source_reference_frames: list[Image.Image] | None,
    projector_polygon: tuple[tuple[float, float], ...],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    source_frames = [source_frame]
    if source_reference_frames:
        source_frames.extend(source_reference_frames)
    residual_views: list[tuple[int, np.ndarray, np.ndarray]] = []
    for frame_index, candidate_source in enumerate(source_frames):
        residual, warped_source = source_subtraction.source_subtracted_residual(
            camera_image,
            source_frame=candidate_source,
            projector_polygon=projector_polygon,
        )
        residual_views.append((frame_index, residual, warped_source > 150))
    return residual_views


def _source_polygons_to_camera_mask(
    *,
    source_polygons: list[tuple[tuple[float, float], ...]],
    source_size: tuple[int, int],
    projector_polygon: tuple[tuple[float, float], ...],
    camera_size: tuple[int, int],
) -> np.ndarray:
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


def _fixed_rect_to_polygon(rect: tuple[int, int, int, int] | None) -> tuple[tuple[float, float], ...] | None:
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    return ((float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1)))


def _detect_residual_occluder_people(
    residual_views: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    threshold: float,
    min_residual_area_px: int,
    min_residual_fraction: float,
    ignored_camera_mask: np.ndarray,
) -> tuple[list[PersonDetection], list[dict[str, Any]]]:
    if not residual_views:
        return [], []
    try:
        import cv2
    except Exception:
        return [], [{"reason": "residual_occluder_unavailable", "error": "cv2 import failed"}]

    height, width = residual_views[0][1].shape
    combined = np.zeros((height, width), dtype=bool)
    for _frame_index, residual, source_bright_mask in residual_views:
        combined |= (residual > threshold) & source_bright_mask
    combined &= ~ignored_camera_mask
    combined[: max(2, int(height * 0.05)), :] = False
    mask_u8 = combined.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    people: list[PersonDetection] = []
    skipped: list[dict[str, Any]] = []
    for label in range(1, count):
        left, top, candidate_width, candidate_height, area = (int(value) for value in stats[label])
        if area < min_residual_area_px:
            continue
        if candidate_width < 60 or candidate_height < 80:
            continue
        if candidate_width > int(width * 0.38) or candidate_height > int(height * 0.85):
            continue
        bbox = (
            float(left),
            float(top),
            float(left + candidate_width),
            float(top + candidate_height),
        )
        bbox_stats = [
            (
                frame_index,
                *_bbox_residual_stats(
                    residual,
                    bbox,
                    threshold=threshold,
                    source_bright_mask=source_bright_mask & ~ignored_camera_mask,
                ),
            )
            for frame_index, residual, source_bright_mask in residual_views
        ]
        best_frame_index, residual_area, residual_fraction = min(bbox_stats, key=lambda item: (item[1], item[2]))
        debug = {
            "source_subtracted_residual_area_px": residual_area,
            "source_subtracted_residual_fraction": residual_fraction,
            "source_subtracted_best_reference_index": best_frame_index,
            "source_subtracted_reference_count": len(residual_views),
            "component_area_px": area,
        }
        if residual_area >= min_residual_area_px and residual_fraction >= min_residual_fraction:
            people.append(
                PersonDetection(
                    bbox_xyxy=bbox,
                    confidence=min(0.99, max(0.2, residual_fraction)),
                    source="source_subtracted_human_occluder",
                    debug=debug,
                )
            )
        else:
            skipped.append(
                {
                    "reason": "residual_matches_projected_source",
                    "bbox_xyxy": bbox,
                    "residual_area_px": residual_area,
                    "residual_fraction": residual_fraction,
                    "best_reference_index": best_frame_index,
                    "reference_count": len(residual_views),
                }
            )
    return people, skipped


def _filter_source_projected_people(
    people: list[PersonDetection],
    *,
    camera_image: Image.Image,
    source_frame: Image.Image,
    source_reference_frames: list[Image.Image] | None = None,
    ignored_source_polygons: list[tuple[tuple[float, float], ...]] | None = None,
    source_size: tuple[int, int] = (1280, 720),
    projector_polygon: tuple[tuple[float, float], ...],
    residual_threshold: float,
    min_residual_area_px: int,
    min_residual_fraction: float,
) -> tuple[list[PersonDetection], list[dict[str, Any]]]:
    residual_views = _build_residual_views(
        camera_image=camera_image,
        source_frame=source_frame,
        source_reference_frames=source_reference_frames,
        projector_polygon=projector_polygon,
    )
    ignored_camera_mask = _source_polygons_to_camera_mask(
        source_polygons=ignored_source_polygons or [],
        source_size=source_size,
        projector_polygon=projector_polygon,
        camera_size=camera_image.size,
    )
    accepted: list[PersonDetection] = []
    skipped: list[dict[str, Any]] = []
    for index, person in enumerate(people):
        stats = [
            (
                frame_index,
                *_bbox_residual_stats(
                    residual,
                    person.bbox_xyxy,
                    threshold=residual_threshold,
                    source_bright_mask=source_bright_mask,
                ),
            )
            for frame_index, residual, source_bright_mask in residual_views
        ]
        best_frame_index, residual_area, residual_fraction = min(
            stats,
            key=lambda item: (item[1], item[2]),
        )
        debug = {
            **person.debug,
            "source_subtracted_residual_area_px": residual_area,
            "source_subtracted_residual_fraction": residual_fraction,
            "source_subtracted_best_reference_index": best_frame_index,
            "source_subtracted_reference_count": len(residual_views),
        }
        enriched = PersonDetection(
            bbox_xyxy=person.bbox_xyxy,
            confidence=person.confidence,
            source=person.source,
            mask=person.mask,
            debug=debug,
        )
        if residual_area >= min_residual_area_px and residual_fraction >= min_residual_fraction:
            accepted.append(enriched)
        else:
            skipped.append(
                {
                    "index": index,
                    "reason": "matches_projected_source",
                    "bbox_xyxy": person.bbox_xyxy,
                    "confidence": person.confidence,
                    "residual_area_px": residual_area,
                    "residual_fraction": residual_fraction,
                    "best_reference_index": best_frame_index,
                    "reference_count": len(residual_views),
                }
            )
    if not accepted:
        occluder_people, occluder_skipped = _detect_residual_occluder_people(
            residual_views,
            threshold=residual_threshold,
            min_residual_area_px=min_residual_area_px,
            min_residual_fraction=min_residual_fraction,
            ignored_camera_mask=ignored_camera_mask,
        )
        accepted.extend(occluder_people)
        skipped.extend(occluder_skipped)
    return accepted, skipped


def _sample_source_reference_frames(
    source: str,
    *,
    source_size: tuple[int, int],
    max_frames: int,
) -> list[Image.Image]:
    if max_frames <= 0:
        return []
    try:
        capture = _open_video_capture(source)
    except Exception:
        return []
    try:
        try:
            import cv2
        except Exception:
            return []
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return []
        positions = np.linspace(0, max(0, total_frames - 1), num=min(max_frames, total_frames), dtype=int)
        frames: list[Image.Image] = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            try:
                frames.append(_frame_from_capture(capture, source_size=source_size))
            except Exception:
                continue
        return frames
    finally:
        capture.release()


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


def _run_renderer(args: argparse.Namespace, *, output_dir: Path, state: SafetyOverlayState) -> None:
    detector: MobileNetPersonDetector | UnavailablePersonDetector
    try:
        detector = MobileNetPersonDetector(
            prototxt=args.human_detector_prototxt.expanduser(),
            model=args.human_detector_model.expanduser(),
            min_confidence=args.person_min_confidence,
        )
    except Exception as exc:
        detector = UnavailablePersonDetector(f"person detector unavailable: {exc}")

    capture = _open_video_capture(str(args.source_video))
    source_reference_frames = _sample_source_reference_frames(
        str(args.source_video),
        source_size=args.source_size,
        max_frames=args.source_reference_frames,
    )
    ffmpeg = _start_ffmpeg_hls(
        output_dir=output_dir,
        source_size=args.source_size,
        fps=args.fps,
        hls_time=args.hls_time,
        hls_list_size=args.hls_list_size,
    )
    assert ffmpeg.stdin is not None
    frame_interval = 1.0 / max(1, args.fps)
    last_camera_at = 0.0
    last_camera: Image.Image | None = None
    last_people: list[PersonDetection] = []
    last_raw_people: list[PersonDetection] = []
    last_source_filter_skipped: list[dict[str, Any]] = []
    last_camera_error: str | None = None
    last_blackout_polygons: list[tuple[tuple[float, float], ...]] = []
    held_result: SafetyOverlayResult | None = None
    held_people: list[PersonDetection] = []
    held_at = 0.0
    frame_index = 0
    last_status_monotonic = time.monotonic()
    last_camera_sample_monotonic: float | None = None
    last_camera_read_ms: float | None = None
    last_detector_ms: float | None = None
    last_source_filter_ms: float | None = None

    while True:
        started = time.monotonic()
        frame_index += 1
        source_frame = _frame_from_capture(capture, source_size=args.source_size)
        now = time.monotonic()
        if last_camera is None or now - last_camera_at >= args.camera_sample_interval:
            try:
                camera_started = time.monotonic()
                last_camera = _read_camera_snapshot(args.camera_snapshot_url, timeout=args.camera_snapshot_timeout)
                last_camera_read_ms = (time.monotonic() - camera_started) * 1000.0
                detector_skipped: list[dict[str, Any]] = []
                try:
                    detector_started = time.monotonic()
                    last_raw_people = detector.detect(last_camera)
                    last_detector_ms = (time.monotonic() - detector_started) * 1000.0
                except Exception as exc:
                    last_raw_people = []
                    last_detector_ms = None
                    detector_skipped = [{"reason": "person_detector_unavailable", "error": str(exc)}]
                ignored_source_polygons = list(last_blackout_polygons)
                fixed_rect_polygon = _fixed_rect_to_polygon(args.fixed_black_rect)
                if fixed_rect_polygon is not None:
                    ignored_source_polygons.append(fixed_rect_polygon)
                source_filter_started = time.monotonic()
                last_people, filter_skipped = _filter_source_projected_people(
                    last_raw_people,
                    camera_image=last_camera,
                    source_frame=source_frame,
                    source_reference_frames=source_reference_frames,
                    ignored_source_polygons=ignored_source_polygons,
                    source_size=args.source_size,
                    projector_polygon=args.projector_polygon,
                    residual_threshold=args.person_residual_threshold,
                    min_residual_area_px=args.person_min_residual_area_px,
                    min_residual_fraction=args.person_min_residual_fraction,
                )
                last_source_filter_ms = (time.monotonic() - source_filter_started) * 1000.0
                last_source_filter_skipped = detector_skipped + filter_skipped
                last_camera_error = None
                last_camera_sample_monotonic = time.monotonic()
            except Exception as exc:
                last_raw_people = []
                last_people = []
                last_source_filter_skipped = []
                last_camera_error = str(exc)
                last_camera_read_ms = None
                last_detector_ms = None
                last_source_filter_ms = None
            last_camera_at = now

        if last_camera is None or last_camera_error:
            result = SafetyOverlayResult("safety_overlay_unavailable", debug={"error": last_camera_error})
        else:
            result = compute_eye_safety_overlay(
                camera_size=last_camera.size,
                source_size=args.source_size,
                projector_polygon=args.projector_polygon,
                people=last_people,
                eye_band_top_fraction=args.eye_band_top_fraction,
                eye_band_bottom_fraction=args.eye_band_bottom_fraction,
                eye_band_left_fraction=args.eye_band_left_fraction,
                eye_band_right_fraction=args.eye_band_right_fraction,
                padding_px=args.padding_px,
                min_overlap_area_px=args.min_overlap_area_px,
            )
            if last_source_filter_skipped:
                result = SafetyOverlayResult(
                    result.status,
                    zones=result.zones,
                    debug={**result.debug, "source_filter_skipped": last_source_filter_skipped},
                )
            result, held_result, held_people, held_at = _apply_eye_safety_hold(
                result,
                current_people=last_people,
                held_result=held_result,
                held_people=held_people,
                held_at=held_at,
                now=time.monotonic(),
                hold_seconds=args.eye_safety_hold_seconds,
            )
            last_blackout_polygons = [zone.polygon for zone in result.zones]
        finished = time.monotonic()
        status_interval_ms = (finished - last_status_monotonic) * 1000.0
        last_status_monotonic = finished
        performance = {
            "frame_index": frame_index,
            "loop_ms": round((finished - started) * 1000.0, 1),
            "status_interval_ms": round(status_interval_ms, 1),
            "target_frame_interval_ms": round(frame_interval * 1000.0, 1),
            "camera_sample_interval_ms": round(args.camera_sample_interval * 1000.0, 1),
            "camera_snapshot_timeout_ms": round(args.camera_snapshot_timeout * 1000.0, 1),
            "camera_age_ms": (
                round((finished - last_camera_sample_monotonic) * 1000.0, 1)
                if last_camera_sample_monotonic is not None
                else None
            ),
            "camera_read_ms": round(last_camera_read_ms, 1) if last_camera_read_ms is not None else None,
            "detector_ms": round(last_detector_ms, 1) if last_detector_ms is not None else None,
            "source_filter_ms": round(last_source_filter_ms, 1) if last_source_filter_ms is not None else None,
            "eye_safety_hold_seconds": args.eye_safety_hold_seconds,
        }
        state.update(
            result,
            people=held_people if result.debug.get("held_after_last_detection") else last_people,
            source_size=args.source_size,
            fixed_black_rect=args.fixed_black_rect,
            camera_image=last_camera,
            camera_error=last_camera_error,
            performance=performance,
        )
        rendered = _render_fixed_black_rect(source_frame, args.fixed_black_rect)
        rendered = render_eye_safety_overlay(rendered, result)
        try:
            ffmpeg.stdin.write(rendered.tobytes())
            ffmpeg.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg HLS writer exited") from exc

        elapsed = time.monotonic() - started
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


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
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for the HLS safety overlay server")
    output_root = args.output_dir or Path(tempfile.mkdtemp(prefix="cat-projector-safety-hls-"))
    output_root.mkdir(parents=True, exist_ok=True)
    state = SafetyOverlayState()
    renderer = threading.Thread(target=_run_renderer, args=(args,), kwargs={"output_dir": output_root, "state": state})
    renderer.daemon = True
    renderer.start()

    handler = lambda *handler_args, **kwargs: OverlayRequestHandler(  # noqa: E731
        *handler_args,
        directory=str(output_root),
        **kwargs,
    )
    OverlayRequestHandler.state = state
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"cat_projector_safety_overlay listening on http://{args.host}:{args.port}/stream.m3u8", flush=True)
    print(f"status: http://{args.host}:{args.port}/status.json", flush=True)
    httpd.serve_forever()
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True, help="Local path or URL to the Cat TV source video")
    parser.add_argument("--camera-snapshot-url", default=DEFAULT_CAMERA_SNAPSHOT_URL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--source-size",
        type=lambda value: tuple(int(part) for part in value.split("x")),
        default=(1280, 720),
    )
    parser.add_argument("--projector-polygon", type=_parse_projector_polygon, default=DEFAULT_PROJECTOR_POLYGON)
    parser.add_argument("--eye-band-top-fraction", type=float, default=0.08)
    parser.add_argument("--eye-band-bottom-fraction", type=float, default=0.22)
    parser.add_argument("--eye-band-left-fraction", type=float, default=0.22)
    parser.add_argument("--eye-band-right-fraction", type=float, default=0.78)
    parser.add_argument("--padding-px", type=int, default=20)
    parser.add_argument("--min-overlap-area-px", type=int, default=24)
    parser.add_argument("--person-residual-threshold", type=float, default=28.0)
    parser.add_argument("--person-min-residual-area-px", type=int, default=1200)
    parser.add_argument("--person-min-residual-fraction", type=float, default=0.10)
    parser.add_argument("--source-reference-frames", type=int, default=36)
    parser.add_argument("--camera-sample-interval", type=float, default=0.12)
    parser.add_argument("--camera-snapshot-timeout", type=float, default=0.35)
    parser.add_argument("--eye-safety-hold-seconds", type=float, default=2.0)
    parser.add_argument("--person-min-confidence", type=float, default=0.35)
    parser.add_argument("--human-detector-prototxt", type=Path, default=DEFAULT_HUMAN_DETECTOR_PROTOTXT)
    parser.add_argument("--human-detector-model", type=Path, default=DEFAULT_HUMAN_DETECTOR_MODEL)
    parser.add_argument("--hls-time", type=int, default=1)
    parser.add_argument("--hls-list-size", type=int, default=4)
    parser.add_argument("--fixed-black-rect", type=_parse_source_rect)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if len(args.source_size) != 2:
        raise SystemExit("--source-size must be WIDTHxHEIGHT")
    args.source_size = (int(args.source_size[0]), int(args.source_size[1]))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
