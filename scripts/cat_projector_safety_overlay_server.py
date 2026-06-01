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
from dataclasses import asdict, dataclass, field
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
            "updated_at": time.time(),
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


def _person_eye_band_bbox_for_filter(
    person: PersonDetection,
    *,
    camera_size: tuple[int, int],
    eye_band_top_fraction: float,
    eye_band_bottom_fraction: float,
    eye_band_left_fraction: float,
    eye_band_right_fraction: float,
    padding_px: int,
) -> tuple[float, float, float, float]:
    width, height = camera_size
    x0, y0, x1, y1 = _clamp_server_bbox(person.bbox_xyxy, camera_size=camera_size)
    person_width = x1 - x0
    person_height = y1 - y0
    left = x0 + person_width * eye_band_left_fraction
    right = x0 + person_width * eye_band_right_fraction
    top = y0 + person_height * eye_band_top_fraction
    bottom = y0 + person_height * eye_band_bottom_fraction
    return _clamp_server_bbox(
        (
            float(np.floor(left - padding_px)),
            float(np.floor(top - padding_px)),
            float(np.ceil(right + padding_px)),
            float(np.ceil(bottom + padding_px)),
        ),
        camera_size=(width, height),
    )


def _bbox_area(bbox_xyxy: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox_xyxy
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


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


def _scale_projector_polygon(
    projector_polygon: tuple[tuple[float, float], ...],
    scale: float,
) -> tuple[tuple[float, float], ...]:
    return tuple((float(x) * scale, float(y) * scale) for x, y in projector_polygon)


def _scale_person_detection(person: PersonDetection, scale: float) -> PersonDetection:
    x0, y0, x1, y1 = person.bbox_xyxy
    return PersonDetection(
        bbox_xyxy=(x0 * scale, y0 * scale, x1 * scale, y1 * scale),
        confidence=person.confidence,
        source=person.source,
        mask=None,
        debug=person.debug,
    )


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
    enable_residual_occluder_fallback: bool = False,
    source_filter_scale: float = 1.0,
    eye_band_top_fraction: float = 0.07,
    eye_band_bottom_fraction: float = 0.19,
    eye_band_left_fraction: float = 0.20,
    eye_band_right_fraction: float = 0.92,
    padding_px: int = 12,
) -> tuple[list[PersonDetection], list[dict[str, Any]]]:
    if not people and not enable_residual_occluder_fallback and not ignored_source_polygons:
        return [], [{"reason": "residual_occluder_fallback_disabled"}]

    scale = max(0.1, min(1.0, float(source_filter_scale)))
    filter_camera_image = camera_image
    filter_projector_polygon = projector_polygon
    filter_people = people
    filter_min_residual_area_px = min_residual_area_px
    if scale < 1.0:
        width, height = camera_image.size
        filter_camera_image = camera_image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
        filter_projector_polygon = _scale_projector_polygon(projector_polygon, scale)
        filter_people = [_scale_person_detection(person, scale) for person in people]
        filter_min_residual_area_px = max(1, int(round(min_residual_area_px * scale * scale)))

    residual_views = _build_residual_views(
        camera_image=filter_camera_image,
        source_frame=source_frame,
        source_reference_frames=source_reference_frames,
        projector_polygon=filter_projector_polygon,
    )
    ignored_camera_mask = _source_polygons_to_camera_mask(
        source_polygons=ignored_source_polygons or [],
        source_size=source_size,
        projector_polygon=filter_projector_polygon,
        camera_size=filter_camera_image.size,
    )
    accepted: list[PersonDetection] = []
    skipped: list[dict[str, Any]] = []
    for index, person in enumerate(filter_people):
        original_person = people[index]
        eye_band_bbox = _person_eye_band_bbox_for_filter(
            person,
            camera_size=filter_camera_image.size,
            eye_band_top_fraction=eye_band_top_fraction,
            eye_band_bottom_fraction=eye_band_bottom_fraction,
            eye_band_left_fraction=eye_band_left_fraction,
            eye_band_right_fraction=eye_band_right_fraction,
            padding_px=max(0, int(round(padding_px * scale))),
        )
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
        eye_stats = [
            (
                frame_index,
                *_bbox_residual_stats(
                    residual,
                    eye_band_bbox,
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
        best_eye_frame_index, eye_residual_area, eye_residual_fraction = min(
            eye_stats,
            key=lambda item: (item[1], item[2]),
        )
        debug = {
            **original_person.debug,
            "source_subtracted_residual_area_px": residual_area,
            "source_subtracted_residual_fraction": residual_fraction,
            "source_subtracted_best_reference_index": best_frame_index,
            "source_subtracted_reference_count": len(residual_views),
            "source_subtracted_eye_band_residual_area_px": eye_residual_area,
            "source_subtracted_eye_band_residual_fraction": eye_residual_fraction,
            "source_subtracted_eye_band_best_reference_index": best_eye_frame_index,
            "source_filter_scale": scale,
        }
        enriched = PersonDetection(
            bbox_xyxy=original_person.bbox_xyxy,
            confidence=original_person.confidence,
            source=original_person.source,
            mask=original_person.mask,
            debug=debug,
        )
        eye_min_residual_area_px = max(
            1,
            min(
                filter_min_residual_area_px,
                int(round(_bbox_area(eye_band_bbox) * min_residual_fraction)),
            ),
        )
        if (
            residual_area >= filter_min_residual_area_px
            and residual_fraction >= min_residual_fraction
        ) or (
            eye_residual_area >= eye_min_residual_area_px
            and eye_residual_fraction >= min_residual_fraction
        ):
            accepted.append(enriched)
        else:
            skipped.append(
                {
                    "index": index,
                    "reason": "matches_projected_source",
                    "bbox_xyxy": original_person.bbox_xyxy,
                    "confidence": original_person.confidence,
                    "residual_area_px": residual_area,
                    "residual_fraction": residual_fraction,
                    "eye_band_residual_area_px": eye_residual_area,
                    "eye_band_residual_fraction": eye_residual_fraction,
                    "best_reference_index": best_frame_index,
                    "eye_band_best_reference_index": best_eye_frame_index,
                    "reference_count": len(residual_views),
                    "source_filter_scale": scale,
                }
            )
    if not accepted and enable_residual_occluder_fallback:
        occluder_people, occluder_skipped = _detect_residual_occluder_people(
            residual_views,
            threshold=residual_threshold,
            min_residual_area_px=filter_min_residual_area_px,
            min_residual_fraction=min_residual_fraction,
            ignored_camera_mask=ignored_camera_mask,
        )
        if scale < 1.0:
            occluder_people = [
                PersonDetection(
                    bbox_xyxy=tuple(value / scale for value in person.bbox_xyxy),
                    confidence=person.confidence,
                    source=person.source,
                    mask=person.mask,
                    debug={**person.debug, "source_filter_scale": scale},
                )
                for person in occluder_people
            ]
        accepted.extend(occluder_people)
        skipped.extend(occluder_skipped)
    elif not accepted and not people and not ignored_camera_mask.any():
        skipped.append({"reason": "residual_occluder_fallback_disabled"})
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


@dataclass(frozen=True)
class SafetyComputationSnapshot:
    result: SafetyOverlayResult = field(default_factory=lambda: SafetyOverlayResult("starting"))
    people: list[PersonDetection] = field(default_factory=list)
    camera_image: Image.Image | None = None
    camera_error: str | None = None
    performance: dict[str, Any] = field(default_factory=dict)
    blackout_polygons: list[tuple[tuple[float, float], ...]] = field(default_factory=list)
    updated_at: float | None = None


class SafetyRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source_frame: Image.Image | None = None
        self._source_frame_index = 0
        self._source_updated_at: float | None = None
        self._snapshot = SafetyComputationSnapshot()
        self._stop = False

    def stop(self) -> None:
        with self._lock:
            self._stop = True

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def update_source_frame(self, source_frame: Image.Image, frame_index: int) -> None:
        with self._lock:
            self._source_frame = source_frame
            self._source_frame_index = frame_index
            self._source_updated_at = time.monotonic()

    def source_frame_snapshot(self) -> tuple[Image.Image | None, int, float | None]:
        with self._lock:
            return self._source_frame, self._source_frame_index, self._source_updated_at

    def update_snapshot(self, snapshot: SafetyComputationSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def snapshot(self) -> SafetyComputationSnapshot:
        with self._lock:
            return self._snapshot


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
    return _clamp_server_bbox(
        _shift_bbox(track.bbox_xyxy, velocity_x * elapsed, velocity_y * elapsed),
        camera_size=camera_size,
    )


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
        predicted_count += 1
        predicted = _track_to_detection(
            track,
            now=now,
            max_missing_seconds=max_missing_seconds,
            camera_size=camera_size,
            max_speed_px_s=max_speed_px_s,
        )
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
    source_reference_frames = _sample_source_reference_frames(
        str(args.source_video),
        source_size=args.source_size,
        max_frames=args.source_reference_frames,
    )
    recent_eye_zone_results: list[tuple[float, SafetyOverlayResult]] = []
    held_result: SafetyOverlayResult | None = None
    held_people: list[PersonDetection] = []
    held_at = 0.0
    previous_people: list[PersonDetection] = []
    previous_people_at: float | None = None
    physics_tracks: list[SmoothedPersonTrack] = []
    last_blackout_polygons: list[tuple[tuple[float, float], ...]] = []
    sample_index = 0

    while not runtime.should_stop():
        source_frame, source_frame_index, source_updated_at = runtime.source_frame_snapshot()
        if source_frame is None:
            time.sleep(min(0.02, max(0.001, args.camera_sample_interval)))
            continue

        started = time.monotonic()
        sample_index += 1
        camera_image: Image.Image | None = None
        camera_error: str | None = None
        raw_people: list[PersonDetection] = []
        people: list[PersonDetection] = []
        source_filter_skipped: list[dict[str, Any]] = []
        camera_read_ms: float | None = None
        detector_ms: float | None = None
        source_filter_ms: float | None = None
        result: SafetyOverlayResult

        try:
            camera_started = time.monotonic()
            camera_image = _read_camera_snapshot(args.camera_snapshot_url, timeout=args.camera_snapshot_timeout)
            camera_read_ms = (time.monotonic() - camera_started) * 1000.0
            detector_skipped: list[dict[str, Any]] = []
            try:
                detector_started = time.monotonic()
                raw_people = detector.detect(camera_image)
                detector_ms = (time.monotonic() - detector_started) * 1000.0
            except Exception as exc:
                detector_skipped = [{"reason": "person_detector_unavailable", "error": str(exc)}]
            ignored_source_polygons = list(last_blackout_polygons)
            fixed_rect_polygon = _fixed_rect_to_polygon(args.fixed_black_rect)
            if fixed_rect_polygon is not None:
                ignored_source_polygons.append(fixed_rect_polygon)
            source_filter_started = time.monotonic()
            people, filter_skipped = _filter_source_projected_people(
                raw_people,
                camera_image=camera_image,
                source_frame=source_frame,
                source_reference_frames=source_reference_frames,
                ignored_source_polygons=ignored_source_polygons,
                source_size=args.source_size,
                projector_polygon=args.projector_polygon,
                residual_threshold=args.person_residual_threshold,
                min_residual_area_px=args.person_min_residual_area_px,
                min_residual_fraction=args.person_min_residual_fraction,
                enable_residual_occluder_fallback=args.enable_residual_occluder_fallback,
                source_filter_scale=args.source_filter_scale,
                eye_band_top_fraction=args.eye_band_top_fraction,
                eye_band_bottom_fraction=args.eye_band_bottom_fraction,
                eye_band_left_fraction=args.eye_band_left_fraction,
                eye_band_right_fraction=args.eye_band_right_fraction,
                padding_px=args.padding_px,
            )
            source_filter_ms = (time.monotonic() - source_filter_started) * 1000.0
            source_filter_skipped = detector_skipped + filter_skipped
        except Exception as exc:
            camera_error = str(exc)

        now = time.monotonic()
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

        if camera_image is None or camera_error:
            result = SafetyOverlayResult("safety_overlay_unavailable", debug={"error": camera_error})
        else:
            result = compute_eye_safety_overlay(
                camera_size=camera_image.size,
                source_size=args.source_size,
                projector_polygon=args.projector_polygon,
                people=people,
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
                    "source_filter_skipped": source_filter_skipped,
                    "raw_person_count": len(raw_people),
                    "prediction_horizon_seconds": round(horizon_seconds, 3),
                    "prediction_padding_px": args.eye_safety_prediction_padding_px,
                    "max_prediction_px": args.eye_safety_max_prediction_px,
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
                current_people=people,
                held_result=held_result,
                held_people=held_people,
                held_at=held_at,
                now=now,
                hold_seconds=args.eye_safety_hold_seconds,
            )
            last_blackout_polygons = [zone.polygon for zone in result.zones]

        finished = time.monotonic()
        performance = {
            "safety_sample_index": sample_index,
            "safety_worker_loop_ms": round((finished - started) * 1000.0, 1),
            "camera_sample_interval_ms": round(args.camera_sample_interval * 1000.0, 1),
            "camera_snapshot_timeout_ms": round(args.camera_snapshot_timeout * 1000.0, 1),
            "camera_read_ms": round(camera_read_ms, 1) if camera_read_ms is not None else None,
            "detector_ms": round(detector_ms, 1) if detector_ms is not None else None,
            "source_filter_ms": round(source_filter_ms, 1) if source_filter_ms is not None else None,
            "source_frame_index": source_frame_index,
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
            **physics_debug,
        }
        runtime.update_snapshot(
            SafetyComputationSnapshot(
                result=result,
                people=held_people if result.debug.get("held_after_last_detection") else people,
                camera_image=camera_image,
                camera_error=camera_error,
                performance=performance,
                blackout_polygons=list(last_blackout_polygons),
                updated_at=finished,
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
            runtime.update_source_frame(source_frame, frame_index)
            snapshot = runtime.snapshot()
            result = snapshot.result
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
            result = _expire_stale_active_result(
                snapshot.result,
                now=finished,
                updated_at=snapshot.updated_at,
                max_age_seconds=args.max_active_overlay_age,
            )
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
                fixed_black_rect=args.fixed_black_rect,
                camera_image=snapshot.camera_image,
                camera_error=snapshot.camera_error,
                performance=performance,
            )
            elapsed = time.monotonic() - started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        runtime.stop()
        capture.release()


def _run_status_only(args: argparse.Namespace, *, state: SafetyOverlayState) -> None:
    capture = _open_video_capture(str(args.source_video))
    status_interval = 1.0 / max(1, args.fps)
    source_interval = 1.0 / max(1, args.source_tracking_fps)
    runtime = SafetyRuntime()
    safety_worker = threading.Thread(target=_run_safety_worker, args=(args,), kwargs={"runtime": runtime})
    safety_worker.daemon = True
    safety_worker.start()
    frame_index = 0
    last_status_monotonic = time.monotonic()
    next_source_at = 0.0

    try:
        while True:
            started = time.monotonic()
            if started >= next_source_at:
                frame_index += 1
                source_frame = _frame_from_capture(capture, source_size=args.source_size)
                runtime.update_source_frame(source_frame, frame_index)
                next_source_at = started + source_interval
            snapshot = runtime.snapshot()
            finished = time.monotonic()
            status_interval_ms = (finished - last_status_monotonic) * 1000.0
            last_status_monotonic = finished
            safety_age_ms = (
                round((finished - snapshot.updated_at) * 1000.0, 1) if snapshot.updated_at is not None else None
            )
            result = _expire_stale_active_result(
                snapshot.result,
                now=finished,
                updated_at=snapshot.updated_at,
                max_age_seconds=args.max_active_overlay_age,
            )
            performance = {
                **snapshot.performance,
                "frame_index": frame_index,
                "status_only": True,
                "native_video_playback": True,
                "source_tracking_fps": args.source_tracking_fps,
                "status_target_interval_ms": round(status_interval * 1000.0, 1),
                "status_loop_ms": round((finished - started) * 1000.0, 1),
                "status_interval_ms": round(status_interval_ms, 1),
                "safety_result_age_ms": safety_age_ms,
                "max_active_overlay_age_ms": round(args.max_active_overlay_age * 1000.0, 1),
                "hls_encoder_enabled": False,
            }
            state.update(
                result,
                people=snapshot.people if result.status == snapshot.result.status else [],
                source_size=args.source_size,
                fixed_black_rect=args.fixed_black_rect,
                camera_image=snapshot.camera_image,
                camera_error=snapshot.camera_error,
                performance=performance,
            )
            elapsed = time.monotonic() - started
            if elapsed < status_interval:
                time.sleep(status_interval - elapsed)
    finally:
        runtime.stop()
        capture.release()


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
    parser.add_argument("--eye-band-top-fraction", type=float, default=0.07)
    parser.add_argument("--eye-band-bottom-fraction", type=float, default=0.19)
    parser.add_argument("--eye-band-left-fraction", type=float, default=0.20)
    parser.add_argument("--eye-band-right-fraction", type=float, default=0.92)
    parser.add_argument("--padding-px", type=int, default=12)
    parser.add_argument("--min-overlap-area-px", type=int, default=24)
    parser.add_argument("--person-residual-threshold", type=float, default=28.0)
    parser.add_argument("--person-min-residual-area-px", type=int, default=1200)
    parser.add_argument("--person-min-residual-fraction", type=float, default=0.10)
    parser.add_argument("--source-filter-scale", type=float, default=0.35)
    parser.add_argument(
        "--enable-residual-occluder-fallback",
        dest="enable_residual_occluder_fallback",
        action="store_true",
    )
    parser.add_argument(
        "--disable-residual-occluder-fallback",
        dest="enable_residual_occluder_fallback",
        action="store_false",
    )
    parser.set_defaults(enable_residual_occluder_fallback=True)
    parser.add_argument("--source-reference-frames", type=int, default=3)
    parser.add_argument("--camera-sample-interval", type=float, default=0.06)
    parser.add_argument("--camera-snapshot-timeout", type=float, default=2.0)
    parser.add_argument("--eye-safety-trail-seconds", type=float, default=0.15)
    parser.add_argument("--eye-safety-hold-seconds", type=float, default=0.0)
    parser.add_argument("--eye-safety-prediction-seconds", type=float, default=0.25)
    parser.add_argument("--eye-safety-prediction-padding-px", type=float, default=16.0)
    parser.add_argument("--eye-safety-max-prediction-px", type=float, default=220.0)
    parser.add_argument("--max-active-overlay-age", type=float, default=0.9)
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
