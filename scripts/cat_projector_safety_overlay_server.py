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

    def update(
        self,
        result: SafetyOverlayResult,
        *,
        people: list[PersonDetection],
        camera_image: Image.Image | None = None,
        camera_error: str | None = None,
    ) -> None:
        debug_camera_jpeg = _render_debug_camera_jpeg(
            camera_image,
            result=result,
            people=people,
        )
        with self._lock:
            self.payload = {
                "status": result.status,
                "zone_count": len(result.zones),
                "zones": [asdict(zone) for zone in result.zones],
                "person_count": len(people),
                "people": [asdict(person) for person in people],
                "debug": result.debug,
                "camera_error": camera_error,
                "updated_at": time.time(),
            }
            self.debug_camera_jpeg = debug_camera_jpeg

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.payload)

    def debug_camera_snapshot(self) -> bytes | None:
        with self._lock:
            return self.debug_camera_jpeg


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
    last_camera_error: str | None = None

    while True:
        started = time.monotonic()
        source_frame = _frame_from_capture(capture, source_size=args.source_size)
        now = time.monotonic()
        if last_camera is None or now - last_camera_at >= args.camera_sample_interval:
            try:
                last_camera = _read_camera_snapshot(args.camera_snapshot_url)
                last_people = detector.detect(last_camera)
                last_camera_error = None
            except Exception as exc:
                last_people = []
                last_camera_error = str(exc)
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
                padding_px=args.padding_px,
                min_overlap_area_px=args.min_overlap_area_px,
            )
        state.update(result, people=last_people, camera_image=last_camera, camera_error=last_camera_error)
        rendered = render_eye_safety_overlay(source_frame, result)
        try:
            ffmpeg.stdin.write(rendered.tobytes())
            ffmpeg.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg HLS writer exited") from exc

        elapsed = time.monotonic() - started
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


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
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--source-size",
        type=lambda value: tuple(int(part) for part in value.split("x")),
        default=(1280, 720),
    )
    parser.add_argument("--projector-polygon", type=_parse_projector_polygon, default=DEFAULT_PROJECTOR_POLYGON)
    parser.add_argument("--eye-band-top-fraction", type=float, default=0.10)
    parser.add_argument("--eye-band-bottom-fraction", type=float, default=0.42)
    parser.add_argument("--padding-px", type=int, default=18)
    parser.add_argument("--min-overlap-area-px", type=int, default=24)
    parser.add_argument("--camera-sample-interval", type=float, default=0.35)
    parser.add_argument("--person-min-confidence", type=float, default=0.35)
    parser.add_argument("--human-detector-prototxt", type=Path, default=DEFAULT_HUMAN_DETECTOR_PROTOTXT)
    parser.add_argument("--human-detector-model", type=Path, default=DEFAULT_HUMAN_DETECTOR_MODEL)
    parser.add_argument("--hls-time", type=int, default=1)
    parser.add_argument("--hls-list-size", type=int, default=4)
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
