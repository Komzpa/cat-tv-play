#!/usr/bin/env python3
"""Prove that the Android projector overlay can draw a live black source rectangle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_SERVER_PATH = REPO_ROOT / "scripts" / "cat_projector_safety_overlay_server.py"
DEFAULT_ADB_SERIAL = "192.168.100.39:5555"
DEFAULT_CAMERA_SNAPSHOT_URL = "http://192.168.100.39:8081/shot.jpg"
DEFAULT_VIDEO_URL = "http://192.168.100.74:8787/stream.m3u8"
DEFAULT_LIVE_STATUS_URL = "http://192.168.100.74:8787/status.json"


def _load_overlay_server() -> Any:
    spec = importlib.util.spec_from_file_location("cat_projector_overlay_server_for_rect_proof", OVERLAY_SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load overlay server module from {OVERLAY_SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


overlay_server = _load_overlay_server()


@dataclass
class RectState:
    source_size: tuple[int, int]
    rect: tuple[int, int, int, int]
    active: bool = False
    activated_at_monotonic: float | None = None
    requests: list[float] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        now = time.monotonic()
        self.requests.append(now)
        x0, y0, x1, y1 = self.rect
        zones = [
            {
                "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "source": "realtime_rect_proof",
            }
        ]
        return {
            "status": "active" if self.active else "no_person",
            "zone_count": 1 if self.active else 0,
            "zones": zones if self.active else [],
            "source_size": list(self.source_size),
            "fixed_black_rect": None,
            "updated_at": time.time(),
            "debug": {
                "proof_active": self.active,
                "proof_activated_at_monotonic": self.activated_at_monotonic,
            },
        }


def _rect_polygon(rect: tuple[int, int, int, int]) -> tuple[tuple[float, float], ...]:
    x0, y0, x1, y1 = rect
    return ((float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1)))


def _read_bytes_url(url: str, *, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _read_camera(url: str, *, timeout: float) -> Image.Image:
    return Image.open(urllib.request.urlopen(url, timeout=timeout)).convert("RGB")


def _adb(serial: str, *args: str, timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["adb", "-s", serial, *args],
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _adb_shell(serial: str, command: str, *, timeout: float = 8.0) -> None:
    _adb(serial, "shell", command, timeout=timeout)


def _adb_screencap(serial: str) -> bytes:
    return subprocess.check_output(["adb", "-s", serial, "exec-out", "screencap", "-p"], timeout=8)


def _start_android_overlay(serial: str, *, video_url: str, status_url: str, source_size: tuple[int, int]) -> None:
    width, height = source_size
    _adb(
        serial,
        "shell",
        "am",
        "start",
        "-a",
        "by.openclaw.catprojectorcamera.DISPLAY_RECT",
        "-n",
        "by.openclaw.catprojectorcamera/.ProjectorDisplayActivity",
        "--es",
        "video_url",
        video_url,
        "--es",
        "overlay_status_url",
        status_url,
        "--ei",
        "source_width",
        str(width),
        "--ei",
        "source_height",
        str(height),
        timeout=8,
    )


def _ensure_projector_lamp(serial: str) -> None:
    _adb_shell(
        serial,
        "pm enable com.zhiying.powerservice >/dev/null 2>&1 || true; "
        "am startservice -n com.zhiying.powerservice/.PowerService --ez standby false >/dev/null; "
        "input keyevent 224; sleep 0.4; "
        "am force-stop com.zhiying.powerservice >/dev/null 2>&1 || true",
        timeout=12,
    )


def _projected_mask(
    *,
    image_size: tuple[int, int],
    source_size: tuple[int, int],
    source_polygon: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    mask = overlay_server._source_polygons_to_camera_mask(
        source_polygons=[source_polygon],
        source_size=source_size,
        projector_polygon=overlay_server.DEFAULT_PROJECTOR_POLYGON,
        camera_size=image_size,
    )
    try:
        import cv2
    except Exception:
        return mask, []

    source_width, source_height = source_size
    source_points = np.float32(
        [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]]
    )
    camera_points = np.float32(overlay_server.DEFAULT_PROJECTOR_POLYGON)
    homography = cv2.getPerspectiveTransform(source_points, camera_points)
    projected = cv2.perspectiveTransform(np.float32(source_polygon).reshape(-1, 1, 2), homography).reshape(-1, 2)
    return mask, [(float(x), float(y)) for x, y in projected]


def _luma_mean(image: Image.Image, mask: np.ndarray) -> float:
    values = np.asarray(image.convert("L"), dtype=np.uint8)[mask]
    if values.size == 0:
        return 255.0
    return float(values.mean())


def _annotate(image: Image.Image, projected_polygon: list[tuple[float, float]], label: str) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    if projected_polygon:
        points = list(projected_polygon)
        points.append(projected_polygon[0])
        draw.line(points, fill=(0, 255, 0), width=4)
    draw.text((12, 12), label, fill=(255, 255, 0))
    return annotated


def _screen_rect_stats(
    screen: Image.Image,
    rect: tuple[int, int, int, int],
    source_size: tuple[int, int],
) -> dict[str, Any]:
    width, height = screen.size
    source_width, source_height = source_size
    x0, y0, x1, y1 = rect
    scaled = (
        int(round(x0 * width / source_width)),
        int(round(y0 * height / source_height)),
        int(round(x1 * width / source_width)),
        int(round(y1 * height / source_height)),
    )
    arr = np.asarray(screen.convert("L"), dtype=np.uint8)
    patch = arr[scaled[1] : scaled[3], scaled[0] : scaled[2]]
    return {
        "screen_size": [width, height],
        "scaled_rect": list(scaled),
        "mean_luma": round(float(patch.mean()), 2) if patch.size else None,
        "p90_luma": round(float(np.percentile(patch, 90)), 2) if patch.size else None,
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_handler(state: RectState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path != "/status.json":
                self.send_error(404)
                return
            body = json.dumps(state.payload()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def _serve_status(state: RectState, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, name="cat-projector-rect-proof-status", daemon=True)
    thread.start()
    return server


def run(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_size = tuple(args.source_size)
    rect = tuple(args.rect)
    state = RectState(source_size=source_size, rect=rect)
    server = _serve_status(state, args.host, args.port)
    status_url = f"http://{args.advertise_host}:{args.port}/status.json"
    source_polygon = _rect_polygon(rect)
    samples: list[dict[str, Any]] = []
    first_seen_ms: float | None = None
    first_seen_path: str | None = None
    try:
        if args.ensure_lamp:
            _ensure_projector_lamp(args.adb_serial)
        _start_android_overlay(
            args.adb_serial,
            video_url=args.video_url,
            status_url=status_url,
            source_size=source_size,
        )
        time.sleep(args.pre_arm_seconds)
        baseline = _read_camera(args.camera_snapshot_url, timeout=args.http_timeout)
        baseline_path = out_dir / "baseline.jpg"
        baseline.save(baseline_path, quality=92)
        mask, projected_polygon = _projected_mask(
            image_size=baseline.size,
            source_size=source_size,
            source_polygon=source_polygon,
        )
        baseline_luma = _luma_mean(baseline, mask)
        state.active = True
        state.activated_at_monotonic = time.monotonic()
        deadline = state.activated_at_monotonic + args.duration
        index = 0
        while time.monotonic() < deadline:
            captured_at = time.monotonic()
            image = _read_camera(args.camera_snapshot_url, timeout=args.http_timeout)
            mean_luma = _luma_mean(image, mask)
            delta = baseline_luma - mean_luma
            seen = delta >= args.min_darkening_luma or mean_luma <= args.max_active_luma
            sample_path = None
            if index == 0 or seen and first_seen_ms is None or index % max(1, int(args.save_every)) == 0:
                sample_path = out_dir / f"sample-{index:03d}.jpg"
                image.save(sample_path, quality=90)
            if seen and first_seen_ms is None:
                first_seen_ms = (captured_at - state.activated_at_monotonic) * 1000.0
                first_seen_path = str(sample_path or out_dir / f"first-seen-{index:03d}.jpg")
                if sample_path is None:
                    image.save(first_seen_path, quality=90)
                _annotate(image, projected_polygon, f"first seen {first_seen_ms:.1f}ms").save(
                    out_dir / "first-seen-annotated.jpg",
                    quality=92,
                )
            samples.append(
                {
                    "index": index,
                    "captured_ms_after_activation": round((captured_at - state.activated_at_monotonic) * 1000.0, 1),
                    "expected_rect_mean_luma": round(mean_luma, 2),
                    "darkening_from_baseline": round(delta, 2),
                    "seen": seen,
                    "image": str(sample_path) if sample_path else None,
                }
            )
            index += 1
            time.sleep(args.sample_interval)
        screen_path = out_dir / "screen-after-render.png"
        screen_path.write_bytes(_adb_screencap(args.adb_serial))
        screen_image = Image.open(screen_path).convert("RGB")
        screen_stats = _screen_rect_stats(screen_image, rect, source_size)
        _annotate(baseline, projected_polygon, "baseline expected rect").save(
            out_dir / "baseline-annotated.jpg",
            quality=92,
        )
        summary = {
            "accepted": (
                first_seen_ms is not None
                and screen_stats["mean_luma"] is not None
                and screen_stats["mean_luma"] <= 10.0
            ),
            "status_url": status_url,
            "video_url": args.video_url,
            "restored_status_url": args.restore_status_url,
            "source_size": list(source_size),
            "source_rect": list(rect),
            "camera_size": list(baseline.size),
            "projected_camera_polygon": [[round(x, 2), round(y, 2)] for x, y in projected_polygon],
            "baseline_expected_rect_mean_luma": round(baseline_luma, 2),
            "first_seen_ms": round(first_seen_ms, 1) if first_seen_ms is not None else None,
            "first_seen_path": first_seen_path,
            "screen_rect_stats": screen_stats,
            "status_requests": len(state.requests),
            "status_request_rate_hz": round(len(state.requests) / max(0.001, args.pre_arm_seconds + args.duration), 2),
            "camera_sample_count": len(samples),
            "camera_sample_rate_hz": round(len(samples) / max(0.001, args.duration), 2),
            "samples": samples,
        }
        _save_json(out_dir / "proof-summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["accepted"] else 1
    finally:
        try:
            _start_android_overlay(
                args.adb_serial,
                video_url=args.video_url,
                status_url=args.restore_status_url,
                source_size=source_size,
            )
        finally:
            server.shutdown()
            server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb-serial", default=DEFAULT_ADB_SERIAL)
    parser.add_argument("--camera-snapshot-url", default=DEFAULT_CAMERA_SNAPSHOT_URL)
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--restore-status-url", default=DEFAULT_LIVE_STATUS_URL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--advertise-host", default="192.168.100.74")
    parser.add_argument("--port", type=int, default=8798)
    parser.add_argument(
        "--source-size",
        type=lambda value: tuple(int(part) for part in value.split("x")),
        default=(1280, 720),
    )
    parser.add_argument(
        "--rect",
        type=lambda value: tuple(int(part) for part in value.split(",")),
        default=(520, 260, 760, 420),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=1.8)
    parser.add_argument("--pre-arm-seconds", type=float, default=0.5)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--http-timeout", type=float, default=1.0)
    parser.add_argument("--min-darkening-luma", type=float, default=18.0)
    parser.add_argument("--max-active-luma", type=float, default=95.0)
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--ensure-lamp", action="store_true")
    args = parser.parse_args()
    if len(args.source_size) != 2:
        raise SystemExit("--source-size must be WIDTHxHEIGHT")
    if len(args.rect) != 4:
        raise SystemExit("--rect must be x0,y0,x1,y1")
    if args.rect[2] <= args.rect[0] or args.rect[3] <= args.rect[1]:
        raise SystemExit("--rect must satisfy x1 > x0 and y1 > y0")
    return args


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
