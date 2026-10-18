#!/usr/bin/env python3
"""Causal optical round-trip harness: video marker -> camera -> nucat -> overlay
status -> physical renderer -> camera-visible blackout.

One mode only. The proof status server stays ``no_person`` until the harness
observes a fresh, non-duplicate input-marker edge in the continuously drained
camera MJPEG stream; only then does it publish the blackout polygon. The same
stream is then read until the output ROI darkens. Latency is first-input-frame
to first-output-frame on the same local receive timeline, with lower/upper
bounds from the actual neighboring frame receive times. Those bounds cover
frame sampling only: they are NOT strict optical onset bounds, because
exposure/readout and pre-callback delay are unknown and common-mode
cancellation across the two observations is an assumption, not a proof.
``X-Frame-Age-Ms`` is callback age, retained per frame as diagnostics.

This is a fixed-marker transport/render closed-loop benchmark. It bypasses the
production MobileNet-SSD person detector, source-subtraction, tracking and
eye-band geometry: a projected person would be suppressed by production
source-projected-person filtering, so no production-detector number is claimed
here. Real-model validation stays with the proof owner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_SERVER_PATH = REPO_ROOT / "scripts" / "cat_projector_safety_overlay_server.py"
DEFAULT_ADB_SERIAL = "192.168.100.39:5555"
DEFAULT_CAMERA_SNAPSHOT_URL = "http://192.168.100.39:8081/shot.jpg"
DEFAULT_CAMERA_STREAM_URL = "http://192.168.100.39:8081/video?fps=25"
DEFAULT_LIVE_STATUS_URL = "http://192.168.100.74:8787/status.json"
MARKER_NAME = "optical-marker.mp4"


def _load_overlay_server() -> Any:
    spec = importlib.util.spec_from_file_location(
        "cat_projector_overlay_server_for_optical_proof", OVERLAY_SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load overlay server module from {OVERLAY_SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


overlay_server = _load_overlay_server()


def _rect(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(p) for p in value.split(","))
    if len(parts) != 4 or parts[2] <= parts[0] or parts[3] <= parts[1]:
        raise argparse.ArgumentTypeError("rect must be x0,y0,x1,y1 with x1>x0, y1>y0")
    return parts  # type: ignore[return-value]


def _polygon(rect: tuple[int, int, int, int]) -> tuple[tuple[float, float], ...]:
    x0, y0, x1, y1 = rect
    return ((float(x0), float(y0)), (float(x1), float(y0)),
            (float(x1), float(y1)), (float(x0), float(y1)))


def _mask(image_size: tuple[int, int], source_size: tuple[int, int],
          rect: tuple[int, int, int, int]) -> np.ndarray:
    # Existing calibration: production homography/mask builder, unchanged.
    return overlay_server._source_polygons_to_camera_mask(
        source_polygons=[_polygon(rect)], source_size=source_size,
        projector_polygon=overlay_server.DEFAULT_PROJECTOR_POLYGON,
        camera_size=image_size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(len(ordered) - 1, low + 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _stats(values: list[float]) -> dict[str, Any]:
    return {"n": len(values),
            "p50_ms": round(v, 1) if (v := _percentile(values, 50)) is not None else None,
            "p95_ms": round(v, 1) if (v := _percentile(values, 95)) is not None else None,
            "max_ms": round(max(values), 1) if values else None}


def _adb(serial: str, *args: str, timeout: float = 8.0) -> None:
    subprocess.run(["adb", "-s", serial, *args], check=True, text=True,
                   capture_output=True, timeout=timeout)


def _start_android(serial: str, *, video_url: str, status_url: str,
                   source_size: tuple[int, int]) -> None:
    _adb(serial, "shell", "am", "start", "-a",
         "camera.catprojector.publisher.DISPLAY_RECT", "-n",
         "camera.catprojector.display/camera.catprojector.publisher.ProjectorDisplayActivity",
         "--es", "video_url", video_url, "--es", "overlay_status_url", status_url,
         "--ei", "source_width", str(source_size[0]),
         "--ei", "source_height", str(source_size[1]))


def _stop_android(serial: str) -> None:
    _adb(serial, "shell", "am", "start", "-a",
         "camera.catprojector.publisher.STOP_DISPLAY", "-n",
         "camera.catprojector.display/camera.catprojector.publisher.ProjectorDisplayActivity")


# ---------------------------------------------------------------------------
# Marker video: gray field; input ROI white for the high part of each period.
# Generated with the existing ffmpeg runtime; served by the proof HTTP server.
# ---------------------------------------------------------------------------

def generate_marker(*, output: Path, source_size: tuple[int, int],
                    input_rect: tuple[int, int, int, int], duration: float,
                    fps: int, low_seconds: float, high_seconds: float) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    period = low_seconds + high_seconds
    x0, y0, x1, y1 = input_rect
    enable = f"between(mod(t\\,{period:.3f})\\,{low_seconds:.3f}\\,{period:.3f})"
    filt = (f"color=c=0x808080:size={source_size[0]}x{source_size[1]}:rate={fps}"
            f":duration={duration:.3f},drawbox=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}"
            f":color=white:t=fill:enable='{enable}'")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", filt, "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-g", str(fps), "-movflags", "+faststart",
                    str(output)], check=True, capture_output=True, text=True)
    return {"path": str(output), "sha256": _sha256(output), "bytes": output.stat().st_size,
            "duration_s": duration, "fps": fps, "period_s": period,
            "low_s": low_seconds, "high_s": high_seconds}


# ---------------------------------------------------------------------------
# Proof status + marker HTTP server. Status stays clear until the harness
# calls activate() from an observed fresh input edge; nothing else can set it.
# ---------------------------------------------------------------------------

@dataclass
class ProofState:
    source_size: tuple[int, int]
    rect: tuple[int, int, int, int]
    active: bool = False
    trial: int = -1
    input_meta: dict[str, Any] | None = None
    activated_at_monotonic: float | None = None
    requests: int = 0

    def clear(self) -> None:
        self.active = False
        self.input_meta = None
        self.activated_at_monotonic = None

    def activate(self, trial: int, input_meta: dict[str, Any]) -> None:
        self.trial = trial
        self.input_meta = input_meta
        self.activated_at_monotonic = time.monotonic()
        self.active = True

    def payload(self) -> dict[str, Any]:
        self.requests += 1
        x0, y0, x1, y1 = self.rect
        zone = {"polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "source": "optical_marker_roundtrip"}
        return {"status": "active" if self.active else "no_person",
                "zone_count": 1 if self.active else 0,
                "zones": [zone] if self.active else [],
                "source_size": list(self.source_size), "fixed_black_rect": None,
                "updated_at": time.time(),
                "performance": {"proof_trial": self.trial,
                                "input_camera_frame": self.input_meta,
                                "activated_at_monotonic": self.activated_at_monotonic},
                "debug": {"probe_kind": "fixed_marker_transport_render_NOT_production_detector"}}


def _serve(state: ProofState, marker_path: Path, host: str, port: int) -> ThreadingHTTPServer:
    marker_bytes = marker_path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/status.json":
                body = json.dumps(state.payload()).encode()
                self._reply(200, body, "application/json")
            elif path == "/" + MARKER_NAME:
                self._video()
            else:
                self.send_error(404)

        def _reply(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)

        def _video(self) -> None:
            range_header = self.headers.get("Range", "")
            if range_header.startswith("bytes="):
                try:
                    start_s, _, end_s = range_header[6:].partition("-")
                    start = int(start_s or 0)
                    end = int(end_s) if end_s else len(marker_bytes) - 1
                except ValueError:
                    return self.send_error(416)
                end = min(end, len(marker_bytes) - 1)
                if start > end:
                    return self.send_error(416)
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(marker_bytes)}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(marker_bytes[start:end + 1])
            else:
                self._reply(200, marker_bytes, "video/mp4")

        def log_message(self, *_a: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="optical-proof-http").start()
    return server


def _frame_meta(frame: Any) -> dict[str, Any]:
    return {"sequence": frame.sequence, "connection": frame.connection,
            "received_at_monotonic": frame.received_at,
            "received_wall_at": frame.received_wall_at,
            "read_started_at_monotonic": frame.read_started_at,
            "callback_age_ms": frame.age_ms, "encode_ms": frame.encode_ms,
            "evidence_at_monotonic": frame.evidence_at}


def _is_output_decline(output_base: float, out_luma: float, *,
                       min_darkening: float, max_active_luma: float) -> bool:
    return (output_base > max_active_luma
            and output_base - out_luma >= min_darkening
            and out_luma <= max_active_luma)

def run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_size: tuple[int, int] = tuple(args.source_size)
    marker_path = args.marker_video.expanduser() if args.marker_video else out_dir / MARKER_NAME
    if args.marker_video:
        video_url = args.video_url or f"http://{args.advertise_host}:{args.port}/{MARKER_NAME}"
        marker_info = {"path": str(marker_path), "sha256": _sha256(marker_path),
                       "served": args.video_url is None}
    else:
        marker_info = generate_marker(
            output=marker_path, source_size=source_size, input_rect=args.input_rect,
            duration=args.marker_duration, fps=args.marker_fps,
            low_seconds=args.marker_low, high_seconds=args.marker_high)
        video_url = f"http://{args.advertise_host}:{args.port}/{MARKER_NAME}"
    if args.generate_marker_only:
        print(json.dumps(marker_info, sort_keys=True))
        return 0

    state = ProofState(source_size=source_size, rect=args.rect)
    server = _serve(state, marker_path, args.host, args.port)
    status_url = f"http://{args.advertise_host}:{args.port}/status.json"
    stream_url = overlay_server._camera_stream_url(args.camera_snapshot_url,
                                                    args.camera_stream_url)
    if not stream_url:
        raise SystemExit("no MJPEG stream URL: pass --camera-stream-url or a /shot.jpg snapshot URL")
    restore_video = args.restore_video_url
    trials: list[dict[str, Any]] = []
    valid_points: list[float] = []
    valid_lowers: list[float] = []
    valid_uppers: list[float] = []
    dupes = stale = 0
    restore_error: str | None = None
    restore_status = "not_needed"
    started = False
    run_deadline = time.monotonic() + args.run_timeout
    failed_error: str | None = None
    try:
        _start_android(args.adb_serial, video_url=video_url, status_url=status_url,
                       source_size=source_size)
        started = True
        request = urllib.request.Request(
            stream_url, headers={"Cache-Control": "no-cache", "Accept-Encoding": "identity"})
        input_mask: np.ndarray | None = None
        output_mask: np.ndarray | None = None
        last_key: tuple[int, int] | None = None
        attempt = 0
        # Per-attempt state machine: baseline -> wait_input -> wait_output.
        phase = "baseline"
        low_since: float | None = None
        input_low = output_base = 0.0
        prev: Any = None            # previous accepted frame (for interval bounds)
        in_prev: Any = None
        in_frame: Any = None
        phase_deadline = time.monotonic() + args.input_timeout + args.baseline_arm + 5.0
        with urllib.request.urlopen(request, timeout=args.camera_frame_timeout) as response:
            # Production parser reuse: bounded multipart parts, required
            # X-Frame-Sequence / X-Frame-Age-Ms, receive timestamps on nucat.
            for frame in overlay_server._iter_mjpeg_frames(
                    response, timeout=args.camera_frame_timeout, connection=1):
                if time.monotonic() >= run_deadline:
                    trials.append({"attempt": attempt, "valid": False, "reason": "run_timeout"})
                    break
                if len(valid_points) >= args.trials or attempt > args.max_attempts:
                    break
                if frame.key is not None and frame.key == last_key:
                    dupes += 1
                    continue  # duplication never creates an onset
                if frame.key is not None and last_key is not None and frame.key[1] <= last_key[1]:
                    dupes += 1
                    continue
                if frame.age_ms is None or frame.age_ms > args.max_frame_age_ms:
                    stale += 1
                    continue  # stale evidence cannot trigger or close a trial
                last_key = frame.key
                image = Image.open(BytesIO(frame.jpeg)).convert("L")
                if input_mask is None:
                    input_mask = _mask(image.size, source_size, args.input_rect)
                    output_mask = _mask(image.size, source_size, args.rect)
                arr = np.asarray(image)
                in_luma = float(arr[input_mask].mean())
                out_luma = float(arr[output_mask].mean())
                now = frame.received_at  # single local receive timeline for both edges

                if phase == "baseline":
                    if state.active:
                        state.clear()
                    if now > phase_deadline:
                        attempt += 1
                        trials.append({"attempt": attempt, "valid": False,
                                       "reason": "baseline_timeout"})
                        low_since = None
                        phase_deadline = now + args.input_timeout + args.baseline_arm + 5.0
                    elif in_luma < args.input_low_max_luma:
                        if low_since is None:
                            low_since = now
                        input_low, output_base = in_luma, out_luma
                        if now - low_since >= args.baseline_arm:
                            attempt += 1
                            phase = "wait_input"
                            phase_deadline = now + args.input_timeout
                    else:
                        low_since = None
                elif now > phase_deadline:
                    trials.append({"attempt": attempt, "valid": False,
                                   "reason": f"{phase}_timeout"})
                    state.clear()
                    phase, low_since = "baseline", None
                    phase_deadline = now + args.input_timeout + args.baseline_arm + 5.0
                elif phase == "wait_input":
                    if in_luma >= input_low + args.input_min_brightening:
                        in_prev, in_frame = prev, frame
                        # Causal gate: blackout exists only after this observed edge.
                        state.activate(attempt, _frame_meta(frame))
                        (out_dir / f"trial-{attempt:02d}-input.jpg").write_bytes(frame.jpeg)
                        phase = "wait_output"
                        phase_deadline = now + args.output_timeout
                elif phase == "wait_output":
                    dark = _is_output_decline(
                        output_base, out_luma, min_darkening=args.min_darkening,
                        max_active_luma=args.max_active_luma)
                    if dark and in_frame is not None and in_prev is not None and prev is not None:
                        point = (frame.received_at - in_frame.received_at) * 1000.0
                        lower = (prev.received_at - in_frame.received_at) * 1000.0
                        upper = (frame.received_at - in_prev.received_at) * 1000.0
                        row: dict[str, Any] = {
                            "attempt": attempt, "valid": point >= 0,
                            "point_ms": round(point, 1), "lower_ms": round(lower, 1),
                            "upper_ms": round(upper, 1),
                            "output_baseline_luma": round(output_base, 2),
                            "output_luma": round(out_luma, 2),
                            "min_darkening_threshold": args.min_darkening,
                            "max_active_luma_threshold": args.max_active_luma,
                            "input_frame": _frame_meta(in_frame),
                            "input_prev_frame": _frame_meta(in_prev),
                            "output_frame": _frame_meta(frame),
                            "output_prev_frame": _frame_meta(prev),
                            "activated_at_monotonic": state.activated_at_monotonic}
                        if point < 0:
                            row["reason"] = "negative_delta_rejected"
                        else:
                            valid_points.append(point)
                            valid_lowers.append(lower)
                            valid_uppers.append(upper)
                            (out_dir / f"trial-{attempt:02d}-output.jpg").write_bytes(frame.jpeg)
                        trials.append(row)
                        in_frame = in_prev = None
                        state.clear()
                        phase, low_since = "baseline", None
                        phase_deadline = now + args.input_timeout + args.baseline_arm + 5.0
                prev = frame
    except Exception as exc:
        failed_error = repr(exc)
    finally:
        state.clear()
        if started:
            try:
                if args.restore_off:
                    _stop_android(args.adb_serial)
                    restore_status = "off_restored"
                else:
                    _start_android(args.adb_serial, video_url=restore_video,
                                   status_url=args.restore_status_url, source_size=source_size)
                    restore_status = "video_restored"
            except Exception as exc:
                restore_error = repr(exc)
                restore_status = "failed"
        server.shutdown()
        server.server_close()

    summary: dict[str, Any] = {
        "accepted": len(valid_points) >= args.trials and restore_error is None and failed_error is None,
        "measurement": "fixed_marker_same_camera_input_frame_to_output_frame",
        "bypassed_production_stage": "MobileNet-SSD person detector + source-subtraction + "
                                     "tracking + eye-band geometry (fixed ROI brightening "
                                     "recognition instead); not a production safety latency",
        "clock_limitations": [
            "point/lower/upper use nucat monotonic receive times of actual neighboring frames",
            "bounds cover frame sampling only; exposure/readout and pre-callback delay unknown, "
            "so they are not strict optical onset bounds",
            "X-Frame-Age-Ms is Android callback age, retained per frame, never summed into latency",
            "common camera delay cancellation between input and output observations is assumed, not proven"],
        "stats": {"point": _stats(valid_points), "lower_bound": _stats(valid_lowers),
                  "upper_bound": _stats(valid_uppers)},
        "trials": trials, "valid_trials": len(valid_points),
        "invalid_trials": sum(1 for t in trials if not t["valid"]),
        "duplicate_frames_skipped": dupes,
        "stale_frames_skipped": stale,
        "marker": marker_info, "video_url": video_url, "status_url": status_url,
        "camera_stream_url": stream_url,
        "source_size": list(source_size), "input_rect": list(args.input_rect),
        "output_rect": list(args.rect), "status_requests": state.requests,
        "restored_video_url": restore_video, "restored_status_url": args.restore_status_url,
        "restore_error": restore_error, "restore_status": restore_status,
        "failed_error": failed_error}
    summary["artifact_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode()).hexdigest()
    (out_dir / "proof-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["accepted"] else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--adb-serial", default=DEFAULT_ADB_SERIAL)
    p.add_argument("--camera-snapshot-url", default=DEFAULT_CAMERA_SNAPSHOT_URL)
    p.add_argument("--camera-stream-url", default=DEFAULT_CAMERA_STREAM_URL)
    p.add_argument("--camera-frame-timeout", type=float, default=2.0)
    p.add_argument("--max-frame-age-ms", type=float, default=900.0)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--advertise-host", default="192.168.100.74")
    p.add_argument("--port", type=int, default=8798)
    p.add_argument("--source-size", type=lambda v: tuple(int(x) for x in v.split("x")),
                   default=(1280, 720))
    p.add_argument("--rect", type=_rect, default=(520, 260, 760, 420),
                   help="output blackout ROI in source coordinates")
    p.add_argument("--input-rect", type=_rect, default=(120, 120, 320, 220),
                   help="distinct input marker ROI in source coordinates")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max-attempts", type=int, default=25)
    p.add_argument("--baseline-arm", type=float, default=0.5)
    p.add_argument("--input-timeout", type=float, default=10.0)
    p.add_argument("--output-timeout", type=float, default=5.0)
    p.add_argument("--run-timeout", type=float, default=180.0,
                   help="maximum whole-run wall time, independent of accepted frames")
    p.add_argument("--input-low-max-luma", type=float, default=170.0)
    p.add_argument("--input-min-brightening", type=float, default=25.0)
    p.add_argument("--min-darkening", type=float, default=18.0)
    p.add_argument("--max-active-luma", type=float, default=95.0)
    p.add_argument("--marker-video", type=Path, help="pre-generated marker MP4 (default: generate)")
    p.add_argument("--video-url", help="external marker URL when --marker-video is already hosted")
    p.add_argument("--generate-marker-only", action="store_true")
    p.add_argument("--marker-duration", type=float, default=300.0)
    p.add_argument("--marker-fps", type=int, default=30)
    p.add_argument("--marker-low", type=float, default=1.0)
    p.add_argument("--marker-high", type=float, default=2.0)
    p.add_argument("--restore-video-url",
                   help="production video URL recorded before the run; omit with --restore-off")
    p.add_argument("--restore-status-url", default=DEFAULT_LIVE_STATUS_URL)
    p.add_argument("--restore-off", action="store_true",
                   help="restore baseline with no display task")
    args = p.parse_args()
    if len(args.source_size) != 2:
        raise SystemExit("--source-size must be WIDTHxHEIGHT")
    ix0, iy0, ix1, iy1 = args.input_rect
    ox0, oy0, ox1, oy1 = args.rect
    if not (ix1 <= ox0 or ox1 <= ix0 or iy1 <= oy0 or oy1 <= iy0):
        raise SystemExit("--input-rect and --rect must be disjoint")
    if args.marker_low < args.baseline_arm:
        raise SystemExit("--marker-low must be >= --baseline-arm")
    if args.marker_high <= 0 or args.trials < 1 or args.max_attempts < args.trials:
        raise SystemExit("invalid marker/trial counts")
    if args.run_timeout <= 0:
        raise SystemExit("--run-timeout must be positive")
    if args.restore_off and args.restore_video_url:
        raise SystemExit("--restore-video-url cannot be combined with --restore-off")
    if not args.restore_off and not args.restore_video_url:
        raise SystemExit("provide --restore-video-url or --restore-off")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
