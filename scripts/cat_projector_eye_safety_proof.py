#!/usr/bin/env python3
"""Capture proof that live projector blackout covers a detected person's eyes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_STATUS_URL = "http://127.0.0.1:8787/status.json"
DEFAULT_CAMERA_FRAME_URL = "http://127.0.0.1:8787/camera.jpg"
DEFAULT_DEBUG_CAMERA_URL = "http://127.0.0.1:8787/debug-camera.jpg"
DEFAULT_CAMERA_SNAPSHOT_URL = "http://192.168.100.39:8081/shot.jpg"
DEFAULT_ADB_SERIAL = "192.168.100.39:5555"
RESIDUAL_ONLY_SOURCE = "source_subtracted_human_occluder"


def _read_json_url(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_bytes_url(url: str, *, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _is_direct_detector_active(status: dict[str, Any]) -> tuple[bool, str]:
    if status.get("status") != "active":
        return False, f"status={status.get('status')}"
    debug = status.get("debug") if isinstance(status.get("debug"), dict) else {}
    if debug.get("held_after_last_detection"):
        return False, "held_after_last_detection"
    people = status.get("people")
    if not isinstance(people, list) or not people:
        return False, "no_people"
    detector_people = [person for person in people if person.get("source") != RESIDUAL_ONLY_SOURCE]
    if not detector_people:
        return False, "residual_only_people"
    zones = status.get("zones")
    if not isinstance(zones, list) or not zones:
        return False, "no_zones"
    return True, "direct_detector_active"


def _wait_for_active_status(args: argparse.Namespace, out_dir: Path) -> dict[str, Any] | None:
    deadline = time.monotonic() + args.timeout
    last_status: dict[str, Any] | None = None
    last_reason = "not_checked"
    samples = 0
    while time.monotonic() < deadline:
        samples += 1
        try:
            status = _read_json_url(args.status_url, timeout=args.http_timeout)
        except Exception as exc:
            last_reason = f"status_read_error:{exc}"
            time.sleep(args.poll_interval)
            continue
        last_status = status
        accepted, reason = _is_direct_detector_active(status)
        last_reason = reason
        if accepted:
            status["proof_wait_samples"] = samples
            return status
        time.sleep(args.poll_interval)
    summary = {
        "accepted": False,
        "last_reject_reason": last_reason,
        "samples": samples,
        "last_status": last_status,
    }
    _save_json(out_dir / "proof-timeout-summary.json", summary)
    return None


def _adb_screencap(serial: str) -> bytes:
    return subprocess.check_output(["adb", "-s", serial, "exec-out", "screencap", "-p"], timeout=8)


def _polygon_mask(size: tuple[int, int], polygon: list[list[float]]) -> np.ndarray:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in polygon], fill=255)
    return np.asarray(mask, dtype=np.uint8) > 0


def _bbox_mask(size: tuple[int, int], bbox: list[float]) -> np.ndarray:
    width, height = size
    left, top, right, bottom = (int(round(float(value))) for value in bbox)
    left = max(0, min(width, left))
    right = max(left, min(width, right))
    top = max(0, min(height, top))
    bottom = max(top, min(height, bottom))
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _luma_stats(image: Image.Image, mask: np.ndarray) -> dict[str, float | int | None]:
    luma = np.asarray(image.convert("L"), dtype=np.uint8)
    values = luma[mask]
    if values.size == 0:
        return {"pixels": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "pixels": int(values.size),
        "mean": round(float(values.mean()), 2),
        "p10": round(float(np.percentile(values, 10)), 2),
        "p50": round(float(np.percentile(values, 50)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
    }


def _annotate_physical(image: Image.Image, status: dict[str, Any]) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for zone in status.get("zones") or []:
        eye_band = zone.get("camera_eye_band_xyxy")
        if isinstance(eye_band, list) and len(eye_band) == 4:
            draw.rectangle([float(value) for value in eye_band], outline=(255, 0, 0), width=3)
        projected = zone.get("camera_projected_polygon")
        if isinstance(projected, list) and len(projected) >= 3:
            points = [(float(x), float(y)) for x, y in projected]
            points.append((float(projected[0][0]), float(projected[0][1])))
            draw.line(points, fill=(0, 255, 0), width=3)
    return annotated


def _scale_source_polygon_to_screen(
    polygon: list[list[float]],
    *,
    source_size: list[int] | tuple[int, int],
    screen_size: tuple[int, int],
) -> list[list[float]]:
    source_width, source_height = int(source_size[0]), int(source_size[1])
    screen_width, screen_height = screen_size
    return [
        [float(x) * screen_width / source_width, float(y) * screen_height / source_height]
        for x, y in polygon
    ]


def _analyze_physical(image: Image.Image, status: dict[str, Any]) -> dict[str, Any]:
    zone_summaries: list[dict[str, Any]] = []
    for index, zone in enumerate(status.get("zones") or []):
        projected = zone.get("camera_projected_polygon")
        eye_band = zone.get("camera_eye_band_xyxy")
        if not isinstance(projected, list) or len(projected) < 3:
            continue
        projected_mask = _polygon_mask(image.size, projected)
        if isinstance(eye_band, list) and len(eye_band) == 4:
            eye_mask = _bbox_mask(image.size, eye_band)
        else:
            eye_mask = projected_mask
        overlap_mask = projected_mask & eye_mask
        zone_summaries.append(
            {
                "zone_index": index,
                "camera_bbox_xyxy": zone.get("camera_bbox_xyxy"),
                "camera_eye_band_xyxy": eye_band,
                "camera_projected_polygon": projected,
                "projected_polygon_luma": _luma_stats(image, projected_mask),
                "eye_band_luma": _luma_stats(image, eye_mask),
                "projected_eye_overlap_luma": _luma_stats(image, overlap_mask),
            }
        )
    darkest = None
    for item in zone_summaries:
        mean = item["projected_eye_overlap_luma"].get("mean")
        if mean is not None and (darkest is None or mean < darkest):
            darkest = mean
    return {
        "zone_count": len(zone_summaries),
        "zones": zone_summaries,
        "darkest_projected_eye_overlap_mean": darkest,
        "passes_darkness_threshold": darkest is not None and darkest <= 75.0,
    }


def _analyze_screen(screen: Image.Image, status: dict[str, Any]) -> dict[str, Any]:
    source_size = status.get("source_size") or [1280, 720]
    zone_summaries: list[dict[str, Any]] = []
    for index, zone in enumerate(status.get("zones") or []):
        source_polygon = zone.get("polygon")
        if not isinstance(source_polygon, list) or len(source_polygon) < 3:
            continue
        screen_polygon = _scale_source_polygon_to_screen(
            source_polygon,
            source_size=source_size,
            screen_size=screen.size,
        )
        mask = _polygon_mask(screen.size, screen_polygon)
        luma = _luma_stats(screen, mask)
        zone_summaries.append(
            {
                "zone_index": index,
                "source_polygon": source_polygon,
                "screen_polygon": screen_polygon,
                "screen_polygon_luma": luma,
                "passes_black_screen_threshold": luma.get("p90") is not None and luma["p90"] <= 10.0,
            }
        )
    return {
        "zone_count": len(zone_summaries),
        "zones": zone_summaries,
        "passes_black_screen_threshold": bool(zone_summaries)
        and all(item["passes_black_screen_threshold"] for item in zone_summaries),
    }


def _analyze_geometry(status: dict[str, Any]) -> dict[str, Any]:
    zone_summaries: list[dict[str, Any]] = []
    for index, zone in enumerate(status.get("zones") or []):
        coverage = zone.get("camera_eye_band_coverage")
        zone_summaries.append(
            {
                "zone_index": index,
                "camera_eye_band_coverage": coverage,
                "camera_eye_band_xyxy": zone.get("camera_eye_band_xyxy"),
                "camera_projected_polygon": zone.get("camera_projected_polygon"),
                "passes_eye_band_coverage_threshold": isinstance(coverage, int | float) and float(coverage) >= 0.9,
            }
        )
    return {
        "zone_count": len(zone_summaries),
        "zones": zone_summaries,
        "passes_eye_band_coverage_threshold": bool(zone_summaries)
        and all(item["passes_eye_band_coverage_threshold"] for item in zone_summaries),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    parser.add_argument("--camera-frame-url", default=DEFAULT_CAMERA_FRAME_URL)
    parser.add_argument("--debug-camera-url", default=DEFAULT_DEBUG_CAMERA_URL)
    parser.add_argument("--camera-snapshot-url", default=DEFAULT_CAMERA_SNAPSHOT_URL)
    parser.add_argument("--adb-serial", default=DEFAULT_ADB_SERIAL)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--poll-interval", type=float, default=0.08)
    parser.add_argument("--render-wait", type=float, default=0.45)
    parser.add_argument("--http-timeout", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    status = _wait_for_active_status(args, out_dir)
    if status is None:
        print(json.dumps({"accepted": False, "out_dir": str(out_dir)}, sort_keys=True))
        return 2

    _save_json(out_dir / "status-active.json", status)
    detector_camera_path = out_dir / "detector-camera.jpg"
    detector_camera_path.write_bytes(_read_bytes_url(args.camera_frame_url, timeout=args.http_timeout))
    detector_debug_path = out_dir / "detector-debug-camera.jpg"
    detector_debug_path.write_bytes(_read_bytes_url(args.debug_camera_url, timeout=args.http_timeout))
    time.sleep(args.render_wait)
    render_status = _read_json_url(args.status_url, timeout=args.http_timeout)
    render_status["proof_render_wait_seconds"] = args.render_wait
    _save_json(out_dir / "status-after-render.json", render_status)
    physical_bytes = _read_bytes_url(args.camera_snapshot_url, timeout=args.http_timeout)
    physical_path = out_dir / "physical-after-render.jpg"
    physical_path.write_bytes(physical_bytes)
    screen_path = out_dir / "screen-after-render.png"
    screen_path.write_bytes(_adb_screencap(args.adb_serial))

    physical = Image.open(physical_path).convert("RGB")
    screen = Image.open(screen_path).convert("RGB")
    detector_camera = Image.open(detector_camera_path).convert("RGB")
    detector_debug = Image.open(detector_debug_path).convert("RGB")
    annotated = _annotate_physical(physical, render_status)
    annotated.save(out_dir / "physical-after-render-annotated.jpg", quality=92)
    detector_annotated = _annotate_physical(detector_debug, status)
    detector_annotated.save(out_dir / "detector-debug-camera-annotated.jpg", quality=92)
    detector_camera_annotated = _annotate_physical(detector_camera, status)
    detector_camera_annotated.save(out_dir / "detector-camera-annotated.jpg", quality=92)
    analysis = _analyze_physical(physical, render_status)
    detector_camera_analysis = _analyze_physical(detector_camera, status)
    screen_analysis = _analyze_screen(screen, render_status)
    geometry_analysis = _analyze_geometry(render_status)
    passes_eye_safety_proof = (
        render_status.get("status") == "active"
        and screen_analysis["passes_black_screen_threshold"]
        and geometry_analysis["passes_eye_band_coverage_threshold"]
    )
    summary = {
        "accepted": passes_eye_safety_proof,
        "out_dir": str(out_dir),
        "render_wait_seconds": args.render_wait,
        "status_path": str(out_dir / "status-active.json"),
        "status_after_render_path": str(out_dir / "status-after-render.json"),
        "detector_camera_path": str(detector_camera_path),
        "detector_debug_path": str(detector_debug_path),
        "physical_path": str(physical_path),
        "screen_path": str(screen_path),
        "analysis": analysis,
        "detector_camera_analysis": detector_camera_analysis,
        "screen_analysis": screen_analysis,
        "geometry_analysis": geometry_analysis,
        "passes_eye_safety_proof": passes_eye_safety_proof,
        "people_sources": [person.get("source") for person in status.get("people") or []],
        "performance": status.get("performance"),
    }
    _save_json(out_dir / "proof-summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if passes_eye_safety_proof else 1


if __name__ == "__main__":
    raise SystemExit(main())
