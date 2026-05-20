#!/usr/bin/env python3
"""Local Segment Anything service for Cat Projector label review.

The review backend proxies prompt requests to this local-only HTTP service so
the browser never uploads household frames to a cloud API.
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_PREDICTOR: Any | None = None
_MODEL_INFO: dict[str, Any] = {}


def _normalise_point(point: Any) -> tuple[float, float] | None:
    if isinstance(point, dict):
        if "x" in point and "y" in point:
            return float(point["x"]), float(point["y"])
        if "point" in point:
            return _normalise_point(point["point"])
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return None


def _component_polygon(mask: np.ndarray, max_vertices: int = 64) -> list[dict[str, float]]:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        cv2 = None

    if cv2 is not None:
        contours, _hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.5, cv2.arcLength(contour, True) / max_vertices)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        return [{"x": round(float(x), 2), "y": round(float(y), 2)} for x, y in points]

    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return []
    x0, y0 = float(xs.mean()), float(ys.mean())
    boundary: list[tuple[int, int]] = []
    height, width = mask.shape
    for y, x in zip(ys, xs, strict=True):
        if x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1 or not mask[y - 1 : y + 2, x - 1 : x + 2].all():
            boundary.append((int(x), int(y)))
    buckets: dict[int, tuple[float, float, float]] = {}
    for x, y in boundary:
        angle = np.arctan2(y - y0, x - x0)
        bucket = int((angle + np.pi) / (2 * np.pi) * max_vertices)
        distance = (x - x0) ** 2 + (y - y0) ** 2
        if bucket not in buckets or distance > buckets[bucket][2]:
            buckets[bucket] = (float(x), float(y), float(distance))
    return [
        {"x": round(x, 2), "y": round(y, 2)}
        for x, y, _distance in sorted(buckets.values(), key=lambda item: np.arctan2(item[1] - y0, item[0] - x0))
    ]


def _bbox(mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    return {
        "x": round(float(xs.min()), 2),
        "y": round(float(ys.min()), 2),
        "width": round(float(xs.max() - xs.min() + 1), 2),
        "height": round(float(ys.max() - ys.min() + 1), 2),
    }


def _load_predictor() -> Any:
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    checkpoint = os.environ.get("CAT_PROJECTOR_SAM_CHECKPOINT", "").strip()
    if not checkpoint:
        raise RuntimeError("CAT_PROJECTOR_SAM_CHECKPOINT is required")
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"SAM checkpoint not found: {checkpoint_path}")

    model_type = os.environ.get("CAT_PROJECTOR_SAM_MODEL_TYPE", "vit_b").strip() or "vit_b"
    device = os.environ.get("CAT_PROJECTOR_SAM_DEVICE", "auto").strip() or "auto"

    try:
        import torch  # type: ignore[import-not-found]
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("Install official Segment Anything dependencies: torch and segment-anything") from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    model.to(device=device)
    _PREDICTOR = SamPredictor(model)
    _MODEL_INFO.update({"model_type": model_type, "checkpoint": str(checkpoint_path), "device": device})
    return _PREDICTOR


def segment(payload: dict[str, Any]) -> dict[str, Any]:
    positives = [_normalise_point(point) for point in payload.get("positive_points") or []]
    positives = [point for point in positives if point is not None]
    negatives = [_normalise_point(point) for point in payload.get("negative_points") or []]
    negatives = [point for point in negatives if point is not None]
    if not positives:
        raise ValueError("positive_points is required")

    image_path = Path(str(payload.get("image_path") or "")).expanduser()
    if not image_path.is_file():
        raise ValueError(f"image_path does not exist: {image_path}")

    predictor = _load_predictor()
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))

    predictor.set_image(rgb)
    points = positives + negatives
    point_coords = np.asarray(points, dtype=np.float32)
    point_labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    masks, scores, _logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    order = np.argsort(scores)[::-1]
    best_mask = masks[int(order[0])].astype(bool)
    polygon = _component_polygon(best_mask)
    if len(polygon) < 3:
        raise ValueError("SAM returned an empty mask")
    return {
        "kind": "cat_projector_label_review_mask_v1",
        "source": "official_segment_anything",
        "polygon": polygon,
        "bbox_xywh": _bbox(best_mask),
        "score": round(float(scores[int(order[0])]), 4),
        "model": _MODEL_INFO,
    }


class SamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[cat-projector-sam] {fmt % args}", flush=True)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            try:
                _load_predictor()
                self._send_json({"ok": True, "model": _MODEL_INFO})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/segment":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            self._send_json(segment(payload))
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--warmup", action="store_true", help="Load the SAM checkpoint before accepting requests.")
    args = parser.parse_args()
    if args.warmup:
        _load_predictor()
    server = ThreadingHTTPServer((args.host, args.port), SamHandler)
    print(f"cat_projector_sam listening on http://{args.host}:{args.port}/segment", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
