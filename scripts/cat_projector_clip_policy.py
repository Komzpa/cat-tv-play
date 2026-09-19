"""Shared Cat Projector camera-recording policy."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_config as site_config  # noqa: E402

DEFAULT_HUMAN_DETECTOR_DIR = site_config.models_root() / "opencv-mobilenet-ssd"
DEFAULT_HUMAN_DETECTOR_PROTOTXT = DEFAULT_HUMAN_DETECTOR_DIR / "MobileNetSSD_deploy.prototxt"
DEFAULT_HUMAN_DETECTOR_MODEL = DEFAULT_HUMAN_DETECTOR_DIR / "MobileNetSSD_deploy.caffemodel"
PERSON_CLASS_ID = 15


@dataclass(frozen=True)
class CameraHumanCheck:
    detected: bool
    reason: str
    checked_frames: int = 0
    frame: str | None = None
    confidence: float | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    error: str | None = None

    @property
    def should_skip(self) -> bool:
        return self.detected or self.reason == "human_check_unavailable"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _extract_sample_frames(chunks: list[Path], frames_dir: Path, *, sample_fps: int, scale_width: int) -> list[Path]:
    for chunk in chunks:
        if not chunk.exists():
            continue
        pattern = frames_dir / f"{chunk.stem}_%05d.jpg"
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
            return [
                frames_dir
                / f"__ffmpeg_failed__{chunk.stem}__{result.stderr[-160:].replace('/', '_').replace(chr(10), ' ')}"
            ]
    return sorted(path for path in frames_dir.glob("*.jpg") if path.is_file())


def detect_camera_human_presence(
    chunks: list[Path],
    *,
    sample_fps: int = 1,
    scale_width: int = 640,
    min_confidence: float = 0.35,
    prototxt: Path = DEFAULT_HUMAN_DETECTOR_PROTOTXT,
    model: Path = DEFAULT_HUMAN_DETECTOR_MODEL,
) -> CameraHumanCheck:
    if not chunks:
        return CameraHumanCheck(False, "no_chunks")
    if not prototxt.exists() or not model.exists():
        return CameraHumanCheck(
            True,
            "human_check_unavailable",
            error=f"missing OpenCV MobileNet SSD model files under {prototxt.parent}",
        )
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception as exc:
        return CameraHumanCheck(True, "human_check_unavailable", error=f"cv2 import failed: {exc}")

    try:
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
    except Exception as exc:
        return CameraHumanCheck(True, "human_check_unavailable", error=f"failed to load human detector: {exc}")

    with tempfile.TemporaryDirectory(prefix="cat-projector-human-check-") as tmpdir:
        frames = _extract_sample_frames(
            chunks,
            Path(tmpdir),
            sample_fps=sample_fps,
            scale_width=scale_width,
        )
        if len(frames) == 1 and frames[0].name.startswith("__ffmpeg_failed__"):
            return CameraHumanCheck(True, "human_check_unavailable", error=frames[0].name)
        for frame_index, frame_path in enumerate(frames, start=1):
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for index in range(detections.shape[2]):
                confidence = float(detections[0, 0, index, 2])
                class_id = int(detections[0, 0, index, 1])
                if class_id != PERSON_CLASS_ID or confidence < min_confidence:
                    continue
                x0, y0, x1, y1 = (detections[0, 0, index, 3:7] * np.array([width, height, width, height])).astype(int)
                return CameraHumanCheck(
                    True,
                    "camera_human_detected",
                    checked_frames=frame_index,
                    frame=frame_path.name,
                    confidence=round(confidence, 4),
                    bbox_xyxy=(int(x0), int(y0), int(x1), int(y1)),
                )
    return CameraHumanCheck(False, "no_camera_human_detected", checked_frames=len(frames))
