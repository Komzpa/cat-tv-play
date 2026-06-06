"""Cat detector abstraction for segmentation-first jump measurement.

The legacy contrast/CatBoost detector is kept as an explicit backend for
fallback, debugging, and hard-negative mining. Runtime measurement code should
consume masks from segmentation detections when they are available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

BBoxXYWH = tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectorContext:
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    source_path: str | None = None
    source_video_path: str | None = None
    recording_id: str | None = None
    calibration_id: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CatDetection:
    bbox_xywh: BBoxXYWH
    score: float
    source: str
    model_id: str
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    mask: np.ndarray | None = None
    mask_polygon: tuple[tuple[float, float], ...] = ()
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def has_mask(self) -> bool:
        return self.mask is not None or bool(self.mask_polygon)

    def mask_area_px(self) -> int | None:
        if self.mask is None:
            return None
        return int(np.asarray(self.mask, dtype=bool).sum())

    def to_debug_row(self) -> dict[str, Any]:
        return {
            "bbox": _format_bbox_xywh(self.bbox_xywh),
            "p": round(float(self.score), 4),
            "source": self.source,
            "model_id": self.model_id,
            "has_mask": self.has_mask,
            "mask_area_px": self.mask_area_px(),
            "debug": self.debug,
        }


class CatDetector(Protocol):
    model_id: str
    source: str

    def detect(self, frame: Image.Image, context: DetectorContext | None = None) -> list[CatDetection]:
        """Return cat detections for one frame."""


@dataclass(frozen=True)
class DetectorConfig:
    backend: str = "auto"
    model_path: str | None = None
    device: str | None = None
    confidence_threshold: float = 0.5
    legacy_model_path: str | None = None
    legacy_metadata_path: str | None = None
    allow_fake: bool = False


class DetectorUnavailableError(RuntimeError):
    """Raised when the configured detector cannot run in this environment."""


class FakeSegmentationDetector:
    """Tiny deterministic detector for tests and smoke checks."""

    source = "fake_segmentation"
    model_id = "fake_segmentation_v1"

    def __init__(
        self,
        *,
        bbox_xywh: BBoxXYWH = (20.0, 30.0, 40.0, 50.0),
        mask: np.ndarray | None = None,
        score: float = 0.9,
    ) -> None:
        self.bbox_xywh = bbox_xywh
        self._mask = mask
        self.score = score

    def detect(self, frame: Image.Image, context: DetectorContext | None = None) -> list[CatDetection]:
        mask = self._mask
        if mask is None:
            mask = np.zeros((frame.height, frame.width), dtype=bool)
            x, y, width, height = (int(round(value)) for value in self.bbox_xywh)
            mask[max(0, y) : min(frame.height, y + height), max(0, x) : min(frame.width, x + width)] = True
        return [
            CatDetection(
                bbox_xywh=self.bbox_xywh,
                score=self.score,
                source=self.source,
                model_id=self.model_id,
                frame_index=context.frame_index if context else None,
                timestamp_seconds=context.timestamp_seconds if context else None,
                mask=mask,
                debug={"test_detector": True},
            )
        ]


class LegacyContrastDetector:
    """Explicit legacy contrast/components + CatBoost candidate scorer."""

    source = "legacy_contrast_catboost"

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        min_probability: float = 0.0,
    ) -> None:
        from scripts import cat_projector_frame_detector as legacy

        self._legacy = legacy
        self._model, self._metadata = legacy.load_model(
            model_path or legacy.DEFAULT_MODEL_PATH,
            metadata_path or legacy.DEFAULT_METADATA_PATH,
        )
        self.model_id = str(self._metadata.get("model_path") or model_path or legacy.DEFAULT_MODEL_PATH)
        self.min_probability = min_probability

    def _source_explains_candidate(
        self,
        frame: Image.Image,
        bbox_xywh: BBoxXYWH,
        reference_frames: list[Image.Image],
        *,
        residual_threshold: float = 26.0,
        source_bright_threshold: int = 150,
        max_dark_fraction: float = 0.16,
        min_bright_coverage: float = 0.25,
    ) -> bool:
        x, y, width, height = bbox_xywh
        left = max(0, min(frame.width, int(np.floor(x))))
        top = max(0, min(frame.height, int(np.floor(y))))
        right = max(left + 1, min(frame.width, int(np.ceil(x + width))))
        bottom = max(top + 1, min(frame.height, int(np.ceil(y + height))))
        for source_frame in reference_frames:
            residual, warped_source = self._legacy.source_subtracted_residual(
                frame.convert("RGB"),
                source_frame=source_frame,
            )
            residual_patch = residual[top:bottom, left:right]
            source_patch = warped_source[top:bottom, left:right]
            bright = source_patch > source_bright_threshold
            bright_coverage = float(bright.mean()) if bright.size else 0.0
            if bright_coverage < min_bright_coverage:
                continue
            dark_fraction = float(((residual_patch > residual_threshold) & bright).mean())
            if dark_fraction <= max_dark_fraction:
                return True
        return False

    def detect(self, frame: Image.Image, context: DetectorContext | None = None) -> list[CatDetection]:
        candidates = None
        source_frame = (context.debug or {}).get("projector_source_frame") if context else None
        reference_frames = list((context.debug or {}).get("projector_source_reference_frames") or []) if context else []
        if source_frame is not None:
            candidates = self._legacy.detect_source_subtracted_candidate_components(
                frame.convert("RGB"),
                source_frame=source_frame,
                room_background=(context.debug or {}).get("room_background") if context else None,
                residual_baseline=(context.debug or {}).get("residual_baseline") if context else None,
            )
        predictions = sorted(
            self._legacy.score_candidates(
                frame.convert("RGB"),
                model=self._model,
                metadata=self._metadata,
                candidates=candidates,
            ),
            key=lambda item: item.cat_probability,
            reverse=True,
        )
        detector_source = self.source
        if candidates is not None:
            detector_source = "legacy_contrast_catboost_source_subtracted"
        detections: list[CatDetection] = []
        for prediction in predictions:
            if prediction.candidate is None or prediction.cat_probability < self.min_probability:
                continue
            bbox_xywh = tuple(float(value) for value in prediction.candidate.bbox_xywh)
            if candidates is not None and self._source_explains_candidate(frame, bbox_xywh, reference_frames):
                continue
            context_debug = dict((context.debug or {}) if context else {})
            context_debug.pop("projector_source_frame", None)
            context_debug.pop("projector_source_reference_frames", None)
            detections.append(
                CatDetection(
                    bbox_xywh=bbox_xywh,
                    score=float(prediction.cat_probability),
                    source=prediction.candidate.source,
                    model_id=str(prediction.model_path or self.model_id),
                    frame_index=context.frame_index if context else None,
                    timestamp_seconds=context.timestamp_seconds if context else None,
                    mask=None,
                    debug={
                        "backend": detector_source,
                        "features": prediction.features,
                        **context_debug,
                    },
                )
            )
        return detections


class UltralyticsSegmentationDetector:
    """Ultralytics YOLO segmentation backend.

    The dependency and weights are optional. Construction fails cleanly when they
    are absent; tests use FakeSegmentationDetector and never download weights.
    """

    source = "ultralytics_yolo_segmentation"

    def __init__(
        self,
        model_path: Path,
        *,
        device: str | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        if not model_path.exists():
            raise DetectorUnavailableError(f"segmentation model does not exist: {model_path}")
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional runtime dependency.
            raise DetectorUnavailableError("ultralytics is not installed; install it for YOLO segmentation") from exc
        self._model = YOLO(str(model_path))
        self.model_id = str(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: Image.Image, context: DetectorContext | None = None) -> list[CatDetection]:
        kwargs: dict[str, Any] = {"conf": self.confidence_threshold, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self._model.predict(np.asarray(frame.convert("RGB")), **kwargs)
        detections: list[CatDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            if boxes is None or masks is None or masks.data is None:
                continue
            names = getattr(result, "names", {}) or {}
            box_xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else [None] * len(scores)
            mask_data = masks.data.cpu().numpy().astype(bool)
            polygons = getattr(masks, "xy", None) or []
            for index, score in enumerate(scores):
                class_id = int(classes[index]) if classes[index] is not None else None
                class_name = str(names.get(class_id, class_id if class_id is not None else "cat"))
                if class_name not in {"cat", "sher", "Sher", "kitten"} and len(names) > 1:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box_xyxy[index])
                polygon = tuple((float(x), float(y)) for x, y in polygons[index]) if index < len(polygons) else ()
                mask = _mask_in_frame_coordinates(mask_data[index], polygon, width=frame.width, height=frame.height)
                detections.append(
                    CatDetection(
                        bbox_xywh=(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                        score=float(score),
                        source=self.source,
                        model_id=self.model_id,
                        frame_index=context.frame_index if context else None,
                        timestamp_seconds=context.timestamp_seconds if context else None,
                        mask=mask,
                        mask_polygon=polygon,
                        debug={"class_id": class_id, "class_name": class_name},
                    )
                )
        return sorted(detections, key=lambda item: item.score, reverse=True)


def build_detector(config: DetectorConfig) -> CatDetector:
    backend = (config.backend or "auto").strip().lower()
    if backend == "fake":
        if not config.allow_fake and os.environ.get("CAT_PROJECTOR_ALLOW_FAKE_DETECTOR") != "1":
            raise DetectorUnavailableError(
                "fake detector requires allow_fake=True or CAT_PROJECTOR_ALLOW_FAKE_DETECTOR=1"
            )
        return FakeSegmentationDetector(score=config.confidence_threshold)
    if backend in {"segmentation", "yolo", "ultralytics"} or (backend == "auto" and config.model_path):
        if not config.model_path:
            raise DetectorUnavailableError("segmentation backend requires model_path")
        return UltralyticsSegmentationDetector(
            Path(config.model_path).expanduser(),
            device=config.device,
            confidence_threshold=config.confidence_threshold,
        )
    if backend in {"auto", "legacy"}:
        return LegacyContrastDetector(
            model_path=Path(config.legacy_model_path).expanduser() if config.legacy_model_path else None,
            metadata_path=Path(config.legacy_metadata_path).expanduser() if config.legacy_metadata_path else None,
            min_probability=0.0,
        )
    raise ValueError(f"unknown cat detector backend: {config.backend}")


def _format_bbox_xywh(bbox: BBoxXYWH) -> str:
    return ",".join(str(int(round(float(value)))) for value in bbox)


def _mask_in_frame_coordinates(
    raw_mask: np.ndarray,
    polygon: tuple[tuple[float, float], ...],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if polygon:
        image = Image.new("1", (width, height), 0)
        ImageDraw.Draw(image).polygon(list(polygon), fill=1)
        return np.asarray(image, dtype=bool)
    mask = np.asarray(raw_mask, dtype=bool)
    if mask.shape == (height, width):
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0
