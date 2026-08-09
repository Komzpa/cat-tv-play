from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts import cat_projector_sam_service as service


class FakePredictor:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.set_image_calls = 0
        self.predict_calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.delay = delay
        self._lock = threading.Lock()

    def set_image(self, image: np.ndarray) -> None:
        assert image.shape == (8, 10, 3)
        self.set_image_calls += 1
        time.sleep(self.delay)

    def predict(self, *, point_coords: Any, point_labels: Any, multimask_output: bool) -> Any:
        assert multimask_output is True
        assert point_coords.shape[1] == 2
        assert point_labels.shape[0] == point_coords.shape[0]
        with self._lock:
            self.predict_calls += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(self.delay)
            masks = np.zeros((3, 8, 10), dtype=bool)
            masks[:, 2:6, 3:7] = True
            scores = np.asarray([0.2, 0.8, 0.4], dtype=np.float32)
            return masks, scores, None
        finally:
            with self._lock:
                self.active_calls -= 1


def _reset_service(predictor: FakePredictor) -> None:
    service._PREDICTOR = predictor  # noqa: SLF001
    service._CACHED_IMAGE_KEY = None  # noqa: SLF001
    service._MODEL_INFO.clear()


def _image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (10, 8), color).save(path)


def test_reuses_embedding_by_content_across_paths(tmp_path: Path) -> None:
    predictor = FakePredictor()
    _reset_service(predictor)
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _image(first, (10, 20, 30))
    second.write_bytes(first.read_bytes())

    first_result = service.segment({"image_path": str(first), "positive_points": [[4, 3]]})
    second_result = service.segment({"image_path": str(second), "positive_points": [[5, 3]]})

    assert predictor.set_image_calls == 1
    assert predictor.predict_calls == 2
    assert first_result["cache"] == {"hit": False, "key": second_result["cache"]["key"], "capacity": 1}
    assert second_result["cache"]["hit"] is True
    assert first_result["timing"]["set_image_ms"] >= 0
    assert second_result["timing"]["set_image_ms"] == 0
    assert second_result["timing"]["total_ms"] >= second_result["timing"]["predict_ms"]


def test_changed_content_at_same_path_invalidates_embedding(tmp_path: Path) -> None:
    predictor = FakePredictor()
    _reset_service(predictor)
    image_path = tmp_path / "frame.jpg"
    _image(image_path, (10, 20, 30))
    first = service.segment({"image_path": str(image_path), "positive_points": [[4, 3]]})
    _image(image_path, (30, 20, 10))
    second = service.segment({"image_path": str(image_path), "positive_points": [[4, 3]]})

    assert predictor.set_image_calls == 2
    assert first["cache"]["key"] != second["cache"]["key"]
    assert second["cache"]["hit"] is False


def test_predictor_and_embedding_are_serialized_for_concurrent_prompts(tmp_path: Path) -> None:
    predictor = FakePredictor(delay=0.01)
    _reset_service(predictor)
    image_path = tmp_path / "frame.jpg"
    _image(image_path, (10, 20, 30))

    from concurrent.futures import ThreadPoolExecutor

    payloads = [
        {"image_path": str(image_path), "positive_points": [[4, 3]]},
        {"image_path": str(image_path), "positive_points": [[5, 3]]},
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(service.segment, payloads))

    assert len(results) == 2
    assert predictor.set_image_calls == 1
    assert predictor.predict_calls == 2
    assert predictor.max_active_calls == 1
    assert {result["cache"]["hit"] for result in results} == {False, True}
