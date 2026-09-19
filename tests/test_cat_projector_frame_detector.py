from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_frame_detector.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("cat_projector_frame_detector", SCRIPT_PATH)
detector = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = detector
SCRIPT_SPEC.loader.exec_module(detector)

#: The household corpus is not published with this repository; the tests that
#: read it skip unless an installation put a copy beside the checkout.
CORPUS_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "cat-tv-learning"
requires_corpus = pytest.mark.skipif(
    not (CORPUS_ROOT / "detector-training").is_dir(),
    reason="local household corpus under datasets/cat-tv-learning is not present",
)


def test_extract_candidate_features_marks_projection_overlap() -> None:
    image = Image.new("RGB", (1280, 720), "white")
    features = detector.extract_candidate_features(image, (100, 120, 80, 90))

    assert features["area_ratio"] > 0
    assert features["inside_projection_area_ratio"] > 0.5
    assert features["outside_projection_area_ratio"] < 0.5


def test_detect_dark_candidate_components_returns_candidate_top() -> None:
    image = Image.new("RGB", (320, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 40, 105, 150), fill="black")

    candidates = detector.detect_dark_candidate_components(image)

    assert candidates
    assert candidates[0].top_y_px <= 42
    assert candidates[0].area_px > 100


def test_source_subtracted_candidates_ignore_projected_prey_and_keep_cat() -> None:
    source = Image.new("RGB", (1280, 720), "white")
    ImageDraw.Draw(source).ellipse((500, 220, 570, 280), fill="black")
    camera = Image.fromarray(detector._warp_source_to_camera(source, 1280, 720)).convert("RGB")  # noqa: SLF001
    ImageDraw.Draw(camera).rectangle((620, 430, 700, 620), fill="black")

    candidates = detector.detect_source_subtracted_candidate_components(camera, source_frame=source)

    assert candidates
    best = candidates[0]
    x, y, width, height = best.bbox_xywh
    assert x <= 625 <= x + width
    assert y <= 435 <= y + height
    assert all(not (390 <= candidate.bbox_xywh[0] <= 470 and candidate.bbox_xywh[1] < 320) for candidate in candidates)


def test_source_subtracted_candidates_accept_residual_baseline() -> None:
    source = Image.new("RGB", (1280, 720), "white")
    camera = Image.fromarray(detector._warp_source_to_camera(source, 1280, 720)).convert("RGB")  # noqa: SLF001
    ImageDraw.Draw(camera).rectangle((620, 430, 700, 620), fill="black")
    residual, _warped = detector.source_subtracted_residual(camera, source_frame=source)
    baseline = residual.copy()
    baseline[430:620, 620:700] = 0

    candidates = detector.detect_source_subtracted_candidate_components(
        camera,
        source_frame=source,
        residual_baseline=baseline,
    )

    assert candidates
    x, y, width, height = candidates[0].bbox_xywh
    assert x <= 625 <= x + width
    assert y <= 435 <= y + height


def test_source_subtracted_candidates_use_room_background_for_outside_projection_motion() -> None:
    source = Image.new("RGB", (1280, 720), "white")
    background = Image.new("RGB", (1280, 720), (200, 200, 200))
    ImageDraw.Draw(background).rectangle((700, 640, 760, 690), fill=(70, 70, 70))
    camera = background.copy()
    ImageDraw.Draw(camera).rectangle((500, 650, 580, 710), fill="black")

    assert not detector.detect_source_subtracted_candidate_components(camera, source_frame=source)

    candidates = detector.detect_source_subtracted_candidate_components(
        camera,
        source_frame=source,
        room_background=background,
    )

    assert candidates
    x, y, width, height = candidates[0].bbox_xywh
    assert x <= 505 <= x + width
    assert y <= 655 <= y + height
    assert all(not (700 <= candidate.bbox_xywh[0] <= 760) for candidate in candidates)


def test_load_model_falls_back_to_bundled_detector(tmp_path: Path, monkeypatch) -> None:
    local_model = tmp_path / "missing-local" / "cat_projector_candidate_detector_v1.cbm"
    local_metadata = tmp_path / "missing-local" / "cat_projector_candidate_detector_v1.metadata.json"
    bundled_model = tmp_path / "repo-model" / "cat_projector_candidate_detector_v1.cbm"
    bundled_metadata = tmp_path / "repo-model" / "cat_projector_candidate_detector_v1.metadata.json"
    bundled_model.parent.mkdir(parents=True)
    bundled_model.write_bytes(b"fake bundled catboost model")
    bundled_metadata.write_text(json.dumps({"created_at": "bundled-test"}), encoding="utf-8")
    loaded_paths: list[str] = []

    class FakeCatBoostClassifier:
        def load_model(self, path: str) -> None:
            loaded_paths.append(path)

    monkeypatch.setattr(detector, "DEFAULT_MODEL_PATH", local_model)
    monkeypatch.setattr(detector, "DEFAULT_METADATA_PATH", local_metadata)
    monkeypatch.setattr(detector, "BUNDLED_MODEL_PATH", bundled_model)
    monkeypatch.setattr(detector, "BUNDLED_METADATA_PATH", bundled_metadata)
    monkeypatch.setattr(detector, "CatBoostClassifier", FakeCatBoostClassifier)
    monkeypatch.setattr(detector, "Pool", object())

    assert detector.default_model_available()

    _model, metadata = detector.load_model(local_model, local_metadata)

    assert loaded_paths == [str(bundled_model)]
    assert metadata["created_at"] == "bundled-test"
    assert metadata["model_path"] == str(bundled_model)
    assert metadata["metadata_path"] == str(bundled_metadata)


def test_load_model_does_not_fall_back_for_missing_explicit_custom_path(tmp_path: Path, monkeypatch) -> None:
    default_model = tmp_path / "default" / "cat_projector_candidate_detector_v1.cbm"
    default_metadata = tmp_path / "default" / "cat_projector_candidate_detector_v1.metadata.json"
    custom_model = tmp_path / "custom" / "cat_projector_candidate_detector_v1.cbm"
    custom_metadata = tmp_path / "custom" / "cat_projector_candidate_detector_v1.metadata.json"
    bundled_model = tmp_path / "repo-model" / "cat_projector_candidate_detector_v1.cbm"
    bundled_metadata = tmp_path / "repo-model" / "cat_projector_candidate_detector_v1.metadata.json"
    bundled_model.parent.mkdir(parents=True)
    bundled_model.write_bytes(b"fake bundled catboost model")
    bundled_metadata.write_text(json.dumps({"created_at": "bundled-test"}), encoding="utf-8")

    monkeypatch.setattr(detector, "DEFAULT_MODEL_PATH", default_model)
    monkeypatch.setattr(detector, "DEFAULT_METADATA_PATH", default_metadata)
    monkeypatch.setattr(detector, "BUNDLED_MODEL_PATH", bundled_model)
    monkeypatch.setattr(detector, "BUNDLED_METADATA_PATH", bundled_metadata)
    monkeypatch.setattr(detector, "Pool", object())

    resolved_model, resolved_metadata = detector._resolve_model_paths(custom_model, custom_metadata)  # noqa: SLF001

    assert resolved_model == custom_model
    assert resolved_metadata == custom_metadata


@requires_corpus
@pytest.mark.skipif(detector.CatBoostClassifier is None or detector.Pool is None, reason="CatBoost is required")
def test_bundled_detector_keeps_right_side_blurred_jump_positive() -> None:
    image_path = (
        CORPUS_ROOT
        / "detector-training/cat-projector-right-blurred-jump-20260518/"
        / "frames/chunk_0154_t1.00_right_blurred_jump.jpg"
    )
    image = Image.open(image_path).convert("RGB")
    model, metadata = detector.load_model(detector.BUNDLED_MODEL_PATH, detector.BUNDLED_METADATA_PATH)
    candidate = detector.Candidate((638, 402, 132, 214), 638, 402, 132 * 214, "regression")

    predictions = detector.score_candidates(image, model=model, metadata=metadata, candidates=[candidate])

    assert predictions
    assert predictions[0].cat_probability >= 0.60


@requires_corpus
@pytest.mark.skipif(detector.CatBoostClassifier is None or detector.Pool is None, reason="CatBoost is required")
def test_bundled_detector_keeps_live1269_visible_sher_positive() -> None:
    dataset = CORPUS_ROOT / "detector-training/cat-projector-live1269-no-cat-20260518"
    model, metadata = detector.load_model(detector.BUNDLED_MODEL_PATH, detector.BUNDLED_METADATA_PATH)
    probabilities: list[float] = []

    with (dataset / "labels.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "candidate_bbox" not in row["candidate_reason"]:
                continue
            image = Image.open(dataset / row["image_relpath"]).convert("RGB")
            bbox = detector._parse_bbox_xywh(row["candidate_bbox_xywh"])  # noqa: SLF001
            assert bbox is not None
            candidate = detector.Candidate(bbox, bbox[0], bbox[1], int(bbox[2] * bbox[3]), "regression")
            predictions = detector.score_candidates(image, model=model, metadata=metadata, candidates=[candidate])
            assert predictions
            probabilities.append(predictions[0].cat_probability)

    assert probabilities
    assert min(probabilities) >= 0.60


@requires_corpus
@pytest.mark.skipif(detector.CatBoostClassifier is None or detector.Pool is None, reason="CatBoost is required")
def test_bundled_detector_rejects_projected_source_video_objects() -> None:
    dataset = CORPUS_ROOT / "detector-training/cat-projector-projected-source-not-sher-20260518"
    model, metadata = detector.load_model(detector.BUNDLED_MODEL_PATH, detector.BUNDLED_METADATA_PATH)
    probabilities: list[float] = []
    highlight_probabilities: list[float] = []

    with (dataset / "labels.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image = Image.open(dataset / row["image_relpath"]).convert("RGB")
            bbox = detector._parse_bbox_xywh(row["candidate_bbox_xywh"])  # noqa: SLF001
            assert bbox is not None
            candidate = detector.Candidate(bbox, bbox[0], bbox[1], int(bbox[2] * bbox[3]), "regression")
            predictions = detector.score_candidates(image, model=model, metadata=metadata, candidates=[candidate])
            assert predictions
            probability = predictions[0].cat_probability
            probabilities.append(probability)
            if "highlight" in row["candidate_reason"]:
                highlight_probabilities.append(probability)

    assert probabilities
    assert highlight_probabilities
    assert max(probabilities) < 0.55
    assert max(highlight_probabilities) < 0.40


@requires_corpus
@pytest.mark.skipif(detector.CatBoostClassifier is None or detector.Pool is None, reason="CatBoost is required")
def test_bundled_detector_accepts_known1448_bottom_edge_sher() -> None:
    dataset = CORPUS_ROOT / "detector-training/cat-projector-known1448-no-cat-20260518"
    image = Image.open(dataset / "frames/known1448_frame_011_chunk_0156_1020ms.jpg").convert("RGB")
    model, metadata = detector.load_model(detector.BUNDLED_MODEL_PATH, detector.BUNDLED_METADATA_PATH)
    predictions = sorted(
        detector.score_candidates(
            image,
            model=model,
            metadata=metadata,
            candidates=detector._bottom_edge_window_candidates(image.width, image.height),  # noqa: SLF001
        ),
        key=lambda item: item.cat_probability,
        reverse=True,
    )

    assert predictions
    bottom_edge_predictions = [
        prediction
        for prediction in predictions
        if prediction.candidate is not None and prediction.candidate.source == "bottom_edge_window_proposal"
    ]

    assert bottom_edge_predictions
    assert bottom_edge_predictions[0].cat_probability >= 0.90


@requires_corpus
@pytest.mark.skipif(detector.CatBoostClassifier is None or detector.Pool is None, reason="CatBoost is required")
def test_bundled_detector_rejects_bottom_edge_true_no_cat_windows() -> None:
    dataset = CORPUS_ROOT / "detector-training/cat-projector-bottom-window-true-no-cat-20260518"
    model, metadata = detector.load_model(detector.BUNDLED_MODEL_PATH, detector.BUNDLED_METADATA_PATH)
    probabilities: list[float] = []

    with (dataset / "labels.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image = Image.open(dataset / row["image_relpath"]).convert("RGB")
            bbox = detector._parse_bbox_xywh(row["candidate_bbox_xywh"])  # noqa: SLF001
            assert bbox is not None
            candidate = detector.Candidate(bbox, bbox[0], bbox[1], int(bbox[2] * bbox[3]), "regression")
            predictions = detector.score_candidates(image, model=model, metadata=metadata, candidates=[candidate])
            assert predictions
            probabilities.append(predictions[0].cat_probability)

    assert probabilities
    assert max(probabilities) < 0.55


def test_bootstrap_candidates_writes_candidate_labels(tmp_path: Path) -> None:
    package = tmp_path / "source"
    frames = package / "frames"
    frames.mkdir(parents=True)
    image = Image.new("RGB", (320, 180), "white")
    ImageDraw.Draw(image).rectangle((60, 40, 105, 150), fill="black")
    image.save(frames / "positive.jpg")
    labels = package / "labels.csv"
    labels.write_text(
        "image_relpath,label_cat_present,label_cat_playing,bbox_xywh,occlusion,confidence,notes,source_recording_dir,source_chunk,source_offset_seconds,ha_session_id,frigate_event_id,video_slug,candidate_reason\n"
        "frames/positive.jpg,yes,watching,,,,,,,,,,,\n",
        encoding="utf-8",
    )

    args = detector.parse_args(
        [
            "bootstrap-candidates",
            "--labels",
            str(labels),
            "--output-root",
            str(tmp_path / "datasets"),
            "--package-id",
            "boot",
        ]
    )
    detector.bootstrap_candidates(args)

    with (tmp_path / "datasets" / "boot" / "labels.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["label_candidate_is_cat"] == "unsure"
    assert rows[0]["candidate_bbox_xywh"]


def test_bootstrap_live_jump_events_writes_review_candidates_not_positive_labels(tmp_path: Path) -> None:
    source = tmp_path / "live"
    frames = source / "frames"
    frames.mkdir(parents=True)
    image_path = frames / "jump.jpg"
    Image.new("RGB", (320, 180), "white").save(image_path)
    events = source / "events.jsonl"
    events.write_text(
        json.dumps({"image_path": str(image_path), "component_bbox_xyxy": [60, 40, 120, 150]}) + "\n",
        encoding="utf-8",
    )

    args = detector.parse_args(
        [
            "bootstrap-live-jump-events",
            "--events",
            str(events),
            "--output-root",
            str(tmp_path / "datasets"),
            "--package-id",
            "live",
        ]
    )
    detector.bootstrap_live_jump_events(args)

    with (tmp_path / "datasets" / "live" / "labels.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["label_cat_present"] == "unsure"
    assert rows[0]["label_candidate_is_cat"] == "unsure"
    assert rows[0]["bbox_xywh"] == ""


def test_render_frame_cat_report_uses_frame_level_answer(tmp_path: Path) -> None:
    package = tmp_path / "dataset"
    frames = package / "frames"
    frames.mkdir(parents=True)
    Image.new("RGB", (320, 180), "white").save(frames / "cat.jpg")
    Image.new("RGB", (320, 180), "white").save(frames / "missing_bbox.jpg")
    Image.new("RGB", (320, 180), "white").save(frames / "candidate_only.jpg")
    Image.new("RGB", (320, 180), "white").save(frames / "empty.jpg")
    labels = package / "labels.csv"
    fieldnames = [
        "image_relpath",
        "label_cat_present",
        "label_cat_playing",
        "review_status",
        "candidate_bbox_xywh",
        "label_candidate_is_cat",
        "negative_reason",
        "bbox_xywh",
        "occlusion",
        "confidence",
        "notes",
        "source_recording_dir",
        "source_chunk",
        "source_offset_seconds",
        "ha_session_id",
        "frigate_event_id",
        "video_slug",
        "candidate_reason",
    ]
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "image_relpath": "frames/cat.jpg",
                    "label_cat_present": "yes",
                    "candidate_bbox_xywh": "10,10,20,20",
                    "label_candidate_is_cat": "no",
                    "negative_reason": "missed_candidate",
                    "bbox_xywh": "40,50,60,70",
                    "review_status": "human_reviewed",
                },
                {
                    "image_relpath": "frames/missing_bbox.jpg",
                    "label_cat_present": "yes",
                    "candidate_bbox_xywh": "10,10,20,20",
                    "label_candidate_is_cat": "no",
                    "negative_reason": "missed_candidate",
                    "bbox_xywh": "",
                    "review_status": "human_reviewed",
                },
                {
                    "image_relpath": "frames/candidate_only.jpg",
                    "label_cat_present": "yes",
                    "candidate_bbox_xywh": "10,10,20,20",
                    "label_candidate_is_cat": "yes",
                    "negative_reason": "",
                    "bbox_xywh": "",
                    "review_status": "human_reviewed",
                },
                {
                    "image_relpath": "frames/empty.jpg",
                    "label_cat_present": "no",
                    "candidate_bbox_xywh": "10,10,20,20",
                    "label_candidate_is_cat": "no",
                    "negative_reason": "empty_wall",
                    "bbox_xywh": "",
                    "review_status": "human_reviewed",
                },
            ]
        )

    out = tmp_path / "report.jpg"
    args = detector.parse_args(["render-frame-cat-report", "--labels", str(labels), "--out", str(out)])
    detector.render_frame_cat_report(args)

    summary = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["cat"] == 1
    assert summary["missing_cat_bbox"] == 2
    assert summary["no_cat"] == 1
    assert out.exists()


def test_training_samples_keep_candidate_negative_and_human_positive(tmp_path: Path) -> None:
    package = tmp_path / "dataset"
    frames = package / "frames"
    frames.mkdir(parents=True)
    Image.new("RGB", (320, 180), "white").save(frames / "cat.jpg")
    labels = package / "labels.csv"
    fieldnames = [
        "image_relpath",
        "label_cat_present",
        "label_cat_playing",
        "review_status",
        "candidate_bbox_xywh",
        "label_candidate_is_cat",
        "negative_reason",
        "bbox_xywh",
        "occlusion",
        "confidence",
        "notes",
        "source_recording_dir",
        "source_chunk",
        "source_offset_seconds",
        "ha_session_id",
        "frigate_event_id",
        "video_slug",
        "candidate_reason",
    ]
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "image_relpath": "frames/cat.jpg",
                "label_cat_present": "yes",
                "candidate_bbox_xywh": "10,10,20,20",
                "label_candidate_is_cat": "no",
                "negative_reason": "missed_candidate",
                "bbox_xywh": "80,50,40,90",
                "review_status": "human_reviewed",
            }
        )

    samples = detector._training_samples_from_labels([labels])

    assert [(sample.label, sample.sample_role, detector._format_bbox_xywh(sample.bbox_xywh)) for sample in samples] == [
        (0, "candidate_bbox", "10,10,20,20"),
        (1, "human_bbox", "80,50,40,90"),
    ]


def test_mine_post_positive_hard_negatives_masks_cat_then_records_false_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "source"
    frames = package / "frames"
    frames.mkdir(parents=True)
    image = Image.new("RGB", (320, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 100, 150), fill="black")  # confirmed cat
    draw.rectangle((190, 75, 235, 155), fill="black")  # stool/table false candidate
    image.save(frames / "cat_and_stool.jpg")
    labels = package / "labels.csv"
    fieldnames = [
        "image_relpath",
        "label_cat_present",
        "label_cat_playing",
        "review_status",
        "candidate_bbox_xywh",
        "label_candidate_is_cat",
        "negative_reason",
        "bbox_xywh",
        "occlusion",
        "confidence",
        "notes",
        "source_recording_dir",
        "source_chunk",
        "source_offset_seconds",
        "ha_session_id",
        "frigate_event_id",
        "video_slug",
        "candidate_reason",
    ]
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "image_relpath": "frames/cat_and_stool.jpg",
                "label_cat_present": "yes",
                "label_cat_playing": "yes",
                "bbox_xywh": "40,40,60,110",
                "review_status": "human_reviewed",
            }
        )

    monkeypatch.setattr(detector, "load_model", lambda *_args: ("model", {"created_at": "test-model"}))

    def fake_score_candidates(_image, *, model, metadata):
        assert model == "model"
        assert metadata["created_at"] == "test-model"
        return [
            detector.CandidatePrediction(
                candidate=detector.Candidate((190, 75, 45, 80), 190, 75, 3600, "fake_stool"),
                cat_probability=0.91,
                model_path="model",
                model_version="test-model",
                features={},
            ),
            detector.CandidatePrediction(
                candidate=detector.Candidate((45, 45, 10, 10), 45, 45, 100, "cat_tiny_overlap"),
                cat_probability=0.99,
                model_path="model",
                model_version="test-model",
                features={},
            ),
        ]

    monkeypatch.setattr(detector, "score_candidates", fake_score_candidates)
    args = detector.parse_args(
        [
            "mine-post-positive-hard-negatives",
            "--labels",
            str(labels),
            "--output-root",
            str(tmp_path / "datasets"),
            "--package-id",
            "post-positive",
            "--replace-existing",
        ]
    )
    detector.mine_post_positive_hard_negatives(args)

    with (tmp_path / "datasets" / "post-positive" / "labels.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["label_cat_present"] == "yes"
    assert rows[0]["candidate_bbox_xywh"] == "190,75,45,80"
    assert rows[0]["label_candidate_is_cat"] == "no"
    assert rows[0]["negative_reason"] == "post_positive_erased_cat_false_positive"
    assert rows[0]["bbox_xywh"] == "40,40,60,110"
    assert rows[0]["candidate_reason"] == "post_positive_erased_cat_false_positive"
