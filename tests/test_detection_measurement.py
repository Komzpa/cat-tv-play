from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


detection = _load_module("cat_tv_play_detection_test", ROOT / "custom_components" / "cat_tv_play" / "detection.py")
measurement = _load_module(
    "cat_tv_play_measurement_test", ROOT / "custom_components" / "cat_tv_play" / "measurement.py"
)
active_learning = _load_module(
    "cat_projector_active_learning_test", ROOT / "scripts" / "cat_projector_active_learning.py"
)
frame_detector = _load_module(
    "cat_projector_frame_detector_test", ROOT / "scripts" / "cat_projector_frame_detector.py"
)
export_yolo = _load_module(
    "cat_projector_yolo_export_test", ROOT / "scripts" / "export_cat_projector_yolo_segmentation.py"
)
yolo_seg = _load_module("cat_projector_yolo_segmentation_test", ROOT / "scripts" / "cat_projector_yolo_segmentation.py")


def test_fake_segmentation_detector_returns_mask_detection() -> None:
    image = Image.new("RGB", (120, 90), (200, 200, 200))
    detector = detection.build_detector(
        detection.DetectorConfig(backend="fake", confidence_threshold=0.82, allow_fake=True)
    )

    rows = detector.detect(image, detection.DetectorContext(frame_index=7, timestamp_seconds=1.2))

    assert len(rows) == 1
    assert rows[0].has_mask
    assert rows[0].score == 0.82
    assert rows[0].frame_index == 7
    assert rows[0].source == "fake_segmentation"


def test_detection_debug_row_serialises_mask_polygon() -> None:
    mask = np.zeros((80, 120), dtype=bool)
    mask[30:70, 40:75] = True
    row = detection.CatDetection(
        bbox_xywh=(40.0, 30.0, 35.0, 40.0),
        score=0.9,
        source="test",
        model_id="test-model",
        mask=mask,
    ).to_debug_row()

    assert row["has_mask"] is True
    assert row["mask_area_px"] == int(mask.sum())
    assert len(row["mask_polygon"]) >= 4
    xs = [point["x"] for point in row["mask_polygon"]]
    ys = [point["y"] for point in row["mask_polygon"]]
    assert min(xs) <= 40
    assert max(xs) >= 74
    assert min(ys) <= 30
    assert max(ys) >= 69


def test_yolo_eval_reads_all_segmentation_polygons(tmp_path: Path) -> None:
    label = tmp_path / "multi.txt"
    label.write_text(
        "\n".join(
            [
                "0 0.100000 0.100000 0.200000 0.100000 0.200000 0.200000 0.100000 0.200000",
                "0 0.700000 0.700000 0.800000 0.700000 0.800000 0.800000 0.700000 0.800000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    polygons = yolo_seg._read_yolo_polygon_labels(label, 100, 100)  # noqa: SLF001
    mask = yolo_seg._polygons_mask(polygons, (100, 100))  # noqa: SLF001

    assert len(polygons) == 2
    assert mask[15, 15]
    assert mask[75, 75]


def test_segmentation_config_fails_cleanly_when_weights_are_absent(tmp_path: Path) -> None:
    try:
        detection.build_detector(
            detection.DetectorConfig(
                backend="segmentation",
                model_path=str(tmp_path / "missing.pt"),
            )
        )
    except detection.DetectorUnavailableError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing segmentation weights unexpectedly loaded")


def test_yolo_masks_are_projected_to_source_frame_coordinates() -> None:
    raw_mask = np.zeros((8, 8), dtype=bool)
    raw_mask[1:4, 1:4] = True
    polygon = ((20.0, 10.0), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0))

    mask = detection._mask_in_frame_coordinates(raw_mask, polygon, width=100, height=80)  # noqa: SLF001

    assert mask.shape == (80, 100)
    assert mask[15, 25]
    assert not mask[2, 2]


def test_auto_config_without_model_uses_legacy_fallback(monkeypatch) -> None:
    class DummyLegacyDetector:
        source = "legacy_contrast_catboost"
        model_id = "dummy"

        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(detection, "LegacyContrastDetector", DummyLegacyDetector)

    detector = detection.build_detector(detection.DetectorConfig(backend="auto"))

    assert detector.source == "legacy_contrast_catboost"


def test_fake_detector_requires_explicit_dev_permission() -> None:
    try:
        detection.build_detector(detection.DetectorConfig(backend="fake"))
    except detection.DetectorUnavailableError as exc:
        assert "allow_fake" in str(exc)
    else:
        raise AssertionError("fake detector unexpectedly loaded without dev permission")


def test_active_learning_auto_without_segmentation_model_selects_legacy() -> None:
    config = active_learning._normalised_detector_config(  # noqa: SLF001
        SimpleNamespace(
            detector_backend="auto",
            segmentation_model=None,
            device=None,
            confidence_threshold=0.5,
            allow_fake_detector=False,
        )
    )

    assert config["backend"] == "legacy"
    assert config["model_path"] is None


def test_active_learning_auto_with_segmentation_model_selects_segmentation(tmp_path: Path) -> None:
    config = active_learning._normalised_detector_config(  # noqa: SLF001
        SimpleNamespace(
            detector_backend="auto",
            segmentation_model=tmp_path / "sher.pt",
            device="cpu",
            confidence_threshold=0.5,
            allow_fake_detector=False,
        )
    )

    assert config["backend"] == "segmentation"
    assert config["requested_backend"] == "auto"


def test_legacy_source_subtracted_candidate_mask_becomes_detection_mask() -> None:
    mask = np.zeros((80, 120), dtype=bool)
    mask[30:70, 40:75] = True
    candidate = frame_detector.Candidate(
        bbox_xywh=(40.0, 30.0, 35.0, 40.0),
        top_x_px=57.0,
        top_y_px=30.0,
        area_px=int(mask.sum()),
        source="source_subtracted_projector_residual",
        mask=mask,
    )

    class FakeLegacy:
        def detect_source_subtracted_candidate_components(self, *_args, **_kwargs):
            return [candidate]

        def score_candidates(self, *_args, **_kwargs):
            return [
                SimpleNamespace(
                    candidate=candidate,
                    cat_probability=0.91,
                    model_path="/tmp/fake.cbm",
                    features={"area_ratio": 0.01},
                )
            ]

    detector = object.__new__(detection.LegacyContrastDetector)
    detector._legacy = FakeLegacy()  # noqa: SLF001
    detector._model = object()  # noqa: SLF001
    detector._metadata = {"model_path": "/tmp/fake.cbm"}  # noqa: SLF001
    detector.model_id = "/tmp/fake.cbm"
    detector.min_probability = 0.0

    rows = detector.detect(
        Image.new("RGB", (120, 80), (200, 200, 200)),
        detection.DetectorContext(debug={"projector_source_frame": Image.new("RGB", (120, 80), "white")}),
    )

    assert len(rows) == 1
    assert rows[0].has_mask
    assert rows[0].mask is mask
    point, warning = active_learning._measurement_for_detection(rows[0])  # noqa: SLF001
    assert warning == ""
    assert point.point_type == "mask_top_p5"
    assert point.source == "segmentation_mask"


def test_mask_top_measurement_ignores_single_noise_pixel() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:80, 30:70] = True
    mask[2, 99] = True

    point = measurement.mask_top_measurement_point(mask, score=0.9, top_fraction=0.05)

    assert point is not None
    assert point.point_type == "mask_top_p5"
    assert 39 <= point.image_y <= 43
    assert 45 <= point.image_x <= 55
    assert point.confidence == 0.9


def test_mask_top_measurement_uses_largest_component_not_small_high_island() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:80, 30:70] = True
    mask[2:12, 80:90] = True

    point = measurement.mask_top_measurement_point(mask, score=0.9, top_fraction=0.05)

    assert point is not None
    assert 39 <= point.image_y <= 43
    assert point.debug["component_count"] == 2
    assert point.debug["discarded_component_area_px"] == 100


def test_mask_top_measurement_rejects_tiny_or_degenerate_masks() -> None:
    tiny = np.zeros((100, 100), dtype=bool)
    tiny[10:12, 10:12] = True
    strip = np.zeros((100, 100), dtype=bool)
    strip[10:90, 20:22] = True

    assert measurement.mask_top_measurement_point(tiny, score=0.9) is None
    assert measurement.mask_top_measurement_point(strip, score=0.9) is None


def test_mask_measurement_transforms_to_wall_centimeters() -> None:
    point = measurement.MeasurementPoint(
        point_type="mask_top_p5",
        image_x=25,
        image_y=20,
        confidence=0.8,
        source="segmentation_mask",
    )
    # wall_x = image_x, wall_y = 100 - image_y
    homography = (1.0, 0.0, 0.0, 0.0, -1.0, 100.0, 0.0, 0.0)

    wall_point = measurement.with_wall_coordinates(point, homography)

    assert wall_point.wall_x_cm == 25
    assert wall_point.wall_y_cm == 80


def test_fake_segmentation_smoke_measures_from_mask_not_bbox() -> None:
    image_path = "/tmp/fake-mask-frame.jpg"
    active_learning._init_worker(
        "", "", json.dumps({"backend": "fake", "confidence_threshold": 0.9, "allow_fake": True})
    )  # noqa: SLF001

    Image.new("RGB", (120, 100), (200, 200, 200)).save(image_path)
    row = active_learning._score_one(image_path)  # noqa: SLF001

    assert row["detector_backend"] == "fake_segmentation"
    assert row["best_has_mask"] is True
    assert row["best_measurement_point"]["point_type"] == "mask_top_p5"
    assert row["best_measurement_point"]["source"] == "segmentation_mask"
    Path(image_path).unlink()


def test_segmentation_active_learning_does_not_train_legacy_catboost(tmp_path: Path, monkeypatch) -> None:
    def fail_legacy_train(*_args, **_kwargs):
        raise AssertionError("segmentation rescore must not train CatBoost")

    monkeypatch.setattr(active_learning, "_train_model", fail_legacy_train)
    monkeypatch.setattr(
        active_learning,
        "_materialize_review_training_package",
        lambda: {"labels_csv": str(tmp_path / "labels.csv"), "copied": []},
    )
    monkeypatch.setattr(active_learning, "_collect_labels", lambda _labels: [])
    monkeypatch.setattr(active_learning, "_reviewable_frame_paths", lambda: [tmp_path / "frame.jpg"])
    monkeypatch.setattr(
        active_learning,
        "_rescore_frames",
        lambda *_args, **_kwargs: [
            {
                "raw_path": str(tmp_path / "frame.jpg"),
                "global_frame": 0,
                "best_probability": 0.9,
                "best_bbox": "",
                "detector_backend": "fake_segmentation",
            }
        ],
    )
    monkeypatch.setattr(
        active_learning,
        "_remeasure_jump_heights",
        lambda *_args, **_kwargs: {"kind": "jump_heights", "status": "done", "videos": []},
    )
    monkeypatch.setattr(active_learning, "RESCORES_ROOT", tmp_path / "rescores")

    manifest = active_learning.run_iteration(
        SimpleNamespace(
            run_id="seg-no-catboost",
            output_root=tmp_path / "rescores",
            replace_existing=False,
            labels=[],
            detector_backend="fake",
            segmentation_model=None,
            allow_fake_detector=True,
            device=None,
            confidence_threshold=0.5,
            model=tmp_path / "legacy.cbm",
            metadata=tmp_path / "legacy.json",
            jobs=1,
            calibration=tmp_path / "calibration.json",
            height_min_probability=0.5,
        )
    )

    assert manifest["model"]["training_status"] == "skipped_legacy_catboost_training"
    assert manifest["model"]["backend"] == "fake"


def test_yolo_export_splits_by_recording_and_writes_polygon_labels(tmp_path: Path) -> None:
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50), (10, 10, 10)).save(image)
    (labels_root / "case.json").write_text(
        json.dumps(
            {
                "label": "cat",
                "image_path": str(image),
                "source_recording_dir": "/recordings/session-a",
                "review_decision": "bad_geometry",
                "geometry_status": "corrected",
                "masks": [
                    {
                        "polygon": [
                            {"x": 10, "y": 10},
                            {"x": 30, "y": 10},
                            {"x": 20, "y": 30},
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    items = export_yolo.collect_items(labels_root)
    manifest = export_yolo.export_yolo_segmentation(items, tmp_path / "yolo", val_fraction=0.0, copy=True)

    assert manifest["item_count"] == 1
    label_path = Path(manifest["rows"][0]["label"])
    assert label_path.read_text(encoding="utf-8").startswith("0 0.100000 0.200000")
    assert manifest["split_policy"] == "recording_session_hash"
    assert manifest["rows"][0]["split_key_source"] == "source_recording_dir"
    assert manifest["rows"][0]["image_sha256"]


def test_yolo_export_loads_sidecar_masks_and_reports_skips(tmp_path: Path) -> None:
    labels_root = tmp_path / "labels"
    masks_root = tmp_path / "masks" / "case.sidecar"
    labels_root.mkdir()
    masks_root.mkdir(parents=True)
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50), (10, 10, 10)).save(image)
    mask_path = masks_root / "cat.mask1.json"
    mask_path.write_text(
        json.dumps(
            {
                "mask_id": "cat.mask1",
                "polygon": [
                    {"x": 10, "y": 10},
                    {"x": 30, "y": 10},
                    {"x": 20, "y": 30},
                ],
            }
        ),
        encoding="utf-8",
    )
    (labels_root / "case.sidecar.json").write_text(
        json.dumps(
            {
                "case_id": "case.sidecar",
                "label": "cat",
                "image_path": str(image),
                "video_id": "video.a",
                "mask_refs": [{"id": "cat.mask1", "path": str(mask_path)}],
            }
        ),
        encoding="utf-8",
    )
    (labels_root / "case.cat-no-mask.json").write_text(
        json.dumps({"case_id": "case.cat-no-mask", "label": "cat", "image_path": str(image)}),
        encoding="utf-8",
    )

    items, skipped = export_yolo.collect_items_with_skips(labels_root)
    validation = export_yolo.validate_items(items, skipped)
    manifest = export_yolo.export_yolo_segmentation(items, tmp_path / "yolo-sidecar", val_fraction=0.0, copy=True)

    assert len(items) == 1
    assert items[0].mask_ids == ("cat.mask1",)
    assert validation["issue_counts"]["cat_without_valid_polygon"] == 1
    assert manifest["rows"][0]["mask_ref_paths"] == [str(mask_path)]
    assert manifest["source_mask_ref_count"] == 1


def test_yolo_export_loads_relative_sidecar_mask_refs_from_sibling_masks_root(tmp_path: Path) -> None:
    labels_root = tmp_path / "labels"
    masks_root = tmp_path / "masks"
    labels_root.mkdir()
    mask_dir = masks_root / "case.relative"
    mask_dir.mkdir(parents=True)
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50), (10, 10, 10)).save(image)
    (mask_dir / "cat.mask1.json").write_text(
        json.dumps(
            {
                "mask_id": "cat.mask1",
                "polygon": [
                    {"x": 10, "y": 10},
                    {"x": 30, "y": 10},
                    {"x": 20, "y": 30},
                ],
            }
        ),
        encoding="utf-8",
    )
    (labels_root / "case.relative.json").write_text(
        json.dumps(
            {
                "case_id": "case.relative",
                "label": "cat",
                "image_path": str(image),
                "mask_refs": [{"id": "cat.mask1", "path": "cat.mask1.json"}],
            }
        ),
        encoding="utf-8",
    )
    items, skipped = export_yolo.collect_items_with_skips(labels_root)

    assert not skipped
    assert len(items) == 1
    assert items[0].mask_ids == ("cat.mask1",)


def test_yolo_export_records_discarded_masks_on_not_cat(tmp_path: Path) -> None:
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50), (10, 10, 10)).save(image)
    (labels_root / "case.notcat.json").write_text(
        json.dumps(
            {
                "label": "not_cat",
                "image_path": str(image),
                "masks": [
                    {
                        "id": "old",
                        "polygon": [
                            {"x": 10, "y": 10},
                            {"x": 30, "y": 10},
                            {"x": 20, "y": 30},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    items = export_yolo.collect_items(labels_root)
    manifest = export_yolo.export_yolo_segmentation(items, tmp_path / "yolo-notcat", val_fraction=0.0, copy=True)

    assert manifest["hard_negative_count"] == 1
    assert manifest["discarded_mask_count"] == 1
    assert manifest["rows"][0]["had_discarded_mask"] is True


def test_yolo_train_refuses_missing_base_model_before_ultralytics_import(tmp_path: Path) -> None:
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n", encoding="utf-8")
    args = yolo_seg.parse_args(
        [
            "train",
            "--dataset",
            str(dataset_yaml),
            "--base-model",
            str(tmp_path / "missing.pt"),
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )

    try:
        yolo_seg.train(args)
    except FileNotFoundError as exc:
        assert "base model" in str(exc)
    else:
        raise AssertionError("training unexpectedly accepted a missing base model")


def test_yolo_train_refuses_generic_base_without_new_lineage_flag(tmp_path: Path) -> None:
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n", encoding="utf-8")
    base_model = tmp_path / "base" / "yolo11n-seg.pt"
    base_model.parent.mkdir()
    base_model.write_bytes(b"not a real model")
    args = yolo_seg.parse_args(
        [
            "train",
            "--dataset",
            str(dataset_yaml),
            "--base-model",
            str(base_model),
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )

    try:
        yolo_seg.train(args)
    except ValueError as exc:
        assert "--allow-new-sher-lineage" in str(exc)
        assert "latest Sher .pt" in str(exc)
    else:
        raise AssertionError("training unexpectedly accepted a generic YOLO base")


def test_yolo_train_classifies_existing_sher_model_as_fine_tune(tmp_path: Path) -> None:
    model = tmp_path / "models" / "sher-yolo-seg-first.pt"
    model.parent.mkdir()
    model.write_bytes(b"not a real model")

    assert (
        yolo_seg._training_mode_for_base_model(model, allow_new_sher_lineage=False)  # noqa: SLF001
        == "fine_tune_existing_sher"
    )


def test_yolo_train_allows_explicit_generic_new_lineage(tmp_path: Path) -> None:
    model = tmp_path / "models" / "base" / "yolo11n-seg.pt"

    assert (
        yolo_seg._training_mode_for_base_model(model, allow_new_sher_lineage=True)  # noqa: SLF001
        == "new_sher_lineage_from_generic_yolo"
    )


def test_yolo_train_parses_gentle_fine_tune_knobs(tmp_path: Path) -> None:
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n", encoding="utf-8")
    args = yolo_seg.parse_args(
        [
            "train",
            "--dataset",
            str(dataset_yaml),
            "--base-model",
            str(tmp_path / "sher-yolo-seg-first.pt"),
            "--optimizer",
            "AdamW",
            "--lr0",
            "0.0002",
            "--lrf",
            "0.1",
            "--warmup-epochs",
            "0",
            "--freeze",
            "10",
        ]
    )

    assert args.optimizer == "AdamW"
    assert args.lr0 == 0.0002
    assert args.lrf == 0.1
    assert args.warmup_epochs == 0
    assert args.freeze == 10


def test_yolo_eval_defaults_to_safe_sher_confidence_threshold(tmp_path: Path) -> None:
    args = yolo_seg.parse_args(
        [
            "eval",
            "--dataset",
            str(tmp_path / "dataset.yaml"),
            "--model",
            str(tmp_path / "model.pt"),
            "--out",
            str(tmp_path / "eval"),
        ]
    )

    assert args.confidence_threshold == 0.55


def test_yolo_report_renderer_consumes_fake_eval(tmp_path: Path) -> None:
    manifest = {"dataset_hash": "abc", "item_count": 1}
    eval_data = {
        "metrics": {"cat_presence_precision": 0.5, "legacy_false_positive_hard_negative_count": 3},
        "rows": [
            {
                "buckets": ["false_positive", "segmentation_legacy_disagreement"],
                "image": "/tmp/frame.jpg",
                "score": 0.7,
                "legacy_score": 0.1,
                "legacy_present": False,
                "mask_iou": 0.0,
                "overlay_paths": ["overlays/false_positive/00000.jpg"],
            }
        ],
    }
    out = tmp_path / "report.html"

    yolo_seg.render_report_from_files(manifest, eval_data, out, limit=10)

    text = out.read_text(encoding="utf-8")
    assert "false_positive" in text
    assert "segmentation_legacy_disagreement" in text
    assert "cat_presence_precision" in text
    assert "<img src='overlays/false_positive/00000.jpg'>" in text
    assert "legacy_false_positive_hard_negative_count" in text
