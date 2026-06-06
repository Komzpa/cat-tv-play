from __future__ import annotations

import csv
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_label_review_server.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("cat_projector_label_review_server", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
server = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = server
SCRIPT_SPEC.loader.exec_module(server)

ACTIVE_LEARNING_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_active_learning.py"
ACTIVE_LEARNING_SPEC = importlib.util.spec_from_file_location("cat_projector_active_learning", ACTIVE_LEARNING_PATH)
assert ACTIVE_LEARNING_SPEC is not None
active_learning = importlib.util.module_from_spec(ACTIVE_LEARNING_SPEC)
assert ACTIVE_LEARNING_SPEC.loader is not None
sys.modules[ACTIVE_LEARNING_SPEC.name] = active_learning
ACTIVE_LEARNING_SPEC.loader.exec_module(active_learning)


def test_fake_smoke_builds_review_queue_and_action_records(tmp_path: Path) -> None:
    assert server.run_fake_smoke(tmp_path) == 0
    label_files = sorted((tmp_path / "state" / "label-review" / "labels").glob("*.json"))
    action_files = sorted((tmp_path / "state" / "label-review" / "actions").glob("*.json"))
    mask_files = sorted((tmp_path / "state" / "label-review" / "masks").glob("*/*.json"))
    video_status_files = sorted((tmp_path / "state" / "label-review" / "videos").glob("*.json"))
    training_labels = sorted((tmp_path / "state" / "datasets").glob("cat-projector-review-ui-*/labels.csv"))
    assert len(label_files) == 2
    assert len(action_files) == 3
    assert len(mask_files) == 1
    assert len(video_status_files) == 1
    assert len(training_labels) == 1
    saved_labels = [json.loads(path.read_text(encoding="utf-8")) for path in label_files]
    assert {row["label_cat_present"] for row in saved_labels} == {"yes", "no"}
    assert {row["label_candidate_is_cat"] for row in saved_labels} == {"no"}
    assert {row["video_id"] for row in saved_labels} != {None}
    training_text = training_labels[0].read_text(encoding="utf-8")
    assert "label_candidate_is_cat" in training_text
    assert "yes" in training_text
    assert "no" in training_text
    decisions = {row["review_decision"] for row in saved_labels}
    assert decisions == {"false_positive", "missed_cat"}
    assert any(row["cat_present"] is True and row["geometry_status"] == "corrected" for row in saved_labels)


def test_missed_cat_label_is_not_treated_as_empty_frame() -> None:
    assert not server._frame_says_no_cat(  # noqa: SLF001
        {
            "review_status": "saved",
            "review_decision": "missed_cat",
            "label": "cat",
            "human_label": "cat",
            "cat_present": True,
            "candidate_is_cat": False,
            "label_cat_present": "yes",
            "label_candidate_is_cat": "no",
        }
    )


def test_live_retrain_job_materializes_training_package(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(image_path)
    original_allowed = server.ALLOWED_ROOTS
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_jobs = server.JOBS_ROOT
    original_training = server.TRAINING_DATASETS_ROOT
    original_allow_live = server.ALLOW_LIVE_JOBS
    original_active = server._ACTIVE_JOB_ID
    original_live_command = server.LIVE_JOB_COMMAND
    try:
        server.ALLOWED_ROOTS = (tmp_path,)
        server.REVIEW_ROOT = tmp_path / "review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.JOBS_ROOT = server.REVIEW_ROOT / "jobs"
        server.TRAINING_DATASETS_ROOT = tmp_path / "datasets"
        server.ALLOW_LIVE_JOBS = True
        server._ACTIVE_JOB_ID = None
        server.LIVE_JOB_COMMAND = [
            sys.executable,
            "-c",
            "import json; print(json.dumps({'frame_count': 3}))",
        ]
        server._save_label(
            {
                "case_id": "case.live_retrain",
                "image_path": str(image_path),
                "review_decision": "good",
                "label": "cat",
                "review_status": "saved",
            }
        )

        job = server._start_or_queue_job({"action": "retrain_model", "case_id": "case.live_retrain"})

        assert job["kind"] == "cat_projector_label_review_job_v1"
        assert job["status"] == "running"
        assert job["payload"]["training_package"]["copied"]
        assert Path(job["payload"]["training_package"]["labels_csv"]).exists()
        for _ in range(20):
            if server._ACTIVE_JOB_ID is None:
                break
            time.sleep(0.01)
        finished = json.loads((server.JOBS_ROOT / f"{job['id']}.json").read_text(encoding="utf-8"))
        assert finished["status"] == "done"
        assert finished["result"]["frame_count"] == 3
    finally:
        server.ALLOWED_ROOTS = original_allowed
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.JOBS_ROOT = original_jobs
        server.TRAINING_DATASETS_ROOT = original_training
        server.ALLOW_LIVE_JOBS = original_allow_live
        server._ACTIVE_JOB_ID = original_active
        server.LIVE_JOB_COMMAND = original_live_command


def test_stale_running_job_is_marked_failed_after_restart(tmp_path: Path) -> None:
    original_jobs = server.JOBS_ROOT
    original_active = server._ACTIVE_JOB_ID
    try:
        server.JOBS_ROOT = tmp_path / "jobs"
        server.JOBS_ROOT.mkdir(parents=True)
        server._ACTIVE_JOB_ID = None
        job_path = server.JOBS_ROOT / "stale.json"
        job_path.write_text(
            json.dumps(
                {
                    "id": "stale",
                    "kind": "cat_projector_label_review_job_v1",
                    "action": "rescore_recording",
                    "status": "running",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "log": ["started by review UI"],
                }
            ),
            encoding="utf-8",
        )

        jobs = server._list_jobs()
    finally:
        server.JOBS_ROOT = original_jobs
        server._ACTIVE_JOB_ID = original_active

    assert jobs[0]["status"] == "failed"
    assert "marked failed after server restart" in jobs[0]["log"][-1]


def test_segment_endpoint_requires_configured_sam_without_explicit_degraded_fallback(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"not-an-image")
    old_endpoint = server.SAM_ENDPOINT
    try:
        server.SAM_ENDPOINT = ""
        try:
            server._segment_with_optional_sam(
                image_path,
                [{"x": 1, "y": 1}],
                [],
                [],
            )
        except ValueError as exc:
            assert "CAT_PROJECTOR_SAM_ENDPOINT is not configured" in str(exc)
        else:
            raise AssertionError("segment unexpectedly fell back without allow_fallback")
    finally:
        server.SAM_ENDPOINT = old_endpoint


def test_segment_endpoint_uses_click_contour_when_fallback_is_explicit(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image = server.Image.new("RGB", (100, 80), (20, 20, 20))
    for x in range(35, 65):
        for y in range(25, 55):
            image.putpixel((x, y), (180, 180, 180))
    image.save(image_path)
    old_endpoint = server.SAM_ENDPOINT
    try:
        server.SAM_ENDPOINT = ""
        payload = server._segment_with_optional_sam(
            image_path,
            [{"x": 50, "y": 40}],
            [],
            [],
            allow_fallback=True,
        )
        assert payload["source"] == "server_click_contour"
        assert len(payload["polygon"]) >= 3
        assert payload["bbox_xywh"]["width"] > 1
        assert payload["bbox_xywh"]["height"] > 1
    finally:
        server.SAM_ENDPOINT = old_endpoint


def test_review_decision_mapping_preserves_legacy_label_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(image_path)
    original_allowed = server.ALLOWED_ROOTS
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    try:
        server.ALLOWED_ROOTS = (tmp_path,)
        server.LABELS_ROOT = tmp_path / "labels"
        server.MASKS_ROOT = tmp_path / "masks"
        cases = {
            "good": {
                "label": "cat",
                "label_cat_present": "yes",
                "label_candidate_is_cat": "yes",
                "geometry_status": "ok",
            },
            "false_positive": {"label": "not_cat", "label_cat_present": "no", "label_candidate_is_cat": "no"},
            "missed_cat": {
                "label": "cat",
                "label_cat_present": "yes",
                "label_candidate_is_cat": "no",
                "geometry_status": "missing",
            },
            "bad_geometry": {
                "label": "cat",
                "label_cat_present": "yes",
                "label_candidate_is_cat": "yes",
                "geometry_status": "bad",
            },
            "unsure": {"label": "unsure", "label_cat_present": "", "label_candidate_is_cat": ""},
        }
        for decision, expected in cases.items():
            saved = server._save_label(
                {
                    "case_id": f"case.{decision}",
                    "image_path": str(image_path),
                    "review_decision": decision,
                    "label": "cat",
                    "review_status": "saved",
                }
            )
            for key, value in expected.items():
                assert saved[key] == value
            assert saved["review_decision"] == decision
            assert saved["reviewer_source"] == "local_ui"
    finally:
        server.ALLOWED_ROOTS = original_allowed
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks


def test_parse_bbox_accepts_saved_json_list_and_dict() -> None:
    assert server._parse_bbox([743, 153, 100, 144]) == (743.0, 153.0, 100.0, 144.0)  # noqa: SLF001
    assert server._parse_bbox({"x": 1, "y": 2, "width": 3, "height": 4}) == (1.0, 2.0, 3.0, 4.0)  # noqa: SLF001


def test_bad_geometry_training_package_uses_corrected_mask_bbox(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(image_path)
    original_allowed = server.ALLOWED_ROOTS
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_training = server.TRAINING_DATASETS_ROOT
    try:
        server.ALLOWED_ROOTS = (tmp_path,)
        server.REVIEW_ROOT = tmp_path / "review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.TRAINING_DATASETS_ROOT = tmp_path / "datasets"
        server._save_label(
            {
                "case_id": "case.bad_geometry_mask_bbox",
                "image_path": str(image_path),
                "review_decision": "bad_geometry",
                "label": "cat",
                "review_status": "saved",
                "candidate_bbox_xywh": [1, 2, 3, 4],
                "masks": [
                    {
                        "id": "corrected",
                        "label": "Sher",
                        "kind": "cat",
                        "bbox_xywh": {"x": 10, "y": 11, "width": 12, "height": 13},
                        "polygon": [{"x": 10, "y": 11}, {"x": 22, "y": 24}],
                    }
                ],
            }
        )

        package = server._materialize_review_labels_as_training_package({"action": "retrain_model"})
        labels_path = Path(package["labels_csv"])
        rows = list(csv.DictReader(labels_path.open(encoding="utf-8")))

        corrected_rows = [row for row in rows if row["candidate_bbox_xywh"] == "10,11,12,13"]
        old_candidate_rows = [row for row in rows if row["candidate_bbox_xywh"] == "1,2,3,4"]
        assert corrected_rows
        assert corrected_rows[0]["label_candidate_is_cat"] == "yes"
        assert corrected_rows[0]["bbox_xywh"] == "10,11,12,13"
        assert old_candidate_rows
        assert old_candidate_rows[0]["label_candidate_is_cat"] == "no"
        assert old_candidate_rows[0]["negative_reason"] == "old_candidate_no_overlap_corrected_mask"
        assert old_candidate_rows[0]["bbox_xywh"] == "10,11,12,13"
    finally:
        server.ALLOWED_ROOTS = original_allowed
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.TRAINING_DATASETS_ROOT = original_training


def test_training_package_uses_mask_ref_sidecar_when_embedded_masks_missing(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(image_path)
    original_allowed = server.ALLOWED_ROOTS
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_training = server.TRAINING_DATASETS_ROOT
    try:
        server.ALLOWED_ROOTS = (tmp_path,)
        server.REVIEW_ROOT = tmp_path / "review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.TRAINING_DATASETS_ROOT = tmp_path / "datasets"
        case_id = "case.sidecar_mask_bbox"
        mask_dir = server.MASKS_ROOT / case_id
        mask_dir.mkdir(parents=True)
        (mask_dir / "corrected.json").write_text(
            json.dumps(
                {
                    "mask_id": "corrected",
                    "bbox_xywh": {"x": 10, "y": 11, "width": 12, "height": 13},
                    "polygon": [{"x": 10, "y": 11}, {"x": 22, "y": 11}, {"x": 22, "y": 24}],
                }
            ),
            encoding="utf-8",
        )
        server.LABELS_ROOT.mkdir(parents=True)
        (server.LABELS_ROOT / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "image_path": str(image_path),
                    "label": "cat",
                    "label_cat_present": "yes",
                    "label_candidate_is_cat": "yes",
                    "review_decision": "bad_geometry",
                    "review_status": "saved",
                    "candidate_bbox_xywh": [1, 2, 3, 4],
                    "mask_refs": [{"id": "corrected", "path": "corrected.json"}],
                }
            ),
            encoding="utf-8",
        )

        package = server._materialize_review_labels_as_training_package({"action": "retrain_model"})
        with Path(package["labels_csv"]).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    finally:
        server.ALLOWED_ROOTS = original_allowed
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.TRAINING_DATASETS_ROOT = original_training

    assert rows[0]["candidate_bbox_xywh"] == "10,11,12,13"
    assert rows[0]["bbox_xywh"] == "10,11,12,13"


def test_file_api_supports_get_and_head_for_discovered_images(tmp_path: Path) -> None:
    fake_dataset = server.build_fake_corpus(tmp_path)
    fake_state = tmp_path / "state"
    original_dataset = server.DATASET_ROOT
    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.DATASET_ROOT = fake_dataset
        server.STATE_ROOT = fake_state
        server.REVIEW_ROOT = fake_state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.SCAN_ROOTS = (fake_dataset / "datasets",)
        server.ALLOWED_ROOTS = (fake_dataset, fake_state, server.REVIEW_ROOT)

        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.CatProjectorLabelReviewHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urlopen(f"{base_url}/api/cat-projector-label-review/cases?limit=1", timeout=10) as response:
                image_url = json.loads(response.read().decode("utf-8"))["cases"][0]["image_url"]
            with urlopen(f"{base_url}/api/cat-projector-label-review/videos?limit=1", timeout=10) as response:
                video = json.loads(response.read().decode("utf-8"))["videos"][0]
            with urlopen(
                f"{base_url}/api/cat-projector-label-review/videos/{video['id']}/frames?limit=1",
                timeout=10,
            ) as response:
                frame = json.loads(response.read().decode("utf-8"))["frames"][0]
            with urlopen(f"{base_url}{image_url}", timeout=10) as response:
                assert response.status == 200
                assert response.headers["Content-Type"] == "image/jpeg"
                assert response.read(16)
            with urlopen(f"{base_url}{frame['model_output_url']}", timeout=10) as response:
                assert response.status == 200
                assert response.headers["Content-Type"] == "image/jpeg"
                assert response.read(16)
            head_request = Request(f"{base_url}{image_url}", method="HEAD")
            with urlopen(head_request, timeout=10) as response:
                assert response.status == 200
                assert response.headers["Content-Type"] == "image/jpeg"
                assert int(response.headers["Content-Length"]) > 0
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
    finally:
        server.DATASET_ROOT = original_dataset
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed


def test_video_review_pairs_real_batch_input_with_ordered_annotated_outputs(tmp_path: Path) -> None:
    batch = tmp_path / "state" / "batch_reviews" / "review" / "clip"
    raw = batch / "raw"
    annotated = batch / "annotated_frames"
    raw.mkdir(parents=True)
    annotated.mkdir(parents=True)
    for index in range(3):
        image = server.Image.new("RGB", (80, 60), (40 + index, 40, 40))
        image.save(raw / f"raw_{index:04d}.jpg")
    for index in range(2):
        image = server.Image.new("RGB", (80, 60), (90, 20 + index, 20))
        image.save(annotated / f"ann_{index + 1:04d}.jpg")

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = tmp_path / "state"
        server.REVIEW_ROOT = server.STATE_ROOT / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (server.STATE_ROOT / "batch_reviews",)
        server.ALLOWED_ROOTS = (tmp_path, server.STATE_ROOT, server.REVIEW_ROOT)

        video = server._discover_videos(1)[0]
        assert video.label == "clip"
        assert video.frame_count == 3
        assert video.output_frame_count == 2
        _video, frames = server._frames_for_video(video.id, limit=3)
        assert frames[0]["model_output_path"].endswith("ann_0001.jpg")
        assert frames[1]["model_output_path"].endswith("ann_0002.jpg")
        assert frames[2]["model_output_path"] is None
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed


def test_batch_review_frames_infer_full_recording_context(tmp_path: Path) -> None:
    state = tmp_path / "state"
    recording = state / "recordings" / "20260518T184855_session"
    recording.mkdir(parents=True)
    (recording / "manifest.json").write_text("{}", encoding="utf-8")
    (recording / "chunk_0014.mp4").write_bytes(b"fake mp4")
    batch = state / "batch_reviews" / "final" / "20260518T184855_live1269_no_cat_probe" / "raw"
    batch.mkdir(parents=True)
    image = server.Image.new("RGB", (80, 60), (40, 40, 40))
    frame_path = batch / "chunk_0014_00043.jpg"
    image.save(frame_path)

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.REVIEW_ROOT = state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (state / "batch_reviews",)
        server.ALLOWED_ROOTS = (tmp_path, state, server.REVIEW_ROOT)

        source_video, source_recording = server._recording_context(frame_path)
        assert source_recording == recording.resolve()
        assert source_video == (recording / "chunk_0014.mp4").resolve()
        video = server._discover_videos(1)[0]
        assert video.source_recording_dir == recording.resolve()
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed


def test_direct_recording_review_frame_points_to_matching_chunk(tmp_path: Path) -> None:
    state = tmp_path / "state"
    recording = state / "recordings" / "20260606T000431_session"
    review_frames = recording / "review_frames"
    review_frames.mkdir(parents=True)
    (recording / "manifest.json").write_text("{}", encoding="utf-8")
    (recording / "chunk_0000.mp4").write_bytes(b"first chunk")
    (recording / "chunk_0204.mp4").write_bytes(b"highlight chunk")
    frame_path = review_frames / "chunk_0204_00105.jpg"
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(frame_path)
    (state / "sessions.jsonl").write_text(
        json.dumps(
            {
                "kind": "cat_projector_telegram_live_notification_sent",
                "telegram_live_action": "sent",
                "recording_dir": str(recording.resolve()),
                "jump_highlight": {
                    "bbox": [10, 20, 70, 120],
                    "chunk": 204,
                    "height_cm": 99.4,
                    "offset_seconds": 3.5,
                    "reason": "calibrated_high_jump",
                    "top_px": [42.0, 26.0],
                    "wall_x_mm": -25.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    original_allowed = server.ALLOWED_ROOTS
    original_state = server.STATE_ROOT
    try:
        server.STATE_ROOT = state
        server.ALLOWED_ROOTS = (tmp_path, state)
        video = server.ReviewVideo(
            id=server._video_id_for_path(recording),  # noqa: SLF001
            label=recording.name,
            source="recordings",
            mtime=recording.stat().st_mtime,
            source_recording_dir=recording.resolve(),
            source_video_path=(recording / "chunk_0000.mp4").resolve(),
        )

        frames = server._frame_payloads_for_review_video(video, [frame_path.resolve()])  # noqa: SLF001
    finally:
        server.ALLOWED_ROOTS = original_allowed
        server.STATE_ROOT = original_state

    assert frames[0]["source_video_path"] == str((recording / "chunk_0204.mp4").resolve())
    assert frames[0]["candidate_bbox_xywh"] == (10.0, 20.0, 60.0, 100.0)
    assert frames[0]["detector_backend"] == "telegram_jump_highlight"
    assert frames[0]["best_top_height_cm"] == 99.4
    assert frames[0]["measurement_point"]["image_x"] == 42.0
    assert frames[0]["measurement_point"]["image_y"] == 26.0


def test_timeline_materializes_requested_recording_frame_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "state"
    recording = state / "recordings" / "20260606T003437_session"
    recording.mkdir(parents=True)
    (recording / "manifest.json").write_text("{}", encoding="utf-8")
    (recording / "chunk_0095.mp4").write_bytes(b"fake chunk")
    (state / "sessions.jsonl").write_text(
        json.dumps(
            {
                "kind": "cat_projector_telegram_live_notification_sent",
                "telegram_live_action": "sent",
                "recording_dir": str(recording.resolve()),
                "jump_highlight": {
                    "bbox": [588, 360, 672, 488],
                    "chunk": 95,
                    "height_cm": 117.1,
                    "offset_seconds": 3.0,
                    "top_px": [610.0, 374.0],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        commands.append(command)
        if command[0] == "ffprobe":
            if "frame=best_effort_timestamp_time,pkt_pts_time" in command:
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 0,
                        "stdout": json.dumps(
                            {"frames": [{"best_effort_timestamp_time": f"{index / 30:.6f}"} for index in range(92)]}
                        ),
                    },
                )()
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": json.dumps({"streams": [{"avg_frame_rate": "30/1"}]})},
            )()
        output_pattern = Path(command[-1])
        for index in range(92):
            server.Image.new("RGB", (80, 60), (40, 40, 40)).save(
                output_pattern.with_name(output_pattern.name.replace("%05d", f"{index:05d}"))
            )
        return type("Completed", (), {"returncode": 0})()

    original_state = server.STATE_ROOT
    original_dataset = server.DATASET_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.DATASET_ROOT = tmp_path / "missing-dataset-root"
        server.SCAN_ROOTS = (state / "recordings",)
        server.ALLOWED_ROOTS = (tmp_path, state)
        server._clear_discovery_caches()  # noqa: SLF001
        monkeypatch.setattr(server.subprocess, "run", fake_run)
        monkeypatch.setattr(server, "_current_model_row_for_image", lambda _path: None)  # noqa: SLF001

        video_id = server._video_id_for_path(recording.resolve())  # noqa: SLF001
        _video, frames, suspect_queue, _reviewed, resolved_label, frame_rate = server._timeline_for_video(  # noqa: SLF001
            video_id,
            requested_frame_label="chunk_0095_00045.jpg",
        )
        current_rescore = state / "label-review" / "rescores" / "current"
        current_rescore.mkdir(parents=True)
        current_probe = current_rescore / "probe_rows.json"
        current_probe.write_text(
            json.dumps(
                [
                    {
                        "raw_path": str(recording / "review_frames" / "chunk_0095_00069.jpg"),
                        "best_probability": 0.91,
                        "best_bbox": "10,11,12,13",
                        "best_mask_polygon": [
                            {"x": 10, "y": 11},
                            {"x": 22, "y": 11},
                            {"x": 22, "y": 24},
                            {"x": 10, "y": 24},
                        ],
                        "detector_backend": "current_segmentation",
                        "detector_model_id": "current-model-test",
                        "model_created_at": "2026-06-06T17:56:42+00:00",
                        "model_path": "/tmp/current-model-test.pt",
                        "measurement_source": "current_mask_top",
                        "best_top_height_cm": 77.7,
                        "best_measurement_point": {
                            "image_x": 16.0,
                            "image_y": 17.0,
                            "wall_y_cm": 77.7,
                            "point_type": "current_mask_top",
                            "source": "current_segmentation",
                        },
                        "top_candidates": [
                            {
                                "p": 0.91,
                                "bbox": "10,11,12,13",
                                "source": "current_segmentation",
                                "mask_polygon": [
                                    {"x": 10, "y": 11},
                                    {"x": 22, "y": 11},
                                    {"x": 22, "y": 24},
                                    {"x": 10, "y": 24},
                                ],
                            },
                            {
                                "p": 0.37,
                                "bbox": "40,41,42,43",
                                "source": "current_segmentation",
                                "mask_polygon": [
                                    {"x": 40, "y": 41},
                                    {"x": 82, "y": 41},
                                    {"x": 82, "y": 84},
                                    {"x": 40, "y": 84},
                                ],
                            },
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        latest = state / "label-review" / "rescores" / "latest.json"
        latest.write_text(json.dumps({"probe_rows": str(current_probe)}), encoding="utf-8")
        _video2, explicit_frames, _suspect_queue2, _reviewed2, explicit_label, _frame_rate2 = server._timeline_for_video(  # noqa: SLF001
            video_id,
            requested_frame_label="chunk_0095_00069.jpg",
        )
        explicit_case = next(
            case
            for case in server._discover_cases(500)  # noqa: SLF001
            if case.image_path == (recording / "review_frames" / "chunk_0095_00069.jpg").resolve()
        )
        explicit_case_payload = server._case_to_payload(explicit_case, include_size=True)  # noqa: SLF001
    finally:
        server.STATE_ROOT = original_state
        server.DATASET_ROOT = original_dataset
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed
        server._clear_discovery_caches()  # noqa: SLF001

    frame_path = recording / "review_frames" / "chunk_0095_00090.jpg"
    assert frame_path.exists()
    assert (recording / "review_frames" / "chunk_0095_00089.jpg").exists()
    assert (recording / "review_frames" / "chunk_0095_00091.jpg").exists()
    assert frames[90]["label"] == "chunk_0095_00090.jpg"
    assert frames[90]["source_video_path"] == str((recording / "chunk_0095.mp4").resolve())
    assert frames[90]["candidate_bbox_xywh"] == (588.0, 360.0, 84.0, 128.0)
    assert frames[90]["current_model_overlay"] is None
    assert frames[90]["original_model_overlay"]["bbox_xywh"] == (588.0, 360.0, 84.0, 128.0)
    assert frames[90]["original_model_overlay"]["role"] == "original_model"
    assert frames[90]["best_top_height_cm"] == 117.1
    assert frames[90]["review_priority_score"] == 35.0
    assert frames[69]["label"] == "chunk_0095_00069.jpg"
    assert frames[69]["candidate_bbox_xywh"] is None
    assert frames[69]["current_model_overlay"] is None
    assert frames[69]["original_model_overlay"] is None
    assert frames[69]["model_overlays"] == []
    assert frames[69]["detector_backend"] == ""
    assert frames[69]["measurement_point"] is None
    assert frames[69]["best_top_height_cm"] is None
    assert frames[69]["review_priority_score"] is None
    assert any(item["frame_index"] == 90 for item in suspect_queue)
    assert resolved_label == "chunk_0095_00090.jpg"
    assert frame_rate == 30.0
    ffmpeg_command = next(command for command in commands if command[0] == "ffmpeg")
    assert "-vsync" in ffmpeg_command
    assert "-vf" not in ffmpeg_command
    assert explicit_label == "chunk_0095_00069.jpg"
    assert explicit_frames[69]["candidate_bbox_xywh"] == (10.0, 11.0, 12.0, 13.0)
    assert explicit_frames[69]["current_model_overlay"]["bbox_xywh"] == (10.0, 11.0, 12.0, 13.0)
    assert explicit_frames[69]["current_model_overlay"]["polygon"] == [
        {"x": 10.0, "y": 11.0},
        {"x": 22.0, "y": 11.0},
        {"x": 22.0, "y": 24.0},
        {"x": 10.0, "y": 24.0},
    ]
    assert explicit_frames[69]["current_model_overlay"]["detector_backend"] == "current_segmentation"
    assert explicit_frames[69]["current_model_overlay"]["measurement_point"]["image_x"] == 16.0
    assert explicit_frames[69]["best_top_height_cm"] == 77.7
    assert explicit_frames[69]["original_model_overlay"] is None
    assert [overlay["role"] for overlay in explicit_frames[69]["model_overlays"]] == [
        "current_model",
        "current_model_candidate",
    ]
    assert explicit_frames[69]["model_overlays"][1]["bbox_xywh"] == (40.0, 41.0, 42.0, 43.0)
    assert explicit_case_payload["detector_backend"] == "current_segmentation"
    assert explicit_case_payload["model_created_at"] == "2026-06-06T17:56:42+00:00"
    assert explicit_case_payload["model_path"] == "/tmp/current-model-test.pt"
    assert explicit_case_payload["best_top_height_cm"] == 77.7
    assert explicit_case_payload["current_model_overlay"]["detector_backend"] == "current_segmentation"
    assert [overlay["role"] for overlay in explicit_case_payload["model_overlays"]] == [
        "current_model",
        "current_model_candidate",
    ]
    assert explicit_case_payload["current_model_overlay"]["polygon"] == [
        {"x": 10.0, "y": 11.0},
        {"x": 22.0, "y": 11.0},
        {"x": 22.0, "y": 24.0},
        {"x": 10.0, "y": 24.0},
    ]
    assert len(explicit_frames) == 92
    for frame in explicit_frames:
        if frame["label"] == "chunk_0095_00069.jpg":
            assert frame["detector_backend"] == "current_segmentation"
            assert frame["measurement_source"] == "current_mask_top"
            assert frame["measurement_point"]["image_x"] == 16.0
            continue
        if frame["label"] == "chunk_0095_00090.jpg":
            assert frame["candidate_bbox_xywh"] == (588.0, 360.0, 84.0, 128.0)
            assert frame["current_model_overlay"] is None
            assert frame["model_overlays"] == [frame["original_model_overlay"]]
            assert frame["detector_backend"] == "telegram_jump_highlight"
            assert frame["measurement_source"] == "telegram_jump_highlight"
            assert frame["measurement_point"]["image_x"] == 610.0
            assert frame["measurement_point"]["image_y"] == 374.0
            assert frame["original_model_overlay"]["bbox_xywh"] == (588.0, 360.0, 84.0, 128.0)
            continue
        else:
            assert frame["candidate_bbox_xywh"] is None
            assert frame["current_model_overlay"] is None
            assert frame["original_model_overlay"] is None
            assert frame["model_overlays"] == []
            assert frame["detector_backend"] == ""
            assert frame["measurement_source"] == ""
            assert frame["measurement_point"] is None
    non_peak_frames = [
        frame
        for frame in explicit_frames
        if frame["label"] not in {"chunk_0095_00069.jpg", "chunk_0095_00090.jpg"}
    ]
    assert all(frame["best_top_height_cm"] is None for frame in non_peak_frames)
    assert all(frame["review_priority_score"] is None for frame in non_peak_frames)
    assert all(frame["original_model_overlay"] is None for frame in non_peak_frames)


def test_frame_payload_uses_telegram_render_original_overlays_per_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    recording = state / "recordings" / "20260606T033955_session"
    frame_dir = recording / "review_frames"
    frame_dir.mkdir(parents=True)
    frames = [
        frame_dir / "chunk_0114_00095.jpg",
        frame_dir / "chunk_0114_00096.jpg",
    ]
    for frame in frames:
        server.Image.new("RGB", (80, 60), (40, 40, 40)).save(frame)
    (recording / "chunk_0114.mp4").write_bytes(b"fake chunk")
    (state / "sessions.jsonl").write_text(
        json.dumps(
            {
                "kind": "cat_projector_telegram_live_notification_sent",
                "telegram_live_action": "sent",
                "recording_dir": str(recording.resolve()),
                "jump_highlight": {
                    "bbox": [520, 368, 672, 508],
                    "chunk": 114,
                    "height_cm": 115.7,
                    "offset_seconds": 4.0,
                    "top_px": [618.0, 378.0],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (recording / "telegram_live_notification.json").write_text(
        json.dumps(
            {
                "render_stats": {
                    "original_model_overlays": [
                        {
                            "frame_label": "chunk_0114_00095.jpg",
                            "source_frame_label": "chunk_0114_00095.jpg",
                            "chunk": 114,
                            "frame_index": 95,
                            "offset_seconds": 4.75,
                            "bbox": [500, 360, 640, 506],
                            "top_px": [604.0, 369.0],
                            "height_cm": 111.2,
                            "wall_x_mm": -80.0,
                            "cat_probability": 0.78,
                            "model_path": "/models/original.cbm",
                            "reason": "telegram_render_detector",
                        },
                        {
                            "frame_label": "chunk_0114_00096.jpg",
                            "source_frame_label": "chunk_0114_00096.jpg",
                            "chunk": 114,
                            "frame_index": 96,
                            "offset_seconds": 4.8,
                            "bbox": [520, 368, 672, 508],
                            "top_px": [618.0, 378.0],
                            "height_cm": 115.7,
                            "wall_x_mm": -74.6,
                            "cat_probability": 0.83,
                            "model_path": "/models/original.cbm",
                            "reason": "telegram_render_detector",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    original_allowed = server.ALLOWED_ROOTS
    original_state = server.STATE_ROOT
    try:
        server.STATE_ROOT = state
        server.ALLOWED_ROOTS = (tmp_path, state)
        monkeypatch.setattr(server, "_current_model_row_for_image", lambda _path: None)  # noqa: SLF001
        video = server.ReviewVideo(
            id=server._video_id_for_path(recording),  # noqa: SLF001
            label=recording.name,
            source="recordings",
            mtime=recording.stat().st_mtime,
            source_recording_dir=recording.resolve(),
            source_video_path=(recording / "chunk_0114.mp4").resolve(),
        )

        payloads = server._frame_payloads_for_review_video(video, [frame.resolve() for frame in frames])  # noqa: SLF001
    finally:
        server.ALLOWED_ROOTS = original_allowed
        server.STATE_ROOT = original_state
        server._clear_discovery_caches()  # noqa: SLF001

    assert [frame["label"] for frame in payloads] == ["chunk_0114_00095.jpg", "chunk_0114_00096.jpg"]
    assert payloads[0]["original_model_overlay"]["bbox_xywh"] == (500.0, 360.0, 140.0, 146.0)
    assert payloads[0]["original_model_overlay"]["detector_backend"] == "telegram_render_original"
    assert payloads[0]["original_model_overlay"]["detector_probability"] == 0.78
    assert payloads[0]["measurement_point"]["source"] == "telegram_render_original"
    assert payloads[1]["original_model_overlay"]["bbox_xywh"] == (520.0, 368.0, 152.0, 140.0)
    assert payloads[1]["original_model_overlay"]["top_height_cm"] == 115.7


def test_frame_payload_adds_on_demand_current_model_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    frame_dir = state / "recordings" / "session" / "review_frames"
    frame_dir.mkdir(parents=True)
    frame_path = frame_dir / "chunk_0001_00008.jpg"
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(frame_path)

    def fake_current_row(path: Path) -> dict[str, object] | None:
        assert path == frame_path.resolve()
        return {
            "detector_cat_probability": "0.87",
            "candidate_bbox_xywh": "10,11,12,13",
            "best_mask_polygon": [
                {"x": 10, "y": 11},
                {"x": 22, "y": 11},
                {"x": 22, "y": 24},
                {"x": 10, "y": 24},
            ],
            "detector_backend": "current_model_test",
            "detector_model_id": "current-model-v1",
            "measurement_source": "current_bbox_top",
            "best_measurement_point": {
                "image_x": 16.0,
                "image_y": 11.0,
                "point_type": "current_bbox_top",
                "source": "current_model_test",
            },
        }

    original_allowed = server.ALLOWED_ROOTS
    original_state = server.STATE_ROOT
    try:
        server.STATE_ROOT = state
        server.ALLOWED_ROOTS = (tmp_path, state)
        server._clear_discovery_caches()  # noqa: SLF001
        monkeypatch.setattr(server, "_current_model_row_for_image", fake_current_row)  # noqa: SLF001
        video = server.ReviewVideo(
            id="video.test",
            label="session",
            source="recordings",
            mtime=frame_path.stat().st_mtime,
            source_recording_dir=frame_path.parents[1],
        )

        frames = server._frame_payloads_for_review_video(video, [frame_path.resolve()])  # noqa: SLF001
    finally:
        server.ALLOWED_ROOTS = original_allowed
        server.STATE_ROOT = original_state
        server._clear_discovery_caches()  # noqa: SLF001

    assert frames[0]["current_model_overlay"]["role"] == "current_model"
    assert frames[0]["current_model_overlay"]["bbox_xywh"] == (10.0, 11.0, 12.0, 13.0)
    assert frames[0]["current_model_overlay"]["polygon"] == [
        {"x": 10.0, "y": 11.0},
        {"x": 22.0, "y": 11.0},
        {"x": 22.0, "y": 24.0},
        {"x": 10.0, "y": 24.0},
    ]
    assert frames[0]["current_model_overlay"]["detector_backend"] == "current_model_test"
    assert frames[0]["current_model_overlay"]["measurement_point"]["image_x"] == 16.0
    assert frames[0]["model_overlays"] == [frames[0]["current_model_overlay"]]


def test_video_review_counts_reprocessed_session_artifacts(tmp_path: Path) -> None:
    state = tmp_path / "state"
    recording = state / "recordings" / "20260518T184855_session"
    recording.mkdir(parents=True)
    (recording / "manifest.json").write_text("{}", encoding="utf-8")
    (recording / "chunk_0001.mp4").write_bytes(b"fake chunk")
    batch = state / "batch_reviews" / "final" / "20260518T184855_review" / "raw"
    batch.mkdir(parents=True)
    image = server.Image.new("RGB", (80, 60), (40, 40, 40))
    image.save(batch / "chunk_0001_00001.jpg")

    reprocessed = state / "reprocessed" / "review-ui-video-test"
    reprocessed.mkdir(parents=True)
    horizontal = reprocessed / "annotated_full_session.mp4"
    vertical = reprocessed / "annotated_full_session_vertical.mp4"
    horizontal.write_bytes(b"fake horizontal")
    vertical.write_bytes(b"fake vertical")
    manifest = {
        "kind": "cat_projector_review_reprocess_output_v1",
        "review_video_id": "video.test",
        "source_recording_dir": str(recording.resolve()),
        "outputs": [str(horizontal), str(vertical)],
    }
    (reprocessed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.REVIEW_ROOT = state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (state / "batch_reviews",)
        server.ALLOWED_ROOTS = (tmp_path, state, server.REVIEW_ROOT)

        video = server._discover_videos(1)[0]
        payload = server._video_payload(video)
        assert video.output_frame_count == 0
        assert payload["output_artifact_count"] == 2
        assert [artifact["path"] for artifact in payload["output_artifacts"]] == [
            str(horizontal.resolve()),
            str(vertical.resolve()),
        ]
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed


def test_untimestamped_batch_review_infers_recording_from_chunk_and_output_manifest(tmp_path: Path) -> None:
    state = tmp_path / "state"
    first_recording = state / "recordings" / "20260518T010000_first"
    target_recording = state / "recordings" / "20260518T184855_target"
    for recording in (first_recording, target_recording):
        recording.mkdir(parents=True)
        (recording / "manifest.json").write_text("{}", encoding="utf-8")
        (recording / "chunk_0154.mp4").write_bytes(b"fake chunk")
    image = server.Image.new("RGB", (80, 60), (10, 10, 10))
    image.save(target_recording / "startup_projector_camera.jpg")
    reprocessed = state / "reprocessed" / "review-ui-target"
    reprocessed.mkdir(parents=True)
    (reprocessed / "annotated_full_session.mp4").write_bytes(b"fake output")
    (reprocessed / "manifest.json").write_text(
        json.dumps({"source_recording_dir": str(target_recording.resolve())}),
        encoding="utf-8",
    )
    raw = state / "batch_reviews" / "final_review_20260518" / "no_cat_frame_candidate_probe" / "raw"
    raw.mkdir(parents=True)
    frame_path = raw / "chunk_0154_00019.jpg"
    image = server.Image.new("RGB", (80, 60), (40, 40, 40))
    image.save(frame_path)

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.REVIEW_ROOT = state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (state / "batch_reviews",)
        server.ALLOWED_ROOTS = (tmp_path, state, server.REVIEW_ROOT)

        source_video, source_recording = server._recording_context(frame_path)
        assert source_recording == target_recording.resolve()
        assert source_video == (target_recording / "chunk_0154.mp4").resolve()
        video = server._discover_videos(1)[0]
        payload = server._video_payload(video)
        assert payload["source_recording_dir"] == str(target_recording.resolve())
        assert payload["frame_count"] == 1
        assert payload["output_artifact_count"] == 1
        _video, frames = server._frames_for_video(video.id, limit=10)
        assert [frame["image_path"] for frame in frames] == [str(frame_path.resolve())]
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed


def test_video_frames_skip_rendered_output_artifacts_for_labeling(tmp_path: Path) -> None:
    state = tmp_path / "state"
    recording = state / "recordings" / "20260518T184855_target"
    recording.mkdir(parents=True)
    (recording / "manifest.json").write_text("{}", encoding="utf-8")
    (recording / "chunk_0154.mp4").write_bytes(b"fake chunk")
    review = state / "batch_reviews" / "final_review_20260518" / "20260518T184855_probe"
    raw = review / "raw"
    raw.mkdir(parents=True)
    image = server.Image.new("RGB", (80, 60), (40, 40, 40))
    input_frame = raw / "chunk_0154_00019.jpg"
    image.save(input_frame)
    image.save(review / "frame_0145_hold_aspect_fix.jpg")
    image.save(review / "annotated_120_145_sheet.jpg")

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.REVIEW_ROOT = state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (state / "batch_reviews",)
        server.ALLOWED_ROOTS = (tmp_path, state, server.REVIEW_ROOT)

        video = server._discover_videos(1)[0]
        _video, frames = server._frames_for_video(video.id, limit=10)
        assert [frame["image_path"] for frame in frames] == [str(input_frame.resolve())]
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed


def test_generated_review_training_packages_are_not_review_queue_inputs(tmp_path: Path) -> None:
    state = tmp_path / "state"
    real_frames = state / "datasets" / "real-corpus" / "frames"
    generated_frames = state / "datasets" / "cat-projector-review-ui-all-review-labels-20260522" / "frames"
    real_frames.mkdir(parents=True)
    generated_frames.mkdir(parents=True)
    image = server.Image.new("RGB", (80, 60), (40, 40, 40))
    real_image = real_frames / "real.jpg"
    generated_image = generated_frames / "copy.jpg"
    image.save(real_image)
    image.save(generated_image)

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.REVIEW_ROOT = state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (state / "datasets",)
        server.ALLOWED_ROOTS = (tmp_path, state, server.REVIEW_ROOT)

        cases = server._discover_cases(10)
        assert [case.image_path for case in cases] == [real_image.resolve()]
        videos = server._discover_videos(10)
        assert all("cat-projector-review-ui-" not in str(video.source_recording_dir or video.label) for video in videos)
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed


def test_review_metadata_reads_latest_rescore_only(tmp_path: Path) -> None:
    state = tmp_path / "state"
    image_path = state / "datasets" / "real-corpus" / "frames" / "real.jpg"
    image_path.parent.mkdir(parents=True)
    image = server.Image.new("RGB", (80, 60), (40, 40, 40))
    image.save(image_path)
    old_run = state / "label-review" / "rescores" / "old"
    new_run = state / "label-review" / "rescores" / "new"
    old_run.mkdir(parents=True)
    new_run.mkdir(parents=True)
    old_probe = old_run / "probe_rows.json"
    new_probe = new_run / "probe_rows.json"
    old_probe.write_text(
        json.dumps([{"raw_path": str(image_path), "best_probability": 0.1, "best_bbox": "1,1,10,10"}]),
        encoding="utf-8",
    )
    new_probe.write_text(
        json.dumps([{"raw_path": str(image_path), "best_probability": 0.5, "best_bbox": "2,2,20,20"}]),
        encoding="utf-8",
    )
    latest = state / "label-review" / "rescores" / "latest.json"
    latest.write_text(json.dumps({"probe_rows": str(new_probe)}), encoding="utf-8")

    original_state = server.STATE_ROOT
    original_dataset = server.DATASET_ROOT
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.DATASET_ROOT = tmp_path / "missing-dataset-root"
        server.ALLOWED_ROOTS = (tmp_path, state)

        rows = server._review_metadata_rows_by_image()
        assert rows[image_path.resolve()]["detector_cat_probability"] == "0.5"
        assert rows[image_path.resolve()]["candidate_bbox_xywh"] == "2,2,20,20"
    finally:
        server.STATE_ROOT = original_state
        server.DATASET_ROOT = original_dataset
        server.ALLOWED_ROOTS = original_allowed


def test_timeline_keeps_saved_not_cat_out_of_unreviewed_suspects(tmp_path: Path) -> None:
    state = tmp_path / "state"
    frames_dir = state / "datasets" / "session" / "frames"
    frames_dir.mkdir(parents=True)
    saved_frame = frames_dir / "saved-not-cat.jpg"
    fresh_frame = frames_dir / "fresh-suspect.jpg"
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(saved_frame)
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(fresh_frame)

    rescore = state / "label-review" / "rescores" / "new"
    rescore.mkdir(parents=True)
    probe = rescore / "probe_rows.json"
    probe.write_text(
        json.dumps(
            [
                {
                    "raw_path": str(saved_frame),
                    "best_probability": 0.51,
                    "detector_backend": "ultralytics_yolo_segmentation",
                    "best_top_height_cm": 180.0,
                    "measurement_source": "mask_top_p5",
                    "review_priority_score": 80.0,
                    "review_priority_reasons": ["high reviewed height"],
                },
                {
                    "raw_path": str(fresh_frame),
                    "best_probability": 0.5,
                    "detector_backend": "ultralytics_yolo_segmentation",
                    "best_top_height_cm": 150.0,
                    "measurement_source": "mask_top_p5",
                },
            ]
        ),
        encoding="utf-8",
    )
    latest = state / "label-review" / "rescores" / "latest.json"
    latest.write_text(json.dumps({"probe_rows": str(probe)}), encoding="utf-8")

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.REVIEW_ROOT = state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (state / "datasets",)
        server.ALLOWED_ROOTS = (tmp_path, state, server.REVIEW_ROOT)
        server.LABELS_ROOT.mkdir(parents=True)
        case_id = server._case_id_for_path(saved_frame)  # noqa: SLF001
        (server.LABELS_ROOT / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "image_path": str(saved_frame),
                    "label": "not_cat",
                    "label_cat_present": "no",
                    "label_candidate_is_cat": "no",
                    "cat_present": False,
                    "candidate_is_cat": False,
                    "review_decision": "false_positive",
                    "review_status": "saved",
                }
            ),
            encoding="utf-8",
        )

        video = server._discover_videos(1)[0]
        _video, frames, suspect_queue, reviewed_suspect_queue, _resolved_label, _frame_rate = server._timeline_for_video(  # noqa: SLF001
            video.id
        )
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed

    saved = next(frame for frame in frames if frame["image_path"] == str(saved_frame.resolve()))
    fresh = next(frame for frame in frames if frame["image_path"] == str(fresh_frame.resolve()))
    assert saved["reviewed"] is True
    assert fresh["reviewed"] is False
    assert fresh["current_model_overlay"]["role"] == "current_model"
    assert fresh["current_model_overlay"]["detector_backend"] == "ultralytics_yolo_segmentation"
    assert fresh["current_model_overlay"]["top_height_cm"] == 150.0
    assert fresh["original_model_overlay"] is None
    assert fresh["model_overlays"] == [fresh["current_model_overlay"]]
    assert {item["frame_id"] for item in suspect_queue} == {fresh["id"]}
    assert {item["frame_id"] for item in reviewed_suspect_queue} == {saved["id"]}
    assert "model output missing" not in fresh["suspicion_reasons"]


def test_active_learning_remeasures_jump_heights_from_mask_measurement_before_bbox(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "kind": "cat_projector_wall_plane_calibration_v1",
                "image_points_px": [[0, 0], [100, 0], [0, 100], [100, 100]],
                "wall_points_cm": [[0, 100], [100, 100], [0, 0], [100, 0]],
            }
        ),
        encoding="utf-8",
    )
    frame = tmp_path / "recordings" / "rec-high" / "frames" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    server.Image.new("RGB", (100, 100), (40, 40, 40)).save(frame)

    rows = [
        {
            "raw_path": str(frame),
            "global_frame": 3,
            "best_bbox": "10,20,30,40",
            "best_probability": 0.8,
            "best_source": "fresh_model",
            "best_measurement_point": {
                "point_type": "mask_top_p5",
                "image_x": 25.0,
                "image_y": 10.0,
                "confidence": 0.9,
                "source": "segmentation_mask",
            },
        }
    ]

    original_state = server.STATE_ROOT
    original_allowed = server.ALLOWED_ROOTS
    original_active_state = active_learning.review.STATE_ROOT
    original_active_allowed = active_learning.review.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = tmp_path
        server.ALLOWED_ROOTS = (tmp_path,)
        active_learning.review.STATE_ROOT = tmp_path
        active_learning.review.ALLOWED_ROOTS = (tmp_path,)
        summary = active_learning._remeasure_jump_heights(  # noqa: SLF001
            rows,
            calibration_path=calibration_path,
            min_probability=0.5,
        )
    finally:
        server.STATE_ROOT = original_state
        server.ALLOWED_ROOTS = original_allowed
        active_learning.review.STATE_ROOT = original_active_state
        active_learning.review.ALLOWED_ROOTS = original_active_allowed

    assert rows[0]["best_top_height_cm"] == 90.0
    assert rows[0]["legacy_bbox_top_height_cm"] == 80.0
    assert rows[0]["measurement_source"] == "mask_top_p5"
    assert rows[0]["tracker_status"] == "accepted"
    assert rows[0]["tracker_reason"] == "initialized"
    assert summary["status"] == "done"
    assert summary["videos"][0]["max_jump_height_cm"] == 90.0


def test_active_learning_does_not_measure_reviewed_not_cat_as_jump(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "kind": "cat_projector_wall_plane_calibration_v1",
                "image_points_px": [[0, 0], [100, 0], [0, 100], [100, 100]],
                "wall_points_cm": [[0, 100], [100, 100], [0, 0], [100, 0]],
            }
        ),
        encoding="utf-8",
    )
    frame = tmp_path / "recordings" / "rec-false-jump" / "frames" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    server.Image.new("RGB", (100, 100), (40, 40, 40)).save(frame)
    rows = [
        {
            "raw_path": str(frame),
            "global_frame": 3,
            "best_bbox": "10,20,30,40",
            "best_probability": 0.9,
            "best_source": "fresh_model",
            "best_measurement_point": {
                "point_type": "mask_top_p5",
                "image_x": 25.0,
                "image_y": 10.0,
                "confidence": 0.95,
                "source": "segmentation_mask",
            },
        }
    ]

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_allowed = server.ALLOWED_ROOTS
    original_active_state = active_learning.review.STATE_ROOT
    original_active_review = active_learning.review.REVIEW_ROOT
    original_active_labels = active_learning.review.LABELS_ROOT
    original_active_allowed = active_learning.review.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = tmp_path
        server.REVIEW_ROOT = tmp_path / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.ALLOWED_ROOTS = (tmp_path,)
        active_learning.review.STATE_ROOT = tmp_path
        active_learning.review.REVIEW_ROOT = server.REVIEW_ROOT
        active_learning.review.LABELS_ROOT = server.LABELS_ROOT
        active_learning.review.ALLOWED_ROOTS = (tmp_path,)
        server.LABELS_ROOT.mkdir(parents=True)
        case_id = server._case_id_for_path(frame)  # noqa: SLF001
        (server.LABELS_ROOT / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "image_path": str(frame),
                    "label": "not_cat",
                    "label_cat_present": "no",
                    "label_candidate_is_cat": "no",
                    "cat_present": False,
                    "candidate_is_cat": False,
                    "review_decision": "false_positive",
                    "review_status": "saved",
                }
            ),
            encoding="utf-8",
        )

        summary = active_learning._remeasure_jump_heights(  # noqa: SLF001
            rows,
            calibration_path=calibration_path,
            min_probability=0.5,
        )
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.ALLOWED_ROOTS = original_allowed
        active_learning.review.STATE_ROOT = original_active_state
        active_learning.review.REVIEW_ROOT = original_active_review
        active_learning.review.LABELS_ROOT = original_active_labels
        active_learning.review.ALLOWED_ROOTS = original_active_allowed

    assert summary["status"] == "done"
    assert summary["measured_frame_count"] == 0
    assert summary["accepted_frame_count"] == 0
    assert summary["videos"] == []
    assert rows[0]["review_excluded_from_jump_height"] is True
    assert rows[0]["review_exclusion_reason"] == "human_review_not_cat"
    assert "best_top_height_cm" not in rows[0]


def test_active_learning_scales_frame_coordinates_to_calibration_size(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "kind": "cat_projector_wall_plane_calibration_v1",
                "image_size_px": [1280, 720],
                "image_points_px": [[0, 0], [1280, 0], [0, 720], [1280, 720]],
                "wall_points_cm": [[0, 720], [1280, 720], [0, 0], [1280, 0]],
            }
        ),
        encoding="utf-8",
    )
    frame = tmp_path / "recordings" / "rec-scaled" / "frames" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    server.Image.new("RGB", (640, 360), (40, 40, 40)).save(frame)
    rows = [
        {
            "raw_path": str(frame),
            "global_frame": 3,
            "best_bbox": "190,300,20,40",
            "best_probability": 0.95,
            "best_source": "fresh_model",
            "source_size_px": {"width": 640, "height": 360},
            "best_measurement_point": {
                "point_type": "mask_top_p5",
                "image_x": 200.0,
                "image_y": 329.0,
                "confidence": 0.95,
                "source": "segmentation_mask",
            },
        }
    ]

    summary = active_learning._remeasure_jump_heights(  # noqa: SLF001
        rows,
        calibration_path=calibration_path,
        min_probability=0.99,
    )

    assert summary["status"] == "done"
    assert summary["measured_frame_count"] == 1
    assert rows[0]["best_top_height_cm"] == 62.0
    assert rows[0]["best_measurement_point"]["debug"]["coordinate_transform"]["calibration_image_x"] == 400.0
    assert rows[0]["best_measurement_point"]["debug"]["coordinate_transform"]["calibration_image_y"] == 658.0


def test_active_learning_aligns_current_frame_to_calibration_reference(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "kind": "cat_projector_wall_plane_calibration_v1",
                "image_size_px": [200, 160],
                "image_points_px": [[0, 0], [200, 0], [0, 160], [200, 160]],
                "wall_points_cm": [[0, 160], [200, 160], [0, 0], [200, 0]],
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(20260523)
    reference = np.full((160, 200), 80, dtype=np.uint8)
    for x, y in rng.integers([10, 10], [190, 150], size=(120, 2)):
        cv2.circle(reference, (int(x), int(y)), 2, 220, -1)
    transform = np.float32([[1, 0, 7], [0, 1, 10]])
    current = cv2.warpAffine(reference, transform, (200, 160), borderValue=80)
    reference_path = tmp_path / "reference.jpg"
    frame = tmp_path / "recordings" / "rec-aligned" / "frames" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(frame), current)
    rows = [
        {
            "raw_path": str(frame),
            "global_frame": 3,
            "best_bbox": "52,45,20,30",
            "best_probability": 0.95,
            "best_source": "fresh_model",
            "source_size_px": {"width": 200, "height": 160},
            "best_measurement_point": {
                "point_type": "mask_top_p5",
                "image_x": 57.0,
                "image_y": 50.0,
                "confidence": 0.95,
                "source": "segmentation_mask",
            },
        }
    ]

    summary = active_learning._remeasure_jump_heights(  # noqa: SLF001
        rows,
        calibration_path=calibration_path,
        min_probability=0.99,
        alignment_reference_path=reference_path,
    )

    alignment = rows[0]["best_measurement_point"]["debug"]["coordinate_transform"]["alignment"]
    assert summary["frame_alignment"]["applied_frame_count"] == 1
    assert alignment["applied"] is True
    assert abs(alignment["aligned_calibration_image_x"] - 50.0) < 2.0
    assert abs(alignment["aligned_calibration_image_y"] - 40.0) < 2.0
    assert abs(rows[0]["best_top_height_cm"] - 120.0) < 2.0


def test_projection_edge_alignment_uses_outer_screen_edges() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    image = np.full((260, 360), 40, dtype=np.uint8)
    cv2.rectangle(image, (40, 25), (320, 230), 230, 3)
    cv2.line(image, (45, 130), (315, 130), 230, 3)

    edges = active_learning._detect_projection_edges(image)  # noqa: SLF001

    assert edges is not None
    assert abs(edges["top_y"] - 25) < 5
    assert abs(edges["bottom_y"] - 230) < 5
    assert abs(edges["left_x"] - 40) < 5
    assert abs(edges["right_x"] - 320) < 5
    assert (
        active_learning._edge_alignment_from_edges(  # noqa: SLF001
            {"top_y": 20, "bottom_y": 155, "left_x": 30, "right_x": 220},
            {"top_y": 20, "bottom_y": 155, "left_x": 30, "right_x": 460},
        )
        is None
    )


def test_discover_videos_sorts_by_latest_remeasured_jump_height(tmp_path: Path) -> None:
    state = tmp_path / "state"
    low = state / "recordings" / "20260518T010000_low"
    high = state / "recordings" / "20260518T020000_high"
    for recording in (low, high):
        frames = recording / "frames"
        frames.mkdir(parents=True)
        (recording / "manifest.json").write_text("{}", encoding="utf-8")
        (recording / "chunk_0000.mp4").write_bytes(b"fake chunk")
        server.Image.new("RGB", (80, 60), (40, 40, 40)).save(frames / "frame.jpg")

    latest = state / "label-review" / "rescores" / "jump_heights_latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(
        json.dumps(
            {
                "kind": "cat_projector_active_learning_jump_heights_v1",
                "status": "done",
                "videos": [
                    {
                        "recording_dir": str(low.resolve()),
                        "max_jump_height_cm": 42.0,
                    },
                    {
                        "recording_dir": str(high.resolve()),
                        "max_jump_height_cm": 150.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    stale = state / "batch_reviews" / "old" / "scan_results.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        json.dumps([{"recording": str(low.resolve()), "max_height_cm": 999.0}]),
        encoding="utf-8",
    )

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.REVIEW_ROOT = state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (state / "recordings",)
        server.ALLOWED_ROOTS = (tmp_path, state, server.REVIEW_ROOT)

        videos = server._discover_videos(10)
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed

    assert [video.source_recording_dir for video in videos[:2]] == [high.resolve(), low.resolve()]
    assert [video.max_jump_height_cm for video in videos[:2]] == [150.5, 42.0]


def test_discover_videos_uses_previous_rescore_when_latest_height_index_is_empty(tmp_path: Path) -> None:
    state = tmp_path / "state"
    recording = state / "recordings" / "20260518T020000_high"
    frames = recording / "frames"
    frames.mkdir(parents=True)
    (recording / "manifest.json").write_text("{}", encoding="utf-8")
    (recording / "chunk_0000.mp4").write_bytes(b"fake chunk")
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(frames / "frame.jpg")

    latest = state / "label-review" / "rescores" / "jump_heights_latest.json"
    previous = state / "label-review" / "rescores" / "active-learning-previous" / "jump_heights.json"
    previous.parent.mkdir(parents=True)
    latest.write_text(
        json.dumps({"kind": "cat_projector_active_learning_jump_heights_v1", "status": "done", "videos": []}),
        encoding="utf-8",
    )
    previous.write_text(
        json.dumps(
            {
                "kind": "cat_projector_active_learning_jump_heights_v1",
                "status": "done",
                "videos": [{"recording_dir": str(recording.resolve()), "max_jump_height_cm": 202.6}],
            }
        ),
        encoding="utf-8",
    )
    stale = state / "batch_reviews" / "old" / "scan_results.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps([{"recording": str(recording.resolve()), "max_height_cm": 999.0}]), encoding="utf-8")

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.REVIEW_ROOT = state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (state / "recordings",)
        server.ALLOWED_ROOTS = (tmp_path, state, server.REVIEW_ROOT)

        videos = server._discover_videos(10)
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed

    assert len(videos) == 1
    assert videos[0].max_jump_height_cm == 202.6
    assert "active-learning-previous" in videos[0].max_jump_height_source


def test_discover_videos_suppresses_latest_height_when_max_frame_is_reviewed_not_cat(tmp_path: Path) -> None:
    state = tmp_path / "state"
    recording = state / "recordings" / "20260517T203508_false_jump"
    frames = recording / "frames"
    frames.mkdir(parents=True)
    frame = frames / "frame.jpg"
    artifact = recording / "telegram_notification" / "203_height_first_known_window.sheet.jpg"
    (recording / "manifest.json").write_text("{}", encoding="utf-8")
    (recording / "chunk_0000.mp4").write_bytes(b"fake chunk")
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(frame)
    artifact.parent.mkdir(parents=True)
    server.Image.new("RGB", (80, 60), (40, 40, 40)).save(artifact)

    latest = state / "label-review" / "rescores" / "jump_heights_latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(
        json.dumps(
            {
                "kind": "cat_projector_active_learning_jump_heights_v1",
                "status": "done",
                "videos": [
                    {
                        "recording_dir": str(recording.resolve()),
                        "max_jump_height_cm": 206.0,
                        "max_frame_path": str(frame.resolve()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stale = state / "batch_reviews" / "old" / "scan_results.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        json.dumps([{"recording": str(recording.resolve()), "max_height_cm": 999.0}]),
        encoding="utf-8",
    )

    original_state = server.STATE_ROOT
    original_review = server.REVIEW_ROOT
    original_labels = server.LABELS_ROOT
    original_masks = server.MASKS_ROOT
    original_queue = server.QUEUE_ROOT
    original_video_status = server.VIDEO_STATUS_ROOT
    original_scan_roots = server.SCAN_ROOTS
    original_allowed = server.ALLOWED_ROOTS
    try:
        server.STATE_ROOT = state
        server.REVIEW_ROOT = state / "label-review"
        server.LABELS_ROOT = server.REVIEW_ROOT / "labels"
        server.MASKS_ROOT = server.REVIEW_ROOT / "masks"
        server.QUEUE_ROOT = server.REVIEW_ROOT / "actions"
        server.VIDEO_STATUS_ROOT = server.REVIEW_ROOT / "videos"
        server.SCAN_ROOTS = (state / "recordings",)
        server.ALLOWED_ROOTS = (tmp_path, state, server.REVIEW_ROOT)
        server.LABELS_ROOT.mkdir(parents=True)
        case_id = server._case_id_for_path(frame)  # noqa: SLF001
        (server.LABELS_ROOT / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "image_path": str(frame),
                    "label": "not_cat",
                    "label_cat_present": "no",
                    "label_candidate_is_cat": "no",
                    "cat_present": False,
                    "candidate_is_cat": False,
                    "review_decision": "false_positive",
                    "review_status": "saved",
                }
            ),
            encoding="utf-8",
        )

        videos = server._discover_videos(10)
        review_frames = server._recording_frame_paths(recording)  # noqa: SLF001
    finally:
        server.STATE_ROOT = original_state
        server.REVIEW_ROOT = original_review
        server.LABELS_ROOT = original_labels
        server.MASKS_ROOT = original_masks
        server.QUEUE_ROOT = original_queue
        server.VIDEO_STATUS_ROOT = original_video_status
        server.SCAN_ROOTS = original_scan_roots
        server.ALLOWED_ROOTS = original_allowed

    assert len(videos) == 1
    assert review_frames == [frame.resolve()]
    assert videos[0].max_jump_height_cm is None
    assert videos[0].max_jump_height_source == ""
