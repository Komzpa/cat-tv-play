from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_label_review_server.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("cat_projector_label_review_server", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
server = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = server
SCRIPT_SPEC.loader.exec_module(server)


def test_fake_smoke_builds_review_queue_and_action_records(tmp_path: Path) -> None:
    assert server.run_fake_smoke(tmp_path) == 0
    label_files = sorted((tmp_path / "state" / "label-review" / "labels").glob("*.json"))
    action_files = sorted((tmp_path / "state" / "label-review" / "actions").glob("*.json"))
    mask_files = sorted((tmp_path / "state" / "label-review" / "masks").glob("*/*.json"))
    video_status_files = sorted((tmp_path / "state" / "label-review" / "videos").glob("*.json"))
    training_labels = sorted((tmp_path / "state" / "datasets").glob("cat-projector-review-ui-*/labels.csv"))
    assert len(label_files) == 2
    assert len(action_files) == 2
    assert len(mask_files) == 1
    assert len(video_status_files) == 1
    assert len(training_labels) == 1
    saved_labels = [json.loads(path.read_text(encoding="utf-8")) for path in label_files]
    assert {row["label_candidate_is_cat"] for row in saved_labels} == {"yes", "no"}
    assert {row["video_id"] for row in saved_labels} != {None}
    training_text = training_labels[0].read_text(encoding="utf-8")
    assert "label_candidate_is_cat" in training_text
    assert "yes" in training_text
    assert "no" in training_text


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
