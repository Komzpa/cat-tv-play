from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_cat_projector_labeling_package.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("build_cat_projector_labeling_package", SCRIPT_PATH)
builder = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = builder
SCRIPT_SPEC.loader.exec_module(builder)


def _write_recording(root: Path, name: str, *, session_id: str, video_url: str = "") -> Path:
    recording_dir = root / name
    recording_dir.mkdir(parents=True)
    (recording_dir / "manifest.json").write_text(
        json.dumps(
            {
                "frigate_event_id": "event-1",
                "ha_session_id": session_id,
                "video_url": video_url,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (recording_dir / "chunk_0000.mp4").write_bytes(b"fake mp4")
    return recording_dir


def test_discover_completed_chunks_includes_mkv_and_skips_part_files(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    recording_dir.mkdir()
    (recording_dir / "chunk_0000.mkv").write_bytes(b"fake mkv")
    (recording_dir / "chunk_0001.mp4").write_bytes(b"fake mp4")
    (recording_dir / "chunk_0002.part.mp4").write_bytes(b"partial")
    (recording_dir / "chunk_0003.txt").write_text("not video", encoding="utf-8")

    chunks = builder._discover_completed_chunks(recording_dir)

    assert [item.name for item in chunks] == ["chunk_0000.mkv", "chunk_0001.mp4"]


def test_discover_recordings_prioritizes_known_false_negative(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    _write_recording(
        root,
        "20260516T003119_20260516T003115-cat-sher-pounce-lanes",
        session_id="20260516T003115-cat-sher-pounce-lanes",
        video_url="http://ha/local/cat-tv/cat-sher-pounce-lanes.mp4",
    )
    _write_recording(
        root,
        "20260516T000507_20260516T000333-cats-tv-real-bugs-GkWGkmnMBCw-seed30",
        session_id="20260516T000333-cats-tv-real-bugs-GkWGkmnMBCw-seed30",
    )

    recordings = builder.discover_recordings(
        root,
        false_negative_sessions=("20260516T003115-cat-sher-pounce-lanes",),
    )

    assert [item.path.name for item in recordings] == [
        "20260516T003119_20260516T003115-cat-sher-pounce-lanes",
        "20260516T000507_20260516T000333-cats-tv-real-bugs-GkWGkmnMBCw-seed30",
    ]
    assert recordings[0].candidate_reason == "user_observed_false_negative_window"
    assert recordings[0].video_slug == "cat-sher-pounce-lanes.mp4"


def test_discover_recordings_filters_session_patterns(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    _write_recording(root, "one", session_id="cat-sher-pounce-lanes")
    _write_recording(root, "two", session_id="cats-tv-real-bugs")

    recordings = builder.discover_recordings(root, session_patterns=["real-bugs"])

    assert [item.path.name for item in recordings] == ["two"]


def test_discover_recordings_limits_repeated_session_windows(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    for index in range(3):
        _write_recording(root, f"repeat-{index}", session_id="same-session")
    _write_recording(root, "fresh", session_id="fresh-session")

    recordings = builder.discover_recordings(root, max_recordings_per_session=1)

    assert [item.path.name for item in recordings] == ["fresh", "repeat-0"]


def test_label_schema_documents_local_no_cloud_path(tmp_path: Path) -> None:
    schema_path = tmp_path / "LABEL_SCHEMA.md"
    plan_path = tmp_path / "LOCAL_MODEL_PLAN.md"

    builder.write_label_schema(schema_path)
    builder.write_local_model_plan(plan_path)

    assert "label_cat_present" in schema_path.read_text(encoding="utf-8")
    plan = plan_path.read_text(encoding="utf-8")
    assert "local model, no Frigate+ upload" in plan
    assert "state classifier" in plan


def _build_args(recordings_root: Path, output_root: Path, **overrides: object) -> Namespace:
    payload = {
        "contact_sheet_frames": 0,
        "false_negative_session": list(builder.DEFAULT_FALSE_NEGATIVE_SESSIONS),
        "max_frames_per_recording": 10,
        "max_recordings": 0,
        "max_recordings_per_session": 0,
        "max_total_frames": 10,
        "offsets": (0.5, 1.0, 1.5),
        "output_root": output_root,
        "package_id": "package",
        "recordings_root": recordings_root,
        "replace_existing": False,
        "session_pattern": [],
    }
    payload.update(overrides)
    return Namespace(**payload)


def test_build_package_fails_when_no_frames_extract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recordings_root = tmp_path / "recordings"
    _write_recording(recordings_root, "one", session_id="session")
    monkeypatch.setattr(builder, "_ffprobe_duration_seconds", lambda _path: 2.0)
    monkeypatch.setattr(builder, "_extract_frame", lambda _source, _offset, _output: False)

    with pytest.raises(RuntimeError, match="no frames were extracted"):
        builder.build_package(_build_args(recordings_root, tmp_path / "datasets"))


def test_build_package_does_not_write_unlisted_frames_past_total_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings_root = tmp_path / "recordings"
    _write_recording(recordings_root, "one", session_id="session")
    monkeypatch.setattr(builder, "_ffprobe_duration_seconds", lambda _path: 2.0)

    extracted: list[Path] = []

    def fake_extract(_source: Path, _offset: float, output: Path) -> bool:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"jpeg enough bytes for test")
        extracted.append(output)
        return True

    monkeypatch.setattr(builder, "_extract_frame", fake_extract)

    builder.build_package(
        _build_args(
            recordings_root,
            tmp_path / "datasets",
            max_total_frames=2,
        )
    )

    package_dir = tmp_path / "datasets" / "package"
    rows = (package_dir / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()
    frame_files = sorted((package_dir / "frames").glob("*.jpg"))
    assert len(rows) == 2
    assert len(frame_files) == 2
    assert extracted == frame_files


def test_build_package_uses_collision_free_offset_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recordings_root = tmp_path / "recordings"
    _write_recording(recordings_root, "one", session_id="session")
    monkeypatch.setattr(builder, "_ffprobe_duration_seconds", lambda _path: 2.0)

    def fake_extract(_source: Path, _offset: float, output: Path) -> bool:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"jpeg enough bytes for test")
        return True

    monkeypatch.setattr(builder, "_extract_frame", fake_extract)

    builder.build_package(
        _build_args(
            recordings_root,
            tmp_path / "datasets",
            offsets=(1.04, 1.05),
        )
    )

    frame_files = sorted((tmp_path / "datasets" / "package" / "frames").glob("*.jpg"))
    assert [item.name for item in frame_files] == [
        "one__chunk_0000__001040ms.jpg",
        "one__chunk_0000__001050ms.jpg",
    ]


def test_build_package_refuses_to_overwrite_existing_labels(tmp_path: Path) -> None:
    recordings_root = tmp_path / "recordings"
    _write_recording(recordings_root, "one", session_id="session")
    package_dir = tmp_path / "datasets" / "package"
    package_dir.mkdir(parents=True)
    (package_dir / "labels.csv").write_text("human,label\nkept,yes\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to overwrite existing package"):
        builder.build_package(_build_args(recordings_root, tmp_path / "datasets"))


def test_contact_sheet_requires_pillow_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "Image", None)
    monkeypatch.setattr(builder, "ImageDraw", None)
    monkeypatch.setattr(builder, "ImageFont", None)

    with pytest.raises(RuntimeError, match="Pillow is required"):
        builder._write_contact_sheet(tmp_path, [{"image_relpath": "frame.jpg"}])


def test_replace_existing_clears_stale_frame_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recordings_root = tmp_path / "recordings"
    _write_recording(recordings_root, "one", session_id="session")
    package_dir = tmp_path / "datasets" / "package"
    stale_frame = package_dir / "frames" / "stale.jpg"
    stale_frame.parent.mkdir(parents=True)
    stale_frame.write_bytes(b"old frame")
    monkeypatch.setattr(builder, "_ffprobe_duration_seconds", lambda _path: 2.0)

    def fake_extract(_source: Path, _offset: float, output: Path) -> bool:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"new frame")
        return True

    monkeypatch.setattr(builder, "_extract_frame", fake_extract)

    builder.build_package(
        _build_args(
            recordings_root,
            tmp_path / "datasets",
            max_total_frames=1,
            replace_existing=True,
        )
    )

    frame_files = sorted((package_dir / "frames").glob("*.jpg"))
    assert [item.name for item in frame_files] == ["one__chunk_0000__000500ms.jpg"]
