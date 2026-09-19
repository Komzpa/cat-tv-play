#!/usr/bin/env python3
"""Build a local labeling package from Cat TV projector-camera recordings.

The package is intentionally local-first: it extracts JPEG candidates plus
metadata for human labels, without uploading frames to Frigate+ or any other
cloud annotator.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - pillow is a runtime dependency for sheets
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_config as site_config  # noqa: E402
from cat_tv_play_atomic_io import read_json_or  # noqa: E402

DEFAULT_RECORDINGS_ROOT = site_config.recordings_root()
DEFAULT_DATASETS_ROOT = site_config.datasets_root()
DEFAULT_OFFSETS = (0.7, 2.5, 4.2)
#: Session ids whose frames are known false negatives for this corpus.
DEFAULT_FALSE_NEGATIVE_SESSIONS = tuple(site_config.get("corpus.false_negative_sessions") or ())
VIDEO_CHUNK_SUFFIXES = {".mp4", ".mkv", ".mov"}

LABEL_FIELDNAMES = [
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


@dataclass(frozen=True)
class Recording:
    path: Path
    manifest: dict[str, Any]
    chunks: tuple[Path, ...]
    priority: int
    candidate_reason: str

    @property
    def ha_session_id(self) -> str:
        return str(self.manifest.get("ha_session_id") or "")

    @property
    def frigate_event_id(self) -> str:
        return str(self.manifest.get("frigate_event_id") or "")

    @property
    def video_slug(self) -> str:
        video_url = str(self.manifest.get("video_url") or "")
        if "/" in video_url:
            return video_url.rsplit("/", 1)[-1]
        return video_url


def _safe_slug(value: str) -> str:
    value = value.strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:140]


def _parse_offsets(raw_offsets: str) -> tuple[float, ...]:
    offsets: list[float] = []
    for raw in raw_offsets.split(","):
        raw = raw.strip()
        if not raw:
            continue
        offset = float(raw)
        if offset < 0:
            raise argparse.ArgumentTypeError("frame offsets must be non-negative")
        offsets.append(offset)
    if not offsets:
        raise argparse.ArgumentTypeError("at least one frame offset is required")
    return tuple(offsets)


def _offset_slug(offset_seconds: float) -> str:
    return f"{int(round(offset_seconds * 1000)):06d}ms"


def _discover_completed_chunks(recording_dir: Path) -> tuple[Path, ...]:
    chunks: list[Path] = []
    for path in sorted(recording_dir.glob("chunk_*")):
        if ".part." in path.name:
            continue
        if path.suffix.lower() not in VIDEO_CHUNK_SUFFIXES:
            continue
        chunks.append(path)
    return tuple(chunks)


def _ffprobe_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return 0.0


def _extract_frame(source: Path, offset_seconds: float, output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{offset_seconds:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    return result.returncode == 0 and output.exists() and output.stat().st_size > 1024


def _recording_reason(
    *,
    recording_dir: Path,
    manifest: dict[str, Any],
    false_negative_sessions: Iterable[str],
) -> tuple[int, str]:
    session_id = str(manifest.get("ha_session_id") or "")
    name = recording_dir.name
    for needle in false_negative_sessions:
        if needle and (needle in session_id or needle in name):
            return 0, "user_observed_false_negative_window"
    if "cat-sher-pounce-lanes" in session_id or "cat-sher-pounce-lanes" in name:
        return 1, "current_local_experiment"
    if "cats-tv-real-bugs" in session_id or "cats-tv-real-bugs" in name:
        return 2, "rejected_sparse_projection_comparison"
    return 3, "background_cat_tv_recording"


def discover_recordings(
    recordings_root: Path,
    *,
    false_negative_sessions: Iterable[str] = DEFAULT_FALSE_NEGATIVE_SESSIONS,
    session_patterns: Iterable[str] = (),
    max_recordings_per_session: int = 0,
) -> list[Recording]:
    patterns = tuple(pattern for pattern in session_patterns if pattern)
    recordings: list[Recording] = []
    for manifest_path in sorted(recordings_root.glob("*/manifest.json")):
        recording_dir = manifest_path.parent
        manifest = read_json_or(manifest_path)
        session_id = str(manifest.get("ha_session_id") or "")
        haystack = f"{recording_dir.name}\n{session_id}\n{manifest.get('video_url') or ''}"
        if patterns and not any(pattern in haystack for pattern in patterns):
            continue
        chunks = _discover_completed_chunks(recording_dir)
        if not chunks:
            continue
        priority, reason = _recording_reason(
            recording_dir=recording_dir,
            manifest=manifest,
            false_negative_sessions=false_negative_sessions,
        )
        recordings.append(
            Recording(
                path=recording_dir,
                manifest=manifest,
                chunks=chunks,
                priority=priority,
                candidate_reason=reason,
            )
        )
    recordings.sort(key=lambda item: (item.priority, item.path.name))
    if max_recordings_per_session > 0:
        kept: list[Recording] = []
        counts: dict[str, int] = defaultdict(int)
        for recording in recordings:
            key = recording.ha_session_id or recording.path.name
            if counts[key] >= max_recordings_per_session:
                continue
            kept.append(recording)
            counts[key] += 1
        recordings = kept
    return recordings


def _iter_frame_candidates(
    recording: Recording,
    *,
    offsets: tuple[float, ...],
    max_frames_per_recording: int,
) -> Iterable[tuple[Path, float]]:
    yielded = 0
    for chunk in recording.chunks:
        duration = _ffprobe_duration_seconds(chunk)
        if duration <= 0:
            continue
        for offset in offsets:
            if offset >= duration:
                continue
            yield chunk, offset
            yielded += 1
            if yielded >= max_frames_per_recording:
                return


def write_label_schema(path: Path) -> None:
    path.write_text(
        """# Cat Projector Local Label Schema

Fill `labels.csv` in place. Keep ambiguous rows; use `confidence=low` and notes.

- `label_cat_present`: `yes`, `no`, or `unsure`
- `label_cat_playing`: `yes`, `watching`, `no`, or `unsure`
- `review_status`: `blank`, `machine_prefilled_needs_human_review`, or `human_reviewed`
- `candidate_bbox_xywh`: optional detector/scorer candidate box `x,y,w,h` in
  image pixels. This can be a false-positive box even when Sher is visible
  elsewhere in the frame.
- `label_candidate_is_cat`: `yes`, `no`, or `unsure` for the candidate box.
  Use this for retraining the automatic top-point detector.
- `negative_reason`: for `label_cat_present=no`, for example
  `projected_prey`, `static_furniture`, `shadow`, `screen_edge`, or
  `detector_false_positive_not_cat`
- `bbox_xywh`: optional `x,y,w,h` in image pixels around Sher. Leave blank for `no`.
- `occlusion`: optional short note, for example `edge`, `shadow`, `blur`, `partial`.
- `confidence`: `high`, `medium`, or `low`

For the first local model, the most useful split is:

- positive: `label_cat_present=yes` with Sher visible anywhere in the projector-camera frame
- strong positive for session logic: `label_cat_playing=yes` or `watching`
- negative: `label_cat_present=no`
- hard negative: a previous detector/scorer fired on a non-cat candidate;
  store its box in `candidate_bbox_xywh` and set
  `label_candidate_is_cat=no`. This is different from `label_cat_present=no`:
  a frame may contain Sher while the candidate box is still projected prey.
  Keep `review_status=machine_prefilled_needs_human_review` until a human
  confirms it, then set `review_status=human_reviewed`.
- hold out `unsure` until a second pass

No cloud upload is required. This package is meant for a local YOLO/state
classifier comparison after labels exist.
""",
        encoding="utf-8",
    )


def write_local_model_plan(path: Path) -> None:
    path.write_text(
        """# Local Cat Detector Plan

Chosen path: local model, no Frigate+ upload.

1. Label this package first: cat present, cat playing/watching, optional bbox.
2. Train a small local state classifier as the first decision gate:
   `empty`, `cat_visible_idle`, `cat_engaged`.
3. Compare it against the current Frigate `cat` detector on the same held-out
   frames. The live automation should keep treating detector loss as advisory,
   not as a projector-stop command.
4. If bboxes are consistently available and the state classifier is too coarse,
   train/export a local YOLO/ONNX detector and evaluate it before wiring Frigate
   to it.

Frigate+ fine-tune is deliberately not used here because it uploads household
camera frames to a third-party cloud.
""",
        encoding="utf-8",
    )


def build_package(args: argparse.Namespace) -> int:
    recordings_root = args.recordings_root.expanduser()
    datasets_root = args.output_root.expanduser()
    package_id = args.package_id or f"cat-projector-local-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}"
    package_dir = datasets_root / _safe_slug(package_id)
    frames_dir = package_dir / "frames"
    if package_dir.exists() and not args.replace_existing and any(package_dir.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite existing package {package_dir}; "
            "use a new --package-id or pass --replace-existing before human labels exist"
        )
    if package_dir.exists() and args.replace_existing:
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(
        recordings_root,
        false_negative_sessions=args.false_negative_session,
        session_patterns=args.session_pattern,
        max_recordings_per_session=args.max_recordings_per_session,
    )
    if args.max_recordings:
        recordings = recordings[: args.max_recordings]
    if not recordings:
        raise RuntimeError(f"no usable recordings found under {recordings_root}")

    rows: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    for recording in recordings:
        rec_slug = _safe_slug(recording.path.name)
        extracted_for_recording = 0
        for chunk, offset in _iter_frame_candidates(
            recording,
            offsets=args.offsets,
            max_frames_per_recording=args.max_frames_per_recording,
        ):
            if len(rows) >= args.max_total_frames:
                break
            image_name = f"{rec_slug}__{chunk.stem}__{_offset_slug(offset)}.jpg"
            image_path = frames_dir / image_name
            if not _extract_frame(chunk, offset, image_path):
                continue
            extracted_for_recording += 1
            relpath = image_path.relative_to(package_dir).as_posix()
            row = {
                "image_relpath": relpath,
                "label_cat_present": "",
                "label_cat_playing": "",
                "review_status": "",
                "candidate_bbox_xywh": "",
                "label_candidate_is_cat": "",
                "negative_reason": "",
                "bbox_xywh": "",
                "occlusion": "",
                "confidence": "",
                "notes": "",
                "source_recording_dir": str(recording.path),
                "source_chunk": str(chunk),
                "source_offset_seconds": f"{offset:.3f}",
                "ha_session_id": recording.ha_session_id,
                "frigate_event_id": recording.frigate_event_id,
                "video_slug": recording.video_slug,
                "candidate_reason": recording.candidate_reason,
            }
            rows.append(row)
            frame_records.append(dict(row))
        if extracted_for_recording == 0:
            print(f"warning: no frames extracted from {recording.path}", file=sys.stderr)
        if len(rows) >= args.max_total_frames:
            break
    if not rows:
        raise RuntimeError(
            f"no frames were extracted from {len(recordings)} candidate recordings under {recordings_root}"
        )

    labels_path = package_dir / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frame_count": len(rows),
        "label_schema": "cat_projector_local_v1",
        "labels_csv": str(labels_path),
        "local_only": True,
        "no_cloud_upload": True,
        "offsets_seconds": list(args.offsets),
        "package_dir": str(package_dir),
        "recordings_root": str(recordings_root),
        "recordings": [
            {
                "candidate_reason": recording.candidate_reason,
                "chunk_count": len(recording.chunks),
                "ha_session_id": recording.ha_session_id,
                "path": str(recording.path),
                "priority": recording.priority,
                "video_slug": recording.video_slug,
            }
            for recording in recordings
        ],
    }
    (package_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (package_dir / "frames.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in frame_records),
        encoding="utf-8",
    )
    write_label_schema(package_dir / "LABEL_SCHEMA.md")
    write_local_model_plan(package_dir / "LOCAL_MODEL_PLAN.md")
    _write_contact_sheet(package_dir, frame_records[: args.contact_sheet_frames])

    print(
        json.dumps(
            {
                "frame_count": len(rows),
                "labels_csv": str(labels_path),
                "manifest": str(package_dir / "manifest.json"),
                "package_dir": str(package_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _write_contact_sheet(package_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required to write contact_sheet.jpg")
    thumbs: list[tuple[Image.Image, str]] = []
    for row in rows:
        image_path = package_dir / str(row["image_relpath"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((240, 135))
        caption = f"{row['candidate_reason']} | {Path(str(row['source_chunk'])).name}"
        thumbs.append((image.copy(), caption[:60]))
    if not thumbs:
        return
    columns = 4
    cell_w, cell_h = 260, 180
    rows_count = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows_count * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (thumb, caption) in enumerate(thumbs):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        sheet.paste(thumb, (x + 10, y + 10))
        draw.text((x + 10, y + 150), caption, fill=(0, 0, 0))
    sheet.save(package_dir / "contact_sheet.jpg", quality=92)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", type=Path, default=DEFAULT_RECORDINGS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--package-id", default="")
    parser.add_argument("--session-pattern", action="append", default=[])
    parser.add_argument(
        "--false-negative-session",
        action="append",
        default=list(DEFAULT_FALSE_NEGATIVE_SESSIONS),
        help="Session/name substring known to contain a cat-present false negative window.",
    )
    parser.add_argument("--offsets", type=_parse_offsets, default=DEFAULT_OFFSETS)
    parser.add_argument("--max-recordings", type=int, default=8)
    parser.add_argument("--max-recordings-per-session", type=int, default=4)
    parser.add_argument("--max-frames-per-recording", type=int, default=18)
    parser.add_argument("--max-total-frames", type=int, default=120)
    parser.add_argument("--contact-sheet-frames", type=int, default=48)
    parser.add_argument("--replace-existing", action="store_true")
    parser.set_defaults(func=build_package)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
