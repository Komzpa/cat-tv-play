#!/usr/bin/env python3
"""Build Telegram-ready clips from reviewed cat-projector labeling rows.

This prepares a local outbox package only. The actual send step can use the
manual Telegram sender once the operator chooses the chat and reviews captions.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_config as site_config  # noqa: E402
from cat_tv_play_time import now_local  # noqa: E402

DEFAULT_DATASETS_ROOT = site_config.datasets_root()
DEFAULT_OUTBOX_ROOT = site_config.learning_root() / "telegram-clips"


@dataclass(frozen=True)
class ClipCandidate:
    image_relpath: str
    source_chunk: Path
    source_offset_seconds: float
    ha_session_id: str
    frigate_event_id: str
    video_slug: str
    confidence: str
    notes: str


def _truthy_label(value: str) -> bool:
    return value.strip().lower() in {"yes", "true", "1", "playing", "play"}


def _float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_play_candidates(labels_csv: Path, *, include_unsure: bool = False) -> list[ClipCandidate]:
    candidates: list[ClipCandidate] = []
    with labels_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            playing = str(row.get("label_cat_playing") or "")
            present = str(row.get("label_cat_present") or "")
            if not _truthy_label(playing) and (
                not include_unsure or playing.strip().lower() != "unsure" or not _truthy_label(present)
            ):
                continue
            source_chunk = str(row.get("source_chunk") or "").strip()
            if not source_chunk:
                continue
            ha_session_id = str(row.get("ha_session_id") or "").strip()
            video_slug = str(row.get("video_slug") or "").strip()
            candidates.append(
                ClipCandidate(
                    image_relpath=str(row.get("image_relpath") or "").strip(),
                    source_chunk=Path(source_chunk).expanduser(),
                    source_offset_seconds=_float(str(row.get("source_offset_seconds") or "0")),
                    ha_session_id=ha_session_id,
                    frigate_event_id=str(row.get("frigate_event_id") or "").strip(),
                    video_slug=video_slug,
                    confidence=str(row.get("confidence") or "").strip(),
                    notes=str(row.get("notes") or "").strip(),
                )
            )
    return candidates


def _clip_name(candidate: ClipCandidate, index: int) -> str:
    slug = _display_video_slug(candidate).replace(".mp4", "") or candidate.ha_session_id or "cat-projector"
    safe_slug = "".join(char if char.isalnum() or char in "._-" else "_" for char in slug)[:96]
    offset_ms = int(round(candidate.source_offset_seconds * 1000))
    return f"{index:03d}_{safe_slug}_{offset_ms:06d}ms.mp4"


def _session_slug(candidate: ClipCandidate) -> str:
    if "-" not in candidate.ha_session_id:
        return ""
    prefix, slug = candidate.ha_session_id.split("-", 1)
    if len(prefix) == 15 and prefix[8] == "T" and slug:
        return slug
    return ""


def _display_video_slug(candidate: ClipCandidate) -> str:
    session_slug = _session_slug(candidate)
    if session_slug and session_slug not in candidate.video_slug:
        return f"{session_slug}.mp4"
    return candidate.video_slug


def _caption(candidate: ClipCandidate, clip_path: Path) -> str:
    bits = ["Sher projector play clip"]
    display_video_slug = _display_video_slug(candidate)
    if display_video_slug:
        bits.append(f"video: {display_video_slug}")
    if candidate.video_slug and candidate.video_slug != display_video_slug:
        bits.append(f"recorded video_url field was stale: {candidate.video_slug}")
    if candidate.ha_session_id:
        bits.append(f"session: {candidate.ha_session_id}")
    bits.append(f"camera time: +{candidate.source_offset_seconds:.1f}s in {candidate.source_chunk.name}")
    if candidate.confidence:
        bits.append(f"label confidence: {candidate.confidence}")
    if candidate.notes:
        bits.append(f"review note: {candidate.notes}")
    bits.append(f"clip: {clip_path.name}")
    return "\n".join(bits)


def build_clip(
    candidate: ClipCandidate,
    *,
    output_path: Path,
    preroll_seconds: float,
    duration_seconds: float,
) -> None:
    start_seconds = max(0.0, candidate.source_offset_seconds - preroll_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(candidate.source_chunk),
        "-t",
        f"{duration_seconds:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 1024:
        raise RuntimeError(f"failed to build clip {output_path}: {result.stderr[-400:]}")


def build_outbox(
    *,
    labels_csv: Path,
    output_dir: Path,
    max_clips: int,
    preroll_seconds: float,
    duration_seconds: float,
    include_unsure: bool,
) -> dict[str, object]:
    candidates = load_play_candidates(labels_csv, include_unsure=include_unsure)[:max_clips]
    if not candidates:
        raise RuntimeError("no reviewed cat-playing rows found in labels.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        clip_path = output_dir / _clip_name(candidate, index)
        build_clip(candidate, output_path=clip_path, preroll_seconds=preroll_seconds, duration_seconds=duration_seconds)
        caption = _caption(candidate, clip_path)
        caption_path = clip_path.with_suffix(".caption.txt")
        caption_path.write_text(caption + "\n", encoding="utf-8")
        clips.append(
            {
                "clip_path": str(clip_path),
                "caption_path": str(caption_path),
                "caption": caption,
                "source_chunk": str(candidate.source_chunk),
                "source_offset_seconds": candidate.source_offset_seconds,
                "ha_session_id": candidate.ha_session_id,
                "frigate_event_id": candidate.frigate_event_id,
                "video_slug": candidate.video_slug,
            }
        )

    manifest = {
        "kind": "cat_projector_telegram_clip_outbox_v1",
        "created_at": now_local().isoformat(timespec="seconds"),
        "labels_csv": str(labels_csv),
        "clip_count": len(clips),
        "clips": clips,
        "send_hint": "Review captions, then send manually with src/send_telegram.py --text/--attachment.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Dataset package directory containing labels.csv.")
    parser.add_argument("--labels-csv", type=Path, help="Explicit labels.csv path.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument("--preroll-seconds", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    parser.add_argument("--include-unsure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_csv = args.labels_csv
    if labels_csv is None:
        if args.dataset is None:
            raise SystemExit("Use --dataset or --labels-csv")
        labels_csv = args.dataset / "labels.csv"
    output_dir = args.output_dir
    if output_dir is None:
        dataset_name = (args.dataset or labels_csv.parent).name
        output_dir = DEFAULT_OUTBOX_ROOT / dataset_name
    manifest = build_outbox(
        labels_csv=labels_csv,
        output_dir=output_dir,
        max_clips=args.max_clips,
        preroll_seconds=args.preroll_seconds,
        duration_seconds=args.duration_seconds,
        include_unsure=args.include_unsure,
    )
    print(f"built {manifest['clip_count']} Telegram-ready cat projector clips in {output_dir}")


if __name__ == "__main__":
    main()
