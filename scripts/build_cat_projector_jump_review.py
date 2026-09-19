#!/usr/bin/env python3
"""Build jump-review sheets and run calibrated height measurements.

The scan step is intentionally dumb and local: it turns direct projector-camera
chunks into contact sheets plus frame candidates, then writes a JSONL template
for human top-point annotations. The measure step delegates geometry to
measure_cat_projector_jump_height.py so jump heights keep using the wall-plane
homography instead of a one-off pixel scale.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - pillow is a runtime dependency for render
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_config as site_config  # noqa: E402
from cat_tv_play_atomic_io import read_json_or  # noqa: E402
from scripts.cat_projector_frame_detector import (  # noqa: E402
    DEFAULT_METADATA_PATH as DEFAULT_DETECTOR_METADATA_PATH,
)
from scripts.cat_projector_frame_detector import DEFAULT_MODEL_PATH as DEFAULT_DETECTOR_MODEL_PATH  # noqa: E402
from scripts.cat_projector_frame_detector import (  # noqa: E402
    best_cat_candidate_from_model,
    default_model_available,
    load_model,
)

DEFAULT_RECORDINGS_ROOT = site_config.recordings_root()
DEFAULT_OUTPUT_ROOT = site_config.learning_root() / "jump-review"
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov"}


@dataclass(frozen=True)
class ExtractSpec:
    chunk_index: int
    offset_seconds: float
    label: str


def _safe_slug(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return safe[:160] or "cat-projector-jump-review"


def _latest_recording(recordings_root: Path = DEFAULT_RECORDINGS_ROOT) -> Path:
    manifests = sorted(recordings_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no recording manifests under {recordings_root}")
    return manifests[-1].parent


def _completed_chunks(recording_dir: Path) -> list[Path]:
    chunks: list[Path] = []
    for path in sorted(recording_dir.glob("chunk_*")):
        if ".part." in path.name:
            continue
        if path.suffix.lower() in VIDEO_SUFFIXES:
            chunks.append(path)
    if not chunks:
        raise FileNotFoundError(f"no completed video chunks under {recording_dir}")
    return chunks


def _chunk_index(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"chunk name does not look like chunk_NNNN: {path}") from exc


def _concat_line(path: Path) -> str:
    return "file '" + str(path).replace("'", "'\\''") + "'\n"


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")


def _build_sheet(
    *,
    chunks: list[Path],
    output_path: Path,
    fps: float,
    scale_width: int,
    tile: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = output_path.with_suffix(".concat.txt")
    concat_list.write_text("".join(_concat_line(chunk) for chunk in chunks), encoding="utf-8")
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-vf",
            f"fps={fps},scale={scale_width}:-1,tile={tile}",
            "-frames:v",
            "1",
            str(output_path),
        ]
    )


def _extract_frame(source: Path, offset_seconds: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
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
            str(output_path),
        ]
    )
    if output_path.stat().st_size < 1024:
        raise RuntimeError(f"extracted frame is too small to trust: {output_path}")


def _parse_extract_spec(raw: str) -> ExtractSpec:
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError("--extract must be CHUNK:OFFSET[:LABEL]")
    try:
        chunk_index = int(parts[0])
        offset_seconds = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--extract chunk and offset must be numeric") from exc
    if chunk_index < 0 or offset_seconds < 0:
        raise argparse.ArgumentTypeError("--extract chunk and offset must be non-negative")
    label = parts[2] if len(parts) == 3 and parts[2] else f"chunk_{chunk_index:04d}_{offset_seconds:.1f}s"
    return ExtractSpec(chunk_index=chunk_index, offset_seconds=offset_seconds, label=_safe_slug(label))


def _parse_optional_path(raw: str) -> Path | None:
    raw = raw.strip()
    return Path(raw).expanduser() if raw else None


def _format_bbox_xywh(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(str(int(round(value))) for value in bbox)


def _detector_top_point_row(
    frame_path: Path,
    *,
    detector_model: Path,
    detector_loaded_model: Any,
    detector_loaded_metadata: dict[str, Any],
    detector_min_probability: float,
) -> dict[str, Any]:
    if Image is None:
        raise RuntimeError("Pillow is required for detector top-point proposal")
    image = Image.open(frame_path).convert("RGB")
    prediction = best_cat_candidate_from_model(
        image,
        model=detector_loaded_model,
        metadata=detector_loaded_metadata,
        fallback_model_path=detector_model,
        min_probability=detector_min_probability,
    )
    if prediction.candidate is None:
        return {
            "top_x_px": None,
            "top_y_px": None,
            "measurement_role": "detector_candidate_top",
            "include_in_max": False,
            "confidence": "detector_no_candidate",
            "detector_cat_probability": round(prediction.cat_probability, 4),
            "detector_model_path": prediction.model_path,
            "candidate_bbox_xywh": "",
            "notes": "detector did not accept a cat candidate; fill top_x_px/top_y_px manually if Sher is visible",
        }
    candidate = prediction.candidate
    return {
        "top_x_px": round(candidate.top_x_px, 2),
        "top_y_px": round(candidate.top_y_px, 2),
        "measurement_role": "detector_candidate_top",
        "include_in_max": True,
        "confidence": "detector_candidate_needs_human_review",
        "detector_cat_probability": round(prediction.cat_probability, 4),
        "detector_model_path": prediction.model_path,
        "candidate_bbox_xywh": _format_bbox_xywh(candidate.bbox_xywh),
        "notes": "detector-proposed highest candidate point; verify visually before treating as final jump height",
    }


def scan(args: argparse.Namespace) -> None:
    recording_dir = args.recording_dir.expanduser() if args.recording_dir else _latest_recording(args.recordings_root)
    manifest_path = recording_dir / "manifest.json"
    manifest = read_json_or(manifest_path) if manifest_path.exists() else {}
    chunks = _completed_chunks(recording_dir)
    chunks_by_index = {_chunk_index(path): path for path in chunks}
    output_dir = (
        args.output_dir.expanduser() if args.output_dir else DEFAULT_OUTPUT_ROOT / _safe_slug(recording_dir.name)
    )
    sheets_dir = output_dir / "sheets"
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_chunks = chunks[args.start_chunk :]
    if args.end_chunk is not None:
        selected_chunks = [chunk for chunk in selected_chunks if _chunk_index(chunk) <= args.end_chunk]

    sheet_records: list[dict[str, Any]] = []
    for block_start in range(0, len(selected_chunks), args.sheet_chunk_count):
        block = selected_chunks[block_start : block_start + args.sheet_chunk_count]
        if not block:
            continue
        first_index = _chunk_index(block[0])
        last_index = _chunk_index(block[-1])
        sheet_path = sheets_dir / f"chunks_{first_index:04d}_{last_index:04d}.jpg"
        _build_sheet(
            chunks=block, output_path=sheet_path, fps=args.sheet_fps, scale_width=args.sheet_width, tile=args.tile
        )
        sheet_records.append(
            {
                "sheet_path": str(sheet_path),
                "first_chunk": first_index,
                "last_chunk": last_index,
                "chunk_count": len(block),
            }
        )

    observation_rows: list[dict[str, Any]] = []
    detector_model = args.detector_model
    detector_loaded_model = None
    detector_loaded_metadata = None
    if args.propose_detector_top_points:
        if detector_model is None and default_model_available():
            detector_model = DEFAULT_DETECTOR_MODEL_PATH
        if detector_model is None:
            raise FileNotFoundError(
                f"detector model is missing: {DEFAULT_DETECTOR_MODEL_PATH}; "
                "pass --detector-model or disable --propose-detector-top-points"
            )
        detector_loaded_model, detector_loaded_metadata = load_model(
            detector_model,
            args.detector_metadata,
        )
    for spec in args.extract:
        source = chunks_by_index.get(spec.chunk_index)
        if source is None:
            raise FileNotFoundError(f"chunk_{spec.chunk_index:04d} is not present in {recording_dir}")
        frame_path = (
            frames_dir / f"chunk_{spec.chunk_index:04d}_{int(round(spec.offset_seconds * 1000)):06d}ms_{spec.label}.jpg"
        )
        _extract_frame(source, spec.offset_seconds, frame_path)
        proposal: dict[str, Any] = {
            "top_x_px": None,
            "top_y_px": None,
            "measurement_role": "visible_top",
            "include_in_max": True,
            "confidence": "todo",
            "notes": (
                "fill top_x_px/top_y_px with the highest visible paw/head/body point; "
                "set include_in_max=false for blur, shadow, or empty-wall points"
            ),
        }
        if args.propose_detector_top_points:
            proposal = _detector_top_point_row(
                frame_path,
                detector_model=detector_model,
                detector_loaded_model=detector_loaded_model,
                detector_loaded_metadata=detector_loaded_metadata,
                detector_min_probability=args.detector_min_probability,
            )
        observation_rows.append(
            {
                "frame_id": str(frame_path.relative_to(output_dir)),
                **proposal,
                "source": str(source),
                "source_offset_seconds": spec.offset_seconds,
            }
        )

    observations_template = output_dir / "observations_template.jsonl"
    with observations_template.open("w", encoding="utf-8") as handle:
        for row in observation_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "kind": "cat_projector_jump_review_scan_v1",
        "recording_dir": str(recording_dir),
        "ha_session_id": manifest.get("ha_session_id"),
        "video_url": manifest.get("video_url"),
        "output_dir": str(output_dir),
        "sheets": sheet_records,
        "observations_template": str(observations_template),
        "extracted_frame_count": len(observation_rows),
    }
    (output_dir / "scan_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def measure(args: argparse.Namespace) -> None:
    observations = args.observations.expanduser()
    output_json = args.output_json.expanduser()
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "measure_cat_projector_jump_height.py"),
        "--calibration",
        str(args.calibration.expanduser()),
        "--observations",
        str(observations),
        "--output-json",
        str(output_json),
    ]
    if args.output_csv:
        command.extend(["--output-csv", str(args.output_csv.expanduser())])
    result = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:] or result.stdout[-1200:])
    print(result.stdout.strip())


def _text_length(draw: Any, text: str, font: Any) -> float:
    try:
        return float(draw.textlength(text, font=font))
    except Exception:
        return float(len(text) * 8)


def _load_fonts() -> tuple[Any, Any, Any]:
    if ImageFont is None:
        return None, None, None
    try:
        return (
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38),
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25),
            ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18),
        )
    except Exception:
        return None, None, None


def _resolve_frame_path(review_dir: Path, frame_id: str) -> Path:
    frame_path = Path(frame_id)
    if frame_path.is_absolute():
        return frame_path
    return review_dir / frame_path


def render(args: argparse.Namespace) -> None:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for render; install python3-pil/Pillow")
    measurements = read_json_or(args.measurements.expanduser())
    review_dir = args.review_dir.expanduser() if args.review_dir else args.measurements.expanduser().parent
    records = list(measurements.get("records") or [])
    included = [record for record in records if record.get("include_in_max") is True]
    if not included:
        raise RuntimeError("measurements contain no included visible-top records")
    maximum = max(included, key=lambda record: float(record["cat_top_height_mm"]))
    frame_path = _resolve_frame_path(review_dir, str(maximum["frame_id"]))
    output_path = args.output.expanduser() if args.output else review_dir / "sher_jump_annotated_visible_max.jpg"

    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_big, font_med, font_small = _load_fonts()

    draw.rectangle((0, 0, img.width, 84), fill=(12, 16, 20))
    draw.text((24, 14), "Sher jump height - visible top", fill=(245, 248, 250), font=font_big)
    draw.text(
        (26, 56), "calibrated wall homography; excluded blur is not counted", fill=(180, 196, 210), font=font_small
    )

    x = float(maximum["cat_top_x_px"])
    y = float(maximum["cat_top_y_px"])
    height_cm = float(maximum["cat_top_height_cm"])
    color = (255, 210, 60)
    radius = 10
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
    label_left = min(img.width - 468, int(x + 135))
    label_top = max(96, int(y - 26))
    draw.line((x + radius, y, label_left - 12, label_top + 25), fill=color, width=3)
    draw.rounded_rectangle(
        (label_left, label_top, img.width - 28, label_top + 52), radius=8, fill=(12, 16, 20), outline=color, width=2
    )
    draw.text((label_left + 16, label_top + 12), f"visible top {height_cm:.1f} cm", fill=(245, 248, 250), font=font_med)

    excluded = [record for record in records if record.get("include_in_max") is not True]
    if excluded:
        best_excluded = max(excluded, key=lambda record: float(record["cat_top_height_mm"]))
        excluded_text = (
            f"Excluded from max: {best_excluded['cat_top_height_cm']} cm "
            f"({best_excluded.get('exclusion_reason') or best_excluded.get('measurement_role')})."
        )
        draw.text((24, img.height - 60), excluded_text[:150], fill=(230, 230, 230), font=font_small)

    footer = (
        f"source: {Path(str(maximum.get('source') or '')).name}   corrected visible measurement: {height_cm:.1f} cm"
    )
    width = _text_length(draw, footer, font_small)
    draw.rectangle((0, img.height - 36, img.width, img.height), fill=(12, 16, 20))
    draw.text((max(18, int((img.width - width) // 2)), img.height - 28), footer, fill=(210, 220, 230), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=94)
    print(str(output_path))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Build sheets and optional frame annotation templates.")
    scan_parser.add_argument("--recording-dir", type=Path)
    scan_parser.add_argument("--recordings-root", type=Path, default=DEFAULT_RECORDINGS_ROOT)
    scan_parser.add_argument("--output-dir", type=Path)
    scan_parser.add_argument("--start-chunk", type=int, default=0)
    scan_parser.add_argument("--end-chunk", type=int)
    scan_parser.add_argument("--sheet-chunk-count", type=int, default=30)
    scan_parser.add_argument("--sheet-fps", type=float, default=1.0)
    scan_parser.add_argument("--sheet-width", type=int, default=256)
    scan_parser.add_argument("--tile", default="10x15")
    scan_parser.add_argument("--extract", type=_parse_extract_spec, action="append", default=[])
    scan_parser.add_argument("--propose-detector-top-points", action="store_true")
    scan_parser.add_argument("--detector-model", type=_parse_optional_path)
    scan_parser.add_argument("--detector-metadata", type=_parse_optional_path, default=DEFAULT_DETECTOR_METADATA_PATH)
    scan_parser.add_argument("--detector-min-probability", type=float, default=0.30)
    scan_parser.set_defaults(func=scan)

    measure_parser = subparsers.add_parser("measure", help="Measure filled top-point observations with calibration.")
    measure_parser.add_argument("--calibration", type=Path, required=True)
    measure_parser.add_argument("--observations", type=Path, required=True)
    measure_parser.add_argument("--output-json", type=Path, required=True)
    measure_parser.add_argument("--output-csv", type=Path)
    measure_parser.set_defaults(func=measure)

    render_parser = subparsers.add_parser("render", help="Render a clean annotated image from included measurements.")
    render_parser.add_argument("--measurements", type=Path, required=True)
    render_parser.add_argument("--review-dir", type=Path)
    render_parser.add_argument("--output", type=Path)
    render_parser.set_defaults(func=render)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.func(args)


if __name__ == "__main__":
    main()
