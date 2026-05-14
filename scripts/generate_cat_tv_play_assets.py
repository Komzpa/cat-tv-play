#!/usr/bin/env python3
"""Generate simple Cat TV Play test assets.

This script is optional. It creates local MP4 stimuli that can be copied to
Home Assistant's `/config/www/cat-tv/` directory. The integration itself does
not depend on Pillow or ffmpeg.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw

WIDTH = 1280
HEIGHT = 720
FPS = 20
SEGMENT_SECONDS = 80
LOOP_SECONDS = 30 * 60

FrameRenderer = Callable[[int], Image.Image]


def _ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 0.5 - 0.5 * math.cos(value * math.pi)


def _draw_moth(draw: ImageDraw.ImageDraw, x: float, y: float, scale: float = 1.0) -> None:
    radius = 11 * scale
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(12, 12, 12))
    draw.ellipse((x - radius * 2.8, y - radius * 1.2, x - radius * 0.3, y + radius * 1.0), fill=(18, 18, 18))
    draw.ellipse((x + radius * 0.3, y - radius * 1.2, x + radius * 2.8, y + radius * 1.0), fill=(18, 18, 18))


def render_jump_ladder(frame_index: int) -> Image.Image:
    """Render a left-side vertical chase that climbs in reachable steps."""

    t = frame_index / FPS
    image = Image.new("RGB", (WIDTH, HEIGHT), (238, 235, 220))
    draw = ImageDraw.Draw(image)

    # Source pixels are not physical centimeters. These lanes are visual
    # waypoints; real height is measured by Cat TV Play calibration afterwards.
    steps = [
        (250, 500),
        (270, 430),
        (245, 360),
        (275, 285),
        (235, 215),
        (260, 160),
    ]
    cycle = 12.0
    local = t % cycle
    step_index = min(len(steps) - 2, int(local // 2.0))
    phase = (local % 2.0) / 2.0

    start = steps[step_index]
    end = steps[step_index + 1]
    if phase < 0.45:
        p = _ease(phase / 0.45)
    elif phase < 0.78:
        p = 1.0
    else:
        p = _ease((phase - 0.78) / 0.22)

    x = start[0] + (end[0] - start[0]) * p + 8 * math.sin(t * 10.0)
    y = start[1] + (end[1] - start[1]) * p + 4 * math.sin(t * 13.0)
    _draw_moth(draw, x, y, 1.15)

    if local > 10.5:
        # Give a low catch moment before the loop restarts.
        _draw_moth(draw, 245 + 20 * math.sin(t * 8), 565, 1.3)

    return image


def write_video(path: Path, renderer: FrameRenderer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        assert process.stdin is not None
        for frame_index in range(FPS * SEGMENT_SECONDS):
            process.stdin.write(renderer(frame_index).tobytes())
        process.stdin.close()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def loop_video(segment: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "30",
            "-i",
            str(segment),
            "-t",
            str(LOOP_SECONDS),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/cat-tv-play-assets"))
    args = parser.parse_args()

    segment = args.output_dir / "segments" / "cat-tv-jump-ladder.mp4"
    output = args.output_dir / "cat-tv-jump-ladder.mp4"
    write_video(segment, render_jump_ladder)
    loop_video(segment, output)
    print(output)


if __name__ == "__main__":
    main()

