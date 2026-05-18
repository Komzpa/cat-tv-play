"""Review-frame overlays for cat jump clips."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class JumpPeakHold:
    """A frozen crop around a jump peak for the review overlay."""

    t: float
    height_cm: float
    image: Image.Image
    peak_x_px: float
    peak_y_px: float
    label: str | None = None


def crop_jump_peak_hold(
    frame: Image.Image,
    *,
    t: float,
    height_cm: float,
    peak_x_px: float,
    peak_y_px: float,
    crop_width_px: int = 220,
    crop_height_px: int = 160,
    label: str | None = None,
) -> JumpPeakHold:
    """Crop a hold frame around the visual peak point."""

    if crop_width_px <= 0 or crop_height_px <= 0:
        raise ValueError("crop size must be positive")

    source = frame.convert("RGB")
    width, height = source.size
    left = int(round(peak_x_px - crop_width_px / 2))
    top = int(round(peak_y_px - crop_height_px / 2))
    left = max(0, min(left, max(0, width - crop_width_px)))
    top = max(0, min(top, max(0, height - crop_height_px)))
    right = min(width, left + crop_width_px)
    bottom = min(height, top + crop_height_px)

    return JumpPeakHold(
        t=t,
        height_cm=height_cm,
        image=source.crop((left, top, right, bottom)),
        peak_x_px=peak_x_px,
        peak_y_px=peak_y_px,
        label=label,
    )


def update_top_jump_holds(
    holds: Iterable[JumpPeakHold],
    candidate: JumpPeakHold,
    *,
    limit: int = 3,
    min_separation_s: float = 0.7,
) -> list[JumpPeakHold]:
    """Insert a peak hold into the top list, merging nearby frames of one jump."""

    if limit <= 0:
        return []

    merged: list[JumpPeakHold] = []
    inserted = False
    for hold in holds:
        if abs(hold.t - candidate.t) <= min_separation_s:
            merged.append(candidate if candidate.height_cm > hold.height_cm else hold)
            inserted = True
        else:
            merged.append(hold)

    if not inserted:
        merged.append(candidate)

    return sorted(merged, key=lambda hold: (hold.height_cm, -hold.t), reverse=True)[:limit]


def render_top_jump_holds_overlay(
    frame: Image.Image,
    holds: Iterable[JumpPeakHold],
    *,
    panel_width_px: int = 250,
    padding_px: int = 10,
    thumbnail_height_px: int = 125,
) -> Image.Image:
    """Draw top-3 jump peak holds in the right side of the frame."""

    if panel_width_px <= padding_px * 2:
        raise ValueError("panel_width_px must leave room for padding")

    output = frame.convert("RGBA")
    width, height = output.size
    panel_left = max(0, width - panel_width_px)
    panel = Image.new("RGBA", (width - panel_left, height), (0, 0, 0, 130))
    output.alpha_composite(panel, (panel_left, 0))

    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    title = "TOP JUMPS"
    draw.text((panel_left + padding_px, padding_px), title, fill=(255, 255, 255, 255), font=font)

    y = padding_px + 18
    max_thumb_width = max(1, panel_width_px - padding_px * 2)
    for rank, hold in enumerate(sorted(holds, key=lambda item: item.height_cm, reverse=True)[:3], start=1):
        if y >= height - padding_px:
            break

        label = hold.label or f"#{rank} {hold.height_cm:.1f} cm"
        draw.text((panel_left + padding_px, y), label, fill=(255, 255, 255, 255), font=font)
        y += 14

        thumb = _fit_thumbnail(hold.image, max_thumb_width, thumbnail_height_px)
        thumb_x = panel_left + padding_px + (max_thumb_width - thumb.width) // 2
        output.alpha_composite(thumb.convert("RGBA"), (thumb_x, y))
        draw.rectangle(
            (thumb_x, y, thumb_x + thumb.width - 1, y + thumb.height - 1),
            outline=(255, 215, 64, 255),
            width=2,
        )
        y += thumb.height + padding_px

    return output.convert("RGB")


def _fit_thumbnail(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    source = image.convert("RGB")
    width, height = source.size
    scale = min(max_width / width, max_height / height)
    resized = source.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    return resized
