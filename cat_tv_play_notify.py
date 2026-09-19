#!/usr/bin/env python3
"""Outbound clip notifications.

This project does not own a messaging transport. An installation names one in
``notify.command`` (see ``deploy/private.example.yml``): an argv list where the
literal ``{text}`` is replaced by the caption and the media file paths are
appended as trailing arguments. With no command configured, delivery is a
recorded no-op, so a fresh checkout never messages anybody.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_config as site_config  # noqa: E402

DEFAULT_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class DeliveryResult:
    delivered: bool
    audited: bool
    source: str
    delivery_class: str
    reason: str
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def build_command(argv_template: list[str], text: str, media: list[Path | str]) -> list[str]:
    command = [part.replace("{text}", text) for part in argv_template]
    command.extend(str(item) for item in media)
    return command


def deliver_notification(
    content: str | None,
    *,
    source: str,
    delivery_class: str = "user_facing",
    reason: str = "",
    media: list[Path | str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> DeliveryResult:
    """Send one notification through the configured command."""

    text = str(content or "").strip()
    items = list(media or [])
    argv_template = site_config.notify_command()
    if not argv_template:
        return DeliveryResult(
            delivered=False,
            audited=True,
            source=source,
            delivery_class=delivery_class,
            reason="notify_command_not_configured",
        )

    command = build_command(argv_template, text, items)
    try:
        completed = subprocess.run(  # noqa: S603 - argv comes from the operator's own config
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return DeliveryResult(
            delivered=False,
            audited=True,
            source=source,
            delivery_class=delivery_class,
            reason=f"notify_command_failed: {type(exc).__name__}",
            returncode=1,
            stderr=str(exc),
        )

    return DeliveryResult(
        delivered=completed.returncode == 0,
        audited=True,
        source=source,
        delivery_class=delivery_class,
        reason=reason or "notify_command",
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
