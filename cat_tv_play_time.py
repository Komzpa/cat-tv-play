"""Timestamp helpers.

The installation's wall clock decides local time: this project reads the
system timezone rather than naming a place, so nothing here says where the
cat lives. Set ``TZ`` to pin a different zone for one process.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

FUTURE_SKEW_TOLERANCE = timedelta(minutes=5)


def local_timezone() -> tzinfo:
    """Return the system's current local timezone."""

    resolved = datetime.now().astimezone().tzinfo
    return resolved if resolved is not None else UTC


def now_local() -> datetime:
    """Return the current instant in the system's local timezone."""

    return datetime.now(local_timezone())


def parse_iso_timestamp(
    value: Any,
    *,
    assume_tz: tzinfo | None = UTC,
    allow_epoch: bool = False,
) -> datetime | None:
    """Parse ISO-8601 input into an aware UTC instant.

    Naive ISO values use ``assume_tz`` (UTC by default), or are rejected when it
    is ``None``. Epoch numbers are accepted only by callers that opt in.
    """

    if allow_epoch and isinstance(value, int | float) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        if assume_tz is None:
            return None
        parsed = parsed.replace(tzinfo=assume_tz)
    return parsed.astimezone(UTC)
