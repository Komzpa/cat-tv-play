"""Notification retry policy for Cat TV jump-highlight delivery.

The integration owns the behavior decision; house-specific glue may still own
the actual Telegram delivery transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

TEMPORARY_LIVE_SKIP_ACTIONS = frozenset({"camera_human_detected", "human_check_unavailable"})


@dataclass(frozen=True)
class LiveNotificationDecision:
    action: str
    retry_allowed: bool
    age_seconds: float | None = None
    previous_action: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "telegram_live_action": self.action,
            "retry_allowed": self.retry_allowed,
        }
        if self.age_seconds is not None:
            payload["age_seconds"] = round(float(self.age_seconds), 1)
        if self.previous_action:
            payload["previous_telegram_live_action"] = self.previous_action
        return payload


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def decide_live_notification_from_marker(
    marker: dict[str, Any],
    *,
    now: datetime,
    sent_cooldown_seconds: float,
    scan_cooldown_seconds: float,
) -> LiveNotificationDecision | None:
    """Return a cooldown decision for an existing live notification marker.

    Human-presence skips are intentionally not terminal: the next scan after
    the scan cooldown may find a later jump window where the person left the
    projector camera frame.
    """

    action = str(marker.get("telegram_live_action") or ("sent" if marker.get("clip") else ""))
    last_sent = _parse_dt(marker.get("last_sent_at") or marker.get("sent_at"))
    last_scan = _parse_dt(marker.get("last_scan_at"))

    if action in TEMPORARY_LIVE_SKIP_ACTIONS and last_scan is not None:
        age_seconds = (now - last_scan).total_seconds()
        if age_seconds < scan_cooldown_seconds:
            return LiveNotificationDecision(
                action="scan_cooldown",
                retry_allowed=False,
                age_seconds=age_seconds,
                previous_action=action,
            )
        return None

    if action == "sent" and last_sent is not None:
        age_seconds = (now - last_sent).total_seconds()
        if age_seconds < sent_cooldown_seconds:
            return LiveNotificationDecision(action="cooldown", retry_allowed=False, age_seconds=age_seconds)
        return None

    if action in {"no_curated_highlight", "no_jump_highlight"} and last_scan is not None:
        age_seconds = (now - last_scan).total_seconds()
        if age_seconds < scan_cooldown_seconds:
            return LiveNotificationDecision(action="scan_cooldown", retry_allowed=False, age_seconds=age_seconds)
        return None

    return None
