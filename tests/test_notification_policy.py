import sys
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "cat_tv_play" / "notification_policy.py"
SPEC = spec_from_file_location("cat_tv_play_notification_policy", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
notification_policy = module_from_spec(SPEC)
sys.modules[SPEC.name] = notification_policy
SPEC.loader.exec_module(notification_policy)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def test_camera_human_live_skip_rescans_after_scan_cooldown() -> None:
    marker = {
        "telegram_live_action": "camera_human_detected",
        "last_scan_at": "2026-05-26T01:34:03+04:00",
    }

    decision = notification_policy.decide_live_notification_from_marker(
        marker,
        now=_dt("2026-05-26T01:35:03+04:00"),
        sent_cooldown_seconds=300,
        scan_cooldown_seconds=180,
    )

    assert decision is not None
    assert decision.to_payload() == {
        "age_seconds": 60.0,
        "previous_telegram_live_action": "camera_human_detected",
        "retry_allowed": False,
        "telegram_live_action": "scan_cooldown",
    }
    assert (
        notification_policy.decide_live_notification_from_marker(
            marker,
            now=_dt("2026-05-26T01:38:03+04:00"),
            sent_cooldown_seconds=300,
            scan_cooldown_seconds=180,
        )
        is None
    )


def test_sent_live_marker_keeps_delivery_cooldown() -> None:
    marker = {
        "telegram_live_action": "sent",
        "last_sent_at": "2026-05-26T01:34:03+04:00",
    }

    decision = notification_policy.decide_live_notification_from_marker(
        marker,
        now=_dt("2026-05-26T01:35:03+04:00"),
        sent_cooldown_seconds=300,
        scan_cooldown_seconds=180,
    )

    assert decision is not None
    assert decision.action == "cooldown"
    assert decision.retry_allowed is False


def test_no_jump_live_marker_uses_scan_cooldown() -> None:
    marker = {
        "telegram_live_action": "no_jump_highlight",
        "last_scan_at": "2026-05-26T01:34:03+04:00",
    }

    decision = notification_policy.decide_live_notification_from_marker(
        marker,
        now=_dt("2026-05-26T01:35:03+04:00"),
        sent_cooldown_seconds=300,
        scan_cooldown_seconds=180,
    )

    assert decision is not None
    assert decision.to_payload() == {
        "age_seconds": 60.0,
        "retry_allowed": False,
        "telegram_live_action": "scan_cooldown",
    }
