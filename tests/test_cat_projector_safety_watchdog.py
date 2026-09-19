from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_safety_watchdog.py"
SPEC = spec_from_file_location("cat_projector_safety_watchdog", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
watchdog = module_from_spec(SPEC)
sys.modules[SPEC.name] = watchdog
SPEC.loader.exec_module(watchdog)

WatchdogInputs = watchdog.WatchdogInputs
WatchdogLatch = watchdog.WatchdogLatch
decide_watchdog_action = watchdog.decide_watchdog_action
LAMP_STANDBY_TRUE_COMMAND = watchdog.LAMP_STANDBY_TRUE_COMMAND


def _decide(
    *,
    session_on: bool = True,
    display_active: bool = True,
    overlay_status: str = "no_person",
    updated_at: float | None = 100.0,
    camera_error: str | None = None,
    now: float = 100.0,
    latch: WatchdogLatch | None = None,
):
    return decide_watchdog_action(
        WatchdogInputs(
            session_on=session_on,
            display_active=display_active,
            overlay_status=overlay_status,
            overlay_updated_at=updated_at,
            camera_error=camera_error,
            now=now,
        ),
        latch if latch is not None else WatchdogLatch(),
        clear_delay_seconds=1.5,
        danger_delay_seconds=8.0,
        max_status_age_seconds=1.5,
    )


def test_active_eye_overlay_preserves_lamp_and_session() -> None:
    decision = _decide(overlay_status="active")

    assert decision.action == "none"
    assert decision.reason == "protected_by_overlay:active"
    assert decision.latch.lamp_held_off is False


def test_stale_overlay_status_waits_before_fail_closed_while_display_is_active() -> None:
    decision = _decide(overlay_status="no_person", updated_at=90.0, now=100.0)

    assert decision.action == "none"
    assert decision.reason == "danger_wait:status_stale"
    assert decision.latch.lamp_held_off is False
    assert decision.latch.danger_since == 100.0


def test_sustained_stale_overlay_status_fails_closed_while_display_is_active() -> None:
    latch = WatchdogLatch(danger_since=90.0)

    decision = _decide(overlay_status="no_person", updated_at=90.0, now=100.0, latch=latch)

    assert decision.action == "lamp_off"
    assert decision.reason == "danger_status:status_stale"
    assert decision.latch.lamp_held_off is True
    assert decision.latch.danger_since == 90.0


def test_lamp_turns_back_on_after_clear_hold_without_clearing_session() -> None:
    latch = WatchdogLatch(lamp_held_off=True, clear_since=98.0)

    decision = _decide(overlay_status="no_projection_overlap", now=100.0, latch=latch)

    assert decision.action == "lamp_on"
    assert decision.reason == "clear_after_hold"
    assert decision.latch.lamp_held_off is False


def test_clear_status_waits_before_relighting_after_person_leaves() -> None:
    latch = WatchdogLatch(lamp_held_off=True, clear_since=None)

    decision = _decide(overlay_status="no_person", now=100.0, latch=latch)

    assert decision.action == "none"
    assert decision.reason == "clear_wait"
    assert decision.latch == WatchdogLatch(lamp_held_off=True, clear_since=100.0)


def test_clear_status_resets_pending_danger_without_lamp_command() -> None:
    latch = WatchdogLatch(danger_since=98.0)

    decision = _decide(overlay_status="no_person", now=100.0, latch=latch)

    assert decision.action == "none"
    assert decision.reason == "no_action:no_person"
    assert decision.latch == WatchdogLatch()


def test_session_off_resets_latch_without_lamp_command() -> None:
    latch = WatchdogLatch(lamp_held_off=True, clear_since=98.0)

    decision = _decide(session_on=False, overlay_status="active", latch=latch)

    assert decision.action == "none"
    assert decision.reason == "session_off"
    assert decision.latch == WatchdogLatch()


def test_lamp_off_command_uses_projector_standby_without_session_stop() -> None:
    assert "standby true" in LAMP_STANDBY_TRUE_COMMAND


def test_lamp_off_command_uses_canonical_ha_script(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, str]]] = []

    def fake_post_ha_service(ha_url: str, token: str, domain: str, service: str, payload: dict[str, str]) -> None:
        calls.append((domain, service, payload))

    monkeypatch.setattr(watchdog, "_post_ha_service", fake_post_ha_service)

    watchdog._send_lamp_off("http://ha", "token")

    assert calls == [
        ("script", "turn_on", {"entity_id": "script.cat_projector_lamp_off"}),
    ]
