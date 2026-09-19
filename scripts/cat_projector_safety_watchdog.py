#!/usr/bin/env python3
"""Keep cat-projector eye safety fail-closed without ending the play session."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_config as site_config  # noqa: E402

DEFAULT_HA_URL = site_config.home_assistant_url()
DEFAULT_STATUS_URL = site_config.service_url("overlay_status_url")
DEFAULT_STATE_PATH = site_config.learning_root() / "cat-projector-safety-watchdog.json"
DEFAULT_CONFIG_PATH = site_config.home_assistant_config_path() or Path("config.py")
#: Media player that speaks adb to the projector-side Android device.
PROJECTOR_ENTITY_ID = site_config.entity("adb_media_player")
#: Vendor shell command that puts the projector lamp into standby.
LAMP_STANDBY_TRUE_COMMAND = site_config.get(
    "projector_app.lamp_standby_command",
    "am startservice -n <vendor.power.service>/.PowerService --ez standby true",
)
SESSION_ENTITY_ID = site_config.entity("session_boolean")
DISPLAY_ACTIVE_ENTITY_ID = site_config.entity("display_active_binary_sensor")
LAMP_OFF_SCRIPT_ENTITY_ID = site_config.entity("lamp_off_script")
LAMP_ON_SCRIPT_ENTITY_ID = site_config.entity("lamp_on_script")

PROTECTED_STATUSES = {"active"}
CLEAR_STATUSES = {"active", "no_person", "no_projection_overlap"}
DANGER_STATUSES = {"safety_overlay_unavailable", "starting", "status_error", "status_stale"}


@dataclass(frozen=True)
class WatchdogInputs:
    session_on: bool
    display_active: bool
    overlay_status: str
    overlay_updated_at: float | None
    camera_error: str | None
    now: float


@dataclass(frozen=True)
class WatchdogLatch:
    lamp_held_off: bool = False
    clear_since: float | None = None
    danger_since: float | None = None


@dataclass(frozen=True)
class WatchdogDecision:
    action: str
    reason: str
    latch: WatchdogLatch


def decide_watchdog_action(
    inputs: WatchdogInputs,
    latch: WatchdogLatch,
    *,
    clear_delay_seconds: float,
    danger_delay_seconds: float,
    max_status_age_seconds: float,
) -> WatchdogDecision:
    if not inputs.session_on:
        return WatchdogDecision("none", "session_off", WatchdogLatch())

    overlay_status = inputs.overlay_status
    if inputs.overlay_updated_at is None or inputs.now - inputs.overlay_updated_at > max_status_age_seconds:
        overlay_status = "status_stale"

    protected = overlay_status in PROTECTED_STATUSES and not inputs.camera_error
    danger = overlay_status in DANGER_STATUSES or bool(inputs.camera_error)
    clear = overlay_status in CLEAR_STATUSES and not inputs.camera_error

    if inputs.display_active and danger:
        danger_since = latch.danger_since or inputs.now
        if inputs.now - danger_since < danger_delay_seconds:
            return WatchdogDecision(
                "none",
                f"danger_wait:{overlay_status}",
                WatchdogLatch(
                    lamp_held_off=latch.lamp_held_off,
                    clear_since=None,
                    danger_since=danger_since,
                ),
            )
        return WatchdogDecision(
            "lamp_off",
            f"danger_status:{overlay_status}",
            WatchdogLatch(lamp_held_off=True, clear_since=None, danger_since=danger_since),
        )

    if latch.lamp_held_off:
        if clear:
            clear_since = latch.clear_since or inputs.now
            if inputs.now - clear_since >= clear_delay_seconds:
                return WatchdogDecision("lamp_on", "clear_after_hold", WatchdogLatch())
            return WatchdogDecision(
                "none",
                "clear_wait",
                WatchdogLatch(lamp_held_off=True, clear_since=clear_since),
            )
        return WatchdogDecision(
            "none",
            f"held_off_waiting_for_clear:{overlay_status}",
            WatchdogLatch(lamp_held_off=True, danger_since=latch.danger_since),
        )

    if protected:
        return WatchdogDecision("none", f"protected_by_overlay:{overlay_status}", WatchdogLatch())

    return WatchdogDecision("none", f"no_action:{overlay_status}", WatchdogLatch())


def _load_token(config_path: Path) -> str:
    spec = importlib.util.spec_from_file_location("cat_tv_play_site_secrets", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    token = (
        getattr(module, "HOME_ASSISTANT_TOKEN", None)
        or getattr(module, "HASS_TOKEN", None)
        or getattr(module, "HA_TOKEN", None)
    )
    if not token:
        raise RuntimeError(f"Home Assistant token not found in {config_path}")
    return str(token)


def _fetch_json_url(url: str, *, timeout: float, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_ha_service(ha_url: str, token: str, domain: str, service: str, payload: dict[str, Any]) -> None:
    request = Request(
        f"{ha_url.rstrip('/')}/api/services/{domain}/{service}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        response.read()


def _send_lamp_off(ha_url: str, token: str) -> None:
    _post_ha_service(ha_url, token, "script", "turn_on", {"entity_id": LAMP_OFF_SCRIPT_ENTITY_ID})


def _ha_state(ha_url: str, token: str, entity_id: str) -> str | None:
    """Return the entity state, or None when Home Assistant is unreachable.

    A safety watchdog must outlive a Home Assistant restart: a transient error
    here used to raise and kill the process, and systemd's start limit then made
    that death permanent.
    """

    try:
        payload = _fetch_json_url(
            f"{ha_url.rstrip('/')}/api/states/{entity_id}",
            timeout=8,
            headers={"Authorization": f"Bearer {token}"},
        )
    except (OSError, TimeoutError, URLError, json.JSONDecodeError):
        return None
    return str(payload.get("state", "unknown"))


def _load_latch(path: Path) -> WatchdogLatch:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return WatchdogLatch()
    except json.JSONDecodeError:
        return WatchdogLatch()
    return WatchdogLatch(
        lamp_held_off=bool(payload.get("lamp_held_off", False)),
        clear_since=payload.get("clear_since"),
        danger_since=payload.get("danger_since"),
    )


def _save_latch(path: Path, latch: WatchdogLatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "lamp_held_off": latch.lamp_held_off,
                "clear_since": latch.clear_since,
                "danger_since": latch.danger_since,
                "updated_at": time.time(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_overlay_status(status_url: str, *, timeout: float) -> tuple[str, float | None, str | None]:
    try:
        payload = _fetch_json_url(status_url, timeout=timeout)
    except (OSError, TimeoutError, URLError, json.JSONDecodeError) as exc:
        return "status_error", None, str(exc)
    return (
        str(payload.get("status", "status_error")),
        payload.get("updated_at"),
        payload.get("camera_error"),
    )


def run_once(args: argparse.Namespace) -> WatchdogDecision:
    token = _load_token(args.config_path)
    now = time.time()
    session_state = _ha_state(args.ha_url, token, SESSION_ENTITY_ID)
    display_state = _ha_state(args.ha_url, token, DISPLAY_ACTIVE_ENTITY_ID)
    overlay_status, overlay_updated_at, camera_error = _read_overlay_status(
        args.overlay_status_url, timeout=args.timeout
    )
    latch = _load_latch(args.state_path)

    if session_state is None or display_state is None:
        # Home Assistant is unreachable: hold the current latch instead of
        # clearing it, and skip the lamp commands that would fail anyway.
        decision = WatchdogDecision("none", "ha_unavailable", latch)
        print(json.dumps({"action": decision.action, "reason": decision.reason}, sort_keys=True), flush=True)
        return decision

    decision = decide_watchdog_action(
        WatchdogInputs(
            session_on=session_state == "on",
            display_active=display_state == "on",
            overlay_status=overlay_status,
            overlay_updated_at=overlay_updated_at,
            camera_error=camera_error,
            now=now,
        ),
        latch,
        clear_delay_seconds=args.clear_delay_seconds,
        danger_delay_seconds=args.danger_delay_seconds,
        max_status_age_seconds=args.max_status_age_seconds,
    )

    _save_latch(args.state_path, decision.latch)

    try:
        if decision.action == "lamp_off":
            _send_lamp_off(args.ha_url, token)
        elif decision.action == "lamp_on":
            _post_ha_service(args.ha_url, token, "script", "turn_on", {"entity_id": LAMP_ON_SCRIPT_ENTITY_ID})
    except (OSError, TimeoutError, URLError) as exc:
        payload = {"action": decision.action, "reason": f"lamp_command_failed: {exc}"}
        print(json.dumps(payload, sort_keys=True), flush=True)
        return decision

    print(json.dumps({"action": decision.action, "reason": decision.reason}, sort_keys=True), flush=True)
    return decision


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ha-url", default=DEFAULT_HA_URL)
    parser.add_argument("--overlay-status-url", default=DEFAULT_STATUS_URL)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--interval-seconds", type=float, default=0.35)
    parser.add_argument("--clear-delay-seconds", type=float, default=5.0)
    parser.add_argument("--danger-delay-seconds", type=float, default=8.0)
    parser.add_argument("--max-status-age-seconds", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.state_path = args.state_path.expanduser()
    while True:
        run_once(args)
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
