#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, parse, request

import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cat_tv_play_config as site_config  # noqa: E402

DEFAULT_HA_URL = site_config.home_assistant_url()
DEFAULT_EVENT_TYPE = "acoustic_event_cat_meow_candidate"
DEFAULT_SENSOR_ENTITY_ID = site_config.entity("acoustic_events_sensor")
DEFAULT_SESSION_ENTITY_ID = site_config.entity("session_boolean")
DEFAULT_RENDERER_ENTITY_ID = site_config.entity("media_player")
DEFAULT_SCRIPT_ENTITY_ID = site_config.entity("start_script")
DEFAULT_MIN_MEOW_SCORE = 0.62
DEFAULT_COOLDOWN_SECONDS = 3 * 60 * 60
#: Microphone source ids whose meow candidates may start a session.
DEFAULT_SOURCE_IDS = set(site_config.get("audio.meow_source_ids") or ())
DEFAULT_BLOCKED_BROAD_LABELS = {
    "announcement",
    "fan",
    "keyboard",
    "keyboard_typing",
    "music",
    "robot_vacuum",
    "scrape_or_thump",
    "silence_noise",
    "snore",
    "speech",
}
DEFAULT_BLOCKED_INTENT_LABELS = {
    "attention",
    "complaint",
    "distress",
}
DEFAULT_BLOCKED_INTENT_CONFIDENCE = 0.0
DEFAULT_BLOCKED_INTENT_MAX_PLAY_PROBABILITY = 0.25


@dataclass(frozen=True)
class MeowDecision:
    accepted: bool
    reason: str


def _csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _float_payload(payload: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key) or default)
    except (TypeError, ValueError):
        return default


def _mapping_payload(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def should_accept_meow_intent(
    payload: dict[str, object],
    *,
    blocked_intent_labels: set[str],
    blocked_intent_confidence: float,
    blocked_intent_max_play_probability: float,
) -> MeowDecision:
    intent = _mapping_payload(payload, "intent_hint")
    if not intent:
        return MeowDecision(True, "intent_missing")

    label = str(intent.get("label") or "")
    confidence = _float_payload(intent, "confidence")
    probabilities = _mapping_payload(intent, "class_probabilities")
    play_probability = _float_payload(probabilities, "play")

    if (
        label in blocked_intent_labels
        and confidence >= blocked_intent_confidence
        and play_probability <= blocked_intent_max_play_probability
    ):
        return MeowDecision(False, f"blocked_intent:{label}:{confidence:.3f}:play={play_probability:.3f}")

    return MeowDecision(True, f"intent:{label or 'unknown'}")


def should_accept_meow_event(
    payload: dict[str, object],
    *,
    source_ids: set[str],
    min_score: float,
    blocked_broad_labels: set[str],
    broad_block_confidence: float,
    blocked_intent_labels: set[str] = DEFAULT_BLOCKED_INTENT_LABELS,
    blocked_intent_confidence: float = DEFAULT_BLOCKED_INTENT_CONFIDENCE,
    blocked_intent_max_play_probability: float = DEFAULT_BLOCKED_INTENT_MAX_PLAY_PROBABILITY,
) -> MeowDecision:
    source_id = str(payload.get("source_id") or "")
    if source_id not in source_ids:
        return MeowDecision(False, f"source_not_allowed:{source_id or 'unknown'}")

    score = _float_payload(payload, "meow_score")
    if score < min_score:
        return MeowDecision(False, f"score_below_threshold:{score:.3f}")

    broad = str(payload.get("broad_acoustic_label") or "")
    broad_confidence = _float_payload(payload, "broad_acoustic_confidence")
    if broad in blocked_broad_labels and broad_confidence >= broad_block_confidence:
        return MeowDecision(False, f"blocked_broad_label:{broad}:{broad_confidence:.3f}")

    intent_decision = should_accept_meow_intent(
        payload,
        blocked_intent_labels=blocked_intent_labels,
        blocked_intent_confidence=blocked_intent_confidence,
        blocked_intent_max_play_probability=blocked_intent_max_play_probability,
    )
    if not intent_decision.accepted:
        return intent_decision

    return MeowDecision(True, "accepted")


def should_start_projector(*, session_state: str, renderer_state: str) -> MeowDecision:
    if session_state == "off":
        return MeowDecision(True, "session_off")
    if renderer_state == "unavailable":
        return MeowDecision(True, "session_on_renderer_unavailable")
    return MeowDecision(False, f"already_active:{session_state}:{renderer_state}")


def _load_home_assistant_credentials(cwd: Path, url: str, token: str, token_file: Path | None) -> tuple[str, str]:
    if token_file:
        token = token or token_file.expanduser().read_text(encoding="utf-8").strip()
    if url and token:
        return url.rstrip("/"), token

    config_path = cwd / "config.py"
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("cat_projector_meow_watchdog_config", config_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            url = url or str(getattr(module, "HOME_ASSISTANT_URL", "") or "")
            token = token or str(getattr(module, "HOME_ASSISTANT_TOKEN", "") or "")

    if not url:
        url = DEFAULT_HA_URL
    if not token:
        raise RuntimeError("Home Assistant token is not configured")
    return url.rstrip("/"), token


def _ha_request(
    *,
    ha_url: str,
    ha_token: str,
    path: str,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    timeout: float,
) -> object:
    raw = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(
        f"{ha_url}{path}",
        data=raw,
        headers={"Authorization": f"Bearer {ha_token}", "Content-Type": "application/json"},
        method=method,
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def _ha_state(*, ha_url: str, ha_token: str, entity_id: str, timeout: float) -> str:
    try:
        data = _ha_request(
            ha_url=ha_url,
            ha_token=ha_token,
            path=f"/api/states/{parse.quote(entity_id, safe='')}",
            timeout=timeout,
        )
    except error.HTTPError as exc:
        if exc.code == 404:
            return "not_found"
        raise
    if not isinstance(data, dict):
        return "unknown"
    return str(data.get("state") or "unknown")


class CatProjectorMeowWatchdog:
    def __init__(
        self,
        *,
        ha_url: str,
        ha_token: str,
        event_type: str,
        sensor_entity_id: str,
        session_entity_id: str,
        renderer_entity_id: str,
        script_entity_id: str,
        source_ids: set[str],
        min_meow_score: float,
        blocked_broad_labels: set[str],
        broad_block_confidence: float,
        blocked_intent_labels: set[str],
        blocked_intent_confidence: float,
        blocked_intent_max_play_probability: float,
        cooldown_seconds: float,
        ha_timeout: float,
    ) -> None:
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.event_type = event_type
        self.sensor_entity_id = sensor_entity_id
        self.session_entity_id = session_entity_id
        self.renderer_entity_id = renderer_entity_id
        self.script_entity_id = script_entity_id
        self.source_ids = source_ids
        self.min_meow_score = min_meow_score
        self.blocked_broad_labels = blocked_broad_labels
        self.broad_block_confidence = broad_block_confidence
        self.blocked_intent_labels = blocked_intent_labels
        self.blocked_intent_confidence = blocked_intent_confidence
        self.blocked_intent_max_play_probability = blocked_intent_max_play_probability
        self.cooldown_seconds = cooldown_seconds
        self.ha_timeout = ha_timeout
        self._last_start_at = 0.0

    async def run_forever(self) -> None:
        while True:
            try:
                await self._run_once()
            except Exception as exc:
                print(f"watchdog reconnect after {type(exc).__name__}: {str(exc)[-400:]}", flush=True)
                await asyncio.sleep(5.0)

    async def _run_once(self) -> None:
        ws_url = self.ha_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        async with websockets.connect(f"{ws_url}/api/websocket", open_timeout=self.ha_timeout) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "auth", "access_token": self.ha_token}))
            auth = json.loads(await ws.recv())
            if auth.get("type") != "auth_ok":
                raise RuntimeError(f"HA websocket auth failed: {auth}")

            await ws.send(json.dumps({"id": 1, "type": "subscribe_events", "event_type": self.event_type}))
            await self._expect_success(ws, 1)
            await ws.send(json.dumps({"id": 2, "type": "subscribe_events", "event_type": "state_changed"}))
            await self._expect_success(ws, 2)
            print("cat projector meow watchdog connected", flush=True)

            async for raw in ws:
                message = json.loads(raw)
                if message.get("type") != "event":
                    continue
                await self._handle_ha_event(message.get("event") or {})

    async def _expect_success(self, ws: object, message_id: int) -> None:
        raw = await ws.recv()
        message = json.loads(raw)
        if message.get("id") != message_id or not message.get("success"):
            raise RuntimeError(f"unexpected websocket response for {message_id}: {message}")

    async def _handle_ha_event(self, event: dict[str, object]) -> None:
        event_type = str(event.get("event_type") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        assert isinstance(data, dict)

        if event_type == self.event_type:
            decision = should_accept_meow_event(
                data,
                source_ids=self.source_ids,
                min_score=self.min_meow_score,
                blocked_broad_labels=self.blocked_broad_labels,
                broad_block_confidence=self.broad_block_confidence,
                blocked_intent_labels=self.blocked_intent_labels,
                blocked_intent_confidence=self.blocked_intent_confidence,
                blocked_intent_max_play_probability=self.blocked_intent_max_play_probability,
            )
            if not decision.accepted:
                print(f"ignore meow event: {decision.reason}", flush=True)
                return
            await self._maybe_start_projector(f"event:{decision.reason}")
            return

        if event_type == "state_changed" and data.get("entity_id") == self.sensor_entity_id:
            new_state = data.get("new_state")
            if isinstance(new_state, dict) and new_state.get("state") == "cat_dialog":
                await self._maybe_start_projector("sensor:cat_dialog")

    async def _maybe_start_projector(self, trigger_reason: str) -> None:
        now = time.monotonic()
        session_state = _ha_state(
            ha_url=self.ha_url,
            ha_token=self.ha_token,
            entity_id=self.session_entity_id,
            timeout=self.ha_timeout,
        )
        renderer_state = _ha_state(
            ha_url=self.ha_url,
            ha_token=self.ha_token,
            entity_id=self.renderer_entity_id,
            timeout=self.ha_timeout,
        )
        decision = should_start_projector(session_state=session_state, renderer_state=renderer_state)
        if not decision.accepted:
            print(f"skip projector start trigger={trigger_reason} reason={decision.reason}", flush=True)
            return
        if decision.reason != "session_on_renderer_unavailable" and now - self._last_start_at < self.cooldown_seconds:
            print(f"skip projector start cooldown trigger={trigger_reason}", flush=True)
            return

        _ha_request(
            ha_url=self.ha_url,
            ha_token=self.ha_token,
            path="/api/services/script/turn_on",
            method="POST",
            payload={"entity_id": self.script_entity_id},
            timeout=self.ha_timeout,
        )
        self._last_start_at = now
        print(
            f"started projector trigger={trigger_reason} reason={decision.reason} "
            f"session={session_state} renderer={renderer_state}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the cat projector from HA meow events when HA package deploy is unavailable."
    )
    parser.add_argument("--ha-url", default=os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_HA_URL", DEFAULT_HA_URL))
    parser.add_argument("--ha-token", default=os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_HA_TOKEN", ""))
    parser.add_argument("--ha-token-file", type=Path, default=os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_HA_TOKEN_FILE"))
    parser.add_argument("--cwd", type=Path, default=Path(os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_CWD", os.getcwd())))
    parser.add_argument("--event-type", default=os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_EVENT_TYPE", DEFAULT_EVENT_TYPE))
    parser.add_argument(
        "--sensor-entity-id",
        default=os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_SENSOR_ENTITY_ID", DEFAULT_SENSOR_ENTITY_ID),
    )
    parser.add_argument(
        "--session-entity-id",
        default=os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_SESSION_ENTITY_ID", DEFAULT_SESSION_ENTITY_ID),
    )
    parser.add_argument(
        "--renderer-entity-id",
        default=os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_RENDERER_ENTITY_ID", DEFAULT_RENDERER_ENTITY_ID),
    )
    parser.add_argument(
        "--script-entity-id",
        default=os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_SCRIPT_ENTITY_ID", DEFAULT_SCRIPT_ENTITY_ID),
    )
    parser.add_argument(
        "--source-ids",
        default=os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_SOURCE_IDS", ",".join(sorted(DEFAULT_SOURCE_IDS))),
    )
    parser.add_argument(
        "--min-meow-score",
        type=float,
        default=float(os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_MIN_MEOW_SCORE", DEFAULT_MIN_MEOW_SCORE)),
    )
    parser.add_argument(
        "--blocked-broad-labels",
        default=os.getenv(
            "CAT_PROJECTOR_MEOW_WATCHDOG_BLOCKED_BROAD_LABELS", ",".join(sorted(DEFAULT_BLOCKED_BROAD_LABELS))
        ),
    )
    parser.add_argument(
        "--broad-block-confidence",
        type=float,
        default=float(os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_BROAD_BLOCK_CONFIDENCE", "0.25")),
    )
    parser.add_argument(
        "--blocked-intent-labels",
        default=os.getenv(
            "CAT_PROJECTOR_MEOW_WATCHDOG_BLOCKED_INTENT_LABELS", ",".join(sorted(DEFAULT_BLOCKED_INTENT_LABELS))
        ),
    )
    parser.add_argument(
        "--blocked-intent-confidence",
        type=float,
        default=float(
            os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_BLOCKED_INTENT_CONFIDENCE", DEFAULT_BLOCKED_INTENT_CONFIDENCE)
        ),
    )
    parser.add_argument(
        "--blocked-intent-max-play-probability",
        type=float,
        default=float(
            os.getenv(
                "CAT_PROJECTOR_MEOW_WATCHDOG_BLOCKED_INTENT_MAX_PLAY_PROBABILITY",
                DEFAULT_BLOCKED_INTENT_MAX_PLAY_PROBABILITY,
            )
        ),
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=float(os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_COOLDOWN_SECONDS", DEFAULT_COOLDOWN_SECONDS)),
    )
    parser.add_argument(
        "--ha-timeout", type=float, default=float(os.getenv("CAT_PROJECTOR_MEOW_WATCHDOG_HA_TIMEOUT", "10"))
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ha_url, ha_token = _load_home_assistant_credentials(args.cwd, args.ha_url, args.ha_token, args.ha_token_file)
    watchdog = CatProjectorMeowWatchdog(
        ha_url=ha_url,
        ha_token=ha_token,
        event_type=args.event_type,
        sensor_entity_id=args.sensor_entity_id,
        session_entity_id=args.session_entity_id,
        renderer_entity_id=args.renderer_entity_id,
        script_entity_id=args.script_entity_id,
        source_ids=_csv_set(args.source_ids),
        min_meow_score=args.min_meow_score,
        blocked_broad_labels=_csv_set(args.blocked_broad_labels),
        broad_block_confidence=args.broad_block_confidence,
        blocked_intent_labels=_csv_set(args.blocked_intent_labels),
        blocked_intent_confidence=args.blocked_intent_confidence,
        blocked_intent_max_play_probability=args.blocked_intent_max_play_probability,
        cooldown_seconds=args.cooldown_seconds,
        ha_timeout=args.ha_timeout,
    )
    asyncio.run(watchdog.run_forever())


if __name__ == "__main__":
    main()
