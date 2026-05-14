"""Cat TV Play integration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.typing import ConfigType

from .calibration import image_to_wall_homography, points_from_service_data, transform_image_point
from .const import (
    CONF_CAMERA_ENTITY_ID,
    CONF_DEFAULT_MEDIA_URL,
    CONF_MEDIA_PLAYER_ENTITY_ID,
    CONF_RECORDER_SWITCH_ENTITY_IDS,
    CONF_SNAPSHOT_SWITCH_ENTITY_IDS,
    DOMAIN,
    EVENT_CALIBRATION_SAVED,
    EVENT_OBSERVATION_RECORDED,
    EVENT_SESSION_STARTED,
    EVENT_SESSION_STOPPED,
    SERVICE_MEASURE_IMAGE_POINT,
    SERVICE_RECORD_OBSERVATION,
    SERVICE_SAVE_CALIBRATION,
    SERVICE_START_SESSION,
    SERVICE_STOP_SESSION,
)
from .sensor import SIGNAL_UPDATED
from .store import CatTvPlayStore

PLATFORMS = ["sensor"]


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _entry_config(entry: ConfigEntry) -> dict[str, Any]:
    data = dict(entry.data)
    data.update(entry.options)
    return data


def _split_entity_ids(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up global Cat TV Play services."""

    hass.data.setdefault(DOMAIN, {})
    store = CatTvPlayStore(hass)
    await store.async_load()
    hass.data[DOMAIN]["store"] = store

    async def _turn_switches(entity_ids: list[str], state: bool) -> None:
        if not entity_ids:
            return
        await hass.services.async_call(
            "switch",
            "turn_on" if state else "turn_off",
            {ATTR_ENTITY_ID: entity_ids},
            blocking=True,
        )

    async def _start_session(call: ServiceCall) -> dict[str, Any] | None:
        entry_id = call.data.get("entry_id")
        entry = _find_entry(hass, entry_id)
        config = _entry_config(entry)
        media_player = str(call.data.get("media_player_entity_id") or config[CONF_MEDIA_PLAYER_ENTITY_ID])
        media_url = str(call.data.get("media_url") or config.get(CONF_DEFAULT_MEDIA_URL) or "")
        if not media_url:
            raise vol.Invalid("media_url is required when no default media URL is configured")

        default_session_id = f"cat-tv-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        session_id = str(call.data.get("session_id") or default_session_id)
        now = _utc_now()
        active_session = {
            "session_id": session_id,
            "started_at": now,
            "media_player_entity_id": media_player,
            "media_url": media_url,
            "camera_entity_id": call.data.get("camera_entity_id") or config.get(CONF_CAMERA_ENTITY_ID),
            "config_entry_id": entry.entry_id,
        }

        await _turn_switches(_split_entity_ids(config.get(CONF_RECORDER_SWITCH_ENTITY_IDS)), True)
        await _turn_switches(_split_entity_ids(config.get(CONF_SNAPSHOT_SWITCH_ENTITY_IDS)), True)
        await hass.services.async_call(
            "media_player",
            "play_media",
            {
                ATTR_ENTITY_ID: media_player,
                "media_content_id": media_url,
                "media_content_type": call.data.get("media_content_type", "video/mp4"),
            },
            blocking=True,
        )

        store.data["active_session"] = active_session
        store.data.setdefault("sessions", []).append(active_session)
        await store.async_save()
        hass.bus.async_fire(EVENT_SESSION_STARTED, active_session)
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return {"session_id": session_id, "started_at": now}

    async def _stop_session(call: ServiceCall) -> dict[str, Any] | None:
        entry = _find_entry(hass, call.data.get("entry_id"))
        config = _entry_config(entry)
        active_session = store.data.get("active_session") or {}
        media_player = str(
            call.data.get("media_player_entity_id")
            or active_session.get("media_player_entity_id")
            or config[CONF_MEDIA_PLAYER_ENTITY_ID]
        )

        await hass.services.async_call("media_player", "media_stop", {ATTR_ENTITY_ID: media_player}, blocking=True)
        await _turn_switches(_split_entity_ids(config.get(CONF_RECORDER_SWITCH_ENTITY_IDS)), False)
        await _turn_switches(_split_entity_ids(config.get(CONF_SNAPSHOT_SWITCH_ENTITY_IDS)), False)

        stopped = dict(active_session)
        stopped["stopped_at"] = _utc_now()
        stopped["stop_reason"] = call.data.get("reason", "manual")
        store.data["active_session"] = None
        store.data.setdefault("sessions", []).append(stopped | {"kind": "session_stopped"})
        await store.async_save()
        hass.bus.async_fire(EVENT_SESSION_STOPPED, stopped)
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return {"session_id": stopped.get("session_id"), "stopped_at": stopped["stopped_at"]}

    async def _record_observation(call: ServiceCall) -> None:
        active_session = store.data.get("active_session") or {}
        observation = {
            "recorded_at": _utc_now(),
            "session_id": call.data.get("session_id") or active_session.get("session_id"),
            "behavior": call.data["behavior"],
            "confidence": call.data.get("confidence"),
            "image_x": call.data.get("image_x"),
            "image_y": call.data.get("image_y"),
            "wall_x_cm": call.data.get("wall_x_cm"),
            "wall_y_cm": call.data.get("wall_y_cm"),
            "jump_height_cm": call.data.get("jump_height_cm"),
            "note": call.data.get("note"),
        }
        store.data.setdefault("observations", []).append(observation)
        await store.async_save()
        hass.bus.async_fire(EVENT_OBSERVATION_RECORDED, observation)
        async_dispatcher_send(hass, SIGNAL_UPDATED)

    async def _save_calibration(call: ServiceCall) -> dict[str, Any]:
        calibration_id = str(call.data.get("calibration_id") or "default")
        points = points_from_service_data(call.data["points"])
        homography = image_to_wall_homography(points)
        payload = {
            "calibration_id": calibration_id,
            "created_at": _utc_now(),
            "points": [point.__dict__ for point in points],
            "homography": list(homography),
            "note": call.data.get("note"),
        }
        store.data.setdefault("calibrations", {})[calibration_id] = payload
        await store.async_save()
        hass.bus.async_fire(EVENT_CALIBRATION_SAVED, payload)
        async_dispatcher_send(hass, SIGNAL_UPDATED)
        return {
            "calibration_id": calibration_id,
            "point_count": len(points),
            "homography": list(homography),
        }

    async def _measure_image_point(call: ServiceCall) -> dict[str, Any]:
        calibration_id = str(call.data.get("calibration_id") or "default")
        calibration = (store.data.get("calibrations") or {}).get(calibration_id)
        if not calibration:
            raise vol.Invalid(f"unknown calibration_id: {calibration_id}")
        wall_x, wall_y = transform_image_point(
            tuple(float(value) for value in calibration["homography"]),
            float(call.data["image_x"]),
            float(call.data["image_y"]),
        )
        return {
            "calibration_id": calibration_id,
            "wall_x_cm": wall_x,
            "wall_y_cm": wall_y,
            "jump_height_cm": wall_y,
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_START_SESSION,
        _start_session,
        schema=START_SESSION_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_STOP_SESSION,
        _stop_session,
        schema=STOP_SESSION_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RECORD_OBSERVATION,
        _record_observation,
        schema=RECORD_OBSERVATION_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_SAVE_CALIBRATION,
        _save_calibration,
        schema=SAVE_CALIBRATION_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MEASURE_IMAGE_POINT,
        _measure_image_point,
        schema=MEASURE_IMAGE_POINT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Cat TV Play from a config entry."""

    hass.data.setdefault(DOMAIN, {}).setdefault("entries", {})[entry.entry_id] = entry
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a Cat TV Play config entry."""

    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        hass.data[DOMAIN].get("entries", {}).pop(entry.entry_id, None)
    return unloaded


def _find_entry(hass: HomeAssistant, entry_id: str | None) -> ConfigEntry:
    entries = hass.data.get(DOMAIN, {}).get("entries", {})
    if entry_id:
        if entry_id not in entries:
            raise vol.Invalid(f"unknown Cat TV Play entry_id: {entry_id}")
        return entries[entry_id]
    if len(entries) != 1:
        raise vol.Invalid("entry_id is required when more than one Cat TV Play entry exists")
    return next(iter(entries.values()))


START_SESSION_SCHEMA = vol.Schema(
    {
        vol.Optional("entry_id"): str,
        vol.Optional("session_id"): str,
        vol.Optional("media_player_entity_id"): str,
        vol.Optional("camera_entity_id"): str,
        vol.Optional("media_url"): str,
        vol.Optional("media_content_type", default="video/mp4"): str,
    }
)

STOP_SESSION_SCHEMA = vol.Schema(
    {
        vol.Optional("entry_id"): str,
        vol.Optional("media_player_entity_id"): str,
        vol.Optional("reason", default="manual"): str,
    }
)

RECORD_OBSERVATION_SCHEMA = vol.Schema(
    {
        vol.Required("behavior"): vol.In(
            ["noticed", "watching", "stalking", "paw", "jump", "left", "ignored", "other"]
        ),
        vol.Optional("session_id"): str,
        vol.Optional("confidence"): vol.Coerce(float),
        vol.Optional("image_x"): vol.Coerce(float),
        vol.Optional("image_y"): vol.Coerce(float),
        vol.Optional("wall_x_cm"): vol.Coerce(float),
        vol.Optional("wall_y_cm"): vol.Coerce(float),
        vol.Optional("jump_height_cm"): vol.Coerce(float),
        vol.Optional("note"): str,
    }
)

CALIBRATION_POINT_SCHEMA = vol.Schema(
    {
        vol.Required("image_x"): vol.Coerce(float),
        vol.Required("image_y"): vol.Coerce(float),
        vol.Required("wall_x_cm"): vol.Coerce(float),
        vol.Required("wall_y_cm"): vol.Coerce(float),
    }
)

SAVE_CALIBRATION_SCHEMA = vol.Schema(
    {
        vol.Optional("calibration_id", default="default"): str,
        vol.Required("points"): vol.All([CALIBRATION_POINT_SCHEMA], vol.Length(min=4)),
        vol.Optional("note"): str,
    }
)

MEASURE_IMAGE_POINT_SCHEMA = vol.Schema(
    {
        vol.Optional("calibration_id", default="default"): str,
        vol.Required("image_x"): vol.Coerce(float),
        vol.Required("image_y"): vol.Coerce(float),
    }
)
