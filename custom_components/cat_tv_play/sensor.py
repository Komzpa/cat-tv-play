"""Sensors for Cat TV Play."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

SIGNAL_UPDATED = f"{DOMAIN}_updated"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Cat TV Play sensors."""

    async_add_entities([CatTvPlaySessionSensor(hass, entry)])


class CatTvPlaySessionSensor(SensorEntity):
    """Expose the active Cat TV session."""

    _attr_icon = "mdi:cat"
    _attr_name = "Cat TV Play session"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_session"
        self._remove_signal: Callable[[], None] | None = None

    async def async_added_to_hass(self) -> None:
        self._remove_signal = async_dispatcher_connect(self.hass, SIGNAL_UPDATED, self._handle_update)

    async def async_will_remove_from_hass(self) -> None:
        if self._remove_signal is not None:
            self._remove_signal()

    @callback
    def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> str:
        store = self.hass.data[DOMAIN]["store"]
        active_session = store.data.get("active_session")
        if not active_session:
            return "idle"
        return str(active_session.get("session_id") or "active")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        store = self.hass.data[DOMAIN]["store"]
        return {
            "active_session": store.data.get("active_session"),
            "last_observation": (store.data.get("observations") or [None])[-1],
            "calibrations": list((store.data.get("calibrations") or {}).keys()),
        }
