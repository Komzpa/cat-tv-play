"""Persistent storage for Cat TV Play."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import STORAGE_KEY, STORAGE_VERSION


class PetTvStore:
    """Small JSON-backed store for sessions, observations, and calibrations."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self.data: dict[str, Any] = {
            "active_session": None,
            "sessions": [],
            "observations": [],
            "calibrations": {},
        }

    async def async_load(self) -> None:
        loaded = await self._store.async_load()
        if isinstance(loaded, dict):
            self.data.update(loaded)

    async def async_save(self) -> None:
        await self._store.async_save(self.data)

