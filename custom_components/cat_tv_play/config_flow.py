"""Config flow for Cat TV Play."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import selector

from .const import (
    CONF_CAMERA_ENTITY_ID,
    CONF_DEFAULT_MEDIA_URL,
    CONF_MEDIA_PLAYER_ENTITY_ID,
    CONF_RECORDER_SWITCH_ENTITY_IDS,
    CONF_SNAPSHOT_SWITCH_ENTITY_IDS,
    DOMAIN,
)


def _schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    defaults = defaults or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_MEDIA_PLAYER_ENTITY_ID,
                default=defaults.get(CONF_MEDIA_PLAYER_ENTITY_ID),
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="media_player")),
            vol.Optional(
                CONF_CAMERA_ENTITY_ID,
                default=defaults.get(CONF_CAMERA_ENTITY_ID, ""),
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="camera")),
            vol.Optional(
                CONF_DEFAULT_MEDIA_URL,
                default=defaults.get(CONF_DEFAULT_MEDIA_URL, ""),
            ): str,
            vol.Optional(
                CONF_RECORDER_SWITCH_ENTITY_IDS,
                default=defaults.get(CONF_RECORDER_SWITCH_ENTITY_IDS, ""),
            ): str,
            vol.Optional(
                CONF_SNAPSHOT_SWITCH_ENTITY_IDS,
                default=defaults.get(CONF_SNAPSHOT_SWITCH_ENTITY_IDS, ""),
            ): str,
        }
    )


class CatTvPlayConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a Cat TV Play config flow."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> config_entries.FlowResult:
        if user_input is not None:
            await self.async_set_unique_id(str(user_input[CONF_MEDIA_PLAYER_ENTITY_ID]))
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title="Cat TV Play", data=user_input)

        return self.async_show_form(step_id="user", data_schema=_schema())

    @staticmethod
    def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> config_entries.OptionsFlow:
        return CatTvPlayOptionsFlow(config_entry)


class CatTvPlayOptionsFlow(config_entries.OptionsFlow):
    """Edit Cat TV Play options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> config_entries.FlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        defaults = dict(self.config_entry.data)
        defaults.update(self.config_entry.options)
        return self.async_show_form(step_id="init", data_schema=_schema(defaults))
