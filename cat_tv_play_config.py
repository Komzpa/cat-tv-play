#!/usr/bin/env python3
"""Site configuration for Cat TV Play.

Public code in this repository carries no hostnames, IP addresses, storage
paths, device names, entity ids, or credentials. One installation puts those in
``deploy/private.yml``; git ignores that file and ``deploy/private.example.yml``
documents its shape with placeholder values.

Resolution order for every key, first hit wins:

1. the environment variable named in ``ENV_OVERRIDES`` for that key;
2. ``deploy/private.yml`` (or the file named by ``CAT_TV_PLAY_CONFIG``);
3. the neutral built-in default, which is either empty or a local loopback
   address, so an unconfigured checkout never reaches somebody's house.

Scripts call the small accessors at the bottom of this module instead of
reading the mapping directly, so a renamed key changes one place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "deploy" / "private.yml"
#: Used when no installation configured a learning root.
FALLBACK_LEARNING_ROOT = Path("~/.local/share/cat-tv-play/learning")
EXAMPLE_CONFIG_PATH = REPO_ROOT / "deploy" / "private.example.yml"

#: Dotted config key -> environment variable that overrides it.
ENV_OVERRIDES: dict[str, str] = {
    "storage.learning_root": "CAT_TV_LEARNING_ROOT",
    "storage.recordings_root": "CAT_TV_RECORDINGS_ROOT",
    "storage.datasets_root": "CAT_TV_DATASETS_ROOT",
    "storage.models_root": "CAT_TV_MODELS_ROOT",
    "storage.audio_events_root": "CAT_TV_AUDIO_EVENTS_ROOT",
    "storage.model_cache_root": "CAT_TV_MODEL_CACHE_ROOT",
    "storage.calibration_path": "CAT_TV_CALIBRATION_PATH",
    "storage.extra_corpus_roots": "CAT_TV_EXTRA_CORPUS_ROOTS",
    "home_assistant.url": "CAT_TV_HA_URL",
    "home_assistant.config_path": "CAT_TV_HA_CONFIG_PATH",
    "home_assistant.token": "CAT_TV_HA_TOKEN",
    "home_assistant.token_file": "CAT_TV_HA_TOKEN_FILE",
    "camera.snapshot_url": "CAT_TV_CAMERA_SNAPSHOT_URL",
    "camera.video_url": "CAT_TV_CAMERA_VIDEO_URL",
    "camera.stream_url": "CAT_TV_CAMERA_STREAM_URL",
    "camera.adb_serial": "CAT_TV_CAMERA_ADB_SERIAL",
    "projector_app.package": "CAT_TV_PROJECTOR_APP_PACKAGE",
    "services.overlay_status_url": "CAT_TV_OVERLAY_STATUS_URL",
    "services.overlay_video_url": "CAT_TV_OVERLAY_VIDEO_URL",
    "services.overlay_advertise_host": "CAT_TV_OVERLAY_ADVERTISE_HOST",
    "services.review_ui_url": "CAT_TV_REVIEW_UI_URL",
    "services.sam_endpoint": "CAT_PROJECTOR_SAM_ENDPOINT",
    "database.url": "CAT_TV_DATABASE_URL",
    "notify.command": "CAT_TV_NOTIFY_COMMAND",
}

#: Neutral defaults. Nothing here points at a real installation.
DEFAULTS: dict[str, Any] = {
    "corpus": {
        "false_negative_sessions": [],
    },
    "audio": {
        "meow_source_ids": [],
    },
    "storage": {
        "learning_root": "",
        "calibration_path": "",
        "recordings_root": "",
        "datasets_root": "",
        "models_root": "",
        "audio_events_root": "",
        "model_cache_root": "",
        "extra_corpus_roots": [],
    },
    "home_assistant": {
        "url": "",
        "config_path": "",
        "token": "",
        "token_file": "",
        "entities": {
            "session_boolean": "input_boolean.cat_projector_session",
            "session_id_text": "input_text.cat_projector_session_id",
            "frigate_event_id_text": "input_text.cat_projector_frigate_event_id",
            "video_url_text": "input_text.cat_projector_video_url",
            "display_active_binary_sensor": "binary_sensor.cat_projector_display_active",
            "start_script": "script.cat_projector_start",
            "lamp_on_script": "script.cat_projector_lamp_on",
            "lamp_off_script": "script.cat_projector_lamp_off",
            "media_player": "",
            "adb_media_player": "",
            "acoustic_events_sensor": "",
            "treat_dispenser_button": "",
            "vacuum_status_sensor": "",
            "quiet_hours_booleans": [],
        },
    },
    "camera": {
        "snapshot_url": "",
        "video_url": "",
        "stream_url": "",
        "adb_serial": "",
    },
    "projector_app": {
        "package": "",
        "lamp_standby_command": "",
    },
    "services": {
        "overlay_status_url": "http://127.0.0.1:8787/status.json",
        "overlay_video_url": "",
        "overlay_advertise_host": "127.0.0.1",
        "review_ui_url": "http://127.0.0.1:8790/calibration-tools/projector-wall-calibrator.html",
        "sam_endpoint": "http://127.0.0.1:8766/segment",
    },
    "database": {
        "url": "",
    },
    "notify": {
        "command": [],
    },
}

_CACHE: dict[str, Any] | None = None
_CACHE_SOURCE: Path | None = None


def config_path() -> Path:
    configured = os.environ.get("CAT_TV_PLAY_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_CONFIG_PATH


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError(f"reading {path} needs PyYAML; install the project dependencies") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


def load(*, refresh: bool = False) -> dict[str, Any]:
    """Return the merged configuration mapping."""

    global _CACHE, _CACHE_SOURCE
    path = config_path()
    if not refresh and _CACHE is not None and path == _CACHE_SOURCE:
        return _CACHE
    overlay = _load_yaml(path) if path.is_file() else {}
    _CACHE = _merge(DEFAULTS, overlay)
    _CACHE_SOURCE = path
    return _CACHE


def reset_cache() -> None:
    """Forget the loaded configuration; tests call this after changing the environment."""

    global _CACHE, _CACHE_SOURCE
    _CACHE = None
    _CACHE_SOURCE = None


def get(dotted_key: str, default: Any = None) -> Any:
    """Return one configuration value, honouring the environment override first."""

    env_name = ENV_OVERRIDES.get(dotted_key)
    if env_name:
        raw = os.environ.get(env_name)
        if raw:
            return _coerce_like(raw, _lookup(DEFAULTS, dotted_key))
    value = _lookup(load(), dotted_key)
    if value in (None, "", [], {}):
        return default if default is not None else value
    return value


def _lookup(mapping: dict[str, Any], dotted_key: str) -> Any:
    node: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _coerce_like(raw: str, sample: Any) -> Any:
    if isinstance(sample, list):
        return [item for item in (piece.strip() for piece in raw.split(os.pathsep)) if item]
    return raw


def _path_or_none(dotted_key: str) -> Path | None:
    value = get(dotted_key)
    if not value:
        return None
    return Path(str(value)).expanduser()


# --- storage -----------------------------------------------------------------


def learning_root() -> Path:
    """Root of the small mutable learning state (labels, queues, models)."""

    configured = _path_or_none("storage.learning_root")
    return configured if configured is not None else FALLBACK_LEARNING_ROOT.expanduser()


def recordings_root() -> Path:
    """Root of the raw session recordings, which may sit on another filesystem."""

    configured = _path_or_none("storage.recordings_root")
    return configured if configured is not None else learning_root() / "recordings"


def datasets_root() -> Path:
    configured = _path_or_none("storage.datasets_root")
    return configured if configured is not None else learning_root() / "datasets"


def models_root() -> Path:
    configured = _path_or_none("storage.models_root")
    return configured if configured is not None else learning_root() / "models"


def audio_events_root() -> Path:
    """Root of the acoustic-event sidecars produced by the audio project."""

    configured = _path_or_none("storage.audio_events_root")
    return configured if configured is not None else learning_root() / "audio-events"


def model_cache_root() -> Path:
    configured = _path_or_none("storage.model_cache_root")
    return configured if configured is not None else learning_root() / "model-cache"


def extra_corpus_roots() -> list[Path]:
    """Extra read-only corpus roots the review backend may serve files from."""

    values = get("storage.extra_corpus_roots") or []
    if isinstance(values, str):
        values = [values]
    return [Path(str(value)).expanduser() for value in values if str(value).strip()]


# --- Home Assistant ----------------------------------------------------------


def home_assistant_url(default: str = "") -> str:
    return str(get("home_assistant.url", default) or default)


def home_assistant_config_path() -> Path | None:
    return _path_or_none("home_assistant.config_path")


def home_assistant_token() -> str:
    """Return the long-lived access token from the configured source.

    Order: an inline ``home_assistant.token``, a ``home_assistant.token_file``,
    then a Python module named by ``home_assistant.config_path`` that defines
    ``HOME_ASSISTANT_TOKEN``, ``HASS_TOKEN`` or ``HA_TOKEN``.
    """

    inline = str(get("home_assistant.token") or "").strip()
    if inline:
        return inline

    token_file = _path_or_none("home_assistant.token_file")
    if token_file is not None and token_file.is_file():
        found = token_file.read_text(encoding="utf-8").strip()
        if found:
            return found

    module_path = home_assistant_config_path()
    if module_path is not None and module_path.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location("cat_tv_play_site_secrets", module_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name in ("HOME_ASSISTANT_TOKEN", "HASS_TOKEN", "HA_TOKEN"):
                found = str(getattr(module, name, "") or "").strip()
                if found:
                    return found

    raise RuntimeError(
        "no Home Assistant token configured: set home_assistant.token, "
        f"home_assistant.token_file or home_assistant.config_path in {config_path()}"
    )


def calibration_path() -> Path:
    """Wall-plane calibration written by the review UI."""

    configured = _path_or_none("storage.calibration_path")
    if configured is not None:
        return configured
    return learning_root() / "calibration" / "active_wall_calibration.json"


def entity(name: str, default: str = "") -> str:
    """Return one configured entity id, or ``default`` when the home has none."""

    value = _lookup(load(), f"home_assistant.entities.{name}")
    if value in (None, ""):
        return default
    return str(value)


def entity_list(name: str) -> list[str]:
    value = _lookup(load(), f"home_assistant.entities.{name}") or []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


# --- devices and services ----------------------------------------------------


def camera_snapshot_url(default: str = "") -> str:
    return str(get("camera.snapshot_url", default) or default)


def camera_video_url(default: str = "") -> str:
    return str(get("camera.video_url", default) or default)


def camera_stream_url(default: str = "") -> str:
    value = get("camera.stream_url", "")
    return str(value or camera_video_url(default))


def camera_adb_serial(default: str = "") -> str:
    return str(get("camera.adb_serial", default) or default)


def projector_app_package(default: str = "") -> str:
    """Android application id of the companion projector/camera app."""

    return str(get("projector_app.package", default) or default)


def projector_app_action(name: str) -> str:
    package = projector_app_package()
    return f"{package}.{name}" if package else ""


def projector_app_component(activity: str) -> str:
    package = projector_app_package()
    return f"{package}/.{activity}" if package else ""


def service_url(name: str, default: str = "") -> str:
    return str(get(f"services.{name}", default) or default)


def database_url(default: str = "") -> str:
    return str(get("database.url", default) or default)


def notify_command() -> list[str]:
    value = get("notify.command") or []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value]
