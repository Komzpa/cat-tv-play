#!/usr/bin/env python3
"""Bridge to the acoustic-event tagging helpers owned by the audio project.

Meow scoring reads acoustic-event sidecars and embeddings that a separate audio
project writes. That project owns the sidecar format, the operational label set,
and the embedding lookup, so this repository imports them instead of keeping a
second copy that could drift.

Point ``CAT_TV_AUDIO_TAGGING_MODULE`` at the importable module name and, when it
is not installed on ``sys.path``, ``CAT_TV_AUDIO_TAGGING_PATH`` at its checkout.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

DEFAULT_MODULE = "acoustic_event_tagging_models"
DEFAULT_CAPTURE_MODULE = "home_audio_mesh.capture.event_capture"


class AudioTaggingUnavailable(RuntimeError):
    """The acoustic tagging helpers are not installed."""


def module_name() -> str:
    return os.environ.get("CAT_TV_AUDIO_TAGGING_MODULE", DEFAULT_MODULE)


def load() -> ModuleType:
    """Import and return the acoustic tagging helper module."""

    _add_extra_path()
    name = module_name()
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the host install
        raise AudioTaggingUnavailable(
            f"acoustic tagging module {name!r} is not importable; install the audio project or set "
            "CAT_TV_AUDIO_TAGGING_MODULE / CAT_TV_AUDIO_TAGGING_PATH"
        ) from exc


def _add_extra_path() -> None:
    extra_path = os.environ.get("CAT_TV_AUDIO_TAGGING_PATH")
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)


def meow_feature_names() -> list[str]:
    """Return the handcrafted meow feature names the audio project extracts."""

    _add_extra_path()
    name = os.environ.get("CAT_TV_MEOW_FEATURES_MODULE", DEFAULT_CAPTURE_MODULE)
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the host install
        raise AudioTaggingUnavailable(
            f"meow feature module {name!r} is not importable; install the audio project or set "
            "CAT_TV_MEOW_FEATURES_MODULE / CAT_TV_AUDIO_TAGGING_PATH"
        ) from exc
    return list(module.MEOW_FEATURE_NAMES)


def __getattr__(name: str):
    """Expose the helper module's public names lazily."""

    if name.startswith("_"):
        raise AttributeError(name)
    return getattr(load(), name)
