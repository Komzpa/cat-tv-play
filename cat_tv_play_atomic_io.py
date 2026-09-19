"""Durable JSON/text persistence primitives.

Writes are replace-based: the old target stays intact until the new file is
fully written and synced. Readers keep each caller's fallback policy, so an
omitted default re-raises the original error instead of hiding a corrupt file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

_MISSING = object()


def atomic_write_text(path: Path, text: str, *, fsync: bool = True, mode: int | None = None) -> bool:
    """Atomically replace ``path`` when ``text`` changed, returning whether it did."""

    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        if fsync:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def atomic_write_json(
    path: Path,
    obj: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    fsync: bool = True,
    mode: int | None = None,
) -> bool:
    """Serialize JSON consistently and atomically, with a trailing newline."""

    return atomic_write_text(
        path,
        json.dumps(obj, ensure_ascii=ensure_ascii, indent=2, sort_keys=sort_keys) + "\n",
        fsync=fsync,
        mode=mode,
    )


def read_json_or[T](path: Path, default: T = _MISSING) -> Any | T:
    """Read JSON or return the caller's fallback; without one, re-raise."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        if default is _MISSING:
            raise
        return default


def read_json_or_none(path: Path) -> Any | None:
    """Read JSON, returning ``None`` for a missing, unreadable, or bad file."""

    return read_json_or(path, None)
