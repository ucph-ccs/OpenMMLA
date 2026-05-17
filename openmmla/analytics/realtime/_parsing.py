"""helpers for parsing measurement payloads in realtime analytics."""

from __future__ import annotations

import json
from typing import Any


def coerce_json_list(value: Any, default: list[Any] | None = None) -> list[Any]:
    """Return a list from either a parsed list or a JSON string."""
    if default is None:
        default = []

    if value is None:
        return list(default)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return list(default)
        return parsed if isinstance(parsed, list) else list(default)
    return list(default)


def coerce_json_dict(value: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a dict from either a parsed dict or a JSON string."""
    if default is None:
        default = {}

    if value is None:
        return dict(default)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return dict(default)
        return parsed if isinstance(parsed, dict) else dict(default)
    return dict(default)
