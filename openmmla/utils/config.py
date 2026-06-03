from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


SYSTEM_SERVICES_REL_PATH = Path("config") / "system_services.yml"
SYSTEM_SERVICE_SECTIONS = ("InfluxDB", "MongoDB", "MQTT", "Redis")


def load_yaml_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def find_system_services_config(
    start_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    explicit = os.environ.get("OPENMMLA_SYSTEM_SERVICES_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    if start_path is None:
        current = Path.cwd().resolve()
    else:
        current = Path(start_path).expanduser().resolve()
        if current.is_file():
            current = current.parent

    for parent in (current, *current.parents):
        candidate = parent / SYSTEM_SERVICES_REL_PATH
        if candidate.is_file():
            return candidate
    return None


def load_system_services_config(
    start_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    config_path = find_system_services_config(start_path)
    if config_path is None:
        return {}
    return load_yaml_config(config_path)


def merge_system_services(
    config: dict[str, Any],
    start_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    merged = dict(config)
    system_config = load_system_services_config(start_path)
    for section in SYSTEM_SERVICE_SECTIONS:
        section_data = system_config.get(section)
        if isinstance(section_data, dict):
            merged[section] = dict(section_data)
    return merged


def load_config_with_system_services(
    config_path: str | os.PathLike[str],
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    return merge_system_services(config, config_path)
