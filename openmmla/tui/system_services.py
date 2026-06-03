from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from openmmla.tui.schema.definitions import SHARED_SECTIONS, get_shared_defaults
from openmmla.tui.schema.loader import load_existing_config


SYSTEM_SERVICES_REL_PATH = os.path.join("config", "system_services.yml")

SYSTEM_SERVICE_SOURCE_CONFIG_RELS = (
    os.path.join("pipelines", "asr-base", "config.yml"),
    os.path.join("pipelines", "vfa-base", "config.yml"),
    os.path.join("pipelines", "ips-base", "config.yml"),
    os.path.join("pipelines", "uber-server", "dashboard", "flask-backend", "config.yml"),
)


def system_services_config_path(root: str | os.PathLike[str]) -> str:
    return str(Path(root) / SYSTEM_SERVICES_REL_PATH)


def usable_system_service_value(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text and "<" not in text and ">" not in text)


def _section_from_flat_values(section_name: str, values: dict[str, object]) -> dict[str, object]:
    info = SHARED_SECTIONS.get(section_name, {})
    section_data: dict[str, object] = {}
    for key, field in info.get("fields", {}).items():
        path = f"{section_name}.{key}"
        value = values.get(path, field.get("default", ""))
        if value is not None:
            section_data[key] = value
    return section_data


def flat_values_to_config(values: dict[str, object]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for section_name in SHARED_SECTIONS:
        section_data = _section_from_flat_values(section_name, values)
        if section_data:
            config[section_name] = section_data
    return config


def config_to_flat_values(config: dict[str, Any], *, include_defaults: bool = True) -> dict[str, object]:
    values = get_shared_defaults() if include_defaults else {}
    for section_name, info in SHARED_SECTIONS.items():
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, field in info.get("fields", {}).items():
            value = section.get(key)
            if value is None and include_defaults:
                value = field.get("default", "")
            if value is not None:
                values[f"{section_name}.{key}"] = value
    return values


def load_system_services_config(root: str | os.PathLike[str]) -> dict[str, Any]:
    return load_existing_config(system_services_config_path(root))


def harvest_system_services_from_pipeline_configs(root: str | os.PathLike[str]) -> dict[str, object]:
    root_path = Path(root)
    values = get_shared_defaults()
    seen: set[str] = set()
    for rel_path in SYSTEM_SERVICE_SOURCE_CONFIG_RELS:
        config_path = root_path / rel_path
        if not config_path.is_file():
            continue
        config = load_existing_config(str(config_path))
        for section_name, info in SHARED_SECTIONS.items():
            section = config.get(section_name)
            if not isinstance(section, dict):
                continue
            for key in info.get("fields", {}):
                path = f"{section_name}.{key}"
                if path in seen:
                    continue
                value = section.get(key)
                if usable_system_service_value(value):
                    values[path] = value
                    seen.add(path)
    return values


def load_system_service_values(root: str | os.PathLike[str]) -> dict[str, object]:
    config = load_system_services_config(root)
    if config:
        return config_to_flat_values(config)
    return harvest_system_services_from_pipeline_configs(root)


def save_system_services_config(root: str | os.PathLike[str], values: dict[str, object]) -> str:
    config_path = system_services_config_path(root)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(flat_values_to_config(values), file, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return config_path


def save_system_service_section(
    root: str | os.PathLike[str],
    section_name: str,
    section_data: dict[str, object],
) -> str:
    config = load_system_services_config(root)
    if not isinstance(config, dict):
        config = {}
    config[section_name] = dict(section_data)
    values = config_to_flat_values(config)
    return save_system_services_config(root, values)
