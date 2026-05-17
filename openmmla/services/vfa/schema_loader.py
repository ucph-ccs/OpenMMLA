"""helpers for loading centralized VFA action schemas."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(slots=True)
class VFAActionSchema:
    """resolved prompt-side action schema for the VFA server."""

    schema_name: str
    schema_path: str
    action_definitions: dict[str, str]
    decision_process: str


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _find_repo_root(start_path: str | None = None) -> str:
    current = os.path.abspath(start_path or os.getcwd())
    while True:
        if os.path.isfile(os.path.join(current, "pyproject.toml")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.getcwd()
        current = parent


def _resolve_path(path: str | None, repo_root: str, project_dir: str | None = None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path

    repo_candidate = os.path.join(repo_root, path)
    if os.path.exists(repo_candidate):
        return repo_candidate

    if project_dir:
        project_candidate = os.path.join(project_dir, path)
        if os.path.exists(project_candidate):
            return project_candidate

    return repo_candidate


def load_vfa_action_schema(
    analyzer_config: dict[str, Any],
    project_dir: str | None,
) -> VFAActionSchema:
    """load centralized task-independent VFA action schema."""

    legacy_keys = {"action_definitions", "decision_process"} & analyzer_config.keys()
    if legacy_keys:
        raise ValueError(
            "Inline VFA action definitions are no longer supported. "
            "Move action labels and decision_process into config/vfa/action_schemas.yml "
            "and set VLLMFrameAnalyzer.action_schema when a non-default schema is needed."
        )

    repo_root = _find_repo_root(project_dir or os.getcwd())
    schema_path = _resolve_path(
        analyzer_config.get("action_schema_path", "config/vfa/action_schemas.yml"),
        repo_root=repo_root,
        project_dir=project_dir,
    )
    schema_name = analyzer_config.get("action_schema")

    if not schema_path or not os.path.exists(schema_path):
        raise FileNotFoundError(
            "VFA action schema file not found. Set VLLMFrameAnalyzer.action_schema_path "
            "to config/vfa/action_schemas.yml or another centralized schema file."
        )

    schema_data = _load_yaml(schema_path)
    schema_name = schema_name or schema_data.get("default_schema")
    schemas = schema_data.get("schemas", {})
    if schema_name not in schemas:
        raise ValueError(f"Unknown VFA action schema '{schema_name}' in {schema_path}")
    schema = schemas[schema_name]
    labels = schema.get("labels", {})
    if not labels:
        raise ValueError(f"Schema '{schema_name}' in {schema_path} has no labels")

    return VFAActionSchema(
        schema_name=schema_name,
        schema_path=schema_path,
        action_definitions={str(key): str(value) for key, value in labels.items()},
        decision_process=str(schema.get("decision_process", "")),
    )
