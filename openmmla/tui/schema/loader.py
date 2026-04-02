import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml


PLACEHOLDER_RE = re.compile(r'^<.*>$')
URL_PLACEHOLDER_RE = re.compile(r'https?://.*<.*>')
INLINE_COMMENT_RE = re.compile(r'^[^#]*#\s*(.*?)\s*$')


@dataclass
class FieldDef:
    path: str
    field_type: str
    default: Any
    description: str
    required: bool
    section: str
    choices: list = field(default_factory=list)


@dataclass
class PipelineDef:
    name: str
    template_path: str
    config_path: str
    fields: list = field(default_factory=list)
    sections: list = field(default_factory=list)


def _infer_type(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        if URL_PLACEHOLDER_RE.match(value) or value.startswith("http"):
            return "url"
        if PLACEHOLDER_RE.match(value):
            return "str"
    return "str"


def _is_placeholder(value):
    if isinstance(value, str):
        return bool(PLACEHOLDER_RE.match(value)) or bool(URL_PLACEHOLDER_RE.match(value))
    return value is None


def _extract_comments(filepath):
    """extract inline comments keyed by the yaml key on that line."""
    comments = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                key_match = re.match(r'^(\s*)([A-Za-z_][\w\-\[\]]*)\s*:', line)
                if not key_match:
                    continue
                indent = len(key_match.group(1))
                key_name = key_match.group(2)
                comment_match = INLINE_COMMENT_RE.match(line)
                desc = comment_match.group(1) if comment_match else ""
                comments[(indent, key_name)] = desc
    except OSError:
        pass
    return comments


def _walk_yaml(data, path_parts, section, comments, fields, indent_level=0):
    """recursively walk yaml dict and collect leaf FieldDefs."""
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        current_path = path_parts + [str(key)]
        dot_path = ".".join(current_path)
        child_indent = (indent_level + 1) * 2
        desc = comments.get((child_indent, str(key)), "")
        if desc == "":
            desc = comments.get((indent_level * 2, str(key)), "")

        if isinstance(value, dict):
            _walk_yaml(value, current_path, section, comments, fields, indent_level + 1)
        else:
            ft = _infer_type(value)
            req = _is_placeholder(value)
            fields.append(FieldDef(
                path=dot_path,
                field_type=ft,
                default=value,
                description=desc,
                required=req,
                section=section,
            ))


def load_template(filepath):
    """parse a config_template.yml and return list of FieldDef and section names."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    comments = _extract_comments(filepath)
    fields = []
    sections = list(data.keys())

    for section_key, section_value in data.items():
        if isinstance(section_value, dict):
            _walk_yaml(
                section_value,
                [section_key],
                section_key,
                comments,
                fields,
                indent_level=0,
            )
        else:
            ft = _infer_type(section_value)
            req = _is_placeholder(section_value)
            desc = comments.get((0, section_key), "")
            fields.append(FieldDef(
                path=section_key,
                field_type=ft,
                default=section_value,
                description=desc,
                required=req,
                section=section_key,
            ))

    return fields, sections


def _find_project_root():
    """walk up from this file to find the repo root containing pyproject.toml."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


def discover_pipelines():
    """find all config_template.yml files and build PipelineDef list."""
    root = _find_project_root()
    pipelines = []

    registry = [
        ("ASR Base", "base_stations/asr"),
        ("VFA Base", "base_stations/vfa"),
        ("IPS Base", "base_stations/ips"),
        ("ASR Server", "servers/asr"),
        ("VFA Server", "servers/vfa"),
    ]

    for name, rel_dir in registry:
        template = os.path.join(root, rel_dir, "config_template.yml")
        config = os.path.join(root, rel_dir, "config.yml")
        if not os.path.isfile(template):
            continue
        fields, sections = load_template(template)
        pipelines.append(PipelineDef(
            name=name,
            template_path=template,
            config_path=config,
            fields=fields,
            sections=sections,
        ))

    return pipelines


def load_existing_config(config_path):
    """load an existing config.yml and return its data dict, or empty dict if missing."""
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def get_nested_value(data, dot_path):
    """retrieve a value from nested dict by dot-separated path."""
    keys = dot_path.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


def set_nested_value(data, dot_path, value):
    """set a value in nested dict by dot-separated path, creating intermediate dicts."""
    keys = dot_path.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def save_config(config_path, fields, values):
    """build a yaml dict from field values and write to config_path."""
    data = {}
    for f in fields:
        val = values.get(f.path, f.default)
        if val is not None:
            set_nested_value(data, f.path, val)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
