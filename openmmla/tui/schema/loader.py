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
    entry_schema: dict = field(default_factory=dict)


@dataclass
class PipelineDef:
    name: str
    template_path: str
    config_path: str
    fields: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    base_template: list = field(default_factory=list)
    base_section: str = ""
    base_template_prefix: str = ""


@dataclass
class StreamDef:
    name: str
    ssh_profile: str
    device: str
    target: str
    codec: str = ""
    resolution: str = ""
    fps: int = 30
    format: str = ""
    rate: int = 0
    channels: int = 0


def _clean_stream_optional(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in ("", "Select.NULL", "None", "null"):
        return ""
    return text


def streams_from_config(data: dict) -> list[StreamDef]:
    """load stream definitions from a parsed pipeline config."""
    streams_data = data.get("Streams", {})
    if not isinstance(streams_data, dict):
        return []
    streams = []
    for name, props in streams_data.items():
        if not isinstance(props, dict):
            continue
        ssh_profile = _clean_stream_optional(props.get("ssh_profile", ""))
        device = _clean_stream_optional(props.get("device", ""))
        target = props.get("target", "")
        if not target:
            continue
        streams.append(StreamDef(
            name=name,
            ssh_profile=ssh_profile,
            device=device,
            target=target,
            codec=props.get("codec", ""),
            resolution=props.get("resolution", ""),
            fps=int(props.get("fps", 30)),
            format=props.get("format", ""),
            rate=int(props.get("rate", 0)),
            channels=int(props.get("channels", 0)),
        ))
    return streams


def load_streams(config_path: str) -> list[StreamDef]:
    """load stream definitions from a pipeline config.yml."""
    return streams_from_config(load_existing_config(config_path))


def get_stream_targets(config_path: str, protocol: str = "rtmp") -> list[str]:
    """extract target URLs from Streams config, filtered by protocol prefix."""
    streams = load_streams(config_path)
    return [s.target for s in streams if s.target.startswith(f"{protocol}://")]


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
    """extract inline comments keyed by dot-path (e.g. 'SpeechTranscriber.azure.model')."""
    comments = {}
    path_stack: list[tuple[int, str]] = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                key_match = re.match(r'^(\s*)([A-Za-z_<][\w\-\[\]<>]*)\s*:', line)
                if not key_match:
                    continue
                indent = len(key_match.group(1))
                key_name = key_match.group(2)
                while path_stack and path_stack[-1][0] >= indent:
                    path_stack.pop()
                path_stack.append((indent, key_name))
                dot_path = ".".join(k for _, k in path_stack)
                comment_match = INLINE_COMMENT_RE.match(line)
                desc = comment_match.group(1) if comment_match else ""
                comments[dot_path] = desc
    except OSError:
        pass
    return comments


def _infer_entry_schema(entries):
    """infer dict-entry schema from a list of dicts, preferring non-placeholder values."""
    schema = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for k, v in entry.items():
            if k not in schema:
                schema[k] = _infer_type(v)
            elif schema[k] == "str" and not _is_placeholder(v):
                schema[k] = _infer_type(v)
    return schema


def _walk_yaml(data, path_parts, section, comments, fields, indent_level=0):
    """recursively walk yaml dict and collect leaf FieldDefs."""
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if PLACEHOLDER_RE.match(str(key)):
            continue
        current_path = path_parts + [str(key)]
        dot_path = ".".join(current_path)
        desc = comments.get(dot_path, "")

        if isinstance(value, dict):
            sub_section = f"{section}.{key}"
            _walk_yaml(value, current_path, sub_section, comments, fields, indent_level + 1)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            schema = _infer_entry_schema(value)
            fields.append(FieldDef(
                path=dot_path,
                field_type="list_of_dicts",
                default=value,
                description=desc,
                required=False,
                section=section,
                entry_schema=schema,
            ))
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


def fields_from_config_section(section_key, section_value):
    """generate FieldDefs from an arbitrary config section not in the template."""
    fields = []
    if isinstance(section_value, dict):
        _walk_yaml(section_value, [section_key], section_key, {}, fields)
    return fields


def _find_placeholder_child(section_value):
    """find a placeholder child key only when it is the sole child.

    This distinguishes a genuine base template section (e.g. Base: {<Device>: ...})
    from a section that merely includes a placeholder example alongside real children
    (e.g. upstreams: {infer: ..., <your-service>: ...}).
    """
    if not isinstance(section_value, dict):
        return None
    placeholder = None
    for child_key in section_value:
        if PLACEHOLDER_RE.match(str(child_key)):
            placeholder = str(child_key)
        else:
            return None
    return placeholder


def load_template(filepath):
    """parse a config_template.yml and return fields, sections, base_template, and base metadata."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    comments = _extract_comments(filepath)
    fields = []
    base_template = []
    base_section = ""
    base_template_prefix = ""
    sections = list(data.keys())

    for section_key, section_value in data.items():
        if isinstance(section_value, dict):
            tpl_child = _find_placeholder_child(section_value)
            if tpl_child is not None:
                child_value = section_value[tpl_child]
                base_section = section_key
                base_template_prefix = f"{section_key}.{tpl_child}"
                if isinstance(child_value, dict):
                    _walk_yaml(
                        child_value,
                        [section_key, tpl_child],
                        base_template_prefix,
                        comments,
                        base_template,
                        indent_level=0,
                    )
                continue
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

    return fields, sections, base_template, base_section, base_template_prefix


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
        ("ASR Base", "pipelines/asr-base"),
        ("VFA Base", "pipelines/vfa-base"),
        ("IPS Base", "pipelines/ips-base"),
        ("ASR Server", "pipelines/asr-server"),
        ("VFA Server", "pipelines/vfa-server"),
        ("Nginx", "pipelines/uber-server/nginx"),
        ("Flask Backend", "pipelines/uber-server/dashboard/flask-backend"),
    ]

    for name, rel_dir in registry:
        template = os.path.join(root, rel_dir, "config_template.yml")
        config = os.path.join(root, rel_dir, "config.yml")
        if not os.path.isfile(template):
            continue
        fields, sections, base_template, base_sec, base_tpl_prefix = load_template(template)
        pipelines.append(PipelineDef(
            name=name,
            template_path=template,
            config_path=config,
            fields=fields,
            sections=sections,
            base_template=base_template,
            base_section=base_sec,
            base_template_prefix=base_tpl_prefix,
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
