from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import yaml
from rich.markup import escape as rich_escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.widget import Widget
from textual.widgets import (
    Static, Tree, Button, Select, Input, Label, TabbedContent, TabPane, TextArea,
)

from openmmla.tui.schema.loader import (
    FieldDef as LoaderFieldDef,
    discover_pipelines, load_existing_config, get_nested_value,
    save_config, PipelineDef, _find_project_root, fields_from_config_section,
    load_streams, streams_from_config,
)
from openmmla.tui.schema.definitions import (
    SHARED_SECTIONS, SHARED_SECTION_NAMES, apply_shared_values,
)
from openmmla.tui.system_services import (
    SYSTEM_SERVICE_SOURCE_CONFIG_RELS,
    load_system_service_values,
    load_system_services_config,
    save_system_service_section,
    pipeline_section_overrides,
    shared_section_drift,
)
from openmmla.tui.ssh import (
    REFRESH_TARGETS_OPTION, TARGET_STATES, is_select_sentinel, probe_all_profiles, probe_ssh_endpoint, summarize_states, target_options, target_state_label,
    load_ssh_profiles, get_profile_by_name, ssh_run_sync,
    scp_file_async, scp_from_remote_async, ssh_run_async, ssh_check_port, ssh_check_tmux,
    ssh_test_connection,
    wrap_local, wrap_remote,
)
from openmmla.tui.artifacts import (
    collection_artifact_dir, merge_tree, pipeline_artifact_dir,
    safe_segment, update_collection_manifest, update_pipeline_manifest,
)
from openmmla.utils.artifact_paths import NON_SESSION_ARTIFACT_DIRS
from openmmla.collection.recording import (
    DEFAULT_AUDIO_CHANNEL,
    DEFAULT_AUDIO_DEVICE_LINUX,
    DEFAULT_AUDIO_DEVICE_MACOS,
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_AUDIO_INPUT_FORMAT_LINUX,
    DEFAULT_AUDIO_INPUT_FORMAT_MACOS,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_VIDEO_BITRATE_LINUX,
    DEFAULT_VIDEO_BITRATE_MACOS,
    DEFAULT_VIDEO_BUFSIZE_LINUX,
    DEFAULT_VIDEO_BUFSIZE_MACOS,
    DEFAULT_VIDEO_DEVICE_LINUX,
    DEFAULT_VIDEO_DEVICE_MACOS,
    DEFAULT_VIDEO_FRAMERATE,
    DEFAULT_VIDEO_INPUT_FORMAT_LINUX,
    DEFAULT_VIDEO_INPUT_FORMAT_MACOS,
    DEFAULT_VIDEO_MAXRATE_LINUX,
    DEFAULT_VIDEO_MAXRATE_MACOS,
    DEFAULT_VIDEO_PRESET,
    DEFAULT_VIDEO_SIZE,
    DEFAULT_VIDEO_SOURCE_FORMAT_LINUX,
    DEFAULT_VIDEO_SOURCE_FORMAT_MACOS,
)
from openmmla.utils.experiments import (
    get_active_experiments, get_groups_for_experiment, get_participant_aliases,
    load_experiments,
)
from openmmla.tui.widgets.command_session import CommandSession
from openmmla.tui.widgets.config_form import ConfigForm
from openmmla.tui.widgets.experiment_form import ExperimentForm
from openmmla.tui.widgets.service_card import ServiceCard, ServiceDef, ParamDef, ComponentDef
from openmmla.tui.widgets.ssh_form import SSHForm
from openmmla.tui.widgets.stream_panel import StreamPanel
from openmmla.tui.widgets.session_control import SessionControlPanel
from openmmla.tui.widgets.task_form import TaskForm


_SVC_PIPELINE_NAMES: dict[str, str] = {
    "Uber: Nginx": "Nginx",
    "Uber: Flask": "Flask Backend",
}

_STREAM_PIPELINES = {"ASR Base", "IPS Base", "VFA Base"}

_GLOBAL_DEFAULT_NAV_ORDER = (
    "SSH Profiles",
    "Experiments",
    "Tasks",
    "MongoDB",
    "InfluxDB",
    "MQTT",
    "Redis",
    "Gateway",
)
_SYSTEM_SERVICES_LABEL = "System Services"

_MLLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_MLLM_PORT = 8010
_MLLM_MAX_MODEL_LEN = 8192
_MLLM_IMAGE_LIMIT = '{"image":4}'
_MLLM_GPU_MEMORY_UTILIZATION = "0.80"
_MLLM_API_KEY = "EMPTY"
_MLLM_CONFIG_REL_PATH = os.path.join("config", "mllm_server.yml")
_REMOTE_COLLECTION_RUNTIME = "~/.openmmla/collection-runtime"
_REMOTE_COLLECTION_RUNTIME_ENV = "$HOME/.openmmla/collection-runtime"

_ARTIFACT_CONFIG_RELS = SYSTEM_SERVICE_SOURCE_CONFIG_RELS

_REMOTE_COLLECTION_FILES = (
    "openmmla/__init__.py",
    "openmmla/collection/__init__.py",
    "openmmla/collection/recording.py",
    "openmmla/commands/__init__.py",
    "openmmla/commands/collect/__init__.py",
    "openmmla/commands/collect/audio.py",
    "openmmla/commands/collect/video.py",
    "openmmla/utils/__init__.py",
    "openmmla/utils/artifact_paths.py",
)
_NEW_COLLECTION_SESSION_CHOICE = "Create MongoDB Session"
_COLLECTION_HIDDEN_PRESET_FLAGS = {
    "--audio-interactive",
    "--audio-input-format",
    "--audio-device",
    "--audio-channels",
    "--audio-channel",
    "--sample-rate",
    "--audio-format",
    "--video-interactive",
    "--video-input-format",
    "--video-device",
    "--video-source-format",
    "--framerate",
    "--size",
    "--bitrate",
    "--maxrate",
    "--bufsize",
    "--preset",
    "--camera-label",
}
_LAUNCHER_UI_WORKER_GROUP = "launcher-ui"
_LAUNCHER_STATUS_WORKER_GROUP = "launcher-status"
_LAUNCHER_DOWNLOAD_WORKER_GROUP = "launcher-downloads"
_LAUNCHER_REMOTE_DELETE_WORKER_GROUP = "launcher-remote-delete"
_LAUNCHER_REMOTE_STOP_WORKER_GROUP = "launcher-remote-stop"
_LAUNCHER_COLLECTION_STOP_WORKER_GROUP = "launcher-collection-stop"

_MLLM_FIELDS = [
    LoaderFieldDef(
        path="server.model",
        field_type="str",
        default=_MLLM_MODEL,
        description="Hugging Face model id for vLLM serve",
        required=True,
        section="server",
    ),
    LoaderFieldDef(
        path="server.port",
        field_type="int",
        default=_MLLM_PORT,
        description="OpenAI-compatible API port",
        required=True,
        section="server",
    ),
    LoaderFieldDef(
        path="server.host",
        field_type="str",
        default="0.0.0.0",
        description="Bind host for vLLM",
        required=True,
        section="server",
    ),
    LoaderFieldDef(
        path="server.dtype",
        field_type="str",
        default="auto",
        description="vLLM dtype argument",
        required=True,
        section="server",
    ),
    LoaderFieldDef(
        path="server.max_model_len",
        field_type="int",
        default=_MLLM_MAX_MODEL_LEN,
        description="Maximum model context length",
        required=True,
        section="server",
    ),
    LoaderFieldDef(
        path="server.limit_mm_per_prompt",
        field_type="str",
        default=_MLLM_IMAGE_LIMIT,
        description='vLLM multimodal limit JSON, e.g. {"image":4}',
        required=True,
        section="server",
    ),
    LoaderFieldDef(
        path="server.gpu_memory_utilization",
        field_type="float",
        default=float(_MLLM_GPU_MEMORY_UTILIZATION),
        description="GPU memory utilization fraction",
        required=True,
        section="server",
    ),
    LoaderFieldDef(
        path="server.api_key",
        field_type="str",
        default=_MLLM_API_KEY,
        description="OpenAI-compatible API key",
        required=True,
        section="server",
    ),
]

_STREAM_FIELDS_TEMPLATE = [
    ("target", "str", "", "rtmp://<host>/<app>/<stream> or udp://<host>:<port>", False),
    ("ssh_profile", "str", "", "SSH profile for remote stream management", True),
    ("device", "str", "", "device path, e.g. /dev/video0 (video) or hw:1,0 (audio)", False),
]


def _mllm_config_path(root: str) -> str:
    return os.path.join(root, _MLLM_CONFIG_REL_PATH)


def _mllm_form_values(root: str) -> dict:
    existing = load_existing_config(_mllm_config_path(root))
    values = {}
    for field in _MLLM_FIELDS:
        existing_value = get_nested_value(existing, field.path)
        values[field.path] = field.default if existing_value is None else existing_value
    return values


def _coerce_int(value, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _quote_remote_path(path: str) -> str:
    text = str(path).strip()
    if text == "~" or text == "$HOME":
        return "$HOME"
    if text.startswith("~/"):
        return _remote_path_join("$HOME", *(shlex.quote(part) for part in text[2:].split("/") if part))
    if text.startswith("$HOME/"):
        return _remote_path_join("$HOME", *(shlex.quote(part) for part in text[6:].split("/") if part))
    return shlex.quote(text)


def _remote_path_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[part.strip("/") for part in parts if part]])


def _replace_loopback_url(url: object, host: str) -> object:
    text = str(url or "").strip()
    if not text:
        return url
    try:
        parts = urlsplit(text)
    except ValueError:
        return url
    if parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return url

    userinfo = ""
    if parts.username:
        userinfo = quote(unquote(parts.username), safe="")
        if parts.password is not None:
            userinfo += f":{quote(unquote(parts.password), safe='')}"
        userinfo += "@"
    host_part = f"[{host}]" if ":" in host and not host.startswith("[") else host
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit((parts.scheme, f"{userinfo}{host_part}{port}", parts.path, parts.query, parts.fragment))


def _config_for_local_db_access(config: dict, profile=None) -> dict:
    local_config = copy.deepcopy(config)
    if profile is None:
        return local_config
    for section in ("MongoDB", "InfluxDB"):
        section_config = local_config.get(section)
        if isinstance(section_config, dict) and "url" in section_config:
            section_config["url"] = _replace_loopback_url(section_config["url"], profile.host)
    return local_config


def _remote_home(profile) -> str | None:
    try:
        result = ssh_run_sync(profile, 'printf "%s" "$HOME"', timeout=10.0)
    except Exception:
        return None
    home = (result.stdout or "").strip()
    if result.returncode == 0 and home.startswith("/"):
        return home.rstrip("/") or "/"
    return None


def _expand_remote_home_path(path: str, remote_home: str | None) -> str:
    text = str(path).strip()
    if not remote_home:
        return text
    home = remote_home.rstrip("/") or "/"
    if text == "~" or text == "$HOME":
        return home
    if text.startswith("~/"):
        return _remote_path_join(home, text[2:])
    if text.startswith("$HOME/"):
        return _remote_path_join(home, text[6:])
    return text


def _safe_session_id(value: str | None, default: str = "") -> str:
    raw = str(value or "").strip()
    if not raw:
        return default
    if raw in {".", ".."} or "/" in raw or "\\" in raw:
        return ""
    segment = safe_segment(raw, "")
    if segment:
        return segment
    return default


def _is_new_collection_session_choice(value: object) -> bool:
    return str(value or "").strip() in {_NEW_COLLECTION_SESSION_CHOICE, "New Session"}


def _non_self_matching_regex(pattern: str) -> str:
    """escape a process pattern while preventing pkill -f from matching itself."""
    parts = []
    replaced = False
    for char in str(pattern):
        if not replaced and char.isalnum():
            parts.append(f"[{re.escape(char)}]")
            replaced = True
        else:
            parts.append(re.escape(char))
    return "".join(parts) if parts else pattern


def _non_self_matching_process_pattern(pattern: str) -> str:
    """prevent pgrep/pkill patterns from matching their own shell command."""
    parts = []
    replaced = False
    for char in str(pattern):
        if not replaced and char.isalnum():
            parts.append(f"[{char}]")
            replaced = True
        else:
            parts.append(char)
    return "".join(parts) if parts else pattern


def _mllm_config(root: str, values: dict | None = None) -> dict:
    raw = values or _mllm_form_values(root)
    return {
        "model": str(raw.get("server.model") or _MLLM_MODEL),
        "port": _coerce_int(raw.get("server.port"), _MLLM_PORT),
        "host": str(raw.get("server.host") or "0.0.0.0"),
        "dtype": str(raw.get("server.dtype") or "auto"),
        "max_model_len": _coerce_int(raw.get("server.max_model_len"), _MLLM_MAX_MODEL_LEN),
        "limit_mm_per_prompt": str(raw.get("server.limit_mm_per_prompt") or _MLLM_IMAGE_LIMIT),
        "gpu_memory_utilization": _coerce_float(
            raw.get("server.gpu_memory_utilization"),
            float(_MLLM_GPU_MEMORY_UTILIZATION),
        ),
        "api_key": str(raw.get("server.api_key") or _MLLM_API_KEY),
    }


def _asr_server_config_path(root: str) -> str:
    return os.path.join(root, "pipelines", "asr-server", "config.yml")


def _asr_audio_inferer_backend_from_config(config: dict) -> str:
    backend = get_nested_value(config, "AudioInferer.backend")
    if backend is None:
        return "nemo"
    return str(backend).strip().lower() or "nemo"


def _asr_audio_inferer_backend(root: str) -> str:
    return _asr_audio_inferer_backend_from_config(
        load_existing_config(_asr_server_config_path(root))
    )


def _asr_server_conda_env_from_config(config: dict) -> str:
    if _asr_audio_inferer_backend_from_config(config) == "wespeaker":
        return "asr-server-wespeaker"
    return "asr-server-nemo"


def _asr_server_conda_env(root: str) -> str:
    return _asr_server_conda_env_from_config(
        load_existing_config(_asr_server_config_path(root))
    )


def _artifact_session_choices(root: str) -> list[str]:
    configs = []
    for rel_path in _ARTIFACT_CONFIG_RELS:
        config_path = os.path.join(root, rel_path)
        if os.path.isfile(config_path):
            configs.append(load_existing_config(config_path))
    local_sessions = _local_artifact_session_ids(root) + _local_collection_session_ids(root)
    return _artifact_session_choices_from_configs(configs, local_sessions)


def _artifact_session_choices_from_configs(configs: list[dict], local_sessions: list[str] | None = None) -> list[str]:
    choices: list[str] = []
    seen: set[str] = set()

    def add_many(values: list[str]) -> None:
        for value in values:
            session_id = str(value or "").strip()
            if not session_id or session_id in seen:
                continue
            seen.add(session_id)
            choices.append(session_id)

    # most pipeline configs point at the same databases; query each unique
    # endpoint only once instead of once per config
    seen_mongo: set[tuple[str, str]] = set()
    seen_influx: set[tuple[str, str, str, str]] = set()
    for config in configs:
        if not isinstance(config, dict):
            continue
        mongo_cfg = config.get("MongoDB")
        if isinstance(mongo_cfg, dict):
            mongo_sig = (str(mongo_cfg.get("url") or ""), str(mongo_cfg.get("db") or ""))
            if mongo_sig not in seen_mongo:
                seen_mongo.add(mongo_sig)
                add_many(_mongodb_session_ids_from_config(config))
        influx_cfg = config.get("InfluxDB")
        if isinstance(influx_cfg, dict):
            influx_sig = (
                str(influx_cfg.get("url") or ""),
                str(influx_cfg.get("token") or ""),
                str(influx_cfg.get("org") or ""),
                str(influx_cfg.get("bucket") or ""),
            )
            if influx_sig not in seen_influx:
                seen_influx.add(influx_sig)
                add_many(_influxdb_session_ids_from_config(config))

    add_many(local_sessions or [])
    return choices[:100]


def _mongodb_session_ids(config_path: str) -> list[str]:
    return _mongodb_session_ids_from_config(load_existing_config(config_path))


def _mongodb_session_ids_from_config(config: dict) -> list[str]:
    try:
        from pymongo import MongoClient
        from openmmla.utils.constants import MONGODB_DEFAULT_DB

        mongo_config = config.get("MongoDB", {})
        if not isinstance(mongo_config, dict):
            return []
        url = str(mongo_config.get("url") or "").strip()
        if not url or "<" in url:
            return []
        db_name = str(mongo_config.get("db") or MONGODB_DEFAULT_DB)
        client = MongoClient(
            url,
            serverSelectionTimeoutMS=800,
            connectTimeoutMS=800,
        )
        try:
            client.admin.command("ping")
            sessions = client[db_name]["sessions"]
            return [
                str(session.get("session_id"))
                for session in sessions.find({}, {"_id": 0, "session_id": 1}).sort("start_time", -1)
                if isinstance(session, dict) and session.get("session_id")
            ]
        finally:
            client.close()
    except Exception:
        return []


def _influxdb_session_ids(config_path: str) -> list[str]:
    return _influxdb_session_ids_from_config(load_existing_config(config_path))


def _influxdb_session_ids_from_config(config: dict) -> list[str]:
    try:
        from influxdb_client import InfluxDBClient
        from openmmla.utils.constants import INFLUXDB_DEFAULT_BUCKET, INFLUXDB_MEASUREMENT

        influx_config = config.get("InfluxDB", {})
        if not isinstance(influx_config, dict):
            return []
        url = str(influx_config.get("url") or "").strip()
        token = str(influx_config.get("token") or "").strip()
        org = str(influx_config.get("org") or "").strip()
        if not url or not token or not org or "<" in url or "<" in token or "<" in org:
            return []
        bucket = str(influx_config.get("bucket") or INFLUXDB_DEFAULT_BUCKET)
        client = InfluxDBClient(url=url, token=token, org=org, timeout=1000)
        try:
            query = f'''
                from(bucket: "{bucket}")
                |> range(start: -365d)
                |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}")
                |> keep(columns: ["session_id"])
                |> distinct(column: "session_id")
            '''
            result = client.query_api().query(org=org, query=query)
            session_ids = set()
            for table in result:
                for record in table.records:
                    session_id = record.values.get("session_id")
                    if session_id:
                        session_ids.add(str(session_id))
            return sorted(session_ids, reverse=True)
        finally:
            client.close()
    except Exception:
        return []


def _local_artifact_session_ids(root: str) -> list[str]:
    artifacts_dir = os.path.join(root, "artifacts")
    if not os.path.isdir(artifacts_dir):
        return []
    return sorted(
        [
            name for name in os.listdir(artifacts_dir)
            if name not in _NON_SESSION_ARTIFACT_NAMES
            and os.path.isdir(os.path.join(artifacts_dir, name))
        ],
        reverse=True,
    )


def _local_collection_session_ids(root: str) -> list[str]:
    collection_dir = os.path.join(root, "collection")
    if not os.path.isdir(collection_dir):
        return []
    return sorted(
        [
            name for name in os.listdir(collection_dir)
            if os.path.isdir(os.path.join(collection_dir, name))
        ],
        reverse=True,
    )


def _remote_artifact_session_ids(profile) -> list[str]:
    artifacts_dir = _remote_path_join(profile.remote_project_path, "artifacts")
    quoted_dir = _quote_remote_path(artifacts_dir)
    cmd = (
        f"if [ -d {quoted_dir} ]; then "
        f"find {quoted_dir} -mindepth 1 -maxdepth 1 -type d -exec basename {{}} \\; 2>/dev/null; "
        "fi"
    )
    try:
        result = ssh_run_sync(profile, cmd, timeout=8.0)
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return sorted(
        [
            line.strip()
            for line in (result.stdout or "").splitlines()
            if line.strip() and line.strip() not in _NON_SESSION_ARTIFACT_NAMES
        ],
        reverse=True,
    )


def _ips_transform_local_dir(root: str) -> str:
    return os.path.join(root, "pipelines", "ips-base", "camera_sync")


def _ips_cameras_local_dir(root: str) -> str:
    """directory where camera calibration captures per-camera image folders."""
    return os.path.join(root, "pipelines", "ips-base", "camera_calib", "cameras")


def _is_transform_matrix_file(name: str) -> bool:
    return name.startswith("transformation_matrices") and name.endswith(".json")


def _local_transform_matrix_files(local_dir: str) -> list[str]:
    if not os.path.isdir(local_dir):
        return []
    return sorted(
        name for name in os.listdir(local_dir)
        if _is_transform_matrix_file(name)
        and os.path.isfile(os.path.join(local_dir, name))
    )


def _remote_transform_matrix_files(profile, remote_dir: str) -> list[str]:
    quoted_dir = _quote_remote_path(remote_dir)
    cmd = (
        f"if [ -d {quoted_dir} ]; then "
        f"find {quoted_dir} -maxdepth 1 -type f -name 'transformation_matrices*.json' -exec basename {{}} \\; "
        "2>/dev/null; fi"
    )
    try:
        result = ssh_run_sync(profile, cmd, timeout=8.0)
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return sorted(
        line.strip()
        for line in (result.stdout or "").splitlines()
        if line.strip()
    )


_FILE_MISSING_SENTINEL = "__OPENMMLA_FILE_MISSING__"


def _remote_list_files(profile, remote_dir: str, suffix: str) -> list[str]:
    """list files in a remote directory matching *suffix (basename only)."""
    quoted = _quote_remote_path(remote_dir)
    cmd = (
        f"if [ -d {quoted} ]; then "
        f"find {quoted} -maxdepth 1 -type f -name '*{suffix}' -exec basename {{}} \\; "
        "2>/dev/null; fi"
    )
    try:
        result = ssh_run_sync(profile, cmd, timeout=8.0)
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return sorted(
        line.strip()
        for line in (result.stdout or "").splitlines()
        if line.strip() and not line.strip().startswith(".")
    )


def _remote_read_file(profile, remote_path: str) -> str | None:
    """read a remote file's contents, or None if missing/unreadable."""
    quoted = _quote_remote_path(remote_path)
    cmd = f"if [ -f {quoted} ]; then cat {quoted}; else printf '{_FILE_MISSING_SENTINEL}'; fi"
    try:
        result = ssh_run_sync(profile, cmd, timeout=10.0)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = result.stdout or ""
    if out.strip() == _FILE_MISSING_SENTINEL:
        return None
    return out


def _remote_write_file(profile, remote_path: str, content: str) -> tuple[bool, str]:
    """write content to a remote file (creating parent dirs). Returns (ok, error)."""
    quoted = _quote_remote_path(remote_path)
    remote_dir = remote_path.rsplit("/", 1)[0] if "/" in remote_path else "."
    cmd = f"mkdir -p {_quote_remote_path(remote_dir)} && cat > {quoted}"
    args = profile.base_ssh_args() + [cmd]
    try:
        proc = subprocess.run(
            args, input=content, capture_output=True, text=True, timeout=20.0
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or "").strip() or f"exit code {proc.returncode}"
    return True, ""


class TransformMatrixPanel(Widget):
    """IPS transform matrix file overview and sync controls."""

    DEFAULT_CSS = """
    TransformMatrixPanel {
        height: auto;
        padding: 1 2;
    }
    TransformMatrixPanel .tm-title {
        text-style: bold;
        margin-bottom: 1;
    }
    TransformMatrixPanel .tm-muted {
        color: $text-muted;
    }
    TransformMatrixPanel .tm-file {
        padding-left: 2;
    }
    TransformMatrixPanel .tm-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    TransformMatrixPanel .tm-actions Select {
        width: 1fr;
    }
    TransformMatrixPanel .tm-actions Button {
        min-width: 20;
        margin-left: 1;
    }
    TransformMatrixPanel #tm-editor {
        height: 22;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        *,
        local_dir: str,
        target: str,
        remote_dir: str | None,
        local_files: list[str],
        remote_files: list[str],
        ssh_profiles: list[str],
        ssh_profile=None,
    ) -> None:
        super().__init__()
        self.local_dir = local_dir
        self.target = target
        self.remote_dir = remote_dir
        self.local_files = local_files
        self.remote_files = remote_files
        self.ssh_profiles = ssh_profiles
        # when set (host is remote), the panel lists/reads/writes the matrix
        # files on the selected remote host instead of the local disk
        self._ssh_profile = ssh_profile
        self._current_file: str | None = None

    @property
    def _is_remote(self) -> bool:
        return (
            self.target != "local"
            and self._ssh_profile is not None
            and self.remote_dir is not None
        )

    @property
    def _dir(self) -> str:
        return self.remote_dir if self._is_remote else self.local_dir

    def _file_path(self, name: str) -> str:
        if self._is_remote:
            return _remote_path_join(self.remote_dir, name)
        return os.path.join(self.local_dir, name)

    def _tm_files(self) -> list[str]:
        return self.remote_files if self._is_remote else self.local_files

    def compose(self) -> ComposeResult:
        # make the active source unambiguous: when the host is remote, the list
        # and editor operate on that host's files (read/written over SSH); when
        # local, on the local disk.
        host_label = self.target if self._is_remote else "Local"
        files = self._tm_files()
        yield Static("[b]Transform Matrix[/b]", classes="tm-title")
        yield Static(f"Editing on {host_label}: {self._dir}", classes="tm-muted")
        if self._is_remote:
            yield Static(f"(Local copy: {self.local_dir})", classes="tm-muted")
        yield Static(
            "Generated by camera sync; edit here only to inspect/correct.",
            classes="tm-muted",
        )

        if files:
            yield Select(
                [(name, name) for name in files],
                prompt="Select a transform matrix file...",
                id="tm-file-select",
            )
            yield TextArea("", id="tm-editor", read_only=True)
            with Horizontal(classes="tm-actions"):
                yield Button("Save", variant="primary", id="btn-tm-save", disabled=True)
                yield Button("Reload", id="btn-tm-reload", disabled=True)
            yield Static("", id="tm-status", classes="tm-muted")
        else:
            where = host_label
            yield Static(
                f"No transformation_matrices*.json files found on {where}.",
                classes="tm-muted",
            )

        # Sync is only offered from the local host (push local -> remote).
        if self.target == "local":
            if self.ssh_profiles:
                with Horizontal(classes="tm-actions"):
                    yield Select(
                        [(name, name) for name in self.ssh_profiles],
                        prompt="Select SSH profile...",
                        id="transform-sync-profile-select",
                    )
                    yield Button("Sync to Remote", variant="warning", id="btn-sync-transform-remote")
            else:
                yield Static("No SSH profiles configured for sync.", classes="tm-muted")

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#tm-status", Static).update(text)
        except Exception:
            pass

    def _load_current(self) -> None:
        editor = self.query_one("#tm-editor", TextArea)
        if not self._current_file:
            editor.load_text("")
            editor.read_only = True
            return
        path = self._file_path(self._current_file)
        host_label = self.target if self._is_remote else "local"
        if self._is_remote:
            content = _remote_read_file(self._ssh_profile, path)
            if content is None:
                editor.load_text("")
                editor.read_only = True
                self._set_status(f"Could not read {self._current_file} on {host_label}")
                return
            editor.load_text(content)
            editor.read_only = False
            self._set_status(f"Loaded {self._current_file} from {host_label}")
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                editor.load_text(fh.read())
            editor.read_only = False
            self._set_status(f"Loaded {self._current_file} from {host_label}")
        except OSError as exc:
            editor.load_text("")
            editor.read_only = True
            self._set_status(f"Could not read {self._current_file}: {exc}")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "tm-file-select":
            return
        value = event.value
        self._current_file = None if value in (None, Select.BLANK) else str(value)
        has_file = self._current_file is not None
        self.query_one("#btn-tm-save", Button).disabled = not has_file
        self.query_one("#btn-tm-reload", Button).disabled = not has_file
        self._load_current()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-tm-reload":
            self._load_current()
        elif event.button.id == "btn-tm-save":
            if not self._current_file:
                return
            editor = self.query_one("#tm-editor", TextArea)
            text = editor.text
            try:
                json.loads(text)  # reject invalid JSON before writing
            except Exception as exc:
                self._set_status(f"Not saved — invalid JSON: {exc}")
                return
            path = self._file_path(self._current_file)
            host_label = self.target if self._is_remote else "local"
            if self._is_remote:
                ok, err = _remote_write_file(self._ssh_profile, path, text)
                if ok:
                    self._set_status(f"Saved {self._current_file} to {host_label}")
                else:
                    self._set_status(f"Save failed on {host_label}: {err}")
                return
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                self._set_status(f"Saved {self._current_file} to {host_label}")
            except OSError as exc:
                self._set_status(f"Save failed: {exc}")


class CameraManagerPanel(Widget):
    """Browse and delete camera calibration image folders (local host only).

    Calibration writes captured checkerboard images to
    camera_calib/cameras/<camera_name>/. This panel lists those folders, shows
    the images in a chosen camera, and lets the user delete a single image or a
    whole camera folder without leaving the TUI.
    """

    DEFAULT_CSS = """
    CameraManagerPanel {
        height: auto;
        padding: 1 2;
    }
    CameraManagerPanel .cm-title {
        text-style: bold;
        margin-bottom: 1;
    }
    CameraManagerPanel .cm-muted {
        color: $text-muted;
    }
    CameraManagerPanel .cm-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    CameraManagerPanel .cm-actions Button {
        min-width: 18;
        margin-right: 1;
    }
    CameraManagerPanel #cm-images {
        height: 12;
        margin-top: 1;
    }
    """

    def __init__(self, *, cameras_dir: str, target: str) -> None:
        super().__init__()
        self.cameras_dir = cameras_dir
        self.target = target
        self._current_camera: str | None = None

    # ── filesystem helpers ───────────────────────────────────────
    def _cameras(self) -> list[str]:
        try:
            return sorted(
                name for name in os.listdir(self.cameras_dir)
                if os.path.isdir(os.path.join(self.cameras_dir, name)) and not name.startswith(".")
            )
        except OSError:
            return []

    def _images(self, camera: str | None) -> list[str]:
        if not camera:
            return []
        cam_dir = os.path.join(self.cameras_dir, camera)
        try:
            return sorted(
                name for name in os.listdir(cam_dir)
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
        except OSError:
            return []

    # ── compose ──────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Static("[b]Calibration Cameras[/b]", classes="cm-title")
        if self.target != "local":
            yield Static(
                "Camera image management is available on the Local host only.",
                classes="cm-muted",
            )
            return
        yield Static(f"Folder: {self.cameras_dir}", classes="cm-muted")
        cameras = self._cameras()
        if not cameras:
            yield Static("No captured cameras yet (run a Capture from calibration).", classes="cm-muted")
            return
        yield Select(
            [(c, c) for c in cameras],
            prompt="Select a camera...",
            id="cm-camera-select",
        )
        yield Static("", id="cm-info", classes="cm-muted")
        yield Select([], prompt="Select an image...", id="cm-image-select")
        with Horizontal(classes="cm-actions"):
            yield Button("Delete Image", variant="error", id="btn-cm-del-image", disabled=True)
            yield Button("Delete Camera", variant="error", id="btn-cm-del-camera", disabled=True)
            yield Button("Refresh", id="btn-cm-refresh")
        yield Static("", id="cm-status", classes="cm-muted")

    # ── helpers ──────────────────────────────────────────────────
    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#cm-status", Static).update(text)
        except Exception:
            pass

    def _refresh_cameras(self) -> None:
        """re-list cameras after a deletion (rebuild the camera dropdown)."""
        try:
            sel = self.query_one("#cm-camera-select", Select)
            sel.set_options([(c, c) for c in self._cameras()])
            sel.value = Select.BLANK
        except Exception:
            pass
        self._current_camera = None
        self._refresh_images()
        try:
            self.query_one("#btn-cm-del-camera", Button).disabled = True
        except Exception:
            pass

    def _refresh_images(self) -> None:
        images = self._images(self._current_camera)
        try:
            img_sel = self.query_one("#cm-image-select", Select)
            img_sel.set_options([(n, n) for n in images])
            img_sel.value = Select.BLANK
        except Exception:
            pass
        try:
            self.query_one("#cm-info", Static).update(
                f"{self._current_camera}: {len(images)} image(s)" if self._current_camera else ""
            )
        except Exception:
            pass
        try:
            self.query_one("#btn-cm-del-image", Button).disabled = True
        except Exception:
            pass

    # ── events ───────────────────────────────────────────────────
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "cm-camera-select":
            self._current_camera = None if event.value is Select.BLANK else str(event.value)
            self._refresh_images()
            try:
                self.query_one("#btn-cm-del-camera", Button).disabled = self._current_camera is None
            except Exception:
                pass
        elif event.select.id == "cm-image-select":
            has = event.value not in (None, Select.BLANK)
            try:
                self.query_one("#btn-cm-del-image", Button).disabled = not has
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "btn-cm-refresh":
            event.stop()
            self._refresh_cameras()
            self._set_status("Refreshed.")
        elif bid == "btn-cm-del-image":
            event.stop()
            self._delete_image()
        elif bid == "btn-cm-del-camera":
            event.stop()
            self._delete_camera()

    def _safe_under_cameras(self, path: str) -> bool:
        root = os.path.abspath(self.cameras_dir)
        target = os.path.abspath(path)
        try:
            return os.path.commonpath([root, target]) == root and target != root
        except ValueError:
            return False

    def _delete_image(self) -> None:
        if not self._current_camera:
            return
        try:
            img = self.query_one("#cm-image-select", Select).value
        except Exception:
            return
        if img in (None, Select.BLANK):
            return
        path = os.path.join(self.cameras_dir, self._current_camera, str(img))
        if not self._safe_under_cameras(path):
            self._set_status("Refusing to delete a path outside the cameras folder.")
            return
        try:
            os.remove(path)
            self._set_status(f"Deleted image {img}")
        except OSError as exc:
            self._set_status(f"Delete failed: {exc}")
        self._refresh_images()

    def _delete_camera(self) -> None:
        if not self._current_camera:
            return
        path = os.path.join(self.cameras_dir, self._current_camera)
        if not self._safe_under_cameras(path):
            self._set_status("Refusing to delete a path outside the cameras folder.")
            return
        name = self._current_camera
        try:
            shutil.rmtree(path)
            self._set_status(f"Deleted camera folder '{name}'. Note: its entry under config 'Cameras' (if any) is left untouched.")
        except OSError as exc:
            self._set_status(f"Delete failed: {exc}")
        self._refresh_cameras()


class PromptsPanel(Widget):
    """VFA prompt template browser and editor."""

    DEFAULT_CSS = """
    PromptsPanel {
        height: auto;
        padding: 1 2;
    }
    PromptsPanel .pp-title {
        text-style: bold;
        margin-bottom: 1;
    }
    PromptsPanel .pp-muted {
        color: $text-muted;
    }
    PromptsPanel #prompt-editor {
        height: 24;
        margin-top: 1;
    }
    PromptsPanel .pp-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    PromptsPanel .pp-actions Button {
        min-width: 16;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        *,
        prompts_dir: str,
        active_files: list[str],
        profile: str,
        end_to_end: bool,
        target: str = "local",
        ssh_profiles: list[str] | None = None,
        ssh_profile=None,
        remote_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.prompts_dir = prompts_dir
        self.active_files = set(active_files)
        self.profile = profile
        self.end_to_end = end_to_end
        self.target = target
        self.ssh_profiles = ssh_profiles or []
        # When ssh_profile is set, the panel reads/writes prompt files on the
        # selected remote host instead of the local disk.
        self._ssh_profile = ssh_profile
        self._remote_dir = remote_dir
        self._current_file: str | None = None

    @property
    def _is_remote(self) -> bool:
        return self._ssh_profile is not None and self._remote_dir is not None

    @property
    def _dir(self) -> str:
        return self._remote_dir if self._is_remote else self.prompts_dir

    def _file_path(self, name: str) -> str:
        if self._is_remote:
            return _remote_path_join(self._remote_dir, name)
        return os.path.join(self.prompts_dir, name)

    def _prompt_files(self) -> list[str]:
        if self._is_remote:
            return _remote_list_files(self._ssh_profile, self._remote_dir, ".txt")
        try:
            return sorted(
                name for name in os.listdir(self.prompts_dir)
                if name.endswith(".txt") and not name.startswith(".")
            )
        except OSError:
            return []

    def compose(self) -> ComposeResult:
        host_label = self.target if self._is_remote else "Local"
        yield Static("[b]Prompt Templates[/b]", classes="pp-title")
        yield Static(f"{host_label}: {self._dir}", classes="pp-muted")
        yield Static(self._profile_line(), id="pp-profile-line", classes="pp-muted")
        files = self._prompt_files()
        if not files:
            yield Static("No .txt prompt templates found.", classes="pp-muted")
            return
        yield Select(
            self._build_options(),
            prompt="Select a prompt template...",
            id="prompt-file-select",
        )
        yield TextArea("", id="prompt-editor", read_only=True)
        with Horizontal(classes="pp-actions"):
            yield Button("Save", variant="primary", id="btn-prompt-save", disabled=True)
            yield Button("Reload", id="btn-prompt-reload", disabled=True)
        yield Static("", id="prompt-status", classes="pp-muted")
        # Sync is only offered from the local host (push local prompt files ->
        # remote). On a remote host no sync button is shown.
        if self.target == "local" and self.ssh_profiles:
            with Horizontal(classes="pp-actions"):
                yield Select(
                    [(name, name) for name in self.ssh_profiles],
                    prompt="Select SSH profile...",
                    id="prompts-sync-profile-select",
                )
                yield Button("Sync to Remote", variant="warning", id="btn-sync-prompts-remote")

    def _profile_line(self) -> str:
        mode = "end-to-end" if self.end_to_end else "two-step (VLM + LLM)"
        return (
            f"Active profile: [b]{self.profile}[/b] ({mode}) — "
            "change via Config > prompt_profile / end_to_end"
        )

    def _build_options(self) -> list[tuple[Text, str]]:
        """build Select options, highlighting the currently-active prompt files."""
        opts: list[tuple[Text, str]] = []
        for name in self._prompt_files():
            if name in self.active_files:
                opts.append((Text(f"● {name}  (active)", style="bold green"), name))
            else:
                opts.append((Text(name), name))
        return opts

    def refresh_active(self, active_files: list[str], profile: str, end_to_end: bool) -> None:
        """recompute the active prompt set after a config change and re-render."""
        self.active_files = set(active_files)
        self.profile = profile
        self.end_to_end = end_to_end
        try:
            self.query_one("#pp-profile-line", Static).update(self._profile_line())
        except Exception:
            pass
        try:
            self.query_one("#prompt-file-select", Select).set_options(self._build_options())
        except Exception:
            pass

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#prompt-status", Static).update(text)
        except Exception:
            pass

    def _load_current(self) -> None:
        editor = self.query_one("#prompt-editor", TextArea)
        if not self._current_file:
            editor.load_text("")
            editor.read_only = True
            return
        path = self._file_path(self._current_file)
        if self._is_remote:
            content = _remote_read_file(self._ssh_profile, path)
            if content is None:
                editor.load_text("")
                editor.read_only = True
                self._set_status(f"Could not read {self.target}:{path}")
                return
            editor.load_text(content)
            editor.read_only = False
            self._set_status(f"Loaded {self._current_file} from {self.target}")
            return
        try:
            with open(path, "r", encoding="utf-8") as file:
                editor.load_text(file.read())
            editor.read_only = False
            self._set_status(f"Loaded {self._current_file}")
        except OSError as exc:
            editor.load_text("")
            editor.read_only = True
            self._set_status(f"Could not read {self._current_file}: {exc}")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "prompt-file-select":
            return
        value = event.value
        self._current_file = None if value in (None, Select.BLANK) else str(value)
        has_file = self._current_file is not None
        self.query_one("#btn-prompt-save", Button).disabled = not has_file
        self.query_one("#btn-prompt-reload", Button).disabled = not has_file
        self._load_current()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-prompt-reload":
            self._load_current()
        elif event.button.id == "btn-prompt-save":
            if not self._current_file:
                return
            path = self._file_path(self._current_file)
            editor = self.query_one("#prompt-editor", TextArea)
            if self._is_remote:
                ok, err = _remote_write_file(self._ssh_profile, path, editor.text)
                if ok:
                    self._set_status(
                        f"Saved {self._current_file} to {self.target} (restart VFA Server to apply)"
                    )
                else:
                    self._set_status(f"Save to {self.target} failed: {err}")
                return
            try:
                with open(path, "w", encoding="utf-8") as file:
                    file.write(editor.text)
                self._set_status(f"Saved {self._current_file} (restart VFA Server to apply)")
            except OSError as exc:
                self._set_status(f"Save failed: {exc}")


class ActionSchemaPanel(Widget):
    """VFA action-schema editor (single centralized YAML file)."""

    DEFAULT_CSS = """
    ActionSchemaPanel {
        height: auto;
        padding: 1 2;
    }
    ActionSchemaPanel .as-title {
        text-style: bold;
        margin-bottom: 1;
    }
    ActionSchemaPanel .as-muted {
        color: $text-muted;
    }
    ActionSchemaPanel #action-schema-editor {
        height: 28;
        margin-top: 1;
    }
    ActionSchemaPanel .as-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    ActionSchemaPanel .as-actions Button {
        min-width: 16;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        *,
        schema_path: str,
        target: str = "local",
        ssh_profiles: list[str] | None = None,
        ssh_profile=None,
        remote_path: str | None = None,
    ) -> None:
        super().__init__()
        self.schema_path = schema_path
        self.target = target
        self.ssh_profiles = ssh_profiles or []
        # When ssh_profile is set, the panel reads/writes the schema on the
        # selected remote host instead of the local disk.
        self._ssh_profile = ssh_profile
        self._remote_path = remote_path

    @property
    def _is_remote(self) -> bool:
        return self._ssh_profile is not None and self._remote_path is not None

    @property
    def _path(self) -> str:
        return self._remote_path if self._is_remote else self.schema_path

    def compose(self) -> ComposeResult:
        host_label = self.target if self._is_remote else "Local"
        yield Static("[b]Action Schema[/b]", classes="as-title")
        yield Static(f"{host_label}: {self._path}", classes="as-muted")
        yield Static(
            "This file can define multiple named schemas under 'schemas:'; the one "
            "named by 'default_schema' (top of file) is the active one. Override "
            "per-pipeline with VLLMFrameAnalyzer.action_schema.",
            classes="as-muted",
        )
        yield TextArea("", id="action-schema-editor", read_only=True)
        with Horizontal(classes="as-actions"):
            yield Button("Save", variant="primary", id="btn-aschema-save")
            yield Button("Reload", id="btn-aschema-reload")
        yield Static("", id="aschema-status", classes="as-muted")
        # Sync is only offered from the local host (push local schema file ->
        # remote). On a remote host no sync button is shown.
        if self.target == "local" and self.ssh_profiles:
            with Horizontal(classes="as-actions"):
                yield Select(
                    [(name, name) for name in self.ssh_profiles],
                    prompt="Select SSH profile...",
                    id="aschema-sync-profile-select",
                )
                yield Button("Sync to Remote", variant="warning", id="btn-sync-aschema-remote")

    def on_mount(self) -> None:
        self._load()

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#aschema-status", Static).update(text)
        except Exception:
            pass

    def _load(self) -> None:
        editor = self.query_one("#action-schema-editor", TextArea)
        if self._is_remote:
            content = _remote_read_file(self._ssh_profile, self._remote_path)
            if content is None:
                editor.load_text("")
                editor.read_only = True
                self._set_status(f"Could not read {self.target}:{self._remote_path}")
                return
            editor.load_text(content)
            editor.read_only = False
            self._set_status(f"Loaded from {self.target}")
            return
        try:
            with open(self.schema_path, "r", encoding="utf-8") as fh:
                editor.load_text(fh.read())
            editor.read_only = False
            self._set_status(f"Loaded {os.path.basename(self.schema_path)}")
        except OSError as exc:
            editor.load_text("")
            editor.read_only = True
            self._set_status(f"Could not read {self.schema_path}: {exc}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-aschema-reload":
            self._load()
        elif event.button.id == "btn-aschema-save":
            editor = self.query_one("#action-schema-editor", TextArea)
            text = editor.text
            try:
                yaml.safe_load(text)  # reject invalid YAML before writing
            except yaml.YAMLError as exc:
                self._set_status(f"Not saved — invalid YAML: {exc}")
                return
            if self._is_remote:
                ok, err = _remote_write_file(self._ssh_profile, self._remote_path, text)
                if ok:
                    self._set_status(f"Saved to {self.target} (restart VFA Server to apply)")
                else:
                    self._set_status(f"Save to {self.target} failed: {err}")
                return
            try:
                os.makedirs(os.path.dirname(self.schema_path), exist_ok=True)
                with open(self.schema_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                self._set_status("Saved (restart VFA Server to apply)")
            except OSError as exc:
                self._set_status(f"Save failed: {exc}")


def _make_stream_fields(stream_name: str) -> list[LoaderFieldDef]:
    """create FieldDef list for a single stream entry."""
    section = f"Streams.{stream_name}"
    ssh_profile_names = ["local"] + [p.name for p in load_ssh_profiles()]
    fields = []
    for key, ftype, default, desc, is_choices in _STREAM_FIELDS_TEMPLATE:
        choices = ssh_profile_names if is_choices else []
        fields.append(LoaderFieldDef(
            path=f"Streams.{stream_name}.{key}",
            field_type=ftype,
            default=default,
            description=desc,
            required=(key == "target"),
            section=section,
            choices=choices,
        ))
    return fields


def _build_service_registry(root: str) -> list[ServiceDef]:
    """build the registry of all launchable services."""
    services = []

    services.append(ServiceDef(
        name="ASR Base",
        category="ASR",
        conda_env="asr-base",
        config_dir=os.path.join(root, "pipelines", "asr-base"),
        launch_type="bash",
        description="Real-time audio analysis base stations and synchronizer",
        params=[
            ParamDef("-nb", "Num Bases", "int", 1),
            ParamDef("-ns", "Num Synchronizers", "int", 1),
            ParamDef("-sid", "Session", "str", ""),
            ParamDef("--experiment-group", "Experiment Group", "str", ""),
            ParamDef("-m", "Mode", "str", "live", ["live", "capture", "analyze"]),
            ParamDef("-s", "Store Audio", "bool", True),
            ParamDef("-vad", "VAD", "bool", True),
            ParamDef("-nr", "Noise Reduce", "bool", True),
            ParamDef("-tr", "Transcribe", "bool", True),
            ParamDef("-sp", "Speech Separate", "bool", False),
            ParamDef("-d", "Dominant Speaker", "bool", False),
            ParamDef("-hsr", "Half-Scaled Recognition", "bool", True),
        ],
        components=[
            ComponentDef("base", "mmla asr-base", "-nb",
                         ["-sid", "-m", "-s", "-vad", "-nr", "-tr", "-sp", "-hsr"]),
            ComponentDef("synchronizer", "mmla asr-sync", "-ns",
                         ["-sid", "-d", "-sp"]),
        ],
        artifact_pipeline="asr-base",
    ))

    services.append(ServiceDef(
        name="VFA Base",
        category="VFA",
        conda_env="vfa-base",
        config_dir=os.path.join(root, "pipelines", "vfa-base"),
        launch_type="bash",
        description="Video frame analysis base stations and synchronizer",
        params=[
            ParamDef("-nb", "Num Bases", "int", 1),
            ParamDef("-ns", "Num Synchronizers", "int", 1),
            ParamDef("-sid", "Session", "str", ""),
            ParamDef("--experiment-group", "Experiment Group", "str", ""),
            ParamDef("-m", "Mode", "str", "live", ["live", "capture", "analyze"]),
            ParamDef("-g", "Graphics", "bool", True),
            ParamDef("-s", "Store Frames", "bool", True),
            ParamDef("-v", "Verbose", "bool", True),
        ],
        components=[
            ComponentDef("base", "mmla vfa-base", "-nb",
                         ["-sid", "-m", "-g", "-s", "-v"]),
            ComponentDef("synchronizer", "mmla vfa-sync", "-ns", ["-sid"]),
        ],
        artifact_pipeline="vfa-base",
    ))

    services.append(ServiceDef(
        name="IPS Base",
        category="IPS",
        conda_env="ips-base",
        config_dir=os.path.join(root, "pipelines", "ips-base"),
        launch_type="bash",
        description="Indoor positioning system base stations, synchronizer, and visualizer",
        params=[
            ParamDef("-nb", "Num Bases", "int", 1),
            ParamDef("-ns", "Num Synchronizers", "int", 1),
            ParamDef("-nv", "Num Visualizers", "int", 1),
            ParamDef("-sid", "Session", "str", ""),
            ParamDef("--experiment-group", "Experiment Group", "str", ""),
            ParamDef("-g", "Graphics", "bool", True),
            ParamDef("-s", "Store Frames", "bool", True),
            ParamDef("-v", "Verbose", "bool", True),
        ],
        components=[
            ComponentDef("base", "mmla ips-base", "-nb",
                         ["-sid", "-g", "-s", "-v"]),
            ComponentDef("synchronizer", "mmla ips-sync", "-ns",
                         ["-sid", "-v"]),
            ComponentDef("visualizer", "mmla ips-vis", "-nv",
                         ["-sid", "-s"]),
        ],
        artifact_pipeline="ips-base",
    ))

    services.append(ServiceDef(
        name="IPS Camera Calibration",
        category="IPS",
        conda_env="ips-base",
        config_dir=os.path.join(root, "pipelines", "ips-base"),
        launch_type="bash",
        description="Calibrate camera intrinsic parameters (interactive, run before IPS sessions)",
        params=[
            ParamDef("-n", "Num Calibrators", "int", 1),
        ],
        components=[
            ComponentDef("calibrator", "mmla ips-ccal", "-n", []),
        ],
    ))

    services.append(ServiceDef(
        name="IPS Camera Sync",
        category="IPS",
        conda_env="ips-base",
        config_dir=os.path.join(root, "pipelines", "ips-base"),
        launch_type="bash",
        description="Synchronize multi-camera coordinates (tag detectors + sync manager)",
        params=[
            ParamDef("-nc", "Num Tag Detectors", "int", 2),
            ParamDef("-ns", "Num Sync Managers", "int", 1),
        ],
        components=[
            ComponentDef("tag detector", "mmla ips-ctag", "-nc", []),
            ComponentDef("sync manager", "mmla ips-csync", "-ns", []),
        ],
    ))

    services.append(ServiceDef(
        name="Collection Session",
        category="Collection",
        conda_env="",
        config_dir=root,
        launch_type="collection",
        description="Record raw audio/video files for post-time processing",
        params=[
            ParamDef("-na", "Num Audio", "int", 0),
            ParamDef("-nv", "Num Video", "int", 0),
            ParamDef("--session-id", "Session ID", "str", ""),
            ParamDef("--experiment-group", "Experiment Group", "str", ""),
            ParamDef("--output-root", "Output Root", "str", "artifacts"),
        ],
        components=[
            ComponentDef(
                "audio",
                "scripts/collection/audio_recording.sh",
                "-na",
                [
                    "--session-id", "--output-root", "--host-label",
                    "--audio-interactive", "--sample-rate", "--audio-format",
                ],
            ),
            ComponentDef(
                "video",
                "scripts/collection/video_recording.sh",
                "-nv",
                [
                    "--session-id", "--output-root", "--host-label",
                    "--video-interactive",
                    "--framerate", "--size", "--bitrate", "--maxrate", "--bufsize",
                    "--preset", "--camera-label",
                ],
            ),
        ],
    ))

    services.append(ServiceDef(
        name="ASR Server",
        category="ASR",
        conda_env="docker",
        config_dir=os.path.join(root, "pipelines", "asr-server"),
        launch_type="tmux",
        display_type="docker",
        description=(
            "ASR inference services, one container per service "
            f"(AudioInferer: {_asr_audio_inferer_backend(root)})"
        ),
    ))

    services.append(ServiceDef(
        name="VFA Server",
        category="VFA",
        conda_env="docker",
        config_dir=os.path.join(root, "pipelines", "vfa-server"),
        launch_type="tmux",
        display_type="docker",
        description="VFA inference services, one container per service (VLLM frame analyzer, ...)",
    ))

    mllm_config = _mllm_config(root)
    services.append(ServiceDef(
        name="MLLM Server",
        category="VFA",
        conda_env="vfa-vllm",
        config_dir=root,
        launch_type="vllm",
        description=(
            "OpenAI-compatible vLLM server "
            f"({mllm_config['model']} on :{mllm_config['port']})"
        ),
    ))

    uber_dir = os.path.join(root, "pipelines", "uber-server")
    if os.path.isdir(uber_dir):
        for svc_name, desc in [
            ("InfluxDB", "Time series database"),
            ("MongoDB", "Document database"),
            ("Redis", "In-memory data store and message broker"),
            ("Mosquitto", "MQTT message broker"),
            ("Nginx", "Reverse proxy and load balancer"),
            ("Flask", "Dashboard (backend API + web frontend)"),
            ("Celery", "Async task worker"),
        ]:
            name = f"Uber: {svc_name}"
            # only the services with a container equivalent in the infra compose
            # file get a run-mode switch; the rest stay Makefile-only
            params = (
                [ParamDef("--mode", "Run mode", "str", "native", ["native", "docker"])]
                if _make_target_for(name) in _INFRA_COMPOSE_SERVICES
                else []
            )
            services.append(ServiceDef(
                name=name,
                category="Infrastructure",
                conda_env="uber-server",
                config_dir=uber_dir,
                launch_type="make",
                description=desc,
                params=params,
            ))

    return services


def _check_tmux_session(session_name: str) -> bool:
    """check if a tmux session with the given name exists."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _capture_tmux_pane(session_name: str, lines: int = 80) -> str:
    """capture recent output from a tmux session."""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session_name, "-p", "-S", f"-{lines}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout
        return f"(could not capture pane for session '{session_name}')"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "(tmux not available or session not found)"


def _get_system_service_log(service_name: str, lines: int = 80) -> str:
    """get log output for a brew-managed system service."""
    try:
        prefix = subprocess.run(
            ["brew", "--prefix"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        prefix = "/opt/homebrew"

    log_paths: dict[str, list[str]] = {
        "influxdb": [f"{prefix}/var/log/influxdb2/influxd_output.log"],
        "mongodb": [f"{prefix}/var/log/mongodb/mongo.log"],
        "redis": [f"{prefix}/var/log/redis.log"],
        "nginx": [
            f"{prefix}/var/log/nginx/error.log",
            f"{prefix}/var/log/nginx/access.log",
        ],
        "mosquitto": [f"{prefix}/var/log/mosquitto/mosquitto.log"],
    }

    paths = log_paths.get(service_name, [])
    for path in paths:
        if os.path.isfile(path):
            try:
                result = subprocess.run(
                    ["tail", "-n", str(lines), path],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return f"[log file: {path}]\n{result.stdout}"
            except Exception:
                continue

    try:
        proc_name = "influxd" if service_name == "influxdb" else "mongod" if service_name == "mongodb" else service_name
        result = subprocess.run(
            ["log", "show", "--predicate", f'process == "{proc_name}"',
             "--last", "5m", "--style", "compact"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"[macOS system log]\n{result.stdout}"
    except Exception:
        pass

    return f"(no log file found for {service_name})"


def _check_config_exists(config_dir: str) -> bool:
    return os.path.isfile(os.path.join(config_dir, "config.yml"))


def _tmux_component_session_name(service_name: str) -> str:
    raw = service_name.lower().replace(" ", "_")
    return "".join(ch for ch in raw if ch.isalnum() or ch in "_-")


def _stack_service_specs_from_config(config: dict) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    if not isinstance(config, dict):
        return specs
    for service_name, service_config in config.items():
        if not isinstance(service_config, dict) or "port" not in service_config:
            continue
        port = _coerce_int(service_config.get("port"), 0)
        if port <= 0:
            continue
        specs.append({
            "name": str(service_name),
            "session": _tmux_component_session_name(str(service_name)),
            "port": port,
        })
    return specs


def _stack_service_specs(config_dir: str) -> list[dict[str, object]]:
    config = load_existing_config(os.path.join(config_dir, "config.yml"))
    return _stack_service_specs_from_config(config)


# Built-in WSGI app module for each stack component. These are shipped with the
# toolkit and fixed per component, so users no longer fill an `app` path in the
# config; the launcher resolves it here. A user may still override by setting an
# explicit `app:` in config.yml.
_DEFAULT_STACK_APPS = {
    "AudioInferer": "openmmla.services.asr.apps.serve_audio_inferer",
    "AudioResampler": "openmmla.services.asr.apps.serve_audio_resampler",
    "SpeechEnhancer": "openmmla.services.asr.apps.serve_speech_enhancer",
    "SpeechSeparator": "openmmla.services.asr.apps.serve_speech_separator",
    "SpeechTranscriber": "openmmla.services.asr.apps.serve_speech_transcriber",
    "VoiceActivityDetector": "openmmla.services.asr.apps.serve_voice_activity_detector",
    "VLLMFrameAnalyzer": "openmmla.services.vfa.apps.serve_multi_angle_vllm_frame_analyzer",
}


def _is_placeholder_value(value: str) -> bool:
    """True for unfilled template placeholders like <path-to-...>."""
    return bool(re.fullmatch(r"<[^>]*>", str(value).strip()))


def _stack_launch_specs_from_config(config: dict) -> list[dict[str, object]]:
    """Like _stack_service_specs_from_config, but include the workers/app fields
    needed to build gunicorn launch commands (ported from bash/services.sh)."""
    specs: list[dict[str, object]] = []
    if not isinstance(config, dict):
        return specs
    for service_name, service_config in config.items():
        if not isinstance(service_config, dict) or "port" not in service_config:
            continue
        port = _coerce_int(service_config.get("port"), 0)
        if port <= 0:
            continue
        app = str(service_config.get("app") or "").strip()
        if not app or _is_placeholder_value(app):
            # fall back to the built-in toolkit module for this component
            app = _DEFAULT_STACK_APPS.get(str(service_name), "")
        if not app:
            continue
        specs.append({
            "name": str(service_name),
            "session": _tmux_component_session_name(str(service_name)),
            "port": port,
            "workers": max(1, _coerce_int(service_config.get("workers"), 1)),
            "app": app,
        })
    return specs


def _split_app_target(app: str, check_filesystem: bool) -> tuple[str, str]:
    """Resolve a config 'app' value into (working_dir, gunicorn module).

    Module paths like openmmla.services.asr.apps.serve_audio_inferer pass
    through unchanged. File paths (absolute, ~, or containing a separator)
    are split into a cd directory and a bare module name."""
    looks_like_path = "/" in app or app.startswith("~")
    if check_filesystem:
        expanded = os.path.expanduser(app)
        looks_like_path = looks_like_path and (
            os.path.exists(expanded) or os.path.exists(expanded + ".py")
        )
        app = expanded if looks_like_path else app
    if not looks_like_path:
        return "", app
    module = os.path.basename(app)
    if module.endswith(".py"):
        module = module[:-3]
    return os.path.dirname(app), module


def _stack_service_shell_command(
    spec: dict[str, object],
    project_dir: str,
    config_path: str,
    *,
    remote: bool = False,
) -> str:
    """Build the gunicorn command for one server-stack service."""
    workdir, module = _split_app_target(str(spec["app"]), check_filesystem=not remote)
    quote = _quote_remote_path if remote else shlex.quote
    cd_part = f"cd {quote(workdir)} && " if workdir else ""
    # The WSGI app factories read the un-prefixed PROJECT_DIR / CONFIG_PATH
    # (see openmmla/commands/asr/*.py and openmmla/services/**/apps/serve_*.py).
    # --timeout 0 disables gunicorn's worker timeout: these inference services
    # load multi-GB models at boot (first run also downloads them), which can far
    # exceed the 30s default and would otherwise get the worker killed mid-load.
    return (
        f"{cd_part}"
        f"PROJECT_DIR={quote(project_dir)} "
        f"CONFIG_PATH={quote(config_path)} "
        f"gunicorn -k gevent --timeout 0 -w {spec['workers']} "
        f"-b 0.0.0.0:{spec['port']} {module}:app"
    )


def _kill_port_processes(port: int) -> None:
    """Force-kill any local processes listening on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    for pid in result.stdout.split():
        pid = pid.strip()
        if pid:
            subprocess.run(["kill", "-9", pid], capture_output=True)


def _remote_kill_port_snippet(port: int) -> str:
    """POSIX shell snippet that kills processes on a port (remote side)."""
    return (
        f'for _p in $(lsof -ti:{port} 2>/dev/null); do kill -9 "$_p" 2>/dev/null; done'
    )


def _is_stack_service(svc: ServiceDef) -> bool:
    """ASR/VFA Server stacks are managed via docker compose (one container per
    sub-service); status detection stays port-based."""
    return svc.launch_type == "tmux" and svc.name in ("ASR Server", "VFA Server")


# compose file (relative to the repo root) per stack service
_STACK_COMPOSE_FILES = {
    "ASR Server": "docker/docker-compose.asr.yml",
    "VFA Server": "docker/docker-compose.vfa.yml",
}

# config section name -> docker compose service name
_STACK_COMPOSE_SERVICES = {
    "AudioInferer": "audio-inferer",
    "AudioResampler": "audio-resampler",
    "SpeechEnhancer": "speech-enhancer",
    "SpeechSeparator": "speech-separator",
    "SpeechTranscriber": "speech-transcriber",
    "VoiceActivityDetector": "voice-activity-detector",
    "VLLMFrameAnalyzer": "frame-analyzer",
}

# central infrastructure compose stack (relative to the repo root); the Uber
# cards listed below can drive either the bare-metal Makefile or these containers
_INFRA_COMPOSE_FILE = "docker/docker-compose.infra.yml"

# _make_target_for() name -> docker compose service name
_INFRA_COMPOSE_SERVICES = {
    "influxdb": "influxdb",
    "mongodb": "mongodb",
}


def _stack_compose_rel_file(svc: ServiceDef) -> str | None:
    return _STACK_COMPOSE_FILES.get(svc.name)


def _compose_service_names(specs: list[dict[str, object]], config: dict | None) -> list[str]:
    """map selected config sections to compose service names; AudioInferer maps
    to its nemo variant when the config backend is 'nemo'."""
    backend = str(((config or {}).get("AudioInferer") or {}).get("backend") or "").strip().lower()
    names = []
    for spec in specs:
        name = _STACK_COMPOSE_SERVICES.get(str(spec["name"]))
        if not name:
            continue
        if name == "audio-inferer" and backend == "nemo":
            name = "audio-inferer-nemo"
        names.append(name)
    return names


def _compose_command(compose_rel_file: str, args: str, profiles: list[str] | None = None) -> str:
    """build a `docker compose` command relative to the repo root."""
    profile_part = "".join(f" --profile {p}" for p in (profiles or []))
    return f"docker compose -f {shlex.quote(compose_rel_file)}{profile_part} {args}"


def _infra_compose_service(svc: ServiceDef) -> str | None:
    """compose service name for an Uber card that has a container equivalent."""
    if svc.launch_type != "make":
        return None
    return _INFRA_COMPOSE_SERVICES.get(_make_target_for(svc.name))


def _infra_docker_mode(svc: ServiceDef, params: dict | None = None) -> bool:
    """whether this card's Run mode asks for the container instead of the
    bare-metal Makefile; anything but an explicit "docker" stays native."""
    if _infra_compose_service(svc) is None:
        return False
    return str((params or {}).get("--mode") or "native").strip().lower() == "docker"


def _stack_legacy_session(svc: ServiceDef) -> str | None:
    if svc.name == "ASR Server":
        return "asr-services"
    if svc.name == "VFA Server":
        return "vfa-services"
    return None


def _stack_sessions(svc: ServiceDef) -> list[str]:
    return _stack_sessions_from_specs(svc, _stack_service_specs(svc.config_dir))


def _stack_sessions_from_specs(svc: ServiceDef, specs: list[dict[str, object]]) -> list[str]:
    sessions = [_service_session_name(svc)]
    legacy = _stack_legacy_session(svc)
    if legacy:
        sessions.append(legacy)
    sessions.extend(str(spec["session"]) for spec in specs)
    return list(dict.fromkeys(sessions))


def _stack_ports(svc: ServiceDef) -> list[int]:
    return _stack_ports_from_specs(_stack_service_specs(svc.config_dir))


def _stack_ports_from_specs(specs: list[dict[str, object]]) -> list[int]:
    return [int(spec["port"]) for spec in specs]


_SESSION_CHOICE_TTL_SEC = 20.0
_TARGET_PROBE_INTERVAL_SEC = 30.0

_probe_ssh_endpoint = probe_ssh_endpoint

_SYSTEM_SVC_PORTS: dict[str, int] = {
    "influxdb": 8086,
    "mongodb": 27017,
    "redis": 6379,
    "mosquitto": 1883,
    "nginx": 8080,
}
_NON_SESSION_ARTIFACT_NAMES = {*NON_SESSION_ARTIFACT_DIRS, ".DS_Store"}

_MAKE_TARGET_OVERRIDES: dict[str, str] = {}

# (port, expected_command) for app services that need port cleanup on stop
_APP_PORT_CMDS: dict[str, tuple[int, str]] = {
    "flask": (5050, "gunicorn"),
}


def _make_target_for(svc_name: str) -> str:
    """derive the Makefile target name from a service name."""
    if svc_name in _MAKE_TARGET_OVERRIDES:
        return _MAKE_TARGET_OVERRIDES[svc_name]
    return svc_name.replace("Uber: ", "").lower().replace(".", "")


def _check_port_in_use(port: int) -> bool:
    """check if a TCP port is in use on localhost."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


_KNOWN_CONDA_ENVS: set[str] = set()


def _check_conda_env(env_name: str) -> bool:
    """check that a conda env exists.

    Positive results are cached for the lifetime of the TUI (envs are rarely
    deleted mid-session); on a cache miss the list is re-queried, so freshly
    created envs are always picked up."""
    if env_name in _KNOWN_CONDA_ENVS:
        return True
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts:
            _KNOWN_CONDA_ENVS.add(parts[0])
    return env_name in _KNOWN_CONDA_ENVS


def _service_session_name(svc: ServiceDef) -> str:
    return svc.name.lower().replace(" ", "-")


def _collection_session_prefix(svc: ServiceDef) -> str:
    return _service_session_name(svc) + "-"


def _collection_sessions_local(svc: ServiceDef) -> list[str]:
    prefix = _collection_session_prefix(svc)
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        return [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().startswith(prefix)
        ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _service_requires_config(svc: ServiceDef) -> bool:
    return svc.launch_type not in ("make", "vllm", "collection")


def _service_python_hint(svc: ServiceDef) -> str:
    return "3.12" if svc.conda_env == "vfa-vllm" else "3.10"


def _vllm_serve_command(config: dict | None = None) -> str:
    cfg = config or _mllm_config(_find_project_root())
    args = [
        "vllm", "serve", cfg["model"],
        "--host", cfg["host"],
        "--port", str(cfg["port"]),
        "--dtype", cfg["dtype"],
        "--max-model-len", str(cfg["max_model_len"]),
        "--limit-mm-per-prompt", cfg["limit_mm_per_prompt"],
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--api-key", cfg["api_key"],
    ]
    return " ".join(shlex.quote(arg) for arg in args)


class ServicePanel(Widget):

    DEFAULT_CSS = """
    ServicePanel {
        layout: horizontal;
        width: 1fr;
        height: 1fr;
    }
    #svc-sidebar {
        width: 32;
        border-right: solid $primary;
        padding: 1;
        background: $panel;
    }
    #svc-sidebar Tree {
        width: 1fr;
        scrollbar-size: 1 1;
    }
    #svc-main {
        width: 1fr;
        height: 1fr;
    }
    #svc-target-bar {
        layout: horizontal;
        height: auto;
        padding: 0 1;
    }
    #svc-target-bar Label {
        width: 10;
        padding-top: 1;
    }
    #svc-target-bar Select {
        width: 1fr;
    }
    #svc-target-refresh {
        width: 5;
        min-width: 5;
        height: 3;
        margin-left: 1;
    }
    #svc-content-area {
        width: 1fr;
        height: 1fr;
    }
    #svc-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
        text-style: italic;
    }
    .svc-config-scroll {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }
    .svc-launch-scroll {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }
    #svc-sub-tabs {
        width: 1fr;
        height: 1fr;
    }
    /* TabPane defaults to height:auto, which lets pane content escape the
       viewport: inner VerticalScrolls then never scroll and the overflow is
       clipped (e.g. when the command log is resized taller). Pin the pane
       and switcher to the tab body height so scrolling works. */
    #svc-sub-tabs ContentSwitcher {
        height: 1fr;
    }
    #svc-sub-tabs TabPane {
        height: 1fr;
    }
    .sync-bar {
        layout: horizontal;
        height: auto;
        padding: 1 0;
        margin-top: 1;
    }
    .sync-bar Select {
        width: 1fr;
    }
    .sync-bar Button {
        margin: 0 1;
        min-width: 18;
    }
    .preflight-warning {
        color: $warning;
        padding: 0 2;
    }
    .add-base-bar, .add-stream-bar {
        layout: horizontal;
        height: auto;
        padding: 1 0;
    }
    .add-base-bar Input, .add-stream-bar Input {
        width: 1fr;
    }
    .add-base-bar Button, .add-stream-bar Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._root = _find_project_root()
        self._services = _build_service_registry(self._root)
        self._svc_map: dict[str, ServiceDef] = {s.name: s for s in self._services}
        self._svc_states: dict[str, bool] = {}
        self._pipelines: list[PipelineDef] = []
        self._pipeline_map: dict[str, PipelineDef] = {}
        self._shared_values: dict[str, object] = load_system_service_values(self._root)
        self._current_pipeline: PipelineDef | None = None
        self._current_form: ConfigForm | None = None
        self._current_shared_section: str | None = None
        self._current_config_local_path: str | None = None
        self._config_container: VerticalScroll | None = None
        self._ssh_profile_names: list[str] = [
            p.name for p in load_ssh_profiles()
        ]
        self._collection_last_params: dict[tuple[str, str], dict] = {}
        # collection choices that describe the *recording session* rather than
        # the machine doing the recording: they follow the user from host to
        # host, so setting up a multi-machine session is one pass, not one
        # full re-entry per host.
        self._collection_sticky: dict[str, object] = {}
        # ...and the ones that are per-machine (output root, host label), kept
        # so a rebuild on the same host does not discard local edits
        self._collection_target_sticky: dict[str, dict[str, object]] = {}
        self._collection_role: str = "audio"
        # (target, service name) -> "native" | "docker" for the infra cards; the
        # card is rebuilt on every tree/host change and would otherwise fall back
        # to the ParamDef default, silently sending Stop down the make path
        self._infra_mode: dict[tuple[str, str], str] = {}
        # session id -> hosts this TUI launched it on, so "Stop All Hosts"
        # reaches a machine even when it is currently unreachable
        self._collection_launch_targets: dict[str, set[str]] = {}
        self._pending_collection_delete: tuple[str, str] | None = None
        self._target_config_cache: dict[tuple[str, str], dict] = {}
        self._target_platform_cache: dict[str, str] = {}
        self._current_service_name: str | None = None
        self._target_states: dict[str, str] = {}
        self._target_probe_timer = None
        self._last_target: str = "local"
        self._suppress_target_change: bool = False
        self._target_states: dict[str, str] = {}  # profile name -> online/offline
        self._target_probe_timer = None
        self._last_target: str = "local"
        self._suppress_target_change: bool = False

    def compose(self) -> ComposeResult:
        target_options = [("Local", "local")] + [
            (name, name) for name in self._ssh_profile_names
        ]
        with Vertical(id="svc-sidebar"):
            yield Static("[b]Launcher[/b]", classes="status-info")
            tree: Tree[str] = Tree("OpenMMLA", id="svc-tree")
            tree.root.expand()
            yield tree
        with Vertical(id="svc-main"):
            with Horizontal(id="svc-target-bar"):
                yield Label("Host:")
                yield Select(target_options, value="local", id="svc-target-select")
                yield Button("↻", variant="primary", compact=True, id="svc-target-refresh")
            with Vertical(id="svc-content-area"):
                yield Static(
                    "Select a service from the sidebar.",
                    id="svc-empty",
                )
            yield CommandSession(show_target=False, id="svc-cmd-session")

    def on_mount(self) -> None:
        self._pipelines = discover_pipelines()
        self._pipeline_map = {p.name: p for p in self._pipelines}
        self._build_tree()
        # populate running markers asynchronously; the tree itself renders
        # instantly from cached states
        self._refresh_visible_statuses()
        self._probe_targets()

    # ── target reachability ──────────────────────────────────────

    def _probe_targets(self) -> None:
        self.run_worker(
            self._async_probe_targets(),
            group="launcher-target-probe",
            exclusive=True,
        )

    async def _async_test_single_host(self, name: str) -> None:
        from openmmla.tui.ssh import test_profile_by_name
        success, msg = await asyncio.to_thread(test_profile_by_name, name)
        self._target_states = dict(TARGET_STATES)
        self._refresh_target_options()
        color = "green" if success else "red"
        self._log(f"[{color}]'{name}': {msg}[/{color}]")

    async def _async_manual_probe(self) -> None:
        states = await asyncio.to_thread(probe_all_profiles)
        self._target_states = states
        self._refresh_target_options()
        self._log(f"[green]Connection test finished: {summarize_states(states)}.[/green]")

    async def _async_probe_targets(self) -> None:
        try:
            states = await asyncio.to_thread(probe_all_profiles)
            if states != self._target_states:
                for name, state in states.items():
                    previous = self._target_states.get(name)
                    if previous is not None and previous != state and state in ("online", "offline"):
                        color = "green" if state == "online" else "red"
                        self._log(f"[{color}]Host '{name}' is now {state}.[/{color}]")
                self._target_states = states
                self._refresh_target_options()
        finally:
            # the re-probe loop must survive any failure above
            if self._target_probe_timer is not None:
                self._target_probe_timer.stop()
            self._target_probe_timer = self.set_timer(_TARGET_PROBE_INTERVAL_SEC, self._probe_targets)

    def on_show(self) -> None:
        self._refresh_target_options()
        if self._current_service_name == "Collection Session":
            self.run_worker(
                self._reload_current_service_view(),
                group=_LAUNCHER_UI_WORKER_GROUP,
                exclusive=True,
            )

    def on_ssh_form_profiles_changed(self, event: SSHForm.ProfilesChanged) -> None:
        event.stop()
        self._refresh_target_options()
        self._probe_targets()

    def on_ssh_form_connection_tested(self, event: SSHForm.ConnectionTested) -> None:
        """a single-host test updated TARGET_STATES; mirror it in the dropdown."""
        event.stop()
        self._target_states = dict(TARGET_STATES)
        self._refresh_target_options()

    def _target_option_label(self, name: str) -> str:
        return target_state_label(name)

    def _refresh_target_options(self) -> None:
        self._ssh_profile_names = [p.name for p in load_ssh_profiles()]
        try:
            sel = self.query_one("#svc-target-select", Select)
            options = target_options()
            current = sel.value
            self._suppress_target_change = True
            try:
                sel.set_options(options)
                if any(v == current for _, v in options):
                    sel.value = current
                else:
                    sel.value = "local"
                    self._last_target = "local"
                    self.query_one("#svc-cmd-session", CommandSession).set_target("local")
            finally:
                self.call_after_refresh(self._clear_target_suppression)
        except Exception:
            self._suppress_target_change = False

    def _clear_target_suppression(self) -> None:
        self._suppress_target_change = False

    def _get_panel_target(self) -> str:
        """get the current target from the unified selector."""
        try:
            sel = self.query_one("#svc-target-select", Select)
            val = sel.value
            if is_select_sentinel(val):
                return "local"
            return str(val)
        except Exception:
            return "local"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "svc-target-select":
            if self._suppress_target_change:
                return
            if is_select_sentinel(event.value):
                return  # transient placeholder while options are rebuilt
            val = str(event.value)
            if val == REFRESH_TARGETS_OPTION:
                self._log("[yellow]Testing connections to all hosts...[/yellow]")
                self._suppress_target_change = True
                event.select.value = self._last_target
                self.call_after_refresh(self._clear_target_suppression)
                self.run_worker(
                    self._async_manual_probe(),
                    group="launcher-target-probe",
                    exclusive=True,
                )
                return
            if val == self._last_target:
                return
            if val != "local" and self._target_states.get(val) == "offline":
                self._log(
                    f"[red]Host '{val}' is offline; staying on '{self._last_target}'. "
                    f"Re-testing it now...[/red]"
                )
                self._suppress_target_change = True
                event.select.value = self._last_target
                self.call_after_refresh(self._clear_target_suppression)
                self.run_worker(
                    self._async_test_single_host(val),
                    group="launcher-target-probe",
                    exclusive=False,
                )
                return
            # snapshot against the host the card still belongs to, so per-host
            # values (output root, host label) are not filed under the new one
            self._capture_collection_card_state(self._last_target)
            self._capture_infra_mode(self._last_target)
            self._last_target = val
            if val != "local":
                self._log(f"[yellow]Switching host to '{val}' — loading remote state...[/yellow]")
            cmd = self.query_one("#svc-cmd-session", CommandSession)
            cmd.set_target(val)
            # stale states from the previous target should not linger
            self._svc_states.clear()
            self.run_worker(
                self._reload_current_service_view(capture=False),
                group=_LAUNCHER_UI_WORKER_GROUP,
                exclusive=True,
            )
            self._refresh_visible_statuses()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tabbed_content.id != "svc-sub-tabs":
            return
        self._set_command_session_visible(event.pane.id != "svc-tab-config")

    def _set_command_session_visible(self, visible: bool) -> None:
        try:
            self.query_one("#svc-cmd-session", CommandSession).display = visible
        except Exception:
            pass

    def _pipeline_for_service(self, svc_name: str) -> PipelineDef | None:
        """find the PipelineDef associated with a service, if any."""
        pipeline_name = _SVC_PIPELINE_NAMES.get(svc_name, svc_name)
        return self._pipeline_map.get(pipeline_name)

    # ── sidebar tree ─────────────────────────────────────────────

    def _build_tree(self) -> None:
        tree = self.query_one("#svc-tree", Tree)
        tree.clear()

        shared_node = tree.root.add(_SYSTEM_SERVICES_LABEL, data="__shared__")
        shared_node.expand()
        added_shared_sections: set[str] = set()
        for item in _GLOBAL_DEFAULT_NAV_ORDER:
            if item == "SSH Profiles":
                shared_node.add_leaf("SSH Profiles", data="__ssh_profiles__")
            elif item == "Experiments":
                shared_node.add_leaf("Experiments", data="__experiments__")
            elif item == "Tasks":
                shared_node.add_leaf("Tasks", data="__tasks__")
            elif item in SHARED_SECTIONS:
                shared_node.add_leaf(item, data=f"__shared__{item}")
                added_shared_sections.add(item)
        for sec in SHARED_SECTIONS:
            if sec not in added_shared_sections:
                shared_node.add_leaf(sec, data=f"__shared__{sec}")

        categories: dict[str, list[ServiceDef]] = {}
        for svc in self._services:
            categories.setdefault(svc.category, []).append(svc)

        infra_svcs = categories.get("Infrastructure", [])
        if infra_svcs:
            infra_node = tree.root.add("Infrastructure", data="__cat_Infrastructure")
            infra_node.expand()
            for svc in infra_svcs:
                infra_node.add_leaf(f"{svc.name}{self._svc_markers(svc)}", data=svc.name)

        collection_svcs = categories.get("Collection", [])
        if collection_svcs:
            collection_node = tree.root.add("Collection", data="__cat_Collection")
            collection_node.expand()
            for svc in collection_svcs:
                collection_node.add_leaf(f"{svc.name}{self._svc_markers(svc)}", data=svc.name)

        _PIPELINE_CATS = ["ASR", "IPS", "VFA"]

        pipeline_node = tree.root.add("Pipelines", data="__cat_Pipelines")
        pipeline_node.expand()
        for cat in _PIPELINE_CATS:
            svcs = categories.get(cat, [])
            if not svcs:
                continue
            cat_node = pipeline_node.add(cat, data=f"__cat_{cat}")
            cat_node.expand()
            for svc in svcs:
                cat_node.add_leaf(f"{svc.name}{self._svc_markers(svc)}", data=svc.name)
        pipeline_node.add_leaf("Session Control", data="__session_control__")

    def _svc_markers(self, svc: ServiceDef) -> str:
        """build status marker string for a service tree leaf.

        Reads the cached running state only — live detection (subprocess/
        socket/SSH probes) happens in the _refresh_visible_statuses worker,
        never on the UI thread while rendering the tree."""
        is_running = self._svc_states.get(svc.name, False)
        pipeline = self._pipeline_for_service(svc.name)
        markers = ""
        if pipeline and os.path.isfile(pipeline.config_path):
            markers += " [green]\\[OK][/green]"
        if is_running:
            markers += " [green](R)[/green]"
        return markers

    # ── tree node selection ──────────────────────────────────────

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node_data = event.node.data
        if node_data is None:
            return
        node_str = str(node_data)

        if node_str.startswith("__cat_") or node_str == "__shared__":
            return

        self._capture_collection_card_state()
        self._capture_infra_mode()
        content_area = self.query_one("#svc-content-area", Vertical)
        await content_area.remove_children()
        self._current_pipeline = None
        self._current_form = None
        self._current_shared_section = None
        self._current_config_local_path = None
        self._config_container = None
        self._current_service_name = None

        if node_str.startswith("__shared__"):
            self._set_command_session_visible(False)
            section_name = node_str.replace("__shared__", "")
            scroll = VerticalScroll(classes="svc-config-scroll")
            await content_area.mount(scroll)
            self._config_container = scroll
            self._current_shared_section = section_name
            self._show_shared_form(scroll, section_name)
            return

        if node_str == "__experiments__":
            self._set_command_session_visible(False)
            await content_area.mount(ExperimentForm())
            return

        if node_str == "__tasks__":
            self._set_command_session_visible(False)
            await content_area.mount(TaskForm())
            return

        if node_str == "__ssh_profiles__":
            self._set_command_session_visible(False)
            await content_area.mount(SSHForm())
            return

        if node_str == "__session_control__":
            self._set_command_session_visible(False)
            target = self._get_panel_target()
            choices = await asyncio.to_thread(self._artifact_session_choices_for_target, target)
            choices = [c for c in choices if c and c != _NEW_COLLECTION_SESSION_CHOICE]
            from openmmla.tui.system_services import system_services_config_path
            scroll = VerticalScroll(classes="svc-launch-scroll")
            await content_area.mount(scroll)
            await scroll.mount(SessionControlPanel(choices, system_services_config_path(self._root)))
            return

        svc = self._svc_map.get(node_str)
        if svc is None:
            return
        self._current_service_name = node_str
        base_svc = svc
        self._set_command_session_visible(True)

        target = self._get_panel_target()
        svc, is_running = await self._service_view_state(base_svc, target)
        self._svc_states[base_svc.name] = is_running

        await self._mount_service_content(content_area, svc, is_running)

    async def _reload_current_service_view(self, capture: bool = True) -> None:
        """rebuild the selected service panel after target-dependent state changes.

        `capture` is off for reloads that follow a launch: the launcher has
        already recorded the resolved session id, and the card on screen still
        shows the pre-launch selection."""
        if capture:
            self._capture_collection_card_state()
            self._capture_infra_mode()
        if not self._current_service_name:
            self._build_tree()
            return
        svc = self._svc_map.get(self._current_service_name)
        if svc is None:
            self._build_tree()
            return
        target = self._get_panel_target()
        display_svc, is_running = await self._service_view_state(svc, target)
        self._svc_states[svc.name] = is_running

        content_area = self.query_one("#svc-content-area", Vertical)
        await content_area.remove_children()
        self._current_pipeline = None
        self._current_form = None
        self._current_shared_section = None
        self._current_config_local_path = None
        self._config_container = None
        self._set_command_session_visible(True)
        await self._mount_service_content(
            content_area,
            display_svc,
            is_running,
        )
        self._build_tree()

    async def _service_view_state(self, svc: ServiceDef, target: str) -> tuple[ServiceDef, bool]:
        """resolve target-dependent service metadata without blocking the UI loop."""
        return await asyncio.to_thread(self._service_view_state_sync, svc, target)

    def _service_view_state_sync(self, svc: ServiceDef, target: str) -> tuple[ServiceDef, bool]:
        if target == "local":
            is_running = self._detect_running(svc)
        else:
            is_running = self._detect_running_remote(svc, target)
        self._svc_states[svc.name] = is_running
        return self._service_for_target(svc, target), is_running

    async def _populate_deferred_panes(
        self,
        tabs: TabbedContent,
        svc: ServiceDef,
        pipeline: PipelineDef,
        config_scroll: VerticalScroll,
    ) -> None:
        """fill the Config/Streams/Transform/Prompts panes after the first paint.

        The user may have already navigated to another tree node; in that case
        the containers are detached and this becomes a no-op."""
        if not tabs.is_attached or not config_scroll.is_attached:
            return
        try:
            self._show_pipeline_form(config_scroll, pipeline)

            streams = self._streams_for_pipeline_target(pipeline)
            if streams or svc.name in ("ASR Base", "IPS Base", "VFA Base"):
                stream_scroll = VerticalScroll(classes="svc-launch-scroll")
                stream_pane = TabPane("Streams", stream_scroll, id="svc-tab-streams")
                await tabs.add_pane(stream_pane)
                stream_config_path = pipeline.config_path if self._get_panel_target() == "local" else ""
                panel = StreamPanel(streams, config_path=stream_config_path, project_dir=self._root)
                await stream_scroll.mount(panel)

            if svc.name == "IPS Base":
                transform_scroll = VerticalScroll(classes="svc-launch-scroll")
                transform_pane = TabPane("Transform Matrix", transform_scroll, id="svc-tab-transform")
                await tabs.add_pane(transform_pane)
                await transform_scroll.mount(self._transform_matrix_panel())

            if svc.name == "VFA Server":
                prompts_scroll = VerticalScroll(classes="svc-launch-scroll")
                prompts_pane = TabPane("Prompts", prompts_scroll, id="svc-tab-prompts")
                await tabs.add_pane(prompts_pane)
                await prompts_scroll.mount(self._vfa_prompts_panel(svc))

                aschema_scroll = VerticalScroll(classes="svc-launch-scroll")
                aschema_pane = TabPane("Action Schema", aschema_scroll, id="svc-tab-action-schema")
                await tabs.add_pane(aschema_pane)
                await aschema_scroll.mount(self._vfa_action_schema_panel(svc))
        except Exception:
            # containers can disappear mid-population when the user switches
            # nodes quickly; never let that take down the screen
            pass

    async def _mount_service_content(
        self,
        content_area: Vertical,
        svc: ServiceDef,
        is_running: bool,
    ) -> None:
        pipeline = self._pipeline_for_service(svc.name)

        if pipeline:
            self._current_pipeline = pipeline
            tabs = TabbedContent(id="svc-sub-tabs")
            await content_area.mount(tabs)

            launch_scroll = VerticalScroll(classes="svc-launch-scroll")
            card = ServiceCard(
                svc,
                is_running=is_running,
                stack_components=self._stack_component_names(svc, self._get_panel_target()),
            )
            launch_pane = TabPane("Launch", launch_scroll, id="svc-tab-launch")
            await tabs.add_pane(launch_pane)
            await launch_scroll.mount(card)

            config_scroll = VerticalScroll(classes="svc-config-scroll")
            config_pane = TabPane("Config", config_scroll, id="svc-tab-config")
            await tabs.add_pane(config_pane)
            self._config_container = config_scroll
            self._current_config_local_path = pipeline.config_path
            # defer the heavy widget building (config form has dozens of field
            # rows; streams/transform/prompts may hit disk or SSH) so the Launch
            # card paints immediately when navigating the tree
            self.call_after_refresh(self._populate_deferred_panes, tabs, svc, pipeline, config_scroll)
        elif svc.launch_type == "vllm":
            tabs = TabbedContent(id="svc-sub-tabs")
            await content_area.mount(tabs)

            launch_scroll = VerticalScroll(classes="svc-launch-scroll")
            card = ServiceCard(
                svc,
                is_running=is_running,
            )
            launch_pane = TabPane("Launch", launch_scroll, id="svc-tab-launch")
            await tabs.add_pane(launch_pane)
            await launch_scroll.mount(card)

            config_scroll = VerticalScroll(classes="svc-config-scroll")
            config_pane = TabPane("Config", config_scroll, id="svc-tab-config")
            await tabs.add_pane(config_pane)
            self._config_container = config_scroll
            self._current_config_local_path = _mllm_config_path(self._root)
            self._show_mllm_form(config_scroll)
        else:
            scroll = VerticalScroll(classes="svc-launch-scroll")
            await content_area.mount(scroll)
            await scroll.mount(ServiceCard(
                svc,
                is_running=is_running,
                initial_collection_role=self._collection_role,
            ))
            # calibration captures per-camera image folders; offer a manager to
            # browse/delete them without digging into the project on disk
            if svc.name == "IPS Camera Calibration":
                await scroll.mount(CameraManagerPanel(
                    cameras_dir=_ips_cameras_local_dir(self._root),
                    target=self._get_panel_target(),
                ))

    def _service_with_session_choices(self, svc: ServiceDef, target: str | None = None) -> ServiceDef:
        if not svc.artifact_pipeline:
            return svc
        target = target or self._get_panel_target()
        session_choices = self._artifact_session_choices_for_target(target)
        choices = [_NEW_COLLECTION_SESSION_CHOICE]
        for session_id in session_choices:
            if session_id and session_id not in choices:
                choices.append(session_id)
        experiment_group_choices = self._collection_experiment_group_choices()
        params = []
        for param in svc.params:
            if param.flag in ("-sid", "--session-id", "--artifact-session-id"):
                params.append(ParamDef(param.flag, "Session", param.param_type, _NEW_COLLECTION_SESSION_CHOICE, choices))
            elif param.flag == "--experiment-group" and experiment_group_choices:
                params.append(replace(param, default=experiment_group_choices[0], choices=experiment_group_choices))
            else:
                params.append(param)
        return replace(svc, params=params)

    def _service_with_artifact_choices(self, svc: ServiceDef) -> ServiceDef:
        return self._service_with_session_choices(svc)

    def _artifact_session_choices_for_target(self, target: str, conda_env: str = "") -> list[str]:
        # session discovery hits MongoDB/InfluxDB and is by far the slowest part
        # of building a service card; cache per target so tree navigation stays
        # snappy. Refresh on a card invalidates the cache.
        cache: dict[str, tuple[float, list[str]]] = getattr(self, "_session_choice_cache", None) or {}
        self._session_choice_cache = cache
        cached = cache.get(target)
        if cached and time.monotonic() - cached[0] < _SESSION_CHOICE_TTL_SEC:
            return cached[1]
        choices = self._fetch_artifact_session_choices(target)
        cache[target] = (time.monotonic(), choices)
        return choices

    def _invalidate_session_choice_cache(self, target: str | None = None) -> None:
        cache = getattr(self, "_session_choice_cache", None)
        if not cache:
            return
        if target is None:
            cache.clear()
        else:
            cache.pop(target, None)

    def _fetch_artifact_session_choices(self, target: str) -> list[str]:
        if target == "local":
            return _artifact_session_choices(self._root)

        profile = get_profile_by_name(target)
        if profile is None:
            return _artifact_session_choices(self._root)

        configs = []
        for rel_path in _ARTIFACT_CONFIG_RELS:
            config_path = os.path.join(self._root, rel_path)
            config, _ = self._load_config_for_target(
                config_path,
                show_status=False,
                target=target,
            )
            if config:
                configs.append(_config_for_local_db_access(config, profile))

        local_sessions = _remote_artifact_session_ids(profile) + _artifact_session_choices(self._root)
        choices = _artifact_session_choices_from_configs(configs, local_sessions)
        return choices or _artifact_session_choices(self._root)

    def _collection_experiment_group_choices(self) -> list[str]:
        data = load_experiments(self._root)
        choices: list[str] = []
        seen: set[str] = set()
        for experiment in get_active_experiments(data):
            exp_id = str(experiment.get("experiment_id") or "").strip()
            if not exp_id:
                continue
            for group_id in get_groups_for_experiment(exp_id, data):
                choice = f"{exp_id}/{group_id}"
                if choice in seen:
                    continue
                seen.add(choice)
                choices.append(choice)
        return choices

    @staticmethod
    def _collection_session_scoped_flags(svc: ServiceDef) -> set[str]:
        """flags that describe the recording session, not the recording host.

        These carry over when the Host selector changes: the same session is
        normally recorded by several machines at once."""
        return {comp.count_flag for comp in svc.components} | {
            "--session-id",
            "--experiment-group",
        }

    def _capture_infra_mode(self, target: str | None = None) -> None:
        """remember an infra card's Run mode before the card is rebuilt."""
        target = target or self._get_panel_target()
        try:
            cards = list(self.query(ServiceCard))
        except Exception:
            return
        for card in cards:
            svc = card.service_def
            if _infra_compose_service(svc) is None:
                continue
            try:
                mode = str((card.collect_params() or {}).get("--mode") or "").strip().lower()
            except Exception:
                continue
            if mode in ("native", "docker"):
                self._infra_mode[(target, svc.name)] = mode

    def _capture_collection_card_state(self, target: str | None = None) -> None:
        """remember the collection card's choices before the card is rebuilt.

        `target` names the host the card was built for; on a host switch that
        is the *previous* one, since the Select has already moved on."""
        target = target or self._get_panel_target()
        try:
            cards = list(self.query(ServiceCard))
        except Exception:
            return
        for card in cards:
            if card.service_def.launch_type != "collection":
                continue
            snapshot = card.collection_snapshot()
            if not snapshot:
                continue
            self._collection_role = snapshot.get("role") or self._collection_role
            shared_flags = self._collection_session_scoped_flags(card.service_def)
            target_values = self._collection_target_sticky.setdefault(target, {})
            for flag, value in (snapshot.get("values") or {}).items():
                if flag == "--session-id":
                    # "" records an explicit "Create MongoDB Session" pick, which
                    # must survive a host switch just like a real session id does
                    self._collection_sticky[flag] = (
                        "" if _is_new_collection_session_choice(value) else _safe_session_id(value)
                    )
                    continue
                if not isinstance(value, (bool, int)) and not str(value or "").strip():
                    continue
                if flag in shared_flags:
                    self._collection_sticky[flag] = value
                else:
                    target_values[flag] = value
            break

    def _remember_collection_session(self, session_id: object) -> None:
        """make the session a launch just resolved the default everywhere.

        The first host creates the MongoDB session; every host after it should
        record into that same session rather than creating another one."""
        session_id = _safe_session_id(session_id)
        if not session_id:
            return
        self._collection_sticky["--session-id"] = session_id
        # a brand new session is in no target's cached choice list yet
        self._invalidate_session_choice_cache()
        # keep the card on screen in step: a capture before the next rebuild
        # would otherwise read back a stale "Create MongoDB Session"
        try:
            cards = list(self.query(ServiceCard))
        except Exception:
            return
        for card in cards:
            if card.service_def.launch_type == "collection":
                card.select_collection_session(session_id)

    def _remember_collection_launch(self, target: str, session_id: object) -> None:
        session_id = _safe_session_id(session_id)
        if not session_id:
            return
        self._collection_launch_targets.setdefault(session_id, set()).add(target)

    def _service_with_collection_defaults(self, svc: ServiceDef, target: str | None = None) -> ServiceDef:
        if svc.launch_type != "collection":
            return svc

        target = target or self._get_panel_target()
        defaults = self._collection_defaults_for_current_target(target)
        session_choices = self._artifact_session_choices_for_target(target)
        experiment_group_choices = self._collection_experiment_group_choices()
        last_params = self._collection_last_params.get((target, svc.name), {})
        sticky = self._collection_sticky
        target_sticky = self._collection_target_sticky.get(target, {})
        shared_flags = self._collection_session_scoped_flags(svc)

        if "--session-id" in sticky:
            # an explicit pick (or a session a launch just created): always
            # offered, even before the databases list it
            last_session = _safe_session_id(sticky["--session-id"])
        else:
            last_session = self._collection_session_id(last_params)
            if (
                last_session
                and last_session not in session_choices
                and not self._svc_states.get(svc.name, False)
            ):
                last_session = ""
        last_experiment_group = str(
            sticky.get("--experiment-group")
            or last_params.get("--experiment-group")
            or ""
        ).strip()
        params = []
        for param in self._collection_param_defs(svc, defaults):
            default = defaults.get(param.flag, param.default)
            choices = list(param.choices)
            if param.flag in shared_flags and param.flag in sticky:
                default = sticky[param.flag]
            elif param.flag in target_sticky:
                default = target_sticky[param.flag]
            if param.flag == "--session-id":
                choices = [_NEW_COLLECTION_SESSION_CHOICE]
                for session_id in [last_session, *session_choices]:
                    if session_id and session_id not in choices:
                        choices.append(session_id)
                default = last_session or _NEW_COLLECTION_SESSION_CHOICE
            elif param.flag == "--experiment-group" and experiment_group_choices:
                choices = experiment_group_choices
                default = (
                    last_experiment_group
                    if last_experiment_group in experiment_group_choices
                    else experiment_group_choices[0]
                )
            params.append(replace(param, default=default, choices=choices))
        return replace(svc, params=params)

    @staticmethod
    def _collection_param_defs(svc: ServiceDef, defaults: dict[str, object]) -> list[ParamDef]:
        params = list(svc.params)
        existing = {param.flag for param in params}
        labels = {
            "--host-label": "Host Label",
            "--audio-interactive": "Terminal Setup",
            "--audio-input-format": "Input Format",
            "--audio-device": "Device",
            "--audio-channels": "Channel Count",
            "--audio-channel": "Channel",
            "--sample-rate": "Sample Rate",
            "--audio-format": "Audio Format",
            "--video-interactive": "Terminal Setup",
            "--video-input-format": "Input Format",
            "--video-device": "Device",
            "--video-source-format": "Video Format",
            "--framerate": "Framerate",
            "--size": "Frame Size",
            "--bitrate": "Bitrate",
            "--maxrate": "Maxrate",
            "--bufsize": "Bufsize",
            "--preset": "Preset",
            "--camera-label": "Camera Label",
        }
        video_source_choices = (
            [""]
            if str(defaults.get("--video-input-format") or "").strip() == "avfoundation"
            else ["mjpeg", "yuyv422", ""]
        )
        choices = {
            "--audio-input-format": ["alsa", "avfoundation"],
            "--audio-channel": ["mix", "0", "1", "2", "3"],
            "--sample-rate": ["8000", "16000", "22050", "24000", "44100", "48000"],
            "--audio-format": ["wav", "flac", "aac"],
            "--video-input-format": ["v4l2", "avfoundation"],
            "--video-source-format": video_source_choices,
            "--preset": ["ultrafast", "veryfast", "faster", "fast", "medium"],
        }
        bool_flags = {"--audio-interactive", "--video-interactive"}
        for component in svc.components:
            for flag in component.flags:
                if (
                    flag in existing
                    or flag in {"--session-id", "--experiment-group", "--output-root"}
                    or flag in _COLLECTION_HIDDEN_PRESET_FLAGS
                ):
                    continue
                default = defaults.get(flag, "")
                param_type = "bool" if flag in bool_flags else "str"
                params.append(ParamDef(flag, labels.get(flag, flag.lstrip("-")), param_type, default, choices.get(flag, [])))
                existing.add(flag)
        return params

    def _service_for_current_target(self, svc: ServiceDef) -> ServiceDef:
        return self._service_for_target(svc, self._get_panel_target())

    def _service_for_target(self, svc: ServiceDef, target: str) -> ServiceDef:
        svc = self._service_with_collection_defaults(svc, target)
        svc = self._service_with_session_choices(svc, target)
        svc = self._service_with_infra_mode(svc, target)
        if svc.name != "ASR Server":
            return svc
        config_path = os.path.join(svc.config_dir, "config.yml")
        config, _ = self._load_config_for_target(config_path, show_status=False, target=target)
        backend = _asr_audio_inferer_backend_from_config(config)
        return replace(
            svc,
            conda_env=_asr_server_conda_env_from_config(config),
            description=f"ASR inference services (AudioInferer: {backend})",
        )

    def _service_with_infra_mode(self, svc: ServiceDef, target: str) -> ServiceDef:
        """re-seed a rebuilt infra card with the Run mode last chosen for this host."""
        if _infra_compose_service(svc) is None:
            return svc
        mode = self._infra_mode.get((target, svc.name))
        if mode is None:
            return svc
        return replace(
            svc,
            params=[
                replace(p, default=mode) if p.flag == "--mode" else p
                for p in svc.params
            ],
        )

    def _streams_for_pipeline_target(self, pipeline: PipelineDef) -> list:
        target = self._get_panel_target()
        if target == "local":
            return load_streams(pipeline.config_path)
        config, _ = self._load_config_for_target(pipeline.config_path, show_status=False, target=target)
        return streams_from_config(config)

    def _vfa_prompts_panel(self, svc: ServiceDef) -> PromptsPanel:
        from openmmla.services.vfa.prompt_profiles import (
            DEFAULT_PROMPT_PROFILE, active_prompt_files,
        )
        target = self._get_panel_target()
        config_path = os.path.join(svc.config_dir, "config.yml")
        # read the config of the selected host so active set / profile reflect it
        config, _ = self._load_config_for_target(config_path, show_status=False, target=target)
        analyzer_config = (config or {}).get("VLLMFrameAnalyzer") or {}
        prompts_dir = str(analyzer_config.get("prompt_templates_dir") or "prompts")
        if isinstance(prompts_dir, str) and prompts_dir.strip().startswith("<") and prompts_dir.strip().endswith(">"):
            prompts_dir = "prompts"
        if not os.path.isabs(prompts_dir):
            prompts_dir = os.path.join(svc.config_dir, prompts_dir)
        profile = str(analyzer_config.get("prompt_profile") or DEFAULT_PROMPT_PROFILE)
        end_to_end = bool(analyzer_config.get("end_to_end", False))
        ssh_profile = None
        remote_dir = None
        if target != "local":
            ssh_profile = get_profile_by_name(target)
            if ssh_profile is not None:
                remote_dir = self._remote_dir_for_local(prompts_dir, ssh_profile)
        return PromptsPanel(
            prompts_dir=prompts_dir,
            active_files=active_prompt_files(profile, end_to_end),
            profile=profile,
            end_to_end=end_to_end,
            target=target,
            ssh_profiles=[p.name for p in load_ssh_profiles()] if target == "local" else [],
            ssh_profile=ssh_profile,
            remote_dir=remote_dir,
        )

    def _vfa_action_schema_panel(self, svc: ServiceDef) -> ActionSchemaPanel:
        target = self._get_panel_target()
        config_path = os.path.join(svc.config_dir, "config.yml")
        config, _ = self._load_config_for_target(config_path, show_status=False, target=target)
        analyzer_config = (config or {}).get("VLLMFrameAnalyzer") or {}
        schema_rel = str(analyzer_config.get("action_schema_path") or "config/vfa/action_schemas.yml")
        if schema_rel.strip().startswith("<") and schema_rel.strip().endswith(">"):
            schema_rel = "config/vfa/action_schemas.yml"
        # action schemas live under the repo root (centralized definitions)
        schema_path = schema_rel if os.path.isabs(schema_rel) else os.path.join(self._root, schema_rel)
        ssh_profile = None
        remote_path = None
        if target != "local":
            ssh_profile = get_profile_by_name(target)
            if ssh_profile is not None:
                remote_path = self._remote_config_path(schema_path, ssh_profile)
        return ActionSchemaPanel(
            schema_path=schema_path,
            target=target,
            ssh_profiles=[p.name for p in load_ssh_profiles()] if target == "local" else [],
            ssh_profile=ssh_profile,
            remote_path=remote_path,
        )

    def _transform_matrix_panel(self) -> TransformMatrixPanel:
        target = self._get_panel_target()
        local_dir = _ips_transform_local_dir(self._root)
        remote_dir = None
        remote_files: list[str] = []
        profile = None
        if target != "local":
            profile = get_profile_by_name(target)
            if profile is not None:
                remote_dir = self._ips_transform_remote_dir(profile)
                remote_files = _remote_transform_matrix_files(profile, remote_dir)
        return TransformMatrixPanel(
            local_dir=local_dir,
            target=target,
            remote_dir=remote_dir,
            local_files=_local_transform_matrix_files(local_dir),
            remote_files=remote_files,
            ssh_profiles=self._ssh_profile_names,
            ssh_profile=profile,
        )

    # ── config logic ─────────────────────────────────────────────

    def _show_shared_form(self, container: VerticalScroll, section_name: str) -> None:
        sec_info = SHARED_SECTIONS.get(section_name, {})
        fields = []
        for key, fdef in sec_info.get("fields", {}).items():
            fields.append(LoaderFieldDef(
                path=f"{section_name}.{key}",
                field_type=fdef["field_type"],
                default=fdef["default"],
                description=fdef["description"],
                required=True,
                section=section_name,
            ))
        values = {f.path: self._shared_values.get(f.path, f.default) for f in fields}
        form = ConfigForm(f"shared:{section_name}", fields, values)
        container.mount(form)
        self._current_form = form
        self._show_sync_bar(shared_section=section_name)

    # media file extensions used to populate the per-base file dropdown
    _SOURCE_FILE_EXTS = (
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm",
        ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac",
    )

    def _list_source_files(self, existing: dict, target: str) -> list[str]:
        """list media filenames in the configured Base.file_dir(s) so the Config
        form can offer them as a dropdown when a base's source is 'file'. Only
        resolves local, absolute directories; remote/unknown dirs fall back to a
        free-text field in the form."""
        if target != "local" or not isinstance(existing, dict):
            return []
        base = existing.get("Base")
        dirs: list[str] = []
        if isinstance(base, dict):
            fd = base.get("file_dir")
            if isinstance(fd, str):
                dirs.append(fd)
            # ASR keeps Base as a map of device -> settings (with file_dir each)
            for v in base.values():
                if isinstance(v, dict) and isinstance(v.get("file_dir"), str):
                    dirs.append(v["file_dir"])
        files: list[str] = []
        seen: set[str] = set()
        for d in dirs:
            if not d or not os.path.isabs(d) or not os.path.isdir(d):
                continue
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for fn in names:
                low = fn.lower()
                if fn not in seen and any(low.endswith(e) for e in self._SOURCE_FILE_EXTS):
                    seen.add(fn)
                    files.append(fn)
        return files

    def _show_pipeline_form(self, container: VerticalScroll, pipeline: PipelineDef) -> None:
        existing, source_message = self._load_config_for_target(pipeline.config_path)
        apply_shared_values(pipeline.fields, self._shared_values)

        # populate Bases entry dropdowns from the config (camera <- Cameras,
        # base_type <- Base) so users pick existing values instead of typing
        if isinstance(existing, dict):
            cameras = sorted((existing.get("Cameras") or {}).keys())
            base_types = sorted((existing.get("Base") or {}).keys())
            source_types = {
                "ASR Base": ["udp", "tcp", "pyaudio", "rtmp", "lsl", "file"],
                "IPS Base": ["opencv", "rtmp", "lsl", "file"],
                "VFA Base": ["opencv", "rtmp", "lsl", "file"],
            }.get(pipeline.name, [])
            source_files = self._list_source_files(existing, self._get_panel_target())
            for f in pipeline.fields:
                if f.field_type == "list_of_dicts" and f.path == "Bases":
                    choices = {}
                    if "camera" in (f.entry_schema or {}):
                        choices["camera"] = cameras
                    if "base_type" in (f.entry_schema or {}):
                        choices["base_type"] = base_types
                    if "source" in (f.entry_schema or {}) and source_types:
                        choices["source"] = source_types
                    # files for the per-base file dropdown (consumed by
                    # _source_index_widget only when that base's source is 'file')
                    if "source_index" in (f.entry_schema or {}):
                        choices["source_index"] = source_files
                    f.entry_field_choices = choices

        _src_target = self._get_panel_target()
        is_local_host = _src_target == "local"
        overrides = pipeline_section_overrides(existing)
        readonly_paths: set[str] = set()
        values = {}
        sources: dict[str, str] = {}
        for f in pipeline.fields:
            top_section = f.path.split(".")[0]
            is_shared = top_section in SHARED_SECTION_NAMES
            existing_val = get_nested_value(existing, f.path)
            # origin of the displayed value: the target's own config ("target"),
            # the local System Services store used as a gap-filler ("shared"),
            # or the template default ("default").
            if existing_val is not None:
                values[f.path] = existing_val
                origin = "target"
            elif self._shared_values.get(f.path) is not None:
                values[f.path] = self._shared_values[f.path]
                origin = "shared"
            else:
                values[f.path] = f.default
                origin = "default"

            if is_shared and top_section not in overrides:
                # managed centrally; read-only. The tag reflects where the shown
                # value actually comes from so a remote view distinguishes a real
                # remote value from a local fallback stand-in.
                readonly_paths.add(f.path)
                if is_local_host:
                    sources[f.path] = "[yellow]· managed in System Services[/yellow]"
                elif origin == "target":
                    sources[f.path] = f"[cyan]· from {_src_target}[/cyan]"
                elif origin == "shared":
                    sources[f.path] = "[yellow]· local shared (fallback)[/yellow]"
                else:
                    sources[f.path] = "[dim]· default[/dim]"
            elif is_shared and top_section in overrides:
                sources[f.path] = "[magenta]· override (pinned here)[/magenta]"
            elif origin == "target":
                sources[f.path] = (
                    "[cyan]· config.yml[/cyan]" if is_local_host
                    else f"[cyan]· from {_src_target}[/cyan]"
                )
            elif origin == "shared":
                sources[f.path] = "[yellow]· local shared[/yellow]"
            else:
                sources[f.path] = "[dim]· default[/dim]"

        dynamic_sections: dict[str, list[LoaderFieldDef]] = {}
        if pipeline.base_template and pipeline.base_section:
            base_data = existing.get(pipeline.base_section, {})
            if isinstance(base_data, dict):
                for device_name in base_data:
                    section_name = f"{pipeline.base_section}.{device_name}"
                    fields = self._clone_template_fields(pipeline, section_name)
                    dynamic_sections[section_name] = fields
                    for f in fields:
                        val = get_nested_value(existing, f.path)
                        if val is not None:
                            values[f.path] = val

        if pipeline.name == "Nginx":
            existing_upstreams = existing.get("upstreams", {})
            template_names = {
                f.path.split(".", 1)[1]
                for f in pipeline.fields if f.path.startswith("upstreams.")
            }
            for svc_name, entries in existing_upstreams.items():
                if svc_name not in template_names and isinstance(entries, list):
                    values[f"upstreams.{svc_name}"] = entries

        known_sections = {f.path.split(".")[0] for f in pipeline.fields}
        if pipeline.base_section:
            known_sections.add(pipeline.base_section)
        for key, value in existing.items():
            if key not in known_sections and isinstance(value, dict):
                extra_fields = fields_from_config_section(key, value)
                if extra_fields:
                    dynamic_sections[key] = extra_fields
                    for f in extra_fields:
                        val = get_nested_value(existing, f.path)
                        if val is not None:
                            values[f.path] = val

        if pipeline.name in _STREAM_PIPELINES:
            streams_data = existing.get("Streams", {})
            if isinstance(streams_data, dict):
                for stream_name, stream_props in streams_data.items():
                    if not isinstance(stream_props, dict):
                        continue
                    section_name = f"Streams.{stream_name}"
                    s_fields = _make_stream_fields(stream_name)
                    dynamic_sections[section_name] = s_fields
                    for f in s_fields:
                        val = get_nested_value(existing, f.path)
                        if val is not None:
                            values[f.path] = val
            fields_to_remove = [f for f in pipeline.fields if f.path.startswith("Streams")]
            for f in fields_to_remove:
                pipeline.fields.remove(f)

        group_add_buttons = {}
        if pipeline.base_template and pipeline.base_section:
            group_add_buttons[pipeline.base_section] = ("+ Add Base", "btn-add-base")
        if pipeline.name in _STREAM_PIPELINES:
            group_add_buttons["Streams"] = ("+ Add Stream", "btn-add-stream")

        form = ConfigForm(pipeline.name, pipeline.fields, values, dynamic_sections,
                          group_add_buttons=group_add_buttons,
                          base_section=pipeline.base_section or None,
                          sources=sources,
                          readonly_paths=readonly_paths,
                          shared_sections=set(SHARED_SECTION_NAMES),
                          overridden_sections=overrides,
                          allow_override_toggle=True)
        container.mount(form)
        self._current_form = form
        if source_message:
            self._show_status(source_message)
        self._show_sync_bar(pipeline)

    def _show_mllm_form(self, container: VerticalScroll) -> None:
        values = _mllm_form_values(self._root)
        form = ConfigForm("MLLM Server", _MLLM_FIELDS, values)
        container.mount(form)
        self._current_form = form
        self._show_sync_bar(local_path=_mllm_config_path(self._root))

    def on_config_form_override_toggled(self, event: ConfigForm.OverrideToggled) -> None:
        """toggle SystemServicesOverride for a shared section on the current
        target's pipeline config, then reload so the fields flip
        read-only/editable. Works on the local host and on a remote host."""
        pipeline = self._current_pipeline
        if pipeline is None:
            return
        section = event.section_name
        target = self._get_panel_target()
        config, _ = self._load_config_for_target(
            pipeline.config_path, show_status=False, target=target)
        if not isinstance(config, dict):
            config = {}
        else:
            config = dict(config)
        overrides = set(pipeline_section_overrides(config))
        if section in overrides:
            overrides.discard(section)
            action = f"{section} is now managed centrally (System Services)"
        else:
            overrides.add(section)
            action = f"{section} is now overridden on {pipeline.name} (edit it here)"
        if overrides:
            config["SystemServicesOverride"] = sorted(overrides)
        else:
            config.pop("SystemServicesOverride", None)

        cache_key = self._config_cache_key(pipeline.config_path, target)
        if target == "local":
            os.makedirs(os.path.dirname(pipeline.config_path), exist_ok=True)
            with open(pipeline.config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
            self._target_config_cache.pop(cache_key, None)
            self._show_status(action)
            self.run_worker(self._reload_current_service_view(), exclusive=True)
            return

        # remote host: write the toggled config back over scp
        profile = get_profile_by_name(target)
        if profile is None:
            self._show_status(f"SSH profile '{target}' not found; override not changed.")
            return
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".yml", prefix="openmmla-override-", delete=False, encoding="utf-8")
        with tmp:
            yaml.safe_dump(config, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
        remote_path = self._remote_config_path(pipeline.config_path, profile)
        # optimistically reflect the new state so the reloaded form is correct;
        # _run_scp confirms/keeps it on success.
        self._target_config_cache[cache_key] = config
        self._show_status(f"{action}; saving to {target} ...")
        self.run_worker(
            self._run_scp(target, tmp.name, remote_path, cleanup_local=True,
                          cache_key=cache_key, cache_config=config),
            group="override-scp", exclusive=True,
        )
        self.run_worker(self._reload_current_service_view(), exclusive=True)

    def on_config_form_saved(self, event: ConfigForm.Saved) -> None:
        if event.pipeline_name.startswith("shared:"):
            for path, val in event.values.items():
                self._shared_values[path] = val
            section_name = event.pipeline_name.replace("shared:", "", 1)
            config_path = save_system_service_section(
                self._root,
                section_name,
                self._shared_section_data(section_name),
            )
            updated = self._apply_shared_section_to_local_configs(section_name)
            self._show_status(
                f"{section_name} system service saved to {config_path} and {updated} local pipeline config(s)"
            )
            return

        if event.pipeline_name == "MLLM Server":
            config_path = _mllm_config_path(self._root)
            save_config(config_path, _MLLM_FIELDS, event.values)
            self._services = _build_service_registry(self._root)
            self._svc_map = {s.name: s for s in self._services}
            self._show_status(f"Saved launch config locally to {config_path}")
            self._build_tree()
            return

        pipeline = self._pipeline_map.get(event.pipeline_name)
        if pipeline is None:
            return

        form = self._current_form
        all_fields = form.all_fields if form else pipeline.fields
        self._save_pipeline_config_for_target(pipeline, all_fields, event.values)

        if pipeline.name == "ASR Server":
            self._services = _build_service_registry(self._root)
            self._svc_map = {s.name: s for s in self._services}
            self._refresh_service_cards()

        if self._get_panel_target() == "local":
            self._refresh_stream_panels(pipeline)
        self._refresh_vfa_prompts(pipeline)
        self._show_sync_bar(pipeline)
        self._build_tree()

    def _refresh_vfa_prompts(self, pipeline: PipelineDef) -> None:
        """recompute Prompts-tab active set after a VFA Server config save."""
        if pipeline.name != "VFA Server":
            return
        try:
            from openmmla.services.vfa.prompt_profiles import (
                DEFAULT_PROMPT_PROFILE, active_prompt_files,
            )
            config, _ = self._load_config_for_target(
                pipeline.config_path, show_status=False, target=self._get_panel_target()
            )
            analyzer = (config or {}).get("VLLMFrameAnalyzer") or {}
            profile = str(analyzer.get("prompt_profile") or DEFAULT_PROMPT_PROFILE)
            end_to_end = bool(analyzer.get("end_to_end", False))
            active = active_prompt_files(profile, end_to_end)
        except Exception:
            return
        for panel in self.query(PromptsPanel):
            panel.refresh_active(active, profile, end_to_end)

    def _refresh_stream_panels(self, pipeline: PipelineDef) -> None:
        """refresh stream tabs after a pipeline config save."""
        if pipeline.name not in _STREAM_PIPELINES:
            return
        streams = load_streams(pipeline.config_path)
        for panel in self.query(StreamPanel):
            panel.update_streams(streams)

    def _refresh_service_cards(self) -> None:
        """refresh metadata on any mounted service card after registry changes."""
        for card in self.query(ServiceCard):
            service = self._svc_map.get(card.service_def.name)
            if service is not None:
                card.update_service_def(self._service_for_current_target(service))

    def _apply_shared_section_to_local_configs(self, section_name: str) -> int:
        section_data = self._shared_section_data(section_name)
        if not section_data:
            return 0
        if not self._pipelines:
            self._pipelines = discover_pipelines()
            self._pipeline_map = {pipeline.name: pipeline for pipeline in self._pipelines}
        updated = 0
        for pipeline in self._pipelines:
            if not self._pipeline_has_section(pipeline, section_name):
                continue
            config = load_existing_config(pipeline.config_path)
            if not isinstance(config, dict):
                config = {}
            # a pipeline that pins this section (override) manages it itself and
            # must not be overwritten from the central store.
            if section_name in pipeline_section_overrides(config):
                continue
            if config.get(section_name) == section_data:
                continue
            config[section_name] = dict(section_data)
            os.makedirs(os.path.dirname(pipeline.config_path), exist_ok=True)
            with open(pipeline.config_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
            updated += 1
        self._target_config_cache.clear()
        self._shared_values.update(
            {f"{section_name}.{key}": value for key, value in section_data.items()}
        )
        return updated

    def _show_status(self, message: str) -> None:
        container = self._config_container
        if container is None:
            return
        for old in container.query(".status-saved"):
            old.remove()
        container.mount(Static(f" {message}", classes="status-saved"))

    def _show_sync_bar(
        self,
        pipeline: PipelineDef | None = None,
        shared_section: str | None = None,
        local_path: str | None = None,
    ) -> None:
        container = self._config_container
        if container is None:
            return
        for old in container.query(".sync-bar"):
            old.remove()
        target = self._get_panel_target()
        if target != "local":
            # On a remote host, Save already writes directly to that host, so we
            # don't show a separate sync button here. To push a locally-edited
            # config to a remote, switch Host to local and use the SSH-profile
            # picker + "Sync to Remote" below.
            return

        if pipeline is None and local_path is None and shared_section is None:
            return
        profiles = load_ssh_profiles()
        if not profiles:
            return
        options = [(p.name, p.name) for p in profiles]
        selected_profile = target if any(p.name == target for p in profiles) else None
        select_kwargs = {}
        if selected_profile is not None:
            select_kwargs["value"] = selected_profile
        bar = Horizontal(
            Select(
                options,
                prompt="Select SSH profile...",
                id="sync-profile-select",
                **select_kwargs,
            ),
            Button("Sync to Remote", variant="warning", id="btn-sync-remote"),
            classes="sync-bar",
        )
        container.mount(bar)

    def _clone_template_fields(self, pipeline: PipelineDef, new_name: str) -> list[LoaderFieldDef]:
        """clone base_template fields with paths rewritten to a new section name."""
        old_prefix = pipeline.base_template_prefix
        fields = []
        for f in pipeline.base_template:
            new_path = new_name + f.path[len(old_prefix):]
            new_section = new_name + f.section[len(old_prefix):]
            fields.append(LoaderFieldDef(
                path=new_path,
                field_type=f.field_type,
                default=f.default,
                description=f.description,
                required=f.required,
                section=new_section,
                choices=list(f.choices),
            ))
        return fields

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "svc-target-refresh":
            self._log("[yellow]Testing connections to all hosts...[/yellow]")
            self.run_worker(
                self._async_manual_probe(),
                group="launcher-target-probe",
                exclusive=True,
            )
            return
        if event.button.id == "btn-sync-remote":
            self._sync_to_remote()
        elif event.button.id == "btn-sync-local-target":
            self._sync_local_to_selected_target()
        elif event.button.id == "btn-sync-transform-remote":
            self._sync_transform_to_remote()
        elif event.button.id == "btn-sync-transform-local-target":
            self._sync_transform_local_to_selected_target()
        elif event.button.id == "btn-sync-prompts-remote":
            self._sync_prompts_to_remote()
        elif event.button.id == "btn-sync-aschema-remote":
            self._sync_action_schema_to_remote()
        elif event.button.id == "btn-add-base":
            self._show_add_base_input()
        elif event.button.id == "btn-confirm-add-base":
            self._confirm_add_base()
        elif event.button.id == "btn-cancel-add-base":
            self._cancel_add_base()
        elif event.button.id == "btn-add-stream":
            self._show_add_stream_input()
        elif event.button.id == "btn-confirm-add-stream":
            self._confirm_add_stream()
        elif event.button.id == "btn-cancel-add-stream":
            self._cancel_add_stream()

    def _show_add_base_input(self) -> None:
        form = self._current_form
        if form is None:
            return
        try:
            form.query_one("#btn-add-base").remove()
        except Exception:
            pass
        bar = Horizontal(
            Input(placeholder="Device name (e.g. Jabra)", id="add-base-input"),
            Button("Add", variant="success", id="btn-confirm-add-base"),
            Button("Cancel", id="btn-cancel-add-base"),
            classes="add-base-bar",
        )
        try:
            form.mount(bar, before=form.query_one(".form-actions"))
        except Exception:
            form.mount(bar)

    def _confirm_add_base(self) -> None:
        try:
            inp = self.query_one("#add-base-input", Input)
            name = inp.value.strip()
        except Exception:
            return
        if not name:
            return

        pipeline = self._current_pipeline
        form = self._current_form
        if not pipeline or not pipeline.base_template or not form:
            return

        section_name = f"{pipeline.base_section}.{name}"
        fields = self._clone_template_fields(pipeline, section_name)
        form.add_section(section_name, fields, {})
        self._restore_add_base_button()

    def _cancel_add_base(self) -> None:
        self._restore_add_base_button()

    def _restore_add_base_button(self) -> None:
        form = self._current_form
        if form is None:
            return
        for old in form.query(".add-base-bar"):
            old.remove()
        btn = Button("+ Add Base", variant="success", id="btn-add-base")
        try:
            form.mount(btn, before=form.query_one(".form-actions"))
        except Exception:
            form.mount(btn)

    # ── stream add/delete ─────────────────────────────────────────

    def _show_add_stream_input(self) -> None:
        form = self._current_form
        if form is None:
            return
        try:
            form.query_one("#btn-add-stream").remove()
        except Exception:
            pass
        bar = Horizontal(
            Input(placeholder="Stream name (e.g. cam-1)", id="add-stream-input"),
            Button("Add", variant="success", id="btn-confirm-add-stream"),
            Button("Cancel", id="btn-cancel-add-stream"),
            classes="add-stream-bar",
        )
        try:
            form.mount(bar, before=form.query_one(".form-actions"))
        except Exception:
            form.mount(bar)

    def _confirm_add_stream(self) -> None:
        try:
            inp = self.query_one("#add-stream-input", Input)
            name = inp.value.strip()
        except Exception:
            return
        if not name:
            return

        form = self._current_form
        if form is None:
            return

        section_name = f"Streams.{name}"
        fields = _make_stream_fields(name)
        form.add_section(section_name, fields, {})
        self._restore_add_stream_button()

    def _cancel_add_stream(self) -> None:
        self._restore_add_stream_button()

    def _restore_add_stream_button(self) -> None:
        form = self._current_form
        if form is None:
            return
        for old in form.query(".add-stream-bar"):
            old.remove()
        btn = Button("+ Add Stream", variant="success", id="btn-add-stream")
        try:
            form.mount(btn, before=form.query_one(".form-actions"))
        except Exception:
            form.mount(btn)

    def _config_cache_key(self, local_path: str, target: str | None = None) -> tuple[str, str]:
        return (target or self._get_panel_target(), os.path.abspath(local_path))

    def _load_config_for_target(
        self,
        local_path: str,
        show_status: bool = True,
        target: str | None = None,
    ) -> tuple[dict, str]:
        target = target or self._get_panel_target()
        key = self._config_cache_key(local_path, target)
        if key in self._target_config_cache:
            return self._target_config_cache[key], ""

        if target == "local":
            config = load_existing_config(local_path)
            self._target_config_cache[key] = config
            return config, ""

        profile = get_profile_by_name(target)
        if profile is None:
            return {}, f"SSH profile '{target}' not found; using defaults."

        remote_path = self._remote_config_path(local_path, profile)
        quoted_path = _quote_remote_path(remote_path)
        cmd = (
            f"if [ -f {quoted_path} ]; then "
            f"cat {quoted_path}; "
            "else printf '__OPENMMLA_CONFIG_MISSING__\\n'; fi"
        )
        try:
            result = ssh_run_sync(profile, cmd, timeout=10.0)
        except Exception as e:
            return {}, f"Could not read {target}:{remote_path}: {e}. Using defaults."

        if result.returncode != 0:
            error = (result.stderr or "").strip() or f"exit code {result.returncode}"
            return {}, f"Could not read {target}:{remote_path}: {error}. Using defaults."

        raw = result.stdout or ""
        if raw.strip() == "__OPENMMLA_CONFIG_MISSING__":
            config = {}
            self._target_config_cache[key] = config
            return config, f"No remote config at {target}:{remote_path}; using defaults."

        try:
            config = yaml.safe_load(raw) or {}
        except yaml.YAMLError as e:
            return {}, f"Invalid remote config at {target}:{remote_path}: {e}. Using defaults."
        if not isinstance(config, dict):
            config = {}
        self._target_config_cache[key] = config
        return (
            config,
            f"Loaded config from {target}:{remote_path}" if show_status else "",
        )

    def _save_pipeline_config_for_target(
        self,
        pipeline: PipelineDef,
        fields: list[LoaderFieldDef],
        values: dict,
    ) -> None:
        target = self._get_panel_target()
        if target == "local":
            save_config(pipeline.config_path, fields, values)
            self._target_config_cache[self._config_cache_key(pipeline.config_path, "local")] = (
                load_existing_config(pipeline.config_path)
            )
            self._show_status(f"Saved locally to {pipeline.config_path}")
            return

        profile = get_profile_by_name(target)
        if profile is None:
            self._show_status(f"SSH profile '{target}' not found; config was not saved.")
            return

        tmp = tempfile.NamedTemporaryFile(
            "w",
            suffix=".yml",
            prefix="openmmla-config-",
            delete=False,
        )
        tmp_path = tmp.name
        tmp.close()
        save_config(tmp_path, fields, values)
        cache_key = self._config_cache_key(pipeline.config_path, target)
        cache_config = load_existing_config(tmp_path)

        remote_path = self._remote_config_path(pipeline.config_path, profile)
        self._show_status(f"Saving to {target}:{remote_path} ...")
        self.run_worker(
            self._run_scp(
                target,
                tmp_path,
                remote_path,
                cleanup_local=True,
                cache_key=cache_key,
                cache_config=cache_config,
            ),
            exclusive=True,
        )

    def _sync_to_remote(self) -> None:
        try:
            sel = self.query_one("#sync-profile-select", Select)
            val = sel.value
            if val is Select.BLANK or val is None:
                self._show_status("Select an SSH profile first.")
                return
            profile_name = str(val)
        except Exception:
            return

        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._show_status(f"SSH profile '{profile_name}' not found.")
            return

        # System Services shared-section view: push just this section to the
        # selected host (respecting any per-pipeline overrides on that host).
        if self._current_shared_section:
            self._sync_shared_section_to_target(self._current_shared_section, profile_name)
            return

        local_path = self._current_pipeline.config_path if self._current_pipeline is not None else self._current_config_local_path
        if not local_path:
            return
        self._sync_local_config_path_to_profile(local_path, profile_name)

    def _sync_local_to_selected_target(self) -> None:
        target = self._get_panel_target()
        if target == "local":
            return
        if self._current_shared_section:
            self._sync_shared_section_to_target(self._current_shared_section, target)
            return
        local_path = self._current_pipeline.config_path if self._current_pipeline is not None else self._current_config_local_path
        if not local_path:
            self._show_status("No local config is selected for syncing.")
            return
        self._sync_local_config_path_to_profile(local_path, target)

    def _sync_local_config_path_to_profile(self, local_path: str, profile_name: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._show_status(f"SSH profile '{profile_name}' not found.")
            return
        if not os.path.isfile(local_path):
            self._show_status(f"Local config not found: {local_path}. Save first.")
            return
        remote_path = self._remote_config_path(local_path, profile)
        cache_key = self._config_cache_key(local_path, profile_name)
        cache_config = load_existing_config(local_path)
        self._show_status(f"Syncing to {profile_name}:{remote_path} ...")
        self.run_worker(
            self._run_scp(
                profile_name,
                local_path,
                remote_path,
                cache_key=cache_key,
                cache_config=cache_config,
            ),
            exclusive=True,
        )

    def _sync_transform_to_remote(self) -> None:
        try:
            sel = self.query_one("#transform-sync-profile-select", Select)
            val = sel.value
            if val is Select.BLANK or val is None:
                self._show_status("Select an SSH profile first.")
                return
            profile_name = str(val)
        except Exception:
            return
        self._sync_transform_dir_to_profile(profile_name)

    def _sync_transform_local_to_selected_target(self) -> None:
        target = self._get_panel_target()
        if target == "local":
            return
        self._sync_transform_dir_to_profile(target)

    def _sync_transform_dir_to_profile(self, profile_name: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._show_status(f"SSH profile '{profile_name}' not found.")
            return
        local_dir = _ips_transform_local_dir(self._root)
        files = _local_transform_matrix_files(local_dir)
        if not files:
            self._show_status(f"No transformation_matrices*.json files found in {local_dir}.")
            return
        remote_dir = self._ips_transform_remote_dir(profile)
        self._show_status(f"Syncing IPS transform matrices to {profile_name}:{remote_dir} ...")
        self.run_worker(
            self._run_transform_matrix_sync(profile_name, local_dir, remote_dir, files),
            exclusive=True,
        )

    def _remote_dir_for_local(self, local_dir: str, profile) -> str:
        rel = os.path.relpath(local_dir, self._root)
        return _remote_path_join(profile.remote_project_path, rel)

    def _sync_prompts_to_remote(self) -> None:
        try:
            sel = self.query_one("#prompts-sync-profile-select", Select)
            val = sel.value
            if val is Select.BLANK or val is None:
                self._show_status("Select an SSH profile first.")
                return
            profile_name = str(val)
        except Exception:
            return
        try:
            panel = self.query_one(PromptsPanel)
        except Exception:
            return
        local_dir = panel.prompts_dir
        if not os.path.isdir(local_dir):
            self._show_status(f"No prompts directory: {local_dir}")
            return
        files = [
            f for f in os.listdir(local_dir)
            if f.endswith(".txt") and not f.startswith(".")
        ]
        if not files:
            self._show_status("No prompt files to sync.")
            return
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._show_status(f"SSH profile '{profile_name}' not found.")
            return
        remote_dir = self._remote_dir_for_local(local_dir, profile)
        self._show_status(f"Syncing prompts to {profile_name}:{remote_dir} ...")
        self.run_worker(
            self._run_files_sync(profile_name, local_dir, remote_dir, files, "prompt"),
            exclusive=True,
        )

    def _sync_action_schema_to_remote(self) -> None:
        try:
            sel = self.query_one("#aschema-sync-profile-select", Select)
            val = sel.value
            if val is Select.BLANK or val is None:
                self._show_status("Select an SSH profile first.")
                return
            profile_name = str(val)
        except Exception:
            return
        try:
            panel = self.query_one(ActionSchemaPanel)
        except Exception:
            return
        local_path = panel.schema_path
        if not os.path.isfile(local_path):
            self._show_status(f"No action schema file: {local_path}")
            return
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._show_status(f"SSH profile '{profile_name}' not found.")
            return
        remote_path = self._remote_config_path(local_path, profile)
        self._show_status(f"Syncing action schema to {profile_name}:{remote_path} ...")
        self.run_worker(
            self._run_scp(profile_name, local_path, remote_path),
            exclusive=True,
        )

    async def _run_files_sync(
        self,
        profile_name: str,
        local_dir: str,
        remote_dir: str,
        files: list[str],
        label: str,
    ) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        mkdir_proc = await ssh_run_async(profile, f"mkdir -p {_quote_remote_path(remote_dir)}")
        await mkdir_proc.wait()
        copied = 0
        failures: list[str] = []
        for name in files:
            local_path = os.path.join(local_dir, name)
            remote_path = _remote_path_join(remote_dir, name)
            proc = await scp_file_async(profile, local_path, remote_path)
            assert proc.stdout is not None
            output = ""
            async for line in proc.stdout:
                output += line.decode(errors="replace")
            rc = await proc.wait()
            if rc == 0:
                copied += 1
            else:
                detail = output.strip() or f"exit code {rc}"
                failures.append(f"{name}: {detail}")
        if failures:
            self._show_status(f"{label} sync failed: {'; '.join(failures[:2])}")
            return
        self._show_status(f"Synced {copied} {label} file(s) to {profile_name}:{remote_dir}")

    async def _run_remote_streamed(
        self, profile_name: str, cmd: str, ok_msg: str, fail_msg: str
    ) -> None:
        """Run a remote command directly over SSH, streaming output to the log.

        Used for tmux start/stop and other remote control actions. Sending the
        command straight through ssh args (rather than an interactive Terminal +
        AppleScript) avoids the nested-quote truncation that previously broke
        remote launches."""
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        self._log(ok_msg if rc == 0 else f"[red]{fail_msg} (exit {rc}).[/red]")
        self._refresh_service_cards()

    def _sync_shared_section_to_target(self, section_name: str, target: str) -> None:
        profile = get_profile_by_name(target)
        if profile is None:
            self._show_status(f"SSH profile '{target}' not found.")
            return
        section_data = self._shared_section_data(section_name)
        if not section_data:
            self._show_status(f"No shared defaults found for {section_name}.")
            return

        entries: list[tuple[str, str, tuple[str, str], dict]] = []
        temp_paths: list[str] = []
        for pipeline in self._pipelines:
            if not self._pipeline_has_section(pipeline, section_name):
                continue
            remote_config, _ = self._load_config_for_target(
                pipeline.config_path,
                show_status=False,
                target=target,
            )
            if not isinstance(remote_config, dict):
                remote_config = {}
            # respect a pipeline that pins this section locally.
            if section_name in pipeline_section_overrides(remote_config):
                continue
            remote_config[section_name] = dict(section_data)

            tmp = tempfile.NamedTemporaryFile(
                "w",
                suffix=".yml",
                prefix="openmmla-shared-config-",
                delete=False,
                encoding="utf-8",
            )
            with tmp:
                yaml.safe_dump(remote_config, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
            temp_paths.append(tmp.name)
            remote_path = self._remote_config_path(pipeline.config_path, profile)
            cache_key = self._config_cache_key(pipeline.config_path, target)
            entries.append((tmp.name, remote_path, cache_key, remote_config))

        if not entries:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self._show_status(f"No pipeline configs contain {section_name}.")
            return

        for key, value in section_data.items():
            self._shared_values[f"{section_name}.{key}"] = value

        self._show_status(f"Syncing {section_name} system service to {target} ...")
        self.run_worker(
            self._run_scp_batch(target, entries, cleanup_local=True, success_message=f"Synced {section_name} system service"),
            exclusive=True,
        )

    def _shared_section_data(self, section_name: str) -> dict[str, object]:
        info = SHARED_SECTIONS.get(section_name, {})
        values = self._current_form.collect_values() if self._current_form is not None else {}
        section_data: dict[str, object] = {}
        for key, fdef in info.get("fields", {}).items():
            path = f"{section_name}.{key}"
            section_data[key] = values.get(path, self._shared_values.get(path, fdef.get("default", "")))
        return section_data

    def _central_shared_sections(self) -> dict[str, dict]:
        """Return shared sections the user has explicitly saved to the central
        System Services store (``config/system_services.yml``).

        Only sections actually present in that file are returned, so a section
        that has never been saved centrally is left untouched at launch (its
        pipeline value is never clobbered by an unset default).
        """
        stored = load_system_services_config(self._root)
        result: dict[str, dict] = {}
        if isinstance(stored, dict):
            for name in SHARED_SECTION_NAMES:
                if name == "Sudo":
                    continue  # local credential; never synced into pipeline configs
                data = stored.get(name)
                if isinstance(data, dict) and data:
                    result[name] = data
        return result

    def _apply_central_sections_to_config_file(self, config_path: str, central: dict[str, dict],
                                               sections: list[str]) -> None:
        """Rewrite the named shared sections in a local pipeline config from the
        central store, in place."""
        config = load_existing_config(config_path)
        if not isinstance(config, dict):
            config = {}
        for section_name in sections:
            if section_name in central:
                config[section_name] = dict(central[section_name])
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        self._target_config_cache.pop(self._config_cache_key(config_path, "local"), None)

    def _sync_shared_sections_to_target(self, section_names: list[str], target: str,
                                        central: dict[str, dict]) -> None:
        """Push the named central shared sections into every remote pipeline
        config on ``target`` (respecting per-pipeline overrides), in one scp
        batch."""
        profile = get_profile_by_name(target)
        if profile is None:
            self._show_status(f"SSH profile '{target}' not found.")
            return
        section_names = [s for s in section_names if s in central]
        if not section_names:
            return
        if not self._pipelines:
            self._pipelines = discover_pipelines()
            self._pipeline_map = {p.name: p for p in self._pipelines}
        entries: list[tuple[str, str, tuple[str, str], dict]] = []
        temp_paths: list[str] = []
        for pipeline in self._pipelines:
            relevant = [s for s in section_names if self._pipeline_has_section(pipeline, s)]
            if not relevant:
                continue
            remote_config, _ = self._load_config_for_target(
                pipeline.config_path, show_status=False, target=target)
            if not isinstance(remote_config, dict):
                remote_config = {}
            overrides = pipeline_section_overrides(remote_config)
            wrote = False
            for s in relevant:
                if s in overrides:
                    continue
                remote_config[s] = dict(central[s])
                wrote = True
            if not wrote:
                continue
            tmp = tempfile.NamedTemporaryFile(
                "w", suffix=".yml", prefix="openmmla-shared-config-", delete=False, encoding="utf-8")
            with tmp:
                yaml.safe_dump(remote_config, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
            temp_paths.append(tmp.name)
            remote_path = self._remote_config_path(pipeline.config_path, profile)
            cache_key = self._config_cache_key(pipeline.config_path, target)
            entries.append((tmp.name, remote_path, cache_key, remote_config))
        if not entries:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            return
        self._show_status(f"Syncing {', '.join(section_names)} to {target} ...")
        self.run_worker(
            self._run_scp_batch(target, entries, cleanup_local=True,
                                success_message=f"Synced system services to {target}"),
            exclusive=True,
        )

    def _reconcile_shared_sections_before_launch(self, svc: ServiceDef, target: str,
                                                is_remote: bool) -> bool:
        """Reconcile a pipeline's shared system-service sections with the central
        store just before launch. Returns True to proceed, False to abort.

        Local: drifted sections are rewritten from the central store in place,
        then launch proceeds. Remote: on drift the latest values are pushed and
        this launch is aborted so the operator relaunches against the now
        up-to-date remote config (avoids racing the async push).
        """
        central = self._central_shared_sections()
        if not central:
            return True  # nothing saved centrally; leave pipeline configs as-is
        config_path = os.path.join(svc.config_dir, "config.yml")
        if not is_remote:
            config = load_existing_config(config_path)
            overrides = pipeline_section_overrides(config)
            drifted = shared_section_drift(central, config, overrides=overrides)
            if drifted:
                self._apply_central_sections_to_config_file(config_path, central, drifted)
                self._log(
                    f"[yellow]Updated {', '.join(drifted)} from System Services before "
                    f"launch (local config was out of date).[/yellow]"
                )
            return True
        # remote
        if get_profile_by_name(target) is None:
            return True  # cannot verify; existing checks already warned
        remote_config, _ = self._load_config_for_target(config_path, show_status=False, target=target)
        overrides = pipeline_section_overrides(remote_config)
        drifted = shared_section_drift(central, remote_config, overrides=overrides)
        if not drifted:
            return True
        self._log(
            f"[yellow]System-services config on '{target}' is out of date "
            f"({', '.join(drifted)}); pushing latest from System Services...[/yellow]"
        )
        self._sync_shared_sections_to_target(drifted, target, central)
        self._log(f"[yellow]Relaunch {svc.name} once the sync above completes.[/yellow]")
        return False

    @staticmethod
    def _pipeline_has_section(pipeline: PipelineDef, section_name: str) -> bool:
        prefix = f"{section_name}."
        return any(
            field.path.startswith(prefix)
            or field.section == section_name
            or field.section.startswith(prefix)
            for field in pipeline.fields
        )

    def _remote_config_path(self, local_path: str, profile) -> str:
        rel_path = os.path.relpath(local_path, self._root)
        return _remote_path_join(profile.remote_project_path, rel_path)

    def _remote_config_exists(self, svc: ServiceDef, target: str) -> tuple[bool | None, str]:
        """Check over SSH whether the remote config.yml for a service exists.

        Returns a (exists, remote_path) tuple. ``exists`` is True/False when it
        could be determined, or None when the check could not run (missing SSH
        profile or SSH error) so the caller does not block the launch.
        """
        profile = get_profile_by_name(target)
        if profile is None:
            return None, "config.yml"
        local_path = os.path.join(svc.config_dir, "config.yml")
        remote_path = self._remote_config_path(local_path, profile)
        quoted = _quote_remote_path(remote_path)
        cmd = f"if [ -f {quoted} ]; then printf FOUND; else printf MISSING; fi"
        try:
            result = ssh_run_sync(profile, cmd, timeout=8.0)
        except Exception:
            return None, remote_path
        if result.returncode != 0:
            return None, remote_path
        out = (result.stdout or "").strip()
        if "MISSING" in out:
            return False, remote_path
        if "FOUND" in out:
            return True, remote_path
        return None, remote_path

    def _ips_transform_remote_dir(self, profile) -> str:
        rel_path = os.path.relpath(_ips_transform_local_dir(self._root), self._root)
        return _remote_path_join(profile.remote_project_path, rel_path)

    def _stack_component_names(self, svc: ServiceDef, target: str) -> list[str]:
        """names of the sub-services of a stack service, read from its config."""
        if not _is_stack_service(svc):
            return []
        try:
            cfg, _ = self._load_config_for_target(
                os.path.join(svc.config_dir, "config.yml"),
                show_status=False,
                target=target,
            )
            return [str(s["name"]) for s in _stack_launch_specs_from_config(cfg or {})]
        except Exception:
            return []

    def _stack_running_counts(self, svc: ServiceDef, target: str) -> tuple[int, int] | None:
        """(up, total) listening ports for a stack service, or None if not one."""
        if not _is_stack_service(svc):
            return None
        if target == "local":
            ports = _stack_ports(svc)
            if not ports:
                return None
            return (sum(1 for p in ports if _check_port_in_use(p)), len(ports))
        profile = get_profile_by_name(target)
        if profile is None:
            return None
        ports = self._stack_ports_for_target(svc, target)
        if not ports:
            return None
        return (sum(1 for p in ports if ssh_check_port(profile, p)), len(ports))

    def _stack_service_specs_for_target(self, svc: ServiceDef, target: str) -> list[dict[str, object]]:
        if target == "local":
            return _stack_service_specs(svc.config_dir)
        config_path = os.path.join(svc.config_dir, "config.yml")
        config, _ = self._load_config_for_target(config_path, show_status=False, target=target)
        return _stack_service_specs_from_config(config)

    def _stack_sessions_for_target(self, svc: ServiceDef, target: str) -> list[str]:
        specs = self._stack_service_specs_for_target(svc, target)
        return _stack_sessions_from_specs(svc, specs)

    def _stack_ports_for_target(self, svc: ServiceDef, target: str) -> list[int]:
        specs = self._stack_service_specs_for_target(svc, target)
        return _stack_ports_from_specs(specs)

    async def _run_scp(
        self,
        profile_name: str,
        local_path: str,
        remote_path: str,
        cleanup_local: bool = False,
        cache_key: tuple[str, str] | None = None,
        cache_config: dict | None = None,
    ) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        try:
            remote_dir = remote_path.rsplit("/", 1)[0]
            mkdir_proc = await ssh_run_async(profile, f"mkdir -p {_quote_remote_path(remote_dir)}")
            await mkdir_proc.wait()

            proc = await scp_file_async(profile, local_path, remote_path)
            assert proc.stdout is not None
            output = ""
            async for line in proc.stdout:
                output += line.decode(errors="replace")
            rc = await proc.wait()
            if rc == 0:
                if cache_key is not None and cache_config is not None:
                    self._target_config_cache[cache_key] = cache_config
                    self._refresh_service_cards()
                self._show_status(f"Saved to {profile_name}:{remote_path}")
                await self._maybe_push_master_key(profile, local_path)
            else:
                self._show_status(f"Save failed: {output.strip()}")
        finally:
            if cleanup_local:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

    async def _maybe_push_master_key(self, profile, local_config_path: str) -> None:
        """sync ~/.openmmla/master.key to the remote host when an uploaded
        config contains ENC(...) values, so remote services can decrypt them."""
        try:
            with open(local_config_path, "r", encoding="utf-8") as fh:
                if "ENC(" not in fh.read():
                    return
        except OSError:
            return
        try:
            from openmmla.utils.crypto import MASTER_KEY_PATH
        except ImportError:
            return
        if not os.path.exists(MASTER_KEY_PATH):
            return
        try:
            mkdir_proc = await ssh_run_async(profile, "mkdir -p ~/.openmmla && chmod 700 ~/.openmmla")
            await mkdir_proc.wait()
            scp_proc = await scp_file_async(profile, MASTER_KEY_PATH, ".openmmla/master.key")
            rc = await scp_proc.wait()
            if rc == 0:
                chmod_proc = await ssh_run_async(profile, "chmod 600 ~/.openmmla/master.key")
                await chmod_proc.wait()
                self._show_status(
                    f"Saved to {profile.name} (encrypted values; master key synced to remote ~/.openmmla/)"
                )
        except Exception:
            self._show_status(
                "Config has encrypted values but master key sync failed; "
                "copy ~/.openmmla/master.key to the remote manually."
            )

    async def _run_transform_matrix_sync(
        self,
        profile_name: str,
        local_dir: str,
        remote_dir: str,
        files: list[str],
    ) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return

        mkdir_proc = await ssh_run_async(profile, f"mkdir -p {_quote_remote_path(remote_dir)}")
        await mkdir_proc.wait()

        copied = 0
        failures: list[str] = []
        for name in files:
            local_path = os.path.join(local_dir, name)
            remote_path = _remote_path_join(remote_dir, name)
            proc = await scp_file_async(profile, local_path, remote_path)
            assert proc.stdout is not None
            output = ""
            async for line in proc.stdout:
                output += line.decode(errors="replace")
            rc = await proc.wait()
            if rc == 0:
                copied += 1
            else:
                detail = output.strip() or f"exit code {rc}"
                failures.append(f"{name}: {detail}")

        if failures:
            self._show_status(f"Transform matrix sync failed: {'; '.join(failures[:2])}")
            return
        self._show_status(f"Synced {copied} transform matrix file(s) to {profile_name}:{remote_dir}")

    async def _run_scp_batch(
        self,
        profile_name: str,
        entries: list[tuple[str, str, tuple[str, str], dict]],
        cleanup_local: bool = False,
        success_message: str = "Synced config",
    ) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        saved = 0
        failures: list[str] = []
        try:
            for local_path, remote_path, cache_key, cache_config in entries:
                remote_dir = remote_path.rsplit("/", 1)[0]
                mkdir_proc = await ssh_run_async(profile, f"mkdir -p {_quote_remote_path(remote_dir)}")
                await mkdir_proc.wait()

                proc = await scp_file_async(profile, local_path, remote_path)
                assert proc.stdout is not None
                output = ""
                async for line in proc.stdout:
                    output += line.decode(errors="replace")
                rc = await proc.wait()
                if rc == 0:
                    self._target_config_cache[cache_key] = cache_config
                    saved += 1
                else:
                    failures.append(f"{remote_path}: {output.strip() or f'exit code {rc}'}")

            if failures:
                self._show_status(f"{success_message} partially failed: {'; '.join(failures[:2])}")
            else:
                self._show_status(f"{success_message} to {profile_name} ({saved} file(s))")
            if saved:
                self._refresh_service_cards()
        finally:
            if cleanup_local:
                for local_path, _, _, _ in entries:
                    try:
                        os.unlink(local_path)
                    except OSError:
                        pass

    # ── launch logic ─────────────────────────────────────────────

    @property
    def _cmd(self) -> CommandSession:
        return self.query_one("#svc-cmd-session", CommandSession)

    def _log(self, message: str) -> None:
        try:
            self._cmd.log(message)
        except Exception:
            pass

    def on_stream_panel_stream_log(self, event: StreamPanel.StreamLog) -> None:
        self._log(event.text)

    def _detect_running(self, svc: ServiceDef) -> bool:
        if svc.launch_type == "vllm":
            return _check_port_in_use(_mllm_config(self._root)["port"])
        if svc.launch_type == "tmux":
            if _is_stack_service(svc):
                ports = _stack_ports(svc)
                if ports:
                    return all(_check_port_in_use(port) for port in ports)
                return any(_check_tmux_session(session) for session in _stack_sessions(svc))
            session_name = _service_session_name(svc)
            if _check_tmux_session(session_name):
                return True
            if "ASR" in svc.name:
                return _check_tmux_session("asr-services")
            if svc.name == "VFA Server":
                return _check_tmux_session("vfa-services")
        elif svc.launch_type == "make":
            target = _make_target_for(svc.name)
            port = _SYSTEM_SVC_PORTS.get(target)
            if port is not None:
                return _check_port_in_use(port)
            return _check_tmux_session(target)
        elif svc.launch_type == "collection":
            return False
        return False

    def on_service_card_start_requested(self, event: ServiceCard.StartRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return
        # capture before launching: the launch overwrites the sticky session id
        # with the one it resolves, and the reload afterwards skips the capture
        self._capture_collection_card_state()
        self._capture_infra_mode()
        svc = self._service_for_current_target(svc)

        target = self._get_panel_target()
        is_remote = target != "local"

        if _service_requires_config(svc):
            if not is_remote:
                if not _check_config_exists(svc.config_dir):
                    self._log(f"[yellow]WARNING: config.yml not found in {svc.config_dir}[/yellow]")
                    self._log("[yellow]Please configure this pipeline first.[/yellow]")
                    return
            else:
                exists, remote_path = self._remote_config_exists(svc, target)
                if exists is False:
                    self._log(
                        f"[red]WARNING: config.yml not found on '{target}' at {remote_path}[/red]"
                    )
                    self._log(
                        f"[yellow]Open the Config tab, set your values, and click Save "
                        f"to push config.yml to '{target}' before launching.[/yellow]"
                    )
                    return
                if exists is None:
                    self._log(
                        f"[yellow]NOTE: could not verify config.yml on '{target}' "
                        f"({remote_path}); launching anyway.[/yellow]"
                    )

            # keep shared system-service sections consistent with the central
            # System Services store before launching (single source of truth).
            if not self._reconcile_shared_sections_before_launch(svc, target, is_remote):
                return

        if not is_remote and svc.launch_type != "make" and svc.conda_env:
            has_conda = shutil.which("conda") is not None
            if has_conda and not _check_conda_env(svc.conda_env):
                self._log(f"[red]conda env '{svc.conda_env}' not found. Aborting launch.[/red]")
                self._log(
                    f"[yellow]Create it with: conda create -n {svc.conda_env} "
                    f"python={_service_python_hint(svc)}[/yellow]"
                )
                return
        elif not is_remote and svc.launch_type == "make" and svc.conda_env:
            has_conda = shutil.which("conda") is not None
            need_env = _make_target_for(svc.name) in ("flask", "celery", "nginx")
            if need_env and has_conda and not _check_conda_env(svc.conda_env):
                self._log(f"[red]conda env '{svc.conda_env}' not found. Aborting launch.[/red]")
                self._log(f"[yellow]Create it with: conda create -n {svc.conda_env} python=3.10[/yellow]")
                return

        launch_params = dict(event.params)
        if not self._ensure_pipeline_session_for_launch(svc, launch_params, target=target):
            self._log("[red]Could not resolve a launch session id.[/red]")
            return

        target_label = f"on '{target}'" if is_remote else "locally"
        self._log(f"[green]Starting {svc.name} {target_label}...[/green]")

        if is_remote:
            self._launch_remote(svc, launch_params, target)
        else:
            self._launch_service(svc, launch_params)
        self.set_timer(3.0, self._refresh_visible_statuses)

    def on_service_card_stop_requested(self, event: ServiceCard.StopRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return

        target = self._get_panel_target()
        is_remote = target != "local"
        target_label = f"on '{target}'" if is_remote else "locally"
        self._log(f"[red]Stopping {svc.name} {target_label}...[/red]")

        if is_remote:
            self._stop_remote(svc, target, event.params)
        else:
            self._stop_service(svc, event.params)
        self.set_timer(2.0, self._refresh_visible_statuses)

    def on_service_card_stop_all_requested(self, event: ServiceCard.StopAllRequested) -> None:
        """stop one collection session's audio and video on every host at once."""
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None or svc.launch_type != "collection":
            return

        target = self._get_panel_target()
        params = self._collection_params_for_action(svc, event.params, target)
        session_id = self._collection_session_id(params)
        if not session_id:
            self._log("[yellow]Select a collection session before stopping it on every host.[/yellow]")
            return

        targets = self._collection_stop_targets(session_id)
        self._log(
            f"[red]Stopping collection session '{session_id}' on {len(targets)} host(s): "
            f"{', '.join(targets)}[/red]"
        )
        self.run_worker(
            self._run_collection_stop_all(session_id, targets),
            name=f"collection-stop-all:{session_id}",
            group=_LAUNCHER_COLLECTION_STOP_WORKER_GROUP,
            exclusive=False,
        )
        self.set_timer(2.0, self._refresh_visible_statuses)

    def on_service_card_refresh_requested(self, event: ServiceCard.RefreshRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return
        self._invalidate_session_choice_cache(self._get_panel_target())
        if svc.launch_type == "collection":
            self.run_worker(
                self._reload_current_service_view(),
                group=_LAUNCHER_UI_WORKER_GROUP,
                exclusive=True,
            )
            self._log("Refreshing Collection Session choices.")
            return
        target = self._get_panel_target()
        self.run_worker(
            self._async_refresh_single_status(svc, target),
            group=_LAUNCHER_STATUS_WORKER_GROUP,
            exclusive=False,
        )

    async def _async_refresh_single_status(self, svc: ServiceDef, target: str) -> None:
        """probe one service's running state off the UI thread."""
        counts = None
        if _is_stack_service(svc):
            counts = await asyncio.to_thread(self._stack_running_counts, svc, target)
            is_running = bool(counts) and counts[1] > 0 and counts[0] == counts[1]
        elif target == "local":
            is_running = await asyncio.to_thread(self._detect_running, svc)
        else:
            is_running = await asyncio.to_thread(self._detect_running_remote, svc, target)
        if target != self._get_panel_target():
            return
        self._svc_states[svc.name] = is_running
        for card in self.query(ServiceCard):
            if card.service_def.name == svc.name:
                if counts is not None:
                    card.update_stack_status(counts[0], counts[1])
                else:
                    card.update_status(is_running)
        self._build_tree()
        if counts is not None:
            self._log(f"{svc.name} ({target}): {counts[0]}/{counts[1]} running")
        else:
            status = "[green]Running[/green]" if is_running else "[red]Stopped[/red]"
            self._log(f"{svc.name} ({target}): {status}")

    def on_session_control_panel_refresh_requested(self, event: SessionControlPanel.RefreshRequested) -> None:
        """↻ on the Session Control panel: re-query the active session list."""
        event.stop()
        target = self._get_panel_target()
        self._invalidate_session_choice_cache(target)
        self._log("Refreshing session list...")
        self.run_worker(
            self._async_refresh_session_control_choices(target),
            group=_LAUNCHER_UI_WORKER_GROUP,
            exclusive=False,
        )

    async def _async_refresh_session_control_choices(self, target: str) -> None:
        choices = await asyncio.to_thread(self._artifact_session_choices_for_target, target)
        choices = [c for c in choices if c and c != _NEW_COLLECTION_SESSION_CHOICE]
        if target != self._get_panel_target():
            return
        for panel in self.query(SessionControlPanel):
            panel.update_session_choices(choices)
        self._log(f"Session list updated ({len(choices)} session(s)).")

    def on_service_card_session_refresh_requested(self, event: ServiceCard.SessionRefreshRequested) -> None:
        """↻ next to a session Select: bypass the TTL cache and re-query the
        active session list, then swap the dropdown options in place."""
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return
        target = self._get_panel_target()
        self._invalidate_session_choice_cache(target)
        self._log("Refreshing session list...")
        self.run_worker(
            self._async_refresh_session_choices(svc.name, target),
            group=_LAUNCHER_UI_WORKER_GROUP,
            exclusive=False,
        )

    async def _async_refresh_session_choices(self, service_name: str, target: str) -> None:
        session_ids = await asyncio.to_thread(self._artifact_session_choices_for_target, target)
        if target != self._get_panel_target():
            return
        choices = [_NEW_COLLECTION_SESSION_CHOICE]
        for session_id in session_ids:
            if session_id and session_id not in choices:
                choices.append(session_id)
        for card in self.query(ServiceCard):
            if card.service_def.name == service_name:
                card.update_session_choices(choices)
        self._log(f"Session list updated ({len(choices) - 1} session(s)).")

    def on_service_card_download_requested(self, event: ServiceCard.DownloadRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return
        target = self._get_panel_target()
        if svc.launch_type == "collection":
            if target == "local":
                self._log("[yellow]Download is only needed for remote collection targets.[/yellow]")
                return
            params = self._collection_params_for_action(svc, event.params, target)
            session_id = self._collection_session_id(params)
            if not session_id:
                self._log("[yellow]No valid collection session id. Enter one or start a collection first.[/yellow]")
                return
            self.run_worker(
                self._run_collection_download(target, params),
                name=f"collection-download:{target}:{session_id}",
                group=_LAUNCHER_DOWNLOAD_WORKER_GROUP,
                exclusive=False,
            )
            return

        if not svc.artifact_pipeline:
            return
        if target == "local":
            self._log("[yellow]Artifact download is only available for remote base targets.[/yellow]")
            return
        session_id = self._artifact_session_id(event.params)
        if not session_id:
            self._log("[yellow]Enter a valid Artifact Session before downloading base artifacts.[/yellow]")
            return
        self.run_worker(
            self._run_pipeline_artifacts_download(target, svc, session_id),
            name=f"pipeline-download:{target}:{svc.name}:{session_id}",
            group=_LAUNCHER_DOWNLOAD_WORKER_GROUP,
            exclusive=False,
        )

    def on_service_card_delete_files_requested(self, event: ServiceCard.DeleteFilesRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None or svc.launch_type != "collection":
            return
        target = self._get_panel_target()
        if target == "local":
            self._log("[yellow]Delete Remote is only available for remote collection targets.[/yellow]")
            return
        params = self._collection_params_for_action(svc, event.params, target)
        session_id = self._collection_session_id(params)
        if not session_id:
            self._log("[yellow]No valid collection session id. Enter one or start a collection first.[/yellow]")
            return
        delete_key = (target, session_id)
        if self._pending_collection_delete != delete_key:
            self._pending_collection_delete = delete_key
            self._log(
                f"[yellow]Press Delete Remote again to permanently delete remote collection '{session_id}' on '{target}'.[/yellow]"
            )
            return
        self._pending_collection_delete = None
        self.run_worker(
            self._run_collection_remote_delete(target, params),
            name=f"collection-remote-delete:{target}:{session_id}",
            group=_LAUNCHER_REMOTE_DELETE_WORKER_GROUP,
            exclusive=False,
        )

    async def _run_collection_download(self, profile_name: str, params: dict) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._log(f"[red]SSH profile '{profile_name}' not found.[/red]")
            return
        session_id = self._collection_session_id(params)
        remote_path = self._collection_remote_path(profile, params)
        remote_transfer_path = await asyncio.to_thread(
            lambda: _expand_remote_home_path(remote_path, _remote_home(profile))
        )
        local_path = self._collection_local_path(params, profile_name)
        self._log(f"[cyan]Downloading {profile_name}:{remote_transfer_path} -> {local_path}[/cyan]")
        with tempfile.TemporaryDirectory(prefix="openmmla-collection-") as tmp_dir:
            proc = await scp_from_remote_async(profile, remote_transfer_path, tmp_dir)
            assert proc.stdout is not None
            output = ""
            async for line in proc.stdout:
                output += line.decode(errors="replace")
            rc = await proc.wait()
            if rc != 0:
                self._log(f"[red]Download failed (exit {rc}).[/red]")
                for line in output.strip().splitlines():
                    self._log(rich_escape(line))
                return

            downloaded = Path(tmp_dir) / os.path.basename(remote_transfer_path.rstrip("/"))
            if not downloaded.exists():
                children = [path for path in Path(tmp_dir).iterdir() if path.name != ".DS_Store"]
                downloaded = children[0] if len(children) == 1 else downloaded
            if not downloaded.exists():
                self._log("[red]Download completed, but no collection directory was found in the transfer.[/red]")
                return

            stats = await asyncio.to_thread(
                merge_tree,
                downloaded,
                Path(local_path),
                conflict_label=profile_name,
            )
        manifest = await asyncio.to_thread(
            update_collection_manifest,
            self._root,
            session_id=session_id,
            host_name=safe_segment(params.get("--host-label") or profile_name, "host"),
            remote_path=remote_transfer_path,
            local_path=Path(local_path),
        )
        self._log(
            f"[green]Downloaded collection to {local_path} "
            f"(copied {stats['copied']}, skipped {stats['skipped']}, conflicts {stats['conflicted']}).[/green]"
        )
        self._log(f"[green]Updated session manifest: {manifest}[/green]")

    async def _run_collection_remote_delete(self, profile_name: str, params: dict) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._log(f"[red]SSH profile '{profile_name}' not found.[/red]")
            return
        session_id = self._collection_session_id(params)
        if not session_id or session_id in (".", "..", "/"):
            self._log("[red]Refusing to delete invalid collection session path.[/red]")
            return
        remote_path = self._collection_remote_path(profile, params)
        remote_delete_path = _expand_remote_home_path(remote_path, _remote_home(profile))
        quoted_path = _quote_remote_path(remote_delete_path)
        cmd = (
            f"if [ -d {quoted_path} ]; then "
            f"rm -rf -- {quoted_path} && echo DELETED; "
            "else echo MISSING; fi"
        )
        self._log(f"[red]Deleting remote collection: {profile_name}:{remote_delete_path}[/red]")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        output = ""
        async for line in proc.stdout:
            output += line.decode(errors="replace")
        rc = await proc.wait()
        for line in output.strip().splitlines():
            self._log(rich_escape(line))
        if rc == 0:
            self._log("[green]Remote collection delete command completed.[/green]")
        else:
            self._log(f"[red]Remote collection delete failed (exit {rc}).[/red]")

    async def _mark_session_ended(self, session_id: str, target: str = "local") -> None:
        """record the stop in MongoDB without blocking the event loop."""
        note = await asyncio.to_thread(self._mark_mongodb_session_ended, session_id, target)
        if note:
            self._log(note)

    async def _collection_stop_on_target(self, target: str, session_id: str) -> tuple[int, str]:
        """run the session-scoped stop command on one host.

        The command matches both the audio and the video recorders of that one
        session, so a single run stops everything the host records for it."""
        command = self._collection_stop_command(session_id)
        if target == "local":
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self._root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        else:
            profile = get_profile_by_name(target)
            if profile is None:
                return 127, f"SSH profile '{target}' not found."
            proc = await ssh_run_async(
                profile,
                f"bash -lc {shlex.quote(f'cd $HOME && {command}')}",
            )
        output = ""
        if proc.stdout is not None:
            async for line in proc.stdout:
                output += line.decode(errors="replace")
        rc = await proc.wait()
        return rc, output

    async def _run_collection_remote_stop(self, profile_name: str, session_id: str) -> None:
        if get_profile_by_name(profile_name) is None:
            self._log(f"[red]SSH profile '{profile_name}' not found.[/red]")
            return

        self._log(
            f"[red]Stopping collection session '{session_id}' on '{profile_name}' ...[/red]"
        )
        rc, output = await self._collection_stop_on_target(profile_name, session_id)
        for line in output.strip().splitlines():
            self._log(rich_escape(line))
        if rc == 0:
            self._log(
                f"[green]Stop command completed for collection session '{session_id}' on '{profile_name}'.[/green]"
            )
        else:
            self._log(f"[red]Remote collection stop failed (exit {rc}).[/red]")
        await self._mark_session_ended(session_id, profile_name)
        await self._reload_current_service_view()

    async def _run_collection_local_stop(self, session_id: str) -> None:
        self._log(f"[red]Stopping collection session '{session_id}' locally ...[/red]")
        rc, output = await self._collection_stop_on_target("local", session_id)
        for line in output.strip().splitlines():
            self._log(rich_escape(line))
        if rc == 0:
            self._log(f"[green]Stop command completed for collection session '{session_id}'.[/green]")
        else:
            self._log(f"[yellow]Collection stop command finished with warnings (exit {rc}).[/yellow]")
        await self._mark_session_ended(session_id, "local")
        await self._reload_current_service_view()

    def _collection_stop_targets(self, session_id: str) -> list[str]:
        """every host that could still be recording this session.

        Hosts this TUI launched the session on come first and are always
        included; the rest of the configured hosts are swept too (the stop
        command is session-scoped, so it is a no-op where nothing matches),
        minus the ones a probe has shown to be offline."""
        targets: list[str] = ["local"]
        for name in sorted(self._collection_launch_targets.get(session_id, set())):
            if name not in targets:
                targets.append(name)
        for profile in load_ssh_profiles():
            if profile.name in targets:
                continue
            if self._target_states.get(profile.name) == "offline":
                continue
            targets.append(profile.name)
        return targets

    async def _run_collection_stop_all(self, session_id: str, targets: list[str]) -> None:
        results = await asyncio.gather(
            *(self._collection_stop_on_target(target, session_id) for target in targets),
            return_exceptions=True,
        )
        stopped = 0
        for target, result in zip(targets, results):
            label = "local" if target == "local" else f"'{target}'"
            if isinstance(result, BaseException):
                self._log(f"[red]{label}: stop failed ({result}).[/red]")
                continue
            rc, output = result
            for line in output.strip().splitlines():
                # a bare "[host]" prefix would be swallowed as rich markup
                self._log(f"  [cyan]{rich_escape(target)}[/cyan]  {rich_escape(line)}")
            if rc == 0:
                stopped += 1
                self._log(f"[green]{label}: stop command completed.[/green]")
            else:
                self._log(f"[yellow]{label}: stop command finished with warnings (exit {rc}).[/yellow]")
        if stopped == len(targets):
            self._log(
                f"[green]Collection session '{session_id}' stopped on all {len(targets)} host(s).[/green]"
            )
        else:
            self._log(
                f"[yellow]Collection session '{session_id}': {stopped}/{len(targets)} host(s) "
                f"stopped cleanly; check the lines above.[/yellow]"
            )
        await self._mark_session_ended(session_id, "local")
        await self._reload_current_service_view()

    @staticmethod
    def _artifact_session_id(params: dict) -> str:
        raw = params.get("--artifact-session-id") or params.get("--session-id") or params.get("-sid")
        if _is_new_collection_session_choice(raw):
            return ""
        return _safe_session_id(raw)

    async def _run_pipeline_artifacts_download(
        self,
        profile_name: str,
        svc: ServiceDef,
        session_id: str,
    ) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._log(f"[red]SSH profile '{profile_name}' not found.[/red]")
            return
        pipeline_name = svc.artifact_pipeline or svc.name
        artifact_host = await self._resolve_remote_pipeline_artifact_host(
            profile,
            session_id,
            pipeline_name,
            fallback=profile_name,
        )
        local_root = pipeline_artifact_dir(
            self._root,
            session_id,
            pipeline_name,
            artifact_host,
        )
        downloaded_paths: list[str] = []
        total = {"copied": 0, "skipped": 0, "conflicted": 0}
        remote_root = self._remote_pipeline_artifact_root(profile, session_id, pipeline_name, artifact_host)
        self._log(f"[cyan]Downloading {svc.name} artifacts from {profile_name}:{remote_root}[/cyan]")

        for remote_rel, local_rel in self._pipeline_artifact_sources():
            remote_path = f"{remote_root.rstrip('/')}/{remote_rel}"
            exists = await self._remote_path_exists(profile, remote_path)
            if not exists:
                self._log(f"  [yellow]Missing remote path: {remote_rel}[/yellow]")
                continue
            destination = local_root / local_rel
            stats = await self._download_remote_item(
                profile,
                remote_path,
                destination,
                conflict_label=profile_name,
            )
            if stats is None:
                continue
            downloaded_paths.append(remote_rel)
            for key, value in stats.items():
                total[key] += value

        if not downloaded_paths:
            self._log(f"[yellow]No artifacts found for session '{session_id}' on {profile_name}.[/yellow]")
            return

        manifest = update_pipeline_manifest(
            self._root,
            session_id=session_id,
            pipeline_name=pipeline_name,
            host_name=artifact_host,
            remote_root=remote_root,
            local_path=local_root,
            downloaded_paths=downloaded_paths,
        )
        self._log(
            f"[green]Downloaded {svc.name} artifacts to {local_root} "
            f"(copied {total['copied']}, skipped {total['skipped']}, conflicts {total['conflicted']}).[/green]"
        )
        self._log(f"[green]Updated session manifest: {manifest}[/green]")

    def _remote_pipeline_root(self, profile, svc: ServiceDef) -> str:
        rel_dir = os.path.relpath(svc.config_dir, self._root)
        return f"{profile.remote_project_path.rstrip('/')}/{rel_dir}"

    @staticmethod
    def _remote_pipeline_artifact_base(profile, session_id: str, pipeline_name: str) -> str:
        return (
            f"{profile.remote_project_path.rstrip('/')}/artifacts/"
            f"{safe_segment(session_id, 'session')}/pipelines/"
            f"{safe_segment(pipeline_name, 'pipeline')}"
        )

    @classmethod
    def _remote_pipeline_artifact_root(cls, profile, session_id: str, pipeline_name: str, host_name: str) -> str:
        return f"{cls._remote_pipeline_artifact_base(profile, session_id, pipeline_name)}/{safe_segment(host_name, 'host')}"

    @staticmethod
    def _pipeline_artifact_sources() -> list[tuple[str, Path]]:
        return [
            ("real-time/runtime", Path("real-time") / "runtime"),
            ("real-time/profiles", Path("real-time") / "profiles"),
            ("post-time", Path("post-time")),
            ("logger", Path("logger")),
            ("config", Path("config")),
            ("visualizations", Path("visualizations")),
        ]

    async def _resolve_remote_pipeline_artifact_host(
        self,
        profile,
        session_id: str,
        pipeline_name: str,
        *,
        fallback: str,
    ) -> str:
        base_root = self._remote_pipeline_artifact_base(profile, session_id, pipeline_name)
        candidates: list[str] = []
        for candidate in (fallback, await self._remote_short_hostname(profile)):
            safe = safe_segment(candidate, "")
            if safe and safe not in candidates:
                candidates.append(safe)

        for candidate in candidates:
            if await self._remote_path_exists(profile, f"{base_root.rstrip('/')}/{candidate}"):
                return candidate

        child_dirs = await self._remote_child_dirs(profile, base_root)
        if len(child_dirs) == 1:
            return child_dirs[0]
        if child_dirs:
            self._log(
                f"[yellow]Multiple remote artifact host dirs found under {base_root}; "
                f"using {child_dirs[0]}.[/yellow]"
            )
            return child_dirs[0]
        return safe_segment(fallback, "host")

    async def _remote_short_hostname(self, profile) -> str:
        proc = await ssh_run_async(profile, "hostname -s 2>/dev/null || hostname")
        assert proc.stdout is not None
        output = ""
        async for line in proc.stdout:
            output += line.decode(errors="replace")
        rc = await proc.wait()
        if rc != 0:
            return ""
        return safe_segment(output.strip().splitlines()[0] if output.strip() else "", "")

    async def _remote_child_dirs(self, profile, remote_path: str) -> list[str]:
        quoted_path = _quote_remote_path(remote_path)
        cmd = (
            f"if [ -d {quoted_path} ]; then "
            f"find {quoted_path} -mindepth 1 -maxdepth 1 -type d -exec basename {{}} \\; 2>/dev/null; "
            "fi"
        )
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        output = ""
        async for line in proc.stdout:
            output += line.decode(errors="replace")
        rc = await proc.wait()
        if rc != 0:
            return []
        dirs = []
        for line in output.splitlines():
            name = safe_segment(line.strip(), "")
            if name and name not in _NON_SESSION_ARTIFACT_NAMES and name not in dirs:
                dirs.append(name)
        return sorted(dirs)

    async def _remote_path_exists(self, profile, remote_path: str) -> bool:
        cmd = f"if [ -e {_quote_remote_path(remote_path)} ]; then echo EXISTS; else echo MISSING; fi"
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        output = ""
        async for line in proc.stdout:
            output += line.decode(errors="replace")
        rc = await proc.wait()
        return rc == 0 and "EXISTS" in output

    async def _download_remote_item(
        self,
        profile,
        remote_path: str,
        destination: Path,
        *,
        conflict_label: str,
    ) -> dict[str, int] | None:
        remote_transfer_path = await asyncio.to_thread(
            lambda: _expand_remote_home_path(remote_path, _remote_home(profile))
        )
        with tempfile.TemporaryDirectory(prefix="openmmla-artifact-") as tmp_dir:
            proc = await scp_from_remote_async(profile, remote_transfer_path, tmp_dir)
            assert proc.stdout is not None
            output = ""
            async for line in proc.stdout:
                output += line.decode(errors="replace")
            rc = await proc.wait()
            if rc != 0:
                self._log(f"[red]Download failed for {remote_path} (exit {rc}).[/red]")
                for line in output.strip().splitlines():
                    self._log(rich_escape(line))
                return None

            downloaded = Path(tmp_dir) / os.path.basename(remote_transfer_path.rstrip("/"))
            if not downloaded.exists():
                children = [path for path in Path(tmp_dir).iterdir() if path.name != ".DS_Store"]
                downloaded = children[0] if len(children) == 1 else downloaded
            if not downloaded.exists():
                self._log(f"[red]Download completed, but no artifact was found for {remote_path}.[/red]")
                return None
            return await asyncio.to_thread(
                merge_tree,
                downloaded,
                destination,
                conflict_label=conflict_label,
            )

    def _refresh_visible_statuses(self) -> None:
        self.run_worker(
            self._async_refresh_visible_statuses(self._get_panel_target()),
            group=_LAUNCHER_STATUS_WORKER_GROUP,
            exclusive=True,
        )

    async def _async_refresh_visible_statuses(self, target: str) -> None:
        states = await asyncio.to_thread(self._detect_visible_statuses, target)
        if target != self._get_panel_target():
            return
        for svc_name, (is_running, counts) in states.items():
            self._svc_states[svc_name] = is_running
            for card in self.query(ServiceCard):
                if card.service_def.name == svc_name:
                    if counts is not None:
                        card.update_stack_status(counts[0], counts[1])
                    else:
                        card.update_status(is_running)
        self._build_tree()

    def _detect_visible_statuses(self, target: str) -> dict[str, tuple[bool, tuple[int, int] | None]]:
        states: dict[str, tuple[bool, tuple[int, int] | None]] = {}
        for svc in self._services:
            counts = self._stack_running_counts(svc, target)
            if counts is not None:
                up, total = counts
                states[svc.name] = (total > 0 and up == total, counts)
            elif target == "local":
                states[svc.name] = (self._detect_running(svc), None)
            else:
                states[svc.name] = (self._detect_running_remote(svc, target), None)
        return states

    def _detect_running_remote(self, svc: ServiceDef, profile_name: str) -> bool:
        """check if a service is running on a remote host via SSH."""
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return False
        if svc.launch_type == "make":
            target = _make_target_for(svc.name)
            port = _SYSTEM_SVC_PORTS.get(target)
            if port is not None:
                return ssh_check_port(profile, port)
            return ssh_check_tmux(profile, target)
        elif svc.launch_type == "tmux":
            if _is_stack_service(svc):
                ports = self._stack_ports_for_target(svc, profile_name)
                if ports:
                    return all(ssh_check_port(profile, port) for port in ports)
                return any(
                    ssh_check_tmux(profile, session)
                    for session in self._stack_sessions_for_target(svc, profile_name)
                )
            session_name = _service_session_name(svc)
            return ssh_check_tmux(profile, session_name)
        elif svc.launch_type == "vllm":
            return ssh_check_port(profile, _mllm_config(self._root)["port"])
        elif svc.launch_type == "collection":
            return False
        return False

    def on_service_card_view_logs_requested(self, event: ServiceCard.ViewLogsRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return

        target = self._get_panel_target()
        is_remote = target != "local"

        if is_remote:
            is_running = self._detect_running_remote(svc, target)
        else:
            is_running = self._detect_running(svc)

        if not is_running and not self._logs_available(svc, target):
            self._log(f"[yellow]{svc.name} is not running ({target}). Start the service first.[/yellow]")
            return

        # the Logs message carries no params, so read the card's controls directly
        params = self._card_params(svc.name)
        if is_remote:
            self._view_logs_remote(svc, params)
        else:
            self._view_logs_local(svc, params)

    def _card_params(self, service_name: str) -> dict:
        """current control values of the mounted card for a service, if any."""
        for card in self.query(ServiceCard):
            if card.service_def.name == service_name:
                try:
                    return card.collect_params()
                except Exception:
                    return {}
        return {}

    def _logs_available(self, svc: ServiceDef, target: str) -> bool:
        if _infra_compose_service(svc) is not None and _infra_docker_mode(
            svc, self._card_params(svc.name)
        ):
            # a container that crash-loops never opens its port, and that is
            # exactly when its logs are worth reading
            if target == "local":
                return os.path.isfile(os.path.join(self._root, _INFRA_COMPOSE_FILE))
            return get_profile_by_name(target) is not None
        if svc.launch_type not in ("tmux", "vllm", "collection"):
            return False
        if _is_stack_service(svc):
            # docker compose logs work whether or not containers are running
            rel = _stack_compose_rel_file(svc)
            if target == "local":
                return bool(rel) and os.path.isfile(os.path.join(self._root, rel))
            return get_profile_by_name(target) is not None
        if target == "local":
            if svc.launch_type == "collection":
                return False
            return _check_tmux_session(_service_session_name(svc))
        profile = get_profile_by_name(target)
        if profile is None:
            return False
        if svc.launch_type == "collection":
            return False
        return ssh_check_tmux(profile, _service_session_name(svc))

    def _view_logs_local(self, svc: ServiceDef, params: dict | None = None) -> None:
        if svc.launch_type == "make":
            if _infra_docker_mode(svc, params):
                compose_path = self._infra_compose_path()
                if compose_path is None:
                    return
                cmd = _compose_command(
                    compose_path,
                    f"logs --tail 80 --no-color {_infra_compose_service(svc)}",
                )
                self._log(f"  Running: {cmd}")
                self._cmd.run(cmd)
                return
            target = _make_target_for(svc.name)
            if target in _SYSTEM_SVC_PORTS:
                self._log(f"[cyan]── Logs for {svc.name} ──[/cyan]")
                output = _get_system_service_log(target)
                for line in output.splitlines():
                    self._log(line)
                self._log(f"[cyan]── End of logs ──[/cyan]")
                return
            session_name = target
        elif _is_stack_service(svc):
            rel = _stack_compose_rel_file(svc)
            if rel and os.path.isfile(os.path.join(self._root, rel)):
                cmd = _compose_command(os.path.join(self._root, rel),
                                       "logs --tail 40 --no-color", ["nemo"])
                self._cmd.run(cmd)
            else:
                self._log(f"[yellow]Compose file not found: {rel}[/yellow]")
            return
        elif svc.launch_type == "collection":
            self._log(f"[cyan]── Logs for {svc.name} ──[/cyan]")
            sessions = _collection_sessions_local(svc)
            if not sessions:
                self._log("(no collection sessions found)")
            for session_name in sessions:
                self._log(f"[cyan]── session: {session_name} ──[/cyan]")
                output = _capture_tmux_pane(session_name)
                for line in output.splitlines():
                    self._log(line)
            self._log(f"[cyan]── End of logs ──[/cyan]")
            return
        elif svc.launch_type in ("tmux", "vllm"):
            session_name = _service_session_name(svc)
        else:
            self._log(f"[yellow]No logs available for {svc.name}[/yellow]")
            return

        self._log(f"[cyan]── Logs for {svc.name} (session: {session_name}) ──[/cyan]")
        output = _capture_tmux_pane(session_name)
        for line in output.splitlines():
            self._log(line)
        self._log(f"[cyan]── End of logs ──[/cyan]")

    def _view_logs_remote(self, svc: ServiceDef, params: dict | None = None) -> None:
        if svc.launch_type == "make":
            if _infra_docker_mode(svc, params):
                profile = get_profile_by_name(self._get_panel_target())
                remote_root = profile.remote_project_path if profile else "~/OpenMMLA"
                # the infra compose file lives at the repo root, not in
                # pipelines/uber-server like the Makefile does. no "cd" prefix:
                # CommandSession routes those to its directory-change path, which
                # discards the output we are trying to show
                compose_path = _quote_remote_path(
                    _remote_path_join(remote_root, _INFRA_COMPOSE_FILE)
                )
                service = _infra_compose_service(svc)
                cmd = f"docker compose -f {compose_path} logs --tail 80 --no-color {service}"
                self._log(f"  Running: {cmd}")
                self._cmd.run(cmd)
                return
            target = _make_target_for(svc.name)
            if target in _SYSTEM_SVC_PORTS:
                log_cmds = {
                    "influxdb": "journalctl -u influxdb -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/influxdb/influxd.log 2>/dev/null || echo '(no influxdb logs found)'",
                    "mongodb": "journalctl -u mongod -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/mongodb/mongod.log 2>/dev/null || echo '(no mongodb logs found)'",
                    "redis": "journalctl -u redis-server -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/redis/redis-server.log 2>/dev/null || echo '(no redis logs found)'",
                    "mosquitto": "journalctl -u mosquitto -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/mosquitto/mosquitto.log 2>/dev/null || echo '(no mosquitto logs found)'",
                    "nginx": "journalctl -u nginx -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/nginx/error.log 2>/dev/null || echo '(no nginx logs found)'",
                }
                cmd = log_cmds.get(target, f"echo '(no log command for {target})'")
                self._cmd.run(cmd)
                return
            session_name = target
        elif _is_stack_service(svc):
            profile = get_profile_by_name(self._get_panel_target())
            remote_root = profile.remote_project_path if profile else "~/OpenMMLA"
            rel = _stack_compose_rel_file(svc)
            compose_path = _quote_remote_path(_remote_path_join(remote_root, rel))
            cmd = f"docker compose -f {compose_path} --profile nemo logs --tail 40 --no-color"
            self._cmd.run(cmd)
            return
        elif svc.launch_type == "collection":
            prefix = shlex.quote(_collection_session_prefix(svc))
            cmd = (
                f"for s in $(tmux list-sessions -F '#{{session_name}}' 2>/dev/null | grep '^{prefix}'); do "
                "echo \"── session: $s ──\"; tmux capture-pane -t \"$s\" -p -S -80; "
                "done"
            )
            self._cmd.run(cmd)
            return
        elif svc.launch_type in ("tmux", "vllm"):
            session_name = _service_session_name(svc)
        else:
            self._log(f"[yellow]No logs available for {svc.name}[/yellow]")
            return

        self._cmd.run(f"tmux capture-pane -t {session_name} -p -S -80")

    def _launch_service(self, svc: ServiceDef, params: dict) -> None:
        try:
            if svc.launch_type == "bash":
                self._launch_bash(svc, params)
            elif svc.launch_type == "tmux":
                self._launch_tmux_server(svc, params)
            elif svc.launch_type == "vllm":
                self._launch_vllm_server(svc)
            elif svc.launch_type == "make":
                self._launch_make(svc, params)
            elif svc.launch_type == "collection":
                self._launch_collection(svc, params)
        except Exception as e:
            self._log(f"[red]Error launching {svc.name}: {e}[/red]")

    def _collection_defaults_for_current_target(self, target: str) -> dict[str, object]:
        return self._collection_defaults_for_target(
            target,
            platform_name=self._collection_platform_for_target(target),
        )

    def _collection_platform_for_target(self, target: str) -> str:
        if target == "local":
            return sys.platform

        cache = getattr(self, "_target_platform_cache", None)
        if cache is None:
            cache = {}
            self._target_platform_cache = cache
        if target in cache:
            return cache[target]

        lower_target = str(target or "").lower()
        if lower_target.startswith(("raspi", "rpi", "pi")):
            cache[target] = "linux"
            return cache[target]

        profile = get_profile_by_name(target)
        if profile is None:
            cache[target] = "linux"
            return cache[target]
        try:
            result = ssh_run_sync(profile, "uname -s", timeout=2.0)
            raw = (result.stdout or "").strip().lower()
        except Exception:
            raw = ""
        if "darwin" in raw:
            platform_name = "darwin"
        elif "linux" in raw:
            platform_name = "linux"
        else:
            platform_name = "linux"
        cache[target] = platform_name
        return platform_name

    @staticmethod
    def _collection_defaults_for_target(
        target: str,
        platform_name: str | None = None,
    ) -> dict[str, object]:
        platform_text = (platform_name or (sys.platform if target == "local" else "linux")).lower()
        is_macos = platform_text.startswith("darwin")
        defaults: dict[str, object] = {
            "--output-root": ServicePanel._collection_default_output_root(target),
            "--host-label": ServicePanel._collection_default_host_label(target),
            "--audio-interactive": True,
            "--audio-input-format": (
                DEFAULT_AUDIO_INPUT_FORMAT_MACOS if is_macos else DEFAULT_AUDIO_INPUT_FORMAT_LINUX
            ),
            "--audio-device": DEFAULT_AUDIO_DEVICE_MACOS if is_macos else DEFAULT_AUDIO_DEVICE_LINUX,
            "--audio-channels": "",
            "--audio-channel": DEFAULT_AUDIO_CHANNEL,
            "--sample-rate": str(DEFAULT_AUDIO_SAMPLE_RATE),
            "--audio-format": DEFAULT_AUDIO_FORMAT,
            "--video-interactive": True,
            "--video-input-format": (
                DEFAULT_VIDEO_INPUT_FORMAT_MACOS if is_macos else DEFAULT_VIDEO_INPUT_FORMAT_LINUX
            ),
            "--video-device": DEFAULT_VIDEO_DEVICE_MACOS if is_macos else DEFAULT_VIDEO_DEVICE_LINUX,
            "--video-source-format": (
                DEFAULT_VIDEO_SOURCE_FORMAT_MACOS if is_macos else DEFAULT_VIDEO_SOURCE_FORMAT_LINUX
            ),
            "--framerate": DEFAULT_VIDEO_FRAMERATE,
            "--size": DEFAULT_VIDEO_SIZE,
            "--bitrate": DEFAULT_VIDEO_BITRATE_MACOS if is_macos else DEFAULT_VIDEO_BITRATE_LINUX,
            "--maxrate": DEFAULT_VIDEO_MAXRATE_MACOS if is_macos else DEFAULT_VIDEO_MAXRATE_LINUX,
            "--bufsize": DEFAULT_VIDEO_BUFSIZE_MACOS if is_macos else DEFAULT_VIDEO_BUFSIZE_LINUX,
            "--preset": DEFAULT_VIDEO_PRESET,
            "--camera-label": "",
        }
        return defaults

    @staticmethod
    def _is_default_collection_output_root(output_root: str) -> bool:
        return output_root in {
            "",
            "collection",
            "~/collection",
            "artifacts",
            "~/artifacts",
            "post-time/recordings",
            "~/post-time/recordings",
        }

    def _collection_launch_params(
        self,
        params: dict,
        target: str = "local",
        service_name: str = "Collection Session",
    ) -> dict:
        last = self._collection_last_params.get((target, service_name), {})
        prepared = dict(last)
        for key, value in params.items():
            if key == "--session-id" and not str(value or "").strip() and prepared.get(key):
                continue
            if key == "--session-id" and _is_new_collection_session_choice(value):
                prepared[key] = ""
                continue
            if isinstance(value, (bool, int)):
                prepared[key] = value
            elif str(value or "").strip() or key not in prepared:
                prepared[key] = value
        defaults = self._collection_defaults_for_current_target(target)
        for flag, value in defaults.items():
            if prepared.get(flag) in (None, ""):
                prepared[flag] = value
        for flag in _COLLECTION_HIDDEN_PRESET_FLAGS:
            if flag in defaults:
                prepared[flag] = defaults[flag]
        raw_session_id = str(prepared.get("--session-id") or "").strip()
        if raw_session_id:
            prepared["--session-id"] = _safe_session_id(raw_session_id)
        else:
            prepared["--session-id"] = ""
        output_root = str(prepared.get("--output-root") or "").strip()
        if self._is_default_collection_output_root(output_root):
            prepared["--output-root"] = self._collection_default_output_root(target)
        prepared.pop("--initial-sync-time", None)
        return prepared

    def _collection_params_for_action(self, svc: ServiceDef, params: dict, target: str) -> dict:
        last = self._collection_last_params.get((target, svc.name), {})
        merged = dict(last)
        for key, value in params.items():
            if key == "--session-id" and _is_new_collection_session_choice(value):
                continue
            if isinstance(value, (bool, int)):
                merged[key] = value
            elif str(value or "").strip():
                merged[key] = value
        defaults = self._collection_defaults_for_current_target(target)
        for flag, value in defaults.items():
            if merged.get(flag) in (None, ""):
                merged[flag] = value
        output_root = str(merged.get("--output-root") or "").strip()
        if self._is_default_collection_output_root(output_root):
            merged["--output-root"] = self._collection_default_output_root(target)
        merged["--session-id"] = _safe_session_id(merged.get("--session-id"))
        merged.pop("--initial-sync-time", None)
        return merged

    @staticmethod
    def _collection_default_output_root(target: str) -> str:
        return "~/artifacts" if target != "local" else "artifacts"

    @staticmethod
    def _collection_default_host_label(target: str) -> str:
        if target != "local":
            return safe_segment(target, "host")
        return safe_segment(socket.gethostname().split(".", 1)[0], "host")

    @staticmethod
    def _collection_session_id(params: dict) -> str:
        return _safe_session_id(params.get("--session-id"))

    def _collection_local_path(self, params: dict, host_name: str) -> str:
        session_id = self._collection_session_id(params)
        artifact_host = safe_segment(params.get("--host-label") or host_name, "host")
        return str(collection_artifact_dir(self._root, session_id, artifact_host))

    def _collection_remote_path(self, profile, params: dict) -> str:
        output_root = str(params.get("--output-root") or "artifacts").strip()
        session_id = _safe_session_id(params.get("--session-id"))
        host_name = safe_segment(
            params.get("--host-label") or getattr(profile, "name", None),
            "host",
        )
        if output_root.rstrip("/") in {"artifacts", "~/artifacts"}:
            root = output_root.rstrip("/")
            if root.startswith("/") or root.startswith("~"):
                return f"{root}/{session_id}/collection/{host_name}"
            return f"~/{root}/{session_id}/collection/{host_name}"
        if output_root.startswith("/") or output_root.startswith("~"):
            return f"{output_root.rstrip('/')}/{session_id}"
        return f"~/{output_root.rstrip('/')}/{session_id}"

    @staticmethod
    def _collection_count(params: dict, flag: str) -> int:
        try:
            return max(0, int(params.get(flag, 0)))
        except (TypeError, ValueError):
            return 0

    def _collection_requested_component_count(self, svc: ServiceDef, params: dict) -> int:
        return sum(self._collection_count(params, comp.count_flag) for comp in svc.components)

    @staticmethod
    def _usable_collection_mongodb_config(config: dict) -> dict | None:
        mongo_config = config.get("MongoDB", {})
        if not isinstance(mongo_config, dict):
            return None
        url = str(mongo_config.get("url") or "").strip()
        if url and "<" not in url:
            return dict(mongo_config)
        return None

    def _collection_mongodb_config(self, target: str = "local") -> tuple[dict, str] | tuple[None, None]:
        if target != "local":
            profile = get_profile_by_name(target)
            if profile is not None:
                for rel_path in _ARTIFACT_CONFIG_RELS:
                    config_path = os.path.join(self._root, rel_path)
                    remote_config, _ = self._load_config_for_target(
                        config_path,
                        show_status=False,
                        target=target,
                    )
                    if not remote_config:
                        continue
                    local_access_config = _config_for_local_db_access(remote_config, profile)
                    mongo_config = self._usable_collection_mongodb_config(local_access_config)
                    if mongo_config:
                        return mongo_config, f"{target}:{self._remote_config_path(config_path, profile)}"

        for rel_path in _ARTIFACT_CONFIG_RELS:
            config_path = os.path.join(self._root, rel_path)
            if not os.path.isfile(config_path):
                continue
            config = load_existing_config(config_path)
            mongo_config = self._usable_collection_mongodb_config(config)
            if mongo_config:
                return mongo_config, config_path
        shared_url = str(self._shared_values.get("MongoDB.url") or "").strip()
        if shared_url and "<" not in shared_url:
            return {
                "url": shared_url,
                "db": str(self._shared_values.get("MongoDB.db") or "openmmla").strip(),
            }, f"{_SYSTEM_SERVICES_LABEL} / MongoDB"
        return None, None

    @staticmethod
    def _parse_collection_experiment_group(value: object) -> tuple[str, str]:
        text = str(value or "").strip()
        if "/" in text:
            exp_id, group_id = text.split("/", 1)
        elif ":" in text:
            exp_id, group_id = text.split(":", 1)
        else:
            return "", ""
        return exp_id.strip(), group_id.strip()

    def _create_mongodb_session(
        self,
        experiment_group: object,
        target: str = "local",
        *,
        created_by: str = "tui_launcher",
    ) -> str:
        exp_id, group_id = self._parse_collection_experiment_group(experiment_group)
        if not exp_id or not group_id:
            self._log("[yellow]Select an Experiment Group before creating a MongoDB session.[/yellow]")
            return ""

        mongo_config, config_path = self._collection_mongodb_config(target)
        if not mongo_config:
            self._log("[red]No usable MongoDB config found for collection session creation.[/red]")
            return ""

        try:
            from pymongo import ASCENDING, MongoClient
            from pymongo.errors import DuplicateKeyError
            from openmmla.utils.constants import MONGODB_DEFAULT_DB
            from openmmla.utils.input import _make_session_id
        except ModuleNotFoundError as e:
            self._log(f"[red]Cannot create MongoDB session: {e}[/red]")
            return ""

        session_id = _make_session_id(exp_id, group_id, datetime.now(timezone.utc))
        data = load_experiments(self._root)
        participants = list(get_participant_aliases(exp_id, group_id, data).values())
        db_name = str(mongo_config.get("db") or MONGODB_DEFAULT_DB)
        url = str(mongo_config.get("url") or "").strip()
        try:
            client = MongoClient(
                url,
                serverSelectionTimeoutMS=1500,
                connectTimeoutMS=1500,
            )
            try:
                client.admin.command("ping")
                sessions = client[db_name]["sessions"]
                sessions.create_index([("session_id", ASCENDING)], unique=True)
                sessions.insert_one({
                    "session_id": session_id,
                    "experiment_id": exp_id,
                    "group_id": group_id,
                    "participants": participants,
                    "start_time": datetime.now(timezone.utc),
                    "end_time": None,
                    "status": "active",
                    "metadata": {"created_by": created_by},
                })
            except DuplicateKeyError:
                self._log(f"[yellow]MongoDB session already exists; using {session_id}.[/yellow]")
            finally:
                client.close()
        except Exception as e:
            # MongoDB being down should not block a launch: the session id is
            # generated locally; only the registration record is skipped.
            self._log(f"[yellow]MongoDB unreachable ({e}); using locally generated session id.[/yellow]")
            self._log(f"[yellow]Session {session_id} is NOT registered in MongoDB — start Uber: MongoDB and re-create it if you need it in the session list.[/yellow]")
            return session_id

        self._log(f"[green]MongoDB session ready: {session_id} ({exp_id}/{group_id})[/green]")
        self._invalidate_session_choice_cache()
        return session_id

    def _mark_mongodb_session_ended(self, session_id: str, target: str = "local") -> str:
        """flip one session's MongoDB record to ended and return a log line.

        Stopping the recorders is the real work; this record keeping must never
        break a stop, so a missing config, a missing pymongo, an unreachable
        server and a session that was never registered all come back as a
        warning line instead of an exception. Runs off the event loop (the
        remote config lookup can go over SSH), hence the returned string rather
        than a direct self._log call."""
        if not session_id:
            return ""

        mongo_config, _ = self._collection_mongodb_config(target)
        if not mongo_config:
            return (
                f"[yellow]No usable MongoDB config found; session '{session_id}' "
                f"still shows as active.[/yellow]"
            )

        try:
            from pymongo import MongoClient

            from openmmla.utils.constants import MONGODB_DEFAULT_DB
        except ModuleNotFoundError as e:
            return f"[yellow]Cannot mark session '{session_id}' ended: {e}[/yellow]"

        db_name = str(mongo_config.get("db") or MONGODB_DEFAULT_DB)
        url = str(mongo_config.get("url") or "").strip()
        try:
            client = MongoClient(
                url,
                serverSelectionTimeoutMS=1500,
                connectTimeoutMS=1500,
            )
            try:
                client.admin.command("ping")
                result = client[db_name]["sessions"].update_one(
                    {"session_id": session_id},
                    {"$set": {
                        "end_time": datetime.now(timezone.utc),
                        "status": "ended",
                    }},
                )
            finally:
                client.close()
        except Exception as e:
            return (
                f"[yellow]MongoDB unreachable ({e}); session '{session_id}' "
                f"still shows as active.[/yellow]"
            )

        if result.matched_count:
            return f"[green]Session '{session_id}' marked ended in MongoDB.[/green]"
        # sessions started while MongoDB was down were never registered
        return (
            f"[yellow]Session '{session_id}' is not registered in MongoDB; "
            f"nothing to mark ended.[/yellow]"
        )

    def _create_collection_mongodb_session(self, experiment_group: object, target: str = "local") -> str:
        return self._create_mongodb_session(experiment_group, target=target, created_by="tui_collection")

    def _ensure_collection_session_for_launch(self, params: dict, target: str = "local") -> bool:
        session_id = self._collection_session_id(params)
        if session_id:
            params["--session-id"] = session_id
            return True
        session_id = self._create_collection_mongodb_session(params.get("--experiment-group"), target=target)
        if not session_id:
            return False
        params["--session-id"] = session_id
        return True

    def _ensure_pipeline_session_for_launch(self, svc: ServiceDef, params: dict, target: str = "local") -> bool:
        if not svc.artifact_pipeline:
            return True
        raw_session = params.get("-sid") or params.get("--session-id") or params.get("--artifact-session-id")
        session_id = "" if _is_new_collection_session_choice(raw_session) else _safe_session_id(raw_session)
        if session_id:
            params["-sid"] = session_id
            return True

        session_id = self._create_mongodb_session(
            params.get("--experiment-group"),
            target=target,
            created_by=f"tui_{safe_segment(svc.artifact_pipeline, 'pipeline')}",
        )
        if not session_id:
            return False
        params["-sid"] = session_id
        return True

    def _collection_component_command(
        self,
        comp: ComponentDef,
        params: dict,
        root: str,
    ) -> str:
        script_path = comp.script if os.path.isabs(comp.script) else os.path.join(root, comp.script)
        args = ["bash", script_path]
        for flag in comp.flags:
            value = params.get(flag)
            if value is None:
                continue
            if isinstance(value, bool):
                args.extend([flag, "true" if value else "false"])
                continue
            text = str(value).strip()
            if text == "":
                continue
            args.extend([flag, text])
        return " ".join(shlex.quote(arg) for arg in args)

    def _collection_remote_component_command(self, comp: ComponentDef, params: dict) -> str:
        module = {
            "audio": "openmmla.commands.collect.audio",
            "video": "openmmla.commands.collect.video",
        }.get(comp.role)
        if not module:
            raise ValueError(f"Unsupported collection component: {comp.role}")

        args = ["python3", "-m", module]
        for flag in comp.flags:
            value = params.get(flag)
            if value is None:
                continue
            if isinstance(value, bool):
                args.extend([flag, "true" if value else "false"])
                continue
            text = str(value).strip()
            if text == "":
                continue
            args.extend([flag, text])
        command = " ".join(shlex.quote(arg) for arg in args)
        return f"PYTHONPATH={_REMOTE_COLLECTION_RUNTIME_ENV}:$PYTHONPATH {command}"

    @staticmethod
    def _interactive_ssh_args(profile) -> list[str]:
        args = profile.base_ssh_args()
        try:
            ssh_index = args.index("ssh")
        except ValueError:
            return args
        if "-tt" not in args:
            args.insert(ssh_index + 1, "-tt")
        return args

    def _collection_remote_terminal_command(self, profile, command: str) -> str:
        return self._remote_terminal_command(profile, command)

    def _remote_terminal_command(self, profile, command: str, cwd: str = "~") -> str:
        quoted_cwd = _quote_remote_path(cwd)
        remote_script = (
            f"cd {quoted_cwd} && {command}; "
            "rc=$?; "
            "printf '\\n[OpenMMLA] remote command exited with code %s\\n' \"$rc\"; "
            "exec \"${SHELL:-bash}\" -l"
        )
        args = self._interactive_ssh_args(profile) + ["bash", "-lc", remote_script]
        return " ".join(shlex.quote(arg) for arg in args)

    def _remote_bash_terminal_commands(self, svc: ServiceDef, params: dict, profile) -> list[tuple[str, str]]:
        rel_dir = os.path.relpath(svc.config_dir, self._root)
        remote_root = profile.remote_project_path
        remote_config_dir = _remote_path_join(remote_root, rel_dir)
        remote_config_path = _remote_path_join(remote_config_dir, "config.yml")
        tab_cmds: list[tuple[str, str]] = []

        for comp in svc.components:
            count = params.get(comp.count_flag, 0)
            if isinstance(count, str):
                try:
                    count = int(count)
                except ValueError:
                    count = 0
            if count <= 0:
                continue

            flag_parts = []
            for flag in comp.flags:
                value = params.get(flag)
                if value is None:
                    continue
                if isinstance(value, bool):
                    flag_parts.extend([flag, "true" if value else "false"])
                    continue
                text = str(value).strip()
                if text:
                    flag_parts.extend([flag, text])

            command = (
                f"{comp.script} "
                f"-p {_quote_remote_path(remote_config_dir)} "
                f"-c {_quote_remote_path(remote_config_path)}"
            )
            if flag_parts:
                command += " " + " ".join(shlex.quote(part) for part in flag_parts)
            run_cmd = (
                f"cd {_quote_remote_path(remote_config_dir)} && "
                f"export PYTHONPATH={_quote_remote_path(remote_root)}:$PYTHONPATH && "
                f"{command}"
            )
            wrapped_cmd = wrap_remote(run_cmd, svc.conda_env)

            for index in range(count):
                label = f"{comp.role} {index + 1}" if count > 1 else comp.role
                tab_cmds.append((label, self._remote_terminal_command(profile, wrapped_cmd)))

        return tab_cmds

    @staticmethod
    def _remote_bash_stop_command(svc: ServiceDef) -> str:
        patterns = []
        for comp in svc.components:
            if comp.script:
                patterns.append(comp.script)
        if not patterns:
            patterns = [svc.name]
        quoted_patterns = " ".join(
            shlex.quote(_non_self_matching_regex(pattern))
            for pattern in patterns
        )
        return (
            f"for pattern in {quoted_patterns}; do "
            "echo \"Stopping remote process matching: $pattern\"; "
            "pkill -INT -f \"$pattern\" 2>/dev/null || true; "
            "done; "
            "sleep 2; "
            f"for pattern in {quoted_patterns}; do "
            "pkill -TERM -f \"$pattern\" 2>/dev/null || true; "
            "done; "
            "echo \"Stop signal sent for remote bash service.\""
        )

    @staticmethod
    def _collection_stop_command(session_id: str) -> str:
        quoted_session = shlex.quote(session_id)

        # the session id reaches each pattern through $SESSION_ID rather than
        # being baked into the pattern text, so no complete pattern ever
        # appears in this command's own argv. Otherwise the shell running it
        # matches its own patterns -- "audio_recording.sh.*--session-id <id>"
        # is spelled out inside the "scripts/collection/..." pattern next to
        # it -- and linux pgrep/pkill -f, which scan every /proc/PID/cmdline
        # and only skip themselves, would report the recorders as still alive
        # and signal the stop command itself.
        def pattern_word(prefix: str) -> str:
            return shlex.quote(_non_self_matching_process_pattern(prefix)) + '"$SESSION_ID"'

        recorder_patterns = " ".join(
            pattern_word(prefix)
            for prefix in (
                "openmmla.commands.collect.audio.*--session-id ",
                "openmmla.commands.collect.video.*--session-id ",
                "scripts/collection/audio_recording.sh.*--session-id ",
                "scripts/collection/video_recording.sh.*--session-id ",
                "audio_recording.sh.*--session-id ",
                "video_recording.sh.*--session-id ",
            )
        )
        ffmpeg_pattern = pattern_word("ffmpeg.*")
        return (
            f"SESSION_ID={quoted_session}; "
            "echo \"Stopping OpenMMLA collection session: $SESSION_ID\"; "
            f"for pattern in {recorder_patterns}; do "
            "pkill -INT -f \"$pattern\" 2>/dev/null || true; "
            "done; "
            "grace=20; "
            "while [ \"$grace\" -gt 0 ]; do "
            "alive=0; "
            f"for pattern in {recorder_patterns}; do "
            "if pgrep -f \"$pattern\" >/dev/null 2>&1; then alive=1; fi; "
            "done; "
            "[ \"$alive\" -eq 0 ] && break; "
            "sleep 1; grace=$((grace - 1)); "
            "done; "
            "alive=0; "
            f"for pattern in {recorder_patterns}; do "
            "if pgrep -f \"$pattern\" >/dev/null 2>&1; then alive=1; fi; "
            "done; "
            "if [ \"$alive\" -ne 0 ]; then "
            "echo \"Recorders did not stop after grace period; not force-killing to avoid corrupting files.\"; "
            "exit 1; "
            "fi; "
            f"if pgrep -f {ffmpeg_pattern} >/dev/null 2>&1; then "
            "echo \"Recorder wrapper stopped but ffmpeg is still running; sending Ctrl+C-equivalent SIGINT.\"; "
            f"pkill -INT -f {ffmpeg_pattern} 2>/dev/null || true; "
            "ffmpeg_grace=10; "
            "while [ \"$ffmpeg_grace\" -gt 0 ]; do "
            f"if ! pgrep -f {ffmpeg_pattern} >/dev/null 2>&1; then break; fi; "
            "sleep 1; ffmpeg_grace=$((ffmpeg_grace - 1)); "
            "done; "
            f"if pgrep -f {ffmpeg_pattern} >/dev/null 2>&1; then "
            "echo \"ffmpeg is still running; leaving it alive to avoid corrupting files.\"; "
            "exit 1; "
            "fi; "
            "fi; "
            "echo \"Stop signal sent for collection session: $SESSION_ID\""
        )

    def _ensure_remote_collection_runtime(self, profile) -> bool:
        remote_home = _remote_home(profile)
        remote_runtime = _expand_remote_home_path(_REMOTE_COLLECTION_RUNTIME, remote_home)
        dirs = sorted({os.path.dirname(path) for path in _REMOTE_COLLECTION_FILES})
        mkdir_parts = [
            _remote_path_join(remote_runtime, directory)
            for directory in dirs
        ]
        mkdir_cmd = "mkdir -p " + " ".join(_quote_remote_path(path) for path in mkdir_parts)
        try:
            result = ssh_run_sync(profile, mkdir_cmd, timeout=15.0)
        except Exception as e:
            self._log(f"[red]Failed to prepare remote collection runtime: {e}[/red]")
            return False
        if result.returncode != 0:
            self._log(f"[red]Failed to prepare remote collection runtime: {result.stderr.strip()}[/red]")
            return False

        for rel_path in _REMOTE_COLLECTION_FILES:
            local_path = os.path.join(self._root, rel_path)
            remote_path = _remote_path_join(remote_runtime, rel_path)
            try:
                result = subprocess.run(
                    profile.base_scp_args() + [local_path, f"{profile.ssh_destination()}:{remote_path}"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            except Exception as e:
                self._log(f"[red]Failed to upload collection runtime file {rel_path}: {e}[/red]")
                return False
            if result.returncode != 0:
                self._log(f"[red]Failed to upload collection runtime file {rel_path}: {result.stderr.strip()}[/red]")
                return False
        return True

    def _collection_session_name(self, svc: ServiceDef, role: str, index: int, count: int) -> str:
        suffix = role if count <= 1 else f"{role}-{index + 1}"
        return _collection_session_prefix(svc) + suffix

    def _launch_collection(self, svc: ServiceDef, params: dict) -> None:
        if not svc.components:
            self._log(f"[red]No collection components defined for {svc.name}[/red]")
            return

        prepared = self._collection_launch_params(params, service_name=svc.name)
        if self._collection_requested_component_count(svc, prepared) <= 0:
            self._log("[yellow]No collection components launched.[/yellow]")
            return
        if not self._ensure_collection_session_for_launch(prepared, target="local"):
            self._log("[red]Could not resolve a collection session id.[/red]")
            return
        self._collection_last_params[(self._get_panel_target(), svc.name)] = dict(prepared)
        self._remember_collection_session(prepared.get("--session-id"))
        self._remember_collection_launch("local", prepared.get("--session-id"))
        launched = 0
        self._log(f"  Session ID: {prepared['--session-id']}")
        self._log("  Sync time: auto; manifest will use the earliest common replay time")

        tab_cmds: list[tuple[str, str]] = []
        for comp in svc.components:
            count = self._collection_count(prepared, comp.count_flag)
            for index in range(count):
                label = self._collection_session_name(svc, comp.role, index, count)
                command = self._collection_component_command(comp, prepared, self._root)
                run_cmd = f"cd {shlex.quote(self._root)} && {command}"
                tab_cmds.append((label, run_cmd))
                self._log(rich_escape(f"    [{label}] {command}"))
        if tab_cmds and self._open_collection_terminal(tab_cmds):
            launched = len(tab_cmds)

        if launched:
            output_root = str(prepared.get("--output-root") or "collection")
            self._log(f"[green]Collection recording started; files will be written under {output_root}.[/green]")
            self.run_worker(
                self._reload_current_service_view(capture=False),
                group=_LAUNCHER_UI_WORKER_GROUP,
                exclusive=True,
            )
        else:
            self._log("[yellow]No collection components launched.[/yellow]")

    def _launch_bash(self, svc: ServiceDef, params: dict) -> None:
        if not svc.components:
            self._log(f"[red]No components defined for {svc.name}[/red]")
            return

        python_path = self._root
        conda_env = svc.conda_env
        config_path = os.path.join(svc.config_dir, "config.yml")
        project_arg = f"-p {shlex.quote(svc.config_dir)}"
        config_arg = f"-c {shlex.quote(config_path)}"
        preamble = (
            f"export PYTHONPATH={shlex.quote(python_path)}/:$PYTHONPATH && "
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {shlex.quote(conda_env)}"
        )

        tab_cmds: list[tuple[str, str]] = []
        for comp in svc.components:
            count = params.get(comp.count_flag, 0)
            if isinstance(count, str):
                try:
                    count = int(count)
                except ValueError:
                    count = 0
            if count <= 0:
                continue

            flag_parts = []
            for f in comp.flags:
                val = params.get(f)
                if val is None:
                    continue
                if isinstance(val, bool):
                    flag_parts.append(f"{f} {'true' if val else 'false'}")
                else:
                    text = str(val).strip()
                    if text:
                        flag_parts.append(f"{f} {shlex.quote(text)}")
            flag_str = " ".join(flag_parts)

            if os.path.sep in comp.script or comp.script.endswith(".py"):
                script_path = os.path.join(svc.config_dir, comp.script)
                py_cmd = f"python3 {script_path}"
            else:
                py_cmd = comp.script
            py_cmd += f" {project_arg} {config_arg}"
            if flag_str:
                py_cmd += f" {flag_str}"

            full_cmd = f"{preamble} && {py_cmd}"

            for i in range(count):
                label = f"{comp.role} {i + 1}" if count > 1 else comp.role
                tab_cmds.append((label, full_cmd))

        if not tab_cmds:
            self._log("[yellow]No components to launch (all counts are 0).[/yellow]")
            return

        self._log(f"  Launching {len(tab_cmds)} tab(s)...")
        for label, cmd in tab_cmds:
            self._log(rich_escape(f"    [{label}] {cmd.split(' && ')[-1]}"))

        if sys.platform == "darwin":
            self._open_tabs_mac(tab_cmds)
        elif self._is_ubuntu():
            self._open_tabs_gnome(tab_cmds)
        elif self._is_raspberry_pi():
            self._open_tabs_lxterminal(tab_cmds)
        else:
            self._log("[yellow]Unsupported OS for terminal tab launch.[/yellow]")
            return

        self._log(f"[green]{svc.name} launched in new terminal window.[/green]")

    def _open_collection_terminal(self, tab_cmds: list[tuple[str, str]]) -> bool:
        if sys.platform == "darwin":
            self._open_tabs_mac(tab_cmds)
            return True
        if self._is_ubuntu():
            self._open_tabs_gnome(tab_cmds)
            return True
        if self._is_raspberry_pi():
            self._open_tabs_lxterminal(tab_cmds)
            return True
        self._log("[yellow]Unsupported OS for terminal tab launch.[/yellow]")
        return False

    def _open_tabs_mac(self, tab_cmds: list[tuple[str, str]]) -> None:
        """open one Terminal.app window with N tabs on macOS."""
        script_lines = []
        _, first_cmd = tab_cmds[0]
        escaped = first_cmd.replace("\\", "\\\\").replace('"', '\\"')
        script_lines.append(f'tell application "Terminal" to do script "{escaped}"')
        script_lines.append('tell application "Terminal" to activate')

        for _, cmd in tab_cmds[1:]:
            escaped = cmd.replace("\\", "\\\\").replace('"', '\\"')
            script_lines.append("delay 0.5")
            script_lines.append(
                'tell application "System Events" to keystroke "t" using command down'
            )
            script_lines.append("delay 0.3")
            script_lines.append(
                f'tell application "Terminal" to do script "{escaped}" in the front window'
            )

        args = ["osascript"]
        for line in script_lines:
            args.extend(["-e", line])
        subprocess.Popen(args)

    def _open_tabs_gnome(self, tab_cmds: list[tuple[str, str]]) -> None:
        """open one gnome-terminal window with N tabs."""
        args = ["gnome-terminal", "--window"]
        for _, cmd in tab_cmds:
            args.extend(["--tab", "--", "bash", "-c", f"{cmd}; exec bash"])
        subprocess.Popen(args)

    def _open_tabs_lxterminal(self, tab_cmds: list[tuple[str, str]]) -> None:
        """open one lxterminal window per component (no multi-tab support)."""
        for _, cmd in tab_cmds:
            shell_cmd = "bash -lc " + shlex.quote(f"{cmd}; exec bash")
            subprocess.Popen([
                "lxterminal",
                f"--command={shell_cmd}",
            ])

    @staticmethod
    def _is_ubuntu() -> bool:
        try:
            with open("/etc/os-release") as f:
                return "ID=ubuntu" in f.read()
        except FileNotFoundError:
            return False

    @staticmethod
    def _is_raspberry_pi() -> bool:
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            return "ID=debian" in os_release and "Raspberry Pi" in cpuinfo
        except FileNotFoundError:
            return False

    @staticmethod
    def _filter_specs_by_selection(specs: list, params: dict | None):
        """keep only specs whose component name is in params['__components__'].

        When no selection is provided (non-stack service or older card), all
        specs are kept."""
        selection = (params or {}).get("__components__")
        if selection is None:
            return specs
        selset = set(selection)
        return [s for s in specs if str(s["name"]) in selset]

    def _launch_tmux_server(self, svc: ServiceDef, params: dict | None = None) -> None:
        if _is_stack_service(svc):
            self._launch_docker_stack(svc, params)
            return
        self._log(f"[yellow]No launch method for {svc.name}.[/yellow]")

    def _launch_docker_stack(self, svc: ServiceDef, params: dict | None = None) -> None:
        """start the selected stack sub-services with docker compose (one
        container per service; replaces the old tmux+gunicorn flow)."""
        rel = _stack_compose_rel_file(svc)
        if not rel or not os.path.isfile(os.path.join(self._root, rel)):
            self._log(f"[red]Compose file not found: {rel} (run from a repo with docker/ assets)[/red]")
            return
        config_path = os.path.join(svc.config_dir, "config.yml")
        config = load_existing_config(config_path)
        specs = _stack_launch_specs_from_config(config)
        if not specs:
            self._log(f"[red]No launchable services (sections with port/app) found in {config_path}[/red]")
            return
        specs = self._filter_specs_by_selection(specs, params)
        if not specs:
            self._log("[yellow]No sub-services selected to start.[/yellow]")
            return
        services = _compose_service_names(specs, config)
        profiles = ["nemo"] if "audio-inferer-nemo" in services else []
        # absolute -f path: CommandSession treats "cd ..."-prefixed input as a
        # plain directory change and would swallow the command output
        cmd = _compose_command(os.path.join(self._root, rel),
                               f"up -d --build {' '.join(services)}", profiles)
        self._log(f"  Running: {cmd}")
        self._log("  [yellow]First build downloads several GB of images; progress streams below.[/yellow]")
        self._cmd.run(cmd)

    def _launch_vllm_server(self, svc: ServiceDef) -> None:
        session_name = _service_session_name(svc)
        config = _mllm_config(self._root)
        vllm_cmd = _vllm_serve_command(config)
        run_cmd = f"cd {shlex.quote(self._root)} && {vllm_cmd}; exec bash"
        wrapped_cmd = wrap_local(run_cmd, svc.conda_env)
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            capture_output=True,
        )
        subprocess.Popen(
            ["tmux", "new-session", "-d", "-s", session_name, wrapped_cmd],
            cwd=self._root,
        )
        self._log(f"  Command: {vllm_cmd}")
        self._log(f"[green]{svc.name} tmux session '{session_name}' started on port {config['port']}.[/green]")

    @staticmethod
    def _infra_container_exists(compose_path: str, service: str) -> bool:
        """whether the compose project has a container for this service."""
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", compose_path, "ps", "-a", "-q", service],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # docker missing or wedged: let the stop command itself report it
            return True
        return bool(result.stdout.strip())

    def _infra_compose_path(self) -> str | None:
        """absolute path to the infra compose file, or None (logged) if missing."""
        compose_path = os.path.join(self._root, _INFRA_COMPOSE_FILE)
        if not os.path.isfile(compose_path):
            self._log(f"[red]Compose file not found: {compose_path}[/red]")
            return None
        return compose_path

    def _launch_make(self, svc: ServiceDef, params: dict | None = None) -> None:
        if _infra_docker_mode(svc, params):
            compose_path = self._infra_compose_path()
            if compose_path is None:
                return
            # absolute -f path: CommandSession treats "cd ..."-prefixed input as a
            # plain directory change and would swallow the command output
            cmd = _compose_command(compose_path, f"up -d {_infra_compose_service(svc)}")
            self._log(f"  Running: {cmd}")
            self._cmd.run(cmd)
            return

        target = _make_target_for(svc.name)
        make_dir = svc.config_dir
        makefile = os.path.join(make_dir, "Makefile")
        if not os.path.isfile(makefile):
            self._log(f"[red]Makefile not found: {makefile}[/red]")
            return

        if target in _SYSTEM_SVC_PORTS:
            self._log(f"  Running: make {target}")
            self._log("  [yellow]If it pauses at a Password: prompt, it is auto-filled from System Services → Sudo (or type it in the command box below and press Enter).[/yellow]")
            self._cmd.run(f'make -C {make_dir} {target} SUDO="sudo -S"')
        else:
            self._log(f"  Running: make {target}")
            subprocess.Popen(
                ["make", target],
                cwd=make_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._log(f"[green]Uber {target} started.[/green]")

    @staticmethod
    def _bash_run_flag_str(svc: ServiceDef, params: dict) -> str:
        launch_flags: set[str] = set()
        for comp in svc.components:
            launch_flags.add(comp.count_flag)
            launch_flags.update(comp.flags)

        ordered_flags = [param.flag for param in svc.params if param.flag in launch_flags]
        parts = []
        for flag in ordered_flags:
            value = params.get(flag)
            if value is None:
                continue
            if isinstance(value, bool):
                parts.extend([flag, "true" if value else "false"])
                continue
            text = str(value).strip()
            if text:
                parts.extend([flag, text])
        return "".join(f" {shlex.quote(part)}" for part in parts)

    def _stop_service(self, svc: ServiceDef, params: dict | None = None) -> None:
        try:
            if svc.launch_type == "collection":
                stop_params = self._collection_params_for_action(svc, params or {}, "local")
                session_id = self._collection_session_id(stop_params)
                if not session_id:
                    self._log("[yellow]No collection session has been started from this target yet.[/yellow]")
                    return
                self.run_worker(
                    self._run_collection_local_stop(session_id),
                    name=f"collection-local-stop:{session_id}",
                    group=_LAUNCHER_COLLECTION_STOP_WORKER_GROUP,
                    exclusive=False,
                )
            elif svc.launch_type in ("tmux", "vllm"):
                if _is_stack_service(svc):
                    rel = _stack_compose_rel_file(svc)
                    if rel and os.path.isfile(os.path.join(self._root, rel)):
                        # --profile nemo so profile-gated containers stop too
                        cmd = _compose_command(os.path.join(self._root, rel), "down", ["nemo"])
                        self._log(f"  Running: {cmd}")
                        self._cmd.run(cmd)
                    else:
                        self._log(f"[red]Compose file not found: {rel}[/red]")
                    return
                sessions = [_service_session_name(svc)]
                for session_name in sessions:
                    subprocess.run(["tmux", "send-keys", "-t", session_name, "C-c"], capture_output=True)
                    subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)
                self._log(f"[red]{svc.name} stopped.[/red]")
            elif svc.launch_type == "make":
                if _infra_docker_mode(svc, params):
                    compose_path = self._infra_compose_path()
                    if compose_path is None:
                        return
                    service = _infra_compose_service(svc)
                    if not self._infra_container_exists(compose_path, service):
                        # `compose stop` exits 0 and prints nothing for a service
                        # with no container, which otherwise reads as a success
                        self._log(
                            f"[yellow]No container for '{service}' in this compose project. "
                            f"If {svc.name} is running on its port, it is the native "
                            f"service — switch Run mode to native to stop it.[/yellow]"
                        )
                        return
                    # stop, not down: both infra cards share one compose file, so a
                    # down from the InfluxDB card would tear down MongoDB as well
                    cmd = _compose_command(compose_path, f"stop {service}")
                    self._log(f"  Running: {cmd}")
                    self._cmd.run(cmd)
                    return
                make_name = _make_target_for(svc.name)
                stop_target = "stop-" + make_name
                if make_name in _SYSTEM_SVC_PORTS:
                    self._cmd.run(f'make -C {svc.config_dir} {stop_target} SUDO="sudo -S"')
                else:
                    subprocess.run(
                        ["make", stop_target],
                        cwd=svc.config_dir,
                        capture_output=True,
                        timeout=15,
                    )
                    self._log(f"[red]{svc.name} stopped.[/red]")
                    kill_info = _APP_PORT_CMDS.get(make_name)
                    if kill_info:
                        self._kill_port(kill_info[0], kill_info[1])
            elif svc.launch_type == "bash":
                self._log(f"[yellow]Bash-launched services must be stopped from their terminal windows.[/yellow]")
        except Exception as e:
            self._log(f"[red]Error stopping {svc.name}: {e}[/red]")

    def _kill_port(self, port: int, expected_cmd: str) -> None:
        """force-kill processes on a port whose command name matches expected_cmd."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip()
            if not pids:
                return
            killed = 0
            for pid in pids.splitlines():
                pid = pid.strip()
                if not pid:
                    continue
                ps = subprocess.run(
                    ["ps", "-p", pid, "-o", "comm="],
                    capture_output=True, text=True, timeout=5,
                )
                cmd = ps.stdout.strip().lower()
                if expected_cmd.lower() not in cmd:
                    continue
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                killed += 1
            if killed:
                self._log(f"  Cleaned {killed} process(es) on port {port}")
        except Exception:
            pass

    # ── remote operations ────────────────────────────────────────

    def _launch_remote(self, svc: ServiceDef, params: dict, profile_name: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._log(f"[red]SSH profile '{profile_name}' not found.[/red]")
            return
        remote_root = profile.remote_project_path
        try:
            if svc.launch_type == "bash":
                tab_cmds = self._remote_bash_terminal_commands(svc, params, profile)
                for label, command in tab_cmds:
                    self._log(rich_escape(
                        f"    [{label}] ssh {profile.ssh_destination()} "
                        f"{command.split(' bash -lc ', 1)[-1]}"
                    ))
                if tab_cmds and self._open_collection_terminal(tab_cmds):
                    self._log(f"[green]{svc.name} launched in SSH terminal(s).[/green]")
                else:
                    self._log("[yellow]No remote components launched.[/yellow]")

            elif svc.launch_type == "tmux":
                rel_dir = os.path.relpath(svc.config_dir, self._root)
                remote_config_dir = _remote_path_join(remote_root, rel_dir)
                remote_config_path = _remote_path_join(remote_config_dir, "config.yml")
                config, _ = self._load_config_for_target(
                    os.path.join(svc.config_dir, "config.yml"),
                    show_status=False,
                    target=profile_name,
                )
                specs = _stack_launch_specs_from_config(config)
                if not specs:
                    self._log(f"[red]No launchable services found in remote {remote_config_path}[/red]")
                    return
                specs = self._filter_specs_by_selection(specs, params)
                if not specs:
                    self._log("[yellow]No sub-services selected to start.[/yellow]")
                    return
                rel = _stack_compose_rel_file(svc)
                if not rel:
                    self._log(f"[red]No compose file mapped for {svc.name}.[/red]")
                    return
                services = _compose_service_names(specs, config)
                profiles = ["nemo"] if "audio-inferer-nemo" in services else []
                compose_cmd = _compose_command(rel, f"up -d --build {' '.join(services)}", profiles)
                run_cmd = f"cd {_quote_remote_path(remote_root)} && {compose_cmd}"
                for spec, service in zip(specs, services):
                    self._log(f"  {spec['name']}: container '{service}' on port {spec['port']}")
                self._log(f"  Launching {len(specs)} container(s) on {profile_name} via docker compose...")
                self._log("  [yellow]First build downloads several GB of images.[/yellow]")
                self.run_worker(
                    self._run_remote_streamed(
                        profile_name,
                        run_cmd,
                        f"[green]{svc.name}: containers started on {profile_name} "
                        "(services are loading; click Refresh in a moment).[/green]",
                        f"{svc.name} remote launch failed",
                    )
                )

            elif svc.launch_type == "vllm":
                session_name = _service_session_name(svc)
                config = _mllm_config(self._root)
                vllm_cmd = _vllm_serve_command(config)
                run_cmd = f"cd {_quote_remote_path(remote_root)} && {vllm_cmd}; exec bash"
                wrapped_cmd = wrap_remote(run_cmd, svc.conda_env)
                cmd = (
                    f"tmux kill-session -t {session_name} 2>/dev/null; "
                    f"tmux new-session -d -s {session_name} {shlex.quote(wrapped_cmd)}; "
                    f"tmux attach -t {session_name}"
                )
                ssh_cmd = self._remote_terminal_command(profile, cmd)
                self._log(f"  Remote terminal: ssh {profile.ssh_destination()} {vllm_cmd}")
                if self._open_collection_terminal([(svc.name, ssh_cmd)]):
                    self._log(f"  Command: {vllm_cmd}")
                    self._log(f"[green]{svc.name} tmux session opened remotely on port {config['port']}.[/green]")
                else:
                    self._log("[yellow]Could not open remote MLLM terminal.[/yellow]")

            elif svc.launch_type == "collection":
                prepared = self._collection_launch_params(params, target=profile_name, service_name=svc.name)
                if self._collection_requested_component_count(svc, prepared) <= 0:
                    self._log("[yellow]No remote collection components launched.[/yellow]")
                    return
                if not self._ensure_collection_session_for_launch(prepared, target=profile_name):
                    self._log("[red]Could not resolve a collection session id.[/red]")
                    return
                self._collection_last_params[(profile_name, svc.name)] = dict(prepared)
                self._remember_collection_session(prepared.get("--session-id"))
                self._remember_collection_launch(profile_name, prepared.get("--session-id"))
                self._log(f"  Session ID: {prepared['--session-id']}")
                self._log("  Sync time: auto; manifest will use the earliest common replay time")
                success, msg = ssh_test_connection(profile)
                if not success:
                    self._log(f"[red]SSH connection failed: {msg}[/red]")
                    return
                if not self._ensure_remote_collection_runtime(profile):
                    return
                launched = 0
                tab_cmds: list[tuple[str, str]] = []
                for comp in svc.components:
                    count = self._collection_count(prepared, comp.count_flag)
                    for index in range(count):
                        label = self._collection_session_name(svc, comp.role, index, count)
                        command = self._collection_remote_component_command(comp, prepared)
                        ssh_cmd = self._collection_remote_terminal_command(profile, command)
                        tab_cmds.append((label, ssh_cmd))
                        self._log(rich_escape(f"    [{label}] ssh {profile.ssh_destination()} {command}"))
                if tab_cmds and self._open_collection_terminal(tab_cmds):
                    launched = len(tab_cmds)
                    self._log(f"[green]{svc.name} launched in SSH terminal(s).[/green]")
                    self.run_worker(
                        self._reload_current_service_view(capture=False),
                        group=_LAUNCHER_UI_WORKER_GROUP,
                        exclusive=True,
                    )
                else:
                    self._log("[yellow]No remote collection components launched.[/yellow]")

            elif svc.launch_type == "make":
                if _infra_docker_mode(svc, params):
                    # the infra compose file lives at the repo root, not in
                    # pipelines/uber-server like the Makefile does
                    compose_cmd = _compose_command(
                        _INFRA_COMPOSE_FILE, f"up -d {_infra_compose_service(svc)}"
                    )
                    run_cmd = f"cd {_quote_remote_path(remote_root)} && {compose_cmd}"
                else:
                    target = _make_target_for(svc.name)
                    remote_dir = f"{remote_root}/{os.path.relpath(svc.config_dir, self._root)}"
                    run_cmd = f"cd {_quote_remote_path(remote_dir)} && make {shlex.quote(target)}"
                ssh_cmd = self._remote_terminal_command(profile, run_cmd)
                self._log(f"  Remote terminal: ssh {profile.ssh_destination()} {run_cmd}")
                if self._open_collection_terminal([(svc.name, ssh_cmd)]):
                    self._log(f"[green]{svc.name} start opened in SSH terminal.[/green]")
                else:
                    self._log("[yellow]Could not open remote make terminal.[/yellow]")
        except Exception as e:
            self._log(f"[red]Remote launch error: {e}[/red]")

    def _stop_remote(self, svc: ServiceDef, profile_name: str, params: dict | None = None) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._log(f"[red]SSH profile '{profile_name}' not found.[/red]")
            return
        remote_root = profile.remote_project_path
        try:
            if svc.launch_type == "collection":
                stop_params = self._collection_params_for_action(svc, params or {}, profile_name)
                session_id = self._collection_session_id(stop_params)
                if not session_id:
                    self._log("[yellow]No collection session has been started from this remote target yet.[/yellow]")
                    return
                self.run_worker(
                    self._run_collection_remote_stop(profile_name, session_id),
                    name=f"collection-remote-stop:{profile_name}:{session_id}",
                    group=_LAUNCHER_REMOTE_STOP_WORKER_GROUP,
                    exclusive=False,
                )

            elif svc.launch_type in ("tmux", "vllm"):
                if _is_stack_service(svc):
                    rel = _stack_compose_rel_file(svc)
                    compose_cmd = _compose_command(rel, "down", ["nemo"])
                    cmd = f"cd {_quote_remote_path(remote_root)} && {compose_cmd}"
                else:
                    session = _service_session_name(svc)
                    cmd = (
                        f"tmux send-keys -t {shlex.quote(session)} C-c 2>/dev/null; "
                        f"tmux kill-session -t {shlex.quote(session)} 2>/dev/null"
                    )
                self._log(f"  Stopping {svc.name} on {profile_name}...")
                self.run_worker(
                    self._run_remote_streamed(
                        profile_name, cmd,
                        f"[green]{svc.name} stopped on {profile_name}.[/green]",
                        f"{svc.name} remote stop failed",
                    ),
                    group=_LAUNCHER_REMOTE_STOP_WORKER_GROUP,
                    exclusive=False,
                )

            elif svc.launch_type == "make":
                if _infra_docker_mode(svc, params):
                    # stop, not down: both infra cards share one compose file, so a
                    # down from the InfluxDB card would tear down MongoDB as well.
                    # the file lives at the repo root, not in pipelines/uber-server
                    compose_cmd = _compose_command(
                        _INFRA_COMPOSE_FILE, f"stop {_infra_compose_service(svc)}"
                    )
                    run_cmd = f"cd {_quote_remote_path(remote_root)} && {compose_cmd}"
                    self._log(f"  Running: {run_cmd}")
                else:
                    target = "stop-" + _make_target_for(svc.name)
                    remote_dir = f"{remote_root}/{os.path.relpath(svc.config_dir, self._root)}"
                    run_cmd = f"cd {_quote_remote_path(remote_dir)} && make {shlex.quote(target)}"
                self._log(f"  Stopping {svc.name} on {profile_name}...")
                self.run_worker(
                    self._run_remote_streamed(
                        profile_name, run_cmd,
                        f"[green]{svc.name} stopped on {profile_name}.[/green]",
                        f"{svc.name} remote stop failed",
                    ),
                    group=_LAUNCHER_REMOTE_STOP_WORKER_GROUP,
                    exclusive=False,
                )

            elif svc.launch_type == "bash":
                command = self._remote_bash_stop_command(svc)
                self._log(f"  Stopping {svc.name} on {profile_name}...")
                self.run_worker(
                    self._run_remote_streamed(
                        profile_name, command,
                        f"[green]{svc.name} stopped on {profile_name}.[/green]",
                        f"{svc.name} remote stop failed",
                    ),
                    group=_LAUNCHER_REMOTE_STOP_WORKER_GROUP,
                    exclusive=False,
                )
        except Exception as e:
            self._log(f"[red]Remote stop error: {e}[/red]")
