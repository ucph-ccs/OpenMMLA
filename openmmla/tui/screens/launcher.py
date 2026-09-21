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
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import yaml
from rich.markup import escape as rich_escape
from textual.markup import escape as markup_escape
from rich.text import Text
from textual import on
from textual.message import Message
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
    load_streams, streams_from_config, move_source_settings, PLACEHOLDER_RE,
)
from openmmla.tui.schema.definitions import (
    CONSOLE_ONLY_SECTIONS, PRIVATE_SECTIONS, SHARED_SECTIONS, SHARED_SECTION_NAMES, apply_shared_values,
)
from openmmla.tui.system_services import (
    SYSTEM_SERVICE_SOURCE_CONFIG_RELS,
    SYSTEM_SERVICE_DEFAULT_PORTS,
    SYSTEM_SERVICE_LABELS,
    complete_stream_urls,
    hosts_match,
    is_loopback_host,
    is_stream_path,
    repoint_stream_url,
    stream_server_path,
    stream_server_urls,
    system_service_endpoint,
    system_service_port_conflict,
    system_service_probe_ports,
    system_service_reachable,
    target_for_service_host,
    config_to_flat_values,
    harvest_system_services_from_configs,
    load_system_service_values,
    load_system_services_config,
    get_sudo_password,
    save_system_service_section,
    pipeline_section_overrides,
    shared_section_drift,
)
from openmmla.tui.ssh import (
    REFRESH_TARGETS_OPTION, TARGET_PLATFORMS, TARGET_STATES, WINDOWS_HOST_NOTE, is_select_sentinel, remote_platform, probe_all_profiles, probe_ssh_endpoint, summarize_states, target_options, target_state_label,
    load_ssh_profiles, get_profile_by_name, ssh_run_sync,
    scp_file_async, scp_from_remote_async, ssh_run_async, ssh_check_port, ssh_check_tmux,
    ssh_test_connection,
    wrap_local, wrap_remote,
)
from openmmla.tui.artifacts import (
    collection_artifact_dir, merge_tree, safe_segment, update_collection_manifest,
)
from openmmla.tui import download as dl
from openmmla.tui import recordings, stream_export
from openmmla.utils.artifact_paths import (
    ARTIFACTS_DIR, CAPTURE_STREAMS_DIR, NON_SESSION_ARTIFACT_DIRS, SERVER_RECORD_REL, SERVER_STREAMS_DIR,
    STREAMS_DIR,
)
from openmmla.utils.yaml_dump import dump_yaml_pretty
from openmmla.utils.constants import get_stream_sources, normalize_source
from openmmla.utils.config import get_bases, get_base_by_id, decrypt_config_values
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
from openmmla.tui.widgets.config_form import ConfigForm, DictListField, FieldRow
from openmmla.tui.widgets.recordings_panel import StreamServerRecordingsPanel
from openmmla.tui.widgets.experiment_form import ExperimentForm
from openmmla.tui.widgets.service_card import ServiceCard, ServiceDef, ParamDef, ComponentDef, host_params
from openmmla.tui.widgets.ssh_form import SSHForm
from openmmla.tui.widgets.stream_panel import StreamPanel, _with_stream_path
from openmmla.tui.widgets.session_control import SessionControlPanel
from openmmla.tui.widgets.speaker_profiles import SpeakerProfilesScreen
from openmmla.tui import speakers as asr_speakers
from openmmla.tui import devices as capture_devices
from openmmla.bases.asr.speaker_profiles import join_speakers
from openmmla.tui.widgets.task_form import TaskForm
from openmmla.tui.screens.environment import ENV_GROUPS, env_statuses_local, env_statuses_remote


_SVC_PIPELINE_NAMES: dict[str, str] = {
    "Uber: Nginx": "Nginx",
    "Uber: Flask": "Flask Backend",
}

_STREAM_PIPELINES = {"ASR Base", "IPS Base", "VFA Base"}

# the settings tree: fixed headings (not collapsible) over the forms they sort
_SETTINGS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Hosts", ("SSH Profiles",)),
    ("Study", ("Experiments", "Tasks")),
    # the order of the System Services cards below them
    ("Connections", ("InfluxDB", "MongoDB", "Redis", "MQTT", "Gateway", "StreamServer", "Dashboard")),
    ("Credentials", ("Sudo",)),
)
_SYSTEM_SERVICES_LABEL = "System Settings"

# the infrastructure cards' internal "Uber: <name>" key is jargon: it stays the
# key, and everything the user reads shows SYSTEM_SERVICE_LABELS instead

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
    "openmmla/utils/mac_desktop.py",
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
# its own group: Cancel targets downloads, and must not kill the sweep
_LAUNCHER_STAGING_SWEEP_WORKER_GROUP = "launcher-staging-sweep"
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

def _gateway_base_url(values: dict) -> str:
    """scheme://host:port of the Gateway section in the flat values of a form."""
    scheme = str(values.get("Gateway.scheme") or "").strip() or "http"
    host = str(values.get("Gateway.host") or "").strip() or "localhost"
    return f"{scheme}://{host}:{values.get('Gateway.http_port') or 8080}"


def _is_server_entry(field: LoaderFieldDef) -> bool:
    return field.path.startswith("Server.") and field.field_type in ("str", "url")


def _server_entry_hint(value: object, gateway: str) -> str:
    """where the value of a Server entry leads: through the Gateway or straight
    to a server. It is shown while the value is typed, so a half-written URL must
    not raise, and what was typed must not be read as markup."""
    text = str(value or "").strip()
    if not text or "<" in text:
        return "a bare name goes through the Gateway, a full URL connects directly"
    if "://" in text:
        try:
            server = urlsplit(text).netloc
        except ValueError:
            server = ""
        return f"direct connection to {markup_escape(server or text)}, without the Gateway"
    return f"through the Gateway: {markup_escape(gateway)}/{markup_escape(text.lstrip('/'))}"


def _server_section_note(gateway: str) -> str:
    gateway = markup_escape(gateway)
    return (
        f"One line per AI service, in either of two forms. A bare name (infer) goes through the "
        f"Gateway: {gateway}/infer, which Nginx spreads over the servers listed on its card. A "
        f"full URL (http://gpu-box:5001/infer) is used as it is: a direct connection to that "
        f"server, without Nginx."
    )


# {publish} is the stream server's URL of <app>/<name>, see _make_stream_fields.
# Kept to a line or two: they stand above every stream of the form. The capture
# host (ssh_profile) and its device are not among them: both are picked in the
# Streams tab's table (_STREAMS_TAB_KEYS)
_STREAM_FIELDS_TEMPLATE = [
    ("target", "str", "",
     "where the device publishes. Just the path (ips/cam-1) becomes {publish} on Save; "
     "or a full rtmp/rtsp/srt URL, or udp://<base>:<port> for raw audio to an ASR base"),
    ("read_target", "str", "",
     "where the bases pull it: usually the same path over RTSP, which connects faster than RTMP. Empty = pull the target"),
    ("kind", "str", "",
     "audio or video. Empty = by the device (hw:1,0 or a Mac's :0 is audio) or a udp/tcp target (audio), "
     "else {kind}, as on this card"),
    ("record", "bool", False,
     "also record on the capture host, under <record_root>/streams/capture/<date>/<host>/"),
    ("record_keep_days", "int", 0,
     "days the capture host keeps those recordings: older ones are deleted at the stream's Start and at "
     "Refresh on the Streams tab, whose Manage lists them. 0 = until deleted"),
]

# stream fields shown as a dropdown; empty stays a choice (the blank)
_STREAM_FIELD_CHOICES = {"kind": ["audio", "video"]}
# what the Streams tab picks for a stream, and a Config tab Save keeps
_STREAMS_TAB_KEYS = ("ssh_profile", "device")


def _card_stream_kind(card_name: str) -> str:
    """what a stream of a pipeline card is when neither its kind, its device nor
    its target tells: audio on ASR (a Mac's microphone pushed over RTMP names no
    device), video on IPS and VFA."""
    return "audio" if card_name == "ASR Base" else "video"


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


def _host_label(host: str) -> str:
    """how a host reads on screen; 'local' is this machine."""
    return "Local" if host == "local" else host


async def _process_output(proc) -> str:
    """everything an scp or ssh child wrote, read in chunks: a long line with
    no newline in it would overrun the stream reader's limit."""
    assert proc.stdout is not None
    chunks = []
    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        chunks.append(chunk.decode(errors="replace"))
    return "".join(chunks)


def _sync_destination_options(source: str, ssh_profiles: list[str]) -> list[tuple[str, str]]:
    """the hosts the files shown for `source` can be copied to: every other
    machine, and this one whenever the files on screen are another host's. What
    was edited on a server comes back the way it went out."""
    options = [(name, name) for name in ssh_profiles if name != source]
    if source != "local":
        options.insert(0, (_host_label("local"), "local"))
    return options


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
        # these configs are read raw, so the token is still ENC(...); sent as-is
        # influx answers 401 and the except below turns it into "no sessions"
        influx_config = decrypt_config_values(influx_config)
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
    return _remote_transform_matrix_listing(profile, remote_dir) or []


def _remote_transform_matrix_listing(profile, remote_dir: str) -> list[str] | None:
    """the transformation_matrices*.json files of a remote folder; None when
    the host could not be asked (a missing folder is an empty list)."""
    quoted_dir = _quote_remote_path(remote_dir)
    cmd = (
        f"if [ -d {quoted_dir} ]; then "
        f"find {quoted_dir} -maxdepth 1 -type f -name 'transformation_matrices*.json' -exec basename {{}} \\; "
        "2>/dev/null; fi"
    )
    try:
        result = ssh_run_sync(profile, cmd, timeout=8.0)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return sorted(
        line.strip()
        for line in (result.stdout or "").splitlines()
        if line.strip()
    )


# the base cards, and the pipeline their bases note in a session's sources
_BASE_CARD_PIPELINES = {"ASR Base": "asr", "VFA Base": "vfa", "IPS Base": "ips"}

# the card whose bases recognize speakers from the profiles of its host (Speakers)
_ASR_BASE_CARD = "ASR Base"

# the synchronizer's own count of the bases it merges each time slice from
# (the long form of -nb of mmla asr-sync and vfa-sync): Sync Waits For on the
# card, apart from the card's Num Bases, since bases of the session may run on
# other hosts. Too high, a slice waits for bases that never report and is
# merged only when it expires; too low, it is merged before the rest report.
_SYNC_WAIT_FLAG = "--num_bases"
_SYNC_WAIT_CARDS = ("ASR Base", "VFA Base")

# the Base option that passes no -b: that base asks in its own window which
# Bases entry it is (for a config whose Bases list has no entry for it)
_ASK_BASE_LABEL = "ask in its window"

# a flag the config decides unless the card says otherwise (the VFA synchronizer's
# Action Labels, Pose and Gaze): the first option passes nothing
_CONFIG_OR_ON_OFF = [("as the config says", ""), ("on", "true"), ("off", "false")]

# what the ASR Base card's Language offers (-lang): the language each of its
# bases has its speech transcribed in, sent with every request, which leaves
# the speech transcriber's own configured language for the first option. The
# common ones; -lang itself takes any code (yue) or locale (en-GB) the
# service's backend knows
_ASR_LANGUAGES = [
    ("the server's own language", ""),
    ("English (en)", "en"),
    ("Danish (da)", "da"),
    ("Chinese (zh)", "zh"),
    ("Japanese (ja)", "ja"),
    ("Korean (ko)", "ko"),
    ("German (de)", "de"),
    ("French (fr)", "fr"),
    ("Spanish (es)", "es"),
    ("Italian (it)", "it"),
    ("Portuguese (pt)", "pt"),
    ("Dutch (nl)", "nl"),
    ("Swedish (sv)", "sv"),
    ("Norwegian (nb)", "nb"),
    ("Finnish (fi)", "fi"),
    ("Russian (ru)", "ru"),
    ("Arabic (ar)", "ar"),
]

# transformation_matrices_<id>.json: camera sync's matrices into base <id>'s
# coordinates, the one the IPS synchronizer takes as its main camera
_MATRIX_FILE_PREFIX = "transformation_matrices_"


def _shown_base_value(value) -> str:
    """a Bases value worth showing on a dropdown: not empty, not a <placeholder>
    and not a dropdown left unset."""
    text = str(value if value is not None else "").strip()
    if not text or text == "Select.NULL" or (text.startswith("<") and text.endswith(">")):
        return ""
    return text


def _is_main_base(base: dict) -> bool:
    return str(base.get("main")).strip().lower() in ("true", "1", "yes", "on")


def _base_choice_label(pipeline: str, base: dict) -> str:
    """what a Base dropdown shows for one Bases entry, `0 · macbook-air ·
    stream ips-cam-1`: its id, the camera of an IPS or VFA base (the
    base_type of an ASR one), then its source and source_index."""
    what = _shown_base_value(base.get("base_type") if pipeline == "asr" else base.get("camera"))
    source = _shown_base_value(base.get("source"))
    index = _shown_base_value(base.get("source_index"))
    if index and normalize_source(source) == "file":
        index = os.path.basename(index.rstrip("/")) or index  # the file, not its path
    where = " ".join(part for part in (source, index) if part)
    return " · ".join(part for part in (str(base.get("id")), what, where) if part)


def _base_choices(pipeline: str, config: dict) -> list[tuple[str, str]]:
    """the options of a Base dropdown: every Bases entry of the config in its
    order (value: the id, as -b takes it), then asking in the base's window."""
    options = [(_base_choice_label(pipeline, base), str(base.get("id"))) for base in get_bases(config)]
    return options + [(_ASK_BASE_LABEL, "")]


def _asr_base_types(config: dict) -> list[str]:
    """the device types of an ASR config: the keys of its Base section."""
    section = (config or {}).get("Base")
    if not isinstance(section, dict):
        return []
    return [str(key) for key, value in section.items() if isinstance(value, dict) and _shown_base_value(key)]


def _matrix_file_ids(names: list[str]) -> list[str]:
    """the base ids that have a transformation_matrices_<id>.json: the id is
    all that stands between the prefix and .json (camera sync's raw
    transformation_matrices.json has none and is left out)."""
    ids: list[str] = []
    for name in names:
        if name.startswith(_MATRIX_FILE_PREFIX) and name.endswith(".json"):
            base_id = name[len(_MATRIX_FILE_PREFIX):-len(".json")]
            if base_id and base_id not in ids:
                ids.append(base_id)
    return sorted(ids, key=lambda value: (0, int(value), "") if value.isdigit() else (1, 0, value))


def _main_camera_choices(ids: list[str], config: dict) -> tuple[list[tuple[str, str]], str]:
    """the Main Camera dropdown of the IPS synchronizer: one option per matrix
    file on the card's host, and the default, the Bases entry with main: true
    when it has a file, else the first there is ("" with none)."""
    bases = {str(base.get("id")): base for base in get_bases(config)}
    options = []
    for base_id in ids:
        base = bases.get(base_id)
        if base is None:
            label = f"{base_id} · no Bases entry"
        else:
            camera = _shown_base_value(base.get("camera"))
            label = " · ".join(part for part in (base_id, camera, "main" if _is_main_base(base) else "") if part)
        options.append((label, base_id))
    main = next((base_id for base_id, base in bases.items() if _is_main_base(base)), None)
    default = main if main in ids else (ids[0] if ids else "")
    return options, default


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


def _remote_delete_file(profile, remote_path: str) -> tuple[bool, str]:
    """delete a remote file; one that is not there counts as deleted. Returns (ok, error)."""
    try:
        result = ssh_run_sync(profile, f"rm -f {_quote_remote_path(remote_path)}", timeout=10.0)
    except Exception as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, (result.stderr or "").strip() or f"exit code {result.returncode}"
    return True, ""


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


def _streams_to_carry(local_config: object, remote_config: object) -> dict | None:
    """the Streams entries of a local pipeline config that a remote copy of it
    should have: None when the local config has none, or the remote's already
    match. The rest of the remote config is not looked at."""
    streams = local_config.get("Streams") if isinstance(local_config, dict) else None
    if not isinstance(streams, dict) or not streams:
        return None
    remote_streams = remote_config.get("Streams") if isinstance(remote_config, dict) else None
    return None if remote_streams == streams else copy.deepcopy(streams)


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
        # the file whose Delete has been pressed once
        self._pending_delete: str | None = None

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
                yield Button("Delete", variant="error", id="btn-tm-delete", disabled=True)
            yield Static("", id="tm-status", classes="tm-muted")
        else:
            where = host_label
            yield Static(
                f"No transformation_matrices*.json files found on {where}.",
                classes="tm-muted",
            )

        # the files of the host on screen go to any other machine, this one
        # included: a matrix made on a base station is brought back here the
        # same way this machine's are sent out.
        destinations = _sync_destination_options(self.target, self.ssh_profiles)
        if destinations:
            with Horizontal(classes="tm-actions"):
                yield Select(
                    destinations,
                    prompt="Select destination host...",
                    id="transform-sync-host-select",
                )
                yield Button("Sync to Host", variant="warning", id="btn-sync-transform-host")
        else:
            yield Static("No SSH profiles configured for sync.", classes="tm-muted")

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#tm-status", Static).update(text)
        except Exception:
            pass

    def set_status(self, text: str) -> None:
        """what a sync started from this tab has to say, said on this tab: the
        status line of the Config tab is not on screen while this one is."""
        self._set_status(text)

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
        # a Select rebuilding its options (a file deleted) emits its placeholder,
        # which is Select.BLANK or Select.NULL by textual version
        self._current_file = None if is_select_sentinel(event.value) else str(event.value)
        self._pending_delete = None
        has_file = self._current_file is not None
        for button in ("#btn-tm-save", "#btn-tm-reload", "#btn-tm-delete"):
            self.query_one(button, Button).disabled = not has_file
        self._load_current()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-tm-reload":
            self._load_current()
        elif event.button.id == "btn-tm-delete":
            self._delete_current()
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

    def _delete_current(self) -> None:
        """delete the file on screen, on the host the tab edits, at the second
        press: camera sync writes these, and a base station that loses its own
        has no matrices until they are made or synced again."""
        name = self._current_file
        if not name or not _is_transform_matrix_file(name):
            return
        host_label = self.target if self._is_remote else "Local"
        if self._pending_delete != name:
            self._pending_delete = name
            base_id = name[len(_MATRIX_FILE_PREFIX):-len(".json")] if name.startswith(_MATRIX_FILE_PREFIX) else ""
            who = f"base {base_id} there" if base_id else "the IPS bases there"
            self._set_status(
                f"Press Delete again to delete {name} on {host_label}: {who} then has no matrices until "
                f"camera sync writes them again. The other hosts keep their copy.")
            return
        self._pending_delete = None
        if self._is_remote:
            ok, err = _remote_delete_file(self._ssh_profile, self._file_path(name))
        else:
            try:
                os.remove(self._file_path(name))
                ok, err = True, ""
            except FileNotFoundError:
                ok, err = True, ""  # gone already
            except OSError as exc:
                ok, err = False, str(exc)
        if not ok:
            self._set_status(f"Delete failed on {host_label}: {err}")
            return
        self._forget_file(name)
        left = self._tm_files()
        self._set_status(f"Deleted {name} on {host_label}." if left
                         else f"Deleted {name} on {host_label}: no transform matrix file left there.")

    def _forget_file(self, name: str) -> None:
        """take a deleted file off the list, leaving none picked."""
        files = self._tm_files()
        if name in files:
            files.remove(name)
        try:
            self.query_one("#tm-file-select", Select).set_options([(item, item) for item in files])
        except Exception:
            pass
        self._current_file = None
        try:
            editor = self.query_one("#tm-editor", TextArea)
            editor.load_text("")
            editor.read_only = True
            for button in ("#btn-tm-save", "#btn-tm-reload", "#btn-tm-delete"):
                self.query_one(button, Button).disabled = True
        except Exception:
            pass


# the template's example camera, left in configs saved before Cameras came from
# the config: letters where the calibrator writes numbers
_TEMPLATE_CAMERA_PARAMS = ["fx", "fy", "cx", "cy"]


def _is_template_camera(entry) -> bool:
    return isinstance(entry, dict) and entry.get("params") == _TEMPLATE_CAMERA_PARAMS


def _make_camera_fields(camera: str) -> list[LoaderFieldDef]:
    """the fields of one Cameras entry of an IPS or VFA base config, as the
    calibrator writes them."""
    section = f"Cameras.{camera}"
    specs = (
        ("fisheye", "bool", False, "Set to true if the camera uses a fisheye lens"),
        ("params", "list", [], "intrinsics fx, fy, cx, cy (what the tag detector uses)"),
        ("K", "list", [], "3x3 intrinsic matrix, one row per bracket: [fx, 0, cx], [0, fy, cy], [0, 0, 1]"),
        ("D", "list", [], "distortion coefficients in brackets, e.g. [k1, k2, p1, p2, k3]; K and D undistort a fisheye"),
    )
    return [
        LoaderFieldDef(path=f"{section}.{key}", field_type=ftype, default=default,
                       description=desc, required=False, section=section)
        for key, ftype, default, desc in specs
    ]


def _calibrated_cameras(config: dict) -> dict[str, dict]:
    """the Cameras entries of an IPS base config that hold parameters: numbers
    under `params` (the template's camera_name placeholder holds letters)."""
    cameras = config.get("Cameras") if isinstance(config, dict) else None
    if not isinstance(cameras, dict):
        return {}
    found: dict[str, dict] = {}
    for name, entry in cameras.items():
        params = entry.get("params") if isinstance(entry, dict) else None
        if isinstance(params, list) and params and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in params):
            found[str(name)] = entry
    return found


def _bases_using_camera(config: dict, camera: str) -> list[str]:
    """ids of the Bases entries of an IPS base config that use `camera`."""
    bases = config.get("Bases") if isinstance(config, dict) else None
    if isinstance(bases, dict):
        entries = [dict(entry, id=entry.get("id", key)) for key, entry in bases.items() if isinstance(entry, dict)]
    elif isinstance(bases, list):
        entries = [entry for entry in bases if isinstance(entry, dict)]
    else:
        entries = []
    return [str(entry.get("id")) for entry in entries if str(entry.get("camera")) == camera]


class CameraManagerPanel(Widget):
    """the cameras of the IPS camera calibration on this machine.

    Capture writes a camera's checkerboard images to
    camera_calib/cameras/<name>/ and Calibrate writes its parameters to
    Cameras.<name> of the IPS base config. The two are not one to one (a
    camera captured but not calibrated yet, parameters typed in by hand), so
    the list holds both. A camera's images open in the file browser, where
    they can be looked at; a camera is deleted with its parameters (not while
    a base uses it); and its parameters go to the host that will capture with
    it, without the rest of this machine's config."""

    class SyncRequested(Message):
        """Sync to Host: this camera's parameters into that host's config."""

        def __init__(self, panel: "CameraManagerPanel", camera: str, profile_name: str) -> None:
            super().__init__()
            self.panel = panel
            self.camera = camera
            self.profile_name = profile_name

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
    CameraManagerPanel #cm-sync-profile {
        width: 40;
        margin-right: 1;
    }
    """

    def __init__(
        self, *, cameras_dir: str, target: str, config_path: str = "", ssh_profiles: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.cameras_dir = cameras_dir
        self.target = target
        self.config_path = config_path
        self.ssh_profiles = list(ssh_profiles or [])
        self._current_camera: str | None = None
        # a camera whose Delete Camera has been pressed once
        self._pending_delete: str | None = None

    # ── what is here ─────────────────────────────────────────────
    def _config(self) -> dict:
        return load_existing_config(self.config_path) if self.config_path else {}

    def _folders(self) -> list[str]:
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
        try:
            return sorted(
                name for name in os.listdir(os.path.join(self.cameras_dir, camera))
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
        except OSError:
            return []

    def _cameras(self) -> list[str]:
        return sorted(set(self._folders()) | set(_calibrated_cameras(self._config())))

    def _describe(self, camera: str) -> str:
        config = self._config()
        folder = os.path.join(self.cameras_dir, camera)
        parts = [f"{len(self._images(camera))} image(s)" if os.path.isdir(folder) else "no images here"]
        params = _calibrated_cameras(config).get(camera)
        if params:
            parts.append(f"calibrated (fisheye: {'yes' if params.get('fisheye') else 'no'})")
        else:
            parts.append("not calibrated yet (Calibrate in the calibrator writes its parameters)")
        users = _bases_using_camera(config, camera)
        if users:
            parts.append(f"used by base {', '.join(users)}")
        return f"{camera}: " + " · ".join(parts)

    # ── compose ──────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Static("[b]Calibration Cameras[/b]", classes="cm-title")
        if self.target != "local":
            yield Static(
                "The cameras calibrated on this machine are managed with Host = Local.",
                classes="cm-muted",
            )
            return
        yield Static(
            f"Images: {self.cameras_dir}\nParameters: Cameras in {self.config_path}", classes="cm-muted")
        cameras = self._cameras()
        yield Select(
            [(c, c) for c in cameras],
            prompt="Select a camera..." if cameras else "No camera yet: Start, capture, calibrate",
            id="cm-camera-select",
        )
        yield Static("", id="cm-info", classes="cm-muted")
        with Horizontal(classes="cm-actions"):
            yield Button("Open Folder", id="btn-cm-open", disabled=True)
            yield Button("Delete Camera", variant="error", id="btn-cm-del-camera", disabled=True)
            yield Button("Refresh", id="btn-cm-refresh")
        if self.ssh_profiles:
            with Horizontal(classes="cm-actions"):
                yield Select(
                    [(name, name) for name in self.ssh_profiles],
                    prompt="Host that captures with it...",
                    id="cm-sync-profile",
                )
                yield Button("Sync to Host", variant="warning", id="btn-cm-sync", disabled=True)
        yield Static(
            "After calibrating, Sync to Host gives the host that runs the IPS base this camera's "
            "parameters (only them: the rest of its config stays as it is).",
            classes="cm-muted",
        )
        yield Static("", id="cm-status", classes="cm-muted")

    # ── state ────────────────────────────────────────────────────
    def set_status(self, text: str) -> None:
        try:
            self.query_one("#cm-status", Static).update(text)
        except Exception:
            pass

    def _sync_profile(self) -> str | None:
        try:
            value = self.query_one("#cm-sync-profile", Select).value
        except Exception:
            return None
        return None if value in (None, Select.BLANK) else str(value)

    def _update_actions(self) -> None:
        camera = self._current_camera
        calibrated = bool(camera) and camera in _calibrated_cameras(self._config())
        states = {
            "#btn-cm-open": not (camera and os.path.isdir(os.path.join(self.cameras_dir, camera))),
            "#btn-cm-del-camera": camera is None,
            "#btn-cm-sync": not (calibrated and self._sync_profile()),
        }
        for selector, disabled in states.items():
            try:
                self.query_one(selector, Button).disabled = disabled
            except Exception:
                pass
        try:
            self.query_one("#cm-info", Static).update(self._describe(camera) if camera else "")
        except Exception:
            pass

    def _refresh_cameras(self) -> None:
        """read the folders and the config again, keeping the camera on screen
        when it is still there."""
        cameras = self._cameras()
        keep = self._current_camera if self._current_camera in cameras else None
        try:
            sel = self.query_one("#cm-camera-select", Select)
            sel.set_options([(c, c) for c in cameras])
            sel.value = keep if keep else Select.BLANK
        except Exception:
            pass
        self._current_camera = keep
        self._update_actions()

    # ── events ───────────────────────────────────────────────────
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "cm-camera-select":
            event.stop()
            self._current_camera = None if event.value in (None, Select.BLANK) else str(event.value)
            self._pending_delete = None
            self._update_actions()
        elif event.select.id == "cm-sync-profile":
            event.stop()
            self._update_actions()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "btn-cm-refresh":
            event.stop()
            self._pending_delete = None
            self._refresh_cameras()
            self.set_status("Refreshed.")
        elif bid == "btn-cm-open":
            event.stop()
            self._open_folder()
        elif bid == "btn-cm-del-camera":
            event.stop()
            self._delete_camera()
        elif bid == "btn-cm-sync":
            event.stop()
            profile = self._sync_profile()
            if self._current_camera and profile:
                self.set_status(f"Syncing '{self._current_camera}' to {profile} ...")
                self.post_message(self.SyncRequested(self, self._current_camera, profile))

    def _safe_under_cameras(self, path: str) -> bool:
        root = os.path.abspath(self.cameras_dir)
        target = os.path.abspath(path)
        try:
            return os.path.commonpath([root, target]) == root and target != root
        except ValueError:
            return False

    def _open_folder(self) -> None:
        """the images in the file browser of this machine: the console cannot
        show a picture, and that is where a bad one is found and removed."""
        if not self._current_camera:
            return
        folder = os.path.join(self.cameras_dir, self._current_camera)
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.Popen([opener, folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.set_status(f"Opened {folder}")
        except OSError:
            self.set_status(f"No file browser to open it with; the images are in {folder}")

    def _delete_camera(self) -> None:
        camera = self._current_camera
        if not camera:
            return
        config = self._config()
        users = _bases_using_camera(config, camera)
        if users:
            self._pending_delete = None
            self.set_status(
                f"Not deleted: base {', '.join(users)} uses '{camera}' (Bases of the IPS Base config). "
                f"Give it another camera first.")
            return
        folder = os.path.join(self.cameras_dir, camera)
        has_folder = os.path.isdir(folder) and self._safe_under_cameras(folder)
        cameras = config.get("Cameras")
        has_params = isinstance(cameras, dict) and camera in cameras
        what = " and ".join(part for part in (
            f"its {len(self._images(camera))} image(s)" if has_folder else "",
            "its parameters in config.yml" if has_params else "",
        ) if part)
        if not what:
            self.set_status(f"'{camera}' has nothing left here to delete.")
            self._refresh_cameras()
            return
        if self._pending_delete != camera:
            self._pending_delete = camera
            self.set_status(
                f"Press Delete Camera again to delete '{camera}': {what}. Other hosts keep their copy.")
            return
        self._pending_delete = None
        problems = []
        if has_folder:
            try:
                shutil.rmtree(folder)
            except OSError as exc:
                problems.append(f"images: {exc}")
        if has_params:
            try:
                del cameras[camera]
                dump_yaml_pretty(config, self.config_path)
            except OSError as exc:
                problems.append(f"config.yml: {exc}")
        self.set_status(
            f"Deleted '{camera}': {what}." if not problems
            else f"Deleting '{camera}' went wrong: {'; '.join(problems)}")
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
        # the prompt files of the host on screen go to any other machine, this
        # one included: what was edited on a server comes back here.
        destinations = _sync_destination_options(self.target, self.ssh_profiles)
        if destinations:
            with Horizontal(classes="pp-actions"):
                yield Select(
                    destinations,
                    prompt="Select destination host...",
                    id="prompts-sync-host-select",
                )
                yield Button("Sync to Host", variant="warning", id="btn-sync-prompts-host")

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

    def set_status(self, text: str) -> None:
        """what a sync started from this tab has to say, said on this tab: the
        status line of the Config tab is not on screen while this one is."""
        self._set_status(text)

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
        # the schema of the host on screen goes to any other machine, this one
        # included: what was edited on a server comes back here.
        destinations = _sync_destination_options(self.target, self.ssh_profiles)
        if destinations:
            with Horizontal(classes="as-actions"):
                yield Select(
                    destinations,
                    prompt="Select destination host...",
                    id="aschema-sync-host-select",
                )
                yield Button("Sync to Host", variant="warning", id="btn-sync-aschema-host")

    def on_mount(self) -> None:
        self._load()

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#aschema-status", Static).update(text)
        except Exception:
            pass

    def set_status(self, text: str) -> None:
        """what a sync started from this tab has to say, said on this tab: the
        status line of the Config tab is not on screen while this one is."""
        self._set_status(text)

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


_MEDIAMTX_CONFIG_REL = os.path.join("pipelines", "uber-server", "mediamtx", "mediamtx.yml")


def _mediamtx_server_recording(text: str) -> tuple[int, bool] | None:
    """(line index, on?) of `record:` under `pathDefaults:` in a mediamtx.yml,
    or None when the file has no such line."""
    found = recordings.path_default_line(text, "record")
    if found is None or not re.fullmatch(r"[A-Za-z]+", found[1]):
        return None
    return found[0], found[1].lower() in ("yes", "true", "on")


class StreamServerConfigPanel(Widget):
    """the MediaMTX config of the host the Stream Server card is on, as text,
    with the one switch most people come for on top: recording on the server."""

    DEFAULT_CSS = """
    StreamServerConfigPanel {
        height: auto;
        padding: 1 2;
    }
    StreamServerConfigPanel .ss-title {
        text-style: bold;
        margin-bottom: 1;
    }
    StreamServerConfigPanel .ss-muted {
        color: $text-muted;
    }
    StreamServerConfigPanel #mediamtx-editor {
        height: 24;
        margin-top: 1;
    }
    StreamServerConfigPanel .ss-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    StreamServerConfigPanel .ss-actions Button {
        min-width: 16;
        margin-right: 1;
    }
    StreamServerConfigPanel .ss-actions Label {
        padding: 1 1 0 1;
    }
    StreamServerConfigPanel .ss-actions Select {
        width: 34;
    }
    """

    def __init__(self, *, config_path: str, target: str = "local", ssh_profile=None,
                 remote_path: str | None = None) -> None:
        super().__init__()
        self.config_path = config_path
        self.target = target
        self._ssh_profile = ssh_profile
        self._remote_path = remote_path

    @property
    def _is_remote(self) -> bool:
        return self._ssh_profile is not None and self._remote_path is not None

    def compose(self) -> ComposeResult:
        where = f"{self.target}: {self._remote_path}" if self._is_remote else f"Local: {self.config_path}"
        yield Static("[b]Stream Server config (mediamtx.yml)[/b]", classes="ss-title")
        yield Static(where, classes="ss-muted")
        yield Static(
            "MediaMTX records every stream that is published to it while `record` under "
            "`pathDefaults` is on: ten-minute segments under artifacts/streams/server/<app>/<name>/ of the "
            "project on this host (the same folder for a docker and a native run). A segment is deleted "
            "`recordDeleteAfter` after it began, by MediaMTX itself, so a session's footage has to be "
            "exported before then (Sessions → Export Streams; the Recordings tab shows what is held). "
            "This is the server-side copy; recording on the capture device is the `record` field of a "
            "stream (Streams tab of a base card). The two are independent. The ports here have to match "
            "System Settings → Stream Server.",
            classes="ss-muted",
        )
        with Horizontal(classes="ss-actions"):
            yield Button("Server-side recording: ?", id="btn-mediamtx-record")
            yield Label("Keep recordings for:")
            yield Select([(label, seconds) for label, seconds in recordings.RETENTION_CHOICES],
                         allow_blank=False, id="mediamtx-retention")
        yield TextArea("", id="mediamtx-editor", read_only=True)
        with Horizontal(classes="ss-actions"):
            yield Button("Save", variant="primary", id="btn-mediamtx-save")
            yield Button("Reload", id="btn-mediamtx-reload")
        yield Static("", id="mediamtx-status", classes="ss-muted")

    def on_mount(self) -> None:
        self._load()

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#mediamtx-status", Static).update(text)
        except Exception:
            pass

    def _show_recording_state(self) -> None:
        state = _mediamtx_server_recording(self.query_one("#mediamtx-editor", TextArea).text)
        button = self.query_one("#btn-mediamtx-record", Button)
        if state is None:
            button.label, button.variant, button.disabled = "Server-side recording: not set", "default", True
        else:
            button.label = f"Server-side recording: {'ON' if state[1] else 'OFF'}"
            button.variant, button.disabled = ("success" if state[1] else "default"), False
        self._show_retention_state()

    def _show_retention_state(self) -> None:
        """the choice follows `recordDeleteAfter` in the editor: one of the
        offered spans, or the value as written when it is none of them."""
        select = self.query_one("#mediamtx-retention", Select)
        found = recordings.path_default_line(self.query_one("#mediamtx-editor", TextArea).text, "recordDeleteAfter")
        options = list(recordings.RETENTION_CHOICES)
        with select.prevent(Select.Changed):
            if found is None:
                select.set_options([("not set: MediaMTX keeps a day", -1.0)])
                select.value = -1.0
                select.disabled = True
                return
            seconds = recordings.parse_duration(found[1])
            if seconds is None or seconds not in {value for _label, value in options}:
                options.append((f"as written: {found[1] or 'empty'}", -1.0))
                seconds = -1.0
            select.set_options(options)
            select.value = seconds
            select.disabled = False

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "mediamtx-retention" or event.value is Select.BLANK or float(event.value) < 0:
            return
        # the Select announces its first option as it mounts, and that message
        # arrives after the editor was loaded and the choice set to the file's
        # value: a change that no longer describes the Select is not the user's
        if event.value != event.select.value:
            return
        editor = self.query_one("#mediamtx-editor", TextArea)
        found = recordings.path_default_line(editor.text, "recordDeleteAfter")
        if found is not None and recordings.parse_duration(found[1]) == float(event.value):
            return  # the file already says so
        text = recordings.with_path_default(
            editor.text, "recordDeleteAfter", recordings.format_duration(float(event.value)))
        if text is None:
            return
        editor.load_text(text)
        self._show_retention_state()
        self._set_status("Changed in the editor: press Save to write it.")

    def _load(self) -> None:
        editor = self.query_one("#mediamtx-editor", TextArea)
        if self._is_remote:
            content = _remote_read_file(self._ssh_profile, self._remote_path)
        else:
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
            except OSError:
                content = None
        if content is None:
            editor.load_text("")
            editor.read_only = True
            self._set_status(f"Could not read {self._remote_path if self._is_remote else self.config_path}")
        else:
            editor.load_text(content)
            editor.read_only = False
            self._set_status("Loaded.")
        self._show_recording_state()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        editor = self.query_one("#mediamtx-editor", TextArea)
        if event.button.id == "btn-mediamtx-reload":
            self._load()
        elif event.button.id == "btn-mediamtx-record":
            state = _mediamtx_server_recording(editor.text)
            if state is None:
                return
            editor.load_text(recordings.with_path_default(editor.text, "record", "no" if state[1] else "yes"))
            self._show_recording_state()
            self._set_status("Changed in the editor: press Save to write it.")
        elif event.button.id == "btn-mediamtx-save":
            text = editor.text
            try:
                yaml.safe_load(text)  # reject invalid YAML before writing
            except yaml.YAMLError as exc:
                self._set_status(f"Not saved: invalid YAML: {exc}")
                return
            if self._is_remote:
                ok, err = _remote_write_file(self._ssh_profile, self._remote_path, text)
            else:
                try:
                    with open(self.config_path, "w", encoding="utf-8") as fh:
                        fh.write(text)
                    ok, err = True, ""
                except OSError as exc:
                    ok, err = False, str(exc)
            self._set_status(
                "Saved. MediaMTX reloads the file when it changes; Stop and Start on the Launch tab "
                "if a change does not show." if ok else f"Save failed: {err}"
            )
            self._show_recording_state()


def _make_stream_fields(stream_name: str, stream_server: dict | None = None,
                        default_kind: str = "video") -> list[LoaderFieldDef]:
    """create FieldDef list for a single stream entry. With the Stream Server
    section of System Settings, the help names its real address; default_kind
    is what the card takes a stream with an empty kind for."""
    section = f"Streams.{stream_name}"
    publish, _pull = stream_server_urls(stream_server or {"host": "<stream-server>"}, "<app>/<name>")
    fields = []
    for key, ftype, default, desc in _STREAM_FIELDS_TEMPLATE:
        choices = list(_STREAM_FIELD_CHOICES.get(key, []))
        fields.append(LoaderFieldDef(
            path=f"Streams.{stream_name}.{key}",
            field_type=ftype,
            default=default,
            description=desc.replace("{publish}", publish).replace("{kind}", default_kind),
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
            # how many bases the synchronizer merges each time slice from, on
            # every host: the Bases entries of the card host's config
            # (_service_with_base_choices), else (None) Num Bases
            ParamDef(_SYNC_WAIT_FLAG, "Sync Waits For", "int", None, follows="-nb"),
            ParamDef("-sid", "Session", "str", ""),
            ParamDef("--experiment-group", "Experiment Group", "str", ""),
            # choices, defaults and follow_values come from the card host's
            # config (_service_with_base_choices)
            ParamDef("-b", "Base", "choice", "", per_instance="-nb"),
            # the speaker profiles each base recognizes (-spk): a line per base,
            # which the launcher writes and Manage changes (_put_speakers)
            ParamDef("--speakers", "Speakers", "speakers", None, per_instance="-nb"),
            ParamDef("-bt", "Synchronizer Base Type", "choice", "", follows="-b"),
            ParamDef("-m", "Mode", "str", "live", ["live", "capture", "analyze"]),
            ParamDef("-s", "Store Audio", "bool", True),
            ParamDef("-vad", "VAD", "bool", True),
            ParamDef("-nr", "Noise Reduce", "bool", True),
            ParamDef("-tr", "Transcribe", "bool", True),
            ParamDef("-lang", "Language", "str", "", _ASR_LANGUAGES),
            # anonymous speaker turns with every transcript (pyannote on the speech
            # transcriber, local WhisperX only): who-of-how-many spoke when, no names
            ParamDef("-dia", "Diarize", "bool", False),
            ParamDef("-sp", "Speech Separate", "bool", False),
            ParamDef("-d", "Dominant Speaker", "bool", False),
            ParamDef("-hsr", "Half-Scaled Recognition", "bool", True),
        ],
        components=[
            ComponentDef("base", "mmla asr-base", "-nb",
                         ["-sid", "-b", "--speakers", "-m", "-s", "-vad", "-nr", "-tr", "-lang", "-dia", "-sp", "-hsr"]),
            ComponentDef("synchronizer", "mmla asr-sync", "-ns",
                         ["-sid", _SYNC_WAIT_FLAG, "-bt", "-d", "-sp"]),
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
            ParamDef(_SYNC_WAIT_FLAG, "Sync Waits For", "int", None, follows="-nb"),
            ParamDef("-sid", "Session", "str", ""),
            ParamDef("--experiment-group", "Experiment Group", "str", ""),
            ParamDef("-b", "Base", "choice", "", per_instance="-nb"),
            ParamDef("-m", "Mode", "str", "live", ["live", "capture", "analyze"]),
            ParamDef("-g", "Graphics", "bool", True),
            ParamDef("-s", "Store Frames", "bool", True),
            ParamDef("-v", "Verbose", "bool", True),
            # what the synchronizer asks the frame analyzer for: action labels
            # (the VLM, vfa_action), the pose (skeletons, tags, head yaws;
            # vfa_features, one per frame set) and the gazes with it; the
            # config's Synchronizer.actions / pose / gaze decide unless the card says
            ParamDef("-a", "Action Labels", "str", "", _CONFIG_OR_ON_OFF),
            ParamDef("-pose", "Pose", "str", "", _CONFIG_OR_ON_OFF),
            ParamDef("-gaze", "Gaze", "str", "", _CONFIG_OR_ON_OFF),
        ],
        components=[
            ComponentDef("base", "mmla vfa-base", "-nb",
                         ["-sid", "-b", "-m", "-g", "-s", "-v"]),
            ComponentDef("synchronizer", "mmla vfa-sync", "-ns", ["-sid", _SYNC_WAIT_FLAG, "-a", "-pose", "-gaze"]),
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
            ParamDef("-b", "Base", "choice", "", per_instance="-nb"),
            ParamDef("-mc", "Main Camera", "choice", ""),
            ParamDef("-d", "Visualizer View", "str", "2d", ["2d", "3d"]),
            ParamDef("-g", "Graphics", "bool", True),
            ParamDef("-s", "Store Frames", "bool", True),
            ParamDef("-v", "Verbose", "bool", True),
        ],
        components=[
            ComponentDef("base", "mmla ips-base", "-nb",
                         ["-sid", "-b", "-g", "-s", "-v"]),
            ComponentDef("synchronizer", "mmla ips-sync", "-ns",
                         ["-sid", "-mc", "-v"]),
            ComponentDef("visualizer", "mmla ips-vis", "-nv",
                         ["-sid", "-d", "-s"]),
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
            ParamDef("-na", "Num Audio", "int", 1),
            ParamDef("-nv", "Num Video", "int", 1),
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
        # its services run in docker containers: no conda env to check for or activate
        conda_env="",
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
        # its services run in docker containers: no conda env to check for or activate
        conda_env="",
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
            ("MediaMTX", "Streaming server: RTMP/SRT/RTSP in, RTSP/SRT/RTMP out, recording"),
            ("Flask", "Dashboard (backend API + web frontend)"),
            ("Celery", "Async task worker"),
        ]:
            name = f"Uber: {svc_name}"
            # only the services with a container equivalent in the infra compose
            # file get a run-mode switch; the rest stay Makefile-only
            params = (
                [ParamDef("--mode", "Run mode", "str", _INFRA_DEFAULT_MODE, ["native", "docker"])]
                if _make_target_for(name) in _INFRA_COMPOSE_SERVICES
                else []
            )
            # the docker stack mints the influx admin token on the container host;
            # this pulls it into System Settings instead of a copy-paste over ssh
            extra_actions = (
                [("Fetch Token", _INFRA_FETCH_TOKEN_ACTION)]
                if _make_target_for(name) == "influxdb"
                else []
            )
            services.append(ServiceDef(
                name=name,
                category="System Services",
                # nginx renders its config with the env, gunicorn and celery run
                # in it; a database, a broker or MediaMTX is a systemd, brew or
                # docker process and needs none
                conda_env="" if _make_target_for(name) in _INFRA_NO_ENV_TARGETS else "uber-server",
                config_dir=uber_dir,
                launch_type="make",
                description=desc,
                params=params,
                extra_actions=extra_actions,
                label=SYSTEM_SERVICE_LABELS.get(_make_target_for(name), svc_name),
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
    if service_name == "mediamtx":
        # a native MediaMTX runs inside the tmux session `make mediamtx` opened
        return _capture_tmux_pane("mediamtx")
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

# cards whose services the Gateway routes: starting one renders the Gateway's
# config again and reloads it, so a machine it could not reach when it last
# rendered gets a route without anyone pressing Start on the Gateway card. Not
# the MLLM Server: the frame analyzer calls that one straight, not through the
# Gateway
_GATEWAY_ROUTED_CARDS = frozenset({"ASR Server", "VFA Server"})

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

# what the Run mode select starts on, and what a missing value means
_INFRA_DEFAULT_MODE = "docker"

_INFRA_FETCH_TOKEN_ACTION = "fetch-token"

# prints the influx admin token of the docker stack on the host it runs on: the
# running container's CLI config first (covers a token influx generated itself),
# else docker/.env. runs from the repo root. prints nothing when neither has one.
_INFRA_READ_TOKEN_SH = (
    r"t=$(docker compose -f docker/docker-compose.infra.yml exec -T influxdb "
    r"cat /etc/influxdb2/influx-configs 2>/dev/null"
    r''' | sed -n 's/^[[:space:]]*token[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1); '''
    r'''[ -n "$t" ] || t=$(sed -n 's/^INFLUXDB_INIT_ADMIN_TOKEN=//p' docker/.env 2>/dev/null | head -n1); '''
    r'''printf '%s' "$t"'''
)


def _mask_secret(value: str) -> str:
    return f"{value[:4]}…{value[-4:]}" if len(value) >= 12 else "…"


# _make_target_for() name -> docker compose service name
_INFRA_COMPOSE_SERVICES = {
    "influxdb": "influxdb",
    "mongodb": "mongodb",
    "mediamtx": "mediamtx",
}

# System Settings field that names the machine a system service runs on, keyed
# by make target. The worker has no address of its own: it lives next to Flask
_SERVICE_HOST_FIELDS: dict[str, str] = {
    "influxdb": "InfluxDB.url",
    "mongodb": "MongoDB.url",
    "redis": "Redis.host",
    "mosquitto": "MQTT.host",
    "nginx": "Gateway.host",
    "mediamtx": "StreamServer.host",
    "flask": "Dashboard.host",
    "celery": "Dashboard.host",
}

_SETTINGS_HOST_NOTE = "Local  (System Settings live in this project)"
# settings that only mean something on the machine the console runs on; the
# Connections forms have a Host selector instead (every machine's services read
# that machine's own settings)
_LOCAL_SETTINGS_NOTES: dict[str, str] = {
    "__ssh_profiles__": "Local  (the machines this console can reach)",
    "__experiments__": "Local  (a session takes its participants to every host through MongoDB)",
    "__tasks__": "Local  (task definitions are read by this console only)",
    "__shared__Sudo": "Local  (this machine's admin password; a remote host uses its SSH profile's)",
    "__shared__StreamServer": "Local  (System Settings live in this project; Sync to Host gives another machine "
                              "this address and the stream URLs it completed)",
}
_SESSION_CONTROL_HOST_NOTE = "Not host-specific  (START and STOP travel over Redis)"

# the host each launcher node was last pointed at, kept across restarts
_NODE_HOSTS_REL_PATH = os.path.join("config", "launcher_hosts.yml")


@dataclass(frozen=True)
class NodeHost:
    """where one launcher node opens.

    Every node has its own host. A system service whose System Settings address
    names a machine opens on that machine, so by default its card, its sidebar
    marker and its Start/Stop mean the same place; the user may still move the
    card for a one-off on another host, and it is back on the configured
    machine the next time it is opened. Anything else opens where the user
    last pointed it, which is remembered per node."""
    target: str = "local"
    # a system service whose address names a machine: the settings field
    # ("InfluxDB.url"), the machine as written there, and the console target
    # that reaches it ("" when it is neither this machine nor a usable profile)
    follows: str = ""
    machine: str = ""
    machine_target: str = ""
    # why `target` is not the host one would expect
    fallback_from: str = ""


def _encrypt_secrets(config: dict) -> None:
    """ENC(...) the token/password values of a config that is about to be
    written to another machine (what the user just typed is still plaintext)."""
    try:
        from openmmla.utils.crypto import encrypt_sensitive_values, ensure_master_key
        encrypt_sensitive_values(config, ensure_master_key())
    except Exception:
        pass  # crypto unavailable: written as it is, like the local store


def _node_hosts_path(root: str) -> str:
    return os.path.join(root, _NODE_HOSTS_REL_PATH)


def load_node_hosts(root: str) -> dict[str, str]:
    """node name -> SSH profile name; nodes on Local are simply absent."""
    try:
        with open(_node_hosts_path(root), encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(name): str(target) for name, target in data.items()
        if str(target or "").strip() and str(target) != "local"
    }


def save_node_hosts(root: str, hosts: dict[str, str]) -> None:
    path = _node_hosts_path(root)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(dict(sorted(hosts.items())), file, default_flow_style=False, allow_unicode=True)
    except OSError:
        pass  # a read-only checkout only loses the memory, never the action


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
    bare-metal Makefile; a missing value means _INFRA_DEFAULT_MODE."""
    if _infra_compose_service(svc) is None:
        return False
    mode = str((params or {}).get("--mode") or _INFRA_DEFAULT_MODE).strip().lower()
    return mode == "docker"


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

# conventional ports of the Uber system services, keyed by make target; the
# reachability probe itself lives in openmmla.tui.system_services
_SYSTEM_SVC_PORTS: dict[str, int] = SYSTEM_SERVICE_DEFAULT_PORTS
_NON_SESSION_ARTIFACT_NAMES = {*NON_SESSION_ARTIFACT_DIRS, ".DS_Store"}

_MAKE_TARGET_OVERRIDES: dict[str, str] = {}

# (port, expected_command) for app services that need port cleanup on stop
_APP_PORT_CMDS: dict[str, tuple[int, str]] = {
    "flask": (5050, "gunicorn"),
}


_ENV_NAMES = {entry["env"] for entry in ENV_GROUPS}
# databases and brokers are brew/systemd/docker processes; their card's
# conda_env is only the Makefile's home, not something they run in
_INFRA_NO_ENV_TARGETS = {"influxdb", "mongodb", "redis", "mosquitto", "mediamtx"}


def _shared_section_label(section: str) -> str:
    info = SHARED_SECTIONS.get(section) or {}
    return str(info.get("label") or section)


def _service_uses_conda_env(svc: ServiceDef) -> bool:
    """whether the service actually runs Python in a conda env: pipelines and
    the dashboard/nginx helpers do; databases, brokers and docker stacks do not."""
    if svc.conda_env not in _ENV_NAMES:
        return False
    return not (svc.launch_type == "make" and _make_target_for(svc.name) in _INFRA_NO_ENV_TARGETS)


# how old a host's env states may be when the Launcher comes back into view
_ENV_MARKER_MAX_AGE = 60.0


def _env_marker_color(status: str | None) -> str | None:
    if not status or status == "Unknown":
        return None
    if status == "Ready":
        return "green"
    return "yellow" if status.startswith("Partial") else "red"


def _make_extra_vars(root: str, target: str) -> list[str]:
    """make variables a target needs from System Settings."""
    if target == "flask":
        _, port = system_service_endpoint(root, "flask") or ("", 5050)
        return [f"DASHBOARD_PORT={port}"]
    return []


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


# what a live recorder looks like in the process table. Recorders run in a
# terminal window of their own (a local Terminal tab, an ssh session on a
# remote host), never in tmux, so their processes are the only trace of them:
# the names below are the ones the stop command signals as well
_COLLECTION_RECORDER_MARKS = (
    "openmmla.commands.collect.audio",
    "openmmla.commands.collect.video",
    "collection/audio_recording.sh",
    "collection/video_recording.sh",
)
# every process with its full command line, on macOS and Linux alike
_PS_ALL_COMMANDS = "ps -A -ww -o pid=,args="


def _collection_recorders(ps_output: str) -> list[tuple[str, str, str]]:
    """(role, session id, pid) of the recorders alive in a `ps -o pid=,args=`
    listing, one entry per recorder (the wrapper script and the python module
    it runs are the same recorder)."""
    found: dict[tuple[str, str], str] = {}
    for line in ps_output.splitlines():
        if "--session-id" not in line:
            continue  # an editor open on the script is not a recorder
        if "pkill" in line or "pgrep" in line or "SESSION_ID=" in line:
            continue  # the stop command spells the same names out
        mark = next((m for m in _COLLECTION_RECORDER_MARKS if m in line), None)
        if mark is None:
            continue
        pid, _, args = line.strip().partition(" ")
        match = re.search(r"--session-id[ =]+(\S+)", args)
        key = ("audio" if "audio" in mark else "video", match.group(1) if match else "?")
        # prefer the python process: it is the one that holds ffmpeg
        if key not in found or mark.startswith("openmmla."):
            found[key] = pid
    return [(role, session, pid) for (role, session), pid in found.items()]


def _pid_order(pid: object) -> int:
    """recorders sorted by pid: the higher one was started later (close enough
    to tell the current take from one that was left running)."""
    try:
        return int(str(pid).strip())
    except ValueError:
        return -1


def _collection_recorders_local() -> list[tuple[str, str, str]]:
    try:
        result = subprocess.run(
            shlex.split(_PS_ALL_COMMANDS), capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return _collection_recorders(result.stdout) if result.returncode == 0 else []


def _collection_recorders_remote(profile) -> list[tuple[str, str, str]]:
    try:
        result = ssh_run_sync(profile, _PS_ALL_COMMANDS, timeout=8.0)
    except (subprocess.TimeoutExpired, Exception):
        return []
    return _collection_recorders(result.stdout or "")


def _service_requires_config(svc: ServiceDef) -> bool:
    return svc.launch_type not in ("make", "vllm", "collection")


# make targets that read a config.yml of their own (gitignored, so a pulled
# checkout has none), by the pipeline whose Config form writes it: nginx
# renders its upstreams from it, gunicorn and the celery worker load it at import
_MAKE_CONFIG_PIPELINES: dict[str, str] = {
    "nginx": "Nginx",
    "flask": "Flask Backend",
    "celery": "Flask Backend",
}


def _service_python_hint(svc: ServiceDef) -> str:
    return "3.12" if svc.conda_env == "vfa-vllm" else "3.10"


_MLLM_KEY_UNDECRYPTABLE = (
    "[red]The MLLM api key cannot be decrypted on this machine: "
    "~/.openmmla/master.key is missing or is not the key it was saved with. "
    "Re-enter the key on the MLLM Server Config tab.[/red]"
)


def _undecryptable_api_key(config: dict) -> bool:
    """whether the MLLM api key is still ENC(...) after a decrypt attempt.

    That means this machine's ~/.openmmla/master.key is not the one the key was
    saved with. Launching anyway would bring vllm up demanding a token the frame
    analyzer cannot produce, and the 401s would look like a model problem."""
    key = str(decrypt_config_values(config.get("api_key") or ""))
    return key.startswith("ENC(") and key.endswith(")")


def _vllm_serve_command(config: dict | None = None, *, mask: bool = False) -> str:
    """the `vllm serve` command line, with the api key decrypted for vllm itself.

    The form keeps the key as save_config wrote it, ENC(...) and all, the way the
    System Settings forms keep theirs; it is decrypted here, at the one place that
    hands it to a server. Undecryptable, it stays ENC(...) rather than becoming
    empty, so _launch_vllm_server can say so instead of starting a server whose
    key nobody holds. mask=True is for the log pane: the same line with the key
    shortened, since the command is echoed there in full."""
    cfg = config or _mllm_config(_find_project_root())
    api_key = str(decrypt_config_values(cfg["api_key"]))
    args = [
        "vllm", "serve", cfg["model"],
        "--host", cfg["host"],
        "--port", str(cfg["port"]),
        "--dtype", cfg["dtype"],
        "--max-model-len", str(cfg["max_model_len"]),
        "--limit-mm-per-prompt", cfg["limit_mm_per_prompt"],
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--api-key", _mask_secret(api_key) if mask else api_key,
    ]
    return " ".join(shlex.quote(arg) for arg in args)


# one Terminal window per component on a Mac, told what to run once its shell
# is ready for it. Terminal has no AppleScript call for a new tab, so each
# component after the first asks for one with a Cmd-T keystroke; what comes
# back is a tab on some Macs and a window of its own on others, and either is
# fine as long as it is that new shell the command goes to. The command is
# typed into the shell found by its tty, never into "the front window", which
# is how a component's command used to land in the window of the component
# before it, where its program was already running: the command was read as
# that program's input and the component never started, while the log said it
# had launched. A shell still reading its startup files (conda, oh-my-zsh)
# drops what is typed into it, so each one is given until it is idle. When no
# new shell comes -- no new tab, or System Events not allowed to press Cmd-T
# -- the component gets a window of its own instead of a busy tab, and the
# commands that got nowhere are named in the log.
# argv: one shell command per component. Returns one line each: tab, window,
# or failed.
_MAC_TABS_SCRIPT = r"""
on run argv
	set outcomes to {}
	try
		tell application "Terminal"
			activate
			do script (item 1 of argv)
		end tell
		set end of outcomes to "window"
	on error
		set end of outcomes to "failed"
	end try
	set canKeystroke to false
	if (count of argv) > 1 then set canKeystroke to my waitUntilFront()
	repeat with i from 2 to (count of argv)
		set shellTab to missing value
		if canKeystroke then set shellTab to my openShell()
		try
			if shellTab is missing value then
				tell application "Terminal" to do script (item i of argv)
				set end of outcomes to "window"
			else
				my waitUntilReady(shellTab)
				tell application "Terminal" to do script (item i of argv) in shellTab
				set end of outcomes to "tab"
			end if
		on error
			set end of outcomes to "failed"
		end try
	end repeat
	return my joinText(outcomes)
end run

-- Cmd-T goes wherever the keyboard is: false when Terminal will not take it,
-- and every component then gets a window of its own
on waitUntilFront()
	repeat 30 times
		try
			tell application "System Events"
				if frontmost of process "Terminal" then return true
				set frontmost of process "Terminal" to true
			end tell
		on error
			return false
		end try
		delay 0.1
	end repeat
	return false
end waitUntilFront

on ttysNow()
	set found to {}
	tell application "Terminal"
		repeat with w in windows
			try
				repeat with t in tabs of w
					try
						set end of found to tty of t
					end try
				end repeat
			end try
		end repeat
	end tell
	return found
end ttysNow

-- the shell a Cmd-T made, wherever Terminal put it: the one tab whose tty was
-- not open before. Missing value when the keystroke made none
on openShell()
	try
		set before_ to my ttysNow()
	on error
		return missing value
	end try
	try
		tell application "System Events" to tell process "Terminal"
			set frontmost to true
			keystroke "t" using command down
		end tell
	on error
		return missing value
	end try
	repeat 60 times
		delay 0.05
		try
			tell application "Terminal"
				repeat with w in windows
					repeat with t in tabs of w
						set thisTty to ""
						try
							set thisTty to tty of t
						end try
						if thisTty is not "" and before_ does not contain thisTty then
							return contents of t
						end if
					end repeat
				end repeat
			end tell
		end try
	end repeat
	return missing value
end openShell

-- idle means the startup files are read and the prompt is up: what is typed
-- before that is lost
on waitUntilReady(theTab)
	repeat 60 times
		try
			tell application "Terminal"
				if (count of processes of theTab) > 0 and not busy of theTab then return true
			end tell
		end try
		delay 0.1
	end repeat
	return false
end waitUntilReady

on joinText(theList)
	set saved to AppleScript's text item delimiters
	set AppleScript's text item delimiters to linefeed
	set out to theList as text
	set AppleScript's text item delimiters to saved
	return out
end joinText
"""

# two cards started at once would trade keystrokes and new shells, so their
# windows are opened one launch at a time
_MAC_TABS_LOCK = threading.Lock()


class ServicePanel(Widget):

    DEFAULT_CSS = """
    ServicePanel {
        layout: horizontal;
        width: 1fr;
        height: 1fr;
    }
    /* wide enough for the longest leaf with all three of its markers: at 32 the
       (R) of "Dashboard (Flask) [E] [C] (R)" and of every longer leaf was cut
       off. The tree is drawn with 3-column guides for the same reason */
    #svc-sidebar {
        width: 44;
        max-width: 40%;
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
    .svc-legend {
        color: $text-muted;
        padding: 0 1;
    }
    #svc-target-refresh {
        width: 5;
        min-width: 5;
        height: 3;
        margin-left: 1;
    }
    #svc-target-local-note {
        display: none;
        width: 1fr;
        height: 3;
        padding-top: 1;
        color: $text-muted;
        text-style: italic;
    }
    /* nodes whose host is not the user's to pick (System Settings are edited
       on this machine, a system service runs where System Settings put it,
       Session Control has no host): swap the selector for a note, keeping the
       bar's height so the content below does not jump */
    #svc-target-bar.host-fixed Label {
        color: $text-muted;
    }
    #svc-target-bar.host-fixed Select,
    #svc-target-bar.host-fixed Button {
        display: none;
    }
    #svc-target-bar.host-fixed #svc-target-local-note {
        display: block;
    }
    #svc-target-local-note {
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    /* the bar must stay one row: never let a long "name  (offline ✗)" label
       wrap at narrow terminal widths */
    #svc-target-select SelectCurrent #label {
        text-wrap: nowrap;
        text-overflow: ellipsis;
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
    /* the config form scrolls internally (ConfigForm is 1fr), so its container
       is a plain Vertical: the sync bar and status line mounted after the form
       stay on screen below it instead of being pushed past the viewport */
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
    .sync-note {
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }
    /* whose settings a Connections form shows when it is not this machine's */
    .settings-origin {
        height: auto;
        margin-bottom: 1;
        color: $warning;
    }
    /* the note sits right on top of the picker */
    .sync-bar-noted {
        margin-top: 0;
        padding-top: 0;
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
        self._config_container: Vertical | None = None
        self._ssh_profile_names: list[str] = [
            p.name for p in load_ssh_profiles()
        ]
        self._collection_last_params: dict[tuple[str, str], dict] = {}
        # collection choices that describe the *recording session* rather than
        # the machine doing the recording: they follow the user from host to
        # host, so setting up a multi-machine session is one pass, not one
        # full re-entry per host.
        self._collection_sticky: dict[str, object] = {}
        # host -> sessions its recorders belong to, as the last status probe
        # found them; and the one the card was last pointed at for that host
        self._collection_host_sessions: dict[str, list[str]] = {}
        self._collection_followed: dict[str, str] = {}
        # ...and the ones that are per-machine (output root, host label), kept
        # so a rebuild on the same host does not discard local edits
        self._collection_target_sticky: dict[str, dict[str, object]] = {}
        self._collection_role: str = "audio"
        # (target, service name) -> "native" | "docker" for the infra cards; the
        # card is rebuilt on every tree/host change and would otherwise fall back
        # to the ParamDef default, silently sending Stop down the other path
        self._infra_mode: dict[tuple[str, str], str] = {}
        # node name -> SSH profile the user last pointed that node at; every
        # node keeps its own host, across restarts. System services whose
        # address names a machine are not in here: see _derive_node_host
        self._node_hosts: dict[str, str] = load_node_hosts(self._root)
        # node name -> NodeHost as last resolved off the UI thread; the tree
        # markers read this rather than resolve names while rendering
        self._node_host_cache: dict[str, NodeHost] = {}
        self._current_node_host: NodeHost = NodeHost()
        # (node name, host) of a system service card the user moved off the
        # machine System Settings name: it lasts until another node is opened
        self._host_override: tuple[str, str] | None = None
        # whose System Settings the Connections forms show and save; shared by
        # those forms so one pick lets the user read through a host's settings,
        # and back on Local with every new console
        self._settings_target: str = "local"
        # card and host the next log lines belong to, and the one the log last
        # drew a divider for (see _log)
        self._log_context: str = ""
        self._log_context_shown: str = ""
        self._fallback_noted: dict[str, str] = {}
        # (card, host) whose config keeps in Base what the Bases entries now hold,
        # said once each (move_source_settings)
        self._source_moves_noted: set[tuple[str, str]] = set()
        # host -> the differences between its own System Settings and Local's
        # that a Start has already pointed out
        self._own_settings_noted: dict[str, list[str]] = {}
        # target -> {conda env: Ready | Partial: ... | Missing}; feeds the tree's
        # [E] markers and is refreshed off the UI thread (conda is slow)
        self._env_statuses: dict[str, dict[str, str]] = {}
        self._env_statuses_at: dict[str, float] = {}  # when each host's were read
        # remote host -> {config file, as its local path -> whether that host
        # has it}; feeds the [C] markers of the nodes on it (Local's are
        # read from disk as the tree is drawn)
        self._config_presence: dict[str, dict[str, bool]] = {}
        # session id -> hosts this TUI launched it on, so "Stop All Hosts"
        # reaches a machine even when it is currently unreachable
        self._collection_launch_targets: dict[str, set[str]] = {}
        self._pending_collection_delete: tuple[str, str] | None = None
        # (host, session id) of a Start that was held back because the session
        # has ended; the second press on the same pair goes ahead
        self._pending_ended_start: tuple[str, str] | None = None
        # (target, session, host) keys of downloads already running, so a second
        # press — or the other collection tab's Download — is refused instead of
        # racing the first one into the same staging directory
        self._downloads_in_flight: set[str] = set()
        # (target, session) of a Delete Remote that is running: a Download of
        # that folder waits for it, as a Delete waits for a Download
        self._remote_deletes_in_flight: set[tuple[str, str]] = set()
        self._staging_swept: bool = False
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
            yield Static("\\[E] env  \\[C] config  (R) running", classes="svc-legend")
            yield Static("green ok · yellow partial · red missing", classes="svc-legend")
            tree: Tree[str] = Tree("OpenMMLA", id="svc-tree")
            tree.guide_depth = 3
            tree.root.expand()
            yield tree
        with Vertical(id="svc-main"):
            with Horizontal(id="svc-target-bar"):
                yield Label("Host:")
                yield Select(target_options, value="local", id="svc-target-select")
                yield Static(_SETTINGS_HOST_NOTE, id="svc-target-local-note")
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
        self._sweep_staging_once()
        # populate running markers asynchronously; the tree itself renders
        # instantly from cached states
        self._refresh_visible_statuses()
        self._probe_targets()
        self._kick_env_status_refresh("local")

    def refresh_env_markers(self, target: str) -> None:
        """the Environment tab created, removed or filled an env on `target`:
        the [E] markers of the nodes on that host are stale."""
        self._kick_env_status_refresh(target)

    def _kick_env_status_refresh(self, target: str) -> None:
        # a group per host: nodes sit on different hosts, and one host's check
        # must not cancel another's
        self.run_worker(
            self._refresh_env_statuses(target),
            group=f"launcher-env-status:{target}",
            exclusive=True,
        )

    async def _refresh_env_statuses(self, target: str) -> None:
        """recompute the conda env states behind the tree's [E] markers."""
        try:
            if target == "local":
                statuses = await asyncio.to_thread(env_statuses_local, self._root)
            else:
                profile = get_profile_by_name(target)
                if profile is None:
                    return
                statuses = await env_statuses_remote(profile)
        except Exception as e:
            self._log(f"[yellow]Environment check on {target} failed: {e}[/yellow]")
            return
        self._env_statuses[target] = statuses
        self._env_statuses_at[target] = time.time()
        self._build_tree()

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
        self._kick_env_status_refresh(self._get_panel_target())

    async def _async_probe_targets(self) -> None:
        try:
            # a host is taken for offline on its second miss in a row
            states = await asyncio.to_thread(probe_all_profiles, confirm_offline=2)
            if states != self._target_states:
                for name, state in states.items():
                    previous = self._target_states.get(name)
                    if previous is not None and previous != state and state in ("online", "offline"):
                        color = "green" if state == "online" else "red"
                        self._log(f"[{color}]Host '{name}' is now {state}.[/{color}]")
                self._target_states = states
                self._refresh_target_options()
                self._refresh_visible_statuses()
                # a group of its own: only an actual rebuild may replace the
                # UI worker, a mere look must not cancel a reload in flight
                self.run_worker(
                    self._async_rebind_current_node(),
                    group="launcher-rebind",
                    exclusive=True,
                )
        finally:
            # the re-probe loop must survive any failure above
            if self._target_probe_timer is not None:
                self._target_probe_timer.stop()
            self._target_probe_timer = self.set_timer(_TARGET_PROBE_INTERVAL_SEC, self._probe_targets)

    async def _async_rebind_current_node(self) -> None:
        """rebuild the card on screen when its host is no longer the one it
        was built for (System Settings saved, a host came up or went down)."""
        name = self._current_service_name
        svc = self._svc_map.get(name) if name else None
        if svc is None:
            return
        before = self._current_node_host
        node = await asyncio.to_thread(self._resolve_node_host, svc)
        if node != before and self._current_service_name == name:
            self.run_worker(
                self._reload_current_service_view(),
                group=_LAUNCHER_UI_WORKER_GROUP,
                exclusive=True,
            )

    def on_show(self) -> None:
        self._refresh_target_options()
        # an env may have been installed meanwhile, in the Environment tab or by
        # hand over ssh: the markers of every host a node sits on catch up, but
        # not on every flip of the tabs
        now = time.time()
        for target in list(self._env_statuses):
            if (now - self._env_statuses_at.get(target, 0.0) > _ENV_MARKER_MAX_AGE
                    and TARGET_STATES.get(target) != "offline"):
                self._kick_env_status_refresh(target)
        if self._current_service_name == "Collection Session":
            self.run_worker(
                self._reload_current_service_view(),
                group=_LAUNCHER_UI_WORKER_GROUP,
                exclusive=True,
            )

    # bound with @on: Textual names this message's handler "on_sshform_...", so
    # the "on_ssh_form_..." spelling was never called and a saved profile only
    # reached the Host selector with the next background probe
    @on(SSHForm.ProfilesChanged)
    def on_ssh_form_profiles_changed(self, event: SSHForm.ProfilesChanged) -> None:
        event.stop()
        renamed = getattr(event, "renamed", None)
        if renamed and renamed[0] in self._node_hosts.values():
            # the cards that were pointed at this profile follow its new name
            self._node_hosts = {
                node: (renamed[1] if target == renamed[0] else target)
                for node, target in self._node_hosts.items()
            }
            save_node_hosts(self._root, self._node_hosts)
        self._refresh_target_options()
        self._probe_targets()

    @on(SSHForm.ConnectionTested)
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

    def _show_host_bar(self, note: str | None = None) -> None:
        """show the Host selector, or `note` in its place on a node whose host
        is not the user's to pick.

        System Settings live in the local project, a system service runs where
        System Settings put it, and Session Control has no host at all. A
        greyed-out selector was misleading there, so the selector and its
        refresh button are hidden instead."""
        try:
            bar = self.query_one("#svc-target-bar", Horizontal)
            select = self.query_one("#svc-target-select", Select)
            refresh = self.query_one("#svc-target-refresh", Button)
            label = self.query_one("#svc-target-local-note", Static)
        except Exception:
            return
        fixed = note is not None
        if fixed:
            label.update(note)
        bar.set_class(fixed, "host-fixed")
        select.disabled = fixed
        refresh.disabled = fixed

    # ── per-node hosts ───────────────────────────────────────────

    def _unusable_reason(self, target: str, profiles: list) -> str:
        """why a remote host cannot serve a card right now, or "". Asks a host
        it has not met for its platform, so it runs off the UI thread."""
        if TARGET_STATES.get(target) == "offline":
            return "is offline"
        if TARGET_PLATFORMS.get(target) is None and TARGET_STATES.get(target) == "online":
            profile = next((p for p in profiles if p.name == target), None)
            if profile is not None and hasattr(profile, "base_ssh_args"):
                remote_platform(profile)
        if TARGET_PLATFORMS.get(target) == "windows":
            return WINDOWS_HOST_NOTE
        return ""

    def _derive_node_host(self, svc: ServiceDef, profiles: list) -> NodeHost:
        if svc.launch_type == "make":
            make_target = _make_target_for(svc.name)
            endpoint = system_service_endpoint(
                self._root, "flask" if make_target == "celery" else make_target)
            host = endpoint[0] if endpoint else ""
            # a loopback address names no machine: that card is the user's pick
            if not is_loopback_host(host):
                field = _SERVICE_HOST_FIELDS.get(make_target, "System Settings")
                bound = target_for_service_host(host, profiles)
                if not bound:
                    return NodeHost(
                        follows=field, machine=host,
                        fallback_from=(
                            f"System Settings put it on {host} ({field}), which is neither this "
                            f"machine nor a saved SSH profile"))
                reason = self._unusable_reason(bound, profiles) if bound != "local" else ""
                if reason:
                    return NodeHost(
                        follows=field, machine=host,
                        fallback_from=f"System Settings put it on '{bound}' ({field}), which {reason}")
                return NodeHost(bound, follows=field, machine=host, machine_target=bound)

        saved = self._node_hosts.get(svc.name, "local")
        if saved == "local":
            return NodeHost()
        if not any(profile.name == saved for profile in profiles):
            return NodeHost(fallback_from=f"'{saved}', where it was last used, is no longer a saved SSH profile")
        reason = self._unusable_reason(saved, profiles)
        if reason:
            return NodeHost(fallback_from=f"'{saved}', where it was last used, {reason}")
        return NodeHost(saved)

    def _resolve_node_host(self, svc: ServiceDef, profiles: list | None = None) -> NodeHost:
        """where this node's actions run. Reads config files and may resolve
        names, so it belongs off the UI thread; the tree reads the cached result."""
        node = self._derive_node_host(svc, load_ssh_profiles() if profiles is None else profiles)
        self._node_host_cache[svc.name] = node
        return node

    def _card_target(self, svc: ServiceDef, node: NodeHost) -> str:
        """the host the card is on: the node's own, unless the user moved this
        system service card somewhere else for the time being."""
        override = self._host_override
        if override is None or override[0] != svc.name:
            return node.target
        target = override[1]
        usable = target == "local" or (
            TARGET_STATES.get(target) != "offline"
            and any(value == target for _, value in target_options())
        )
        if not usable or target == node.target:
            self._host_override = None
            return node.target
        return target

    def _apply_node_host(self, svc: ServiceDef, node: NodeHost) -> None:
        """point the Host bar, and everything that follows it, at a node's host."""
        self._current_node_host = node
        self._show_host_bar(None)
        target = self._card_target(svc, node)
        self._set_log_context(svc, target)
        # said once per cause: the card is rebuilt often, the reason stays the same
        if node.fallback_from and self._fallback_noted.get(svc.name) != node.fallback_from:
            self._log(f"[yellow]{svc.display_name}: {node.fallback_from}. Showing Local.[/yellow]")
        self._fallback_noted[svc.name] = node.fallback_from
        if self._point_host_select(target):
            # as on a manual host switch: the [E] markers of that host catch up
            self._kick_env_status_refresh(self._last_target)

    def _point_host_select(self, target: str) -> bool:
        """move the Host selector to `target` on the console's own account (a
        node was opened); True when that changed the current host."""
        try:
            select = self.query_one("#svc-target-select", Select)
        except Exception:
            return False
        changed = target != self._last_target
        # not a pick by the user: the Select moves without a Changed message,
        # and its handler would ignore the current target anyway
        self._last_target = target
        if select.value != target:
            with select.prevent(Select.Changed):
                try:
                    select.value = target
                except Exception:
                    # a profile saved after the options were last built
                    try:
                        select.set_options(target_options())
                        select.value = target
                    except Exception:
                        target = self._last_target = "local"
                        select.value = target
        if changed:
            try:
                self.query_one("#svc-cmd-session", CommandSession).set_target(target)
            except Exception:
                pass
        return changed

    def _remember_node_host(self, target: str) -> None:
        """file the user's Host pick under the node it was made on."""
        name = self._current_service_name
        if not name:
            return
        if self._current_node_host.follows:
            # System Settings stay this card's home: the move is a one-off
            self._host_override = None if target == self._current_node_host.target else (name, target)
            return
        if target == "local":
            self._node_hosts.pop(name, None)
        else:
            self._node_hosts[name] = target
        node = NodeHost(target)
        self._node_host_cache[name] = node
        self._current_node_host = node
        save_node_hosts(self._root, self._node_hosts)

    def _off_configured_machine(self, svc: ServiceDef, target: str) -> bool:
        """whether `target` is another machine than the one System Settings
        name for this system service. The card then reports and controls that
        host's own port; the sidebar marker keeps following the configured
        endpoint, which is what the pipelines connect to."""
        node = self._node_host_cache.get(svc.name)
        return bool(node and node.follows and target != node.machine_target)

    @staticmethod
    def _is_local_only_node(node_str: str) -> bool:
        """tree nodes whose content ignores the Host selector."""
        return node_str.startswith("__shared__") or node_str in (
            "__ssh_profiles__", "__experiments__", "__tasks__",
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "svc-target-select" and any(
                isinstance(node, ServiceCard) and node.service_def.name == _ASR_BASE_CARD
                for node in event.select.ancestors):
            # the Session, the Bases and the Mode decide the card's Speakers line
            self.call_after_refresh(self._show_speakers_summary)
            return
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
            if val != "local" and TARGET_PLATFORMS.get(val) is None:
                # the first visit to this host: find out what it is before any
                # card starts talking bash to it
                self._log(f"[yellow]Checking host '{val}'...[/yellow]")
                self.run_worker(
                    self._async_switch_after_platform_check(val),
                    group="launcher-host-platform",
                    exclusive=True,
                )
                return
            if self._refuse_unsupported_host(val):
                return
            self._switch_host(val)

    def _refuse_unsupported_host(self, val: str) -> bool:
        """put the Host selector back when `val` is a machine the console
        cannot drive, and say why."""
        if val == "local" or TARGET_PLATFORMS.get(val) != "windows":
            return False
        self._log(f"[red]Host '{val}' {WINDOWS_HOST_NOTE}. Staying on '{self._last_target}'.[/red]")
        try:
            select = self.query_one("#svc-target-select", Select)
            with select.prevent(Select.Changed):
                select.set_options(target_options())  # the label now says so too
                select.value = self._last_target
        except Exception:
            pass
        return True

    async def _async_switch_after_platform_check(self, val: str) -> None:
        profile = get_profile_by_name(val)
        if profile is not None:
            await asyncio.to_thread(remote_platform, profile)
        if self._get_panel_target() != val:
            return  # the user picked something else while ssh was at work
        if not self._refuse_unsupported_host(val):
            self._switch_host(val)

    def _switch_host(self, val: str) -> None:
        """the Host selector moved to `val` by the user's hand."""
        # snapshot against the host the card still belongs to, so per-host
        # values (output root, host label) are not filed under the new one
        self._capture_collection_card_state(self._last_target)
        self._capture_infra_mode(self._last_target)
        self._last_target = val
        if self._current_shared_section:
            self._settings_target = val
            self.query_one("#svc-cmd-session", CommandSession).set_target(val)
            self.run_worker(
                self._reload_shared_form(),
                group=_LAUNCHER_UI_WORKER_GROUP,
                exclusive=True,
            )
            return
        self._remember_node_host(val)
        current = self._svc_map.get(self._current_service_name or "")
        if current is not None:
            self._set_log_context(current, val)
        if val != "local":
            self._log(f"[yellow]Switching host to '{val}' — loading remote state...[/yellow]")
        cmd = self.query_one("#svc-cmd-session", CommandSession)
        cmd.set_target(val)
        # only this node moved: its state from the previous host must not
        # linger, every other node still sits where it was
        if self._current_service_name:
            self._svc_states.pop(self._current_service_name, None)
        self.run_worker(
            self._reload_current_service_view(capture=False),
            group=_LAUNCHER_UI_WORKER_GROUP,
            exclusive=True,
        )
        self._refresh_visible_statuses()
        self._kick_env_status_refresh(val)

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tabbed_content.id != "svc-sub-tabs":
            return
        self._set_command_session_visible(event.pane.id != "svc-tab-config")
        if event.pane.id == "svc-tab-config":
            # a stream started or stopped on the Streams tab since the form was
            # built: its dropdown says what the Stream Server says now
            self._remark_live_streams()

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
        try:
            tree = self.query_one("#svc-tree", Tree)
        except Exception:
            return  # a worker that finished while the console was closing
        tree.clear()

        shared_node = tree.root.add(_SYSTEM_SERVICES_LABEL, data="__shared__")
        shared_node.expand()
        added_shared_sections: set[str] = set()
        for group_label, items in _SETTINGS_GROUPS:
            # headings only sort the forms: always open, never collapsible
            group = shared_node.add(f"[dim]{group_label}[/dim]", data=f"__cat_settings_{group_label}")
            group.allow_expand = False
            group.expand()
            for item in items:
                if item == "SSH Profiles":
                    group.add_leaf("SSH Profiles", data="__ssh_profiles__")
                elif item == "Experiments":
                    group.add_leaf("Experiments", data="__experiments__")
                elif item == "Tasks":
                    group.add_leaf("Tasks", data="__tasks__")
                elif item in SHARED_SECTIONS:
                    group.add_leaf(_shared_section_label(item), data=f"__shared__{item}")
                    added_shared_sections.add(item)
        for sec in SHARED_SECTIONS:
            if sec not in added_shared_sections:
                shared_node.add_leaf(_shared_section_label(sec), data=f"__shared__{sec}")

        categories: dict[str, list[ServiceDef]] = {}
        for svc in self._services:
            categories.setdefault(svc.category, []).append(svc)

        infra_svcs = categories.get("System Services", [])
        if infra_svcs:
            # where they run: the host most of them share goes on the heading
            # (a label on every leaf would not fit the sidebar), and a service
            # elsewhere says so on its own line
            hosts = {svc.name: self._svc_host_label(svc) for svc in infra_svcs}
            known = [label for label in hosts.values() if label]
            shared = Counter(known).most_common(1)[0][0] if known else ""
            infra_node = tree.root.add(f"System Services{shared}", data="__cat_System Services")
            infra_node.expand()
            for svc in infra_svcs:
                own = hosts[svc.name] if hosts[svc.name] != shared else ""
                infra_node.add_leaf(f"{svc.display_name}{self._svc_markers(svc)}{own}", data=svc.name)

        collection_svcs = categories.get("Collection", [])
        if collection_svcs:
            collection_node = tree.root.add("Collection", data="__cat_Collection")
            collection_node.expand()
            for svc in collection_svcs:
                collection_node.add_leaf(f"{svc.display_name}{self._svc_markers(svc)}", data=svc.name)

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
                cat_node.add_leaf(f"{svc.display_name}{self._svc_markers(svc)}", data=svc.name)
        # a session start/stop spans every pipeline, so it is not filed under one
        tree.root.add_leaf("Session Control", data="__session_control__")

    def _svc_markers(self, svc: ServiceDef) -> str:
        """build status marker string for a service tree leaf.

        Reads the cached running state only — live detection (subprocess/
        socket/SSH probes) happens in the _refresh_visible_statuses worker,
        never on the UI thread while rendering the tree."""
        markers = ""
        node = self._node_host_cache.get(svc.name) or NodeHost(self._node_hosts.get(svc.name, "local"))
        # no [E] for a machine the console cannot look into
        if _service_uses_conda_env(svc) and not (node.follows and not node.machine_target):
            statuses = self._env_statuses.get(node.target) or {}
            color = _env_marker_color(statuses.get(svc.conda_env))
            if color:
                markers += f" [{color}]\\[E][/{color}]"
        config_file = self._svc_config_file(svc)
        # a config file this service needs, on the host the node sits on: red
        # when it has not been made there yet, none while that is not known
        if config_file and not (node.follows and not node.machine_target):
            if node.target == "local":
                present: bool | None = os.path.isfile(config_file)
            else:
                present = self._config_presence.get(node.target, {}).get(os.path.abspath(config_file))
            if present is not None:
                color = "green" if present else "red"
                markers += f" [{color}]\\[C][/{color}]"
        if self._svc_states.get(svc.name, False):
            markers += " [green](R)[/green]"
        return markers

    def _svc_host_label(self, svc: ServiceDef) -> str:
        """" @ server-01": where a system service runs, as the status probe
        last resolved it (nothing before that, rather than a guess). It is
        the machine System Settings put it on, offline or not, else the host
        its card was last pointed at."""
        node = self._node_host_cache.get(svc.name)
        if node is None:
            return ""
        if node.machine_target:
            host = "Local" if node.machine_target == "local" else node.machine_target
        elif node.follows:
            # a machine the console cannot use: the address says where it runs
            host = node.machine
            if "offline" in node.fallback_from:
                return f" [red]@ {rich_escape(host)} (offline)[/red]"
        else:
            host = "Local" if node.target == "local" else node.target
        return f" [dim]@ {rich_escape(host)}[/dim]"

    def _svc_config_file(self, svc: ServiceDef) -> str:
        """the config file a node needs, as its local path; "" for none."""
        if svc.launch_type == "make":
            make_target = _make_target_for(svc.name)
            if make_target == "mediamtx":
                return os.path.join(self._root, _MEDIAMTX_CONFIG_REL)
            pipeline = self._pipeline_map.get(_MAKE_CONFIG_PIPELINES.get(make_target, ""))
        else:
            pipeline = self._pipeline_for_service(svc.name)
        return pipeline.config_path if pipeline else ""

    def _remote_config_presence(self, profile) -> dict[str, bool] | None:
        """which of the nodes' config files the host of `profile` has, asked in
        one ssh round trip; None when it could not be told. Runs in a worker
        thread."""
        files = sorted({os.path.abspath(f) for f in map(self._svc_config_file, self._services) if f})
        if not files:
            return {}
        command = "; ".join(
            f"[ -f {_quote_remote_path(self._remote_config_path(f, profile))} ] && echo Y || echo N"
            for f in files
        )
        try:
            result = ssh_run_sync(profile, command, timeout=8.0)
        except Exception:
            return None
        answers = (result.stdout or "").split()
        # a shell that is not sh (Windows) answers something else entirely
        if result.returncode != 0 or len(answers) != len(files) or not set(answers) <= {"Y", "N"}:
            return None
        return {path: answer == "Y" for path, answer in zip(files, answers)}

    def _note_config_presence(self, target: str, local_path: str, exists: bool) -> None:
        """what a save, a sync or the check before a Start just found out about
        a config file on `target`: its [C] marker says so at once."""
        if target == "local":
            return
        known = self._config_presence.setdefault(target, {})
        path = os.path.abspath(local_path)
        if known.get(path) != exists:
            known[path] = exists
            self._build_tree()

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
        self._current_node_host = NodeHost()
        self._host_override = None
        if node_str in _LOCAL_SETTINGS_NOTES:
            self._show_host_bar(_LOCAL_SETTINGS_NOTES[node_str])
        elif node_str == "__session_control__":
            self._show_host_bar(_SESSION_CONTROL_HOST_NOTE)

        if node_str.startswith("__shared__"):
            self._set_command_session_visible(False)
            section_name = node_str.replace("__shared__", "")
            scroll = Vertical(classes="svc-config-scroll")
            await content_area.mount(scroll)
            self._config_container = scroll
            self._current_shared_section = section_name
            await self._mount_shared_form(scroll, section_name)
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
            # no host here: the signal is published to the Redis of System
            # Settings from this machine, whichever machines the bases run on
            choices = await asyncio.to_thread(self._artifact_session_choices_for_target, "local")
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

        # every node has its own host: the one System Settings give a system
        # service, else the one this node was last pointed at
        node = await asyncio.to_thread(self._resolve_node_host, base_svc)
        self._apply_node_host(base_svc, node)
        target = self._get_panel_target()
        svc, is_running = await self._service_view_state(base_svc, target)

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
        # System Settings or the host states may have changed since the card
        # was built, so the node's host is worked out again
        node = await asyncio.to_thread(self._resolve_node_host, svc)
        if self._current_service_name != svc.name:
            self._build_tree()
            return
        self._apply_node_host(svc, node)
        target = self._get_panel_target()
        display_svc, is_running = await self._service_view_state(svc, target)
        if self._current_service_name != svc.name:
            # the user moved to another tree node while the (possibly remote)
            # probe ran; that node owns the content area now, so only refresh
            # the tree markers
            self._build_tree()
            return

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
        self._note_card_state(svc, target, is_running)
        if svc.name in _BASE_CARD_PIPELINES and target != "local":
            # the Bases the dropdowns offer: read here, off the UI thread, and
            # kept for the Start and the redraws that follow
            self._load_config_for_target(os.path.join(svc.config_dir, "config.yml"), show_status=False, target=target)
        if svc.name == "IPS Base":
            # the Main Camera choices, likewise
            self._transform_matrix_ids(target, refresh=True)
        return self._service_for_target(svc, target), is_running

    def _note_card_state(self, svc: ServiceDef, target: str, is_running: bool) -> None:
        """let the card's state feed the sidebar marker, unless the card was
        moved off the machine that marker stands for."""
        if not self._off_configured_machine(svc, target):
            self._svc_states[svc.name] = is_running

    async def _populate_deferred_panes(
        self,
        tabs: TabbedContent,
        svc: ServiceDef,
        pipeline: PipelineDef,
        config_scroll: Vertical,
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
                panel = StreamPanel(
                    streams,
                    config_path=stream_config_path,
                    project_dir=self._root,
                    stream_server=self._stream_server_address,
                    default_kind=_card_stream_kind(svc.name),
                )
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
            if svc.name == _ASR_BASE_CARD:
                self._list_speakers(self._get_panel_target())

            config_scroll = Vertical(classes="svc-config-scroll")
            config_pane = TabPane("Config", config_scroll, id="svc-tab-config")
            await tabs.add_pane(config_pane)
            self._config_container = config_scroll
            self._current_config_local_path = pipeline.config_path
            # defer the heavy widget building (config form has dozens of field
            # rows; streams/transform/prompts may hit disk or SSH) so the Launch
            # card paints immediately when navigating the tree
            self.call_after_refresh(self._populate_deferred_panes, tabs, svc, pipeline, config_scroll)
        elif svc.launch_type == "make" and _make_target_for(svc.name) == "mediamtx":
            tabs = TabbedContent(id="svc-sub-tabs")
            await content_area.mount(tabs)
            launch_scroll = VerticalScroll(classes="svc-launch-scroll")
            await tabs.add_pane(TabPane("Launch", launch_scroll, id="svc-tab-launch"))
            await launch_scroll.mount(ServiceCard(svc, is_running=is_running))

            # mediamtx.yml of the host the card is on: ports, and recording on the server
            target = self._get_panel_target()
            local_path = os.path.join(self._root, _MEDIAMTX_CONFIG_REL)
            profile = get_profile_by_name(target) if target != "local" else None
            config_scroll = VerticalScroll(classes="svc-launch-scroll")
            await tabs.add_pane(TabPane("Config", config_scroll, id="svc-tab-config"))
            await config_scroll.mount(StreamServerConfigPanel(
                config_path=local_path, target=target, ssh_profile=profile,
                remote_path=self._remote_config_path(local_path, profile) if profile is not None else None,
            ))
            # what the server holds and the way to make room: asked over HTTP,
            # sized over a shell on the card's host
            recordings_scroll = VerticalScroll(classes="svc-launch-scroll")
            await tabs.add_pane(TabPane("Recordings", recordings_scroll, id="svc-tab-recordings"))
            await recordings_scroll.mount(self._stream_server_recordings_panel(profile))
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

            config_scroll = Vertical(classes="svc-config-scroll")
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
            # the cameras calibration made on this machine: their images, their
            # parameters, and the sync of those to the host that captures
            if svc.name == "IPS Camera Calibration":
                await scroll.mount(CameraManagerPanel(
                    cameras_dir=_ips_cameras_local_dir(self._root),
                    target=self._get_panel_target(),
                    config_path=self._ips_base_config_path(),
                    ssh_profiles=[
                        profile.name for profile in load_ssh_profiles()
                        if TARGET_PLATFORMS.get(profile.name) != "windows"
                    ],
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

    def _note_host_recorders(self, target: str, recorders: list) -> bool:
        """keep which sessions a host is recording (newest recorder first);
        returns whether it records at all. Called from the status probes."""
        sessions: list[str] = []
        for _role, session, _pid in sorted(recorders, key=lambda item: _pid_order(item[2]), reverse=True):
            session = _safe_session_id(session)
            if session and session not in sessions:
                sessions.append(session)
        self.__dict__.setdefault("_collection_host_sessions", {})[target] = sessions
        return bool(recorders)

    def _host_recording_session(self, target: str, shown: object = "") -> str:
        """the session the card of `target` should open on because that host is
        recording it, or "" when it records nothing. One session id follows the
        user from host to host (one take, several machines); with two groups
        recording at once on different hosts that id is the other group's, and
        Stop would look for it here in vain. What the host records wins."""
        sessions = self.__dict__.get("_collection_host_sessions", {}).get(target) or []
        if not sessions:
            return ""
        shown = _safe_session_id(shown)
        return shown if shown in sessions else sessions[0]

    def _follow_host_recording_session(self, target: str) -> None:
        """after a status probe: point the card on screen at the session its
        host records. Once per change, so a session the user picks on purpose
        while the host records (to download an older one) is left alone."""
        followed = self.__dict__.setdefault("_collection_followed", {})
        sessions = self.__dict__.get("_collection_host_sessions", {}).get(target) or []
        if not sessions:
            followed.pop(target, None)
            return
        try:
            cards = [card for card in self.query(ServiceCard) if card.service_def.launch_type == "collection"]
        except Exception:
            return
        for card in cards:
            shown = ((card.collection_snapshot() or {}).get("values") or {}).get("--session-id")
            session = self._host_recording_session(target, shown)
            if followed.get(target) == session:
                continue
            followed[target] = session
            if _safe_session_id(shown) != session:
                card.select_collection_session(session)
                self._log(
                    f"[cyan]{'This machine' if target == 'local' else target} is recording session "
                    f"{session}: the card shows it, so Stop and Download act on it.[/cyan]"
                )

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

        recording = self._host_recording_session(target, sticky.get("--session-id"))
        if recording:
            # this host is in the middle of a take: that is the session its card is about
            last_session = recording
            self.__dict__.setdefault("_collection_followed", {})[target] = recording
        elif "--session-id" in sticky:
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
        svc = self._service_with_endpoint_note(svc, target)
        svc = self._service_with_base_choices(svc, target)
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

    def _service_with_endpoint_note(self, svc: ServiceDef, target: str = "local") -> ServiceDef:
        """say on the card which address the status probe connects to."""
        if svc.launch_type != "make":
            return svc
        endpoint = system_service_endpoint(self._root, _make_target_for(svc.name))
        if endpoint is None:
            return svc
        host, port = endpoint
        if self._off_configured_machine(svc, target):
            node = self._node_host_cache[svc.name]
            here = "this machine" if target == "local" else f"'{target}'"
            return replace(svc, description=(
                f"{svc.description}. [yellow]System Settings point the pipelines at {host}:{port} "
                f"({node.follows}), which is not {here}: this card shows and controls {here}'s own "
                f"port {port}, the sidebar marker still follows {host}:{port}.[/yellow]"))
        where = (
            f"loopback:{port} on the selected host" if is_loopback_host(host)
            else f"{host}:{port}"
        )
        return replace(svc, description=f"{svc.description} (status probe: {where})")

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

    def _service_with_base_choices(self, svc: ServiceDef, target: str) -> ServiceDef:
        """what a base card's dropdowns offer on the card's host: its config's
        Bases entries for each Base (-b), the matrix files for the IPS
        synchronizer's Main Camera (-mc), the Base keys for the ASR
        synchronizer's base type (-bt), which follows Base 1.

        The ASR and VFA synchronizers' Sync Waits For starts on the number of
        Bases entries.

        A remote host is asked nothing from here: this also runs on the UI
        thread (Start, a config Save). Its config comes from the config cache
        and its matrix files from _transform_matrix_ids, both read when the
        card was built (_service_view_state_sync) or refreshed, off it."""
        pipeline = _BASE_CARD_PIPELINES.get(svc.name)
        if pipeline is None:
            return svc
        config_path = os.path.join(svc.config_dir, "config.yml")
        try:
            if target == "local":
                config, _ = self._load_config_for_target(config_path, show_status=False, target=target)
            else:
                config = self._target_config_cache.get(self._config_cache_key(config_path, target))
        except Exception:
            config = None  # an unreadable config offers nothing but asking in the window
        config = config if isinstance(config, dict) else {}
        bases = get_bases(config)
        base_options = _base_choices(pipeline, config)
        params = []
        for param in svc.params:
            if param.flag == "-b" and param.per_instance:
                params.append(replace(param, choices=base_options, default=[value for _, value in base_options if value]))
            elif param.flag == "-bt" and pipeline == "asr":
                types = _asr_base_types(config)
                follow = {
                    str(base.get("id")): str(base.get("base_type"))
                    for base in bases if str(base.get("base_type") or "") in types
                }
                first = str(bases[0].get("id")) if bases else ""
                default = follow.get(first) or (types[0] if len(types) == 1 else "")
                params.append(replace(param, choices=[(name, name) for name in types], default=default,
                                      follow_values=follow))
            elif param.flag == "-mc" and pipeline == "ips":
                options, default = _main_camera_choices(self._transform_matrix_ids(target) or [], config)
                params.append(replace(param, choices=options, default=default))
            elif param.flag == _SYNC_WAIT_FLAG:
                # every base of the session, wherever it runs, is a Bases
                # entry; with none listed, the card's Num Bases (None)
                params.append(replace(param, default=len(bases) or None))
            else:
                params.append(param)
        return replace(svc, params=params)

    def _transform_matrix_ids(self, target: str, refresh: bool = False) -> list[str] | None:
        """the base ids with a transformation_matrices_<id>.json in camera_sync/
        on the card's host; None when that host's folder could not be listed.
        This machine's folder is read each time; a remote one only with
        `refresh` (off the UI thread: a card being built, its Refresh), and
        what it said is kept for the Start and the redraws that follow. A
        host that does not answer a refresh keeps the list it gave before, as
        its config does, and is noted in _matrix_ids_stale."""
        if target == "local":
            return _matrix_file_ids(_local_transform_matrix_files(_ips_transform_local_dir(self._root)))
        cache: dict[str, list[str] | None] | None = getattr(self, "_matrix_ids_cache", None)
        if cache is None:
            cache = self._matrix_ids_cache = {}
        stale: set[str] | None = getattr(self, "_matrix_ids_stale", None)
        if stale is None:
            stale = self._matrix_ids_stale = set()
        if refresh:
            profile = get_profile_by_name(target)
            listing = (
                _remote_transform_matrix_listing(profile, self._ips_transform_remote_dir(profile))
                if profile is not None else None
            )
            if listing is not None:
                cache[target] = _matrix_file_ids(listing)
                stale.discard(target)
            elif cache.get(target) is not None:
                # not answering now: the Main Camera keeps the list and the pick it had
                stale.add(target)
            else:
                cache[target] = None
        return cache.get(target)

    def _matrix_folder(self, target: str) -> str:
        """camera_sync/ of the card's host, as the log names it."""
        if target == "local":
            return _ips_transform_local_dir(self._root)
        profile = get_profile_by_name(target)
        return self._ips_transform_remote_dir(profile) if profile is not None else "pipelines/ips-base/camera_sync"

    @staticmethod
    def _base_entry_ids(svc: ServiceDef) -> list[str]:
        """the Bases entries of the card host's config, as its Base dropdowns
        offer them (_service_with_base_choices)."""
        param = next((param for param in svc.params if param.flag == "-b" and param.per_instance), None)
        if param is None:
            return []
        values = []
        for choice in param.choices:
            value = choice[1] if isinstance(choice, (tuple, list)) and len(choice) == 2 else choice
            if str(value or ""):
                values.append(str(value))
        return values

    # ── ASR Base: the speakers its bases recognize ───────────────

    def _speaker_answers(self) -> dict[str, asr_speakers.Answer]:
        """what each host said last about its speaker profiles."""
        return self.__dict__.setdefault("_asr_speaker_answers", {})

    def _speaker_picks(self) -> dict[tuple[str, str], list[str] | None]:
        """the speakers ticked under Manage, per host and base (_speaker_key);
        None or absent: nobody ticked any, and the session's group decides."""
        return self.__dict__.setdefault("_asr_speaker_picks", {})

    def _list_speakers(self, target: str) -> None:
        """ask the card's host which speaker profiles it has, off the UI thread;
        the card's Speakers lines say what is known meanwhile."""
        self.call_after_refresh(self._show_speakers_summary)
        self.run_worker(self._async_list_speakers(target), group="launcher-asr-speakers", exclusive=True)

    async def _async_list_speakers(self, target: str) -> None:
        answer = await asyncio.to_thread(asr_speakers.list_profiles, asr_speakers.SpeakerHost(target, self._root))
        answers = self._speaker_answers()
        before = answers.get(target)
        if answer.listing is None and before is not None and before.listing is not None:
            # silent now: keep the profiles it named before
            self._log(f"[yellow]Speakers: {rich_escape(answer.problem)}[/yellow]")
        else:
            answers[target] = answer
        if target == self._get_panel_target():
            self._show_speakers_summary()

    def _asr_card_config(self, target: str, svc: ServiceDef | None = None) -> dict:
        """the ASR Base config of the card's host: read here for this machine,
        from the config cache for another (filled when the card was built)."""
        svc = svc or getattr(self, "_svc_map", {}).get(_ASR_BASE_CARD)
        if svc is None:
            return {}
        config_path = os.path.join(svc.config_dir, "config.yml")
        try:
            if target == "local":
                config, _ = self._load_config_for_target(config_path, show_status=False, target=target)
            else:
                config = self._target_config_cache.get(self._config_cache_key(config_path, target))
        except Exception:
            config = None
        return config if isinstance(config, dict) else {}

    @staticmethod
    def _base_ids(params: dict) -> list[str]:
        """the Bases entry each base of the card is, row by row ("" for one
        left on 'ask in its window')."""
        count = _coerce_int(params.get("-nb"), 0)
        picked = [str(value or "") for value in (params.get("-b") if isinstance(params.get("-b"), list) else [])]
        return (picked + [""] * count)[:count]

    def _base_entry_id(self, params: dict, target: str, index: int) -> str:
        """the Bases entry base `index` runs as: the row's pick, else the
        config's only entry (what a base given no -b takes); "" when it asks."""
        ids = self._base_ids(params)
        entry = ids[index] if index < len(ids) else ""
        if not entry:
            entries = get_bases(self._asr_card_config(target))
            if len(entries) == 1:
                entry = str(entries[0].get("id"))
        return entry

    def _speaker_key(self, params: dict, target: str, index: int) -> tuple[str, str]:
        """what base `index`'s pick is kept under: its Bases entry (a pick goes
        with the microphone, whichever row it is on), else its row."""
        entry = self._base_entry_id(params, target, index)
        return (target, f"id:{entry}" if entry else f"row:{index}")

    def _base_label(self, params: dict, target: str, index: int) -> str:
        """'Base 1 · voice_badge_0 (Nicla)': a base of the card, for the Manage
        screen and the log."""
        entry_id = self._base_entry_id(params, target, index)
        entry = get_base_by_id(self._asr_card_config(target), entry_id) if entry_id else None
        label = f"Base {index + 1}"
        if entry_id:
            label += f" · {entry_id}"
        if entry is not None and str(entry.get("base_type") or ""):
            label += f" ({entry.get('base_type')})"
        return label

    def _speaker_frame(self, params: dict, target: str) -> asr_speakers.Context:
        """the card's session group, its participants and the host's profiles:
        what every base's Speakers line starts from (its choice: the group's)."""
        session = params.get("-sid")
        found = asr_speakers.session_group(
            _is_new_collection_session_choice(session), session, params.get("--experiment-group"),
            self._collection_experiment_group_choices())
        participants: list[dict] = []
        if found:
            try:
                participants = list(get_participant_aliases(found[0], found[1], load_experiments(self._root)).values())
            except Exception:
                participants = []
        answer = self._speaker_answers().get(target)
        listing = answer.listing if answer else None
        return asr_speakers.Context(
            group="/".join(found) if found else "",
            participants=participants,
            listing=listing,
            problem=(answer.problem if answer and listing is None else ""),
            choice=asr_speakers.choose(None, listing, participants),
        )

    def _speaker_context(self, params: dict, target: str, index: int,
                         frame: asr_speakers.Context | None = None) -> asr_speakers.Context:
        """what a Start would pass base `index`: its own ticks, else the frame's group choice."""
        frame = frame or self._speaker_frame(params, target)
        picked = self._speaker_picks().get(self._speaker_key(params, target, index))
        return replace(frame, choice=asr_speakers.choose(picked, frame.listing, frame.participants))

    def _base_verifies(self, params: dict, target: str, index: int, svc: ServiceDef | None = None) -> bool | None:
        """whether base `index` recognizes speakers (its base type is asr_scope
        individual); None when the host's config is not known."""
        config = self._asr_card_config(target, svc)
        if not config:
            return None
        ids = self._base_ids(params)
        return asr_speakers.verifies_speakers(config, ids[index] if index < len(ids) else "")

    def _speakers_unused(self, params: dict, target: str, index: int) -> str:
        """why base `index` would recognize nobody; "" when it would."""
        if str(params.get("-m") or "live") == "capture":
            return "capture mode records only"
        if self._base_verifies(params, target, index) is False:
            return "this base is asr_scope group"
        return ""

    def _show_speakers_summary(self) -> None:
        """write the Speakers line of every base of the ASR Base card."""
        card = next((card for card in self.query(ServiceCard) if card.service_def.name == _ASR_BASE_CARD), None)
        if card is None:
            return
        target = self._get_panel_target()
        params = card.collect_params()
        where = "this machine" if target == "local" else target
        frame = self._speaker_frame(params, target)
        for index in range(_coerce_int(params.get("-nb"), 0)):
            context = self._speaker_context(params, target, index, frame)
            card.show_speakers(index, asr_speakers.summary(context, where, self._speakers_unused(params, target, index)))

    def on_service_card_speakers_requested(self, event: ServiceCard.SpeakersRequested) -> None:
        """Manage on a Speakers row of the ASR Base card: the speaker profiles
        of its host, and which of them that base recognizes."""
        svc = self._svc_map.get(event.service_name)
        if svc is None:
            return
        target = self._get_panel_target()
        index = event.index
        frame = self._speaker_frame(event.params, target)
        shown = self._service_with_base_choices(svc, target)
        base_param = next((param for param in shown.params if param.flag == "-b" and param.per_instance), None)
        bases = [
            (str(choice[0]), str(choice[1])) for choice in (base_param.choices if base_param else [])
            if isinstance(choice, (tuple, list)) and len(choice) == 2 and str(choice[1] or "")
        ]
        key = self._speaker_key(event.params, target, index)

        def done(result) -> None:
            if not isinstance(result, dict):
                return
            self._speaker_picks()[key] = result.get("picked")
            if isinstance(result.get("answer"), asr_speakers.Answer):
                self._speaker_answers()[target] = result["answer"]
            if result.get("start_dir"):
                self.__dict__["_asr_speaker_files_dir"] = result["start_dir"]
            self._show_speakers_summary()

        self.app.push_screen(SpeakerProfilesScreen(
            asr_speakers.SpeakerHost(target, self._root),
            answer=self._speaker_answers().get(target),
            picked=self._speaker_picks().get(key),
            participants=frame.participants,
            group=frame.group,
            bases=bases,
            base=self._base_entry_id(event.params, target, index),
            base_label=self._base_label(event.params, target, index),
            vad=bool(event.params.get("-vad", True)),
            nr=bool(event.params.get("-nr", True)),
            store=bool(event.params.get("-s", True)),
            start_dir=self.__dict__.get("_asr_speaker_files_dir"),
        ), done)

    def _put_speakers(self, params: dict, target: str) -> None:
        """the -spk a Start of the ASR Base card passes each of its bases (a
        list, one entry per base): the speakers ticked for it under Manage,
        else the participants of the session's group that have a profile on
        the host; none ("": the base takes every profile) when neither names
        anyone.

        This machine's profiles are read again first (one registered from a
        base's own menu since counts). Another host's are what it said last;
        one that never answered `mmla asr-speakers` gets no -spk, since a
        checkout from before it has a `mmla asr-base` that would refuse it."""
        count = _coerce_int(params.get("-nb"), 0)
        if target == "local":
            self._speaker_answers()["local"] = asr_speakers.list_profiles(asr_speakers.SpeakerHost("local", self._root))
        answer = self._speaker_answers().get(target)
        if target != "local" and (answer is None or answer.listing is None):
            params["--speakers"] = [""] * count
            return
        frame = self._speaker_frame(params, target)
        speakers = []
        for index in range(count):
            choice = self._speaker_context(params, target, index, frame).choice
            speakers.append(join_speakers(choice.names) if choice.names else "")
        params["--speakers"] = speakers

    def _speakers_start_problem(self, params: dict, target: str, svc: ServiceDef | None = None) -> str:
        """why a base of the ASR Base card would stop at its menu for want of
        speakers (asr_scope individual in live or analyze mode); "" when none would."""
        mode = str(params.get("-m") or "live")
        count = _coerce_int(params.get("-nb"), 0)
        if count <= 0 or mode == "capture":
            return ""
        config = self._asr_card_config(target, svc)
        if not config:
            return ""
        frame = self._speaker_frame(params, target)
        listing = frame.listing
        where = "this machine" if target == "local" else f"'{target}'"
        ids = self._base_ids(params)
        for index in range(count):
            if not asr_speakers.verifies_speakers(config, ids[index]):
                continue
            base = f"base {index + 1}" + (f" ({ids[index]})" if ids[index] else "")
            choice = self._speaker_context(params, target, index, frame).choice
            if choice.names == []:
                return (f"[yellow]No speaker is ticked under Speakers {index + 1} → Manage, and {base} recognizes "
                        f"speakers (asr_scope: individual) in {mode} mode. Tick them there, or press Use Group there "
                        f"to let the session's group decide, or switch Mode to capture.[/yellow]")
            if listing is None:
                return ""   # the host did not say: the bases decide, and ask in their window if they must
            if not listing.names:
                return (f"[yellow]No speaker profile is registered on {where}, and {base} recognizes speakers "
                        f"(asr_scope: individual) in {mode} mode. Register them under Speakers → Manage; or switch "
                        f"Mode to capture, or set asr_scope: group for its base type on the Config tab.[/yellow]")
            if choice.names and not any(name in listing.names for name in choice.names):
                return (f"[yellow]None of the speakers ticked under Speakers {index + 1} → Manage "
                        f"({rich_escape(', '.join(choice.names))}) is registered on {where}. Tick others there, "
                        f"or register them.[/yellow]")
        return ""

    def _speakers_start_notes(self, params: dict, target: str) -> list[str]:
        """what a Start of the ASR Base card says about the speakers it passes each base."""
        count = _coerce_int(params.get("-nb"), 0)
        if count <= 0 or str(params.get("-m") or "live") == "capture":
            return []
        frame = self._speaker_frame(params, target)
        listing = frame.listing
        where = "this machine" if target == "local" else target
        if listing is None and target != "local":
            answer = self._speaker_answers().get(target)
            why = f" ({answer.problem})" if answer and answer.problem else ""
            return [f"  [yellow]Speakers: {rich_escape(where)} did not say which profiles it has{rich_escape(why)}, so "
                    f"no -spk is passed and each base takes every profile there.[/yellow]"]
        notes = []
        followed_group = False
        for index in range(count):
            if self._base_verifies(params, target, index) is False:
                continue
            choice = self._speaker_context(params, target, index, frame).choice
            label = self._base_label(params, target, index)
            if choice.names:
                why = {"picked": "ticked under Manage",
                       "group": f"the participants of {frame.group}"}.get(choice.how, "")
                notes.append(f"  {rich_escape(label)} speakers: {rich_escape(', '.join(choice.names))}"
                             + (f" ({rich_escape(why)})" if why else ""))
                followed_group = followed_group or choice.how == "group"
                if listing is not None:
                    missing = [name for name in choice.names if name not in listing.names]
                    if missing:
                        notes.append(f"  [yellow]Not registered on {rich_escape(where)}, so not recognized: "
                                     f"{rich_escape(', '.join(missing))}[/yellow]")
            else:
                notes.append(f"  {rich_escape(label)} speakers: every profile on {rich_escape(where)}")
        if listing is not None and followed_group:
            absent = asr_speakers.unregistered(listing.names, frame.participants)
            if absent:
                notes.append(f"  [yellow]Participants of {rich_escape(frame.group)} with no profile on "
                             f"{rich_escape(where)}: {rich_escape(', '.join(absent))} (Speakers → Manage "
                             f"registers them)[/yellow]")
        return notes

    def _base_card_start_notes(self, svc: ServiceDef, params: dict) -> list[str]:
        """what a Start of the ASR or VFA base card should say about the
        bases it does not start: those the config lists and the synchronizer
        waits for have to run on other hosts, in this session."""
        if svc.name not in _SYNC_WAIT_CARDS:
            return []
        num_bases = _coerce_int(params.get("-nb"), 0)
        syncs = _coerce_int(params.get("-ns"), 0)
        if num_bases <= 0 and syncs <= 0:
            return []
        listed = len(self._base_entry_ids(svc))
        waits = (_coerce_int(params.get(_SYNC_WAIT_FLAG), 0) or listed) if syncs > 0 else 0
        notes = []
        if listed > num_bases:
            notes.append(
                f"  The config lists {listed} Bases entries and this card starts {num_bases} base(s): the other "
                f"{listed - num_bases} must run elsewhere (the base card of another host, in this session)."
            )
        if waits > num_bases:
            notes.append(
                f"  The synchronizer waits for {waits} bases (Sync Waits For) and this card starts {num_bases}: "
                f"the other {waits - num_bases} have to report from other hosts, or each time slice is merged "
                f"only when it expires, late and without them."
            )
        return notes

    def _base_card_start_problem(self, svc: ServiceDef, params: dict, target: str) -> str:
        """why a base card cannot start as it stands, said plainly; "" when it
        can. Every process the card opens takes its choices from the card and
        asks nothing, so what it would have asked must be answerable here."""
        if svc.name not in _BASE_CARD_PIPELINES:
            return ""
        num_bases = _coerce_int(params.get("-nb"), 0)
        picked = [str(value or "") for value in (params.get("-b") if isinstance(params.get("-b"), list) else [])]
        picked = (picked + [""] * num_bases)[:num_bases]
        entries = self._base_entry_ids(svc)
        # a base given no -b takes the config's only Bases entry, when it has
        # exactly one (it asks in its window only when there are several)
        only = entries[0] if len(entries) == 1 else ""
        seen: dict[str, int] = {}
        for index, value in enumerate(picked):
            entry = value or only
            if not entry:
                continue
            if entry in seen:
                first = seen[entry]
                if picked[first] and value:
                    return (
                        f"[yellow]Base {first + 1} and Base {index + 1} are both Bases entry {entry}: each base "
                        f"needs an entry of its own. Pick another one for either, or lower Num Bases.[/yellow]"
                    )
                return (
                    f"[yellow]Base {first + 1} and Base {index + 1} would both be Bases entry {entry}: the config "
                    f"lists that one entry only, and a base left on '{_ASK_BASE_LABEL}' takes the only entry there "
                    f"is. Add a Bases entry for the other base on the Config tab and Save, or lower Num Bases to "
                    f"1.[/yellow]"
                )
            seen[entry] = index
        if svc.name in _SYNC_WAIT_CARDS and _coerce_int(params.get("-ns"), 0) > 0:
            waits = _coerce_int(params.get(_SYNC_WAIT_FLAG), 0)
            counted = waits or len(entries)
            if not counted:
                return (
                    "[yellow]Sync Waits For is 0, so the synchronizer would count the Bases entries of its config, "
                    "and the config lists none. Set Sync Waits For to the number of bases in this session, on "
                    "every host, or set Num Synchronizers to 0.[/yellow]"
                )
            if num_bases > counted:
                why = "" if waits else " (the Bases entries of its config, as Sync Waits For is 0)"
                return (
                    f"[yellow]This card starts {num_bases} bases, but the synchronizer waits for {counted}{why}: it "
                    f"would merge each time slice as soon as {counted} of them report, before the others do. Set "
                    f"Sync Waits For to {num_bases} or more (with the bases of this session on other hosts), or "
                    f"lower Num Bases.[/yellow]"
                )
        if svc.name == _ASR_BASE_CARD:
            problem = self._speakers_start_problem(params, target, svc)
            if problem:
                return problem
        if svc.name != "IPS Base" or _coerce_int(params.get("-ns"), 0) <= 0 or str(params.get("-mc") or "").strip():
            return ""
        where = "this machine" if target == "local" else f"'{target}'"
        folder = self._matrix_folder(target)
        ids = self._transform_matrix_ids(target)
        if ids is None:
            return (
                f"[red]The IPS synchronizer has no Main Camera: the matrix files in {folder} on {where} could not "
                f"be listed. Press Refresh on this card once {where} answers.[/red]"
            )
        if not ids:
            fetch = (
                "" if target == "local"
                else f", then copy them to {where} with Sync to Host on the Transform Matrix tab (Host: Local)"
            )
            return (
                f"[red]No transformation_matrices_<id>.json in {folder} on {where}: the IPS synchronizer takes its "
                f"Main Camera from one, and every base needs its own file there too. Make them with IPS Camera "
                f"Sync{fetch}, and press Refresh on this card. To start the bases alone, set Num Synchronizers "
                f"to 0.[/red]"
            )
        return (
            "[yellow]Pick the synchronizer's Main Camera first (press Refresh on this card when its list "
            "is out of date), or set Num Synchronizers to 0.[/yellow]"
        )

    def _refresh_base_card_choices(self) -> None:
        """a mounted base card's dropdowns follow a config just saved; they
        read it from the config cache, which the save has filled."""
        target = self._get_panel_target()
        for card in self.query(ServiceCard):
            service = self._svc_map.get(card.service_def.name)
            if service is None or service.name not in _BASE_CARD_PIPELINES:
                continue
            fresh = self._service_with_base_choices(service, target)
            card.update_param_choices(host_params(fresh.params))
        self._show_speakers_summary()

    async def _async_refresh_base_choices(self, svc: ServiceDef, target: str) -> None:
        """Refresh on a base card: read its host's config and matrix files
        again, off the UI thread, and give its dropdowns the fresh options."""
        config_path = os.path.join(svc.config_dir, "config.yml")

        def fetch() -> None:
            key = self._config_cache_key(config_path, target)
            old = self._target_config_cache.pop(key, None)
            self._load_config_for_target(config_path, show_status=False, target=target)
            if key not in self._target_config_cache and old is not None:
                self._target_config_cache[key] = old  # not readable now: keep what it said last
            if svc.name == "IPS Base":
                self._transform_matrix_ids(target, refresh=True)

        await asyncio.to_thread(fetch)
        if svc.name == "IPS Base" and target in getattr(self, "_matrix_ids_stale", set()):
            self._log(
                f"[yellow]The matrix files in {self._matrix_folder(target)} on '{target}' could not be listed "
                f"just now: the Main Camera list is the one read before, and may be out of date. Press Refresh "
                f"again once '{target}' answers.[/yellow]"
            )
        if self._get_panel_target() == target:
            self._refresh_base_card_choices()

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
            ssh_profiles=self._ssh_profile_names,
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
            ssh_profiles=self._ssh_profile_names,
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

    def _settings_host(self, section_name: str) -> str:
        """whose settings a System Settings form shows: the Connections forms
        follow their Host selector, everything else is this machine's."""
        if f"__shared__{section_name}" in _LOCAL_SETTINGS_NOTES:
            return "local"
        target = self._settings_target
        if target != "local" and (
            TARGET_STATES.get(target) == "offline"
            or not any(value == target for _, value in target_options())
        ):
            target = self._settings_target = "local"
        return target

    def _settings_values_for(self, target: str) -> tuple[dict[str, object], str]:
        """(flat shared values, where they come from) of one machine. A remote
        machine is read over ssh, so this runs off the UI thread."""
        if target == "local":
            return self._shared_values, ""
        configs = []
        for rel_path in SYSTEM_SERVICE_SOURCE_CONFIG_RELS:
            config, _ = self._load_config_for_target(
                os.path.join(self._root, rel_path), show_status=False, target=target)
            if config:
                configs.append(config)
        values = harvest_system_services_from_configs(configs)
        store, error = self._load_config_for_target(
            self._remote_settings_path(), show_status=False, target=target)
        if isinstance(store, dict) and store:
            # what its services read at startup; the pipeline configs only fill
            # in the sections that file does not have
            values.update(config_to_flat_values(store, include_defaults=False))
            return values, f"'{target}' has System Settings of its own (config/system_services.yml): its services use these."
        if error and "No remote config" not in error:
            return values, f"[red]{rich_escape(error)}[/red]"
        if configs:
            return values, (
                f"'{target}' has no config/system_services.yml: these values are from its pipeline "
                f"configs, which a Start from this console keeps in step with Local. Save gives it "
                f"settings of its own."
            )
        return values, f"'{target}' has neither settings nor pipeline configs yet: these are the defaults."

    async def _mount_shared_form(self, container: Vertical, section_name: str) -> None:
        target = self._settings_host(section_name)
        if f"__shared__{section_name}" not in _LOCAL_SETTINGS_NOTES:
            self._show_host_bar(None)
            self._point_host_select(target)
        values, origin = self._shared_values, ""
        if target != "local":
            loading = Static(f" Reading the settings of '{target}' ...", classes="status-saved")
            await container.mount(loading)
            values, origin = await asyncio.to_thread(self._settings_values_for, target)
            if not container.is_attached or self._current_shared_section != section_name:
                return  # the user moved on while ssh was at work
            await loading.remove()
        if origin:
            await container.mount(Static(origin, classes="settings-origin"))
        self._show_shared_form(container, section_name, values)

    async def _reload_shared_form(self) -> None:
        """the Host selector moved while a Connections form is open."""
        section_name = self._current_shared_section
        container = self._config_container
        if not section_name or container is None or not container.is_attached:
            return
        await container.remove_children()
        self._current_form = None
        await self._mount_shared_form(container, section_name)

    def _show_shared_form(self, container: Vertical, section_name: str,
                          values: dict[str, object] | None = None) -> None:
        shared_values = self._shared_values if values is None else values
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
        values = {f.path: shared_values.get(f.path, f.default) for f in fields}
        form = ConfigForm(f"shared:{section_name}", fields, values)
        container.mount(form)
        self._current_form = form
        self._show_sync_bar(shared_section=section_name)

    def _show_pipeline_form(self, container: Vertical, pipeline: PipelineDef) -> None:
        existing, source_message = self._load_config_for_target(pipeline.config_path)
        apply_shared_values(pipeline.fields, self._shared_values)
        # what only one source uses, and an older config keeps in Base (a udp/tcp
        # base type's host and packet_format, a file_dir), is shown in the Bases
        # entries that use it; the next Save writes it there (move_source_settings)
        if isinstance(existing, dict):
            existing = copy.deepcopy(existing)
            moved = move_source_settings(existing)
            noted = (pipeline.name, self._get_panel_target())
            if moved and noted not in self._source_moves_noted:
                self._source_moves_noted.add(noted)
                self._log(
                    f"[yellow]{rich_escape(pipeline.name)}: Base holds settings only one source uses; the next "
                    f"Save moves them into the Bases entries that use them, and drops the others:[/yellow]\n"
                    + "\n".join(f"  {rich_escape(line)}" for line in moved))

        # populate Bases entry dropdowns from the config (camera <- Cameras,
        # base_type <- Base) so users pick existing values instead of typing
        if isinstance(existing, dict):
            cameras = sorted(
                name for name, entry in (existing.get("Cameras") or {}).items()
                if not _is_template_camera(entry))
            base_types = sorted((existing.get("Base") or {}).keys())
            # 'rtmp' is not offered: it is the old name of 'stream' (an entry
            # that still says it is shown as stream)
            source_types = {
                "ASR Base": ["udp", "tcp", "pyaudio", "stream", "lsl", "file"],
                "IPS Base": ["opencv", "stream", "lsl", "file"],
                "VFA Base": ["opencv", "stream", "lsl", "file"],
            }.get(pipeline.name, [])
            for f in pipeline.fields:
                if f.field_type == "list_of_dicts" and f.path == "Bases":
                    choices = {}
                    if "camera" in (f.entry_schema or {}):
                        choices["camera"] = cameras
                    if "base_type" in (f.entry_schema or {}):
                        choices["base_type"] = base_types
                    if "source" in (f.entry_schema or {}) and source_types:
                        choices["source"] = source_types
                    if "packet_format" in (f.entry_schema or {}):
                        choices["packet_format"] = ["auto", "timestamped", "raw"]
                    if "camera_angle" in (f.entry_schema or {}):
                        # the viewing angles of Base.angle_config, named there
                        # with what a camera at that angle sees
                        choices["camera_angle"] = [
                            str(angle) for angle in ((existing.get("Base") or {}).get("angle_config") or {})
                            if not PLACEHOLDER_RE.match(str(angle))]
                    if "source_index" in (f.entry_schema or {}):
                        # a file entry's folder is listed, and Browse… opens, where
                        # its files are this machine's
                        choices["source_index:file_here"] = self._get_panel_target() == "local"
                        # the streams a 'stream' base can pull, in the order
                        # its source_index counts them
                        choices["source_index:stream"] = get_stream_sources(existing)
                        # the devices of the card's host, as it said last (a
                        # text box until it has: _probe_form_devices asks)
                        for kind in self._device_kinds(pipeline.name):
                            choices[f"source_index:{kind}"] = self._device_options(self._get_panel_target(), kind)
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
            # the local System Settings store used as a gap-filler ("shared"),
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
                    sources[f.path] = "[yellow]· managed in System Settings[/yellow]"
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

        # the cameras are the config's, one group each: the calibrator writes
        # them, Calibration Cameras syncs them, + Add Camera adds one by hand
        has_cameras = "Cameras" in pipeline.sections
        known_sections = {f.path.split(".")[0] for f in pipeline.fields}
        if has_cameras:
            known_sections.add("Cameras")
        if pipeline.base_section:
            known_sections.add(pipeline.base_section)
        if pipeline.name in _STREAM_PIPELINES:
            # drawn below as a group of its own, one entry per stream
            known_sections.add("Streams")
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
                    s_fields = _make_stream_fields(
                        stream_name, self._stream_server_address(), _card_stream_kind(pipeline.name))
                    dynamic_sections[section_name] = s_fields
                    for f in s_fields:
                        val = get_nested_value(existing, f.path)
                        if val is not None:
                            values[f.path] = val
        leftover_cameras: list[str] = []
        if has_cameras and isinstance(existing.get("Cameras"), dict):
            for camera, entry in existing["Cameras"].items():
                if _is_template_camera(entry):
                    # the template's example: not a camera, gone at the next Save
                    leftover_cameras.append(f"Cameras.{camera}")
                    continue
                if not isinstance(entry, dict):
                    continue
                c_fields = _make_camera_fields(str(camera))
                dynamic_sections[f"Cameras.{camera}"] = c_fields
                for f in c_fields:
                    val = get_nested_value(existing, f.path)
                    if val is not None:
                        values[f.path] = val
        # the template's empty `Streams:` key is not a field of the form: the
        # streams are the group above. Filtered for this form only; taking it out
        # of pipeline.fields for good made the section unknown the next time the
        # tab was opened, and the config's Streams showed up a second time as a
        # plain section
        form_fields = [
            f for f in pipeline.fields
            if not (pipeline.name in _STREAM_PIPELINES and f.path.startswith("Streams"))
            and not (has_cameras and f.path.startswith("Cameras."))
        ]

        group_add_buttons = {}
        if pipeline.base_template and pipeline.base_section:
            group_add_buttons[pipeline.base_section] = ("+ Add Base", "btn-add-base")
        if pipeline.name in _STREAM_PIPELINES:
            group_add_buttons["Streams"] = ("+ Add Stream", "btn-add-stream")
        if has_cameras:
            group_add_buttons["Cameras"] = ("+ Add Camera", "btn-add-camera")

        section_titles, section_notes = self._pipeline_section_help(
            form_fields, values, self._stream_server_address())
        if has_cameras:
            section_notes["Cameras"] = (
                "The cameras of this config: written by IPS Camera Calibration (Calibrate), synced from "
                "another machine (Calibration Cameras, Sync to Host), or added here with + Add Camera.")
        form = ConfigForm(pipeline.name, form_fields, values, dynamic_sections,
                          group_add_buttons=group_add_buttons,
                          base_section=pipeline.base_section or None,
                          sources=sources,
                          readonly_paths=readonly_paths,
                          shared_sections=set(SHARED_SECTION_NAMES),
                          overridden_sections=overrides,
                          allow_override_toggle=True,
                          section_titles=section_titles,
                          section_notes=section_notes,
                          entry_factories={"Cameras": _make_camera_fields} if has_cameras else None)
        for section_name in leftover_cameras:
            form.mark_removed(section_name)
        container.mount(form)
        self._current_form = form
        if source_message:
            self._show_status(source_message)
        self._show_sync_bar(pipeline)
        if any(f.path == "Bases" and (f.entry_field_choices or {}).get("source_index:stream") for f in form_fields):
            self.run_worker(self._mark_live_streams(form), group="stream-live-marks", exclusive=True)
        if isinstance(existing, dict):
            self.run_worker(self._probe_form_devices(form, pipeline, existing), group="config-devices",
                            exclusive=True)

    # ── the capture devices of the hosts a form names ────────────

    def _device_answers(self) -> dict[str, capture_devices.Devices]:
        """what each host said last about its devices (capture_devices), kept
        until Refresh on a base card."""
        return self.__dict__.setdefault("_capture_device_answers", {})

    def _device_options(self, host: str, kind: str) -> list[tuple[str, str]]:
        answer = self._device_answers().get(host)
        return answer.options(kind) if answer is not None else []

    @staticmethod
    def _device_kinds(pipeline_name: str) -> set[str]:
        """the device a Bases entry's source_index names on this card: pyaudio's
        input devices on ASR Base, opencv's cameras on IPS and VFA Base."""
        if pipeline_name == "ASR Base":
            return {"pyaudio"}
        if pipeline_name in ("IPS Base", "VFA Base"):
            return {"opencv"}
        return set()

    def _form_device_hosts(self, pipeline: PipelineDef, existing: dict, target: str) -> dict[str, set[str]]:
        """which hosts a form's dropdowns ask, and for what: the card's host for
        the Bases entries. A stream's device is picked on the Streams tab, which
        asks its capture host itself."""
        kinds = self._device_kinds(pipeline.name)
        if kinds and any(f.path == "Bases" for f in pipeline.fields):
            return {target: set(kinds)}
        return {}

    async def _probe_form_devices(self, form, pipeline: PipelineDef, existing: dict, force: bool = False) -> None:
        """ask the hosts a form names for the devices it has no answer from yet
        (every one of them with `force`), off the UI thread, and give the
        form's fields the dropdowns as each answers."""
        target = self._get_panel_target()
        answers = self._device_answers()
        for host, kinds in self._form_device_hosts(pipeline, existing, target).items():
            known = answers.get(host)
            if known is not None and not force:
                kinds = kinds - set(known.found) - set(known.problems)
            if not kinds:
                continue
            answer = await asyncio.to_thread(capture_devices.list_devices, host, kinds, self._root)
            if known is not None and not force:
                known.found.update(answer.found)
                known.problems.update(answer.problems)
                known.platform = known.platform or answer.platform
            else:
                answers[host] = answer
            # a host that answers at once (this machine) beats the form's mounting:
            # its fields take the answer once they are there
            for _ in range(60):
                if self._current_form is not form:
                    break
                if form.is_attached and (form.query(FieldRow) or form.query(DictListField)):
                    break
                await asyncio.sleep(0.05)
            if not form.is_attached or self._current_form is not form:
                return  # another card, or another tab of it, took the form down meanwhile
            self._apply_device_choices(form, pipeline, existing, host)

    def _apply_device_choices(self, form, pipeline: PipelineDef, existing: dict, host: str) -> None:
        """put what `host` said into the form: the Bases dropdowns when it is
        the card's host."""
        answer = self._device_answers().get(host)
        if answer is None:
            return
        where = "this machine" if host == "local" else host
        if host == self._get_panel_target():
            for kind in self._device_kinds(pipeline.name):
                for field in form.query(DictListField):
                    if field.field_def.path == "Bases":
                        field.set_source_choices(kind, answer.options(kind), answer.note(kind, where))

    def _reprobe_form_devices(self) -> None:
        """Refresh on a base card: forget what the hosts said about their
        devices, and ask again for the form on screen, if one is."""
        self._device_answers().clear()
        form, pipeline = self._current_form, self._current_pipeline
        if form is None or pipeline is None or not form.is_attached:
            return
        existing, _ = self._load_config_for_target(pipeline.config_path, show_status=False)
        if isinstance(existing, dict):
            self.run_worker(self._probe_form_devices(form, pipeline, existing, force=True), group="config-devices",
                            exclusive=True)

    def _remark_live_streams(self) -> None:
        """ask the Stream Server again which streams of the Bases form on
        screen are live, as when the form was built."""
        form = self._current_form
        if form is None or not form.is_attached:
            return
        if any(field.field_def.path == "Bases" and field.stream_choices for field in form.query(DictListField)):
            self.run_worker(self._mark_live_streams(form), group="stream-live-marks", exclusive=True)

    async def _mark_live_streams(self, form) -> None:
        """which streams of a Bases form the Stream Server has live now, asked
        off the UI thread once the form is up: the stream dropdowns mark them
        (● live, ○ not publishing now), and the paths it has live that no
        Streams entry of this config names are listed below them."""
        server = self._stream_server_address()
        host = str(server.get("host") or "").strip()
        if not host:
            return
        try:
            live = await asyncio.to_thread(
                recordings.live_paths, host, int(server.get("api_port") or recordings.API_PORT))
        except recordings.RecordingsError:
            live = None
        try:
            fields = [f for f in form.query(DictListField) if f.field_def.path == "Bases"]
        except Exception:
            return  # the form was replaced meanwhile
        for field in fields:
            pullable = field.stream_choices
            paths = await asyncio.to_thread(
                lambda: {name: stream_server_path(url, server) for name, url in pullable})
            if live is None:
                field.set_stream_states(
                    {}, f"The Stream Server ({host}) does not answer, so whether these streams are live "
                        f"now is not known.")
                continue
            states = {name: ("live" if path in live else "idle") for name, path in paths.items() if path}
            # the server is shared: only paths under the apps of this config's streams
            apps = {path.split("/", 1)[0] for path in paths.values() if path}
            elsewhere = sorted(path for path in live - set(paths.values()) if path.split("/", 1)[0] in apps)
            field.set_stream_states(states, (
                f"Live on the Stream Server but in no Streams entry here: {', '.join(elsewhere)}. "
                f"Add it under Streams to pull it." if elsewhere else ""))

    @staticmethod
    def _pipeline_section_help(
        fields: list[LoaderFieldDef], values: dict, stream_server: dict | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """what the sections of a pipeline config are called in the form, and a
        note on the ones that only make sense together: Gateway, Server and
        Streams. Also says, under each Server entry, where its value leads."""
        publish, pull = stream_server_urls(stream_server or {"host": "<stream-server>"}, "ips/cam-1")
        titles = {
            name: str(info.get("label") or name)
            for name, info in SHARED_SECTIONS.items() if info.get("label") and info["label"] != name
        }
        titles["Server"] = "Server  (the AI services this base calls)"
        gateway = _gateway_base_url(values)
        notes = {
            "Gateway": (
                "The Nginx load balancer in front of the AI services. Whether a request goes through "
                "it is decided per service, in the Server section."
            ),
            "Server": _server_section_note(gateway),
            "Streams": (
                "One entry per camera or microphone stream. target is where the capture device "
                "publishes, read_target what the bases pull. Write the path alone (ips/cam-1) and Save "
                f"completes both with the Stream Server of System Settings: {publish} and {pull}. "
                "A full URL is kept as written. record: true also records on the capture device; the "
                "Stream Server records on its side whatever reaches it (its card, Config tab). The "
                "machine that captures a stream is picked on the Streams tab, in its SSH Profile column."
            ),
        }
        for field in fields:
            if _is_server_entry(field):
                field.description = _server_entry_hint(values.get(field.path), gateway)
        return titles, notes

    @on(ConfigForm.FieldEdited)
    def _on_config_form_field_edited(self, event: ConfigForm.FieldEdited) -> None:
        """the line under a Server entry says where its value leads: it follows
        the value as it is typed, and every entry follows the Gateway."""
        form = self._current_form
        if form is None or event.path.split(".", 1)[0] not in ("Server", "Gateway"):
            return
        values = form.collect_values()
        values[event.path] = event.value
        gateway = _gateway_base_url(values)
        if event.path.startswith("Server."):
            form.set_field_description(event.path, _server_entry_hint(event.value, gateway))
            return
        form.set_section_note("Server", _server_section_note(gateway))
        for field in form.all_fields:
            if _is_server_entry(field):
                form.set_field_description(field.path, _server_entry_hint(values.get(field.path), gateway))

    def _write_stream_entry(self, stream_name: str, key: str, value) -> dict | None:
        """set one key of a stream from the Streams tab (Record, SSH Profile)
        in the config of the host the card is on, and show it there.
        Returns the config written, None when nothing was."""
        return self._write_stream_entries({stream_name: {key: value}})

    def _write_stream_entries(self, changes: dict[str, dict]) -> dict | None:
        """the same for several streams in one write (Manage's keep time)."""
        pipeline = self._current_pipeline
        if pipeline is None:
            return None
        target = self._get_panel_target()
        config, _ = self._load_config_for_target(pipeline.config_path, show_status=False, target=target)
        config = copy.deepcopy(config) if isinstance(config, dict) else {}
        streams = config.get("Streams") or {}
        missing = [f"'{name}'" for name in changes if not isinstance(streams.get(name), dict)]
        if missing:
            which = f"Stream {missing[0]} is" if len(missing) == 1 else f"Streams {', '.join(missing)} are"
            self._log(f"[red]{which} not in the config of {target}; Save the Config tab first.[/red]")
            return None
        for name, values in changes.items():
            streams[name].update(values)
        cache_key = self._config_cache_key(pipeline.config_path, target)
        if target == "local":
            with open(pipeline.config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
            self._target_config_cache[cache_key] = config
        else:
            profile = get_profile_by_name(target)
            if profile is None:
                self._log(f"[red]SSH profile '{target}' not found; not changed.[/red]")
                return None
            tmp = tempfile.NamedTemporaryFile(
                "w", suffix=".yml", prefix="openmmla-stream-", delete=False, encoding="utf-8")
            with tmp:
                yaml.safe_dump(config, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
            self._target_config_cache[cache_key] = config
            # one group for both: each copy carries the whole file, and the last one must win
            self.run_worker(
                self._run_scp(target, tmp.name, self._remote_config_path(pipeline.config_path, profile),
                              cleanup_local=True, cache_key=cache_key, cache_config=config),
                group="stream-entry-scp", exclusive=True,
            )
        for panel in self.query(StreamPanel):
            panel.update_streams(streams_from_config(config))
        return config

    def on_stream_panel_record_toggle_requested(self, event: StreamPanel.RecordToggleRequested) -> None:
        """the Record column of the Streams tab: write the stream's `record` into
        the config of the host the card is on, then show it in both tabs."""
        event.stop()
        if self._write_stream_entry(event.stream_name, "record", bool(event.record)) is None:
            return
        state = "on" if event.record else "off"
        self._log(
            f"[green]{event.stream_name}: recording on the capture device is {state} "
            f"(takes effect at its next Start).[/green]"
        )
        self._reshow_config_form()

    def _reshow_config_form(self) -> None:
        """the Config tab holds the old value of what the Streams tab just
        wrote: a Save from there would undo it, so it is drawn again."""
        container = self._config_container
        if container is not None and container.is_attached:
            container.remove_children()
            self._current_form = None
            self.call_after_refresh(self._show_pipeline_form, container, self._current_pipeline)

    def on_stream_panel_keep_days_change_requested(self, event: StreamPanel.KeepDaysChangeRequested) -> None:
        """Keep recordings for, in Manage on the Streams tab: `record_keep_days`
        of every stream there, written into the config of the host the card is on."""
        event.stop()
        if not event.keep_days:
            return
        changes = {name: {"record_keep_days": max(int(days), 0)} for name, days in event.keep_days.items()}
        if self._write_stream_entries(changes) is None:
            return
        days = max(int(value) for value in event.keep_days.values())
        self._log(
            f"[green]{', '.join(event.keep_days)}: kept on the capture host until deleted.[/green]" if not days else
            f"[green]{', '.join(event.keep_days)}: recordings on the capture host are kept {days} day(s); "
            f"older ones are deleted at a stream's Start and at Refresh on the Streams tab.[/green]"
        )
        self._reshow_config_form()

    def on_stream_panel_ssh_profile_change_requested(self, event: StreamPanel.SshProfileChangeRequested) -> None:
        """SSH Profile in the Streams tab: the machine that runs the stream's
        ffmpeg, written into the config of the host the card is on. The Config
        tab does not show it, so its form stays as it is, unsaved edits and all.
        The device picked on the machine before is none of this one's: it goes."""
        event.stop()
        changes: dict[str, object] = {"ssh_profile": event.ssh_profile}
        before = self._stream_entry(event.stream_name)
        dropped = str(before.get("device") or "").strip()
        if dropped and str(before.get("ssh_profile") or "") != event.ssh_profile:
            changes["device"] = ""
        else:
            dropped = ""
        config = self._write_stream_entries({event.stream_name: changes})
        if config is None:
            return
        profile = event.ssh_profile
        if not profile:
            self._log(f"[green]{event.stream_name} is external now: someone else publishes it, the bases pull it.[/green]")
            return
        self._log(
            f"[green]{event.stream_name}: Start runs its ffmpeg on "
            f"{'this machine' if profile == 'local' else profile} now.[/green]"
            + (f" [yellow]Its device ({rich_escape(dropped)}) was the other machine's: pick one in the Device "
               f"column, or Start takes the first.[/yellow]" if dropped else "")
        )
        target = str(config["Streams"][event.stream_name].get("target") or "")
        if profile != "local" and "://" in target and is_loopback_host(urlsplit(target).hostname):
            # a pick here does not pass through Save, which keeps localhost out of a stream captured elsewhere
            self._log(
                f"[yellow]It publishes to {rich_escape(target)}, and on {profile} that address is {profile} "
                f"itself. Give the Stream Server under System Settings a name {profile} can reach "
                f"(e.g. {socket.gethostname()}).[/yellow]"
            )

    def on_stream_panel_device_change_requested(self, event: StreamPanel.DeviceChangeRequested) -> None:
        """Device in the Streams tab: the camera or microphone the stream's
        ffmpeg opens on its machine, written into the config of the host the
        card is on; the Config tab does not show it."""
        event.stop()
        if self._write_stream_entry(event.stream_name, "device", event.device) is None:
            return
        self._log(f"[green]{event.stream_name}: " + (
            f"Start opens {rich_escape(event.device)}.[/green]" if event.device else
            "no device picked: Start takes the first of its machine.[/green]"))

    def _stream_entry(self, name: str) -> dict:
        """a stream's entry in the config of the host the card is on, as last read."""
        pipeline = self._current_pipeline
        if pipeline is None:
            return {}
        config, _ = self._load_config_for_target(pipeline.config_path, show_status=False,
                                                 target=self._get_panel_target())
        streams = config.get("Streams") if isinstance(config, dict) else None
        entry = streams.get(name) if isinstance(streams, dict) else None
        return entry if isinstance(entry, dict) else {}

    def _show_mllm_form(self, container: Vertical) -> None:
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
            action = f"{section} is now managed centrally (System Settings)"
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
            section_name = event.pipeline_name.replace("shared:", "", 1)
            target = self._settings_host(section_name)
            if target != "local":
                # that machine's own settings file and its pipeline configs
                self._sync_shared_section_to_target(section_name, target, save=True)
                return
            old_stream_server = self._stream_server_address() if section_name == "StreamServer" else None
            for path, val in event.values.items():
                self._shared_values[path] = val
            config_path = save_system_service_section(
                self._root,
                section_name,
                self._shared_section_data(section_name),
            )
            updated = self._apply_shared_section_to_local_configs(section_name)
            message = f"{section_name} system service saved to {config_path} and {updated} local pipeline config(s)"
            if old_stream_server is not None:
                message = f"{section_name} system service saved to {config_path}"
                moved, configs = self._repoint_local_streams(old_stream_server, self._stream_server_address())
                if moved:
                    message += (
                        f"; {moved} stream URL(s) in {configs} local pipeline config(s) followed it to "
                        f"{self._stream_server_address().get('host')} (Sync to Host on a pipeline's Config tab "
                        f"takes them to another host)"
                    )
            self._show_status(message)
            # the address may now name another machine: move the markers along
            self._refresh_visible_statuses()
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
        all_fields = all_fields + self._stored_capture_hosts(pipeline, event.values)
        stream_note = self._complete_stream_targets(pipeline, event.values)
        saved = self._save_pipeline_config_for_target(pipeline, all_fields, event.values, note=stream_note)

        if pipeline.name == "ASR Server":
            self._services = _build_service_registry(self._root)
            self._svc_map = {s.name: s for s in self._services}
            self._refresh_service_cards()
        elif pipeline.name in _BASE_CARD_PIPELINES and saved is not None and self._get_panel_target() == "local":
            # the Base dropdowns list the Bases just saved; a save to another
            # host refreshes the cards once its copy has landed (_run_scp)
            self._refresh_base_card_choices()

        # the Streams tab prunes by what it holds (record_keep_days): it gets
        # what was saved, on another host as well as here
        if saved is not None:
            self._refresh_stream_panels(pipeline, saved)
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

    def _refresh_stream_panels(self, pipeline: PipelineDef, config: dict | None = None) -> None:
        """refresh stream tabs after a pipeline config save, from the config
        that was saved (another host's), else from this machine's file."""
        if pipeline.name not in _STREAM_PIPELINES:
            return
        streams = streams_from_config(config) if config is not None else load_streams(pipeline.config_path)
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
        for old in container.query(".sync-bar, .sync-note"):
            old.remove()
        if shared_section in PRIVATE_SECTIONS:
            return  # this machine's admin password never leaves it
        if shared_section is not None and self._settings_host(shared_section) != "local":
            # the form shows that machine's settings and Save writes them there
            return
        target = self._get_panel_target()
        if pipeline is None and local_path is None and shared_section is None:
            return
        # a pipeline config is synced from the host it was read on, so the Host
        # selector picks the source and the picker below the destination: what
        # was edited on server-01 goes back to this machine, or on to another
        # host. System Settings and the MLLM launch config are different: their
        # Save writes this machine's project whatever the Host selector says,
        # so their source is this machine and the picker defaults to the
        # selected host.
        source = target if pipeline is not None else "local"
        options = _sync_destination_options(source, [p.name for p in load_ssh_profiles()])
        if not options:
            return
        preselect = target if shared_section is not None and any(
            name == target for _, name in options) else None
        select_kwargs = {"value": preselect} if preselect is not None else {}
        bar = Horizontal(
            Select(
                options,
                prompt="Select destination host...",
                id="sync-host-select",
                **select_kwargs,
            ),
            Button("Sync to Host", variant="warning", id="btn-sync-host"),
            classes="sync-bar" if shared_section is None else "sync-bar sync-bar-noted",
        )
        if shared_section == "StreamServer":
            container.mount(Static(
                "Save writes this machine's project and moves the stream URLs of its pipeline configs "
                "with the address. Sync to Host gives another machine this address (its own "
                "config/system_services.yml, which a console there reads) and those Streams entries "
                "(what its bases pull), and leaves the rest of its configs as they are.",
                classes="sync-note",
            ))
        elif shared_section is not None:
            container.mount(Static(
                f"Save writes this machine's project. Sync to Host copies the {shared_section} "
                f"section to another machine: into its pipeline configs that carry it, and into its own "
                f"config/system_services.yml (created when it has none), so what runs there connects to "
                f"the same service.",
                classes="sync-note",
            ))
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
        if event.button.id == "btn-sync-host":
            self._sync_to_host()
        elif event.button.id == "btn-sync-transform-host":
            self._sync_transform_to_host()
        elif event.button.id == "btn-sync-prompts-host":
            self._sync_prompts_to_host()
        elif event.button.id == "btn-sync-aschema-host":
            self._sync_action_schema_to_host()
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
        pipeline = self._current_pipeline
        fields = _make_stream_fields(
            name, self._stream_server_address(), _card_stream_kind(pipeline.name if pipeline is not None else ""))
        form.add_section(section_name, fields, self._new_stream_values(name))
        self._restore_add_stream_button()

    def _stream_server_address(self) -> dict[str, object]:
        """host and ports of System Settings → Stream Server, as this console holds them."""
        return {
            key: self._shared_values.get(f"StreamServer.{key}", fdef.get("default", ""))
            for key, fdef in SHARED_SECTIONS["StreamServer"]["fields"].items()
        }

    def _stored_capture_hosts(self, pipeline: PipelineDef, values: dict) -> list[LoaderFieldDef]:
        """each stream's ssh_profile and device are picked on the Streams tab,
        so the form does not hold them. Save still writes them: the URL
        completion reads the ssh_profile, and the config of another host is
        written from the form alone. Adds the stored values and returns the
        fields that carry them."""
        if pipeline.name not in _STREAM_PIPELINES:
            return []
        target = self._get_panel_target()
        if target == "local":
            config = load_existing_config(pipeline.config_path)
        else:
            config, _ = self._load_config_for_target(pipeline.config_path, show_status=False, target=target)
        streams = config.get("Streams") if isinstance(config, dict) else None
        if not isinstance(streams, dict):
            return []
        fields = []
        for key in [key for key in values if key.startswith("Streams.") and key.endswith(".target")]:
            name = key[len("Streams."):-len(".target")]
            entry = streams.get(name)
            for picked in _STREAMS_TAB_KEYS:
                path = f"Streams.{name}.{picked}"
                if path in values or not isinstance(entry, dict) or entry.get(picked) is None:
                    continue
                values[path] = entry[picked]
                fields.append(LoaderFieldDef(path=path, field_type="str", default="", description="",
                                             required=False, section=f"Streams.{name}"))
        return fields

    def _complete_stream_targets(self, pipeline: PipelineDef, values: dict) -> str:
        """Streams written as a bare path (ips/cam-1) get the Stream Server address
        of System Settings before the config is written, and the form shows what
        they became. Returns what to tell the user, to go after the save message."""
        if pipeline.name not in _STREAM_PIPELINES:
            return ""
        server = self._stream_server_address()
        host = str(server.get("host") or "").strip() or "localhost"
        names = [
            key[len("Streams."):-len(".target")] for key in values
            if key.startswith("Streams.") and key.endswith(".target")
        ]
        held_back: dict[str, str] = {}
        if is_loopback_host(host):
            # localhost in a URL is the machine that opens it: right for a stream
            # captured and pulled on this machine, wrong for any other
            config_host = self._get_panel_target()
            for name in names:
                short = any(is_stream_path(values.get(f"Streams.{name}.{key}")) for key in ("target", "read_target"))
                capture = str(values.get(f"Streams.{name}.ssh_profile") or "").strip()
                if is_select_sentinel(capture) or capture == "local":
                    capture = ""
                elsewhere = capture or (config_host if config_host != "local" else "")
                if short and elsewhere:
                    held_back[name] = elsewhere
        open_values = {key: val for key, val in values.items() if not any(
            key.startswith(f"Streams.{name}.") for name in held_back)}
        completed = complete_stream_urls(open_values, server)
        form = self._current_form
        for name in completed:
            for key in ("target", "read_target"):
                path = f"Streams.{name}.{key}"
                values[path] = open_values.get(path, values.get(path))
                if form is not None:
                    form.set_field_value(path, values[path])
        notes = []
        if completed:
            reach = " (reachable from this machine only)" if is_loopback_host(host) else ""
            notes.append(f"{', '.join(completed)}: completed with the Stream Server of System Settings, {host}{reach}")
        if held_back:
            who = ", ".join(sorted(set(held_back.values())))
            notes.append(
                f"{', '.join(held_back)}: left as a path, because the Stream Server is '{host}' under System "
                f"Settings and {who} cannot reach that. Put this machine's name there "
                f"(e.g. {socket.gethostname()}) and Save here again"
            )
        return "".join(f". {note}" for note in notes)

    def _repoint_local_streams(self, old: dict, new: dict) -> tuple[int, int]:
        """the Stream Server moved: the stream URLs of the local pipeline configs
        that named its old address follow it. (URLs moved, configs written)"""
        if all(str(old.get(key)) == str(new.get(key)) for key in ("host", "rtmp_port", "rtsp_port")):
            return 0, 0
        if not self._pipelines:
            self._pipelines = discover_pipelines()
            self._pipeline_map = {pipeline.name: pipeline for pipeline in self._pipelines}
        moved = configs = 0
        for pipeline in self._pipelines:
            if pipeline.name not in _STREAM_PIPELINES or not os.path.isfile(pipeline.config_path):
                continue
            config = load_existing_config(pipeline.config_path)
            streams = config.get("Streams") if isinstance(config, dict) else None
            if not isinstance(streams, dict):
                continue
            changed = 0
            for entry in streams.values():
                if not isinstance(entry, dict):
                    continue
                for key in ("target", "read_target"):
                    url = str(entry.get(key) or "").strip()
                    repointed = repoint_stream_url(url, old, new) if url else url
                    if repointed != url:
                        entry[key] = repointed
                        changed += 1
            if not changed:
                continue
            with open(pipeline.config_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
            moved += changed
            configs += 1
        if configs:
            self._target_config_cache.clear()
        return moved, configs

    def _new_stream_values(self, name: str) -> dict[str, object]:
        """a new stream opens on the Stream Server's URLs of <pipeline>/<name>, so
        the address typed once under System Settings is not typed again. With a
        loopback address there, only the path is filled in: whether localhost is
        right depends on the capture host, which Save looks at. A stream added
        on the ASR card says kind: audio, so a microphone that names no device
        (a Mac's first) is captured as one wherever the entry is read."""
        pipeline = self._current_pipeline
        app = pipeline.name.split()[0].lower() if pipeline is not None else "stream"
        path = f"{app}/{safe_segment(name, 'stream')}"
        kind = {f"Streams.{name}.kind": "audio"} if pipeline is not None and pipeline.name == "ASR Base" else {}
        server = self._stream_server_address()
        if is_loopback_host(server.get("host")):
            return {f"Streams.{name}.target": path, **kind}
        publish, pull = stream_server_urls(server, path)
        return {f"Streams.{name}.target": publish, f"Streams.{name}.read_target": pull, **kind}

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
        note: str = "",
    ) -> dict | None:
        """write the pipeline config on the card's host; returns what was
        written (on another host: what the copy carries), None when nothing was."""
        target = self._get_panel_target()
        if target == "local":
            save_config(pipeline.config_path, fields, values, transform=move_source_settings)
            saved = load_existing_config(pipeline.config_path)
            self._target_config_cache[self._config_cache_key(pipeline.config_path, "local")] = saved
            self._show_status(f"Saved locally to {pipeline.config_path}{note}")
            return saved

        profile = get_profile_by_name(target)
        if profile is None:
            self._show_status(f"SSH profile '{target}' not found; config was not saved.")
            return None

        tmp = tempfile.NamedTemporaryFile(
            "w",
            suffix=".yml",
            prefix="openmmla-config-",
            delete=False,
        )
        tmp_path = tmp.name
        tmp.close()
        save_config(tmp_path, fields, values, transform=move_source_settings)
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
                note=note,
            ),
            exclusive=True,
        )
        return cache_config

    # ── sync between hosts ───────────────────────────────────────
    #
    # Every file the console edits — a pipeline config, the prompt templates,
    # the action schema, the transform matrices — belongs to the host the Host
    # selector names, and a Sync to Host copies it from there to another
    # machine. Either end may be this one: what was edited on server-01 comes
    # back here, or goes on to another host (through this machine, the one
    # place both are reachable).

    def _host_profile(self, host: str):
        """the SSH profile of a host; None is this machine."""
        return None if host == "local" else get_profile_by_name(host)

    def _host_path(self, local_path: str, host: str) -> str:
        """where a file of this project lives on `host`."""
        profile = self._host_profile(host)
        return local_path if profile is None else self._remote_config_path(local_path, profile)

    def _host_join(self, host: str, directory: str, name: str) -> str:
        return os.path.join(directory, name) if host == "local" else _remote_path_join(directory, name)

    def _host_files(self, host: str, local_dir: str, suffix: str) -> list[str]:
        """the files of one directory of this project on `host`."""
        if self._host_profile(host) is None:
            try:
                return sorted(
                    name for name in os.listdir(local_dir)
                    if name.endswith(suffix) and not name.startswith(".")
                )
            except OSError:
                return []
        return _remote_list_files(self._host_profile(host), self._host_path(local_dir, host), suffix)

    def _source_is_reachable(self, source: str, say=None) -> bool:
        """a host whose profile has gone since the card was drawn has nothing
        to send: copying this machine's file under its name would be a lie."""
        if source == "local" or self._host_profile(source) is not None:
            return True
        (say or self._show_status)(f"SSH profile '{source}' not found; nothing was synced.")
        return False

    def _sync_selection(self, selector: str, say=None) -> str | None:
        """the host picked next to a Sync to Host button, or None with a note
        in the status line when there is nothing usable to sync to."""
        say = say or self._show_status
        try:
            value = self.query_one(selector, Select).value
        except Exception:
            return None
        if is_select_sentinel(value):
            say("Select a destination host first.")
            return None
        host = str(value)
        if host != "local" and get_profile_by_name(host) is None:
            say(f"SSH profile '{host}' not found.")
            return None
        return host

    def _sync_to_host(self) -> None:
        dest = self._sync_selection("#sync-host-select")
        if dest is None:
            return

        # System Settings shared-section view: push just this section to the
        # selected host (respecting any per-pipeline overrides on that host).
        if self._current_shared_section == "StreamServer":
            self._sync_streams_to_target(dest)
            return
        if self._current_shared_section:
            self._sync_shared_section_to_target(self._current_shared_section, dest)
            return

        if self._current_pipeline is not None:
            self._sync_config_between_hosts(
                self._current_pipeline.config_path, self._get_panel_target(), dest)
        elif self._current_config_local_path:
            # a config this machine keeps for every host (the MLLM launch
            # config): its form reads and writes the local file, so that is
            # what goes out whatever the Host selector says
            self._sync_config_between_hosts(self._current_config_local_path, "local", dest)

    def _sync_config_between_hosts(self, local_path: str, source: str, dest: str) -> None:
        """copy one config file from the host it was read on to another."""
        if source == dest or not self._source_is_reachable(source):
            return
        if source == "local" and not os.path.isfile(local_path):
            self._show_status(f"Local config not found: {local_path}. Save first.")
            return
        self._show_status(
            f"Syncing {os.path.relpath(local_path, self._root)} from {_host_label(source)} "
            f"to {_host_label(dest)} ..."
        )
        self.run_worker(self._run_config_copy(source, dest, local_path), exclusive=True)

    def _sync_files_between_hosts(
        self, source: str, dest: str, local_dir: str, files: list[str], label: str, say=None
    ) -> None:
        say = say or self._show_status
        if source == dest or not self._source_is_reachable(source, say):
            return
        dest_dir = self._host_path(local_dir, dest)
        say(
            f"Syncing {len(files)} {label} file(s) from {_host_label(source)} "
            f"to {_host_label(dest)}:{dest_dir} ..."
        )
        self.run_worker(self._run_files_copy(source, dest, local_dir, files, label, say), exclusive=True)

    def _sync_transform_to_host(self) -> None:
        try:
            panel = self.query_one(TransformMatrixPanel)
        except Exception:
            return
        dest = self._sync_selection("#transform-sync-host-select", panel.set_status)
        if dest is None:
            return
        source = panel.target
        local_dir = panel.local_dir
        profile = self._host_profile(source)
        files = (
            _local_transform_matrix_files(local_dir) if profile is None
            else _remote_transform_matrix_files(profile, self._host_path(local_dir, source))
        )
        if not files:
            panel.set_status(
                f"No transformation_matrices*.json files on {_host_label(source)}: "
                f"{self._host_path(local_dir, source)}"
            )
            return
        self._sync_files_between_hosts(
            source, dest, local_dir, files, "transform matrix", panel.set_status)

    def _sync_prompts_to_host(self) -> None:
        try:
            panel = self.query_one(PromptsPanel)
        except Exception:
            return
        dest = self._sync_selection("#prompts-sync-host-select", panel.set_status)
        if dest is None:
            return
        source = panel.target
        local_dir = panel.prompts_dir
        files = self._host_files(source, local_dir, ".txt")
        if not files:
            panel.set_status(
                f"No prompt files on {_host_label(source)}: {self._host_path(local_dir, source)}")
            return
        self._sync_files_between_hosts(source, dest, local_dir, files, "prompt", panel.set_status)

    def _sync_action_schema_to_host(self) -> None:
        try:
            panel = self.query_one(ActionSchemaPanel)
        except Exception:
            return
        dest = self._sync_selection("#aschema-sync-host-select", panel.set_status)
        if dest is None:
            return
        source = panel.target
        local_path = panel.schema_path
        if source == "local" and not os.path.isfile(local_path):
            panel.set_status(f"No action schema file: {local_path}")
            return
        self._sync_files_between_hosts(
            source, dest, os.path.dirname(local_path), [os.path.basename(local_path)],
            "action schema", panel.set_status)

    async def _stage_from_host(self, host: str, paths: list[str]) -> tuple[str, dict[str, str], list[str]]:
        """bring files of `host` onto this machine, from where they can be
        written to any other: (staging dir, {path on the host: local copy},
        failures). A local source is staged as itself and leaves no staging
        dir; two remote hosts have no route to each other, so a copy between
        them goes through here."""
        profile = self._host_profile(host)
        if profile is None:
            staged = {path: path for path in paths if os.path.isfile(path)}
            return "", staged, [
                f"{os.path.basename(path)}: not on this machine" for path in paths if path not in staged
            ]
        staging = tempfile.mkdtemp(prefix="openmmla-sync-")
        staged: dict[str, str] = {}
        failures: list[str] = []
        for index, path in enumerate(paths):
            # the index keeps two files of the same name (another directory,
            # another host) from landing on each other in the staging dir
            local_copy = os.path.join(staging, f"{index}-{os.path.basename(path)}")
            proc = await scp_from_remote_async(profile, path, local_copy)
            output = await _process_output(proc)
            rc = await proc.wait()
            if rc == 0 and os.path.isfile(local_copy):
                staged[path] = local_copy
            else:
                failures.append(f"{os.path.basename(path)}: {output.strip() or f'exit code {rc}'}")
        return staging, staged, failures

    async def _place_on_host(self, host: str, pairs: list[tuple[str, str]]) -> tuple[int, list[str]]:
        """write files that are on this machine onto `host`: (written, failures)."""
        profile = self._host_profile(host)
        written = 0
        failures: list[str] = []
        for local_path, dest_path in pairs:
            if profile is None:
                try:
                    if os.path.abspath(local_path) != os.path.abspath(dest_path):
                        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
                        shutil.copyfile(local_path, dest_path)
                    written += 1
                except OSError as exc:
                    failures.append(f"{os.path.basename(dest_path)}: {exc}")
                continue
            remote_dir = dest_path.rsplit("/", 1)[0]
            mkdir_proc = await ssh_run_async(profile, f"mkdir -p {_quote_remote_path(remote_dir)}")
            await mkdir_proc.wait()
            proc = await scp_file_async(profile, local_path, dest_path)
            output = await _process_output(proc)
            rc = await proc.wait()
            if rc == 0:
                written += 1
            else:
                failures.append(f"{os.path.basename(dest_path)}: {output.strip() or f'exit code {rc}'}")
        return written, failures

    async def _run_config_copy(self, source: str, dest: str, local_path: str) -> None:
        """one config file from one host to another. What lands is what the
        console then holds for the destination, and what its [C] marker says."""
        source_path = self._host_path(local_path, source)
        dest_path = self._host_path(local_path, dest)
        staging, staged, failures = await self._stage_from_host(source, [source_path])
        try:
            if source_path not in staged:
                self._show_status(
                    f"Sync failed: {failures[0] if failures else f'{_host_label(source)}:{source_path}'}")
                return
            written, failures = await self._place_on_host(dest, [(staged[source_path], dest_path)])
            if not written:
                self._show_status(f"Sync failed: {'; '.join(failures[:2])}")
                return
            self._target_config_cache[self._config_cache_key(local_path, dest)] = (
                load_existing_config(staged[source_path]))
            self._note_config_presence(dest, local_path, True)
            profile = self._host_profile(dest)
            if profile is not None:
                await self._maybe_push_master_key(profile, staged[source_path])
            self._show_status(
                f"Synced {os.path.relpath(local_path, self._root)} from {_host_label(source)} "
                f"to {_host_label(dest)}: {dest_path}"
            )
            self._refresh_service_cards()
        finally:
            if staging:
                shutil.rmtree(staging, ignore_errors=True)

    async def _run_files_copy(
        self, source: str, dest: str, local_dir: str, files: list[str], label: str, say=None
    ) -> None:
        """the named files of one directory of this project, from one host to
        another."""
        say = say or self._show_status
        source_dir = self._host_path(local_dir, source)
        dest_dir = self._host_path(local_dir, dest)
        paths = [self._host_join(source, source_dir, name) for name in files]
        staging, staged, failures = await self._stage_from_host(source, paths)
        try:
            written, placed_failures = await self._place_on_host(dest, [
                (staged[path], self._host_join(dest, dest_dir, name))
                for path, name in zip(paths, files) if path in staged
            ])
            failures += placed_failures
            if failures:
                say(
                    f"{label} sync to {_host_label(dest)}: {written} file(s) copied, "
                    f"{'; '.join(failures[:2])}"
                )
                return
            say(
                f"Synced {written} {label} file(s) from {_host_label(source)} "
                f"to {_host_label(dest)}:{dest_dir}"
            )
        finally:
            if staging:
                shutil.rmtree(staging, ignore_errors=True)

    def _remote_dir_for_local(self, local_dir: str, profile) -> str:
        rel = os.path.relpath(local_dir, self._root)
        return _remote_path_join(profile.remote_project_path, rel)

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

    def _sync_shared_section_to_target(self, section_name: str, target: str, save: bool = False) -> None:
        """write the section on screen into `target`: its pipeline configs and
        its own settings file. `save` is the Save of a form that shows that
        machine's settings (the file is created when it has none); otherwise
        this is Sync to Host, which copies Local's and only ever updates a
        settings file that is already there."""
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
        carriers = [pipeline for pipeline in self._pipelines if self._pipeline_has_section(pipeline, section_name)]
        for pipeline in carriers:
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
            _encrypt_secrets(remote_config)

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

        # the host's own settings file gets the section too, and is created
        # when the host has none: its services read it at startup, a console
        # there shows it, and both then say what this one says
        own_store = self._remote_settings_entry(
            target, profile, {section_name: section_data}, create=True)
        if own_store is not None:
            temp_paths.append(own_store[0])
            entries.append(own_store)

        if not entries:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self._show_status(
                f"'{target}' already has these {section_name} settings: its config/system_services.yml says "
                f"the same, and every pipeline config there that carries the section says the same or pins "
                f"its own.")
            return

        if not save:
            for key, value in section_data.items():
                self._shared_values[f"{section_name}.{key}"] = value

        loopback = [
            f"{section_name}.{key}" for key, value in section_data.items()
            if key in ("host", "url") and is_loopback_host(
                urlsplit(str(value)).hostname if "://" in str(value) else value)
        ]
        success_message = f"{'Saved' if save else 'Synced'} {section_name} system service"
        if loopback and not save:
            # the log pane is hidden on this node, so the warning rides along
            success_message = (
                f"Note: {', '.join(loopback)} is a loopback address, which on '{target}' means "
                f"'{target}' itself and not this machine; machines that share one service need "
                f"its real host name. {success_message}"
            )
        self._show_status(f"{'Saving' if save else 'Syncing'} {section_name} system service to {target} ...")
        self.run_worker(
            self._run_scp_batch(target, entries, cleanup_local=True, success_message=success_message),
            exclusive=True,
        )

    def _sync_streams_to_target(self, target: str) -> None:
        """Sync to Host on the Stream Server form: that machine gets this
        address in its own settings file (no pipeline config carries the
        section; a console there reads it from the file, created when it has
        none) and the stream URLs the address completed, which are what
        a base there pulls: the Streams entries of the local pipeline configs
        go into that machine's copies, and the rest of each config stays as it
        is there."""
        profile = get_profile_by_name(target)
        if profile is None:
            self._show_status(f"SSH profile '{target}' not found.")
            return
        if not self._pipelines:
            self._pipelines = discover_pipelines()
            self._pipeline_map = {pipeline.name: pipeline for pipeline in self._pipelines}
        entries: list[tuple[str, str, tuple[str, str], dict]] = []
        carried: list[str] = []
        same: list[str] = []
        for pipeline in self._pipelines:
            if pipeline.name not in _STREAM_PIPELINES or not os.path.isfile(pipeline.config_path):
                continue
            local_config = load_existing_config(pipeline.config_path)
            remote_config, _ = self._load_config_for_target(pipeline.config_path, show_status=False, target=target)
            streams = _streams_to_carry(local_config, remote_config)
            if streams is None:
                if isinstance(local_config, dict) and local_config.get("Streams"):
                    same.append(pipeline.name)
                continue
            remote_config = dict(remote_config) if isinstance(remote_config, dict) else {}
            remote_config["Streams"] = streams
            _encrypt_secrets(remote_config)
            tmp = tempfile.NamedTemporaryFile(
                "w", suffix=".yml", prefix="openmmla-streams-", delete=False, encoding="utf-8")
            with tmp:
                yaml.safe_dump(remote_config, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
            entries.append((tmp.name, self._remote_config_path(pipeline.config_path, profile),
                            self._config_cache_key(pipeline.config_path, target), remote_config))
            carried.append(pipeline.name)
        own_store = self._remote_settings_entry(
            target, profile, {"StreamServer": self._shared_section_data("StreamServer")}, create=True)
        if own_store is not None:
            entries.append(own_store)
        if not entries:
            self._show_status(
                f"Nothing to sync: '{target}' already has this Stream Server address, and "
                + (f"the Streams entries of {', '.join(same)} there match this machine's." if same
                   else "no pipeline config on this machine has Streams entries."))
            return
        what = ["the Stream Server address"] if own_store is not None else []
        if carried:
            what.append(f"the stream URLs of {', '.join(carried)}")
        what = " and ".join(what)
        self._show_status(f"Syncing {what} to {target} ...")
        self.run_worker(
            self._run_scp_batch(target, entries, cleanup_local=True, success_message=f"Synced {what}"),
            exclusive=True,
        )

    def _remote_settings_path(self) -> str:
        from openmmla.tui.system_services import system_services_config_path
        return system_services_config_path(self._root)

    def _remote_settings_entry(self, target: str, profile, sections: dict[str, dict], create: bool = False):
        """scp entry that writes `sections` into the host's own
        config/system_services.yml, or None when there is nothing to write.

        The services read that file on top of their pipeline config (a section
        the pipeline pins stays its own), so a copy left behind on a host, by a
        console that once ran there, would silently beat everything this
        console pushes if it were left out: Sync to Host and a Save on that
        host's own form both bring it in step, and create it when the host has
        none, so what its services read and what a console there shows are
        what this one says. `create` False only looks (nothing is written for
        a host without the file)."""
        local_path = self._remote_settings_path()
        remote_store, _ = self._load_config_for_target(local_path, show_status=False, target=target)
        if not isinstance(remote_store, dict):
            remote_store = {}
        if not remote_store and not create:
            return None
        updated = dict(remote_store)
        for name, data in sections.items():
            if name not in PRIVATE_SECTIONS:
                updated[name] = dict(data)
        if updated == remote_store:
            return None
        _encrypt_secrets(updated)
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".yml", prefix="openmmla-remote-settings-", delete=False, encoding="utf-8")
        with tmp:
            yaml.safe_dump(updated, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return (
            tmp.name,
            self._remote_config_path(local_path, profile),
            self._config_cache_key(local_path, target),
            updated,
        )

    def _settings_reference_for(self, target: str, central: dict[str, dict]) -> dict[str, dict]:
        """the shared sections a remote host's pipeline configs should mirror:
        its own settings where it has them, this console's everywhere else."""
        reference = dict(central)
        remote_store, _ = self._load_config_for_target(
            self._remote_settings_path(), show_status=False, target=target)
        if isinstance(remote_store, dict):
            for name in SHARED_SECTION_NAMES:
                data = remote_store.get(name)
                if name not in CONSOLE_ONLY_SECTIONS and isinstance(data, dict) and data:
                    reference[name] = data
        return reference

    def _remote_settings_drift(self, target: str, central: dict[str, dict], carried: set[str]) -> list[str]:
        """connection sections that the host's own config/system_services.yml
        sets to something else than System Settings here, as "Section → host".
        Secrets are compared decrypted and never shown."""
        from openmmla.utils.config import SYSTEM_SERVICE_SECTIONS, decrypt_config_values
        remote_store, _ = self._load_config_for_target(
            self._remote_settings_path(), show_status=False, target=target)
        if not isinstance(remote_store, dict) or not remote_store:
            return []
        drifted = []
        for name in SYSTEM_SERVICE_SECTIONS:
            if name not in carried or name not in central or not isinstance(remote_store.get(name), dict):
                continue
            if decrypt_config_values(remote_store[name]) != decrypt_config_values(central[name]):
                where = remote_store[name].get("url") or remote_store[name].get("host") or "other values"
                if "://" in str(where):
                    where = urlsplit(str(where)).netloc.rsplit("@", 1)[-1]
                drifted.append(f"{name} → {where}")
        return drifted

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
        System Settings store (``config/system_services.yml``).

        Only sections actually present in that file are returned, so a section
        that has never been saved centrally is left untouched at launch (its
        pipeline value is never clobbered by an unset default).
        """
        stored = load_system_services_config(self._root)
        result: dict[str, dict] = {}
        if isinstance(stored, dict):
            for name in SHARED_SECTION_NAMES:
                if name in CONSOLE_ONLY_SECTIONS:
                    continue  # read by this console only; never synced into pipeline configs
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
        up-to-date remote config (avoids racing the async push). A remote host
        with System Settings of its own is reconciled with those, section by
        section: they are what its services read, and what its Connections
        forms show and save.
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
                    f"[yellow]Updated {', '.join(drifted)} from System Settings before "
                    f"launch (local config was out of date).[/yellow]"
                )
            return True
        # remote
        if get_profile_by_name(target) is None:
            return True  # cannot verify; existing checks already warned
        remote_config, _ = self._load_config_for_target(config_path, show_status=False, target=target)
        overrides = pipeline_section_overrides(remote_config)
        carried = {name for name in central if name in (remote_config or {}) and name not in overrides}
        own_drift = self._remote_settings_drift(target, central, carried)
        # said once per host and difference, not at every Start
        if own_drift and self._own_settings_noted.get(target) != own_drift:
            self._own_settings_noted[target] = own_drift
            self._log(
                f"[yellow]'{target}' has System Settings of its own that differ from this machine's: "
                f"{'; '.join(own_drift)}. What runs there connects to those. Open the Connections "
                f"forms with Host = {target} to review them, or press Sync to Host on the Local "
                f"forms to replace them.[/yellow]"
            )
        reference = self._settings_reference_for(target, central)
        drifted = shared_section_drift(reference, remote_config, overrides=overrides)
        if not drifted:
            return True
        self._log(
            f"[yellow]System-services config on '{target}' is out of date "
            f"({', '.join(drifted)}); pushing latest from System Settings...[/yellow]"
        )
        self._sync_shared_sections_to_target(drifted, target, reference)
        self._log(f"[yellow]Relaunch {svc.display_name} once the sync above completes.[/yellow]")
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

    def _stream_server_recordings_panel(self, profile) -> StreamServerRecordingsPanel:
        """the server's inventory. Its API is asked on the card's host: the
        address of System Settings on Local, the SSH host of a remote card
        (which shows and controls that machine's own MediaMTX). The record
        folder is artifacts/streams/server of the project on that host, which
        is where both run modes put it."""
        server = self._stream_server_address()
        api_port = int(server.get("api_port") or recordings.API_PORT)
        if profile is None:
            host = str(server.get("host") or "localhost")
            quoted_root = shlex.quote(os.path.join(self._root, ARTIFACTS_DIR, SERVER_RECORD_REL))

            def run_shell(command: str) -> str | None:
                try:
                    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60.0)
                except Exception:
                    return None
                return result.stdout if result.returncode == 0 else None
        else:
            host = profile.host
            quoted_root = _quote_remote_path(
                _remote_path_join(profile.remote_project_path, ARTIFACTS_DIR, SERVER_RECORD_REL))

            def run_shell(command: str) -> str | None:
                try:
                    result = ssh_run_sync(profile, command, timeout=60.0)
                except Exception:
                    return None
                return result.stdout if result.returncode == 0 else None
        return StreamServerRecordingsPanel(host=host, api_port=api_port, quoted_root=quoted_root, run_shell=run_shell)

    def _remote_config_exists(self, svc: ServiceDef, target: str) -> tuple[bool | None, str]:
        """Check over SSH whether the remote config.yml for a service exists.

        Returns a (exists, remote_path) tuple. ``exists`` is True/False when it
        could be determined, or None when the check could not run (missing SSH
        profile or SSH error) so the caller does not block the launch.
        """
        return self._remote_file_exists(os.path.join(svc.config_dir, "config.yml"), target)

    def _remote_file_exists(self, local_path: str, target: str) -> tuple[bool | None, str]:
        """whether the file at `local_path`'s place in the project exists on
        `target`, as (exists, remote_path); None when that could not be told."""
        profile = get_profile_by_name(target)
        if profile is None:
            return None, os.path.basename(local_path)
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

    def _make_config_ready(self, svc: ServiceDef, target: str) -> bool:
        """False, having said why, when a make target that reads a config.yml
        would start on a host that has none: make would get as far as a
        traceback (nginx) or a tmux session that dies at once (flask, celery)."""
        pipeline = self._pipeline_map.get(_MAKE_CONFIG_PIPELINES.get(_make_target_for(svc.name), ""))
        if pipeline is None:
            return True
        if target == "local":
            exists, where = os.path.isfile(pipeline.config_path), pipeline.config_path
        else:
            exists, where = self._remote_file_exists(pipeline.config_path, target)
            if exists is not None:
                self._note_config_presence(target, pipeline.config_path, exists)
        if exists is None:
            self._log(f"[yellow]NOTE: could not verify {where} on '{target}'; launching anyway.[/yellow]")
            return True
        if exists:
            return True
        on_host = "here" if target == "local" else f"on '{target}'"
        self._log(f"[red]{svc.display_name}: config.yml not found {on_host} at {where}[/red]")
        form = SYSTEM_SERVICE_LABELS["nginx" if pipeline.name == "Nginx" else "flask"]
        hint = (
            f"Open the Config tab of {form}, set your values, and click Save "
            f"to write config.yml {on_host} before launching"
        )
        if target != "local" and os.path.isfile(pipeline.config_path):
            # the form on that host starts from defaults; this one is filled in
            hint += f", or copy this machine's with Host = Local and Sync to Host to '{target}'"
        self._log(f"[yellow]{hint}.[/yellow]")
        return False

    def _camera_sync_ready(self, target: str) -> bool:
        """False, having said why, when the IPS base config of the card's host
        cannot run a camera sync (it takes the main base and at least one
        other), rather than a terminal that opens only to show a traceback."""
        from openmmla.utils.config import camera_sync_problem
        path = self._ips_base_config_path()
        if target == "local":
            config, note = load_existing_config(path), ""
        else:
            config, note = self._load_config_for_target(path, show_status=False, target=target)
        if not config and note:
            return True  # not readable from here: the sync says it itself
        problem = camera_sync_problem(config)
        if problem:
            where = "this machine" if target == "local" else f"'{target}'"
            self._log(f"[yellow]IPS Camera Sync on {where}: {rich_escape(problem)}[/yellow]")
            return False
        return True

    def _ips_base_config_path(self) -> str:
        pipeline = self._pipeline_map.get("IPS Base")
        return pipeline.config_path if pipeline else os.path.join(self._root, "pipelines", "ips-base", "config.yml")

    @on(CameraManagerPanel.SyncRequested)
    def on_camera_sync_requested(self, event: CameraManagerPanel.SyncRequested) -> None:
        event.stop()
        self.run_worker(
            self._sync_camera_to_host(event.panel, event.camera, event.profile_name),
            group="camera-sync",
            exclusive=True,
        )

    async def _sync_camera_to_host(self, panel, camera: str, profile_name: str) -> None:
        """Sync to Host on the Calibration Cameras panel: that host's IPS base
        config gets this camera's parameters, Cameras.<name>, and the rest of
        it (its Bases and Streams, which belong to that host) stays as it is.
        The config's own Sync to Host copies the whole file instead."""
        def report(text: str, color: str) -> None:
            panel.set_status(text)
            self._log(f"[{color}]{rich_escape(text)}[/{color}]")

        local_path = self._ips_base_config_path()
        params = _calibrated_cameras(load_existing_config(local_path)).get(camera)
        if not params:
            report(f"'{camera}' has no calibrated parameters here to sync.", "yellow")
            return
        profile = get_profile_by_name(profile_name)
        if profile is None:
            report(f"SSH profile '{profile_name}' not found.", "red")
            return
        exists, remote_path = await asyncio.to_thread(self._remote_file_exists, local_path, profile_name)
        if exists is None:
            report(f"Could not reach {profile_name} to read {remote_path}.", "red")
            return
        if not exists:
            report(
                f"{profile_name} has no IPS base config yet ({remote_path}). Save one there first: IPS Base "
                f"with Host = {profile_name}, Config tab, Save; then sync the camera.", "yellow")
            return
        # read afresh rather than from the cache: someone may have saved it since
        result = await asyncio.to_thread(ssh_run_sync, profile, f"cat {_quote_remote_path(remote_path)}", 15.0)
        try:
            config = yaml.safe_load(result.stdout) if result.returncode == 0 else None
        except yaml.YAMLError:
            config = None
        if not isinstance(config, dict):
            report(f"Could not read {profile_name}:{remote_path}; it was left as it is.", "red")
            return
        cameras = config.get("Cameras")
        if not isinstance(cameras, dict):
            cameras = config["Cameras"] = {}
        if cameras.get(camera) == params:
            report(f"{profile_name} already has the same parameters for '{camera}'.", "green")
            return
        existed = camera in cameras
        cameras[camera] = copy.deepcopy(params)
        # the calibrator's own writer: the rest of the file is written back as read
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", prefix="openmmla-camera-", delete=False)
        tmp.close()
        try:
            await asyncio.to_thread(dump_yaml_pretty, config, tmp.name)
            proc = await scp_file_async(profile, tmp.name, remote_path)
            output = (await proc.stdout.read()).decode(errors="replace") if proc.stdout else ""
            rc = await proc.wait()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        if rc != 0:
            report(f"Sync of '{camera}' to {profile_name} failed: {output.strip() or f'scp exited {rc}'}", "red")
            return
        self._target_config_cache[self._config_cache_key(local_path, profile_name)] = config
        report(
            f"'{camera}' synced to {profile_name}: {'updated' if existed else 'added'} Cameras.{camera} in "
            f"{remote_path}; the rest of that config is as it was.", "green")

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
        note: str = "",
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
                    self._note_config_presence(*cache_key, True)
                self._show_status(f"Saved to {profile_name}:{remote_path}{note}")
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
            for local_path, _, _, _ in entries:
                # one key serves them all: push it with the first file that needs it
                try:
                    with open(local_path, "r", encoding="utf-8") as fh:
                        needs_key = "ENC(" in fh.read()
                except OSError:
                    needs_key = False
                if needs_key:
                    await self._maybe_push_master_key(profile, local_path)
                    break
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
                    self._note_config_presence(*cache_key, True)
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
            # the log is one transcript for the whole tab (commands keep
            # streaming after the user moves on), so a divider says which card
            # and host the lines below belong to. Drawn lazily: browsing the
            # tree without doing anything leaves no trail of empty headers
            if self._log_context and self._log_context != self._log_context_shown:
                self._log_context_shown = self._log_context
                self._cmd.log(f"[dim]── {self._log_context} ──[/dim]")
            self._cmd.log(message)
        except Exception:
            pass

    def _set_log_context(self, svc: ServiceDef, target: str) -> None:
        self._log_context = f"{svc.display_name} · {'Local' if target == 'local' else target}"

    # the command session is mounted once by compose() and never rebuilt, so a
    # card rebuild mid-download cannot orphan the progress row
    def _progress_start(self, label: str, total: int | None) -> None:
        try:
            self._cmd.start_progress(label, total)
        except Exception:
            pass

    def _progress_update(self, done: int, total: int, detail: str) -> None:
        try:
            self._cmd.update_progress(done, total, detail)
        except Exception:
            pass

    def _progress_end(self) -> None:
        # one row is shared by every download; the finishing worker still holds
        # its own key here, so anything above one means another is still running
        if len(self._downloads_in_flight) > 1:
            return
        try:
            self._cmd.end_progress()
        except Exception:
            pass

    def _sweep_staging_once(self) -> None:
        """drop abandoned download staging trees, once per process."""
        if self._staging_swept:
            return
        self._staging_swept = True
        self.run_worker(
            asyncio.to_thread(dl.sweep_staging, self._root, 14),
            group=_LAUNCHER_STAGING_SWEEP_WORKER_GROUP,
            exclusive=True,
        )

    def on_command_session_download_cancel_requested(
        self, event: CommandSession.DownloadCancelRequested
    ) -> None:
        self._log("[yellow]Cancelling download; the data already fetched is kept for resume.[/yellow]")
        try:
            self.workers.cancel_group(self, _LAUNCHER_DOWNLOAD_WORKER_GROUP)
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
            # a card moved off the configured machine reports this machine's own port
            own_port = self._off_configured_machine(svc, "local")
            if target in _SYSTEM_SVC_PORTS:
                # the address every pipeline is configured with, probed from here;
                # a loopback url means "this machine" and keeps the local check
                return system_service_reachable(self._root, target, own_port=own_port)
            if target == "flask":
                return system_service_reachable(self._root, "flask", own_port=own_port)
            return _check_tmux_session(target)
        elif svc.launch_type == "collection":
            # a recorder process alive on this machine means a recording is
            # in progress, whichever session it belongs to
            return self._note_host_recorders("local", _collection_recorders_local())
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
                if exists is not None:
                    self._note_config_presence(target, os.path.join(svc.config_dir, "config.yml"), exists)
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
            # System Settings store before launching (single source of truth).
            if not self._reconcile_shared_sections_before_launch(svc, target, is_remote):
                return
        elif svc.launch_type == "make" and not self._make_config_ready(svc, target):
            return
        if svc.name == "IPS Camera Sync" and not self._camera_sync_ready(target):
            return
        if svc.name == _ASR_BASE_CARD:
            self._put_speakers(event.params, target)
        problem = self._base_card_start_problem(svc, event.params, target)
        if problem:
            self._log(problem)
            return
        for note in self._base_card_start_notes(svc, event.params):
            self._log(note)
        if svc.name == _ASR_BASE_CARD:
            for note in self._speakers_start_notes(event.params, target):
                self._log(note)

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

        if svc.launch_type == "collection" and not self._confirm_start_into_ended_session(event.params, target):
            return

        launch_params = dict(event.params)
        if not self._ensure_pipeline_session_for_launch(svc, launch_params, target=target):
            self._log("[red]Could not resolve a launch session id.[/red]")
            return

        target_label = f"on '{target}'" if is_remote else "locally"
        self._log(f"[green]Starting {svc.display_name} {target_label}...[/green]")
        self._note_port_conflict(svc, target)

        if is_remote:
            self._launch_remote(svc, launch_params, target)
        else:
            self._launch_service(svc, launch_params)
        if svc.name in _GATEWAY_ROUTED_CARDS:
            self.run_worker(
                self._reload_gateway_after_start(svc.display_name),
                group="launcher-gateway-reload", exclusive=True,
            )
        self.set_timer(3.0, self._refresh_visible_statuses)

    def on_service_card_stop_requested(self, event: ServiceCard.StopRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return

        target = self._get_panel_target()
        is_remote = target != "local"
        target_label = f"on '{target}'" if is_remote else "locally"
        self._log(f"[red]Stopping {svc.display_name} {target_label}...[/red]")
        self._note_port_conflict(svc, target)

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
        if svc.name in _BASE_CARD_PIPELINES:
            # the Bases of its host's config and, for IPS, its matrix files
            self.run_worker(
                self._async_refresh_base_choices(svc, target),
                group="launcher-base-choices",
                exclusive=True,
            )
            self._reprobe_form_devices()
        if svc.name == _ASR_BASE_CARD:
            self._list_speakers(target)

    def _port_conflict(self, svc: ServiceDef, target: str) -> str:
        """explain a system service whose ports answer only in part: another
        program holds one of them. Probes, so it runs off the UI thread."""
        if svc.launch_type != "make" or _make_target_for(svc.name) not in _SYSTEM_SVC_PORTS:
            return ""
        check = None
        profile = get_profile_by_name(target) if target != "local" else None
        if profile is not None:
            check = lambda port: ssh_check_port(profile, port)
        return system_service_port_conflict(
            self._root, _make_target_for(svc.name), check,
            own_port=self._off_configured_machine(svc, target),
        )

    async def _async_note_port_conflict(self, svc: ServiceDef, target: str) -> None:
        conflict = await asyncio.to_thread(self._port_conflict, svc, target)
        if conflict:
            self._log(f"[yellow]{svc.display_name}: {conflict}.[/yellow]")

    def _note_port_conflict(self, svc: ServiceDef, target: str) -> None:
        if svc.launch_type != "make":
            return
        self.run_worker(
            self._async_note_port_conflict(svc, target),
            group="launcher-port-conflict",
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
        self._note_card_state(svc, target, is_running)
        for card in self.query(ServiceCard):
            if card.service_def.name == svc.name:
                if counts is not None:
                    card.update_stack_status(counts[0], counts[1])
                else:
                    card.update_status(is_running)
        self._build_tree()
        if counts is not None:
            self._log(f"{svc.display_name} ({target}): {counts[0]}/{counts[1]} running")
        else:
            status = "[green]Running[/green]" if is_running else "[red]Stopped[/red]"
            self._log(f"{svc.display_name} ({target}): {status}")
        await self._async_note_port_conflict(svc, target)

    def on_session_control_panel_refresh_requested(self, event: SessionControlPanel.RefreshRequested) -> None:
        """↻ on the Session Control panel: re-query the active session list."""
        event.stop()
        target = "local"
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
            if (target, session_id) in self._remote_deletes_in_flight:
                self._log(
                    f"[yellow]'{session_id}' is being deleted on '{target}' right now; there is nothing "
                    f"left to download there.[/yellow]"
                )
                return
            host_label = safe_segment(params.get("--host-label") or target, "host")
            key = f"collection-download:{target}:{session_id}:{host_label}"
            if key in self._downloads_in_flight:
                self._log(
                    "[yellow]A download for this session and host is already running; it takes the "
                    "audio and the video alike, whichever tab Download was pressed on.[/yellow]"
                )
                return
            self._downloads_in_flight.add(key)
            self.run_worker(
                self._run_collection_download(target, params, key),
                name=key,
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
        if self._collection_download_running(target, session_id):
            self._pending_collection_delete = None
            self._log(
                f"[yellow]'{session_id}' is being downloaded from '{target}' right now: Delete Remote waits "
                f"until that download is done or cancelled.[/yellow]"
            )
            return
        delete_key = (target, session_id)
        if delete_key in self._remote_deletes_in_flight:
            self._log(f"[yellow]'{session_id}' is already being deleted on '{target}'.[/yellow]")
            return
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

    async def _run_collection_download(self, profile_name: str, params: dict, key: str = "") -> None:
        try:
            profile = get_profile_by_name(profile_name)
            if profile is None:
                self._log(f"[red]SSH profile '{profile_name}' not found.[/red]")
                return
            session_id = self._collection_session_id(params)
            host_label = safe_segment(params.get("--host-label") or profile_name, "host")
            remote_path = self._collection_remote_path(profile, params)
            remote_transfer_path = await asyncio.to_thread(
                lambda: _expand_remote_home_path(remote_path, _remote_home(profile))
            )
            local_path = Path(self._collection_local_path(params, profile_name))
            await self._transfer_collection_tree(
                profile, profile_name, session_id, host_label, remote_transfer_path, local_path)
        except asyncio.CancelledError:
            self._progress_end()
            self._log("[yellow]Download cancelled; press Download again to resume.[/yellow]")
            raise
        finally:
            self._downloads_in_flight.discard(key)

    def _export_callbacks(self) -> stream_export.ExportCallbacks:
        """a transfer's log lines and progress go to this panel's log and its
        progress row; Cancel there cancels the download worker."""
        return stream_export.ExportCallbacks(
            log=self._log,
            progress_start=self._progress_start,
            progress_update=self._progress_update,
            progress_end=self._progress_end,
        )

    async def _transfer_collection_tree(
        self,
        profile,
        profile_name: str,
        session_id: str,
        host_label: str,
        remote_transfer_path: str,
        local_path: Path,
    ) -> bool:
        """scan, download (staged, resumable) and merge one remote
        <folder>/collection/<host label> tree: a session's recordings, audio
        and video alike, then note them in the session's manifest. Only what is
        not here yet is fetched. True when it is all here afterwards."""

        def note_in_manifest() -> str:
            manifest = update_collection_manifest(
                self._root,
                session_id=session_id,
                host_name=host_label,
                remote_path=remote_transfer_path,
                local_path=local_path,
            )
            return f"[green]Updated session manifest: {manifest}[/green]"

        return await stream_export.fetch_tree(
            profile,
            remote_transfer_path,
            local_path,
            staging=dl.staging_root(self._root, session_id, "collection", host_label),
            label=host_label,
            where=profile_name,
            what=session_id,
            callbacks=self._export_callbacks(),
            merge=merge_tree,
            after_merge=note_in_manifest,
        )

    def _collection_download_running(self, target: str, session_id: str) -> bool:
        """a Download of this session from this host is running, whatever the
        host label: with an Output Root of its own, Delete Remote removes the
        whole session folder."""
        prefix = f"collection-download:{target}:{session_id}:"
        return any(key.startswith(prefix) for key in self._downloads_in_flight)

    async def _run_collection_remote_delete(self, profile_name: str, params: dict) -> None:
        session_id = self._collection_session_id(params)
        key = (profile_name, session_id)
        # the press checked for a running download; this closes the window
        # between that press and the worker starting
        if self._collection_download_running(profile_name, session_id):
            self._log(f"[yellow]'{session_id}' is being downloaded from '{profile_name}': not deleted.[/yellow]")
            return
        self._remote_deletes_in_flight.add(key)
        try:
            await self._delete_collection_remote(profile_name, params)
        finally:
            self._remote_deletes_in_flight.discard(key)

    async def _delete_collection_remote(self, profile_name: str, params: dict) -> None:
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
        words = output.split()
        if rc == 0 and "DELETED" in words:
            self._log(f"[green]Deleted {profile_name}:{remote_delete_path}.[/green]")
        elif rc == 0 and "MISSING" in words:
            self._log(
                f"[yellow]Nothing to delete: {profile_name}:{remote_delete_path} is not there "
                f"(deleted before, or never recorded on this host).[/yellow]"
            )
        else:
            for line in output.strip().splitlines():
                self._log(rich_escape(line))
            self._log(f"[red]Remote collection delete failed (exit {rc}).[/red]")

    async def _mark_session_ended(self, session_id: str, target: str = "local") -> None:
        """record the stop in MongoDB without blocking the event loop."""
        note = await asyncio.to_thread(self._mark_mongodb_session_ended, session_id, target)
        if note:
            self._log(note)
        self._release_collection_session(session_id)

    def _release_collection_session(self, session_id: str) -> None:
        """the recording is over on every host: the cards go back to `Create
        MongoDB Session`, so the next Start is a new take and not a second
        helping of this one. Download and Delete Remote do not need the id on
        the card: they fall back to the session last recorded on their host."""
        session_id = _safe_session_id(session_id)
        if not session_id:
            return
        if _safe_session_id(self._collection_sticky.get("--session-id")) == session_id:
            self._collection_sticky["--session-id"] = ""
        try:
            cards = list(self.query(ServiceCard))
        except Exception:
            return
        for card in cards:
            if card.service_def.launch_type != "collection":
                continue
            shown = ((card.collection_snapshot() or {}).get("values") or {}).get("--session-id")
            if _safe_session_id(shown) == session_id:
                # the card is captured before it is rebuilt: it has to agree
                card.select_collection_session(_NEW_COLLECTION_SESSION_CHOICE)

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
        await self._end_session_unless_still_recording(session_id, profile_name)
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
        await self._end_session_unless_still_recording(session_id, "local")
        await self._reload_current_service_view()

    def _hosts_still_recording(self, session_id: str, stopped: str) -> list[str]:
        """the other hosts this console started the session on (and this
        machine) where one of its recorders is still alive. Asks each of them
        for its processes, so it runs off the event loop."""
        candidates = {"local", *self._collection_launch_targets.get(session_id, set())} - {stopped}
        busy = []
        for host in sorted(candidates):
            if host == "local":
                recorders = _collection_recorders_local()
            else:
                profile = get_profile_by_name(host)
                if (profile is None or self._target_states.get(host) == "offline"
                        or TARGET_PLATFORMS.get(host) == "windows"):
                    continue
                recorders = _collection_recorders_remote(profile)
            if any(session == session_id for _, session, _ in recorders):
                busy.append("this machine" if host == "local" else f"'{host}'")
        return busy

    async def _end_session_unless_still_recording(self, session_id: str, stopped: str) -> None:
        """Stop on one host: the session as a whole has only ended when no
        other host is still recording it. It used to be marked ended at the
        first Stop, in the middle of a recording that went on elsewhere."""
        # the recorders of this session are gone from the host that was stopped:
        # its card must not open on the session again before the next probe
        recording = self.__dict__.get("_collection_host_sessions", {}).get(stopped)
        if recording and session_id in recording:
            recording.remove(session_id)
        busy = await asyncio.to_thread(self._hosts_still_recording, session_id, stopped)
        if busy:
            self._log(
                f"[yellow]Session '{session_id}' is still being recorded on {', '.join(busy)}: not marked "
                f"ended. Stop All Hosts ends it everywhere.[/yellow]"
            )
            return
        await self._mark_session_ended(session_id, stopped)

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
            if TARGET_PLATFORMS.get(profile.name) == "windows":
                continue  # no recorder can run there, and no bash to stop one with
            targets.append(profile.name)
        return targets

    async def _stop_all_on_host(self, target: str, session_id: str) -> tuple[int, str] | None:
        """the stop command on one host of a Stop All Hosts; None for a Windows
        host. Only a host that has been picked somewhere is known to be one, so
        the others are asked first (once: the answer is kept)."""
        if target != "local" and target not in TARGET_PLATFORMS:
            profile = get_profile_by_name(target)
            if profile is not None:
                await asyncio.to_thread(remote_platform, profile)
        if TARGET_PLATFORMS.get(target) == "windows":
            return None
        return await self._collection_stop_on_target(target, session_id)

    async def _run_collection_stop_all(self, session_id: str, targets: list[str]) -> None:
        results = await asyncio.gather(
            *(self._stop_all_on_host(target, session_id) for target in targets),
            return_exceptions=True,
        )
        stopped = 0
        windows: list[str] = []
        for target, result in zip(targets, results):
            label = "local" if target == "local" else f"'{target}'"
            if isinstance(result, BaseException):
                self._log(f"[red]{label}: stop failed ({result}).[/red]")
                continue
            if result is None:
                # no bash there to stop anything with, and no recorder to stop
                windows.append(target)
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
        asked = len(targets) - len(windows)
        skipped = (
            f"; skipped {', '.join(windows)} (Windows, where no recorder runs)" if windows else ""
        )
        if stopped == asked:
            self._log(
                f"[green]Collection session '{session_id}' stopped on all {asked} host(s){skipped}.[/green]"
            )
        else:
            self._log(
                f"[yellow]Collection session '{session_id}': {stopped}/{asked} host(s) "
                f"stopped cleanly; check the lines above{skipped}.[/yellow]"
            )
        await self._mark_session_ended(session_id, "local")
        await self._reload_current_service_view()

    def _refresh_visible_statuses(self) -> None:
        self.run_worker(
            self._async_refresh_visible_statuses(),
            group=_LAUNCHER_STATUS_WORKER_GROUP,
            exclusive=True,
        )

    async def _async_refresh_visible_statuses(self) -> None:
        states = await asyncio.to_thread(self._detect_visible_statuses)
        for svc_name, (probed_target, is_running, counts) in states.items():
            node = self._node_host_cache.get(svc_name)
            if node is not None and node.target != probed_target:
                continue  # the node was moved to another host while this ran
            self._svc_states[svc_name] = is_running
            svc = self._svc_map.get(svc_name)
            card_target = self._get_panel_target()
            if svc is None or card_target != probed_target or self._off_configured_machine(svc, card_target):
                continue  # the card on screen reports another host than the marker
            for card in self.query(ServiceCard):
                if card.service_def.name == svc_name:
                    if counts is not None:
                        card.update_stack_status(counts[0], counts[1])
                    else:
                        card.update_status(is_running)
            if svc.launch_type == "collection":
                self._follow_host_recording_session(probed_target)
        self._build_tree()
        # the [E] markers need the env states of every host a node sits on
        for target in {node.target for node in self._node_host_cache.values()}:
            if target not in self._env_statuses and TARGET_STATES.get(target) != "offline":
                self._kick_env_status_refresh(target)

    def _detect_visible_statuses(self) -> dict[str, tuple[str, bool, tuple[int, int] | None]]:
        """node name -> (host it was probed on, running, stack counts); every
        node is probed on its own host. Runs in a worker thread."""
        states: dict[str, tuple[str, bool, tuple[int, int] | None]] = {}
        profiles = load_ssh_profiles()
        for svc in self._services:
            node = self._resolve_node_host(svc, profiles)
            is_running, counts = self._detect_node_status(svc, node)
            states[svc.name] = (node.target, is_running, counts)
        # the [C] markers of the nodes on another host read that host's files
        for target in {state[0] for state in states.values()} - {"local"}:
            profile = next((p for p in profiles if p.name == target), None)
            if profile is None or TARGET_STATES.get(target) == "offline":
                continue
            presence = self._remote_config_presence(profile)
            if presence is not None:
                self._config_presence[target] = presence
        return states

    def _detect_node_status(self, svc: ServiceDef, node: NodeHost) -> tuple[bool, tuple[int, int] | None]:
        target = node.target
        if node.follows and not node.machine_target:
            # a machine the console cannot log into: only the address itself,
            # probed from here, says anything (a tmux session on this machine
            # would be someone else's)
            make_target = _make_target_for(svc.name)
            port_probed = make_target in _SYSTEM_SVC_PORTS or make_target == "flask"
            return (system_service_reachable(self._root, make_target) if port_probed else False), None
        if target != "local" and TARGET_STATES.get(target) == "offline":
            return False, None  # no ssh timeouts for a host known to be down
        counts = self._stack_running_counts(svc, target)
        if counts is not None:
            up, total = counts
            return (total > 0 and up == total), counts
        if target == "local":
            return self._detect_running(svc), None
        return self._detect_running_remote(svc, target), None

    def _status_needs_ssh(self, svc: ServiceDef, node: NodeHost) -> bool:
        """whether this node's state can only be had by logging into its host
        (an address that names a machine is probed from here instead)."""
        if node.target == "local":
            return False
        if svc.launch_type != "make" or not node.machine_target:
            return True
        make_target = _make_target_for(svc.name)
        return not (make_target in _SYSTEM_SVC_PORTS or make_target == "flask")

    def status_rows(self, use_ssh: bool = True) -> list[dict]:
        """what the Status tab lists: every service the console can start, on
        its own host and probed the way its sidebar marker is, so the two tabs
        cannot disagree. Runs in a worker thread.

        Without `use_ssh` (the tab's five-second refresh) a node that can only
        be asked over ssh keeps the state of the last full pass; `running` is
        None when it has never been asked."""
        rows: list[dict] = []
        profiles = load_ssh_profiles()
        memo: dict[str, tuple] = self.__dict__.setdefault("_status_memo", {})
        for svc in self._services:
            if svc.launch_type == "bash":
                continue  # interactive: lives in its own terminal, nothing to probe
            node = self._resolve_node_host(svc, profiles)
            if use_ssh or not self._status_needs_ssh(svc, node):
                running, counts = self._detect_node_status(svc, node)
                memo[svc.name] = (node.target, running, counts)
                self._svc_states[svc.name] = running
            else:
                known = memo.get(svc.name)
                running, counts = (known[1], known[2]) if known and known[0] == node.target else (None, None)

            port, session, make_target = "-", "", ""
            if svc.launch_type == "make":
                make_target = _make_target_for(svc.name)
                ports = system_service_probe_ports(self._root, "flask" if make_target == "celery" else make_target)
                port = "/".join(str(p) for p in ports) if make_target != "celery" and ports else "-"
                session = make_target if make_target in ("flask", "celery") else ""
            elif svc.launch_type == "vllm":
                port = str(_mllm_config(self._root)["port"])
                session = _service_session_name(svc)
            rows.append({
                "name": svc.display_name,
                "key": svc.name,
                "make_target": make_target,
                # the machine as System Settings write it, else the card's host
                "host": node.machine or ("local" if node.target == "local" else node.target),
                "target": node.target,
                # placed by System Settings on a named machine: part of the
                # deployment, so its being down is worth a row of its own
                "placed": bool(node.follows),
                "running": running,
                "counts": counts,
                "port": port,
                "session": session,
            })
        return rows

    def _detect_running_remote(self, svc: ServiceDef, profile_name: str) -> bool:
        """check if a service is running on a remote host via SSH."""
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return False
        if svc.launch_type == "make":
            target = _make_target_for(svc.name)
            own_port = self._off_configured_machine(svc, profile_name)
            if target in _SYSTEM_SVC_PORTS:
                # probed from this machine at the configured address; a loopback
                # url, or a card moved off the configured machine, asks the host
                # about its own port instead
                return system_service_reachable(
                    self._root, target,
                    remote_loopback_check=lambda port: ssh_check_port(profile, port),
                    own_port=own_port,
                )
            if target == "flask":
                return system_service_reachable(
                    self._root, "flask",
                    remote_loopback_check=lambda port: ssh_check_port(profile, port),
                    own_port=own_port,
                )
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
            return self._note_host_recorders(profile_name, _collection_recorders_remote(profile))
        return False

    def on_service_card_action_requested(self, event: ServiceCard.ActionRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None or event.action != _INFRA_FETCH_TOKEN_ACTION:
            return
        if not _infra_docker_mode(svc, event.params):
            self._log(
                "[yellow]Fetch Token reads the token the docker stack was set up with "
                "(container config or docker/.env). Switch Run mode to docker; a native "
                "InfluxDB manages its tokens in its own web UI.[/yellow]"
            )
            return
        target = self._get_panel_target()
        self._log(f"  Reading the InfluxDB admin token from {target}...")
        self.run_worker(
            self._fetch_influx_token(target),
            name="fetch-influx-token",
            group="launcher-fetch-token",
            exclusive=True,
        )

    async def _fetch_influx_token(self, target: str) -> None:
        """read the docker stack's influx admin token on `target` and store it
        (encrypted) as InfluxDB.token in System Settings. The token itself never
        reaches the log pane."""
        profile = None
        try:
            if target == "local":
                result = await asyncio.to_thread(
                    subprocess.run, ["sh", "-c", _INFRA_READ_TOKEN_SH],
                    cwd=self._root, capture_output=True, text=True, timeout=30,
                )
                origin = "this machine"
            else:
                profile = get_profile_by_name(target)
                if profile is None:
                    self._log(f"[red]SSH profile '{target}' not found.[/red]")
                    return
                cmd = f"cd {_quote_remote_path(profile.remote_project_path)} && {_INFRA_READ_TOKEN_SH}"
                result = await asyncio.to_thread(ssh_run_sync, profile, cmd, 30.0)
                origin = f"{profile.name} ({profile.host})"
        except (subprocess.TimeoutExpired, OSError) as e:
            self._log(f"[red]Could not read the token: {e}[/red]")
            return

        token = result.stdout.strip().strip('"').strip("'")
        if not token or any(ch.isspace() for ch in token) or len(token) < 16:
            detail = (result.stderr or "").strip().splitlines()
            self._log(
                f"[red]No InfluxDB token found on {origin}: neither the running container's "
                f"/etc/influxdb2/influx-configs nor docker/.env has one. Has the stack been "
                f"started there?[/red]"
            )
            if detail:
                self._log(f"  {detail[-1]}")
            return

        config = load_system_services_config(self._root)
        section = dict((config.get("InfluxDB") if isinstance(config, dict) else None) or {})
        current = str(section.get("token") or "")
        try:
            from openmmla.utils.crypto import is_encrypted, decrypt_value
            if is_encrypted(current):
                current = decrypt_value(current)
        except Exception:
            current = ""
        if current == token:
            self._log(f"[green]InfluxDB.token already matches {origin} ({_mask_secret(token)}).[/green]")
            return

        section["token"] = token
        path = save_system_service_section(self._root, "InfluxDB", section)
        # remote pipeline configs are re-synced from System Settings on launch
        self._target_config_cache.clear()
        self._log(
            f"[green]InfluxDB.token updated from {origin}: {_mask_secret(token)} "
            f"→ stored encrypted in {os.path.relpath(path, self._root)}.[/green]"
        )

        url_host, _ = system_service_endpoint(self._root, "influxdb") or ("", 0)
        if target != "local" and not is_loopback_host(url_host) and profile is not None:
            if not (hosts_match(url_host, profile.host) or hosts_match(url_host, profile.name)):
                self._log(
                    f"[yellow]Note: InfluxDB.url points at {url_host}, but this token came from "
                    f"{profile.host}. Point the url at the same host or the token will be "
                    f"rejected.[/yellow]"
                )

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
            self._log(f"[yellow]{svc.display_name} is not running ({target}). Start the service first.[/yellow]")
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
                self._log(f"[cyan]── Logs for {svc.display_name} ──[/cyan]")
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
            self._log_collection_recorders(svc, _collection_recorders_local(), "this machine")
            return
        elif svc.launch_type in ("tmux", "vllm"):
            session_name = _service_session_name(svc)
        else:
            self._log(f"[yellow]No logs available for {svc.display_name}[/yellow]")
            return

        self._log(f"[cyan]── Logs for {svc.display_name} (session: {session_name}) ──[/cyan]")
        output = _capture_tmux_pane(session_name)
        for line in output.splitlines():
            self._log(line)
        self._log(f"[cyan]── End of logs ──[/cyan]")

    def _log_collection_recorders(self, svc: ServiceDef, recorders: list, where: str) -> None:
        """a recorder prints into the terminal window it opened, so the card's
        Logs can only say which recorders are alive."""
        self._log(f"[cyan]── {svc.display_name} on {where} ──[/cyan]")
        if not recorders:
            self._log("(no recorder is running)")
            return
        for role, session, pid in sorted(recorders):
            self._log(f"  {role} recorder · session {session} · pid {pid}")
        self._log("  Each recorder prints into the terminal window it was started in.")

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
                    "mediamtx": "if tmux has-session -t mediamtx 2>/dev/null; then tmux capture-pane -p -t mediamtx | tail -n 80; else echo '(no mediamtx tmux session found)'; fi",
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
            profile = get_profile_by_name(self._get_panel_target())
            if profile is not None:
                self._log_collection_recorders(svc, _collection_recorders_remote(profile), profile.name)
            return
        elif svc.launch_type in ("tmux", "vllm"):
            session_name = _service_session_name(svc)
        else:
            self._log(f"[yellow]No logs available for {svc.display_name}[/yellow]")
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
            self._log(f"[red]Error launching {svc.display_name}: {e}[/red]")

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
        # a host that does not say is taken for linux, as before
        platform_name = remote_platform(profile, timeout=4.0) or "linux"
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
                # the id this console was recording under is lost with a
                # restart, and another console never had it: say so before a
                # live session is split in two
                running = sessions.find_one(
                    {"experiment_id": exp_id, "group_id": group_id, "status": "active"},
                    sort=[("start_time", -1)],
                )
                if running and running.get("session_id") != session_id:
                    self._log(
                        f"[yellow]{exp_id}/{group_id} already has an active session, "
                        f"'{running.get('session_id')}'. A new one is created; to join that one, pick it "
                        f"under Session ID instead.[/yellow]"
                    )
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

    def _ensure_session_registered(self, session_id: str, experiment_group: object,
                                   target: str, created_by: str) -> None:
        """a launch under an existing session id: register it when MongoDB does
        not know it (any more). The id may come from a list that is a few
        seconds old, from artifacts on disk, or from a session another console
        deleted; recording under it regardless left files that belong to no
        session, which Stop then could not mark ended. Never blocks a launch."""
        try:
            mongo_config, _ = self._collection_mongodb_config(target)
            from pymongo import MongoClient
            from openmmla.utils.constants import MONGODB_DEFAULT_DB
        except Exception:
            return  # no config, no pymongo: nothing to check the id against
        if not mongo_config:
            return
        db_name = str(mongo_config.get("db") or MONGODB_DEFAULT_DB)
        try:
            client = MongoClient(
                str(mongo_config.get("url") or "").strip(),
                serverSelectionTimeoutMS=1500, connectTimeoutMS=1500,
            )
            try:
                sessions = client[db_name]["sessions"]
                if sessions.find_one({"session_id": session_id}, {"_id": 1}) is not None:
                    return
                exp_id, group_id = self._parse_collection_experiment_group(experiment_group)
                # the id names its experiment and group; the card may show another
                if not exp_id or not group_id or not session_id.startswith(f"{exp_id}_{group_id}_"):
                    self._log(
                        f"[yellow]Session '{session_id}' is not registered in MongoDB, and the selected "
                        f"Experiment Group is not the one it belongs to: not registered, it will only "
                        f"show up through its artifacts.[/yellow]"
                    )
                    return
                data = load_experiments(self._root)
                sessions.insert_one({
                    "session_id": session_id,
                    "experiment_id": exp_id,
                    "group_id": group_id,
                    "participants": list(get_participant_aliases(exp_id, group_id, data).values()),
                    "start_time": datetime.now(timezone.utc),
                    "end_time": None,
                    "status": "active",
                    "metadata": {"created_by": created_by, "registered_again": True},
                })
            finally:
                client.close()
        except Exception as e:
            self._log(f"[yellow]MongoDB unreachable ({e}); could not check session '{session_id}'.[/yellow]")
            return
        self._log(
            f"[yellow]Session '{session_id}' was not in MongoDB (deleted, or created while it was "
            f"down): registered it again for {exp_id}/{group_id}.[/yellow]"
        )
        self._invalidate_session_choice_cache()

    def _session_record(self, session_id: str, target: str, update: dict | None = None) -> dict | None:
        """the MongoDB document of a session (after `update`, a $set, when one
        is given); None when there is none or MongoDB cannot be asked."""
        try:
            mongo_config, _ = self._collection_mongodb_config(target)
            from pymongo import MongoClient
            from openmmla.utils.constants import MONGODB_DEFAULT_DB
        except Exception:
            return None
        if not mongo_config:
            return None
        try:
            client = MongoClient(
                str(mongo_config.get("url") or "").strip(),
                serverSelectionTimeoutMS=1500, connectTimeoutMS=1500,
            )
            try:
                sessions = client[str(mongo_config.get("db") or MONGODB_DEFAULT_DB)]["sessions"]
                if update:
                    sessions.update_one({"session_id": session_id}, {"$set": update})
                return sessions.find_one({"session_id": session_id})
            finally:
                client.close()
        except Exception:
            return None

    def _confirm_start_into_ended_session(self, params: dict, target: str) -> bool:
        """hold a collection Start back when its session has already ended.

        The session id follows the user from host to host, and it still does
        after Stop All Hosts: the next Start, meant as a new take, would quietly
        go into the finished session. The second press goes ahead and makes
        the session active again."""
        session_id = self._collection_session_id(params)
        if not session_id:
            return True  # "Create MongoDB Session"
        record = self._session_record(session_id, target)
        if not record or record.get("status") != "ended":
            self._pending_ended_start = None
            return True
        if self._pending_ended_start == (target, session_id):
            self._pending_ended_start = None
            self._session_record(session_id, target, {"status": "active", "end_time": None})
            self._log(f"[yellow]Recording into '{session_id}' again: it is active once more.[/yellow]")
            return True
        self._pending_ended_start = (target, session_id)
        ended = record.get("end_time")
        when = f" at {ended.strftime('%H:%M UTC')}" if isinstance(ended, datetime) else ""
        self._log(
            f"[yellow]Session '{session_id}' has ended{when}. Pick 'Create MongoDB Session' for a new "
            f"take, or press Start again to record into it all the same.[/yellow]"
        )
        return False

    def forget_session(self, session_id: str) -> None:
        """a session was deleted in the Sessions tab: stop offering its id. The
        cards kept it as their selection, so the next Start recorded into a
        session MongoDB no longer had."""
        session_id = _safe_session_id(session_id)
        if not session_id:
            return
        if self._collection_sticky.get("--session-id") == session_id:
            self._collection_sticky.pop("--session-id", None)
        for key, params in list(self._collection_last_params.items()):
            if self._collection_session_id(params) == session_id:
                del self._collection_last_params[key]
        self._collection_launch_targets.pop(session_id, None)
        self._invalidate_session_choice_cache()
        if self._current_service_name:
            self.run_worker(
                self._reload_current_service_view(capture=False),
                group=_LAUNCHER_UI_WORKER_GROUP,
                exclusive=True,
            )

    def _ensure_collection_session_for_launch(self, params: dict, target: str = "local") -> bool:
        session_id = self._collection_session_id(params)
        if session_id:
            params["--session-id"] = session_id
            self._ensure_session_registered(
                session_id, params.get("--experiment-group"), target, "tui_collection")
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
            self._ensure_session_registered(
                session_id, params.get("--experiment-group"), target,
                f"tui_{safe_segment(svc.artifact_pipeline, 'pipeline')}")
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
        # the recorders run ffmpeg, which Homebrew installs outside the PATH of
        # the login bash they start in: on macOS only zsh's profile adds it
        return _with_stream_path(f"PYTHONPATH={_REMOTE_COLLECTION_RUNTIME_ENV}:$PYTHONPATH {command}")

    @staticmethod
    def _interactive_ssh_args(profile) -> list[str]:
        args = profile.base_ssh_args()
        # the ssh binary comes resolved (/usr/bin/ssh): found by its name
        ssh_index = next((i for i, arg in enumerate(args) if os.path.basename(arg) == "ssh"), None)
        if ssh_index is None:
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
                if value is None or isinstance(value, (list, tuple)):
                    continue  # a per-instance value goes to its own instance below
                if self._count_left_out(svc, comp, flag, value):
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

            for index in range(count):
                instance_parts = self._instance_flag_parts(comp, params, index)
                instance_command = command
                if instance_parts:
                    instance_command += " " + " ".join(shlex.quote(part) for part in instance_parts)
                run_cmd = (
                    f"cd {_quote_remote_path(remote_config_dir)} && "
                    f"export PYTHONPATH={_quote_remote_path(remote_root)}:$PYTHONPATH && "
                    f"{instance_command}"
                )
                wrapped_cmd = wrap_remote(run_cmd, svc.conda_env)
                label = f"{comp.role} {index + 1}" if count > 1 else comp.role
                tab_cmds.append((label, self._remote_terminal_command(profile, wrapped_cmd)))

        return tab_cmds

    @staticmethod
    def _instance_flag_parts(comp: ComponentDef, params: dict, index: int) -> list[str]:
        """the flags one instance of a component gets for itself: a card's
        per-instance value (a list, one entry per instance, such as the Bases
        entry of each base: -b 0, -b 1), when that instance has one."""
        parts: list[str] = []
        for flag in comp.flags:
            values = params.get(flag)
            if not isinstance(values, (list, tuple)) or index >= len(values):
                continue
            text = str(values[index] if values[index] is not None else "").strip()
            if text:
                parts.extend([flag, text])
        return parts

    @staticmethod
    def _count_left_out(svc: ServiceDef, comp: ComponentDef, flag: str, value) -> bool:
        """True for a count at 0 that a component is better without: the
        synchronizer's --num_bases (Sync Waits For) at 0, or another
        component's count at 0. The synchronizer then counts the bases from
        its config's Bases, rather than wait for none."""
        if flag != _SYNC_WAIT_FLAG and not any(
                other.count_flag == flag for other in svc.components if other is not comp):
            return False
        return _coerce_int(value, 0) <= 0

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
            self._log(f"[red]No collection components defined for {svc.display_name}[/red]")
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
            self._log(f"[red]No components defined for {svc.display_name}[/red]")
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
                if val is None or isinstance(val, (list, tuple)):
                    continue  # a per-instance value goes to its own instance below
                if self._count_left_out(svc, comp, f, val):
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

            for i in range(count):
                instance_parts = self._instance_flag_parts(comp, params, i)
                instance_cmd = py_cmd
                if instance_parts:
                    instance_cmd += " " + " ".join(shlex.quote(part) for part in instance_parts)
                label = f"{comp.role} {i + 1}" if count > 1 else comp.role
                tab_cmds.append((label, f"{preamble} && {instance_cmd}"))

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

        self._log(f"[green]{svc.display_name} launched in new terminal window.[/green]")

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
        """open a Terminal.app tab or window per component on macOS.

        The commands go to osascript as arguments (_MAC_TABS_SCRIPT), never
        quoted into the script itself, and the launch runs in a thread of its
        own: what it opens takes a few seconds, as a new shell is given the
        time its startup files need before it is typed into, and the launcher
        stays live meanwhile."""
        if not tab_cmds:
            return
        app = None
        try:
            app = self.app
        except Exception:
            pass
        threading.Thread(
            target=self._run_mac_tabs,
            args=([label for label, _ in tab_cmds], [cmd for _, cmd in tab_cmds], app),
            daemon=True,
        ).start()

    def _run_mac_tabs(self, labels: list[str], cmds: list[str], app) -> None:
        """run the window opener and say in the log what got nowhere."""
        def report(message: str) -> None:
            try:
                app.call_from_thread(self._log, message)
            except Exception:
                pass

        with _MAC_TABS_LOCK:
            try:
                done = subprocess.run(
                    ["osascript", "-", *cmds],
                    input=_MAC_TABS_SCRIPT, capture_output=True, text=True,
                    timeout=60 + 20 * len(cmds),
                )
            except Exception as e:
                report(f"[red]Could not open Terminal windows: {e}[/red]")
                return

        places = [line.strip() for line in (done.stdout or "").splitlines() if line.strip()]
        if done.returncode != 0:
            detail = (done.stderr or "").strip().splitlines()
            report(f"[red]Terminal could not be told to run these: "
                   f"{detail[-1] if detail else 'osascript failed'}[/red]")
        lost = [label for i, label in enumerate(labels)
                if i >= len(places) or places[i] == "failed"]
        if lost:
            report(f"[red]No window ran: {', '.join(lost)} — start them again.[/red]")
        elif places[1:].count("window") == len(places) - 1 and len(places) > 1:
            # Terminal made no tab: allow the app running the launcher under
            # Privacy & Security > Accessibility to keep them in one window
            report("[dim]No new tabs: each component got a window of its own.[/dim]")

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
        self._log(f"[yellow]No launch method for {svc.display_name}.[/yellow]")

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
        if _undecryptable_api_key(config):
            self._log(_MLLM_KEY_UNDECRYPTABLE)
            return
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
        self._log(f"  Command: {_vllm_serve_command(config, mask=True)}")
        self._log(f"[green]{svc.display_name} tmux session '{session_name}' started on port {config['port']}.[/green]")

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
            if _infra_compose_service(svc) == "mediamtx":
                # made as this user: a missing bind source is made by docker, as root,
                # and so would artifacts/streams/, where captures and copies go too
                for sub in (SERVER_STREAMS_DIR, CAPTURE_STREAMS_DIR):
                    os.makedirs(os.path.join(self._root, ARTIFACTS_DIR, STREAMS_DIR, sub), exist_ok=True)
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
            self._log("  [yellow]If it pauses at a Password: prompt, it is auto-filled from System Settings → Sudo (or type it in the command box below and press Enter).[/yellow]")
            self._cmd.run(f'make -C {make_dir} {target} SUDO="sudo -S"')
        else:
            self._log(f"  Running: make {target}")
            subprocess.Popen(
                ["make", target, *_make_extra_vars(self._root, target)],
                cwd=make_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._log(f"[green]Uber {target} started.[/green]")

    async def _reload_gateway_after_start(self, started: str) -> None:
        """render the Gateway's config again and reload it, after a card whose
        services it routes started.

        The rendered config leaves out a machine whose name did not resolve then,
        and the reload also clears what nginx learned of the servers, so one that
        has just started is tried at once instead of after fail_timeout. Nothing
        to do while the Gateway does not run: its own Start renders it."""
        label = SYSTEM_SERVICE_LABELS["nginx"]
        host = (system_service_endpoint(self._root, "nginx") or ("", 0))[0]
        if not await asyncio.to_thread(system_service_reachable, self._root, "nginx"):
            self._log(
                f"[yellow]{label} does not answer at {host or 'its configured address'}: start it "
                f"to route {started}.[/yellow]"
            )
            return
        if is_loopback_host(host):
            target = "local"
        else:
            target = await asyncio.to_thread(target_for_service_host, host, load_ssh_profiles())
        if not target:
            self._log(
                f"[yellow]{label} runs on '{host}', which this console has no way onto: press Start on "
                f"its card there if {started} is not routed.[/yellow]"
            )
            return
        where = "locally" if target == "local" else f"on '{target}'"
        self._log(f"  Reloading {label} {where} so it routes {started}...")
        ok, said = await asyncio.to_thread(self._gateway_reload_sync, target)
        if ok:
            self._log(f"[green]{label} reloaded {where}.[/green]")
        else:
            self._log(
                f"[yellow]Could not reload {label} {where}: {said}. Press Start on its card if "
                f"{started} is not routed.[/yellow]"
            )

    def _gateway_reload_sync(self, target: str) -> tuple[bool, str]:
        """`make nginx` on the machine the Gateway is on, with a sudo prompt
        answered as the console's command line answers it: the stored local
        password here, the SSH profile's password on another machine. Returns
        whether it worked and, when it did not, the last thing it said."""
        uber_rel = "pipelines/uber-server"
        try:
            if target == "local":
                password = get_sudo_password(self._root) or ""
                result = subprocess.run(
                    ["make", "-C", os.path.join(self._root, *uber_rel.split("/")),
                     "nginx", "SUDO=sudo -S"],
                    input=f"{password}\n", capture_output=True, text=True, timeout=180,
                )
            else:
                profile = get_profile_by_name(target)
                if profile is None:
                    return False, f"no SSH profile named '{target}'"
                remote_dir = f"{profile.remote_project_path}/{uber_rel}"
                command = wrap_remote(
                    f'cd {_quote_remote_path(remote_dir)} && make nginx SUDO="sudo -S"', "uber-server"
                )
                result = subprocess.run(
                    profile.base_ssh_args() + [command],
                    input=f"{getattr(profile, 'password', '') or ''}\n",
                    capture_output=True, text=True, timeout=180,
                )
        except subprocess.TimeoutExpired:
            # never the exception itself: its repr holds the ssh arguments, which
            # begin with the password
            return False, "it did not finish in 180 s"
        except Exception as exc:
            return False, str(exc)
        if result.returncode == 0:
            return True, ""
        said = [line for line in ((result.stderr or "") + (result.stdout or "")).splitlines() if line.strip()]
        return False, said[-1].strip()[:200] if said else f"make nginx exited {result.returncode}"

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
            if value is None or isinstance(value, (list, tuple)):
                continue  # per instance: no single value stands for them all
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
                self._log(f"[red]{svc.display_name} stopped.[/red]")
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
                            f"If {svc.display_name} is running on its port, it is the native "
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
                    self._log(f"[red]{svc.display_name} stopped.[/red]")
                    kill_info = _APP_PORT_CMDS.get(make_name)
                    if kill_info:
                        port = kill_info[0]
                        if make_name == "flask":
                            port = (system_service_endpoint(self._root, "flask") or ("", port))[1]
                        self._kill_port(port, kill_info[1])
            elif svc.launch_type == "bash":
                self._log(f"[yellow]Bash-launched services must be stopped from their terminal windows.[/yellow]")
        except Exception as e:
            self._log(f"[red]Error stopping {svc.display_name}: {e}[/red]")

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
                    self._log(f"[green]{svc.display_name} launched in SSH terminal(s).[/green]")
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
                    self._log(f"[red]No compose file mapped for {svc.display_name}.[/red]")
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
                        f"[green]{svc.display_name}: containers started on {profile_name} "
                        "(services are loading; click Refresh in a moment).[/green]",
                        f"{svc.display_name} remote launch failed",
                    )
                )

            elif svc.launch_type == "vllm":
                session_name = _service_session_name(svc)
                config = _mllm_config(self._root)
                if _undecryptable_api_key(config):
                    self._log(_MLLM_KEY_UNDECRYPTABLE)
                    return
                vllm_cmd = _vllm_serve_command(config)
                run_cmd = f"cd {_quote_remote_path(remote_root)} && {vllm_cmd}; exec bash"
                wrapped_cmd = wrap_remote(run_cmd, svc.conda_env)
                cmd = (
                    f"tmux kill-session -t {session_name} 2>/dev/null; "
                    f"tmux new-session -d -s {session_name} {shlex.quote(wrapped_cmd)}; "
                    f"tmux attach -t {session_name}"
                )
                ssh_cmd = self._remote_terminal_command(profile, cmd)
                masked_cmd = _vllm_serve_command(config, mask=True)
                self._log(f"  Remote terminal: ssh {profile.ssh_destination()} {masked_cmd}")
                if self._open_collection_terminal([(svc.name, ssh_cmd)]):
                    self._log(f"  Command: {masked_cmd}")
                    self._log(f"[green]{svc.display_name} tmux session opened remotely on port {config['port']}.[/green]")
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
                if remote_platform(profile) == "darwin":
                    self._log(
                        f"  [yellow]{profile_name} is a Mac: its recorders start FFmpeg from a Terminal window "
                        "on its own screen, as macOS lets nothing started over SSH use the camera or the "
                        "microphone; the window closes by itself once FFmpeg runs. Someone has to be logged "
                        "in there, with Terminal allowed under Privacy & Security (Camera, Microphone).[/yellow]"
                    )
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
                    self._log(f"[green]{svc.display_name} launched in SSH terminal(s).[/green]")
                    self.run_worker(
                        self._reload_current_service_view(capture=False),
                        group=_LAUNCHER_UI_WORKER_GROUP,
                        exclusive=True,
                    )
                else:
                    self._log("[yellow]No remote collection components launched.[/yellow]")

            elif svc.launch_type == "make":
                # in the command session, as on Local: its shell has conda (a
                # login shell over ssh does not reach the conda init of .bashrc),
                # and a sudo prompt is answered with the SSH profile's password
                if _infra_docker_mode(svc, params):
                    # the infra compose file lives at the repo root, not in
                    # pipelines/uber-server like the Makefile does
                    compose_cmd = _compose_command(
                        _INFRA_COMPOSE_FILE, f"up -d {_infra_compose_service(svc)}"
                    )
                    if _infra_compose_service(svc) == "mediamtx":
                        # as the SSH user, before docker would make it as root
                        streams = f"{ARTIFACTS_DIR}/{STREAMS_DIR}"
                        compose_cmd = (f"mkdir -p {streams}/{SERVER_STREAMS_DIR} {streams}/{CAPTURE_STREAMS_DIR} "
                                       f"&& {compose_cmd}")
                    run_cmd = f"cd {_quote_remote_path(remote_root)} && {compose_cmd}"
                else:
                    target = _make_target_for(svc.name)
                    remote_dir = f"{remote_root}/{os.path.relpath(svc.config_dir, self._root)}"
                    extra = "".join(f" {shlex.quote(v)}" for v in _make_extra_vars(self._root, target))
                    run_cmd = f"cd {_quote_remote_path(remote_dir)} && make {shlex.quote(target)}{extra}"
                    if target in _SYSTEM_SVC_PORTS:
                        run_cmd += ' SUDO="sudo -S"'
                        self._log(
                            "  [yellow]If it pauses at a Password: prompt, it is answered with the SSH "
                            "profile's password (or type it in the command box below and press Enter).[/yellow]"
                        )
                self._log(f"  Running on '{profile_name}': {run_cmd}")
                self._cmd.run_on(profile_name, run_cmd)
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
                self._log(f"  Stopping {svc.display_name} on {profile_name}...")
                self.run_worker(
                    self._run_remote_streamed(
                        profile_name, cmd,
                        f"[green]{svc.display_name} stopped on {profile_name}.[/green]",
                        f"{svc.display_name} remote stop failed",
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
                    make_name = _make_target_for(svc.name)
                    remote_dir = f"{remote_root}/{os.path.relpath(svc.config_dir, self._root)}"
                    run_cmd = f"cd {_quote_remote_path(remote_dir)} && make {shlex.quote('stop-' + make_name)}"
                    if make_name in _SYSTEM_SVC_PORTS:
                        # systemctl needs sudo: the command session answers the
                        # prompt with the SSH profile's password, as on Local
                        self._log(f"  Stopping {svc.display_name} on {profile_name}...")
                        self._cmd.run_on(profile_name, run_cmd + ' SUDO="sudo -S"')
                        return
                self._log(f"  Stopping {svc.display_name} on {profile_name}...")
                self.run_worker(
                    self._run_remote_streamed(
                        profile_name, run_cmd,
                        f"[green]{svc.display_name} stopped on {profile_name}.[/green]",
                        f"{svc.display_name} remote stop failed",
                    ),
                    group=_LAUNCHER_REMOTE_STOP_WORKER_GROUP,
                    exclusive=False,
                )

            elif svc.launch_type == "bash":
                command = self._remote_bash_stop_command(svc)
                self._log(f"  Stopping {svc.display_name} on {profile_name}...")
                self.run_worker(
                    self._run_remote_streamed(
                        profile_name, command,
                        f"[green]{svc.display_name} stopped on {profile_name}.[/green]",
                        f"{svc.display_name} remote stop failed",
                    ),
                    group=_LAUNCHER_REMOTE_STOP_WORKER_GROUP,
                    exclusive=False,
                )
        except Exception as e:
            self._log(f"[red]Remote stop error: {e}[/red]")
