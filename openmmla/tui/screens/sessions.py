from __future__ import annotations

import asyncio

import copy
import json
import os
import shlex
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import yaml
from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, DataTable, RichLog, Button, Select, Label, ProgressBar

from openmmla.tui import base_files, recordings, stream_export
from openmmla.utils import session_sources
from openmmla.utils.artifact_paths import (
    NON_SESSION_ARTIFACT_DIRS, capture_host_label, capture_record_root, session_capture_streams_dir,
    session_server_streams_dir,
)
from openmmla.utils.constants import stream_kind


@dataclass
class SessionConfigSource:
    config_path: str
    config: dict
    label: str
    target: str
    remote_path: str = ""


def _normalize_target(value) -> str:
    if value is Select.BLANK or value is None:
        return "local"
    return str(value)


def _target_options() -> list[tuple[str, str]]:
    from openmmla.tui.ssh import target_options
    return target_options()


def _remote_path_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[part.strip("/") for part in parts if part]])


def _quote_remote_path(path: str) -> str:
    text = str(path).strip()
    if text == "~" or text == "$HOME":
        return "$HOME"
    if text.startswith("~/"):
        return _remote_path_join("$HOME", *(shlex.quote(part) for part in text[2:].split("/") if part))
    if text.startswith("$HOME/"):
        return _remote_path_join("$HOME", *(shlex.quote(part) for part in text[6:].split("/") if part))
    return shlex.quote(text)


def _has_database_config(config: dict) -> bool:
    return bool(isinstance(config, dict) and (config.get("MongoDB") or config.get("InfluxDB")))


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


def _config_for_local_access(config: dict, profile=None) -> dict:
    local_config = copy.deepcopy(config)
    if profile is None:
        return local_config
    for section in ("MongoDB", "InfluxDB"):
        section_config = local_config.get(section)
        if isinstance(section_config, dict) and "url" in section_config:
            section_config["url"] = _replace_loopback_url(section_config["url"], profile.host)
    return local_config


def _find_config_path() -> str | None:
    """find the first pipeline config.yml that contains MongoDB and InfluxDB sections."""
    from openmmla.tui.schema.loader import discover_pipelines, load_existing_config
    for pipeline in discover_pipelines():
        data = load_existing_config(pipeline.config_path)
        if _has_database_config(data):
            return pipeline.config_path
    return None


def _read_remote_config(profile, remote_path: str) -> dict:
    import yaml
    from openmmla.tui.ssh import ssh_run_sync

    quoted_path = _quote_remote_path(remote_path)
    cmd = (
        f"if [ -f {quoted_path} ]; then "
        f"cat {quoted_path}; "
        "else printf '__OPENMMLA_CONFIG_MISSING__\\n'; fi"
    )
    result = ssh_run_sync(profile, cmd, timeout=10.0)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "").strip() or f"exit code {result.returncode}")
    raw = result.stdout or ""
    if raw.strip() == "__OPENMMLA_CONFIG_MISSING__":
        return {}
    data = yaml.safe_load(raw) or {}
    return data if isinstance(data, dict) else {}


def _find_config_source(target: str) -> SessionConfigSource | None:
    from openmmla.tui.schema.loader import discover_pipelines, load_existing_config, _find_project_root
    from openmmla.tui.ssh import get_profile_by_name
    from openmmla.tui.system_services import load_system_services_config, system_services_config_path

    root = _find_project_root()
    system_config = load_system_services_config(root)
    if _has_database_config(system_config):
        return SessionConfigSource(
            config_path=system_services_config_path(root),
            config=_config_for_local_access(system_config),
            label=system_services_config_path(root),
            target=target,
        )

    pipelines = discover_pipelines()
    if target == "local":
        for pipeline in pipelines:
            config = load_existing_config(pipeline.config_path)
            if _has_database_config(config):
                return SessionConfigSource(
                    config_path=pipeline.config_path,
                    config=_config_for_local_access(config),
                    label=pipeline.config_path,
                    target=target,
                )
        return None

    profile = get_profile_by_name(target)
    if profile is None:
        raise RuntimeError(f"SSH profile '{target}' not found.")
    for pipeline in pipelines:
        rel_path = os.path.relpath(pipeline.config_path, root)
        remote_path = _remote_path_join(profile.remote_project_path, rel_path)
        config = _read_remote_config(profile, remote_path)
        if _has_database_config(config):
            return SessionConfigSource(
                config_path=pipeline.config_path,
                config=_config_for_local_access(config, profile),
                label=f"{target}:{remote_path}",
                target=target,
                remote_path=remote_path,
            )
    return None


def _local_db_config() -> dict:
    """the local config whose MongoDB/InfluxDB sections this panel reads.

    Same precedence as _find_config_source: System Settings first, then the
    first pipeline config that still carries the database sections."""
    from openmmla.tui.schema.loader import discover_pipelines, load_existing_config, _find_project_root
    from openmmla.tui.system_services import load_system_services_config

    system_config = load_system_services_config(_find_project_root())
    if _has_database_config(system_config):
        return system_config
    for pipeline in discover_pipelines():
        config = load_existing_config(pipeline.config_path)
        if _has_database_config(config):
            return config
    return {}


def _target_for_db_host(host: str, profiles: list) -> str:
    """target that serves `host`: an SSH profile name, "local" for this
    machine, or "" when no saved profile matches."""
    from openmmla.tui.system_services import target_for_service_host

    return target_for_service_host(host, profiles)


def _resolve_default_target() -> str:
    """host this panel should open on.

    The sessions live in MongoDB, so the machine its address in System
    Settings names decides, whichever host the Launcher is on: a database on
    another machine lists its sessions without a manual switch. A host known
    to be offline is never picked. Blocking: reads config files, resolves
    names and may probe the derived host."""
    from openmmla.tui.ssh import (
        TARGET_STATES, get_profile_by_name, load_ssh_profiles, probe_ssh_endpoint,
    )

    def usable(name: str) -> bool:
        return bool(name) and name != "local" and TARGET_STATES.get(name) != "offline"

    derived = _target_for_db_host(_db_host_from_config(_local_db_config()), load_ssh_profiles())
    if not usable(derived):
        return "local"
    if TARGET_STATES.get(derived) == "online":
        return derived
    # a host nobody has probed yet: check it rather than strand the panel on an
    # unreachable host it picked on the user's behalf
    profile = get_profile_by_name(derived)
    if profile is None or not probe_ssh_endpoint(profile.host, profile.port):
        return "local"
    return derived


def _write_temp_config(config: dict) -> str:
    handle = tempfile.NamedTemporaryFile(
        "w",
        suffix=".yml",
        prefix="openmmla-sessions-config-",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        yaml.safe_dump(config, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return handle.name


def _coerce_start_timestamp(value) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.timestamp()
    except ValueError:
        return 0.0


def _format_start_time(value) -> str:
    """a session's start as the table shows it. MongoDB gives a datetime; a
    session known only from its artifacts has the manifest's epoch seconds or
    ISO text, which used to be printed raw (1789654864.817)."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    stamp = _coerce_start_timestamp(value)
    if stamp > 0:
        return datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return str(value or "-")


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            if path.suffix == ".json":
                data = json.load(file)
            else:
                data = yaml.safe_load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, yaml.YAMLError, ValueError):
        return {}


def _local_artifact_sessions(root: str | os.PathLike[str]) -> list[dict]:
    artifacts_dir = Path(root) / "artifacts"
    if not artifacts_dir.is_dir():
        return []
    sessions = []
    for session_dir in sorted(artifacts_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not session_dir.is_dir() or session_dir.name in {*NON_SESSION_ARTIFACT_DIRS, ".DS_Store"}:
            continue
        manifest = _read_manifest(session_dir / "manifest.yml") or _read_manifest(session_dir / "manifest.json")
        sessions.append({
            "session_id": str(manifest.get("session_id") or session_dir.name),
            "experiment_id": manifest.get("experiment_id") or "",
            "group_id": manifest.get("group_id") or "",
            "status": manifest.get("status") or "artifact",
            "start_time": manifest.get("created_at") or manifest.get("initial_sync_time") or "",
            "_artifact_dir": str(session_dir),
            "_source_kinds": {"Artifacts"},
        })
    return sessions


def _local_collection_sessions(root: str | os.PathLike[str]) -> list[dict]:
    collection_dir = Path(root) / "collection"
    if not collection_dir.is_dir():
        return []
    sessions = []
    for session_dir in sorted(collection_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not session_dir.is_dir() or session_dir.name in {".DS_Store"}:
            continue
        manifest = _read_manifest(session_dir / "manifest.yml") or _read_manifest(session_dir / "manifest.json")
        sessions.append({
            "session_id": str(manifest.get("session_id") or session_dir.name),
            "experiment_id": manifest.get("experiment_id") or "",
            "group_id": manifest.get("group_id") or "",
            "status": manifest.get("status") or "collection",
            "start_time": manifest.get("created_at") or manifest.get("initial_sync_time") or "",
            "_artifact_dir": str(session_dir),
            "_source_kinds": {"Collection Files"},
        })
    return sessions


def _source_label(kinds: set[str], db_host: str = "") -> str:
    parts = []
    for kind in ("MongoDB", "Artifacts", "Collection Files"):
        if kind in kinds:
            if kind == "MongoDB" and db_host:
                parts.append(f"MongoDB@{db_host}")
            else:
                parts.append(kind)
    return " + ".join(parts) or "-"


def _db_host_from_config(config: dict) -> str:
    """hostname of the MongoDB endpoint actually used for the session list."""
    url = str((config.get("MongoDB") or {}).get("url") or "")
    netloc = urlsplit(url).netloc if url else ""
    return netloc.split("@")[-1].split(":")[0] if netloc else ""


def _db_endpoint_summary(config: dict) -> str:
    """e.g. 'MongoDB@ericli.local:27017 · InfluxDB@server-01:8086'."""
    parts = []
    for section, name in (("MongoDB", "MongoDB"), ("InfluxDB", "InfluxDB")):
        url = str((config.get(section) or {}).get("url") or "")
        if url:
            netloc = urlsplit(url).netloc or url
            parts.append(f"{name}@{netloc.split('@')[-1]}")
    return " · ".join(parts)


def _merge_session_rows(mongo_sessions: list[dict], artifact_sessions: list[dict], db_host: str = "") -> list[dict]:
    merged: dict[str, dict] = {}
    for session in mongo_sessions:
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            continue
        row = dict(session)
        row["_source_kinds"] = {"MongoDB"}
        merged[session_id] = row

    for session in artifact_sessions:
        session_id = str(session.get("session_id") or "").strip()
        if not session_id:
            continue
        if session_id not in merged:
            merged[session_id] = dict(session)
            continue
        row = merged[session_id]
        for key in ("experiment_id", "group_id", "status", "start_time"):
            if not row.get(key) and session.get(key):
                row[key] = session[key]
        row["_artifact_dir"] = session.get("_artifact_dir")
        kinds = set(row.get("_source_kinds") or [])
        kinds.update(session.get("_source_kinds") or [])
        row["_source_kinds"] = kinds

    rows = list(merged.values())
    for row in rows:
        row["_source"] = _source_label(set(row.get("_source_kinds") or []), db_host)
    return sorted(rows, key=lambda row: _coerce_start_timestamp(row.get("start_time")), reverse=True)


# ---- Export Streams ----

# the worker group of Export Streams, and of Export All, which ends with it: a
# group of its own, so Refresh does not stop it and Cancel stops nothing else
_STREAMS_WORKER_GROUP = "sessions-streams"

# the pipelines whose Streams a session that noted none may have taken, and the
# kind a stream of theirs is when it does not say (the card's)
_CAPTURE_PIPELINES = (("pipelines/asr-base", "audio"), ("pipelines/ips-base", "video"), ("pipelines/vfa-base", "video"))

# what the end of a session's window is (recordings.session_end), as the log says it
_WINDOW_REASONS = {
    "ended": "when it was ended",
    "left": "never ended: when its last base left",
    "running": "still running: up to now",
}


class _ExportStopped(Exception):
    """Cancel was pressed while a clip was being downloaded."""


def _shown_path(project_root, path) -> str:
    """a folder of the project as the log names it, artifacts/<session>/...,
    escaped for markup."""
    from openmmla.tui.artifacts import relative_to_root

    return escape(relative_to_root(project_root, path))


@dataclass
class _ServerSources:
    """the sources of a session as the Stream Server of System Settings sees them."""
    paths: list[str] = field(default_factory=list)                  # its paths there, each once, in the order noted
    elsewhere: list[tuple[str, str]] = field(default_factory=list)  # (stream, URL) published to another server
    by_stream: dict[str, str] = field(default_factory=dict)         # stream name -> its path there
    direct: list[str] = field(default_factory=list)                 # `asr:0 mic-1`: taken through no server at all


def _server_sources(record: dict | None, server: dict) -> _ServerSources:
    """sort the sources of a session record by where their streams went: the
    URL each base pulled is looked up on the Stream Server of System Settings
    (system_services.stream_server_path, which checks the host). Blocking: a
    host name may be resolved."""
    from openmmla.tui.system_services import hosts_match, is_loopback_host, stream_server_path

    server_host = str((server or {}).get("host") or "").strip().strip("[]")
    found = _ServerSources()
    for entry in session_sources.session_sources(record):
        name = str(entry.get("stream") or entry.get("key") or "?")
        url = str(entry.get("url") or "").strip()
        # only what a stream server serves (rtmp, rtsp, srt): udp and tcp go straight to a base
        served = session_sources.stream_url_path(url) if url else None
        if served is None:
            found.direct.append(f"{entry.get('key')} {entry.get('stream') or entry.get('source') or '?'}")
            continue
        path = stream_server_path(url, server)
        if path is None and is_loopback_host(urlsplit(url).hostname) and hosts_match(entry.get("host"), server_host):
            # localhost in the URL of a base that runs on the Stream Server's host is that server
            path = served
        if path is None:
            if (name, url) not in found.elsewhere:
                found.elsewhere.append((name, url))
            continue
        found.by_stream.setdefault(name, path)
        if path not in found.paths:
            found.paths.append(path)
    return found


def _configured_recorded_streams(project_root) -> list[stream_export.RecordedStream]:
    """the streams with Record on that the console runs (an SSH Profile set) in
    this machine's ASR, IPS and VFA configs: what a session that noted no
    streams may have taken. A stream in two configs (one camera for IPS and
    VFA) is taken once."""
    from openmmla.tui.schema.loader import load_streams

    found: list[stream_export.RecordedStream] = []
    for rel_dir, default_kind in _CAPTURE_PIPELINES:
        try:
            streams = load_streams(os.path.join(str(project_root), rel_dir, "config.yml"))
        except (TypeError, ValueError):  # a config a hand edit broke: its streams are not known
            continue
        for stream in streams:
            if not stream.ssh_profile or not stream.record:
                continue
            recorded = stream_export.RecordedStream(
                stream.name, stream.ssh_profile,
                capture_record_root(stream.ssh_profile, stream.record_root, project_root),
                capture_host_label(stream.ssh_profile), stream_kind(stream, default_kind))
            if recorded not in found:
                found.append(recorded)
    return found


class SessionsPanel(Widget):

    # seconds the Stream Server keeps a recording, as it last said; None while
    # it has not answered. 0 is for ever
    _retention: float | None = None
    # the MongoDB client of the host shown; None while none is connected
    _mongo_client = None
    # the session whose streams or base files are being exported (Export
    # Streams, Export Base Files, Export All), what Cancel sets to stop it, and
    # the button that started it; None while none is
    _streams_export_session: str | None = None
    _streams_cancel: threading.Event | None = None
    _streams_export_button: str = "Export Streams"

    class SessionDeleted(Message):
        """a session's database records were deleted here: the Launcher must
        stop offering its id, or the next recording goes to a session MongoDB
        no longer knows."""

        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    DEFAULT_CSS = """
    SessionsPanel {
        width: 1fr;
        height: 1fr;
    }
    #sessions-summary {
        height: 3;
        padding: 0 2;
        background: $panel;
        content-align: center middle;
        text-style: bold;
    }
    /* unified target bar: identical placement/style across Launcher,
       Environment, and Sessions (top of the panel, full-width select) */
    #sessions-target-bar {
        layout: horizontal;
        height: auto;
        padding: 0 1;
    }
    #sessions-target-bar Label {
        width: 10;
        padding-top: 1;
    }
    #sessions-target-select {
        width: 1fr;
    }
    #sessions-target-refresh {
        width: 5;
        min-width: 5;
        height: 3;
        margin-left: 1;
    }
    #sessions-table {
        height: 1fr;
    }
    #sessions-actions {
        height: 3;
        padding: 0 1;
    }
    #sessions-actions Button {
        margin: 0 1;
    }
    /* the progress of Export Streams, up while it runs; height auto because a
       bare Horizontal defaults to 1fr */
    #sessions-progress {
        layout: horizontal;
        height: auto;
        min-height: 1;
        padding: 0 1;
        display: none;
    }
    #sessions-progress.active {
        display: block;
    }
    #ses-progress-label {
        width: 34;
        height: auto;
        color: $text-muted;
    }
    #sessions-progress ProgressBar {
        width: 1fr;
        height: auto;
    }
    #sessions-progress Bar {
        width: 1fr;
    }
    #ses-progress-detail {
        width: 40;
        height: auto;
        color: $text-muted;
        text-align: right;
    }
    #btn-ses-export-cancel {
        width: 9;
        min-width: 9;
        height: auto;
        min-height: 1;
        margin-left: 1;
    }
    #sessions-log {
        height: 14;
        border-top: solid $primary;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._config_path: str | None = None
        self._config_source: SessionConfigSource | None = None
        self._mongo_client = None
        self._influx_client = None
        self._sessions: list[dict] = []
        self._selected_session_id: str | None = None
        self._pending_delete_session_id: str | None = None
        self._pending_delete_artifacts_session_id: str | None = None
        self._target = "local"
        self._suppress_select = False
        self._bootstrapping = False

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="sessions-target-bar"):
                yield Label("Host:")
                yield Select(_target_options(), value="local", id="sessions-target-select")
                yield Button("↻", variant="primary", compact=True, id="sessions-target-refresh")
            yield Static("Discovering database configuration...", id="sessions-summary")
            yield DataTable(id="sessions-table")
            with Horizontal(id="sessions-actions"):
                yield Button("Refresh", variant="primary", id="btn-ses-refresh")
                yield Button("Export Measurements", variant="success", id="btn-ses-export-logs")
                yield Button("Export Visualizations", variant="success", id="btn-ses-export-vis")
                # the footage of a session is a time range of the streams its bases
                # noted in its record, as the Stream Server and their capture hosts
                # recorded them
                yield Button("Export Streams", variant="success", id="btn-ses-export-streams")
                # what the bases wrote on the machines they ran on (base_files)
                yield Button("Export Base Files", variant="success", id="btn-ses-export-base-files")
                yield Button("Export All", variant="warning", id="btn-ses-export-all")
                yield Button("Delete Session", variant="error", id="btn-ses-delete")
                yield Button("Delete Artifacts", variant="error", id="btn-ses-delete-artifacts")
            with Horizontal(id="sessions-progress"):
                yield Static("", id="ses-progress-label")
                yield ProgressBar(total=None, show_eta=False, id="ses-progress-bar")
                yield Static("", id="ses-progress-detail")
                yield Button("Cancel", variant="error", compact=True, id="btn-ses-export-cancel")
            yield RichLog(id="sessions-log", highlight=True, markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.add_columns("Session ID", "Experiment", "Group", "Status", "Started", "Recordings until", "Source")
        table.cursor_type = "row"
        self._start_bootstrap()

    def on_show(self) -> None:
        self._refresh_target_options()
        # the host stays where it is: the MongoDB host from the start, or the
        # one picked here by hand; the Launcher's host has no say in it
        if self._mongo_client is None and not self._bootstrapping:
            # nothing connected: retry on the way in, as rebuilding the Host
            # options used to do as a side effect
            self.run_worker(self._async_init(self._target), exclusive=True)
        elif not self._bootstrapping and self.is_attached:
            # sessions are created and recorded in the Launcher: coming back
            # here has to show them without a press on Refresh
            self.run_worker(self._async_reload(), group="sessions-reload", exclusive=True)

    async def _async_reload(self) -> None:
        """list what the databases and the disk hold now, with the clients that
        are already connected; the query stays off the UI thread."""
        target = self._target
        rows = await asyncio.to_thread(self._gather_sessions)
        if target == self._target and not self._bootstrapping:
            self._render_sessions(rows)

    def _start_bootstrap(self) -> None:
        self._bootstrapping = True
        self.run_worker(self._async_bootstrap(), exclusive=True)

    async def _async_bootstrap(self) -> None:
        """open on the host serving MongoDB."""
        try:
            target = await asyncio.to_thread(_resolve_default_target)
            if self._apply_target(target):
                self._log(f"[cyan]Host follows the MongoDB endpoint: '{target}'.[/cyan]")
            await self._async_init(self._target)
        finally:
            self._bootstrapping = False

    def _apply_target(self, target: str) -> bool:
        """point the panel and its Host selector at `target` without
        re-entering the selector's change handler."""
        if target == self._target:
            return False
        try:
            select = self.query_one("#sessions-target-select", Select)
            if not any(value == target for _, value in _target_options()):
                return False
            self._suppress_select = True
            select.value = target
            self.call_after_refresh(self._clear_select_suppression)
        except Exception:
            self._suppress_select = False
            return False
        self._target = target
        self._config_path = None
        self._config_source = None
        return True

    def on_unmount(self) -> None:
        self._close_clients()

    async def _async_init(self, target: str) -> None:
        """initialize db clients and load sessions in a background worker."""
        import asyncio
        loop = asyncio.get_event_loop()
        self._close_clients()
        self._clear_sessions()
        self._update_summary(f"Connecting to {target} database config...")

        try:
            config_source = await loop.run_in_executor(None, _find_config_source, target)
        except Exception as e:
            self._update_summary(f"Database config lookup failed for {target}: {e}")
            return

        if target != self._target:
            return

        if not config_source:
            self._update_summary(f"No pipeline config with MongoDB/InfluxDB found for {target}.")
            self._refresh_sessions()
            return

        self._config_path = config_source.config_path
        self._config_source = config_source

        temp_path = await loop.run_in_executor(None, _write_temp_config, config_source.config)
        try:
            from openmmla.utils.client import MongoDBClientWrapper
            try:
                self._mongo_client = await loop.run_in_executor(
                    None, MongoDBClientWrapper, temp_path,
                )
            except Exception as e:
                self._mongo_client = None
                self._log(f"[yellow]MongoDB connection unavailable: {e}[/yellow]")
        except Exception as e:
            self._mongo_client = None
            self._log(f"[yellow]MongoDB client unavailable: {e}[/yellow]")

        try:
            from openmmla.utils.client import InfluxDBClientWrapper
            try:
                self._influx_client = await loop.run_in_executor(
                    None, InfluxDBClientWrapper, temp_path,
                )
            except Exception as e:
                self._influx_client = None
                self._log(f"[yellow]InfluxDB connection unavailable: {e}[/yellow]")
        except Exception as e:
            self._influx_client = None
            self._log(f"[yellow]InfluxDB client unavailable: {e}[/yellow]")
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        if self._mongo_client is None and self._influx_client is None:
            self._update_summary("Database connection unavailable; showing local sessions only.")
            self._refresh_sessions()
            return

        if target != self._target:
            self._close_clients()
            return

        self._refresh_sessions()
        self._log(f"[cyan]Database config: {config_source.label}[/cyan]")

    # ---- UI helpers ----

    def _update_summary(self, text: str) -> None:
        try:
            summary = self.query_one("#sessions-summary", Static)
            summary.update(Text(str(text)))
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        try:
            log = self.query_one("#sessions-log", RichLog)
            log.write(msg)
        except Exception:
            pass

    def _clear_sessions(self) -> None:
        self._sessions = []
        self._selected_session_id = None
        self._pending_delete_session_id = None
        self._pending_delete_artifacts_session_id = None
        try:
            self.query_one("#sessions-table", DataTable).clear()
        except Exception:
            pass

    def _close_clients(self) -> None:
        for client in (self._mongo_client, self._influx_client):
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
        self._mongo_client = None
        self._influx_client = None

    def _refresh_target_options(self) -> None:
        try:
            select = self.query_one("#sessions-target-select", Select)
            current = _normalize_target(select.value)
            options = _target_options()
            select.set_options(options)
            if any(value == current for _, value in options):
                select.value = current
                self._target = current
            else:
                select.value = "local"
                self._target = "local"
        except Exception:
            pass

    async def _async_test_single_host(self, name: str) -> None:
        from openmmla.tui.ssh import test_profile_by_name
        success, msg = await asyncio.to_thread(test_profile_by_name, name)
        self._refresh_target_options()
        color = "green" if success else "red"
        self.query_one("#sessions-log", RichLog).write(f"[{color}]'{name}': {msg}[/{color}]")

    async def _async_probe_hosts(self) -> None:
        from openmmla.tui.ssh import probe_all_profiles, summarize_states
        states = await asyncio.to_thread(probe_all_profiles)
        self._refresh_target_options()
        self.query_one("#sessions-log", RichLog).write(
            f"[green]Connection test finished: {summarize_states(states)}.[/green]"
        )

    # ---- session listing ----

    def _refresh_sessions(self) -> None:
        self._render_sessions(self._gather_sessions())

    def _gather_sessions(self) -> list[dict]:
        """the session rows of the current host: MongoDB, plus the artifacts on
        this machine when the host is Local. Blocking (a database query)."""
        from openmmla.tui.schema.loader import _find_project_root

        try:
            mongo_sessions = self._mongo_client.get_all_sessions() if self._mongo_client else []
        except Exception:
            mongo_sessions = []  # a connection that went away: the disk is still worth listing
        # how long the Stream Server keeps a recording, from the server itself:
        # the table says until when each session's footage can still be exported
        self._retention = self._ask_retention()
        artifact_sessions = []
        if self._target == "local":
            root = _find_project_root()
            artifact_sessions = _local_artifact_sessions(root) + _local_collection_sessions(root)
        db_host = _db_host_from_config(self._config_source.config) if self._config_source else ""
        return _merge_session_rows(mongo_sessions, artifact_sessions, db_host)

    def _ask_retention(self) -> float | None:
        """`recordDeleteAfter` of the running Stream Server; None when it does
        not answer. Blocking (HTTP)."""
        from openmmla.tui.schema.loader import _find_project_root
        from openmmla.tui.system_services import stream_server_address

        server = stream_server_address(_find_project_root())
        try:
            return recordings.retention(str(server.get("host") or "localhost"),
                                        int(server.get("api_port") or recordings.API_PORT), timeout=2.0)
        except recordings.RecordingsError:
            return None

    def _recordings_until(self, session: dict) -> Text | str:
        """when the Stream Server begins to delete this session's footage: its
        start plus the retention. `kept` while nothing is deleted, `gone` once
        it has passed; a session MongoDB does not know has no window to export."""
        if "MongoDB" not in set(session.get("_source_kinds") or []):
            return "-"
        start = recordings.parse_time(session.get("start_time"))
        if self._retention is None or start is None:
            return "-"
        if self._retention <= 0:
            return "kept"
        until = recordings.expiry(start, self._retention)
        now = datetime.now(timezone.utc)
        if until <= now:
            return Text("gone", style="dim")
        text = until.strftime("%Y-%m-%d %H:%M UTC")
        return Text(text, style="yellow") if until - now < timedelta(hours=24) else text

    def _render_sessions(self, sessions: list[dict]) -> None:
        self._sessions = sessions
        table = self.query_one("#sessions-table", DataTable)
        selected = self._selected_session_id
        table.clear()

        for ses in self._sessions:
            sid = ses.get("session_id", "")
            exp = ses.get("experiment_id", "")
            grp = ses.get("group_id", "")
            status = ses.get("status", "unknown")
            start = ses.get("start_time")
            start_str = _format_start_time(start)
            table.add_row(sid, exp, grp, status, start_str, self._recordings_until(ses), ses.get("_source", "-"))

        source = self._config_source.label if self._config_source else self._target
        detail = f"Config: {source}"
        if self._config_source:
            endpoints = _db_endpoint_summary(self._config_source.config)
            if endpoints:
                detail += f" | {endpoints}"
        if self._target == "local":
            detail += " + local artifacts"
        if self._retention is not None:
            detail += f" | recordings kept {recordings.describe_retention(self._retention)}"
        self._update_summary(f"Sessions: {len(self._sessions)} found | {detail}")
        # a reload must not move the cursor off the session the user was on
        ids = [ses.get("session_id", "") for ses in self._sessions]
        if selected in ids:
            table.move_cursor(row=ids.index(selected))

    # ---- event handlers ----

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#sessions-table", DataTable)
        try:
            row_data = table.get_row(event.row_key)
            self._selected_session_id = str(row_data[0])
            self._pending_delete_session_id = None
            self._pending_delete_artifacts_session_id = None
        except Exception:
            self._selected_session_id = None

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "sessions-target-select":
            return
        if self._suppress_select:
            return
        from openmmla.tui.ssh import REFRESH_TARGETS_OPTION, TARGET_STATES, is_select_sentinel
        if is_select_sentinel(event.value):
            return
        if str(event.value) == REFRESH_TARGETS_OPTION:
            self.query_one("#sessions-log", RichLog).write("[yellow]Testing connections to all hosts...[/yellow]")
            self._revert_select(event.select)
            self.run_worker(self._async_probe_hosts(), group="sessions-host-probe", exclusive=True)
            return
        target = _normalize_target(event.value)
        if target != "local" and TARGET_STATES.get(target) == "offline":
            self.query_one("#sessions-log", RichLog).write(
                f"[red]Host '{target}' is offline; staying on '{self._target}'. "
                f"Re-testing it now...[/red]"
            )
            self._revert_select(event.select)
            self.run_worker(self._async_test_single_host(target), group="sessions-host-probe", exclusive=False)
            return
        # rebuilding the options re-emits Changed for the host already shown;
        # reloading there would cancel the worker that is resolving the default
        if target == self._target:
            return
        self._target = target
        self._config_path = None
        self._config_source = None
        self.run_worker(self._async_init(target), exclusive=True)

    def _revert_select(self, select: Select) -> None:
        self._suppress_select = True
        select.value = self._target
        self.call_after_refresh(self._clear_select_suppression)

    def _clear_select_suppression(self) -> None:
        self._suppress_select = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "sessions-target-refresh":
            self.query_one("#sessions-log", RichLog).write("[yellow]Testing connections to all hosts...[/yellow]")
            self.run_worker(self._async_probe_hosts(), group="sessions-host-probe", exclusive=True)
            return
        if bid == "btn-ses-refresh":
            self.run_worker(self._async_init(self._target), exclusive=True)
            return
        if bid == "btn-ses-export-cancel":
            self._cancel_streams_export()
            return

        if not self._selected_session_id:
            self._log("[yellow]Select a session row first.[/yellow]")
            return

        session_id = self._selected_session_id

        if bid == "btn-ses-export-logs":
            self.run_worker(self._run_export(session_id, logs=True, vis=False), exclusive=True)
        elif bid == "btn-ses-export-vis":
            self.run_worker(self._run_export(session_id, logs=True, vis=True), exclusive=True)
        elif bid == "btn-ses-export-all":
            self._start_streams_export(session_id, self._run_export_all, "Export All · measurements")
        elif bid == "btn-ses-export-streams":
            self._start_streams_export(session_id, self._run_export_streams, "Export Streams")
        elif bid == "btn-ses-export-base-files":
            self._start_streams_export(session_id, self._run_export_base_files, "Export Base Files")
        elif bid == "btn-ses-delete":
            if self._pending_delete_session_id != session_id:
                self._pending_delete_session_id = session_id
                self._pending_delete_artifacts_session_id = None
                self._log(
                    f"[red]Delete session '{session_id}' will remove MongoDB metadata, "
                    "and InfluxDB measurements. Local artifacts and downloaded logs are preserved. "
                    "Click Delete Session again to confirm.[/red]"
                )
                return
            self._pending_delete_session_id = None
            self.run_worker(self._run_delete(session_id), exclusive=True)
        elif bid == "btn-ses-delete-artifacts":
            if self._pending_delete_artifacts_session_id != session_id:
                self._pending_delete_artifacts_session_id = session_id
                self._pending_delete_session_id = None
                self._log(
                    f"[red]Delete artifacts for '{session_id}' will remove local artifacts/files only. "
                    "MongoDB metadata and InfluxDB measurements are preserved. "
                    "Click Delete Artifacts again to confirm.[/red]"
                )
                return
            self._pending_delete_artifacts_session_id = None
            self.run_worker(self._run_delete_artifacts(session_id), exclusive=True)

    # ---- export logic ----

    async def _run_export(self, session_id: str, *, logs: bool, vis: bool) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._do_export, session_id, logs, vis)

    async def _run_export_all(self, session_id: str) -> None:
        """Export All: the measurements and visualizations, then the streams
        and the base files."""
        await self._run_export(session_id, logs=True, vis=True)
        await self._run_export_streams(session_id)
        if self._streams_cancel is not None and self._streams_cancel.is_set():
            raise asyncio.CancelledError()
        await self._run_export_base_files(session_id)

    # ---- Export Streams ----

    def _start_streams_export(self, session_id: str, run, label: str) -> None:
        """start Export Streams, Export Base Files, or Export All, which ends
        with both, in a worker of its own group with the progress row up, so
        its Cancel can be reached the whole time; one at a time."""
        if self._streams_export_session is not None:
            self._log(
                f"[yellow]{self._streams_export_button} of '{escape(self._streams_export_session)}' is still "
                f"running. Wait for it to finish, or press Cancel next to its progress.[/yellow]"
            )
            return
        self._streams_export_session = session_id
        self._streams_cancel = threading.Event()
        self._streams_export_button = label.split(" · ")[0]
        self._progress_show(label)
        self.run_worker(self._streams_worker(session_id, run), group=_STREAMS_WORKER_GROUP, exclusive=False)

    async def _streams_worker(self, session_id: str, run) -> None:
        try:
            await run(session_id)
        except asyncio.CancelledError:
            if self._streams_cancel is not None:
                self._streams_cancel.set()  # a clip downloading in a thread stops at its next chunk
            staged = ("" if self._streams_export_button == "Export Base Files" else
                      ", and cuts made on a capture host stay there until they are fetched")
            self._log(
                f"[yellow]The export of '{escape(session_id)}' stopped. What arrived is kept{staged}: press "
                f"{self._streams_export_button} again to go on.[/yellow]"
            )
            raise
        except Exception as error:  # a failure is this export's, not the app's
            self._log(f"[red]✗ The export of '{escape(session_id)}' failed: {escape(str(error))}[/red]")
        finally:
            self._progress_hide()
            self._streams_export_session = None
            self._streams_cancel = None

    def _cancel_streams_export(self) -> None:
        """Cancel next to the progress: stop the export of streams that runs."""
        if self._streams_export_session is None:
            return
        if self._streams_cancel is not None:
            self._streams_cancel.set()
        self._log("[yellow]Stopping the export...[/yellow]")
        self.workers.cancel_group(self, _STREAMS_WORKER_GROUP)

    def _progress_show(self, label: str) -> None:
        """put the progress row up, its bar busy until a transfer gives a size."""
        try:
            self._progress_start(label, None)
            self.query_one("#sessions-progress", Horizontal).add_class("active")
        except Exception:
            pass

    def _progress_start(self, label: str, total: int | None) -> None:
        try:
            bar = self.query_one("#ses-progress-bar", ProgressBar)
            bar.update(total=None)
            bar.update(total=float(total) if total else None, progress=0)
            self.query_one("#ses-progress-label", Static).update(Text(label))
            self.query_one("#ses-progress-detail", Static).update("")
        except Exception:
            pass

    def _progress_update(self, done: int, total: int, detail: str) -> None:
        """bytes so far of a transfer of `total` bytes (0: a size nobody knows,
        the bar stays busy), and a `12.1 MB · 3.2 MB/s` detail."""
        try:
            if total:
                bar = self.query_one("#ses-progress-bar", ProgressBar)
                if bar.total != float(total):
                    bar.update(total=float(total))
                bar.update(progress=min(max(0, int(done)), int(total)))
            self.query_one("#ses-progress-detail", Static).update(Text(detail))
        except Exception:
            pass

    def _progress_end(self) -> None:
        """a transfer is over; the row stays up for as long as the export runs."""
        self._progress_start(self._streams_export_button, None)

    def _progress_hide(self) -> None:
        try:
            self.query_one("#sessions-progress", Horizontal).remove_class("active")
            self.query_one("#ses-progress-bar", ProgressBar).update(total=None, progress=0)
            self.query_one("#ses-progress-detail", Static).update("")
        except Exception:
            pass

    def _session_record(self, session_id: str) -> tuple[dict, bool]:
        """the session's document as MongoDB holds it now, and whether it is the
        table's row instead. A base notes the stream it takes when it joins,
        which can be after the table was listed, so the row may not have it
        yet: it stands in only when MongoDB does not give the document back
        (True). Empty for a session MongoDB never knew. Blocking (a query)."""
        row = self._session_by_id(session_id)
        if self._mongo_client is not None:
            try:
                record = self._mongo_client.get_session(session_id)
            except Exception:
                record = None
            if isinstance(record, dict) and record:
                return record, False
        if "MongoDB" in set(row.get("_source_kinds") or []):
            return row, True
        return {}, False

    async def _run_export_streams(self, session_id: str) -> None:
        """Export Streams: both copies of the session's part of every stream it
        used, from its `sources` (read again from MongoDB at the press).

        A stream is shared by the sessions that pull it and is recorded whether
        or not one runs: by the Stream Server (every path published to it) and,
        with Record on, by the machine that captures it. Nothing of either
        belongs to a session; each base notes in the session's document the
        stream it takes (openmmla.utils.session_sources), and the session's
        start and end (recordings.session_end) pick the part of it:
          - the server copy, over HTTP: each noted path on the Stream Server of
            System Settings, one file per unbroken stretch, into
            artifacts/<session>/streams/server/<app>/<name>_<start>.mp4;
          - the capture copy, over SSH: each noted stream's recording, cut on
            its capture host and fetched (stream_export), into
            artifacts/<session>/streams/capture/<host label>/<video|audio>/.
        A session without that note (one begun before the bases wrote it) takes
        every path the server recorded while it ran and every stream with Record
        on in this machine's pipeline configs, and says so. Cancel stops it
        between steps (a clip being downloaded at once); what arrived stays."""
        from openmmla.tui.schema.loader import _find_project_root
        from openmmla.tui.system_services import stream_server_address

        cancel = self._streams_cancel or threading.Event()
        shown = escape(session_id)
        self._log(f"[bold]Exporting the streams of session: {shown}[/bold]")
        self._progress_start("Export Streams · reading the session", None)
        record, stale = await asyncio.to_thread(self._session_record, session_id)
        if stale:
            self._log(
                f"  [dim]Could not read '{shown}' from MongoDB again, so this goes by the table as it was last "
                f"listed, which may not have the streams its bases noted since. Refresh and export again to be "
                f"sure.[/dim]"
            )
        start = recordings.parse_time(record.get("start_time"))
        if start is None:
            self._log(
                f"  [yellow]'{shown}' has no start time in MongoDB (it is known from its artifacts only), so there "
                f"is no time range to cut out of the streams.[/yellow]"
            )
            return
        end, why = recordings.session_end(record)
        until = f"{end:%H:%M:%S}" if end.date() == start.date() else f"{end:%Y-%m-%d %H:%M:%S}"
        self._log(f"  {start:%Y-%m-%d %H:%M:%S} to {until} UTC ({_WINDOW_REASONS[why]})")
        if not session_sources.session_sources(record):
            self._log(
                f"  [yellow]'{shown}' does not say which streams its bases took (it began before the console noted "
                f"them, or no base noted its stream in it), so this takes every path the Stream Server recorded "
                f"while it ran, and every stream with Record on in this machine's ASR, IPS and VFA configs.[/yellow]"
            )

        root = _find_project_root()
        server = await asyncio.to_thread(stream_server_address, root)
        sources = await asyncio.to_thread(_server_sources, record, server)
        from_server = await self._export_server_copy(
            root, session_id, record, server, sources, (start, end), why == "ended", cancel)
        if cancel.is_set():
            raise asyncio.CancelledError()
        from_capture = await self._export_capture_copy(root, session_id, record, sources, (start, end), cancel)

        folder = _shown_path(root, session_server_streams_dir(root, session_id).parent)
        if from_server or from_capture:
            self._log(
                f"[bold green]Streams of {shown} exported: {from_server} file(s) from the Stream Server, "
                f"{from_capture} from the capture hosts, under {folder}[/bold green]\n"
                f"  [dim]File names carry the time they start at, so a base replays them with source: file and "
                f"Base.file_dir on one of these folders.[/dim]"
            )
        else:
            self._log(f"[yellow]Nothing of the streams of {shown} was exported (see above).[/yellow]")

    async def _export_server_copy(
        self, root, session_id: str, record: dict, server: dict, sources: _ServerSources,
        window: tuple[datetime, datetime], ended: bool, cancel: threading.Event,
    ) -> int:
        """the server copy: each unbroken stretch of the session's paths on the
        Stream Server within its window, asked of the playback server over HTTP
        (no SSH). A clip already here in full is not fetched again. How many
        clips are here afterwards."""
        from openmmla.tui.artifacts import copy_covers, ensure_session_layout

        start, end = window
        shown = escape(session_id)
        host = str(server.get("host") or "localhost")
        api_port = int(server.get("api_port") or recordings.API_PORT)
        playback_port = int(server.get("playback_port") or recordings.PLAYBACK_PORT)
        out_dir = session_server_streams_dir(root, session_id)
        noted = bool(session_sources.session_sources(record))
        self._log(f"[cyan]From the Stream Server {escape(host)} into {_shown_path(root, out_dir)}[/cyan]")
        for name, url in sources.elsewhere:
            self._log(
                f"  [yellow]- {escape(name)}: published to another server ({escape(url)}), not to the Stream "
                f"Server of System Settings, so it is skipped here.[/yellow]"
            )
        paths = list(sources.paths)
        if noted:
            if not paths:
                if not sources.elsewhere:
                    self._log(
                        f"  [yellow]None of the bases of '{shown}' took a stream through the Stream Server "
                        f"({escape(', '.join(sources.direct))}), so it holds nothing of this session.[/yellow]"
                    )
                return 0
            self._log(f"  Its streams, as its bases noted them: {escape(', '.join(paths))}")

        def ask() -> dict:
            # the session's own paths need no listing: the playback server is
            # asked for those, as they are named; without them, for everything
            listed = paths if noted else recordings.recorded_paths(host, api_port)
            return {path: recordings.timespans(host, path, playback_port) for path in listed}

        self._progress_start(f"Stream Server · asking {host}", None)
        try:
            spans = await asyncio.to_thread(ask)
        except recordings.RecordingsError as error:
            self._log(
                f"  [red]✗ The Stream Server does not answer: {escape(str(error))}[/red]\n"
                f"  [dim]Its host and its API/playback ports are under Launcher → System Settings → Stream Server; "
                f"the API and the playback server are switched on in its mediamtx.yml.[/dim]"
            )
            return 0
        finally:
            self._progress_end()
        clips = recordings.clips_for_window(spans, start, end)
        if not clips:
            expired = recordings.expiry(start, self._retention)
            if expired is not None and expired <= datetime.now(timezone.utc):
                self._log(
                    f"  [yellow]Nothing left on the server: it keeps a recording for "
                    f"{recordings.describe_retention(self._retention)}, and this session's footage was deleted "
                    f"from {expired:%Y-%m-%d %H:%M} UTC on. The retention is set on the Stream Server card, "
                    f"Config tab; its Recordings tab shows what the server still holds.[/yellow]"
                )
                return 0
            if noted:
                held = [path for path in paths if spans.get(path)]
                found = ("Nothing of its streams was recorded on the server in that time"
                         + (f" (it holds {', '.join(held)} from other times)." if held
                            else " (it holds no recording of them)."))
            else:
                held = list(spans)
                found = ("Nothing was recorded on the server in that time"
                         + (f" (it holds {', '.join(held[:6])}{' ...' if len(held) > 6 else ''})." if held
                            else " (it holds no recording at all)."))
            self._log(
                f"  [yellow]{escape(found)} Server-side recording is the switch on the Stream Server card, "
                f"Config tab.[/yellow]"
            )
            return 0

        await asyncio.to_thread(ensure_session_layout, root, session_id)
        loop = asyncio.get_running_loop()
        done = 0
        for index, clip in enumerate(clips, 1):
            if cancel.is_set():
                raise asyncio.CancelledError()
            destination = out_dir / recordings.clip_relpath(clip)
            label = escape(f"{clip.path}  {clip.start:%H:%M:%S} +{clip.duration:.0f}s")
            size_here = destination.stat().st_size if destination.is_file() else 0
            if size_here:
                # a clip is named after its start only, so a copy exported while
                # the session was still going looks like the full one: its length
                # tells. Without ffprobe here, an ended session's clip counts as final
                covered = await asyncio.to_thread(copy_covers, destination, clip.duration)
                if covered or (covered is None and ended):
                    self._log(f"  [dim]- {label}: already exported[/dim]")
                    done += 1
                    continue
                if covered is False:
                    self._log(
                        f"  [cyan]{label}: the copy here stops short (exported while the session was still "
                        f"going); fetching it in full[/cyan]"
                    )
            self._progress_start(f"Stream Server · {index} of {len(clips)}", None)
            try:
                size = await asyncio.to_thread(
                    recordings.download_clip, host, clip, str(destination), playback_port,
                    progress=self._clip_progress(loop, clip, cancel))
            except recordings.RecordingsError as error:
                self._log(f"  [red]✗ {label}: {escape(str(error))}[/red]")
                continue
            except _ExportStopped:
                raise asyncio.CancelledError() from None
            finally:
                self._progress_end()
            done += 1
            self._log(f"  [green]✓[/green] {label} -> {escape(recordings.clip_relpath(clip))} ({size / 1e6:.1f} MB)")
        self._log(f"  [green]{done} of {len(clips)} clip(s) are under {_shown_path(root, out_dir)}[/green]")
        return done

    def _clip_progress(self, loop, clip: recordings.Clip, cancel: threading.Event):
        """the progress callback of one clip's download, which runs in its
        thread: it stops the download once Cancel was pressed, and moves the
        progress row on the app's loop a few times a second."""
        began = time.monotonic()
        shown = [0.0]

        def progress(written: int) -> None:
            if cancel.is_set():
                raise _ExportStopped()
            now = time.monotonic()
            if now - shown[0] < 0.25:
                return
            shown[0] = now
            rate = written / max(now - began, 1e-3)
            detail = f"{clip.path} · {recordings.human_size(written)} · {recordings.human_size(rate)}/s"
            try:
                loop.call_soon_threadsafe(self._progress_update, written, 0, detail)
            except RuntimeError:  # the app's loop is closed
                pass

        return progress

    async def _export_capture_copy(
        self, root, session_id: str, record: dict, sources: _ServerSources,
        window: tuple[datetime, datetime], cancel: threading.Event,
    ) -> int:
        """the capture copy: each noted stream's recording on the machine that
        captured it, cut there to the window without re-encoding and fetched
        (stream_export.export_session, which says what it does line by line).
        How many cuts are here afterwards."""
        start, end = window
        folder = session_capture_streams_dir(root, session_id)
        self._log(f"[cyan]From the capture hosts into {_shown_path(root, folder)}[/cyan]")
        found = stream_export.session_streams(record, root)
        if found.noted:
            for name, why in found.skipped:
                reason = ("someone else publishes it (it has no SSH Profile), so no capture host of this console "
                          "records it" if why == "external" else
                          "Record was off for it, so its capture host holds no recording of it")
                path = sources.by_stream.get(name)
                other = next((url for stream, url in sources.elsewhere if stream == name), None)
                if path:
                    where = f"on the Stream Server ({escape(path)}, above)"
                elif other:
                    where = f"on the server it was published to ({escape(other)})"
                else:
                    where = "on the Stream Server, if it went through one"
                self._log(f"  [dim]- {escape(name)}: {reason}; its only copy is {where}[/dim]")
            streams = found.streams
            if not streams:
                if not found.skipped:
                    self._log(
                        "  [dim]- Its bases took no stream the console captures (a camera or microphone of their "
                        "own, or a file), so there is nothing to cut.[/dim]"
                    )
                return 0
        else:
            streams = await asyncio.to_thread(_configured_recorded_streams, root)
            if not streams:
                self._log(
                    "  [dim]- No stream in this machine's ASR, IPS or VFA config records on its capture host "
                    "(Record on), so there is nothing to cut.[/dim]"
                )
                return 0
            self._log(f"  Streams with Record on here: {escape(', '.join(stream.name for stream in streams))}")

        callbacks = stream_export.ExportCallbacks(
            log=self._log,
            progress_start=self._progress_start,
            progress_update=self._progress_update,
            progress_end=self._progress_end,
            cancelled=cancel.is_set,
        )
        self._progress_start("Capture hosts · cutting", None)
        try:
            result = await stream_export.export_session(
                root, session_id, streams, start.timestamp(), end.timestamp(), callbacks)
        finally:
            self._progress_end()
        if result.here:
            self._log(
                f"  [green]{result.here} cut(s) are under {_shown_path(root, result.folder)}"
                + (f" ({result.present} of them were already here)" if result.present else "")
                + "[/green]"
            )
        if result.unfetched:
            hosts = ", ".join(dict.fromkeys(result.unfetched))
            self._log(
                f"  [yellow]The cuts made on {escape(hosts)} did not all arrive. They stay there: press Export "
                f"Streams again to fetch them.[/yellow]"
            )
        elif not result.here and all(cut.listed and not cut.failed for cut in result.cuts):
            self._log(
                "  [yellow]None of these streams was being recorded on its capture host during the "
                "session.[/yellow]"
            )
        return result.here

    # ---- Export Base Files ----

    async def _run_export_base_files(self, session_id: str) -> None:
        """Export Base Files: what the session's bases, synchronizers and IPS
        visualizer wrote on the machines they ran on (their logs, the config
        they ran with, what they recorded), from the host of every SSH profile,
        into artifacts/<session>/pipelines/<pipeline>/<host>/ (base_files).
        Those run on this machine wrote there in the first place. Cancel stops
        it between folders (a transfer at once); what arrived stays, and a
        transfer cut short resumes at the next press."""
        from openmmla.tui.schema.loader import _find_project_root
        from openmmla.tui.ssh import load_ssh_profiles

        cancel = self._streams_cancel or threading.Event()
        shown = escape(session_id)
        root = _find_project_root()
        folder = _shown_path(root, base_files.session_pipelines_dir(root, session_id))
        self._log(f"[bold]Exporting the base files of session: {shown}[/bold]")
        self._progress_start("Export Base Files · reading the session", None)
        record, _stale = await asyncio.to_thread(self._session_record, session_id)
        profiles = await asyncio.to_thread(load_ssh_profiles)
        here = await asyncio.to_thread(base_files.local_parts, root, session_id)
        if here:
            self._log(f"  [dim]This machine's own are in {folder} already: {escape(', '.join(here))}[/dim]")
        if not profiles:
            self._log(
                "  [yellow]No SSH profile to ask: a base run on another machine keeps its files there. Add that "
                "machine under System Settings → Hosts → SSH Profiles.[/yellow]"
            )
            return
        self._log(f"[cyan]Asking {len(profiles)} SSH host(s) what they hold of it, into {folder}[/cyan]")
        callbacks = stream_export.ExportCallbacks(
            log=self._log,
            progress_start=self._progress_start,
            progress_update=self._progress_update,
            progress_end=self._progress_end,
            cancelled=cancel.is_set,
        )
        result = await base_files.export_session(root, session_id, profiles, callbacks, record=record)
        for host, keys in result.missing.items():
            self._log(
                f"  [yellow]{escape(', '.join(keys))} ran on {escape(host)}, which no SSH profile reached: its "
                f"files stay there.[/yellow]"
            )
        if result.fetched:
            parts = ", ".join(f"{part} from {profile}" for profile, part in result.fetched)
            self._log(f"[bold green]Base files of {shown} exported: {escape(parts)}, under {folder}[/bold green]")
        if result.incomplete:
            self._log(
                f"[yellow]Not all of it arrived from {escape(', '.join(result.incomplete))} (see above). What was "
                f"staged is kept: press Export Base Files again to go on.[/yellow]"
            )
        elif not result.fetched:
            self._log(f"[yellow]No SSH host holds base files of {shown}.[/yellow]")

    async def _run_delete(self, session_id: str) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._do_delete, session_id)
        self.post_message(self.SessionDeleted(session_id))
        self._refresh_sessions()

    async def _run_delete_artifacts(self, session_id: str) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._do_delete_artifacts, session_id)
        self._refresh_sessions()

    def _session_by_id(self, session_id: str) -> dict:
        for session in self._sessions:
            if str(session.get("session_id") or "") == session_id:
                return session
        return {}

    def _do_delete(self, session_id: str) -> None:
        from openmmla.tui.artifacts import artifact_session_dir
        from openmmla.tui.schema.loader import _find_project_root

        session = self._session_by_id(session_id)
        self._log(f"[bold red]Deleting session: {session_id}[/bold red]")

        if self._influx_client is not None:
            try:
                if self._influx_client.delete_session_data(session_id):
                    self._log("  [green]✓[/green] InfluxDB measurements deleted")
                else:
                    self._log("  [yellow]- InfluxDB measurements not deleted or not found[/yellow]")
            except Exception as e:
                self._log(f"  [red]✗ InfluxDB delete failed: {e}[/red]")
        else:
            self._log("  [dim]- InfluxDB not connected[/dim]")

        if self._mongo_client is not None:
            try:
                if self._mongo_client.delete_session(session_id):
                    self._log("  [green]✓[/green] MongoDB session deleted")
                else:
                    self._log("  [yellow]- MongoDB session not found[/yellow]")
            except Exception as e:
                self._log(f"  [red]✗ MongoDB delete failed: {e}[/red]")
        else:
            self._log("  [dim]- MongoDB not connected[/dim]")

        artifact_dir = Path(session.get("_artifact_dir") or artifact_session_dir(_find_project_root(), session_id))
        if artifact_dir.exists():
            self._log(f"  [dim]- Local artifacts preserved: {artifact_dir}[/dim]")
        else:
            self._log("  [dim]- No local artifact directory[/dim]")

        self._log(f"[bold green]Delete complete for {session_id}[/bold green]")

    def _local_artifact_path_for_session(self, session_id: str) -> Path:
        from openmmla.tui.artifacts import artifact_session_dir
        from openmmla.tui.schema.loader import _find_project_root

        session = self._session_by_id(session_id)
        if session.get("_artifact_dir"):
            return Path(str(session["_artifact_dir"])).expanduser().resolve()
        return artifact_session_dir(_find_project_root(), session_id).resolve()

    @staticmethod
    def _is_safe_local_artifact_path(path: Path, project_root: Path) -> bool:
        allowed_roots = [
            (project_root / "artifacts").resolve(),
            (project_root / "collection").resolve(),
        ]
        try:
            return any(path == root or path.is_relative_to(root) for root in allowed_roots)
        except ValueError:
            return False

    def _do_delete_artifacts(self, session_id: str) -> None:
        from openmmla.tui.schema.loader import _find_project_root

        project_root = Path(_find_project_root()).resolve()
        artifact_path = self._local_artifact_path_for_session(session_id)
        self._log(f"[bold red]Deleting local artifacts for session: {session_id}[/bold red]")

        if not self._is_safe_local_artifact_path(artifact_path, project_root):
            self._log(f"  [red]✗ Refusing to delete path outside local artifact roots: {artifact_path}[/red]")
            return
        if not artifact_path.exists():
            self._log(f"  [yellow]- No local artifact directory found: {artifact_path}[/yellow]")
            return
        if not artifact_path.is_dir():
            self._log(f"  [red]✗ Refusing to delete non-directory artifact path: {artifact_path}[/red]")
            return

        try:
            shutil.rmtree(artifact_path)
            self._log(f"  [green]✓[/green] Local artifacts deleted: {artifact_path}")
            self._log(f"[bold green]Artifact delete complete for {session_id}[/bold green]")
        except Exception as e:
            self._log(f"  [red]✗ Local artifact delete failed: {e}[/red]")

    def _do_export(self, session_id: str, logs: bool, vis: bool) -> None:
        import time
        import yaml
        from openmmla.tui.artifacts import artifact_session_dir, ensure_session_layout, relative_to_root
        from openmmla.tui.schema.loader import _find_project_root
        from openmmla.utils.constants import (
            EVENT_TYPE_ASR_RECOGNITION, EVENT_TYPE_ASR_TRANSCRIPTION,
            EVENT_TYPE_VFA_ACTION,
            EVENT_TYPE_IPS_TRANSLATION, EVENT_TYPE_IPS_ROTATION, EVENT_TYPE_IPS_RELATION,
        )
        from openmmla.utils.querys import fetch_and_process_data, save_to_json_file

        root = _find_project_root()
        ensure_session_layout(root, session_id)
        session_dir = artifact_session_dir(root, session_id)
        measurements_dir = os.path.join(session_dir, "measurements")
        analysis_dir = os.path.join(session_dir, "analysis")
        features_dir = os.path.join(analysis_dir, "features")
        vis_dir = os.path.join(analysis_dir, "visualizations")

        self._log(f"[bold]Exporting session: {session_id}[/bold]")
        self._log(f"Output directory: {session_dir}")

        # ---- export measurements ----
        if logs:
            os.makedirs(measurements_dir, exist_ok=True)

            event_types = {
                "ASR Recognition": (EVENT_TYPE_ASR_RECOGNITION, "speaker_recognition"),
                "ASR Transcription": (EVENT_TYPE_ASR_TRANSCRIPTION, "speaker_transcription"),
                "VFA Action": (EVENT_TYPE_VFA_ACTION, "action_recognition"),
                "IPS Translation": (EVENT_TYPE_IPS_TRANSLATION, "badge_translation"),
                "IPS Rotation": (EVENT_TYPE_IPS_ROTATION, "badge_rotation"),
                "IPS Relation": (EVENT_TYPE_IPS_RELATION, "badge_relation"),
            }

            exported_files: dict[str, str] = {}
            for label, (evt, suffix) in event_types.items():
                try:
                    data = fetch_and_process_data(session_id, evt, self._influx_client)
                    if data:
                        path = save_to_json_file(session_id, data, suffix, measurements_dir)
                        exported_files[label] = path
                        self._log(f"  [green]✓[/green] {label}: {len(data)} records -> {os.path.basename(path)}")
                    else:
                        self._log(f"  [dim]- {label}: no data[/dim]")
                except Exception as e:
                    self._log(f"  [red]✗ {label}: {e}[/red]")

            # convert transcription to txt
            if "ASR Transcription" in exported_files:
                try:
                    from openmmla.analytics.asr.transcription import convert_transcription_json_to_txt
                    convert_transcription_json_to_txt(exported_files["ASR Transcription"])
                    self._log("  [green]✓[/green] ASR Transcription -> .txt")
                except Exception as e:
                    self._log(f"  [red]✗ Transcription txt conversion: {e}[/red]")

        # ---- export visualizations ----
        if vis:
            os.makedirs(vis_dir, exist_ok=True)

            # ASR visualizations
            recognition_path = os.path.join(measurements_dir, f"{session_id}_speaker_recognition.json")
            if os.path.isfile(recognition_path):
                try:
                    from openmmla.analytics.asr.analyze import (
                        plot_speaker_diarization_interactive,
                        plot_speaking_interaction_network,
                    )
                    plot_speaker_diarization_interactive(recognition_path, vis_dir)
                    self._log("  [green]✓[/green] ASR: speaker diarization chart")
                    plot_speaking_interaction_network(recognition_path, vis_dir)
                    self._log("  [green]✓[/green] ASR: speaking interaction network")
                except Exception as e:
                    self._log(f"  [red]✗ ASR visualizations: {e}[/red]")
            else:
                self._log("  [dim]- ASR visualizations: no recognition log file[/dim]")

            # IPS visualizations
            translation_path = os.path.join(measurements_dir, f"{session_id}_badge_translation.json")
            relation_path = os.path.join(measurements_dir, f"{session_id}_badge_relation.json")
            if os.path.isfile(translation_path):
                try:
                    from openmmla.analytics.ips.analyze import (
                        plot_badge_locations_and_trajectories,
                        plot_2d_heatmap,
                        plot_physical_interaction_network,
                    )
                    plot_badge_locations_and_trajectories(translation_path, vis_dir)
                    self._log("  [green]✓[/green] IPS: badge locations and trajectories")
                    plot_2d_heatmap(translation_path, vis_dir)
                    self._log("  [green]✓[/green] IPS: 2D heatmap")
                    if os.path.isfile(relation_path):
                        plot_physical_interaction_network(relation_path, vis_dir)
                        self._log("  [green]✓[/green] IPS: physical interaction network")
                except Exception as e:
                    self._log(f"  [red]✗ IPS visualizations: {e}[/red]")
            else:
                self._log("  [dim]- IPS visualizations: no translation log file[/dim]")

            # VFA has no visualizations currently
            self._log("  [dim]- VFA visualizations: not available[/dim]")

        manifest_path = os.path.join(session_dir, "manifest.yml")
        manifest = {}
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as file:
                    loaded_manifest = yaml.safe_load(file) or {}
                    if isinstance(loaded_manifest, dict):
                        manifest = loaded_manifest
            except yaml.YAMLError:
                manifest = {}
        manifest["session_id"] = session_id
        manifest.pop("visualizations_path", None)
        if self._config_source:
            manifest["database_source"] = {
                "target": self._config_source.target,
                "config": self._config_source.label,
            }
        if logs:
            manifest["measurements_path"] = relative_to_root(root, measurements_dir)
        analysis = manifest.get("analysis")
        if not isinstance(analysis, dict):
            analysis = {}
        analysis["features_path"] = relative_to_root(root, features_dir)
        analysis["visualizations_path"] = relative_to_root(root, vis_dir)
        if vis:
            analysis["visualizations_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["analysis"] = analysis
        with open(manifest_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(manifest, file, sort_keys=False, allow_unicode=True)

        self._log(f"[bold green]Export complete for {session_id}[/bold green]")
