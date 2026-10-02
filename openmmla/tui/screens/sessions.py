from __future__ import annotations

import asyncio

import copy
import json
import os
import shlex
import shutil
import tempfile
import threading
from dataclasses import dataclass
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

from openmmla.tui import recordings, stream_export
from openmmla.utils import session_sources
from openmmla.utils.artifact_paths import NON_SESSION_ARTIFACT_DIRS, safe_segment, short_hostname


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


def _database_url_set(config: dict, section: str) -> bool:
    """whether a config says where MongoDB or InfluxDB is: its url is filled
    in (an unfilled mongodb://<uber-server>:27017 names no machine)."""
    from openmmla.tui.system_services import section_address_set

    return isinstance(config, dict) and section_address_set(config.get(section), section)


def _has_database_config(config: dict) -> bool:
    return _database_url_set(config, "MongoDB") or _database_url_set(config, "InfluxDB")


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


def _sealed_here(config: dict, profile) -> dict:
    """another host's config as this machine reads it: each ENC(...) value,
    sealed with that host's own master key, opened with it (read over ssh, in
    memory only) and sealed again with this machine's; one this machine's
    key opens already, or none opens, stays as it is."""
    from openmmla.utils import crypto

    if not crypto.enc_tokens(config):
        return config
    from openmmla.tui.screens.launcher import _local_master_key, _read_host_master_key

    host_key, _ = _read_host_master_key(profile)
    local_key = _local_master_key()
    opened, _ = crypto.plan_reseal(config, local_key, [host_key])
    if not opened:
        return config
    try:
        return crypto.apply_reseal(config, opened, local_key or crypto.ensure_master_key())
    except Exception:
        return config  # no key here to seal with: the values stay as they came


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
                config=_sealed_here(_config_for_local_access(config, profile), profile),
                label=f"{target}:{remote_path}",
                target=target,
                remote_path=remote_path,
            )
    return None


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


# ---- Export and Archive ----

# the worker group of Export and of Archive: a group of its own, so Refresh
# does not stop them and Cancel stops nothing else
_JOB_WORKER_GROUP = "sessions-transfer"


# what the end End Session writes is (_end_by_hand), as the log says it
_END_REASONS = {
    "left": "when its last base left",
    "measurement": "its last measurement in InfluxDB",
    "source": "when a base last joined or left it",
    "log": "when a log of it was last written on this machine",
    "now": "now: neither its record, InfluxDB nor a log here says when it stopped",
}


def _end_by_hand(record: dict, last_measurement: datetime | None, now: datetime | None = None,
                 last_log: datetime | None = None) -> tuple[datetime, str]:
    """the end_time End Session writes into a session that was never ended,
    and what it is (_END_REASONS). When every base that joined it has left,
    that moment, which is where Export already took it to end, so the window
    it cuts stays the same; otherwise the latest moment the session is
    known to have been alive: its last measurement, a base joining or leaving,
    or a log its components wrote on this machine; now when nothing says. A
    time from before the session began (a replay stamps its measurements with
    the original time, a copied file keeps its own) says nothing about its end."""
    try:
        left = recordings.parse_time(session_sources.last_left(record))
    except (TypeError, ValueError):  # left_at of kinds that do not compare
        left = None
    if left is not None:
        return left, "left"
    start = recordings.parse_time(record.get("start_time"))
    moments = []
    for moment, why in ((last_measurement, "measurement"), (last_log, "log")):
        if moment is not None and (start is None or moment >= start):
            moments.append((moment, why))
    for entry in session_sources.session_sources(record):
        for key in ("joined_at", "left_at"):
            moment = recordings.parse_time(entry.get(key))
            if moment is not None:
                moments.append((moment, "source"))
    if moments:
        return max(moments, key=lambda moment: moment[0])
    return now or datetime.now(timezone.utc), "now"


def _last_local_log(artifact_dir: Path) -> datetime | None:
    """when a log the session's components wrote on this machine was last
    written (pipelines/<pipeline>/<this host>/logger/ of its artifacts); None
    without one. The folders of other hosts are left out: a log fetched from
    one without rsync carries the time it arrived."""
    times = []
    for path in Path(artifact_dir).glob(f"pipelines/*/{short_hostname()}/logger/*"):
        try:
            if path.is_file():
                times.append(path.stat().st_mtime)
        except OSError:
            continue
    return datetime.fromtimestamp(max(times), tz=timezone.utc) if times else None


class SessionsPanel(Widget):

    # seconds the Stream Server keeps a recording, as it last said; None while
    # it has not answered. 0 is for ever
    _retention: float | None = None
    # the MongoDB client of the host shown; None while none is connected
    _mongo_client = None
    # the session being exported or archived, what Cancel sets to stop it,
    # and the button that started it; None while none is
    _job_session: str | None = None
    _job_cancel: threading.Event | None = None
    _job_button: str = "Export"
    # database clients let go of (Refresh, a Host change) while a job held
    # them: closed once it is over
    _clients_after_job: tuple = ()

    class SessionEnded(Message):
        """End Session marked a session ended here: the Launcher's base cards
        must stop opening on it."""

        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

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
    /* a short label keeps its own width, so the whole row fits 86 columns */
    #sessions-actions Button {
        margin: 0 1;
        min-width: 10;
    }
    /* below that, the buttons take two lines of three (_fit_actions) */
    #sessions-actions.-two-lines {
        layout: grid;
        grid-size: 3;
        grid-rows: 3;
        height: 6;
    }
    #sessions-actions.-two-lines Button {
        width: 1fr;
    }
    /* the progress of Export and Archive, up while one runs; height auto
       because a bare Horizontal defaults to 1fr */
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
        # the session End Session was pressed on once, and the end it announced
        self._pending_end: tuple[str, datetime] | None = None
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
                # everything of the session gathered onto this console (mmla ses-export):
                # its measurements, recordings, streams and base files
                yield Button("Export", variant="success", id="btn-ses-export")
                # the console's copy sent to the System Settings host (mmla ses-archive)
                yield Button("Archive", variant="success", id="btn-ses-archive")
                # a session left active (its console gone before Stop) set to ended
                yield Button("End Session", variant="warning", id="btn-ses-end")
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

    def on_resize(self, event) -> None:
        self._fit_actions(event.size.width)

    def _fit_actions(self, width: int) -> None:
        """the action row on one line when it holds every button (each its
        label and a cell either side, at least 10; a cell between two; the
        row's padding and outer margins), else on two."""
        try:
            row = self.query_one("#sessions-actions", Horizontal)
        except Exception:
            return
        buttons = list(row.query(Button))
        needed = sum(max(10, len(str(button.label)) + 2) for button in buttons) + len(buttons) + 3
        row.set_class(width < needed, "-two-lines")

    def on_show(self) -> None:
        self._refresh_target_options()
        # the host stays where it is: this machine, or the one picked here by
        # hand; the Launcher's host has no say in it
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
        """open on this machine.

        Whichever host is picked, the databases are the ones System Settings
        names, so following the MongoDB address only ever moved the panel off
        the one host that also lists what lies in this checkout's artifacts/."""
        try:
            await self._async_init(self._target)
        finally:
            self._bootstrapping = False

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
            from openmmla.tui.system_services import unset_address_note

            self._update_summary(
                f"No MongoDB or InfluxDB address for {target}: {unset_address_note('MongoDB')}, and no "
                f"pipeline config there says where it is.")
            self._refresh_sessions()
            return

        self._config_path = config_source.config_path
        self._config_source = config_source

        temp_path = await loop.run_in_executor(None, _write_temp_config, config_source.config)
        from openmmla.tui.system_services import unset_address_note

        # a database whose url is not filled in is not asked: there is no machine to ask
        unset = [name for name in ("MongoDB", "InfluxDB") if not _database_url_set(config_source.config, name)]
        for name in unset:
            self._log(f"[yellow]{name} not connected: {unset_address_note(name)}.[/yellow]")
        try:
            from openmmla.utils.client import MongoDBClientWrapper
            try:
                if "MongoDB" not in unset:
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
                if "InfluxDB" not in unset:
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
        self._pending_end = None
        try:
            self.query_one("#sessions-table", DataTable).clear()
        except Exception:
            pass

    def _close_clients(self) -> None:
        clients = tuple(client for client in (self._mongo_client, self._influx_client) if client is not None)
        self._mongo_client = None
        self._influx_client = None
        if self._job_session is not None:
            # the Export or Archive that runs is still querying them
            self._clients_after_job += clients
            return
        for client in clients:
            try:
                client.close()
            except Exception:
                pass

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
        if not server.get("host"):
            return None  # System Settings have no address for it yet
        try:
            return recordings.retention(str(server.get("host")),
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
            self._pending_end = None
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
            self._cancel_job()
            return

        if not self._selected_session_id:
            self._log("[yellow]Select a session row first.[/yellow]")
            return

        session_id = self._selected_session_id

        if bid == "btn-ses-export":
            self._start_export(session_id)
        elif bid == "btn-ses-archive":
            self._start_archive(session_id)
        elif bid == "btn-ses-end":
            self._pending_delete_session_id = None
            self._pending_delete_artifacts_session_id = None
            pending, self._pending_end = self._pending_end, None
            if pending is not None and pending[0] == session_id:
                self.run_worker(self._run_end(session_id, pending[1]), group="sessions-end", exclusive=True)
            else:
                self.run_worker(self._run_end_ask(session_id), group="sessions-end", exclusive=True)
        elif bid == "btn-ses-delete":
            if self._pending_delete_session_id != session_id:
                self._pending_delete_session_id = session_id
                self._pending_delete_artifacts_session_id = None
                self._pending_end = None
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
                self._pending_end = None
                self._log(
                    f"[red]Delete artifacts for '{session_id}' will remove local artifacts/files only. "
                    "MongoDB metadata and InfluxDB measurements are preserved. "
                    "Click Delete Artifacts again to confirm.[/red]"
                )
                return
            self._pending_delete_artifacts_session_id = None
            self.run_worker(self._run_delete_artifacts(session_id), exclusive=True)

    # ---- Export ----

    def _start_export(self, session_id: str) -> None:
        """Export: everything of the session gathered onto this console (mmla
        ses-export), with the progress row and its Cancel. A row without a
        session id that names one folder has nothing to export."""
        if not str(session_id or "").strip() or safe_segment(session_id, "") != session_id:
            self._log(f"[yellow]'{escape(str(session_id))}' is no session id that names one folder: nothing to "
                      f"export.[/yellow]")
            return
        self._start_job(session_id, self._run_export, "Export")

    async def _run_export(self, session_id: str) -> None:
        """what mmla ses-export does, in this worker, with the session's
        document as MongoDB holds it now (else the table's row), the InfluxDB
        the table was listed from, and the Stream Server's retention as it
        last said."""
        from openmmla.commands.ses import export

        record, stale = await asyncio.to_thread(self._session_record, session_id)
        # the table's own keys (_source_kinds, a set) are no part of the session's document
        record = {key: value for key, value in record.items() if not str(key).startswith("_")}
        if stale:
            self._log(
                f"  [dim]Could not read '{escape(session_id)}' from MongoDB again, so this goes by the table as it "
                f"was last listed, which may not have the streams its bases noted since. Refresh and export again "
                f"to be sure.[/dim]"
            )
        source = None
        if self._config_source:
            source = {"target": self._config_source.target, "config": self._config_source.label}
        await export.export_session(
            session_id, callbacks=self._job_callbacks(), record=record or None, influx=self._influx_client,
            retention=self._retention, database_source=source)

    def _job_callbacks(self) -> stream_export.ExportCallbacks:
        """the log lines and the progress of Export and Archive go to this
        panel's log and its progress row; Cancel there stops them."""
        cancel = self._job_cancel or threading.Event()
        return stream_export.ExportCallbacks(
            log=self._log,
            progress_start=self._progress_start,
            progress_update=self._progress_update,
            progress_end=self._progress_end,
            cancelled=cancel.is_set,
        )

    def _start_job(self, session_id: str, run, label: str) -> None:
        """start Export or Archive in a worker of its own group with the
        progress row up, so its Cancel can be reached the whole time; one at a
        time."""
        if self._job_session is not None:
            self._log(
                f"[yellow]{self._job_button} of '{escape(self._job_session)}' is still running. Wait for it to "
                f"finish, or press Cancel next to its progress.[/yellow]"
            )
            return
        self._job_session = session_id
        self._job_cancel = threading.Event()
        self._job_button = label
        self._progress_show(label)
        self.run_worker(self._job_worker(session_id, run), group=_JOB_WORKER_GROUP, exclusive=False)

    async def _job_worker(self, session_id: str, run) -> None:
        try:
            await run(session_id)
        except asyncio.CancelledError:
            if self._job_cancel is not None:
                self._job_cancel.set()  # a clip downloading in a thread stops at its next chunk
            staged = ("" if self._job_button == "Archive" else
                      ", and cuts made on a capture host stay there until they are fetched")
            self._log(
                f"[yellow]The {self._job_button.lower()} of '{escape(session_id)}' stopped. What arrived is "
                f"kept{staged}: press {self._job_button} again to go on.[/yellow]"
            )
            raise
        except Exception as error:  # a failure is this job's, not the app's
            self._log(f"[red]✗ The {self._job_button.lower()} of '{escape(session_id)}' failed: "
                      f"{escape(str(error))}[/red]")
        finally:
            self._progress_hide()
            self._job_session = None
            self._job_cancel = None
            clients, self._clients_after_job = self._clients_after_job, ()
            for client in clients:
                try:
                    client.close()
                except Exception:
                    pass

    def _cancel_job(self) -> None:
        """Cancel next to the progress: stop the Export or Archive that runs."""
        if self._job_session is None:
            return
        if self._job_cancel is not None:
            self._job_cancel.set()
        self._log(f"[yellow]Stopping the {self._job_button.lower()}...[/yellow]")
        self.workers.cancel_group(self, _JOB_WORKER_GROUP)

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
        """a transfer is over; the row stays up for as long as the job runs."""
        self._progress_start(self._job_button, None)

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

    # ---- Archive ----

    def _start_archive(self, session_id: str) -> None:
        """Archive: send the session's raw files from this checkout to the
        System Settings host (mmla ses-archive), with the progress row and
        Cancel of the exports. A session with no folder here has nothing to
        send yet."""
        from openmmla.commands.ses import archive

        if not archive.local_session_dir(session_id).is_dir():
            self._log(f"[yellow]{escape(archive.no_local_folder(session_id))}.[/yellow]")
            return
        self._start_job(session_id, self._run_archive, "Archive")

    async def _run_archive(self, session_id: str) -> None:
        """what mmla ses-archive does, in this worker, with the MongoDB the
        table was listed from."""
        from openmmla.commands.ses import archive

        try:
            await archive.archive_session(session_id, callbacks=self._job_callbacks(), mongo=self._mongo_client)
        except archive.ArchiveError as error:
            self._log(f"[red]✗ The archive of '{escape(session_id)}' was not made: {escape(str(error))}[/red]")

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

    # ---- End Session ----

    async def _run_end_ask(self, session_id: str) -> None:
        """End Session, first press: read the session's document again and say
        the end the second press writes; nothing changes yet."""
        shown = escape(session_id)
        record = await self._endable_record(session_id)
        if record is None:
            return
        last_measurement = None
        if self._influx_client is not None:
            try:
                last_measurement = await asyncio.to_thread(self._influx_client.last_event_time, session_id)
            except Exception as error:
                self._log(f"  [dim]InfluxDB did not say when its last measurement was: {escape(str(error))}[/dim]")
        last_log = await asyncio.to_thread(_last_local_log, self._local_artifact_path_for_session(session_id))
        end, why = _end_by_hand(record, last_measurement, last_log=last_log)
        # the conclusion first: the log does not wrap
        self._log(
            f"[yellow]Click End Session again to mark '{shown}' ended at {end:%Y-%m-%d %H:%M:%S} UTC, "
            f"{_END_REASONS[why]}.[/yellow]"
        )
        still_in = [str(entry.get("key") or "?") for entry in session_sources.session_sources(record)
                    if entry.get("left_at") is None]
        if still_in:
            self._log(
                f"  [yellow]Never noted leaving: {escape(', '.join(still_in))}. If still running, stop it "
                f"instead (STOP in Session Control), which ends the session too.[/yellow]"
            )
        self._pending_end = (session_id, end)

    async def _run_end(self, session_id: str, end: datetime) -> None:
        """End Session, second press: the end announced at the first."""
        shown = escape(session_id)
        if await self._endable_record(session_id) is None:
            return
        if await asyncio.to_thread(self._mongo_client.end_session, session_id, end):
            self._log(f"[green]✓ Session '{shown}' marked ended at {end:%Y-%m-%d %H:%M:%S} UTC.[/green]")
            # its paths on the Stream Server stop recording, as at STOP
            from openmmla.tui.schema.loader import _find_project_root
            from openmmla.tui.system_services import system_services_config_path
            from openmmla.utils.stream_recording import end_session_recording

            recording = await asyncio.to_thread(
                end_session_recording, session_id, mongo_db=self._mongo_client, now=end,
                start_path=str(system_services_config_path(_find_project_root())))
            if recording["text"]:
                color = "yellow" if recording["warnings"] else "dim"
                self._log(f"  [{color}]{escape(recording['text'])}[/{color}]")
            self.post_message(self.SessionEnded(session_id))
        else:
            self._log(f"[red]✗ Session '{shown}' could not be marked ended: MongoDB did not take it.[/red]")
        await self._async_reload()

    async def _endable_record(self, session_id: str) -> dict | None:
        """the session's MongoDB document when End Session can end it; None,
        and the reason in the log, when it is not there or has ended already."""
        shown = escape(session_id)
        if self._mongo_client is None:
            self._log(f"[yellow]MongoDB is not connected: '{shown}' cannot be ended here.[/yellow]")
            return None
        record = await asyncio.to_thread(self._mongo_client.get_session, session_id)
        if not record:
            self._log(
                f"[yellow]'{shown}' is not in MongoDB (it is known from its files only, or MongoDB did not "
                f"answer): it has no status to change.[/yellow]"
            )
            return None
        if record.get("status") == "ended":
            ended = recordings.parse_time(record.get("end_time"))
            when = f" at {ended:%Y-%m-%d %H:%M:%S} UTC" if ended else ""
            self._log(f"[yellow]'{shown}' has already ended{when}.[/yellow]")
            return None
        return record

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
