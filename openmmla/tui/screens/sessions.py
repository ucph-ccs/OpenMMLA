from __future__ import annotations

import copy
import json
import os
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import yaml
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widget import Widget
from textual.widgets import Static, DataTable, RichLog, Button, Select, Label

from openmmla.utils.artifact_paths import NON_SESSION_ARTIFACT_DIRS


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
    from openmmla.tui.ssh import load_ssh_profiles
    return [("Local", "local")] + [(profile.name, profile.name) for profile in load_ssh_profiles()]


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


def _source_label(kinds: set[str]) -> str:
    return " + ".join(kind for kind in ("MongoDB", "Artifacts", "Collection Files") if kind in kinds) or "-"


def _merge_session_rows(mongo_sessions: list[dict], artifact_sessions: list[dict]) -> list[dict]:
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
        row["_source"] = _source_label(set(row.get("_source_kinds") or []))
    return sorted(rows, key=lambda row: _coerce_start_timestamp(row.get("start_time")), reverse=True)


class SessionsPanel(Widget):

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
    #sessions-target-bar {
        height: 3;
        padding: 0 1;
        background: $panel;
        align: center middle;
    }
    #sessions-target-bar Label {
        width: 10;
    }
    #sessions-target-select {
        width: 40;
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

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="sessions-target-bar"):
                yield Label("Target:")
                yield Select(_target_options(), value="local", id="sessions-target-select")
            yield Static("Discovering database configuration...", id="sessions-summary")
            yield DataTable(id="sessions-table")
            with Horizontal(id="sessions-actions"):
                yield Button("Refresh", variant="primary", id="btn-ses-refresh")
                yield Button("Export Measurements", variant="success", id="btn-ses-export-logs")
                yield Button("Export Visualizations", variant="success", id="btn-ses-export-vis")
                yield Button("Export All", variant="warning", id="btn-ses-export-all")
                yield Button("Delete Session", variant="error", id="btn-ses-delete")
                yield Button("Delete Artifacts", variant="error", id="btn-ses-delete-artifacts")
            yield RichLog(id="sessions-log", highlight=True, markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.add_columns("Session ID", "Experiment", "Group", "Status", "Started", "Source")
        table.cursor_type = "row"
        self.run_worker(self._async_init(self._target), exclusive=True)

    def on_show(self) -> None:
        self._refresh_target_options()

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

    # ---- session listing ----

    def _refresh_sessions(self) -> None:
        from openmmla.tui.schema.loader import _find_project_root

        mongo_sessions = self._mongo_client.get_all_sessions() if self._mongo_client else []
        artifact_sessions = []
        if self._target == "local":
            root = _find_project_root()
            artifact_sessions = _local_artifact_sessions(root) + _local_collection_sessions(root)
        self._sessions = _merge_session_rows(mongo_sessions, artifact_sessions)
        table = self.query_one("#sessions-table", DataTable)
        table.clear()

        for ses in self._sessions:
            sid = ses.get("session_id", "")
            exp = ses.get("experiment_id", "")
            grp = ses.get("group_id", "")
            status = ses.get("status", "unknown")
            start = ses.get("start_time")
            start_str = start.strftime("%Y-%m-%d %H:%M UTC") if isinstance(start, datetime) else str(start or "-")
            table.add_row(sid, exp, grp, status, start_str, ses.get("_source", "-"))

        source = self._config_source.label if self._config_source else self._target
        detail = f"Source: {source}"
        if self._target == "local":
            detail += " + local artifacts"
        self._update_summary(f"Sessions: {len(self._sessions)} found | {detail}")

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
        target = _normalize_target(event.value)
        if target == self._target and self._mongo_client is not None:
            return
        self._target = target
        self._config_path = None
        self._config_source = None
        self.run_worker(self._async_init(target), exclusive=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-ses-refresh":
            self.run_worker(self._async_init(self._target), exclusive=True)
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
            self.run_worker(self._run_export(session_id, logs=True, vis=True), exclusive=True)
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

    async def _run_delete(self, session_id: str) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._do_delete, session_id)
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
