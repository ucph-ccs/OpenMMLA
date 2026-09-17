from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static, DataTable, RichLog, Button

from openmmla.tui.schema.loader import _find_project_root
from openmmla.tui.system_services import (
    SYSTEM_SERVICE_LABELS,
    hosts_match, is_loopback_host, is_this_machine, system_service_endpoint, system_service_reachable,
)
from openmmla.tui.ssh import load_ssh_profiles, probe_ssh_endpoint, ssh_check_port, ssh_check_tmux, ssh_run_sync, get_profile_by_name, SSHProfile


KNOWN_SERVICES = [
    {"name": "InfluxDB", "port": 8086, "type": "system", "target": "influxdb"},
    {"name": "MongoDB", "port": 27017, "type": "system", "target": "mongodb"},
    {"name": "Redis", "port": 6379, "type": "system", "target": "redis"},
    {"name": SYSTEM_SERVICE_LABELS["mosquitto"], "port": 1883, "type": "system", "target": "mosquitto"},
    {"name": SYSTEM_SERVICE_LABELS["nginx"], "port": 8080, "type": "system", "target": "nginx"},
    {"name": SYSTEM_SERVICE_LABELS["mediamtx"], "port": 1935, "type": "system", "target": "mediamtx"},
    {"name": SYSTEM_SERVICE_LABELS["flask"], "port": 5050, "type": "tmux", "session": "flask", "target": "flask"},
    {"name": SYSTEM_SERVICE_LABELS["celery"], "port": None, "type": "tmux", "session": "celery", "target": "celery"},
]
# the ASR / VFA servers and the MLLM server are not in this list: the Launcher
# knows which host each of them was put on, and status_rows() asks it


def _check_port(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """check if a port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


def _profile_for_host(host: str) -> SSHProfile | None:
    """the SSH profile whose host is `host`, when the Host column names a
    machine rather than a profile."""
    for profile in load_ssh_profiles():
        if hosts_match(profile.host, host):
            return profile
    return None


def _infra_container_logs_local(svc_key: str) -> str:
    """logs of the docker infra container for a system service, or "" when the
    compose project has no container for it (bare-metal deployment)."""
    from openmmla.tui.screens.launcher import _INFRA_COMPOSE_FILE, _INFRA_COMPOSE_SERVICES
    service = _INFRA_COMPOSE_SERVICES.get(svc_key)
    compose = os.path.join(_find_project_root(), _INFRA_COMPOSE_FILE)
    if not service or not os.path.isfile(compose):
        return ""
    try:
        ps = subprocess.run(
            ["docker", "compose", "-f", compose, "ps", "-a", "-q", service],
            capture_output=True, text=True, timeout=10,
        )
        if not ps.stdout.strip():
            return ""
        logs = subprocess.run(
            ["docker", "compose", "-f", compose, "logs", "--tail", "80", "--no-color", service],
            capture_output=True, text=True, timeout=15,
        )
        return logs.stdout or logs.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _infra_container_logs_remote_cmd(svc_key: str, remote_root: str, fallback: str) -> str:
    """shell that prints the docker infra container's logs when one exists on
    the remote host, else runs the bare-metal log command."""
    from openmmla.tui.screens.launcher import _INFRA_COMPOSE_FILE, _INFRA_COMPOSE_SERVICES
    service = _INFRA_COMPOSE_SERVICES.get(svc_key)
    if not service:
        return fallback
    compose = f"{remote_root.rstrip('/')}/{_INFRA_COMPOSE_FILE}"
    return (
        f'if [ -n "$(docker compose -f {compose} ps -a -q {service} 2>/dev/null)" ]; '
        f"then docker compose -f {compose} logs --tail 80 --no-color {service}; "
        f"else {fallback}; fi"
    )


def _list_tmux_sessions() -> dict[str, str]:
    """return dict of { session_name: creation_time_str }."""
    sessions = {}
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}|#{session_created}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.split("|", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        ts = int(parts[1].strip())
                        created = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                    except (ValueError, OSError):
                        created = parts[1].strip()
                    sessions[name] = created
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return sessions


def _capture_tmux_pane(session_name: str, lines: int = 50) -> str:
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
        return "(tmux not available)"


class StatusPanel(Widget):

    DEFAULT_CSS = """
    StatusPanel {
        width: 1fr;
        height: 1fr;
    }
    #status-summary {
        height: 3;
        padding: 0 2;
        background: $panel;
        content-align: center middle;
        text-style: bold;
    }
    #status-table {
        height: 1fr;
    }
    #status-actions {
        height: 3;
        padding: 0 1;
    }
    #status-actions Button {
        margin: 0 1;
    }
    #log-panel {
        height: 14;
        border-top: solid $primary;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._tmux_sessions: dict[str, str] = {}
        self._selected_row: tuple[str, str, str, str] | None = None  # (name, host, status, tmux session)
        # row key -> tmux session name, for View Logs (the cell shows more than the name)
        self._row_sessions: dict = {}
        self._ssh_profiles: list[SSHProfile] = []
        self._summary_text: str = ""
        # a stopped service gets a row only when System Settings put it on a
        # named machine (its being down matters); "Show all" lists the rest
        self._show_all: bool = False

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Loading status...", id="status-summary")
            yield DataTable(id="status-table")
            from textual.containers import Horizontal
            with Horizontal(id="status-actions"):
                yield Button("Refresh", variant="primary", id="btn-refresh")
                yield Button("View Logs", variant="default", id="btn-view-logs")
                yield Button("Show all", variant="default", id="btn-show-all")
            yield RichLog(id="log-panel", highlight=True, markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#status-table", DataTable)
        # "tmux", not "Session": a session is a recording everywhere else in the
        # console, and this column is only the tmux session a service runs in
        table.add_columns("Service", "Host", "Status", "Port", "tmux")
        table.cursor_type = "row"
        self._is_visible = False
        self._schedule_refresh()

    def on_show(self) -> None:
        self._is_visible = True
        self._refresh_status()

    def on_hide(self) -> None:
        self._is_visible = False

    def _schedule_refresh(self) -> None:
        self.set_timer(5.0, self._auto_refresh)

    def _auto_refresh(self) -> None:
        # don't probe ports/tmux while the Status tab is hidden
        if getattr(self, "_is_visible", False):
            self._refresh_status(include_remote=False)
        self._schedule_refresh()

    def _refresh_status(self, include_remote: bool = False) -> None:
        """refresh service status off the UI thread; remote checks only when requested."""
        self.run_worker(
            self._async_refresh_status(include_remote),
            group="status-refresh",
            exclusive=True,
        )

    def _launcher(self):
        """the Launcher panel of this console, which knows every service's host."""
        try:
            from openmmla.tui.screens.launcher import ServicePanel
            return self.app.query_one(ServicePanel)
        except Exception:
            return None

    async def _async_refresh_status(self, include_remote: bool) -> None:
        launcher = self._launcher()
        if launcher is not None:
            tmux_sessions, profiles, rows, running_count, hidden = await asyncio.to_thread(
                self._gather_launcher_status, launcher, include_remote, self._show_all
            )
            summary_text = f"Services: {running_count} running"
            if hidden:
                summary_text += f" · {hidden} stopped (Show all)"
        else:
            tmux_sessions, profiles, rows, running_count, port_count = await asyncio.to_thread(
                self._gather_local_status
            )
            summary_text = f"Services: {running_count} running | Ports: {port_count} active"
        self._tmux_sessions = tmux_sessions
        self._ssh_profiles = profiles

        table = self.query_one("#status-table", DataTable)
        table.clear()
        self._row_sessions.clear()
        for row in rows:
            self._add_row(table, *row)

        self._summary_text = f"{summary_text} | tmux: {len(tmux_sessions)} | SSH profiles: {len(profiles)}"
        summary = self.query_one("#status-summary", Static)
        summary.update(self._summary_text)

        if include_remote and profiles:
            self.run_worker(self._refresh_remote_status(), exclusive=True)

    @staticmethod
    def _gather_launcher_status(launcher, use_ssh: bool, show_all: bool) -> tuple[dict, list, list[tuple], int, int]:
        """rows from the Launcher's own picture of the deployment (worker thread):
        each service on the host its card is on, probed like its sidebar marker."""
        tmux_sessions = _list_tmux_sessions()
        profiles = load_ssh_profiles()
        rows: list[tuple] = []
        running_count = hidden = 0
        listed_sessions: set[str] = set()
        # the system services first, in the order of the Launcher's tree
        entries = sorted(launcher.status_rows(use_ssh=use_ssh), key=lambda entry: not entry["make_target"])
        for entry in entries:
            running, counts = entry["running"], entry["counts"]
            if entry["session"]:
                listed_sessions.add(entry["session"])
            if running:
                running_count += 1
            elif not (entry["placed"] or show_all):
                hidden += 1
                continue
            if running is None:
                status = "? (Refresh)"  # its host is only asked on a Refresh
            elif counts is not None:
                status = f"Running {counts[0]}/{counts[1]}" if counts[0] else "Stopped"
            else:
                status = "Running" if running else "Stopped"
            session = entry["session"]
            started = tmux_sessions.get(session, "-") if session and entry["target"] == "local" else "-"
            rows.append((entry["name"], entry["host"], status, entry["port"], session or "-", started))

        for session_name in sorted(set(tmux_sessions) - listed_sessions):
            if session_name in ("asr-services", "vfa-services"):
                continue
            rows.append((f"[tmux] {session_name}", "local", "Running", "-", session_name, tmux_sessions[session_name]))
            running_count += 1
        return tmux_sessions, profiles, rows, running_count, hidden

    @staticmethod
    def _gather_local_status() -> tuple[dict[str, str], list, list[tuple], int, int]:
        """collect all local probe results (runs in a worker thread)."""
        tmux_sessions = _list_tmux_sessions()
        profiles = load_ssh_profiles()
        root = _find_project_root()
        rows: list[tuple] = []
        running_count = 0
        port_count = 0

        for svc in KNOWN_SERVICES:
            name = svc["name"]
            port = svc.get("port")
            session = svc.get("session", "")

            host_label = "local"
            configured = svc["type"] == "system" or svc.get("target") == "flask"
            if configured:
                # the address pipelines (or the browser) are configured with,
                # probed from here; the Host column names that machine, not
                # where the TUI runs
                host, port = system_service_endpoint(root, svc["target"]) or ("", port)
                port_ok = system_service_reachable(root, svc["target"])
                port_label = str(port)
                if not is_loopback_host(host):
                    host_label = host
            else:
                port_ok = _check_port(port) if port else False
                port_label = str(port) if port else "-"
            session_ok = session in tmux_sessions if session else False
            is_up = port_ok if configured else session_ok

            if is_up:
                running_count += 1
            if port_ok:
                port_count += 1

            rows.append((
                name,
                host_label,
                "Running" if is_up else "Stopped",
                port_label,
                session if session else "-",
                tmux_sessions.get(session, "-") if session else "-",
            ))

        extra_sessions = set(tmux_sessions.keys()) - {
            s.get("session", "") for s in KNOWN_SERVICES if s.get("session")
        }
        for session_name in sorted(extra_sessions):
            if session_name in ("asr-services", "vfa-services"):
                continue
            rows.append((
                f"[tmux] {session_name}", "local", "Running", "-",
                session_name, tmux_sessions[session_name],
            ))
            running_count += 1

        return tmux_sessions, profiles, rows, running_count, port_count

    async def _refresh_remote_status(self) -> None:
        """check remote services in background (slow SSH calls).

        Hosts whose SSH port is unreachable are skipped after a 1s TCP probe
        instead of paying a full SSH timeout per service check."""
        table = self.query_one("#status-table", DataTable)
        log = self.query_one("#log-panel", RichLog)
        added = 0
        root = _find_project_root()

        reachable = []
        for profile in self._ssh_profiles:
            ok = await asyncio.to_thread(probe_ssh_endpoint, profile.host, profile.port)
            if ok:
                reachable.append(profile)
            else:
                log.write(f"[yellow]Skipping '{profile.name}' ({profile.host}): SSH port unreachable.[/yellow]")

        shown = {(str(table.get_row(key)[0]), str(table.get_row(key)[1])) for key in table.rows}
        for profile in reachable:
            from openmmla.tui.ssh import TARGET_PLATFORMS
            if TARGET_PLATFORMS.get(profile.name) == "windows":
                log.write(f"[yellow]Skipping '{profile.name}': it runs Windows, which the console cannot ask.[/yellow]")
                continue
            log.write(f"Checking services on '{profile.name}'...")
            for svc in KNOWN_SERVICES:
                port = svc.get("port")
                session = svc.get("session", "")

                r_port_ok = False
                r_session_ok = False

                loop = asyncio.get_event_loop()
                if svc["type"] == "system" or svc.get("target") == "flask":
                    host, _ = system_service_endpoint(root, svc["target"]) or ("", None)
                    if not is_loopback_host(host):
                        # one configured address for everyone: the local row
                        # already shows it as host:port, a per-host copy would
                        # just repeat the same probe result
                        continue
                    # a loopback url means this host's own port, asked over ssh
                    r_port_ok = await asyncio.to_thread(
                        system_service_reachable, root, svc["target"],
                        lambda p, _profile=profile: ssh_check_port(_profile, p),
                    )
                    port = (system_service_endpoint(root, svc["target"]) or ("", port))[1]
                elif port:
                    r_port_ok = await loop.run_in_executor(
                        None, ssh_check_port, profile, port,
                    )
                if session:
                    r_session_ok = await loop.run_in_executor(
                        None, ssh_check_tmux, profile, session,
                    )

                if svc["type"] == "system":
                    r_up = r_port_ok
                else:
                    r_up = r_session_ok or r_port_ok

                if not r_up or (svc["name"], profile.name) in shown:
                    continue

                if svc["type"] == "system":
                    # only loopback-configured services reach here
                    _, eport = system_service_endpoint(root, svc["target"]) or ("", port)
                    port_str = str(eport)
                else:
                    port_str = str(port) if port else "-"
                self._add_row(table, svc["name"], profile.name, "Running", port_str, session or "-", "-")
                added += 1

        if added > 0:
            summary = self.query_one("#status-summary", Static)
            summary.update(f"{self._summary_text} (+{added} remote)")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-show-all":
            self._show_all = not self._show_all
            event.button.label = "Running only" if self._show_all else "Show all"
            self._refresh_status(include_remote=False)
        elif event.button.id == "btn-refresh":
            self._refresh_status(include_remote=True)
        elif event.button.id == "btn-view-logs":
            self._view_selected_logs()

    def _add_row(self, table: DataTable, name: str, host: str, status: str, port: str,
                 session: str, started: str) -> None:
        """one table row. The last column says which tmux session a service
        runs in and since when; services that are brew, systemd or docker
        processes have none and show "-"."""
        session = "" if session == "-" else session
        if not session:
            shown = "-"
        elif started and started != "-":
            # "2026-09-17 13:40:12" -> "09-17 13:40"
            shown = f"{session} · since {started[5:16]}"
        else:
            shown = session
        key = table.add_row(name, host, status, port, shown)
        self._row_sessions[key] = session

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#status-table", DataTable)
        row_key = event.row_key
        try:
            row_data = table.get_row(row_key)
            name = str(row_data[0])
            host = str(row_data[1])
            status = str(row_data[2])
            session = self._row_sessions.get(row_key, "")
            self._selected_row = (name, host, status, session)
        except Exception:
            self._selected_row = None

    def _view_selected_logs(self) -> None:
        log = self.query_one("#log-panel", RichLog)
        if not self._selected_row:
            log.write("[yellow]Select a service row first.[/yellow]")
            return

        name, host, status, session = self._selected_row

        if "Stopped" in status:
            log.write(f"[yellow]{name} ({host}) is not running. Start the service first.[/yellow]")
            return

        log.clear()
        svc_def = next((s for s in KNOWN_SERVICES if s["name"] == name), None)
        svc_type = svc_def["type"] if svc_def else ("tmux" if session else "unknown")
        svc_key = svc_def.get("target", "") if svc_def and svc_type == "system" else ""

        # from here on the name is only printed: "[tmux] x" would read as markup
        from rich.markup import escape
        name = escape(name)
        if host == "local" or (svc_type == "system" and is_this_machine(host)):
            self._view_logs_local(log, name, svc_type, svc_key, session)
        else:
            self._view_logs_remote(log, name, host, svc_type, svc_key, session)

    def _view_logs_local(self, log: RichLog, name: str, svc_type: str, svc_key: str, session: str) -> None:
        if svc_type == "system" and svc_key:
            from openmmla.tui.screens.launcher import _get_system_service_log
            log.write(f"[bold]Logs for {name} (local)[/bold]\n")
            output = _infra_container_logs_local(svc_key) or _get_system_service_log(svc_key)
            for line in output.splitlines():
                log.write(line)
        elif session:
            log.write(f"[bold]Logs for {name} (session: {session})[/bold]\n")
            output = _capture_tmux_pane(session, lines=40)
            for line in output.splitlines():
                log.write(line)
        else:
            log.write(f"[yellow]No logs here for {name}: open its card in the Launcher and press Logs.[/yellow]")

    def _view_logs_remote(self, log: RichLog, name: str, host: str, svc_type: str, svc_key: str, session: str) -> None:
        profile = get_profile_by_name(host) or _profile_for_host(host)
        if profile is None:
            log.write(
                f"[red]No SSH profile reaches '{host}'. Add one whose host is '{host}' "
                f"under Launcher → SSH Profiles to read its logs.[/red]"
            )
            return

        _REMOTE_LOG_CMDS: dict[str, str] = {
            "influxdb": "journalctl -u influxdb -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/influxdb/influxd.log 2>/dev/null || echo '(no influxdb logs found)'",
            "mongodb": "journalctl -u mongod -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/mongodb/mongod.log 2>/dev/null || echo '(no mongodb logs found)'",
            "redis": "journalctl -u redis-server -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/redis/redis-server.log 2>/dev/null || echo '(no redis logs found)'",
            "mosquitto": "journalctl -u mosquitto -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/mosquitto/mosquitto.log 2>/dev/null || echo '(no mosquitto logs found)'",
            "nginx": "journalctl -u nginx -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/nginx/error.log 2>/dev/null || echo '(no nginx logs found)'",
            "mediamtx": "if tmux has-session -t mediamtx 2>/dev/null; then tmux capture-pane -p -t mediamtx | tail -n 80; else echo '(no mediamtx tmux session found)'; fi",
        }

        if svc_type == "system" and svc_key in _REMOTE_LOG_CMDS:
            log.write(f"[bold]Logs for {name} ({host})[/bold]\n")
            cmd = _infra_container_logs_remote_cmd(
                svc_key, profile.remote_project_path, _REMOTE_LOG_CMDS[svc_key],
            )
            try:
                result = ssh_run_sync(profile, cmd, timeout=15.0)
                output = result.stdout if result.returncode == 0 else result.stderr
                for line in output.splitlines():
                    log.write(line)
            except Exception as e:
                log.write(f"[red]Failed to fetch remote logs: {e}[/red]")
        elif session:
            log.write(f"[bold]Logs for {name} ({host}, session: {session})[/bold]\n")
            try:
                result = ssh_run_sync(profile, f"tmux capture-pane -t {session} -p -S -80", timeout=10.0)
                output = result.stdout if result.returncode == 0 else f"(could not capture tmux session '{session}')"
                for line in output.splitlines():
                    log.write(line)
            except Exception as e:
                log.write(f"[red]Failed to fetch remote logs: {e}[/red]")
        else:
            log.write(f"[yellow]No logs available for {name} on {host}[/yellow]")
