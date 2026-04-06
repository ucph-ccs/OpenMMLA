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
from openmmla.tui.ssh import load_ssh_profiles, ssh_check_port, ssh_check_tmux, ssh_run_sync, get_profile_by_name, SSHProfile


KNOWN_SERVICES = [
    {"name": "InfluxDB", "port": 8086, "type": "system"},
    {"name": "MongoDB", "port": 27017, "type": "system"},
    {"name": "Redis", "port": 6379, "type": "system"},
    {"name": "Mosquitto", "port": 1883, "type": "system"},
    {"name": "Nginx", "port": 8080, "type": "system"},
    {"name": "Flask Dashboard", "port": 5050, "type": "tmux", "session": "flask"},
    {"name": "Next.js Frontend", "port": 3000, "type": "tmux", "session": "next"},
    {"name": "Celery Worker", "port": None, "type": "tmux", "session": "celery"},
    {"name": "AudioInferer", "port": 5001, "type": "tmux", "session": "audioinferer"},
    {"name": "AudioResampler", "port": 5002, "type": "tmux", "session": "audioresampler"},
    {"name": "SpeechEnhancer", "port": 5003, "type": "tmux", "session": "speechenhancer"},
    {"name": "SpeechSeparator", "port": 5004, "type": "tmux", "session": "speechseparator"},
    {"name": "SpeechTranscriber", "port": 5005, "type": "tmux", "session": "speechtranscriber"},
    {"name": "VoiceActivityDetector", "port": 5006, "type": "tmux", "session": "voiceactivitydetector"},
]


def _check_port(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """check if a port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


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
        self._selected_row: tuple[str, str, str, str] | None = None  # (name, host, status, session)
        self._ssh_profiles: list[SSHProfile] = []
        self._summary_text: str = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Loading status...", id="status-summary")
            yield DataTable(id="status-table")
            from textual.containers import Horizontal
            with Horizontal(id="status-actions"):
                yield Button("Refresh", variant="primary", id="btn-refresh")
                yield Button("View Logs", variant="default", id="btn-view-logs")
            yield RichLog(id="log-panel", highlight=True, markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#status-table", DataTable)
        table.add_columns("Service", "Host", "Status", "Port", "Session", "Started")
        table.cursor_type = "row"
        self._refresh_status()
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        self.set_timer(5.0, self._auto_refresh)

    def _auto_refresh(self) -> None:
        self._refresh_status(include_remote=False)
        self._schedule_refresh()

    def _refresh_status(self, include_remote: bool = False) -> None:
        """refresh service status. remote checks only when explicitly requested."""
        self._tmux_sessions = _list_tmux_sessions()
        self._ssh_profiles = load_ssh_profiles()
        table = self.query_one("#status-table", DataTable)
        table.clear()

        running_count = 0
        port_count = 0

        for svc in KNOWN_SERVICES:
            name = svc["name"]
            port = svc.get("port")
            session = svc.get("session", "")

            port_ok = _check_port(port) if port else False
            session_ok = session in self._tmux_sessions if session else False

            if svc["type"] == "system":
                is_up = port_ok
            else:
                is_up = session_ok

            status_str = "Running" if is_up else "Stopped"
            port_str = str(port) if port else "-"
            session_str = session if session else "-"
            started_str = self._tmux_sessions.get(session, "-") if session else "-"

            if is_up:
                running_count += 1
            if port_ok:
                port_count += 1

            table.add_row(name, "local", status_str, port_str, session_str, started_str)

        extra_sessions = set(self._tmux_sessions.keys()) - {
            s.get("session", "") for s in KNOWN_SERVICES if s.get("session")
        }
        for session_name in sorted(extra_sessions):
            if session_name in ("asr-services", "vfa-services"):
                continue
            table.add_row(
                f"[tmux] {session_name}",
                "local",
                "Running",
                "-",
                session_name,
                self._tmux_sessions[session_name],
            )
            running_count += 1

        remote_count = len(self._ssh_profiles)
        self._summary_text = (
            f"Services: {running_count} running | Ports: {port_count} active | "
            f"tmux: {len(self._tmux_sessions)} | SSH profiles: {remote_count}"
        )
        summary = self.query_one("#status-summary", Static)
        summary.update(self._summary_text)

        if include_remote and self._ssh_profiles:
            self.run_worker(self._refresh_remote_status(), exclusive=True)

    async def _refresh_remote_status(self) -> None:
        """check remote services in background (slow SSH calls)."""
        table = self.query_one("#status-table", DataTable)
        added = 0

        for profile in self._ssh_profiles:
            for svc in KNOWN_SERVICES:
                port = svc.get("port")
                session = svc.get("session", "")

                r_port_ok = False
                r_session_ok = False

                loop = asyncio.get_event_loop()
                if port:
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

                if not r_up:
                    continue

                port_str = str(port) if port else "-"
                session_str = session if session else "-"
                table.add_row(
                    svc["name"], profile.name, "Running", port_str, session_str, "-",
                )
                added += 1

        if added > 0:
            summary = self.query_one("#status-summary", Static)
            summary.update(f"{self._summary_text} (+{added} remote)")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh":
            self._refresh_status(include_remote=True)
        elif event.button.id == "btn-view-logs":
            self._view_selected_logs()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#status-table", DataTable)
        row_key = event.row_key
        try:
            row_data = table.get_row(row_key)
            name = str(row_data[0])
            host = str(row_data[1])
            status = str(row_data[2])
            session = str(row_data[4]) if row_data[4] and row_data[4] != "-" else ""
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
        svc_key = name.lower().split()[0] if svc_def and svc_type == "system" else ""

        if host == "local":
            self._view_logs_local(log, name, svc_type, svc_key, session)
        else:
            self._view_logs_remote(log, name, host, svc_type, svc_key, session)

    def _view_logs_local(self, log: RichLog, name: str, svc_type: str, svc_key: str, session: str) -> None:
        if svc_type == "system" and svc_key:
            from openmmla.tui.screens.launcher import _get_system_service_log
            log.write(f"[bold]Logs for {name} (local)[/bold]\n")
            output = _get_system_service_log(svc_key)
            for line in output.splitlines():
                log.write(line)
        elif session:
            log.write(f"[bold]Logs for {name} (session: {session})[/bold]\n")
            output = _capture_tmux_pane(session, lines=40)
            for line in output.splitlines():
                log.write(line)
        else:
            log.write(f"[yellow]No logs available for {name}[/yellow]")

    def _view_logs_remote(self, log: RichLog, name: str, host: str, svc_type: str, svc_key: str, session: str) -> None:
        profile = get_profile_by_name(host)
        if profile is None:
            log.write(f"[red]SSH profile '{host}' not found.[/red]")
            return

        _REMOTE_LOG_CMDS: dict[str, str] = {
            "influxdb": "journalctl -u influxdb -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/influxdb/influxd.log 2>/dev/null || echo '(no influxdb logs found)'",
            "mongodb": "journalctl -u mongod -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/mongodb/mongod.log 2>/dev/null || echo '(no mongodb logs found)'",
            "redis": "journalctl -u redis-server -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/redis/redis-server.log 2>/dev/null || echo '(no redis logs found)'",
            "mosquitto": "journalctl -u mosquitto -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/mosquitto/mosquitto.log 2>/dev/null || echo '(no mosquitto logs found)'",
            "nginx": "journalctl -u nginx -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/nginx/error.log 2>/dev/null || echo '(no nginx logs found)'",
        }

        if svc_type == "system" and svc_key in _REMOTE_LOG_CMDS:
            log.write(f"[bold]Logs for {name} ({host})[/bold]\n")
            try:
                result = ssh_run_sync(profile, _REMOTE_LOG_CMDS[svc_key], timeout=15.0)
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
