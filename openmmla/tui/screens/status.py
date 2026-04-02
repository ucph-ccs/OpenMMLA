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
from openmmla.tui.ssh import load_ssh_profiles, ssh_check_port, ssh_check_tmux, SSHProfile


KNOWN_SERVICES = [
    {"name": "InfluxDB", "port": 8086, "type": "system"},
    {"name": "Redis", "port": 6379, "type": "system"},
    {"name": "Mosquitto", "port": 1883, "type": "system"},
    {"name": "Nginx", "port": 8080, "type": "system"},
    {"name": "Flask Dashboard", "port": 5000, "type": "tmux", "session": "flask"},
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
        self._selected_session: str | None = None
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
        self._refresh_status()
        self._schedule_refresh()

    def _refresh_status(self) -> None:
        """refresh local services only (fast, non-blocking)."""
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
                is_up = session_ok or port_ok

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

        if self._ssh_profiles:
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
            self._refresh_status()
        elif event.button.id == "btn-view-logs":
            self._view_selected_logs()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one("#status-table", DataTable)
        row_key = event.row_key
        try:
            row_data = table.get_row(row_key)
            session = row_data[3]
            if session and session != "-":
                self._selected_session = str(session)
        except Exception:
            pass

    def _view_selected_logs(self) -> None:
        log = self.query_one("#log-panel", RichLog)
        if not self._selected_session:
            log.write("[yellow]Select a row with a tmux session first.[/yellow]")
            return

        log.clear()
        log.write(f"[bold]Logs for session: {self._selected_session}[/bold]\n")
        output = _capture_tmux_pane(self._selected_session, lines=40)
        for line in output.splitlines():
            log.write(line)
