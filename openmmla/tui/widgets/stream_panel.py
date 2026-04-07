from __future__ import annotations

import asyncio
import subprocess

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, DataTable

from openmmla.tui.schema.loader import StreamDef, load_streams
from openmmla.tui.ssh import get_profile_by_name, ssh_run_sync, ssh_check_tmux


def _check_local_tmux(session_name: str) -> bool:
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _tmux_session_name(stream_name: str) -> str:
    return f"mmla-stream-{stream_name}"


def _build_ffmpeg_cmd(stream: StreamDef) -> str:
    """build the ffmpeg command string from a stream definition."""
    target = stream.target
    is_audio = target.startswith("udp://") or target.startswith("tcp://")

    if is_audio:
        fmt = stream.format or "s16le"
        rate = stream.rate or 16000
        channels = stream.channels or 1
        device = stream.device or "hw:0,0"
        proto = "udp" if target.startswith("udp://") else "tcp"
        addr = target.split("://", 1)[1]
        return (
            f"ffmpeg -f alsa -ac {channels} -ar {rate} -i {device} "
            f"-c:a aac -b:a 128k "
            f"-f flv {proto}://{addr}"
        )

    device = stream.device or "/dev/video0"
    codec = stream.codec or "libx264"
    resolution = stream.resolution or "1920x1080"
    fps = stream.fps or 30
    return (
        f"ffmpeg -use_wallclock_as_timestamps 1 "
        f"-f v4l2 -input_format mjpeg -framerate {fps} -video_size {resolution} -i {device} "
        f"-c:v {codec} -preset ultrafast -tune zerolatency "
        f"-g {fps} -keyint_min {fps} -sc_threshold 0 "
        f'-x264-params "keyint={fps}:min-keyint={fps}:no-scenecut=1:repeat-headers=1" '
        f"-b:v 1M -maxrate 2M -bufsize 2M "
        f"-f flv {target}"
    )


class StreamPanel(Widget):
    """panel for managing remote streams via SSH (start/stop only)."""

    class StreamLog(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    DEFAULT_CSS = """
    StreamPanel {
        height: 1fr;
        width: 1fr;
    }
    #stream-table {
        height: 1fr;
    }
    #stream-actions {
        height: auto;
        padding: 1 0;
    }
    #stream-actions Button {
        margin: 0 1;
    }
    """

    def __init__(self, streams: list[StreamDef], config_path: str = "") -> None:
        super().__init__()
        self._streams = list(streams)
        self._config_path = config_path
        self._statuses: dict[str, bool] = {}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield DataTable(id="stream-table")
            with Horizontal(id="stream-actions"):
                yield Button("Start", variant="success", id="stream-btn-start")
                yield Button("Stop", variant="error", id="stream-btn-stop")
                yield Button("Start All", variant="success", id="stream-btn-start-all")
                yield Button("Stop All", variant="error", id="stream-btn-stop-all")
                yield Button("Refresh", variant="primary", id="stream-btn-refresh")

    def on_mount(self) -> None:
        table = self.query_one("#stream-table", DataTable)
        table.add_columns("Name", "SSH Profile", "Device", "Target", "Status")
        table.cursor_type = "row"
        if self._streams:
            self._refresh_all()
        else:
            self._rebuild_table()

    def _log(self, msg: str) -> None:
        self.post_message(self.StreamLog(msg))

    def _refresh_all(self) -> None:
        self.run_worker(self._async_refresh_all(), exclusive=True)

    async def _async_refresh_all(self) -> None:
        loop = asyncio.get_event_loop()
        for stream in self._streams:
            if not stream.ssh_profile:
                continue
            session = _tmux_session_name(stream.name)
            if stream.ssh_profile == "local":
                is_running = await loop.run_in_executor(
                    None, _check_local_tmux, session,
                )
            else:
                profile = get_profile_by_name(stream.ssh_profile)
                if profile is None:
                    self._statuses[stream.name] = False
                    continue
                is_running = await loop.run_in_executor(
                    None, ssh_check_tmux, profile, session,
                )
            self._statuses[stream.name] = is_running
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        try:
            table = self.query_one("#stream-table", DataTable)
        except Exception:
            return
        table.clear()
        if not self._streams:
            return
        for stream in self._streams:
            if not stream.ssh_profile:
                status = "External"
            elif self._statuses.get(stream.name, False):
                status = "Running"
            else:
                status = "Stopped"
            table.add_row(
                stream.name,
                stream.ssh_profile or "-",
                stream.device or "-",
                stream.target,
                status,
            )

    def _get_selected_stream(self) -> StreamDef | None:
        try:
            table = self.query_one("#stream-table", DataTable)
            row_key = table.cursor_row
            if row_key < 0 or row_key >= len(self._streams):
                return None
            return self._streams[row_key]
        except Exception:
            return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id or ""
        if btn == "stream-btn-start":
            stream = self._get_selected_stream()
            if stream and stream.ssh_profile:
                self._start_stream(stream)
            elif stream and not stream.ssh_profile:
                self._log(f"[yellow]{stream.name} is an external stream (no SSH profile).[/yellow]")
            else:
                self._log("[yellow]Select a stream row first.[/yellow]")
        elif btn == "stream-btn-stop":
            stream = self._get_selected_stream()
            if stream and stream.ssh_profile:
                self._stop_stream(stream)
            elif stream and not stream.ssh_profile:
                self._log(f"[yellow]{stream.name} is an external stream (no SSH profile).[/yellow]")
            else:
                self._log("[yellow]Select a stream row first.[/yellow]")
        elif btn == "stream-btn-start-all":
            for s in self._streams:
                if s.ssh_profile:
                    self._start_stream(s)
        elif btn == "stream-btn-stop-all":
            for s in self._streams:
                if s.ssh_profile:
                    self._stop_stream(s)
        elif btn == "stream-btn-refresh":
            self._reload_and_refresh()

    def _reload_and_refresh(self) -> None:
        """re-read streams from config.yml and refresh status."""
        if self._config_path:
            self._streams = list(load_streams(self._config_path))
        self._refresh_all()

    def _start_stream(self, stream: StreamDef) -> None:
        self.run_worker(self._async_start(stream))

    def _stop_stream(self, stream: StreamDef) -> None:
        self.run_worker(self._async_stop(stream))

    async def _async_start(self, stream: StreamDef) -> None:
        session = _tmux_session_name(stream.name)
        is_local = stream.ssh_profile == "local"
        loop = asyncio.get_event_loop()

        if is_local:
            already_running = await loop.run_in_executor(None, _check_local_tmux, session)
        else:
            profile = get_profile_by_name(stream.ssh_profile)
            if profile is None:
                self._log(f"[red]SSH profile '{stream.ssh_profile}' not found.[/red]")
                return
            already_running = await loop.run_in_executor(None, ssh_check_tmux, profile, session)

        if already_running:
            self._log(f"[yellow]{stream.name} is already running.[/yellow]")
            self._statuses[stream.name] = True
            self._rebuild_table()
            return

        ffmpeg_cmd = _build_ffmpeg_cmd(stream)
        tmux_cmd = f"tmux new-session -d -s {session} '{ffmpeg_cmd}; exec bash'"
        target_label = "locally" if is_local else f"on {stream.ssh_profile}"
        self._log(f"[green]Starting {stream.name} {target_label}...[/green]")
        self._log(f"  {ffmpeg_cmd}")

        try:
            if is_local:
                result = await loop.run_in_executor(
                    None, lambda: subprocess.run(
                        tmux_cmd, shell=True, capture_output=True, text=True, timeout=15,
                    ),
                )
            else:
                result = await loop.run_in_executor(
                    None, ssh_run_sync, profile, tmux_cmd, 15.0,
                )
            if result.returncode == 0:
                self._log(f"[green]{stream.name} started.[/green]")
                self._statuses[stream.name] = True
            else:
                output = result.stdout.strip() if result.stdout else result.stderr.strip()
                self._log(f"[red]Failed to start {stream.name}: {output}[/red]")
                self._statuses[stream.name] = False
        except Exception as e:
            self._log(f"[red]Error starting {stream.name}: {e}[/red]")
            self._statuses[stream.name] = False
        self._rebuild_table()

    async def _async_stop(self, stream: StreamDef) -> None:
        session = _tmux_session_name(stream.name)
        is_local = stream.ssh_profile == "local"

        if not is_local:
            profile = get_profile_by_name(stream.ssh_profile)
            if profile is None:
                self._log(f"[red]SSH profile '{stream.ssh_profile}' not found.[/red]")
                return

        stop_cmd = (
            f"tmux send-keys -t {session} C-c 2>/dev/null; "
            f"sleep 1; "
            f"tmux kill-session -t {session} 2>/dev/null; "
            f"echo DONE"
        )
        target_label = "locally" if is_local else f"on {stream.ssh_profile}"
        self._log(f"[red]Stopping {stream.name} {target_label}...[/red]")

        loop = asyncio.get_event_loop()
        try:
            if is_local:
                result = await loop.run_in_executor(
                    None, lambda: subprocess.run(
                        stop_cmd, shell=True, capture_output=True, text=True, timeout=15,
                    ),
                )
            else:
                result = await loop.run_in_executor(
                    None, ssh_run_sync, profile, stop_cmd, 15.0,
                )
            if "DONE" in (result.stdout or ""):
                self._log(f"[red]{stream.name} stopped.[/red]")
                self._statuses[stream.name] = False
            else:
                self._log(f"[yellow]{stream.name} may still be running.[/yellow]")
        except Exception as e:
            self._log(f"[red]Error stopping {stream.name}: {e}[/red]")
        self._rebuild_table()

    def update_streams(self, streams: list[StreamDef]) -> None:
        """replace the stream list and refresh."""
        self._streams = list(streams)
        self._statuses.clear()
        if streams:
            self._refresh_all()
        else:
            self._rebuild_table()
