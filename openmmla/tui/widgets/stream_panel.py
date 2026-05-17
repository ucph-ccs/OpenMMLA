from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import time

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, DataTable

from openmmla.tui.schema.loader import StreamDef, load_streams
from openmmla.tui.ssh import get_profile_by_name, ssh_run_sync
from openmmla.utils.stream_registry import register_stream_start, mark_stream_stopped


STREAM_REMOTE_PATH = "/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _with_stream_path(command: str) -> str:
    """run stream commands with a predictable PATH for non-interactive SSH shells."""
    return f"export PATH={STREAM_REMOTE_PATH}:$PATH; {command}"


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


def _check_remote_tmux(profile, session_name: str) -> bool:
    try:
        cmd = _with_stream_path(
            f"tmux has-session -t {shlex.quote(session_name)} 2>/dev/null && echo OK || echo FAIL"
        )
        result = ssh_run_sync(profile, cmd, timeout=8.0)
        return "OK" in result.stdout
    except Exception:
        return False


def _stream_start_file(session_name: str) -> str:
    return f"$HOME/.openmmla/streams/{session_name}.start"


def _project_root_from_config(config_path: str) -> str:
    if not config_path:
        return os.getcwd()
    path = os.path.abspath(config_path)
    marker = f"{os.sep}pipelines{os.sep}"
    if marker in path:
        return path.split(marker, 1)[0]
    return os.path.dirname(path)


def _build_tmux_stream_cmd(session: str, ffmpeg_cmd: str) -> str:
    start_file = _stream_start_file(session)
    inner_cmd = (
        f"mkdir -p $HOME/.openmmla/streams; "
        f"START_TIME=$(python3 -c \"import time; print('%.6f' % time.time())\" 2>/dev/null || date +%s); "
        f"printf '%s\\n' \"$START_TIME\" > {start_file}; "
        f"{ffmpeg_cmd}; exec bash"
    )
    return _with_stream_path(f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(inner_cmd)}")


def _read_local_stream_start_time(session: str) -> float | None:
    path = os.path.expanduser(f"~/.openmmla/streams/{session}.start")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_remote_stream_start_time(profile, session: str) -> float | None:
    cmd = _with_stream_path(f"cat {_stream_start_file(session)} 2>/dev/null")
    try:
        result = ssh_run_sync(profile, cmd, timeout=5.0)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (ValueError, Exception):
        return None
    return None


def _read_stream_start_time_with_retry(
    profile,
    session: str,
    is_local: bool,
    attempts: int = 20,
    delay: float = 0.1,
) -> float | None:
    """wait briefly for the tmux child shell to persist its start timestamp."""
    for attempt in range(max(1, attempts)):
        start_time = (
            _read_local_stream_start_time(session)
            if is_local
            else _read_remote_stream_start_time(profile, session)
        )
        if start_time is not None:
            return start_time
        if attempt < attempts - 1:
            time.sleep(delay)
    return None


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
        f"ffmpeg -fflags +genpts -use_wallclock_as_timestamps 1 "
        f"-f v4l2 -input_format mjpeg -framerate {fps} -video_size {resolution} -i {device} "
        f"-c:v {codec} -preset ultrafast -tune zerolatency "
        f"-g {fps} -keyint_min {fps} -sc_threshold 0 "
        f'-x264-params "keyint={fps}:min-keyint={fps}:no-scenecut=1:repeat-headers=1" '
        f"-b:v 1M -maxrate 2M -bufsize 2M "
        f"-f flv {target}"
    )


def _probe_rtmp_target(target: str, timeout: float = 10.0) -> tuple[bool, str]:
    """probe an RTMP target by asking ffmpeg to decode a short sample."""
    if not target.startswith("rtmp://"):
        return False, "Probe currently supports RTMP targets only."
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", target, "-t", "2", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return False, "ffmpeg command not found on this machine."
    except subprocess.TimeoutExpired:
        return False, "Probe timed out while reading the RTMP target."

    if result.returncode == 0:
        return True, "RTMP target is readable."
    output = (result.stderr or result.stdout or "").strip()
    if not output:
        output = f"ffmpeg exited with code {result.returncode}"
    return False, output


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
        self._project_dir = _project_root_from_config(config_path)
        self._statuses: dict[str, bool] = {}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield DataTable(id="stream-table")
            with Horizontal(id="stream-actions"):
                yield Button("Start", variant="success", id="stream-btn-start")
                yield Button("Stop", variant="error", id="stream-btn-stop")
                yield Button("Logs", variant="primary", id="stream-btn-logs")
                yield Button("Probe", variant="warning", id="stream-btn-probe")
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
                    None, _check_remote_tmux, profile, session,
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
        elif btn == "stream-btn-logs":
            stream = self._get_selected_stream()
            if stream and stream.ssh_profile:
                self._view_stream_logs(stream)
            elif stream and not stream.ssh_profile:
                self._log(f"[yellow]{stream.name} is external; no managed tmux logs.[/yellow]")
            else:
                self._log("[yellow]Select a stream row first.[/yellow]")
        elif btn == "stream-btn-probe":
            stream = self._get_selected_stream()
            if stream:
                self._probe_stream(stream)
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
            self._log(f"[cyan]Reloaded {len(self._streams)} stream(s) from {self._config_path}.[/cyan]")
        self._refresh_all()

    def _start_stream(self, stream: StreamDef) -> None:
        self.run_worker(self._async_start(stream))

    def _stop_stream(self, stream: StreamDef) -> None:
        self.run_worker(self._async_stop(stream))

    def _view_stream_logs(self, stream: StreamDef) -> None:
        self.run_worker(self._async_view_logs(stream))

    def _probe_stream(self, stream: StreamDef) -> None:
        self.run_worker(self._async_probe_stream(stream))

    async def _async_start(self, stream: StreamDef) -> None:
        session = _tmux_session_name(stream.name)
        is_local = stream.ssh_profile == "local"
        profile = None
        loop = asyncio.get_event_loop()

        if is_local:
            already_running = await loop.run_in_executor(None, _check_local_tmux, session)
        else:
            profile = get_profile_by_name(stream.ssh_profile)
            if profile is None:
                self._log(f"[red]SSH profile '{stream.ssh_profile}' not found.[/red]")
                return
            already_running = await loop.run_in_executor(None, _check_remote_tmux, profile, session)

        if already_running:
            self._log(f"[yellow]{stream.name} is already running.[/yellow]")
            self._statuses[stream.name] = True
            self._rebuild_table()
            return

        ffmpeg_cmd = _build_ffmpeg_cmd(stream)
        tmux_cmd = _build_tmux_stream_cmd(session, ffmpeg_cmd)
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
                start_time = await loop.run_in_executor(
                    None,
                    lambda: _read_stream_start_time_with_retry(profile, session, is_local),
                )
                if start_time is None:
                    start_time = time.time()
                    self._log(
                        "[yellow]Could not read capture-side stream_start_time; "
                        "using local registration time.[/yellow]"
                    )
                register_stream_start(
                    stream.name,
                    stream.target,
                    start_time,
                    project_dir=self._project_dir,
                    ssh_profile=stream.ssh_profile,
                    device=stream.device,
                )
                self._log(f"[green]{stream.name} started.[/green]")
                self._log(f"  stream_start_time={start_time:.6f}")
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

        quoted_session = shlex.quote(session)
        stop_cmd = (
            f"tmux send-keys -t {quoted_session} C-c 2>/dev/null; "
            f"sleep 1; "
            f"tmux kill-session -t {quoted_session} 2>/dev/null; "
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
                    None, ssh_run_sync, profile, _with_stream_path(stop_cmd), 15.0,
                )
            if "DONE" in (result.stdout or ""):
                self._log(f"[red]{stream.name} stopped.[/red]")
                mark_stream_stopped(stream.name, project_dir=self._project_dir)
                self._statuses[stream.name] = False
            else:
                self._log(f"[yellow]{stream.name} may still be running.[/yellow]")
        except Exception as e:
            self._log(f"[red]Error stopping {stream.name}: {e}[/red]")
        self._rebuild_table()

    async def _async_view_logs(self, stream: StreamDef) -> None:
        session = _tmux_session_name(stream.name)
        is_local = stream.ssh_profile == "local"
        loop = asyncio.get_event_loop()
        self._log(f"[cyan]── Logs for {stream.name} ({session}) ──[/cyan]")
        try:
            if is_local:
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["tmux", "capture-pane", "-t", session, "-p", "-S", "-120"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    ),
                )
            else:
                profile = get_profile_by_name(stream.ssh_profile)
                if profile is None:
                    self._log(f"[red]SSH profile '{stream.ssh_profile}' not found.[/red]")
                    return
                cmd = _with_stream_path(f"tmux capture-pane -t {shlex.quote(session)} -p -S -120")
                result = await loop.run_in_executor(None, ssh_run_sync, profile, cmd, 10.0)
            output = (result.stdout or result.stderr or "").strip()
            if result.returncode != 0:
                self._log(f"[red]Could not read tmux logs: {output or result.returncode}[/red]")
                return
            if not output:
                self._log("[yellow]No tmux output captured yet.[/yellow]")
                return
            for line in output.splitlines()[-80:]:
                self._log(line)
        except Exception as e:
            self._log(f"[red]Error reading logs for {stream.name}: {e}[/red]")

    async def _async_probe_stream(self, stream: StreamDef) -> None:
        self._log(f"[cyan]Probing {stream.target}...[/cyan]")
        loop = asyncio.get_event_loop()
        success, message = await loop.run_in_executor(None, _probe_rtmp_target, stream.target)
        if success:
            self._log(f"[green]{stream.name}: {message}[/green]")
        else:
            self._log(f"[red]{stream.name}: {message}[/red]")

    def update_streams(self, streams: list[StreamDef]) -> None:
        """replace the stream list and refresh."""
        self._streams = list(streams)
        self._statuses.clear()
        if streams:
            self._refresh_all()
        else:
            self._rebuild_table()
