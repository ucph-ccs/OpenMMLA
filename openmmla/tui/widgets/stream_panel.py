from __future__ import annotations

import asyncio
import datetime
from dataclasses import dataclass
import os
import re
import shlex
import socket
import subprocess
import time

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, DataTable, Label, Select

from openmmla.tui.schema.loader import StreamDef, load_streams
from openmmla.tui.ssh import get_profile_by_name, ssh_run_sync
from openmmla.utils.artifact_paths import safe_segment
from openmmla.utils.constants import STREAM_URL_SCHEMES
from openmmla.utils.stream_registry import load_stream_registry, register_stream_start, mark_stream_stopped


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


# ALSA-style device names mark an audio stream when no kind is given
_ALSA_DEVICE_PREFIXES = ("hw:", "plughw:", "default", "sysdefault", "dsnoop", "plug:", "pulse")

# recording root on a remote streaming host when record_root is unset; the same
# root the Collection card records under, so Collection -> Download fetches both
STREAM_RECORD_ROOT = "$HOME/artifacts"

# seconds Stop waits for ffmpeg to finalize its files after Ctrl-C before the
# tmux session is killed
STREAM_STOP_GRACE_SECONDS = 8


# the Recordings select: whole files of one capture host, or a session's part of every stream
ALL_RECORDINGS = "__everything__"


@dataclass(frozen=True)
class RecordedStream:
    """where a managed stream keeps its recordings: enough to find them again."""
    name: str
    ssh_profile: str   # "local" or an SSH profile
    record_root: str   # on the capture host; $HOME/... for a remote one
    host_label: str    # the folder under collection/
    kind: str          # video | audio


def _stream_start_file(session_name: str) -> str:
    return f"$HOME/.openmmla/streams/{session_name}.start"


def _stream_record_file(session_name: str) -> str:
    """holds the expanded recording path of a managed stream on its host."""
    return f"$HOME/.openmmla/streams/{session_name}.record"


def _project_root_from_config(config_path: str) -> str:
    if not config_path:
        return os.getcwd()
    path = os.path.abspath(config_path)
    marker = f"{os.sep}pipelines{os.sep}"
    if marker in path:
        return path.split(marker, 1)[0]
    return os.path.dirname(path)


def _build_tmux_stream_cmd(
    session: str,
    ffmpeg_cmd: str,
    record_dir: str | None = None,
    record_path: str | None = None,
) -> str:
    """wrap the ffmpeg command in a detached tmux session that first notes the
    capture-side start time and, when recording, creates the recording folder and
    notes the file path (the ${START_TIME} in it expands on the host)."""
    start_file = _stream_start_file(session)
    record_file = _stream_record_file(session)
    if record_dir and record_path:
        record_part = (
            f'mkdir -p "{record_dir}"; '
            f"printf '%s\\n' \"{record_path}\" > {record_file}; "
        )
    else:
        record_part = f"rm -f {record_file}; "
    inner_cmd = (
        f"mkdir -p $HOME/.openmmla/streams; "
        f"START_TIME=$(python3 -c \"import time; print('%.6f' % time.time())\" 2>/dev/null || date +%s); "
        f"printf '%s\\n' \"$START_TIME\" > {start_file}; "
        f"{record_part}"
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


def _build_stop_stream_cmd(session: str) -> str:
    """Ctrl-C the ffmpeg in the stream's tmux session, wait until it has exited
    (it finalizes the recording's index and duration on the way out), then close
    the session. ffmpeg is a child of the pane's shell, which is what the wait
    polls; the shell itself is what tmux reports as the pane's command."""
    quoted_session = shlex.quote(session)
    return (
        f"tmux send-keys -t {quoted_session} C-c 2>/dev/null; "
        f"p=$(tmux list-panes -t {quoted_session} -F '#{{pane_pid}}' 2>/dev/null | head -n1); i=0; "
        f'while [ -n "$p" ] && pgrep -P "$p" -x ffmpeg >/dev/null 2>&1 '
        f"&& [ $i -lt {STREAM_STOP_GRACE_SECONDS * 2} ]; do sleep 0.5; i=$((i+1)); done; "
        f"tmux kill-session -t {quoted_session} 2>/dev/null; "
        f"echo DONE"
    )


def _read_stream_record_path(profile, session: str, is_local: bool, attempts: int = 10, delay: float = 0.1) -> str | None:
    """the recording path the tmux wrapper noted, once it exists."""
    path = f"~/.openmmla/streams/{session}.record"
    for attempt in range(max(1, attempts)):
        value = None
        if is_local:
            try:
                with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
                    value = f.read().strip()
            except OSError:
                value = None
        else:
            try:
                result = ssh_run_sync(profile, _with_stream_path(f"cat {_stream_record_file(session)} 2>/dev/null"), timeout=5.0)
                value = result.stdout.strip() if result.returncode == 0 else None
            except Exception:
                value = None
        if value:
            return value
        if attempt < attempts - 1:
            time.sleep(delay)
    return None


# raw PCM sample formats the ASR base can receive (AudioStream SUPPORTED_FORMATS),
# keyed by both the ffmpeg spelling and the base config spelling
_PCM_SAMPLE_FORMATS = {
    "s16le": "s16le", "int16": "s16le",
    "s32le": "s32le", "int32": "s32le",
    "f32le": "f32le", "float32": "f32le",
}


def _pcm_sample_format(value: str) -> str:
    """map a Streams 'format' value to the ffmpeg raw PCM sample format."""
    key = (value or "s16le").strip().lower()
    if key not in _PCM_SAMPLE_FORMATS:
        supported = ", ".join(sorted(set(_PCM_SAMPLE_FORMATS.values())))
        raise ValueError(f"unsupported audio stream format '{value}', use one of: {supported}")
    return _PCM_SAMPLE_FORMATS[key]


def _stream_kind(stream: StreamDef) -> str:
    """'audio' or 'video': the explicit kind, else inferred from target and device."""
    kind = (stream.kind or "").strip().lower()
    if kind in ("audio", "video"):
        return kind
    if stream.target.startswith(("udp://", "tcp://")):
        return "audio"
    if (stream.device or "").strip().lower().startswith(_ALSA_DEVICE_PREFIXES):
        return "audio"
    return "video"


def _publish_muxer(target: str) -> tuple[str, dict[str, str]]:
    """(ffmpeg output format, its options) for a publish target URL."""
    if target.startswith("rtmp://"):
        return "flv", {}
    if target.startswith("rtsp://"):
        return "rtsp", {"rtsp_transport": "tcp"}
    if target.startswith("srt://"):
        return "mpegts", {}
    raise ValueError(
        f"unsupported stream target '{target}': use rtmp://, rtsp:// or srt://, "
        "or udp:// / tcp:// for raw audio to an ASR base")


def _cli_options(options: dict[str, str]) -> str:
    return "".join(f"-{key} {value} " for key, value in options.items())


def _tee_slave(muxer: str, options: dict[str, str], url: str, onfail_ignore: bool = False) -> str:
    """one output of the tee muxer: [f=<muxer>:<opt>=<val>:...]<url>."""
    parts = [f"f={muxer}", *(f"{key}={value}" for key, value in options.items())]
    if onfail_ignore:
        parts.append("onfail=ignore")
    return f"[{':'.join(parts)}]{url}"


def _double_rate(rate: str) -> str:
    """'1M' -> '2M', '800k' -> '1600k': maxrate and bufsize follow the bitrate."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", rate or "")
    if not match:
        raise ValueError(f"invalid bitrate '{rate}', use e.g. 1M or 800k")
    return f"{float(match.group(1)) * 2:g}{match.group(2)}"


def _record_path(stream: StreamDef, record_dir: str) -> str:
    """the recording file on the streaming host; ${START_TIME} expands in the tmux
    wrapper so the name carries the capture-side start time the file source expects."""
    extension = "wav" if _stream_kind(stream) == "audio" else "mkv"
    return f"{record_dir.rstrip('/')}/{stream.name}_${{START_TIME}}.{extension}"


def _build_ffmpeg_cmd(stream: StreamDef, record_dir: str | None = None) -> str:
    """build the ffmpeg command string from a stream definition.

    With record_dir the same capture is also written to a file there, so a
    session keeps its raw recording next to the live stream: one encode with two
    outputs, via the tee muxer for video and a second PCM output for audio.
    """
    target = stream.target
    record_to = _record_path(stream, record_dir) if record_dir else None

    if _stream_kind(stream) == "audio":
        rate = stream.rate or 16000
        channels = stream.channels or 1
        device = stream.device or "hw:0,0"
        capture = f"ffmpeg -f alsa -ac {channels} -ar {rate} -i {device} "
        if target.startswith(("udp://", "tcp://")):
            # the ASR base reads a header-less PCM byte stream on udp/tcp (see
            # AudioStream._read_socket_chunk), so send raw samples rather than AAC
            # in FLV. ffmpeg flushes once per ALSA period and keeps udp datagrams
            # below the MTU; the base re-frames whatever arrives to its own chunk_size.
            fmt = _pcm_sample_format(stream.format)
            proto = "udp" if target.startswith("udp://") else "tcp"
            addr = target.split("://", 1)[1]
            live = f"-c:a pcm_{fmt} -f {fmt} {proto}://{addr}"
            file_codec = f"pcm_{fmt}"
        else:
            muxer, options = _publish_muxer(target)
            live = f"-c:a aac -b:a 128k -f {muxer} {_cli_options(options)}{target}"
            file_codec = "pcm_s16le"
        if not record_to:
            return capture + live
        return f"{capture}-map 0:a {live} -map 0:a -c:a {file_codec} -f wav {record_to}"

    device = stream.device or "/dev/video0"
    codec = stream.codec or "libx264"
    resolution = stream.resolution or "1920x1080"
    fps = stream.fps or 30
    bitrate = stream.bitrate or "1M"
    peak = _double_rate(bitrate)
    muxer, options = _publish_muxer(target)
    encode = (
        f"ffmpeg -fflags +genpts -use_wallclock_as_timestamps 1 "
        f"-f v4l2 -input_format mjpeg -framerate {fps} -video_size {resolution} -i {device} "
        f"-c:v {codec} -preset ultrafast -tune zerolatency "
        f"-g {fps} -keyint_min {fps} -sc_threshold 0 "
        f'-x264-params "keyint={fps}:min-keyint={fps}:no-scenecut=1:repeat-headers=1" '
        f"-b:v {bitrate} -maxrate {peak} -bufsize {peak} "
    )
    if not record_to:
        return f"{encode}-f {muxer} {_cli_options(options)}{target}"
    # one encode, two outputs: the tee muxer needs the global-header flag spelled
    # out, and onfail=ignore keeps the recording going when the server is down
    return (
        f"{encode}-flags +global_header -map 0:v -f tee "
        f'"{_tee_slave(muxer, options, target, onfail_ignore=True)}|[f=matroska]{record_to}"'
    )


def _probe_stream_target(target: str, timeout: float = 10.0) -> tuple[bool, str]:
    """probe a pullable stream URL by asking ffmpeg to decode a short sample."""
    if not target.startswith(tuple(f"{scheme}://" for scheme in STREAM_URL_SCHEMES)):
        return False, "Probe supports rtmp://, rtsp:// and srt:// URLs only."
    command = ["ffmpeg", "-v", "error"]
    if target.startswith("rtsp://"):
        command += ["-rtsp_transport", "tcp"]
    command += ["-i", target, "-t", "2", "-f", "null", "-"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return False, "ffmpeg command not found on this machine."
    except subprocess.TimeoutExpired:
        return False, "Probe timed out while reading the stream."

    if result.returncode == 0:
        return True, "Stream is readable."
    output = (result.stderr or result.stdout or "").strip()
    if not output:
        output = f"ffmpeg exited with code {result.returncode}"
    return False, output


# name from before MediaMTX, when every pullable stream was RTMP
_probe_rtmp_target = _probe_stream_target


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
    #stream-recordings {
        height: 3;
        padding: 0 1;
    }
    #stream-recordings Label {
        width: 13;
        padding-top: 1;
    }
    #stream-session-select {
        width: 1fr;
    }
    #stream-recordings Button {
        margin: 0 1;
    }
    #stream-actions {
        height: auto;
        padding: 1 0;
    }
    #stream-actions Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        streams: list[StreamDef],
        config_path: str = "",
        project_dir: str | None = None,
        session_choices: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._streams = list(streams)
        self._config_path = config_path
        self._project_dir = project_dir or _project_root_from_config(config_path)
        self._statuses: dict[str, bool] = {}
        # sessions whose part of the recordings Download can cut out
        self._session_choices = [s for s in (session_choices or []) if s]

    def _session_options(self) -> list[tuple[str, str]]:
        return [
            ("Everything the selected stream's host has recorded (whole files, every day)", ALL_RECORDINGS),
            *((f"Session {session}: its part of every stream here, cut by its start and end", session)
              for session in self._session_choices),
        ]

    def set_session_choices(self, sessions: list[str]) -> None:
        self._session_choices = [s for s in sessions if s]
        try:
            select = self.query_one("#stream-session-select", Select)
        except Exception:
            return
        current = select.value
        select.set_options(self._session_options())
        select.value = current if current in self._session_choices else ALL_RECORDINGS

    def recorded_streams(self) -> list[RecordedStream]:
        """every stream of this card the console runs, and so may hold a recording of."""
        return [
            RecordedStream(stream.name, stream.ssh_profile, self._record_root(stream),
                           self._record_host_label(stream), _stream_kind(stream))
            for stream in self._streams if stream.ssh_profile
        ]

    @staticmethod
    def _record_session() -> str:
        """the folder a stream's recording is filed under: the day, never a
        session. A stream outlives sessions and is shared by them (several groups
        in one room pull the same camera), so its recording belongs to none of
        them; the start time in the file name is what ties it to a session."""
        return f"streams-{datetime.date.today():%Y%m%d}"

    @staticmethod
    def _record_host_label(stream: StreamDef) -> str:
        """host folder under <session>/collection/, the same the Collection card uses."""
        if stream.ssh_profile == "local":
            return safe_segment(socket.gethostname().split(".", 1)[0], "host")
        return safe_segment(stream.ssh_profile, "host")

    def _record_root(self, stream: StreamDef) -> str:
        """the folder on the capture host that holds the streams-<date> folders."""
        if stream.record_root:
            root = stream.record_root.rstrip("/")
            if root == "~" or root.startswith("~/"):
                root = "$HOME" + root[1:]
            return root
        if stream.ssh_profile == "local":
            return os.path.join(self._project_dir, "artifacts")
        return STREAM_RECORD_ROOT

    def _record_dir(self, stream: StreamDef) -> str | None:
        """recording folder on the streaming host, or None when the stream does not record."""
        if not stream.record:
            return None
        kind = _stream_kind(stream)
        return (
            f"{self._record_root(stream)}/{self._record_session()}/collection/"
            f"{self._record_host_label(stream)}/{kind}"
        )

    @staticmethod
    def _fetch_hint(stream: StreamDef) -> str:
        return "." if stream.ssh_profile == "local" else f" on {stream.ssh_profile}; Download on this tab copies it here."

    def _registered_record_path(self, stream_name: str) -> str:
        """the recording file noted when the stream was started, if it records."""
        try:
            entry = load_stream_registry(self._project_dir).get("streams", {}).get(stream_name)
        except Exception:
            return ""
        if not isinstance(entry, dict) or entry.get("status") != "running":
            return ""
        return str(entry.get("record_path") or "").strip()

    class DownloadRequested(Message):
        """fetch the recordings a capture host holds; the launcher owns the
        transfer (progress row, staging, resume), as it does for Collection."""

        def __init__(self, ssh_profile: str, record_root: str, host_label: str) -> None:
            super().__init__()
            self.ssh_profile = ssh_profile
            self.record_root = record_root
            self.host_label = host_label

    class SessionChoicesRequested(Message):
        """Refresh: a session created since the tab was opened belongs in the
        Recordings select; the launcher knows the sessions."""

    class SessionDownloadRequested(Message):
        """fetch one session's part of the recordings of every stream here: the
        launcher knows the session's start and end, and owns the transfer."""

        def __init__(self, session_id: str, streams: list[RecordedStream]) -> None:
            super().__init__()
            self.session_id = session_id
            self.streams = streams

    class RecordToggleRequested(Message):
        """flip a stream's capture-side recording; the launcher owns the config
        of the host the card is on and writes it there."""

        def __init__(self, stream_name: str, record: bool) -> None:
            super().__init__()
            self.stream_name = stream_name
            self.record = record

    HELP = (
        "A stream is one entry under Streams in this card's config: a camera or microphone that ffmpeg "
        "publishes to the Stream Server (target), and that the bases pull from it (read_target). Add one "
        "with + Add Stream on the Config tab; its URLs come from the Stream Server of System Settings.\n"
        "Recording has two independent switches. On the capture device: the Record column, which "
        "Record on/off flips for the selected stream (written next to the stream while it is pushed, so "
        "it survives a network drop). On the server: MediaMTX records every stream that reaches it, set "
        "on the Stream Server (MediaMTX) card, Config tab.\n"
        "A stream is shared: start it once and any number of sessions can pull it, one after another or "
        "at the same time. Its recording is therefore filed by day (streams-<date>), not under a session; "
        "the start time in the file name and the session's start and end tell which part belongs to which. "
        "Download, with a session chosen under Recordings, cuts that part out of every stream here on its "
        "capture host and copies it into artifacts/<session>/; without one it copies the whole files."
    )

    # how a stream stops being external
    _EXTERNAL_HINT = (
        "To run it from here, set ssh_profile of {name} on the Config tab (Streams) to the machine its "
        "device is attached to, local or an SSH profile, and Save."
    )

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.HELP, id="stream-help")
            yield DataTable(id="stream-table")
            yield Static("", id="stream-empty")
            # a row of its own: Download is about what was recorded, the row
            # below about the streams themselves
            with Horizontal(id="stream-recordings"):
                yield Label("Recordings:")
                yield Select(self._session_options(), value=ALL_RECORDINGS, allow_blank=False,
                             id="stream-session-select")
                yield Button("Download", variant="warning", id="stream-btn-download")
            with Horizontal(id="stream-actions"):
                yield Button("Start", variant="success", id="stream-btn-start")
                yield Button("Stop", variant="error", id="stream-btn-stop")
                yield Button("Logs", variant="primary", id="stream-btn-logs")
                yield Button("Probe", variant="warning", id="stream-btn-probe")
                yield Button("Record on/off", variant="default", id="stream-btn-record")
                yield Button("Start All", variant="success", id="stream-btn-start-all")
                yield Button("Stop All", variant="error", id="stream-btn-stop-all")
                yield Button("Refresh", variant="primary", id="stream-btn-refresh")

    def on_mount(self) -> None:
        table = self.query_one("#stream-table", DataTable)
        table.add_columns("Name", "SSH Profile", "Device", "Target", "Record", "Status")
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
        try:
            self.query_one("#stream-empty", Static).update(
                "" if self._streams else
                "No streams yet. Open the Config tab, expand Streams and press + Add Stream, then Save."
            )
        except Exception:
            pass
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
                # nobody records an external stream here: the console does not run its ffmpeg
                ("yes" if stream.ssh_profile else "n/a") if stream.record else "-",
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
                self._log(
                    f"[yellow]{stream.name} is an external stream (no SSH profile): someone else publishes it. "
                    f"{self._EXTERNAL_HINT.format(name=stream.name)}[/yellow]"
                )
            else:
                self._log("[yellow]Select a stream row first.[/yellow]")
        elif btn == "stream-btn-stop":
            stream = self._get_selected_stream()
            if stream and stream.ssh_profile:
                self._stop_stream(stream)
            elif stream and not stream.ssh_profile:
                self._log(
                    f"[yellow]{stream.name} is an external stream (no SSH profile): someone else publishes it. "
                    f"{self._EXTERNAL_HINT.format(name=stream.name)}[/yellow]"
                )
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
        elif btn == "stream-btn-record":
            stream = self._get_selected_stream()
            if stream is None:
                self._log("[yellow]Select a stream row first.[/yellow]")
            elif not stream.ssh_profile:
                self._log(
                    f"[yellow]{stream.name} is external: the console does not run its ffmpeg, so it "
                    f"cannot make it record. {self._EXTERNAL_HINT.format(name=stream.name)}[/yellow]"
                )
            elif self._statuses.get(stream.name, False):
                self._log(f"[yellow]Stop {stream.name} first: its ffmpeg was started without the change.[/yellow]")
            else:
                self.post_message(self.RecordToggleRequested(stream.name, not stream.record))
        elif btn == "stream-btn-download":
            try:
                choice = self.query_one("#stream-session-select", Select).value
            except Exception:
                choice = ALL_RECORDINGS
            if choice not in (ALL_RECORDINGS, Select.BLANK, None):
                streams = self.recorded_streams()
                if streams:
                    self.post_message(self.SessionDownloadRequested(str(choice), streams))
                else:
                    self._log(
                        "[yellow]Every stream here is external: the console did not start them, so no host "
                        "it manages holds a recording. The Stream Server may: Sessions → Export Recordings.[/yellow]"
                    )
                return
            stream = self._get_selected_stream()
            if stream is None:
                self._log("[yellow]Select a stream row first.[/yellow]")
            elif not stream.ssh_profile:
                self._log(
                    f"[yellow]{stream.name} is external: the console did not start it, so no host "
                    f"it manages holds a recording of it.[/yellow]"
                )
            elif stream.ssh_profile == "local":
                self._log(
                    f"[cyan]{stream.name} is captured on this machine: its recordings are already here, under "
                    f"{self._record_root(stream)}/streams-<date>/collection/{self._record_host_label(stream)}/.[/cyan]"
                )
            else:
                self.post_message(self.DownloadRequested(
                    stream.ssh_profile, self._record_root(stream), self._record_host_label(stream)))
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
        else:
            self._log("[cyan]Refreshing stream status from the current target config.[/cyan]")
        self.post_message(self.SessionChoicesRequested())
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

        record_dir = self._record_dir(stream)
        try:
            ffmpeg_cmd = _build_ffmpeg_cmd(stream, record_dir)
        except ValueError as e:
            self._log(f"[red]Cannot start {stream.name}: {e}[/red]")
            return
        record_path = _record_path(stream, record_dir) if record_dir else None
        tmux_cmd = _build_tmux_stream_cmd(session, ffmpeg_cmd, record_dir, record_path)
        target_label = "locally" if is_local else f"on {stream.ssh_profile}"
        self._log(f"[green]Starting {stream.name} {target_label}...[/green]")
        self._log(f"  {ffmpeg_cmd}")
        if record_dir:
            self._log(f"  Recording to {record_dir}/ on the streaming host")

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
                recorded_file = None
                if record_dir:
                    recorded_file = await loop.run_in_executor(
                        None,
                        lambda: _read_stream_record_path(profile, session, is_local),
                    )
                register_stream_start(
                    stream.name,
                    stream.target,
                    start_time,
                    project_dir=self._project_dir,
                    ssh_profile=stream.ssh_profile,
                    device=stream.device,
                    read_target=stream.read_target,
                    record_path=recorded_file or "",
                )
                self._log(f"[green]{stream.name} started.[/green]")
                self._log(f"  stream_start_time={start_time:.6f}")
                if recorded_file:
                    self._log(f"  recording={recorded_file}")
                elif record_dir:
                    self._log("[yellow]Could not read back the recording path; check the stream logs.[/yellow]")
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

        stop_cmd = _build_stop_stream_cmd(session)
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
                recorded = self._registered_record_path(stream.name)
                mark_stream_stopped(stream.name, project_dir=self._project_dir)
                self._statuses[stream.name] = False
                record_dir = self._record_dir(stream)
                if recorded:
                    # where Start put it: the card's Session may have changed since,
                    # and the file did not move with it
                    self._log(f"  Recording kept at {recorded}{self._fetch_hint(stream)}")
                elif record_dir:
                    self._log(f"  Recording kept under {record_dir}/{self._fetch_hint(stream)}")
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
        url = stream.read_url
        self._log(f"[cyan]Probing {url}...[/cyan]")
        loop = asyncio.get_event_loop()
        success, message = await loop.run_in_executor(None, _probe_stream_target, url)
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
