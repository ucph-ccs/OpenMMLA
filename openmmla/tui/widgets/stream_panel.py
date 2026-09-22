from __future__ import annotations

import asyncio
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Callable
from urllib.parse import urlsplit

from rich.cells import cell_len
from rich.markup import escape as rich_escape
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ItemGrid, Vertical, Horizontal
from textual.geometry import Region
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Static, Button, DataTable, Input, Label, OptionList

from openmmla.tui import capture_recordings, recordings
from openmmla.tui import devices as capture_devices
from openmmla.tui.schema.loader import StreamDef, load_streams
from openmmla.tui.ssh import get_profile_by_name, load_ssh_profiles, remote_platform, ssh_run_sync
from openmmla.tui.system_services import stream_server_path
from openmmla.tui.widgets.stream_recordings import StreamRecordingsScreen
from openmmla.utils.artifact_paths import (
    REMOTE_RECORD_ROOT, capture_day, capture_host_label, capture_record_dir, capture_record_root,
)
from openmmla.utils.constants import STREAM_URL_SCHEMES, stream_kind
from openmmla.utils.mac_desktop import window_script
from openmmla.utils.stream_registry import load_stream_registry, register_stream_start, mark_stream_stopped


STREAM_REMOTE_PATH = "/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _with_stream_path(command: str) -> str:
    """run stream commands with a predictable PATH for non-interactive SSH shells."""
    return f"export PATH={STREAM_REMOTE_PATH}:$PATH; {command}"


def _stream_app(url: object, server: dict) -> str:
    """the app a stream URL names, the path segment before the stream's own
    name (rtmp://server-01:1935/vfa/raspi5-05 -> vfa): which card a stream
    belongs with. Read as the Stream Server reads it when it speaks for the
    URL (rtmp, rtsp and srt alike), else off the URL's path."""
    path = stream_server_path(url, server)
    if path is None:
        path = urlsplit(str(url or "").strip()).path.lstrip("/")
    segments = [part for part in str(path).split("/") if part]
    return segments[0] if len(segments) > 1 else ""


def _tmux_session_name(stream_name: str) -> str:
    return f"mmla-stream-{stream_name}"


def _tmux_session_target(session: str) -> str:
    """the -t of a tmux command that takes a session. tmux also takes a bare
    name as the start of one (mmla-stream-cam-1 finds mmla-stream-cam-10 when
    cam-1 is not running); "=" asks for that name only. Quoted by hand, as zsh
    expands a word that starts with "=", and shlex.quote leaves it bare."""
    return "'=" + session.replace("'", "'\\''") + "'"


def _tmux_pane_target(session: str) -> str:
    """the same for a tmux command that takes a window or a pane: the session's current one."""
    return "'=" + session.replace("'", "'\\''") + ":'"


def _stream_file(session_name: str, suffix: str) -> str:
    """a file of a managed stream on its host, next to its .start and .record."""
    return f"$HOME/.openmmla/streams/{session_name}.{suffix}"


def _run_on_host(profile, command: str, timeout: float) -> subprocess.CompletedProcess:
    """run a shell command on a stream's host: this machine when profile is None."""
    if profile is None:
        return subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    return ssh_run_sync(profile, command, timeout=timeout)


# what a managed stream's host says of it (_stream_state_cmd), and "unknown"
# when the host did not answer
STREAM_RUNNING = "running"
STREAM_STARTING = "starting"
STREAM_EXITED = "exited"
STREAM_STOPPED = "stopped"
STREAM_UNKNOWN = "unknown"


def _noted_ffmpeg_cmd(session: str) -> str:
    """a shell test: the ffmpeg a Mac's Terminal window started, whose pid was
    noted, runs; it leaves the pid in $q. A pid noted by a run long gone may
    belong to another process by now, hence the name check."""
    return (
        f'{{ q=$(cat "{_stream_file(session, "pid")}" 2>/dev/null); '
        f'[ -n "$q" ] && ps -p "$q" -o comm= 2>/dev/null | grep -q ffmpeg; }}'
    )


def _stream_state_cmd(session: str) -> str:
    """print the state of a managed stream on its host. Its tmux session is no
    proof of it: the session outlives its ffmpeg, which may have stopped on
    its first line. RUNNING while ffmpeg runs (a child of the pane, or the one
    a Mac's Terminal window started, whose pid was noted: that one does not
    end with the session, so it counts without one), STARTING while that
    window is still to start it, EXITED when the session is all that is left
    (the pane keeps what ffmpeg said), STOPPED without a session."""
    noted = _noted_ffmpeg_cmd(session)
    return (
        f"if ! tmux has-session -t {_tmux_session_target(session)} 2>/dev/null; then "
        f"if {noted}; then echo RUNNING; else echo STOPPED; fi; else "
        f"p=$(tmux list-panes -t {_tmux_pane_target(session)} -F '#{{pane_pid}}' 2>/dev/null | head -n1); "
        f'if {{ [ -n "$p" ] && pgrep -P "$p" -x ffmpeg >/dev/null 2>&1; }} || {noted}; then echo RUNNING; '
        f'elif [ -e "{_stream_file(session, "opening")}" ]; then echo STARTING; '
        f"else echo EXITED; fi; fi"
    )


def _parse_stream_state(output: str) -> str | None:
    for word in (output or "").split():
        if word in ("RUNNING", "STARTING", "EXITED", "STOPPED"):
            return word.lower()
    return None


def _stream_state(profile, session: str) -> str | None:
    """the state of a managed stream on its host (profile None: this machine),
    None when the host did not answer."""
    try:
        result = _run_on_host(profile, _with_stream_path(_stream_state_cmd(session)), 8.0)
    except Exception:
        return None
    return _parse_stream_state(result.stdout)


# the last line the wrappers write into a stream's pane: what comes before it
# is what ffmpeg said last
_EXIT_MARK = "[OpenMMLA] ffmpeg exited with status"
# what the pane of a stream started from a Terminal window on a Mac says first
_DESKTOP_NOTE = (
    "ffmpeg is started from a Terminal window on this Mac's screen (over ssh macOS lets nothing use the "
    "camera or the microphone) and runs on when the window has closed; its output follows here, and C-c "
    "here stops it."
)
# what ffmpeg prints of itself on every start, and on a Mac of every camera
_FFMPEG_NOISE = ("ffmpeg version", "built with", "configuration:", "libav", "libsw", "libpostproc")
_MAC_CAMERA_WARNING = "NSCameraUseContinuityCameraDeviceType"


def _last_words(pane: str, lines: int = 8) -> list[str]:
    """what ffmpeg said last in a stream's pane, up to the wrapper's exit mark:
    its banner, the pane's own note, blank lines and the idle shell after it
    left out."""
    text = pane.splitlines()
    marks = [index for index, line in enumerate(text) if _EXIT_MARK in line]
    if marks:
        text = text[:marks[-1] + 1]
    kept = [
        line.rstrip() for line in text
        if line.strip() and not line.strip().startswith(_FFMPEG_NOISE)
        and _MAC_CAMERA_WARNING not in line and line.strip() != _DESKTOP_NOTE
    ]
    return kept[-lines:]


def _stream_pane(profile, session: str) -> str:
    """what a stream's pane shows, wrapped lines joined."""
    command = _with_stream_path(f"tmux capture-pane -t {_tmux_pane_target(session)} -p -J -S -300 2>/dev/null")
    try:
        return _run_on_host(profile, command, 10.0).stdout or ""
    except Exception:
        return ""


def _stream_platform(stream: StreamDef, profile) -> str:
    """"darwin" or "linux": the capture command the stream's host takes."""
    if stream.ssh_profile == "local":
        return "darwin" if sys.platform == "darwin" else "linux"
    return (remote_platform(profile) if profile is not None else "") or "linux"


def _needs_desktop_session(platform: str, is_local: bool) -> bool:
    """a Mac reached over ssh. macOS gives the camera and the microphone only to
    an app someone allowed, in the desktop session; anything started over ssh
    counts as sshd, which is never asked, and its ffmpeg waits for frames
    forever or records silence. Such a stream is started from a Terminal window
    on the Mac's own screen, where the Collection recorder runs too."""
    return platform == "darwin" and (not is_local or bool(os.environ.get("SSH_CONNECTION")))


# recording root on a remote streaming host when record_root is unset: the one
# the Collection card records under too (its recordings go to <session>/collection/,
# a stream's to streams/capture/<day>/)
STREAM_RECORD_ROOT = REMOTE_RECORD_ROOT

# seconds Stop waits for ffmpeg to finalize its files after Ctrl-C before the
# tmux session is killed
STREAM_STOP_GRACE_SECONDS = 8


def stream_record_root(ssh_profile: str, record_root: str, project_dir: str) -> str:
    """the folder on a stream's capture host that holds its streams/capture/
    tree: its record_root (~ spelled $HOME, which the remote shell expands),
    else artifacts/ of the project here for a stream captured on this machine,
    $HOME/artifacts on any other."""
    return capture_record_root(ssh_profile, record_root, project_dir)


def stream_host_label(ssh_profile: str) -> str:
    """the host folder a stream's recordings are filed under, below
    streams/capture/<day>/: this machine's short name for 'local', else the
    SSH profile (the one the Collection card uses)."""
    return capture_host_label(ssh_profile)


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


# the pane stays after ffmpeg, for Logs: it says how ffmpeg ended, then a shell
# waits (the variable keeps macOS's bash from telling of zsh as it starts)
_AFTER_FFMPEG = f'echo "{_EXIT_MARK} $?"; BASH_SILENCE_DEPRECATION_WARNING=1 exec bash'


def _prepare_stream_files(session: str) -> str:
    """forget what a previous run of the stream left (a start time read back
    before the new one is written would be the old one), and mark the stream
    as coming up until its ffmpeg is started: until then no ffmpeg is no sign
    that it stopped (_stream_state_cmd)."""
    stale = " ".join(_stream_file(session, suffix) for suffix in ("start", "record", "pid", "rc"))
    return f'mkdir -p $HOME/.openmmla/streams; rm -f {stale}; touch "{_stream_file(session, "opening")}"'


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
        f'rm -f "{_stream_file(session, "opening")}"; '
        f"{ffmpeg_cmd}; "
        f"{_AFTER_FFMPEG}"
    )
    return _with_stream_path(
        f"{_prepare_stream_files(session)}; "
        f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(inner_cmd)}"
    )


# seconds a Terminal window on a Mac has to start ffmpeg
DESKTOP_START_TIMEOUT = 30

def _build_desktop_stream_cmd(
    session: str,
    ffmpeg_cmd: str,
    record_dir: str | None = None,
    record_path: str | None = None,
    name: str = "",
) -> str:
    """the stream on a Mac reached over ssh (_needs_desktop_session): ffmpeg is
    started from a .command script that `open -a Terminal` runs in a window on
    the Mac's own screen, where macOS lets it use the camera. The script notes
    the start time and the recording path as the tmux wrapper does, leaves
    ffmpeg to a keeper in a session of its own (openmmla.utils.mac_desktop)
    and closes its window once ffmpeg runs: nothing stays on the screen, and
    no window closed by hand can stop a stream. The tmux session is still
    what the console manages: its pane follows ffmpeg's output, passes a C-c
    (Stop) on to that ffmpeg, and says how it ended."""
    if record_dir and record_path:
        record_part = f'mkdir -p "{record_dir}"\nprintf \'%s\\n\' "{record_path}" > "$F.record"\n'
    else:
        record_part = 'rm -f "$F.record"\n'
    script = window_script(
        f'"$HOME/.openmmla/streams/{session}"',
        ffmpeg_cmd,
        about=f"ffmpeg for the OpenMMLA stream {name}",
        setup=(
            f"export PATH={STREAM_REMOTE_PATH}:$PATH\n"
            "START_TIME=$(python3 -c \"import time; print('%.6f' % time.time())\" 2>/dev/null || date +%s)\n"
            "export START_TIME\n"
            "printf '%s\\n' \"$START_TIME\" > \"$F.start\"\n"
            f"{record_part}"
        ),
    )
    follower = (
        f'F="$HOME/.openmmla/streams/{session}"; '
        'if open -a Terminal "$F.command"; then '
        f'echo "{_DESKTOP_NOTE}"; '
        "trap 'kill -INT \"$(cat \"$F.pid\" 2>/dev/null)\" 2>/dev/null' INT TERM HUP; "
        f'i=0; while [ -e "$F.opening" ] && [ $i -lt {DESKTOP_START_TIMEOUT * 2} ]; do sleep 0.5; i=$((i+1)); done; '
        'if [ -e "$F.opening" ]; then rm -f "$F.opening"; '
        f'echo "The Terminal window did not start ffmpeg within {DESKTOP_START_TIMEOUT} s."; '
        'tail -n 5 "$F.log" 2>/dev/null; '
        'else tail -n +1 -f "$F.log" & t=$!; '
        'while kill -0 "$(cat "$F.pid" 2>/dev/null)" 2>/dev/null; do sleep 0.5; done; '
        f'sleep 1; kill $t 2>/dev/null; echo "{_EXIT_MARK} $(cat "$F.rc" 2>/dev/null)"; fi; '
        'else rm -f "$F.opening"; '
        "echo \"Could not open a Terminal window on this Mac's screen, the only place macOS lets ffmpeg "
        'use the camera and the microphone: is someone logged in there?"; fi; '
        "BASH_SILENCE_DEPRECATION_WARNING=1 exec bash"
    )
    command_file = _stream_file(session, "command")
    return _with_stream_path(
        f"{_prepare_stream_files(session)}; "
        f"cat > \"{command_file}\" <<'OPENMMLA_STREAM_EOF'\n{script}OPENMMLA_STREAM_EOF\n"
        f'chmod +x "{command_file}"; '
        f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(follower)}"
    )


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
    polls; the shell itself is what tmux reports as the pane's command. On a Mac
    reached over ssh ffmpeg was started from a Terminal window and is no child
    of the pane: it gets the Ctrl-C by its noted pid (the pane passes one on
    too, and it is reached so even when its session is gone), the wait polls
    that pid, and as that ffmpeg does not end with the session, one still there
    after the wait is killed."""
    pane = _tmux_pane_target(session)
    noted = _noted_ffmpeg_cmd(session)
    return (
        f"tmux send-keys -t {pane} C-c 2>/dev/null; "
        # a Terminal window that has not started its ffmpeg yet no longer will
        f'rm -f "{_stream_file(session, "opening")}"; '
        f'{noted} && kill -INT "$q" 2>/dev/null; '
        f"p=$(tmux list-panes -t {pane} -F '#{{pane_pid}}' 2>/dev/null | head -n1); i=0; "
        f'while {{ {{ [ -n "$p" ] && pgrep -P "$p" -x ffmpeg >/dev/null 2>&1; }} || {noted}; }} '
        f"&& [ $i -lt {STREAM_STOP_GRACE_SECONDS * 2} ]; do sleep 0.5; i=$((i+1)); done; "
        f'{noted} && kill -9 "$q" 2>/dev/null; rm -f "{_stream_file(session, "pid")}"; '
        f"tmux kill-session -t {_tmux_session_target(session)} 2>/dev/null; "
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


def _stream_kind(stream: StreamDef, default: str = "video") -> str:
    """'audio' or 'video': its kind, else a udp/tcp target or a device that
    captures sound, else `default`, the card's (constants.stream_kind, the rule
    a session's sources go by too). A Mac's microphone pushed over RTMP names
    no device, so only the card that pulls it can tell: audio on ASR."""
    return stream_kind(stream, default)


# a Mac's first camera and first microphone (ffmpeg -f avfoundation -list_devices true -i "")
MAC_CAPTURE_DEVICE = "0"


def _avfoundation_input(device: str, kind: str) -> str:
    """the -i of AVFoundation, "<camera>:<microphone>", each an index or a name:
    a video stream takes no sound, an audio stream no picture."""
    device = str(device).strip()
    if ":" in device:
        return device
    return f"{device}:none" if kind == "video" else f":{device}"


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


def _record_path(stream: StreamDef, record_dir: str, kind: str = "") -> str:
    """the recording file on the streaming host; ${START_TIME} expands in the tmux
    wrapper so the name carries the capture-side start time the file source expects.
    `kind`: the stream's as its card tells it (_stream_kind with the card's default)."""
    extension = "wav" if (kind or _stream_kind(stream)) == "audio" else "mkv"
    return f"{record_dir.rstrip('/')}/{stream.name}_${{START_TIME}}.{extension}"


def _build_ffmpeg_cmd(stream: StreamDef, record_dir: str | None = None, platform: str = "linux",
                      kind: str = "") -> str:
    """build the ffmpeg command string from a stream definition.

    With record_dir the same capture is also written to a file there, so a
    session keeps its raw recording next to the live stream: one encode with two
    outputs, via the tee muxer for video and a second PCM output for audio.
    platform is the capture host's: "darwin" captures through AVFoundation, as
    V4L2 and ALSA are Linux's. kind is the stream's as its card tells it
    (_stream_kind with the card's default); empty, the rule without a card.
    """
    kind = kind or _stream_kind(stream)
    target = stream.target
    record_to = _record_path(stream, record_dir, kind) if record_dir else None
    mac = platform == "darwin"

    if kind == "audio":
        rate = stream.rate or 16000
        channels = stream.channels or 1
        if mac:
            device = _avfoundation_input(stream.device or MAC_CAPTURE_DEVICE, "audio")
            capture = f"ffmpeg -f avfoundation -i {shlex.quote(device)} "
            # AVFoundation delivers what the microphone gives: each output converts it
            convert = f"-ac {channels} -ar {rate} "
        else:
            device = stream.device or "hw:0,0"
            capture = f"ffmpeg -f alsa -ac {channels} -ar {rate} -i {device} "
            convert = ""
        if target.startswith(("udp://", "tcp://")):
            # the ASR base reads a header-less PCM byte stream on udp/tcp (see
            # AudioStream._read_socket_chunk), so send raw samples rather than AAC
            # in FLV. ffmpeg flushes once per ALSA period and keeps udp datagrams
            # below the MTU; the base re-frames whatever arrives to its own chunk_size.
            fmt = _pcm_sample_format(stream.format)
            proto = "udp" if target.startswith("udp://") else "tcp"
            addr = target.split("://", 1)[1]
            live = f"{convert}-c:a pcm_{fmt} -f {fmt} {proto}://{addr}"
            file_codec = f"pcm_{fmt}"
        else:
            muxer, options = _publish_muxer(target)
            live = f"{convert}-c:a aac -b:a 128k -f {muxer} {_cli_options(options)}{target}"
            file_codec = "pcm_s16le"
        if not record_to:
            return capture + live
        return f"{capture}-map 0:a {live} -map 0:a {convert}-c:a {file_codec} -f wav {record_to}"

    codec = stream.codec or "libx264"
    resolution = stream.resolution or "1920x1080"
    fps = stream.fps or 30
    bitrate = stream.bitrate or "1M"
    peak = _double_rate(bitrate)
    muxer, options = _publish_muxer(target)
    if mac:
        # nv12 is 4:2:0, which Mac cameras deliver (one that does not gets its
        # own format from ffmpeg, hence the yuv420p the players expect). Unlike
        # V4L2, AVFoundation states no frame rate: with wallclock timestamps
        # ffmpeg took 1000k fps and duplicated frames without end, so the
        # output is held at the camera's
        device = _avfoundation_input(stream.device or MAC_CAPTURE_DEVICE, "video")
        capture = (
            f"-f avfoundation -pixel_format nv12 -framerate {fps} -video_size {resolution} "
            f"-i {shlex.quote(device)} "
        )
        output = f"-pix_fmt yuv420p -r {fps} "
    else:
        device = stream.device or "/dev/video0"
        capture = f"-f v4l2 -input_format mjpeg -framerate {fps} -video_size {resolution} -i {device} "
        output = ""
    encode = (
        f"ffmpeg -fflags +genpts -use_wallclock_as_timestamps 1 {capture}"
        f"-c:v {codec} {output}-preset ultrafast -tune zerolatency "
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


# the columns of the Streams table whose cells are dropdowns: the machine that
# captures the stream, its device on that machine, and whether it records there
PROFILE_COLUMN = 1
DEVICE_COLUMN = 2
RECORD_COLUMN = 4


class StreamTable(DataTable):
    """the Streams table. A click on a row's SSH Profile cell, or Enter on a
    row, asks for the list of machines that can capture the stream; a click
    on its Device cell, for the devices of that machine; on its Record cell,
    for whether it records there."""

    class ProfileMenuRequested(Message):
        def __init__(self, row: int) -> None:
            super().__init__()
            self.row = row

    class DeviceMenuRequested(Message):
        def __init__(self, row: int) -> None:
            super().__init__()
            self.row = row

    class RecordMenuRequested(Message):
        def __init__(self, row: int) -> None:
            super().__init__()
            self.row = row

    def on_click(self, event: events.Click) -> None:
        # runs before DataTable's own handler, which moves the cursor to the row
        meta = event.style.meta
        row = meta.get("row")
        if not isinstance(row, int) or row < 0 or meta.get("out_of_bounds", False):
            return
        if meta.get("column") == PROFILE_COLUMN:
            self.post_message(self.ProfileMenuRequested(row))
        elif meta.get("column") == DEVICE_COLUMN:
            self.post_message(self.DeviceMenuRequested(row))
        elif meta.get("column") == RECORD_COLUMN:
            self.post_message(self.RecordMenuRequested(row))

    def action_select_cursor(self) -> None:
        super().action_select_cursor()
        if self.row_count:
            self.post_message(self.ProfileMenuRequested(self.cursor_row))

    def cell_region(self, row: int, column: int) -> Region:
        """where a cell is on the screen, for a list to open under it."""
        columns = self.ordered_columns
        if len(columns) <= column:
            return Region(self.content_region.x, self.content_region.y, 0, 1)
        x = sum(col.get_render_width(self) for col in columns[:column])
        y = (self.header_height if self.show_header else 0) + sum(r.height for r in self.ordered_rows[:row])
        area = self.content_region
        return Region(area.x + x - round(self.scroll_x), area.y + y - round(self.scroll_y),
                      columns[column].get_render_width(self), 1)

    def profile_cell_region(self, row: int) -> Region:
        return self.cell_region(row, PROFILE_COLUMN)

    def device_cell_region(self, row: int) -> Region:
        return self.cell_region(row, DEVICE_COLUMN)

    def record_cell_region(self, row: int) -> Region:
        return self.cell_region(row, RECORD_COLUMN)


class StreamProfileMenu(ModalScreen):
    """the dropdown of an SSH Profile or a Device cell: opens under the cell,
    like the list of a Select. Enter or a click picks one; Escape or a click
    beside the list leaves the row as it was, and the screen returns None."""

    DEFAULT_CSS = """
    StreamProfileMenu {
        background: transparent;
    }
    StreamProfileMenu > OptionList {
        border: tall $border;
        background: $surface;
    }
    """

    BINDINGS = [Binding("escape", "close", "Close", show=False)]

    # options shown at once; more scroll
    VISIBLE_OPTIONS = 10

    def __init__(self, options: list[tuple[str, str]], current: str, anchor: Region) -> None:
        super().__init__()
        self._labels = [label for label, _value in options]
        self._values = [value for _label, value in options]
        self._current = current
        self._anchor = anchor

    def compose(self) -> ComposeResult:
        yield OptionList(*self._labels)

    def on_mount(self) -> None:
        menu = self.query_one(OptionList)
        screen_width, screen_height = self.app.size
        shown = min(len(self._labels), self.VISIBLE_OPTIONS)
        height = shown + 2
        # label, the padding of option and list, the border, and a scrollbar when one is needed
        width = max(cell_len(label) for label in self._labels) + 6 + (2 if shown < len(self._labels) else 0)
        width = min(max(width, self._anchor.width), screen_width)
        x = max(0, min(self._anchor.x, screen_width - width))
        # under the cell, or above it when the screen ends first
        y = self._anchor.bottom if self._anchor.bottom + height <= screen_height else max(0, self._anchor.y - height)
        menu.styles.width = width
        menu.styles.height = height
        menu.styles.offset = (x, y)
        if self._current in self._values:
            menu.highlighted = self._values.index(self._current)
        menu.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self._values[event.option_index])

    def on_click(self, event: events.Click) -> None:
        if not self.query_one(OptionList).region.contains(event.screen_x, event.screen_y):
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class StreamDeviceInput(ModalScreen):
    """a device the machine did not list, typed in a box under its cell. Enter
    gives what was typed (empty: none), Escape or a click beside it None."""

    DEFAULT_CSS = """
    StreamDeviceInput {
        background: transparent;
    }
    StreamDeviceInput > Input {
        width: 40;
    }
    """

    BINDINGS = [Binding("escape", "close", "Close", show=False)]

    def __init__(self, current: str, anchor: Region, placeholder: str = "") -> None:
        super().__init__()
        self._current = current
        self._anchor = anchor
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        yield Input(value=self._current, placeholder=self._placeholder)

    def on_mount(self) -> None:
        box = self.query_one(Input)
        screen_width, screen_height = self.app.size
        width = min(max(40, self._anchor.width), screen_width)
        box.styles.width = width
        box.styles.offset = (max(0, min(self._anchor.x, screen_width - width)),
                             self._anchor.bottom if self._anchor.bottom + 3 <= screen_height
                             else max(0, self._anchor.y - 3))
        box.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(event.value.strip())

    def on_click(self, event: events.Click) -> None:
        if not self.query_one(Input).region.contains(event.screen_x, event.screen_y):
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


# a Device list's last option: type one the machine did not list
TYPE_DEVICE = "\x00type"


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
    #stream-recordings Button {
        margin: 0 1;
    }
    /* the buttons wrap onto another line instead of running off the right
       edge when the pane is narrow: ItemGrid fits as many columns as there
       is room for, and gives up one at a time as it shrinks */
    #stream-actions {
        height: auto;
        padding: 1 1;
        grid-gutter: 1 2;
    }
    #stream-actions Button {
        width: 1fr;
        height: 3;
    }
    """

    def __init__(
        self,
        streams: list[StreamDef],
        config_path: str = "",
        project_dir: str | None = None,
        stream_server: Callable[[], dict] | None = None,
        default_kind: str = "video",
        app: str = "",
    ) -> None:
        super().__init__()
        self._streams = list(streams)
        self._config_path = config_path
        # the app this card's streams live under on the Stream Server (ips,
        # asr, vfa): which left-over captures are its own to stop
        self._app = str(app or "").strip().lower()
        # what a stream with no kind that neither its target nor its device
        # tells is: the card's ("audio" on ASR, whose Mac microphone pushed
        # over RTMP names no device; "video" on IPS and VFA)
        self._default_kind = default_kind if default_kind in ("audio", "video") else "video"
        self._project_dir = project_dir or _project_root_from_config(config_path)
        # whether a stream's ffmpeg runs (or is coming up), and what its host
        # said of it in full (STREAM_*)
        self._statuses: dict[str, bool] = {}
        self._states: dict[str, str] = {}
        # System Settings -> Stream Server, and what it says of each stream it
        # carries: "live", "idle" (nobody publishes it) or "unknown" (no answer)
        self._stream_server = stream_server
        self._live: dict[str, str] = {}
        # what each capture host said of its devices, until Refresh
        self._device_answers: dict[str, capture_devices.Devices] = {}
        # captures this console started that the config's Streams no longer
        # name, listed below them so they can be stopped (_left_over_captures)
        self._left_over: list[StreamDef] = []

    def _rows(self) -> list[StreamDef]:
        """what the table shows: this config's Streams, then the captures left
        over from entries it no longer has."""
        return self._streams + self._left_over

    def _is_left_over(self, stream: StreamDef) -> bool:
        return any(other.name == stream.name for other in self._left_over)

    def _left_over_captures(self) -> list[StreamDef]:
        """captures this console started that this config's Streams no longer
        name: an entry deleted (or renamed) on the Config tab leaves its
        ffmpeg and its tmux session running, and with no row there is nothing
        to stop them with. They come from the stream registry, which notes
        every stream started from here and the machine it runs on, and are
        kept to the apps of this card's streams so that another card's stream
        stays on its own tab."""
        try:
            entries = load_stream_registry(self._project_dir).get("streams", {})
        except Exception:
            return []
        known = {stream.name for stream in self._streams}
        server = self._server_address()
        apps = {app for app in
                [self._app] + [_stream_app(stream.read_url, server) for stream in self._streams] if app}
        if not apps:
            return []  # nothing to tell this card's streams from another's
        found = []
        for name, entry in entries.items():
            if not isinstance(entry, dict) or entry.get("status") != "running" or str(name) in known:
                continue
            if not str(entry.get("ssh_profile") or "").strip():
                continue  # no machine to stop it on
            target = str(entry.get("target") or "")
            read_target = str(entry.get("read_target") or "")
            if _stream_app(read_target or target, server) not in apps:
                continue  # another card's stream
            found.append(StreamDef(
                name=str(name),
                ssh_profile=str(entry.get("ssh_profile") or ""),
                device=str(entry.get("device") or ""),
                target=target,
                read_target=read_target,
                kind=str(entry.get("kind") or ""),
            ))
        return found

    def capture_streams(self) -> list[capture_recordings.CaptureStream]:
        """every stream of this card the console runs, as Manage and the keep
        time see them: where they record, and for how long their recordings
        stay there."""
        return [
            capture_recordings.CaptureStream(
                stream.name,
                capture_recordings.CaptureFolder(
                    stream.ssh_profile, self._record_root(stream), self._record_host_label(stream)),
                self._kind(stream), stream.record_keep_days)
            for stream in self._streams if stream.ssh_profile
        ]

    def _kind(self, stream: StreamDef) -> str:
        """'audio' or 'video' of a stream of this card: everything here that
        depends on it (the capture, the recording's file and folder, Manage
        and the copies it makes) goes by this."""
        return _stream_kind(stream, self._default_kind)

    def _live_record_paths(self) -> set[str]:
        """the files the streams started from here are writing, as their Start noted them."""
        try:
            entries = load_stream_registry(self._project_dir).get("streams", {})
        except Exception:
            return set()
        return {
            str(entry["record_path"]).strip() for entry in entries.values()
            if isinstance(entry, dict) and entry.get("status") == "running" and entry.get("record_path")
        }

    @staticmethod
    def _record_day() -> str:
        """the folder a stream's recording is filed under: the day (the console's
        date, YYYY-MM-DD), never a session. A stream outlives sessions and is
        shared by them (several groups in one room pull the same camera), so its
        recording belongs to none of them; the start time in the file name is
        what ties it to a session."""
        return capture_day()

    @staticmethod
    def _record_host_label(stream: StreamDef) -> str:
        """host folder under streams/capture/<day>/, the same label the Collection card uses."""
        return stream_host_label(stream.ssh_profile)

    def _record_root(self, stream: StreamDef) -> str:
        """the folder on the capture host that holds streams/capture/<day>/."""
        return stream_record_root(stream.ssh_profile, stream.record_root, self._project_dir)

    def _record_dir(self, stream: StreamDef) -> str | None:
        """recording folder on the streaming host,
        <record_root>/streams/capture/<day>/<host label>/<video|audio>, or None
        when the stream does not record."""
        if not stream.record:
            return None
        return capture_record_dir(
            self._record_root(stream), self._record_day(), self._record_host_label(stream), self._kind(stream))

    @staticmethod
    def _on_host(stream: StreamDef) -> str:
        """the end of the line naming a stopped stream's recording: the host it is
        on, unless that is this machine."""
        if stream.ssh_profile == "local":
            return "."
        return f" on {stream.ssh_profile}."

    def _registered_record_path(self, stream_name: str) -> str:
        """the recording file noted when the stream was started, if it records."""
        try:
            entry = load_stream_registry(self._project_dir).get("streams", {}).get(stream_name)
        except Exception:
            return ""
        if not isinstance(entry, dict) or entry.get("status") != "running":
            return ""
        return str(entry.get("record_path") or "").strip()

    class RecordToggleRequested(Message):
        """a row's Record was picked: whether the stream also records on the
        machine that captures it. The launcher owns the config of the host the
        card is on and writes it there."""

        def __init__(self, stream_name: str, record: bool) -> None:
            super().__init__()
            self.stream_name = stream_name
            self.record = record

    class KeepDaysChangeRequested(Message):
        """Manage set how long the recordings of the streams here stay on their
        capture hosts; the launcher writes it into the config of the host the
        card is on, as it does for the Record column."""

        def __init__(self, keep_days: dict[str, int]) -> None:
            super().__init__()
            self.keep_days = keep_days

    class SshProfileChangeRequested(Message):
        """a row's SSH Profile was picked: the machine that runs the stream's
        ffmpeg, "" for an external stream. The launcher writes it into the
        config of the host the card is on, as it does for the Record column."""

        def __init__(self, stream_name: str, ssh_profile: str) -> None:
            super().__init__()
            self.stream_name = stream_name
            self.ssh_profile = ssh_profile

    class DeviceChangeRequested(Message):
        """a row's Device was picked: the camera or microphone its ffmpeg opens
        on its machine, "" for none (Start takes the first). The launcher
        writes it into the config of the host the card is on."""

        def __init__(self, stream_name: str, device: str) -> None:
            super().__init__()
            self.stream_name = stream_name
            self.device = device

    # three lines: it stands above the table every time the tab is opened
    HELP = (
        "A stream is a Streams entry of this card's config (Config tab, + Add Stream): ffmpeg publishes a camera "
        "or microphone to the Stream Server, the bases pull it. Started once, it serves any number of sessions. "
        "Status is its ffmpeg (Exited: it stopped by itself, Logs says why); Stream Server, what the server receives. "
        "A row marked not in Streams is a capture left over from an entry the config no longer has: Stop is all "
        "it is here for.\n"
        "SSH Profile (click it, or Enter on a row): the machine whose ffmpeg publishes it, - for a stream someone "
        "else publishes; Device (click it): its camera or microphone there; Record (click it): whether it also "
        "records there. The Stream Server records on its side whatever reaches it (its card, Config tab).\n"
        "Recordings are filed by day on the capture device, not by session. Manage lists them there, deletes "
        "them, and sets how long they are kept. A session's part of them (and of the Stream Server's) is "
        "Sessions → Export Streams."
    )

    # what a Record cell offers: record on the capture device next to the push, or not
    _RECORD_OPTIONS = [
        ("yes  (also record on the capture device)", "yes"),
        ("no  (publish only)", "no"),
    ]

    # how a stream stops being external
    _EXTERNAL_HINT = (
        "To run it from here, click its SSH Profile (or press Enter on its row) and pick the machine its "
        "device is attached to, local or an SSH profile."
    )

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.HELP, id="stream-help")
            yield StreamTable(id="stream-table")
            yield Static("", id="stream-empty")
            # a row of its own: Manage is about what was recorded, the row
            # below about the streams themselves
            with Horizontal(id="stream-recordings"):
                yield Label("Recordings:")
                yield Button("Manage", variant="primary", id="stream-btn-manage")
            with ItemGrid(id="stream-actions", min_column_width=13, max_column_width=16):
                yield Button("Start", variant="success", id="stream-btn-start")
                yield Button("Stop", variant="error", id="stream-btn-stop")
                yield Button("Logs", variant="primary", id="stream-btn-logs")
                yield Button("Probe", variant="warning", id="stream-btn-probe")
                yield Button("Start All", variant="success", id="stream-btn-start-all")
                yield Button("Stop All", variant="error", id="stream-btn-stop-all")
                yield Button("Refresh", variant="primary", id="stream-btn-refresh")

    def on_mount(self) -> None:
        table = self.query_one("#stream-table", DataTable)
        table.add_columns("Name", "SSH Profile", "Device", "Target", "Record", "Status", "Stream Server")
        table.cursor_type = "row"
        self._rebuild_table()
        # even with no Streams entry there may be a capture left over from one
        self._refresh_all()

    def _log(self, msg: str) -> None:
        self.post_message(self.StreamLog(msg))

    def _refresh_all(self) -> None:
        self.run_worker(self._async_refresh_all(), exclusive=True)

    def _set_state(self, name: str, state: str | None) -> None:
        """note what a stream's host said of it; None: it did not answer."""
        self._states[name] = state or STREAM_UNKNOWN
        self._statuses[name] = state in (STREAM_RUNNING, STREAM_STARTING)

    async def _async_refresh_all(self) -> None:
        loop = asyncio.get_event_loop()
        self._left_over = await asyncio.to_thread(self._left_over_captures)
        for stream in self._rows():
            if not stream.ssh_profile:
                continue
            profile = None
            if stream.ssh_profile != "local":
                profile = get_profile_by_name(stream.ssh_profile)
                if profile is None:
                    self._set_state(stream.name, None)
                    continue
            state = await loop.run_in_executor(None, _stream_state, profile, _tmux_session_name(stream.name))
            self._set_state(stream.name, state)
        # a left-over whose host has nothing left of it (no session, no
        # ffmpeg) is over: the registry is told so, and its row goes
        done = [stream for stream in self._left_over
                if stream.ssh_profile and self._states.get(stream.name) == STREAM_STOPPED]
        for stream in done:
            mark_stream_stopped(stream.name, project_dir=self._project_dir)
        if done:
            names = {stream.name for stream in done}
            self._left_over = [stream for stream in self._left_over if stream.name not in names]
        self._live = await self._server_states()
        self._rebuild_table()

    def _server_path(self, stream: StreamDef, server: dict) -> str | None:
        """the stream's path on the Stream Server of System Settings, None when
        it goes elsewhere (another server, or udp/tcp to an ASR base)."""
        return stream_server_path(stream.target, server) or stream_server_path(stream.read_url, server)

    def _server_address(self) -> dict:
        try:
            return dict(self._stream_server() or {}) if self._stream_server else {}
        except Exception:
            return {}

    async def _server_states(self) -> dict[str, str]:
        """what the Stream Server says of each stream it carries: "live" while
        someone publishes it, "idle" when nobody does, "unknown" when it does
        not answer. A stream that goes elsewhere has no entry."""
        server = self._server_address()
        host = str(server.get("host") or "").strip()
        paths = {stream.name: self._server_path(stream, server) for stream in self._rows()} if host else {}
        paths = {name: path for name, path in paths.items() if path}
        if not paths:
            return {}
        try:
            api_port = int(server.get("api_port") or recordings.API_PORT)
            live = await asyncio.to_thread(recordings.live_paths, host, api_port)
        except (recordings.RecordingsError, ValueError):
            return {name: "unknown" for name in paths}
        return {name: "live" if path in live else "idle" for name, path in paths.items()}

    def _rebuild_table(self) -> None:
        try:
            table = self.query_one("#stream-table", DataTable)
        except Exception:
            return
        # clear() puts the cursor back on the first row: it stays on its stream
        try:
            selected = str(table.get_row_at(table.cursor_row)[0]) if table.row_count else ""
        except Exception:
            selected = ""
        table.clear()
        rows = self._rows()
        try:
            self.query_one("#stream-empty", Static).update(
                "" if rows else
                "No streams yet. Open the Config tab, expand Streams, press + Add Stream and Save; "
                "then pick its SSH Profile and Device here."
            )
        except Exception:
            pass
        if not rows:
            return
        # the arrows line up at the right edge of the column, under its heading
        width = max([len("SSH Profile") - 3] + [cell_len(stream.ssh_profile or "-") for stream in rows])
        device_width = max([len("Device") - 3] + [cell_len(stream.device or "-") for stream in rows])
        record_width = max([len("Record") - 3] + [cell_len(self._record_cell(stream)) for stream in rows])
        for stream in rows:
            # a left-over capture has no entry to edit: its cells are plain,
            # and only Stop, Logs and Probe do anything on its row
            left_over = self._is_left_over(stream)
            record = self._record_cell(stream)
            state = self._states.get(stream.name)
            if not stream.ssh_profile:
                status = "External"
            elif self._statuses.get(stream.name, False):
                status = "Starting" if state == STREAM_STARTING else "Running"
            elif state == STREAM_EXITED:
                # the tmux session outlived its ffmpeg
                status = "Exited"
            elif state == STREAM_UNKNOWN:
                status = "No answer"
            else:
                status = "Stopped"
            live = self._live.get(stream.name)
            profile = stream.ssh_profile or "-"
            table.add_row(
                self._name_cell(stream),
                # drawn as the dropdown it is
                profile if left_over else
                Text.assemble(profile + " " * (width - cell_len(profile)), ("  ▾", "dim")),
                # a dropdown of the devices of that machine; an external stream's is its publisher's
                (stream.device or "-") if left_over else
                Text.assemble((stream.device or "-") + " " * (device_width - cell_len(stream.device or "-")),
                              ("  ▾", "dim")) if stream.ssh_profile else "-",
                stream.target,
                # a dropdown too; nobody records an external stream here, as the
                # console does not run its ffmpeg
                "n/a" if left_over else
                Text.assemble(record + " " * (record_width - cell_len(record)),
                              ("  ▾", "dim")) if stream.ssh_profile else "n/a",
                status,
                Text("● live", "green") if live == "live" else
                Text("○ not live", "dim") if live == "idle" else
                Text("no answer", "dim") if live == "unknown" else "-",
            )
        for row, stream in enumerate(rows):
            if str(self._name_cell(stream)) == selected:
                table.move_cursor(row=row)
                break

    def _name_cell(self, stream: StreamDef):
        """the Name column: a capture the config's Streams no longer name says
        so, as it is only there to be stopped."""
        if not self._is_left_over(stream):
            return stream.name
        return Text.assemble(stream.name, ("  (not in Streams)", "yellow"))

    def _left_over_note(self, stream: StreamDef) -> str:
        """why a left-over row takes no edits: its Streams entry is gone, so
        there is nothing to write a machine, a device or Record into."""
        return (
            f"[yellow]{stream.name} is a capture left over from a Streams entry this config no longer has: "
            f"Stop it (and Logs to see what it was doing), or add the entry back on the Config tab to run "
            f"it from here again.[/yellow]"
        )

    @staticmethod
    def _profile_options(current: str) -> list[tuple[str, str]]:
        """what an SSH Profile cell offers: this machine, every SSH profile, and
        none, for a stream someone else publishes. (label, value) pairs."""
        names = ["local"] + [profile.name for profile in load_ssh_profiles()]
        options = [("local  (this machine)" if name == "local" else name, name) for name in names]
        if current and current not in names:
            # renamed or deleted since: shown as what it is, and kept unless another is picked
            options.append((f"{current}  (not in the list any more)", current))
        options.append(("-  (external: someone else publishes it)", ""))
        return options

    def on_stream_table_profile_menu_requested(self, event: StreamTable.ProfileMenuRequested) -> None:
        event.stop()
        rows = self._rows()
        if not 0 <= event.row < len(rows):
            return
        stream = rows[event.row]
        if self._is_left_over(stream):
            self._log(self._left_over_note(stream))
            return
        if self._statuses.get(stream.name, False):
            where = "this machine" if stream.ssh_profile == "local" else stream.ssh_profile
            self._log(
                f"[yellow]Stop {stream.name} first: its ffmpeg runs on {where}, and Stop looks for it on "
                f"the machine its SSH Profile names.[/yellow]"
            )
            return
        table = self.query_one("#stream-table", StreamTable)
        menu = StreamProfileMenu(self._profile_options(stream.ssh_profile), stream.ssh_profile,
                                 table.profile_cell_region(event.row))

        def picked(profile: str | None) -> None:
            if profile is not None and profile != stream.ssh_profile:
                self.post_message(self.SshProfileChangeRequested(stream.name, profile))

        self.app.push_screen(menu, picked)

    def on_stream_table_device_menu_requested(self, event: StreamTable.DeviceMenuRequested) -> None:
        """the devices of the machine the row's SSH Profile names, of the
        stream's kind: asked once and kept until Refresh, then offered under
        the cell."""
        event.stop()
        rows = self._rows()
        if not 0 <= event.row < len(rows):
            return
        stream = rows[event.row]
        if self._is_left_over(stream):
            self._log(self._left_over_note(stream))
            return
        if not stream.ssh_profile:
            self._log(f"[yellow]{stream.name} is external: its device is up to whoever publishes it. "
                      f"{self._EXTERNAL_HINT}[/yellow]")
            return
        kind = self._kind(stream)
        answer = self._device_answers.get(stream.ssh_profile)
        if answer is not None and (kind in answer.found or kind in answer.problems):
            self._open_device_menu(stream, kind, answer)
            return
        where = "this machine" if stream.ssh_profile == "local" else stream.ssh_profile
        self._log(f"Asking {where} for its {'microphones' if kind == 'audio' else 'cameras'}...")
        self.run_worker(self._ask_devices(stream, kind), group="stream-devices", exclusive=True)

    async def _ask_devices(self, stream: StreamDef, kind: str) -> None:
        host = stream.ssh_profile
        answer = await asyncio.to_thread(capture_devices.list_devices, host, {kind}, self._project_dir or "")
        known = self._device_answers.setdefault(host, answer)
        if known is not answer:
            known.found.update(answer.found)
            known.problems.update(answer.problems)
            known.platform = known.platform or answer.platform
        # the row may have gone, or moved to another machine, while its host was asked
        current = next((s for s in self._streams if s.name == stream.name), None)
        if self.is_attached and current is not None and current.ssh_profile == host:
            self._open_device_menu(current, kind, known)

    def _open_device_menu(self, stream: StreamDef, kind: str, answer: capture_devices.Devices) -> None:
        where = "this machine" if stream.ssh_profile == "local" else stream.ssh_profile
        options = list(answer.options(kind))
        if not options:
            # nothing listed: why (the host could not be asked, or has none of that kind)
            problem = answer.problems.get(kind)
            self._log(f"[yellow]{rich_escape(where)}: {rich_escape(problem)}[/yellow]" if problem else
                      f"[yellow]No {kind} device found on {rich_escape(where)}.[/yellow]")
        current = stream.device or ""
        if current and not any(value == current for _label, value in options):
            options.append((f"{current}  (not found on {where})", current))
        options.append(("-  (none: Start takes the first)", ""))
        options.append(("type another…", TYPE_DEVICE))
        row = next((i for i, s in enumerate(self._streams) if s.name == stream.name), 0)
        table = self.query_one("#stream-table", StreamTable)
        anchor = table.device_cell_region(row)

        def changed(device: str | None) -> None:
            if device is None or device == current:
                return
            self.post_message(self.DeviceChangeRequested(stream.name, device))
            if self._statuses.get(stream.name, False):
                self._log(f"[yellow]{stream.name} runs now: it opens {device or 'its first device'} "
                          f"at its next Start.[/yellow]")

        def picked(device: str | None) -> None:
            if device == TYPE_DEVICE:
                example = "hw:1,0 or :0" if kind == "audio" else "/dev/video0 or 0"
                self.app.push_screen(StreamDeviceInput(current, anchor, f"a {kind} device, e.g. {example}"), changed)
                return
            changed(device)

        self.app.push_screen(StreamProfileMenu(options, current, anchor), picked)

    def on_stream_table_record_menu_requested(self, event: StreamTable.RecordMenuRequested) -> None:
        """a Record cell was clicked: whether the stream also records on the
        machine that captures it, next to the push. The keep time of those
        recordings stays Manage's, for every stream of the card at once."""
        event.stop()
        rows = self._rows()
        if not 0 <= event.row < len(rows):
            return
        stream = rows[event.row]
        if self._is_left_over(stream):
            self._log(self._left_over_note(stream))
            return
        if not stream.ssh_profile:
            self._log(
                f"[yellow]{stream.name} is external: the console does not run its ffmpeg, so it cannot "
                f"make it record. {self._EXTERNAL_HINT}[/yellow]"
            )
            return
        if self._statuses.get(stream.name, False):
            self._log(f"[yellow]Stop {stream.name} first: its ffmpeg was started without the change.[/yellow]")
            return
        table = self.query_one("#stream-table", StreamTable)
        menu = StreamProfileMenu(self._RECORD_OPTIONS, "yes" if stream.record else "no",
                                 table.record_cell_region(event.row))

        def picked(choice: str | None) -> None:
            if choice is not None and (choice == "yes") != bool(stream.record):
                self.post_message(self.RecordToggleRequested(stream.name, choice == "yes"))

        self.app.push_screen(menu, picked)

    def _get_selected_stream(self) -> StreamDef | None:
        try:
            table = self.query_one("#stream-table", DataTable)
            rows = self._rows()
            row_key = table.cursor_row
            if row_key < 0 or row_key >= len(rows):
                return None
            return rows[row_key]
        except Exception:
            return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id or ""
        if btn == "stream-btn-start":
            stream = self._get_selected_stream()
            if stream and self._is_left_over(stream):
                self._log(self._left_over_note(stream))
            elif stream and stream.ssh_profile:
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
        elif btn == "stream-btn-manage":
            self._open_recordings_manager()
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
            for s in self._rows():
                if s.ssh_profile:
                    self._stop_stream(s)
        elif btn == "stream-btn-refresh":
            # the machines are asked for their devices again, at the next Device list
            self._device_answers.clear()
            self._reload_and_refresh()

    def _reload_and_refresh(self) -> None:
        """re-read streams from config.yml and refresh status."""
        if self._config_path:
            self._streams = list(load_streams(self._config_path))
            self._log(f"[cyan]Reloaded {len(self._streams)} stream(s) from {self._config_path}.[/cyan]")
        else:
            self._log("[cyan]Refreshing stream status from the current target config.[/cyan]")
        self.run_worker(self._prune_recordings(), group="stream-prune", exclusive=True)
        self._refresh_all()

    @staticmethod
    def _record_cell(stream: StreamDef) -> str:
        """the Record column of a stream the console runs: whether it also records
        on the machine that captures it, and how long those recordings stay there."""
        if not stream.record:
            return "no"
        return f"yes, {stream.record_keep_days} d" if stream.record_keep_days > 0 else "yes"

    def _open_recordings_manager(self) -> None:
        """Manage: what the streams here recorded on the machines that capture them."""
        if not self.capture_streams():
            self._log(
                "[yellow]Every stream here is external: the console did not start them, so no host it "
                "manages holds a recording. The Stream Server's own are on its card, Recordings tab.[/yellow]"
            )
            return
        self.app.push_screen(
            StreamRecordingsScreen(self.capture_streams, self._live_record_paths, self._request_keep_days))

    def _request_keep_days(self, days: int) -> None:
        self.post_message(self.KeepDaysChangeRequested(
            {stream.name: days for stream in self._streams if stream.ssh_profile}))

    async def _prune_recordings(self, names: set[str] | None = None) -> None:
        """delete what a stream recorded longer ago than its record_keep_days,
        on its capture host; the file a stream is writing stays. Nothing on
        the host does it by itself: it happens at a stream's Start and at Refresh."""
        streams = [s for s in self.capture_streams() if s.keep_days > 0 and (names is None or s.name in names)]
        if not streams:
            return
        results = await asyncio.to_thread(capture_recordings.prune, streams, self._live_record_paths())
        for line in capture_recordings.prune_report(results, streams):
            self._log(line)

    def _start_stream(self, stream: StreamDef) -> None:
        self.run_worker(self._async_start(stream))

    def _stop_stream(self, stream: StreamDef) -> None:
        self.run_worker(self._async_stop(stream))

    def _view_stream_logs(self, stream: StreamDef) -> None:
        self.run_worker(self._async_view_logs(stream))

    def _probe_stream(self, stream: StreamDef) -> None:
        self.run_worker(self._async_probe_stream(stream))

    # seconds ffmpeg has to keep running after Start to count as started: a wrong
    # option, device or size stops it within the first second or two
    STREAM_SETTLE_SECONDS = 3.0
    # seconds Start waits for the Stream Server to receive a stream it started
    SERVER_LIVE_TIMEOUT = 12.0

    async def _await_ffmpeg(self, profile, session: str, desktop: bool) -> str | None:
        """what became of ffmpeg after Start: STREAM_RUNNING once it has run for
        STREAM_SETTLE_SECONDS, else the state it ended in (None: the host did
        not answer). A Terminal window on a Mac has DESKTOP_START_TIMEOUT to
        start it."""
        loop = asyncio.get_event_loop()
        wait = (DESKTOP_START_TIMEOUT + 5 if desktop else 10) + self.STREAM_SETTLE_SECONDS
        deadline = time.monotonic() + wait
        running_since = None
        state = None
        while time.monotonic() < deadline:
            state = await loop.run_in_executor(None, _stream_state, profile, session)
            if state == STREAM_RUNNING:
                running_since = running_since or time.monotonic()
                if time.monotonic() - running_since >= self.STREAM_SETTLE_SECONDS:
                    return state
            elif state in (STREAM_EXITED, STREAM_STOPPED):
                return state
            else:
                running_since = None
            await asyncio.sleep(0.5)
        return state

    async def _report_exit(self, stream: StreamDef, profile, session: str, where: str) -> None:
        """say that a stream's ffmpeg stopped by itself, and what it said last."""
        loop = asyncio.get_event_loop()
        pane = ""
        for attempt in range(4):
            pane = await loop.run_in_executor(None, _stream_pane, profile, session)
            if _EXIT_MARK in pane or attempt == 3:
                break
            # the pane follows an ffmpeg a Terminal window started: its last lines may still be on their way
            await asyncio.sleep(0.5)
        said = _last_words(pane)
        self._log(f"[red]{stream.name}: ffmpeg on {where} stopped by itself. What it said last:[/red]")
        for line in said or ["(nothing)"]:
            self._log(f"  {rich_escape(line)}")
        self._log("  [dim]Logs shows all of it; Start runs it again.[/dim]")

    async def _await_server(self, stream: StreamDef) -> str | None:
        """whether the Stream Server receives a stream just started: "live" once
        it does, "idle" if it still does not after SERVER_LIVE_TIMEOUT, "unknown"
        when it does not answer; None for a stream that goes elsewhere."""
        server = self._server_address()
        host = str(server.get("host") or "").strip()
        path = self._server_path(stream, server) if host else None
        if not path:
            return None
        try:
            api_port = int(server.get("api_port") or recordings.API_PORT)
        except ValueError:
            return None
        deadline = time.monotonic() + self.SERVER_LIVE_TIMEOUT
        while True:
            try:
                live = await asyncio.to_thread(recordings.live_paths, host, api_port)
            except recordings.RecordingsError:
                return "unknown"
            if path in live:
                return "live"
            if time.monotonic() >= deadline:
                return "idle"
            await asyncio.sleep(1.0)

    async def _check_server_side(self, stream: StreamDef, profile, session: str, desktop: bool, where: str) -> None:
        """after Start: does what ffmpeg publishes reach the Stream Server?"""
        seen = await self._await_server(stream)
        if seen is None:
            return
        self._live[stream.name] = seen
        if seen == "live":
            self._log(f"[green]{stream.name} is live on the Stream Server.[/green]")
        elif seen == "unknown":
            self._log(f"[yellow]The Stream Server does not answer, so whether {stream.name} reaches it is not known.[/yellow]")
        else:
            loop = asyncio.get_event_loop()
            state = await loop.run_in_executor(None, _stream_state, profile, session)
            if state == STREAM_EXITED:
                await self._report_exit(stream, profile, session, where)
                mark_stream_stopped(stream.name, project_dir=self._project_dir)
                self._set_state(stream.name, state)
            else:
                hint = (
                    " The first time, the Mac may be asking on its screen whether Terminal may use the camera "
                    "or the microphone." if desktop else ""
                )
                self._log(
                    f"[yellow]{stream.name}: ffmpeg runs on {where}, but nothing has reached the Stream Server "
                    f"yet. Logs shows what ffmpeg says.{hint}[/yellow]"
                )
        self._rebuild_table()

    async def _async_start(self, stream: StreamDef) -> None:
        session = _tmux_session_name(stream.name)
        is_local = stream.ssh_profile == "local"
        profile = None
        loop = asyncio.get_event_loop()

        if not is_local:
            profile = get_profile_by_name(stream.ssh_profile)
            if profile is None:
                self._log(f"[red]SSH profile '{stream.ssh_profile}' not found.[/red]")
                return
        where = "this machine" if is_local else stream.ssh_profile

        state = await loop.run_in_executor(None, _stream_state, profile, session)
        if state in (STREAM_RUNNING, STREAM_STARTING, None):
            if state is None:
                self._log(f"[red]{where} does not answer, so {stream.name} was not started.[/red]")
            else:
                self._log(f"[yellow]{stream.name} is already running.[/yellow]")
            self._set_state(stream.name, state)
            self._rebuild_table()
            return

        if stream.record_keep_days > 0:
            # room first: what it recorded before and kept long enough goes
            await self._prune_recordings({stream.name})
        platform = await loop.run_in_executor(None, _stream_platform, stream, profile)
        desktop = _needs_desktop_session(platform, is_local)
        record_dir = self._record_dir(stream)
        try:
            ffmpeg_cmd = _build_ffmpeg_cmd(stream, record_dir, platform, self._kind(stream))
        except ValueError as e:
            self._log(f"[red]Cannot start {stream.name}: {e}[/red]")
            return
        record_path = _record_path(stream, record_dir, self._kind(stream)) if record_dir else None
        if desktop:
            launch_cmd = _build_desktop_stream_cmd(session, ffmpeg_cmd, record_dir, record_path, stream.name)
        else:
            launch_cmd = _build_tmux_stream_cmd(session, ffmpeg_cmd, record_dir, record_path)
        if state == STREAM_EXITED:
            # the tmux session outlived its ffmpeg: a new one takes its name
            self._log(f"[yellow]{stream.name}: its ffmpeg on {where} had stopped by itself; starting it again.[/yellow]")
            launch_cmd = _with_stream_path(f"tmux kill-session -t {_tmux_session_target(session)} 2>/dev/null; {launch_cmd}")
        target_label = "locally" if is_local else f"on {stream.ssh_profile}"
        self._log(f"[green]Starting {stream.name} {target_label}...[/green]")
        self._log(f"  {rich_escape(ffmpeg_cmd)}")
        if desktop:
            self._log(
                f"  [yellow]{where} is a Mac: ffmpeg is started from a Terminal window on its own screen, as "
                "macOS lets nothing started over SSH use the camera or the microphone; the window closes by "
                "itself once ffmpeg runs. Someone has to be logged in there, with Terminal allowed under "
                "Privacy & Security (Camera, Microphone).[/yellow]"
            )
        if record_dir:
            self._log(f"  Recording to {record_dir}/ on the streaming host")

        try:
            result = await loop.run_in_executor(None, _run_on_host, profile, launch_cmd, 15.0)
            if result.returncode == 0:
                self._set_state(stream.name, STREAM_STARTING)
                self._rebuild_table()
                # tmux runs, which says nothing of ffmpeg: it may stop on its first line
                state = await self._await_ffmpeg(profile, session, desktop)
                if state != STREAM_RUNNING:
                    if state is None:
                        self._log(
                            f"[yellow]{where} does not answer, so whether {stream.name} runs is not known; "
                            f"Refresh asks again.[/yellow]"
                        )
                    else:
                        await self._report_exit(stream, profile, session, where)
                    self._set_state(stream.name, state)
                    self._live = await self._server_states()
                    self._rebuild_table()
                    return
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
                self._set_state(stream.name, STREAM_RUNNING)
                self._rebuild_table()
                await self._check_server_side(stream, profile, session, desktop, where)
            else:
                output = result.stdout.strip() if result.stdout else result.stderr.strip()
                self._log(f"[red]Failed to start {stream.name}: {rich_escape(output)}[/red]")
                self._set_state(stream.name, STREAM_STOPPED)
        except Exception as e:
            self._log(f"[red]Error starting {stream.name}: {rich_escape(str(e))}[/red]")
            self._set_state(stream.name, None)
        self._rebuild_table()

    async def _async_stop(self, stream: StreamDef) -> None:
        session = _tmux_session_name(stream.name)
        is_local = stream.ssh_profile == "local"

        if not is_local:
            profile = get_profile_by_name(stream.ssh_profile)
            if profile is None:
                self._log(f"[red]SSH profile '{stream.ssh_profile}' not found.[/red]")
                return

        stop_cmd = _with_stream_path(_build_stop_stream_cmd(session))
        target_label = "locally" if is_local else f"on {stream.ssh_profile}"
        self._log(f"[red]Stopping {stream.name} {target_label}...[/red]")

        loop = asyncio.get_event_loop()
        try:
            # the grace for ffmpeg to finish its files, and the ssh round trip
            result = await loop.run_in_executor(
                None, _run_on_host, None if is_local else profile, stop_cmd, STREAM_STOP_GRACE_SECONDS + 7.0,
            )
            if "DONE" in (result.stdout or ""):
                self._log(f"[red]{stream.name} stopped.[/red]")
                recorded = self._registered_record_path(stream.name)
                mark_stream_stopped(stream.name, project_dir=self._project_dir)
                self._set_state(stream.name, STREAM_STOPPED)
                if self._is_left_over(stream):
                    # it was on the tab to be stopped: its row goes with it
                    self._left_over = [s for s in self._left_over if s.name != stream.name]
                self._live = await self._server_states()
                record_dir = self._record_dir(stream)
                if recorded:
                    # where Start put it: the card's Session may have changed since,
                    # and the file did not move with it
                    self._log(f"  Recording kept at {recorded}{self._on_host(stream)}")
                elif record_dir:
                    self._log(f"  Recording kept under {record_dir}/{self._on_host(stream)}")
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
                        ["tmux", "capture-pane", "-t", f"={session}:", "-p", "-S", "-120"],
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
                cmd = _with_stream_path(f"tmux capture-pane -t {_tmux_pane_target(session)} -p -S -120")
                result = await loop.run_in_executor(None, ssh_run_sync, profile, cmd, 10.0)
            output = (result.stdout or result.stderr or "").strip()
            if result.returncode != 0:
                self._log(f"[red]Could not read tmux logs: {output or result.returncode}[/red]")
                return
            if not output:
                self._log("[yellow]No tmux output captured yet.[/yellow]")
                return
            # ffmpeg's lines start with "[in#0 @ 0x...]", which the log would take for markup
            for line in output.splitlines()[-80:]:
                self._log(rich_escape(line))
        except Exception as e:
            self._log(f"[red]Error reading logs for {stream.name}: {e}[/red]")

    async def _async_probe_stream(self, stream: StreamDef) -> None:
        url = stream.read_url
        self._log(f"[cyan]Probing {url}...[/cyan]")
        loop = asyncio.get_event_loop()
        success, message = await loop.run_in_executor(None, _probe_stream_target, url)
        if success:
            self._log(f"[green]{stream.name}: {message}[/green]")
            return
        self._log(f"[red]{stream.name}: {rich_escape(message)}[/red]")
        if "404" in message:
            follow = (
                "Status says whether its ffmpeg runs, and Logs why it stopped." if stream.ssh_profile
                else "It is external: whoever publishes it has to start it."
            )
            self._log(
                f"  [yellow]The Stream Server has nothing at that path: nobody is publishing {stream.name} "
                f"now. {follow}[/yellow]"
            )

    def update_streams(self, streams: list[StreamDef]) -> None:
        """replace the stream list and refresh. Shown at once, a stream whose
        machine is the same with the status it had, then checked again."""
        hosts = {stream.name: stream.ssh_profile for stream in self._rows()}
        self._streams = list(streams)
        # an entry taken out of the config is a left-over capture until it is
        # stopped, and keeps the status it had until the refresh says otherwise
        self._left_over = [stream for stream in self._left_over
                           if not any(other.name == stream.name for other in self._streams)]
        self._statuses = {
            stream.name: self._statuses[stream.name] for stream in self._rows()
            if stream.name in self._statuses and hosts.get(stream.name) == stream.ssh_profile
        }
        self._states = {name: self._states[name] for name in self._statuses if name in self._states}
        self._live = {stream.name: self._live[stream.name] for stream in self._rows() if stream.name in self._live}
        self._rebuild_table()
        self._refresh_all()
