"""the capture devices of a host, for the dropdowns of the Config tab: what a
Streams entry's `device` can be on its capture host, and what a Bases entry's
`source_index` can be for a pyaudio or an opencv source on the card's host.

Each host is asked with the tools it has, over SSH for another machine (the
console knows what each host runs, see ssh.remote_platform):
  a Mac      ffmpeg -f avfoundation -list_devices: its cameras (0, 1, ...) and
             microphones (:0, :1, ...), the values its streams take
  Linux      arecord -l for the microphones (hw:<card>,<device>), v4l2-ctl
             --list-devices for the cameras (/dev/videoN, the first node of
             each), or the /dev/video* nodes alone without v4l2-ctl
  pyaudio    the input devices PortAudio reports, asked in the asr-base env of
             the host, since its indexes are PortAudio's own
An opencv index is the camera list again: OpenCV opens /dev/video<N> for index
N on Linux, and AVFoundation's Nth device on a Mac.

What a host says is a snapshot: a device plugged in later shows up at the next
Refresh of the card. The parsers are pure and tested on real output."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

from openmmla.tui.ssh import TARGET_PLATFORMS, get_profile_by_name, remote_platform, ssh_run_sync, wrap_local, \
    wrap_remote
from openmmla.tui.stream_cuts import bash, with_tool_path

# the kinds of device a host is asked for
STREAM_KINDS = ("audio", "video")     # a Streams entry's device, by the stream's kind
SOURCE_KINDS = ("pyaudio", "opencv")  # a Bases entry's source_index, by its source
CONDA_ENV = "asr-base"                # where pyaudio is

TIMEOUT = 20.0

# the lines the scripts mark their parts with
_AUDIO_MARK, _VIDEO_MARK, _NODES_MARK, _PYAUDIO_MARK = "@audio", "@video", "@nodes", "@pyaudio"

MAC_SCRIPT = 'ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1; true'
LINUX_SCRIPT = (
    f'echo {_AUDIO_MARK}; arecord -l 2>&1; echo {_VIDEO_MARK}; v4l2-ctl --list-devices 2>&1; '
    f'echo {_NODES_MARK}; ls -1 /dev/video* 2>/dev/null; true'
)
# run by the python of the asr-base env; prints one @pyaudio line
PYAUDIO_SCRIPT = f'''
import json
try:
    import pyaudio
except ImportError:
    print("{_PYAUDIO_MARK} " + json.dumps({{"error": "pyaudio is not installed in this env"}}))
    raise SystemExit(0)
p = pyaudio.PyAudio()
found = []
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    if int(d.get("maxInputChannels") or 0) > 0:
        found.append({{"index": i, "name": str(d.get("name") or ""), "channels": int(d["maxInputChannels"])}})
p.terminate()
print("{_PYAUDIO_MARK} " + json.dumps({{"devices": found}}))
'''


@dataclass(frozen=True)
class Device:
    value: str   # what the config stores
    label: str   # what the dropdown shows beside it


@dataclass
class Devices:
    """what one host has, as it answered: the devices of each kind asked for,
    and why a kind is not known."""
    found: dict[str, list[Device]] = field(default_factory=dict)
    problems: dict[str, str] = field(default_factory=dict)
    platform: str = ""

    def options(self, kind: str) -> list[tuple[str, str]]:
        """(label, value) pairs for a dropdown; [] when the kind is not known."""
        return [(f"{d.value}  {d.label}".rstrip() if d.label else d.value, d.value) for d in self.found.get(kind, [])]

    def note(self, kind: str, where: str) -> str:
        """one line under a field: what was found on `where`, or why nothing is known."""
        if kind in self.problems:
            return f"{where}: {self.problems[kind]}"
        devices = self.found.get(kind)
        if devices is None:
            return ""
        if not devices:
            return f"No {kind} device found on {where}; Refresh on the Launch tab looks again."
        shown = ", ".join(f"{d.value} {d.label}".strip() for d in devices[:4])
        more = f" +{len(devices) - 4} more" if len(devices) > 4 else ""
        return f"Found on {where}: {shown}{more}. Refresh on the Launch tab looks again."


# ---- parsing what the tools print ----

_AVF_LINE = re.compile(r"\[AVFoundation indev @ [^\]]+\] (.*)")
_AVF_DEVICE = re.compile(r"^\[(\d+)\] (.+)$")


def parse_avfoundation(text: str) -> tuple[list[Device], list[Device]]:
    """(cameras, microphones) of `ffmpeg -f avfoundation -list_devices true`:
    a camera is its index (what a video stream's device is), a microphone
    ':<index>' (an audio stream's)."""
    cameras: list[Device] = []
    microphones: list[Device] = []
    current: list[Device] | None = None
    prefix = ""
    for raw in (text or "").splitlines():
        match = _AVF_LINE.search(raw)
        if not match:
            continue
        line = match.group(1).strip()
        if line.endswith("video devices:"):
            current, prefix = cameras, ""
        elif line.endswith("audio devices:"):
            current, prefix = microphones, ":"
        elif current is not None:
            device = _AVF_DEVICE.match(line)
            if device:
                current.append(Device(prefix + device.group(1), device.group(2).strip()))
    return cameras, microphones


_ARECORD = re.compile(r"^card (\d+): (.+?) \[(.*?)\], device (\d+): (.+?) \[(.*?)\]", re.M)


def parse_arecord(text: str) -> list[Device]:
    """the capture devices `arecord -l` lists, as hw:<card>,<device>."""
    devices = []
    for card, _card_id, card_name, index, _device_id, device_name in _ARECORD.findall(text or ""):
        label = card_name.strip()
        if device_name.strip() and device_name.strip() != card_name.strip():
            label = f"{label} — {device_name.strip()}"
        devices.append(Device(f"hw:{card},{index}", label))
    return devices


def parse_v4l2(text: str, nodes: str = "") -> list[Device]:
    """the cameras `v4l2-ctl --list-devices` groups (the first /dev/videoN of
    each, named), else the /dev/video* nodes of `nodes`, unnamed."""
    devices: list[Device] = []
    name = ""
    taken = False
    for raw in (text or "").splitlines():
        if not raw.strip():
            name, taken = "", False
            continue
        if not raw.startswith(("\t", " ")):
            name, taken = raw.strip().rstrip(":"), False
            continue
        node = raw.strip()
        if node.startswith("/dev/video") and not taken:
            devices.append(Device(node, name))
            taken = True
    if devices:
        return devices
    for raw in (nodes or "").splitlines():
        node = raw.strip()
        if node.startswith("/dev/video"):
            devices.append(Device(node, ""))
    return sorted(devices, key=lambda d: _node_number(d.value))


def _node_number(node: str) -> int:
    digits = re.sub(r"\D", "", node.rsplit("/", 1)[-1])
    return int(digits) if digits else 0


def opencv_from_cameras(cameras: list[Device], platform: str) -> list[Device]:
    """the opencv indexes of the cameras found: /dev/video<N> is index N on
    Linux, and a Mac's cameras are indexed as AVFoundation lists them."""
    devices = []
    for camera in cameras:
        if platform == "linux":
            index = str(_node_number(camera.value)) if camera.value.startswith("/dev/video") else camera.value
            label = f"{camera.label} ({camera.value})" if camera.label else camera.value
        else:
            index, label = camera.value, camera.label
        devices.append(Device(index, label))
    return devices


def parse_pyaudio(text: str) -> tuple[list[Device], str]:
    """(input devices, problem) from the @pyaudio line PYAUDIO_SCRIPT prints."""
    for raw in reversed((text or "").splitlines()):
        line = raw.strip()
        if line.startswith(_PYAUDIO_MARK):
            try:
                answer = json.loads(line[len(_PYAUDIO_MARK):])
            except ValueError:
                break
            if answer.get("error"):
                return [], str(answer["error"])
            devices = [
                Device(str(d.get("index")), f"{d.get('name', '')} ({d.get('channels', '?')} ch)".strip())
                for d in answer.get("devices", []) if d.get("index") is not None
            ]
            return devices, ""
    tail = [line.strip() for line in (text or "").splitlines() if line.strip()][-2:]
    if any("EnvironmentNameNotFound" in line or "Could not find conda environment" in line for line in tail):
        return [], f"no conda env '{CONDA_ENV}' to ask pyaudio in"
    return [], "pyaudio did not answer" + (f" ({' / '.join(tail)})" if tail else "")


def _split_linux(text: str) -> tuple[str, str, str]:
    """the arecord, v4l2-ctl and ls parts of LINUX_SCRIPT's output."""
    audio = video = nodes = ""
    part = None
    for raw in (text or "").splitlines():
        if raw.strip() == _AUDIO_MARK:
            part = "audio"
        elif raw.strip() == _VIDEO_MARK:
            part = "video"
        elif raw.strip() == _NODES_MARK:
            part = "nodes"
        elif part == "audio":
            audio += raw + "\n"
        elif part == "video":
            video += raw + "\n"
        elif part == "nodes":
            nodes += raw + "\n"
    return audio, video, nodes


# ---- asking a host ----

def platform_of(target: str) -> str:
    """"darwin", "linux", "windows" or "" (not known) for this machine or an SSH profile."""
    if target == "local":
        return "darwin" if sys.platform == "darwin" else ("windows" if sys.platform.startswith("win") else "linux")
    cached = TARGET_PLATFORMS.get(target)
    if cached:
        return cached
    profile = get_profile_by_name(target)
    return remote_platform(profile) if profile is not None else ""


def _run(target: str, command: str, timeout: float = TIMEOUT) -> str | None:
    """stdout (with stderr) of a shell command on `target`; None when it could not run."""
    try:
        if target == "local":
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        else:
            profile = get_profile_by_name(target)
            if profile is None:
                return None
            result = ssh_run_sync(profile, command, timeout=timeout)
    except Exception:
        # the ssh argument list holds the password: never in a message
        return None
    return (result.stdout or "") + (result.stderr or "")


def pyaudio_command(python_path: str = "") -> str:
    """the shell line that asks pyaudio in the asr-base env: the script rides
    in base64, so no quoting of it survives to the shells it passes through."""
    encoded = base64.b64encode(PYAUDIO_SCRIPT.encode()).decode()
    path = f"export PYTHONPATH={python_path}:$PYTHONPATH && " if python_path else ""
    return f'{path}python -c "import base64;exec(base64.b64decode(\'{encoded}\').decode())"'


def list_devices(target: str, kinds, root: str = "") -> Devices:
    """ask `target` ("local" or an SSH profile) for the devices of `kinds`
    (audio, video, pyaudio, opencv). Blocking: run it off the UI thread."""
    kinds = set(kinds)
    where = "this machine" if target == "local" else target
    answer = Devices(platform=platform_of(target))
    if not answer.platform:
        for kind in kinds:
            answer.problems[kind] = f"{where} did not say what it runs, so its devices were not asked for"
        return answer
    if answer.platform == "windows":
        for kind in kinds:
            answer.problems[kind] = "devices on a Windows host are not listed here: type the device"
        return answer

    if kinds & {"audio", "video", "opencv"}:
        script = MAC_SCRIPT if answer.platform == "darwin" else LINUX_SCRIPT
        text = _run(target, bash(with_tool_path(script)))
        if text is None:
            for kind in kinds & {"audio", "video", "opencv"}:
                answer.problems[kind] = f"{where} did not answer, so its devices are not known"
        else:
            if answer.platform == "darwin":
                cameras, microphones = parse_avfoundation(text)
                if not cameras and not microphones and "avfoundation" not in text.lower():
                    for kind in kinds & {"audio", "video", "opencv"}:
                        answer.problems[kind] = f"ffmpeg on {where} did not list its devices (is it installed?)"
                    return _ask_pyaudio(answer, target, kinds, root, where)
            else:
                audio, video, nodes = _split_linux(text)
                microphones = parse_arecord(audio)
                cameras = parse_v4l2(video, nodes)
            if "audio" in kinds:
                answer.found["audio"] = microphones
            if "video" in kinds:
                answer.found["video"] = cameras
            if "opencv" in kinds:
                answer.found["opencv"] = opencv_from_cameras(cameras, answer.platform)
    return _ask_pyaudio(answer, target, kinds, root, where)


def _ask_pyaudio(answer: Devices, target: str, kinds: set[str], root: str, where: str) -> Devices:
    if "pyaudio" not in kinds:
        return answer
    command = pyaudio_command(root if target == "local" else "")
    wrapped = wrap_local(command, CONDA_ENV) if target == "local" else wrap_remote(command, CONDA_ENV)
    text = _run(target, wrapped)
    if text is None:
        answer.problems["pyaudio"] = f"{where} did not answer, so its input devices are not known"
        return answer
    devices, problem = parse_pyaudio(text)
    if problem:
        answer.problems["pyaudio"] = problem
    else:
        answer.found["pyaudio"] = devices
    return answer


def stream_hosts(config: dict) -> dict[str, str]:
    """the capture host of each Streams entry the console runs ("local" or an
    SSH profile): what its device dropdown has to ask. External streams have none."""
    hosts: dict[str, str] = {}
    for name, entry in ((config or {}).get("Streams") or {}).items():
        if not isinstance(entry, dict):
            continue
        host = str(entry.get("ssh_profile") or "").strip()
        if host:
            hosts[str(name)] = host
    return hosts


def device_path(config_dir: str) -> str:
    """this checkout's root, the PYTHONPATH a local pyaudio probe runs with."""
    return os.path.dirname(os.path.dirname(os.path.abspath(config_dir)))
