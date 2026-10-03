"""the programs a stream needs on the machine that captures it, and what each
conda env of the Environment tab needs of its host, and installing the ones a
host lacks.

A stream's ffmpeg runs in a tmux session on the host its camera or
microphone is attached to (widgets.stream_panel), and the Device column asks
that host's v4l2-ctl (cameras) and arecord (microphones) what it has
(devices). A freshly installed Raspberry Pi has no tmux, and may lack the
rest: Start asks the host first and installs what is missing, with apt-get
on Linux (sudo answered with the password it is handed: the SSH profile's,
or System Settings -> Sudo for this machine) and with Homebrew on a Mac.
Without tmux or ffmpeg nothing can start; the other two only fill the
Device lists, so a host still lacking them starts its stream all the same.

The Environment tab asks a host for what its envs' programs need (ENV_TOOLS:
an ASR base's ffmpeg and PortAudio, the tmux of the dashboard and the MLLM
Server, ...) and installs it the same way, on a button.

Everything here builds commands or reads what they printed, and is tested
without a host; running them is the caller's, one install at a time on a host
(install_lock)."""

from __future__ import annotations

import asyncio
import shlex
import weakref
from dataclasses import dataclass

# program -> (Debian package, Homebrew formula); "" where there is none. A
# Mac's C compiler comes with Xcode's command line tools, not from Homebrew
PACKAGES = {
    "tmux": ("tmux", "tmux"),
    "ffmpeg": ("ffmpeg", "ffmpeg"),
    "v4l2-ctl": ("v4l-utils", ""),
    "arecord": ("alsa-utils", ""),
    "portaudio": ("portaudio19-dev", "portaudio"),
    "cc": ("build-essential", ""),
    "git": ("git", "git"),
    "sshpass": ("sshpass", "sshpass"),
}
# what is asked for by a test of its own rather than by `command -v`: PortAudio
# is a library, and pip builds PyAudio against its header
_PROBES = {
    "portaudio": " || ".join(
        f"[ -f {folder}/include/portaudio.h ]"
        for folder in ("/usr", "/usr/local", "/opt/homebrew", "/opt/local")
    ),
}
# conda env group -> platform -> what its programs need on the host: an ASR
# base pulls its stream through ffmpeg and imports PyAudio, which pip builds
# against PortAudio with a C compiler (a Mac has one with Xcode's tools); the
# dashboard, Celery and the MLLM Server run in tmux; the analysis commands cut
# media with ffmpeg; the console reaches hosts with sshpass and clones with
# git. The VFA and IPS bases read video through OpenCV, which brings its own
# FFmpeg, and need nothing
ENV_TOOLS = {
    "asr-base": {"darwin": ["ffmpeg", "portaudio"], "linux": ["ffmpeg", "portaudio", "cc"]},
    "vfa-vllm-runtime": {"darwin": ["tmux"], "linux": ["tmux"]},
    "uber-base": {"darwin": ["ffmpeg"], "linux": ["ffmpeg"]},
    "uber-server": {"darwin": ["tmux"], "linux": ["tmux"]},
    "tui": {"darwin": ["git", "sshpass"], "linux": ["git", "sshpass"]},
}
# what no stream starts without
REQUIRED = ("tmux", "ffmpeg")
# seconds the check and an install may take: ffmpeg brings about a hundred
# packages along, over whatever network the host is on
CHECK_TIMEOUT = 15.0
INSTALL_TIMEOUT = 900.0
# seconds apt waits for another install (a desktop's updater) to let go of its lock
LOCK_WAIT = 120

# the last line check_command prints: a host that did not run it through prints none
_CHECKED = "OPENMMLA_TOOLS_CHECKED"


def env_tools(group: str, platform: str) -> list[str]:
    """what the env of `group` needs on a host of `platform` ("darwin" or "linux")."""
    return list(ENV_TOOLS.get(group, {}).get(platform, []))


def all_env_tools() -> list[str]:
    """everything any env needs on any platform, for one check of a host."""
    return list(dict.fromkeys(tool for platforms in ENV_TOOLS.values()
                              for tools in platforms.values() for tool in tools))


def stream_programs(platform: str, kind: str) -> list[str]:
    """the programs a stream of `kind` ("audio" or "video") needs on a host of
    `platform` ("darwin" or "linux"). A Mac lists its devices with ffmpeg itself."""
    if platform == "darwin":
        return list(REQUIRED)
    return [*REQUIRED, "arecord" if kind == "audio" else "v4l2-ctl"]


@dataclass
class HostTools:
    """what a host said: the programs it lacks, what it installs with
    ("apt", "brew", or "" for neither) and what it is ("darwin", "linux", or
    "" when it did not say)."""
    missing: list[str]
    installer: str
    platform: str = ""

    @property
    def lacking_required(self) -> list[str]:
        return [program for program in self.missing if program in REQUIRED]


def check_command(programs: list[str]) -> str:
    """a shell command that prints MISSING <program> for each program the host
    lacks, INSTALLER apt|brew for what it installs with, PLATFORM and an end mark."""
    probes = "".join(
        f"{{ {_PROBES[program]}; }} || echo MISSING {shlex.quote(program)}; " if program in _PROBES
        else f"command -v {shlex.quote(program)} >/dev/null 2>&1 || echo MISSING {shlex.quote(program)}; "
        for program in programs
    )
    return (
        f"{probes}"
        'echo PLATFORM "$(uname -s)"; '
        "if command -v apt-get >/dev/null 2>&1; then echo INSTALLER apt; "
        "elif command -v brew >/dev/null 2>&1; then echo INSTALLER brew; fi; "
        f"echo {_CHECKED}"
    )


def parse_check(output: str) -> HostTools | None:
    """what check_command printed; None when it did not run through."""
    lines = [line.split() for line in (output or "").splitlines()]
    if [_CHECKED] not in lines:
        return None
    missing = [words[1] for words in lines if len(words) == 2 and words[0] == "MISSING"]
    installer = next((words[1] for words in lines if len(words) == 2 and words[0] == "INSTALLER"), "")
    system = next((words[1] for words in lines if len(words) == 2 and words[0] == "PLATFORM"), "")
    return HostTools(missing, installer, {"Darwin": "darwin", "Linux": "linux"}.get(system, ""))


def packages(tools: HostTools) -> list[str]:
    """the packages that give the host what it lacks, as its installer names them."""
    column = {"apt": 0, "brew": 1}.get(tools.installer)
    if column is None:
        return []
    names = [PACKAGES.get(program, ("", ""))[column] for program in tools.missing]
    return list(dict.fromkeys(name for name in names if name))


def install_command(installer: str, names: list[str]) -> str:
    """the command that installs `names` with `installer`. apt-get runs under
    sudo -S, which reads the password from stdin if it asks for one; the
    install itself reads nothing from there (exec </dev/null), so a password
    sudo did not need never reaches apt-get or a package's scripts. Nothing
    asks a question on the way either: debconf is told not to, and a config
    file someone changed on the host stays as it is."""
    quoted = " ".join(shlex.quote(name) for name in names)
    if installer == "brew":
        return f"brew install {quoted}"
    lock = f"-o DPkg::Lock::Timeout={LOCK_WAIT}"
    script = (
        "exec </dev/null; export DEBIAN_FRONTEND=noninteractive; "
        f"apt-get {lock} update -qq; "
        f"apt-get {lock} install -y -qq "
        f"-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold {quoted}"
    )
    return f"sudo -S -p '' sh -c {shlex.quote(script)}"


# one install at a time on a host, whichever tab asked: apt takes one at a
# time, and the second then finds the programs in place. Per event loop, as an
# asyncio lock belongs to the one it was first used in
_INSTALL_LOCKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def install_lock(host: str) -> asyncio.Lock:
    """the lock an install on `host` ("local" or an SSH profile's name) holds."""
    locks = _INSTALL_LOCKS.setdefault(asyncio.get_running_loop(), {})
    return locks.setdefault(host, asyncio.Lock())


def manual_command(installer: str, names: list[str]) -> str:
    """what someone at the host types to install `names` by hand."""
    if installer == "brew":
        return f"brew install {' '.join(names)}"
    return f"sudo apt install -y {' '.join(names)}"


def install_failure(output: str, password_source: str) -> str:
    """why an install failed, from what it printed; `password_source` says
    where the password sudo was handed came from."""
    lowered = (output or "").lower()
    if "incorrect password" in lowered or "sorry, try again" in lowered:
        return f"sudo refused the password of {password_source}"
    if "no password was provided" in lowered or "a password is required" in lowered:
        return f"sudo asks for a password, and {password_source} has none"
    if "could not get lock" in lowered:
        return f"another install kept apt busy for longer than {LOCK_WAIT} s"
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    return " / ".join(lines[-3:]) or "it printed nothing"
