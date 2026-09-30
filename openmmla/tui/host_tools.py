"""the programs a stream needs on the machine that captures it, and installing
the ones that machine lacks.

A stream's ffmpeg runs in a tmux session on the host its camera or
microphone is attached to (widgets.stream_panel), and the Device column asks
that host's v4l2-ctl (cameras) and arecord (microphones) what it has
(devices). A freshly installed Raspberry Pi has no tmux, and may lack the
rest: Start asks the host first and installs what is missing, with apt-get
on Linux (sudo answered with the password it is handed: the SSH profile's,
or System Settings -> Sudo for this machine) and with Homebrew on a Mac.
Without tmux or ffmpeg nothing can start; the other two only fill the
Device lists, so a host still lacking them starts its stream all the same.

Everything here builds commands or reads what they printed, and is tested
without a host; running them is the caller's."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

# program -> (Debian package, Homebrew formula); "" where there is none
PACKAGES = {
    "tmux": ("tmux", "tmux"),
    "ffmpeg": ("ffmpeg", "ffmpeg"),
    "v4l2-ctl": ("v4l-utils", ""),
    "arecord": ("alsa-utils", ""),
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


def stream_programs(platform: str, kind: str) -> list[str]:
    """the programs a stream of `kind` ("audio" or "video") needs on a host of
    `platform` ("darwin" or "linux"). A Mac lists its devices with ffmpeg itself."""
    if platform == "darwin":
        return list(REQUIRED)
    return [*REQUIRED, "arecord" if kind == "audio" else "v4l2-ctl"]


@dataclass
class HostTools:
    """what a host said: the programs it lacks, and what it installs with
    ("apt", "brew", or "" for neither)."""
    missing: list[str]
    installer: str

    @property
    def lacking_required(self) -> list[str]:
        return [program for program in self.missing if program in REQUIRED]


def check_command(programs: list[str]) -> str:
    """a shell command that prints MISSING <program> for each program the host
    lacks, INSTALLER apt|brew for what it installs with, and an end mark."""
    probes = "".join(
        f"command -v {shlex.quote(program)} >/dev/null 2>&1 || echo MISSING {shlex.quote(program)}; "
        for program in programs
    )
    return (
        f"{probes}"
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
    return HostTools(missing, installer)


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
