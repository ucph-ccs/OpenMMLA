from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict

import yaml

from openmmla.tui.schema.loader import _find_project_root

PROFILES_DIR = "config"
PROFILES_FILE = "ssh_profiles.yml"
_LOCAL_COMMAND_DIRS = (
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
)


def _resolve_local_command(command: str) -> str:
    path = shutil.which(command)
    if path:
        return path
    for directory in _LOCAL_COMMAND_DIRS:
        candidate = os.path.join(directory, command)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return command


def _missing_local_command(args: list[str]) -> str | None:
    for command in ("sshpass", "ssh", "scp"):
        if any(os.path.basename(arg) == command for arg in args) and _resolve_local_command(command) == command:
            return command
    return None


@dataclass
class SSHProfile:
    name: str
    host: str
    user: str
    port: int = 22
    password: str = ""
    key_path: str = ""
    remote_project_path: str = "~/OpenMMLA"

    def ssh_destination(self) -> str:
        return f"{self.user}@{self.host}"

    def _sshpass_prefix(self) -> list[str]:
        """return sshpass prefix if password auth is configured."""
        if self.password:
            return [_resolve_local_command("sshpass"), "-p", self.password]
        return []

    def _control_path(self) -> str:
        return f"/tmp/ssh-openmmla-{self.user}@{self.host}:{self.port}"

    def _common_ssh_opts(self) -> list[str]:
        return [
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self._control_path()}",
            "-o", "ControlPersist=300",
        ]

    def base_ssh_args(self) -> list[str]:
        """return the common ssh argument list (without the remote command)."""
        args = self._sshpass_prefix()
        args.extend([_resolve_local_command("ssh")] + self._common_ssh_opts())
        if self.key_path:
            args.extend(["-i", os.path.expanduser(self.key_path)])
        if self.port != 22:
            args.extend(["-p", str(self.port)])
        args.append(self.ssh_destination())
        return args

    def base_scp_args(self) -> list[str]:
        """return the common scp argument list (without src/dest)."""
        args = self._sshpass_prefix()
        args.extend([_resolve_local_command("scp")] + self._common_ssh_opts())
        if self.key_path:
            args.extend(["-i", os.path.expanduser(self.key_path)])
        if self.port != 22:
            args.extend(["-P", str(self.port)])
        return args


def _profiles_path() -> str:
    root = _find_project_root()
    return os.path.join(root, PROFILES_DIR, PROFILES_FILE)


def load_ssh_profiles() -> list[SSHProfile]:
    """load ssh profiles from .openmmla/ssh_profiles.yml."""
    path = _profiles_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            return []
        profiles = []
        for entry in data:
            if isinstance(entry, dict) and "name" in entry and "host" in entry:
                profiles.append(SSHProfile(
                    name=entry["name"],
                    host=entry["host"],
                    user=entry.get("user", ""),
                    port=int(entry.get("port", 22)),
                    password=entry.get("password", ""),
                    key_path=entry.get("key_path", ""),
                    remote_project_path=entry.get("remote_project_path", "~/OpenMMLA"),
                ))
        return profiles
    except Exception:
        return []


def save_ssh_profiles(profiles: list[SSHProfile]) -> None:
    """save ssh profiles to .openmmla/ssh_profiles.yml."""
    path = _profiles_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [asdict(p) for p in profiles]
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_profile_by_name(name: str) -> SSHProfile | None:
    for p in load_ssh_profiles():
        if p.name == name:
            return p
    return None


def ssh_run_sync(profile: SSHProfile, command: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    """run a command on the remote host synchronously."""
    args = profile.base_ssh_args() + [command]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


async def ssh_run_async(
    profile: SSHProfile,
    command: str,
    pipe_stdin: bool = False,
) -> asyncio.subprocess.Process:
    """start a command on the remote host asynchronously, returning the process."""
    args = profile.base_ssh_args() + [command]
    return await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if pipe_stdin else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


def ssh_check_port(profile: SSHProfile, port: int) -> bool:
    """check if a port on the remote host is accepting connections."""
    cmd = f"nc -z 127.0.0.1 {port} 2>/dev/null && echo OK || echo FAIL"
    try:
        result = ssh_run_sync(profile, cmd, timeout=8.0)
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return False


def ssh_check_tmux(profile: SSHProfile, session_name: str) -> bool:
    """check if a tmux session exists on the remote host."""
    cmd = f"tmux has-session -t {session_name} 2>/dev/null && echo OK || echo FAIL"
    try:
        result = ssh_run_sync(profile, cmd, timeout=8.0)
        return "OK" in result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return False


def ssh_test_connection(profile: SSHProfile) -> tuple[bool, str]:
    """test ssh connectivity. returns (success, message)."""
    args = profile.base_ssh_args()
    missing_command = _missing_local_command(args)
    if missing_command:
        return False, f"{missing_command} command not found"
    try:
        result = subprocess.run(
            args + ["echo CONNECTION_OK"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if result.returncode == 0 and "CONNECTION_OK" in result.stdout:
            return True, "Connection successful"
        return False, result.stderr.strip() or f"exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "Connection timed out"
    except FileNotFoundError:
        return False, "ssh command not found"
    except Exception as e:
        return False, str(e)


async def scp_file_async(
    profile: SSHProfile,
    local_path: str,
    remote_path: str,
) -> asyncio.subprocess.Process:
    """copy a local file to the remote host via scp, returning the process."""
    args = profile.base_scp_args()
    args.extend([local_path, f"{profile.ssh_destination()}:{remote_path}"])
    return await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


async def scp_from_remote_async(
    profile: SSHProfile,
    remote_path: str,
    local_path: str,
) -> asyncio.subprocess.Process:
    """copy a remote file or directory to the local host via scp."""
    args = profile.base_scp_args()
    args.extend(["-r", f"{profile.ssh_destination()}:{remote_path}", local_path])
    return await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


CONDA_INIT = (
    'if command -v conda >/dev/null 2>&1; then '
    '_base="$(conda info --base 2>/dev/null)" && '
    '[ -f "$_base/etc/profile.d/conda.sh" ] && . "$_base/etc/profile.d/conda.sh"; '
    'fi; '
    'if ! command -v conda >/dev/null 2>&1; then '
    'for _p in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" '
    '"$HOME/mambaforge" /home/*/miniforge3 /home/*/miniconda3 /opt/conda; do '
    '[ -f "$_p/etc/profile.d/conda.sh" ] && . "$_p/etc/profile.d/conda.sh" && break; '
    'done; '
    'fi'
)

REMOTE_SHELL_INIT = f'export LANG=en_US.UTF-8 PYTHONUNBUFFERED=1; {CONDA_INIT}'


def wrap_local(cmd: str, conda_env: str = "") -> str:
    """wrap a local command with conda init and optional activate."""
    activate = f"conda activate {conda_env} && " if conda_env else ""
    escaped = cmd.replace("'", "'\\''")
    return f"bash -c 'export PYTHONUNBUFFERED=1; {CONDA_INIT}; {activate}{escaped}'"


def wrap_remote(cmd: str, conda_env: str = "") -> str:
    """wrap a shell command with locale, conda init and optional activate for remote SSH."""
    activate = f"conda activate {conda_env} && " if conda_env else ""
    escaped = cmd.replace("'", "'\\''")
    return f"bash -c '{REMOTE_SHELL_INIT}; {activate}{escaped}'"


def git_remote_url() -> str:
    """return the git remote origin url of the local project."""
    root = _find_project_root()
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5, cwd=root,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
