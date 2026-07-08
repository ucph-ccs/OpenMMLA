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
    # fast-fail on unreachable hosts before paying the full SSH handshake
    # timeout (ConnectTimeout=5 / subprocess cap 10s)
    if not probe_ssh_endpoint(profile.host, profile.port, timeout=1.5):
        return False, f"host unreachable ({profile.host}:{profile.port})"
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


_RESOLVED_SSH_ENDPOINTS: dict[tuple[str, int], tuple[str, int]] = {}


def resolve_ssh_endpoint(host: str, port: int) -> tuple[str, int]:
    """resolve the effective hostname/port via `ssh -G`, honoring ~/.ssh/config
    aliases (Host server-01 -> HostName 192.168.x.x) that a plain socket
    lookup cannot see. Results are cached."""
    key = (host, port)
    cached = _RESOLVED_SSH_ENDPOINTS.get(key)
    if cached:
        return cached
    resolved_host, resolved_port = host, port
    try:
        result = subprocess.run(
            [_resolve_local_command("ssh"), "-G", "-p", str(port), host],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                k, v = parts[0].lower(), parts[1].strip()
                if k == "hostname" and v:
                    resolved_host = v
                elif k == "port":
                    try:
                        resolved_port = int(v)
                    except ValueError:
                        pass
    except Exception:
        pass
    _RESOLVED_SSH_ENDPOINTS[key] = (resolved_host, resolved_port)
    return resolved_host, resolved_port


# profile name -> "online" / "offline"; written by the launcher's background
# probe loop, read by every screen's target dropdown
TARGET_STATES: dict[str, str] = {}


def is_select_sentinel(value) -> bool:
    """True for the placeholder values a Select emits while (re)building its
    options (Select.BLANK / Select.NULL depending on textual version)."""
    if value is None:
        return True
    return str(value) in ("Select.BLANK", "Select.NULL")


def target_state_label(name: str) -> str:
    """render a profile name with its cached reachability state."""
    state = TARGET_STATES.get(name)
    if state == "online":
        return f"{name}  (online)"
    if state == "offline":
        return f"{name}  (offline ✗)"
    return name


# legacy sentinel: connectivity tests are now triggered by the ↻ button next
# to each Host selector; kept so existing Select.Changed handlers stay valid
REFRESH_TARGETS_OPTION = "__refresh_targets__"


def target_options() -> list[tuple[str, str]]:
    """unified (label, value) options for all Host selectors."""
    return (
        [("Local", "local")]
        + [(target_state_label(p.name), p.name) for p in load_ssh_profiles()]
    )


def probe_all_profiles(timeout: float = 1.0, deadline: float = 15.0) -> dict[str, str]:
    """probe every SSH profile concurrently and update TARGET_STATES.

    The connect timeout does not bound DNS/mDNS resolution (a dead .local
    name can take ~5s to fail), so an overall deadline caps the wall time:
    anything unresolved by then is reported offline."""
    from concurrent.futures import ThreadPoolExecutor, wait
    profiles = load_ssh_profiles()
    if not profiles:
        TARGET_STATES.clear()
        return {}

    pool = ThreadPoolExecutor(max_workers=min(16, len(profiles)))
    futures = {
        pool.submit(probe_ssh_endpoint, profile.host, profile.port, timeout): profile.name
        for profile in profiles
    }
    done, _pending = wait(futures, timeout=deadline)
    states: dict[str, str] = {}
    for future, name in futures.items():
        if future in done:
            try:
                ok = bool(future.result())
            except Exception:
                ok = False
            states[name] = "online" if ok else "offline"
        else:
            # still resolving when the deadline hit (e.g. slow mDNS): we don't
            # know either way — never mislabel a reachable host as offline
            states[name] = "unknown"
    # do not wait for stragglers stuck in mDNS resolution
    pool.shutdown(wait=False, cancel_futures=True)
    TARGET_STATES.clear()
    TARGET_STATES.update(states)
    return states


def test_profile_by_name(name: str) -> tuple[bool, str]:
    """run a full SSH connectivity test for one saved profile and record the
    definitive result in TARGET_STATES."""
    profile = get_profile_by_name(name)
    if profile is None:
        return False, f"profile '{name}' not found"
    success, msg = ssh_test_connection(profile)
    TARGET_STATES[name] = "online" if success else "offline"
    return success, msg


def summarize_states(states: dict[str, str]) -> str:
    """human-readable summary like '2 online, 5 offline, 1 unknown'."""
    counts = {"online": 0, "offline": 0, "unknown": 0}
    for state in states.values():
        counts[state] = counts.get(state, 0) + 1
    parts = [f"{count} {state}" for state, count in counts.items() if count]
    return ", ".join(parts) if parts else "no hosts configured"


def probe_ssh_endpoint(host: str, port: int, timeout: float = 1.0) -> bool:
    """cheap reachability check: TCP connect to the (ssh-config resolved) SSH port."""
    import socket
    host, port = resolve_ssh_endpoint(host, port)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


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
