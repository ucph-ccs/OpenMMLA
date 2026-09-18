from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import stat
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
    # kept out of the repr: a crash prints the locals of every frame, and a
    # profile in scope would put its password on the terminal
    password: str = field(default="", repr=False)
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

    def rsync_shell_arg(self) -> str:
        """return the ssh command line rsync should use for -e.

        rsync word-splits the -e value, so every token is shell-quoted here.
        Note ssh spells the port -p, unlike scp's -P."""
        parts = [_resolve_local_command("ssh")] + self._common_ssh_opts()
        if self.key_path:
            parts.extend(["-i", os.path.expanduser(self.key_path)])
        if self.port != 22:
            parts.extend(["-p", str(self.port)])
        return " ".join(shlex.quote(part) for part in parts)

    def base_rsync_args(self) -> list[str]:
        """return the rsync argument prefix; the caller appends -e and paths.

        sshpass wraps rsync itself rather than living inside -e: a password
        containing a space would be mangled by rsync's word-splitting."""
        args = self._sshpass_prefix()
        args.append(_resolve_local_command("rsync"))
        return args


def _profiles_path() -> str:
    root = _find_project_root()
    return os.path.join(root, PROFILES_DIR, PROFILES_FILE)


def _decrypt_password(value) -> str:
    """return a usable plaintext password from a stored profile value.

    Profiles written before passwords were encrypted still hold plaintext, so
    a non-ENC value is passed through unchanged. A value that is encrypted but
    undecryptable yields "" rather than the ciphertext, so ssh falls back to
    key auth instead of authenticating with the ENC(...) blob."""
    text = str(value or "")
    if not text:
        return ""
    try:
        from openmmla.utils.crypto import is_encrypted, decrypt_value
    except Exception:
        return text  # crypto unavailable: the value can only be plaintext
    if not is_encrypted(text):
        return text
    try:
        return decrypt_value(text)
    except Exception:
        return ""


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
                    password=_decrypt_password(entry.get("password", "")),
                    key_path=entry.get("key_path", ""),
                    remote_project_path=entry.get("remote_project_path", "~/OpenMMLA"),
                ))
        return profiles
    except Exception:
        return []


def save_ssh_profiles(profiles: list[SSHProfile]) -> None:
    """save ssh profiles to .openmmla/ssh_profiles.yml.

    Passwords are encrypted to ENC(...) with the master key before hitting
    disk, so a profile store that is copied or shared never carries a readable
    password. The file is gitignored; ssh_profiles_template.yml is the tracked
    stand-in."""
    path = _profiles_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [asdict(p) for p in profiles]
    try:
        from openmmla.utils.crypto import encrypt_sensitive_values, ensure_master_key
        encrypt_sensitive_values(data, ensure_master_key())
    except Exception:
        pass  # crypto unavailable: fall back to writing values as-is
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


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
        # never inherit: that hands the console's own terminal to the remote
        # command, which then eats the user's keystrokes, and a prompt on it
        # (sudo) waits for an answer that cannot arrive
        stdin=asyncio.subprocess.PIPE if pipe_stdin else asyncio.subprocess.DEVNULL,
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
            note = verify_stored_password(profile) if profile.password else None
            return True, "Connection successful" + (f"; {note}" if note else "")
        return False, result.stderr.strip() or f"exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "Connection timed out"
    except FileNotFoundError:
        return False, "ssh command not found"
    except Exception as e:
        return False, str(e)


def verify_stored_password(profile: SSHProfile, timeout: float = 10.0) -> str | None:
    """None when the stored password logs in by itself, else why it could not.

    A key in authorized_keys logs in whatever the profile's password says, so
    a placeholder left there passes every connection test while sudo on that
    host, which asks for the account's password, refuses it at the first
    Start. This logs in with the password alone, on a connection of its own:
    a multiplexed one would ride the key's login."""
    args = profile._sshpass_prefix() + [
        _resolve_local_command("ssh"),
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password,keyboard-interactive",
        "-o", "NumberOfPasswordPrompts=1",
    ]
    if profile.port != 22:
        args.extend(["-p", str(profile.port)])
    args.extend([profile.ssh_destination(), "echo PASSWORD_OK"])
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "the stored password could not be checked (timed out)"
    except Exception as e:
        return f"the stored password could not be checked ({e})"
    if result.returncode == 0 and "PASSWORD_OK" in result.stdout:
        return None
    stderr = result.stderr.strip()
    # sshd names the methods it takes: "Permission denied (publickey,password)"
    if "Permission denied" in stderr and "password" not in stderr.rsplit("Permission denied", 1)[-1]:
        return ("the stored password could not be checked: this host's sshd takes keys only "
                "(sudo there still asks for the account's password)")
    return ("the stored password is not the account's: a key logged you in, but sudo on this host "
            "will refuse it. Put the account's real password here, or give the account passwordless "
            "sudo on that host")


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

# profile name -> failed probes in a row while TARGET_STATES still says online
_PROBE_MISSES: dict[str, int] = {}


# host the Launcher's Host bar is on; other screens open on the same host
# instead of always falling back to "local"
_CURRENT_TARGET = "local"


def is_select_sentinel(value) -> bool:
    """True for the placeholder values a Select emits while (re)building its
    options (Select.BLANK / Select.NULL depending on textual version)."""
    if value is None:
        return True
    return str(value) in ("Select.BLANK", "Select.NULL")


def set_current_target(name) -> None:
    """record the host the Launcher's Host bar moved to."""
    global _CURRENT_TARGET
    if is_select_sentinel(name):
        return
    text = str(name or "").strip()
    if not text or text == REFRESH_TARGETS_OPTION:
        return
    _CURRENT_TARGET = text


def current_target() -> str:
    """host the Launcher's Host bar is on ("local" until it is switched)."""
    return _CURRENT_TARGET


# profile name -> "linux" | "darwin" | "windows", learnt the first time a host is
# asked (it costs an ssh round trip) and forgotten when the profile is edited
TARGET_PLATFORMS: dict[str, str] = {}

# everything the console does on a remote host goes through a POSIX shell
# (bash -lc, tmux, conda, test/cat/mkdir -p) and the recorders need POSIX file
# locks and signals, so the cmd.exe or PowerShell of Windows' own OpenSSH server
# cannot be driven. WSL2 as the machine's ssh shell answers `uname` as Linux
WINDOWS_HOST_NOTE = (
    "runs Windows, and the console needs a POSIX shell on a remote host (bash, tmux, conda). "
    "Use a Linux or macOS machine; with WSL2 as the machine's ssh shell it counts as Linux, "
    "though cameras and microphones are not visible in there"
)


def remote_platform(profile: SSHProfile, timeout: float = 8.0) -> str:
    """"linux", "darwin" or "windows"; "" when the host did not say.

    `uname -s` answers wherever there is a POSIX shell. Windows' own OpenSSH
    server starts cmd.exe or PowerShell, which do not know it; `echo %OS%
    $env:OS` prints Windows_NT in either of them (and nothing of the kind in
    sh), which tells such a host from one that simply did not answer."""
    cached = TARGET_PLATFORMS.get(profile.name)
    if cached:
        return cached
    platform_name = ""
    try:
        raw = (ssh_run_sync(profile, "uname -s", timeout=timeout).stdout or "").lower()
        if "darwin" in raw:
            platform_name = "darwin"
        elif "linux" in raw:
            platform_name = "linux"
        else:
            raw = ssh_run_sync(profile, "echo %OS% $env:OS", timeout=timeout).stdout or ""
            if "windows_nt" in raw.lower():
                platform_name = "windows"
    except Exception:
        return ""
    if platform_name:
        TARGET_PLATFORMS[profile.name] = platform_name
    return platform_name


def target_state_label(name: str) -> str:
    """render a profile name with its cached reachability state."""
    state = TARGET_STATES.get(name)
    if TARGET_PLATFORMS.get(name) == "windows":
        return f"{name}  (Windows: not supported ✗)"
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


def probe_all_profiles(
    timeout: float = 1.0, deadline: float = 15.0, confirm_offline: int = 1,
) -> dict[str, str]:
    """probe every SSH profile concurrently and update TARGET_STATES.

    The connect timeout does not bound DNS/mDNS resolution (a dead .local
    name can take ~5s to fail), so an overall deadline caps the wall time:
    anything unresolved by then is reported offline.

    A host that was online goes offline only after `confirm_offline` failed
    probes in a row. A background loop passes 2, so one blip of this
    machine's network (every host failing at once) does not move cards to
    Local; a probe the user asked for passes the default and is definitive."""
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
    for name, state in states.items():
        if state == "offline" and TARGET_STATES.get(name) == "online":
            misses = _PROBE_MISSES.get(name, 0) + 1
            if misses < confirm_offline:
                _PROBE_MISSES[name] = misses
                states[name] = "online"
                continue
        _PROBE_MISSES.pop(name, None)
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
