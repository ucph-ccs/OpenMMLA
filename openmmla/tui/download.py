"""resumable, progress-reporting downloads of remote artifacts.

The console used to pull remote recordings with a single ``scp -r`` into a
``TemporaryDirectory``: no way to tell how far along it was, and nothing
survived an interruption because the context manager wiped the temp tree on
the way out. This module replaces that with a three-part transfer:

1. one ssh round trip that *plans* the transfer — does the remote path exist,
   does the remote have rsync, and what is every file's size and mtime;
2. a transfer into a staging directory that lives next to the artifacts (so it
   survives a crash, a quit, or a dropped link) using rsync when both ends
   have it and scp otherwise;
3. a completeness gate — the staged tree is merged into ``artifacts/`` only
   once every planned file is there at its full remote size.

Progress is measured by polling the staging directory's byte count against the
planned total, never by parsing rsync/scp output: openrsync, GNU rsync and scp
all print different things, and the byte count is true for all three.

No textual import here on purpose: everything below is headless-testable.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from openmmla.tui.artifacts import artifact_session_dir, safe_segment
from openmmla.tui.ssh import _resolve_local_command, ssh_run_async, wrap_remote

PLAN_VERSION = 1
STAGING_DIR_NAME = ".staging"
PLAN_SUFFIX = ".plan.json"

# above this many files the per-file plan is dropped: the probe output (and the
# reconcile pass over it) stops being cheap, so we keep only the byte total
MAX_PLANNED_FILES = 5000
# above this many holes the scp path restarts recursively instead of issuing
# one ssh connection per file
SCP_PER_FILE_LIMIT = 200
# the recursive scp landing zone, kept inside staging so the progress poller
# sees the bytes arrive
SCP_SCRATCH_NAME = ".scp-incoming"
RSYNC_IO_TIMEOUT = 60
POLL_FAST = 0.5
POLL_SLOW = 1.5
RATE_WINDOW = 10
MAX_OUTPUT_BYTES = 64 * 1024

# rsync exits 1 on an unrecognized option and 2 on a protocol mismatch; both
# mean "this rsync pairing cannot work", so fall back instead of retrying
RSYNC_GIVE_UP_RCS = frozenset({1, 2})
# 24 is "some files vanished before they could be transferred", which is normal
# when a recorder is still writing; the completeness gate has the final say
RSYNC_OK_RCS = frozenset({0, 24})

RSYNC_EXCLUDES = (
    ".DS_Store",
    "__pycache__/",
    ".manifest.lock",
    "*.tmp",
    "*.part",
)

# a remote root safe to hand to rsync, which passes it through a remote shell
_SIMPLE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_@%+=:,./-]*$")


def is_excluded(rel: str) -> bool:
    """True for a remote file the transports are told to skip.

    This has to agree with RSYNC_EXCLUDES exactly: a file the probe plans but
    rsync never sends would leave the completeness gate permanently unsatisfied,
    and nothing would ever be merged."""
    parts = rel.split("/")
    name = parts[-1]
    if "__pycache__" in parts[:-1]:
        return True
    return (
        name in {".DS_Store", ".manifest.lock"}
        or name.endswith((".tmp", ".part"))
    )
# rels we refuse to plan: they cannot be quoted safely for the per-file scp path
_UNSAFE_REL = re.compile(r'["\\\n\r]')

# profile names whose remote rsync turned out to be unusable this process
_NO_RSYNC: set[str] = set()


# ── data model ───────────────────────────────────────────────


@dataclass(frozen=True)
class RemoteFile:
    rel: str
    size: int
    mtime: int


@dataclass
class RemotePlan:
    exists: bool = False
    has_rsync: bool = False
    files: list[RemoteFile] = field(default_factory=list)
    total_bytes: int = 0
    file_count: int = 0
    # False once the file list was dropped for being too large; the gate then
    # falls back to a byte total and resume degrades to "start over"
    exact: bool = True
    rejected: list[str] = field(default_factory=list)
    # files the transports are told to skip; planned files and transferred
    # files must be the same set or the gate can never be satisfied
    excluded: int = 0
    # set when ssh itself failed, so "not exists" is not mistaken for
    # "the directory is not there"
    unreachable: str = ""
    saw_marker: bool = False

    def by_rel(self) -> dict[str, RemoteFile]:
        return {item.rel: item for item in self.files}


@dataclass
class TransferResult:
    status: str  # complete | failed | incomplete | growing | empty
    tool: str = ""
    staged_root: Path | None = None
    rc: int = 0
    output: str = ""
    missing: list[str] = field(default_factory=list)
    grew: list[str] = field(default_factory=list)
    bytes_done: int = 0
    total_bytes: int = 0


class RateWindow:
    """trailing byte-rate estimator over the last few progress samples."""

    MIN_SAMPLES = 4

    def __init__(self, size: int = RATE_WINDOW) -> None:
        self._size = max(2, size)
        self._samples: list[tuple[float, int]] = []

    def observe(self, done: int) -> None:
        self._samples.append((time.monotonic(), done))
        if len(self._samples) > self._size:
            del self._samples[0]

    def rate(self) -> float:
        """bytes per second, or 0.0 until the window has enough samples."""
        if len(self._samples) < self.MIN_SAMPLES:
            return 0.0
        (first_at, first_done), (last_at, last_done) = self._samples[0], self._samples[-1]
        elapsed = last_at - first_at
        if elapsed <= 0:
            return 0.0
        moved = last_done - first_done
        return moved / elapsed if moved > 0 else 0.0

    def eta(self, done: int, total: int) -> float | None:
        rate = self.rate()
        if rate <= 0 or total <= 0 or done >= total:
            return None
        return (total - done) / rate


# ── formatting ───────────────────────────────────────────────


def fmt_bytes(value: float) -> str:
    """render a byte count the way transfer tools do (1000-based)."""
    number = float(value or 0)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(number) < 1000 or unit == "TB":
            if unit == "B":
                return f"{int(number)} B"
            return f"{number:.1f} {unit}"
        number /= 1000.0
    return f"{number:.1f} TB"


def fmt_rate(bytes_per_second: float) -> str:
    if not bytes_per_second or bytes_per_second <= 0:
        return "—"
    return f"{fmt_bytes(bytes_per_second)}/s"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    total = int(seconds)
    if total < 60:
        return f"~{total}s"
    if total < 3600:
        return f"~{total // 60}m{total % 60:02d}s"
    return f"~{total // 3600}h{(total % 3600) // 60:02d}m"


def progress_detail(done: int, total: int, window: RateWindow) -> str:
    if total <= 0:
        return f"{fmt_bytes(done)} · starting…"
    rate = window.rate()
    if rate <= 0:
        return f"{fmt_bytes(done)}/{fmt_bytes(total)} · starting…"
    return (
        f"{fmt_bytes(done)}/{fmt_bytes(total)} · {fmt_rate(rate)} · "
        f"{fmt_duration(window.eta(done, total))}"
    )


# ── staging paths ────────────────────────────────────────────


def staging_root(project_root: str | os.PathLike[str], session_id: str, *parts: str) -> Path:
    """return the staging tree for one download.

    Nested under the session directory rather than under ``artifacts/``: both
    session pickers enumerate ``artifacts/*`` one level deep and would offer a
    top-level ``.staging`` as a selectable session id."""
    root = artifact_session_dir(project_root, session_id) / STAGING_DIR_NAME
    for part in parts:
        root = root / safe_segment(part, "part")
    return root


def plan_path(staging: Path) -> Path:
    """the resume ledger, kept *beside* the staged tree.

    merge_tree only skips .DS_Store/.manifest.lock, so a state file inside the
    staged tree would be copied straight into artifacts/."""
    return staging.parent / f"{staging.name}{PLAN_SUFFIX}"


def finalize(staging: Path) -> None:
    """drop a staged tree and its ledger after a successful merge."""
    shutil.rmtree(staging, ignore_errors=True)
    with contextlib.suppress(OSError):
        plan_path(staging).unlink()
    _prune_empty_parents(staging.parent)


def _prune_empty_parents(start: Path) -> None:
    """remove empty directories upwards, stopping at (and including) .staging."""
    current = start
    for _ in range(8):
        if not current.is_dir() or not current.name:
            return
        try:
            next(current.iterdir())
            return  # still holds another download's data
        except StopIteration:
            pass
        except OSError:
            return
        last = current.name == STAGING_DIR_NAME
        with contextlib.suppress(OSError):
            current.rmdir()
        if last:
            return
        current = current.parent


def sweep_staging(project_root: str | os.PathLike[str], max_age_days: int = 14) -> int:
    """remove staged trees nobody came back to. returns how many were removed.

    Abandoned staging lives under a gitignored directory and can be gigabytes,
    so it is swept rather than kept forever."""
    artifacts = Path(project_root) / "artifacts"
    if not artifacts.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for session_dir in _iterdir(artifacts):
        staging_dir = session_dir / STAGING_DIR_NAME
        if not staging_dir.is_dir():
            continue
        for tree, ledger in _staged_trees(staging_dir):
            try:
                stamp = (ledger if ledger.exists() else tree).stat().st_mtime
            except OSError:
                continue
            if stamp >= cutoff:
                continue
            shutil.rmtree(tree, ignore_errors=True)
            with contextlib.suppress(OSError):
                ledger.unlink()
            removed += 1
        _prune_empty_tree(staging_dir)
    return removed


def _iterdir(path: Path) -> list[Path]:
    try:
        return [child for child in path.iterdir() if child.is_dir()]
    except OSError:
        return []


def _staged_trees(staging_dir: Path) -> list[tuple[Path, Path]]:
    """(tree, ledger) pairs under .staging, plus abandoned scp scratch dirs."""
    found: list[tuple[Path, Path]] = []
    for root, dirs, files in os.walk(staging_dir):
        for name in files:
            if name.endswith(PLAN_SUFFIX):
                tree = Path(root) / name[: -len(PLAN_SUFFIX)]
                found.append((tree, Path(root) / name))
        for name in dirs:
            if name.endswith(".scp"):
                scratch = Path(root) / name
                found.append((scratch, plan_path(scratch)))
    return found


def _prune_empty_tree(staging_dir: Path) -> None:
    """remove every empty directory under (and including) .staging."""
    for root, dirs, _files in os.walk(staging_dir, topdown=False):
        for name in dirs:
            with contextlib.suppress(OSError):
                (Path(root) / name).rmdir()
    with contextlib.suppress(OSError):
        staging_dir.rmdir()


# ── remote probe ─────────────────────────────────────────────


def quote_remote_path(path: str) -> str:
    """shell-quote a remote path, keeping a literal $HOME for the remote shell.

    A local copy of launcher._quote_remote_path; importing the launcher here
    would be a circular import."""
    text = str(path).strip()
    if text in ("~", "$HOME"):
        return "$HOME"
    if text.startswith("~/"):
        rest = text[2:]
    elif text.startswith("$HOME/"):
        rest = text[6:]
    else:
        return shlex.quote(text)
    parts = [shlex.quote(part) for part in rest.split("/") if part]
    return "/".join(["$HOME", *parts])


def build_probe_command(remote_path: str) -> str:
    """one shell command that plans a transfer in a single ssh round trip.

    Prints OK/MISSING, then RSYNC=0|1, then one `<size> <mtime> <path>` line
    per file. du is deliberately not used: it reports allocated blocks, so the
    progress bar would never reach 100%."""
    quoted = quote_remote_path(remote_path)
    return (
        f"if [ -d {quoted} ]; then "
        "printf 'OK\\n'; "
        "if command -v rsync >/dev/null 2>&1; then printf 'RSYNC=1\\n'; "
        "else printf 'RSYNC=0\\n'; fi; "
        f"cd {quoted} || exit 0; "
        # GNU coreutils stat and BSD stat spell every format character
        # differently, and the remote may be either
        "if stat -c '%s' . >/dev/null 2>&1; then "
        "find . -type f -exec stat -c '%s %Y %n' {} + 2>/dev/null; "
        "else "
        "find . -type f -exec stat -f '%z %m %N' {} + 2>/dev/null; "
        "fi; "
        "else printf 'MISSING\\n'; fi"
    )


def parse_probe(text: str) -> RemotePlan:
    """read probe output, tolerating motd/conda noise around it."""
    plan = RemotePlan()
    saw_marker = False
    for raw in str(text or "").splitlines():
        line = raw.rstrip("\r")
        stripped = line.strip()
        if stripped == "OK":
            plan.exists = True
            saw_marker = True
            continue
        if stripped == "MISSING":
            plan.exists = False
            saw_marker = True
            continue
        if stripped == "RSYNC=1":
            plan.has_rsync = True
            continue
        if stripped == "RSYNC=0":
            plan.has_rsync = False
            continue
        if not saw_marker or not plan.exists:
            continue
        parts = line.split(" ", 2)
        if len(parts) != 3:
            continue
        size_text, mtime_text, name = parts
        if not size_text.isdigit() or not mtime_text.lstrip("-").isdigit():
            continue
        rel = name[2:] if name.startswith("./") else name
        if not rel or rel.startswith("/") or _UNSAFE_REL.search(rel):
            plan.rejected.append(name)
            continue
        if is_excluded(rel):
            plan.excluded += 1
            continue
        plan.files.append(RemoteFile(rel, int(size_text), int(mtime_text)))

    plan.saw_marker = saw_marker
    plan.file_count = len(plan.files)
    plan.total_bytes = sum(item.size for item in plan.files)
    if plan.file_count > MAX_PLANNED_FILES:
        # keep the totals, drop the list: reconcile and the per-file gate stop
        # being worth their cost at this size
        plan.exact = False
        plan.files = []
    return plan


async def probe_remote(profile, remote_path: str) -> RemotePlan:
    """plan a transfer over the profile's shared ssh connection.

    An ssh that never reached the remote shell is reported as unreachable
    rather than as a missing directory, which is what the bare output would
    otherwise look like."""
    proc = await ssh_run_async(profile, wrap_remote(build_probe_command(remote_path)))
    output = await _drain(proc)
    rc = await proc.wait()
    plan = parse_probe(output)
    if rc != 0 and not plan.saw_marker:
        plan.unreachable = output.strip() or f"ssh exited {rc}"
    return plan


async def _drain(proc, limit: int = 4 * 1024 * 1024) -> str:
    """read a child's stdout in chunks.

    Never readline(): rsync-shaped output can carry more than StreamReader's
    64 KiB limit between newlines and raises ValueError."""
    if proc.stdout is None:
        return ""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        if size < limit:
            chunks.append(chunk)
            size += len(chunk)
    return b"".join(chunks).decode(errors="replace")


# ── local measurement, ledger, reconcile ─────────────────────


def dir_bytes(path: Path) -> int:
    """sum every regular file under path, dotfiles included.

    openrsync writes to a hidden `.<name>.XXXXXXXX` and renames only at the
    end, so skipping dotfiles would pin the bar at 0% for a whole file."""
    total = 0
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def read_ledger(staging: Path) -> tuple[str, dict[str, RemoteFile]]:
    """return (remote_root, {rel: RemoteFile}) recorded by the previous run."""
    path = plan_path(staging)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return "", {}
    if not isinstance(data, dict) or data.get("version") != PLAN_VERSION:
        return "", {}
    files = data.get("files")
    if not isinstance(files, dict):
        return "", {}
    recorded: dict[str, RemoteFile] = {}
    for rel, pair in files.items():
        if isinstance(pair, list) and len(pair) == 2:
            with contextlib.suppress(TypeError, ValueError):
                recorded[rel] = RemoteFile(rel, int(pair[0]), int(pair[1]))
    return str(data.get("remote_root") or ""), recorded


def write_ledger(staging: Path, plan: RemotePlan, remote_root: str) -> None:
    """record the plan beside the staged tree, atomically."""
    path = plan_path(staging)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PLAN_VERSION,
        "remote_root": remote_root,
        "exact": plan.exact,
        "total_bytes": plan.total_bytes,
        "file_count": plan.file_count,
        "files": {item.rel: [item.size, item.mtime] for item in plan.files},
    }
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


def reconcile_staging(
    staged_root: Path,
    plan: RemotePlan,
    ledger: dict[str, RemoteFile],
    remote_root: str,
    ledger_root: str,
) -> tuple[int, int]:
    """drop staged data that cannot be safely resumed. returns (dropped, resumable).

    openrsync's --append happily leaves a destination that is *larger* than the
    source in place and still exits 0, so an oversized stub has to go."""
    if not staged_root.exists():
        return 0, 0
    if not plan.exact:
        # too many files to reconcile one by one: keep what is there and let
        # rsync's own size+mtime check decide, with --append never running
        return 0, 0
    if (ledger_root and ledger_root != remote_root) or not ledger:
        dropped = _wipe_contents(staged_root)
        return dropped, 0

    planned = plan.by_rel()
    dropped = 0
    resumable = 0
    for rel, remote in planned.items():
        staged = staged_root / rel
        try:
            size = staged.stat().st_size
        except OSError:
            continue
        recorded = ledger.get(rel)
        if recorded is None or recorded.mtime != remote.mtime:
            # the remote file was re-recorded under the same name
            _unlink(staged)
            dropped += 1
            continue
        if size > remote.size:
            _unlink(staged)
            dropped += 1
        elif size < remote.size:
            resumable += 1
    # anything staged that the remote no longer plans is stale, and merge_tree
    # would copy it straight into artifacts/
    for path in _staged_files(staged_root):
        rel = path.relative_to(staged_root).as_posix()
        if rel not in planned and not rel.startswith(f"{SCP_SCRATCH_NAME}/"):
            _unlink(path)
            dropped += 1
    return dropped, resumable


def _staged_files(staged_root: Path) -> list[Path]:
    found: list[Path] = []
    for root, _dirs, files in os.walk(staged_root):
        for name in files:
            found.append(Path(root) / name)
    return found


def _wipe_contents(staged_root: Path) -> int:
    dropped = 0
    for path in _staged_files(staged_root):
        _unlink(path)
        dropped += 1
    return dropped


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def missing_files(staged_root: Path, plan: RemotePlan) -> list[str]:
    """planned files that are absent or short."""
    holes = []
    for item in plan.files:
        try:
            size = (staged_root / item.rel).stat().st_size
        except OSError:
            holes.append(item.rel)
            continue
        if size < item.size:
            holes.append(item.rel)
    return holes


def grown_files(staged_root: Path, plan: RemotePlan) -> list[str]:
    """planned files that arrived larger than planned — still being recorded."""
    grew = []
    for item in plan.files:
        try:
            size = (staged_root / item.rel).stat().st_size
        except OSError:
            continue
        if size > item.size:
            grew.append(item.rel)
    return grew


# ── transports ───────────────────────────────────────────────


def local_rsync_path() -> str:
    resolved = _resolve_local_command("rsync")
    return "" if resolved == "rsync" else resolved


def build_rsync_argv(profile, remote_root: str, staging: Path, *, append: bool) -> list[str] | None:
    """return the rsync argv, or None when rsync cannot be used for this pair.

    Every flag here is load-bearing against macOS's openrsync: -t is what makes
    a second pass cost nothing, --partial is what leaves anything behind on an
    interrupt, and --append is the only thing that actually resumes a file
    (openrsync implements no delta algorithm). --info=progress2 and
    --append-verify are GNU-only and make openrsync exit 1."""
    if not local_rsync_path():
        return None
    if not _SIMPLE_REMOTE_PATH.match(str(remote_root)):
        return None
    transport = profile.rsync_shell_arg()
    if not transport:
        return None
    argv = profile.base_rsync_args()
    argv.extend(["-rt", "--partial", f"--timeout={RSYNC_IO_TIMEOUT}"])
    if append:
        argv.append("--append")
    for pattern in RSYNC_EXCLUDES:
        argv.append(f"--exclude={pattern}")
    argv.extend(["-e", transport])
    # trailing slashes on both sides: the remote directory's *contents* land in
    # the staging root, so there is no basename guessing afterwards
    argv.append(f"{profile.ssh_destination()}:{str(remote_root).rstrip('/')}/")
    argv.append(f"{str(staging).rstrip('/')}/")
    return argv


def build_scp_recursive_argv(profile, remote_root: str, destination: Path) -> list[str]:
    """the legacy recursive form, argv-identical to scp_from_remote_async."""
    argv = profile.base_scp_args()
    argv.extend(["-r", f"{profile.ssh_destination()}:{remote_root}", str(destination)])
    return argv


def build_scp_file_argv(profile, remote_root: str, rel: str, target: Path) -> list[str]:
    """fetch one planned file.

    The remote path is passed through unquoted, exactly as the recursive form
    does: modern scp speaks SFTP and looks the path up literally, so added
    quotes would become part of the filename."""
    remote_file = f"{str(remote_root).rstrip('/')}/{rel}"
    argv = profile.base_scp_args()
    argv.extend([f"{profile.ssh_destination()}:{remote_file}", str(target)])
    return argv


# ── child process control ────────────────────────────────────


def _killpg(proc, sig) -> None:
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), sig)


async def _run_child(argv: list[str]) -> tuple[int, str]:
    """run a transfer child in its own process group and return (rc, output).

    The process group is what lets a cancelled download reap sshpass, rsync and
    the ssh it forks; today's code never signals the scp child at all."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output = await _drain(proc, limit=MAX_OUTPUT_BYTES)
        rc = await proc.wait()
        return rc, output
    finally:
        if proc.returncode is None:
            _killpg(proc, signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), 3)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                _killpg(proc, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(proc.wait()), 1)


# ── driver ───────────────────────────────────────────────────

OnProgress = Callable[[int, int, str], None]
OnLog = Callable[[str], None]


async def download_tree(
    profile,
    remote_root: str,
    *,
    staging: Path,
    plan: RemotePlan,
    on_progress: OnProgress,
    log: OnLog,
    poll_interval: float | None = None,
) -> TransferResult:
    """transfer a remote tree into staging, reporting progress as it goes.

    Never raises on a transfer failure — the caller reads TransferResult.status
    and only merges on "complete". On cancellation the child is killed and the
    staged data is left in place so the next press resumes."""
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    total = plan.total_bytes

    ledger_root, ledger = read_ledger(staging)
    dropped, resumable = await asyncio.to_thread(
        reconcile_staging, staging, plan, ledger, remote_root, ledger_root
    )
    await asyncio.to_thread(write_ledger, staging, plan, remote_root)

    already = await asyncio.to_thread(dir_bytes, staging)
    if dropped:
        log(f"  [yellow]Discarded {dropped} unusable staged file(s).[/yellow]")
    if already:
        percent = int(already * 100 / total) if total else 0
        log(f"  [cyan]Resuming: {fmt_bytes(already)} of {fmt_bytes(total)} already staged ({percent}%).[/cyan]")

    window = RateWindow()
    window.observe(already)
    on_progress(already, total, progress_detail(already, total, window))

    if poll_interval is None:
        poll_interval = POLL_FAST if plan.file_count <= 500 else POLL_SLOW
    poller = asyncio.ensure_future(
        _poll_loop(staging, total, on_progress, poll_interval, window)
    )

    tool = ""
    rc = 0
    output = ""
    try:
        # any other reason rsync cannot be used (an unquotable remote path, an
        # unusable transport) surfaces as a give-up exit code and falls back
        use_rsync = (
            plan.has_rsync
            and getattr(profile, "name", "") not in _NO_RSYNC
            and bool(local_rsync_path())
        )
        if use_rsync:
            tool = "rsync"
            rc, output = await _rsync_transfer(profile, remote_root, staging, resumable, log)
            if rc in RSYNC_GIVE_UP_RCS:
                log(f"  [yellow]rsync unusable here (exit {rc}); falling back to scp.[/yellow]")
                _NO_RSYNC.add(getattr(profile, "name", ""))
                tool = "scp"
                rc, output = await _scp_transfer(profile, remote_root, staging, plan, log)
        else:
            tool = "scp"
            if plan.has_rsync:
                log("  [yellow]rsync unavailable locally; using scp.[/yellow]")
            rc, output = await _scp_transfer(profile, remote_root, staging, plan, log)
    finally:
        poller.cancel()
        # suppress broadly: a progress callback that blew up must not mask the
        # transfer's own result
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await poller

    done = await asyncio.to_thread(dir_bytes, staging)
    window.observe(done)
    on_progress(done, total, progress_detail(done, total, window))

    result = TransferResult(
        status="complete",
        tool=tool,
        staged_root=staging,
        rc=rc,
        output=output,
        bytes_done=done,
        total_bytes=total,
    )
    if rc not in RSYNC_OK_RCS:
        result.status = "failed"
        return result
    if plan.exact:
        result.grew = await asyncio.to_thread(grown_files, staging, plan)
        result.missing = await asyncio.to_thread(missing_files, staging, plan)
        if result.grew:
            result.status = "growing"
        elif result.missing:
            result.status = "incomplete"
    elif done < total:
        result.status = "incomplete"
    return result


async def _poll_loop(
    staging: Path,
    total: int,
    on_progress: OnProgress,
    interval: float,
    window: RateWindow,
) -> None:
    while True:
        await asyncio.sleep(interval)
        done = await asyncio.to_thread(dir_bytes, staging)
        window.observe(done)
        on_progress(done, total, progress_detail(done, total, window))


async def _rsync_transfer(
    profile,
    remote_root: str,
    staging: Path,
    resumable: int,
    log: OnLog,
) -> tuple[int, str]:
    """one rsync pass, or an --append pass followed by a plain verify pass.

    The verify pass exists because --append trusts that whatever is already on
    disk is a prefix of the source. It costs almost nothing on a tree rsync has
    just synced, because -t already matched every mtime."""
    if resumable:
        argv = build_rsync_argv(profile, remote_root, staging, append=True)
        if argv is None:
            return 1, "rsync argv unavailable"
        rc, output = await _run_child(argv)
        if rc in RSYNC_GIVE_UP_RCS:
            return rc, output
        if rc not in RSYNC_OK_RCS:
            return rc, output
        log("  [cyan]Verifying resumed files…[/cyan]")
    argv = build_rsync_argv(profile, remote_root, staging, append=False)
    if argv is None:
        return 1, "rsync argv unavailable"
    return await _run_child(argv)


async def _scp_transfer(
    profile,
    remote_root: str,
    staging: Path,
    plan: RemotePlan,
    log: OnLog,
) -> tuple[int, str]:
    """scp cannot resume a file, but it can skip the ones already complete."""
    if plan.exact:
        holes = await asyncio.to_thread(missing_files, staging, plan)
        if not holes:
            return 0, ""
        if len(holes) <= SCP_PER_FILE_LIMIT and len(holes) < plan.file_count:
            log(f"  [cyan]scp: fetching {len(holes)} of {plan.file_count} file(s).[/cyan]")
            for rel in holes:
                target = staging / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                _unlink(target)  # scp has no resume; a short file must go
                rc, output = await _run_child(build_scp_file_argv(profile, remote_root, rel, target))
                if rc != 0:
                    return rc, output
            return 0, ""
    return await _scp_recursive(profile, remote_root, staging)


async def _scp_recursive(profile, remote_root: str, staging: Path) -> tuple[int, str]:
    """pull the whole tree, then move it into the staging root.

    scp -r drops the remote directory *itself* into the destination, so the
    transfer lands in a scratch dir and is relocated afterwards; rsync's
    trailing slashes give the contents-in-staging shape directly.

    The scratch sits *inside* staging for two reasons: the progress poller
    measures staging, so a sibling directory would leave the bar at 0% for the
    whole transfer; and whatever arrived is relocated even when the transfer
    fails or is cancelled, so the completed files survive for the next press."""
    scratch = staging / SCP_SCRATCH_NAME
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        return await _run_child(build_scp_recursive_argv(profile, remote_root, scratch))
    finally:
        landed = scratch / os.path.basename(str(remote_root).rstrip("/"))
        if not landed.is_dir():
            children = [path for path in scratch.iterdir() if path.name != ".DS_Store"]
            landed = children[0] if len(children) == 1 else landed
        if landed.is_dir():
            await asyncio.to_thread(_relocate_into, landed, staging)
        shutil.rmtree(scratch, ignore_errors=True)


def _relocate_into(source: Path, destination: Path) -> None:
    """move every file under source into destination, replacing what is there."""
    destination.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(source):
        rel_root = Path(root).relative_to(source)
        target_root = destination / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            with contextlib.suppress(OSError):
                os.replace(Path(root) / name, target_root / name)
