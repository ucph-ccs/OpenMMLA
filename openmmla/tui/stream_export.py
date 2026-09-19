"""capture-side stream recordings, taken off the machines that capture them: a
session's part of the streams it used.

A managed stream with `record: true` writes one file per run on its capture
host, <record_root>/streams/capture/<YYYY-MM-DD>/<host label>/<video|audio>/
<name>_<start>.<mkv|wav>, filed by day because the stream is shared by the
sessions that pull it. A session's part is cut there (stream_cuts: no
re-encoding, a video cut moved back onto a keyframe and named after it),
staged under <record_root>/streams/.session-cuts/<session>/<host label>/,
fetched with the resumable staged transfer (download), and lands in
artifacts/<session>/streams/capture/<host label>/<video|audio>/; the staging is
removed once it has arrived. A stream captured on this machine is cut straight
into place. The recordings themselves are never touched.

Nothing here draws. Whoever runs it hands in ExportCallbacks for its log lines
(Rich markup) and its progress row, and cancels by cancelling the task that
runs it (a Textual worker's cancel) or by answering True from
callbacks.cancelled(), which is asked between steps. Either way the coroutine
raises asyncio.CancelledError with the progress row closed; what was staged
stays, so the next run resumes it."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from rich.markup import escape

from openmmla.tui import download as dl
from openmmla.tui import stream_cuts
from openmmla.tui.artifacts import METADATA_FILENAMES, copy_covers, merge_tree, update_session_manifest
from openmmla.tui.ssh import get_profile_by_name
from openmmla.utils import session_sources
from openmmla.utils.artifact_paths import (
    CAPTURE_KINDS, CAPTURE_STREAMS_DIR, STREAMS_DIR, capture_host_label, capture_record_root, safe_segment,
    session_capture_streams_dir,
)


# ---- what is recorded where ----

@dataclass(frozen=True)
class RecordedStream:
    """where a managed stream keeps its recordings: enough to find them again."""
    name: str
    ssh_profile: str   # "local" or an SSH profile
    record_root: str   # on the capture host; $HOME/... for a remote one
    host_label: str    # the folder under streams/capture/<day>/
    kind: str          # video | audio


@dataclass
class SessionStreams:
    """the streams a session's bases noted they took, as the capture side sees them."""
    streams: list[RecordedStream] = field(default_factory=list)   # to cut: the console runs them, Record on
    # (name, why) of the others, in the order the bases noted them: "external"
    # (someone else runs its ffmpeg) or "not recorded" (Record was off for it);
    # only the Stream Server may have those
    skipped: list[tuple[str, str]] = field(default_factory=list)


def session_streams(record: dict | None, project_root) -> SessionStreams:
    """the capture-side streams of a session record (session_sources.captured_streams):
    record_root with ~ spelled $HOME, empty meaning artifacts/ of the project for
    'local' and $HOME/artifacts on any other host; the host label of 'local' is
    this machine's short name, else the SSH profile."""
    found = SessionStreams()
    for entry in session_sources.captured_streams(record):
        name, profile = entry["name"], entry["ssh_profile"]
        if not profile:
            found.skipped.append((name, "external"))
        elif not entry["record"]:
            found.skipped.append((name, "not recorded"))
        else:
            found.streams.append(RecordedStream(
                name, profile, capture_record_root(profile, entry["record_root"], project_root),
                capture_host_label(profile), entry["kind"] if entry["kind"] in CAPTURE_KINDS else "video"))
    return found


# ---- how it reports ----

def _nothing(*_args) -> None:
    return None


def _never() -> bool:
    return False


@dataclass
class ExportCallbacks:
    """how an export tells whoever runs it what it does.

    log(line): one line of Rich markup.
    progress_start(label, total): a transfer begins; total is its bytes, None
        while it is still being sized (it may be called again with the total).
    progress_update(done, total, detail): bytes arrived so far, and a
        `12.1 MB/40.0 MB · 3.2 MB/s · ~9s` detail.
    progress_end(): the transfer is over, whatever became of it.
    cancelled(): asked between steps; True stops the export (CancelledError)."""
    log: Callable[[str], None] = _nothing
    progress_start: Callable[[str, "int | None"], None] = _nothing
    progress_update: Callable[[int, int, str], None] = _nothing
    progress_end: Callable[[], None] = _nothing
    cancelled: Callable[[], bool] = _never


def _check(callbacks: ExportCallbacks) -> None:
    if callbacks.cancelled():
        raise asyncio.CancelledError()


# ---- running a script where a stream records ----

HostRunner = Callable[[str, str, float], Awaitable["str | None"]]


def _kill(proc, sig) -> None:
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), sig)


async def run_on_host(ssh_profile: str, script: str, timeout: float = 60.0) -> str | None:
    """stdout and stderr of a script run through bash on a stream's capture
    host ('local': this machine), with the PATH its ffmpeg has; None when it
    could not be run there (no such SSH profile, ssh or bash did not start, or
    it took longer than `timeout`). Cancelled, it stops what it started here."""
    command = stream_cuts.with_tool_path(stream_cuts.bash(script))
    if ssh_profile == "local":
        argv = ["bash", "-c", command]
    else:
        profile = get_profile_by_name(ssh_profile)
        if profile is None:
            return None
        argv = [*profile.base_ssh_args(), command]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    except OSError:
        return None
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        return None
    finally:
        if proc.returncode is None:
            # its own process group: ssh, and sshpass before it, go with it
            _kill(proc, signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), 3)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                _kill(proc, signal.SIGKILL)
    return output.decode(errors="replace")


# ---- the transfer: staged, resumable, merged once complete ----

# files a transfer lists one by one before it starts; the rest are counted
LIST_MAX = 12


def describe_files(rels) -> str:
    """"1 video, 1 audio, 2 manifest(s)" for paths of a recordings tree."""
    counts: dict[str, int] = {}
    for rel in rels:
        if os.path.basename(rel) in METADATA_FILENAMES:
            kind = "manifest(s)"
        else:
            head = rel.split("/", 1)[0] if "/" in rel else ""
            kind = head if head in CAPTURE_KINDS else "other"
        counts[kind] = counts.get(kind, 0) + 1
    order = ("video", "audio", "other", "manifest(s)")
    return ", ".join(f"{counts[kind]} {kind}" for kind in order if kind in counts) or "no files"


def files_already_here(local_path: Path, plan) -> list[str]:
    """the planned files already at their place here, at the remote size.

    A recording is named after the moment it started and is not written again
    once it is over, so a copy of that size is that file. The manifests are
    fetched every time: they are small, and a stop rewrites them in place."""
    here = []
    for item in plan.files:
        if os.path.basename(item.rel) in METADATA_FILENAMES:
            continue
        try:
            if (Path(local_path) / item.rel).stat().st_size == item.size:
                here.append(item.rel)
        except OSError:
            continue
    return here


def log_plan(log: Callable[[str], None], plan, here: list[str], describe: Callable = describe_files) -> None:
    """what a transfer holds, file by file, before it starts. `describe` sorts
    the files for the first line; "" leaves it at their count."""
    size = dl.fmt_bytes(plan.total_bytes)
    kinds = describe([item.rel for item in plan.files]) if plan.exact else ""
    log(f"  {plan.file_count} file(s), {size}" + (f": {kinds}" if kinds else ""))
    if not plan.exact:
        return
    here_set = set(here)
    media = [item for item in plan.files if os.path.basename(item.rel) not in METADATA_FILENAMES]
    for item in media[:LIST_MAX]:
        note = "  [dim](already here)[/dim]" if item.rel in here_set else ""
        log(f"    {escape(item.rel)}  {dl.fmt_bytes(item.size)}{note}")
    if len(media) > LIST_MAX:
        log(f"    … and {len(media) - LIST_MAX} more")


def report_incomplete(log: Callable[[str], None], result) -> None:
    """explain a transfer that must not be merged, and how to continue."""
    if result.status == "growing":
        names = ", ".join(result.grew[:3])
        more = f" (+{len(result.grew) - 3} more)" if len(result.grew) > 3 else ""
        log(f"[yellow]{len(result.grew)} file(s) are still being written on the remote host: "
            f"{escape(names)}{more}.[/yellow]")
        log("[yellow]Stop the recorders, then download again.[/yellow]")
    elif result.status == "incomplete":
        log(f"[red]Download incomplete via {result.tool}: {len(result.missing)} file(s) did not arrive in full.[/red]")
    else:
        log(f"[red]Download failed via {result.tool} (exit {result.rc}).[/red]")
    for line in str(result.output or "").strip().splitlines()[-20:]:
        log(escape(line))
    log("[yellow]Nothing was merged into artifacts/. The data already fetched is kept — "
        "press Download again to resume.[/yellow]")


async def fetch_tree(
    profile,
    remote_dir: str,
    local_dir: Path,
    *,
    staging: Path,
    label: str,
    where: str,
    callbacks: ExportCallbacks,
    what: str | None = None,
    merge: Callable[..., dict] | None = None,
    after_merge: Callable[[], "str | None"] | None = None,
    describe: Callable = describe_files,
) -> bool:
    """scan, download (staged, resumable) and merge one remote tree into
    `local_dir`: only what is not here yet at the remote size is fetched, and
    nothing is merged before every planned file has arrived in full. True when
    it is all here afterwards.

    `remote_dir` is absolute on the host (or $HOME/...); `staging` is this
    machine's resume folder for it (download.staging_root); `label` heads the
    progress row, `where` (the SSH profile) names the host in the log and the
    conflict copies, `what` (default: label) names the tree in the log.
    `merge(source, destination, conflict_label=...)` defaults to
    artifacts.merge_tree; `after_merge()`, run between the merge and the removal
    of the staging (in a thread), may return a line for the log. `describe(rels)`
    names a set of files in the log (describe_files: by recording kind); an
    empty answer names them by their count."""
    merge = merge or merge_tree
    log = callbacks.log

    def named(files) -> str:
        rels = [item.rel for item in files]
        return describe(rels) or f"{len(rels)} file(s)"

    log(f"[cyan]Downloading {escape(where)}:{escape(remote_dir)} -> {escape(str(local_dir))}[/cyan]")
    # the row goes up before the remote scan, so Cancel is reachable while a
    # large tree is being sized
    callbacks.progress_start(f"{label} · scanning…", None)
    try:
        plan = await dl.probe_remote(profile, remote_dir)
        if plan.unreachable:
            log(f"[red]Could not reach {escape(where)}: {escape(plan.unreachable)}[/red]")
            return False
        if not plan.exists:
            log(f"[red]Remote folder not found: {escape(remote_dir)}[/red]")
            return False
        if not plan.file_count:
            log("[yellow]The remote folder is empty; nothing to download.[/yellow]")
            return False
        if plan.rejected:
            log(f"[yellow]Skipping {len(plan.rejected)} remote file(s) with unsupported names.[/yellow]")
        here = await asyncio.to_thread(files_already_here, local_dir, plan) if plan.exact else []
        log_plan(log, plan, here, describe)
        if here:
            plan = dl.leave_out(plan, here)
            if not plan.file_count:
                log(f"[green]{escape(what or label)}: all {len(here)} file(s) are already here; nothing to fetch.[/green]")
                return True
            log(f"  [cyan]{len(here)} file(s) are already here at the same size and are not fetched "
                f"again; fetching {named(plan.files)}.[/cyan]")
        _check(callbacks)
        callbacks.progress_start(
            f"{label} · {named(plan.files)}" if plan.exact else f"{label} · {plan.file_count} file(s)",
            plan.total_bytes,
        )
        result = await dl.download_tree(
            profile, remote_dir, staging=staging, plan=plan, on_progress=callbacks.progress_update, log=log)
    finally:
        callbacks.progress_end()

    if result.status != "complete":
        report_incomplete(log, result)
        return False

    stats = await asyncio.to_thread(merge, result.staged_root, local_dir, conflict_label=where)
    note = await asyncio.to_thread(after_merge) if after_merge is not None else None
    await asyncio.to_thread(dl.finalize, staging)
    fetched = named(plan.files) if plan.exact else f"{plan.file_count} file(s)"
    log(
        f"[green]Downloaded {fetched} to {escape(str(local_dir))} via {result.tool} "
        f"(copied {stats['copied']}, unchanged {stats['skipped']}, conflicts {stats['conflicted']})"
        + (f"; {len(here)} file(s) were already here" if here else "")
        + ".[/green]"
    )
    if note:
        log(note)
    return True


# ---- a session's part of its streams ----

@dataclass
class StreamCut:
    """what one stream gave a session."""
    stream: RecordedStream
    made: int = 0       # cuts made: here for 'local', staged on the capture host otherwise
    present: int = 0    # already here in full, not cut again
    failed: int = 0     # ffmpeg could not cut them
    listed: bool = True  # False: its capture host could not be asked


@dataclass
class SessionExport:
    """what export_session() did."""
    session_id: str
    folder: Path                                    # artifacts/<session>/streams/capture
    cuts: list[StreamCut] = field(default_factory=list)
    fetched: int = 0                                # cuts that arrived here this time
    present: int = 0                                # already here in full
    unfetched: list[str] = field(default_factory=list)  # hosts whose staged cuts did not arrive

    @property
    def here(self) -> int:
        return self.fetched + self.present


def _where(stream) -> str:
    return "this machine" if stream.ssh_profile == "local" else stream.ssh_profile


async def cut_stream(
    project_root,
    session_id: str,
    stream: RecordedStream,
    start: float,
    end: float,
    callbacks: ExportCallbacks | None = None,
    *,
    run: HostRunner | None = None,
) -> StreamCut:
    """cut one stream's recordings to the window (unix times) on its capture
    host: straight into artifacts/<session>/streams/capture/<host label>/<kind>/
    for 'local', into its staging there otherwise. A cut already here in full
    is not made again; a shorter copy (taken while the session went on) is
    replaced."""
    callbacks = callbacks or ExportCallbacks()
    run = run or run_on_host
    log = callbacks.log
    where = escape(_where(stream))
    name_markup = escape(stream.name)
    listing = await run(
        stream.ssh_profile, stream_cuts.list_script(stream.record_root, stream.host_label, stream.kind, stream.name),
        30.0)
    files = stream_cuts.parse_listing(listing or "", stream.name)
    if files is None:
        log(f"  [red]✗ {name_markup}: could not list its recordings on {where}.[/red]")
        return StreamCut(stream, listed=False)
    cuts = stream_cuts.cuts_for_window(files, start, end)
    if not cuts:
        log(f"  [dim]- {name_markup}: nothing recorded on {where} in that time[/dim]")
        return StreamCut(stream)

    here_dir = session_capture_streams_dir(project_root, session_id, stream.host_label) / stream.kind
    if stream.ssh_profile == "local":
        folder, quoted_dir = str(here_dir), False
    else:
        folder, quoted_dir = stream_cuts.staging_dir(stream.record_root, session_id, stream.host_label, stream.kind), True
    result = StreamCut(stream)
    for cut in cuts:
        _check(callbacks)
        if stream.kind == "video":
            probed = await run(stream.ssh_profile, stream_cuts.keyframe_script(cut), 60.0)
            moved = stream_cuts.on_keyframe(cut, probed or "")
            if moved is None:
                log(f"  [yellow]{name_markup}: ffprobe found no keyframe to start on (is it installed on {where}?); "
                    f"the cut may begin up to a second before the time in its name.[/yellow]")
            else:
                cut = moved
        name = stream_cuts.cut_name(stream.name, cut, stream.kind)
        # a cut is named after its first frame only: a copy made while the
        # session was still going has this name too, and its length tells
        here = here_dir / name
        covered = await asyncio.to_thread(copy_covers, here, cut.duration)
        if covered:
            result.present += 1
            log(f"  [dim]- {name_markup}: {escape(name)} is already here in full[/dim]")
            continue
        output = await run(
            stream.ssh_profile, stream_cuts.cut_script(cut, folder, stream.name, stream.kind, quoted_dir=quoted_dir),
            900.0)
        words = (output or "").split()
        if "CUT" in words:
            result.made += 1
            if covered is False and here.exists() and stream.ssh_profile != "local":
                # kept until the full cut has arrived: the fetch's merge puts it in place
                # (a local cut already went over it, whole)
                log(f"  [cyan]{name_markup}: the copy of {escape(name)} here stopped short; "
                    f"the full cut replaces it once it is here.[/cyan]")
            if "KEPT" in words:
                log(f"  [green]✓[/green] {name_markup}: {escape(name)} is still staged on {where} from an earlier "
                    f"export, whole; it is fetched as it is")
            else:
                log(f"  [green]✓[/green] {name_markup}: {cut.duration:.0f}s from {where} -> {escape(name)}")
        else:
            result.failed += 1
            detail = " ".join((output or "no answer").split())[-200:]
            log(f"  [red]✗ {name_markup}: ffmpeg could not cut {escape(name)} on {where}: {escape(detail)}[/red]")
    return result


def _note_in_manifest(project_root, session_id: str, host_label: str) -> str | None:
    """name the folders of a host's cuts as file sources in the session's
    manifest (audio for ASR, video for IPS and VFA), as a Collection download
    does for its recordings; a line for the log, None when there is nothing."""
    local_dir = session_capture_streams_dir(project_root, session_id, host_label)
    sources = {}
    for kind, pipelines in (("audio", ("asr",)), ("video", ("ips", "vfa"))):
        if (local_dir / kind).is_dir():
            for pipeline in pipelines:
                sources[pipeline] = {"host": host_label, "pipeline": STREAMS_DIR, "source": "file",
                                     "file_dir": str(local_dir / kind)}
    if not sources:
        return None
    manifest = update_session_manifest(project_root, session_id=session_id, file_sources=sources)
    return f"  [dim]Named them in the session manifest: {escape(str(manifest))}[/dim]"


def merge_replacing(source: Path, destination: Path, *, conflict_label: str) -> dict:
    """merge a fetched tree whose remote side is the whole file: a copy here of
    the same name but another size (a cut taken while the session was still
    going, a log fetched while its base still wrote it) gives way to the fetched
    one. Only a cut this copy did not cover is ever made, and a base only ever
    adds to its log. Done once every planned file has arrived, so a fetch that
    fails keeps the copy there was."""
    for root, _dirs, files in os.walk(source):
        for filename in files:
            if filename.endswith((".part", ".tmp")) or filename in METADATA_FILENAMES:
                continue
            fetched = Path(root) / filename
            here = destination / fetched.relative_to(source)
            try:
                if here.is_file() and here.stat().st_size != fetched.stat().st_size:
                    here.unlink()
            except OSError:
                continue
    return merge_tree(source, destination, conflict_label=conflict_label)


async def fetch_session_cuts(
    project_root,
    session_id: str,
    ssh_profile: str,
    record_root: str,
    host_label: str,
    callbacks: ExportCallbacks | None = None,
    *,
    run: HostRunner | None = None,
    fetch: Callable[..., Awaitable[bool]] | None = None,
    expected: bool = True,
) -> bool:
    """fetch the cuts of a session staged on a capture host into
    artifacts/<session>/streams/capture/<host label>/, then remove that staging
    there (a transfer that did not finish keeps it, so the next run resumes
    instead of cutting anew). True when they are all here. `expected`: this run
    cut something there; when not, a host with nothing staged is no error (an
    earlier export may still have left cuts there, and they are fetched)."""
    callbacks = callbacks or ExportCallbacks()
    run = run or run_on_host
    fetch = fetch or fetch_tree
    log = callbacks.log
    profile = get_profile_by_name(ssh_profile)
    if profile is None:
        log(f"[red]SSH profile '{escape(ssh_profile)}' not found: the cuts staged there stay until it is back.[/red]")
        return False
    quoted = stream_cuts.staging_dir(record_root, session_id, host_label)
    answer = await run(ssh_profile, stream_cuts.resolve_script(quoted), 30.0)
    remote_dir = stream_cuts.parse_resolved(answer)
    if remote_dir is None:
        if "RESOLVED" in str(answer or "").split() and not expected:
            return True  # nothing was cut there, now or by an earlier export
        log(f"[red]Could not find the cuts staged on {escape(ssh_profile)} ({escape(quoted)}).[/red]")
        return False
    _check(callbacks)
    local_dir = session_capture_streams_dir(project_root, session_id, host_label)
    staging = dl.staging_root(project_root, session_id, STREAMS_DIR, CAPTURE_STREAMS_DIR, host_label)
    fetched = await fetch(
        profile, remote_dir, local_dir, staging=staging, label=host_label, where=ssh_profile, what=session_id,
        callbacks=callbacks, merge=merge_replacing,
        after_merge=lambda: _note_in_manifest(project_root, session_id, host_label))
    if fetched:
        # only what was staged for this session and host
        await run(ssh_profile, stream_cuts.cleanup_script(record_root, session_id, host_label), 30.0)
    return fetched


async def export_session(
    project_root,
    session_id: str,
    streams: list[RecordedStream],
    start: float,
    end: float,
    callbacks: ExportCallbacks | None = None,
    *,
    run: HostRunner | None = None,
    fetch: Callable[..., Awaitable[bool]] | None = None,
) -> SessionExport:
    """a session's part (start to end, unix times) of the recordings of these
    streams, into artifacts/<session>/streams/capture/<host label>/<video|audio>/:
    every stream is cut on its capture host, then each remote host's cuts are
    fetched at once. `run` (default run_on_host) runs a script where a stream
    records; `fetch` (default fetch_tree) is the transfer. Hosts that cannot be
    asked are logged and left out, never raised."""
    callbacks = callbacks or ExportCallbacks()
    run = run or run_on_host
    fetch = fetch or fetch_tree
    session_id = safe_segment(session_id, "session")
    result = SessionExport(session_id, session_capture_streams_dir(project_root, session_id))
    # every capture host a remote stream records on: what an earlier export
    # staged there and could not fetch is fetched too, even when nothing is cut now
    staged: dict[tuple[str, str, str], int] = {}
    here_labels: list[str] = []
    for stream in streams:
        _check(callbacks)
        cut = await cut_stream(project_root, session_id, stream, start, end, callbacks, run=run)
        result.cuts.append(cut)
        result.present += cut.present
        if stream.ssh_profile == "local":
            result.fetched += cut.made
            if cut.made and stream.host_label not in here_labels:
                here_labels.append(stream.host_label)
        elif cut.listed:
            group = (stream.ssh_profile, stream.record_root, stream.host_label)
            staged[group] = staged.get(group, 0) + cut.made
    for host_label in here_labels:
        note = await asyncio.to_thread(_note_in_manifest, project_root, session_id, host_label)
        if note:
            callbacks.log(note)
    for (ssh_profile, record_root, host_label), count in staged.items():
        _check(callbacks)
        if await fetch_session_cuts(project_root, session_id, ssh_profile, record_root, host_label, callbacks,
                                    run=run, fetch=fetch, expected=count > 0):
            result.fetched += count
        else:
            result.unfetched.append(ssh_profile)
    return result

