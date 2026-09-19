"""the files a session's bases leave on the machines they run on, taken off
those machines: Sessions -> Export Base Files.

A base, and a synchronizer or the IPS visualizer beside it, keeps what it
writes for a session in the checkout it runs from, under
artifacts/<session>/pipelines/<pipeline>/<host>/ (<host> is the machine's
short name; artifact_paths.pipeline_section_dir): its logs (logger/), the
config it ran with (config/), what it recorded (real-time/runtime/: an ASR
base's speech segments with Store Audio on and the speaker profiles it
recognized, the frames VFA and IPS keep) and the IPS visualizer's plots
(visualizations/). A process run on this machine writes into this console's
artifacts/ in the first place; one run over SSH leaves them on its host.

This asks every SSH profile at once what it holds of the session (one round
trip each), fetches it folder by folder into the same place here with the
staged, resumable transfer of stream_export.fetch_tree, and names each host's
folders in the session's manifest. real-time/temp/, frames on their way to a
server, stays where it is. The path is the session's own, so whatever is there
is the session's: unlike its streams, nothing in its record is needed to know
what to take. The hosts its bases noted they ran on (its sources) only tell
which of them no profile reached.

Nothing here draws: whoever runs it hands in stream_export.ExportCallbacks."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from rich.markup import escape

from openmmla.tui import download as dl
from openmmla.tui import stream_export
from openmmla.tui.artifacts import update_pipeline_manifest
from openmmla.utils import session_sources
from openmmla.utils.artifact_paths import ARTIFACTS_DIR, safe_segment, short_hostname

PIPELINES_DIR = "pipelines"

# the folders of a host's part that are taken, below <pipeline>/<host>/;
# real-time/profiles/ and post-time/ are where older bases wrote
FOLDERS = ("real-time/runtime", "real-time/profiles", "post-time", "logger", "config", "visualizations")

# seconds a host has to say what it holds; all are asked at once
ASK_TIMEOUT = 20.0

# a <pipeline>/<host>/<folder> of the listing, named as the bases name them (safe_segment)
_FOLDER_LINE = re.compile(r"^([A-Za-z0-9_.:-]+)/([A-Za-z0-9_.:-]+)/(.+)$")


def session_pipelines_dir(project_root, session_id: str) -> Path:
    """artifacts/<session>/pipelines/ of a checkout."""
    return Path(project_root) / ARTIFACTS_DIR / safe_segment(session_id, "session") / PIPELINES_DIR


# ---- what a host holds ----

def list_script(project_path: str | None, session_id: str) -> str:
    """print what a host holds of a session: ROOT <absolute
    artifacts/<session>/pipelines> when it is there, DIR <pipeline>/<host>/<folder>
    for each of FOLDERS with a file the transfer would take, HOST <the machine's
    short name>, and LISTED when done. `project_path`: the checkout there
    (~ spelled either way)."""
    path = (f"{str(project_path or '~/OpenMMLA').rstrip('/')}/{ARTIFACTS_DIR}/"
            f"{safe_segment(session_id, 'session')}/{PIPELINES_DIR}")
    return (
        f"if cd {dl.quote_remote_path(path)} 2>/dev/null; then "
        'echo "ROOT $(pwd -P)"; '
        "for part in */*/; do "
        f"for folder in {' '.join(FOLDERS)}; do "
        'if [ -n "$(find "$part$folder" -type f ! -name .DS_Store ! -name .manifest.lock '
        "! -name '*.tmp' ! -name '*.part' 2>/dev/null | head -n 1)\" ]; then "
        'echo "DIR $part$folder"; fi; '
        "done; done; fi; "
        'echo "HOST $(hostname -s 2>/dev/null || hostname)"; echo LISTED'
    )


@dataclass
class HostFiles:
    """what one machine holds of a session, as list_script() printed it."""
    host: str = ""                                    # its short name
    root: str | None = None                           # its absolute artifacts/<session>/pipelines
    folders: list[str] = field(default_factory=list)  # <pipeline>/<host>/<folder>, each with a file

    def parts(self) -> dict[str, list[str]]:
        """<pipeline>/<host> -> its folders, in the order they were listed."""
        found: dict[str, list[str]] = {}
        for folder in self.folders:
            pipeline, host, rest = _FOLDER_LINE.match(folder).groups()
            found.setdefault(f"{pipeline}/{host}", []).append(rest)
        return found


def parse_listing(text: str | None) -> HostFiles | None:
    """what list_script() printed; None when the host could not be asked."""
    lines = [line.strip() for line in str(text or "").splitlines()]
    if "LISTED" not in lines:
        return None
    found = HostFiles()
    for line in lines:
        if line.startswith("HOST "):
            found.host = line[5:].strip()
        elif line.startswith("ROOT /"):
            found.root = line[5:].strip().rstrip("/") or "/"
        elif line.startswith("DIR "):
            folder = line[4:].strip()
            match = _FOLDER_LINE.match(folder)
            if match and match.group(3) in FOLDERS and folder not in found.folders:
                found.folders.append(folder)
    if found.root is None:
        found.folders = []
    return found


def noted_hosts(record: dict | None) -> dict[str, list[str]]:
    """the machines the session's bases noted they ran on (its sources), each
    with the keys of those bases."""
    found: dict[str, list[str]] = {}
    for entry in session_sources.session_sources(record):
        host = str(entry.get("host") or "").strip()
        if host:
            found.setdefault(host, []).append(str(entry.get("key") or entry.get("pipeline") or "?"))
    return found


def _machine(name: str | None) -> str:
    return safe_segment(name, "").lower()


def local_parts(project_root, session_id: str) -> list[str]:
    """the <pipeline>/<host> folders of this machine's own processes, which
    wrote them here in the first place."""
    here = session_pipelines_dir(project_root, session_id)
    me = _machine(short_hostname())
    try:
        return sorted(f"{pipeline.name}/{host.name}" for pipeline in here.iterdir() if pipeline.is_dir()
                      for host in pipeline.iterdir() if host.is_dir() and _machine(host.name) == me)
    except OSError:
        return []


# ---- taking it ----

def _by_count(_rels) -> str:
    """base files have no kinds worth sorting them by: fetch_tree counts them."""
    return ""


@dataclass
class BaseFilesExport:
    """what export_session() did."""
    session_id: str
    folder: Path                                                 # artifacts/<session>/pipelines here
    fetched: list[tuple[str, str]] = field(default_factory=list)  # (profile, <pipeline>/<host>) all here now
    incomplete: list[str] = field(default_factory=list)          # profiles whose folders did not all arrive
    unasked: list[str] = field(default_factory=list)             # profiles that could not be asked
    missing: dict[str, list[str]] = field(default_factory=dict)  # noted host -> its bases, reached by none


async def export_session(
    project_root,
    session_id: str,
    profiles,
    callbacks: stream_export.ExportCallbacks | None = None,
    *,
    record: dict | None = None,
    run: stream_export.HostRunner | None = None,
    fetch: Callable[..., Awaitable[bool]] | None = None,
) -> BaseFilesExport:
    """take a session's base files off the hosts of these SSH profiles into
    artifacts/<session>/pipelines/<pipeline>/<host>/ here. `record`: the
    session's document, whose sources name the hosts its bases ran on (one no
    profile reached is reported). `run` (default stream_export.run_on_host)
    runs the listing on a host; `fetch` (default stream_export.fetch_tree) is
    the transfer. Hosts that cannot be asked are logged and left out, never
    raised."""
    callbacks = callbacks or stream_export.ExportCallbacks()
    run = run or stream_export.run_on_host
    fetch = fetch or stream_export.fetch_tree
    log = callbacks.log
    session_id = safe_segment(session_id, "session")
    here = session_pipelines_dir(project_root, session_id)
    result = BaseFilesExport(session_id, here)
    profiles = list(profiles)

    callbacks.progress_start(f"Base files · asking {len(profiles)} host(s)", None)
    try:
        answers = await asyncio.gather(*(
            run(profile.name, list_script(profile.remote_project_path, session_id), ASK_TIMEOUT)
            for profile in profiles))
    finally:
        callbacks.progress_end()

    this_folder = os.path.realpath(here)
    reached = {_machine(short_hostname())}
    # (ssh destination, folder there) -> the profile that answered for it first
    seen: dict[tuple[str, str], str] = {}
    for profile, answer in zip(profiles, answers):
        name = escape(profile.name)
        listing = parse_listing(answer)
        if listing is None:
            result.unasked.append(profile.name)
            log(f"  [dim]- {name}: could not be asked (offline, or no bash there); whatever it holds stays "
                f"there[/dim]")
            continue
        reached.add(_machine(listing.host))
        if not listing.folders:
            log(f"  [dim]- {name}: nothing of this session[/dim]")
            continue
        if listing.root == this_folder:
            log(f"  [dim]- {name}: reaches this machine's own folder, where they are already[/dim]")
            continue
        key = (f"{profile.ssh_destination()}:{profile.port}", listing.root)
        if key in seen:
            log(f"  [dim]- {name}: the same folder as {escape(seen[key])}[/dim]")
            continue
        seen[key] = profile.name
        complete = True
        for part, folders in listing.parts().items():
            if await _fetch_part(project_root, session_id, profile, listing.root, part, folders, callbacks, fetch):
                result.fetched.append((profile.name, part))
            else:
                complete = False
        if not complete:
            result.incomplete.append(profile.name)

    result.missing = {host: keys for host, keys in noted_hosts(record).items() if _machine(host) not in reached}
    return result


async def _fetch_part(project_root, session_id: str, profile, root: str, part: str, folders: list[str],
                      callbacks: stream_export.ExportCallbacks, fetch) -> bool:
    """fetch the folders of one <pipeline>/<host> part and name what arrived in
    the session's manifest; True when all of them are here."""
    log = callbacks.log
    log(f"[cyan]{escape(profile.name)}: {escape(part)} ({escape(', '.join(folders))})[/cyan]")
    here = session_pipelines_dir(project_root, session_id) / part
    arrived: list[str] = []
    for folder in folders:
        if callbacks.cancelled():
            raise asyncio.CancelledError()
        if await fetch(
            profile, f"{root}/{part}/{folder}", here / folder,
            staging=dl.staging_root(project_root, session_id, PIPELINES_DIR, *part.split("/"), *folder.split("/")),
            label=f"{profile.name} · {folder}", where=profile.name, what=f"{part}/{folder}",
            callbacks=callbacks, merge=stream_export.merge_replacing, describe=_by_count,
        ):
            arrived.append(folder)
    if arrived:
        pipeline, host = part.split("/")
        manifest = await asyncio.to_thread(
            update_pipeline_manifest, project_root, session_id=session_id, pipeline_name=pipeline,
            host_name=host, remote_root=f"{root}/{part}", local_path=here, downloaded_paths=arrived)
        log(f"  [dim]Named them in the session manifest: {escape(str(manifest))}[/dim]")
    return len(arrived) == len(folders)
