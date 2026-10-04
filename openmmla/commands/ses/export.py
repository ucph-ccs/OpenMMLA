"""mmla ses-export: everything of a session gathered onto this console.

A session leaves its data on several machines: its measurements in InfluxDB,
its Collection recordings on the hosts that recorded them, its part of the
streams on the Stream Server and on the hosts that captured them, and what its
bases wrote on the machines they ran on. Export (the Sessions tab's button, or
this command) gathers all of it into artifacts/<session>/ of this console's
checkout, in the layout each part has always had there:

  measurements  one JSON file per event type from InfluxDB, a text transcript
                and <session>_parameters.json, in measurements/
  collection    each recording host's folder, in collection/<host label>/,
                with the transfer of the Collection card's Download
  streams       the Stream Server's cut, over its playback server, in
                streams/server/, and the cuts made on the capture hosts, in
                streams/capture/<host label>/ (stream_export)
  base          what the bases and synchronizers wrote on the machines they
                ran on, in pipelines/<pipeline>/<host>/ (base_files)

The recording hosts are the ones Collection Start noted in the session's
MongoDB document (collection_hosts, each recorder under collection_recorders),
else the hosts of the recordings the session's manifest here names, else every
SSH profile, asked whether it holds artifacts/<session>/collection/.

A host that is offline, or no longer holds a file, is named in the log, and
the export goes on with the others; what the archive host (the System Settings
host, as mmla ses-archive picks it) holds of that part under
artifacts/<session>/ is fetched from there instead: only what did not arrive
from its origin, and beside a different copy here, never over it (the archive
may hold the older one). A file already here at the same size is not fetched
again (and, when its source states a sha256, as the archive host's ledger
does, only with that sha256), so a second export fetches only what is
missing. The end says per part what was fetched, what was here already and
what is missing and why; the exit status is 1 only when something that should
exist is not here and could not be fetched from anywhere.

Test hooks: OPENMMLA_EXPORT_LOCAL_ROOT is the project root whose artifacts/
receives the session (default this checkout), OPENMMLA_EXPORT_REMOTE_ROOT the
project root a session's folders are looked for under on every SSH host
(default its home for the Collection recordings, its profile's
remote_project_path for the base files); the archive host's root is
ses-archive's OPENMMLA_ARCHIVE_REMOTE_ROOT. The log says so, in yellow,
whenever one of them is set."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import glob
import hashlib
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ENV_LOCAL_ROOT = "OPENMMLA_EXPORT_LOCAL_ROOT"
ENV_REMOTE_ROOT = "OPENMMLA_EXPORT_REMOTE_ROOT"

# the parts of an export, in the order they run
CATEGORIES = ("measurements", "collection", "streams", "base")
TITLES = {"measurements": "Measurements", "collection": "Collection", "streams": "Streams", "base": "Base files"}

# the session document's note of where Collection Start recorded: the hosts by
# name, and each recorder with its host's folder and its device
COLLECTION_HOSTS_FIELD = "collection_hosts"
COLLECTION_RECORDERS_FIELD = "collection_recorders"

# seconds a host has to say what it holds; all are asked at once
ASK_TIMEOUT = 20.0

# the event types of InfluxDB, as each is named in the log and in its file
_EVENT_FILES = (
    ("ASR Recognition", "EVENT_TYPE_ASR_RECOGNITION", "speaker_recognition"),
    ("ASR Transcription", "EVENT_TYPE_ASR_TRANSCRIPTION", "speaker_transcription"),
    ("VFA Action", "EVENT_TYPE_VFA_ACTION", "action_recognition"),
    ("VFA Features", "EVENT_TYPE_VFA_FEATURES", "features"),
    ("IPS Translation", "EVENT_TYPE_IPS_TRANSLATION", "badge_translation"),
    ("IPS Rotation", "EVENT_TYPE_IPS_ROTATION", "badge_rotation"),
    ("IPS Relation", "EVENT_TYPE_IPS_RELATION", "badge_relation"),
)

# what the end of a session's window is (recordings.session_end), as the log says it
_WINDOW_REASONS = {
    "ended": "when it was ended",
    "left": "never ended: when its last base left",
    "running": "still running: up to now",
}

# the Stream Server's retention when the caller does not know it: asked of the
# server itself, once it is needed to tell footage deleted from never recorded
ASK = object()

# what `find` passes over when it looks for a file of the session: nothing a
# transfer takes (merge_tree and the transfer skip them too)
_FILE_IGNORED = ("! -name .DS_Store ! -name .manifest.lock ! -name '*.tmp' ! -name '*.part'")


def _escape(text) -> str:
    from rich.markup import escape

    return escape(str(text))


# ---- what an export did ----

@dataclass
class Missing:
    """something of a session that is not here after the export."""
    what: str
    why: str
    counted: bool = True   # it should exist: the exit status says so


@dataclass
class PartResult:
    """what one part of the export did (in a dry run: would do)."""
    name: str
    fetched: int = 0                                         # files fetched or written this time
    fetched_bytes: int = 0
    from_archive: int = 0                                    # of those, from the archive host
    present: int = 0                                         # here already, not fetched again
    missing: list[Missing] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(item.counted for item in self.missing)


@dataclass
class ExportResult:
    session_id: str
    dry_run: bool = False
    parts: dict[str, PartResult] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 1 if any(part.failed for part in self.parts.values()) else 0


# ---- where it goes ----

def _settings_root():
    from openmmla.tui.schema.loader import _find_project_root

    return _find_project_root()


def local_project_root():
    """the project root whose artifacts/ receives the session:
    OPENMMLA_EXPORT_LOCAL_ROOT, else this checkout."""
    configured = os.environ.get(ENV_LOCAL_ROOT, "").strip()
    return os.path.abspath(os.path.expanduser(configured)) if configured else _settings_root()


def hook_notes() -> list[str]:
    """a log line (markup) for each test hook that moves the export: the
    environment is inherited, and a stray variable would fill or ask another
    folder without a word."""
    from openmmla.commands.ses import archive

    notes = []
    for name, what in ((ENV_LOCAL_ROOT, "the session is gathered under {} here, not this checkout"),
                       (ENV_REMOTE_ROOT, "every SSH host is asked under {}, not its own folders"),
                       (archive.ENV_REMOTE_ROOT, "the archive host is asked under {}, not its checkout")):
        value = os.environ.get(name, "").strip()
        if value:
            notes.append(f"  [yellow]{name} is set: {what.format(_escape(value))}.[/yellow]")
    return notes


def _remote_root() -> str:
    return os.environ.get(ENV_REMOTE_ROOT, "").strip()


def _profiles(remote_root: str = "") -> list:
    """the SSH profiles, each pointing at `remote_root` instead of its own
    checkout when the test hook sets one."""
    from openmmla.tui.ssh import load_ssh_profiles

    profiles = load_ssh_profiles()
    if remote_root:
        profiles = [dataclasses.replace(profile, remote_project_path=remote_root) for profile in profiles]
    return profiles


def _shown_path(project_root, path) -> str:
    """a folder of the project as the log names it, artifacts/<session>/...,
    escaped for markup."""
    from openmmla.tui.artifacts import relative_to_root

    return _escape(relative_to_root(project_root, path))


def _media(rels) -> list[str]:
    """the paths of a transfer that are counted: not the manifests and
    configs, which are fetched every time."""
    from openmmla.tui.artifacts import METADATA_FILENAMES

    return [rel for rel in rels if os.path.basename(rel) not in METADATA_FILENAMES]


def _files_in(folder: Path) -> list[Path]:
    """the files of a folder here that count as its content."""
    found = []
    for path in Path(folder).rglob("*"):
        name = path.name
        if (path.is_file() and not name.startswith(".") and not name.endswith((".tmp", ".part"))
                and ".staging" not in path.parts):
            found.append(path)
    return found


def _merge_counts(source: Path, stats: dict) -> tuple[int, int]:
    """(files copied, files here already) of a merge_tree of `source` on this
    machine, the manifests and configs it copies every time left out."""
    files = [str(path) for path in _files_in(source)]
    meta = len(files) - len(_media(files))
    return max(0, stats.get("copied", 0) + stats.get("conflicted", 0) - meta), stats.get("skipped", 0)


# ---- skipping what is here already ----

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def present_files(local_dir, plan, digests: dict | None = None, conflict_label: str = "") -> list[str]:
    """the planned files already here: at the remote size (as
    stream_export.files_already_here), and with the sha256 the source states
    for one, when it states one (the archive host's ledger). The copy a
    transfer from `conflict_label` kept beside a different file here
    (<stem>_<label><ext>, as merge_tree keeps both) counts as well, so it is
    not fetched again and again."""
    from openmmla.tui import stream_export

    if not digests and not conflict_label:
        return stream_export.files_already_here(local_dir, plan)
    digests = digests or {}
    here = []
    for item in plan.files:
        if not _media([item.rel]):
            continue
        path = Path(local_dir) / item.rel
        want = digests.get(item.rel)
        candidates = [path]
        if conflict_label:
            from openmmla.utils.artifact_paths import safe_segment

            label = glob.escape(safe_segment(conflict_label, "remote"))
            candidates += sorted(path.parent.glob(f"{glob.escape(path.stem)}_{label}*{glob.escape(path.suffix)}"))
        for candidate in candidates:
            try:
                if candidate.stat().st_size != item.size:
                    continue
                if not want or _sha256(candidate) == want:
                    here.append(item.rel)
                    break
            except OSError:
                continue
    return here


def _same_content(left: Path, right: Path) -> bool:
    try:
        return left.stat().st_size == right.stat().st_size and _sha256(left) == _sha256(right)
    except OSError:
        return False


def _merge_archived(source: Path, destination: Path, *, conflict_label: str) -> dict:
    """merge_tree for what comes from the archive host, which holds what this
    console had when Archive was pressed and may be older than the copy here
    (a log a base still wrote, a cut taken while the session went on): a file
    here is never replaced, the archived one goes beside it as
    <name>_<archive host><ext> (as present_files counts it), and only once."""
    from openmmla.tui.artifacts import METADATA_FILENAMES, merge_tree
    from openmmla.utils.artifact_paths import safe_segment

    stats = {"copied": 0, "skipped": 0, "conflicted": 0}
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        return merge_tree(source, destination, conflict_label=conflict_label)
    label = glob.escape(safe_segment(conflict_label, "remote"))
    for path in sorted(source.rglob("*")):
        name = path.name
        if (not path.is_file() or name in (".DS_Store", ".manifest.lock") or name.endswith((".tmp", ".part"))
                or "__pycache__" in path.relative_to(source).parts):
            continue
        here = destination / path.relative_to(source)
        if here.is_file() and name not in METADATA_FILENAMES and not _same_content(path, here):
            beside = here.parent.glob(f"{glob.escape(here.stem)}_{label}*{glob.escape(here.suffix)}")
            if any(_same_content(path, copy) for copy in beside):
                stats["skipped"] += 1  # kept beside it by an earlier export
                continue
        for key, value in merge_tree(path, here, conflict_label=conflict_label).items():
            stats[key] += value
    return stats


# a cut or clip exported before is named after its first frame, which a video
# cut may move up to a second before the window (onto a keyframe)
_HERE_SLACK_SECONDS = 2.0


def _exported_here(folder: Path, name: str, suffix: str, start: float, end: float) -> list[Path]:
    """the files here of a stream's part of a window (unix times), as an
    earlier export named them: <name>_<start><suffix>, <start> within it."""
    found = []
    try:
        candidates = sorted(Path(folder).glob(f"{glob.escape(name)}_*{suffix}"))
    except OSError:
        return []
    for path in candidates:
        try:
            began = float(path.name[len(name) + 1:len(path.name) - len(suffix)])
        except ValueError:
            continue  # another stream's (cam_x for cam), or a copy kept beside one
        if start - _HERE_SLACK_SECONDS <= began < end and path.is_file():
            found.append(path)
    return found


class _Counted:
    """stream_export.fetch_tree, counting into a part what each transfer found
    here already and what it fetched; the shape base_files and stream_export
    take for their `fetch`."""

    def __init__(self, part: PartResult, digests: dict | None = None, archived: bool = False) -> None:
        self.part = part
        self.digests = digests
        self.archived = archived

    async def __call__(self, profile, remote_dir, local_dir, **kwargs):
        from openmmla.tui import stream_export

        counts = {}

        def present(here_dir, plan):
            here = present_files(here_dir, plan, self.digests, kwargs.get("where") or "")
            wanted = [item for item in plan.files if _media([item.rel]) and item.rel not in set(here)]
            counts.update(present=len(here), fetched=len(wanted), bytes=sum(item.size for item in wanted))
            return here

        ok = await stream_export.fetch_tree(profile, remote_dir, local_dir, present=present, **kwargs)
        self.part.present += counts.get("present", 0)
        if ok:
            self.part.fetched += counts.get("fetched", 0)
            self.part.fetched_bytes += counts.get("bytes", 0)
            if self.archived:
                self.part.from_archive += counts.get("fetched", 0)
        return ok


async def _plan_counts(profile, remote_dir: str, local_dir: Path, digests: dict | None = None,
                       label: str = "") -> tuple[int, int, int] | None:
    """what a transfer would fetch, for a dry run: (files, bytes, here
    already), asked over SSH and changing nothing; None when the folder could
    not be asked or is not there."""
    from openmmla.tui import download as dl

    plan = await dl.probe_remote(profile, remote_dir)
    if plan.unreachable or not plan.exists:
        return None
    here = await asyncio.to_thread(present_files, local_dir, plan, digests, label) if plan.exact else []
    wanted = [item for item in plan.files if _media([item.rel]) and item.rel not in set(here)]
    return len(wanted), sum(item.size for item in wanted), len(here)


# ---- the archive host, when a part is not where it came from ----

class _Archive:
    """the archive host's copy of the session (mmla ses-archive): looked up at
    the first part that needs it, then kept."""

    def __init__(self, session_id: str, root, settings_root, callbacks, dry_run: bool) -> None:
        self.session_id = session_id
        self.root = root
        self.settings_root = settings_root
        self.callbacks = callbacks
        self.dry_run = dry_run
        self.target = None
        self.why = ""                       # why there is none to ask
        self.session_dir = ""               # the session's folder there, absolute
        self.digests: dict[str, str] = {}   # path below it -> sha256, from its ledger
        self.holds: dict[str, bool] = {}
        self.failed: dict[str, str] = {}    # folder below it -> why its transfer did not complete
        self._resolved = False

    @property
    def name(self) -> str:
        return self.target.name if self.target is not None else "the archive host"

    def _runner(self) -> str:
        return "local" if self.target.here else self.target.profile.name

    async def _resolve(self) -> bool:
        from openmmla.commands.ses import archive

        if not self._resolved:
            self._resolved = True
            try:
                self.target = await asyncio.to_thread(archive.resolve_target, None, self.settings_root)
            except archive.ArchiveError as error:
                self.why = f"no archive host to ask ({error})"
        return self.target is not None

    async def ask(self, rels: list[str]) -> dict[str, bool]:
        """which of these folders below the session's folder the archive host
        holds a file of (one round trip for those not asked yet)."""
        from openmmla.tui import download as dl
        from openmmla.tui import stream_export
        from openmmla.utils.artifact_paths import safe_segment

        rels = [rel for rel in dict.fromkeys(rels) if rel]
        if not await self._resolve():
            return {rel: False for rel in rels}
        new = [rel for rel in rels if rel not in self.holds]
        if new:
            session_path = self.target.session_path(safe_segment(self.session_id, "session"))
            checks = " ".join(shlex.quote(rel) for rel in new)
            script = (
                f"if cd {dl.quote_remote_path(session_path)} 2>/dev/null; then echo \"ROOT $(pwd -P)\"; "
                f"for rel in {checks}; do if [ -d \"$rel\" ] && "
                f"[ -n \"$(find \"$rel\" -type f {_FILE_IGNORED} 2>/dev/null | head -n 1)\" ]; then "
                "echo \"HAS $rel\"; fi; done; "
                "if [ -f .archive/ledger.json ]; then echo LEDGER-BEGIN; cat .archive/ledger.json; echo; "
                "echo LEDGER-END; fi; fi; echo ASKED"
            )
            self.callbacks.progress_start(f"Archive · asking {self.name}", None)
            try:
                answer = await stream_export.run_on_host(self._runner(), script, ASK_TIMEOUT)
            finally:
                self.callbacks.progress_end()
            lines = str(answer or "").splitlines()
            if "ASKED" not in [line.strip() for line in lines]:
                self.why = f"{self.name} did not answer"
                for rel in new:
                    self.holds[rel] = False
            else:
                found, ledger, inside = set(), [], False
                for line in lines:
                    text = line.rstrip("\r")
                    if text == "LEDGER-BEGIN":
                        inside = True
                    elif text == "LEDGER-END":
                        inside = False
                    elif inside:
                        ledger.append(text)
                    elif text.startswith("ROOT /"):
                        self.session_dir = text[5:].strip()
                    elif text.startswith("HAS "):
                        found.add(text[4:].strip())
                with contextlib.suppress(ValueError, TypeError, AttributeError):
                    self.digests = {rel: str(entry.get("sha256") or "")
                                    for rel, entry in json.loads("\n".join(ledger) or "{}").items()
                                    if isinstance(entry, dict) and entry.get("sha256")}
                same = (self.target.here and bool(self.session_dir) and os.path.realpath(self.session_dir)
                        == os.path.realpath(Path(self.root) / "artifacts" / safe_segment(self.session_id, "session")))
                if same:
                    self.why = f"{self.name} keeps it in this very folder"
                elif not self.session_dir:
                    self.why = f"{self.name} holds nothing of the session"
                for rel in new:
                    self.holds[rel] = rel in found and not same
        return {rel: self.holds.get(rel, False) for rel in rels}

    def digests_below(self, rel: str) -> dict[str, str]:
        prefix = rel.rstrip("/") + "/"
        return {key[len(prefix):]: value for key, value in self.digests.items() if key.startswith(prefix)}

    async def subfolders(self, rel: str) -> list[str]:
        """the folders right below <session>/<rel> on the archive host that
        hold a file (one round trip); none when it holds nothing there or
        could not be asked."""
        from openmmla.tui import stream_export
        from openmmla.utils.artifact_paths import safe_segment

        if not (await self.ask([rel])).get(rel):
            return []
        script = (
            f"if cd {shlex.quote(f'{self.session_dir}/{rel}')} 2>/dev/null; then for sub in */; do "
            "sub=\"${sub%/}\"; "
            f"if [ -d \"$sub\" ] && [ -n \"$(find \"$sub\" -type f {_FILE_IGNORED} 2>/dev/null | head -n 1)\" ]; "
            "then echo \"SUB $sub\"; fi; done; fi; echo LISTED")
        self.callbacks.progress_start(f"Archive · asking {self.name}", None)
        try:
            answer = await stream_export.run_on_host(self._runner(), script, ASK_TIMEOUT)
        finally:
            self.callbacks.progress_end()
        lines = [line.rstrip("\r") for line in str(answer or "").splitlines()]
        if "LISTED" not in [line.strip() for line in lines]:
            return []
        names = [line[4:].strip() for line in lines if line.startswith("SUB ")]
        return [name for name in dict.fromkeys(names) if name and safe_segment(name, "") == name]

    async def fetch(self, rel: str, local_dir: Path, part: PartResult, *, after_merge=None) -> bool:
        """the archive host's <session>/<rel> into local_dir: only what is not
        here yet (same size, same sha256 as its ledger says), merged beside a
        different copy here, never over it (_merge_archived). True when it is
        all here afterwards."""
        from openmmla.tui import download as dl

        if not (await self.ask([rel])).get(rel):
            return False
        remote = f"{self.session_dir}/{rel}"
        log = self.callbacks.log
        if self.dry_run:
            if self.target.here:
                counts = (len(_files_in(Path(remote))), 0, 0)
            else:
                counts = await _plan_counts(self.target.profile, remote, local_dir, self.digests_below(rel),
                                            self.name)
            if counts is None:
                return False
            part.fetched += counts[0]
            part.fetched_bytes += counts[1]
            part.from_archive += counts[0]
            part.present += counts[2]
            log(f"  [cyan]Would fetch {counts[0]} file(s) of {_escape(rel)}/ from the archive host "
                f"{_escape(self.name)}[/cyan]")
            return True
        log(f"[cyan]From the archive host {_escape(self.name)}: {_escape(rel)}/[/cyan]")
        if self.target.here:
            stats = await asyncio.to_thread(_merge_archived, Path(remote), local_dir,
                                            conflict_label=self.name)
            copied, skipped = await asyncio.to_thread(_merge_counts, Path(remote), stats)
            part.fetched += copied
            part.from_archive += copied
            part.present += skipped
            if after_merge is not None:
                note = await asyncio.to_thread(after_merge)
                if note:
                    log(note)
            log(f"  [green]Copied {copied} file(s) of {_escape(rel)}/ from this machine's archive "
                f"({skipped} here already)[/green]")
            return True
        fetch = _Counted(part, self.digests_below(rel), archived=True)
        ok = await fetch(
            self.target.profile, remote, local_dir,
            staging=dl.staging_root(self.root, self.session_id, "archive", *rel.split("/")),
            label=f"Archive · {rel.split('/')[-1]}", where=self.name, what=f"{rel} (archived)",
            callbacks=self.callbacks, merge=_merge_archived, after_merge=after_merge)
        if not ok:
            self.failed[rel] = f"its copy on {self.name} did not all arrive (Export again to go on)"
        return ok

    def none_note(self, rel: str = "") -> str:
        """why the archive host had nothing to give, for the summary."""
        return self.failed.get(rel) or self.why or f"{self.name} does not hold it either"


# ---- measurements ----

def _digest_or_none(path: str) -> str | None:
    try:
        return _sha256(Path(path))
    except OSError:
        return None


async def export_measurements(root, session_id: str, influx, record: dict | None, callbacks, *,
                              dry_run: bool = False, database_source: dict | None = None) -> PartResult:
    """one JSON file per event type from InfluxDB into
    artifacts/<session>/measurements/ (openmmla.utils.querys), a text
    transcript, <session>_parameters.json from the session's document
    (openmmla.utils.session_provenance), and the session manifest's
    measurements_path. A file that comes out as it was is counted as here
    already."""
    from openmmla.tui.artifacts import artifact_session_dir, ensure_session_layout
    from openmmla.utils import constants

    part = PartResult("measurements")
    log = callbacks.log
    session_dir = artifact_session_dir(root, session_id)
    measurements_dir = os.path.join(session_dir, "measurements")
    if influx is None:
        kept = [name for name in (os.listdir(measurements_dir) if os.path.isdir(measurements_dir) else [])
                if name.endswith(".json") and not name.endswith("_parameters.json")]
        if kept:
            part.present += len(kept)
            log(f"  [yellow]InfluxDB is not connected: the {len(kept)} measurement file(s) exported before stay "
                f"as they are.[/yellow]")
        else:
            part.missing.append(Missing("the InfluxDB events", "InfluxDB is not connected"))
            log("  [yellow]InfluxDB is not connected: no measurement is exported.[/yellow]")
    else:
        from openmmla.utils.querys import fetch_and_process_data, save_to_json_file

        if not dry_run:
            await asyncio.to_thread(ensure_session_layout, root, session_id)
            os.makedirs(measurements_dir, exist_ok=True)
        exported: dict[str, str] = {}
        for label, constant, suffix in _EVENT_FILES:
            if callbacks.cancelled():
                raise asyncio.CancelledError()
            event = getattr(constants, constant)
            callbacks.progress_start(f"Measurements · {label}", None)
            try:
                data = await asyncio.to_thread(fetch_and_process_data, session_id, event, influx)
                if not data:
                    log(f"  [dim]- {label}: no data[/dim]")
                    continue
                if dry_run:
                    part.fetched += 1
                    log(f"  {label}: {len(data)} records")
                    continue
                path = os.path.join(measurements_dir, f"{session_id}_{suffix}.json")
                before = await asyncio.to_thread(_digest_or_none, path)
                # the features of a session (a skeleton per person per frame) are large:
                # written compactly, they are a fraction of the indented size
                path = await asyncio.to_thread(save_to_json_file, session_id, data, suffix, measurements_dir,
                                               event == constants.EVENT_TYPE_VFA_FEATURES)
                exported[label] = path
                if before is not None and before == await asyncio.to_thread(_digest_or_none, path):
                    part.present += 1
                    log(f"  [dim]- {label}: {len(data)} records, as exported before[/dim]")
                else:
                    part.fetched += 1
                    part.fetched_bytes += os.path.getsize(path)
                    log(f"  [green]✓[/green] {label}: {len(data)} records -> {os.path.basename(path)}")
            except Exception as error:  # one event type that fails leaves the others
                part.missing.append(Missing(f"{label} events", str(error) or type(error).__name__))
                log(f"  [red]✗ {label}: {_escape(str(error))}[/red]")
            finally:
                callbacks.progress_end()
        if "ASR Transcription" in exported:
            try:
                from openmmla.analytics.asr.transcription import convert_transcription_json_to_txt

                await asyncio.to_thread(convert_transcription_json_to_txt, exported["ASR Transcription"])
                log("  [green]✓[/green] ASR Transcription -> .txt")
            except Exception as error:
                log(f"  [red]✗ Transcription txt conversion: {_escape(str(error))}[/red]")
    if not dry_run:
        lines: list[str] = []
        await asyncio.to_thread(_export_parameters, session_id, record, measurements_dir, part, lines.append)
        for line in lines:
            log(line)
        await asyncio.to_thread(_note_measurements, root, session_id, database_source, measurements_dir)
    return part


def _export_parameters(session_id: str, record: dict | None, measurements_dir: str, part: PartResult, log) -> None:
    """write <session>_parameters.json (openmmla.utils.session_provenance): the
    session's fields, the streams its bases took, and what each component ran
    with; and say in `log` (lines kept for the caller's loop) what is there."""
    from openmmla.utils import session_provenance

    if not isinstance(record, dict) or not record:
        log("  [dim]- Parameters: the session is not in MongoDB, so what it ran with is not known[/dim]")
        return
    try:
        os.makedirs(measurements_dir, exist_ok=True)
        before = _digest_or_none(os.path.join(measurements_dir, f"{session_id}_parameters.json"))
        path = session_provenance.write_session_parameters(record, measurements_dir)
        if before is not None and before == _digest_or_none(path):
            part.present += 1
        else:
            part.fetched += 1
            part.fetched_bytes += os.path.getsize(path)
        components = session_provenance.session_components(record)
        log(f"  [green]✓[/green] Parameters: {len(components)} component(s) -> {os.path.basename(path)}")
        for entry in components:
            log(f"    [dim]{_escape(session_provenance.component_summary(entry))}[/dim]")
        if not components:
            log("  [dim]  no component noted what it ran with (bases from before they did, or none joined)[/dim]")
    except Exception as error:
        log(f"  [red]✗ Parameters: {_escape(str(error))}[/red]")


def _note_measurements(root, session_id: str, database_source: dict | None, measurements_dir: str) -> None:
    """the session manifest's measurements_path (and where the databases were)."""
    import yaml

    from openmmla.tui.artifacts import artifact_session_dir, relative_to_root

    session_dir = artifact_session_dir(root, session_id)
    analysis_dir = os.path.join(session_dir, "analysis")
    manifest_path = os.path.join(session_dir, "manifest.yml")
    manifest = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file) or {}
            if isinstance(loaded, dict):
                manifest = loaded
        except yaml.YAMLError:
            manifest = {}
    manifest["session_id"] = session_id
    manifest.pop("visualizations_path", None)
    if database_source:
        manifest["database_source"] = dict(database_source)
    manifest["measurements_path"] = relative_to_root(root, measurements_dir)
    analysis = manifest.get("analysis")
    if not isinstance(analysis, dict):
        analysis = {}
    analysis["features_path"] = relative_to_root(root, os.path.join(analysis_dir, "features"))
    analysis["visualizations_path"] = relative_to_root(root, os.path.join(analysis_dir, "visualizations"))
    manifest["analysis"] = analysis
    os.makedirs(session_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False, allow_unicode=True)


# ---- the Collection recordings ----

@dataclass
class RecordingHost:
    """a host the session's Collection recordings lie on, as the session names it."""
    host: str          # its name: an SSH profile, or the console's short name for its own recorders
    label: str         # its folder under collection/
    folder: str = ""   # where they lie there, as Start wrote them ("" for that host's default)


def collection_hosts(record: dict | None, manifest: dict | None, local_session_dir=None
                     ) -> tuple[list[RecordingHost], str]:
    """the hosts of a session's recordings, and what named them: the session
    document's collection_hosts (Collection Start), else the recordings of
    the manifest here (and the hosts its downloads came from)."""
    from openmmla.utils.artifact_paths import safe_segment

    found: dict[str, RecordingHost] = {}
    names = (record or {}).get(COLLECTION_HOSTS_FIELD)
    recorders = (record or {}).get(COLLECTION_RECORDERS_FIELD)
    recorders = [item for item in recorders if isinstance(item, dict)] if isinstance(recorders, list) else []
    for name in names if isinstance(names, list) else []:
        name = str(name or "").strip()
        if not name or name in found:
            continue
        own = next((item for item in recorders if str(item.get("host") or "") == name), {})
        found[name] = RecordingHost(name, safe_segment(own.get("host_label") or name, "host"),
                                    str(own.get("folder") or ""))
    if found:
        return list(found.values()), COLLECTION_HOSTS_FIELD
    manifest = manifest or {}
    here = str(local_session_dir or "")
    downloads = ((manifest.get("artifacts") or {}).get("collection") if isinstance(manifest.get("artifacts"), dict)
                 else None)
    remote_paths = {}
    for entry in downloads if isinstance(downloads, list) else []:
        if isinstance(entry, dict) and entry.get("host") and entry.get("remote_path"):
            remote_paths[safe_segment(str(entry["host"]), "host")] = str(entry["remote_path"])
    rows = manifest.get("recordings") if isinstance(manifest.get("recordings"), list) else []
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") == "stream" or not str(row.get("host") or "").strip():
            continue
        label = safe_segment(str(row["host"]).strip(), "host")
        if label in found:
            continue
        folder = remote_paths.get(label, "")
        path = str(row.get("path") or "")
        marker = f"/collection/{label}/"
        if not folder and marker in path and not (here and path.startswith(here.rstrip("/") + "/")):
            folder = path[:path.index(marker) + len(marker) - 1]
        found[label] = RecordingHost(label, label, folder)
    for label, folder in remote_paths.items():
        found.setdefault(label, RecordingHost(label, label, folder))
    return list(found.values()), "the manifest here" if found else ""


def _reach(host: str, profiles) -> str:
    """'local' for this machine, the SSH profile that reaches `host`, or ''.
    Blocking: a host name may be resolved."""
    from openmmla.tui.system_services import target_for_service_host
    from openmmla.utils.artifact_paths import safe_segment, short_hostname

    if host.lower() == short_hostname().lower():
        return "local"
    for profile in profiles:
        if profile.name == host:
            return profile.name
    for profile in profiles:
        if safe_segment(profile.name, "host") == host:
            return profile.name
    return target_for_service_host(host, profiles)


def _candidate_folders(host: RecordingHost, profile, session_id: str, remote_root: str) -> list[str]:
    """where a host's recordings of the session may lie: where Start wrote
    them, then the recorders' default output root (~/artifacts) and the
    profile's checkout."""
    folders = [host.folder] if host.folder else []
    roots = [remote_root] if remote_root else ["~", profile.remote_project_path or "~/OpenMMLA"]
    for root in roots:
        folders.append(f"{str(root).rstrip('/')}/artifacts/{session_id}/collection/{host.label}")
    return list(dict.fromkeys(folders))


def _collection_script(specs: list[tuple[int, list[str]]], discover: list[str], session_id: str) -> str:
    """print FOUND <key> <absolute folder> for the first folder of each spec
    that holds a file, SUB <absolute folder> for every folder below
    <root>/artifacts/<session>/collection/ of each root to discover that holds
    one, then LISTED."""
    from openmmla.tui import download as dl

    has_file = f"find . -type f {_FILE_IGNORED} 2>/dev/null | head -n 1"
    parts = []
    for key, folders in specs:
        tries = " || ".join(
            f"( cd {dl.quote_remote_path(folder)} 2>/dev/null && [ -n \"$({has_file})\" ] && echo \"FOUND {key} $(pwd -P)\" )"
            for folder in folders)
        parts.append(f"{{ {tries}; }} || true")
    for root in discover:
        quoted = dl.quote_remote_path(f"{str(root).rstrip('/')}/artifacts/{session_id}/collection")
        parts.append(
            f"( if cd {quoted} 2>/dev/null; then for sub in */; do sub=\"${{sub%/}}\"; "
            f"if [ -d \"$sub\" ] && [ -n \"$(cd \"$sub\" && {has_file})\" ]; then echo \"SUB $(pwd -P)/$sub\"; fi; "
            "done; fi )")
    parts.append("echo LISTED")
    return "; ".join(parts)


def _parse_collection(text: str | None) -> tuple[dict[int, str], list[str]] | None:
    lines = [line.rstrip("\r") for line in str(text or "").splitlines()]
    if "LISTED" not in [line.strip() for line in lines]:
        return None
    found: dict[int, str] = {}
    subs: list[str] = []
    for line in lines:
        if line.startswith("FOUND "):
            key, _, path = line[6:].partition(" ")
            if key.isdigit() and path.startswith("/"):
                found.setdefault(int(key), path.rstrip("/"))
        elif line.startswith("SUB /"):
            subs.append(line[4:].rstrip("/"))
    return found, subs


async def export_collection(root, session_id: str, record: dict | None, callbacks, archive: _Archive, *,
                            dry_run: bool = False, all_profiles: bool = False) -> PartResult:
    """each recording host's folder into artifacts/<session>/collection/<host
    label>/, with the transfer and the manifest note of the Collection card's
    Download."""
    from openmmla.commands.ses.archive import _local_manifest_texts, _parse_manifest
    from openmmla.tui import download as dl
    from openmmla.tui import stream_export
    from openmmla.tui.artifacts import artifact_session_dir, collection_artifact_dir, merge_tree, update_collection_manifest

    part = PartResult("collection")
    log = callbacks.log
    remote_root = _remote_root()
    profiles = await asyncio.to_thread(_profiles, remote_root)
    session_dir = artifact_session_dir(root, session_id)
    manifest = await asyncio.to_thread(lambda: _parse_manifest(_local_manifest_texts(session_dir)))
    named, via = collection_hosts(record, manifest, session_dir)
    discover = all_profiles or not named
    if named:
        log(f"  Recorded on {_escape(', '.join(host.host for host in named))} (as {_escape(via)} names them)")
    if discover:
        why = "" if named else ": neither the session's document nor a manifest here names its hosts"
        if profiles:
            log(f"  Asking every SSH profile ({len(profiles)}) whether it holds "
                f"artifacts/{_escape(session_id)}/collection/{why}")
        else:
            log(f"  [yellow]No SSH profile to ask whether it holds artifacts/{_escape(session_id)}/collection/"
                f"{why}.[/yellow]")
    reached = await asyncio.to_thread(lambda: {host.host: _reach(host.host, profiles) for host in named})
    by_name = {profile.name: profile for profile in profiles}
    # labels whose recordings did not come from their host: (label, what is wrong, the host)
    wanting: list[tuple[str, str, str]] = []
    # labels whose recordings are all here, from wherever they came
    done: set[str] = set()

    def note_manifest(label: str, remote_path: str, local_path: Path):
        def note() -> str:
            manifest_path = update_collection_manifest(root, session_id=session_id, host_name=label,
                                                       remote_path=remote_path, local_path=local_path)
            return f"[green]Updated session manifest: {_escape(manifest_path)}[/green]"
        return note

    # this machine's own recorders wrote here in the first place
    for host in named:
        if reached.get(host.host) != "local":
            continue
        local_dir = collection_artifact_dir(root, session_id, host.label)
        count = len(_media([str(path) for path in await asyncio.to_thread(_files_in, local_dir)]))
        elsewhere = Path(host.folder).expanduser() if host.folder else None
        if count:
            part.present += count
            done.add(host.label)
            log(f"  [dim]- {_escape(host.host)}: recorded on this machine, {count} file(s) in "
                f"{_shown_path(root, local_dir)}[/dim]")
        elif elsewhere is not None and elsewhere.is_dir() and elsewhere.resolve() != local_dir.resolve():
            # an output root of its own on this machine: copied into place
            if dry_run:
                files = await asyncio.to_thread(_files_in, elsewhere)
                part.fetched += len(files)
                log(f"  Would copy {len(files)} file(s) from {_escape(host.folder)}")
            else:
                stats = await asyncio.to_thread(merge_tree, elsewhere, local_dir, conflict_label=host.host)
                copied, skipped = await asyncio.to_thread(_merge_counts, elsewhere, stats)
                part.fetched += copied
                part.present += skipped
                log(f"  [green]{_escape(host.host)}: copied {copied} file(s) from {_escape(host.folder)}[/green]")
            done.add(host.label)
        else:
            wanting.append((host.label, f"{host.host} (this machine) holds no recording of it", host.host))
    for host in named:
        if not reached.get(host.host):
            log(f"  [yellow]No SSH profile reaches {_escape(host.host)}: its recordings of {_escape(session_id)} "
                f"(collection/{_escape(host.label)}/) are not fetched from it.[/yellow]")
            wanting.append((host.label, f"no SSH profile reaches {host.host}", host.host))

    # every profile asked once, all at once
    asks: dict[str, dict] = {}
    for index, host in enumerate(named):
        name = reached.get(host.host)
        if name and name != "local" and name in by_name:
            entry = asks.setdefault(name, {"specs": [], "discover": []})
            entry["specs"].append((index, _candidate_folders(host, by_name[name], session_id, remote_root)))
    if discover:
        for profile in profiles:
            entry = asks.setdefault(profile.name, {"specs": [], "discover": []})
            roots = [remote_root] if remote_root else ["~", profile.remote_project_path or "~/OpenMMLA"]
            entry["discover"] = list(dict.fromkeys(roots))
    names = list(asks)
    callbacks.progress_start(f"Collection · asking {len(names)} host(s)", None)
    try:
        answers = await asyncio.gather(*(
            stream_export.run_on_host(name, _collection_script(asks[name]["specs"], asks[name]["discover"],
                                                               session_id), ASK_TIMEOUT)
            for name in names))
    finally:
        callbacks.progress_end()

    # (profile, folder there, label, the host it came from) to fetch, each folder once
    sources: list[tuple[object, str, str, str]] = []
    seen: dict[tuple[str, str], str] = {}
    for name, answer in zip(names, answers):
        profile = by_name[name]
        parsed = _parse_collection(answer)
        own = [named[index] for index, _folders in asks[name]["specs"]]
        if parsed is None:
            for host in own:
                log(f"  [yellow]{_escape(name)} did not answer (offline?): its recordings of {_escape(session_id)} "
                    f"(collection/{_escape(host.label)}/) are not fetched from it.[/yellow]")
                wanting.append((host.label, f"{name} did not answer", host.host))
            if asks[name]["discover"] and not own:
                log(f"  [yellow]{_escape(name)}: could not be asked (offline?); whatever it holds of "
                    f"{_escape(session_id)} stays there, and what the archive host holds is fetched below.[/yellow]")
            continue
        found, subs = parsed
        for index, _folders in asks[name]["specs"]:
            host = named[index]
            if index in found:
                sources.append((profile, found[index], host.label, host.host))
            else:
                log(f"  [yellow]{_escape(name)} no longer holds the recordings of {_escape(session_id)} "
                    f"(collection/{_escape(host.label)}/).[/yellow]")
                wanting.append((host.label, f"{name} no longer holds them", host.host))
        for folder in subs:
            sources.append((profile, folder, os.path.basename(folder), name))
    for profile, folder, label, origin in sources:
        key = (f"{profile.ssh_destination()}:{profile.port}", folder)
        local_dir = collection_artifact_dir(root, session_id, label)
        if key in seen:
            continue
        if Path(folder).is_dir() and os.path.realpath(folder) == os.path.realpath(local_dir):
            # a profile that reaches this machine: they are in place already
            done.add(label)
            continue
        seen[key] = profile.name
        if callbacks.cancelled():
            raise asyncio.CancelledError()
        if dry_run:
            counts = await _plan_counts(profile, folder, local_dir)
            if counts is None:
                wanting.append((label, f"{profile.name} could not be asked again", origin))
                continue
            part.fetched += counts[0]
            part.fetched_bytes += counts[1]
            part.present += counts[2]
            done.add(label)
            log(f"  Would fetch {counts[0]} file(s) from {_escape(profile.name)}:{_escape(folder)} "
                f"({counts[2]} here already)")
            continue
        ok = await _Counted(part)(
            profile, folder, local_dir,
            staging=dl.staging_root(root, session_id, "collection", label),
            label=label, where=profile.name, what=f"collection/{label}",
            callbacks=callbacks, merge=merge_tree, after_merge=note_manifest(label, folder, local_dir))
        if ok:
            done.add(label)
        else:
            wanting.append((label, f"the transfer from {profile.name} did not complete", origin))

    # what did not come from its host, from the archive host; a folder a host
    # was named for, or was found holding, is known to exist: counted when missing
    for label, why, origin in wanting:
        if label in done:
            continue  # another profile reaching the same machine brought it
        done.add(label)
        local_dir = collection_artifact_dir(root, session_id, label)
        rel = f"collection/{label}"
        if await archive.fetch(rel, local_dir, part,
                               after_merge=note_manifest(label, f"{archive.name}:{rel}", local_dir)):
            continue
        part.missing.append(Missing(f"collection/{label}/ ({origin})", f"{why}; {archive.none_note(rel)}"))
    if discover:
        # a host that could not be asked, or no longer holds its folder, may have
        # left it on the archive host: each folder there that no host brought
        labels = [label for label in await archive.subfolders("collection") if label not in done]
        await archive.ask([f"collection/{label}" for label in labels])
        for label in labels:
            if callbacks.cancelled():
                raise asyncio.CancelledError()
            done.add(label)
            local_dir = collection_artifact_dir(root, session_id, label)
            rel = f"collection/{label}"
            if not await archive.fetch(rel, local_dir, part,
                                       after_merge=note_manifest(label, f"{archive.name}:{rel}", local_dir)):
                part.missing.append(Missing(f"collection/{label}/ ({archive.name})", archive.none_note(rel)))
        if not done:
            log(f"  [dim]- No host holds Collection recordings of {_escape(session_id)}.[/dim]")
    return part


# ---- the streams ----

class _ExportStopped(Exception):
    """Cancel was pressed while a clip was being downloaded."""


@dataclass
class ServerSources:
    """the sources of a session as the Stream Server of System Settings sees them."""
    paths: list[str] = field(default_factory=list)                  # its paths there, each once, in the order noted
    elsewhere: list[tuple[str, str]] = field(default_factory=list)  # (stream, URL) published to another server
    by_stream: dict[str, str] = field(default_factory=dict)         # stream name -> its path there
    direct: list[str] = field(default_factory=list)                 # `asr:0 mic-1`: taken through no server at all


def server_sources(record: dict | None, server: dict) -> ServerSources:
    """sort the sources of a session record by where their streams went: the
    URL each base pulled is looked up on the Stream Server of System Settings
    (system_services.stream_server_path, which checks the host). Blocking: a
    host name may be resolved."""
    from urllib.parse import urlsplit

    from openmmla.tui.system_services import hosts_match, is_loopback_host, stream_server_path
    from openmmla.utils import session_sources

    server_host = str((server or {}).get("host") or "").strip().strip("[]")
    found = ServerSources()
    for entry in session_sources.session_sources(record):
        name = str(entry.get("stream") or entry.get("key") or "?")
        url = str(entry.get("url") or "").strip()
        # only what a stream server serves (rtmp, rtsp, srt): udp and tcp go straight to a base
        served = session_sources.stream_url_path(url) if url else None
        if served is None:
            found.direct.append(f"{entry.get('key')} {entry.get('stream') or entry.get('source') or '?'}")
            continue
        path = stream_server_path(url, server)
        if path is None and is_loopback_host(urlsplit(url).hostname) and hosts_match(entry.get("host"), server_host):
            # localhost in the URL of a base that runs on the Stream Server's host is that server
            path = served
        if path is None:
            if (name, url) not in found.elsewhere:
                found.elsewhere.append((name, url))
            continue
        found.by_stream.setdefault(name, path)
        if path not in found.paths:
            found.paths.append(path)
    return found


@dataclass
class _ServerCopy:
    here: int = 0                              # clips here afterwards
    failed: str = ""                           # why the server copy could not be had at all
    failed_clips: list[str] = field(default_factory=list)


async def export_streams(root, session_id: str, record: dict | None, callbacks, archive: _Archive | None = None, *,
                         dry_run: bool = False, retention=ASK) -> PartResult:
    """both copies of the session's part of every stream it used, from its
    `sources`.

    A stream is shared by the sessions that pull it: the Stream Server records
    its path while a session that pulls it runs (START to STOP,
    openmmla.utils.stream_recording), and with Record on the machine that
    captures it records it whether or not one runs. Nothing of either belongs
    to a session; each base notes in the session's document the stream it
    takes (openmmla.utils.session_sources), and the session's start and end
    (recordings.session_end) pick the part of it:
      - the server copy, over HTTP: each noted path on the Stream Server of
        System Settings, one file per unbroken stretch, into
        artifacts/<session>/streams/server/<app>/<name>_<start>.mp4;
      - the capture copy, over SSH: each noted stream's recording, cut on its
        capture host and fetched (stream_export), into
        artifacts/<session>/streams/capture/<host label>/<video|audio>/.
    A session without that note (one begun before the bases wrote it, or one
    no base joined) has nothing to export, and says so. Cancel stops it
    between steps (a clip being downloaded at once); what arrived stays.
    `retention`: the Stream Server's, in seconds (None: not known), ASK to ask
    the server when it is needed."""
    from openmmla.tui import recordings
    from openmmla.tui.system_services import stream_server_address
    from openmmla.utils import session_sources
    from openmmla.utils.artifact_paths import session_server_streams_dir

    part = PartResult("streams")
    log = callbacks.log
    shown = _escape(session_id)
    record = record or {}
    start = recordings.parse_time(record.get("start_time"))
    if start is None:
        log(f"  [yellow]'{shown}' has no start time in MongoDB (it is known from its artifacts only), so there is "
            f"no time range to cut out of the streams.[/yellow]")
        if archive is not None:
            # what the archive host holds goes beside a different copy here, never over it
            await archive.fetch("streams", Path(root) / "artifacts" / session_id / "streams", part)
        return part
    if not session_sources.session_sources(record):
        # the conclusion first: the log does not wrap
        log(f"  [yellow]Nothing to export: '{shown}' names no stream (it began before the bases noted theirs, or "
            f"no base joined it).[/yellow]")
        return part
    end, why = recordings.session_end(record)
    until = f"{end:%H:%M:%S}" if end.date() == start.date() else f"{end:%Y-%m-%d %H:%M:%S}"
    log(f"  {start:%Y-%m-%d %H:%M:%S} to {until} UTC ({_WINDOW_REASONS[why]})")

    server = await asyncio.to_thread(stream_server_address, _settings_root())
    # without an address no URL is known to be the Stream Server's
    sources = await asyncio.to_thread(server_sources, record, server) if server.get("host") else ServerSources()
    copy = await _server_copy(root, session_id, server, sources, (start, end), why == "ended", callbacks,
                              retention, part, dry_run)
    if callbacks.cancelled():
        raise asyncio.CancelledError()
    if copy.failed or copy.failed_clips:
        why = copy.failed or ", ".join(copy.failed_clips)
        if archive is None or not await archive.fetch("streams/server", session_server_streams_dir(root, session_id),
                                                      part):
            part.missing.append(Missing("streams/server/ (the Stream Server's copy)",
                                        why + (f"; {archive.none_note('streams/server')}" if archive is not None
                                               else "")))
    from_capture = await _capture_copy(root, session_id, record, sources, (start, end), callbacks, part, archive,
                                       dry_run)

    folder = _shown_path(root, session_server_streams_dir(root, session_id).parent)
    archived = f", {part.from_archive} from the archive host" if part.from_archive else ""
    if dry_run:
        log(f"  Would have {copy.here} clip(s) from the Stream Server and {from_capture} cut(s) from the capture "
            f"hosts{archived} under {folder}")
    elif copy.here or from_capture or part.fetched or part.present:
        log(f"[bold green]Streams of {shown} exported: {copy.here} file(s) from the Stream Server, {from_capture} "
            f"from the capture hosts{archived}, under {folder}[/bold green]\n"
            f"  [dim]File names carry the time they start at, so a base replays one with source: file and its full "
            f"path as source_index.[/dim]")
    else:
        log(f"[yellow]Nothing of the streams of {shown} was exported (see above).[/yellow]")
    return part


def _server_clips_here(out_dir: Path, path: str, start: datetime, end: datetime) -> list[Path]:
    """the clips of a Stream Server path an earlier export brought here for
    this window: <app>/<name>_<start>.mp4 under streams/server/
    (recordings.clip_relpath)."""
    parts = [part for part in path.split("/") if part and part not in (".", "..")]
    folder = Path(out_dir).joinpath(*parts[:-1]) if len(parts) > 1 else Path(out_dir)
    return _exported_here(folder, parts[-1] if parts else "stream", ".mp4", start.timestamp(), end.timestamp())


async def _server_copy(root, session_id: str, server: dict, sources: ServerSources,
                       window: tuple[datetime, datetime], ended: bool, callbacks, retention,
                       part: PartResult, dry_run: bool = False) -> _ServerCopy:
    """the server copy: each unbroken stretch of the session's paths on the
    Stream Server within its window, asked of the playback server over HTTP
    (no SSH). A clip already here in full is not fetched again."""
    from openmmla.tui import recordings
    from openmmla.tui.artifacts import copy_covers, ensure_session_layout
    from openmmla.utils.artifact_paths import session_server_streams_dir

    result = _ServerCopy()
    log = callbacks.log
    start, end = window
    shown = _escape(session_id)
    host = str(server.get("host") or "")
    if not host:
        from openmmla.tui.system_services import unset_address_note

        log(f"[yellow]Not asked of the Stream Server: {unset_address_note('StreamServer')}.[/yellow]")
        return result
    playback_port = int(server.get("playback_port") or recordings.PLAYBACK_PORT)
    out_dir = session_server_streams_dir(root, session_id)
    log(f"[cyan]From the Stream Server {_escape(host)} into {_shown_path(root, out_dir)}[/cyan]")
    for name, url in sources.elsewhere:
        log(f"  [yellow]- {_escape(name)}: published to another server ({_escape(url)}), not to the Stream Server "
            f"of System Settings, so it is skipped here.[/yellow]")
    paths = list(sources.paths)
    if not paths:
        if not sources.elsewhere:
            log(f"  [yellow]None of the bases of '{shown}' took a stream through the Stream Server "
                f"({_escape(', '.join(sources.direct))}), so it holds nothing of this session.[/yellow]")
        return result
    log(f"  Its streams, as its bases noted them: {_escape(', '.join(paths))}")

    def ask() -> dict:
        # the session's own paths need no listing: the playback server is
        # asked for those, as they are named
        return {path: recordings.timespans(host, path, playback_port) for path in paths}

    callbacks.progress_start(f"Stream Server · asking {host}", None)
    try:
        spans = await asyncio.to_thread(ask)
    except recordings.RecordingsError as error:
        log(f"  [red]✗ The Stream Server does not answer: {_escape(error)}[/red]\n"
            f"  [dim]Its host and its API/playback ports are under Launcher → System Settings → Stream Server; the "
            f"API and the playback server are switched on in its mediamtx.yml.[/dim]")
        result.failed = f"the Stream Server {host} did not answer"
        return result
    finally:
        callbacks.progress_end()
    clips = recordings.clips_for_window(spans, start, end)
    if not clips:
        # the clips an earlier export brought of each path are here still, whatever the server holds now
        kept = await asyncio.to_thread(
            lambda: {path: len(_server_clips_here(out_dir, path, start, end)) for path in paths})
        result.here += sum(kept.values())
        part.present += sum(kept.values())
        gone = [path for path in paths if not kept[path]]
        here_note = (f"; the {sum(kept.values())} clip(s) exported before are here"
                     + (f", and nothing of {', '.join(gone)}" if gone else "") if any(kept.values()) else "")
        if retention is ASK:
            # how long the server keeps a recording tells deleted from never recorded
            retention = None
            with contextlib.suppress(recordings.RecordingsError, ValueError, TypeError):
                retention = await asyncio.to_thread(
                    recordings.retention, host, int(server.get("api_port") or recordings.API_PORT), 2.0)
        expired = recordings.expiry(start, retention)
        if expired is not None and expired <= datetime.now(timezone.utc):
            color = "yellow" if gone else "dim"
            log(f"  [{color}]Nothing left on the server: it keeps a recording for "
                f"{recordings.describe_retention(retention)}, and this session's footage was deleted from "
                f"{expired:%Y-%m-%d %H:%M} UTC on{_escape(here_note)}. The retention is set on the Stream Server "
                f"card, Config tab; its Recordings tab shows what the server still holds.[/{color}]")
            if gone:
                result.failed = (f"past the Stream Server's retention ({recordings.describe_retention(retention)}), "
                                 f"and nothing of {', '.join(gone)} is here")
            return result
        held = [path for path in paths if spans.get(path)]
        held_note = f" (it holds {', '.join(held)} from other times)" if held else " (it holds no recording of them)"
        if not gone:
            log(f"  [dim]- Nothing of its streams is on the server in that time any more"
                f"{_escape(held_note + here_note)}.[/dim]")
            return result
        found = "Nothing of its streams was recorded on the server in that time" + held_note + here_note + "."
        log(f"  [yellow]{_escape(found)} The server records a session's paths from its START to its STOP (Session "
            f"Control); a session never STARTed, or a server restarted meanwhile, has none.[/yellow]")
        return result

    if not dry_run:
        await asyncio.to_thread(ensure_session_layout, root, session_id)
    loop = asyncio.get_running_loop()
    for index, clip in enumerate(clips, 1):
        if callbacks.cancelled():
            raise asyncio.CancelledError()
        destination = out_dir / recordings.clip_relpath(clip)
        label = _escape(f"{clip.path}  {clip.start:%H:%M:%S} +{clip.duration:.0f}s")
        size_here = destination.stat().st_size if destination.is_file() else 0
        if size_here:
            # a clip is named after its start only, so a copy exported while
            # the session was still going looks like the full one: its length
            # tells. Without ffprobe here, an ended session's clip counts as final
            covered = await asyncio.to_thread(copy_covers, destination, clip.duration)
            if covered or (covered is None and ended):
                log(f"  [dim]- {label}: already exported[/dim]")
                result.here += 1
                part.present += 1
                continue
            if covered is False:
                log(f"  [cyan]{label}: the copy here stops short (exported while the session was still going); "
                    f"fetching it in full[/cyan]")
        if dry_run:
            log(f"  Would fetch {label}")
            result.here += 1
            part.fetched += 1
            continue
        callbacks.progress_start(f"Stream Server · {index} of {len(clips)}", None)
        try:
            size = await asyncio.to_thread(
                recordings.download_clip, host, clip, str(destination), playback_port,
                progress=_clip_progress(loop, clip, callbacks))
        except recordings.RecordingsError as error:
            log(f"  [red]✗ {label}: {_escape(error)}[/red]")
            result.failed_clips.append(str(error))
            continue
        except _ExportStopped:
            raise asyncio.CancelledError() from None
        finally:
            callbacks.progress_end()
        result.here += 1
        part.fetched += 1
        part.fetched_bytes += int(size or 0)
        log(f"  [green]✓[/green] {label} -> {_escape(recordings.clip_relpath(clip))} ({size / 1e6:.1f} MB)")
    log(f"  [green]{result.here} of {len(clips)} clip(s) are under {_shown_path(root, out_dir)}[/green]")
    return result


def _clip_progress(loop, clip, callbacks):
    """the progress callback of one clip's download, which runs in its
    thread: it stops the download once Cancel was pressed, and moves the
    progress row on the app's loop a few times a second."""
    from openmmla.tui import recordings

    began = time.monotonic()
    shown = [0.0]

    def progress(written: int) -> None:
        if callbacks.cancelled():
            raise _ExportStopped()
        now = time.monotonic()
        if now - shown[0] < 0.25:
            return
        shown[0] = now
        rate = written / max(now - began, 1e-3)
        detail = f"{clip.path} · {recordings.human_size(written)} · {recordings.human_size(rate)}/s"
        try:
            loop.call_soon_threadsafe(callbacks.progress_update, written, 0, detail)
        except RuntimeError:  # the app's loop is closed
            pass

    return progress


async def _capture_copy(root, session_id: str, record: dict, sources: ServerSources,
                        window: tuple[datetime, datetime], callbacks, part: PartResult,
                        archive: _Archive | None = None, dry_run: bool = False) -> int:
    """the capture copy: each noted stream's recording on the machine that
    captured it, cut there to the window without re-encoding and fetched
    (stream_export.export_session, which says what it does line by line). How
    many cuts are here afterwards.

    Every stream the console captures is asked for, whatever Record its bases
    noted: a base notes it from the config of the machine it runs on, which
    need not be the config the stream was started with. One noted off that its
    host holds nothing of, or whose host does not answer, is said in a dim
    line and is not missing; one whose recording ffmpeg could not cut is."""
    from openmmla.tui import stream_cuts, stream_export
    from openmmla.utils.artifact_paths import session_capture_streams_dir

    log = callbacks.log
    start, end = window
    folder = session_capture_streams_dir(root, session_id)
    log(f"[cyan]From the capture hosts into {_shown_path(root, folder)}[/cyan]")
    found = stream_export.session_streams(record, root)

    def only_copy(name: str) -> str:
        """where the copy of a stream no capture host gave is (markup)."""
        path = sources.by_stream.get(name)
        other = next((url for stream, url in sources.elsewhere if stream == name), None)
        if path:
            return f"on the Stream Server ({_escape(path)}, above)"
        if other:
            return f"on the server it was published to ({_escape(other)})"
        return "on the Stream Server, if it went through one"

    for name, _why in found.skipped:
        log(f"  [dim]- {_escape(name)}: someone else publishes it (it has no SSH Profile), so no capture host of "
            f"this console records it; its only copy is {only_copy(name)}[/dim]")
    streams = found.streams
    if not streams:
        if not found.skipped:
            log("  [dim]- Its bases took no stream the console captures (a camera or microphone of their own, or a "
                "file), so there is nothing to cut.[/dim]")
        return 0
    # a host that could not be asked, no longer holds a stream's recording, or
    # whose cuts did not arrive: what the archive holds of it
    lost: dict[str, tuple[str, str]] = {}
    # the cuts this export's transfers brought: counted as fetched, never as here already
    arrived: set[Path] = set()

    async def kept_here(stream) -> tuple[list[Path], list[Path]]:
        """the cuts of a stream's window here: those an earlier export brought,
        and those a transfer of this one brought that this run did not make
        (an earlier export staged them, and its fetch did not finish)."""
        found = await asyncio.to_thread(
            _exported_here, folder / stream.host_label / stream.kind, stream.name,
            f".{stream_cuts.EXTENSIONS.get(stream.kind, 'mkv')}", start.timestamp(), end.timestamp())
        return [path for path in found if path not in arrived], [path for path in found if path in arrived]

    def said_here(before: list[Path], now: list[Path]) -> str:
        """which of a stream's cuts are here, as kept_here() splits them."""
        said = [f"the {len(before)} cut(s) exported before are here"] if before else []
        if now:
            said.append(f"the {len(now)} cut(s) an earlier export left staged there were fetched now")
        return "; ".join(said)

    async def none_there(stream, where: str) -> int:
        """a capture host that answered with no recording of the window: the
        cuts an earlier export brought are here (how many; one fetched now
        counts as fetched), else what the archive host holds of it is
        fetched; for a stream noted Record off, where its only copy is."""
        before, now = await kept_here(stream)
        if before or now:
            part.present += len(before)
            log(f"  [dim]- {_escape(stream.name)}: {_escape(where)} holds no recording of that time any more; "
                f"{said_here(before, now)}[/dim]")
            return len(before)
        if stream.noted_off:
            log(f"  [dim]- {_escape(stream.name)}: Record was off for it, and {_escape(where)} holds no recording of "
                f"it from that time; its only copy is {only_copy(stream.name)}[/dim]")
            return 0
        log(f"  [yellow]{_escape(where)} holds no recording of {_escape(stream.name)} from the session's time "
            f"(deleted since, or never recorded): the cut of it (streams/capture/{_escape(stream.host_label)}/) "
            f"is not made there.[/yellow]")
        lost.setdefault(stream.host_label, (where, f"{where} holds no recording of {stream.name} from that time"))
        return 0

    async def not_asked(stream, where: str) -> int:
        """a capture host that did not answer for a stream noted Record off:
        not missing, since nothing says it was recorded there; the cuts an
        earlier export brought are here (how many; one fetched now counts as
        fetched)."""
        before, now = await kept_here(stream)
        if before or now:
            part.present += len(before)
            log(f"  [dim]- {_escape(stream.name)}: {_escape(where)} could not be asked; "
                f"{said_here(before, now)}[/dim]")
            return len(before)
        log(f"  [dim]- {_escape(stream.name)}: Record was off for it by its base's config, and {_escape(where)} "
            f"could not be asked whether it recorded it anyway; its only copy known is "
            f"{only_copy(stream.name)}[/dim]")
        return 0

    if dry_run:
        here = 0
        for stream in streams:
            listing = await stream_export.run_on_host(
                stream.ssh_profile,
                stream_cuts.list_script(stream.record_root, stream.host_label, stream.kind, stream.name), 30.0)
            files = stream_cuts.parse_listing(listing or "", stream.name)
            where = "this machine" if stream.ssh_profile == "local" else stream.ssh_profile
            if files is None:
                if stream.noted_off:
                    here += await not_asked(stream, where)
                    continue
                log(f"  [yellow]{_escape(where)} did not answer (offline?): {_escape(stream.name)} would not be cut "
                    f"there (streams/capture/{_escape(stream.host_label)}/).[/yellow]")
                lost.setdefault(stream.host_label, (where, f"{where} did not answer"))
                continue
            cuts = stream_cuts.cuts_for_window(files, start.timestamp(), end.timestamp())
            if not cuts:
                here += await none_there(stream, where)
                continue
            if stream.noted_off:
                log(stream_export.recorded_anyway(stream))
            here += len(cuts)
            # no bytes: a cut has no size before it is made, and the listing gives none of the recordings
            part.fetched += len(cuts)
            log(f"  {_escape(stream.name)}: {len(cuts)} cut(s) to make on {_escape(where)}")
        await _capture_from_archive(lost, folder, part, archive)
        return here

    callbacks.progress_start("Capture hosts · cutting", None)
    try:
        result = await stream_export.export_session(root, session_id, streams, start.timestamp(), end.timestamp(),
                                                    callbacks)
    finally:
        callbacks.progress_end()
    part.fetched += result.fetched
    # a cut made here counts with its size, as a file Measurements writes here does
    part.fetched_bytes += result.fetched_bytes
    part.present += result.present
    arrived.update(result.arrived)
    if result.here:
        log(f"  [green]{result.here} cut(s) are under {_shown_path(root, result.folder)}"
            + (f" ({result.present} of them were already here)" if result.present else "") + "[/green]")
    kept = 0
    for cut in result.cuts:
        stream = cut.stream
        where = "this machine" if stream.ssh_profile == "local" else stream.ssh_profile
        if not cut.listed and stream.noted_off:
            kept += await not_asked(stream, where)
        elif not cut.listed:
            log(f"  [yellow]{_escape(where)} did not answer (offline?): the cut of {_escape(stream.name)} "
                f"(streams/capture/{_escape(stream.host_label)}/) is not made there.[/yellow]")
            lost.setdefault(stream.host_label, (where, f"{where} did not answer"))
        elif cut.failed:
            # a recording that is there and could not be cut, whatever Record was noted
            lost.setdefault(stream.host_label, (where, f"ffmpeg could not cut {stream.name} on {where}"))
        elif not (cut.made or cut.present):
            kept += await none_there(stream, where)
    for profile in result.unfetched:
        for cut in result.cuts:
            # a stream noted Record off is missing only for a cut made of it
            if cut.stream.ssh_profile == profile and (cut.made or not cut.stream.noted_off):
                lost.setdefault(cut.stream.host_label, (profile, f"the cuts made on {profile} did not arrive"))
    if result.unfetched:
        hosts = ", ".join(dict.fromkeys(result.unfetched))
        log(f"  [yellow]The cuts made on {_escape(hosts)} did not all arrive. They stay there: press Export again "
            f"to fetch them.[/yellow]")
    await _capture_from_archive(lost, folder, part, archive)
    return result.here + kept


async def _capture_from_archive(lost: dict[str, tuple[str, str]], folder: Path, part: PartResult,
                                archive: _Archive | None) -> None:
    """the capture-side cuts of the hosts in `lost` (host label -> (host, what
    went wrong)) from the archive host; what it does not hold is missing."""
    for label, (where, why) in lost.items():
        rel = f"streams/capture/{label}"
        if archive is not None and await archive.fetch(rel, folder / label, part):
            continue
        part.missing.append(Missing(f"{rel}/ ({where})", why + (f"; {archive.none_note(rel)}" if archive else "")))


# ---- the base files ----

async def export_base(root, session_id: str, record: dict | None, callbacks, archive: _Archive | None = None, *,
                      dry_run: bool = False) -> PartResult:
    """what the session's bases and synchronizers (and an older session's IPS
    visualizer) wrote on the machines they ran on (their logs, the config they
    ran with, what they recorded), from the host of every SSH profile, into
    artifacts/<session>/pipelines/<pipeline>/<host>/ (base_files). Those run on
    this machine wrote there in the first place. A transfer cut short resumes
    at the next export. What did not arrive from its host (a folder whose
    transfer did not complete, a host its bases noted that no profile reached
    or that holds none of their files now) is looked for on the archive host."""
    from openmmla.tui import base_files
    from openmmla.utils.artifact_paths import short_hostname

    part = PartResult("base")
    log = callbacks.log
    shown = _escape(session_id)
    pipelines_dir = base_files.session_pipelines_dir(root, session_id)
    folder = _shown_path(root, pipelines_dir)
    profiles = await asyncio.to_thread(_profiles, _remote_root())
    here = await asyncio.to_thread(base_files.local_parts, root, session_id)
    if here:
        count = sum(len(_media([str(path) for path in _files_in(pipelines_dir / p)])) for p in here)
        part.present += count
        log(f"  [dim]This machine's own are in {folder} already: {_escape(', '.join(here))}[/dim]")
    # (profile, <pipeline>/<host>/<folder>) found on a host and not here afterwards
    unfetched: list[tuple[str, str]] = []
    if profiles:
        log(f"[cyan]Asking {len(profiles)} SSH host(s) what they hold of it, into {folder}[/cyan]")
        fetch = _Counted(part) if not dry_run else _dry_fetch(part)
        result = await base_files.export_session(root, session_id, profiles, callbacks, record=record, fetch=fetch)
        missing, incomplete, fetched = result.missing, list(result.incomplete), result.fetched
        unfetched = list(result.unfetched)
    else:
        log("  [yellow]No SSH profile to ask: a base run on another machine keeps its files there. Add that machine "
            "under System Settings → Hosts → SSH Profiles.[/yellow]")
        me = base_files._machine(short_hostname())
        missing = {host: keys for host, keys in base_files.noted_hosts(record).items()
                   if base_files._machine(host) != me}
        incomplete, fetched = [], []
    for host, keys in missing.items():
        log(f"  [yellow]{_escape(', '.join(keys))} ran on {_escape(host)}, which no SSH profile reached: its files stay "
            f"there.[/yellow]")
    # a host its bases noted that answered (or this machine) and holds none of their files, none here either
    held = {base_files._machine(name.split("/")[1]) for _profile, name in [*fetched, *unfetched]}
    held |= await asyncio.to_thread(_machines_here, pipelines_dir)
    gone = {host: keys for host, keys in base_files.noted_hosts(record).items()
            if host not in missing and base_files._machine(host) not in held}
    for host, keys in gone.items():
        log(f"  [yellow]{_escape(', '.join(keys))} ran on {_escape(host)}, and none of its files are here or on that "
            f"host now (deleted since, or kept under another checkout).[/yellow]")
    if dry_run:
        incomplete, unfetched = [], []  # nothing was meant to arrive
    elif fetched:
        parts = ", ".join(f"{name} from {profile}" for profile, name in fetched)
        log(f"[bold green]Base files of {shown} exported: {_escape(parts)}, under {folder}[/bold green]")
    if incomplete:
        log(f"[yellow]Not all of it arrived from {_escape(', '.join(incomplete))} (see above). What was "
            f"staged is kept: press Export again to go on.[/yellow]")
    elif not fetched and not dry_run and profiles:
        log(f"[yellow]No SSH host holds base files of {shown}.[/yellow]")
    if not (missing or gone or unfetched):
        return part
    # only what did not arrive from its origin: never what came from it, nor this machine's own
    machines = {base_files._machine(host) for host in [*missing, *gone]}
    taken = (await _base_from_archive(root, session_id, callbacks, archive, part, machines,
                                      {name for _profile, name in unfetched}) if archive else {})
    by_machine: dict[str, bool] = {}
    for name, arrived in taken.items():
        machine = base_files._machine(name.split("/")[1])
        by_machine[machine] = by_machine.get(machine, True) and arrived

    def note(arrived: bool | None) -> str:
        if archive is None:
            return ""
        if arrived is False:
            return f"; its copy on {archive.name} did not all arrive (Export again to go on)"
        return f"; {archive.none_note()}"

    for host, keys in missing.items():
        arrived = by_machine.get(base_files._machine(host))
        if not arrived:
            part.missing.append(Missing(f"pipelines/*/{host}/ ({', '.join(keys)})",
                                        "no SSH profile reached it" + note(arrived)))
    for host, keys in gone.items():
        arrived = by_machine.get(base_files._machine(host))
        if not arrived:
            part.missing.append(Missing(f"pipelines/*/{host}/ ({', '.join(keys)})",
                                        f"{host} holds none of its files now" + note(arrived)))
    for profile, name in unfetched:
        if not taken.get(name):
            part.missing.append(Missing(f"pipelines/{name}/ ({profile})",
                                        f"its transfer from {profile} did not complete" + note(taken.get(name))))
    return part


def _machines_here(pipelines_dir: Path) -> set[str]:
    """the machines with a <pipeline>/<host>/ folder here that holds a file."""
    from openmmla.tui import base_files

    found = set()
    try:
        for pipeline in Path(pipelines_dir).iterdir():
            for host in pipeline.iterdir() if pipeline.is_dir() else []:
                if host.is_dir() and _files_in(host):
                    found.add(base_files._machine(host.name))
    except OSError:
        pass
    return found


def _dry_fetch(part: PartResult, archived: bool = False):
    """the `fetch` of base_files for a dry run: what a transfer would fetch,
    counted, and nothing fetched (False, so nothing is noted in a manifest)."""

    async def fetch(profile, remote_dir, local_dir, **kwargs):
        counts = await _plan_counts(profile, remote_dir, local_dir)
        if counts is not None:
            part.fetched += counts[0]
            part.fetched_bytes += counts[1]
            part.present += counts[2]
            if archived:
                part.from_archive += counts[0]
            kwargs["callbacks"].log(f"  Would fetch {counts[0]} file(s) of {_escape(kwargs.get('what') or remote_dir)}"
                                    f" ({counts[2]} here already)")
        return False

    return fetch


async def _base_from_archive(root, session_id: str, callbacks, archive: _Archive, part: PartResult,
                             machines: set[str], folders: set[str]) -> dict[str, bool]:
    """the base files the archive host holds of the session: every folder of
    the machines in `machines`, and the <pipeline>/<host>/<folder> in
    `folders`; never one of this machine's own here, which are the origin
    copy. What arrives goes beside a different copy here, never over it
    (_merge_archived). <pipeline>/<host>/<folder> -> whether it is all here
    afterwards, for each folder taken."""
    from openmmla.tui import base_files, stream_export

    if not (machines or folders) or not (await archive.ask(["pipelines"])).get("pipelines"):
        return {}
    target = archive.target
    root_there = archive.session_dir.rsplit("/", 2)[0]  # <project>/artifacts/<session> -> <project>
    answer = await stream_export.run_on_host(archive._runner(), base_files.list_script(root_there, session_id),
                                             ASK_TIMEOUT)
    listing = base_files.parse_listing(answer)
    if listing is None or not listing.folders:
        return {}
    local_dir = base_files.session_pipelines_dir(root, session_id)
    own = {name for name in await asyncio.to_thread(base_files.local_parts, root, session_id)
           if await asyncio.to_thread(_files_in, local_dir / name)}
    taken: dict[str, bool] = {}
    for name, held in listing.parts().items():
        if name in own:
            continue
        machine = base_files._machine(name.split("/", 1)[1])
        wanted = held if machine in machines else [sub for sub in held if f"{name}/{sub}" in folders]
        if not wanted:
            continue
        if callbacks.cancelled():
            raise asyncio.CancelledError()
        if target.here:
            copied = skipped = 0
            for sub in wanted:
                source = Path(listing.root) / name / sub
                if archive.dry_run:
                    copied += len(await asyncio.to_thread(_files_in, source))
                else:
                    stats = await asyncio.to_thread(_merge_archived, source, local_dir / name / sub,
                                                    conflict_label=archive.name)
                    counts = await asyncio.to_thread(_merge_counts, source, stats)
                    copied, skipped = copied + counts[0], skipped + counts[1]
                taken[f"{name}/{sub}"] = True
            part.fetched += copied
            part.from_archive += copied
            part.present += skipped
            what = (f"{copied} file(s) of pipelines/{_escape(name)}/ ({_escape(', '.join(wanted))}) from this "
                    f"machine's archive")
            callbacks.log(f"  Would copy {what}" if archive.dry_run
                          else f"  [green]Copied {what} ({skipped} here already)[/green]")
        else:
            callbacks.log(f"[cyan]From the archive host {_escape(archive.name)}:[/cyan]")
            fetch = (_dry_fetch(part, archived=True) if archive.dry_run
                     else _Counted(part, archive.digests_below(f"pipelines/{name}"), archived=True))
            missed: list[str] = []
            await base_files._fetch_part(root, session_id, target.profile, listing.root, name, wanted, callbacks,
                                         fetch, merge=_merge_archived, missed=missed)
            for sub in wanted:
                taken[f"{name}/{sub}"] = archive.dry_run or sub not in missed
    return taken


# ---- the export ----

def _read_record(mongo, session_id: str) -> tuple[dict | None, str]:
    """(the session's document or None, and why MongoDB could not be read)."""
    try:
        if hasattr(mongo, "get_session"):
            record = mongo.get_session(session_id)
        else:
            record = mongo.sessions.find_one({"session_id": session_id}, {"_id": 0})
    except Exception as error:  # its text may carry the URL: only its kind is said
        return None, f"MongoDB did not answer ({type(error).__name__})"
    return (record if isinstance(record, dict) and record else None), ""


def parse_only(text: str | None) -> list[str]:
    """the parts --only names (all of them when it names none); ValueError for
    one that is no part."""
    if not str(text or "").strip():
        return list(CATEGORIES)
    names = [name.strip().lower() for name in str(text).split(",") if name.strip()]
    unknown = [name for name in names if name not in CATEGORIES]
    if unknown:
        raise ValueError(f"no such part: {', '.join(unknown)} (the parts are {', '.join(CATEGORIES)})")
    return [name for name in CATEGORIES if name in names]


async def export_session(session_id: str, *, only=None, dry_run: bool = False, all_profiles: bool = False,
                         callbacks=None, mongo=None, record: dict | None = None, influx=None, retention=ASK,
                         database_source: dict | None = None, local_root=None, settings_root=None) -> ExportResult:
    """export one session (the module's docstring says how). `callbacks` has
    the shape of stream_export.ExportCallbacks; `record` is the session's
    document (read from `mongo`, anything with get_session or a `sessions`
    collection, when None); `influx` the InfluxDB client of the measurements
    (None: they are not exported); `retention` the Stream Server's as the
    caller knows it (None: not known; ASK: asked of the server when needed). Hosts that cannot be asked are logged and left out,
    never raised; a part that fails is said and the next one runs."""
    from openmmla.tui import stream_export
    from openmmla.utils.artifact_paths import safe_segment

    callbacks = callbacks or stream_export.ExportCallbacks()
    log = callbacks.log
    root = local_root or local_project_root()
    settings_root = settings_root or _settings_root()
    sid = safe_segment(session_id, "session")
    names = list(only) if only else list(CATEGORIES)
    result = ExportResult(sid, dry_run=dry_run)
    for line in hook_notes():
        log(line)
    log(f"[bold]Exporting session: {_escape(sid)}[/bold] into {_shown_path(root, Path(root) / 'artifacts' / sid)}"
        + (" (dry run: nothing is fetched or written)" if dry_run else ""))
    if record is None and mongo is not None:
        record, error = await asyncio.to_thread(_read_record, mongo, sid)
        if error:
            log(f"  [yellow]{_escape(error)}: what the session's document says is not known.[/yellow]")
    if record is None:
        log(f"  [dim]'{_escape(sid)}' is not in MongoDB (or MongoDB was not reached): its streams and the hosts it "
            f"recorded on are not known from it.[/dim]")
    archive = _Archive(sid, root, settings_root, callbacks, dry_run)
    runners = {
        "measurements": lambda: export_measurements(root, sid, influx, record, callbacks, dry_run=dry_run,
                                                    database_source=database_source),
        "collection": lambda: export_collection(root, sid, record, callbacks, archive, dry_run=dry_run,
                                                all_profiles=all_profiles),
        "streams": lambda: export_streams(root, sid, record, callbacks, archive, dry_run=dry_run,
                                          retention=retention),
        "base": lambda: export_base(root, sid, record, callbacks, archive, dry_run=dry_run),
    }
    for name in names:
        if callbacks.cancelled():
            raise asyncio.CancelledError()
        log(f"[bold cyan]{TITLES[name]}[/bold cyan]")
        callbacks.progress_start(f"Export · {TITLES[name]}", None)
        try:
            result.parts[name] = await runners[name]()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # a part that fails is that part's, the next one runs
            log(f"  [red]✗ {TITLES[name]} failed: {_escape(error)}[/red]")
            result.parts[name] = PartResult(name, missing=[Missing(TITLES[name].lower(), str(error))])
        finally:
            callbacks.progress_end()
    summarize(result, log)
    return result


def summarize(result: ExportResult, log) -> None:
    from openmmla.tui.recordings import human_size

    shown = _escape(result.session_id)
    if result.dry_run:
        log(f"[bold]Dry run of {shown}: nothing was fetched or written[/bold]")
    else:
        status = "complete" if not result.exit_code else "incomplete"
        color = "green" if not result.exit_code else "yellow"
        log(f"[bold {color}]Export of {shown}: {status}[/bold {color}]")
    fetched_word = "to fetch" if result.dry_run else "fetched"
    for name, part in result.parts.items():
        archived = f", {part.from_archive} of them from the archive host" if part.from_archive else ""
        line = (f"  {TITLES[name] + ':':<14}{part.fetched} {fetched_word} ({human_size(part.fetched_bytes)}{archived}), "
                f"{part.present} here already")
        log(_escape(line))
        for item in part.missing:
            color = "red" if item.counted else "dim"
            log(f"    [{color}]missing: {_escape(item.what)}: {_escape(item.why)}[/{color}]")
    if result.exit_code and not result.dry_run:
        log("  [yellow]Export again to fetch what is missing once its host is back: what arrived stays, and is not "
            "fetched again.[/yellow]")


# ---- noting the collection hosts at Start ----

def collection_recorders(host_name: str, params: dict, folder: str) -> list[dict]:
    """one entry per recorder of a host's Collection Start params: the host,
    its folder for the session's recordings, and each recorder's kind, Device
    Label and device."""
    from openmmla.utils.artifact_paths import safe_segment

    label = safe_segment(params.get("--host-label") or host_name, "host")
    entries = []
    for kind, counter in (("audio", "-na"), ("video", "-nv")):
        try:
            count = max(0, int(params.get(counter) or 0))
        except (TypeError, ValueError):
            count = 0
        labels = params.get(f"--{kind}-device-label")
        devices = params.get(f"--{kind}-device")
        for index in range(count):
            device_label = labels[index] if isinstance(labels, (list, tuple)) and index < len(labels) else ""
            device = devices[index] if isinstance(devices, (list, tuple)) and index < len(devices) else devices
            entries.append({"host": host_name, "host_label": label, "folder": str(folder or ""), "kind": kind,
                            "device_label": str(device_label or ""),
                            "device": str(device or "") if not isinstance(device, (list, tuple)) else ""})
    if not entries:
        entries.append({"host": host_name, "host_label": label, "folder": str(folder or ""), "kind": "",
                        "device_label": "", "device": ""})
    return entries


def note_collection_hosts(session_id: str, hosts: list[tuple[str, dict, str]], *, settings_root=None,
                          sessions=None, timeout: float = 2.0) -> str:
    """note in the session's MongoDB document where a Collection Start
    records: `collection_hosts` (the host names, "local" written as this
    console's short name) and `collection_recorders` (each recorder with its
    host's folder and its device), added to what an earlier Start noted. One
    update that never makes a document; never raises. A log line (markup), ""
    when it was noted. `hosts`: (host, its params, its folder) per host."""
    from openmmla.utils.artifact_paths import short_hostname

    names, recorders = [], []
    for host, params, folder in hosts:
        name = short_hostname() if host == "local" else str(host)
        if name not in names:
            names.append(name)
        recorders.extend(collection_recorders(name, params or {}, folder))
    if not session_id or not names:
        return ""
    client = None
    try:
        if sessions is None:
            from openmmla.tui.system_services import system_services_config_path
            from openmmla.utils.stream_recording import _open_sessions, _settings

            start_path = system_services_config_path(settings_root or _settings_root())
            client, sessions = _open_sessions(_settings(None, start_path), timeout)
        update = {"$addToSet": {COLLECTION_HOSTS_FIELD: {"$each": names},
                                COLLECTION_RECORDERS_FIELD: {"$each": recorders}}}
        result = sessions.update_one({"session_id": session_id}, update)
    except Exception as error:  # its text may carry the URL: only its kind is said
        return (f"[yellow]Could not note the recording hosts in MongoDB ({type(error).__name__}): Sessions → Export "
                f"finds them from the manifests, or by asking every SSH profile.[/yellow]")
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()
    if not getattr(result, "matched_count", 0):
        return (f"[yellow]Session {session_id} is not in MongoDB yet: its recording hosts are not noted there "
                f"(Sessions → Export finds them by asking every SSH profile).[/yellow]")
    return ""


# ---- the command ----

def open_influx(settings_root=None):
    """(the InfluxDB client of System Settings, or None, and why not)."""
    from openmmla.tui.system_services import (
        load_system_services_config, section_address_set, system_services_config_path, unset_address_note,
    )

    settings_root = settings_root or _settings_root()
    if not section_address_set((load_system_services_config(settings_root) or {}).get("InfluxDB"), "InfluxDB"):
        return None, unset_address_note("InfluxDB")
    try:
        from openmmla.utils.client.influx_client import InfluxDBClientWrapper
    except ModuleNotFoundError:
        return None, "influxdb-client is not installed here"
    try:
        client = InfluxDBClientWrapper(str(system_services_config_path(settings_root)))
        if not client.client.ping():
            with contextlib.suppress(Exception):
                client.close()
            return None, "InfluxDB did not answer"
    except Exception as error:  # its text may carry the URL: only its kind is said
        return None, f"InfluxDB did not answer ({type(error).__name__})"
    return client, ""


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-export",
        description="Gather everything of a session into this console's artifacts/<session>/: its measurements from "
                    "InfluxDB, its Collection recordings from each recording host, its part of the streams from the "
                    "Stream Server and the capture hosts, and what its bases wrote on their machines. A host that is "
                    "offline is named and skipped, and what the archive host holds of it is fetched from there; a "
                    "file already here is not fetched again.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=120),
    )
    parser.add_argument("session_id", help="the session (its folder here is artifacts/<session>/)")
    parser.add_argument("--only", default="", metavar="PARTS",
                        help=f"comma-separated parts to export: {','.join(CATEGORIES)} (default all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would be fetched and from where, and fetch or write nothing")
    parser.add_argument("--all-profiles", action="store_true",
                        help="also ask every SSH profile for Collection recordings of the session, not only the "
                             "hosts its document or manifest names")
    return parser


def main():
    from openmmla.commands.ses import archive
    from openmmla.utils.artifact_paths import safe_segment

    parser = get_parser()
    args = parser.parse_args()
    try:
        only = parse_only(args.only)
    except ValueError as error:
        parser.error(str(error))
    if not str(args.session_id or "").strip() or safe_segment(args.session_id, "") != args.session_id.strip():
        parser.error(f"'{args.session_id}' is no session id")
    callbacks = archive._ConsoleCallbacks()
    mongo, why = archive.open_mongo()
    if mongo is None:
        callbacks.log(f"[yellow]MongoDB: {_escape(why)}.[/yellow]")
    influx = None
    if "measurements" in only:
        influx, why = open_influx()
        if influx is None:
            callbacks.log(f"[yellow]InfluxDB: {_escape(why)}.[/yellow]")
    from openmmla.tui.system_services import system_services_config_path

    source = {"target": "local", "config": str(system_services_config_path(_settings_root()))}
    try:
        result = asyncio.run(export_session(
            args.session_id.strip(), only=only, dry_run=args.dry_run, all_profiles=args.all_profiles,
            callbacks=callbacks, mongo=mongo, influx=influx, database_source=source))
    except KeyboardInterrupt:
        callbacks.log("[yellow]Stopped. What arrived is kept, and what was staged resumes: run it again to go "
                      "on.[/yellow]")
        sys.exit(130)
    finally:
        for client in (mongo, influx):
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close()
    sys.exit(result.exit_code)
