"""capture-side stream recordings on the machines that capture them: what is
there, and making room.

A managed stream with `record: true` writes one file per run on its capture
host, <record_root>/streams/capture/<YYYY-MM-DD>/<host label>/{video,audio}/
<name>_<start>.<ext> (StreamPanel._record_dir), and nothing there removes it:
Download copies it here and leaves it where it is. This lists those files per
capture host with the room left on its disk, deletes the ones picked, and
prunes a stream's recordings kept longer than its `record_keep_days`.

A file a stream still writes is never deleted: removing it frees nothing
until the stream stops (ffmpeg goes on writing into it) and loses that take.
The host itself tells: a file named on the command line of a running ffmpeg,
or one written in the last LIVE_SECONDS, is being recorded.

The scripts run through bash (stream_cuts.bash) on the capture host, or on
this machine for a local stream, and keep to what bash 3.2 (a Mac's) and the
stat of macOS and of Linux both take; the rest is parsing and planning, which
is tested without a host."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime

from rich.markup import escape

from openmmla.tui.recordings import human_size
from openmmla.tui.ssh import get_profile_by_name, ssh_run_sync
from openmmla.tui.stream_cuts import _quote_root, bash
from openmmla.utils.artifact_paths import (
    CAPTURE_DAY_GLOB, CAPTURE_KINDS, CAPTURE_RECORD_REL, CAPTURE_STREAMS_DIR, STREAMS_DIR, capture_day, is_capture_day,
)

# a file written this recently is being recorded, whatever ps says (the check
# by ffmpeg's command line comes first; this is for when it misses): ffmpeg 7
# writes in 256 KiB blocks, about every 2 s for 1 Mbps video and every 8 s for
# 16 kHz mono audio, 16 s at 8 kHz; a stopped stream's file is free a minute after
LIVE_SECONDS = 60

DAY_SECONDS = 86400

# what `Keep recordings for` offers, in days (the Stream Server card's spans); 0 keeps them
KEEP_CHOICES: list[tuple[str, int]] = [
    ("1 day", 1), ("3 days", 3), ("7 days", 7), ("14 days", 14), ("30 days", 30), ("for ever", 0),
]


@dataclass(frozen=True)
class CaptureFolder:
    """where one capture host files stream recordings: <record_root>/streams/capture/<day>/<host_label>/."""
    ssh_profile: str   # "local" or an SSH profile
    record_root: str   # on that host; $HOME/... for a remote one
    host_label: str

    @property
    def where(self) -> str:
        return "this machine" if self.ssh_profile == "local" else self.ssh_profile


@dataclass(frozen=True)
class CaptureStream:
    """a stream the console runs: whose recordings they are, and how long they stay."""
    name: str
    folder: CaptureFolder
    kind: str           # video | audio
    keep_days: int = 0  # 0: until someone deletes them


@dataclass(frozen=True)
class CaptureFile:
    folder: CaptureFolder
    path: str             # absolute, on the capture host
    day: str              # YYYY-MM-DD, the day folder
    kind: str             # video | audio
    stream: str           # the name before _<start>
    start: float | None   # unix time in the file name; None when it has none
    written: float        # last write (mtime), on the host's clock
    size: int             # bytes
    live: bool            # a running ffmpeg writes it, or it was written moments ago
    writing: bool = False  # a running ffmpeg names it on its command line


@dataclass
class Listing:
    """what one capture host holds, as it said it."""
    folder: CaptureFolder
    files: list[CaptureFile]
    free: int | None   # bytes free on the disk of the record root
    now: float         # the host's clock when it was asked


@dataclass
class Deletion:
    """what a delete script did on one host."""
    folder: CaptureFolder
    removed: list[tuple[str, int]] = field(default_factory=list)  # (path, bytes)
    live: list[str] = field(default_factory=list)      # being recorded: left alone
    gone: list[str] = field(default_factory=list)      # deleted before, by someone else
    refused: list[str] = field(default_factory=list)   # not a stream recording of that folder
    failed: list[str] = field(default_factory=list)
    answered: bool = True                              # False: the host could not be asked

    @property
    def freed(self) -> int:
        return sum(size for _path, size in self.removed)


def by_folder(streams: list[CaptureStream]) -> dict[CaptureFolder, list[CaptureStream]]:
    """the streams grouped by the folder they record into, in their order."""
    folders: dict[CaptureFolder, list[CaptureStream]] = {}
    for stream in streams:
        folders.setdefault(stream.folder, []).append(stream)
    return folders


# ---- on the host ----

# shell variables every script starts from: the tree, the host's clock, and the
# command lines of the ffmpegs running there. pgrep picks them by name: a plain
# `ps | grep ffmpeg` also finds the shell that runs the script, whose command
# line names every file it was given
def _prelude(folder: CaptureFolder) -> str:
    return (
        f"root={_quote_root(folder.record_root)}; label={shlex.quote(folder.host_label)}; "
        f'cap="$root"/{CAPTURE_RECORD_REL}; '
        "now=$(date +%s); "
        "running() { for p in $(pgrep -x ffmpeg 2>/dev/null); do ps -ww -o args= -p \"$p\" 2>/dev/null; done; }; "
        "procs=$(running); "
        # size and mtime: GNU stat, then BSD stat
        "st() { stat -c '%s %Y' \"$1\" 2>/dev/null || stat -f '%z %m' \"$1\" 2>/dev/null; }; "
    )


def inventory_script(folder: CaptureFolder) -> str:
    """print `NOW <host time>`, `FILE <bytes> <mtime> <1 if an ffmpeg names it> <path>`
    for every recording in the folder, `FREE <kB>` of the disk it is on, LISTED when done."""
    return _prelude(folder) + (
        'echo "NOW $now"; '
        f'for f in "$cap"/{CAPTURE_DAY_GLOB}/"$label"/video/* "$cap"/{CAPTURE_DAY_GLOB}/"$label"/audio/*; do '
        '[ -f "$f" ] || continue; s=$(st "$f"); [ -n "$s" ] || continue; w=0; '
        'case "$procs" in *"$f"*) w=1;; esac; '
        'printf "FILE %s %s %s\\n" "$s" "$w" "$f"; done; '
        # the root may not be there yet: its nearest folder that is shares its disk
        'd=$root; while [ ! -d "$d" ] && [ "$d" != / ] && [ "$d" != . ]; do d=$(dirname "$d"); done; '
        "printf 'FREE %s\\n' \"$(df -Pk \"$d\" 2>/dev/null | awk 'NR==2{print $4}')\"; "
        "echo LISTED"
    )


# folders left empty go, day by day, bottom up: never today's, which a stream
# starting now may just have made (by the host's date and by the console's,
# which names the folder), and never one a running ffmpeg writes into.
# streams/capture/ itself stays: a Start's mkdir -p may be making a day in it
def _tidy(console_day: str) -> str:
    return (
        f"procs=$(running); today=$(date +%Y-%m-%d); here={shlex.quote(console_day)}; "
        f'for d in "$cap"/{CAPTURE_DAY_GLOB}; do [ -d "$d" ] || continue; '
        'case "$d" in */"$today"|*/"$here") continue;; esac; '
        'case "$procs" in *"$d/"*) continue;; esac; '
        'c="$d/$label"; rmdir "$c/video" "$c/audio" 2>/dev/null; '
        'rmdir "$c" 2>/dev/null && rmdir "$d" 2>/dev/null; done; '
    )


def delete_script(folder: CaptureFolder, paths: list[str], console_day: str | None = None) -> str:
    """delete these recordings of the folder, each checked again on the host:
    `REMOVED <bytes> <path>`, or LIVE / GONE / REFUSED / FAILED `<path>`; then
    the folders left empty, and DELETED when done. A path has to be a file
    right in <root>/streams/capture/<day>/<label>/video or audio, with no . or
    .. below the root (the root itself may have them)."""
    listed = " ".join(shlex.quote(path) for path in paths)
    console_day = console_day or capture_day()
    return _prelude(folder) + (
        f"for f in {listed}; do "
        f'case "$f" in "$cap"/{CAPTURE_DAY_GLOB}/"$label"/video/*|"$cap"/{CAPTURE_DAY_GLOB}/"$label"/audio/*) ;; '
        '*) echo "REFUSED $f"; continue;; esac; '
        'rest=${f#"$root"/}; case "/$rest/" in */../*|*/./*) echo "REFUSED $f"; continue;; esac; '
        # the day glob is one folder, the label has no /: one more / is a file in a folder below video/ or audio/
        'rest=${f#"$cap"/}; case "$rest" in */*/*/*/*) echo "REFUSED $f"; continue;; esac; '
        'if [ ! -f "$f" ]; then echo "GONE $f"; continue; fi; '
        's=$(st "$f"); m=${s##* }; '
        'case "$procs" in *"$f"*) echo "LIVE $f"; continue;; esac; '
        f'if [ -n "$m" ] && [ $((now - m)) -lt {LIVE_SECONDS} ]; then echo "LIVE $f"; continue; fi; '
        'if rm -f -- "$f" && [ ! -e "$f" ]; then echo "REMOVED ${s%% *} $f"; else echo "FAILED $f"; fi; done; '
        + _tidy(console_day) + "echo DELETED"
    )


# ---- what the host said ----

def _day_folder(path: str) -> tuple[str, str] | None:
    """(YYYY-MM-DD, video|audio) of a recording's path,
    .../streams/capture/<day>/<label>/<kind>/<file>; None for anything else."""
    parts = path.split("/")
    if len(parts) < 7 or parts[-6:-4] != [STREAMS_DIR, CAPTURE_STREAMS_DIR] or not is_capture_day(parts[-4]):
        return None
    if parts[-2] not in CAPTURE_KINDS:
        return None
    return parts[-4], parts[-2]


def _stream_and_start(path: str) -> tuple[str, float | None]:
    """<name>_<start>.<ext>: the stream's name and the start in it."""
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    name, _, start = stem.rpartition("_")
    try:
        return (name, float(start)) if name else (stem, None)
    except ValueError:
        return stem, None


def parse_listing(text: str | None, folder: CaptureFolder) -> Listing | None:
    """None when the listing never finished: the host could not be asked, which is not the same as holding nothing."""
    lines = [line.rstrip("\r") for line in str(text or "").splitlines()]
    if "LISTED" not in (line.strip() for line in lines):
        return None
    files: list[CaptureFile] = []
    free, now = None, None
    for line in lines:
        tag, _, rest = line.partition(" ")
        if tag == "NOW" and rest.strip().isdigit():
            now = float(rest.strip())
        elif tag == "FREE" and rest.strip().isdigit():
            free = int(rest.strip()) * 1024
        elif tag == "FILE":
            fields = rest.split(" ", 3)
            if len(fields) != 4 or not (fields[0].isdigit() and fields[1].isdigit()):
                continue
            size, written, live, path = fields
            where = _day_folder(path)
            if not path.startswith("/") or where is None:
                continue
            stream, start = _stream_and_start(path)
            files.append(CaptureFile(folder, path, where[0], where[1], stream, start, float(written), int(size),
                                     live == "1", writing=live == "1"))
    if now is None:
        now = max([f.written for f in files], default=0.0)
    listing = Listing(folder, files, free, now)
    # a file written moments ago is being recorded, whether or not an ffmpeg names it
    listing.files = [
        f if f.live or now - f.written >= LIVE_SECONDS else replace(f, live=True)
        for f in files
    ]
    return listing


def parse_deletion(text: str | None, folder: CaptureFolder) -> Deletion:
    lines = [line.rstrip("\r") for line in str(text or "").splitlines()]
    result = Deletion(folder, answered="DELETED" in (line.strip() for line in lines))
    for line in lines:
        tag, _, rest = line.partition(" ")
        if tag == "REMOVED":
            size, _, path = rest.partition(" ")
            if size.isdigit() and path:
                result.removed.append((path, int(size)))
        elif tag in ("LIVE", "GONE", "REFUSED", "FAILED") and rest:
            getattr(result, tag.lower()).append(rest)
    return result


# ---- what to delete ----

def is_live(item: CaptureFile, live_paths=()) -> bool:
    """being recorded, as the host says or as the console noted it started."""
    return item.live or item.path in set(live_paths or ())


def due(listing: Listing, streams: list[CaptureStream], live_paths=()) -> list[CaptureFile]:
    """the recordings of these streams kept longer than their keep_days, counted
    from the last write on the host's clock. A file is matched by the stream's
    name and kind; never one being recorded."""
    keep = {(s.name, s.kind): s.keep_days for s in streams if s.folder == listing.folder and s.keep_days > 0}
    picked = []
    for item in listing.files:
        days = keep.get((item.stream, item.kind), 0)
        if days and not is_live(item, live_paths) and listing.now - item.written >= days * DAY_SECONDS:
            picked.append(item)
    return picked


def expires(item: CaptureFile, streams: list[CaptureStream]) -> float | None:
    """when a recording is past its stream's keep time; None when it is kept."""
    for stream in streams:
        if (stream.folder, stream.name, stream.kind) == (item.folder, item.stream, item.kind) and stream.keep_days > 0:
            return item.written + stream.keep_days * DAY_SECONDS
    return None


def older_than(listings: list[Listing], seconds: float, live_paths=()) -> list[CaptureFile]:
    """every recording last written longer ago than that, on its host's clock."""
    return [
        item for listing in listings for item in listing.files
        if not is_live(item, live_paths) and listing.now - item.written >= seconds
    ]


# ---- running it ----

def run_script(ssh_profile: str, script: str, timeout: float = 30.0) -> str | None:
    """stdout of a script run through bash on a capture host ("local": this
    machine); None when it could not be run there."""
    command = bash(script)
    try:
        if ssh_profile == "local":
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        else:
            profile = get_profile_by_name(ssh_profile)
            if profile is None:
                return None
            result = ssh_run_sync(profile, command, timeout=timeout)
    except Exception:
        return None
    return result.stdout or ""


def list_folder(folder: CaptureFolder, timeout: float = 30.0) -> Listing | None:
    return parse_listing(run_script(folder.ssh_profile, inventory_script(folder), timeout), folder)


def delete(folder: CaptureFolder, paths: list[str], timeout: float = 60.0) -> Deletion:
    if not paths:
        return Deletion(folder)
    return parse_deletion(run_script(folder.ssh_profile, delete_script(folder, paths), timeout), folder)


def delete_files(items: list[CaptureFile]) -> list[Deletion]:
    """delete the picked recordings, one script per host."""
    folders: dict[CaptureFolder, list[str]] = {}
    for item in items:
        folders.setdefault(item.folder, []).append(item.path)
    return [delete(folder, paths) for folder, paths in folders.items()]


@dataclass
class Pruned:
    """what a stream's keep time removed from one host."""
    folder: CaptureFolder
    deletion: Deletion | None = None   # None: nothing was due
    unreachable: bool = False


def prune(streams: list[CaptureStream], live_paths=()) -> list[Pruned]:
    """delete what these streams recorded longer ago than their keep_days,
    folder by folder; a folder with nothing due gets no delete script."""
    results = []
    for folder, owned in by_folder([s for s in streams if s.keep_days > 0]).items():
        listing = list_folder(folder)
        if listing is None:
            results.append(Pruned(folder, unreachable=True))
            continue
        picked = due(listing, owned, live_paths)
        results.append(Pruned(folder, delete(folder, [item.path for item in picked]) if picked else None))
    return results


def prune_report(results: list[Pruned], streams: list[CaptureStream]) -> list[str]:
    """log lines for the Streams tab: only what was removed, or could not be."""
    lines = []
    keep = {s.name: s.keep_days for s in streams}
    for result in results:
        where = result.folder.where
        if result.unreachable:
            lines.append(f"[yellow]Could not ask {escape(where)} for recordings past their keep time; "
                         "they stay until the next Start or Refresh.[/yellow]")
            continue
        deletion = result.deletion
        if deletion is None:
            continue
        if not deletion.answered:
            lines.append(f"[yellow]{escape(where)} did not finish deleting the recordings past their keep time.[/yellow]")
        if deletion.removed:
            names = sorted({_stream_and_start(path)[0] for path, _size in deletion.removed})
            spans = ", ".join(f"{escape(name)} ({keep.get(name, '?')} d)" for name in names)
            lines.append(
                f"[cyan]Removed {len(deletion.removed)} recording(s) past their keep time on {escape(where)}, "
                f"{human_size(deletion.freed)}: {spans}.[/cyan]")
        for path in deletion.failed:
            lines.append(f"[red]Could not delete {escape(path)} on {escape(where)}.[/red]")
        if deletion.refused:
            lines.append(
                f"[red]{len(deletion.refused)} recording(s) past their keep time on {escape(where)} were not "
                f"deleted: they are not where the stream records ({escape(deletion.refused[0])}).[/red]")
    return lines


# ---- telling it ----

def day_label(day: str) -> str:
    """the day of a row: its folder's name, 2026-09-19."""
    return str(day or "-")


def stamp(moment: float | None) -> str:
    return datetime.fromtimestamp(moment).strftime("%m-%d %H:%M") if moment else "-"


def span(seconds: float | None) -> str:
    """`42 s`, `23 min`, `2 h 05`, `3 d 4 h`."""
    if seconds is None or seconds < 0:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min"
    if seconds < DAY_SECONDS:
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d}"
    return f"{seconds // DAY_SECONDS} d {seconds % DAY_SECONDS // 3600} h"
