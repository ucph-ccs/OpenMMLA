"""mmla ses-archive: a session's raw files, sent from this console's checkout
to the System Settings host, where the dashboard and replays read them.

A session's files reach the console when its recordings are downloaded
(Collection card, Download) and its streams and base files exported (Sessions
tab), often the day after it ran. Archive (the Sessions tab's button, or this
command) sends what is raw of artifacts/<session>/ to the same place in the
checkout of the archive host: the Dashboard's host of System Settings, else
its Stream Server's, or the SSH profile --host names. Nothing runs by itself,
and nothing is deleted, here or there.

What is sent: collection/ (every host), streams/ (the exported stream cuts),
raw/, pipelines/<pipeline>/<host>/config/ and logger/ (with the JSON of what
each component ran with) and measurements/<session>_parameters.json; with
--with-runtime-media also what the bases stored (real-time/, post-time/,
visualizations/). A profiles/ folder (speaker profiles: voices) is never sent.
A file already there with the same sha256 is not sent again; a different file
of the same name there is kept, and this one goes beside it as
<name>_<console host><ext>, as artifacts.merge_tree keeps both here. A host's
manifest or a config.yml too, which merge_tree replaces here: the copy there
may hold rows this one lacks (another console, or a fuller download, archived
the session first), and nothing there is overwritten.

Files arrive in <session>/.archive/incoming/ on the archive host, with rsync
(resumable) or scp, and each is moved into place only once its sha256 there is
the one it has here; one whose sum differs is dropped and named. Then, on that
host: the session's part of the Stream Server's recordings is cut from its own
playback server into streams/server/, the session's manifest is rewritten with
the paths the files have there (relpath, bytes, sha256 and origin added to each
recording; the stream cuts listed under `stream_cuts`, not as recordings), and
the session's MongoDB document gets an `archive` field
(location, status, files, bytes, verified_at).

The archive host's part lies between the two marks below: standard library
only, sent over SSH with each request and run there by python3, so it does not
depend on what that host's checkout holds.

Test hooks: OPENMMLA_ARCHIVE_LOCAL_ROOT is the project root whose artifacts/
is sent (default this checkout), OPENMMLA_ARCHIVE_REMOTE_ROOT the project root
on the archive host (default its SSH profile's remote_project_path). The log
says so, in yellow, whenever one of them is set."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import copy
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ENV_LOCAL_ROOT = "OPENMMLA_ARCHIVE_LOCAL_ROOT"
ENV_REMOTE_ROOT = "OPENMMLA_ARCHIVE_REMOTE_ROOT"

ARTIFACTS_DIR = "artifacts"
# what is sent of artifacts/<session>/: these folders whole, two sections of
# each pipelines/<pipeline>/<host>/, and the session's parameters
RAW_FOLDERS = ("collection", "streams", "raw")
PIPELINE_SECTIONS = ("config", "logger")
# what the bases stored: sent with --with-runtime-media only
RUNTIME_SECTIONS = ("real-time", "post-time", "visualizations")
# real-time/temp/: frames on their way to a server, never anything to keep
TRANSIENT_SECTION = "temp"
# speaker profiles hold voices: they stay with the host that uses them
PRIVATE_FOLDER = "profiles"
PARAMETERS_SUFFIX = "_parameters.json"
SESSION_MANIFESTS = ("manifest.json", "manifest.yml")
# the session manifest's list of the Stream Server cuts the archive made there,
# kept apart from `recordings` (the collection's, by host and device)
STREAM_CUTS_KEY = "stream_cuts"
_STREAM_KIND = "stream"


class ArchiveError(Exception):
    """the archive could not be made at all: no folder here, no host to send
    to, or the archive host could not be asked."""


# ---- archive host: begin ----
# everything from here to the end mark also runs on the archive host, as a
# program of its own (_remote_program): standard library only, and nothing
# from outside this section but the imports _PROGRAM_HEADER makes

_ARCHIVE_DIR = ".archive"
_INCOMING_DIR = "incoming"
_LEDGER_NAME = "ledger.json"
_REPLY = "ARCHIVE-REPLY "
_PROGRESS = "ARCHIVE-PROGRESS "
_CHUNK = 1024 * 1024
# a stream cut written this long after its window closed holds all of it
_CUT_SETTLE_SECONDS = 5.0
_MAX_CONFLICTS = 1000


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _expand(path):
    """a path as the archive host spells it: ~ and $HOME expanded, absolute."""
    text = str(path or "").strip()
    if text == "$HOME" or text.startswith("$HOME/"):
        text = "~" + text[len("$HOME"):]
    return os.path.abspath(os.path.expanduser(text))


def _sha256_file(path, progress=None):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            if progress is not None:
                progress(len(chunk))
    return digest.hexdigest()


def _rel_ok(rel):
    """a path relative to the session folder that names nothing outside it."""
    text = str(rel or "")
    if not text or text.startswith("/") or any(char in text for char in "\\\n\r\0"):
        return False
    return all(part not in ("", ".", "..") for part in text.split("/"))


def _inside(real_root, path):
    real = os.path.realpath(path)
    return real == real_root or real.startswith(real_root.rstrip(os.sep) + os.sep)


def _conflict_rel(rel, label, index):
    """<stem>_<label><ext>, then <stem>_<label>_2<ext>, ...: the name
    artifacts.merge_tree gives a copy that differs from the file there."""
    head, _, name = rel.rpartition("/")
    stem, ext = os.path.splitext(name)
    renamed = "%s_%s%s" % (stem, label, ext) if index == 1 else "%s_%s_%d%s" % (stem, label, index, ext)
    return head + "/" + renamed if head else renamed


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return None


def _manifest_texts(session_dir):
    """the session's manifest.json and manifest.yml as written, each with its
    mtime; None for one that is not there."""
    found = {}
    for name in ("manifest.json", "manifest.yml"):
        path = os.path.join(session_dir, name)
        text = _read_text(path)
        found[name] = None if text is None else {"text": text, "mtime": os.path.getmtime(path)}
    return found


def _prune_empty(folder):
    for root, _dirs, _files in os.walk(folder, topdown=False):
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass


class _Progress:
    """progress lines for the console, two a second at most."""

    def __init__(self, total, label):
        self.total = int(total or 0)
        self.done = 0
        self.label = label
        self.shown = 0.0

    def add(self, count, detail=""):
        self.done += count
        now = time.monotonic()
        if now - self.shown >= 0.5:
            self.shown = now
            sys.stdout.write(_PROGRESS + json.dumps([self.done, self.total, detail or self.label]) + "\n")
            sys.stdout.flush()


class _Ledger:
    """the sha256 of each file the archive put or found in a session folder,
    with its size and mtime then: a file untouched since is not read again."""

    def __init__(self, session_dir):
        self.path = os.path.join(session_dir, _ARCHIVE_DIR, _LEDGER_NAME)
        self.items = {}
        self.changed = False
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                self.items = data
        except (OSError, ValueError):
            pass

    def sha256(self, session_dir, rel, progress=None):
        stat = os.stat(os.path.join(session_dir, rel))
        known = self.items.get(rel)
        if (isinstance(known, dict) and known.get("sha256") and known.get("size") == stat.st_size
                and known.get("mtime_ns") == stat.st_mtime_ns):
            if progress is not None:
                progress(stat.st_size)
            return known["sha256"]
        digest = _sha256_file(os.path.join(session_dir, rel), progress)
        self.note(session_dir, rel, digest)
        return digest

    def note(self, session_dir, rel, digest, verified_at=None):
        stat = os.stat(os.path.join(session_dir, rel))
        entry = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
        if verified_at:
            entry["verified_at"] = verified_at
        self.items[rel] = entry
        self.changed = True

    def save(self):
        if not self.changed:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        partial = self.path + ".tmp"
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump(self.items, handle, indent=1, sort_keys=True)
        os.replace(partial, self.path)
        self.changed = False


def _op_check(request):
    """where each file goes there, and whether it is there already: `send`
    (to its own name, or beside a different file of that name), `skip` (a
    file with its sha256 is there) or `refused`. The staging folders of the
    files to send are made, and a staged copy longer than the file dropped,
    except in a dry run, which changes nothing."""
    session_dir = _expand(request["session_dir"])
    dry = bool(request.get("dry_run"))
    label = str(request.get("label") or "console")
    real = os.path.realpath(session_dir)
    incoming = os.path.join(session_dir, _ARCHIVE_DIR, _INCOMING_DIR)
    ledger = _Ledger(session_dir)
    items = request.get("files") or []
    progress = _Progress(sum(int(item.get("size") or 0) for item in items), "comparing")
    taken = set()
    files = {}
    for item in items:
        rel = str(item.get("rel") or "")
        size = int(item.get("size") or 0)
        digest = str(item.get("sha256") or "")
        decision = None if _rel_ok(rel) else {"action": "refused", "why": "a name the archive host cannot take"}
        index = 0
        while decision is None:
            if index > _MAX_CONFLICTS:
                decision = {"action": "refused", "why": "too many files of that name there"}
                break
            target = rel if index == 0 else _conflict_rel(rel, label, index)
            path = os.path.join(session_dir, target)
            index += 1
            if not _inside(real, path):
                decision = {"action": "refused", "why": "it would land outside the session folder"}
            elif target in taken or os.path.islink(path) or (os.path.lexists(path) and not os.path.isfile(path)):
                continue
            elif not os.path.lexists(path):
                decision = {"action": "send", "target": target}
            elif os.path.getsize(path) == size and ledger.sha256(session_dir, target, progress.add) == digest:
                decision = {"action": "skip", "target": target}
        if decision["action"] == "send":
            taken.add(decision["target"])
            staged = os.path.join(incoming, decision["target"])
            staged_size = os.path.getsize(staged) if os.path.isfile(staged) else None
            if staged_size is not None and staged_size > size and not dry:
                os.unlink(staged)
                staged_size = None
            decision["staged"] = staged_size
            if not dry:
                os.makedirs(os.path.dirname(staged), exist_ok=True)
        files[rel] = decision
    if not dry:
        ledger.save()
    probe = session_dir
    while not os.path.isdir(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    return {
        "session_dir": session_dir,
        "exists": os.path.isdir(session_dir),
        "free": shutil.disk_usage(probe).free,
        "has_rsync": bool(shutil.which("rsync")),
        "python": sys.version.split()[0],
        "files": files,
        "manifest": _manifest_texts(session_dir),
    }


def _op_place(request):
    """check each staged file by its sha256 and move it into place: one that
    differs is dropped; one whose place was taken since the check goes beside
    it, and one whose content is there already is dropped as well."""
    session_dir = _expand(request["session_dir"])
    real = os.path.realpath(session_dir)
    incoming = os.path.join(session_dir, _ARCHIVE_DIR, _INCOMING_DIR)
    label = str(request.get("label") or "console")
    ledger = _Ledger(session_dir)
    items = request.get("items") or []
    progress = _Progress(sum(int(item.get("size") or 0) for item in items), "checking")
    now = _utc_now()
    placed = {}
    for item in items:
        rel, target = str(item.get("rel") or ""), str(item.get("target") or "")
        size, digest = int(item.get("size") or 0), str(item.get("sha256") or "")
        staged = os.path.join(incoming, target)
        if not (_rel_ok(rel) and _rel_ok(target) and _inside(real, os.path.join(session_dir, target))):
            placed[rel] = {"ok": False, "why": "a name the archive host cannot take"}
            continue
        if not os.path.isfile(staged):
            placed[rel] = {"ok": False, "why": "it did not arrive"}
            continue
        got = _sha256_file(staged, progress.add)
        if got != digest:
            os.unlink(staged)
            placed[rel] = {"ok": False, "why": "its sha256 there differs", "sha256": got}
            continue
        final, index, same = target, 0, False
        while True:
            path = os.path.join(session_dir, final)
            if not os.path.lexists(path):
                break
            if os.path.isfile(path) and not os.path.islink(path):
                if os.path.getsize(path) == size and ledger.sha256(session_dir, final) == digest:
                    same = True
                    break
            index += 1
            if index > _MAX_CONFLICTS:
                break
            final = _conflict_rel(rel, label, index)
        if index > _MAX_CONFLICTS:
            placed[rel] = {"ok": False, "why": "too many files of that name there"}
            continue
        if same:
            os.unlink(staged)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.replace(staged, path)
        ledger.note(session_dir, final, digest, verified_at=now)
        placed[rel] = {"ok": True, "final": final, "bytes": os.path.getsize(path), "sha256": digest}
    _prune_empty(incoming)
    ledger.save()
    return {"placed": placed, "verified_at": now}


def _download(url, destination, timeout, progress):
    """one clip of the playback server into place; nothing is left behind by
    a download that fails."""
    partial = destination + ".part"
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(partial, "wb") as handle:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                progress(len(chunk))
        os.replace(partial, destination)
    except BaseException:
        try:
            os.unlink(partial)
        except OSError:
            pass
        raise


def _op_cut(request):
    """the session's part of the Stream Server's recordings, from the playback
    server as this host reaches it: a clip already there and written after
    its window closed is kept, a shorter one (cut while the session went on)
    replaced."""
    session_dir = _expand(request["session_dir"])
    real = os.path.realpath(session_dir)
    ledger = _Ledger(session_dir)
    origins = [str(origin).rstrip("/") for origin in request.get("origins") or []]
    timeout = float(request.get("timeout") or 60.0)
    progress = _Progress(0, "cutting")
    done = []
    for clip in request.get("clips") or []:
        rel = str(clip.get("rel") or "")
        path = os.path.join(session_dir, rel)
        if not (_rel_ok(rel) and _inside(real, path)):
            done.append({"rel": rel, "status": "failed", "why": "a name the archive host cannot take"})
            continue
        status = "cut"
        if os.path.isfile(path) and not os.path.islink(path):
            if (os.path.getsize(path) > 0
                    and os.path.getmtime(path) >= float(clip.get("end") or 0) + _CUT_SETTLE_SECONDS):
                done.append({"rel": rel, "status": "kept", "bytes": os.path.getsize(path),
                             "sha256": ledger.sha256(session_dir, rel)})
                continue
            status = "replaced"
        why = "no address of the playback server was given"
        for origin in list(origins):
            try:
                _download(origin + "/get?" + str(clip.get("query") or ""), path, timeout,
                          lambda count, rel=rel: progress.add(count, rel))
            except urllib.error.HTTPError as error:
                why = "HTTP %d from %s" % (error.code, origin)
                break  # the server answered: another of its addresses says the same
            except (urllib.error.URLError, OSError) as error:
                why = "%s: %s" % (origin, getattr(error, "reason", error))
                continue
            why = None
            origins.remove(origin)
            origins.insert(0, origin)
            break
        if why is not None:
            done.append({"rel": rel, "status": "failed", "why": why})
            continue
        digest = _sha256_file(path)
        ledger.note(session_dir, rel, digest, verified_at=_utc_now())
        done.append({"rel": rel, "status": status, "bytes": os.path.getsize(path), "sha256": digest})
    ledger.save()
    return {"clips": done}


def _op_write(request):
    """write the session's manifest.json and manifest.yml, unless they changed
    since they were read (then they are handed back, to be merged again)."""
    session_dir = _expand(request["session_dir"])
    current = _manifest_texts(session_dir)
    expected = request.get("expected") or {}
    for name in ("manifest.json", "manifest.yml"):
        if (current.get(name) or {}).get("text") != expected.get(name):
            return {"changed": True, "manifest": current}
    os.makedirs(session_dir, exist_ok=True)
    for name, key in (("manifest.yml", "yml"), ("manifest.json", "json")):
        partial = os.path.join(session_dir, ".%s.archive-tmp" % name)
        with open(partial, "w", encoding="utf-8") as handle:
            handle.write(request[key])
        os.replace(partial, os.path.join(session_dir, name))
    return {"written": True}


_OPS = {"check": _op_check, "place": _op_place, "cut": _op_cut, "write": _op_write}


def _remote_main(encoded):
    try:
        request = json.loads(base64.b64decode(encoded).decode("utf-8"))
        reply = _OPS[request["op"]](request)
    except Exception as error:  # the console says it, and stops
        reply = {"error": "%s: %s" % (type(error).__name__, error)}
    sys.stdout.write(_REPLY + json.dumps(reply) + "\n")
    sys.stdout.flush()

# ---- archive host: end ----


_BEGIN_MARK = "# ---- archive host: " + "begin ----"
_END_MARK = "# ---- archive host: " + "end ----"
_PROGRAM_HEADER = (
    "from __future__ import annotations\n"
    "import base64, hashlib, json, os, shutil, sys, time\n"
    "import urllib.error, urllib.request\n"
)
# the archive host's python3, reading its program on stdin
_REMOTE_PYTHON = ("command -v python3 >/dev/null 2>&1 || { echo 'python3 is not installed there' >&2; exit 127; }; "
                  "exec python3 -")
# what the program's reply line may weigh: the check of a large session lists every file
_READ_LIMIT = 256 * 1024 * 1024
# a staging path rsync takes as it is (it passes it through a remote shell)
_SIMPLE_PATH = re.compile(r"^/[A-Za-z0-9_@%+=:,./-]*$")
# seconds between two looks at how much has arrived
_POLL_SECONDS = 3.0


@functools.lru_cache(maxsize=1)
def _remote_section() -> str:
    source = Path(__file__).read_text(encoding="utf-8")
    return source[source.index(_BEGIN_MARK):source.index(_END_MARK)]


def _remote_program(request: dict) -> str:
    """the archive host's part of this file, run with one request."""
    encoded = base64.b64encode(json.dumps(request, default=str).encode("utf-8")).decode("ascii")
    return f"{_PROGRAM_HEADER}{_remote_section()}\n_remote_main({encoded!r})\n"


# ---- where the archive goes ----

def _settings_root():
    from openmmla.tui.schema.loader import _find_project_root

    return _find_project_root()


def local_project_root():
    """the project root whose artifacts/ is archived: OPENMMLA_ARCHIVE_LOCAL_ROOT,
    else this checkout."""
    configured = os.environ.get(ENV_LOCAL_ROOT, "").strip()
    return os.path.abspath(os.path.expanduser(configured)) if configured else _settings_root()


def hook_notes(local_root=None) -> list[str]:
    """a log line (markup) for each test hook that moves the archive: the
    environment is inherited, and a stray variable would send another folder,
    or into another one, without a word."""
    from rich.markup import escape

    notes = []
    local = os.environ.get(ENV_LOCAL_ROOT, "").strip()
    if local and local_root is None:
        notes.append(f"  [yellow]{ENV_LOCAL_ROOT} is set: the session folder is read under {escape(local)}, "
                     f"not this checkout.[/yellow]")
    remote = os.environ.get(ENV_REMOTE_ROOT, "").strip()
    if remote:
        notes.append(f"  [yellow]{ENV_REMOTE_ROOT} is set: the archive goes under {escape(remote)} on the archive "
                     f"host, not its checkout.[/yellow]")
    return notes


def local_session_dir(session_id: str, project_root=None) -> Path:
    from openmmla.utils.artifact_paths import safe_segment

    return Path(project_root or local_project_root()) / ARTIFACTS_DIR / safe_segment(session_id, "session")


@dataclass
class ArchiveTarget:
    """the host an archive goes to."""
    name: str          # its SSH profile; this machine's short name when it is this machine
    profile: object    # the SSH profile; None when this machine is the archive host
    root: str          # the project root there, as written (~/OpenMMLA)
    via: str           # what named it

    @property
    def here(self) -> bool:
        return self.profile is None

    def session_path(self, session_id: str) -> str:
        return f"{self.root.rstrip('/')}/{ARTIFACTS_DIR}/{session_id}"

    def location(self, session_id: str) -> str:
        return f"{self.name}:{self.session_path(session_id)}"


def resolve_target(host: str | None = None, settings_root=None) -> ArchiveTarget:
    """the archive host: the SSH profile `host` names ('local' for this
    machine, or a host name an SSH profile reaches), else the Dashboard's host
    of System Settings, else its Stream Server's. Blocking: host names may be
    resolved."""
    from openmmla.tui.ssh import load_ssh_profiles
    from openmmla.tui.system_services import (
        load_system_services_config, stream_server_section, target_for_service_host, usable_system_service_value,
    )
    from openmmla.utils.artifact_paths import short_hostname

    settings_root = settings_root or _settings_root()
    profiles = load_ssh_profiles()
    remote_root = os.environ.get(ENV_REMOTE_ROOT, "").strip()
    host = str(host or "").strip()
    if host:
        via = "--host"
        if host == "local" or any(profile.name == host for profile in profiles):
            named = host
        else:
            named = target_for_service_host(host, profiles)
        if not named:
            raise ArchiveError(f"'{host}' is no SSH profile, and none reaches it: add it under System Settings → "
                               f"Hosts → SSH Profiles")
    else:
        config = load_system_services_config(settings_root) or {}
        value = via = ""
        for section, label in (("Dashboard", "Dashboard"), ("StreamServer", "Stream Server")):
            fields = stream_server_section(config) if section == "StreamServer" else config.get(section)
            if isinstance(fields, dict) and usable_system_service_value(fields.get("host")):
                value, via = str(fields["host"]).strip(), f"System Settings → {label}"
                break
        if not value:
            raise ArchiveError("System Settings name no host for the Dashboard or the Stream Server: set one under "
                               "System Settings → Connections, or name the archive host with --host")
        named = target_for_service_host(value, profiles)
        if not named:
            raise ArchiveError(f"No SSH profile reaches {value} ({via}): add one under System Settings → Hosts → "
                               f"SSH Profiles, or name the archive host with --host")
    if named == "local":
        return ArchiveTarget(short_hostname(), None, remote_root or str(settings_root), via)
    profile = next(profile for profile in profiles if profile.name == named)
    return ArchiveTarget(profile.name, profile, remote_root or profile.remote_project_path or "~/OpenMMLA", via)


# ---- what is sent ----

@dataclass
class LocalFile:
    rel: str           # below artifacts/<session>/, with /
    size: int
    sha256: str = ""


@dataclass
class Selection:
    """what of a session folder is sent, and what stays here."""
    files: list[LocalFile] = field(default_factory=list)
    profiles: int = 0                                      # files under a profiles/ folder: never sent
    runtime: int = 0                                       # what the bases stored, without --with-runtime-media
    runtime_bytes: int = 0
    links: list[str] = field(default_factory=list)         # symbolic links: not followed
    unnamed: list[str] = field(default_factory=list)       # names that cannot be written in UTF-8
    left: dict[str, int] = field(default_factory=dict)     # folder -> files that are not raw (derived, exports)


def _kind_of(parts: list[str], with_runtime_media: bool) -> str:
    """send, profiles, runtime, transient or left: what becomes of a file at
    artifacts/<session>/<parts>."""
    if PRIVATE_FOLDER in parts[:-1]:
        return "profiles"
    top = parts[0]
    if top in RAW_FOLDERS and len(parts) > 1:
        return "send"
    if top == "pipelines" and len(parts) >= 4:
        section = parts[3]
        if len(parts) == 4 or section in PIPELINE_SECTIONS:
            return "send"
        if section in RUNTIME_SECTIONS:
            if section == "real-time" and len(parts) > 5 and parts[4] == TRANSIENT_SECTION:
                return "transient"
            return "send" if with_runtime_media else "runtime"
    if top == "measurements" and len(parts) == 2 and parts[1].endswith(PARAMETERS_SUFFIX):
        return "send"
    return "left"


def select_files(session_dir, with_runtime_media: bool = False) -> Selection:
    """the files of a session folder to send, in a stable order. Hidden files
    and folders (staging, .DS_Store) and half-written ones (*.tmp, *.part) are
    passed over; a symbolic link is not followed."""
    session_dir = Path(session_dir)
    chosen = Selection()
    for root, dirs, names in os.walk(session_dir):
        rel_root = os.path.relpath(root, session_dir)
        parts_root = [] if rel_root == "." else rel_root.split(os.sep)
        kept = []
        for name in sorted(dirs):
            if name.startswith(".") or name == "__pycache__":
                continue
            if os.path.islink(os.path.join(root, name)):
                chosen.links.append("/".join(parts_root + [name]))
                continue
            kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            parts = parts_root + [name]
            rel = "/".join(parts)
            path = os.path.join(root, name)
            if name.startswith(".") or name.endswith((".tmp", ".part")):
                continue
            if len(parts) == 1 and name in SESSION_MANIFESTS:
                continue  # the session's manifest is merged there, not sent
            if os.path.islink(path):
                chosen.links.append(rel)
                continue
            kind = _kind_of(parts, with_runtime_media)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if kind == "send":
                try:
                    rel.encode("utf-8")
                except UnicodeEncodeError:
                    chosen.unnamed.append(rel.encode("utf-8", "replace").decode("utf-8"))
                    continue
                chosen.files.append(LocalFile(rel, size))
            elif kind == "profiles":
                chosen.profiles += 1
            elif kind == "runtime":
                chosen.runtime += 1
                chosen.runtime_bytes += size
            elif kind == "left":
                folder = f"{parts[0]}/" if len(parts) > 1 else "(top level)"
                chosen.left[folder] = chosen.left.get(folder, 0) + 1
    return chosen


# ---- talking to the archive host ----

class _Quiet:
    """callbacks that say nothing (stream_export.ExportCallbacks has the same shape)."""

    def log(self, _line: str) -> None:
        pass

    def progress_start(self, _label: str, _total) -> None:
        pass

    def progress_update(self, _done: int, _total: int, _detail: str) -> None:
        pass

    def progress_end(self) -> None:
        pass

    def cancelled(self) -> bool:
        return False


def _stop_if_cancelled(callbacks) -> None:
    if callbacks.cancelled():
        raise asyncio.CancelledError()


def _kill_group(proc, sig) -> None:
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), sig)


async def _end_process(proc) -> None:
    """stop a child that is still running, and everything it started (ssh,
    sshpass): it runs in a process group of its own."""
    if proc.returncode is not None:
        return
    _kill_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), 3)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        _kill_group(proc, signal.SIGKILL)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


async def _ask(target: ArchiveTarget, request: dict, *, timeout: float = 6 * 3600.0, progress=None) -> dict:
    """run one request on the archive host (this machine's own python when it
    is this machine) and return its reply; `progress(done, total, detail)`
    gets the program's progress lines. Cancelled, it stops the program."""
    program = _remote_program(request).encode("utf-8")
    if target.here:
        argv = [sys.executable or "python3", "-"]
    else:
        argv = [*target.profile.base_ssh_args(), _REMOTE_PYTHON]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True, limit=_READ_LIMIT)
    except OSError as error:
        raise ArchiveError(f"{target.name} could not be asked: {error.strerror or type(error).__name__}") from None
    reply: dict | None = None
    errors: list[str] = []

    async def feed() -> None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            proc.stdin.write(program)
            await proc.stdin.drain()
        proc.stdin.close()

    async def read_replies() -> None:
        nonlocal reply
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip("\r\n")
            if text.startswith(_REPLY):
                with contextlib.suppress(ValueError):
                    reply = json.loads(text[len(_REPLY):])
            elif text.startswith(_PROGRESS) and progress is not None:
                with contextlib.suppress(ValueError, TypeError):
                    progress(*json.loads(text[len(_PROGRESS):]))

    async def read_errors() -> None:
        errors.append((await proc.stderr.read()).decode("utf-8", "replace"))

    try:
        await asyncio.wait_for(asyncio.gather(feed(), read_replies(), read_errors()), timeout)
        code = await proc.wait()
    except asyncio.TimeoutError:
        raise ArchiveError(f"{target.name} did not answer within {timeout:g} s") from None
    finally:
        await _end_process(proc)
    if not isinstance(reply, dict):
        last = _last_line("".join(errors))
        raise ArchiveError(f"{target.name} could not be asked (exit {code})" + (f": {last}" if last else ""))
    if reply.get("error"):
        raise ArchiveError(f"{target.name}: {reply['error']}")
    return reply


async def _run_quiet(argv: list[str]) -> tuple[int, str]:
    """a transfer child in a process group of its own: (exit code, output).
    Its argument list is never shown: with a password in the profile it
    starts with sshpass -p <password>."""
    from openmmla.tui import download as dl

    return await dl._run_child(argv)


async def _staged_bytes(target: ArchiveTarget, incoming: str) -> int | None:
    """how much lies in the staging folder there now; None when it could not be asked."""
    script = ("import os,sys\n"
              "print(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(sys.argv[1]) for f in fs))")
    if target.here:
        argv = [sys.executable or "python3", "-c", script, incoming]
    else:
        argv = [*target.profile.base_ssh_args(), f"python3 -c {shlex.quote(script)} {shlex.quote(incoming)}"]
    try:
        code, output = await asyncio.wait_for(_run_quiet(argv), 30)
    except asyncio.TimeoutError:
        return None
    try:
        return int(output.strip().splitlines()[-1]) if code == 0 else None
    except (ValueError, IndexError):
        return None


async def _poll_staged(target: ArchiveTarget, incoming: str, total: int, callbacks) -> None:
    from openmmla.tui.recordings import human_size

    first: tuple[int, float] | None = None
    while True:
        await asyncio.sleep(_POLL_SECONDS)
        done = await _staged_bytes(target, incoming)
        if done is None:
            continue
        now = time.monotonic()
        first = first or (done, now)
        rate = (done - first[0]) / max(now - first[1], 1e-3)
        callbacks.progress_update(min(done, total), total,
                                  f"{human_size(done)} of {human_size(total)} · {human_size(rate)}/s")


def _rsync_argv(profile, tree: Path, incoming: str, append: bool) -> list[str]:
    """rsync of the link tree (each link the file it names, -L) into the
    staging folder; --partial keeps what arrived of a file cut off, and
    --append goes on from it (openrsync has no other way to resume)."""
    from openmmla.tui import download as dl

    argv = profile.base_rsync_args()
    argv.extend(["-rtL", "--partial", f"--timeout={dl.RSYNC_IO_TIMEOUT}"])
    if append:
        argv.append("--append")
    argv.extend(["-e", profile.rsync_shell_arg(), f"{tree}/", f"{profile.ssh_destination()}:{incoming}/"])
    return argv


async def _send_over_ssh(target: ArchiveTarget, session_dir: Path, sends: list, incoming: str, has_rsync: bool,
                         callbacks) -> str | None:
    """send the files into the staging folder there, with rsync when both
    ends have it, else with scp; None when the transfer ended well, else what
    went wrong. What arrived is checked afterwards either way."""
    from openmmla.tui import download as dl

    profile = target.profile
    tree = Path(tempfile.mkdtemp(prefix="openmmla-archive-"))
    try:
        # a tree of links under the names the files take there: one rsync sends
        # them all, a file going beside another of its name included
        for item, decision in sends:
            link = tree / decision["target"]
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.abspath(session_dir / item.rel), link)
        if has_rsync and dl.local_rsync_path() and _SIMPLE_PATH.match(incoming):
            resumable = any(decision.get("staged") and decision["staged"] < item.size for item, decision in sends)
            gave_up = False
            if resumable:
                callbacks.log("  [cyan]Going on with what arrived of them before.[/cyan]")
            for append in ([True] if resumable else []) + [False]:
                code, output = await _run_quiet(_rsync_argv(profile, tree, incoming, append))
                if code in dl.RSYNC_GIVE_UP_RCS:
                    callbacks.log(f"  [yellow]rsync could not be used (exit {code}); sending with scp.[/yellow]")
                    gave_up = True
                    break
                if code not in dl.RSYNC_OK_RCS:
                    return _last_line(output) or f"rsync exited {code}"
            if not gave_up:
                callbacks.log(f"  Sent {len(sends)} file(s) with rsync; checking them there.")
                return None
        for item, decision in sends:
            _stop_if_cancelled(callbacks)
            if decision.get("staged") == item.size:
                continue  # it arrived whole before; its sum is checked like any other
            code, output = await _run_quiet(profile.base_scp_args() + [
                str(session_dir / item.rel), f"{profile.ssh_destination()}:{incoming}/{decision['target']}"])
            if code != 0:
                return _last_line(output) or f"scp exited {code}"
        callbacks.log(f"  Sent {len(sends)} file(s) with scp; checking them there.")
        return None
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


async def _push(target: ArchiveTarget, session_dir: Path, sends: list, remote_dir: str, has_rsync: bool,
                callbacks) -> str | None:
    """send each file into <session there>/.archive/incoming/<its name there>."""
    from openmmla.tui.recordings import human_size

    incoming = f"{remote_dir.rstrip('/')}/{_ARCHIVE_DIR}/{_INCOMING_DIR}"
    total = sum(item.size for item, _ in sends)
    callbacks.progress_start(f"Archive · sending {len(sends)} file(s)", total or None)
    try:
        if target.here:
            done = 0
            for index, (item, decision) in enumerate(sends, 1):
                _stop_if_cancelled(callbacks)
                await asyncio.to_thread(_copy_file, session_dir / item.rel, Path(incoming) / decision["target"])
                done += item.size
                callbacks.progress_update(done, total, f"{index} of {len(sends)} · {human_size(done)}")
            return None
        poller = asyncio.ensure_future(_poll_staged(target, incoming, total, callbacks))
        try:
            return await _send_over_ssh(target, session_dir, sends, incoming, has_rsync, callbacks)
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await poller
    finally:
        callbacks.progress_end()


async def _hash_files(session_dir: Path, files: list[LocalFile], callbacks) -> None:
    from openmmla.tui.recordings import human_size

    total = sum(item.size for item in files)
    callbacks.progress_start("Archive · reading the files here", total or None)
    done = 0
    try:
        for index, item in enumerate(files, 1):
            _stop_if_cancelled(callbacks)
            item.sha256 = await asyncio.to_thread(_sha256_file, str(session_dir / item.rel))
            done += item.size
            callbacks.progress_update(done, total, f"{index} of {len(files)} · {human_size(done)}")
    finally:
        callbacks.progress_end()


def _progress_relay(callbacks, label: str):
    """the program's progress lines, onto the progress row."""
    from openmmla.tui.recordings import human_size

    state = {"total": None}

    def relay(done, total, detail) -> None:
        if total and state["total"] != total:
            state["total"] = total
            callbacks.progress_start(label, total)
        shown = f"{human_size(done)} of {human_size(total)}" if total else human_size(done)
        callbacks.progress_update(int(done), int(total or 0), f"{shown} · {detail}")

    return relay


# ---- the session's part of the Stream Server's recordings ----

@dataclass
class StreamPlan:
    """the clips of the session's server paths to cut on the archive host."""
    clips: list = field(default_factory=list)                    # recordings.Clip
    kinds: dict[str, str] = field(default_factory=dict)          # path -> audio | video, as its bases noted
    notes: list[str] = field(default_factory=list)               # log lines (markup)
    failed: str = ""                                             # why the server could not be asked
    running: bool = False                                        # the session has not ended


def _server_paths(record: dict, server: dict) -> tuple[list[str], list[tuple[str, str]], dict[str, str]]:
    """the session's paths on the Stream Server of System Settings, the
    (stream, URL) its bases took from another server, and each path's kind:
    the paths START switched recording on for (stream_recording). Blocking: a
    host name may be resolved."""
    from openmmla.utils.stream_recording import session_server_paths

    return session_server_paths(record, server)


def session_windows(record: dict, now: datetime | None = None) -> tuple[list, str]:
    """the stretches of the session its recordings are taken from, each as
    (start, end, the server paths it covers or None for all): its
    `recording_windows` when it has them (one {paths, start, end} per START
    the Stream Server recorded it for, openmmla.utils.stream_recording; an
    open end is the session's end), else its start to its end. Also returns
    what they are: `recording_windows`, or why the session ends where it does
    (recordings.session_end)."""
    from openmmla.tui import recordings

    end_default, why = recordings.session_end(record, now)
    windows = []
    raw = (record or {}).get("recording_windows")
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            start, end, paths = item.get("start"), item.get("end"), item.get("paths")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            (start, end), paths = item, None
        else:
            continue
        start, end = recordings.parse_time(start), recordings.parse_time(end) or end_default
        if isinstance(paths, (list, tuple)):
            paths = frozenset(str(path).strip("/") for path in paths)
        else:
            paths = None
        if start is not None and end > start:
            windows.append((start, end, paths))
    if windows:
        return windows, "recording_windows"
    start = recordings.parse_time((record or {}).get("start_time"))
    return ([(start, end_default, None)] if start is not None and end_default > start else []), why


def plan_stream_cuts(record: dict | None, server: dict, now: datetime | None = None) -> StreamPlan:
    """what of the Stream Server's recordings is the session's: each path its
    bases took through the Stream Server of System Settings, cut to its
    windows, one clip per unbroken stretch (recordings.clips_for_window). The
    playback server is asked from here, over HTTP. Blocking."""
    from rich.markup import escape

    from openmmla.tui import recordings

    plan = StreamPlan()
    if not record:
        plan.notes.append("  [dim]- The session is not in MongoDB (or MongoDB was not reached), so its streams are "
                          "not known: nothing is cut from the Stream Server.[/dim]")
        return plan
    host = str((server or {}).get("host") or "").strip()
    if not host:
        plan.notes.append("  [dim]- System Settings have no Stream Server host: nothing is cut from it.[/dim]")
        return plan
    paths, elsewhere, plan.kinds = _server_paths(record, server)
    for name, url in elsewhere:
        plan.notes.append(f"  [dim]- {escape(name)}: published to another server ({escape(url)}), not cut[/dim]")
    if not paths:
        plan.notes.append("  [dim]- The session took no stream through the Stream Server: nothing to cut.[/dim]")
        return plan
    windows, _why = session_windows(record, now)
    plan.running = recordings.session_end(record, now)[1] == "running"
    port = int(server.get("playback_port") or recordings.PLAYBACK_PORT)
    try:
        spans = {path: recordings.timespans(host, path, port) for path in paths}
    except recordings.RecordingsError as error:
        plan.failed = str(error)
        return plan
    seen = set()
    for path in paths:
        for start, end, covered in windows:
            if covered is not None and path not in covered:
                continue
            for clip in recordings.clips_for_window({path: spans[path]}, start, end):
                if (clip.path, clip.start) not in seen:
                    seen.add((clip.path, clip.start))
                    plan.clips.append(clip)
    if not plan.clips:
        plan.notes.append(f"  [dim]- Nothing of {escape(', '.join(paths))} was recorded on the Stream Server in the "
                          f"session's time.[/dim]")
    return plan


def _clip_request(clip) -> dict:
    """a clip as the archive host fetches it: the query of its playback URL,
    and where it lands below the session folder."""
    from openmmla.tui import recordings

    query = recordings.clip_url("localhost", clip, recordings.PLAYBACK_PORT).split("?", 1)[1]
    rel = f"streams/server/{recordings.clip_relpath(clip).replace(os.sep, '/')}"
    return {"rel": rel, "query": query, "end": clip.end.timestamp()}


def _playback_origins(server: dict) -> list[str]:
    """the playback server as the archive host may reach it: by its name in
    System Settings (on that host itself it resolves to an address the server
    listens on), then by loopback addresses."""
    from openmmla.tui import recordings

    port = int((server or {}).get("playback_port") or recordings.PLAYBACK_PORT)
    origins = []
    for host in (str((server or {}).get("host") or "").strip(), "127.0.0.1", "127.0.1.1", "localhost"):
        if host:
            origin = f"http://[{host}]:{port}" if ":" in host and not host.startswith("[") else f"http://{host}:{port}"
            if origin not in origins:
                origins.append(origin)
    return origins


# ---- MongoDB ----

def open_mongo(settings_root=None):
    """(the sessions collection System Settings → MongoDB names, as an object
    with `sessions` and `close`, or None, and why not)."""
    from openmmla.tui.system_services import (
        load_system_services_config, section_address_set, system_services_config_path, unset_address_note,
    )
    from openmmla.utils.config import load_config_with_system_services
    from openmmla.utils.constants import MONGODB_DEFAULT_DB

    settings_root = settings_root or _settings_root()
    if not section_address_set((load_system_services_config(settings_root) or {}).get("MongoDB"), "MongoDB"):
        return None, unset_address_note("MongoDB")
    try:
        from pymongo import MongoClient
    except ModuleNotFoundError:
        return None, "pymongo is not installed here"
    try:
        section = load_config_with_system_services(system_services_config_path(settings_root)).get("MongoDB") or {}
        client = MongoClient(section["url"], serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        client.admin.command("ping")
    except Exception as error:  # its text may carry the URL: only its kind is said
        return None, f"MongoDB did not answer ({type(error).__name__})"
    sessions = client[section.get("db") or MONGODB_DEFAULT_DB]["sessions"]
    return SimpleNamespace(sessions=sessions, close=client.close), ""


def _find_record(mongo, session_id: str) -> tuple[dict | None, str]:
    """(the session's document or None, and why MongoDB could not be read)."""
    try:
        record = mongo.sessions.find_one({"session_id": session_id}, {"_id": 0})
    except Exception as error:  # its text may carry the URL: only its kind is said
        return None, f"MongoDB did not answer ({type(error).__name__})"
    return (record if isinstance(record, dict) and record else None), ""


def _note_in_mongo(mongo, session_id: str, archive: dict) -> str:
    """set `archive` on the session's document, which has to be there: one
    update that touches nothing else and never makes a document."""
    try:
        result = mongo.sessions.update_one({"session_id": session_id}, {"$set": {"archive": archive}})
    except Exception as error:
        return f"failed ({type(error).__name__})"
    return "noted" if getattr(result, "matched_count", 0) else "missing"


# ---- the manifest there ----

@dataclass
class Archived:
    """a file as the archive host holds it."""
    final: str         # below the session folder there
    bytes: int
    sha256: str


def _parse_manifest(texts: dict | None) -> dict:
    """the newer of manifest.json and manifest.yml that parses ({} for none)."""
    import yaml

    best, best_time = {}, None
    for name in SESSION_MANIFESTS:
        entry = (texts or {}).get(name)
        if not entry:
            continue
        try:
            data = json.loads(entry["text"]) if name.endswith(".json") else yaml.safe_load(entry["text"])
        except (ValueError, yaml.YAMLError):
            continue
        if isinstance(data, dict) and (best_time is None or float(entry.get("mtime") or 0) > best_time):
            best, best_time = data, float(entry.get("mtime") or 0)
    return best


def _local_manifest_texts(session_dir: Path) -> dict:
    found = {}
    for name in SESSION_MANIFESTS:
        path = session_dir / name
        try:
            found[name] = {"text": path.read_text(encoding="utf-8"), "mtime": path.stat().st_mtime}
        except (OSError, UnicodeDecodeError):
            found[name] = None
    return found


def _same_entry(old, new) -> bool:
    if not (isinstance(old, dict) and isinstance(new, dict)):
        return old == new
    if old.get("id") is not None and new.get("id") is not None:
        return old.get("id") == new.get("id")

    def lasting(entry: dict) -> dict:
        return {key: value for key, value in entry.items() if not str(key).endswith("_at")}

    return lasting(old) == lasting(new)


def merge_manifest(base: dict, update: dict) -> dict:
    """`update` laid over `base`: every key of either kept, mappings merged
    key by key, a list's entries matched by their id (else by what they hold
    apart from their *_at times) and merged, the rest appended; a value of
    `update` wins over one of `base` unless it is empty."""
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in (update or {}).items():
        merged[key] = _merged(merged[key], value) if key in merged else copy.deepcopy(value)
    return merged


def _merged(old, new):
    if isinstance(old, dict) and isinstance(new, dict):
        return merge_manifest(old, new)
    if isinstance(old, list) and isinstance(new, list):
        out = copy.deepcopy(old)
        for item in new:
            for index, existing in enumerate(out):
                if _same_entry(existing, item):
                    out[index] = _merged(existing, item)
                    break
            else:
                out.append(copy.deepcopy(item))
        return out
    return copy.deepcopy(new) if new not in (None, "") else old


def _under(path: str, folder: str) -> str | None:
    """`path` relative to `folder` when it lies in it ('' for the folder itself)."""
    text, base = str(path or "").rstrip("/"), str(folder).rstrip("/")
    if text == base:
        return ""
    return text[len(base) + 1:] if text.startswith(base + "/") else None


def _row_rel(row: dict, local_dirs: tuple[str, ...], remote_dir: str, session_id: str) -> str | None:
    """the place below the session folder of the file a recording row names."""
    path = str(row.get("path") or "").strip()
    for folder in (*local_dirs, remote_dir):
        rel = _under(path, folder) if path else None
        if rel:
            return rel
    parts = path.replace("\\", "/").split("/")
    if session_id in parts[:-1]:
        # written on another machine: what follows the session's folder there
        index = len(parts) - 1 - parts[::-1].index(session_id)
        return "/".join(parts[index + 1:]) or None
    rel = str(row.get("relpath") or "").strip().strip("/")
    if rel and _rel_ok(rel):
        return rel
    if path and not os.path.isabs(path) and _rel_ok(path):
        return path
    return None


def rewrite_manifest(local: dict, remote: dict, *, session_id: str, local_dirs: tuple[str, ...], remote_dir: str,
                     archived: dict[str, Archived], console: str, same_place: bool, cut_rows: list[dict],
                     archive: dict) -> tuple[dict, int]:
    """the session's manifest as the archive host is to hold it: its own
    there, with this console's laid over it after every path of this
    console's was turned into the one its file has there. Each recording row
    gets path (absolute there), relpath, bytes, sha256 and origin (where the
    file was sent from, unless a row already names one); a file source folder
    the archive holds files of points there. The stream cuts are listed under
    `stream_cuts`, apart from the recordings: ses-code, ses-align and
    ses-calibrate take every row of `recordings` for a device's recording of
    the collection (with its host and device, from Collection Start on), and
    a row an earlier archive put there is moved over. `archive` notes this
    archive. Also returns how many rows name a file the archive does not
    hold."""
    local = copy.deepcopy(local or {})
    remote = remote or {}
    remote_origins = {row.get("id"): row.get("origin") for row in remote.get("recordings") or []
                      if isinstance(row, dict) and isinstance(row.get("origin"), dict)}
    unresolved = 0
    for row in local.get("recordings") or []:
        if not isinstance(row, dict) or row.get("kind") == _STREAM_KIND:
            continue
        rel = _row_rel(row, local_dirs, remote_dir, session_id)
        item = archived.get(rel) if rel else None
        if item is None:
            unresolved += 1
            continue
        origin = remote_origins.get(row.get("id")) or (row.get("origin") if isinstance(row.get("origin"), dict) else None)
        if origin is None and not same_place:
            origin = {"host": console, "path": f"{local_dirs[0]}/{rel}"}
        row.update({"path": f"{remote_dir}/{item.final}", "relpath": item.final, "bytes": item.bytes,
                    "sha256": item.sha256})
        if origin:
            row["origin"] = origin
    finals = [item.final for item in archived.values()]
    sources = local.get("file_sources")
    for entries in (sources.values() if isinstance(sources, dict) else []):
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            for key in ("file_dir", "frame_dir"):
                rel = next((found for folder in local_dirs
                            if (found := _under(str(entry.get(key) or ""), folder)) is not None), None)
                if rel is not None and any(final == rel or final.startswith(rel + "/") or not rel for final in finals):
                    entry[key] = f"{remote_dir}/{rel}" if rel else remote_dir
    merged = merge_manifest(remote, local)
    rows = merged.get("recordings") if isinstance(merged.get("recordings"), list) else []
    moved = [row for row in rows if isinstance(row, dict) and row.get("kind") == _STREAM_KIND]
    merged["recordings"] = [row for row in rows if not (isinstance(row, dict) and row.get("kind") == _STREAM_KIND)]
    cuts = merged.get(STREAM_CUTS_KEY) if isinstance(merged.get(STREAM_CUTS_KEY), list) else []
    cuts = _merged(_merged(cuts, moved), cut_rows)
    if cuts:
        merged[STREAM_CUTS_KEY] = cuts
    merged["session_id"] = merged.get("session_id") or session_id
    merged["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    merged["archive"] = archive
    return merged, unresolved


def _cut_row(clip, result: dict, remote_dir: str, server_host: str, kind: str) -> dict:
    """the manifest row of a stream cut (under `stream_cuts`), as the archive
    host holds it."""
    rel = result["rel"]
    below = rel.split("streams/server/", 1)[-1]
    return {
        "id": "stream_" + os.path.splitext(below)[0].replace("/", "_"),
        "modality": kind,
        "kind": _STREAM_KIND,
        "stream_path": clip.path,
        "start_time": clip.start.timestamp(),
        "duration": clip.duration,
        "format": os.path.splitext(rel)[1].lstrip(".") or "mp4",
        "path": f"{remote_dir}/{rel}",
        "relpath": rel,
        "bytes": int(result.get("bytes") or 0),
        "sha256": result.get("sha256"),
        "origin": {"host": "mediamtx", "server": server_host, "path": clip.path},
    }


def _render_manifest(manifest: dict) -> tuple[str, str]:
    import yaml

    return (json.dumps(manifest, indent=2, sort_keys=False, default=str) + "\n",
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))


# ---- the archive ----

@dataclass
class ArchiveResult:
    session_id: str
    location: str = ""
    status: str = ""                                          # complete | partial; "" for a dry run
    to_send: int = 0                                          # what a dry run would send
    to_send_bytes: int = 0
    sent: int = 0                                             # sent and verified there
    sent_bytes: int = 0
    beside: int = 0                                           # sent beside another file of its name
    present: int = 0                                          # there already, same sha256
    present_bytes: int = 0
    mismatched: list[str] = field(default_factory=list)       # arrived with another sha256 (dropped there)
    failed: list[str] = field(default_factory=list)           # did not arrive, or refused there
    cuts: dict[str, int] = field(default_factory=dict)        # cut | replaced | kept | failed -> clips
    cut_bytes: int = 0
    clips: int = 0                                            # clips a dry run would cut
    stream_failed: str = ""
    running: bool = False
    manifest: str = ""                                        # written | not written (why)
    mongo: str = ""                                           # noted | missing | not connected | failed (...)

    @property
    def files(self) -> int:
        return self.sent + self.present + sum(count for status, count in self.cuts.items() if status != "failed")

    @property
    def bytes(self) -> int:
        return self.sent_bytes + self.present_bytes + self.cut_bytes

    @property
    def exit_code(self) -> int:
        return 0 if self.status in ("", "complete") else 1


# what to do about a session this console holds nothing of
DOWNLOAD_FIRST = ("download its recordings first (the Collection card's Download), and export its streams and base "
                  "files (Sessions tab) if it has any, then archive it")


def no_local_folder(session_id: str) -> str:
    from openmmla.utils.artifact_paths import safe_segment

    return (f"'{session_id}' has no folder here ({ARTIFACTS_DIR}/{safe_segment(session_id, 'session')}/): "
            f"{DOWNLOAD_FIRST}")


def _describe_selection(selection: Selection) -> list[str]:
    from rich.markup import escape

    from openmmla.tui.recordings import human_size

    by_folder: dict[str, list[int]] = {}
    for item in selection.files:
        folder = item.rel.split("/", 1)[0]
        count = by_folder.setdefault(folder, [0, 0])
        count[0] += 1
        count[1] += item.size
    lines = []
    if by_folder:
        parts = ", ".join(f"{folder}/ {count} ({human_size(size)})" for folder, (count, size) in by_folder.items())
        lines.append(f"  Raw files here: {escape(parts)}")
    if selection.profiles:
        lines.append(f"  [yellow]- {selection.profiles} file(s) under profiles/ (speaker profiles) are never sent: "
                     f"they stay here.[/yellow]")
    if selection.runtime:
        lines.append(f"  [dim]- What the bases stored ({selection.runtime} file(s), "
                     f"{human_size(selection.runtime_bytes)}, pipelines/*/*/real-time/ and post-time/) stays here: "
                     f"--with-runtime-media sends it.[/dim]")
    if selection.left:
        parts = ", ".join(f"{folder} {count}" for folder, count in sorted(selection.left.items()))
        lines.append(f"  [dim]- Not raw, stays here: {escape(parts)}[/dim]")
    if selection.links:
        shown = ", ".join(selection.links[:5]) + (" …" if len(selection.links) > 5 else "")
        lines.append(f"  [yellow]- {len(selection.links)} symbolic link(s) are not followed: {escape(shown)}[/yellow]")
    if selection.unnamed:
        lines.append(f"  [yellow]- {len(selection.unnamed)} file(s) whose names are not UTF-8 are not sent.[/yellow]")
    return lines


def _still_recording(session_dir: Path) -> list[str]:
    """the recordings the session's manifest says are still being written."""
    rows = _parse_manifest(_local_manifest_texts(session_dir)).get("recordings") or []
    return [str(row.get("id") or row.get("path") or "?") for row in rows
            if isinstance(row, dict) and row.get("status") == "recording"]


async def archive_session(session_id: str, *, host: str | None = None, dry_run: bool = False,
                          with_runtime_media: bool = False, callbacks=None, mongo=None, record: dict | None = None,
                          local_root=None, settings_root=None) -> ArchiveResult:
    """archive one session (the module's docstring says how). `callbacks` has
    the shape of stream_export.ExportCallbacks; `mongo` an object whose
    `sessions` is the MongoDB collection (None: the session's streams are not
    known and nothing is noted there); `record` stands in for the session's
    document. Raises ArchiveError when nothing could be archived; a file that
    did not arrive whole, or a stream that could not be cut, makes the status
    partial."""
    from rich.markup import escape

    from openmmla.tui.recordings import human_size
    from openmmla.tui.system_services import stream_server_address
    from openmmla.utils.artifact_paths import safe_segment, short_hostname

    callbacks = callbacks or _Quiet()
    log = callbacks.log
    for line in hook_notes(local_root):
        log(line)
    settings_root = settings_root or _settings_root()
    sid = safe_segment(session_id, "session")
    session_dir = local_session_dir(sid, local_root)
    if not session_dir.is_dir():
        raise ArchiveError(no_local_folder(session_id))
    console = short_hostname()
    label = safe_segment(console, "console")
    result = ArchiveResult(session_id)

    log(f"[bold]Archiving session: {escape(session_id)}[/bold]" + (" (dry run: nothing is sent)" if dry_run else ""))
    target = await asyncio.to_thread(resolve_target, host, settings_root)
    result.location = target.location(sid)
    log(f"  Archive host: {escape(target.name)} ({escape(target.via)}), into {escape(target.session_path(sid))}")

    selection = await asyncio.to_thread(select_files, session_dir, with_runtime_media)
    for line in _describe_selection(selection):
        log(line)
    recording = await asyncio.to_thread(_still_recording, session_dir)
    if recording:
        log(f"  [yellow]The manifest says {len(recording)} recording(s) were still being written when they were "
            f"downloaded ({escape(', '.join(recording[:3]))}): if so, download them again after Stop.[/yellow]")
    await _hash_files(session_dir, selection.files, callbacks)
    _stop_if_cancelled(callbacks)

    # without the session's document its streams are not known: not complete
    streams_unknown = False
    if mongo is None:
        streams_unknown = record is None
        unknown = "the session's streams are not known, and " if streams_unknown else ""
        log(f"  [yellow]MongoDB is not connected: {unknown}the archive is not noted there.[/yellow]")
        result.mongo = "not connected"
    elif record is None:
        record, error = await asyncio.to_thread(_find_record, mongo, session_id)
        if error:
            streams_unknown = True
            result.mongo = "unreachable"
            mongo = None
            log(f"  [yellow]{escape(error)}: the session's streams are not known, and the archive is not noted "
                f"there.[/yellow]")
    server = await asyncio.to_thread(stream_server_address, settings_root)
    plan = await asyncio.to_thread(plan_stream_cuts, record, server)
    for line in plan.notes:
        log(line)
    if plan.failed:
        log(f"  [red]✗ The Stream Server does not answer: {escape(plan.failed)}[/red]")
    if plan.running:
        log("  [yellow]The session is still running: its streams are cut up to now. Archive it again once it has "
            "ended.[/yellow]")
    # a stream cut exported here that the archive host can still cut from the Stream Server itself is
    # left here: the host's own cut is the copy kept, so an export made before or after an earlier
    # archive never lands beside it as a second file. what the Stream Server no longer holds (past
    # its retention) is sent as exported
    cut_there = {_clip_request(clip)["rel"] for clip in plan.clips}
    exported = [item for item in selection.files if item.rel in cut_there]
    if exported:
        selection.files = [item for item in selection.files if item.rel not in cut_there]
        log(f"  [dim]- {len(exported)} exported stream cut(s) stay here: {escape(target.name)} cuts them from the "
            f"Stream Server itself.[/dim]")

    # where each file goes there, and what is there already
    same_place = target.here and os.path.realpath(_expand(target.session_path(sid))) == os.path.realpath(session_dir)
    callbacks.progress_start(f"Archive · asking {target.name}", None)
    try:
        check = await _ask(target, {
            "op": "check", "session_dir": target.session_path(sid), "label": label, "dry_run": dry_run,
            "files": [] if same_place else [{"rel": item.rel, "size": item.size, "sha256": item.sha256}
                                            for item in selection.files],
        }, progress=_progress_relay(callbacks, f"Archive · comparing on {target.name}"))
    finally:
        callbacks.progress_end()
    remote_dir = str(check["session_dir"])
    decisions = check.get("files") or {}
    archived: dict[str, Archived] = {}
    sends = []
    for item in selection.files:
        decision = {"action": "skip", "target": item.rel} if same_place else decisions.get(item.rel) or {}
        if decision.get("action") == "skip":
            archived[item.rel] = Archived(decision["target"], item.size, item.sha256)
            result.present += 1
            result.present_bytes += item.size
        elif decision.get("action") == "send":
            sends.append((item, decision))
        else:
            result.failed.append(item.rel)
            log(f"  [red]✗ {escape(item.rel)}: {escape(str(decision.get('why') or 'no answer for it'))}[/red]")
    result.to_send = len(sends)
    result.to_send_bytes = sum(item.size for item, _ in sends)
    beside = [item.rel for item, decision in sends if decision["target"] != item.rel]
    log(f"  To send: {len(sends)} file(s), {human_size(result.to_send_bytes)}; there already: {result.present} "
        f"({human_size(result.present_bytes)})" + (f"; {len(beside)} go beside a different file of the same name"
                                                    if beside else ""))
    for rel in beside[:5]:
        target_name = next(decision["target"] for item, decision in sends if item.rel == rel)
        log(f"    [yellow]{escape(rel)} differs from the file there: sent as {escape(target_name)}[/yellow]")

    result.clips = len(plan.clips)
    result.running = plan.running
    if plan.clips:
        log(f"  Stream Server: {len(plan.clips)} clip(s) of {escape(', '.join(dict.fromkeys(c.path for c in plan.clips)))} "
            f"to cut on {escape(target.name)}")
    if not selection.files and not plan.clips:
        raise ArchiveError(f"Nothing raw to archive in {ARTIFACTS_DIR}/{sid}/, and no Stream Server recording of the "
                           f"session: {DOWNLOAD_FIRST}")
    if dry_run:
        _summarize(result, log)
        return result

    free = int(check.get("free") or 0)
    if free and result.to_send_bytes > free * 0.98:
        raise ArchiveError(f"{target.name} has {human_size(free)} free, and the files to send take "
                           f"{human_size(result.to_send_bytes)}")

    # the transfer, into the staging folder there
    if sends:
        _stop_if_cancelled(callbacks)
        failure = await _push(target, session_dir, sends, remote_dir, bool(check.get("has_rsync")), callbacks)
        if failure:
            log(f"  [red]✗ The transfer to {escape(target.name)} stopped: {escape(failure)}[/red]")
        _stop_if_cancelled(callbacks)
        callbacks.progress_start(f"Archive · checking on {target.name}", None)
        try:
            placed = await _ask(target, {
                "op": "place", "session_dir": remote_dir, "label": label,
                "items": [{"rel": item.rel, "target": decision["target"], "size": item.size, "sha256": item.sha256}
                          for item, decision in sends],
            }, progress=_progress_relay(callbacks, f"Archive · checking on {target.name}"))
        finally:
            callbacks.progress_end()
        for item, decision in sends:
            answer = (placed.get("placed") or {}).get(item.rel) or {}
            if answer.get("ok"):
                archived[item.rel] = Archived(answer["final"], int(answer.get("bytes") or item.size), item.sha256)
                result.sent += 1
                result.sent_bytes += item.size
                result.beside += answer["final"] != item.rel
            elif "differs" in str(answer.get("why") or ""):
                result.mismatched.append(item.rel)
                log(f"  [red]✗ {escape(item.rel)}: its sha256 there is not the one here; the copy there was "
                    f"dropped.[/red]")
            else:
                result.failed.append(item.rel)
                log(f"  [red]✗ {escape(item.rel)}: {escape(str(answer.get('why') or 'no answer for it'))}[/red]")

    # the session's part of the Stream Server's recordings, cut there
    cut_rows = []
    if plan.clips:
        _stop_if_cancelled(callbacks)
        requests = [_clip_request(clip) for clip in plan.clips]
        callbacks.progress_start(f"Archive · cutting on {target.name}", None)
        try:
            reply = await _ask(target, {"op": "cut", "session_dir": remote_dir, "origins": _playback_origins(server),
                                        "clips": requests},
                               progress=_progress_relay(callbacks, f"Archive · cutting on {target.name}"))
        finally:
            callbacks.progress_end()
        for clip, answer in zip(plan.clips, reply.get("clips") or []):
            status = str(answer.get("status") or "failed")
            result.cuts[status] = result.cuts.get(status, 0) + 1
            label_text = escape(f"{clip.path}  {clip.start:%H:%M:%S} +{clip.duration:.0f}s")
            if status == "failed":
                log(f"  [red]✗ {label_text}: {escape(str(answer.get('why') or ''))}[/red]")
                continue
            result.cut_bytes += int(answer.get("bytes") or 0)
            cut_rows.append(_cut_row(clip, answer, remote_dir, str(server.get("host") or ""),
                                     plan.kinds.get(clip.path, "video")))
            log(f"  [green]✓[/green] {label_text} -> {escape(answer['rel'])} "
                f"({human_size(answer.get('bytes'))}{', already there' if status == 'kept' else ''})")
    result.stream_failed = plan.failed

    complete = not (result.mismatched or result.failed or result.cuts.get("failed") or plan.failed or plan.running
                    or streams_unknown)
    result.status = "complete" if complete else "partial"
    verified_at = datetime.now(timezone.utc)
    archive = {"location": result.location, "status": result.status, "files": result.files, "bytes": result.bytes,
               "verified_at": verified_at}

    # the manifest there, with the paths the files have there
    _stop_if_cancelled(callbacks)
    local_manifest = _parse_manifest(_local_manifest_texts(session_dir))
    local_dirs = tuple(dict.fromkeys((str(session_dir), os.path.realpath(session_dir))))
    texts = check.get("manifest") or {}
    note = dict(archive, verified_at=verified_at.strftime("%Y-%m-%dT%H:%M:%SZ"), sent_from={
        "host": console, "path": str(session_dir)})
    result.manifest = "not written"
    for _attempt in range(2):
        merged, unresolved = rewrite_manifest(
            local_manifest, _parse_manifest(texts), session_id=sid, local_dirs=local_dirs, remote_dir=remote_dir,
            archived=archived, console=console, same_place=same_place, cut_rows=cut_rows, archive=note)
        json_text, yml_text = _render_manifest(merged)
        reply = await _ask(target, {
            "op": "write", "session_dir": remote_dir, "json": json_text, "yml": yml_text,
            "expected": {name: (texts.get(name) or {}).get("text") for name in SESSION_MANIFESTS},
        }, timeout=120.0)
        if reply.get("written"):
            result.manifest = "written"
            rows = len(merged.get("recordings") or [])
            cuts = len(merged.get(STREAM_CUTS_KEY) or [])
            log(f"  [green]✓[/green] Manifest written there: {rows} recording(s) with their paths there"
                + (f", {cuts} stream cut(s) under {STREAM_CUTS_KEY}" if cuts else "")
                + (f" ({unresolved} row(s) of this manifest name a file the archive does not hold)" if unresolved
                   else ""))
            break
        texts = reply.get("manifest") or {}
    else:
        log("  [yellow]The manifest there kept changing while it was merged: it was left as it is.[/yellow]")

    # the session's document
    if mongo is not None:
        if record is None:
            result.mongo = "missing"
            log(f"  [yellow]MongoDB has no document of '{escape(session_id)}': the archive is noted in the manifest "
                f"there only.[/yellow]")
        else:
            result.mongo = await asyncio.to_thread(_note_in_mongo, mongo, session_id, archive)
            if result.mongo != "noted":
                log(f"  [yellow]The archive could not be noted in MongoDB: {escape(result.mongo)}.[/yellow]")
    _summarize(result, log)
    return result


def _summarize(result: ArchiveResult, log) -> None:
    from rich.markup import escape

    from openmmla.tui.recordings import human_size

    shown = escape(result.session_id)
    if not result.status:
        log(f"[bold]Dry run of {shown}: {result.to_send} file(s) ({human_size(result.to_send_bytes)}) would be sent, "
            f"{result.present} are there already, {result.clips} stream clip(s) would be cut, into "
            f"{escape(result.location)}[/bold]")
        return
    cuts = ", ".join(f"{count} {status}" for status, count in sorted(result.cuts.items())) or "none"
    lines = [
        f"  sent:           {result.sent} file(s), {human_size(result.sent_bytes)}"
        + (f" ({result.beside} beside a different file of the same name)" if result.beside else ""),
        f"  already there:  {result.present} file(s), {human_size(result.present_bytes)}",
        f"  verified:       {result.sent + result.present} file(s) by sha256 there",
        f"  mismatches:     {len(result.mismatched)}" + (f", did not arrive: {len(result.failed)}" if result.failed else ""),
        f"  stream cuts:    {cuts}" + (" (the Stream Server did not answer)" if result.stream_failed else ""),
        f"  manifest:       {result.manifest}; MongoDB: {result.mongo}",
    ]
    color = "green" if result.status == "complete" else "yellow"
    log(f"[bold {color}]Archive of {shown}: {result.status}, {result.files} file(s), {human_size(result.bytes)} at "
        f"{escape(result.location)}[/bold {color}]")
    for line in lines:
        log(escape(line))
    if result.status != "complete":
        log("  [yellow]Archive again to send what is missing: what arrived whole stays there and is not sent "
            "again.[/yellow]")


# ---- the command ----

class _ConsoleCallbacks:
    """the archive's log on this terminal (Rich markup), its progress on one
    line of stderr."""

    def __init__(self) -> None:
        try:
            from rich.console import Console

            self._console = Console(highlight=False, soft_wrap=True)
        except ModuleNotFoundError:
            self._console = None
        self._tty = sys.stderr.isatty()
        self._label = ""
        self._shown = 0.0
        self._open = False

    def log(self, line: str) -> None:
        self._close_line()
        if self._console is not None:
            self._console.print(line)
        else:
            print(re.sub(r"\[/?[a-z #0-9]*\]", "", line))

    def progress_start(self, label: str, _total) -> None:
        self._label = label

    def progress_update(self, _done: int, _total: int, detail: str) -> None:
        now = time.monotonic()
        if now - self._shown < (0.5 if self._tty else 15.0):
            return
        self._shown = now
        if self._tty:
            sys.stderr.write(f"\r\033[K  {self._label}: {detail}")
            self._open = True
        else:
            sys.stderr.write(f"  {self._label}: {detail}\n")
        sys.stderr.flush()

    def progress_end(self) -> None:
        self._close_line()

    def _close_line(self) -> None:
        if self._open:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
            self._open = False

    def cancelled(self) -> bool:
        return False


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-archive",
        description="Send a session's raw files from this console's artifacts/<session>/ to the System Settings host "
                    "(the Dashboard's, else the Stream Server's), check each there by its sha256, cut the session's "
                    "part of the Stream Server's recordings there, rewrite the session's manifest there, and note the "
                    "archive in MongoDB. Nothing is deleted.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=120),
    )
    parser.add_argument("session_id", help="the session (its folder here is artifacts/<session>/)")
    parser.add_argument("--host", default=None,
                        help="the SSH profile of the archive host, or 'local'; if not set, the Dashboard's host of "
                             "System Settings, else the Stream Server's")
    parser.add_argument("--dry-run", action="store_true", help="say what would be sent and cut, and change nothing")
    parser.add_argument("--with-runtime-media", action="store_true",
                        help="also send what the bases stored: pipelines/<pipeline>/<host>/real-time/, post-time/ and "
                             "visualizations/ (speaker profiles never)")
    return parser


def main():
    args = get_parser().parse_args()
    callbacks = _ConsoleCallbacks()
    mongo, why = open_mongo()
    if mongo is None:
        callbacks.log(f"[yellow]MongoDB: {why}.[/yellow]")
    try:
        result = asyncio.run(archive_session(
            args.session_id, host=args.host, dry_run=args.dry_run, with_runtime_media=args.with_runtime_media,
            callbacks=callbacks, mongo=mongo))
    except ArchiveError as error:
        callbacks.log(f"[red]✗ {_escape(str(error))}[/red]")
        sys.exit(2)
    except KeyboardInterrupt:
        callbacks.log("[yellow]Stopped. What arrived on the archive host is kept: run it again to go on.[/yellow]")
        sys.exit(130)
    finally:
        if mongo is not None:
            with contextlib.suppress(Exception):
                mongo.close()
    sys.exit(result.exit_code)


def _escape(text: str) -> str:
    try:
        from rich.markup import escape
    except ModuleNotFoundError:
        return text
    return escape(text)
