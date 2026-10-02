"""mmla ses-delete: a session removed, either everywhere central or from the
files of one host.

Without --files-on it is the Sessions tab's Delete Session: the session goes
from where the console keeps it for everyone. In this order, stopping at the
first step that fails (the session then stays listed, and the same command or
press goes on from there):

  archive   artifacts/<session>/ of the archive host's checkout (the folder
            mmla ses-archive writes, on the host archive.resolve_target names;
            its ledger lies inside it, in .archive/), and the dashboard's
            cached report of the session on that host
            (pipelines/uber-server/dashboard/flask-backend/cache/<session>/,
            its default place; a dashboard started with DASHBOARD_CACHE_DIR
            keeps it elsewhere, and that copy is not found)
  InfluxDB  every event of the session, counted again afterwards
  MongoDB   the session's document

Copies on other machines stay: this console's own folder (unless this console
is the archive host), the capture hosts' folders and the Stream Server's
recordings, which streams share between sessions. When the archive host cannot
be reached, or cannot be named, nothing is deleted.

With --files-on HOST it is Delete Files: the session's folders on that one
host ('local' for this machine, else an SSH profile) and nowhere else. On this
machine they are artifacts/<session>/ and collection/<session>/ of its
checkout; on another host also ~/artifacts/<session>/ (where the Collection
recorders write by default), the stream cuts staged under
~/artifacts/streams/.session-cuts/<session>/, and the folders the session's
MongoDB document names on that host (a recorder's own output root, a stream's
record_root).

Every folder deleted is a directory that is a direct child, named exactly the
session id (letter case included, also where the file system ignores case), of
one of those roots, as the host itself resolves it (symbolic links in the root
are followed, a link by the session's name is refused, and the root itself is
never deleted). The first run (or press) says what would
go, with sizes, and deletes nothing; with --yes (the second press) exactly
those folders are deleted, each checked again just before.

The host side is a bash script run over SSH (or here), with commands macOS and
Linux share; everything else is planning and parsing, tested without a host.
A host that stops answering once a delete was sent may have deleted part of
it: what it said before is logged, and nothing claims the rest is untouched.

Test hooks: ses-archive's OPENMMLA_ARCHIVE_REMOTE_ROOT moves the archive (and
the dashboard cache looked for beside it), ses-export's
OPENMMLA_EXPORT_REMOTE_ROOT moves the folders of every SSH host. The log says
so, in yellow, whenever one of them is set."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ARTIFACTS_DIR = "artifacts"
COLLECTION_DIR = "collection"
# below the archive host's checkout: the dashboard's report cache, one folder per session
DASHBOARD_CACHE_REL = "pipelines/uber-server/dashboard/flask-backend/cache"

# seconds a host has: to list its session folders, to size one session's, to delete them
LIST_TIMEOUT = 60.0
PLAN_TIMEOUT = 120.0
DELETE_TIMEOUT = 900.0

# what a root holds, as the log names it
WHAT_ARTIFACTS = "artifacts"
WHAT_COLLECTION = "collection"
WHAT_RECORDERS = "Collection recordings"
WHAT_CUTS = "stream cuts"
WHAT_ARCHIVE = "archive"
WHAT_CACHE = "dashboard cache"


def _escape(text) -> str:
    from rich.markup import escape

    return escape(str(text))


def _human(size) -> str:
    from openmmla.tui.recordings import human_size

    return human_size(size)


# ---- the session id ----

def session_id_problem(session_id) -> str:
    """why `session_id` cannot name the one folder a deletion takes ('' when
    it can): it must be one path segment as the console writes them
    (artifact_paths.safe_segment), and none of the folders artifacts/ holds
    for every session (streams/, runtime/, recordings/)."""
    from openmmla.utils.artifact_paths import NON_SESSION_ARTIFACT_DIRS, safe_segment

    text = str(session_id if session_id is not None else "")
    if not text.strip():
        return "no session id"
    if any(char in text for char in "/\\\0\n\r") or text in (".", ".."):
        return f"'{text}' is no session id that names one folder"
    if safe_segment(text, "") != text:
        return f"'{text}' is no session id that names one folder"
    if text in NON_SESSION_ARTIFACT_DIRS:
        return f"'{text}' is a folder every session shares, not a session"
    return ""


# ---- where a host keeps a session's files ----

@dataclass(frozen=True)
class FilesRoot:
    """a folder whose children named after a session are that session's files."""
    path: str    # as that host spells it: absolute, or ~/... and $HOME/... for its home
    what: str    # what it holds, as the log names it


def _spelled(path: str) -> str:
    """one spelling of a host path: ~ as $HOME, no trailing or doubled slash."""
    import re

    text = re.sub(r"/{2,}", "/", str(path or "").strip())
    if text == "~" or text.startswith("~/"):
        text = "$HOME" + text[1:]
    return text.rstrip("/") or "/"


def _shown(path: str) -> str:
    """a host path as the log names it: $HOME as ~."""
    text = str(path)
    return "~" + text[len("$HOME"):] if text == "$HOME" or text.startswith("$HOME/") else text


def _join(root: str, *parts: str) -> str:
    return "/".join([str(root).rstrip("/"), *[part.strip("/") for part in parts if part]])


def _export_remote_root() -> str:
    from openmmla.commands.ses.export import ENV_REMOTE_ROOT

    return os.environ.get(ENV_REMOTE_ROOT, "").strip()


def hook_notes() -> list[str]:
    """a log line (markup) for each test hook that moves what is deleted."""
    from openmmla.commands.ses import archive, export

    notes = []
    for name, what in ((archive.ENV_REMOTE_ROOT, "the archive is looked for under {} on the archive host"),
                       (export.ENV_REMOTE_ROOT, "the files of every SSH host are looked for under {}")):
        value = os.environ.get(name, "").strip()
        if value:
            notes.append(f"  [yellow]{name} is set: {what.format(_escape(value))}, not its own folders.[/yellow]")
    return notes


def _record_roots(host: str, session_id: str, record: dict | None, project_root) -> list[FilesRoot]:
    """the roots the session's document names on `host` (an SSH profile):
    the folder each of its Collection recorders wrote into there, and the
    staging folder of the cuts of each stream captured there with Record on."""
    from openmmla.commands.ses.export import COLLECTION_RECORDERS_FIELD
    from openmmla.utils import session_sources
    from openmmla.utils.artifact_paths import SESSION_CUTS_DIR, STREAMS_DIR, capture_record_root

    roots = []
    recorders = (record or {}).get(COLLECTION_RECORDERS_FIELD)
    for item in recorders if isinstance(recorders, list) else []:
        if not isinstance(item, dict) or str(item.get("host") or "") != host:
            continue
        folder = _spelled(str(item.get("folder") or ""))
        parts = folder.split("/")
        if session_id in parts[1:]:
            # the folder holding <session>/: what Start wrote into is below it
            index = len(parts) - 1 - parts[::-1].index(session_id)
            root = "/".join(parts[:index])
            if root and root != "/":
                roots.append(FilesRoot(root, WHAT_RECORDERS))
    for entry in session_sources.captured_streams(record):
        if entry["ssh_profile"] == host and entry["record"]:
            record_root = capture_record_root(host, entry["record_root"], project_root)
            roots.append(FilesRoot(_join(_spelled(record_root), STREAMS_DIR, SESSION_CUTS_DIR), WHAT_CUTS))
    return roots


def host_roots(host: str, project_root=None, record: dict | None = None, session_id: str = "") -> list[FilesRoot]:
    """the roots a session's files lie under on `host`: on this machine
    ('local') artifacts/ and collection/ of its checkout; on an SSH host
    artifacts/ and collection/ of the checkout its profile names, ~/artifacts
    (the Collection recorders' default output root), the staged stream cuts
    under ~/artifacts/streams/.session-cuts, and what the session's document
    names there. Each once, in that order."""
    from openmmla.utils.artifact_paths import SESSION_CUTS_DIR, STREAMS_DIR

    if host == "local":
        if project_root is None:
            from openmmla.tui.schema.loader import _find_project_root

            project_root = _find_project_root()
        base = os.path.abspath(str(project_root))
        roots = [FilesRoot(os.path.join(base, ARTIFACTS_DIR), WHAT_ARTIFACTS),
                 FilesRoot(os.path.join(base, COLLECTION_DIR), WHAT_COLLECTION)]
    else:
        from openmmla.tui.ssh import get_profile_by_name

        hooked = _export_remote_root()
        if hooked:
            checkout = home = _spelled(hooked)
        else:
            profile = get_profile_by_name(host)
            checkout = _spelled((getattr(profile, "remote_project_path", "") or "~/OpenMMLA"))
            home = "$HOME"
        roots = [FilesRoot(_join(checkout, ARTIFACTS_DIR), WHAT_ARTIFACTS),
                 FilesRoot(_join(checkout, COLLECTION_DIR), WHAT_COLLECTION),
                 FilesRoot(_join(home, ARTIFACTS_DIR), WHAT_RECORDERS),
                 FilesRoot(_join(home, ARTIFACTS_DIR, STREAMS_DIR, SESSION_CUTS_DIR), WHAT_CUTS)]
        if record and session_id and not hooked:
            roots += _record_roots(host, session_id, record, project_root)
    seen, unique = set(), []
    for root in roots:
        key = _spelled(root.path)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


# ---- the host side: bash, the same on macOS and Linux ----

def _quote_root(path: str) -> str:
    """a root for the shell: $HOME stays expandable, everything else is quoted."""
    text = _spelled(path)
    if text == "$HOME":
        return '"$HOME"'
    if text.startswith("$HOME/"):
        return '"$HOME"/' + "/".join(shlex.quote(part) for part in text[len("$HOME/"):].split("/") if part)
    return shlex.quote(text)


def _check_id(session_id: str) -> None:
    problem = session_id_problem(session_id)
    if problem:
        raise ValueError(problem)


# every script starts here: no CDPATH (a relative cd would print the folder),
# and an id that names one folder, checked again on the host
_PRELUDE = (
    "unset CDPATH\n"
    "case \"$sid\" in ''|.*|*/*) echo 'REFUSED - id -'; echo PLANNED; echo DONE; exit 0;; esac\n"
    # the folder a root resolves to on that host, '' when there is none
    # (an empty name would be the folder the script runs in: never a root)
    "resolve() { [ -n \"$1\" ] || return 0; (cd -- \"$1\" 2>/dev/null && pwd -P) || true; }\n"
    # whether folder $1 holds an entry named exactly $sid as it is stored there:
    # on a file system that ignores case (macOS) $1/$sid also finds a folder whose
    # name differs in case, and pwd -P spells it as typed; a glob gives the stored
    # name (an id never starts with a dot, so * sees it)
    "named() { for e in \"$1\"/*; do [ \"${e##*/}\" = \"$sid\" ] && return 0; done; return 1; }\n"
    # the machine, the same whichever profile or address reached it
    "machine() {\n"
    "  m=$(cat /etc/machine-id 2>/dev/null || cat /var/lib/dbus/machine-id 2>/dev/null)\n"
    "  [ -n \"$m\" ] || m=$(ioreg -rd1 -c IOPlatformExpertDevice 2>/dev/null"
    " | awk -F'\"' '/IOPlatformUUID/{print $4}')\n"
    "  echo \"MACHINE ${m:--} $(hostname 2>/dev/null || echo -)\"\n"
    "}\n"
)

_LOOK = r'''
look() {
  i=$1; r=$(resolve "$2")
  if [ -z "$r" ]; then echo "ROOT $i -"; return 0; fi
  echo "ROOT $i $r"
  if [ "$r" = "/" ]; then echo "REFUSED $i root $r"; return 0; fi
  c="$r/$sid"
  # a folder whose name differs from the id in case only is another session's
  if ! named "$r"; then return 0; fi
  if [ -L "$c" ]; then echo "REFUSED $i link $c"; return 0; fi
  if [ ! -e "$c" ]; then return 0; fi
  if [ ! -d "$c" ]; then echo "REFUSED $i notdir $c"; return 0; fi
  if [ "$(resolve "$c")" != "$c" ]; then echo "REFUSED $i moved $c"; return 0; fi
  k=$(du -sk "$c" 2>/dev/null | cut -f1)
  n=$(find "$c" -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "HAS $i ${k:-0} ${n:-0} $c"
}
'''

_DELETE = r'''
del() {
  i=$1; want=$3; r=$(resolve "$2"); c="$r/$sid"
  if [ -z "$r" ] || [ "$c" != "$want" ]; then
    if [ -e "$want" ] || [ -L "$want" ]; then echo "REFUSED $i moved $want"; return 1; fi
    echo "GONE $i $want"; return 0
  fi
  if [ "$r" = "/" ]; then echo "REFUSED $i root $want"; return 1; fi
  if ! named "$r"; then
    if [ -e "$c" ] || [ -L "$c" ]; then echo "REFUSED $i case $want"; return 1; fi
    echo "GONE $i $want"; return 0
  fi
  if [ -L "$c" ]; then echo "REFUSED $i link $want"; return 1; fi
  if [ ! -e "$c" ]; then echo "GONE $i $want"; return 0; fi
  if [ ! -d "$c" ]; then echo "REFUSED $i notdir $want"; return 1; fi
  if [ "$(resolve "$c")" != "$c" ]; then echo "REFUSED $i moved $want"; return 1; fi
  rm -rf -- "$c" 2>/dev/null
  if [ -e "$c" ] || [ -L "$c" ]; then echo "FAILED $i $want"; return 1; fi
  echo "DELETED $i $want"
}
'''


def _roots_lines(roots: list[FilesRoot]) -> str:
    return "".join(f"d{index}={_quote_root(root.path)}\n" for index, root in enumerate(roots))


def _probe_lines(probes: list[str]) -> str:
    """PROBE <j> <the folder probe j resolves to, or ->: read, never deleted."""
    return "".join(f"p={_quote_root(path)}; p=$(resolve \"$p\"); echo \"PROBE {index} ${{p:--}}\"\n"
                   for index, path in enumerate(probes))


def plan_script(roots: list[FilesRoot], session_id: str, probes: list[str] = ()) -> str:
    """MACHINE <id> <hostname> first; for each root: ROOT <i> <the folder it
    resolves to, or ->, then HAS <i> <kB> <files> <path> for the session's
    folder there, or REFUSED <i> <link|notdir|moved|root> <path> for one that
    may not be deleted; PROBE <j> <path> for each of `probes` (folders only
    read); PLANNED at the end."""
    _check_id(session_id)
    calls = "".join(f"look {index} \"$d{index}\"\n" for index in range(len(roots)))
    return (f"sid={shlex.quote(session_id)}\n{_PRELUDE}{_LOOK}machine\n{_roots_lines(roots)}{calls}"
            f"{_probe_lines(list(probes))}echo PLANNED\n")


def probe_script(probes: list[str]) -> str:
    """MACHINE <id> <hostname>, PROBE <j> <path> for each of `probes`, then
    PROBED: what a host is and where folders lie there, nothing else."""
    return f"sid=-\n{_PRELUDE}machine\n{_probe_lines(list(probes))}echo PROBED\n"


def delete_script(roots: list[FilesRoot], session_id: str, folders: list[tuple[int, str]]) -> str:
    """delete each (root index, path) of a plan, in order, once its root still
    resolves so that the path is its child named the session id; stop at the
    first that fails. DELETED, GONE (not there any more), REFUSED <why> or
    FAILED per folder, STOPPED after a failure, DONE at the end."""
    _check_id(session_id)
    lines = []
    for index, path in folders:
        if not 0 <= index < len(roots):
            raise ValueError(f"no root {index}")
        lines.append(f"del {index} \"$d{index}\" {shlex.quote(path)} || {{ echo STOPPED; echo DONE; exit 0; }}\n")
    return f"sid={shlex.quote(session_id)}\n{_PRELUDE}{_DELETE}{_roots_lines(roots)}{''.join(lines)}echo DONE\n"


def listing_script(roots: list[FilesRoot]) -> str:
    """for each root: ROOT <i> <the folder it resolves to, or ->, then
    DIR <i> <kB> <name> for each folder in it (no link, nothing hidden, none of
    the folders every session shares); LISTED at the end."""
    from openmmla.utils.artifact_paths import NON_SESSION_ARTIFACT_DIRS

    shared = "|".join(shlex.quote(name) for name in sorted(NON_SESSION_ARTIFACT_DIRS))
    body = (
        "unset CDPATH\n"
        "resolve() { [ -n \"$1\" ] || return 0; (cd -- \"$1\" 2>/dev/null && pwd -P) || true; }\n"
        "lst() {\n"
        "  i=$1; r=$(resolve \"$2\")\n"
        "  if [ -z \"$r\" ] || [ \"$r\" = \"/\" ]; then echo \"ROOT $i -\"; return 0; fi\n"
        "  echo \"ROOT $i $r\"\n"
        "  for c in \"$r\"/*; do\n"
        "    n=${c##*/}\n"
        f"    case \"$n\" in {shared}|.*|'*') continue;; esac\n"
        "    if [ -L \"$c\" ] || [ ! -d \"$c\" ]; then continue; fi\n"
        "    k=$(du -sk \"$c\" 2>/dev/null | cut -f1)\n"
        "    echo \"DIR $i ${k:-0} $n\"\n"
        "  done\n"
        "}\n"
    )
    calls = "".join(f"lst {index} \"$d{index}\"\n" for index in range(len(roots)))
    return f"{body}{_roots_lines(roots)}{calls}echo LISTED\n"


def run_script(host: str, script: str, timeout: float, done: str, profile=None) -> tuple[str | None, str]:
    """(stdout, '') of a script run through bash on `host` ('local': this
    machine, else the SSH profile of that name, or `profile` when given) that
    ran to its `done` line. Otherwise (out, why), why never empty: out is
    None when the script was never started, else what it printed before it
    stopped ('' for nothing, or nothing that could be read), since a delete
    cut off midway may already have deleted what it printed. Its argument
    list is never shown: with a password in the profile it starts with
    sshpass -p."""
    from openmmla.tui.ssh import get_profile_by_name, ssh_error_text, ssh_run_sync
    from openmmla.tui.stream_cuts import bash

    if host != "local" and profile is None:
        profile = get_profile_by_name(host)
        if profile is None:
            return None, f"there is no SSH profile '{host}'"
    try:
        if host == "local":
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout)
        else:
            # an empty stdin: ssh must not read the console's own terminal
            result = ssh_run_sync(profile, bash(script), timeout=timeout, input_text="")
    except subprocess.TimeoutExpired as error:  # ssh_error_text: never the command line
        partial = error.stdout
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        return str(partial or ""), ssh_error_text(error) or "it timed out"
    except OSError as error:  # bash, ssh or sshpass could not be started
        return None, ssh_error_text(error) or type(error).__name__
    except Exception as error:  # it ran, but what it said cannot be read
        return "", ssh_error_text(error) or type(error).__name__
    out = result.stdout or ""
    if done not in [line.strip() for line in out.splitlines()]:
        lines = [line.strip() for line in str(result.stderr or "").splitlines() if line.strip()]
        return out, lines[-1] if lines else f"exit code {result.returncode}"
    return out, ""


# ---- what a host holds: the Sessions tab's listing ----

@dataclass
class HostListing:
    """the session folders of one host, with what they weigh."""
    host: str
    ok: bool = False
    why: str = ""                                            # why the host could not be asked
    roots: list[FilesRoot] = field(default_factory=list)
    resolved: dict[int, str] = field(default_factory=dict)   # root index -> the folder there
    sessions: dict[str, dict] = field(default_factory=dict)  # id -> {"bytes", "whats"}

    def size(self, session_id: str) -> int | None:
        entry = self.sessions.get(str(session_id or ""))
        return None if entry is None else int(entry["bytes"])


def parse_listing(text: str, listing: HostListing) -> HostListing:
    for line in str(text or "").splitlines():
        parts = line.rstrip("\r").split(" ", 3)
        if parts[0] == "ROOT" and len(parts) >= 3 and parts[1].isdigit():
            path = line.rstrip("\r").split(" ", 2)[2]
            if path != "-":
                listing.resolved[int(parts[1])] = path
        elif parts[0] == "DIR" and len(parts) == 4 and parts[1].isdigit() and parts[2].isdigit():
            index, kb, name = int(parts[1]), int(parts[2]), parts[3]
            if session_id_problem(name) or not 0 <= index < len(listing.roots):
                continue
            entry = listing.sessions.setdefault(name, {"bytes": 0, "whats": set(), "paths": set()})
            path = f"{listing.resolved.get(index, listing.roots[index].path)}/{name}"
            if path in entry["paths"]:
                continue  # two roots that are one folder there
            entry["paths"].add(path)
            entry["bytes"] += kb * 1024
            entry["whats"].add(listing.roots[index].what)
    return listing


def list_host_files(host: str, project_root=None) -> HostListing:
    """the session folders `host` holds under its roots (host_roots), each
    with its size, in one call to the host (du there). Blocking."""
    listing = HostListing(host, roots=host_roots(host, project_root))
    out, why = run_script(host, listing_script(listing.roots), LIST_TIMEOUT, "LISTED")
    if why or out is None:
        listing.why = why or "it did not answer"
        return listing
    listing.ok = True
    return parse_listing(out, listing)


# ---- one session's folders on one host ----

@dataclass
class Folder:
    """a folder of the session that a deletion takes."""
    root: int        # index of its root
    path: str        # as the host resolves it
    what: str
    kb: int
    files: int

    @property
    def bytes(self) -> int:
        return self.kb * 1024


_REFUSALS = {
    "link": "a symbolic link by the session's name, not its folder",
    "notdir": "a file by the session's name, not a folder",
    "moved": "it does not resolve to the folder of that name in its root",
    "root": "its root is the top of the file system",
    "id": "the id names no single folder",
    "case": "a folder whose name differs from the session id in letter case only, not its folder",
}


@dataclass
class HostPlan:
    """what of a session one host holds, as the host itself resolves it."""
    host: str                                                 # 'local' or an SSH profile
    session_id: str
    roots: list[FilesRoot] = field(default_factory=list)
    folders: list[Folder] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)   # (path, why)
    reached: bool = False
    why: str = ""                                             # why the host could not be asked
    machine: str = ""                                         # what the host says it is ('' unknown)
    probed: dict[int, str] = field(default_factory=dict)      # probe index -> the folder it resolves to

    @property
    def bytes(self) -> int:
        return sum(folder.bytes for folder in self.folders)


def _parse_probes(line: str, head: str, plan) -> None:
    """MACHINE and PROBE lines into `plan` (anything with machine and probed)."""
    if head == "MACHINE":
        parts = line.split(" ", 2)
        # the machine id and the hostname together: a cloned SD card shares its id
        if len(parts) == 3 and parts[1] != "-":
            plan.machine = f"{parts[1]} {parts[2]}"
    elif head == "PROBE":
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2] != "-":
            plan.probed[int(parts[1])] = parts[2]


def parse_plan(text: str, plan: HostPlan) -> HostPlan:
    seen = set()
    for line in str(text or "").splitlines():
        line = line.rstrip("\r")
        head = line.split(" ", 1)[0]
        _parse_probes(line, head, plan)
        if head == "HAS":
            parts = line.split(" ", 4)
            if len(parts) == 5 and parts[1].isdigit() and parts[2].isdigit() and parts[3].isdigit():
                index = int(parts[1])
                if 0 <= index < len(plan.roots) and parts[4] not in seen:
                    seen.add(parts[4])
                    plan.folders.append(Folder(index, parts[4], plan.roots[index].what, int(parts[2]),
                                               int(parts[3])))
        elif head == "REFUSED":
            parts = line.split(" ", 3)
            if len(parts) == 4:
                plan.refused.append((parts[3], _REFUSALS.get(parts[2], parts[2])))
    return plan


def plan_host(host: str, session_id: str, roots: list[FilesRoot], profile=None, probes: list[str] = ()) -> HostPlan:
    """the session's folders under `roots` on `host`, with sizes and file
    counts, what machine it is, and where each of `probes` resolves there,
    in one call. Blocking."""
    plan = HostPlan(host, session_id, roots=list(roots))
    out, why = run_script(host, plan_script(plan.roots, session_id, probes), PLAN_TIMEOUT, "PLANNED",
                          profile=profile)
    if why or out is None:
        plan.why = why or "it did not answer"
        return plan
    plan.reached = True
    return parse_plan(out, plan)


@dataclass
class HostProbe:
    """what machine a host is, and where folders resolve there."""
    machine: str = ""
    probed: dict[int, str] = field(default_factory=dict)
    why: str = ""


def probe_host(host: str, probes: list[str], profile=None) -> HostProbe:
    """the machine `host` is and where each of `probes` resolves there; reads
    only. Blocking."""
    probe = HostProbe()
    out, why = run_script(host, probe_script(probes), LIST_TIMEOUT, "PROBED", profile=profile)
    if why or out is None:
        probe.why = why or "it did not answer"
        return probe
    for line in out.splitlines():
        line = line.rstrip("\r")
        _parse_probes(line, line.split(" ", 1)[0], probe)
    return probe


@dataclass
class Outcome:
    """what became of each folder of a plan."""
    deleted: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)   # (path, why)
    why: str = ""                                                  # the host could not be asked, or stopped
    cut_off: bool = False                                          # it stopped once the delete was sent

    @property
    def ok(self) -> bool:
        return not self.why and not self.failed


def delete_on_host(plan: HostPlan, folders: list[Folder], profile=None) -> Outcome:
    """delete `folders` of `plan` on its host, in order, each checked again
    there first; stops at the first that fails. A host that stops answering
    once the delete was sent (cut_off) may have deleted some: what it said
    before is kept. Blocking."""
    outcome = Outcome()
    if not folders:
        return outcome
    script = delete_script(plan.roots, plan.session_id, [(folder.root, folder.path) for folder in folders])
    out, why = run_script(plan.host, script, DELETE_TIMEOUT, "DONE", profile=profile)
    if why or out is None:
        outcome.why = why or "it did not answer"
        outcome.cut_off = out is not None
    for line in (out or "").splitlines():
        line = line.rstrip("\r")
        head = line.split(" ", 1)[0]
        parts = line.split(" ", 2)
        if head == "DELETED" and len(parts) == 3:
            outcome.deleted.append(parts[2])
        elif head == "GONE" and len(parts) == 3:
            outcome.gone.append(parts[2])
        elif head == "FAILED" and len(parts) == 3:
            outcome.failed.append((parts[2], "it could not be removed whole (permissions, or a file still open)"))
        elif head == "REFUSED":
            parts = line.split(" ", 3)
            if len(parts) == 4:
                outcome.failed.append((parts[3], _REFUSALS.get(parts[2], parts[2])))
    done = set(outcome.deleted) | set(outcome.gone) | {path for path, _ in outcome.failed}
    if not outcome.why and not outcome.failed and any(folder.path not in done for folder in folders):
        outcome.failed.append(("", "the host did not say what became of every folder"))
    return outcome


def _where(host: str) -> str:
    from openmmla.utils.artifact_paths import short_hostname

    return f"this machine ({short_hostname()})" if host == "local" else host


# ---- Delete Files: one host ----

@dataclass
class FilesPlan:
    """Delete Files: the session's folders on the host picked."""
    found: HostPlan
    archive_copies: set[str] = field(default_factory=set)    # paths that are the archive's copy too
    notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers and bool(self.found.folders)


def plan_files(host: str, session_id: str, *, project_root=None, record: dict | None = None,
               settings_root=None) -> FilesPlan:
    """what Delete Files takes of `session_id` on `host`. Blocking (one call to
    the host, the archive host's name resolved, and, when the host picked is
    not the archive host's own login, one read on the archive host)."""
    from concurrent.futures import ThreadPoolExecutor

    from openmmla.commands.ses import archive

    plan = FilesPlan(HostPlan(host, session_id))
    problem = session_id_problem(session_id)
    if problem:
        plan.blockers.append(f"{problem}: nothing is deleted")
        return plan
    plan.notes.extend(hook_notes())
    roots = host_roots(host, project_root, record, session_id)
    # the archive host's own copy is the folder ses-archive writes there. Which
    # machine the host picked is, and where that folder lies, both hosts say
    # themselves: another SSH profile, or an address, may reach the archive host
    try:
        target = archive.resolve_target(None, settings_root)
    except Exception:
        target = None
    archive_root = _join(_spelled(target.root), ARTIFACTS_DIR) if target is not None else ""
    same_login = target is not None and (target.name == host or (target.here and host == "local"))
    archive_probe = None
    if target is not None and not same_login:
        with ThreadPoolExecutor(max_workers=1) as pool:
            asked = pool.submit(probe_host, "local" if target.here else target.name, [archive_root],
                                None if target.here else target.profile)
            plan.found = plan_host(host, session_id, roots)
            archive_probe = asked.result()
    else:
        plan.found = plan_host(host, session_id, roots, probes=[archive_root] if same_login else [])
    if not plan.found.reached:
        plan.blockers.append(f"{_where(host)} could not be asked ({plan.found.why}): nothing is deleted")
        return plan
    for path, why in plan.found.refused:
        plan.blockers.append(f"{path}: {why}. Nothing is deleted until it is sorted out by hand")
    resolved = ""
    if same_login:
        resolved = plan.found.probed.get(0, "")
    elif archive_probe is not None and archive_probe.why:
        if plan.found.folders:
            plan.notes.append(f"  [yellow]The archive host {_escape(target.name)} could not be asked whether one "
                              f"of these folders is its archive copy ({_escape(archive_probe.why)}).[/yellow]")
    elif archive_probe is not None and archive_probe.machine and archive_probe.machine == plan.found.machine:
        resolved = archive_probe.probed.get(0, "")
    if resolved and resolved != "/":
        for folder in plan.found.folders:
            if folder.path == f"{resolved}/{session_id}":
                plan.archive_copies.add(folder.path)
    if record and str(record.get("status") or "") == "active":
        plan.notes.append("  [yellow]The session is still active in MongoDB: a recorder or base may still write "
                          "into these folders.[/yellow]")
    return plan


def describe_files_plan(plan: FilesPlan, lead: str = "Click Delete Files again to delete") -> list[str]:
    """the first press's lines (markup): what goes, or why nothing can."""
    sid, where = _escape(plan.found.session_id), _escape(_where(plan.found.host))
    lines = list(plan.notes)
    if plan.blockers:
        return lines + [f"[red]✗ {_escape(text)}.[/red]" for text in plan.blockers]
    if not plan.found.folders:
        roots = ", ".join(_escape(_shown(root.path)) for root in plan.found.roots)
        return lines + [f"[yellow]{where} holds no files of '{sid}' (looked in {roots}): nothing to delete."
                        f"[/yellow]"]
    # the log does not wrap: the conclusion first, and each folder's size before its path
    count = len(plan.found.folders)
    lines.insert(0, f"[red]{lead} the files of '{sid}' on {where}: {count} folder(s), {_human(plan.found.bytes)}. "
                    f"Nothing elsewhere is touched.[/red]")
    if plan.archive_copies:
        lines.insert(1, "  [red]One of them is the archive copy too: the copy the dashboard and replays read, and "
                        "deleting it cannot be undone.[/red]")
    for folder in plan.found.folders:
        copy = ", the archive copy too" if folder.path in plan.archive_copies else ""
        lines.append(f"  [red]{_escape(folder.what)}, {folder.files} file(s), {_human(folder.bytes)}{copy}: "
                     f"{_escape(folder.path)}[/red]")
    return lines


def delete_files(plan: FilesPlan, log, header: bool = True) -> bool:
    """Delete Files, second press: exactly the folders the first press named,
    each checked again on the host first; True when all of them are gone."""
    sid, where = _escape(plan.found.session_id), _escape(_where(plan.found.host))
    if header:
        log(f"[bold red]Deleting the files of '{sid}' on {where}[/bold red]")
    outcome = delete_on_host(plan.found, plan.found.folders)
    return _log_outcome(outcome, where, log, f"Files of '{sid}' deleted on {where}")


def _log_outcome(outcome: Outcome, where: str, log, done_text: str) -> bool:
    for path in outcome.deleted:
        log(f"  [green]✓[/green] Deleted {_escape(path)}")
    for path in outcome.gone:
        log(f"  [dim]- {_escape(path)} was not there any more[/dim]")
    for path, why in outcome.failed:
        log(f"  [red]✗ {_escape(path) + ': ' if path else ''}{_escape(why)}; nothing after it is deleted[/red]")
    if outcome.why and outcome.cut_off:
        log(f"  [red]✗ {where} stopped answering during the delete ({_escape(outcome.why)}): what is listed "
            f"above was deleted, the rest may be partly deleted. Look again to see what is left[/red]")
    elif outcome.why:
        log(f"  [red]✗ {where} could not be asked ({_escape(outcome.why)}): nothing is deleted[/red]")
    if outcome.ok:
        log(f"[bold green]{done_text}[/bold green]")
    return outcome.ok


# ---- Delete Session: everywhere central ----

@dataclass
class SessionPlan:
    """Delete Session: what goes, read just before the question."""
    session_id: str
    archive: object = None                                    # archive.ArchiveTarget, None when not known
    archive_plan: HostPlan | None = None                      # its archive folder and dashboard cache there
    influx_counts: dict[str, int] | None = None               # None: no InfluxDB in System Settings
    mongo_exists: bool | None = None                          # None: no MongoDB in System Settings
    mongo_status: str = ""
    kept: list[str] = field(default_factory=list)             # copies that stay (markup)
    notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def folders(self, what: str) -> list[Folder]:
        if self.archive_plan is None:
            return []
        return [folder for folder in self.archive_plan.folders if folder.what == what]

    @property
    def events(self) -> int:
        return sum((self.influx_counts or {}).values())

    @property
    def empty(self) -> bool:
        return not (self.archive_plan and self.archive_plan.folders) and not self.events and not self.mongo_exists

    @property
    def ready(self) -> bool:
        return not self.blockers and not self.empty


def _local_size(path: Path) -> int | None:
    """bytes of a local folder, links not followed; None when it is not there
    under exactly its name (macOS also finds a name that differs in case)."""
    if path.is_symlink() or not path.is_dir():
        return None
    try:
        if path.name not in os.listdir(path.parent):
            return None
    except OSError:
        return None
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _mongo_document(mongo, session_id: str) -> dict | None:
    return mongo.sessions.find_one(
        {"session_id": session_id},
        {"_id": 0, "session_id": 1, "status": 1, "collection_hosts": 1, "sources": 1})


def plan_session(session_id: str, *, influx=None, influx_configured: bool = True, mongo=None,
                 mongo_configured: bool = True, project_root=None, settings_root=None) -> SessionPlan:
    """what Delete Session takes of `session_id`. `influx` is an
    InfluxDBClientWrapper and `mongo` anything whose `sessions` is the MongoDB
    collection; each None while it is not connected, which stops the delete
    when System Settings name it (`*_configured`). Blocking: the archive host
    is asked over SSH, both databases are read."""
    from openmmla.commands.ses import archive
    from openmmla.commands.ses.export import COLLECTION_HOSTS_FIELD
    from openmmla.utils import session_sources

    plan = SessionPlan(session_id)
    problem = session_id_problem(session_id)
    if problem:
        plan.blockers.append(f"{problem}: nothing is deleted")
        return plan
    plan.notes.extend(hook_notes())
    if project_root is None:
        from openmmla.tui.schema.loader import _find_project_root

        project_root = _find_project_root()

    # the archive host and what it holds; nothing goes while it cannot be asked
    try:
        target = archive.resolve_target(None, settings_root)
    except Exception as error:  # ArchiveError, or a profile file that cannot be read
        plan.blockers.append(f"The archive host is not known ({error}): nothing is deleted")
        target = None
    if target is not None:
        plan.archive = target
        root = target.root.rstrip("/")
        roots = [FilesRoot(f"{root}/{ARTIFACTS_DIR}", WHAT_ARCHIVE),
                 FilesRoot(f"{root}/{DASHBOARD_CACHE_REL}", WHAT_CACHE)]
        host = "local" if target.here else target.name
        plan.archive_plan = plan_host(host, session_id, roots, profile=None if target.here else target.profile)
        if not plan.archive_plan.reached:
            plan.blockers.append(f"The archive host {target.name} could not be asked ({plan.archive_plan.why}): "
                                 f"nothing is deleted")
        for path, why in plan.archive_plan.refused:
            plan.blockers.append(f"{target.name}:{path}: {why}. Nothing is deleted until it is sorted out by hand")

    # InfluxDB: the events by type
    if influx is None:
        if influx_configured:
            plan.blockers.append("InfluxDB is not connected, so its events of the session cannot be deleted: "
                                 "nothing is deleted. Refresh, or start InfluxDB")
    else:
        try:
            plan.influx_counts = dict(influx.count_session_events_by_type(session_id))
        except Exception as error:  # its text may carry the URL: only its kind is said
            plan.blockers.append(f"InfluxDB did not answer ({type(error).__name__}): nothing is deleted")

    # MongoDB: the document
    record = None
    if mongo is None:
        if mongo_configured:
            plan.blockers.append("MongoDB is not connected, so the session's document cannot be deleted: nothing "
                                 "is deleted. Refresh, or start MongoDB")
    else:
        try:
            record = _mongo_document(mongo, session_id)
        except Exception as error:
            plan.blockers.append(f"MongoDB did not answer ({type(error).__name__}): nothing is deleted")
        else:
            plan.mongo_exists = bool(record)
            plan.mongo_status = str((record or {}).get("status") or "")
    if plan.mongo_status == "active":
        plan.notes.append("  [yellow]The session is still active: a base or recorder that is still running may "
                          "write into it again.[/yellow]")

    # what stays: this console's copy, unless it is the archive, and every other machine's
    archive_paths = set()
    if target is not None and target.here:
        archive_paths = {os.path.realpath(folder.path) for folder in plan.folders(WHAT_ARCHIVE)}
    for folder in (ARTIFACTS_DIR, COLLECTION_DIR):
        path = Path(project_root) / folder / session_id
        size = _local_size(path)
        if size is None:
            continue
        if os.path.realpath(path) in archive_paths:
            plan.notes.append(f"  [red]This console is the archive host: its {folder}/{_escape(session_id)}/ is the "
                              f"archive, and goes with it.[/red]")
            continue
        plan.kept.append(f"this console's {folder}/{_escape(session_id)}/ ({_human(size)})")
    hosts = [str(name) for name in (record or {}).get(COLLECTION_HOSTS_FIELD) or [] if str(name or "").strip()]
    hosts += [entry["ssh_profile"] for entry in session_sources.captured_streams(record)
              if entry["record"] and entry["ssh_profile"] and entry["ssh_profile"] != "local"]
    named = ", ".join(dict.fromkeys(hosts))
    plan.kept.append("the files on the capture hosts" + (f" ({_escape(named)})" if named else "")
                     + " and on other machines (Delete Files, host by host)")
    plan.kept.append("the Stream Server's recordings, which the sessions that use a stream share")
    return plan


def describe_session_plan(plan: SessionPlan, lead: str = "Click Delete Session again to delete") -> list[str]:
    """the first press's lines (markup): what goes, in red, and what stays; or
    why nothing can go."""
    sid = _escape(plan.session_id)
    lines = list(plan.notes)
    if plan.blockers:
        return lines + [f"[red]✗ {_escape(text)}.[/red]" for text in plan.blockers]
    if plan.empty:
        return lines + [f"[yellow]Nothing of '{sid}' is kept centrally (no archive, no InfluxDB events, no MongoDB "
                        f"document): nothing to delete. Delete Files removes its files on the host picked.[/yellow]"]
    target_name = _escape(plan.archive.name) if plan.archive is not None else "?"
    archive = plan.folders(WHAT_ARCHIVE)
    goes = []
    if archive:
        goes.append(f"its archive on {target_name}")
    if plan.events:
        goes.append(f"{plan.events} InfluxDB event(s)")
    if plan.mongo_exists:
        goes.append("its MongoDB document")
    goes = goes or ["its dashboard cache"]
    listed = goes[0] if len(goes) == 1 else f"{', '.join(goes[:-1])} and {goes[-1]}"
    # the log does not wrap: the conclusion first, and each folder's size before its path
    lines.insert(0, f"[red]{lead} '{sid}' everywhere central: {listed}.[/red]")
    if archive:
        lines.insert(1, "  [red]Deleting the archive cannot be undone; for an imported classroom session it holds "
                        "the raw videos.[/red]")
    for folder in archive:
        lines.append(f"  [red]Archive on {target_name}, {folder.files} file(s), {_human(folder.bytes)} (with its "
                     f"ledger): {_escape(folder.path)}[/red]")
    if not archive and plan.archive is not None:
        lines.append(f"  [dim]- No archive on {target_name}.[/dim]")
    for folder in plan.folders(WHAT_CACHE):
        lines.append(f"  [red]Dashboard cache on {target_name}, {_human(folder.bytes)}: {_escape(folder.path)}[/red]")
    if plan.influx_counts is None:
        lines.append("  [dim]- InfluxDB: System Settings name none.[/dim]")
    elif plan.events:
        by_type = ", ".join(f"{_escape(kind)} {count}" for kind, count in sorted(plan.influx_counts.items()))
        lines.append(f"  [red]InfluxDB: {plan.events} event(s) ({by_type})[/red]")
    else:
        lines.append("  [dim]- InfluxDB: no events of it.[/dim]")
    if plan.mongo_exists is None:
        lines.append("  [dim]- MongoDB: System Settings name none.[/dim]")
    elif plan.mongo_exists:
        status = f" (status {_escape(plan.mongo_status)})" if plan.mongo_status else ""
        lines.append(f"  [red]MongoDB: its document{status}[/red]")
    else:
        lines.append("  [dim]- MongoDB: no document of it.[/dim]")
    for text in plan.kept:
        lines.append(f"  Stays: {text}")
    return lines


def delete_session(plan: SessionPlan, *, influx=None, mongo=None, log=print, header: bool = True) -> bool:
    """Delete Session, second press: the archive (and the dashboard cache) on
    the archive host, then InfluxDB, then MongoDB, each step logged, stopping
    at the first that fails. True once all of it is gone. Blocking."""
    sid = plan.session_id
    _check_id(sid)  # never a delete whose predicate or path is anything but this one session
    shown = _escape(sid)
    if header:
        log(f"[bold red]Deleting session: {shown}[/bold red]")

    # 1. the archive host: the archive folder first, then the dashboard cache
    if plan.archive_plan is not None and plan.archive_plan.folders:
        target = plan.archive
        folders = plan.folders(WHAT_ARCHIVE) + plan.folders(WHAT_CACHE)
        outcome = delete_on_host(plan.archive_plan, folders, profile=None if target.here else target.profile)
        where = _escape(target.name)
        for path in outcome.deleted:
            what = "Archive" if any(folder.path == path for folder in plan.folders(WHAT_ARCHIVE)) else "Dashboard cache"
            log(f"  [green]✓[/green] {what} deleted on {where}: {_escape(path)}")
        for path in outcome.gone:
            log(f"  [dim]- {where}:{_escape(path)} was not there any more[/dim]")
        if not outcome.ok:
            for path, why in outcome.failed:
                log(f"  [red]✗ {where}:{_escape(path)}: {_escape(why)}[/red]" if path else
                    f"  [red]✗ {where}: {_escape(why)}[/red]")
            if outcome.why and outcome.cut_off:
                log(f"  [red]✗ {where} stopped answering during the delete ({_escape(outcome.why)}): what is "
                    f"listed above was deleted, the rest may be partly deleted[/red]")
            elif outcome.why:
                log(f"  [red]✗ {where} could not be asked ({_escape(outcome.why)})[/red]")
            log(f"[red]Stopped: nothing after it was deleted, so InfluxDB and MongoDB still hold '{shown}'. Delete "
                f"it again to go on.[/red]")
            return False
    else:
        log("  [dim]- No archive to delete[/dim]")

    # 2. InfluxDB, counted again afterwards
    if plan.influx_counts is not None:
        if influx is None:
            log("  [red]✗ InfluxDB is not connected any more[/red]")
            log(f"[red]Stopped: MongoDB still holds '{shown}'. Delete it again to go on.[/red]")
            return False
        try:
            before = sum(influx.count_session_events_by_type(sid).values())
            if before:
                if not influx.delete_session_data(sid):
                    raise RuntimeError("InfluxDB did not take the delete")
                left = sum(influx.count_session_events_by_type(sid).values())
                if left:
                    raise RuntimeError(f"{left} event(s) are still there")
                log(f"  [green]✓[/green] InfluxDB: {before} event(s) deleted")
            else:
                log("  [dim]- InfluxDB: no events of it[/dim]")
        except Exception as error:
            text = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            log(f"  [red]✗ InfluxDB: {_escape(text)}[/red]")
            log(f"[red]Stopped: MongoDB still holds '{shown}'. Delete it again to go on.[/red]")
            return False

    # 3. MongoDB
    if plan.mongo_exists is not None:
        if mongo is None:
            log("  [red]✗ MongoDB is not connected any more[/red]")
            return False
        try:
            result = mongo.sessions.delete_one({"session_id": sid})
        except Exception as error:
            log(f"  [red]✗ MongoDB did not answer ({type(error).__name__})[/red]")
            return False
        if getattr(result, "deleted_count", 0):
            log("  [green]✓[/green] MongoDB: its document deleted")
        else:
            log("  [dim]- MongoDB: no document of it[/dim]")
    log(f"[bold green]Session '{shown}' deleted everywhere central.[/bold green]")
    return True


# ---- the command ----

def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-delete",
        description="Delete a session. Without --files-on: everywhere central, as the Sessions tab's Delete Session "
                    "does: its archive on the archive host (and the dashboard's cached report there), then its "
                    "InfluxDB events, then its MongoDB document, stopping at the first step that fails. With "
                    "--files-on HOST: its folders on that one host, as Delete Files does. Without --yes it says "
                    "what would go and deletes nothing.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=40, width=120),
    )
    parser.add_argument("session_id", help="the session")
    parser.add_argument("--files-on", default=None, metavar="HOST",
                        help="delete the session's files on HOST ('local' for this machine, or an SSH profile) "
                             "instead of the central delete")
    parser.add_argument("--yes", action="store_true", help="delete; without it nothing is deleted")
    return parser


def main():
    import contextlib

    from openmmla.commands.ses import archive, export

    parser = get_parser()
    args = parser.parse_args()
    session_id = str(args.session_id or "").strip()
    problem = session_id_problem(session_id)
    if problem:
        parser.error(problem)
    # an empty host (--files-on "$HOST" with HOST unset) is a mistake, never the central delete
    if args.files_on is not None and not str(args.files_on).strip():
        parser.error("--files-on needs a host: local or an SSH profile")
    callbacks = archive._ConsoleCallbacks()
    log = callbacks.log
    settings_root = archive._settings_root()
    mongo, why = archive.open_mongo(settings_root)
    if mongo is None:
        log(f"[yellow]MongoDB: {_escape(why)}.[/yellow]")
    influx = None
    try:
        if args.files_on is not None:
            host = str(args.files_on).strip()
            from openmmla.tui.ssh import get_profile_by_name

            if host != "local" and get_profile_by_name(host) is None:
                log(f"[red]✗ There is no SSH profile '{_escape(host)}': name one of System Settings → Hosts → "
                    f"SSH Profiles, or 'local'.[/red]")
                sys.exit(2)
            record = None
            if mongo is not None:
                with contextlib.suppress(Exception):
                    record = archive._find_record(mongo, session_id)[0]
            plan = plan_files(host, session_id, project_root=settings_root, record=record,
                              settings_root=settings_root)
            for line in describe_files_plan(plan, "Run again with --yes to delete" if not args.yes else "Deleting"):
                log(line)
            if plan.blockers:
                sys.exit(2)
            if not plan.ready or not args.yes:
                if plan.ready:
                    log("[yellow]Nothing was deleted: run it again with --yes.[/yellow]")
                sys.exit(0)
            sys.exit(0 if delete_files(plan, log, header=False) else 1)
        from openmmla.tui.system_services import load_system_services_config, section_address_set

        config = load_system_services_config(settings_root) or {}
        influx_configured = section_address_set(config.get("InfluxDB"), "InfluxDB")
        mongo_configured = section_address_set(config.get("MongoDB"), "MongoDB")
        if influx_configured:
            influx, why = export.open_influx(settings_root)
            if influx is None:
                log(f"[yellow]InfluxDB: {_escape(why)}.[/yellow]")
        plan = plan_session(session_id, influx=influx, influx_configured=influx_configured, mongo=mongo,
                            mongo_configured=mongo_configured, project_root=settings_root,
                            settings_root=settings_root)
        for line in describe_session_plan(plan, "Run again with --yes to delete" if not args.yes else "Deleting"):
            log(line)
        if plan.blockers:
            sys.exit(2)
        if not plan.ready or not args.yes:
            if plan.ready:
                log("[yellow]Nothing was deleted: run it again with --yes.[/yellow]")
            sys.exit(0)
        sys.exit(0 if delete_session(plan, influx=influx, mongo=mongo, log=log, header=False) else 1)
    finally:
        for client in (mongo, influx):
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close()
