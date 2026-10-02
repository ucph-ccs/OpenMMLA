"""On-disk cache of computed report parts and job states.

The dashboard computes a session's report once and serves it from here, so a page load never waits
on InfluxDB for data that has not changed. The cache is plain files under one directory, shared by
the web process and the Celery worker on the same host: `<sid>/<part>.json` holds a part in an
envelope that remembers which InfluxDB state it was computed from (`source_last_event`), and
`<sid>/<job>.status.json` says whether a job is queued, running, done or failed. Every write goes
through a temporary file and `os.replace`, so a reader never sees half a file.

A store made without a folder (`ReportStore()`: every process that loads dashboard.py, the web
process, the worker and a `python dashboard.py precompute` run alike) also adds the cache folder it
uses to `cache.location` beside this file: one absolute path a line, the latest last, at most
LOCATION_MAX_LINES. `mmla ses-delete` (Delete Session) deletes a session's cache in each folder listed
there and at the default place, so a cache that `DASHBOARD_CACHE_DIR` put elsewhere is found even
after a process with another folder started.
"""

import json
import logging
import os
import re
import tempfile
import threading
import time

logger = logging.getLogger("dashboard.store")

JOBS = {"light": ("speech", "space"), "video": ("attention", "timeline")}
PART_JOB = {part: job for job, parts in JOBS.items() for part in parts}
JOB_STATES = ("queued", "running", "done", "error")

# the same rule as openmmla.analytics.report.common.SESSION_ID_RE, repeated so the store never
# builds a path from an unchecked id even when that package cannot be imported
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_STATUS_FIELDS = ("state", "step", "done", "total", "started_at", "updated_at", "runner", "error",
                  "source_last_event", "pid")


_HEADER_RE = re.compile(r'^\{"part":"[a-z]+","computed_at":([-0-9.eE+]+|null),"source_last_event":([-0-9.eE+]+|null)'
                        r'(?:,"version":([0-9]+|null))?,"data":')


def _number(text: str) -> float | None:
    try:
        return None if text == "null" else float(text)
    except ValueError:
        return None


def default_cache_dir() -> str:
    """`DASHBOARD_CACHE_DIR` when set, else `cache/` next to this file."""
    configured = os.environ.get("DASHBOARD_CACHE_DIR", "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


# the pointer to the cache folders in use, beside this file (gitignored): every folder a store of
# this checkout kept its cache in, so one process started with another DASHBOARD_CACHE_DIR never
# hides the folder a running dashboard uses; openmmla's commands/ses/delete.py reads it at
# DASHBOARD_CACHE_POINTER_REL with the same limits (POINTER_MAX_BYTES, POINTER_MAX_LINES)
LOCATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.location")
LOCATION_MAX_BYTES = 8192
LOCATION_MAX_LINES = 16


def _nameable(line: bytes) -> bool:
    """whether a pointer line can hold this folder: an absolute path without control characters
    that fits in the file with its newline."""
    return (line.startswith(b"/") and len(line) < LOCATION_MAX_BYTES
            and not any(byte < 32 or byte == 127 for byte in line))


def read_locations(path: str | None = None) -> list[bytes]:
    """the folders the pointer file lists (LOCATION_FILE unless `path`), as bytes, oldest first, each
    once; a line that cannot name a folder, a line without its newline, a link and a file that cannot
    be read count as nothing."""
    path = path or LOCATION_FILE
    try:
        if os.path.islink(path):
            return []
        with open(path, "rb") as handle:
            data = handle.read(LOCATION_MAX_BYTES + 1)
    except OSError:
        return []
    if b"\0" in data:
        return []
    lines = []
    for line in data[:LOCATION_MAX_BYTES].split(b"\n")[:-1]:
        if _nameable(line) and line not in lines:
            lines.append(line)
    return lines


def write_location(cache_dir: str, path: str | None = None) -> bool:
    """add `cache_dir` to the pointer file (LOCATION_FILE unless `path`) as its last line: its absolute
    path and a newline, the folders listed before kept (the oldest dropped past LOCATION_MAX_LINES or
    LOCATION_MAX_BYTES), through a temporary file and `os.replace`; left alone when it already ends
    with it. False when it cannot: a checkout that cannot be written, or a folder no line can name (a
    control character, or too long), only means a delete does not look there."""
    path = path or LOCATION_FILE
    folder = os.path.abspath(cache_dir)
    # bytes as the file system has them: a name that is no UTF-8 still goes as it is
    line = os.fsencode(folder)
    if not _nameable(line):
        logger.warning("the cache folder %r cannot be named in %s: Delete Session does not look there", folder, path)
        return False
    lines = [entry for entry in read_locations(path) if entry != line] + [line]
    while len(lines) > LOCATION_MAX_LINES or sum(len(entry) + 1 for entry in lines) > LOCATION_MAX_BYTES:
        lines.pop(0)
    data = b"".join(entry + b"\n" for entry in lines)
    try:
        if not os.path.islink(path):
            with open(path, "rb") as handle:
                if handle.read(LOCATION_MAX_BYTES + 1) == data:
                    return True
    except OSError:
        pass
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=".cache.location.", dir=os.path.dirname(path))
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        # only paths: readable by whoever deletes a session over SSH
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        return True
    except (OSError, ValueError) as exc:
        logger.warning("could not write %s (%s): Delete Session does not look in %s unless it is listed already "
                       "or is the default place", path, exc, folder)
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def _dumps(obj) -> str:
    try:
        return json.dumps(obj, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        # a part that slipped a numpy scalar or a NaN through is still worth keeping
        from openmmla.analytics.report.common import jsonable
        return json.dumps(jsonable(obj), allow_nan=False, separators=(",", ":"))


class ReportStore:
    """Report parts, job states and the fused window table of each session, one folder per session."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = os.path.abspath(cache_dir or default_cache_dir())
        self._memo: dict[str, tuple[tuple, dict]] = {}
        self._memo_lock = threading.Lock()
        if not cache_dir:
            # the store of every process that loads dashboard.py: add where its cache is. A store
            # given its folder (a test, a script) leaves the pointer alone
            try:
                write_location(self.cache_dir)
            except Exception as exc:  # the pointer helps a delete; it never stops the dashboard
                logger.warning("could not write the cache pointer: %s", exc)

    def session_dir(self, sid: str, create: bool = False) -> str:
        if not isinstance(sid, str) or not _SESSION_ID_RE.match(sid) or sid in (".", ".."):
            raise ValueError(f"invalid session id {sid!r}")
        path = os.path.join(self.cache_dir, sid)
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def _write_atomic(self, path: str, text: str) -> None:
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=os.path.splitext(path)[1], dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_json(self, path: str) -> dict | None:
        """the parsed file, memoised by mtime and size so a large part is parsed once per change."""
        try:
            stat = os.stat(path)
        except OSError:
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        with self._memo_lock:
            hit = self._memo.get(path)
            if hit and hit[0] == key:
                return hit[1]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("unreadable cache file %s: %s", path, exc)
            return None
        if not isinstance(value, dict):
            return None
        with self._memo_lock:
            if len(self._memo) > 256:
                self._memo.clear()
            self._memo[path] = (key, value)
        return value

    def part_path(self, sid: str, part: str) -> str:
        if part not in PART_JOB:
            raise ValueError(f"unknown report part {part!r}")
        return os.path.join(self.session_dir(sid), f"{part}.json")

    def read_part(self, sid: str, part: str) -> dict | None:
        """the envelope {part, computed_at, source_last_event, data}, or None when not computed."""
        envelope = self._read_json(self.part_path(sid, part))
        if envelope is None or "data" not in envelope:
            return None
        return envelope

    def read_part_header(self, sid: str, part: str) -> dict | None:
        """{computed_at, source_last_event, version} of a cached part without parsing its data.

        write_part puts both before `data`, so the first few hundred bytes hold them; the session
        list asks this for every session and part, and a part can be megabytes."""
        path = self.part_path(sid, part)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                head = handle.read(512)
        except OSError:
            return None
        match = _HEADER_RE.match(head)
        if match:
            version = _number(match.group(3)) if match.group(3) else None
            return {"computed_at": _number(match.group(1)), "source_last_event": _number(match.group(2)),
                    "version": int(version) if version is not None else None}
        envelope = self.read_part(sid, part)
        if envelope is None:
            return None
        return {"computed_at": envelope.get("computed_at"), "source_last_event": envelope.get("source_last_event"),
                "version": envelope.get("version")}

    def write_part(self, sid: str, part: str, data, source_last_event: float | None,
                   version: int | None = None) -> dict:
        envelope = {"part": part, "computed_at": round(time.time(), 3),
                    "source_last_event": source_last_event, "version": version, "data": data}
        self.session_dir(sid, create=True)
        self._write_atomic(self.part_path(sid, part), _dumps(envelope))
        return envelope

    def status_path(self, sid: str, job: str) -> str:
        if job not in JOBS:
            raise ValueError(f"unknown report job {job!r}")
        return os.path.join(self.session_dir(sid), f"{job}.status.json")

    def read_status(self, sid: str, job: str) -> dict | None:
        status = self._read_json(self.status_path(sid, job))
        if status is None or status.get("state") not in JOB_STATES:
            return None
        return status

    def write_status(self, sid: str, job: str, **fields) -> dict:
        """merge `fields` into the job's status and stamp `updated_at`."""
        status = dict(self.read_status(sid, job) or {})
        status.update({key: value for key, value in fields.items() if key in _STATUS_FIELDS})
        status["updated_at"] = round(time.time(), 3)
        for key in _STATUS_FIELDS:
            status.setdefault(key, None)
        self.session_dir(sid, create=True)
        self._write_atomic(self.status_path(sid, job), _dumps(status))
        return status

    def window_features_path(self, sid: str) -> str:
        return os.path.join(self.session_dir(sid), "window_features.csv")

    def has_window_features(self, sid: str) -> bool:
        return os.path.isfile(self.window_features_path(sid))

    def write_window_rows(self, sid: str, rows: list[dict]) -> str:
        """the fused window table as CSV, through window_features.write_table and an atomic rename."""
        from openmmla.analytics.fusion.window_features import write_table
        path = self.window_features_path(sid)
        folder = self.session_dir(sid, create=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".csv", dir=folder)
        os.close(fd)
        try:
            write_table(rows, tmp)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path
