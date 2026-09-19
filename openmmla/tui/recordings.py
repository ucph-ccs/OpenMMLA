"""server-side recordings: what the stream server (MediaMTX) recorded, by time.

MediaMTX records every published path whether or not a session runs, and its
playback server returns any time range of a path as one file. A stream is
shared by the sessions that pull it, so the footage of one session is a query
and not a folder: the paths its bases pulled (they note them in the session's
document, openmmla.utils.session_sources), each cut to the window between the
session's start and end.

The server keeps a segment for `recordDeleteAfter` and then deletes it itself,
so a session's footage has to be exported before then; this module also asks
the server what it holds and for how long, and removes segments through its
API, for the Recordings tab of the Stream Server card."""

from __future__ import annotations

import json
import os
import re
import shlex
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from openmmla.utils.session_sources import last_left

# MediaMTX defaults; System Settings → Stream Server holds the ones in use
API_PORT = 9997
PLAYBACK_PORT = 9996

# a stream that stopped as the session began leaves a sliver: not worth a file
MIN_CLIP_SECONDS = 1.0


class RecordingsError(Exception):
    """the stream server could not be asked, or refused."""


@dataclass(frozen=True)
class Clip:
    path: str        # the stream's path on the server, <app>/<name>
    start: datetime  # aware, UTC
    duration: float  # seconds

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration)


def parse_time(value) -> datetime | None:
    """a session or recording time as an aware UTC datetime. MongoDB hands back
    naive datetimes that are UTC; MediaMTX writes RFC 3339 with a Z."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    # exactly six fraction digits: Go writes up to nine, and fromisoformat
    # before Python 3.11 takes three or six and nothing else
    text = re.sub(r"\.(\d+)", lambda found: "." + (found.group(1) + "000000")[:6], text, count=1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def session_end(record: dict | None, now: datetime | None = None) -> tuple[datetime, str]:
    """where a session's footage ends, and why: its end_time (`ended`); for a
    session that was never ended, the moment its last base left, once every
    base that joined it has (`left`, from the session's sources); otherwise
    now, as it is still running (`running`)."""
    ended = parse_time((record or {}).get("end_time"))
    if ended is not None:
        return ended, "ended"
    try:
        left = parse_time(last_left(record))
    except (TypeError, ValueError):  # left_at of kinds that do not compare: no end to go by
        left = None
    if left is not None:
        return left, "left"
    return now or datetime.now(timezone.utc), "running"


def _rfc3339(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _origin(host: str, port: int) -> str:
    host = str(host or "").strip() or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(port)}"


def _get_json(url: str, timeout: float):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RecordingsError(f"{url}: HTTP {error.code} {_error_text(error)}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise RecordingsError(f"{url}: {getattr(error, 'reason', error)}") from error


def _error_text(error: urllib.error.HTTPError) -> str:
    try:
        return str(json.loads(error.read().decode("utf-8")).get("error") or "").strip()
    except Exception:
        return ""


@dataclass(frozen=True)
class Recorded:
    """one path's recording on the server: when each of its segments began."""
    path: str
    segments: tuple[datetime, ...]  # aware UTC, oldest first


def inventory(host: str, api_port: int = API_PORT, timeout: float = 10.0) -> list[Recorded]:
    """every path the stream server holds a recording of, with the start of
    each segment (control API)."""
    found: dict[str, set[datetime]] = {}
    page = 0
    while True:
        data = _get_json(f"{_origin(host, api_port)}/v3/recordings/list?itemsPerPage=100&page={page}", timeout)
        for item in data.get("items") or []:
            name = str(item.get("name") or "")
            if not name:
                continue
            starts = found.setdefault(name, set())
            for segment in item.get("segments") or []:
                start = parse_time((segment or {}).get("start"))
                if start is not None:
                    starts.add(start)
        page += 1
        if page >= int(data.get("pageCount") or 0):
            break
    return [Recorded(name, tuple(sorted(found[name]))) for name in sorted(found)]


def live_paths(host: str, api_port: int = API_PORT, timeout: float = 3.0) -> set[str]:
    """the paths being published to the stream server right now (control API:
    a path is ready while its source sends)."""
    live: set[str] = set()
    page = 0
    while True:
        data = _get_json(f"{_origin(host, api_port)}/v3/paths/list?itemsPerPage=100&page={page}", timeout)
        for item in data.get("items") or []:
            if item.get("ready") and item.get("name"):
                live.add(str(item["name"]))
        page += 1
        if page >= int(data.get("pageCount") or 0):
            break
    return live


def recorded_paths(host: str, api_port: int = API_PORT, timeout: float = 10.0) -> list[str]:
    """every path the stream server holds a recording of (control API)."""
    return [recorded.path for recorded in inventory(host, api_port, timeout)]


def segments_before(recorded: list[Recorded], cutoff: datetime) -> list[tuple[str, datetime]]:
    """(path, start) of every segment that began before `cutoff`, oldest first."""
    old = [(item.path, start) for item in recorded for start in item.segments if start < cutoff]
    return sorted(old, key=lambda pair: (pair[1], pair[0]))


def delete_segment(host: str, path: str, start: datetime, api_port: int = API_PORT, timeout: float = 30.0) -> None:
    """remove one segment on the server (control API): the one that began at
    `start`, so the time has to be the one the listing gave."""
    query = urllib.parse.urlencode({"path": path, "start": _rfc3339(start)})
    request = urllib.request.Request(
        f"{_origin(host, api_port)}/v3/recordings/deletesegment?{query}", method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            pass
    except urllib.error.HTTPError as error:
        raise RecordingsError(
            f"{path} {start:%Y-%m-%d %H:%M:%S}: HTTP {error.code} {_error_text(error)}") from error
    except (urllib.error.URLError, OSError) as error:
        raise RecordingsError(f"{path}: {getattr(error, 'reason', error)}") from error


def timespans(host: str, path: str, playback_port: int = PLAYBACK_PORT,
              timeout: float = 10.0) -> list[tuple[datetime, float]]:
    """the stretches a path was recorded without a break (playback server);
    segments that follow one another count as one stretch."""
    url = f"{_origin(host, playback_port)}/list?{urllib.parse.urlencode({'path': path})}"
    try:
        data = _get_json(url, timeout)
    except RecordingsError as error:
        # a path without segments answers 400/404, which is "nothing", not a failure
        if isinstance(error.__cause__, urllib.error.HTTPError) and error.__cause__.code in (400, 404):
            return []
        raise
    spans = []
    for item in data if isinstance(data, list) else []:
        start = parse_time(item.get("start"))
        try:
            duration = float(item.get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        if start is not None and duration > 0:
            spans.append((start, duration))
    return sorted(spans)


def clips_for_window(spans_by_path: dict[str, list[tuple[datetime, float]]],
                     start: datetime, end: datetime) -> list[Clip]:
    """what was recorded between start and end: one clip per unbroken stretch
    of a path, cut to the window. A stream that was restarted gives two."""
    clips = []
    for path in sorted(spans_by_path):
        for span_start, duration in spans_by_path[path]:
            clip_start = max(start, span_start)
            clip_end = min(end, span_start + timedelta(seconds=duration))
            seconds = (clip_end - clip_start).total_seconds()
            if seconds >= MIN_CLIP_SECONDS:
                clips.append(Clip(path, clip_start, round(seconds, 3)))
    return clips


def clip_url(host: str, clip: Clip, playback_port: int = PLAYBACK_PORT) -> str:
    # the default format (fMP4) keeps the file's clock on the requested start:
    # its first frame is at the offset it was recorded at, which is what makes
    # <name>_<start>.mp4 plus the frame time a wall-clock time again
    query = urllib.parse.urlencode({"path": clip.path, "start": _rfc3339(clip.start), "duration": f"{clip.duration:.3f}"})
    return f"{_origin(host, playback_port)}/get?{query}"


def clip_relpath(clip: Clip) -> str:
    """<app>/<name>_<start>.mp4: the streams of a pipeline share a folder, and
    the name is the one the `file` source reads the start time from."""
    parts = [part for part in clip.path.split("/") if part and part not in (".", "..")]
    name = f"{parts[-1] if parts else 'stream'}_{clip.start.timestamp():.6f}.mp4"
    return os.path.join(*parts[:-1], name) if len(parts) > 1 else name


def download_clip(host: str, clip: Clip, destination: str, playback_port: int = PLAYBACK_PORT,
                  progress=None, timeout: float = 60.0) -> int:
    """write one clip to `destination`; returns its size. The file appears under
    its name only when it is complete: an error, or one raised by `progress`
    (called with the bytes written so far, which is how a download is
    stopped), leaves nothing behind."""
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    partial = destination + ".part"
    written = 0
    try:
        with urllib.request.urlopen(clip_url(host, clip, playback_port), timeout=timeout) as response, \
                open(partial, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if progress is not None:
                    progress(written)
        os.replace(partial, destination)
    except urllib.error.HTTPError as error:
        _discard(partial)
        raise RecordingsError(f"{clip.path}: HTTP {error.code} {_error_text(error)}") from error
    except (urllib.error.URLError, OSError) as error:
        _discard(partial)
        raise RecordingsError(f"{clip.path}: {getattr(error, 'reason', error)}") from error
    except BaseException:
        _discard(partial)
        raise
    return written


def _discard(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ---- how long the server keeps a recording ----

# MediaMTX deletes a segment `recordDeleteAfter` after it began. Its own
# default is a day; 0 keeps everything
DEFAULT_RETENTION = 24 * 3600.0

# what the Stream Server card offers for `recordDeleteAfter`
RETENTION_CHOICES: list[tuple[str, float]] = [
    ("1 day", 86400.0), ("3 days", 3 * 86400.0), ("7 days", 7 * 86400.0),
    ("14 days", 14 * 86400.0), ("30 days", 30 * 86400.0), ("for ever", 0.0),
]

# Go's units, plus the days MediaMTX adds: it takes `3d` in the file and
# reports whole days that way (`72h` in the file reads back as `3d`)
_DURATION_UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0,
                   "d": 86400.0}
_DURATION_PART = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(ns|us|µs|μs|ms|s|m|h|d)")


def parse_duration(value) -> float | None:
    """seconds of a duration as mediamtx.yml and the API write them: `72h`,
    `3d`, `1h30m`, `0s`, `72h0m0s`. None for anything else, empty included."""
    text = str(value or "").strip().strip("'\"")
    if not text:
        return None
    if text == "0":
        return 0.0
    position, total = 0, 0.0
    while position < len(text):
        part = _DURATION_PART.match(text, position)
        if part is None:
            return None
        total += float(part.group(1)) * _DURATION_UNITS[part.group(2)]
        position = part.end()
    return total


def format_duration(seconds: float) -> str:
    """the shortest Go duration of whole hours, minutes or seconds: 72h, 90m, 0s."""
    seconds = max(0.0, float(seconds))
    if seconds and seconds % 3600 == 0:
        return f"{int(seconds // 3600)}h"
    if seconds and seconds % 60 == 0:
        return f"{int(seconds // 60)}m"
    return f"{seconds:g}s"


def describe_retention(seconds: float | None) -> str:
    """`3 days`, `36 hours`, `for ever`; `unknown` while the server was not asked."""
    if seconds is None:
        return "unknown"
    if seconds <= 0:
        return "for ever"
    for unit, name in ((86400.0, "day"), (3600.0, "hour"), (60.0, "minute")):
        if seconds % unit == 0:
            count = int(seconds // unit)
            return f"{count} {name}{'s' if count != 1 else ''}"
    return format_duration(seconds)


def retention(host: str, api_port: int = API_PORT, timeout: float = 5.0) -> float:
    """seconds the running server keeps a segment for: `recordDeleteAfter` of
    its path defaults. 0 is for ever, which the API reports as an empty string."""
    data = _get_json(f"{_origin(host, api_port)}/v3/config/pathdefaults/get", timeout)
    seconds = parse_duration(data.get("recordDeleteAfter")) if isinstance(data, dict) else None
    return 0.0 if seconds is None else seconds


def expiry(start: datetime | None, retention_seconds: float | None) -> datetime | None:
    """when the server begins to delete a session's footage: the segment that
    holds the session's start began at or before it, and goes `retention`
    after that. None while the retention is unknown, or everything is kept."""
    if start is None or not retention_seconds or retention_seconds <= 0:
        return None
    return start + timedelta(seconds=retention_seconds)


def path_default_line(text: str, key: str) -> tuple[int, str] | None:
    """(line index, value as written) of `key:` under `pathDefaults:` of a
    mediamtx.yml, or None when it has no such line. The file is mostly
    comments, so it is edited as text: a yaml round trip would drop every
    one of them."""
    inside = False
    for index, line in enumerate(text.splitlines()):
        if re.match(r"^pathDefaults:\s*(#.*)?$", line):
            inside = True
            continue
        if inside:
            if line.strip() and not line.startswith((" ", "\t", "#")):
                break  # the next top-level key
            match = re.match(rf"^\s+{re.escape(key)}:\s*([^#]*?)\s*(#.*)?$", line)
            if match:
                return index, match.group(1).strip("'\"")
    return None


def with_path_default(text: str, key: str, value: str) -> str | None:
    """the text with that line's value replaced and its comment kept; None
    when the line is not there (the file is left alone rather than guessed at)."""
    found = path_default_line(text, key)
    if found is None:
        return None
    lines = text.splitlines()
    lines[found[0]] = re.sub(
        rf"^(\s+{re.escape(key)}:\s*)[^#]*?(\s*#.*)?$",
        lambda match: match.group(1) + value + (match.group(2) or ""), lines[found[0]], count=1)
    return "\n".join(lines) + "\n"


# ---- what the recordings take on the server's disk ----

def usage_script(quoted_root: str, paths) -> str:
    """for a shell on the server: `SIZE <kB> <path>` for every recorded path
    under the record root, `FREE <kB>` of the root's disk, USAGE when done.
    The root comes quoted for the shell ($HOME stays expandable)."""
    listed = " ".join(shlex.quote(str(path)) for path in paths)
    return (
        f"root={quoted_root}; set -- {listed}; for p in \"$@\"; do "
        'k=$(du -sk "$root/$p" 2>/dev/null | cut -f1); '
        '[ -n "$k" ] && printf "SIZE %s %s\\n" "$k" "$p"; done; '
        'printf "FREE %s\\n" "$(df -Pk "$root" 2>/dev/null | awk \'NR==2{print $4}\')"; echo USAGE'
    )


def parse_usage(text) -> tuple[dict[str, int], int | None] | None:
    """({path: bytes}, free bytes) from the script's output; None when it
    never finished (the host could not be asked)."""
    lines = [line.rstrip() for line in str(text or "").splitlines()]
    if "USAGE" not in lines:
        return None
    sizes: dict[str, int] = {}
    free = None
    for line in lines:
        kind, _, rest = line.partition(" ")
        if kind == "SIZE":
            kilobytes, _, path = rest.partition(" ")
            if kilobytes.isdigit() and path:
                sizes[path] = int(kilobytes) * 1024
        elif kind == "FREE" and rest.strip().isdigit():
            free = int(rest.strip()) * 1024
    return sizes, free


def human_size(size: int | float | None) -> str:
    """`1.2 GB`; `?` for a size nobody could tell."""
    if size is None:
        return "?"
    amount = float(size)
    for unit in ("B", "kB", "MB", "GB"):
        if amount < 1000:
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1000
    return f"{amount:.1f} TB"
