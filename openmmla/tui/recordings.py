"""server-side recordings: what the stream server (MediaMTX) recorded, by time.

MediaMTX records every published path whether or not a session runs, and its
playback server returns any time range of a path as one file. A stream is
shared by the sessions that pull it, so the footage of one session is a query
and not a folder: the paths that were being recorded between the session's
start and end, each cut to that window."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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


def recorded_paths(host: str, api_port: int = API_PORT, timeout: float = 10.0) -> list[str]:
    """every path the stream server holds a recording of (control API)."""
    names: list[str] = []
    page = 0
    while True:
        data = _get_json(f"{_origin(host, api_port)}/v3/recordings/list?itemsPerPage=100&page={page}", timeout)
        names.extend(str(item.get("name") or "") for item in data.get("items") or [])
        page += 1
        if page >= int(data.get("pageCount") or 0):
            break
    return sorted(name for name in set(names) if name)


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


def matches(path: str, patterns) -> bool:
    """no pattern takes every path; `ips/*` takes the streams of one pipeline."""
    wanted = [str(p).strip().strip("/") for p in (patterns or []) if str(p).strip()]
    return not wanted or any(fnmatch.fnmatchcase(path, pattern) for pattern in wanted)


def clips_for_window(spans_by_path: dict[str, list[tuple[datetime, float]]],
                     start: datetime, end: datetime, patterns=()) -> list[Clip]:
    """what was recorded between start and end: one clip per unbroken stretch
    of a path, cut to the window. A stream that was restarted gives two."""
    clips = []
    for path in sorted(spans_by_path):
        if not matches(path, patterns):
            continue
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
    its name only when it is complete."""
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
    return written


def _discard(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
