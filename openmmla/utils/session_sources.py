"""what a session used: one entry per base, kept in the session's MongoDB
document under `sources`.

A base writes its entry when it joins a session (which Bases entry it is, the
stream it takes, where that stream is captured and recorded) and notes when it
leaves. Sessions -> Export Streams reads the entries, so it takes the
session's own streams and nothing else, from the Stream Server and from the
capture hosts: a stream is
shared infrastructure, and a session is tied to it by what its bases pulled.

Writing never stops a base: MongoDB being down, or a session document the
console did not create, is a warning in the base's log and nothing more."""

from __future__ import annotations

import logging
import socket
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlsplit

from openmmla.utils.constants import STREAM_URL_SCHEMES, normalize_source, resolve_stream_source, stream_kind

logger = logging.getLogger(__name__)

# the field of the session document
SOURCES_FIELD = "sources"


def source_key(pipeline: str, base_id, stream: str | None = None) -> str:
    """one entry per base of a pipeline and the stream it takes: `ips:0@ips-cam-1`,
    `asr:1` for one that takes none. A base started again in the same session
    on another stream gets an entry of its own, so the first stream's part of
    the session is still exported."""
    key = f"{pipeline}:{base_id}"
    return f"{key}@{stream}" if stream else key


# what a Select leaves in a config field nobody filled in
_UNSET = ("", "Select.NULL", "Select.BLANK", "None", "null")


def _clean(value) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text in _UNSET else text


def _yes(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def stream_url_path(url) -> str | None:
    """the path a pulled stream has on the Stream Server: `ips/cam-1` of
    rtsp://server:8554/ips/cam-1 or rtmp://server/ips/cam-1, and of an SRT
    URL's streamid (read:ips/cam-1, or #!::r=ips/cam-1,m=request). None for
    anything else, such as raw audio pushed over udp or tcp."""
    text = str(url or "").strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in STREAM_URL_SCHEMES:
        return None
    if scheme == "srt":
        streamid = dict(parse_qsl(parts.query)).get("streamid", "")
        if not streamid and parts.fragment.startswith("!::"):
            streamid = f"#{parts.fragment}"  # an unescaped # starts the URL's fragment
        if streamid.startswith("#!::"):
            fields = dict(item.split("=", 1) for item in streamid[4:].split(",") if "=" in item)
            path = fields.get("r", "")
        else:
            path = streamid.split(":")[1] if streamid.count(":") >= 1 else ""
    else:
        path = parts.path
    return path.strip("/") or None


def _pushed_to_port(config: dict, source: str, port, host: str = "") -> tuple[str, dict, str] | None:
    """the Streams entry that pushes raw audio to this port (an ASR base on
    udp/tcp). Bases on several machines may listen on the same port: then the
    one pushed to this base's own host, and none when that cannot be told."""
    try:
        wanted = int(port)
    except (TypeError, ValueError):
        return None
    streams = (config or {}).get("Streams") or {}
    if not isinstance(streams, dict):
        return None
    found = []
    for name, entry in streams.items():
        if not isinstance(entry, dict):
            continue
        target = str(entry.get("target") or "").strip()
        try:
            parts = urlsplit(target)
            if parts.scheme.lower() == source and parts.port == wanted:
                found.append((str(name), entry, target, (parts.hostname or "").lower()))
        except ValueError:
            continue
    if len(found) == 1:
        return found[0][:3]
    here = {name for name in (host.lower(), host.lower().split(".", 1)[0]) if name}
    mine = [item for item in found if item[3] in here or item[3].split(".", 1)[0] in here]
    return mine[0][:3] if len(mine) == 1 else None


def stream_for_base(config: dict, base: dict, host: str = "") -> tuple[str, dict, str] | None:
    """(name, Streams entry, URL) of the stream a Bases entry takes; None when
    it takes none (a local camera or microphone, a file) or names none that
    is there. `host`: the machine the base runs on, for a udp/tcp base."""
    source = normalize_source((base or {}).get("source"))
    streams = (config or {}).get("Streams") or {}
    if source == "stream":
        try:
            name, url = resolve_stream_source(config or {}, (base or {}).get("source_index"))
        except ValueError:
            return None
        entry = streams.get(name) if isinstance(streams, dict) else None
        return name, entry if isinstance(entry, dict) else {}, url
    if source in ("udp", "tcp"):
        return _pushed_to_port(config, source, (base or {}).get("port"), host)
    return None


def _kind(entry: dict, pipeline: str, url: str) -> str:
    """where the Streams tab files the stream's recording: the same rule, with
    the card's default (an ASR card's streams are audio)."""
    fields = dict(entry or {})
    fields.setdefault("target", url or "")
    return stream_kind(fields, default="audio" if pipeline == "asr" else "video")


def source_entry(pipeline: str, base: dict, config: dict, *, stream: str | None = None,
                 url: str | None = None, host: str | None = None,
                 now: datetime | None = None) -> dict:
    """the entry a base writes for itself when it joins a session.

    `stream` and `url` are what the base resolved, when it resolved them
    itself (an ASR base whose source_index is empty picks its stream from a
    menu); otherwise they are looked up from its Bases entry. `capture` is
    what the console needs to find the capture-side recording again: the
    Streams entry's machine (ssh_profile), whether it records, and where."""
    base = base or {}
    host = host or socket.gethostname().split(".", 1)[0]
    found = stream_for_base(config, base, host)
    entry: dict = {}
    if stream:
        streams = (config or {}).get("Streams") or {}
        entry = streams.get(stream) if isinstance(streams, dict) and isinstance(streams.get(stream), dict) else {}
        if found and found[0] == stream:
            entry = found[1]
            url = url or found[2]
    elif found:
        stream, entry, found_url = found
        url = url or found_url
    capture = None
    if stream:
        capture = {
            "ssh_profile": _clean(entry.get("ssh_profile")),
            "record": _yes(entry.get("record")),
            "record_root": _clean(entry.get("record_root")),
            "kind": _kind(entry, pipeline, url or ""),
        }
    source_index = base.get("source_index")
    return {
        "key": source_key(pipeline, base.get("id"), stream),
        "pipeline": pipeline,
        "base_id": str(base.get("id")),
        "source": normalize_source(base.get("source")),
        "source_index": None if source_index is None else str(source_index),
        "stream": stream or None,
        "url": url or None,
        "server_path": stream_url_path(url),
        "capture": capture,
        "host": host,
        "joined_at": now or datetime.now(timezone.utc),
        "left_at": None,
    }


def record_joined(mongo, session_id: str | None, entry: dict, log=None) -> bool:
    """write a base's entry into its session; never raises."""
    log = log or logger
    if mongo is None or not session_id:
        return False
    try:
        written = bool(mongo.add_session_source(session_id, entry))
    except Exception as error:  # a base runs on without it
        log.warning("Could not note in session %s which stream base %s takes: %s", session_id, entry.get("key"), error)
        return False
    if not written:
        log.warning("Session %s is not in MongoDB: which stream base %s takes is not noted.", session_id, entry.get("key"))
    return written


def record_left(mongo, session_id: str | None, key: str, log=None) -> bool:
    """note that a base left its session; never raises."""
    log = log or logger
    if mongo is None or not session_id:
        return False
    try:
        return bool(mongo.mark_session_source_left(session_id, key))
    except Exception as error:
        log.warning("Could not note in session %s that base %s left: %s", session_id, key, error)
        return False


# ---- reading it back ----

def session_sources(record: dict | None) -> list[dict]:
    """the entries of a session document, in the order they were written."""
    sources = (record or {}).get(SOURCES_FIELD)
    return [entry for entry in sources if isinstance(entry, dict)] if isinstance(sources, list) else []


def server_paths(record: dict | None) -> list[str]:
    """the Stream Server paths the session's bases pulled, each once."""
    paths: list[str] = []
    for entry in session_sources(record):
        path = str(entry.get("server_path") or "").strip("/")
        if path and path not in paths:
            paths.append(path)
    return paths


def captured_streams(record: dict | None) -> list[dict]:
    """the streams the session's bases took that the console captures (a
    Streams entry with an ssh_profile), each once: name, ssh_profile, record,
    record_root, kind."""
    found: dict[str, dict] = {}
    for entry in session_sources(record):
        name = str(entry.get("stream") or "").strip()
        capture = entry.get("capture") if isinstance(entry.get("capture"), dict) else None
        if not name or capture is None or name in found:
            continue
        found[name] = {
            "name": name,
            "ssh_profile": str(capture.get("ssh_profile") or "").strip(),
            "record": bool(capture.get("record")),
            "record_root": str(capture.get("record_root") or "").strip(),
            "kind": str(capture.get("kind") or "video"),
        }
    return list(found.values())


def last_left(record: dict | None):
    """when the last base of the session left, once every base that joined it
    has; None while one is still in (or none ever joined). What a session that
    was never ended runs until."""
    sources = session_sources(record)
    if not sources or any(entry.get("left_at") is None for entry in sources):
        return None
    return max(entry["left_at"] for entry in sources)
