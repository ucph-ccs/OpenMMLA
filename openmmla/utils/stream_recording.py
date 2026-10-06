"""the Stream Server records a session's streams only while the session runs.

mediamtx.yml records no path by default (`record: no` under pathDefaults).
START, sent from Session Control in the TUI or with `mmla ses-ctl`, switches
recording on for the paths the session's bases pull through the Stream Server
of System Settings (the `server_path` of each of its `sources` whose URL names
that server, openmmla.utils.session_sources; a stream pulled from another
server or straight from a camera is no path of this one), each through a named
path entry of the server's control API that holds `record: true`; STOP, and a
session ended any other way (Sessions -> End Session, the Collection card's
Stop), switches them off and removes the entries, so each path falls back to
`all_others`.

Every path is its own entry and nothing touches pathDefaults: two sessions in
two rooms at once never switch each other's paths, and a path that two running
sessions share stays on until the last of them stops. A named entry that
mediamtx.yml itself holds (one with settings of its own) has its `record`
switched and is never removed. A path that a regular-expression entry of
mediamtx.yml covers is given the path defaults while it records.

The session's MongoDB document keeps when recording was on: under
`recording_windows`, one {paths, start, end} per START, its end set at STOP. A
second START extends the open window with any path it adds.

The control API changes the running server only. A server restarted during a
session records nothing more of it until the next START, and so does one that
reloads its mediamtx.yml: MediaMTX reads the file again whenever it changes (a
Save or a Sync of the Stream Server card's Config tab on a native run, a git
pull) and drops every entry the API added. reapply_open_windows switches the
paths of the running sessions on again; the Config tab calls it after a Save
or a Sync. Nothing here raises: a server or a MongoDB that cannot be reached
becomes a warning in the summary, and START and STOP go out regardless."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from openmmla.utils.constants import MONGODB_DEFAULT_DB
from openmmla.utils.session_sources import server_paths, session_sources, stream_url_path

logger = logging.getLogger(__name__)

# seconds to wait on the control API, and on MongoDB to answer
API_TIMEOUT = 3.0

# the field of the session document
WINDOWS_FIELD = "recording_windows"

# the keys an entry added here differs from the path defaults in; an entry of
# mediamtx.yml differs in more
_OWN_KEYS = ("name", "record")


class _Unreachable(Exception):
    """the control API did not answer at all."""


class _Refused(Exception):
    """the control API answered with an error."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status} {message}".strip())
        self.status = status


# ---- the control API ----

def _quote(name: str) -> str:
    return urllib.parse.quote(str(name).strip("/"), safe="/~")


def _call(api_base: str, method: str, route: str, body: dict | None = None,
          timeout: float = API_TIMEOUT) -> dict:
    """the JSON the API answers, {} for none. Raises _Refused for an HTTP error
    and _Unreachable when the server cannot be reached."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}{route}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        try:
            message = str(json.loads(error.read().decode("utf-8")).get("error") or "")
        except Exception:
            message = ""
        raise _Refused(error.code, message) from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise _Unreachable(str(getattr(error, "reason", error))) from error
    try:
        parsed = json.loads(text) if text.strip() else {}
    except ValueError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def path_defaults(api_base: str, timeout: float = API_TIMEOUT) -> dict:
    """the pathDefaults the running server holds."""
    return _call(api_base, "GET", "/v3/config/pathdefaults/get", timeout=timeout)


def record_on(api_base: str, path: str, timeout: float = API_TIMEOUT) -> str:
    """switch recording on for one path: `added` (a named entry of its own),
    or `patched` (an entry that was there already). The publisher stays."""
    try:
        _call(api_base, "POST", f"/v3/config/paths/add/{_quote(path)}", {"record": True}, timeout)
        return "added"
    except _Refused as error:
        if error.status != 400 or "exists" not in str(error):
            raise
    _call(api_base, "PATCH", f"/v3/config/paths/patch/{_quote(path)}", {"record": True}, timeout)
    return "patched"


def _is_own_entry(entry: dict, defaults: dict) -> bool:
    """whether a named entry is the path defaults plus a record switch, the
    entry record_on adds; one mediamtx.yml holds differs in more."""
    strip = lambda conf: {key: value for key, value in conf.items() if key not in _OWN_KEYS}
    return bool(defaults) and strip(entry) == strip(defaults)


def record_off(api_base: str, path: str, defaults: dict, timeout: float = API_TIMEOUT) -> str:
    """switch recording off for one path: `removed` (the entry record_on added
    is gone, the path falls back to all_others), `patched` (an entry of
    mediamtx.yml, record off) or `absent` (no entry of its own)."""
    route = _quote(path)
    try:
        entry = _call(api_base, "GET", f"/v3/config/paths/get/{route}", timeout=timeout)
    except _Refused as error:
        if error.status == 404:
            return "absent"
        raise
    _call(api_base, "PATCH", f"/v3/config/paths/patch/{route}", {"record": False}, timeout)
    if not _is_own_entry(entry, defaults):
        return "patched"
    try:
        _call(api_base, "DELETE", f"/v3/config/paths/delete/{route}", timeout=timeout)
    except _Refused as error:
        if error.status != 404:
            raise
    return "removed"


# ---- where the server and the database are ----

def _settings(config: dict | None, start_path) -> dict:
    """System Settings as one dict: the store (StreamServer stays there only)
    under the pipeline config merged with it (MongoDB as the bases see it)."""
    if config is not None:
        return config if isinstance(config, dict) else {}
    from openmmla.utils.config import load_config_with_system_services, load_system_services_config

    settings: dict = {}
    try:
        settings.update(load_system_services_config(start_path) or {})
    except Exception as error:
        logger.debug("System Settings unreadable: %s", error)
    if start_path:
        try:
            settings.update(load_config_with_system_services(start_path) or {})
        except Exception as error:
            logger.debug("config %s unreadable: %s", start_path, error)
    return settings


def stream_api_base(config: dict | None = None, start_path=None) -> str | None:
    """http://<host>:<api_port> of System Settings → Stream Server (MediaMTX),
    None while that form has no host."""
    from openmmla.tui.system_services import stream_server_section, usable_system_service_value

    section = stream_server_section(_settings(config, start_path))
    host = str(section.get("host") or "").strip()
    if not usable_system_service_value(host):
        return None
    try:
        port = int(section.get("api_port") or 9997)
    except (TypeError, ValueError):
        port = 9997
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _open_sessions(settings: dict, timeout: float):
    """(client, sessions collection) of the MongoDB of System Settings, with
    short timeouts: a database that does not answer costs seconds, not thirty."""
    from openmmla.utils.config import holds_placeholder

    section = settings.get("MongoDB") if isinstance(settings.get("MongoDB"), dict) else {}
    url = str(section.get("url") or "").strip()
    if not url or holds_placeholder(url):
        raise RuntimeError("System Settings → Connections → MongoDB has no url yet")
    from pymongo import MongoClient

    millis = int(timeout * 1000)
    client = MongoClient(url, serverSelectionTimeoutMS=millis, connectTimeoutMS=millis,
                         socketTimeoutMS=millis * 2)
    return client, client[section.get("db") or MONGODB_DEFAULT_DB]["sessions"]


# ---- the session ----

def session_server_paths(record: dict | None, server: dict | None
                         ) -> tuple[list[str], list[tuple[str, str]], dict[str, str]]:
    """the session's paths on the Stream Server `server` ({host, ...} of System
    Settings), the (stream, URL) its bases took from another server, and each
    path's kind (audio | video: the first kind its sources noted; else audio for
    a path only ASR sources took, video for any other, as the dashboard's
    mongo_devices decides it). A source counts when its
    URL names that server (by name, alias or address), or names localhost from
    a base that ran on the server's host; one with a server path and no URL is
    taken at its word. Blocking: a host name may be resolved."""
    from openmmla.tui.system_services import hosts_match, is_loopback_host, stream_server_path

    server_host = str((server or {}).get("host") or "").strip().strip("[]")
    paths: list[str] = []
    elsewhere: list[tuple[str, str]] = []
    kinds: dict[str, str] = {}
    # a kind guessed for a source that noted none gives way to one another source noted
    guessed: dict[str, str] = {}
    for entry in session_sources(record):
        url = str(entry.get("url") or "").strip()
        served = str(entry.get("server_path") or "").strip("/") or (stream_url_path(url) if url else None)
        if not served:
            continue  # udp or tcp straight to a base, or a device of its own
        path = stream_server_path(url, server) if url and server_host else None
        if (path is None and url and is_loopback_host(urllib.parse.urlsplit(url).hostname)
                and hosts_match(entry.get("host"), server_host)):
            path = served  # localhost in the URL of a base that runs on the Stream Server's host
        if path is None and not url:
            path = served
        if path is None:
            name = str(entry.get("stream") or entry.get("key") or "?")
            if (name, url) not in elsewhere:
                elsewhere.append((name, url))
            continue
        capture = entry.get("capture") if isinstance(entry.get("capture"), dict) else {}
        kind = capture.get("kind")
        if kind in ("audio", "video"):
            kinds.setdefault(path, kind)
        else:
            # a source noted without its capture's kind: an ASR base pulls sound, any other video,
            # and a path a camera's base pulls too is video (a camera's stream may carry its microphone)
            guess = "audio" if entry.get("pipeline") == "asr" or path.split("/", 1)[0] == "asr" else "video"
            if guessed.get(path) != "video":
                guessed[path] = guess
        if path not in paths:
            paths.append(path)
    for path, kind in guessed.items():
        kinds.setdefault(path, kind)
    return paths, elsewhere, kinds


def _open_window(record: dict | None) -> dict | None:
    windows = (record or {}).get(WINDOWS_FIELD)
    for window in reversed(windows if isinstance(windows, list) else []):
        if isinstance(window, dict) and window.get("end") is None:
            return window
    return None


def _shared_with(sessions, session_id: str, path: str) -> str | None:
    """another session that has not ended and records this path now; None
    also when MongoDB cannot say, as the session that stops has ended."""
    try:
        other = sessions.find_one(
            {"session_id": {"$ne": session_id}, "status": {"$ne": "ended"},
             WINDOWS_FIELD: {"$elemMatch": {"end": None, "paths": path}}},
            {"_id": 0, "session_id": 1})
    except Exception as error:
        logger.debug("could not ask MongoDB who else records %s: %s", path, error)
        return None
    return str(other.get("session_id")) if other else None


def _note_window(sessions, session_id: str, on: bool, paths: list[str], now: datetime) -> str | None:
    """open, extend or close the session's recording window."""
    open_one = {"session_id": session_id, WINDOWS_FIELD: {"$elemMatch": {"end": None}}}
    if not on:
        result = sessions.update_one(open_one, {"$set": {f"{WINDOWS_FIELD}.$.end": now}})
        return "closed" if result.matched_count else None
    if not paths:
        return None
    result = sessions.update_one(open_one, {"$addToSet": {f"{WINDOWS_FIELD}.$.paths": {"$each": paths}}})
    if result.matched_count:
        return "extended"
    result = sessions.update_one(
        {"session_id": session_id},
        {"$push": {WINDOWS_FIELD: {"paths": list(paths), "start": now, "end": None}}})
    return "opened" if result.matched_count else None


def set_session_recording(session_id: str, on: bool, *, config: dict | None = None, mongo_db=None,
                          api_base: str | None = None, paths: list[str] | None = None, start_path=None,
                          timeout: float = API_TIMEOUT, now: datetime | None = None) -> dict:
    """switch the Stream Server's recording of a session's streams on (START)
    or off (STOP). Never raises.

    `config`: System Settings as a dict (StreamServer, MongoDB); read from the
    store found from `start_path` when None. `mongo_db`: anything with a
    `sessions` collection (MongoDBClientWrapper, a pymongo Database); opened
    from System Settings when None, unless `paths` is given, which then stands
    for the session's paths and leaves MongoDB alone.

    Returns {session_id, on, api, paths, done, kept, elsewhere, window,
    warnings, text}: `done` the paths switched, `kept` the paths left on
    because another running session records them, `elsewhere` the streams the
    bases took from another server, `text` one line for the caller's output."""
    now = now or datetime.now(timezone.utc)
    summary: dict = {"session_id": session_id, "on": bool(on), "api": None, "paths": [], "done": [],
                     "kept": [], "elsewhere": [], "window": None, "warnings": [], "text": ""}
    warnings: list[str] = summary["warnings"]
    notes: list[str] = []
    client = None
    try:
        settings = _settings(config, start_path)
        sessions = getattr(mongo_db, "sessions", None) if mongo_db is not None else None
        record = None
        if sessions is None and paths is None and mongo_db is not None:
            warnings.append("the MongoDB client has no sessions collection: the session's streams are unknown")
        elif sessions is None and paths is None:
            try:
                client, sessions = _open_sessions(settings, timeout)
            except Exception as error:
                warnings.append(f"MongoDB not reached ({error}): the session's streams are unknown")
        if sessions is not None:
            try:
                record = sessions.find_one({"session_id": session_id},
                                           {"_id": 0, "sources": 1, WINDOWS_FIELD: 1})
                if record is None and paths is None:
                    warnings.append(f"session {session_id} is not in MongoDB: its streams are unknown")
            except Exception as error:
                warnings.append(f"MongoDB not reached ({error}): the session's streams are unknown")
                sessions = None

        if paths is not None:
            wanted = list(paths)
        else:
            from openmmla.tui.system_services import stream_server_section, usable_system_service_value

            server = stream_server_section(settings)
            if usable_system_service_value(str(server.get("host") or "").strip()):
                # only the paths of this server: a camera pulled straight, or another
                # room's server, may carry a path of the same name
                wanted, elsewhere, _kinds = session_server_paths(record, server)
                summary["elsewhere"] = [name for name, _url in elsewhere]
            else:
                wanted = server_paths(record)  # no server to tell them apart: its missing host is the warning
        if not on:
            window = _open_window(record)
            for path in (window or {}).get("paths") or []:
                if isinstance(path, str) and path.strip("/") and path.strip("/") not in wanted:
                    wanted.append(path.strip("/"))
        summary["paths"] = wanted

        base = api_base or stream_api_base(settings)
        summary["api"] = base
        if wanted and not base:
            warnings.append("System Settings → Connections → Stream Server (MediaMTX) has no host yet")
        elif wanted:
            try:
                defaults = path_defaults(base, timeout)
            except (_Unreachable, _Refused) as error:
                defaults = None
                warnings.append(f"Stream Server API {base} not reached ({error})")
            if defaults is not None:
                if defaults.get("record"):
                    notes.append("the server records every path regardless: it runs with record: yes under "
                                 "pathDefaults until it is restarted with the current mediamtx.yml")
                for path in wanted:
                    if not on and sessions is not None and _shared_with(sessions, session_id, path):
                        summary["kept"].append(path)
                        continue
                    try:
                        if on:
                            record_on(base, path, timeout)
                        else:
                            record_off(base, path, defaults, timeout)
                        summary["done"].append(path)
                    except _Refused as error:
                        warnings.append(f"{path}: {error}")
                    except _Unreachable as error:
                        warnings.append(f"Stream Server API {base} not reached ({error})")
                        break

        if sessions is not None and record is not None:
            try:
                summary["window"] = _note_window(sessions, session_id, on, summary["done"], now)
            except Exception as error:
                warnings.append(f"recording window not noted in MongoDB ({error})")
    except Exception as error:  # never stop START or STOP
        logger.debug("stream recording of %s failed", session_id, exc_info=True)
        warnings.append(f"stream recording not switched: {error}")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    summary["text"] = _describe(summary, notes)
    logger.debug("stream recording of %s: %s", session_id, summary["text"])
    return summary


def end_session_recording(session_id: str, *, config: dict | None = None, mongo_db=None, start_path=None,
                          now: datetime | None = None, timeout: float = API_TIMEOUT) -> dict:
    """STOP's switch-off for a session that ended any other way (Sessions ->
    End Session, the Collection card's Stop): without it the session's paths
    would record until the server restarts, and no later STOP of the ended
    session would switch them off. Its open window closes at `now` (the end
    the session was given). As set_session_recording, with an empty text when
    the session had nothing on the server and nothing went wrong."""
    summary = set_session_recording(session_id, False, config=config, mongo_db=mongo_db, start_path=start_path,
                                    timeout=timeout, now=now)
    if not summary["paths"] and not summary["warnings"]:
        summary["text"] = ""
    return summary


def reapply_open_windows(*, config: dict | None = None, mongo_db=None, api_base: str | None = None,
                         start_path=None, timeout: float = API_TIMEOUT) -> dict:
    """switch recording on again for the paths of every running session's open
    recording window: what a server that read its mediamtx.yml again (or was
    restarted) has dropped. A path that still records is left as it is. Never
    raises; `config`, `mongo_db` and `start_path` as for set_session_recording.

    Returns {sessions, paths, done, warnings, text}: `sessions` the running
    sessions with an open window, `done` the paths switched on, `text` one
    line, empty when no running session records anything."""
    summary: dict = {"sessions": [], "paths": [], "done": [], "warnings": [], "text": ""}
    warnings: list[str] = summary["warnings"]
    client = None
    try:
        settings = _settings(config, start_path)
        sessions = getattr(mongo_db, "sessions", None) if mongo_db is not None else None
        if sessions is None and mongo_db is not None:
            warnings.append("the MongoDB client has no sessions collection: the running sessions are unknown")
        elif sessions is None:
            try:
                client, sessions = _open_sessions(settings, timeout)
            except Exception as error:
                warnings.append(f"MongoDB not reached ({error}): the running sessions are unknown")
        if sessions is not None:
            try:
                found = list(sessions.find(
                    {"status": {"$ne": "ended"}, WINDOWS_FIELD: {"$elemMatch": {"end": None}}},
                    {"_id": 0, "session_id": 1, WINDOWS_FIELD: 1}))
            except Exception as error:
                warnings.append(f"MongoDB not reached ({error}): the running sessions are unknown")
                found = []
            for record in found:
                window = _open_window(record)
                if window is None:
                    continue
                summary["sessions"].append(str(record.get("session_id") or "?"))
                for path in window.get("paths") or []:
                    path = str(path).strip("/") if isinstance(path, str) else ""
                    if path and path not in summary["paths"]:
                        summary["paths"].append(path)
        base = api_base or stream_api_base(settings)
        if summary["paths"] and not base:
            warnings.append("System Settings → Connections → Stream Server (MediaMTX) has no host yet")
        elif summary["paths"]:
            for path in summary["paths"]:
                try:
                    record_on(base, path, timeout)
                    summary["done"].append(path)
                except _Refused as error:
                    warnings.append(f"{path}: {error}")
                except _Unreachable as error:
                    warnings.append(f"Stream Server API {base} not reached ({error})")
                    break
    except Exception as error:  # never stop what called it
        logger.debug("re-applying the stream recording failed", exc_info=True)
        warnings.append(f"stream recording not switched on again: {error}")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    done, count = summary["done"], len(summary["sessions"])
    if done:
        summary["text"] = (f"Stream Server records {', '.join(done)} again, for {count} running "
                           f"session{'s' if count != 1 else ''}")
    elif summary["paths"]:
        summary["text"] = "Stream Server recording of the running sessions not switched on again"
    elif warnings:
        summary["text"] = "Stream Server recording of the running sessions not checked"
    if warnings:
        summary["text"] += f"; {'; '.join(warnings)}"
    return summary


def _describe(summary: dict, notes: list[str]) -> str:
    """one line: what was switched, then what went wrong."""
    done, kept, warned = summary["done"], summary["kept"], bool(summary["warnings"])
    if summary["on"]:
        if done:
            line = f"Stream Server records {', '.join(done)} until STOP"
        elif not summary["paths"] and not warned and not summary.get("elsewhere"):
            line = "Stream Server records nothing: no base of this session has noted a stream yet"
        else:
            line = "Stream Server records nothing of this session"
    elif done:
        line = f"Stream Server stopped recording {', '.join(done)}"
    elif not summary["paths"] and not warned:
        line = "Stream Server: no stream noted for this session"
    elif kept and not warned:
        line = "Stream Server recording left on"
    else:
        line = "Stream Server recording not switched off"
    if kept:
        line += f"; {', '.join(kept)} still recorded for another running session"
    if summary.get("elsewhere") and summary["on"]:
        line += f"; {', '.join(summary['elsewhere'])} taken from another server, not recorded here"
    if notes:
        line += f" ({'; '.join(notes)})"
    if summary["warnings"]:
        line += f"; {'; '.join(summary['warnings'])}"
    return line
