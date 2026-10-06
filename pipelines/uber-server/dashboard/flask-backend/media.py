"""What MediaMTX can show of a session: its live camera streams and its recorded footage.

The browser plays live video straight from MediaMTX over WebRTC (WHEP) and recorded clips from the
playback server, but it cannot ask the MediaMTX API itself (the API sends no CORS headers), so the
backend asks for it: which paths are publishing now, whether WebRTC and playback are switched on,
and which stretches of a path were recorded. Every probe has a short timeout and a small cache, and
an unreachable MediaMTX only empties the answer and says why: the dashboard works without it.
"""

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

logger = logging.getLogger("dashboard.media")

DEFAULT_PORTS = {"api_port": 9997, "playback_port": 9996, "webrtc_port": 8889}
TIMEOUT = 2.0
PATHS_TTL = 5.0
GLOBAL_TTL = 60.0
ADDRESS_TTL = 60.0
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "127.0.1.1"}
# a microphone's path read under this prefix is its sound as Opus (the listen/ entry of mediamtx.yml)
LISTEN_PREFIX = "listen/"

_FRACTION_RE = re.compile(r"^(.*T\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:?\d\d)?$")


def parse_rfc3339(value) -> float | None:
    """epoch seconds of an RFC 3339 time with any number of fraction digits (MediaMTX writes
    nanoseconds or trims trailing zeros, which datetime.fromisoformat on 3.10 rejects)."""
    if not isinstance(value, str):
        return None
    match = _FRACTION_RE.match(value.strip())
    if not match:
        return None
    base, fraction, zone = match.groups()
    try:
        moment = datetime.fromisoformat(base)
    except ValueError:
        return None
    if zone and zone != "Z":
        sign = 1 if zone[0] == "+" else -1
        digits = zone[1:].replace(":", "")
        offset = sign * (int(digits[:2]) * 3600 + int(digits[2:4]) * 60)
    else:
        offset = 0
    seconds = moment.replace(tzinfo=timezone.utc).timestamp() - offset
    if fraction:
        seconds += float("0." + fraction[:9])
    return seconds


def rfc3339(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _usable(value) -> bool:
    text = str(value or "").strip()
    return bool(text) and "<" not in text and ">" not in text


def stream_server_address(repo_root: str, config: dict | None) -> dict | None:
    """{host, api_port, playback_port, webrtc_port, source} of the stream server.

    System Settings → Stream Server on this machine first (the file the database sections come
    from: OPENMMLA_SYSTEM_SERVICES_CONFIG when set, else config/system_services.yml, read with the
    TUI's defaults), then a `StreamServer` section of the dashboard config, then the InfluxDB host
    with the MediaMTX default ports, since the infrastructure services usually share one machine."""
    try:
        from openmmla.tui.schema.definitions import SHARED_SECTIONS
        from openmmla.tui.schema.loader import load_existing_config
        from openmmla.tui.system_services import config_to_flat_values
        from openmmla.utils.config import find_system_services_config
        path = find_system_services_config(repo_root)
        store = load_existing_config(str(path)) if path is not None else {}
        store = store if isinstance(store, dict) else {}
        # the TUI's defaults fill unset fields (host localhost), so only trust them when System
        # Settings actually holds a Stream Server
        saved = store.get("StreamServer")
        found = {}
        if isinstance(saved, dict) and _usable(saved.get("host")):
            values = config_to_flat_values(store)
            found = {key: values.get(f"StreamServer.{key}") for key in SHARED_SECTIONS["StreamServer"]["fields"]}
        if _usable(found.get("host")):
            return {"host": str(found["host"]).strip(),
                    "api_port": _int(found.get("api_port"), DEFAULT_PORTS["api_port"]),
                    "playback_port": _int(found.get("playback_port"), DEFAULT_PORTS["playback_port"]),
                    "webrtc_port": _int(found.get("webrtc_port"), DEFAULT_PORTS["webrtc_port"]),
                    "source": "system_services"}
    except Exception as exc:
        logger.info("System Settings stream server not readable: %s", exc)
    section = (config or {}).get("StreamServer")
    if isinstance(section, dict) and _usable(section.get("host")):
        return {"host": str(section["host"]).strip(),
                "api_port": _int(section.get("api_port"), DEFAULT_PORTS["api_port"]),
                "playback_port": _int(section.get("playback_port"), DEFAULT_PORTS["playback_port"]),
                "webrtc_port": _int(section.get("webrtc_port"), DEFAULT_PORTS["webrtc_port"]),
                "source": "config"}
    influx = (config or {}).get("InfluxDB")
    url = influx.get("url") if isinstance(influx, dict) else None
    if _usable(url):
        host = urlsplit(str(url)).hostname
        if host:
            return dict(DEFAULT_PORTS, host=host, source="influxdb")
    return None


def _host_for_url(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _get_json(url: str, timeout: float = TIMEOUT):
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8") or "null")


class MediaServer:
    """MediaMTX probes for the dashboard, each cached briefly."""

    def __init__(self, repo_root: str, config: dict | None):
        self.repo_root = repo_root
        self.config = config or {}
        self._lock = threading.Lock()
        self._address = None
        self._address_at = 0.0
        self._paths = (None, None)
        self._paths_at = 0.0
        self._global = None
        self._global_at = 0.0
        self._listen = None
        self._listen_at = 0.0

    def address(self) -> dict | None:
        """re-read every minute, so a Stream Server saved in the TUI reaches a running dashboard."""
        now = time.time()
        if self._address_at and now - self._address_at < ADDRESS_TTL:
            return self._address
        address = stream_server_address(self.repo_root, self.config)
        self._address, self._address_at = address, now
        return address

    def _api(self, path: str):
        address = self.address()
        if address is None:
            raise URLError("no stream server is configured")
        return _get_json(f"http://{_host_for_url(address['host'])}:{address['api_port']}{path}")

    def ready_paths(self) -> tuple[set | None, str | None]:
        """(names of the paths publishing now, None) or (None, reason) when the API did not answer."""
        now = time.time()
        with self._lock:
            if self._paths_at and now - self._paths_at < PATHS_TTL:
                return self._paths
        result = (None, None)
        address = self.address()
        if address is None:
            result = (None, "No stream server is configured.")
        else:
            try:
                ready, page, pages = set(), 0, 1
                while page < pages and page < 20:
                    body = self._api(f"/v3/paths/list?itemsPerPage=200&page={page}") or {}
                    for item in body.get("items") or []:
                        if isinstance(item, dict) and item.get("ready") and item.get("name"):
                            ready.add(str(item["name"]))
                    pages = _int(body.get("pageCount"), 1)
                    page += 1
                result = (ready, None)
            except (URLError, HTTPError, OSError, ValueError) as exc:
                logger.info("MediaMTX API did not answer: %s", exc)
                result = (None, f"MediaMTX did not answer at {address['host']}:{address['api_port']}.")
        with self._lock:
            self._paths, self._paths_at = result, now
        return result

    def global_config(self) -> dict | None:
        """MediaMTX's global configuration (webrtc and playback switches), cached a minute."""
        now = time.time()
        if self._global_at and now - self._global_at < GLOBAL_TTL:
            return self._global
        try:
            value = self._api("/v3/config/global/get")
            value = value if isinstance(value, dict) else None
        except (URLError, HTTPError, OSError, ValueError):
            value = None
        self._global, self._global_at = value, now
        return value

    def listen_entry(self) -> bool | None:
        """whether the running server holds the listen/ entry of mediamtx.yml (a regular expression
        on LISTEN_PREFIX with a runOnDemand), so a microphone can be heard in a browser; None when
        the API did not answer. Cached a minute."""
        now = time.time()
        if self._listen_at and now - self._listen_at < GLOBAL_TTL:
            return self._listen
        value = None
        try:
            found, page, pages = False, 0, 1
            while page < pages and page < 20 and not found:
                body = self._api(f"/v3/config/paths/list?itemsPerPage=200&page={page}") or {}
                for item in body.get("items") or []:
                    name = str((item or {}).get("name") or "") if isinstance(item, dict) else ""
                    if name.startswith(f"~^{LISTEN_PREFIX}") and str(item.get("runOnDemand") or "").strip():
                        found = True
                        break
                pages = _int(body.get("pageCount"), 1)
                page += 1
            value = found
        except (URLError, HTTPError, OSError, ValueError):
            value = None
        self._listen, self._listen_at = value, now
        return value

    def health(self) -> dict:
        address = self.address()
        if address is None:
            return {"ok": None, "host": None}
        ready, reason = self.ready_paths()
        out = {"ok": ready is not None, "host": address["host"]}
        if ready is not None:
            settings = self.global_config() or {}
            out["paths"] = len(ready)
            if "webrtc" in settings:
                out["webrtc"] = bool(settings.get("webrtc"))
        else:
            out["error"] = reason
        return out

    def recordings(self, path: str, start: float, end: float) -> list[list[float]]:
        """[[start_epoch, duration_s], ...] recorded on `path` within [start, end], clipped to it."""
        address = self.address()
        if address is None or end <= start:
            return []
        query = urlencode({"path": path, "start": rfc3339(start), "end": rfc3339(end)})
        url = f"http://{_host_for_url(address['host'])}:{address['playback_port']}/list?{query}"
        try:
            body = _get_json(url)
        except HTTPError as exc:
            if exc.code in (400, 404):
                return []
            logger.info("MediaMTX playback answered %s for %s", exc.code, path)
            return []
        except (URLError, OSError, ValueError) as exc:
            logger.info("MediaMTX playback did not answer: %s", exc)
            return []
        spans = []
        for item in body if isinstance(body, list) else []:
            if not isinstance(item, dict):
                continue
            begin = parse_rfc3339(item.get("start"))
            try:
                duration = float(item.get("duration"))
            except (TypeError, ValueError):
                continue
            if begin is None:
                continue
            a, b = max(begin, start), min(begin + duration, end)
            if b > a:
                spans.append([round(a, 3), round(b - a, 3)])
        return spans

    def browser_base(self, port_key: str, request_host: str | None) -> str | None:
        """the base URL a browser uses for MediaMTX: a loopback stream-server host means MediaMTX
        runs beside the dashboard, so the browser reaches it by the name it reached the dashboard."""
        address = self.address()
        if address is None:
            return None
        host = address["host"]
        if host in LOCAL_HOSTS and request_host:
            host = request_host
        return f"http://{_host_for_url(host)}:{address[port_key]}"

    def session_media(self, doc: dict | None, devices: dict | None, mongo_ok: bool, t0, t1,
                      want_recordings: bool, request_host: str | None) -> dict:
        """the /media answer of section 5.5. Each microphone (kind audio) also names `listen`, the
        path its sound plays from in a browser (LISTEN_PREFIX + its path), while the server has WebRTC
        and the listen/ entry; `listen_reason` says why none has one otherwise."""
        address = self.address()
        out = {"webrtc": None, "playback": None, "streams": [], "recordings": [], "reason": None,
               "listen_reason": None}
        if address is None:
            out["reason"] = "No stream server is configured, so the dashboard cannot show video."
            return out
        settings = self.global_config()
        webrtc_on = settings is None or bool(settings.get("webrtc"))
        playback_on = settings is None or bool(settings.get("playback"))
        out["webrtc"] = self.browser_base("webrtc_port", request_host) if webrtc_on else None
        out["playback"] = self.browser_base("playback_port", request_host) if playback_on else None
        if doc is None:
            out["reason"] = ("MongoDB is not reachable, so this session's streams are unknown."
                             if not mongo_ok else
                             "This session has no MongoDB record, so its streams are unknown.")
            return out
        streams = [s for s in ((devices or {}).get("streams") or []) if isinstance(s, dict) and s.get("path")]
        if not streams:
            kinds = {str((source or {}).get("source") or "") for source in (doc.get("sources") or [])
                     if isinstance(source, dict)}
            if kinds and kinds <= {"file"}:
                out["reason"] = "This session read its sources from files, so it has no live streams."
            elif kinds:
                out["reason"] = "This session's sources did not go through the stream server."
            else:
                out["reason"] = "This session lists no sources."
            return out
        ready, api_reason = self.ready_paths()
        kinds = [stream.get("kind") or ("audio" if stream.get("pipeline") == "asr" else "video") for stream in streams]
        listen = None
        if "audio" in kinds:
            listen = self.listen_entry() if webrtc_on else False
            if not webrtc_on:
                out["listen_reason"] = "MediaMTX has WebRTC turned off, so the microphones cannot play in the browser."
            elif listen is None:
                out["listen_reason"] = "MediaMTX did not say whether it can turn the microphones into sound for the browser."
            elif not listen:
                out["listen_reason"] = ("The stream server has no listen/ entry in its mediamtx.yml, so the microphones "
                                        "cannot play in the browser (see the dashboard docs).")
        for stream, kind in zip(streams, kinds):
            path = str(stream.get("path"))
            out["streams"].append({
                "path": path, "pipeline": stream.get("pipeline"), "base_id": stream.get("base_id"),
                "camera": stream.get("base_id") if kind == "video" else None, "kind": kind,
                # the stream's name (the camera's tile key) and how its capture turned the picture
                "stream": stream.get("stream") or None, "rotate": stream.get("rotate") or 0,
                "ready": bool(ready is not None and path in ready),
                # where a browser hears a microphone: its sound as Opus, made when first read
                "listen": f"{LISTEN_PREFIX}{path.strip('/')}" if kind == "audio" and listen else None,
            })
        if ready is None:
            out["reason"] = api_reason
        elif not any(s["ready"] for s in out["streams"]):
            out["reason"] = "None of this session's streams is publishing now."
        elif not webrtc_on:
            out["reason"] = ("MediaMTX has WebRTC turned off, so live video cannot play in the browser "
                             "(see the dashboard docs).")
        if want_recordings and t0 is not None and t1 is not None:
            for stream in out["streams"]:
                spans = self.recordings(stream["path"], float(t0), float(t1))
                if spans:
                    out["recordings"].append({"path": stream["path"], "spans": spans})
        return out
