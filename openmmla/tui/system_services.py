from __future__ import annotations

import re
import socket
import time
from typing import Callable
from urllib.parse import urlsplit

import os
from pathlib import Path
from typing import Any

import yaml

from openmmla.tui.schema.definitions import SHARED_SECTIONS, get_shared_defaults
from openmmla.tui.schema.loader import load_existing_config


SYSTEM_SERVICES_REL_PATH = os.path.join("config", "system_services.yml")

SYSTEM_SERVICE_SOURCE_CONFIG_RELS = (
    os.path.join("pipelines", "asr-base", "config.yml"),
    os.path.join("pipelines", "vfa-base", "config.yml"),
    os.path.join("pipelines", "ips-base", "config.yml"),
    os.path.join("pipelines", "uber-server", "dashboard", "flask-backend", "config.yml"),
)


def system_services_config_path(root: str | os.PathLike[str]) -> str:
    return str(Path(root) / SYSTEM_SERVICES_REL_PATH)


def usable_system_service_value(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text and "<" not in text and ">" not in text)


# Top-level list key in a pipeline config naming shared sections that this
# pipeline manages itself (an explicit escape hatch). Sections listed here are
# NOT auto-synced from the central System Settings store and stay editable in
# the pipeline's own Config tab.
SYSTEM_SERVICES_OVERRIDE_KEY = "SystemServicesOverride"


def pipeline_section_overrides(config: dict | None) -> set[str]:
    """Return the set of shared-section names a pipeline config pins locally.

    Only names that are actually shared sections are honored; anything else in
    the override list is ignored.
    """
    if not isinstance(config, dict):
        return set()
    raw = config.get(SYSTEM_SERVICES_OVERRIDE_KEY) or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {str(name) for name in raw if str(name) in SHARED_SECTIONS}


def shared_section_drift(
    central_sections: dict[str, dict],
    target_config: dict | None,
    *,
    overrides: set[str] | None = None,
) -> list[str]:
    """Return the shared-section names whose value in ``target_config`` differs
    from the central store, skipping overridden sections.

    A section counts as drifted when the pipeline declares it (or the central
    store has a value for it) and the two do not match. Sections the pipeline
    does not use at all are ignored.

    Args:
        central_sections: mapping of section name -> canonical section dict.
        target_config: the pipeline config dict to compare against.
        overrides: shared-section names the pipeline pins locally (skipped).
    """
    target_config = target_config if isinstance(target_config, dict) else {}
    overrides = overrides or set()
    drifted: list[str] = []
    for section_name, central_data in central_sections.items():
        if section_name in overrides:
            continue
        if section_name not in target_config:
            # pipeline does not carry this section; nothing to reconcile.
            continue
        if target_config.get(section_name) != central_data:
            drifted.append(section_name)
    return drifted


def _section_from_flat_values(section_name: str, values: dict[str, object]) -> dict[str, object]:
    info = SHARED_SECTIONS.get(section_name, {})
    section_data: dict[str, object] = {}
    for key, field in info.get("fields", {}).items():
        path = f"{section_name}.{key}"
        value = values.get(path, field.get("default", ""))
        if value is not None:
            section_data[key] = value
    return section_data


def flat_values_to_config(values: dict[str, object]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for section_name in SHARED_SECTIONS:
        section_data = _section_from_flat_values(section_name, values)
        if section_data:
            config[section_name] = section_data
    return config


def stream_server_section(config: dict[str, Any]) -> dict[str, Any]:
    """the StreamServer section of a settings store. A store written before
    MediaMTX had a section of its own keeps its host and ports under Gateway
    (one machine for both): that is what it meant, so that is what it yields."""
    section = config.get("StreamServer") if isinstance(config, dict) else None
    if isinstance(section, dict) and section:
        return section
    gateway = config.get("Gateway") if isinstance(config, dict) else None
    if not isinstance(gateway, dict):
        return {}
    return {key: gateway[key] for key in ("host", "rtmp_port", "rtsp_port") if gateway.get(key) is not None}


# what MediaMTX accepts as a path name, one or more segments
_STREAM_PATH_RE = re.compile(r"^[A-Za-z0-9_~-][A-Za-z0-9._~-]*(?:/[A-Za-z0-9._~-]+)*$")


def is_stream_path(value: object) -> bool:
    """a Streams target in its short form: the <app>/<name> path on the stream
    server, without scheme or host. A first segment with a dot is taken for a
    host that lost its scheme (ericli.local/ips/cam-1), not for a path."""
    text = str(value or "").strip().strip("/")
    if not text or "://" in text or not _STREAM_PATH_RE.match(text):
        return False
    return "." not in text.split("/", 1)[0]


def _stream_server_port(value: object, default: int) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return default
    return port if 0 < port < 65536 else default


def stream_server_urls(section: dict[str, Any], path: str) -> tuple[str, str]:
    """(publish URL, pull URL) of a path on the stream server: capture devices
    publish it over RTMP and the bases pull the same path over RTSP."""
    host = str((section or {}).get("host") or "").strip() or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    rtmp_port = _stream_server_port((section or {}).get("rtmp_port"), 1935)
    rtsp_port = _stream_server_port((section or {}).get("rtsp_port"), 8554)
    path = str(path).strip().strip("/")
    return f"rtmp://{host}:{rtmp_port}/{path}", f"rtsp://{host}:{rtsp_port}/{path}"


def complete_stream_urls(values: dict[str, object], stream_server: dict[str, Any]) -> list[str]:
    """expand, in place, the short form in the Streams values of a pipeline form
    (flat paths, Streams.<name>.target): a target that is a bare path is
    published to the stream server of System Settings and, unless read_target
    says otherwise, pulled from it; a bare read_target is completed the same way.
    Full URLs are left as they are, and so is an empty read_target next to a full
    target (it means: pull the target). Returns the streams it completed."""
    completed: list[str] = []
    for key in list(values):
        if not (key.startswith("Streams.") and key.endswith(".target")):
            continue
        name = key[len("Streams."):-len(".target")]
        read_key = f"Streams.{name}.read_target"
        target = str(values.get(key) or "").strip()
        read_target = str(values.get(read_key) or "").strip()
        changed = False
        if is_stream_path(target):
            publish, pull = stream_server_urls(stream_server, target)
            values[key] = publish
            if not read_target:
                values[read_key] = pull
            changed = True
        if is_stream_path(read_target):
            values[read_key] = stream_server_urls(stream_server, read_target)[1]
            changed = True
        if changed:
            completed.append(name)
    return completed


_STREAM_SERVER_URL_PORTS = {"rtmp": ("rtmp_port", 1935), "rtsp": ("rtsp_port", 8554)}


def repoint_stream_url(url: object, old: dict[str, Any], new: dict[str, Any]) -> str:
    """the same stream on the stream server's new address, when the URL named
    the old one (host, and the port of its scheme); any other URL comes back
    unchanged: a stream that was pointed somewhere else on purpose stays there."""
    text = str(url or "").strip()
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return text
    if parts.scheme not in _STREAM_SERVER_URL_PORTS or not parts.hostname:
        return text
    key, default_port = _STREAM_SERVER_URL_PORTS[parts.scheme]
    old_host = str((old or {}).get("host") or "").strip().strip("[]").lower()
    same_host = parts.hostname.lower() == old_host or (
        is_loopback_host(parts.hostname) and is_loopback_host(old_host))
    if not old_host or not same_host:
        return text
    if (port or default_port) != _stream_server_port((old or {}).get(key), default_port):
        return text
    publish, pull = stream_server_urls(new, "")
    base = (publish if parts.scheme == "rtmp" else pull).rstrip("/")
    userinfo = f"{parts.netloc.rsplit('@', 1)[0]}@" if "@" in parts.netloc else ""
    scheme, address = base.split("://", 1)
    rest = parts.path + (f"?{parts.query}" if parts.query else "")
    return f"{scheme}://{userinfo}{address}{rest}"


def config_to_flat_values(config: dict[str, Any], *, include_defaults: bool = True) -> dict[str, object]:
    values = get_shared_defaults() if include_defaults else {}
    for section_name, info in SHARED_SECTIONS.items():
        section = stream_server_section(config) if section_name == "StreamServer" else config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, field in info.get("fields", {}).items():
            value = section.get(key)
            if value is None and include_defaults:
                value = field.get("default", "")
            if value is not None:
                values[f"{section_name}.{key}"] = value
    return values


def load_system_services_config(root: str | os.PathLike[str]) -> dict[str, Any]:
    return load_existing_config(system_services_config_path(root))


def harvest_system_services_from_pipeline_configs(root: str | os.PathLike[str]) -> dict[str, object]:
    root_path = Path(root)
    configs = []
    for rel_path in SYSTEM_SERVICE_SOURCE_CONFIG_RELS:
        config_path = root_path / rel_path
        if config_path.is_file():
            configs.append(load_existing_config(str(config_path)))
    return harvest_system_services_from_configs(configs)


def harvest_system_services_from_configs(configs: list[dict]) -> dict[str, object]:
    """flat shared values taken from pipeline configs, first usable value wins;
    the configs may have been read on this machine or on another one."""
    values = get_shared_defaults()
    seen: set[str] = set()
    for config in configs:
        if not isinstance(config, dict):
            continue
        for section_name, info in SHARED_SECTIONS.items():
            section = stream_server_section(config) if section_name == "StreamServer" else config.get(section_name)
            if not isinstance(section, dict):
                continue
            for key in info.get("fields", {}):
                path = f"{section_name}.{key}"
                if path in seen:
                    continue
                value = section.get(key)
                if usable_system_service_value(value):
                    values[path] = value
                    seen.add(path)
    return values


def load_system_service_values(root: str | os.PathLike[str]) -> dict[str, object]:
    config = load_system_services_config(root)
    if config:
        return config_to_flat_values(config)
    return harvest_system_services_from_pipeline_configs(root)


def save_system_services_config(root: str | os.PathLike[str], values: dict[str, object]) -> str:
    config_path = system_services_config_path(root)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config = flat_values_to_config(values)
    try:
        from openmmla.utils.crypto import encrypt_sensitive_values, ensure_master_key
        encrypt_sensitive_values(config, ensure_master_key())
    except Exception:
        pass  # crypto unavailable: fall back to writing values as-is
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return config_path


def get_sudo_password(root: str | os.PathLike[str]) -> str | None:
    """Return the decrypted local sudo password from the System Settings store,
    or None if unset/unusable."""
    config = load_system_services_config(root)
    section = config.get("Sudo") if isinstance(config, dict) else None
    value = section.get("password") if isinstance(section, dict) else None
    if not usable_system_service_value(value):
        return None
    value = str(value)
    try:
        from openmmla.utils.crypto import is_encrypted, decrypt_value
        if is_encrypted(value):
            value = decrypt_value(value)
    except Exception:
        return None
    return value or None


def save_system_service_section(
    root: str | os.PathLike[str],
    section_name: str,
    section_data: dict[str, object],
) -> str:
    config = load_system_services_config(root)
    if not isinstance(config, dict):
        config = {}
    config[section_name] = dict(section_data)
    values = config_to_flat_values(config)
    return save_system_services_config(root, values)


# ── system service endpoints ──────────────────────────────────────

# conventional ports of the Uber system services, keyed by make target
SYSTEM_SERVICE_DEFAULT_PORTS: dict[str, int] = {
    "influxdb": 8086,
    "mongodb": 27017,
    "redis": 6379,
    "mosquitto": 1883,
    "nginx": 8080,
    "mediamtx": 1935,
}

# what a system service is called wherever the console shows it (its card, the
# sidebar, the Status tab, the log), keyed by make target: "Role (Product)",
# where the role is the System Settings → Connections section that holds its
# address. The Connections forms are labelled the same way in
# schema/definitions.py, so a card and the form with its address share a name
# and nobody has to know that Mosquitto is the MQTT broker
SYSTEM_SERVICE_LABELS: dict[str, str] = {
    "influxdb": "InfluxDB",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "mosquitto": "MQTT (Mosquitto)",
    "nginx": "Gateway (Nginx)",
    "mediamtx": "Stream Server (MediaMTX)",
    "flask": "Dashboard (Flask)",
    "celery": "Dashboard (Celery)",
}

# the dashboard backend is a make target too, but not a port-probed system
# service; its port only matters for the Dashboard section and `make flask`
DASHBOARD_DEFAULT_PORT = 5050

# hosts that mean "this machine" rather than one specific address
LOOPBACK_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1", "0.0.0.0"})


def is_loopback_host(host: object) -> bool:
    return str(host or "").strip().strip("[]").lower() in LOOPBACK_HOSTS


def is_this_machine(host: object) -> bool:
    """loopback, this machine's own hostname (bare, .local, or fully qualified),
    or any name or address that resolves to one of its interfaces (a LAN or
    tailnet IP written into a url is still this machine). May resolve names:
    call it off the UI thread."""
    if is_loopback_host(host):
        return True
    name = str(host or "").strip().strip("[]").lower().rstrip(".")
    try:
        me = socket.gethostname().lower().rstrip(".")
    except OSError:
        me = ""
    short = me.split(".")[0]
    if me and name in {me, short, f"{short}.local"}:
        return True
    return any(_is_local_address(address) for address in _resolved_addresses(name))


def _is_local_address(address: str) -> bool:
    """whether an IP address belongs to one of this machine's interfaces: only
    a local address can be bound."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((address, 0))
        return True
    except (OSError, ValueError):
        return False


# host -> (expiry, addresses): bindings and markers are recomputed on every
# status refresh, and a dead .local name takes seconds to fail each time
_RESOLVE_TTL_SEC = 30.0
_RESOLVED_ADDRESSES: dict[str, tuple[float, set[str]]] = {}


def _resolved_addresses(host: str) -> set[str]:
    cached = _RESOLVED_ADDRESSES.get(host)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except (OSError, ValueError):
        addresses = set()
    _RESOLVED_ADDRESSES[host] = (time.monotonic() + _RESOLVE_TTL_SEC, addresses)
    return addresses


def hosts_match(a: object, b: object) -> bool:
    """whether two host spellings name the same machine: equal names, equal
    short names, or at least one shared resolved address (so a tailnet IP in a
    url still matches an SSH profile that uses the hostname)."""
    x = str(a or "").strip().lower().rstrip(".")
    y = str(b or "").strip().lower().rstrip(".")
    if not x or not y:
        return False
    if x == y or x.split(".")[0] == y.split(".")[0]:
        return True
    return bool(_resolved_addresses(x) & _resolved_addresses(y))


def target_for_service_host(host: object, profiles: list) -> str:
    """console target that reaches the machine `host` names: "local" for this
    machine, the name of the saved SSH profile that points at it, or "" when
    the console has no way onto that machine."""
    host = str(host or "").strip()
    if not host:
        return ""
    if is_this_machine(host):
        return "local"
    for profile in profiles:
        if hosts_match(host, profile.host) or hosts_match(host, profile.name):
            return profile.name
    return ""


def _url_host_port(url: object) -> tuple[str, int | None]:
    """hostname and port of a URL; port is None when absent or unparseable."""
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return "", None
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        # e.g. a mongodb replica-set list "h1:27017,h2:27017"
        port = None
    return host, port


def system_service_endpoint(root: str | os.PathLike[str], target: str) -> tuple[str, int] | None:
    """(host, port) that clients of an Uber system service are configured with.

    Read from System Settings, which is what every pipeline config is synced
    from. The port falls back to the conventional default; the host is "" when
    unset, and a loopback host means "wherever the service runs" rather than a
    specific machine (see is_loopback_host)."""
    default = DASHBOARD_DEFAULT_PORT if target == "flask" else SYSTEM_SERVICE_DEFAULT_PORTS.get(target)
    if default is None:
        return None
    try:
        config = load_system_services_config(root) or {}
    except Exception:
        config = {}

    def section(name: str) -> dict:
        value = config.get(name)
        return value if isinstance(value, dict) else {}

    if target == "flask":
        host, port = section("Dashboard").get("host"), section("Dashboard").get("port")
    elif target == "influxdb":
        host, port = _url_host_port(section("InfluxDB").get("url"))
    elif target == "mongodb":
        host, port = _url_host_port(section("MongoDB").get("url"))
    elif target == "redis":
        host, port = section("Redis").get("host"), section("Redis").get("port")
    elif target == "mosquitto":
        host, port = section("MQTT").get("host"), section("MQTT").get("port")
    elif target == "mediamtx":
        stream_server = stream_server_section(config)
        host, port = stream_server.get("host"), stream_server.get("rtmp_port")
    else:
        host, port = section("Gateway").get("host"), section("Gateway").get("http_port")
    host = str(host or "").strip()
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = default
    if not 0 < port < 65536:
        port = default
    return host, port


def check_endpoint(host: str, port: int, timeout: float = 1.5) -> bool:
    """TCP-connect probe from this machine; resolves names the way clients do."""
    try:
        with socket.create_connection((host or "127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


MEDIAMTX_DEFAULT_RTSP_PORT = 8554


def system_service_probe_ports(root: str | os.PathLike[str], target: str) -> list[int]:
    """every port that has to answer before the service counts as running.

    One port for most services. MediaMTX needs its RTSP port next to RTMP: an
    nginx built with the RTMP module (the gateway before MediaMTX) answers on
    1935 as well, and would read as a running MediaMTX that Stop never finds."""
    endpoint = system_service_endpoint(root, target)
    if endpoint is None:
        return []
    ports = [endpoint[1]]
    if target == "mediamtx":
        try:
            stream_server = stream_server_section(load_system_services_config(root) or {})
        except Exception:
            stream_server = {}
        rtsp = stream_server.get("rtsp_port")
        try:
            rtsp = int(rtsp)
        except (TypeError, ValueError):
            rtsp = MEDIAMTX_DEFAULT_RTSP_PORT
        if not 0 < rtsp < 65536:
            rtsp = MEDIAMTX_DEFAULT_RTSP_PORT
        if rtsp not in ports:
            ports.append(rtsp)
    return ports


def system_service_port_states(
    root: str | os.PathLike[str],
    target: str,
    remote_loopback_check: Callable[[int], bool] | None = None,
    own_port: bool = False,
) -> dict[int, bool]:
    """which of the service's probe ports answer at its configured endpoint.

    Probed from this machine at the configured host:port, i.e. the path the
    pipelines actually take. A loopback host names no machine in particular, so
    it is checked on the selected host instead: through remote_loopback_check
    (an ssh probe of that host's own port) when given, else locally. own_port
    asks for that same check of the selected host whatever the address says:
    a card moved off the configured machine reports the host it is on."""
    endpoint = system_service_endpoint(root, target)
    if endpoint is None:
        return {}
    host = endpoint[0]
    states: dict[int, bool] = {}
    for port in system_service_probe_ports(root, target):
        if not own_port and not is_loopback_host(host):
            states[port] = check_endpoint(host, port)
        elif remote_loopback_check is not None:
            states[port] = bool(remote_loopback_check(port))
        else:
            states[port] = check_endpoint("127.0.0.1", port)
    return states


def system_service_reachable(
    root: str | os.PathLike[str],
    target: str,
    remote_loopback_check: Callable[[int], bool] | None = None,
    own_port: bool = False,
) -> bool:
    """whether the configured endpoint of an Uber system service answers, on
    every port that service is made of (see system_service_port_states)."""
    states = system_service_port_states(root, target, remote_loopback_check, own_port)
    return bool(states) and all(states.values())


def system_service_port_conflict(
    root: str | os.PathLike[str],
    target: str,
    remote_loopback_check: Callable[[int], bool] | None = None,
    own_port: bool = False,
) -> str:
    """explain a service whose ports answer only in part, or "" when they all
    agree. Half-open means another program holds one of its ports."""
    if len(system_service_probe_ports(root, target)) < 2:
        return ""
    states = system_service_port_states(root, target, remote_loopback_check, own_port)
    if all(states.values()) or not any(states.values()):
        return ""
    up = ", ".join(str(port) for port, ok in states.items() if ok)
    down = ", ".join(str(port) for port, ok in states.items() if not ok)
    message = f"port {up} answers but {down} does not, so another program holds {up}"
    if target == "mediamtx":
        message += (
            ": usually an nginx built with the RTMP module, left over from before MediaMTX. "
            f"Press Start on the {SYSTEM_SERVICE_LABELS['nginx']} card to render its config again (the current template "
            "has no rtmp block), or stop Nginx"
        )
    return message
