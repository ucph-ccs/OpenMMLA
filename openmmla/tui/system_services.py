from __future__ import annotations

import socket
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
# NOT auto-synced from the central System Services store and stay editable in
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


def config_to_flat_values(config: dict[str, Any], *, include_defaults: bool = True) -> dict[str, object]:
    values = get_shared_defaults() if include_defaults else {}
    for section_name, info in SHARED_SECTIONS.items():
        section = config.get(section_name)
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
    values = get_shared_defaults()
    seen: set[str] = set()
    for rel_path in SYSTEM_SERVICE_SOURCE_CONFIG_RELS:
        config_path = root_path / rel_path
        if not config_path.is_file():
            continue
        config = load_existing_config(str(config_path))
        for section_name, info in SHARED_SECTIONS.items():
            section = config.get(section_name)
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
    """Return the decrypted local sudo password from the System Services store,
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
}

# hosts that mean "this machine" rather than one specific address
LOOPBACK_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1", "0.0.0.0"})


def is_loopback_host(host: object) -> bool:
    return str(host or "").strip().strip("[]").lower() in LOOPBACK_HOSTS


def is_this_machine(host: object) -> bool:
    """loopback, or this machine's own hostname (bare, .local, or fully qualified)."""
    if is_loopback_host(host):
        return True
    name = str(host or "").strip().lower().rstrip(".")
    try:
        me = socket.gethostname().lower().rstrip(".")
    except OSError:
        return False
    short = me.split(".")[0]
    return name in {me, short, f"{short}.local"}


def _resolved_addresses(host: str) -> set[str]:
    try:
        return {info[4][0] for info in socket.getaddrinfo(host, None)}
    except (OSError, ValueError):
        return set()


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

    Read from System Services, which is what every pipeline config is synced
    from. The port falls back to the conventional default; the host is "" when
    unset, and a loopback host means "wherever the service runs" rather than a
    specific machine (see is_loopback_host)."""
    default = SYSTEM_SERVICE_DEFAULT_PORTS.get(target)
    if default is None:
        return None
    try:
        config = load_system_services_config(root) or {}
    except Exception:
        config = {}

    def section(name: str) -> dict:
        value = config.get(name)
        return value if isinstance(value, dict) else {}

    if target == "influxdb":
        host, port = _url_host_port(section("InfluxDB").get("url"))
    elif target == "mongodb":
        host, port = _url_host_port(section("MongoDB").get("url"))
    elif target == "redis":
        host, port = section("Redis").get("host"), section("Redis").get("port")
    elif target == "mosquitto":
        host, port = section("MQTT").get("host"), section("MQTT").get("port")
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


def system_service_reachable(
    root: str | os.PathLike[str],
    target: str,
    remote_loopback_check: Callable[[int], bool] | None = None,
) -> bool:
    """whether the configured endpoint of an Uber system service answers.

    Probed from this machine at the configured host:port, i.e. the path the
    pipelines actually take. A loopback host names no machine in particular, so
    it is checked on the selected host instead: through remote_loopback_check
    (an ssh probe of that host's own port) when given, else locally."""
    endpoint = system_service_endpoint(root, target)
    if endpoint is None:
        return False
    host, port = endpoint
    if is_loopback_host(host):
        if remote_loopback_check is not None:
            return bool(remote_loopback_check(port))
        return check_endpoint("127.0.0.1", port)
    return check_endpoint(host, port)
