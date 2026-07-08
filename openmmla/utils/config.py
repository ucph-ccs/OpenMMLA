from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


SYSTEM_SERVICES_REL_PATH = Path("config") / "system_services.yml"
SYSTEM_SERVICE_SECTIONS = ("InfluxDB", "MongoDB", "MQTT", "Redis")


def get_bases(config: dict) -> list[dict]:
    """Return the configured base nodes from the 'Bases' config section.

    Shared by ASR and IPS: each base is a dict with an 'id' (base identity;
    numeric for ASR so udp/tcp ports can be derived, string for IPS).
    Modality-specific keys differ — IPS: 'camera'/'source_index'/'main';
    ASR: 'base_type'/'source_index'/'channel'. This list is the single source
    of truth for base identity and per-device bindings.
    """
    bases = (config or {}).get("Bases") or []
    result = []
    for b in bases:
        if not isinstance(b, dict):
            continue
        bid = b.get("id")
        if bid is None:
            continue
        # ignore the shipped template entry (id left as a <...> placeholder)
        if isinstance(bid, str) and bid.strip().startswith("<") and bid.strip().endswith(">"):
            continue
        result.append(b)
    return result


def get_base_by_id(config: dict, base_id) -> dict | None:
    """Return the base entry whose id matches, or None."""
    for base in get_bases(config):
        if str(base.get("id")) == str(base_id):
            return base
    return None


def coerce_source_index(value, default: int = 0) -> int:
    """Return an int index for index-based sources (opencv/rtmp/pyaudio).

    source_index is overloaded: a numeric index for index-based sources, a file
    name for 'file' sources, and a stream name for 'lsl'. This coerces only the
    numeric case, falling back to `default` for names/None.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def select_source_by_index_or_name(value, available: list):
    """Pick an entry from `available` by integer index or by name.

    Used by video bases when resolving source_index against the discovered
    sources: an integer selects by position; a non-numeric value (e.g. a file
    name for 'file' sources) matches by basename (or exact string).
    """
    if not available:
        return None
    try:
        idx = int(value)
        return available[idx] if 0 <= idx < len(available) else available[0]
    except (TypeError, ValueError):
        pass
    target = str(value)
    for src in available:
        if os.path.basename(str(src)) == target or str(src) == target:
            return src
    return available[0]


def compute_initial_sync_time(file_dir: str, configured=None, exts=None) -> float:
    """Resolve the file-replay sync reference time.

    Returns the explicit ``configured`` value when set; otherwise auto-computes
    the latest (max) file start time parsed from ``_<timestamp>.<ext>`` file
    names in ``file_dir`` — the common point at which every file has begun.
    """
    import re

    if configured is not None and str(configured).strip() != "":
        return float(configured)
    starts: list[float] = []
    try:
        names = os.listdir(file_dir)
    except OSError:
        names = []
    for name in names:
        if exts and not name.lower().endswith(tuple(exts)):
            continue
        match = re.search(r"_(\d+(?:\.\d+)?)\.", name)
        if match:
            starts.append(float(match.group(1)))
    if not starts:
        raise ValueError(
            f"Could not auto-compute initial_sync_time: no timestamped files "
            f"(<name>_<unixtime>.<ext>) found in {file_dir}. Name files with a "
            f"trailing _<timestamp> or set initial_sync_time explicitly.")
    return max(starts)


def load_yaml_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def find_system_services_config(
    start_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    explicit = os.environ.get("OPENMMLA_SYSTEM_SERVICES_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    if start_path is None:
        current = Path.cwd().resolve()
    else:
        current = Path(start_path).expanduser().resolve()
        if current.is_file():
            current = current.parent

    for parent in (current, *current.parents):
        candidate = parent / SYSTEM_SERVICES_REL_PATH
        if candidate.is_file():
            return candidate
    return None


def load_system_services_config(
    start_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    config_path = find_system_services_config(start_path)
    if config_path is None:
        return {}
    return load_yaml_config(config_path)


def merge_system_services(
    config: dict[str, Any],
    start_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    merged = dict(config)
    system_config = load_system_services_config(start_path)
    for section in SYSTEM_SERVICE_SECTIONS:
        section_data = system_config.get(section)
        if isinstance(section_data, dict):
            merged[section] = dict(section_data)
    return merged


def decrypt_config_values(data: Any) -> Any:
    """Return a copy of a nested config structure with ENC(...) leaves
    decrypted (best-effort: values stay encrypted if the crypto module or
    master key is unavailable, and a warning is logged so a client using a
    literal 'ENC(...)' credential is diagnosable)."""
    try:
        from openmmla.utils.crypto import is_encrypted, decrypt_value
    except ImportError:
        return data

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str) and is_encrypted(node):
            try:
                return decrypt_value(node)
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "Could not decrypt an ENC(...) config value — wrong or missing "
                    "master key (~/.openmmla/master.key)? The encrypted form will be "
                    "used as-is and will be rejected by the target service."
                )
                return node
        return node

    return walk(data)


def load_config_with_system_services(
    config_path: str | os.PathLike[str],
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    merged = merge_system_services(config, config_path)
    # client wrappers consume this directly: credentials must be plaintext here
    return decrypt_config_values(merged)
