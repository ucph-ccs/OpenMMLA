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
    ASR: 'base_type'/'source_index'/'channel_select'/'port'. This list is the single source
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



def asr_segment_durations(config: dict, sp: bool = False, base_types=None) -> dict[str, float]:
    """how long the segments are that each ASR base type records: its recognize_duration, or
    recognize_sp_duration with speech separation. For `base_types`, else for the types the
    config's Bases entries use; a type without a number is left out."""
    blocks = (config or {}).get("Base")
    blocks = blocks if isinstance(blocks, dict) else {}
    if base_types is None:
        base_types = [str(entry.get("base_type")) for entry in get_bases(config)
                      if entry.get("base_type") is not None]
    key = "recognize_sp_duration" if sp else "recognize_duration"
    durations: dict[str, float] = {}
    for name in base_types:
        block = blocks.get(str(name))
        if not isinstance(block, dict) or str(name) in durations:
            continue
        try:
            durations[str(name)] = float(block.get(key))
        except (TypeError, ValueError):
            continue
    return durations


def shared_segment_duration(durations: dict) -> float | None:
    """the segment length most of these base types share (on a tie, the first one's); None for
    none."""
    values = list(durations.values())
    if not values:
        return None
    return max(values, key=lambda value: (values.count(value), -values.index(value)))

def get_base_by_id(config: dict, base_id) -> dict | None:
    """Return the base entry whose id matches, or None."""
    for base in get_bases(config):
        if str(base.get("id")) == str(base_id):
            return base
    return None


def is_main_base(base: dict) -> bool:
    return str(base.get("main")).lower() in ("true", "1", "yes")


def camera_sync_problem(config: dict) -> str:
    """why IPS camera sync cannot run on this config's 'Bases', or "".

    Sync puts a second camera in the main one's coordinates: the two bases run
    and both see the same tag, so it takes the base marked main: true (exactly
    one) and at least one other. The console checks this before it starts the
    sync, and the sync itself raises with the same words."""
    bases = get_bases(config)
    listed = ", ".join(
        f"{base.get('id')} (camera {base.get('camera') or '?'}{', main' if is_main_base(base) else ''})"
        for base in bases
    )
    mains = [base for base in bases if is_main_base(base)]
    if not bases:
        return ("No bases under 'Bases'. Camera sync needs two: the main one (main: true) and the one "
                "whose camera is synced to it.")
    if not mains:
        return f"No base is marked main: true (Bases: {listed}); mark exactly one."
    if len(mains) > 1:
        return f"More than one base is marked main: true ({', '.join(str(m.get('id')) for m in mains)}); mark exactly one."
    if len(bases) < 2:
        return (f"Only one base is defined (Bases: {listed}). Camera sync puts a second camera in the main "
                f"one's coordinates: add a base for it under 'Bases' (IPS Base, Config tab, + Add Entry) with its "
                f"camera and source and main: false, start both bases, then the sync.")
    return ""


def coerce_source_index(value, default: int = 0) -> int:
    """Return an int index for index-based sources (opencv/stream/pyaudio).

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
    # sections the pipeline pins (SystemServicesOverride) stay its own: the TUI
    # leaves them alone when it syncs, and so does the merge at startup
    pinned = config.get("SystemServicesOverride") or []
    if isinstance(pinned, str):
        pinned = [pinned]
    pinned = {str(name) for name in pinned} if isinstance(pinned, (list, tuple, set)) else set()
    for section in SYSTEM_SERVICE_SECTIONS:
        if section in pinned and isinstance(config.get(section), dict):
            continue
        section_data = system_config.get(section)
        if isinstance(section_data, dict):
            merged[section] = dict(section_data)
    return merged


def _holds_encrypted_values(node: Any) -> bool:
    """whether any leaf is an ENC(...) string, recognised without importing crypto."""
    if isinstance(node, dict):
        return any(_holds_encrypted_values(v) for v in node.values())
    if isinstance(node, list):
        return any(_holds_encrypted_values(v) for v in node)
    return isinstance(node, str) and node.startswith("ENC(") and node.endswith(")")


def decrypt_config_values(data: Any) -> Any:
    """Return a copy of a nested config structure with ENC(...) leaves
    decrypted (best-effort: values stay encrypted if the crypto module or
    master key is unavailable, and a warning is logged so a client using a
    literal 'ENC(...)' credential is diagnosable)."""
    try:
        from openmmla.utils.crypto import is_encrypted, decrypt_value
    except ImportError:
        # an env whose extra forgot cryptography reaches here, and silence would
        # send the literal ENC(...) to the service as a credential: influxdb
        # answers 401 and the caller reports "no data" rather than "bad token"
        if _holds_encrypted_values(data):
            import logging
            logging.getLogger(__name__).warning(
                "Config holds ENC(...) values but 'cryptography' is not installed in "
                "this environment, so they stay encrypted and the service they are sent "
                "to will reject them. Reinstall this environment's extra, e.g. "
                "pip install -e '.[uber-server]'."
            )
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
