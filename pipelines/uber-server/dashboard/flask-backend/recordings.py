"""A session's raw capture recordings on the dashboard's machine, for the Analysis page's downloads.

The recorders keep each file as artifacts/<session>/collection/<host>/<audio|video>/<id>.<format>,
and artifacts/<session>/manifest.json lists them with their device, start, length and, for audio,
scope and wearer. A file is found by the manifest's `path` when that path exists inside the session
folder (the path was written on the machine that recorded or imported the session, often another
one), else by its place in collection/; a session without a manifest is read from collection/ alone.

Nothing else of the session folder is ever listed or served: not the speaker profiles (voice
biometrics of children), the coding clips under labels/, raw/, analysis/, pipelines/ or the
manifests themselves, and nothing a symlink or a manifest path leads to outside the session folder.
Every file is checked on its real path (os.path.realpath), whichever way it was found.

Standard library only.
"""

import json
import os
import re
import wave

MEDIA_TYPES = {
    "wav": "audio/wav", "flac": "audio/flac", "m4a": "audio/mp4", "mp3": "audio/mpeg",
    "mp4": "video/mp4", "mov": "video/quicktime", "mkv": "video/x-matroska", "webm": "video/webm",
}
MODALITIES = ("video", "audio")
AUDIO_SCOPES = ("group", "personal")
REC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
# the capture recorders' file names: <modality>_<host>_<device>_<start epoch>.<ext>
RECORDING_NAME_RE = re.compile(r"^(?P<modality>audio|video)_(?P<host>.+?)_(?P<device>[^_]+)_(?P<start>\d+(?:\.\d+)?)$")
# folders of a session that hold more than capture recordings: a path through one of them is
# refused even inside collection/ (a host folder that happens to carry such a name included)
PRIVATE_PARTS = frozenset({"profiles", "labels", "raw", "analysis", "pipelines"})
MANIFEST = "manifest.json"


def repo_root() -> str:
    """the repository this backend belongs to, four levels above this folder (as media.py's caller
    finds it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def artifacts_root() -> str:
    """DASHBOARD_ARTIFACTS_DIR when set, else <repo>/artifacts."""
    configured = (os.environ.get("DASHBOARD_ARTIFACTS_DIR") or "").strip()
    return os.path.abspath(os.path.expanduser(configured)) if configured else os.path.join(repo_root(), "artifacts")


def media_type(path: str) -> str:
    return MEDIA_TYPES.get(_extension(path), "application/octet-stream")


def download_name(sid: str, rec_id: str, path: str) -> str:
    """<session>_<recording>.<ext>, with anything but letters, digits, `_`, `.` and `-` replaced."""
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{sid}_{rec_id}")
    return f"{stem}.{_extension(path) or 'bin'}"


def list_recordings(root: str, sid: str) -> list[dict]:
    """the session's recordings that exist on this machine, video first, then audio, by device:
    {id, modality, host, device, format, size, duration, start, scope, participant, channels,
    sample_rate, label}; [] for a session folder that does not exist."""
    records = [_record(path, host, modality, rec_id, row) for rec_id, modality, host, path, row in _found(root, sid)]
    records.sort(key=lambda r: (MODALITIES.index(r["modality"]), (r["device"] or "").lower(),
                                (r["host"] or "").lower(), r["start"] or 0.0, r["id"]))
    return records


def resolve(root: str, sid: str, rec_id: str) -> str | None:
    """the real path of one recording list_recordings lists, None for any other id."""
    if not isinstance(rec_id, str) or not REC_ID_RE.fullmatch(rec_id):
        return None
    return next((path for found_id, _, _, path, _ in _found(root, sid) if found_id == rec_id), None)


def label(record: dict) -> str:
    """the line a person reads: Camera c920-05 on raspi5-01, Group mic jabra-0, Worn mic vimo-0,
    Tag 0, Worn mic badge-0, Microphone mic-9."""
    device, host = record.get("device"), record.get("host")
    if record.get("modality") == "video":
        if device and host:
            return f"Camera {device} on {host}"
        return f"Camera {device}" if device else f"Camera on {host}" if host else "Camera"
    name = device or (f"on {host}" if host else None)
    scope = record.get("scope")
    if scope == "group":
        return f"Group mic {name}" if name else "Group mic"
    if scope == "personal":
        text = f"Worn mic {name}" if name else "Worn mic"
        participant = record.get("participant")
        return f"{text}, Tag {participant}" if participant is not None else text
    return f"Microphone {name}" if name else "Microphone"


def _extension(path: str) -> str:
    _, dot, ext = os.path.basename(str(path)).rpartition(".")
    return ext.lower() if dot else ""


def _session_dir(root: str, sid: str) -> str | None:
    """the real path of artifacts/<sid>/, None when sid is no plain folder name or it is missing.
    The folder itself may be a link (a session moved to another disk); what is inside it may not
    lead out of it."""
    if not isinstance(sid, str) or not REC_ID_RE.fullmatch(sid):
        return None
    path = os.path.join(root, sid)
    try:
        return os.path.realpath(path) if os.path.isdir(path) else None
    except (OSError, ValueError):
        return None


def _capture_file(base: str, candidate: str) -> tuple[str, str, str, str] | None:
    """(real path, host folder, modality, id) when `candidate` is a regular capture recording of
    the session at base, i.e. its real path is base/collection/<host>/<audio|video>/<id>.<ext>
    with a media extension; None for anything else."""
    try:
        real = os.path.realpath(candidate)
        if os.path.commonpath([base, real]) != base:
            return None
        parts = os.path.relpath(real, base).split(os.sep)
    except (OSError, ValueError):
        return None
    if len(parts) != 4 or parts[0] != "collection" or parts[2] not in MODALITIES:
        return None
    if any(part.lower() in PRIVATE_PARTS or part.startswith(".") for part in parts):
        return None
    stem, dot, ext = parts[3].rpartition(".")
    if not dot or ext.lower() not in MEDIA_TYPES or not REC_ID_RE.fullmatch(stem):
        return None
    try:
        if not os.path.isfile(real):
            return None
    except (OSError, ValueError):
        return None
    return real, parts[1], parts[2], stem


def _read_manifest(base: str) -> list | None:
    """the manifest's recordings list, None when there is no readable manifest with one."""
    path = os.path.join(base, MANIFEST)
    try:
        real = os.path.realpath(path)
        if os.path.commonpath([base, real]) != base or not os.path.isfile(real):
            return None
        with open(real, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    rows = data.get("recordings") if isinstance(data, dict) else None
    return rows if isinstance(rows, list) else None


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _count(value) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number > 0 and number == int(number) else None


def _text(value) -> str | None:
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    text = str(value).strip()
    return text or None


def _wav_header(path: str) -> tuple[int, int, float | None] | None:
    """(channels, sample rate, seconds) of a PCM wav from its header, None for anything else."""
    try:
        with wave.open(path, "rb") as handle:
            channels, rate, frames = handle.getnchannels(), handle.getframerate(), handle.getnframes()
    except (OSError, EOFError, wave.Error, ValueError):
        return None
    return channels, rate, (round(frames / rate, 3) if rate > 0 else None)


def _record(path: str, host: str | None, modality: str, rec_id: str, row: dict) -> dict:
    """the listing entry of one file in the host folder `host`, from its manifest row (empty for a
    scanned file) and, where the row says nothing, from the recorder's file name."""
    named = RECORDING_NAME_RE.fullmatch(rec_id)
    named = named.groupdict() if named and named.group("modality") == modality else {}
    device = _text(row.get("device")) or named.get("device")
    start = _number(row.get("start_time"))
    if start is None and named.get("start"):
        start = _number(named["start"])
    duration = _number(row.get("duration"))
    header = _wav_header(path) if _extension(path) == "wav" else None
    if not duration or duration < 0:
        duration = header[2] if header else None
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    record = {
        "id": rec_id, "modality": modality, "host": host or _text(row.get("host")) or named.get("host"),
        "device": device, "format": _extension(path), "size": size,
        "duration": round(duration, 3) if duration else None,
        "start": round(start, 3) if start is not None else None,
        "scope": None, "participant": None, "channels": None, "sample_rate": None,
    }
    if modality == "audio":
        scope = row.get("scope")
        record["scope"] = scope if scope in AUDIO_SCOPES else None
        record["participant"] = _text(row.get("participant"))
        record["channels"] = _count(row.get("channels")) or (header[0] if header else None)
        record["sample_rate"] = _count(row.get("sample_rate")) or (header[1] if header else None)
    record["label"] = label(record)
    return record


def _lexically_inside(path: str, folders: tuple[str, ...]) -> bool:
    """whether a path names something inside one of the folders, by its spelling alone (a path
    written on another machine, /home/server-01/... say, may sit on an automounted folder here
    that takes a tenth of a second to ask)."""
    path = os.path.normpath(path)
    try:
        return any(os.path.commonpath([folder, path]) == folder for folder in folders)
    except ValueError:
        return False


def _from_manifest(base: str, rows: list, folders: tuple[str, ...]) -> list[tuple[str, str, str, str, dict]]:
    """the manifest rows' files; a row's `path` is only followed when it is spelled inside the
    session folder (one of `folders`), else its file is looked up in collection/."""
    found = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rec_id, modality = row.get("id"), row.get("modality")
        if not isinstance(rec_id, str) or not REC_ID_RE.fullmatch(rec_id) or modality not in MODALITIES:
            continue
        candidates = []
        path = _text(row.get("path"))
        if path:
            path = path if os.path.isabs(path) else os.path.join(base, path)
            if _lexically_inside(path, folders):
                candidates.append(path)
        host = _text(row.get("host"))
        ext = (_text(row.get("format")) or (_extension(path) if path else "") or "").lower().lstrip(".")
        if host and ext:
            candidates.append(os.path.join(base, "collection", host, modality, f"{rec_id}.{ext}"))
        for candidate in candidates:
            checked = _capture_file(base, candidate)
            # the file has to be this row's: its own id, in its modality's folder
            if checked and checked[2] == modality and checked[3] == rec_id:
                found.append((rec_id, modality, checked[1], checked[0], row))
                break
    return found


def _scan(base: str) -> list[tuple[str, str, str, str, dict]]:
    found = []
    collection = os.path.join(base, "collection")
    try:
        hosts = sorted(entry.name for entry in os.scandir(collection) if entry.is_dir())
    except OSError:
        return found
    for host in hosts:
        for modality in MODALITIES:
            try:
                names = sorted(entry.name for entry in os.scandir(os.path.join(collection, host, modality)))
            except OSError:
                continue
            for name in names:
                stem, dot, ext = name.rpartition(".")
                if not dot or ext.lower() not in MEDIA_TYPES or not REC_ID_RE.fullmatch(stem):
                    continue
                checked = _capture_file(base, os.path.join(collection, host, modality, name))
                if checked and checked[2] == modality and checked[3] == stem:
                    found.append((stem, modality, checked[1], checked[0], {}))
    return found


def _found(root: str, sid: str) -> list[tuple[str, str, str, str, dict]]:
    """(id, modality, host folder, real path, manifest row) of each capture recording of the
    session, from its manifest when it has one, else from collection/; the first of an id only."""
    base = _session_dir(root, sid)
    if base is None:
        return []
    rows = _read_manifest(base)
    folders = (os.path.abspath(os.path.join(root, sid)), base)
    found = _from_manifest(base, rows, folders) if rows is not None else _scan(base)
    unique, seen = [], set()
    for item in found:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    return unique
