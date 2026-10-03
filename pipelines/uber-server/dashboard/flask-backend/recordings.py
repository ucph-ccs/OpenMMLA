"""A session's raw capture recordings on the dashboard's machine, for the Analysis page's downloads.

The recorders keep each file as artifacts/<session>/collection/<host>/<audio|video>/<id>.<format>,
and artifacts/<session>/manifest.json lists them with their device, start, length and, for audio,
scope and wearer. A file is found by the manifest's `path` when that path exists inside the session
folder (the path was written on the machine that recorded or imported the session, often another
one), else by its place in collection/; a session without a manifest is read from collection/ alone.

The cuts `mmla ses-archive` took of the session's streams from the stream server are listed too:
artifacts/<session>/streams/server/<stream path>_<start>.<ext> (fMP4), from the manifest's
`stream_cuts` rows (their `relpath`, or a `path` spelled inside the session folder), else from a
scan of streams/server/. Each is listed with `source` "stream" and its stream path; the capture
files have `source` "collection".

Every file is listed with a length: the manifest row's `duration`, else what the file holds (a wav's
header, else ffprobe's, kept per path, size and modification time so a listing probes a file once),
else what the recorder noted (`audio_seconds`, or `stopped_at` - `start_time`). The replay ends a file
where its listed length does, so a file listed without one would hold the clock to the session's end.

Nothing else of the session folder is ever listed or served: not the speaker profiles (voice
biometrics of children), the coding clips under labels/, raw/, analysis/, pipelines/, the archive's
ledger under .archive/ or the manifests themselves, and nothing a symlink or a manifest path leads
to outside the session folder. Every file is checked on its real path (os.path.realpath), whichever
way it was found.

Standard library only (and ffprobe, when it is on the PATH).
"""

import json
import os
import re
import shutil
import subprocess
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
# where the archive keeps the stream server's cuts: streams/server/<stream path>_<start>.<ext>
STREAM_CUTS_DIR = ("streams", "server")
STREAM_CUT_NAME_RE = re.compile(r"^(?P<name>.+)_(?P<start>\d+(?:\.\d+)?)$")
# a stream path is at most this many folders deep below streams/server/ (MediaMTX's are app/name)
STREAM_CUT_DEPTH = 4
SOURCES = ("collection", "stream")
# a stream server's recording is the same stretch as an archived cut of its path when it starts
# within this many seconds of the cut, or lies inside it give or take as much
CUT_MATCH_SECONDS = 2.0
# how long one ffprobe of a file's length may take, and how many lengths are kept (the oldest go)
PROBE_TIMEOUT_SECONDS = 20.0
PROBED_MAX = 4096


def repo_root() -> str:
    """the repository this backend belongs to, four levels above this folder (as media.py's caller
    finds it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def artifacts_root(repo: str | None = None) -> str:
    """DASHBOARD_ARTIFACTS_DIR when set, else <repo>/artifacts (this backend's repository unless `repo`
    names another)."""
    configured = (os.environ.get("DASHBOARD_ARTIFACTS_DIR") or "").strip()
    return os.path.abspath(os.path.expanduser(configured)) if configured else os.path.join(repo or repo_root(), "artifacts")


def media_type(path: str) -> str:
    return MEDIA_TYPES.get(_extension(path), "application/octet-stream")


def download_name(sid: str, rec_id: str, path: str) -> str:
    """<session>_<recording>.<ext>, with anything but letters, digits, `_`, `.` and `-` replaced."""
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{sid}_{rec_id}")
    return f"{stem}.{_extension(path) or 'bin'}"


def list_recordings(root: str, sid: str) -> list[dict]:
    """the session's recordings that exist on this machine: the capture files first, then the
    archived stream cuts, each video first, then audio, by device: {id, source, modality, kind,
    host, device, stream_path, format, size, duration, start, scope, participant, channels,
    sample_rate, label}; [] for a session folder that does not exist."""
    records = [_record(path, host, modality, rec_id, row) for rec_id, modality, host, path, row in _found(root, sid)]
    taken = {record["id"] for record in records}
    records += [_cut_record(cut) for cut in _found_cuts(root, sid) if cut["id"] not in taken]
    records.sort(key=lambda r: (SOURCES.index(r["source"]), MODALITIES.index(r["modality"]),
                                (r["device"] or "").lower(), (r["host"] or r["stream_path"] or "").lower(),
                                r["start"] or 0.0, r["id"]))
    return records


def resolve(root: str, sid: str, rec_id: str) -> str | None:
    """the real path of one recording list_recordings lists, None for any other id."""
    if not isinstance(rec_id, str) or not REC_ID_RE.fullmatch(rec_id):
        return None
    found = next((path for found_id, _, _, path, _ in _found(root, sid) if found_id == rec_id), None)
    if found is not None:
        return found
    return next((cut["path"] for cut in _found_cuts(root, sid) if cut["id"] == rec_id), None)


def manifest_archive(root: str, sid: str) -> dict | None:
    """the `archive` block of the session's manifest on this machine (what `mmla ses-archive`
    wrote there when this machine holds the archive), None without one."""
    base = _session_dir(root, sid)
    data = _manifest(base) if base is not None else None
    archive = data.get("archive") if data else None
    return archive if isinstance(archive, dict) else None


def place_cuts(records: list[dict], sources) -> list[dict]:
    """give each archived stream cut among `records` (in place) what the session's document says of
    its stream path (`sources`, the document's sources[]): `camera`, the IPS or VFA base that took
    it (None for sound, or a path no base took), `pipeline`, and as `device` the session's name of
    the stream (sources[].stream) when it has one. The capture files get `camera` None."""
    by_path: dict[str, list[dict]] = {}
    for entry in sources if isinstance(sources, list) else []:
        if isinstance(entry, dict):
            path = str(entry.get("server_path") or "").strip("/")
            if path:
                by_path.setdefault(path, []).append(entry)
    for record in records:
        record.setdefault("camera", None)
        if record.get("source") != "stream":
            continue
        entries = by_path.get(str(record.get("stream_path") or ""), [])
        if not entries:
            record["camera"] = record["pipeline"] = None
            continue
        # a camera's base names the cut; an IPS and a VFA base on one path take the VFA one
        bases = sorted((e for e in entries if e.get("pipeline") in ("vfa", "ips") and _text(e.get("base_id"))),
                       key=lambda e: 0 if e.get("pipeline") == "vfa" else 1)
        first = bases[0] if bases else entries[0]
        record["camera"] = _text(bases[0].get("base_id")) if bases and record.get("modality") == "video" else None
        record["pipeline"] = _text(first.get("pipeline"))
        stream = next((_text(e.get("stream")) for e in entries if _text(e.get("stream"))), None)
        if stream:
            record["device"] = stream
    return records


def unarchived_spans(server: list[dict], cuts: list[dict], tolerance: float = CUT_MATCH_SECONDS) -> list[dict]:
    """the stream server's recordings (one span each, {path, spans: [[start, seconds]]}) that no
    archived cut of the same path holds: a cut holds a span that starts within `tolerance` seconds
    of it or lies inside it, give or take as much."""
    by_path: dict[str, list[tuple[float, float]]] = {}
    for cut in cuts or []:
        path, start = cut.get("stream_path"), _number(cut.get("start"))
        if path and start is not None:
            by_path.setdefault(str(path), []).append((start, max(_number(cut.get("duration")) or 0.0, 0.0)))

    def held(path: str, span) -> bool:
        try:
            start, seconds = float(span[0]), float(span[1])
        except (TypeError, ValueError, IndexError):
            return False
        for cut_start, cut_seconds in by_path.get(path, ()):
            if abs(start - cut_start) <= tolerance:
                return True
            if start >= cut_start - tolerance and start + seconds <= cut_start + cut_seconds + tolerance:
                return True
        return False

    out = []
    for entry in server or []:
        spans = [span for span in entry.get("spans") or [] if not held(str(entry.get("path")), span)]
        if spans:
            out.append(dict(entry, spans=spans))
    return out


def label(record: dict) -> str:
    """the line a person reads: Camera c920-05 on raspi5-01, Group mic jabra-0, Worn mic vimo-0,
    Tag 0, Worn mic badge-0, Microphone mic-9; Stream vfa/c920-05 for an archived stream cut."""
    if record.get("source") == "stream":
        return f"Stream {record.get('stream_path') or record.get('device') or ''}".rstrip()
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


def _manifest(base: str) -> dict | None:
    """the session's manifest.json, None when there is no readable one inside the session folder."""
    path = os.path.join(base, MANIFEST)
    try:
        real = os.path.realpath(path)
        if os.path.commonpath([base, real]) != base or not os.path.isfile(real):
            return None
        with open(real, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_manifest(base: str, key: str = "recordings") -> list | None:
    """one list of the manifest (its recordings unless `key` says otherwise), None when there is no
    readable manifest with one."""
    data = _manifest(base)
    rows = data.get(key) if data else None
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


# (real path, size, mtime_ns) -> seconds or None, so a file is probed once while it stays the same
_PROBED: dict[tuple[str, int, int], float | None] = {}


def _probe_seconds(path: str) -> float | None:
    """the length ffprobe reads from a media file (its container's duration; a fragmented MP4's
    fragments counted through), None without ffprobe or when it cannot say."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    lines = (result.stdout or "").split()
    seconds = _number(lines[0]) if result.returncode == 0 and lines else None
    return seconds if seconds is not None and seconds > 0 else None


def _media_seconds(path: str) -> float | None:
    """the length of the file at path by ffprobe, kept while its size and modification time stay."""
    try:
        real = os.path.realpath(path)
        stat = os.stat(real)
    except (OSError, ValueError):
        return None
    key = (real, stat.st_size, stat.st_mtime_ns)
    if key in _PROBED:
        return _PROBED[key]
    seconds = _probe_seconds(real)
    if len(_PROBED) >= PROBED_MAX:
        _PROBED.pop(next(iter(_PROBED)))
    _PROBED[key] = seconds
    return seconds


def _noted_seconds(row: dict) -> float | None:
    """the length a capture recorder noted in its manifest row: the samples its audio file holds, else
    the time from its start to its stop; None when it noted neither."""
    audio = _number(row.get("audio_seconds"))
    if audio is not None and audio > 0:
        return audio
    start, stopped = _number(row.get("start_time")), _number(row.get("stopped_at"))
    if start is not None and stopped is not None and stopped > start:
        return stopped - start
    return None


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
    if not duration or duration < 0:
        duration = _media_seconds(path) or _noted_seconds(row)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    record = {
        "id": rec_id, "source": "collection", "modality": modality, "kind": modality,
        "host": host or _text(row.get("host")) or named.get("host"),
        "device": device, "stream_path": None, "camera": None, "format": _extension(path), "size": size,
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
    session: its manifest's rows, then what collection/ holds besides (a file the manifest does not
    list, a manifest with an empty list); the first of an id and of a file only."""
    base = _session_dir(root, sid)
    if base is None:
        return []
    rows = _read_manifest(base)
    folders = (os.path.abspath(os.path.join(root, sid)), base)
    found = (_from_manifest(base, rows, folders) if rows is not None else []) + _scan(base)
    unique, seen, paths = [], set(), set()
    for item in found:
        if item[0] not in seen and item[3] not in paths:
            seen.add(item[0])
            paths.add(item[3])
            unique.append(item)
    return unique


# ---- the archived stream cuts ----

def _stream_cut_file(base: str, candidate: str) -> dict | None:
    """{id, path (real), stream_path, start, format} when `candidate` is an archived stream cut of
    the session at base, i.e. its real path is base/streams/server/<stream path>_<start>.<ext> with
    a media extension, no dot or private folder on the way; None for anything else. The id is the
    archive's: stream_<stream path with / as _>_<start>."""
    try:
        real = os.path.realpath(candidate)
        if os.path.commonpath([base, real]) != base:
            return None
        parts = os.path.relpath(real, base).split(os.sep)
    except (OSError, ValueError):
        return None
    head = len(STREAM_CUTS_DIR)
    if tuple(parts[:head]) != STREAM_CUTS_DIR or not head + 1 < len(parts) <= head + STREAM_CUT_DEPTH:
        return None
    if any(not part or part.startswith(".") or part.lower() in PRIVATE_PARTS for part in parts):
        return None
    stem, dot, ext = parts[-1].rpartition(".")
    named = STREAM_CUT_NAME_RE.fullmatch(stem) if dot else None
    if not named or ext.lower() not in MEDIA_TYPES:
        return None
    stream_path = "/".join(parts[head:-1] + [named.group("name")])
    rec_id = f"stream_{stream_path.replace('/', '_')}_{named.group('start')}"
    if not REC_ID_RE.fullmatch(rec_id):
        return None
    try:
        if not os.path.isfile(real):
            return None
    except (OSError, ValueError):
        return None
    return {"id": rec_id, "path": real, "stream_path": stream_path, "start": float(named.group("start")),
            "format": ext.lower()}


def _cuts_from_manifest(base: str, rows: list, folders: tuple[str, ...]) -> list[dict]:
    """the files of the manifest's stream_cuts rows: a row's `relpath`, else its `path` when that is
    spelled inside the session folder; the file has to carry the row's id."""
    found = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rec_id = row.get("id")
        if not isinstance(rec_id, str) or not REC_ID_RE.fullmatch(rec_id):
            continue
        candidates = []
        relpath = _text(row.get("relpath"))
        if relpath and not os.path.isabs(relpath):
            candidates.append(os.path.join(base, relpath))
        path = _text(row.get("path"))
        if path:
            path = path if os.path.isabs(path) else os.path.join(base, path)
            if _lexically_inside(path, folders):
                candidates.append(path)
        for candidate in candidates:
            if not _lexically_inside(os.path.abspath(candidate), folders):
                continue
            checked = _stream_cut_file(base, candidate)
            if checked and checked["id"] == rec_id:
                found.append(dict(checked, row=row))
                break
    return found


def _scan_cuts(base: str) -> list[dict]:
    found = []
    top = os.path.join(base, *STREAM_CUTS_DIR)
    if not os.path.isdir(top):
        return found
    for folder, dirs, names in os.walk(top):
        depth = len(os.path.relpath(folder, top).split(os.sep)) if folder != top else 0
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d.lower() not in PRIVATE_PARTS
                         and depth + 1 < STREAM_CUT_DEPTH)
        for name in sorted(names):
            checked = _stream_cut_file(base, os.path.join(folder, name))
            if checked:
                found.append(dict(checked, row={}))
    return found


def _found_cuts(root: str, sid: str) -> list[dict]:
    """the session's archived stream cuts on this machine, {id, path, stream_path, start, format,
    row}: the manifest's stream_cuts rows, then what streams/server/ holds besides; the first of an
    id and of a file only."""
    base = _session_dir(root, sid)
    if base is None:
        return []
    rows = _read_manifest(base, "stream_cuts")
    folders = (os.path.abspath(os.path.join(root, sid)), base)
    found = (_cuts_from_manifest(base, rows, folders) if rows is not None else []) + _scan_cuts(base)
    unique, seen, paths = [], set(), set()
    for item in found:
        if item["id"] not in seen and item["path"] not in paths:
            seen.add(item["id"])
            paths.add(item["path"])
            unique.append(item)
    return unique


def _cut_record(cut: dict) -> dict:
    """the listing entry of one archived stream cut: its modality from its manifest row, else from
    the stream's app (an asr/ stream is sound); its device the stream's name (the last part of its
    path, which the dashboard replaces with the session's own name of the stream when it knows it)."""
    row = cut.get("row") or {}
    stream_path = _text(row.get("stream_path")) or cut["stream_path"]
    modality = row.get("modality") if row.get("modality") in MODALITIES else (
        "audio" if stream_path.split("/", 1)[0] == "asr" else "video")
    start = _number(row.get("start_time"))
    duration = _number(row.get("duration"))
    if not duration or duration < 0:
        duration = _media_seconds(cut["path"])
    try:
        size = os.path.getsize(cut["path"])
    except OSError:
        size = None
    record = {
        "id": cut["id"], "source": "stream", "modality": modality, "kind": modality, "host": None,
        "device": stream_path.rsplit("/", 1)[-1], "stream_path": stream_path, "camera": None, "format": cut["format"],
        "size": size,
        "duration": round(duration, 3) if duration and duration > 0 else None,
        "start": round(start if start is not None else cut["start"], 6),
        "scope": None, "participant": None, "channels": None, "sample_rate": None,
    }
    record["label"] = label(record)
    return record
