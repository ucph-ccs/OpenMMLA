"""The sessions the dashboard lists, and the description of one.

The list comes from one Flux request with independent branches per aggregate: per session and
event type the record count and the first window start (count/min of `window_start_time`), the
last window end and the newest point time (max of `window_end_time`), how many transcripts carry
a `participant` or `diarization`, and the distinct `speakers` lists of the recognition buckets.
Each branch is its own from |> range |> filter |> group |> aggregate chain, which InfluxDB answers
from its storage engine without reading the rows (sharing one `from` between the branches
defeats that and takes over ten times longer). The speakers branch reads values, but only one
short string per 3 s bucket.

A session's span is its data's: t0 the earliest window start and t1 the latest window end of
ASR recognition, IPS translation and VFA features. The session id's minute stamp and MongoDB's
`metadata.initial_sync_time` say when recording started, while the document's start_time and
end_time are when someone ran it (a replay runs years after the recording), so they are never
the span. MongoDB is optional: it adds the session's status, devices, streams, the IPS main
camera and provenance, and lists sessions created before any data arrived. It is only read, with
the masked pipeline config (`components.config`, most of a document) left out."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openmmla.analytics.report.common import (
    EVENT_BUCKETS, FLUX_FAR_FUTURE, MODALITY_EVENT_TYPES, SILENT_LABELS, is_pupil_tag, parse_experiment_id,
    parse_session_id, query_tables, round_or_none, to_epoch, valid_session_id,
)
from openmmla.utils.asr_scope import participant_of
from openmmla.utils.constants import INFLUXDB_MEASUREMENT

LIVE_SECONDS = 20.0
# the record counts a session list entry shows
INDEX_COUNTS = ('asr_recognition', 'asr_transcription', 'ips_translation', 'vfa_features')
# the event types whose windows make a session's span; the others only when none of these ran
SPAN_EVENT_TYPES = ('asr_recognition', 'ips_translation', 'vfa_features')
GROUP_ID_RE = re.compile(r'^group_\d+$')
_SESSION_PROJECTION = {'_id': 0, 'components.config': 0}
# what the session list needs from a document
_INDEX_PROJECTION = {
    '_id': 0, 'session_id': 1, 'experiment_id': 1, 'group_id': 1, 'status': 1, 'metadata.initial_sync_time': 1,
    'sources.pipeline': 1, 'sources.base_id': 1, 'sources.server_path': 1, 'sources.capture.kind': 1,
    'components.key': 1, 'components.pipeline': 1, 'components.role': 1, 'components.id': 1,
}


# ---- InfluxDB ----

def _stats_query(bucket: str, sid: str | None = None, pose: bool = False) -> str:
    where = f' and r.session_id == "{sid}"' if sid else ''
    source = (f'from(bucket: "{bucket}") |> range(start: 0, stop: {FLUX_FAR_FUTURE})'
              f' |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}"{where}')
    by_type = '|> group(columns: ["session_id", "event_type"])'
    lines = [
        f'{source} and r._field == "window_start_time") {by_type} |> count() |> yield(name: "n")',
        f'{source} and r._field == "window_start_time") {by_type} |> min() |> yield(name: "first")',
        f'{source} and r._field == "window_end_time") {by_type} |> max() |> yield(name: "last")',
        f'{source} and r.event_type == "asr_transcription" and (r._field == "participant" or r._field == "diarization"))'
        f' |> group(columns: ["session_id", "_field"]) |> count() |> yield(name: "fields")',
        f'{source} and r.event_type == "asr_recognition" and r._field == "speakers")'
        f' |> group(columns: ["session_id"]) |> distinct() |> yield(name: "speakers")',
        # the distinct wearers and chunk labels: a worn microphone that is not a badge (a test mic)
        # makes no wearer session, and a group chunk's label says a group microphone ran
        f'{source} and r.event_type == "asr_transcription" and (r._field == "participant" or r._field == "speaker"))'
        f' |> group(columns: ["session_id", "_field"]) |> distinct() |> yield(name: "names")',
    ]
    if pose:
        lines.append(f'{source} and r.event_type == "vfa_features" and r._field == "pose_model") |> last()'
                     f' |> yield(name: "pose")')
    return '\n'.join(lines)


def _speaker_names(value) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, list):
        return [str(name) for name in value if name is not None]
    return [] if value is None else [str(value)]


def _empty_stats() -> dict:
    return {'types': {}, 'fields': {}, 'speakers': set(), 'participants': set(), 'chunk_speakers': set(),
            'pose_model': None}


def _session_stats(client, sid: str | None = None, pose: bool = False) -> dict[str, dict]:
    """{session_id: {"types": {event_type: {n, first, last, newest}}, "fields": {field: n},
    "speakers": set, "participants": set, "chunk_speakers": set, "pose_model": str | None}} from
    one request."""
    out: dict[str, dict] = {}

    def session(key) -> dict:
        return out.setdefault(key, _empty_stats())

    for table in query_tables(client, _stats_query(client.bucket, sid, pose)):
        for record in table.records:
            values = record.values
            key = values.get('session_id')
            if not key:
                continue
            name, value = values.get('result'), values.get('_value')
            entry = session(key)
            if name in ('n', 'first', 'last'):
                event_type = values.get('event_type')
                if not event_type:
                    continue
                stats = entry['types'].setdefault(event_type, {'n': 0, 'first': None, 'last': None, 'newest': None})
                if name == 'n':
                    stats['n'] += int(value or 0)
                elif name == 'first':
                    stats['first'] = to_epoch(value)
                else:
                    stats['last'] = to_epoch(value)
                    stats['newest'] = to_epoch(values.get('_time'))
            elif name == 'fields':
                entry['fields'][str(values.get('_field'))] = int(value or 0)
            elif name == 'speakers':
                entry['speakers'].update(_speaker_names(value))
            elif name == 'names' and value not in (None, ''):
                held = 'participants' if values.get('_field') == 'participant' else 'chunk_speakers'
                entry[held].add(str(value))
            elif name == 'pose' and value:
                entry['pose_model'] = str(value)
    return out


def _span(types: dict) -> tuple[float | None, float | None]:
    """(t0, t1): the earliest window start and the latest window end of the span event types (all
    types when none of them has records)."""
    chosen = [types[name] for name in SPAN_EVENT_TYPES if types.get(name, {}).get('n')]
    if not chosen:
        chosen = [stats for stats in types.values() if stats.get('n')]
    firsts = [stats['first'] for stats in chosen if stats.get('first') is not None]
    lasts = [stats['last'] for stats in chosen if stats.get('last') is not None]
    t0 = min(firsts) if firsts else None
    t1 = max(lasts) if lasts else None
    if t0 is not None and t1 is not None and t1 < t0:
        t1 = t0
    return t0, t1


def _newest(types: dict) -> float | None:
    times = [stats['newest'] for stats in types.values() if stats.get('newest') is not None]
    return max(times) if times else None


def _is_group_id(name: str, group_id: str | None) -> bool:
    return bool(name) and (GROUP_ID_RE.match(name) is not None or (bool(group_id) and name == group_id))


def speech_mode(has_asr: bool, participant_transcripts: int, recognition_names,
                group_id: str | None = None, participants=None, chunk_speakers=()) -> str | None:
    """how a session heard speech: 'wearer' (worn microphones: transcripts carry a participant and
    no recognition bucket names a group), 'wearer+group' (both), 'group' (recognition names only
    group ids, `group_NN` or the session's group_id, or silence), 'individual' (named speakers of
    speaker verification); None without ASR.

    With `participants` (the distinct values transcripts carry) only a badge id makes a wearer: a
    worn microphone labelled otherwise (a test mic such as `badge-0`) neither makes the session a
    wearer session nor counts as a named speaker. `chunk_speakers` are the labels transcripts carry:
    a group id among them says a group microphone ran even where its buckets named nobody. The
    analysis (speech.speech_mode) and the session list use this one rule."""
    if not has_asr:
        return None
    names = {str(name) for name in recognition_names or ()} - SILENT_LABELS
    if participants is not None:
        worn = {participant_of(value) for value in participants} - {None}
        wearers = {tag for tag in worn if is_pupil_tag(tag)}
        names -= worn - wearers
    else:
        wearers = bool(participant_transcripts and participant_transcripts > 0)
    groups = {name for name in names if _is_group_id(name, group_id)}
    groups |= {str(name) for name in chunk_speakers or () if _is_group_id(str(name), group_id)}
    if wearers:
        return 'wearer+group' if groups else 'wearer'
    if not names - groups:
        return 'group'
    return 'individual'


def speech_mode_of(recognitions: list[dict], transcriptions: list[dict], group_id: str | None = None) -> str | None:
    """speech_mode from fetched records, for code that holds them rather than the index."""
    names: set[str] = set()
    for record in recognitions:
        names.update(_speaker_names(record.get('speakers')))
    participants = {participant_of(record.get('participant')) for record in transcriptions} - {None}
    speakers = {str(record.get('speaker')) for record in transcriptions
                if participant_of(record.get('participant')) is None and record.get('speaker') is not None}
    worn = sum(1 for record in transcriptions if participant_of(record.get('participant')) is not None)
    return speech_mode(bool(recognitions or transcriptions), worn, names, group_id, participants=participants,
                       chunk_speakers=speakers)


# ---- MongoDB ----

def open_mongo(config: dict, timeout_ms: int = 1500):
    """the session database of a config (the merged dashboard config, or its MongoDB section),
    None when pymongo is missing, nothing is configured or the server does not answer within
    `timeout_ms` (1.5 s for a page; a report job, whose result is cached, waits longer, since a
    first connection over the tailnet can take 1.7 s). Read-only by use: unlike
    MongoDBClientWrapper it creates no index."""
    section = (config or {}).get('MongoDB', config) if isinstance(config, dict) else None
    url = section.get('url') if isinstance(section, dict) else None
    if not url or not isinstance(url, str) or url.startswith('<'):
        return None
    try:
        from pymongo import MongoClient
    except ImportError:
        return None
    client = None
    try:
        client = MongoClient(url, serverSelectionTimeoutMS=timeout_ms, connectTimeoutMS=timeout_ms,
                             socketTimeoutMS=10000, appname='openmmla-dashboard')
        client.admin.command('ping')
        return client[str(section.get('db') or 'openmmla')]
    except Exception:  # unreachable, refused, a malformed URL: the dashboard runs without it
        if client is not None:
            try:
                client.close()  # its monitor threads would keep knocking otherwise
            except Exception:
                pass
        return None


def mongo_session(mongo_db, sid: str) -> dict | None:
    """the session's document without its masked pipeline configs; None when it is not there, the
    id is not a session id, or MongoDB does not answer."""
    if mongo_db is None or not valid_session_id(sid):
        return None
    try:
        return mongo_db['sessions'].find_one({'session_id': sid}, _SESSION_PROJECTION)
    except Exception:
        return None


def _mongo_index(mongo_db) -> tuple[dict[str, dict], str | None]:
    if mongo_db is None:
        return {}, None
    try:
        docs = list(mongo_db['sessions'].find({}, _INDEX_PROJECTION))
    except Exception as error:
        return {}, f'MongoDB did not answer ({type(error).__name__}), so the list shows InfluxDB sessions only.'
    return {doc['session_id']: doc for doc in docs if isinstance(doc.get('session_id'), str)}, None


def _natural_key(text: str) -> tuple:
    return tuple((0, int(part), '') if part.isascii() and part.isdigit() else (1, 0, part)
                 for part in re.split(r'(\d+)', str(text)) if part)


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _components(doc: dict | None) -> list[dict]:
    return [entry for entry in _list((doc or {}).get('components')) if isinstance(entry, dict)]


def _started(entry: dict) -> float:
    moment = to_epoch(entry.get('started_at'))
    return moment if moment is not None else float('-inf')


def _newest_first(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=_started, reverse=True)


def _ips_synchronizers(doc: dict | None) -> list[dict]:
    return _newest_first([entry for entry in _components(doc)
                          if entry.get('pipeline') == 'ips' and entry.get('role') == 'synchronizer'])


def _matrices_of(entry: dict) -> dict | None:
    files = entry.get('files')
    found = files.get('transformation_matrices') if isinstance(files, dict) else None
    return found if isinstance(found, dict) else None


def _ips_main(doc: dict | None) -> str | None:
    """the camera whose frame the session's IPS positions are in: the synchronizer's (the newest
    one's, since a relaunch can leave a stale entry), else what the bases say."""
    for entry in _ips_synchronizers(doc):
        parameters = entry.get('parameters') if isinstance(entry.get('parameters'), dict) else {}
        arguments = entry.get('arguments') if isinstance(entry.get('arguments'), dict) else {}
        matrices = _matrices_of(entry) or {}
        for value in (parameters.get('main_id'), matrices.get('main_id'), arguments.get('main_camera')):
            if value not in (None, ''):
                return str(value)
    for entry in _newest_first(_components(doc)):
        parameters = entry.get('parameters') if isinstance(entry.get('parameters'), dict) else {}
        if entry.get('pipeline') == 'ips' and parameters.get('main_id') not in (None, ''):
            return str(parameters['main_id'])
    return None


def _turn(value) -> int:
    """0, 90, 180 or 270 of a source's capture.rotate (how the capture host turned the picture,
    clockwise); anything else, and a session from before the capture turned pictures, is 0."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number in (0, 90, 180, 270) else 0


def _text(value) -> str | None:
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    text = str(value).strip()
    return text or None


def _video_items(sources: list[dict], components: list[dict]) -> list[dict]:
    """one item per camera the session's IPS and VFA bases took: {key, label, vfa, ips, paths,
    rotate}. The bases are grouped by the stream's name (sources[].stream), else by the file a
    file-mode base read, else by the base id, so the IPS and the VFA base on one camera share an
    item; its key is the stream's name when there is one, else the VFA base id, else the IPS one
    (a file-mode camera's key is its base id, not its file). `vfa` and `ips` are
    the bases' ids (the VFA one names the camera in the frame sets), `paths` the stream paths on the
    Stream Server, `rotate` how the capture turned the picture. VFA cameras first, then IPS-only
    ones, each by name."""
    items: dict[str, dict] = {}
    names: dict[str, str] = {}
    for entry in sources:
        pipeline = entry.get('pipeline')
        base_id = _text(entry.get('base_id'))
        if pipeline not in ('ips', 'vfa') or base_id is None:
            continue
        capture = entry.get('capture') if isinstance(entry.get('capture'), dict) else {}
        if capture.get('kind') not in (None, '', 'video'):
            continue
        stream = _text(entry.get('stream'))
        source_file = _text(entry.get('source_index')) if entry.get('source') == 'file' else None
        group = f'stream:{stream}' if stream else f'file:{source_file}' if source_file else f'base:{base_id}'
        item = items.get(group)
        if item is None:
            item = items[group] = {'key': None, 'label': None, 'vfa': None, 'ips': None, 'paths': [], 'rotate': 0}
            names[group] = stream or ''
        if item[pipeline] is None:
            item[pipeline] = base_id
        path = str(entry.get('server_path') or '').strip('/')
        if path and path not in item['paths']:
            item['paths'].append(path)
        item['rotate'] = item['rotate'] or _turn(capture.get('rotate'))
    if not sources:
        for entry in components:
            parameters = entry.get('parameters') if isinstance(entry.get('parameters'), dict) else {}
            base_id = _text(entry.get('id') or parameters.get('base_id') or parameters.get('id'))
            pipeline = entry.get('pipeline')
            if entry.get('role') != 'base' or pipeline not in ('ips', 'vfa') or base_id is None:
                continue
            item = items.setdefault(f'base:{base_id}', {'key': None, 'label': None, 'vfa': None, 'ips': None,
                                                        'paths': [], 'rotate': 0})
            names.setdefault(f'base:{base_id}', '')
            item[pipeline] = item[pipeline] or base_id
    by_key: dict[str, dict] = {}
    for group, item in items.items():
        key = names[group] or item['vfa'] or item['ips']
        kept = by_key.get(key)
        if kept is None:
            by_key[key] = dict(item, key=key)
            continue
        # two groups that came to one key (a stream name that is also another camera's base id)
        # are one camera
        for pipeline in ('vfa', 'ips'):
            kept[pipeline] = kept[pipeline] or item[pipeline]
        kept['paths'] += [path for path in item['paths'] if path not in kept['paths']]
        kept['rotate'] = kept['rotate'] or item['rotate']
    out = list(by_key.values())
    for item in out:
        both = 'VFA + IPS' if item['vfa'] and item['ips'] else 'VFA' if item['vfa'] else 'IPS'
        item['label'] = f"{item['key']} · {both}"
    out.sort(key=lambda item: (item['vfa'] is None, _natural_key(item['key'])))
    return out


def archive_summary(value) -> dict | None:
    """{status, files, bytes, location, verified_at (epoch)} of an `archive` block `mmla
    ses-archive` wrote (the session document's or the manifest's); None for anything else."""
    if not isinstance(value, dict) or not value:
        return None
    status = _text(value.get('status'))
    if status is None and value.get('location') in (None, ''):
        return None

    def count(key):
        number = value.get(key)
        return int(number) if isinstance(number, (int, float)) and not isinstance(number, bool) and number >= 0 else None

    return {'status': status, 'files': count('files'), 'bytes': count('bytes'),
            'location': _text(value.get('location')), 'verified_at': to_epoch(value.get('verified_at'))}


def mongo_devices(doc: dict | None) -> dict:
    """the cameras (IPS and VFA bases) and microphones (ASR bases) a session used, the Stream Server
    paths its bases pulled (each with the stream's name and how its capture turned the picture),
    one video item per camera (_video_items), and its IPS main camera; from the document's
    `sources`, else from its base components."""
    cameras: list[str] = []
    microphones: list[str] = []
    streams: list[dict] = []

    def note(pipeline, base_id) -> None:
        if base_id in (None, ''):
            return
        target = microphones if pipeline == 'asr' else cameras if pipeline in ('ips', 'vfa') else None
        if target is not None and str(base_id) not in target:
            target.append(str(base_id))

    sources = [entry for entry in _list((doc or {}).get('sources')) if isinstance(entry, dict)]
    for entry in sources:
        pipeline = entry.get('pipeline')
        note(pipeline, entry.get('base_id'))
        path = str(entry.get('server_path') or '').strip('/')
        if path and all(stream['path'] != path for stream in streams):
            capture = entry.get('capture') if isinstance(entry.get('capture'), dict) else {}
            kind = capture.get('kind') or ('audio' if pipeline == 'asr' else 'video')
            streams.append({'path': path, 'pipeline': pipeline, 'base_id': str(entry.get('base_id') or ''),
                            'kind': str(kind), 'stream': _text(entry.get('stream')),
                            'rotate': _turn(capture.get('rotate'))})
    if not sources:
        for entry in _components(doc):
            if entry.get('role') == 'base':
                parameters = entry.get('parameters') if isinstance(entry.get('parameters'), dict) else {}
                note(entry.get('pipeline'), entry.get('id') or parameters.get('base_id') or parameters.get('id'))
    return {
        'cameras': sorted(cameras, key=_natural_key),
        'microphones': sorted(microphones, key=_natural_key),
        'streams': sorted(streams, key=lambda stream: _natural_key(stream['path'])),
        'video': _video_items(sources, _components(doc)),
        'ips_main': _ips_main(doc),
    }


def _vector(value) -> list[float] | None:
    """[x, y, z] of a 3x1 column ([[x], [y], [z]]) or a flat triple."""
    try:
        flat = [row[0] if isinstance(row, (list, tuple)) else row for row in value]
        if len(flat) != 3:
            return None
        out = [float(v) for v in flat]
    except (TypeError, ValueError, IndexError):
        return None
    return out if all(v == v and abs(v) != float('inf') for v in out) else None


def _optical_axis(rotation) -> list[float] | None:
    """camera k's optical axis (its z) in the main camera's frame: column 2 of R_k, unit length."""
    try:
        axis = [float(rotation[i][2]) for i in range(3)]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    norm = sum(v * v for v in axis) ** 0.5
    if not norm or norm != norm:
        return None
    return [round(v / norm, 6) for v in axis]


def ips_cameras(doc: dict | None) -> list[dict]:
    """the session's IPS cameras placed in the main camera's frame (metres, OpenCV axes: x right,
    y down, z forward), from the synchronizer's transformation matrices (p_main = R_k p_k + T_k):
    the main camera at the origin looking along z, each other at T_k looking along R_k's column 2.
    Only the cameras the session's IPS bases used, when the document lists them; [] without
    matrices or a main camera."""
    main = _ips_main(doc)
    matrices: dict = {}
    for entry in _ips_synchronizers(doc):
        found = _matrices_of(entry)
        if found:
            main = main or (str(found['main_id']) if found.get('main_id') else None)
            matrices = found.get('matrices') if isinstance(found.get('matrices'), dict) else {}
            break
    if not main:
        return []
    used = {str(entry.get('base_id')) for entry in _list((doc or {}).get('sources'))
            if isinstance(entry, dict) and entry.get('pipeline') == 'ips' and entry.get('base_id') not in (None, '')}
    out = [{'id': main, 'main': True, 'position': [0.0, 0.0, 0.0], 'axis': [0.0, 0.0, 1.0]}]
    for camera in sorted(matrices, key=_natural_key):
        if str(camera) == main or (used and str(camera) not in used):
            continue
        matrix = matrices[camera] if isinstance(matrices[camera], dict) else {}
        position = _vector(matrix.get('T'))
        if position is None:
            continue
        out.append({'id': str(camera), 'main': False, 'position': [round(v, 4) for v in position],
                    'axis': _optical_axis(matrix.get('R'))})
    return out


def _services(entry: dict) -> dict:
    return entry.get('services') if isinstance(entry.get('services'), dict) else {}


def _provenance(doc: dict | None, pose_model: str | None) -> dict | None:
    """what made the session's data: the transcriber and its language, diarization, the pose model
    and whether gaze ran, and each component's software (newest first)."""
    components = _newest_first(_components(doc))
    if not components and not pose_model:
        return None
    transcriber = language = gaze = diarize = None
    for entry in components:
        services = _services(entry)
        arguments = entry.get('arguments') if isinstance(entry.get('arguments'), dict) else {}
        parameters = entry.get('parameters') if isinstance(entry.get('parameters'), dict) else {}
        if entry.get('pipeline') == 'asr' and entry.get('role') == 'base':
            speech = services.get('speech_transcriber')
            speech = speech if isinstance(speech, dict) else {}
            transcriber = transcriber or speech.get('model')
            language = language or speech.get('language') or parameters.get('language') or arguments.get('language')
            if isinstance(arguments.get('diarize'), bool):
                diarize = bool(diarize) or arguments['diarize']
        if entry.get('pipeline') == 'vfa' and entry.get('role') == 'synchronizer':
            analyzer = services.get('vllm_frame_analyzer')
            analyzer = analyzer if isinstance(analyzer, dict) else {}
            features = analyzer.get('features') if isinstance(analyzer.get('features'), dict) else {}
            pose_model = pose_model or features.get('pose_model')
            if gaze is None and isinstance(arguments.get('gaze'), bool):
                gaze = arguments['gaze']
    listed = []
    for entry in components:
        software = entry.get('software') if isinstance(entry.get('software'), dict) else {}
        listed.append({'key': entry.get('key'), 'pipeline': entry.get('pipeline'), 'role': entry.get('role'),
                       'id': entry.get('id'), 'host': entry.get('host'), 'git_commit': software.get('git_commit'),
                       'openmmla': software.get('openmmla'), 'started_at': to_epoch(entry.get('started_at'))})
    newest = next((entry for entry in listed if entry['git_commit']), {})
    return {'transcriber': transcriber, 'language': language, 'diarize': diarize, 'pose_model': pose_model,
            'gaze': gaze, 'git_commit': newest.get('git_commit'), 'openmmla': newest.get('openmmla'),
            'components': listed}


# ---- building the list and the meta ----

def _identity(sid: str, doc: dict | None) -> dict:
    parsed = parse_session_id(sid)
    doc = doc or {}
    if isinstance(doc.get('experiment_id'), str) and doc['experiment_id']:
        experiment = parse_experiment_id(doc['experiment_id'])
        parsed['experiment'] = doc['experiment_id']
        parsed['date'] = experiment['date'] or parsed['date']
        parsed['task'] = experiment['task'] or parsed['task']
    if isinstance(doc.get('group_id'), str) and doc['group_id']:
        parsed['group'] = doc['group_id']
    metadata = doc.get('metadata') if isinstance(doc.get('metadata'), dict) else {}
    synced = to_epoch(metadata.get('initial_sync_time'))
    if synced is not None and synced > 0:
        parsed['recorded_start'] = synced
    return parsed


def _state(newest: float | None, now: float) -> dict:
    live = newest is not None and now - newest < LIVE_SECONDS
    return {'live': live, 'last_event': newest, 'lag': round_or_none(now - newest, 1) if newest is not None else None,
            'label': 'live' if live else 'ended'}


def _entry(sid: str, stats: dict | None, doc: dict | None, now: float) -> dict:
    stats = stats or _empty_stats()
    types = stats['types']
    identity = _identity(sid, doc)
    t0, t1 = _span(types)
    counts = {name: int(types.get(name, {}).get('n') or 0) for name in INDEX_COUNTS}
    modalities = [name for name, event_types in MODALITY_EVENT_TYPES.items()
                  if any(types.get(event_type, {}).get('n') for event_type in event_types)]
    newest = _newest(types)
    mode = speech_mode('asr' in modalities, stats['fields'].get('participant', 0), stats['speakers'],
                       identity['group'], participants=stats.get('participants'),
                       chunk_speakers=stats.get('chunk_speakers'))
    mongo = None
    if doc is not None:
        devices = mongo_devices(doc)
        mongo = {'status': doc.get('status'), 'cameras': devices['cameras'], 'microphones': devices['microphones'],
                 'streams': len(devices['streams'])}
    if identity['date'] is None and (identity['recorded_start'] or t0):
        identity['date'] = time.strftime('%Y-%m-%d', time.gmtime(identity['recorded_start'] or t0))
    return {
        'id': sid, **identity,
        't0': t0, 't1': t1, 'duration': round_or_none(t1 - t0, 3) if t0 is not None and t1 is not None else None,
        'counts': counts, 'modalities': modalities, 'speech_mode': mode,
        'last_event': newest, 'state': _state(newest, now)['label'], 'mongo': mongo,
    }


def _newest_order(entry: dict) -> tuple:
    moment = entry.get('recorded_start') or entry.get('t0')
    return (moment is not None, moment or 0.0, entry['id'])


def session_index(client, mongo_db=None) -> dict:
    """every session in InfluxDB (and the MongoDB-only ones), newest recording first, with its
    span, record counts, modalities, speech mode and live state. Raises InfluxUnavailable when
    InfluxDB cannot be asked; MongoDB trouble is a warning."""
    now = time.time()
    warnings: list[str] = []
    stats = _session_stats(client)
    docs, problem = _mongo_index(mongo_db)
    if problem:
        warnings.append(problem)
    skipped = sorted(sid for sid in set(stats) | set(docs) if not valid_session_id(sid))
    if skipped:
        warnings.append(f'{len(skipped)} session id(s) hold characters the dashboard does not accept '
                        'and are not listed.')
    sessions = [_entry(sid, stats.get(sid), docs.get(sid), now)
                for sid in set(stats) | set(docs) if valid_session_id(sid)]
    sessions.sort(key=_newest_order, reverse=True)
    return {'sessions': sessions, 'generated_at': now, 'warnings': warnings}


def _coverage(n: int, duration: float | None, bucket: float) -> float | None:
    if not n or not duration or duration <= 0:
        return None
    return round(min(n / (duration / bucket), 1.0), 4)


def _type_summary(stats: dict | None, duration: float | None, bucket: float) -> dict:
    stats = stats or {}
    n = int(stats.get('n') or 0)
    return {'n': n, 'first': stats.get('first'), 'last': stats.get('last'), 'coverage': _coverage(n, duration, bucket)}


def session_meta(client, sid: str, mongo_db=None) -> dict | None:
    """the description of one session (the meta object of the dashboard's API): identity, span,
    live state, per-modality counts and coverage, and, from MongoDB, devices, streams, the video
    items (one per camera), where the session is archived, the IPS main camera and its cameras'
    placement, and provenance. None when neither InfluxDB nor MongoDB
    knows the session. Raises ValueError for an invalid id, InfluxUnavailable when InfluxDB cannot
    be asked. `report` is left None for the server to fill."""
    if not valid_session_id(sid):
        raise ValueError('not a valid session id')
    now = time.time()
    stats = _session_stats(client, sid, pose=True).get(sid)
    doc = mongo_session(mongo_db, sid)
    if stats is None and doc is None:
        return None
    entry = _entry(sid, stats, doc, now)
    stats = stats or _empty_stats()
    types = stats['types']
    duration = entry['duration']
    modalities: dict[str, Any] = {'asr': None, 'ips': None, 'vfa': None}
    if 'asr' in entry['modalities']:
        modalities['asr'] = {
            'recognition': _type_summary(types.get('asr_recognition'), duration, EVENT_BUCKETS['asr_recognition']),
            'transcription': {'n': int(types.get('asr_transcription', {}).get('n') or 0)},
            'mode': entry['speech_mode'],
            'worn_transcripts': int(stats['fields'].get('participant', 0)),
            'diarized_transcripts': int(stats['fields'].get('diarization', 0)),
        }
    if 'ips' in entry['modalities']:
        modalities['ips'] = _type_summary(types.get('ips_translation'), duration, EVENT_BUCKETS['ips_translation'])
    if 'vfa' in entry['modalities']:
        modalities['vfa'] = _type_summary(types.get('vfa_features'), duration, EVENT_BUCKETS['vfa_features'])
    devices = mongo_devices(doc) if doc is not None else None
    return {
        'id': sid, 'experiment': entry['experiment'], 'date': entry['date'], 'task': entry['task'],
        'group': entry['group'], 'recorded_start': entry['recorded_start'],
        't0': entry['t0'], 't1': entry['t1'], 'duration': duration,
        'state': _state(entry['last_event'], now),
        'modalities': modalities,
        'devices': {'cameras': devices['cameras'], 'microphones': devices['microphones']} if devices else None,
        'streams': devices['streams'] if devices else [],
        'video': devices['video'] if devices else [],
        'archive': archive_summary(doc.get('archive')) if doc is not None else None,
        'ips_main': devices['ips_main'] if devices else None,
        'ips_cameras': ips_cameras(doc) if doc is not None else [],
        'provenance': _provenance(doc, stats.get('pose_model')),
        'report': None,
        'mongo': doc is not None,
    }
