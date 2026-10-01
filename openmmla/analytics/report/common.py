"""What every part of the dashboard's report shares: reading a session's events from InfluxDB,
turning values into plain JSON, the order of tags, and the keys of diarized voices.

Events are read straight from the query API, never through InfluxDBClientWrapper._execute_query,
because that swallows every error into an empty list, and a dashboard has to tell "no data" from
"database down". Each row comes back pivoted, with the Influx point time as `_t` (epoch seconds;
it is the window's end) instead of the wrapper's datetime `time`, which plain JSON cannot carry,
and with its JSON fields parsed.

Voices are the anonymous speakers a group microphone's diarization linked across chunks. Their
numbers count within one stream (a microphone and the registry its base started), so the same
number from a second stream is another person: VoiceKeys keeps the first stream's keys plain
(`v:2`) and marks the others (`v:2@2`), the same way for the analysis and the live stream."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from datetime import date, datetime, timezone
from typing import Any, Callable

from openmmla.utils.constants import INFLUXDB_MEASUREMENT
from openmmla.utils.querys import deep_parse_json

SESSION_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')
# event types are pasted into Flux like session ids, so they are held to plain names
EVENT_TYPE_RE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')

MODALITY_EVENT_TYPES = {
    'asr': ('asr_recognition', 'asr_transcription'),
    'ips': ('ips_translation', 'ips_rotation', 'ips_relation'),
    'vfa': ('vfa_features',),
}
EVENT_TYPES = tuple(event_type for types in MODALITY_EVENT_TYPES.values() for event_type in types)
# the length of one record of the types that set a session's span and coverage, in seconds
EVENT_BUCKETS = {'asr_recognition': 3.0, 'ips_translation': 1.0, 'vfa_features': 1.0}
# the trust bound of a badge id (fusion.window_features.MAX_PUPIL_TAG): a higher id is a misread
MAX_PUPIL_TAG = 12
SILENT_LABELS = frozenset({'silent', 'unknown', ''})

# the upper bound of every unbounded range: a base whose clock runs ahead of the server's must
# still be read, which range()'s default stop (now) would not do
FLUX_FAR_FUTURE = '2200-01-01T00:00:00Z'
# fields that hold free text: never parsed as JSON, even when a transcript reads like a list
_TEXT_FIELDS = frozenset({'text', 'speaker', 'participant', 'attribution', 'voice_registry', 'pose_model',
                          'session_id', 'event_type'})
_DROPPED_COLUMNS = frozenset({'_start', '_stop', '_time', '_measurement', 'result', 'table', 'time'})
# a word the aligner gave a start and no end lasts this long (window_features.WORD_SECONDS)
WORD_SECONDS = 0.3

UNLINKED_KEY = 'v:?'
_VOICE_KEY_RE = re.compile(r'^v:(?P<voice>-?\d+)(?:@(?P<stream>\d+))?$')


class InfluxUnavailable(RuntimeError):
    """InfluxDB could not be asked (unreachable, refused the token, rejected the query). The
    message says why in a line fit for the page; it never holds the token."""


def valid_session_id(sid) -> bool:
    return isinstance(sid, str) and SESSION_ID_RE.fullmatch(sid) is not None


def valid_event_type(event_type) -> bool:
    return isinstance(event_type, str) and EVENT_TYPE_RE.fullmatch(event_type) is not None


def require_valid(sid, event_type=None) -> None:
    """ValueError unless sid is a session id and event_type (when given) an event type name: both
    are pasted into Flux, so nothing else may reach a query."""
    if not valid_session_id(sid):
        raise ValueError('not a valid session id')
    if event_type is not None and not valid_event_type(event_type):
        raise ValueError('not a valid event type')


_STAMP_RE = re.compile(r'_(\d{6}T\d{4})Z\Z')
_GROUP_RE = re.compile(r'_(group_[A-Za-z0-9-]+)\Z')
_EXPERIMENT_RE = re.compile(r'^exp_(\d{4})(\d{2})(\d{2})(?:_(.+))?$')


def parse_session_id(sid: str) -> dict:
    """what a session id `<exp>_<group>_<YYMMDDTHHMMZ>` says: the experiment, its date and task,
    the group, and the recording start to the minute (epoch). Parts it does not hold are None."""
    out = {'experiment': None, 'date': None, 'task': None, 'group': None, 'recorded_start': None}
    head = str(sid or '')
    stamp = _STAMP_RE.search(head)
    if stamp:
        try:
            moment = datetime.strptime(stamp.group(1), '%y%m%dT%H%M').replace(tzinfo=timezone.utc)
            out['recorded_start'] = moment.timestamp()
        except ValueError:
            pass
        head = head[:stamp.start()]
    group = _GROUP_RE.search(head)
    if group:
        out['group'] = group.group(1)
        head = head[:group.start()]
    if head and (group or stamp or head.startswith('exp_')):
        out['experiment'] = head
        out.update(parse_experiment_id(head))
    return out


def parse_experiment_id(experiment: str) -> dict:
    """the date (ISO) and task an experiment id `exp_<YYYYMMDD>_<task>` names, None where it
    names none."""
    out = {'date': None, 'task': None}
    match = _EXPERIMENT_RE.match(str(experiment or ''))
    if match:
        try:
            out['date'] = date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
        except ValueError:
            pass
        out['task'] = match.group(4)
    return out


_RFC3339_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})(?:[Tt ](\d{2}:\d{2})(:\d{2})?(?:\.(\d+))?)?'
                         r'\s*([Zz]|[+-]\d{2}:?\d{2})?$')


def to_epoch(value) -> float | None:
    """epoch seconds of a datetime (naive is UTC, as pymongo gives it), a number, or a string
    (a number or an RFC 3339 time, nanoseconds allowed); None for anything else."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        try:
            return value.timestamp()
        except (ValueError, OverflowError, OSError):  # pandas NaT, out of range
            return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            number = None
        if number is not None:
            return number if math.isfinite(number) else None
        match = _RFC3339_RE.match(text)
        if not match:
            return None
        day, minutes, seconds, fraction, zone = match.groups()
        iso = f'{day}T{minutes or "00:00"}{seconds or ":00"}.{(fraction or "")[:6].ljust(6, "0")}'
        if zone in (None, 'Z', 'z'):
            zone = '+00:00'
        elif ':' not in zone:
            zone = f'{zone[:3]}:{zone[3:]}'
        try:
            return datetime.fromisoformat(iso + zone).timestamp()
        except (ValueError, OverflowError, OSError):
            return None
    item = getattr(value, 'item', None)  # numpy scalars
    if callable(item):
        try:
            return to_epoch(item())
        except (TypeError, ValueError):
            return None
    return None


def _set_order(value) -> tuple:
    return (type(value).__name__, str(value))


def _json_key(key) -> str:
    if isinstance(key, str):
        return key
    if key is None:
        return 'null'
    if isinstance(key, bool):
        return 'true' if key else 'false'
    if hasattr(key, 'item') and callable(key.item):
        key = key.item()
    return str(key)


def jsonable(obj):
    """a deep copy of obj that json.dumps takes as it is and that a browser parses: numpy values
    become Python ones, datetimes epoch seconds, NaN and infinities None, tuples lists, sets sorted
    lists, keys strings."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {_json_key(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, (set, frozenset)):
        return [jsonable(value) for value in sorted(obj, key=_set_order)]
    if isinstance(obj, datetime):
        return to_epoch(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if type(obj).__name__ in ('NAType', 'NaTType'):
        return None
    tolist = getattr(obj, 'tolist', None)  # numpy arrays and scalars, pandas series
    if callable(tolist):
        try:
            return jsonable(tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode('utf-8', 'replace')
    try:
        number = float(obj)  # a Decimal or a Fraction
    except (TypeError, ValueError):
        return str(obj)
    return number if math.isfinite(number) else None


def round_or_none(value, digits: int = 2):
    """value rounded, None when it is not a finite number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def tag_sort_key(tag) -> tuple:
    """tags in numeric order where they are numbers, the others after them by name."""
    text = str(tag)
    if text.isascii() and text.isdigit():
        return (0, int(text), text)
    return (1, 0, text)


def sort_tags(tags) -> list[str]:
    """the distinct tags as strings, in tag order (tag_sort_key)."""
    return sorted({str(tag) for tag in tags if tag is not None}, key=tag_sort_key)


def is_pupil_tag(tag) -> bool:
    """whether a tag can be a participant's badge: a number no higher than the trust bound."""
    text = str(tag) if tag is not None else ''
    return text.isascii() and text.isdigit() and int(text) <= MAX_PUPIL_TAG


def finite_number(value) -> float | None:
    """value as a float, None when it is not a finite number (bools are not numbers here)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parsed_value(value):
    """a field as structured data: rows fetched without parsing still hold JSON strings."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ('{', '['):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
    return value


# ---- reading events ----

def _flux_time(epoch, default: str) -> str:
    if epoch is None:
        return default
    number = finite_number(epoch)
    if number is None:
        raise ValueError('time bound is not a number')
    # the stamps are microseconds from a float epoch, so a bound in microseconds is exact
    return f'time(v: {int(round(number * 1_000_000)) * 1000})'


def influx_reason(error: BaseException) -> str:
    """one line on why a query failed, from what the client raised (HTTP status and InfluxDB's
    message, or the connection error), without the token, which only travels in a header."""
    status = getattr(error, 'status', None)
    body = getattr(error, 'body', None)
    message = None
    if body:
        try:
            message = json.loads(body).get('message')
        except (TypeError, ValueError, AttributeError):
            message = None
    if status:
        reason = message or getattr(error, 'reason', None) or type(error).__name__
        text = f'HTTP {status}: {reason}'
    else:
        text = str(error).strip() or type(error).__name__
    text = ' '.join(text.split())
    return text if len(text) <= 300 else text[:297] + '...'


_gzip_apis: dict = {}
_gzip_lock = threading.Lock()


def _event_query_api(client):
    """a query API on the client's server that asks for gzip-compressed answers. Event rows are
    mostly JSON text: 5 min of video features are 7.7 MB as InfluxDB sends them and 1.5 MB
    compressed, which over the tailnet is the difference between 6-28 s and 2.5 s. One per server
    and token, kept for the process; the client's own API when it does not say where it points."""
    url, token, org = getattr(client, 'url', None), getattr(client, 'token', None), getattr(client, 'org', None)
    if not isinstance(url, str) or not isinstance(token, str):
        return client.query_api
    key = (url, org, hashlib.sha256(token.encode()).hexdigest())
    api = _gzip_apis.get(key)
    if api is None:
        with _gzip_lock:
            api = _gzip_apis.get(key)
            if api is None:
                try:
                    from influxdb_client import InfluxDBClient
                    api = InfluxDBClient(url=url, token=token, org=org, enable_gzip=True,
                                         timeout=(5_000, 120_000)).query_api()
                except Exception:  # an older client: the uncompressed API still answers
                    api = client.query_api
                _gzip_apis[key] = api
    return api


def _records(client, query: str):
    """the records of a query, as the query API streams them; InfluxUnavailable on any failure."""
    try:
        stream = _event_query_api(client).query_stream(query=query, org=client.org)
        for record in stream:
            yield record
    except InfluxUnavailable:
        raise
    except Exception as error:  # the client raises urllib3, ApiException and parsing errors alike
        raise InfluxUnavailable(influx_reason(error)) from error


def query_tables(client, query: str):
    """the tables of a Flux query (several yields come back in one list); InfluxUnavailable on
    any failure."""
    try:
        return client.query_api.query(query=query, org=client.org)
    except Exception as error:
        raise InfluxUnavailable(influx_reason(error)) from error


def _events_query(client, sid: str, event_type: str, start, end) -> str:
    return (
        f'from(bucket: "{client.bucket}")\n'
        f'  |> range(start: {_flux_time(start, "0")}, stop: {_flux_time(end, FLUX_FAR_FUTURE)})\n'
        f'  |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}" and r.session_id == "{sid}"'
        f' and r.event_type == "{event_type}")\n'
        f'  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
    )


def _row(values: dict, parse: bool) -> dict:
    row = {key: value for key, value in values.items() if key not in _DROPPED_COLUMNS}
    moment = values.get('_time')
    row['_t'] = moment.timestamp() if isinstance(moment, datetime) else to_epoch(moment)
    if parse:
        for key, value in row.items():
            if key not in _TEXT_FIELDS and isinstance(value, str) and value.lstrip()[:1] in ('{', '['):
                row[key] = deep_parse_json(value)
    return row


def _start_order(row: dict) -> tuple:
    stamp = finite_number(row.get('_t'))
    stamp = stamp if stamp is not None else math.inf
    start = finite_number(row.get('window_start_time'))
    return (start if start is not None else stamp, stamp)


def fetch(client, sid: str, event_type: str, start: float | None = None, end: float | None = None,
          parse: bool = True) -> list[dict]:
    """every record of one event type of a session whose point time (`_t`, the window's end)
    lies in [start, end), unbounded where None: pivoted, its JSON fields parsed (unless `parse` is
    False), sorted by window_start_time. Raises ValueError for an id that cannot be a session's
    and InfluxUnavailable when InfluxDB cannot be asked."""
    require_valid(sid, event_type)
    query = _events_query(client, sid, event_type, start, end)
    rows = [_row(record.values, parse) for record in _records(client, query)]
    rows.sort(key=_start_order)
    return rows


def event_span(client, sid: str, event_type: str) -> tuple[float, float] | None:
    """the first and last point time of one event type of a session, None when it has none."""
    require_valid(sid, event_type)
    source = (f'from(bucket: "{client.bucket}") |> range(start: 0, stop: {FLUX_FAR_FUTURE})'
              f' |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}" and r.session_id == "{sid}"'
              f' and r.event_type == "{event_type}" and r._field == "window_end_time")')
    query = f'{source} |> first() |> yield(name: "first")\n{source} |> last() |> yield(name: "last")'
    found: dict[str, float] = {}
    for table in query_tables(client, query):
        for record in table.records:
            moment = to_epoch(record.values.get('_time'))
            name = record.values.get('result')
            if moment is None:
                continue
            if name == 'first':
                found['first'] = min(moment, found.get('first', moment))
            else:
                found['last'] = max(moment, found.get('last', moment))
    if 'first' not in found or 'last' not in found:
        return None
    return found['first'], found['last']


def fetch_chunked(client, sid: str, event_type: str, start: float | None, end: float | None,
                  chunk: float = 300.0, progress: Callable[[float, float], Any] | None = None,
                  parse: bool = True) -> list[dict]:
    """fetch() in pieces of `chunk` seconds of point time, so a session's video features (about
    90 MB an hour) come in steps a progress bar can follow: progress(done_seconds, total_seconds)
    after each piece. The last piece takes `end` itself too, so the session's first and last
    point times (event_span, or the meta's t0 and state.last_event) bound every record of it.
    Without start or end, the event type's own span is looked up first."""
    require_valid(sid, event_type)
    if start is None or end is None:
        span = event_span(client, sid, event_type)
        if span is None:
            if progress is not None:
                progress(0.0, 0.0)
            return []
        start = span[0] if start is None else start
        end = span[1] if end is None else end
    start, end = float(start), float(end)
    if end < start:
        return []
    chunk = max(float(chunk), 1.0)
    total = end - start
    rows: list[dict] = []
    first = start
    while True:
        last = min(first + chunk, end)
        final = last >= end
        rows.extend(fetch(client, sid, event_type, first, last + 1e-6 if final else last, parse=parse))
        if progress is not None:
            progress(round(last - start, 3), round(total, 3))
        if final:
            break
        first = last
    rows.sort(key=_start_order)
    return rows


def last_event_time(client, sid: str) -> float | None:
    """the newest point time (epoch) of any event of the session, None when it has none."""
    require_valid(sid)
    query = (f'from(bucket: "{client.bucket}") |> range(start: 0, stop: {FLUX_FAR_FUTURE})'
             f' |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}" and r.session_id == "{sid}"'
             f' and r._field == "window_end_time") |> last()')
    newest = None
    for table in query_tables(client, query):
        for record in table.records:
            moment = to_epoch(record.values.get('_time'))
            if moment is not None and (newest is None or moment > newest):
                newest = moment
    return newest


def last_event_times(client, sid: str) -> dict[str, float]:
    """the newest point time (epoch) of each event type of the session, in one query: {event_type:
    epoch}, without the types it has none of. The live stream starts each type from its own newest
    record, since the types come from different synchronizers and video trails speech."""
    require_valid(sid)
    query = (f'from(bucket: "{client.bucket}") |> range(start: 0, stop: {FLUX_FAR_FUTURE})'
             f' |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}" and r.session_id == "{sid}"'
             f' and r._field == "window_end_time") |> last()')
    newest: dict[str, float] = {}
    for table in query_tables(client, query):
        for record in table.records:
            moment = to_epoch(record.values.get('_time'))
            event_type = record.values.get('event_type')
            if moment is None or not isinstance(event_type, str):
                continue
            if event_type not in newest or moment > newest[event_type]:
                newest[event_type] = moment
    return newest


# ---- voices ----

_voice_stream_of = None


def voice_stream(record: dict) -> str:
    """the stream a chunk's voice numbers count in (fusion.window_features._voice_stream)."""
    global _voice_stream_of
    if _voice_stream_of is None:
        from openmmla.analytics.fusion.window_features import _voice_stream
        _voice_stream_of = _voice_stream
    return _voice_stream_of(record)


def _voice_number(voice) -> int | None:
    if voice is None or isinstance(voice, bool):
        return None
    if isinstance(voice, int):
        return voice
    if isinstance(voice, float):
        return int(voice) if voice.is_integer() else None
    text = str(voice).strip()
    digits = text[1:] if text.startswith('-') else text
    return int(text) if digits.isascii() and digits.isdigit() else None


def entity_label(key: str) -> str:
    """the label of a speech entity key: voices (see VoiceKeys), `tag:0` Tag 0, `spk:<name>` the
    name, `group:<id>` Group mic, `other` Other voices; anything else as it is."""
    key = str(key)
    if key == UNLINKED_KEY:
        return 'Unlinked'
    match = _VOICE_KEY_RE.match(key)
    if match:
        stream = match.group('stream')
        label = f'Voice {match.group("voice")}'
        return f'{label} (mic {stream})' if stream else label
    if key.startswith('tag:'):
        return f'Tag {key[4:]}'
    if key.startswith('spk:'):
        return key[4:]
    if key.startswith('group:') or key == 'group':
        return 'Group mic'
    if key == 'other':
        return 'Other voices'
    return key


class VoiceKeys:
    """stable keys for the diarized voices of a session, whatever order its chunks are read in
    after the first: the first voice stream met keeps plain keys (`v:2`), each later one is
    numbered (`v:2@2`, `v:2@3`). Feed the chunks in time order, so the first is the earliest."""

    def __init__(self):
        self._streams: dict[str, int] = {}

    @property
    def streams(self) -> list[str]:
        """the voice streams met so far, in the order they got their number."""
        return list(self._streams)

    def key(self, record: dict, voice) -> str:
        number = _voice_number(voice)
        if number is None:
            return UNLINKED_KEY
        stream = voice_stream(record)
        index = self._streams.get(stream)
        if index is None:
            index = self._streams[stream] = len(self._streams) + 1
        return f'v:{number}' if index == 1 else f'v:{number}@{index}'

    @staticmethod
    def label(key: str) -> str:
        return entity_label(key)


def _linked_voice(linked: dict, speaker):
    """the voice the chunk's `voices` links a chunk-local SPEAKER_NN to, None when it links none."""
    entry = linked.get(speaker) if speaker is not None else None
    return entry.get('voice') if isinstance(entry, dict) else None


def chunk_turns(record: dict, voices: VoiceKeys) -> list[tuple[float, float, str]]:
    """the diarized turns of a transcription chunk, absolute (epoch), each with its voice key
    (UNLINKED_KEY for a turn no voice was linked to), sorted; [] for a chunk without turns."""
    start = finite_number(record.get('window_start_time'))
    turns = parsed_value(record.get('diarization'))
    if start is None or not isinstance(turns, list):
        return []
    linked = parsed_value(record.get('voices'))
    linked = linked if isinstance(linked, dict) else {}
    out = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        first, last = finite_number(turn.get('start')), finite_number(turn.get('end'))
        if first is None or last is None or last < first:
            continue
        voice = turn.get('voice')
        if voice is None:
            voice = _linked_voice(linked, turn.get('speaker'))
        out.append((start + first, start + last, voices.key(record, voice)))
    out.sort()
    return out


def chunk_words(record: dict, voices: VoiceKeys) -> list[tuple[str, float, float, str | None]]:
    """every word of a transcription chunk as (word, absolute start, absolute end, voice key).

    Times follow window_features.word_spans: a word's own start and end (WORD_SECONDS long when
    it has no end); a word the aligner could not place takes the span of the stamped word before
    it; words with no stamped word before them, and every word of a chunk with no stamps at all
    (or with only `text`), get an even share of the chunk by their place in it. The voice key is
    the word's voice, UNLINKED_KEY for a diarized word (it names a speaker) without one, None for a
    word of an undiarized chunk (a worn microphone's)."""
    start = finite_number(record.get('window_start_time'))
    if start is None:
        return []
    end = finite_number(record.get('window_end_time'))
    length = max((end if end is not None else start) - start, 0.0)
    entries = parsed_value(record.get('words'))
    if not isinstance(entries, list) or not entries:
        words = str(record.get('text') or '').split()
        share = length / len(words) if words else 0.0
        return [(word, start + share * i, start + share * (i + 1), None) for i, word in enumerate(words)]
    linked = parsed_value(record.get('voices'))
    linked = linked if isinstance(linked, dict) else {}
    share = length / len(entries)
    stamped = any(isinstance(entry, dict) and finite_number(entry.get('start')) is not None for entry in entries)
    out, last = [], None
    for i, entry in enumerate(entries):
        entry = entry if isinstance(entry, dict) else {'word': entry}
        word = entry.get('word')
        if word is None:
            word = entry.get('text')
        first = finite_number(entry.get('start')) if stamped else None
        if first is not None:
            until = finite_number(entry.get('end'))
            span = (first, until if until is not None and until > first else first + WORD_SECONDS)
            last = span
        elif last is not None:
            span = last
        else:
            span = (share * i, share * (i + 1))
        voice = entry.get('voice')
        if voice is None:
            voice = _linked_voice(linked, entry.get('speaker'))
        if _voice_number(voice) is not None:
            key = voices.key(record, voice)
        elif entry.get('speaker') is not None:
            key = UNLINKED_KEY
        else:
            key = None
        out.append(('' if word is None else str(word), start + span[0], start + span[1], key))
    return out
