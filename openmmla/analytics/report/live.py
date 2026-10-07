"""The records of the live stream, slimmed to what the live page draws.

A browser following a session gets every record its pipelines write, about one IPS window and
one VFA frame set a second and a recognition bucket every 3 s, for as long as it watches. Sent
as stored, a frame set alone is about 24 KB (every body's skeleton with its scores, every pair of
bodies in a full room); slimmed it is a few KB: tagged persons keep their skeletons and gaze, untagged ones a
box, their track and the keypoints the pose model is sure of (the rest null), pairs only among
tagged persons with distances in frame widths. Recognition drops the level traces (about 5 MB a
session), transcripts keep words, their times and voice keys.

Times stay absolute (epoch seconds, 3 decimals); word and turn times are relative to their
chunk's start (2 decimals). A worn microphone's words carry their class (wearer, crosstalk,
other) as the analysis decides it, from the level traces of every worn microphone over the last
few minutes the stream has seen (WornWords), so the live page can dim the neighbours' words. IPS positions are laid on the floor with space.project, so the live
map and the analysis map share one coordinate system. The VFA gaze targets are remade with the
fusion's hand circle (window_features.relabel_hand_circles), so what the live page counts as a
look at someone's hands is what the analysis counts. A tagged body says whether its tag was read
in that frame or only kept on its track (`rm`), so the live page can give a kept tag the fusion's
time limit."""

from __future__ import annotations

import math

from openmmla.analytics.report.common import (
    VoiceKeys, chunk_turns, chunk_words, finite_number, is_pupil_tag, parsed_value, round_or_none, tag_sort_key,
)
from openmmla.utils.asr_scope import participant_of

# an untagged body keeps only the keypoints the pose model scored at least this (the live page
# draws no keypoint below it either, cameras.js KEYPOINT_MIN_CONF)
UNTAGGED_MIN_CONFIDENCE = 0.3


def _space():
    # imported here: the space module is written beside this one and may not exist yet
    from openmmla.analytics.report import space
    return space


def _start(record: dict, length: float = 0.0) -> float | None:
    """the record's window start, else its point time less its length."""
    start = finite_number(record.get('window_start_time'))
    if start is not None:
        return start
    stamp = finite_number(record.get('_t'))
    return stamp - length if stamp is not None else None


def _r3(value) -> float | None:
    return round_or_none(value, 3)


def slim_recognition(rec: dict) -> dict | None:
    """a recognition bucket: start, length, the speakers with their voiced seconds, the worn
    microphones' SNRs (None without them). None for a record without a time."""
    start = _start(rec, 3.0)
    if start is None:
        return None
    end = finite_number(rec.get('window_end_time'))
    speakers = parsed_value(rec.get('speakers'))
    speakers = [str(name) for name in speakers] if isinstance(speakers, list) else (
        [] if speakers is None else [str(speakers)])
    durations = parsed_value(rec.get('durations'))
    durations = durations if isinstance(durations, list) else []
    durations = [_r3(value) for value in durations[:len(speakers)]]
    durations += [None] * (len(speakers) - len(durations))
    energies = parsed_value(rec.get('energies'))
    snrs = None
    if isinstance(energies, dict):
        snrs = {str(tag): round_or_none(snr, 1) for tag, snr in energies.items()}
    return {'t': _r3(start), 'd': _r3(end - start) if end is not None and end >= start else 3.0,
            'sp': speakers, 'du': durations, 'en': snrs}


def slim_transcription(rec: dict, voices: VoiceKeys, classes: list | None = None) -> dict | None:
    """a transcription chunk: its span, label, wearer (worn microphones), text, and each word and
    diarized turn relative to the chunk's start with its voice key. None for a record without a
    time. `voices` is the stream's own VoiceKeys, fed every chunk in time order; `classes` the
    word classes WornWords gave the chunk (in word_times order), None when it has none: words are
    [word, start, end, voice key, class]."""
    start = _start(rec)
    if start is None:
        return None
    end = finite_number(rec.get('window_end_time'))
    end = end if end is not None and end >= start else start
    record = rec if finite_number(rec.get('window_start_time')) is not None else dict(rec, window_start_time=start)
    turns = [[round(first - start, 2), round(last - start, 2), key] for first, last, key in chunk_turns(record, voices)]
    spoken = chunk_words(record, voices)
    if classes is not None:
        from openmmla.analytics.report.speech import _align
        classes = _align(record, spoken, classes)
    else:
        classes = [None] * len(spoken)
    words = [[word, round(first - start, 2), round(last - start, 2), key, cls]
             for (word, first, last, key), cls in zip(spoken, classes)]
    text = rec.get('text')
    text = text.strip() if isinstance(text, str) else None
    speaker = rec.get('speaker')
    return {'t': _r3(start), 'e': _r3(end), 'sp': None if speaker is None else str(speaker),
            'pt': participant_of(rec.get('participant')), 'text': text or None, 'w': words, 'turns': turns}


class WornWords:
    """the class of each word a worn microphone transcribes (wearer, crosstalk, other), decided as
    the analysis decides it (window_features.personal_speech) over the recognition buckets and worn
    chunks of the last KEEP seconds the stream has seen. A word is judged on every worn microphone's
    level while it was said, and the buckets holding those levels arrive before the chunk (the
    synchronizer writes a bucket every 3 s, a transcript only once it is transcribed). Sessions
    whose microphones left no level traces get no classes (None), as in the analysis. One per
    connection; feed it in time order."""

    KEEP = 180.0

    def __init__(self):
        self._recognitions: list[dict] = []
        self._chunks: list[dict] = []

    @property
    def active(self) -> bool:
        """whether a worn microphone's chunk has come (the session has worn microphones)."""
        return bool(self._chunks)

    @staticmethod
    def _trim(records: list[dict], newest: float, keep: float) -> list[dict]:
        return [record for record in records if (finite_number(record.get('window_start_time')) or 0.0) >= newest - keep]

    def add_recognitions(self, records) -> None:
        records = [record for record in records or () if finite_number(record.get('window_start_time')) is not None]
        if not records:
            return
        self._recognitions.extend(records)
        newest = max(finite_number(record.get('window_start_time')) for record in self._recognitions)
        self._recognitions = self._trim(self._recognitions, newest, self.KEEP)

    def classify(self, records) -> dict[int, list | None]:
        """{id(record): classes in word_times order | None} for the worn chunks among `records`
        (the chunks of one batch); {} when none is worn or the classes cannot be decided."""
        worn = [record for record in records or ()
                if participant_of(record.get('participant')) is not None
                and finite_number(record.get('window_start_time')) is not None]
        if not worn:
            return {}
        self._chunks.extend(worn)
        newest = max(finite_number(record.get('window_start_time')) for record in self._chunks)
        self._chunks = self._trim(self._chunks, newest, self.KEEP)
        try:
            from openmmla.analytics.fusion.window_features import _end_time, _time, personal_speech
            personal = personal_speech(self._recognitions, self._chunks)
            if personal is None or not personal.by_word:
                return {}
            return {id(record): personal.word_classes(record, _time(record), _end_time(record, 1.0))
                    for record in worn}
        except Exception:  # the classes dim words on the page; a chunk without them still shows
            return {}


def merge_ips(translations: list, rotations: list, relations: list) -> list[dict]:
    """the three IPS records of each synchronizer bucket joined by their window start (the three
    share it exactly), sorted: [{"t", "tr", "rot", "graph"}], {} for a part a bucket lacks."""
    windows: dict[float, dict] = {}
    for part, records, field in (('tr', translations, 'translations'), ('rot', rotations, 'rotations'),
                                 ('graph', relations, 'graph')):
        for record in records or ():
            start = _start(record, 1.0)
            if start is None:
                continue
            window = windows.get(start)
            if window is None:
                window = windows[start] = {'t': start, 'tr': {}, 'rot': {}, 'graph': {}}
            value = parsed_value(record.get(field))
            window[part] = value if isinstance(value, dict) else {}
    return [windows[start] for start in sorted(windows)]


def _facing(rotation) -> tuple[float, float, float] | None:
    """a badge's outward normal (the wearer's facing) in the camera frame: -column 2 of R."""
    rotation = parsed_value(rotation)
    try:
        facing = (-float(rotation[0][2]), -float(rotation[1][2]), -float(rotation[2][2]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return facing if all(math.isfinite(v) for v in facing) else None


def slim_ips(window: dict, basis: dict | None) -> dict:
    """an IPS bucket of merge_ips on the floor: per tag (u, v, h) metres (space.project; the
    camera's x-z plane without a basis), the heading of the badge's facing on the floor in
    radians (atan2(f·ex, f·ef), 0 = away from the main camera, positive towards u), and the
    directed facing edges."""
    space = _space()
    positions, headings = {}, {}
    translations = window.get('tr') if isinstance(window.get('tr'), dict) else {}
    for tag in sorted(translations, key=tag_sort_key):
        point = space.position(translations[tag])
        if point is None:
            continue
        u, v, h = space.project(point, basis)
        positions[str(tag)] = [_r3(u), _r3(v), _r3(h)]
    rotations = window.get('rot') if isinstance(window.get('rot'), dict) else {}
    for tag in sorted(rotations, key=tag_sort_key):
        facing = _facing(rotations[tag])
        if facing is None:
            continue
        u, v, _ = space.project(facing, basis)
        if math.hypot(u, v) > 1e-9:
            headings[str(tag)] = _r3(math.atan2(u, v))
    graph = window.get('graph') if isinstance(window.get('graph'), dict) else {}
    edges = sorted({(str(source), str(target)) for source, targets in graph.items()
                    for target in (targets if isinstance(targets, list) else []) if str(target) != str(source)},
                   key=lambda edge: (tag_sort_key(edge[0]), tag_sort_key(edge[1])))
    return {'t': _r3(window.get('t')), 'p': positions, 'hd': headings, 'f': [list(edge) for edge in edges]}


def _int_points(values, size: int) -> list[int] | None:
    if not isinstance(values, (list, tuple)) or len(values) < size:
        return None
    out = []
    for value in values[:size]:
        number = finite_number(value)
        if number is None:
            return None
        out.append(int(round(number)))
    return out


def _person_tag(person: dict) -> str | None:
    """the badge a body carries: its tag_id, else a person_id that is a tag number."""
    tag = person.get('tag_id')
    if tag is not None and not isinstance(tag, bool):
        number = finite_number(tag)
        if number is not None and number.is_integer():
            return str(int(number))
        return str(tag)
    name = person.get('person_id')
    text = str(name) if name is not None else ''
    return text if text.isascii() and text.isdigit() else None


def _keypoints(values, min_confidence: float | None = None) -> list | None:
    """[x, y, confidence] per keypoint (pixels as ints, confidence 2 decimals). With
    `min_confidence`, a keypoint scored below it (or unreadable) is None, which keeps the
    untagged bodies' skeletons small."""
    if not isinstance(values, list) or not values:
        return None
    out = []
    for point in values:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            out.append(None if min_confidence is not None else [0, 0, 0.0])
            continue
        x, y = finite_number(point[0]), finite_number(point[1])
        confidence = finite_number(point[2]) if len(point) > 2 else None
        if min_confidence is not None and (x is None or y is None or confidence is None
                                           or confidence < min_confidence):
            out.append(None)
            continue
        out.append([int(round(x)) if x is not None else 0, int(round(y)) if y is not None else 0,
                    round(confidence, 2) if confidence is not None else 0.0])
    if min_confidence is not None and all(point is None for point in out):
        return None
    return out


def _gaze(person: dict) -> dict:
    gaze = person.get('gaze') if isinstance(person.get('gaze'), dict) else {}
    target = gaze.get('target') if isinstance(gaze.get('target'), dict) else {}
    looked = target.get('person_id')
    if looked is None:
        to = None
    elif is_pupil_tag(looked):
        to = str(looked)
    else:
        to = 'other'
    return {'p': _int_points(gaze.get('point'), 2), 'cat': str(target.get('category') or 'unknown'), 'to': to}


def _slim_person(person: dict) -> tuple[dict, str | None]:
    """a body: tagged ones with every keypoint, head yaw, gaze and face box, and `rm`: 0 when the
    camera read the tag in this frame (`tag_match` torso or box, or an older event that does not
    say), 1 when the features endpoint only kept it on the body's track (`tag_match` track), which
    the live page trusts for TAG_MEMORY_SECONDS after the track last read it, as the fusion does
    (window_features.expire_track_tags); untagged ones with their person_id (`id`, track_<n> or
    unknown_<n>) and the keypoints scored at least UNTAGGED_MIN_CONFIDENCE, so the live page can
    draw them as people without a badge read."""
    tag = _person_tag(person)
    track = finite_number(person.get('track_id'))
    slim = {'tag': tag, 'tr': int(track) if track is not None and track.is_integer() else None,
            'b': _int_points(person.get('bbox'), 4), 'k': None, 'yaw': None, 'g': None}
    if tag is not None:
        slim['rm'] = 1 if person.get('tag_match') == 'track' else 0
        slim['k'] = _keypoints(person.get('keypoints'))
        slim['yaw'] = round_or_none(person.get('head_yaw'), 1)
        slim['g'] = _gaze(person)
        slim['fb'] = _int_points(person.get('face_bbox'), 4)
    else:
        slim['k'] = _keypoints(person.get('keypoints'), UNTAGGED_MIN_CONFIDENCE)
        name = person.get('person_id')
        slim['id'] = str(name) if name is not None and str(name) else None
    return slim, tag


def _pair_values(entry, width: float) -> list:
    entry = entry if isinstance(entry, dict) else {}
    out = []
    for name in ('gaze_distance', 'hand_distance'):
        value = finite_number(entry.get(name))
        out.append(round(value / width, 4) if value is not None and width else None)
    return out


def _camera_order(frame: dict) -> tuple:
    return (str(frame.get('camera') or frame.get('angle') or ''),)


def slim_vfa(rec: dict, relabel: bool = True) -> dict | None:
    """a frame set: per camera (by id) its size, the AprilTag centres it decoded, every body
    (tagged ones with skeleton, head yaw, face box, gaze and whether the tag was read or kept;
    untagged ones with a box, their person_id and the skeleton's sure keypoints) and the
    pairs of tagged bodies (`pr`, gaze and hand distance in frame widths). The top-level `pr`
    holds each pair once per frame set: from the first camera that measured its gaze distance,
    else the first that holds the pair. None for a record without a time."""
    start = _start(rec)
    if start is None:
        return None
    if relabel:
        try:
            from openmmla.analytics.fusion.window_features import relabel_hand_circles
            rec = relabel_hand_circles([rec])[0]
        except Exception:  # a frame the relabelling cannot read is sent as the server made it
            pass
    frames = parsed_value(rec.get('features'))
    frames = [frame for frame in frames if isinstance(frame, dict)] if isinstance(frames, list) else []
    cameras, pairs = [], {}
    for index, frame in enumerate(sorted(frames, key=_camera_order)):
        width = finite_number(frame.get('width')) or 0.0
        height = finite_number(frame.get('height')) or 0.0
        tags = frame.get('tags') if isinstance(frame.get('tags'), dict) else {}
        centres = {}
        for tag in sorted(tags, key=tag_sort_key):
            point = _int_points(tags[tag], 2)
            if point is not None:
                centres[str(tag)] = point
        persons, tagged = [], set()
        for person in frame.get('persons') if isinstance(frame.get('persons'), list) else []:
            if not isinstance(person, dict):
                continue
            slim, tag = _slim_person(person)
            persons.append(slim)
            if tag is not None:
                tagged.add(tag)
        camera_pairs = {}
        stored = frame.get('pairs') if isinstance(frame.get('pairs'), dict) else {}
        for key, entry in stored.items():
            parts = str(key).split('|')
            if len(parts) != 2 or parts[0] == parts[1] or not (parts[0] in tagged and parts[1] in tagged):
                continue
            first, second = sorted(parts, key=tag_sort_key)
            camera_pairs[f'{first}|{second}'] = _pair_values(entry, width)
        camera = frame.get('camera') or frame.get('angle') or f'camera {index + 1}'
        cameras.append({'id': str(camera), 'w': int(width), 'h': int(height), 'tg': centres, 'ps': persons,
                        'pr': camera_pairs})
        for key, values in camera_pairs.items():
            if key not in pairs or (pairs[key][0] is None and values[0] is not None):
                pairs[key] = values
    return {'t': _r3(start), 'c': cameras, 'pr': {key: pairs[key] for key in sorted(pairs)}}
