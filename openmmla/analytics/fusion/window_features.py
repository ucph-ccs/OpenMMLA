"""The fusion table of a session: one row per window (10 s by default) with, side by side, what
the ASR, IPS and VFA pipelines saw in it, as plain numbers with a coverage count per modality.

Speech comes from `asr_recognition` (who-or-whether someone spoke, 3 s buckets) and
`asr_transcription` (talk spurts, words, and the anonymous speaker turns of a diarized chunk);
space from `ips_translation` and `ips_relation` (positions, facing, movement, presence); body
and gaze from `vfa_features` (head yaw, wrist speed, where gazes land, joint attention, hand
distance); the semantic layer from `vfa_action` (the VLM's labels). Every person is a tag id,
every pair a sorted tag pair; persons the pose model saw without a tag are left out.

The events come from InfluxDB (a session id) or from a Sessions -> Export Measurements folder
(`<session>_<suffix>.json`), so a table can be built offline from an export.

Body and gaze follow each camera on its own. Replayed cameras share one angle name, and the
frames of one frame set share its moment, so a sequence keyed by the angle alternates between
two viewpoints: every wrist speed became a jump from one camera to the other, the gaze switches
counted camera flips, and the yaw spread mixed two views of one head. A frame's camera is the
`camera` the features endpoint echoes; for older events it is the angle, numbered by its place
in the frame set when several cameras share it. How many cameras and frame sets saw a person is
counted beside the values (`_cameras`, `_frame_sets`), since frame counts double with a second
camera, and a frame set that lost or gained a frame is counted in `n_vfa_incomplete`, because
numbering by place can then give one camera's frame to another.
"""
from __future__ import annotations

import bisect
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Iterable

from openmmla.utils.constants import (
    EVENT_TYPE_ASR_RECOGNITION, EVENT_TYPE_ASR_TRANSCRIPTION, EVENT_TYPE_IPS_RELATION,
    EVENT_TYPE_IPS_ROTATION, EVENT_TYPE_IPS_TRANSLATION, EVENT_TYPE_VFA_ACTION, EVENT_TYPE_VFA_FEATURES,
)

# the file suffix Sessions -> Export Measurements gives each event type
EXPORT_SUFFIXES = {
    EVENT_TYPE_ASR_RECOGNITION: 'speaker_recognition',
    EVENT_TYPE_ASR_TRANSCRIPTION: 'speaker_transcription',
    EVENT_TYPE_VFA_ACTION: 'action_recognition',
    EVENT_TYPE_VFA_FEATURES: 'features',
    EVENT_TYPE_IPS_TRANSLATION: 'badge_translation',
    EVENT_TYPE_IPS_ROTATION: 'badge_rotation',
    EVENT_TYPE_IPS_RELATION: 'badge_relation',
}
SILENT_LABELS = {'silent', 'unknown', ''}
GAZE_CATEGORIES = ('partner_face', 'partner_hands', 'own_hands', 'zone', 'elsewhere', 'out_of_frame', 'unknown')
# a joint attention: two gazes landing within this share of the frame's width of each other
JOINT_ATTENTION_WIDTH = 0.05
# the action label of a window is the newest one up to its end, remembered for this long
ACTION_MEMORY = 60.0
ASR_BUCKET = 3.0
COCO_WRISTS = {'left': 9, 'right': 10}
# the events that are instants (a frame set, a label), not spans: each belongs to one window
INSTANT_EVENT_TYPES = {EVENT_TYPE_VFA_FEATURES, EVENT_TYPE_VFA_ACTION}


# ---- loading ----

def export_suffixes(suffix: str) -> tuple[str, ...]:
    """the file suffixes an event type was exported under: the older per-session-bucket
    exports wrote the IPS files in the plural."""
    return (suffix, suffix + 's') if suffix.startswith('badge_') else (suffix,)


def export_files(measurements_dir: str, session_id: str | None = None) -> dict[str, list[str]]:
    """the export files of the folder, by event type: `<session>_<suffix>.json` (any session
    when none is named), sorted."""
    names = sorted(os.listdir(measurements_dir))
    files: dict[str, list[str]] = {}
    for event_type, suffix in EXPORT_SUFFIXES.items():
        files[event_type] = [os.path.join(measurements_dir, name) for name in names
                             if any(name.endswith(f"_{s}.json") for s in export_suffixes(suffix))
                             and (session_id is None or name.startswith(f"{session_id}_"))]
    return files


def session_of_export(measurements_dir: str) -> str | None:
    """the session an export folder holds, from its file names (`<session>_<suffix>.json`), else
    from the folder's parent when the folder is a session's `measurements`; None when neither
    says."""
    for name in sorted(os.listdir(measurements_dir)):
        for suffix in list(EXPORT_SUFFIXES.values()) + ['parameters']:
            for s in export_suffixes(suffix):
                tail = f"_{s}.json"
                if name.endswith(tail) and len(name) > len(tail):
                    return name[:-len(tail)]
    folder = os.path.abspath(measurements_dir)
    if os.path.basename(folder) == 'measurements':
        return os.path.basename(os.path.dirname(folder)) or None
    return None


def load_events_from_influx(session_id: str, influx_client) -> dict[str, list[dict]]:
    """every event of the session, by type, sorted by window start, JSON fields parsed."""
    from openmmla.utils.querys import fetch_and_process_data
    return {event_type: fetch_and_process_data(session_id, event_type, influx_client) for event_type in EXPORT_SUFFIXES}


def load_events_from_export(measurements_dir: str, session_id: str | None = None) -> dict[str, list[dict]]:
    """every event of the session, by type, from a Sessions -> Export Measurements folder; a type
    with no file is empty."""
    from openmmla.utils.querys import deep_parse_json
    events: dict[str, list[dict]] = {}
    for event_type, paths in export_files(measurements_dir, session_id).items():
        records: list[dict] = []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                records.extend(deep_parse_json(record) for record in loaded if isinstance(record, dict))
        records.sort(key=_time)
        events[event_type] = records
    return events


def _time(record: dict) -> float:
    for key in ('window_start_time', 'segment_start_time', 'chunk_start_time'):
        value = record.get(key)
        if value not in (None, ''):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _end_time(record: dict, default_length: float) -> float:
    for key in ('window_end_time', 'chunk_end_time'):
        value = record.get(key)
        if value not in (None, ''):
            try:
                end = float(value)
                if end > _time(record):
                    return end
            except (TypeError, ValueError):
                continue
    return _time(record) + default_length


class EventIndex:
    """the records of one event type, sorted by start, found by window without a rescan: a span
    (a bucket, a chunk, an IPS second) overlaps a window, an instant (a frame set, a label)
    belongs to the one window its moment is in."""

    def __init__(self, records: list[dict], default_length: float = 1.0, instant: bool = False):
        self.instant = instant
        rows = sorted(((_time(r), _time(r) if instant else _end_time(r, default_length), r) for r in records
                       if _time(r) > 0), key=lambda row: row[0])
        self.starts = [row[0] for row in rows]
        self.ends = [row[1] for row in rows]
        self.records = [row[2] for row in rows]
        self.longest = max((end - start for start, end, _ in rows), default=0.0)

    def __len__(self) -> int:
        return len(self.records)

    def between(self, ws: float, we: float) -> list[tuple[dict, float, float]]:
        """the records in [ws, we): overlapping it (spans), or starting in it (instants)."""
        first = bisect.bisect_left(self.starts, ws - self.longest - 1e-9) if not self.instant else bisect.bisect_left(self.starts, ws)
        last = bisect.bisect_left(self.starts, we)
        found = []
        for i in range(first, last):
            start, end = self.starts[i], self.ends[i]
            if self.instant or end > ws:
                found.append((self.records[i], start, end))
        return found

    def before(self, moment: float, memory: float) -> dict | None:
        """the newest record starting before `moment` and within `memory` seconds of it."""
        i = bisect.bisect_left(self.starts, moment) - 1
        if i >= 0 and self.starts[i] >= moment - memory:
            return self.records[i]
        return None


def session_span(events: dict[str, list[dict]]) -> tuple[float, float] | None:
    """the first and last moment any event covers; None for no events."""
    starts, ends = [], []
    for event_type, records in events.items():
        for record in records:
            start = _time(record)
            if start <= 0:
                continue
            starts.append(start)
            ends.append(start if event_type in INSTANT_EVENT_TYPES
                        else _end_time(record, ASR_BUCKET if event_type.startswith('asr') else 1.0))
    if not starts:
        return None
    return min(starts), max(max(ends), min(starts) + 1e-6)


def windows(start: float, end: float, window: float, step: float) -> list[tuple[int, float, float]]:
    """(index, start, end) of every window from `start` until `end` is covered."""
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be greater than 0")
    out, index, cursor = [], 0, start
    while cursor < end:
        out.append((index, cursor, cursor + window))
        index += 1
        cursor += step
    return out


def _word_stamps(record: dict) -> list[float] | None:
    """the start of every word of a transcribed chunk, in seconds from the chunk's start, in the
    order they were said; None when the chunk carries no stamped words (no `words`, or none with a
    start). A word the aligner could not place takes the stamp of the word before it."""
    entries = record.get('words')
    if isinstance(entries, str):
        try:
            entries = json.loads(entries)
        except json.JSONDecodeError:
            return None
    if not isinstance(entries, list) or not entries:
        return None
    stamps, last = [], None
    for entry in entries:
        try:
            last = float(entry['start'])
        except (KeyError, TypeError, ValueError):
            pass
        if last is not None:
            stamps.append(last)
    return stamps or None


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _std(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _round(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _entropy(shares: dict) -> float | None:
    total = sum(v for v in shares.values() if v > 0)
    if total <= 0:
        return None
    return -sum((v / total) * math.log(v / total) for v in shares.values() if v > 0)


def _union_overlap(intervals: list[tuple[float, float]]) -> float:
    """how much time of a set of intervals is covered by more than one of them, counted once
    however many cover it: a sweep over their edges."""
    edges = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    covered_twice, depth, last = 0.0, 0, None
    for moment, change in edges:
        if last is not None and depth >= 2:
            covered_twice += moment - last
        depth += change
        last = moment
    return covered_twice


# ---- participants ----

def participants_of(events: dict[str, list[dict]]) -> list[str]:
    """the tag ids the session has: the badges IPS positioned, and the persons the features
    endpoint matched to a tag; sorted numerically where they are numbers."""
    tags: set[str] = set()
    for record in events.get(EVENT_TYPE_IPS_TRANSLATION, []):
        translations = record.get('translations') or {}
        if isinstance(translations, dict):
            tags.update(str(tag) for tag in translations)
    for record in events.get(EVENT_TYPE_VFA_FEATURES, []):
        for frame in _frames_of(record):
            for person in frame.get('persons', []):
                if person.get('tag_id') is not None:
                    tags.add(str(person['tag_id']))
    return sorted(tags, key=lambda tag: (not tag.lstrip('-').isdigit(), int(tag) if tag.lstrip('-').isdigit() else 0, tag))


def _pairs(participants: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(participants) for b in participants[i + 1:]]


def _frames_of(record: dict) -> list[dict]:
    frames = record.get('features')
    if isinstance(frames, str):
        try:
            frames = json.loads(frames)
        except json.JSONDecodeError:
            frames = []
    return frames if isinstance(frames, list) else []


# ---- speech ----

def speech_features(recognition: EventIndex, transcription: EventIndex, ws: float, we: float,
                    speakers: list[str]) -> dict:
    """what was said in the window: how much of it held speech, by whom when the speakers are
    named, the talk spurts and words that started in it, and the anonymous turns of its
    diarized chunks. A bucket several microphones reported the same speaker in counts that
    speaker once, and no bucket counts for more than the time it covers."""
    length = we - ws
    out: dict[str, Any] = {}
    buckets = recognition.between(ws, we)
    out['n_asr_recognition'] = len(buckets)
    speech, silent, by_speaker = 0.0, 0.0, defaultdict(float)
    for record, start, end in buckets:
        bucket = max(end - start, 1e-9)
        fraction = _overlap(ws, we, start, end) / bucket
        names = record.get('speakers') or []
        durations = record.get('durations') or []
        segments = record.get('segment_start_times') or []
        if isinstance(names, str):
            names = [names]
        # one entry per (speaker, segment): the copies of it from several bases fold into one
        heard: dict[tuple[str, Any], float] = {}
        for i, name in enumerate(names):
            try:
                duration = float(durations[i]) if i < len(durations) else bucket
            except (TypeError, ValueError):
                duration = bucket
            key = (str(name), segments[i] if i < len(segments) else None)
            heard[key] = max(heard.get(key, 0.0), min(duration, bucket))
        bucket_speech = min(sum(d for (name, _), d in heard.items() if name not in SILENT_LABELS), bucket)
        bucket_silent = min(sum(d for (name, _), d in heard.items() if name in SILENT_LABELS), bucket - bucket_speech)
        speech += bucket_speech * fraction
        silent += bucket_silent * fraction
        for (name, _), duration in heard.items():
            if name not in SILENT_LABELS:
                by_speaker[name] += min(duration, bucket) * fraction
    out['speech_ratio'] = _round(min(speech / length, 1.0)) if buckets else None
    out['silence_ratio'] = _round(min(silent / length, 1.0)) if buckets else None
    out['n_speakers_named'] = len(by_speaker) if buckets else None
    for name in speakers:
        out[f'spk_{name}_ratio'] = _round(min(by_speaker.get(name, 0.0) / length, 1.0)) if buckets else None

    chunks = transcription.between(ws, we)
    out['n_asr_transcription'] = len(chunks)
    started = [(record, start, end) for record, start, end in chunks if ws <= start < we]
    out['n_spurts'] = len(started)
    out['mean_spurt_seconds'] = _round(_mean(end - start for _, start, end in started))
    # a word counts in the window it was spoken in when the transcriber stamped it (word_level:
    # seconds from the chunk's start); a chunk without stamps gives its words to the window it
    # started in, and a chunk of minutes would otherwise give them all to one window
    words = 0
    for record, start, _ in chunks:
        stamps = _word_stamps(record)
        if stamps is None:
            if ws <= start < we:
                words += len(str(record.get('text') or '').split())
        else:
            words += sum(1 for stamp in stamps if ws <= start + stamp < we)
    out['words'] = words

    # the anonymous turns: labels hold within a chunk, so the counts are per chunk; the entropy
    # is averaged over the chunks, the switches and the overlap summed
    entropies, overlap_time, switches, most_speakers = [], 0.0, 0, 0
    for record, start, end in chunks:
        turns = record.get('diarization')
        if isinstance(turns, str):
            try:
                turns = json.loads(turns)
            except json.JSONDecodeError:
                turns = None
        if not isinstance(turns, list) or not turns:
            continue
        clipped, chunk_time = [], defaultdict(float)
        for turn in turns:
            try:
                t0, t1 = start + float(turn['start']), start + float(turn['end'])
            except (KeyError, TypeError, ValueError):
                continue
            inside = _overlap(ws, we, t0, t1)
            if inside > 0:
                clipped.append((max(t0, ws), min(t1, we), str(turn.get('speaker'))))
                chunk_time[str(turn.get('speaker'))] += inside
        if not clipped:
            continue
        clipped.sort()
        most_speakers = max(most_speakers, len(chunk_time))
        switches += sum(1 for (_, _, la), (_, _, lb) in zip(clipped, clipped[1:]) if la != lb)
        overlap_time += _union_overlap([(a0, a1) for a0, a1, _ in clipped])
        entropies.append(_entropy(chunk_time))
    diarized = bool(entropies)
    out['dia_speakers'] = most_speakers if diarized else None
    out['dia_switches'] = switches if diarized else None
    out['dia_overlap_ratio'] = _round(min(overlap_time / length, 1.0)) if diarized else None
    out['dia_share_entropy'] = _round(_mean(entropies)) if diarized else None
    return out


# ---- space ----

def _position(translation) -> tuple[float, float, float] | None:
    """a badge position as (x, y, z) from the synchronizer's [[x], [y], [z]] or a flat [x, y, z]."""
    try:
        values = [float(v[0]) if isinstance(v, (list, tuple)) else float(v) for v in translation]
    except (TypeError, ValueError, IndexError):
        return None
    return (values[0], values[1], values[2]) if len(values) >= 3 else None


def space_features(translations: EventIndex, relations: EventIndex, ws: float, we: float,
                   participants: list[str]) -> dict:
    """where everyone was in the window: presence and movement per person, distance per pair,
    and who faced whom (the IPS relation graph)."""
    out: dict[str, Any] = {}
    records = translations.between(ws, we)
    out['n_ips'] = len(records)
    positions: dict[str, list[tuple[float, tuple[float, float, float]]]] = defaultdict(list)
    for record, start, _ in records:
        translation = record.get('translations') or {}
        if not isinstance(translation, dict):
            continue
        for tag, value in translation.items():
            position = _position(value)
            if position is not None:
                positions[str(tag)].append((start, position))
    for tag in participants:
        seen = sorted(positions.get(tag, []))
        out[f'p{tag}_present_ratio'] = _round(len(seen) / len(records)) if records else None
        out[f'p{tag}_path_m'] = _round(sum(math.dist(a[1], b[1]) for a, b in zip(seen, seen[1:]))) if len(seen) > 1 else None
    for a, b in _pairs(participants):
        at_a = {start: position for start, position in positions.get(a, [])}
        distances = [math.dist(at_a[start], position) for start, position in positions.get(b, []) if start in at_a]
        out[f'pair{a}_{b}_dist_mean_m'] = _round(_mean(distances))
        out[f'pair{a}_{b}_dist_min_m'] = _round(min(distances)) if distances else None

    graphs = relations.between(ws, we)
    out['n_ips_relation'] = len(graphs)
    facing: dict[tuple[str, str], int] = defaultdict(int)
    mutual: dict[tuple[str, str], int] = defaultdict(int)
    both: dict[tuple[str, str], int] = defaultdict(int)
    for record, _, _ in graphs:
        graph = record.get('graph') or {}
        if not isinstance(graph, dict):
            continue
        faces = {str(tag): {str(t) for t in (targets or [])} for tag, targets in graph.items()}
        for a, b in _pairs(participants):
            if a in faces and b in faces:
                both[(a, b)] += 1
                ab, ba = b in faces[a], a in faces[b]
                facing[(a, b)] += ab
                facing[(b, a)] += ba
                mutual[(a, b)] += ab and ba
    for a, b in _pairs(participants):
        n = both[(a, b)]
        out[f'pair{a}_{b}_face_ab_ratio'] = _round(facing[(a, b)] / n) if n else None
        out[f'pair{a}_{b}_face_ba_ratio'] = _round(facing[(b, a)] / n) if n else None
        out[f'pair{a}_{b}_face_mutual_ratio'] = _round(mutual[(a, b)] / n) if n else None
    return out


# ---- body and gaze ----

def _wrists(person: dict) -> dict[str, tuple[float, float]]:
    """the wrists seen with confidence, by side."""
    keypoints = person.get('keypoints') or []
    wrists = {}
    for side, index in COCO_WRISTS.items():
        if index < len(keypoints) and len(keypoints[index]) >= 3 and float(keypoints[index][2]) >= 0.3:
            wrists[side] = (float(keypoints[index][0]), float(keypoints[index][1]))
    return wrists


def _angle(frame: dict) -> str:
    return str(frame.get('angle') or 'frame')


def _echoed_camera(frame: dict) -> str | None:
    camera = frame.get('camera')
    return None if camera in (None, '') else str(camera)


def frame_set_layout(records: Iterable[dict]) -> tuple[int | None, frozenset[str]]:
    """what the frame sets of a session hold: the modal number of frames in one (a set with
    another count lost or gained a camera; a tie goes to the larger count), and the angles some
    set repeats among the frames that do not name their camera, which several cameras share.
    Decided once for the session, so a set that lost one of those cameras still numbers its
    frame rather than making it a camera of its own."""
    counts: dict[int, int] = defaultdict(int)
    shared: set[str] = set()
    for record in records:
        frames = _frames_of(record)
        counts[len(frames)] += 1
        angles = [_angle(frame) for frame in frames if _echoed_camera(frame) is None]
        shared.update(angle for angle in set(angles) if angles.count(angle) > 1)
    modal = max(counts, key=lambda n: (counts[n], n)) if counts else None
    return modal, frozenset(shared)


def camera_keys(frames: list[dict], shared_angles: Iterable[str] = ()) -> list[str]:
    """the camera each frame of a frame set came from: the `camera` the features endpoint echoed;
    else, for an angle several cameras share, `<angle>#<k>` for its k-th frame in the set (the
    synchronizer sends them in sorted base order, so k is the same camera from set to set while
    none is missing); else the angle."""
    shared_angles = set(shared_angles)
    keys, taken = [], defaultdict(int)
    for frame in frames:
        camera, angle = _echoed_camera(frame), _angle(frame)
        if camera is not None:
            keys.append(camera)
        elif angle in shared_angles:
            keys.append(f"{angle}#{taken[angle]}")
            taken[angle] += 1
        else:
            keys.append(angle)
    return keys


def body_gaze_features(features: EventIndex, ws: float, we: float, participants: list[str],
                       layout: tuple[int | None, frozenset[str]] | None = None) -> dict:
    """what the bodies and gazes did in the window, per person and per pair, from the frames the
    features endpoint answered. Every sequence (the wrist speed, following each hand from one
    frame to the next; the gaze switches; the yaw spread) is taken within one camera, and the
    cameras are then pooled, each frame counting once in the shares and the mean yaw. Distances
    are in shares of the frame's width, so cameras compare. `layout` is the session's
    frame_set_layout, worked out from the whole index when not given."""
    out: dict[str, Any] = {}
    records = features.between(ws, we)
    out['n_vfa_features'] = len(records)
    modal, shared = layout if layout is not None else frame_set_layout(features.records)
    # camera -> person -> [(time, frame set, person dict, width)], and camera -> [(time, frame set, frame)]
    seen: dict[str, dict[str, list[tuple[float, int, dict, float]]]] = defaultdict(lambda: defaultdict(list))
    frames_by_camera: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
    angles, incomplete = set(), 0
    for number, (record, start, _) in enumerate(records):
        frames = _frames_of(record)
        incomplete += modal is not None and len(frames) != modal
        for frame, camera in zip(frames, camera_keys(frames, shared)):
            angles.add(_angle(frame))
            width = float(frame.get('width') or 0.0) or None
            frames_by_camera[camera].append((start, number, frame))
            for person in frame.get('persons', []):
                if person.get('tag_id') is None:
                    continue
                seen[camera][str(person['tag_id'])].append((start, number, person, width))
    out['n_vfa_angles'] = len(angles)
    out['n_vfa_cameras'] = len(frames_by_camera)
    out['n_vfa_incomplete'] = incomplete

    for tag in participants:
        yaws, spreads, speeds, switches, categories = [], [], [], [], []
        cameras, frame_sets = 0, set()
        for camera, persons in seen.items():
            rows = sorted(persons.get(tag, []), key=lambda row: row[0])
            if not rows:
                continue
            cameras += 1
            frame_sets.update(number for _, number, _, _ in rows)
            camera_yaws = [float(person['head_yaw']) for _, _, person, _ in rows if person.get('head_yaw') is not None]
            yaws.extend(camera_yaws)
            if len(camera_yaws) > 1:
                spreads.append((_std(camera_yaws), len(camera_yaws)))
            sequence = [((person.get('gaze') or {}).get('target') or {}).get('category') or 'unknown' for _, _, person, _ in rows]
            categories.extend(sequence)
            switches.append(sum(1 for a, b in zip(sequence, sequence[1:]) if a != b))
            # each hand against itself from one frame to the next, in frame widths per second
            for (t0, _, p0, width), (t1, _, p1, _) in zip(rows, rows[1:]):
                w0, w1 = _wrists(p0), _wrists(p1)
                moved = [math.dist(w0[side], w1[side]) for side in w0 if side in w1]
                if moved and width and t1 > t0:
                    speeds.append(_mean(moved) / width / (t1 - t0))
        out[f'p{tag}_frames'] = len(categories)
        out[f'p{tag}_frame_sets'] = len(frame_sets)
        out[f'p{tag}_cameras'] = cameras
        out[f'p{tag}_yaw_mean'] = _round(_mean(yaws), 1)
        out[f'p{tag}_yaw_abs_mean'] = _round(_mean(abs(y) for y in yaws), 1)
        # two views of one head differ by where the cameras stand, not by any turn: the spread
        # is taken per camera and averaged, weighted by the yaws each gave
        out[f'p{tag}_yaw_std'] = _round(sum(std * n for std, n in spreads) / sum(n for _, n in spreads), 1) if spreads else None
        out[f'p{tag}_wrist_speed'] = _round(_mean(speeds))
        out[f'p{tag}_gaze_switches'] = _round(_mean(switches), 2)
        for category in GAZE_CATEGORIES:
            out[f'p{tag}_gaze_{category}_ratio'] = _round(categories.count(category) / len(categories)) if categories else None

    for a, b in _pairs(participants):
        hand, gaze_dist, joint, mutual, frame_sets = [], [], [], [], set()
        for rows in frames_by_camera.values():
            for _, number, frame in rows:
                width = float(frame.get('width') or 0.0)
                persons = {str(p['tag_id']): p for p in frame.get('persons', []) if p.get('tag_id') is not None}
                if a not in persons or b not in persons:
                    continue
                frame_sets.add(number)
                pairs = frame.get('pairs') or {}
                pair = pairs.get(f'{a}|{b}') or pairs.get(f'{b}|{a}') or {}
                if width:
                    if pair.get('hand_distance') is not None:
                        hand.append(float(pair['hand_distance']) / width)
                    if pair.get('gaze_distance') is not None:
                        gaze_dist.append(float(pair['gaze_distance']) / width)
                        joint.append(1.0 if float(pair['gaze_distance']) / width <= JOINT_ATTENTION_WIDTH else 0.0)
                targets = [((persons[x].get('gaze') or {}).get('target') or {}) for x in (a, b)]
                mutual.append(1.0 if targets[0].get('category') == 'partner_face' and str(targets[0].get('person_id')) == b
                              and targets[1].get('category') == 'partner_face' and str(targets[1].get('person_id')) == a else 0.0)
        out[f'pair{a}_{b}_frames'] = len(mutual)
        out[f'pair{a}_{b}_frame_sets'] = len(frame_sets)
        out[f'pair{a}_{b}_hand_dist_min'] = _round(min(hand)) if hand else None
        out[f'pair{a}_{b}_hand_dist_mean'] = _round(_mean(hand))
        out[f'pair{a}_{b}_gaze_dist_mean'] = _round(_mean(gaze_dist))
        out[f'pair{a}_{b}_joint_attention_ratio'] = _round(_mean(joint))
        out[f'pair{a}_{b}_mutual_gaze_ratio'] = _round(_mean(mutual))
    return out


# ---- semantic ----

def action_features(actions: EventIndex, ws: float, we: float, participants: list[str]) -> dict:
    """the VLM's label per person for the window: the newest one up to the window's end,
    remembered for at most ACTION_MEMORY seconds before its start; and whether a pair
    manipulates."""
    out: dict[str, Any] = {}
    inside = actions.between(ws, we)
    out['n_vfa_action'] = len(inside)
    chosen = actions.before(we, we - ws + ACTION_MEMORY)
    labels: dict[str, str] = {}
    if chosen is not None:
        classifications = chosen.get('action_recognition') or {}
        if isinstance(classifications, dict):
            classifications = classifications.get('classifications') or {}
        if isinstance(classifications, dict):
            labels = {str(k): str(v) for k, v in classifications.items()}
    for tag in participants:
        out[f'p{tag}_action'] = labels.get(tag)
    for a, b in _pairs(participants):
        out[f'pair{a}_{b}_co_manipulating'] = (1 if labels.get(a) == 'Manipulating' and labels.get(b) == 'Manipulating' else 0) \
            if a in labels and b in labels else None
    return out


# ---- the table ----

def window_features(events: dict[str, list[dict]], window: float = 10.0, step: float = 10.0,
                    participants: list[str] | None = None, speakers: list[str] | None = None) -> list[dict]:
    """the fusion table: one row per window over the session's span."""
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be greater than 0")
    span = session_span(events)
    if span is None:
        return []
    participants = list(participants) if participants else participants_of(events)
    if speakers is None:
        named = set()
        for record in events.get(EVENT_TYPE_ASR_RECOGNITION, []):
            names = record.get('speakers') or []
            named.update(str(n) for n in (names if isinstance(names, list) else [names]) if str(n) not in SILENT_LABELS)
        speakers = sorted(named)
    recognition = EventIndex(events.get(EVENT_TYPE_ASR_RECOGNITION, []), ASR_BUCKET)
    transcription = EventIndex(events.get(EVENT_TYPE_ASR_TRANSCRIPTION, []), 1.0)
    translations = EventIndex(events.get(EVENT_TYPE_IPS_TRANSLATION, []), 1.0)
    relations = EventIndex(events.get(EVENT_TYPE_IPS_RELATION, []), 1.0)
    features = EventIndex(events.get(EVENT_TYPE_VFA_FEATURES, []), instant=True)
    layout = frame_set_layout(features.records)
    actions = EventIndex(events.get(EVENT_TYPE_VFA_ACTION, []), instant=True)
    rows = []
    for index, ws, we in windows(span[0], span[1], window, step):
        row: dict[str, Any] = {'window_index': index, 'window_start': round(ws, 3), 'window_end': round(we, 3)}
        row.update(speech_features(recognition, transcription, ws, we, speakers))
        row.update(space_features(translations, relations, ws, we, participants))
        row.update(body_gaze_features(features, ws, we, participants, layout))
        row.update(action_features(actions, ws, we, participants))
        rows.append(row)
    return rows


def write_table(rows: list[dict], path: str) -> str:
    """the rows as CSV (.csv) or JSON lines (anything else); every row gets every column."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    if path.lower().endswith('.csv'):
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: ('' if row.get(key) is None else row.get(key)) for key in columns})
    else:
        with open(path, 'w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps({key: row.get(key) for key in columns}) + '\n')
    return path
