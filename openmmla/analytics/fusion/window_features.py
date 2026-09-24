"""The fusion table of a session: one row per window (10 s by default) with, side by side, what
the ASR, IPS and VFA pipelines saw in it, as plain numbers with a coverage count per modality.

Speech comes from `asr_recognition` (who-or-whether someone spoke, 3 s buckets) and
`asr_transcription` (talk spurts, words, and the anonymous speaker turns of a diarized chunk);
space from `ips_translation` and `ips_relation` (positions, facing, movement, presence); body
and gaze from `vfa_features` (head yaw, wrist speed, where gazes land, joint attention, hand
distance); the semantic layer from `vfa_action` (the VLM's labels). Every person is a tag id,
every pair a sorted tag pair; persons the pose model saw without a tag count only in the seat
trace.

A tag names more frames than the ones it was read in. The features endpoint tracks every person
per camera and keeps a read tag on the track from then on; the fusion carries it further, over
the whole session: the frames of a track before its first read, and those after the server lost
its memory of it, take the tag of the track's nearest read (propagate_track_tags). Such a person
counts like a tagged one everywhere but in learning the seats, and `n_vfa_propagated` says how
many of the window's person frames were named that way; a session whose persons were not tracked
has no such column.

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

The seat trace keeps what the untagged bodies say about presence. A participant's seat on a
camera is the median centre of their own boxes there (SEAT_MIN_BOXES at least; the boxes the
server tagged, read or kept on the track, never the propagated ones), and a body without a tag is
at the seat within half their median box width. `p<tag>_untagged_at_seat_ratio` is the share of
the window's frame sets (all of them, so a camera that dropped out lowers it) in which such a body
stood at the seat while the tag was not seen on that camera (someone there whose tag was not
read), empty when no camera holding the seat gave a frame; `n_untagged_at_seats` counts those
bodies per frame set. A table of a session without VFA has neither.

A session with personal microphones (transcripts that carry a `participant`) counts each
wearer's words only in the 3 s buckets the synchronizer's energy vote gave them
(`p<tag>_words`). When the worn microphones left level traces (`levels` in the buckets and their
transcripts, since 2026-09-24) each word is decided on its own instead, from the levels of every
worn microphone while it was said (attribution.attribute_word): it counts for the wearer when their
microphone led every other by the margin, and not when another wearer's microphone led (cross-talk)
or none did (the teacher, or wearers talking over each other), and a word two microphones
transcribed counts once (attribution.count_once). The bucket vote's counts stay beside it in
`p<tag>_vote_words` (and `vote_words`, their sum, without a group microphone), which the layout
drops. Without a group microphone, `words` is then every word the worn microphones transcribed,
each spoken word once whoever said it (attribution.once_across), as a group microphone counts every
word it hears. Its spurts, words and turns stay the group microphone's, and so, since
2026-09-24, do its speech_ratio, silence_ratio and n_speakers_named: they read only the group
microphone's entries of each bucket, as in a session with the group microphone alone (a bucket of
wearers' entries is the group's silence only while the group microphone was naming speech around
it, and a group microphone that names speech after a wearer leaves the speech pooled). A session
without them gives the same table as before.

In-group gaze. A partner is another pupil of the session: `pupils`, the command's -tags, else the
pupils the session's manifest declares, else the participants whose tags are at most MAX_PUPIL_TAG
(the IPS trust bound). A gaze the server put on the face or hands of anyone else (an untagged body,
track_<n> or unknown_<n>, or a tag outside the pupils: the teacher, another group, a misread
badge) is `other_face` or `other_hands`, except an untagged body that stands at the seat (on that
camera) of a pupil whose tag the frame does not hold: that is almost certainly the pupil with an
unread badge, and a partner (`n_vfa_seat_partners` counts those gaze frames). The partner columns of
a table fused before 2026-09-23 counted every other person: they equal partner_* + other_* of this
table. `p<tag>_in_group` says whether the tag is one of the pupils.

The work area. Before any window is cut, every camera learns where the pupils' hands have been, up
to and including each frame (openmmla.services.vfa.work_area), and a gaze the server called
`elsewhere` that lands in that area is `work_area` (label_work_areas). Faces, hands, zones,
out_of_frame and unknown keep priority, the area learns from the tags the server gave (never the
propagated ones), and a frame the server already labelled (it carries `work_area`) is kept as it
is. `p<tag>_work_area_ready_ratio` is the share of the person's gaze frames taken on a camera whose
area was ready. The gaze switches count changes between the labels, so a gaze moving from the
work area to beyond it is a switch, as is one moving from a partner to the teacher.

Joint attention against its own past. Two gazes close together are joint attention, but pupils
who sit close look at the same table a lot: `pair<a>_<b>_joint_attention_baseline` is how often
a's gaze in the window met b's gaze on the same camera 20, 30 and 40 s earlier (and b's met a's),
the pair's own rate of meeting by seating and task, past the decay of a joint episode;
`_joint_attention_excess` is the window's joint attention above it. The baseline is causal in
time only: it compares gaze points from before the window, but who a gaze point belongs to, like
the partner/other split and every tagged column, comes from the offline naming over the whole
session (a track takes the tag of its nearest read, which can come later, and the seats are learned
from the whole session). A later badge read can therefore change an earlier window's values, and a
server with only forward track memory would not reproduce them exactly.

The hand circle, remade. The features endpoint scores a gaze against a circle around each hand and
measures a pair's hands between the circles' centres (features.hand_regions). Version 2 of the
circle (2026-09-24) sits 0.33 shoulder widths past the wrist, where the hand is, rather than 0.14;
every stored event was made with version 1 (its frames say nothing of the circle; a server with
version 2 says so in each frame's `scoring`). So before anything else the fusion makes every frame's
gaze targets and pair hand distances again from what the frame stores (relabel_hand_circles;
`hand_relabel=False` keeps them as stored, and the work area and hand columns then read the circle
the frames were made with, table_hand_circle). With the version 1 circle the remake gives back
99.975 % of the stored targets of the 20 replayed sessions and 99.92 % of their hand distances
within 0.5 px; every other one is reproduced by moving the stored numbers within their rounding.
With version 2, 6.8 % of the targets change, most of them from `elsewhere` to a person's own or a
partner's hands.

The hands in body units (2026-09-24). Each pupil's wrists are followed in the image, in their own
shoulder widths, over 1 s steps on one camera (wrist_moves, hand_status): whether the hands were
active or still (a lean over resting hands leaves them still, hands carried along with the body
move), the wrist speed in shoulder widths, the hands of a body outside the group near theirs, and
per pair the steps with one active and the other still, and how often the still one's gaze was on
the active one's hands (body_gaze_features).
The wrist speed and hand distances in frame widths stay in the table (the classifier's
duplicate-skeleton gate reads one), but the camera's distance and field of view move them (the pixels
cancel out: a 540p and a 1080p camera from the same place give the same values). These are
2D positions at one frame set a second: hands close together are not hands touching, and a handover
of a second or less is seen once or not at all.
"""
from __future__ import annotations

import bisect
import csv
import json
import math
import os
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from openmmla.bases.asr.attribution import ENERGY_MARGIN_DB, WORD_WEARER, as_levels, attribute_word, count_once, once_across, power_db, span_powers, word_lead
from openmmla.services.vfa import features as vfa_features
from openmmla.services.vfa.work_area import WorkArea, apply_work_area
from openmmla.utils.asr_scope import participant_of
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
# where a gaze lands, as the table counts it: partner_* is another pupil, other_* anyone else,
# work_area an `elsewhere` inside the camera's work area
GAZE_CATEGORIES = ('partner_face', 'partner_hands', 'other_face', 'other_hands', 'own_hands', 'zone',
                   'work_area', 'elsewhere', 'out_of_frame', 'unknown')
# the IPS trust bound (layout.MAX_TAG): a higher tag is a mis-decoded badge, never a pupil
MAX_PUPIL_TAG = 12
# a joint attention: two gazes landing within this share of the frame's width of each other
JOINT_ATTENTION_WIDTH = 0.05
# the joint-attention baseline compares a pupil's gaze with the partner's this many seconds
# earlier: past the decay of a joint episode (by 20 s the rate has made most of its drop to the
# 40 s level), still within the same seating and task phase
JOINT_BASELINE_LAGS = (20.0, 30.0, 40.0)
# the partner's frame set nearest t - d must lie within this many seconds of it
JOINT_BASELINE_SLACK = 0.5
# fewer lag comparisons than this give no baseline
JOINT_BASELINE_MIN = 10
# the action label of a window is the newest one up to its end, remembered for this long
ACTION_MEMORY = 60.0
ASR_BUCKET = 3.0
# in a session with a group microphone beside worn ones, a bucket holding only wearers' entries is read
# as the group microphone's silence only when the group microphone named speech within this many seconds
# before it and within this many after it. The merged bucket does not say which base reported, and a
# group base that stopped, started late or dropped out leaves the worn ones reporting alone. On the
# replayed sessions with both, every such bucket (72) had the group's speech within 15 s on both sides,
# and the group's speech buckets were at most 18 s apart; this is twice the 15 s
GROUP_SILENCE_REACH = 30.0
COCO_WRISTS = {'left': 9, 'right': 10}
# the events that are instants (a frame set, a label), not spans: each belongs to one window
INSTANT_EVENT_TYPES = {EVENT_TYPE_VFA_FEATURES, EVENT_TYPE_VFA_ACTION}
# a tag's seat on a camera is the median centre of its own boxes there; a body is at the seat
# within this many of those boxes' median widths of it
SEAT_RADIUS_WIDTHS = 0.5
# a camera learns a tag's seat only from at least this many of the tag's boxes (10 s at the bases'
# one frame set a second), so a tag misread for a few frames makes no seat
SEAT_MIN_BOXES = 10
# a track id seen again after this long is another track: ByteTrack keeps a lost track for 30
# frames (30 s at the bases' one frame set a second; no track of the 21 replayed sessions was
# away longer than 32 s), so a longer absence means the server restarted or forgot the camera and
# handed the id out again
TRACK_GAP_SECONDS = 60.0
# the tag_match of a person whose tag the fusion carried along their track
PROPAGATED = 'propagated'
# the features endpoint's keypoint_confidence and inout_threshold, which every replay used (the
# server's defaults): the hand relabel must score as the server did, and reads them from a frame that
# states them (its `scoring`, every answer from version 2 of the hand circle on); these stand in for
# a frame that does not
VFA_KEYPOINT_CONFIDENCE = 0.3
VFA_INOUT_THRESHOLD = 0.5
# the wrists and shoulders the body-normalised hand columns read: at this confidence at least one
# wrist is seen in 94-98 % of the pupils' frames and both shoulders in 94-96 %, by setup (the 20
# replayed sessions)
BODY_CONFIDENCE = 0.5
# a step is two frame sets in a row, this far apart: the bounds below are displacements in a 1 s step
# (the frame sets are 1.0 s apart in every replay, 56,270 gaps), so a session at another cadence
# (keyframe_interval 0.5) has no hand steps rather than steps its bounds were not measured for
MIN_STEP_SECONDS = 0.75
MAX_STEP_SECONDS = 1.5
# a step is still when every wrist seen moved less than HAND_STILL_SW shoulder widths in the image, and
# active when one moved at least HAND_ACTIVE_SW; between the two it is neither. Measured on the pupils'
# steps of the 20 replayed sessions as the table takes them (the tags carried along the tracks), no
# label read: a wrist whose arm stayed still in the image in that step (its own elbow and shoulder each
# moved under 0.05 shoulder widths, nothing subtracted) moved a median 0.024-0.027, p90 0.078-0.084, p95
# 0.109-0.118 and p99 0.22-0.24 shoulder widths in a 1 s step, by setup (100,487 wrists). Still is under
# that p90 and active is twice it, the logic of the torso frame's bounds (one step). That p90 is a
# convention of the definition, not the keypoints' noise alone. It follows the arm's bound (0.059 under
# 0.03, 0.110 under 0.08), since a looser arm lets real movement in, and one still step takes in arms
# that are just stopping or about to move: those whose previous or next step moved give a p90 of
# 0.089-0.098, while arms still on the previous and next step as well give 0.055-0.061 (44,224 wrists;
# 0.08 is about their p95, 0.16 above their p99 of 0.13-0.16). Bounds of 0.06 and 0.12 would read 8
# points more steps active (46.5, 43.6 and 54.0 % against 38.4, 35.9 and 45.9 % at 540p micro:bit,
# 1080p micro:bit and 1080p microscope) and 8 to 9 fewer still; two cameras agree a little less there
# (three-way kappa 0.24, 0.33 and 0.30 against 0.26, 0.36 and 0.33), so reliability does not settle it,
# and the definition fixed first is kept. The wrist is not in the definition, so its own spread is not
# cut (a bound under 0.30 on it moves the p90 by 0.002, while one under 0.10 would pin its p99 at
# 0.097). A step takes the largest of its wrists' moves, so a step whose both arms stayed still reads
# still in 85-90 % of cases and active in 1.8-2.7 %, by setup. In hand lengths, still is under 0.17
# and active from 0.35.
# Over every pupil step, 36-46 % read active, 35-45 % still and 19-20 % between, by setup. The same
# bounds held against the shoulder midpoint (the frame until 2026-09-24's review, where the arm-still
# p90 was 0.077-0.081); in the image a lean over resting hands no longer counts: of the steps active
# against the shoulders, 2.9-4.3 % are still in the image and 11-12 % between
HAND_STILL_SW = 0.08
HAND_ACTIVE_SW = 0.16
# a hand's length (wrist to the farthest fingertip) in shoulder widths: the whole-body hand model's
# confident hands on 128 replayed frames (368 hands, mean score >= 0.7, plausible) are a median 0.46
# shoulder widths long (IQR 0.37-0.55; 0.48, 0.47 and 0.40 at 540p micro:bit, 1080p micro:bit and
# 1080p microscope)
HAND_LENGTH_SW = 0.46
# a body whose box overlaps a pupil's this much (IoU) is a second skeleton of that pupil, no one else
DUPLICATE_BODY_IOU = 0.5
# a body whose box lies this much inside a pupil's (the share of its own area) is part of that pupil:
# an arm or a head the pose model saw apart. Of the bodies whose hands the first rule (the IoU alone)
# counted near a pupil's on the 20 replayed sessions, 7 %, 13 % and 24 % lay so (540p micro:bit,
# 1080p micro:bit, 1080p microscope)
CONTAINED_BODY_SHARE = 0.6
# a body with a hand circle this close to one of a pupil's (its centre within this many of the pupil's
# shoulder widths, a fifth of a hand's length) shares the pupil's wrist: a second skeleton of the
# pupil; 9-15 % of those bodies had one, by setup
DUPLICATE_HAND_SW = 0.1
# a pair's follow_ratio needs at least this many steps with one active and the other still: a share of
# one or two steps can only be 0, 0.5 or 1, and 37-48 % of the windows with such a step had no more
# than two, by setup
FOLLOW_MIN_STEPS = 3


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
    return sorted(tags, key=_tag_key)


def default_pupils(participants: Iterable[str]) -> list[str]:
    """the pupils of a session nobody declared: the participants whose tags are numbers no higher
    than MAX_PUPIL_TAG (a higher one is a mis-decoded badge)."""
    return [str(tag) for tag in participants if str(tag).isdigit() and int(str(tag)) <= MAX_PUPIL_TAG]


def _tag_key(tag: str) -> tuple:
    """tags sorted numerically where they are numbers, the others after them by name."""
    return (not tag.lstrip('-').isdigit(), int(tag) if tag.lstrip('-').isdigit() else 0, tag)


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

def _speaker_names(record: dict) -> list[str]:
    """a record's `speakers` as a list of text: one name is a list of it, none is empty."""
    names = record.get('speakers')
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return [str(names)]


def _count_words(chunks: list[tuple[dict, float, float]], ws: float, we: float) -> int:
    """the words of the chunks said in [ws, we)."""
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
    return words


def _json_field(record: dict, name: str):
    value = record.get(name)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _voice_stream(record: dict) -> str:
    """the microphone and registry whose session voices a linked chunk's are: a wearer's, else the
    chunk's speaker (the group id of a group-level base), and the base's voice registry
    (`voice_registry`: a base launched again into the session numbers its voices anew)."""
    tag = participant_of(record.get('participant'))
    stream = f'participant {tag}' if tag is not None else f'speaker {record.get("speaker")}'
    return f'{stream} {record.get("voice_registry") or ""}'.rstrip()


def _diarization_features(chunks: list[tuple[dict, float, float]], ws: float, we: float, length: float) -> dict:
    """the anonymous turns of the diarized chunks in the window (dia_*).

    A chunk whose speakers the base linked into the voices of its session (a `voices` field, each
    turn its `voice`) is read by voice, together with the other linked chunks of its microphone and
    voice registry: the speakers, switches, overlap and shares are then the window's, across its
    chunks. A speaker the base left without a voice (no usable embedding, or too little speech to
    start a voice) is a speaker of its own chunk only, as every speaker of a chunk from before the
    linking is. The speakers are the most any one reading has, the switches and the overlap are
    summed, the entropy averaged."""
    out: dict[str, Any] = {}
    readings: list[list[tuple[float, float, str]]] = []
    linked: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for index, (record, start, end) in enumerate(chunks):
        turns = _json_field(record, 'diarization')
        if not isinstance(turns, list) or not turns:
            continue
        voices = isinstance(_json_field(record, 'voices'), dict)
        clipped = []
        for turn in turns:
            try:
                t0, t1 = start + float(turn['start']), start + float(turn['end'])
            except (KeyError, TypeError, ValueError):
                continue
            if _overlap(ws, we, t0, t1) > 0:
                if voices and turn.get('voice') is not None:
                    label = f'voice {turn["voice"]}'
                else:
                    label = f'chunk {index} {turn.get("speaker")}'  # a label holds within its chunk only
                clipped.append((max(t0, ws), min(t1, we), label))
        if not clipped:
            continue
        if voices:
            linked[_voice_stream(record)].extend(clipped)
        else:
            readings.append(clipped)
    readings.extend(linked.values())
    entropies, overlap_time, switches, most_speakers = [], 0.0, 0, 0
    for clipped in readings:
        clipped.sort()
        time = defaultdict(float)
        for a0, a1, label in clipped:
            time[label] += a1 - a0
        most_speakers = max(most_speakers, len(time))
        switches += sum(1 for (_, _, la), (_, _, lb) in zip(clipped, clipped[1:]) if la != lb)
        overlap_time += _union_overlap([(a0, a1) for a0, a1, _ in clipped])
        entropies.append(_entropy(time))
    diarized = bool(entropies)
    out['dia_speakers'] = most_speakers if diarized else None
    out['dia_switches'] = switches if diarized else None
    out['dia_overlap_ratio'] = _round(min(overlap_time / length, 1.0)) if diarized else None
    out['dia_share_entropy'] = _round(_mean(entropies)) if diarized else None
    return out


class LevelTraces:
    """one worn microphone's level traces (attribution.levels_record, each from its start), found by
    moment: the synchronizer's buckets hold one for each of its segments, speech or silence, its
    transcripts one for each of its chunks; a moment is read from a bucket's, and from a
    transcript's where no bucket's holds it (a base run without a synchronizer, a bucket the
    synchronizer lost)."""

    def __init__(self):
        self._sources: dict[str, list[tuple[float, dict]]] = {'bucket': [], 'chunk': []}
        self._index: dict[str, tuple[list[float], list[tuple[float, dict]], float]] | None = None

    def add(self, source: str, start: float, levels: dict):
        self._sources[source].append((float(start), levels))
        self._index = None

    def __bool__(self) -> bool:
        return any(self._sources.values())

    def has(self, source: str) -> bool:
        """whether any trace came from `source` ('bucket' or 'chunk')."""
        return bool(self._sources.get(source))

    def at(self, moment: float) -> tuple[float, dict] | None:
        """(start, levels) of the trace holding `moment`, a bucket's before a transcript's; None
        when none does."""
        if self._index is None:
            self._index = {}
            for source, held in self._sources.items():
                held = sorted(held, key=lambda pair: pair[0])
                longest = max((len(levels['db']) * levels['hop'] for _, levels in held), default=0.0)
                self._index[source] = ([start for start, _ in held], held, longest)
        for source in ('bucket', 'chunk'):
            starts, held, longest = self._index[source]
            i = bisect_right(starts, moment) - 1
            while i >= 0 and starts[i] >= moment - longest:
                start, levels = held[i]
                if moment < start + len(levels['db']) * levels['hop']:
                    return start, levels
                i -= 1
        return None


@dataclass
class PersonalSpeech:
    """the personal microphones of a session: who wore one, the buckets each won, whether a
    group microphone transcribed too, when that one named speech, and the level traces of the worn
    microphones (from the buckets and their transcripts), which decide their words one by one."""
    participants: list[str]
    won: dict[str, tuple[list[float], list[float]]]  # per participant: sorted bucket starts, their ends
    has_group: bool
    # the sorted starts of the buckets holding a group microphone's speech (an entry neither silent nor
    # named after a wearer); None without a group microphone, or when its speech is named after a
    # wearer too, so that its entries cannot be told from the worn ones'
    group_speech: list[float] | None = None
    # per participant: the level traces of their microphone, from the buckets and their transcripts
    traces: dict[str, LevelTraces] | None = None
    # whether the words are decided one by one: every wearer who left transcripts left level traces,
    # and the buckets carry them wherever a synchronizer voted (a session transcribed before 2026-09-24
    # has none, and one whose synchronizer kept no levels keeps the bucket vote)
    by_word: bool = False
    margin_db: float = ENERGY_MARGIN_DB
    # where the levels that decide the words come from: 'buckets' (the transcripts' fill the gaps) or
    # 'transcripts' (a base run without a synchronizer); None when the words are not decided one by one
    levels_from: str | None = None

    def __post_init__(self):
        self._classes: dict[int, list[str] | None] = {}
        self._counted: dict[int, list[bool]] = {}  # per worn transcript: its words that count once across microphones
        # every microphone with a level: a worn one that never passed its speech gate still leads
        self._level_tags = sorted(set(self.participants) | set(self.traces or {}), key=_tag_key)

    def won_at(self, participant: str, moment: float) -> bool:
        """whether `moment` falls in a bucket the participant won."""
        starts, ends = self.won.get(participant, ([], []))
        i = bisect_right(starts, moment) - 1
        return i >= 0 and moment < ends[i]

    def snr(self, participant: str, first: float, last: float) -> float | None:
        """how far the participant's microphone stood over its floor from `first` to `last` (session
        time): the power mean of the steps of theirs the span covers, from the trace that holds
        `first` (LevelTraces.at) and from the ones that follow it where the span runs past its end,
        less the floor of the first; None when no trace holds `first` (attribute_word then counts the
        microphone as at its floor)."""
        traces = (self.traces or {}).get(participant)
        if not traces:
            return None
        powers, floor, moment = [], None, first
        for _ in range(64):  # a word covers a few traces at most
            found = traces.at(moment)
            if found is None:
                break
            start, levels = found
            floor = levels['floor_db'] if floor is None else floor
            until = start + len(levels['db']) * levels['hop']
            powers += span_powers(levels['db'], levels['hop'], moment - start, last - start)
            if last <= until or until <= moment:
                break
            moment = until + 1e-6  # the next trace of this microphone, a bucket's or a transcript's
        level = power_db(powers)
        return None if level is None else level - floor

    def _judge(self, record: dict, start: float, end: float) -> list[tuple[float, float, str, str, float]]:
        """(start, end, wearer, class, lead) of each word of a worn microphone's transcript, in the
        order of word_times: attribute_word and word_lead over the levels of every worn microphone
        while it was said."""
        wearer = participant_of(record.get('participant'))
        judged = []
        for first, last in word_spans(record, start, end):
            t0, t1 = start + first, start + last
            snrs = {tag: self.snr(tag, t0, t1) for tag in self._level_tags}
            judged.append((t0, t1, wearer, attribute_word(wearer, snrs, self.margin_db), word_lead(wearer, snrs)))
        return judged

    def decide_words(self, transcriptions: list[dict]):
        """decides every word of the session's worn transcripts at once (by_word), so that a word two
        worn microphones transcribed counts for one wearer at most (count_once), and once among all
        the words spoken (once_across)."""
        if not self.by_word:
            return
        judged, owners = [], []
        for record in transcriptions:
            if participant_of(record.get('participant')) is None:
                continue
            start = _time(record)
            if start <= 0:
                continue
            words = self._judge(record, start, _end_time(record, 1.0))  # as window_features indexes a transcript
            owners.append((id(record), len(judged), len(words)))
            judged.extend(words)
        classes = count_once(judged)
        counted = once_across(judged, classes)
        for key, first, n in owners:
            self._classes[key] = classes[first:first + n]
            self._counted[key] = counted[first:first + n]

    def words_counted(self, record: dict) -> list[bool] | None:
        """which words of a worn microphone's transcript count when every spoken word counts once,
        whichever microphones transcribed it (decide_words, once_across), in the order of word_times;
        None when the words are not decided one by one."""
        return self._counted.get(id(record)) if self.by_word else None

    def word_classes(self, record: dict, start: float, end: float) -> list[str] | None:
        """whose each word of a worn microphone's transcript is (attribution.attribute_word), in the
        order of word_times, from the levels of every worn microphone while it was said, a word two
        microphones transcribed counted once (decide_words); None when the words are not decided one
        by one (by_word)."""
        key = id(record)
        if key not in self._classes:
            classes = None
            if self.by_word and participant_of(record.get('participant')) is not None:
                classes = [judged[3] for judged in self._judge(record, start, end)]
            self._classes[key] = classes
        return self._classes[key]

    def group_near(self, moment: float) -> bool:
        """whether the group microphone named speech within GROUP_SILENCE_REACH seconds before
        `moment` and within that after it: it was reporting then."""
        starts = self.group_speech or []
        i = bisect_left(starts, moment)
        if i < len(starts) and starts[i] == moment:
            return True
        return (i > 0 and moment - starts[i - 1] <= GROUP_SILENCE_REACH
                and i < len(starts) and starts[i] - moment <= GROUP_SILENCE_REACH)


def _energies_of(record: dict) -> dict | None:
    """a merged bucket's energies ({participant: snr}), None when it carries none."""
    energies = record.get('energies')
    if isinstance(energies, str):
        try:
            energies = json.loads(energies)
        except json.JSONDecodeError:
            return None
    return energies if isinstance(energies, dict) else None


def won_buckets(recognitions: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """for each participant, the [start, end) of the merged buckets that list them among the
    speakers and, when the bucket carries energies, among those too."""
    won: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in recognitions:
        try:
            start = float(record.get('window_start_time'))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start):
            continue
        try:
            end = float(record.get('window_end_time'))
        except (TypeError, ValueError):
            end = start + ASR_BUCKET
        if not end > start:
            end = start + ASR_BUCKET
        energies = _energies_of(record)
        voted = {str(tag) for tag in energies} if energies is not None else None
        for name in set(_speaker_names(record)):
            if name in SILENT_LABELS:
                continue
            if voted is not None and name not in voted:
                continue
            won[name].append((start, end))
    return {name: sorted(spans) for name, spans in won.items()}


def word_times(record: dict, start: float, end: float) -> list[float]:
    """when each word of a chunk was said, in seconds from the chunk's start: its stamp when the
    transcriber gave one, else the words spread evenly over the chunk."""
    stamps = _word_stamps(record)
    if stamps is not None:
        return stamps
    n = len(str(record.get('text') or '').split())
    return [(end - start) * i / n for i in range(n)]


# a word the aligner gave a start and no end lasts this long
WORD_SECONDS = 0.3


def word_spans(record: dict, start: float, end: float) -> list[tuple[float, float]]:
    """from when to when each word of word_times was said, in seconds from the chunk's start: its
    start and end when the transcriber gave them (WORD_SECONDS when it gave no end), the span of the
    word before it for a word the aligner could not place; the even share of the chunk each word of
    an unstamped chunk has."""
    stamps = _word_stamps(record)
    if stamps is None:
        n = len(str(record.get('text') or '').split())
        share = (end - start) / n if n else 0.0
        return [(share * i, share * (i + 1)) for i in range(n)]
    entries = _json_field(record, 'words')
    spans, last = [], None
    for entry in entries if isinstance(entries, list) else []:
        try:
            first = float(entry['start'])
        except (KeyError, TypeError, ValueError):
            if last is not None:
                spans.append(last)
            continue
        try:
            until = float(entry['end'])
        except (KeyError, TypeError, ValueError):
            until = first + WORD_SECONDS
        last = (first, until if until > first else first + WORD_SECONDS)
        spans.append(last)
    return spans


def personal_speech(recognitions: list[dict], transcriptions: list[dict]) -> PersonalSpeech | None:
    """the session's personal microphones, None when no transcript carries a participant (every
    older session); with level traces, every worn word is decided here (decide_words)."""
    tags = {participant_of(t.get('participant')) for t in transcriptions} - {None}
    if not tags:
        return None
    spans = won_buckets(recognitions)
    voted: set[str] = set()
    for record in recognitions:
        energies = _energies_of(record)
        if energies:
            voted.update(str(tag) for tag in energies)
    participants = sorted(tags | voted, key=_tag_key)
    won = {tag: ([s for s, _ in spans.get(tag, [])], [e for _, e in spans.get(tag, [])]) for tag in participants}
    has_group = any(participant_of(t.get('participant')) is None for t in transcriptions)
    # the level traces of the worn microphones: every segment's in the buckets, every chunk's in the transcripts
    traces: dict[str, LevelTraces] = defaultdict(LevelTraces)
    for record in recognitions:
        held = _json_field(record, 'levels')
        for tag, value in (held.items() if isinstance(held, dict) else ()):
            levels, start = as_levels(value), value.get('start') if isinstance(value, dict) else None
            try:
                start = float(start) if start is not None else _time(record)
            except (TypeError, ValueError):
                continue
            if levels is not None and start > 0 and math.isfinite(start):
                traces[str(tag)].add('bucket', start, levels)
    for record in transcriptions:
        tag, levels = participant_of(record.get('participant')), as_levels(record.get('levels'))
        start = _time(record)
        if tag is not None and levels is not None and start > 0:
            traces[tag].add('chunk', start, levels)
    traces = {tag: held for tag, held in traces.items() if held}
    # the names the group microphone gave its own chunks: its entries are told from the worn ones'
    # by name, so a group microphone that names speech after a wearer (speaker verification with
    # profiles named after the tags) cannot be read apart, and its session keeps the pooled speech
    group_names = {str(t.get('speaker')) for t in transcriptions
                   if participant_of(t.get('participant')) is None and t.get('speaker') is not None} - SILENT_LABELS
    group_speech = None
    if has_group and not group_names & set(participants):
        group_speech = sorted({_time(record) for record in recognitions
                               if any(name not in SILENT_LABELS and name not in participants
                                      for name in _speaker_names(record))})
    # the words are decided one by one when every wearer left level traces; where a synchronizer voted
    # (energies) its buckets must carry them too, since a transcript's trace covers only its own speech,
    # so that another microphone would stand at its floor between its chunks
    bucket_levels = any(held.has('bucket') for held in traces.values())
    by_word = bool(traces) and tags <= set(traces) and (bucket_levels or not voted)
    personal = PersonalSpeech(participants=participants, won=won, has_group=has_group, group_speech=group_speech,
                              traces=traces or None, by_word=by_word,
                              levels_from=('buckets' if bucket_levels else 'transcripts') if by_word else None)
    personal.decide_words(transcriptions)
    return personal


def _heard(entries: list[tuple[str, float, Any]], bucket: float) -> dict[tuple[str, Any], float]:
    """a bucket's (name, duration, segment start) entries as {(name, segment start): seconds}: the
    copies of one speaker's segment from several bases fold into the longest, none longer than the
    bucket."""
    heard: dict[tuple[str, Any], float] = {}
    for name, duration, segment in entries:
        key = (name, segment)
        heard[key] = max(heard.get(key, 0.0), min(duration, bucket))
    return heard


def group_speech_entries(entries: list[tuple[str, float, Any]], wearers: set[str], bucket: float,
                         group_near: bool = True) -> list[tuple[str, float, Any]] | None:
    """the group microphone's entries of a merged bucket (name, duration, segment start), in a
    session where it ran beside worn ones; None when the bucket cannot say. The bucket does not say
    which base an entry came from, but a worn microphone names its speech with its wearer's tag
    (`wearers`, personal_speech's participants) and the group microphone with the group's id
    (`group_01` or `group_02` in the replays), so the group's are the entries not named after a
    wearer (personal_speech checks the group's own chunks are not named after one). The synchronizer
    keeps a silent entry only when no microphone heard speech, so a bucket with a wearer's speech and
    none of the group's is one the group microphone called silent, if it reported that bucket at all:
    it becomes one silent entry of the bucket, as the group microphone's own report was (all 3,895
    silent entries of the 20 replays cover their 3 s bucket), when the group microphone named speech
    near it (`group_near`, PersonalSpeech.group_near), and None otherwise, since a group base that
    stopped or dropped out leaves the worn ones reporting alone. On the four replays with worn
    microphones the group base logged every bucket, and said silent in each of the 72 buckets
    holding only wearers' entries. A bucket without entries stays without."""
    kept = [entry for entry in entries if entry[0] not in wearers]
    if kept or not entries:
        return kept
    return [('silent', bucket, None)] if group_near else None


def speech_features(recognition: EventIndex, transcription: EventIndex, ws: float, we: float,
                    speakers: list[str], personal: PersonalSpeech | None = None) -> dict:
    """what was said in the window: how much of it held speech, by whom when the speakers are
    named, the talk spurts and words that started in it, and the anonymous turns of its
    diarized chunks. A bucket several microphones reported the same speaker in counts that
    speaker once, and no bucket counts for more than the time it covers.

    With personal microphones (`personal`), each wearer's words count only in the buckets they
    won (`p<tag>_words`), or, when the worn microphones left level traces (personal.by_word), only
    the words their microphone led on (the bucket vote's count then stays in `p<tag>_vote_words`),
    and the spurts, words and turns are the group microphone's. Without one the spurts and turns
    come from every chunk, and `words` is, with level traces, every word the worn microphones
    transcribed, each spoken word once whoever said it (PersonalSpeech.words_counted), as a group
    microphone counts every word it hears (the vote's sum stays in `vote_words`); before level
    traces it is the wearers' sum.

    With a group microphone beside the worn ones whose entries can be told apart
    (`personal.group_speech`), speech_ratio, silence_ratio and n_speakers_named read only its entries
    of each bucket (group_speech_entries), so the window's speech is what the room microphone heard,
    as in a session that had only that one; a bucket of wearers' entries far from the group's speech
    (GROUP_SILENCE_REACH) counts neither speech nor silence, as a missing bucket, and a window with no
    bucket left to read has none of the three. spk_<name>_ratio reads one name's entries in every
    session, a wearer's its own microphone's."""
    length = we - ws
    out: dict[str, Any] = {}
    buckets = recognition.between(ws, we)
    out['n_asr_recognition'] = len(buckets)
    wearers = set(personal.participants) if personal is not None and personal.group_speech is not None else None
    speech, silent, by_speaker, named, read = 0.0, 0.0, defaultdict(float), set(), 0
    for record, start, end in buckets:
        bucket = max(end - start, 1e-9)
        fraction = _overlap(ws, we, start, end) / bucket
        names = record.get('speakers') or []
        durations = record.get('durations') or []
        segments = record.get('segment_start_times') or []
        if isinstance(names, str):
            names = [names]
        entries = []
        for i, name in enumerate(names):
            try:
                duration = float(durations[i]) if i < len(durations) else bucket
            except (TypeError, ValueError):
                duration = bucket
            entries.append((str(name), duration, segments[i] if i < len(segments) else None))
        # one entry per (speaker, segment): the copies of it from several bases fold into one
        heard = _heard(entries, bucket)
        for (name, _), duration in heard.items():
            if name not in SILENT_LABELS:
                by_speaker[name] += min(duration, bucket) * fraction
        # the speech and silence of the bucket, from the group microphone alone when there is one
        if wearers is None:
            counted = heard
        else:
            group = group_speech_entries(entries, wearers, bucket, personal.group_near(start))
            if group is None:
                continue
            counted = _heard(group, bucket)
        read += 1
        bucket_speech = min(sum(d for (name, _), d in counted.items() if name not in SILENT_LABELS), bucket)
        bucket_silent = min(sum(d for (name, _), d in counted.items() if name in SILENT_LABELS), bucket - bucket_speech)
        speech += bucket_speech * fraction
        silent += bucket_silent * fraction
        named.update(name for name, _ in counted if name not in SILENT_LABELS)
    out['speech_ratio'] = _round(min(speech / length, 1.0)) if read else None
    out['silence_ratio'] = _round(min(silent / length, 1.0)) if read else None
    out['n_speakers_named'] = len(named) if read else None
    for name in speakers:
        out[f'spk_{name}_ratio'] = _round(min(by_speaker.get(name, 0.0) / length, 1.0)) if buckets else None

    chunks = transcription.between(ws, we)
    if personal is None:
        out['n_asr_transcription'] = len(chunks)
        started = [(record, start, end) for record, start, end in chunks if ws <= start < we]
        out['n_spurts'] = len(started)
        out['mean_spurt_seconds'] = _round(_mean(end - start for _, start, end in started))
        out['words'] = _count_words(chunks, ws, we)
        out.update(_diarization_features(chunks, ws, we, length))
        return out

    # personal microphones: the spurts and turns are the group microphone's when there is one
    group = [c for c in chunks if participant_of(c[0].get('participant')) is None]
    basis = group if personal.has_group else chunks
    out['n_asr_transcription'] = len(basis)
    started = [(record, start, end) for record, start, end in basis if ws <= start < we]
    out['n_spurts'] = len(started)
    out['mean_spurt_seconds'] = _round(_mean(end - start for _, start, end in started))
    # a wearer's word counts when their microphone led while it was said, and, before level traces,
    # when it was said in a bucket the energy vote gave them
    worn: dict[str, int] = {}
    voted: dict[str, int] = {}
    spoken = 0  # every worn word, each spoken word once (with level traces)
    for tag in personal.participants:
        count = by_vote = 0
        for record, start, end in chunks:
            if participant_of(record.get('participant')) != tag:
                continue
            classes = personal.word_classes(record, start, end)
            counted = personal.words_counted(record)
            for i, offset in enumerate(word_times(record, start, end)):
                moment = start + offset
                if not ws <= moment < we:
                    continue
                won = personal.won_at(tag, moment)
                by_vote += won
                count += won if classes is None or i >= len(classes) else classes[i] == WORD_WEARER
                spoken += counted[i] if counted is not None and i < len(counted) else 1
        worn[f'p{tag}_words'] = count
        voted[f'p{tag}_vote_words'] = by_vote
    if personal.has_group:
        out['words'] = _count_words(group, ws, we)
    else:
        out['words'] = spoken if personal.by_word else sum(worn.values())
    out.update(worn)
    if personal.by_word:
        if not personal.has_group:
            out['vote_words'] = sum(voted.values())
        out.update(voted)
    out.update(_diarization_features(basis, ws, we, length))
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


COCO_SHOULDERS = {'left': 5, 'right': 6}


def _body_points(person: dict, indices: dict[str, int]) -> dict[str, tuple[float, float]]:
    """the named keypoints seen with at least BODY_CONFIDENCE, by side."""
    keypoints = person.get('keypoints') or []
    points = {}
    for side, index in indices.items():
        try:
            x, y, confidence = (float(v) for v in keypoints[index][:3])
        except (IndexError, TypeError, ValueError):
            continue
        if confidence >= BODY_CONFIDENCE and math.isfinite(x) and math.isfinite(y):
            points[side] = (x, y)
    return points


def body_scale(person: dict) -> tuple[tuple[float, float], float] | None:
    """the person's shoulder midpoint and shoulder width, in pixels, from both shoulders seen with
    at least BODY_CONFIDENCE; None otherwise."""
    shoulders = _body_points(person, COCO_SHOULDERS)
    if len(shoulders) < 2:
        return None
    (lx, ly), (rx, ry) = shoulders['left'], shoulders['right']
    width = math.hypot(lx - rx, ly - ry)
    return (((lx + rx) / 2.0, (ly + ry) / 2.0), width) if width > 1e-6 else None


def wrist_moves(before: dict, after: dict) -> list[float] | None:
    """how far each wrist moved in the image from one frame of a person to the next, in the
    person's shoulder widths (the mean of the two frames'): only wrists and shoulders seen with at
    least BODY_CONFIDENCE in both. In shoulder widths, so a pupil near the camera moves no faster
    than one far from it. In the image, not against the shoulders: a pupil who leans in or turns
    with the hands resting on the table moves no wrist, but hands carried along with the body (a
    pupil shifting in the chair with the hands in the lap) do move, so a movement here is the
    hand's, whether the arm or the body carried it, not work on the artifact. None when the
    shoulders are not seen in both frames or no wrist is."""
    scales = body_scale(before), body_scale(after)
    if None in scales:
        return None
    width = (scales[0][1] + scales[1][1]) / 2.0
    wrists0, wrists1 = _body_points(before, COCO_WRISTS), _body_points(after, COCO_WRISTS)
    moves = [math.dist(wrists0[side], wrists1[side]) / width for side in wrists0 if side in wrists1]
    return moves or None


ACTIVE, STILL, BETWEEN = 'active', 'still', 'between'


def hand_status(moves: list[float] | None) -> str | None:
    """a step's hands: active when a wrist moved at least HAND_ACTIVE_SW, still when every wrist
    seen moved less than HAND_STILL_SW, between otherwise; None without a wrist to judge."""
    if not moves:
        return None
    if max(moves) >= HAND_ACTIVE_SW:
        return ACTIVE
    if max(moves) < HAND_STILL_SW:
        return STILL
    return BETWEEN


def _box_iou(a, b) -> float:
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
        bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    except (TypeError, ValueError, IndexError):
        return 0.0
    w, h = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _hand_circles(person: dict, nudge: float | None) -> list:
    """the person's hand circles as the server scores them (features.hand_regions at its keypoint
    confidence, placed by `nudge`); none for a body without keypoints or a box."""
    if not person.get('keypoints'):
        return []
    try:
        return vfa_features.hand_regions(person, VFA_KEYPOINT_CONFIDENCE, nudge)
    except (KeyError, TypeError, ValueError, IndexError):
        return []


def _inside_share(inner, outer) -> float:
    """the share of box `inner`'s area that lies inside box `outer` (0 for a box that is not one)."""
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in inner[:4])
        bx1, by1, bx2, by2 = (float(v) for v in outer[:4])
    except (TypeError, ValueError, IndexError):
        return 0.0
    w, h = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    area = (ax2 - ax1) * (ay2 - ay1)
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


def _outsiders(bodies: list[dict], person: dict, pupils: set[str], seats_here: dict | None,
               tracks_here: set | frozenset, nudge: float | None = None) -> list[dict]:
    """the bodies of a frame that belong to no one of the group, as far as the frame can tell: not
    `person`, not a pupil (by tag), not a second skeleton or a part of one of them (a box
    overlapping theirs by DUPLICATE_BODY_IOU, or lying CONTAINED_BODY_SHARE inside it, or a hand
    circle centred within DUPLICATE_HAND_SW of one of theirs, a shared wrist), not a body at a
    pupil's seat on this camera (`seats_here`, the pupil seen there or not), and not one on a track
    the server gave a pupil's tag to at some time of the session (`tracks_here`, this camera's track
    ids). `nudge` places the hand circles."""
    group = [p for p in bodies if p.get('tag_id') is not None and str(p['tag_id']) in pupils]
    if not any(p is person for p in group):
        group.append(person)
    # the group's hands, each with how close another's must come to be the same wrist: DUPLICATE_HAND_SW
    # of the shoulder width the circle's radius was made from
    hands = [(centre, DUPLICATE_HAND_SW / vfa_features.HAND_RADIUS_SHOULDERS * radius)
             for p in group for centre, radius in _hand_circles(p, nudge)]
    seats = list((seats_here or {}).values())
    out = []
    for body in bodies:
        if any(body is p for p in group):
            continue
        if body.get('track_id') is not None and body['track_id'] in tracks_here:
            continue
        box = body.get('bbox')
        if any(_box_iou(box, p.get('bbox')) >= DUPLICATE_BODY_IOU or _inside_share(box, p.get('bbox')) >= CONTAINED_BODY_SHARE
               for p in group):
            continue
        centre = _box_centre(body)
        if centre is not None and any(_at_seat(centre, seat) for seat in seats):
            continue
        if any(math.dist(mine, theirs) <= same for mine, _ in _hand_circles(body, nudge) for theirs, same in hands):
            continue
        out.append(body)
    return out


def _other_hands_near(person: dict, others: list[dict], nudge: float | None) -> bool | None:
    """whether a hand circle of one of `others` has its centre within HAND_LENGTH_SW of the
    person's shoulder width from the centre of one of the person's; None when the person shows no
    hand circle or no shoulder width to judge by."""
    scale = body_scale(person)
    own = _hand_circles(person, nudge)
    if scale is None or not own:
        return None
    reach = HAND_LENGTH_SW * scale[1]
    return any(math.dist(centre, mine) <= reach for other in others for centre, _ in _hand_circles(other, nudge)
               for mine, _ in own)


def _follows(watcher: dict, worker: dict, tag: str, frame: dict, nudge: float | None) -> bool:
    """whether the watcher's gaze in this frame is on the worker's hands: a target of the worker's
    hands (partner_hands naming `tag`), or a gaze point in frame within reach of one of the
    worker's hand circles as gaze_target scores it (the circle grown by the gaze tolerance), unless
    the target is the watcher's own hands, which the server found nearer."""
    gaze = watcher.get('gaze') or {}
    target = gaze.get('target') or {}
    category = target.get('category') or 'unknown'
    if category == 'partner_hands' and str(target.get('person_id')) == tag:
        return True
    point = gaze.get('point')
    if category in _NO_POINT or category == 'own_hands' or not point or len(point) < 2:
        return False
    try:
        tolerance = vfa_features.gaze_tolerance(float(frame.get('width') or 0.0), float(frame.get('height') or 0.0))
        where = (float(point[0]), float(point[1]))
    except (TypeError, ValueError):
        return False
    return any(math.dist(where, centre) - radius <= tolerance for centre, radius in _hand_circles(worker, nudge))


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


# ---- identity along the tracks ----

def _decoded(person: dict) -> bool:
    """whether the person's tag was read in this frame (`torso`, `box`, or an older event that
    does not say), not carried by the server's track memory (`track`) or by the fusion."""
    return person.get('tag_id') is not None and person.get('tag_match') not in ('track', PROPAGATED)


def _track_segments(occurrences: list[tuple], gap: float) -> list[list[tuple]]:
    """a track's occurrences (sorted by time) cut wherever the track id was not seen for more
    than `gap` seconds: the id was handed out again."""
    segments: list[list[tuple]] = []
    for occurrence in occurrences:
        if not segments or occurrence[0] - segments[-1][-1][0] > gap:
            segments.append([])
        segments[-1].append(occurrence)
    return segments


def _nearest_tag(anchors: list[tuple[float, Any]], times: list[float], moment: float):
    """the tag of the read nearest `moment` (the earlier one on a tie)."""
    i = bisect.bisect_left(times, moment)
    before = anchors[i - 1] if i > 0 else None
    after = anchors[i] if i < len(anchors) else None
    if before is None:
        return after[1]
    if after is None or moment - before[0] <= after[0] - moment:
        return before[1]
    return after[1]


def _renamed(frame: dict, persons: list[dict], names: dict[str, str]) -> dict:
    """a copy of the frame with `persons`, and its pairs and gaze targets renamed from the
    persons' old ids to their tags."""
    frame = dict(frame, persons=persons)
    pairs = frame.get('pairs')
    if isinstance(pairs, dict):
        frame['pairs'] = {'|'.join(names.get(part, part) for part in str(key).split('|')): value
                          for key, value in pairs.items()}
    for n, person in enumerate(persons):
        target = (person.get('gaze') or {}).get('target')
        if isinstance(target, dict) and str(target.get('person_id')) in names:
            gaze = dict(person['gaze'], target=dict(target, person_id=names[str(target['person_id'])]))
            persons[n] = dict(person, gaze=gaze)
    return frame


def propagate_track_tags(records: Iterable[dict], layout: tuple[int | None, frozenset[str]] | None = None,
                         gap: float = TRACK_GAP_SECONDS) -> list[dict]:
    """the vfa_features records with the tags carried along the tracks, offline, over the whole
    session: within a camera, a person without a tag whose track has a read tag somewhere takes
    the tag of the nearest read of that track (`tag_match: propagated`, `person_id` the tag), so
    the frames before a track's first read and those after the server's memory of it was lost
    are named too. A track that read two tags is split at the switch that way, each untagged
    frame going to the read nearest it. A track id not seen for more than `gap` seconds is
    another track from then on. A tag is only ever given to a person without one, and not in a
    frame where another person already carries it or where two tracks would get it (then
    neither does); a track id twice in one frame names nobody. The pairs and gaze targets of a
    renamed person's frame follow the new name. Records nothing changes in are returned as they
    were; `layout` is the session's frame_set_layout, worked out from the records when not
    given."""
    records = list(records)
    _, shared = layout if layout is not None else frame_set_layout(records)
    parsed = [_frames_of(record) for record in records]
    # (camera, track id) -> [(time, record, frame, person)]
    tracks: dict[tuple[str, Any], list[tuple[float, int, int, int]]] = defaultdict(list)
    for i, (record, frames) in enumerate(zip(records, parsed)):
        moment = _time(record)
        for j, (frame, camera) in enumerate(zip(frames, camera_keys(frames, shared))):
            persons = frame.get('persons') or []
            counts = defaultdict(int)
            for person in persons:
                if person.get('track_id') is not None:
                    counts[person['track_id']] += 1
            for k, person in enumerate(persons):
                track = person.get('track_id')
                if track is not None and counts[track] == 1:
                    tracks[(camera, track)].append((moment, i, j, k))
    # (record, frame) -> [(person, tag)]
    proposed: dict[tuple[int, int], list[tuple[int, Any]]] = defaultdict(list)
    for occurrences in tracks.values():
        occurrences.sort()
        for segment in _track_segments(occurrences, gap):
            anchors = [(moment, parsed[i][j]['persons'][k]['tag_id']) for moment, i, j, k in segment
                       if _decoded(parsed[i][j]['persons'][k])]
            if not anchors:
                continue
            times = [moment for moment, _ in anchors]
            for moment, i, j, k in segment:
                if parsed[i][j]['persons'][k].get('tag_id') is None:
                    proposed[(i, j)].append((k, _nearest_tag(anchors, times, moment)))
    if not proposed:
        return records
    out = list(records)
    changed: dict[int, dict[int, dict]] = defaultdict(dict)
    for (i, j), wanted in proposed.items():
        frame = parsed[i][j]
        persons = list(frame.get('persons') or [])
        held = {str(p['tag_id']) for p in persons if p.get('tag_id') is not None}
        asked = defaultdict(int)
        for _, tag in wanted:
            asked[str(tag)] += 1
        names: dict[str, str] = {}
        for k, tag in wanted:
            if str(tag) in held or asked[str(tag)] > 1:
                continue
            person = persons[k]
            if person.get('person_id') is not None:
                names[str(person['person_id'])] = str(tag)
            persons[k] = dict(person, tag_id=tag, tag_match=PROPAGATED, person_id=str(tag))
        if any(p is not q for p, q in zip(persons, frame.get('persons') or [])):
            changed[i][j] = _renamed(frame, persons, names)
    for i, frames in changed.items():
        out[i] = dict(records[i], features=[frames.get(j, frame) for j, frame in enumerate(parsed[i])])
    return out


def _tracked(records: Iterable[dict]) -> bool:
    """whether the features endpoint tracked the persons of these records (some carry a
    `track_id`)."""
    return any(person.get('track_id') is not None for record in records for frame in _frames_of(record)
               for person in frame.get('persons') or [])


def _propagated_count(features: EventIndex, ws: float, we: float) -> int:
    """the persons of the window's frames whose tag the fusion carried along their track."""
    return sum(1 for record, _, _ in features.between(ws, we) for frame in _frames_of(record)
               for person in frame.get('persons') or [] if person.get('tag_match') == PROPAGATED)


# ---- the hand circles, the work area and the gaze points ----

def relabel_hand_circles(records: Iterable[dict], nudge: float | None = None,
                         min_confidence: float = VFA_KEYPOINT_CONFIDENCE,
                         inout_threshold: float = VFA_INOUT_THRESHOLD) -> list[dict]:
    """the vfa_features records with every frame's gaze targets and pair hand distances made again
    from what the frame stores, with the hand circle placed by `nudge` (features.HAND_NUDGE when not
    given; features.relabel_answer), so a change of the hand circle needs no replay of the video.
    With the circle the server used (features.HAND_NUDGE_V1 for every event stored before its image
    was rebuilt) it gives back the server's targets and distances. Run it on the records as stored:
    the server's names (person_id, the pair keys) are what it rebuilds from, so it comes before
    propagate_track_tags, and before label_work_areas, since the circle decides which gazes are
    `elsewhere`. A frame the server already made with this circle (its `scoring` says so,
    features.hand_circle_of), a frame the server already labelled with a work area (it carries
    `work_area`), and one that cannot be remade (no width, a person without keypoints), are kept as
    they are; records nothing changed in are returned as they were, aligned to the input.
    `min_confidence` and `inout_threshold` stand in for the server's where a frame does not state
    its own."""
    records = list(records)
    out = list(records)
    nudge = vfa_features.HAND_NUDGE if nudge is None else float(nudge)
    for i, record in enumerate(records):
        frames = _frames_of(record)
        remade = [frame if not isinstance(frame, dict) or 'work_area' in frame
                  or vfa_features.HAND_NUDGES.get(vfa_features.hand_circle_of(frame)) == nudge
                  else vfa_features.relabel_answer(frame, min_confidence, inout_threshold, nudge) for frame in frames]
        if any(new is not old for new, old in zip(remade, frames)):
            out[i] = dict(record, features=remade)
    return out


def stored_hand_circles(records: Iterable[dict]) -> Counter:
    """the stored frames by the version of the hand circle the server made them with
    (features.hand_circle_of: 1 for a frame that does not say, None for a version this code does
    not know)."""
    return Counter(vfa_features.hand_circle_of(frame) for record in records for frame in _frames_of(record)
                   if isinstance(frame, dict))


def table_hand_circle(records: Iterable[dict], hand_relabel: bool = True) -> int:
    """the version of the hand circle a table's gaze targets, hand distances, work area and hand
    columns come from: the current one (features.HAND_CIRCLE_VERSION) with the relabel, else the one
    most stored frames were made with (1 when none says)."""
    if hand_relabel:
        return vfa_features.HAND_CIRCLE_VERSION
    known = [(n, version) for version, n in stored_hand_circles(records).items() if version is not None]
    return max(known)[1] if known else 1


def pupil_tracks_of(records: Iterable[dict], pupils: Iterable[str],
                    layout: tuple[int | None, frozenset[str]] | None = None) -> dict[str, frozenset]:
    """camera -> the track ids the server gave a pupil's tag to (read, or kept on the track) in some
    frame of the session: a body on such a track is that pupil, or a second skeleton of them, even
    where it carries no tag."""
    records = list(records)
    _, shared = layout if layout is not None else frame_set_layout(records)
    wanted = {str(tag) for tag in pupils}
    tracks: dict[str, set] = defaultdict(set)
    for record in records:
        frames = _frames_of(record)
        for frame, camera in zip(frames, camera_keys(frames, shared)):
            for person in frame.get('persons') or []:
                if person.get('track_id') is not None and person.get('tag_id') is not None \
                        and str(person['tag_id']) in wanted:
                    tracks[camera].add(person['track_id'])
    return {camera: frozenset(ids) for camera, ids in tracks.items()}


def label_work_areas(records: Iterable[dict], pupils: Iterable[str],
                     layout: tuple[int | None, frozenset[str]] | None = None, nudge: float | None = None) -> list[dict]:
    """the vfa_features records with every camera's work area applied (work_area.apply_work_area):
    in time order, each frame's pupils' hands teach its camera's area (one per camera and frame
    size), and a gaze the server called `elsewhere` that lands in the ready area becomes
    `work_area`; the frame gets `work_area`, the box used or None. The records come back aligned
    to the input. The area learns from the tags the server gave, so feed the records before
    propagate_track_tags, and from the hand circles placed by `nudge` (features.HAND_NUDGE when not
    given), which should be the circle the targets were made with. A frame that already carries
    `work_area` (a server that labels online) is kept as it is, and so is a frame without a width;
    records nothing changed in are returned as they were. `layout` is the session's
    frame_set_layout, worked out from the records when not given."""
    records = list(records)
    _, shared = layout if layout is not None else frame_set_layout(records)
    wanted = [str(tag) for tag in pupils or ()]
    areas: dict[tuple, WorkArea] = {}
    out = list(records)
    # a record without a time never reaches a window (EventIndex drops it), so it teaches no area
    for i in sorted((k for k in range(len(records)) if _time(records[k]) > 0), key=lambda k: _time(records[k])):
        frames = _frames_of(records[i])
        labelled, changed = [], False
        for frame, camera in zip(frames, camera_keys(frames, shared)):
            try:
                width, height = float(frame.get('width') or 0.0), float(frame.get('height') or 0.0)
            except (TypeError, ValueError):
                width, height = 0.0, 0.0
            if 'work_area' in frame or not width:
                labelled.append(frame)
                continue
            area = areas.setdefault((camera, width, height), WorkArea(width, height))
            labelled.append(apply_work_area(frame, area, wanted, nudge))
            changed = True
        if changed:
            out[i] = dict(records[i], features=labelled)
    return out


# out of frame or unreadable: no point to compare, as the server's pairwise needs both gazes in frame
_NO_POINT = ('unknown', 'out_of_frame')


def gaze_points(records: Iterable[dict], layout: tuple[int | None, frozenset[str]] | None = None) -> dict:
    """{(camera, tag): (times, points)}: every tagged person's gaze point in frame, by camera, in
    time order, with the frame's width beside it ((x, y, width)); a target that is unknown or
    out_of_frame, or a frame without a width, gives none."""
    records = sorted(records, key=_time)
    _, shared = layout if layout is not None else frame_set_layout(records)
    index: dict[tuple[str, str], tuple[list[float], list[tuple[float, float, float]]]] = {}
    for record in records:
        moment = _time(record)
        if moment <= 0:
            continue
        frames = _frames_of(record)
        for frame, camera in zip(frames, camera_keys(frames, shared)):
            try:
                width = float(frame.get('width') or 0.0)
            except (TypeError, ValueError):
                width = 0.0
            if not width:
                continue
            for person in frame.get('persons') or []:
                if person.get('tag_id') is None:
                    continue
                gaze = person.get('gaze') or {}
                target = gaze.get('target') or {}
                point = gaze.get('point')
                if (target.get('category') or 'unknown') in _NO_POINT or not point or len(point) < 2:
                    continue
                times, points = index.setdefault((camera, str(person['tag_id'])), ([], []))
                times.append(moment)
                points.append((float(point[0]), float(point[1]), width))
    return index


def _nearest(times: list[float], moment: float) -> int | None:
    """the index of the time nearest `moment` (the earlier one on a tie); None for no times."""
    if not times:
        return None
    i = bisect.bisect_left(times, moment)
    if i == 0:
        return 0
    if i == len(times):
        return i - 1
    return i - 1 if moment - times[i - 1] <= times[i] - moment else i


def joint_attention_baseline(gaze_index: dict, a: str, b: str, ws: float, we: float) -> float | None:
    """how often a's gaze in [ws, we) met b's on the same camera JOINT_BASELINE_LAGS seconds
    earlier, and b's met a's: per camera, per gaze point of one at time t in the window and per
    lag d, the other's point nearest t - d (within JOINT_BASELINE_SLACK) is one comparison, a hit
    within JOINT_ATTENTION_WIDTH of the frame's width. Only points before `ws` are compared, so a
    window longer than the smallest lag never meets itself. Hits over comparisons, pooled over
    cameras, lags and both directions; None below JOINT_BASELINE_MIN comparisons."""
    comparisons, hits = 0, 0
    cameras = {camera for camera, _ in gaze_index}
    for camera in cameras:
        for x, y in ((a, b), (b, a)):
            source, other = gaze_index.get((camera, x)), gaze_index.get((camera, y))
            if not source or not other:
                continue
            times, points = source
            other_times, other_points = other
            for i in range(bisect.bisect_left(times, ws), bisect.bisect_left(times, we)):
                px, py, width = points[i]
                for lag in JOINT_BASELINE_LAGS:
                    moment = times[i] - lag
                    j = _nearest(other_times, moment)
                    if j is None or abs(other_times[j] - moment) > JOINT_BASELINE_SLACK or other_times[j] >= ws:
                        continue
                    comparisons += 1
                    qx, qy, _ = other_points[j]
                    hits += math.dist((px, py), (qx, qy)) <= JOINT_ATTENTION_WIDTH * width
    return hits / comparisons if comparisons >= JOINT_BASELINE_MIN else None


def _seated_untagged(persons: list[dict], seats_here: dict | None, pupils: set[str]) -> frozenset[str]:
    """the person_ids of the frame's bodies without a tag that stand at the seat (on this camera)
    of a pupil whose tag the frame does not hold: almost certainly that pupil, badge unread."""
    if not seats_here:
        return frozenset()
    tagged = {str(p['tag_id']) for p in persons if p.get('tag_id') is not None}
    missing = [seat for tag, seat in seats_here.items() if tag in pupils and tag not in tagged]
    if not missing:
        return frozenset()
    found = set()
    for person in persons:
        if person.get('tag_id') is not None or person.get('person_id') is None:
            continue
        body = _box_centre(person)
        if body is not None and any(_at_seat(body, seat) for seat in missing):
            found.add(str(person['person_id']))
    return frozenset(found)


def _gaze_label(person: dict, pupils: set[str], work_area: bool = True,
                seated: frozenset[str] = frozenset()) -> str:
    """where a person's gaze landed, as the table counts it: the server's category, with a face or
    hands of anyone but a pupil as other_face or other_hands. A target in `seated` (an untagged body
    at a missing pupil's seat, _seated_untagged) is a pupil."""
    target = (person.get('gaze') or {}).get('target') or {}
    category = target.get('category') or 'unknown'
    if category in ('partner_face', 'partner_hands') and str(target.get('person_id')) not in pupils \
            and str(target.get('person_id')) not in seated:
        return 'other_face' if category == 'partner_face' else 'other_hands'
    if category == 'work_area' and not work_area:
        return 'elsewhere'
    return category


def body_gaze_features(features: EventIndex, ws: float, we: float, participants: list[str],
                       layout: tuple[int | None, frozenset[str]] | None = None, pupils: Iterable[str] | None = None,
                       gaze_index: dict | None = None, work_area: bool = True, seats: dict | None = None,
                       nudge: float | None = None, pupil_tracks: dict | None = None) -> dict:
    """what the bodies and gazes did in the window, per person and per pair, from the frames the
    features endpoint answered. Every sequence (the wrist speed, following each hand from one
    frame to the next; the hand steps; the gaze switches; the yaw spread) is taken within one
    camera, and the cameras are then pooled, each frame (or step) counting once in the shares and
    the means. The older distances and speeds are in shares of the frame's width; the hand columns
    added on 2026-09-24 are in shoulder widths, so neither the camera's distance nor its field of
    view moves them. `layout` is the session's frame_set_layout, worked out from the whole index when
    not given; `pupils` the session's pupils (default_pupils of the participants when not given),
    who alone are partners; `gaze_index` the session's gaze_points, worked out from the whole index
    when not given. `work_area` False (a table without the work area) leaves out
    p<tag>_work_area_ready_ratio. `seats` (seats_of) names a pupil in an untagged gaze target that
    stands at the seat of a pupil the camera's frame does not hold (_seated_untagged), and gives
    `n_vfa_seat_partners`, the gaze frames named that way; without it there is no such column and
    every untagged target is other. `nudge` places the hand circles the hand columns read
    (features.HAND_NUDGE when not given): the circle the gaze targets were made with.
    `pupil_tracks` (pupil_tracks_of) names the bodies on a pupil's track, which are no one else.

    The hands, step by step. A step is two frame sets in a row (MIN_STEP_SECONDS to
    MAX_STEP_SECONDS apart) that both hold the person, and only one body with their tag, on the
    same camera; wrist_moves gives each wrist's movement in the image in the person's shoulder
    widths, and hand_status calls the step active, still or between. Per person: p<tag>_hand_steps
    (steps with a status), p<tag>_hands_active_ratio and p<tag>_hands_still_ratio (of those steps),
    p<tag>_wrist_speed_sw (the mean wrist speed, shoulder widths a second), and
    p<tag>_other_hands_near_ratio (of the person's frames with a hand circle and a shoulder width,
    the share in which a hand circle of a body outside the group (_outsiders: no pupil, no second
    skeleton or part of one, no body at a pupil's seat or on a pupil's track) comes within
    HAND_LENGTH_SW of one of theirs, _other_hands_near). Per pair, over the steps both have a status
    in on one camera (pair<a>_<b>_hand_steps): one_active_ratio (one active, the other still),
    both_active_ratio, both_still_ratio, and follow_ratio, of the one-active steps (at least
    FOLLOW_MIN_STEPS of them) the share in which the still one's gaze at the step's end is on the
    active one's hands (their partner_hands target, or a gaze point within reach of their hand
    circle while the target is not the still one's own hands, _follows). Over the frames that hold
    both with a hand distance and both shoulder widths: hand_dist_sw_min and hand_dist_sw_mean (the
    hand distance over the mean of the two shoulder widths) and hands_close_ratio (the share of
    those frames with the hands within HAND_LENGTH_SW). A ratio with nothing to judge is None, never
    0."""
    out: dict[str, Any] = {}
    records = features.between(ws, we)
    out['n_vfa_features'] = len(records)
    modal, shared = layout if layout is not None else frame_set_layout(features.records)
    pupils = {str(tag) for tag in (pupils if pupils is not None else default_pupils(participants))}
    if gaze_index is None:
        gaze_index = gaze_points(features.records, (modal, shared))
    # camera -> person -> [(time, frame set, person dict, width, area ready, gaze label)], and
    # camera -> [(time, frame set, frame)]
    seen: dict[str, dict[str, list[tuple[float, int, dict, float, bool, str]]]] = defaultdict(lambda: defaultdict(list))
    frames_by_camera: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
    # camera -> frame set -> frame
    frame_at: dict[str, dict[int, dict]] = defaultdict(dict)
    angles, incomplete, seat_named = set(), 0, 0
    for number, (record, start, _) in enumerate(records):
        frames = _frames_of(record)
        incomplete += modal is not None and len(frames) != modal
        for frame, camera in zip(frames, camera_keys(frames, shared)):
            angles.add(_angle(frame))
            width = float(frame.get('width') or 0.0) or None
            ready = frame.get('work_area') is not None
            frames_by_camera[camera].append((start, number, frame))
            persons = frame.get('persons', [])
            seated = _seated_untagged(persons, (seats or {}).get(camera), pupils)
            frame_at[camera][number] = frame
            for person in persons:
                if person.get('tag_id') is None:
                    continue
                label = _gaze_label(person, pupils, work_area, seated)
                if seated and label in ('partner_face', 'partner_hands') \
                        and str(((person.get('gaze') or {}).get('target') or {}).get('person_id')) not in pupils:
                    seat_named += 1
                seen[camera][str(person['tag_id'])].append((start, number, person, width, ready, label))
    out['n_vfa_angles'] = len(angles)
    out['n_vfa_cameras'] = len(frames_by_camera)
    out['n_vfa_incomplete'] = incomplete
    if seats is not None:
        out['n_vfa_seat_partners'] = seat_named

    # camera -> tag -> frame set at a step's end -> (the step's hand status, the person there)
    statuses: dict[str, dict[str, dict[int, tuple[str, dict]]]] = defaultdict(lambda: defaultdict(dict))
    for tag in participants:
        yaws, spreads, speeds, switches, categories, ready = [], [], [], [], [], 0
        cameras, frame_sets = 0, set()
        steps, speeds_sw, near = [], [], []
        for camera, persons in seen.items():
            rows = sorted(persons.get(tag, []), key=lambda row: row[0])
            if not rows:
                continue
            cameras += 1
            frame_sets.update(number for _, number, *_ in rows)
            ready += sum(1 for *_, on, _ in rows if on)
            camera_yaws = [float(person['head_yaw']) for _, _, person, *_ in rows if person.get('head_yaw') is not None]
            yaws.extend(camera_yaws)
            if len(camera_yaws) > 1:
                spreads.append((_std(camera_yaws), len(camera_yaws)))
            sequence = [label for *_, label in rows]
            categories.extend(sequence)
            switches.append(sum(1 for a, b in zip(sequence, sequence[1:]) if a != b))
            # each hand against itself from one frame to the next, in frame widths per second
            for (t0, _, p0, width, *_), (t1, _, p1, *_) in zip(rows, rows[1:]):
                w0, w1 = _wrists(p0), _wrists(p1)
                moved = [math.dist(w0[side], w1[side]) for side in w0 if side in w1]
                if moved and width and t1 > t0:
                    speeds.append(_mean(moved) / width / (t1 - t0))
            # the hands in shoulder widths, over steps of two frame sets in a row on this camera; a frame
            # set with two bodies wearing the tag there (a track's remembered tag and a read one) cannot
            # say which body the step follows
            bodies = Counter(number for _, number, *_ in rows)
            for (t0, n0, p0, *_), (t1, n1, p1, *_) in zip(rows, rows[1:]):
                if n1 != n0 + 1 or bodies[n0] != 1 or bodies[n1] != 1 \
                        or not MIN_STEP_SECONDS <= t1 - t0 <= MAX_STEP_SECONDS:
                    continue
                moves = wrist_moves(p0, p1)
                status = hand_status(moves)
                if status is None:
                    continue
                steps.append(status)
                speeds_sw.append(_mean(moves) / (t1 - t0))
                statuses[camera][tag][n1] = (status, p1)
            # the hands of someone outside the group at the person's own, frame by frame
            seats_here, tracks_here = (seats or {}).get(camera), (pupil_tracks or {}).get(camera, frozenset())
            for _, number, person, *_ in rows:
                others = _outsiders(frame_at[camera][number].get('persons') or [], person, pupils, seats_here, tracks_here,
                                    nudge)
                hit = _other_hands_near(person, others, nudge)
                if hit is not None:
                    near.append(hit)
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
        if work_area:
            out[f'p{tag}_work_area_ready_ratio'] = _round(ready / len(categories)) if categories else None
        out[f'p{tag}_in_group'] = 1 if str(tag) in pupils else 0
        out[f'p{tag}_hand_steps'] = len(steps)
        out[f'p{tag}_hands_active_ratio'] = _round(steps.count(ACTIVE) / len(steps)) if steps else None
        out[f'p{tag}_hands_still_ratio'] = _round(steps.count(STILL) / len(steps)) if steps else None
        out[f'p{tag}_wrist_speed_sw'] = _round(_mean(speeds_sw))
        out[f'p{tag}_other_hands_near_ratio'] = _round(sum(near) / len(near)) if near else None

    for a, b in _pairs(participants):
        hand, gaze_dist, joint, mutual, frame_sets, hand_sw = [], [], [], [], set(), []
        for rows in frames_by_camera.values():
            for _, number, frame in rows:
                width = float(frame.get('width') or 0.0)
                persons = {str(p['tag_id']): p for p in frame.get('persons', []) if p.get('tag_id') is not None}
                if a not in persons or b not in persons:
                    continue
                frame_sets.add(number)
                pairs = frame.get('pairs') or {}
                pair = pairs.get(f'{a}|{b}') or pairs.get(f'{b}|{a}') or {}
                if pair.get('hand_distance') is not None:
                    scales = body_scale(persons[a]), body_scale(persons[b])
                    if None not in scales:
                        hand_sw.append(float(pair['hand_distance']) / ((scales[0][1] + scales[1][1]) / 2.0))
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
        ratio = _round(_mean(joint))
        out[f'pair{a}_{b}_joint_attention_ratio'] = ratio
        out[f'pair{a}_{b}_mutual_gaze_ratio'] = _round(_mean(mutual))
        # the pair's own rate of meeting 20-40 s earlier, and the window's joint attention above it
        baseline = _round(joint_attention_baseline(gaze_index, a, b, ws, we))
        out[f'pair{a}_{b}_joint_attention_baseline'] = baseline
        out[f'pair{a}_{b}_joint_attention_excess'] = _round(ratio - baseline) if ratio is not None and baseline is not None else None
        # the pair's hands step by step, on the cameras that saw both
        both, one, followed = [], 0, 0
        for camera, by_tag in statuses.items():
            here_a, here_b = by_tag.get(a, {}), by_tag.get(b, {})
            for number in sorted(set(here_a) & set(here_b)):
                (sa, pa), (sb, pb) = here_a[number], here_b[number]
                both.append((sa, sb))
                if {sa, sb} == {ACTIVE, STILL}:
                    one += 1
                    watcher, worker, tag = (pb, pa, a) if sa == ACTIVE else (pa, pb, b)
                    followed += _follows(watcher, worker, tag, frame_at[camera][number], nudge)
        out[f'pair{a}_{b}_hand_steps'] = len(both)
        out[f'pair{a}_{b}_one_active_ratio'] = _round(one / len(both)) if both else None
        out[f'pair{a}_{b}_both_active_ratio'] = _round(sum(1 for s in both if s == (ACTIVE, ACTIVE)) / len(both)) if both else None
        out[f'pair{a}_{b}_both_still_ratio'] = _round(sum(1 for s in both if s == (STILL, STILL)) / len(both)) if both else None
        out[f'pair{a}_{b}_follow_ratio'] = _round(followed / one) if one >= FOLLOW_MIN_STEPS else None
        out[f'pair{a}_{b}_hand_dist_sw_min'] = _round(min(hand_sw)) if hand_sw else None
        out[f'pair{a}_{b}_hand_dist_sw_mean'] = _round(_mean(hand_sw))
        out[f'pair{a}_{b}_hands_close_ratio'] = _round(sum(1 for d in hand_sw if d <= HAND_LENGTH_SW) / len(hand_sw)) \
            if hand_sw else None
    return out


# ---- the seats ----

def _box_centre(person: dict) -> tuple[float, float, float] | None:
    """the person's `bbox` [x1, y1, x2, y2] as (centre x, centre y, width); None when the box is
    missing, short, not a number or empty."""
    bbox = person.get('bbox')
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1


def seats_of(records: Iterable[dict], participants: list[str],
             layout: tuple[int | None, frozenset[str]] | None = None) -> dict[str, dict[str, tuple[float, float, float]]]:
    """the seat of every participant on every camera, learned from the whole session: camera ->
    tag -> (x, y, radius), the median centre of the tag's own boxes on that camera and
    SEAT_RADIUS_WIDTHS of their median width. A camera that gave a tag fewer than SEAT_MIN_BOXES
    boxes has no seat for it. `layout` is the session's frame_set_layout, worked out from the
    records when not given."""
    records = list(records)
    _, shared = layout if layout is not None else frame_set_layout(records)
    wanted = set(participants)
    boxes: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        frames = _frames_of(record)
        for frame, camera in zip(frames, camera_keys(frames, shared)):
            for person in frame.get('persons') or []:
                tag = person.get('tag_id')
                if tag is None or str(tag) not in wanted:
                    continue
                box = _box_centre(person)
                if box is not None:
                    boxes[camera][str(tag)].append(box)
    seats: dict[str, dict[str, tuple[float, float, float]]] = {}
    for camera, by_tag in boxes.items():
        for tag, found in by_tag.items():
            if len(found) < SEAT_MIN_BOXES:
                continue
            xs, ys, widths = zip(*found)
            seats.setdefault(camera, {})[tag] = (statistics.median(xs), statistics.median(ys),
                                                 SEAT_RADIUS_WIDTHS * statistics.median(widths))
    return seats


def _at_seat(body: tuple[float, float, float], seat: tuple[float, float, float]) -> bool:
    """whether a body's centre is within the seat's radius (on it counts)."""
    return math.dist(body[:2], seat[:2]) <= seat[2]


def seat_features(features: EventIndex, ws: float, we: float, participants: list[str], seats: dict,
                  layout: tuple[int | None, frozenset[str]]) -> dict:
    """where the pose model saw bodies without a tag at the participants' seats in the window.
    p<tag>_untagged_at_seat_ratio: the share of the window's frame sets in which, on some camera
    holding the tag's seat, a body without a tag stood at the seat while the tag itself was not
    seen on that camera; None when no such camera gave a frame. A frame set that lost those
    cameras counts as one without such a body, so a camera that dropped out cannot make one frame
    the whole window. n_untagged_at_seats: the untagged bodies at any seat, summed over the
    cameras, per frame set of the window; None when no camera holding a seat gave a frame."""
    _, shared = layout
    hits, bases = defaultdict(int), defaultdict(int)
    total, seat_sets = 0, 0
    inside = features.between(ws, we)
    for record, _, _ in inside:
        frames = _frames_of(record)
        with_seat, hit, any_seat, count = set(), set(), False, 0
        for frame, camera in zip(frames, camera_keys(frames, shared)):
            here = seats.get(camera)
            if not here:
                continue
            any_seat = True
            persons = frame.get('persons') or []
            tagged = {str(p['tag_id']) for p in persons if p.get('tag_id') is not None}
            bodies = [box for box in (_box_centre(p) for p in persons if p.get('tag_id') is None) if box is not None]
            count += sum(1 for body in bodies if any(_at_seat(body, seat) for seat in here.values()))
            for tag, seat in here.items():
                with_seat.add(tag)
                if tag not in tagged and any(_at_seat(body, seat) for body in bodies):
                    hit.add(tag)
        if any_seat:
            seat_sets += 1
            total += count
        for tag in with_seat:
            bases[tag] += 1
        for tag in hit:
            hits[tag] += 1
    # the share is of every frame set in the window, the one the presence gate reads
    out = {f'p{tag}_untagged_at_seat_ratio': _round(hits[tag] / len(inside)) if bases[tag] else None
           for tag in participants}
    out['n_untagged_at_seats'] = _round(total / len(inside)) if seat_sets else None
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
                    participants: list[str] | None = None, speakers: list[str] | None = None,
                    track_tags: bool = True, pupils: list[str] | None = None,
                    work_area: bool = True, seat_partners: bool = True, hand_relabel: bool = True) -> list[dict]:
    """the fusion table: one row per window over the session's span. `pupils` are the session's
    pupils, the in-group set whose faces and hands are a partner's (default_pupils of the
    participants when not given). With `hand_relabel` (the default) every stored frame's gaze
    targets and pair hand distances are first made again with the current hand circle
    (relabel_hand_circles, features.HAND_NUDGE); without it they stay as the server stored them,
    and the work area and the hand columns read the circle those were made with
    (table_hand_circle: the version the frames state, version 1 for frames that state none, as every
    frame stored before the server's image was rebuilt). With `work_area`
    (the default) a gaze the server called elsewhere that lands in its camera's work area is
    work_area (label_work_areas). With `track_tags` (the default) the tags are then carried along
    the tracks of the features endpoint (propagate_track_tags), and a session whose persons were
    tracked gets `n_vfa_propagated`. With `seat_partners` (the default) a gaze on an untagged body
    at the seat of a pupil the camera's frame does not hold counts as a partner's
    (n_vfa_seat_partners)."""
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be greater than 0")
    span = session_span(events)
    if span is None:
        return []
    participants = list(participants) if participants else participants_of(events)
    pupils = [str(tag) for tag in pupils] if pupils is not None else default_pupils(participants)
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
    raw = events.get(EVENT_TYPE_VFA_FEATURES, [])
    layout = frame_set_layout(raw)
    # the gaze targets and hand distances made again with the current hand circle, from the frames
    # as the server stored and named them; it touches no box, tag or name. Without the relabel, the
    # work area and the hand columns read the circle the stored targets were made with
    nudge = vfa_features.HAND_NUDGES[table_hand_circle(raw, hand_relabel)]
    circled = relabel_hand_circles(raw, nudge) if hand_relabel and raw else raw
    # the work area of every camera, learned from the pupils' hands as the server tagged them, before
    # any window is cut; it touches no box and no tag
    labelled = label_work_areas(circled, pupils, layout, nudge) if work_area and circled else circled
    # the persons' tags carried along their tracks; a session the server did not track has no
    # propagation column
    tracked = track_tags and _tracked(raw)
    features = EventIndex(propagate_track_tags(labelled, layout) if tracked else labelled, instant=True)
    # every tagged gaze point in frame, by camera, for the joint-attention baseline
    gaze_index = gaze_points(features.records, layout)
    # the seats of the participants, learned once from the whole session and from the tags the
    # server gave (read or kept on the track), not the propagated ones, so a track carried to the
    # wrong person cannot move a seat; None without VFA, so its table has no seat trace
    seats = seats_of(raw, participants, layout) if len(features) else None
    # the pupils' seats, learned the same way, name an untagged gaze target at a missing pupil's seat
    if seats is None or not seat_partners:
        pupil_seats = None
    elif set(pupils) <= set(participants):
        pupil_seats = {camera: {tag: seat for tag, seat in here.items() if tag in pupils} for camera, here in seats.items()}
    else:
        pupil_seats = seats_of(raw, pupils, layout)
    # the tracks the server named a pupil on, whose bodies are never someone outside the group
    pupil_tracks = pupil_tracks_of(raw, pupils, layout) if len(features) else None
    actions = EventIndex(events.get(EVENT_TYPE_VFA_ACTION, []), instant=True)
    # the personal microphones and the buckets each wearer won; None for a session without them
    personal = personal_speech(events.get(EVENT_TYPE_ASR_RECOGNITION, []), events.get(EVENT_TYPE_ASR_TRANSCRIPTION, []))
    rows = []
    for index, ws, we in windows(span[0], span[1], window, step):
        row: dict[str, Any] = {'window_index': index, 'window_start': round(ws, 3), 'window_end': round(we, 3)}
        row.update(speech_features(recognition, transcription, ws, we, speakers, personal=personal))
        row.update(space_features(translations, relations, ws, we, participants))
        row.update(body_gaze_features(features, ws, we, participants, layout, pupils=pupils, gaze_index=gaze_index,
                                      work_area=work_area, seats=pupil_seats, nudge=nudge,
                                      pupil_tracks=pupil_tracks))
        if tracked:
            row['n_vfa_propagated'] = _propagated_count(features, ws, we)
        row.update(action_features(actions, ws, we, participants))
        if seats is not None:
            row.update(seat_features(features, ws, we, participants, seats, layout))
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
