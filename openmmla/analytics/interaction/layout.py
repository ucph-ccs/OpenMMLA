"""The input layout of the 10 s interaction classifier: a fused table (mmla ses-fuse) becomes, per
window, a group token, three availability bits and two small sets, one of at most three persons and
one of at most three pairs. The 118-column pooled view every tabular model reads is made from those
sets, so every model sees the same information.

Tag ids only group a table's columns and run the roster. No feature name, value or slot carries a
tag id, a session id or a column position, so a model cannot learn which badge, seat or session it
is looking at. A slot is a person the roster kept, in descending coverage; nothing learned depends
on the slot order, which only names the persons A, B and C in a text description.

Missing is not zero. A value no sensor could observe is NaN before scaling and 0 after it, and its
mask (a 0/1 column of the token) says which of the two it is; an observed 0 stays 0. So imputation
never turns "IPS was off" into "far apart", or "no transcription chunk" into "no words".

Layout version 2: a present_ratio of 0 in a window a camera saw the person in (and not as the
duplicate gate's copy) is unobserved, not an observed 0. IPS lost the badge, not the person.
Version 1 read it as 0. The seat trace of window_features (p<t>_untagged_at_seat_ratio,
n_untagged_at_seats) is auxiliary: parse_columns sets it apart, no token reads it, and the presence
gate reads it where the table has it.

Layout version 3 (2026-09-23, before any label was read): partner gaze is in-group (another pupil
of the session); the faces and hands of anyone else are other_face and other_hands (mask m_other),
an `elsewhere` gaze inside the camera's work area is work_area (mask m_wa, on where most of the
person's gaze frames had a ready area), and joint attention above the pair's own rate 20-40 s
earlier is joint_attention_excess (mask m_jexcess). The lag column joint_attention_excess_max
replaces joint_attention_ratio_max. A table fused before the split has no p<t>_gaze_other_face_ratio:
its m_other and m_wa are off, its excess is NaN, and its partner gaze still counts every other
person. data_checks' fusion_check lists it as split: false; it is not refused, as a table fused
before the camera fix is not.

Layout version 4 (2026-09-24, before any label was read): the hands in body units. The fusion's
hand columns follow each pupil's wrists in the image, in their own shoulder widths
(window_features.wrist_moves): the share of 1 s steps with the hands active or still, the wrist
speed in shoulder widths, and the hands of a body outside the group near theirs (masks m_hands
and m_ohands); per pair, the steps with one active and the other still, both active or both still,
and how often the still one's gaze was on the active one's hands, given three such steps (m_steps,
m_follow), and the hand distance in shoulder widths with the share within one hand length (m_hand).
The frame-width wrist speed and hand distances are dropped: a pupil near the camera, or a camera
nearer the group or with a narrower field of view, moves them. A table fused before the hand
columns has none of these: its masks are off.

Tables fused before the camera fix have no frame-set or camera counts (p<t>_frame_sets,
p<t>_cameras, pair<a>_<b>_frame_sets, n_vfa_cameras). Their frame counts stand in for them there,
and a second camera inflates those: the seen share is clipped at 1 and the switch rate runs over
interleaved frames. A re-fused table uses the real counts with no change here.

Names: a value is called after the table feature it comes from, without the tag, with `log_` in
front when it is log-transformed (`p5_yaw_std` -> `yaw_std`, `p5_wrist_speed_sw` ->
`log_wrist_speed_sw`); the values the layout derives have names of their own (seen_share,
switch_rate, known_share, the gaze shares of readable gaze, face_any, co_seen_share). A pooled
column is `<value>_<min|mean|max>` for a person or pair value and the value's own name for a group
one; POOLED_COLUMNS lists all 118, and a lag column is `<pooled column>_<suffix>` (LAGS).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from openmmla.services.vfa.work_area import WORK_AREA_READY_MIN

LAYOUT_VERSION = 4
# the IPS trust bound: a higher tag id is a mis-decoded badge
MAX_TAG = 12
N_SLOTS = 3
PAIR_INDEX = ((0, 1), (0, 2), (1, 2))
MODALITIES = ('speech', 'space', 'body_gaze')
# a gaze share is read only where at least this much of the person's gaze could be placed
MIN_KNOWN_SHARE = 0.05
# two skeletons whose hands are this close (frame widths) on average are one person seen twice
DUPLICATE_HAND_WIDTH = 0.01
# the wrist speed is logged above this floor, in shoulder widths a second: under half a still arm's
# median wrist jitter (0.024-0.027 a step) and under every setup's 10th percentile of a window's mean
# speed (0.036-0.061 on the 20 replayed sessions), so the log spreads the still windows without running
# off to minus infinity
WRIST_SW_FLOOR = 1e-2
CLIP = 5.0
# a session's own statistics need this many observed values; before that the global ones serve
MIN_SESSION_VALUES = 30
# below this an IQR is treated as none (a value that is mostly one number, such as overlap)
SPREAD_FLOOR = 1e-9

PERSON_RE = re.compile(r'^p(\d+)_(.+)$')
PAIR_RE = re.compile(r'^pair(\d+)_(\d+)_(.+)$')
SPEAKER_RE = re.compile(r'^spk_.+_ratio$')

_COUNTER = ('a coverage counter: a mask or a normaliser only, never a feature, because the camera count '
            'is a session fingerprint and a posture proxy')
# the seat trace of window_features (untagged bodies at the seats): read by the presence gate, never a model input
AUXILIARY_RE = re.compile(r'^(p\d+_untagged_at_seat_ratio|n_untagged_at_seats)$')
_AUXILIARY = ('an auxiliary presence trace (untagged bodies at the seats) for the presence gate, never a feature: '
              'it is outside the pre-registered feature set')
# the table columns no model reads as a feature, by pattern, with the reason
DROPPED = {
    r'^spk_.+_ratio$': 'named after the group; equals speech_ratio',
    r'^n_speakers_named$': 'a 0/1 copy of speech > 0',
    r'^(n_spurts|mean_spurt_seconds)$': 'artifacts of the 30 s transcription chunk cap',
    r'^p\d+_yaw_mean$': 'its sign depends on which side of the camera the seat is',
    r'^p\d+_wrist_speed$': "in frame widths a second: the camera's distance and field of view move it (a pupil near "
                           'the camera moves faster); p<t>_wrist_speed_sw, in shoulder widths a second, replaces it',
    r'^pair\d+_\d+_hand_dist_(min|mean)$': "in frame widths: the camera's distance and field of view move it; "
                                          'hand_dist_sw_* replace it (the duplicate-skeleton gate still reads hand_dist_mean)',
    r'^p\d+_gaze_zone_ratio$': '0 everywhere (no zones configured); the automatic work area is p<t>_gaze_work_area_ratio',
    r'^p\d+_work_area_ready_ratio$': 'a coverage counter: the m_wa mask only, never a feature',
    r'^p\d+_in_group$': "the fusion's pupil set: data_checks' fusion_check only",
    r'^pair\d+_\d+_joint_attention_baseline$': 'the proximity baseline of joint attention: the raw share and the excess carry it',
    r'^p[\w-]+_words$': "a pupil's words from their worn microphone (the words it led on, or those in buckets it won; "
                        "p<t>_vote_words the bucket vote's beside the first): only sessions with worn microphones "
                        'have it (none of the test sessions, whose audio is the group microphone alone)',
    r'^vote_words$': "the worn microphones' words by the bucket vote, beside `words` in a session without a group "
                     'microphone whose words are decided one by one: a comparison, never a feature',
    r'^pair\d+_\d+_face_(ab|ba)_ratio$': 'replaced by face_any = max(ab, ba), since ab and ba follow the arbitrary tag order',
    # face_both = min(ab, ba) is never built: > 0 in 0-0.6 % of windows
    r'^pair\d+_\d+_face_mutual_ratio$': 'near-dead (> 0 in 0.06 % of IPS pair-windows); its support is in data_checks.json',
    r'^pair\d+_\d+_mutual_gaze_ratio$': 'near-dead (> 0 in 3.8 % of co-visible pair-windows, mean 0.006); its support is in '
                                         'data_checks.json',
    r'^(n_asr_recognition|n_asr_transcription|n_ips|n_ips_relation|n_vfa_features|n_vfa_angles|n_vfa_cameras'
    r'|n_vfa_incomplete|n_vfa_propagated|n_vfa_seat_partners)$': _COUNTER,
    r'^p\d+_(cameras|frames|frame_sets|hand_steps)$': _COUNTER,
    r'^pair\d+_\d+_(frames|frame_sets|hand_steps)$': _COUNTER,
    # the reserved block `semantic`, not built: it joins MODALITIES with --with-actions once the VLM runs
    r'^(n_vfa_action|p\d+_action|pair\d+_\d+_co_manipulating)$': 'empty in this batch; the reserved semantic block',
    AUXILIARY_RE.pattern: _AUXILIARY,
    r'^(window_index|window_start|window_end)$': 'an index, not a feature',
}
# the table columns the tokens are made from
USED = (
    r'^(speech_ratio|silence_ratio|words|dia_speakers|dia_switches|dia_overlap_ratio|dia_share_entropy)$',
    r'^p\d+_(present_ratio|path_m|yaw_abs_mean|yaw_std|wrist_speed_sw|hands_active_ratio|hands_still_ratio'
    r'|other_hands_near_ratio|gaze_switches'
    r'|gaze_(partner_face|partner_hands|other_face|other_hands|own_hands|work_area|elsewhere|out_of_frame|unknown)_ratio)$',
    r'^pair\d+_\d+_(dist_mean_m|dist_min_m|hand_dist_sw_min|hand_dist_sw_mean|hands_close_ratio|gaze_dist_mean'
    r'|joint_attention_ratio|joint_attention_excess|one_active_ratio|both_active_ratio|both_still_ratio|follow_ratio)$',
)

# a token value: its name, what it is made from, the transform (before scaling), the scaling tag
# ('g' global over the training sessions, 's' within the session), its mask and its modality
Value = namedtuple('Value', 'name source transform tag mask modality')

# the lexicon's task_word_share (+lexicon, deferred) would be an eighth group value here, with one
# more pooled speech column
GROUP_VALUES = (
    Value('speech_ratio', 'speech_ratio', 'none', 'g', 'm_asr', 'speech'),
    Value('silence_ratio', 'silence_ratio', 'none', 'g', 'm_asr', 'speech'),
    Value('log_words', 'words', 'log1p', 's', 'm_transcription', 'speech'),
    Value('dia_speakers', 'dia_speakers', 'none', 'g', 'm_dia', 'speech'),
    Value('log_dia_switches', 'dia_switches', 'log1p', 'g', 'm_dia', 'speech'),
    Value('dia_overlap_ratio', 'dia_overlap_ratio', 'none', 'g', 'm_dia', 'speech'),
    Value('dia_share_entropy', 'dia_share_entropy', 'none', 'g', 'm_dia', 'speech'),
)
# the gaze shares of readable gaze: partner_* is another pupil, other_* anyone else (the teacher,
# another group), work_area an elsewhere inside the camera's work area
GAZE_SHARES = ('partner_face', 'partner_hands', 'other_face', 'other_hands', 'own_hands', 'work_area', 'elsewhere',
               'out_of_frame')
# the mask of each gaze share: the split shares have their own, off on a table fused before the split
SHARE_MASKS = {'other_face': 'm_other', 'other_hands': 'm_other', 'work_area': 'm_wa'}
PERSON_VALUES = (
    Value('present_ratio', 'p<t>_present_ratio', 'none', 'g', 'm_ips', 'space'),
    Value('log_path_m', 'p<t>_path_m', 'log1p', 's', 'm_path', 'space'),
    Value('seen_share', 'p<t>_frame_sets / n_vfa_features', 'clip to [0, 1]', 'g', 'm_vfa', 'body_gaze'),
    Value('yaw_abs_mean', 'p<t>_yaw_abs_mean', '/ 90', 'g', 'm_yaw', 'body_gaze'),
    Value('yaw_std', 'p<t>_yaw_std (within camera)', '/ 90', 'g', 'm_yaw', 'body_gaze'),
    # the hands in the body's own units, shoulder widths, so one scale serves every camera
    Value('log_wrist_speed_sw', 'p<t>_wrist_speed_sw', 'log(x + 1e-2)', 'g', 'm_hands', 'body_gaze'),
    Value('hands_active_ratio', 'p<t>_hands_active_ratio', 'none', 'g', 'm_hands', 'body_gaze'),
    Value('hands_still_ratio', 'p<t>_hands_still_ratio', 'none', 'g', 'm_hands', 'body_gaze'),
    Value('other_hands_near_ratio', 'p<t>_other_hands_near_ratio', 'none', 'g', 'm_ohands', 'body_gaze'),
    Value('switch_rate', 'p<t>_gaze_switches / max(p<t>_frames / p<t>_cameras - 1, 1)', 'none', 'g', 'm_vfa', 'body_gaze'),
) + tuple(Value(share, f'p<t>_gaze_{share}_ratio / known_share', 'none', 'g', SHARE_MASKS.get(share, 'm_gaze'),
                'body_gaze') for share in GAZE_SHARES) + (
    Value('known_share', '1 - p<t>_gaze_unknown_ratio', 'none', 'g', 'm_vfa', 'body_gaze'),
)
PAIR_VALUES = (
    Value('dist_mean_m', 'pair<a>_<b>_dist_mean_m', 'none', 'g', 'm_dist', 'space'),
    Value('dist_min_m', 'pair<a>_<b>_dist_min_m', 'none', 'g', 'm_dist', 'space'),
    Value('face_any', 'max(pair<a>_<b>_face_ab_ratio, pair<a>_<b>_face_ba_ratio)', 'none', 'g', 'm_face', 'space'),
    Value('co_seen_share', 'pair<a>_<b>_frame_sets / n_vfa_features', 'clip to [0, 1]', 'g', 'm_covis', 'body_gaze'),
    # the hand distance in shoulder widths: one scale serves every camera
    Value('hand_dist_sw_min', 'pair<a>_<b>_hand_dist_sw_min', 'none', 'g', 'm_hand', 'body_gaze'),
    Value('hand_dist_sw_mean', 'pair<a>_<b>_hand_dist_sw_mean', 'none', 'g', 'm_hand', 'body_gaze'),
    Value('hands_close_ratio', 'pair<a>_<b>_hands_close_ratio', 'none', 'g', 'm_hand', 'body_gaze'),
    # frame widths depend on the camera, so the gaze distance is scaled within the session
    Value('gaze_dist_mean', 'pair<a>_<b>_gaze_dist_mean', 'none', 's', 'm_gazepair', 'body_gaze'),
    Value('joint_attention_ratio', 'pair<a>_<b>_joint_attention_ratio', 'none', 'g', 'm_gazepair', 'body_gaze'),
    # the joint attention above the pair's own rate 20-40 s earlier (the proximity baseline)
    Value('joint_attention_excess', 'pair<a>_<b>_joint_attention_excess', 'none', 'g', 'm_jexcess', 'body_gaze'),
    # the pair's hands step by step: one working while the other is still, and whether the still one watches
    Value('one_active_ratio', 'pair<a>_<b>_one_active_ratio', 'none', 'g', 'm_steps', 'body_gaze'),
    Value('both_active_ratio', 'pair<a>_<b>_both_active_ratio', 'none', 'g', 'm_steps', 'body_gaze'),
    Value('both_still_ratio', 'pair<a>_<b>_both_still_ratio', 'none', 'g', 'm_steps', 'body_gaze'),
    Value('follow_ratio', 'pair<a>_<b>_follow_ratio', 'none', 'g', 'm_follow', 'body_gaze'),
)
GROUP_MASKS = ('m_asr', 'm_transcription', 'm_dia')
PERSON_MASKS = ('m_ips', 'm_path', 'm_vfa', 'm_yaw', 'm_hands', 'm_ohands', 'm_gaze', 'm_other', 'm_wa')
PAIR_MASKS = ('m_dist', 'm_face', 'm_covis', 'm_hand', 'm_gazepair', 'm_jexcess', 'm_steps', 'm_follow')
MASK_MODALITY = {'m_asr': 'speech', 'm_transcription': 'speech', 'm_dia': 'speech',
                 'm_ips': 'space', 'm_path': 'space', 'm_dist': 'space', 'm_face': 'space',
                 'm_vfa': 'body_gaze', 'm_yaw': 'body_gaze', 'm_hands': 'body_gaze', 'm_ohands': 'body_gaze',
                 'm_gaze': 'body_gaze', 'm_other': 'body_gaze', 'm_wa': 'body_gaze',
                 'm_covis': 'body_gaze', 'm_hand': 'body_gaze', 'm_gazepair': 'body_gaze', 'm_jexcess': 'body_gaze',
                 'm_steps': 'body_gaze', 'm_follow': 'body_gaze'}
# the availability bits, one per modality in MODALITIES order
AVAILABILITY = ('speech_ran', 'ips_ran', 'vfa_ran')

# the token columns: values, then masks (then group_size / 3 in G)
G_COLUMNS = tuple(v.name for v in GROUP_VALUES) + GROUP_MASKS + ('group_size',)
P_COLUMNS = tuple(v.name for v in PERSON_VALUES) + PERSON_MASKS
Q_COLUMNS = tuple(v.name for v in PAIR_VALUES) + PAIR_MASKS


def _modality_index() -> dict:
    """where each modality sits in the tokens: its value and mask columns in G, P and Q and its
    availability bit (what modality dropout zeroes to fake an outage)."""
    index = {}
    for position, modality in enumerate(MODALITIES):
        entry = {'avail': position}
        for part, values, masks in (('G', GROUP_VALUES, GROUP_MASKS), ('P', PERSON_VALUES, PERSON_MASKS),
                                    ('Q', PAIR_VALUES, PAIR_MASKS)):
            entry[part] = [i for i, v in enumerate(values) if v.modality == modality] + \
                [len(values) + i for i, m in enumerate(masks) if MASK_MODALITY[m] == modality]
        index[modality] = entry
    return index


MODALITY_INDEX = _modality_index()

_STATS = ('min', 'mean', 'max')


def _pooled_names(values, modality: str) -> tuple:
    return tuple(f'{v.name}_{stat}' for v in values if v.modality == modality for stat in _STATS)


# the pooled view, block by block (10 + 18 + 89 + 1 = 118 columns)
POOLED_BLOCKS = {
    'speech': tuple(v.name for v in GROUP_VALUES) + GROUP_MASKS,
    'space': _pooled_names(PERSON_VALUES, 'space') + _pooled_names(PAIR_VALUES, 'space')
    + ('ips_ran', 'share_present', 'share_pairs_dist'),
    'body_gaze': _pooled_names(PERSON_VALUES, 'body_gaze') + _pooled_names(PAIR_VALUES, 'body_gaze')
    + ('vfa_ran', 'share_seen', 'share_gaze_known', 'share_pairs_covis', 'share_pairs_gaze'),
    'roster': ('group_size',),
}
POOLED_COLUMNS = tuple(column for block in POOLED_BLOCKS.values() for column in block)

# the pooled columns the tabular models also see at their neighbours' windows
LAG_COLUMNS = ('speech_ratio', 'log_words', 'log_dia_switches', 'dia_overlap_ratio',
               'present_ratio_mean', 'dist_mean_m_min', 'face_any_max',
               'partner_face_mean', 'partner_hands_mean', 'own_hands_mean', 'joint_attention_excess_max',
               'log_wrist_speed_sw_mean')
# per temporal mode, each derived column as (suffix, first offset, last offset): one offset is a
# neighbour's value, a range the nanmean over the neighbours in it that exist
LAGS = {
    'T0': (),
    'T1c': (('lag1', -1, -1), ('lag2', -2, -2), ('past5', -4, 0)),
    'T2': (('lag1', -1, -1), ('lead1', 1, 1), ('around5', -2, 2)),
}


# ---- the table ----

def read_table(path) -> pd.DataFrame:
    """a fused table (.csv, else JSON lines, as mmla ses-fuse writes it) in window order, numbers
    as floats; the action columns stay text."""
    path = str(path)
    table = pd.read_csv(path) if path.lower().endswith('.csv') else pd.read_json(path, lines=True)
    for column in table.columns:
        if table[column].dtype == object and not column.endswith('_action'):
            table[column] = pd.to_numeric(table[column], errors='coerce')
    return table.sort_values('window_start', kind='stable').reset_index(drop=True)


def parse_columns(columns) -> dict:
    """the table's columns grouped by the patterns of window_features: per person
    ({tag: {feature: column}}), per pair ({(a, b): {feature: column}}), the named-speaker ratios,
    the auxiliary seat-trace columns (never a person's feature, so a tag named only there gets no
    roster entry), and the rest (group-level values and counters). Tags are ints."""
    persons, pairs = defaultdict(dict), defaultdict(dict)
    speakers, group, auxiliary = [], [], []
    for column in columns:
        pair = PAIR_RE.match(column)
        person = PERSON_RE.match(column)
        if AUXILIARY_RE.match(column):
            auxiliary.append(column)
        elif pair:
            pairs[(int(pair.group(1)), int(pair.group(2)))][pair.group(3)] = column
        elif person:
            persons[int(person.group(1))][person.group(2)] = column
        elif SPEAKER_RE.match(column):
            speakers.append(column)
        else:
            group.append(column)
    return {'persons': dict(sorted(persons.items())), 'pairs': dict(sorted(pairs.items())),
            'speakers': speakers, 'group': group, 'auxiliary': auxiliary}


def column_status(column: str) -> str:
    """'kept' for a column the tokens are made from, the reason for a dropped one, 'unknown' for a
    column this layout does not know (a new fusion column shows up here, not silently)."""
    for pattern, reason in DROPPED.items():
        if re.match(pattern, column):
            return reason
    if any(re.match(pattern, column) for pattern in USED):
        return 'kept'
    return 'unknown'


def _column(table: pd.DataFrame, name: str) -> np.ndarray:
    """a column as floats; one the table does not have is all NaN."""
    if name not in table.columns:
        return np.full(len(table), np.nan)
    return pd.to_numeric(table[name], errors='coerce').to_numpy(dtype=float)


def _frame_sets(table: pd.DataFrame, tag) -> np.ndarray:
    """the frame sets a person appears in (any camera). A table fused before the camera fix has
    only the frame count, which a second camera doubles; it stands in there."""
    name = f'p{tag}_frame_sets'
    return _column(table, name if name in table.columns else f'p{tag}_frames')


def _cameras(table: pd.DataFrame, tag) -> np.ndarray:
    """the cameras that saw a person. Before the camera fix every frame went into one sequence per
    angle, so 1 is the count its switch numbers were made over."""
    name = f'p{tag}_cameras'
    return _column(table, name) if name in table.columns else np.ones(len(table))


def _pair_column(table: pd.DataFrame, a, b, feature: str) -> np.ndarray:
    """a pair's column whichever order the table wrote the pair in (a -tags run keeps the order
    given)."""
    for x, y in ((a, b), (b, a)):
        if f'pair{x}_{y}_{feature}' in table.columns:
            return _column(table, f'pair{x}_{y}_{feature}')
    return np.full(len(table), np.nan)


def _pair_frame_sets(table: pd.DataFrame, a, b) -> np.ndarray:
    """the frame sets in which a pair is seen together; before the camera fix, the frame count."""
    for x, y in ((a, b), (b, a)):
        if f'pair{x}_{y}_frame_sets' in table.columns:
            return _column(table, f'pair{x}_{y}_frame_sets')
    return _pair_column(table, a, b, 'frames')


# ---- the roster ----

@dataclass(eq=False)
class Roster:
    """who gets a slot: `kept` in slot order, `dropped` tag -> reason, `group_size` (persons who
    were there, observed or not, at most 3), `degraded` (someone was never observed), how often
    the duplicate-skeleton gate masked each kept person's camera values, and where the persons came
    from (`source`: 'rules', 'tags' for a command's -tags, 'manifest' for the session's declared
    pupils)."""
    kept: list
    dropped: dict
    group_size: int
    degraded: bool
    gate_counts: dict
    cover: dict = field(default_factory=dict)
    source: str = 'rules'

    def record(self) -> dict:
        """the roster as roster.json holds it (tag ids as text)."""
        return {'kept': [str(tag) for tag in self.kept], 'dropped': {str(t): r for t, r in self.dropped.items()},
                'group_size': self.group_size, 'degraded': self.degraded,
                'gate_counts': {str(t): n for t, n in self.gate_counts.items()},
                'cover': {str(t): c for t, c in self.cover.items()}, 'source': self.source}


def _cover(table: pd.DataFrame, tag) -> dict:
    """the share of windows IPS positioned the person in, a camera saw them in, and either."""
    if len(table) == 0:
        return {'ips': 0.0, 'vfa': 0.0, 'cover': 0.0}
    with np.errstate(invalid='ignore'):
        ips = _column(table, f'p{tag}_present_ratio') > 0
        vfa = _frame_sets(table, tag) > 0
    return {'ips': round(float(ips.mean()), 4), 'vfa': round(float(vfa.mean()), 4),
            'cover': round(float((ips | vfa).mean()), 4)}


def roster(table: pd.DataFrame, max_tag: int = MAX_TAG, vfa_only_cover: float = 0.25, min_cover: float = 0.05,
           max_group: int = N_SLOTS, tags=None, source: str = 'tags') -> Roster:
    """the persons of a session, by rules applied in order: R1 a tag above the IPS trust bound is
    a mis-decoded id; R2 a tag only cameras saw, in under a quarter of the windows, and IPS never
    positioned, in a session where IPS positioned someone, is a mis-decoded low id; R3 a person
    observed in under 5 % of the windows gets no slot but still counts in the group size (and the
    session is flagged degraded); R4 at most three, the best-covered. Slots follow descending
    cover, tag id breaking ties. A `tags` roster replaces the rules: the command's -tags, or the
    pupils a session's manifest declares (`source` 'manifest', see session_roster). Every given
    person keeps a slot, however rarely observed; one the table never names was there but never
    observed (counted in the group size, the session flagged degraded)."""
    parsed = parse_columns(table.columns)
    all_tags = sorted(set(parsed['persons']) | {tag for pair in parsed['pairs'] for tag in pair})
    cover = {tag: _cover(table, tag) for tag in all_tags}

    def order(tag):
        return (-cover[tag]['cover'], tag)

    dropped = {}
    if tags is not None:
        wanted = [int(tag) for tag in tags]
        if len(wanted) > max_group:
            raise ValueError(f"a roster holds at most {max_group} persons, got {len(wanted)}")
        why = 'not a pupil of this session (manifest pupils)' if source == 'manifest' else 'not in the given roster (-tags)'
        for tag in all_tags:
            if tag not in wanted:
                dropped[tag] = why
        kept = sorted((tag for tag in wanted if tag in cover), key=order)
        # a given person the table never names was there but never observed
        unobserved = [tag for tag in wanted if tag not in cover]
        group_size = len(wanted)
    else:
        candidates = []
        for tag in all_tags:
            if tag > max_tag:
                dropped[tag] = f'R1 untrusted id: above {max_tag}, the IPS trust bound'
            else:
                candidates.append(tag)
        if any(cover[tag]['ips'] > 0 for tag in candidates):
            for tag in list(candidates):
                # camera-only: seen at least once, so a person nobody observed is left to R3
                if cover[tag]['ips'] == 0 and 0 < cover[tag]['vfa'] < vfa_only_cover:
                    dropped[tag] = f"R2 camera-only and never positioned (camera cover {cover[tag]['vfa']:.2f})"
                    candidates.remove(tag)
        unobserved = [tag for tag in candidates if cover[tag]['cover'] < min_cover]
        for tag in unobserved:
            dropped[tag] = f"R3 never observed (cover {cover[tag]['cover']:.2f}); counted in the group size"
        ranked = sorted((tag for tag in candidates if tag not in unobserved), key=order)
        for tag in ranked[max_group:]:
            dropped[tag] = f"R4 beyond the group size of {max_group} (cover {cover[tag]['cover']:.2f})"
        kept = ranked[:max_group]
        group_size = min(len(kept) + len(unobserved), max_group)
    gate = duplicate_gate(table, kept)
    return Roster(kept=kept, dropped=dropped, group_size=group_size, degraded=bool(unobserved),
                  gate_counts={tag: int(gate[:, i].sum()) for i, tag in enumerate(kept)}, cover=cover,
                  source='rules' if tags is None else source)


def declared_pupils(session_dir) -> list | None:
    """the pupils a session's manifest declares (`pupils`, tag ids as text, written by mmla ses-tidy
    --pupils), or None when it declares none, has no manifest or the manifest cannot be read. A
    session declares them when its video shows who the pupils were and the roster rules would guess
    wrong (a spare badge on the table read for a few seconds is no third pupil). Raises ValueError
    when an entry is not a tag id."""
    try:
        declared = json.loads((Path(session_dir) / 'manifest.json').read_text(encoding='utf-8')).get('pupils')
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(declared, (str, int)):
        declared = str(declared).split(',')
    if not isinstance(declared, list):
        return None
    pupils = [str(tag).strip() for tag in declared if str(tag).strip()]
    if any(not tag.isdigit() for tag in pupils):
        raise ValueError(f"{Path(session_dir) / 'manifest.json'}: pupils are tag ids, not {declared!r}")
    return pupils or None


def session_roster(table: pd.DataFrame, session_dir=None, tags=None) -> Roster:
    """the roster of a session: a command's own `tags` (-tags) first, then the pupils its manifest
    declares (declared_pupils), else the rules of roster()."""
    if tags is not None:
        return roster(table, tags=tags)
    declared = declared_pupils(session_dir) if session_dir is not None else None
    if declared is not None:
        return roster(table, tags=declared, source='manifest')
    return roster(table)



def roster_digest(kept, group_size) -> str:
    """the sha256 of who holds a slot, in slot order, and the group size: everything a Jev state
    says about who was there. ses-jev writes it on each line of a session's Jev map, so a map asked
    about another roster (the session has since declared its pupils) is known to be stale."""
    body = json.dumps({'kept': [str(tag) for tag in kept], 'group_size': int(group_size)},
                      sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def rules_roster_digest(table: pd.DataFrame, ros: Roster | None = None) -> str:
    """the roster digest of the rules roster of `table` (`ros` itself when it is one): what a Jev
    map line without a roster digest was asked about, since ses-jev read the rules roster before
    a session could declare its pupils."""
    if ros is None or ros.source != 'rules':
        ros = roster(table)
    return roster_digest(ros.kept, ros.group_size)


MIN_TWO_VISIBLE = 0.3  # S1: the share of windows two roster persons must be observed together in


def two_visible(table: pd.DataFrame, kept) -> np.ndarray:
    """per window, whether at least two of the kept persons were observed: positioned by IPS or
    seen by a camera."""
    if len(kept) < 2 or len(table) == 0:
        return np.zeros(len(table), dtype=bool)
    with np.errstate(invalid='ignore'):
        seen = [(_column(table, f'p{tag}_present_ratio') > 0) | (_frame_sets(table, tag) > 0) for tag in kept]
    return np.sum(seen, axis=0) >= 2


def session_inclusion(table: pd.DataFrame, ros: Roster, min_two_visible: float = MIN_TWO_VISIBLE) -> tuple[bool, str]:
    """S1, the inclusion rule of a session, fixed before any label is read: its roster keeps at
    least two persons, and two of them are observed together in at least `min_two_visible` of its
    windows. (included, the reason). A session left out trains nothing and is scored nowhere."""
    if len(ros.kept) < 2:
        return False, f"S1 the roster keeps {len(ros.kept)} person(s); two are needed"
    share = float(two_visible(table, ros.kept).mean()) if len(table) else 0.0
    if share < min_two_visible:
        return False, f"S1 two persons observed together in {share:.2f} of the windows, under {min_two_visible:.2f}"
    return True, f"S1 two persons observed together in {share:.2f} of the windows"


def duplicate_gate(table: pd.DataFrame, kept) -> np.ndarray:
    """(windows, kept persons): where a person's camera values are masked because the pose model
    gave one body two tags. Two kept persons seen in the same frames with their hands under
    DUPLICATE_HAND_WIDTH apart on average are one skeleton; the one with fewer frames is the copy
    (on a tie, the later slot)."""
    gate = np.zeros((len(table), len(kept)), dtype=bool)
    with np.errstate(invalid='ignore'):
        for i, j in combinations(range(len(kept)), 2):
            a, b = kept[i], kept[j]
            fired = (_pair_frame_sets(table, a, b) > 0) & (_pair_column(table, a, b, 'hand_dist_mean') < DUPLICATE_HAND_WIDTH)
            later_is_copy = _column(table, f'p{b}_frames') <= _column(table, f'p{a}_frames')
            gate[:, j] |= fired & later_is_copy
            gate[:, i] |= fired & ~later_is_copy
    return gate


# ---- the tokens ----

@dataclass(eq=False)
class Tokens:
    """one session's tokens, window by window: G (T, 11) the group values, masks and group size / 3;
    avail (T, 3) whether speech, IPS and VFA ran; P (T, 3, 28) the person slots and P_exists (T, 3);
    Q (T, 3, 22) the pair slots, pair_index (3, 2) the slots of each pair, and Q_exists (T, 3);
    window_start (T,); empty (T,) no speech, nobody positioned, nobody seen. `positioned` (T, 3)
    keeps whether IPS placed each slot (share_present needs it after scaling), `window_index` (T,)
    the table's grid index. Unscaled, an unobserved value is NaN; scaled, it is 0."""
    G: np.ndarray
    avail: np.ndarray
    P: np.ndarray
    P_exists: np.ndarray
    Q: np.ndarray
    Q_exists: np.ndarray
    pair_index: np.ndarray
    window_start: np.ndarray
    empty: np.ndarray
    positioned: np.ndarray = None
    window_index: np.ndarray = None
    scaled: bool = False

    def __len__(self) -> int:
        return len(self.window_start)


def _settle(values: np.ndarray, masks: np.ndarray, spec, mask_names) -> tuple[np.ndarray, np.ndarray]:
    """values and masks as a token holds them: a mask is on only where every value under it is a
    number, and a value whose mask is off is NaN, so a mask never claims what is not there."""
    for k, name in enumerate(mask_names):
        covered = [i for i, v in enumerate(spec) if v.mask == name]
        masks[:, k] &= np.isfinite(values[:, covered]).all(axis=1)
    for i, v in enumerate(spec):
        values[~masks[:, mask_names.index(v.mask)], i] = np.nan
    return values, masks


def _ips_missed(present: np.ndarray, frame_sets: np.ndarray, gated: np.ndarray) -> np.ndarray:
    """where IPS gave a person 0 while a camera saw them (and not as the copy the duplicate gate
    masks): the badge was lost, not the person, so that 0 is not an observation."""
    with np.errstate(invalid='ignore'):
        return (present == 0) & (frame_sets > 0) & ~gated


def _person(table, tag, ips_ran, n_vfa, gated):
    present = _column(table, f'p{tag}_present_ratio')
    frame_sets = _frame_sets(table, tag)
    frames = _column(table, f'p{tag}_frames')
    known = 1.0 - _column(table, f'p{tag}_gaze_unknown_ratio')
    # the share of the person's gaze frames on a camera whose work area was ready (NaN before the split)
    ready = _column(table, f'p{tag}_work_area_ready_ratio')
    with np.errstate(divide='ignore', invalid='ignore'):
        seen = (frame_sets > 0) & ~gated
        values = np.column_stack([
            present,
            np.log1p(_column(table, f'p{tag}_path_m')),
            np.clip(frame_sets / n_vfa, 0.0, 1.0),
            _column(table, f'p{tag}_yaw_abs_mean') / 90.0,
            _column(table, f'p{tag}_yaw_std') / 90.0,
            np.log(_column(table, f'p{tag}_wrist_speed_sw') + WRIST_SW_FLOOR),
            _column(table, f'p{tag}_hands_active_ratio'),
            _column(table, f'p{tag}_hands_still_ratio'),
            _column(table, f'p{tag}_other_hands_near_ratio'),
            _column(table, f'p{tag}_gaze_switches') / np.maximum(frames / _cameras(table, tag) - 1.0, 1.0),
            *[_column(table, f'p{tag}_gaze_{share}_ratio') / known for share in GAZE_SHARES],
            known,
        ])
        # present_ratio 0 is an observation where IPS ran and no camera saw the person; where one did,
        # IPS lost the badge and the 0 is unobserved. Camera values need the person seen; the hand values
        # need steps (or frames) with the wrists and shoulders seen, which _settle reads off their NaN
        readable = seen & (known >= MIN_KNOWN_SHARE)
        # the other shares are NaN on a table fused before the split, which _settle turns off; the work
        # area needs most of the person's gaze frames on a camera whose area was ready
        masks = np.column_stack([ips_ran & ~_ips_missed(present, frame_sets, gated), ips_ran, seen, seen, seen, seen,
                                 readable, readable, readable & (ready >= WORK_AREA_READY_MIN)])
    values, masks = _settle(values, masks, PERSON_VALUES, PERSON_MASKS)
    return values, masks, present, frame_sets


def _pair(table, a, b, both_present, n_vfa, gated):
    co_seen = _pair_frame_sets(table, a, b)
    with np.errstate(divide='ignore', invalid='ignore'):
        covisible = (co_seen > 0) & ~gated
        values = np.column_stack([
            _pair_column(table, a, b, 'dist_mean_m'),
            _pair_column(table, a, b, 'dist_min_m'),
            # symmetric in a <-> b: ab and ba follow the tag order, which means nothing
            np.fmax(_pair_column(table, a, b, 'face_ab_ratio'), _pair_column(table, a, b, 'face_ba_ratio')),
            np.clip(co_seen / n_vfa, 0.0, 1.0),
            _pair_column(table, a, b, 'hand_dist_sw_min'),
            _pair_column(table, a, b, 'hand_dist_sw_mean'),
            _pair_column(table, a, b, 'hands_close_ratio'),
            _pair_column(table, a, b, 'gaze_dist_mean'),
            _pair_column(table, a, b, 'joint_attention_ratio'),
            _pair_column(table, a, b, 'joint_attention_excess'),
            _pair_column(table, a, b, 'one_active_ratio'),
            _pair_column(table, a, b, 'both_active_ratio'),
            _pair_column(table, a, b, 'both_still_ratio'),
            _pair_column(table, a, b, 'follow_ratio'),
        ])
        # distances and facing exist only when both were positioned; the excess has its own mask, off
        # where the baseline had too few comparisons (or the table has none); the step shares need steps
        # both had a status in, and following three steps with one active and the other still
        # (window_features.FOLLOW_MIN_STEPS)
        masks = np.column_stack([both_present, both_present, covisible, covisible, covisible, covisible, covisible,
                                 covisible])
    return _settle(values, masks, PAIR_VALUES, PAIR_MASKS)


def window_tokens(table: pd.DataFrame, roster: Roster, speech_measured: bool = True) -> Tokens:
    """the session's tokens, unscaled (log transforms and ratios applied, NaN where unobserved).
    `speech_measured` False marks a session whose microphone missed talk the audio has (the
    low-speech check): its speech values become unobserved instead of silence."""
    T = len(table)
    speech_ratio = _column(table, 'speech_ratio')
    with np.errstate(invalid='ignore'):
        asr_ran = (_column(table, 'n_asr_recognition') > 0) & bool(speech_measured)
        transcribed = _column(table, 'n_asr_transcription') > 0
        ips_ran = _column(table, 'n_ips') > 0
        n_vfa = _column(table, 'n_vfa_features')
        vfa_ran = n_vfa > 0

        g_values = np.column_stack([
            speech_ratio, _column(table, 'silence_ratio'), np.log1p(_column(table, 'words')),
            _column(table, 'dia_speakers'), np.log1p(_column(table, 'dia_switches')),
            _column(table, 'dia_overlap_ratio'), _column(table, 'dia_share_entropy')])
    # words is 0 in a window no chunk covered; m_dia is on where the diarization values exist
    g_masks = np.column_stack([asr_ran, transcribed, np.ones(T, dtype=bool)])
    g_values, g_masks = _settle(g_values, g_masks, GROUP_VALUES, GROUP_MASKS)
    G = np.column_stack([g_values, g_masks.astype(float), np.full(T, roster.group_size / N_SLOTS)])

    kept = list(roster.kept)[:N_SLOTS]
    gate = duplicate_gate(table, kept)
    P = np.full((T, N_SLOTS, len(P_COLUMNS)), np.nan)
    P[:, :, len(PERSON_VALUES):] = 0.0
    P_exists = np.zeros((T, N_SLOTS))
    positioned = np.zeros((T, N_SLOTS), dtype=bool)
    seen_anyone = np.zeros(T, dtype=bool)
    for slot, tag in enumerate(kept):
        values, masks, present, frame_sets = _person(table, tag, ips_ran, n_vfa, gate[:, slot])
        P[:, slot, :] = np.column_stack([values, masks.astype(float)])
        P_exists[:, slot] = 1.0
        with np.errstate(invalid='ignore'):
            positioned[:, slot] = present > 0
            seen_anyone |= frame_sets > 0

    Q = np.full((T, N_SLOTS, len(Q_COLUMNS)), np.nan)
    Q[:, :, len(PAIR_VALUES):] = 0.0
    Q_exists = np.zeros((T, N_SLOTS))
    for q, (i, j) in enumerate(PAIR_INDEX):
        if j >= len(kept):
            continue
        # a masked copy's pairs are masked with it: they would measure one body against itself
        values, masks = _pair(table, kept[i], kept[j], positioned[:, i] & positioned[:, j], n_vfa, gate[:, i] | gate[:, j])
        Q[:, q, :] = np.column_stack([values, masks.astype(float)])
        Q_exists[:, q] = 1.0

    with np.errstate(invalid='ignore'):
        spoken = asr_ran & (speech_ratio > 0)
    window_index = _column(table, 'window_index') if 'window_index' in table.columns else np.arange(T, dtype=float)
    return Tokens(G=G, avail=np.column_stack([asr_ran, ips_ran, vfa_ran]).astype(float),
                  P=P, P_exists=P_exists, Q=Q, Q_exists=Q_exists, pair_index=np.array(PAIR_INDEX, dtype=int),
                  window_start=np.round(_column(table, 'window_start'), 3),
                  empty=~spoken & ~positioned.any(axis=1) & ~seen_anyone,
                  positioned=positioned, window_index=window_index.astype(int))


# ---- scaling ----

_PARTS = (('G', GROUP_VALUES, GROUP_MASKS), ('P', PERSON_VALUES, PERSON_MASKS), ('Q', PAIR_VALUES, PAIR_MASKS))


def _observed(tokens: Tokens, part: str) -> tuple[np.ndarray, np.ndarray]:
    """a part's values as (windows, slots, values) with NaN wherever the value was not observed
    (no slot, mask off, not a number), and that observed mask. G is one slot."""
    spec, masks = {'G': (GROUP_VALUES, GROUP_MASKS), 'P': (PERSON_VALUES, PERSON_MASKS),
                   'Q': (PAIR_VALUES, PAIR_MASKS)}[part]
    if part == 'G':
        array, exists = tokens.G[:, None, :], np.ones((len(tokens.G), 1))
    else:
        array, exists = (tokens.P, tokens.P_exists) if part == 'P' else (tokens.Q, tokens.Q_exists)
    n = len(spec)
    values = array[..., :n].astype(float)
    mask_of = [n + masks.index(v.mask) for v in spec]
    observed = (array[..., mask_of] > 0) & (exists[..., None] > 0) & np.isfinite(values)
    values[~observed] = np.nan
    return values, observed


def _robust(x: np.ndarray) -> tuple[float, float]:
    """median and IQR; a value that is mostly one number has no IQR, and dividing by it would throw
    its rare other values to the clip, so its standard deviation serves, then 1."""
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    spread = q75 - q25
    if spread <= SPREAD_FLOOR:
        spread = float(np.std(x))
    return float(q50), float(spread) if spread > SPREAD_FLOOR else 1.0


@dataclass(eq=False)
class Stats:
    """the [g] statistics: per part ('G', 'P', 'Q') the median and spread of every value, fitted on
    the outer-training sessions' non-empty windows."""
    center: dict
    spread: dict
    n_windows: int = 0
    n_sessions: int = 0

    def record(self) -> dict:
        """the statistics by value name, for config.json."""
        out = {}
        for part, spec, _ in _PARTS:
            for k, v in enumerate(spec):
                out[v.name] = {'median': float(self.center[part][k]), 'spread': float(self.spread[part][k])}
        return {'values': out, 'n_windows': self.n_windows, 'n_sessions': self.n_sessions}


def fit_global_stats(tokens_list) -> Stats:
    """the [g] statistics from the given sessions' unscaled tokens (the caller passes the
    outer-training sessions only): per value, pooled over slots, the observed values of every
    non-empty window. Label-free; the empty windows are left out because they are trivially easy
    and, 70 % of one session, would set the scale."""
    tokens_list = list(tokens_list)
    center, spread = {}, {}
    for part, spec, _ in _PARTS:
        chunks = [np.empty((0, len(spec)))]
        for tokens in tokens_list:
            if tokens.scaled:
                raise ValueError("fit the statistics on unscaled tokens")
            values, _ = _observed(tokens, part)
            chunks.append(values[~tokens.empty].reshape(-1, len(spec)))
        stacked = np.concatenate(chunks)
        center[part], spread[part] = np.zeros(len(spec)), np.ones(len(spec))
        for k in range(len(spec)):
            x = stacked[:, k][np.isfinite(stacked[:, k])]
            if len(x):
                center[part][k], spread[part][k] = _robust(x)
    return Stats(center=center, spread=spread, n_windows=int(sum((~t.empty).sum() for t in tokens_list)),
                 n_sessions=len(tokens_list))


def _global(stats: Stats | None, part: str, k: int, name: str) -> tuple[float, float]:
    if stats is None:
        raise ValueError(f"scaling {name} needs the global statistics (fit_global_stats)")
    return stats.center[part][k], stats.spread[part][k]


def _session_scale(x: np.ndarray, stats, part, k, name) -> tuple[float, float]:
    """a session's own median and spread of a value, over its windows and slots; the global ones
    when the session observed it fewer than MIN_SESSION_VALUES times."""
    observed = x[np.isfinite(x)]
    if len(observed) < MIN_SESSION_VALUES:
        return _global(stats, part, k, name)
    return _robust(observed)


def _running_scale(x: np.ndarray, stats, part, k, name) -> tuple[np.ndarray, np.ndarray]:
    """per window, the median and spread of the values the session observed up to and including
    it (never after): the online normaliser, on the global statistics until MIN_SESSION_VALUES
    values have been seen."""
    center, spread = _global(stats, part, k, name)
    centers, spreads = np.full(len(x), center), np.full(len(x), spread)
    seen: list[float] = []
    for t in range(len(x)):
        new = x[t][np.isfinite(x[t])]
        if len(new):
            seen.extend(new.tolist())
            if len(seen) >= MIN_SESSION_VALUES:
                center, spread = _robust(np.asarray(seen))
        if len(seen) >= MIN_SESSION_VALUES:
            centers[t], spreads[t] = center, spread
    return centers, spreads


def scale(tokens: Tokens, stats: Stats | None = None, mode: str = 'offline', scheme: str = 'mix') -> Tokens:
    """the tokens scaled: (x - median) / spread, clipped to [-5, 5], and 0 wherever the value was
    not observed (its mask says so). A [g] value uses the global statistics, a [s] value its
    session's own, which are label-free and so legitimate on a test session too; `mode` 'causal'
    makes those running (only windows up to the current one). `scheme` 'g' or 's' puts every value
    under one tag (the scaling ablation)."""
    if mode not in ('offline', 'causal'):
        raise ValueError(f"unknown scaling mode {mode!r}: 'offline' or 'causal'")
    if scheme not in ('mix', 'g', 's'):
        raise ValueError(f"unknown scaling scheme {scheme!r}: 'mix', 'g' or 's'")
    if tokens.scaled:
        raise ValueError("these tokens are scaled already")
    arrays = {'G': tokens.G.copy(), 'P': tokens.P.copy(), 'Q': tokens.Q.copy()}
    for part, spec, _ in _PARTS:
        values, observed = _observed(tokens, part)
        z = np.zeros_like(values)
        for k, value in enumerate(spec):
            tag = value.tag if scheme == 'mix' else scheme
            x = values[..., k]
            if not np.isfinite(x).any():
                continue
            if tag == 'g':
                center, spread = _global(stats, part, k, value.name)
            elif mode == 'offline':
                center, spread = _session_scale(x, stats, part, k, value.name)
            else:
                centers, spreads = _running_scale(x, stats, part, k, value.name)
                center, spread = centers[:, None], spreads[:, None]
            with np.errstate(invalid='ignore'):
                z[..., k] = np.clip((x - center) / spread, -CLIP, CLIP)
        z[~observed] = 0.0
        n = len(spec)
        if part == 'G':
            arrays['G'][:, :n] = z[:, 0, :]
        else:
            arrays[part][..., :n] = z
            # a slot nobody holds is all zeros
            arrays[part][(tokens.P_exists if part == 'P' else tokens.Q_exists) == 0] = 0.0
    return replace(tokens, G=arrays['G'], P=arrays['P'], Q=arrays['Q'], scaled=True)


# ---- the pooled view ----

def _pool(x: np.ndarray, stat: str) -> np.ndarray:
    """min, mean or max over the slots that hold a number; NaN where none does."""
    finite = np.isfinite(x)
    count = finite.sum(axis=1)
    if stat == 'min':
        out = np.where(finite, x, np.inf).min(axis=1)
    elif stat == 'max':
        out = np.where(finite, x, -np.inf).max(axis=1)
    else:
        out = np.where(finite, x, 0.0).sum(axis=1) / np.maximum(count, 1)
    return np.where(count > 0, out, np.nan)


def _share(hit: np.ndarray, exists: np.ndarray, ran: np.ndarray) -> np.ndarray:
    """the share of the existing slots a condition holds for; NaN where the modality did not run
    or no slot exists (not observed, rather than none)."""
    n = exists.sum(axis=1)
    return np.where((n > 0) & (ran > 0), (hit & exists).sum(axis=1) / np.maximum(n, 1), np.nan)


def pooled(tokens: Tokens) -> pd.DataFrame:
    """the 118-column pooled view (POOLED_COLUMNS), indexed by window_index: the group values and
    masks, each person value's (min, mean, max) over the slots that observed it, each pair
    value's over the pairs, the availability bits, the shares of slots and pairs that observed a
    modality, and the group size. For at most three slots (min, mean, max) gives back the sorted
    values (the middle one is 3 mean - min - max), so only the binding of values to one person
    is lost. Unscaled tokens give values in the units of the transforms (what the a-priori rule's
    thresholds read); scaled tokens the scaled values. Unobserved is NaN either way."""
    columns = {}
    g_values, _ = _observed(tokens, 'G')
    for k, v in enumerate(GROUP_VALUES):
        columns[v.name] = g_values[:, 0, k]
    for k, name in enumerate(GROUP_MASKS):
        columns[name] = tokens.G[:, len(GROUP_VALUES) + k]
    for part, spec in (('P', PERSON_VALUES), ('Q', PAIR_VALUES)):
        values, _ = _observed(tokens, part)
        for k, v in enumerate(spec):
            for stat in _STATS:
                columns[f'{v.name}_{stat}'] = _pool(values[..., k], stat)

    p_exists, q_exists = tokens.P_exists > 0, tokens.Q_exists > 0
    ips_ran, vfa_ran = tokens.avail[:, 1], tokens.avail[:, 2]

    def person_mask(name):
        return tokens.P[..., len(PERSON_VALUES) + PERSON_MASKS.index(name)] > 0

    def pair_mask(name):
        return tokens.Q[..., len(PAIR_VALUES) + PAIR_MASKS.index(name)] > 0

    positioned = tokens.positioned
    if positioned is None:
        # tokens built by hand: the unscaled present_ratio still says it, a scaled one no longer does
        if tokens.scaled:
            raise ValueError("scaled tokens need `positioned` for share_present")
        with np.errstate(invalid='ignore'):
            positioned = person_mask('m_ips') & (tokens.P[..., P_COLUMNS.index('present_ratio')] > 0)
    columns['ips_ran'] = ips_ran
    columns['share_present'] = _share(positioned, p_exists, ips_ran)
    columns['share_pairs_dist'] = _share(pair_mask('m_dist'), q_exists, ips_ran)
    columns['vfa_ran'] = vfa_ran
    columns['share_seen'] = _share(person_mask('m_vfa'), p_exists, vfa_ran)
    columns['share_gaze_known'] = _share(person_mask('m_gaze'), p_exists, vfa_ran)
    columns['share_pairs_covis'] = _share(pair_mask('m_covis'), q_exists, vfa_ran)
    columns['share_pairs_gaze'] = _share(pair_mask('m_gazepair'), q_exists, vfa_ran)
    columns['group_size'] = np.round(tokens.G[:, -1] * N_SLOTS)
    index = tokens.window_index if tokens.window_index is not None else np.arange(len(tokens))
    frame = pd.DataFrame(columns, index=pd.Index(index, name='window_index'))
    return frame[list(POOLED_COLUMNS)]


def _lag_base(column: str) -> str:
    for suffixes in LAGS.values():
        for suffix, _, _ in suffixes:
            if column.endswith('_' + suffix) and column[:-len(suffix) - 1] in LAG_COLUMNS:
                return column[:-len(suffix) - 1]
    return column


def block_columns(pooled: pd.DataFrame, with_group_size: bool = True) -> dict:
    """the columns of each modality's expert: its block of the pooled view and the lag columns of
    its key columns, in the frame's order; with `with_group_size` each block also has
    group_size (the roster block), which every expert sees."""
    block_of = {column: modality for modality in MODALITIES for column in POOLED_BLOCKS[modality]}
    blocks = {modality: [] for modality in MODALITIES}
    for column in pooled.columns:
        if column == 'group_size':
            if with_group_size:
                for modality in MODALITIES:
                    blocks[modality].append(column)
            continue
        modality = block_of.get(_lag_base(column))
        if modality is not None:
            blocks[modality].append(column)
    return blocks


def temporal_context(pooled: pd.DataFrame, mode: str = 'T0') -> pd.DataFrame:
    """the pooled view of one session (every window of its grid, in order, coded or not) with its
    key columns at neighbouring windows: T0 none, T1c (causal) the two windows before and the
    mean of the last five, T2 (centred) the windows either side and the mean of the five around.
    A neighbour past the session's edge is NaN, and a mean is over the neighbours that exist."""
    if mode not in LAGS:
        raise ValueError(f"unknown temporal mode {mode!r}: one of {', '.join(LAGS)}")
    out = pooled.copy()
    if not LAGS[mode]:
        return out
    key = pooled[list(LAG_COLUMNS)].to_numpy(dtype=float)
    T, reach = len(key), 4
    padded = np.vstack([np.full((reach, key.shape[1]), np.nan), key, np.full((reach, key.shape[1]), np.nan)])
    derived = {}
    for suffix, first, last in LAGS[mode]:
        # rows reach + t + first .. reach + t + last of the padded array are windows t + first .. t + last
        stack = np.stack([padded[reach + offset: reach + offset + T] for offset in range(first, last + 1)])
        finite = np.isfinite(stack)
        count = finite.sum(axis=0)
        mean = np.where(finite, stack, 0.0).sum(axis=0) / np.maximum(count, 1)
        derived[suffix] = np.where(count > 0, mean, np.nan)
    lags = {f'{column}_{suffix}': derived[suffix][:, k]
            for k, column in enumerate(LAG_COLUMNS) for suffix, _, _ in LAGS[mode]}
    return pd.concat([out, pd.DataFrame(lags, index=pooled.index)], axis=1)


# ---- data checks (data_checks.json) ----

def generic_name(column: str) -> str:
    """a column with its tag ids written as * (p5_yaw_std -> p*_yaw_std), so sessions compare."""
    pair = PAIR_RE.match(column)
    if pair:
        return f'pair*_{pair.group(3)}'
    person = PERSON_RE.match(column)
    if person:
        return f'p*_{person.group(2)}'
    return 'spk_*_ratio' if SPEAKER_RE.match(column) else column


def support_table(tables: dict) -> pd.DataFrame:
    """per column (tags as *) and session, the share of windows in which any instance of the
    column is above 0 (text columns: present), with the column's status (kept, or why it is
    dropped); face_both and face_any, which are built from ab and ba, are added."""
    shares = {}
    for session, table in tables.items():
        instances = defaultdict(list)
        for column in table.columns:
            instances[generic_name(column)].append(column)
        row = {}
        for name, columns in instances.items():
            hits = np.zeros(len(table), dtype=bool)
            for column in columns:
                if table[column].dtype == object:
                    hits |= table[column].notna().to_numpy() & (table[column].astype(str) != '')
                else:
                    with np.errstate(invalid='ignore'):
                        hits |= _column(table, column) > 0
            row[name] = float(hits.mean()) if len(table) else np.nan
        both = np.zeros(len(table), dtype=bool)
        either = np.zeros(len(table), dtype=bool)
        for (a, b) in parse_columns(table.columns)['pairs']:
            ab, ba = _pair_column(table, a, b, 'face_ab_ratio'), _pair_column(table, a, b, 'face_ba_ratio')
            with np.errstate(invalid='ignore'):
                both |= np.fmin(ab, ba) > 0
                either |= np.fmax(ab, ba) > 0
        if len(table):
            row['pair*_face_both'], row['pair*_face_any'] = float(both.mean()), float(either.mean())
        shares[session] = row
    frame = pd.DataFrame(shares)
    frame.insert(0, 'status', [_generic_status(name) for name in frame.index])
    return frame.sort_index()


def _generic_status(name: str) -> str:
    if name in ('pair*_face_both', 'pair*_face_any'):
        return 'derived from face_ab and face_ba'
    return column_status(name.replace('pair*', 'pair0_1').replace('p*', 'p0').replace('spk_*', 'spk_0'))


def low_speech(tables: dict, threshold: float = 0.2) -> dict:
    """the sessions whose microphone heard speech in less than `threshold` of the time on average
    (over the windows ASR ran in): the ones to listen to before trusting their silence."""
    out = {}
    for session, table in tables.items():
        with np.errstate(invalid='ignore'):
            ran = _column(table, 'n_asr_recognition') > 0
        speech = _column(table, 'speech_ratio')[ran]
        speech = speech[np.isfinite(speech)]
        if len(speech) and speech.mean() < threshold:
            out[session] = round(float(speech.mean()), 4)
    return out


def listening_sample(table: pd.DataFrame, n: int = 10, seed: int = 0) -> list:
    """the starts of `n` random windows in which ASR ran and heard no speech, in time order, to
    play through ses-code: talk the ASR missed means the session's speech is not measured."""
    with np.errstate(invalid='ignore'):
        silent = (_column(table, 'n_asr_recognition') > 0) & (_column(table, 'speech_ratio') == 0)
    starts = np.round(_column(table, 'window_start')[silent], 3)
    if len(starts) > n:
        starts = np.random.default_rng(seed).choice(starts, size=n, replace=False)
    return sorted(float(s) for s in starts)


def camera_check(table: pd.DataFrame, roster: Roster) -> dict:
    """after the re-fuse: per kept person, the median wrist speed (shoulder widths), yaw spread and
    switch rate in windows one camera saw them in against windows two or more did, and the ratio
    (two / one), which should be near 1 once each camera is followed on its own. Unavailable for a
    table fused before the fix (no p<t>_cameras)."""
    if not any(f'p{tag}_cameras' in table.columns for tag in roster.kept):
        return {'available': False, 'reason': 'the table has no p<t>_cameras (fused before the camera fix)'}
    persons, ratios = {}, defaultdict(list)
    for slot, tag in enumerate(roster.kept):
        cameras = _cameras(table, tag)
        frames = _column(table, f'p{tag}_frames')
        with np.errstate(divide='ignore', invalid='ignore'):
            features = {
                'wrist_speed_sw': _column(table, f'p{tag}_wrist_speed_sw'),
                'yaw_std': _column(table, f'p{tag}_yaw_std'),
                'switch_rate': _column(table, f'p{tag}_gaze_switches') / np.maximum(frames / cameras - 1.0, 1.0),
            }
        entry = {}
        for name, x in features.items():
            one, two = x[(cameras == 1) & np.isfinite(x)], x[(cameras >= 2) & np.isfinite(x)]
            m1 = float(np.median(one)) if len(one) else None
            m2 = float(np.median(two)) if len(two) else None
            ratio = m2 / m1 if m1 and m2 is not None else None
            entry[name] = {'one_camera': m1, 'two_cameras': m2, 'ratio': ratio, 'windows': [len(one), len(two)]}
            if ratio is not None:
                ratios[name].append(ratio)
        persons[str(tag)] = entry
    return {'available': True, 'persons': persons,
            'median_ratio': {name: float(np.median(r)) for name, r in ratios.items()}}


def seat_check(table: pd.DataFrame, roster: Roster) -> dict:
    """the present_ratio rule and the seat trace per kept person: `present_unobserved`, the windows
    whose present_ratio 0 a camera made unobserved; and where the table has the trace,
    `seat_windows` (share of windows a camera holding the person's seat ran in), `at_seat_windows`
    (share with an untagged body at the seat while the tag was missing) and `at_seat_unobserved`
    (that share among the windows no sensor observed the person in; None when there are none),
    with the mean of n_untagged_at_seats."""
    gate = duplicate_gate(table, roster.kept)
    trace = any(f'p{tag}_untagged_at_seat_ratio' in table.columns for tag in roster.kept)
    persons = {}
    for i, tag in enumerate(roster.kept):
        present = _column(table, f'p{tag}_present_ratio')
        frame_sets = _frame_sets(table, tag)
        entry = {'present_unobserved': int(_ips_missed(present, frame_sets, gate[:, i]).sum())}
        if trace:
            ratio = _column(table, f'p{tag}_untagged_at_seat_ratio')
            with np.errstate(invalid='ignore'):
                unobserved = ~((present > 0) | (frame_sets > 0))
                at = ratio > 0
            if len(table):
                entry['seat_windows'] = round(float(np.isfinite(ratio).mean()), 4)
                entry['at_seat_windows'] = round(float(at.mean()), 4)
            else:
                entry['seat_windows'] = entry['at_seat_windows'] = None
            entry['at_seat_unobserved'] = round(float(at[unobserved].mean()), 4) if unobserved.any() else None
        persons[str(tag)] = entry
    n = _column(table, 'n_untagged_at_seats')
    return {'trace': trace, 'persons': persons,
            'n_untagged_at_seats_mean': round(float(np.nanmean(n)), 4) if trace and np.isfinite(n).any() else None}


def fusion_check(table: pd.DataFrame, roster: Roster) -> dict:
    """what the fusion of 2026-09-23 and 2026-09-24 gave the table: `split` (it has the other_face
    columns; a table fused before has not, and its partner gaze counts every other person), `hands`
    (it has the body-normalised hand columns of layout version 4), `pupils` (the tags the
    fusion took for pupils, p<t>_in_group = 1; None before the split), `roster_in_group` (every kept
    person is one of them), `work_area_ready` (per kept person, the mean share of gaze frames on a
    camera whose work area was ready), `joint_baseline_share` (the share of the co-visible
    roster pair-windows with a joint-attention excess) and `seat_partner_frames` (the gaze frames
    whose untagged target the fusion named a pupil by the seat, n_vfa_seat_partners summed; None
    without the column)."""
    parsed = parse_columns(table.columns)
    split = any('gaze_other_face_ratio' in features for features in parsed['persons'].values())
    in_group = {tag: _column(table, column) for tag, features in parsed['persons'].items()
                for name, column in features.items() if name == 'in_group'}
    pupils = sorted(str(tag) for tag, x in in_group.items() if (x == 1).any()) if in_group else None
    ready = {}
    for tag in roster.kept:
        x = _column(table, f'p{tag}_work_area_ready_ratio')
        ready[str(tag)] = round(float(np.nanmean(x)), 4) if np.isfinite(x).any() else None
    covisible, finite = 0, 0
    for a, b in combinations(roster.kept, 2):
        with np.errstate(invalid='ignore'):
            together = _pair_frame_sets(table, a, b) > 0
        covisible += int(together.sum())
        finite += int((together & np.isfinite(_pair_column(table, a, b, 'joint_attention_excess'))).sum())
    hands = any('hands_active_ratio' in features for features in parsed['persons'].values())
    return {'split': bool(split), 'hands': bool(hands), 'pupils': pupils,
            'roster_in_group': None if pupils is None else {str(tag) for tag in roster.kept} <= set(pupils),
            'work_area_ready': ready,
            'joint_baseline_share': round(finite / covisible, 4) if covisible else None,
            'seat_partner_frames': int(np.nansum(_column(table, 'n_vfa_seat_partners')))
            if 'n_vfa_seat_partners' in table.columns else None}


def data_checks(tables: dict, rosters: dict, threshold: float = 0.2, n: int = 10, seed: int = 0) -> dict:
    """what data_checks.json holds, before any training: the low-speech sessions with windows to
    listen to, the support of every kept and dropped column per session, the camera check, the
    present_ratio rule and the seat trace, and the fusion check (in-group gaze, work area,
    joint-attention baseline)."""
    low = low_speech(tables, threshold)
    return {
        'layout_version': LAYOUT_VERSION,
        'low_speech': {session: {'mean_speech_ratio': mean, 'listen': listening_sample(tables[session], n, seed)}
                       for session, mean in low.items()},
        'support': json_ready(support_table(tables)),
        'camera_check': {session: camera_check(table, rosters[session]) for session, table in tables.items()
                         if session in rosters},
        'seat_check': {session: seat_check(table, rosters[session]) for session, table in tables.items()
                       if session in rosters},
        'fusion_check': {session: fusion_check(table, rosters[session]) for session, table in tables.items()
                         if session in rosters},
    }


def json_ready(frame: pd.DataFrame) -> dict:
    """a frame as {row: {column: value}}, NaN as None."""
    return {str(index): {str(k): (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
            for index, row in frame.to_dict(orient='index').items()}
