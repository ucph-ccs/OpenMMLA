"""The video part of the session report: where each pupil looked (attention) and the 10 s session
timeline the interaction-state estimate lives on, built once per session by the video job.

Both come from the fused window table (openmmla.analytics.fusion.window_features), the same table
`mmla ses-fuse` writes, so the dashboard's numbers are the research pipeline's numbers: gaze shares
with the hand circle remade and tags carried along their tracks, joint attention against the
pair's own rate 20 to 40 s earlier, hands in shoulder widths. The roster (who the pupils are) and
the interaction state (codebook rule R0 with the presence gate) are read off that table by
openmmla.analytics.interaction, as in the classifier studies. R0 is a rule fixed from the codebook's
wording, not a trained or validated model, so the state is labelled an estimate.

The table holds shares per looker but not who looked at whom, so one pass over the stored frame sets
counts the looks between pupils (face and hands, deduplicated per frame set across cameras), which
cameras saw which pupil, and the data-quality counters. That pass first remakes each frame's gaze
targets with the fusion's hand circle (window_features.relabel_hand_circles), as the fused shares and
the live view do, so a look at a partner's hands means the same on both pages. The two still count
differently: the allocation shares come from the fused table, which counts camera-frames and carries
tags along their tracks, while the looks count frame sets with the tags the server gave.

Everything is per 10 s window on the fusion's own grid (anchored at the session's earliest event),
with times as offsets from the session's `t0`, so the analysis page can sum any brushed range from
the per-window arrays without asking the server again.
"""
from __future__ import annotations

import bisect
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Callable

from openmmla.analytics.fusion import window_features as WF
from openmmla.analytics.report.common import jsonable, sort_tags
from openmmla.utils.constants import EVENT_TYPE_VFA_FEATURES

WINDOW = 10.0
# the six categories the dashboard shows, made from the fused gaze labels
CATEGORIES = ('partner_face', 'partner_hands', 'other_people', 'task', 'elsewhere', 'unreadable')
CATEGORY_LABELS = {
    'partner_face': ('partner_face',),
    'partner_hands': ('partner_hands',),
    'other_people': ('other_face', 'other_hands'),
    'task': ('own_hands', 'work_area', 'zone'),
    'elsewhere': ('elsewhere',),
    'unreadable': ('out_of_frame', 'unknown'),
}
LOOK_KINDS = ('partner_face', 'partner_hands')
STATE_LABELS = {'0': 'Individual', '1': 'Social', '2': 'Collaborative', '-2': 'Absent'}
ABSENT = -2
# the progress steps, in order (progress(step, done, total))
STEPS = ('Fusing windows', 'Choosing pupils', 'Estimating interaction state', 'Counting looks', 'Assembling')

Progress = Callable[[str, int, int], Any]


def build_video(events: dict[str, list[dict]], t0: float, t1: float, progress: Progress | None = None,
                session_dir: str | None = None) -> tuple[dict, dict, list[dict]]:
    """(attention_data, timeline_data, window_rows) of a session from its parsed events by type
    (rows as common.fetch or window_features.load_events_from_influx give them; the fusion keys on
    window_start_time, so `_t` or `time` is never read). `session_dir` (artifacts/<sid>) lets a
    manifest's declared pupils name the roster. `progress(step, done, total)` is called at each
    stage (STEPS)."""
    def report(step: str, done: int, total: int) -> None:
        if progress is not None:
            progress(step, done, total)

    report(STEPS[0], 0, 1)
    rows = WF.window_features(events, window=WINDOW, step=WINDOW)
    report(STEPS[0], 1, 1)

    report(STEPS[1], 0, 1)
    table = _table(rows)
    ros = _roster(table, session_dir)
    pupils = sort_tags([str(tag) for tag in ros.kept]) if ros is not None else []
    report(STEPS[1], 1, 1)

    report(STEPS[2], 0, 1)
    state, state_error = _state(table, ros, rows)
    report(STEPS[2], 1, 1)

    starts = _grid(events, len(rows))
    records = events.get(EVENT_TYPE_VFA_FEATURES) or []
    # the hand circle the fusion remade the gaze targets with (window_features' default relabel)
    nudge = WF.vfa_features.HAND_NUDGES[WF.table_hand_circle(records, True)]
    scan = _scan(records, pupils, starts, lambda done, total: report(STEPS[3], done, total), nudge)

    report(STEPS[4], 0, 1)
    if t0 is None:
        # no meta start: the offsets count from the fusion's own first window
        t0 = float(rows[0]['window_start']) if rows else 0.0
    offsets = [round(float(row['window_start']) - t0, 2) for row in rows]
    duration = round(t1 - t0, 2) if t1 is not None else None
    attention = _attention(rows, pupils, scan, t0, duration, offsets)
    timeline = _timeline(rows, pupils, ros, state, state_error, t0, duration, offsets)
    report(STEPS[4], 1, 1)
    return jsonable(attention), jsonable(timeline), rows


# ---- the table, the roster and the state ----

def _table(rows: list[dict]):
    """the rows as the interaction code reads a fused CSV (layout.read_table): numbers as floats,
    NaN where unobserved, the action columns left as text."""
    import pandas as pd
    table = pd.DataFrame(rows)
    for column in table.columns:
        if table[column].dtype == object and not column.endswith('_action'):
            table[column] = pd.to_numeric(table[column], errors='coerce')
    return table


def _roster(table, session_dir: str | None):
    """the session's roster: the pupils a manifest under `session_dir` declares, else rules R1 to R4
    over the table; None for an empty table."""
    if len(table) == 0:
        return None
    from openmmla.analytics.interaction import layout
    if session_dir and os.path.isdir(session_dir):
        try:
            return layout.session_roster(table, session_dir)
        except ValueError:
            # a manifest whose pupils cannot be used (not tag ids, more than three) leaves the rules
            pass
    return layout.roster(table)


def _state(table, ros, rows: list[dict]) -> tuple[dict | None, str | None]:
    """R0 per window (0 individual, 1 social, 2 collaborative), -2 where the presence gate finds
    fewer than two pupils observed, None where neither IPS nor VFA ran (nobody could be observed);
    (None, reason) when it cannot be computed."""
    if ros is None or not rows:
        return None, 'No data in this session.'
    if len(ros.kept) < 2:
        return None, 'Fewer than two pupils in the roster, so there is no interaction to estimate.'
    try:
        from openmmla.analytics.interaction import layout, presence, tabular
        predicted = tabular.rule_a_priori(layout.pooled(layout.window_tokens(table, ros)))
        absent = presence.gate(table, ros)
        values: list[int | None] = []
        for row, label, gated in zip(rows, predicted, absent):
            if not (_count(row, 'n_ips') or _count(row, 'n_vfa_features')):
                values.append(None)
            else:
                values.append(ABSENT if gated else int(label))
    except Exception as error:  # the state is an extra: a failure leaves the rest of the report
        return None, f'{type(error).__name__}: {error}'
    computed = [v for v in values if v is not None]
    if not computed:
        return None, 'No IPS or VFA data in this session, so nobody could be observed.'
    shares = {key: round(sum(1 for v in computed if v == int(key)) / len(computed), 4) for key in STATE_LABELS}
    return {'values': values, 'labels': dict(STATE_LABELS),
            'rule': f'R0 v{tabular.RULE_VERSION} (codebook rule, not a trained model)',
            'shares': shares, 'windows': len(computed)}, None


# ---- the scan of the stored frame sets ----

def _grid(events: dict[str, list[dict]], n: int) -> list[float]:
    """the unrounded window starts of the fusion's grid (window_features cuts its rows from the
    same span and steps), so a frame set falls in exactly the window the fusion counted it in."""
    span = WF.session_span(events)
    if span is None:
        return []
    return [ws for _, ws, _ in WF.windows(span[0], span[1], WINDOW, WINDOW)][:n]


def _frames(record: dict) -> list:
    """the frames of a frame set (one per camera); the features field is parsed already, or JSON
    text when the rows were not deep-parsed."""
    frames = record.get('features')
    if isinstance(frames, str):
        try:
            frames = json.loads(frames)
        except ValueError:
            return []
    return frames if isinstance(frames, list) else []


def _rank(person: dict) -> tuple:
    """which of two bodies wearing one tag in a frame is the pupil: a tag read in the frame (torso
    or box) over one the track remembered, then the pose model's confidence."""
    score = person.get('score')
    return (person.get('tag_match') != 'track', float(score) if isinstance(score, (int, float)) else 0.0)


def _scan(records: list[dict], pupils: list[str], starts: list[float], progress: Callable[[int, int], Any],
          nudge: float | None = None) -> dict:
    """one pass over the frame sets: the looks between pupils per window (face, hands; per frame set,
    so two cameras seeing one look count it once), the frame sets each pupil was seen in, which camera
    saw which pupil, and the quality counters. Each frame set's gaze targets are first made again with
    the hand circle placed by `nudge` (WF.relabel_hand_circles, features.HAND_NUDGE when not given),
    one record at a time so the memory stays flat."""
    n_w, n_p = len(starts), len(pupils)
    column = {tag: i for i, tag in enumerate(pupils)}
    other = n_p
    pupil_set = set(pupils)
    looks = {kind: [[[0] * (n_p + 1) for _ in range(n_p)] for _ in range(n_w)] for kind in LOOK_KINDS}
    seen = [[0] * n_p for _ in range(n_w)]
    camera_frames: Counter = Counter()
    camera_sizes: dict[str, Counter] = defaultdict(Counter)
    camera_seen: dict[str, Counter] = defaultdict(Counter)
    models: Counter = Counter()
    quality = Counter()
    modal, shared = WF.frame_set_layout(records)
    total = len(records)
    every = max(1, total // 20)
    progress(0, total)
    for number, record in enumerate(records, 1):
        if number % every == 0 and number < total:
            progress(number, total)
        model = record.get('pose_model')
        if model:
            models[str(model)] += 1
        try:
            record = WF.relabel_hand_circles([record], nudge)[0]
        except Exception:  # a frame set the relabelling cannot read keeps the server's targets, as live.slim_vfa does
            pass
        frames = _frames(record)
        quality['frame_sets'] += 1
        quality['incomplete'] += modal is not None and len(frames) != modal
        try:
            moment = float(record.get('window_start_time'))
        except (TypeError, ValueError):
            moment = None
        w = bisect.bisect_right(starts, moment) - 1 if moment is not None and starts else -1
        if w < 0 or moment >= starts[w] + WINDOW:
            w = None
        seen_here: set[str] = set()
        looked: set[tuple[str, int, str]] = set()
        for frame, camera in zip(frames, WF.camera_keys(frames, shared)):
            if not isinstance(frame, dict):
                continue
            camera_frames[camera] += 1
            camera_sizes[camera][(frame.get('width'), frame.get('height'))] += 1
            quality['camera_frames'] += 1
            quality['gaze_errors'] += bool(frame.get('gaze_error'))
            worn: Counter = Counter()
            best: dict[str, dict] = {}
            for person in frame.get('persons') or []:
                tag = person.get('tag_id')
                if tag is None:
                    quality['untagged'] += 1
                    continue
                tag = str(tag)
                worn[tag] += 1
                if tag in pupil_set and (tag not in best or _rank(person) > _rank(best[tag])):
                    best[tag] = person
            quality['duplicate_tag_frames'] += any(n > 1 for n in worn.values())
            for tag, person in best.items():
                seen_here.add(tag)
                camera_seen[camera][tag] += 1
                target = (person.get('gaze') or {}).get('target') or {}
                kind = target.get('category')
                if kind not in LOOK_KINDS:
                    continue
                looked_at = target.get('person_id')
                looked_at = None if looked_at is None else str(looked_at)
                if looked_at == tag:
                    continue
                looked.add((tag, column[looked_at] if looked_at in pupil_set else other, kind))
        if w is None:
            continue
        for tag in seen_here:
            seen[w][column[tag]] += 1
        for tag, target, kind in looked:
            looks[kind][w][column[tag]][target] += 1
    progress(total, total)
    return {'looks': looks, 'seen': seen, 'camera_frames': camera_frames, 'camera_sizes': camera_sizes,
            'camera_seen': camera_seen, 'pose_model': models.most_common(1)[0][0] if models else None,
            'quality': quality}


# ---- reading the fused rows ----

def _value(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, str):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _count(row: dict, key: str) -> int:
    value = _value(row, key)
    return int(round(value)) if value is not None else 0


def _pair_values(rows: list[dict], a: str, b: str, feature: str) -> list[float | None]:
    """a pair's fused column per window, whichever order the fusion wrote the pair in."""
    key = f'pair{a}_{b}_{feature}'
    if rows and key not in rows[0]:
        key = f'pair{b}_{a}_{feature}'
    return [_value(row, key) for row in rows]


def _counts(values: list[float | None]) -> list[int]:
    return [int(round(value)) if value is not None else 0 for value in values]


def _weighted(items: list[tuple[float | None, float]]) -> float | None:
    """the mean of the values with their weights, over the ones that exist; None for none."""
    total = sum(weight for value, weight in items if value is not None and weight > 0)
    if total <= 0:
        return None
    return round(sum(value * weight for value, weight in items if value is not None and weight > 0) / total, 4)


def _category_frames(row: dict, tag: str) -> tuple[int, dict[str, int]]:
    """a pupil's camera-frames in the window and how many fell in each dashboard category."""
    frames = _count(row, f'p{tag}_frames')
    counts = {}
    for category, labels in CATEGORY_LABELS.items():
        n = 0
        for label in labels:
            ratio = _value(row, f'p{tag}_gaze_{label}_ratio')
            if ratio is not None and frames:
                n += int(round(ratio * frames))
        counts[category] = n
    return frames, counts


def _pairs(pupils: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(pupils) for b in pupils[i + 1:]]


# ---- the parts ----

def _attention(rows: list[dict], pupils: list[str], scan: dict, t0: float, duration: float | None,
               offsets: list[float]) -> dict:
    """the attention part: gaze allocation (fused), looks between pupils (scan), joint attention,
    hands, camera coverage and data quality, with per-window arrays for range sums."""
    allocation, allocation_windows, hands, unknown_share = {}, {}, {}, {}
    for tag in pupils:
        per_window = {'frames': []}
        per_window.update({category: [] for category in CATEGORIES})
        for row in rows:
            frames, counts = _category_frames(row, tag)
            per_window['frames'].append(frames)
            for category in CATEGORIES:
                per_window[category].append(counts[category])
        allocation_windows[tag] = per_window
        total = sum(per_window['frames'])
        allocation[tag] = {category: (round(sum(per_window[category]) / total, 4) if total else None)
                           for category in CATEGORIES}
        unknown_share[tag] = _weighted([(_value(row, f'p{tag}_gaze_unknown_ratio'), _count(row, f'p{tag}_frames'))
                                        for row in rows])
        steps = [_count(row, f'p{tag}_hand_steps') for row in rows]
        hands[tag] = {'active': _weighted([(_value(row, f'p{tag}_hands_active_ratio'), n) for row, n in zip(rows, steps)]),
                      'still': _weighted([(_value(row, f'p{tag}_hands_still_ratio'), n) for row, n in zip(rows, steps)]),
                      'steps': sum(steps)}

    mutual_gaze, joint_attention, pair_hands = [], [], []
    for a, b in _pairs(pupils):
        frames = _counts(_pair_values(rows, a, b, 'frames'))
        mutual = sum(int(round(ratio * n)) for ratio, n in zip(_pair_values(rows, a, b, 'mutual_gaze_ratio'), frames)
                     if ratio is not None)
        mutual_gaze.append({'a': a, 'b': b, 'frames': mutual})
        # ratio, baseline and excess over the same windows (those with a baseline), so the excess is the
        # ratio above the baseline
        both = [(ratio, base, n) for ratio, base, n in zip(_pair_values(rows, a, b, 'joint_attention_ratio'),
                                                          _pair_values(rows, a, b, 'joint_attention_baseline'), frames)
                if ratio is not None and base is not None and n > 0]
        ratio = _weighted([(r, n) for r, _, n in both])
        baseline = _weighted([(base, n) for _, base, n in both])
        joint_attention.append({'a': a, 'b': b, 'ratio': ratio, 'baseline': baseline,
                                'excess': round(ratio - baseline, 4) if ratio is not None and baseline is not None else None,
                                'windows': len(both), 'frames': sum(n for *_, n in both)})
        steps = _counts(_pair_values(rows, a, b, 'hand_steps'))
        one = _pair_values(rows, a, b, 'one_active_ratio')
        pair_hands.append({
            'a': a, 'b': b,
            'one_active': _weighted(list(zip(one, steps))),
            'both_active': _weighted(list(zip(_pair_values(rows, a, b, 'both_active_ratio'), steps))),
            'both_still': _weighted(list(zip(_pair_values(rows, a, b, 'both_still_ratio'), steps))),
            # the follow share is over the steps with one pupil active and the other still
            'follow': _weighted([(follow, (share or 0.0) * n) for follow, share, n
                                 in zip(_pair_values(rows, a, b, 'follow_ratio'), one, steps)]),
            'close': _weighted(list(zip(_pair_values(rows, a, b, 'hands_close_ratio'), frames))),
            'steps': sum(steps)})

    cameras = sorted(scan['camera_frames'])
    camera_list = []
    for camera in cameras:
        size = scan['camera_sizes'][camera].most_common(1)[0][0] if scan['camera_sizes'][camera] else (None, None)
        camera_list.append({'id': camera, 'frames': scan['camera_frames'][camera],
                            'width': size[0], 'height': size[1]})
    coverage = {'cameras': cameras, 'tags': list(pupils),
                'share': [[round(scan['camera_seen'][camera][tag] / scan['camera_frames'][camera], 4)
                           if scan['camera_frames'][camera] else None for tag in pupils] for camera in cameras],
                'frames': [[scan['camera_seen'][camera][tag] for tag in pupils] for camera in cameras]}

    looks = {'targets': list(pupils) + ['other']}
    for kind, key in (('partner_face', 'face'), ('partner_hands', 'hands')):
        windows = scan['looks'][kind]
        looks[key] = {'total': [[sum(window[i][j] for window in windows) for j in range(len(pupils) + 1)]
                                for i in range(len(pupils))],
                      'windows': windows}
    looks['seen'] = {'total': [sum(window[i] for window in scan['seen']) for i in range(len(pupils))],
                     'windows': scan['seen']}

    quality = scan['quality']
    camera_frames = quality['camera_frames']
    return {
        't0': t0, 'duration': duration, 'step': WINDOW, 't': offsets,
        'pupils': list(pupils),
        'cameras': camera_list,
        'camera_coverage': coverage,
        'categories': list(CATEGORIES),
        'allocation': allocation,
        'allocation_windows': allocation_windows,
        'looks': looks,
        'mutual_gaze': mutual_gaze,
        'joint_attention': joint_attention,
        'hands': hands,
        'pair_hands': pair_hands,
        'quality': {'frame_sets': quality['frame_sets'], 'incomplete': quality['incomplete'],
                    'camera_frames': camera_frames, 'gaze_errors': quality['gaze_errors'],
                    'unknown_share': unknown_share,
                    'untagged_per_frame': round(quality['untagged'] / camera_frames, 2) if camera_frames else None,
                    'duplicate_tag_frames': quality['duplicate_tag_frames'],
                    'pose_model': scan['pose_model']},
    }


def _timeline(rows: list[dict], pupils: list[str], ros, state: dict | None, state_error: str | None,
              t0: float, duration: float | None, offsets: list[float]) -> dict:
    """the timeline part: the roster, the state per window and the per-window lanes of speech,
    each pupil and each pair, with which modality ran in each window."""
    asr = [_count(row, 'n_asr_recognition') > 0 for row in rows]
    heard = [ran or _count(row, 'n_asr_transcription') > 0 for row, ran in zip(rows, asr)]
    ips = [_count(row, 'n_ips') > 0 for row in rows]
    vfa = [_count(row, 'n_vfa_features') > 0 for row in rows]

    tags = {}
    for tag in pupils:
        present, social, dominant, active, steps, path = [], [], [], [], [], []
        for row, ips_ran, vfa_ran in zip(rows, ips, vfa):
            positioned = (_value(row, f'p{tag}_present_ratio') or 0.0) > 0
            frames, counts = _category_frames(row, tag)
            if positioned or _count(row, f'p{tag}_frame_sets') > 0:
                present.append(1)
            else:
                present.append(0 if ips_ran or vfa_ran else None)
            social.append(round((counts['partner_face'] + counts['partner_hands']) / frames, 4) if frames else None)
            top = max(CATEGORIES, key=lambda category: counts[category])
            dominant.append(top if counts[top] > 0 else None)
            active.append(_value(row, f'p{tag}_hands_active_ratio'))
            steps.append(_count(row, f'p{tag}_hand_steps'))
            path.append(_value(row, f'p{tag}_path_m'))
        # hand_steps weighs hands_active over a range as the session value is weighed
        tags[tag] = {'present': present, 'social_gaze': social, 'dominant': dominant,
                     'hands_active': active, 'hand_steps': steps, 'path_m': path}

    pairs = {}
    for a, b in _pairs(pupils):
        frames = _counts(_pair_values(rows, a, b, 'frames'))
        mutual = [int(round(ratio * n)) if ratio is not None else None
                  for ratio, n in zip(_pair_values(rows, a, b, 'mutual_gaze_ratio'), frames)]
        pairs[f'{a}|{b}'] = {'dist': _pair_values(rows, a, b, 'dist_mean_m'),
                             'ja': _pair_values(rows, a, b, 'joint_attention_ratio'),
                             'ja_excess': _pair_values(rows, a, b, 'joint_attention_excess'),
                             'face_mutual': mutual,
                             'hands_close': _pair_values(rows, a, b, 'hands_close_ratio'),
                             # frames weighs hands_close over a range as the session value is weighed
                             'frames': frames}

    roster = {'kept': list(pupils), 'dropped': {}, 'group_size': 0, 'source': None, 'degraded': False, 'cover': {}}
    if ros is not None:
        roster = {'kept': list(pupils),
                  'dropped': {str(tag): reason for tag, reason in sorted(ros.dropped.items())},
                  'group_size': int(ros.group_size), 'source': ros.source, 'degraded': bool(ros.degraded),
                  'cover': {str(tag): cover for tag, cover in sorted(ros.cover.items())}}
    return {
        't0': t0, 'duration': duration, 'step': WINDOW, 't': offsets,
        'roster': roster,
        'state': state,
        'state_error': state_error,
        'speech': {'ratio': [_value(row, 'speech_ratio') if ran else None for row, ran in zip(rows, asr)],
                   'words': [_count(row, 'words') if ran else None for row, ran in zip(rows, heard)],
                   'switches': [_value(row, 'dia_switches') for row in rows]},
        'tags': tags,
        'pairs': pairs,
        'coverage': {'asr': [int(v) for v in asr], 'ips': [int(v) for v in ips], 'vfa': [int(v) for v in vfa]},
    }
