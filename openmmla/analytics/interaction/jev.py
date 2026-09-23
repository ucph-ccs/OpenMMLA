"""The zero-shot baseline of the 10 s interaction classifier: Jev reads a plain-text description of
a window's sensor features and picks individual, social or collaborative without seeing a label.

The description (the state) is written from the per-slot numbers of the fused table before any
scaling. Persons are A, B and C in slot order and pairs are named by their two letters. Every
number is rounded and given a word: physical bins for badge distances in metres, and tertiles
fitted once on the dev sessions' windows (label-free, never a test session) for speech, words,
head turn, hand movement and hand distance. A modality that did not run is said in words ("not
measured in this window", "C not in view"), never as 0. No tag id, session id, date, task name or
transcript text reaches the state, so Jev cannot tell sessions apart and nothing said leaves the
machine. The question is the coder's own codebook (`openmmla.commands.ses.code.CODEBOOK`), so the
coder and Jev read one text.

Calls are content-addressed: a request's cache file is named by the sha256 of its body, so a rerun
is free, a re-fused table (a different state) misses the cache by itself, and identical states
(the empty windows) are paid once. The key travels in the Authorization header only; it is never
written to the cache, a log, a URL or a printed request. Each session's map of answers records the
fused table's and the template's (the tertile edges') digests, and a later run from the same table
and template adds to it rather than replacing it, so a --limit run never cuts a full map down.

The input is a slot table (`slot_table`), built here from a fused table and the roster's kept tags
in slot order, so this module needs nothing from the layout but those tags. The criteria-order and
rerun checks of the design are deferred: the criteria keep CODEBOOK's order, and a rerun is a run
without the cache.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from openmmla.commands.ses.code import CODEBOOK

# where Jev is asked: OpenRouter serves TypeSafe's model at its own price through a dedicated
# endpoint (the chat-completions endpoint does not), and any other host that speaks TypeSafe's
# /decide shape can be named with --provider url; jevtypesafeai.com is an unrelated reseller
# (see the docs), kept only as an explicit choice for a comparison
PROVIDERS = {
    'openrouter': {'url': 'https://openrouter.ai/api/v1/systemone', 'key_env': 'OPENROUTER_API_KEY',
                   'model': 'jev-1.13', 'price_per_token': 0.042e-6},
    'url': {'url': None, 'key_env': 'JEV_API_KEY', 'model': 'jev-latest', 'price_per_token': 0.042e-6},
}
DEFAULT_PROVIDER = 'openrouter'
JEV_URL = PROVIDERS[DEFAULT_PROVIDER]['url']
JEV_MODEL = PROVIDERS[DEFAULT_PROVIDER]['model']
# the one place the key is read from (a provider may name another variable)
KEY_ENV = PROVIDERS[DEFAULT_PROVIDER]['key_env']
# dollars per input token (TypeSafe's list price, which OpenRouter passes on); output is free, and
# the real figure is summed from `usage`
PRICE_PER_TOKEN = PROVIDERS[DEFAULT_PROVIDER]['price_per_token']


def provider_settings(provider: str = DEFAULT_PROVIDER, url: str | None = None, key_env: str | None = None,
                      model: str | None = None, price_per_token: float | None = None) -> dict:
    """the endpoint, key variable, model and price of a provider, each overridable: --provider url
    needs its endpoint given."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}: one of {sorted(PROVIDERS)}")
    settings = dict(PROVIDERS[provider])
    for name, value in (('url', url), ('key_env', key_env), ('model', model), ('price_per_token', price_per_token)):
        if value not in (None, ''):
            settings[name] = value
    if not settings['url']:
        raise ValueError("--provider url needs --endpoint, the URL of a host that speaks TypeSafe's decide API")
    return settings
# the classes in the codebook's order (individual, social, collaborative); unclear is offered only
# in j2, and absent never (a window nobody is at is no question for Jev)
UNCLEAR = 'unclear'
ABSENT = 'absent'
CLASSES = tuple(c['label'] for c in CODEBOOK['classes'] if c['label'] not in (UNCLEAR, ABSENT))
# variant -> how many preceding windows the state describes
VARIANTS = {'j0': 0, 'j1': 2, 'j2': 0}
# the codebook sentence a state without context must not carry
CONTEXT_SENTENCE = 'Use the preceding windows as context.'
SENSOR_NOTE = (
    "You cannot see or hear the group. The description comes from sensors: one microphone for the whole group "
    "(how much speech, how many words, how many anonymous voices, never what was said), badges that give "
    "positions in metres and whether one person faced another, and cameras that give where each person's gaze "
    "landed when it could be read and how far apart hands were, in shares of the frame width. A part that says "
    "'not measured' had no data; do not read it as zero or as silence."
)
# the least probability a class keeps after renormalising Jev's answer
PROBA_FLOOR = 1e-4
# characters of the JSON sent per input token, for the estimate
CHARS_PER_TOKEN = 4

SLOT_NAMES = ('A', 'B', 'C')
PAIR_SLOTS = ((0, 1), (0, 2), (1, 2))
GAZE_SHARES = ('partner_face', 'partner_hands', 'own_hands', 'elsewhere', 'out_of_frame')
GAZE_WORDS = {'partner_face': "a partner's face", 'partner_hands': "a partner's hands", 'own_hands': 'own hands',
              'elsewhere': 'elsewhere', 'out_of_frame': 'out of frame'}
# a person's gaze shares are read only when at least this share of their frames had a readable gaze
MIN_KNOWN_SHARE = 0.05
# metres between two badges: close below the first edge, far above the second
DISTANCE_EDGES = (0.6, 1.0)
DISTANCE_WORDS = ('close', 'normal', 'far')
# quantity -> (slot-table column, level, whether 0 has its own words and stays out of the fit, tertile words).
# speech, words and gaze switches are 0 in a quarter of the windows or more, which would pile the lower
# edges on 0; those windows say "no speech detected", "no words transcribed", "gaze stayed on one target"
TERTILES = {
    'speech': ('speech_ratio', 'group', True, ('low', 'medium', 'high')),
    'words': ('words', 'group', True, ('few', 'some', 'many')),
    'head_turn': ('yaw_abs_mean', 'person', False, ('low', 'medium', 'high')),
    'head_variability': ('yaw_std', 'person', False, ('steady', 'shifting', 'restless')),
    'hand_movement': ('wrist_speed', 'person', False, ('still', 'slow', 'fast')),
    'gaze_switches': ('gaze_switches', 'person', True, ('few', 'some', 'many')),
    'hand_distance': ('hand_dist_min', 'pair', False, ('close', 'medium', 'far')),
}
HAND_PHRASES = {'still': 'hands still', 'slow': 'hands moving slowly', 'fast': 'hands moving fast'}
# at least this seen share reads as the whole window
WHOLE_WINDOW = 0.95


# ---- the slot table: the plain numbers a state is written from ----

def _column(table: pd.DataFrame, name: str) -> np.ndarray:
    if name not in table.columns:
        return np.full(len(table), np.nan)
    return pd.to_numeric(table[name], errors='coerce').to_numpy(dtype=float)


def _first_column(table: pd.DataFrame, names) -> np.ndarray:
    """the first of `names` the table has: `frame_sets` after the camera fix, `frames` before it."""
    for name in names:
        if name in table.columns:
            return _column(table, name)
    return np.full(len(table), np.nan)


def _pair_prefix(table: pd.DataFrame, a: str, b: str) -> str:
    """the fused table names a pair by its tags in its own order, which need not be slot order."""
    forward = f'pair{a}_{b}_'
    return forward if any(c.startswith(forward) for c in table.columns) else f'pair{b}_{a}_'


def _share(counts: np.ndarray, of: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.clip(np.nan_to_num(counts) / of, 0.0, 1.0)


def slot_table(table: pd.DataFrame, slots, group_size: int | None = None, vfa_mask=None) -> pd.DataFrame:
    """one row per window of a fused table (`<session>_window_features.csv`), with the numbers a
    state is written from, per slot and pair letter instead of per tag, and the zero-versus-missing
    rules applied (NaN is "could not be observed", 0 is an observation).

    `slots` are the roster's kept tags in slot order (at most 3); `group_size` counts every member,
    slotted or not (default: the slot count); `vfa_mask`, a (windows, slots) boolean array, marks
    where the layout's duplicate-skeleton gate masked a person's camera values.

    Columns: window_index, window_start, group_size; speech_ratio (NaN when ASR did not run), words
    (NaN when no transcription chunk covers the window), dia_speakers, dia_switches,
    dia_overlap_ratio (NaN without a diarized chunk); ips_ran, vfa_ran. Per slot X:
    X_present_ratio (NaN when IPS did not run), X_seen_share (frame sets the person was in over
    the window's, 0 when not seen, NaN when the cameras did not run), X_known_share (share of the
    person's frames with a readable gaze), X_<gaze share> for each of GAZE_SHARES (of the readable
    frames; NaN below MIN_KNOWN_SHARE), X_yaw_abs_mean, X_yaw_std, X_wrist_speed, X_gaze_switches
    (NaN when not seen). Per pair XY: XY_dist_mean_m, XY_dist_min_m (NaN unless both located),
    XY_face_any (either faced the other), XY_co_seen_share, XY_hand_dist_min, XY_hand_dist_mean,
    XY_joint_attention_ratio (NaN unless both seen together)."""
    slots = [str(tag) for tag in slots][:len(SLOT_NAMES)]
    n = len(table)
    out: dict[str, np.ndarray] = {}
    index = _column(table, 'window_index')
    out['window_index'] = np.where(np.isnan(index), np.arange(n), index).astype(int)
    out['window_start'] = _column(table, 'window_start')
    out['group_size'] = np.full(n, int(group_size if group_size is not None else len(slots)))

    out['speech_ratio'] = np.where(_column(table, 'n_asr_recognition') > 0, _column(table, 'speech_ratio'), np.nan)
    # the fused table writes 0 words when no chunk covers the window; that is "not transcribed", not silence
    out['words'] = np.where(_column(table, 'n_asr_transcription') > 0, _column(table, 'words'), np.nan)
    diarized = ~np.isnan(_column(table, 'dia_speakers'))
    for name in ('dia_speakers', 'dia_switches', 'dia_overlap_ratio'):
        out[name] = np.where(diarized, _column(table, name), np.nan)
    ips = _column(table, 'n_ips') > 0
    n_vfa = _column(table, 'n_vfa_features')
    vfa = n_vfa > 0
    out['ips_ran'], out['vfa_ran'] = ips, vfa

    masked = np.zeros((n, len(slots)), dtype=bool) if vfa_mask is None \
        else np.asarray(vfa_mask, dtype=bool).reshape(n, len(slots))
    present, seen = [], []
    for i, tag in enumerate(slots):
        x = SLOT_NAMES[i]
        present.append(np.where(ips, _column(table, f'p{tag}_present_ratio'), np.nan))
        sets = _first_column(table, (f'p{tag}_frame_sets', f'p{tag}_frames'))
        seen.append(vfa & (np.nan_to_num(sets) > 0) & ~masked[:, i])
        out[f'{x}_present_ratio'] = present[i]
        out[f'{x}_seen_share'] = np.where(vfa, np.where(seen[i], _share(sets, n_vfa), 0.0), np.nan)
        known = np.where(seen[i], 1.0 - _column(table, f'p{tag}_gaze_unknown_ratio'), np.nan)
        out[f'{x}_known_share'] = known
        readable = seen[i] & (np.nan_to_num(known) >= MIN_KNOWN_SHARE)
        with np.errstate(divide='ignore', invalid='ignore'):
            for category in GAZE_SHARES:
                out[f'{x}_{category}'] = np.where(readable, _column(table, f'p{tag}_gaze_{category}_ratio') / known, np.nan)
        for name in ('yaw_abs_mean', 'yaw_std', 'wrist_speed', 'gaze_switches'):
            out[f'{x}_{name}'] = np.where(seen[i], _column(table, f'p{tag}_{name}'), np.nan)

    for i, j in PAIR_SLOTS:
        if j >= len(slots):
            continue
        xy = SLOT_NAMES[i] + SLOT_NAMES[j]
        prefix = _pair_prefix(table, slots[i], slots[j])
        located = (np.nan_to_num(present[i]) > 0) & (np.nan_to_num(present[j]) > 0)
        out[f'{xy}_dist_mean_m'] = np.where(located, _column(table, prefix + 'dist_mean_m'), np.nan)
        out[f'{xy}_dist_min_m'] = np.where(located, _column(table, prefix + 'dist_min_m'), np.nan)
        # ab and ba follow the tags' arbitrary order, so only "either faced the other" is kept
        out[f'{xy}_face_any'] = np.where(ips, np.fmax(_column(table, prefix + 'face_ab_ratio'),
                                                      _column(table, prefix + 'face_ba_ratio')), np.nan)
        sets = _first_column(table, (prefix + 'frame_sets', prefix + 'frames'))
        together = vfa & (np.nan_to_num(sets) > 0) & ~masked[:, i] & ~masked[:, j]
        out[f'{xy}_co_seen_share'] = np.where(vfa, np.where(together, _share(sets, n_vfa), 0.0), np.nan)
        for name in ('hand_dist_min', 'hand_dist_mean', 'joint_attention_ratio'):
            out[f'{xy}_{name}'] = np.where(together, _column(table, prefix + name), np.nan)
    return pd.DataFrame(out, index=pd.RangeIndex(n))


def _persons(slots: pd.DataFrame) -> list[str]:
    return [x for x in SLOT_NAMES if f'{x}_present_ratio' in slots.columns]


def _pairs(persons: list[str]) -> list[tuple[str, str]]:
    return [(SLOT_NAMES[i], SLOT_NAMES[j]) for i, j in PAIR_SLOTS if j < len(persons)]


# ---- the words: physical bins and label-free tertiles ----

class Bins:
    """the tertile edges of each quantity in TERTILES, fitted once on the dev sessions' windows and
    then frozen; a quantity the dev windows never measured has no edges, and its number goes
    without a word."""

    def __init__(self, edges: dict):
        self.edges = {name: (None if value is None else (float(value[0]), float(value[1])))
                      for name, value in edges.items()}

    def word(self, name: str, value) -> str | None:
        edges = self.edges.get(name)
        value = _num(value)
        if edges is None or value is None:
            return None
        words = TERTILES[name][3]
        return words[0] if value <= edges[0] else words[1] if value <= edges[1] else words[2]

    def to_dict(self) -> dict:
        return {'edges': {name: (None if e is None else list(e)) for name, e in self.edges.items()},
                'words': {name: list(spec[3]) for name, spec in TERTILES.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> 'Bins':
        return cls(data.get('edges', {}))

    def digest(self) -> str:
        """the sha256 of the edges and words: the template a state was written with, which each
        session map records so answers from two templates are never merged."""
        return hashlib.sha256(canonical_json(self.to_dict()).encode('utf-8')).hexdigest()

    def __eq__(self, other) -> bool:
        return isinstance(other, Bins) and self.edges == other.edges


def _values(slots: pd.DataFrame, column: str, level: str) -> np.ndarray:
    persons = _persons(slots)
    if level == 'group':
        names = [column]
    elif level == 'person':
        names = [f'{x}_{column}' for x in persons]
    else:
        names = [f'{x}{y}_{column}' for x, y in _pairs(persons)]
    arrays = [_column(slots, name) for name in names if name in slots.columns]
    return np.concatenate(arrays) if arrays else np.array([])


def fit_bins(slot_tables) -> Bins:
    """tertile edges from every window of the given slot tables (the dev sessions only; no label is
    read). Edges keep 6 significant digits, so the same tables give the same states, and the same
    cache hits, on any machine."""
    if isinstance(slot_tables, pd.DataFrame):
        slot_tables = [slot_tables]
    edges = {}
    for name, (column, level, zero_apart, _) in TERTILES.items():
        values = np.concatenate([_values(t, column, level) for t in slot_tables] or [np.array([])])
        values = values[~np.isnan(values)]
        if zero_apart:
            values = values[values > 0]
        edges[name] = None if len(values) < 3 else \
            tuple(float(f'{q:.6g}') for q in np.quantile(values, [1 / 3, 2 / 3]))
    return Bins(edges)


def distance_word(metres: float) -> str:
    return DISTANCE_WORDS[0] if metres < DISTANCE_EDGES[0] else DISTANCE_WORDS[1] if metres <= DISTANCE_EDGES[1] \
        else DISTANCE_WORDS[2]


# ---- the state ----

def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _flag(value) -> bool:
    number = _num(value)
    return bool(number) if number is not None else False


def _half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _rounded(value: float, digits: int) -> float:
    """the number as the state shows it; its word is given to this, so the two never disagree."""
    return math.floor(value * 10 ** digits + 0.5) / 10 ** digits


def _pct(share: float) -> str:
    return f"{_half_up(100 * share)}%"


def _count(n: float, one: str, many: str) -> str:
    n = _half_up(n)
    return f"{n} {one if n == 1 else many}"


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return ''.join(names)
    return ', '.join(names[:-1]) + ' and ' + names[-1]


def _tag(bins: Bins, name: str, value) -> str:
    word = bins.word(name, value)
    return f" ({word})" if word else ''


def _opening(row: dict, persons: list[str], window_seconds: float) -> str:
    size = _num(row.get('group_size'))
    size = int(size) if size is not None else len(persons)
    length = 'Ten seconds' if window_seconds == 10 else f"{window_seconds:g} seconds"
    if size <= 1:
        return f"{length} of one student ({_join(persons) or 'not tracked'}) at a table."
    if len(persons) >= size:
        return f"{length} of a group of {size} students ({_join(persons)}) at a table."
    follow = f"{len(persons)} of them ({_join(persons)})" if persons else 'none of them'
    return f"{length} of a group of {size} students at a table; the sensors follow {follow}."


def _speech(row: dict, bins: Bins) -> str:
    speech, words, voices = _num(row.get('speech_ratio')), _num(row.get('words')), _num(row.get('dia_speakers'))
    if speech is None and words is None and voices is None:
        return "Speech: not measured in this window."
    parts = []
    if speech is None:
        parts.append("amount of speech not measured")
    elif speech <= 0:
        parts.append("no speech detected")
    else:
        shown = max(1, _half_up(100 * speech))
        parts.append(f"speech in {shown}% of the window{_tag(bins, 'speech', shown / 100)}")
    if words is None:
        parts.append("words not measured")
    elif words <= 0:
        parts.append("no words transcribed")
    else:
        parts.append(_count(words, 'word', 'words') + _tag(bins, 'words', _half_up(words)))
    if voices is None:
        parts.append("voices not measured")
    else:
        switches, overlap = _num(row.get('dia_switches')) or 0.0, _num(row.get('dia_overlap_ratio')) or 0.0
        changes = "no change of speaker" if _half_up(switches) <= 0 else _count(switches, 'change of speaker', 'changes of speaker')
        talk = "no overlapping talk" if overlap <= 0 else f"{_pct(overlap)} overlapping talk"
        parts.append(f"{_count(voices, 'anonymous voice', 'anonymous voices')}, {changes}, {talk}")
    return "Speech: " + "; ".join(parts) + "."


def _positions(row: dict, persons: list[str], pairs: list[tuple[str, str]]) -> str:
    if not _flag(row.get('ips_ran')):
        return "Positions: not measured in this window."
    located, first = [], True
    for x in persons:
        share = _num(row.get(f'{x}_present_ratio')) or 0.0
        if share <= 0:
            located.append(f"{x} not located")
        else:
            located.append(f"{x} located {_pct(share)}" + (" of the window" if first else ""))
            first = False
    parts = [', '.join(located)] if located else []
    for x, y in pairs:
        metres = _num(row.get(f'{x}{y}_dist_mean_m'))
        if metres is None:
            continue
        metres = _rounded(metres, 1)
        text = f"{x} and {y} {metres:.1f} m apart ({distance_word(metres)})"
        face = _num(row.get(f'{x}{y}_face_any'))
        if face is not None:
            text += f", one faced the other in {_pct(face)} of the window" if face > 0 else ", neither faced the other"
        parts.append(text)
    return "Positions: " + ("; ".join(parts) if parts else "nobody located") + "."


def _person_on_camera(row: dict, x: str, bins: Bins, pilot: bool) -> str:
    seen = _num(row.get(f'{x}_seen_share')) or 0.0
    if seen <= 0:
        return f"{x} not in view."
    # the pilot leaves out how much of the window a person was seen: before the camera fix a frame
    # count doubles with two cameras
    if pilot:
        bits = [f"{x} in view"]
    else:
        bits = [f"{x} in view " + ("all the window" if seen >= WHOLE_WINDOW else f"{_pct(seen)} of the window")]
    known = _num(row.get(f'{x}_known_share'))
    if known is None or known < MIN_KNOWN_SHARE:
        bits.append("gaze not readable")
    else:
        shares = ', '.join(f"{GAZE_WORDS[c]} {_pct(_num(row.get(f'{x}_{c}')) or 0.0)}" for c in GAZE_SHARES)
        bits.append(f"gaze readable in {_pct(known)} of frames: {shares}")
    yaw = _num(row.get(f'{x}_yaw_abs_mean'))
    if yaw is None:
        bits.append("head turn not measured")
    else:
        text = f"head turned {_half_up(yaw)} degrees on average{_tag(bins, 'head_turn', _half_up(yaw))}"
        # head-turn variability, hand movement and gaze switches follow one camera's frames in time,
        # which the two-camera interleave corrupts; the pilot leaves them out
        if not pilot:
            steady = bins.word('head_variability', row.get(f'{x}_yaw_std'))
            text += f", {steady}" if steady else ''
        bits.append(text)
    if not pilot:
        speed = _num(row.get(f'{x}_wrist_speed'))
        if speed is None:
            bits.append("hand movement not measured")
        else:
            word = bins.word('hand_movement', speed)
            bits.append(HAND_PHRASES[word] if word else f"hands moving {speed:.2f} frame widths a second")
        switches = _num(row.get(f'{x}_gaze_switches'))
        if switches is not None:
            if switches <= 0:
                bits.append("gaze stayed on one target")
            else:
                n = max(1, _half_up(switches))
                bits.append(("gaze changed target once" if n == 1 else f"gaze changed target {n} times")
                            + _tag(bins, 'gaze_switches', n))
    return "; ".join(bits) + "."


def _cameras(row: dict, persons: list[str], bins: Bins, pilot: bool) -> str:
    if not _flag(row.get('vfa_ran')):
        return "Cameras: not measured in this window."
    return "Cameras: " + " ".join(_person_on_camera(row, x, bins, pilot) for x in persons)


def _camera_pairs(row: dict, pairs: list[tuple[str, str]], bins: Bins, pilot: bool) -> str | None:
    if not pairs or not _flag(row.get('vfa_ran')):
        return None
    items = []
    for x, y in pairs:
        together = _num(row.get(f'{x}{y}_co_seen_share')) or 0.0
        if together <= 0:
            continue
        bits = [] if pilot else \
            ["in view together " + ("all the window" if together >= WHOLE_WINDOW else f"{_pct(together)} of the window")]
        joint = _num(row.get(f'{x}{y}_joint_attention_ratio'))
        bits.append("gaze points not measured" if joint is None
                    else f"gaze points close together (joint attention) in {_pct(joint)} of shared frames")
        hand = _num(row.get(f'{x}{y}_hand_dist_min'))
        hand = None if hand is None else _rounded(hand, 2)
        bits.append("hands not measured" if hand is None
                    else f"hands {hand:.2f} frame widths apart at the closest{_tag(bins, 'hand_distance', hand)}")
        items.append(f"{x} and {y}: " + "; ".join(bits) + ".")
    return "Pairs on camera: " + (" ".join(items) if items else "no two students in view together.")


def _context(rows: dict, t: int, pairs: list[tuple[str, str]], context: int, window_seconds: float) -> str:
    """the windows before t, oldest first (the codebook's rule reads the preceding windows, never the
    next ones)."""
    parts = []
    for k in range(context, 0, -1):
        when = f"{k * window_seconds:g} s earlier"
        if t - k < 0:
            parts.append(f"{when}: before the recording began.")
            continue
        row = rows[t - k]
        speech, words = _num(row.get('speech_ratio')), _num(row.get('words'))
        joints = [v for v in (_num(row.get(f'{x}{y}_joint_attention_ratio')) for x, y in pairs) if v is not None]
        bits = ["speech not measured" if speech is None else f"speech {_pct(speech)}",
                "words not measured" if words is None else "no words" if words <= 0 else _count(words, 'word', 'words'),
                "joint attention not measured" if not joints else f"joint attention {_pct(max(joints))}"]
        parts.append(f"{when}: " + ", ".join(bits) + ".")
    return "Before this: " + " ".join(parts)


def _describe(rows: dict, t: int, persons: list[str], bins: Bins, context: int, pilot: bool,
              window_seconds: float) -> str:
    row = rows[t]
    pairs = _pairs(persons)
    lines = [_opening(row, persons, window_seconds), _speech(row, bins), _positions(row, persons, pairs),
             _cameras(row, persons, bins, pilot)]
    on_camera = _camera_pairs(row, pairs, bins, pilot)
    if on_camera:
        lines.append(on_camera)
    if context > 0:
        lines.append(_context(rows, t, pairs, context, window_seconds))
    return "\n".join(lines)


def describe_window(slots: pd.DataFrame, t: int, bins: Bins, context: int = 0, pilot: bool = False,
                    window_seconds: float = 10.0) -> str:
    """the plain-text state of window t (a position in the slot table): deterministic, rounded,
    every number with its word, a modality that did not run said in words. `context` preceding
    windows are summarised after it (j1: 2); `pilot` leaves out head-turn variability, gaze
    switches, hand movement and the frame-count shares."""
    if not 0 <= t < len(slots):
        raise IndexError(f"window {t} is outside a table of {len(slots)} windows")
    rows = {i: slots.iloc[i].to_dict() for i in range(max(0, t - context), t + 1)}
    return _describe(rows, t, _persons(slots), bins, context, pilot, window_seconds)


# ---- the question and the request ----

def labels_of(variant: str) -> tuple[str, ...]:
    return CLASSES + ((UNCLEAR,) if variant == 'j2' else ())


def question(variant: str = 'j0') -> dict:
    """the `questions` of a request, from the coder's codebook: its rule (without the sentence about
    preceding windows when the state has none), SENSOR_NOTE, and each class's definition as its
    criterion; j2 offers unclear as well."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; one of {', '.join(VARIANTS)}")
    rule = CODEBOOK['rule']
    if VARIANTS[variant] == 0:
        rule = ' '.join(rule.replace(CONTEXT_SENTENCE, '').split())
    definitions = {c['label']: c['definition'] for c in CODEBOOK['classes']}
    return {'interaction': {'type': 'choice', 'instructions': f"{rule} {SENSOR_NOTE}",
                            'criteria': {label: definitions[label] for label in labels_of(variant)}}}


def request_body(state: str, questions: dict, model: str = JEV_MODEL) -> dict:
    """what is POSTed: the model, the state and the questions; never the key."""
    return {'model': model, 'state': state, 'questions': questions}


def canonical_json(body: dict) -> str:
    """the body as sent and as hashed: sorted keys, no spaces, UTF-8."""
    return json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def body_hash(body: dict) -> str:
    return hashlib.sha256(canonical_json(body).encode('utf-8')).hexdigest()


def redacted_request(body: dict, url: str = JEV_URL) -> dict:
    """the request as it would go out, with the key masked, for --dry-run to print."""
    return {'method': 'POST', 'url': url,
            'headers': {'Authorization': 'Bearer ***', 'Content-Type': 'application/json'}, 'body': body}


# ---- the client and the cache ----

class JevError(RuntimeError):
    """a call Jev did not answer; `status` is the HTTP status (None for a network error). The
    message never holds the key."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class JevClient:
    """POSTs a body to Jev's decide endpoint: 60 s timeout, 3 retries with exponential backoff on
    429, 5xx and network errors, and a thread pool of `workers` for many bodies. Each thread keeps
    its own requests.Session. The key lives on the instance only; it is sent in the Authorization
    header and appears in no repr, URL or error."""

    def __init__(self, api_key: str, workers: int = 4, timeout: float = 60, retries: int = 3, backoff: float = 1.0,
                 url: str = JEV_URL, key_env: str = KEY_ENV):
        if not api_key:
            raise ValueError(f"a Jev API key is needed (from {key_env})")
        # requests would refuse such a header with its value, the key, in the message
        if any(c.isspace() or not c.isprintable() for c in api_key):
            raise ValueError(f"the key in {key_env} holds whitespace or control characters")
        self._api_key = api_key
        self.workers, self.timeout, self.retries, self.backoff, self.url = max(1, int(workers)), timeout, retries, backoff, url
        self._local = threading.local()
        self._sleep = time.sleep

    def __repr__(self) -> str:
        return f"JevClient(url={self.url!r}, workers={self.workers}, timeout={self.timeout})"

    def _session(self):
        session = getattr(self._local, 'session', None)
        if session is None:
            import requests
            session = self._local.session = requests.Session()
        return session

    def _wait(self, attempt: int, response=None) -> float:
        wait = self.backoff * (2 ** attempt)
        retry_after = _num(response.headers.get('Retry-After')) if response is not None else None
        return min(max(wait, retry_after or 0.0), 60.0)

    def decide(self, body: dict) -> dict:
        """Jev's answer to one body, as parsed JSON; JevError when it would not answer."""
        import requests
        data = canonical_json(body).encode('utf-8')
        headers = {'Authorization': f'Bearer {self._api_key}', 'Content-Type': 'application/json'}
        problem = None
        for attempt in range(self.retries + 1):
            response = None
            try:
                response = self._session().post(self.url, data=data, headers=headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as error:
                problem = JevError(f"no answer from Jev ({type(error).__name__})")
            except requests.RequestException as error:
                # the message of a malformed request can quote the headers; only its kind is kept
                raise JevError(f"the request to Jev failed ({type(error).__name__})") from None
            else:
                status = response.status_code
                if status == 429 or status >= 500:
                    problem = JevError(f"Jev answered {status}", status)
                elif status >= 400:
                    raise JevError(f"Jev answered {status}: {response.text[:200]}", status)
                else:
                    try:
                        return response.json()
                    except ValueError:
                        raise JevError(f"Jev answered {status} with something that is not JSON", status) from None
            if attempt < self.retries:
                self._sleep(self._wait(attempt, response))
        raise problem

    def decide_all(self, bodies: list[dict]):
        """(position, answer, error) for every body, as the answers come in, with `workers` calls in
        flight; stopping early cancels what has not started."""
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            futures = {pool.submit(self.decide, body): i for i, body in enumerate(bodies)}
            for future in as_completed(futures):
                try:
                    yield futures[future], future.result(), None
                except JevError as error:
                    yield futures[future], None, error
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


class JevCache:
    """content-addressed answers: `<root>/<sha256 of the body>.json` holds the body, the response,
    its usage and model, and when it was called. The key is never part of a body, so it never
    reaches a cache file."""

    def __init__(self, root):
        self.root = Path(root)

    def path(self, digest: str) -> Path:
        return self.root / f"{digest}.json"

    def get(self, body: dict) -> dict | None:
        path = self.path(body_hash(body))
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None

    def has(self, body: dict) -> bool:
        return self.path(body_hash(body)).exists()

    def put(self, body: dict, response: dict, url: str | None = None) -> dict:
        response = response if isinstance(response, dict) else {'value': response}
        entry = {'hash': body_hash(body), 'body': body, 'response': response, 'usage': response.get('usage'),
                 'model': response.get('model'), 'url': url, 'called_at': datetime.now(timezone.utc).isoformat()}
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(entry['hash'])
        temp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding='utf-8')
        temp.replace(path)
        return entry


# ---- scoring and cost ----

def to_proba(answer, labels=CLASSES) -> dict | None:
    """p over `labels` from Jev's answer (the response, or its `answers.interaction`): the stated
    probabilities, renormalised over the labels and floored at PROBA_FLOOR; without them, the
    choice gets the stated confidence (1 when none is given) and the other labels share the rest
    evenly. None when the answer holds neither a usable probability nor a choice among the labels."""
    if not isinstance(answer, dict):
        return None
    if isinstance(answer.get('answers'), dict):
        answer = answer['answers'].get('interaction')
        if not isinstance(answer, dict):
            return None
    labels = tuple(labels)
    raw = answer.get('probabilities')
    stated = None
    if isinstance(raw, dict):
        stated = {label: _num(raw.get(label)) for label in labels}
    elif isinstance(raw, list):
        stated = {}
        for entry in raw:
            if isinstance(entry, dict):
                label = entry.get('label', entry.get('option', entry.get('choice')))
                stated[label] = _num(entry.get('probability', entry.get('p')))
        stated = {label: stated.get(label) for label in labels}
    if stated is not None and sum(max(v or 0.0, 0.0) for v in stated.values()) > 0:
        p = np.array([max(stated[label] or 0.0, 0.0) for label in labels])
    else:
        choice = answer.get('choice')
        if choice not in labels:
            return None
        confidence = _num(answer.get('confidence'))
        confidence = 1.0 if confidence is None else min(max(confidence, 0.0), 1.0)
        rest = (1.0 - confidence) / (len(labels) - 1) if len(labels) > 1 else 0.0
        p = np.array([confidence if label == choice else rest for label in labels])
    p = p / p.sum()
    p = np.clip(p, PROBA_FLOOR, None)
    p = p / p.sum()
    return {label: float(v) for label, v in zip(labels, p)}


def estimate_cost(bodies, price: float = PRICE_PER_TOKEN) -> dict:
    """input tokens and dollars for sending `bodies`, at about CHARS_PER_TOKEN characters of the JSON
    sent per token; identical bodies are paid once. Output tokens are not priced."""
    unique = {body_hash(body): body for body in bodies}
    tokens = sum(math.ceil(len(canonical_json(body)) / CHARS_PER_TOKEN) for body in unique.values())
    return {'calls': len(bodies), 'unique': len(unique), 'input_tokens': tokens, 'dollars': tokens * price}


# ---- a run ----

def build_requests(sessions: list[dict], variant: str = 'j0', bins: Bins | None = None, pilot: bool = False,
                   limit: int | None = None, seed: int = 0, window_seconds: float = 10.0,
                   model: str = JEV_MODEL) -> list[dict]:
    """one request per window to ask about: {session, window_index, window_start, state, body, hash},
    in session and window order. `limit` keeps that many windows drawn with a fixed seed across all
    the sessions, so a pilot asks about the same windows every time (and hits the cache)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; one of {', '.join(VARIANTS)}")
    if bins is None:
        raise ValueError("fit the bins on the dev sessions first (fit_bins)")
    questions = question(variant)
    picks = []
    for s, session in enumerate(sessions):
        starts = np.round(session['slots']['window_start'].to_numpy(dtype=float), 3)
        only = session.get('only')
        if only is None:
            picks.extend((s, t) for t in range(len(starts)))
        else:
            wanted = {round(float(start), 3) for start in only}
            picks.extend((s, t) for t, start in enumerate(starts) if start in wanted)
    if limit is not None and len(picks) > limit:
        picks = sorted(random.Random(seed).sample(picks, max(0, int(limit))))
    records, persons, asks = {}, {}, []
    for s, t in picks:
        if s not in records:
            slots = sessions[s]['slots']
            records[s], persons[s] = dict(enumerate(slots.to_dict('records'))), _persons(slots)
        state = _describe(records[s], t, persons[s], bins, VARIANTS[variant], pilot, window_seconds)
        body = request_body(state, questions, model)
        row = records[s][t]
        asks.append({'session': sessions[s]['session'], 'window_index': int(row['window_index']),
                     'window_start': round(float(row['window_start']), 3), 'state': state, 'body': body,
                     'hash': body_hash(body)})
    return asks


def _usage_tokens(usage, *names) -> float | None:
    if not isinstance(usage, dict):
        return None
    for name in names:
        value = _num(usage.get(name))
        if value is not None:
            return value
    return None


def run_jev(sessions: list[dict], variant: str = 'j0', bins: Bins | None = None, client: JevClient | None = None,
            cache: JevCache | None = None, limit: int | None = None, dry_run: bool = False, pilot: bool = False,
            seed: int = 0, progress=None, model: str = JEV_MODEL) -> pd.DataFrame:
    """ask Jev about the windows of `sessions` and return one prediction row per window.

    Each session is a dict: `session` (its id), `slots` (its slot_table), `table_sha256` (the
    fused table's digest), `out_dir` (where jev_<variant>.jsonl goes; left out, nothing is
    written) and, for j2, `only` (the window starts of its coded windows). Cached answers are
    reused, identical states are asked once, and a dry run asks nothing and writes nothing. A
    window Jev would not answer keeps its error and no probabilities; an authorisation error stops
    the run. `progress(done, total)` is called after each call."""
    if not dry_run and client is None:
        raise ValueError("a JevClient is needed unless it is a dry run")
    labels = labels_of(variant)
    asks = build_requests(sessions, variant, bins, pilot=pilot, limit=limit, seed=seed, model=model)
    answers: dict[str, tuple] = {}  # hash -> (response, error, from the cache)
    todo: dict[str, dict] = {}
    for ask in asks:
        if ask['hash'] in answers or ask['hash'] in todo:
            continue
        entry = cache.get(ask['body']) if cache is not None else None
        if entry is not None:
            answers[ask['hash']] = (entry.get('response'), None, True)
        else:
            todo[ask['hash']] = ask['body']
    if not dry_run and todo:
        digests = list(todo)
        for done, (i, response, error) in enumerate(client.decide_all([todo[d] for d in digests]), start=1):
            digest = digests[i]
            if error is not None:
                if error.status in (401, 403):
                    raise error
                answers[digest] = (None, str(error), False)
            else:
                if cache is not None:
                    cache.put(todo[digest], response, url=client.url)
                answers[digest] = (response, None, False)
            if progress is not None:
                progress(done, len(digests))

    rows = []
    for ask in asks:
        response, error, cached = answers.get(ask['hash'], (None, None, False))
        p = to_proba(response, labels) if response is not None else None
        if response is not None and p is None and error is None:
            error = 'the answer held no probabilities and no choice among the labels'
        row = {'session': ask['session'], 'window_index': ask['window_index'], 'window_start': ask['window_start'],
               'model': 'jev', 'variant': variant, 'pilot': pilot, 'hash': ask['hash'], 'cached': cached}
        for label in labels:
            row[f'p_{label}'] = p[label] if p else np.nan
        if p:
            row['p_interaction'] = p['social'] + p['collaborative']
            row['y_pred'] = int(np.argmax([p[label] for label in labels]))
            row['y_pred_binary'] = int(row['p_interaction'] >= p['individual'])
        else:
            row['p_interaction'], row['y_pred'], row['y_pred_binary'] = np.nan, None, None
        answer = response.get('answers', {}).get('interaction', {}) if isinstance(response, dict) else {}
        row['choice'] = answer.get('choice') if isinstance(answer, dict) else None
        usage = response.get('usage') if isinstance(response, dict) else None
        row['jev_model'] = response.get('model') if isinstance(response, dict) else None
        row['input_tokens'] = _usage_tokens(usage, 'input_tokens', 'prompt_tokens')
        row['output_tokens'] = _usage_tokens(usage, 'output_tokens', 'completion_tokens')
        row['error'] = error
        rows.append(row)
    predictions = pd.DataFrame(rows)
    if len(predictions):
        for column in ('y_pred', 'y_pred_binary'):
            predictions[column] = predictions[column].astype('Int64')
    if not dry_run:
        # a session none of whose windows were drawn keeps the map it had; one with some drawn
        # (--limit) keeps its other windows' lines, see write_session_map
        for session in sessions:
            asked = predictions[predictions['session'] == session['session']] if len(predictions) else predictions
            if session.get('out_dir') and len(asked):
                write_session_map(asked, session['out_dir'], variant, pilot, session.get('table_sha256'), bins.digest())
    return predictions


def session_map_path(out_dir, variant: str, pilot: bool = False) -> Path:
    """artifacts/<session>/analysis/interaction/jev_<variant>.jsonl; a pilot's state differs, so it
    writes jev_<variant>_pilot.jsonl and never overwrites the real run's map."""
    return Path(out_dir) / f"jev_{variant}{'_pilot' if pilot else ''}.jsonl"


def read_session_map(path) -> list[dict]:
    """a session map's lines; a blank or cut line is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    lines = []
    for text in path.read_text(encoding='utf-8').splitlines():
        try:
            line = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(line, dict) and line.get('window_index') is not None:
            lines.append(line)
    return lines


def write_session_map(predictions: pd.DataFrame, out_dir, variant: str, pilot: bool = False,
                      table_sha256: str | None = None, bins_sha256: str | None = None) -> Path:
    """one line per window asked about: its start, the hash of its request (the cache file), the
    fused table's and the template's digests, and the probabilities it got. The lines of an
    earlier run from the same table and template stay for the windows not asked again, so a
    --limit run adds to a full map instead of cutting it down to the few windows it drew; lines
    from another table or template are dropped, since their states differ."""
    path = session_map_path(out_dir, variant, pilot)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = ['window_index', 'window_start', 'hash'] + [c for c in predictions.columns if c.startswith('p_')] \
        + ['y_pred', 'choice', 'error']
    lines = {}
    for line in read_session_map(path):
        if line.get('table_sha256') == table_sha256 and line.get('bins_sha256') == bins_sha256:
            lines[int(line['window_index'])] = line
    for record in predictions.to_dict('records'):
        line = {key: _plain(record.get(key)) for key in keep}
        line.update({'table_sha256': table_sha256, 'bins_sha256': bins_sha256, 'variant': variant, 'pilot': pilot})
        lines[int(line['window_index'])] = line
    # written aside and moved into place, so a crash never leaves half a map
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(temp, 'w', encoding='utf-8') as handle:
        for index in sorted(lines):
            handle.write(json.dumps(lines[index], ensure_ascii=False) + '\n')
    temp.replace(path)
    return path


def _plain(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    return value


# ---- the sanity check ----

def sanity_view(session: str, slots: pd.DataFrame) -> pd.DataFrame:
    """the two quantities the sanity check reads, per window: speech_ratio, and partner_gaze, the
    mean over the persons with a readable gaze of the share landing on a partner's face or hands."""
    persons = _persons(slots)
    with np.errstate(invalid='ignore'):
        directed = [slots[f'{x}_partner_face'].to_numpy(dtype=float) + slots[f'{x}_partner_hands'].to_numpy(dtype=float)
                    for x in persons]
        stacked = np.vstack(directed) if directed else np.full((1, len(slots)), np.nan)
        counts = np.sum(~np.isnan(stacked), axis=0)
        gaze = np.where(counts > 0, np.nansum(stacked, axis=0) / np.maximum(counts, 1), np.nan)
    return pd.DataFrame({'session': session, 'window_start': np.round(slots['window_start'].to_numpy(dtype=float), 3),
                         'speech_ratio': slots['speech_ratio'].to_numpy(dtype=float), 'partner_gaze': gaze})


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    both = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(both) < 3 or both['a'].nunique() < 2 or both['b'].nunique() < 2:
        return None, len(both)
    ranks = both.rank()
    return float(np.corrcoef(ranks['a'], ranks['b'])[0, 1]), len(both)


def sanity(predictions: pd.DataFrame, pooled: pd.DataFrame, speech: str = 'speech_ratio',
           gaze: str = 'partner_gaze') -> dict:
    """Spearman correlation of p_interaction with the speech ratio and with the partner-directed
    gaze share, over every answered window, before any label; both should be positive. `pooled` is
    any per-window frame with `window_start` (and `session` when there are several) holding the two
    columns: sanity_view's, or the layout's pooled view with its column names passed in."""
    keys = [key for key in ('session', 'window_start') if key in predictions.columns and key in pooled.columns]
    left = predictions[keys + ['p_interaction']].copy()
    right = pooled[keys + [speech, gaze]].copy()
    if 'window_start' in keys:
        left['window_start'] = left['window_start'].astype(float).round(3)
        right['window_start'] = right['window_start'].astype(float).round(3)
    joined = left.merge(right, on=keys, how='inner')
    out = {}
    for name, column in (('speech_ratio', speech), ('partner_gaze', gaze)):
        rho, n = _spearman(joined['p_interaction'], joined[column])
        out[name] = {'rho': rho, 'n': n}
    out['positive'] = all(out[name]['rho'] is not None and out[name]['rho'] > 0 for name in ('speech_ratio', 'partner_gaze'))
    return out
