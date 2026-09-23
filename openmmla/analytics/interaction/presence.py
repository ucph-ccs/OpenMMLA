"""The presence gate of the 10 s interaction classifier: a rule, fixed before any label was read,
that calls a window absent when fewer than two of the roster's kept persons are observed in it. A
person is observed when IPS positioned them, a camera saw their tag, or (on a table fused with the
seat columns) an untagged body stood at their seat in at least half of the window's frame sets.

The coder's 'absent' code is the truth the gate is scored against; the gate never makes a label.
It also gives the strata the metrics are reported in (observed persons, gaze readability) and the
end-to-end state shares in which the gate's errors count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from openmmla.analytics.interaction import labels as L
from openmmla.analytics.interaction import layout as LY
from openmmla.analytics.interaction import metrics as M

# a window with fewer observed kept persons is gated absent
MIN_OBSERVED = 2
# the share of frame sets an untagged body must stand at a person's seat
AT_SEAT_MIN = 0.5
# the fused seat column (untagged body at the tag's seat while the tag is not seen); older tables lack it
AT_SEAT_COLUMN = 'p{tag}_untagged_at_seat_ratio'
OBSERVED_LEVELS = ('0', '1', '2+')
GAZE_LEVELS = ('readable', 'unreadable')
RULE = {'version': 1, 'min_observed': MIN_OBSERVED, 'at_seat_min': AT_SEAT_MIN,
        'observed': 'IPS present_ratio > 0, or camera frame sets > 0, or p<tag>_untagged_at_seat_ratio >= 0.5',
        'fixed': 'before any label was read (2026-09-23)'}


def _kept(roster) -> list:
    """the kept tags of a layout.Roster, or a plain sequence of tags, at most N_SLOTS."""
    kept = roster.kept if hasattr(roster, 'kept') else roster
    return list(kept)[:LY.N_SLOTS]


def observed(table: pd.DataFrame, roster) -> np.ndarray:
    """(windows, kept persons): whether each kept person was observed in each window. NaN and a
    column the table does not have count as not observed."""
    kept = _kept(roster)
    if not kept:
        return np.zeros((len(table), 0), dtype=bool)
    columns = []
    with np.errstate(invalid='ignore'):
        for tag in kept:
            columns.append((LY._column(table, f'p{tag}_present_ratio') > 0)
                           | (LY._frame_sets(table, tag) > 0)
                           | (LY._column(table, AT_SEAT_COLUMN.format(tag=tag)) >= AT_SEAT_MIN))
    return np.column_stack(columns)


def observed_count(table: pd.DataFrame, roster) -> np.ndarray:
    """per window, how many kept persons were observed."""
    return observed(table, roster).sum(axis=1).astype(int)


def gate(table: pd.DataFrame, roster) -> np.ndarray:
    """per window, whether the gate calls it absent: fewer than MIN_OBSERVED kept persons observed."""
    return observed_count(table, roster) < MIN_OBSERVED


def gaze_readable(tokens: LY.Tokens) -> np.ndarray:
    """per window, whether every kept person's gaze was readable: seen, not masked as a duplicate,
    with enough known gaze (the layout's m_gaze mask on the unscaled tokens). A kept person not
    seen makes the window unreadable; a session without kept persons is never readable."""
    col = len(LY.PERSON_VALUES) + LY.PERSON_MASKS.index('m_gaze')
    exists = np.asarray(tokens.P_exists) > 0
    on = np.asarray(tokens.P[:, :, col]) > 0
    return (on | ~exists).all(axis=1) & exists.any(axis=1)


def strata(n_observed, readable) -> dict:
    """the pre-declared strata as bool masks over the rows: observed kept persons (0, 1, 2+; a row
    with -1, unknown, is in no level) and gaze readability."""
    n = np.asarray(n_observed, dtype=float)
    r = np.asarray(readable, dtype=bool)
    return {'observed': {'0': n == 0, '1': n == 1, '2+': n >= 2},
            'gaze': {'readable': r, 'unreadable': ~r}}


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _counts(truth: np.ndarray, gated: np.ndarray) -> dict:
    return {'tp': int((truth & gated).sum()), 'fp': int((~truth & gated).sum()),
            'fn': int((truth & ~gated).sum()), 'tn': int((~truth & ~gated).sum())}


def evaluate_gate(y, gated, sessions=None, n_observed=None) -> dict:
    """the gate scored against the coder's absent code on the windows coded absent or a class;
    unclear and uncoded windows are left out. Precision, recall, F1 and kappa are None where they
    are 0/0. `n_observed` adds the counts per observed-persons level, `sessions` per session."""
    y = np.asarray(y, dtype=float)
    gated = np.asarray(gated, dtype=bool)
    with np.errstate(invalid='ignore'):
        truth = y == L.ABSENT_Y
        coded = np.isfinite(y) & (truth | (y >= 0))
        unclear = y == L.UNCLEAR_Y
    t, g = truth[coded], gated[coded]
    counts = _counts(t, g)
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    out = {'rule': dict(RULE), 'windows': int(len(y)), 'gated_windows': int(gated.sum()),
           'coded': int(coded.sum()), 'unclear_excluded': int(unclear.sum()), 'absent': int(t.sum()),
           'gated': int(g.sum()), **counts,
           'precision': _ratio(tp, tp + fp), 'recall': _ratio(tp, tp + fn), 'f1': _ratio(2 * tp, 2 * tp + fp + fn),
           'kappa': M._float(M.kappa(t.astype(int), g.astype(int), n_classes=2)) if coded.any() else None}
    if n_observed is not None:
        levels = strata(n_observed, np.zeros(len(y), dtype=bool))['observed']
        out['by_observed'] = {level: {'coded': int((coded & mask).sum()), 'absent': int((truth & coded & mask).sum()),
                                      'gated': int((gated & coded & mask).sum())}
                              for level, mask in levels.items()}
    if sessions is not None:
        sessions = np.asarray(sessions, dtype=object)
        rows = []
        for session in pd.unique(sessions):
            keep = coded & (sessions == session)
            rows.append({'session': session, 'coded': int(keep.sum()), 'absent': int(truth[keep].sum()),
                         'gated': int(gated[keep].sum()),
                         **{key: value for key, value in _counts(truth[keep], gated[keep]).items() if key != 'tn'}})
        out['by_session'] = rows
    return out


def state_shares(y, y_pred, gated, groups, names) -> pd.DataFrame:
    """per group (lesson, first-seen order), the coder's shares of absent and each class against
    the predicted ones, over the windows coded absent or a class. A gated window is predicted
    absent, the rest take y_pred; a window neither gated nor answered is left out of both sides
    and counted as unanswered."""
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    gated = np.asarray(gated, dtype=bool)
    groups = np.asarray(groups, dtype=object)
    states = ('absent',) + tuple(names)
    with np.errstate(invalid='ignore'):
        coded = np.isfinite(y) & ((y == L.ABSENT_Y) | (y >= 0))
        answered = gated | (np.isfinite(y_pred) & (y_pred >= 0))
        truth = np.where(y == L.ABSENT_Y, 0, np.where(y >= 0, y + 1, -1))
    truth = np.nan_to_num(truth, nan=-1).astype(int)
    pred = np.where(gated, 0, np.where(np.isfinite(y_pred) & (y_pred >= 0), np.nan_to_num(y_pred) + 1, -1)).astype(int)
    rows = []
    for group in pd.unique(groups):
        mine = coded & (groups == group)
        keep = mine & answered
        n = int(keep.sum())
        row = {'lesson': group, 'n': n, 'unanswered': int((mine & ~answered).sum())}
        errors = []
        for s, state in enumerate(states):
            true_share = float((truth[keep] == s).mean()) if n else np.nan
            pred_share = float((pred[keep] == s).mean()) if n else np.nan
            row[f'true_{state}'], row[f'pred_{state}'] = true_share, pred_share
            row[f'err_{state}'] = abs(pred_share - true_share)
            errors.append(row[f'err_{state}'])
        row['err_mean'] = float(np.mean(errors))
        rows.append(row)
    columns = ['lesson', 'n', 'unanswered'] + [f'{kind}_{s}' for s in states for kind in ('true', 'pred', 'err')] \
        + ['err_mean']
    return pd.DataFrame(rows, columns=columns)


def share_summary(frame: pd.DataFrame) -> dict:
    """state_shares as metrics.json holds it: the states, every lesson's row, the mean error over
    lessons and per state (NaN as None)."""
    states = [column[len('err_'):] for column in frame.columns if column.startswith('err_') and column != 'err_mean']
    records = [{key: (M._float(value) if isinstance(value, (float, np.floating)) else value)
                for key, value in record.items()} for record in frame.to_dict('records')]
    return {'states': states, 'by_lesson': records,
            'err_mean': M._float(frame['err_mean'].mean()) if len(frame) else None,
            'err': {state: M._float(frame[f'err_{state}'].mean()) if len(frame) else None for state in states}}
