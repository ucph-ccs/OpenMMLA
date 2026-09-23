"""The coded labels of the interaction classifier: the class `mmla ses-code` recorded per coder for
each 10 s window, joined to the rows of the session's fused table.

A coder's file (artifacts/<session>/labels/<coder>.jsonl) is an append-only log: a later line for
a window replaces an earlier one, and a line with label null undoes it. The log is replayed in file
order, which is how ses-code itself reads it back, so the model trains on exactly what the coder
last saw; the time stamps cannot order it, since an undo line carries the server's `undone_at` and
no `coded_at`.

Labels are floats over the table's grid: 0 individual, 1 social, 2 collaborative, -1 unclear and
-2 absent (fewer than two members at the group's place; both kept as sequence context, never
scored) and NaN for a window nobody coded. The binary target (interaction = social or
collaborative) is always derived from these, never coded on its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ('individual', 'social', 'collaborative')
UNCLEAR = 'unclear'
UNCLEAR_Y = -1
# a window where fewer than two members are at the group's place: coded, context, never scored
ABSENT = 'absent'
ABSENT_Y = -2
# every code a coder can give, in the order of the five-code agreement, and its y
CODES = CLASSES + (UNCLEAR, ABSENT)
CODE_Y = {**{name: k for k, name in enumerate(CLASSES)}, UNCLEAR: UNCLEAR_Y, ABSENT: ABSENT_Y}
# the file whose labels overrule every coder's for the windows it holds
ADJUDICATED = 'adjudicated'
# `coder` is whose truth a row is part of, `source` the file it came from: in a coder's resolved
# truth the adjudicated rows keep source 'adjudicated' under that coder's name, so the truth is
# still one coder's labels to the join
LABEL_COLUMNS = ('session', 'coder', 'source', 'window_start', 'window_end', 'label', 'y', 'note', 'coded_at')


class LabelJoinError(ValueError):
    """too many labels found no window of the table; `report` says how many and how."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _y(label: str) -> float:
    return float(CODE_Y[label])


def _read_coder(path: Path, session: str) -> tuple[pd.DataFrame, int]:
    """one coder's resolved labels (last line per window wins, a null label undoes), and how many
    lines could not be read (a line cut short by a crash is skipped, not fatal)."""
    labels: dict[int, dict] = {}
    skipped = 0
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = int(round(float(record['window_start']) * 1000))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            skipped += 1
            continue
        label = record.get('label')
        if label is None:
            labels.pop(key, None)  # an undo line
            continue
        if label not in CODE_Y:
            raise ValueError(f"{path}: line {number}: unknown label {label!r}")
        start = round(float(record['window_start']), 3)
        end = record.get('window_end')
        labels[key] = {'session': record.get('session') or session, 'coder': path.stem, 'source': path.stem,
                       'window_start': start,
                       'window_end': round(float(end), 3) if end is not None else np.nan, 'label': label,
                       'y': _y(label), 'note': record.get('note') or '', 'coded_at': record.get('coded_at')}
    frame = pd.DataFrame(list(labels.values()), columns=list(LABEL_COLUMNS))
    return frame.sort_values('window_start', kind='stable').reset_index(drop=True), skipped


def primary_coder(labels: pd.DataFrame) -> str | None:
    """the coder with the most windows (adjudication aside; ties by name): the truth when no
    --coder is given. Pass every session's labels to pick one coder for the whole run."""
    counts = labels[labels['coder'] != ADJUDICATED].groupby('coder').size()
    if counts.empty:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def load_labels(session_dir, coder: str | None = None, all_coders: bool = False) -> pd.DataFrame:
    """a session's labels (LABEL_COLUMNS, one row per window, in time order): the given coder's,
    else the one with the most windows in this session, with adjudicated.jsonl overruling them for
    the windows it holds. Every row of that truth carries the chosen coder's name (it is one
    truth, which join_labels takes), and `source` says which rows adjudication decided. With
    `all_coders` every coder's resolved labels come back (for agreement), one row per coder and
    window, adjudication among them as coder 'adjudicated'. Labels are never averaged across
    coders."""
    session_dir = Path(session_dir)
    folder = session_dir / 'labels'
    frames, skipped = [], 0
    if folder.is_dir():
        for path in sorted(folder.glob('*.jsonl')):
            frame, lost = _read_coder(path, session_dir.name)
            frames.append(frame)
            skipped += lost
    frames = [frame for frame in frames if len(frame)]
    records = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(LABEL_COLUMNS))
    records.attrs['skipped_lines'] = skipped
    if all_coders:
        return records
    chosen = coder or primary_coder(records)
    truth = records[records['coder'] == chosen]
    adjudicated = records[records['coder'] == ADJUDICATED]
    if len(adjudicated):
        truth = pd.concat([truth[~truth['window_start'].isin(adjudicated['window_start'])], adjudicated])
    truth = truth.sort_values('window_start', kind='stable').reset_index(drop=True)
    # adjudicated rows join the chosen coder's truth; a session only adjudication labelled is its own
    truth['coder'] = chosen if chosen is not None else ADJUDICATED
    truth.attrs['skipped_lines'] = skipped
    truth.attrs['coder'] = chosen
    return truth


def _millis(values) -> np.ndarray:
    return np.round(np.asarray(values, dtype=float) * 1000).astype(np.int64)


def join_labels(table: pd.DataFrame, labels: pd.DataFrame, tolerance: float = 0.5, mode: str = 'exact',
                max_unmatched: float = 0.01, min_overlap: float = 5.0) -> tuple[pd.Series, dict]:
    """the labels over the table's grid (a float Series indexed by window_index: 0, 1, 2, -1 for
    unclear, -2 for absent, NaN uncoded) and a report of how they were matched. 'exact' matches
    on the window start to the millisecond (the grids coincide), then the nearest window within `tolerance`
    seconds; 'overlap' (opt-in, for a coding grid whose start moved because a recording was
    missing on the coding machine) maps each label to a window it overlaps by at least
    `min_overlap` seconds, one to one, and reports the offsets. More than `max_unmatched` of the
    labels without a window raises LabelJoinError."""
    if mode not in ('exact', 'overlap'):
        raise ValueError(f"unknown join mode {mode!r}: 'exact' or 'overlap'")
    if 'coder' in labels.columns and labels['coder'].nunique() > 1:
        raise ValueError("join one coder's labels (load_labels without all_coders)")
    label_starts = labels['window_start'].to_numpy(dtype=float)
    label_y = labels['y'].to_numpy(dtype=float)
    if len(np.unique(_millis(label_starts))) < len(label_starts):
        raise ValueError("two labels for one window")
    starts = table['window_start'].to_numpy(dtype=float)
    ends = table['window_end'].to_numpy(dtype=float) if 'window_end' in table.columns else starts + 10.0
    index = pd.Index(table['window_index'].astype(int) if 'window_index' in table.columns else table.index,
                     name='window_index')
    y = np.full(len(table), np.nan)
    taken = np.zeros(len(table), dtype=bool)
    report = {'mode': mode, 'labels': len(labels), 'exact': 0, 'nearest': 0, 'overlap': 0, 'unmatched': 0,
              'unclear': int((label_y == UNCLEAR_Y).sum()), 'absent': int((label_y == ABSENT_Y).sum())}
    offsets = []

    def assign(k, i, how, offset):
        y[i] = label_y[k]
        taken[i] = True
        report[how] += 1
        offsets.append(offset)

    if mode == 'exact':
        position = {ms: i for i, ms in enumerate(_millis(starts))}
        order = np.argsort(starts, kind='stable')
        pending = []
        for k, ms in enumerate(_millis(label_starts)):
            i = position.get(ms)
            if i is not None and not taken[i]:
                assign(k, i, 'exact', 0.0)
            else:
                pending.append(k)
        for k in pending:
            # the nearest window on either side of the label's start
            at = np.searchsorted(starts[order], label_starts[k])
            near = [order[j] for j in (at - 1, at) if 0 <= j < len(order)]
            near = [i for i in near if not taken[i] and abs(starts[i] - label_starts[k]) <= tolerance]
            if near:
                i = min(near, key=lambda i: abs(starts[i] - label_starts[k]))
                assign(k, i, 'nearest', float(label_starts[k] - starts[i]))
            else:
                report['unmatched'] += 1
    else:
        label_ends = labels['window_end'].to_numpy(dtype=float) if 'window_end' in labels.columns \
            else label_starts + 10.0
        label_ends = np.where(np.isfinite(label_ends), label_ends, label_starts + 10.0)
        candidates = []
        for k in range(len(labels)):
            overlap = np.minimum(ends, label_ends[k]) - np.maximum(starts, label_starts[k])
            for i in np.flatnonzero(overlap >= min_overlap):
                candidates.append((-overlap[i], k, i))
        # the largest overlaps first, each label and each window used once
        matched = np.zeros(len(labels), dtype=bool)
        for _, k, i in sorted(candidates):
            if not matched[k] and not taken[i]:
                matched[k] = True
                assign(k, i, 'overlap', float(label_starts[k] - starts[i]))
        report['unmatched'] = int((~matched).sum())
    report['unmatched_share'] = report['unmatched'] / len(labels) if len(labels) else 0.0
    if offsets:
        report['offset_seconds'] = {'median': float(np.median(offsets)), 'min': float(np.min(offsets)),
                                    'max': float(np.max(offsets))}
    if report['unmatched_share'] > max_unmatched:
        raise LabelJoinError(
            f"{report['unmatched']} of {report['labels']} labels found no window of the table "
            f"({report['unmatched_share']:.1%} > {max_unmatched:.0%}); the coding grid may have moved "
            f"(a recording missing on the coding machine): try the overlap join", report)
    return pd.Series(y, index=index, name='y'), report


def scored(y) -> np.ndarray:
    """where a label is a class (not unclear, not absent, not uncoded): the windows loss and
    metrics read."""
    values = np.asarray(y, dtype=float)
    with np.errstate(invalid='ignore'):
        return np.isfinite(values) & (values >= 0)


def to_binary(y):
    """interaction (1: social or collaborative) against individual (0). From labels, unclear (-1),
    absent (-2) and uncoded (NaN) pass through; from (n, 3) class probabilities it is p_social +
    p_collaborative. A Series stays a Series."""
    values = np.asarray(y, dtype=float)
    if values.ndim == 2:
        out = values[:, 1] + values[:, 2]
    else:
        with np.errstate(invalid='ignore'):
            out = np.where(values >= 1, 1.0, values)
    if isinstance(y, pd.Series):
        return pd.Series(out, index=y.index, name=y.name)
    return out


def _kappa(confusion: np.ndarray) -> float:
    n = confusion.sum()
    if n == 0:
        return float('nan')
    observed = np.trace(confusion) / n
    expected = (confusion.sum(axis=1) * confusion.sum(axis=0)).sum() / n ** 2
    return float((observed - expected) / (1 - expected)) if expected < 1 else float('nan')


def agreement(a, b) -> dict:
    """two coders' agreement on the windows both gave a class: Cohen's kappa for the three
    classes and for the derived binary, and the confusion matrices (rows a, columns b), and
    Cohen's kappa over all five codes (the three classes, unclear and absent) on every window both
    coded. Series are aligned on their index (window_start or window_index), anything else by
    position; a window either coder called unclear or absent is counted apart from the
    three-class kappa."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        a, b = a.align(b, join='inner')
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("the two label sets cover different windows")
    both = np.isfinite(a) & np.isfinite(b)
    keep = scored(a) & scored(b)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(confusion, (a[keep].astype(int), b[keep].astype(int)), 1)
    binary = np.zeros((2, 2), dtype=int)
    np.add.at(binary, (to_binary(a[keep]).astype(int), to_binary(b[keep]).astype(int)), 1)
    index = {float(CODE_Y[c]): i for i, c in enumerate(CODES)}
    ai = np.array([index.get(v, -1) if np.isfinite(v) else -1 for v in a], dtype=int)
    bi = np.array([index.get(v, -1) if np.isfinite(v) else -1 for v in b], dtype=int)
    keep5 = (ai >= 0) & (bi >= 0)
    codes = np.zeros((len(CODES), len(CODES)), dtype=int)
    np.add.at(codes, (ai[keep5], bi[keep5]), 1)
    return {'windows': int(keep.sum()), 'both_coded': int(both.sum()),
            'unclear_a': int((both & (a == UNCLEAR_Y)).sum()), 'unclear_b': int((both & (b == UNCLEAR_Y)).sum()),
            'kappa': _kappa(confusion), 'kappa_binary': _kappa(binary),
            'accuracy': float(np.trace(confusion) / keep.sum()) if keep.any() else float('nan'),
            'confusion': confusion.tolist(), 'confusion_binary': binary.tolist(),
            'absent_a': int((both & (a == ABSENT_Y)).sum()), 'absent_b': int((both & (b == ABSENT_Y)).sum()),
            'codes': list(CODES), 'kappa_codes': _kappa(codes),
            'accuracy_codes': float(np.trace(codes) / keep5.sum()) if keep5.any() else float('nan'),
            'confusion_codes': codes.tolist()}


def label_counts(y_by_session: dict) -> pd.DataFrame:
    """windows per class and session (and unclear, absent, uncoded), the table printed before training
    and checked against the 30-window refusal."""
    rows = {}
    for session, y in y_by_session.items():
        values = np.asarray(y, dtype=float)
        row = {name: int((values == k).sum()) for k, name in enumerate(CLASSES)}
        row[UNCLEAR] = int((values == UNCLEAR_Y).sum())
        row[ABSENT] = int((values == ABSENT_Y).sum())
        row['uncoded'] = int((~np.isfinite(values)).sum())
        rows[session] = row
    return pd.DataFrame.from_dict(rows, orient='index', columns=list(CLASSES) + [UNCLEAR, ABSENT, 'uncoded'])
