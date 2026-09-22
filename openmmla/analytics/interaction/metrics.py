"""The scores of the 10 s interaction classifier: the three-class and binary metrics, calibration
and reliability, temporal fidelity, the closing-the-loop measures (state shares, onset latency),
and the uncertainty around them.

Labels are ints (0 individual, 1 social, 2 collaborative, -1 for a window that is not coded or is
unclear, which no score reads) and posteriors (n, 3) arrays; the binary target is always derived,
interaction = social or collaborative, p_interaction = p_social + p_collaborative. Rows are in
session and window order, so that neighbouring rows of one session (and one coded block) are
neighbouring windows.

Windows ten seconds apart are not independent, so every interval here is a session-cluster
bootstrap over lessons, recomputing the pooled metric on each resample, and every difference is
paired on the same resamples; no window-level interval is ever computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# the class names, in label order (the same as labels.CLASSES)
CLASSES = ('individual', 'social', 'collaborative')
WINDOW = 10.0
# a per-session macro-F1 averages only the classes with at least this many true windows there
MIN_SUPPORT = 5
ECE_BINS = 15
FLOOR = 1e-12


def _labels(y, n_classes: int = len(CLASSES)) -> np.ndarray:
    values = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(values) & (values >= 0) & (values < n_classes)
    out = np.full(len(values), -1, dtype=int)
    out[valid] = values[valid].astype(int)
    return out


def _float(value):
    """a plain float for the JSON report, None for NaN."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# ---- hard labels ----

def confusion(y, y_pred, n_classes: int = len(CLASSES)) -> np.ndarray:
    """rows the true class, columns the predicted one, over coded windows with a prediction."""
    y, y_pred = _labels(y, n_classes), _labels(y_pred, n_classes)
    keep = (y >= 0) & (y_pred >= 0)
    return np.bincount(y[keep] * n_classes + y_pred[keep], minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def f1_per_class(y, y_pred, n_classes: int = len(CLASSES)) -> tuple[np.ndarray, np.ndarray]:
    """(F1, support) of every class: F1 = 2 TP / (true windows + predicted windows), 0 when a class
    was neither true nor predicted. A coded window with no prediction counts against recall."""
    y, y_pred = _labels(y, n_classes), _labels(y_pred, n_classes)
    coded = y >= 0
    support = np.bincount(y[coded], minlength=n_classes).astype(float)
    predicted = np.bincount(y_pred[coded & (y_pred >= 0)], minlength=n_classes).astype(float)
    hit = coded & (y == y_pred)
    tp = np.bincount(y[hit], minlength=n_classes).astype(float)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros(n_classes), where=denominator > 0)
    return f1, support


def macro_f1(y, y_pred, min_support: int = 1, n_classes: int = len(CLASSES)) -> float:
    """the mean F1 over the classes with at least `min_support` true windows (all present classes
    by default; MIN_SUPPORT for a single session); NaN when no class qualifies."""
    f1, support = f1_per_class(y, y_pred, n_classes)
    counted = support >= max(min_support, 1)
    return float(f1[counted].mean()) if counted.any() else float('nan')


def balanced_accuracy(y, y_pred, n_classes: int = len(CLASSES)) -> float:
    """the mean recall over the classes that have true windows."""
    y, y_pred = _labels(y, n_classes), _labels(y_pred, n_classes)
    coded = y >= 0
    support = np.bincount(y[coded], minlength=n_classes).astype(float)
    tp = np.bincount(y[coded & (y == y_pred)], minlength=n_classes).astype(float)
    present = support > 0
    return float((tp[present] / support[present]).mean()) if present.any() else float('nan')


def kappa(y, y_pred, n_classes: int = len(CLASSES)) -> float:
    """Cohen's kappa of the prediction against the coder, over coded windows with a prediction."""
    matrix = confusion(y, y_pred, n_classes).astype(float)
    n = matrix.sum()
    if n == 0:
        return float('nan')
    observed = np.trace(matrix) / n
    expected = (matrix.sum(1) * matrix.sum(0)).sum() / n ** 2
    return float((observed - expected) / (1 - expected)) if expected < 1 else float('nan')


# ---- probabilities ----

def nll(y, p) -> float:
    """the mean negative log-likelihood of the true classes."""
    y, p = _labels(y), np.asarray(p, dtype=float)
    keep = (y >= 0) & np.isfinite(p).all(1)
    if not keep.any():
        return float('nan')
    return float(-np.log(np.clip(p[keep, y[keep]], FLOOR, 1.0)).mean())


def brier(y, p) -> float:
    """the multi-class Brier score: the mean squared distance of the posterior to the one-hot truth."""
    y, p = _labels(y, np.shape(p)[1]), np.asarray(p, dtype=float)
    keep = (y >= 0) & np.isfinite(p).all(1)
    if not keep.any():
        return float('nan')
    return float(((p[keep] - np.eye(p.shape[1])[y[keep]]) ** 2).sum(1).mean())


def _equal_mass(confidence: np.ndarray, outcome: np.ndarray, bins: int) -> list[tuple]:
    """(n, mean confidence, observed frequency, lowest, highest confidence) of each of `bins` bins
    holding equally many windows, in confidence order. Equal confidences share a bin (the one their
    first falls in), so a tie is never split by the order the windows came in."""
    n = len(confidence)
    order = np.argsort(confidence, kind='stable')
    ordered = confidence[order]
    first = np.searchsorted(ordered, ordered, side='left')
    index = (first * min(bins, n)) // max(n, 1)
    out = []
    for b in np.unique(index):
        chunk = order[index == b]
        out.append((len(chunk), confidence[chunk].mean(), outcome[chunk].mean(),
                    confidence[chunk].min(), confidence[chunk].max()))
    return out


def reliability(p, y, bins: int = ECE_BINS) -> pd.DataFrame:
    """the reliability table: for the top label (confidence = max p, outcome = the argmax was
    right) and for each class (confidence = p_k, outcome = the window is k), bins of equal mass
    with their mean confidence and observed frequency."""
    p = np.asarray(p, dtype=float)
    k = p.shape[1]
    y = _labels(y, k)
    keep = (y >= 0) & np.isfinite(p).all(1)
    p, y = p[keep], y[keep]
    rows = []
    if len(y):
        kinds = [('top', p.max(1), (p.argmax(1) == y).astype(float))]
        names = CLASSES if k == len(CLASSES) else tuple(str(c) for c in range(k))
        kinds += [(names[c], p[:, c], (y == c).astype(float)) for c in range(k)]
        for kind, confidence, outcome in kinds:
            for b, (n, conf, freq, lo, hi) in enumerate(_equal_mass(confidence, outcome, bins)):
                rows.append({'kind': kind, 'bin': b, 'n': int(n), 'confidence': float(conf),
                             'frequency': float(freq), 'lo': float(lo), 'hi': float(hi)})
    return pd.DataFrame(rows, columns=['kind', 'bin', 'n', 'confidence', 'frequency', 'lo', 'hi'])


def ece(table: pd.DataFrame) -> tuple[float, float]:
    """(top-label ECE, classwise ECE) of a reliability table: the window-weighted mean gap between
    confidence and frequency, the classwise one averaged over the classes."""
    if table.empty:
        return float('nan'), float('nan')
    gaps = pd.DataFrame({'kind': table['kind'], 'n': table['n'],
                         'gap': (table['confidence'] - table['frequency']).abs() * table['n']})
    sums = gaps.groupby('kind', sort=False)[['gap', 'n']].sum()
    per_kind = sums['gap'] / sums['n']
    top = float(per_kind['top']) if 'top' in per_kind.index else float('nan')
    classwise = float(per_kind.drop('top', errors='ignore').mean())
    return top, classwise


# ---- temporal fidelity ----

def _segments(n: int, sessions=None, blocks=None) -> np.ndarray:
    """a segment id per row: a new segment where the session or the block changes, so neighbouring
    rows of one segment are neighbouring windows."""
    change = np.zeros(n, dtype=bool)
    for key in (sessions, blocks):
        if key is not None:
            values = pd.Series(list(key)).astype(str).to_numpy()
            change[1:] |= values[1:] != values[:-1]
    return np.cumsum(change)


def _runs(labels: np.ndarray, segment: np.ndarray, keep: np.ndarray) -> list[int]:
    """the lengths of the runs of one label among kept rows, a run ending where the label, the
    segment or the keeping changes."""
    lengths, current = [], 0
    for t in range(len(labels)):
        if not keep[t]:
            if current:
                lengths.append(current)
            current = 0
            continue
        if current and (labels[t] != labels[t - 1] or segment[t] != segment[t - 1] or not keep[t - 1]):
            lengths.append(current)
            current = 0
        current += 1
    if current:
        lengths.append(current)
    return lengths


def temporal_fidelity(y, y_pred, sessions=None, blocks=None, window: float = WINDOW) -> dict:
    """state switches per coded hour, true and predicted, and the mean run length in seconds, both
    over the coded windows within coded blocks: a smoother that flattens real switches, or a model
    that flickers, shows here and nowhere else. Pass the Viterbi path as y_pred for its run
    lengths."""
    y, y_pred = _labels(y), _labels(y_pred)
    segment = _segments(len(y), sessions, blocks)
    coded = y >= 0
    pair = coded[:-1] & coded[1:] & (segment[:-1] == segment[1:])
    hours = coded.sum() * window / 3600.0
    true_runs = _runs(y, segment, coded)
    pred_runs = _runs(y_pred, segment, coded & (y_pred >= 0))
    return {
        'coded_hours': _float(hours),
        'switches_per_hour_true': _float((y[:-1][pair] != y[1:][pair]).sum() / hours) if hours else None,
        'switches_per_hour_pred': _float((y_pred[:-1][pair] != y_pred[1:][pair]).sum() / hours) if hours else None,
        'mean_run_s_true': _float(np.mean(true_runs) * window) if true_runs else None,
        'mean_run_s_pred': _float(np.mean(pred_runs) * window) if pred_runs else None,
    }


# ---- closing the loop ----

def state_share_error(y, p, sessions) -> pd.DataFrame:
    """per session, the absolute error of each state's time share: the mean posterior against the
    coded share, over the session's coded windows; `mean` averages the three."""
    y, p, sessions = _labels(y), np.asarray(p, dtype=float), np.asarray(sessions, dtype=object)
    rows = []
    for session in pd.unique(sessions):
        keep = (sessions == session) & (y >= 0) & np.isfinite(p).all(1)
        if not keep.any():
            continue
        coded_share = np.bincount(y[keep], minlength=p.shape[1]) / keep.sum()
        error = np.abs(p[keep].mean(0) - coded_share)
        row = {'session': session, 'n': int(keep.sum())}
        row.update({f'err_{name}': float(error[c]) for c, name in enumerate(CLASSES[:p.shape[1]])})
        row['err_mean'] = float(error.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def onset_latency(y, p_filter, blocks=None, sessions=None, prior=None, target: int = 2, min_run: int = 3,
                  min_before: int = 3, window: float = WINDOW) -> dict:
    """for a causal variant: the seconds from the true start of a collaborative run (at least
    `min_run` windows, after at least `min_before` coded windows of another class in the same
    block) to the first window of the run predicted collaborative, and the share of runs never
    caught while they lasted. `p_filter` is the filtered posteriors (decided by argmax p / prior
    when a prior is given, else argmax) or the predicted labels themselves."""
    y = _labels(y)
    p_filter = np.asarray(p_filter, dtype=float)
    if p_filter.ndim == 2:
        ratio = p_filter / (np.asarray(prior, dtype=float) if prior is not None else 1.0)
        predicted = np.where(np.isfinite(ratio).all(1), np.nan_to_num(ratio).argmax(1), -1)
    else:
        predicted = _labels(p_filter)
    segment = _segments(len(y), sessions, blocks)
    latencies, missed = [], 0
    t = 0
    while t < len(y):
        if y[t] != target or (t > 0 and y[t - 1] == target and segment[t - 1] == segment[t]):
            t += 1
            continue
        end = t
        while end < len(y) and y[end] == target and segment[end] == segment[t]:
            end += 1
        before = np.arange(t - min_before, t)
        onset = end - t >= min_run and t >= min_before and (segment[before] == segment[t]).all() \
            and ((y[before] >= 0) & (y[before] != target)).all()
        if onset:
            hits = np.flatnonzero(predicted[t:end] == target)
            if len(hits):
                latencies.append(float(hits[0]) * window)
            else:
                missed += 1
        t = end
    n = len(latencies) + missed
    return {'n_onsets': n, 'n_missed': missed, 'miss_rate': _float(missed / n) if n else None,
            'latency_mean_s': _float(np.mean(latencies)) if latencies else None,
            'latency_median_s': _float(np.median(latencies)) if latencies else None,
            'latencies_s': latencies}


# ---- the report ----

def _binary(y: np.ndarray, p, y_pred_binary: np.ndarray) -> dict:
    coded = y >= 0
    truth = (y > 0).astype(int)
    truth[~coded] = -1
    out = {'f1_interaction': _float(f1_per_class(truth, y_pred_binary, 2)[0][1]),
           'balanced_accuracy': _float(balanced_accuracy(truth, y_pred_binary, 2)),
           'support_interaction': int((truth == 1).sum()), 'auroc': None, 'auprc': None}
    if p is not None:
        # p_social + p_collaborative (or p_interaction itself when the model is a two-class one)
        score = p[:, 1:].sum(1)
        keep = coded & np.isfinite(score)
        if len(np.unique(truth[keep])) == 2:
            from sklearn.metrics import average_precision_score, roc_auc_score
            out['auroc'] = _float(roc_auc_score(truth[keep], score[keep]))
            out['auprc'] = _float(average_precision_score(truth[keep], score[keep]))
    return out


def _decisions(p, y_pred, y_pred_binary) -> tuple[np.ndarray, np.ndarray]:
    """the hard labels to score: y_pred as given, else argmax p; the binary ones as given, else
    y_pred > 0 (-1 where there is no label)."""
    if y_pred is None:
        if p is None:
            raise ValueError("give the posteriors, the hard labels, or both")
        y_pred = np.where(np.isfinite(p).all(1), np.nan_to_num(p).argmax(1), -1)
    y_pred = _labels(y_pred)
    if y_pred_binary is None:
        y_pred_binary = np.where(y_pred >= 0, (y_pred > 0).astype(int), -1)
    return y_pred, _labels(y_pred_binary, 2)


def _scores(y, p, y_pred, y_pred_binary, bins: int, min_support: int = 1) -> dict:
    """every window-level metric of one set of rows; the probabilistic ones are None when there is
    no posterior (R0 gives hard labels only)."""
    f1, support = f1_per_class(y, y_pred)
    out = {
        'n': int((y >= 0).sum()),
        'macro_f1': _float(macro_f1(y, y_pred, min_support)),
        'f1': {name: _float(f1[c]) for c, name in enumerate(CLASSES)},
        'support': {name: int(support[c]) for c, name in enumerate(CLASSES)},
        'balanced_accuracy': _float(balanced_accuracy(y, y_pred)),
        'kappa': _float(kappa(y, y_pred)),
        'confusion': confusion(y, y_pred).tolist(),
        'binary': _binary(y, p, y_pred_binary),
        'nll': None, 'brier': None, 'ece_top': None, 'ece_classwise': None, 'reliability': None,
    }
    if p is not None:
        table = reliability(p, y, bins)
        top, classwise = ece(table)
        out.update({'nll': _float(nll(y, p)), 'brier': _float(brier(y, p)), 'ece_top': _float(top),
                    'ece_classwise': _float(classwise), 'reliability': table.to_dict('records')})
    return out


def per_session(y, p, sessions, y_pred=None, tasks=None, min_support: int = MIN_SUPPORT,
                y_pred_binary=None) -> pd.DataFrame:
    """one row per session with coded windows: its macro-F1 over the classes with at least
    `min_support` true windows there, per-class F1 and support, balanced accuracy, kappa, binary
    F1, NLL and the state-share error. The labels default as in report()."""
    y, sessions = _labels(y), np.asarray(sessions, dtype=object)
    p = None if p is None else np.asarray(p, dtype=float)
    y_pred, y_pred_binary = _decisions(p, y_pred, y_pred_binary)
    tasks = None if tasks is None else np.asarray(tasks, dtype=object)
    rows = []
    for session in pd.unique(sessions):
        rows_of = sessions == session
        coded = rows_of & (y >= 0)
        if not coded.any():
            continue
        f1, support = f1_per_class(y[rows_of], y_pred[rows_of])
        truth = np.where(y[rows_of] >= 0, (y[rows_of] > 0).astype(int), -1)
        row = {'session': session, 'task': tasks[rows_of][0] if tasks is not None else None, 'n': int(coded.sum()),
               'macro_f1': _float(macro_f1(y[rows_of], y_pred[rows_of], min_support))}
        row.update({f'f1_{name}': _float(f1[c]) for c, name in enumerate(CLASSES)})
        row.update({f'support_{name}': int(support[c]) for c, name in enumerate(CLASSES)})
        row.update({'balanced_accuracy': _float(balanced_accuracy(y[rows_of], y_pred[rows_of])),
                    'kappa': _float(kappa(y[rows_of], y_pred[rows_of])),
                    'f1_interaction': _float(f1_per_class(truth, y_pred_binary[rows_of], 2)[0][1])})
        if p is not None:
            row['nll'] = _float(nll(y[rows_of], p[rows_of]))
            shares = state_share_error(y[rows_of], p[rows_of], sessions[rows_of])
            row['share_err'] = _float(shares['err_mean'].iloc[0]) if len(shares) else None
        rows.append(row)
    return pd.DataFrame(rows)


def report(y, p, sessions=None, tasks=None, empty=None, blocks=None, y_pred=None, y_pred_binary=None,
           viterbi_path=None, window: float = WINDOW, bins: int = ECE_BINS, min_support: int = MIN_SUPPORT) -> dict:
    """every metric of 4.4 as a JSON-ready dict: pooled over the held-out windows, without the
    empty windows (no speech, nobody located, nobody seen), per task and per session, with the
    temporal fidelity and the state-share error.

    `p` is the calibrated (n, 3) posteriors, None for a model with hard labels only (R0, whose
    NLL, Brier, ECE, AUROC and AUPRC are then None, i.e. n/a). `y_pred` is the hard label (the
    balanced decision, computed per fold with that fold's prior), argmax p by default;
    `y_pred_binary` the binary decision, y_pred > 0 by default; `viterbi_path` the Viterbi path
    for the switch counts and run lengths, y_pred by default."""
    y = _labels(y)
    p = None if p is None else np.asarray(p, dtype=float)
    y_pred, y_pred_binary = _decisions(p, y_pred, y_pred_binary)
    path = y_pred if viterbi_path is None else _labels(viterbi_path)

    def part(keep):
        return _scores(np.where(keep, y, -1), p, y_pred, y_pred_binary, bins)

    everything = np.ones(len(y), dtype=bool)
    out = {'pooled': part(everything)}
    out['pooled']['temporal'] = temporal_fidelity(y, path, sessions, blocks, window)
    if empty is not None:
        out['without_empty'] = part(~np.asarray(empty, dtype=bool))
        out['empty_share'] = _float(np.asarray(empty, dtype=bool)[y >= 0].mean()) if (y >= 0).any() else None
    if tasks is not None:
        tasks = np.asarray(tasks, dtype=object)
        out['by_task'] = {}
        for task in pd.unique(tasks):
            keep = tasks == task
            scores = part(keep)
            scores['temporal'] = temporal_fidelity(np.where(keep, y, -1), path, sessions, blocks, window)
            out['by_task'][str(task)] = scores
    if sessions is not None:
        table = per_session(y, p, sessions, y_pred, tasks, min_support, y_pred_binary)
        out['per_session'] = table.to_dict('records')
        if p is not None:
            shares = state_share_error(y, p, sessions)
            out['state_share_error'] = {column: _float(shares[column].mean()) for column in shares.columns
                                        if column.startswith('err_')} if len(shares) else None
    return out


# ---- uncertainty ----

def _resamples(groups, n: int, seed: int):
    """n session-cluster resamples: each draws as many groups as there are, with replacement, and
    takes every row of each drawn group (a group drawn twice counts twice)."""
    groups = np.asarray(groups, dtype=object)
    names = pd.unique(groups)
    rows_of = [np.flatnonzero(groups == name) for name in names]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        drawn = rng.integers(0, len(names), len(names))
        yield np.concatenate([rows_of[g] for g in drawn])


def session_bootstrap(fn, groups, n: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """the percentile interval of a pooled metric over lesson resamples: fn(rows) -> float is
    recomputed on the rows of each resample. A resample where the metric is undefined is left out
    and counted."""
    estimate = fn(np.arange(len(groups)))
    samples = np.array([fn(rows) for rows in _resamples(groups, n, seed)], dtype=float)
    finite = samples[np.isfinite(samples)]
    tail = (1 - level) / 2 * 100
    return {'estimate': _float(estimate), 'lo': _float(np.percentile(finite, tail)) if len(finite) else None,
            'hi': _float(np.percentile(finite, 100 - tail)) if len(finite) else None,
            'n': int(n), 'n_undefined': int(n - len(finite)), 'samples': samples}


def paired_delta(fn_a, fn_b, groups, n: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """the difference fn_a - fn_b of a pooled metric, with its interval from the same lesson
    resamples for both and a two-sided bootstrap p-value (the share of resampled differences on
    the other side of 0, doubled, with the +1 correction)."""
    everything = np.arange(len(groups))
    delta = fn_a(everything) - fn_b(everything)
    samples = np.array([fn_a(rows) - fn_b(rows) for rows in _resamples(groups, n, seed)], dtype=float)
    finite = samples[np.isfinite(samples)]
    tail = (1 - level) / 2 * 100
    if len(finite):
        below = (1 + (finite <= 0).sum()) / (len(finite) + 1)
        above = (1 + (finite >= 0).sum()) / (len(finite) + 1)
        p_value = min(1.0, 2 * min(below, above))
    else:
        p_value = float('nan')
    return {'delta': _float(delta), 'lo': _float(np.percentile(finite, tail)) if len(finite) else None,
            'hi': _float(np.percentile(finite, 100 - tail)) if len(finite) else None,
            'p': _float(p_value), 'n': int(n), 'n_undefined': int(n - len(finite)), 'samples': samples}


def holm(pvalues):
    """Holm's step-down adjustment of the confirmatory contrasts' p-values: the i-th smallest is
    multiplied by (m - i + 1), kept monotone and capped at 1. A dict keeps its keys."""
    keys = list(pvalues) if isinstance(pvalues, dict) else None
    values = np.asarray([pvalues[k] for k in keys] if keys is not None else pvalues, dtype=float)
    m = len(values)
    order = np.argsort(values, kind='stable')
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[i]))
        adjusted[i] = running
    return dict(zip(keys, adjusted.tolist())) if keys is not None else adjusted
