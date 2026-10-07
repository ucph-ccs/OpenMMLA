"""Paired contrasts of two variants of the 10 s interaction classifier on the same windows, by unit
(mmla ses-contrast).

A variant is read from a run's predictions.csv by its metrics key, model:temporal:hmm, with
:arm for an ablated arm (the full arm by default); the two sides may come from one run or from two
runs that scored the same windows against the same truth (the same coder, the same join). They are
paired on (session, window_index) over the coded windows, interaction (social or collaborative)
against individual, that both answered. Two sides whose coded windows differ (a session one run
read and the other did not) are refused unless the comparison is allowed on the windows both hold
(subset), and the counts of each side's coded windows are kept either way.

Per unit (the lesson column of predictions.csv: a date with its same_class_as dates in the date,
unit and forward splits, the session in the session split) and per variant:
- C, the within-session concordance: over every (interaction, individual) pair of windows of the
  same session, the share where the variant's p_interaction is higher for the interaction window,
  a tie counting a half; pooled over the pairs of the unit's sessions;
- kappa_binary, Cohen's kappa of y_pred_binary against the binary truth;
- binary_macro_f1, the mean F1 of the two classes over the classes the unit has.

Inference is on the per-unit differences d_u = A - B over the units where both sides define it:
- the paired t-interval, mean d +- t(1 - alpha / 2, n - 1) SD / sqrt(n), and its two-sided p;
- the exact sign-flip test: the share of the 2^n sign assignments whose |mean| reaches the observed
  one (Monte Carlo, with the +1 correction, above MAX_EXACT units);
- the unit-cluster bootstrap, units drawn with replacement and both sides on the same draws: of the
  pooled difference (the metric recomputed on the drawn units' pairs or windows) and of the unit mean;
- Holm over the declared family, the contrasts of one call, metric by metric, on the t-test and
  sign-flip p-values;
- TOST for equivalence with a margin delta (on C by default): equivalent when the (1 - 2 alpha)
  t-interval lies within (-delta, delta), i.e. both one-sided tests reject.

The outcome of a contrast reads its decision metric (C): 'a > b' or 'a < b' when the Holm-adjusted
t-test rejects, else 'equivalent' when TOST does, else 'inconclusive'.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ('C', 'kappa_binary', 'binary_macro_f1')
DECISION = 'C'
ALPHA = 0.05
BOOTSTRAP = 2000
SEED = 20261004
# above this many units the sign-flip p-value is a Monte Carlo one
MAX_EXACT = 20
SIGN_FLIP_DRAWS = 100_000
COLUMNS = ('session', 'lesson', 'window_index', 'y_true', 'p_interaction', 'y_pred_binary')


class ContrastError(ValueError):
    """the two sides cannot be paired: a variant is missing, their truth differs, or they scored
    different coded windows."""


# ---- reading ----

def parse_variant(text: str) -> tuple:
    """(model, temporal, hmm, arm) of a metrics key model:temporal:hmm[:arm]; the arm is 'full' when
    not given."""
    parts = str(text).split(':')
    if len(parts) not in (3, 4) or not all(parts):
        raise ContrastError(f"a variant is model:temporal:hmm or model:temporal:hmm:arm (lr:T0:none), not {text!r}")
    return parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else 'full'


def predictions_path(path) -> Path:
    """a predictions.csv, or the run folder holding one, as an absolute path."""
    path = Path(path).resolve()
    return path / 'predictions.csv' if path.is_dir() else path


def load_variant(path, variant: str) -> pd.DataFrame:
    """one variant's rows of a predictions.csv (or a run folder): session, lesson, window_index,
    y_true, p_interaction, y_pred_binary. A run without an ablation column is the full arm."""
    model, temporal, hmm, arm = parse_variant(variant)
    table = pd.read_csv(predictions_path(path))
    arms = table['ablation'] if 'ablation' in table.columns else pd.Series('full', index=table.index)
    rows = table[(table['model'] == model) & (table['variant'] == f'{temporal}:{hmm}') & (arms == arm)]
    if rows.empty:
        keys = sorted({f"{m}:{v}" + ('' if a == 'full' else f':{a}')
                       for m, v, a in zip(table['model'], table['variant'], arms)})
        raise ContrastError(f"{predictions_path(path)} has no variant {variant}; it has {', '.join(keys)}")
    return rows[list(COLUMNS)].reset_index(drop=True)


def coded_windows(frame: pd.DataFrame) -> set:
    """the (session, window_index) of a variant's coded windows, those whose y_true is a class."""
    truth = pd.to_numeric(frame['y_true'], errors='coerce').to_numpy(dtype=float)
    with np.errstate(invalid='ignore'):
        coded = np.isfinite(truth) & (truth >= 0)
    return set(zip(frame['session'].to_numpy()[coded], frame['window_index'].to_numpy()[coded]))


def coverage(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """how the two sides' coded windows overlap: how many each side holds and both hold, and the
    sessions holding a coded window of one side only."""
    in_a, in_b = coded_windows(a), coded_windows(b)
    return {'a': len(in_a), 'b': len(in_b), 'both': len(in_a & in_b),
            'sessions_a_only': sorted({str(s) for s, _ in in_a - in_b}),
            'sessions_b_only': sorted({str(s) for s, _ in in_b - in_a})}


def paired_frame(a: pd.DataFrame, b: pd.DataFrame, unit: str = 'lesson', subset: bool = False) -> pd.DataFrame:
    """the coded windows both sides hold, joined on (session, window_index), with the binary
    truth, each side's p_interaction and y_pred_binary, and the unit (`unit`: 'lesson', the run's
    unit, or 'session'). A window either side has no answer for (no posterior, no decision) is
    dropped from that metric later, not here. Raises ContrastError when the truth differs, and
    when the two sides' coded windows differ (coverage) unless `subset` allows the windows both
    hold."""
    cover = coverage(a, b)
    if not subset and not cover['a'] == cover['b'] == cover['both']:
        alone = [f"A alone in {', '.join(cover['sessions_a_only'])}"] if cover['sessions_a_only'] else []
        alone += [f"B alone in {', '.join(cover['sessions_b_only'])}"] if cover['sessions_b_only'] else []
        raise ContrastError(f"the two sides scored different coded windows (A {cover['a']}, B {cover['b']}, both "
                            f"{cover['both']}; {'; '.join(alone)}): compare runs over the same sessions, or allow "
                            f"the windows both hold (--allow-subset)")
    joined = a.merge(b, on=['session', 'window_index'], suffixes=('_a', '_b'), how='inner')
    truth_a, truth_b = joined['y_true_a'].to_numpy(dtype=float), joined['y_true_b'].to_numpy(dtype=float)
    both = np.isfinite(truth_a) & np.isfinite(truth_b)
    differ = (both & (truth_a != truth_b)) | (np.isfinite(truth_a) != np.isfinite(truth_b))
    if differ.any():
        raise ContrastError(f"the two sides disagree on the truth of {int(differ.sum())} window(s): they were not "
                            f"scored against the same coder and join")
    coded = both & (truth_a >= 0)
    joined = joined[coded].reset_index(drop=True)
    units = joined['session'] if unit == 'session' else joined['lesson_a']
    return pd.DataFrame({'session': joined['session'], 'unit': units.astype(str), 'window_index': joined['window_index'],
                         'truth': (joined['y_true_a'].to_numpy(dtype=float) > 0).astype(int),
                         'p_a': joined['p_interaction_a'].to_numpy(dtype=float),
                         'p_b': joined['p_interaction_b'].to_numpy(dtype=float),
                         'pred_a': pd.to_numeric(joined['y_pred_binary_a'], errors='coerce').fillna(-1).astype(int),
                         'pred_b': pd.to_numeric(joined['y_pred_binary_b'], errors='coerce').fillna(-1).astype(int)})


# ---- the metrics, from counts that add up over units ----

def concordance_counts(score, truth) -> tuple:
    """(concordant pairs, ties counting a half; pairs) of one session: every (interaction,
    individual) pair of its windows, concordant when the interaction window scores higher."""
    score, truth = np.asarray(score, dtype=float), np.asarray(truth)
    positive, negative = np.sort(score[truth == 1]), np.sort(score[truth == 0])
    if not len(positive) or not len(negative):
        return 0.0, 0
    below = np.searchsorted(negative, positive, side='left')
    tied = np.searchsorted(negative, positive, side='right') - below
    return float(below.sum() + 0.5 * tied.sum()), int(len(positive) * len(negative))


def concordance(score, truth, sessions) -> float:
    """C over windows of possibly several sessions: the concordant pairs within each session over
    the pairs within each session, pooled; NaN without a pair."""
    sessions = np.asarray(sessions, dtype=object)
    concordant, pairs = 0.0, 0
    for session in pd.unique(sessions):
        at = sessions == session
        c, n = concordance_counts(np.asarray(score)[at], np.asarray(truth)[at])
        concordant, pairs = concordant + c, pairs + n
    return concordant / pairs if pairs else float('nan')


def _confusion(truth, pred) -> np.ndarray:
    truth, pred = np.asarray(truth), np.asarray(pred)
    keep = (truth >= 0) & (pred >= 0)
    return np.bincount(truth[keep] * 2 + pred[keep], minlength=4).reshape(2, 2).astype(float)


def kappa_from(confusion) -> float:
    """Cohen's kappa of a confusion matrix (rows the truth); NaN when it is undefined."""
    confusion = np.asarray(confusion, dtype=float)
    n = confusion.sum()
    if n == 0:
        return float('nan')
    observed = np.trace(confusion) / n
    expected = (confusion.sum(1) * confusion.sum(0)).sum() / n ** 2
    return float((observed - expected) / (1 - expected)) if expected < 1 else float('nan')


def macro_f1_from(confusion) -> float:
    """the mean F1 over the classes with a true window, as metrics.macro_f1 counts it; NaN when
    there is none."""
    confusion = np.asarray(confusion, dtype=float)
    support, predicted, hit = confusion.sum(1), confusion.sum(0), np.diag(confusion)
    denominator = support + predicted
    f1 = np.divide(2 * hit, denominator, out=np.zeros(len(hit)), where=denominator > 0)
    counted = support >= 1
    return float(f1[counted].mean()) if counted.any() else float('nan')


def unit_counts(frame: pd.DataFrame) -> dict:
    """per unit (in name order) and side, what every metric adds up from: concordant pairs and
    pairs (each session's own, on the windows both sides gave a posterior), and the binary
    confusion (on the windows both sides gave a decision), with the windows counted."""
    out = {}
    for unit in sorted(frame['unit'].unique()):
        rows = frame[frame['unit'] == unit]
        scored = np.isfinite(rows['p_a'].to_numpy()) & np.isfinite(rows['p_b'].to_numpy())
        decided = (rows['pred_a'].to_numpy() >= 0) & (rows['pred_b'].to_numpy() >= 0)
        record = {'sessions': sorted(rows['session'].unique()), 'windows': int(len(rows)),
                  'scored': int(scored.sum()), 'decided': int(decided.sum())}
        for side in ('a', 'b'):
            concordant, pairs = 0.0, 0
            for session in rows['session'].unique():
                at = (rows['session'] == session).to_numpy() & scored
                c, n = concordance_counts(rows[f'p_{side}'].to_numpy()[at], rows['truth'].to_numpy()[at])
                concordant, pairs = concordant + c, pairs + n
            record[side] = {'concordant': concordant, 'pairs': pairs,
                            'confusion': _confusion(rows['truth'].to_numpy()[decided],
                                                    rows[f'pred_{side}'].to_numpy()[decided])}
        out[unit] = record
    return out


def _metric(counts: dict, metric: str) -> float:
    """one side's metric from its (summed) counts."""
    if metric == 'C':
        return counts['concordant'] / counts['pairs'] if counts['pairs'] else float('nan')
    if metric == 'kappa_binary':
        return kappa_from(counts['confusion'])
    return macro_f1_from(counts['confusion'])


def _summed(records: list, side: str) -> dict:
    return {'concordant': sum(r[side]['concordant'] for r in records), 'pairs': sum(r[side]['pairs'] for r in records),
            'confusion': sum((r[side]['confusion'] for r in records), np.zeros((2, 2)))}


# ---- inference ----

def _float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def t_interval(d, level: float = 0.95) -> dict:
    """the paired t-interval of the mean of the unit differences `d` (df = n - 1) and the two-sided
    one-sample t-test p of mean 0; None where fewer than two units."""
    from scipy import stats
    d = np.asarray(d, dtype=float)
    n = len(d)
    out = {'n': n, 'mean': _float(d.mean()) if n else None, 'sd': _float(d.std(ddof=1)) if n > 1 else None,
           'df': n - 1 if n > 1 else None, 'level': level, 'lo': None, 'hi': None, 'p': None}
    if n < 2:
        return out
    se = d.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * se
    out.update(lo=_float(d.mean() - half), hi=_float(d.mean() + half))
    if se > 0:
        out['p'] = _float(2 * stats.t.sf(abs(d.mean()) / se, n - 1))
    else:
        out['p'] = 1.0 if d.mean() == 0 else 0.0
    return out


def sign_flip(d, max_exact: int = MAX_EXACT, draws: int = SIGN_FLIP_DRAWS, seed: int = SEED) -> dict:
    """the two-sided sign-flip test of mean 0 over the unit differences `d`: exact over every one
    of the 2^n sign assignments up to `max_exact` units (the observed one among them), the Monte
    Carlo share with the +1 correction above."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return {'p': None, 'exact': None, 'assignments': 0}
    observed = abs(d.mean()) - 1e-12
    if n <= max_exact:
        reached, total = 0, 2 ** n
        bits = np.arange(n)
        for start in range(0, total, 1 << 16):
            codes = np.arange(start, min(start + (1 << 16), total))
            signs = 1.0 - 2.0 * ((codes[:, None] >> bits) & 1)
            reached += int((np.abs(signs @ d) / n >= observed).sum())
        return {'p': reached / total, 'exact': True, 'assignments': total}
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(draws, n))
    reached = int((np.abs(signs @ d) / n >= observed).sum())
    return {'p': (reached + 1) / (draws + 1), 'exact': False, 'assignments': draws}


def tost(d, margin: float, alpha: float = ALPHA) -> dict:
    """two one-sided t-tests of |mean| < margin over the unit differences `d`: p = the larger of the
    two one-sided p-values, equivalent when it is below alpha, i.e. when the (1 - 2 alpha)
    t-interval lies within (-margin, margin)."""
    from scipy import stats
    d = np.asarray(d, dtype=float)
    n = len(d)
    out = {'margin': margin, 'alpha': alpha, 'p': None, 'equivalent': None,
           'interval': t_interval(d, 1 - 2 * alpha) if n > 1 else None}
    if n < 2 or margin is None or margin <= 0:
        return out
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se > 0:
        lower = stats.t.sf((mean + margin) / se, n - 1)
        upper = stats.t.cdf((mean - margin) / se, n - 1)
        out['p'] = _float(max(lower, upper))
    else:
        out['p'] = 0.0 if abs(mean) < margin else 1.0
    out['equivalent'] = bool(out['p'] < alpha)
    return out


def holm(pvalues: dict) -> dict:
    """Holm's step-down adjustment of a family's p-values (metrics.holm); None stays None."""
    from openmmla.analytics.interaction import metrics as M
    known = {name: p for name, p in pvalues.items() if p is not None}
    adjusted = M.holm(known) if known else {}
    return {name: adjusted.get(name) for name in pvalues}


def bootstrap(records: list, metric: str, n: int = BOOTSTRAP, seed: int = SEED, level: float = 0.95) -> dict:
    """the unit-cluster bootstrap: units drawn with replacement, both sides on the same draws, of the
    pooled difference (the metric recomputed on the drawn units' summed counts) and of the mean of
    the unit differences; percentile intervals, a draw where a value is undefined left out and
    counted."""
    if len(records) < 2:
        return {'n': 0, 'pooled_lo': None, 'pooled_hi': None, 'mean_lo': None, 'mean_hi': None, 'undefined': 0}
    rng = np.random.default_rng(seed)
    unit_d = np.array([_metric(r['a'], metric) - _metric(r['b'], metric) for r in records])
    pooled, means = [], []
    for _ in range(n):
        drawn = rng.integers(0, len(records), len(records))
        chosen = [records[i] for i in drawn]
        pooled.append(_metric(_summed(chosen, 'a'), metric) - _metric(_summed(chosen, 'b'), metric))
        picked = unit_d[drawn]
        means.append(np.nanmean(picked) if np.isfinite(picked).any() else np.nan)
    pooled, means = np.array(pooled), np.array(means)
    tail = (1 - level) / 2 * 100

    def percentiles(values):
        finite = values[np.isfinite(values)]
        return (_float(np.percentile(finite, tail)), _float(np.percentile(finite, 100 - tail))) if len(finite) \
            else (None, None)

    (pooled_lo, pooled_hi), (mean_lo, mean_hi) = percentiles(pooled), percentiles(means)
    return {'n': n, 'seed': seed, 'level': level, 'pooled_lo': pooled_lo, 'pooled_hi': pooled_hi,
            'mean_lo': mean_lo, 'mean_hi': mean_hi, 'undefined': int((~np.isfinite(pooled)).sum())}


def contrast(frame: pd.DataFrame, margins: dict | None = None, alpha: float = ALPHA, n_boot: int = BOOTSTRAP,
             seed: int = SEED) -> dict:
    """one paired contrast A - B over a paired_frame: per metric the pooled values of both sides and
    their difference, the per-unit values and differences, the t-interval (1 - alpha), the sign-flip
    test, the unit-cluster bootstrap and, where `margins` names the metric, TOST. A unit where a
    side leaves the metric undefined (no pair, one class) stays out of that metric's inference."""
    margins = margins or {}
    counts = unit_counts(frame)
    units = list(counts)
    records = [counts[u] for u in units]
    out = {'windows': int(len(frame)), 'units': units, 'metrics': {}, 'per_unit': []}
    for unit in units:
        record = counts[unit]
        row = {'unit': unit, 'sessions': '+'.join(record['sessions']), 'windows': record['windows'],
               'pairs': record['a']['pairs']}
        for metric in METRICS:
            a, b = _metric(record['a'], metric), _metric(record['b'], metric)
            row.update({f'{metric}_a': _float(a), f'{metric}_b': _float(b), f'{metric}_delta': _float(a - b)})
        out['per_unit'].append(row)
    for metric in METRICS:
        values = [(u, _metric(r['a'], metric), _metric(r['b'], metric), r) for u, r in zip(units, records)]
        defined = [(u, a, b, r) for u, a, b, r in values if np.isfinite(a) and np.isfinite(b)]
        d = np.array([a - b for _, a, b, _ in defined], dtype=float)
        pooled_a, pooled_b = _metric(_summed(records, 'a'), metric), _metric(_summed(records, 'b'), metric)
        out['metrics'][metric] = {
            'pooled_a': _float(pooled_a), 'pooled_b': _float(pooled_b), 'pooled_delta': _float(pooled_a - pooled_b),
            'units_defined': [u for u, _, _, _ in defined],
            'units_undefined': [u for u, a, b, _ in values if not (np.isfinite(a) and np.isfinite(b))],
            'mean_a': _float(np.mean([a for _, a, _, _ in defined])) if defined else None,
            'mean_b': _float(np.mean([b for _, _, b, _ in defined])) if defined else None,
            't': t_interval(d, 1 - alpha), 'sign_flip': sign_flip(d, seed=seed),
            'bootstrap': bootstrap([r for _, _, _, r in defined], metric, n_boot, seed, 1 - alpha),
            'tost': tost(d, margins[metric], alpha) if metric in margins else None}
    return out


def outcome(record: dict, alpha: float = ALPHA, metric: str = DECISION) -> dict:
    """the reading of a contrast on its decision metric, after Holm: 'a > b' or 'a < b' when the
    Holm-adjusted t-test rejects, else 'equivalent' when TOST does, else 'inconclusive'; with the
    two flags beside it."""
    scores = record['metrics'][metric]
    p = scores.get('p_t_holm')
    differs = p is not None and p < alpha
    equivalent = bool((scores.get('tost') or {}).get('equivalent'))
    mean = scores['t']['mean']
    if differs:
        reading = 'a > b' if mean > 0 else 'a < b'
    elif equivalent:
        reading = 'equivalent'
    else:
        reading = 'inconclusive'
    return {'metric': metric, 'reading': reading, 'differs': differs, 'equivalent': equivalent}


def family(contrasts: dict, alpha: float = ALPHA) -> dict:
    """Holm over the declared family (every contrast given), metric by metric, on the t-test and the
    sign-flip p-values, written into each contrast (p_t_holm, p_sign_flip_holm), and each
    contrast's outcome."""
    for metric in METRICS:
        t = holm({name: c['metrics'][metric]['t']['p'] for name, c in contrasts.items()})
        flip = holm({name: c['metrics'][metric]['sign_flip']['p'] for name, c in contrasts.items()})
        for name, c in contrasts.items():
            c['metrics'][metric]['p_t_holm'] = t[name]
            c['metrics'][metric]['p_sign_flip_holm'] = flip[name]
    for c in contrasts.values():
        c['outcome'] = outcome(c, alpha)
    return contrasts


# ---- a call: read, compare, write ----

def _digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_contrasts(specs, out, margin: float | None = None, alpha: float = ALPHA, n_boot: int = BOOTSTRAP,
                  seed: int = SEED, unit: str = 'lesson', log=None, subset: bool = False) -> dict:
    """every contrast of `specs`, a list of (name, (path, variant), (path, variant)), as one Holm
    family: paired, compared and written to `out` as contrasts.json (everything), contrasts.csv (a
    row per contrast and metric) and contrasts_per_unit.csv (a row per contrast and unit). `margin`
    is TOST's margin on C; `subset` compares two sides that scored different coded windows on the
    windows both hold (refused otherwise, paired_frame), each contrast keeping the counts (coded).
    Returns the record contrasts.json holds."""
    say = log or (lambda message: None)
    names = [name for name, _, _ in specs]
    if len(set(names)) != len(names):
        raise ContrastError(f"contrast names must differ: {', '.join(names)}")
    margins = {DECISION: margin} if margin is not None else {}
    contrasts, inputs = {}, {}
    for name, (path_a, variant_a), (path_b, variant_b) in specs:
        for path in (path_a, path_b):
            inputs[str(predictions_path(path))] = _digest(predictions_path(path))
        side_a, side_b = load_variant(path_a, variant_a), load_variant(path_b, variant_b)
        try:
            frame = paired_frame(side_a, side_b, unit, subset)
        except ContrastError as error:
            raise ContrastError(f"{name}: {error}") from None
        if frame.empty:
            raise ContrastError(f"{name}: the two sides share no coded window")
        record = contrast(frame, margins, alpha, n_boot, seed)
        record.update(a={'predictions': str(predictions_path(path_a)), 'variant': variant_a},
                      b={'predictions': str(predictions_path(path_b)), 'variant': variant_b},
                      coded=coverage(side_a, side_b))
        contrasts[name] = record
        cover = record['coded']
        partial = '' if cover['a'] == cover['b'] == cover['both'] else \
            f" (a subset: A holds {cover['a']} coded window(s), B {cover['b']})"
        say(f"{name}: {len(record['units'])} unit(s), {record['windows']} paired coded window(s){partial}")
    family(contrasts, alpha)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'family': names, 'alpha': alpha,
              'level': 1 - alpha, 'margin': margins, 'unit': unit, 'subset': subset, 'bootstrap': n_boot, 'seed': seed,
              'decision_metric': DECISION, 'inputs': inputs, 'contrasts': contrasts,
              'method': (__doc__ or '').split('\n\n', 1)[-1].strip()}
    from openmmla.analytics.interaction.evaluate import write_json
    write_json(out / 'contrasts.json', result)
    rows, units = [], []
    for name, c in contrasts.items():
        for metric, s in c['metrics'].items():
            rows.append({'contrast': name, 'a': c['a']['variant'], 'b': c['b']['variant'], 'metric': metric,
                         'units': s['t']['n'], 'pooled_a': s['pooled_a'], 'pooled_b': s['pooled_b'],
                         'pooled_delta': s['pooled_delta'], 'mean_delta': s['t']['mean'], 'sd': s['t']['sd'],
                         't_lo': s['t']['lo'], 't_hi': s['t']['hi'], 'p_t': s['t']['p'], 'p_t_holm': s['p_t_holm'],
                         'p_sign_flip': s['sign_flip']['p'], 'p_sign_flip_holm': s['p_sign_flip_holm'],
                         'sign_flip_exact': s['sign_flip']['exact'],
                         'boot_pooled_lo': s['bootstrap']['pooled_lo'], 'boot_pooled_hi': s['bootstrap']['pooled_hi'],
                         'boot_mean_lo': s['bootstrap']['mean_lo'], 'boot_mean_hi': s['bootstrap']['mean_hi'],
                         'tost_margin': (s['tost'] or {}).get('margin'), 'tost_p': (s['tost'] or {}).get('p'),
                         'tost_lo': ((s['tost'] or {}).get('interval') or {}).get('lo'),
                         'tost_hi': ((s['tost'] or {}).get('interval') or {}).get('hi'),
                         'equivalent': (s['tost'] or {}).get('equivalent'),
                         'outcome': c['outcome']['reading'] if metric == DECISION else None})
        units += [dict(row, contrast=name) for row in c['per_unit']]
    pd.DataFrame(rows).to_csv(out / 'contrasts.csv', index=False)
    per_unit = pd.DataFrame(units)
    if len(per_unit):
        per_unit = per_unit[['contrast'] + [c for c in per_unit.columns if c != 'contrast']]
    per_unit.to_csv(out / 'contrasts_per_unit.csv', index=False)
    return result


def _show(value, signed: bool = True) -> str:
    return 'n/a' if value is None else (f'{value:+.3f}' if signed else f'{value:.3f}')


def summary(result: dict) -> str:
    """the contrasts as printed: per contrast and metric the mean unit difference, its t-interval,
    the Holm-adjusted p-values and, on C, TOST and the outcome."""
    show = _show
    lines = []
    for name, c in result['contrasts'].items():
        lines.append(f"{name}: {c['a']['variant']} - {c['b']['variant']}, {len(c['units'])} unit(s), "
                     f"{c['windows']} window(s): {c['outcome']['reading']}")
        for metric, s in c['metrics'].items():
            t = s['t']
            text = (f"  {metric}: mean {show(t['mean'])} [{show(t['lo'])}, {show(t['hi'])}] over {t['n']} unit(s), "
                    f"pooled {show(s['pooled_delta'])}, p_t {show(t['p'], False)} (Holm {show(s['p_t_holm'], False)}), "
                    f"sign-flip {show(s['sign_flip']['p'], False)} (Holm {show(s['p_sign_flip_holm'], False)})")
            if s.get('tost'):
                interval = s['tost']['interval'] or {}
                text += (f", TOST +-{s['tost']['margin']}: [{show(interval.get('lo'))}, {show(interval.get('hi'))}] "
                         f"p {show(s['tost']['p'], False)}")
            lines.append(text)
    return '\n'.join(lines)


def ledger_line(result: dict, out) -> dict:
    """the run ledger's line for a look at results through contrasts: what was read and written."""
    out = Path(out).resolve()
    return {'run': out.name, 'run_dir': str(out), 'status': 'looked_at', 'kind': 'contrast',
            'at': datetime.now(timezone.utc).isoformat(), 'family': result['family'],
            'contrasts': {name: {'a': c['a'], 'b': c['b'], 'outcome': c['outcome']['reading'], 'coded': c.get('coded')}
                          for name, c in result['contrasts'].items()},
            'margin': result['margin'], 'alpha': result['alpha'], 'subset': result.get('subset', False),
            'inputs': result['inputs'],
            'outputs': {path.name: _digest(path) for path in sorted(out.iterdir()) if path.is_file()}}
