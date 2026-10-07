"""pmil-lr, an exploratory candidate of the architecture panel: a multiple-instance
logistic model over the pupil pairs of a window. It is never in -m all and stays exploratory
whatever it scores; the declared evaluation does not read it.

An instance is one pair slot that holds a pair: the pair's 14 values and 8 masks (Q), then the
elementwise min and max of its two pupils' 19 values and 9 masks (P), 78 columns that do not
depend on which of the two is which, so no slot order reaches the model. The values are the scaled
tokens' with NaN where a mask says unobserved (layout's own reading), handled as lr handles the
pooled view: the median of the training instances fills a missing value (the mask columns say it
was missing), then standardising; the min and max are over the pupils that observed a value. The
window's speech values are in no instance, since a pair has none of its own.

One linear scorer is shared by every pair, s = w.x + b_g, with one bias per group size g (the
roster's, 2 or 3; a size training never saw takes the nearest one's), and the window's
p(interaction) is the noisy-OR of its pairs, 1 - prod_j (1 - sigmoid(s_j)): any pair can make the
window an interaction, and a triad's three pairs, three chances to fire against a dyad's one (the
inflation of the pooled view's max columns in triads), are taken up by its own bias. The fit is
lr's: balanced class weights, an L2 penalty |w|^2 / (2 C) on the scorer (never on the biases), C
chosen from lr's grid on the inner folds (tabular.select), then the calibrator. A window that holds
no pair (a roster of one, which S1 leaves out) fits nothing and answers 0.5, the balanced fit's no
evidence, which the calibrator's bias turns towards the prior.

It answers p(interaction) only, a noisy-OR having no social against collaborative, so it runs with
the binary target alone: the driver refuses it with the three-class one, and fit refuses any label
but 0 and 1.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from openmmla.analytics.interaction import layout as LY

N_PAIRS = len(LY.PAIR_INDEX)
N_PERSON_VALUES, N_PAIR_VALUES = len(LY.PERSON_VALUES), len(LY.PAIR_VALUES)
# an instance: the pair's values and masks, then the min and the max of its two pupils' values and masks
INSTANCE_COLUMNS = tuple(LY.Q_COLUMNS) + tuple(f'{name}_min' for name in LY.P_COLUMNS) \
    + tuple(f'{name}_max' for name in LY.P_COLUMNS)
D_INSTANCE = len(INSTANCE_COLUMNS)  # 78 in layout version 4
# a window's row: each pair slot's instance (NaN where the slot holds no pair), which slots hold one, the group size
ROW_WIDTH = N_PAIRS * D_INSTANCE + N_PAIRS + 1
# the least probability the noisy-OR answers (tabular.PROBA_FLOOR)
FLOOR = 1e-6
# the least summed softplus of a window's pairs, so log p(interaction) and its gradient stay finite
MIN_EVIDENCE = 1e-12


def window_rows(tokens) -> np.ndarray:
    """a session's windows as the model's rows (T, ROW_WIDTH): per pair slot its instance
    (INSTANCE_COLUMNS, NaN where a value was not observed, and throughout where the slot holds no
    pair), then 1 for each slot that holds a pair, then the roster's group size. `tokens` are
    layout's Tokens, as scaled for the fold."""
    persons, _ = LY._observed(tokens, 'P')
    pairs, _ = LY._observed(tokens, 'Q')
    persons = np.concatenate([persons, np.asarray(tokens.P, dtype=float)[..., N_PERSON_VALUES:]], axis=-1)
    pairs = np.concatenate([pairs, np.asarray(tokens.Q, dtype=float)[..., N_PAIR_VALUES:]], axis=-1)
    index = np.asarray(tokens.pair_index, dtype=int)
    first, second = persons[:, index[:, 0]], persons[:, index[:, 1]]
    # fmin and fmax keep the one pupil's value where the other's is NaN
    instances = np.concatenate([pairs, np.fmin(first, second), np.fmax(first, second)], axis=-1)
    exists = np.asarray(tokens.Q_exists, dtype=float) > 0
    instances[~exists] = np.nan
    size = np.round(np.asarray(tokens.G, dtype=float)[:, -1] * LY.N_SLOTS)
    return np.column_stack([instances.reshape(len(instances), -1), exists.astype(float), size])


def _unpack(X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """window rows as (instances (n, 3, 78), which slots hold a pair (n, 3), group size (n,))."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != ROW_WIDTH:
        raise ValueError(f"pmil-lr reads window rows of {ROW_WIDTH} columns (window_rows), not {X.shape}")
    instances = X[:, :N_PAIRS * D_INSTANCE].reshape(len(X), N_PAIRS, D_INSTANCE)
    return instances, X[:, N_PAIRS * D_INSTANCE:-1] > 0, X[:, -1]


def noisy_or(scores, window, n: int) -> np.ndarray:
    """p(interaction) of n windows from their pairs' scores: 1 - prod (1 - sigmoid(s)) over the
    pairs of each window (`window` gives each score's window), 0 for a window with none."""
    evidence = np.bincount(np.asarray(window, dtype=int), weights=np.logaddexp(0.0, scores), minlength=n)
    return -np.expm1(-evidence)


class PairMIL:
    """the model, as tabular's helpers handle a classifier: fit(X, y, sample_weight) on window rows
    (window_rows) labelled 0 individual or 1 interaction, predict_proba(X) (n, 2), classes_ (0, 1).
    Fitted, `coef_` is the shared scorer's weights over the standardised INSTANCE_COLUMNS and
    `group_bias_` its bias per group size (`sizes_`)."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C, self.max_iter = C, max_iter

    def _standardised(self, x: np.ndarray) -> np.ndarray:
        return (np.where(np.isnan(x), self.median_, x) - self.mean_) / self.scale_

    def _bias_index(self, size: np.ndarray) -> np.ndarray:
        """each window's bias: its group size's, or the nearest size training saw (the smaller on a tie)."""
        distance = np.abs(np.nan_to_num(np.asarray(size, dtype=float), nan=0.0)[:, None] - self.sizes_[None, :])
        return distance.argmin(axis=1)

    def fit(self, X, y, sample_weight=None):
        objective, n_biases = self._prepare(X, y, sample_weight)
        if objective is None:
            return self
        result = minimize(objective, np.zeros(D_INSTANCE + n_biases), jac=True, method='L-BFGS-B',
                          options={'maxiter': int(self.max_iter)})
        self.coef_, self.group_bias_ = result.x[:D_INSTANCE].copy(), result.x[D_INSTANCE:].copy()
        self.n_iter_ = int(result.nit)
        return self

    def _prepare(self, X, y, sample_weight=None):
        """the imputation and standardisation of the training instances, the group sizes, and
        (the objective with its gradient over the scorer's weights then the biases, how many biases),
        or (None, 0) when no window holds a pair. The objective is lr's over the windows that fit:
        sum_i c_i loss_i / n + |w|^2 / (2 C n), c_i the balanced class weight times the sample
        weight, loss_i = -log p_i for an interaction and -log (1 - p_i), the summed softplus of its
        pairs' scores, for an individual window."""
        y = np.asarray(y, dtype=float).reshape(-1)
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("pmil-lr learns individual (0) against interaction (1) only: run it with the binary "
                             "target")
        y = y.astype(int)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        instances, exists, size = _unpack(X)
        self.classes_ = np.array([0, 1])
        observed = instances[exists]
        with warnings.catch_warnings():
            # a column no training instance observed has no median: it is 0, as SimpleImputer keeps an empty one
            warnings.simplefilter('ignore', RuntimeWarning)
            median = np.nanmedian(observed, axis=0) if len(observed) else np.zeros(D_INSTANCE)
        self.median_ = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isnan(observed), self.median_, observed)
        self.mean_ = filled.mean(axis=0) if len(filled) else np.zeros(D_INSTANCE)
        spread = filled.std(axis=0) if len(filled) else np.ones(D_INSTANCE)
        self.scale_ = np.where(spread > 0, spread, 1.0)
        # a window with no pair has nothing to explain its label: it fits nothing
        keep = exists.any(axis=1)
        self.sizes_ = np.unique(np.nan_to_num(size[keep], nan=0.0)) if keep.any() else np.array([2.0])
        self.coef_ = np.zeros(D_INSTANCE)
        self.group_bias_ = np.zeros(len(self.sizes_))
        if not keep.any():
            return None, 0
        labels = y[keep]
        # lr's balanced class weights over the windows that fit, times their sample weights
        counts = np.bincount(labels, minlength=2).astype(float)
        balanced = np.where(counts > 0, len(labels) / (np.count_nonzero(counts) * np.maximum(counts, 1.0)), 0.0)
        c = balanced[labels] * weight[keep]
        rows, slots = np.nonzero(exists[keep])
        Z = self._standardised(instances[keep][rows, slots])
        bias_of = self._bias_index(size[keep])[rows]
        n, d, C, k = len(labels), D_INSTANCE, float(self.C), len(self.sizes_)
        positive = labels == 1

        def objective(theta):
            w, b = theta[:d], theta[d:]
            s = Z @ w + b[bias_of]
            evidence = np.maximum(np.bincount(rows, weights=np.logaddexp(0.0, s), minlength=n), MIN_EVIDENCE)
            # -log p for an interaction, -log (1 - p) = the evidence for an individual window, and their slopes
            # in the evidence (-1 / (e^A - 1), written so that a large A does not overflow, and 1)
            p = -np.expm1(-evidence)
            loss = np.where(positive, -np.log(p), evidence)
            slope = np.where(positive, -np.exp(-evidence) / p, 1.0)
            share = (c * slope)[rows] * expit(s) / n
            value = float((c * loss).sum() / n + w @ w / (2.0 * C * n))
            return value, np.concatenate([Z.T @ share + w / (C * n), np.bincount(bias_of, weights=share, minlength=k)])

        return objective, k

    def pair_scores(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(each pair's score s = w.x + b_g, its window, its slot) over the pairs the windows hold."""
        instances, exists, size = _unpack(X)
        rows, slots = np.nonzero(exists)
        if not len(rows):
            return np.zeros(0), rows, slots
        bias = self.group_bias_[self._bias_index(size)[rows]]
        scores = self._standardised(instances[rows, slots]) @ self.coef_ + bias
        return scores, rows, slots

    def predict_proba(self, X) -> np.ndarray:
        """(n, 2): p(individual), p(interaction), the noisy-OR of the window's pairs; 0.5 each for a
        window with no pair; never below FLOOR."""
        _, exists, _ = _unpack(X)
        scores, rows, _ = self.pair_scores(X)
        p = np.where(exists.any(axis=1), noisy_or(scores, rows, len(exists)), 0.5)
        p = np.clip(p, FLOOR, 1.0 - FLOOR)
        return np.column_stack([1.0 - p, p])

    @property
    def group_bias(self) -> dict:
        """the bias per group size, as metrics.json keeps it."""
        return {int(size): float(bias) for size, bias in zip(self.sizes_, self.group_bias_)}


def make_pmil(C: float = 1.0):
    """pmil-lr with this L2 strength (the grid is lr's, tabular.LR_GRID)."""
    return PairMIL(C=C)
