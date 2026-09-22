"""HMM smoothing of the 10 s interaction posteriors, in numpy, over every window of a session.

The states are the classes. A calibrated posterior p(k|x_t) divided by the prior pi_k is, up to a
constant, the likelihood of the window's features under state k, so it can serve as the HMM's
emission; the exponent gamma (< 1) damps evidence the model has already reused (lag columns, the
network's temporal convolution), and a window where no modality ran gets a flat emission and is
carried by its neighbours. The transitions are counted from coded, adjacent window pairs of the
training sessions with a sticky Dirichlet prior, since coders sample 5-minute blocks and only
pairs within a block exist.

Forward-backward posteriors are the offline answer (every metric uses them), the forward filter
the causal one (online, and for onset latency), and the Viterbi path is used only for run lengths.
Everything works for any number of states, so the binary check is the same code with two.
"""
from __future__ import annotations

import numpy as np

# the emission exponents gamma is chosen from, on inner out-of-fold NLL of the smoothed posteriors
GAMMAS = (0.5, 0.75, 1.0)
# the least probability a posterior may hold before its log is taken
FLOOR = 1e-6


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    top = np.max(a, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return np.squeeze(top, axis=axis) + np.log(np.sum(np.exp(a - top), axis=axis))


def _series(y) -> tuple[np.ndarray, np.ndarray]:
    """(window index, label) of a session's labels: a Series keeps its index, anything else counts
    its positions as consecutive windows."""
    if hasattr(y, 'index') and hasattr(y, 'to_numpy'):
        return np.asarray(y.index, dtype=float), np.asarray(y.to_numpy(), dtype=float)
    values = np.asarray(y, dtype=float).reshape(-1)
    return np.arange(len(values), dtype=float), values


def transitions(y_by_session, alpha: float = 1.0, diagonal: float = 10.0, n_states: int = 3) -> np.ndarray:
    """the transition matrix, rows summing to 1: counts of (y_t, y_t+1) over pairs where both
    windows are coded (a class, not -1 or unclear) and adjacent in window_index, plus a Dirichlet
    prior of `alpha` on every cell and `diagonal` more on the diagonal. `y_by_session` is a dict
    or a list of per-session labels over the session's grid (a Series indexed by window_index,
    or an array whose positions are consecutive windows)."""
    counts = np.zeros((n_states, n_states))
    sessions = y_by_session.values() if isinstance(y_by_session, dict) else y_by_session
    for y in sessions:
        index, values = _series(y)
        coded = np.isfinite(values) & (values >= 0) & (values < n_states)
        pair = coded[:-1] & coded[1:] & (np.diff(index) == 1)
        np.add.at(counts, (values[:-1][pair].astype(int), values[1:][pair].astype(int)), 1)
    counts += alpha + diagonal * np.eye(n_states)
    return counts / counts.sum(1, keepdims=True)


def emissions(p, prior, gamma: float, flat=None) -> np.ndarray:
    """log e_t(k) = gamma * (log p(k|x_t) - log pi_k): the calibrated posterior over the prior as a
    scaled likelihood, damped by gamma. It is 0 (a flat emission) where `flat` is true (no
    modality ran) and where the window has no posterior."""
    p = np.asarray(p, dtype=float)
    log_e = gamma * (np.log(np.clip(p, FLOOR, 1.0)) - np.log(np.clip(np.asarray(prior, dtype=float), FLOOR, 1.0)))
    none = ~np.isfinite(p).all(1)
    if flat is not None:
        none |= np.asarray(flat, dtype=bool)
    log_e[none] = 0.0
    return log_e


def forward_backward(log_e, log_A, log_pi) -> np.ndarray:
    """the smoothed posteriors p(s_t | every window of the session), (T, K), rows summing to 1."""
    log_e, log_A, log_pi = np.asarray(log_e, dtype=float), np.asarray(log_A, dtype=float), np.asarray(log_pi, dtype=float)
    T, K = log_e.shape
    if T == 0:
        return np.zeros((0, K))
    forward = _forward(log_e, log_A, log_pi)
    backward = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        backward[t] = _logsumexp(log_A + (log_e[t + 1] + backward[t + 1])[None, :], axis=1)
    joint = forward + backward
    return np.exp(joint - _logsumexp(joint, axis=1)[:, None])


def _forward(log_e: np.ndarray, log_A: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    T, K = log_e.shape
    forward = np.zeros((T, K))
    forward[0] = log_pi + log_e[0]
    for t in range(1, T):
        forward[t] = log_e[t] + _logsumexp(forward[t - 1][:, None] + log_A, axis=0)
    return forward


def forward_filter(log_e, log_A, log_pi) -> np.ndarray:
    """the filtered posteriors p(s_t | windows up to t), (T, K): what an online classifier knows at
    the end of each window, never reading a later one."""
    log_e, log_A, log_pi = np.asarray(log_e, dtype=float), np.asarray(log_A, dtype=float), np.asarray(log_pi, dtype=float)
    if len(log_e) == 0:
        return np.zeros((0, log_e.shape[1]))
    forward = _forward(log_e, log_A, log_pi)
    return np.exp(forward - _logsumexp(forward, axis=1)[:, None])


def viterbi(log_e, log_A, log_pi) -> np.ndarray:
    """the most probable state path, (T,) ints; used only for run lengths. A tie keeps the lower
    state."""
    log_e, log_A, log_pi = np.asarray(log_e, dtype=float), np.asarray(log_A, dtype=float), np.asarray(log_pi, dtype=float)
    T, K = log_e.shape
    if T == 0:
        return np.zeros(0, dtype=int)
    score = log_pi + log_e[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        options = score[:, None] + log_A
        back[t] = options.argmax(0)
        score = options[back[t], np.arange(K)] + log_e[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def smooth(p, prior, A, gamma: float, flat=None, mode: str = 'fb') -> np.ndarray:
    """one session's posteriors through the HMM: forward-backward (mode 'fb', offline), the forward
    filter ('filter', causal) or the Viterbi path as one-hot rows ('viterbi'). The chain starts
    from the prior, which is also what the emissions are divided by."""
    log_e = emissions(p, prior, gamma, flat)
    log_A = np.log(np.asarray(A, dtype=float))
    log_pi = np.log(np.clip(np.asarray(prior, dtype=float), FLOOR, 1.0))
    if mode == 'fb':
        return forward_backward(log_e, log_A, log_pi)
    if mode == 'filter':
        return forward_filter(log_e, log_A, log_pi)
    if mode == 'viterbi':
        return np.eye(len(log_pi))[viterbi(log_e, log_A, log_pi)]
    raise ValueError(f"mode must be 'fb', 'filter' or 'viterbi', not {mode!r}")


def select_gamma(p_by_session, y_by_session, prior, A, flat_by_session=None, gammas=GAMMAS,
                 mode: str = 'fb') -> tuple[float, dict]:
    """the gamma whose smoothed inner out-of-fold posteriors have the lowest (unweighted) NLL on the
    coded windows, the smaller one on a tie, and the NLL of each. The three arguments by session
    are dicts with the same keys (or lists in the same order), over each session's whole grid."""
    keys = list(p_by_session) if isinstance(p_by_session, dict) else list(range(len(p_by_session)))
    scores = {}
    for gamma in gammas:
        total, count = 0.0, 0
        for key in keys:
            flat = flat_by_session[key] if flat_by_session is not None else None
            posterior = smooth(p_by_session[key], prior, A, gamma, flat, mode)
            _, y = _series(y_by_session[key])
            coded = np.isfinite(y) & (y >= 0) & (y < posterior.shape[1])
            if coded.any():
                chosen = posterior[np.flatnonzero(coded), y[coded].astype(int)]
                total -= np.log(np.clip(chosen, FLOOR, 1.0)).sum()
                count += int(coded.sum())
        scores[float(gamma)] = float(total / count) if count else float('nan')
    finite = {g: s for g, s in scores.items() if np.isfinite(s)}
    best = min(finite, key=lambda g: (finite[g], g)) if finite else float(gammas[-1])
    return best, scores
