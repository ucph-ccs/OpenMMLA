"""The tabular models of the 10 s interaction classifier (Layer A): the a-priori rule, the fitted
two-level tree, logistic regression and gradient boosting on the pooled view, the late fusion of
one expert per modality, the temperature-and-bias calibrator, and the balanced decision.

Every model here reads the pooled view as a DataFrame (NaN where a value could not be observed,
the masks as columns of their own) and labels as ints: 0 individual, 1 social, 2 collaborative,
and -1 for a window that is not coded (or is unclear), which no fit and no score reads. What a
model answers is a log-probability per class (its logits), so the calibrator, the stacker and the
HMM all take one format. Nothing here knows the layout beyond the names it is handed: the experts
get their block's columns from the caller, and the rule's names sit in RULES.

Selection, calibration and stacking all run on the same inner splits, which hold out whole
lessons: windows ten seconds apart are near copies, so a split that scatters them (a random
K-fold, CalibratedClassifierCV, early stopping's validation share) would score a model on windows
it has as good as seen.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

N_CLASSES = 3
# the least probability a class gets: a class no training window had, or a tree leaf that holds
# none of it, would otherwise give a logit of -inf
PROBA_FLOOR = 1e-6

LR_GRID = [{'C': c} for c in (0.01, 0.03, 0.1, 0.3, 1.0)]
HGB_GRID = [{'max_iter': iterations, 'max_leaf_nodes': leaves, 'min_samples_leaf': leaf}
            for iterations in (100, 200, 400) for leaves in (8, 16) for leaf in (40, 100)]

# the a-priori rule, by version: the pooled columns each role reads (layout.pooled of unscaled
# tokens; a role sums its columns) and the thresholds, fixed from the codebook's wording before any
# label was read and never tuned. The thresholds are in the fusion table's units (shares, counts,
# frame widths), so the values must be unscaled; a column named log_* holds log1p of the count and
# is turned back before the threshold
RULES = {
    1: {'fixed': '2026-09-23 (ff81d73), layout version 2, before any label was read',
        'columns': {
            'speech_ratio': ('speech_ratio',),
            'words': ('log_words',),
            'dia_switches': ('log_dia_switches',),
            'm_dia': ('m_dia',),
            # partner gaze counted every other person before layout version 3: partner + other
            'partner_face': ('partner_face_mean', 'other_face_mean'),
            'hands': ('partner_hands_mean', 'other_hands_mean', 'own_hands_mean'),
            'joint_attention': ('joint_attention_ratio_max',),
            'hand_dist_min': ('hand_dist_min_min',),
        },
        'thresholds': {
            'speech_ratio': 0.3,      # talk: at least this share of the window held speech
            'dia_switches': 1,        # ... and the turn passed at least once (diarized windows)
            'words': 5,               # ... or at least this many words (no diarization)
            'partner_face': 0.2,      # look: mean share of readable gaze on a partner's face
            'joint_attention': 0.3,   # shared focus: the best pair's joint-attention share
            'hand_dist_min': 0.05,    # ... or two pairs of hands this close, in frame widths
            'hands': 0.4,             # collaborative: mean share of gaze on hands (own or a partner's)
        }},
    2: {'fixed': '2026-09-23, layout version 3, from the codebook wording, before any label was read',
        'columns': {
            'speech_ratio': ('speech_ratio',),
            'words': ('log_words',),
            'dia_switches': ('log_dia_switches',),
            'm_dia': ('m_dia',),
            'partner_face': ('partner_face_mean',),
            'task_gaze': ('partner_hands_mean', 'own_hands_mean', 'work_area_mean'),
            'watching': ('partner_hands_max',),
            'joint_attention': ('joint_attention_excess_max',),
            'hand_dist_min': ('hand_dist_min_min',),
        },
        'thresholds': {
            'speech_ratio': 0.3,      # talk ("Members interact (talk, ...)"): v1's thresholds
            'dia_switches': 1,
            'words': 5,
            'partner_face': 0.2,      # look ("look at each other"): v1's glance bound, the partner in-group
            'joint_attention': 0.3,   # shared focus: joint attention this far above the pair's own rate 20-40 s earlier
            'hand_dist_min': 0.05,    # ... or two pairs of hands this close ("handing over"), in frame widths
            'task_gaze': 0.4,         # collaborative: gaze on the task (hands, and the work area around them)
            'watching': 0.5,          # a pupil watching a partner's hands for most of the window
        }},
}
RULE_VERSION = 2
# the current rule's columns and thresholds, by their older names
RULE_COLUMNS = RULES[RULE_VERSION]['columns']
RULE_THRESHOLDS = RULES[RULE_VERSION]['thresholds']


# ---- labels and priors ----

def _labels(y, n_classes: int = N_CLASSES) -> np.ndarray:
    """labels as ints, anything that is not a class (NaN, -1, unclear) as -1."""
    values = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(values) & (values >= 0) & (values < n_classes)
    out = np.full(len(values), -1, dtype=int)
    out[valid] = values[valid].astype(int)
    return out


def _rows(X, index):
    return X.iloc[index] if hasattr(X, 'iloc') else np.asarray(X)[index]


def class_prior(y, n_classes: int = N_CLASSES) -> np.ndarray:
    """the class frequencies of the coded windows, each at least PROBA_FLOOR so that its log is
    finite; uniform when nothing is coded."""
    y = _labels(y, n_classes)
    counts = np.bincount(y[y >= 0], minlength=n_classes).astype(float)
    if counts.sum() == 0:
        return np.full(n_classes, 1.0 / n_classes)
    prior = np.maximum(counts / counts.sum(), PROBA_FLOOR)
    return prior / prior.sum()


def nll(logits, y) -> float:
    """the mean negative log-likelihood of the true classes, over coded rows with an answer."""
    logits, y = np.asarray(logits, dtype=float), _labels(y, np.shape(logits)[1])
    keep = (y >= 0) & np.isfinite(logits).all(1)
    if not keep.any():
        return float('nan')
    z = logits[keep]
    log_p = z - _logsumexp(z)
    return float(-log_p[np.arange(len(z)), y[keep]].mean())


def class_weighted_nll(logits, y) -> float:
    """the NLL with each class weighing as much as the others (w_c = N / (K N_c)), so that a
    hyperparameter is not chosen for how well it predicts the commonest class."""
    logits = np.asarray(logits, dtype=float)
    k = logits.shape[1]
    y = _labels(y, k)
    keep = (y >= 0) & np.isfinite(logits).all(1)
    if not keep.any():
        return float('nan')
    z, labels = logits[keep], y[keep]
    counts = np.bincount(labels, minlength=k).astype(float)
    weights = len(labels) / (k * counts[labels])
    log_p = z - _logsumexp(z)
    return float(-(weights * log_p[np.arange(len(z)), labels]).sum() / weights.sum())


def _logsumexp(z: np.ndarray) -> np.ndarray:
    top = z.max(1, keepdims=True)
    return top + np.log(np.exp(z - top).sum(1, keepdims=True))


# ---- the rule ----

def _column(pooled, name: str) -> np.ndarray:
    if name not in pooled.columns:
        raise KeyError(f"the pooled view has no column {name!r} (see RULES)")
    values = pooled[name].to_numpy(dtype=float)
    # a log1p column goes back to its count, so the threshold keeps the codebook's unit
    return np.expm1(values) if name.startswith('log_') else values


def _role(pooled, names) -> np.ndarray:
    """a role's value: the sum of its columns. The first keeps its NaN (a condition on it is then
    false); an added one counts 0 where it is NaN (a table fused before layout version 3 has no
    other_* shares)."""
    names = (names,) if isinstance(names, str) else tuple(names)
    total = _column(pooled, names[0]).copy()
    for name in names[1:]:
        total = total + np.nan_to_num(_column(pooled, name), nan=0.0)
    return total


def rule_a_priori(pooled, columns: dict | None = None, version: int = RULE_VERSION) -> np.ndarray:
    """R0, the no-label rule (the peer of zero-shot Jev), from thresholds fixed by the codebook's
    wording (RULES). A condition on something unobserved (NaN) is false, so a window nobody could
    see or hear is individual. It gives hard labels only (0, 1, 2). `columns` overrides a role's
    columns (a name or a tuple of names).

    Version 2 (layout version 3): talk (speech, and a change of speaker or enough words), look (an
    in-group partner's face), a shared focus (joint attention above the pair's own rate 20-40 s
    earlier, or hands close enough to hand over) or watching (a pupil's gaze on a partner's hands
    for most of the window, silent or not) make an interaction; a shared focus or watching make it
    collaborative, as does an interaction with the gaze on the task (hands and the work area).
    Version 1 (layout version 2): talk, look or the raw joint attention or near hands make an
    interaction, and a shared focus or eyes on the hands make it collaborative; on a version 3
    view it reads partner + other, the partner gaze of layout version 2."""
    if version not in RULES:
        raise ValueError(f"unknown rule version {version!r}: one of {sorted(RULES)}")
    names = dict(RULES[version]['columns'], **(columns or {}))
    th = RULES[version]['thresholds']
    with np.errstate(invalid='ignore'):
        speech = _role(pooled, names['speech_ratio'])
        words = _role(pooled, names['words'])
        switches = _role(pooled, names['dia_switches'])
        diarized = np.nan_to_num(_role(pooled, names['m_dia'])) > 0
        talk = (speech >= th['speech_ratio']) & np.where(diarized, switches >= th['dia_switches'], words >= th['words'])
        look = _role(pooled, names['partner_face']) >= th['partner_face']
        shared = (_role(pooled, names['joint_attention']) >= th['joint_attention']) \
            | (_role(pooled, names['hand_dist_min']) <= th['hand_dist_min'])
        if version == 1:
            hands = _role(pooled, names['hands'])
            interaction = talk | look | shared
            collaborative = interaction & (shared | (hands >= th['hands']))
        else:
            watching = _role(pooled, names['watching']) >= th['watching']
            task = _role(pooled, names['task_gaze']) >= th['task_gaze']
            interaction = talk | look | shared | watching
            collaborative = shared | watching | (interaction & task)
    return np.where(collaborative, 2, np.where(interaction, 1, 0)).astype(int)


# ---- the models ----

def make_rule_tree():
    """R1, the fitted rule: four leaves at most, each holding at least 100 windows; NaN goes down
    the side the tree learned for it (sklearn >= 1.3). Its leaf frequencies are calibrated after."""
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=2, min_samples_leaf=100, class_weight='balanced', random_state=0)


def make_lr(C: float = 1.0):
    """logistic regression on the pooled view: the median of the training fold fills a missing
    value (the mask columns say it was missing), then standardising, then a multinomial L2 model
    with balanced class weights; the calibrator gives the prior back."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # lbfgs is multinomial for three classes by default; multi_class is deprecated in sklearn 1.5
    # an empty column of the training fold is kept (as 0), so the weights keep their columns
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=C, class_weight='balanced', max_iter=2000))


def make_hgb(**params):
    """gradient boosting with NaN handled natively. Early stopping is off: its validation share is
    drawn at random from autocorrelated windows, so it would stop on windows it has as good as
    seen. random_state fixes the column subsampling."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    settings = dict(learning_rate=0.05, l2_regularization=1.0, max_features=0.5, class_weight='balanced',
                    early_stopping=False, random_state=0)
    settings.update(params)
    return HistGradientBoostingClassifier(**settings)


class _Prior:
    """the stand-in for a model whose training windows hold fewer than two classes, or none: it
    answers the training prior everywhere."""

    def __init__(self, prior: np.ndarray):
        self.prior = np.asarray(prior, dtype=float)
        self.classes_ = np.arange(len(self.prior))

    def predict_proba(self, X) -> np.ndarray:
        return np.tile(self.prior, (len(X), 1))


def fit_model(make, params: dict | None, X, y, sample_weight=None, n_classes: int = N_CLASSES):
    """make(**params) fitted on the coded rows, each weighing `sample_weight` when given (the
    session-weight ablation); the training prior instead when those rows hold fewer than two
    classes (an inner fold can lack one)."""
    y = _labels(y, n_classes)
    coded = np.flatnonzero(y >= 0)
    if len(np.unique(y[coded])) < 2:
        return _Prior(class_prior(y, n_classes))
    model = make(**(params or {}))
    if sample_weight is None:
        return model.fit(_rows(X, coded), y[coded])
    # a pipeline hands the weights to its last step by that step's name
    key = f'{model.steps[-1][0]}__sample_weight' if hasattr(model, 'steps') else 'sample_weight'
    return model.fit(_rows(X, coded), y[coded], **{key: np.asarray(sample_weight, dtype=float)[coded]})


def session_weights(sessions, y) -> np.ndarray:
    """1 / sqrt(coded windows of the session) per row, scaled to a mean of 1 over the coded rows:
    the exploratory ablation that keeps one long session from outweighing the rest. A session with
    no coded window weighs 0."""
    sessions, y = np.asarray(sessions, dtype=object), _labels(y)
    weights = np.zeros(len(y))
    for session in np.unique(sessions[y >= 0]):
        rows = sessions == session
        weights[rows] = 1.0 / np.sqrt((y[rows] >= 0).sum())
    coded = y >= 0
    return weights / weights[coded].mean() if coded.any() else weights


def log_proba(model, X, n_classes: int = N_CLASSES) -> np.ndarray:
    """log predict_proba over every class: a class the model never saw, or a probability of 0, is
    floored at PROBA_FLOOR and the row renormalised, so every model answers an (n, 3) array of
    finite logits."""
    if len(X) == 0:
        return np.zeros((0, n_classes))
    proba = np.asarray(model.predict_proba(X), dtype=float)
    full = np.full((len(proba), n_classes), PROBA_FLOOR)
    full[:, np.asarray(model.classes_, dtype=int)] = np.maximum(proba, PROBA_FLOOR)
    return np.log(full / full.sum(1, keepdims=True))


# ---- inner cross-validation ----

def inner_splits(y, groups, inner=4) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train rows, held-out rows) of the inner cross-validation, over every row (coded or not),
    whole groups (lessons) at a time. `inner` is a number of folds (GroupKFold over the groups
    that hold coded windows, balanced by their coded windows), a splitter with split(X, y,
    groups), or the splits themselves: (train, held-out) pairs or objects with .train and .test,
    as row positions or as labels of `groups` (which must then be the same kind: lessons, not
    sessions). A group with no coded window is never held out, so its rows get no out-of-fold
    answer."""
    y, groups = _labels(y), np.asarray(groups)
    if isinstance(inner, (int, np.integer)):
        from sklearn.model_selection import GroupKFold
        coded = y >= 0
        names = np.unique(groups[coded])
        if len(names) < 2:
            raise ValueError("the inner cross-validation needs at least 2 groups with coded windows")
        folds = GroupKFold(n_splits=min(int(inner), len(names)))
        pairs = [(np.unique(groups[coded][train]), np.unique(groups[coded][test]))
                 for train, test in folds.split(np.zeros(coded.sum()), y[coded], groups[coded])]
    elif hasattr(inner, 'split'):
        pairs = [(np.unique(groups[train]), np.unique(groups[test]))
                 for train, test in inner.split(np.zeros(len(y)), y, groups)]
    else:
        # a fold object (splits.Fold) gives its training and held-out groups by name
        pairs = [(fold.train, fold.test) if hasattr(fold, 'train') else fold for fold in inner]
    splits = []
    for train, test in pairs:
        train, test = np.asarray(train), np.asarray(test)
        if train.dtype.kind in 'iu' and test.dtype.kind in 'iu':
            splits.append((train, test))
        else:
            # group labels: every row of those groups
            held = np.flatnonzero(np.isin(groups, test))
            if len(test) and not len(held):
                raise ValueError(f"no row belongs to the held-out groups {list(test)[:3]}: are the splits "
                                 f"in the same groups (lessons or sessions) as `groups`?")
            splits.append((np.flatnonzero(np.isin(groups, train)), held))
    return splits


def select(make, grid, X, y, groups, inner=4, sample_weight=None) -> tuple[dict, np.ndarray]:
    """the grid point whose inner out-of-fold predictions have the lowest class-weighted NLL (the
    first one on a tie), and those predictions: (n, 3) log-probabilities, each row answered by
    the model that did not see its lesson, NaN for a row that was never held out. The same
    out-of-fold logits then fit the calibrator and choose the HMM's gamma. `sample_weight`
    weighs the fits, not the score."""
    y = _labels(y)
    splits = inner_splits(y, groups, inner)
    best = None
    for params in grid:
        oof = np.full((len(y), N_CLASSES), np.nan)
        for train, test in splits:
            if len(test) == 0:
                continue
            weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[train]
            model = fit_model(make, params, _rows(X, train), y[train], weight)
            oof[test] = log_proba(model, _rows(X, test))
        score = class_weighted_nll(oof, y)
        score = score if np.isfinite(score) else np.inf
        if best is None or score < best[0]:
            best = (score, dict(params), oof)
    return best[1], best[2]


# ---- late fusion ----

def presence(*columns):
    """a presence flag from pooled columns: every one of them observed and above 0, e.g.
    presence('ips_ran', 'share_present') for the space expert."""
    def flag(X) -> np.ndarray:
        on = np.ones(len(X), dtype=bool)
        for name in columns:
            on &= np.nan_to_num(X[name].to_numpy(dtype=float)) > 0
        return on
    return flag


# the experts' presence flags, by modality (2.6): speech = m_asr; space = IPS ran and someone was
# present; body = VFA ran and someone was seen. The share columns must be unscaled.
PRESENCE = {
    'speech': presence('m_asr'),
    'space': presence('ips_ran', 'share_present'),
    'body_gaze': presence('vfa_ran', 'share_seen'),
}


class Expert:
    """one modality's model in the late fusion. It reads only its block's columns, learns only from
    the windows where its modality was present, and answers the training log-prior where the
    modality is absent, so an outage reads as 'no evidence' rather than as whatever the imputer
    made of it."""

    def __init__(self, make, grid, columns, flag=None, name: str | None = None):
        self.make, self.grid, self.columns, self.flag, self.name = make, list(grid), list(columns), flag, name

    def present(self, X) -> np.ndarray:
        """whether the modality was there, per row: the flag (a callable or a column name), or
        always when there is none."""
        if self.flag is None:
            return np.ones(len(X), dtype=bool)
        if callable(self.flag):
            return np.asarray(self.flag(X), dtype=bool)
        return np.nan_to_num(X[self.flag].to_numpy(dtype=float)) > 0

    def fit(self, X, y, groups, inner=4, sample_weight=None):
        """chooses the grid point on the inner splits, keeps the out-of-fold logits of every row
        (`oof_logits_`, the fold's training prior where the modality was absent), then refits on
        every coded window where the modality was present."""
        y, groups = _labels(y), np.asarray(groups)
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        splits = inner_splits(y, groups, inner)
        here = self.present(X)
        self.prior_ = class_prior(y)
        rows = np.flatnonzero(here)
        position = np.full(len(y), -1)
        position[rows] = np.arange(len(rows))
        oof = np.full((len(y), N_CLASSES), np.nan)
        for train, test in splits:
            oof[test] = np.log(class_prior(y[train]))
        sub_y = y[rows]
        if len(np.unique(sub_y[sub_y >= 0])) < 2:
            # the modality was (as good as) never there in training: the expert is the prior
            self.params_, self.model_ = None, _Prior(self.prior_)
        else:
            sub_splits = [(position[train[here[train]]], position[test[here[test]]]) for train, test in splits]
            X_here = _rows(X[self.columns], rows)
            sub_weight = None if weight is None else weight[rows]
            self.params_, sub_oof = select(self.make, self.grid, X_here, sub_y, groups[rows], sub_splits, sub_weight)
            answered = np.isfinite(sub_oof).all(1)
            oof[rows[answered]] = sub_oof[answered]
            self.model_ = fit_model(self.make, self.params_, X_here, sub_y, sub_weight)
        self.oof_logits_ = oof
        return self

    def predict_log_proba(self, X) -> np.ndarray:
        """(n, 3) log-probabilities: the model's where the modality was present, the training
        log-prior elsewhere."""
        here = self.present(X)
        out = np.tile(np.log(self.prior_), (len(X), 1))
        rows = np.flatnonzero(here)
        if len(rows):
            out[rows] = log_proba(self.model_, _rows(X[self.columns], rows))
        return out


class LateFusion:
    """the headline: one expert per modality block, and a multinomial logistic stacker (C fixed at
    1, no class weights) over their log-probabilities and presence flags, trained on the experts'
    inner out-of-fold answers so it learns how far to trust each one on windows it did not fit.
    Unweighted, its answers carry the training prior, so it is its own calibrator.

    `blocks` maps a modality to its columns (its block, group_size and its lags); `flags` maps a
    modality to its presence flag (a callable of the pooled frame, or a column name), PRESENCE by
    default. `extra` (fit and predict) adds outside log-probabilities as further experts, e.g.
    Jev's: a dict name -> (n, 3) array, NaN where there is none."""

    def __init__(self, make, grid, blocks: dict, flags: dict | None = None, inner=4, C: float = 1.0):
        self.make, self.grid, self.blocks, self.inner, self.C = make, list(grid), dict(blocks), inner, C
        self.flags = dict(PRESENCE, **(flags or {}))

    def _stack_inputs(self, logits: list[np.ndarray], present: list[np.ndarray]) -> np.ndarray:
        return np.column_stack(logits + [p.astype(float) for p in present])

    def _extra(self, extra: dict | None, n: int, prior: np.ndarray) -> tuple[list, list]:
        if sorted(extra or {}) != self.extra_:
            raise ValueError(f"the stacker was fit with the outside experts {self.extra_}, not {sorted(extra or {})}")
        logits, present = [], []
        for name in self.extra_:
            values = np.array(extra[name], dtype=float).reshape(n, N_CLASSES)
            here = np.isfinite(values).all(1)
            values[~here] = np.log(prior)
            logits.append(values)
            present.append(here)
        return logits, present

    def fit(self, X, y, groups, extra: dict | None = None, sample_weight=None):
        """fits the experts on the inner splits, the stacker on their out-of-fold answers, and
        keeps the stacker's own cross-fitted answers in `oof_logits_`. `sample_weight` (the
        session-weight ablation) weighs the experts' and the stacker's fits."""
        y, groups = _labels(y), np.asarray(groups)
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        splits = inner_splits(y, groups, self.inner)
        self.prior_ = class_prior(y)
        self.experts_ = {}
        logits, present = [], []
        for modality, columns in self.blocks.items():
            expert = Expert(self.make, self.grid, columns, self.flags.get(modality), modality)
            expert.fit(X, y, groups, splits, weight)
            self.experts_[modality] = expert
            logits.append(expert.oof_logits_)
            present.append(expert.present(X))
        self.extra_ = sorted(extra or {})
        if extra:
            # an outside expert's fallback in the out-of-fold inputs is the fold's prior, as for ours
            fallback = np.full((len(y), N_CLASSES), np.nan)
            for train, test in splits:
                fallback[test] = np.log(class_prior(y[train]))
            for name in sorted(extra):
                values = np.array(extra[name], dtype=float).reshape(len(y), N_CLASSES)
                here = np.isfinite(values).all(1)
                values[~here] = fallback[~here]
                logits.append(values)
                present.append(here)
        Z = self._stack_inputs(logits, present)
        answered = np.isfinite(Z).all(1)
        # the stacker's own out-of-fold answers (the same splits), for choosing the HMM's gamma
        oof = np.full((len(y), N_CLASSES), np.nan)
        for train, test in splits:
            train, test = train[answered[train]], test[answered[test]]
            if len(test):
                stacker = fit_model(self._make_stacker, None, Z[train], y[train], None if weight is None else weight[train])
                oof[test] = log_proba(stacker, Z[test])
        self.oof_logits_ = oof
        rows = np.flatnonzero(answered)
        self.stacker_ = fit_model(self._make_stacker, None, Z[rows], y[rows], None if weight is None else weight[rows])
        return self

    def _make_stacker(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=self.C, max_iter=2000)

    def predict_log_proba(self, X, extra: dict | None = None) -> np.ndarray:
        logits = [expert.predict_log_proba(X) for expert in self.experts_.values()]
        present = [expert.present(X) for expert in self.experts_.values()]
        more, here = self._extra(extra, len(X), self.prior_)
        return log_proba(self.stacker_, self._stack_inputs(logits + more, present + here))

    def predict_proba(self, X, extra: dict | None = None) -> np.ndarray:
        return np.exp(self.predict_log_proba(X, extra))


def lr_coefficients(model, columns, blocks: dict | None = None):
    """the fitted logistic regression's weights, one row per input column with its modality block
    (from `blocks`, modality -> columns) and one weight per class: the per-block coefficient report
    of the LR rows. The columns are the standardised ones the model saw."""
    import pandas as pd
    coef = np.asarray((model.steps[-1][1] if hasattr(model, 'steps') else model).coef_)
    block_of = {column: modality for modality, names in (blocks or {}).items() for column in names}
    names = ('individual', 'social', 'collaborative') if coef.shape[0] == 3 else tuple(range(coef.shape[0]))
    table = pd.DataFrame(coef.T, columns=[f'coef_{name}' for name in names])
    table.insert(0, 'block', [block_of.get(column) for column in columns])
    table.insert(0, 'column', list(columns))
    return table


# ---- floors ----

def majority_floor(y_train, n: int) -> tuple[np.ndarray, np.ndarray]:
    """the majority floor for n windows: (posteriors, labels), the training prior as every window's
    posterior and the commonest training class as its label."""
    prior = class_prior(y_train)
    return np.tile(prior, (n, 1)), np.full(n, int(prior.argmax()))


def stratified_floor(y_train, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """the prior-stratified random floor for n windows: (posteriors, labels), labels drawn from the
    training prior with this seed; its metrics are averaged over seeds."""
    prior = class_prior(y_train)
    return np.tile(prior, (n, 1)), np.random.default_rng(seed).choice(len(prior), n, p=prior)


# ---- calibration and the decision ----

class Calibrator:
    """z' = z / T + b with b_individual = 0: a temperature for over- or under-confidence, and a
    per-class bias that gives back the prior the balanced class weights took away, so the HMM's
    p / pi and the balanced decision both see the true prior. It is fit by the unweighted NLL of
    the inner out-of-fold logits (scipy L-BFGS-B); CalibratedClassifierCV is not used, since its
    internal splits are random."""

    def __init__(self):
        self.temperature, self.bias = 1.0, None

    def fit(self, logits, y):
        z = np.asarray(logits, dtype=float)
        k = z.shape[1]
        labels = _labels(y, k)
        keep = (labels >= 0) & np.isfinite(z).all(1)
        z, labels = z[keep], labels[keep]
        self.temperature, self.bias = 1.0, np.zeros(k)
        if not len(z):
            return self
        onehot = np.eye(k)[labels]

        def loss(theta):
            t, b = np.exp(theta[0]), np.concatenate([[0.0], theta[1:]])
            s = z / t + b
            log_p = s - _logsumexp(s)
            d = (np.exp(log_p) - onehot) / len(z)
            return float(-(log_p * onehot).sum() / len(z)), np.concatenate([[(d * (-z / t)).sum()], d.sum(0)[1:]])

        bounds = [(np.log(0.05), np.log(20.0))] + [(-10.0, 10.0)] * (k - 1)
        result = minimize(loss, np.zeros(k), jac=True, method='L-BFGS-B', bounds=bounds)
        self.temperature = float(np.exp(result.x[0]))
        self.bias = np.concatenate([[0.0], result.x[1:]])
        return self

    def transform(self, logits, log: bool = False) -> np.ndarray:
        """the calibrated probabilities (log-probabilities with log=True); a row with no logits
        stays NaN."""
        z = np.asarray(logits, dtype=float)
        bias = self.bias if self.bias is not None else np.zeros(z.shape[1])
        s = z / self.temperature + bias
        out = np.full(z.shape, np.nan)
        keep = np.isfinite(s).all(1)
        if keep.any():
            out[keep] = s[keep] - _logsumexp(s[keep])
        return out if log else np.exp(out)


def balanced_decision(p, prior) -> np.ndarray:
    """the pre-declared hard label, never tuned: argmax_k p(k|x) / pi_k with pi the outer-training
    class frequencies, so a class wins when the window makes it likelier than it is a priori.
    Given p_interaction (1-D) and pi_interaction (a number, or the three-class prior, whose social
    and collaborative shares add up to it), the binary label instead: 1 when
    p / pi >= (1 - p) / (1 - pi). A row with no posterior gets -1."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        prior = np.asarray(prior, dtype=float).reshape(-1)
        pi = float(prior[0]) if len(prior) == 1 else float(prior[1:].sum())
        out = np.where(p / pi >= (1.0 - p) / (1.0 - pi), 1, 0)
        return np.where(np.isfinite(p), out, -1).astype(int)
    ratio = p / np.asarray(prior, dtype=float)
    keep = np.isfinite(ratio).all(1)
    out = np.full(len(p), -1, dtype=int)
    out[keep] = ratio[keep].argmax(1)
    return out
