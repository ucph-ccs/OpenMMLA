"""The evaluation driver of the 10 s interaction classifier (Layer A): it reads every session's
fused table and coded labels, holds out one date at a time (or, once, the four TEST sessions),
fits everything a model needs on the training side only, and writes the run folder
artifacts/_analysis/interaction/<run>/ the results table is made from.

What is fit where; nothing ever reads the held-out date's labels:
- the [g] scaling statistics: the outer-training sessions' windows (label-free); the [s] ones
  each session itself;
- hyperparameters (C, the HGB grid, the network's epoch count): the inner folds over the
  outer-training dates (splits.inner_folds), by the class-weighted NLL of their out-of-fold
  answers;
- the calibrator, the late-fusion stacker and the HMM's gamma: those same out-of-fold answers;
- the HMM's transitions and the prior: the outer-training labels;
- the final model: every outer-training session, applied once to the held-out date.

The unit of every split is splits.class_units: a date, with the dates a manifest's same_class_as
links reach, so the same group on the same date is never on both sides. Its sessions are held out,
grouped in the inner folds and resampled together; SessionData.lesson and the lesson column of
predictions.csv hold the unit, named by its lessons joined with '+'. A run where a DEV session
shares a date or a class with a TEST one is refused.

Every model ends in one format: calibrated p(individual), p(social), p(collaborative), the hard
label by the balanced decision argmax p / pi under the outer-training prior (pre-declared, never
tuned), and the binary target derived as p_social + p_collaborative. The a-priori rule gives hard
labels only, and zero-shot Jev its own probabilities (argmax, since it has no prior).

The online rows (HMM mode 'filter') read no later window of the held-out session anywhere: their
inputs are causal (T0, T1c, the network without its temporal blocks, Jev), the held-out session's
[s] values are scaled by the running normaliser (layout.scale(mode='causal')), and the forward
filter reads only the past. The model, calibrator and gamma stay those fitted on the complete
training sessions, which an online classifier would have had in advance.

A run refuses to train when a class has fewer than 30 coded windows in an outer-training fold:
with that few every learned number is noise, and a model trained anyway would sit in the results
table as if it were not. The label counts are written first, so the refusal says where labels
are missing.

A variant is scored only on the windows it answered, and results.csv and metrics.json give its
coverage (answered over coded held-out windows). Only Jev can fall short, where ses-jev did not
ask about a window or got no answer; a confirmatory contrast is computed only when both of its
variants answered every coded window, so a partial Jev never stands in for a whole one. Jev and
jev-cal are left out of a run, with the reason, when a session they would score (or jev-cal would
train on) has no map made from its current fused table.

Windows the coder called absent (fewer than two members at the group's place) train nothing and
score nothing, like unclear ones. The presence gate (presence.py), fixed before labels were read,
is scored against them as a task of its own. Every variant is also scored per observed-person
stratum and gaze readability, and the headline's decisions with the gate's absent windows give the
end-to-end state shares per lesson (state_shares.csv). A bootstrap interval or contrast needs at
least MIN_BOOTSTRAP_UNITS units; the TEST sessions fall on 2 dates, so a test run reports its
estimates without intervals.

Deferred by decision, with their places kept: the ablation grids of 4.6 (`--ablate`), the causal
network row, the lexicon feature, masked-modality pretraining, and the task-transfer run
(splits.task_transfer exists; the driver does not run it yet).
"""
from __future__ import annotations

import functools
import importlib.util
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from openmmla.analytics.interaction import hmm as H
from openmmla.analytics.interaction import labels as L
from openmmla.analytics.interaction import layout as LY
from openmmla.analytics.interaction import metrics as M
from openmmla.analytics.interaction import presence as P
from openmmla.analytics.interaction import splits as S
from openmmla.analytics.interaction import tabular as TB

# every model a run can name, with the group of the results table it sits in
GROUPS = {
    'r0': 'no labels', 'r0-v1': 'no labels', 'jev': 'no labels',
    'majority': 'label floors', 'stratified': 'label floors',
    'r1': 'few-label', 'jev-cal': 'few-label',
    'lr': 'tabular', 'hgb': 'tabular', 'late-lr': 'tabular', 'late-hgb': 'tabular',
    'pooled-net': 'neural', 'net-notcn': 'neural', 'net': 'neural', 'net-pair': 'neural',
}
MODELS = tuple(GROUPS)
TABULAR = ('r1', 'lr', 'hgb', 'late-lr', 'late-hgb')
NETWORKS = ('pooled-net', 'net-notcn', 'net', 'net-pair')
# the models that read no label: they run even where a learned model is refused
UNLEARNED = ('r0', 'r0-v1', 'jev')
# the a-priori rule's versions by model name: r0 is the current rule, r0-v1 the first one, kept for the record
RULE_MODELS = {'r0': TB.RULE_VERSION, 'r0-v1': 1}
HEADLINE_MODELS = ('late-lr', 'late-hgb')
TEMPORAL = ('T0', 'T1c', 'T2')
HMM_MODES = ('none', 'fb', 'filter')
# 'date' leaves one date out of DEV (the unit of every split); 'test' scores TEST once
SPLITS = ('date', 'test')
TARGETS = ('3class', 'binary')
# the network reads its sequence through its own temporal blocks, not through lag columns
NET_TEMPORAL = {'pooled-net': 'tcn', 'net-notcn': 'T0', 'net': 'tcn', 'net-pair': 'tcn'}
# inputs whose features never read a later window: only these get the forward filter, the online
# answer, and for it the held-out session is scaled by the running normaliser
CAUSAL_INPUTS = ('T0', 'T1c', 'j0', 'j1')
JEV_MODELS = ('jev', 'jev-cal')

MIN_CLASS_WINDOWS = 30
INNER_FOLDS = 4
BOOTSTRAP = 2000
# fewer units than this give no bootstrap interval or contrast: a resample of 2 dates is one of
# them, the other or both, so its percentiles are those dates' own scores, not an interval
MIN_BOOTSTRAP_UNITS = 5
STRATIFIED_SEEDS = (0, 1, 2, 3, 4)
# the pre-registered absent-class policy (4.5): below either bound the confirmatory target is the
# binary one and the three-class results are exploratory
POLICY_SOCIAL_WINDOWS = 200
POLICY_SOCIAL_LESSONS = 6
POLICY_SOCIAL_PER_LESSON = 5
# --quick: small grids and short training, to check the plumbing end to end; never a reported run
QUICK = {
    'lr_grid': [{'C': 0.1}, {'C': 1.0}],
    'hgb_grid': [{'max_iter': 100, 'max_leaf_nodes': 8, 'min_samples_leaf': 40},
                 {'max_iter': 200, 'max_leaf_nodes': 16, 'min_samples_leaf': 100}],
    'max_epochs': 30, 'patience': 5, 'bootstrap': 200,
}
PREDICTION_COLUMNS = ('session', 'lesson', 'task', 'window_index', 'window_start', 'fold', 'model', 'variant',
                      'coded', 'empty_window', 'y_true', 'p_individual', 'p_social', 'p_collaborative',
                      'p_interaction', 'y_pred', 'y_pred_binary', 'temporal', 'hmm', 'y_viterbi',
                      'n_observed', 'presence_gated', 'gaze_readable')


class Refused(RuntimeError):
    """the run would not train: `problems` says which outer-training folds lack which class (or
    lack two dates to hold out), and `run_dir` holds the label counts it read."""

    def __init__(self, message: str, problems: list, run_dir: Path):
        super().__init__(message)
        self.problems, self.run_dir = problems, run_dir


@dataclass
class Config:
    """what a run does (mmla ses-classify fills it from its flags). `models`, `temporal` and `hmm`
    are lists; `coder` is the truth (default: the coder with the most windows over the DEV
    sessions; a test run must name it); `quick` swaps in the QUICK grids and training lengths;
    `epochs` fixes the network's E* (the median of the date folds' for the test model) instead of
    the inner choice; `bootstrap` overrides the number of unit resamples."""
    artifacts: str = 'artifacts'
    sessions: str | None = None
    coder: str | None = None
    models: tuple = ('r0', 'r1', 'lr', 'hgb', 'late-lr', 'late-hgb')
    split: str = 'date'
    temporal: tuple = TEMPORAL
    hmm: tuple = HMM_MODES
    target: str = '3class'
    join: str = 'exact'
    seeds: int = 5
    small: str = 'auto'
    epochs: int | None = None
    jobs: int = 1
    out: str | None = None
    quick: bool = False
    confirm_frozen: bool = False
    jev_variant: str = 'j0'
    bootstrap: int | None = None
    min_class_windows: int = MIN_CLASS_WINDOWS


# ---- sessions ----

@dataclass(eq=False)
class SessionData:
    """one session as the driver holds it: the roster, the unscaled tokens and the unscaled pooled
    view (the rule's input), the coded labels over the table's grid (3-class: 0, 1, 2, -1
    unclear, -2 absent, NaN uncoded) and the target the models learn (the same, or its binary
    form), the join report, the other coders' labels (for agreement), the cached Jev answers and,
    per window, the kept persons observed, whether the presence gate calls it absent, and whether
    every kept person's gaze was readable."""
    session: str
    lesson: str
    task: str | None
    directory: Path
    table_path: Path
    table_sha256: str | None
    roster: LY.Roster
    tokens: LY.Tokens
    raw: pd.DataFrame
    y: np.ndarray
    target: np.ndarray
    join: dict
    label_files: dict = field(default_factory=dict)
    others: dict = field(default_factory=dict)
    jev: np.ndarray | None = None
    jev_note: str | None = None
    n_observed: np.ndarray | None = None
    presence_gated: np.ndarray | None = None
    gaze_readable: np.ndarray | None = None
    same_class_as: tuple = ()
    rules_roster_sha256: str | None = None

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def n_coded(self) -> int:
        return int(L.scored(self.target).sum())

    def series(self, values=None) -> pd.Series:
        """labels over the session's window_index (what the HMM's transitions count pairs by)."""
        return pd.Series(self.target if values is None else values, index=self.tokens.window_index)

    @property
    def flat(self) -> np.ndarray:
        """the windows no modality ran in: the HMM gives them a flat emission."""
        return self.tokens.avail.sum(axis=1) == 0

    @property
    def blocks(self) -> np.ndarray:
        """a name per window for the coded stretch it belongs to (ses-code samples 5-minute blocks),
        '' outside one: switches and run lengths are counted within a stretch only."""
        labelled = np.isfinite(self.y)
        starts = labelled & ~np.concatenate([[False], labelled[:-1]])
        number = np.cumsum(starts)
        return np.where(labelled, np.char.add(f'{self.session}#', number.astype(str)), '')


def session_tables(artifacts, pattern: str | None = None) -> list[tuple[str, Path]]:
    """(session, fused table) for every artifacts/exp_*/analysis/features/<session>_window_features.csv
    whose session id contains `pattern`."""
    found = []
    for session_dir in sorted(Path(artifacts).glob('exp_*')):
        if pattern and pattern not in session_dir.name:
            continue
        path = session_dir / 'analysis' / 'features' / f'{session_dir.name}_window_features.csv'
        if path.exists():
            found.append((session_dir.name, path))
    return found


def primary_coder(directories) -> str | None:
    """the coder with the most windows over every given session, the truth of the whole run (a
    per-session majority could make the truth one coder here and another there). run() passes the
    DEV sessions only, in every split, so how many TEST windows someone coded never decides whose
    labels the models learn from."""
    frames = [L.load_labels(directory, all_coders=True) for directory in directories]
    frames = [frame for frame in frames if len(frame)]
    return L.primary_coder(pd.concat(frames, ignore_index=True)) if frames else None


def same_class_of(directory) -> tuple:
    """the session ids a session's manifest marks as the same school class (`same_class_as`,
    written by mmla ses-tidy --same-class-as); () when it names none or has no manifest."""
    try:
        linked = json.loads((Path(directory) / 'manifest.json').read_text(encoding='utf-8')).get('same_class_as')
    except (OSError, ValueError, AttributeError):
        return ()
    if isinstance(linked, str):
        linked = [linked]
    return tuple(str(s) for s in linked or () if s)


def link_lessons(data: dict) -> None:
    """every session's lesson widened to its unit (splits.class_units over the loaded sessions:
    its date, with the dates their same_class_as reach), so predictions, the bootstrap and the
    refusal group what the folds group."""
    units = S.class_units(list(data), {s: d.same_class_as for s, d in data.items()})
    for session, d in data.items():
        d.lesson = units[session]


def load_session(session: str, table_path, coder: str | None = None, join: str = 'exact',
                 target: str = '3class') -> tuple[SessionData, pd.DataFrame]:
    """a session and its fused table: roster (the manifest's pupils when it declares them),
    unscaled tokens and pooled view, and the coder's labels joined to the table's grid (a join
    that leaves more than 1 % of labels without a window raises labels.LabelJoinError, which
    aborts the run). The other coders' labels are joined too,
    for the inter-coder kappa; one of theirs that does not join is left out, not fatal."""
    from openmmla.utils.session_provenance import file_digest
    table_path = Path(table_path)
    table = LY.read_table(table_path)
    directory = table_path.parents[2]
    # the pupils the session's manifest declares, else the roster rules
    ros = LY.session_roster(table, directory)
    tokens = LY.window_tokens(table, ros)
    truth = L.load_labels(directory, coder=coder)
    if len(truth):
        y, report = L.join_labels(table, truth, mode=join)
        y = y.to_numpy(dtype=float)
    else:
        y, report = np.full(len(table), np.nan), {'mode': join, 'labels': 0}
    report.update(coder=truth.attrs.get('coder'), skipped_lines=truth.attrs.get('skipped_lines', 0))
    others = {}
    everyone = L.load_labels(directory, all_coders=True)
    coders = sorted(set(everyone['coder']) - {L.ADJUDICATED}) if len(everyone) else []
    for name in coders:
        try:
            joined, _ = L.join_labels(table, everyone[everyone['coder'] == name], mode=join)
        except L.LabelJoinError:
            continue
        others[name] = joined.to_numpy(dtype=float)
    label_files = {path.name: file_digest(path) for path in sorted((directory / 'labels').glob('*.jsonl'))} \
        if (directory / 'labels').is_dir() else {}
    n_observed = P.observed_count(table, ros)
    data = SessionData(session=session, lesson=S.lesson_key(session), task=S.task_of(session), directory=directory,
                       table_path=table_path, table_sha256=file_digest(table_path), roster=ros, tokens=tokens,
                       raw=LY.pooled(tokens), y=y, target=L.to_binary(y) if target == 'binary' else y.copy(),
                       join=report, label_files=label_files, others=others, n_observed=n_observed,
                       presence_gated=n_observed < P.MIN_OBSERVED, gaze_readable=P.gaze_readable(tokens),
                       same_class_as=same_class_of(directory),
                       rules_roster_sha256=LY.rules_roster_digest(table, ros))
    return data, table


def jev_log_proba(data: SessionData, variant: str = 'j0') -> tuple[np.ndarray | None, str | None]:
    """the session's cached Jev answers (mmla ses-jev's artifacts/<session>/analysis/interaction/
    jev_<variant>.jsonl) as (T, 3) log-probabilities over the three classes (an unclear share, J2,
    is left out and the rest renormalised), NaN where a window has none; None and why when there
    is no map, or it was made from another version of the fused table or asked about another
    roster (the session has since declared its pupils): its states would differ."""
    path = data.directory / 'analysis' / 'interaction' / f'jev_{variant}.jsonl'
    if not path.exists():
        return None, f'no {path.name}: run mmla ses-jev --variant {variant} first'
    position = {int(w): t for t, w in enumerate(data.tokens.window_index)}
    out = np.full((len(data), len(L.CLASSES)), np.nan)
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            record = json.loads(line)
            t = position.get(int(record['window_index']))
            p = np.array([record.get(f'p_{name}') for name in L.CLASSES], dtype=float)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue  # a blank or cut line
        digest = record.get('table_sha256')
        if digest and data.table_sha256 and digest != data.table_sha256:
            return None, f'{path.name} was made from another version of the fused table: rerun mmla ses-jev'
        # a line without a roster digest was asked about the rules roster (see LY.rules_roster_digest)
        asked = record.get('roster_sha256') or data.rules_roster_sha256
        if asked and asked != LY.roster_digest(data.roster.kept, data.roster.group_size):
            return None, f'{path.name} was asked about another roster than the session now has: rerun mmla ses-jev'
        if t is None or not np.isfinite(p).all() or p.sum() <= 0:
            continue
        p = np.maximum(p / p.sum(), 1e-4)
        out[t] = np.log(p / p.sum())
    return out, None


def jev_template(data: SessionData, variant: str = 'j0') -> str | None:
    """the template digest (bins_sha256) the session's Jev map was written with; None for a map
    from before ses-jev recorded it, or no map."""
    path = data.directory / 'analysis' / 'interaction' / f'jev_{variant}.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines() if path.exists() else []:
        try:
            return json.loads(line).get('bins_sha256')
        except (json.JSONDecodeError, AttributeError):
            continue
    return None


def jev_gaps(folds, data: dict, models, variant: str = 'j0') -> dict:
    """the Jev models a run has to leave out, each with why: zero-shot Jev needs a usable map
    (jev_log_proba's: made from the session's current fused table, asked about its current roster)
    for every session with coded windows it is scored on, and jev-cal also for every one it is
    trained on, all written with one frozen template. Scored anyway, each coded window of a session without one would count as a
    miss and C3 would set the headline against nothing."""
    scored = {s for fold in folds for s in fold.test if data[s].n_coded}
    trained = {s for fold in folds for s in fold.train if data[s].n_coded}
    out = {}
    for model in models:
        if model not in JEV_MODELS:
            continue
        needed = sorted(scored | trained if model == 'jev-cal' else scored)
        missing = [s for s in needed if data[s].jev is None]
        templates = {jev_template(data[s], variant) for s in needed if data[s].jev is not None}
        if missing:
            shown = ', '.join(missing[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3 else '')
            out[model] = (f"no usable jev_{variant}.jsonl for {len(missing)} session(s) with coded windows ({shown}); "
                          f"run mmla ses-jev --variant {variant} on the current fused tables and rosters")
        elif len(templates) > 1:
            out[model] = (f"the jev_{variant}.jsonl maps were written with {len(templates)} different templates "
                          f"(bins_sha256); rerun mmla ses-jev --variant {variant} on every session with one "
                          f"frozen --bins")
    return out


# ---- folds and the refusal ----

def class_names(k: int) -> tuple:
    return L.CLASSES if k == 3 else ('individual', 'interaction')


def make_folds(split: str, data: dict) -> list:
    """the outer folds: one per DEV date with coded windows (date), dates joined by same_class_as
    together, or the one scoring of the TEST sessions (test). Raises splits.SplitError when a DEV
    session shares a date or a same_class_as link with a TEST one."""
    sessions = list(data)
    links = {s: d.same_class_as for s, d in data.items()}
    if split == 'date':
        return S.date_folds(sessions, coded={s: d.n_coded for s, d in data.items()}, same_class=links)
    fold = S.final_fold(sessions, same_class=links)
    return [fold] if fold.test else []


def label_refusal(folds, data: dict, k: int, minimum: int = MIN_CLASS_WINDOWS) -> list:
    """the outer-training folds a learned model may not be fitted on: a class with fewer than
    `minimum` coded windows, or fewer than two units (dates) with coded windows for the inner folds."""
    names = class_names(k)
    problems = []
    for fold in folds:
        counts = np.zeros(k, dtype=int)
        units = set()
        for session in fold.train:
            target = data[session].target
            coded = L.scored(target)
            counts += np.bincount(target[coded].astype(int), minlength=k)[:k]
            if coded.any():
                units.add(data[session].lesson)
        short = [names[c] for c in range(k) if counts[c] < minimum]
        if short or len(units) < 2:
            problems.append({'fold': fold.name, 'counts': dict(zip(names, counts.tolist())), 'short': short,
                             'coded_units': len(units)})
    return problems


def label_counts(data: dict) -> pd.DataFrame:
    """windows per class and session (and unclear, absent, uncoded), with the unit (lesson column),
    task and split role: the table printed before training."""
    counts = L.label_counts({s: d.y for s, d in data.items()})
    counts.insert(0, 'role', ['test' if s in S.TEST_SESSIONS else 'dev' for s in counts.index])
    counts.insert(0, 'task', [data[s].task for s in counts.index])
    counts.insert(0, 'lesson', [data[s].lesson for s in counts.index])
    counts.index.name = 'session'
    return counts


def absent_class_policy(data: dict) -> dict:
    """the pre-registered policy: with fewer than 200 coded social windows on dev, or social at
    least 5 times in fewer than 6 lessons, the confirmatory target becomes the binary one. The
    lessons are counted as registered, by splits.lesson_key, not by the date units the folds hold
    out, so neither the date unit nor a same_class_as link changes the target."""
    social, by_lesson = 0, {}
    for d in data.values():
        if d.session in S.TEST_SESSIONS:
            continue
        n = int((d.y == 1).sum())
        social += n
        # d.lesson is the date unit after link_lessons; the policy counts the lesson itself
        lesson = S.lesson_key(d.session)
        by_lesson[lesson] = by_lesson.get(lesson, 0) + n
    lessons = sum(1 for n in by_lesson.values() if n >= POLICY_SOCIAL_PER_LESSON)
    binary = social < POLICY_SOCIAL_WINDOWS or lessons < POLICY_SOCIAL_LESSONS
    return {'social_windows': social, 'lessons_with_social': lessons,
            'confirmatory_target': 'binary' if binary else '3class'}


# ---- one outer fold ----

def _offsets(sessions) -> dict:
    out, at = {}, 0
    for d in sessions:
        out[d.session] = slice(at, at + len(d))
        at += len(d)
    return out


class _Fold:
    """one outer fold, scaled with the statistics of its training side: the pooled views and their
    lags, the rows of the training side with their units, the inner folds (as row positions for
    the tabular models, as session names for the network), the training prior and transitions."""

    def __init__(self, fold, data: dict, k: int, learned: bool = True):
        self.fold, self.k = fold, k
        self.train = [data[s] for s in fold.train]
        self.test = [data[s] for s in fold.test]
        self.stats = LY.fit_global_stats([d.tokens for d in self.train])
        self.scaled = {d.session: LY.scale(d.tokens, self.stats) for d in self.train + self.test}
        self.views = {s: LY.pooled(t) for s, t in self.scaled.items()}
        self._online = None
        self.at = {'train': _offsets(self.train), 'test': _offsets(self.test)}
        self.y = np.concatenate([d.target for d in self.train])
        sessions = np.concatenate([np.full(len(d), d.session, dtype=object) for d in self.train])
        # a date (with its same_class_as dates) is one group, as in the outer folds
        links = {d.session: d.same_class_as for d in self.train}
        self.units = S.class_units(fold.train, links)
        self.groups = np.concatenate([np.full(len(d), self.units[d.session], dtype=object) for d in self.train])
        # the rule and zero-shot Jev choose nothing, so they need no inner folds (nor two coded dates)
        self.inner = S.inner_folds(fold.train, k=INNER_FOLDS, sizes={d.session: d.n_coded for d in self.train},
                                   same_class=links) if learned else []
        self.pairs = [S.fold_indices(inner, sessions) for inner in self.inner]
        self.prior = TB.class_prior(self.y, k)
        self.A = H.transitions({d.session: d.series() for d in self.train}, n_states=k)
        self._X = {}

    def side(self, side: str) -> list:
        return self.train if side == 'train' else self.test

    def X(self, side: str, temporal: str) -> pd.DataFrame:
        """the pooled view with its lags, each session's computed over its own whole grid."""
        if (side, temporal) not in self._X:
            self._X[side, temporal] = pd.concat([LY.temporal_context(self.views[d.session], temporal)
                                                 for d in self.side(side)], ignore_index=True)
        return self._X[side, temporal]

    def online(self) -> dict:
        """the held-out sessions' tokens as an online classifier would have them at each window: a
        [s] value scaled by the running median and spread of the session's windows so far (the
        [g] statistics until 30 values are seen), never by the windows after it."""
        if self._online is None:
            self._online = {d.session: LY.scale(d.tokens, self.stats, mode='causal') for d in self.test}
        return self._online

    def X_online(self, temporal: str) -> pd.DataFrame:
        """the held-out pooled view with its lags from the online tokens (causal lags only)."""
        if ('online', temporal) not in self._X:
            self._X['online', temporal] = pd.concat(
                [LY.temporal_context(LY.pooled(self.online()[d.session]), temporal) for d in self.test],
                ignore_index=True)
        return self._X['online', temporal]

    def by_session(self, values, side: str) -> dict:
        return {session: values[at] for session, at in self.at[side].items()}

    def flat(self, side: str) -> dict:
        return {d.session: d.flat for d in self.side(side)}

    def labels(self, binary: bool = False) -> dict:
        return {d.session: (L.to_binary(d.target) if binary else d.target) for d in self.train}

    def smooth(self, p, prior, A, gamma, mode) -> np.ndarray:
        """the held-out sessions' posteriors through the HMM, each session on its own."""
        flats = self.flat('test')
        return np.vstack([H.smooth(part, prior, A, gamma, flats[session], mode)
                          for session, part in self.by_session(p, 'test').items()])


def _key(model: str, temporal: str, mode: str) -> str:
    return f'{model}:{temporal}:{mode}'


def _collapse(logits, k: int):
    """three-class logits as the target's: for the binary target, individual against the
    log-sum of the two interaction classes (the models learn from 0/1 labels, so the third holds
    only the floor)."""
    if logits is None or k == 3:
        return logits
    logits = np.asarray(logits, dtype=float)
    return np.column_stack([logits[:, 0], np.logaddexp(logits[:, 1], logits[:, 2])])


def _normalised(logits) -> np.ndarray:
    """log-probabilities as probabilities, renormalised; a row with no answer stays NaN."""
    z = np.asarray(logits, dtype=float)
    out = np.full(z.shape, np.nan)
    keep = np.isfinite(z).all(axis=1)
    if keep.any():
        top = z[keep].max(axis=1, keepdims=True)
        e = np.exp(z[keep] - top)
        out[keep] = e / e.sum(axis=1, keepdims=True)
    return out


def _decide(p, prior, k: int) -> tuple[np.ndarray, np.ndarray]:
    """the balanced decision on the posteriors, and the binary one on p_social + p_collaborative
    under pi_social + pi_collaborative; -1 where there is no posterior."""
    y_pred = TB.balanced_decision(p, prior)
    if k == 2:
        return y_pred, y_pred.copy()
    return y_pred, TB.balanced_decision(p[:, 1] + p[:, 2], prior)


def _finish(fd: _Fold, hmm_modes, model: str, temporal: str, oof, logits, calibrate: bool, details: dict,
            results: dict, online=None):
    """one model's answers through calibration (fit on the inner out-of-fold logits), each HMM mode
    (gamma chosen on the same out-of-fold answers) and the decisions; one result per mode.
    `online` is the held-out logits from the online tokens (_Fold.online), which the forward
    filter reads; without them (Jev, whose answers do not depend on scaling) it reads `logits`."""
    k = fd.k
    oof, logits = _collapse(oof, k), _collapse(logits, k)
    online = logits if online is None else _collapse(online, k)
    if calibrate:
        calibrator = TB.Calibrator().fit(oof, fd.y)
        p_oof, p_test, p_online = calibrator.transform(oof), calibrator.transform(logits), calibrator.transform(online)
        details['calibrator'] = {'temperature': calibrator.temperature, 'bias': calibrator.bias.tolist()}
    else:
        p_oof, p_test, p_online = _normalised(oof), _normalised(logits), _normalised(online)
    for mode in hmm_modes:
        if mode == 'filter' and temporal not in CAUSAL_INPUTS:
            continue
        record = {'p': p_test, 'viterbi': None, 'binary_hmm': None}
        if mode != 'none':
            gamma, scores = H.select_gamma(fd.by_session(p_oof, 'train'), fd.labels(), fd.prior, fd.A,
                                           fd.flat('train'), mode=mode)
            details.setdefault('gamma', {})[mode] = {'gamma': gamma, 'nll': scores}
            record['p'] = fd.smooth(p_online if mode == 'filter' else p_test, fd.prior, fd.A, gamma, mode)
            if mode == 'fb':
                record['viterbi'] = fd.smooth(p_test, fd.prior, fd.A, gamma, 'viterbi').argmax(axis=1)
                if k == 3:
                    record['binary_hmm'] = _binary_hmm(fd, p_oof, p_test)
        record['y_pred'], record['y_pred_binary'] = _decide(record['p'], fd.prior, k)
        results[_key(model, temporal, mode)] = record


def _binary_hmm(fd: _Fold, p_oof, p_test) -> np.ndarray:
    """the check of 2.8: the derived binary posteriors through a 2-state HMM of their own (binary
    transitions, gamma chosen the same way), decided by the binary balanced rule."""
    two_oof = np.column_stack([p_oof[:, 0], p_oof[:, 1] + p_oof[:, 2]])
    two_test = np.column_stack([p_test[:, 0], p_test[:, 1] + p_test[:, 2]])
    prior = np.array([fd.prior[0], 1.0 - fd.prior[0]])
    A = H.transitions({d.session: d.series(L.to_binary(d.target)) for d in fd.train}, n_states=2)
    gamma, _ = H.select_gamma(fd.by_session(two_oof, 'train'), fd.labels(binary=True), prior, A, fd.flat('train'))
    return TB.balanced_decision(fd.smooth(two_test, prior, A, gamma, 'fb'), prior)


def _online_wanted(plan: dict, temporal: str) -> bool:
    return 'filter' in plan['hmm'] and temporal in CAUSAL_INPUTS


def _tabular(fd: _Fold, plan: dict, model: str, temporal: str, details: dict, extras: dict):
    """(inner out-of-fold logits, held-out logits, whether to calibrate, held-out logits from the
    online tokens or None) of a tabular model."""
    X_train, X_test = fd.X('train', temporal), fd.X('test', temporal)
    X_online = fd.X_online(temporal) if _online_wanted(plan, temporal) else None
    if model in ('late-lr', 'late-hgb'):
        make, grid = (TB.make_lr, plan['lr_grid']) if model == 'late-lr' else (TB.make_hgb, plan['hgb_grid'])
        fusion = TB.LateFusion(make, grid, LY.block_columns(X_train), inner=fd.pairs).fit(X_train, fd.y, fd.groups)
        details['params'] = {modality: expert.params_ for modality, expert in fusion.experts_.items()}
        stacker = getattr(fusion.stacker_, 'coef_', None)
        if stacker is not None:
            details['stacker'] = np.asarray(stacker).tolist()
        # the stacker is unweighted, so it carries the training prior: its own calibrator
        online = fusion.predict_log_proba(X_online) if X_online is not None else None
        return fusion.oof_logits_, fusion.predict_log_proba(X_test), False, online
    make, grid = {'r1': (TB.make_rule_tree, [{}]), 'lr': (TB.make_lr, plan['lr_grid']),
                  'hgb': (TB.make_hgb, plan['hgb_grid'])}[model]
    params, oof = TB.select(make, grid, X_train, fd.y, fd.groups, fd.pairs)
    fitted = TB.fit_model(make, params, X_train, fd.y)
    details['params'] = params
    if model == 'lr' and hasattr(fitted, 'steps'):
        blocks = LY.block_columns(X_train, with_group_size=False)
        coefficients = TB.lr_coefficients(fitted, list(X_train.columns), blocks)
        coefficients.insert(0, 'temporal', temporal)
        coefficients.insert(0, 'fold', fd.fold.name)
        extras.setdefault('coefficients', []).append(coefficients)
    online = TB.log_proba(fitted, X_online) if X_online is not None else None
    return oof, TB.log_proba(fitted, X_test), True, online


def _network(fd: _Fold, plan: dict, model: str, details: dict):
    """(inner out-of-fold logits, held-out logits, held-out logits from the online tokens or None)
    of a network rung: E* from the inner folds (seed 0, patience), then the seed ensemble on every
    outer-training session."""
    from openmmla.analytics.interaction import network as N
    pooled_input = model == 'pooled-net'
    first = fd.views[fd.train[0].session]
    blocks = LY.block_columns(first, with_group_size=False)

    def tensors(d, online=False):
        tokens = fd.online()[d.session] if online else fd.scaled[d.session]
        if pooled_input:
            return N.session_tensors(None, y=d.target, pooled=LY.pooled(tokens) if online else fd.views[d.session],
                                     blocks=blocks, session=d.session)
        return N.session_tensors(tokens, y=d.target, session=d.session)

    train, test = [tensors(d) for d in fd.train], [tensors(d) for d in fd.test]
    small = N.use_small(sum(d.n_coded for d in fd.train), plan['small'])
    make = functools.partial(N.make_model, model, small=small, d_in=first.shape[1])
    epochs, oof = N.select_epochs(train, [(inner.train, inner.test) for inner in fd.inner], make=make,
                                  max_epochs=plan['max_epochs'], patience=plan['patience'], seed=0)
    used = plan['epochs'] or epochs
    models = N.fit_ensemble(train, used, seeds=tuple(range(plan['seeds'])), make=make)
    details.update(epochs=epochs, epochs_used=used, small=small, parameters=N.count_parameters(models[0]))
    oof = np.vstack([o if o is not None else np.full((len(d), 3), np.nan) for o, d in zip(oof, fd.train)])
    online = np.vstack([N.predict_net(models, tensors(d, online=True)) for d in fd.test]) \
        if _online_wanted(plan, NET_TEMPORAL[model]) else None
    return oof, np.vstack([N.predict_net(models, t) for t in test]), online


def _jev_rows(sessions) -> np.ndarray:
    return np.vstack([d.jev if d.jev is not None else np.full((len(d), 3), np.nan) for d in sessions])


def _unanswered(record: dict, rows: np.ndarray):
    """no answer where Jev gave none, even where the HMM would carry its neighbours over: a
    window ses-jev never asked about is left out of Jev's scores (and counted in its coverage),
    not scored as the smoother's guess."""
    if not rows.any():
        return
    record['p'] = np.where(rows[:, None], np.nan, record['p'])
    for name in ('y_pred', 'y_pred_binary', 'viterbi', 'binary_hmm'):
        if record.get(name) is not None:
            record[name] = np.where(rows, -1, record[name])


def run_fold(fold, data: dict, plan: dict) -> dict:
    """everything one outer fold gives: per variant (model:temporal:hmm) the held-out rows'
    posteriors and decisions, and what was chosen on the way (parameters, calibrators, gammas,
    epochs)."""
    started = time.time()
    k = plan['k']
    fd = _Fold(fold, data, k, learned=any(model not in UNLEARNED for model in plan['models']))
    results, details, extras = {}, {}, {}
    n_test = sum(len(d) for d in fd.test)
    for model in plan['models']:
        if model in RULE_MODELS:
            # the rule reads the unscaled view: its thresholds are in the table's units
            labels = np.concatenate([TB.rule_a_priori(d.raw, version=RULE_MODELS[model]) for d in fd.test])
            binary = (labels > 0).astype(int)
            results[_key(model, 'T0', 'none')] = {'p': None, 'y_pred': labels if k == 3 else binary,
                                                  'y_pred_binary': binary}
        elif model in ('majority', 'stratified'):
            y3 = np.where(L.scored(fd.y), fd.y, -1)
            p, labels = TB.majority_floor(y3, n_test)
            seeds = [labels]
            if model == 'stratified':
                seeds = [TB.stratified_floor(y3, n_test, seed)[1] for seed in STRATIFIED_SEEDS]
            seeds = [np.minimum(s, k - 1) for s in seeds]
            p = _normalised(_collapse(np.log(p), k))
            binary = [(s > 0).astype(int) for s in seeds]
            results[_key(model, 'T0', 'none')] = {'p': p, 'y_pred': seeds[0], 'y_pred_binary': binary[0],
                                                  'seed_preds': seeds if model == 'stratified' else None}
        elif model in TABULAR:
            for temporal in (('T0',) if model == 'r1' else plan['temporal']):
                where = details.setdefault(f'{model}:{temporal}', {})
                oof, logits, calibrate, online = _tabular(fd, plan, model, temporal, where, extras)
                _finish(fd, plan['hmm'], model, temporal, oof, logits, calibrate, where, results, online)
        elif model in NETWORKS:
            temporal = NET_TEMPORAL[model]
            where = details.setdefault(f'{model}:{temporal}', {})
            oof, logits, online = _network(fd, plan, model, where)
            _finish(fd, plan['hmm'], model, temporal, oof, logits, True, where, results, online)
        elif model == 'jev':
            p = _normalised(_collapse(_jev_rows(fd.test), k))
            answered = np.isfinite(p).all(axis=1)
            y_pred = np.where(answered, np.nan_to_num(p).argmax(axis=1), -1)
            interaction = np.where(answered, (np.nan_to_num(p[:, 1:]).sum(axis=1) >= p[:, 0]).astype(int), -1)
            results[_key('jev', plan['jev_variant'], 'none')] = {'p': p, 'y_pred': y_pred, 'y_pred_binary': interaction}
        elif model == 'jev-cal':
            # Jev never saw a label, so its answers on the training windows are out-of-sample as they are
            where = details.setdefault(f"jev-cal:{plan['jev_variant']}", {})
            held = _jev_rows(fd.test)
            _finish(fd, plan['hmm'], 'jev-cal', plan['jev_variant'], _jev_rows(fd.train), held, True, where, results)
            for key in [key for key in results if key.startswith('jev-cal:')]:
                _unanswered(results[key], ~np.isfinite(held).all(axis=1))
    return {'name': fold.name, 'train': list(fold.train), 'test': list(fold.test),
            'inner': [sorted({fd.units[s] for s in inner.test}) for inner in fd.inner],
            'prior': fd.prior.tolist(), 'transitions': fd.A.tolist(), 'models': details, 'results': results,
            'coefficients': extras.get('coefficients'), 'seconds': round(time.time() - started, 1)}


def _fold_job(fold, data: dict, plan: dict) -> dict:
    """run_fold in a worker: sklearn's convergence notes silenced, and each worker's native thread
    pools kept to its share of the cores so parallel folds do not oversubscribe them."""
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        if plan['threads']:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=plan['threads']):
                return run_fold(fold, data, plan)
        return run_fold(fold, data, plan)


# ---- assembling and scoring ----

def _base_rows(outputs: list, data: dict) -> pd.DataFrame:
    """the held-out windows of every fold, in fold and session order: the rows every variant's
    answers line up with."""
    frames = []
    for out in outputs:
        for session in out['test']:
            d = data[session]
            n = len(d)
            frames.append(pd.DataFrame({
                'session': session, 'lesson': d.lesson, 'task': d.task, 'window_index': d.tokens.window_index,
                'window_start': d.tokens.window_start, 'fold': out['name'], 'coded': L.scored(d.target),
                'empty_window': d.tokens.empty, 'y_true': d.target, 'block': d.blocks,
                'n_observed': d.n_observed if d.n_observed is not None else np.full(n, -1),
                'presence_gated': d.presence_gated if d.presence_gated is not None else np.zeros(n, dtype=bool),
                'gaze_readable': d.gaze_readable if d.gaze_readable is not None else np.zeros(n, dtype=bool)}))
    return pd.concat(frames, ignore_index=True)


def _missing(n: int, k: int) -> dict:
    return {'p': np.full((n, k), np.nan), 'y_pred': np.full(n, -1), 'y_pred_binary': np.full(n, -1)}


def _variants(outputs: list, data: dict, k: int) -> dict:
    """per variant, its answers over the base rows: every fold's, concatenated (a fold that could
    not give one, e.g. a session without Jev answers, gives no answer rather than a gap)."""
    keys = []
    for out in outputs:
        keys += [key for key in out['results'] if key not in keys]
    variants = {}
    for key in keys:
        model, temporal, mode = key.split(':')
        parts = []
        for out in outputs:
            n = sum(len(data[s]) for s in out['test'])
            parts.append(out['results'].get(key) or _missing(n, k))
        record = {'model': model, 'temporal': temporal, 'hmm': mode,
                  'p': None if all(part['p'] is None for part in parts) else
                  np.vstack([part['p'] if part['p'] is not None else np.full((len(part['y_pred']), k), np.nan)
                             for part in parts]),
                  'y_pred': np.concatenate([part['y_pred'] for part in parts]).astype(int),
                  'y_pred_binary': np.concatenate([part['y_pred_binary'] for part in parts]).astype(int)}
        for name in ('viterbi', 'binary_hmm'):
            if all(part.get(name) is not None for part in parts):
                record[name] = np.concatenate([part[name] for part in parts]).astype(int)
        if all(part.get('seed_preds') is not None for part in parts):
            record['seed_preds'] = [np.concatenate([part['seed_preds'][i] for part in parts])
                                    for i in range(len(STRATIFIED_SEEDS))]
        variants[key] = record
    return variants


def _labels_int(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values >= 0), values, -1).astype(int)


def _binary_truth(y: np.ndarray) -> np.ndarray:
    return np.where(y >= 0, (y > 0).astype(int), -1)


def coverage(variant: dict, base: pd.DataFrame) -> dict:
    """how many of the coded held-out windows a variant answered: all of them for every model
    but Jev, which answers only the windows ses-jev asked about and got an answer for.
    `short_sessions` names the sessions with a coded window it did not answer."""
    coded = _labels_int(base['y_true']) >= 0
    answered = coded & (np.asarray(variant['y_pred']) >= 0)
    sessions = base['session'].to_numpy()
    short = sorted(set(sessions[coded & ~answered]))
    return {'answered': int(answered.sum()), 'coded': int(coded.sum()),
            'share': float(answered.sum() / coded.sum()) if coded.any() else None, 'short_sessions': short}


def _complete(variant: dict, base: pd.DataFrame) -> bool:
    share = coverage(variant, base)
    return share['answered'] == share['coded']


def score_variant(variant: dict, base: pd.DataFrame, k: int, n_boot: int) -> tuple[dict, list]:
    """every metric of 4.4 for one variant over the pooled held-out windows it answered
    (metrics.report), the unit-bootstrap interval of its pooled macro-F1 (three-class and
    binary; left out, with the reason, below MIN_BOOTSTRAP_UNITS units), the onset latency of an online variant, the seed mean of the stratified floor, the
    2-state HMM check and its coverage; and its per-session rows. A coded window the variant gave
    no answer for is left out rather than counted as a miss, and `coverage` says how many."""
    # a window without an answer is scored as not coded; its share is the coverage
    y = np.where(np.asarray(variant['y_pred']) >= 0, _labels_int(base['y_true']), -1)
    sessions, lessons = base['session'].to_numpy(), base['lesson'].to_numpy()
    tasks, blocks = base['task'].to_numpy(dtype=object), base['block'].to_numpy()
    p, y_pred, y_binary = variant['p'], variant['y_pred'], variant['y_pred_binary']
    strata = P.strata(base['n_observed'].to_numpy(), base['gaze_readable'].to_numpy(dtype=bool))
    report = M.report(y, p, sessions, tasks, base['empty_window'].to_numpy(), blocks, y_pred=y_pred,
                      y_pred_binary=y_binary, viterbi_path=variant.get('viterbi'), strata=strata)
    per_session = report.pop('per_session', [])
    report['coverage'] = coverage(variant, base)
    coded = y >= 0
    yc, pc, gc = y[coded], y_pred[coded], lessons[coded]
    truth, binary = _binary_truth(y)[coded], y_binary[coded]
    report['macro_f1_ci'] = _interval(lambda rows: M.macro_f1(yc[rows], pc[rows]), gc, n_boot)
    report['binary_macro_f1_ci'] = _interval(
        lambda rows: M.macro_f1(truth[rows], binary[rows], n_classes=2), gc, n_boot)
    if variant['hmm'] == 'filter' and k == 3:
        report['onset_latency'] = M.onset_latency(y, y_pred, blocks, sessions)
    if variant.get('seed_preds'):
        report['macro_f1_seed_mean'] = float(np.nanmean([M.macro_f1(y, s) for s in variant['seed_preds']]))
    if variant.get('binary_hmm') is not None:
        check = variant['binary_hmm']
        report['binary_hmm_check'] = {
            'f1_interaction': M.f1_per_class(_binary_truth(y), check, 2)[0][1],
            'agreement_with_derived': float((check[coded] == y_binary[coded]).mean()) if coded.any() else None}
    return report, per_session


def _drop_samples(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != 'samples'}


def _too_few_units(units: int) -> str | None:
    """why a bootstrap over this many units is left out, or None when it is not."""
    if units >= MIN_BOOTSTRAP_UNITS:
        return None
    return (f"{units} unit(s) (dates) to resample, fewer than {MIN_BOOTSTRAP_UNITS}: "
            f"the percentiles would be those dates' own scores")


def _interval(fn, groups, n_boot: int) -> dict:
    """the unit-bootstrap interval of a pooled metric (metrics.session_bootstrap) with the number
    of units it resampled; with fewer than MIN_BOOTSTRAP_UNITS the estimate stands alone, lo and
    hi are None and `left_out` says why."""
    units = len(pd.unique(np.asarray(groups, dtype=object)))
    why = _too_few_units(units)
    if why is None:
        return dict(_drop_samples(M.session_bootstrap(fn, groups, n=n_boot)), units=units)
    estimate = float(fn(np.arange(len(groups))))
    return {'estimate': estimate if np.isfinite(estimate) else None, 'lo': None, 'hi': None, 'n': 0,
            'n_undefined': 0, 'units': units, 'left_out': why}


def _result_row(key: str, variant: dict, report: dict) -> dict:
    """one row of the results table (4.7)."""
    pooled = report['pooled']
    temporal = pooled.get('temporal') or {}
    ci = report['macro_f1_ci']
    group = 'online' if variant['hmm'] == 'filter' else GROUPS[variant['model']]
    row = {'group': group, 'variant': key, 'model': variant['model'], 'temporal': variant['temporal'],
           'hmm': variant['hmm'], 'n': pooled['n'], 'coverage': (report.get('coverage') or {}).get('share'),
           'macro_f1': pooled['macro_f1'], 'macro_f1_lo': ci['lo'], 'macro_f1_hi': ci['hi']}
    row.update({f'f1_{name}': value for name, value in pooled['f1'].items()})
    row.update({'balanced_accuracy': pooled['balanced_accuracy'], 'kappa': pooled['kappa'],
                'binary_f1': pooled['binary']['f1_interaction'], 'auroc': pooled['binary']['auroc'],
                'nll': pooled['nll'], 'ece': pooled['ece_top'],
                'switches_per_hour_pred': temporal.get('switches_per_hour_pred'),
                'switches_per_hour_true': temporal.get('switches_per_hour_true'),
                'state_share_error': (report.get('state_share_error') or {}).get('err_mean')})
    for task, scores in (report.get('by_task') or {}).items():
        row[f'macro_f1_{task}'] = scores['macro_f1']
    for name, levels in (report.get('strata') or {}).items():
        for level, scores in levels.items():
            row[f'n_{name}_{level}'] = scores['n']
            row[f'macro_f1_{name}_{level}'] = scores['macro_f1']
    return row


def select_headline(variants: dict, reports: dict, binary: bool = False) -> str | None:
    """the pre-registered headline: the best of late-lr and late-hgb, over the temporal modes, with
    and without the HMM, by pooled dev leave-one-date-out macro-F1 (the binary one under the
    absent-class policy);
    the first in run order on a tie."""
    best = None
    for key, variant in variants.items():
        if variant['model'] not in HEADLINE_MODELS or variant['hmm'] not in ('none', 'fb'):
            continue
        score = reports[key]['binary_macro_f1_ci' if binary else 'macro_f1_ci']['estimate']
        if score is not None and (best is None or score > best[0]):
            best = (score, key)
    return None if best is None else best[1]


def share_variant(headline: str | None, variants: dict) -> str | None:
    """the variant whose decisions the end-to-end state shares take: the headline, or where there
    is none (a test run) the first late-lr or late-hgb variant without the forward filter, in run
    order; None when the run has neither."""
    if headline is not None:
        return headline
    for key, variant in variants.items():
        if variant['model'] in HEADLINE_MODELS and variant['hmm'] in ('none', 'fb'):
            return key
    return None


def contrasts(headline: str | None, variants: dict, base: pd.DataFrame, n_boot: int, binary: bool = False) -> dict:
    """the three confirmatory contrasts on pooled dev macro-F1, paired on the same unit (date)
    resamples and Holm-corrected: C1 the headline against the fitted rule R1, C2 the headline's
    model and temporal mode with the HMM against without it, C3 the headline against zero-shot Jev
    J0. A contrast is computed only when both of its variants answered every coded held-out
    window, so both sides are scored on the same windows, and when the coded windows span at least
    MIN_BOOTSTRAP_UNITS units; one the run cannot give comes back with
    `left_out` saying why, and stays out of the Holm family."""
    if headline is None:
        return {}
    model, temporal, mode = headline.split(':')
    wanted = {'C1': (headline, 'r1:T0:none'),
              'C2': (_key(model, temporal, 'fb'), _key(model, temporal, 'none')),
              'C3': (headline, 'jev:j0:none')}
    hints = {'r1:T0:none': 'add -m r1', 'jev:j0:none': 'add -m jev, with answers from mmla ses-jev --variant j0'}
    y = _labels_int(base['y_true'])
    coded = y >= 0
    truth, lessons = (_binary_truth(y) if binary else y)[coded], base['lesson'].to_numpy()[coded]

    def score(key):
        pred = (variants[key]['y_pred_binary'] if binary else variants[key]['y_pred'])[coded]
        return lambda rows: M.macro_f1(truth[rows], pred[rows], n_classes=2 if binary else 3)

    few = _too_few_units(len(pd.unique(np.asarray(lessons, dtype=object))))
    out = {}
    for name, (a, b) in wanted.items():
        absent = [key for key in (a, b) if key not in variants]
        partial = [key for key in (a, b) if key in variants and not _complete(variants[key], base)]
        if few:
            out[name] = {'a': a, 'b': b, 'left_out': few}
        elif absent:
            why = f"the run has no {' or '.join(absent)}" + (f" ({hints[absent[0]]})" if absent[0] in hints else '')
            out[name] = {'a': a, 'b': b, 'left_out': why}
        elif partial:
            share = coverage(variants[partial[0]], base)
            out[name] = {'a': a, 'b': b, 'left_out': f"{partial[0]} answered {share['answered']} of {share['coded']} "
                                                      f"coded windows; both sides must answer every one"}
        else:
            out[name] = dict(_drop_samples(M.paired_delta(score(a), score(b), lessons, n=n_boot)), a=a, b=b,
                             metric='binary macro-F1' if binary else 'macro-F1')
    tested = [name for name, c in out.items() if 'left_out' not in c]
    if tested:
        adjusted = M.holm({name: (out[name]['p'] if out[name]['p'] is not None else 1.0) for name in tested})
        for name in tested:
            out[name]['p_holm'] = adjusted[name]
    return out


def inter_coder(data: dict, primary: str | None) -> dict:
    """Cohen's kappa of every other coder against the primary one, pooled over the windows both
    coded (3-class and binary): the ceiling the results are read against."""
    out = {}
    others = sorted({name for d in data.values() for name in d.others} - {primary, None})
    for name in others:
        a, b = [], []
        for d in data.values():
            if primary in d.others and name in d.others:
                index = pd.MultiIndex.from_arrays([[d.session] * len(d), d.tokens.window_index])
                a.append(pd.Series(d.others[primary], index=index))
                b.append(pd.Series(d.others[name], index=index))
        if a:
            out[name] = L.agreement(pd.concat(a), pd.concat(b))
    return out


def predictions_frame(base: pd.DataFrame, variants: dict, k: int) -> pd.DataFrame:
    """predictions.csv: one row per held-out window and variant, in the columns of 4.7 (with the
    temporal mode, the HMM mode and the Viterbi state after them)."""
    frames = []
    for key, variant in variants.items():
        frame = base.drop(columns=['block']).copy()
        frame['model'] = variant['model']
        frame['variant'] = f"{variant['temporal']}:{variant['hmm']}"
        p = variant['p']
        n = len(frame)
        if p is None:
            p_ind = p_soc = p_col = p_int = np.full(n, np.nan)
        elif k == 3:
            p_ind, p_soc, p_col, p_int = p[:, 0], p[:, 1], p[:, 2], p[:, 1] + p[:, 2]
        else:
            p_ind, p_soc, p_col, p_int = p[:, 0], np.full(n, np.nan), np.full(n, np.nan), p[:, 1]
        frame['p_individual'], frame['p_social'] = p_ind, p_soc
        frame['p_collaborative'], frame['p_interaction'] = p_col, p_int
        frame['y_pred'] = pd.array(np.where(variant['y_pred'] >= 0, variant['y_pred'], -1), dtype='Int64')
        frame['y_pred_binary'] = pd.array(variant['y_pred_binary'], dtype='Int64')
        frame['temporal'], frame['hmm'] = variant['temporal'], variant['hmm']
        viterbi = variant.get('viterbi')
        frame['y_viterbi'] = pd.array(viterbi if viterbi is not None else [pd.NA] * n, dtype='Int64')
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    # uncoded is empty, unclear -1, absent -2
    out['y_true'] = out['y_true'].astype(float).astype('Int64')
    return out[list(PREDICTION_COLUMNS)]


def confusion_frame(reports: dict, k: int) -> pd.DataFrame:
    """confusion.csv: the pooled confusion of every variant, one row per (true, predicted) cell."""
    names = class_names(k)
    rows = []
    for key, report in reports.items():
        matrix = report['pooled']['confusion']
        for i, row in enumerate(matrix[:k]):
            for j, count in enumerate(row[:k]):
                rows.append({'variant': key, 'true': names[i], 'predicted': names[j], 'windows': int(count)})
    return pd.DataFrame(rows, columns=['variant', 'true', 'predicted', 'windows'])


# ---- files ----

def _jsonable(value):
    """a value json can write: numpy scalars and arrays as Python ones, NaN as None."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False), encoding='utf-8')


def _config_record(cfg: Config, plan: dict, data: dict, folds: list, artifacts: Path, run_dir: Path,
                   coder: str | None = None) -> dict:
    """config.json: what the run read (every fused table and label file by sha256), whose labels
    were the truth and how that coder was chosen, with what (feature lists, grids, seeds, the
    layout version) and which software, secrets masked."""
    from openmmla.utils.session_provenance import git_commit, redact_secrets, software_info
    values = LY.GROUP_VALUES + LY.PERSON_VALUES + LY.PAIR_VALUES
    record = {
        'run': run_dir.name, 'created_at': datetime.now(timezone.utc).isoformat(), 'config': asdict(cfg),
        'coder': coder, 'coder_chosen_by': '--coder' if cfg.coder else 'the most windows over the DEV sessions',
        'models_run': list(plan['models']),
        'layout_version': LY.LAYOUT_VERSION,
        'features': {'pooled_columns': list(LY.POOLED_COLUMNS),
                     'pooled_blocks': {name: list(columns) for name, columns in LY.POOLED_BLOCKS.items()},
                     'lag_columns': list(LY.LAG_COLUMNS),
                     'lags': {mode: [suffix for suffix, _, _ in lags] for mode, lags in LY.LAGS.items()},
                     # not called 'tokens': redact_secrets masks any key that names a token
                     'window_layout': {'G': list(LY.G_COLUMNS), 'P': list(LY.P_COLUMNS), 'Q': list(LY.Q_COLUMNS),
                                       'availability': list(LY.AVAILABILITY)},
                     'values': {v.name: {'source': v.source, 'transform': v.transform, 'scale': v.tag, 'mask': v.mask,
                                         'modality': v.modality} for v in values},
                     'dropped': dict(LY.DROPPED)},
        'rule': {'version': TB.RULE_VERSION, 'fixed': TB.RULES[TB.RULE_VERSION]['fixed'],
                 'columns': {role: list(names) for role, names in TB.RULES[TB.RULE_VERSION]['columns'].items()},
                 'thresholds': dict(TB.RULES[TB.RULE_VERSION]['thresholds'])},
        'rule_thresholds': dict(TB.RULE_THRESHOLDS),
        'presence_gate': dict(P.RULE),
        'grids': {'lr': plan['lr_grid'], 'hgb': plan['hgb_grid']},
        'network': {'max_epochs': plan['max_epochs'], 'patience': plan['patience'],
                    'seeds': list(range(plan['seeds'])), 'small': plan['small'], 'epochs': plan['epochs']},
        'hmm': {'gammas': list(H.GAMMAS), 'alpha': 1.0, 'diagonal': 10.0,
                'filter_inputs': f"{', '.join(CAUSAL_INPUTS)}; the held-out sessions scaled by the running normaliser"},
        'min_class_windows': cfg.min_class_windows, 'bootstrap': plan['bootstrap'], 'inner_folds': INNER_FOLDS,
        'folds': [{'name': fold.name, 'train': list(fold.train), 'test': list(fold.test)} for fold in folds],
        'test_sessions': list(S.TEST_SESSIONS),
        'files': {s: {'table': str(d.table_path), 'table_sha256': d.table_sha256, 'labels': d.label_files}
                  for s, d in data.items()},
        'software': software_info(artifacts.parent), 'git_commit': git_commit(artifacts.parent),
    }
    return redact_secrets(_jsonable(record))


def _record_test_run(artifacts: Path, run_dir: Path, cfg: Config, status: str) -> list:
    """the audit trail of the one scoring of the TEST sessions:
    artifacts/_analysis/interaction/test_runs.jsonl gets a line when a test run starts and one when
    it finishes. The earlier lines come back, so the caller can say the set was touched before."""
    path = artifacts / '_analysis' / 'interaction' / 'test_runs.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    earlier = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] \
        if path.exists() else []
    line = {'run': run_dir.name, 'run_dir': str(run_dir), 'status': status,
            'at': datetime.now(timezone.utc).isoformat(), 'models': list(cfg.models), 'coder': cfg.coder,
            'confirm_frozen': bool(cfg.confirm_frozen)}
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(line) + '\n')
    return earlier


# ---- the run ----

def planned_variants(models, temporal=TEMPORAL, hmm=HMM_MODES, jev_variant: str = 'j0') -> dict:
    """model -> the variant keys (model:temporal:hmm) a run with these flags gives it, before
    anything is loaded: r0, the floors and zero-shot Jev give one whatever the flags say, r1 reads
    the T0 view only, a network its own temporal mode, and the forward filter goes only to causal
    inputs. A model with an empty list would be trained on every fold for nothing."""
    out = {}
    for model in dict.fromkeys(models):
        if model in ('r0', 'r0-v1', 'majority', 'stratified'):
            out[model] = [_key(model, 'T0', 'none')]
            continue
        if model == 'jev':
            out[model] = [_key(model, jev_variant, 'none')]
            continue
        modes = [jev_variant] if model == 'jev-cal' else [NET_TEMPORAL[model]] if model in NETWORKS \
            else ['T0'] if model == 'r1' else list(dict.fromkeys(temporal))
        out[model] = [_key(model, t, h) for t in modes for h in dict.fromkeys(hmm)
                      if h != 'filter' or t in CAUSAL_INPUTS]
    return out


def _barren(cfg: Config) -> list:
    """the named models the flags give no variant."""
    planned = planned_variants(cfg.models, cfg.temporal, cfg.hmm, cfg.jev_variant)
    return [model for model, keys in planned.items() if not keys]


def _check(cfg: Config):
    unknown = [m for m in cfg.models if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s) {', '.join(unknown)}: one of {', '.join(MODELS)}")
    if not cfg.models:
        raise ValueError("name at least one model")
    if cfg.split not in SPLITS:
        hint = " ('loso' became 'date': the same group on one date is never on both sides)" \
            if cfg.split == 'loso' else ' (task transfer is deferred)'
        raise ValueError(f"split {cfg.split!r} is not built: one of {', '.join(SPLITS)}{hint}")
    if cfg.target not in TARGETS:
        raise ValueError(f"target must be one of {', '.join(TARGETS)}")
    bad = [t for t in cfg.temporal if t not in TEMPORAL] + [h for h in cfg.hmm if h not in HMM_MODES]
    if bad or not cfg.temporal or not cfg.hmm:
        raise ValueError(f"temporal modes are {', '.join(TEMPORAL)} and HMM modes {', '.join(HMM_MODES)}")
    if cfg.split == 'test' and not cfg.confirm_frozen:
        raise ValueError("the TEST sessions are scored once, after every choice is frozen: confirm with confirm_frozen")
    if cfg.split == 'test' and not cfg.coder:
        raise ValueError("the TEST scoring names its truth coder: pass the coder the date runs were chosen on "
                         "(their config.json 'coder')")
    barren = _barren(cfg)
    if barren:
        raise ValueError(f"{', '.join(barren)} give(s) no variant with temporal {','.join(cfg.temporal)} and HMM "
                         f"{','.join(cfg.hmm)}: the forward filter runs only for inputs that never read a later "
                         f"window ({', '.join(CAUSAL_INPUTS[:2])}, net-notcn, Jev)")
    if any(m in NETWORKS for m in cfg.models) and importlib.util.find_spec('torch') is None:
        raise ModuleNotFoundError("the network variants need torch: pip install torch")


def _plan(cfg: Config) -> dict:
    """the settings every fold needs, as a plain dict a worker process receives."""
    quick = QUICK if cfg.quick else {}
    jobs = max(1, int(cfg.jobs))
    return {'k': 3 if cfg.target == '3class' else 2, 'models': list(dict.fromkeys(cfg.models)),
            'temporal': list(dict.fromkeys(cfg.temporal)), 'hmm': list(dict.fromkeys(cfg.hmm)),
            'lr_grid': quick.get('lr_grid', TB.LR_GRID), 'hgb_grid': quick.get('hgb_grid', TB.HGB_GRID),
            'max_epochs': quick.get('max_epochs', 300), 'patience': quick.get('patience', 25),
            'seeds': int(cfg.seeds), 'small': cfg.small, 'epochs': cfg.epochs, 'jev_variant': cfg.jev_variant,
            'bootstrap': cfg.bootstrap or quick.get('bootstrap', BOOTSTRAP), 'jobs': jobs,
            'threads': max(1, (os.cpu_count() or 1) // jobs) if jobs > 1 else None}


def _run_folds(folds: list, data: dict, plan: dict, say) -> list:
    if plan['jobs'] == 1 or len(folds) == 1:
        outputs = []
        for n, fold in enumerate(folds, start=1):
            outputs.append(_fold_job(fold, data, plan))
            say(f"fold {n}/{len(folds)} {fold.name}: {outputs[-1]['seconds']} s")
        return outputs
    from joblib import Parallel, delayed
    say(f"{len(folds)} folds on {plan['jobs']} workers")
    # each worker gets only the sessions its fold reads
    return Parallel(n_jobs=plan['jobs'])(
        delayed(_fold_job)(fold, {s: data[s] for s in list(fold.train) + list(fold.test)}, plan) for fold in folds)


def run(config, log=None) -> Path:
    """one evaluation run: load, count and check the labels, run every outer fold (in parallel
    with `jobs`), score every variant and write the run folder. Returns the run folder; raises
    Refused (after writing label_counts.csv) when a learned model may not be trained, and
    labels.LabelJoinError when a coder's labels do not fit the table's grid. `log` gets progress
    lines (the command prints them; the library prints nothing)."""
    cfg = config if isinstance(config, Config) else Config(**dict(config))
    say = log or (lambda message: None)
    started = time.time()
    _check(cfg)
    plan = _plan(cfg)
    k = plan['k']
    artifacts = Path(cfg.artifacts).resolve()
    found = session_tables(artifacts, cfg.sessions)
    if cfg.split == 'date':
        # a dev run never opens a TEST session, not even its labels
        found = [(s, p) for s, p in found if s not in S.TEST_SESSIONS]
    if not found:
        raise FileNotFoundError(f"no fused table under {artifacts} for {cfg.sessions or 'any session'}: "
                                f"run mmla ses-fuse")
    # the truth coder is chosen on the DEV sessions in every split, so TEST labels never decide it
    coder = cfg.coder or primary_coder([path.parents[2] for s, path in found if s not in S.TEST_SESSIONS])
    data, tables, excluded = {}, {}, {}
    for session, path in found:
        loaded, table = load_session(session, path, coder, cfg.join, cfg.target)
        # S1, the inclusion rule: a session that never shows two persons together is left out whole
        included, reason = LY.session_inclusion(table, loaded.roster)
        if not included:
            excluded[session] = {**loaded.roster.record(), 'included': False, 'reason': reason}
            say(f"{session} left out: {reason}")
            continue
        data[session], tables[session] = loaded, table
    if not data:
        raise Refused("every session is left out by the inclusion rule S1: "
                      + '; '.join(f"{s} ({r['reason']})" for s, r in excluded.items()))
    link_lessons(data)
    if any(m in JEV_MODELS for m in plan['models']):
        for d in data.values():
            d.jev, d.jev_note = jev_log_proba(d, cfg.jev_variant)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = Path(cfg.out) if cfg.out else artifacts / '_analysis' / 'interaction' / f'{cfg.split}-{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    counts = label_counts(data)
    counts.to_csv(run_dir / 'label_counts.csv')
    say(f"labels of coder {coder or '(none)'}, windows per class and session:\n{counts.to_string()}")
    write_json(run_dir / 'roster.json', {**{s: {**d.roster.record(), 'included': True} for s, d in data.items()}, **excluded})
    write_json(run_dir / 'data_checks.json', LY.data_checks(tables, {s: d.roster for s, d in data.items()}))

    try:
        folds, split_error = make_folds(cfg.split, data), None
    except S.SplitError as error:
        folds, split_error = [], str(error)
    notes = {s: d.jev_note for s, d in data.items() if d.jev_note}
    dropped = jev_gaps(folds, data, plan['models'], cfg.jev_variant)
    if dropped:
        plan['models'] = [m for m in plan['models'] if m not in dropped]
        for model, reason in dropped.items():
            say(f"{model} left out: {reason}")
    learned = [m for m in plan['models'] if m not in UNLEARNED]
    problems = label_refusal(folds, data, k, cfg.min_class_windows) if learned else []
    held = sum(data[s].n_coded for fold in folds for s in fold.test)
    why = None
    if split_error:
        why = split_error
    elif not plan['models']:
        why = f"every model named was left out ({', '.join(dropped)}: no usable Jev maps, see left_out and notes)"
    elif not folds:
        why = 'no TEST session has a fused table' if cfg.split == 'test' else 'no date has coded windows to hold out'
    elif not held:
        why = 'the held-out sessions have no coded window to score'
    elif problems:
        why = f"{len(problems)} outer-training fold(s) lack {cfg.min_class_windows} coded windows of a class " \
              f"(or two coded dates for the inner folds)"
    if why:
        write_json(run_dir / 'metrics.json', {'run': run_dir.name, 'refused': why, 'problems': problems,
                                              'models': plan['models'], 'target': cfg.target,
                                              'left_out': dropped, 'notes': notes})
        raise Refused(f"refusing to train {', '.join(learned) or 'anything'}: {why}", problems, run_dir)

    if cfg.split == 'test':
        earlier = _record_test_run(artifacts, run_dir, cfg, 'started')
        if earlier:
            say(f"the TEST sessions were scored before ({len(earlier)} line(s) in test_runs.jsonl): "
                f"this run is recorded as another look")
    say(f"{len(folds)} fold(s) over {len(data)} session(s): {', '.join(plan['models'])}")
    outputs = _run_folds(folds, data, plan, say)

    base = _base_rows(outputs, data)
    variants = _variants(outputs, data, k)
    if not variants:
        # _check refuses flags that give no variant; this guards the files below
        raise Refused("the folds gave no variant to score", [], run_dir)
    reports, per_session = {}, []
    for key, variant in variants.items():
        reports[key], rows = score_variant(variant, base, k, plan['bootstrap'])
        per_session += [dict(row, variant=key) for row in rows]
    policy = absent_class_policy(data)
    binary_confirmatory = policy['confirmatory_target'] == 'binary' or k == 2
    headline = select_headline(variants, reports, binary_confirmatory) if cfg.split == 'date' else None
    tests = contrasts(headline, variants, base, plan['bootstrap'], binary_confirmatory) if headline else {}
    presence_gated = base['presence_gated'].to_numpy(dtype=bool)
    gate = P.evaluate_gate(base['y_true'].to_numpy(dtype=float), presence_gated, base['session'].to_numpy(),
                           base['n_observed'].to_numpy())
    share_key = share_variant(headline, variants)
    if share_key:
        # per lesson, not per unit: a unit's lessons would pool their share errors and cancel them
        frame = P.state_shares(base['y_true'].to_numpy(dtype=float), variants[share_key]['y_pred'], presence_gated,
                               base['session'].map(S.lesson_key).to_numpy(), class_names(k))
        frame.to_csv(run_dir / 'state_shares.csv', index=False)
        shares = {'variant': share_key, **P.share_summary(frame)}
    else:
        shares = {'left_out': 'the run has no late-lr or late-hgb variant without the forward filter'}

    predictions_frame(base, variants, k).to_csv(run_dir / 'predictions.csv', index=False)
    pd.DataFrame(per_session).to_csv(run_dir / 'per_session.csv', index=False)
    confusion_frame(reports, k).to_csv(run_dir / 'confusion.csv', index=False)
    results = pd.DataFrame([_result_row(key, variants[key], reports[key]) for key in variants])
    ceiling = inter_coder(data, coder)
    for name, agreement in ceiling.items():
        results = pd.concat([results, pd.DataFrame([{'group': 'ceiling', 'variant': f'inter-coder {name}',
                                                     'model': 'coder', 'n': agreement['windows'],
                                                     'kappa': agreement['kappa'],
                                                     'kappa_codes': agreement['kappa_codes']}])], ignore_index=True)
    results.to_csv(run_dir / 'results.csv', index=False)
    coefficients = [frame for out in outputs for frame in (out['coefficients'] or [])]
    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(run_dir / 'coefficients.csv', index=False)

    scored = np.concatenate([d.y for d in data.values()])
    metrics = {
        'run': run_dir.name, 'split': cfg.split, 'target': cfg.target, 'classes': list(class_names(k)),
        'quick': cfg.quick, 'seconds': round(time.time() - started, 1),
        'labels': {'coder': coder, 'coded': int(L.scored(scored).sum()), 'unclear': int((scored == L.UNCLEAR_Y).sum()),
                   'absent': int((scored == L.ABSENT_Y).sum()),
                   'join': {s: d.join for s, d in data.items()}},
        'inter_coder': ceiling, 'absent_class_policy': policy,
        'headline': {'variant': headline, 'selected_on': 'dev leave-one-date-out pooled '
                     + ('binary macro-F1' if binary_confirmatory else 'macro-F1'),
                     'strata': reports[headline].get('strata')} if headline else None,
        'contrasts': tests, 'presence_gate': gate, 'state_shares': shares, 'notes': notes, 'left_out': dropped,
        'folds': [{key: out[key] for key in ('name', 'train', 'test', 'inner', 'prior', 'transitions', 'models',
                                             'seconds')} for out in outputs],
        'variants': {key: dict(reports[key], model=variants[key]['model'], temporal=variants[key]['temporal'],
                               hmm=variants[key]['hmm'], group=_result_row(key, variants[key], reports[key])['group'])
                     for key in variants},
    }
    write_json(run_dir / 'metrics.json', metrics)
    write_json(run_dir / 'config.json', _config_record(cfg, plan, data, folds, artifacts, run_dir, coder))
    if cfg.split == 'test':
        _record_test_run(artifacts, run_dir, cfg, 'finished')
    say(f"{len(variants)} variant(s) scored in {metrics['seconds']} s -> {run_dir}")
    return run_dir
