"""The evaluation driver of the 10 s interaction classifier (Layer A): it reads every session's
fused table and coded labels, holds out one date at a time (or, once, the TEST sessions),
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

`social_decision='infold'` (exploratory, three-class target only) replaces the three-class hard
label of every model with inner out-of-fold answers by a two-step decision
(tabular.two_step_decision): interaction against individual by the balanced binary rule, so every
binary label stays the balanced decision's, then social where p_social / (p_social +
p_collaborative) >= tau, else collaborative. tau is chosen per variant (model, temporal mode, HMM
mode) and outer fold on the calibrated inner out-of-fold answers of the training side, smoothed by
the variant's HMM with its gamma, as the highest social F1 of the same decision over the coded
windows (tabular.social_threshold); a fold whose inner answers hold fewer than
tabular.MIN_SOCIAL_WINDOWS social windows keeps the balanced decision. rule22sep scores its
candidates' three-class macro-F1 with the same decision and taus. The rule, the floors and zero-shot
Jev keep their own labels. Each fold's tau is in metrics.json (social_tau under the fold's models,
and social_decision), the mode in config.json.

`social_decision='infold-all'` (exploratory, three-class target only) decides in one step instead
(tabular.one_step_decision): social where p_social >= tau_all, otherwise the balanced decision
between individual and collaborative (collaborative where p_c / pi_c >= p_i / pi_i). tau_all is
chosen the same way, per variant and outer fold on the same smoothed inner answers, as the highest
social F1 over all coded inner windows (tabular.social_threshold_all), with the same fallback and
records. The binary label is read off the three-class one (interaction is social or
collaborative), so unlike 'infold' it can differ from the balanced binary decision; every
probability, and so every AUROC, is unchanged.

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
least MIN_BOOTSTRAP_UNITS units, so a test run whose TEST sessions fall on fewer dates reports
its estimates without intervals.

The modality ablation of 4.6 (`ablate='modality'`) runs in the same run, on the same folds: every
learned model and the rule are fitted and scored once per arm of MODALITY_ARMS, where an arm's
blocks did not run in any window of any session, train and test alike (layout.ablate: values
unobserved, masks and availability bit 0, as an outage writes them, the empty flag kept). The
blocks are the three modalities and the transcript content of layout version 5 (no_content,
only_content; an only_* arm removes every other block, the content included). The
floors and Jev read no feature and run in the full arm only. The headline, the contrasts and the
state shares are the full arm's, which is the run without the ablation; ablation.csv sets the arms
side by side (exploratory, no Holm correction). An ablated arm's variant key ends in ':<arm>'. Late
fusion has no expert for a block an arm removed or one present in no coded training window (the
content block of tables without its columns).
The gaze-model ablation (`ablate='gaze_model'`, for the sensor-value ladder) runs the
arms of GAZE_MODEL_ARMS the same way: only_body_gaze, only_pose (the cameras' pose values alone:
layout's pseudo-modality gaze_model removed too) and no_gaze_model. The gaze model's values and
masks are unobserved as an outage writes them; vfa_ran, the pose values and the body_gaze expert of
late fusion stay. The diarization ablation (`ablate='dia'`) runs the arm no_dia of DIA_ARMS the same
way: the group microphone's dia_* values and their mask are unobserved (layout's pseudo-modality
dia), while the level, silence and word values, asr_ran and the speech expert of late fusion stay.

The session split (`split='session'`, for when the TEST sessions alone cannot stand for the
model) pools DEV and TEST: every session the named coder
labelled that S1 keeps is held out in turn and the models train on all the others
(splits.session_folds), with leave-one-session-out inner folds within the training sessions
(splits.session_inner_folds) and the session as the unit everywhere (the lesson column, the
bootstrap). Everything fitted is still fitted on a fold's training sessions only, as above. The
truth is the coder's own labels file alone (adjudicated.jsonl is not read for it); a session the
coder did not label, or coded only unclear or absent, is left out with the reason, which names a
labels file there whose name differs from the coder's only in case (case_hint). Each held-out
session is scored on its own (session_scores: macro-F1, accuracy, per-class F1, kappa, binary F1,
AUROC, Brier, NLL), and every variant gets the mean and SD over the sessions beside its pooled
value (session_summary). There is no headline and no contrast, since nothing here was declared in
advance. coder_agreement scores two coders against each other on the same sessions after the
same join, training nothing.

The unit and forward splits (`split='unit'`, `'forward'`) read the sessions the
session split reads, the same way (DEV and TEST alike, the coder's own labels file, the same
sessions left out), but group them as the date split does: the unit is a date with the dates its
sessions' same_class_as links reach (splits.class_units), held out whole, resampled whole and named
in the lesson column. The unit split leaves one unit out (splits.unit_folds_all), the forward split
holds out each date with at least splits.MIN_EARLIER_DATES earlier dates and trains on the earlier
dates only (splits.forward_folds); in both the inner folds leave one training unit out
(splits.unit_inner_folds). Each held-out session and each held-out unit is scored on its own
(per_session.csv, per_unit.csv), results.csv gives the mean and SD over the units, and there is no
headline and no contrast in the run (contrasts.py compares variants on predictions.csv).

Two pseudo-variants choose a model inside each outer fold from the calibrated inner out-of-fold
answers of the variants the run fits, and copy the chosen variant's held-out answers (SELECTORS):
`rule22sep` applies select_headline, the pre-declared headline rule, to them, and `select-all` takes
the lowest class-weighted binary log-loss over every variant of the named models. Each fold's
choice is in metrics.json. `net_oof='common'` gives the networks' inner out-of-fold answers at the
common E* of the refit instead of at each inner split's own best epoch (network.oof_at), so they
enter the calibrator, gamma and select-all as the tabular models' do.

Three exploratory candidates of the architecture panel are named only on purpose,
never by -m all, and sit in the results table's group 'exploratory' (EXPLORATORY_NOTES): pmil-lr, a
noisy-OR over the pupil pairs with a shared linear scorer and a bias per group size (pmil.py; binary
target only), lr-soft, lr fitted on two coders' soft labels (the truth coder's and one other's, half
a window each where both gave a class, none where the truth coder said unclear or absent;
tabular.fit_soft), and net-attn, self-attention over the person slots with the pairs as its bias
(network.AttentionNet). Each is fitted, calibrated and smoothed as the model it extends, and each is
a candidate of select-all.

`features_root` reads the fused tables from another folder (an ablation arm's, re-fused) while the
labels, manifests and Jev maps still come from artifacts/<session>/. Every run of the session, unit
and forward splits writes a line when it starts and one when it finishes to
artifacts/_analysis/interaction/session_runs.jsonl (the commit, the settings, every table and
labels file read and every file written, by sha256); log_run_folders adds run folders made before
the ledger to it, as looked at.

Deferred by decision, with their places kept: the other ablation grids of 4.6 (temporal, fusion,
ladder, weights), the causal network row, the lexicon feature, masked-modality pretraining, and
the task-transfer run (splits.task_transfer exists; the driver does not run it yet).
"""
from __future__ import annotations

import dataclasses
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
# the exploratory candidates of the architecture panel: named only on purpose, never by -m all
# (MODELS), and grouped apart in the results table
EXPLORATORY = ('pmil-lr', 'lr-soft', 'net-attn')
EXPLORATORY_NOTES = {
    'pmil-lr': "exploratory: noisy-OR over the window's pupil pairs of one shared linear scorer, an instance "
               "being a pair's values and masks with the min and max of its two pupils' (78 columns), one bias per "
               "group size, lr's balanced weights and C grid; binary target only",
    'lr-soft': "exploratory: lr fitted on two coders' soft labels, a window both gave a class two rows of "
               "weight 0.5, one only one of them did one row of 1 (none where the truth coder said unclear or "
               "absent), classes balanced over those weights; chosen on the inner folds, calibrated and scored "
               "against the truth coder alone",
    'net-attn': "exploratory: self-attention over the person slots with the pairs as an additive bias, "
                "attention pooling queried by the group token and group size, then the net family's window "
                "layer, temporal blocks and training",
}
# every model that trains through network.py
NEURAL = NETWORKS + ('net-attn',)
# the models that answer p(interaction) only
BINARY_ONLY = ('pmil-lr',)
# the models that read no label: they run even where a learned model is refused
UNLEARNED = ('r0', 'r0-v1', 'jev')
# the a-priori rule's versions by model name: r0 is the current rule, r0-v1 the first one, kept for the record
# (version 2 is no model of its own: on a layout version 4 view it reads version 3's hand distance, and so
# is version 3)
RULE_MODELS = {'r0': TB.RULE_VERSION, 'r0-v1': 1}
HEADLINE_MODELS = ('late-lr', 'late-hgb')
TEMPORAL = ('T0', 'T1c', 'T2')
HMM_MODES = ('none', 'fb', 'filter')
# 'date' leaves one date out of DEV (the unit of every split); 'test' scores TEST once; 'session' leaves
# one session out over every session the coder labelled, DEV and TEST alike; 'unit' leaves one unit (a date
# with its same_class_as dates) out over those sessions, and 'forward' trains on earlier dates only
SPLITS = ('date', 'test', 'session', 'unit', 'forward')
# the splits over every session the coder labelled, DEV and TEST alike, with the coder's own file as the truth
POOLED_SPLITS = ('session', 'unit', 'forward')
# the pseudo-variants that choose, inside each outer fold, among the variants the run fits; never in -m all
SELECTORS = ('rule22sep', 'select-all')
# the HMM modes a selector chooses among (the forward filter is the online row, never chosen)
SELECTED_HMM = ('none', 'fb')
# the models whose variants keep calibrated inner out-of-fold answers, the candidates of select-all
INNER_MODELS = TABULAR + NETWORKS + ('jev-cal',) + EXPLORATORY
# the selectors' variant key: model:nested:nested, since the chosen temporal and HMM mode differ by fold
NESTED = 'nested'
GROUP_OF = dict(GROUPS, **{selector: 'selected' for selector in SELECTORS},
                **{model: 'exploratory' for model in EXPLORATORY})
# what metrics.json says of each selector (select_inner has the whole of it)
SELECTOR_NOTES = {
    'rule22sep': "the pre-declared headline rule (select_headline: late-lr or late-hgb, every temporal mode run, "
                 "HMM none or fb) applied in each outer fold to the candidates' calibrated inner out-of-fold "
                 "answers pooled over the inner folds, by binary macro-F1 under the binary target and by "
                 "three-class macro-F1 under the three-class one, even where the absent-class policy makes binary "
                 "macro-F1 the date split's headline score; known optimism: grid points, calibrators, stackers and "
                 "gamma were chosen on the same inner out-of-fold answers (12 grid points for HGB, 5 for LR), with no "
                 "inner level below, for cost",
    'select-all': "in each outer fold, the variant of the named models (HMM none or fb) with the lowest class-weighted "
                  "binary log-loss of its calibrated inner out-of-fold answers, on the coded training windows every "
                  "candidate answered; the same optimism as rule22sep",
}
# the networks' inner out-of-fold answers: at each inner split's best epoch (best, as before) or at the
# common E* of the refit (common, network.oof_at)
NET_OOF = ('best', 'common')
# the three-class hard label: the pre-declared balanced decision, the two-step decision with a social
# threshold chosen inside each outer fold on the inner out-of-fold answers, or the one-step decision with a
# threshold on p_social chosen the same way (the last two three-class target only)
SOCIAL_DECISIONS = ('balanced', 'infold', 'infold-all')
IN_FOLD_DECISIONS = ('infold', 'infold-all')
SOCIAL_DECISION_NOTES = {
    'balanced': "the pre-declared balanced decision, never tuned: argmax p(k) / pi_k under the outer-training "
                "prior; the binary label is interaction when p_int / pi_int >= (1 - p_int) / (1 - pi_int)",
    'infold': "exploratory, three-class target only: interaction against individual by the balanced binary rule "
              "(every binary label is the balanced decision's), then social where p_social / (p_social + "
              "p_collaborative) >= tau, else collaborative; tau from 0.05 to 0.95 in steps of 0.01, chosen per "
              "variant (model, temporal mode, HMM mode) and outer fold as the highest social F1 of the same "
              "two-step decision on the calibrated inner out-of-fold answers of the coded training windows (through "
              "the variant's HMM with its gamma), the lower tau on a tie; a fold whose inner answers hold fewer than "
              f"{TB.MIN_SOCIAL_WINDOWS} social windows keeps the balanced decision; rule22sep scores its candidates' "
              "three-class macro-F1 with the same decision, and a selector's tau is its winner's; the rule, the "
              "floors and zero-shot Jev keep their own labels. Known optimism: tau is chosen on the inner answers "
              "the calibrator, stacker and gamma were fitted on",
    'infold-all': "exploratory, three-class target only: social where p_social >= tau_all, otherwise the balanced "
                  "decision between individual and collaborative (collaborative where p_collaborative / "
                  "pi_collaborative >= p_individual / pi_individual); the binary label is read off it (interaction "
                  "is social or collaborative), so it can differ from the balanced binary decision, while every "
                  "probability and AUROC stays; tau_all from 0.02 to 0.90 in steps of 0.01, chosen per variant "
                  "(model, temporal mode, HMM mode) and outer fold as the highest social F1 over all coded training "
                  "windows of the calibrated inner out-of-fold answers (through the variant's HMM with its gamma), "
                  "the lower tau_all on a tie; a fold whose inner answers hold fewer than "
                  f"{TB.MIN_SOCIAL_WINDOWS} social windows keeps the balanced decision; rule22sep scores its "
                  "candidates' three-class macro-F1 with the same decision, and a selector's tau_all is its winner's; "
                  "the rule, the floors and zero-shot Jev keep their own labels. Known optimism: tau_all is chosen on "
                  "the inner answers the calibrator, stacker and gamma were fitted on",
}
SESSION_RUNS = 'session_runs.jsonl'
TARGETS = ('3class', 'binary')
# the network reads its sequence through its own temporal blocks, not through lag columns
NET_TEMPORAL = {'pooled-net': 'tcn', 'net-notcn': 'T0', 'net': 'tcn', 'net-pair': 'tcn', 'net-attn': 'tcn'}
# inputs whose features never read a later window: only these get the forward filter, the online
# answer, and for it the held-out session is scaled by the running normaliser
CAUSAL_INPUTS = ('T0', 'T1c', 'j0', 'j1')
JEV_MODELS = ('jev', 'jev-cal')
# the modality ablation (4.6): per arm, the blocks (layout.FUSION_BLOCKS) that did not run in any window of any
# session, train and test alike; the full arm first, and it is the run without the ablation. An only_* arm removes
# every other block, the transcript content included; no_speech keeps the content block
MODALITY_ARMS = {'full': (), 'no_speech': ('speech',), 'no_space': ('space',), 'no_body_gaze': ('body_gaze',),
                 'no_content': ('content',),
                 'only_body_gaze': ('speech', 'space', 'content'), 'only_speech': ('space', 'body_gaze', 'content'),
                 'only_content': ('speech', 'space', 'body_gaze')}
# the gaze-model ablation (the sensor-value ladder): the cameras with the gaze
# model's outputs and without them (layout's pseudo-modality gaze_model), the body_gaze block of late fusion kept
GAZE_MODEL_ARMS = {'full': (), 'only_body_gaze': ('speech', 'space', 'content'),
                   'only_pose': ('speech', 'space', 'content', 'gaze_model'), 'no_gaze_model': ('gaze_model',)}
# the diarization ablation: the group microphone's dia_* values (layout's pseudo-modality dia) with and
# without, the rest of speech kept
DIA_ARMS = {'full': (), 'no_dia': ('dia',)}
ABLATIONS = {'none': {'full': ()}, 'modality': MODALITY_ARMS, 'gaze_model': GAZE_MODEL_ARMS, 'dia': DIA_ARMS}
# the models that read no feature: the floors and Jev run in the full arm only (an ablated arm would repeat them)
FEATURELESS = ('majority', 'stratified') + JEV_MODELS

MIN_CLASS_WINDOWS = 30
INNER_FOLDS = 4
# the session split: a held-out session with fewer class-coded windows than this, or with fewer than two
# classes of at least metrics.MIN_SUPPORT windows, is flagged in per_session.csv and in the summary
MIN_SESSION_WINDOWS = 30
# the per-session scores the session split averages, beside each class's F1 (session_metric_names)
SESSION_METRICS = ('macro_f1', 'accuracy', 'balanced_accuracy', 'kappa', 'f1_interaction', 'binary_macro_f1',
                   'auroc', 'brier', 'nll')
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
                      'ablation', 'coded', 'empty_window', 'y_true', 'p_individual', 'p_social', 'p_collaborative',
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
    are lists; `split` is 'date', 'test' or 'session' (leave one session out over every session
    the coder labelled, DEV and TEST alike); `coder` is the truth (default: the coder with the most
    windows over the DEV sessions; a test or session run must name it, and a session run reads
    that coder's own file only); `test_coder`, in a test run only, scores the TEST sessions
    against another coder than the one that trains (a model's DEV labels train, the human coders'
    TEST labels score; adjudication overrules it as it does any coder); `quick` swaps in the QUICK
    grids and training lengths;
    `epochs` fixes the network's E* (the median of the date folds' for the test model) instead of
    the inner choice; `bootstrap` overrides the number of unit resamples; `device` is where the
    networks train: cpu (the default), cuda, or auto (cuda when torch sees a GPU); `ablate` 'modality'
    adds the arms of MODALITY_ARMS to the run, 'gaze_model' those of GAZE_MODEL_ARMS, 'dia' that of
    DIA_ARMS ('none', the default, runs the full arm only); `scaling`
    is layout.scale's scheme for every session, training and held-out alike: 'mix' (the default, by
    each value's tag), 's' (every value within its own session), 'c' (centred within the session on the
    global spread) or 'g' (every value globally). `split` 'unit' and 'forward' read the sessions 'session'
    reads and must name the coder too; `models` may add the SELECTORS; `net_oof` 'common' gives the
    networks' inner out-of-fold answers at the common E* ('best', the default, at each inner split's best
    epoch); `features_root` is a folder to read the fused tables from instead of
    artifacts/<session>/analysis/features/ (see session_tables). `models` may also name the
    EXPLORATORY candidates; `soft_coder` is lr-soft's second coder (its labels file name; default the
    one human coder besides the truth whose labels the training sessions hold, see soft_coder).
    `social_decision` is the three-class hard label: 'balanced' (the default, the pre-declared
    decision), 'infold' (three-class target only: the two-step decision with a social threshold
    chosen in each outer fold on the inner out-of-fold answers) or 'infold-all' (three-class target
    only: the one-step decision with a threshold on p_social chosen the same way, and the binary label
    read off it); SOCIAL_DECISION_NOTES."""
    artifacts: str = 'artifacts'
    sessions: str | None = None
    coder: str | None = None
    test_coder: str | None = None
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
    device: str = 'cpu'
    out: str | None = None
    quick: bool = False
    confirm_frozen: bool = False
    jev_variant: str = 'j0'
    bootstrap: int | None = None
    min_class_windows: int = MIN_CLASS_WINDOWS
    ablate: str = 'none'
    scaling: str = 'mix'
    net_oof: str = 'best'
    features_root: str | None = None
    soft_coder: str | None = None
    social_decision: str = 'balanced'


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
        """the windows no modality ran in and with no transcript content: the HMM gives them a flat
        emission."""
        return (self.tokens.avail.sum(axis=1) == 0) & ~LY.content_observed(self.tokens)

    @property
    def blocks(self) -> np.ndarray:
        """a name per window for the coded stretch it belongs to (ses-code samples 5-minute blocks),
        '' outside one: switches and run lengths are counted within a stretch only."""
        labelled = np.isfinite(self.y)
        starts = labelled & ~np.concatenate([[False], labelled[:-1]])
        number = np.cumsum(starts)
        return np.where(labelled, np.char.add(f'{self.session}#', number.astype(str)), '')


def session_tables(artifacts, pattern: str | None = None, features_root=None) -> list[tuple[str, Path]]:
    """(session, fused table) for every artifacts/exp_*/analysis/features/<session>_window_features.csv
    whose session id contains `pattern`. With `features_root` the tables come from that folder
    instead (features_table), one for each artifacts/exp_* session folder, which still holds the
    session's labels, manifest and Jev maps; a session with no table there is not found (run()
    refuses a run that would miss one, features_root_gaps)."""
    found = []
    for session_dir in sorted(Path(artifacts).glob('exp_*')):
        if pattern and pattern not in session_dir.name:
            continue
        if features_root is not None:
            path = features_table(features_root, session_dir.name)
            if path is not None:
                found.append((session_dir.name, path))
            continue
        path = session_dir / 'analysis' / 'features' / f'{session_dir.name}_window_features.csv'
        if path.exists():
            found.append((session_dir.name, path))
    return found


def features_table(features_root, session: str) -> Path | None:
    """a session's fused table under another root than artifacts/ (an ablation arm's tables,
    re-fused with mmla ses-fuse -o), the first of <root>/<session>_window_features.csv,
    <root>/<session>/<session>_window_features.csv and
    <root>/<session>/analysis/features/<session>_window_features.csv that exists; None when none
    does."""
    root, name = Path(features_root), f'{session}_window_features.csv'
    for path in (root / name, root / session / name, root / session / 'analysis' / 'features' / name):
        if path.is_file():
            return path
    return None


def features_root_gaps(artifacts, cfg: Config, found) -> dict:
    """the sessions a run with cfg.features_root reads no table for although artifacts/ holds one
    the same run would read (`found`: what it reads), each with why that changes nothing: in the
    session, unit and forward splits, a session without labels/<coder>.jsonl, which every run of
    the split leaves out. Any other such session raises FileNotFoundError naming them all, since
    without it the folds, the training sessions and the scored windows are no longer those of the
    run the arm is compared with."""
    have = {s for s, _ in found}
    gaps = [s for s, _ in session_tables(artifacts, cfg.sessions)
            if s not in have and not (cfg.split == 'date' and s in S.TEST_SESSIONS)]
    left_out, missing = {}, []
    for session in gaps:
        if cfg.split in POOLED_SPLITS and cfg.coder not in label_names([Path(artifacts) / session]):
            left_out[session] = f"no labels/{cfg.coder}.jsonl: every {cfg.split} run leaves it out"
        else:
            missing.append(session)
    if missing:
        raise FileNotFoundError(
            f"the features root {cfg.features_root} has no table for {len(missing)} session(s) the run reads from "
            f"{artifacts}: {', '.join(missing)}; re-fuse them there, copy their own tables there to read them "
            f"unchanged, or narrow the run with --sessions")
    return left_out


def primary_coder(directories) -> str | None:
    """the coder with the most windows over every given session, the truth of the whole run (a
    per-session majority could make the truth one coder here and another there). run() passes the
    DEV sessions only, in every split, so how many TEST windows someone coded never decides whose
    labels the models learn from. A model's labels file (labels.model_names) is never picked: a
    model is the truth only when --coder names it."""
    directories = list(directories)
    models = L.model_names(directories)
    frames = [L.load_labels(directory, all_coders=True) for directory in directories]
    frames = [frame[~frame['coder'].isin(models)] for frame in frames]
    frames = [frame for frame in frames if len(frame)]
    return L.primary_coder(pd.concat(frames, ignore_index=True)) if frames else None


def label_names(directories) -> list:
    """the names of every labels file (labels/<name>.jsonl: coders, adjudicated, model files) under
    the given session folders, sorted."""
    names = set()
    for directory in directories:
        folder = Path(directory) / 'labels'
        if folder.is_dir():
            names |= {path.stem for path in folder.glob('*.jsonl')}
    return sorted(names)


def require_coder(coder: str, directories) -> None:
    """raise FileNotFoundError when no given session has labels/<coder>.jsonl. A coder is the file
    name, matched exactly, case included (labels.load_labels compares the stem), so 'alex' never
    reads Alex.jsonl; the error names the files found and a name that differs only in case."""
    names = label_names(directories)
    if coder in names:
        return
    near = [name for name in names if name.lower() == str(coder).lower()]
    hint = f" (names match exactly, case included: did you mean {near[0]}?)" if near else ''
    raise FileNotFoundError(f"no session has labels/{coder}.jsonl{hint}; the labels files found are "
                            f"{', '.join(names) or 'none'}")


def case_hint(directory, coder: str) -> str | None:
    """what a session that holds no labels of `coder` holds instead: its labels files whose name
    differs from the coder's only in case (labels/Alex.jsonl for coder alex), which are never
    read as that coder's, since ses-code keeps a name as the coder typed it; None when there is
    none. require_coder passes as soon as one session has the exact name, so a session coded under
    another case would otherwise be left out as if nobody had labelled it."""
    near = [name for name in label_names([directory]) if name != coder and name.lower() == str(coder).lower()]
    if not near:
        return None
    files = ' and '.join(f'labels/{name}.jsonl' for name in near)
    return (f"{files} {'is' if len(near) == 1 else 'are'} there, not labels/{coder}.jsonl: coder names match "
            f"exactly, case included, so rename it if it is the same coder")


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


def link_lessons(data: dict, by_session: bool = False) -> None:
    """every session's lesson widened to its unit (splits.class_units over the loaded sessions:
    its date, with the dates their same_class_as reach), so predictions, the bootstrap and the
    refusal group what the folds group. `by_session` (the session split) makes every session its
    own unit instead."""
    if by_session:
        for session, d in data.items():
            d.lesson = session
        return
    units = S.class_units(list(data), {s: d.same_class_as for s, d in data.items()})
    for session, d in data.items():
        d.lesson = units[session]


def ablated(d: SessionData, modalities) -> SessionData:
    """the session with these modalities not run in any window (layout.ablate), and the rule's
    unscaled pooled view made from those tokens. Labels, units, Jev answers and the scoring strata
    (n_observed, presence_gated, gaze_readable, blocks) stay the full data's, so every arm is scored
    on the same windows in the same strata; the arrays they share are never changed after loading."""
    tokens = LY.ablate(d.tokens, modalities)
    return dataclasses.replace(d, tokens=tokens, raw=LY.pooled(tokens))


def load_session(session: str, table_path, coder: str | None = None, join: str = 'exact',
                 target: str = '3class', models: set | None = None,
                 adjudicated: bool = True, directory=None) -> tuple[SessionData, pd.DataFrame]:
    """a session and its fused table: roster (the manifest's pupils when it declares them),
    unscaled tokens and pooled view, and the coder's labels joined to the table's grid (a join
    that leaves more than 1 % of labels without a window raises labels.LabelJoinError, which
    aborts the run). The other coders' labels are joined too,
    for the inter-coder kappa; one of theirs that does not join is left out, not fatal, and so is a
    model's labels file (`models`, run() passes labels.model_names of every session; default: this
    session's). The join report counts the truth's rows from the coder's own file and from
    adjudicated.jsonl; with `adjudicated` False (the session split) adjudicated.jsonl is not read
    for the truth, which is the coder's own file alone. `directory` is the session's folder (labels,
    manifest, Jev maps), by default the one the table sits in (artifacts/<session>/analysis/features/);
    a table read from another root (features_table) names it."""
    from openmmla.utils.session_provenance import file_digest
    table_path = Path(table_path)
    table = LY.read_table(table_path)
    directory = table_path.parents[2] if directory is None else Path(directory)
    # the pupils the session's manifest declares, else the roster rules
    ros = LY.session_roster(table, directory)
    tokens = LY.window_tokens(table, ros)
    truth = L.load_labels(directory, coder=coder, adjudicated=adjudicated)
    if len(truth):
        y, report = L.join_labels(table, truth, mode=join)
        y = y.to_numpy(dtype=float)
    else:
        y, report = np.full(len(table), np.nan), {'mode': join, 'labels': 0}
    adjudicated = int((truth['source'] == L.ADJUDICATED).sum()) if len(truth) else 0
    report.update(coder=truth.attrs.get('coder'), skipped_lines=truth.attrs.get('skipped_lines', 0),
                  own=len(truth) - adjudicated, adjudicated=adjudicated)
    others = {}
    everyone = L.load_labels(directory, all_coders=True)
    # a model's labels file is no coder for the ceiling (it may still be the truth, loaded above)
    models = L.model_coders(directory) if models is None else models
    coders = sorted(set(everyone['coder']) - {L.ADJUDICATED} - models) if len(everyone) else []
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


def soft_coder(data: dict, coder: str | None, named: str | None = None) -> tuple[str | None, str | None]:
    """lr-soft's second coder over `data`, the sessions the run trains on: (its name, None), or
    (None, why there is none). `named` must not be the truth coder and must have given a class to a
    window of those sessions; without a name it is the one human coder besides the truth whose
    labels they hold (their joined labels, SessionData.others, which hold no model's labels file),
    and there is none when no such coder or more than one does."""
    holding = sorted({name for d in data.values() for name, y in d.others.items()
                      if name != coder and L.scored(y).any()})
    if named is not None:
        if named == coder:
            return None, f"lr-soft's second coder {named} is the truth coder: name another"
        if named not in holding:
            return None, (f"lr-soft's second coder {named} gave no window of the training sessions a class"
                          + (f" (the other coders there: {', '.join(holding)})" if holding else ''))
        return named, None
    if not holding:
        return None, (f"lr-soft needs a second coder's labels beside {coder}'s, and no other human coder gave a "
                      f"window of the training sessions a class")
    if len(holding) > 1:
        return None, (f"lr-soft takes one second coder, and the training sessions hold {', '.join(holding)} "
                      f"besides {coder}: name one (soft_coder, --soft-coder)")
    return holding[0], None


# ---- folds and the refusal ----

def class_names(k: int) -> tuple:
    return L.CLASSES if k == 3 else ('individual', 'interaction')


def make_folds(split: str, data: dict) -> list:
    """the outer folds: one per DEV date with coded windows (date), dates joined by same_class_as
    together, the one scoring of the TEST sessions (test), or one per session with coded windows,
    DEV and TEST alike (session), one per unit with coded windows, DEV and TEST alike (unit), or one
    per date with enough earlier dates, trained on those (forward). Raises splits.SplitError when, in
    the date or test split, a DEV session shares a date or a same_class_as link with a TEST one."""
    sessions = list(data)
    if split == 'session':
        return S.session_folds(sessions, coded={s: d.n_coded for s, d in data.items()})
    links = {s: d.same_class_as for s, d in data.items()}
    if split == 'unit':
        return S.unit_folds_all(sessions, coded={s: d.n_coded for s, d in data.items()}, same_class=links)
    if split == 'forward':
        return S.forward_folds(sessions, coded={s: d.n_coded for s, d in data.items()}, same_class=links)
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


def absent_class_policy(data: dict, all_sessions: bool = False) -> dict:
    """the pre-registered policy: with fewer than 200 coded social windows on dev, or social at
    least 5 times in fewer than 6 lessons, the confirmatory target becomes the binary one. The
    lessons are counted as registered, by splits.lesson_key, not by the date units the folds hold
    out, so neither the date unit nor a same_class_as link changes the target. `all_sessions` (the
    session split, which pools DEV and TEST) counts every given session."""
    social, by_lesson = 0, {}
    for d in data.values():
        if d.session in S.TEST_SESSIONS and not all_sessions:
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

    def __init__(self, fold, data: dict, k: int, learned: bool = True, scheme: str = 'mix', split: str = 'date'):
        self.fold, self.k, self.scheme = fold, k, scheme
        self.train = [data[s] for s in fold.train]
        self.test = [data[s] for s in fold.test]
        self.stats = LY.fit_global_stats([d.tokens for d in self.train])
        self.scaled = {d.session: LY.scale(d.tokens, self.stats, scheme=scheme) for d in self.train + self.test}
        self.views = {s: LY.pooled(t) for s, t in self.scaled.items()}
        self._online = None
        self.at = {'train': _offsets(self.train), 'test': _offsets(self.test)}
        self.y = np.concatenate([d.target for d in self.train])
        sessions = np.concatenate([np.full(len(d), d.session, dtype=object) for d in self.train])
        sizes = {d.session: d.n_coded for d in self.train}
        # the rule and zero-shot Jev choose nothing, so they need no inner folds (nor two coded dates)
        if split == 'session':
            # the session split: every training session is its own group, held out alone in the inner folds
            self.units = {s: s for s in fold.train}
            self.inner = S.session_inner_folds(fold.train, sizes=sizes) if learned else []
        elif split in ('unit', 'forward'):
            # a unit (a date with its same_class_as dates) is one group, held out alone in the inner folds;
            # a TEST session is an ordinary one
            links = {d.session: d.same_class_as for d in self.train}
            self.units = S.class_units(fold.train, links)
            self.inner = S.unit_inner_folds(fold.train, sizes=sizes, same_class=links) if learned else []
        else:
            # a date (with its same_class_as dates) is one group, as in the outer folds
            links = {d.session: d.same_class_as for d in self.train}
            self.units = S.class_units(fold.train, links)
            self.inner = S.inner_folds(fold.train, k=INNER_FOLDS, sizes=sizes, same_class=links) if learned else []
        self.groups = np.concatenate([np.full(len(d), self.units[d.session], dtype=object) for d in self.train])
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
            self._online = {d.session: LY.scale(d.tokens, self.stats, mode='causal', scheme=self.scheme)
                            for d in self.test}
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

    def smooth(self, p, prior, A, gamma, mode, side: str = 'test') -> np.ndarray:
        """the held-out sessions' posteriors through the HMM, each session on its own (`side`
        'train': the training sessions' inner out-of-fold ones, for the selectors)."""
        flats = self.flat(side)
        return np.vstack([H.smooth(part, prior, A, gamma, flats[session], mode)
                          for session, part in self.by_session(p, side).items()])


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


def _decide(p, prior, k: int, tau: float | None = None, infold: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """the balanced decision on the posteriors, and the binary one on p_social + p_collaborative
    under pi_social + pi_collaborative; -1 where there is no posterior. With `tau` (the three-class
    target's in-fold social threshold) the three-class label is the two-step decision instead
    (tabular.two_step_decision), whose binary label is the same; with `infold` 'infold-all' it is
    the one-step decision (tabular.one_step_decision), and the binary label is read off it."""
    if tau is not None and k == 3 and infold == 'infold-all':
        y_pred = TB.one_step_decision(p, prior, tau)
        # interaction is social or collaborative, so this binary label can differ from the balanced one
        return y_pred, np.where(y_pred >= 0, (y_pred > 0).astype(int), -1)
    y_pred = TB.two_step_decision(p, prior, tau) if tau is not None and k == 3 else TB.balanced_decision(p, prior)
    if k == 2:
        return y_pred, y_pred.copy()
    return y_pred, TB.balanced_decision(p[:, 1] + p[:, 2], prior)


def _finish(fd: _Fold, hmm_modes, model: str, temporal: str, oof, logits, calibrate: bool, details: dict,
            results: dict, online=None, inner: dict | None = None, infold: str | None = None):
    """one model's answers through calibration (fit on the inner out-of-fold logits), each HMM mode
    (gamma chosen on the same out-of-fold answers) and the decisions; one result per mode.
    `online` is the held-out logits from the online tokens (_Fold.online), which the forward
    filter reads; without them (Jev, whose answers do not depend on scaling) it reads `logits`.
    `inner` (when a selector runs) receives, per variant of SELECTED_HMM, the training side's
    calibrated out-of-fold posteriors through the same HMM (the same gamma) and their decisions
    under the outer-training prior; nothing else changes with it. `infold` (social_decision
    'infold', three-class target) decides each mode's held-out and inner answers by the two-step
    decision with the social threshold chosen on those inner answers (tabular.social_threshold), or
    by the balanced decision where they hold too few social windows, and keeps the choice in
    details['social_tau'] by HMM mode; 'infold-all' does the same with the one-step decision and
    its threshold on p_social (tabular.social_threshold_all)."""
    k = fd.k
    infold = infold if k == 3 else None
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
        p_inner = p_oof
        selected = inner is not None and mode in SELECTED_HMM
        if mode != 'none':
            gamma, scores = H.select_gamma(fd.by_session(p_oof, 'train'), fd.labels(), fd.prior, fd.A,
                                           fd.flat('train'), mode=mode)
            details.setdefault('gamma', {})[mode] = {'gamma': gamma, 'nll': scores}
            record['p'] = fd.smooth(p_online if mode == 'filter' else p_test, fd.prior, fd.A, gamma, mode)
            if mode == 'fb':
                record['viterbi'] = fd.smooth(p_test, fd.prior, fd.A, gamma, 'viterbi').argmax(axis=1)
                if k == 3:
                    record['binary_hmm'] = _binary_hmm(fd, p_oof, p_test)
            if selected or infold:
                # a window no inner fold held out keeps no answer, though the smoother would carry one to it
                answered = np.isfinite(p_oof).all(axis=1)
                p_inner = np.where(answered[:, None], fd.smooth(p_oof, fd.prior, fd.A, gamma, mode, 'train'), np.nan)
        tau = None
        if infold:
            # the variant's own inner answers choose its social threshold (None: too few social windows)
            chosen = (TB.social_threshold_all(p_inner, fd.y) if infold == 'infold-all'
                      else TB.social_threshold(p_inner, fd.y, fd.prior))
            details.setdefault('social_tau', {})[mode] = chosen
            tau = chosen['tau']
        record['y_pred'], record['y_pred_binary'] = _decide(record['p'], fd.prior, k, tau, infold)
        results[_key(model, temporal, mode)] = record
        if selected:
            y_pred, y_binary = _decide(p_inner, fd.prior, k, tau, infold)
            inner[_key(model, temporal, mode)] = {'p': p_inner, 'y_pred': y_pred, 'y_pred_binary': y_binary}


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
        blocks = LY.block_columns(X_train)
        # an ablated block has no expert, and neither has one present in no coded training window (the
        # content block of tables without its columns): it is all unobserved, and a prior-only expert would
        # hand the stacker the inner folds' priors as a feature
        removed = plan.get('ablated') or ()
        coded = TB._labels(fd.y) >= 0
        blocks = {modality: columns for modality, columns in blocks.items()
                  if modality not in removed and (TB.PRESENCE[modality](X_train) & coded).any()}
        fusion = TB.LateFusion(make, grid, blocks, inner=fd.pairs).fit(X_train, fd.y, fd.groups)
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
        coefficients.insert(2, 'ablation', plan.get('ablation', 'full'))
        extras.setdefault('coefficients', []).append(coefficients)
    online = TB.log_proba(fitted, X_online) if X_online is not None else None
    return oof, TB.log_proba(fitted, X_test), True, online


def _soft_labels(d: SessionData, coder: str, k: int) -> np.ndarray:
    """a session's labels from another coder over its windows, in the target's form (binary for k
    2), NaN where they gave none."""
    y = d.others.get(coder)
    if y is None:
        return np.full(len(d), np.nan)
    return L.to_binary(y) if k == 2 else np.asarray(y, dtype=float)


def _vetoed(truth, other) -> tuple[np.ndarray, int]:
    """the second coder's labels for lr-soft with every window the truth coder called unclear or
    absent taken out (NaN), and how many of the second coder's classes that took out: the second
    coder's class counts only where the truth coder gave a class too or gave no code at all, so a
    window the truth coder judged unclear, or without two members at the table, never trains."""
    truth, other = np.asarray(truth, dtype=float), np.asarray(other, dtype=float)
    with np.errstate(invalid='ignore'):
        vetoed = np.isfinite(truth) & (truth < 0) & L.scored(other)
    return np.where(vetoed, np.nan, other), int(vetoed.sum())


def _soft(fd: _Fold, plan: dict, temporal: str, details: dict):
    """(inner out-of-fold logits, held-out logits, whether to calibrate, held-out logits from the
    online tokens or None) of lr-soft: lr on the same view, grid, inner splits and selection score
    (the class-weighted NLL against the truth), every fit on the soft labels of the truth and of
    plan['soft_coder'] over the training windows (tabular.fit_soft, the classes balanced over the
    soft weights), less the windows the truth coder called unclear or absent (_vetoed). The second
    coder's labels of a held-out session are never read."""
    second = plan['soft_coder']
    X_train, X_test = fd.X('train', temporal), fd.X('test', temporal)
    X_online = fd.X_online(temporal) if _online_wanted(plan, temporal) else None
    other, vetoed = _vetoed(fd.y, np.concatenate([_soft_labels(d, second, fd.k) for d in fd.train]))
    make = functools.partial(TB.make_lr, class_weight=None)
    params, oof = TB.select_soft(make, plan['lr_grid'], X_train, fd.y, [other], fd.groups, fd.pairs)
    fitted = TB.fit_soft(make, params, X_train, fd.y, [other])
    truth, given = L.scored(fd.y), L.scored(other)
    details.update(params=params, soft={'coder': second, 'both': int((truth & given).sum()),
                                        'truth_only': int((truth & ~given).sum()),
                                        'second_only': int((~truth & given).sum()), 'vetoed': vetoed})
    online = TB.log_proba(fitted, X_online) if X_online is not None else None
    return oof, TB.log_proba(fitted, X_test), True, online


def _pmil(fd: _Fold, plan: dict, details: dict):
    """(inner out-of-fold logits, held-out logits, whether to calibrate, held-out logits from the
    online tokens or None) of pmil-lr: the window rows of the fold's scaled tokens
    (pmil.window_rows), with C from lr's grid chosen on the inner splits as lr's is; its bias per
    group size goes into the fold's details."""
    from openmmla.analytics.interaction import pmil as PM
    X_train = np.vstack([PM.window_rows(fd.scaled[d.session]) for d in fd.train])
    X_test = np.vstack([PM.window_rows(fd.scaled[d.session]) for d in fd.test])
    params, oof = TB.select(PM.make_pmil, plan['lr_grid'], X_train, fd.y, fd.groups, fd.pairs)
    fitted = TB.fit_model(PM.make_pmil, params, X_train, fd.y)
    details['params'] = params
    if isinstance(fitted, PM.PairMIL):
        details['group_bias'] = fitted.group_bias
    online = None
    if _online_wanted(plan, 'T0'):
        online = TB.log_proba(fitted, np.vstack([PM.window_rows(fd.online()[d.session]) for d in fd.test]))
    return oof, TB.log_proba(fitted, X_test), True, online


def _network(fd: _Fold, plan: dict, model: str, details: dict):
    """(inner out-of-fold logits, held-out logits, held-out logits from the online tokens or None)
    of a network rung: E* from the inner folds (seed 0, patience), then the seed ensemble on every
    outer-training session. The out-of-fold logits are each inner split's at its own best epoch,
    or with plan['net_oof'] 'common' every split's at the refit's epoch count (network.oof_at)."""
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
    device = plan.get('device') or 'cpu'
    keep = {} if plan.get('net_oof') == 'common' else None
    epochs, oof = N.select_epochs(train, [(inner.train, inner.test) for inner in fd.inner], make=make,
                                  max_epochs=plan['max_epochs'], patience=plan['patience'], seed=0, device=device,
                                  keep=keep)
    used = plan['epochs'] or epochs
    if keep is not None:
        # every inner split's answers at the refit's epoch count, rather than at its own best epoch
        oof, at, replayed = N.oof_at(train, keep, used, make=make, seed=0, device=device)
        details.update(oof_epoch=at, oof_replayed=replayed)
    models = N.fit_ensemble(train, used, seeds=tuple(range(plan['seeds'])), make=make, device=device)
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


def _inner_scores(record: dict, y_train) -> dict:
    """the pooled scores select_headline reads, here of one variant's inner out-of-fold decisions on
    the coded training windows it answered, counted as score_variant counts them on held-out ones:
    macro-F1 (the three classes, or the two of a binary target) and binary macro-F1, None where
    undefined."""
    y_pred, y_binary = np.asarray(record['y_pred']), np.asarray(record['y_pred_binary'])
    y = np.where(y_pred >= 0, _labels_int(y_train), -1)
    coded = y >= 0
    return {'n': int(coded.sum()), 'macro_f1': M._float(M.macro_f1(y[coded], y_pred[coded])),
            'binary_macro_f1': M._float(M.macro_f1(_binary_truth(y)[coded], y_binary[coded], n_classes=2))}


def _binary_log_loss(p, truth, rows) -> float | None:
    """the class-weighted binary log-loss of p_interaction (p_social + p_collaborative, or the
    interaction column of a binary target) on `rows`: each class weighs as much as the other
    (tabular.class_weighted_nll); None when no row counts."""
    p = np.asarray(p, dtype=float)
    interaction = np.clip(p[:, 1:].sum(axis=1), M.FLOOR, 1.0 - M.FLOOR)
    loss = TB.class_weighted_nll(np.log(np.column_stack([1.0 - interaction, interaction])),
                                 np.where(rows, truth, -1))
    return M._float(loss)


def select_inner(selector: str, inner: dict, fd: _Fold, models, infold: str | None = None) -> dict:
    """the variant a selector chooses in one outer fold from the calibrated inner out-of-fold
    answers of the training side (`inner`, from _finish: per variant of the HMM modes none and fb,
    the posteriors and the decisions under the outer-training prior), with each candidate's score.
    The caller copies the winner's held-out answers. `infold` says that those decisions are the
    two-step ones with each candidate's in-fold social threshold (social_decision 'infold'), or the
    one-step ones with its in-fold threshold on p_social ('infold-all'), which selected_on then names.

    rule22sep, the pre-declared headline rule: select_headline itself, over the late-lr and late-hgb
    variants (every temporal mode run, HMM none and fb), on their pooled inner out-of-fold binary
    macro-F1 under the binary target and three-class macro-F1 under the three-class one, the first
    in run order on a tie. Under the three-class target it may choose otherwise than the date
    split's headline: there the absent-class policy (too few social windows, or lessons with them)
    makes select_headline rank by binary macro-F1, while rule22sep keeps the target's own score, as
    declared for it. Known optimism, to be disclosed: the grid points, the calibrator,
    the stacker and gamma were chosen on the same inner out-of-fold answers the rule then scores
    (twelve grid points for HGB against five for LR), with no inner level below them, for cost; and
    the fold's decisions use the outer-training prior.

    select-all: over every variant of the named models that has such answers (r1, lr, hgb, late-lr,
    late-hgb, the networks, jev-cal and the EXPLORATORY candidates; HMM none and fb), the lowest
    class-weighted binary log-loss of p_interaction on the coded training windows every candidate
    answered, the first in run order on a tie. The floors, the rule and zero-shot Jev keep no
    out-of-fold answer and are no candidate."""
    names = class_names(fd.k)
    if selector == 'rule22sep':
        candidates = {key: record for key, record in inner.items() if key.split(':')[0] in HEADLINE_MODELS}
        scores = {key: _inner_scores(record, fd.y) for key, record in candidates.items()}
        variants = {key: {'model': key.split(':')[0], 'hmm': key.split(':')[2]} for key in candidates}
        reports = {key: {'macro_f1_ci': {'estimate': score['macro_f1']},
                         'binary_macro_f1_ci': {'estimate': score['binary_macro_f1']}} for key, score in scores.items()}
        winner = select_headline(variants, reports, binary=fd.k == 2)
        how = 'binary macro-F1' if fd.k == 2 else f"macro-F1 over {', '.join(names)}"
        if infold == 'infold-all' and fd.k == 3:
            how += (", each candidate decided by the one-step decision with its in-fold threshold on p_social, chosen "
                    "on the same inner answers")
        elif infold and fd.k == 3:
            how += (", each candidate decided by the two-step decision with its in-fold social threshold, chosen on "
                    "the same inner answers")
        return {'winner': winner, 'candidates': list(candidates), 'scores': scores,
                'selected_on': 'select_headline on the calibrated inner out-of-fold answers pooled over the inner '
                               'folds, ' + how}
    panel = [model for model in models if model in INNER_MODELS]
    candidates = {key: record for key, record in inner.items() if key.split(':')[0] in panel}
    truth = _binary_truth(_labels_int(fd.y))
    rows = truth >= 0
    for record in candidates.values():
        rows &= np.isfinite(np.asarray(record['p'], dtype=float)).all(axis=1)
    losses = {key: _binary_log_loss(record['p'], truth, rows) for key, record in candidates.items()}
    winner = None
    for key, loss in losses.items():
        if loss is not None and (winner is None or loss < losses[winner]):
            winner = key
    return {'winner': winner, 'candidates': list(candidates), 'windows': int(rows.sum()), 'log_loss': losses,
            'selected_on': 'the lowest class-weighted binary log-loss of the calibrated inner out-of-fold answers, on '
                           'the coded training windows every candidate answered'}


def run_fold(fold, data: dict, plan: dict) -> dict:
    """everything one outer fold gives: per variant (model:temporal:hmm) the held-out rows'
    posteriors and decisions, and what was chosen on the way (parameters, calibrators, gammas,
    epochs)."""
    started = time.time()
    k = plan['k']
    fd = _Fold(fold, data, k, learned=any(model not in UNLEARNED for model in plan['models']),
               scheme=plan.get('scaling', 'mix'), split=plan.get('split', 'date'))
    results, details, extras = {}, {}, {}
    # the selectors read the training side's out-of-fold answers, kept only when one runs
    inner = {} if any(model in SELECTORS for model in plan['models']) else None
    # the in-fold social threshold (social_decision 'infold' or 'infold-all'), for the three-class target only
    infold = plan.get('social_decision') if plan.get('social_decision') in IN_FOLD_DECISIONS and k == 3 else None
    n_test = sum(len(d) for d in fd.test)
    for model in plan['models']:
        if model in SELECTORS:
            continue
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
                _finish(fd, plan['hmm'], model, temporal, oof, logits, calibrate, where, results, online, inner,
                        infold=infold)
        elif model == 'lr-soft':
            for temporal in plan['temporal']:
                where = details.setdefault(f'{model}:{temporal}', {})
                oof, logits, calibrate, online = _soft(fd, plan, temporal, where)
                _finish(fd, plan['hmm'], model, temporal, oof, logits, calibrate, where, results, online, inner,
                        infold=infold)
        elif model == 'pmil-lr':
            # the pairs come from the tokens, with no lag columns: T0 only
            where = details.setdefault(f'{model}:T0', {})
            oof, logits, calibrate, online = _pmil(fd, plan, where)
            _finish(fd, plan['hmm'], model, 'T0', oof, logits, calibrate, where, results, online, inner,
                    infold=infold)
        elif model in NEURAL:
            temporal = NET_TEMPORAL[model]
            where = details.setdefault(f'{model}:{temporal}', {})
            oof, logits, online = _network(fd, plan, model, where)
            _finish(fd, plan['hmm'], model, temporal, oof, logits, True, where, results, online, inner,
                    infold=infold)
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
            _finish(fd, plan['hmm'], 'jev-cal', plan['jev_variant'], _jev_rows(fd.train), held, True, where, results,
                    inner=inner, infold=infold)
            for key in [key for key in results if key.startswith('jev-cal:')]:
                _unanswered(results[key], ~np.isfinite(held).all(axis=1))
    for model in [model for model in plan['models'] if model in SELECTORS]:
        choice = select_inner(model, inner, fd, plan['models'], infold=infold)
        details[model] = choice
        if choice.get('winner') is not None:
            results[_key(model, NESTED, NESTED)] = dict(results[choice['winner']])
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


def score_variant(variant: dict, base: pd.DataFrame, k: int, n_boot: int, unit: str = 'date') -> tuple[dict, list]:
    """every metric of 4.4 for one variant over the pooled held-out windows it answered
    (metrics.report), the unit-bootstrap interval of its pooled macro-F1 (three-class and
    binary; left out, with the reason, below MIN_BOOTSTRAP_UNITS units), the onset latency of an online variant, the seed mean of the stratified floor, the
    2-state HMM check and its coverage; and its per-session rows. A coded window the variant gave
    no answer for is left out rather than counted as a miss, and `coverage` says how many. `unit`
    names the bootstrap's unit in the reason an interval is left out ('date', or 'session')."""
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
    report['macro_f1_ci'] = _interval(lambda rows: M.macro_f1(yc[rows], pc[rows]), gc, n_boot, unit)
    report['binary_macro_f1_ci'] = _interval(
        lambda rows: M.macro_f1(truth[rows], binary[rows], n_classes=2), gc, n_boot, unit)
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


def _too_few_units(units: int, unit: str = 'date') -> str | None:
    """why a bootstrap over this many units is left out, or None when it is not; `unit` names what
    a unit is: 'date' (the date and test splits) or 'session' (the session split)."""
    if units >= MIN_BOOTSTRAP_UNITS:
        return None
    return (f"{units} unit(s) ({unit}s) to resample, fewer than {MIN_BOOTSTRAP_UNITS}: "
            f"the percentiles would be those {unit}s' own scores")


def _interval(fn, groups, n_boot: int, unit: str = 'date') -> dict:
    """the unit-bootstrap interval of a pooled metric (metrics.session_bootstrap) with the number
    of units it resampled; with fewer than MIN_BOOTSTRAP_UNITS the estimate stands alone, lo and
    hi are None and `left_out` says why (_too_few_units, naming the `unit`)."""
    units = len(pd.unique(np.asarray(groups, dtype=object)))
    why = _too_few_units(units, unit)
    if why is None:
        return dict(_drop_samples(M.session_bootstrap(fn, groups, n=n_boot)), units=units)
    estimate = float(fn(np.arange(len(groups))))
    return {'estimate': estimate if np.isfinite(estimate) else None, 'lo': None, 'hi': None, 'n': 0,
            'n_undefined': 0, 'units': units, 'left_out': why}


# ---- the session split: every held-out session scored on its own ----

def _session_metrics(y, p, y_pred, y_binary, k: int, min_support: int) -> dict:
    """the scores of one set of rows (a held-out session, or every held-out window pooled), with
    `y` the truth on the coded windows the variant answered and -1 elsewhere. Macro-F1, three-class
    and binary, averages the classes with at least `min_support` true windows; a class's F1 is None
    where it has no true window, and so is the binary F1 without an interaction window; AUROC
    (p_interaction against interaction) needs both sides of the binary target; Brier (multi-class)
    and NLL need posteriors, which the rule does not give."""
    y, y_pred, y_binary = np.asarray(y), np.asarray(y_pred), np.asarray(y_binary)
    names = class_names(k)
    coded = y >= 0
    truth = _binary_truth(y)
    f1, support = M.f1_per_class(y, y_pred, k)
    out = {'n': int(coded.sum()), 'classes': int((support >= max(min_support, 1)).sum()),
           'macro_f1': M._float(M.macro_f1(y, y_pred, min_support, k)),
           'accuracy': float((y_pred[coded] == y[coded]).mean()) if coded.any() else None,
           'balanced_accuracy': M._float(M.balanced_accuracy(y, y_pred, k)),
           'kappa': M._float(M.kappa(y, y_pred, k))}
    out.update({f'f1_{name}': M._float(f1[c]) if support[c] > 0 else None for c, name in enumerate(names)})
    # for the binary target the interaction class's F1 is the binary F1 itself
    out.update({'f1_interaction': M._float(M.f1_per_class(truth, y_binary, 2)[0][1]) if (truth == 1).any() else None,
                'binary_macro_f1': M._float(M.macro_f1(truth, y_binary, min_support, 2)),
                'auroc': None, 'brier': None, 'nll': None})
    out.update({f'support_{name}': int(support[c]) for c, name in enumerate(names)})
    if p is not None:
        p = np.asarray(p, dtype=float)
        score = p[:, 1:].sum(1)
        keep = coded & np.isfinite(score)
        if len(np.unique(truth[keep])) == 2:
            from sklearn.metrics import roc_auc_score
            out['auroc'] = M._float(roc_auc_score(truth[keep], score[keep]))
        out['brier'], out['nll'] = M._float(M.brier(y, p)), M._float(M.nll(y, p))
    return out


def session_metric_names(k: int) -> list:
    """the per-session scores the session split averages: SESSION_METRICS and each class's F1."""
    return list(dict.fromkeys(list(SESSION_METRICS) + [f'f1_{name}' for name in class_names(k)]))


def session_scores(variant: dict, base: pd.DataFrame, k: int, by: str = 'session') -> pd.DataFrame:
    """the session split's per_session.csv rows of one variant: per held-out session with coded
    windows, its scores on the coded windows the variant answered (_session_metrics, macro-F1 over
    the classes with at least metrics.MIN_SUPPORT true windows there), the coder's class-coded
    windows and the coverage, and the flags: few_windows (fewer than MIN_SESSION_WINDOWS scored
    windows), single_class (fewer than two classes with metrics.MIN_SUPPORT true windows, so its
    macro-F1 is one class's F1) and flagged (either). `by` 'lesson' gives the same per held-out unit
    (the unit and forward splits' per_unit.csv), in a 'unit' column, its tasks joined with '+'."""
    truth_all = _labels_int(base['y_true'])
    y_pred, y_binary = np.asarray(variant['y_pred']), np.asarray(variant['y_pred_binary'])
    y = np.where(y_pred >= 0, truth_all, -1)
    sessions, tasks = base[by].to_numpy(), base['task'].to_numpy(dtype=object)
    name = 'session' if by == 'session' else 'unit'
    p = variant['p']
    rows = []
    for session in pd.unique(sessions):
        at = sessions == session
        n_coded = int((truth_all[at] >= 0).sum())
        if not n_coded:
            continue
        scores = _session_metrics(y[at], None if p is None else p[at], y_pred[at], y_binary[at], k, M.MIN_SUPPORT)
        few, single = scores['n'] < MIN_SESSION_WINDOWS, scores['classes'] < 2
        task = tasks[at][0] if by == 'session' else '+'.join(sorted({str(t) for t in tasks[at] if t}))
        rows.append({name: session, 'task': task, 'n': scores['n'], 'n_coded': n_coded,
                     'coverage': scores['n'] / n_coded, 'classes': scores['classes'], 'few_windows': few,
                     'single_class': single, 'flagged': few or single, **scores})
    return pd.DataFrame(rows)


def _spread(values) -> tuple:
    """(mean, sample SD with ddof 1, count) of the finite values; None where there are too few."""
    finite = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return (float(finite.mean()) if len(finite) else None, float(finite.std(ddof=1)) if len(finite) > 1 else None,
            int(len(finite)))


def session_summary(rows: pd.DataFrame, variant: dict, base: pd.DataFrame, k: int, level: str = 'session') -> dict:
    """the session split's figures of one variant (metrics.json `across_sessions`): per score, the
    mean and sample SD over the held-out sessions where it is defined and how many they are, the
    same without the flagged sessions, and the pooled value over every held-out window the variant
    answered (macro-F1 there over every class present, as the pooled report counts it). Every
    held-out session enters the mean of each score it defines, flagged or not: a single-class
    session's macro-F1 is that class's F1, and a session without both sides of the binary target
    has no AUROC and stays out of its mean. `level` 'unit' reads session_scores' rows by unit
    (`across_units`, n_units and units)."""
    truth_all = _labels_int(base['y_true'])
    y_pred = np.asarray(variant['y_pred'])
    pooled = _session_metrics(np.where(y_pred >= 0, truth_all, -1), variant['p'], y_pred,
                              np.asarray(variant['y_pred_binary']), k, 1)
    flagged = rows['flagged'].to_numpy(dtype=bool) if len(rows) else np.zeros(0, dtype=bool)
    out = {f'n_{level}s': int(len(rows)), f'{level}s': list(rows[level]) if len(rows) else [],
           'flagged': sorted(rows.loc[flagged, level]) if len(rows) else [],
           'pooled_windows': pooled['n'], 'metrics': {}}
    for name in session_metric_names(k):
        values = list(rows[name]) if name in rows else []
        mean, sd, n = _spread(values)
        kept = [value for value, flag in zip(values, flagged) if not flag]
        mean_kept, sd_kept, n_kept = _spread(kept)
        out['metrics'][name] = {'mean': mean, 'sd': sd, 'n': n, 'mean_unflagged': mean_kept,
                                'sd_unflagged': sd_kept, 'n_unflagged': n_kept, 'pooled': pooled.get(name)}
    return out


# the session split's columns of results.csv: per score its mean and SD over the sessions
SESSION_RESULT_METRICS = ('macro_f1', 'accuracy', 'kappa', 'f1_interaction', 'auroc', 'brier', 'nll')


def _session_columns(summary: dict, k: int, level: str = 'session') -> dict:
    """results.csv's extra columns in the session split: the number of sessions, each score's mean
    and SD over them, and the pooled accuracy and Brier (the other pooled scores are already in
    the row); `level` 'unit' the same over the units (the unit and forward splits)."""
    scores = summary['metrics']
    out = {f'{level}s': summary[f'n_{level}s'], f'{level}s_flagged': len(summary['flagged'])}
    for name in list(SESSION_RESULT_METRICS) + [f'f1_{name}' for name in class_names(k)]:
        if name in scores and f'{name}_mean' not in out:
            out[f'{name}_mean'], out[f'{name}_sd'] = scores[name]['mean'], scores[name]['sd']
    out['accuracy'], out['brier'] = scores['accuracy']['pooled'], scores['brier']['pooled']
    return out


def _result_row(key: str, variant: dict, report: dict) -> dict:
    """one row of the results table (4.7)."""
    pooled = report['pooled']
    temporal = pooled.get('temporal') or {}
    ci = report['macro_f1_ci']
    group = 'online' if variant['hmm'] == 'filter' else GROUP_OF[variant['model']]
    row = {'group': group, 'variant': key, 'model': variant['model'], 'temporal': variant['temporal'],
           'hmm': variant['hmm'], 'ablation': variant.get('ablation', 'full'), 'n': pooled['n'],
           'coverage': (report.get('coverage') or {}).get('share'),
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


ABLATION_COLUMNS = ('ablation', 'removed', 'variant', 'model', 'temporal', 'hmm', 'group', 'n', 'coverage',
                    'macro_f1', 'macro_f1_lo', 'macro_f1_hi', 'kappa',
                    'binary_macro_f1', 'binary_macro_f1_lo', 'binary_macro_f1_hi',
                    'binary_kappa', 'binary_kappa_lo', 'binary_kappa_hi',
                    'delta_macro_f1', 'delta_macro_f1_lo', 'delta_macro_f1_hi',
                    'delta_binary_macro_f1', 'delta_binary_macro_f1_lo', 'delta_binary_macro_f1_hi',
                    'units', 'left_out')


def ablation_frame(variants: dict, arm_of: dict, reports: dict, base: pd.DataFrame, n_boot: int,
                   arms: dict = MODALITY_ARMS, unit: str = 'date') -> pd.DataFrame:
    """ablation.csv: per model variant (the floors and Jev left out) and arm, the pooled macro-F1
    (three-class) and binary macro-F1 with their unit-bootstrap intervals as score_variant gives
    them, Cohen's kappa and the binary kappa (with its interval on the same resamples), and the
    change against the same variant's full arm, paired on the same unit resamples. Exploratory: no
    Holm correction. Below MIN_BOOTSTRAP_UNITS units the intervals are left out, `left_out` says
    why, naming the `unit` ('date', or 'session'). `arm_of` maps a variant key to (arm, the full
    arm's key); a variant's arms sit together, in the full arm's order, then the arms' order."""
    truth_all = _labels_int(base['y_true'])
    lessons_all = base['lesson'].to_numpy()

    def rows_of(key):
        # the coded windows the variant answered: what score_variant scores it on
        return (truth_all >= 0) & (np.asarray(variants[key]['y_pred']) >= 0)

    def scorer(key, rows, binary):
        truth = (_binary_truth(truth_all) if binary else truth_all)[rows]
        pred = np.asarray(variants[key]['y_pred_binary' if binary else 'y_pred'])[rows]
        return lambda sample: M.macro_f1(truth[sample], pred[sample], n_classes=2 if binary else 3)

    by_base = {}
    for key, (arm, base_key) in arm_of.items():
        if variants[key]['model'] not in FEATURELESS:
            by_base.setdefault(base_key, {})[arm] = key
    order = list(arms)
    out = []
    for base_key, keys in by_base.items():
        full = keys.get('full')
        for arm in sorted(keys, key=lambda name: order.index(name) if name in order else len(order)):
            key = keys[arm]
            variant, report = variants[key], reports[key]
            pooled = report['pooled']
            ci, binary_ci = report['macro_f1_ci'], report['binary_macro_f1_ci']
            rows = rows_of(key)
            truth, binary = _binary_truth(truth_all)[rows], np.asarray(variant['y_pred_binary'])[rows]
            kappa_ci = _interval(lambda sample: M.kappa(truth[sample], binary[sample], n_classes=2),
                                 lessons_all[rows], n_boot, unit)
            row = {'ablation': arm, 'removed': '+'.join(arms.get(arm, ())), 'variant': base_key,
                   'model': variant['model'], 'temporal': variant['temporal'], 'hmm': variant['hmm'],
                   'group': 'online' if variant['hmm'] == 'filter' else GROUP_OF[variant['model']],
                   'n': pooled['n'], 'coverage': (report.get('coverage') or {}).get('share'),
                   'macro_f1': pooled['macro_f1'], 'macro_f1_lo': ci['lo'], 'macro_f1_hi': ci['hi'],
                   'kappa': pooled['kappa'], 'binary_macro_f1': binary_ci['estimate'],
                   'binary_macro_f1_lo': binary_ci['lo'], 'binary_macro_f1_hi': binary_ci['hi'],
                   'binary_kappa': kappa_ci['estimate'], 'binary_kappa_lo': kappa_ci['lo'],
                   'binary_kappa_hi': kappa_ci['hi'], 'units': ci.get('units'), 'left_out': ci.get('left_out')}
            if arm != 'full' and full is not None:
                paired = rows_of(full)
                if np.array_equal(paired, rows):
                    few = _too_few_units(len(pd.unique(np.asarray(lessons_all[rows], dtype=object))), unit)
                    for name, binary_scale in (('delta_macro_f1', False), ('delta_binary_macro_f1', True)):
                        a, b = scorer(key, rows, binary_scale), scorer(full, rows, binary_scale)
                        if few:
                            everything = np.arange(int(rows.sum()))
                            delta = a(everything) - b(everything)
                            row[name] = float(delta) if np.isfinite(delta) else None
                        else:
                            paired_delta = M.paired_delta(a, b, lessons_all[rows], n=n_boot)
                            row[name], row[f'{name}_lo'], row[f'{name}_hi'] = \
                                paired_delta['delta'], paired_delta['lo'], paired_delta['hi']
                else:
                    row['left_out'] = '; '.join(filter(None, [row['left_out'], "no paired change: the arm and the "
                                                              "full arm answered different windows"]))
            out.append(row)
    return pd.DataFrame(out, columns=list(ABLATION_COLUMNS))


def inter_coder(data: dict, primary: str | None) -> dict:
    """Cohen's kappa of every other (human) coder against the primary one, pooled over the windows
    both coded in the given sessions (3-class and binary): the ceiling the results are read
    against. run() passes the sessions it scores and their truth coder."""
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


AGREEMENT_COLUMNS = ('session', 'task', 'windows_a', 'windows_b', 'both_coded', 'windows', 'percent_agreement',
                     'kappa', 'percent_agreement_binary', 'kappa_binary', 'percent_agreement_codes', 'kappa_codes',
                     'unclear_a', 'unclear_b', 'absent_a', 'absent_b')


def _agreement_row(a: pd.Series, b: pd.Series) -> dict:
    """two coders' joined labels (aligned Series) as one row of agreement.csv: labels.agreement's
    kappas with the percent agreement beside each (three classes on the windows both gave a class,
    the derived binary on the same windows, the five codes on every window both coded) and the
    windows each coded."""
    score = L.agreement(a, b)
    binary = np.asarray(score['confusion_binary'])
    return {'windows_a': int(np.isfinite(a.to_numpy(dtype=float)).sum()),
            'windows_b': int(np.isfinite(b.to_numpy(dtype=float)).sum()),
            'both_coded': score['both_coded'], 'windows': score['windows'],
            'percent_agreement': M._float(score['accuracy']), 'kappa': M._float(score['kappa']),
            'percent_agreement_binary': float(np.trace(binary) / binary.sum()) if binary.sum() else None,
            'kappa_binary': M._float(score['kappa_binary']),
            'percent_agreement_codes': M._float(score['accuracy_codes']), 'kappa_codes': M._float(score['kappa_codes']),
            'unclear_a': score['unclear_a'], 'unclear_b': score['unclear_b'], 'absent_a': score['absent_a'],
            'absent_b': score['absent_b'], 'confusion': score['confusion'], 'confusion_binary': score['confusion_binary'],
            'confusion_codes': score['confusion_codes']}


def coder_agreement(artifacts, coders, sessions: str | None = None, join: str = 'exact', out=None, log=None) -> Path:
    """two coders against each other, training nothing (mmla ses-classify --agreement A,B): over
    every session with a fused table, DEV and TEST alike, that the inclusion rule S1 keeps, each
    coder's own labels file (labels/<coder>.jsonl, the last line of a window winning; adjudicated.jsonl
    is not read) is joined to the table's grid by the classifier's join (`join`), and Cohen's kappa
    and the percent agreement are computed on the windows both labelled, per session and pooled
    over the sessions both coded. Writes agreement.csv (a row per session and a pooled row) and
    agreement.json (the same with the confusion matrices, each coder's coded windows per session,
    and the sessions left out and why) to `out` (default artifacts/_analysis/interaction/
    agreement-<time>) and returns that folder. A session whose labels do not join is left out
    with every failing coder's join message, not fatal; each coder's coverage counts every session
    their own labels joined, whatever the other coder's did. A coder with no labels in a session
    where a file differs from their name only in case is told so (case_hint)."""
    say = log or (lambda message: None)
    first, second = coders
    if first == second:
        raise ValueError("name two different coders")
    artifacts = Path(artifacts).resolve()
    found = session_tables(artifacts, sessions)
    if not found:
        raise FileNotFoundError(f"no fused table under {artifacts} for {sessions or 'any session'}: run mmla ses-fuse")
    directories = [path.parents[2] for _, path in found]
    for coder in coders:
        require_coder(coder, directories)
    rows, joined, excluded, coverage = [], {first: [], second: []}, {}, {first: {}, second: {}}
    for session, path in found:
        directory = path.parents[2]
        table = LY.read_table(path)
        included, reason = LY.session_inclusion(table, LY.session_roster(table, directory))
        if not included:
            excluded[session] = reason
            say(f"{session} left out: {reason}")
            continue
        everyone = L.load_labels(directory, all_coders=True)
        # each coder is joined on their own, so one coder's failure never hides the other's coverage
        labels, errors, hints = {}, [], []
        for coder in coders:
            mine = everyone[everyone['coder'] == coder] if len(everyone) else everyone
            if not len(mine):
                hint = case_hint(directory, coder)
                if hint:
                    hints.append(hint)
                continue
            try:
                labels[coder] = L.join_labels(table, mine, mode=join)[0]
            except L.LabelJoinError as error:
                errors.append(f"labels/{coder}.jsonl does not join the fused table: {error}")
                continue
            coverage[coder][session] = int(np.isfinite(labels[coder].to_numpy(dtype=float)).sum())
        if len(labels) < 2:
            head = errors or [f"only {next(iter(labels))} coded it" if labels else "neither coder coded it"]
            excluded[session] = '; '.join(head + hints)
            continue
        a, b = labels[first], labels[second]
        row = _agreement_row(a, b)
        if not row['both_coded']:
            excluded[session] = 'no window both coded'
            continue
        rows.append({'session': session, 'task': S.task_of(session), **row})
        for coder, series in ((first, a), (second, b)):
            joined[coder].append(series.set_axis(pd.MultiIndex.from_arrays([[session] * len(series), series.index])))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = Path(out) if out else artifacts / '_analysis' / 'interaction' / f'agreement-{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    pooled = {'session': 'pooled', 'task': None, **_agreement_row(pd.concat(joined[first]), pd.concat(joined[second]))} \
        if rows else None
    table = pd.DataFrame(rows + ([pooled] if pooled else []), columns=list(AGREEMENT_COLUMNS))
    table.to_csv(run_dir / 'agreement.csv', index=False)
    write_json(run_dir / 'agreement.json', {
        'run': run_dir.name, 'created_at': datetime.now(timezone.utc).isoformat(), 'coders': [first, second],
        'join': join, 'all_sessions': True,
        'labels': "each coder's own labels/<coder>.jsonl (the last line of a window wins, a null label undoes it); "
                  "adjudicated.jsonl and other coders' files are not read",
        'windows': "kappa and percent_agreement: the windows both gave a class (individual, social, collaborative); "
                   "_binary: the same windows as individual against interaction; _codes: every window both coded, "
                   "over the five codes (the three classes, unclear, absent)",
        'sessions': {row['session']: row for row in rows}, 'pooled': pooled, 'excluded': excluded,
        'coverage': coverage,
        'files': {s: {'table': str(path)} for s, path in found}})
    say(f"{len(rows)} session(s) both coded, {len(excluded)} left out -> {run_dir}")
    return run_dir


def predictions_frame(base: pd.DataFrame, variants: dict, k: int) -> pd.DataFrame:
    """predictions.csv: one row per held-out window and variant, in the columns of 4.7 (with the
    temporal mode, the HMM mode and the Viterbi state after them)."""
    frames = []
    for key, variant in variants.items():
        frame = base.drop(columns=['block']).copy()
        frame['model'] = variant['model']
        frame['variant'] = f"{variant['temporal']}:{variant['hmm']}"
        frame['ablation'] = variant.get('ablation', 'full')
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


def confusion_frame(reports: dict, k: int, arms: dict | None = None) -> pd.DataFrame:
    """confusion.csv: the pooled confusion of every variant, one row per (true, predicted) cell,
    with the variant's ablation arm (`arms`, key -> arm; 'full' when a key has none)."""
    names = class_names(k)
    rows = []
    for key, report in reports.items():
        matrix = report['pooled']['confusion']
        arm = (arms or {}).get(key, 'full')
        for i, row in enumerate(matrix[:k]):
            for j, count in enumerate(row[:k]):
                rows.append({'variant': key, 'ablation': arm, 'true': names[i], 'predicted': names[j],
                             'windows': int(count)})
    return pd.DataFrame(rows, columns=['variant', 'ablation', 'true', 'predicted', 'windows'])


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


def _session_split_record(coder: str, folds: list, excluded: dict, soft_coder: str | None = None) -> dict:
    """what config.json and metrics.json say of a session split, so no one takes it for the frozen
    TEST scoring: every session the coder labelled is a fold, DEV and TEST alike, the sessions left
    out with why, whose labels and how the per-session figures are read (with lr-soft's second
    coder, `soft_coder`, among the labels trained on)."""
    sessions = [session for fold in folds for session in fold.test]
    truth = (f"labels/{coder}.jsonl alone: adjudicated.jsonl is not read for the truth, and the other coders' "
             f"files only for the inter-coder ceiling")
    if soft_coder:
        truth = (f"labels/{coder}.jsonl alone: adjudicated.jsonl is not read for the truth; lr-soft also trains on "
                 f"labels/{soft_coder}.jsonl of each fold's training sessions (soft labels, never a held-out "
                 f"session's), and the other coders' files serve only the inter-coder ceiling")
    return {
        'split': 'session', 'all_sessions': True, 'held_out_test': False,
        'note': "leave one session out over every session the coder labelled, DEV and TEST alike (for when the "
                "TEST sessions alone cannot stand for the model): cross-validated "
                "estimates with no untouched hold-out, not the frozen TEST evaluation (split test)",
        'coder': coder, 'truth': truth, 'sessions': sessions, 'tasks': {s: S.task_of(s) for s in sessions},
        'excluded': {s: record.get('reason') for s, record in excluded.items()},
        'inner_folds': 'leave one session out within the training sessions',
        'across_sessions': {
            'n_sessions': len(sessions), 'sessions': sessions,
            'excluded': {s: record.get('reason') for s, record in excluded.items()},
            'macro_f1_per_session': f"the classes with at least {M.MIN_SUPPORT} true windows in the session",
            'flags': {'few_windows': f"fewer than {MIN_SESSION_WINDOWS} scored windows",
                      'single_class': f"fewer than two classes with {M.MIN_SUPPORT} true windows (its macro-F1 is "
                                      f"one class's F1)"},
            'mean': "every held-out session enters the mean and SD (ddof 1) of each score it defines, flagged or "
                    "not; *_unflagged repeat them without the flagged sessions; AUROC needs both sides of the binary "
                    "target, a class's F1 a true window of the class, Brier and NLL posteriors",
            'pooled': "every held-out window the variant answered, macro-F1 over every class present"}}


def split_units(data: dict) -> dict:
    """unit -> its sessions, sorted, in unit order: what the unit and forward splits hold out and
    resample whole (SessionData.lesson after link_lessons)."""
    units = {}
    for session, d in data.items():
        units.setdefault(d.lesson, []).append(session)
    return {unit: sorted(units[unit]) for unit in sorted(units)}


def _unit_split_record(split: str, coder: str, folds: list, excluded: dict, data: dict,
                       soft_coder: str | None = None) -> dict:
    """what config.json and metrics.json say of a unit or forward split, as _session_split_record
    does of the session split, with the units (a date with the dates its same_class_as links reach)
    and the mean over the held-out units beside the mean over the sessions."""
    record = _session_split_record(coder, folds, excluded, soft_coder)
    held = {data[s].lesson for fold in folds for s in fold.test}
    units = split_units(data)
    if split == 'unit':
        note = ("leave one unit out over every session the coder labelled, DEV and TEST alike: "
                "a unit is a recording date with every date its sessions' same_class_as links reach, so the same "
                "pupils are never on both sides; cross-validated estimates with no untouched hold-out, not the frozen "
                "TEST evaluation (split test)")
    else:
        note = (f"forward chaining over every session the coder labelled, DEV and TEST alike: "
                f"each date with at least {S.MIN_EARLIER_DATES} earlier dates with coded windows is held out and the "
                f"models train on the earlier dates only (a training date of the held-out date's unit stays out); "
                f"descriptive, not the frozen TEST evaluation (split test)")
    record.update(split=split, note=note, units=units,
                  inner_folds='leave one unit (a date with its same_class_as dates) out within the training sessions')
    if split == 'forward':
        record['min_earlier_dates'] = S.MIN_EARLIER_DATES
    record['across_units'] = dict(
        record['across_sessions'], n_units=len(held), units=sorted(held),
        macro_f1_per_unit=f"the classes with at least {M.MIN_SUPPORT} true windows in the unit",
        mean="every held-out unit enters the mean and SD (ddof 1) of each score it defines, flagged or not; "
             "*_unflagged repeat them without the flagged units")
    for key in ('n_sessions', 'sessions', 'macro_f1_per_session'):
        record['across_units'].pop(key, None)
    return record


def _config_record(cfg: Config, plan: dict, data: dict, folds: list, artifacts: Path, run_dir: Path,
                   coder: str | None = None, session_record: dict | None = None, not_read: dict | None = None,
                   content: dict | None = None) -> dict:
    """config.json: what the run read (every fused table and label file by sha256), whose labels
    were the truth and how that coder was chosen, with what (feature lists, grids, seeds, the
    layout version) and which software, secrets masked. A session split adds `session_record`
    (_session_split_record) at the top. A run with a features root adds `features_root`: the folder,
    and the sessions with a table under artifacts/ but none there, each with why (`not_read`,
    features_root_gaps). `content` gives, per session, whether its table has the content columns
    and in how many of its windows the block was observed. The decision mode is config.social_decision;
    an infold or infold-all run adds `social_decision` with what the mode does and its tau grid."""
    from openmmla.utils.session_provenance import git_commit, redact_secrets, software_info
    values = LY.GROUP_VALUES + LY.PERSON_VALUES + LY.PAIR_VALUES + LY.CONTENT_VALUES
    record = {
        'run': run_dir.name, 'created_at': datetime.now(timezone.utc).isoformat(), 'config': asdict(cfg),
        'coder': coder, 'coder_chosen_by': '--coder' if cfg.coder else 'the most windows over the DEV sessions',
        'test_coder': (cfg.test_coder or coder) if cfg.split == 'test' else None,
        'models_run': list(plan['models']),
        'ablation': {'grid': cfg.ablate,
                     'arms': {arm: list(removed) for arm, removed in ABLATIONS[cfg.ablate].items()}},
        'scaling': cfg.scaling,
        # the decision mode is in 'config' in every run; an in-fold run also says what it does, while a balanced
        # run's config.json keeps the keys it had before the option
        **({'social_decision': _social_decision_setting(cfg.social_decision)}
           if cfg.social_decision != 'balanced' else {}),
        'layout_version': LY.LAYOUT_VERSION,
        'features': {'pooled_columns': list(LY.POOLED_COLUMNS),
                     'pooled_blocks': {name: list(columns) for name, columns in LY.POOLED_BLOCKS.items()},
                     'lag_columns': list(LY.LAG_COLUMNS),
                     'lags': {mode: [suffix for suffix, _, _ in lags] for mode, lags in LY.LAGS.items()},
                     # not called 'tokens': redact_secrets masks any key that names a token
                     'window_layout': {'G': list(LY.G_COLUMNS), 'P': list(LY.P_COLUMNS), 'Q': list(LY.Q_COLUMNS),
                                       'C': list(LY.C_COLUMNS), 'availability': list(LY.AVAILABILITY)},
                     'values': {v.name: {'source': v.source, 'transform': v.transform, 'scale': v.tag, 'mask': v.mask,
                                         'modality': v.modality} for v in values},
                     'fusion_blocks': list(LY.FUSION_BLOCKS),
                     'content': {'columns': list(LY.CONTENT_COLUMNS), 'sessions': dict(content or {})},
                     'dropped': dict(LY.DROPPED)},
        'rule': {'version': TB.RULE_VERSION, 'fixed': TB.RULES[TB.RULE_VERSION]['fixed'],
                 'columns': {role: list(names) for role, names in TB.RULES[TB.RULE_VERSION]['columns'].items()},
                 'thresholds': dict(TB.RULES[TB.RULE_VERSION]['thresholds'])},
        'rule_thresholds': dict(TB.RULE_THRESHOLDS),
        'presence_gate': dict(P.RULE),
        'grids': {'lr': plan['lr_grid'], 'hgb': plan['hgb_grid']},
        'network': {'max_epochs': plan['max_epochs'], 'patience': plan['patience'],
                    'seeds': list(range(plan['seeds'])), 'small': plan['small'], 'epochs': plan['epochs'],
                    'device': plan['device'], 'torch': plan.get('torch')},
        'hmm': {'gammas': list(H.GAMMAS), 'alpha': 1.0, 'diagonal': 10.0,
                'filter_inputs': f"{', '.join(CAUSAL_INPUTS)}; the held-out sessions scaled by the running normaliser"},
        'min_class_windows': cfg.min_class_windows, 'bootstrap': plan['bootstrap'], 'inner_folds': INNER_FOLDS,
        'folds': [{'name': fold.name, 'train': list(fold.train), 'test': list(fold.test)} for fold in folds],
        'test_sessions': list(S.TEST_SESSIONS),
        'files': {s: {'table': str(d.table_path), 'table_sha256': d.table_sha256, 'labels': d.label_files}
                  for s, d in data.items()},
        'software': software_info(artifacts.parent), 'git_commit': git_commit(artifacts.parent),
    }
    exploratory = [m for m in plan['models'] if m in EXPLORATORY]
    if exploratory:
        record['exploratory'] = _exploratory_record(exploratory, plan)
    if session_record:
        extra = {key: value for key, value in session_record.items() if key not in ('across_sessions', 'across_units')}
        # the split's own record first, after the run's name; its coder and inner folds replace the defaults
        record = {**{key: record[key] for key in ('run', 'created_at')}, **extra,
                  **{key: value for key, value in record.items() if key not in extra}}
    if cfg.features_root is not None:
        record['features_root'] = {'folder': str(Path(cfg.features_root).resolve()), 'not_read': dict(not_read or {})}
    return redact_secrets(_jsonable(record))


def _social_decision_setting(mode: str) -> dict:
    """what config.json says of the three-class hard label: the mode, what it does and, for
    'infold' and 'infold-all', the grid tau is chosen from and the fewest social windows it needs."""
    out = {'mode': mode, 'note': SOCIAL_DECISION_NOTES[mode]}
    if mode in IN_FOLD_DECISIONS:
        taus = TB.SOCIAL_ALL_TAUS if mode == 'infold-all' else TB.SOCIAL_TAUS
        out.update(taus=list(taus), min_social_windows=TB.MIN_SOCIAL_WINDOWS)
    return out


def social_decision_record(outputs: dict, decision: str = 'infold') -> dict:
    """metrics.json's social_decision in an infold or infold-all run (`decision`), from every arm's fold
    outputs ({arm: fold outputs}): per variant (an ablated arm's ending in ':<arm>') and outer fold,
    the threshold chosen (tau, or tau_all for 'infold-all'; None where the fold kept the balanced
    decision, `fallback` saying why), a selector's being its winner's. Each fold's full record (tau,
    its inner social F1, the windows and social windows it was chosen on) is under the fold's
    models, `social_tau` by HMM mode."""
    taus, fallbacks = {}, {}
    for arm, folds in outputs.items():
        suffix = '' if arm == 'full' else f':{arm}'
        for out in folds:
            for name, details in out['models'].items():
                for mode, chosen in (details.get('social_tau') or {}).items():
                    key = f'{name}:{mode}{suffix}'
                    taus.setdefault(key, {})[out['name']] = chosen['tau']
                    if chosen.get('fallback'):
                        fallbacks.setdefault(key, {})[out['name']] = chosen['fallback']
        for out in folds:
            for selector in SELECTORS:
                winner = (out['models'].get(selector) or {}).get('winner')
                if winner is None:
                    continue
                key = _key(selector, NESTED, NESTED) + suffix
                taus.setdefault(key, {})[out['name']] = taus.get(winner + suffix, {}).get(out['name'])
                why = fallbacks.get(winner + suffix, {}).get(out['name'])
                if why:
                    fallbacks.setdefault(key, {})[out['name']] = why
    return dict(_social_decision_setting(decision), tau=taus, fallback=fallbacks)


def _exploratory_record(models, plan: dict) -> dict:
    """what config.json says of the exploratory candidates a run fits: each one's note, pmil-lr's
    instance columns, lr-soft's second coder and net-attn's configurations."""
    out = {'models': {model: EXPLORATORY_NOTES[model] for model in models}}
    if 'pmil-lr' in models:
        from openmmla.analytics.interaction import pmil as PM
        out['pmil-lr'] = {'instance_columns': list(PM.INSTANCE_COLUMNS), 'grid': plan['lr_grid']}
    if 'lr-soft' in models:
        out['lr-soft'] = {'second_coder': plan.get('soft_coder'), 'grid': plan['lr_grid']}
    if 'net-attn' in models:
        from openmmla.analytics.interaction import network as N
        out['net-attn'] = {'default': dict(N.ATTENTION), 'small': dict(N.ATTENTION_SMALL)}
    return out


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
            'test_coder': cfg.test_coder or cfg.coder, 'confirm_frozen': bool(cfg.confirm_frozen),
            'ablate': cfg.ablate, 'scaling': cfg.scaling}
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(line) + '\n')
    return earlier


def ledger_path(artifacts) -> Path:
    """the run ledger: artifacts/_analysis/interaction/session_runs.jsonl."""
    return Path(artifacts) / '_analysis' / 'interaction' / SESSION_RUNS


def _git_state() -> dict:
    """the commit of the checkout this code runs from, in full, and the files under openmmla/ it
    has changed since (None for either when it is no checkout or git is missing)."""
    import subprocess
    folder = Path(__file__).resolve().parents[3]

    def git(*args):
        try:
            done = subprocess.run(['git', '-C', str(folder), *args], capture_output=True, text=True, timeout=5,
                                  check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    commit = git('rev-parse', 'HEAD')
    changed = git('status', '--porcelain', '--', 'openmmla')
    return {'commit': commit.strip() if commit else None,
            'changed': [line[3:] for line in changed.splitlines() if line.strip()] if changed is not None else None}


def _output_digests(run_dir: Path) -> dict:
    """every file of a run folder (not its subfolders) by sha256."""
    from openmmla.utils.session_provenance import file_digest
    return {path.name: file_digest(path) for path in sorted(Path(run_dir).iterdir()) if path.is_file()}


def _append_ledger(artifacts, line: dict) -> None:
    path = ledger_path(artifacts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(_jsonable(line), ensure_ascii=False) + '\n')


def _record_session_run(artifacts: Path, run_dir: Path, cfg: Config, status: str, data: dict, coder: str) -> None:
    """the run ledger of the session, unit and forward splits, in _record_test_run's
    way: artifacts/_analysis/interaction/session_runs.jsonl gets a line when a run starts and one
    when it finishes, with the code's commit (and the files under openmmla/ changed since), the
    run's settings, every fused table and labels file it read by sha256 and, when it finishes, every
    file of the run folder by sha256."""
    from openmmla.utils.session_provenance import redact_secrets
    line = {'run': run_dir.name, 'run_dir': str(Path(run_dir).resolve()), 'status': status,
            'at': datetime.now(timezone.utc).isoformat(), 'split': cfg.split, 'coder': coder, 'target': cfg.target,
            'models': list(cfg.models), 'ablate': cfg.ablate, 'scaling': cfg.scaling, 'net_oof': cfg.net_oof,
            'social_decision': cfg.social_decision,
            'features_root': cfg.features_root, 'quick': bool(cfg.quick), 'git': _git_state(),
            'config': redact_secrets(_jsonable(asdict(cfg))),
            'tables': {s: d.table_sha256 for s, d in data.items()},
            'labels': {s: d.label_files for s, d in data.items()}}
    if status == 'finished':
        line['outputs'] = _output_digests(run_dir)
    _append_ledger(artifacts, line)


def log_run_folders(artifacts, run_dirs, note: str | None = None, status: str = 'looked_at') -> list:
    """add run folders made before the ledger (or by hand) to session_runs.jsonl, each as one line
    with `status` (looked at: their results have been seen) and retro true: the commit, settings
    and tables their config.json records, and every file in the folder by sha256. A folder already
    in the ledger with that status is not added again. Returns (folder, what was done) per folder;
    a folder with no config.json and no metrics.json is skipped as no run folder."""
    path = ledger_path(artifacts)
    logged = set()
    if path.exists():
        for text in path.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if record.get('status') == status:
                logged.add(str(Path(record.get('run_dir') or '').resolve()))
    done = []
    for folder in run_dirs:
        folder = Path(folder).resolve()
        config_file, metrics_file = folder / 'config.json', folder / 'metrics.json'
        if not folder.is_dir() or not (config_file.exists() or metrics_file.exists()):
            done.append((str(folder), 'skipped: no config.json or metrics.json there'))
            continue
        if str(folder) in logged:
            done.append((str(folder), f'already in the ledger as {status}'))
            continue
        config, metrics = {}, {}
        for name, holder in ((config_file, config), (metrics_file, metrics)):
            try:
                holder.update(json.loads(name.read_text(encoding='utf-8')) if name.exists() else {})
            except (OSError, ValueError):
                pass
        settings = config.get('config') or {}
        line = {'run': folder.name, 'run_dir': str(folder), 'status': status, 'retro': True,
                'at': datetime.now(timezone.utc).isoformat(), 'created_at': config.get('created_at'),
                'split': settings.get('split') or metrics.get('split'), 'coder': config.get('coder'),
                'target': settings.get('target') or metrics.get('target'), 'models': settings.get('models'),
                'ablate': settings.get('ablate'), 'scaling': settings.get('scaling'), 'quick': settings.get('quick'),
                'git': {'commit': config.get('git_commit'), 'changed': None, 'from': 'config.json (short)'},
                'config': settings,
                'tables': {s: entry.get('table_sha256') for s, entry in (config.get('files') or {}).items()},
                'labels': {s: entry.get('labels') for s, entry in (config.get('files') or {}).items()},
                'outputs': _output_digests(folder), 'note': note}
        _append_ledger(artifacts, line)
        logged.add(str(folder))
        done.append((str(folder), f'added as {status}'))
    return done


# ---- the run ----

def planned_variants(models, temporal=TEMPORAL, hmm=HMM_MODES, jev_variant: str = 'j0') -> dict:
    """model -> the variant keys (model:temporal:hmm) a run with these flags gives it, before
    anything is loaded: r0, the floors and zero-shot Jev give one whatever the flags say, r1 reads
    the T0 view only, a network its own temporal mode, and the forward filter goes only to causal
    inputs. A model with an empty list would be trained on every fold for nothing. pmil-lr reads the
    tokens' pairs, T0 only, and lr-soft the temporal modes as lr does. A selector gives
    model:nested:nested when the run fits a candidate for it (rule22sep: late-lr or late-hgb;
    select-all: a model of INNER_MODELS) with HMM none or fb."""
    out = {}
    named = list(dict.fromkeys(models))
    chosen = [h for h in dict.fromkeys(hmm) if h in SELECTED_HMM]
    for model in named:
        if model in SELECTORS:
            pool = HEADLINE_MODELS if model == 'rule22sep' else INNER_MODELS
            out[model] = [_key(model, NESTED, NESTED)] if chosen and any(m in pool for m in named) else []
            continue
        if model in ('r0', 'r0-v1', 'majority', 'stratified'):
            out[model] = [_key(model, 'T0', 'none')]
            continue
        if model == 'jev':
            out[model] = [_key(model, jev_variant, 'none')]
            continue
        modes = [jev_variant] if model == 'jev-cal' else [NET_TEMPORAL[model]] if model in NEURAL \
            else ['T0'] if model in ('r1', 'pmil-lr') else list(dict.fromkeys(temporal))
        out[model] = [_key(model, t, h) for t in modes for h in dict.fromkeys(hmm)
                      if h != 'filter' or t in CAUSAL_INPUTS]
    return out


def _barren(cfg: Config) -> list:
    """the named models the flags give no variant."""
    planned = planned_variants(cfg.models, cfg.temporal, cfg.hmm, cfg.jev_variant)
    return [model for model, keys in planned.items() if not keys]


def _selector_gap(cfg: Config) -> str | None:
    """why a named selector has nothing to choose from, or None."""
    chosen = [h for h in cfg.hmm if h in SELECTED_HMM]
    for selector in [m for m in dict.fromkeys(cfg.models) if m in SELECTORS]:
        pool = HEADLINE_MODELS if selector == 'rule22sep' else INNER_MODELS
        if not any(m in pool for m in cfg.models):
            return f"{selector} chooses among {', '.join(pool)}: name at least one of them too"
        if not chosen:
            return f"{selector} chooses among the HMM modes {' and '.join(SELECTED_HMM)}: add one to the HMM modes"
    return None


def _check(cfg: Config):
    named = MODELS + SELECTORS + EXPLORATORY
    unknown = [m for m in cfg.models if m not in named]
    if unknown:
        raise ValueError(f"unknown model(s) {', '.join(unknown)}: one of {', '.join(named)}")
    gap = _selector_gap(cfg)
    if gap:
        raise ValueError(gap)
    if not cfg.models:
        raise ValueError("name at least one model")
    if cfg.split not in SPLITS:
        hint = " ('loso' became 'date': the same group on one date is never on both sides; 'session' leaves one " \
               "session out over DEV and TEST alike)" if cfg.split == 'loso' else ' (task transfer is deferred)'
        raise ValueError(f"split {cfg.split!r} is not built: one of {', '.join(SPLITS)}{hint}")
    if cfg.target not in TARGETS:
        raise ValueError(f"target must be one of {', '.join(TARGETS)}")
    binary_only = [m for m in dict.fromkeys(cfg.models) if m in BINARY_ONLY]
    if binary_only and cfg.target != 'binary':
        raise ValueError(f"{', '.join(binary_only)} answer(s) p(interaction) only (a noisy-OR over the pairs has no "
                         f"social against collaborative): run it with the binary target")
    if cfg.soft_coder is not None and 'lr-soft' not in cfg.models:
        raise ValueError("soft_coder is lr-soft's second coder: it goes with the model lr-soft")
    bad = [t for t in cfg.temporal if t not in TEMPORAL] + [h for h in cfg.hmm if h not in HMM_MODES]
    if bad or not cfg.temporal or not cfg.hmm:
        raise ValueError(f"temporal modes are {', '.join(TEMPORAL)} and HMM modes {', '.join(HMM_MODES)}")
    if cfg.split == 'test' and not cfg.confirm_frozen:
        raise ValueError("the TEST sessions are scored once, after every choice is frozen: confirm with confirm_frozen")
    if cfg.split == 'test' and not cfg.coder:
        raise ValueError("the TEST scoring names its truth coder: pass the coder the date runs were chosen on "
                         "(their config.json 'coder')")
    if cfg.split in POOLED_SPLITS and not cfg.coder:
        raise ValueError(f"the {cfg.split} split reads one coder's labels at a time: name the coder (its labels file "
                         f"name, e.g. 'alex' for labels/alex.jsonl)")
    if cfg.net_oof not in NET_OOF:
        raise ValueError(f"net_oof must be one of {', '.join(NET_OOF)}, not {cfg.net_oof!r}")
    if cfg.social_decision not in SOCIAL_DECISIONS:
        raise ValueError(f"social_decision must be one of {', '.join(SOCIAL_DECISIONS)}, not {cfg.social_decision!r}")
    if cfg.social_decision == 'infold' and cfg.target != '3class':
        raise ValueError("social_decision 'infold' chooses social against collaborative inside each fold: it goes "
                         "with the three-class target")
    if cfg.social_decision == 'infold-all' and cfg.target != '3class':
        raise ValueError("social_decision 'infold-all' chooses a threshold on p_social inside each fold: it goes "
                         "with the three-class target")
    if cfg.features_root is not None and not Path(cfg.features_root).is_dir():
        raise FileNotFoundError(f"the features root {cfg.features_root} is no folder")
    if cfg.test_coder and cfg.split != 'test':
        raise ValueError("test_coder is the truth of the TEST sessions: it goes with split 'test' only")
    barren = _barren(cfg)
    if barren:
        raise ValueError(f"{', '.join(barren)} give(s) no variant with temporal {','.join(cfg.temporal)} and HMM "
                         f"{','.join(cfg.hmm)}: the forward filter runs only for inputs that never read a later "
                         f"window ({', '.join(CAUSAL_INPUTS[:2])}, net-notcn, Jev)")
    if any(m in NEURAL for m in cfg.models) and importlib.util.find_spec('torch') is None:
        raise ModuleNotFoundError("the network variants need torch: pip install torch")
    if cfg.device not in ('cpu', 'cuda', 'auto'):
        raise ValueError(f"device must be cpu, cuda or auto, not {cfg.device!r}")
    if cfg.ablate not in ABLATIONS:
        raise ValueError(f"ablation {cfg.ablate!r} is not built: one of {', '.join(ABLATIONS)} (the temporal, fusion, "
                         f"ladder and weights grids of 4.6 are deferred)")
    if cfg.ablate != 'none' and all(m in FEATURELESS for m in cfg.models):
        raise ValueError(f"the {cfg.ablate} ablation needs a model that reads features: "
                         f"{', '.join(dict.fromkeys(cfg.models))} run in the full arm only")
    if cfg.scaling not in LY.SCHEMES:
        raise ValueError(f"scaling must be one of {', '.join(LY.SCHEMES)}, not {cfg.scaling!r}")


def _device(cfg: Config) -> tuple:
    """(the device the networks train on, as the plan carries it, and torch's version with its CUDA
    build, or None): cpu whenever no network runs, so a tabular run never imports torch; 'cuda' is
    refused here, before anything is read, when torch sees no GPU."""
    if not any(m in NEURAL for m in cfg.models):
        return 'cpu', None
    import torch
    from openmmla.analytics.interaction import network as N
    device = N.resolve_device(cfg.device)
    # the GPU's name would open a CUDA context in this process, which only hands folds out
    return device.type, torch.__version__ + (f", CUDA {torch.version.cuda}" if device.type == 'cuda' else '')


def _plan(cfg: Config) -> dict:
    """the settings every fold needs, as a plain dict a worker process receives."""
    quick = QUICK if cfg.quick else {}
    jobs = max(1, int(cfg.jobs))
    device, build = _device(cfg)
    return {'k': 3 if cfg.target == '3class' else 2, 'models': list(dict.fromkeys(cfg.models)),
            'temporal': list(dict.fromkeys(cfg.temporal)), 'hmm': list(dict.fromkeys(cfg.hmm)),
            'lr_grid': quick.get('lr_grid', TB.LR_GRID), 'hgb_grid': quick.get('hgb_grid', TB.HGB_GRID),
            'max_epochs': quick.get('max_epochs', 300), 'patience': quick.get('patience', 25),
            'seeds': int(cfg.seeds), 'small': cfg.small, 'epochs': cfg.epochs, 'jev_variant': cfg.jev_variant,
            'bootstrap': cfg.bootstrap or quick.get('bootstrap', BOOTSTRAP), 'jobs': jobs, 'scaling': cfg.scaling,
            'split': cfg.split, 'net_oof': cfg.net_oof, 'social_decision': cfg.social_decision,
            'threads': max(1, (os.cpu_count() or 1) // jobs) if jobs > 1 else None,
            'device': device, 'torch': build}


def _run_folds(folds: list, data: dict, plan: dict, say) -> list:
    if plan['jobs'] == 1 or len(folds) == 1:
        outputs = []
        for n, fold in enumerate(folds, start=1):
            outputs.append(_fold_job(fold, data, plan))
            say(f"fold {n}/{len(folds)} {fold.name}: {outputs[-1]['seconds']} s")
        return outputs
    from joblib import Parallel, delayed
    say(f"{len(folds)} folds on {plan['jobs']} workers" + (" sharing the GPU" if plan.get('device') == 'cuda' else ''))
    # each worker gets only the sessions its fold reads. loky starts every worker as a fresh
    # interpreter, never a fork of this one, so on cuda each worker opens its own CUDA context
    # (several hundred MB of GPU memory each, nearly all of it the context) and the folds share
    # the GPU
    return Parallel(n_jobs=plan['jobs'], backend='loky')(
        delayed(_fold_job)(fold, {s: data[s] for s in list(fold.train) + list(fold.test)}, plan) for fold in folds)


def _run_arms(folds: list, arm_data: dict, plans: dict, say) -> dict:
    """every (arm, fold) of the modality ablation, the arms in their order and the folds in theirs:
    one after another, or all of them on one pool of `jobs` workers (a test split has one fold, so
    looping _run_folds per arm would run the arms one by one). Returns {arm: fold outputs in fold
    order}."""
    jobs = [(arm, n, fold) for arm in plans for n, fold in enumerate(folds, start=1)]
    outputs = {arm: [] for arm in plans}
    settings = next(iter(plans.values()))
    if settings['jobs'] == 1 or len(jobs) == 1:
        for arm, n, fold in jobs:
            outputs[arm].append(_fold_job(fold, arm_data[arm], plans[arm]))
            say(f"{arm} fold {n}/{len(folds)} {fold.name}: {outputs[arm][-1]['seconds']} s")
        return outputs
    from joblib import Parallel, delayed
    say(f"{len(plans)} arms x {len(folds)} folds on {settings['jobs']} workers"
        + (" sharing the GPU" if settings.get('device') == 'cuda' else ''))
    # as in _run_folds: each job gets only the sessions its fold reads
    done = Parallel(n_jobs=settings['jobs'], backend='loky')(
        delayed(_fold_job)(fold, {s: arm_data[arm][s] for s in list(fold.train) + list(fold.test)}, plans[arm])
        for arm, _, fold in jobs)
    for (arm, _, _), out in zip(jobs, done):
        outputs[arm].append(out)
    return outputs


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
    found = session_tables(artifacts, cfg.sessions, cfg.features_root)
    if cfg.split == 'date':
        # a dev run never opens a TEST session, not even its labels
        found = [(s, p) for s, p in found if s not in S.TEST_SESSIONS]
    if not found:
        raise FileNotFoundError(f"no fused table under {cfg.features_root or artifacts} for "
                                f"{cfg.sessions or 'any session'}: run mmla ses-fuse")
    # the session's folder (labels, manifest, Jev maps): where its table sits, or under artifacts/ when the
    # tables come from another root
    folder = {s: (artifacts / s if cfg.features_root is not None else path.parents[2]) for s, path in found}
    not_read = None
    if cfg.features_root is not None:
        not_read = features_root_gaps(artifacts, cfg, found)
        say(f"fused tables from {cfg.features_root}: {len(found)} session(s)")
        for session, why in not_read.items():
            say(f"{session}: no table under the features root ({why})")
    by_session = cfg.split == 'session'
    # the session, unit and forward splits: every session the coder labelled, DEV and TEST alike
    pooled = cfg.split in POOLED_SPLITS
    if pooled:
        # one coder's own file is the truth, matched by its exact name
        require_coder(cfg.coder, [folder[s] for s, _ in found])
    # the truth coder is chosen on the DEV sessions in every split, so TEST labels never decide it
    coder = cfg.coder or primary_coder([folder[s] for s, path in found if s not in S.TEST_SESSIONS])
    # a test run may score TEST against another coder than the one that trains; DEV always reads `coder`
    test_coder = (cfg.test_coder or coder) if cfg.split == 'test' else coder
    models = L.model_names([folder[s] for s, _ in found])
    data, tables, excluded = {}, {}, {}
    for session, path in found:
        loaded, table = load_session(session, path, test_coder if session in S.TEST_SESSIONS else coder,
                                     cfg.join, cfg.target, models, adjudicated=not pooled,
                                     directory=folder[session] if cfg.features_root is not None else None)
        # S1, the inclusion rule: a session that never shows two persons together is left out whole
        included, reason = LY.session_inclusion(table, loaded.roster)
        if not included:
            excluded[session] = {**loaded.roster.record(), 'included': False, 'reason': reason}
            if pooled:
                excluded[session]['coded_windows'] = loaded.n_coded
            say(f"{session} left out: {reason}")
            continue
        if pooled and not loaded.n_coded:
            # the session split folds over the sessions the coder labelled; the others train nothing either
            hint = case_hint(loaded.directory, coder)
            reason = (f"no labels of coder {coder} ({hint or f'labels/{coder}.jsonl missing or empty'})"
                      if not loaded.join.get('own') else
                      f"coder {coder} gave no window of it a class (only unclear or absent)")
            excluded[session] = {**loaded.roster.record(), 'included': False, 'reason': reason, 'coded_windows': 0}
            say(f"{session} left out: {reason}")
            continue
        data[session], tables[session] = loaded, table
    if not data:
        if pooled:
            raise Refused(f"every session is left out (the inclusion rule S1, or no class coded by {coder}): "
                          + '; '.join(f"{s} ({r['reason']})" for s, r in excluded.items()), [], None)
        raise Refused("every session is left out by the inclusion rule S1: "
                      + '; '.join(f"{s} ({r['reason']})" for s, r in excluded.items()))
    link_lessons(data, by_session=by_session)
    # the transcript content (layout version 5): a table without its columns reads the block as unobserved
    content = {s: {'columns': LY.has_content(tables[s]), 'observed_windows': int(LY.content_observed(d.tokens).sum()),
                   'windows': len(d)} for s, d in data.items()}
    lacking = [s for s, c in content.items() if not c['columns']]
    if lacking:
        say(f"content block: {len(lacking)} of {len(data)} session table(s) have no content columns "
            f"({', '.join(LY.CONTENT_COLUMNS)}); the block is unobserved there")
    if cfg.split in ('unit', 'forward'):
        for unit, sessions in split_units(data).items():
            say(f"unit {unit}: {', '.join(sessions)}")
    if any(m in JEV_MODELS for m in plan['models']):
        for d in data.values():
            d.jev, d.jev_note = jev_log_proba(d, cfg.jev_variant)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = Path(cfg.out) if cfg.out else artifacts / '_analysis' / 'interaction' / f'{cfg.split}-{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    counts = label_counts(data)
    counts.to_csv(run_dir / 'label_counts.csv')
    say(f"labels of coder {coder or '(none)'}"
        + (f" (TEST sessions: coder {test_coder})" if test_coder != coder else '')
        + f", windows per class and session:\n{counts.to_string()}")
    write_json(run_dir / 'roster.json', {**{s: {**d.roster.record(), 'included': True} for s, d in data.items()}, **excluded})
    write_json(run_dir / 'data_checks.json', LY.data_checks(tables, {s: d.roster for s, d in data.items()}))

    try:
        folds, split_error = make_folds(cfg.split, data), None
    except S.SplitError as error:
        folds, split_error = [], str(error)
    notes = {s: d.jev_note for s, d in data.items() if d.jev_note}
    soft_gap = None
    if 'lr-soft' in plan['models']:
        # the second coder of lr-soft, from the sessions some fold trains on
        trained = {s for fold in folds for s in fold.train}
        plan['soft_coder'], soft_gap = soft_coder({s: data[s] for s in trained}, coder, cfg.soft_coder)
    dropped = jev_gaps(folds, data, plan['models'], cfg.jev_variant)
    if dropped:
        plan['models'] = [m for m in plan['models'] if m not in dropped]
        for model, reason in dropped.items():
            say(f"{model} left out: {reason}")
    learned = [m for m in plan['models'] if m not in UNLEARNED]
    problems = label_refusal(folds, data, k, cfg.min_class_windows) if learned else []
    held = sum(data[s].n_coded for fold in folds for s in fold.test)
    # the TEST truth coder must have coded every TEST session scored with labels of their own: a
    # misspelt or unfinished coder would otherwise be scored on adjudicated.jsonl alone and use up the look
    unmet = sorted(s for s in data if cfg.split == 'test' and s in S.TEST_SESSIONS and not data[s].join.get('own'))
    why = None
    if split_error:
        why = split_error
    elif soft_gap and folds:
        why = soft_gap
    elif not plan['models']:
        why = f"every model named was left out ({', '.join(dropped)}: no usable Jev maps, see left_out and notes)"
    elif not folds:
        why = 'no TEST session has a fused table' if cfg.split == 'test' else \
            f'no session has windows coded by {coder} to hold out' if by_session else \
            f'no unit has windows coded by {coder} to hold out' if cfg.split == 'unit' else \
            (f'no date with windows coded by {coder} has {S.MIN_EARLIER_DATES} earlier dates with coded windows '
             f'to train on') if cfg.split == 'forward' else \
            'no date has coded windows to hold out'
    elif unmet:
        why = (f"the TEST truth coder {test_coder!r} has no labels of their own in {', '.join(unmet)} "
               f"(labels/{test_coder}.jsonl missing or empty there): check the name, or wait until they coded it")
    elif not held:
        why = 'the held-out sessions have no coded window to score'
    elif problems:
        why = f"{len(problems)} outer-training fold(s) lack {cfg.min_class_windows} coded windows of a class " \
              f"(or two coded {'sessions' if by_session else 'units' if pooled else 'dates'} for the inner folds)"
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
    if pooled:
        _record_session_run(artifacts, run_dir, cfg, 'started', data, coder)
    if by_session:
        say(f"leave one session out over every session coder {coder} labelled, DEV and TEST alike "
            f"(inner folds: one training session out); {len(excluded)} session(s) left out")
    elif cfg.split == 'unit':
        say(f"leave one unit out over every session coder {coder} labelled, DEV and TEST alike "
            f"(inner folds: one training unit out); {len(excluded)} session(s) left out")
    elif cfg.split == 'forward':
        say(f"forward chaining over every session coder {coder} labelled, DEV and TEST alike: each date with "
            f"{S.MIN_EARLIER_DATES} earlier dates trains on those (inner folds: one training unit out); "
            f"{len(excluded)} session(s) left out")
        for fold in folds:
            say(f"fold {fold.name}: trains on {len({S.date_of(s) for s in fold.train})} earlier date(s), "
                f"{len(fold.train)} session(s)")
    if 'select-all' in plan['models'] and cfg.net_oof == 'best' and any(m in NEURAL for m in plan['models']):
        say("select-all compares the networks on their inner answers at each split's own best epoch: "
            "net_oof 'common' (--net-oof common) reads every split at the refit's epoch count instead")
    say(f"{len(folds)} fold(s) over {len(data)} session(s): {', '.join(plan['models'])}")
    arms = ABLATIONS[cfg.ablate]
    if len(arms) == 1:
        # the run without an ablation: the full arm alone, on the loaded data and the plan as they are
        outputs = {'full': _run_folds(folds, data, plan, say)}
        arm_data, plans = {'full': data}, {'full': plan}
    else:
        # an ablated arm: its modalities not run in any session, train and test alike; the full arm is
        # the loaded data and the plan untouched, so its numbers are those of a run without the ablation
        arm_data = {arm: data if not removed else {s: ablated(d, removed) for s, d in data.items()}
                    for arm, removed in arms.items()}
        plans = {arm: plan if not removed else dict(plan, ablation=arm, ablated=list(removed),
                                                     models=[m for m in plan['models'] if m not in FEATURELESS])
                 for arm, removed in arms.items()}
        alone = [m for m in plan['models'] if m in FEATURELESS]
        say(f"{cfg.ablate} ablation: {', '.join(arms)}; {', '.join(alone) or 'no model'} in the full arm only")
        outputs = _run_arms(folds, arm_data, plans, say)

    # the rows are the same in every arm: an ablation keeps the grid, the empty flag, the labels and the strata
    base = _base_rows(outputs['full'], data)
    variants, arm_of = {}, {}
    for arm in arms:
        for key, variant in _variants(outputs[arm], arm_data[arm], k).items():
            variant['ablation'] = arm
            merged = key if arm == 'full' else f'{key}:{arm}'
            variants[merged], arm_of[merged] = variant, (arm, key)
    if not variants:
        # _check refuses flags that give no variant; this guards the files below
        raise Refused("the folds gave no variant to score", [], run_dir)
    # the headline, the contrasts and the state shares read the full arm only, as pre-registered
    full = {key: variant for key, variant in variants.items() if variant['ablation'] == 'full'}
    # what the bootstrap resamples, as the reason an interval is left out names it
    unit = 'session' if by_session else 'unit' if cfg.split == 'unit' else 'date'
    reports, per_session, per_unit = {}, [], []
    for key, variant in variants.items():
        reports[key], rows = score_variant(variant, base, k, plan['bootstrap'], unit)
        if pooled:
            # each held-out session is scored on its own, with its flags and the mean and SD over them
            frame = session_scores(variant, base, k)
            reports[key]['across_sessions'] = session_summary(frame, variant, base, k)
            per_session += [{'variant': key, 'model': variant['model'], 'temporal': variant['temporal'],
                             'hmm': variant['hmm'], 'ablation': variant['ablation'], **row}
                            for row in frame.to_dict('records')]
            if not by_session:
                # and so is each held-out unit, the unit and forward splits' inferential unit
                frame = session_scores(variant, base, k, by='lesson')
                reports[key]['across_units'] = session_summary(frame, variant, base, k, level='unit')
                per_unit += [{'variant': key, 'model': variant['model'], 'temporal': variant['temporal'],
                              'hmm': variant['hmm'], 'ablation': variant['ablation'], **row}
                             for row in frame.to_dict('records')]
        else:
            per_session += [dict(row, variant=key, ablation=variant['ablation']) for row in rows]
    policy = absent_class_policy(data, all_sessions=pooled)
    binary_confirmatory = policy['confirmatory_target'] == 'binary' or k == 2
    headline = select_headline(full, reports, binary_confirmatory) if cfg.split == 'date' else None
    tests = contrasts(headline, full, base, plan['bootstrap'], binary_confirmatory) if headline else {}
    presence_gated = base['presence_gated'].to_numpy(dtype=bool)
    gate = P.evaluate_gate(base['y_true'].to_numpy(dtype=float), presence_gated, base['session'].to_numpy(),
                           base['n_observed'].to_numpy())
    share_key = share_variant(headline, full)
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
    if pooled and not by_session:
        pd.DataFrame(per_unit).to_csv(run_dir / 'per_unit.csv', index=False)
    confusion_frame(reports, k, {key: variant['ablation'] for key, variant in variants.items()}).to_csv(
        run_dir / 'confusion.csv', index=False)
    if cfg.ablate != 'none':
        ablation_frame(variants, arm_of, reports, base, plan['bootstrap'], arms, unit).to_csv(
            run_dir / 'ablation.csv', index=False)
    if by_session:
        results = pd.DataFrame([dict(_result_row(key, variants[key], reports[key]),
                                     **_session_columns(reports[key]['across_sessions'], k)) for key in variants])
    elif pooled:
        # the unit and forward splits: the mean and SD over the held-out units
        results = pd.DataFrame([dict(_result_row(key, variants[key], reports[key]),
                                     **_session_columns(reports[key]['across_units'], k, level='unit'))
                                for key in variants])
    else:
        results = pd.DataFrame([_result_row(key, variants[key], reports[key]) for key in variants])
    # the ceiling is read on the sessions this run scores, against their truth coder: in the session and unit
    # splits all of them, in the forward split the held-out dates only (the first dates only ever train)
    scored = {s for fold in folds for s in fold.test}
    ceiling = inter_coder({s: d for s, d in data.items() if s in scored} if pooled else
                          {s: d for s, d in data.items() if (s in S.TEST_SESSIONS) == (cfg.split == 'test')},
                          test_coder if cfg.split == 'test' else coder)
    for name, agreement in ceiling.items():
        results = pd.concat([results, pd.DataFrame([{'group': 'ceiling', 'variant': f'inter-coder {name}',
                                                     'model': 'coder', 'n': agreement['windows'],
                                                     'kappa': agreement['kappa'],
                                                     'kappa_codes': agreement['kappa_codes']}])], ignore_index=True)
    results.to_csv(run_dir / 'results.csv', index=False)
    coefficients = [frame for arm in arms for out in outputs[arm] for frame in (out['coefficients'] or [])]
    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(run_dir / 'coefficients.csv', index=False)

    scored = np.concatenate([d.y for d in data.values()])
    metrics = {
        'run': run_dir.name, 'split': cfg.split, 'target': cfg.target, 'classes': list(class_names(k)),
        'quick': cfg.quick, 'seconds': round(time.time() - started, 1),
        'labels': {'coder': coder, 'test_coder': test_coder if cfg.split == 'test' else None,
                   'coded': int(L.scored(scored).sum()), 'unclear': int((scored == L.UNCLEAR_Y).sum()),
                   'absent': int((scored == L.ABSENT_Y).sum()),
                   'join': {s: d.join for s, d in data.items()}},
        'inter_coder': ceiling,
        'inter_coder_note': (f"no ceiling: the truth coder {(test_coder if cfg.split == 'test' else coder)!r} is a "
                             f"model's labels file") if (test_coder if cfg.split == 'test' else coder) in models else None,
        'absent_class_policy': policy,
        'headline': {'variant': headline, 'selected_on': 'dev leave-one-date-out pooled '
                     + ('binary macro-F1' if binary_confirmatory else 'macro-F1'),
                     'strata': reports[headline].get('strata')} if headline else None,
        'contrasts': tests, 'presence_gate': gate, 'state_shares': shares, 'notes': notes, 'left_out': dropped,
        'folds': [{key: out[key] for key in ('name', 'train', 'test', 'inner', 'prior', 'transitions', 'models',
                                             'seconds')} for out in outputs['full']],
        'variants': {key: dict(reports[key], model=variants[key]['model'], temporal=variants[key]['temporal'],
                               hmm=variants[key]['hmm'], ablation=variants[key]['ablation'],
                               group=_result_row(key, variants[key], reports[key])['group'])
                     for key in variants},
    }
    if cfg.ablate != 'none':
        metrics['ablation'] = {
            'grid': cfg.ablate, 'arms': {arm: list(removed) for arm, removed in arms.items()},
            'full_only': [m for m in plan['models'] if m in FEATURELESS],
            'note': "exploratory: no Holm correction; an arm's modalities did not run in any session, train and test "
                    "alike, and the empty flag and the scoring strata are the full data's",
            # the full arm's folds are 'folds' above
            'folds': {arm: [{key: out[key] for key in ('name', 'models', 'seconds')} for out in outputs[arm]]
                      for arm in arms if arm != 'full'}}
    selected = [m for m in plan['models'] if m in SELECTORS]
    if selected:
        metrics['selectors'] = {selector: {
            'variant': _key(selector, NESTED, NESTED),
            'selected_on': next((out['models'][selector]['selected_on'] for out in outputs['full']
                                 if selector in out['models']), None),
            'winners': {out['name']: out['models'][selector]['winner'] for out in outputs['full']
                        if selector in out['models']},
            'note': SELECTOR_NOTES[selector]} for selector in selected}
    if plan.get('social_decision') in IN_FOLD_DECISIONS:
        metrics['social_decision'] = social_decision_record(outputs, plan['social_decision'])
    exploratory = [m for m in plan['models'] if m in EXPLORATORY]
    if exploratory:
        metrics['exploratory'] = {'models': {model: EXPLORATORY_NOTES[model] for model in exploratory},
                                  'soft_coder': plan.get('soft_coder')}
    session_record = None
    if by_session:
        session_record = _session_split_record(coder, folds, excluded, plan.get('soft_coder'))
        metrics['all_sessions'] = True
        metrics['across_sessions'] = session_record['across_sessions']
    elif pooled:
        session_record = _unit_split_record(cfg.split, coder, folds, excluded, data, plan.get('soft_coder'))
        metrics['all_sessions'] = True
        metrics['across_sessions'] = session_record['across_sessions']
        metrics['across_units'] = session_record['across_units']
        metrics['units'] = session_record['units']
    write_json(run_dir / 'metrics.json', metrics)
    write_json(run_dir / 'config.json', _config_record(cfg, plan, data, folds, artifacts, run_dir, coder,
                                                       session_record, not_read, content))
    if cfg.split == 'test':
        _record_test_run(artifacts, run_dir, cfg, 'finished')
    if pooled:
        _record_session_run(artifacts, run_dir, cfg, 'finished', data, coder)
    say(f"{len(variants)} variant(s) scored in {metrics['seconds']} s -> {run_dir}")
    return run_dir
