"""mmla ses-classify: train and evaluate the 10 s interaction classifier across sessions.

It reads every session's fused table (mmla ses-fuse) and the coded labels (mmla ses-code), and
scores each named model on windows it was not trained on. The unit of every split is the date,
so the same group on the same date is never on both sides: --split date (the default) holds out
one DEV date at a time, every session of it together (two groups of one class, a group's micro:bit
and WeGrow lessons, the two 2025-05-13 takes), with every choice (grid point, epoch count,
calibrator, stacker, HMM gamma) made on inner folds over the other dates; --split test trains on
every DEV session and scores the four TEST sessions, once, after every choice is frozen
(--confirm-frozen and the truth --coder the date runs used; --test-coder scores TEST against
another coder, e.g. a model's DEV labels train and a human coder's TEST labels score; the run is
recorded in artifacts/_analysis/interaction/test_runs.jsonl). --split loso, leave one lesson out, is gone: it
put a group's two lessons of one morning on both sides.

--split session (decided 2026-10-03, when the TEST sessions' data quality turned out too poor to stand
for the model) pools DEV and TEST: every session the --coder labelled (their own labels/<coder>.jsonl,
adjudicated.jsonl not read) that the inclusion rule S1 keeps is held out in turn, the models train on
all the others, and every choice is made on leave-one-session-out inner folds within those. Each
held-out session is scored on its own (per_session.csv, with flags for a session of one class or few
windows) and every variant gets the mean and SD over the sessions beside the pooled value (results.csv,
metrics.json across_sessions); config.json says all_sessions true. There is no headline and no
contrast, and nothing is recorded in test_runs.jsonl: it is not the frozen TEST scoring.

--split unit and --split forward (decided 2026-10-04) read the sessions --split session reads (DEV and
TEST alike, the --coder's own labels file, the same sessions left out) but hold out a unit at a time: a
recording date with every date its sessions' same_class_as links reach (mmla ses-tidy --same-class-as).
unit leaves one unit out; forward holds out each date with at least 4 earlier dates with coded windows
and trains on the earlier dates only. The inner folds leave one training unit out. The units are printed
and kept in config.json; per_unit.csv scores each held-out unit and results.csv gives the mean and SD over
the units beside the pooled value. Every session, unit and forward run writes a line when it starts and
one when it finishes to artifacts/_analysis/interaction/session_runs.jsonl; --ledger-add adds run folders
made before that, as looked at, with --ledger-note kept on each line.

-m rule22sep and -m select-all are pseudo-variants that choose inside each outer fold, from the
calibrated inner out-of-fold answers, among the variants the run fits, and copy the chosen one's
held-out answers: rule22sep by the rule declared on 22 Sep (late-lr or late-hgb, every temporal mode,
HMM none or fb, by binary macro-F1 under --target binary and macro-F1 under 3class), select-all by the
class-weighted binary log-loss over every variant of the named models. Each fold's choice is in
metrics.json. --net-oof common gives the networks' inner out-of-fold answers at the common E* of the
refit rather than at each inner split's own best epoch. --features-root reads the fused tables from
another folder (e.g. an ablation arm's re-fused tables), never artifacts/<session>/analysis/features/;
a session the run would read from artifacts/ but whose table that folder lacks refuses the run (one
without the --coder's labels in a session, unit or forward run excepted, which those leave out anyway).

-m pmil-lr, -m lr-soft and -m net-attn are exploratory candidates of the architecture panel (WP9,
2026-10-04), named only on purpose (never by -m all) and grouped 'exploratory' in the results:
pmil-lr, a noisy-OR over the pupil pairs of one shared linear scorer with a bias per group size,
answers p(interaction) only and needs --target binary; lr-soft is lr fitted on two coders' soft
labels (a window both gave a class counts half for each label, a window the --coder called unclear or
absent not at all), the second coder being --soft-coder or else the one human coder besides --coder
whose labels the training sessions hold; net-attn is self-attention over the persons with the pairs as
its bias, trained as the networks are. All three are calibrated, smoothed and scored as the model they
extend and are candidates of select-all.

--agreement A,B trains nothing: Cohen's kappa and the percent agreement of two coders on the windows
both labelled, per session and pooled, over the same sessions after the same label join, written to
agreement.csv and agreement.json (default folder artifacts/_analysis/interaction/agreement-<time>).

Every run lands in artifacts/_analysis/interaction/<split>-<time>/ (or -o): predictions.csv,
metrics.json, results.csv, per_session.csv, confusion.csv, label_counts.csv, state_shares.csv,
roster.json, data_checks.json, config.json (and coefficients.csv for lr, ablation.csv with --ablate
modality). The label counts are printed first;
when a class has fewer than 30 coded windows in an outer-training fold, the run refuses to train
any learned model and says which folds fall short. jev and jev-cal are left out, with the reason,
when a session they need has no ses-jev map made from its current fused table; a variant is
scored only on the windows it answered, and the coverage is printed when that is not all of them.

--ablate modality fits and scores every learned model and the rule once more per arm (no_speech,
no_space, no_body_gaze, only_body_gaze, only_speech), each with its modalities not run in any
session, train and test alike; the floors and Jev run in the full arm only, the headline and the
contrasts are the full arm's, and ablation.csv sets the arms side by side. It is one run, so with
--split test one look at TEST.

--quick swaps in two-point grids, 30 training epochs and 200 bootstrap resamples: for checking
the plumbing end to end, never for a reported number.
"""
import argparse
import importlib.util
import os
import sys
import time

DEFAULT_MODELS = 'r0,r1,lr,hgb,late-lr,late-hgb'
ABLATIONS = ('none', 'modality', 'temporal', 'fusion', 'ladder', 'weights', 'all')


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mmla ses-classify',
        description="Train and evaluate the 10 s interaction classifier (individual, social, collaborative) across "
                    "sessions: leave one date out on the DEV sessions, score the TEST sessions once, or leave one "
                    "session out over every session a coder labelled, DEV and TEST alike; or, with --agreement, "
                    "score two coders against each other.")
    parser.add_argument('-a', '--artifacts', default=None, help="artifacts root (default <cwd>/artifacts)")
    parser.add_argument('-s', '--sessions', default=None, help="only sessions whose id contains this text")
    parser.add_argument('--coder', default=None,
                        help="whose labels are the truth, by the labels file's exact name (Arthur reads "
                             "labels/Arthur.jsonl; default: the coder with the most windows over the DEV sessions; "
                             "required with --split test, session, unit and forward)")
    parser.add_argument('--test-coder', default=None,
                        help="with --split test: whose labels score the TEST sessions, when not --coder's (e.g. "
                             "train on a model's DEV labels, score on a human coder's TEST labels; "
                             "adjudicated.jsonl overrules it)")
    parser.add_argument('-m', '--models', default=DEFAULT_MODELS,
                        help="comma list of r0 (a-priori rule), r0-v1 (its first version, kept for the record), jev "
                             "(zero-shot, from mmla ses-jev's answers), "
                             "majority, stratified, r1 (fitted tree), jev-cal, lr, hgb, late-lr, late-hgb, pooled-net, "
                             f"net-notcn, net, net-pair (default {DEFAULT_MODELS}); the selectors rule22sep (the "
                             "22 Sep rule over late-lr and late-hgb) and select-all (the lowest inner log-loss over the "
                             "named models), chosen in each fold on the inner out-of-fold answers; and the exploratory "
                             "candidates pmil-lr (pairs, noisy-OR; --target binary only), lr-soft (two coders' soft "
                             "labels) and net-attn (attention over persons), never in all")
    parser.add_argument('--split', choices=('date', 'task', 'test', 'session', 'unit', 'forward'), default='date',
                        help="date: leave one DEV date out, all its sessions together (default); test: train on "
                             "DEV, score TEST once (needs --confirm-frozen and --coder); session: leave one session "
                             "out over every session the --coder labelled, DEV and TEST alike, with the mean and SD "
                             "over the sessions and the pooled value (needs --coder); unit: the same sessions, one "
                             "unit (a date with its same_class_as dates) out at a time (needs --coder); forward: the "
                             "same sessions, each date with 4 earlier dates held out and trained on the earlier "
                             "dates only (needs --coder); task is not built yet")
    parser.add_argument('--agreement', default=None, metavar='CODER_A,CODER_B',
                        help="train nothing: Cohen's kappa and percent agreement of two coders (labels file names) on "
                             "the windows both labelled, per session and pooled, over every session S1 keeps (DEV "
                             "and TEST alike) after the classifier's label join (--join); writes agreement.csv and "
                             "agreement.json to -o")
    parser.add_argument('--temporal', default='all',
                        help="comma list of T0 (no lags), T1c (causal lags), T2 (centred lags), or all (default all)")
    parser.add_argument('--hmm', default='all',
                        help="comma list of none, fb (forward-backward), filter (causal, for T0 and T1c inputs), "
                             "or all (default all)")
    parser.add_argument('--ablate', choices=ABLATIONS, default='none',
                        help="ablation grid: none (default), or modality (every learned model and the rule once more "
                             "without speech, space, body_gaze, speech+space and space+body_gaze, in every session; "
                             "writes ablation.csv); temporal, fusion, ladder, weights and all are not built yet")
    parser.add_argument('--scaling', choices=('mix', 's', 'c', 'g'), default='mix',
                        help="how the feature values are scaled, in every session alike: mix (default, as each "
                             "value is tagged: most by the training sessions' statistics, a few within the session), "
                             "s (every value by its own session's median and spread), c (every value centred on its "
                             "session's median, on the training spread) or g (every value by the training statistics)")
    parser.add_argument('--target', choices=('3class', 'binary'), default='3class',
                        help="3class, or binary (individual against interaction): the fallback when a fold lacks "
                             "a class")
    parser.add_argument('--join', choices=('exact', 'overlap'), default='exact',
                        help="how labels meet the table's windows: exact start (then nearest within 0.5 s), or by at "
                             "least 5 s of overlap when the coding grid moved")
    parser.add_argument('--seeds', type=int, default=5, help="network seeds in the refit ensemble (default 5)")
    parser.add_argument('--small', choices=('auto', 'yes', 'no'), default='auto',
                        help="the network's small configuration: auto below 3,000 coded training windows")
    parser.add_argument('--epochs', type=int, default=None,
                        help="the network's epoch count for the refit, e.g. the median of the date folds' for the "
                             "test model (default: chosen on the inner folds)")
    parser.add_argument('--net-oof', choices=('best', 'common'), default='best',
                        help="the networks' inner out-of-fold answers, which fit their calibrator and HMM gamma and "
                             "enter select-all: best (default) at each inner split's own best epoch; common at the "
                             "refit's epoch count for every split (kept every 5 epochs; a split stopped early is "
                             "replayed to it)")
    parser.add_argument('--features-root', default=None, metavar='FOLDER',
                        help="read each session's fused table from FOLDER (<session>_window_features.csv, or under "
                             "<session>/ or <session>/analysis/features/) instead of artifacts/<session>/analysis/"
                             "features/; labels, manifests and Jev maps still come from artifacts/<session>/. A "
                             "session whose table FOLDER lacks refuses the run (see the module help)")
    parser.add_argument('--ledger-add', nargs='+', default=None, metavar='RUN_DIR',
                        help="train nothing: add these run folders, made before the run ledger, to "
                             "artifacts/_analysis/interaction/session_runs.jsonl as looked at (a folder already there "
                             "is not added again)")
    parser.add_argument('--ledger-note', default=None, help="with --ledger-add: a note kept on each line")
    parser.add_argument('--soft-coder', default=None, metavar='CODER',
                        help="with -m lr-soft: the second coder of its soft labels, by labels file name (default: "
                             "the one human coder besides --coder whose labels the training sessions hold)")
    parser.add_argument('--jev-variant', choices=('j0', 'j1'), default='j0',
                        help="which cached Jev answers jev and jev-cal read (default j0)")
    parser.add_argument('--with-actions', action='store_true',
                        help="add the VLM action block (not built yet: the VLM has not run)")
    parser.add_argument('--jobs', type=int, default=1, help="outer folds run in parallel (default 1)")
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu',
                        help="where the networks train: cpu (default, one thread per fold), cuda, or auto (cuda when "
                             "torch sees a GPU); parallel folds share the GPU. A GPU run repeats itself, but differs from a "
                             "CPU run by floating-point rounding and dropout draws")
    parser.add_argument('--quick', action='store_true',
                        help="two-point grids, 30 epochs, 200 bootstrap resamples: a plumbing check, never a result")
    parser.add_argument('--confirm-frozen', action='store_true',
                        help="with --split test: every choice is frozen, score the TEST sessions")
    parser.add_argument('-o', '--out', default=None,
                        help="run folder (default artifacts/_analysis/interaction/<split>-<time>, or agreement-<time> "
                             "with --agreement)")
    return parser


def _choices(parser, text: str, allowed: tuple, name: str, extra: tuple = ()) -> tuple:
    """a comma list checked against `allowed` and `extra`, with 'all' for every one of `allowed`."""
    items = [item.strip() for item in text.split(',') if item.strip()]
    if items == ['all']:
        return allowed
    unknown = [item for item in items if item not in tuple(allowed) + tuple(extra)]
    if unknown or not items:
        parser.error(f"{name} takes a comma list of {', '.join(tuple(allowed) + tuple(extra))} or all, not {text!r}")
    return tuple(dict.fromkeys(items))


def _missing(modules) -> list:
    return [name for name in modules if importlib.util.find_spec(name) is None]


def _signed(value) -> str:
    return 'n/a' if value is None else f'{value:+.3f}'


def _plain(value) -> str:
    return 'n/a' if value is None else f'{value:.3f}'


def _summary(run_dir) -> str:
    """the results table, one line per variant, as the run folder's results.csv holds it."""
    import pandas as pd
    table = pd.read_csv(run_dir / 'results.csv')
    # the arm column only when the run has an ablated arm, so a run without one prints as before
    arms = 'ablation' in table.columns and table['ablation'].dropna().ne('full').any()
    names = ('group', 'variant') + (('ablation',) if arms else ()) + (
        'n', 'macro_f1', 'macro_f1_lo', 'macro_f1_hi', 'binary_f1', 'kappa', 'nll', 'ece', 'switches_per_hour_pred',
        'switches_per_hour_true')
    columns = [c for c in names if c in table.columns]
    return table[columns].round(3).to_string(index=False, na_rep='n/a')


def _session_summary(run_dir, unit: str = 'session', split: str = 'session') -> str:
    """the session split's results table: per variant the sessions, the mean and SD over them and
    the pooled value of macro-F1 and accuracy, and the means of kappa, AUROC, Brier and NLL; with
    `unit` 'unit' the same over the held-out units of the unit or forward `split`."""
    import pandas as pd
    table = pd.read_csv(run_dir / 'results.csv')
    table = table[table['group'] != 'ceiling']
    arms = 'ablation' in table.columns and table['ablation'].dropna().ne('full').any()
    names = ('group', 'variant') + (('ablation',) if arms else ()) + (
        f'{unit}s', f'{unit}s_flagged', 'macro_f1_mean', 'macro_f1_sd', 'macro_f1', 'accuracy_mean', 'accuracy_sd',
        'accuracy', 'kappa_mean', 'auroc_mean', 'auroc', 'brier_mean', 'nll_mean')
    columns = [c for c in names if c in table.columns]
    head = {'session': "leave one session out over DEV and TEST alike: the mean and SD over the held-out sessions",
            'unit': "leave one unit (a date with its same_class_as dates) out over DEV and TEST alike: the mean and "
                    "SD over the held-out units",
            'forward': "forward chaining over DEV and TEST alike, each date trained on the earlier ones: the mean and "
                       "SD over the held-out units"}[split]
    return (f"{head}, and the pooled value over their windows (macro_f1, accuracy, auroc):\n"
            + table[columns].round(3).to_string(index=False, na_rep='n/a'))


def _agreement_summary(run_dir) -> str:
    """agreement.csv as printed: per session and pooled, the windows both gave a class, kappa and
    percent agreement (three classes, binary, five codes)."""
    import pandas as pd
    table = pd.read_csv(run_dir / 'agreement.csv')
    columns = ['session', 'windows', 'kappa', 'percent_agreement', 'kappa_binary', 'percent_agreement_binary',
               'both_coded', 'kappa_codes', 'percent_agreement_codes']
    return table[columns].round(3).to_string(index=False, na_rep='n/a')


def _agreement(args) -> int:
    """--agreement A,B: the two coders against each other, nothing trained."""
    import json
    from openmmla.analytics.interaction import evaluate as E
    from openmmla.analytics.interaction.labels import LabelJoinError
    coders = [name.strip() for name in args.agreement.split(',') if name.strip()]
    if len(coders) != 2 or coders[0] == coders[1]:
        print(f"--agreement takes two different coders, e.g. Arthur,zaibei, not {args.agreement!r}")
        return 2
    try:
        run_dir = E.coder_agreement(args.artifacts or os.path.join(os.getcwd(), 'artifacts'), coders,
                                    sessions=args.sessions, join=args.join, out=args.out, log=print)
    except (FileNotFoundError, ValueError, LabelJoinError) as error:
        print(str(error))
        return 1
    print(_agreement_summary(run_dir))
    record = json.loads((run_dir / 'agreement.json').read_text(encoding='utf-8'))
    for session, why in record['excluded'].items():
        print(f"left out, {session}: {why}")
    for coder, sessions in record['coverage'].items():
        print(f"{coder} coded {sum(sessions.values())} window(s) in {len(sessions)} session(s) the rule S1 keeps")
    print(f"-> {run_dir}")
    return 0


def _ledger_add(args) -> int:
    """--ledger-add: run folders made before the run ledger added to it as looked at, nothing trained."""
    from openmmla.analytics.interaction import evaluate as E
    artifacts = args.artifacts or os.path.join(os.getcwd(), 'artifacts')
    if not os.path.isdir(artifacts):
        print(f"no artifacts folder at {artifacts}: give it with -a")
        return 1
    done = E.log_run_folders(artifacts, args.ledger_add, note=args.ledger_note)
    for folder, what in done:
        print(f"{folder}: {what}")
    print(f"-> {E.ledger_path(artifacts)}")
    return 0 if all(not what.startswith('skipped') for _, what in done) else 1


def _ablation_summary(run_dir) -> str | None:
    """ablation.csv condensed, one line per variant, one column per arm: the binary macro-F1 with
    its interval and the three-class macro-F1; None when the run has no ablation."""
    import pandas as pd
    path = run_dir / 'ablation.csv'
    if not path.exists():
        return None
    table = pd.read_csv(path)
    if table.empty:
        return None

    def cell(row):
        interval = '' if pd.isna(row['binary_macro_f1_lo']) else \
            f" [{row['binary_macro_f1_lo']:.3f}, {row['binary_macro_f1_hi']:.3f}]"
        binary = 'n/a' if pd.isna(row['binary_macro_f1']) else f"{row['binary_macro_f1']:.3f}"
        three = 'n/a' if pd.isna(row['macro_f1']) else f"{row['macro_f1']:.3f}"
        return f"{binary}{interval} / {three}"

    table['cell'] = table.apply(cell, axis=1)
    arms = list(dict.fromkeys(table['ablation']))
    wide = table.pivot(index='variant', columns='ablation', values='cell')
    wide = wide.reindex(index=list(dict.fromkeys(table['variant'])), columns=arms)
    return ("modality ablation, binary macro-F1 [interval] / macro-F1 (exploratory):\n"
            + wide.to_string(na_rep='-'))


def _metrics(run_dir) -> dict:
    import json
    path = run_dir / 'metrics.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def _caveats(metrics: dict) -> list[str]:
    """what the results table alone does not show: models left out and why, the notes on each
    session's Jev map, the intervals left out for too few units, and the variants scored on only
    part of the coded windows."""
    lines = [f"{model} left out: {reason}" for model, reason in (metrics.get('left_out') or {}).items()]
    lines += [f"note, {session}: {note}" for session, note in (metrics.get('notes') or {}).items()]
    no_interval = next((report['macro_f1_ci']['left_out'] for report in (metrics.get('variants') or {}).values()
                        if (report.get('macro_f1_ci') or {}).get('left_out')), None)
    if no_interval:
        lines.append(f"no bootstrap intervals: {no_interval}")
    for key, report in (metrics.get('variants') or {}).items():
        share = report.get('coverage') or {}
        if share.get('coded') and share.get('answered') != share.get('coded'):
            lines.append(f"{key} answered {share['answered']} of {share['coded']} coded windows and is scored on those "
                         f"only (short in {', '.join(share.get('short_sessions') or [])})")
    return lines


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    # checked before the classifier is imported, which needs them all
    missing = _missing(('numpy', 'pandas', 'sklearn', 'scipy'))
    if missing:
        print(f"missing {', '.join(missing)}: pip install \"openmmla[analysis]\"")
        return 1
    if args.ledger_note and not args.ledger_add:
        parser.error("--ledger-note goes with --ledger-add")
    if args.ledger_add:
        return _ledger_add(args)
    if args.agreement:
        return _agreement(args)
    from openmmla.analytics.interaction import evaluate as E
    models = _choices(parser, args.models, E.MODELS, '-m/--models', E.SELECTORS + E.EXPLORATORY)
    temporal = _choices(parser, args.temporal, E.TEMPORAL, '--temporal')
    hmm = _choices(parser, args.hmm, E.HMM_MODES, '--hmm')
    if args.ablate not in ('none', 'modality'):
        parser.error(f"--ablate {args.ablate} is not built yet: of the ablations of section 4.6 only modality is built")
    if args.ablate == 'modality' and all(model in E.FEATURELESS for model in models):
        parser.error(f"--ablate modality needs a model that reads features: {', '.join(models)} run in the full arm "
                     f"only (the floors and Jev read none)")
    if args.with_actions:
        parser.error("--with-actions is not built yet: the semantic block waits for the VLM to run")
    if args.split == 'task':
        parser.error("--split task is not built yet (splits.task_transfer exists; the driver does not run it)")
    if args.split == 'test' and not args.confirm_frozen:
        parser.error("--split test scores the TEST sessions, once, after every choice is frozen: add --confirm-frozen")
    if args.split == 'test' and not args.coder:
        parser.error("--split test names its truth: add --coder, the coder of the date runs "
                     "(their config.json 'coder')")
    if args.split in ('session', 'unit', 'forward') and not args.coder:
        parser.error(f"--split {args.split} reads one coder's labels at a time: add --coder (the labels file's name, "
                     f"e.g. Arthur or zaibei)")
    if args.test_coder and args.split != 'test':
        parser.error("--test-coder names the truth of the TEST sessions: it goes with --split test")
    binary_only = [model for model in models if model in E.BINARY_ONLY]
    if binary_only and args.target != 'binary':
        parser.error(f"{', '.join(binary_only)} answer(s) p(interaction) only (a noisy-OR over the pairs has no social "
                     f"against collaborative): add --target binary")
    if args.soft_coder and 'lr-soft' not in models:
        parser.error("--soft-coder names lr-soft's second coder: it goes with -m lr-soft")
    if args.seeds < 1 or args.jobs < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("--seeds, --jobs and --epochs must be at least 1")
    gap = E._selector_gap(E.Config(models=models, hmm=hmm))
    if gap:
        parser.error(gap)
    if args.features_root and not os.path.isdir(args.features_root):
        parser.error(f"--features-root {args.features_root} is no folder")
    planned = E.planned_variants(models, temporal, hmm, args.jev_variant)
    barren = [model for model, keys in planned.items() if not keys]
    if barren:
        parser.error(f"{', '.join(barren)} give(s) no variant with --temporal {','.join(temporal)} "
                     f"--hmm {','.join(hmm)}: filter runs only for inputs that never read a later window "
                     f"(T0, T1c, net-notcn, Jev)")

    if any(model in E.NEURAL for model in models) and _missing(('torch',)):
        print("the network variants need torch: pip install torch")
        return 1
    if args.device == 'cuda' and any(model in E.NEURAL for model in models):
        import torch
        if not torch.cuda.is_available():
            print(f"--device cuda, but this torch ({torch.__version__}) sees no CUDA GPU: install a CUDA build of "
                  f"torch, or use --device cpu or auto")
            return 1

    from openmmla.analytics.interaction.labels import LabelJoinError
    config = E.Config(artifacts=args.artifacts or os.path.join(os.getcwd(), 'artifacts'), sessions=args.sessions,
                      coder=args.coder, test_coder=args.test_coder, models=models, split=args.split,
                      temporal=temporal, hmm=hmm,
                      target=args.target, join=args.join, seeds=args.seeds, small=args.small, epochs=args.epochs,
                      jobs=args.jobs, out=args.out, quick=args.quick, confirm_frozen=args.confirm_frozen,
                      jev_variant=args.jev_variant, device=args.device, ablate=args.ablate,
                      scaling=args.scaling, net_oof=args.net_oof, features_root=args.features_root,
                      soft_coder=args.soft_coder)
    started = time.time()
    try:
        run_dir = E.run(config, log=print)
    except E.Refused as refusal:
        print(str(refusal))
        unit = 'session(s)' if args.split == 'session' else 'unit(s)' if args.split in ('unit', 'forward') \
            else 'date(s)'
        for problem in refusal.problems:
            print(f"  fold {problem['fold']}: {problem['counts']} coded windows in training, "
                  f"{problem['coded_units']} coded {unit}")
        if refusal.problems:
            print(f"code more windows (mmla ses-code), or try --target binary; label counts in {refusal.run_dir}")
        for line in _caveats(_metrics(refusal.run_dir) if refusal.run_dir else {}):
            print(line)
        return 1
    except LabelJoinError as error:
        print(str(error))
        return 1
    except FileNotFoundError as error:
        print(str(error))
        return 1

    metrics = _metrics(run_dir)
    if args.split == 'session':
        print(_session_summary(run_dir))
        flagged = sorted({session for report in (metrics.get('variants') or {}).values()
                          for session in (report.get('across_sessions') or {}).get('flagged', [])})
        if flagged:
            print(f"flagged held-out sessions (fewer than {E.MIN_SESSION_WINDOWS} scored windows, or one class): "
                  f"{', '.join(flagged)}; they enter the means, *_unflagged in metrics.json leave them out")
        for session, why in ((metrics.get('across_sessions') or {}).get('excluded') or {}).items():
            print(f"left out, {session}: {why}")
    elif args.split in ('unit', 'forward'):
        print(_session_summary(run_dir, 'unit', args.split))
        flagged = sorted({unit for report in (metrics.get('variants') or {}).values()
                          for unit in (report.get('across_units') or {}).get('flagged', [])})
        if flagged:
            print(f"flagged held-out units (fewer than {E.MIN_SESSION_WINDOWS} scored windows, or one class): "
                  f"{', '.join(flagged)}; they enter the means, *_unflagged in metrics.json leave them out")
        for session, why in ((metrics.get('across_sessions') or {}).get('excluded') or {}).items():
            print(f"left out, {session}: {why}")
    else:
        print(_summary(run_dir))
    ablation = _ablation_summary(run_dir)
    if ablation:
        print(ablation)
    if metrics.get('headline'):
        print(f"headline ({metrics['headline']['selected_on']}): {metrics['headline']['variant']}")
    for selector, record in (metrics.get('selectors') or {}).items():
        chosen = ', '.join(f"{fold} {winner or 'none'}" for fold, winner in record['winners'].items())
        print(f"{selector} chose, per fold: {chosen}")
    for name, contrast in (metrics.get('contrasts') or {}).items():
        if 'left_out' in contrast:
            print(f"{name} {contrast['a']} - {contrast['b']}: left out, {contrast['left_out']}")
            continue
        print(f"{name} {contrast['a']} - {contrast['b']}: {_signed(contrast['delta'])} "
              f"[{_signed(contrast['lo'])}, {_signed(contrast['hi'])}], Holm p {contrast['p_holm']:.3f}")
    gate = metrics.get('presence_gate') or {}
    if gate.get('coded'):
        print(f"presence gate: {gate['absent']} absent of {gate['coded']} coded windows, {gate['gated']} gated; "
              f"precision {_plain(gate['precision'])}, recall {_plain(gate['recall'])}, kappa {_plain(gate['kappa'])}")
    for line in _caveats(metrics):
        print(line)
    policy = metrics['absent_class_policy']
    if args.split in ('session', 'unit', 'forward'):
        print(f"social windows over every session: {policy['social_windows']}, social in "
              f"{policy['lessons_with_social']} lesson(s); no headline or contrast in the {args.split} split")
    elif policy['confirmatory_target'] == 'binary':
        print(f"absent-class policy: {policy['social_windows']} social windows, social in "
              f"{policy['lessons_with_social']} lesson(s): the confirmatory target is binary")
    if config.quick:
        print("--quick: small grids and short training, not a result to report")
    print(f"done in {time.time() - started:.0f} s -> {run_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
