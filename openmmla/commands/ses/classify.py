"""mmla ses-classify: train and evaluate the 10 s interaction classifier across sessions.

It reads every session's fused table (mmla ses-fuse) and the coded labels (mmla ses-code), and
scores each named model on windows it was not trained on: --split loso holds out one DEV lesson at
a time (the two 2025-05-13 takes are one lesson), with every choice (grid point, epoch count,
calibrator, stacker, HMM gamma) made on inner folds over the other lessons; --split test trains
on every DEV session and scores the four TEST sessions, once, after every choice is frozen
(--confirm-frozen and the truth --coder the LOSO runs used; the run is recorded in
artifacts/_analysis/interaction/test_runs.jsonl).

Every run lands in artifacts/_analysis/interaction/<split>-<time>/ (or -o): predictions.csv,
metrics.json, results.csv, per_session.csv, confusion.csv, label_counts.csv, roster.json,
data_checks.json, config.json (and coefficients.csv for lr). The label counts are printed first;
when a class has fewer than 30 coded windows in an outer-training fold, the run refuses to train
any learned model and says which folds fall short. jev and jev-cal are left out, with the reason,
when a session they need has no ses-jev map made from its current fused table; a variant is
scored only on the windows it answered, and the coverage is printed when that is not all of them.

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
                    "sessions: leave one lesson out on the DEV sessions, or score the TEST sessions once.")
    parser.add_argument('-a', '--artifacts', default=None, help="artifacts root (default <cwd>/artifacts)")
    parser.add_argument('-s', '--sessions', default=None, help="only sessions whose id contains this text")
    parser.add_argument('--coder', default=None,
                        help="whose labels are the truth (default: the coder with the most windows over the DEV "
                             "sessions; required with --split test)")
    parser.add_argument('-m', '--models', default=DEFAULT_MODELS,
                        help="comma list of r0 (a-priori rule), jev (zero-shot, from mmla ses-jev's answers), "
                             "majority, stratified, r1 (fitted tree), jev-cal, lr, hgb, late-lr, late-hgb, pooled-net, "
                             f"net-notcn, net, net-pair (default {DEFAULT_MODELS})")
    parser.add_argument('--split', choices=('loso', 'date', 'task', 'test'), default='loso',
                        help="loso: leave one DEV lesson out; test: train on DEV, score TEST once (needs "
                             "--confirm-frozen and --coder); date and task are not built yet")
    parser.add_argument('--temporal', default='all',
                        help="comma list of T0 (no lags), T1c (causal lags), T2 (centred lags), or all (default all)")
    parser.add_argument('--hmm', default='all',
                        help="comma list of none, fb (forward-backward), filter (causal, for T0 and T1c inputs), "
                             "or all (default all)")
    parser.add_argument('--ablate', choices=ABLATIONS, default='none', help="ablation grid (only none is built yet)")
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
                        help="the network's epoch count for the refit, e.g. the median of the LOSO folds' for the "
                             "test model (default: chosen on the inner folds)")
    parser.add_argument('--jev-variant', choices=('j0', 'j1'), default='j0',
                        help="which cached Jev answers jev and jev-cal read (default j0)")
    parser.add_argument('--with-actions', action='store_true',
                        help="add the VLM action block (not built yet: the VLM has not run)")
    parser.add_argument('--jobs', type=int, default=1, help="outer folds run in parallel (default 1)")
    parser.add_argument('--quick', action='store_true',
                        help="two-point grids, 30 epochs, 200 bootstrap resamples: a plumbing check, never a result")
    parser.add_argument('--confirm-frozen', action='store_true',
                        help="with --split test: every choice is frozen, score the TEST sessions")
    parser.add_argument('-o', '--out', default=None,
                        help="run folder (default artifacts/_analysis/interaction/<split>-<time>)")
    return parser


def _choices(parser, text: str, allowed: tuple, name: str) -> tuple:
    """a comma list checked against `allowed`, with 'all' for every one of them."""
    items = [item.strip() for item in text.split(',') if item.strip()]
    if items == ['all']:
        return allowed
    unknown = [item for item in items if item not in allowed]
    if unknown or not items:
        parser.error(f"{name} takes a comma list of {', '.join(allowed)} or all, not {text!r}")
    return tuple(dict.fromkeys(items))


def _missing(modules) -> list:
    return [name for name in modules if importlib.util.find_spec(name) is None]


def _signed(value) -> str:
    return 'n/a' if value is None else f'{value:+.3f}'


def _summary(run_dir) -> str:
    """the results table, one line per variant, as the run folder's results.csv holds it."""
    import pandas as pd
    table = pd.read_csv(run_dir / 'results.csv')
    columns = [c for c in ('group', 'variant', 'n', 'macro_f1', 'macro_f1_lo', 'macro_f1_hi', 'binary_f1', 'kappa',
                           'nll', 'ece', 'switches_per_hour_pred', 'switches_per_hour_true') if c in table.columns]
    return table[columns].round(3).to_string(index=False, na_rep='n/a')


def _metrics(run_dir) -> dict:
    import json
    path = run_dir / 'metrics.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def _caveats(metrics: dict) -> list[str]:
    """what the results table alone does not show: models left out and why, the notes on each
    session's Jev map, and the variants scored on only part of the coded windows."""
    lines = [f"{model} left out: {reason}" for model, reason in (metrics.get('left_out') or {}).items()]
    lines += [f"note, {session}: {note}" for session, note in (metrics.get('notes') or {}).items()]
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
    from openmmla.analytics.interaction import evaluate as E
    models = _choices(parser, args.models, E.MODELS, '-m/--models')
    temporal = _choices(parser, args.temporal, E.TEMPORAL, '--temporal')
    hmm = _choices(parser, args.hmm, E.HMM_MODES, '--hmm')
    if args.ablate != 'none':
        parser.error("--ablate is not built yet: the ablations of section 4.6 are deferred")
    if args.with_actions:
        parser.error("--with-actions is not built yet: the semantic block waits for the VLM to run")
    if args.split in ('date', 'task'):
        parser.error(f"--split {args.split} is not built yet (splits.date_folds and task_transfer exist; the driver "
                     f"does not run them)")
    if args.split == 'test' and not args.confirm_frozen:
        parser.error("--split test scores the TEST sessions, once, after every choice is frozen: add --confirm-frozen")
    if args.split == 'test' and not args.coder:
        parser.error("--split test names its truth: add --coder, the coder of the LOSO runs "
                     "(their config.json 'coder')")
    if args.seeds < 1 or args.jobs < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("--seeds, --jobs and --epochs must be at least 1")
    planned = E.planned_variants(models, temporal, hmm, args.jev_variant)
    barren = [model for model, keys in planned.items() if not keys]
    if barren:
        parser.error(f"{', '.join(barren)} give(s) no variant with --temporal {','.join(temporal)} "
                     f"--hmm {','.join(hmm)}: filter runs only for inputs that never read a later window "
                     f"(T0, T1c, net-notcn, Jev)")

    if any(model in E.NETWORKS for model in models) and _missing(('torch',)):
        print("the network variants need torch: pip install torch")
        return 1

    from openmmla.analytics.interaction.labels import LabelJoinError
    config = E.Config(artifacts=args.artifacts or os.path.join(os.getcwd(), 'artifacts'), sessions=args.sessions,
                      coder=args.coder, models=models, split=args.split, temporal=temporal, hmm=hmm,
                      target=args.target, join=args.join, seeds=args.seeds, small=args.small, epochs=args.epochs,
                      jobs=args.jobs, out=args.out, quick=args.quick, confirm_frozen=args.confirm_frozen,
                      jev_variant=args.jev_variant)
    started = time.time()
    try:
        run_dir = E.run(config, log=print)
    except E.Refused as refusal:
        print(str(refusal))
        for problem in refusal.problems:
            print(f"  fold {problem['fold']}: {problem['counts']} coded windows in training, "
                  f"{problem['coded_lessons']} coded lesson(s)")
        if refusal.problems:
            print(f"code more windows (mmla ses-code), or try --target binary; label counts in {refusal.run_dir}")
        for line in _caveats(_metrics(refusal.run_dir)):
            print(line)
        return 1
    except LabelJoinError as error:
        print(str(error))
        return 1
    except FileNotFoundError as error:
        print(str(error))
        return 1

    metrics = _metrics(run_dir)
    print(_summary(run_dir))
    if metrics.get('headline'):
        print(f"headline ({metrics['headline']['selected_on']}): {metrics['headline']['variant']}")
    for name, contrast in (metrics.get('contrasts') or {}).items():
        if 'left_out' in contrast:
            print(f"{name} {contrast['a']} - {contrast['b']}: left out, {contrast['left_out']}")
            continue
        print(f"{name} {contrast['a']} - {contrast['b']}: {_signed(contrast['delta'])} "
              f"[{_signed(contrast['lo'])}, {_signed(contrast['hi'])}], Holm p {contrast['p_holm']:.3f}")
    for line in _caveats(metrics):
        print(line)
    policy = metrics['absent_class_policy']
    if policy['confirmatory_target'] == 'binary':
        print(f"absent-class policy: {policy['social_windows']} social windows, social in "
              f"{policy['lessons_with_social']} lesson(s): the confirmatory target is binary")
    if config.quick:
        print("--quick: small grids and short training, not a result to report")
    print(f"done in {time.time() - started:.0f} s -> {run_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
