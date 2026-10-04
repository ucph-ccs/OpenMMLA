"""mmla ses-contrast: paired contrasts of two variants of the interaction classifier, by unit.

Each -c NAME A B compares variant A with variant B on the coded windows both hold, where a side is
<predictions.csv or run folder>@<model:temporal:hmm[:arm]> (lr:T0:none, late-lr:T0:fb:no_speech), and
B may name the variant alone to read it from A's file. Two sides that scored different coded windows
(runs over different sessions) are refused unless --allow-subset compares the windows both hold. Per
unit (the run's lesson column: a date with its same_class_as dates, or the session in a session split)
it computes the within-session concordance C, the binary kappa and the binary macro-F1 of both sides
and their difference, then over the units the paired t-interval (df = units - 1), the exact sign-flip
p, the unit-cluster bootstrap and, with --margin, TOST for equivalence on C. Every -c of one call is
one declared family: Holm adjusts its p-values metric by metric. Writes contrasts.json, contrasts.csv
and contrasts_per_unit.csv to -o (default artifacts/_analysis/interaction/contrast-<time>) and adds a
line to the run ledger artifacts/_analysis/interaction/session_runs.jsonl, since a contrast is a look
at results; without an artifacts folder (-a) there is no ledger, and the call needs -o.
"""
import argparse
import os
import sys
from datetime import datetime, timezone


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mmla ses-contrast',
        description="Paired contrasts of two interaction-classifier variants on the same windows, by unit: within-"
                    "session concordance C, binary kappa and binary macro-F1, with the paired t-interval, the exact "
                    "sign-flip test, the unit-cluster bootstrap, Holm over the contrasts of one call and TOST.")
    parser.add_argument('-c', '--contrast', nargs=3, action='append', required=True, metavar=('NAME', 'A', 'B'),
                        help="a contrast A - B: each side <predictions.csv or run folder>@<model:temporal:hmm[:arm]>; "
                             "B may be the variant alone, read from A's file. Repeat for a family (Holm over all)")
    parser.add_argument('--margin', type=float, default=None,
                        help="TOST's equivalence margin on C (e.g. 0.02); without it no equivalence test")
    parser.add_argument('--alpha', type=float, default=0.05,
                        help="the tests' level: (1 - alpha) t-intervals, (1 - 2 alpha) for TOST (default 0.05)")
    parser.add_argument('--bootstrap', type=int, default=2000, help="unit-cluster bootstrap draws (default 2000)")
    parser.add_argument('--seed', type=int, default=20261004, help="the bootstrap's and sign-flip draws' seed")
    parser.add_argument('--unit', choices=('lesson', 'session'), default='lesson',
                        help="what a unit is: lesson, the run's own unit (default), or session")
    parser.add_argument('--allow-subset', action='store_true',
                        help="compare two sides that scored different coded windows (e.g. a run that lacks a "
                             "session) on the windows both hold; without it they are refused")
    parser.add_argument('-a', '--artifacts', default=None,
                        help="artifacts root, for the run ledger and the default -o (default <cwd>/artifacts)")
    parser.add_argument('-o', '--out', default=None,
                        help="folder to write to (default artifacts/_analysis/interaction/contrast-<time>)")
    return parser


def _side(text: str, default_path=None):
    """(path, variant) of one side: path@variant, or the variant alone with `default_path`."""
    if '@' in text:
        path, variant = text.rsplit('@', 1)
        return path, variant
    if default_path is None:
        raise ValueError(f"{text!r} names no predictions: write <predictions.csv or run folder>@<variant>")
    return default_path, text


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 0.5:
        parser.error("--alpha must lie between 0 and 0.5")
    if args.margin is not None and args.margin <= 0:
        parser.error("--margin must be above 0")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be at least 1")
    from openmmla.analytics.interaction import contrasts as C
    from openmmla.analytics.interaction import evaluate as E
    specs = []
    try:
        for name, a, b in args.contrast:
            side_a = _side(a)
            specs.append((name, side_a, _side(b, side_a[0])))
    except ValueError as error:
        parser.error(str(error))
    for _, (path_a, _), (path_b, _) in specs:
        for path in (path_a, path_b):
            if not C.predictions_path(path).exists():
                print(f"no predictions.csv at {path}")
                return 1
    artifacts = args.artifacts or os.path.join(os.getcwd(), 'artifacts')
    # a look at results goes into the run ledger, never into one that the default -o would create
    ledger = os.path.isdir(artifacts)
    if not ledger and not args.out:
        parser.error(f"no artifacts folder at {artifacts}: give it with -a (or a folder to write to with -o)")
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.out or os.path.join(artifacts, '_analysis', 'interaction', f'contrast-{stamp}')
    try:
        result = C.run_contrasts(specs, out, margin=args.margin, alpha=args.alpha, n_boot=args.bootstrap,
                                 seed=args.seed, unit=args.unit, log=print, subset=args.allow_subset)
    except C.ContrastError as error:
        print(str(error))
        return 1
    print(C.summary(result))
    if ledger:
        E._append_ledger(artifacts, C.ledger_line(result, out))
        print(f"logged in {E.ledger_path(artifacts)}")
    else:
        print(f"no run ledger line: there is no artifacts folder at {artifacts} (give it with -a)")
    print(f"-> {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
