"""mmla ses-jev: ask Jev to label each 10 s window from a plain description of its sensor features.

This is the zero-shot baseline of the interaction classifier: no label is read, and Jev sees only a
plain-text state per window (persons as A, B and C, rounded numbers with words, missing modalities
said in words) and the coder's codebook. The tertile words are part of the template, so they are
fitted once, on every dev session under artifacts/ (never a test session, whichever sessions -s
picks), and frozen in artifacts/_analysis/interaction/jev/bins.json (bins_pilot.json for a pilot,
whose tables predate the camera fix). Every later run reads them from there, or from --bins; only
--fit-bins fits them again, and then says so. Answers are cached by the sha256 of the request
under artifacts/_analysis/interaction/jev/cache/, so a rerun costs nothing; each session gets
artifacts/<session>/analysis/interaction/jev_<variant>.jsonl (window -> request hash, with the
fused table's, the template's and the roster's digests; a --limit run adds to it, after dropping
the answers asked about another roster than the session has now), and the run folder gets
predictions.csv, bins.json and sanity.json.

The key is read from the provider's environment variable only (OPENROUTER_API_KEY by default), and the command refuses to call
without it. It is never printed, logged, cached or put in a URL. --dry-run prints three sample
states, one request with the key masked and the token and dollar estimate, and sends nothing.
--pilot runs on dev sessions only, with the quantities the two-camera interleave corrupts left out
of the state; it is for checking the plumbing, wording and cost and is never scored.
"""
import argparse

from openmmla.analytics.interaction.jev import DEFAULT_PROVIDER as J_DEFAULT_PROVIDER, PROVIDERS as J_PROVIDERS
import json
import os
import sys
from pathlib import Path

# how often the run says how far it got
PROGRESS_EVERY = 100


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mmla ses-jev',
        description="Ask Jev to label each 10 s window from a plain description of its sensor features (the "
                    "zero-shot baseline). The key is read from the provider's environment variable only (OPENROUTER_API_KEY by default).")
    parser.add_argument('-a', '--artifacts', default=None, help="artifacts root (default <cwd>/artifacts)")
    parser.add_argument('-s', '--sessions', default=None, help="only sessions whose id contains this text")
    parser.add_argument('--variant', choices=('j0', 'j1', 'j2'), default='j0',
                        help="j0 the window alone; j1 with the two windows before it; j2 j0 with unclear offered, "
                             "on coded windows only, absent ones aside (default j0)")
    parser.add_argument('--pilot', action='store_true',
                        help="dev sessions only, and a state without head-turn variability, gaze switches, hand "
                             "movement or frame shares; never scored")
    parser.add_argument('--limit', type=int, default=None,
                        help="ask about at most N windows, drawn with a fixed seed across the sessions; their "
                             "answers are added to each session's map")
    parser.add_argument('--workers', type=int, default=4, help="requests in flight at once (default 4)")
    parser.add_argument('--dry-run', action='store_true',
                        help="print three sample states, one request with the key masked, and the token and "
                             "dollar estimate; send nothing")
    parser.add_argument('--coder', default=None,
                        help="j2: whose coded windows to ask about (default: the coder with the most windows)")
    parser.add_argument('--bins', default=None,
                        help="the frozen tertile words to use (default artifacts/_analysis/interaction/jev/bins.json, "
                             "or bins_pilot.json with --pilot; fitted and frozen there on the first run)")
    parser.add_argument('--fit-bins', action='store_true',
                        help="fit the tertile words again on the dev sessions and freeze them in place of the old "
                             "ones: a new template, whose states miss the cache")
    parser.add_argument('--order-check', action='store_true', help="criteria order reversed on 200 dev windows (not built yet)")
    parser.add_argument('--rerun-check', action='store_true', help="the same 200 windows asked again past the cache (not built yet)")
    parser.add_argument('--provider', choices=sorted(J_PROVIDERS), default=J_DEFAULT_PROVIDER,
                        help="who serves Jev: openrouter (TypeSafe's model at its list price, the default) or url, any "
                             "host that speaks TypeSafe's decide API, named with --endpoint")
    parser.add_argument('--endpoint', default=None, help="the URL to POST to (required with --provider url)")
    parser.add_argument('--key-env', default=None, help="the environment variable holding the key (the provider's default otherwise)")
    parser.add_argument('--model', default=None, help="the model field of every request (the provider's default otherwise)")
    parser.add_argument('--price', type=float, default=None, help="dollars per million input tokens, for the estimate only")
    parser.add_argument('-o', '--out', default=None,
                        help="run folder for predictions.csv, bins.json and sanity.json "
                             "(default artifacts/_analysis/interaction/jev/<variant>[_pilot])")
    return parser


# ---- what ses-jev takes from the rest of the classifier ----

def _read(path: Path):
    """the fused table as the classifier reads it."""
    from openmmla.analytics.interaction.layout import read_table
    return read_table(path)


def _roster(table, session_dir=None):
    """the kept tags in slot order, the group size and the duplicate-skeleton gate (windows, slots),
    from the classifier's own roster (the pupils `session_dir`'s manifest declares, else the
    rules), so Jev's A, B and C are the persons the models see and a masked copy is not in view
    for either."""
    from openmmla.analytics.interaction.layout import duplicate_gate, session_roster
    kept = session_roster(table, session_dir)
    return [str(tag) for tag in kept.kept], int(kept.group_size), duplicate_gate(table, kept.kept)


def _test_sessions() -> set[str]:
    """the fixed test sessions: never in the bins, never in a pilot."""
    from openmmla.analytics.interaction.splits import TEST_SESSIONS
    return set(TEST_SESSIONS)


def _coded_windows(session_dir: Path, coder: str | None) -> list[float]:
    """the window starts a coder labelled in a session, absent ones aside (j2 asks about those
    only)."""
    from openmmla.analytics.interaction.labels import ABSENT, load_labels
    labels = load_labels(str(session_dir), coder=coder)
    if labels is None or len(labels) == 0:
        return []
    labels = labels[labels['label'] != ABSENT]
    return [float(start) for start in labels['window_start']]


def find_tables(artifacts: Path, pattern: str | None = None) -> list[tuple[str, Path]]:
    """(session, fused table) for every artifacts/exp_*/analysis/features/<session>_window_features.csv."""
    found = []
    for session_dir in sorted(artifacts.glob('exp_*')):
        if pattern and pattern not in session_dir.name:
            continue
        path = session_dir / 'analysis' / 'features' / f'{session_dir.name}_window_features.csv'
        if path.exists():
            found.append((session_dir.name, path))
    return found


def _included(table, slots) -> tuple[bool, str]:
    """the classifier's inclusion rule S1 on a session, from its kept slots: two persons, seen
    together in enough windows; (included, the reason). A session left out fits no bins and is
    asked about nothing, so Jev sees the sessions the models see."""
    from openmmla.analytics.interaction.layout import MIN_TWO_VISIBLE, two_visible
    if len(slots) < 2:
        return False, f"S1 the roster keeps {len(slots)} person(s); two are needed"
    share = float(two_visible(table, [int(tag) for tag in slots]).mean()) if len(table) else 0.0
    if share < MIN_TWO_VISIBLE:
        return False, f"S1 two persons observed together in {share:.2f} of the windows, under {MIN_TWO_VISIBLE:.2f}"
    return True, ''


def _load(path: Path):
    """(the slot table of a session, '', its roster digests), or (None, the reason, None) when the
    inclusion rule leaves it out. The digests are those of the roster the slots come from and of
    the rules roster (layout.roster_digest, rules_roster_digest), which say whether a Jev map was
    asked about the persons the session has now."""
    from openmmla.analytics.interaction import jev as J
    from openmmla.analytics.interaction.layout import roster_digest, rules_roster_digest
    table = _read(path)
    # the table sits at <session>/analysis/features/
    slots, group_size, gate = _roster(table, Path(path).parents[2])
    included, reason = _included(table, slots)
    if not included:
        return None, reason, None
    digests = {'roster': roster_digest(slots, group_size), 'rules': rules_roster_digest(table)}
    return J.slot_table(table, slots, group_size, vfa_mask=gate), '', digests


def _write_lines(path: Path, lines: list[dict]):
    """a session map's lines, written aside and moved into place so a crash never leaves half a map."""
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(temp, 'w', encoding='utf-8') as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + '\n')
    temp.replace(path)


def _asked_roster(line: dict, digests: dict) -> str:
    """the roster digest a map line was asked about: its own, or the rules roster's for a line
    written before ses-jev recorded one (the rules were all it read then)."""
    return line.get('roster_sha256') or digests['rules']


def drop_stale_answers(session: dict, variant: str, pilot: bool) -> int:
    """drop the lines of a session's Jev map that were asked about another roster than the one it
    has now (it has since declared its pupils, see mmla ses-tidy --pupils), before a run adds to
    the map: a --limit run keeps the lines of the windows it does not draw, and those would mix two
    rosters in one map. A map left with no line is removed. Returns how many lines were dropped."""
    from openmmla.analytics.interaction import jev as J
    if not session.get('out_dir'):
        return 0
    path = J.session_map_path(session['out_dir'], variant, pilot)
    lines = J.read_session_map(path)
    keep = [line for line in lines if _asked_roster(line, session['roster_digests']) == session['roster_sha256']]
    if len(keep) == len(lines):
        return 0
    if keep:
        _write_lines(path, keep)
    else:
        path.unlink()
    return len(lines) - len(keep)


def stamp_roster(session: dict, variant: str, pilot: bool):
    """write the session's roster digest on every line of its Jev map that has none: the lines
    this run added, and the earlier ones drop_stale_answers kept."""
    from openmmla.analytics.interaction import jev as J
    if not session.get('out_dir'):
        return
    path = J.session_map_path(session['out_dir'], variant, pilot)
    lines = J.read_session_map(path)
    if lines and any(line.get('roster_sha256') != session['roster_sha256'] for line in lines):
        _write_lines(path, [dict(line, roster_sha256=session['roster_sha256']) for line in lines])


def _frozen_bins(path: Path):
    """(Bins, the sessions they were fitted on) from a frozen bins.json; (None, the quantities it
    lacks) when it does not hold an edge entry for every tertile quantity."""
    from openmmla.analytics.interaction import jev as J
    record = json.loads(path.read_text(encoding='utf-8'))
    missing = [name for name in J.TERTILES if name not in (record.get('edges') or {})]
    if missing:
        return None, missing
    return J.Bins.from_dict(record), record.get('fitted_on') or []


def _sessions_text(n: int) -> str:
    return f"{n} session{'' if n == 1 else 's'}"


def _samples(asks: list[dict]) -> list[tuple[str, dict]]:
    """the longest, a middle and the shortest state: the fullest wording and the most 'not measured'."""
    ordered = sorted(asks, key=lambda ask: (-len(ask['state']), ask['session'], ask['window_index']))
    picks = [('longest', ordered[0]), ('median', ordered[len(ordered) // 2]), ('shortest', ordered[-1])]
    seen, out = set(), []
    for name, ask in picks:
        if id(ask) not in seen:
            seen.add(id(ask))
            out.append((name, ask))
    return out


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.order_check or args.rerun_check:
        parser.error("--order-check and --rerun-check are not built yet")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    import pandas as pd
    from openmmla.analytics.interaction import jev as J
    try:
        provider = J.provider_settings(args.provider, url=args.endpoint, key_env=args.key_env, model=args.model,
                                       price_per_token=args.price / 1e6 if args.price else None)
    except ValueError as error:
        parser.error(str(error))
    # the key is looked up and kept here only, and handed to the client; nothing else sees it
    api_key = os.environ.get(provider['key_env'], '').strip()
    if not api_key and not args.dry_run:
        parser.error(f"set {provider['key_env']} in the environment to call Jev (or use --dry-run to send nothing)")
    print(f"provider {args.provider}: {provider['url']} model {provider['model']} (key from {provider['key_env']})")

    artifacts = Path(args.artifacts or os.path.join(os.getcwd(), 'artifacts')).resolve()
    everything = find_tables(artifacts)
    test = _test_sessions()
    chosen = [(s, p) for s, p in everything if (not args.sessions or args.sessions in s) and not (args.pilot and s in test)]
    if not chosen:
        print(f"no fused tables under {artifacts} for {args.sessions or 'any session'}"
              f"{' outside the test sessions' if args.pilot else ''}")
        return 1

    bins_path = Path(args.bins) if args.bins else \
        artifacts / '_analysis' / 'interaction' / 'jev' / f"bins{'_pilot' if args.pilot else ''}.json"
    frozen = bins_path.exists() and not args.fit_bins
    if args.bins and not bins_path.exists() and not args.fit_bins:
        parser.error(f"no bins at {bins_path}: give the bins.json a run froze, or --fit-bins to fit them there")
    slot_tables, left_out, digests = {}, {}, {}
    for session, path in everything:
        slots, reason, digests[session] = _load(path)
        if slots is None:
            left_out[session] = reason
            print(f"{session} left out: {reason}")
        else:
            slot_tables[session] = slots
    chosen = [(s, p) for s, p in chosen if s not in left_out]
    if not chosen:
        print("every chosen session is left out by the inclusion rule S1")
        return 1
    if frozen:
        bins, fitted_on = _frozen_bins(bins_path)
        if bins is None:
            print(f"{bins_path} holds no tertile edges for {', '.join(fitted_on)}: fit them again with --fit-bins")
            return 1
        print(f"tertile words frozen in {bins_path} (template {bins.digest()[:12]})")
    else:
        # the words are fitted on every dev session the rule kept, whichever sessions are asked about
        dev_tables = {session: slots for session, slots in slot_tables.items() if session not in test}
        if not dev_tables:
            print(f"no dev session under {artifacts} to fit the tertile words on")
            return 1
        bins, fitted_on = J.fit_bins(list(dev_tables.values())), sorted(dev_tables)

    from openmmla.utils.session_provenance import file_digest
    sessions = []
    for session, path in chosen:
        slots = slot_tables[session]
        entry = {'session': session, 'slots': slots, 'table_sha256': file_digest(path),
                 'out_dir': str(artifacts / session / 'analysis' / 'interaction'),
                 'roster_sha256': digests[session]['roster'], 'roster_digests': digests[session]}
        if args.variant == 'j2':
            entry['only'] = _coded_windows(artifacts / session, args.coder)
        sessions.append(entry)

    cache = J.JevCache(artifacts / '_analysis' / 'interaction' / 'jev' / 'cache')
    kind = f"{args.variant}{', pilot' if args.pilot else ''}"
    if args.dry_run:
        if not frozen:
            print(f"tertile words fitted now on {_sessions_text(len(fitted_on))} (template {bins.digest()[:12]}); "
                  f"a real run freezes them in {bins_path}")
        asks = J.build_requests(sessions, args.variant, bins, pilot=args.pilot, limit=args.limit, model=provider['model'])
        if not asks:
            print(f"no windows to ask about ({kind})")
            return 1
        samples = _samples(asks)
        for name, ask in samples:
            print(f"--- sample state ({name}): {ask['session']} window {ask['window_index']} ---")
            print(ask['state'])
        print("--- the request, key masked ---")
        print(json.dumps(J.redacted_request(samples[0][1]['body'], provider['url']), indent=2, ensure_ascii=False))
        uncached = [ask['body'] for ask in asks if not cache.has(ask['body'])]
        cost = J.estimate_cost(uncached, provider['price_per_token'])
        everything_cost = J.estimate_cost([ask['body'] for ask in asks], provider['price_per_token'])
        print(f"{len(asks)} windows of {_sessions_text(len(sessions))} ({kind}); {everything_cost['unique']} distinct "
              f"requests, {everything_cost['unique'] - cost['unique']} already cached")
        print(f"estimate: {cost['input_tokens']:,} input tokens for {cost['unique']} calls, about "
              f"${cost['dollars']:.2f} at ${provider['price_per_token'] * 1e6:.3f} per million (output not priced); nothing sent")
        return 0

    if not frozen:
        replaced = bins_path.exists()
        bins_path.parent.mkdir(parents=True, exist_ok=True)
        bins_path.write_text(json.dumps(dict(bins.to_dict(), fitted_on=fitted_on, sha256=bins.digest()), indent=2),
                             encoding='utf-8')
        print(f"tertile words fitted on {_sessions_text(len(fitted_on))} and frozen in {bins_path} "
              f"(template {bins.digest()[:12]}"
              + ("; they replace the earlier ones, so every state is new and misses the cache)" if replaced else ")"))
    client = J.JevClient(api_key, workers=args.workers, url=provider['url'], key_env=provider['key_env'])

    def progress(done, total):
        if done % PROGRESS_EVERY == 0 or done == total:
            print(f"asked {done} of {total}")

    for session in sessions:
        dropped = drop_stale_answers(session, args.variant, args.pilot)
        if dropped:
            print(f"{session['session']}: {dropped} answers in its {J.session_map_path('', args.variant, args.pilot).name} "
                  f"were asked about another roster and are dropped")
    predictions = J.run_jev(sessions, args.variant, bins, client=client, cache=cache, limit=args.limit,
                            pilot=args.pilot, progress=progress, model=provider['model'])
    for session in sessions:
        stamp_roster(session, args.variant, args.pilot)
    run_dir = Path(args.out) if args.out else \
        artifacts / '_analysis' / 'interaction' / 'jev' / f"{args.variant}{'_pilot' if args.pilot else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(run_dir / 'predictions.csv', index=False)
    (run_dir / 'bins.json').write_text(json.dumps(dict(bins.to_dict(), fitted_on=fitted_on, sha256=bins.digest(),
                                                       frozen_in=str(bins_path)), indent=2), encoding='utf-8')
    pooled = pd.concat([J.sanity_view(s['session'], s['slots']) for s in sessions], ignore_index=True)
    answered = predictions.dropna(subset=['p_interaction']) if len(predictions) else predictions
    check = J.sanity(answered, pooled) if len(answered) else {'speech_ratio': None, 'partner_gaze': None, 'positive': False}
    (run_dir / 'sanity.json').write_text(json.dumps(check, indent=2), encoding='utf-8')

    errors = int(predictions['error'].notna().sum()) if len(predictions) else 0
    cached = int(predictions['cached'].sum()) if len(predictions) else 0
    print(f"{len(predictions)} windows of {_sessions_text(len(sessions))} ({kind}): {cached} from the cache, "
          f"{len(predictions) - cached - errors} answered now, {errors} not answered -> {run_dir}")
    # what this run paid: each request asked now counts once, however many windows share its state
    paid = predictions[~predictions['cached'] & predictions['error'].isna()].drop_duplicates('hash') \
        if len(predictions) else predictions
    tokens = paid['input_tokens'].sum(min_count=1) if len(paid) else None
    if tokens is not None and pd.notna(tokens):
        print(f"usage: {int(tokens):,} input tokens reported for {len(paid)} calls, about "
              f"${tokens * J.PRICE_PER_TOKEN:.2f}")
    if len(answered):
        print(f"sanity: Spearman of p_interaction with speech {check['speech_ratio']['rho']}, with partner-directed "
              f"gaze {check['partner_gaze']['rho']} ({'both positive' if check['positive'] else 'NOT both positive'})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
