"""The windows a locked coding campaign's coders code: a stratified, seeded sample (mmla ses-code --draw-assignment).

The population is the windows of the campaign's sessions (their full grid, code.windows_of) that the named
strata coders coded in the default page: by at least two of them, or by the one when one is named, and with
--assign-filter non-unanimous only the windows they coded differently (the blind adjudication set). Their
labels files are read as they stand now (code.replay); a file a model or a script wrote (code.file_kind) is
refused, so only a person's labels define the strata. A window's stratum joins the keys asked for: `lesson`
(splits.lesson_key), `majority` (the code more than half of the coders who coded the window gave it, else
none) and `unanimity` (unanimous or split).

N windows are allocated to the strata in proportion to their size, each with at least min(--min-per-stratum,
its size), by largest remainder to exactly N; `all` takes the whole population. Within a stratum a simple
random sample without replacement is drawn with random.Random(f"{seed}:{campaign}:{stratum}").

assignment.json (served: which windows, nothing about why) holds one set, `main`, for every coder;
design.json (never served) holds each stratum's population, sample and weight (population / sample), the
coders, the labels files with their sha256, the time they were read, the seed and the filter.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from openmmla.commands.ses import code as C

STRATA = ('lesson', 'majority', 'unanimity')


def lesson_of(session_id: str) -> str:
    """the lesson of a session (splits.lesson_key), or its id when the analytics package is missing"""
    try:
        from openmmla.analytics.interaction.splits import lesson_key
    except Exception:
        return session_id
    return lesson_key(session_id)


def read_coders(sessions: list[dict], coders: list[str], artifacts: Path | None = None) -> tuple[dict, dict]:
    """({session id: {coder: {window key: label}}}, {labels file: sha256}) of the named coders' default-mode
    files as they stand now; a model's file is refused (ValueError)"""
    labels: dict[str, dict[str, dict[str, str]]] = {}
    files: dict[str, str] = {}
    for session in sessions:
        labels[session['id']] = {}
        for coder in coders:
            path = Path(session['dir']) / 'labels' / f'{coder}.jsonl'
            if not path.is_file():
                continue
            raw = path.read_bytes()
            text = raw.decode('utf-8', errors='replace')
            if C.file_kind(text) == 'model':
                raise ValueError(f"{coder!r} holds a model's labels in {session['id']}: the strata are drawn from people's labels")
            current, _ = C.replay(text)
            labels[session['id']][coder] = {k: r['label'] for k, r in current.items() if r.get('label') in C.CODES}
            name = path.relative_to(artifacts).as_posix() if artifacts else str(path)
            files[name] = hashlib.sha256(raw).hexdigest()
    return labels, files


def stratum_of(codes: list[str], lesson: str, strata: tuple[str, ...]) -> str:
    """the stratum key of a window the coders gave `codes`"""
    parts = []
    for name in strata:
        if name == 'lesson':
            parts.append(f'lesson={lesson}')
        elif name == 'majority':
            counts = {c: codes.count(c) for c in set(codes)}
            top = [c for c, n in counts.items() if n * 2 > len(codes)]
            parts.append(f"majority={top[0] if top else 'none'}")
        elif name == 'unanimity':
            parts.append(f"unanimity={'unanimous' if len(set(codes)) == 1 else 'split'}")
    return '|'.join(parts) or 'all'


def allocate(sizes: dict[str, int], n: int, floor: int) -> dict[str, int]:
    """n draws over the strata: proportional to size, at least min(floor, size) each, by largest remainder
    to exactly n (all of every stratum when n reaches the population); ValueError when the floors exceed n"""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    floors = {k: min(floor, s) for k, s in sizes.items()}
    if sum(floors.values()) > n:
        raise ValueError(f'{len(sizes)} strata need at least {sum(floors.values())} windows for their floors, '
                         f'more than the {n} asked for: lower --min-per-stratum or draw more')
    quota = {k: n * s / total for k, s in sizes.items()}
    given = {k: min(sizes[k], max(floors[k], int(quota[k]))) for k in sizes}
    while sum(given.values()) < n:
        room = [k for k in sizes if given[k] < sizes[k]]
        k = max(room, key=lambda k: (quota[k] - given[k], sizes[k], k))
        given[k] += 1
    while sum(given.values()) > n:
        over = [k for k in sizes if given[k] > floors[k]]
        k = min(over, key=lambda k: (quota[k] - given[k], -sizes[k], k))
        given[k] -= 1
    return given


def draw_assignment(campaign: str, sessions: list[dict], window: float, step: float, coders: list[str],
                    n: int | None, strata: tuple[str, ...] = ('lesson', 'majority'), population_filter: str = 'all',
                    min_per_stratum: int = 3, seed: int = 1, artifacts: Path | None = None) -> tuple[dict, dict]:
    """(assignment, design) of a campaign (see the module's docstring); `n` None takes the whole population"""
    if not coders:
        raise ValueError('name the coders whose labels define the strata (--strata-coders)')
    for coder in coders:
        if C.safe_name(coder) != coder or coder.lower() == C.ADJUDICATED:
            raise ValueError(f'{coder!r} is not a coder name')
    unknown = [s for s in strata if s not in STRATA]
    if unknown:
        raise ValueError(f"unknown stratum {unknown[0]!r}: lesson, majority or unanimity")
    if population_filter not in ('all', 'non-unanimous'):
        raise ValueError(f'unknown filter {population_filter!r}')
    if n is not None and n < 1:
        raise ValueError('draw at least one window')
    asof = C.now_utc()
    labels, files = read_coders(sessions, coders, artifacts)
    need = 1 if len(coders) == 1 else 2
    population: dict[str, list[tuple[int, str, float]]] = {}
    for order, session in enumerate(sessions):
        lesson = lesson_of(session['id'])
        by = labels[session['id']]
        for w in C.windows_of(session, window, step, 1.0, 300.0, seed):
            key = f"{w['start']:.3f}"
            codes = [by[c][key] for c in coders if c in by and key in by[c]]
            if len(codes) < need:
                continue
            if population_filter == 'non-unanimous' and len(set(codes)) == 1:
                continue
            population.setdefault(stratum_of(codes, lesson, strata), []).append((order, session['id'], w['start']))
    size = sum(len(v) for v in population.values())
    if not size:
        raise ValueError('no window of the campaign was coded by the strata coders as asked: nothing to draw from')
    sizes = {k: len(v) for k, v in population.items()}
    given = allocate(sizes, size if n is None else n, min_per_stratum)
    drawn, design = [], {}
    for stratum in sorted(population):
        members = sorted(population[stratum])
        chosen = random.Random(f'{seed}:{campaign}:{stratum}').sample(members, given[stratum])
        drawn += chosen
        design[stratum] = {'population': len(members), 'sampled': len(chosen),
                           'weight': len(members) / len(chosen) if chosen else None}
    drawn.sort()
    assignment = {'version': 1, 'campaign': campaign, 'window': window, 'step': step,
                  'sets': {'main': [{'session': sid, 'window_start': start} for _, sid, start in drawn]},
                  'coders': {'*': 'main'}}
    return assignment, {'version': 1, 'campaign': campaign, 'strata': design, 'strata_by': list(strata),
                        'strata_coders': list(coders), 'filter': population_filter, 'n': n, 'population': size,
                        'min_per_stratum': min_per_stratum, 'seed': seed, 'asof': asof, 'label_files': files}
