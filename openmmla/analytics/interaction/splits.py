"""Which sessions train and which score the interaction classifier.

Four sessions are the TEST set, touched once after every choice is frozen: they never reach a
global scaler, the Jev bins, pretraining or prompt work, and no dev split here ever contains them.
The other sessions are DEV.

The unit of every split is the date: the same group on the same date is never on both sides of a
split. Sessions of one date share pupils or a class (a micro:bit and a WeGrow lesson of group_01 on
one morning, two groups of one class, the two 2025-05-13 takes of one group), so all DEV sessions
of a date are held out together and trained on together, in the outer folds, in the inner folds,
in task transfer and in the bootstrap. With 2025-05-20 group_02 voided, the DEV set is 16 sessions
in 15 lessons on 9 dates, so 9 leave-one-date-out folds.

Sessions of one school class on different dates are not independent either: a session manifest's
`same_class_as` (a list of session ids, written by mmla ses-tidy --same-class-as) links them, and
every split merges the dates those links reach, transitively, into one unit. A unit is named by its
lessons (`lesson_key`: the session id without its start suffix) joined with '+'.

A DEV session whose unit reaches a TEST session, by sharing its date or through same_class_as links,
is refused (SplitError) in every split: the one scoring of TEST would train on those pupils.

A fold is (name, train, test), lists of session ids. Only units with coded windows are held out;
a session without labels still trains, since the label-free fits (scalers) read every window.
"""
from __future__ import annotations

import re
from collections import namedtuple

import numpy as np

TEST_SESSIONS = (
    'exp_20260520_microbit_group_01_260520T0834Z',
    'exp_20260520_wegrow_group_01_260520T1054Z',
    'exp_20260603_microbit_group_01_260603T0826Z',
    'exp_20260603_wegrow_group_01_260603T1037Z',
)
LESSON_RE = re.compile(r'^(exp_\d{8}_(?:microbit|wegrow)_group_\d+)_')
TASK_RE = re.compile(r'^exp_\d{8}_(microbit|wegrow)_')
DATE_RE = re.compile(r'^exp_(\d{8})_')

Fold = namedtuple('Fold', 'name train test')


class SplitError(ValueError):
    """a DEV session shares a date, or a same_class_as link, with a TEST session."""


def lesson_key(session_id: str) -> str:
    """the lesson a session belongs to: its id without the start suffix, so two takes of one
    group on one day are one lesson; an id of another shape is its own lesson."""
    match = LESSON_RE.match(session_id)
    return match.group(1) if match else session_id


def task_of(session_id: str) -> str | None:
    """'microbit' or 'wegrow', from the id; None when it names neither."""
    match = TASK_RE.match(session_id)
    return match.group(1) if match else None


def date_of(session_id: str) -> str | None:
    """the session's date (YYYYMMDD), from the id."""
    match = DATE_RE.match(session_id)
    return match.group(1) if match else None


def unit_key(session_id: str) -> str:
    """what a split groups a session by before links: its date; an id without one is its own."""
    return date_of(session_id) or session_id


def linked(same_class, session) -> list:
    """the session ids `same_class` (session -> its manifest's same_class_as, a list or one id)
    links a session to."""
    if not same_class:
        return []
    others = same_class.get(session) or ()
    if isinstance(others, str):
        others = [others]
    return [str(other) for other in others if other and str(other) != session]


def _joined(sessions, same_class=None, extra=()):
    """a function giving each session the root of its unit: its date, joined with the dates the
    same_class_as links of `sessions` reach, transitively (also through a linked session that is
    not among them). `extra` sessions take part without links of their own."""
    parent: dict = {}

    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for session in list(sessions) + list(extra):
        find(unit_key(session))
    for session in sessions:
        for other in linked(same_class, session):
            first, second = find(unit_key(session)), find(unit_key(other))
            if first != second:
                parent[second] = first
    return lambda session: find(unit_key(session))


def check_links(sessions, same_class=None) -> None:
    """raise SplitError when a DEV session's unit reaches a TEST session: it shares the TEST
    session's date, a same_class_as link joins the two, or links reach one from the other through
    other dates. Those pupils would train the model the TEST sessions are scored with, so no split
    can hold them apart. TEST sessions linked or dated only among themselves are fine."""
    test_dates = {date_of(test): test for test in TEST_SESSIONS}
    for session in sessions:
        if session not in TEST_SESSIONS and date_of(session) in test_dates:
            raise SplitError(f"DEV session {session} is on the date of TEST session {test_dates[date_of(session)]}: "
                             f"the TEST scoring would train on the same pupils; void or move a session")
        for other in linked(same_class, session):
            if (session in TEST_SESSIONS) != (other in TEST_SESSIONS):
                test, dev = (session, other) if session in TEST_SESSIONS else (other, session)
                raise SplitError(f"TEST session {test} is marked same_class_as DEV session {dev}: the TEST "
                                 f"scoring would train on that class; drop the link or move a session")
    root = _joined(sessions, same_class, extra=TEST_SESSIONS)
    for session in sessions:
        if session in TEST_SESSIONS:
            continue
        for test in TEST_SESSIONS:
            if root(session) == root(test):
                raise SplitError(f"DEV session {session} reaches TEST session {test} through same_class_as links: "
                                 f"the TEST scoring would train on that class; drop a link or move a session")


def class_units(sessions, same_class=None) -> dict:
    """session -> the unit every split holds out whole: its date, merged with the dates its
    same_class_as links reach, named by the lessons of the given sessions in it joined with '+'
    (exp_20241210_microbit_group_01+exp_20241210_microbit_group_02)."""
    sessions = list(sessions)
    root = _joined(sessions, same_class)
    members: dict = {}
    for session in sessions:
        members.setdefault(root(session), set()).add(lesson_key(session))
    return {session: '+'.join(sorted(members[root(session)])) for session in sessions}


def dev_sessions(sessions) -> list:
    """the sessions that are not TEST, in the order given, each once."""
    seen, out = set(), []
    for session in sessions:
        if session not in TEST_SESSIONS and session not in seen:
            seen.add(session)
            out.append(session)
    return out


def _coded(sessions, coded) -> set:
    """the sessions with coded windows: all of them when `coded` is None, else those a mapping
    counts above 0, or those an iterable names."""
    if coded is None:
        return set(sessions)
    if hasattr(coded, 'items'):
        return {session for session, count in coded.items() if count and count > 0}
    return set(coded)


def date_folds(sessions, coded=None, same_class=None) -> list:
    """leave one date out over the DEV sessions, the outer split: one fold per unit (a date, with
    the dates its same_class_as links reach) with coded windows, holding out all its sessions and
    training on every other DEV session. TEST sessions are left out; a DEV session that shares a
    date or a class with one raises SplitError."""
    check_links(sessions, same_class)
    dev = dev_sessions(sessions)
    held = _coded(dev, coded)
    unit = class_units(dev, same_class)
    folds = []
    for value in sorted({unit[session] for session in dev if session in held}):
        folds.append(Fold(value, [s for s in dev if unit[s] != value], [s for s in dev if unit[s] == value]))
    return folds


def task_transfer(sessions, source: str = 'microbit', target: str = 'wegrow', same_class=None) -> list:
    """train on the DEV sessions of one task and score those of the other (exploratory): one fold.
    A source session whose unit holds a target session does not train, since the same group on the
    same date would be on both sides; today every WeGrow date has a micro:bit lesson, so micro:bit
    -> WeGrow trains on the micro:bit sessions of the other dates only."""
    check_links(sessions, same_class)
    dev = dev_sessions(sessions)
    unit = class_units(dev, same_class)
    test = [s for s in dev if task_of(s) == target]
    scored = {unit[s] for s in test}
    return [Fold(f'{source}->{target}', [s for s in dev if task_of(s) == source and unit[s] not in scored], test)]


def final_fold(sessions, same_class=None) -> Fold:
    """the one scoring of the TEST sessions: train on every DEV session (refused with SplitError
    when a DEV session shares a date or a same_class_as link with a TEST one)."""
    check_links(sessions, same_class)
    return Fold('test', dev_sessions(sessions), [s for s in TEST_SESSIONS if s in set(sessions)])


def inner_folds(sessions, k: int = 4, sizes=None, same_class=None) -> list:
    """GroupKFold(k) over the units (dates, with their same_class_as links) of an outer-training
    set: each unit with coded windows is held out in exactly one inner fold, the heaviest units
    first into the lightest fold (as sklearn's GroupKFold does), weighed by `sizes` (session ->
    coded windows; each session counts 1 when None). A unit with nothing coded always trains. k
    shrinks to the number of units that can be held out; fewer than two raise."""
    dev = dev_sessions(sessions)
    unit = class_units(dev, same_class)
    weight = {s: (1 if sizes is None else (sizes.get(s) or 0)) for s in dev}
    units: dict[str, float] = {}
    for session in dev:
        units[unit[session]] = units.get(unit[session], 0) + weight[session]
    held = [name for name, size in units.items() if size > 0]
    if len(held) < 2:
        raise ValueError(f"inner folds need at least 2 dates with coded windows, got {len(held)}")
    k = min(k, len(held))
    groups: list[list[str]] = [[] for _ in range(k)]
    totals = [0.0] * k
    for name in sorted(held, key=lambda name: (-units[name], name)):
        fold = int(np.argmin(totals))
        groups[fold].append(name)
        totals[fold] += units[name]
    return [Fold(f'inner-{n}', [s for s in dev if unit[s] not in group], [s for s in dev if unit[s] in group])
            for n, group in enumerate(groups)]


def fold_indices(fold: Fold, session_of_rows) -> tuple[np.ndarray, np.ndarray]:
    """the row positions of a fold's training and held-out sessions, from each row's session."""
    rows = np.asarray(session_of_rows)
    return np.flatnonzero(np.isin(rows, list(fold.train))), np.flatnonzero(np.isin(rows, list(fold.test)))
