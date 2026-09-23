"""Which sessions train and which score the interaction classifier.

Four sessions are the TEST set, touched once after every choice is frozen: they never reach a
global scaler, the Jev bins, pretraining or prompt work, and no dev split here ever contains them.
The other 17 are DEV. The unit of a split is the lesson (`lesson_key`: the session id without its
start suffix), since the two 2025-05-13 takes are one group in one lesson and would leak into each
other; that gives 16 leave-one-lesson-out folds, each with inner folds grouped the same way.

Sessions of different pupils from one school class are not independent either: a session
manifest's `same_class_as` (a list of session ids, written by mmla ses-tidy --same-class-as) links
them, and every split merges the lessons those links reach, transitively, into one unit named by
its lessons joined with '+'. A link between a TEST and a DEV session is refused (SplitError) in
every split: the one scoring of TEST would train on that class's other pupils.

A fold is (name, train, test), lists of session ids. Only lessons with coded windows are held out;
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
    """the same_class_as links would put a TEST session's class on the training side."""


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


def linked(same_class, session) -> list:
    """the session ids `same_class` (session -> its manifest's same_class_as, a list or one id)
    links a session to."""
    if not same_class:
        return []
    others = same_class.get(session) or ()
    if isinstance(others, str):
        others = [others]
    return [str(other) for other in others if other and str(other) != session]


def check_links(sessions, same_class=None) -> None:
    """raise SplitError when a session links a TEST session to a DEV one: that class's pupils
    would train the model the TEST sessions are scored with, so no split can hold them apart."""
    for session in sessions:
        for other in linked(same_class, session):
            if (session in TEST_SESSIONS) != (other in TEST_SESSIONS):
                test, dev = (session, other) if session in TEST_SESSIONS else (other, session)
                raise SplitError(f"TEST session {test} is marked same_class_as DEV session {dev}: the TEST "
                                 f"scoring would train on that class; drop the link or move a session")


def _merged(sessions, key, same_class=None) -> dict:
    """session -> its `key` value, merged with the values of every session a same_class_as link
    reaches, transitively (also through a linked session not among `sessions`); a merged value is
    the values of the given sessions joined with '+', sorted."""
    parent: dict = {}

    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def value_of(session):
        return key(session) or session

    for session in sessions:
        find(value_of(session))
        for other in linked(same_class, session):
            first, second = find(value_of(session)), find(value_of(other))
            if first != second:
                parent[second] = first
    members: dict = {}
    for session in sessions:
        members.setdefault(find(value_of(session)), set()).add(value_of(session))
    return {session: '+'.join(sorted(members[find(value_of(session))])) for session in sessions}


def class_units(sessions, same_class=None) -> dict:
    """session -> the unit every split holds out whole: its lesson, merged with the lessons its
    same_class_as links reach (exp_20250520_microbit_group_01+exp_20250520_microbit_group_02)."""
    return _merged(sessions, lesson_key, same_class)


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


def _leave_out(sessions, key, coded, prefix: str, same_class=None) -> list:
    check_links(sessions, same_class)
    dev = dev_sessions(sessions)
    held = _coded(dev, coded)
    unit = _merged(dev, key, same_class)
    folds = []
    for value in sorted({unit[session] for session in dev if session in held}):
        folds.append(Fold(f'{prefix}{value}', [s for s in dev if unit[s] != value], [s for s in dev if unit[s] == value]))
    return folds


def loso_folds(sessions, coded=None, same_class=None) -> list:
    """leave one lesson out over the DEV sessions: one fold per lesson with coded windows, holding
    out all its sessions and training on every other DEV session. TEST sessions are left out.
    Lessons linked by `same_class` (session -> its same_class_as) are one fold, named by both."""
    return _leave_out(sessions, lesson_key, coded, '', same_class)


def date_folds(sessions, coded=None, same_class=None) -> list:
    """leave one date out over the DEV sessions (exploratory): same-day sessions may share
    students and the rig, so a date is held out whole, with the dates of its linked classes."""
    return _leave_out(sessions, date_of, coded, 'date-', same_class)


def task_transfer(sessions, source: str = 'microbit', target: str = 'wegrow') -> list:
    """train on the DEV sessions of one task and score those of the other (exploratory): one fold."""
    dev = dev_sessions(sessions)
    return [Fold(f'{source}->{target}', [s for s in dev if task_of(s) == source],
                 [s for s in dev if task_of(s) == target])]


def final_fold(sessions, same_class=None) -> Fold:
    """the one scoring of the TEST sessions: train on every DEV session (refused with SplitError
    when a same_class_as link joins a TEST session to a DEV one)."""
    check_links(sessions, same_class)
    return Fold('test', dev_sessions(sessions), [s for s in TEST_SESSIONS if s in set(sessions)])


def inner_folds(sessions, k: int = 4, sizes=None, same_class=None) -> list:
    """GroupKFold(k) over the lessons of an outer-training set: each lesson with coded windows is
    held out in exactly one inner fold, the heaviest lessons first into the lightest fold (as
    sklearn's GroupKFold does), weighed by `sizes` (session -> coded windows; each session counts
    1 when None). A lesson with nothing coded always trains. k shrinks to the number of lessons
    that can be held out; fewer than two raise. Lessons linked by `same_class` count as one."""
    dev = dev_sessions(sessions)
    unit = class_units(dev, same_class)
    weight = {s: (1 if sizes is None else (sizes.get(s) or 0)) for s in dev}
    lessons: dict[str, float] = {}
    for session in dev:
        lessons[unit[session]] = lessons.get(unit[session], 0) + weight[session]
    held = [lesson for lesson, size in lessons.items() if size > 0]
    if len(held) < 2:
        raise ValueError(f"inner folds need at least 2 lessons with coded windows, got {len(held)}")
    k = min(k, len(held))
    groups: list[list[str]] = [[] for _ in range(k)]
    totals = [0.0] * k
    for lesson in sorted(held, key=lambda lesson: (-lessons[lesson], lesson)):
        fold = int(np.argmin(totals))
        groups[fold].append(lesson)
        totals[fold] += lessons[lesson]
    return [Fold(f'inner-{n}', [s for s in dev if unit[s] not in group], [s for s in dev if unit[s] in group])
            for n, group in enumerate(groups)]


def fold_indices(fold: Fold, session_of_rows) -> tuple[np.ndarray, np.ndarray]:
    """the row positions of a fold's training and held-out sessions, from each row's session."""
    rows = np.asarray(session_of_rows)
    return np.flatnonzero(np.isin(rows, list(fold.train))), np.flatnonzero(np.isin(rows, list(fold.test)))
