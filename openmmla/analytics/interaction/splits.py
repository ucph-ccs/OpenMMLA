"""Which sessions train and which score the interaction classifier.

Four sessions are the TEST set, touched once after every choice is frozen: they never reach a
global scaler, the Jev bins, pretraining or prompt work, and no dev split here ever contains them.
The other 17 are DEV. The unit of a split is the lesson (`lesson_key`: the session id without its
start suffix), since the two 2025-05-13 takes are one group in one lesson and would leak into each
other; that gives 16 leave-one-lesson-out folds, each with inner folds grouped the same way.

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


def _leave_out(sessions, key, coded, prefix: str) -> list:
    dev = dev_sessions(sessions)
    held = _coded(dev, coded)
    folds = []
    for value in sorted({key(session) for session in dev if session in held}):
        folds.append(Fold(f'{prefix}{value}', [s for s in dev if key(s) != value], [s for s in dev if key(s) == value]))
    return folds


def loso_folds(sessions, coded=None) -> list:
    """leave one lesson out over the DEV sessions: one fold per lesson with coded windows, holding
    out all its sessions and training on every other DEV session. TEST sessions are left out."""
    return _leave_out(sessions, lesson_key, coded, '')


def date_folds(sessions, coded=None) -> list:
    """leave one date out over the DEV sessions (exploratory): same-day sessions may share
    students and the rig, so a date is held out whole."""
    return _leave_out(sessions, date_of, coded, 'date-')


def task_transfer(sessions, source: str = 'microbit', target: str = 'wegrow') -> list:
    """train on the DEV sessions of one task and score those of the other (exploratory): one fold."""
    dev = dev_sessions(sessions)
    return [Fold(f'{source}->{target}', [s for s in dev if task_of(s) == source],
                 [s for s in dev if task_of(s) == target])]


def final_fold(sessions) -> Fold:
    """the one scoring of the TEST sessions: train on every DEV session."""
    return Fold('test', dev_sessions(sessions), [s for s in TEST_SESSIONS if s in set(sessions)])


def inner_folds(sessions, k: int = 4, sizes=None) -> list:
    """GroupKFold(k) over the lessons of an outer-training set: each lesson with coded windows is
    held out in exactly one inner fold, the heaviest lessons first into the lightest fold (as
    sklearn's GroupKFold does), weighed by `sizes` (session -> coded windows; each session counts
    1 when None). A lesson with nothing coded always trains. k shrinks to the number of lessons
    that can be held out; fewer than two raise."""
    dev = dev_sessions(sessions)
    weight = {s: (1 if sizes is None else (sizes.get(s) or 0)) for s in dev}
    lessons: dict[str, float] = {}
    for session in dev:
        lessons[lesson_key(session)] = lessons.get(lesson_key(session), 0) + weight[session]
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
    return [Fold(f'inner-{n}', [s for s in dev if lesson_key(s) not in group], [s for s in dev if lesson_key(s) in group])
            for n, group in enumerate(groups)]


def fold_indices(fold: Fold, session_of_rows) -> tuple[np.ndarray, np.ndarray]:
    """the row positions of a fold's training and held-out sessions, from each row's session."""
    rows = np.asarray(session_of_rows)
    return np.flatnonzero(np.isin(rows, list(fold.train))), np.flatnonzero(np.isin(rows, list(fold.test)))
