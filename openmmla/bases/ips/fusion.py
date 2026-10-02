"""How the IPS synchronizer combines the cameras' raw detections of a time bucket.

Every base publishes the raw pose of each tag it detects, in its own camera's frame. The
synchronizer takes each detection into the main camera's frame with that camera's matrix and,
per tag and bucket:

- keeps one detection per camera: the only one (a file replay reads one frame a second), else,
  for a live camera that sent several frames in the bucket, the one nearest the median of their
  positions (`n` says how many there were);
- fuses the cameras: the positions within `gate` metres of their median are averaged, weighted by
  1 / distance^4 from the camera that saw them (a tag's depth error, the larger part of its error,
  grows with the square of its distance, so its variance with the fourth power: the weight is the
  inverse variance); a camera further off is left out (`out`), and when none lies within the gate the
  best camera is taken alone (two cameras that disagree have no median between them worth
  having). The best camera is the nearest to the tag, the decoder's decision margin breaking a
  tie; the fused rotation is its rotation, as averaging two readings of a small tag's rotation can
  give one neither camera saw;
- says who faces whom: a camera's own view of the pair (both tags on its frames, their raw poses)
  when it saw the facing on at least half of the frames that held both tags, or the fused poses in
  the main camera's frame.

Nothing is carried from one bucket to the next.
"""
from __future__ import annotations

import math

import numpy as np

from .vector import is_tag_looking_at_another_2d

DEFAULT_GATE = 0.25  # metres from the cameras' median position a camera may lie and still be averaged
NEAR_LIMIT = 0.5  # metres: a tag nearer its camera than this weighs as much as one at this distance
FACING_COSINE = -0.94  # is_tag_looking_at_another_2d's cosine threshold, as the bases use it
FACING_DISTANCE = 1.2  # metres on the main camera's x-z plane, as the synchronizer always used
CAMERA_VOTE = 0.5  # a camera says a tag faces another when it saw it on at least this share of the frames with both


def place(R, t, matrix: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """a camera's detection (R 3x3, t 3 or 3x1) in the main camera's frame: (R, t (3,)); `matrix`
    is the camera's {R, T} into the main camera's frame, None for the main camera itself."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    if matrix is None:
        return R, t
    Rm = np.asarray(matrix['R'], dtype=float).reshape(3, 3)
    Tm = np.asarray(matrix['T'], dtype=float).reshape(3)
    return Rm @ R, Rm @ t + Tm


def weight(distance: float) -> float:
    """how much a detection counts in the average: 1 / distance^4 from its camera, the inverse of
    its variance when the error's standard deviation grows with distance^2."""
    return 1.0 / max(float(distance), NEAR_LIMIT) ** 4


def _better(a: dict, b: dict) -> bool:
    """whether detection a is better than b: nearer its camera, else the higher decision margin."""
    if not math.isclose(a['d'], b['d'], abs_tol=1e-9):
        return a['d'] < b['d']
    return (a.get('m') or 0.0) > (b.get('m') or 0.0)


def camera_detection(detections: list[dict]) -> dict:
    """one camera's detection of a tag in a bucket: the only one, else the one nearest the
    component-wise median of their positions, with `n` the number there were."""
    if len(detections) == 1:
        return {**detections[0], 'n': 1}
    positions = np.array([d['t'] for d in detections])
    median = np.median(positions, axis=0)
    nearest = int(np.argmin(np.linalg.norm(positions - median, axis=1)))
    return {**detections[nearest], 'n': len(detections)}


def fuse_tag(by_camera: dict[str, dict], gate: float = DEFAULT_GATE) -> dict:
    """{t (3,), R (3x3), used [cameras], best camera} of a tag from its cameras' detections
    ({camera: {t, R, d, m}} in the main camera's frame)."""
    cameras = sorted(by_camera)
    best_of = lambda names: min(names, key=lambda c: (by_camera[c]['d'], -(by_camera[c].get('m') or 0.0), c))
    if len(cameras) == 1:
        used = cameras
    else:
        positions = np.array([by_camera[c]['t'] for c in cameras])
        median = np.median(positions, axis=0)
        used = [c for c, p in zip(cameras, positions) if float(np.linalg.norm(p - median)) <= gate]
        if not used:
            used = [best_of(cameras)]
    weights = np.array([weight(by_camera[c]['d']) for c in used])
    t = (np.array([by_camera[c]['t'] for c in used]) * weights[:, None]).sum(axis=0) / weights.sum()
    best = best_of(used)
    return {'t': t, 'R': by_camera[best]['R'], 'used': used, 'best': best}


def camera_relations(votes: dict[tuple[str, str], list[int]]) -> set[tuple[str, str]]:
    """the (a, b) a camera saw a facing b on at least CAMERA_VOTE of its frames that held both:
    `votes` is {(a, b): [frames with a facing b, frames with both]}."""
    return {pair for pair, (seen, both) in votes.items() if both and seen >= CAMERA_VOTE * both}


def fused_relations(fused: dict[str, dict]) -> set[tuple[str, str]]:
    """the (a, b) where a faces b on the fused poses ({tag: {t, R}}) in the main camera's frame."""
    out = set()
    for a, pose_a in fused.items():
        tag_a = [pose_a['R'], np.asarray(pose_a['t'], dtype=float).reshape(3, 1)]
        for b, pose_b in fused.items():
            if a == b:
                continue
            tag_b = [pose_b['R'], np.asarray(pose_b['t'], dtype=float).reshape(3, 1)]
            with np.errstate(invalid='ignore', divide='ignore'):
                if is_tag_looking_at_another_2d(tag_a, tag_b, cosine_threshold=FACING_COSINE,
                                                distance_threshold=FACING_DISTANCE):
                    out.add((a, b))
    return out


def _r(values, digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def detection_entry(detection: dict, start: float, used: bool) -> dict:
    """what the stored event keeps of a camera's detection: its position in the main camera's frame
    (t, metres), its facing there (f: the outward normal, -column 2 of R), its distance from the
    camera (d, metres), the decision margin (m), when the frame was taken (dt: seconds after the
    bucket's start), how many frames it stands for (n, when more than one) and whether the fused
    position left it out (out)."""
    entry = {'t': _r(detection['t'], 4), 'f': _r(-np.asarray(detection['R'])[:, 2], 3),
             'd': round(float(detection['d']), 3), 'm': detection.get('m'),
             'dt': round(float(detection['time']) - float(start), 3)}
    if detection.get('n', 1) > 1:
        entry['n'] = int(detection['n'])
    if not used:
        entry['out'] = True
    return entry


def fuse_bucket(bucket: dict, start: float, gate: float = DEFAULT_GATE) -> tuple[dict, dict, dict, dict]:
    """(translations, rotations, graph, detections) of a bucket as the IPS events store them.

    `bucket` holds {'tags': {tag: {camera: [detection]}}, 'votes': {camera: {(a, b): [seen, both]}},
    'nicla': {badge: set of tags}}; a detection is {t (3,), R (3x3), d, m, time} in the main
    camera's frame. translations are {tag: [[x], [y], [z]]}, rotations {tag: 3x3 list}, graph
    {tag: [tags it faces]} with every tag of the bucket as a key, detections {tag: {camera: entry}}.
    """
    fused, detections = {}, {}
    for tag, by_camera_list in bucket['tags'].items():
        by_camera = {camera: camera_detection(found) for camera, found in by_camera_list.items() if found}
        if not by_camera:
            continue
        fused[tag] = fuse_tag(by_camera, gate)
        detections[tag] = {camera: detection_entry(d, start, camera in fused[tag]['used'])
                           for camera, d in sorted(by_camera.items())}
    translations = {tag: [[float(v)] for v in pose['t']] for tag, pose in fused.items()}
    rotations = {tag: np.asarray(pose['R'], dtype=float).tolist() for tag, pose in fused.items()}

    graph = {tag: set() for tag in fused}
    for votes in bucket['votes'].values():
        for a, b in camera_relations(votes):
            graph.setdefault(a, set()).add(b)
    for a, b in fused_relations(fused):
        graph[a].add(b)
    for badge, seen in bucket['nicla'].items():
        graph.setdefault(str(badge), set()).update(str(tag) for tag in seen)
    graph = {tag: sorted(targets, key=str) for tag, targets in graph.items()}
    return translations, rotations, graph, detections
