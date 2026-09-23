"""The work area of a camera: the region of the image where the pupils' hands have been, learned
as a session goes on. A gaze the features endpoint calls `elsewhere` (no face, no hands, no zone
within reach) but that lands inside it is on the table the group works at, so it is called
`work_area` instead. Faces, hands, configured zones, out_of_frame and unknown keep priority: only
`elsewhere` is ever relabelled.

The area is the box between the 2.5th and the 97.5th percentile of the pupils' hand centres on
each axis (a stray sighting in one of forty cannot move an edge), grown on every side by the
median hand radius (the circles gaze_target itself scores) and by the gaze tolerance every target
gets (max(W, H) / 64), and clipped to the frame. It is ready once WORK_AREA_MIN_HANDS hand
sightings have been seen, and before that it contains nothing. It is causal and cumulative: a
frame is labelled with the hands of the frames up to and including it, never later ones, over
the whole session so far. Cameras are fixed within a session; a rig moved between sessions gets
its own area.

Only the pupils' hands teach it, by the tag the server gave (read, or kept on its track), never a
tag the fusion carried along a track, so a track carried to the wrong person cannot stretch it.

In 2D the area is not free of faces (a partner's face often lies inside it, above the table); the
relabel works because faces and hands are scored first. Pure functions on plain dicts, like
features.py, and the same WorkArea serves offline (the fusion, window_features.label_work_areas)
and online (the server's /features; not built yet).
"""
from __future__ import annotations

import bisect
import math

from openmmla.services.vfa import features

GAZE_WORK_AREA = 'work_area'
# the percentiles of the hand centres, per axis, that bound the area
WORK_AREA_LOW, WORK_AREA_HIGH = 2.5, 97.5
# hand sightings before the area is ready: five order statistics beyond each 2.5 % edge; at one
# frame set a second with 2-3 pupils, about 40-60 s
WORK_AREA_MIN_HANDS = 200
# a wrist counts from this confidence, as features' min_confidence default
WORK_AREA_KEYPOINT_CONFIDENCE = 0.3
# a person's work-area share is read only where at least this share of their gaze frames had a
# ready area ("most of")
WORK_AREA_READY_MIN = 0.5
# the fusion's mark on a tag it carried along a track (window_features.PROPAGATED)
_PROPAGATED = 'propagated'


def _rank(values: list, q: float) -> float:
    """the nearest-rank percentile `q` of sorted `values`: index floor(q / 100 * (n - 1) + 0.5)."""
    return values[int(math.floor(q / 100.0 * (len(values) - 1) + 0.5))]


class WorkArea:
    """the work area of one camera of one session, learned causally from the pupils' hands."""

    def __init__(self, width, height, min_hands: int = WORK_AREA_MIN_HANDS,
                 low: float = WORK_AREA_LOW, high: float = WORK_AREA_HIGH):
        self.width, self.height = float(width), float(height)
        self.min_hands, self.low, self.high = int(min_hands), float(low), float(high)
        self.tolerance = features.gaze_tolerance(self.width, self.height)
        # every hand circle seen, each coordinate kept sorted
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.radii: list[float] = []

    def update(self, hands) -> None:
        """adds hand circles, ((x, y), radius) as features.hand_regions gives them. Every radius
        counts in the median, 0 included (the shoulders unseen)."""
        for (x, y), radius in hands:
            bisect.insort(self.xs, float(x))
            bisect.insort(self.ys, float(y))
            bisect.insort(self.radii, float(radius))

    @property
    def n(self) -> int:
        return len(self.xs)

    @property
    def ready(self) -> bool:
        return self.n >= self.min_hands

    def box(self) -> tuple[float, float, float, float] | None:
        """(x1, y1, x2, y2), or None until ready: the percentile box of the hand centres grown by
        the median hand radius plus the gaze tolerance, clipped to the frame."""
        if not self.ready:
            return None
        pad = self.radii[(self.n - 1) // 2] + self.tolerance
        x1, x2 = _rank(self.xs, self.low) - pad, _rank(self.xs, self.high) + pad
        y1, y2 = _rank(self.ys, self.low) - pad, _rank(self.ys, self.high) + pad
        return (min(max(x1, 0.0), self.width), min(max(y1, 0.0), self.height),
                min(max(x2, 0.0), self.width), min(max(y2, 0.0), self.height))

    def contains(self, point) -> bool:
        """whether `point` lies in the ready area (its edges count); False until ready."""
        box = self.box()
        if box is None or point is None:
            return False
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            return False
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]

    def state(self) -> dict:
        box = self.box()
        return {'hands': self.n, 'ready': self.ready, 'box': list(box) if box is not None else None}


def pupil_hands(persons, pupils, min_confidence: float = WORK_AREA_KEYPOINT_CONFIDENCE) -> list:
    """the hand circles (features.hand_regions) of every person whose server tag is one of the
    `pupils`; a person without keypoints, or whose tag the fusion carried along a track, gives
    none."""
    wanted = {str(tag) for tag in pupils or ()}
    hands = []
    for person in persons or []:
        tag = person.get('tag_id')
        if tag is None or str(tag) not in wanted or person.get('tag_match') == _PROPAGATED:
            continue
        if not person.get('keypoints'):
            continue
        try:
            hands.extend(features.hand_regions(person, min_confidence))
        except (KeyError, TypeError, ValueError, IndexError):
            # a body without a box to fall back on for the shoulder width gives no hands
            continue
    return hands


def label(target, point, area: WorkArea) -> dict:
    """the target with `elsewhere` inside the ready area called work_area; anything else as it
    was (faces and hands, zones, out_of_frame and unknown keep priority)."""
    if isinstance(target, dict) and target.get('category') == features.GAZE_ELSEWHERE and area.contains(point):
        return {'category': GAZE_WORK_AREA, 'person_id': None, 'zone': None}
    return target


def apply_work_area(frame: dict, area: WorkArea, pupils) -> dict:
    """a copy of a frame (a /features answer, or a frame of a vfa_features event) with every
    person's gaze target passed through label(): the frame's pupils' hands are added to the area
    first, since the present frame counts. frame['work_area'] is the box used (a list), None while
    the area is not ready. The input is never changed."""
    persons = frame.get('persons') or []
    area.update(pupil_hands(persons, pupils))
    box = area.box()
    out = dict(frame)
    if box is not None:
        labelled = []
        for person in persons:
            gaze = person.get('gaze')
            if isinstance(gaze, dict):
                target = gaze.get('target')
                relabelled = label(target, gaze.get('point'), area)
                if relabelled is not target:
                    person = dict(person, gaze=dict(gaze, target=relabelled))
            labelled.append(person)
        if 'persons' in frame:
            out['persons'] = labelled
    out['work_area'] = list(box) if box is not None else None
    return out
