"""The features the VFA server's /features answers for one frame: who is where (a skeleton per
person, the AprilTag it belongs to), where each head turns, and where each gaze lands, as
geometry on the image. Pure functions on plain dicts, so that a test needs no model.

Coordinates are pixels from the top-left corner of the frame, as the pose model and the gaze
model give them; the frame's width and height come with the answer, so a client may normalize.
"""
from __future__ import annotations

import math

from openmmla.utils.video.pose import COCO_KEYPOINTS

KP = {name: index for index, name in enumerate(COCO_KEYPOINTS)}

# what a gaze may land on: a partner's face wins over any hands, the nearer hands (own or a
# partner's) over a zone of the table, and a zone over nothing in particular
GAZE_PARTNER_FACE = 'partner_face'
GAZE_PARTNER_HANDS = 'partner_hands'
GAZE_OWN_HANDS = 'own_hands'
GAZE_ZONE = 'zone'
GAZE_ELSEWHERE = 'elsewhere'
GAZE_OUT_OF_FRAME = 'out_of_frame'
GAZE_UNKNOWN = 'unknown'  # no face found for the person, or the gaze model gave nothing

# a hand region: this fraction of the shoulder width around a point nudged past the wrist along
# the forearm, so that it covers the fingers rather than the wrist bone
HAND_RADIUS_SHOULDERS = 0.4
HAND_NUDGE = 0.35
# a head: the widest of the ear span, 2.3 times the distance between the eyes, and this
# fraction of the shoulder width
HEAD_WIDTH_SHOULDERS = 0.32
HEAD_WIDTH_EYES = 2.3
# a face box is grown by this much on each side before a gaze point is looked for in it
FACE_MARGIN = 0.2
# the chest, when the hips are hidden (a person seated at a table): a box from the shoulder
# line down this many shoulder widths, this much wider than the shoulders on each side
TORSO_DROP = 1.6
TORSO_MARGIN = 0.15
# how far the gaze model can place a point at all: its heatmap has this many cells across
GAZE_HEATMAP_CELLS = 64
# the yaw read from where the nose sits between two landmarks: the ears span the head's
# diameter (half-angle 90 degrees), the eyes a narrower arc (half-angle about 13 degrees)
EARS_GAIN = 1.0
EYES_GAIN = 1.0 / math.tan(math.radians(13.0))
ONE_EAR_YAW = 70.0


def keypoint(person: dict, name: str, min_confidence: float) -> tuple[float, float] | None:
    """the (x, y) of a person's keypoint when it was seen with at least `min_confidence`; None
    for a keypoint the model does not answer at all."""
    index = KP[name]
    keypoints = person['keypoints']
    if index >= len(keypoints):
        return None
    x, y, confidence = keypoints[index]
    return (float(x), float(y)) if float(confidence) >= min_confidence else None


def point_in_box(point, box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def point_in_polygon(point, polygon) -> bool:
    """whether `point` lies in `polygon` (a list of (x, y), any winding), by ray casting."""
    x, y = point
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if (y1 > y) != (y2 > y):
            cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < cross:
                inside = not inside
    return inside


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_to_segment(point, a, b) -> float:
    """the distance from `point` to the segment a-b."""
    ax, ay = a
    bx, by = b
    px, py = point
    length2 = (bx - ax) ** 2 + (by - ay) ** 2
    if length2 < 1e-12:
        return distance(point, a)
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / length2))
    return distance(point, (ax + t * (bx - ax), ay + t * (by - ay)))


def distance_to_polygon(point, polygon) -> float:
    """how far `point` is from `polygon`: 0 inside, else the distance to its nearest edge."""
    if point_in_polygon(point, polygon):
        return 0.0
    count = len(polygon)
    return min(distance_to_segment(point, polygon[i], polygon[(i + 1) % count]) for i in range(count))


def box_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def shoulder_width(person: dict, min_confidence: float) -> float:
    """the distance between the shoulders; a share of the box width when one is hidden."""
    left, right = keypoint(person, 'left_shoulder', min_confidence), keypoint(person, 'right_shoulder', min_confidence)
    if left and right and distance(left, right) > 1e-6:
        return distance(left, right)
    return 0.3 * max(1.0, person['bbox'][2] - person['bbox'][0])


def head_width(person: dict, min_confidence: float) -> float:
    """how wide the head is on the image: the widest of the ear span, 2.3 times the distance
    between the eyes, and a share of the shoulder width."""
    widths = [HEAD_WIDTH_SHOULDERS * shoulder_width(person, min_confidence)]
    ears = keypoint(person, 'left_ear', min_confidence), keypoint(person, 'right_ear', min_confidence)
    if all(ears):
        widths.append(distance(*ears))
    eyes = keypoint(person, 'left_eye', min_confidence), keypoint(person, 'right_eye', min_confidence)
    if all(eyes):
        widths.append(HEAD_WIDTH_EYES * distance(*eyes))
    return max(widths)


def torso_region(person: dict, min_confidence: float) -> tuple[list[tuple[float, float]], str] | None:
    """where a badge worn on the chest sits: the quadrilateral of the shoulders and hips
    ('hips'), else, seated at a table with the hips hidden, a box hanging from the shoulder
    line ('shoulders'); None when the shoulders are hidden too."""
    shoulders = keypoint(person, 'left_shoulder', min_confidence), keypoint(person, 'right_shoulder', min_confidence)
    if not all(shoulders):
        return None
    hips = keypoint(person, 'right_hip', min_confidence), keypoint(person, 'left_hip', min_confidence)
    if all(hips):
        return [shoulders[0], shoulders[1], hips[0], hips[1]], 'hips'
    width = distance(*shoulders)
    if width < 1e-6:
        return None
    left_x, right_x = min(s[0] for s in shoulders), max(s[0] for s in shoulders)
    top = min(s[1] for s in shoulders)
    margin = TORSO_MARGIN * width
    return [(left_x - margin, top), (right_x + margin, top),
            (right_x + margin, top + TORSO_DROP * width), (left_x - margin, top + TORSO_DROP * width)], 'shoulders'


def polygon_center(polygon) -> tuple[float, float]:
    return sum(p[0] for p in polygon) / len(polygon), sum(p[1] for p in polygon) / len(polygon)


def assign_tags(persons: list[dict], tags: dict, min_confidence: float) -> None:
    """give each person the AprilTag on their chest, one to one: a tag inside a person's torso
    beats one merely inside their box, and among those the tag nearest the middle of the torso
    (or box) wins; every pair is ranked that way and taken greedily, so no tag names two persons
    and no person wears two tags. Sets `tag_id` (int or None), `tag_match` ('torso', 'box' or
    None) and `person_id` (the tag id as text, else unknown_1, unknown_2 ... from left to right)."""
    ranked = []
    for tag_id, centre in tags.items():
        for index, person in enumerate(persons):
            region = torso_region(person, min_confidence)
            if region is not None and point_in_polygon(centre, region[0]):
                ranked.append((0, distance(centre, polygon_center(region[0])), index, int(tag_id), 'torso'))
            elif point_in_box(centre, person['bbox']):
                ranked.append((1, distance(centre, box_center(person['bbox'])), index, int(tag_id), 'box'))
    for person in persons:
        person['tag_id'], person['tag_match'] = None, None
    taken_tags = set()
    for _, _, index, tag_id, how in sorted(ranked, key=lambda item: (item[0], item[1], item[2], item[3])):
        if tag_id in taken_tags or persons[index]['tag_id'] is not None:
            continue
        persons[index]['tag_id'], persons[index]['tag_match'] = tag_id, how
        taken_tags.add(tag_id)
    unknown = 0
    for person in sorted(persons, key=lambda p: p['bbox'][0]):
        if person['tag_id'] is None:
            unknown += 1
            person['person_id'] = f'unknown_{unknown}'
        else:
            person['person_id'] = str(person['tag_id'])


def head_yaw(person: dict, min_confidence: float) -> float | None:
    """how far the head is turned, in degrees: 0 facing the camera, positive turned towards the
    right of the image, negative towards its left, past +-90 when seen from behind. Read from
    where the nose sits between the ears (between the eyes when an ear is hidden, scaled for
    their narrower arc). About +-70 when only one ear is seen, on the side the nose turned to;
    None when the nose, or both ears and eyes, are hidden."""
    nose = keypoint(person, 'nose', min_confidence)
    if nose is None:
        return None
    left_ear, right_ear = keypoint(person, 'left_ear', min_confidence), keypoint(person, 'right_ear', min_confidence)
    if left_ear and right_ear:
        return _yaw_between(nose, left_ear, right_ear, EARS_GAIN)
    left_eye, right_eye = keypoint(person, 'left_eye', min_confidence), keypoint(person, 'right_eye', min_confidence)
    if left_eye and right_eye:
        return _yaw_between(nose, left_eye, right_eye, EYES_GAIN)
    # one ear hidden: the head turned away from it. facing the camera, a person's right ear is on
    # the image's left; their right ear alone in view means they turned towards the image's right
    if right_ear and not left_ear:
        return ONE_EAR_YAW
    if left_ear and not right_ear:
        return -ONE_EAR_YAW
    return None


def _yaw_between(nose, left, right, gain: float) -> float:
    """the yaw from where the nose sits between the person's left and right landmark: the
    nose's offset from their midpoint against their signed half-width (the left landmark sits
    at the larger x when the person faces the camera, so a person seen from behind reads past
    +-90), as an angle."""
    half_width = (left[0] - right[0]) / 2.0
    offset = nose[0] - (left[0] + right[0]) / 2.0
    if abs(half_width) < 1e-6 and abs(offset) < 1e-6:
        return 0.0
    return round(math.degrees(math.atan2(gain * offset, half_width)), 1)


def assign_faces(persons: list[dict], faces: list[dict], min_confidence: float) -> None:
    """give each person the face found on them: the face whose box holds their nose, else the
    person whose box holds the face's centre (the one whose nose, or box centre, is nearest).
    Sets `face` ({bbox, gaze_point, inout}) or None."""
    for person in persons:
        person['face'] = None
    taken = set()
    for face in faces:
        box = face['bbox']
        centre = box_center(box)
        ranked = []
        for index, person in enumerate(persons):
            if index in taken:
                continue
            nose = keypoint(person, 'nose', min_confidence)
            if nose is not None and point_in_box(nose, box):
                ranked.append((0, distance(nose, centre), index))
            elif point_in_box(centre, person['bbox']):
                ranked.append((1, distance(nose or box_center(person['bbox']), centre), index))
        if not ranked:
            continue
        index = min(ranked)[2]
        taken.add(index)
        persons[index]['face'] = {'bbox': [float(v) for v in box],
                                  'gaze_point': [float(v) for v in face['gaze_point']] if face.get('gaze_point') else None,
                                  'inout': float(face['inout']) if face.get('inout') is not None else None}


def hand_regions(person: dict, min_confidence: float) -> list[tuple[tuple[float, float], float]]:
    """where a person's hands are: for each wrist seen, a circle around a point nudged past the
    wrist along the forearm (when the elbow is seen), with a radius from the shoulder width."""
    radius = HAND_RADIUS_SHOULDERS * shoulder_width(person, min_confidence)
    regions = []
    for side in ('left', 'right'):
        wrist = keypoint(person, f'{side}_wrist', min_confidence)
        if wrist is None:
            continue
        elbow = keypoint(person, f'{side}_elbow', min_confidence)
        centre = wrist
        if elbow is not None and distance(elbow, wrist) > 1e-6:
            length = distance(elbow, wrist)
            centre = (wrist[0] + HAND_NUDGE * radius * (wrist[0] - elbow[0]) / length,
                      wrist[1] + HAND_NUDGE * radius * (wrist[1] - elbow[1]) / length)
        regions.append((centre, radius))
    return regions


def gaze_tolerance(width: int, height: int) -> float:
    """how far off a gaze point may be by construction: the gaze model places it in one cell of
    its heatmap, so one cell of the frame's longer side."""
    return max(width, height) / GAZE_HEATMAP_CELLS


def _grown(box, margin: float):
    width, height = box[2] - box[0], box[3] - box[1]
    return [box[0] - margin * width, box[1] - margin * height, box[2] + margin * width, box[3] + margin * height]


def _box_distance(point, box) -> float:
    """how far `point` is from `box`: 0 inside, else the distance to its edge."""
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def gaze_target(person: dict, persons: list[dict], zones: dict, min_confidence: float,
                inout_threshold: float, tolerance: float = 0.0) -> dict:
    """what a person's gaze lands on: {category, person_id, zone}. Everything within reach is
    scored and the best taken: a partner's face (the grown face box, else around their nose)
    beats any hands; the nearest hands, the person's own or a partner's, beat a zone; a named
    zone beats nothing in particular. `tolerance` widens every target by what the gaze model
    cannot resolve. out_of_frame when the model puts the gaze outside the image, unknown when
    there is no gaze for the person."""
    face = person.get('face')
    if not face or face.get('gaze_point') is None or face.get('inout') is None:
        return {'category': GAZE_UNKNOWN, 'person_id': None, 'zone': None}
    if face['inout'] < inout_threshold:
        return {'category': GAZE_OUT_OF_FRAME, 'person_id': None, 'zone': None}
    point = tuple(face['gaze_point'])
    candidates = []  # (priority, distance, target)
    for other in persons:
        if other is person:
            continue
        other_face = other.get('face')
        if other_face:
            gap = _box_distance(point, _grown(other_face['bbox'], FACE_MARGIN))
            if gap <= tolerance:
                candidates.append((0, gap, {'category': GAZE_PARTNER_FACE, 'person_id': other['person_id'], 'zone': None}))
                continue
        nose = keypoint(other, 'nose', min_confidence)
        if nose is not None:
            gap = distance(point, nose) - head_width(other, min_confidence) / 2.0
            if gap <= tolerance:
                candidates.append((0, max(gap, 0.0), {'category': GAZE_PARTNER_FACE, 'person_id': other['person_id'], 'zone': None}))
    for other in persons:
        for centre, radius in hand_regions(other, min_confidence):
            gap = distance(point, centre) - radius
            if gap <= tolerance:
                target = {'category': GAZE_OWN_HANDS, 'person_id': None, 'zone': None} if other is person \
                    else {'category': GAZE_PARTNER_HANDS, 'person_id': other['person_id'], 'zone': None}
                candidates.append((1, max(gap, 0.0), target))
    for name, polygon in (zones or {}).items():
        if len(polygon) >= 3:
            gap = distance_to_polygon(point, polygon)
            if gap <= tolerance:
                candidates.append((2, gap, {'category': GAZE_ZONE, 'person_id': None, 'zone': str(name)}))
    if not candidates:
        return {'category': GAZE_ELSEWHERE, 'person_id': None, 'zone': None}
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _gaze_in_frame(person: dict, inout_threshold: float):
    """a person's gaze point when the gaze model puts it in the frame; None otherwise."""
    face = person.get('face') or {}
    point, inout = face.get('gaze_point'), face.get('inout')
    return tuple(point) if point and inout is not None and inout >= inout_threshold else None


def pairwise(persons: list[dict], min_confidence: float, inout_threshold: float = 0.5) -> dict:
    """for every two persons, how far their gazes land apart (a joint-attention proxy) and how
    close their hands come, in pixels; None when either side lacks the data, which for the gazes
    includes one the model puts out of the frame (below `inout_threshold`)."""
    pairs = {}
    for i, a in enumerate(persons):
        for b in persons[i + 1:]:
            gazes = [_gaze_in_frame(p, inout_threshold) for p in (a, b)]
            gaze_distance = round(distance(gazes[0], gazes[1]), 1) if all(gazes) else None
            hands_a = [centre for centre, _ in hand_regions(a, min_confidence)]
            hands_b = [centre for centre, _ in hand_regions(b, min_confidence)]
            hand_distance = round(min(distance(x, y) for x in hands_a for y in hands_b), 1) \
                if hands_a and hands_b else None
            pairs[f'{a["person_id"]}|{b["person_id"]}'] = {'gaze_distance': gaze_distance,
                                                            'hand_distance': hand_distance}
    return pairs


# a polygon whose coordinates all lie within this band is in [0, 1] units (a little overshoot
# past the frame's edge included); a pixel polygon this small is no polygon at all
NORMALIZED_BAND = 1.5


def scaled_zones(zones: dict | None, width: int, height: int) -> dict:
    """the zones in pixels, clamped to the frame: a polygon whose coordinates all lie within
    [-1.5, 1.5] is taken as normalized to [0, 1] and scaled to the frame; one with bigger
    numbers is in pixels already. A polygon that is not a sequence of at least three [x, y]
    number pairs raises ValueError naming the zone."""
    scaled = {}
    for name, polygon in (zones or {}).items():
        try:
            points = [(float(x), float(y)) for x, y in polygon]
        except (TypeError, ValueError):
            raise ValueError(f"zone '{name}' is not a list of [x, y] pairs")
        if len(points) < 3:
            raise ValueError(f"zone '{name}' has fewer than three points")
        if all(-NORMALIZED_BAND <= x <= NORMALIZED_BAND and -NORMALIZED_BAND <= y <= NORMALIZED_BAND for x, y in points):
            points = [(x * width, y * height) for x, y in points]
        scaled[str(name)] = [(min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))) for x, y in points]
    return scaled


def frame_features(persons: list[dict], tags: dict, faces: list[dict], zones: dict | None,
                   width: int, height: int, angle: str, min_confidence: float = 0.3,
                   inout_threshold: float = 0.5, keypoints: bool = True) -> dict:
    """the features of one frame: `persons`, each with person_id, tag_id, tag_match, bbox, score,
    keypoints (left out with `keypoints=False`), head_yaw, face_bbox and gaze {point, inout,
    target}; `tags` as seen; `zones` as resolved, in pixels; `pairs`; and the frame's angle,
    width and height. `tags` maps tag id -> (x, y) pixel centre; `faces` are [{bbox, gaze_point,
    inout}] from the gaze model."""
    persons = [dict(person) for person in persons]
    assign_tags(persons, tags, min_confidence)
    assign_faces(persons, faces, min_confidence)
    zones_px = scaled_zones(zones, width, height)
    tolerance = gaze_tolerance(width, height)
    answer = []
    for person in persons:
        target = gaze_target(person, persons, zones_px, min_confidence, inout_threshold, tolerance)
        face = person.get('face')
        entry = {
            'person_id': person['person_id'],
            'tag_id': person['tag_id'],
            'tag_match': person['tag_match'],
            'bbox': [round(float(v), 1) for v in person['bbox']],
            'score': round(float(person.get('score', 0.0)), 3),
            'head_yaw': head_yaw(person, min_confidence),
            'face_bbox': [round(v, 1) for v in face['bbox']] if face else None,
            'gaze': {'point': [round(v, 1) for v in face['gaze_point']] if face and face.get('gaze_point') else None,
                     'inout': round(face['inout'], 3) if face and face.get('inout') is not None else None,
                     'target': target},
        }
        if keypoints:
            entry['keypoints'] = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in person['keypoints']]
        answer.append(entry)
    return {
        'angle': angle,
        'width': int(width),
        'height': int(height),
        'tags': {str(tag_id): [round(float(x), 1), round(float(y), 1)] for tag_id, (x, y) in tags.items()},
        'zones': {name: [[round(x, 1), round(y, 1)] for x, y in polygon] for name, polygon in zones_px.items()},
        'persons': answer,
        'pairs': pairwise(persons, min_confidence, inout_threshold),
    }
