"""What a person looks like, for the VFA server's tracker (openmmla.services.vfa.tracking): the
colour of their upper-body clothing and, optionally, their face, so that a track found again
after a gap, and a tag a track carries while it is not read, can be checked against the person.

Two descriptors, each compared by a distance (0 for the same):

- **colour**: an HSV histogram of the upper-body clothing, per channel: the hue in 16 bins, each
  pixel weighted by its saturation (and a pixel darker than V 40 by none, its weight going to a
  grey bin), the saturation in 8 bins and the brightness in 8, each part normalised and given a
  third. It is kept as the square root of the histogram, and two compare by their Hellinger
  distance. The region is the polygon of the shoulders and hips; seated with the hips hidden, a
  box from the shoulder line down 1.2 shoulder widths; with the shoulders hidden too, the band of
  the person's box from 20 % to 50 % of its height, 20 % in from each side; each shrunk by 15 %
  towards its middle, and the crop shrunk to 48 px on its long side.
- **face**: an ArcFace embedding (InsightFace's recognition model, an ONNX file) of the face
  aligned from its five landmarks, which come from the face detector the server runs (RetinaFace
  answers them); two compare by their cosine distance. Only a detector face at least
  `min_face_px` wide and turned at most `max_face_yaw` degrees by its landmarks is used.

A check compares a person with a memory: the face when both sides have one, else the colour (on
the same camera for a tag's gallery: the light differs between cameras), never the colour of a
person whose box overlaps another's (the pixels mix; no memory takes it either). A person
against a track's own looks is `same` at or below the `same` distance, `different` above
`different`, `unknown` in between. A person against a tag (tag_check) is `same` only when that
tag is also the nearest of the session's tags (of those whose gallery holds `rival_looks`
looks), and `different` above `different` or when another tag is within `same` and nearer; a
lost track whose tag has no gallery to compare puts its own looks in that gallery's place. The
distances are the nearest of the memory's looks, which keep the last few, a couple of seconds
apart (TrackLooks, TagGallery).

The defaults come from a calibration on the stored frames of four classroom sessions (14
cameras, 2026-10-02), with galleries of 10 looks 2 s apart: a face cosine distance of 0.55 and a
colour Hellinger distance of 0.12, with the nearest-tag rule, confirmed 90 % of same-person
probes at 0.8 % false accepts. The face separates far better (an AUC of 0.98 to 0.99 against
0.91 for the colour), but only 30 to 39 % of the torso reads show a usable face.

The descriptors live in process memory only: a track's in TrackLooks, a session's tags' in
TagGallery (the frames in which a tag was read on the torso). Nothing here writes a descriptor to
a file, a log or an answer; what leaves is a Check: a kind, a distance and a verdict. A tracker
and its session's gallery are dropped together (the server's idle rule), and a gallery is never
shared between sessions."""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass, field, fields

import numpy as np

from openmmla.services.vfa import features as F

VERDICT_SAME = 'same'
VERDICT_DIFFERENT = 'different'
VERDICT_UNKNOWN = 'unknown'
KIND_COLOUR = 'colour'
KIND_FACE = 'face'
REGION_TORSO = 'torso'
REGION_BOX = 'box'

# the buffalo_l pack of InsightFace: w600k_r50.onnx (ArcFace ResNet-50 trained on WebFace600K)
# and det_10g.onnx (SCRFD-10G with five landmarks); its models are released for non-commercial
# research only
BUFFALO_L_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
FETCH_READ_TIMEOUT = 30.0  # seconds a fetch of the pack (about 280 MB) may wait for its next bytes
FETCH_DEADLINE = 1800.0  # and seconds it may take in all
# where ArcFace wants the five landmarks in its 112 x 112 input: the eye on the image's left, the
# other eye, the nose tip, the mouth corner on the image's left, the other corner
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
ARCFACE_SIZE = 112


# ---- parameters ----

def _unfilled(value) -> bool:
    text = str(value).strip() if value is not None else ''
    return value is None or text == '' or (text.startswith('<') and text.endswith('>'))


def _flag(value, default: bool) -> bool:
    if _unfilled(value):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'true', '1', 'yes', 'y', 'on'}


def _apply(params, config: dict | None):
    """`params` with the keys of `config` that name one of its fields, each converted to the
    field's type; a missing, unfilled or unreadable value keeps the default."""
    if not isinstance(config, dict):
        return params
    for item in fields(params):
        if item.name not in config or isinstance(getattr(params, item.name), (ColourParams, FaceParams)):
            continue
        value, default = config[item.name], getattr(params, item.name)
        try:
            if isinstance(default, bool):
                setattr(params, item.name, _flag(value, default))
            elif isinstance(default, str) and value is not None and not (str(value).strip().startswith('<') and str(value).strip().endswith('>')):
                # a text given empty (or none, off, false) is empty: models_url then never fetches,
                # and no SCRFD is loaded for detection_model; the recognition model has no stand-in
                text = str(value).strip()
                text = '' if text.lower() in {'', 'none', 'off', 'false'} else text
                if text or item.name != 'recognition_model':
                    setattr(params, item.name, text)
            elif _unfilled(value):
                continue
            elif isinstance(default, int):
                setattr(params, item.name, int(float(value)))
            elif isinstance(default, float):
                setattr(params, item.name, float(value))
            else:
                setattr(params, item.name, str(value).strip())
        except (TypeError, ValueError):
            continue
    return params


@dataclass
class ColourParams:
    """the clothing colour check, as calibrated on stored classroom frames (2026-10-02)."""
    enabled: bool = True
    same: float = 0.12  # a Hellinger distance at or below it confirms the person (a tag only when it is also the nearest)
    different: float = 0.35  # above it: someone else; in between, unknown
    hue_bins: int = 16  # the hue, each pixel weighted by its saturation, plus one grey bin
    saturation_bins: int = 8
    value_bins: int = 8
    min_value: int = 40  # a pixel darker than this (OpenCV's V, 0-255) gives its hue weight to the grey bin
    torso_inset: float = 0.15  # the region shrunk by this share towards its middle
    torso_drop: float = 1.2  # hips hidden: the torso hangs this many shoulder widths below the shoulders
    box_top: float = 0.2  # shoulders hidden: the box rows from this share of its height ...
    box_bottom: float = 0.5  # ... to this one
    box_inset: float = 0.2  # ... and this share of its width off each side
    min_shoulder_px: float = 8.0  # shoulders closer together than this count as hidden
    max_side: int = 48  # the crop is shrunk to this many pixels on its long side
    min_pixels: int = 30  # a region of fewer pixels, once shrunk, gives no histogram


@dataclass
class FaceParams:
    """the face check, as calibrated on stored classroom frames (2026-10-02); off unless
    switched on."""
    enabled: bool = False
    same: float = 0.55  # a cosine distance at or below it confirms the person (a tag only when it is also the nearest)
    different: float = 0.65  # above it: someone else; in between, unknown
    min_face_px: float = 56.0  # a face box narrower than this is not used
    max_face_yaw: float = 50.0  # nor one turned further than this by its landmarks (degrees; 50: the nose 0.6 eye distances off the eyes' midpoint)
    recognition_model: str = 'weights/face/w600k_r50.onnx'
    detection_model: str = ''  # SCRFD (weights/face/det_10g.onnx) for persons the face detector missed, from their pose head box; empty: none
    models_url: str = BUFFALO_L_URL  # fetched once, at start, when a model file is missing; empty never fetches
    detection_size: int = 192  # the side of SCRFD's input for a head crop (a multiple of 32)
    detection_threshold: float = 0.5


@dataclass
class AppearanceParams:
    """the appearance checks of the tracker: the colour, the face and their memories."""
    colour: ColourParams = field(default_factory=ColourParams)
    face: FaceParams = field(default_factory=FaceParams)
    descriptor_frames: int = 10  # a track keeps its last this many looks
    gallery_size: int = 10  # a tag keeps this many faces, and this many colours per camera
    sample_spacing_seconds: float = 2.0  # the looks a memory keeps of one camera are at least this far apart
    different_frames: int = 2  # a track in view continues under a new id after this many 'different' tag verdicts in a row (before, the person has a provisional one)
    rival_looks: int = 3  # another tag counts in the nearest-tag rule once its gallery holds this many looks (a badge misread once does not)
    max_overlap: float = 0.3  # a person whose box overlaps another's by this IoU or more adds no colour to a memory, nor is their colour compared

    @classmethod
    def from_config(cls, config: dict | None) -> AppearanceParams:
        """the `appearance` block of the tracking config: `colour` and `face` are each a block
        of their keys or a bare true/false."""
        params = _apply(cls(), config)
        config = config if isinstance(config, dict) else {}
        for name, kind in (('colour', ColourParams), ('face', FaceParams)):
            value = config.get(name)
            if isinstance(value, dict):
                setattr(params, name, _apply(kind(), value))
            elif not _unfilled(value):
                setattr(params, name, kind(enabled=_flag(value, kind().enabled)))
        params.descriptor_frames = max(1, params.descriptor_frames)
        params.gallery_size = max(1, params.gallery_size)
        params.different_frames = max(1, params.different_frames)
        params.rival_looks = max(1, params.rival_looks)
        params.sample_spacing_seconds = max(0.0, params.sample_spacing_seconds)
        return params

    @property
    def active(self) -> bool:
        return bool(self.colour.enabled or self.face.enabled)


# ---- looks, memories and checks ----

class Look:
    """one person in one frame as the checks see them: a colour descriptor (and whether it comes
    from the torso or the box), a face embedding, and whether the person stood clear of the
    others. Held for one frame; its repr never shows the vectors."""
    __slots__ = ('colour', 'region', 'face', 'clear')

    def __init__(self, colour=None, region=None, face=None, clear=True):
        self.colour, self.region, self.face, self.clear = colour, region, face, clear

    def __repr__(self) -> str:
        return (f"Look(colour={'yes' if self.colour is not None else 'no'}, region={self.region}, "
                f"face={'yes' if self.face is not None else 'no'}, clear={self.clear})")


@dataclass(frozen=True)
class Check:
    """the outcome of one comparison: what decided it, its distance (0 for the same), the
    verdict, and the tag it was made against, if any."""
    kind: str
    score: float
    verdict: str
    tag_id: int | None = None

    def as_dict(self) -> dict:
        record = {'kind': self.kind, 'score': round(float(self.score), 3), 'verdict': self.verdict}
        if self.tag_id is not None:
            record['tag_id'] = int(self.tag_id)
        return record


def colour_distance(a: np.ndarray, b: np.ndarray) -> float:
    """the Hellinger distance of two colour descriptors (the square roots of histograms that
    sum to 1): 0 for the same distribution, 1 for two with nothing in common."""
    return math.sqrt(max(0.0, 1.0 - float(np.dot(a, b))))


def face_distance(a: np.ndarray, b: np.ndarray) -> float:
    """the cosine distance of two embeddings: 0 for the same direction."""
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - float(np.dot(a, b)) / norm if norm > 0 else 1.0


def _normalised(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def judge(distance: float, same: float, different: float) -> str:
    return VERDICT_SAME if distance <= same else VERDICT_DIFFERENT if distance > different else VERDICT_UNKNOWN


def decide(params: AppearanceParams, face: float | None, colour: float | None) -> Check | None:
    """a person against one memory, by the nearest distances to its faces and its colours: the
    face when it could be compared, else the colour, else None."""
    if face is not None and params.face.enabled:
        return Check(KIND_FACE, face, judge(face, params.face.same, params.face.different))
    if colour is not None and params.colour.enabled:
        return Check(KIND_COLOUR, colour, judge(colour, params.colour.same, params.colour.different))
    return None


def nearest_tag_verdict(tag: int, distances: dict, same: float, different: float) -> str:
    """the verdict on `tag` among the session's tags' nearest distances: the same person when
    within `same` and nearest of them all; someone else beyond `different`, or when another tag
    is within `same` while this one is not; unknown otherwise."""
    distance = distances[tag]
    others = [value for key, value in distances.items() if key != tag]
    nearest = min(others) if others else None
    if distance <= same and (nearest is None or distance <= nearest):
        return VERDICT_SAME
    if distance > different or (distance > same and nearest is not None and nearest <= same):
        return VERDICT_DIFFERENT
    return VERDICT_UNKNOWN


def tag_check(params: AppearanceParams, gallery: TagGallery | None, tag: int, camera, look: Look | None,
              stand_in: TrackLooks | None = None) -> Check | None:
    """a person whose track claims `tag`, against the galleries of all the session's tags: by
    the face when the person shows one and the tag's gallery holds faces, else by the colour
    against the tags' looks on this camera. Where the tag's gallery holds neither, `stand_in`
    (the looks of the track that remembers the tag) is compared in its place, face first, under
    the same nearest-tag rule against the other tags' galleries. The colour of a person not
    clear of the others is not compared: its pixels mix, and the memories refuse it too. None
    when nothing can be compared."""
    if look is None:
        return None
    tag = int(tag)
    colour = look.colour if look.clear else None
    faces = gallery.face_distances(look.face, tag, params.rival_looks) if gallery is not None and params.face.enabled else {}
    colours = gallery.colour_distances(camera, colour, tag, params.rival_looks) if gallery is not None and params.colour.enabled else {}
    kinds = ((KIND_FACE, faces, params.face, look.face), (KIND_COLOUR, colours, params.colour, colour))
    for kind, distances, limits, _ in kinds:
        if tag in distances:
            return Check(kind, distances[tag], nearest_tag_verdict(tag, distances, limits.same, limits.different), tag)
    if stand_in is not None:
        for kind, distances, limits, mine in kinds:
            if not limits.enabled or mine is None:
                continue
            own = stand_in.face_distance(mine) if kind == KIND_FACE else stand_in.colour_distance(mine)
            if own is not None:
                distances = {**distances, tag: own}
                return Check(kind, own, nearest_tag_verdict(tag, distances, limits.same, limits.different), tag)
    return None


class _Samples:
    """the last `size` vectors of one kind; of one camera only one every `spacing` seconds by
    that camera's clock (a clock that went back, a camera's tracker made again, starts afresh)."""
    __slots__ = ('items', 'spacing', 'last')

    def __init__(self, size: int, spacing: float = 0.0):
        self.items: deque = deque(maxlen=max(1, int(size)))
        self.spacing = float(spacing)
        self.last: dict = {}

    def add(self, vector: np.ndarray, when: float = 0.0, camera=None) -> bool:
        last = self.last.get(camera)
        if last is not None and 0.0 <= when - last < self.spacing - 1e-6:
            return False
        self.last[camera] = when
        self.items.append(vector)
        return True

    def nearest(self, vector: np.ndarray, distance) -> float | None:
        return min(distance(vector, item) for item in self.items) if self.items else None

    def __len__(self) -> int:
        return len(self.items)


class TrackLooks:
    """one camera track's looks: the colours of its last `size` frames in which the person stood
    clear of the others, and its last `size` usable faces, each kind one every `spacing` seconds;
    a person compares with the nearest of them."""

    def __init__(self, size: int, spacing: float = 0.0):
        self.colours = _Samples(size, spacing)
        self.faces = _Samples(size, spacing)

    def add(self, look: Look | None, when: float = 0.0) -> None:
        if look is None:
            return
        if look.colour is not None and look.clear:
            self.colours.add(look.colour, when)
        if look.face is not None:
            self.faces.add(look.face, when)

    def colour_distance(self, colour: np.ndarray | None) -> float | None:
        return self.colours.nearest(colour, colour_distance) if colour is not None else None

    def face_distance(self, face: np.ndarray | None) -> float | None:
        return self.faces.nearest(face, face_distance) if face is not None else None

    def __repr__(self) -> str:
        return f"TrackLooks(colours={len(self.colours)}, faces={len(self.faces)})"


class TagGallery:
    """one session's tags as they looked when read on the torso: per tag, its last `size` faces
    (of any camera) and its last `size` colours of each camera, one every `spacing` seconds of a
    camera. A person compares with the nearest face of each tag, and with the nearest colour of
    each tag on the person's own camera (the light differs between cameras). Shared by the
    trackers of the session's cameras; never written anywhere."""

    def __init__(self, size: int, spacing: float = 0.0):
        self.size, self.spacing = int(size), float(spacing)
        self.faces: dict[int, _Samples] = {}
        self.colours: dict[int, dict[str, _Samples]] = {}
        self.lock = threading.Lock()

    def add(self, tag: int, camera, look: Look | None, when: float = 0.0) -> None:
        if look is None:
            return
        with self.lock:
            if look.colour is not None and look.clear:
                per_camera = self.colours.setdefault(int(tag), {})
                per_camera.setdefault(str(camera), _Samples(self.size, self.spacing)).add(look.colour, when)
            if look.face is not None:
                self.faces.setdefault(int(tag), _Samples(self.size, self.spacing)).add(look.face, when, camera=str(camera))

    def face_distances(self, face: np.ndarray | None, claimed: int | None = None, min_looks: int = 1) -> dict[int, float]:
        """{tag: the distance to its nearest face} for the tags with at least `min_looks` faces,
        and for the `claimed` tag with any."""
        if face is None:
            return {}
        with self.lock:
            return {tag: samples.nearest(face, face_distance) for tag, samples in self.faces.items()
                    if len(samples) >= (1 if tag == claimed else min_looks)}

    def colour_distances(self, camera, colour: np.ndarray | None, claimed: int | None = None,
                         min_looks: int = 1) -> dict[int, float]:
        """{tag: the distance to its nearest colour on `camera`} for the tags with at least
        `min_looks` colours there, and for the `claimed` tag with any."""
        if colour is None:
            return {}
        with self.lock:
            found = {tag: per_camera.get(str(camera)) for tag, per_camera in self.colours.items()}
            return {tag: samples.nearest(colour, colour_distance) for tag, samples in found.items()
                    if samples is not None and len(samples) >= (1 if tag == claimed else min_looks)}

    def clear(self) -> None:
        with self.lock:
            self.faces.clear()
            self.colours.clear()

    def __repr__(self) -> str:
        return f"TagGallery(tags={sorted(set(self.faces) | set(self.colours))})"


# ---- the clothing colour ----

def colour_region(person: dict, min_confidence: float, params: ColourParams) -> tuple[np.ndarray, str] | None:
    """where the upper-body clothing is, as a polygon shrunk by `torso_inset` towards its middle:
    the shoulders and hips; with the hips hidden, a box from the shoulder line down `torso_drop`
    shoulder widths between the shoulders, inside the person's box ('torso'); with the shoulders
    hidden too, a band of the box ('box')."""
    x1, y1, x2, y2 = (float(v) for v in person['bbox'][:4])
    w, h = x2 - x1, y2 - y1
    left_shoulder, right_shoulder = F.keypoint(person, 'left_shoulder', min_confidence), F.keypoint(person, 'right_shoulder', min_confidence)
    left_hip, right_hip = F.keypoint(person, 'left_hip', min_confidence), F.keypoint(person, 'right_hip', min_confidence)
    width = F.distance(left_shoulder, right_shoulder) if left_shoulder and right_shoulder else 0.0
    if left_hip and right_hip and width >= params.min_shoulder_px:
        polygon, region = np.array([left_shoulder, right_shoulder, right_hip, left_hip], dtype=np.float32), REGION_TORSO
    else:
        if width >= params.min_shoulder_px:
            top = min(left_shoulder[1], right_shoulder[1])
            left, right = min(left_shoulder[0], right_shoulder[0]), max(left_shoulder[0], right_shoulder[0])
            bottom, region = min(top + params.torso_drop * width, y2), REGION_TORSO
        else:
            top, bottom = y1 + params.box_top * h, y1 + params.box_bottom * h
            left, right, region = x1 + params.box_inset * w, x2 - params.box_inset * w, REGION_BOX
        if bottom - top < 4 or right - left < 1:
            return None
        polygon = np.array([(left, top), (right, top), (right, bottom), (left, bottom)], dtype=np.float32)
    middle = polygon.mean(axis=0)
    return middle + (polygon - middle) * (1.0 - params.torso_inset), region


def colour_histogram(image: np.ndarray, polygon, params: ColourParams) -> np.ndarray | None:
    """the colour descriptor of the pixels of `image` (BGR) inside `polygon`: the square root of
    the hue (weighted by saturation, plus a grey bin), saturation and brightness histograms, a
    third each. The crop is shrunk to `max_side` pixels on its long side first; None for a
    region of fewer than `min_pixels` pixels then."""
    import cv2

    polygon = np.asarray(polygon, dtype=np.float32)
    height, width = image.shape[:2]
    x0, x1 = max(0, int(math.floor(float(polygon[:, 0].min())))), min(width, int(math.ceil(float(polygon[:, 0].max()))))
    y0, y1 = max(0, int(math.floor(float(polygon[:, 1].min())))), min(height, int(math.ceil(float(polygon[:, 1].max()))))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = image[y0:y1, x0:x1]
    scale = min(1.0, params.max_side / float(max(x1 - x0, y1 - y0))) if params.max_side > 0 else 1.0
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))), interpolation=cv2.INTER_AREA)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round((polygon - [x0, y0]) * scale).astype(np.int32)], 1)
    inside = mask.astype(bool)
    if int(inside.sum()) < max(1, params.min_pixels):
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0][inside].astype(np.int64)
    saturation = hsv[..., 1][inside].astype(np.int64)
    value = hsv[..., 2][inside].astype(np.int64)
    hue_bins, saturation_bins, value_bins = max(1, params.hue_bins), max(1, params.saturation_bins), max(1, params.value_bins)
    weight = (saturation / 255.0) * (value >= params.min_value)
    hues = np.bincount(np.minimum(hue * hue_bins // 180, hue_bins - 1), weights=weight, minlength=hue_bins)
    hues = np.concatenate([hues, [float((1.0 - weight).sum())]])
    saturations = np.bincount(np.minimum(saturation * saturation_bins // 256, saturation_bins - 1), minlength=saturation_bins)
    values = np.bincount(np.minimum(value * value_bins // 256, value_bins - 1), minlength=value_bins)
    parts = [part.astype(np.float64) / float(part.sum()) for part in (hues, saturations, values)]
    return np.sqrt(np.concatenate(parts) / 3.0).astype(np.float32)


# ---- the face ----

def ordered_landmarks(landmarks) -> np.ndarray | None:
    """five landmarks in ArcFace's order (the eye on the image's left, the other eye, the nose,
    the mouth corner on the image's left, the other corner), from RetinaFace's dict (whose left
    and right naming has changed between releases, so the points are ordered by x) or a list of
    five points already in that order."""
    if isinstance(landmarks, dict):
        try:
            eyes = sorted((landmarks['left_eye'], landmarks['right_eye']), key=lambda p: float(p[0]))
            mouth = sorted((landmarks['mouth_left'], landmarks['mouth_right']), key=lambda p: float(p[0]))
            points = [eyes[0], eyes[1], landmarks['nose'], mouth[0], mouth[1]]
        except (KeyError, TypeError, IndexError):
            return None
    else:
        points = landmarks
    try:
        array = np.asarray(points, dtype=np.float32).reshape(5, 2)
    except (TypeError, ValueError):
        return None
    return array if np.all(np.isfinite(array)) else None


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """the 2 x 3 rotation, uniform scale and shift that best takes `source` points onto
    `target` points in the least-squares sense (Umeyama's method)."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    s, t = source - mean_s, target - mean_t
    covariance = t.T @ s / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = (s ** 2).sum() / len(source)
    scale = float((singular * sign).sum() / variance) if variance > 0 else 1.0
    matrix = np.zeros((2, 3))
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = mean_t - scale * rotation @ mean_s
    return matrix


def align_face(image: np.ndarray, landmarks) -> np.ndarray | None:
    """the face as ArcFace takes it: 112 x 112 (BGR), warped so its five landmarks land on the
    template."""
    import cv2

    points = ordered_landmarks(landmarks)
    if points is None:
        return None
    matrix = similarity_transform(points, ARCFACE_TEMPLATE)
    return cv2.warpAffine(image, matrix, (ARCFACE_SIZE, ARCFACE_SIZE), borderValue=0.0)


def landmark_yaw(landmarks) -> float | None:
    """how far a face is turned, in degrees, from where the nose sits between the eyes: its
    offset from their midpoint against half their distance, as an angle (0 facing the camera;
    50 degrees is an offset of 0.6 eye distances, the calibration's limit for a usable face)."""
    points = ordered_landmarks(landmarks)
    if points is None:
        return None
    left, right, nose = points[0], points[1], points[2]
    half = (right[0] - left[0]) / 2.0
    if half <= 1e-6:
        return None
    return round(math.degrees(math.atan2(nose[0] - (left[0] + right[0]) / 2.0, half)), 1)


def _onnx_session(path: str):
    """an ONNX Runtime session on the GPU when ONNX Runtime has CUDA, else on the CPU."""
    import onnxruntime

    preload = getattr(onnxruntime, 'preload_dlls', None)
    available = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in available and callable(preload):
        try:
            # the CUDA and cuDNN libraries of the torch wheels, which ONNX Runtime (1.21+) can share
            preload()
        except Exception:  # noqa: BLE001 - a CPU session still works
            pass
    providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in available] or available
    return onnxruntime.InferenceSession(path, providers=providers)


class FaceEmbedder:
    """InsightFace's ArcFace recognition model (an ONNX file): `embed(image, landmarks)` gives
    the L2-normalised embedding of the face the landmarks mark. The model is loaded at the first
    `load()` (the server calls it at start, never in a request)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.provider = None

    def load(self) -> FaceEmbedder:
        if self.session is None:
            self.session = _onnx_session(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.provider = (self.session.get_providers() or ['?'])[0]
        return self

    def embed(self, image: np.ndarray, landmarks) -> np.ndarray | None:
        aligned = align_face(image, landmarks)
        if aligned is None:
            return None
        self.load()
        # the input w600k_r50 was trained on: RGB, (x - 127.5) / 127.5, NCHW
        blob = ((aligned[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        output = np.asarray(self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0], dtype=np.float32)
        return _normalised(output.reshape(-1))


class ScrfdDetector:
    """InsightFace's SCRFD face detector (det_10g.onnx: three strides, two anchors per cell, five
    landmarks), run on a crop around a head box for a person the server's face detector missed
    (only when `detection_model` names it: faces found in the pose's head boxes separated poorly
    in the calibration, an AUC of 0.898 against 0.984 for the detector's faces):
    `detect(image, box)` gives (score, face box, landmarks) of the best face whose centre lies in
    the box, in the image's pixels, or None."""

    STRIDES = (8, 16, 32)
    ANCHORS = 2

    def __init__(self, model_path: str, size: int = 192, threshold: float = 0.5):
        self.model_path = model_path
        self.size = max(32, int(size) // 32 * 32)
        self.threshold = float(threshold)
        self.session = None
        self.input_name = None

    def load(self) -> ScrfdDetector:
        if self.session is None:
            self.session = _onnx_session(self.model_path)
            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name
            shape = list(model_input.shape)
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self.size = int(shape[2])  # an export with a fixed input
            if len(self.session.get_outputs()) != 3 * len(self.STRIDES):
                raise ValueError(f"{os.path.basename(self.model_path)} is not an SCRFD export with landmarks "
                                 f"(it has {len(self.session.get_outputs())} outputs, not 9)")
        return self

    @classmethod
    def decode(cls, outputs: list, size: int, threshold: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """the faces in SCRFD's raw outputs (scores, box distances and landmark offsets per
        stride, in that order) as (score, [x1, y1, x2, y2], 5 x 2 landmarks) in input pixels."""
        count = len(cls.STRIDES)
        found = []
        for index, stride in enumerate(cls.STRIDES):
            scores, boxes, points = (np.asarray(outputs[index + k * count], dtype=np.float32) for k in range(3))
            if scores.ndim == 3:  # a batched export
                scores, boxes, points = scores[0], boxes[0], points[0]
            cells = size // stride
            centres = np.stack(np.mgrid[:cells, :cells][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            centres = np.repeat(centres, cls.ANCHORS, axis=0)
            scores = scores.reshape(-1)
            keep = np.flatnonzero(scores >= threshold)
            boxes, points = boxes.reshape(-1, 4) * stride, points.reshape(-1, 10) * stride
            for i in keep:
                if i >= len(centres):
                    continue
                cx, cy = centres[i]
                box = np.array([cx - boxes[i, 0], cy - boxes[i, 1], cx + boxes[i, 2], cy + boxes[i, 3]], dtype=np.float32)
                marks = np.stack([cx + points[i, 0::2], cy + points[i, 1::2]], axis=1)
                found.append((float(scores[i]), box, marks))
        return found

    def detect(self, image: np.ndarray, box) -> tuple[float, np.ndarray, np.ndarray] | None:
        import cv2

        self.load()
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        # the head box and half as much again around it, so a face it cuts is found whole
        grow = 0.25 * max(x2 - x1, y2 - y1)
        cx1, cy1 = int(max(0, math.floor(x1 - grow))), int(max(0, math.floor(y1 - grow)))
        cx2, cy2 = int(min(width, math.ceil(x2 + grow))), int(min(height, math.ceil(y2 + grow)))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None
        crop = image[cy1:cy2, cx1:cx2]
        scale = self.size / float(max(crop.shape[:2]))
        resized = cv2.resize(crop, (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))))
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized[:self.size, :self.size]
        blob = ((canvas[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outputs = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})
        best = None
        for score, face, marks in self.decode(outputs, self.size, self.threshold):
            face = face / scale + [cx1, cy1, cx1, cy1]
            marks = marks / scale + [cx1, cy1]
            centre = ((face[0] + face[2]) / 2.0, (face[1] + face[3]) / 2.0)
            if F.point_in_box(centre, (x1, y1, x2, y2)) and (best is None or score > best[0]):
                best = (score, face, marks)
        return best


def ensure_face_models(params: FaceParams, base_dir: str, log=None) -> tuple[str, str | None]:
    """the paths of the recognition model and of SCRFD (None when its file is absent and cannot
    be fetched), resolved against `base_dir`. A missing file is fetched once from `models_url`
    (InsightFace's buffalo_l pack), each model taken out of the archive by its file name."""
    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(base_dir, path)

    recognition, detection = resolve(params.recognition_model), resolve(params.detection_model) if params.detection_model else None
    missing = [path for path in (recognition, detection) if path and not os.path.exists(path)]
    if missing and params.models_url:
        _fetch_models(params.models_url, missing, log)
    if not os.path.exists(recognition):
        raise FileNotFoundError(f"the face recognition model is not at {recognition}")
    return recognition, detection if detection and os.path.exists(detection) else None


def _download(url: str, path: str) -> None:
    """`url` into `path`, failing (TimeoutError) when the connection stays silent for
    FETCH_READ_TIMEOUT seconds or the whole takes longer than FETCH_DEADLINE: the server's start
    waits for it, and a stalled fetch must end in the face check off, not a start that never ends."""
    deadline = time.monotonic() + FETCH_DEADLINE
    with urllib.request.urlopen(url, timeout=FETCH_READ_TIMEOUT) as response, open(path, 'wb') as sink:  # noqa: S310 - a URL from the config
        # what has come, not a full chunk: a trickle must still reach the deadline check
        read = getattr(response, 'read1', response.read)
        while True:
            chunk = read(1 << 16)
            if not chunk:
                return
            sink.write(chunk)
            if time.monotonic() > deadline:
                raise TimeoutError(f"the face models took longer than {FETCH_DEADLINE:g} s to fetch")


def _fetch_models(url: str, paths: list[str], log=None) -> None:
    wanted = {os.path.basename(path): path for path in paths}
    os.makedirs(os.path.dirname(paths[0]) or '.', exist_ok=True)
    if log is not None:
        log.info(f"Fetching the face models {sorted(wanted)} once from {url}")
    with tempfile.TemporaryDirectory(dir=os.path.dirname(paths[0]) or '.') as scratch:
        archive = os.path.join(scratch, 'models.zip')
        _download(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                target = wanted.get(os.path.basename(member))
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
                partial = target + '.part'
                with bundle.open(member) as source, open(partial, 'wb') as sink:
                    shutil.copyfileobj(source, sink)
                os.replace(partial, target)


# ---- one frame's persons ----

def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = F.box_area(a) + F.box_area(b) - inter
    return inter / union if union > 0 else 0.0


def describe_persons(image: np.ndarray, persons: list[dict], faces: list[dict], min_confidence: float,
                     params: AppearanceParams, embedder: FaceEmbedder | None = None,
                     landmarker: ScrfdDetector | None = None, errors: list | None = None) -> list[Look]:
    """a Look per person of the frame, in their order. The colour comes from the torso (or the
    upper box); the face from the detector face that goes to the person (as assign_faces gives
    them out) with its landmarks (with a `landmarker`, SCRFD on the person's head box when the
    detector gave none), and only when it is wide enough and turned little enough by its
    landmarks. A person whose box overlaps another's by `max_overlap` or more is not clear. A
    face model that fails (a GPU out of memory, say) leaves that person and the rest of the
    frame without a face, their colours kept, and its error is added to `errors` when given."""
    copies = [dict(person) for person in persons]
    use_face = params.face.enabled and embedder is not None
    if use_face:
        F.assign_faces(copies, faces, min_confidence)
    boxes = [[float(v) for v in person['bbox'][:4]] for person in persons]
    looks = []
    for index, person in enumerate(persons):
        look = Look()
        look.clear = all(_box_iou(boxes[index], other) < params.max_overlap
                         for j, other in enumerate(boxes) if j != index)
        if params.colour.enabled:
            region = colour_region(person, min_confidence, params.colour)
            if region is not None:
                look.colour = colour_histogram(image, region[0], params.colour)
                look.region = region[1] if look.colour is not None else None
        if use_face:
            try:
                look.face = _face_embedding(image, person, copies[index].get('face'), min_confidence, params.face,
                                            embedder, landmarker)
            except Exception as error:  # noqa: BLE001 - the colour check goes on without the face
                use_face = False
                if errors is not None:
                    errors.append(error)
        looks.append(look)
    return looks


def _face_embedding(image, person, face, min_confidence, params: FaceParams, embedder, landmarker):
    landmarks, width = None, None
    if face is not None and face.get('landmarks') is not None and face.get('face_source', F.FACE_SOURCE_DETECTOR) == F.FACE_SOURCE_DETECTOR:
        landmarks, width = face['landmarks'], float(face['bbox'][2]) - float(face['bbox'][0])
    elif landmarker is not None:
        box = face['bbox'] if face is not None else F.head_box(person, min_confidence)
        if box is not None and float(box[2]) - float(box[0]) >= params.min_face_px:
            found = landmarker.detect(image, box)
            if found is not None:
                landmarks, width = found[2], float(found[1][2] - found[1][0])
    if landmarks is None or width is None or width < params.min_face_px:
        return None
    # the turn by the landmarks alone: the pose's head yaw is too coarse a gate (faces it puts
    # past 60 degrees still separated with an AUC of 0.98 in the calibration)
    yaw = landmark_yaw(landmarks)
    if yaw is None or abs(yaw) > params.max_face_yaw:
        return None
    return embedder.embed(image, landmarks)
