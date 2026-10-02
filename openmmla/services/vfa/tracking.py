"""Tracking the persons of one camera across its frames, for the VFA server's /features: a
ByteTrack (Ultralytics) over the pose model's boxes gives every person a track id that holds
while they stay in view and for a while out of it, and a memory of the AprilTag each track wore
gives a person their tag back while it is hidden. One tracker per camera of a session, made
when the camera's first frame comes. Ultralytics is imported when a tracker is made, so that a
server without the vfa-server extra, and a test, can import this module.

ByteTrack alone compares boxes only, in one global assignment over the tracks in view and the
lost ones, and a lost track's box keeps moving at its last velocity; so a lost track could take a
detection from the track in view of the same person, drift onto someone else, and hand them its
tag. The tracker here (a small subclass of Ultralytics' BYTETracker, below) adds:

- **split** (`split_gap_seconds` 3, `split_on` 'different'): a lost track found again 3 s or
  more after it last saw its person (two missed frame sets at one a second) continues under a
  new track id, which inherits no tag, when the appearance calls the person someone else; with
  `split_on` 'unconfirmed' (the first rule) it does unless the appearance confirms the person;
- **cascade** (off): the tracks in view take the frame's persons first; a lost track is offered
  only a person none of them took;
- **freeze_lost** (off): a lost track stays where its person was last seen, with no velocity,
  and a track found again starts its motion afresh from where it is found.

The cascade and freezing were weighed in a replay of the stored boxes of 20 sessions (36.9
camera-hours, 2026-10-02), with the first split rule: a track found again after 3 to 4 s already
holds someone else as often as after a longer gap (14 to 17 % of the reads across it disagree,
against 4 to 5 % after 2 s and 0.1 % in view), and splitting every such track cut the fused
frames whose tag a read on the same track contradicts from 2.84 % to 1.02 %. The cascade keeps a
person whose box a lost track would take on their own track, but swaps more persons between
tracks in view, which no split catches (1.52 % with the split); freezing adds errors in every
pairing. Both stay as switches. A pilot re-run of three sessions then found that splitting every
re-found track the face did not confirm cost more presence than it saved in wrong tags (the face
had nothing to compare for 3,978 of 4,013 splits, and 84 % of the re-finds after 3 s were the
same person), so by default only a 'different' verdict splits.

The appearance (openmmla.services.vfa.appearance: the face when it is switched on, and the
clothing colour, off by default) also blocks a lost track from a person it says is someone
else, and for the REFUSAL_HOLD_FRAMES frames after from any person it does not confirm, so a
refused newcomer whose face is not seen in the next frame is not taken by it either (while it
is on, a lost track overlapping a track in view also gives way to it in ByteTrack's duplicate
check), and checks a track's remembered tag against that tag's gallery:
the verdict is recorded, and only with `tag_check_acts` withholds the tag and splits the track."""
from __future__ import annotations

import inspect
import threading
import time
import weakref
from types import SimpleNamespace

import numpy as np

from openmmla.services.vfa.appearance import (SPLIT_ON_UNCONFIRMED, SPLIT_RULES, VERDICT_DIFFERENT, VERDICT_SAME,
                                              AppearanceParams, Check, Look, TagGallery, TrackLooks, decide, tag_check)

DEFAULT_BUFFER_FRAMES = 30  # frames a person out of view keeps their track (and their tag)
DEFAULT_IDLE_SECONDS = 600.0  # a camera not heard from for this long forgets its tracks
DEFAULT_SPLIT_GAP_SECONDS = 3.0  # a lost track found again sooner keeps its id and tag; one found later is split as split_on says
DEFAULT_FRAME_SECONDS = 1.0  # the time between two frames of a camera: the synchronizer sends one frame set a second
# frames after the appearance refused a lost track a person in which that track takes only a
# person the appearance confirms: a newcomer's track is confirmed on its second frame, so the
# refused person has one of their own by then even when the pose model misses them once
REFUSAL_HOLD_FRAMES = 3


class _Detections:
    """the pose model's persons as ByteTrack reads detections: `conf`, `xywh` (centre and size)
    and `cls` arrays, indexable by a boolean mask. Every detection clears the tracker's first
    threshold (0, below), so the `idx` its answer carries is the index in the list given."""

    def __init__(self, xyxy, conf):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index):
        return _Detections(self.xyxy[index], self.conf[index])


_CLASSES = None


def _tracker_classes():
    """Ultralytics' BYTETracker and STrack, each with the few methods the tracker changes
    overridden, made once at the first tracker (ultralytics is imported then).

    Written against byte_tracker.py of ultralytics 8.3.223 (one `update`) and 8.4.157 (the same
    steps split into hooks: _first_association, _apply_match, _second_association,
    _unconfirmed_association, merge_track_pools). Both versions:
    - predict the pool of tracks in view and lost through `self.multi_predict(pool)`, once per frame;
    - make the first assignment over that pool from `self.get_dists(pool, detections)` with
      `matching.linear_assignment(dists, thresh=args.match_thresh)`, and the unconfirmed tracks'
      (a newcomer seen once) from `self.get_dists(unconfirmed, leftover)`, whose tracks are not
      yet `is_activated` while every track of the pool is;
    - re-activate a matched lost track with `track.re_activate(det, frame_id, new_id=False)`, and
      turn an unmatched track in view lost with `track.mark_lost()`;
    - make tracks of detections with `self.init_track(results, img)` and give a new track
      `self.next_id()` in `activate` (and in `re_activate` with new_id);
    - drop, of a tracked and a lost track whose boxes overlap by an IoU above 0.85, the one with
      the smaller `frame_id - start_frame` (remove_duplicate_stracks).
    So the overrides are: `get_dists` (the cascade and the appearance gate: it decides the first
    assignment itself and hands back a matrix that admits exactly the pairs it chose),
    `multi_predict` (it notes each track's box before the prediction, for freezing), `init_track`
    (its tracks are PersonTrack), and on the track `mark_lost`, `re_activate` and `next_id`. Should
    a later ultralytics lack one of them, or use one of the other names the two classes add (8.4's
    update calls `self._first_association(...)`, a name a helper here once had, which then
    failed every frame), making a tracker fails with a message naming it."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    from ultralytics.trackers import byte_tracker
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.trackers.utils import matching

    BYTETracker, STrack = byte_tracker.BYTETracker, byte_tracker.STrack
    missing = [f'BYTETracker.{name}' for name in ('update', 'get_dists', 'multi_predict', 'init_track')
               if not callable(getattr(BYTETracker, name, None))]
    missing += [f'STrack.{name}' for name in ('activate', 're_activate', 'mark_lost', 'mark_removed', 'convert_coords', 'next_id')
                if not callable(getattr(STrack, name, None))]
    missing += [f'matching.{name}' for name in ('linear_assignment',) if not callable(getattr(matching, name, None))]
    missing += [f'TrackState.{name}' for name in ('Tracked', 'Lost') if not hasattr(TrackState, name)]
    if 'BYTETracker.get_dists' not in missing and len(inspect.signature(BYTETracker.get_dists).parameters) != 3:
        missing.append('BYTETracker.get_dists(tracks, detections)')
    if missing:
        import ultralytics
        raise RuntimeError(f"ultralytics {getattr(ultralytics, '__version__', '?')} has no {', '.join(missing)}, which "
                           f"the person tracker overrides (openmmla.services.vfa.tracking, written for 8.3 and 8.4)")

    class PersonTrack(STrack):
        """a track of PersonByteTracker: its ids come from its own camera's counter, a lost one
        stays where it was last seen, and one found again may continue under a new id."""
        _owner = None  # a weak reference to the PersonByteTracker that made it
        _observed = None  # the box state the last detection left, before the next prediction
        _started = None  # the true start frame while lost (start_frame then says it never outlives a track in view)
        _refused = None  # the frame the appearance last refused it a person while lost (the refusal holds REFUSAL_HOLD_FRAMES)

        def next_id(self):
            owner = self._owner() if self._owner is not None else None
            return owner.new_id() if owner is not None else STrack.next_id()

        def mark_lost(self):
            super().mark_lost()
            owner = self._owner() if self._owner is not None else None
            if owner is None:
                return
            if owner.freeze and self._observed is not None:
                # where its person was last seen, standing still: no drift onto someone else
                self.mean = self._observed.copy()
                self.mean[4:] = 0.0
            if owner.freeze or owner.cascade or owner.gated:
                # a frozen box, or one the appearance refused a person at, sits where the next
                # person is likely to stand: remove_duplicate_stracks must drop it rather than the
                # track in view that overlaps it, so while lost it counts as younger than any
                self._started, self.start_frame = self.start_frame, self.frame_id + 1

        def re_activate(self, new_track, frame_id, new_id=False):
            owner = self._owner() if self._owner is not None else None
            split = owner.on_refound(self, new_track) if owner is not None else False
            if self._started is not None:
                self.start_frame, self._started = self._started, None
            self._refused = None
            super().re_activate(new_track, frame_id, new_id=bool(new_id or split))
            if owner is not None and owner.freeze:
                # the motion before the gap says nothing about the motion after it
                self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(new_track.tlwh))

    class PersonByteTracker(BYTETracker):
        """BYTETracker with the cascade, frozen lost tracks, splits and the appearance gate (see
        _tracker_classes); `owner` is the PersonTracker whose appearance it asks."""
        track_class = PersonTrack  # 8.4 makes its tracks from this; 8.3's init_track is wrapped below

        def __init__(self, args, owner, cascade: bool = False, freeze: bool = False):
            # ultralytics 8.4 keeps a lost track args.track_buffer frames, 8.3 frame_rate / 30 *
            # track_buffer, so it is told frame_rate 30 when it asks
            if 'frame_rate' in inspect.signature(BYTETracker.__init__).parameters:
                super().__init__(args, frame_rate=30)
            else:
                super().__init__(args)
            self.owner = weakref.ref(owner)
            self.cascade, self.freeze = bool(cascade), bool(freeze)
            self.gated = False  # whether the appearance may refuse a lost track a person (set by the owner)
            self.ids = 0  # this camera's track ids (ultralytics shares one counter across trackers and resets it per tracker)
            self.found = {}  # id(lost track) -> (detection index, gap in frames, Check or None), this frame

        def new_id(self) -> int:
            self.ids += 1
            return self.ids

        def init_track(self, results, img=None):
            tracks = super().init_track(results, img)
            owner = weakref.ref(self)
            for track in tracks:
                if not isinstance(track, PersonTrack):
                    if type(track) is not STrack:
                        raise RuntimeError(f"ultralytics made a {type(track).__name__} track, not the STrack the person tracker extends")
                    track.__class__ = PersonTrack
                track._owner = owner
            return tracks

        def multi_predict(self, tracks):
            if self.freeze:
                for track in tracks:
                    if track.state == TrackState.Tracked and track.mean is not None:
                        track._observed = track.mean.copy()
            super().multi_predict(tracks)

        def get_dists(self, tracks, detections):
            dists = super().get_dists(tracks, detections)
            if not len(tracks) or not len(detections) or not all(track.is_activated for track in tracks):
                return dists  # the unconfirmed tracks' pass, or nothing to match: as ultralytics has it
            return self._decide_first_pass(tracks, detections, np.asarray(dists, dtype=np.float64))

        def _assign(self, costs, rows, columns, threshold):
            rows, columns = list(rows), list(columns)
            if not rows or not columns:
                return []
            matches, _, _ = matching.linear_assignment(costs[np.ix_(rows, columns)], thresh=threshold)
            return [(rows[int(r)], columns[int(c)]) for r, c in matches]

        def _pairs(self, costs, in_view, lost, count, threshold):
            """the first assignment over `costs`: one over every track, or with the cascade the
            tracks in view first and the lost ones over the persons they left."""
            if not self.cascade:
                return self._assign(costs, sorted(in_view + lost), range(count), threshold)
            pairs = self._assign(costs, in_view, range(count), threshold)
            taken = {j for _, j in pairs}
            return pairs + self._assign(costs, lost, [j for j in range(count) if j not in taken], threshold)

        def _decide_first_pass(self, tracks, detections, dists):
            """the first assignment, decided here: a lost track is never matched to a person its
            appearance calls someone else, nor, for REFUSAL_HOLD_FRAMES frames after such a
            refusal, to one its appearance does not confirm; with the cascade the tracks in view
            go first. Answers a matrix that admits exactly the pairs chosen (0 for them, above
            the threshold elsewhere), so ultralytics' own assignment on it applies these."""
            threshold = float(self.args.match_thresh)
            owner = self.owner()
            blocked = max(1.0, threshold) + 1.0
            in_view = [i for i, track in enumerate(tracks) if track.state == TrackState.Tracked]
            lost = [i for i, track in enumerate(tracks) if track.state != TrackState.Tracked]
            gated, checks, refused = dists.copy(), {}, {}  # refused: (i, j) -> the check that refused it (None: nothing compared)
            for i in lost:
                # a refusal holds: the person it refused may show no face in the next frames,
                # before a track of their own is confirmed
                held = tracks[i]._refused is not None and self.frame_id - tracks[i]._refused <= REFUSAL_HOLD_FRAMES
                for j in range(len(detections)):
                    if gated[i, j] >= threshold or owner is None:
                        continue
                    check = owner.reactivation_check(tracks[i], int(detections[j].idx))
                    if check is not None:
                        checks[(i, j)] = check
                    verdict = check.verdict if check is not None else None
                    if verdict == VERDICT_DIFFERENT or (held and verdict != VERDICT_SAME):
                        gated[i, j] = blocked
                        refused[(i, j)] = check
            pairs = self._pairs(gated, in_view, lost, len(detections), threshold)
            decided = np.full(dists.shape, blocked)
            for i, j in pairs:
                decided[i, j] = 0.0
                if tracks[i].state != TrackState.Tracked:
                    self.found[id(tracks[i])] = (int(detections[j].idx), self.frame_id - tracks[i].frame_id, checks.get((i, j)))
            if owner is not None and refused:
                matched = {j for _, j in pairs}
                # a refusal is recorded on a person no track took, and on one it kept from the
                # lost track that would have taken them (who then went to another track); such
                # a 'different' refusal starts (or renews) the track's hold
                decisive = set(refused) & set(self._pairs(dists, in_view, lost, len(detections), threshold))
                order = sorted(refused.items(), key=lambda item: (item[1] is None or item[1].verdict != VERDICT_DIFFERENT,
                                                                  item[1].score if item[1] is not None else 0.0))
                for (i, j), check in order:
                    if j not in matched or (i, j) in decisive:
                        if check is not None and check.verdict == VERDICT_DIFFERENT:
                            tracks[i]._refused = self.frame_id
                        owner.note(int(detections[j].idx), 'track', check, gap_frames=self.frame_id - tracks[i].frame_id,
                                   blocked=True)
            return decided

        def on_refound(self, track, detection) -> bool:
            """a lost track is found again: whether it continues under a new id."""
            index, gap, check = self.found.pop(id(track), (int(detection.idx), self.frame_id - track.frame_id, None))
            owner = self.owner()
            return owner.refound(index, gap, check) if owner is not None else False

    # a name of these classes that ultralytics also uses replaces its own without a word: only
    # the overrides above may (with the attributes their instances set, below in __init__ and
    # on the tracks, compared with a probe of each base)
    args = SimpleNamespace(track_high_thresh=0.0, track_low_thresh=0.0, new_track_thresh=0.25, track_buffer=30,
                           match_thresh=0.8, fuse_score=True)
    shadowed = _shadowed(PersonByteTracker, BYTETracker, {'get_dists', 'multi_predict', 'init_track', 'track_class'},
                         ('owner', 'cascade', 'freeze', 'gated', 'ids', 'found'), lambda: BYTETracker(args))
    shadowed += _shadowed(PersonTrack, STrack, {'mark_lost', 're_activate', 'next_id'}, (),
                          lambda: STrack(np.array([10.0, 10.0, 4.0, 4.0, 0.0], dtype=np.float32), 0.9, 0.0))
    if shadowed:
        import ultralytics
        raise RuntimeError(f"ultralytics {getattr(ultralytics, '__version__', '?')} has its own {', '.join(shadowed)}, "
                           f"which the person tracker's would replace (openmmla.services.vfa.tracking, written for 8.3 "
                           f"and 8.4): rename the tracker's")
    _CLASSES = (PersonByteTracker, PersonTrack, TrackState)
    return _CLASSES


def _shadowed(cls, base, overrides, state, probe=None) -> list[str]:
    """the names `cls` adds to `base` (its methods and class attributes, and `state`, the
    attributes its instances set) that `base` already has, `overrides` aside: those of its class,
    and those an instance `probe()` makes holds (left out when no probe can be made)."""
    names = {name for name in vars(cls) if not (name.startswith('__') and name.endswith('__'))} | set(state)
    known = set(dir(base))
    if probe is not None:
        try:
            known |= set(vars(probe()))
        except Exception:  # noqa: BLE001 - the class's own names are still compared
            pass
    return sorted(f'{base.__name__}.{name}' for name in names - set(overrides) if name in known)


class PersonTracker:
    """the tracks of one camera of one session. `track(persons, looks)` gives every person of the
    next frame their `track_id` (None for one ByteTrack has not confirmed yet: a newcomer's first
    frame), `assign(persons)` runs once the frame's tags are matched and gives a tracked person
    without a tag the one their track wore (`tag_match: track`), remembering the tags seen now. A
    memory lives as long as ByteTrack keeps the track, in view or lost for up to `buffer_frames`;
    a tag seen on another track moves to it, as the newer sighting wins.

    `looks` (openmmla.services.vfa.appearance.describe_persons, one per person) feed the
    appearance checks when `appearance` is given: each track's own looks (its last few, a
    couple of seconds apart), and the session's `gallery` of the tags read on the torso, shared
    by the session's cameras. Without them every check is skipped: a lost track keeps its id and
    tag whenever it is found again (with `split_on` 'unconfirmed', one found after
    `split_gap_seconds` is split), and a remembered tag is carried as before.

    `split_on` and `tag_check_acts` default to the `appearance` parameters' (or their defaults
    without them): which verdict splits a lost track found again after `split_gap_seconds`
    ('different', or 'unconfirmed': any but 'same'), and whether a 'different' verdict on a
    remembered tag withholds it and splits the track, or is only recorded."""

    def __init__(self, buffer_frames: int = DEFAULT_BUFFER_FRAMES, new_track_threshold: float = 0.25,
                 match_threshold: float = 0.8, cascade: bool = False, freeze_lost: bool = False,
                 split_gap_seconds: float | None = DEFAULT_SPLIT_GAP_SECONDS,
                 frame_seconds: float = DEFAULT_FRAME_SECONDS, appearance: AppearanceParams | None = None,
                 gallery: TagGallery | None = None, camera: str | None = None, split_on: str | None = None,
                 tag_check_acts: bool | None = None):
        rules = appearance if appearance is not None else AppearanceParams()
        self.split_on = str(rules.split_on if split_on is None else split_on).strip().lower()
        if self.split_on not in SPLIT_RULES:
            raise ValueError(f"split_on is {self.split_on!r}, not one of {', '.join(SPLIT_RULES)}")
        self.tag_check_acts = bool(rules.tag_check_acts if tag_check_acts is None else tag_check_acts)
        PersonByteTracker, _, _ = _tracker_classes()  # the vfa-server extra
        # the first threshold at 0: the pose model already kept the persons it believes in, and
        # a detection below it would be matched in a second pass whose indices are its own
        args = SimpleNamespace(track_high_thresh=0.0, track_low_thresh=0.0,
                               new_track_thresh=float(new_track_threshold), track_buffer=int(buffer_frames),
                               match_thresh=float(match_threshold), fuse_score=True)
        self.tracker = PersonByteTracker(args, self, cascade=cascade, freeze=freeze_lost)
        self.buffer_frames = int(buffer_frames)
        self.split_gap_seconds = float(split_gap_seconds) if split_gap_seconds is not None and float(split_gap_seconds) > 0 else None
        self.frame_seconds = float(frame_seconds) if frame_seconds and float(frame_seconds) > 0 else DEFAULT_FRAME_SECONDS
        self.appearance = appearance if appearance is not None and appearance.active else None
        self.gallery = gallery if self.appearance is not None else None
        # the gate: with the appearance checks on, a lost track is refused a person its check
        # calls someone else, and gives way to a track in view in ByteTrack's duplicate check,
        # whatever split_on and tag_check_acts say
        self.tracker.gated = self.appearance is not None
        self.camera = camera
        self.tags: dict[int, tuple[int, int]] = {}  # track id -> (tag id, the frame it was seen on)
        self.doubts: dict[int, int] = {}  # track id -> 'different' verdicts in a row on the tag it carries (tag_check_acts)
        self.provisional: dict[int, int] = {}  # track id -> the id its doubted person has meanwhile (tag_check_acts)
        self.descriptors: dict[int, TrackLooks] = {}  # track id -> its last looks (memory only)
        self.frame = 0
        self.touched = time.monotonic()
        self.lock = threading.Lock()  # a camera's frames go through one at a time
        self._looks: list[Look | None] | None = None  # this frame's, until assign
        self._reid: dict[int, dict] = {}  # person index -> {check: record}, this frame

    def track(self, persons: list[dict], looks: list[Look | None] | None = None) -> None:
        """the next frame of this camera: sets `track_id` on every person of it, and `reid` on a
        person a lost track was checked against."""
        self.frame += 1
        self.touched = time.monotonic()
        self._looks = list(looks) if self.appearance is not None and looks is not None and len(looks) == len(persons) else None
        self._reid = {}
        self.tracker.found = {}
        for person in persons:
            person['track_id'] = None
            person.pop('reid', None)
        detections = _Detections([person['bbox'] for person in persons],
                                 [person.get('score', 1.0) for person in persons])
        for row in self.tracker.update(detections):
            # [x1, y1, x2, y2, track id, score, cls, idx]: idx is the detection the track matched
            index = int(row[-1])
            if 0 <= index < len(persons):
                persons[index]['track_id'] = int(row[4])
        for index, record in self._reid.items():
            persons[index]['reid'] = {check: dict(value) for check, value in record.items()}
        alive = self.alive()
        self.descriptors = {track: looks_ for track, looks_ in self.descriptors.items() if track in alive}
        if self._looks is not None:
            for index, person in enumerate(persons):
                if person['track_id'] is not None:
                    self._descriptor(person['track_id']).add(self._looks[index], self.now())

    def now(self) -> float:
        """this camera's clock, in seconds: its frames counted at `frame_seconds` each."""
        return self.frame * self.frame_seconds

    def alive(self) -> set[int]:
        """the ids of the tracks ByteTrack still keeps, in view or lost."""
        return {track.track_id for track in self.tracker.tracked_stracks + self.tracker.lost_stracks}

    def assign(self, persons: list[dict]) -> None:
        """once the frame's tags are matched: a tag seen now is remembered by its track (and
        forgotten by any other), and feeds the session's gallery when read on the torso; a
        tracked person without a tag gets the one their track wore, and is checked against that
        tag's gallery, the verdict going into their `reid`. Only with `tag_check_acts` does a
        'different' verdict act: it withholds the tag in that frame and answers the person under
        a provisional id of their own, so that the fusion, which carries reads along a track id,
        gives that frame no tag either; the `different_frames`-th in a row (2) continues the
        track under that id, without the tag, and a verdict that is not 'different' (or a read)
        puts the person back on their track."""
        looks = self._looks
        alive = self.alive()
        self.tags = {track: seen for track, seen in self.tags.items() if track in alive}
        self.doubts = {track: count for track, count in self.doubts.items() if track in alive}
        self.provisional = {track: other for track, other in self.provisional.items() if track in self.doubts}
        for index, person in enumerate(persons):
            track, tag = person.get('track_id'), person.get('tag_id')
            if tag is None:
                continue
            if self.gallery is not None and looks is not None and person.get('tag_match') == 'torso':
                self.gallery.add(int(tag), self.camera, looks[index], self.now())
            if track is None:
                continue
            self.tags = {other: seen for other, seen in self.tags.items() if seen[0] != tag or other == track}
            self.tags[track] = (int(tag), self.frame)
            self._settle(track)
        for index, person in enumerate(persons):
            track = person.get('track_id')
            if track is None or person.get('tag_id') is not None or track not in self.tags:
                continue
            tag = self.tags[track][0]
            look = looks[index] if looks is not None else None
            check = self._tag_check(tag, look)
            if check is not None:
                record = check.as_dict()
                if check.verdict == VERDICT_DIFFERENT and self.tag_check_acts:
                    self.doubts[track] = self.doubts.get(track, 0) + 1
                    if track not in self.provisional:
                        self.provisional[track] = self.tracker.new_id()
                    if self.doubts[track] >= self.appearance.different_frames:
                        record['split'] = True
                        person['track_id'] = self._split(track, look, self.provisional[track])
                    else:
                        record['withheld'] = True
                        person['track_id'] = self.provisional[track]
                    person['reid'] = dict(person.get('reid') or {}, tag=record)
                    continue
                self._settle(track)
                person['reid'] = dict(person.get('reid') or {}, tag=record)
            person['tag_id'], person['tag_match'] = tag, 'track'
        self._looks = None

    def _settle(self, track: int) -> None:
        """the doubt on the person of `track` is over: they stay on it."""
        self.doubts.pop(track, None)
        self.provisional.pop(track, None)

    # ---- the appearance checks ----

    def _descriptor(self, track: int) -> TrackLooks:
        if track not in self.descriptors:
            self.descriptors[track] = TrackLooks(self.appearance.descriptor_frames, self.appearance.sample_spacing_seconds)
        return self.descriptors[track]

    def reactivation_check(self, track, index: int) -> Check | None:
        """the person at `index` of this frame against the lost `track`. When the track remembers
        a tag: against the galleries of the session's tags, its tag confirmed only when it is also
        the nearest; where the tag's gallery holds nothing to compare (a tag read on the box only,
        or not on this camera), the track's own looks stand in for it under the same rule, so a
        person another tag's gallery holds nearer is not confirmed. Else against the track's own
        looks. The colour of a person not clear of the others is not compared (the face is)."""
        if self._looks is None or not 0 <= index < len(self._looks) or self._looks[index] is None:
            return None
        look = self._looks[index]
        known = self.descriptors.get(track.track_id)
        remembered = self.tags.get(track.track_id)
        if remembered is not None:
            return tag_check(self.appearance, self.gallery, remembered[0], self.camera, look, stand_in=known)
        if known is None:
            return None
        return decide(self.appearance, known.face_distance(look.face),
                      known.colour_distance(look.colour) if look.clear else None)

    def _tag_check(self, tag: int, look: Look | None) -> Check | None:
        """a person their track remembers `tag` for, against the session's tag galleries."""
        if look is None or self.gallery is None:
            return None
        return tag_check(self.appearance, self.gallery, tag, self.camera, look)

    def refound(self, index: int, gap_frames: int, check: Check | None) -> bool:
        """a lost track was found again on the person at `index`, `gap_frames` after it last saw
        its person: whether it continues under a new id. Only after a gap of split_gap_seconds or
        more, and then on a 'different' verdict (split_on 'different'), or on any verdict but
        'same', none included ('unconfirmed'). A re-find after that gap is recorded whatever
        happens to it (`verdict` 'unknown' and `kind` None when nothing could be compared), a
        sooner one when a check ran."""
        gap = gap_frames * self.frame_seconds
        after = self.split_gap_seconds is not None and gap >= self.split_gap_seconds
        verdict = check.verdict if check is not None else None
        if self.split_on == SPLIT_ON_UNCONFIRMED:
            split = after and verdict != VERDICT_SAME
        else:
            # the gate refuses a lost track a person its check calls someone else before the
            # match is made (that person starts a track of their own, and for
            # REFUSAL_HOLD_FRAMES after the track takes only a person it confirms), so no
            # re-find reaches here with 'different' while it does
            split = after and verdict == VERDICT_DIFFERENT
        if check is not None or after:
            self.note(index, 'track', check, gap_frames=gap_frames, split=split)
        return split

    def note(self, index: int, kind: str, check: Check | None, gap_frames: int | None = None,
             split: bool = False, blocked: bool = False) -> None:
        """what a check on the person at `index` decided, for the answer: kind, score and
        verdict, the gap in seconds, and whether the track was split or the match blocked. A
        person found by a lost track keeps that record over one a block left."""
        record = check.as_dict() if check is not None else {'kind': None, 'score': None, 'verdict': 'unknown'}
        if gap_frames is not None:
            record['gap'] = round(gap_frames * self.frame_seconds, 1)
        if split:
            record['split'] = True
        if blocked:
            record['blocked'] = True
            if kind in self._reid.get(index, {}):
                return
        self._reid.setdefault(index, {})[kind] = record

    def _split(self, track: int, look: Look | None, new: int | None = None) -> int:
        """the track in view `track` continues under a new id (`new`, given), without its memory
        and looks."""
        self.tags.pop(track, None)
        self._settle(track)
        self.descriptors.pop(track, None)
        for strack in self.tracker.tracked_stracks:
            if strack.track_id == track:
                strack.track_id = new if new is not None else self.tracker.new_id()
                if self.appearance is not None:
                    self._descriptor(strack.track_id).add(look, self.now())
                return strack.track_id
        return track

    def drop(self) -> None:
        """forget every track, tag and look (the server's idle rule)."""
        self.tags.clear()
        self.doubts.clear()
        self.provisional.clear()
        self.descriptors.clear()
        self._looks, self._reid = None, {}
