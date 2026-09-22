"""Tracking the persons of one camera across its frames, for the VFA server's /features: a
ByteTrack (Ultralytics) over the pose model's boxes gives every person a track id that holds
while they stay in view and for a while out of it, and a memory of the AprilTag each track wore
gives a person their tag back while it is hidden. One tracker per camera of a session, made
when the camera's first frame comes. Ultralytics is imported when a tracker is made, so that a
server without the vfa-server extra, and a test, can import this module."""
from __future__ import annotations

import inspect
import threading
import time
from types import SimpleNamespace

import numpy as np

DEFAULT_BUFFER_FRAMES = 30  # frames a person out of view keeps their track (and their tag)
DEFAULT_IDLE_SECONDS = 600.0  # a camera not heard from for this long forgets its tracks


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


class PersonTracker:
    """the tracks of one camera of one session. `track(persons)` gives every person of the next
    frame their `track_id` (None for one ByteTrack has not confirmed yet: a newcomer's first
    frame), `assign(persons)` runs once the frame's tags are matched and gives a tracked person
    without a tag the one their track wore before (`tag_match: track`), remembering the tags
    seen now. A memory lives as long as ByteTrack keeps the track, in view or lost for up to
    `buffer_frames`; a tag seen on another track moves to it, as the newer sighting wins."""

    def __init__(self, buffer_frames: int = DEFAULT_BUFFER_FRAMES, new_track_threshold: float = 0.25,
                 match_threshold: float = 0.8):
        from ultralytics.trackers.byte_tracker import BYTETracker  # the vfa-server extra
        # the first threshold at 0: the pose model already kept the persons it believes in, and
        # a detection below it would be matched in a second pass whose indices are its own
        args = SimpleNamespace(track_high_thresh=0.0, track_low_thresh=0.0,
                               new_track_thresh=float(new_track_threshold), track_buffer=int(buffer_frames),
                               match_thresh=float(match_threshold), fuse_score=True)
        # the buffer counts frames as given: ultralytics 8.4 keeps a lost track args.track_buffer
        # frames, 8.3 frame_rate / 30 * track_buffer, so it is told frame_rate 30 when it asks
        parameters = inspect.signature(BYTETracker.__init__).parameters
        self.tracker = BYTETracker(args, frame_rate=30) if 'frame_rate' in parameters else BYTETracker(args)
        self.buffer_frames = int(buffer_frames)
        self.tags: dict[int, tuple[int, int]] = {}  # track id -> (tag id, the frame it was seen on)
        self.frame = 0
        self.touched = time.monotonic()
        self.lock = threading.Lock()  # a camera's frames go through one at a time

    def track(self, persons: list[dict]) -> None:
        """the next frame of this camera: sets `track_id` on every person of it."""
        self.frame += 1
        self.touched = time.monotonic()
        for person in persons:
            person['track_id'] = None
        detections = _Detections([person['bbox'] for person in persons],
                                 [person.get('score', 1.0) for person in persons])
        for row in self.tracker.update(detections):
            # [x1, y1, x2, y2, track id, score, cls, idx]: idx is the detection the track matched
            index = int(row[-1])
            if 0 <= index < len(persons):
                persons[index]['track_id'] = int(row[4])

    def alive(self) -> set[int]:
        """the ids of the tracks ByteTrack still keeps, in view or lost."""
        return {track.track_id for track in self.tracker.tracked_stracks + self.tracker.lost_stracks}

    def assign(self, persons: list[dict]) -> None:
        """once the frame's tags are matched: a tag seen now is remembered by its track (and
        forgotten by any other), and a tracked person without a tag gets the one their track wore."""
        alive = self.alive()
        self.tags = {track: seen for track, seen in self.tags.items() if track in alive}
        for person in persons:
            track, tag = person.get('track_id'), person.get('tag_id')
            if track is None or tag is None:
                continue
            self.tags = {other: seen for other, seen in self.tags.items() if seen[0] != tag or other == track}
            self.tags[track] = (int(tag), self.frame)
        for person in persons:
            track = person.get('track_id')
            if track is not None and person.get('tag_id') is None and track in self.tags:
                person['tag_id'], person['tag_match'] = self.tags[track][0], 'track'
