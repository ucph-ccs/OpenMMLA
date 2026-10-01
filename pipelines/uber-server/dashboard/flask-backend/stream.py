"""The live data stream of a session, as Server-Sent Events.

Records go out in the slim formats of openmmla.analytics.report.live. Follow mode sends the last
few minutes of a session, then what its pipelines write as they write it. Replay mode walks a
virtual clock through a finished (or earlier) part of a session at 1 to 16 times real time and
releases each record when the clock reaches it: a window when it starts, a transcript chunk when
it ends, since that is when its text exists.

The followers of one session share one poller, a SessionFeed. It starts with the first follower,
asks InfluxDB once a second for what each event type wrote after its newest record (six small
queries side by side; every 5 s once nothing new came for a minute), slims the records once and
hands the batch to every follower through a bounded queue. It stops 10 s after the last follower
left, so a page reload finds it still running. InfluxDB therefore sees the same six queries a
second whether one browser follows a session or thirty. Each follower still reads its own history:
it joins the feed first, notes how far the feed has read each event type, reads its backfill up
to there, then passes on the feed's batches, leaving out the records its backfill already held, so
nothing is missing or sent twice. The feed also keeps what slimming needs across batches: the
voice keys (which every follower shares, so a voice has one key on every page), the floor basis
and the worn microphones' recent levels. A follower that falls behind by more than its queue holds
(about five minutes of batches) is dropped; its page reconnects and reads the gap as history.

The shared feed needs gevent (gunicorn's gevent worker, or `python dashboard.py serve`): the
poller is a greenlet beside the request greenlets, so the report package's InfluxDB clients are
only ever used from the hub's thread, never from a native thread. Without gevent each follower
polls on its own. Replays never share: each has its own clock, cursors, voice keys and floor basis,
and reads ahead 10 to 32 s of session time at a time, which keeps batches flowing while the next
piece arrives.
"""

import json
import logging
import time

logger = logging.getLogger("dashboard.stream")

ASR_RECOGNITION = "asr_recognition"
ASR_TRANSCRIPTION = "asr_transcription"
IPS_TRANSLATION = "ips_translation"
IPS_ROTATION = "ips_rotation"
IPS_RELATION = "ips_relation"
VFA_FEATURES = "vfa_features"
EVENT_TYPES = (ASR_RECOGNITION, ASR_TRANSCRIPTION, IPS_TRANSLATION, IPS_ROTATION, IPS_RELATION, VFA_FEATURES)
IPS_TYPES = (IPS_TRANSLATION, IPS_ROTATION, IPS_RELATION)

PING_EVERY = 15.0
STATUS_EVERY = 5.0
FOLLOW_TICK = 1.0
FOLLOW_IDLE_TICK = 5.0
FOLLOW_IDLE_AFTER = 60.0
REPLAY_TICK = 0.5
BACKFILL_CHUNK = 60.0
BACKFILL_DEFAULT = 300.0
BACKFILL_MAX = 900.0
LIVE_SECONDS = 20.0
MIN_FLOOR_OBS = 30
# a window's InfluxDB time is its end, at most a recognition window (3 s) after its start
MAX_EMIT_LAG = 5.0
# transcript chunks of several microphones can be written out of order, so follow mode re-reads
# the last minute of them; other types are written in time order by their synchronizer
LOOKBACK = {ASR_TRANSCRIPTION: 60.0}
IPS_JOIN_WAIT = 3.0
MAX_ERRORS = 5
FLOOR_KEEP = 2000
# the point times are microseconds, so a range that must take its upper bound stops one later
EDGE = 1e-6
# a shared feed outlives its last follower by this long, and holds this many batches per follower
FEED_GRACE = 10.0
FEED_QUEUE = 300
FEED_READY_WAIT = 30.0
# how far before the oldest cursor the feed may still send a record: the transcript look-back, an
# IPS window waiting for its other records, and a margin
FEED_OVERLAP = max(LOOKBACK.values()) + IPS_JOIN_WAIT + 5.0


def sse(event: str, data) -> str:
    try:
        text = json.dumps(data, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        from openmmla.analytics.report.common import jsonable
        text = json.dumps(jsonable(data), allow_nan=False, separators=(",", ":"))
    return f"event: {event}\ndata: {text}\n\n"


def _r3(value):
    return None if value is None else round(float(value), 3)


def _emit_time(event_type: str, record: dict) -> float:
    key = "window_end_time" if event_type == ASR_TRANSCRIPTION else "window_start_time"
    value = record.get(key)
    if value is None:
        value = record.get("_t")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(record.get("_t") or 0.0)


def _record_time(record: dict) -> float | None:
    try:
        return float(record["_t"])
    except (KeyError, TypeError, ValueError):
        return None


def _window_start(record: dict, length: float = 1.0) -> float | None:
    """the window start merge_ips groups a record under: its window start, else its point time
    less the window's length."""
    try:
        return float(record["window_start_time"])
    except (KeyError, TypeError, ValueError):
        stamp = _record_time(record)
        return None if stamp is None else stamp - length


def _key(event_type: str, record: dict) -> tuple:
    """what identifies a record across reads: its type, window start and point time."""
    return (event_type, record.get("window_start_time"), _record_time(record))


def _usable_basis(floor) -> dict | None:
    """a floor object of a cached space part, unless it is the camera x-z fallback."""
    if not isinstance(floor, dict) or floor.get("method") == "camera-xz":
        return None
    if not all(isinstance(floor.get(key), (list, tuple)) for key in ("g", "ex", "ef")):
        return None
    return floor


def _gevent_active() -> bool:
    try:
        from gevent import monkey
        return bool(monkey.is_module_patched("socket"))
    except Exception:
        return False


def fetch_many(fetch, ranges: dict) -> dict:
    """{event_type: rows} for {event_type: (begin, end)}; the queries run side by side under gevent,
    so one poll costs about one round trip to InfluxDB rather than six."""
    if len(ranges) > 1 and _gevent_active():
        import gevent

        def read(event_type, begin, end):
            # the error comes back as a value: a greenlet dying of it would print a traceback per type
            try:
                return True, fetch(event_type, begin, end)
            except Exception as exc:
                return False, exc

        jobs = {t: gevent.spawn(read, t, begin, end) for t, (begin, end) in ranges.items()}
        try:
            gevent.joinall(list(jobs.values()))
            out = {}
            for event_type, job in jobs.items():
                ok, value = job.get()
                if not ok:
                    raise value
                out[event_type] = value
            return out
        finally:
            for job in jobs.values():
                if not job.ready():
                    job.kill(block=False)
    return {t: fetch(t, begin, end) for t, (begin, end) in ranges.items()}


class FloorTracker:
    """the floor basis of a stream: fixed once known, else built from rotations as they come."""

    def __init__(self, basis: dict | None):
        self.basis = basis
        self._rotations: list = []

    def adopt(self, basis: dict | None) -> bool:
        """take a basis found elsewhere when none is known yet; whether it was taken."""
        if self.basis is not None or basis is None:
            return False
        self.basis = basis
        self._rotations = []
        return True

    def add(self, rotations: list) -> dict | None:
        """feed rotation records; returns the basis the first time one can be built."""
        if self.basis is not None or not rotations:
            return None
        self._rotations.extend(rotations)
        if len(self._rotations) > FLOOR_KEEP:
            self._rotations = self._rotations[-FLOOR_KEEP:]
        try:
            from openmmla.analytics.report.space import floor_basis
            basis = floor_basis(self._rotations, min_obs=MIN_FLOOR_OBS)
        except Exception as exc:
            logger.warning("floor basis failed: %s", exc)
            return None
        if basis is not None:
            self.basis = basis
            self._rotations = []
            return basis
        return None


class Slimmer:
    """turns raw records into the slim stream formats. It keeps the worn microphones' recent
    levels (for the word classes), and the voice keys unless it is given shared ones."""

    def __init__(self, voices=None):
        from openmmla.analytics.report import common, live
        self.live = live
        self.voices = voices if voices is not None else common.VoiceKeys()
        self.worn = live.WornWords()
        self._warned = set()

    def _each(self, name: str, fn, items) -> list:
        out = []
        for item in items:
            try:
                slim = fn(item)
            except Exception as exc:
                if name not in self._warned:
                    self._warned.add(name)
                    logger.warning("could not slim a %s record: %s: %s", name, type(exc).__name__, exc)
                continue
            if slim is not None:
                out.append((slim, item))
        return out

    def _ips_windows(self, records: dict) -> tuple[list, dict]:
        """the merged IPS windows of the records, and the keys of the records behind each."""
        parts = [records.get(t) or [] for t in IPS_TYPES]
        try:
            windows = self.live.merge_ips(*parts)
        except Exception as exc:
            logger.warning("could not merge IPS windows: %s", exc)
            return [], {}
        keys: dict = {}
        for event_type, rows in zip(IPS_TYPES, parts):
            for record in rows:
                start = _window_start(record, 1.0)
                if start is not None:
                    keys.setdefault(start, []).append(_key(event_type, record))
        return windows, {start: tuple(found) for start, found in keys.items()}

    def items(self, records: dict, basis: dict | None) -> dict:
        """{"asr" | "tr" | "ips" | "vfa": [(slim, keys, window)]}, each list sorted by `t`: keys are
        the _key of the raw records behind the slim one, window the merged IPS window (to slim it
        again on another floor basis), None for the other kinds."""
        live = self.live
        recognitions = records.get(ASR_RECOGNITION) or []
        chunks = records.get(ASR_TRANSCRIPTION) or []
        asr = [(slim, (_key(ASR_RECOGNITION, r),), None)
               for slim, r in self._each("recognition", live.slim_recognition, recognitions)]
        # the buckets first: a worn chunk's words are judged on the levels they hold
        self.worn.add_recognitions(recognitions)
        classes = self.worn.classify(chunks)
        tr = [(slim, (_key(ASR_TRANSCRIPTION, r),), None)
              for slim, r in self._each("transcription",
                                        lambda r: live.slim_transcription(r, self.voices, classes.get(id(r))), chunks)]
        windows, window_keys = self._ips_windows(records)
        ips = [(slim, window_keys.get(w.get("t"), ()), w)
               for slim, w in self._each("ips", lambda w: live.slim_ips(w, basis), windows)]
        vfa = [(slim, (_key(VFA_FEATURES, r),), None)
               for slim, r in self._each("vfa", live.slim_vfa, records.get(VFA_FEATURES) or [])]
        out = {}
        for name, found in (("asr", asr), ("tr", tr), ("ips", ips), ("vfa", vfa)):
            found.sort(key=lambda item: (item[0].get("t") is None, item[0].get("t") or 0.0))
            out[name] = found
        return out

    def batch(self, records: dict, basis: dict | None) -> dict:
        return _plain(self.items(records, basis))


def _plain(items: dict) -> dict:
    return {name: [item[0] for item in found] for name, found in items.items()}


class IpsJoiner:
    """holds IPS records of a window until its translation, rotation and relation have all come.

    The synchronizer writes the three a few milliseconds apart, so a poll can land between them; a
    window still incomplete after a few seconds is sent as it is."""

    def __init__(self):
        self.pending: dict = {}

    def add(self, event_type: str, records: list, now: float) -> None:
        for record in records:
            key = record.get("window_start_time")
            if key is None:
                continue
            entry = self.pending.setdefault(key, {"since": now})
            entry[event_type] = record

    def ready(self, now: float, flush: bool = False) -> dict:
        out = {IPS_TRANSLATION: [], IPS_ROTATION: [], IPS_RELATION: []}
        for key in sorted(self.pending):
            entry = self.pending[key]
            complete = all(t in entry for t in out)
            if flush or complete or now - entry["since"] >= IPS_JOIN_WAIT:
                for t in out:
                    if t in entry:
                        out[t].append(entry[t])
                del self.pending[key]
        return out


class PollState:
    """what a follow poll keeps from one poll to the next: per event type the newest point time
    read (its cursor) and the stamps of the last minute (transcripts are read again that far
    back), the IPS windows still waiting for their other records, the floor basis and the
    slimmer. `lower`: nothing stamped before it is sent (the history the viewer asked for)."""

    def __init__(self, cursors: dict, slimmer: Slimmer, tracker: FloorTracker, *, lower=None, seen=None):
        self.cursors = {t: cursors.get(t) for t in EVENT_TYPES}
        self.seen = {t: set((seen or {}).get(t) or ()) for t in EVENT_TYPES}
        self.slimmer = slimmer
        self.tracker = tracker
        self.joiner = IpsJoiner()
        self.lower = lower

    @property
    def clock(self) -> float | None:
        stamps = [c for c in self.cursors.values() if c is not None]
        return max(stamps) if stamps else None

    def poll(self, fetch, wall: float) -> tuple[dict, dict | None, bool]:
        """one poll: (Slimmer.items of what is new, the floor basis when this poll found it,
        whether any record came). Raises InfluxUnavailable when InfluxDB cannot be asked."""
        ranges = {t: (None if c is None else c - LOOKBACK.get(t, 0.0), None) for t, c in self.cursors.items()}
        fetched = fetch_many(fetch, ranges)
        records = {}
        for event_type in EVENT_TYPES:
            cursor = self.cursors[event_type]
            lookback = LOOKBACK.get(event_type, 0.0)
            seen = self.seen[event_type]
            fresh = []
            for row in fetched.get(event_type) or ():
                stamp = _record_time(row)
                if stamp is None or stamp in seen:
                    continue
                if cursor is not None and stamp <= cursor - lookback:
                    continue
                if self.lower is not None and stamp < self.lower:
                    # the transcript look-back must not reach before the history asked for
                    continue
                seen.add(stamp)
                fresh.append(row)
            if fresh:
                newest = max(_record_time(r) for r in fresh)
                self.cursors[event_type] = newest if cursor is None else max(cursor, newest)
                horizon = self.cursors[event_type] - lookback - 5.0
                self.seen[event_type] = {s for s in seen if s >= horizon}
            records[event_type] = fresh
        new = any(records[t] for t in EVENT_TYPES)
        for event_type in IPS_TYPES:
            self.joiner.add(event_type, records[event_type], wall)
        ready = self.joiner.ready(wall)
        found = self.tracker.add(ready[IPS_ROTATION]) if self.tracker.basis is None else None
        records.update(ready)
        return self.slimmer.items(records, self.tracker.basis), found, new


def _batch_event(clock, backfill: bool, slim: dict) -> str:
    body = {"clock": _r3(clock), "backfill": backfill}
    body.update(slim)
    return sse("batch", body)


def _has_any(slim: dict) -> bool:
    return any(slim.get(key) for key in ("asr", "tr", "ips", "vfa"))


def clamp_speed(value) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0
    if speed != speed:
        return 1.0
    return min(16.0, max(1.0, speed))


def clamp_backfill(value) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return BACKFILL_DEFAULT
    if seconds != seconds:
        return BACKFILL_DEFAULT
    return min(BACKFILL_MAX, max(0.0, seconds))


def _error_message(exc: BaseException) -> str:
    from openmmla.analytics.report.common import InfluxUnavailable
    return str(exc) if isinstance(exc, InfluxUnavailable) else f"{type(exc).__name__}: {exc}"


def live_stream(client, sid: str, *, mode: str, t0, t1, group_id, at=None, speed=1.0,
                backfill=BACKFILL_DEFAULT, floor=None, last_event_fn=None, sleep=time.sleep,
                now=time.time, shared=None, hub=None):
    """the SSE text of one connection; ends on replay end, on repeated InfluxDB errors, or when
    the client goes away (GeneratorExit at a yield). Follow mode joins the session's shared feed
    when gevent runs (`shared` None), or when `shared` is true."""
    from openmmla.analytics.report import common

    speed = clamp_speed(speed)
    backfill = clamp_backfill(backfill)

    def fetch(event_type, start, end):
        return common.fetch(client, sid, event_type, start, end)

    def last_event():
        if last_event_fn is not None:
            return last_event_fn()
        return common.last_event_time(client, sid)

    def newest_times():
        return common.last_event_times(client, sid)

    try:
        if mode == "replay":
            yield from _replay(fetch, last_event, sid, Slimmer(), FloorTracker(_usable_basis(floor)), t0=t0, t1=t1,
                               group_id=group_id, at=at, speed=speed, backfill=backfill, sleep=sleep, now=now)
        elif _gevent_active() if shared is None else shared:
            yield from _follow_shared(hub or FEEDS, fetch, newest_times, sid, t0=t0, t1=t1, group_id=group_id,
                                      backfill=backfill, floor=floor, now=now)
        else:
            yield from _follow(fetch, last_event, sid, Slimmer(), FloorTracker(_usable_basis(floor)), t0=t0, t1=t1,
                               group_id=group_id, backfill=backfill, sleep=sleep, now=now)
    except GeneratorExit:
        raise
    except Exception as exc:
        logger.exception("stream of %s failed", sid)
        yield sse("end", {"reason": "error", "message": _error_message(exc)})


def _hello(sid, mode, t0, t1, live, speed, clock, basis, group_id) -> str:
    return sse("hello", {"session": sid, "mode": mode, "t0": _r3(t0), "t1": _r3(t1), "live": bool(live),
                         "speed": speed, "clock": _r3(clock), "floor": basis, "group_id": group_id})


def _status(clock, last, wall) -> str:
    live = last is not None and wall - last < LIVE_SECONDS
    lag = None if last is None else round(wall - last, 3)
    return sse("status", {"clock": _r3(clock), "live": live, "lag": lag, "last_event": _r3(last)})


def _end_of(t1, last):
    ends = [float(x) for x in (t1, last) if x is not None]
    return max(ends) if ends else None


def _follow(fetch, last_event, sid, slimmer, tracker, *, t0, t1, group_id, backfill, sleep, now):
    """follow mode on its own: this connection reads its backfill and then polls for itself (used
    when gevent does not run, so no feed can be shared)."""
    from openmmla.analytics.report.common import InfluxUnavailable

    last = last_event()
    wall = now()
    clock = last if last is not None else wall
    cursors = {t: last for t in EVENT_TYPES}
    seen = {t: set() for t in EVENT_TYPES}
    prefetched_rotations = None
    start = None if last is None else last - backfill
    if last is not None and backfill > 0 and tracker.basis is None:
        # the basis goes out with hello, so the backfilled positions are already in floor coordinates
        prefetched_rotations = fetch(IPS_ROTATION, start, last + 0.001)
        tracker.add(prefetched_rotations)
    live = last is not None and wall - last < LIVE_SECONDS
    yield _hello(sid, "follow", t0, _end_of(t1, last), live, 1.0, clock, tracker.basis, group_id)

    if last is not None and backfill > 0:
        chunk_start = start
        while chunk_start <= last:
            chunk_end = chunk_start + BACKFILL_CHUNK
            if chunk_end >= last:
                # the last chunk takes the newest record itself (ranges end before their stop)
                chunk_end = last + 0.001
            wanted = {t: (chunk_start, chunk_end) for t in EVENT_TYPES
                      if not (t == IPS_ROTATION and prefetched_rotations is not None)}
            records = fetch_many(fetch, wanted)
            if prefetched_rotations is not None:
                records[IPS_ROTATION] = [r for r in prefetched_rotations
                                         if chunk_start <= (_record_time(r) or 0) < chunk_end]
            for event_type in EVENT_TYPES:
                for record in records[event_type]:
                    stamp = _record_time(record)
                    if stamp is not None:
                        seen[event_type].add(stamp)
            if tracker.basis is None and prefetched_rotations is None:
                basis = tracker.add(records[IPS_ROTATION])
                if basis is not None:
                    yield sse("floor", basis)
            yield _batch_event(min(chunk_end, last), True, slimmer.batch(records, tracker.basis))
            chunk_start = chunk_end
            if chunk_end >= last:
                break
        # each type resumes after its own newest record: the types come from different
        # synchronizers, and video features trail speech by a second or two, so a cursor at the
        # newest event of any type would skip the video records still to come below it
        for event_type in EVENT_TYPES:
            if seen[event_type]:
                cursors[event_type] = max(seen[event_type])
    state = PollState(cursors, slimmer, tracker, lower=start, seen=seen)

    errors = 0
    last_ping = last_status = now()
    last_new = now() if live else 0.0
    while True:
        wall = now()
        tick = FOLLOW_TICK if wall - last_new < FOLLOW_IDLE_AFTER else FOLLOW_IDLE_TICK
        sleep(tick)
        wall = now()
        try:
            items, found, new = state.poll(fetch, wall)
            errors = 0
        except InfluxUnavailable as exc:
            errors += 1
            logger.warning("follow poll of %s failed (%d in a row): %s", sid, errors, exc)
            if errors >= MAX_ERRORS:
                yield sse("end", {"reason": "error", "message": str(exc)})
                return
            continue
        if new:
            last_new = wall
        if found is not None:
            yield sse("floor", found)
        clock = state.clock if state.clock is not None else clock
        slim = _plain(items)
        if _has_any(slim):
            yield _batch_event(clock, False, slim)
        if wall - last_status >= STATUS_EVERY:
            last_status = wall
            yield _status(clock, state.clock, wall)
        if wall - last_ping >= PING_EVERY:
            last_ping = wall
            yield ": ping\n\n"


# ---- the shared follow feed ----

class FeedMessage:
    """what a feed hands its followers: a batch (its SSE text, ready to send as it is, plus the
    slim records with their keys for a follower that must leave some out), a floor basis, or the
    end of the feed."""

    __slots__ = ("kind", "text", "items", "basis", "clock", "oldest")

    def __init__(self, kind: str, text: str, items=None, basis=None, clock=None, oldest=None):
        self.kind = kind
        self.text = text
        self.items = items
        self.basis = basis
        self.clock = clock
        self.oldest = oldest

    @classmethod
    def batch(cls, items: dict, clock, basis) -> "FeedMessage | None":
        if not any(items.values()):
            return None
        stamps = [key[2] for found in items.values() for item in found for key in item[1] if key[2] is not None]
        return cls("batch", _batch_event(clock, False, _plain(items)), items, basis, clock,
                   min(stamps) if stamps else None)


class Follower:
    """one connection on a shared feed: its queue of the feed's messages, the floor basis its page
    knows, the oldest stamp it asked for, and the keys of the records its own backfill sent (with
    the newest of their stamps), so the feed's batches leave those out."""

    def __init__(self, maxsize: int | None = None):
        import gevent.queue
        self.queue = gevent.queue.Queue(FEED_QUEUE if maxsize is None else maxsize)
        self.dropped = False
        self.basis = None
        self.lower = None
        self.sent: set = set()
        self.sent_until = None

    def note_sent(self, event_type: str, records) -> None:
        for record in records:
            key = _key(event_type, record)
            if key[2] is None:
                continue
            self.sent.add(key)
            if self.sent_until is None or key[2] > self.sent_until:
                self.sent_until = key[2]

    def next(self, timeout: float):
        """the next message, None when none came within `timeout` seconds."""
        import gevent.queue
        try:
            return self.queue.get(timeout=max(0.0, timeout))
        except gevent.queue.Empty:
            return None

    def render(self, message: FeedMessage) -> str | None:
        """the SSE text of a feed batch for this follower: the feed's own text when nothing in it
        concerns this follower alone, else the batch without what the backfill already sent or
        what lies before the history asked for, with the IPS windows on this page's floor basis."""
        moved = message.basis is not self.basis and bool(message.items.get("ips"))
        oldest = message.oldest
        overlaps = bool(self.sent) and (oldest is None or self.sent_until is None or oldest <= self.sent_until)
        early = self.lower is not None and (oldest is None or oldest < self.lower)
        if not (moved or overlaps or early):
            return message.text
        body = {}
        for name, found in message.items.items():
            kept = []
            for slim, keys, window in found:
                if overlaps and any(key in self.sent for key in keys):
                    continue
                stamps = [key[2] for key in keys if key[2] is not None]
                if early and stamps and max(stamps) < self.lower:
                    continue
                if name == "ips" and moved and window is not None:
                    try:
                        from openmmla.analytics.report.live import slim_ips
                        slim = slim_ips(window, self.basis)
                    except Exception as exc:
                        logger.warning("could not slim an IPS window again: %s", exc)
                        continue
                kept.append(slim)
            body[name] = kept
        if not _has_any(body):
            return None
        return _batch_event(message.clock, False, body)

    def forget_sent(self, cursors: dict) -> None:
        """drop the backfill's keys once no feed batch can hold those records any more."""
        if not self.sent:
            return
        stamps = [c for c in cursors.values() if c is not None]
        if stamps and self.sent_until is not None and min(stamps) - FEED_OVERLAP > self.sent_until:
            self.sent.clear()


class SessionFeed:
    """the one poller of a session in follow mode, shared by all its followers (see the module
    docstring). It runs in its own greenlet."""

    def __init__(self, hub: "FeedHub", sid: str, fetch, newest_times, floor=None, now=time.time):
        self.hub = hub
        self.sid = sid
        self.fetch = fetch
        self.newest_times = newest_times
        self.now = now
        self.followers: set = set()
        self.slimmer = Slimmer()
        self.tracker = FloorTracker(_usable_basis(floor))
        self.state: PollState | None = None
        import gevent.event
        self.ready = gevent.event.Event()
        self.error = None
        self.closed = False
        self.empty_since = None
        self.started_at = now()
        self.polled_at = None
        self.polls = 0
        self.greenlet = None

    @property
    def voices(self):
        return self.slimmer.voices

    @property
    def basis(self) -> dict | None:
        return self.tracker.basis

    @property
    def clock(self) -> float | None:
        return None if self.state is None else self.state.clock

    def cursors(self) -> dict:
        return dict(self.state.cursors) if self.state is not None else {t: None for t in EVENT_TYPES}

    def pending_windows(self) -> set:
        """the window starts of IPS records the feed read but has not sent yet (waiting for the
        window's other records): they will come in a feed batch, whole."""
        return set(self.state.joiner.pending) if self.state is not None else set()

    def adopt(self, basis: dict | None) -> dict | None:
        """take a floor basis a follower found (from the cached space part or its backfill) when
        the feed has none; the followers hear of it. Returns the feed's basis."""
        if self.tracker.adopt(basis):
            self._publish(FeedMessage("floor", sse("floor", basis), basis=basis))
        return self.tracker.basis

    def subscribe(self) -> Follower:
        follower = Follower()
        self.followers.add(follower)
        self.empty_since = None
        return follower

    def unsubscribe(self, follower: Follower) -> None:
        self.followers.discard(follower)
        if not self.followers and self.empty_since is None:
            self.empty_since = self.now()

    def _publish(self, message: FeedMessage) -> None:
        import gevent.queue
        for follower in list(self.followers):
            try:
                follower.queue.put_nowait(message)
            except gevent.queue.Full:
                follower.dropped = True
                self.unsubscribe(follower)
                logger.info("a follower of %s fell %d batches behind and was dropped", self.sid, FEED_QUEUE)

    def _start(self) -> None:
        """the cursors from each type's newest record, the transcripts and recognition buckets of
        the last few minutes (the look-back must not send those again, and the worn microphones'
        word classes need their levels), and rotations for the floor basis when none is cached."""
        newest = self.newest_times() or {}
        cursors = {t: newest.get(t) for t in EVENT_TYPES}
        # an IPS window's three records are written a moment apart: the three types start from the
        # oldest of their newest records, so a window caught half written is read whole later
        ips = [cursors[t] for t in IPS_TYPES if cursors[t] is not None]
        for event_type in IPS_TYPES:
            if cursors[event_type] is not None:
                cursors[event_type] = min(ips)
        seen = {t: set() for t in EVENT_TYPES}
        ranges = {}
        keep = self.slimmer.worn.KEEP
        for event_type in (ASR_RECOGNITION, ASR_TRANSCRIPTION):
            if cursors[event_type] is not None:
                ranges[event_type] = (cursors[event_type] - keep, cursors[event_type] + EDGE)
        if self.tracker.basis is None and cursors[IPS_ROTATION] is not None:
            ranges[IPS_ROTATION] = (cursors[IPS_ROTATION] - BACKFILL_DEFAULT, cursors[IPS_ROTATION] + EDGE)
        fetched = fetch_many(self.fetch, ranges) if ranges else {}
        self.slimmer.worn.add_recognitions(fetched.get(ASR_RECOGNITION) or [])
        chunks = fetched.get(ASR_TRANSCRIPTION) or []
        self.slimmer.worn.classify(chunks)
        seen[ASR_TRANSCRIPTION] = {s for s in (_record_time(r) for r in chunks) if s is not None}
        self.tracker.add(fetched.get(IPS_ROTATION) or [])
        self.state = PollState(cursors, self.slimmer, self.tracker, seen=seen)

    def run(self) -> None:
        import gevent
        from openmmla.analytics.report.common import InfluxUnavailable
        try:
            try:
                self._start()
            except Exception as exc:
                self.error = _error_message(exc)
                logger.warning("live feed of %s could not start: %s", self.sid, self.error)
                return
            finally:
                self.ready.set()
            errors = 0
            last_new = self.now()
            while True:
                wall = self.now()
                gevent.sleep(FOLLOW_TICK if wall - last_new < FOLLOW_IDLE_AFTER else FOLLOW_IDLE_TICK)
                wall = self.now()
                if not self.followers:
                    if self.empty_since is None:
                        self.empty_since = wall
                    if wall - self.empty_since >= FEED_GRACE:
                        return
                    # nobody listens: no queries until someone does, the next poll catches up
                    continue
                try:
                    items, found, new = self.state.poll(self.fetch, wall)
                    errors = 0
                except InfluxUnavailable as exc:
                    errors += 1
                    logger.warning("live feed of %s: poll failed (%d in a row): %s", self.sid, errors, exc)
                    if errors >= MAX_ERRORS:
                        self.error = str(exc)
                        self._publish(FeedMessage("end", sse("end", {"reason": "error", "message": str(exc)})))
                        return
                    continue
                self.polled_at = wall
                self.polls += 1
                if new:
                    last_new = wall
                if found is not None:
                    self._publish(FeedMessage("floor", sse("floor", found), basis=found))
                message = FeedMessage.batch(items, self.state.clock, self.tracker.basis)
                if message is not None:
                    self._publish(message)
        except Exception as exc:
            logger.exception("live feed of %s failed", self.sid)
            self.error = _error_message(exc)
            self._publish(FeedMessage("end", sse("end", {"reason": "error", "message": self.error})))
        finally:
            self._close()

    def _close(self) -> None:
        self.closed = True
        self.hub.remove(self)


class FeedHub:
    """the shared follow feeds of this process, one per followed session."""

    def __init__(self, feed_class=SessionFeed):
        self.feeds: dict = {}
        self.feed_class = feed_class

    def join(self, sid: str, fetch, newest_times, floor=None, now=time.time) -> tuple[SessionFeed, Follower]:
        """the session's feed (started when it has none) and a new follower on it. Nothing here
        waits, so under gevent a feed cannot close between finding it and joining it."""
        import gevent
        feed = self.feeds.get(sid)
        if feed is None or feed.closed:
            feed = self.feed_class(self, sid, fetch, newest_times, floor, now)
            self.feeds[sid] = feed
            feed.greenlet = gevent.spawn(feed.run)
        return feed, feed.subscribe()

    def remove(self, feed: SessionFeed) -> None:
        if self.feeds.get(feed.sid) is feed:
            del self.feeds[feed.sid]

    def newest(self, sid: str, max_age: float, now: float | None = None) -> float | None:
        """the newest event time of a session whose feed polled in the last `max_age` seconds,
        else None (then ask InfluxDB)."""
        feed = self.feeds.get(sid)
        now = time.time() if now is None else now
        if feed is None or feed.closed or feed.polled_at is None or now - feed.polled_at > max_age:
            return None
        return feed.clock

    def stats(self) -> dict:
        feeds = [feed for feed in list(self.feeds.values()) if not feed.closed]
        return {"feeds": len(feeds), "followers": sum(len(feed.followers) for feed in feeds)}


FEEDS = FeedHub()


def _follow_shared(hub: FeedHub, fetch, newest_times, sid, *, t0, t1, group_id, backfill, floor, now):
    """follow mode on the session's shared feed: hello, this follower's own backfill up to where
    the feed has read each type, then the feed's batches without what the backfill already held."""
    feed, follower = hub.join(sid, fetch, newest_times, floor, now)
    try:
        if not feed.ready.wait(FEED_READY_WAIT):
            yield sse("end", {"reason": "error", "message": "InfluxDB did not answer in time."})
            return
        if feed.error is not None and feed.state is None:
            yield sse("end", {"reason": "error", "message": feed.error})
            return
        # where the feed has read each type: up to there this follower reads its own history,
        # after it the feed's batches carry everything, as this follower already listens
        cursors = feed.cursors()
        pending = feed.pending_windows()
        stamps = [c for c in cursors.values() if c is not None]
        last = max(stamps) if stamps else None
        start = None if last is None else last - backfill
        if start is not None:
            follower.lower = min([start] + stamps)
        basis = feed.basis
        if basis is None:
            basis = feed.adopt(_usable_basis(floor))
        rotations = None
        upper = cursors[IPS_ROTATION]
        if basis is None and start is not None and backfill > 0 and upper is not None and upper >= start:
            # the basis goes out with hello, so the backfilled positions are already in floor coordinates
            rotations = [r for r in fetch(IPS_ROTATION, start, upper + EDGE) if _window_start(r) not in pending]
            basis = feed.adopt(FloorTracker(None).add(rotations)) or feed.basis
        follower.basis = basis
        wall = now()
        live = last is not None and wall - last < LIVE_SECONDS
        yield _hello(sid, "follow", t0, _end_of(t1, last), live, 1.0, last if last is not None else wall, basis,
                     group_id)

        if start is not None and backfill > 0:
            slimmer = Slimmer(voices=feed.voices)
            if feed.slimmer.worn.active:
                # a worn chunk's words are judged on the levels of the minutes before it too, as the
                # feed judges them, so the backfill reads those first (and sends none of them)
                keep = slimmer.worn.KEEP
                context = fetch_many(fetch, {t: (start - keep, start) for t in (ASR_RECOGNITION, ASR_TRANSCRIPTION)
                                             if cursors[t] is not None})
                slimmer.worn.add_recognitions(context.get(ASR_RECOGNITION) or [])
                slimmer.worn.classify(context.get(ASR_TRANSCRIPTION) or [])
            chunk_start = start
            while not follower.dropped:
                chunk_end = chunk_start + BACKFILL_CHUNK
                final = chunk_end >= last
                wanted = {}
                for event_type, top in cursors.items():
                    if top is None or top < chunk_start or (event_type == IPS_ROTATION and rotations is not None):
                        continue
                    wanted[event_type] = (chunk_start, top + EDGE if final or top < chunk_end else chunk_end)
                records = fetch_many(fetch, wanted)
                for event_type in EVENT_TYPES:
                    records.setdefault(event_type, [])
                if rotations is not None:
                    stop = upper + EDGE if final else chunk_end
                    records[IPS_ROTATION] = [r for r in rotations if chunk_start <= (_record_time(r) or 0) < stop]
                for event_type in IPS_TYPES:
                    # a window the feed still holds comes whole in its batch
                    records[event_type] = [r for r in records[event_type] if _window_start(r) not in pending]
                for event_type in EVENT_TYPES:
                    follower.note_sent(event_type, records[event_type])
                if follower.basis is None and feed.basis is not None:
                    follower.basis = feed.basis
                    yield sse("floor", follower.basis)
                yield _batch_event(min(chunk_end, last), True, slimmer.batch(records, follower.basis))
                if final:
                    break
                chunk_start = chunk_end

        last_ping = last_status = now()
        while True:
            if follower.dropped:
                yield sse("end", {"reason": "error",
                                  "message": "This view fell behind the live stream; it reconnects."})
                return
            wall = now()
            message = follower.next(min(last_status + STATUS_EVERY, last_ping + PING_EVERY) - wall)
            if follower.dropped:
                continue
            if message is None and feed.closed:
                # the feed stopped without a word (the server shuts down): the page reconnects
                yield sse("end", {"reason": "error", "message": "The live feed of this session stopped; it reconnects."})
                return
            if message is not None:
                if message.kind == "end":
                    yield message.text
                    return
                if message.kind == "floor":
                    if follower.basis is None:
                        follower.basis = message.basis
                        yield message.text
                else:
                    text = follower.render(message)
                    follower.forget_sent(feed.cursors())
                    if text:
                        yield text
            wall = now()
            if wall - last_status >= STATUS_EVERY:
                last_status = wall
                clock = feed.clock
                yield _status(clock if clock is not None else wall, clock, wall)
            if wall - last_ping >= PING_EVERY:
                last_ping = wall
                yield ": ping\n\n"
    finally:
        feed.unsubscribe(follower)


# ---- replay ----

class ReplayBuffer:
    """records read ahead of the virtual clock, released in emit-time order.

    Under gevent the read-ahead runs in a greenlet, so batches keep flowing from what is already
    buffered while the next piece arrives (a minute of video features is about 1.5 MB); elsewhere
    it reads in place."""

    def __init__(self, fetch, start: float, lower: float):
        self.fetch = fetch
        self.fetched_until = start
        self.lower = lower
        self.pending = {t: [] for t in EVENT_TYPES}
        self._job = None
        self._job_until = None
        self._background = _gevent_active()

    def _read(self, begin: float, until: float) -> dict:
        rows = fetch_many(self.fetch, {t: (begin, until) for t in EVENT_TYPES})
        return {t: [r for r in rows[t] if _emit_time(t, r) > self.lower] for t in EVENT_TYPES}

    def _merge(self, rows: dict, until: float) -> None:
        for event_type, new_rows in rows.items():
            if new_rows:
                merged = self.pending[event_type] + new_rows
                merged.sort(key=lambda r, _t=event_type: _emit_time(_t, r))
                self.pending[event_type] = merged
        self.fetched_until = max(self.fetched_until, until)

    def _collect(self, wait: bool) -> None:
        """take in a finished read-ahead (re-raising its error); with `wait`, wait for it first."""
        if self._job is None:
            return
        if not wait and not self._job.ready():
            return
        job, until = self._job, self._job_until
        self._job = self._job_until = None
        self._merge(job.get(), until)

    def ensure(self, until: float) -> None:
        """everything stored before `until` is in the buffer when this returns."""
        self._collect(wait=True)
        if until > self.fetched_until:
            begin = self.fetched_until
            self._merge(self._read(begin, until), until)

    def prefetch(self, until: float, piece: float) -> None:
        """start reading the next `piece` seconds towards `until` without waiting (in place when
        gevent is not running); small pieces keep batches flowing on a slow link."""
        self._collect(wait=False)
        if self._job is not None or until <= self.fetched_until:
            return
        begin = self.fetched_until
        until = min(until, begin + piece)
        if self._background:
            import gevent
            self._job, self._job_until = gevent.spawn(self._read, begin, until), until
        else:
            self._merge(self._read(begin, until), until)

    def close(self) -> None:
        if self._job is not None:
            try:
                self._job.kill(block=False)
            except Exception:
                pass
            self._job = None

    def pop_due(self, clock: float) -> dict:
        out = {}
        for event_type, rows in self.pending.items():
            cut = 0
            while cut < len(rows) and _emit_time(event_type, rows[cut]) <= clock:
                cut += 1
            out[event_type], self.pending[event_type] = rows[:cut], rows[cut:]
        return out


def _replay(fetch, last_event, sid, slimmer, tracker, *, t0, t1, group_id, at, speed, backfill, sleep, now):
    from openmmla.analytics.report.common import InfluxUnavailable

    last = last_event()
    known = [float(x) for x in (t0, t1, last) if x is not None]
    if not known:
        yield _hello(sid, "replay", t0, t1, False, speed, at, tracker.basis, group_id)
        yield sse("end", {"reason": "replay_end", "message": "This session has no measurements."})
        return
    end_bound = max(known)
    start_at = float(t0) if t0 is not None else min(known)
    try:
        clock = float(at) if at is not None else start_at
    except (TypeError, ValueError):
        clock = start_at
    if clock != clock:
        clock = start_at
    clock = min(max(clock, start_at), end_bound)

    if tracker.basis is None:
        tracker.add(fetch(IPS_ROTATION, clock - 300.0, clock + 300.0))
    wall = now()
    live = last is not None and wall - last < LIVE_SECONDS
    yield _hello(sid, "replay", t0, end_bound, live, speed, clock, tracker.basis, group_id)

    # with a backfill, [at - backfill, at] goes out first in 60 s chunks; with none, the previous
    # connection already sent everything up to `at`, so only records after it go out
    lower = clock - backfill
    buffer = ReplayBuffer(fetch, lower - 0.001, lower - 1e-6 if backfill > 0 else clock)
    try:
        if backfill > 0:
            boundary = lower
            while boundary < clock:
                boundary = min(boundary + BACKFILL_CHUNK, clock)
                buffer.ensure(boundary + MAX_EMIT_LAG)
                due = buffer.pop_due(boundary)
                if tracker.basis is None:
                    basis = tracker.add(due.get(IPS_ROTATION) or [])
                    if basis is not None:
                        yield sse("floor", basis)
                yield _batch_event(boundary, True, slimmer.batch(due, tracker.basis))
        buffer.ensure(clock + MAX_EMIT_LAG)
        yield from _replay_run(buffer, last_event, sid, slimmer, tracker, clock=clock, end_bound=end_bound,
                               speed=speed, sleep=sleep, now=now, error_type=InfluxUnavailable)
    finally:
        buffer.close()


def _replay_run(buffer, last_event, sid, slimmer, tracker, *, clock, end_bound, speed, sleep, now, error_type):
    """the playing part of a replay: the clock advances `speed` times wall time, but never past what
    has been read (it waits, like a video buffering, rather than skip records)."""
    ahead = max(30.0, 15.0 * speed)
    piece = max(10.0, 2.0 * speed)
    last_wall = last_ping = last_status = now()
    errors = 0
    while True:
        sleep(REPLAY_TICK)
        wall = now()
        proposed = clock + (wall - last_wall) * speed
        last_wall = wall
        if proposed > end_bound:
            fresh_last = last_event()
            if fresh_last is not None and fresh_last > end_bound:
                end_bound = fresh_last
        horizon = end_bound + MAX_EMIT_LAG + 0.001
        try:
            buffer.prefetch(min(max(proposed, clock) + MAX_EMIT_LAG + ahead, horizon), piece)
            errors = 0
        except error_type as exc:
            errors += 1
            logger.warning("replay read of %s failed (%d in a row): %s", sid, errors, exc)
            if errors >= MAX_ERRORS:
                yield sse("end", {"reason": "error", "message": str(exc)})
                return
            continue
        # a window is stored under its end time, so all that is due by t has been read once the
        # buffer passes t + MAX_EMIT_LAG
        safe = end_bound if buffer.fetched_until >= horizon else buffer.fetched_until - MAX_EMIT_LAG
        clock = max(clock, min(proposed, safe, end_bound))
        due = buffer.pop_due(clock)
        if tracker.basis is None:
            basis = tracker.add(due.get(IPS_ROTATION) or [])
            if basis is not None:
                yield sse("floor", basis)
        slim = slimmer.batch(due, tracker.basis)
        if _has_any(slim):
            yield _batch_event(clock, False, slim)
        if proposed > end_bound and clock >= end_bound:
            yield _status(end_bound, last_event(), wall)
            yield sse("end", {"reason": "replay_end", "message": "The replay reached the end of the session."})
            return
        if wall - last_status >= STATUS_EVERY:
            last_status = wall
            yield _status(clock, last_event(), wall)
        if wall - last_ping >= PING_EVERY:
            last_ping = wall
            yield ": ping\n\n"
