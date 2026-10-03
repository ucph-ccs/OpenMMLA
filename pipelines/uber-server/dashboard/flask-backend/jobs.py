"""Report jobs: computing a session's report parts and deciding when to recompute them.

Two jobs fill the cache. `light` reads ASR and IPS and writes the speech and space parts in a few
seconds; `video` reads every event type (VFA in time chunks, since a 1 h session is about 90 MB of
JSON) and runs the window fusion for the attention and timeline parts, which takes minutes. A job
runs on the Celery worker when one is listening on the dashboard's own queue, else in a process the
web process starts, so a dashboard without a worker still produces its report.

A cached part is served as long as InfluxDB has nothing newer than what it was computed from; a
live session's part is also served while it is younger than a minute (light) or five (video), so a
running session is recomputed at that pace rather than on every page poll.
"""

import logging
import os
import subprocess
import threading
import time
from collections import deque

from openmmla.analytics.report import REPORT_VERSION
from recordings import artifacts_root
from store import JOBS, PART_JOB, ReportStore

logger = logging.getLogger("dashboard.jobs")

DEAD_AFTER = 15 * 60.0
LIVE_MAX_AGE = {"light": 60.0, "video": 300.0}
WORKER_TTL = 30.0
WORKER_WAIT = 2.5
ERROR_RETRY_AFTER = 300.0
# local job processes at a time, per job: a video job reads a whole session's video features
LOCAL_SLOTS = {"light": 3, "video": 1}
STATUS_WRITE_INTERVAL = 1.0
# a job's space part keeps the camera placement it was computed with until the data changes, so a
# job waits longer for MongoDB than a page does
JOB_MONGO_TIMEOUT_MS = 5000

ASR_TYPES = ("asr_recognition", "asr_transcription")
IPS_TYPES = ("ips_translation", "ips_rotation", "ips_relation")
VFA_TYPES = ("vfa_features",)

_STEP_LABELS = {
    "asr_recognition": "Reading speech windows",
    "asr_transcription": "Reading transcripts",
    "ips_translation": "Reading badge positions",
    "ips_rotation": "Reading badge rotations",
    "ips_relation": "Reading facing relations",
    "vfa_features": "Reading video features",
}


class NoMeasurements(RuntimeError):
    """the session has no InfluxDB events, so there is nothing to compute."""


def _same_time(a, b) -> bool:
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) < 1e-3
    except (TypeError, ValueError):
        return False


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return True
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def is_active(status: dict | None, now: float | None = None) -> bool:
    """queued or running, and still alive: touched in the last 15 min, by a process that exists.

    A local job's pid only proves anything on the host that wrote it, which is this one: the cache
    folder is not shared between machines."""
    if not status or status.get("state") not in ("queued", "running"):
        return False
    now = time.time() if now is None else now
    try:
        if now - float(status.get("updated_at") or 0) > DEAD_AFTER:
            return False
    except (TypeError, ValueError):
        return False
    if status.get("runner") == "local" and not _pid_alive(status.get("pid")):
        return False
    return True


def part_is_fresh(envelope: dict | None, job: str, last_event: float | None, live: bool,
                  now: float | None = None) -> bool:
    if not envelope:
        return False
    if envelope.get("version") != REPORT_VERSION:
        # computed by code that has changed since: recompute it whatever the data
        return False
    if _same_time(envelope.get("source_last_event"), last_event):
        return True
    if live:
        now = time.time() if now is None else now
        try:
            return now - float(envelope.get("computed_at") or 0) < LIVE_MAX_AGE[job]
        except (TypeError, ValueError):
            return False
    return False


def job_summary(store: ReportStore, sid: str, job: str, last_event: float | None, live: bool,
                now: float | None = None) -> str:
    """running | ready | stale | error | missing, for the session list and the meta object (a job
    that recomputes parts still shown counts as running, so a refresh is visible)."""
    now = time.time() if now is None else now
    envelopes = [store.read_part_header(sid, part) for part in JOBS[job]]
    status = store.read_status(sid, job)
    if is_active(status, now):
        return "running"
    if all(envelopes) and all(part_is_fresh(env, job, last_event, live, now) for env in envelopes):
        return "ready"
    if any(envelopes):
        return "stale"
    if status and status.get("state") == "error":
        return "error"
    return "missing"


def _progress_body(status: dict | None) -> dict:
    status = status or {}
    return {"step": status.get("step"), "done": status.get("done"), "total": status.get("total")}


class _StatusWriter:
    """writes a job's progress to its status file, at most once a second unless the step changes."""

    def __init__(self, store: ReportStore, sid: str, job: str):
        self.store, self.sid, self.job = store, sid, job
        self._last_write = 0.0
        self._last_step = None

    def __call__(self, step: str, done=None, total=None, force: bool = False) -> None:
        now = time.time()
        if not force and step == self._last_step and now - self._last_write < STATUS_WRITE_INTERVAL:
            return
        self._last_step, self._last_write = step, now
        try:
            self.store.write_status(self.sid, self.job, state="running", step=step,
                                    done=None if done is None else int(done),
                                    total=None if total is None else int(total))
        except Exception as exc:
            logger.warning("could not write the %s status of %s: %s", self.job, self.sid, exc)


def _fetch_types(client, sid: str, types, start: float, end: float, report, chunked: bool) -> dict:
    from openmmla.analytics.report import common
    events = {}
    for event_type in types:
        label = _STEP_LABELS.get(event_type, f"Reading {event_type}")
        if chunked:
            total = max(1.0, end - start)
            report(label, 0, total, force=True)
            events[event_type] = common.fetch_chunked(
                client, sid, event_type, start, end, chunk=300.0,
                progress=lambda done, total_s, _label=label: report(_label, done, total_s))
        else:
            report(label, None, None, force=True)
            events[event_type] = common.fetch(client, sid, event_type, start, end)
    return events


def _session_span(client, sid: str):
    """(t0, t1, group id, last event) of the session from InfluxDB, or NoMeasurements."""
    from openmmla.analytics.report import common, sessions
    last_event = common.last_event_time(client, sid)
    if last_event is None:
        raise NoMeasurements("This session has no measurements in InfluxDB.")
    meta = sessions.session_meta(client, sid, None) or {}
    t0, t1 = meta.get("t0"), meta.get("t1")
    if t0 is None:
        raise NoMeasurements("This session has no ASR, IPS or VFA windows in InfluxDB.")
    if t1 is None:
        t1 = last_event
    return float(t0), float(t1), meta.get("group"), float(last_event)


def compute_job(client, store: ReportStore, sid: str, job: str, *, runner: str,
                mongo_doc: dict | None = None, repo_root: str | None = None) -> dict:
    """compute one job's parts into the store and return the final status.

    Every failure ends as an `error` status rather than an exception, so the caller (the precompute
    process or the Celery task) never loses the reason. In the light job each part is built on its own, so a
    bug in one leaves the other usable."""
    from openmmla.analytics.report.common import jsonable

    report = _StatusWriter(store, sid, job)
    store.write_status(sid, job, state="running", step="Starting", done=None, total=None,
                       started_at=round(time.time(), 3), runner=runner, error=None, pid=os.getpid())
    errors = []
    last_event = None
    try:
        t0, t1, group_id, last_event = _session_span(client, sid)
        # windows are stamped with their end time, and a transcript chunk ends up to ~30 s after
        # it starts, so read a little either side of the span
        start, end = t0 - 60.0, max(t1, last_event) + 1.0
        if job == "light":
            asr = _fetch_types(client, sid, ASR_TYPES, start, end, report, chunked=False)
            ips = _fetch_types(client, sid, IPS_TYPES, start, end, report, chunked=True)
            report("Computing speech", None, None, force=True)
            try:
                from openmmla.analytics.report.speech import build_speech
                speech = build_speech(asr["asr_recognition"], asr["asr_transcription"], t0, t1,
                                      group_id=group_id)
                store.write_part(sid, "speech", jsonable(speech), last_event, REPORT_VERSION)
            except Exception as exc:
                logger.exception("speech part of %s failed", sid)
                errors.append(f"speech: {type(exc).__name__}: {exc}")
            report("Computing space", None, None, force=True)
            try:
                from openmmla.analytics.report.sessions import ips_cameras
                from openmmla.analytics.report.space import build_space, main_camera_turn
                cameras = ips_cameras(mongo_doc) if mongo_doc else None
                # the main camera's turn sets the fallback plan of a session with too few badge rotations
                space = build_space(ips["ips_translation"], ips["ips_rotation"], ips["ips_relation"],
                                    t0, t1, cameras=cameras, main_turn=main_camera_turn(mongo_doc) if mongo_doc else 0)
                store.write_part(sid, "space", jsonable(space), last_event, REPORT_VERSION)
            except Exception as exc:
                logger.exception("space part of %s failed", sid)
                errors.append(f"space: {type(exc).__name__}: {exc}")
        elif job == "video":
            events = _fetch_types(client, sid, ASR_TYPES + IPS_TYPES, start, end, report, chunked=False)
            events.update(_fetch_types(client, sid, VFA_TYPES, start, end, report, chunked=True))
            from openmmla.analytics.report.video import build_video
            session_dir = None
            if repo_root:
                # the session folder the recordings route reads too (DASHBOARD_ARTIFACTS_DIR moves both)
                candidate = os.path.join(artifacts_root(repo_root), sid)
                session_dir = candidate if os.path.isdir(candidate) else None
            attention, timeline, rows = build_video(
                events, t0, t1, session_dir=session_dir,
                progress=lambda step, done, total: report(step, done, total))
            del events
            report("Saving", None, None, force=True)
            store.write_part(sid, "attention", jsonable(attention), last_event, REPORT_VERSION)
            store.write_part(sid, "timeline", jsonable(timeline), last_event, REPORT_VERSION)
            store.write_window_rows(sid, rows)
        else:
            raise ValueError(f"unknown job {job!r}")
    except NoMeasurements as exc:
        errors.append(str(exc))
    except Exception as exc:
        logger.exception("%s job of %s failed", job, sid)
        errors.append(f"{type(exc).__name__}: {exc}")
    # a job that failed before it read the data keeps the data state it was submitted at, so the
    # error backoff of _should_retry still applies to it
    seen = {} if last_event is None else {"source_last_event": last_event}
    if errors:
        return store.write_status(sid, job, state="error", step=None, error="; ".join(errors), **seen)
    return store.write_status(sid, job, state="done", step=None, done=None, total=None, error=None, **seen)


class JobRunner:
    """submits jobs to Celery or to a local process and answers report requests from the cache.

    A local job runs as its own process (`command(sid, job)`, the precompute command of
    dashboard.py), not as a thread of the web process: the video job is a minute of CPU that would
    hold the GIL against every page and stream, and under gevent a native thread must not touch
    sockets (or a client the report package keeps for the process) that belong to the web
    process's hub. The process reports through the same status file as a Celery job, and a few
    run at a time (LOCAL_SLOTS), the rest wait their turn here."""

    def __init__(self, store: ReportStore, command=None, *, mode: str | None = None, celery_app=None,
                 task=None, queue: str = "mmla-dashboard", cwd: str | None = None):
        self.store = store
        self.command = command
        self.mode = (mode or os.environ.get("DASHBOARD_JOBS") or "auto").strip().lower()
        if self.mode not in ("auto", "celery", "local"):
            logger.warning("DASHBOARD_JOBS=%s is not celery, local or auto: using auto", self.mode)
            self.mode = "auto"
        self.celery_app = celery_app
        self.task = task
        self.queue = queue
        self.cwd = cwd
        self._lock = threading.Lock()
        self._procs: dict = {}
        self._waiting: deque = deque()
        self._worker = None
        self._worker_checked = 0.0
        self._probe = None

    def worker_status(self, refresh: bool = False) -> dict:
        """how many Celery workers consume the dashboard queue, cached 30 s.

        A plain ping would also count an old worker of the previous dashboard, which listens on the
        default `celery` queue and would never take these tasks, so this asks for active queues.
        The probe runs aside and is waited for at most WORKER_WAIT seconds: kombu retries a refused
        Redis connection for several seconds, and a page request must not wait for that."""
        if self.mode == "local" or self.celery_app is None or self.task is None:
            return {"ok": None, "workers": 0, "queue": self.queue, "mode": self.mode}
        now = time.time()
        if not refresh and self._worker is not None and now - self._worker_checked < WORKER_TTL:
            return self._worker
        with self._lock:
            probe = self._probe
            if probe is None or not probe.is_alive():
                done = threading.Event()
                probe = threading.Thread(target=self._probe_workers, args=(done,), name="celery-probe",
                                         daemon=True)
                probe.done = done
                self._probe = probe
                probe.start()
        probe.done.wait(WORKER_WAIT)
        if not probe.done.is_set():
            pending = {"ok": False, "workers": 0, "queue": self.queue, "mode": self.mode,
                       "error": "Still waiting for Redis to answer."}
            if self._worker is None or now - self._worker_checked >= WORKER_TTL:
                self._worker, self._worker_checked = pending, now
        return self._worker

    def _probe_workers(self, done: threading.Event) -> None:
        result = {"ok": False, "workers": 0, "queue": self.queue, "mode": self.mode}
        try:
            replies = self.celery_app.control.inspect(timeout=1.0).active_queues() or {}
            workers = [name for name, queues in replies.items()
                       if any((q or {}).get("name") == self.queue for q in (queues or []))]
            result.update(ok=bool(workers), workers=len(workers))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        self._worker, self._worker_checked = result, time.time()
        done.set()

    def runner_for_submit(self) -> str:
        if self.mode == "local" or self.celery_app is None or self.task is None:
            return "local"
        if self.mode == "celery":
            return "celery"
        return "celery" if self.worker_status().get("ok") else "local"

    def busy(self, sid: str, job: str) -> bool:
        """whether this process runs or holds back the job."""
        with self._lock:
            return (sid, job) in self._procs or (sid, job) in self._waiting

    def submit(self, sid: str, job: str, force: bool = False, last_event: float | None = None) -> dict:
        """queue `job` (light, video or all) unless it already runs; returns the status of the
        (last) job. `force` is accepted for symmetry with the refresh route: a job that runs is
        never started twice. `last_event` is the data state the caller saw, kept in the status so
        a job that fails before it reads InfluxDB still counts as a failure on that data."""
        if job == "all":
            status = None
            for name in JOBS:
                status = self.submit(sid, name, force=force, last_event=last_event)
            return status
        if job not in JOBS:
            raise ValueError(f"unknown job {job!r}")
        self.store.session_dir(sid)
        runner = self.runner_for_submit()
        with self._lock:
            status = self.store.read_status(sid, job)
            if (sid, job) in self._procs or (sid, job) in self._waiting or is_active(status):
                return status or {"state": "running", "runner": "local"}
            now = round(time.time(), 3)
            status = self.store.write_status(sid, job, state="queued", step="Waiting to start", done=None,
                                             total=None, started_at=now, runner=runner, error=None,
                                             pid=os.getpid() if runner == "local" else None,
                                             source_last_event=last_event)
            if runner == "local":
                self._waiting.append((sid, job))
        if runner == "celery":
            try:
                self.task.apply_async(args=[sid, job], queue=self.queue)
                return status
            except Exception as exc:
                logger.warning("could not queue the %s job of %s on Celery (%s): running it here",
                               job, sid, exc)
                with self._lock:
                    self._waiting.append((sid, job))
                status = self.store.write_status(sid, job, runner="local", pid=os.getpid())
        self._start_waiting()
        return status

    def _start_waiting(self) -> None:
        launch = []
        with self._lock:
            for item in list(self._waiting):
                running = sum(1 for key in self._procs if key[1] == item[1])
                if running >= LOCAL_SLOTS.get(item[1], 1):
                    continue
                self._waiting.remove(item)
                self._procs[item] = None
                launch.append(item)
        for sid, job in launch:
            self._launch(sid, job)

    def _launch(self, sid: str, job: str) -> None:
        try:
            if self.command is None:
                raise RuntimeError("no command to run a local job with")
            proc = subprocess.Popen(self.command(sid, job), cwd=self.cwd, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, start_new_session=True)
        except Exception as exc:
            logger.exception("could not start the %s job of %s", job, sid)
            with self._lock:
                self._procs.pop((sid, job), None)
            self.store.write_status(sid, job, state="error", step=None,
                                    error=f"The report process did not start ({type(exc).__name__}: {exc}).")
            return
        with self._lock:
            self._procs[(sid, job)] = proc
        threading.Thread(target=self._reap, args=(sid, job, proc), name=f"report-{job}", daemon=True).start()

    def _reap(self, sid: str, job: str, proc) -> None:
        """wait for a local job's process; a process that died without saying so leaves an error."""
        try:
            code = proc.wait()
        except Exception:
            code = None
        with self._lock:
            self._procs.pop((sid, job), None)
        try:
            status = self.store.read_status(sid, job)
            if code != 0 and status and status.get("state") in ("queued", "running"):
                self.store.write_status(sid, job, state="error", step=None,
                                        error=f"The report process stopped (exit code {code}) before it finished.")
        except Exception as exc:
            logger.warning("could not check the %s job of %s: %s", job, sid, exc)
        self._start_waiting()

    def serve_part(self, sid: str, part: str, last_event: float | None, live: bool,
                   submit: bool = True) -> tuple[int, dict]:
        """the HTTP status and body of GET .../report/<part> (section 4.2 of the spec). With `submit`
        False it only reads: a page that merely borrows a part (the live page takes its colours from
        the speech and timeline parts) never starts a job, and a part never computed answers 404."""
        job = PART_JOB[part]
        now = time.time()
        envelope = self.store.read_part(sid, part)
        status = self.store.read_status(sid, job)
        active = is_active(status, now) or self.busy(sid, job)
        if envelope is not None:
            fresh = part_is_fresh(envelope, job, last_event, live, now)
            resubmitted = False
            if submit and not fresh and not active and self._should_retry(status, job, last_event, live, now):
                self.submit(sid, job, last_event=last_event)
                resubmitted = True
            body = {"status": "ready", "part": part, "computed_at": envelope.get("computed_at"),
                    "stale": not fresh, "data": envelope.get("data")}
            if not fresh and not active and not resubmitted and status and status.get("state") == "error":
                # the older part is all there is: say why the newer one is not, rather than
                # letting the page promise a recompute that already failed
                body["job_error"] = status.get("error") or "The job failed."
            return 200, body
        if active:
            return 202, {"status": (status or {}).get("state") or "running", "part": part,
                         "progress": _progress_body(status), "runner": (status or {}).get("runner")}
        if status and status.get("state") == "error" and not self._should_retry(status, job, last_event, live, now):
            return 500, {"status": "error", "part": part, "error": status.get("error") or "The job failed."}
        if not submit:
            return 404, {"status": "missing", "part": part}
        status = self.submit(sid, job, last_event=last_event)
        return 202, {"status": (status or {}).get("state") or "queued", "part": part,
                     "progress": _progress_body(status), "runner": (status or {}).get("runner")}

    @staticmethod
    def _should_retry(status: dict | None, job: str, last_event, live: bool, now: float) -> bool:
        """a failed job is retried when the data changed since, or after five minutes; a stale
        part whose last run succeeded is always recomputed. A failure that does not know its data
        state, or one of a live session (whose newest event moves every second), waits out the
        time alone: the live recompute pace, else five minutes."""
        if not status or status.get("state") != "error":
            return True
        source = status.get("source_last_event")
        if source is not None and not live and not _same_time(source, last_event):
            return True
        wait = LIVE_MAX_AGE.get(job, ERROR_RETRY_AFTER) if live else ERROR_RETRY_AFTER
        try:
            return now - float(status.get("updated_at") or 0) > wait
        except (TypeError, ValueError):
            return True


def run_in_worker(store: ReportStore, config_path: str, sid: str, job: str, repo_root: str | None) -> dict:
    """the Celery task body: its own InfluxDB and MongoDB clients, closed afterwards."""
    from openmmla.analytics.report.sessions import mongo_session, open_mongo
    from openmmla.utils.client import InfluxDBClientWrapper
    from openmmla.utils.config import load_config_with_system_services

    client = InfluxDBClientWrapper(config_path)
    mongo_db = None
    try:
        doc = None
        if job == "light":
            mongo_db = open_mongo(load_config_with_system_services(config_path), timeout_ms=JOB_MONGO_TIMEOUT_MS)
            doc = mongo_session(mongo_db, sid) if mongo_db is not None else None
        return compute_job(client, store, sid, job, runner="celery", mongo_doc=doc, repo_root=repo_root)
    finally:
        try:
            client.client.close()
        except Exception:
            pass
        if mongo_db is not None:
            try:
                mongo_db.client.close()
            except Exception:
                pass
