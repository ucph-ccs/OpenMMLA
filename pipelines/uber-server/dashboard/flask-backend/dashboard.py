"""The OpenMMLA dashboard: session explorer, live view and analysis report.

One Flask app serves the static pages under ../frontend, a JSON API over the sessions in InfluxDB
(with MongoDB's session documents when it answers), a Server-Sent Events stream for the live and
replay views, and the report parts, which openmmla.analytics.report computes and store.py caches.
The report jobs run on the Celery worker of `celery -A dashboard.celery worker` when one listens on
the dashboard's queue, else in a `python dashboard.py precompute` process this one starts.

Run it as the Makefile does, from this folder: `gunicorn -k gevent -w 1 -b 0.0.0.0:5050 dashboard:app`
(one worker: the job bookkeeping and the live feeds that every follower of a session shares live in
this process, see stream.py). For development, `python dashboard.py serve`; to fill the cache ahead
of a meeting, `python dashboard.py precompute --all`.
"""

import os
import sys

if __name__ == "__main__" and sys.argv[1:2] == ["serve"]:
    # gevent's server needs the standard library patched before anything opens a socket, as
    # gunicorn's gevent worker does for the deployed app
    from gevent import monkey
    monkey.patch_all()

import argparse
import gzip
import json
import logging
import mimetypes
import threading
import time
from urllib.parse import quote, urlencode, urlsplit

from celery import Celery
from flask import Flask, Response, request, send_file, send_from_directory

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import jobs as report_jobs  # noqa: E402
import recordings as raw_recordings  # noqa: E402
from media import MediaServer, rfc3339  # noqa: E402
from store import JOBS, PART_JOB, ReportStore  # noqa: E402
from stream import FEEDS, clamp_backfill, clamp_speed, live_stream  # noqa: E402

from openmmla.analytics.report import common, sessions  # noqa: E402
from openmmla.analytics.report.common import InfluxUnavailable, valid_session_id  # noqa: E402

logger = logging.getLogger("dashboard")

FRONTEND_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend"))
ASSETS_DIR = os.path.join(FRONTEND_DIR, "assets")
REPO_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, "..", "..", "..", ".."))
project_dir = BACKEND_DIR if os.path.isfile(os.path.join(BACKEND_DIR, "config.yml")) else os.getcwd()
config_path = os.path.join(project_dir, "config.yml")

QUEUE = os.environ.get("DASHBOARD_CELERY_QUEUE", "").strip() or "mmla-dashboard"
DEFAULT_PORT = 5050
INDEX_TTL = 15.0
LAST_EVENT_TTL = 5.0
# a failed InfluxDB query is answered from memory this long, so the requests that waited for it
# fail with it rather than each waiting out the client timeout again
INFLUX_ERROR_TTL = 5.0
STATE_TTL = 2.0
HEALTH_TTL = 10.0
GZIP_MIN_BYTES = 32 * 1024
EXPORT_NAMES = ("transcript.txt", "transcript.srt", "window_features.csv", "report.json")
# a stream recording shorter than this is not offered (the TUI's Export Streams skips it too)
SERVER_CLIP_MIN_SECONDS = 1.0

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")


def _usable(value) -> bool:
    text = str(value or "").strip()
    return bool(text) and "<" not in text and ">" not in text


def _load_config() -> tuple[dict, str | None]:
    """the dashboard config merged with System Settings (and decrypted); ({}, reason) when unreadable."""
    if not os.path.isfile(config_path):
        return {}, f"{config_path} does not exist: copy config_template.yml or save it from the TUI."
    try:
        from openmmla.utils.config import load_config_with_system_services
        return load_config_with_system_services(config_path) or {}, None
    except Exception as exc:
        return {}, f"config.yml could not be read: {type(exc).__name__}"


CONFIG, CONFIG_ERROR = _load_config()

RAW_MEDIA_OFF = "Raw recordings are turned off on this dashboard (Exports.raw_media in config.yml)."
RAW_MEDIA_OFF_ENV = "Raw recordings are turned off on this dashboard (DASHBOARD_RAW_MEDIA)."
RAW_MEDIA_OFF_UNREAD = ("Raw recordings are turned off on this dashboard because its configuration could not "
                        "be read (config.yml or System Settings).")


def _switch(value) -> bool | None:
    """a yes/no setting as a bool, None when it says neither."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return None


def raw_media_state() -> tuple[bool, str | None]:
    """(whether the raw recordings are listed and served, why not). DASHBOARD_RAW_MEDIA overrides
    Exports.raw_media of the merged config; on when neither says anything, off when either holds
    something that is neither yes nor no, an empty value included (they guard footage of children, so
    a typo keeps them off), and off when config.yml exists but the merged config could not be read
    (it may say false)."""
    env = (os.environ.get("DASHBOARD_RAW_MEDIA") or "").strip()
    if env:
        on = _switch(env)
        return (True, None) if on else (False, RAW_MEDIA_OFF_ENV)
    if CONFIG_ERROR and os.path.isfile(config_path):
        return False, RAW_MEDIA_OFF_UNREAD
    if "Exports" not in CONFIG or CONFIG.get("Exports") is None:
        return True, None
    section = CONFIG.get("Exports")
    if not isinstance(section, dict):
        return False, RAW_MEDIA_OFF
    if "raw_media" not in section:
        return True, None
    if _switch(section.get("raw_media")) is True:
        return True, None
    return False, RAW_MEDIA_OFF


def _redis_url(config: dict) -> str | None:
    section = config.get("Redis") if isinstance(config.get("Redis"), dict) else {}
    host = section.get("host")
    if not _usable(host):
        return None
    try:
        port = int(section.get("port") or 6379)
        db = int(section.get("db") or 0)
    except (TypeError, ValueError):
        return None
    return f"redis://{str(host).strip()}:{port}/{db}"


REDIS_URL = _redis_url(CONFIG)


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name) or default))
    except ValueError:
        return default


def make_celery(flask_app: Flask) -> Celery:
    """the report worker's Celery app, on Redis from the merged config and on its own queue, so a
    worker of an older dashboard (which consumes the default `celery` queue) never takes its tasks."""
    broker = REDIS_URL or "memory://"
    app_celery = Celery(flask_app.import_name if flask_app.import_name != "__main__" else "dashboard",
                        broker=broker, backend=REDIS_URL or None)
    app_celery.conf.update(
        task_default_queue=QUEUE,
        task_ignore_result=True,
        task_track_started=False,
        worker_prefetch_multiplier=1,
        # a video job holds a whole session's video features: two at a time is what one server takes
        worker_concurrency=_int_env("DASHBOARD_WORKER_CONCURRENCY", 2),
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=3,
        broker_transport_options={"socket_connect_timeout": 3},
    )
    return app_celery


app = Flask(__name__, static_folder=ASSETS_DIR, static_url_path="/assets")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = None
app.json.sort_keys = False
celery = make_celery(app)
STORE = ReportStore()
MEDIA = MediaServer(REPO_ROOT, CONFIG)


@celery.task(name="dashboard.build_report_task", ignore_result=True)
def build_report_task(sid, job):
    """compute one report job of one session into the cache (light: speech + space, video:
    attention + timeline)."""
    if not valid_session_id(sid) or job not in JOBS:
        logger.warning("ignored a report task with an invalid session id or job")
        return None
    status = report_jobs.run_in_worker(STORE, config_path, sid, job, REPO_ROOT)
    return (status or {}).get("state")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


_client = None
_client_lock = threading.Lock()


def influx_client():
    """the web process's InfluxDB client wrapper, made on first use."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            if CONFIG_ERROR:
                raise InfluxUnavailable(CONFIG_ERROR)
            try:
                from openmmla.utils.client import InfluxDBClientWrapper
                _client = InfluxDBClientWrapper(config_path)
            except Exception as exc:
                raise InfluxUnavailable(f"InfluxDB is not configured in config.yml ({type(exc).__name__})") from exc
    return _client


def influx_label() -> str:
    section = CONFIG.get("InfluxDB") if isinstance(CONFIG.get("InfluxDB"), dict) else {}
    url = section.get("url") if _usable(section.get("url")) else "n/a"
    bucket = section.get("bucket") or "mmla-data"
    return f"InfluxDB at {url} (bucket {bucket})"


_mongo = None
_mongo_checked = 0.0
_mongo_lock = threading.Lock()
MONGO_RETRY = 30.0


def mongo_configured() -> bool:
    section = CONFIG.get("MongoDB") if isinstance(CONFIG.get("MongoDB"), dict) else {}
    return _usable(section.get("url"))


def mongo_db():
    """the MongoDB database of the session documents, None when not configured or unreachable
    (retried every 30 s)."""
    global _mongo, _mongo_checked
    if _mongo is not None or not mongo_configured():
        return _mongo
    now = time.time()
    if now - _mongo_checked < MONGO_RETRY:
        return None
    with _mongo_lock:
        if _mongo is None and now - _mongo_checked >= MONGO_RETRY:
            _mongo_checked = now
            try:
                _mongo = sessions.open_mongo(CONFIG)
            except Exception as exc:
                logger.info("MongoDB not available: %s", type(exc).__name__)
                _mongo = None
    return _mongo


_last_events: dict = {}
_last_lock = threading.Lock()
_last_queries: dict = {}
_last_errors: dict = {}


def _cached_last(sid: str, max_age: float, now: float):
    with _last_lock:
        hit = _last_events.get(sid)
    return hit if hit and now - hit[0] < max_age else None


def _keep_last(sid: str, at: float, value) -> None:
    with _last_lock:
        if len(_last_events) > 2048:
            _last_events.clear()
        _last_events[sid] = (at, value)


def last_event(sid: str, refresh: bool = False, max_age: float = LAST_EVENT_TTL) -> float | None:
    """the newest event time of a session, at most `max_age` seconds old (5 s by default; every
    live view asks it). A session whose live feed polled within that time answers from the feed
    without a query; otherwise one request asks InfluxDB while the others for the session wait,
    and when it fails they fail with it (INFLUX_ERROR_TTL)."""
    now = time.time()
    if not refresh:
        hit = _cached_last(sid, max_age, now)
        if hit is not None:
            return hit[1]
        fed = FEEDS.newest(sid, max_age, now)
        if fed is not None:
            _keep_last(sid, now, fed)
            return fed
    with _last_lock:
        lock = _last_queries.get(sid)
        if lock is None:
            if len(_last_queries) > 2048:
                _last_queries.clear()
            lock = _last_queries[sid] = threading.Lock()
    with lock:
        now = time.time()
        if not refresh:
            hit = _cached_last(sid, max_age, now)
            if hit is not None:
                return hit[1]
        with _last_lock:
            failed = _last_errors.get(sid)
        if failed is not None and now - failed[0] < INFLUX_ERROR_TTL:
            raise InfluxUnavailable(failed[1])
        try:
            value = common.last_event_time(influx_client(), sid)
        except InfluxUnavailable as exc:
            with _last_lock:
                if len(_last_errors) > 2048:
                    _last_errors.clear()
                _last_errors[sid] = (time.time(), str(exc))
            raise
        _keep_last(sid, now, value)
        with _last_lock:
            _last_errors.pop(sid, None)
    return value


def is_live(last: float | None, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return last is not None and now - last < sessions.LIVE_SECONDS


_index = None
_index_at = 0.0
_index_error = None
_index_lock = threading.Lock()


def session_index(refresh: bool = False) -> dict:
    """sessions.session_index, cached 15 s; one request computes it while the others wait, and
    when it fails they fail with it (INFLUX_ERROR_TTL)."""
    global _index, _index_at, _index_error
    if not refresh and _index is not None and time.time() - _index_at < INDEX_TTL:
        return _index
    with _index_lock:
        if refresh or _index is None or time.time() - _index_at >= INDEX_TTL:
            failed = _index_error
            if failed is not None and time.time() - failed[0] < INFLUX_ERROR_TTL:
                raise InfluxUnavailable(failed[1])
            try:
                value = sessions.session_index(influx_client(), mongo_db())
            except InfluxUnavailable as exc:
                _index_error = (time.time(), str(exc))
                raise
            _index, _index_at, _index_error = value, time.time(), None
            now = time.time()
            with _last_lock:
                for entry in value.get("sessions") or []:
                    if entry.get("id") and entry.get("last_event") is not None:
                        _last_events.setdefault(entry["id"], (now, entry["last_event"]))
    return _index


def index_entry(sid: str) -> dict | None:
    """the session's entry in the cached index, without querying when the cache is cold."""
    index = _index
    if index is None:
        return None
    for entry in index.get("sessions") or []:
        if entry.get("id") == sid:
            return entry
    return None


def report_summary(sid: str, last: float | None, live: bool) -> dict:
    now = time.time()
    return {job: report_jobs.job_summary(STORE, sid, job, last, live, now) for job in JOBS}


def job_command(sid: str, job: str) -> list[str]:
    """the command of a local report job: this file's precompute, in this interpreter."""
    return [sys.executable, os.path.join(BACKEND_DIR, "dashboard.py"), "precompute", "--force", "--job", job, sid]


RUNNER = report_jobs.JobRunner(STORE, job_command, celery_app=celery if REDIS_URL else None,
                               task=build_report_task, queue=QUEUE, cwd=BACKEND_DIR)


def _dumps(body) -> str:
    try:
        return json.dumps(body, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps(common.jsonable(body), allow_nan=False, separators=(",", ":"))


def json_response(body, status: int = 200) -> Response:
    text = _dumps(body).encode("utf-8")
    response = Response(text, status=status, mimetype="application/json")
    if len(text) >= GZIP_MIN_BYTES and "gzip" in (request.headers.get("Accept-Encoding") or ""):
        response.set_data(gzip.compress(text, compresslevel=5))
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"
    response.headers["Cache-Control"] = "no-store"
    return response


def check_sid(sid: str) -> str:
    if not valid_session_id(sid):
        raise ApiError(400, "That is not a valid session id.")
    return sid


def influx_error(exc: Exception) -> ApiError:
    return ApiError(503, f"{influx_label()}: {exc}")


@app.errorhandler(ApiError)
def _api_error(exc: ApiError):
    return json_response({"error": exc.message}, exc.status)


@app.errorhandler(404)
def _not_found(exc):
    if request.path.startswith("/api/"):
        return json_response({"error": "Not found."}, 404)
    return Response("Not found.", status=404, mimetype="text/plain")


@app.errorhandler(405)
def _not_allowed(exc):
    if request.path.startswith("/api/"):
        return json_response({"error": "Method not allowed."}, 405)
    return Response("Method not allowed.", status=405, mimetype="text/plain")


@app.errorhandler(500)
def _server_error(exc):
    if request.path.startswith("/api/"):
        return json_response({"error": "The dashboard failed on this request; its log says why."}, 500)
    return Response("Server error.", status=500, mimetype="text/plain")


@app.after_request
def _cache_headers(response: Response) -> Response:
    path = request.path
    if path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    elif path.startswith("/assets/") or path.endswith(".html") or path in ("/", "/live", "/realtime",
                                                                            "/analysis", "/posttime"):
        response.headers["Cache-Control"] = "no-cache"
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response


def _page(name: str) -> Response:
    if not os.path.isfile(os.path.join(FRONTEND_DIR, name)):
        return Response(f"{name} is missing from the dashboard frontend.", status=404, mimetype="text/plain")
    return send_from_directory(FRONTEND_DIR, name)


@app.route("/")
def index_page():
    return _page("index.html")


@app.route("/live")
@app.route("/realtime")
def live_page():
    return _page("live.html")


@app.route("/analysis")
@app.route("/posttime")
def analysis_page():
    return _page("analysis.html")


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


_health = None
_health_at = 0.0
_health_lock = threading.Lock()


@app.route("/api/health")
def api_health():
    """reachability of everything the dashboard reads, for the explorer's status chips; never 5xx.
    The checks run at most every 10 s, however many pages ask; `checked_at` says when they ran,
    `stream` how many sessions are followed live now and by how many connections."""
    global _health, _health_at
    if _health is None or time.time() - _health_at >= HEALTH_TTL:
        with _health_lock:
            if _health is None or time.time() - _health_at >= HEALTH_TTL:
                _health = _check_health()
                _health_at = time.time()
    out = dict(_health)
    out["checked_at"] = round(_health_at, 3)
    out["time"] = round(time.time(), 3)
    out["stream"] = FEEDS.stats()
    return json_response(out)


def _check_health() -> dict:
    out = {"influx": {"ok": False, "url": None, "bucket": None}, "mongo": {"ok": None},
           "worker": {"ok": None, "workers": 0, "queue": QUEUE}, "media": {"ok": None, "host": None}}
    section = CONFIG.get("InfluxDB") if isinstance(CONFIG.get("InfluxDB"), dict) else {}
    out["influx"]["url"] = section.get("url") if _usable(section.get("url")) else None
    out["influx"]["bucket"] = section.get("bucket") or "mmla-data"
    try:
        client = influx_client()
        common.query_tables(client, f'from(bucket: "{client.bucket}") |> range(start: -1m) |> limit(n: 1)')
        out["influx"]["ok"] = True
    except Exception as exc:
        out["influx"]["error"] = str(exc) if isinstance(exc, InfluxUnavailable) else type(exc).__name__
    if mongo_configured():
        try:
            db = mongo_db()
            if db is None:
                out["mongo"] = {"ok": False, "error": "MongoDB did not answer."}
            else:
                db.command("ping")
                out["mongo"] = {"ok": True}
        except Exception as exc:
            out["mongo"] = {"ok": False, "error": f"MongoDB did not answer ({type(exc).__name__})."}
    try:
        worker = dict(RUNNER.worker_status())
        out["worker"] = {"ok": worker.get("ok"), "workers": worker.get("workers", 0), "queue": QUEUE,
                         "mode": worker.get("mode")}
        if worker.get("error"):
            out["worker"]["error"] = str(worker["error"])[:300]
    except Exception as exc:
        out["worker"]["error"] = type(exc).__name__
    try:
        out["media"] = MEDIA.health()
    except Exception as exc:
        out["media"] = {"ok": False, "host": None, "error": type(exc).__name__}
    return out


@app.route("/api/sessions")
def api_sessions():
    try:
        index = session_index()
    except InfluxUnavailable as exc:
        raise influx_error(exc)
    except Exception as exc:
        logger.exception("session index failed")
        raise influx_error(f"{type(exc).__name__}: {exc}")
    now = time.time()
    entries = []
    for entry in index.get("sessions") or []:
        entry = dict(entry)
        last = entry.get("last_event")
        try:
            entry["report"] = report_summary(entry["id"], last, is_live(last, now))
        except Exception as exc:
            logger.warning("report state of %s: %s", entry.get("id"), exc)
            entry["report"] = {job: "missing" for job in JOBS}
        entries.append(entry)
    body = dict(index)
    body["sessions"] = entries
    return json_response(body)


@app.route("/api/get_sessions")
def api_get_sessions():
    """the old plain list of session ids, kept for anything scripted against it."""
    try:
        index = session_index()
    except Exception as exc:
        raise influx_error(exc)
    ids = sorted(entry["id"] for entry in index.get("sessions") or []
                 if entry.get("id") and any((entry.get("counts") or {}).values()))
    return json_response(ids)


def _meta(sid: str) -> dict:
    """the meta object of a session, 404 when neither InfluxDB nor MongoDB knows it."""
    try:
        meta = sessions.session_meta(influx_client(), sid, mongo_db())
    except InfluxUnavailable as exc:
        raise influx_error(exc)
    if not meta:
        raise ApiError(404, "No session with this id in InfluxDB or MongoDB.")
    return meta


@app.route("/api/sessions/<sid>")
def api_session(sid):
    check_sid(sid)
    meta = dict(_meta(sid))
    state = meta.get("state") or {}
    last = state.get("last_event")
    if last is not None:
        with _last_lock:
            _last_events[sid] = (time.time(), last)
    meta["report"] = report_summary(sid, last, bool(state.get("live")))
    return json_response(meta)


@app.route("/api/sessions/<sid>/state")
def api_session_state(sid):
    """whether a session is live and how far behind its newest data is, for pages that poll every
    few seconds (the explorer's live band): one small InfluxDB query per session at most every 2 s
    (none while the session's live feed runs), the span from the cached session list."""
    check_sid(sid)
    try:
        last = last_event(sid, max_age=STATE_TTL)
    except InfluxUnavailable as exc:
        raise influx_error(exc)
    entry = index_entry(sid)
    if last is None and entry is None:
        raise ApiError(404, "No session with this id in InfluxDB.")
    now = time.time()
    t0 = entry.get("t0") if entry else None
    ends = [float(x) for x in ((entry or {}).get("t1"), last) if x is not None]
    return json_response({"live": is_live(last, now), "last_event": None if last is None else round(last, 3),
                          "lag": None if last is None else round(now - last, 3),
                          "t0": None if t0 is None else round(float(t0), 3),
                          "t1": round(max(ends), 3) if ends else None})


def _session_last(sid: str) -> float:
    """the session's newest event time, 404 when it has no measurements at all."""
    try:
        last = last_event(sid)
    except InfluxUnavailable as exc:
        raise influx_error(exc)
    if last is None:
        raise ApiError(404, "This session has no measurements in InfluxDB.")
    return last


@app.route("/api/sessions/<sid>/report/<part>")
def api_report_part(sid, part):
    check_sid(sid)
    if part not in PART_JOB:
        raise ApiError(404, "Report parts are speech, space, attention and timeline.")
    last = _session_last(sid)
    # ?submit=0 reads the cache only and never starts a job
    submit = request.args.get("submit", "1") != "0"
    status, body = RUNNER.serve_part(sid, part, last, is_live(last), submit=submit)
    return json_response(body, status)


@app.route("/api/sessions/<sid>/report/refresh", methods=["POST"])
def api_report_refresh(sid):
    check_sid(sid)
    payload = request.get_json(silent=True) or {}
    job = payload.get("job") if isinstance(payload, dict) else None
    job = job or "all"
    if job not in ("light", "video", "all"):
        raise ApiError(400, "job must be light, video or all.")
    last = _session_last(sid)
    states = {}
    for name in (JOBS if job == "all" else (job,)):
        status = RUNNER.submit(sid, name, force=True, last_event=last) or {}
        states[name] = {"state": status.get("state"), "runner": status.get("runner")}
    return json_response({"status": "queued", "jobs": states}, 202)


def _span(sid: str) -> dict:
    """t0, t1 and the group of a session, from the cached index when it holds the session."""
    entry = index_entry(sid)
    if entry is None:
        meta = _meta(sid)
        entry = {"t0": meta.get("t0"), "t1": meta.get("t1"), "group": meta.get("group"),
                 "last_event": (meta.get("state") or {}).get("last_event")}
    return entry


def _float_arg(name: str):
    value = request.args.get(name)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except ValueError:
        raise ApiError(400, f"{name} must be a number.")
    if number != number or number in (float("inf"), float("-inf")):
        raise ApiError(400, f"{name} must be a number.")
    return number


@app.route("/api/sessions/<sid>/stream")
def api_stream(sid):
    check_sid(sid)
    mode = (request.args.get("mode") or "follow").strip().lower()
    if mode not in ("follow", "replay"):
        raise ApiError(400, "mode must be follow or replay.")
    at = _float_arg("at")
    speed = clamp_speed(request.args.get("speed") or 1)
    backfill = clamp_backfill(request.args.get("backfill") if request.args.get("backfill") not in (None, "")
                              else 300)
    try:
        client = influx_client()
        span = _span(sid)
    except InfluxUnavailable as exc:
        raise influx_error(exc)
    floor = None
    try:
        space = STORE.read_part(sid, "space")
        floor = ((space or {}).get("data") or {}).get("floor")
    except Exception:
        floor = None

    def cached_last():
        try:
            return last_event(sid)
        except InfluxUnavailable:
            return None

    generator = live_stream(client, sid, mode=mode, t0=span.get("t0"), t1=span.get("t1"),
                            group_id=span.get("group"), at=at, speed=speed, backfill=backfill,
                            floor=floor, last_event_fn=cached_last)
    response = Response(generator, mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


def _request_hostname() -> str | None:
    try:
        return urlsplit("//" + (request.host or "")).hostname
    except ValueError:
        return None


def _media_inputs(sid: str) -> tuple:
    """(MongoDB answered, the session's document, its devices, t0, t1, the ApiError of a span
    lookup that failed) for the stream server's view of a session: the span from the session list
    or InfluxDB, else from the document."""
    db = mongo_db()
    doc = None
    if db is not None:
        try:
            doc = sessions.mongo_session(db, sid)
        except Exception as exc:
            logger.info("MongoDB read of %s failed: %s", sid, type(exc).__name__)
    devices = sessions.mongo_devices(doc) if doc else None
    t0 = t1 = None
    failed = None
    try:
        span = _span(sid)
        t0, t1 = span.get("t0"), span.get("last_event") or span.get("t1")
    except InfluxUnavailable:
        pass
    except ApiError as exc:
        failed = exc
    if t0 is None and doc:
        t0 = common.to_epoch(doc.get("start_time"))
        t1 = common.to_epoch(doc.get("end_time")) or time.time()
    return db is not None, doc, devices, t0, t1, failed


@app.route("/api/sessions/<sid>/media")
def api_media(sid):
    check_sid(sid)
    # while Exports.raw_media is off, neither the recorded stretches nor the playback server that
    # serves them go out (the live view only needs webrtc)
    raw_media_on, _ = raw_media_state()
    want_recordings = raw_media_on and request.args.get("recordings") in ("1", "true", "yes")
    mongo_ok, doc, devices, t0, t1, failed = _media_inputs(sid)
    if failed is not None and (failed.status not in (404, 503) or (failed.status == 404 and doc is None)):
        raise failed
    body = MEDIA.session_media(doc, devices, mongo_ok, t0, t1, want_recordings, _request_hostname())
    if not raw_media_on:
        body["playback"] = None
    return json_response(body)


def _server_recordings(mongo_ok: bool, doc, devices, t0, t1) -> list[dict]:
    """MediaMTX's recordings of the session's stream paths within its span: one entry per unbroken
    stretch of a path (a stream restarted during the session gives two), each with the playback
    server's URL of that stretch (fMP4, as Sessions -> Export Streams fetches it); [] without the
    session's streams, a stream server that answers or its playback server."""
    if doc is None or t0 is None or t1 is None:
        return []
    ready, _ = MEDIA.ready_paths()
    if ready is None:
        # the API did not answer, and the playback server beside it would make every path wait
        # out its timeout
        return []
    body = MEDIA.session_media(doc, devices, mongo_ok, t0, t1, True, _request_hostname())
    playback = body.get("playback")
    if not playback:
        return []
    kinds = {stream.get("path"): stream.get("kind") for stream in body.get("streams") or []}
    out = []
    for item in body.get("recordings") or []:
        for start, duration in item.get("spans") or []:
            if duration < SERVER_CLIP_MIN_SECONDS:
                continue
            query = urlencode({"path": item["path"], "start": rfc3339(start), "duration": f"{duration:.3f}"})
            out.append({"path": item["path"], "kind": kinds.get(item["path"]) or "video",
                        "spans": [[start, duration]], "url": f"{playback}/get?{query}"})
    return out


@app.route("/api/sessions/<sid>/recordings")
def api_recordings(sid):
    """the session's raw camera and microphone files on this machine (artifacts/<sid>/collection/)
    and MediaMTX's recordings of its streams, for the downloads; nothing while Exports.raw_media
    (or DASHBOARD_RAW_MEDIA) turns them off. The files are listed without InfluxDB or MongoDB;
    those only add each file's offset from the session start and the stream recordings."""
    check_sid(sid)
    enabled, reason = raw_media_state()
    if not enabled:
        return json_response({"enabled": False, "files": [], "server": [], "reason": reason})
    files = raw_recordings.list_recordings(raw_recordings.artifacts_root(), sid)
    try:
        mongo_ok, doc, devices, t0, t1, _ = _media_inputs(sid)
    except Exception as exc:
        logger.info("span of %s for its recordings: %s", sid, type(exc).__name__)
        mongo_ok, doc, devices, t0, t1 = False, None, None, None, None
    for record in files:
        record["url"] = f"/api/sessions/{sid}/recordings/{quote(record['id'], safe='')}"
        start = record.get("start")
        record["offset"] = round(start - float(t0), 3) if start is not None and t0 is not None else None
    try:
        server = _server_recordings(mongo_ok, doc, devices, t0, t1)
    except Exception as exc:
        logger.info("stream recordings of %s: %s", sid, type(exc).__name__)
        server = []
    return json_response({"enabled": True, "files": files, "server": server, "reason": None})


@app.route("/api/sessions/<sid>/recordings/<rec_id>")
def api_recording_file(sid, rec_id):
    """one file the recordings route lists, with HTTP ranges (a player seeks in it); a download
    unless ?inline=1."""
    check_sid(sid)
    if not raw_recordings.REC_ID_RE.fullmatch(rec_id or ""):
        raise ApiError(400, "That is not a valid recording id.")
    enabled, reason = raw_media_state()
    if not enabled:
        raise ApiError(403, reason)
    path = raw_recordings.resolve(raw_recordings.artifacts_root(), sid, rec_id)
    if path is None:
        raise ApiError(404, "This session has no recording with this id on the dashboard's machine.")
    inline = request.args.get("inline") in ("1", "true", "yes")
    response = send_file(path, mimetype=raw_recordings.media_type(path), as_attachment=not inline,
                         download_name=raw_recordings.download_name(sid, rec_id, path), conditional=True,
                         max_age=None)
    response.headers["Cache-Control"] = "private, no-store"
    return response


def _attachment(response: Response, sid: str, name: str) -> Response:
    response.headers["Content-Disposition"] = f'attachment; filename="{sid}_{name}"'
    response.headers["Cache-Control"] = "no-store"
    return response


def _speech_for_export(sid: str) -> dict:
    """the speech part's data for the transcript downloads; a part not computed yet goes through
    the report route's rules (its job queued unless it runs or failed a moment ago)."""
    envelope = STORE.read_part(sid, "speech")
    if envelope is not None:
        return envelope.get("data") or {}
    last = _session_last(sid)
    code, body = RUNNER.serve_part(sid, "speech", last, is_live(last))
    if code == 200:
        return body.get("data") or {}
    if code == 500:
        raise ApiError(500, f"The speech report failed: {body.get('error')}")
    raise ApiError(404, "The speech report has not been computed yet; it is being computed now.")


@app.route("/api/sessions/<sid>/export/<name>")
def api_export(sid, name):
    check_sid(sid)
    if name.endswith(".jsonl"):
        event_type = name[:-len(".jsonl")]
        if event_type not in common.EVENT_TYPES:
            raise ApiError(404, "Unknown event type.")
        _session_last(sid)
        from openmmla.analytics.report import exports
        try:
            lines = exports.iter_jsonl(influx_client(), sid, event_type)
        except InfluxUnavailable as exc:
            raise influx_error(exc)

        def body():
            try:
                for line in lines:
                    yield line if line.endswith("\n") else line + "\n"
            except InfluxUnavailable as exc:
                logger.warning("export of %s %s stopped: %s", sid, event_type, exc)
                # the server then drops the connection without the closing chunk, so the client
                # reports an incomplete download rather than a whole file cut short
                raise

        return _attachment(Response(body(), mimetype="application/x-ndjson"), sid, name)
    if name not in EXPORT_NAMES:
        raise ApiError(404, "Unknown export.")
    if name == "window_features.csv":
        path = STORE.window_features_path(sid)
        if not os.path.isfile(path):
            raise ApiError(404, "The window features exist once the video analysis has run.")
        return _attachment(send_file(path, mimetype="text/csv", conditional=False), sid, name)
    if name in ("transcript.txt", "transcript.srt"):
        from openmmla.analytics.report import exports
        speech = _speech_for_export(sid)
        transcript = speech.get("transcript") or []
        if name == "transcript.txt":
            text = exports.transcript_text(transcript, speech.get("t0") or 0.0)
        else:
            text = exports.transcript_srt(transcript)
        kind = "text/plain" if name.endswith(".txt") else "application/x-subrip"
        return _attachment(Response(text, content_type=f"{kind}; charset=utf-8"), sid, name)
    meta = _meta(sid)
    parts = {}
    for part in PART_JOB:
        envelope = STORE.read_part(sid, part)
        if envelope is not None:
            parts[part] = {"computed_at": envelope.get("computed_at"),
                           "source_last_event": envelope.get("source_last_event"), "data": envelope.get("data")}
    text = _dumps({"session": sid, "exported_at": round(time.time(), 3), "meta": meta, "parts": parts})
    return _attachment(Response(text, mimetype="application/json"), sid, name)


def _precompute(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="dashboard.py precompute",
                                     description="compute report parts into the cache")
    parser.add_argument("sessions", nargs="*", help="session ids")
    parser.add_argument("--all", action="store_true", help="every session with InfluxDB data")
    parser.add_argument("--job", choices=("light", "video", "all"), default="all")
    parser.add_argument("--force", action="store_true", help="run even when the job looks active")
    args = parser.parse_args(argv)
    client = influx_client()
    # one connection for the run, with a job's patience: a part keeps the cameras it was computed with
    db = sessions.open_mongo(CONFIG, timeout_ms=report_jobs.JOB_MONGO_TIMEOUT_MS) if mongo_configured() else None
    if mongo_configured() and db is None:
        print("MongoDB did not answer: the space parts are computed without camera placement", file=sys.stderr)
    ids = list(args.sessions)
    if args.all:
        index = sessions.session_index(client, db)
        ids += [e["id"] for e in index.get("sessions") or [] if any((e.get("counts") or {}).values())]
    if not ids:
        parser.error("name session ids or pass --all")
    failures = 0
    for sid in dict.fromkeys(ids):
        if not valid_session_id(sid):
            print(f"{sid}: not a valid session id", file=sys.stderr)
            failures += 1
            continue
        doc = sessions.mongo_session(db, sid) if db is not None else None
        for job in (JOBS if args.job == "all" else (args.job,)):
            if not args.force and report_jobs.is_active(STORE.read_status(sid, job)):
                print(f"{sid} {job}: already running elsewhere, skipped", file=sys.stderr)
                continue
            started = time.time()
            status = report_jobs.compute_job(client, STORE, sid, job, runner="local", mongo_doc=doc,
                                             repo_root=REPO_ROOT)
            took = time.time() - started
            if status.get("state") == "done":
                print(f"{sid} {job}: done in {took:.1f} s", file=sys.stderr)
            else:
                failures += 1
                print(f"{sid} {job}: {status.get('error')}", file=sys.stderr)
    if db is not None:
        db.client.close()
    return 1 if failures else 0


def _serve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="dashboard.py serve", description="development server (gevent)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT") or DEFAULT_PORT))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args(argv)
    from gevent.pywsgi import WSGIServer
    print(f"dashboard on http://{args.host}:{args.port} (cache {STORE.cache_dir}, jobs {RUNNER.mode})",
          file=sys.stderr)
    server = WSGIServer((args.host, args.port), app)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python dashboard.py serve [--port N] | precompute <sid>... | --all [--job light|video|all]")
        return 0
    command, rest = argv[0], argv[1:]
    if command == "serve":
        return _serve(rest)
    if command == "precompute":
        return _precompute(rest)
    print(f"unknown command {command!r}: serve or precompute", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
