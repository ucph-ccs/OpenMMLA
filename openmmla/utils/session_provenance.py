"""what a session ran with: one entry per component (a base, a synchronizer, the
IPS visualizer), kept in the session's MongoDB document under `components` and,
next to the config it copied, as
artifacts/<session>/pipelines/<pipeline>/<host>/config/<role>[_<id>].json.

A component writes its entry as soon as it knows its session and what it runs
with: the flags it was started with (`arguments`), the values it resolved from
them and its config (`parameters`: thresholds, durations, the camera and its
intrinsics, the transformation matrices, the speaker profiles it recognizes,
the stream it takes), the files it read that are not in the config (`files`),
the config itself with its secrets masked (`config`, and the file's digest) and
the software it runs (`software`: the openmmla version, the checkout's commit).
What the servers it calls run (the transcriber's model and language, the frame
analyzer's models and prompt profile) is asked after, in a thread, and set into
the entry as `services` when it comes: a server that has no /info yet, or does
not answer, is noted as such.

The measurements in InfluxDB carry the session id and nothing else about the
run, so this is where a session's numbers are compared to another's, or a run
is set up again. Sessions -> Export Measurements writes it out next to them as
<session>_parameters.json. Writing never stops a component: MongoDB being down,
or a session document the console did not create, is a warning in its log and
nothing more."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from openmmla.utils.artifact_paths import pipeline_section_dir, safe_segment, short_hostname

logger = logging.getLogger(__name__)

# the field of the session document
COMPONENTS_FIELD = "components"
# what an entry says instead of a secret
MASK = "***"
# the path a server answers about itself on, below its endpoint
INFO_PATH = "info"

_SECRET_KEY = re.compile(r"(token|api[_-]?key|secret|passw(or)?d|subscription[_-]?key|access[_-]?key|private[_-]?key)", re.I)


def component_key(pipeline: str, role: str, component_id=None) -> str:
    """one entry per component of a pipeline: `asr:base:1`, `ips:synchronizer`,
    `asr:synchronizer:Jabra` (the base type it merges), `ips:visualizer`."""
    key = f"{pipeline}:{role}"
    if component_id is None or str(component_id).strip() == "":
        return key
    return f"{key}:{component_id}"


def plain(value):
    """a copy that JSON and MongoDB take: numpy scalars and arrays as numbers and
    lists, tuples and sets as lists, paths as text, datetimes as they are, keys as
    text without the characters MongoDB refuses in a field name; anything else
    as its text."""
    if value is None or isinstance(value, (bool, int, float, str, datetime)):
        return value
    if isinstance(value, dict):
        return {_field_name(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, Path):
        return str(value)
    tolist = getattr(value, "tolist", None)  # a numpy array or scalar
    if callable(tolist):
        try:
            return plain(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)  # a numpy scalar without tolist
    if callable(item):
        try:
            return plain(item())
        except Exception:
            pass
    return str(value)


def _field_name(key) -> str:
    text = str(key)
    if text.startswith("$"):
        text = "_" + text[1:]
    return text.replace(".", "_")


def redact_secrets(value):
    """the same structure with every secret masked: the value of a key that names
    one (token, api_key, password, subscription_key ...), an ENC(...) value
    wherever it is, and the password of a URL."""
    if isinstance(value, dict):
        return {key: (MASK if _SECRET_KEY.search(str(key)) and item not in (None, "") else redact_secrets(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        if value.startswith("ENC(") and value.endswith(")"):
            return MASK
        return _mask_url_password(value)
    return value


def _mask_url_password(text: str) -> str:
    if "://" not in text or "@" not in text:
        return text
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if not parts.password:
        return text
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    netloc = f"{parts.username or ''}:{MASK}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def file_digest(path) -> str | None:
    """the sha256 of a file, to tell at a glance whether two runs read the same config."""
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()
    except OSError:
        return None


def git_commit(project_dir=None) -> str | None:
    """the commit of the checkout a project folder is in, or None (not a checkout, no git)."""
    try:
        done = subprocess.run(
            ["git", "-C", str(project_dir or "."), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def software_info(project_dir=None) -> dict:
    """the software a component runs: openmmla's version, the checkout's commit, python, the OS."""
    try:
        version = importlib.metadata.version("openmmla")
    except Exception:
        version = None
    info = {"openmmla": version, "python": platform.python_version(), "platform": platform.platform()}
    commit = git_commit(project_dir)
    if commit:
        info["git_commit"] = commit
    return info


def component_entry(pipeline: str, role: str, component_id=None, *, arguments: dict | None = None,
                    parameters: dict | None = None, files: dict | None = None, services: dict | None = None,
                    config: dict | None = None, config_path=None, project_dir=None, host: str | None = None,
                    now: datetime | None = None) -> dict:
    """the entry a component writes for itself when it joins a session.

    `arguments`: the flags it was started with. `parameters`: what it resolved
    and runs with. `files`: what it read besides the config (the transformation
    matrices, the speaker profiles). `config`: the config it loaded, kept with
    its secrets masked. `services`: what its servers answered, when asked
    before (record_services_later asks after)."""
    host = host or short_hostname()
    return {
        "key": component_key(pipeline, role, component_id),
        "pipeline": pipeline,
        "role": role,
        "id": None if component_id is None else str(component_id),
        "host": host,
        "pid": os.getpid(),
        "started_at": now or datetime.now(timezone.utc),
        "software": software_info(project_dir),
        "arguments": plain(arguments or {}),
        "parameters": plain(parameters or {}),
        "files": plain(files or {}),
        "services": None if services is None else plain(services),
        "config": None if config is None else redact_secrets(plain(config)),
        "config_path": None if config_path is None else str(config_path),
        "config_sha256": file_digest(config_path),
    }


def entry_file(project_dir, session_id: str, pipeline_name: str, entry: dict) -> Path:
    """where the entry goes on the component's machine: the config folder of the
    session's pipeline artifacts, as <role>.json or <role>_<id>.json."""
    name = safe_segment(str(entry.get("role") or "component"), "component")
    if entry.get("id") not in (None, ""):
        name = f"{name}_{safe_segment(str(entry['id']), 'id')}"
    return pipeline_section_dir(project_dir, session_id, pipeline_name, "config", entry.get("host")) / f"{name}.json"


def write_entry_file(project_dir, session_id: str, pipeline_name: str, entry: dict, log=None) -> Path | None:
    """write the entry next to the config the component copied; never raises."""
    log = log or logger
    try:
        path = entry_file(project_dir, session_id, pipeline_name, entry)
        path.write_text(json.dumps(entry, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception as error:
        log.warning("Could not write what %s runs with into the session's artifacts: %s", entry.get("key"), error)
        return None


def record_component(mongo, session_id: str | None, entry: dict, project_dir=None, pipeline_name: str | None = None,
                     log=None) -> bool:
    """note in the session what a component runs with: into MongoDB, and as a
    file under the session's artifacts when a pipeline is named. Never raises;
    True when MongoDB took it."""
    log = log or logger
    if project_dir is not None and pipeline_name:
        write_entry_file(project_dir, session_id or "session", pipeline_name, entry, log=log)
    if mongo is None or not session_id:
        return False
    try:
        written = bool(mongo.add_session_component(session_id, entry))
    except Exception as error:  # a component runs on without it
        log.warning("Could not note in session %s what %s runs with: %s", session_id, entry.get("key"), error)
        return False
    if not written:
        log.warning("Session %s is not in MongoDB: what %s runs with is not noted there.", session_id, entry.get("key"))
    return written


# ---- what the servers run ----

def info_url(url: str) -> str:
    """the address a server answers about itself on: its endpoint's /info."""
    text = str(url or "").strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return text.rstrip("/") + "/" + INFO_PATH
    path = parts.path.rstrip("/") + "/" + INFO_PATH
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def describe_services(urls: dict[str, str], timeout: float = 3.0) -> dict:
    """ask each server what it runs (GET <url>/info): its answer per name, with
    the url asked; an `error` instead for one that has no /info (an older
    server) or does not answer."""
    described: dict[str, dict] = {}
    for name, url in (urls or {}).items():
        if not url:
            continue
        described[name] = _describe_service(str(url), timeout)
    return described


def _describe_service(url: str, timeout: float) -> dict:
    asked = info_url(url)
    try:
        import requests
        response = requests.get(asked, timeout=timeout)
    except Exception as error:
        # requests nests the cause several times over; the first line says enough
        return {"url": url, "error": f"{type(error).__name__}: {str(error).split(' (Caused by', 1)[0][:200]}"}
    if response.status_code == 404:
        return {"url": url, "error": "no /info: the server runs an openmmla without it"}
    if response.status_code != 200:
        return {"url": url, "error": f"HTTP {response.status_code}"}
    try:
        answer = response.json()
    except ValueError:
        return {"url": url, "error": "not a JSON answer"}
    if not isinstance(answer, dict):
        return {"url": url, "error": "not a JSON object"}
    return {"url": url, **redact_secrets(plain(answer))}


def record_services_later(mongo, session_id: str | None, entry: dict, urls: dict[str, str], project_dir=None,
                          pipeline_name: str | None = None, log=None, timeout: float = 3.0) -> threading.Thread | None:
    """ask the component's servers what they run and set the answers into its
    entry, in MongoDB and in its file, without holding the component up: the
    asking happens in a daemon thread. Returns the thread, None when there is
    no server to ask."""
    log = log or logger
    urls = {name: url for name, url in (urls or {}).items() if url}
    if not urls:
        return None

    def work():
        services = describe_services(urls, timeout=timeout)
        entry["services"] = services
        if mongo is not None and session_id:
            try:
                if not mongo.set_session_component_field(session_id, entry["key"], "services", services):
                    log.warning("Session %s is not in MongoDB: what the servers of %s run is not noted there.",
                                session_id, entry.get("key"))
            except Exception as error:
                log.warning("Could not note in session %s what the servers of %s run: %s",
                            session_id, entry.get("key"), error)
        if project_dir is not None and pipeline_name:
            write_entry_file(project_dir, session_id or "session", pipeline_name, entry, log=log)

    thread = threading.Thread(target=work, name=f"provenance-{entry.get('key')}", daemon=True)
    thread.start()
    return thread


# ---- reading it back ----

def session_components(record: dict | None) -> list[dict]:
    """the entries of a session document, in the order they were written."""
    components = (record or {}).get(COMPONENTS_FIELD)
    return [entry for entry in components if isinstance(entry, dict)] if isinstance(components, list) else []


def component_summary(entry: dict) -> str:
    """one line for a log: the component, its host, its flags, and what its servers run."""
    parts = [str(entry.get("key") or "?"), f"@ {entry.get('host') or '?'}"]
    started = entry.get("started_at")
    if isinstance(started, datetime):
        parts.append(started.strftime("%Y-%m-%d %H:%M:%SZ"))
    elif started:
        parts.append(str(started))
    arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
    flags = " ".join(f"{name}={value}" for name, value in arguments.items()
                     if value not in (None, "") and name != "session_id")
    if flags:
        parts.append(f"· {flags}")
    services = entry.get("services") if isinstance(entry.get("services"), dict) else {}
    told = []
    for name, answer in services.items():
        if not isinstance(answer, dict):
            continue
        if answer.get("error"):
            told.append(f"{name}: {answer['error']}")
            continue
        model = answer.get("model") or answer.get("vlm_model") or answer.get("backend")
        told.append(f"{name}: {model}" if model else name)
    if told:
        parts.append("· " + "; ".join(told))
    return " ".join(parts)


def session_parameters(record: dict) -> dict:
    """what Export Measurements writes next to a session's measurements: the
    session's own fields, the streams its bases took, and what every component ran with."""
    core = {name: record.get(name) for name in
            ("session_id", "experiment_id", "group_id", "participants", "start_time", "end_time", "status", "metadata")}
    return {
        "session": core,
        "sources": [entry for entry in (record.get("sources") or []) if isinstance(entry, dict)]
        if isinstance(record.get("sources"), list) else [],
        COMPONENTS_FIELD: session_components(record),
    }


def write_session_parameters(record: dict, out_dir) -> Path:
    """write <session>_parameters.json into a folder (the session's measurements)."""
    session_id = str(record.get("session_id") or "session")
    path = Path(out_dir) / f"{safe_segment(session_id, 'session')}_parameters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session_parameters(record), indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return path


# ---- the analysis layer: what was made from the measurements, and from what ----

ANALYSIS_RECORD_NAME = "parameters.json"


def _file_note(path, root=None) -> dict:
    """a file as an input or output of an analysis: its path (relative to a root
    when given), size and digest, and for a measurements export its record count."""
    file = Path(path)
    note: dict = {"path": str(file)}
    if root is not None:
        try:
            note["path"] = file.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    try:
        note["bytes"] = file.stat().st_size
    except OSError:
        note["missing"] = True
        return note
    note["sha256"] = file_digest(file)
    if file.suffix == ".json":
        try:
            loaded = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, list):
            note["records"] = len(loaded)
    return note


def analysis_record(session_id: str, *, inputs, outputs, steps, parameters: dict | None = None,
                    root=None, project_dir=None, now: datetime | None = None) -> dict:
    """what an analysis of a session was made from and with: the measurement files
    it read (with their digests and record counts), the files it wrote, the steps
    it ran, its parameters (none yet: the plots take no thresholds) and the
    software. Sessions -> Export Visualizations writes it as analysis/parameters.json,
    so a plot can be traced to the measurements export it came from, which
    <session>_parameters.json traces to the components that made them."""
    return {
        "session_id": session_id,
        "layer": "analysis",
        "produced_at": now or datetime.now(timezone.utc),
        "software": software_info(project_dir),
        "steps": [str(step) for step in steps],
        "parameters": plain(parameters or {}),
        "inputs": [_file_note(path, root) for path in inputs],
        "outputs": [_file_note(path, root) for path in outputs],
    }


def write_analysis_record(record: dict, analysis_dir) -> Path:
    """write analysis/parameters.json."""
    path = Path(analysis_dir) / ANALYSIS_RECORD_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return path
