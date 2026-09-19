from __future__ import annotations

import os
import re
import shutil
import socket
from datetime import date
from pathlib import Path


ARTIFACTS_DIR = "artifacts"
RUNTIME_ARTIFACTS_DIR = "runtime"

# stream recordings. A stream is shared by the sessions that pull it, so what
# records it files by day (capture side) or by stream path (server side), under
# streams/ of an artifacts folder; a session's part of them is filed under
# artifacts/<session>/streams/ with the same two sub-folders:
#   <record_root>/streams/capture/<YYYY-MM-DD>/<host label>/<video|audio>/<stream>_<start>.<mkv|wav>
#       on a capture host (record_root: ~/artifacts on a remote one, artifacts/ of
#       the project for 'local'); the day is the console's local date at Start
#   artifacts/streams/server/<app>/<name>/<YYYY-MM-DD_HH-MM-SS-ffffff>.mp4
#       MediaMTX, on the Stream Server's host (mediamtx.yml recordPath; the docker
#       bind is MEDIAMTX_STREAMS_DIR in docker/.env.example)
#   artifacts/streams/capture/<YYYY-MM-DD>/<host label>/<video|audio>/...
#       whole-file copies of the capture side on the console, tied to no session
#   artifacts/<session>/streams/server/<app>/<name>_<start>.mp4
#   artifacts/<session>/streams/capture/<host label>/<video|audio>/<stream>_<start>.<mkv|wav>
STREAMS_DIR = "streams"
SERVER_STREAMS_DIR = "server"
CAPTURE_STREAMS_DIR = "capture"
# below <record_root>/streams/ of a capture host: the cuts of a session, until fetched
SESSION_CUTS_DIR = ".session-cuts"
# the server-side record folder, relative to an artifacts folder
SERVER_RECORD_REL = f"{STREAMS_DIR}/{SERVER_STREAMS_DIR}"
# the capture-side record folder, relative to a record root
CAPTURE_RECORD_REL = f"{STREAMS_DIR}/{CAPTURE_STREAMS_DIR}"
# a capture host's record root when its stream names none (a remote host; 'local'
# records under artifacts/ of the project)
REMOTE_RECORD_ROOT = "$HOME/artifacts"
# the day folder of a capture-side recording, and a shell glob that matches one
# (one path segment only: none of its characters matches a /)
CAPTURE_DAY_FORMAT = "%Y-%m-%d"
CAPTURE_DAY_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
CAPTURE_KINDS = ("video", "audio")

# where the Stream Server recorded before streams/server/ (not read any more,
# but still no session)
LEGACY_RECORDINGS_DIR = "recordings"
NON_SESSION_ARTIFACT_DIRS = {RUNTIME_ARTIFACTS_DIR, STREAMS_DIR, LEGACY_RECORDINGS_DIR}

_CAPTURE_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def safe_segment(value: str | None, default: str = "item") -> str:
    """Return a string that is safe to use as one filesystem path segment."""
    raw = str(value or "").strip() or default
    segment = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("._-")
    if segment in {"", ".", ".."}:
        return default
    return segment


def short_hostname() -> str:
    return safe_segment(socket.gethostname().split(".", 1)[0], "host")


# ---- stream recordings ----
# the capture side is named in the shell of the capture host (a str, $HOME kept
# for it to expand); the console side takes the project root as it is given
# (the console's own checkout), without looking further up for pyproject.toml

def capture_day(moment: date | None = None) -> str:
    """the day folder of a capture-side recording started now (or on `moment`):
    the console's local date, which names the folder at Start."""
    return (moment or date.today()).strftime(CAPTURE_DAY_FORMAT)


def is_capture_day(name: str | None) -> bool:
    """True for a day folder of the capture side, 2026-09-19."""
    return bool(_CAPTURE_DAY.match(str(name or "")))


def capture_record_root(ssh_profile: str | None, record_root: str | None, project_root) -> str:
    """the record root on a stream's capture host: its record_root (~ spelled
    $HOME, which the remote shell expands), else artifacts/ of the project here
    for a stream captured on this machine ('local'), $HOME/artifacts on any other."""
    root = str(record_root or "").strip()
    if root:
        # one spelling of it everywhere: ffmpeg's path and the listings' must match
        root = re.sub(r"/{2,}", "/", root).rstrip("/") or "/"
        if root == "~" or root.startswith("~/"):
            root = "$HOME" + root[1:]
        return root
    if ssh_profile == "local":
        return os.path.join(str(project_root), ARTIFACTS_DIR)
    return REMOTE_RECORD_ROOT


def capture_host_label(ssh_profile: str | None) -> str:
    """the host folder a stream's recordings are filed under on its capture host
    and here: this machine's short name for 'local', else the SSH profile."""
    if ssh_profile == "local":
        return short_hostname()
    return safe_segment(ssh_profile, "host")


def capture_record_dir(record_root: str, day: str, host_label: str, kind: str) -> str:
    """<record_root>/streams/capture/<day>/<host label>/<kind>: where a stream
    started on `day` records on its capture host (not quoted for a shell)."""
    return f"{str(record_root).rstrip('/')}/{CAPTURE_RECORD_REL}/{day}/{host_label}/{kind}"


def streams_artifact_dir(project_root) -> Path:
    """artifacts/streams/ of the project: stream recordings tied to no session."""
    return Path(project_root) / ARTIFACTS_DIR / STREAMS_DIR


def server_record_dir(project_root) -> Path:
    """artifacts/streams/server/: where MediaMTX records on the Stream Server's host."""
    return streams_artifact_dir(project_root) / SERVER_STREAMS_DIR


def capture_copy_dir(project_root, day: str, host_label: str) -> Path:
    """artifacts/streams/capture/<day>/<host label>/: the whole files of one
    capture host's day, copied here (tied to no session)."""
    return (streams_artifact_dir(project_root) / CAPTURE_STREAMS_DIR / safe_segment(day, "day")
            / safe_segment(host_label, "host"))


def session_streams_dir(project_root, session_id: str) -> Path:
    """artifacts/<session>/streams/: a session's part of the stream recordings."""
    return Path(project_root) / ARTIFACTS_DIR / safe_segment(session_id, "session") / STREAMS_DIR


def session_server_streams_dir(project_root, session_id: str) -> Path:
    """artifacts/<session>/streams/server/: its clips from the Stream Server,
    <app>/<name>_<start>.mp4 below it."""
    return session_streams_dir(project_root, session_id) / SERVER_STREAMS_DIR


def session_capture_streams_dir(project_root, session_id: str, host_label: str | None = None) -> Path:
    """artifacts/<session>/streams/capture/[<host label>/]: its cuts of the
    capture-side recordings, <video|audio>/<stream>_<start>.<mkv|wav> below a host."""
    path = session_streams_dir(project_root, session_id) / CAPTURE_STREAMS_DIR
    return path / safe_segment(host_label, "host") if host_label is not None else path


def project_root_for_artifacts(project_dir: str | os.PathLike[str] | None) -> Path:
    """Resolve the OpenMMLA repository root for artifact storage.

    Runtime commands often pass a pipeline directory as project_dir. Artifacts are
    intentionally rooted at the repository level so sessions can aggregate output
    across pipelines and hosts.
    """
    start = Path(project_dir or os.getcwd()).expanduser().resolve()
    if start.is_file():
        start = start.parent

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "openmmla").is_dir():
            return candidate
    return start


def artifact_root(project_dir: str | os.PathLike[str] | None) -> Path:
    return project_root_for_artifacts(project_dir) / ARTIFACTS_DIR


def session_artifact_dir(
    project_dir: str | os.PathLike[str] | None,
    session_id: str,
) -> Path:
    return artifact_root(project_dir) / safe_segment(session_id, "session")


def collection_artifact_dir(
    project_dir: str | os.PathLike[str] | None,
    session_id: str,
    host_name: str | None = None,
) -> Path:
    return (
        session_artifact_dir(project_dir, session_id)
        / "collection"
        / safe_segment(host_name, short_hostname())
    )


def pipeline_artifact_dir(
    project_dir: str | os.PathLike[str] | None,
    session_id: str,
    pipeline_name: str,
    host_name: str | None = None,
) -> Path:
    return (
        session_artifact_dir(project_dir, session_id)
        / "pipelines"
        / safe_segment(pipeline_name, "pipeline")
        / safe_segment(host_name, short_hostname())
    )


def pipeline_section_dir(
    project_dir: str | os.PathLike[str] | None,
    session_id: str,
    pipeline_name: str,
    section: str,
    host_name: str | None = None,
) -> Path:
    path = pipeline_artifact_dir(project_dir, session_id, pipeline_name, host_name) / section
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_pipeline_artifact_dir(
    project_dir: str | os.PathLike[str] | None,
    pipeline_name: str,
    *parts: str,
    host_name: str | None = None,
) -> Path:
    """Return the cross-session runtime artifact directory for one pipeline."""
    path = (
        artifact_root(project_dir)
        / RUNTIME_ARTIFACTS_DIR
        / "pipelines"
        / safe_segment(pipeline_name, "pipeline")
        / safe_segment(host_name, short_hostname())
    )
    for part in parts:
        path = path / safe_segment(part, "item")
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_config_snapshot(
    config_path: str | os.PathLike[str] | None,
    project_dir: str | os.PathLike[str] | None,
    session_id: str,
    pipeline_name: str,
    host_name: str | None = None,
) -> Path | None:
    if not config_path:
        return None
    source = Path(config_path).expanduser()
    if not source.is_file():
        return None
    config_dir = pipeline_section_dir(project_dir, session_id, pipeline_name, "config", host_name)
    destination = config_dir / source.name
    shutil.copy2(source, destination)
    return destination
