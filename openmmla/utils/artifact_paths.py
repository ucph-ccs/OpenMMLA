from __future__ import annotations

import os
import re
import shutil
import socket
from pathlib import Path


ARTIFACTS_DIR = "artifacts"
RUNTIME_ARTIFACTS_DIR = "runtime"
NON_SESSION_ARTIFACT_DIRS = {RUNTIME_ARTIFACTS_DIR}


def safe_segment(value: str | None, default: str = "item") -> str:
    """Return a string that is safe to use as one filesystem path segment."""
    raw = str(value or "").strip() or default
    segment = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("._-")
    if segment in {"", ".", ".."}:
        return default
    return segment


def short_hostname() -> str:
    return safe_segment(socket.gethostname().split(".", 1)[0], "host")


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
