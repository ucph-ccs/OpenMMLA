from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml


ARTIFACTS_DIR = "artifacts"
METADATA_FILENAMES = {"manifest.json", "manifest.yml", "config.yml"}
PIPELINE_SLUGS = {
    "ASR Base": "asr-base",
    "VFA Base": "vfa-base",
    "IPS Base": "ips-base",
}


def safe_segment(value: str | None, default: str) -> str:
    """return a filesystem-safe path segment."""
    raw = str(value or "").strip() or default
    segment = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("._-")
    if segment in {"", ".", ".."}:
        return default
    return segment


def pipeline_slug(name: str) -> str:
    return PIPELINE_SLUGS.get(name, safe_segment(name.lower().replace(" ", "-"), "pipeline"))


def artifact_session_dir(project_root: str | os.PathLike[str], session_id: str) -> Path:
    return Path(project_root) / ARTIFACTS_DIR / safe_segment(session_id, "session")


def collection_artifact_dir(
    project_root: str | os.PathLike[str],
    session_id: str,
    host_name: str,
) -> Path:
    return artifact_session_dir(project_root, session_id) / "collection" / safe_segment(host_name, "host")


def pipeline_artifact_dir(
    project_root: str | os.PathLike[str],
    session_id: str,
    pipeline_name: str,
    host_name: str,
) -> Path:
    return (
        artifact_session_dir(project_root, session_id)
        / "pipelines"
        / pipeline_slug(pipeline_name)
        / safe_segment(host_name, "host")
    )


def manifest_path(project_root: str | os.PathLike[str], session_id: str) -> Path:
    return artifact_session_dir(project_root, session_id) / "manifest.yml"


def manifest_json_path(project_root: str | os.PathLike[str], session_id: str) -> Path:
    return artifact_session_dir(project_root, session_id) / "manifest.json"


def ensure_session_layout(project_root: str | os.PathLike[str], session_id: str) -> Path:
    session_dir = artifact_session_dir(project_root, session_id)
    for rel_path in (
        Path("collection"),
        Path("pipelines"),
        Path("measurements"),
        Path("exports") / "influxdb",
        Path("exports") / "mongodb",
        Path("visualizations"),
        Path("analysis") / "features",
        Path("analysis") / "visualizations",
    ):
        (session_dir / rel_path).mkdir(parents=True, exist_ok=True)
    return session_dir


def relative_to_root(project_root: str | os.PathLike[str], path: str | os.PathLike[str]) -> str:
    try:
        return os.path.relpath(Path(path), Path(project_root))
    except ValueError:
        return str(path)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_file(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return False
    if left.stat().st_size != right.stat().st_size:
        return False
    return _file_digest(left) == _file_digest(right)


def _coerce_sync_time(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _file_source_sync_times(data_sources: dict[str, Any]) -> list[float]:
    sync_times: list[float] = []
    for sources in data_sources.values():
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict) or source.get("source") != "file":
                continue
            sync_time = _coerce_sync_time(source.get("initial_sync_time"))
            if sync_time is not None:
                sync_times.append(sync_time)
    return sync_times


def _normalize_file_source_sync_times(data_sources: dict[str, Any], initial_sync_time: float) -> None:
    for sources in data_sources.values():
        if not isinstance(sources, list):
            continue
        for source in sources:
            if isinstance(source, dict) and source.get("source") == "file":
                source["initial_sync_time"] = initial_sync_time


def _manifest_initial_sync_time(root: Path) -> float | None:
    for path in (root / "manifest.json", root / "manifest.yml"):
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as file:
                if path.suffix == ".json":
                    data = json.load(file)
                else:
                    data = yaml.safe_load(file)
        except (OSError, json.JSONDecodeError, yaml.YAMLError):
            continue
        if isinstance(data, dict):
            sync_time = _coerce_sync_time(data.get("initial_sync_time"))
            if sync_time is not None:
                return sync_time
    return None


def _nearest_initial_sync_time(path: Path) -> float | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        sync_time = _manifest_initial_sync_time(candidate)
        if sync_time is not None:
            return sync_time
    return None


def _conflict_path(path: Path, label: str) -> Path:
    safe_label = safe_segment(label, "remote")
    candidate = path.with_name(f"{path.stem}_{safe_label}{path.suffix}")
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{safe_label}_{index}{path.suffix}")
        index += 1
    return candidate


def merge_tree(source: Path, destination: Path, *, conflict_label: str) -> dict[str, int]:
    """merge source into destination without clobbering media/data files."""
    stats = {"copied": 0, "skipped": 0, "conflicted": 0}
    if source.is_file():
        _merge_file(source, destination, conflict_label=conflict_label, stats=stats)
        return stats

    destination.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(source):
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        rel_root = Path(root).relative_to(source)
        dest_root = destination / rel_root
        dest_root.mkdir(parents=True, exist_ok=True)
        for filename in files:
            if filename in {".DS_Store", ".manifest.lock"}:
                continue
            _merge_file(
                Path(root) / filename,
                dest_root / filename,
                conflict_label=conflict_label,
                stats=stats,
            )
    return stats


def _merge_file(source: Path, destination: Path, *, conflict_label: str, stats: dict[str, int]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
        stats["copied"] += 1
        return
    if destination.name in METADATA_FILENAMES:
        shutil.copy2(source, destination)
        stats["copied"] += 1
        return
    if _same_file(source, destination):
        stats["skipped"] += 1
        return
    conflict = _conflict_path(destination, conflict_label)
    shutil.copy2(source, conflict)
    stats["conflicted"] += 1


def update_collection_manifest(
    project_root: str | os.PathLike[str],
    *,
    session_id: str,
    host_name: str,
    remote_path: str,
    local_path: Path,
) -> Path:
    entry = {
        "host": host_name,
        "remote_path": remote_path,
        "local_path": relative_to_root(project_root, local_path),
        "downloaded_at": _timestamp(),
    }
    file_sources = {}
    audio_dir = local_path / "audio"
    video_dir = local_path / "video"
    collection_manifest = _read_manifest_pair(local_path / "manifest.yml", local_path / "manifest.json")
    initial_sync_time = _coerce_sync_time(collection_manifest.get("initial_sync_time"))
    if audio_dir.exists():
        file_sources["asr"] = _file_source_entry(
            host_name,
            "collection",
            audio_dir,
            initial_sync_time=initial_sync_time,
        )
    if video_dir.exists():
        file_sources["ips"] = _file_source_entry(
            host_name,
            "collection",
            video_dir,
            initial_sync_time=initial_sync_time,
        )
        file_sources["vfa"] = _file_source_entry(
            host_name,
            "collection",
            video_dir,
            initial_sync_time=initial_sync_time,
        )
    return update_session_manifest(
        project_root,
        session_id=session_id,
        collection_entry=entry,
        file_sources=file_sources,
        recordings=_collection_recordings(local_path, collection_manifest),
    )


def update_pipeline_manifest(
    project_root: str | os.PathLike[str],
    *,
    session_id: str,
    pipeline_name: str,
    host_name: str,
    remote_root: str,
    local_path: Path,
    downloaded_paths: list[str],
) -> Path:
    entry = {
        "host": host_name,
        "remote_root": remote_root,
        "local_path": relative_to_root(project_root, local_path),
        "downloaded_paths": downloaded_paths,
        "downloaded_at": _timestamp(),
    }
    file_sources = _pipeline_file_sources(
        session_id=session_id,
        pipeline_name=pipeline_name,
        host_name=host_name,
        local_path=local_path,
    )
    return update_session_manifest(
        project_root,
        session_id=session_id,
        pipeline_name=pipeline_name,
        pipeline_entry=entry,
        file_sources=file_sources,
    )


def _first_existing_dir(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_dir():
            return path
    return None


def _pipeline_file_sources(
    *,
    session_id: str,
    pipeline_name: str,
    host_name: str,
    local_path: Path,
) -> dict[str, dict[str, Any]]:
    slug = pipeline_slug(pipeline_name)
    safe_session = safe_segment(session_id, "session")
    sources: dict[str, dict[str, Any]] = {}

    if slug == "asr-base":
        file_dir = _first_existing_dir([
            local_path / "real-time" / "runtime",
            local_path / "post-time",
            local_path / "collection" / safe_session / "audio",
            local_path / "post-time" / "recordings" / safe_session / "audio",
            local_path / "post-time" / safe_session / "audio",
            local_path / "real-time" / "runtime" / safe_session,
        ])
        if file_dir is not None:
            sources["asr"] = _file_source_entry(host_name, slug, file_dir)
        return sources

    if slug in {"ips-base", "vfa-base"}:
        modality = "ips" if slug == "ips-base" else "vfa"
        video_dir = _first_existing_dir([
            local_path / "post-time" / "video",
            local_path / "collection" / safe_session / "video",
            local_path / "post-time" / "recordings" / safe_session / "video",
            local_path / "post-time" / safe_session / "video",
        ])
        if video_dir is not None:
            sources[modality] = _file_source_entry(host_name, slug, video_dir)
        elif slug == "vfa-base":
            frame_dir = _first_existing_dir([
                local_path / "real-time" / "runtime",
                local_path / "real-time" / "runtime" / safe_session,
            ])
            if frame_dir is not None:
                sources["vfa"] = _frame_source_entry(host_name, slug, frame_dir)
    return sources


def _file_source_entry(
    host_name: str,
    pipeline_name: str,
    file_dir: Path,
    *,
    initial_sync_time: float | None = None,
) -> dict[str, Any]:
    if initial_sync_time is None:
        initial_sync_time = _nearest_initial_sync_time(file_dir)
    entry = {
        "host": host_name,
        "pipeline": pipeline_name,
        "source": "file",
        "file_dir": str(file_dir),
    }
    if initial_sync_time is not None:
        entry["initial_sync_time"] = initial_sync_time
    return entry


def _frame_source_entry(host_name: str, pipeline_name: str, frame_dir: Path) -> dict[str, Any]:
    return {
        "host": host_name,
        "pipeline": pipeline_name,
        "source": "frames",
        "frame_dir": str(frame_dir),
    }


def update_session_manifest(
    project_root: str | os.PathLike[str],
    *,
    session_id: str,
    collection_entry: dict[str, Any] | None = None,
    pipeline_name: str | None = None,
    pipeline_entry: dict[str, Any] | None = None,
    file_sources: dict[str, dict[str, Any]] | None = None,
    recordings: list[dict[str, Any]] | None = None,
) -> Path:
    ensure_session_layout(project_root, session_id)
    path = manifest_path(project_root, session_id)
    json_path = manifest_json_path(project_root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_manifest_pair(path, json_path)
    data["session_id"] = session_id
    data["updated_at"] = _timestamp()
    artifacts = data.setdefault("artifacts", {})

    if collection_entry:
        collection = artifacts.setdefault("collection", [])
        _upsert_entry(collection, collection_entry, keys=("host", "local_path"))

    if pipeline_name and pipeline_entry:
        pipelines = artifacts.setdefault("pipelines", {})
        entries = pipelines.setdefault(pipeline_slug(pipeline_name), [])
        _upsert_entry(entries, pipeline_entry, keys=("host", "local_path"))

    if file_sources:
        data_sources = data.setdefault("file_sources", {})
        for modality, source in file_sources.items():
            sources = data_sources.get(modality)
            if not isinstance(sources, list):
                sources = []
                data_sources[modality] = sources
            _upsert_entry(sources, source, keys=_source_keys(source))
        sync_times = _file_source_sync_times(data_sources)
        existing_sync_time = _coerce_sync_time(data.get("initial_sync_time"))
        if existing_sync_time is not None:
            sync_times.append(existing_sync_time)
        if sync_times:
            data["initial_sync_time"] = max(sync_times)
            _normalize_file_source_sync_times(data_sources, data["initial_sync_time"])

    if recordings:
        existing_recordings = data.setdefault("recordings", [])
        if not isinstance(existing_recordings, list):
            existing_recordings = []
            data["recordings"] = existing_recordings
        for recording in recordings:
            _upsert_entry(existing_recordings, recording, keys=("id",))

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=False)
        file.write("\n")
    return path


def _read_manifest_pair(yml_path: Path, json_path: Path) -> dict[str, Any]:
    data = _read_yaml(yml_path)
    if data:
        return data
    return _read_json(json_path)


def _collection_recordings(local_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    recordings = manifest.get("recordings")
    if not isinstance(recordings, list):
        return []
    normalized = []
    for recording in recordings:
        if not isinstance(recording, dict):
            continue
        item = dict(recording)
        path = str(item.get("path") or "").strip()
        if path:
            item["path"] = _local_collection_recording_path(local_path, path)
        normalized.append(item)
    return normalized


def _local_collection_recording_path(local_path: Path, original_path: str) -> str:
    original = Path(original_path)
    if not original.is_absolute():
        return str((local_path / original).resolve())
    for marker in ("audio", "video"):
        if marker in original.parts:
            index = original.parts.index(marker)
            return str(local_path.joinpath(*original.parts[index:]))
    return str(original)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _upsert_entry(entries: list[dict[str, Any]], entry: dict[str, Any], *, keys: tuple[str, ...]) -> None:
    for index, existing in enumerate(entries):
        if not isinstance(existing, dict):
            continue
        if all(existing.get(key) == entry.get(key) for key in keys):
            merged = dict(existing)
            merged.update(entry)
            entries[index] = merged
            return
    entries.append(entry)


def _source_keys(source: dict[str, Any]) -> tuple[str, ...]:
    if source.get("source") == "frames":
        return ("host", "source", "frame_dir")
    return ("host", "source", "file_dir")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
