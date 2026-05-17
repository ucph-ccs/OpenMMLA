from __future__ import annotations

import os
import time
from typing import Any

import yaml


REGISTRY_PATH = os.path.join("real-time", "runtime", "stream_registry.yml")


def stream_registry_path(project_dir: str | None = None, registry_path: str | None = None) -> str:
    """return the runtime stream registry path."""
    if registry_path:
        return registry_path
    root = project_dir or os.getcwd()
    return os.path.join(root, REGISTRY_PATH)


def load_stream_registry(project_dir: str | None = None, registry_path: str | None = None) -> dict[str, Any]:
    """load runtime stream metadata."""
    path = stream_registry_path(project_dir, registry_path)
    if not os.path.isfile(path):
        return {"streams": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        return {"streams": {}}
    if not isinstance(data, dict):
        return {"streams": {}}
    streams = data.get("streams", {})
    if not isinstance(streams, dict):
        data["streams"] = {}
    return data


def save_stream_registry(
    data: dict[str, Any],
    project_dir: str | None = None,
    registry_path: str | None = None,
) -> None:
    """save runtime stream metadata."""
    path = stream_registry_path(project_dir, registry_path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def register_stream_start(
    stream_name: str,
    target: str,
    stream_start_time: float,
    *,
    project_dir: str | None = None,
    registry_path: str | None = None,
    ssh_profile: str = "",
    device: str = "",
) -> dict[str, Any]:
    """record that a managed stream has started."""
    data = load_stream_registry(project_dir, registry_path)
    streams = data.setdefault("streams", {})
    entry = {
        "name": stream_name,
        "target": target,
        "ssh_profile": ssh_profile,
        "device": device,
        "stream_start_time": float(stream_start_time),
        "registered_at": time.time(),
        "status": "running",
    }
    streams[stream_name] = entry
    save_stream_registry(data, project_dir, registry_path)
    return entry


def mark_stream_stopped(
    stream_name: str,
    *,
    project_dir: str | None = None,
    registry_path: str | None = None,
) -> None:
    """record that a managed stream has stopped."""
    data = load_stream_registry(project_dir, registry_path)
    entry = data.setdefault("streams", {}).get(stream_name)
    if isinstance(entry, dict):
        entry["status"] = "stopped"
        entry["stopped_at"] = time.time()
        save_stream_registry(data, project_dir, registry_path)


def resolve_stream_by_target(
    target: str,
    *,
    project_dir: str | None = None,
    registry_path: str | None = None,
) -> dict[str, Any] | None:
    """resolve the newest running stream metadata for a target URL."""
    data = load_stream_registry(project_dir, registry_path)
    matches = [
        entry
        for entry in data.get("streams", {}).values()
        if isinstance(entry, dict)
        and entry.get("target") == target
        and entry.get("status") == "running"
        and entry.get("stream_start_time") is not None
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: float(item.get("registered_at") or 0))


def resolve_rtmp_timestamp(
    pts_ms: float,
    received_time: float,
    *,
    stream_start_time: float | None = None,
    receiver_pts_offset: float | None = None,
    stream_name: str = "",
) -> tuple[float, dict[str, Any], float | None]:
    """resolve an RTMP frame timestamp from stream-start metadata and media PTS."""
    metadata: dict[str, Any] = {
        "received_time": received_time,
        "rtmp_pts_ms": pts_ms,
    }
    if pts_ms and pts_ms > 0:
        pts_seconds = pts_ms / 1000.0
        metadata["rtmp_pts_seconds"] = pts_seconds
        if stream_start_time is not None and pts_seconds < 30 * 24 * 60 * 60:
            metadata["timestamp_source"] = "rtmp_stream_start_pts"
            metadata["stream_start_time"] = stream_start_time
            metadata["stream_name"] = stream_name
            return stream_start_time + pts_seconds, metadata, receiver_pts_offset
        if pts_seconds > 1_000_000_000:
            metadata["timestamp_source"] = "rtmp_absolute_pts"
            return pts_seconds, metadata, receiver_pts_offset
        if receiver_pts_offset is None:
            receiver_pts_offset = received_time - pts_seconds
        metadata["timestamp_source"] = "rtmp_receiver_calibrated_pts"
        metadata["receiver_pts_offset"] = receiver_pts_offset
        return pts_seconds + receiver_pts_offset, metadata, receiver_pts_offset

    metadata["timestamp_source"] = "receiver_wallclock"
    return received_time, metadata, receiver_pts_offset
