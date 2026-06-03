"""data-source readers for OpenMMLA-GD dataset export."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openmmla.utils.constants import (
    EVENT_TYPE_ASR_RECOGNITION,
    EVENT_TYPE_ASR_TRANSCRIPTION,
    EVENT_TYPE_IPS_RELATION,
    EVENT_TYPE_IPS_ROTATION,
    EVENT_TYPE_IPS_TRANSLATION,
    EVENT_TYPE_VFA_ACTION,
)
from openmmla.utils.querys import deep_parse_json

MEASUREMENT_FILENAME_HINTS = {
    "asr_recognition": EVENT_TYPE_ASR_RECOGNITION,
    "speaker_recognition": EVENT_TYPE_ASR_RECOGNITION,
    "asr_transcription": EVENT_TYPE_ASR_TRANSCRIPTION,
    "speaker_transcription": EVENT_TYPE_ASR_TRANSCRIPTION,
    "ips_translation": EVENT_TYPE_IPS_TRANSLATION,
    "badge_translation": EVENT_TYPE_IPS_TRANSLATION,
    "ips_rotation": EVENT_TYPE_IPS_ROTATION,
    "badge_rotation": EVENT_TYPE_IPS_ROTATION,
    "ips_relation": EVENT_TYPE_IPS_RELATION,
    "badge_relation": EVENT_TYPE_IPS_RELATION,
    "vfa_action": EVENT_TYPE_VFA_ACTION,
    "action_recognition": EVENT_TYPE_VFA_ACTION,
}

OPENMMLA_GD_EVENT_TYPES = (
    EVENT_TYPE_ASR_RECOGNITION,
    EVENT_TYPE_ASR_TRANSCRIPTION,
    EVENT_TYPE_IPS_RELATION,
    EVENT_TYPE_IPS_ROTATION,
    EVENT_TYPE_IPS_TRANSLATION,
    EVENT_TYPE_VFA_ACTION,
)


@dataclass(frozen=True, slots=True)
class RecordingRef:
    id: str
    modality: str
    path: str
    host: str
    start_time: float
    end_time: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def overlaps(self, start_time: float, end_time: float) -> bool:
        recording_end = self.end_time if self.end_time is not None else end_time
        return self.start_time < end_time and recording_end > start_time

    def as_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modality": self.modality,
            "path": self.path,
            "host": self.host,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


def load_session_manifest(project_dir: str, session_id: str, artifacts_root: str = "artifacts") -> dict[str, Any]:
    session_dir = session_artifact_dir(project_dir, session_id, artifacts_root)
    manifest_path = os.path.join(session_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Session manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        return deep_parse_json(json.load(manifest_file))


def session_artifact_dir(project_dir: str, session_id: str, artifacts_root: str = "artifacts") -> str:
    root = artifacts_root if os.path.isabs(artifacts_root) else os.path.join(project_dir, artifacts_root)
    return os.path.join(root, session_id)


def load_recordings(project_dir: str, session_id: str, artifacts_root: str = "artifacts") -> list[RecordingRef]:
    session_dir = session_artifact_dir(project_dir, session_id, artifacts_root)
    manifests = _load_all_recording_manifests(session_dir)
    recordings: list[RecordingRef] = []
    seen: set[tuple[str, str]] = set()

    for manifest_dir, manifest in manifests:
        for item in manifest.get("recordings", []) or []:
            if not isinstance(item, dict):
                continue
            recording = _recording_from_item(session_dir, manifest_dir, item)
            if recording is None:
                continue
            key = (recording.id, recording.path)
            if key in seen:
                continue
            seen.add(key)
            recordings.append(recording)

    return sorted(recordings, key=lambda item: (item.start_time, item.modality, item.host))


def infer_session_bounds(manifest: dict[str, Any], recordings: list[RecordingRef], default_duration: float = 30.0) -> tuple[float, float]:
    starts = [recording.start_time for recording in recordings if recording.start_time > 0]
    ends = [recording.end_time for recording in recordings if recording.end_time is not None and recording.end_time > 0]

    if starts and ends:
        return min(starts), max(float(end_time) for end_time in ends)

    initial_sync_time = _coerce_float(manifest.get("initial_sync_time") or manifest.get("created_at"))
    if initial_sync_time is not None:
        return initial_sync_time, initial_sync_time + default_duration

    raise ValueError("Cannot infer session bounds from manifest or recordings.")


def collect_measurement_events(
    project_dir: str,
    session_id: str,
    artifacts_root: str = "artifacts",
    source: str = "artifacts",
    config_path: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if source not in {"artifacts", "influx", "both"}:
        raise ValueError("source must be one of: artifacts, influx, both.")

    events: dict[str, list[dict[str, Any]]] = {event_type: [] for event_type in OPENMMLA_GD_EVENT_TYPES}
    if source in {"artifacts", "both"}:
        _merge_events(events, _load_artifact_measurements(project_dir, session_id, artifacts_root))
    if source in {"influx", "both"}:
        if not config_path:
            raise ValueError("config_path is required when source is 'influx' or 'both'.")
        _merge_events(events, _query_influx_measurements(config_path, session_id))

    for event_type in list(events):
        events[event_type] = sorted(
            (deep_parse_json(event) for event in events[event_type]),
            key=lambda item: event_start_time(item),
        )
    return events


def event_start_time(event: dict[str, Any]) -> float:
    for key in ("window_start_time", "segment_start_time", "start_time"):
        value = _coerce_float(event.get(key))
        if value is not None:
            return value
    time_value = event.get("time")
    if isinstance(time_value, datetime):
        return time_value.timestamp()
    value = _coerce_float(time_value)
    return value if value is not None else 0.0


def event_end_time(event: dict[str, Any]) -> float:
    value = _coerce_float(event.get("window_end_time") or event.get("chunk_end_time") or event.get("end_time"))
    return value if value is not None else event_start_time(event)


def events_for_window(
    events_by_type: dict[str, list[dict[str, Any]]],
    window_start: float,
    window_end: float,
) -> dict[str, list[dict[str, Any]]]:
    window_events: dict[str, list[dict[str, Any]]] = {}
    for event_type, events in events_by_type.items():
        window_events[event_type] = [
            event for event in events
            if event_end_time(event) >= window_start and event_start_time(event) <= window_end
        ]
    return window_events


def infer_participants(events_by_type: dict[str, list[dict[str, Any]]], configured: list[str] | None = None) -> list[str]:
    participants = {str(item) for item in configured or [] if str(item).strip()}
    for event in events_by_type.get(EVENT_TYPE_IPS_TRANSLATION, []):
        translations = event.get("translations", {})
        if isinstance(translations, dict):
            participants.update(str(key) for key in translations)
    for event in events_by_type.get(EVENT_TYPE_IPS_RELATION, []):
        graph = event.get("graph", {})
        if isinstance(graph, dict):
            participants.update(str(key) for key in graph)
            for targets in graph.values():
                if isinstance(targets, list):
                    participants.update(str(target) for target in targets)
    return sorted(participants)


def raw_refs_for_window(recordings: list[RecordingRef], window_start: float, window_end: float) -> dict[str, list[dict[str, Any]]]:
    refs: dict[str, list[dict[str, Any]]] = {"audio": [], "video": []}
    for recording in recordings:
        if recording.modality in refs and recording.overlaps(window_start, window_end):
            refs[recording.modality].append(recording.as_record())
    return refs


def _load_all_recording_manifests(session_dir: str) -> list[tuple[str, dict[str, Any]]]:
    manifests: list[tuple[str, dict[str, Any]]] = []
    root_manifest = os.path.join(session_dir, "manifest.json")
    if os.path.exists(root_manifest):
        manifests.append((session_dir, _read_json(root_manifest)))

    collection_dir = os.path.join(session_dir, "collection")
    if os.path.isdir(collection_dir):
        for host in sorted(os.listdir(collection_dir)):
            manifest_path = os.path.join(collection_dir, host, "manifest.json")
            if os.path.exists(manifest_path):
                manifests.append((os.path.dirname(manifest_path), _read_json(manifest_path)))
    return manifests


def _recording_from_item(session_dir: str, manifest_dir: str, item: dict[str, Any]) -> RecordingRef | None:
    modality = str(item.get("modality") or "").strip().lower()
    if modality not in {"audio", "video"}:
        return None
    start_time = _coerce_float(item.get("start_time") or item.get("created_at"))
    if start_time is None:
        return None
    end_time = _coerce_float(item.get("stopped_at") or item.get("end_time"))
    path = _resolve_recording_path(session_dir, manifest_dir, item)
    return RecordingRef(
        id=str(item.get("id") or os.path.basename(path)),
        modality=modality,
        path=path,
        host=str(item.get("host") or os.path.basename(manifest_dir)),
        start_time=start_time,
        end_time=end_time,
        metadata={key: value for key, value in item.items() if key not in {"id", "modality", "path", "host", "start_time", "stopped_at", "end_time"}},
    )


def _resolve_recording_path(session_dir: str, manifest_dir: str, item: dict[str, Any]) -> str:
    original = str(item.get("path") or "").strip()
    if original and os.path.exists(original):
        return original
    if original and not os.path.isabs(original):
        candidate = os.path.abspath(os.path.join(manifest_dir, original))
        if os.path.exists(candidate):
            return candidate

    modality = str(item.get("modality") or "").strip().lower()
    host = str(item.get("host") or os.path.basename(manifest_dir))
    filename = os.path.basename(original) if original else str(item.get("id") or "")
    local_candidate = os.path.join(session_dir, "collection", host, modality, filename)
    return os.path.abspath(local_candidate)


def _load_artifact_measurements(project_dir: str, session_id: str, artifacts_root: str) -> dict[str, list[dict[str, Any]]]:
    session_dir = session_artifact_dir(project_dir, session_id, artifacts_root)
    events: dict[str, list[dict[str, Any]]] = {event_type: [] for event_type in OPENMMLA_GD_EVENT_TYPES}
    for relative_dir in ("measurements", os.path.join("exports", "influxdb")):
        root = os.path.join(session_dir, relative_dir)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                event_type = _event_type_from_filename(filename)
                if event_type is None:
                    continue
                path = os.path.join(dirpath, filename)
                events[event_type].extend(_read_measurement_file(path))
    return events


def _query_influx_measurements(config_path: str, session_id: str) -> dict[str, list[dict[str, Any]]]:
    from openmmla.utils.client import InfluxDBClientWrapper

    client = InfluxDBClientWrapper(config_path)
    try:
        return {
            event_type: [deep_parse_json(event) for event in client.query_events(session_id, event_type)]
            for event_type in OPENMMLA_GD_EVENT_TYPES
        }
    finally:
        client.close()


def _event_type_from_filename(filename: str) -> str | None:
    lowered = filename.lower()
    for hint, event_type in MEASUREMENT_FILENAME_HINTS.items():
        if hint in lowered:
            return event_type
    return None


def _read_measurement_file(path: str) -> list[dict[str, Any]]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        data = deep_parse_json(_read_json(path))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            records = data.get("records") or data.get("events")
            if isinstance(records, list):
                return [item for item in records if isinstance(item, dict)]
            return [data]
    if suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if text:
                    records.append(deep_parse_json(json.loads(text)))
        return [item for item in records if isinstance(item, dict)]
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8") as file:
            return [deep_parse_json(dict(row)) for row in csv.DictReader(file)]
    return []


def _merge_events(target: dict[str, list[dict[str, Any]]], source: dict[str, list[dict[str, Any]]]) -> None:
    for event_type, events in source.items():
        target.setdefault(event_type, []).extend(events)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return deep_parse_json(data)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
