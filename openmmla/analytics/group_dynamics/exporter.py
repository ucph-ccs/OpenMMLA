"""OpenMMLA-GD dataset exporter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .features import build_group_dynamics_features
from .schema import GROUP_STATE_LABELS, SCHEMA_VERSION, empty_labels
from .sources import (
    collect_measurement_events,
    events_for_window,
    infer_participants,
    infer_session_bounds,
    load_recordings,
    load_session_manifest,
    raw_refs_for_window,
    session_artifact_dir,
)
from .windowing import TimeWindow, build_windows


@dataclass(frozen=True, slots=True)
class ExportResult:
    output_dir: str
    windows_path: str
    manifest_path: str
    window_count: int


class GroupDynamicsExporter:
    """export classroom group-dynamics windows from OpenMMLA session artifacts."""

    def __init__(
        self,
        project_dir: str,
        session_id: str,
        config_path: str | None = None,
        artifacts_root: str = "artifacts",
        output_dir: str | None = None,
        source: str = "artifacts",
        window_size: float = 30.0,
        step_size: float = 15.0,
        participants: list[str] | None = None,
        group_id: str | None = None,
        audio_scope: str = "group",
    ):
        self.project_dir = os.path.abspath(project_dir)
        self.session_id = session_id
        self.config_path = config_path
        self.artifacts_root = artifacts_root
        self.source = source
        self.window_size = float(window_size)
        self.step_size = float(step_size)
        self.participants = participants or []
        self.group_id = group_id
        self.audio_scope = _normalize_audio_scope(audio_scope)
        self.session_dir = session_artifact_dir(self.project_dir, session_id, artifacts_root)
        self.output_dir = output_dir or os.path.join(self.session_dir, "analysis", "group_dynamics")

    def export(self) -> ExportResult:
        manifest = load_session_manifest(self.project_dir, self.session_id, self.artifacts_root)
        recordings = load_recordings(self.project_dir, self.session_id, self.artifacts_root)
        session_start, session_end = infer_session_bounds(manifest, recordings, default_duration=self.window_size)
        events = collect_measurement_events(
            project_dir=self.project_dir,
            session_id=self.session_id,
            artifacts_root=self.artifacts_root,
            source=self.source,
            config_path=self.config_path,
        )
        participants = infer_participants(events, self.participants)
        group_id = self.group_id or _infer_group_id(self.session_id, manifest)
        windows = build_windows(session_start, session_end, self.window_size, self.step_size)
        records = [
            self._record_for_window(window, recordings, events, participants, group_id, session_start)
            for window in windows
        ]
        return self._write_export(records, manifest, session_start, session_end, participants, group_id)

    def _record_for_window(
        self,
        window: TimeWindow,
        recordings,
        events: dict[str, list[dict[str, Any]]],
        participants: list[str],
        group_id: str,
        session_start: float,
    ) -> dict[str, Any]:
        raw_refs = raw_refs_for_window(recordings, window.start, window.end)
        window_events = events_for_window(events, window.start, window.end)
        features, modality_mask, quality = build_group_dynamics_features(
            window_events=window_events,
            raw_refs=raw_refs,
            participants=participants,
            audio_scope=self.audio_scope,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "group_id": group_id,
            "window_id": window.index,
            "window_start": window.start,
            "window_end": window.end,
            "relative_window_start": window.start - session_start,
            "relative_window_end": window.end - session_start,
            "participants": participants,
            "modality_mask": modality_mask,
            "quality": quality,
            "raw_refs": raw_refs,
            "features": features,
            "labels": empty_labels(),
        }

    def _write_export(
        self,
        records: list[dict[str, Any]],
        manifest: dict[str, Any],
        session_start: float,
        session_end: float,
        participants: list[str],
        group_id: str,
    ) -> ExportResult:
        os.makedirs(self.output_dir, exist_ok=True)
        windows_path = os.path.join(self.output_dir, "windows.jsonl")
        manifest_path = os.path.join(self.output_dir, "dataset_manifest.json")

        with open(windows_path, "w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        dataset_manifest = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "group_id": group_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "window_count": len(records),
            "session_start": session_start,
            "session_end": session_end,
            "participants": participants,
            "audio_scope": self.audio_scope,
            "state_taxonomy": list(GROUP_STATE_LABELS),
            "raw_manifest_session_id": manifest.get("session_id"),
        }
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(dataset_manifest, manifest_file, ensure_ascii=False, indent=2, sort_keys=True)

        return ExportResult(
            output_dir=self.output_dir,
            windows_path=windows_path,
            manifest_path=manifest_path,
            window_count=len(records),
        )


def export_group_dynamics_dataset(**kwargs) -> ExportResult:
    return GroupDynamicsExporter(**kwargs).export()


def _normalize_audio_scope(audio_scope: str) -> str:
    resolved = str(audio_scope or "group").strip().lower()
    if resolved not in {"participant", "group", "none"}:
        raise ValueError("audio_scope must be one of: participant, group, none.")
    return resolved


def _infer_group_id(session_id: str, manifest: dict[str, Any]) -> str:
    for key in ("group_id", "group"):
        value = manifest.get(key)
        if value:
            return str(value)
    marker = "_group_"
    if marker in session_id:
        suffix = session_id.split(marker, 1)[1]
        parts = suffix.split("_")
        if len(parts) >= 2:
            return f"group_{parts[0]}"
    return ""
