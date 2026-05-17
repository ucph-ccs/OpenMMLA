"""hybrid feature pipeline using centralized config-driven status rules."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from openmmla.analytics.realtime._parsing import coerce_json_dict, coerce_json_list
from openmmla.analytics.realtime.status_engine import (
    StatusEngine,
    load_runtime_analytics_config,
    load_status_engine_config,
    normalize_asr_scope,
    normalize_participant_aliases,
)
from openmmla.utils.client import InfluxDBClientWrapper
from openmmla.utils.constants import (
    EVENT_TYPE_ASR_RECOGNITION,
    EVENT_TYPE_ASR_TRANSCRIPTION,
    EVENT_TYPE_GROUP_INDICATORS,
    EVENT_TYPE_GROUP_SUMMARY,
    EVENT_TYPE_IPS_RELATION,
    EVENT_TYPE_IPS_ROTATION,
    EVENT_TYPE_IPS_TRANSLATION,
    EVENT_TYPE_PARTICIPANT_INDICATORS,
    EVENT_TYPE_PARTICIPANT_SUMMARY,
    EVENT_TYPE_VFA_ACTION,
)
from openmmla.utils.logger import get_logger

logger = get_logger("hybrid_feature_pipeline")

FEATURE_EVENT_TYPE_TO_BUCKET_KEY = {
    EVENT_TYPE_ASR_RECOGNITION: "speaker_recognition",
    EVENT_TYPE_ASR_TRANSCRIPTION: "speaker_transcription",
    EVENT_TYPE_IPS_TRANSLATION: "badge_translation",
    EVENT_TYPE_IPS_ROTATION: "badge_rotation",
    EVENT_TYPE_IPS_RELATION: "badge_relation",
    EVENT_TYPE_VFA_ACTION: "action_recognition",
}

FEATURE_EVENT_QUERY_ALIASES = {
    EVENT_TYPE_ASR_RECOGNITION: [EVENT_TYPE_ASR_RECOGNITION, "speaker_recognition"],
    EVENT_TYPE_ASR_TRANSCRIPTION: [EVENT_TYPE_ASR_TRANSCRIPTION, "speaker_transcription"],
    EVENT_TYPE_IPS_TRANSLATION: [EVENT_TYPE_IPS_TRANSLATION, "badge_translation"],
    EVENT_TYPE_IPS_ROTATION: [EVENT_TYPE_IPS_ROTATION, "badge_rotation"],
    EVENT_TYPE_IPS_RELATION: [EVENT_TYPE_IPS_RELATION, "badge_relation"],
    EVENT_TYPE_VFA_ACTION: [EVENT_TYPE_VFA_ACTION, "action_recognition"],
}


@dataclass(slots=True)
class HybridFeatureConfig:
    window_size: int
    step_size: int
    participant_ids: list[str]
    participant_aliases: dict[str, dict[str, str]]
    status_profile: str
    task_config_path: str
    asr_scope: str
    group_id: str


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _code_summary(codes: list[int]) -> dict[str, float | int]:
    total = len(codes)
    counter = Counter(codes)
    return {
        "count": total,
        "mean": _safe_mean([float(code) for code in codes]),
        "max": max(codes) if codes else 0,
        "ratio_0": counter.get(0, 0) / total if total else 0.0,
        "ratio_1": counter.get(1, 0) / total if total else 0.0,
        "ratio_2": counter.get(2, 0) / total if total else 0.0,
    }


class HybridFeaturePipeline:
    """build realtime/offline hybrid features using the new status engine."""

    def __init__(
        self,
        config_path: str,
        project_dir: str | None = None,
        window_size: int | None = None,
        step_size: int | None = None,
        participant_ids: list[str] | None = None,
        participant_aliases: dict[str, dict[str, Any]] | None = None,
        task_config: str | None = None,
        status_profile: str | None = None,
        experiment_id: str | None = None,
        group_id: str | None = None,
        asr_scope: str | None = None,
    ):
        project_dir = project_dir or os.getcwd()
        self.project_dir = project_dir
        self.config_path = config_path if os.path.isabs(config_path) else os.path.join(project_dir, config_path)
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        runtime_analytics = load_runtime_analytics_config(
            runtime_config_path=self.config_path,
            project_dir=project_dir,
            task_config=task_config,
            status_profile=status_profile,
            participant_ids=participant_ids,
            participant_aliases=participant_aliases,
            experiment_id=experiment_id,
            group_id=group_id,
            asr_scope=asr_scope,
        )
        self.runtime_analytics = runtime_analytics
        self.status_config = load_status_engine_config(project_dir, runtime_analytics)
        self.status_engine = StatusEngine(self.status_config)
        self.feature_config = HybridFeatureConfig(
            window_size=window_size or self._window_size_from_runtime(default=60),
            step_size=step_size or self._step_size_from_runtime(default=30),
            participant_ids=runtime_analytics.participant_ids,
            participant_aliases=runtime_analytics.participant_aliases,
            status_profile=runtime_analytics.status_profile,
            task_config_path=runtime_analytics.task_config_path,
            asr_scope=runtime_analytics.asr_scope,
            group_id=runtime_analytics.group_id,
        )

    def _runtime_config(self) -> dict[str, Any]:
        import yaml

        with open(self.config_path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file) or {}

    def _window_size_from_runtime(self, default: int) -> int:
        config = self._runtime_config()
        return int(config.get("Analytics", {}).get("window_size", config.get("Construct_Analysis", {}).get("context_window", default)))

    def _step_size_from_runtime(self, default: int) -> int:
        config = self._runtime_config()
        return int(config.get("Analytics", {}).get("step_size", config.get("Construct_Analysis", {}).get("step_size", default)))

    def collect_measurements(
        self,
        influx_client: InfluxDBClientWrapper,
        session_id: str,
        bucket_start: float,
        bucket_end: float,
    ) -> dict[str, list[dict[str, Any]]]:
        start_dt = datetime.fromtimestamp(bucket_start, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(bucket_end, tz=timezone.utc)
        bucket_data: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for event_type, aliases in FEATURE_EVENT_QUERY_ALIASES.items():
            bucket_key = FEATURE_EVENT_TYPE_TO_BUCKET_KEY[event_type]
            merged: list[dict[str, Any]] = []
            seen = set()
            for alias in aliases:
                events = influx_client.query_events(session_id, alias, start_time=start_dt, end_time=end_dt)
                for event in events:
                    identifier = (
                        event.get("window_start_time"),
                        event.get("window_end_time"),
                        event.get("speaker"),
                        event.get("text"),
                        event.get("time"),
                    )
                    if identifier in seen:
                        continue
                    seen.add(identifier)
                    merged.append(event)
            merged.sort(key=lambda item: item.get("window_start_time", 0.0))
            bucket_data[bucket_key] = merged

        return dict(bucket_data)

    def build_time_bucket(
        self,
        bucket_index: int,
        bucket_start: float,
        bucket_end: float,
        measurements: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        time_bucket: dict[str, Any] = {
            "bucket_index": bucket_index,
            "bucket_start": bucket_start,
            "bucket_end": bucket_end,
            "bucket_duration": bucket_end - bucket_start,
        }
        for bucket_key in FEATURE_EVENT_TYPE_TO_BUCKET_KEY.values():
            entries = measurements.get(bucket_key, [])
            time_bucket[bucket_key] = [
                entry for entry in entries
                if entry.get("window_end_time", bucket_end) >= bucket_start
                and entry.get("window_start_time", bucket_start) <= bucket_end
            ]
        return time_bucket

    def encode_status_vectors(self, time_bucket: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return self.status_engine.encode_bucket(
            time_bucket,
            self.feature_config.participant_ids,
            self.feature_config.participant_aliases,
            asr_scope=self.feature_config.asr_scope,
        )

    def encode_group_level_status(
        self,
        time_bucket: dict[str, Any],
        group_status: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if self.feature_config.asr_scope == "group":
            content_status = self.status_engine.encode_group_content_status(time_bucket)
        else:
            codes: list[int] = []
            metadata: list[str] = []
            for participant_id in self.feature_config.participant_ids:
                content_status = group_status.get(participant_id, {}).get("content_status", {})
                codes.extend(int(code) for code in content_status.get("code", []))
                metadata.extend(str(item) for item in content_status.get("metadata", []))
            content_status = {"code": codes, "metadata": metadata}
        return {"content_status": content_status}

    def _alias_lookup(self, alias_key: str) -> dict[str, str]:
        aliases = normalize_participant_aliases(
            self.feature_config.participant_ids,
            self.feature_config.participant_aliases,
        )
        return {alias_values[alias_key]: participant_id for participant_id, alias_values in aliases.items()}

    def build_hybrid_features(
        self,
        time_bucket: dict[str, Any],
        group_status: dict[str, dict[str, Any]],
        group_level_status: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        participant_features = {}
        asr_scope = normalize_asr_scope(self.feature_config.asr_scope)

        recognition_entries = time_bucket.get("speaker_recognition", [])
        transcription_entries = time_bucket.get("speaker_transcription", [])
        relation_entries = time_bucket.get("badge_relation", [])
        action_entries = time_bucket.get("action_recognition", [])
        tag_to_participant = self._alias_lookup("tag_id")

        turn_counts: Counter[str] = Counter()
        overlap_events = 0
        for entry in recognition_entries:
            speakers = [str(item) for item in coerce_json_list(entry.get("speakers", []))]
            active_speakers = [speaker for speaker in speakers if speaker and speaker.lower() != "silent"]
            if asr_scope == "participant" and len(active_speakers) > 1:
                overlap_events += 1
            for speaker in active_speakers:
                if asr_scope == "participant":
                    participant_id = tag_to_participant.get(speaker)
                    if participant_id:
                        turn_counts[participant_id] += 1

        utterance_counts: Counter[str] = Counter()
        utterance_word_counts: Counter[str] = Counter()
        if asr_scope == "participant":
            for entry in transcription_entries:
                speaker = str(entry.get("speaker", ""))
                participant_id = tag_to_participant.get(speaker)
                if not participant_id:
                    continue
                utterance_counts[participant_id] += 1
                utterance_word_counts[participant_id] += len(str(entry.get("text", "")).split())

        relation_edge_counts = []
        for entry in relation_entries:
            graph = coerce_json_dict(entry.get("graph", {}))
            relation_edge_counts.append(sum(len(targets) for targets in graph.values()))

        for participant_id in self.feature_config.participant_ids:
            participant_aliases = self.feature_config.participant_aliases[participant_id]
            tag_id = participant_aliases["tag_id"]
            status_bundle = group_status[participant_id]
            action_codes = status_bundle["action_status"]["code"]
            content_codes = status_bundle["content_status"]["code"]
            speaking_codes = status_bundle["speaking_status"]["code"]
            proximity_codes = status_bundle["proximity_status"]["code"]

            participant_action_labels = status_bundle["action_status"].get("metadata", [])
            dominant_action = Counter(participant_action_labels).most_common(1)

            speaking_measurements = 0
            similarity_values = []
            if asr_scope == "participant":
                for entry in recognition_entries:
                    speakers = [str(item) for item in coerce_json_list(entry.get("speakers", []))]
                    if tag_id in speakers:
                        speaking_measurements += 1
                        similarities = coerce_json_list(entry.get("similarities", []))
                        for speaker, similarity in zip(speakers, similarities):
                            if str(speaker) == tag_id and isinstance(similarity, (int, float)):
                                similarity_values.append(float(similarity))

            action_measurements = 0
            for entry in action_entries:
                action_recognition = coerce_json_dict(entry.get("action_recognition", {}))
                classifications = action_recognition.get("classifications", {})
                if tag_id in classifications:
                    action_measurements += 1

            participant_features[participant_id] = {
                "indicator_features": {
                    "action_status": _code_summary(action_codes),
                    "content_status": _code_summary(content_codes),
                    "speaking_status": _code_summary(speaking_codes),
                    "proximity_status": _code_summary(proximity_codes),
                },
                "measurement_summaries": {
                    "recognition_segments": speaking_measurements,
                    "avg_similarity": _safe_mean(similarity_values),
                    "utterance_count": utterance_counts.get(participant_id, 0),
                    "word_count": utterance_word_counts.get(participant_id, 0),
                    "relevant_utterance_ratio": (
                        sum(1 for code in content_codes if code == 2) / len(content_codes)
                        if content_codes else 0.0
                    ),
                    "in_zone_ratio": (
                        sum(1 for code in proximity_codes if code >= 1) / len(proximity_codes)
                        if proximity_codes else 0.0
                    ),
                    "close_to_peer_ratio": (
                        sum(1 for code in proximity_codes if code == 2) / len(proximity_codes)
                        if proximity_codes else 0.0
                    ),
                    "action_measurement_count": action_measurements,
                    "dominant_action": dominant_action[0][0] if dominant_action else "",
                    "asr_scope": asr_scope,
                    "content_status_available": asr_scope == "participant",
                },
            }

        total_turns = sum(turn_counts.values())
        speaking_balance = 0.0
        if total_turns and self.feature_config.participant_ids:
            expected = total_turns / max(len(self.feature_config.participant_ids), 1)
            speaking_balance = 1.0 - (
                sum(abs(turn_counts.get(pid, 0) - expected) for pid in self.feature_config.participant_ids)
                / (2 * total_turns)
            )

        group_content_status = group_level_status.get("content_status", {})
        group_content_codes = [int(code) for code in group_content_status.get("code", [])]
        group_content_metadata = group_content_status.get("metadata", [])
        if asr_scope == "group":
            group_utterance_count = len(group_content_codes)
            group_word_count = sum(
                len(str(item.get("text", "") if isinstance(item, dict) else item).split())
                for item in group_content_metadata
            )
        else:
            group_utterance_count = sum(utterance_counts.values())
            group_word_count = sum(utterance_word_counts.values())

        group_features = {
            "group_id": self.feature_config.group_id,
            "asr_scope": asr_scope,
            "participant_count": len(self.feature_config.participant_ids),
            "active_speaker_count": sum(1 for pid in self.feature_config.participant_ids if turn_counts.get(pid, 0) > 0),
            "total_turns": total_turns,
            "total_utterances": group_utterance_count,
            "total_words": group_word_count,
            "group_content_status": _code_summary(group_content_codes),
            "group_relevant_utterance_ratio": (
                sum(1 for code in group_content_codes if code == 2) / len(group_content_codes)
                if group_content_codes else 0.0
            ),
            "group_utterance_count": group_utterance_count,
            "group_word_count": group_word_count,
            "speaking_overlap_ratio": overlap_events / len(recognition_entries) if recognition_entries else 0.0,
            "speaking_balance": speaking_balance,
            "relation_edge_density": (
                _safe_mean([float(value) for value in relation_edge_counts])
                / max(len(self.feature_config.participant_ids) * max(len(self.feature_config.participant_ids) - 1, 1), 1)
                if relation_edge_counts else 0.0
            ),
            "measurement_counts": {
                bucket_key: len(time_bucket.get(bucket_key, []))
                for bucket_key in FEATURE_EVENT_TYPE_TO_BUCKET_KEY.values()
            },
        }

        return {
            "bucket_index": time_bucket["bucket_index"],
            "bucket_start": time_bucket["bucket_start"],
            "bucket_end": time_bucket["bucket_end"],
            "bucket_duration": time_bucket["bucket_duration"],
            "status_profile": self.feature_config.status_profile,
            "task_config_path": self.feature_config.task_config_path,
            "participants": self.feature_config.participant_aliases,
            "participant_features": participant_features,
            "group_features": group_features,
            "group_status": group_status,
            "group_level_status": group_level_status,
        }

    def generate_window_snapshot(
        self,
        influx_client: InfluxDBClientWrapper,
        session_id: str,
        bucket_start: float,
        bucket_end: float,
        bucket_index: int = 0,
    ) -> dict[str, Any]:
        measurements = self.collect_measurements(influx_client, session_id, bucket_start, bucket_end)
        time_bucket = self.build_time_bucket(bucket_index, bucket_start, bucket_end, measurements)
        group_status = self.encode_status_vectors(time_bucket)
        group_level_status = self.encode_group_level_status(time_bucket, group_status)
        return self.build_hybrid_features(time_bucket, group_status, group_level_status)


class RealtimeIndicatorEncoder:
    """poll InfluxDB, encode current indicators, and optionally write them back for dashboard use."""

    def __init__(
        self,
        config_path: str,
        session_id: str,
        project_dir: str | None = None,
        window_size: int | None = None,
        step_size: int | None = None,
        participant_ids: list[str] | None = None,
        participant_aliases: dict[str, dict[str, Any]] | None = None,
        task_config: str | None = None,
        status_profile: str | None = None,
        experiment_id: str | None = None,
        group_id: str | None = None,
        asr_scope: str | None = None,
        writeback: bool = True,
    ):
        self.project_dir = project_dir or os.getcwd()
        self.session_id = session_id
        self.writeback = writeback
        session_context = self._session_context(self.project_dir, config_path, session_id)
        experiment_id = experiment_id or session_context.get("experiment_id")
        group_id = group_id or session_context.get("group_id")
        session_participant_aliases = self._participant_aliases_from_session(session_context)
        participant_aliases = participant_aliases or session_participant_aliases or None
        participant_ids = participant_ids or list(session_participant_aliases.keys()) or None
        self.pipeline = HybridFeaturePipeline(
            config_path=config_path,
            project_dir=self.project_dir,
            window_size=window_size,
            step_size=step_size,
            participant_ids=participant_ids,
            participant_aliases=participant_aliases,
            task_config=task_config,
            status_profile=status_profile,
            experiment_id=experiment_id,
            group_id=group_id,
            asr_scope=asr_scope,
        )
        self.influx_client = InfluxDBClientWrapper(self.pipeline.config_path)
        self._last_written_window_end: float | None = None

    @staticmethod
    def _session_context(project_dir: str, config_path: str, session_id: str) -> dict[str, Any]:
        resolved_config_path = config_path if os.path.isabs(config_path) else os.path.join(project_dir, config_path)
        try:
            from openmmla.utils.client import MongoDBClientWrapper

            mongo_client = MongoDBClientWrapper(resolved_config_path)
            try:
                session_context = mongo_client.get_session(session_id) or {}
                if session_context:
                    return session_context
            finally:
                mongo_client.close()
        except Exception:
            pass
        return RealtimeIndicatorEncoder._session_context_from_experiments(project_dir, session_id)

    @staticmethod
    def _session_context_from_experiments(project_dir: str, session_id: str) -> dict[str, str]:
        try:
            from openmmla.utils.experiments import get_groups_for_experiment, load_experiments

            experiments = load_experiments(project_dir)
            for experiment in experiments.get("active_experiments", []):
                experiment_id = str(experiment.get("experiment_id", ""))
                if not experiment_id:
                    continue
                for group_id in get_groups_for_experiment(experiment_id, experiments):
                    if session_id.startswith(f"{experiment_id}_{group_id}_"):
                        return {"experiment_id": experiment_id, "group_id": group_id}
        except Exception:
            return {}
        return {}

    @staticmethod
    def _participant_aliases_from_session(session_context: dict[str, Any]) -> dict[str, dict[str, str]]:
        aliases: dict[str, dict[str, str]] = {}
        for participant in session_context.get("participants", []) or []:
            if not isinstance(participant, dict):
                continue
            participant_id = str(participant.get("participant_id") or participant.get("name") or "").strip()
            tag_id = str(participant.get("tag_id") or "").strip()
            if not participant_id or not tag_id:
                continue
            aliases[participant_id] = {
                "participant_id": participant_id,
                "tag_id": tag_id,
            }
            if participant.get("description"):
                aliases[participant_id]["description"] = str(participant["description"]).strip()
        return aliases

    def _next_window(self, end_time: float | None = None) -> tuple[float, float]:
        end_time = end_time or time.time()
        return end_time - self.pipeline.feature_config.window_size, end_time

    def encode_once(self, end_time: float | None = None) -> dict[str, Any]:
        bucket_start, bucket_end = self._next_window(end_time=end_time)
        snapshot = self.pipeline.generate_window_snapshot(
            self.influx_client,
            self.session_id,
            bucket_start=bucket_start,
            bucket_end=bucket_end,
            bucket_index=0,
        )
        if self.writeback:
            self.write_snapshot(snapshot)
        return snapshot

    def write_snapshot(self, snapshot: dict[str, Any]) -> None:
        bucket_start = snapshot["bucket_start"]
        bucket_end = snapshot["bucket_end"]
        if self._last_written_window_end is not None and bucket_end <= self._last_written_window_end:
            logger.info("Skipping already-written window ending at %.3f", bucket_end)
            return

        for participant_id, features in snapshot["participant_features"].items():
            status_bundle = snapshot["group_status"][participant_id]
            indicator_fields = {
                "window_start_time": bucket_start,
                "window_end_time": bucket_end,
                "participant_id": participant_id,
                "tag_id": self.pipeline.feature_config.participant_aliases[participant_id]["tag_id"],
                "status_profile": snapshot["status_profile"],
                "task_config_path": snapshot["task_config_path"],
                "asr_scope": self.pipeline.feature_config.asr_scope,
                "action_status": json.dumps(status_bundle["action_status"]["code"]),
                "content_status": json.dumps(status_bundle["content_status"]["code"]),
                "speaking_status": json.dumps(status_bundle["speaking_status"]["code"]),
                "proximity_status": json.dumps(status_bundle["proximity_status"]["code"]),
            }
            summary_fields = {
                "window_start_time": bucket_start,
                "window_end_time": bucket_end,
                "participant_id": participant_id,
                "participant": json.dumps(self.pipeline.feature_config.participant_aliases[participant_id]),
                "status_profile": snapshot["status_profile"],
                "task_config_path": snapshot["task_config_path"],
                "asr_scope": self.pipeline.feature_config.asr_scope,
                "features_json": json.dumps(features),
            }
            self.influx_client.write_event(self.session_id, EVENT_TYPE_PARTICIPANT_INDICATORS, indicator_fields)
            self.influx_client.write_event(self.session_id, EVENT_TYPE_PARTICIPANT_SUMMARY, summary_fields)

        group_indicator_fields = {
            "window_start_time": bucket_start,
            "window_end_time": bucket_end,
            "status_profile": snapshot["status_profile"],
            "task_config_path": snapshot["task_config_path"],
            "asr_scope": self.pipeline.feature_config.asr_scope,
            "group_status_json": json.dumps(snapshot["group_status"]),
            "group_level_status_json": json.dumps(snapshot.get("group_level_status", {})),
            "participant_ids": json.dumps(self.pipeline.feature_config.participant_ids),
            "participants": json.dumps(self.pipeline.feature_config.participant_aliases),
        }
        group_summary_fields = {
            "window_start_time": bucket_start,
            "window_end_time": bucket_end,
            "status_profile": snapshot["status_profile"],
            "task_config_path": snapshot["task_config_path"],
            "asr_scope": self.pipeline.feature_config.asr_scope,
            "group_features_json": json.dumps(snapshot["group_features"]),
        }
        self.influx_client.write_event(self.session_id, EVENT_TYPE_GROUP_INDICATORS, group_indicator_fields)
        self.influx_client.write_event(self.session_id, EVENT_TYPE_GROUP_SUMMARY, group_summary_fields)
        self._last_written_window_end = bucket_end

    def run_forever(self, poll_interval: float | None = None, iterations: int | None = None) -> None:
        poll_interval = poll_interval or self.pipeline.feature_config.step_size
        completed = 0
        while iterations is None or completed < iterations:
            self.encode_once()
            completed += 1
            time.sleep(poll_interval)
