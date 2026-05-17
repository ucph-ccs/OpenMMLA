"""new centralized status engine for IMWUT-oriented multimodal inference."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import yaml

from openmmla.analytics.realtime._parsing import coerce_json_dict, coerce_json_list


@dataclass(slots=True)
class RuntimeAnalyticsConfig:
    participant_ids: list[str]
    participant_aliases: dict[str, dict[str, str]]
    task_config_path: str
    status_profile: str
    asr_scope: str
    group_id: str


@dataclass(slots=True)
class StatusEngineConfig:
    profile_name: str
    speaking: dict[str, Any]
    proximity: dict[str, Any]
    content: dict[str, Any]
    action_map: dict[str, int]
    task_semantics: dict[str, Any]


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve(project_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(project_dir, path)


def normalize_participant_aliases(
    participant_ids: list[str],
    participant_aliases: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, str]]:
    """normalize participant ids to the shared tag_id used by ASR, IPS, and VFA."""
    participant_aliases = participant_aliases or {}
    normalized: dict[str, dict[str, str]] = {}
    for participant_id in participant_ids:
        pid = str(participant_id)
        aliases = participant_aliases.get(pid, {})
        tag_id = str(aliases.get("tag_id") or pid)
        normalized[pid] = {
            "participant_id": pid,
            "tag_id": tag_id,
        }
        if aliases.get("description"):
            normalized[pid]["description"] = str(aliases["description"])
    return normalized


def normalize_asr_scope(asr_scope: str | None = None) -> str:
    resolved_scope = str(asr_scope or "participant").strip().lower()
    if resolved_scope not in {"participant", "group"}:
        raise ValueError("Analytics.asr_scope must be 'participant' or 'group'.")
    return resolved_scope


def _analytics_participant_aliases(analytics_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    aliases: dict[str, dict[str, Any]] = {}

    for item in analytics_config.get("participants", []) or []:
        if not isinstance(item, dict):
            continue
        participant_id = item.get("participant_id") or item.get("name") or item.get("id")
        if participant_id is None:
            continue
        aliases[str(participant_id)] = dict(item)

    configured_aliases = analytics_config.get("participant_aliases", {}) or {}
    if isinstance(configured_aliases, dict):
        for participant_id, alias_config in configured_aliases.items():
            if isinstance(alias_config, dict):
                aliases[str(participant_id)] = {**aliases.get(str(participant_id), {}), **alias_config}

    return aliases


def _experiment_participant_aliases(project_dir: str, experiment_id: str | None, group_id: str | None) -> dict[str, dict[str, str]]:
    if not experiment_id or not group_id:
        return {}
    try:
        from openmmla.utils.experiments import get_participant_aliases, load_experiments

        return get_participant_aliases(experiment_id, group_id, data=load_experiments(project_dir))
    except Exception:
        return {}


def load_runtime_analytics_config(
    runtime_config_path: str,
    project_dir: str,
    task_config: str | None = None,
    status_profile: str | None = None,
    participant_ids: list[str] | None = None,
    participant_aliases: dict[str, dict[str, Any]] | None = None,
    experiment_id: str | None = None,
    group_id: str | None = None,
    asr_scope: str | None = None,
) -> RuntimeAnalyticsConfig:
    runtime_config = _load_yaml(runtime_config_path)
    analytics_config = runtime_config.get("Analytics", {})
    construct_analysis = runtime_config.get("Construct_Analysis", {})
    resolved_experiment_id = experiment_id or analytics_config.get("experiment_id")
    resolved_group_id = group_id or analytics_config.get("group_id")
    aliases = {
        **_experiment_participant_aliases(project_dir, resolved_experiment_id, resolved_group_id),
        **_analytics_participant_aliases(analytics_config),
        **(participant_aliases or {}),
    }

    resolved_task_config = (
        task_config
        or analytics_config.get("task_config")
        or "config/tasks/programming.yaml"
    )
    resolved_status_profile = (
        status_profile
        or analytics_config.get("status_profile")
        or "imwut_v1"
    )
    resolved_asr_scope = normalize_asr_scope(asr_scope or analytics_config.get("asr_scope"))
    resolved_participants = (
        participant_ids
        or analytics_config.get("participant_ids")
        or construct_analysis.get("participant_ids")
        or list(aliases.keys())
        or []
    )
    resolved_participant_ids = [str(item) for item in resolved_participants]

    return RuntimeAnalyticsConfig(
        participant_ids=resolved_participant_ids,
        participant_aliases=normalize_participant_aliases(resolved_participant_ids, aliases),
        task_config_path=_resolve(project_dir, resolved_task_config),
        status_profile=resolved_status_profile,
        asr_scope=resolved_asr_scope,
        group_id=str(resolved_group_id or ""),
    )


def load_status_engine_config(
    project_dir: str,
    runtime_analytics: RuntimeAnalyticsConfig,
) -> StatusEngineConfig:
    profiles_path = os.path.join(project_dir, "config", "analytics", "status_profiles.yml")
    action_taxonomy_path = os.path.join(project_dir, "config", "analytics", "action_taxonomy.yml")

    profiles_config = _load_yaml(profiles_path)
    action_taxonomy = _load_yaml(action_taxonomy_path).get("maps", {})

    profile_name = runtime_analytics.status_profile or profiles_config.get("default_profile")
    profiles = profiles_config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown status profile: {profile_name}")

    profile = profiles[profile_name]
    action_map_ref = profile.get("action", {}).get("action_map_ref", "default_v1")
    action_map = action_taxonomy.get(action_map_ref)
    if action_map is None:
        raise ValueError(f"Unknown action map reference: {action_map_ref}")

    task_semantics = _load_yaml(runtime_analytics.task_config_path)
    return StatusEngineConfig(
        profile_name=profile_name,
        speaking=profile.get("speaking", {}),
        proximity=profile.get("proximity", {}),
        content=profile.get("content", {}),
        action_map=action_map,
        task_semantics=task_semantics,
    )


class StatusEngine:
    """encode participant-level statuses from a framework-compatible time bucket."""

    def __init__(self, config: StatusEngineConfig):
        self.config = config
        semantic_rules = config.task_semantics.get("semantic_mapping_rules", {})
        audio_keyword_rules = semantic_rules.get("audio_keywords", [])
        self.relevant_keywords = sorted(
            {
                keyword.lower()
                for rule in audio_keyword_rules
                for keyword in rule.get("keywords", [])
                if isinstance(keyword, str)
            }
        )
        self.off_topic_keywords = [keyword.lower() for keyword in config.content.get("off_topic_keywords", [])]
        self.filler_words = [word.lower() for word in config.content.get("filler_words", [])]
        self.transcription_artifacts = [re.compile(pattern, re.IGNORECASE) for pattern in config.content.get("transcription_artifacts", [])]

    def encode_bucket(
        self,
        time_bucket: dict[str, Any],
        participant_ids: list[str],
        participant_aliases: dict[str, dict[str, str]] | None = None,
        asr_scope: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        aliases = normalize_participant_aliases(participant_ids, participant_aliases)
        resolved_asr_scope = normalize_asr_scope(asr_scope)
        return {
            participant_id: {
                "action_status": self.encode_action_status(time_bucket, participant_id, aliases),
                "content_status": (
                    self.encode_content_status(time_bucket, participant_id, aliases)
                    if resolved_asr_scope == "participant"
                    else {"code": [], "metadata": []}
                ),
                "speaking_status": (
                    self.encode_speaking_status(time_bucket, participant_id, aliases)
                    if resolved_asr_scope == "participant"
                    else {"code": [], "metadata": []}
                ),
                "proximity_status": self.encode_proximity_status(time_bucket, participant_id, aliases),
            }
            for participant_id in participant_ids
        }

    def encode_action_status(
        self,
        time_bucket: dict[str, Any],
        participant_id: str,
        participant_aliases: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        default_code = int(self.config.action_map.get("default", 0))
        aliases = normalize_participant_aliases([participant_id], participant_aliases)
        tag_id = aliases[participant_id]["tag_id"]
        codes: list[int] = []
        labels: list[str] = []
        for entry in time_bucket.get("action_recognition", []):
            payload = coerce_json_dict(entry.get("action_recognition", {}))
            classifications = payload.get("classifications", {})
            if tag_id not in classifications:
                continue
            label = str(classifications[tag_id])
            labels.append(label)
            codes.append(int(self.config.action_map.get(label, default_code)))
        return {"code": codes, "metadata": labels}

    def encode_content_status(
        self,
        time_bucket: dict[str, Any],
        participant_id: str,
        participant_aliases: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        aliases = normalize_participant_aliases([participant_id], participant_aliases)
        tag_id = aliases[participant_id]["tag_id"]

        codes: list[int] = []
        texts: list[str] = []
        for entry in time_bucket.get("speaker_transcription", []):
            if str(entry.get("speaker", "")) != tag_id:
                continue
            text = str(entry.get("text", "")).strip()
            texts.append(text)
            codes.append(self._encode_transcript_text(text))

        return {"code": codes, "metadata": texts}

    def encode_group_content_status(self, time_bucket: dict[str, Any]) -> dict[str, Any]:
        codes: list[int] = []
        metadata: list[dict[str, str]] = []
        for entry in time_bucket.get("speaker_transcription", []):
            text = str(entry.get("text", "")).strip()
            speaker = str(entry.get("speaker", "")).strip()
            codes.append(self._encode_transcript_text(text))
            metadata.append({"speaker": speaker, "text": text})
        return {"code": codes, "metadata": metadata}

    def encode_speaking_status(
        self,
        time_bucket: dict[str, Any],
        participant_id: str,
        participant_aliases: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        inactive_code = int(self.config.speaking.get("inactive_code", 0))
        speaking_code = int(self.config.speaking.get("speaking_code", 1))
        focused_code = int(self.config.speaking.get("focused_code", 2))
        silent_labels = {str(item) for item in self.config.speaking.get("silent_labels", ["silent"])}
        time_margin = float(self.config.speaking.get("facing_time_margin_seconds", 5.0))
        aliases = normalize_participant_aliases([participant_id], participant_aliases)
        tag_id = aliases[participant_id]["tag_id"]

        codes: list[int] = []
        speakers_metadata: list[list[str]] = []
        for entry in time_bucket.get("speaker_recognition", []):
            speakers = [str(item) for item in coerce_json_list(entry.get("speakers", []))]
            speakers_metadata.append(speakers)
            if tag_id not in speakers or tag_id in silent_labels:
                codes.append(inactive_code)
                continue

            relation = self._closest_relation_entry(time_bucket.get("badge_relation", []), entry.get("window_start_time", 0.0), time_margin)
            if relation and relation.get(tag_id):
                codes.append(focused_code)
            else:
                codes.append(speaking_code)

        return {"code": codes, "metadata": speakers_metadata}

    def encode_proximity_status(
        self,
        time_bucket: dict[str, Any],
        participant_id: str,
        participant_aliases: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        zone_center = self.config.proximity.get("zone_center", [0.0, 0.0, 0.0])
        zone_radius = float(self.config.proximity.get("zone_radius", 4.0))
        close_distance = float(self.config.proximity.get("close_distance", 1.0))
        time_margin = float(self.config.proximity.get("time_margin_seconds", 2.0))
        close_code = int(self.config.proximity.get("close_code", 2))
        in_zone_code = int(self.config.proximity.get("in_zone_code", 1))
        out_code = int(self.config.proximity.get("out_of_zone_code", 0))

        codes: list[int] = []
        metadata: list[dict[str, Any]] = []
        translations_data = time_bucket.get("badge_translation", [])
        aliases = normalize_participant_aliases([participant_id], participant_aliases)
        tag_id = aliases[participant_id]["tag_id"]

        for entry in translations_data:
            translations = coerce_json_dict(entry.get("translations", {}))
            if tag_id not in translations:
                codes.append(out_code)
                metadata.append({"position": None, "closest_participant": None, "min_distance": None})
                continue

            position = self._flatten_position(translations.get(tag_id))
            if position is None:
                codes.append(out_code)
                metadata.append({"position": None, "closest_participant": None, "min_distance": None})
                continue

            timestamp = float(entry.get("window_start_time", 0.0))
            distance_to_zone = self._distance(position, zone_center)
            inside_zone = distance_to_zone <= zone_radius

            closest_participant = None
            min_distance = float("inf")
            for other_id in self._discover_participant_ids(translations_data):
                if other_id == tag_id:
                    continue
                other_position = self._closest_position(translations_data, other_id, timestamp, time_margin)
                if other_position is None:
                    continue
                distance = self._distance(position, other_position)
                if distance < min_distance:
                    min_distance = distance
                    closest_participant = other_id

            if inside_zone and min_distance <= close_distance:
                codes.append(close_code)
            elif inside_zone:
                codes.append(in_zone_code)
            else:
                codes.append(out_code)

            metadata.append(
                {
                    "position": position,
                    "closest_participant": closest_participant,
                    "min_distance": None if min_distance == float("inf") else min_distance,
                    "distance_to_zone": distance_to_zone,
                }
            )

        return {"code": codes, "metadata": metadata}

    def _closest_relation_entry(self, relation_entries: list[dict[str, Any]], timestamp: float, time_margin: float) -> dict[str, Any] | None:
        best_graph = None
        min_delta = float("inf")
        for entry in relation_entries:
            delta = abs(float(entry.get("window_start_time", 0.0)) - timestamp)
            if delta > time_margin or delta >= min_delta:
                continue
            best_graph = coerce_json_dict(entry.get("graph", {}))
            min_delta = delta
        return best_graph

    def _discover_participant_ids(self, translations_data: list[dict[str, Any]]) -> set[str]:
        participant_ids = set()
        for entry in translations_data:
            participant_ids.update(str(key) for key in coerce_json_dict(entry.get("translations", {})).keys())
        return participant_ids

    def _closest_position(
        self,
        translations_data: list[dict[str, Any]],
        participant_id: str,
        target_timestamp: float,
        time_margin: float,
    ) -> list[float] | None:
        best_position = None
        min_delta = float("inf")
        for entry in translations_data:
            delta = abs(float(entry.get("window_start_time", 0.0)) - target_timestamp)
            if delta > time_margin or delta >= min_delta:
                continue
            translations = coerce_json_dict(entry.get("translations", {}))
            if participant_id not in translations:
                continue
            position = self._flatten_position(translations.get(participant_id))
            if position is None:
                continue
            best_position = position
            min_delta = delta
        return best_position

    def _looks_invalid_transcript(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 2:
            return True
        for pattern in self.transcription_artifacts:
            if pattern.match(stripped):
                return True
        cleaned = stripped.lower()
        for filler in self.filler_words:
            cleaned = cleaned.replace(filler, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return len(cleaned) < 2

    def _encode_transcript_text(self, text: str) -> int:
        invalid_code = int(self.config.content.get("invalid_code", 0))
        off_topic_code = int(self.config.content.get("off_topic_code", 1))
        relevant_code = int(self.config.content.get("task_relevant_code", 2))
        min_meaningful_words = int(self.config.content.get("min_meaningful_words", 3))

        if self._looks_invalid_transcript(text):
            return invalid_code

        normalized = text.lower()
        words = [word for word in re.findall(r"\b\w+\b", normalized) if word]
        content_words = [word for word in words if word not in self.filler_words]
        if len(content_words) < min_meaningful_words:
            return off_topic_code if content_words else invalid_code

        relevant_hits = sum(1 for keyword in self.relevant_keywords if keyword in normalized)
        off_topic_hits = sum(1 for keyword in self.off_topic_keywords if keyword in normalized)
        return relevant_code if relevant_hits > off_topic_hits and relevant_hits > 0 else off_topic_code

    @staticmethod
    def _flatten_position(value: Any) -> list[float] | None:
        if not isinstance(value, list) or len(value) < 3:
            return None
        flattened = []
        for coordinate in value[:3]:
            if isinstance(coordinate, list):
                if not coordinate or not isinstance(coordinate[0], (int, float)):
                    return None
                flattened.append(float(coordinate[0]))
            elif isinstance(coordinate, (int, float)):
                flattened.append(float(coordinate))
            else:
                return None
        return flattened

    @staticmethod
    def _distance(pos1: list[float], pos2: list[float]) -> float:
        return sum((float(a) - float(b)) ** 2 for a, b in zip(pos1[:3], pos2[:3])) ** 0.5
