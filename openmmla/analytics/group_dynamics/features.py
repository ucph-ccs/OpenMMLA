"""compact group-context features for OpenMMLA-GD."""

from __future__ import annotations

import math
from itertools import combinations
from statistics import mean
from typing import Any

from openmmla.utils.constants import (
    EVENT_TYPE_ASR_TRANSCRIPTION,
    EVENT_TYPE_IPS_RELATION,
    EVENT_TYPE_IPS_TRANSLATION,
    EVENT_TYPE_VFA_ACTION,
)


def build_group_dynamics_features(
    window_events: dict[str, list[dict[str, Any]]],
    raw_refs: dict[str, list[dict[str, Any]]],
    participants: list[str],
    audio_scope: str = "group",
) -> tuple[dict[str, Any], dict[str, bool], dict[str, float]]:
    geometry, geometry_quality = extract_geometry_features(window_events, participants)
    pose_video, pose_video_quality = extract_pose_video_features(window_events, raw_refs)
    audio_text, audio_text_quality = extract_audio_text_features(window_events, raw_refs, audio_scope)

    modality_mask = {
        "audio": bool(raw_refs.get("audio")),
        "transcript": bool(window_events.get(EVENT_TYPE_ASR_TRANSCRIPTION)),
        "video": bool(raw_refs.get("video") or window_events.get(EVENT_TYPE_VFA_ACTION)),
        "pose": bool(pose_video.get("pose_available")),
        "trajectory": bool(window_events.get(EVENT_TYPE_IPS_TRANSLATION)),
        "proximity": bool(window_events.get(EVENT_TYPE_IPS_RELATION) or window_events.get(EVENT_TYPE_IPS_TRANSLATION)),
    }
    quality = {
        "geometry": geometry_quality,
        "pose_video": pose_video_quality,
        "audio_text": audio_text_quality,
    }
    features = {
        "geometry": geometry,
        "pose_video": pose_video,
        "audio_text": audio_text,
    }
    return features, modality_mask, quality


def extract_geometry_features(window_events: dict[str, list[dict[str, Any]]], participants: list[str]) -> tuple[dict[str, Any], float]:
    translation_samples = [
        _parse_positions(event.get("translations", {}))
        for event in window_events.get(EVENT_TYPE_IPS_TRANSLATION, [])
    ]
    translation_samples = [sample for sample in translation_samples if sample]
    relation_graphs = [
        event.get("graph", {})
        for event in window_events.get(EVENT_TYPE_IPS_RELATION, [])
        if isinstance(event.get("graph", {}), dict)
    ]

    pairwise_distances = []
    spreads = []
    centroids = []
    cluster_counts = []
    for positions in translation_samples:
        distances = _pairwise_distances(positions)
        pairwise_distances.extend(distances)
        spreads.append(_group_spread(positions))
        centroids.append(_centroid(positions))
        cluster_counts.append(_cluster_count(positions, distance_threshold=1.2))

    centroid_speeds = []
    approach_rates = []
    for previous, current in zip(translation_samples, translation_samples[1:]):
        previous_centroid = _centroid(previous)
        current_centroid = _centroid(current)
        centroid_speeds.append(_distance(previous_centroid, current_centroid))
        approach_rates.extend(_pairwise_approach_rates(previous, current))

    edge_densities = [_graph_density(graph, participants) for graph in relation_graphs]
    edge_density_delta = abs(edge_densities[-1] - edge_densities[0]) if len(edge_densities) >= 2 else 0.0

    features = {
        "sample_count": len(translation_samples),
        "participant_count": len(participants),
        "mean_pairwise_distance": _safe_mean(pairwise_distances),
        "min_pairwise_distance": min(pairwise_distances) if pairwise_distances else 0.0,
        "max_pairwise_distance": max(pairwise_distances) if pairwise_distances else 0.0,
        "mean_group_spread": _safe_mean(spreads),
        "centroid_movement": sum(centroid_speeds),
        "mean_centroid_speed": _safe_mean(centroid_speeds),
        "mean_approach_rate": _safe_mean(approach_rates),
        "cluster_count": _safe_mean([float(value) for value in cluster_counts]),
        "graph_density": _safe_mean(edge_densities),
        "graph_density_change": edge_density_delta,
        "relation_event_count": len(relation_graphs),
    }
    quality = min(1.0, len(translation_samples) / 3.0) if translation_samples else 0.0
    return features, quality


def extract_pose_video_features(window_events: dict[str, list[dict[str, Any]]], raw_refs: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], float]:
    vfa_events = window_events.get(EVENT_TYPE_VFA_ACTION, [])
    video_refs = raw_refs.get("video", [])
    features = {
        "video_ref_count": len(video_refs),
        "vfa_event_count": len(vfa_events),
        "pose_available": False,
        "visible_participant_count": 0,
        "frozen_embedding_available": False,
    }
    quality = 1.0 if video_refs else min(1.0, len(vfa_events) / 2.0)
    return features, quality


def extract_audio_text_features(
    window_events: dict[str, list[dict[str, Any]]],
    raw_refs: dict[str, list[dict[str, Any]]],
    audio_scope: str = "group",
) -> tuple[dict[str, Any], float]:
    transcriptions = window_events.get(EVENT_TYPE_ASR_TRANSCRIPTION, [])
    texts = [str(event.get("text") or "").strip() for event in transcriptions]
    texts = [text for text in texts if text]
    word_counts = [len(text.split()) for text in texts]
    features = {
        "audio_scope": audio_scope,
        "audio_ref_count": len(raw_refs.get("audio", [])),
        "utterance_count": len(texts),
        "word_count": sum(word_counts),
        "avg_words_per_utterance": _safe_mean([float(value) for value in word_counts]),
        "transcript_char_count": sum(len(text) for text in texts),
        "transcript_embedding_available": False,
    }
    if texts:
        quality = min(1.0, len(texts) / 3.0)
    elif raw_refs.get("audio"):
        quality = 0.25
    else:
        quality = 0.0
    return features, quality


def _parse_positions(raw_translations: Any) -> dict[str, tuple[float, float, float]]:
    positions = {}
    if not isinstance(raw_translations, dict):
        return positions
    for participant_id, coords in raw_translations.items():
        point = _coerce_point(coords)
        if point is not None:
            positions[str(participant_id)] = point
    return positions


def _coerce_point(coords: Any) -> tuple[float, float, float] | None:
    try:
        x = _nested_float(coords, 0)
        y = _nested_float(coords, 1)
        z = _nested_float(coords, 2)
        return x, y, z
    except (TypeError, ValueError, IndexError):
        return None


def _nested_float(coords: Any, index: int) -> float:
    value = coords[index]
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return float(value)


def _pairwise_distances(positions: dict[str, tuple[float, float, float]]) -> list[float]:
    return [
        _distance(positions[left], positions[right])
        for left, right in combinations(sorted(positions), 2)
    ]


def _pairwise_approach_rates(
    previous: dict[str, tuple[float, float, float]],
    current: dict[str, tuple[float, float, float]],
) -> list[float]:
    rates = []
    shared = sorted(set(previous).intersection(current))
    for left, right in combinations(shared, 2):
        previous_distance = _distance(previous[left], previous[right])
        current_distance = _distance(current[left], current[right])
        rates.append(previous_distance - current_distance)
    return rates


def _group_spread(positions: dict[str, tuple[float, float, float]]) -> float:
    if not positions:
        return 0.0
    centroid = _centroid(positions)
    return _safe_mean([_distance(point, centroid) for point in positions.values()])


def _centroid(positions: dict[str, tuple[float, float, float]]) -> tuple[float, float, float]:
    if not positions:
        return 0.0, 0.0, 0.0
    return (
        mean(point[0] for point in positions.values()),
        mean(point[1] for point in positions.values()),
        mean(point[2] for point in positions.values()),
    )


def _cluster_count(positions: dict[str, tuple[float, float, float]], distance_threshold: float) -> int:
    nodes = set(positions)
    clusters = 0
    while nodes:
        clusters += 1
        stack = [nodes.pop()]
        while stack:
            node = stack.pop()
            linked = {
                other for other in list(nodes)
                if _distance(positions[node], positions[other]) <= distance_threshold
            }
            nodes.difference_update(linked)
            stack.extend(linked)
    return clusters


def _graph_density(graph: dict[str, Any], participants: list[str]) -> float:
    nodes = set(participants)
    nodes.update(str(node) for node in graph)
    for targets in graph.values():
        if isinstance(targets, list):
            nodes.update(str(target) for target in targets)
    possible_edges = len(nodes) * max(len(nodes) - 1, 0)
    if possible_edges == 0:
        return 0.0
    edge_count = 0
    for targets in graph.values():
        if isinstance(targets, list):
            edge_count += len({str(target) for target in targets})
    return edge_count / possible_edges


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(
        (left[0] - right[0]) ** 2
        + (left[1] - right[1]) ** 2
        + (left[2] - right[2]) ** 2
    )


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0
