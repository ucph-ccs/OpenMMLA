"""shared schema constants for OpenMMLA-GD datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SCHEMA_VERSION = "openmmla-gd.v1"

GROUP_STATE_LABELS = (
    "forming",
    "dispersing",
    "communicating",
    "co_working",
    "approaching_or_merging",
    "idle_or_off_task",
)

EMPTY_LABELS: dict[str, Any] = {
    "human": None,
    "teacher": None,
    "final": None,
    "confidence": None,
}

DEFAULT_MODALITY_MASK = {
    "audio": False,
    "transcript": False,
    "video": False,
    "pose": False,
    "trajectory": False,
    "proximity": False,
}

DEFAULT_QUALITY = {
    "geometry": 0.0,
    "pose_video": 0.0,
    "audio_text": 0.0,
}


def empty_labels() -> dict[str, Any]:
    return deepcopy(EMPTY_LABELS)


def default_modality_mask() -> dict[str, bool]:
    return deepcopy(DEFAULT_MODALITY_MASK)


def default_quality() -> dict[str, float]:
    return deepcopy(DEFAULT_QUALITY)
