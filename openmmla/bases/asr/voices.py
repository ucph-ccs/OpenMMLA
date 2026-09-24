"""Session-wide voices from the anonymous speakers of diarized chunks.

The speech transcriber diarizes each chunk on its own, so the SPEAKER_00 of one chunk need not be
the SPEAKER_00 of the next. With the turns it returns one speaker embedding per SPEAKER_NN (the
diarization pipeline's centroid of that speaker in the chunk). A base keeps a VoiceRegistry for its
session: each chunk's speakers are matched to the voices it has heard so far by the cosine
similarity of their embeddings to the voices' running centroids, one speaker to one voice, the
most similar pairs first, and only at VOICE_LINK_THRESHOLD or above; a speaker left over starts a
new voice when it spoke at least VOICE_MIN_SECONDS in the chunk, and has no voice otherwise. Only
chunks already transcribed are read, so a voice is the same live and in replay.

Voices are numbered 1, 2, 3 ... in the order they were first heard, per base and session. The base
keeps its registry through a run restarted after a recording error; a base launched again into the
session starts a new one, and its transcripts name the registry (voice_registry) so that readers
keep the two numberings apart.

The threshold comes from unlabelled chunks (docs/pipelines/asr.md): 30 s of a group microphone
diarized whole say which of its speakers are one and which are two, and its two 15 s halves,
diarized apart, are two consecutive chunks. On the centroids the pipeline returns for such halves
(the vectors this registry compares), one speaker's two halves and two speakers' halves meet at an
equal error rate of 24 % at a similarity of 0.355; a registry run over the halves of six lessons in
order gives one speaker's two halves two voices and two speakers one voice about equally often
(25 and 24 %) at 0.35.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

VOICE_LINK_THRESHOLD = 0.35  # cosine similarity at which a chunk's speaker is a voice already heard
VOICE_MIN_SECONDS = 1.0  # the speech a speaker needs in a chunk to start a voice of its own


def unit_vector(vector) -> np.ndarray | None:
    """`vector` scaled to length 1, or None when it is empty, not numbers, not finite or all zero
    (a speaker the pipeline has no centroid for)."""
    try:
        array = np.asarray(vector, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else None


def speaker_seconds(turns) -> dict[str, float]:
    """how long each speaker of a chunk's turns ({start, end, speaker}) spoke, in seconds."""
    seconds: dict[str, float] = {}
    for turn in turns or []:
        try:
            length = float(turn['end']) - float(turn['start'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(length) and length > 0:
            label = str(turn.get('speaker'))
            seconds[label] = seconds.get(label, 0.0) + length
    return seconds


class VoiceRegistry:
    """the voices one base has heard in a session: per voice, the sum of its speakers' unit
    embeddings weighted by their seconds of speech, and that weight."""

    def __init__(self, threshold: float = VOICE_LINK_THRESHOLD, min_seconds: float = VOICE_MIN_SECONDS):
        self.threshold = float(threshold)
        self.min_seconds = float(min_seconds)
        self._sums: list[np.ndarray] = []
        self._weights: list[float] = []

    def __len__(self) -> int:
        return len(self._sums)

    def centroids(self) -> np.ndarray:
        """the voices' centroids as unit vectors, one row per voice (voice n is row n - 1)."""
        if not self._sums:
            return np.zeros((0, 0))
        sums = np.stack(self._sums)
        return sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)

    def link(self, embeddings: dict | None, seconds: dict | None = None) -> dict[str, dict[str, Any]]:
        """the session voice of each speaker of one chunk: {speaker: {'voice': n, 'similarity': s}},
        s the cosine similarity to the voice's centroid before this chunk, None for a voice this
        chunk started. `embeddings` is {speaker: vector}, `seconds` {speaker: seconds of speech}
        (speaker_seconds of the chunk's turns). Two speakers of one chunk are never one voice; a
        speaker without a usable embedding, or matched to no voice with too little speech to start
        one, is left out. The voices' centroids take in this chunk's speakers once all are decided."""
        seconds = seconds or {}
        units = {str(label): unit for label, unit in
                 ((label, unit_vector(vector)) for label, vector in (embeddings or {}).items()) if unit is not None}
        if not units:
            return {}
        labels = sorted(units)
        links: dict[str, dict[str, Any]] = {}
        centroids = self.centroids()
        if len(centroids) and centroids.shape[1] == len(units[labels[0]]):
            similarity = np.stack([units[label] for label in labels]) @ centroids.T
            pairs = sorted(((float(similarity[i, j]), i, j) for i in range(len(labels)) for j in range(len(centroids))),
                           key=lambda pair: (-pair[0], pair[1], pair[2]))
            used_voices: set[int] = set()
            for value, i, j in pairs:
                if value < self.threshold:
                    break
                if labels[i] in links or j in used_voices:
                    continue
                links[labels[i]] = {'voice': j + 1, 'similarity': round(value, 3)}
                used_voices.add(j)
        dimension = len(units[labels[0]])
        for label in labels:
            if label in links or float(seconds.get(label, 0.0)) < self.min_seconds:
                continue
            if self._sums and len(self._sums[0]) != dimension:
                continue  # an embedding of another model: it cannot be one of these voices
            self._sums.append(np.zeros(dimension))
            self._weights.append(0.0)
            links[label] = {'voice': len(self._sums), 'similarity': None}
        for label, link in links.items():
            weight = max(float(seconds.get(label, 0.0)), 0.1)
            self._sums[link['voice'] - 1] = self._sums[link['voice'] - 1] + weight * units[label]
            self._weights[link['voice'] - 1] += weight
        return links


def with_voices(items, links: dict) -> list:
    """copies of `items` (a chunk's turns or words) with the session voice of their speaker as
    `voice`, when it has one; the rest unchanged."""
    out = []
    for item in items or []:
        if isinstance(item, dict) and str(item.get('speaker')) in links and item.get('speaker') is not None:
            item = dict(item, voice=links[str(item['speaker'])]['voice'])
        out.append(item)
    return out
