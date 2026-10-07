"""Session-wide voices from the anonymous speakers of diarized chunks.

The speech transcriber diarizes each chunk on its own, so the SPEAKER_00 of one chunk need not be
the SPEAKER_00 of the next. With the turns it returns one speaker embedding per SPEAKER_NN, and
names their kind (speaker_embedding_kind). Of kind `speech` (SPEECH_EMBEDDING), each is the
embedding of that speaker's own speech in the chunk: its turns minus the moments another speaker
talks too, put one after the other and embedded whole by the diarization pipeline's WeSpeaker
ResNet34 (256 numbers; openmmla.utils.audio.transcriber.speaker_speech_embeddings). A service that
names no kind returns pyannote's clustering centroids (CENTROID_EMBEDDING), which depend on the
length of the chunk: pyannote embeds 10 s windows, pads a chunk shorter than that with zeros, and
WeSpeaker subtracts the mean of the features over the whole window, padding included. One speaker's
centroids of chunks under 10 s and over then hardly resemble each other, so its short chunks became
one voice and its long ones another. The speech embeddings no longer part by chunk length, but how
well one speaker's embeddings match still grows with the seconds of speech embedded.

A base keeps a VoiceRegistry for its session: each chunk's speakers are matched to the voices it
has heard so far by the cosine similarity of their embeddings to the voices' running centroids, one
speaker to one voice, the most similar pairs first, and only at VOICE_LINK_THRESHOLD or above. A
speaker left over starts a new voice when it spoke at least VOICE_MIN_SECONDS in the chunk with
nobody else talking (speaker_alone_seconds), and has no voice otherwise: a speaker heard mostly over
another is embedded from a mix of the two, which may join a voice but must not start one. A registry
holds embeddings of one kind and one length, those of its first voice: a chunk whose embeddings are
of another kind or length is linked to none of its voices and starts none. Only chunks already
transcribed are read, so a voice is the same live and in replay.

Voices are numbered 1, 2, 3 ... in the order they were first heard, per base and session. The base
keeps its registry through a run restarted after a recording error; a base launched again into the
session starts a new one, and its transcripts name the registry (voice_registry) so that readers
keep the two numberings apart, and the kind of embedding it linked by (voice_embedding).

The threshold and the minimum were chosen on recorded classroom group-microphone audio without
labels: each session's chunks, diarized and embedded as the service does, were linked in order at
a range of thresholds, against the speakers of the same session diarized whole. They sit where the
registry splits a speaker about as much speech as it joins of two, with few voices from short
remnants. The whole-session clustering stands in for the truth, so the values balance the linking's
two errors and do not measure how well voices follow people: on a far-field microphone in a busy
classroom, a voice is a stable sound, not a pupil.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

VOICE_LINK_THRESHOLD = 0.30  # cosine similarity at which a chunk's speaker is a voice already heard
VOICE_MIN_SECONDS = 2.0  # the speech a speaker needs alone in a chunk to start a voice of its own

# the kinds of speaker embedding a speech transcriber answers with (its speaker_embedding_kind)
SPEECH_EMBEDDING = "speech"  # each speaker's own speech in the chunk, embedded whole
CENTROID_EMBEDDING = "centroid"  # pyannote's clustering centroid: an answer that names no kind
VOICE_EMBEDDING = SPEECH_EMBEDDING  # the kind a base expects, for which the two values above were chosen


def embedding_kind(value) -> str:
    """the kind of the speaker embeddings of a speech transcriber's answer, from the kind it names
    (speaker_embedding_kind): CENTROID_EMBEDDING when it names none, as a service from before the
    kind was named returned pyannote's centroids."""
    text = str(value).strip() if value is not None else ''
    return text or CENTROID_EMBEDDING


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


def speaker_alone_seconds(turns) -> dict[str, float]:
    """how long each speaker of a chunk's turns ({start, end, speaker}) spoke with no other speaker
    talking, in seconds: the speech its embedding is made of (speaker_speech_embeddings), so a
    speaker heard only over another has none."""
    spans: dict[str, list[tuple[float, float]]] = {}
    for turn in turns or []:
        try:
            start, end = float(turn['start']), float(turn['end'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            spans.setdefault(str(turn.get('speaker')), []).append((start, end))
    # the moments where the number of speakers talking changes, in order
    events = sorted((moment, change, label) for label, pieces in spans.items()
                    for start, end in pieces for moment, change in ((start, 1), (end, -1)))
    alone = {label: 0.0 for label in spans}
    talking: dict[str, int] = {}
    previous = None
    for moment, change, label in events:
        if previous is not None and moment > previous:
            active = [name for name, count in talking.items() if count > 0]
            if len(active) == 1:
                alone[active[0]] += moment - previous
        talking[label] = talking.get(label, 0) + change
        previous = moment
    return alone


class VoiceRegistry:
    """the voices one base has heard in a session: per voice, the sum of its speakers' unit
    embeddings weighted by their seconds of speech, and that weight; and the kind of embedding they
    are all of (`kind`, that of the first voice's chunk, None before it)."""

    def __init__(self, threshold: float = VOICE_LINK_THRESHOLD, min_seconds: float = VOICE_MIN_SECONDS):
        self.threshold = float(threshold)
        self.min_seconds = float(min_seconds)
        self.kind = None
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

    def link(self, embeddings: dict | None, seconds: dict | None = None, kind: str | None = None,
             start_seconds: dict | None = None) -> dict[str, dict[str, Any]]:
        """the session voice of each speaker of one chunk: {speaker: {'voice': n, 'similarity': s}},
        s the cosine similarity to the voice's centroid before this chunk, None for a voice this
        chunk started. `embeddings` is {speaker: vector}, `seconds` {speaker: seconds of speech}
        (speaker_seconds of the chunk's turns), which weigh the speakers in their voices' centroids,
        `start_seconds` {speaker: seconds} the speech that decides whether a speaker left over starts
        a voice (speaker_alone_seconds of the turns; `seconds` when not given), `kind` the kind of
        the embeddings (embedding_kind). Two speakers of one chunk are never one voice; a speaker
        without a usable embedding, or matched to no voice with too little speech to start one, is
        left out. Embeddings of another kind than the voices' are compared with none of them and
        start none, as embeddings of another length are. The voices' centroids take in this chunk's
        speakers once all are decided."""
        seconds = seconds or {}
        start_seconds = seconds if start_seconds is None else start_seconds
        units = {str(label): unit for label, unit in
                 ((label, unit_vector(vector)) for label, vector in (embeddings or {}).items()) if unit is not None}
        if not units:
            return {}
        if self._sums and kind != self.kind:
            return {}  # embeddings of another kind: their similarity to these voices means nothing
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
            if label in links or float(start_seconds.get(label, 0.0)) < self.min_seconds:
                continue
            if self._sums and len(self._sums[0]) != dimension:
                continue  # an embedding of another model: it cannot be one of these voices
            if not self._sums:
                self.kind = kind  # the first voice sets the kind of all
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
