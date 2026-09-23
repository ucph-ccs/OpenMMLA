"""Energy attribution of personal microphones: the level of a segment, a base's running noise
floor, and the synchronizer's vote over the personal channels of a bucket.

A microphone worn by one participant (a Bases entry with `participant`) hears its wearer loudest
and the neighbours as cross-talk. Every base measures the level of each raw segment in dBFS and
keeps its own noise floor; the synchronizer compares, per bucket, how far each worn microphone's
speech stands above its floor, and only the loudest (and those within a tie of it) keep their
speech. A group or room microphone never votes.
"""
from __future__ import annotations

import json
import math
import zlib
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np

MIN_DB = -100.0
FULL_SCALE = 32768.0
FLOOR_SECONDS, FLOOR_PERCENTILE, FLOOR_MIN_COUNT = 60.0, 10.0, 5
ENERGY_MARGIN_DB, ENERGY_TIE_DB = 6.0, 3.0
SILENT_SPEAKERS = ('silent', 'unknown')


def level_db(amplitude: float) -> float:
    """20*log10(amplitude / 32768), never below MIN_DB (no -inf for silence)."""
    try:
        amplitude = float(amplitude)
    except (TypeError, ValueError):
        return MIN_DB
    if not math.isfinite(amplitude) or amplitude <= 0:
        return MIN_DB
    return max(20.0 * math.log10(amplitude / FULL_SCALE), MIN_DB)


def segment_energy(frames: bytes) -> tuple[float, float]:
    """(rms_db, peak_db) of 16-bit mono PCM, computed in float64 so -32768 squared does not
    overflow; an odd trailing byte is dropped; empty or all-zero frames give (MIN_DB, MIN_DB)."""
    frames = bytes(frames or b'')
    samples = np.frombuffer(frames[:len(frames) // 2 * 2], dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return MIN_DB, MIN_DB
    rms = math.sqrt(float(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    return level_db(rms), level_db(peak)


class NoiseFloor:
    """a base's running noise floor: the FLOOR_PERCENTILE-th percentile of the rms_db of its
    segments over the last FLOOR_SECONDS, silent ones included."""

    def __init__(self, seconds: float = FLOOR_SECONDS, percentile: float = FLOOR_PERCENTILE,
                 min_count: int = FLOOR_MIN_COUNT):
        self.seconds = float(seconds)
        self.percentile = float(percentile)
        self.min_count = int(min_count)
        self._levels: deque[tuple[float, float]] = deque()

    def __len__(self) -> int:
        return len(self._levels)

    def update(self, moment: float, rms_db: float) -> float:
        """adds (moment, rms_db) unless it is digital silence (<= MIN_DB), forgets what is older
        than moment - seconds, and returns the floor: the percentile (np.percentile) once
        min_count values are held, else their minimum, else rms_db itself."""
        moment, rms_db = float(moment), float(rms_db)
        if rms_db > MIN_DB:
            self._levels.append((moment, rms_db))
        # a gap longer than the window empties it, so the floor starts again
        while self._levels and self._levels[0][0] < moment - self.seconds:
            self._levels.popleft()
        levels = [level for _, level in self._levels]
        if len(levels) >= self.min_count:
            return float(np.percentile(levels, self.percentile))
        if levels:
            return min(levels)
        return rms_db


def energy_record(rms_db: float, peak_db: float, floor_db: float) -> dict:
    """{'rms_db', 'peak_db', 'floor_db'} rounded to 2, as a recognition carries it."""
    return {'rms_db': round(float(rms_db), 2), 'peak_db': round(float(peak_db), 2),
            'floor_db': round(float(floor_db), 2)}


def _finite(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_energy(value) -> dict | None:
    """an energy from a payload (a dict, or its JSON string) with numeric, finite rms_db and
    floor_db (floats); None otherwise."""
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict):
        return None
    rms, floor = _finite(value.get('rms_db')), _finite(value.get('floor_db'))
    if rms is None or floor is None:
        return None
    energy = {'rms_db': rms, 'floor_db': floor}
    peak = _finite(value.get('peak_db'))
    if peak is not None:
        energy['peak_db'] = peak
    return energy


def snr_db(energy) -> float | None:
    """rms_db - floor_db of an energy, None when as_energy gives None."""
    energy = as_energy(energy)
    if energy is None:
        return None
    return energy['rms_db'] - energy['floor_db']


def as_decibels(value, default: float) -> float:
    """a dB setting from a config: a number, else the default (None, blank, <...>, text)."""
    number = _finite(value.strip() if isinstance(value, str) else value)
    return float(default) if number is None else number


def energy_vote(snrs: dict[str, float], margin_db: float = ENERGY_MARGIN_DB,
                tie_db: float = ENERGY_TIE_DB) -> set[str]:
    """the participants who spoke in a bucket: the loudest over its floor when it is at least
    margin_db up, and every other within tie_db of it; empty when none reaches the margin."""
    if not snrs:
        return set()
    loudest = max(snrs.values())
    if loudest < margin_db:
        return set()
    return {participant for participant, snr in snrs.items() if snr >= loudest - tie_db}


def _is_personal(result) -> bool:
    return isinstance(result, dict) and result.get('participant') not in (None, '')


def _has_speech(result: dict) -> bool:
    speakers = result.get('speakers') or []
    if isinstance(speakers, str):
        speakers = [speakers]
    return any(str(speaker) not in SILENT_SPEAKERS for speaker in speakers)


def attribute_bucket(frame_results: dict[str, dict], margin_db: float, tie_db: float,
                     silent_duration: float) -> tuple[dict[str, dict], dict[str, float]]:
    """(the bucket with every personal channel that lost the vote made silent, {participant:
    snr rounded to 1} of the personal channels that had speech and a usable energy)."""
    voters: dict[str, tuple[str, float]] = {}  # base id -> (participant, snr)
    snrs: dict[str, float] = {}
    for base_id, result in frame_results.items():
        if not _is_personal(result) or not _has_speech(result):
            continue
        snr = snr_db(result.get('energy'))
        if snr is None:
            continue  # speech without a usable level: kept as it is, and it does not vote
        participant = str(result['participant'])
        voters[base_id] = (participant, snr)
        snrs[participant] = max(snr, snrs.get(participant, snr))
    if not voters:
        return frame_results, {}
    winners = energy_vote(snrs, margin_db, tie_db)
    voted: dict[str, dict] = {}
    for base_id, result in frame_results.items():
        if base_id in voters and voters[base_id][0] not in winners:
            result = {**result, 'speakers': ['silent'], 'similarities': [0.0], 'durations': [silent_duration]}
        voted[base_id] = result
    return voted, {participant: round(snr, 1) for participant, snr in snrs.items()}


def transcript_time(chunk_end_time: float, base_key: str) -> datetime:
    """the moment a wearer base's transcript is stored at: its chunk end plus 1-99 999 us fixed
    per base (zlib.crc32(base_key.encode()) % 99_999 + 1), so the transcripts of several bases
    that end together do not overwrite each other in InfluxDB (tz=utc)."""
    offset = zlib.crc32(str(base_key).encode()) % 99_999 + 1
    return datetime.fromtimestamp(float(chunk_end_time), tz=timezone.utc) + timedelta(microseconds=offset)
