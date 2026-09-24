"""Energy attribution of personal microphones: the level of a segment, a base's running noise
floor, and the synchronizer's vote over the personal channels of a bucket.

A microphone worn by one participant (a Bases entry with `participant`) hears its wearer loudest
and the neighbours as cross-talk. Every base measures the level of each raw segment in dBFS and
keeps its own noise floor; the synchronizer compares, per bucket, how far each worn microphone's
speech stands above its floor, and only the loudest (and those within a tie of it) keep their
speech. A group or room microphone never votes. A base may also gate its speech on that snr
(speech_gate: relative).

A 3 s bucket is coarse: worn microphones hear each other and the teacher, and the loudest leads
the next by a few dB most of the time, so the vote often credits both or the wrong one. A worn
base therefore also measures its level every 100 ms (level_trace), with its floor: its transcripts
carry the trace of their chunk, its recognitions that of their segment, which the synchronizer
keeps in the bucket for every worn microphone, speech or silence. Each word is then decided from
the levels of all worn microphones while it was said (attribute_word): the wearer's own when their
microphone leads by the margin, another wearer's (cross-talk) when theirs does, nobody's when none
does.
"""
from __future__ import annotations

import bisect
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
SPEECH_GATES = ('absolute', 'relative')
SPEECH_GATE_SNR_DB = 6.0
SILENT_SPEAKERS = ('silent', 'unknown')
LEVEL_HOP_SECONDS = 0.1  # the step of a worn microphone's level trace (level_trace)
# whose a word on a worn microphone is (attribute_word)
WORD_WEARER, WORD_CROSSTALK, WORD_OTHER = 'wearer', 'crosstalk', 'other'


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


def level_trace(frames: bytes, framerate: int = 16000, hop: float = LEVEL_HOP_SECONDS) -> list[float]:
    """the level of 16-bit mono PCM every `hop` seconds: the rms of each stretch in dBFS (level_db,
    never below MIN_DB) rounded to 0.1 dB, the last stretch shorter when the frames end inside it;
    empty for no frames."""
    frames = bytes(frames or b'')
    samples = np.frombuffer(frames[:len(frames) // 2 * 2], dtype=np.int16).astype(np.float64)
    step = max(int(round(float(framerate) * float(hop))), 1)
    if samples.size == 0:
        return []
    whole = samples.size // step
    powers = list((samples[:whole * step].reshape(whole, step) ** 2).mean(axis=1)) if whole else []
    if samples.size > whole * step:
        rest = samples[whole * step:]
        powers.append(float(np.mean(rest * rest)))
    return [round(level_db(math.sqrt(float(power))), 1) for power in powers]


class NoiseFloor:
    """a base's running noise floor: the FLOOR_PERCENTILE-th percentile of the rms_db of its
    segments over the last FLOOR_SECONDS, silent ones included. Until it holds min_count
    segments it is the lowest level of the segments before the one being measured, and the first
    segment of a run (or after a gap longer than the window) has none: a segment is never measured
    against its own level. `last` is the floor the latest update returned (None before any)."""

    def __init__(self, seconds: float = FLOOR_SECONDS, percentile: float = FLOOR_PERCENTILE,
                 min_count: int = FLOOR_MIN_COUNT):
        self.seconds = float(seconds)
        self.percentile = float(percentile)
        self.min_count = int(min_count)
        self._levels: deque[tuple[float, float]] = deque()
        self.last: float | None = None

    def __len__(self) -> int:
        return len(self._levels)

    def update(self, moment: float, rms_db: float) -> float | None:
        """forgets what is older than moment - seconds, adds (moment, rms_db) unless it is digital
        silence (<= MIN_DB), and returns the floor: the percentile (np.percentile) of the values held
        once min_count are, else the lowest of those held before this one, else None."""
        moment, rms_db = float(moment), float(rms_db)
        # a gap longer than the window empties it, so the floor starts again
        while self._levels and self._levels[0][0] < moment - self.seconds:
            self._levels.popleft()
        earlier = [level for _, level in self._levels]
        if rms_db > MIN_DB:
            self._levels.append((moment, rms_db))
        if len(self._levels) >= self.min_count:
            floor = float(np.percentile([level for _, level in self._levels], self.percentile))
        else:
            floor = min(earlier) if earlier else None
        self.last = floor
        return floor


def energy_record(rms_db: float, peak_db: float, floor_db: float | None) -> dict:
    """{'rms_db', 'peak_db', 'floor_db'} rounded to 2, as a recognition carries it; floor_db None
    when the base had no floor yet (as_energy then finds no usable energy: the segment neither
    votes nor passes the relative gate)."""
    floor = _finite(floor_db)
    return {'rms_db': round(float(rms_db), 2), 'peak_db': round(float(peak_db), 2),
            'floor_db': None if floor is None else round(floor, 2)}


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
    return as_number(value, default)


def as_number(value, default: float) -> float:
    """a number from a config: the value as a float, else the default (None, blank, <...>, text,
    a yes/no)."""
    number = _finite(value.strip() if isinstance(value, str) else value)
    return float(default) if number is None else number


def speech_gate_of(value) -> str:
    """the speech gate a Base block names: 'absolute' (the default: nothing, a blank or an unfilled
    placeholder) or 'relative'; case and spaces do not matter; anything else is an error."""
    if value is None:
        return 'absolute'
    text = str(value).strip()
    if not text or (text.startswith('<') and text.endswith('>')):
        return 'absolute'
    gate = text.lower()
    if gate in SPEECH_GATES:
        return gate
    raise ValueError(f"speech_gate must be 'absolute' or 'relative', not '{value}'.")


def relative_speech(energy, min_snr_db: float) -> bool:
    """whether a segment's raw level stands at least min_snr_db over its base's noise floor (snr_db
    of its energy); False when the energy is not usable."""
    snr = snr_db(energy)
    return snr is not None and snr >= float(min_snr_db)


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


def levels_record(trace: list[float], floor_db: float | None, hop: float = LEVEL_HOP_SECONDS) -> dict:
    """{'hop', 'floor_db', 'db'}: a level trace (level_trace, from the start of what it measures)
    and its base's noise floor then (None when it had none), as a worn microphone's transcript
    carries it for its chunk and its recognition for its segment."""
    floor = _finite(floor_db)
    return {'hop': float(hop), 'floor_db': None if floor is None else round(floor, 2),
            'db': [float(level) for level in trace]}


def as_levels(value) -> dict | None:
    """levels (levels_record; a dict, or its JSON string) with a positive hop, a finite floor_db
    and a list of numbers in db, as floats; None otherwise."""
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict):
        return None
    hop, floor, trace = _finite(value.get('hop')), _finite(value.get('floor_db')), value.get('db')
    if hop is None or hop <= 0 or floor is None or not isinstance(trace, list):
        return None
    levels = [_finite(level) for level in trace]
    if any(level is None for level in levels):
        return None
    return {'hop': hop, 'floor_db': floor, 'db': levels}


def span_powers(trace: list[float], hop: float, first: float, last: float) -> list[float]:
    """the powers (10^(dB/10)) of the steps of a trace (dB every `hop` seconds from its start) that a
    span from `first` to `last` seconds after that start overlaps (the step holding `first` when the
    span has no length); empty when the span misses the trace. A span that runs past the trace's
    end is cut there, and the caller reads the rest from the trace that follows."""
    if not trace or not hop > 0 or _finite(first) is None or _finite(last) is None:
        return []
    if last < 0 or first >= len(trace) * hop:
        return []
    # a millionth of a step of slack: session times near 1.7e9 s carry rounding of about 1e-7 s, which
    # would otherwise take in the step after a span that ends on a step's edge
    begin = max(int(math.floor(max(first, 0.0) / hop + 1e-6)), 0)
    end = int(math.ceil(last / hop - 1e-6)) if last > first else begin + 1
    end = min(max(end, begin + 1), len(trace))
    return [10.0 ** (float(level) / 10.0) for level in trace[begin:end]]


def power_db(powers: list[float]) -> float | None:
    """the level of the power mean of `powers` in dB; None for none."""
    return 10.0 * math.log10(sum(powers) / len(powers)) if powers else None


def span_level(trace: list[float], hop: float, first: float, last: float) -> float | None:
    """the level of a trace (dB every `hop` seconds from its start) while something lasted from
    `first` to `last` seconds after that start: the power mean of the steps the span overlaps
    (span_powers); None when the span misses the trace."""
    return power_db(span_powers(trace, hop, first, last))


def _floored(snrs: dict[str, float | None], wearer: str) -> dict[str, float]:
    """the snrs with a missing, None or negative one at 0 (at its floor), the wearer's included."""
    levels = {str(tag): max(_finite(snr) or 0.0, 0.0) for tag, snr in (snrs or {}).items()}
    levels.setdefault(str(wearer), 0.0)
    return levels


def _lead(tag: str, levels: dict[str, float]) -> float:
    return levels[tag] - max([level for other, level in levels.items() if other != tag] + [0.0])


def word_lead(wearer: str, snrs: dict[str, float | None]) -> float:
    """how far the wearer's microphone stood over every other worn microphone and over its own
    floor while a word was said, in dB (attribute_word's snrs; negative when another stood higher)."""
    return _lead(str(wearer), _floored(snrs, wearer))


def attribute_word(wearer: str, snrs: dict[str, float | None], margin_db: float = ENERGY_MARGIN_DB) -> str:
    """whose a word on a worn microphone is, from how far each worn microphone of the session stood
    over its own floor while the word was said (`snrs`, participant -> dB; a microphone missing
    or None gave no level then and counts as at its floor, as does one below it):
    WORD_WEARER when the wearer's microphone stands margin_db over every other and over its own
    floor (word_lead); WORD_CROSSTALK when another's does (that wearer said it, and it is theirs);
    WORD_OTHER when none does: someone without a microphone (the teacher), wearers talking over each
    other, or a word too quiet to tell. At most one microphone leads at a moment (count_once settles
    two words whose spans differ)."""
    levels = _floored(snrs, wearer)
    if _lead(str(wearer), levels) >= float(margin_db):
        return WORD_WEARER
    if any(_lead(tag, levels) >= float(margin_db) for tag in levels if tag != str(wearer)):
        return WORD_CROSSTALK
    return WORD_OTHER


def count_once(words: list[tuple[float, float, str, str, float]]) -> list[str]:
    """the classes of worn microphones' words, (start, end, wearer, attribute_word's class,
    word_lead) each, with a spoken word counted once: of two WORD_WEARER words of different wearers
    whose spans overlap by at least half the shorter one (the same word on two microphones, each
    leading over its own span), the one whose microphone led by less becomes WORD_CROSSTALK (on a
    tie the one given first stays). The classes come back in the order given."""
    classes = [str(word[3]) for word in words]
    order = sorted((i for i, word in enumerate(words) if classes[i] == WORD_WEARER), key=lambda i: words[i][0])
    active: list[int] = []
    for i in order:
        start, end, wearer = float(words[i][0]), float(words[i][1]), str(words[i][2])
        active = [j for j in active if float(words[j][1]) > start and classes[j] == WORD_WEARER]
        for j in active:
            other_start, other_end = float(words[j][0]), float(words[j][1])
            shared = min(end, other_end) - max(start, other_start)
            if str(words[j][2]) == wearer or shared < 0.5 * min(end - start, other_end - other_start):
                continue
            loser = i if float(words[i][4]) < float(words[j][4]) or (float(words[i][4]) == float(words[j][4]) and j < i) else j
            classes[loser] = WORD_CROSSTALK
            if loser == i:
                break
        if classes[i] == WORD_WEARER:
            active.append(i)
    return classes


def once_across(words: list[tuple[float, float, str, str, float]], classes: list[str]) -> list[bool]:
    """which words of worn microphones count when every spoken word counts once, whoever said it:
    the words as count_once takes them, (start, end, wearer, class, word_lead), and their classes
    after count_once. A spoken word is on each microphone at most once, so a word stands for at most
    one word of every other microphone: taken a WORD_WEARER word first, then by how far its
    microphone led (on a tie the one given first), a word whose span overlaps a counted word of
    another wearer's microphone by at least half the shorter one, and that counted word stands for
    no word of this microphone yet, is that word (the one it shares most time with) and does not
    count. The flags come back in the order given."""
    order = sorted(range(len(words)), key=lambda i: (classes[i] != WORD_WEARER, -float(words[i][4]), i))
    longest = max((float(word[1]) - float(word[0]) for word in words), default=0.0)
    kept: dict[str, tuple[list[float], list[float], list[int]]] = {}  # per microphone, by start
    stands_for: dict[int, set[str]] = {}  # per counted word: the microphones whose word it is too
    counts = [False] * len(words)
    for i in order:
        start, end, wearer = float(words[i][0]), float(words[i][1]), str(words[i][2])
        same = None
        for mic, (starts, ends, held) in kept.items():
            if mic == wearer:
                continue
            j = bisect.bisect_left(starts, start - longest)
            while j < len(starts) and starts[j] < end:
                shared = min(end, ends[j]) - max(start, starts[j])
                if (shared > 0 and shared >= 0.5 * min(end - start, ends[j] - starts[j])
                        and wearer not in stands_for[held[j]] and (same is None or shared > same[0])):
                    same = (shared, held[j])
                j += 1
        if same is not None:
            stands_for[same[1]].add(wearer)
            continue
        counts[i] = True
        stands_for[i] = set()
        starts, ends, held = kept.setdefault(wearer, ([], [], []))
        j = bisect.bisect_right(starts, start)
        starts.insert(j, start)
        ends.insert(j, end)
        held.insert(j, i)
    return counts


def transcript_time(chunk_end_time: float, base_key: str) -> datetime:
    """the moment a wearer base's transcript is stored at: its chunk end plus 1-99 999 us fixed
    per base (zlib.crc32(base_key.encode()) % 99_999 + 1), so the transcripts of several bases
    that end together do not overwrite each other in InfluxDB (tz=utc)."""
    offset = zlib.crc32(str(base_key).encode()) % 99_999 + 1
    return datetime.fromtimestamp(float(chunk_end_time), tz=timezone.utc) + timedelta(microseconds=offset)
