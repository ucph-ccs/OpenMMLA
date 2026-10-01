"""The speech part of a session report: how much of the session held speech, who spoke, for how long,
after whom, and what was said, from the ASR recognition buckets and transcription chunks.

Who a speaker is depends on the microphones (the mode). One group microphone knows only anonymous
voices: its entities are the diarized session voices, the five longest on their own and the rest,
with the turns no voice was linked to, folded into "Other voices". Worn microphones name their
wearer: their entities are the tags, with the seconds of the buckets each one won, and words only
where the levels give a word to the wearer, because most of what a worn microphone transcribes is
its neighbours'. A group microphone running beside worn ones is a lane of its own: it decides the
session's speech and silence, as in a session that had only it, and stays out of the shares.

A few rules from the data hold throughout: several microphones' durations overlap in time, so a
bucket never counts for more than its own length; overlapping turns of one voice count once; a
missing bucket is a gap, never silence. Times in the output are seconds from the session's t0.
"""
from __future__ import annotations

import json
import math
import re
from bisect import bisect_right
from collections import defaultdict

from openmmla.analytics.fusion.window_features import _end_time, _time, group_speech_entries, personal_speech
from openmmla.analytics.report.common import (
    SILENT_LABELS, VoiceKeys, chunk_turns, chunk_words, is_pupil_tag, jsonable, sort_tags,
)
from openmmla.analytics.report.sessions import speech_mode_of
from openmmla.bases.asr.attribution import WORD_WEARER
from openmmla.utils.asr_scope import participant_of

GROUP_LABEL_RE = re.compile(r'^group_\d+$')
# the synchronizer's bucket when the rows do not say
BUCKET_SECONDS = 3.0
# a bucket starting more than one bucket and this much after the one before leaves a gap
GAP_TOLERANCE = 0.5
# a turn this soon after the one before it, of the same voice, continues it
MERGE_GAP = 0.5
# a change of voice this soon after the last turn is a switch, and a transition
SWITCH_GAP = 2.0
# a change of wearer counts across this much silence
BRIDGE_SECONDS = 9.0
TOP_VOICES = 5
PAUSE_EDGES = [3, 6, 9, 12, 15, 20, 30, 60]
TURN_EDGES = [0.5, 1, 2, 4, 8, 16, 32]


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _r(value, digits: int = 2):
    return None if value is None else round(float(value), digits)


def _is_group_label(name: str, group_id: str | None) -> bool:
    return bool(name) and (name == group_id or GROUP_LABEL_RE.match(name) is not None)


def _entries(record: dict, bucket: float) -> list[tuple[str, float, object]]:
    """a bucket's aligned speakers/durations/segment starts as (name, seconds, segment start)."""
    names = record.get('speakers')
    if names is None:
        names = []
    elif isinstance(names, str):
        names = [names]
    elif not isinstance(names, (list, tuple)):
        names = [names]
    durations = record.get('durations')
    durations = durations if isinstance(durations, (list, tuple)) else [durations] if durations is not None else []
    segments = record.get('segment_start_times')
    segments = segments if isinstance(segments, (list, tuple)) else [segments] if segments is not None else []
    out = []
    for i, name in enumerate(names):
        seconds = _num(durations[i]) if i < len(durations) else None
        segment = segments[i] if i < len(segments) else None
        out.append((str(name), bucket if seconds is None else max(seconds, 0.0), segment))
    return out


def _windows(recognitions: list[dict]) -> list[list]:
    """the recognition buckets as [start, end, entries], sorted, one per start (a bucket written
    twice keeps the entries of both)."""
    by_start: dict[float, list] = {}
    for record in recognitions:
        start = _num(record.get('window_start_time'))
        if start is None or start <= 0:
            continue
        end = _num(record.get('window_end_time'))
        if end is None or end <= start:
            end = start + BUCKET_SECONDS
        entries = _entries(record, end - start)
        key = round(start, 3)
        if key in by_start:
            by_start[key][1] = max(by_start[key][1], end)
            by_start[key][2].extend(entries)
        else:
            by_start[key] = [start, end, entries]
    return sorted(by_start.values(), key=lambda window: window[0])


def _bucket_length(windows: list[list]) -> float:
    lengths = sorted(end - start for start, end, _ in windows)
    return round(lengths[len(lengths) // 2], 3) if lengths else BUCKET_SECONDS


def _heard(entries, bucket: float) -> dict:
    """{(name, segment start): seconds}: one speaker's segment reported by several bases counts once."""
    heard: dict = {}
    for name, seconds, segment in entries:
        key = (name, segment)
        heard[key] = max(heard.get(key, 0.0), min(seconds, bucket))
    return heard


def _voiced_seconds(entries, bucket: float) -> float:
    return min(sum(s for (name, _), s in _heard(entries, bucket).items() if name not in SILENT_LABELS), bucket)


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _covered_twice(intervals: list[tuple[float, float]]) -> float:
    """time covered by at least two of the intervals, counted once: a sweep over their edges."""
    edges = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    total, depth, last = 0.0, 0, None
    for moment, change in edges:
        if last is not None and depth >= 2:
            total += moment - last
        depth += change
        last = moment
    return total


def _histogram(values, edges) -> list[int]:
    """counts below the first edge, between each pair of edges, and from the last edge on."""
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[bisect_right(edges, value + 1e-9)] += 1
    return counts


def _entropy_balance(seconds: list[float]) -> float | None:
    """normalised Shannon entropy of the shares, None for fewer than two speakers."""
    positive = [s for s in seconds if s > 0]
    if len(positive) < 2:
        return None
    total = sum(positive)
    return -sum((s / total) * math.log(s / total) for s in positive) / math.log(len(positive))


def _text(value) -> str | None:
    if value is None or isinstance(value, str):
        return value
    # deep parsing turns a text that reads as JSON into a list; give it back as text
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def speech_mode(recognitions: list[dict], transcriptions: list[dict], group_id: str | None = None) -> str | None:
    """'wearer' when transcripts carry a badge as participant and no group microphone named speech,
    'wearer+group' when both, 'group' when the buckets name only the group id (or nothing),
    'individual' when they name speakers (verification profiles); None without ASR. The session
    list's rule (sessions.speech_mode), so both name a session alike."""
    return speech_mode_of(recognitions, transcriptions, group_id)


def _lane_label(tag: str) -> str:
    # a worn microphone labelled other than a badge (a test mic) keeps its own label
    return f'Tag {tag} (worn mic)' if is_pupil_tag(tag) else f'{tag} (worn mic)'


def _align(record: dict, words: list, values: list | None) -> list:
    """per word of chunk_words, the value window_features gave it in word_times order (a class, a
    counted flag); word_spans skips the words before the first stamped one, so those get None."""
    if values is None:
        return [None] * len(words)
    if len(values) == len(words):
        return list(values)
    entries = record.get('words')
    out, index, stamped = [], 0, False
    for entry in entries if isinstance(entries, list) else []:
        has_start = isinstance(entry, dict) and _num(entry.get('start')) is not None
        if has_start or stamped:
            stamped = True
            out.append(values[index] if index < len(values) else None)
            index += 1
        else:
            out.append(None)
    out = out[:len(words)]
    return out + [None] * (len(words) - len(out))


def _word_moments(record: dict, words: list) -> list[float]:
    """when each word was said (epoch): its stamp, or the chunk's text spread evenly over the chunk
    when the transcriber gave no words."""
    if words:
        return [start for _, start, _, _ in words]
    tokens = (_text(record.get('text')) or '').split()
    start, end = _time(record), _end_time(record, 1.0)
    return [start + (end - start) * i / len(tokens) for i in range(len(tokens))]


def _merge_turns(turns: list[tuple[float, float, str]]) -> list[list]:
    """the turns in order of their start, a turn of the same voice as the one before it and at most
    MERGE_GAP after it joined to it; a turn of another voice in between (a backchannel) keeps them
    apart, as the live view counts them."""
    merged: list[list] = []
    for start, end, key in sorted(turns):
        if merged and merged[-1][2] == key and start - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, key])
    return merged


def _runs(windows: list[list], names_of, bucket: float) -> list[list]:
    """per name, the maximal runs of contiguous buckets naming it, as [start, end, name]."""
    runs: list[list] = []
    open_runs: dict[str, list] = {}
    previous = None
    for start, end, _ in windows:
        contiguous = previous is not None and start - previous <= bucket + GAP_TOLERANCE
        names = names_of(start)
        for name in list(open_runs):
            if not contiguous or name not in names:
                runs.append(open_runs.pop(name))
        for name in names:
            if name in open_runs:
                open_runs[name][1] = end
            else:
                open_runs[name] = [start, end, name]
        previous = start
    runs.extend(open_runs.values())
    return sorted(runs, key=lambda run: (run[0], run[1], run[2]))


def build_speech(recognitions: list[dict], transcriptions: list[dict], t0: float, t1: float,
                 group_id: str | None = None) -> dict:
    """the speech part's data (SPEC 3.5) from a session's parsed asr_recognition and
    asr_transcription rows (common.fetch), with times as offsets from `t0`; `group_id` is the
    group microphone's label when the session document gives it, else the buckets' group label."""
    t0, t1 = float(t0), float(t1)
    duration = max(t1 - t0, 0.0)
    notes: list[str] = []
    recognitions = sorted((r for r in recognitions if (_num(r.get('window_start_time')) or 0) > 0),
                          key=lambda r: _num(r.get('window_start_time')))
    chunks = sorted((r for r in transcriptions if _time(r) > 0), key=lambda r: (_time(r), _end_time(r, 1.0)))
    windows = _windows(recognitions)
    bucket = _bucket_length(windows)
    mode = speech_mode(recognitions, chunks, group_id)

    def off(moment: float) -> float:
        return round(moment - t0, 2)

    # the group id: the one given, else the group label the buckets or group chunks use most
    if not group_id:
        seen: dict[str, int] = defaultdict(int)
        for _, _, entries in windows:
            for name, _, _ in entries:
                if GROUP_LABEL_RE.match(name):
                    seen[name] += 1
        for record in chunks:
            speaker = str(record.get('speaker') or '')
            if participant_of(record.get('participant')) is None and GROUP_LABEL_RE.match(speaker):
                seen[speaker] += 1
        group_id = max(seen, key=lambda name: (seen[name], name)) if seen else None

    worn_chunks = [r for r in chunks if participant_of(r.get('participant')) is not None]
    group_chunks = [r for r in chunks if participant_of(r.get('participant')) is None]
    worn_labels = {participant_of(r.get('participant')) for r in worn_chunks}
    wearer_tags = sort_tags(tag for tag in worn_labels if is_pupil_tag(tag))
    # worn microphones that are not badges: never wearers, nor named speakers
    stray = worn_labels - set(wearer_tags)
    personal = None
    if wearer_tags and mode in ('wearer', 'wearer+group'):
        try:
            personal = personal_speech(recognitions, chunks)
        except Exception as exc:  # the attribution is a refinement: the report stands without it
            notes.append(f'Word attribution of the worn microphones failed ({type(exc).__name__}); '
                         'every word a worn microphone transcribed counts for its wearer.')
            personal = None
        if personal is not None and not personal.by_word:
            notes.append('The worn microphones left no level traces: a wearer\'s words are the ones said '
                         'in the buckets the energy vote gave them.')

    # the buckets give speech, silence and who was named; with a group microphone beside worn ones,
    # speech and silence are the group microphone's alone
    wearers_apart, near = None, (lambda moment: True)
    if mode == 'wearer+group':
        if personal is not None and personal.group_speech is not None:
            wearers_apart, near = set(personal.participants), personal.group_near
        elif personal is None:
            wearers_apart = {name for _, _, entries in windows for name, _, _ in entries
                             if name not in SILENT_LABELS and not _is_group_label(name, group_id)} | set(wearer_tags)
    named_mode = mode in ('wearer', 'wearer+group', 'individual')

    activity_t, activity_v, voiced_raw, silent_flags = [], [], [], []
    voiced_total, group_seconds, has_group_entries = 0.0, 0.0, False
    named_seconds: dict[str, float] = defaultdict(float)
    named_by_window: dict[float, set] = {}
    named_spans: list[tuple[float, float, str, float]] = []  # (start, end, name, seconds) for per-minute bins
    for start, end, entries in windows:
        length = end - start
        basis = entries if wearers_apart is None else group_speech_entries(entries, wearers_apart, length, near(start))
        if basis:
            heard = _heard(basis, length)
            spoken = min(sum(s for (name, _), s in heard.items() if name not in SILENT_LABELS), length)
            voiced = spoken / length
            silent = all(name in SILENT_LABELS for name, _ in heard)
            voiced_total += spoken
        else:
            voiced, silent = None, None
        activity_t.append(off(start))
        activity_v.append(_r(voiced, 3))
        voiced_raw.append(voiced)
        silent_flags.append(silent)
        group_entries = [e for e in entries if _is_group_label(e[0], group_id)]
        if group_entries:
            has_group_entries = True
            group_seconds += _voiced_seconds(group_entries, length)
        if named_mode:
            longest: dict[str, float] = {}
            for name, seconds, _ in entries:
                if name in SILENT_LABELS or name in stray or _is_group_label(name, group_id):
                    continue
                longest[name] = max(longest.get(name, 0.0), min(seconds, length))
            named_by_window[start] = set(longest)
            for name, seconds in longest.items():
                named_seconds[name] += seconds
                named_spans.append((start, end, name, seconds))

    readable = [v for v in voiced_raw if v is not None]
    flags = [f for f in silent_flags if f is not None]
    speech_ratio = sum(readable) / len(readable) if readable else None
    silence_ratio = sum(1 for f in flags if f) / len(flags) if flags else None

    # pauses: runs of contiguous silent buckets
    pauses, run, previous = [], 0, None
    for (start, _, _), silent in zip(windows, silent_flags):
        contiguous = previous is not None and start - previous <= bucket + GAP_TOLERANCE
        if silent and (contiguous or run == 0):
            run += 1
        else:
            if run:
                pauses.append(run * bucket)
            run = 1 if silent else 0
        previous = start
    if run:
        pauses.append(run * bucket)

    # coverage and gaps
    gaps = [[off(a_end), off(b_start)] for (a_start, a_end, _), (b_start, _, _) in zip(windows, windows[1:])
            if b_start - a_start > bucket + GAP_TOLERANCE]
    expected = int(round(duration / bucket)) if duration > 0 else 0
    coverage = {'windows': len(windows), 'expected': expected,
                'ratio': _r(min(len(windows) / expected, 1.0), 3) if expected else None, 'gaps': gaps}

    # the transcripts give the words, the voices' turns and the lanes
    voices = VoiceKeys()
    voice_turns: list[tuple[float, float, str]] = []
    transcript, lanes_seen = [], set()
    word_moments: list[float] = []  # the words the session counts
    voice_words: dict[str, int] = defaultdict(int)
    wearer_words: dict[str, int] = defaultdict(int)
    speaker_words: dict[str, int] = defaultdict(int)
    for record in chunks:
        start, end = _time(record), _end_time(record, 1.0)
        tag = participant_of(record.get('participant'))
        if tag is None:
            voice_turns.extend(chunk_turns(record, voices))
        words = chunk_words(record, voices)
        moments = _word_moments(record, words)
        classes = [None] * len(words)
        if tag is not None:
            lane, label = f'tag:{tag}', _lane_label(tag)
            got, failed = None, personal is None
            if personal is not None:
                try:
                    got = personal.word_classes(record, start, end)
                except Exception as exc:
                    failed = True
                    notes.append(f'Word classes of a worn transcript failed ({type(exc).__name__}); '
                                 'its words count for its wearer.')
            classes = _align(record, words, got)
            # a chunk with text and no words has its classes per text word, as its moments are
            per_moment = classes if words else (got or [])
            if got is not None:
                own = [c == WORD_WEARER for c in per_moment]
            elif failed:
                own = [True] * len(moments)
            else:
                own = [personal.won_at(tag, moment) for moment in moments]
            wearer_words[tag] += sum(own)
            if mode == 'wearer':
                # the session's words: each spoken word once, whichever microphones transcribed it
                counted = personal.words_counted(record) if got is not None else None
                if counted is not None:
                    flags = _align(record, words, counted) if words else counted
                    word_moments.extend(m for m, flag in zip(moments, flags) if flag)
                else:
                    word_moments.extend(m for m, flag in zip(moments, own) if flag)
        else:
            lane = 'group'
            speaker = str(record.get('speaker') or '')
            label = speaker if mode == 'individual' and speaker else 'Group mic'
            word_moments.extend(moments)
            for _, _, _, key in words:
                if key is not None:
                    voice_words[key] += 1
            speaker_words[speaker] += len(moments)
        lanes_seen.add(lane)
        transcript.append({
            't0': off(start), 't1': off(end), 'lane': lane, 'speaker': str(record.get('speaker') or ''),
            'label': label, 'text': _text(record.get('text')),
            'words': [[str(word), off(ws), off(we), key, cls] for (word, ws, we, key), cls in zip(words, classes)],
        })
    lanes = []
    if 'group' in lanes_seen:
        lanes.append({'key': 'group', 'label': 'Named speakers' if mode == 'individual' else 'Group mic'})
    for tag in sort_tags(lane[4:] for lane in lanes_seen if lane.startswith('tag:')):
        lanes.append({'key': f'tag:{tag}', 'label': _lane_label(tag)})
    words_total = len(word_moments)

    n_bins = max(1, int(math.ceil(duration / 60.0))) if duration > 0 else 1
    by_entity: dict[str, list[float]] = {}

    def minute_add(series: list[float], a: float, b: float, seconds: float | None = None):
        """adds [a, b) (session offsets) to the minute bins; `seconds` spreads that amount over the
        span in proportion instead of the span's own length."""
        if b <= a:
            return
        scale = 1.0 if seconds is None else seconds / (b - a)
        first, last = max(int(a // 60), 0), min(int(b // 60), n_bins - 1)
        for i in range(first, last + 1):
            overlap = min(b, (i + 1) * 60.0) - max(a, i * 60.0)
            if overlap > 0:
                series[i] += overlap * scale

    entities, other, turn_list, transitions = [], None, [], {'keys': [], 'counts': []}
    turn_lengths, overlap_ratio, switches, balance, n_entities = [], None, None, None, 0
    if mode == 'group':
        merged = _merge_turns(voice_turns)
        per_key: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for start, end, key in voice_turns:
            per_key[key].append((start, end))
        unions = {key: _union(spans) for key, spans in per_key.items()}
        seconds = {key: sum(b - a for a, b in spans) for key, spans in unions.items()}
        union_all = _union([(a, b) for a, b, _ in voice_turns])
        spoken = sum(b - a for a, b in union_all)
        if voice_turns:
            overlap_ratio = _covered_twice([(a, b) for a, b, _ in voice_turns]) / spoken if spoken else None
        named_voices = sorted((k for k in seconds if k != 'v:?'), key=lambda k: (-seconds[k], k))
        top = named_voices[:TOP_VOICES]
        folded = named_voices[TOP_VOICES:] + (['v:?'] if 'v:?' in seconds else [])
        fold = {key: (key if key in top else 'other') for key in seconds}
        turn_count = defaultdict(int)
        turn_seconds = defaultdict(float)
        for start, end, key in merged:
            turn_count[key] += 1
            turn_seconds[key] += end - start
        total_seconds = sum(seconds.values())
        for rank, key in enumerate(top, 1):
            spoken_key = seconds[key]
            entities.append({
                'key': key, 'label': voices.label(key), 'kind': 'voice', 'rank': rank, 'slot': rank,
                'seconds': _r(spoken_key, 1), 'share': _r(spoken_key / total_seconds, 4) if total_seconds else None,
                'turns': turn_count[key],
                'mean_turn': _r(turn_seconds[key] / turn_count[key]) if turn_count[key] else None,
                'words': voice_words.get(key, 0),
                'wpm': _r(voice_words.get(key, 0) / (spoken_key / 60.0), 1) if spoken_key > 0 else None,
            })
            series = [0.0] * n_bins
            for a, b in unions[key]:
                minute_add(series, a - t0, b - t0)
            by_entity[key] = series
        if folded:
            other_seconds = sum(seconds[k] for k in folded)
            other = {'key': 'other', 'label': 'Other voices', 'members': folded,
                     'seconds': _r(other_seconds, 1),
                     'share': _r(other_seconds / total_seconds, 4) if total_seconds else None,
                     'turns': sum(turn_count[k] for k in folded), 'words': sum(voice_words.get(k, 0) for k in folded)}
            series = [0.0] * n_bins
            for key in folded:
                for a, b in unions[key]:
                    minute_add(series, a - t0, b - t0)
            by_entity['other'] = series
        keys = top + (['other'] if folded else [])
        index = {key: i for i, key in enumerate(keys)}
        counts = [[0] * len(keys) for _ in keys]
        switches = 0
        for x, y in zip(merged, merged[1:]):
            if x[2] != y[2] and y[0] - x[1] <= SWITCH_GAP:
                switches += 1
                counts[index[fold[x[2]]]][index[fold[y[2]]]] += 1
        transitions = {'keys': keys, 'counts': counts}
        turn_list = [[off(a), off(b), key] for a, b, key in merged]
        turn_lengths = [b - a for a, b, _ in merged]
        balance = _entropy_balance([seconds[k] for k in named_voices])
        n_entities = sum(1 for k in named_voices if seconds[k] > 0)
        if not voice_turns:
            switches = None
            if chunks:
                notes.append('The transcripts carry no diarized turns, so there are no voices to tell apart.')
        elif not named_voices:
            # a speaker label holds within its chunk only, so unlinked turns cannot say who followed whom
            switches, transitions = None, {'keys': [], 'counts': []}
            notes.append('No diarized turn was linked to a session voice, so speakers cannot be told apart '
                         'across chunks: all speech counts as unlinked.')
    elif named_mode:
        prefix, kind = ('spk:', 'speaker') if mode == 'individual' else ('tag:', 'wearer')
        names = set(named_seconds)
        if mode != 'individual':
            names |= set(wearer_tags)
        names = sort_tags(names)
        runs = _runs(windows, lambda moment: named_by_window.get(moment, set()), bucket)
        run_count, run_seconds = defaultdict(int), defaultdict(float)
        for start, end, name in runs:
            run_count[name] += 1
            run_seconds[name] += end - start
        ranked = sorted(names, key=lambda n: (-named_seconds.get(n, 0.0), n))
        rank_of = {name: i for i, name in enumerate(ranked, 1)}
        order = ranked if mode == 'individual' else names
        total_seconds = sum(named_seconds.get(n, 0.0) for n in names)
        per_name_series = {name: [0.0] * n_bins for name in names}
        for start, end, name, spoken_name in named_spans:
            minute_add(per_name_series[name], start - t0, end - t0, spoken_name)
        for name in order:
            spoken_name = named_seconds.get(name, 0.0)
            count = wearer_words.get(name, 0) if mode != 'individual' else speaker_words.get(name, 0)
            entities.append({
                'key': prefix + name, 'label': name if mode == 'individual' else f'Tag {name}', 'kind': kind,
                'rank': rank_of[name],
                # a tag's colour comes from the roster in the browser; a named speaker's from its rank
                'slot': rank_of[name] if mode == 'individual' and rank_of[name] <= TOP_VOICES else None,
                'seconds': _r(spoken_name, 1), 'share': _r(spoken_name / total_seconds, 4) if total_seconds else None,
                'turns': run_count[name],
                'mean_turn': _r(run_seconds[name] / run_count[name]) if run_count[name] else None,
                'words': count, 'wpm': _r(count / (spoken_name / 60.0), 1) if spoken_name > 0 else None,
            })
            by_entity[prefix + name] = per_name_series[name]
        # who follows whom: a name new in a bucket follows the names of the last bucket that named
        # anyone and dropped out, across up to BRIDGE_SECONDS of silence
        keys = [prefix + name for name in order]
        index = {name: i for i, name in enumerate(order)}
        counts = [[0] * len(order) for _ in order]
        last, last_end, switches = None, None, 0
        for start, end, _ in windows:
            current = named_by_window.get(start) or set()
            if not current:
                continue
            if last and start - last_end <= BRIDGE_SECONDS:
                for a in last - current:
                    for b in current - last:
                        counts[index[a]][index[b]] += 1
                        switches += 1
            last, last_end = current, end
        transitions = {'keys': keys, 'counts': counts}
        turn_list = [[off(a), off(b), prefix + name] for a, b, name in runs]
        turn_lengths = [b - a for a, b, _ in runs]
        with_one = sum(1 for names_here in named_by_window.values() if len(names_here) >= 1)
        with_two = sum(1 for names_here in named_by_window.values() if len(names_here) >= 2)
        overlap_ratio = with_two / with_one if with_one else None
        balance = _entropy_balance([named_seconds.get(n, 0.0) for n in names])
        n_entities = sum(1 for n in names if named_seconds.get(n, 0.0) > 0)

    group_mic = None
    if mode in ('group', 'wearer+group') and (has_group_entries or group_chunks):
        group_mic = {'key': f'group:{group_id}' if group_id else 'group', 'label': 'Group mic',
                     'seconds': _r(group_seconds, 1)}

    speech_sum, speech_n, words_per_minute = [0.0] * n_bins, [0] * n_bins, [0] * n_bins
    for (start, _, _), voiced in zip(windows, voiced_raw):
        if voiced is None:
            continue
        i = min(max(int((start - t0) // 60), 0), n_bins - 1)
        speech_sum[i] += voiced
        speech_n[i] += 1
    for moment in word_moments:
        words_per_minute[min(max(int((moment - t0) // 60), 0), n_bins - 1)] += 1
    per_minute = {
        't': [i * 60 for i in range(n_bins)],
        'speech': [_r(speech_sum[i] / speech_n[i], 3) if speech_n[i] else None for i in range(n_bins)],
        'words': words_per_minute,
        'by_entity': {key: [_r(v, 1) for v in series] for key, series in by_entity.items()},
    }

    minutes = duration / 60.0
    merged_count = len(turn_lengths)
    # the words worn microphones transcribe are mostly the teacher's and the neighbours', while
    # their buckets count only the speech a wearer won, so without a group microphone the two do
    # not make a speech rate
    wpm = words_total / (voiced_total / 60.0) if voiced_total > 0 and mode != 'wearer' else None
    if mode == 'wearer' and words_total:
        notes.append('Without a group microphone there is no session speech rate: the worn microphones '
                     'transcribe speech their buckets do not count as a wearer\'s.')
    kpis = {
        'speech_ratio': _r(speech_ratio, 3),
        'silence_ratio': _r(silence_ratio, 3),
        'overlap_ratio': _r(overlap_ratio, 3),
        'switches_per_min': _r(switches / minutes, 2) if switches is not None and minutes > 0 else None,
        'turns': merged_count if (mode == 'group' and voice_turns) or named_mode else None,
        'mean_turn': _r(sum(turn_lengths) / merged_count) if merged_count else None,
        'mean_pause': _r(sum(pauses) / len(pauses)) if pauses else None,
        'words': words_total,
        'wpm': _r(wpm, 1),
        'balance': _r(balance, 3),
        'n_entities': n_entities,
    }

    data = {
        't0': round(t0, 3), 'duration': _r(duration), 'mode': mode, 'coverage': coverage,
        'entities': entities,
        'group_mic': group_mic, 'kpis': kpis,
        'activity': {'step': bucket, 't': activity_t, 'v': activity_v},
        'per_minute': per_minute, 'turns': turn_list, 'transitions': transitions,
        'pauses': {'edges': PAUSE_EDGES, 'counts': _histogram(pauses, PAUSE_EDGES)},
        'turn_lengths': {'edges': TURN_EDGES, 'counts': _histogram(turn_lengths, TURN_EDGES)},
        'lanes': lanes, 'transcript': transcript, 'notes': sorted(set(notes)),
    }
    if other is not None:
        data['other'] = other
    return jsonable(data)
