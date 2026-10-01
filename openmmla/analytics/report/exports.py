"""The downloads of a session: its raw events as JSON lines, and its transcript as plain text
and as SRT subtitles.

The events are streamed in pieces of point time, so a whole session's video features (about
90 MB) never sit in memory at once, with the point time as epoch seconds (`_t`) in place of the
datetime plain JSON cannot hold. The transcript comes from the speech part (speech.transcript),
in session offsets: a diarized chunk is split where its words change voice, so each block names
the voice that spoke it; a chunk without voices (a worn microphone's, an undiarized one) is one
block under its lane's label."""

from __future__ import annotations

import json
import math
import time
from typing import Iterator

from openmmla.analytics.report.common import entity_label, event_span, fetch, finite_number, jsonable, require_valid

# seconds of point time per query of an export: a minute of video features is about 1.5 MB
EXPORT_CHUNKS = {'vfa_features': 300.0}
EXPORT_CHUNK = 3600.0
# the shortest a subtitle stays on screen
MIN_CUE_SECONDS = 0.5


def iter_jsonl(client, sid: str, event_type: str, chunk: float | None = None) -> Iterator[str]:
    """every record of one event type of a session as a JSON line (parsed fields, `_t` the point
    time in epoch seconds), in time order, read from InfluxDB a piece at a time. The ids are
    checked and the span asked for before this returns, so ValueError (invalid ids) and
    InfluxUnavailable come before the first line; a failure later breaks the iteration."""
    require_valid(sid, event_type)
    span = event_span(client, sid, event_type)
    if span is None:
        return iter(())
    step = max(float(chunk or EXPORT_CHUNKS.get(event_type, EXPORT_CHUNK)), 1.0)
    return _jsonl_lines(client, sid, event_type, span, step)


def _jsonl_lines(client, sid: str, event_type: str, span: tuple[float, float], step: float) -> Iterator[str]:
    start, last = span
    while True:
        end = min(start + step, last)
        final = end >= last
        for row in fetch(client, sid, event_type, start, end + 1e-6 if final else end):
            yield json.dumps(jsonable(row), ensure_ascii=False, separators=(',', ':')) + '\n'
        if final:
            break
        start = end


def _clock(seconds: float, millis: bool = False) -> str:
    seconds = max(float(seconds), 0.0)
    whole = int(math.floor(seconds))
    text = f'{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}'
    if millis:
        text += f',{min(int(round((seconds - whole) * 1000)), 999):03d}'
    return text


def _word(entry) -> tuple[str, float | None, float | None, str | None]:
    if isinstance(entry, dict):
        return (str(entry.get('word') or entry.get('text') or ''), finite_number(entry.get('start')),
                finite_number(entry.get('end')), entry.get('voice') or entry.get('key'))
    if isinstance(entry, (list, tuple)) and entry:
        values = list(entry) + [None] * 4
        word = '' if values[0] is None else str(values[0])
        return word, finite_number(values[1]), finite_number(values[2]), values[3]
    return str(entry or ''), None, None, None


def _blocks(chunk: dict) -> list[tuple[float, float, str, str]]:
    """(start, end, label, text) of a transcript chunk, in session offsets: one block per run of
    words of one voice when its words carry voice keys (a word without one stays in the run it is
    in), else the whole chunk under its lane's label. [] for a chunk with no text."""
    start = finite_number(chunk.get('t0'))
    end = finite_number(chunk.get('t1'))
    start = start if start is not None else 0.0
    end = end if end is not None and end >= start else start
    label = str(chunk.get('label') or chunk.get('speaker') or 'Unknown speaker')
    words = [_word(entry) for entry in chunk.get('words') or []]
    if not any(key for _, _, _, key in words):
        text = chunk.get('text')
        text = text.strip() if isinstance(text, str) else ''
        if not text:
            text = ' '.join(word for word, _, _, _ in words if word).strip()
        return [(start, end, label, text)] if text else []
    first_key = next(key for _, _, _, key in words if key)
    blocks: list[list] = []
    current = None
    for word, first, last, key in words:
        key = key or current or first_key
        first = first if first is not None else (blocks[-1][1] if blocks else start)
        last = last if last is not None and last >= first else first
        if key != current or not blocks:
            blocks.append([first, last, key, []])
            current = key
        block = blocks[-1]
        block[1] = max(block[1], last)
        if word:
            block[3].append(word)
    return [(block[0], block[1], entity_label(block[2]), ' '.join(block[3]).strip()) for block in blocks if block[3]]


def _ordered(transcript: list[dict]) -> list[dict]:
    def order(chunk: dict) -> tuple:
        start = finite_number(chunk.get('t0'))
        return (start if start is not None else math.inf, str(chunk.get('lane') or ''))
    return sorted((chunk for chunk in transcript or [] if isinstance(chunk, dict)), key=order)


def transcript_text(transcript: list[dict], t0: float) -> str:
    """the transcript as readable text: one `[HH:MM:SS] Label: text` block per voice run or chunk,
    elapsed from the session's start, which the first line gives in UTC."""
    begin = finite_number(t0)
    lines = []
    if begin is not None and begin > 0:  # 0 is what a caller without the start passes
        moment = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(begin))
        lines += [f'Times are elapsed from the session start, {moment} UTC.', '']
    for chunk in _ordered(transcript):
        for start, _, label, text in _blocks(chunk):
            lines += [f'[{_clock(start)}] {label}: {text}', '']
    return '\n'.join(lines).rstrip('\n') + '\n' if lines else ''


def transcript_srt(transcript: list[dict]) -> str:
    """the transcript as SRT subtitles in session offsets, one cue per block of transcript_text
    (cues of overlapping microphones overlap, which players show stacked)."""
    cues = []
    for chunk in _ordered(transcript):
        for start, end, label, text in _blocks(chunk):
            start = max(start, 0.0)
            end = max(end, start + MIN_CUE_SECONDS)
            cues.append(f'{len(cues) + 1}\n{_clock(start, True)} --> {_clock(end, True)}\n{label}: {text}\n')
    return '\n'.join(cues)
