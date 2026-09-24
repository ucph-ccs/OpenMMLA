"""What was said in a coding window, in Danish and in English, for the mmla ses-code page.

The Danish comes from the session's asr_transcription events: the words stamped inside the window,
with a little context before and after it, grouped by who said them (the group microphone, one of
its anonymous diarized voices, or the worn microphone of a pupil). A voice is the session voice the
base linked the chunk's speaker to ('voice 3' is the same voice in every chunk); a base launched
again into the session numbers its voices anew, and those of such a later registry are named
'voice 3 (set 2)'. A chunk from before the linking names its SPEAKER_NN, which holds within that
chunk only. A chunk without word stamps
gives its whole text, marked approximate. The English is a local MarianMT translation
(Helsinki-NLP/opus-mt-da-en on the CPU): nothing leaves the machine. Translations are cached per
session in artifacts/<session>/analysis/transcripts/windows_<window>s.json with the hash of the
Danish they came from, so a transcript run again is translated again. Without the translation
stack (pip install -e '.[coding]') the Danish is still shown.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from openmmla.utils.asr_scope import participant_of

MODEL_NAME = 'Helsinki-NLP/opus-mt-da-en'
CONTEXT = 2.0
DEFAULT_INFLUX_CONFIG = 'pipelines/ips-base/config_pilot_260603.yml'
EVENT_TYPE = 'asr_transcription'
NO_EVENTS = ('no transcripts in InfluxDB for this session: none were recorded, '
             'or the database cannot be read now (see the log)')
NO_EVENTS_RETRY = 30.0
PARTS = ('before', 'inside', 'after')


# ---- the Danish of a window ----

def _float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _chunk_span(record: dict) -> tuple[float, float] | None:
    start = _float(record.get('window_start_time'))
    if start is None or start <= 0:
        return None
    end = _float(record.get('window_end_time'))
    return start, end if end is not None and end > start else start + 1.0


def _linked(record: dict) -> bool:
    """whether the base linked the chunk's speakers into the voices of its session (`voices`)."""
    voices = record.get('voices')
    if isinstance(voices, str):
        try:
            voices = json.loads(voices)
        except json.JSONDecodeError:
            return False
    return isinstance(voices, dict)


def _voice_sets(records: list[dict]) -> dict[tuple[str, str], str]:
    """what the voices of each voice registry of a microphone are named after their number: nothing
    for the microphone's first registry (by its first chunk), ' (set 2)', ' (set 3)' ... for the
    registries of a base launched again into the session."""
    first: dict[tuple[str, str], float] = {}
    for record in records:
        span = _chunk_span(record)
        if span is None or not _linked(record):
            continue
        key = (source_of(record), str(record.get('voice_registry') or ''))
        first[key] = min(first.get(key, span[0]), span[0])
    names, seen = {}, {}
    for key in sorted(first, key=first.get):
        seen[key[0]] = seen.get(key[0], 0) + 1
        names[key] = '' if seen[key[0]] == 1 else f' (set {seen[key[0]]})'
    return names


def _words(record: dict, voice_set: str = '') -> list[dict] | None:
    """the chunk's words with a start in seconds from the chunk's start and the name of their voice
    (`voice_set` after the number of a linked one, _voice_sets); None when none is stamped. A word
    the aligner could not place takes the stamp of the word before it."""
    entries = record.get('words')
    if isinstance(entries, str):
        try:
            entries = json.loads(entries)
        except json.JSONDecodeError:
            return None
    if not isinstance(entries, list):
        return None
    words, last, linked = [], None, _linked(record)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stamp = _float(entry.get('start'))
        if stamp is not None:
            last = stamp
        text = str(entry.get('word') or entry.get('text') or '').strip()
        if last is None or not text:
            continue
        if linked:
            # the session voice; a speaker the base could not link has none
            voice = f"voice {entry['voice']}{voice_set}" if entry.get('voice') not in (None, '') else None
        else:
            voice = _voice(entry.get('speaker'))
        words.append({'word': text, 'start': last, 'voice': voice})
    return words or None


def _voice(label) -> str | None:
    """a chunk's own diarized speaker as a short name: SPEAKER_01 is 'voice 1'."""
    if label in (None, ''):
        return None
    text = str(label)
    digits = ''.join(c for c in text if c.isdigit())
    return f'voice {int(digits)}' if text.upper().startswith('SPEAKER') and digits else text


def source_of(record: dict) -> str:
    """who a chunk is from: a pupil's worn microphone, or the group microphone."""
    tag = participant_of(record.get('participant'))
    return f'pupil {tag}' if tag is not None else 'group mic'


def window_text(records: Iterable[dict], start: float, end: float, context: float = CONTEXT) -> list[dict]:
    """the lines said in [start, end), with the words said up to `context` seconds before and after.

    One line is a run of words of one chunk and one speaker, in time order: {'speaker', 'start',
    'approximate', 'before', 'inside', 'after'} with the Danish of each part (empty when nothing
    was said in it); its English, added by TextSource.text as 'en', is of the whole line; a line of words said only before or after the window is all context. A
    chunk without stamped words that overlaps the window is one approximate line holding its whole
    text as 'inside'."""
    lines = []
    records = list(records)
    voice_sets = _voice_sets(records)
    for record in records:
        span = _chunk_span(record)
        if span is None:
            continue
        chunk_start, chunk_end = span
        if chunk_end <= start - context or chunk_start >= end + context:
            continue
        source = source_of(record)
        words = _words(record, voice_sets.get((source, str(record.get('voice_registry') or '')), ''))
        if words is None:
            text = ' '.join(str(record.get('text') or '').split())
            if text and chunk_start < end and chunk_end > start:
                lines.append({'speaker': source, 'start': round(chunk_start, 3), 'approximate': True,
                              'before': '', 'inside': text, 'after': ''})
            continue
        run: dict[str, Any] | None = None
        runs = []
        for word in words:
            moment = chunk_start + word['start']
            if moment < start - context or moment >= end + context:
                part = None
            elif moment < start:
                part = 'before'
            elif moment < end:
                part = 'inside'
            else:
                part = 'after'
            if part is None:
                continue
            voice = word['voice']
            speaker = f'{source} · {voice}' if voice and source == 'group mic' else source
            if run is None or run['speaker'] != speaker:
                run = {'speaker': speaker, 'start': round(moment, 3), 'approximate': False,
                       'before': [], 'inside': [], 'after': []}
                runs.append(run)
            run[part].append(word['word'])
        for run in runs:
            lines.append({**run, **{part: ' '.join(run[part]) for part in PARTS}})
    lines.sort(key=lambda line: line['start'])
    return lines


def source_hash(lines: list[dict]) -> str:
    """the hash of a window's Danish: a cached translation of other words is stale."""
    basis = [[line['speaker'], line['approximate']] + [line[part] for part in PARTS] for line in lines]
    return hashlib.sha256(json.dumps(basis, ensure_ascii=False).encode()).hexdigest()[:16]


# ---- the English ----

class Translator:
    """Danish to English with a MarianMT model on the CPU, loaded once, on first use. The page
    loads it in the background (start_loading) so no request waits the minute a first download
    can take; prepare waits for it (load)."""

    def __init__(self, model_name: str = MODEL_NAME, threads: int = 4):
        self.model_name = model_name
        self.threads = threads
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._error: str | None = None
        self._loader: threading.Thread | None = None

    def load(self) -> str | None:
        """None when the model is ready, else why the translation is unavailable."""
        with self._lock:
            if self._model is not None or self._error is not None:
                return self._error
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except Exception as error:
                self._error = f"translation unavailable: {type(error).__name__}: {error} (pip install -e '.[coding]')"
                return self._error
            try:
                torch.set_num_threads(max(1, self.threads))
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).eval()
            except Exception as error:
                self._error = f'translation unavailable: {self.model_name} could not be loaded ({type(error).__name__}: {error})'
            return self._error

    def start_loading(self) -> None:
        """load the model in a background thread, once; a missing stack is known at once."""
        if self.ready or self._error is not None or (self._loader and self._loader.is_alive()):
            return
        missing = [name for name in ('torch', 'transformers') if importlib.util.find_spec(name) is None]
        if missing:
            self._error = (f"translation unavailable: {' and '.join(missing)} not installed "
                           f"(pip install -e '.[coding]')")
            return
        self._loader = threading.Thread(target=self.load, name='ses-code-translator', daemon=True)
        self._loader.start()

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def busy(self) -> bool:
        """whether a load or a translation holds the model now."""
        return self._lock.locked()

    @property
    def error(self) -> str | None:
        return self._error

    def translate(self, texts: list[str], batch: int = 16) -> list[str]:
        """the English of each text; an empty text stays empty. Raises RuntimeError when the
        model cannot be loaded."""
        error = self.load()
        if error:
            raise RuntimeError(error)
        import torch
        out = [''] * len(texts)
        todo = [i for i, text in enumerate(texts) if text.strip()]
        for first in range(0, len(todo), batch):
            index = todo[first:first + batch]
            with self._lock, torch.no_grad():
                encoded = self._tokenizer([texts[i] for i in index], return_tensors='pt', padding=True,
                                          truncation=True, max_length=512)
                generated = self._model.generate(**encoded, num_beams=4)
                english = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            for i, text in zip(index, english):
                out[i] = text.strip()
        return out


# ---- the cache ----

def cache_path(session_dir: Path, window: float) -> Path:
    return Path(session_dir) / 'analysis' / 'transcripts' / f'windows_{window:g}s.json'


def window_key(start: float) -> str:
    return f'{float(start):.3f}'


def read_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {'windows': {}}
    if not isinstance(data, dict) or not isinstance(data.get('windows'), dict):
        return {'windows': {}}
    return data


_write_lock = threading.Lock()


def write_cache(path: Path, entries: dict[str, dict], **header) -> None:
    """merge `entries` ({window key: entry}) into the cache file and write it whole. The file is
    re-read under a lock first, so the page and a --prepare-text run can both write it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix('.lock')
    with _write_lock, open(lock_path, 'a') as lock_file:
        try:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except ImportError:  # no file locks on this platform: the thread lock still holds
            pass
        data = read_cache(path)
        data.update(header)
        data['windows'].update(entries)
        temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        temp.replace(path)


def _entry(lines: list[dict], english: list[str], model: str) -> dict:
    return {'hash': source_hash(lines), 'model': model, 'translated': True,
            'lines': [{**line, 'en': text} for line, text in zip(lines, english)]}


def _pieces(lines: list[dict]) -> list[str]:
    """what is translated: each line whole, its context with it, since a sentence cut at the
    window's edge translates badly in halves."""
    return [' '.join(line[part] for part in PARTS if line[part]) for line in lines]


# ---- the page's text source ----

class TextSource:
    """the Danish and English of windows, per session: the session's transcripts are read from
    InfluxDB once, the translations from and to the cache."""

    def __init__(self, window: float, context: float = CONTEXT, influx_config: str | None = DEFAULT_INFLUX_CONFIG,
                 translator: Translator | None = None,
                 events: Callable[[str], list[dict]] | None = None):
        self.window = window
        self.context = context
        self.influx_config = influx_config
        self.translator = translator if translator is not None else Translator()
        self._events_of = events
        self._events: dict[str, list[dict]] = {}
        self._empty_at: dict[str, float] = {}
        self._client = None
        self._lock = threading.Lock()

    def events(self, session_id: str) -> list[dict]:
        """the session's transcribed chunks; raises when they cannot be read. The InfluxDB client
        answers a failed query with no events, so no events is not kept: it is asked again after
        NO_EVENTS_RETRY seconds, and said as a LookupError."""
        with self._lock:
            if session_id not in self._events:
                if time.monotonic() - self._empty_at.get(session_id, -NO_EVENTS_RETRY) < NO_EVENTS_RETRY:
                    raise LookupError(NO_EVENTS)
                if self._events_of is not None:
                    records = self._events_of(session_id)
                else:
                    if self._client is None:
                        if not self.influx_config or not os.path.exists(self.influx_config):
                            raise FileNotFoundError(f'no InfluxDB config at {self.influx_config} (--influx-config)')
                        from openmmla.utils.client.influx_client import InfluxDBClientWrapper
                        self._client = InfluxDBClientWrapper(self.influx_config)
                    records = self._client.query_events(session_id, EVENT_TYPE)
                if not records:
                    self._empty_at[session_id] = time.monotonic()
                    raise LookupError(NO_EVENTS)
                self._events[session_id] = sorted(records, key=lambda r: _float(r.get('window_start_time')) or 0.0)
            return self._events[session_id]

    def text(self, session: dict[str, Any], start: float, translate: bool = True,
             prefetch: bool = False) -> dict[str, Any]:
        """the window's lines with their English when it can be had: {'start', 'end', 'context',
        'lines', 'translated', 'model', 'note', 'pending'}; 'note' says why there is no transcript
        or no translation, 'pending' that the model is still loading and the page may ask again.
        No request waits for the model to load, and a prefetch does not wait for another
        translation: the window on screen goes first."""
        end = start + self.window
        out = {'start': start, 'end': end, 'context': self.context, 'lines': [], 'translated': False,
               'model': None, 'note': None, 'pending': False}
        path = cache_path(Path(session['dir']), self.window)
        cached = read_cache(path)['windows'].get(window_key(start))
        try:
            lines = window_text(self.events(session['id']), start, end, self.context)
        except Exception as error:  # an unreachable database must not stop the coding page
            if cached and cached.get('translated'):
                return {**out, 'lines': cached['lines'], 'translated': True, 'model': cached.get('model'),
                        'note': 'from the cache: the transcripts cannot be read now'}
            reason = str(error) if isinstance(error, LookupError) else f'{type(error).__name__}: {error}'
            return {**out, 'note': f'transcript unavailable: {reason}'}
        out['lines'] = [{**line, 'en': None} for line in lines]
        if not lines:
            return out
        digest = source_hash(lines)
        if cached and cached.get('translated') and cached.get('hash') == digest \
                and cached.get('model') == self.translator.model_name:
            return {**out, 'lines': cached['lines'], 'translated': True, 'model': cached['model']}
        if not translate:
            return out
        translator = self.translator
        if not translator.ready:
            translator.start_loading()
            if translator.error:
                return {**out, 'note': translator.error}
            if not translator.ready:
                return {**out, 'pending': True,
                        'note': 'translation loading: the English follows once the model is ready'}
        if prefetch and translator.busy:
            return out
        try:
            english = self.translator.translate(_pieces(lines))
        except Exception as error:
            return {**out, 'note': str(error) if isinstance(error, RuntimeError) else
                    f'translation failed: {type(error).__name__}: {error}'}
        entry = _entry(lines, english, self.translator.model_name)
        write_cache(path, {window_key(start): entry}, model=self.translator.model_name, context=self.context,
                    window=self.window, session=session['id'])
        return {**out, 'lines': entry['lines'], 'translated': True, 'model': self.translator.model_name}

    def prepare(self, session: dict[str, Any], starts: list[float], batch_windows: int = 16,
                say: Callable[[str], None] = print) -> tuple[int, int]:
        """translate every window of `starts` not yet cached; (translated, already cached or silent).
        Raises when the transcripts or the model cannot be had."""
        error = self.translator.load()
        if error:
            raise RuntimeError(error)
        records = self.events(session['id'])
        path = cache_path(Path(session['dir']), self.window)
        cached = read_cache(path)['windows']
        todo = []
        for start in starts:
            lines = window_text(records, start, start + self.window, self.context)
            entry = cached.get(window_key(start))
            if not lines or (entry and entry.get('translated') and entry.get('hash') == source_hash(lines)
                             and entry.get('model') == self.translator.model_name):
                continue
            todo.append((start, lines))
        done = 0
        for first in range(0, len(todo), batch_windows):
            group = todo[first:first + batch_windows]
            english = self.translator.translate([piece for _, lines in group for piece in _pieces(lines)])
            entries, at = {}, 0
            for start, lines in group:
                entries[window_key(start)] = _entry(lines, english[at:at + len(lines)], self.translator.model_name)
                at += len(lines)
            write_cache(path, entries, model=self.translator.model_name, context=self.context, window=self.window,
                        session=session['id'])
            done += len(group)
            say(f"{session['id']}: {done} of {len(todo)} windows translated")
        return done, len(starts) - len(todo)
