"""The transcription audit of mmla ses-code (a plan whose task is 'transcript'): a listener who knows the
language writes down, blind to every system output, what is said in sampled 10 s windows of the fused
grid (the content model's unit), and answers the content model's own questions about them; only once a
logged operator step closes their blind pass are they shown the text the content model read, and rate it.
The page and the scorer are audit_page's and audit_score's (dispatching on the plan's task); this module
draws the items, freezes each version's text and closes the blind passes.

The windows. A window is [ws, ws + 10) on the display version's fused grid. Its clip is [ws - 10, ws + 12):
the content model's 10 s of context, the target and a 2 s tail, with the sensing audit's camera grid and
the session's sound: the group microphone; in a session whose worn microphones fed the text (at least
WORN_MIN_SHARE of the version's words) a second clip with their sum ('dual'); in a session without a group
microphone the mix of its worn microphones, analysis/group_mix/audio_mix-0_<t>.wav, its note's sha256
recorded ('mix'). The population of a session is every grid window inside its sound and videos, less the
margin at either end, so a clip always lies inside the recordings.

The sample, per lesson (splits.lesson_key):
  R      random windows. With --audit-windows-from SOURCE (a sensing audit), the source's who-speaks
         windows of every lesson it drew from, its aliases and recording order; each window must start on
         the display grid (within MATCH_TOLERANCE), the source design's microphone must be the session's
         group microphone now, and the window must lie inside the population, else the sample is refused
         (never thinned). A lesson the source drew none of gets --audit-speech fresh windows. The windows
         are ranked within the lesson: the source's reliability windows (in a fresh lesson FRESH_HEAD drawn
         ones) first in a seeded order, the rest after them in another, so the ranks are a uniformly random
         permutation of a simple random sample and every prefix 1..k is a simple random sample of the
         lesson. Sweep 1 is ranks 1 to --audit-first-ranks, sweep 2 the rest; both are frozen and rendered,
         and the page serves sweep 2 only when started with --audit-sweeps 2. Weight: the lesson's
         population over its R items (the scorer recomputes it over the items it keeps).
  S      windows the coders named by --audit-social-coders labelled social (their last label; a labels
         file a model wrote is refused): --audit-social of them, a share --audit-social-both labelled
         social by every named coder and the rest by exactly one, each at least --audit-gap seconds from
         every R and practice window and from the S windows drawn before it in its session; a pool that
         runs out moves its shortfall to the other (printed and recorded). All in sweep 1, no design
         weight, the pools' sizes recorded.
  P      practice, never scored nor revealed: the source's practice windows (without a source,
         audit.PRACTICE_SPEECH windows in the first session of the first kind), and one window in the first
         session of each other kind of sound, at least --audit-gap from every item.
  The reliability subset: R ranks 1 to --audit-reliability-ranks of every lesson and a share
  --audit-reliability of the S windows (the page gives everyone the practice block).

Versions. --audit-sample freezes the display version, rerun (the events at the freeze, after the re-run,
with each session's current table; the primary version and the one the reveal shows), and --audit-freeze
the other, reported (the events the content model read, with the reported tables), with
--audit-asr-events naming each version's asr_transcription events: a pattern with {sid} (a backup's
JSON lines, gzipped or not, or a JSON list) or 'influx'. Per item and version (pipeline_<version>.json,
mode 0600): the table's speech measures (as the sensing audit freezes them) and

  asr: {consumed: {cur, prev, n_words, n_words_prev, n_lines, n_approx, sha},   every source
        group: {the same} | null,                                               the group channel only
        timed: [[word, start, end | null, 'g' | 'w'], ...]}                     offsets from ws, [ws-2, ws+12)
  content: {arm: {context: {teacher, peer_task, peer_other, no_text}}}          with --audit-content-scores

where cur is code_text.window_text of [ws, ws + 10) with the inside parts joined by newlines, exactly the
text the content model read, prev the same of [ws - 10, ws), and sha its sha256's first 16 digits. The
checks: the events' count against the count the version's table was fused from (a problem, refused unless
--audit-allow-drift). These refuse the session and no flag overrides them: with --audit-asr-reference, a cur
or prev that is not the reference's row (source 'all'); a row of a content arm that carries cur_sha whose
cur_sha is missing or not sha; timed words inside the window that are not the words of cur's stamped lines (a
reader drifted from code_text). The read version (every version but the display one: the text the content model
read) is frozen only with --audit-asr-reference and an arm that carries cur_sha, so neither check can be left
out, and the scorer takes its file only when both ran on every window (read_problems). A window on no row of the
version's table keeps its text and is listed in the file's off_grid.

What the server may serve (view.json): per item only {item, part 'speech', kind 'transcript', practice,
reliability, sweep, render} and, from --audit-render, span (CLIP_SPAN, offsets in the clip), clips
({main, worn when dual}) and, for an item not practice, reveal ({prev, cur}: the reveal version's consumed
text). Never the stratum, rank, weight, agreement, labels, session id, word counts, speech measures or
scores; those are in the design (pipeline.json, never served) for the scorer. The page answers phases
'transcribe' and 'reveal' of these items, and serves the sweeps up to --audit-sweeps (default the plan's
sizes['sweeps']).

What the scorer reads besides the frozen files: the plan (task, display_version and reveal_version, sizes,
lessons with their populations, social with its pools and inclusion shares, primary, declared_roles,
declared = DECLARED_TRANSCRIPT, normaliser) and per session the design (lesson, audio_kind, worn_share,
lesson_population, source, labels' sha256, and per item window_start, window_end, practice, reliability,
sweep, stratum 'R' | 'S' | 'P', rank, weight, agreement, pool, source_item). content_arms reads a version's
content scores supplied at scoring the way the freeze reads them (ContentArm.row, with its cur_sha);
read_blind_closed and blind_lines give each name's close and its last blind lines before it.

The blind close (--audit-close-blind ID --audit-auditor NAME): under the request log's lock (the one the
server's saves hold) runtime/audit/<id>/blind_closed.json gets {name: {seq, at}}, seq the close's own log
line; from then on the page refuses the name's blind answers and may reveal to it the items whose
transcription it saved before seq. Names are taken as the open page takes them (NFC). The log's close lines
are the record (logged_closes): the server does not start and a close refuses while the file disagrees with
them (close_problems), the scorer and the references count by them, and a close the file holds but the log does
not (the step stopped between its two writes) is logged by closing that name again. The references
(--audit-export-references ID --audit-auditor NAME --audit-out FILE): the name's last blind line per scored
item before its close, with each frozen version's text, in a file of mode 0600 whose sha256 is logged.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from openmmla.commands.ses import audit as A
from openmmla.commands.ses import code as C
from openmmla.commands.ses import code_locked as L
from openmmla.commands.ses import code_text as T

L.register_loaded(__name__, __file__)

TASK = 'transcript'
# the sizes of a transcription plan a flag does not give (audit.DEFAULTS gives the seed, the margin and the
# practice switch); 'sweeps' is how many sweeps the page serves unless --audit-sweeps says otherwise
TRANSCRIPT_DEFAULTS = {'speech': 10, 'first_ranks': 3, 'sweeps': 1, 'social': 28, 'social_both': 0.65, 'gap': 20.0,
                       'reliability': 0.25, 'reliability_ranks': 1}
# the version frozen at sampling, the primary of the declared analyses and the one the reveal shows
PRIMARY_VERSION = 'rerun'
# the content model's window and the context its words are placed with (code_text)
WINDOW = 10.0
CONTEXT = T.CONTEXT
# a clip is the window with the model's context before it and a tail after it
CLIP_BEFORE, CLIP_AFTER = 10.0, 12.0
CLIP_SPAN = {'length': CLIP_BEFORE + CLIP_AFTER, 'context': [0.0, CLIP_BEFORE],
             'target': [CLIP_BEFORE, CLIP_BEFORE + WINDOW], 'tail': [CLIP_BEFORE + WINDOW, CLIP_BEFORE + CLIP_AFTER]}
# in a lesson the source audit drew nothing of, this many fresh windows take the first ranks, as a source's
# reliability pair does
FRESH_HEAD = 2
# a session's personal microphones fed its text when their chunks hold at least this share of the version's
# words: a stray chunk or two of a badge that was not transcribed does not make a session 'dual'
WORN_MIN_SHARE = 0.05
AUDIO_KINDS = ('group', 'dual', 'mix')
SOCIAL = 'social'
AGREEMENTS = ('both', 'one')
BLIND_CLOSED = 'blind_closed.json'
CLOSE_EVENT = 'audit-close-blind'
# the columns that hold the hash of the text a content row scored
SHA_COLUMNS = ('cur_sha', 'cur_sha256')
RULE ='word start; unplaced words take the previous stamp'
NO_EVENTS_FLAG = "give --audit-asr-events PATTERN|influx: the version's asr_transcription events ({sid} in a pattern)"
# the analyses the transcription audit's scorer reports, fixed here before any answer is scored
DECLARED_TRANSCRIPT = [
    'versions: rerun (V1, the asr_transcription events at its freeze with each session\'s current fused table) is '
    'the primary version and the one the reveal shows; reported (V0, the events the content model read, checked '
    'against its texts and cur_sha, with the reported tables) gives the secondary row',
    'hypothesis: the version\'s consumed text (every source; the inside parts of code_text.window_text of the '
    'window joined by newlines, the text the content model reads); group (the group channel only) as a secondary. '
    'Reference: the primary auditor\'s last blind transcript before their blind close. Normaliser N1 (audit_text), '
    'strict',
    'the primary auditor: --audit-auditor, else the one name whose answers chose the full audit; names that chose '
    'the reliability subset give the agreement; answers of different auditors are never averaged or merged; a '
    'roles note (--log-note) is expected before the first scored answer, else every header warns',
    'left out of every analysis: items flagged, not rendered or not answered, practice items, lines saved after an '
    'auditor\'s blind close (counted); in the WER an item with no reference word (it counts in the detection rates)',
    'weights: an R item\'s design weight is its lesson\'s population over the lesson\'s R items kept in the '
    'analysis (answered, unflagged, rendered, in the sweeps analysed); S items carry no design weight and enter the '
    'conditional analyses only',
    'P1 accuracy: the pooled WER of V1 consumed on R items, sum(w E) / sum(w N) over items with N >= 1, '
    'design-weighted with the unweighted value beside it; the substitution, deletion and insertion shares on the '
    'same denominator; CER over characters; the reference words and the hypothesis tokens',
    'P2 content validity: the within-lesson concordance C of the V1 content scores with the blind judgements on R '
    'and S, unweighted: P2a (headline) peer_other, peer = other against peer = task; P2b teacher, adult = yes '
    'against no; P2c peer_task, peer = task against peer = none; no_text scores 0; also by stratum, with a pooled '
    'AUROC beside it. The V1 arms are supplied at scoring (--audit-content-scores), each row\'s cur_sha equal to '
    'the frozen V1 sha (a mismatch refuses the arm); without one, P2 and P3 are not available for V1 and V0\'s are '
    'shown as the secondary row',
    'P3 the ASR\'s cost: C recomputed from the frozen scorer run on the primary\'s reference in LLM form, the ASR '
    'prev kept and only cur replaced; delta = C_ref - C_ASR (paired) and TR = (C_ASR - 0.5) / (C_ref - 0.5), not '
    'estimable when C_ref <= 0.55; for P2a to P2c',
    'P4 who speaks: on R items of the lessons with a group microphone with V1 speech_ratio > 0 and who not '
    'cannot_tell, the primary\'s weighted shares of member, adult, other_group and none, and the paired difference '
    'from the comparison audit\'s primary (--audit-compare-with, joined on session and window_start); Cohen\'s kappa '
    'over the four classes and member against not; the shares over all lessons beside them',
    'decision bands (descriptive): P2a tracks off-task talk a listener hears when C\'s lower 95 % bound is above '
    '0.5; P3 the ASR costs little when delta\'s upper bound is below 0.05, part of the signal when its lower bound '
    'is above 0, else not resolved; TR at least 0.9 little lost, 0.6 to 0.9 part lost, below 0.6 most lost; with '
    'fewer than 15 peer = other items P2a and P3 for peer_other are descriptive only; the WER has no pass mark',
    'S1 speech detection on R, weighted: the miss rate (cur empty when status = speech and N >= 1, and N >= 5), '
    'the phantom rate (at least 1, and 3, hypothesis tokens when status = none), hypothesis words per minute of none '
    'audio, the same for unintelligible items apart, the share of none items whose hypothesis repeats a 3-gram',
    'S2 the group source\'s WER for both versions where the version has a group channel; by kind of sound (group, '
    'dual, mix) for consumed and group; in dual sessions consumed minus group and the insertion share',
    'S3 V0 (reported): P1, S1 and S2, V1 - V0 paired, and P2 and P3 with V0\'s frozen content arms',
    'S4 by source: recall per reference tag (M, T, O, ?), the source mix of the hits, WER by who, teacher recall '
    'minus member recall (paired)',
    'S5 the words feature of each version\'s table against the N0 reference token count: the ratio of sums, the mean '
    'signed error, Spearman within lessons of at least 5 items; speech_ratio > 0, > 0.1 and > 0.3 against status',
    'S6 content words (N1 less the frozen stopwords and the response words): multiset precision, recall and F1',
    'S7 score invariance between ASR text and reference text: Lin\'s CCC, Spearman and MAE for teacher, peer_task '
    'and peer_other; kappa on the peer argmax letter and on teacher > 0.5; context 1 (ASR prev), context 0, and both '
    'texts in N1 surface form; reproducibility: the ASR cur scored again against the frozen arm (max |delta|)',
    'S8 the coders\' labels against the heard topic on coded R and S items: peer by coder label per coder, kappa of '
    'social and collaborative against other and task, C of peer_other for coder-social against coder-collaborative '
    'beside P2a (the coders could read a translation of the ASR text)',
    'S9 between the transcribers on the items both answered: the WER of each against the other, the system\'s WER '
    'against each and its excess over that floor (paired), kappa for status, who, adult, peer and other_offtask',
    'S10 who speaks further: the confusion matrices against the comparison audit\'s auditors, the dominant source by '
    'transcript word share against who on the items with a reference word, other_offtask = yes against peer_other '
    'where peer is not other',
    'S11 the reveal ratings gist and invented on R (weighted) and on S, as ratings of the version the reveal showed',
    'S12 cloud arms: P2 with each version\'s frozen cloud scores; P3 with a cloud model only if approved, apart',
    'S13 the S stratum alone: P1 and S1, descriptive',
    'sensitivities: N0 for N1; an edge tolerance of 0.5 s (the timed tokens starting in [-0.5, 10.5), those within '
    '0.5 s of an edge optional and so those starting before it that end inside); split and merge; the least WER over '
    'the orders of up to 4 reference lines where overlap is ticked; items with [x] or overlap left out; the '
    'insertions of items with N = 0 added; the lesson macro mean; leave one lesson out; sweep 1 only against every '
    'sweep served; P2 on R only (weighted); a variant map v2 if any; every primary without the sessions '
    '--audit-exclude-sessions names',
    'statistics: a lesson-cluster bootstrap over the lessons, 10,000 draws, percentile 95 % intervals, undefined '
    'draws counted; WER, CER, the rates, C, kappa and CCC recomputed each draw, every ratio a ratio of sums; paired '
    'draws for V1 against V0, consumed against group, C_ref against C_ASR, the auditor against the comparison audit, '
    'the transcribers against each other, teacher recall against member recall; per-lesson rows give counts and '
    'point values only; a leave-one-lesson-out range beside each primary; no p-values',
    'fixed rules: answered R items are a random sample of their lesson as long as the order the page serves them in '
    'is followed (a seeded shuffle of each sweep, sweep 1 first; an item left unanswered before an answered one is a '
    'skip, logged and counted); no interim score: the score refuses before the primary\'s blind close unless '
    '--audit-interim, logged and named in every header; outputs hold counts, never transcript text',
]


class TranscriptRefusal(A.AuditError):
    """a refusal no flag overrides: a frozen text is not the text it must be"""


# ---- the plan's kind and sizes ----

def transcript_plan(plan: dict) -> bool:
    return plan.get('task') == TASK


def sizes_of(args) -> dict:
    """a transcription plan's sizes: the flags, else TRANSCRIPT_DEFAULTS, else the sensing audit's defaults"""
    sizes = {}
    for name in ('seed', 'margin', 'practice', *TRANSCRIPT_DEFAULTS):
        value = getattr(args, f'audit_{name}', None)
        sizes[name] = value if value is not None else TRANSCRIPT_DEFAULTS.get(name, A.DEFAULTS.get(name))
    if not 0 <= sizes['social_both'] <= 1 or not 0 <= sizes['reliability'] <= 1:
        raise A.AuditError('--audit-social-both and --audit-reliability are shares from 0 to 1')
    if sizes['first_ranks'] < 1 or sizes['reliability_ranks'] < 0 or sizes['speech'] < 0 or sizes['social'] < 0:
        raise A.AuditError('--audit-first-ranks is 1 or more; --audit-reliability-ranks, --audit-speech and '
                           '--audit-social are 0 or more')
    return sizes


def normaliser_state() -> dict:
    """the normaliser the scorer applies (audit_text), as the plan records it before any answer"""
    try:
        from openmmla.commands.ses import audit_text
    except ImportError:
        raise A.AuditError("this checkout has no normaliser (openmmla/commands/ses/audit_text.py): it is frozen into "
                           "the plan at sampling") from None
    return {'module': audit_text.__name__, 'version': getattr(audit_text, 'NORMALISER_VERSION', None),
            'sha256': L.file_sha256(audit_text.__file__)}


def _name(raw, what: str = '--audit-auditor') -> str:
    """an auditor's name as the open page takes it (NFC, no control or line-break character)"""
    from openmmla.commands.ses import audit_page
    name, why = audit_page.typed_name(raw)
    if name is None:
        raise A.AuditError(f'give {what} NAME: {why}')
    return name


def write_private_json(path: Path, data: Any) -> None:
    """a JSON file holding transcript text, written whole (a temp file and a rename) and readable by its owner only"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as file:
        file.write(json.dumps(data, indent=1, ensure_ascii=False, default=A._plain) + '\n')
    os.chmod(temp, 0o600)
    os.replace(temp, path)


# ---- the events ----

def _when(value) -> float | None:
    """an event's own time (a datetime from InfluxDB, ISO text from a backup) as epoch seconds"""
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip().replace('Z', '+00:00')).timestamp()
        except ValueError:
            return None
    return None


def _sorted_records(records: list) -> list[dict]:
    """the chunks by their start, then by their own time, then as stored (code_text.TextSource sorts by the start
    alone, keeping the query's time order for chunks that start together)"""
    def key(pair):
        index, record = pair
        when = _when(record.get('time'))
        return (T._float(record.get('window_start_time')) or 0.0, math.inf if when is None else when, index)
    return [r for _, r in sorted(((i, r) for i, r in enumerate(records) if isinstance(r, dict)), key=key)]


def load_asr(sid: str, spec: str | None, influx_config: str | None) -> tuple[list[dict], dict]:
    """a session's asr_transcription records, as stored (JSON fields may be text), and where they came from:
    InfluxDB (spec 'influx', through the repository's client) or a file the pattern names ({sid} for the session):
    JSON lines (a backup, gzipped or not) or a JSON list. Provenance: the source, the sha256 of the file's bytes
    (of the records as canonical JSON for InfluxDB) and the count."""
    if not spec:
        raise A.AuditError(NO_EVENTS_FLAG)
    if spec == 'influx':
        if not influx_config or not os.path.exists(influx_config):
            raise A.AuditError(f'no InfluxDB config at {influx_config} (--influx-config)')
        from openmmla.utils.client.influx_client import InfluxDBClientWrapper
        records = InfluxDBClientWrapper(influx_config).query_events(sid, T.EVENT_TYPE)
        if not records:
            raise A.AuditError(f'no asr_transcription events of {sid} in InfluxDB: none were recorded, or the '
                               'database cannot be read now')
        body = json.dumps(records, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')
        return _sorted_records(records), {'source': 'influxdb', 'sha256': L.sha256_hex(body), 'count': len(records)}
    path = Path(os.path.expanduser(spec.format(sid=sid))).resolve()
    try:
        data = path.read_bytes()
    except OSError:
        raise A.AuditError(f'no asr_transcription events of {sid} at {path}') from None
    text = (gzip.decompress(data) if path.name.endswith('.gz') else data).decode('utf-8')
    if path.name.endswith(('.jsonl', '.jsonl.gz')):
        loaded = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            loaded = loaded.get(T.EVENT_TYPE) or loaded.get('records') or []
    if not isinstance(loaded, list):
        raise A.AuditError(f'{path} holds no list of asr_transcription records')
    records = _sorted_records(loaded)
    return records, {'source': str(path), 'sha256': L.sha256_hex(data), 'count': len(records)}


def stamped_words(record: dict) -> list[tuple[str, float, float | None]] | None:
    """a chunk's words as code_text._words reads them (a word the aligner could not place takes the stamp of the
    word before it; words before the first stamp and empty words are left out), each with the seconds of its
    start and of its own end (None without one) from the chunk's start; None when none is stamped"""
    entries = record.get('words')
    if isinstance(entries, str):
        try:
            entries = json.loads(entries)
        except json.JSONDecodeError:
            return None
    if not isinstance(entries, list):
        return None
    out, last = [], None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stamp = T._float(entry.get('start'))
        if stamp is not None:
            last = stamp
        text = str(entry.get('word') or entry.get('text') or '').strip()
        if last is None or not text:
            continue
        out.append((text, last, T._float(entry.get('end'))))
    return out or None


def timed_words(records: list[dict], ws: float) -> list[tuple]:
    """the words of every chunk starting in [ws - 2, ws + 12) (window_text's context around the window), as
    (moment, chunk order, word order, word, end moment | None, 'g' for the group microphone | 'w' for a worn
    one), in time order; the chunks window_text leaves out are left out"""
    lo, hi = ws - CONTEXT, ws + WINDOW + CONTEXT
    out = []
    for order, record in enumerate(records):
        span = T._chunk_span(record)
        if span is None or span[1] <= lo or span[0] >= hi:
            continue
        words = stamped_words(record)
        if not words:
            continue
        channel = 'g' if T.source_of(record) == 'group mic' else 'w'
        for k, (word, start, end) in enumerate(words):
            moment = span[0] + start
            if lo <= moment < hi:
                out.append((moment, order, k, word, None if end is None else span[0] + end, channel))
    out.sort(key=lambda w: (w[0], w[1], w[2]))
    return out


def worn_share(records: list[dict]) -> float:
    """the share of the version's words a worn microphone's chunks hold"""
    total = worn = 0
    for record in records:
        words = stamped_words(record)
        n = len(words) if words else len(str(record.get('text') or '').split())
        total += n
        worn += n if T.source_of(record) != 'group mic' else 0
    return worn / total if total else 0.0


# ---- the sessions ----

def find_mix(session_dir: Path) -> dict | None:
    """a session's mix of its worn microphones (analysis/group_mix/audio_<device>_<start>.wav with its .json note,
    as replay writes it for a session without a group microphone), or None"""
    folder = Path(session_dir) / 'analysis' / 'group_mix'
    wavs = sorted(folder.glob('audio_*_*.wav')) if folder.is_dir() else []
    if not wavs:
        return None
    if len(wavs) > 1:
        raise A.AuditError(f'{len(wavs)} mixes in analysis/group_mix: which one the text came from is not known')
    wav, note = wavs[0], wavs[0].with_suffix('.json')
    match = A.FILE_START.search(wav.name)
    if not match:
        raise A.AuditError(f'{wav.name} names no <device>_<start>.wav')
    try:
        about = json.loads(note.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise A.AuditError(f'the mix {wav.name} has no readable note ({note.name})') from None
    seconds = T._float(about.get('seconds'))
    if seconds is None:
        import wave
        with wave.open(str(wav), 'rb') as handle:
            seconds = handle.getnframes() / float(handle.getframerate())
    device = re.match(r'audio_(.+)_\d', wav.name)
    return {'device': device.group(1) if device else 'mix', 'path': str(wav), 'start_time': float(match.group(1)),
            'duration': seconds, 'note': {'path': str(note), 'sha256': L.file_sha256(note), 'method': about.get('method'),
                                          'sources': [os.path.basename(str(s)) for s in about.get('sources') or []],
                                          'clipped_samples': about.get('clipped_samples'), 'rate': about.get('rate')}}


def _recording(r: dict) -> dict:
    return {'device': r.get('device'), 'path': r['path'], 'start_time': r['start_time'], 'duration': r.get('duration')}


def asr_problems(table_file: Path, params: dict, asr: dict) -> list[str]:
    """why a table may not describe these events (an empty list when it does): its sha256 against the one its
    parameters recorded, and the events' count against the count the fusion read"""
    problems = []
    recorded = params['outputs'].get(table_file.name)
    if recorded and recorded != L.file_sha256(table_file):
        problems.append('the table changed since it was fused (its sha256 is not the one parameters.json recorded)')
    fused = params['events'].get(T.EVENT_TYPE)
    if fused is not None and int(fused) != int(asr['count']):
        problems.append(f"the fusion read {fused} asr_transcription events, the events here are {asr['count']}")
    return problems


class SpeechSession:
    """a session as the transcription audit draws and freezes it: its videos, the sound of its clips (the group
    microphone, else the mix of its worn microphones), its personal microphones, the version's fused table and
    parameters and the version's asr_transcription events. Nothing of the vision pipeline: no replay config,
    vfa_features, fused view or keyframes."""

    def __init__(self, artifacts: Path, sid: str, sources: dict):
        from openmmla.analytics.interaction import layout as Y
        from openmmla.analytics.interaction import splits
        from openmmla.collection.recording import natural_device_key
        self.id, self.dir = sid, Path(artifacts) / sid
        self.lesson, self.task = splits.lesson_key(sid), splits.task_of(sid)
        self.split = 'TEST' if sid in splits.TEST_SESSIONS else 'DEV'
        self.videos, audios = A.manifest_recordings(self.dir)
        if not self.videos:
            raise A.AuditError('no video recording')
        self.mic = A.group_mic(audios)
        self.mix = None if self.mic else find_mix(self.dir)
        if self.mic is None and self.mix is None:
            raise A.AuditError('no group microphone, and no mix of the worn microphones (analysis/group_mix)')
        self.personal = sorted((a for a in audios if a.get('scope') == 'personal'),
                               key=lambda a: natural_device_key(a.get('device')))
        self.table_file = A.table_path(artifacts, sid, sources['tables'])
        if not self.table_file.exists():
            raise A.AuditError(f'no fused table at {self.table_file}')
        self.params_file = A.parameters_path(self.table_file, sid, sources['parameters'])
        self.params = A.fusion_parameters(self.params_file)
        self.table = Y.read_table(self.table_file)
        self.kept = [str(t) for t in Y.session_roster(self.table, self.dir).kept]
        self.records, self.asr = load_asr(sid, sources['asr'], sources['influx'])
        self.problems = asr_problems(self.table_file, self.params, self.asr)

    @property
    def sound(self) -> dict:
        return self.mic or self.mix

    def population(self, margin: float) -> list[tuple[float, float]]:
        """the grid's windows inside the sound and every video, less `margin` seconds at either end"""
        sound = self.sound
        lo = max([sound['start_time']] + [v['start_time'] for v in self.videos])
        hi = min([sound['start_time'] + (sound.get('duration') or 0)] +
                 [v['start_time'] + (v.get('duration') or 0) for v in self.videos])
        return [(float(a), float(b)) for a, b in zip(self.table['window_start'], self.table['window_end'])
                if float(a) >= lo + margin and float(b) <= hi - margin]

    def audio_kind(self) -> tuple[str, list[dict], float]:
        """(the kind of the session's sound, the worn microphones of its second clip, their share of the words)"""
        share = round(worn_share(self.records), 4)
        if self.mic is None:
            return 'mix', [], share
        if self.personal and share >= WORN_MIN_SHARE:
            return 'dual', [_recording(a) for a in self.personal], share
        return 'group', [], share

    def sources(self) -> dict:
        return {'asr': dict(self.asr, fusion_count=self.params['events'].get(T.EVENT_TYPE),
                            code_text_sha256=L.file_sha256(T.__file__), context=CONTEXT, rule=RULE),
                'table': str(self.table_file), 'table_sha256': L.file_sha256(self.table_file),
                'parameters': str(self.params_file), 'parameters_sha256': L.file_sha256(self.params_file),
                'problems': self.problems}


def version_sources(args) -> dict:
    return {'asr': args.audit_asr_events, 'tables': args.audit_tables, 'parameters': args.audit_parameters,
            'influx': args.influx_config or os.path.join(os.getcwd(), C.DEFAULT_INFLUX_CONFIG)}


def load_session(artifacts: Path, sid: str, sources: dict) -> SpeechSession:
    """a SpeechSession, any failure of its files said as an AuditError naming its kind"""
    try:
        return SpeechSession(artifacts, sid, sources)
    except A.AuditError:
        raise
    except Exception as error:  # the session's failure is said, not the whole step's
        raise A.AuditError(f'{type(error).__name__}: {str(error)[:300]}') from error


# ---- the source audit ----

class SourceAudit:
    """the sensing audit whose who-speaks windows the transcription audit takes: its aliases, recording order and
    sizes, and per session its design's windows, each design the file the source's sampling logged"""

    def __init__(self, artifacts: Path, audit_id: str):
        self.id = audit_id
        self.plan = A.load_plan(artifacts, audit_id)
        if transcript_plan(self.plan):
            raise A.AuditError(f'{audit_id} is a transcription audit: the windows come from a sensing audit')
        self.plan_sha256 = L.file_sha256(A.audit_dir(artifacts, audit_id) / A.PLAN_FILE)
        self.alias = {s['id']: s['alias'] for s in self.plan['sessions']}
        by_alias = {alias: sid for sid, alias in self.alias.items()}
        self.order = [by_alias[a] for a in self.plan['order']]
        logged = A.logged_files(artifacts, audit_id, events=('audit-sample',))
        self.designs, self.design_sha256 = {}, {}
        for sid in self.order:
            path = A.session_audit_dir(artifacts, sid, audit_id) / A.DESIGN_FILE
            digest = L.file_sha256(path)
            lines = logged.get(A._rel(artifacts, path)) or []
            if digest is None or not lines or lines[-1][1] != digest:
                raise A.AuditError(f"{self.alias[sid]} of {audit_id}: its design is not the file the source's sampling "
                                   'logged, so its windows may not be the ones it drew')
            self.designs[sid] = A.read_json(path)
            self.design_sha256[sid] = digest

    def drew(self, sid: str) -> bool:
        """whether the source drew who-speaks windows from the session's population (it had a group microphone)"""
        design = self.designs.get(sid)
        return bool(design and design.get('mic'))

    def windows(self, sid: str, practice: bool) -> list[dict]:
        return [it for it in (self.designs.get(sid) or {}).get('speech', []) if bool(it.get('practice')) == practice]


def _on_grid(population: list[tuple[float, float]], start: float) -> tuple[float, float] | None:
    best = min(population, key=lambda w: abs(w[0] - start), default=None)
    return best if best is not None and abs(best[0] - start) <= A.MATCH_TOLERANCE else None


def reuse_window(session: SpeechSession, window: dict, population: list, source: SourceAudit) -> tuple[float, float]:
    """a source's window on the display grid: refused unless the source's microphone is the session's group
    microphone now, the window starts on the grid (within MATCH_TOLERANCE) and lies inside the population"""
    mic = (source.designs[session.id] or {}).get('mic') or {}
    alias = source.alias[session.id]
    if session.mic is None or os.path.realpath(mic.get('path', '')) != os.path.realpath(session.mic['path']):
        raise A.AuditError(f"{alias}: the group microphone of {source.id}'s windows is not the session's group "
                           'microphone now: its windows are not reused')
    starts = [float(a) for a in session.table['window_start']]
    if not any(abs(a - float(window['window_start'])) <= A.MATCH_TOLERANCE for a in starts):
        raise A.AuditError(f"{alias}: {window['item']} of {source.id} starts on no window of this version's table")
    found = _on_grid(population, float(window['window_start']))
    if found is None:
        raise A.AuditError(f"{alias}: {window['item']} of {source.id} lies outside the population (the sound, the "
                           'videos and the margin)')
    return found


# ---- drawing ----

def rank_lesson(windows: list[tuple], head: list[tuple], seed, audit_id: str, lesson: str) -> list[tuple]:
    """a lesson's windows in rank order: the `head` windows first, in a seeded order, then the rest in another;
    with the head a uniformly random subset of a simple random sample (a source's reliability windows), every
    prefix of the ranks is a simple random sample of the lesson"""
    def key(w):
        return (w[0], w[1])
    first = sorted(head, key=key)
    rest = sorted((w for w in windows if w not in head), key=key)
    A._rng(seed, audit_id, 'rank-rel', lesson).shuffle(first)
    A._rng(seed, audit_id, 'rank', lesson).shuffle(rest)
    return first + rest


def fresh_windows(members: list[str], pops: dict, size: int, seed, audit_id: str) -> list[tuple]:
    """`size` windows of a lesson, over its sessions in proportion to their populations, each session's drawn
    at random: (session, start, end, None)"""
    per = A.allocate(size, {sid: len(pops[sid]) for sid in members})
    out = []
    for sid in members:
        if per.get(sid):
            drawn = A._rng(seed, audit_id, 'speech', sid).sample(pops[sid], per[sid])
            out += [(sid, a, b, None) for a, b in sorted(drawn)]
    return out


def _apart(start: float, taken, gap: float) -> bool:
    return all(abs(start - t) >= gap for t in taken)


def draw_practice(ids: list[str], kinds: dict, pops: dict, items: dict, source, sessions: dict, sizes: dict, seed,
                  audit_id: str) -> dict[str, list[dict]]:
    """the practice windows per session: the source's, and one in the first session of each kind of sound they do
    not cover (without a source's, audit.PRACTICE_SPEECH in the first session of the first kind), apart from
    every item"""
    out: dict[str, list[dict]] = defaultdict(list)
    if sizes['practice'] <= 0:
        return out
    covered = set()
    for sid in ids:
        for window in (source.windows(sid, True) if source else []):
            a, b = reuse_window(sessions[sid], window, pops[sid], source)
            out[sid].append(_item(a, b, 'P', source_item=window['item'], source_sha=source.design_sha256[sid]))
            covered.add(kinds[sid][0])
    for kind in AUDIO_KINDS:
        first = next((sid for sid in ids if kinds[sid][0] == kind and pops[sid]), None)
        if kind in covered or first is None:
            continue
        n = 1 if any(out.values()) else A.PRACTICE_SPEECH
        taken = [it['window_start'] for it in items[first] + out[first]]
        candidates = [w for w in pops[first] if _apart(w[0], taken, sizes['gap'])]
        for a, b in sorted(A._rng(seed, audit_id, 'practice', first).sample(candidates, min(n, len(candidates)))):
            out[first].append(_item(a, b, 'P'))
        covered.add(kind)
    return out


def coder_labels(session_dir: Path, coders: list[str]) -> tuple[dict[str, dict[float, str]], dict[str, str | None]]:
    """per named coder, their last label per window start, and the sha256 of each coder's file; a coder whose file
    a model or a script wrote is refused"""
    from openmmla.analytics.interaction import labels as LB
    hashes = {}
    for coder in coders:
        path = Path(session_dir) / 'labels' / f'{coder}.jsonl'
        if path.exists() and LB.file_kind(path) == 'model':
            raise A.AuditError(f"{coder}'s labels of {Path(session_dir).name} were written by a model, not by the "
                               'coding page: not a coder of the social windows')
        hashes[coder] = L.file_sha256(path)
    try:
        frame = LB.load_labels(session_dir, all_coders=True)
    except ValueError as error:
        raise A.AuditError(f'the labels of {Path(session_dir).name} cannot be read: {error}') from None
    out: dict[str, dict[float, str]] = {coder: {} for coder in coders}
    for coder, start, label in zip(frame['coder'], frame['window_start'], frame['label']):
        if coder in out:
            out[coder][float(start)] = str(label)
    return out, hashes


def _label_at(labels: dict[float, str], start: float) -> str | None:
    near = min(labels, key=lambda t: abs(t - start), default=None)
    return labels[near] if near is not None and abs(near - start) <= A.MATCH_TOLERANCE else None


def social_pools(artifacts: Path, ids: list[str], pops: dict, taken: dict, coders: list[str],
                 gap: float) -> tuple[dict, dict, dict]:
    """(the windows labelled social by every named coder ('both') and by exactly one ('one'), each apart from the
    session's R and practice windows; the pools' sizes, total and eligible; the labels files' sha256 per session)"""
    pools: dict[str, list] = {a: [] for a in AGREEMENTS}
    sizes = {a: {'total': 0, 'eligible': 0} for a in AGREEMENTS}
    hashes = {}
    for sid in ids:
        labels, hashes[sid] = coder_labels(Path(artifacts) / sid, coders)
        for a, b in pops[sid]:
            social = sum(_label_at(labels[c], a) == SOCIAL for c in coders)
            agreement = 'both' if social == len(coders) else 'one' if social == 1 else None
            if agreement is None:
                continue
            sizes[agreement]['total'] += 1
            if _apart(a, taken[sid], gap):
                sizes[agreement]['eligible'] += 1
                pools[agreement].append((sid, a, b))
    return pools, sizes, hashes


def draw_social(pools: dict, sizes: dict, n: int, both: float, gap: float, seed, audit_id: str) -> tuple[list, dict]:
    """the social windows: round(n * both) from the 'both' pool and the rest from the 'one' pool, each in a seeded
    order and taken when at least `gap` from those already taken in its session; a shortfall of 'both' moves to
    'one'. ([(session, start, end, agreement)], what the draw wanted and got)"""
    wanted = {'both': int(math.floor(n * both + 0.5))}
    wanted['one'] = n - wanted['both']
    accepted: dict[str, list] = defaultdict(list)
    out, got = [], {}

    def take(agreement: str, want: int) -> int:
        order = sorted(pools[agreement])
        A._rng(seed, audit_id, 'social', agreement).shuffle(order)
        count = 0
        for sid, a, b in order:
            if count >= want:
                break
            if _apart(a, accepted[sid], gap):
                accepted[sid].append(a)
                out.append((sid, a, b, agreement))
                count += 1
        return count

    got['both'] = take('both', wanted['both'])
    moved = wanted['both'] - got['both']
    got['one'] = take('one', wanted['one'] + moved)
    summary = {'n': n, 'both_share': both, 'gap': gap, 'wanted': wanted, 'drawn': got, 'moved_to_one': moved,
               'shortfall': n - got['both'] - got['one'], 'pools': sizes,
               'pi': {a: (round(got[a] / sizes[a]['eligible'], 4) if sizes[a]['eligible'] else None) for a in AGREEMENTS}}
    return out, summary


def _item(start: float, end: float, stratum: str, **fields) -> dict:
    """a design item before its id: its window, its stratum ('R' random, 'S' social, 'P' practice) and fields"""
    return {'item': None, 'window_start': start, 'window_end': end, 'practice': stratum == 'P', 'reliability': False,
            'sweep': 1, 'stratum': stratum, 'rank': fields.get('rank'), 'weight': fields.get('weight'),
            'agreement': fields.get('agreement'), 'pool': fields.get('pool'),
            'source_item': fields.get('source_item'), 'source_design_sha256': fields.get('source_sha')}


def order_items(items: list[dict], seed, audit_id: str, sid: str, alias: str) -> list[dict]:
    """a recording's items with their ids, in the order the page shows them: sweep 1 (R and S shuffled together),
    then sweep 2, numbered <audit>-<alias>-t-NNN; then the practice items in time order, <audit>-<alias>-pt-NN"""
    regular = [it for it in items if not it['practice']]
    ordered = []
    for sweep in (1, 2):
        chosen = sorted((it for it in regular if it['sweep'] == sweep), key=lambda it: it['window_start'])
        A._rng(seed, audit_id, 'order', sid, sweep).shuffle(chosen)
        ordered += chosen
    practising = sorted((it for it in items if it['practice']), key=lambda it: it['window_start'])
    for n, it in enumerate(ordered, 1):
        it['item'] = f'{audit_id}-{alias}-t-{n:03d}'
    for n, it in enumerate(practising, 1):
        it['item'] = f'{audit_id}-{alias}-pt-{n:02d}'
    return ordered + practising


def view_item(item: dict) -> dict:
    """what the server may serve of an item before the render: no stratum, rank, weight or pipeline value"""
    return {'item': item['item'], 'part': 'speech', 'kind': TASK, 'practice': item['practice'],
            'reliability': item['reliability'], 'sweep': item['sweep'], 'render': None}


def sample_transcript(artifacts: Path, audit_id: str, sessions: list[SpeechSession], sizes: dict, version: str,
                      source: SourceAudit | None = None, coders=(), primary: str | None = None,
                      argv=None) -> tuple[dict, dict, dict]:
    """draw the items: (the plan, the view of each session, the design of each session)"""
    seed, margin, gap = sizes['seed'], sizes['margin'], sizes['gap']
    by_id = {s.id: s for s in sessions}
    if source:
        ids = [sid for sid in source.order if sid in by_id]
        alias = {sid: source.alias[sid] for sid in ids}
    else:
        ids = sorted(by_id)
        A._rng(seed, audit_id, 'aliases').shuffle(ids)
        alias = {sid: f'R{n + 1:02d}' for n, sid in enumerate(ids)}
    kinds = {sid: by_id[sid].audio_kind() for sid in ids}
    pops = {sid: by_id[sid].population(margin) for sid in ids}
    lessons: dict[str, list[str]] = defaultdict(list)
    for sid in ids:
        lessons[by_id[sid].lesson].append(sid)
    items: dict[str, list[dict]] = defaultdict(list)
    lesson_rows = {}
    for lesson, members in lessons.items():
        frame = [sid for sid in members if source and source.drew(sid)]
        reused = bool(frame)
        if reused:
            # the source's windows, its reliability windows ranked first
            windows, head = [], []
            for sid in frame:
                for window in source.windows(sid, False):
                    a, b = reuse_window(by_id[sid], window, pops[sid], source)
                    entry = (sid, a, b, window['item'])
                    windows.append(entry)
                    if window.get('reliability'):
                        head.append(entry)
            for sid in members:
                if sid not in frame:
                    print(f'{alias[sid]}: no random window: {source.id} drew its lesson from the other recordings')
        else:
            frame = [sid for sid in members if pops[sid]]
            windows = fresh_windows(frame, pops, sizes['speech'], seed, audit_id)
            head = A._rng(seed, audit_id, 'rank-head', lesson).sample(sorted(windows, key=lambda w: (w[0], w[1])),
                                                                      min(FRESH_HEAD, len(windows)))
        population = sum(len(pops[sid]) for sid in frame)
        ranked = rank_lesson(windows, head, seed, audit_id, lesson)
        lesson_rows[lesson] = {'population': population, 'items': len(ranked), 'reused': reused}
        for rank, (sid, a, b, from_item) in enumerate(ranked, 1):
            it = _item(a, b, 'R', rank=rank, weight=population / len(ranked), source_item=from_item,
                       source_sha=source.design_sha256[sid] if from_item else None)
            it['sweep'] = 1 if rank <= sizes['first_ranks'] else 2
            it['reliability'] = rank <= sizes['reliability_ranks']
            items[sid].append(it)
    practice = draw_practice(ids, kinds, pops, items, source, by_id, sizes, seed, audit_id)
    social, label_hashes = None, {}
    if sizes['social'] > 0:
        if not coders:
            raise A.AuditError('give --audit-social-coders NAME,NAME: whose social labels draw the S windows (or '
                               '--audit-social 0)')
        taken = {sid: [it['window_start'] for it in items[sid] + practice[sid]] for sid in ids}
        pools, pool_sizes, label_hashes = social_pools(artifacts, ids, pops, taken, list(coders), gap)
        drawn, social = draw_social(pools, pool_sizes, sizes['social'], sizes['social_both'], gap, seed, audit_id)
        social['coders'] = list(coders)
        chosen = []
        for sid, a, b, agreement in drawn:
            it = _item(a, b, 'S', agreement=agreement, pool=pool_sizes[agreement])
            items[sid].append(it)
            chosen.append(it)
        A._flag_share(chosen, sizes['reliability'], A._rng(seed, audit_id, 'reliability', 'S'), 'reliability')
        if social['moved_to_one'] or social['shortfall']:
            print(f"social: {social['drawn']['both']} of {social['wanted']['both']} windows both coders labelled social "
                  f"({social['moved_to_one']} moved to one coder's), {social['drawn']['one']} one coder's"
                  + (f"; {social['shortfall']} short" if social['shortfall'] else ''))
    plan_sessions, views, designs = [], {}, {}
    for sid in ids:
        s, a = by_id[sid], alias[sid]
        ordered = order_items(items[sid] + practice[sid], seed, audit_id, sid, a)
        kind, worn, share = kinds[sid]
        regular = [it for it in ordered if not it['practice']]
        designs[sid] = {'audit_id': audit_id, 'session': sid, 'alias': a, 'lesson': s.lesson, 'split': s.split,
                        'task': s.task, 'audio_kind': kind, 'display_version': version,
                        'videos': [_recording(v) for v in s.videos],
                        'mic': _recording(s.mic) if s.mic else None,
                        'sound': _recording(s.mic) if s.mic else dict(s.mix), 'worn': worn,
                        'worn_share': share, 'population': len(pops[sid]),
                        'lesson_population': lesson_rows[s.lesson]['population'],
                        'source': {'audit_id': source.id, 'plan_sha256': source.plan_sha256,
                                   'design_sha256': source.design_sha256.get(sid)} if source else None,
                        'labels': label_hashes.get(sid) or {}, 'speech': ordered,
                        'sources': {version: s.sources()}}
        views[sid] = {'audit_id': audit_id, 'alias': a, 'mode': 'blind', 'kind': TASK,
                      'items': [view_item(it) for it in ordered]}
        plan_sessions.append({'id': sid, 'alias': a, 'lesson': s.lesson, 'split': s.split, 'task': s.task,
                              'audio_kind': kind, 'population': len(pops[sid]),
                              'random': sum(1 for it in regular if it['stratum'] == 'R'),
                              'social': sum(1 for it in regular if it['stratum'] == 'S'),
                              'sweep_1': sum(1 for it in regular if it['sweep'] == 1),
                              'sweep_2': sum(1 for it in regular if it['sweep'] == 2),
                              'reliability': sum(1 for it in regular if it['reliability']),
                              'practice': sum(1 for it in ordered if it['practice'])})
    plan = {'audit_id': audit_id, 'task': TASK, 'created_at': C.now_utc(), 'seed': seed, 'mode': 'blind',
            'display_version': version, 'reveal_version': version, 'sizes': sizes, 'sessions': plan_sessions,
            'order': [alias[sid] for sid in ids], 'practice_sessions': [alias[sid] for sid in ids if practice.get(sid)],
            'source_audit': {'id': source.id, 'plan_sha256': source.plan_sha256} if source else None,
            'lessons': lesson_rows, 'social': social, 'primary': primary, 'declared_roles': [],
            'declared': DECLARED_TRANSCRIPT, 'normaliser': normaliser_state(), 'clip': CLIP_SPAN,
            'worn_min_share': WORN_MIN_SHARE, 'software': L.software(), 'argv': list(argv or [])}
    plan['estimate_hours'] = A.estimate(plan, designs.values())
    return plan, views, designs


# ---- a version's text, frozen ----

def text_of(lines: list[dict]) -> dict:
    """the window's words without speaker labels, as the content model read them: each line's in-window part, in
    window_text's time order, one per row"""
    parts = [line['inside'] for line in lines if line['inside']]
    return {'text': '\n'.join(parts), 'n_lines': len(parts), 'n_words': sum(len(p.split()) for p in parts),
            'n_approx': sum(1 for line in lines if line['inside'] and line['approximate'])}


def window_entry(records: list[dict], ws: float) -> tuple[dict, list[str]]:
    """(the text of [ws, ws + 10) and of the 10 s before it, as frozen; the words of its stamped lines)"""
    lines = T.window_text(records, ws, ws + WINDOW, CONTEXT)
    cur, prev = text_of(lines), text_of(T.window_text(records, ws - WINDOW, ws, CONTEXT))
    stamped = ' '.join(line['inside'] for line in lines if line['inside'] and not line['approximate']).split()
    return {'cur': cur['text'], 'prev': prev['text'], 'n_words': cur['n_words'], 'n_words_prev': prev['n_words'],
            'n_lines': cur['n_lines'], 'n_approx': cur['n_approx'],
            'sha': L.sha256_hex(cur['text'])[:16]}, stamped


class Reference:
    """the texts a content model read (JSON lines of session, window_start, source, cur and prev): a frozen
    window's text must equal its row of source 'all'"""

    def __init__(self, path):
        self.path = Path(os.path.expanduser(str(path))).resolve()
        self.sha256 = L.file_sha256(self.path)
        if self.sha256 is None:
            raise A.AuditError(f'no reference texts at {self.path}')
        self.rows: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
        for line in self.path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get('source') == 'all':
                self.rows[str(row.get('session'))].append((float(row['window_start']), row.get('cur') or '',
                                                           row.get('prev') or ''))

    def row(self, sid: str, ws: float) -> tuple[str, str] | None:
        near = min(self.rows.get(sid, []), key=lambda r: abs(r[0] - ws), default=None)
        return (near[1], near[2]) if near is not None and abs(near[0] - ws) <= A.MATCH_TOLERANCE else None


def row_sha(row: dict) -> str | None:
    """the hash of the text a content row scored (cur_sha, or cur_sha256 cut to its first 16 digits), None without one"""
    for column in SHA_COLUMNS:
        value = (row.get(column) or '').strip()
        if value:
            return value[:16]
    return None


class ContentArm:
    """a content model's scores (a CSV of session, window_start, source, context, no_text and the letter
    probabilities teacher, peer_task and peer_other, with cur_sha where it carries one: `hashed`), rows of source
    'all'"""

    def __init__(self, name: str, path):
        self.name, self.path = name, Path(os.path.expanduser(str(path))).resolve()
        self.sha256 = L.file_sha256(self.path)
        if self.sha256 is None:
            raise A.AuditError(f'no content scores of arm {name} at {self.path}')
        self.rows: dict[tuple[str, str], list[tuple[float, dict]]] = defaultdict(list)
        with self.path.open(encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            self.hashed = any(column in (reader.fieldnames or []) for column in SHA_COLUMNS)
            for row in reader:
                if row.get('source') == 'all':
                    key = (row['session'], str(int(float(row['context']))))
                    self.rows[key].append((float(row['window_start']), row))
        for key, rows in self.rows.items():
            starts = sorted(round(ws, 3) for ws, _ in rows)
            if len(set(starts)) != len(starts):
                raise A.AuditError(f'arm {name}: two rows for one window of {key[0]} (context {key[1]})')

    def contexts(self, sid: str) -> list[str]:
        return sorted(context for session, context in self.rows if session == sid)

    def row(self, sid: str, ws: float, context: str) -> dict | None:
        near = min(self.rows.get((sid, context), []), key=lambda r: abs(r[0] - ws), default=None)
        return near[1] if near is not None and abs(near[0] - ws) <= A.MATCH_TOLERANCE else None


def content_arms(spec: str | None) -> dict[str, ContentArm]:
    """--audit-content-scores ARM=FILE,ARM=FILE: each arm's scores"""
    arms = {}
    for part in (spec or '').split(','):
        if not part.strip():
            continue
        name, _, path = part.partition('=')
        if not name.strip() or not path.strip():
            raise A.AuditError('give --audit-content-scores as ARM=FILE,ARM=FILE')
        arms[name.strip()] = ContentArm(name.strip(), path.strip())
    return arms


def _score(row: dict) -> dict:
    """a content row's scores: no_text gives 0 (the model was not asked); a score missing otherwise is None"""
    no_text = int(float(row.get('no_text') or 0))

    def value(name):
        if no_text:
            return 0.0
        number = T._float(row.get(name))
        return None if number is None else number
    return {'teacher': value('teacher'), 'peer_task': value('peer_task'), 'peer_other': value('peer_other'),
            'no_text': no_text}


def freeze_transcript(audit_id: str, design: dict, version: str, session: SpeechSession,
                      reference: Reference | None = None, content: dict | None = None) -> dict:
    """version `version`'s text and speech measures for every item of a session (practice included), checked:
    the timed words inside each window must be the words of its text; with `reference` every text must be its
    row's; every content arm's cur_sha must be the frozen sha. TranscriptRefusal when a check fails."""
    table = session.table
    starts = [float(v) for v in table['window_start']]
    words_columns = [f'p{t}_words' for t in session.kept if f'p{t}_words' in table.columns]
    worn = bool(words_columns) and bool((session.params.get('speech') or {}).get('wearers'))
    group_records = [r for r in session.records if T.source_of(r) == 'group mic']
    out, off_grid, matched, differ, missing, sha_off = {}, [], 0, [], [], []
    # per arm the rows found and missing, and of an arm that carries cur_sha the rows whose hash was checked
    used = {name: {'rows': 0, 'missing': 0, 'hashed': arm.hashed, 'checked': 0} for name, arm in (content or {}).items()}
    for it in design['speech']:
        ws = float(it['window_start'])
        consumed, stamped = window_entry(session.records, ws)
        timed = timed_words(session.records, ws)
        inside = Counter(token for moment, _, _, word, _, _ in timed if ws <= moment < ws + WINDOW
                         for token in word.split())
        if inside != Counter(stamped):
            raise TranscriptRefusal(f"the timed words of {it['item']} are not the words of its text "
                                    f"({sum(inside.values())} against {len(stamped)}): the reader has drifted from code_text")
        entry: dict[str, Any] = {'row': False}
        at = min(range(len(starts)), key=lambda i: abs(starts[i] - ws), default=None)
        if at is not None and abs(starts[at] - ws) <= A.MATCH_TOLERANCE:
            found = table.iloc[at]

            def value(name):
                v = found.get(name) if name in table.columns else None
                return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
            member = [value(c) for c in words_columns]
            entry = {'row': True, 'window_start': starts[at], 'speech_ratio': value('speech_ratio'),
                     'words': value('words'), 'n_asr_recognition': value('n_asr_recognition'), 'worn': worn,
                     'member_words': sum(v for v in member if v is not None) if worn else None}
        else:
            off_grid.append(it['item'])
        entry['asr'] = {'consumed': consumed,
                        'group': window_entry(group_records, ws)[0] if group_records else None,
                        'timed': [[word, round(moment - ws, 3), None if end is None else round(end - ws, 3), channel]
                                  for moment, _, _, word, end, channel in timed]}
        if reference is not None:
            row = reference.row(session.id, ws)
            if row is None:
                missing.append(it['item'])
            elif row != (consumed['cur'], consumed['prev']):
                differ.append(it['item'])
            else:
                matched += 1
        if content:
            scores = {}
            for name, arm in content.items():
                for context in arm.contexts(session.id):
                    row = arm.row(session.id, ws, context)
                    if row is None:
                        used[name]['missing'] += 1
                        continue
                    used[name]['rows'] += 1
                    if arm.hashed:
                        # a file that carries the hash must carry it on every row: a row without one is not checked
                        if row_sha(row) != consumed['sha']:
                            sha_off.append((name, it['item']))
                        else:
                            used[name]['checked'] += 1
                    scores.setdefault(name, {})[context] = _score(row)
            entry['content'] = scores
        out[it['item']] = entry
    if reference is not None and (differ or missing):
        raise TranscriptRefusal(f"{len(differ)} of {len(design['speech'])} windows differ from the reference texts and "
                                f'{len(missing)} have no row there: this version is not the text the reference read')
    if sha_off:
        arms = sorted({name for name, _ in sha_off})
        raise TranscriptRefusal(f"{len(sha_off)} content rows ({', '.join(arms)}) carry no cur_sha, or one that is not "
                                'the frozen text\'s: the scores are of another text')
    sources = session.sources()
    if content:
        sources['content'] = {name: {'file': str(arm.path), 'sha256': arm.sha256, **used[name]}
                              for name, arm in content.items()}
    if reference is not None:
        sources['reference'] = {'file': str(reference.path), 'sha256': reference.sha256, 'matched': matched,
                                'of': len(design['speech'])}
    return {'audit_id': audit_id, 'session': session.id, 'version': version, 'task': TASK, 'frozen_at': C.now_utc(),
            'sources': sources, 'kept': session.kept, 'group_channel': bool(group_records), 'off_grid': off_grid,
            'speech': out}


# ---- the commands ----

def _print_plan(plan: dict) -> None:
    print(f"transcription audit {plan['audit_id']} (display version {plan['display_version']}"
          + (f", windows of {plan['source_audit']['id']}" if plan.get('source_audit') else '') + f"): "
          f"{len(plan['sessions'])} sessions")
    for s in plan['sessions']:
        print(f"  {s['alias']} {s['split'] or '-'} {s['audio_kind']}: {s['random']} random, {s['social']} social "
              f"(sweep 1 {s['sweep_1']}, sweep 2 {s['sweep_2']}), {s['reliability']} reliability"
              + (f", {s['practice']} practice" if s['practice'] else ''))
    print('estimate: ' + ', '.join(f'{k} {v} h' for k, v in plan['estimate_hours'].items()))


def _sessions(artifacts: Path, args, source: SourceAudit | None, sources: dict) -> list[SpeechSession]:
    """the sessions drawn from: with a source, its sessions (or those --audit-sessions names of them), every one of
    which must load, since the source's windows are never thinned; else the candidates, a failing one left out"""
    if source:
        named = [s.strip() for s in (args.audit_sessions or '').split(',') if s.strip()]
        stray = [sid for sid in named if sid not in source.alias]
        if stray:
            raise A.AuditError(f'{stray[0]} is not a session of {source.id}, whose aliases and order the audit takes')
        ids = named or list(source.order)
    else:
        ids = A.candidate_sessions(artifacts, args)
    sessions = []
    for sid in ids:
        try:
            session = load_session(artifacts, sid, sources)
            if session.problems and not args.audit_allow_drift:
                raise A.AuditError(f"{'; '.join(session.problems)} (--audit-allow-drift keeps it, recorded)")
        except A.AuditError as error:
            if source:
                raise A.AuditError(f"{source.alias[sid]} ({sid}) cannot be audited: {error}; the source's windows are "
                                   'not thinned (--audit-sessions names the sessions to keep)') from None
            print(f'left out: {sid}: {error}')
            continue
        sessions.append(session)
    if not sessions:
        raise A.AuditError('no session can be audited')
    return sessions


def cmd_sample(args, argv) -> int:
    artifacts, audit_id = A._artifacts(args), args.audit_sample
    if args.audit_version != PRIMARY_VERSION:
        raise A.AuditError(f'give --audit-version {PRIMARY_VERSION}: the declared analyses take it as the primary '
                           'version and the one the reveal shows; freeze the other with --audit-freeze after the sample')
    folder = A.audit_dir(artifacts, audit_id)
    if (folder / A.PLAN_FILE).exists():
        raise A.AuditError(f'{folder / A.PLAN_FILE} exists already: an audit is drawn once')
    if not args.audit_asr_events:
        raise A.AuditError(NO_EVENTS_FLAG)
    sizes = sizes_of(args)
    normaliser_state()
    primary = _name(args.audit_primary, '--audit-primary') if args.audit_primary else None
    coders = [c.strip() for c in (args.audit_social_coders or '').split(',') if c.strip()]
    source = SourceAudit(artifacts, args.audit_windows_from) if args.audit_windows_from else None
    sources = version_sources(args)
    reference = Reference(args.audit_asr_reference) if args.audit_asr_reference else None
    content = content_arms(args.audit_content_scores)
    sessions = _sessions(artifacts, args, source, sources)
    plan, views, designs = sample_transcript(artifacts, audit_id, sessions, sizes, args.audit_version, source, coders,
                                             primary, argv)
    by_id = {s.id: s for s in sessions}
    frozen = {}
    for sid, design in designs.items():
        try:
            frozen[sid] = freeze_transcript(audit_id, design, args.audit_version, by_id[sid], reference, content)
        except A.AuditError as error:
            raise A.AuditError(f"the display version cannot be frozen for {design['alias']}: {error}; nothing was "
                               'written') from None
    written = []
    for sid, design in designs.items():
        here = A.session_audit_dir(artifacts, sid, audit_id)
        target = A.version_file(artifacts, sid, audit_id, args.audit_version)
        A.write_json(here / A.DESIGN_FILE, design)
        A.write_json(here / A.VIEW_FILE, views[sid])
        write_private_json(target, frozen[sid])
        written += [here / A.DESIGN_FILE, here / A.VIEW_FILE, target]
    A.write_json(folder / A.PLAN_FILE, plan)
    campaign = L.Campaign(folder)
    campaign.update(lambda d: d.update({
        'campaign': audit_id, 'kind': 'audit', 'audit_id': audit_id, 'created_at': plan['created_at'],
        'artifacts': str(artifacts), 'window': WINDOW, 'step': WINDOW, 'seed': sizes['seed'],
        'sessions': [{'id': s['id'], 'alias': s['alias']} for s in plan['sessions']], 'cookie_days': L.COOKIE_DAYS,
        'closed_at': None, 'released_at': None, 'coders': []}), create=True)
    A._event(artifacts, audit_id, 'audit-sample', plan_sha256=L.file_sha256(folder / A.PLAN_FILE), task=TASK,
             version=args.audit_version, sessions=len(designs), files=A.file_hashes(artifacts, written), argv=list(argv))
    _print_plan(plan)
    return 0


def read_version(plan: dict, version: str) -> bool:
    """whether `version` is the text a content model read (every version but the display one): it is frozen only
    against the texts that model read and the cur_sha of its scores, checks no flag overrides"""
    return version != plan.get('display_version', PRIMARY_VERSION)


def read_problems(data: dict, design: dict) -> list[str]:
    """why a read version's frozen file was not checked against what the content model read (none when it was):
    every window's text against its reference row, and a content arm with cur_sha checked on every row it has"""
    sources = data.get('sources') or {}
    reference, n = sources.get('reference') or {}, len(design['speech'])
    out = []
    if not reference:
        out.append('its text was checked against no reference texts')
    elif reference.get('matched') != n or reference.get('of') != n:
        out.append(f"{reference.get('matched')} of its {n} windows matched the reference texts")
    if not any(arm.get('hashed') and arm.get('checked') and arm.get('checked') == arm.get('rows')
               for arm in (sources.get('content') or {}).values()):
        out.append("no content arm's cur_sha was checked")
    return out


def cmd_freeze(args, argv, plan: dict) -> int:
    artifacts, audit_id, version = A._artifacts(args), args.audit_freeze, args.audit_version
    if not version:
        raise A.AuditError('give --audit-version reported|rerun')
    if not args.audit_asr_events:
        raise A.AuditError(NO_EVENTS_FLAG)
    reference = Reference(args.audit_asr_reference) if args.audit_asr_reference else None
    content = content_arms(args.audit_content_scores)
    if read_version(plan, version) and (reference is None or not any(arm.hashed for arm in content.values())):
        # without both, a wrong pattern or a retry after a refusal would freeze an unchecked text as the one read
        raise A.AuditError(f'{version} is the text the content model read: it is frozen only against the texts that '
                           'model read (--audit-asr-reference FILE) and the cur_sha of its scores (--audit-content-scores '
                           'ARM=FILE, a file with a cur_sha column), checks no flag overrides')
    answered = A.after_answers(artifacts, plan, args, 'freezing a version')
    rendered = (A.audit_dir(artifacts, audit_id) / A.RENDER_INDEX).exists()
    sources = version_sources(args)
    before = A.logged_files(artifacts, audit_id)
    frozen, refused, skipped, written, refusals = 0, 0, 0, [], []
    for entry in plan['sessions']:
        sid, alias = entry['id'], entry['alias']
        design = A.read_json(A.session_audit_dir(artifacts, sid, audit_id) / A.DESIGN_FILE)
        target = A.version_file(artifacts, sid, audit_id, version)
        earlier = [seq for seq, _, _ in before.get(A._rel(artifacts, target), [])]
        if earlier:
            # frozen and logged once: a file deleted and frozen again would be a version chosen afterwards
            print(f'{alias}: {version} was frozen at seq {earlier[0]} of the log: a version is frozen once')
            skipped += 1
            continue
        if target.exists() and not A.read_json(target).get('refused'):
            print(f'{alias}: {target.name} exists already: a version is frozen once')
            skipped += 1
            continue
        # a refused session may be frozen again (its refusal stays in the log); the read version only through its
        # checks, which every freeze of it runs
        stub = {'audit_id': audit_id, 'session': sid, 'version': version, 'task': TASK}
        try:
            session = load_session(artifacts, sid, sources)
            if session.problems and not args.audit_allow_drift:
                raise A.AuditError('; '.join(session.problems))
            data = freeze_transcript(audit_id, design, version, session, reference, content)
        except A.AuditError as error:
            write_private_json(target, {**stub, 'refused': str(error),
                                        'no_override': isinstance(error, TranscriptRefusal)})
            refusals.append(target)
            print(f'{alias}: refused: {error}')
            refused += 1
            continue
        except Exception as error:  # one session's failure leaves it out, not the others
            write_private_json(target, {**stub, 'refused': f'{type(error).__name__}: {str(error)[:300]}'})
            refusals.append(target)
            print(f'{alias}: refused: {type(error).__name__}: {error}')
            refused += 1
            continue
        data.update(after_answers=answered, after_render=rendered)
        write_private_json(target, data)
        written.append(target)
        frozen += 1
        print(f'{alias}: {version} frozen' + (f" ({len(data['off_grid'])} windows on no row of its table)"
                                              if data['off_grid'] else ''))
    A._event(artifacts, audit_id, 'audit-freeze', task=TASK, version=version, frozen=frozen, refused=refused,
             skipped=skipped, after_answers=answered, after_render=rendered, files=A.file_hashes(artifacts, written),
             refused_files=A.file_hashes(artifacts, refusals),
             reference_sha256=reference.sha256 if reference else None,
             content={name: arm.sha256 for name, arm in content.items()}, argv=list(argv))
    return 0 if frozen else 1


def unfrozen(artifacts: Path, plan: dict) -> list[str]:
    """why the render may not run yet: per recording and version, a file no logged step froze nor refused"""
    logged = A.logged_files(artifacts, plan['audit_id'])
    missing = []
    for entry in plan['sessions']:
        for version in A.VERSIONS:
            path = A.version_file(artifacts, entry['id'], plan['audit_id'], version)
            if not path.exists():
                missing.append(f"{entry['alias']} {version}")
            elif not A.read_json(path).get('refused') and not logged.get(A._rel(artifacts, path)):
                missing.append(f"{entry['alias']} {version} (frozen by no logged step)")
    return missing


# ---- the blind close and the references ----

def blind_closed_path(artifacts: Path, audit_id: str) -> Path:
    return A.audit_dir(artifacts, audit_id) / BLIND_CLOSED


def read_blind_closed(artifacts: Path, audit_id: str) -> dict[str, dict]:
    """{name: {'seq', 'at'}} of the auditors whose blind pass is closed (read under the request log's lock when a
    save depends on it)"""
    path = blind_closed_path(artifacts, audit_id)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    except ValueError:
        raise A.AuditError(f'{path} does not parse: whose blind pass is closed is not known') from None
    return data if isinstance(data, dict) else {}


def logged_closes(artifacts: Path, audit_id: str) -> tuple[dict[str, dict], str | None]:
    """({name: {'seq', 'at'}} of the blind closes the request log holds, each name's first; the sha256 of
    blind_closed.json the last of them wrote): the closes as the operator's logged steps made them"""
    path = A.audit_dir(artifacts, audit_id) / L.LOG_FILE
    closes, last = {}, None
    try:
        data = path.read_bytes()
    except OSError:
        return closes, last
    for line in data.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get('event') != CLOSE_EVENT or not isinstance(record.get('auditor'), str):
            continue
        closes.setdefault(record['auditor'], {'seq': record.get('seq'), 'at': record.get('t')})
        last = record.get('file_sha256')
    return closes, last


def close_problems(closed: dict, logged: dict) -> list[str]:
    """why blind_closed.json (`closed`) is not the closes the request log holds (`logged`): a name closed in one
    and not in the other, or at another seq"""
    out = []
    for name in sorted(set(closed) | set(logged)):
        entry = closed.get(name)
        seq = entry.get('seq') if isinstance(entry, dict) else None
        if name not in logged:
            out.append(f'{name} is closed in {BLIND_CLOSED} (seq {seq}) by no logged {CLOSE_EVENT}')
        elif name not in closed:
            out.append(f"{name}'s logged close (seq {logged[name]['seq']}) is not in {BLIND_CLOSED}")
        elif seq != logged[name]['seq']:
            out.append(f"{name} is closed at seq {seq} in {BLIND_CLOSED}, at seq {logged[name]['seq']} in the log")
    return out


def _links(campaign: dict) -> dict[str, str]:
    """token id -> name of every claimed audit link"""
    return {t['token_id']: c['name'] for c in campaign.get('coders', []) for t in c.get('tokens', [])
            if t.get('scope') == 'audit' and t.get('claimed_at')}


def blind_lines(artifacts: Path, plan: dict, campaign: dict, name: str, before: int | None) -> dict[str, dict]:
    """item -> the name's last transcription line saved before `before` (its close's seq; every line when None),
    of the lines the open page saved under the name or a claimed audit link of the name saved"""
    links = _links(campaign)
    out: dict[str, dict] = {}
    for _, path in A.record_files(artifacts, plan['audit_id'], [s['id'] for s in plan['sessions']]):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or record.get('auditor') != name or record.get('phase') != 'transcribe':
                continue
            linked = links.get(str(record.get('token_id'))) == name
            typed = record.get('open') is True and record.get('token_id') is None
            seq = record.get('request_seq')
            if not (linked or typed) or not isinstance(seq, int) or (before is not None and seq >= before):
                continue
            if record.get('item') not in out or seq > out[record['item']]['request_seq']:
                out[record['item']] = record
    return out


def cmd_close_blind(args, argv) -> int:
    artifacts, audit_id = A._artifacts(args), args.audit_close_blind
    plan = A.load_plan(artifacts, audit_id)
    if not transcript_plan(plan):
        raise A.AuditError(f'{audit_id} is a sensing audit: a blind pass is closed in a transcription audit')
    name = _name(args.audit_auditor)
    folder = A.audit_dir(artifacts, audit_id)
    campaign = L.Campaign(folder)
    data = campaign.data()
    if data.get('closed_at'):
        raise A.AuditError(f"the audit was closed at {data['closed_at']}")
    check = L.checked_log(campaign, artifacts, bool(getattr(args, 'despite_log_failure', False)), 'close')
    if not blind_lines(artifacts, plan, data, name, None):
        raise A.AuditError(f'{name} saved no transcription in audit {audit_id}: check the name (typed names are '
                           'taken in NFC, as the open page takes them)')
    path = blind_closed_path(artifacts, audit_id)
    log = L.RequestLog(folder / L.LOG_FILE)
    try:
        with log.held():
            # under the lock a save of the server also holds: a save is judged wholly before or after the close
            closed = read_blind_closed(artifacts, audit_id)
            logged, _ = logged_closes(artifacts, audit_id)
            if name in logged:
                raise A.AuditError(f"{name}'s blind pass was closed at seq {logged[name]['seq']}")
            others = close_problems({k: v for k, v in closed.items() if k != name}, logged)
            if others:
                raise A.AuditError('; '.join(others) + f': {BLIND_CLOSED} is not the closes the log holds (close a name '
                                   'it holds unlogged with this step; put back by hand what the log holds)')
            # a close the file holds and the log does not (a step stopped between the two writes) is logged now, at
            # this step's seq
            unlogged = closed.get(name)
            closed[name] = {'seq': log.next_seq(), 'at': C.now_utc()}
            A.write_json(path, closed)
            entry = {'event': CLOSE_EVENT, 'auditor': name, 'file_sha256': L.file_sha256(path), 'pid': os.getpid(),
                     'log_check': check, 'argv': list(argv)}
            if unlogged is not None:
                entry['unlogged_seq'] = unlogged.get('seq') if isinstance(unlogged, dict) else None
            line = log.write(entry, sync=True)
    finally:
        log.close()
    print(f"{name}'s blind pass of audit {audit_id} is closed at seq {line['seq']}"
          + (f" (its close at seq {line['unlogged_seq']} was in {BLIND_CLOSED} but not in the log)"
             if 'unlogged_seq' in line else '')
          + ': the page refuses their transcriptions from now on and, if they answer the full audit, shows them the '
          "text the content model read (the declared rule: the primary is closed only once the second auditor's blind pass is done, else a logged "
          'note says why)')
    return 0


def cmd_export_references(args, argv) -> int:
    artifacts, audit_id = A._artifacts(args), args.audit_export_references
    plan = A.load_plan(artifacts, audit_id)
    if not transcript_plan(plan):
        raise A.AuditError(f'{audit_id} is a sensing audit: references are exported from a transcription audit')
    name = _name(args.audit_auditor)
    if not args.audit_out:
        raise A.AuditError('give --audit-out FILE: where the references go (mode 0600)')
    out = Path(os.path.expanduser(args.audit_out)).resolve()
    if out.exists():
        raise A.AuditError(f'{out} exists already: give another --audit-out')
    folder = A.audit_dir(artifacts, audit_id)
    campaign = L.Campaign(folder)
    check = L.checked_log(campaign, artifacts, bool(getattr(args, 'despite_log_failure', False)), 'export')
    # the close as the operator's logged step made it (the scorer counts by the same)
    closed = logged_closes(artifacts, audit_id)[0].get(name)
    lines = blind_lines(artifacts, plan, campaign.data(), name, closed['seq'] if closed else None)
    try:
        from openmmla.commands.ses import audit_text
        llm = getattr(audit_text, 'llm_form', None)
    except ImportError:
        llm = None
    rows, versions = [], set()
    for entry in plan['sessions']:
        design = A.read_json(A.session_audit_dir(artifacts, entry['id'], audit_id) / A.DESIGN_FILE)
        frozen = {}
        for version in A.VERSIONS:
            path = A.version_file(artifacts, entry['id'], audit_id, version)
            data = A.read_json(path) if path.exists() else {}
            if data and not data.get('refused'):
                frozen[version] = data
        for it in design['speech']:
            record = lines.get(it['item'])
            if it['practice'] or record is None:
                continue
            answer = record.get('answer') or {}
            asr = {}
            for version, data in frozen.items():
                consumed = ((data['speech'].get(it['item']) or {}).get('asr') or {}).get('consumed')
                if consumed:
                    asr[version] = {'prev': consumed['prev'], 'cur': consumed['cur'], 'sha': consumed['sha']}
                    versions.add(version)
            row = {'item': it['item'], 'alias': entry['alias'], 'session': entry['id'],
                   'window_start': it['window_start'], 'stratum': it['stratum'], 'status': answer.get('status'),
                   'transcript': answer.get('transcript'), 'flag': bool(answer.get('flag')),
                   'request_seq': record['request_seq'], 'asr': asr}
            if llm is not None and isinstance(answer.get('transcript'), str):
                try:
                    row['llm'] = llm(answer['transcript'])
                except ValueError:  # a line the page's check let through and the grammar refuses: no LLM form
                    row['llm'] = None
            rows.append(row)
    body = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as file:
        file.write(body)
        file.flush()
        os.fsync(file.fileno())
    digest = L.sha256_hex(body)
    A._event(artifacts, audit_id, 'audit-export-references', auditor=name, file=str(out), sha256=digest,
             items=len(rows), closed_seq=closed['seq'] if closed else None, versions=sorted(versions),
             llm_form=llm is not None, log_check=check, argv=list(argv))
    print(f"{len(rows)} references of {name} written to {out} (sha256 {digest})"
          + ('' if closed else f": {name}'s blind pass is not closed, so every transcription line counts"))
    return 0
