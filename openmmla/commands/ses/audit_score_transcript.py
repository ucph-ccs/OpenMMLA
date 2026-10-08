"""The scores of a transcription audit (mmla ses-code --audit-score ID --audit-version rerun|reported, for a plan
whose task is 'transcript'; audit_score hands it over): one frozen version's text, audit_speech's
pipeline_<version>.json, against the primary auditor's blind transcripts and judgements.

What counts. The answers are those audit_score.read_answers counts: the lines a claimed audit link saved for its
own auditor and, in an open audit, those the open page saved, under the name typed. Of an item's transcription
the last line saved before its auditor's blind close counts (the close's own line in the request log,
--audit-close-blind: blind_closed.json, which the page acted on, must agree with the log, else every header
names where it does not); a line saved at or after the close, which the page refuses, is left out and counted. A
reveal rating counts only when saved after its auditor's close (the last such line), and is a rating of the
plan's reveal version, whichever version is scored. Nothing is scored before the primary's close unless
--audit-interim, which the score's log line and every header name. The request log must verify with the
answers files (code_locked.checked_log), each frozen file must be the one its freeze logged
(audit_score.frozen_checks), the version the content model read (reported) must have been frozen against the
texts it read and a content arm's cur_sha in every session (audit_speech.read_problems: else it is not scored, nor
scored beside), and the normaliser must be the one the plan froze (--audit-allow-drift scores with another,
named). Practice items, items not rendered, not answered or flagged, items the version holds no text of
and transcripts the grammar (audit_text) now refuses are left out of every analysis and counted; so are the
lines after a close, and the skips (an R item left unanswered before an item answered after it, in the order the
page served its recording: sweep by sweep, each in the view's seeded order) are counted.

The primary auditor is --audit-auditor, else the plan's primary (--audit-primary at sampling), else the one name
whose answers chose the full audit (audit_score's rule, open or with links); the names whose answers chose the
reliability subset give the agreement and any other name is counted and left out. Answers of two auditors are
never averaged or merged. An operator's note of the roles (--log-note, its text naming the roles) is expected
before the first scored answer; every header warns when there is none. In an open audit the names were typed,
not authenticated, and every header says so.

An item's counts, never a word of either text: the primary's transcript and the version's text, each normalised
by N1 (audit_text) and aligned, for each source of the text (consumed, every source: the text the content
model read; group, the group microphone alone): the reference words N, the hits, substitutions, deletions and
insertions, the same per reference tag and over the characters (CER), with N0, with the edge tolerance (the
timed words starting in [-0.5, 10.5) s, those within 0.5 s of an edge and those starting before it that end
inside the window optional), with split and merge tolerated, and the least over the orders of at most four
lines when the auditor ticked overlap; the content words both share; then the answers (status, overlap, who,
adult, peer, other_offtask), the reveal ratings (gist, invented), the version's table measures and the content
scores. A content arm's scores of the version's text are the frozen ones (read again from their file, while
it is the one frozen, for the letters the frozen values lack) or those of a file --audit-content-scores names
for the version scored: every row's cur_sha must be the frozen text's sha, else the arm is refused with the
count, no flag overriding it. The scores of the primary's references (--audit-content-scores-ref ARM=FILE:
rescore.py's CSV, the frozen scorer run again on the exported transcripts) must carry the hashes of the
counted transcript (and its request seq) and of the version's text, else that variant is refused with the count.

The analyses are the plan's declared ones (audit_speech.DECLARED_TRANSCRIPT): P1 to P4 on the version scored
(rerun, V1, is the declared primary; reported, V0, gives the secondary row), S1 to S13 and the sensitivities;
the other version, when frozen and checked, is scored beside it with the paired differences V1 - V0. An R
item weighs its lesson's population over the lesson's R items kept here; S items carry no weight and enter the
conditional analyses only. WER, CER and every rate are ratios of sums (sum w E over sum w N, over the items
with N >= 1), C the within-lesson concordance (the pairs of a positive and a negative of one lesson, a tie a
half) with the AUROC over every pair beside it, TR = (C_ASR - 0.5) / (C_ref - 0.5) when C_ref > 0.55, then
Cohen's kappa and Lin's CCC. Every pooled value has a lesson-cluster bootstrap interval
(metrics.session_bootstrap, BOOT draws unless --audit-boot, the undefined draws counted), every difference
paired draws (metrics.paired_delta), each primary its leave-one-lesson-out range; the per-lesson rows give
counts and point values only, and no p-value is given.

The outputs go to artifacts/runtime/audit/<id>/scores/<version>_<UTC time>/ and hold counts only, never a word
of a transcript or of the version's text: report.txt, units_<version>.csv (per item and source),
pooled_<check>.csv and lesson_<check>.csv per check, interauditor.csv, compare_<audit>.csv (--audit-compare-with:
a sensing audit's who-speaks answers joined on the session and the window start, its designs and its request log
verified first, read only), content_<arm>.csv per content arm and summary.json, each headed by the plan's
sha256 and its anchors' sha256, the version, the primary, the roles, the blind closes, --audit-interim, the
deviations and the lines left out.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from openmmla.commands.ses import audit as A
from openmmla.commands.ses import audit_score as S
from openmmla.commands.ses import audit_speech as SP
from openmmla.commands.ses import audit_text as X
from openmmla.commands.ses import code as C
from openmmla.commands.ses import code_locked as L

L.register_loaded(__name__, __file__)

# the bootstrap's draws unless --audit-boot says otherwise
BOOT = 10000
# the edge tolerance: within this many seconds of either edge of the window a timed word is optional
EDGE = 0.5
# the transfer ratio is not estimable at or below this C_ref
TR_FLOOR = 0.55
# with fewer peer = other items P2a and P3 of peer_other are descriptive only
MIN_POSITIVES = 15
# the words feature's Spearman takes the lessons of at least this many items
MIN_LESSON_ITEMS = 5
# with overlap ticked, the orders of at most this many lines are tried
OVERLAP_LINES = 4
# the content arm of P2 and P3; the others are S12's
PRIMARY_ARM = 'qwen'
SOURCES = ('consumed', 'group')
WHO = S.SPEAKERS
ANSWERED_WHO = WHO + ('cannot_tell',)
WHO_OF_TAG = {'M': 'member', 'T': 'adult', 'O': 'other_group', '?': 'cannot_tell'}
# a reference tag as part of a column's name
TAG_KEY = {'M': 'M', 'T': 'T', 'O': 'O', '?': 'Q'}
PEER = ('task', 'other', 'none', 'cannot_tell')
GIST = ('yes', 'partly', 'no', 'nothing_said')
INVENTED = ('none', 'some', 'most', 'cannot_tell')
SCORES = ('teacher', 'peer_task', 'peer_other')
LETTER_FIELDS = ('teacher', 'teacher_pB', 'teacher_pC', 'peer_task', 'peer_other', 'peer_pC', 'peer_pD')
# the peer question's options as the content model's letters, in its order
PEER_OPTIONS = (('A', 'peer_task'), ('B', 'peer_other'), ('C', 'peer_pC'), ('D', 'peer_pD'))
# the questions of P2: (name, the score, the answer, its value of the positives, of the negatives)
QUESTIONS = (('P2a', 'peer_other', 'peer', 'other', 'task'), ('P2b', 'teacher', 'adult', 'yes', 'no'),
             ('P2c', 'peer_task', 'peer', 'task', 'none'))
# rescore.py's variants: the reference in its LLM form with the version's prev (ctx1) or without one (ctx0), the
# version's own text again (reproducibility), and both in N1's surface form; per variant (the hash of cur, of
# prev or None, whether the row names the transcript, the context)
REF_VARIANTS = {'ref_ctx1': ('llm', 'prev', True, '1'), 'ref_ctx0': ('llm', None, True, '0'),
                'asr_ctx1': ('cur', 'prev', False, '1'), 'ref_n1': ('ref_n1', 'prev_n1', True, '1'),
                'asr_n1': ('cur_n1', 'prev_n1', False, '1')}
# an operator's note naming the roles
ROLES = re.compile(r'\brole', re.IGNORECASE)
VERSION_NAMES = {'rerun': 'V1 rerun', 'reported': 'V0 reported'}
CHECKS = (('accuracy', 'ACCURACY (P1)'), ('detection', 'DETECTION (S1)'), ('sources', 'SOURCES (S4)'),
          ('group', 'THE GROUP SOURCE AND THE KIND OF SOUND (S2)'), ('words', 'WORDS FEATURE (S5)'),
          ('content', 'CONTENT (P2, P3, S7, S12)'), ('who', 'WHO SPEAKS (P4, S10)'),
          ('reveal', 'REVEAL (S11)'), ('coders', "THE CODERS' LABELS AGAINST THE HEARD TOPIC (S8)"),
          ('social', 'THE SOCIAL STRATUM ALONE (S13)'), ('versions', 'V1 AGAINST V0 (S3)'),
          ('sensitivity', 'SENSITIVITIES'))
POOLED_COLUMNS = ['check', 'version', 'analysis', 'arm', 'by', 'group', 'metric', 'value', 'n', 'positives',
                  'negatives', 'lessons', 'lo', 'hi', 'interval', 'lolo_lo', 'lolo_hi', 'band']
LESSON_COLUMNS = ['check', 'version', 'analysis', 'arm', 'lesson', 'aliases', 'metric', 'value', 'n', 'num', 'den']
INTERAUDITOR_COLUMNS = ['auditor', 'question', 'n', 'agree', 'kappa', 'value', 'lo', 'hi', 'interval']
UNIT_FIELDS = ['item', 'alias', 'session', 'lesson', 'split', 'stratum', 'sweep', 'rank', 'reliability', 'agreement',
               'audio_kind', 'window_start', 'weight', 'status', 'overlap', 'who', 'adult', 'peer', 'other_offtask',
               'lines', 'markers_x', 'markers_bg', 'dominant', 'n0_ref', 'gist', 'invented', 'row', 'speech_ratio',
               'words', 'n_asr_recognition', 'member_words']
COUNT_FIELDS = (['N', 'H', 'S', 'D', 'I', 'E'] + [f'{k}_{TAG_KEY[t]}' for t in X.TAGS for k in ('N', 'H', 'S', 'D')]
                + ['cN', 'cH', 'cS', 'cD', 'cI', 'cE', 'hyp', 'hyp_words', 'hyp_numbers', 'empty']
                + [f'{p}{k}' for p in ('tol_', 'n0_', 'sm_', 'ovl_') for k in ('N', 'H', 'S', 'D', 'I', 'E')]
                + ['tol_optional', 'bag_hits', 'bag_ref', 'bag_hyp', 'repeat3'])


def sha16(text: str) -> str:
    """the first 16 hex digits of a text's sha256, as content_v1's cur_sha and the frozen sha are"""
    return L.sha256_hex(text or '')[:16]


def _seq(record: dict) -> int | None:
    seq = record.get('request_seq')
    return seq if isinstance(seq, int) and not isinstance(seq, bool) else None


def _number(value) -> float | None:
    if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---- the surface forms rescore.py scores (it defines them alike: a row's hash is checked against these) ----

def n1_reference(lines) -> str:
    """a transcript's lines in N1's surface form: each line's normalised words joined by spaces, one per turn"""
    out = (' '.join(X.normalise(line.tokens)) for line in lines)
    return '\n'.join(text for text in out if text)


def n1_text(text: str) -> str:
    """a version's text in N1's surface form, line by line"""
    out = (' '.join(X.normalise(line.split())) for line in (text or '').split('\n'))
    return '\n'.join(part for part in out if part)


# ---- reading ----

def log_records(artifacts: Path, audit_id: str) -> list[dict]:
    """the request log's lines that parse, in order"""
    path = A.audit_dir(artifacts, audit_id) / L.LOG_FILE
    try:
        data = path.read_bytes()
    except OSError:
        return []
    out = []
    for line in data.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def roles_notes(records: list[dict]) -> list[dict]:
    """the operator's notes (--log-note) that name the roles: {seq, text}"""
    return [{'seq': r.get('seq'), 'text': str(r.get('text') or '')} for r in records
            if r.get('event') == 'note' and ROLES.search(str(r.get('text') or ''))]


def counted_answers(artifacts: Path, plan: dict, campaign: dict, typed: bool,
                    closed: dict) -> tuple[dict, Counter, dict, dict]:
    """(auditor -> item -> phase -> the record that counts, what was left out and why, auditor -> the subsets
    their counted answers were saved under, what the lines show): audit_score.read_answers, then of a
    transcription the last line before its auditor's blind close and of a reveal rating the last line after it.
    The info: per name the transcriptions saved at or after their close ('late') and their last transcription's
    seq ('last_transcribe'), and the seq of the first scored (not practice) line of any name ('first_scored')."""
    answers, left, scopes = S.read_answers(artifacts, plan, campaign, typed)
    links = S.audit_links(campaign)
    history: dict[tuple, list] = defaultdict(list)
    first = None
    for _, path in A.record_files(artifacts, plan['audit_id'], [s['id'] for s in plan['sessions']]):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(line)
                auditor, item, phase = record['auditor'], record['item'], record['phase']
            except (ValueError, KeyError, TypeError):
                continue
            link = links.get(str(record.get('token_id')))
            # the lines read_answers counts: the open page's under the name typed, a link's for its own auditor
            opened = typed and link is None and record.get('open') is True and record.get('token_id') is None \
                and isinstance(auditor, str)
            if not opened and (link is None or link['name'] != auditor):
                continue
            seq = _seq(record)
            if not record.get('practice') and seq is not None:
                first = seq if first is None else min(first, seq)
            if phase in ('transcribe', 'reveal'):
                history[(auditor, item, phase)].append(record)
    late, last = Counter(), {}
    for (auditor, item, phase), records in history.items():
        close = (closed.get(auditor) or {}).get('seq')
        if phase == 'transcribe':
            kept = [r for r in records if close is None or (_seq(r) is not None and _seq(r) < close)]
            late[auditor] += len(records) - len(kept)
            seqs = [_seq(r) for r in records if _seq(r) is not None]
            if seqs:
                last[auditor] = max(last.get(auditor, -1), max(seqs))
        else:
            kept = [r for r in records if close is not None and _seq(r) is not None and _seq(r) > close]
            if len(kept) < len(records):
                left["reveal ratings saved before their auditor's blind close"] += len(records) - len(kept)
        phases = (answers.get(auditor) or {}).get(item)
        if phases is None:
            continue
        if kept:
            phases[phase] = kept[-1]
        else:
            phases.pop(phase, None)
    for name, count in sorted(late.items()):
        if count:
            left[f'transcriptions saved at or after the blind close of {name}'] += count
    return answers, left, scopes, {'late': {k: v for k, v in late.items() if v}, 'last_transcribe': last,
                                   'first_scored': first}


# ---- the units ----

def hyp_tokens(text: str, level: str = 'N1') -> list[str]:
    """a version's text normalised line by line (a variant never joins two lines)"""
    return [token for line in (text or '').split('\n') for token in X.normalise(line.split(), level)]


def edge_tokens(timed: list, channels: tuple) -> tuple[list[str], list[bool]]:
    """the edge-tolerant hypothesis, in time order and N1-normalised: the timed words of `channels` starting in
    [-EDGE, WINDOW + EDGE) s of the window, those within EDGE of an edge optional, and those starting before
    -EDGE that end inside the window, optional too; a token normalised from several words is optional when
    every one of them is. A chunk without word stamps (an approximate line of the text) has no timed words."""
    words, free = [], []
    for word, start, end, channel in timed:
        if channel not in channels or start is None:
            continue
        if -EDGE <= start < SP.WINDOW + EDGE:
            optional = start < EDGE or start >= SP.WINDOW - EDGE
        elif start < -EDGE and end is not None and end > 0:
            optional = True
        else:
            continue
        for piece in str(word).split():
            words.append(piece)
            free.append(optional)
    tokens, origins = X.normalise_tracked(words)
    return tokens, [all(free[i] for i in origin) for origin in origins]


def _counts(alignment, prefix: str = '') -> dict:
    return {f'{prefix}N': alignment.N, f'{prefix}H': alignment.H, f'{prefix}S': alignment.S, f'{prefix}D': alignment.D,
            f'{prefix}I': alignment.I, f'{prefix}E': alignment.E}


def _repeats(tokens: list[str]) -> bool:
    """whether a 3-gram of the tokens comes twice"""
    grams = Counter(zip(tokens, tokens[1:], tokens[2:]))
    return any(count > 1 for count in grams.values())


def source_counts(lines: list, text: str, timed: list, channels: tuple, overlap: bool) -> dict:
    """a transcript's lines against one source's text of the window: the counts of every alignment, no token"""
    per_line = [X.normalise(line.tokens) for line in lines]
    ref = [token for words in per_line for token in words]
    tags = [line.tag for line, words in zip(lines, per_line) for _ in words]
    hyp = hyp_tokens(text)
    strict = X.align(ref, hyp)
    out = _counts(strict)
    for tag, row in X.by_tag(strict, tags).items():
        for key, value in row.items():
            out[f'{key}_{TAG_KEY[tag]}'] = value
    out.update(_counts(X.cer(ref, hyp), 'c'))
    out.update(hyp=len(hyp), hyp_words=len((text or '').split()), hyp_numbers=X.number_tokens(text or ''),
               empty=int(not (text or '').strip()))
    ref0 = [token for line in lines for token in X.normalise(line.tokens, 'N0')]
    out.update(_counts(X.align(ref0, hyp_tokens(text, 'N0')), 'n0_'))
    tolerant, optional = edge_tokens(timed, channels)
    out.update(_counts(X.align(ref, tolerant, optional), 'tol_'), tol_optional=sum(optional))
    merged = X.split_merge(ref, hyp)
    out.update(_counts(X.align(merged.ref, merged.hyp), 'sm_'))
    best = strict
    if overlap and 1 < len(lines) <= OVERLAP_LINES:
        for order in itertools.permutations(range(len(lines))):
            tried = X.align([token for k in order for token in per_line[k]], hyp)
            if tried.E < best.E:
                best = tried
    out.update(_counts(best, 'ovl_'))
    shared = X.bag(X.content_words(ref), X.content_words(hyp))
    out.update(bag_hits=shared.hits, bag_ref=shared.ref, bag_hyp=shared.hyp, repeat3=int(_repeats(hyp)))
    return out


def _dominant(words: Counter) -> str:
    """the source of most of a transcript's words, as a who-speaks answer: none without words, cannot_tell on a tie"""
    if not sum(words.values()):
        return 'none'
    top = max(words.values())
    best = [tag for tag in X.TAGS if words[tag] == top]
    return WHO_OF_TAG[best[0]] if len(best) == 1 else 'cannot_tell'


def item_unit(design: dict, it: dict, entry: dict, said: dict, lines: list, phases: dict) -> dict:
    """one scored item: its design, the answers, the reveal ratings, the version's measures and the counts of
    each source of its text; no word of either"""
    asr = entry['asr']
    timed = [tuple(t) for t in asr.get('timed') or []]
    overlap = bool(said.get('overlap'))
    words = Counter()
    for line in lines:
        words[line.tag] += len(X.normalise(line.tokens))
    reveal = (phases.get('reveal') or {}).get('answer') or {}
    rated = bool(reveal) and not reveal.get('flag')
    group = asr.get('group')
    return {'item': it['item'], 'alias': design['alias'], 'session': design['session'], 'lesson': design['lesson'],
            'split': design.get('split'), 'stratum': it['stratum'], 'sweep': it.get('sweep') or 1, 'rank': it.get('rank'),
            'reliability': bool(it.get('reliability')), 'agreement': it.get('agreement'),
            'audio_kind': design.get('audio_kind'), 'window_start': float(it['window_start']), 'weight': None,
            'status': said.get('status'), 'overlap': overlap, 'who': said.get('who'), 'adult': said.get('adult'),
            'peer': said.get('peer'), 'other_offtask': said.get('other_offtask'), 'lines': len(lines),
            'markers_x': sum(line.markers.count('[x]') for line in lines),
            'markers_bg': sum(line.markers.count('[bg]') for line in lines), 'dominant': _dominant(words),
            'n0_ref': sum(len(X.normalise(line.tokens, 'N0')) for line in lines),
            'gist': reveal.get('gist') if rated else None, 'invented': reveal.get('invented') if rated else None,
            'reveal_flagged': bool(reveal.get('flag')), 'row': bool(entry.get('row')),
            'speech_ratio': _number(entry.get('speech_ratio')), 'words': _number(entry.get('words')),
            'n_asr_recognition': _number(entry.get('n_asr_recognition')),
            'member_words': _number(entry.get('member_words')),
            'consumed': source_counts(lines, asr['consumed']['cur'], timed, ('g', 'w'), overlap),
            'group': source_counts(lines, group['cur'], timed, ('g',), overlap) if group else None,
            'scores': {}, 'refs': {}, 'labels': {}, 'compare': {}, 'compare_item': None}


def item_hashes(said: dict, lines: list, entry: dict, record: dict) -> dict:
    """the hashes a re-scored reference row must carry: of the transcript as saved, its LLM and N1 forms, and of the
    version's text and prev, as they are and in N1's form (never written out)"""
    consumed = entry['asr']['consumed']
    return {'ref': sha16(said.get('transcript') or ''), 'llm': sha16(X.llm_form(lines)), 'ref_n1': sha16(n1_reference(lines)),
            'cur': consumed['sha'], 'prev': sha16(consumed['prev']), 'cur_n1': sha16(n1_text(consumed['cur'])),
            'prev_n1': sha16(n1_text(consumed['prev'])), 'request_seq': _seq(record)}


def skipped(served: list[tuple[str, bool]]) -> int:
    """the R items of a recording left unanswered before an item answered after them, in the order the page served
    them ((stratum, answered) per item): the answered R items are a random sample only while that order is kept"""
    last = max((i for i, (_, done) in enumerate(served) if done), default=-1)
    return sum(1 for stratum, done in served[:last] if stratum == 'R' and not done)


def build_units(designs: dict, versions: dict, render: dict, answers: dict,
                auditor: str) -> tuple[list[dict], Counter, dict, dict]:
    """the auditor's scored items of a version, one unit each; what was left out and why; per lesson the skips
    (R items left unanswered before an item answered after them in their recording's order on the page: sweep by
    sweep, each in its view's seeded order); and per item the hashes its re-scored reference is checked by"""
    units, left, skips, hashes = [], Counter(), defaultdict(int), {}
    mine = answers.get(auditor, {})
    for sid, design in designs.items():
        data = versions.get(sid)
        rendered = (render.get('sessions') or {}).get(design['alias'])
        served = []
        # the page's order of the recording: sweep by sweep, each sweep in the view's order (the design's)
        for it in sorted(design['speech'], key=lambda it: it.get('sweep') or 1):
            if it['practice']:
                left['practice items'] += 1
                continue
            if render and (not rendered or rendered.get('failed') or it['item'] in (rendered.get('errors') or {})):
                left['items not rendered'] += 1
                continue
            phases = mine.get(it['item']) or {}
            record = phases.get('transcribe')
            served.append((it['stratum'], record is not None))
            if record is None:
                left['items not answered'] += 1
                continue
            said = record.get('answer') or {}
            if said.get('flag'):
                left['items flagged'] += 1
                continue
            if data is None or data.get('refused'):
                left['items of a session the version refused'] += 1
                continue
            entry = (data.get('speech') or {}).get(it['item']) or {}
            if not (entry.get('asr') or {}).get('consumed'):
                left['items the version holds no text of'] += 1
                continue
            try:
                lines = X.parse_reference(said.get('transcript') or '')
            except X.TranscriptError:
                left['transcripts the grammar refuses'] += 1
                continue
            if X.status_error(said.get('status'), lines):
                left['transcripts their status disagrees with'] += 1
                continue
            units.append(item_unit(design, it, entry, said, lines, phases))
            hashes[it['item']] = item_hashes(said, lines, entry, record)
            if units[-1]['reveal_flagged']:
                left['reveal ratings flagged'] += 1
        if served:
            skips[design['lesson']] += skipped(served)
    return units, left, dict(skips), hashes


def weigh(units: list[dict], populations: dict) -> list[dict]:
    """copies of the units, each R item weighing its lesson's population over the lesson's R items among them"""
    kept = Counter(u['lesson'] for u in units if u['stratum'] == 'R')
    return [dict(u, weight=populations[u['lesson']] / kept[u['lesson']] if u['stratum'] == 'R' else None) for u in units]


def populations_of(plan: dict, designs: dict) -> dict:
    """each lesson's population (summed over its sessions), as the plan recorded it"""
    out = {lesson: row.get('population') for lesson, row in (plan.get('lessons') or {}).items()}
    for design in designs.values():
        if out.get(design['lesson']) is None:
            out[design['lesson']] = design.get('lesson_population') or design.get('population')
    return out


# ---- the content scores ----

def row_scores(row: dict) -> dict:
    """a content row's letter probabilities: no_text gives 0 (the model was not asked), a missing one None"""
    no_text = int(_number(row.get('no_text')) or 0)
    out = {name: (0.0 if no_text else _number(row.get(name))) for name in LETTER_FIELDS}
    out['no_text'] = no_text
    return out


def _row_sha(row: dict) -> str | None:
    for column in ('cur_sha', 'cur_sha256'):
        value = (row.get(column) or '').strip()
        if value:
            return value[:16]
    return None


class Arm:
    """one content model's scores of a version's text, item -> context -> scores, with where they came from and what
    was refused or missing"""

    def __init__(self, name: str, origin: str, file=None, sha256=None):
        self.name, self.origin, self.file, self.sha256 = name, origin, file, sha256
        self.scores: dict[str, dict[str, dict]] = {}
        self.rows = self.missing = self.mismatched = 0
        self.refused: str | None = None
        self.note: str | None = None

    def summary(self) -> dict:
        return {'origin': self.origin, 'file': None if self.file is None else str(self.file), 'sha256': self.sha256,
                'rows': self.rows, 'missing': self.missing, 'mismatched': self.mismatched, 'refused': self.refused,
                'note': self.note}


def _consumed(versions: dict, sid: str, item: str) -> dict | None:
    data = versions.get(sid) or {}
    if data.get('refused'):
        return None
    return (((data.get('speech') or {}).get(item) or {}).get('asr') or {}).get('consumed')


def frozen_arms(designs: dict, versions: dict) -> dict[str, Arm]:
    """the content arms frozen with a version: each item's frozen scores, and the letters too where the arm's file is
    still the one frozen (read again, a row's cur_sha the frozen sha where the file carries one, else the arm is
    refused with the count)"""
    arms: dict[str, Arm] = {}
    tables = {}
    for data in versions.values():
        if data.get('refused'):
            continue
        for name, source in ((data.get('sources') or {}).get('content') or {}).items():
            if name not in arms:
                arms[name] = Arm(name, 'frozen', source.get('file'), source.get('sha256'))
                tables[name] = None
                if source.get('file') and L.file_sha256(source['file']) == source.get('sha256'):
                    try:
                        tables[name] = SP.ContentArm(name, source['file'])
                    except A.AuditError as error:
                        arms[name].note = f'its file cannot be read again ({error}): the frozen values, no letters'
                else:
                    arms[name].note = 'its file is gone or changed since the freeze: the frozen values, no letters'
            elif (source.get('file'), source.get('sha256')) != (arms[name].file, arms[name].sha256):
                arms[name].refused = 'frozen from two files'
    for name, arm in arms.items():
        unhashed = 0
        for sid, design in designs.items():
            data = versions.get(sid) or {}
            if data.get('refused'):
                continue
            for it in design['speech']:
                entry = (data.get('speech') or {}).get(it['item']) or {}
                consumed = (entry.get('asr') or {}).get('consumed')
                frozen = (entry.get('content') or {}).get(name)
                if consumed is None or not frozen:
                    arm.missing += consumed is not None
                    continue
                for context, values in frozen.items():
                    row = tables[name].row(sid, float(it['window_start']), context) if tables[name] else None
                    arm.rows += 1
                    if row is not None:
                        sha = _row_sha(row)
                        if sha is not None and sha != consumed['sha']:
                            arm.mismatched += 1
                        unhashed += sha is None
                        scores = row_scores(row)
                    else:
                        scores = {field: None for field in LETTER_FIELDS}
                        scores.update({k: values.get(k) for k in ('teacher', 'peer_task', 'peer_other')},
                                      no_text=int(values.get('no_text') or 0))
                    arm.scores.setdefault(it['item'], {})[str(context)] = scores
        if arm.mismatched and not arm.refused:
            arm.refused = (f'{arm.mismatched} rows of its file carry a cur_sha that is not the frozen text\'s: the scores '
                           'are of another text')
        if unhashed and not arm.note:
            arm.note = f'{unhashed} rows carry no cur_sha: matched by their window alone, as at the freeze'
    return arms


def scoring_arms(spec: str | None, designs: dict, versions: dict) -> dict[str, Arm]:
    """the content arms --audit-content-scores names for the version scored: every item's row (session, window
    start within MATCH_TOLERANCE, source 'all', each of the file's contexts), its cur_sha the frozen text's sha,
    else the arm is refused with the count (no override); a file without cur_sha cannot be checked, so is refused"""
    out = {}
    for name, table in SP.content_arms(spec).items():
        arm = out[name] = Arm(name, 'scoring', str(table.path), table.sha256)
        hashed = True
        for sid, design in designs.items():
            contexts = table.contexts(sid)
            for it in design['speech']:
                consumed = _consumed(versions, sid, it['item'])
                if consumed is None:
                    continue
                if not contexts:
                    arm.missing += 1
                for context in contexts:
                    row = table.row(sid, float(it['window_start']), context)
                    if row is None:
                        arm.missing += 1
                        continue
                    arm.rows += 1
                    sha = _row_sha(row)
                    if sha is None:
                        hashed = False
                    elif sha != consumed['sha']:
                        arm.mismatched += 1
                    arm.scores.setdefault(it['item'], {})[context] = row_scores(row)
        if not hashed:
            arm.refused = 'its rows carry no cur_sha: they cannot be checked against the frozen text'
        elif arm.mismatched:
            arm.refused = (f'{arm.mismatched} rows carry a cur_sha that is not the frozen text\'s: the scores are of '
                           'another text')
    return out


def _ref_row_ok(variant: str, row: dict, hashes: dict) -> bool:
    cur, prev, named, context = REF_VARIANTS[variant]
    if str(row.get('context') or '').strip() not in (context, f'{context}.0'):
        return False
    if (row.get('cur_sha') or '').strip() != hashes[cur]:
        return False
    if prev and (row.get('prev_sha') or '').strip() != hashes[prev]:
        return False
    if named and ((row.get('ref_sha') or '').strip() != hashes['ref']
                  or str(row.get('request_seq') or '').strip() != str(hashes['request_seq'])):
        return False
    return True


def reference_arms(spec: str | None, version: str, hashes: dict, normaliser: str | None) -> dict[str, dict]:
    """--audit-content-scores-ref ARM=FILE,...: per arm and variant (REF_VARIANTS) the scores of the primary's
    references, rescore.py's rows of the version scored (ref_ctx0's of any), each checked by its hashes and request
    seq against the counted transcript and the version's text; a variant with a row that fails, or two rows of one
    item, is refused with the count, and the N1 variants when the file's meta names another normaliser than the
    plan's"""
    out = {}
    for part in (spec or '').split(','):
        if not part.strip():
            continue
        name, _, raw = part.partition('=')
        name = name.strip()
        if not name or not raw.strip():
            raise A.AuditError('give --audit-content-scores-ref as ARM=FILE,ARM=FILE')
        path = Path(os.path.expanduser(raw.strip())).resolve()
        digest = L.file_sha256(path)
        if digest is None:
            raise A.AuditError(f'no reference scores of arm {name} at {path}')
        meta_path = path.with_name(path.name + '.meta.json')
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            meta = {}
        variants = {v: {'scores': {}, 'rows': 0, 'missing': 0, 'mismatched': 0, 'refused': None} for v in REF_VARIANTS}
        seen = set()
        with path.open(encoding='utf-8', newline='') as file:
            for row in csv.DictReader(file):
                variant, item = row.get('variant'), row.get('item')
                if variant not in variants or item not in hashes:
                    continue
                if variant != 'ref_ctx0' and row.get('version') != version:
                    continue
                if variant == 'ref_ctx0' and row.get('version') not in ('', None, version):
                    continue
                entry = variants[variant]
                if (item, variant) in seen:
                    entry['refused'] = 'two rows of one item'
                    continue
                seen.add((item, variant))
                entry['rows'] += 1
                if not _ref_row_ok(variant, row, hashes[item]):
                    entry['mismatched'] += 1
                    continue
                entry['scores'][item] = row_scores(row)
        for variant, entry in variants.items():
            entry['missing'] = sum(1 for item in hashes if item not in entry['scores']) - entry['mismatched']
            if entry['mismatched'] and not entry['refused']:
                entry['refused'] = (f"{entry['mismatched']} rows whose hashes or request seq are not those of the counted "
                                    "transcript and the version's text")
        if meta.get('normaliser_sha256') and normaliser and meta['normaliser_sha256'] != normaliser:
            for variant in ('ref_n1', 'asr_n1'):
                variants[variant]['refused'] = variants[variant]['refused'] or \
                    "its N1 forms are another normaliser's than the plan's"
        out[name] = {'file': str(path), 'sha256': digest, 'variants': variants,
                     'meta': {k: meta.get(k) for k in ('scorer', 'scorer_sha256', 'normaliser_sha256', 'model',
                                                       'revision', 'input_sha256', 'rows')}}
    return out


def attach_scores(units: list[dict], arms: dict[str, Arm], refs: dict[str, dict]) -> None:
    for u in units:
        for name, arm in arms.items():
            if arm.refused:
                continue
            for context, values in (arm.scores.get(u['item']) or {}).items():
                u['scores'][(name, context)] = values
        for name, ref in refs.items():
            for variant, entry in ref['variants'].items():
                if not entry['refused'] and u['item'] in entry['scores']:
                    u['refs'][(name, variant)] = entry['scores'][u['item']]


def _arm(name: str, context: str, key: str) -> Callable:
    def get(u):
        values = u['scores'].get((name, context))
        return None if values is None else values.get(key)
    return get


def _ref(name: str, variant: str, key: str) -> Callable:
    def get(u):
        values = u['refs'].get((name, variant))
        return None if values is None else values.get(key)
    return get


def peer_letter(values: dict | None) -> str | None:
    """the peer question's most likely letter, None when the model was not asked or a letter is missing"""
    if values is None or values.get('no_text'):
        return None
    probs = [(values.get(field), letter) for letter, field in PEER_OPTIONS]
    if any(p is None for p, _ in probs):
        return None
    return max(probs, key=lambda p: p[0])[1]


# ---- the comparison audit and the coders' labels ----

class Comparison:
    """the sensing audit --audit-compare-with names: its designs the ones its sampling logged, its request log
    verified with its answers files (both read only), and its who-speaks answers by name"""

    def __init__(self, artifacts: Path, audit_id: str, despite_log_failure: bool = False):
        self.id = audit_id
        self.source = SP.SourceAudit(artifacts, audit_id)
        campaign = L.Campaign(A.audit_dir(artifacts, audit_id))
        if not campaign.exists():
            raise A.AuditError(f'no campaign.yml in {campaign.folder}: whose answers {audit_id} holds is not known')
        self.log = L.checked_log(campaign, artifacts, despite_log_failure, 'score')
        data = campaign.data()
        self.open = bool(S.open_serves(artifacts, audit_id))
        answers, self.left, scopes = S.read_answers(artifacts, self.source.plan, data, typed=self.open)
        self.primary = (S.primary_typed(answers, None, scopes) if self.open else
                        S.primary_auditor(answers, None, data))
        self.speakers = {name: {item: phases['speech'].get('answer') or {} for item, phases in items.items()
                                if 'speech' in phases} for name, items in answers.items()}
        self.names = sorted(self.speakers)
        self.plan_sha256 = self.source.plan_sha256

    def item_at(self, sid: str, start: float) -> str | None:
        """the comparison's who-speaks window of the session starting at `start` (within MATCH_TOLERANCE)"""
        windows = self.source.windows(sid, False)
        near = min(windows, key=lambda w: abs(float(w['window_start']) - start), default=None)
        return near['item'] if near is not None and abs(float(near['window_start']) - start) <= A.MATCH_TOLERANCE else None

    def speaker(self, name: str, item: str | None) -> str | None:
        answer = self.speakers.get(name, {}).get(item) if item else None
        if not answer or answer.get('flag'):
            return None
        return answer.get('speaker')

    def attach(self, units: list[dict]) -> int:
        """each unit's comparison item and every name's speaker there; the number joined"""
        joined = 0
        for u in units:
            u['compare_item'] = self.item_at(u['session'], u['window_start'])
            u['compare'] = {name: self.speaker(name, u['compare_item']) for name in self.names}
            joined += u['compare_item'] is not None
        return joined


def attach_labels(artifacts: Path, plan: dict, designs: dict, units: list[dict]) -> tuple[list[str], list[str]]:
    """each unit's last label of every coder the plan's social draw named (audit_speech.coder_labels); (the coders,
    the labels files changed since the sample or unreadable now)"""
    coders = list((plan.get('social') or {}).get('coders') or [])
    changed = []
    if not coders:
        return coders, changed
    by_session = defaultdict(list)
    for u in units:
        by_session[u['session']].append(u)
    for sid, chosen in sorted(by_session.items()):
        alias = designs[sid]['alias']
        try:
            labels, digests = SP.coder_labels(Path(artifacts) / sid, coders)
        except A.AuditError as error:
            changed.append(f'{alias}: {error}')
            continue
        recorded = designs[sid].get('labels') or {}
        changed += [f'{alias} {coder}' for coder in coders if digests.get(coder) != recorded.get(coder)]
        for u in chosen:
            u['labels'] = {coder: SP._label_at(labels[coder], u['window_start']) for coder in coders}
    return coders, changed


# ---- metrics on additive statistics (a lesson bootstrap sums lessons), and those that are not ----

def _keeps(metric, u) -> bool:
    keeps = getattr(metric, 'keeps', None)
    return keeps(u) if keeps is not None else metric.keep(u)


class SumRatio:
    """a ratio of sums: sum of w * num(u) over sum of w * den(u) over the units `keep` keeps (a WER, a CER, a share,
    a rate); w is 1, or the design weight when `weighted` (R items only)"""
    total = False

    def __init__(self, name: str, keep: Callable, num: Callable, den: Callable | None = None, weighted: bool = False):
        self.name = name + (' (weighted)' if weighted else '')
        self.keep, self.num, self.den, self.weighted = keep, num, den or (lambda u: 1.0), weighted

    def _w(self, u) -> float:
        return float(u['weight']) if self.weighted else 1.0

    def stats(self, units):
        import numpy as np
        num = den = 0.0
        for u in units:
            if self.keep(u):
                w = self._w(u)
                num += w * float(self.num(u))
                den += w * float(self.den(u))
        return np.array([num, den])

    @staticmethod
    def final(stats) -> float:
        return float(stats[0] / stats[1]) if stats[1] > 0 else float('nan')

    def count(self, units) -> int:
        return sum(1 for u in units if self.keep(u))


class Total(SumRatio):
    """a sum (the reference words, the hypothesis tokens), given with no interval"""
    total = True

    def stats(self, units):
        import numpy as np
        return np.array([sum(float(self.num(u)) for u in units if self.keep(u))])

    @staticmethod
    def final(stats) -> float:
        return float(stats[0])


class Concordance:
    """the within-lesson concordance C of score(u): over the pairs of a positive and a negative of one lesson, the
    share where the positive scores higher, a tie counting a half; with `weighted` a pair counts w_i w_j"""

    def __init__(self, name: str, keep: Callable, score: Callable, positive: Callable, negative: Callable,
                 weighted: bool = False):
        self.name = name + (' (weighted)' if weighted else '')
        self.keep, self.score, self.positive, self.negative, self.weighted = keep, score, positive, negative, weighted

    def keeps(self, u) -> bool:
        return self.keep(u) and self.score(u) is not None and (self.positive(u) or self.negative(u))

    def _sides(self, units) -> dict:
        by: dict = defaultdict(lambda: ([], []))
        for u in units:
            if self.keeps(u):
                w = float(u['weight']) if self.weighted else 1.0
                by[u['lesson']][0 if self.positive(u) else 1].append((float(self.score(u)), w))
        return by

    @staticmethod
    def _pairs(positives, negatives) -> tuple[float, float]:
        num = den = 0.0
        for sp, wp in positives:
            for sn, wn in negatives:
                num += wp * wn * (1.0 if sp > sn else 0.5 if sp == sn else 0.0)
                den += wp * wn
        return num, den

    def stats(self, units):
        import numpy as np
        num = den = 0.0
        for positives, negatives in self._sides(units).values():
            a, b = self._pairs(positives, negatives)
            num, den = num + a, den + b
        return np.array([num, den])

    @staticmethod
    def final(stats) -> float:
        return float(stats[0] / stats[1]) if stats[1] > 0 else float('nan')

    def count(self, units) -> int:
        return sum(1 for u in units if self.keeps(u))

    def sides(self, units) -> tuple[int, int]:
        chosen = [u for u in units if self.keeps(u)]
        return sum(1 for u in chosen if self.positive(u)), sum(1 for u in chosen if not self.positive(u))


class PooledAUC(Concordance):
    """the AUROC of score(u): C over every pair of a positive and a negative, across lessons too"""

    def resampler(self, units, lessons):
        import numpy as np
        at = {lesson: i for i, lesson in enumerate(lessons)}
        k = len(lessons)
        by = self._sides(units)
        num, den = np.zeros((k, k)), np.zeros((k, k))
        for lp, (positives, _) in by.items():
            for ln, (_, negatives) in by.items():
                if lp in at and ln in at:
                    num[at[lp], at[ln]], den[at[lp], at[ln]] = self._pairs(positives, negatives)

        def fn(rows):
            drawn = np.bincount(rows, minlength=k).astype(float)
            total = drawn @ den @ drawn
            return float(drawn @ num @ drawn / total) if total > 0 else float('nan')
        return fn


class Transfer:
    """TR = (C_ASR - 0.5) / (C_ref - 0.5) of two concordances on the same units; not estimable (NaN) when C_ref is
    at most TR_FLOOR"""
    weighted = False

    def __init__(self, name: str, asr: Concordance, ref: Concordance):
        self.name, self.asr, self.ref = name, asr, ref

    def keeps(self, u) -> bool:
        return self.asr.keeps(u) and self.ref.keeps(u)

    def stats(self, units):
        import numpy as np
        return np.concatenate([self.asr.stats(units), self.ref.stats(units)])

    @staticmethod
    def final(stats) -> float:
        c_asr = stats[0] / stats[1] if stats[1] > 0 else float('nan')
        c_ref = stats[2] / stats[3] if stats[3] > 0 else float('nan')
        if not (math.isfinite(c_asr) and math.isfinite(c_ref)) or c_ref <= TR_FLOOR:
            return float('nan')
        return float((c_asr - 0.5) / (c_ref - 0.5))

    def count(self, units) -> int:
        return sum(1 for u in units if self.keeps(u))


class CCC:
    """Lin's concordance correlation coefficient of x(u) and y(u): 2 cov / (var x + var y + (mean x - mean y)^2), the
    moments over n"""
    weighted = False

    def __init__(self, name: str, keep: Callable, x: Callable, y: Callable):
        self.name, self.keep, self.x, self.y = name, keep, x, y

    def keeps(self, u) -> bool:
        return self.keep(u) and self.x(u) is not None and self.y(u) is not None

    def stats(self, units):
        import numpy as np
        out = np.zeros(6)
        for u in units:
            if self.keeps(u):
                x, y = float(self.x(u)), float(self.y(u))
                out += (1.0, x, y, x * x, y * y, x * y)
        return out

    @staticmethod
    def final(stats) -> float:
        n = stats[0]
        if n < 2:
            return float('nan')
        mx, my = stats[1] / n, stats[2] / n
        vx, vy = stats[3] / n - mx * mx, stats[4] / n - my * my
        cov = stats[5] / n - mx * my
        den = vx + vy + (mx - my) ** 2
        return float(2 * cov / den) if den > 1e-15 else float('nan')

    def count(self, units) -> int:
        return sum(1 for u in units if self.keeps(u))


def _ranks(values):
    """average ranks, ties sharing their mean rank"""
    import numpy as np
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    inverse = np.empty(len(values), dtype=int)
    inverse[order] = np.arange(len(values))
    ordered = values[order]
    starts = np.r_[True, ordered[1:] != ordered[:-1]]
    dense = starts.cumsum()[inverse]
    edges = np.r_[np.flatnonzero(starts), len(values)]
    return 0.5 * (edges[dense] + edges[dense - 1] + 1)


def spearman(xs, ys) -> float:
    """Spearman's rho: Pearson's correlation of the average ranks (NaN under 3 pairs or without variation)"""
    if len(xs) < 3:
        return float('nan')
    rx, ry = _ranks(xs), _ranks(ys)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float('nan')


class Spearman:
    """Spearman's rho of x(u) and y(u) over the units `keep` keeps (recomputed from the items of each draw)"""
    weighted = False

    def __init__(self, name: str, keep: Callable, x: Callable, y: Callable):
        self.name, self.keep, self.x, self.y = name, keep, x, y

    def keeps(self, u) -> bool:
        return self.keep(u) and self.x(u) is not None and self.y(u) is not None

    def count(self, units) -> int:
        return sum(1 for u in units if self.keeps(u))

    def resampler(self, units, lessons):
        import numpy as np
        at = {lesson: i for i, lesson in enumerate(lessons)}
        xs, ys = [[] for _ in lessons], [[] for _ in lessons]
        for u in units:
            if self.keeps(u) and u['lesson'] in at:
                xs[at[u['lesson']]].append(float(self.x(u)))
                ys[at[u['lesson']]].append(float(self.y(u)))
        xs, ys = [np.array(v) for v in xs], [np.array(v) for v in ys]

        def fn(rows):
            return spearman(np.concatenate([xs[r] for r in rows]), np.concatenate([ys[r] for r in rows]))
        return fn


class LessonSpearman:
    """the mean of Spearman's rho of x(u) and y(u) within each lesson of at least MIN_LESSON_ITEMS units, weighted
    by the lesson's units"""
    weighted = False

    def __init__(self, name: str, keep: Callable, x: Callable, y: Callable):
        self.name, self.keep, self.x, self.y = name, keep, x, y

    def keeps(self, u) -> bool:
        return self.keep(u) and self.x(u) is not None and self.y(u) is not None

    def stats(self, units):
        import numpy as np
        by = defaultdict(list)
        for u in units:
            if self.keeps(u):
                by[u['lesson']].append((float(self.x(u)), float(self.y(u))))
        num = den = 0.0
        for pairs in by.values():
            if len(pairs) >= MIN_LESSON_ITEMS:
                rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
                if math.isfinite(rho):
                    num, den = num + len(pairs) * rho, den + len(pairs)
        return np.array([num, den])

    @staticmethod
    def final(stats) -> float:
        return float(stats[0] / stats[1]) if stats[1] > 0 else float('nan')

    def count(self, units) -> int:
        return sum(1 for u in units if self.keeps(u))


class MacroMean:
    """the mean over lessons of a ratio of sums within each (the lesson macro mean)"""
    weighted = False

    def __init__(self, name: str, ratio: SumRatio):
        self.name, self.ratio = name, ratio

    def keeps(self, u) -> bool:
        return self.ratio.keep(u)

    def count(self, units) -> int:
        return self.ratio.count(units)

    def resampler(self, units, lessons):
        import numpy as np
        by = defaultdict(list)
        for u in units:
            by[u['lesson']].append(u)
        table = np.array([self.ratio.stats(by[lesson]) for lesson in lessons]).reshape(len(lessons), 2)

        def fn(rows):
            chosen = table[rows]
            ok = chosen[:, 1] > 0
            return float(np.mean(chosen[ok, 0] / chosen[ok, 1])) if ok.any() else float('nan')
        return fn


def _fn(metric, units: list[dict], lessons: list) -> Callable:
    """fn(rows) of a metric over the lessons `rows` draws (a lesson drawn twice counts twice)"""
    if hasattr(metric, 'resampler'):
        return metric.resampler(units, lessons)
    import numpy as np
    by = defaultdict(list)
    for u in units:
        by[u['lesson']].append(u)
    table = np.array([metric.stats(by[lesson]) for lesson in lessons])
    return lambda rows: metric.final(table[rows].sum(axis=0))


def _lolo(fn: Callable, k: int) -> tuple[float | None, float | None]:
    """the least and the greatest value with one lesson left out"""
    import numpy as np
    if k < 2:
        return None, None
    everything = np.arange(k)
    values = [fn(np.delete(everything, i)) for i in range(k)]
    finite = [v for v in values if math.isfinite(v)]
    return (S._value(min(finite)), S._value(max(finite))) if finite else (None, None)


def evaluate(metric, units: list[dict], boot: int, seed: int, lolo: bool = False) -> dict:
    """a metric over the units: its value, n, the lessons it rests on, a lesson-cluster bootstrap interval
    (metrics.session_bootstrap over those lessons) and, with `lolo`, its leave-one-lesson-out range"""
    import numpy as np
    from openmmla.analytics.interaction.metrics import session_bootstrap
    lessons = sorted({u['lesson'] for u in units if _keeps(metric, u)})
    out = {'metric': metric.name, 'value': None, 'n': metric.count(units), 'lessons': len(lessons), 'lo': None,
           'hi': None, 'interval': None, 'lolo_lo': None, 'lolo_hi': None}
    if not lessons:
        return out
    fn = _fn(metric, units, lessons)
    out['value'] = S._value(fn(np.arange(len(lessons))))
    if out['value'] is None:
        return out
    if boot > 0 and not getattr(metric, 'total', False):
        result = session_bootstrap(fn, np.array(lessons, dtype=object), n=boot, seed=seed)
        out.update(lo=S._value(result['lo']), hi=S._value(result['hi']),
                   interval=f"lesson bootstrap 95 % ({len(lessons)} lessons, {result['n_undefined']} of {result['n']} "
                            'undefined)')
    if lolo:
        out['lolo_lo'], out['lolo_hi'] = _lolo(fn, len(lessons))
    return out


def paired(name: str, a, units_a: list[dict], b, units_b: list[dict], boot: int, seed: int, lolo: bool = False) -> dict:
    """a - b with its interval from the same lesson draws for both (metrics.paired_delta), the lessons those of
    either side; no p-value"""
    import numpy as np
    from openmmla.analytics.interaction.metrics import paired_delta
    lessons = sorted({u['lesson'] for u in units_a if _keeps(a, u)} | {u['lesson'] for u in units_b if _keeps(b, u)})
    out = {'metric': name, 'value': None, 'n': a.count(units_a), 'lessons': len(lessons), 'lo': None, 'hi': None,
           'interval': None, 'lolo_lo': None, 'lolo_hi': None}
    if not lessons:
        return out
    fa, fb = _fn(a, units_a, lessons), _fn(b, units_b, lessons)
    everything = np.arange(len(lessons))
    out['value'] = S._value(fa(everything) - fb(everything))
    if out['value'] is None:
        return out
    if boot > 0:
        result = paired_delta(fa, fb, np.array(lessons, dtype=object), n=boot, seed=seed)
        out.update(lo=S._value(result['lo']), hi=S._value(result['hi']),
                   interval=f"paired lesson bootstrap 95 % ({len(lessons)} lessons, {result['n_undefined']} of "
                            f"{result['n']} undefined)")
    if lolo:
        out['lolo_lo'], out['lolo_hi'] = _lolo(lambda rows: fa(rows) - fb(rows), len(lessons))
    return out


class Tables:
    """the rows of every check: pooled ones with their intervals, and per lesson the counts and point values"""

    def __init__(self, boot: int, seed: int, aliases: dict):
        self.boot, self.seed, self.aliases = boot, seed, aliases
        self.pooled: dict[str, list] = defaultdict(list)
        self.lessons: dict[str, list] = defaultdict(list)

    def add(self, check: str, analysis: str, version: str, metric, units: list[dict], by: str = 'pooled',
            group: str = 'all', arm: str | None = None, lolo: bool = False, band: Callable | None = None) -> dict:
        row = {'check': check, 'version': version, 'analysis': analysis, 'arm': arm, 'by': by, 'group': group,
               **evaluate(metric, units, self.boot, self.seed, lolo)}
        if hasattr(metric, 'sides'):
            row['positives'], row['negatives'] = metric.sides(units)
        row['band'] = band(row) if band else None
        self.pooled[check].append(row)
        if by == 'pooled':
            self.lessons[check] += self._per_lesson(check, analysis, version, arm, metric, units)
        return row

    def delta(self, check: str, analysis: str, version: str, name: str, a, units_a: list[dict], b, units_b: list[dict],
              by: str = 'pooled', group: str = 'all', arm: str | None = None, lolo: bool = False,
              band: Callable | None = None) -> dict:
        row = {'check': check, 'version': version, 'analysis': analysis, 'arm': arm, 'by': by, 'group': group,
               **paired(name, a, units_a, b, units_b, self.boot, self.seed, lolo)}
        row['band'] = band(row) if band else None
        self.pooled[check].append(row)
        return row

    def _per_lesson(self, check, analysis, version, arm, metric, units) -> list[dict]:
        import numpy as np
        rows = []
        for lesson in sorted({u['lesson'] for u in units if _keeps(metric, u)}):
            chosen = [u for u in units if u['lesson'] == lesson]
            if hasattr(metric, 'resampler'):
                value, stats = metric.resampler(chosen, [lesson])(np.array([0])), None
            else:
                stats = metric.stats(chosen)
                value = metric.final(stats)
            two = stats is not None and len(stats) == 2
            rows.append({'check': check, 'version': version, 'analysis': analysis, 'arm': arm, 'lesson': lesson,
                         'aliases': self.aliases.get(lesson), 'metric': metric.name, 'value': S._value(value),
                         'n': metric.count(chosen), 'num': S._value(stats[0]) if two else None,
                         'den': S._value(stats[1]) if two else None})
        return rows


# ---- the bands of the declared decisions (descriptive) ----

def few(row: dict) -> str:
    return f'descriptive only (fewer than {MIN_POSITIVES} peer = other items)'


def band_p2a(row: dict) -> str | None:
    if row['lo'] is None:
        return None
    return ('tracks off-task talk a listener hears (lower bound above 0.5)' if row['lo'] > 0.5 else
            'not shown (lower bound at or below 0.5)')


def band_p3(row: dict) -> str | None:
    if row['lo'] is None:
        return None
    if row['hi'] < 0.05:
        return 'the ASR costs little (upper bound below 0.05)' + (', its cost above 0' if row['lo'] > 0 else '')
    if row['lo'] > 0:
        return 'the ASR costs part of the signal (lower bound above 0)'
    return 'not resolved'


def band_tr(row: dict) -> str:
    value = row['value']
    if value is None:
        return f'not estimable (C_ref at or below {TR_FLOOR})'
    return 'little lost' if value >= 0.9 else 'part lost' if value >= 0.6 else 'most lost'


# ---- the analyses ----

def _has(source: str, prefix: str = '', least: int = 1) -> Callable:
    return lambda u: u.get(source) is not None and u[source][f'{prefix}N'] >= least


def _get(source: str, key: str) -> Callable:
    return lambda u: u[source][key]


def rate(name: str, source: str, num: str, den: str = 'N', weighted: bool = False, keep: Callable | None = None,
         prefix: str = '') -> SumRatio:
    """a ratio of sums of a source's counts over its items with N >= 1 (of the `prefix` alignment) and `keep`"""
    has = _has(source, prefix)
    return SumRatio(name, (lambda u: has(u) and keep(u)) if keep else has, _get(source, num), _get(source, den), weighted)


def wer(name: str, source: str = 'consumed', prefix: str = '', weighted: bool = False, keep=None) -> SumRatio:
    return rate(name, source, f'{prefix}E', f'{prefix}N', weighted, keep, prefix)


def accuracy(tables: Tables, check: str, analysis: str, version: str, units: list[dict], source: str = 'consumed',
             weights=(True, False), lolo: bool = False) -> None:
    """the WER of a source with its substitution, deletion and insertion shares, the CER and the totals (P1)"""
    for weighted in weights:
        tables.add(check, analysis, version, wer('WER', source, weighted=weighted), units, lolo=lolo)
    for weighted in weights:
        for key, title in (('S', 'substitution share'), ('D', 'deletion share'), ('I', 'insertion share')):
            tables.add(check, analysis, version, rate(title, source, key, weighted=weighted), units)
        tables.add(check, analysis, version, rate('CER', source, 'cE', 'cN', weighted), units, lolo=lolo)
    has = _has(source)
    tables.add(check, analysis, version, Total('reference words (N1)', has, _get(source, 'N')), units)
    tables.add(check, analysis, version, Total('hypothesis tokens (N1)', has, _get(source, 'hyp')), units)


def content_words(tables: Tables, version: str, R: list[dict]) -> None:
    """S6: the content words (N1 less the stopwords and the response words) both texts share, as multisets: precision
    over the hypothesis's (the items without reference words included), recall over the reference's, and F1"""
    def has(u):
        return u['consumed'] is not None
    for weighted in (True, False):
        tables.add('accuracy', 'S6', version, SumRatio('content words: precision', has, _get('consumed', 'bag_hits'),
                                                       _get('consumed', 'bag_hyp'), weighted), R)
        tables.add('accuracy', 'S6', version, SumRatio('content words: recall', has, _get('consumed', 'bag_hits'),
                                                       _get('consumed', 'bag_ref'), weighted), R)
        tables.add('accuracy', 'S6', version, SumRatio(
            'content words: F1', has, lambda u: 2 * u['consumed']['bag_hits'],
            lambda u: u['consumed']['bag_ref'] + u['consumed']['bag_hyp'], weighted), R)


def detection(tables: Tables, check: str, analysis: str, version: str, units: list[dict], weights=(True, False)) -> None:
    """the miss and phantom rates, the hypothesis tokens per minute of silence and of unintelligible speech, the
    repeated 3-grams (S1)"""
    def speech(least):
        return lambda u: u['status'] == 'speech' and u['consumed']['N'] >= least

    def status(value):
        return lambda u: u['status'] == value
    for weighted in weights:
        for least in (1, 5):
            tables.add(check, analysis, version, SumRatio(f'miss rate (the text empty; speech, N >= {least})',
                                                          speech(least), _get('consumed', 'empty'), weighted=weighted), units)
        for value in ('none', 'unintelligible'):
            for least in (1, 3):
                tables.add(check, analysis, version, SumRatio(
                    f'phantom rate ({value}: at least {least} hypothesis tokens)', status(value),
                    (lambda k: lambda u: u['consumed']['hyp'] >= k)(least), weighted=weighted), units)
            tables.add(check, analysis, version, SumRatio(
                f'hypothesis tokens per minute of {value} audio', status(value), _get('consumed', 'hyp'),
                lambda u: SP.WINDOW / 60.0, weighted), units)
    tables.add(check, analysis, version, SumRatio('none items whose hypothesis repeats a 3-gram', status('none'),
                                                  _get('consumed', 'repeat3')), units)


def sources(tables: Tables, version: str, R: list[dict]) -> None:
    """recall per reference tag, the source mix of the hits, the WER by who speaks, teacher against member recall (S4)"""
    for weighted in (True, False):
        for tag in X.TAGS:
            k = TAG_KEY[tag]
            tables.add('sources', 'S4', version, SumRatio(
                f'recall of the {tag} words', (lambda k: lambda u: u['consumed'][f'N_{k}'] >= 1)(k),
                _get('consumed', f'H_{k}'), _get('consumed', f'N_{k}'), weighted), R, by='tag', group=tag)
            tables.add('sources', 'S4', version, SumRatio(
                f'share of the hits that are {tag} words', lambda u: u['consumed']['H'] >= 1, _get('consumed', f'H_{k}'),
                _get('consumed', 'H'), weighted), R, by='tag', group=tag)
    for who in ANSWERED_WHO:
        tables.add('sources', 'S4', version, wer('WER', weighted=True, keep=(lambda w: lambda u: u['who'] == w)(who)), R,
                   by='who', group=who)
    teacher = SumRatio('recall of the T words', lambda u: u['consumed']['N_T'] >= 1, _get('consumed', 'H_T'),
                       _get('consumed', 'N_T'), True)
    member = SumRatio('recall of the M words', lambda u: u['consumed']['N_M'] >= 1, _get('consumed', 'H_M'),
                      _get('consumed', 'N_M'), True)
    tables.delta('sources', 'S4', version, 'recall of the T words - recall of the M words (weighted, paired)', teacher, R,
                 member, R)


def group_source(tables: Tables, version: str, R: list[dict]) -> None:
    """the group source's WER, both sources by the kind of sound, and consumed against group in dual sessions (S2)"""
    for weighted in (True, False):
        tables.add('group', 'S2', version, wer('WER of the group source', 'group', weighted=weighted), R)
    for kind in SP.AUDIO_KINDS:
        chosen = [u for u in R if u['audio_kind'] == kind]
        for source in SOURCES:
            if any(u[source] is not None for u in chosen):
                tables.add('group', 'S2', version, wer(f'WER of {source}', source, weighted=True), chosen, by='kind',
                           group=kind)
    dual = [u for u in R if u['audio_kind'] == 'dual' and u['group'] is not None]
    if dual:
        tables.delta('group', 'S2', version, 'WER of consumed - WER of group (weighted, paired)',
                     wer('WER of consumed', weighted=True), dual, wer('WER of group', 'group', weighted=True), dual,
                     by='kind', group='dual')
        for source in SOURCES:
            tables.add('group', 'S2', version, rate(f'insertion share of {source}', source, 'I', weighted=True), dual,
                       by='kind', group='dual')


def words_feature(tables: Tables, version: str, R: list[dict]) -> None:
    """the table's words feature against the reference words (N0), and speech_ratio against the speech heard (S5)"""
    def measured(u):
        return u['words'] is not None

    def ratio(u):
        return u['speech_ratio'] is not None
    for weighted in (True, False):
        tables.add('words', 'S5', version, SumRatio('words feature / reference words (N0)', measured,
                                                    lambda u: u['words'], lambda u: u['n0_ref'], weighted), R)
        tables.add('words', 'S5', version, SumRatio('mean signed error, words feature - reference words (N0)', measured,
                                                    lambda u: u['words'] - u['n0_ref'], weighted=weighted), R)
    tables.add('words', 'S5', version, LessonSpearman(
        f'Spearman of the words feature and the reference words within lessons (at least {MIN_LESSON_ITEMS} items)',
        measured, lambda u: u['words'], lambda u: u['n0_ref']), R)
    for t in S.THRESHOLDS:
        tables.add('words', 'S5', version, SumRatio(f'sensitivity of speech_ratio > {t:g} to speech heard',
                                                    lambda u: ratio(u) and u['status'] != 'none',
                                                    (lambda t: lambda u: u['speech_ratio'] > t)(t)), R)
        tables.add('words', 'S5', version, SumRatio(f'specificity of speech_ratio > {t:g} to speech heard',
                                                    lambda u: ratio(u) and u['status'] == 'none',
                                                    (lambda t: lambda u: u['speech_ratio'] <= t)(t)), R)
        tables.add('words', 'S5', version, S.Kappa(f'kappa of speech_ratio > {t:g} and speech heard', ratio,
                                                   lambda u: u['status'] != 'none',
                                                   (lambda t: lambda u: u['speech_ratio'] > t)(t), (False, True)), R)


def _question(score: Callable, field: str, positive: str, negative: str):
    return (lambda u: score(u) is not None), (lambda u: u[field] == positive), (lambda u: u[field] == negative)


def content(tables: Tables, version: str, units: list[dict], arms: dict[str, Arm], refs: dict[str, dict]) -> None:
    """per content arm: P2 (the primary arm; S12 for the others) C and the AUROC of each score against the blind
    judgements, by stratum, and on R alone weighted (a sensitivity); P3 with the arm's re-scored references: C_ref,
    delta = C_ref - C_ASR (paired), TR, and the same at context 0; S7: the scores of the ASR text against those of
    the reference text (CCC, Spearman, MAE, kappa of the peer letter and of teacher > 0.5) at context 1, context 0
    and in N1's surface form, and the reproducibility of the arm (max |delta| of its text scored again)"""
    R = [u for u in units if u['stratum'] == 'R']
    others = sum(1 for u in units if u['peer'] == 'other')
    for name, arm in arms.items():
        if arm.refused:
            continue
        analysis = 'P2' if name == PRIMARY_ARM else 'S12'
        for q, key, field, pos, neg in QUESTIONS:
            score = _arm(name, '1', key)
            keep, positive, negative = _question(score, field, pos, neg)
            title = f'{q}: C of {key}, {field} = {pos} against {neg}'
            descriptive = key == 'peer_other' and others < MIN_POSITIVES
            band = few if descriptive else band_p2a if q == 'P2a' else None
            tables.add('content', analysis, version, Concordance(title, keep, score, positive, negative), units,
                       arm=name, lolo=analysis == 'P2', band=band)
            tables.add('content', analysis, version, PooledAUC(f'{q}: AUROC of {key} (every pair)', keep, score,
                                                               positive, negative), units, arm=name)
            for stratum in ('R', 'S'):
                chosen = [u for u in units if u['stratum'] == stratum]
                tables.add('content', analysis, version, Concordance(title, keep, score, positive, negative), chosen,
                           by='stratum', group=stratum, arm=name)
            tables.add('sensitivity', f'{analysis} on R alone', version,
                       Concordance(title, keep, score, positive, negative, weighted=True), R, arm=name)
        reference_content(tables, version, units, name, refs.get(name), others)


def reference_content(tables: Tables, version: str, units: list[dict], name: str, ref: dict | None, others: int) -> None:
    """P3 (S12 for an arm not the primary) and S7 of one content arm against its re-scored references"""
    if ref is None:
        return
    analysis = 'P3' if name == PRIMARY_ARM else 'S12'
    for label, context, variant in (('', '1', 'ref_ctx1'), (', context 0', '0', 'ref_ctx0')):
        for q, key, field, pos, neg in QUESTIONS:
            asr, human = _arm(name, context, key), _ref(name, variant, key)
            positive, negative = (lambda f, p: lambda u: u[f] == p)(field, pos), (lambda f, p: lambda u: u[f] == p)(field, neg)
            both = (lambda a, h: lambda u: a(u) is not None and h(u) is not None)(asr, human)
            c_asr = Concordance(f'{q}: C_ASR of {key}{label}', both, asr, positive, negative)
            c_ref = Concordance(f'{q}: C_ref of {key}{label}', both, human, positive, negative)
            descriptive = key == 'peer_other' and others < MIN_POSITIVES
            main = analysis if not label else f'{analysis}{label}'
            tables.add('content', main, version, c_asr, units, arm=name)
            tables.add('content', main, version, c_ref, units, arm=name)
            tables.delta('content', main, version, f'{q}: delta = C_ref - C_ASR of {key}{label} (paired)', c_ref, units,
                         c_asr, units, arm=name, lolo=not label, band=few if descriptive else band_p3)
            tables.add('content', main, version, Transfer(f'{q}: TR of {key}{label}', c_asr, c_ref), units, arm=name,
                       band=few if descriptive else band_tr)
    for label, asr_of, ref_of in (('context 1', lambda k: _arm(name, '1', k), lambda k: _ref(name, 'ref_ctx1', k)),
                                  ('context 0', lambda k: _arm(name, '0', k), lambda k: _ref(name, 'ref_ctx0', k)),
                                  ("N1's surface form", lambda k: _ref(name, 'asr_n1', k), lambda k: _ref(name, 'ref_n1', k))):
        for key in SCORES:
            x, y = asr_of(key), ref_of(key)
            both = (lambda x, y: lambda u: x(u) is not None and y(u) is not None)(x, y)
            tables.add('content', 'S7', version, CCC(f'CCC of {key}', both, x, y), units, by='invariance', group=label,
                       arm=name)
            tables.add('content', 'S7', version, Spearman(f'Spearman of {key}', both, x, y), units, by='invariance',
                       group=label, arm=name)
            tables.add('content', 'S7', version, SumRatio(f'MAE of {key}', both, (lambda x, y: lambda u: abs(x(u) - y(u)))(x, y)),
                       units, by='invariance', group=label, arm=name)
        variant_asr, variant_ref = {'context 1': ((name, '1'), 'ref_ctx1'), 'context 0': ((name, '0'), 'ref_ctx0'),
                                    "N1's surface form": (None, 'ref_n1')}[label]

        def asr_values(u, at=variant_asr):
            return u['scores'].get(at) if at else u['refs'].get((name, 'asr_n1'))

        def ref_values(u, variant=variant_ref):
            return u['refs'].get((name, variant))

        def asked(u, a=asr_values, r=ref_values):
            return peer_letter(a(u)) is not None and peer_letter(r(u)) is not None

        tables.add('content', 'S7', version, S.Kappa('kappa of the peer letter', asked, lambda u, r=ref_values: peer_letter(r(u)),
                                                     lambda u, a=asr_values: peer_letter(a(u)), 'ABCD'),
                   units, by='invariance', group=label, arm=name)

        def teacher_asked(u, a=asr_values, r=ref_values):
            return all(v is not None and not v.get('no_text') and v.get('teacher') is not None for v in (a(u), r(u)))
        tables.add('content', 'S7', version, S.Kappa('kappa of teacher > 0.5', teacher_asked,
                                                     lambda u, r=ref_values: r(u)['teacher'] > 0.5,
                                                     lambda u, a=asr_values: a(u)['teacher'] > 0.5, (False, True)),
                   units, by='invariance', group=label, arm=name)
    # the arm's own text scored again: the largest difference from the arm's scores
    for key in SCORES:
        diffs = [abs(_ref(name, 'asr_ctx1', key)(u) - _arm(name, '1', key)(u)) for u in units
                 if _ref(name, 'asr_ctx1', key)(u) is not None and _arm(name, '1', key)(u) is not None]
        tables.pooled['content'].append({'check': 'content', 'version': version, 'analysis': 'S7', 'arm': name,
                                         'by': 'reproducibility', 'group': 'context 1',
                                         'metric': f'max |delta| of {key}, the text scored again',
                                         'value': S._value(max(diffs)) if diffs else None, 'n': len(diffs)})


def worded(u) -> bool:
    """whether the transcript has a word (N1) to take a dominant source from"""
    return u['consumed']['N'] >= 1


def who_speaks(tables: Tables, version: str, units: list[dict], comparison: Comparison | None,
               everyone: dict[str, list[dict]]) -> None:
    """P4 the primary's shares of who speaks where the version measured speech (the lessons with a group microphone,
    and every lesson), against the comparison audit's primary (paired) with Cohen's kappa; S10 the comparison's
    other names, test-retest where a name answered both audits, the dominant source of the transcript's words
    against who (on the items with a word), and other_offtask against peer_other where peer is not other"""
    R = [u for u in units if u['stratum'] == 'R']

    def measured(u):
        return u['speech_ratio'] is not None and u['speech_ratio'] > 0 and u['who'] in WHO
    mic = [u for u in R if measured(u) and u['audio_kind'] != 'mix']
    every = [u for u in R if measured(u)]
    for analysis, chosen in (('P4', mic), ('P4, every lesson', every)):
        for weighted in (True, False):
            for who in WHO:
                tables.add('who', analysis, version, SumRatio(f'share {who}', lambda u: True,
                                                              (lambda w: lambda u: u['who'] == w)(who), weighted=weighted),
                           chosen, lolo=analysis == 'P4' and weighted)
    if comparison is not None and comparison.primary:
        cp = comparison.primary
        joined = [u for u in mic if u['compare'].get(cp) in WHO]
        for who in WHO:
            tables.delta('who', 'P4', version, f'share {who}: the primary - {cp} (weighted, paired)',
                         SumRatio(f'share {who}', lambda u: True, (lambda w: lambda u: u['who'] == w)(who), weighted=True),
                         joined, SumRatio(f'share {who} of {cp}', lambda u: True,
                                          (lambda w: lambda u: u['compare'][cp] == w)(who), weighted=True), joined,
                         lolo=True)
        tables.add('who', 'P4', version, S.Kappa(f'kappa against {cp}, four classes', lambda u: True, lambda u: u['who'],
                                                 lambda u: u['compare'][cp], WHO), joined)
        tables.add('who', 'P4', version, S.Kappa(f'kappa against {cp}, member against not', lambda u: True,
                                                 lambda u: u['who'] == 'member', lambda u: u['compare'][cp] == 'member',
                                                 (False, True)), joined)
        for name in comparison.names:
            if name == cp:
                continue
            chosen = [u for u in R if u['who'] in WHO and u['compare'].get(name) in WHO]
            tables.add('who', 'S10', version, S.Kappa(f'kappa against {name}, four classes', lambda u: True,
                                                      lambda u: u['who'], (lambda n: lambda u: u['compare'][n])(name), WHO),
                       chosen)
        for name, their in everyone.items():
            if name in comparison.names:
                chosen = [u for u in their if u['who'] in WHO and u['compare'].get(name) in WHO]
                tables.add('who', 'S10', version, S.Kappa(f'kappa of {name} against their own {comparison.id} answers '
                                                          '(test-retest)', lambda u: True, lambda u: u['who'],
                                                          (lambda n: lambda u: u['compare'][n])(name), WHO), chosen)
    # a dominant source by word share needs words: an item without a reference word would agree or differ by design
    words = [u for u in R if worded(u) and u['who'] in WHO and u['dominant'] in WHO]
    tables.add('who', 'S10', version, SumRatio("the transcript's dominant source is who", lambda u: True,
                                               lambda u: u['dominant'] == u['who']), words)
    tables.add('who', 'S10', version, S.Kappa("kappa of the transcript's dominant source and who", lambda u: True,
                                              lambda u: u['who'], lambda u: u['dominant'], WHO), words)
    score = _arm(PRIMARY_ARM, '1', 'peer_other')
    tables.add('who', 'S10', version, Concordance(
        'C of peer_other, other_offtask = yes against no, where peer is not other',
        lambda u: u['peer'] != 'other' and score(u) is not None, score, lambda u: u['other_offtask'] == 'yes',
        lambda u: u['other_offtask'] == 'no'), units, arm=PRIMARY_ARM)


def confusions(units: list[dict], comparison: Comparison | None, everyone: dict[str, list[dict]]) -> dict:
    """who speaks against each comparison name, test-retest, and against the transcript's dominant source (on the
    items with a word)"""
    def matrix(pairs):
        return [[sum(1 for a, b in pairs if a == r and b == c) for c in ANSWERED_WHO] for r in ANSWERED_WHO]
    R = [u for u in units if u['stratum'] == 'R']
    out = {'who_against_dominant_source': {'rows': list(ANSWERED_WHO), 'columns': list(ANSWERED_WHO),
                                           'matrix': matrix([(u['who'], u['dominant']) for u in R if worded(u)])}}
    if comparison is not None:
        for name in comparison.names:
            pairs = [(u['who'], u['compare'].get(name)) for u in R if u['compare'].get(name) in ANSWERED_WHO]
            out[f'who_against_{name}'] = {'rows': list(ANSWERED_WHO), 'columns': list(ANSWERED_WHO), 'matrix': matrix(pairs)}
        for name, their in everyone.items():
            if name in comparison.names:
                pairs = [(u['who'], u['compare'].get(name)) for u in their if u['compare'].get(name) in ANSWERED_WHO]
                out[f'test_retest_{name}'] = {'rows': list(ANSWERED_WHO), 'columns': list(ANSWERED_WHO),
                                              'matrix': matrix(pairs)}
    return out


def coder_labels(tables: Tables, version: str, units: list[dict], coders: list[str]) -> None:
    """S8: the heard topic (peer) of the items each coder labelled social or collaborative, kappa of social and
    collaborative against other and task, and C of peer_other for coder-social against coder-collaborative"""
    score = _arm(PRIMARY_ARM, '1', 'peer_other')
    for coder in coders:
        def label(u, c=coder):
            return (u['labels'] or {}).get(c)
        for value in ('social', 'collaborative'):
            for peer in PEER:
                tables.add('coders', 'S8', version, SumRatio(
                    f'peer = {peer} of the items {coder} labelled {value}', (lambda v, f=label: lambda u: f(u) == v)(value),
                    (lambda p: lambda u: u['peer'] == p)(peer)), units, by='coder', group=coder)
        tables.add('coders', 'S8', version, S.Kappa(
            f"kappa of {coder}'s social and collaborative against the heard other and task",
            lambda u, f=label: f(u) in ('social', 'collaborative') and u['peer'] in ('other', 'task'),
            lambda u, f=label: 'other' if f(u) == 'social' else 'task', lambda u: u['peer'], ('other', 'task')),
            units, by='coder', group=coder)
        tables.add('coders', 'S8', version, Concordance(
            f'C of peer_other, {coder} social against collaborative', lambda u: score(u) is not None, score,
            lambda u, f=label: f(u) == 'social', lambda u, f=label: f(u) == 'collaborative'), units, by='coder',
            group=coder, arm=PRIMARY_ARM)


def reveal(tables: Tables, version: str, units: list[dict]) -> None:
    """S11: the reveal ratings on R (weighted) and on S, of `version`, the one the reveal showed"""
    for stratum, weighted in (('R', True), ('R', False), ('S', False)):
        chosen = [u for u in units if u['stratum'] == stratum]
        for field, values in (('gist', GIST), ('invented', INVENTED)):
            for value in values:
                tables.add('reveal', 'S11', version, SumRatio(
                    f'{field} = {value}', (lambda f: lambda u: u[f] is not None)(field),
                    (lambda f, v: lambda u: u[f] == v)(field, value), weighted=weighted), chosen, by='stratum', group=stratum)


def sensitivities(tables: Tables, version: str, R: list[dict], populations: dict) -> None:
    """the declared sensitivities of P1: N0, the edge tolerance, split and merge, the orders of overlapping lines,
    without the items with [x] or overlap, with the insertions of the items without reference words, the lesson
    macro mean, and sweep 1 alone"""
    for prefix, title in (('n0_', 'WER, N0'), ('tol_', 'WER, edge tolerance 0.5 s'), ('sm_', 'WER, split and merge'),
                          ('ovl_', 'WER, the least over the orders of overlapping lines')):
        for weighted in (True, False):
            tables.add('sensitivity', 'P1', version, wer(title, prefix=prefix, weighted=weighted), R)
    for weighted in (True, False):
        tables.add('sensitivity', 'P1', version, wer('WER without the items with [x] or overlap', weighted=weighted,
                                                     keep=lambda u: not u['markers_x'] and not u['overlap']), R)
        tables.add('sensitivity', 'P1', version, SumRatio('WER with the insertions of the items without reference words',
                                                          lambda u: True, _get('consumed', 'E'), _get('consumed', 'N'),
                                                          weighted), R)
    tables.add('sensitivity', 'P1', version, MacroMean('WER, the lesson macro mean', wer('WER')), R)
    first = weigh([u for u in R if u['sweep'] == 1], populations)
    for weighted in (True, False):
        tables.add('sensitivity', 'P1', version, wer('WER, sweep 1 alone', weighted=weighted), first, by='sweep', group='1')


def core_primaries(tables: Tables, check: str, analysis: str, version: str, units: list[dict], refs: dict) -> None:
    """P1 to P4 at their headline (a sensitivity's rows): the weighted WER, C of the primary arm, delta, the shares"""
    R = [u for u in units if u['stratum'] == 'R']
    tables.add(check, analysis, version, wer('WER', weighted=True), R, lolo=True)
    for q, key, field, pos, neg in QUESTIONS:
        score = _arm(PRIMARY_ARM, '1', key)
        keep, positive, negative = _question(score, field, pos, neg)
        tables.add(check, analysis, version, Concordance(f'{q}: C of {key}', keep, score, positive, negative), units,
                   arm=PRIMARY_ARM)
        if PRIMARY_ARM in refs:
            human = _ref(PRIMARY_ARM, 'ref_ctx1', key)
            both = (lambda a, h: lambda u: a(u) is not None and h(u) is not None)(score, human)
            tables.delta(check, analysis, version, f'{q}: delta = C_ref - C_ASR of {key} (paired)',
                         Concordance('ref', both, human, positive, negative), units,
                         Concordance('asr', both, score, positive, negative), units, arm=PRIMARY_ARM)
    mic = [u for u in R if u['speech_ratio'] is not None and u['speech_ratio'] > 0 and u['who'] in WHO
           and u['audio_kind'] != 'mix']
    for who in WHO:
        tables.add(check, analysis, version, SumRatio(f'share {who}', lambda u: True,
                                                      (lambda w: lambda u: u['who'] == w)(who), weighted=True), mic)


def version_analyses(tables: Tables, version: str, units: list[dict], arms: dict, refs: dict, populations: dict,
                     scored: bool) -> None:
    """every analysis of one version's units; the version scored also gets the sources, the S stratum alone and
    the sensitivities (the other version's rows are its P1, S1, S2, S5 and content); the reveal is the reveal
    version's (score)"""
    R = [u for u in units if u['stratum'] == 'R']
    accuracy(tables, 'accuracy', 'P1', version, R, lolo=True)
    content_words(tables, version, R)
    detection(tables, 'detection', 'S1', version, R)
    group_source(tables, version, R)
    words_feature(tables, version, R)
    content(tables, version, units, arms, refs)
    if scored:
        sources(tables, version, R)
        S_units = [u for u in units if u['stratum'] == 'S']
        accuracy(tables, 'social', 'S13', version, S_units, weights=(False,))
        detection(tables, 'social', 'S13', version, S_units, weights=(False,))
        sensitivities(tables, version, R, populations)


def kept_by_both(metric, a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict]]:
    """each side's units of the items `metric` keeps in both (the group text, a content score: what one version has
    and the other lacks), so that a paired difference compares the same items"""
    items = {u['item'] for u in a if _keeps(metric, u)} & {u['item'] for u in b if _keeps(metric, u)}
    return [u for u in a if u['item'] in items], [u for u in b if u['item'] in items]


def version_deltas(tables: Tables, units: dict[str, list[dict]], populations: dict) -> None:
    """S3: V1 - V0 on the items both versions hold, paired, each difference on the items its metric keeps in both
    (the weights those of the items both versions hold)"""
    common = {u['item'] for u in units['rerun']} & {u['item'] for u in units['reported']}
    v1 = weigh([u for u in units['rerun'] if u['item'] in common], populations)
    v0 = weigh([u for u in units['reported'] if u['item'] in common], populations)
    R1, R0 = [u for u in v1 if u['stratum'] == 'R'], [u for u in v0 if u['stratum'] == 'R']
    pairs = [(wer('WER', weighted=True), wer('WER', weighted=False), True),
             (rate('CER', 'consumed', 'cE', 'cN', True), None, False),
             (wer('WER of the group source', 'group', weighted=True), None, False),
             (SumRatio('miss rate (speech, N >= 1)', lambda u: u['status'] == 'speech' and u['consumed']['N'] >= 1,
                       _get('consumed', 'empty'), weighted=True), None, False),
             (SumRatio('phantom rate (none: at least 1 hypothesis token)', lambda u: u['status'] == 'none',
                       lambda u: u['consumed']['hyp'] >= 1, weighted=True), None, False)]
    for metric, unweighted, lolo in pairs:
        for m in (metric, unweighted):
            if m is not None:
                a, b = kept_by_both(m, R1, R0)
                tables.delta('versions', 'S3', 'rerun - reported', f'{m.name}: V1 - V0 (paired)', m, a, m, b, lolo=lolo)
    for q, key, field, pos, neg in QUESTIONS:
        score = _arm(PRIMARY_ARM, '1', key)
        keep, positive, negative = _question(score, field, pos, neg)
        metric = Concordance(f'{q}: C of {key}', keep, score, positive, negative)
        a, b = kept_by_both(metric, v1, v0)
        tables.delta('versions', 'S3', 'rerun - reported', f'{q}: C of {key}: V1 - V0 (paired)', metric, a, metric, b,
                     arm=PRIMARY_ARM)


# ---- the two transcribers ----

def pair_counts(designs: dict, answers: dict, a: str, b: str, items: set) -> dict[str, dict]:
    """per item both answered, the counts of b's transcript against a's as the reference ('ab') and of a's against
    b's ('ba'), N1 and over the characters"""
    out = {}
    for design in designs.values():
        for it in design['speech']:
            if it['item'] not in items:
                continue
            refs = []
            for name in (a, b):
                record = ((answers.get(name) or {}).get(it['item']) or {}).get('transcribe') or {}
                lines = X.parse_reference((record.get('answer') or {}).get('transcript') or '')
                refs.append([token for line in lines for token in X.normalise(line.tokens)])
            out[it['item']] = {'ab': {**_counts(X.align(refs[0], refs[1])), **_counts(X.cer(refs[0], refs[1]), 'c')},
                               'ba': {**_counts(X.align(refs[1], refs[0])), **_counts(X.cer(refs[1], refs[0]), 'c')}}
    return out


def interauditor(designs: dict, answers: dict, primary: str, mine: list[dict], partners: dict[str, list[dict]],
                 boot: int, seed: int) -> list[dict]:
    """S9 per partner, on the items both answered (practice and flagged ones left out): the WER of each against the
    other as the reference, the system's WER against each on the same items and its excess over that human floor
    (paired), and Cohen's kappa of status, who, adult, peer, other_offtask and overlap"""
    rows = []
    by_item = {u['item']: u for u in mine}
    for name, theirs in sorted(partners.items()):
        shared = [(by_item[u['item']], u) for u in theirs if u['item'] in by_item]
        counts = pair_counts(designs, answers, primary, name, {a['item'] for a, _ in shared})
        units = [{'lesson': a['lesson'], 'item': a['item'], 'ab': counts[a['item']]['ab'], 'ba': counts[a['item']]['ba'],
                  'sa': a['consumed'], 'sb': b['consumed'], 'a': a, 'b': b} for a, b in shared]
        floor_a = SumRatio(f'WER of {name} against {primary}', lambda u: u['ab']['N'] >= 1, lambda u: u['ab']['E'],
                           lambda u: u['ab']['N'])
        floor_b = SumRatio(f'WER of {primary} against {name}', lambda u: u['ba']['N'] >= 1, lambda u: u['ba']['E'],
                           lambda u: u['ba']['N'])
        system_a = SumRatio(f'WER of the system against {primary}', lambda u: u['ab']['N'] >= 1, lambda u: u['sa']['E'],
                            lambda u: u['sa']['N'])
        system_b = SumRatio(f'WER of the system against {name}', lambda u: u['ba']['N'] >= 1, lambda u: u['sb']['E'],
                            lambda u: u['sb']['N'])
        metrics = [floor_a, floor_b, system_a, system_b,
                   SumRatio(f'CER of {name} against {primary}', lambda u: u['ab']['N'] >= 1, lambda u: u['ab']['cE'],
                            lambda u: u['ab']['cN']),
                   SumRatio(f'CER of {primary} against {name}', lambda u: u['ba']['N'] >= 1, lambda u: u['ba']['cE'],
                            lambda u: u['ba']['cN'])]
        for metric in metrics:
            result = evaluate(metric, units, boot, seed)
            rows.append({'auditor': name, 'question': result['metric'], 'n': result['n'], 'value': result['value'],
                         'lo': result['lo'], 'hi': result['hi'], 'interval': result['interval']})
        for system, floor, whom in ((system_a, floor_a, primary), (system_b, floor_b, name)):
            result = paired(f'the system\'s excess over the floor against {whom} (paired)', system, units, floor, units,
                            boot, seed)
            rows.append({'auditor': name, 'question': result['metric'], 'n': result['n'], 'value': result['value'],
                         'lo': result['lo'], 'hi': result['hi'], 'interval': result['interval']})
        for field in ('status', 'who', 'adult', 'peer', 'other_offtask', 'overlap'):
            pairs = [(str(u['a'][field]), str(u['b'][field])) for u in units]
            categories = sorted({v for pair in pairs for v in pair})
            kappa = S.Kappa(f'kappa of {field}', lambda u: True, (lambda f: lambda u: str(u['a'][f]))(field),
                            (lambda f: lambda u: str(u['b'][f]))(field), categories)
            result = evaluate(kappa, units, boot, seed) if len(categories) > 1 else {'value': None, 'lo': None,
                                                                                    'hi': None, 'interval': None}
            rows.append({'auditor': name, 'question': f'kappa of {field}', 'n': len(pairs),
                         'agree': round(sum(1 for x, y in pairs if x == y) / len(pairs), 4) if pairs else None,
                         'kappa': result['value'], 'value': result['value'], 'lo': result['lo'], 'hi': result['hi'],
                         'interval': result['interval']})
    return rows


# ---- writing ----

def _header(context: dict) -> str:
    """the line that heads every output: never a word of a transcript"""
    closes = ', '.join(f'{name} at seq {row.get("seq")}' for name, row in sorted(context['closes'].items())) or 'none'
    parts = [f"# transcription audit {context['audit_id']}", f"plan sha256 {context['plan_sha256']}",
             f"anchors sha256 {context['anchors_sha256']}",
             f"version {context['version']} ({context['role']})", f"primary auditor {context['auditor']}",
             f"reliability {', '.join(context['partners']) or 'none'}",
             'roles: ' + (', '.join(f"note at seq {n['seq']}" for n in context['roles_notes']) or 'no note'),
             f'blind closes (logged): {closes}']
    parts += context.get('read_checks') or []
    if context.get('roles_warning'):
        parts.append(f"WARNING: {context['roles_warning']}")
    if context.get('interim'):
        parts.append("INTERIM: scored before the primary's blind close (--audit-interim)")
    if context.get('open'):
        parts.append('the audit was open (names typed, not authenticated)')
    parts.append(f"lines left out after a blind close {sum(context['late'].values())}")
    log = context.get('log') or {}
    parts.append('request log ' + ('DOES NOT VERIFY, scored anyway' if log.get('despite_failure') else
                                   f"{len(log.get('cuts') or [])} crash cut(s)" if log.get('cuts') else 'verified')
                 + f", head {log.get('head')}")
    if context['deviations']:
        parts.append('deviations: ' + ' | '.join(context['deviations']))
    return '; '.join(parts).replace('\n', ' ')


def _plain(value):
    if isinstance(value, bool):
        return int(value)
    return '' if value is None else value


def _csv(path: Path, context: dict, rows: list[dict], columns: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        handle.write(_header(context) + '\n')
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _plain(row.get(k)) for k in columns})


def unit_rows(units: list[dict]) -> tuple[list[dict], list[str]]:
    """the units as rows per item and source, counts, answers, measures and scores only"""
    scores = sorted({f'{arm}_ctx{context}_{key}' for u in units for (arm, context) in u['scores']
                     for key in SCORES + ('no_text',)})
    refs = sorted({f'{arm}_{variant}_{key}' for u in units for (arm, variant) in u['refs'] for key in SCORES + ('no_text',)})
    coders = sorted({coder for u in units for coder in u['labels']})
    names = sorted({name for u in units for name in u['compare']})
    columns = (UNIT_FIELDS + ['source'] + COUNT_FIELDS + [f'label_{c}' for c in coders] + ['compare_item']
               + [f'compare_{n}' for n in names] + scores + refs)
    rows = []
    for u in units:
        base = {k: u.get(k) for k in UNIT_FIELDS}
        base.update({f'label_{c}': u['labels'].get(c) for c in coders}, compare_item=u['compare_item'])
        base.update({f'compare_{n}': u['compare'].get(n) for n in names})
        for (arm, context), values in u['scores'].items():
            base.update({f'{arm}_ctx{context}_{k}': values.get(k) for k in SCORES + ('no_text',)})
        for (arm, variant), values in u['refs'].items():
            base.update({f'{arm}_{variant}_{k}': values.get(k) for k in SCORES + ('no_text',)})
        for source in SOURCES:
            if u[source] is not None:
                rows.append({**base, 'source': source, **{k: u[source].get(k) for k in COUNT_FIELDS}})
    return rows, columns


def _fmt(row: dict) -> str:
    if row.get('value') is None:
        return f"{'-':>8} (n {row.get('n')})"
    text = f"{row['value']:>8.3f} (n {row.get('n')}"
    if row.get('positives') is not None:
        text += f", {row['positives']} positive, {row['negatives']} negative"
    text += ')'
    if row.get('lo') is not None:
        text += f" [{row['lo']}, {row['hi']}]"
    if row.get('lolo_lo') is not None:
        text += f" leaving a lesson out [{row['lolo_lo']}, {row['lolo_hi']}]"
    if row.get('band'):
        text += f"  {row['band']}"
    return text


def report_text(context: dict, tables: Tables, agreement: list[dict], confusion: dict, left: dict) -> str:
    lines = [_header(context), '', 'ANCHORS'] + [f'    {name:<28} {digest}' for name, digest in context['anchors']]
    lines += ['', 'ROLES', f"    plan primary: {context['plan_primary'] or 'none'}; declared roles: "
                           f"{'; '.join(context['declared_roles']) or 'none'}"]
    lines += [f"    note at seq {n['seq']}: {n['text'][:300]}".replace('\n', ' ') for n in context['roles_notes']]
    if context.get('roles_warning'):
        lines.append(f"    WARNING: {context['roles_warning']}")
    lines += ['', 'BLIND CLOSES'] + ([f"    {name:<24} seq {row.get('seq')} at {row.get('at')}"
                                     for name, row in sorted(context['closes'].items())] or ['    none'])
    lines += ['', 'CONTENT ARMS'] + ([f"    {version} {name}: {row['origin']} {row['file']} sha256 {row['sha256']}, "
                                      f"{row['rows']} rows, {row['missing']} missing, {row['mismatched']} mismatched"
                                      + (f"; REFUSED: {row['refused']}" if row['refused'] else '')
                                      + (f"; {row['note']}" if row['note'] else '')
                                      for version, arms in context['arms'].items() for name, row in arms.items()]
                                     or ['    none'])
    for version, refs in context['refs'].items():
        for name, ref in refs.items():
            lines.append(f"    {version} {name} references: {ref['file']} sha256 {ref['sha256']}, scorer sha256 "
                         f"{ref['meta'].get('scorer_sha256')}")
            lines += [f"      {variant}: {v['rows']} rows, {v['missing']} missing, {v['mismatched']} mismatched"
                      + (f"; REFUSED: {v['refused']}" if v['refused'] else '') for variant, v in ref['variants'].items()]
    lines += ['', f"declared analyses ({len(context['declared'])}):"] + [f'  - {d}' for d in context['declared']]
    reliability = ['', 'RELIABILITY (S9)'] + ([
        f"    {r['auditor']:<16} {r['question']:<60} n {r['n']:<5}"
        + (f" agree {r['agree']}" if r.get('agree') is not None else '') + f" {_fmt(r)}" for r in agreement]
        or ['    no second auditor answered the same items'])
    for check, title in CHECKS:
        if check == 'reveal':
            # the report's order: who speaks, then the agreement of the transcribers, then the reveal
            lines += reliability
            reliability = []
        rows = tables.pooled.get(check) or []
        if not rows and check != 'content':
            continue
        lines += ['', title]
        if check == 'content':
            for version, arms in context['arms'].items():
                if PRIMARY_ARM not in arms or arms[PRIMARY_ARM]['refused']:
                    lines.append(f'  {VERSION_NAMES.get(version, version)}: P2 and P3 not available (no {PRIMARY_ARM} arm of '
                                 "this version's text: --audit-content-scores)")
        for key in dict.fromkeys((r['version'], r['analysis'], r.get('arm'), r['by'], r['group']) for r in rows):
            version, analysis, arm, by, group = key
            lines.append(f"  {VERSION_NAMES.get(version, version)}, {analysis}" + (f', arm {arm}' if arm else '')
                         + ('' if by == 'pooled' else f', {by} {group}'))
            lines += [f"    {r['metric']:<84} {_fmt(r)}" for r in rows
                      if (r['version'], r['analysis'], r.get('arm'), r['by'], r['group']) == key]
        if check == 'who':
            for name, data in confusion.items():
                lines += [f'  CONFUSION {name} (rows the auditor, columns {", ".join(data["columns"])})']
                lines += ['    ' + ' '.join(f'{v:>5}' for v in row) for row in data['matrix']]
        if check == 'sensitivity':
            lines.append(f'  map v2: none (the variant map is audit_text v{X.NORMALISER_VERSION}\'s, no later addition)')
    lines += reliability
    lines += ['', 'SKIPS (R items left unanswered before an item answered after them, in the order the page served them)']
    lines += [f'    {lesson}: {n}' for lesson, n in sorted(context['skips'].items()) if n] or ['    none']
    lines += ['', 'LEFT OUT'] + [f'    {k}: {v}' for k, v in sorted(left.items())]
    return '\n'.join(lines) + '\n'


# ---- the score ----

def excluded_sessions(plan: dict, spec: str | None) -> list[str]:
    """--audit-exclude-sessions ID,...: the sessions (ids or aliases) of the S-X sensitivity"""
    by_alias = {s['alias']: s['id'] for s in plan['sessions']}
    ids = {s['id'] for s in plan['sessions']}
    out = []
    for name in (n.strip() for n in (spec or '').split(',')):
        if not name:
            continue
        sid = by_alias.get(name, name)
        if sid not in ids:
            raise A.AuditError(f'{name} is no session of audit {plan["audit_id"]} (--audit-exclude-sessions takes ids or '
                               'aliases)')
        out.append(sid)
    return out


def unread_problems(plan: dict, designs: dict, version: str, versions: dict) -> list[str]:
    """of the version the content model read (every version but the display one), the sessions frozen without its
    checks: every window's text against the reference texts and a content arm's cur_sha (audit_speech.read_problems)"""
    if not SP.read_version(plan, version):
        return []
    return [f"{designs[sid]['alias']} {version}: {problem}" for sid, data in sorted(versions.items())
            if not data.get('refused') for problem in SP.read_problems(data, designs[sid])]


def read_checks(plan: dict, versions: dict[str, dict]) -> list[str]:
    """per version the content model read, what its text was checked against: the reference texts' sha256 and the
    arms whose cur_sha was checked, over the sessions frozen"""
    out = []
    for version, frozen in versions.items():
        kept = [d for d in frozen.values() if not d.get('refused')]
        if not SP.read_version(plan, version) or not kept:
            continue
        references = sorted({str((d['sources'].get('reference') or {}).get('sha256'))[:16] for d in kept})
        hashed = sorted({name for d in kept for name, arm in (d['sources'].get('content') or {}).items()
                         if arm.get('hashed') and arm.get('checked')})
        out.append(f"{version} checked against the texts the content model read (reference sha256 "
                   f"{', '.join(references)}) and the cur_sha of {', '.join(hashed)} in its {len(kept)} sessions")
    return out


def closes_of(artifacts: Path, audit_id: str) -> tuple[dict, list[str]]:
    """the blind closes as the request log holds them ({name: {'seq', 'at'}}), and the deviations where
    blind_closed.json, which the page acted on, is not them"""
    logged, last = SP.logged_closes(artifacts, audit_id)
    path = SP.blind_closed_path(artifacts, audit_id)
    problems = SP.close_problems(SP.read_blind_closed(artifacts, audit_id), logged)
    if not problems and logged and L.file_sha256(path) != last:
        problems.append(f'{SP.BLIND_CLOSED} changed since the last close logged it')
    return logged, [f'{problem}: scored by the logged closes' for problem in problems]


def _versions(artifacts: Path, plan: dict, audit_id: str, version: str) -> dict:
    out = {}
    for entry in plan['sessions']:
        path = A.version_file(artifacts, entry['id'], audit_id, version)
        if path.exists():
            out[entry['id']] = A.read_json(path)
    return out


def score(artifacts: Path, audit_id: str, version: str, auditor: str | None = None, boot: int = BOOT, seed: int = 0,
          allow_drift: bool = False, out: Path | None = None, despite_log_failure: bool = False, interim: bool = False,
          compare_with: str | None = None, content_spec: str | None = None, content_ref: str | None = None,
          exclude: str | None = None) -> tuple[Path, dict]:
    plan = A.load_plan(artifacts, audit_id)
    if not SP.transcript_plan(plan):
        raise A.AuditError(f'{audit_id} is a sensing audit: audit_score scores it')
    if version not in A.VERSIONS:
        raise A.AuditError(f"give --audit-version {'|'.join(A.VERSIONS)}: the version to score")
    folder = A.audit_dir(artifacts, audit_id)
    campaign = L.Campaign(folder)
    if not campaign.exists():
        raise A.AuditError(f'no campaign.yml in {campaign.folder}: whose answers count is not known')
    log = L.checked_log(campaign, artifacts, despite_log_failure, 'score')
    deviations = []
    frozen_normaliser = (plan.get('normaliser') or {}).get('sha256')
    if frozen_normaliser != L.file_sha256(X.__file__):
        why = (f'the normaliser (audit_text, sha256 {L.file_sha256(X.__file__)}) is not the one the plan froze '
               f'({frozen_normaliser})')
        if not allow_drift:
            raise A.AuditError(f'{why}: --audit-allow-drift scores with it anyway, named in every header')
        deviations.append(why)
    designs = {entry['id']: A.read_json(A.session_audit_dir(artifacts, entry['id'], audit_id) / A.DESIGN_FILE)
               for entry in plan['sessions']}
    frozen = {v: _versions(artifacts, plan, audit_id, v) for v in A.VERSIONS}
    if not frozen[version]:
        raise A.AuditError(f'no session of audit {audit_id} has {version} frozen (--audit-freeze {audit_id} '
                           f'--audit-version {version})')
    problems, after = S.frozen_checks(artifacts, audit_id, plan, version, frozen[version])
    if problems:
        raise A.AuditError('; '.join(problems) + ': the frozen outputs are not the ones logged, nothing is scored')
    unread = unread_problems(plan, designs, version, frozen[version])
    if unread:
        raise A.AuditError('; '.join(unread) + f': {version} is scored as the text the content model read only when '
                           'it was frozen against that text (--audit-asr-reference and a content arm with cur_sha), '
                           'nothing is scored')
    drifted = S.drift({sid: d for sid, d in frozen[version].items() if not d.get('refused')})
    if drifted and not allow_drift:
        raise A.AuditError('; '.join(drifted) + ' (--audit-allow-drift scores anyway, recorded)')
    deviations += [f'{d} (allowed)' for d in drifted]
    other = next(v for v in A.VERSIONS if v != version)
    if frozen[other]:
        other_problems, other_after = S.frozen_checks(artifacts, audit_id, plan, other, frozen[other])
        other_problems += unread_problems(plan, designs, other, frozen[other])
        other_drift = S.drift({sid: d for sid, d in frozen[other].items() if not d.get('refused')})
        if other_problems or (other_drift and not allow_drift):
            deviations.append(f"{other} is not scored beside: {'; '.join(other_problems + other_drift)}")
            frozen[other] = {}
        else:
            deviations += [f'{d} ({other}, allowed)' for d in other_drift]
            after = {'after_answers': {**after['after_answers'], **{f'{a} {other}': n for a, n in
                                                                   other_after['after_answers'].items()}},
                     'after_render': after['after_render'] + [f'{a} {other}' for a in other_after['after_render']]}
    for alias, count in sorted(after['after_answers'].items()):
        deviations.append(f'{alias} frozen after {count} scored answers (never the primary version)')
    if after['after_render']:
        deviations.append('frozen after the render: ' + ', '.join(sorted(after['after_render'])))
    index_path = folder / A.RENDER_INDEX
    render = A.read_json(index_path) if index_path.exists() else {}
    if render.get('after_answers'):
        deviations.append(f"rendered after {render['after_answers']} scored answers")
    data = campaign.data()
    opened = S.open_serves(artifacts, audit_id)
    closed, deviations_of_closes = closes_of(artifacts, audit_id)
    deviations += deviations_of_closes
    answers, left, scopes, info = counted_answers(artifacts, plan, data, bool(opened), closed)
    chosen = auditor or plan.get('primary')
    if opened:
        primary = S.primary_typed(answers, chosen, scopes)
    else:
        primary = S.primary_auditor(answers, chosen, data)
    if primary is None:
        raise A.AuditError(f'no answers by {chosen}' if chosen else
                           'no answers yet under a name that chose the full audit (or give --audit-auditor)')
    if plan.get('primary') and primary != plan['primary']:
        deviations.append(f"the plan named {plan['primary']} the primary; {primary} is scored as it (--audit-auditor)")
    partners = sorted(n for n in answers if n != primary and scopes.get(n) == {'reliability'})
    elsewhere = [n for n in answers if n != primary and n not in partners]
    if elsewhere:
        left['names other than the primary and the reliability subset, not scored'] = len(elsewhere)
    if primary not in closed:
        if not interim:
            raise A.AuditError(f"{primary}'s blind pass is not closed (--audit-close-blind {audit_id} --audit-auditor "
                               f"{primary}): no interim score (--audit-interim scores anyway, logged and named in every "
                               'header)')
        deviations.append(f"interim: {primary}'s blind pass is not closed")
    else:
        close = closed[primary]['seq']
        for name in partners:
            last = info['last_transcribe'].get(name)
            if last is not None and last > close:
                deviations.append(f"{primary} was closed at seq {close} before {name}'s blind pass ended (their last "
                                  f'transcription at seq {last})')
    notes = roles_notes(log_records(artifacts, audit_id))
    first = info['first_scored']
    roles_warning = None
    if first is not None and not any(isinstance(n['seq'], int) and n['seq'] < first for n in notes):
        roles_warning = f'no note of the roles (--log-note) precedes the first scored answer (seq {first})'
    excluded = excluded_sessions(plan, exclude)
    populations = populations_of(plan, designs)
    aliases = defaultdict(list)
    for entry in plan['sessions']:
        aliases[designs[entry['id']]['lesson']].append(entry['alias'])
    tables = Tables(boot, seed, {lesson: ' '.join(sorted(a)) for lesson, a in aliases.items()})
    comparison = Comparison(artifacts, compare_with, despite_log_failure) if compare_with else None
    normaliser = L.file_sha256(X.__file__)
    scored = [version] + ([other] if frozen[other] else [])
    # the reveal showed one version's text: the ratings are that version's, whichever is scored
    reveal_version = plan.get('reveal_version') or SP.PRIMARY_VERSION
    units, arms, refs, skips = {}, {}, {}, {}
    for v in scored:
        built, unit_left, lesson_skips, hashes = build_units(designs, frozen[v], render, answers, primary)
        if v == version:
            left.update(unit_left)
            skips = lesson_skips
        else:
            # the answers' reasons are the scored version's; only what the other version lacks is its own
            for reason in ('items of a session the version refused', 'items the version holds no text of'):
                if unit_left.get(reason):
                    left[f'{reason} ({other})'] = unit_left[reason]
        if v != reveal_version:
            for u in built:
                u['gist'] = u['invented'] = None
        built = weigh(built, populations)
        arms[v] = frozen_arms(designs, frozen[v])
        if v == version:
            for name, arm in scoring_arms(content_spec, designs, frozen[v]).items():
                if name in arms[v]:
                    raise A.AuditError(f'arm {name} is frozen with {v} already: name the arm read at scoring otherwise')
                arms[v][name] = arm
        refs[v] = reference_arms(content_ref, v, hashes, frozen_normaliser)
        attach_scores(built, arms[v], refs[v])
        if comparison is not None:
            comparison.attach(built)
        coders, changed = attach_labels(artifacts, plan, designs, built)
        if v == version and changed:
            deviations.append('the labels changed since the sample or cannot be read: ' + ', '.join(changed))
        units[v] = built
    mine = units[version]
    for name, arm in arms[version].items():
        if arm.refused:
            deviations.append(f'arm {name} refused: {arm.refused}')
    partner_units = {}
    for name in partners:
        theirs, _, _, _ = build_units(designs, frozen[version], render, answers, name)
        if comparison is not None:
            comparison.attach(theirs)
        partner_units[name] = theirs
    for v in scored:
        version_analyses(tables, v, units[v], arms[v], refs[v], populations, v == version)
    if reveal_version in units:
        reveal(tables, reveal_version, units[reveal_version])
    everyone = {primary: mine, **partner_units}
    who_speaks(tables, version, mine, comparison, everyone)
    coder_labels(tables, version, mine, coders)
    if len(scored) == 2:
        version_deltas(tables, units, populations)
    if excluded:
        kept = weigh([u for u in mine if u['session'] not in excluded], populations)
        core_primaries(tables, 'sensitivity', 'S-X without ' + ', '.join(designs[s]['alias'] for s in excluded),
                       version, kept, refs[version])
    agreement = interauditor(designs, answers, primary, mine, partner_units, boot, seed)
    confusion = confusions(mine, comparison, everyone)
    if excluded:
        deviations.append('S-X leaves out ' + ', '.join(designs[s]['alias'] for s in excluded) + ' (--audit-exclude-sessions)')
    if comparison is not None and comparison.log.get('despite_failure'):
        deviations.append(f'the request log of {comparison.id} does not verify, compared anyway')
    anchors = A.anchor_hashes(artifacts, audit_id)
    context = {'audit_id': audit_id, 'task': SP.TASK, 'plan_sha256': L.file_sha256(folder / A.PLAN_FILE),
               'anchors': anchors, 'anchors_sha256': L.sha256_hex(json.dumps(anchors)), 'version': version,
               'role': ('the declared primary, V1' if version == SP.PRIMARY_VERSION else 'the secondary row, V0')
               + (f'; {other} scored beside' if len(scored) == 2 else ''),
               'other': other if len(scored) == 2 else None, 'auditor': primary, 'partners': partners,
               'plan_primary': plan.get('primary'), 'declared_roles': list(plan.get('declared_roles') or []),
               'roles_notes': notes, 'roles_warning': roles_warning, 'closes': closed, 'interim': primary not in closed,
               'reveal_version': reveal_version, 'read_checks': read_checks(plan, {v: frozen[v] for v in scored}),
               'late': info['late'], 'deviations': deviations, 'skips': skips, 'excluded': excluded,
               'declared': plan.get('declared') or [], 'normaliser': {**(plan.get('normaliser') or {}), 'now': normaliser},
               'boot': boot, 'seed': seed, 'scored_at': C.now_utc(),
               'log': {k: log[k] for k in ('ok', 'message', 'head', 'seq', 'cuts', 'despite_failure')},
               'arms': {v: {name: arm.summary() for name, arm in arms[v].items()} for v in scored},
               'refs': {v: {name: {'file': ref['file'], 'sha256': ref['sha256'], 'meta': ref['meta'], 'variants': {
                   variant: {k: e[k] for k in ('rows', 'missing', 'mismatched', 'refused')}
                   for variant, e in ref['variants'].items()}} for name, ref in refs[v].items()} for v in scored},
               'compare': None if comparison is None else {
                   'audit_id': comparison.id, 'plan_sha256': comparison.plan_sha256, 'primary': comparison.primary,
                   'names': comparison.names, 'open': comparison.open,
                   'log': {k: comparison.log[k] for k in ('ok', 'message', 'head', 'seq', 'despite_failure')},
                   'joined': sum(1 for u in mine if u['compare_item'])}}
    if opened:
        context['open'] = {'served_open_at_seq': opened, 'names': {
            name: {'scope': S._scope_of(scopes.get(name, set())), 'items': len(answers[name]),
                   'role': 'primary' if name == primary else 'reliability' if name in partners else 'left out'}
            for name in sorted(answers)}, 'shared_browser': S.shared_browser(artifacts, audit_id, primary)['names']}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = Path(out) if out else folder / A.SCORES_DIR / f'{version}_{stamp}'
    target.mkdir(parents=True, exist_ok=True)
    for v in scored:
        rows, columns = unit_rows(units[v])
        _csv(target / f'units_{v}.csv', context, rows, columns)
    for check, rows in tables.pooled.items():
        _csv(target / f'pooled_{check}.csv', context, rows, POOLED_COLUMNS)
    for check, rows in tables.lessons.items():
        _csv(target / f'lesson_{check}.csv', context, rows, LESSON_COLUMNS)
    _csv(target / 'interauditor.csv', context, agreement, INTERAUDITOR_COLUMNS)
    if comparison is not None:
        names = comparison.names
        joined = [dict({k: u[k] for k in ('item', 'alias', 'session', 'lesson', 'window_start', 'stratum', 'rank',
                                          'sweep', 'who', 'compare_item')}, **{f'speaker_{n}': u['compare'].get(n)
                                                                               for n in names})
                  for u in mine if u['compare_item']]
        _csv(target / f'compare_{comparison.id}.csv', context, joined,
             ['item', 'alias', 'session', 'lesson', 'window_start', 'stratum', 'rank', 'sweep', 'who', 'compare_item']
             + [f'speaker_{n}' for n in names])
    for name in sorted({name for v in scored for name in arms[v]}):
        rows = [r for check in ('content', 'who', 'coders', 'sensitivity', 'versions') for r in tables.pooled.get(check, [])
                if r.get('arm') == name]
        _csv(target / f'content_{name}.csv', context, rows, POOLED_COLUMNS)
    summary = {**context, 'left_out': dict(left), 'confusion': confusion, 'interauditor': agreement,
               'tables': dict(tables.pooled)}
    A.write_json(target / 'summary.json', summary)
    (target / 'report.txt').write_text(report_text(context, tables, agreement, confusion, dict(left)), encoding='utf-8')
    return target, summary


def cmd_score(args, argv) -> int:
    artifacts, audit_id = A._artifacts(args), args.audit_score
    if not args.audit_version:
        raise A.AuditError(f'give --audit-version rerun|reported: the version to score ({SP.PRIMARY_VERSION} is the '
                           'declared primary)')
    boot = BOOT if args.audit_boot is None else args.audit_boot
    folder, summary = score(artifacts, audit_id, args.audit_version, args.audit_auditor, boot, A._size(args, 'boot_seed'),
                            bool(args.audit_allow_drift), Path(args.audit_out) if args.audit_out else None,
                            bool(args.despite_log_failure), bool(args.audit_interim), args.audit_compare_with,
                            args.audit_content_scores, args.audit_content_scores_ref, args.audit_exclude_sessions)
    A._event(artifacts, audit_id, 'audit-score', task=SP.TASK, version=args.audit_version, auditor=summary['auditor'],
             interim=bool(summary['interim']), compare_with=args.audit_compare_with,
             excluded=summary['excluded'], deviations=summary['deviations'], log_check=summary['log'],
             late=summary['late'], skips={lesson: n for lesson, n in summary['skips'].items() if n},
             content={v: {name: row['sha256'] for name, row in arms.items()} for v, arms in summary['arms'].items()},
             content_ref={v: {name: row['sha256'] for name, row in refs.items()} for v, refs in summary['refs'].items()},
             summary_sha256=L.file_sha256(folder / 'summary.json'), argv=list(argv),
             **({'open': True} if summary.get('open') else {}))
    print(f"transcription audit {audit_id}, {VERSION_NAMES.get(args.audit_version)} ({summary['role']}), primary auditor "
          f"{summary['auditor']}" + (', INTERIM' if summary['interim'] else '')
          + (', the audit was open (names typed, not authenticated)' if summary.get('open') else ''))
    if summary.get('roles_warning'):
        print(f"  WARNING: {summary['roles_warning']}")
    for deviation in summary['deviations']:
        print(f'  deviation: {deviation}')
    arms = summary['arms'].get(args.audit_version) or {}
    if PRIMARY_ARM not in arms or arms[PRIMARY_ARM]['refused']:
        print(f'  P2 and P3 not available for {args.audit_version}: no {PRIMARY_ARM} arm of its text (--audit-content-scores)')
    pooled = summary['tables']
    wanted = (('accuracy', 'P1', None, 'WER (weighted)'), ('accuracy', 'P1', None, 'CER (weighted)'),
              ('content', 'P2', PRIMARY_ARM, 'P2a: C of peer_other, peer = other against task'),
              ('content', 'P3', PRIMARY_ARM, 'P2a: delta = C_ref - C_ASR of peer_other (paired)'),
              ('who', 'P4', None, 'share member (weighted)'))
    for check, analysis, arm, metric in wanted:
        row = next((r for r in pooled.get(check, []) if r['version'] == args.audit_version and r['analysis'] == analysis
                    and r.get('arm') == arm and r['by'] == 'pooled' and r['metric'] == metric), None)
        if row:
            print(f'  {analysis} {metric}: {_fmt(row)}')
    print(f'written to {folder}')
    return 0
