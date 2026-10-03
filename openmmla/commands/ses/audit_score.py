"""The scores of a sensing audit (mmla ses-code --audit-score ID --audit-version reported|rerun): one
frozen pipeline version (audit.py's pipeline_<version>.json) against the primary auditor's answers.

What is scored, and only that: the answers saved through a claimed audit link of the audit's
campaign.yml, the record's auditor the link's (an answer without a link, or of another's link, is
counted and left out); of an item's identity answers the first (the page locks it; a later line is
counted as a lock broken). The request log must verify with the answers files both ways first
(code_locked.checked_log; --despite-log-failure scores anyway, named in the header), and each frozen
version's file must be the one its freeze logged (a file changed or frozen twice is refused). A version
frozen or a render made after scored answers existed is named in the header.

The primary auditor is --audit-auditor, else the one auditor whose audit link has no subset (two such
need --audit-auditor); the others feed the inter-auditor agreement only.

An open audit (a start line of its request log served it with --audit-open) is scored by typed name: an
answer saved by the open page counts under the name it was saved under, and an answer of a claimed audit
link as above. The primary auditor is --audit-auditor, else the one name whose answers chose the full
audit (two such need --audit-auditor); the agreement is the primary's with each name that chose the
reliability subset, on the items both answered; any other name is counted and left out. Every output's
header says the audit was open: the names were typed, not authenticated. Since one person may type two
names, the report lists the saves under other names from an address and browser of the primary's own
saves, and among them the identity answers of a frame saved before the primary's (shared_browser).

Practice items, items not rendered or whose frame failed its tag check, frames and clips the auditor
flagged, frames the version holds no frame of, pupils without a letter of the display roster, and
cannot-tell answers are left out of every score and counted. The answers' notes are never read into an
output.

Identity (a pipeline member box: the version's person at a displayed box carries a tag of the
version's roster):
  precision        correct / (correct + swapped + outside the group + not a person), with the error
                   mix; by where the tag came from (read in the frame, the server's track memory, carried
                   along the track by the fusion), for the untagged boxes standing at a missing pupil's
                   seat (the seat rule), and for the server's stored naming beside the fused one
  recall           of the members the auditor saw (a box given that pupil, or a member without a box),
                   found when the version's person at the auditor's box carries the pupil's tag; the
                   misses split into undetected (no box at all, or no person of the version at the
                   box), untagged and mistagged; and the missed members per frame
Gaze (a pipeline member box the auditor answered the gaze of):
  readability      the version's unknown against the auditor's cannot tell
  agreement        on the boxes both could read: accuracy, Cohen's kappa and the confusion matrix (rows the
                   auditor, columns the version) over the eight classes, and member-directed (a member's
                   face or hands) against not; between answers apart (agreeing when the version's class is
                   one of the two); the same on the boxes whose identity is correct; by face source, face
                   height and head brightness terciles (the plan's cuts)
Who speaks (a window of a recording with a group microphone):
  speech present   the auditor's speaker other than no one, against speech_ratio > 0, > 0.1 and > 0.3:
                   sensitivity, specificity, kappa
  not the group's  of the windows with measured speech, the share whose speaker is the teacher or an
                   adult or another group, unweighted and weighted by speech_ratio, and at speech_ratio
                   >= 0.3; where worn microphones ran, member-attributed words (at least half the words)
                   against the speaker

Every table comes per lesson first (with Wilson intervals, item-level and approximate), then by
split, camera count and group size, then pooled, each with a lesson-cluster bootstrap interval
(metrics.session_bootstrap over the lessons in the row) and design-weighted values beside the
unweighted ones. The outputs go to artifacts/runtime/audit/<id>/scores/<version>_<UTC time>/: summary.json,
report.txt, per_lesson_<check>.csv, by_<dimension>_<check>.csv, pooled_<check>.csv, confusion_<name>.csv
and interauditor.csv, each headed by the plan's sha256, the version, the drift check and the auditor.
"""
from __future__ import annotations

import csv
import json
import math
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from openmmla.commands.ses import audit as A
from openmmla.commands.ses import code as C
from openmmla.commands.ses import code_locked as L

L.register_loaded(__name__, __file__)

# the pipeline's gaze labels as the audit's classes: a zone and the work area are the task material
GAZE_OF_LABEL = {'partner_face': 'member_face', 'other_face': 'other_face', 'own_hands': 'own_hands',
                 'partner_hands': 'member_hands', 'other_hands': 'other_hands', 'work_area': 'task', 'zone': 'task',
                 'elsewhere': 'elsewhere', 'out_of_frame': 'out_of_frame', 'unknown': None}
GAZE_CLASSES = ('member_face', 'other_face', 'own_hands', 'member_hands', 'other_hands', 'task', 'elsewhere', 'out_of_frame')
MEMBER_DIRECTED = ('member_face', 'member_hands')
JUDGED = ('correct', 'swap', 'outside', 'not_person')
MISSES = ('undetected', 'untagged', 'mistagged')
SPEAKERS = ('member', 'adult', 'other_group', 'none')
THRESHOLDS = (0.0, 0.1, 0.3)
TALK = 0.3
DIMENSIONS = (('lesson', 'lesson'), ('split', 'split'), ('cameras', 'camera count'), ('group_size', 'group size'))


# ---- reading ----

def audit_links(campaign: dict) -> dict[str, dict]:
    """token id -> {'name', 'subset'} of every claimed audit link of campaign.yml"""
    return {t['token_id']: {'name': c['name'], 'subset': t.get('subset')} for c in campaign.get('coders', [])
            for t in c.get('tokens', []) if t.get('scope') == 'audit' and t.get('claimed_at')}


def open_serves(artifacts: Path, audit_id: str) -> list:
    """the seqs of the start lines in the audit's request log whose server served it open (--audit-open)"""
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
        if isinstance(record, dict) and record.get('event') == 'start' and record.get('kind') == 'audit' \
                and record.get('open') is True:
            out.append(record.get('seq'))
    return out


def shared_browser(artifacts: Path, audit_id: str, primary: str) -> dict:
    """the saves the request log names under another name than `primary` from an address and browser one of
    the primary's own saves came from: in an open audit one person may type two names, and an identity
    answer of a frame under a second name shows which boxes the frozen versions call pupils before the
    primary's own identity answer of it is locked. {'names': name -> saves, 'identity_before': [{'name',
    'alias', 'item', 'seq', 'primary_seq'}, ...] (an identity save of a frame before the primary's)}. The
    same address and browser are no proof of one person (a shared machine, an SSH forward's 127.0.0.1), and
    different ones no proof of two."""
    path = A.audit_dir(artifacts, audit_id) / L.LOG_FILE
    try:
        data = path.read_bytes()
    except OSError:
        data = b''
    saves = []
    for line in data.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get('label_sha256') and isinstance(record.get('coder'), str) \
                and isinstance(record.get('seq'), int) and isinstance(record.get('ip'), str):
            saves.append(record)
    mine = {(r.get('ip'), r.get('ua')) for r in saves if r['coder'] == primary}
    first: dict = {}
    for r in saves:
        if r['coder'] == primary and r.get('phase') == 'identity':
            first.setdefault(r.get('item'), r['seq'])
    names: Counter = Counter()
    before = []
    for r in saves:
        if r['coder'] == primary or (r.get('ip'), r.get('ua')) not in mine:
            continue
        names[r['coder']] += 1
        if r.get('phase') == 'identity' and r.get('item') in first and r['seq'] < first[r['item']]:
            before.append({'name': r['coder'], 'alias': r.get('alias'), 'item': r['item'], 'seq': r['seq'],
                           'primary_seq': first[r['item']]})
    return {'names': dict(sorted(names.items())), 'identity_before': before}


def read_answers(artifacts: Path, plan: dict, campaign: dict,
                 typed: bool = False) -> tuple[dict[str, dict[str, dict[str, dict]]], Counter, dict[str, set]]:
    """(auditor -> item -> phase -> the record that counts: an identity's first, any other phase's last; what
    was left out and why; auditor -> the subsets their counted answers were saved under): only records a
    claimed audit link of `campaign` (campaign.yml) saved for its own auditor count, and with `typed` (an
    open audit) also those the open page saved, under the name each was saved under"""
    links = audit_links(campaign)
    out: dict[str, dict[str, dict[str, dict]]] = defaultdict(dict)
    scopes: dict[str, set] = defaultdict(set)
    left = Counter()
    for _, path in A.record_files(artifacts, plan['audit_id'], [s['id'] for s in plan['sessions']]):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(line)
                auditor, item, phase = record['auditor'], record['item'], record['phase']
            except (ValueError, KeyError, TypeError):
                left['answer lines that do not parse'] += 1
                continue
            link = links.get(str(record.get('token_id')))
            if typed and link is None and record.get('open') is True and record.get('token_id') is None \
                    and isinstance(auditor, str):
                subset = record.get('subset') if record.get('subset') in L.SUBSETS else None
            elif link is None or link['name'] != auditor:
                left['answers saved without a claimed audit link of their auditor' if not typed else
                     'answers saved neither by the open page nor through a claimed audit link of their auditor'] += 1
                continue
            else:
                subset = link['subset']
            phases = out[auditor].setdefault(item, {})
            if phase == 'identity' and 'identity' in phases:
                left['identity answers saved after the lock (the first counts)'] += 1
                continue
            phases[phase] = record
            scopes[auditor].add(subset)
    return dict(out), left, dict(scopes)


def _scope_of(subsets: set) -> str:
    """what a name's answers chose: 'full', 'reliability', or 'mixed' when they were saved under both"""
    if len(subsets) > 1:
        return 'mixed'
    return 'reliability' if subsets == {'reliability'} else 'full'


def primary_typed(answers: dict, chosen: str | None, scopes: dict[str, set]) -> str | None:
    """an open audit's primary auditor: `chosen`, else the one name with answers that chose the full audit"""
    if chosen:
        # the open page keeps names in NFC: a name given in another form is the same name
        chosen = unicodedata.normalize('NFC', chosen)
        return chosen if chosen in answers else None
    full = sorted(name for name in answers if None in scopes.get(name, ()))
    if len(full) > 1:
        raise A.AuditError(f"{len(full)} names answered the full audit ({', '.join(full)}): name the primary one with "
                           '--audit-auditor')
    return full[0] if full else None


def primary_auditor(answers: dict, chosen: str | None, campaign: dict) -> str | None:
    """the auditor scored: `chosen`, else the one auditor with answers whose audit links carry no subset"""
    if chosen:
        return chosen if chosen in answers else None
    subsets = defaultdict(set)
    for link in audit_links(campaign).values():
        subsets[link['name']].add(link['subset'])
    whole = sorted(name for name in answers if None in subsets.get(name, ()))
    if len(whole) > 1:
        raise A.AuditError(f"{len(whole)} auditors answered on links without a subset ({', '.join(whole)}): "
                           'name the primary one with --audit-auditor')
    return whole[0] if whole else None


def frozen_checks(artifacts: Path, audit_id: str, plan: dict, version: str, versions: dict) -> tuple[list[str], dict]:
    """(why the frozen files of `version` cannot be scored, what the freezes recorded): each file must be the
    one the last freeze of it logged, frozen once"""
    logged = A.logged_files(artifacts, audit_id)
    problems, after = [], {'after_answers': {}, 'after_render': []}
    for entry in plan['sessions']:
        sid = entry['id']
        if sid not in versions or versions[sid].get('refused'):
            continue
        rel = A._rel(artifacts, A.version_file(artifacts, sid, audit_id, version))
        lines = logged.get(rel) or []
        if not lines:
            problems.append(f"{entry['alias']}: its {version} file was frozen by no logged step")
            continue
        if len(lines) > 1:
            problems.append(f"{entry['alias']}: its {version} file was frozen {len(lines)} times (seq "
                            f"{', '.join(str(seq) for seq, _, _ in lines)})")
            continue
        if L.file_sha256(A.version_file(artifacts, sid, audit_id, version)) != lines[-1][1]:
            problems.append(f"{entry['alias']}: its {version} file changed since its freeze at seq {lines[-1][0]}")
            continue
        if versions[sid].get('after_answers'):
            after['after_answers'][entry['alias']] = versions[sid]['after_answers']
        if versions[sid].get('after_render'):
            after['after_render'].append(entry['alias'])
    return problems, after


def drift(versions: dict) -> list[str]:
    """the sessions whose table or parameters changed since the version was frozen"""
    out = []
    for sid, data in versions.items():
        sources = data.get('sources') or {}
        for name in ('table', 'parameters'):
            path, recorded = sources.get(name), sources.get(f'{name}_sha256')
            if path and recorded and L.file_sha256(path) != recorded:
                out.append(f'{sid}: its {name} changed since the freeze')
    return out


# ---- the units ----

def _source(match) -> str:
    """where a person's tag came from: read in the frame (torso or box), the server's track memory, or the
    fusion carrying it along the track"""
    return {'track': 'track memory', 'propagated': 'propagated'}.get(match, 'read')


def _tercile(value, cuts) -> str | None:
    if value is None:
        return None
    return 'low' if value < cuts[0] else 'middle' if value < cuts[1] else 'high'


def build_units(plan: dict, designs: dict, versions: dict, render: dict, answers: dict, auditor: str) -> dict:
    """the scored units of every check, and the counts of what was left out"""
    units = defaultdict(list)
    left = Counter()
    cuts = plan.get('cuts') or {'face_px': list(A.FACE_PX_CUTS), 'luma': list(A.LUMA_CUTS)}
    mine = answers.get(auditor, {})
    for sid, design in designs.items():
        data = versions.get(sid)
        if data is None or data.get('refused'):
            left['sessions refused for this version'] += 1
            continue
        letters = design['letters']
        letter_of = {tag: letter for letter, tag in letters.items()}
        kept = set(data.get('kept') or [])
        base = {'lesson': design['lesson'], 'split': design['split'] or 'none', 'cameras': design['n_cameras'],
                'group_size': design['group_size'], 'session': sid, 'alias': design['alias']}
        rendered = (render.get('sessions') or {}).get(design['alias'], {})
        heads = rendered.get('items') or {}
        for item in design['vision']:
            if item['practice']:
                left['practice frames'] += 1
                continue
            if item['item'] in (rendered.get('errors') or {}) or (render and item['item'] not in heads):
                left['frames not rendered'] += 1
                continue
            if ((heads.get(item['item']) or {}).get('tag_check') or {}).get('verdict') == 'mismatch':
                left['frames whose tags were not where they were stored'] += 1
                continue
            identity = (mine.get(item['item']) or {}).get('identity')
            if identity is None:
                left['frames not answered'] += 1
                continue
            said = identity['answer']
            if said.get('flag'):
                left['frames flagged'] += 1
                continue
            found = data['vision'].get(item['item']) or {}
            if not found.get('frame'):
                # the version holds no frame here: nothing of it can be scored (not as misses either)
                left["frames without the version's frame"] += 1
                continue
            boxes = found.get('boxes') or {}
            u0 = dict(base, item=item['item'], camera=item['base'], weight=item['weight'] or 1.0)
            given = said.get('boxes') or {}
            # precision: the version's member boxes, of the pupils the auditor knows by a letter (the display
            # version's roster); a pupil only this version keeps has no reference pictures, so no answer can name them
            for number, values in boxes.items():
                answer = given.get(number)
                if values.get('member'):
                    if values['member'] not in letter_of:
                        left['identity: member boxes of a pupil without a letter'] += 1
                    else:
                        outcome = ('correct' if answer == letter_of[values['member']] else 'swap' if answer in letters
                                   else answer)
                        if outcome == 'cannot_tell':
                            left['identity: member boxes answered cannot tell'] += 1
                        units['identity'].append(dict(u0, box=number, outcome=outcome, source=_source(values.get('match'))))
                elif values.get('seat_of'):
                    if values['seat_of'] not in letter_of:
                        left['identity: seat boxes of a pupil without a letter'] += 1
                    else:
                        outcome = ('correct' if answer == letter_of[values['seat_of']] else 'swap' if answer in letters
                                   else answer)
                        units['seat'].append(dict(u0, box=number, outcome=outcome, source='seat'))
                stored = (values.get('stored') or {}).get('tag')
                if stored in kept and stored in letter_of:
                    outcome = ('correct' if answer == letter_of[stored] else 'swap' if answer in letters else answer)
                    units['stored'].append(dict(u0, box=number, outcome=outcome, source=_source((values.get('stored') or {}).get('match'))))
            if found.get('extra'):
                left['identity: version persons at no displayed box'] += len(found['extra'])
            # recall: the members the auditor saw
            seen, hit = 0, 0
            for letter in sorted(set(v for v in given.values() if v in letters)):
                tag = letters[letter]
                at = [n for n, v in given.items() if v == letter]
                persons = [boxes.get(n) for n in at]
                if any(p and p.get('tag') == tag for p in persons):
                    kind = 'found'
                elif any(p and p.get('tag') is not None for p in persons):
                    kind = 'mistagged'
                elif any(p for p in persons):
                    kind = 'untagged'
                else:
                    kind = 'undetected'
                units['recall'].append(dict(u0, member=letter, kind=kind))
                seen += 1
                hit += kind == 'found'
            missed = said.get('missed') or 0
            for _ in range(int(missed)):
                units['recall'].append(dict(u0, member=None, kind='undetected'))
            units['frames'].append(dict(u0, seen=seen + int(missed), found=hit, missed=seen + int(missed) - hit))
            # gaze
            gaze = (mine.get(item['item']) or {}).get('gaze')
            if gaze is None:
                member_boxes = [n for n, v in boxes.items() if v.get('member')]
                left['gaze: member boxes without a gaze answer'] += len(member_boxes)
                continue
            if gaze['answer'].get('flag'):
                left['gaze: frames flagged'] += 1
                continue
            answered = gaze['answer'].get('boxes') or {}
            looks = (heads.get(item['item']) or {}).get('boxes') or {}
            for number, values in boxes.items():
                if not values.get('member') or values['member'] not in letter_of:
                    continue
                if number not in answered:
                    left['gaze: member boxes not asked'] += 1
                    continue
                said_class = answered[number]
                v = values.get('gaze') or {}
                units['gaze'].append(dict(
                    u0, box=number, truth=said_class['class'], between=said_class.get('between'),
                    pipeline=GAZE_OF_LABEL.get(v.get('label'), v.get('label')), label=v.get('label'),
                    identity_correct=given.get(number) == letter_of.get(values['member']),
                    face_source=v.get('face_source') or 'none', face_height=_tercile(v.get('face_h'), cuts['face_px']),
                    brightness=_tercile((looks.get(number) or {}).get('luma'), cuts['luma'])))
            for number, value in given.items():
                if value in letters:
                    values = boxes.get(number) or {}
                    measured = values.get('member') == letters[value]
                    units['members'].append(dict(u0, box=number, measured=measured,
                                                 readable=measured and (values.get('gaze') or {}).get('label') != 'unknown'))
        for item in design['speech']:
            if item['practice']:
                left['practice windows'] += 1
                continue
            if item['item'] in (rendered.get('errors') or {}):
                left['windows not rendered'] += 1
                continue
            record = (mine.get(item['item']) or {}).get('speech')
            if record is None:
                left['windows not answered'] += 1
                continue
            said = record['answer']
            if said.get('flag'):
                left['windows flagged'] += 1
                continue
            if said['speaker'] == 'cannot_tell':
                left['windows answered cannot tell'] += 1
                continue
            row = data['speech'].get(item['item']) or {}
            if not row.get('row') or row.get('speech_ratio') is None:
                left["windows without the version's measures"] += 1
                continue
            words, member = row.get('words'), row.get('member_words')
            units['speech'].append(dict(base, item=item['item'], weight=item['weight'] or 1.0, speaker=said['speaker'],
                                        ratio=row['speech_ratio'], worn=bool(row.get('worn')) and member is not None,
                                        attributed=bool(words) and member is not None and member >= 0.5 * words))
    return {'units': dict(units), 'left': dict(left)}


# ---- metrics on additive statistics, so a lesson bootstrap sums lessons ----

class Ratio:
    """a share (or a mean): sum of w * hit(u) over sum of w, over the units `keep` keeps; w is 1, the
    design weight (`weighted`), or what `weight` gives"""

    def __init__(self, name: str, keep: Callable, hit: Callable, weighted: bool = False, weight: Callable | None = None):
        self.name = name + (' (weighted)' if weighted else '')
        self.keep, self.hit, self.weighted, self.weight = keep, hit, weighted or weight is not None, weight

    def _w(self, u) -> float:
        if self.weight is not None:
            return float(self.weight(u))
        return float(u.get('weight') or 1.0) if self.weighted else 1.0

    def stats(self, units):
        import numpy as np
        num = den = 0.0
        for u in units:
            if self.keep(u):
                w = self._w(u)
                num += w * float(self.hit(u))
                den += w
        return np.array([num, den])

    @staticmethod
    def final(stats) -> float:
        return float(stats[0] / stats[1]) if stats[1] > 0 else float('nan')

    def count(self, units) -> int:
        return sum(1 for u in units if self.keep(u))


class Kappa:
    """Cohen's kappa of truth(u) against pred(u) over the units `keep` keeps"""

    def __init__(self, name: str, keep: Callable, truth: Callable, pred: Callable, categories):
        self.name, self.keep, self.truth, self.pred, self.categories = name, keep, truth, pred, list(categories)
        self.weighted = False

    def matrix(self, units):
        at = {c: i for i, c in enumerate(self.categories)}
        m = [[0] * len(at) for _ in at]
        for u in units:
            if self.keep(u):
                m[at[self.truth(u)]][at[self.pred(u)]] += 1
        return m

    def stats(self, units):
        import numpy as np
        return np.array(self.matrix(units), dtype=float).ravel()

    def final(self, stats) -> float:
        k = len(self.categories)
        value = C.kappa([[int(stats[i * k + j]) for j in range(k)] for i in range(k)])
        return float('nan') if value is None else float(value)

    def count(self, units) -> int:
        return sum(1 for u in units if self.keep(u))


def _readable(u) -> bool:
    return u['pipeline'] is not None and u['truth'] not in ('cannot_tell', 'between')


def _member(c) -> str:
    return 'member' if c in MEMBER_DIRECTED else 'not'


METRICS: dict[str, list] = {
    'identity': [
        Ratio('precision', lambda u: u['outcome'] in JUDGED, lambda u: u['outcome'] == 'correct'),
        Ratio('precision', lambda u: u['outcome'] in JUDGED, lambda u: u['outcome'] == 'correct', weighted=True),
        Ratio('swapped share', lambda u: u['outcome'] in JUDGED, lambda u: u['outcome'] == 'swap'),
        Ratio('outside share', lambda u: u['outcome'] in JUDGED, lambda u: u['outcome'] == 'outside'),
        Ratio('not a person share', lambda u: u['outcome'] in JUDGED, lambda u: u['outcome'] == 'not_person'),
    ],
    'recall': [
        Ratio('recall', lambda u: True, lambda u: u['kind'] == 'found'),
        Ratio('recall', lambda u: True, lambda u: u['kind'] == 'found', weighted=True),
    ] + [Ratio(f'{kind} share', lambda u: True, (lambda kind: lambda u: u['kind'] == kind)(kind)) for kind in MISSES],
    'frames': [
        Ratio('missed members per frame', lambda u: True, lambda u: u['missed']),
        Ratio('missed members per frame', lambda u: True, lambda u: u['missed'], weighted=True),
    ],
    'gaze': [
        Ratio('version readable', lambda u: True, lambda u: u['pipeline'] is not None),
        Ratio('auditor readable', lambda u: True, lambda u: u['truth'] != 'cannot_tell'),
        Ratio('accuracy, 8 classes', _readable, lambda u: u['truth'] == u['pipeline']),
        Ratio('accuracy, 8 classes', _readable, lambda u: u['truth'] == u['pipeline'], weighted=True),
        Kappa('kappa, 8 classes', _readable, lambda u: u['truth'], lambda u: u['pipeline'], GAZE_CLASSES),
        Ratio('accuracy, member-directed', _readable, lambda u: _member(u['truth']) == _member(u['pipeline'])),
        Kappa('kappa, member-directed', _readable, lambda u: _member(u['truth']), lambda u: _member(u['pipeline']),
              ('member', 'not')),
        Ratio('between answers agreeing', lambda u: u['truth'] == 'between' and u['pipeline'] is not None,
              lambda u: u['pipeline'] in (u['between'] or ())),
        Ratio('accuracy, 8 classes, identity correct', lambda u: _readable(u) and u['identity_correct'],
              lambda u: u['truth'] == u['pipeline']),
        Kappa('kappa, 8 classes, identity correct', lambda u: _readable(u) and u['identity_correct'],
              lambda u: u['truth'], lambda u: u['pipeline'], GAZE_CLASSES),
        Ratio('accuracy, member-directed, identity correct', lambda u: _readable(u) and u['identity_correct'],
              lambda u: _member(u['truth']) == _member(u['pipeline'])),
    ],
    'members': [
        Ratio('placed members the version measured', lambda u: True, lambda u: u['measured']),
        Ratio('placed members the version read', lambda u: True, lambda u: u['readable']),
    ],
    'speech': [m for threshold in THRESHOLDS for m in (
        Ratio(f'sensitivity at speech_ratio > {threshold:g}', lambda u: u['speaker'] != 'none',
              (lambda t: lambda u: u['ratio'] > t)(threshold)),
        Ratio(f'specificity at speech_ratio > {threshold:g}', lambda u: u['speaker'] == 'none',
              (lambda t: lambda u: u['ratio'] <= t)(threshold)),
        Kappa(f'kappa at speech_ratio > {threshold:g}', lambda u: True, lambda u: u['speaker'] != 'none',
              (lambda t: lambda u: u['ratio'] > t)(threshold), (False, True)))] + [
        Ratio("not the group's, of measured speech", lambda u: u['ratio'] > 0, lambda u: u['speaker'] in ('adult', 'other_group')),
        Ratio("not the group's, of measured speech", lambda u: u['ratio'] > 0, lambda u: u['speaker'] in ('adult', 'other_group'),
              weighted=True),
        # the measured group speech time that is not the group's
        Ratio("not the group's, weighted by speech_ratio", lambda u: u['ratio'] > 0,
              lambda u: u['speaker'] in ('adult', 'other_group'), weight=lambda u: u['ratio']),
        Ratio('no one speaking, of measured speech', lambda u: u['ratio'] > 0, lambda u: u['speaker'] == 'none'),
        Ratio(f"not the group's at speech_ratio >= {TALK:g}", lambda u: u['ratio'] >= TALK,
              lambda u: u['speaker'] in ('adult', 'other_group')),
    ],
}
METRICS['seat'] = [METRICS['identity'][0], METRICS['identity'][2], METRICS['identity'][3]]
METRICS['stored'] = [METRICS['identity'][0], METRICS['identity'][1]]
# the extra breakdowns of a check besides lesson, split, camera count and group size
BREAKDOWNS = {'identity': (('source', 'tag source'),), 'stored': (('source', 'tag source'),),
              'gaze': (('face_source', 'face source'), ('face_height', 'face height'), ('brightness', 'head brightness'))}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if not n:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def _value(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def evaluate(metric, units: list[dict], boot: int, seed: int, interval: str) -> dict:
    """a metric over units: its value, n, and an interval: 'wilson' (item-level, for one lesson's share) or
    'bootstrap' (resampling the units' lessons, metrics.session_bootstrap)"""
    import numpy as np
    stats = metric.stats(units)
    out = {'metric': metric.name, 'value': _value(metric.final(stats)), 'n': metric.count(units), 'lo': None, 'hi': None,
           'interval': None}
    if out['value'] is None or not out['n']:
        return out
    if interval == 'wilson' and isinstance(metric, Ratio) and not metric.weighted and 0 <= out['value'] <= 1 \
            and all(float(metric.hit(u)) in (0.0, 1.0) for u in units if metric.keep(u)):
        out['lo'], out['hi'] = wilson(out['value'] * out['n'], out['n'])
        out['interval'] = 'Wilson 95 %, item-level, approximate'
    elif interval == 'bootstrap' and boot > 0:
        from openmmla.analytics.interaction.metrics import session_bootstrap
        lessons = sorted({u['lesson'] for u in units})
        table = np.array([metric.stats([u for u in units if u['lesson'] == lesson]) for lesson in lessons])
        result = session_bootstrap(lambda rows: metric.final(table[rows].sum(axis=0)), np.array(lessons, dtype=object),
                                   n=boot, seed=seed)
        out.update(lo=_value(result['lo']), hi=_value(result['hi']), lessons=len(lessons),
                   interval=f"lesson bootstrap 95 % ({len(lessons)} lessons, {result['n_undefined']} of {result['n']} undefined)")
    return out


def check_tables(check: str, units: list[dict], boot: int, seed: int) -> list[dict]:
    """the rows of a check: per lesson first, then by split, camera count, group size (and the check's own
    breakdowns), then pooled"""
    rows = []
    dimensions = list(DIMENSIONS) + list(BREAKDOWNS.get(check, ()))
    for key, title in dimensions:
        groups = sorted({u.get(key) for u in units}, key=lambda v: (v is None, str(v)))
        for group in groups:
            chosen = [u for u in units if u.get(key) == group]
            for metric in METRICS[check]:
                rows.append({'check': check, 'by': title, 'group': 'none' if group is None else str(group),
                             **evaluate(metric, chosen, boot, seed, 'wilson' if key == 'lesson' else 'bootstrap')})
    for metric in METRICS[check]:
        rows.append({'check': check, 'by': 'pooled', 'group': 'all', **evaluate(metric, units, boot, seed, 'bootstrap')})
    return rows


def confusions(units: dict) -> dict[str, dict]:
    gaze = [u for u in units.get('gaze', []) if _readable(u)]
    full = Kappa('', lambda u: True, lambda u: u['truth'], lambda u: u['pipeline'], GAZE_CLASSES)
    member = Kappa('', lambda u: True, lambda u: _member(u['truth']), lambda u: _member(u['pipeline']), ('member', 'not'))
    readability = [[0, 0], [0, 0]]
    for u in units.get('gaze', []):  # a gaze between two targets was read
        readability[u['truth'] == 'cannot_tell'][u['pipeline'] is None] += 1
    worn = [u for u in units.get('speech', []) if u['worn']]
    attributed = [[sum(1 for u in worn if u['attributed'] == a and u['speaker'] == s) for s in SPEAKERS] for a in (True, False)]
    return {'gaze_8_classes': {'rows': list(GAZE_CLASSES), 'columns': list(GAZE_CLASSES), 'matrix': full.matrix(gaze)},
            'gaze_member_directed': {'rows': ['member', 'not'], 'columns': ['member', 'not'], 'matrix': member.matrix(gaze)},
            'gaze_readability': {'rows': ['auditor read it', 'auditor cannot tell'],
                                 'columns': ['version read it', 'version unknown'], 'matrix': readability},
            'speech_worn_attribution': {'rows': ['member-attributed', 'unattributed'], 'columns': list(SPEAKERS),
                                        'matrix': attributed}}


def interauditor(plan: dict, answers: dict, primary: str, partners=None) -> list[dict]:
    """per other auditor (of `partners`, default every one) and question, on the items both answered
    (practice and flagged ones left out): the pairs, the share agreeing and Cohen's kappa where it is defined"""
    rows = []
    mine = answers.get(primary, {})
    for other in sorted(set(answers if partners is None else partners) - {primary}):
        theirs = answers[other]
        pairs: dict[str, list] = defaultdict(list)
        for item, phases in mine.items():
            if item not in theirs:
                continue
            for phase, record in phases.items():
                twin = theirs[item].get(phase)
                if twin is None or record.get('practice'):
                    continue
                a, b = record['answer'], twin['answer']
                if a.get('flag') or b.get('flag'):
                    continue
                if phase == 'roster':
                    pairs['roster: wears the badge'].append((a['wears'], b['wears']))
                elif phase == 'speech':
                    pairs['who speaks'].append((a['speaker'], b['speaker']))
                elif phase == 'identity':
                    for n in sorted(set(a.get('boxes') or {}) & set(b.get('boxes') or {}), key=int):
                        pairs['identity: who is in a box'].append((a['boxes'][n], b['boxes'][n]))
                    if a.get('missed') is not None and b.get('missed') is not None:
                        pairs['identity: members without a box'].append((a['missed'], b['missed']))
                else:
                    for n in sorted(set(a.get('boxes') or {}) & set(b.get('boxes') or {}), key=int):
                        pairs['gaze: where'].append((a['boxes'][n]['class'], b['boxes'][n]['class']))
        for question, values in sorted(pairs.items()):
            categories = sorted({str(v) for pair in values for v in pair})
            at = {c: i for i, c in enumerate(categories)}
            m = [[0] * len(at) for _ in at]
            for x, y in values:
                m[at[str(x)]][at[str(y)]] += 1
            k = C.kappa(m) if len(categories) > 1 else None
            rows.append({'auditor': other, 'question': question, 'n': len(values),
                         'agree': round(sum(1 for x, y in values if x == y) / len(values), 4),
                         'kappa': None if k is None else round(k, 4)})
    return rows


# ---- writing ----

def _header(context: dict) -> str:
    after = context.get('after') or {}
    late = ''
    if after.get('after_answers'):
        late += '; frozen after scored answers: ' + ', '.join(f'{a} ({n})' for a, n in sorted(after['after_answers'].items()))
    if after.get('after_render'):
        late += '; frozen after the render: ' + ', '.join(sorted(after['after_render']))
    if context.get('rendered_after_answers'):
        late += f"; rendered after {context['rendered_after_answers']} scored answers"
    log = context.get('log') or {}
    checked = ('DOES NOT VERIFY, scored anyway' if log.get('despite_failure') else
               f"{len(log.get('cuts') or [])} crash cut(s)" if log.get('cuts') else 'verified')
    opened = '; the audit was open (names typed, not authenticated)' if context.get('open') else ''
    shared = (context.get('open') or {}).get('shared_browser') or {}
    if shared.get('names'):
        opened += (f"; {sum(shared['names'].values())} saves under other names from the primary's address and browser"
                   f" ({len(shared['identity_before'])} identity answers of a frame before the primary's)")
    return (f"# audit {context['audit_id']}; plan sha256 {context['plan_sha256']}; version {context['version']}; "
            f"drift check {context['drift']}; primary auditor {context['auditor']}{opened}; request log {checked}, head "
            f"{log.get('head')}{late}")


def _csv(path: Path, context: dict, rows: list[dict], columns: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        handle.write(_header(context) + '\n')
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if row.get(k) is None else row.get(k) for k in columns})


def _fmt(row: dict) -> str:
    if row['value'] is None:
        return f"{'-':>8} (n {row['n']})"
    interval = f" [{row['lo']}, {row['hi']}]" if row['lo'] is not None else ''
    return f"{row['value']:>8.3f} (n {row['n']}){interval}"


def report_text(context: dict, tables: dict, confusion: dict, agreement: list, left: dict) -> str:
    lines = [_header(context), f"declared analyses ({len(context['declared'])}):"]
    lines += [f'  - {d}' for d in context['declared']]
    titles = {'identity': 'IDENTITY: precision of the member boxes', 'stored': "IDENTITY: the server's stored naming",
              'seat': "IDENTITY: untagged boxes at a missing pupil's seat", 'recall': 'IDENTITY: recall of the members seen',
              'frames': 'IDENTITY: missed members per frame', 'gaze': 'GAZE', 'members': 'GAZE: members the auditor placed',
              'speech': 'WHO SPEAKS'}
    for check, rows in tables.items():
        lines += ['', titles.get(check, check.upper())]
        for by in [b for b in dict.fromkeys(r['by'] for r in rows)]:
            lines.append(f'  by {by}' if by != 'pooled' else '  pooled')
            for row in (r for r in rows if r['by'] == by):
                lines.append(f"    {row['group']:<34} {row['metric']:<48} {_fmt(row)}")
    for name, data in confusion.items():
        lines += ['', f'CONFUSION {name} (rows {", ".join(data["rows"])}; columns {", ".join(data["columns"])})']
        lines += ['    ' + ' '.join(f'{v:>5}' for v in row) for row in data['matrix']]
    if context.get('open'):
        lines += ['', 'OPEN AUDIT: THE NAMES TYPED (not authenticated)']
        lines += [f"    {name:<24} {row['scope']:<12} {row['items']:>5} items  {row['role']}"
                  for name, row in context['open']['names'].items()]
        shared = context['open'].get('shared_browser') or {}
        lines += ['', "OPEN AUDIT: SAVES UNDER OTHER NAMES FROM THE PRIMARY'S ADDRESS AND BROWSER (one person may type "
                      'two names; a shared machine looks alike)']
        lines += [f'    {name:<24} {count:>5} saves' for name, count in (shared.get('names') or {}).items()] or ['    none']
        if shared.get('identity_before'):
            lines += ['    identity answers of a frame under another name before the primary\'s own (the boxes the '
                      'versions call pupils were then shown):']
            lines += [f"      {r['name']:<22} {r['alias']} {r['item']} at seq {r['seq']}, the primary's at seq "
                      f"{r['primary_seq']}" for r in shared['identity_before']]
    lines += ['', 'INTER-AUDITOR AGREEMENT']
    lines += [f"    {r['auditor']:<16} {r['question']:<36} n {r['n']:<5} agree {r['agree']} kappa {r['kappa']}"
              for r in agreement] or ['    no second auditor answered the same items']
    lines += ['', 'LEFT OUT'] + [f'    {k}: {v}' for k, v in sorted(left.items())]
    return '\n'.join(lines) + '\n'


def score(artifacts: Path, audit_id: str, version: str, auditor: str | None = None, boot: int = 2000, seed: int = 0,
          allow_drift: bool = False, out: Path | None = None, despite_log_failure: bool = False) -> tuple[Path, dict]:
    plan = A.load_plan(artifacts, audit_id)
    campaign = L.Campaign(A.audit_dir(artifacts, audit_id))
    if not campaign.exists():
        raise A.AuditError(f'no campaign.yml in {campaign.folder}: the answers are scored by their links')
    log = L.checked_log(campaign, artifacts, despite_log_failure, 'score')
    designs, versions = {}, {}
    for entry in plan['sessions']:
        folder = A.session_audit_dir(artifacts, entry['id'], audit_id)
        designs[entry['id']] = A.read_json(folder / A.DESIGN_FILE)
        path = A.version_file(artifacts, entry['id'], audit_id, version)
        if path.exists():
            versions[entry['id']] = A.read_json(path)
    if not versions:
        raise A.AuditError(f'no session of audit {audit_id} has {version} frozen (--audit-freeze {audit_id} '
                           f'--audit-version {version})')
    problems, after = frozen_checks(artifacts, audit_id, plan, version, versions)
    if problems:
        raise A.AuditError('; '.join(problems) + ': the frozen outputs are not the ones logged, nothing is scored')
    drifted = drift({sid: d for sid, d in versions.items() if not d.get('refused')})
    if drifted and not allow_drift:
        raise A.AuditError('; '.join(drifted) + ' (--audit-allow-drift scores anyway, recorded)')
    index_path = A.audit_dir(artifacts, audit_id) / A.RENDER_INDEX
    render = A.read_json(index_path) if index_path.exists() else {}
    data = campaign.data()
    # served open at least once: the answers are scored by the names typed
    opened = open_serves(artifacts, audit_id)
    answers, refused, scopes = read_answers(artifacts, plan, data, typed=bool(opened))
    partners = None
    if opened:
        primary = primary_typed(answers, auditor, scopes)
        if primary is None:
            raise A.AuditError(f'no answers by {auditor}' if auditor else
                               'no answers yet under a name that chose the full audit (or give --audit-auditor)')
        partners = sorted(n for n in answers if n != primary and scopes.get(n) == {'reliability'})
    else:
        primary = primary_auditor(answers, auditor, data)
        if primary is None:
            raise A.AuditError(f'no answers by {auditor} through an audit link' if auditor else
                               'no answers yet through an audit link without a subset (or give --audit-auditor)')
    built = build_units(plan, designs, versions, render, answers, primary)
    units, left = built['units'], built['left']
    left.update(refused)
    if opened:
        others = [n for n in answers if n != primary and n not in partners]
        if others:
            left['open audit: names other than the primary and the reliability subset, not scored'] = len(others)
    tables = {check: check_tables(check, units.get(check, []), boot, seed)
              for check in ('identity', 'stored', 'seat', 'recall', 'frames', 'gaze', 'members', 'speech')}
    confusion = confusions(units)
    agreement = interauditor(plan, answers, primary, partners)
    context = {'audit_id': audit_id, 'plan_sha256': L.file_sha256(A.audit_dir(artifacts, audit_id) / A.PLAN_FILE),
               'version': version, 'drift': ('; '.join(drifted) + ' (allowed)') if drifted else 'none',
               'auditor': primary, 'declared': plan.get('declared') or [], 'boot': boot, 'seed': seed,
               'mode': plan.get('mode'), 'scored_at': C.now_utc(), 'after': after,
               'rendered_after_answers': render.get('after_answers') or 0,
               'log': {k: log[k] for k in ('ok', 'message', 'head', 'seq', 'cuts', 'despite_failure')}}
    if opened:
        # names typed, not authenticated: each name with the scope its answers chose and how many items it answered
        context['open'] = {'served_open_at_seq': opened, 'names': {
            name: {'scope': _scope_of(scopes.get(name, set())), 'items': len(answers[name]),
                   'role': 'primary' if name == primary else 'reliability' if name in partners else 'left out'}
            for name in sorted(answers)}, 'shared_browser': shared_browser(artifacts, audit_id, primary)}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = Path(out) if out else A.audit_dir(artifacts, audit_id) / A.SCORES_DIR / f'{version}_{stamp}'
    folder.mkdir(parents=True, exist_ok=True)
    columns = ['check', 'by', 'group', 'metric', 'value', 'n', 'lo', 'hi', 'interval']
    for check, rows in tables.items():
        _csv(folder / f'per_lesson_{check}.csv', context, [r for r in rows if r['by'] == 'lesson'], columns)
        for key, title in list(DIMENSIONS[1:]) + list(BREAKDOWNS.get(check, ())):
            _csv(folder / f'by_{key}_{check}.csv', context, [r for r in rows if r['by'] == title], columns)
        _csv(folder / f'pooled_{check}.csv', context, [r for r in rows if r['by'] == 'pooled'], columns)
    for name, data in confusion.items():
        with (folder / f'confusion_{name}.csv').open('w', newline='', encoding='utf-8') as handle:
            handle.write(_header(context) + '\n')
            writer = csv.writer(handle)
            writer.writerow(['truth \\ version'] + data['columns'])
            for label, row in zip(data['rows'], data['matrix']):
                writer.writerow([label] + row)
    _csv(folder / 'interauditor.csv', context, agreement, ['auditor', 'question', 'n', 'agree', 'kappa'])
    summary = {**context, 'left_out': left, 'confusion': confusion, 'interauditor': agreement,
               'tables': {check: rows for check, rows in tables.items()}}
    A.write_json(folder / 'summary.json', summary)
    (folder / 'report.txt').write_text(report_text(context, tables, confusion, agreement, left), encoding='utf-8')
    return folder, summary


def cmd_score(args, argv) -> int:
    artifacts, audit_id = A._artifacts(args), args.audit_score
    if not args.audit_version:
        raise A.AuditError('give --audit-version reported|rerun: the version to score')
    folder, summary = score(artifacts, audit_id, args.audit_version, args.audit_auditor, A._size(args, 'boot'),
                            A._size(args, 'boot_seed'), bool(args.audit_allow_drift),
                            Path(args.audit_out) if args.audit_out else None, bool(args.despite_log_failure))
    A._event(artifacts, audit_id, 'audit-score', version=args.audit_version, auditor=summary['auditor'],
             drift=summary['drift'], log_check=summary['log'], summary_sha256=L.file_sha256(folder / 'summary.json'),
             argv=list(argv), **({'open': True} if summary.get('open') else {}))
    pooled = {(r['check'], r['metric']): r for rows in summary['tables'].values() for r in rows if r['by'] == 'pooled'}
    print(f"audit {audit_id}, {args.audit_version}, primary auditor {summary['auditor']}"
          + (', the audit was open (names typed, not authenticated)' if summary.get('open') else '')
          + (f", drift: {summary['drift']}" if summary['drift'] != 'none' else '')
          + (f", the request log does not verify ({summary['log']['message']})" if summary['log']['despite_failure'] else ''))
    shared = (summary.get('open') or {}).get('shared_browser') or {}
    if shared.get('names'):
        print("  saves under other names from the primary's address and browser: "
              + ', '.join(f'{name} {count}' for name, count in shared['names'].items())
              + (f"; {len(shared['identity_before'])} identity answers of a frame before the primary's"
                 if shared['identity_before'] else '') + ' (report.txt lists them)')
    for key in (('identity', 'precision'), ('recall', 'recall'), ('gaze', 'accuracy, 8 classes'),
                ('gaze', 'kappa, 8 classes'), ('gaze', 'version readable'), ('speech', 'sensitivity at speech_ratio > 0'),
                ('speech', "not the group's, of measured speech")):
        row = pooled.get(key)
        if row:
            print(f'  {key[0]}: {key[1]}: {_fmt(row)}')
    print(f'written to {folder}')
    return 0
