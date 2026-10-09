"""The page of a transcription audit (mmla ses-code --audit ID, a plan whose task is 'transcript'; audit_speech
draws and freezes it): a listener who knows the language writes down what is said in each sampled 10 s window
and answers the content model's questions about it, blind to every system output, and only once an operator
closes their blind pass rates the text the content model read.

The server is audit_page's (its guard and request log, links of scope audit, or with --audit-open the names
typed with the full audit or the reliability subset, and the same rules of names and scopes); this module gives
its routes for a transcription audit's items (TranscriptHandler) and its page (TRANSCRIPT_PAGE, OPEN_TRANSCRIPT_PAGE
open, which starts and changes its name as the sensing audit's open page does: audit_page.OPEN_FLOW). The sensing
audit's pages are audit_page's alone. What the server reads: the plan, each session's view,
the answers files and blind_closed.json; never a pipeline*.json. The only system output it relays is an item's
reveal, the reveal version's text of the window and of the 10 s before it, which audit_render copied into the
view. It is sent in the item's answer only, only to a name whose blind pass the operator closed
(--audit-close-blind ID --audit-auditor NAME: the link's name, or the name as typed, in NFC), only for an item not
practice whose transcription that name saved before the close, and only to a name that answers the full audit
(the reliability subset is never shown it). Until then no response carries any system output: no text, word
count, score, stratum, rank, weight or session id.

Served beside another audit at one address (audit_page: --audit ID --audit-with THIS), every route is under a path
prefix (/t/audit, /t/api/audit/..., /t/audit/clip/...): the handler's `base_path`, which the server puts before
every address it sends and writes into the page (based()), so the page sends its requests there and its browser
keeps the name and scope under keys of their own. Served alone, the base path is '' and the pages are
TRANSCRIPT_PAGE and OPEN_TRANSCRIPT_PAGE as they are. Served beside another audit, the page also links back to the
entry from its header (homed), and leaving through that link asks first when the item shown has changes not saved,
and does nothing while a save is sent or an item loads, as a move to another item does; open, the page hands the
name it audits under to the entry, which fills it in.

The sequence: the practice block (the practice windows of each recording, in the plan's order), then sweep by
sweep each recording's windows of that sweep in its view's order, the recordings in the plan's order (reversed
for the reliability subset, which gets only the flagged windows). Only the sweeps up to --audit-sweeps (default
the plan's, 1) are served: an item of a later sweep is not in the sequence, not counted and not sent (404), and
neither is its clip, so a later sweep can be served without drawing, freezing or rendering after any answer.
Each item has one clip of the camera grid with the table microphone (/audit/clip/<item>) and, in a session whose
personal microphones fed the text, one with their sum (?channel=worn); the view's span says where in the clip
the window lies.

Answers are lines of artifacts/<session>/audit/<id>/answers/<auditor>.jsonl (readable by its owner only, since
it holds transcripts) as the sensing audit's (the item,
the phase, the session, the auditor and link, or open and the scope, the codebook version, practice,
reliability, the sweep, the seconds spent, the page's and the server's times and the request's seq); the last
line of an item's phase counts:

  transcribe  {"status": "speech" | "unintelligible" | "none", "transcript": "M: ...\\nT: ...", "overlap": bool,
               "who": "none" | "member" | "adult" | "other_group" | "cannot_tell",
               "adult": "yes" | "no" | "cannot_tell", "peer": "task" | "other" | "none" | "cannot_tell",
               "other_offtask": "yes" | "no" | "cannot_tell", "flag": bool, "note": "",
               "listening": {"seen": [[from, to], ...], "played_seconds": s, "slowest_rate": r | null,
                             "highest_gain_db": g, "channels": ["main", "worn"], "target_cover": c}}
  reveal      {"gist": "yes" | "partly" | "no" | "nothing_said",
               "invented": "none" | "some" | "most" | "cannot_tell", "flag": bool, "note": ""}

The transcript is checked by audit_text.parse_reference, the grammar the scorer reads it with, and kept as it
cleans it (NFC, each line trimmed and its runs of spaces one); its status agrees with it (none: empty;
unintelligible: only [x] and [bg]; speech: a word at least), status none goes with who none and the other way
round, and no one speaking with no adult, no pupil talk, no other group's and no overlap; an adult speaking most
(who adult) goes with an adult question that is not no. Transcriptions may be revised until
the name's blind pass is closed; from then the server refuses them (409, judged under the request log's lock,
which the close holds too, so a save is wholly before or after it). A reveal rating is taken only where the
reveal is sent; it may be changed.

The listening record (the parts of the clip played, the seconds played, the slowest speed, the highest gain and
the microphones) is what the browser reports: the server refuses a transcription whose record covers less than
TARGET_COVER of the shaded window unless it is flagged, which is a convenience for the auditor, not a control.
The page keeps no draft: an unsaved transcription lives in the page only, and leaving it asks first.
"""
from __future__ import annotations

import json
import math
import os
import urllib.parse
from pathlib import Path
from typing import Any

from openmmla.commands.ses import audit as A
from openmmla.commands.ses import audit_page as P
from openmmla.commands.ses import audit_speech as S
from openmmla.commands.ses import audit_text as X
from openmmla.commands.ses import code as C
from openmmla.commands.ses import code_locked as L

L.register_loaded(__name__, __file__)

TRANSCRIPT_PHASES = ('transcribe', 'reveal')
# the version after the sensing audit's codebook, so that a line says which codebook it answered
TRANSCRIPT_CODEBOOK_VERSION = A.CODEBOOK_VERSION + 1
YES_NO = ('yes', 'no', 'cannot_tell')
PEER = ('task', 'other', 'none', 'cannot_tell')
GIST = ('yes', 'partly', 'no', 'nothing_said')
INVENTED = ('none', 'some', 'most', 'cannot_tell')
# the questions after the transcript, each with its values (who speaks is the sensing audit's question)
QUESTIONS = {'who': P.SPEAKERS, 'adult': YES_NO, 'peer': PEER, 'other_offtask': YES_NO}
# what a window's answers are when no one speaks in it
NOBODY = {'who': 'none', 'adult': 'no', 'peer': 'none', 'other_offtask': 'no'}
# the clips of an item: the table microphone's, and the personal microphones' sum in a dual session
CHANNELS = ('main', 'worn')
# the share of the shaded window the listening record must cover before an unflagged transcription is saved
TARGET_COVER = L.MIN_COVER
MAX_SEEN = 200
# a played part may end this far past the clip's span (an encoded clip runs a frame or two long)
SEEN_SLACK = 0.5
SPEEDS = (1.0, 0.75, 0.5)
GAINS_DB = (0, 6, 12, 18)
MIN_RATE, MAX_RATE = 0.25, 4.0
# a transcript is refused unread past this many characters, twice what it may hold once cleaned
MAX_RAW = 2 * X.MAX_TEXT
# the guide's rules, its English twin (docs/analytics/transcription_guide.md gives them in Danish too)
GUIDE = [
    'Write every word that begins inside the shaded 10 seconds. A word already under way when the shading starts '
    'is left out; a word that begins before the shading ends is written in full.',
    'Standard Danish spelling (Retskrivningsordbogen), also for reduced pronunciations: hvad, ikke, jeg, det. '
    'English words in English spelling. Numbers as words, exactly as said (digits are refused). Names as heard, in '
    'their usual spelling.',
    'Natural sentence punctuation (. , ? !) and capitals for sentence starts and names; nothing else.',
    'One line per turn, a new line at each change of speaker, each line starting with its tag. When people speak '
    'at once, each gets their own line, in the order they start, and you tick overlap. Distant speech that you can '
    'understand is written and tagged O or T.',
    'Leave out fillers: øh, øhm, æh, eh, hm, mm, mhm. Write ja, jo, nej, næ, nå, okay.',
    'Write repetitions and corrections in full; a cut-off word with a hyphen: mi-.',
    '[x] one stretch you cannot understand; {ord} your best guess (never parentheses); [bg] background talk you '
    'cannot understand.',
    'Status: speech if you wrote at least one word, unintelligible if someone speaks but nothing can be written, '
    'none if nobody speaks.',
    'Listen as often as you need: slow playback, the loop, the gain and the other microphone are there to be used. '
    'Use headphones. No automatic transcription or translation.',
    'Answer the questions after transcribing. For this group against another group, use the picture and the '
    'loudness. Choose cannot tell when you cannot tell.',
    'Your answers can be changed until the operator closes your blind pass. The full audit is then shown the '
    'automatic text of each window it transcribed, to rate; the reliability subset never sees it.',
]
PEER_RULE = [
    'About the task means the hands-on work or the lesson: programming the board, the microscope, the worksheet, '
    'asking for or giving help with them, handing material over.',
    'Something else means jokes, games, chat about other things, teasing, other pupils, phones.',
    'If both happen, choose what fills more of the 10 seconds.',
    'Talk addressed only to the teacher is not pupil-to-pupil talk.',
]
TRANSCRIPT_CODEBOOK = {
    'version': TRANSCRIPT_CODEBOOK_VERSION,
    'titles': {'transcribe': 'Transcribe the shaded 10 seconds, then answer the questions.',
               'reveal': 'Rate the automatic text of the shaded 10 seconds.'},
    'status': {'question': 'Is anyone speaking in the shaded 10 seconds?', 'short': 'whether anyone speaks',
               'answers': [{'value': 'speech', 'title': 'speech (I wrote words)'},
                           {'value': 'unintelligible', 'title': 'unintelligible (speech, but no word can be written)'},
                           {'value': 'none', 'title': 'none (no one speaks)'}]},
    'transcript': {'question': 'Write every word that begins inside the shaded 10 seconds, one line per turn; each '
                               'line starts with a tag.',
                   'tags': [{'tag': 'M', 'title': 'a pupil of this group'},
                            {'tag': 'T', 'title': 'the teacher or another adult'},
                            {'tag': 'O', 'title': 'a pupil of another group'}, {'tag': '?', 'title': 'cannot tell'}],
                   'markers': [{'marker': '[x]', 'title': 'one stretch you cannot understand'},
                               {'marker': '[bg]', 'title': 'background talk you cannot understand'},
                               {'marker': '{}', 'title': 'your best guess, inside the braces'}],
                   'rules': GUIDE},
    'overlap': {'question': 'Two or more people speak at the same time during part of the shaded 10 seconds.'},
    'who': {'question': P.AUDIT_CODEBOOK['speech']['question'], 'short': 'who speaks',
            'answers': [{'value': a['value'], 'title': a['title']} for a in P.AUDIT_CODEBOOK['speech']['answers']]},
    'adult': {'question': 'In the CURRENT 10 seconds, is an adult (the teacher) speaking, for example instructing, '
                          'explaining, or asking the class or the group?', 'short': 'whether an adult speaks',
              'answers': [{'value': 'yes', 'title': 'yes'}, {'value': 'no', 'title': 'no'},
                          {'value': 'cannot_tell', 'title': 'cannot tell'}]},
    'peer': {'question': 'In the CURRENT 10 seconds, are pupils of THIS GROUP talking with each other (one pupil '
                         'addressing another)?', 'short': 'whether pupils of this group talk',
             'answers': [{'value': 'task', 'title': 'yes, about the task'},
                         {'value': 'other', 'title': 'yes, about something else'},
                         {'value': 'none', 'title': 'no pupil-to-pupil talk in this group: silence, only the teacher, '
                                                    'or a pupil talking to themselves'},
                         {'value': 'cannot_tell', 'title': 'cannot tell'}],
             'rule': PEER_RULE},
    'other_offtask': {'question': 'Can you hear pupils of ANOTHER group talking about something other than the task?',
                      'short': "whether another group's pupils talk off the task",
                      'answers': [{'value': 'yes', 'title': 'yes'}, {'value': 'no', 'title': 'no'},
                                  {'value': 'cannot_tell', 'title': 'cannot tell'}]},
    'flag': 'Flag: an audio problem, or anything odd',
    'reveal': {'mine': 'Your transcript', 'model': 'What the content model read',
               'context': 'the 10 seconds before (its context)', 'no_context': '(no speech)', 'empty': 'the system wrote nothing; the content model was not asked',
               'gist': {'question': 'Does the automatic text convey what was said in the shaded 10 seconds?',
                        'answers': [{'value': 'yes', 'title': 'yes'}, {'value': 'partly', 'title': 'partly'},
                                    {'value': 'no', 'title': 'no'},
                                    {'value': 'nothing_said', 'title': 'nothing was said'}]},
               'invented': {'question': 'Does it contain words or phrases that nobody said?',
                            'answers': [{'value': 'none', 'title': 'none'}, {'value': 'some', 'title': 'some'},
                                        {'value': 'most', 'title': 'most'},
                                        {'value': 'cannot_tell', 'title': 'cannot tell'}]}},
    'closed': {'practice': 'Your blind pass is closed. Practice items are not revealed: nothing is asked here.',
               'subset': 'Your blind pass is closed. Thank you: nothing more is asked of the reliability subset.',
               'none': 'Your blind pass is closed, and no transcription of this item was saved before it: nothing is '
                       'asked here.'},
    'player': {'speeds': list(SPEEDS), 'gains_db': list(GAINS_DB),
               'channels': {'main': 'the table microphone', 'worn': 'the personal microphones'}},
    'cover': TARGET_COVER,
    'limits': {'lines': X.MAX_LINES, 'line': X.MAX_LINE, 'text': X.MAX_TEXT, 'note': P.MAX_NOTE},
}


# ---- what is served ----

def load_transcript_audit(artifacts: Path, audit_id: str, sweeps: int | None = None) -> dict:
    """what the server serves of a transcription audit: audit_page.load_audit's, with the sweeps served (`sweeps`,
    else the plan's) and only their items reachable"""
    plan = A.load_plan(artifacts, audit_id)
    if not S.transcript_plan(plan):
        raise A.AuditError(f'{audit_id} is a sensing audit, not a transcription audit')
    audit = P.load_audit(artifacts, audit_id)
    present = set()
    for _, item in audit['items'].values():
        if item.get('kind') != S.TASK or item.get('part') != 'speech':
            raise A.AuditError(f"{item.get('item')} is not a window of a transcription audit: its views hold its "
                               'windows only')
        present.add(_sweep(item))
    wanted = sweeps if sweeps is not None else (plan.get('sizes') or {}).get('sweeps', S.TRANSCRIPT_DEFAULTS['sweeps'])
    top = max(present, default=1)
    if isinstance(wanted, bool) or not isinstance(wanted, int) or not 1 <= wanted <= top:
        raise A.AuditError(f"--audit-sweeps is a sweep of the plan's, 1 to {top}")
    audit['items'] = {k: v for k, v in audit['items'].items() if _sweep(v[1]) <= wanted}
    audit.update(task=S.TASK, sweeps=wanted)
    return audit


def _sweep(item: dict) -> int:
    sweep = item.get('sweep')
    return sweep if isinstance(sweep, int) and not isinstance(sweep, bool) else 1


def sequence(audit: dict, subset: str | None) -> list[str]:
    """the item ids an auditor answers, in order: the practice block (each recording's practice windows, the
    recordings in the plan's order), then sweep by sweep each recording's windows of that sweep in its view's
    order, the recordings in the plan's order; with subset 'reliability' only the flagged windows, the recordings
    in reverse order. Only rendered items of the sweeps served."""
    served = audit['items']

    def ready(item):
        return item['item'] in served and item.get('render') == 'ok'
    views = {alias: audit['sessions'][alias]['view'] for alias in audit['order']}
    out = [i['item'] for alias in audit['practice'] for i in views[alias]['items'] if i.get('practice') and ready(i)]
    order = list(reversed(audit['order'])) if subset == 'reliability' else audit['order']
    for sweep in range(1, audit['sweeps'] + 1):
        for alias in order:
            out += [i['item'] for i in views[alias]['items'] if not i.get('practice') and ready(i)
                    and _sweep(i) == sweep and (subset != 'reliability' or i.get('reliability'))]
    return out


def phase_of(item: dict, own: dict, closed: int | None, subset: str | None) -> str | None:
    """the phase an auditor answers of an item now: 'transcribe' until their blind pass is closed (at seq
    `closed`); after it 'reveal' for an item not practice that has a reveal, when they answer the full audit and
    their transcription that counts was saved before the close; else None (nothing is left to answer)"""
    if closed is None:
        return 'transcribe'
    if item.get('practice') or subset is not None or not isinstance(item.get('reveal'), dict):
        return None
    seq = (own.get('transcribe') or {}).get('request_seq')
    return 'reveal' if isinstance(seq, int) and not isinstance(seq, bool) and seq < closed else None


def reveal_refusal(item: dict, closed: int | None, subset: str | None) -> str:
    """why an item takes no reveal rating from an auditor"""
    if item.get('practice'):
        return 'practice items are never revealed'
    if subset is not None:
        return 'the reliability subset is never shown the automatic text'
    if closed is None:
        return 'the automatic text is shown once your blind pass is closed'
    return 'no transcription of this item was saved before your blind pass was closed'


# ---- the answers ----

def union(pairs) -> list[list[float]]:
    """the parts of a clip played, merged where they touch or overlap, in order"""
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in pairs if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def target_cover(seen, target) -> float:
    """the share of the target [from, to] that the played parts cover"""
    a, b = float(target[0]), float(target[1])
    if b <= a:
        return 0.0
    return sum(max(0.0, min(y, b) - max(x, a)) for x, y in union(seen)) / (b - a)


def _span(item: dict) -> dict:
    span = item.get('span')
    return span if isinstance(span, dict) and 'target' in span and 'length' in span else S.CLIP_SPAN


def clean_listening(raw, item: dict) -> tuple[dict | None, str | None]:
    """(the listening record as saved, None) or (None, why it is refused): the numbers checked as a coding page's
    label line's are (code_locked.clean_record), the played parts inside the clip, the target's cover computed here"""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return None, 'listening is not an object'
    span = _span(item)
    top = float(span['length']) + SEEN_SLACK
    seen = raw.get('seen')
    if seen is None:
        seen = []
    if not isinstance(seen, list) or len(seen) > MAX_SEEN:
        return None, f'seen is a list of at most {MAX_SEEN} [from, to] pairs'
    pairs = []
    for pair in seen:
        if not isinstance(pair, list) or len(pair) != 2:
            return None, 'seen is a list of [from, to] pairs'
        a, b = P._number(pair[0]), P._number(pair[1])
        if a is None or b is None or not 0.0 <= a <= b <= top:
            return None, f'seen: each [from, to] lies in the clip, 0 to {top:g} s'
        pairs.append((a, b))
    merged = union(pairs)
    out: dict[str, Any] = {'seen': [[round(a, 3), round(b, 3)] for a, b in merged]}
    played = raw.get('played_seconds')
    if played is not None:
        number = P._number(played)
        if number is None:
            return None, 'played_seconds is not a number'
        played = round(min(max(number, 0.0), L.MAX_SECONDS), 2)
    out['played_seconds'] = played
    rate = raw.get('slowest_rate')
    if rate is not None:
        rate = P._number(rate)
        if rate is None or not MIN_RATE <= rate <= MAX_RATE:
            return None, f'slowest_rate is not a rate from {MIN_RATE:g} to {MAX_RATE:g}'
        rate = round(rate, 3)
    out['slowest_rate'] = rate
    gain = raw.get('highest_gain_db', 0)
    gain = P._number(gain) if gain is not None else 0.0
    if gain is None or not 0.0 <= gain <= GAINS_DB[-1]:
        return None, f'highest_gain_db is not a gain from 0 to {GAINS_DB[-1]} dB'
    out['highest_gain_db'] = round(gain, 1)
    channels = raw.get('channels')
    if channels is None:
        channels = []
    offered = [c for c in CHANNELS if c == 'main' or (item.get('clips') or {}).get(c)]
    if not isinstance(channels, list) or len(set(map(str, channels))) != len(channels) \
            or any(c not in offered for c in channels):
        return None, f"channels lists the microphones played, of {', '.join(offered)}"
    out['channels'] = [c for c in CHANNELS if c in channels]
    out['target_cover'] = round(target_cover(merged, span['target']), 3)
    return out, None


def clean_transcript_answer(raw, item: dict) -> tuple[dict | None, str | None]:
    """(a transcription as saved, None) or (None, why it is refused, as the page can say it): the transcript in
    audit_text's grammar, kept as it cleans it; the status agreeing with it and with the questions; only the
    values each question has; the listening record covering the shaded window unless the item is flagged"""
    if not isinstance(raw, dict):
        return None, 'the answer is not an object'
    common, error = P._flag_and_note(raw)
    if error:
        return None, error
    status = raw.get('status')
    if status not in X.STATUSES:
        return None, f"status is one of {', '.join(X.STATUSES)}"
    text = raw.get('transcript')
    if text is None:
        text = ''
    if not isinstance(text, str):
        return None, 'the transcript is not text'
    if len(text) > MAX_RAW:
        return None, f'the transcript is longer than {X.MAX_TEXT} characters'
    try:
        lines = X.parse_reference(text)
    except X.TranscriptError as refused:
        return None, str(refused)
    error = X.status_error(status, lines)
    if error:
        return None, error
    overlap = raw.get('overlap', False)
    if not isinstance(overlap, bool):
        return None, 'overlap is not true or false'
    values = {}
    for field, allowed in QUESTIONS.items():
        if raw.get(field) not in allowed:
            return None, f"{field} is one of {', '.join(allowed)}"
        values[field] = raw[field]
    if status == 'none' and values['who'] != 'none':
        return None, 'no one speaks: who speaks is none'
    if status != 'none' and values['who'] == 'none':
        return None, 'who speaks is none only when no one speaks (status none)'
    if status == 'none' and any(values[f] != v for f, v in NOBODY.items()):
        return None, 'no one speaks: no adult speaks, no pupils of this group talk, and none of another group'
    if status == 'none' and overlap:
        return None, 'no one speaks: no voices overlap (untick overlap)'
    if values['who'] == 'adult' and values['adult'] == 'no':
        return None, 'an adult speaks most (who speaks): the adult question is not no'
    listening, error = clean_listening(raw.get('listening'), item)
    if error:
        return None, error
    if not common['flag'] and listening['target_cover'] < TARGET_COVER:
        target = _span(item)['target']
        left = (1.0 - listening['target_cover']) * (float(target[1]) - float(target[0]))
        return None, (f'listen to the whole shaded window first ({math.ceil(left * 10) / 10:g} s of it not heard), '
                      'or flag the item')
    return {'status': status, 'transcript': X.reference_text(lines), 'overlap': overlap, **values, **common,
            'listening': listening}, None


def clean_reveal_answer(raw) -> tuple[dict | None, str | None]:
    """(a reveal rating as saved, None) or (None, why it is refused)"""
    if not isinstance(raw, dict):
        return None, 'the answer is not an object'
    common, error = P._flag_and_note(raw)
    if error:
        return None, error
    if raw.get('gist') not in GIST:
        return None, f"gist is one of {', '.join(GIST)}"
    if raw.get('invented') not in INVENTED:
        return None, f"invented is one of {', '.join(INVENTED)}"
    return {'gist': raw['gist'], 'invented': raw['invented'], **common}, None


# ---- the server ----

def private_file(path: Path) -> None:
    """an auditor's answers file, created readable by its owner only (it holds their transcripts), or made so, before
    a line is appended to it (code_locked's append opens it as it is)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
    os.chmod(path, 0o600)


class TranscriptHandler(P.AuditHandler):
    """audit_page's handler for a transcription audit's windows: its sequence by sweep, its two phases, the blind
    close and the reveal it releases"""
    TITLE = 'Transcription audit'
    CHOICE = ('Danish transcription', 'What is said in a short clip, written down word for word, then a few questions '
                                      'about it.')

    def _visible(self, subset) -> list[str]:
        return sequence(self.audit, subset)

    def _closed_at(self, name: str) -> int | None:
        """the seq of the name's blind close, None while it is open (a close without a readable seq closes it at
        -1: nothing before it is revealed)"""
        entry = S.read_blind_closed(self.artifacts, self.audit['audit_id']).get(name)
        if entry is None:
            return None
        seq = entry.get('seq') if isinstance(entry, dict) else None
        return seq if isinstance(seq, int) and not isinstance(seq, bool) else -1

    @staticmethod
    def _state(item: dict, own: dict, closed: int | None, subset: str | None) -> tuple[str | None, bool]:
        """(the phase the auditor answers of the item now, whether it is done: that phase answered, or none left)"""
        phase = phase_of(item, own, closed, subset)
        return phase, phase is None or phase in own

    # routes ----

    def _page(self, request: L.Request) -> None:
        page = OPEN_TRANSCRIPT_PAGE if self.open else TRANSCRIPT_PAGE
        self.send_html(based(homed(page, self.open) if self.together else page, self.base_path))

    def _boot(self, request: L.Request) -> None:
        who = self._auditor()
        if who is None:
            return
        name, subset = who
        mine = self._answers(name)
        closed = self._closed_at(name)
        rows = []
        for item_id in self._visible(subset):
            alias, item = self.audit['items'][item_id]
            phase, done = self._state(item, mine.get(item_id, {}), closed, subset)
            rows.append({'item': item_id, 'part': item['part'], 'alias': alias, 'practice': bool(item.get('practice')),
                         'sweep': _sweep(item), 'phase': phase, 'done': done})
        self.send_json({'audit_id': self.audit['audit_id'], 'mode': self.audit['mode'], 'kind': S.TASK,
                        'codebook': TRANSCRIPT_CODEBOOK, 'closed': self.closed(), 'blind_closed': closed is not None,
                        'auditor': name, 'subset': subset, 'sweeps': self.audit['sweeps'], 'sequence': rows,
                        **({'open': True} if self.open else {})})

    def _progress(self, request: L.Request) -> None:
        who = self._auditor()
        if who is None:
            return
        name, subset = who
        mine = self._answers(name)
        closed = self._closed_at(name)
        progress: dict[str, dict[str, dict[str, int]]] = {}
        for item_id in self._visible(subset):
            alias, item = self.audit['items'][item_id]
            phase, done = self._state(item, mine.get(item_id, {}), closed, subset)
            if phase is None:
                continue
            part = 'practice' if item.get('practice') else f'sweep_{_sweep(item)}'
            row = progress.setdefault(alias, {}).setdefault(part, {'items': 0, 'done': 0})
            row['items'] += 1
            row['done'] += done
        self.send_json({'progress': progress, 'phase': 'transcribe' if closed is None else 'reveal'})

    def _clip_url(self, item_id: str, channel: str) -> str:
        # under the prefix the audit is served at (audit_page: --audit-with), as every address the page is sent
        query = self._media_query()
        if channel == 'main':
            return f'{self.base_path}/audit/clip/{item_id}{query}'
        return (f"{self.base_path}/audit/clip/{item_id}{query}{'&' if query else '?'}"
                f"{urllib.parse.urlencode({'channel': channel})}")

    def _item(self, request: L.Request) -> None:
        who = self._auditor()
        if who is None:
            return
        name, subset = who
        visible = self._visible(subset)
        item_id = P._first(request.query, 'item')
        if item_id not in self.audit['items'] or item_id not in visible:
            return self.send_json({'error': 'no such item'}, 404)
        alias, item = self.audit['items'][item_id]
        self.note(session=self.audit['sessions'][alias]['id'], alias=alias, item=item_id)
        own = self._answers(name).get(item_id, {})
        closed = self._closed_at(name)
        phase, _ = self._state(item, own, closed, subset)
        clips = {c: self._clip_url(item_id, c) for c in CHANNELS if (item.get('clips') or {}).get(c)}
        span = _span(item)
        blind = own.get('transcribe')
        out: dict[str, Any] = {
            'item': item_id, 'part': item['part'], 'kind': S.TASK, 'alias': alias, 'practice': bool(item.get('practice')),
            'sweep': _sweep(item), 'position': visible.index(item_id), 'count': len(visible),
            'span': {k: span[k] for k in ('length', 'context', 'target', 'tail') if k in span}, 'clips': clips,
            'phase': phase, 'blind_closed': closed is not None, 'answer': blind.get('answer') if blind else None,
            'reveal': None}
        if phase == 'reveal':
            # the one system output the page shows: once the name's blind pass is closed, after its transcription
            shown, rated = item['reveal'], own.get('reveal')
            out['reveal'] = {'prev': str(shown.get('prev') or ''), 'cur': str(shown.get('cur') or ''),
                             'answer': rated.get('answer') if rated else None}
        self.send_json(out)

    def _image(self, request: L.Request) -> None:
        # a window has no pictures; a closed audit says so as for any picture
        if self._visible_item(request.rest.partition('/')[0]) is not None:
            self.send_json({'error': 'not found'}, 404)

    def _clip(self, request: L.Request) -> None:
        found = self._visible_item(request.rest)
        if found is None:
            return
        alias, item, _ = found
        self.note(session=self.audit['sessions'][alias]['id'], alias=alias, item=request.rest)
        channel = P._first(request.query, 'channel') or 'main'
        if channel != 'main':
            self.note(channel=channel[:20])
        name = (item.get('clips') or {}).get(channel) if channel in CHANNELS else None
        path = self.folder / A.MEDIA_DIR / alias / str(name or '')
        if not name or not path.is_file():
            return self.send_json({'error': 'not found'}, 404)
        self.send_file(path, 'video/mp4')

    def _answer(self, request: L.Request) -> None:
        body = request.body
        who = self._auditor()
        if who is None:
            return
        name, subset = who
        if self.closed():
            return self.send_json({'error': 'the audit is closed'}, 409)
        item_id, phase = body.get('item'), body.get('phase')
        if not isinstance(item_id, str) or item_id not in self.audit['items'] or item_id not in self._visible(subset):
            return self.send_json({'error': 'no such item'}, 404)
        alias, item = self.audit['items'][item_id]
        sid = self.audit['sessions'][alias]['id']
        self.note(session=sid, alias=alias, item=item_id, phase=phase)
        if phase not in TRANSCRIPT_PHASES:
            return self.send_json({'error': f'transcript items take no {str(phase)[:20]} answer'}, 400)

        def check(record: dict) -> tuple[int, str] | None:
            # under the request log's lock, which the blind close holds too: the close and the answers as they are
            # now, so a save is judged wholly before or wholly after a close
            if self.open:
                refused = self._open_refusal(name, subset)
                if refused:
                    return refused
            closed = self._closed_at(name)
            if phase == 'transcribe':
                if closed is not None:
                    return 409, 'the blind pass is closed: your transcriptions are no longer changed'
                answer, error = clean_transcript_answer(body.get('answer'), item)
            else:
                own = self._answers(name).get(item_id, {})
                if phase_of(item, own, closed, subset) != 'reveal':
                    return 409, reveal_refusal(item, closed, subset)
                answer, error = clean_reveal_answer(body.get('answer'))
            if error:
                return 400, error
            try:
                private_file(path)
            except OSError as failed:
                return 500, f'the record cannot be written: {failed.strerror or "file error"}'
            record['answer'] = answer
            return None

        path = self._answers_path(alias, name)
        record = {'audit_id': self.audit['audit_id'], 'item': item_id, 'part': item['part'], 'phase': phase,
                  'session': sid, 'alias': alias, 'auditor': name, 'token_id': self.identity.token_id, 'subset': subset,
                  'mode': self.audit['mode'], 'codebook': TRANSCRIPT_CODEBOOK_VERSION,
                  'practice': bool(item.get('practice')), 'reliability': bool(item.get('reliability')),
                  'sweep': _sweep(item), 'answer': None}
        if self.open:
            # saved under the typed name: no link, the scope as chosen
            record.update(open=True, scope='reliability' if subset == 'reliability' else 'full')
        spent = P._number(body.get('seconds_spent'))
        if spent is not None:
            record['seconds_spent'] = round(min(max(spent, 0.0), L.MAX_SECONDS), 3)
        answered = body.get('answered_at')
        if isinstance(answered, str) and len(answered) <= 40 and C.parse_time(answered) is not None:
            record['answered_at'] = answered
        record['saved_at'] = C.now_utc()
        self.append_record(path, path.relative_to(self.artifacts).as_posix(), record, check=check)


TRANSCRIPT_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transcription audit</title>
<style>
:root{--bg:#111;--panel:#1b1b1b;--line:#333;--text:#eee;--dim:#999;--accent:#ffd400;--good:#5fd38d;--bad:#ff6b6b;--context:#3a3a3a;--target:#8a7300}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,Helvetica,Arial,sans-serif}
header{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:8px 16px;border-bottom:1px solid var(--line)}
header b{font-size:15px}#where{color:var(--dim)}#practice{color:var(--accent);font-weight:600}
main{display:flex;gap:16px;padding:12px 16px;align-items:flex-start;flex-wrap:wrap}
#left{flex:3 1 560px;min-width:0}#right{flex:2 1 380px;min-width:300px}
video{display:block;max-width:100%;max-height:58vh;background:#000}
#timeline{position:relative;display:flex;height:22px;margin-top:6px;border:1px solid var(--line);cursor:pointer}
#t_context,#t_tail{background:var(--context)}#t_target{background:var(--target)}
#playhead{position:absolute;top:-4px;bottom:-4px;width:2px;background:#fff;left:0}
#controls{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}#player{color:var(--dim);margin-top:6px;min-height:20px}
button{font:inherit;background:#2a2a2a;color:var(--text);border:1px solid var(--line);border-radius:4px;padding:3px 9px;cursor:pointer}
kbd{background:#333;border-radius:3px;padding:0 5px}
#title{font-size:16px;margin-bottom:6px}.q{margin:10px 0 2px;font-weight:600}.choices label{display:block;padding:1px 0}
.check{display:block;margin-top:8px}#tags{margin:4px 0}
#transcript{width:100%;height:150px;font:15px/1.5 Menlo,Consolas,monospace;background:var(--panel);color:var(--text);border:1px solid var(--line)}
#note{width:100%;height:44px;margin-top:8px;background:var(--panel);color:var(--text);border:1px solid var(--line)}
#mine{white-space:pre-wrap;background:var(--panel);border:1px solid var(--line);padding:6px;font:14px/1.5 Menlo,Consolas,monospace}
#prev{white-space:pre-wrap;color:var(--dim);margin-top:4px}#cur{white-space:pre-wrap;font-size:15px;margin-top:4px}
#nav{display:flex;gap:8px;margin-top:10px}#status{min-height:20px;margin-top:6px}#status.bad{color:var(--bad)}#status.good{color:var(--good)}
#keys{color:var(--dim);margin-top:8px}details{margin-top:12px;color:var(--dim)}details li{margin:2px 0}details b{color:var(--text)}
</style></head><body>
<header><b>Transcription audit</b><span id="who"></span><span id="practice"></span><span id="where"></span><span id="progress"></span></header>
<main><section id="left"><video id="video" playsinline preload="auto"></video>
<div id="timeline"><div id="t_context"></div><div id="t_target"></div><div id="t_tail"></div><span id="playhead"></span></div>
<div id="controls"><button type="button" id="b_window">Play window <kbd>space</kbd></button>
<button type="button" id="b_before">From 2 s before <kbd>a</kbd></button>
<button type="button" id="b_previous">With the previous 10 s <kbd>p</kbd></button>
<button type="button" id="b_back">Back 2 s <kbd>b</kbd></button>
<button type="button" id="b_loop">Loop window <kbd>l</kbd></button>
<button type="button" id="b_speed">Speed <kbd>r</kbd></button>
<button type="button" id="b_gain">Gain <kbd>g</kbd></button>
<button type="button" id="b_mic" style="display:none">Microphone <kbd>c</kbd></button>
<button type="button" id="b_stop">Stop <kbd>s</kbd></button></div>
<div id="player"></div>
<details id="help"><summary id="helptoggle">How to transcribe, and the keys</summary><div id="helpbody"></div></details></section>
<section id="right"><div id="title"></div>
<div id="blind" style="display:none">
<div class="q" id="q_status"></div><div class="choices" id="f_status"></div>
<div class="q" id="q_transcript"></div>
<div id="tags"><button type="button" id="tag_M">M:</button> <button type="button" id="tag_T">T:</button> <button type="button" id="tag_O">O:</button> <button type="button" id="tag_q">?:</button> · <button type="button" id="mark_x">[x]</button> <button type="button" id="mark_bg">[bg]</button> <button type="button" id="mark_guess">{ }</button></div>
<textarea id="transcript" spellcheck="false" autocomplete="off" autocapitalize="off"></textarea>
<label class="check"><input type="checkbox" id="overlap"> <span id="q_overlap"></span></label>
<div class="q" id="q_who"></div><div class="choices" id="f_who"></div>
<div class="q" id="q_adult"></div><div class="choices" id="f_adult"></div>
<div class="q" id="q_peer"></div><div class="choices" id="f_peer"></div>
<div class="q" id="q_other_offtask"></div><div class="choices" id="f_other_offtask"></div></div>
<div id="reveal" style="display:none">
<div class="q" id="q_mine"></div><div id="mine"></div>
<div class="q" id="q_model"></div><div id="prev"></div><div id="cur"></div>
<div class="q" id="q_gist"></div><div class="choices" id="f_gist"></div>
<div class="q" id="q_invented"></div><div class="choices" id="f_invented"></div></div>
<div id="common" style="display:none"><label class="check"><input type="checkbox" id="flag"> <span id="q_flag"></span></label>
<textarea id="note" placeholder="note"></textarea></div>
<div id="nav"><button type="button" id="prev_item">← previous</button><button type="button" id="save" style="display:none">Save <kbd>Ctrl/⌘ Enter</kbd></button><button type="button" id="next_item">next →</button><button type="button" id="first_open" style="display:none">the first item not answered</button></div>
<div id="status"></div><div id="keys"></div></section></main>
<script>
const $ = id => document.getElementById(id);
// loading: a key pressed while the next item loads would act on the one before it, so it is dropped; loads counts the
// loads begun, so that one overtaken by a later one is dropped when it answers
let boot = null, auditor = null, cb = null, seq = [], pos = 0, item = null, phase = null, saving = false, loading = false, loads = 0, shownAt = 0;
// the form's radio buttons by question and value, and the form as loaded or last saved: a change asks before leaving
let groups = {}, savedForm = null;
// what the auditor heard of the item: the parts of the clip played, the seconds played, the slowest speed, the
// highest gain and the microphones; carried on from the saved answer, kept nowhere but in this page
let listen = null;
// the player: the part playing ([from, to] in clip seconds, null to the end), the loop, the speed and gain (their
// indices), the microphone, the WebAudio gain, the playhead's last time and where to go once another clip loads
let range = null, looping = false, speed = 0, gain = 0, channel = 'main', audio = null, last = null, pendingSeek = null;
const ASKED = ['status', 'who', 'adult', 'peer', 'other_offtask'];
const NOBODY = [['who', 'none'], ['adult', 'no'], ['peer', 'none'], ['other_offtask', 'no']];
const LEAVE = 'This item has changes that are not saved. Leave it without saving them?';
function said(text, mood) { $('status').textContent = text; $('status').className = mood || ''; }
function show(id, on) { $(id).style.display = on ? '' : 'none'; }
// Enter and space on a button, a link or the help's summary that has the focus are its own: the browser presses,
// follows or folds it, and no key of the page acts on them (any other key still does; a click hands the focus back)
function controlKey(e) { return (e.key === 'Enter' || e.key === ' ') && ['BUTTON', 'A', 'SELECT', 'SUMMARY'].includes((e.target || {}).tagName); }
// where the server serves this page's routes: the path prefix of an audit served beside another, set by the server
const BASE = '';
// the auditor is the one the personal link named: the page sends no name, and keeps none
async function api(path) {
  const r = await fetch(BASE + path); const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `status ${r.status}`); return data;
}
async function post(body) {
  const r = await fetch(BASE + '/api/audit/answer', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body)});
  const data = await r.json().catch(() => ({}));
  if (!r.ok || !data.ok) throw new Error(data.error || `status ${r.status}`); return data;
}
function node(tag, text) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; }
function list(texts) { const ul = node('ul'); for (const t of texts) ul.appendChild(node('li', t)); return ul; }

// ---- the form ----

function question(field) { return field === 'gist' || field === 'invented' ? cb.reveal[field] : cb[field]; }
function build() {
  groups = {};
  for (const field of ASKED.concat(['gist', 'invented'])) {
    const q = question(field), box = $(`f_${field}`);
    $(`q_${field}`).textContent = q.question; box.replaceChildren(); groups[field] = {};
    for (const a of q.answers) {
      const label = node('label'), input = node('input');
      input.type = 'radio'; input.name = field; input.value = a.value;
      // a choice made hands the keys back to the player, where space would choose it again
      input.addEventListener('change', () => { if (input.blur) input.blur(); chosen(field, a.value); });
      label.append(input, ' ' + a.title); box.appendChild(label); groups[field][a.value] = input;
    }
  }
  $('q_transcript').textContent = cb.transcript.question; $('q_overlap').textContent = cb.overlap.question;
  $('q_flag').textContent = cb.flag; $('q_mine').textContent = cb.reveal.mine; $('q_model').textContent = cb.reveal.model;
  for (const t of cb.transcript.tags) $(`tag_${t.tag === '?' ? 'q' : t.tag}`).title = t.title;
  const marks = {'[x]': 'mark_x', '[bg]': 'mark_bg', '{}': 'mark_guess'};
  for (const m of cb.transcript.markers) $(marks[m.marker]).title = m.title;
  const help = $('helpbody'), heading = text => { const p = node('p'); p.appendChild(node('b', text)); return p; };
  help.replaceChildren(heading('The transcript'), list(cb.transcript.rules), heading('Tags'),
    list(cb.transcript.tags.map(t => `${t.tag}: ${t.title}`)), heading('Markers'),
    list(cb.transcript.markers.map(m => `${m.marker} ${m.title}`)), heading('Pupils of this group: the task or something else'),
    list(cb.peer.rule), heading('Keys'), list(['space: play the shaded window; a: from 2 s before it to the end; p: with the previous 10 s; b: back 2 s; l: loop the window; r: speed; g: gain; c: the other microphone; s: stop',
    't: write the transcript; n: the note; ← →: the previous or next item',
    'In the transcript and the note: Ctrl/⌘ Enter saves, Esc plays the window, Alt ← goes back 2 s; every other key types']));
  $('keys').innerHTML = '<kbd>←</kbd> <kbd>→</kbd> move · <kbd>t</kbd> transcript · <kbd>n</kbd> note · in the text: <kbd>Ctrl/⌘ Enter</kbd> save, <kbd>Esc</kbd> window, <kbd>Alt ←</kbd> back 2 s';
}
function pick(field, value) { const g = groups[field] || {}; for (const v of Object.keys(g)) g[v].checked = v === value; }
function picked(field) { const g = groups[field] || {}; return Object.keys(g).find(v => g[v].checked) || null; }
// no one speaking answers the questions about who speaks too, and no voices overlap
function chosen(field, value) { if (field === 'status' && value === 'none') { for (const [f, v] of NOBODY) pick(f, v); $('overlap').checked = false; } }
function readForm() {
  const common = {flag: !!$('flag').checked, note: $('note').value};
  if (phase === 'transcribe') return Object.assign({status: picked('status'), transcript: $('transcript').value, overlap: !!$('overlap').checked,
    who: picked('who'), adult: picked('adult'), peer: picked('peer'), other_offtask: picked('other_offtask')}, common);
  if (phase === 'reveal') return Object.assign({gist: picked('gist'), invented: picked('invented')}, common);
  return null;
}
function fillForm() {
  const blind = item.answer || {}, rated = (item.reveal && item.reveal.answer) || {}, now = phase === 'reveal' ? rated : blind;
  for (const f of ASKED) pick(f, blind[f] || null);
  $('transcript').value = blind.transcript || ''; $('overlap').checked = !!blind.overlap;
  pick('gist', rated.gist || null); pick('invented', rated.invented || null);
  $('flag').checked = !!now.flag; $('note').value = now.note || '';
  savedForm = phase ? JSON.stringify(readForm()) : null;
}
function dirty() { return !!(item && phase && savedForm !== null && JSON.stringify(readForm()) !== savedForm); }
// a tag at the start of the caret's line (in place of the tag it has), the caret kept in the words
function tagLine(tag) {
  if (phase !== 'transcribe') return;
  const t = $('transcript'), text = t.value, at = Math.min(t.selectionStart ?? text.length, text.length);
  const start = at > 0 ? text.lastIndexOf('\n', at - 1) + 1 : 0, old = /^(M|T|O|\?): ?/.exec(text.slice(start)), cut = old ? old[0].length : 0;
  t.value = text.slice(0, start) + `${tag}: ` + text.slice(start + cut);
  const caret = start + tag.length + 2 + Math.max(0, at - start - cut);
  t.focus(); t.selectionStart = t.selectionEnd = caret;
}
// a marker at the caret, or braces round the selection, apart from the words around it
function mark(marker) {
  if (phase !== 'transcribe') return;
  const t = $('transcript'), text = t.value, a = Math.min(t.selectionStart ?? text.length, text.length);
  const b = Math.max(a, Math.min(t.selectionEnd ?? a, text.length)), before = text.slice(0, a), inner = text.slice(a, b), after = text.slice(b);
  const pad = before && !/\s$/.test(before) ? ' ' : '', tail = after && !/^[\s.,?!]/.test(after) ? ' ' : '';
  const body = marker === '{}' ? `{${inner}}` : marker;
  t.value = before + pad + body + tail + after;
  const caret = before.length + pad.length + (marker === '{}' && !inner ? 1 : body.length);
  t.focus(); t.selectionStart = t.selectionEnd = caret;
}

// ---- the player ----

function union(ranges) {
  const out = [];
  for (const [a, b] of ranges.filter(r => r[1] > r[0]).sort((x, y) => x[0] - y[0])) {
    const end = out[out.length - 1];
    if (end && a <= end[1]) end[1] = Math.max(end[1], b); else out.push([a, b]);
  }
  return out;
}
// the parts of the clip played, kept before its source changes (which empties video.played)
function keepSeen() {
  const v = $('video'), p = v.played; if (!listen || !p) return;
  const more = []; for (let i = 0; i < p.length; i++) more.push([p.start(i), p.end(i)]);
  listen.seen = union(listen.seen.concat(more));
}
function restore(saved) {
  const l = saved || {};
  listen = {seen: union((l.seen || []).map(p => [p[0], p[1]])), played: l.played_seconds || 0,
            slowest: typeof l.slowest_rate === 'number' ? l.slowest_rate : null, gain: l.highest_gain_db || 0, channels: (l.channels || []).slice()};
}
// what the record keeps of playing: the microphone, the speed and the gain it played at
function heard() {
  const v = $('video'); if (!listen || v.paused) return;
  if (!listen.channels.includes(channel)) listen.channels.push(channel);
  const s = v.playbackRate || 1; listen.slowest = listen.slowest === null ? s : Math.min(listen.slowest, s);
  listen.gain = Math.max(listen.gain, cb.player.gains_db[gain]);
}
function notHeard() {
  keepSeen(); const [a, b] = item.span.target; let s = 0;
  for (const [x, y] of listen.seen) s += Math.max(0, Math.min(y, b) - Math.max(x, a));
  return s >= cb.cover * (b - a) ? 0 : (b - a) - s;
}
function record() {
  keepSeen(); const r3 = x => Math.round(x * 1000) / 1000;
  return {seen: listen.seen.slice(0, 200).map(([a, b]) => [r3(a), r3(b)]), played_seconds: Math.round(listen.played * 10) / 10,
          slowest_rate: listen.slowest, highest_gain_db: listen.gain, channels: listen.channels.slice()};
}
function rate() {
  const v = $('video'), s = cb.player.speeds[speed];
  v.defaultPlaybackRate = s; v.playbackRate = s; v.preservesPitch = true; v.mozPreservesPitch = true; v.webkitPreservesPitch = true;
}
function playNow() {
  const v = $('video');
  if (audio && audio.ctx.state === 'suspended' && audio.ctx.resume) audio.ctx.resume().catch(() => {});
  const p = v.play(); if (p && p.catch) p.catch(() => {});
  heard(); drawPlayer();
}
function begin(from, to) {
  if (!item) return;
  const v = $('video'); keepSeen();
  range = to === null ? null : [from, to]; last = null;
  // a seek before the clip's metadata is not one: the browser would count the clip as played from its start
  if (v.readyState === 0) { pendingSeek = {t: Math.max(0, from), play: true}; drawPlayer(); return; }
  v.currentTime = Math.max(0, from); playNow();
}
function playWindow() { if (item) begin(item.span.target[0], item.span.target[1]); }
function fromBefore() { if (item) begin(item.span.target[0] - 2, item.span.length); }
function withPrevious() { if (item) begin(item.span.context[0], item.span.target[1]); }
function back() { if (item) begin(Math.max(0, ($('video').currentTime || 0) - 2), range ? range[1] : null); }
function stop() { const v = $('video'); if (v.pause) v.pause(); range = null; keepSeen(); drawPlayer(); }
function toggleLoop() { looping = !looping; said(looping ? 'the window loops' : 'the loop is off', 'plain'); if (looping) playWindow(); else drawPlayer(); }
function nextSpeed() { speed = (speed + 1) % cb.player.speeds.length; rate(); heard(); drawPlayer(); }
// the gain goes through WebAudio, made at the first raise: the file is not changed
function nextGain() {
  const next = (gain + 1) % cb.player.gains_db.length, db = cb.player.gains_db[next];
  if (db > 0 && !audio) {
    try {
      const AC = window.AudioContext || window.webkitAudioContext, ctx = new AC(), amp = ctx.createGain();
      ctx.createMediaElementSource($('video')).connect(amp); amp.connect(ctx.destination); audio = {ctx, node: amp};
    } catch (e) { said('this browser cannot raise the sound', 'bad'); return; }
  }
  gain = next;
  if (audio) { audio.node.gain.value = Math.pow(10, db / 20); if (audio.ctx.state === 'suspended' && audio.ctx.resume) audio.ctx.resume().catch(() => {}); }
  heard(); drawPlayer();
}
// the other microphone's clip, at the same moment, playing if this one played
function switchMic() {
  if (!item) return;
  if (!item.clips.worn) { said('this window has one microphone', 'plain'); return; }
  const v = $('video'); keepSeen();
  pendingSeek = {t: v.currentTime || 0, play: !v.paused};
  channel = channel === 'main' ? 'worn' : 'main';
  v.src = item.clips[channel]; v.load(); rate(); last = null;
  said(`now ${cb.player.channels[channel]}`, 'plain'); drawPlayer();
}
function loaded() {
  if (!pendingSeek) return;
  const v = $('video'), p = pendingSeek; pendingSeek = null;
  v.currentTime = p.t; rate(); if (p.play) playNow(); else drawPlayer();
}
function tick() {
  const v = $('video'), t = v.currentTime || 0;
  if (listen && !v.paused && last !== null && t > last && t - last < 1.5) listen.played += t - last;
  last = t;
  if (range && t >= range[1]) {
    if (looping && item) { range = item.span.target.slice(); v.currentTime = range[0]; last = null; }
    else { range = null; if (v.pause) v.pause(); keepSeen(); }
  }
  drawPlayer();
}
function ended() { range = null; keepSeen(); if (looping) playWindow(); else drawPlayer(); }
function seekTo(e) {
  const box = $('timeline'), r = box.getBoundingClientRect ? box.getBoundingClientRect() : null;
  if (!item || !r || !r.width) return;
  begin(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * item.span.length, null);
}
function drawTimeline() {
  const s = item ? item.span : null, width = p => s ? `${100 * (p[1] - p[0]) / s.length}%` : '0%';
  $('t_context').style.width = width(s && s.context); $('t_target').style.width = width(s && s.target); $('t_tail').style.width = width(s && s.tail);
}
function drawPlayer() {
  const v = $('video'), t = v.currentTime || 0;
  if (!item) { $('player').textContent = ''; $('playhead').style.left = '0%'; return; }
  const s = item.span, where = t < s.target[0] ? 'before the window' : t < s.target[1] ? 'in the window' : 'after the window';
  $('playhead').style.left = `${Math.max(0, Math.min(100, 100 * t / s.length))}%`;
  $('player').textContent = `${t.toFixed(1)} s of ${s.length} · ${where} · speed ${cb.player.speeds[speed]}× · gain +${cb.player.gains_db[gain]} dB · loop ${looping ? 'on' : 'off'}` +
    (item.clips.worn ? ` · ${cb.player.channels[channel]}` : '');
}

// ---- the items ----

function progressText() {
  const rows = seq.filter(s => s.phase), done = rows.filter(s => s.done).length;
  $('progress').textContent = rows.length ? `${done} of ${rows.length} ${boot && boot.blind_closed ? 'rated' : 'transcribed'}` : '';
}
// the items still to answer, each in the phase it is in
function unanswered() { return seq.filter(s => s.phase && !s.done).length; }
// the end of the sequence: thanks only when nothing is left, else how much is left and a way back to it
function endTitle() {
  const left = unanswered();
  return left ? `${left} ${left === 1 ? 'item is' : 'items are'} not answered yet.` : 'Every item is answered. Thank you.';
}
function paint() {
  $('practice').textContent = item && item.practice ? 'practice' : '';
  $('where').textContent = item ? `${item.alias} · item ${item.position + 1} of ${item.count}${item.sweep > 1 ? ` · sweep ${item.sweep}` : ''}` : '';
  show('blind', phase === 'transcribe'); show('reveal', phase === 'reveal'); show('common', !!phase); show('save', !!phase);
  show('b_mic', !!(item && item.clips.worn)); show('first_open', !item && unanswered() > 0);
  const c = cb.closed;
  $('title').textContent = !item ? endTitle() : phase ? cb.titles[phase] : item.practice ? c.practice : boot.subset ? c.subset : c.none;
  if (phase === 'reveal') {
    const mine = item.answer || {};
    $('mine').textContent = mine.transcript || (mine.status === 'none' ? '(no one speaks)' : '(no word written)');
    $('prev').textContent = `${cb.reveal.context}: ${item.reveal.prev || cb.reveal.no_context}`;
    $('cur').textContent = item.reveal.cur || cb.reveal.empty;
  } else { $('mine').textContent = ''; $('prev').textContent = ''; $('cur').textContent = ''; }
  drawTimeline(); drawPlayer(); progressText();
}
function stopPlayer() { const v = $('video'); if (v.pause) v.pause(); range = null; looping = false; pendingSeek = null; }
async function open(at) {
  const ticket = ++loads; loading = true;
  try { await load(at, ticket); } finally { if (ticket === loads) loading = false; }
}
// the item at `at` (the end past the last), shown only once it has come: a fetch that fails leaves the item shown
// where it was, and a load overtaken by a later one is dropped
async function load(at, ticket) {
  const to = Math.max(0, Math.min(at, seq.length));
  let next = null;
  if (to < seq.length) {
    try { next = await api(`/api/audit/item?item=${encodeURIComponent(seq[to].item)}`); }
    catch (e) { if (ticket === loads) said(`cannot load the item: ${e.message}`, 'bad'); return; }
    if (ticket !== loads) return;
  }
  stopPlayer(); pos = to; item = next;
  if (!item) { phase = null; savedForm = null; listen = null; paint(); return; }
  said(''); phase = item.phase; seq[pos].phase = item.phase;
  restore(item.answer && item.answer.listening);
  channel = 'main'; last = null;
  const v = $('video'); v.src = item.clips.main; v.load(); rate();
  fillForm(); shownAt = Date.now(); paint();
}
// another item, once the auditor agrees to leave changes unsaved; nothing while an item loads or a save is sent
async function go(at) {
  if (loading || saving) return;
  if (dirty() && !confirm(LEAVE)) return;
  await open(at);
}
function firstOpen() { const i = seq.findIndex(s => s.phase && !s.done); return i < 0 ? seq.length : i; }
// the next item to answer after this one, else the first before it: an item passed over comes round again
function nextOpen() { const i = seq.findIndex((s, k) => k > pos && s.phase && !s.done); return i < 0 ? firstOpen() : i; }
async function save() {
  if (!item || !phase || saving || loading) return;
  const answer = readForm();
  if (phase === 'transcribe') {
    const missing = ASKED.filter(f => !answer[f]);
    if (missing.length) { said(`NOT saved: answer ${missing.map(f => cb[f].short).join(', ')}`, 'bad'); return; }
    const left = notHeard();
    if (!answer.flag && left > 0) { said(`NOT saved: listen to the whole shaded window first (${left.toFixed(1)} s of it not heard), or flag the item`, 'bad'); return; }
    answer.listening = record();
  } else if (!answer.gist || !answer.invented) { said('NOT saved: answer both questions', 'bad'); return; }
  // the item saved, kept before the wait: the answer is marked on it, wherever the page is when the save returns
  const state = JSON.stringify(readForm()), row = seq[pos];
  saving = true; said('saving…');
  const body = {phase, item: item.item, answer, seconds_spent: Math.round((Date.now() - shownAt) / 100) / 10, answered_at: new Date().toISOString()};
  try { await post(body); } catch (e) { said(`NOT saved: ${e.message}`, 'bad'); saving = false; return; }
  saving = false; row.done = true; if (seq[pos] === row) savedForm = state; said('saved', 'good');
  await open(nextOpen());
}
const KEYS = {' ': playWindow, a: fromBefore, p: withPrevious, b: back, l: toggleLoop, r: nextSpeed, g: nextGain, c: switchMic, s: stop,
              t: () => { if (phase === 'transcribe') $('transcript').focus(); }, n: () => { if (phase) $('note').focus(); }};

async function onKey(e) {
  if (!boot) return;
  const k = e.key, typing = e.target === $('transcript') || e.target === $('note');
  // what the text fields pass on, taken before a key with a modifier is let through
  if (k === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); return save(); }
  // a button reached with Tab: space presses it rather than playing the window (Ctrl/⌘ Enter, above, still saves, as
  // the Save button says)
  if (controlKey(e)) return;
  if (typing) {
    if (k === 'Escape') { e.preventDefault(); return playWindow(); }
    if (k === 'ArrowLeft' && e.altKey) { e.preventDefault(); return back(); }
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey || loading) return;
  if (k === 'ArrowLeft') { e.preventDefault(); return go(pos - 1); }
  if (k === 'ArrowRight') { e.preventDefault(); return go(pos + 1); }
  const act = item && !e.repeat ? KEYS[k] : null;
  if (act) { e.preventDefault(); act(); }
}

async function start() {
  try { boot = await api('/api/audit/boot'); }
  catch (e) { said(`cannot start: ${e.message}`, 'bad'); return; }
  auditor = boot.auditor; seq = boot.sequence; cb = boot.codebook; build();
  $('who').textContent = `Auditing as ${auditor}${boot.subset ? ` (${boot.subset} items)` : ''}${boot.blind_closed ? ' · your blind pass is closed' : ''}`;
  if (boot.closed) said('This audit is closed: answers are no longer saved.', 'bad');
  const first = seq.findIndex(s => s.phase && !s.done);
  await open(first < 0 ? seq.length : first);
}

const v0 = $('video');
for (const [event, act] of [['timeupdate', tick], ['ended', ended], ['pause', () => { keepSeen(); drawPlayer(); }], ['play', heard],
                            ['seeking', () => { last = null; }], ['loadedmetadata', loaded]]) v0.addEventListener(event, act);
// a button pressed hands the keys back to the page, where space would press it again
for (const [id, act] of [['b_window', playWindow], ['b_before', fromBefore], ['b_previous', withPrevious], ['b_back', back], ['b_loop', toggleLoop],
                         ['b_speed', nextSpeed], ['b_gain', nextGain], ['b_mic', switchMic], ['b_stop', stop], ['tag_M', () => tagLine('M')],
                         ['tag_T', () => tagLine('T')], ['tag_O', () => tagLine('O')], ['tag_q', () => tagLine('?')], ['mark_x', () => mark('[x]')],
                         ['mark_bg', () => mark('[bg]')], ['mark_guess', () => mark('{}')], ['save', save], ['prev_item', () => go(pos - 1)],
                         ['next_item', () => go(pos + 1)], ['first_open', () => go(firstOpen())]]) $(id).addEventListener('click', () => { if ($(id).blur) $(id).blur(); act(); });
$('overlap').addEventListener('change', () => { if ($('overlap').blur) $('overlap').blur(); });
$('flag').addEventListener('change', () => { if ($('flag').blur) $('flag').blur(); });
// the help's summary, clicked, folds the help and hands the keys back to the page, where space would fold it again
$('helptoggle').addEventListener('click', () => { if ($('helptoggle').blur) $('helptoggle').blur(); });
$('timeline').addEventListener('click', seekTo);
window.addEventListener('beforeunload', e => { if (dirty()) { e.preventDefault(); e.returnValue = ''; } });
document.addEventListener('keydown', e => { onKey(e); });
start();
</script></body></html>
"""


def _patched(page: str, patches: list[tuple[str, str]],
             where: str = 'audit_transcript_page.OPEN_TRANSCRIPT_PAGE') -> str:
    for old, new in patches:
        if page.count(old) != 1:
            raise RuntimeError(f'the transcription page changed: {old[:60]!r} is not in it once; update {where}')
        page = page.replace(old, new)
    return page


# the open audit's page: the link's page with a name form before it, every request naming the auditor and scope; opened
# from the entry, or reloaded in a tab that audits, it starts without the form, and change name edits the name in place
# (audit_page's OPEN_FLOW, as the sensing audit's open page)
OPEN_TRANSCRIPT_PAGE = _patched(TRANSCRIPT_PAGE, [
    ('</style>', P.OPEN_STYLE + '</style>'),
    ('<header><b>Transcription audit</b><span id="who"></span>',
     '<header><b>Transcription audit</b><span id="who"></span>' + P.OPEN_RENAMING),
    ('<main><section id="left">',
     '<form id="named" style="display:none" autocomplete="off"><p>Type your name and choose what you answer: the full '
     'audit, or the reliability subset of the second auditor. Your answers are kept under this name, so type it the same '
     'way each time; this browser remembers it (and nothing of your answers).</p>\n'
     '<label>Name <input id="auditor" maxlength="100" autocomplete="off" spellcheck="false"></label>\n'
     '<label>You answer <select id="scope"><option value="">choose…</option><option value="full">the full audit</option>'
     '<option value="reliability">the reliability subset</option></select></label>\n'
     '<button type="submit">Start</button><div id="namestatus"></div></form>\n'
     '<main id="work" style="display:none"><section id="left">'),
    ("// the auditor is the one the personal link named: the page sends no name, and keeps none\n"
     "async function api(path) {\n  const r = await fetch(BASE + path);",
     "// the auditor is the name typed in (an open audit): every request carries it and the scope chosen, and the\n"
     "// browser remembers both for the next visit (an audit served beside another under keys of its own)\n"
     "let scope = null;\n"
     "function recall(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }\n"
     "function remember(key, value) { try { localStorage.setItem(key, value); } catch (e) {} }\n"
     "function stored(key) { return BASE ? `${BASE}/${key}` : key; }\n"
     "// an item with changes not saved is left only once the auditor agrees\n"
     "function mayLeave() { return !dirty() || confirm(LEAVE); }\n"
     "// the item shown, dropped (its player stopped) when the page starts under another name or asks for one\n"
     "function forgetItem() { stopPlayer(); item = null; phase = null; savedForm = null; listen = null; }\n"
     + P.OPEN_NAMED +
     "async function api(path, as) {\n  const r = await fetch(BASE + named(path, as));"),
    ("const r = await fetch(BASE + '/api/audit/answer', {", "const r = await fetch(BASE + named('/api/audit/answer'), {"),
    ("async function start() {\n"
     "  try { boot = await api('/api/audit/boot'); }\n"
     "  catch (e) { said(`cannot start: ${e.message}`, 'bad'); return; }\n",
     "// the name form, filled with what this browser remembers (or `kept`, a name and scope just refused); Enter or Start\n"
     "// begins\n"
     "function askName(text, kept) {\n"
     "  if (saving || !mayLeave()) return;\n"
     "  boot = null; forgetItem();\n"
     "  show('work', false); show('rename', false); show('named', true); $('who').textContent = '';\n"
     "  const [name, chosen] = kept || [auditor || recall(stored('auditor')), scope || recall(stored('auditScope'))];\n"
     "  $('auditor').value = name || ''; $('scope').value = chosen || '';\n"
     "  $('namestatus').textContent = text || ''; $('auditor').focus();\n"
     "}\n" + P.OPEN_FLOW),
    ("  $('who').textContent = `Auditing as ${auditor}${boot.subset ? ` (${boot.subset} items)` : ''}"
     "${boot.blind_closed ? ' · your blind pass is closed' : ''}`;\n",
     "  $('whorest').textContent = `${boot.subset ? ` (${boot.subset} items)` : ''}${boot.blind_closed ? ' · your blind pass is closed' : ''}`;\n"
     "  $('who').textContent = `Auditing as ${auditor}${$('whorest').textContent}`;\n"),
    ("document.addEventListener('keydown', e => { onKey(e); });\nstart();\n", P.OPEN_TAIL),
])


# the line of the pages' script that says where their routes are: an audit served beside another (audit_page,
# --audit-with) is served under a path prefix, which the server writes into it
BASE_LINE = "const BASE = '';"


def based(page: str, base: str) -> str:
    """a transcription page as served under the path prefix `base`, its requests sent there; the page as it is when the
    audit is served alone (base '')"""
    if not base:
        return page
    if page.count(BASE_LINE) != 1:
        raise RuntimeError(f'the transcription page changed: {BASE_LINE!r} is not in it once; update '
                           'audit_transcript_page.based')
    return page.replace(BASE_LINE, f'const BASE = {json.dumps(base)};')


# what a transcription page served beside another audit adds to audit_page.homed's link back to the entry: leaving
# through it leaves the item as a move to another item does, asking first when the item has changes not saved
# (mayLeave(): the open page's own, the link's page's added here), and the browser does not ask a second time as the
# page goes
UNLOAD_LINE = "window.addEventListener('beforeunload', e => { if (dirty()) { e.preventDefault(); e.returnValue = ''; } });\n"
UNLOAD_LEAVING = (
    "// leaving: the auditor agreed to leave the item through the link back to the entry, which the browser does not ask\n"
    "// again\n"
    "let leaving = false;\n"
    "window.addEventListener('beforeunload', e => { if (!leaving && dirty()) { e.preventDefault(); e.returnValue = ''; } });\n")
MAY_LEAVE = ("// an item with changes not saved is left only once the auditor agrees, as on the open page\n"
             "function mayLeave() { return !dirty() || confirm(LEAVE); }\n")
HOME_LEAVE = (
    "// the link back to the entry leaves the item as a move to another item does: nothing while an item loads or a save\n"
    "// is sent (the save not yet stored, nor yet refused), and with changes not saved, only once the auditor agrees. A\n"
    "// click that opens it in a tab or window of its own leaves nothing, and a step back to this page asks again\n"
    "$('home').addEventListener('click', e => {\n"
    "  if (e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;\n"
    "  e.preventDefault();\n"
    "  if (loading || saving) return;\n"
    "  if (mayLeave()) { leaving = true; location.assign($('home').href); }\n"
    "});\n"
    "window.addEventListener('pageshow', () => { leaving = false; });\n")


def homed(page: str, opened: bool) -> str:
    """TRANSCRIPT_PAGE, or OPEN_TRANSCRIPT_PAGE when `opened`, as served beside another audit: audit_page.homed's link
    back to the entry, which leaves the item as a move to another item does (based() then puts in the base)"""
    page = _patched(page, [(UNLOAD_LINE, UNLOAD_LEAVING)], 'audit_transcript_page.homed')
    return P.homed(page, opened, ('' if opened else MAY_LEAVE) + HOME_LEAVE)
