"""The page of a sensing audit (mmla ses-code --audit ID): one auditor at a time answers the roster,
identity, gaze and who-speaks questions of audit.py, on a server of its own (port 8766 unless -p, as a
campaign's).

The server is code_locked's guarded server, and an auditor is always someone a one-time link of scope
audit names (unless the audit is served open, below): the links are issued in the audit's own folder
(--campaign <that folder> --issue-token NAME --token-scope audit, --token-subset reliability for a
second auditor who answers the reliability subset only), whether the server binds 127.0.0.1 (behind
an SSH forward) or the tailnet address with --allow-from. No answer is saved under a name the page
gives, so no auditor reads or answers as another, and the scorer takes only the answers of claimed
audit links. Every request is logged in the audit's hash-chained request log, and every answer is
appended with its log line (append_record), so --verify-log checks the answers files both ways. A
closed audit is not served, and its pictures and clips are refused (410) by a server still running.

What the server reads: plan.json, each session's view.json and the answers files. It never opens a
pipeline*.json, so no pipeline output can reach an answer; in blind mode the only pipeline-derived
fact the page learns is which boxes a frozen version calls a pupil, and only once the frame's
identity answers are saved, which locks them. The locks (a frame's identity once saved, a recording's
roster once a frame of it is answered) are checked under the request log's lock against the answers
file as it is then, so two saves at once cannot both pass.

The sequence an auditor goes through: the practice items first (the roster of their recordings, then
their frames and windows), then each recording in the plan's order (reversed for the reliability
subset): its roster, its frames (identity, then gaze), its who-speaks windows. Answers:

  roster    {"wears": "yes" | "no" | "cannot_tell"}; changeable until the recording's first identity answer
  identity  {"boxes": {"1": "A" | "B" | "C" | "outside" | "not_person" | "cannot_tell", ...},
             "missed": 0..group size, "flag": false, "note": ""}; locked once saved
  gaze      {"boxes": {"1": {"class": <a GAZE_CLASSES value> | "between" | "cannot_tell",
                             "between": [two classes] when between}, ...}, "flag": false, "note": ""}
  speech    {"speaker": "none" | "member" | "adult" | "other_group" | "cannot_tell", "played_seconds": s,
             "flag": false, "note": ""}

Each is a line of artifacts/<session>/audit/<id>/answers/<auditor>.jsonl with the item, the phase, the
session, the auditor and link, the mode, the codebook version, whether it is practice or in the
reliability subset, the seconds spent, the page's and the server's times and the request's sequence
number in the log; the last line of an item's phase counts, but for identity, whose first line counts.

The page's viewing gate (a who-speaks clip played before an answer) trusts what the browser reports:
it is a convenience for the auditor, not a control.

The open audit (--audit ID --audit-open) is served as the default coding page is: no link, claim or
cookie. The page (OPEN_AUDIT_PAGE) asks for the auditor's name (code.name_error checks it, as the
default page checks a coder's; the browser remembers it) and whether they answer the full audit or the
reliability subset, and every request but the page itself carries both (?auditor=NAME&scope=full|
reliability; the server puts them into the picture and clip addresses it sends). A name is taken in
NFC, without control, formatting or line-break characters. Answers go to answers/<the name's file
form>.jsonl with `open`, the scope and no link; a default-page coder's name may be typed (the answers
are a file of their own). A name's file belongs to that name: a name whose file (in any case or Unicode
form) holds another name's answers or an answer saved through a link is refused, and so is a scope
other than the one its answers were saved under, all checked again under the log's lock when an answer
is saved; a refusal never says the other name. Once an answer is saved open, the audit is not served
with links again. The request log keeps the typed name on every request and save (its `coder`, which
--verify-log checks against the `auditor` of each open line), so --verify-log still checks the answers
files both ways; the start line records `open`. --allow-from is optional: a tailnet bind without it
lets every tailnet address in (0.0.0.0 and LAN addresses stay refused). Every rule of what an auditor
sees holds per typed name; that the person typing a name is that auditor is not checked, so one person
may answer a frame's identity under a second name, see which boxes the frozen versions call pupils,
then answer it under their own (the scorer lists the saves under other names from the primary's
address and browser).

A transcription audit (a plan whose task is 'transcript', drawn by audit_speech) is served by this
command too, with links or open as above, but with the routes and the page of audit_transcript_page
(TranscriptHandler, TRANSCRIPT_PAGE); --audit-sweeps says how many of its sweeps are served.
"""
from __future__ import annotations

import ipaddress
import json
import os
import threading
import unicodedata
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from openmmla.commands.ses import audit as A
from openmmla.commands.ses import code as C
from openmmla.commands.ses import code_locked as L

L.register_loaded(__name__, __file__)

# the campaign server's port: a campaign and an audit served at once need -p for one of them
DEFAULT_PORT = L.DEFAULT_PORT
AUDIT_MODULES = ('openmmla.commands.ses.code', 'openmmla.commands.ses.code_locked', 'openmmla.commands.ses.audit',
                 'openmmla.commands.ses.audit_page')
# a transcription audit's server runs these as well: its page and routes, the grammar its transcripts are checked
# with and the blind closes it reads (a sensing audit's start line names AUDIT_MODULES only, as before)
TRANSCRIPT_MODULES = AUDIT_MODULES + ('openmmla.commands.ses.audit_text', 'openmmla.commands.ses.audit_speech',
                                      'openmmla.commands.ses.audit_transcript_page')
# where a gaze lands, in the order the pipeline breaks ties (features.gaze_target, window_features._gaze_label):
# any face before any hands, among hands the nearest, then the work area or a zone, then elsewhere, then out of frame
GAZE_CLASSES = [
    {'key': '1', 'value': 'member_face', 'title': "a group member's face"},
    {'key': '2', 'value': 'other_face', 'title': 'the face of someone not in the group'},
    {'key': '3', 'value': 'own_hands', 'title': 'their own hands, or what they hold'},
    {'key': '4', 'value': 'member_hands', 'title': "a group member's hands, or what they hold"},
    {'key': '5', 'value': 'other_hands', 'title': 'the hands of someone not in the group, or what they hold'},
    {'key': '6', 'value': 'task', 'title': "the task material or work area on the table, in nobody's hands"},
    {'key': '7', 'value': 'elsewhere', 'title': 'somewhere else in the picture'},
    {'key': '8', 'value': 'out_of_frame', 'title': 'outside the picture'},
]
GAZE_VALUES = tuple(c['value'] for c in GAZE_CLASSES)
WEARS = ('yes', 'no', 'cannot_tell')
BOX_VALUES = ('outside', 'not_person', 'cannot_tell')
SPEAKERS = ('none', 'member', 'adult', 'other_group', 'cannot_tell')
AUDIT_CODEBOOK = {
    'version': A.CODEBOOK_VERSION,
    'roster': {'question': 'Is the badge marked with the diamond worn by the person in the yellow box?',
               'answers': [{'key': 'y', 'value': 'yes', 'title': 'yes'}, {'key': 'n', 'value': 'no', 'title': 'no'},
                           {'key': 'x', 'value': 'cannot_tell', 'title': 'cannot tell'}]},
    'identity': {'question': 'Who is in this box? Compare with the reference pictures of each pupil.',
                 'answers': [{'key': 'o', 'value': 'outside', 'title': 'someone not in the group'},
                             {'key': 'p', 'value': 'not_person', 'title': 'not a person'},
                             {'key': 'x', 'value': 'cannot_tell', 'title': 'cannot tell'}],
                 'missed': 'How many group members are in the picture without a box of their own?',
                 'rule': 'Give a pupil to every box that shows them, also when two boxes show one pupil. Answers are '
                         'locked once saved.'},
    'gaze': {'question': 'Where is the person in the yellow box looking?',
             'rule': 'Take the first that applies: a face (anyone\'s) before any hands; among hands, whoever\'s hands '
                     'the gaze lands nearest (their own, a member\'s or someone else\'s); then the task material or work '
                     'area; then somewhere else in the picture; then outside the picture. When it lands between two of '
                     'these and you cannot choose, press 9 and name both.',
             'classes': GAZE_CLASSES, 'between': {'key': '9', 'value': 'between', 'title': 'between two of these'},
             'cannot_tell': {'key': 'x', 'value': 'cannot_tell', 'title': 'cannot tell'}},
    'speech': {'question': 'Who speaks in these 10 seconds? When several speak, who speaks most?',
               'rule': 'Judge by eye and ear in the clip.',
               'answers': [{'key': '0', 'value': 'none', 'title': 'no one'},
                           {'key': '1', 'value': 'member', 'title': 'a group member'},
                           {'key': '2', 'value': 'adult', 'title': 'the teacher or another adult'},
                           {'key': '3', 'value': 'other_group', 'title': 'pupils of another group'},
                           {'key': 'x', 'value': 'cannot_tell', 'title': 'cannot tell'}]},
}
MAX_NOTE = 2000
PHASES = {'roster': 'roster', 'identity': 'vision', 'gaze': 'vision', 'speech': 'speech'}
# what an open audit's auditor chooses to answer: the scope's subset (audit_page.sequence)
SCOPES = {'full': None, 'reliability': 'reliability'}


def _first(query: dict, name: str) -> str | None:
    values = query.get(name)
    return values[0] if values else None


def typed_name(raw) -> tuple[str | None, str | None]:
    """(an open audit's auditor name, None) or (None, why it is refused): the name as typed, trimmed, in NFC
    (a name typed in two Unicode forms is one name, as a file system that ignores the form opens one file
    for both), not empty, without a control, formatting or line-break character (it would forge or hide
    lines of the scores' report) and short enough to name a file (code.name_error, as the default page
    checks its coder names)"""
    name = unicodedata.normalize('NFC', raw).strip() if isinstance(raw, str) else ''
    if not name:
        return None, 'type your name first'
    if any(unicodedata.category(c).startswith('C') or unicodedata.category(c) in ('Zl', 'Zp') for c in name):
        return None, 'the name holds a control, formatting or line-break character: type it again'
    error = C.name_error(name)
    return (None, error.replace('coder name', 'name')) if error else (name, None)


def folded(stem: str) -> str:
    """an answers file's name as a file system blind to case and Unicode form sees it: two names whose files
    fold alike share one file there"""
    return unicodedata.normalize('NFC', stem).casefold()


def scope_text(subset: str | None) -> str:
    return 'the reliability subset' if subset == 'reliability' else 'the full audit'


def open_bind(bind: str, allow_from: str | None, allow_wide: bool = False) -> tuple:
    """the networks that may connect to an open audit: as code_locked.check_bind, but a tailnet bind needs no
    --allow-from (then every tailnet address may connect, as anyone who reaches the default page may)"""
    if not (allow_from or '').strip():
        try:
            address = ipaddress.ip_address('127.0.0.1' if bind == 'localhost' else bind)
        except ValueError:
            address = None  # check_bind says why
        if address is not None and any(address.version == n.version and address in n for n in L.TAILNET):
            return L.TAILNET
    return L.check_bind(bind, allow_from, allow_wide)


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float('inf') else None


def load_audit(artifacts: Path, audit_id: str) -> dict:
    """what the server serves of an audit: the plan's order and each session's view, never a pipeline file"""
    plan = A.load_plan(artifacts, audit_id)
    sessions, items = {}, {}
    for entry in plan['sessions']:
        view = A.read_json(A.session_audit_dir(artifacts, entry['id'], audit_id) / A.VIEW_FILE)
        sessions[entry['alias']] = {'id': entry['id'], 'view': view}
        for item in view['items']:
            items[item['item']] = (entry['alias'], item)
    practice = [entry['alias'] for entry in plan['sessions']
                if any(i.get('practice') for i in sessions[entry['alias']]['view']['items'])]
    return {'audit_id': audit_id, 'mode': plan['mode'], 'order': list(plan['order']), 'sessions': sessions,
            'items': items, 'practice': practice}


def saved_open(artifacts: Path, audit: dict) -> int:
    """how many answer lines of the audit the open page saved (a link's page would show those typed under its
    auditor's name as theirs, so the audit is not served with links again)"""
    count = 0
    for _, path in A.record_files(artifacts, audit['audit_id'], [s['id'] for s in audit['sessions'].values()]):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            count += isinstance(record, dict) and record.get('open') is True
    return count


def sequence(audit: dict, subset: str | None) -> list[str]:
    """the item ids an auditor answers, in order: the practice block, then each recording's roster, frames
    and windows; with subset 'reliability' only the flagged items, the recordings in reverse order"""
    def ready(item):
        return item.get('render') == 'ok'

    out, done_roster = [], set()
    views = {alias: audit['sessions'][alias]['view'] for alias in audit['order']}
    for alias in audit['practice']:
        out += [i['item'] for i in views[alias]['items'] if i['part'] == 'roster' and ready(i)]
        done_roster.add(alias)
    for part in ('vision', 'speech'):
        for alias in audit['practice']:
            out += [i['item'] for i in views[alias]['items'] if i['part'] == part and i.get('practice') and ready(i)]
    order = list(reversed(audit['order'])) if subset == 'reliability' else audit['order']
    for alias in order:
        chosen = [i for i in views[alias]['items'] if i['part'] in ('vision', 'speech') and not i.get('practice')
                  and ready(i) and (subset != 'reliability' or i.get('reliability'))]
        if not chosen:
            continue
        if alias not in done_roster:
            out += [i['item'] for i in views[alias]['items'] if i['part'] == 'roster' and ready(i)]
        out += [i['item'] for i in chosen if i['part'] == 'vision'] + [i['item'] for i in chosen if i['part'] == 'speech']
    return out


class AuditHandler(L.GuardedHandler, BaseHTTPRequestHandler):
    """the audit's page and API, for the auditor a link binds (or, open, the name a request carries)"""
    KIND = 'audit'
    HOME = '/audit'
    TITLE = 'Sensing audit'
    ROUTES = {('GET', '/'): '_home', ('GET', '/audit'): '_page', ('GET', '/api/audit/boot'): '_boot',
              ('GET', '/api/audit/item'): '_item', ('GET', '/api/audit/progress'): '_progress',
              ('POST', '/api/audit/answer'): '_answer'}
    PREFIX_ROUTES = (('GET', '/audit/img/', '_image'), ('GET', '/audit/clip/', '_clip'))
    audit: dict = {}
    folder: Path | None = None
    artifacts: Path | None = None
    cache: dict = {}
    cache_lock = threading.Lock()
    # an open audit (--audit-open): no link, the auditor is the name each request carries
    open: bool = False

    # who ----

    def _identity(self) -> L.Identity | None:
        """the link's auditor (GuardedHandler); open, the name and scope of the request's query, never None: the
        page itself needs neither, and the routes that do refuse a request without them (_auditor)"""
        if not self.open:
            return super()._identity()
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        name, self._typed_error = typed_name(_first(query, 'auditor'))
        scope = _first(query, 'scope')
        if scope is not None:
            self.note(scope=scope[:20])
        if self._typed_error is None and scope not in SCOPES:
            self._typed_error = 'choose the full audit or the reliability subset (scope=full or scope=reliability)'
        return L.Identity(name, None, self.KIND, SCOPES.get(scope), None)

    def note_query(self, query: dict) -> None:
        if self.open and self._fields.get('device', '') is None:
            del self._fields['device']  # an open audit binds no device: its log lines name none
        super().note_query(query)

    def _claim(self, method: str, token: str) -> None:
        if not self.open:
            return super()._claim(method, token)
        self.send_message('/c/', 'This audit takes no links: open /audit and type your name.', 404)

    def _auditor(self) -> tuple[str, str | None] | None:
        """(the auditor's name, their subset), or None after answering why not: the link's (GuardedHandler
        answers 401 without one); open, the name and scope the request carries, when _open_refusal allows them"""
        if not self.open:
            return self.identity.name, self.identity.subset
        refused = (400, self._typed_error) if self._typed_error else \
            self._open_refusal(self.identity.name, self.identity.subset)
        if refused:
            self.send_json({'error': refused[1]}, refused[0])
            return None
        return self.identity.name, self.identity.subset

    def _open_refusal(self, name: str, subset: str | None) -> tuple[int, str] | None:
        """why a typed name cannot be used now, or None: the answers files of its file form (in any case and
        Unicode form, as a file system blind to them would open them) hold no other name's answers and no
        answer saved through a link, the file the name writes is listed under that very name, and its answers
        so far were saved under the scope it chose. The other name is never said: typing it would show its
        answers."""
        form, own = folded(C.safe_name(name)), f'{C.safe_name(name)}.jsonl'
        taken = (409, 'another name already uses the file of this name (it differs in case, punctuation or accents): '
                      'type another name')
        saved: set = set()
        for alias in self.audit['order']:
            folder = A.answers_dir(self.artifacts, self.audit['sessions'][alias]['id'], self.audit['audit_id'])
            # asked before the listing: a file the name's first save creates meanwhile is then listed too
            exists = (folder / own).exists()
            try:
                listed = os.listdir(folder)
            except (FileNotFoundError, NotADirectoryError):
                continue
            # a file system blind to case or Unicode form opens a file listed under another spelling
            if exists and own not in listed:
                return taken
            for entry in listed:
                path = folder / entry
                if path.suffix != '.jsonl' or folded(path.stem) != form or not path.is_file():
                    continue
                found = self._parsed(path)
                if found is None:
                    continue
                if found[2]:
                    return 409, 'the answers under this name were saved through an audit link: type another name'
                if any(n != name for n in found[1]):
                    return taken
                if name in found[1]:
                    saved |= found[1][name][1]
        if saved and subset not in saved:
            return 409, (f'{name} answers {scope_text(sorted(saved, key=str)[0])}: choose it, or type another name')
        return None

    def _parsed(self, path: Path) -> tuple[dict, dict, bool] | None:
        """an answers file read (again when it changed), None when there is none: (item -> phase -> the record
        that counts, an identity's first line and any other phase's last; auditor -> (the same of their own
        lines, the subsets they were saved under); whether a line was saved through a link, not open)"""
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        stamp = (stat.st_mtime_ns, stat.st_size)
        with self.cache_lock:
            cached = self.cache.get(path)
        if cached is None or cached[0] != stamp:
            parsed: dict[str, dict[str, dict]] = {}
            by_name: dict[str, tuple[dict, set]] = {}
            linked = False
            for line in path.read_text(encoding='utf-8').splitlines():
                try:
                    record = json.loads(line)
                    phases = parsed.setdefault(record['item'], {})
                    if record['phase'] != 'identity' or 'identity' not in phases:
                        phases[record['phase']] = record
                except (ValueError, KeyError, TypeError):
                    continue
                linked = linked or record.get('open') is not True
                if isinstance(record.get('auditor'), str):
                    own, subsets = by_name.setdefault(record['auditor'], ({}, set()))
                    phases = own.setdefault(record['item'], {})
                    if record['phase'] != 'identity' or 'identity' not in phases:
                        phases[record['phase']] = record
                    subset = record.get('subset')
                    subsets.add(subset if isinstance(subset, str) else None)
            cached = (stamp, parsed, by_name, linked)
            with self.cache_lock:
                self.cache[path] = cached
        return cached[1], cached[2], cached[3]

    def _answers(self, name: str) -> dict[str, dict[str, dict]]:
        """the auditor's answers, item -> phase -> the record that counts (an identity's first line, any
        other phase's last), from every session's answers file; open, of the lines saved under the name only"""
        out: dict[str, dict[str, dict]] = {}
        for alias in self.audit['order']:
            found = self._parsed(self._answers_path(alias, name))
            if found is None:
                continue
            chosen = (found[1].get(name) or ({}, set()))[0] if self.open else found[0]
            for item, phases in chosen.items():
                out.setdefault(item, {}).update(phases)
        return out

    def _answers_path(self, alias: str, name: str) -> Path:
        sid = self.audit['sessions'][alias]['id']
        return A.answers_dir(self.artifacts, sid, self.audit['audit_id']) / f'{C.safe_name(name)}.jsonl'

    def _media_query(self) -> str:
        """what the addresses of pictures and clips carry: nothing for a link's auditor (the cookie names them);
        open, the auditor and scope, since the browser fetches them by address"""
        if not self.open:
            return ''
        scope = next(k for k, v in SCOPES.items() if v == self.identity.subset)
        return '?' + urllib.parse.urlencode({'auditor': self.identity.name, 'scope': scope})

    def _visible(self, subset) -> list[str]:
        return sequence(self.audit, subset)

    @staticmethod
    def _asked(item: dict, identity: dict | None) -> list[str]:
        """the boxes the gaze question asks about: every box the auditor gave a pupil and every box a frozen
        version calls one; none for a frame flagged broken"""
        if not identity or (identity.get('answer') or {}).get('flag'):
            return []
        mine = {n for n, value in ((identity.get('answer') or {}).get('boxes') or {}).items() if value in A.LETTERS}
        return sorted((mine | set(item.get('ask_gaze') or [])) & set(item.get('boxes') or {}), key=int)

    def _done(self, item: dict, mine: dict) -> bool:
        if item['part'] == 'roster':
            return 'roster' in mine
        if item['part'] == 'speech':
            return 'speech' in mine
        identity = mine.get('identity')
        return identity is not None and ('gaze' in mine or not self._asked(item, identity))

    # routes ----

    def _home(self, request: L.Request) -> None:
        self.send_body(303, 'text/plain', b'', (('location', '/audit'),))

    def _page(self, request: L.Request) -> None:
        self.send_html(OPEN_AUDIT_PAGE if self.open else AUDIT_PAGE)

    def _boot(self, request: L.Request) -> None:
        who = self._auditor()
        if who is None:
            return
        name, subset = who
        mine = self._answers(name)
        rows = []
        for item_id in self._visible(subset):
            alias, item = self.audit['items'][item_id]
            rows.append({'item': item_id, 'part': item['part'], 'alias': alias, 'practice': bool(item.get('practice')),
                         'done': self._done(item, mine.get(item_id, {}))})
        self.send_json({'audit_id': self.audit['audit_id'], 'mode': self.audit['mode'], 'codebook': AUDIT_CODEBOOK,
                        'closed': self.closed(), 'auditor': name, 'subset': subset, 'sequence': rows,
                        **({'open': True} if self.open else {})})

    def _progress(self, request: L.Request) -> None:
        who = self._auditor()
        if who is None:
            return
        name, subset = who
        mine = self._answers(name)
        progress: dict[str, dict[str, dict[str, int]]] = {}
        for item_id in self._visible(subset):
            alias, item = self.audit['items'][item_id]
            part = 'practice' if item.get('practice') else item['part']
            row = progress.setdefault(alias, {}).setdefault(part, {'items': 0, 'done': 0})
            row['items'] += 1
            row['done'] += self._done(item, mine.get(item_id, {}))
        self.send_json({'progress': progress})

    def _url(self, item_id: str, name: str | None) -> str | None:
        return f'/audit/img/{item_id}/{name}{self._media_query()}' if name else None

    def _item(self, request: L.Request) -> None:
        who = self._auditor()
        if who is None:
            return
        name, subset = who
        visible = self._visible(subset)
        item_id = _first(request.query, 'item')
        if item_id not in self.audit['items'] or item_id not in visible:
            return self.send_json({'error': 'no such item'}, 404)
        alias, item = self.audit['items'][item_id]
        self.note(session=self.audit['sessions'][alias]['id'], alias=alias, item=item_id)
        mine = self._answers(name)
        own = mine.get(item_id, {})
        out: dict[str, Any] = {'item': item_id, 'part': item['part'], 'alias': alias, 'practice': bool(item.get('practice')),
                               'position': visible.index(item_id), 'count': len(visible)}
        if item['part'] == 'roster':
            out.update(pupil=item['pupil'], image=self._url(item_id, item.get('image')),
                       answer=(own.get('roster') or {}).get('answer'))
        elif item['part'] == 'speech':
            out.update(clip=f'/audit/clip/{item_id}{self._media_query()}',
                       answer=(own.get('speech') or {}).get('answer'))
        else:
            view = self.audit['sessions'][alias]['view']
            images = item.get('images') or {}
            gallery = {letter: [] for letter in view['pupils']}
            for other in view['items']:
                said = ((mine.get(other['item']) or {}).get('roster') or {}).get('answer') or {}
                if other['part'] == 'roster' and other.get('render') == 'ok' and said.get('wears') == 'yes':
                    gallery.setdefault(other['pupil'], []).append(self._url(other['item'], other.get('image')))
            identity = own.get('identity')
            out.update(size=item['size'], boxes=item['boxes'], image=self._url(item_id, images.get('ids')),
                       pupils=view['pupils'], group_size=view['group_size'], gallery=gallery,
                       identity=identity.get('answer') if identity else None, gaze=None)
            if identity is not None:
                asked = self._asked(item, identity)
                out['gaze'] = {'ask': asked, 'answer': (own.get('gaze') or {}).get('answer'),
                               'images': {n: {'highlight': self._url(item_id, (images.get('gaze') or {}).get(n)),
                                              'head': self._url(item_id, (images.get('head') or {}).get(n))}
                                          for n in asked}}
        self.send_json(out)

    def _visible_item(self, item_id: str) -> tuple[str, dict, str] | None:
        """(alias, item, auditor) of an item the requester may see, else None after answering 404, or 410
        once the audit is closed (its pictures and clips are deleted, and none is sent again)"""
        if self.closed():
            self.send_json({'error': 'the audit is closed'}, 410)
            return None
        who = self._auditor()
        if who is None:
            return None
        name, subset = who
        if item_id not in self.audit['items'] or item_id not in self._visible(subset):
            self.send_json({'error': 'not found'}, 404)
            return None
        alias, item = self.audit['items'][item_id]
        return alias, item, name

    def _image(self, request: L.Request) -> None:
        item_id, _, name = request.rest.partition('/')
        found = self._visible_item(item_id)
        if found is None:
            return
        alias, item, auditor = found
        self.note(session=self.audit['sessions'][alias]['id'], alias=alias, item=item_id)
        images = item.get('images') or {}
        allowed = {item.get('image'), images.get('ids')}
        if any(name == n for n in list((images.get('gaze') or {}).values()) + list((images.get('head') or {}).values())):
            # the pictures of the gaze question only once the frame's identity answers are locked
            if 'identity' in self._answers(auditor).get(item_id, {}):
                allowed.add(name)
        if not name or name not in allowed:
            return self.send_json({'error': 'not found'}, 404)
        path = self.folder / A.MEDIA_DIR / alias / name
        if not path.is_file():
            return self.send_json({'error': 'not found'}, 404)
        self.send_file(path, 'image/jpeg')

    def _clip(self, request: L.Request) -> None:
        found = self._visible_item(request.rest)
        if found is None:
            return
        alias, item, _ = found
        self.note(session=self.audit['sessions'][alias]['id'], alias=alias, item=request.rest)
        path = self.folder / A.MEDIA_DIR / alias / str(item.get('clip') or '')
        if item['part'] != 'speech' or not item.get('clip') or not path.is_file():
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
        if PHASES.get(phase) != item['part']:
            return self.send_json({'error': f'{item["part"]} items take no {str(phase)[:20]} answer'}, 400)
        view = self.audit['sessions'][alias]['view']

        def check(record: dict) -> tuple[int, str] | None:
            # under the request log's lock: the answers as saved by now, so two saves at once are judged one
            # after the other and a lock cannot be passed by both
            if self.open:
                # a name's first saves under two scopes at once, or two names of one file form, are judged so too
                refused = self._open_refusal(name, subset)
                if refused:
                    return refused
            mine = self._answers(name)
            answer, error = clean_answer(phase, body.get('answer'), item, view, mine.get(item_id, {}), self._asked)
            if error:
                return (409 if error.startswith('locked') else 400), error
            if phase == 'roster' and any('identity' in (mine.get(i['item']) or {}) for i in view['items']
                                         if i['part'] == 'vision'):
                return 409, 'locked: the roster of this recording is fixed once a frame of it is answered'
            if phase == 'identity' and any(i['part'] == 'roster' and i.get('render') == 'ok'
                                           and 'roster' not in (mine.get(i['item']) or {}) for i in view['items']):
                return 409, 'answer the roster of this recording first'
            record['answer'] = answer
            return None

        record = {'audit_id': self.audit['audit_id'], 'item': item_id, 'part': item['part'], 'phase': phase,
                  'session': sid, 'alias': alias, 'auditor': name, 'token_id': self.identity.token_id, 'subset': subset,
                  'mode': self.audit['mode'], 'codebook': A.CODEBOOK_VERSION, 'practice': bool(item.get('practice')),
                  'reliability': bool(item.get('reliability')), 'answer': None}
        if self.open:
            # saved under the typed name: no link, the scope as chosen
            record.update(open=True, scope='reliability' if subset == 'reliability' else 'full')
        spent = _number(body.get('seconds_spent'))
        if spent is not None:
            record['seconds_spent'] = round(min(max(spent, 0.0), L.MAX_SECONDS), 3)
        answered = body.get('answered_at')
        if isinstance(answered, str) and len(answered) <= 40 and C.parse_time(answered) is not None:
            record['answered_at'] = answered
        record['saved_at'] = C.now_utc()
        path = self._answers_path(alias, name)
        self.append_record(path, path.relative_to(self.artifacts).as_posix(), record, check=check)


def _flag_and_note(raw: dict) -> tuple[dict | None, str | None]:
    flag = raw.get('flag', False)
    if not isinstance(flag, bool):
        return None, 'flag is not true or false'
    note = raw.get('note', '')
    if note is None:
        note = ''
    if not isinstance(note, str) or len(note) > MAX_NOTE:
        return None, f'the note is not text of at most {MAX_NOTE} characters'
    return {'flag': flag, 'note': note}, None


def clean_answer(phase: str, raw, item: dict, view: dict, own: dict, asked) -> tuple[dict | None, str | None]:
    """(the answer as saved, None) or (None, why it is refused): only the values the item has, nothing more"""
    if not isinstance(raw, dict):
        return None, 'the answer is not an object'
    common, error = _flag_and_note(raw)
    if error:
        return None, error
    if phase == 'roster':
        if raw.get('wears') not in WEARS:
            return None, 'wears is yes, no or cannot_tell'
        return {'wears': raw['wears']}, None
    if phase == 'speech':
        if raw.get('speaker') not in SPEAKERS:
            return None, f"speaker is one of {', '.join(SPEAKERS)}"
        out = {'speaker': raw['speaker'], **common}
        played = _number(raw.get('played_seconds'))
        if played is not None:
            out['played_seconds'] = round(min(max(played, 0.0), L.MAX_SECONDS), 2)
        return out, None
    boxes = raw.get('boxes') or {}
    if not isinstance(boxes, dict):
        return None, 'boxes is not an object'
    numbers = set(item.get('boxes') or {})
    if phase == 'identity':
        if 'identity' in own:
            return None, 'locked: the identity answers of this frame are saved'
        letters = set(view['pupils'])
        unknown = sorted(set(boxes) - numbers, key=str)
        if unknown:
            return None, f'the frame has no box {str(unknown[0])[:8]}'
        wrong = next((n for n, v in boxes.items() if v not in letters and v not in BOX_VALUES), None)
        if wrong is not None:
            return None, f'box {wrong}: give a pupil of the recording, outside, not_person or cannot_tell'
        missed = raw.get('missed')
        if not common['flag']:
            if set(boxes) != numbers:
                return None, 'answer every box'
            if isinstance(missed, bool) or not isinstance(missed, int) or not 0 <= missed <= max(1, int(view['group_size'])):
                return None, f"missed is a count from 0 to {view['group_size']}"
        elif missed is not None and (isinstance(missed, bool) or not isinstance(missed, int) or not 0 <= missed <= 3):
            return None, 'missed is a count'
        return {'boxes': dict(boxes), 'missed': missed, **common}, None
    # gaze
    identity = own.get('identity')
    if identity is None:
        return None, 'answer who is in the boxes first'
    if (identity.get('answer') or {}).get('flag'):
        return None, 'this frame was flagged: it takes no gaze answers'
    wanted = set(asked(item, identity))
    if set(boxes) != wanted:
        return None, 'answer the gaze of every box asked about'
    out = {}
    for number, value in boxes.items():
        if not isinstance(value, dict) or value.get('class') not in GAZE_VALUES + ('between', 'cannot_tell'):
            return None, f'box {number}: give a gaze class, between or cannot_tell'
        entry = {'class': value['class']}
        if value['class'] == 'between':
            pair = value.get('between')
            if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1] \
                    or any(p not in GAZE_VALUES for p in pair):
                return None, f'box {number}: between names two different gaze classes'
            entry['between'] = list(pair)
        out[number] = entry
    return {'boxes': out, **common}, None


def cmd_serve(args, argv) -> int:
    artifacts, audit_id = A._artifacts(args), args.audit
    folder = A.audit_dir(artifacts, audit_id)
    sweeps = getattr(args, 'audit_sweeps', None)
    base, modules, task = AuditHandler, AUDIT_MODULES, {}
    if A.load_plan(artifacts, audit_id).get('task') == 'transcript':
        # a transcription audit (audit_speech) has routes and a page of its own, and serves the sweeps asked for
        from openmmla.commands.ses import audit_speech as SP
        from openmmla.commands.ses import audit_transcript_page as TP
        audit = TP.load_transcript_audit(artifacts, audit_id, sweeps)
        # the page acts on blind_closed.json: it must be the closes the operator's logged steps made
        problems = SP.close_problems(SP.read_blind_closed(artifacts, audit_id), SP.logged_closes(artifacts, audit_id)[0])
        if problems:
            raise A.AuditError('; '.join(problems) + f' (--audit-close-blind {audit_id} --audit-auditor NAME logs a '
                               'close the file holds unlogged; put back by hand what the log holds)')
        base, modules, task = TP.TranscriptHandler, TRANSCRIPT_MODULES, {'task': audit['task'], 'sweeps': audit['sweeps']}
    elif sweeps is not None:
        raise A.AuditError(f'{audit_id} is a sensing audit: --audit-sweeps serves the sweeps of a transcription audit')
    else:
        audit = load_audit(artifacts, audit_id)
    # the links are the audit's own, issued in its folder: --campaign may name it, never another
    if args.campaign and Path(args.campaign).expanduser().resolve() != folder.resolve():
        raise A.AuditError(f"an audit's links are issued in its own folder: --campaign {folder}")
    opened = bool(getattr(args, 'audit_open', None))
    if not (folder / L.CAMPAIGN_FILE).exists():
        why = 'it holds whether the audit is closed (--audit-sample writes it)' if opened else 'the links are issued there'
        raise A.AuditError(f'no campaign.yml in {folder}: {why}')
    campaign = L.Campaign(folder)
    data = campaign.data()
    if data.get('closed_at'):
        raise A.AuditError(f"the audit was closed at {data['closed_at']}: it is not served again")
    # open, the auditors type their names: no link is needed, and links issued are not served
    if not opened and not any(t.get('scope') == 'audit' and not t.get('revoked_at') for c in data['coders']
                              for t in c.get('tokens', [])):
        raise A.AuditError(f'no audit link is issued: --campaign {folder} --issue-token NAME --token-scope audit '
                           '(or --audit-open: the auditors type their names)')
    typed = 0 if opened else saved_open(artifacts, audit)
    if typed:
        raise A.AuditError(f'{typed} answers of audit {audit_id} were typed on the open page: serve it with --audit-open '
                           '(a link would show the answers typed under its name as its own)')
    allow = (open_bind if opened else L.check_bind)(args.bind, args.allow_from, args.allow_wide)
    port = args.port if L._explicit_port(argv) else DEFAULT_PORT
    others = L.other_instances(artifacts)
    if others and not args.i_know_another_instance_runs:
        listed = '; '.join(f"pid {o['pid']}: {o['command'][:120]}" for o in others)
        raise A.AuditError(f'another ses-code may serve {artifacts} ({listed}): limit the auditors to this port, then '
                           'give --i-know-another-instance-runs')
    unrendered = sum(1 for _, item in audit['items'].values() if item.get('render') != 'ok')
    if unrendered == len(audit['items']):
        raise A.AuditError(f'nothing of audit {audit_id} is rendered (--audit-render {audit_id})')
    log = L.RequestLog(folder / L.LOG_FILE)
    handler = L.handler_class(base, campaign=campaign, log=log, allow=allow, audit=audit, folder=folder,
                              artifacts=artifacts, cache={}, cache_lock=threading.Lock(),
                              **({'open': True} if opened else {}))
    L.watch_campaign(campaign, log)
    extra = {'audit_id': audit_id, 'plan_sha256': L.file_sha256(folder / A.PLAN_FILE), 'mode': audit['mode'],
             'campaign_sha256': L.file_sha256(campaign.path), 'another_instances': others or None,
             'allow_wide': bool(args.allow_wide), **task}
    if opened:
        # the scorer reads the audit as open from this line: names typed, not authenticated
        extra['open'] = True
    served = '' if not task else ' (sweep 1)' if task['sweeps'] == 1 else f" (sweeps 1 to {task['sweeps']})"
    print(f"audit {audit_id} ({audit['mode']}{', open: the auditors type their names' if opened else ''}): "
          f"{len(audit['sessions'])} recordings, {len(audit['items']) - unrendered} items served{served}; "
          f"http://{args.bind}:{port}/audit for {', '.join(str(n) for n in allow)} (Ctrl-C stops)")
    try:
        return L.serve(handler, args.bind, port, argv, modules, extra)
    finally:
        campaign.watch = None
        log.close()


AUDIT_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sensing audit</title>
<style>
:root{--bg:#111;--panel:#1b1b1b;--line:#333;--text:#eee;--dim:#999;--accent:#ffd400;--good:#5fd38d;--bad:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,Helvetica,Arial,sans-serif}
header{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:8px 16px;border-bottom:1px solid var(--line)}
header b{font-size:15px}#where{color:var(--dim)}#practice{color:var(--accent);font-weight:600}
main{display:flex;gap:16px;padding:12px 16px;align-items:flex-start;flex-wrap:wrap}
#left{flex:3 1 560px;min-width:0}#right{flex:1 1 300px;min-width:260px}
#stage{position:relative;display:inline-block;max-width:100%}#picture{display:block;max-width:100%;max-height:72vh}
#layer{position:absolute;left:0;top:0}.box{position:absolute;border:2px solid transparent;cursor:pointer}
.box span{position:absolute;left:0;top:-20px;background:#000c;color:#fff;font-size:12px;padding:0 4px;white-space:nowrap}
.box.current{border-color:var(--accent)}video{display:block;max-width:100%;max-height:72vh;background:#000}
#close img{display:block;height:200px;margin-top:8px;border:1px solid var(--line)}
#question{font-size:16px;margin-bottom:6px}#rule{color:var(--dim);margin-bottom:8px}
.choice{display:block;padding:2px 0}.choice kbd,#keys kbd{background:#333;border-radius:3px;padding:0 5px}
.row{padding:2px 6px;border-radius:4px;cursor:pointer}.row.current{background:#333}
#gallery{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}#gallery figure{margin:0}#gallery img{height:90px;display:block}
#gallery figcaption{color:var(--dim)}#note{width:100%;height:44px;margin-top:10px;background:var(--panel);color:var(--text);border:1px solid var(--line)}
#status{min-height:20px;margin-top:6px}#status.bad{color:var(--bad)}#status.good{color:var(--good)}#keys{color:var(--dim);margin-top:8px}
</style></head><body>
<header><b>Sensing audit</b><span id="who"></span><span id="practice"></span><span id="where"></span><span id="progress"></span></header>
<main><section id="left"><div id="stage"><img id="picture" alt="" style="display:none"><div id="layer"></div><video id="video" controls playsinline style="display:none"></video></div>
<div id="close" style="display:none"><img id="head" alt=""></div></section>
<section id="right"><div id="question"></div><div id="rule"></div><div id="choices"></div><div id="rows"></div><div id="gallery"></div>
<textarea id="note" placeholder="note"></textarea><div id="status"></div><div id="keys"></div></section></main>
<script>
const $ = id => document.getElementById(id);
// loading: a key pressed while the next item loads would act on the one before it, so it is dropped
let boot = null, auditor = null, seq = [], pos = 0, item = null, phase = null, st = null, saving = false, loading = false, shownAt = 0;
function said(text, mood) { $('status').textContent = text; $('status').className = mood || ''; }
function show(id, on) { $(id).style.display = on ? '' : 'none'; }
// the auditor is the one the personal link named: the page sends no name, and keeps none
async function api(path) {
  const r = await fetch(path); const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `status ${r.status}`); return data;
}
async function post(body) {
  const r = await fetch('/api/audit/answer', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body)});
  const data = await r.json().catch(() => ({}));
  if (!r.ok || !data.ok) throw new Error(data.error || `status ${r.status}`); return data;
}
function choiceLine(key, title) { const d = document.createElement('div'), k = document.createElement('kbd'); d.className = 'choice'; k.textContent = key; d.append(k, ' ' + title); return d; }
function setChoices(list) { $('choices').replaceChildren(...list.map(c => choiceLine(c[0], c[1]))); }
function keys(text) { $('keys').innerHTML = text; }
function gazeTitle(value) { const c = boot.codebook.gaze.classes.find(c => c.value === value); return c ? c.title : value; }
function shortAnswer(v) { return v === 'outside' ? 'not in group' : v === 'not_person' ? 'not a person' : v === 'cannot_tell' ? 'cannot tell' : v; }
function progressText() { const done = seq.filter(s => s.done).length; $('progress').textContent = `${done} of ${seq.length} answered`; }

// the numbered boxes over the picture, scaled to it as shown
function drawLayer() {
  const layer = $('layer'), pic = $('picture');
  layer.replaceChildren();
  if (!item || item.part !== 'vision' || !pic.naturalWidth) return;
  const k = pic.clientWidth / item.size[0];
  layer.style.width = pic.clientWidth + 'px'; layer.style.height = pic.clientHeight + 'px';
  const numbers = phase === 'gaze' ? st.ask : Object.keys(item.boxes);
  for (const n of numbers) {
    const [x1, y1, x2, y2] = item.boxes[n], d = document.createElement('div');
    d.className = 'box' + (st && st.current === n ? ' current' : '');
    Object.assign(d.style, {left: x1 * k + 'px', top: y1 * k + 'px', width: (x2 - x1) * k + 'px', height: (y2 - y1) * k + 'px'});
    const label = document.createElement('span');
    const given = phase === 'identity' ? st.boxes[n] : phase === 'gaze' && st.boxes[n] ? st.boxes[n].class : null;
    label.textContent = given ? `${n} · ${shortAnswer(given)}` : n;
    d.appendChild(label);
    d.addEventListener('click', () => { if (!st.locked) { st.current = n; paint(); } });
    layer.appendChild(d);
  }
}

function rowsOf(numbers, text) {
  $('rows').replaceChildren(...numbers.map(n => {
    const r = document.createElement('div'); r.className = 'row' + (st.current === n ? ' current' : '');
    r.textContent = `Box ${n}: ${text(n)}`; r.addEventListener('click', () => { if (!st.locked) { st.current = n; paint(); } }); return r;
  }));
}

function gallery() {
  const g = $('gallery'); g.replaceChildren();
  if (!item || item.part !== 'vision' || phase !== 'identity') return;
  for (const letter of item.pupils) {
    const f = document.createElement('figure'), cap = document.createElement('figcaption');
    cap.textContent = `Pupil ${letter}`; f.appendChild(cap);
    const urls = (item.gallery || {})[letter] || [];
    if (!urls.length) { const none = document.createElement('div'); none.textContent = 'no reference'; f.appendChild(none); }
    for (const u of urls.slice(0, 4)) { const img = document.createElement('img'); img.src = u; f.appendChild(img); }
    g.appendChild(f);
  }
}

function paint() {
  const cb = boot.codebook;
  $('practice').textContent = item && item.practice ? 'practice' : '';
  $('where').textContent = item ? `${item.alias} · item ${item.position + 1} of ${item.count}` : '';
  $('rule').textContent = '';
  show('video', item && item.part === 'speech'); show('picture', item && item.part !== 'speech'); show('close', phase === 'gaze');
  $('rows').replaceChildren(); $('choices').replaceChildren();
  if (!item) { $('question').textContent = 'Every item is answered. Thank you.'; keys(''); drawLayer(); gallery(); return; }
  if (item.part === 'roster') {
    $('question').textContent = `Pupil ${item.pupil}: ${cb.roster.question}`;
    setChoices(cb.roster.answers.map(a => [a.key, a.title]));
    if (st.answer) said(`saved: ${shortAnswer(st.answer)}`, 'good');
    keys('<kbd>y</kbd> <kbd>n</kbd> <kbd>x</kbd> answer · <kbd>←</kbd> <kbd>→</kbd> move');
  } else if (item.part === 'speech') {
    $('question').textContent = cb.speech.question; $('rule').textContent = cb.speech.rule;
    setChoices(cb.speech.answers.map(a => [a.key, a.title]));
    keys('<kbd>0</kbd>–<kbd>3</kbd> <kbd>x</kbd> answer · <kbd>space</kbd> replay · <kbd>f</kbd> flag · <kbd>n</kbd> note · <kbd>←</kbd> <kbd>→</kbd> move');
  } else if (phase === 'identity') {
    const names = item.pupils.map(l => [l.toLowerCase(), `pupil ${l}`]);
    const all = Object.keys(item.boxes), answered = all.every(n => st.boxes[n]);
    $('question').textContent = st.locked ? 'Who is in each box (saved, locked)' : all.length && !answered ? cb.identity.question : cb.identity.missed;
    $('rule').textContent = cb.identity.rule;
    setChoices(all.length && !answered ? names.concat(cb.identity.answers.map(a => [a.key, a.title])) :
               Array.from({length: Math.max(1, item.group_size) + 1}, (_, i) => [String(i), `${i} member${i === 1 ? '' : 's'} without a box`]));
    rowsOf(all, n => st.boxes[n] ? shortAnswer(st.boxes[n]) : '…');
    if (answered && !st.locked) { const r = document.createElement('div'); r.className = 'row'; r.textContent = `Members without a box: ${st.missed === null ? '…' : st.missed}${st.flag ? ' · flagged' : ''}`; $('rows').appendChild(r); }
    keys(st.locked ? '<kbd>←</kbd> <kbd>→</kbd> move' : `${item.pupils.map(l => `<kbd>${l.toLowerCase()}</kbd>`).join(' ')} <kbd>o</kbd> <kbd>p</kbd> <kbd>x</kbd> box · click a box or <kbd>⌫</kbd> to go back · <kbd>0</kbd>–<kbd>${item.group_size}</kbd> members without a box · <kbd>f</kbd> flag · <kbd>Enter</kbd> save and lock · <kbd>←</kbd> <kbd>→</kbd> move`);
  } else {
    $('question').textContent = st.ask.length ? cb.gaze.question : 'No gaze is asked about in this frame.';
    $('rule').textContent = cb.gaze.rule;
    setChoices(cb.gaze.classes.map(c => [c.key, c.title]).concat([[cb.gaze.between.key, cb.gaze.between.title + (st.between ? ` (${st.between.map(gazeTitle).join(' and ') || 'name the first'})` : '')], [cb.gaze.cannot_tell.key, cb.gaze.cannot_tell.title]]));
    rowsOf(st.ask, n => st.boxes[n] ? (st.boxes[n].class === 'between' ? `between ${st.boxes[n].between.map(gazeTitle).join(' and ')}` : gazeTitle(st.boxes[n].class)) : '…');
    const images = item.gaze.images[st.current] || {};
    if (images.highlight) $('picture').src = images.highlight;
    $('head').src = images.head || ''; show('close', !!images.head);
    keys('<kbd>1</kbd>–<kbd>8</kbd> where · <kbd>9</kbd> between two (then both) · <kbd>x</kbd> cannot tell · <kbd>⌫</kbd> back · <kbd>f</kbd> flag · <kbd>Enter</kbd> save · <kbd>←</kbd> <kbd>→</kbd> move');
  }
  drawLayer(); gallery(); progressText();
}

async function open(at) {
  loading = true;
  try { await load(at); } finally { loading = false; }
}
async function load(at) {
  pos = Math.max(0, Math.min(at, seq.length));
  if (pos >= seq.length) { item = null; phase = null; st = null; paint(); return; }
  try { item = await api(`/api/audit/item?item=${encodeURIComponent(seq[pos].item)}`); }
  catch (e) { said(`cannot load the item: ${e.message}`, 'bad'); return; }
  said('');
  $('note').value = '';
  if (item.part === 'roster') { phase = 'roster'; st = {answer: item.answer ? item.answer.wears : null}; $('picture').src = item.image; }
  else if (item.part === 'speech') {
    phase = 'speech'; st = {answer: item.answer ? item.answer.speaker : null, flag: !!(item.answer && item.answer.flag), ranges: []};
    if (item.answer) { $('note').value = item.answer.note || ''; said(`saved: ${item.answer.speaker}`, 'good'); }
    const v = $('video'); v.src = item.clip; v.load(); if (v.play) v.play().catch(() => {});
  } else if (!item.identity) {
    phase = 'identity'; const all = Object.keys(item.boxes);
    st = {boxes: {}, current: all[0] || null, missed: null, flag: false, locked: false};
    $('picture').src = item.image;
  } else {
    phase = 'gaze'; const g = item.gaze, prior = g.answer ? g.answer.boxes : {};
    st = {ask: g.ask, boxes: Object.assign({}, prior), current: g.ask[0] || null, between: null, flag: !!(g.answer && g.answer.flag), locked: false};
    if (g.answer) { $('note').value = g.answer.note || ''; said('saved: change any box and press Enter to save again', 'good'); }
    if (item.identity.flag) { said('this frame was flagged as broken: no gaze is asked', 'plain'); st.ask = []; }
    if (!st.ask.length) $('picture').src = item.image;
  }
  shownAt = Date.now();
  paint();
}

function played() { const v = $('video'), p = v.played; let s = 0; if (p) for (let i = 0; i < p.length; i++) s += p.end(i) - p.start(i); return s; }
function nextOpen() { const i = seq.findIndex((s, k) => k > pos && !s.done); return i < 0 ? seq.length : i; }
async function save(body, then) {
  if (saving) return; saving = true; said('saving…');
  body.item = item.item; body.seconds_spent = Math.round((Date.now() - shownAt) / 100) / 10; body.answered_at = new Date().toISOString();
  try { await post(body); } catch (e) { said(`NOT saved: ${e.message}`, 'bad'); saving = false; return; }
  saving = false; said('saved', 'good'); await then();
}

async function onKey(e) {
  if (!boot || !item && e.key !== 'ArrowLeft') return;
  if (e.target === $('note')) { if (e.key === 'Escape') $('note').blur(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey || loading) return;
  const k = e.key;
  if (k === 'ArrowLeft') { e.preventDefault(); return open(pos - 1); }
  if (k === 'ArrowRight') { e.preventDefault(); return open(pos + 1); }
  // n is the roster's "no", and a roster item takes no note
  if (k === 'n' && phase !== 'roster') { e.preventDefault(); $('note').focus(); return; }
  if (saving) return;
  const note = $('note').value;
  if (phase === 'roster') {
    const a = boot.codebook.roster.answers.find(a => a.key === k); if (!a) return;
    return save({phase: 'roster', answer: {wears: a.value}}, async () => { seq[pos].done = true; st.answer = a.value; await open(pos + 1); });
  }
  if (phase === 'speech') {
    if (k === ' ') { e.preventDefault(); const v = $('video'); v.currentTime = 0; v.play().catch(() => {}); return; }
    if (k === 'f') { st.flag = !st.flag; said(st.flag ? 'flagged: the clip is broken or cannot be judged' : 'flag removed', 'plain'); return; }
    const a = boot.codebook.speech.answers.find(a => a.key === k); if (!a) return;
    const watched = played();
    if (!st.flag && !item.answer && watched < 9.5) { said(`NOT saved: play the whole clip first (${Math.ceil(9.5 - watched)} s left)`, 'bad'); return; }
    return save({phase: 'speech', answer: {speaker: a.value, played_seconds: Math.round(watched * 10) / 10, flag: st.flag, note}},
                async () => { seq[pos].done = true; await open(nextOpen()); });
  }
  if (phase === 'identity') {
    if (st.locked) return;
    const all = Object.keys(item.boxes), answered = () => all.every(n => st.boxes[n]);
    if (k === 'f') { st.flag = !st.flag; said(st.flag ? 'flagged: the frame is broken (it can be saved as it is)' : 'flag removed', 'plain'); paint(); return; }
    if (k === 'Backspace') { e.preventDefault(); const i = all.indexOf(st.current); if (i > 0) st.current = all[i - 1]; else if (i < 0 && all.length) st.current = all[all.length - 1]; paint(); return; }
    if (k === 'Enter') {
      if (!st.flag && (!answered() || st.missed === null)) { said('NOT saved: answer every box and how many members have no box', 'bad'); return; }
      return save({phase: 'identity', answer: {boxes: st.boxes, missed: st.missed, flag: st.flag, note}},
                  async () => { await open(pos); if (item && item.gaze && !item.gaze.ask.length) { seq[pos].done = true; await open(nextOpen()); } });
    }
    const letter = item.pupils.find(l => l.toLowerCase() === k);
    const other = boot.codebook.identity.answers.find(a => a.key === k);
    if ((letter || other) && st.current && (!answered() || st.boxes[st.current])) {
      st.boxes[st.current] = letter || other.value;
      const next = all.find(n => !st.boxes[n]); st.current = next || null; paint(); return;
    }
    if (/^[0-9]$/.test(k) && answered() && Number(k) <= Math.max(1, item.group_size)) { st.missed = Number(k); paint(); return; }
    return;
  }
  if (phase === 'gaze') {
    if (k === 'f') { st.flag = !st.flag; said(st.flag ? 'flagged' : 'flag removed', 'plain'); return; }
    if (k === 'Backspace') { e.preventDefault(); const i = st.ask.indexOf(st.current); if (i > 0) st.current = st.ask[i - 1]; else if (i < 0 && st.ask.length) st.current = st.ask[st.ask.length - 1]; st.between = null; paint(); return; }
    if (k === 'Enter') {
      if (!st.ask.length) { seq[pos].done = true; return open(nextOpen()); }
      if (!st.ask.every(n => st.boxes[n])) { said('NOT saved: answer every box', 'bad'); return; }
      return save({phase: 'gaze', answer: {boxes: st.boxes, flag: st.flag, note}}, async () => { seq[pos].done = true; await open(nextOpen()); });
    }
    if (!st.current) return;
    const c = boot.codebook.gaze.classes.find(c => c.key === k);
    if (st.between) {
      if (k === 'Escape') { st.between = null; paint(); return; }
      if (!c || st.between.includes(c.value)) return;
      st.between.push(c.value);
      if (st.between.length < 2) { paint(); return; }
      st.boxes[st.current] = {class: 'between', between: st.between}; st.between = null;
    } else if (k === boot.codebook.gaze.between.key) { st.between = []; paint(); return; }
    else if (k === boot.codebook.gaze.cannot_tell.key) st.boxes[st.current] = {class: 'cannot_tell'};
    else if (c) st.boxes[st.current] = {class: c.value};
    else return;
    const next = st.ask.find(n => !st.boxes[n]); if (next) st.current = next; paint();
  }
}

async function start() {
  try { boot = await api('/api/audit/boot'); }
  catch (e) { said(`cannot start: ${e.message}`, 'bad'); return; }
  auditor = boot.auditor; seq = boot.sequence;
  $('who').textContent = `Auditing as ${auditor}${boot.subset ? ` (${boot.subset} items)` : ''}${boot.mode === 'verify' ? ' · verify mode' : ''}`;
  if (boot.closed) said('This audit is closed: answers are no longer saved.', 'bad');
  const first = seq.findIndex(s => !s.done);
  await open(first < 0 ? seq.length : first);
}

$('picture').addEventListener('load', drawLayer);
window.addEventListener('resize', drawLayer);
document.addEventListener('keydown', e => { onKey(e); });
start();
</script></body></html>
"""


def _patched(page: str, patches: list[tuple[str, str]]) -> str:
    for old, new in patches:
        if page.count(old) != 1:
            raise RuntimeError(f'the audit page changed: {old[:60]!r} is not in it once; update audit_page.OPEN_AUDIT_PAGE')
        page = page.replace(old, new)
    return page


# the open audit's page: the link's page with a name form before it, every request naming the auditor and scope
OPEN_AUDIT_PAGE = _patched(AUDIT_PAGE, [
    ('</style>',
     '#named{max-width:560px;margin:32px auto;padding:0 16px}#named p{color:var(--dim)}#named label{display:block;margin:12px 0}\n'
     '#named input,#named select{font:inherit;background:var(--panel);color:var(--text);border:1px solid var(--line);'
     'border-radius:4px;padding:6px 8px;min-width:260px}\n'
     '#named button{font:inherit;background:#2d6cdf;color:#fff;border:0;border-radius:6px;padding:8px 18px;cursor:pointer}\n'
     '#namestatus{min-height:20px;margin-top:8px;color:var(--bad)}'
     '#rename{font:inherit;background:none;border:1px solid var(--line);color:var(--dim);border-radius:4px;padding:1px 8px;cursor:pointer}\n'
     '</style>'),
    ('<header><b>Sensing audit</b><span id="who"></span>',
     '<header><b>Sensing audit</b><span id="who"></span><button id="rename" type="button" style="display:none">change name</button>'),
    ('<main><section id="left">',
     '<form id="named" style="display:none" autocomplete="off"><p>Type your name and choose what you answer: the full '
     'audit, or the reliability subset of the second auditor. Your answers are kept under this name, so type it the same '
     'way each time; this browser remembers it.</p>\n'
     '<label>Name <input id="auditor" maxlength="100" autocomplete="off" spellcheck="false"></label>\n'
     '<label>You answer <select id="scope"><option value="">choose…</option><option value="full">the full audit</option>'
     '<option value="reliability">the reliability subset</option></select></label>\n'
     '<button type="submit">Start</button><div id="namestatus"></div></form>\n'
     '<main id="work" style="display:none"><section id="left">'),
    ("// the auditor is the one the personal link named: the page sends no name, and keeps none\n"
     "async function api(path) {\n  const r = await fetch(path);",
     "// the auditor is the name typed in (an open audit): every request carries it and the scope chosen, and the\n"
     "// browser remembers both for the next visit\n"
     "let scope = null;\n"
     "function recall(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }\n"
     "function remember(key, value) { try { localStorage.setItem(key, value); } catch (e) {} }\n"
     "function named(path) { return `${path}${path.includes('?') ? '&' : '?'}auditor=${encodeURIComponent(auditor)}&scope=${encodeURIComponent(scope)}`; }\n"
     "async function api(path) {\n  const r = await fetch(named(path));"),
    ("const r = await fetch('/api/audit/answer', {", "const r = await fetch(named('/api/audit/answer'), {"),
    ("async function start() {\n"
     "  try { boot = await api('/api/audit/boot'); }\n"
     "  catch (e) { said(`cannot start: ${e.message}`, 'bad'); return; }\n",
     "// the name form, filled with what this browser remembers; Enter or Start begins\n"
     "function askName(text) {\n"
     "  if (saving) return;\n"
     "  boot = null; item = null; phase = null; st = null;\n"
     "  const v = $('video'); if (v.pause) v.pause();\n"
     "  show('work', false); show('rename', false); show('named', true);\n"
     "  $('auditor').value = auditor || recall('auditor') || ''; $('scope').value = scope || recall('auditScope') || '';\n"
     "  $('namestatus').textContent = text || ''; $('auditor').focus();\n"
     "}\n"
     "async function start() {\n"
     "  const name = $('auditor').value.trim(), chosen = $('scope').value;\n"
     "  if (!name) { $('namestatus').textContent = 'type your name first'; return; }\n"
     "  if (chosen !== 'full' && chosen !== 'reliability') { $('namestatus').textContent = 'choose the full audit or the reliability subset'; return; }\n"
     "  auditor = name; scope = chosen; $('namestatus').textContent = 'starting…';\n"
     "  try { boot = await api('/api/audit/boot'); }\n"
     "  catch (e) { boot = null; $('namestatus').textContent = `cannot start: ${e.message}`; return; }\n"
     "  remember('auditor', auditor); remember('auditScope', scope);\n"
     "  $('namestatus').textContent = ''; $('auditor').blur();\n"
     "  show('named', false); show('work', true); show('rename', true);\n"),
    ("document.addEventListener('keydown', e => { onKey(e); });\nstart();\n",
     "document.addEventListener('keydown', e => { onKey(e); });\n"
     "$('named').addEventListener('submit', e => { e.preventDefault(); start(); });\n"
     "$('rename').addEventListener('click', () => askName());\n"
     "askName();\n"),
])
