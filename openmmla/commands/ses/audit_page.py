"""The page of a sensing audit (mmla ses-code --audit ID): one auditor at a time answers the roster,
identity, gaze and who-speaks questions of audit.py, on a server of its own (port 8766 unless -p, as a
campaign's).

The server is code_locked's guarded server, and an auditor is always someone a one-time link of scope
audit names: the links are issued in the audit's own folder (--campaign <that folder> --issue-token
NAME --token-scope audit, --token-subset reliability for a second auditor who answers the reliability
subset only), whether the server binds 127.0.0.1 (behind an SSH forward) or the tailnet address with
--allow-from. No answer is saved under a name the page gives, so no auditor reads or answers as
another, and the scorer takes only the answers of claimed audit links. Every request is logged in the
audit's hash-chained request log, and every answer is appended with its log line (append_record), so
--verify-log checks the answers files both ways. A closed audit is not served, and its pictures and
clips are refused (410) by a server still running.

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
"""
from __future__ import annotations

import json
import threading
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


def _first(query: dict, name: str) -> str | None:
    values = query.get(name)
    return values[0] if values else None


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
    """the audit's page and API, for the auditor a link binds"""
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

    # who ----

    def _auditor(self) -> tuple[str, str | None]:
        """(the auditor's name, their subset): always the link's (GuardedHandler answers 401 without one)"""
        return self.identity.name, self.identity.subset

    def _answers(self, name: str) -> dict[str, dict[str, dict]]:
        """the auditor's answers, item -> phase -> the record that counts (an identity's first line, any
        other phase's last), from every session's answers file"""
        out: dict[str, dict[str, dict]] = {}
        for alias in self.audit['order']:
            path = self._answers_path(alias, name)
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            stamp = (stat.st_mtime_ns, stat.st_size)
            with self.cache_lock:
                cached = self.cache.get(path)
            if cached is None or cached[0] != stamp:
                parsed: dict[str, dict[str, dict]] = {}
                for line in path.read_text(encoding='utf-8').splitlines():
                    try:
                        record = json.loads(line)
                        phases = parsed.setdefault(record['item'], {})
                        if record['phase'] != 'identity' or 'identity' not in phases:
                            phases[record['phase']] = record
                    except (ValueError, KeyError, TypeError):
                        continue
                cached = (stamp, parsed)
                with self.cache_lock:
                    self.cache[path] = cached
            for item, phases in cached[1].items():
                out.setdefault(item, {}).update(phases)
        return out

    def _answers_path(self, alias: str, name: str) -> Path:
        sid = self.audit['sessions'][alias]['id']
        return A.answers_dir(self.artifacts, sid, self.audit['audit_id']) / f'{C.safe_name(name)}.jsonl'

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
        self.send_html(AUDIT_PAGE)

    def _boot(self, request: L.Request) -> None:
        name, subset = self._auditor()
        mine = self._answers(name)
        rows = []
        for item_id in self._visible(subset):
            alias, item = self.audit['items'][item_id]
            rows.append({'item': item_id, 'part': item['part'], 'alias': alias, 'practice': bool(item.get('practice')),
                         'done': self._done(item, mine.get(item_id, {}))})
        self.send_json({'audit_id': self.audit['audit_id'], 'mode': self.audit['mode'], 'codebook': AUDIT_CODEBOOK,
                        'closed': self.closed(), 'auditor': name, 'subset': subset, 'sequence': rows})

    def _progress(self, request: L.Request) -> None:
        name, subset = self._auditor()
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
        return f'/audit/img/{item_id}/{name}' if name else None

    def _item(self, request: L.Request) -> None:
        name, subset = self._auditor()
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
            out.update(clip=f'/audit/clip/{item_id}', answer=(own.get('speech') or {}).get('answer'))
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
        name, subset = self._auditor()
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
        name, subset = self._auditor()
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
    audit = load_audit(artifacts, audit_id)
    # the links are the audit's own, issued in its folder: --campaign may name it, never another
    if args.campaign and Path(args.campaign).expanduser().resolve() != folder.resolve():
        raise A.AuditError(f"an audit's links are issued in its own folder: --campaign {folder}")
    if not (folder / L.CAMPAIGN_FILE).exists():
        raise A.AuditError(f'no campaign.yml in {folder}: the links are issued there')
    campaign = L.Campaign(folder)
    data = campaign.data()
    if data.get('closed_at'):
        raise A.AuditError(f"the audit was closed at {data['closed_at']}: it is not served again")
    if not any(t.get('scope') == 'audit' and not t.get('revoked_at') for c in data['coders'] for t in c.get('tokens', [])):
        raise A.AuditError(f'no audit link is issued: --campaign {folder} --issue-token NAME --token-scope audit')
    allow = L.check_bind(args.bind, args.allow_from, args.allow_wide)
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
    handler = L.handler_class(AuditHandler, campaign=campaign, log=log, allow=allow, audit=audit, folder=folder,
                              artifacts=artifacts, cache={}, cache_lock=threading.Lock())
    L.watch_campaign(campaign, log)
    extra = {'audit_id': audit_id, 'plan_sha256': L.file_sha256(folder / A.PLAN_FILE), 'mode': audit['mode'],
             'campaign_sha256': L.file_sha256(campaign.path), 'another_instances': others or None,
             'allow_wide': bool(args.allow_wide)}
    print(f"audit {audit_id} ({audit['mode']}): {len(audit['sessions'])} recordings, {len(audit['items']) - unrendered} "
          f"items served; http://{args.bind}:{port}/audit for {', '.join(str(n) for n in allow)} (Ctrl-C stops)")
    try:
        return L.serve(handler, args.bind, port, argv, AUDIT_MODULES, extra)
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
