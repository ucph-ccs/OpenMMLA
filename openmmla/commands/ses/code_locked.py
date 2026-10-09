"""Locked coding campaigns for mmla ses-code: blind coding by named coders over one-time personal links.

A campaign is a folder (--campaign DIR, best under artifacts/runtime/coding/<name>/; its name names the
labels folder) that lists its sessions, its coders and, optionally, the windows each coder codes. Its
server (--locked) shows a coder only their own labels: no other coder's, model's or adjudicated labels,
notes or counts reach any answer, the agreement routes answer as an unknown path does, the sessions are
named by aliases (R01, R02 ...) and every time the page sees is an offset into the recording. Labels go
to artifacts/<session>/labels/locked/<campaign>/<coder>.jsonl, which neither the default page nor
labels.load_labels reads (both read labels/*.jsonl only), until --release-campaign copies them to
labels/ after --close-campaign.

The folder holds:

  campaign.yml       settings, the sessions with their aliases, the coders with the sha256 of their
                     tokens and devices (written under a file lock, through a temp file and a rename)
  assignment.json    the windows each coder codes (--draw-assignment, code_assign); none: every window
  design.json        how the assignment was drawn; never read by the server
  requests.jsonl     the request log (below)
  links/<name>.txt   a coder's link, readable by its owner only, to hand over; never printed
  clips/             the campaign's clip cache (no container metadata), deleted by --close-campaign
  release.json       what --release-campaign copied where, with sha256s

Links. --issue-token writes a link http://<host>/c/<token> to links/<name>.txt. Opening it shows a
button; pressing it claims the link: the server keeps the sha256 of a new random device secret, sets
that secret as an HttpOnly cookie, and the link is spent. A second claim shows "this link was already
used" and is logged. A lost link or browser is replaced by --revoke-token and --issue-token, both
logged (issue_token and revoke_token log their own lines, so no link is made or taken back unlogged).

The request log (RequestLog) is append-only JSONL, one line per request and per operator action
(init, token issue and revoke, assignment, note, close, release, anchor). Every line carries `seq` and
`prev`, the sha256 of the bytes of the line before it, so an edited, removed or inserted line breaks
the chain. Every writer (the server's threads, the operator's commands, a restarted server) takes an
flock around reading the last line and appending, so the chain continues across processes and
restarts. A request that saved a label line carries that line's sha256 (label_sha256) and its file
(label_file). The claim of a link and every request a claimed device makes carry `device`, the first
16 hex digits of the sha256 of the device secret, and a server logs `campaign-changed` with the new
sha256 whenever campaign.yml changes under it. --verify-log checks the chain (a line a crash cut short,
chained over by the next writer, is reported apart: exit 3), both directions of the saved lines (every
label line has its request, every such request its line, in the logged order and of the same link),
and the links: a link is claimed once and only after its logged issue, every request of a link comes
from the device its logged claim set up, and none follows its logged revoke. So an operator who edits
campaign.yml to act as a coder (a device hash of their own, a claim or a revoke undone) shows in the
log. The start line records the sha256 of the handler modules as they were imported (LOADED), their
git state and whether the checkout is dirty. --anchor prints only hashes (the chain head and each coder
file's sha256) for sending to someone outside the project. --close-campaign and --release-campaign
verify the log first and refuse when it fails, unless --despite-log-failure (logged).

The viewing gate of min_view full (a class key counts once the window's clip has played) rests on what
the page reports of its playback: a convenience that keeps a coder from coding a window unseen by
accident, not a control; the viewing record each label line carries is the page's word too.

Serving refuses 0.0.0.0 and LAN addresses: it binds 127.0.0.1 (reached through an SSH forward) or the
tailnet address, and with the tailnet address only the hosts of --allow-from may connect (a network
needs --allow-wide). A closed campaign is not served. It also refuses to start while another ses-code
serves the same artifacts, unless --i-know-another-instance-runs (a convenience check: the control is
that a coder's device reaches no other port of the server).

Interface reused by the sensing audit's server (audit_page.py):

  Campaign(folder)        .data() (read again whenever campaign.yml changed; .watch(sha256) is called
                          then), .update(change), .link_state(token, kind), .claim(token, kind),
                          .authorize(secret, kind) -> Identity(name, token_id, scope, subset, device)
                          | None
  RequestLog(path)        .append(entry), .event(name, **fields), .held() (the log's lock; inside it
                          .next_seq() and .write(entry)), .head(), .sync(), .close();
                          verify_log(path, files) -> (ok, message, info)
  GuardedHandler          a mixin placed before BaseHTTPRequestHandler (or code.Handler): the
                          allow-from check, /c/<token> claims, the cookie (every other request needs
                          it), the POST checks (JSON, same origin, 64 KiB), no-store and the other
                          headers, one log line per request, 404 for any path not routed. A subclass
                          sets KIND ('code' or 'audit': the scope a token must hold), HOME, ROUTES
                          {(method, path): name} and PREFIX_ROUTES ((method, prefix, name), ...); a
                          route method takes a Request(path, query, body, rest) and answers with
                          send_json, send_html, send_file (Range) or append_record (a record line and
                          its log line, under the log's lock, after an optional check under it);
                          self.identity is the bound coder, self.note(**fields) adds to the request's
                          log line, note_query(query) says what of a query is logged.
  handler_class(base, **attributes)   a subclass of `base` with its class attributes set, one per server
  check_bind(bind, allow_from, allow_wide)   the networks that may connect, or CampaignError
  other_instances(artifacts)          other ses-code processes that may serve the same artifacts
  serve(handler_class, bind, port, argv, modules, extra)   start line, serve until stopped, stop line
  issue_token(campaign, name, scope, subset), revoke_token(campaign, name, scope)
                                      links of scope 'code' or 'audit' (an audit link may carry the
                                      subset 'reliability'), each logged
  register_loaded(name, file)         a module's sha256 as it is imported, for the start line
  checked_log(campaign, artifacts, override, step)   the log verified before a step that relies on it

The flags --campaign, --allow-from, --allow-wide, --i-know-another-instance-runs,
--despite-log-failure and --issue-token with --token-scope audit and --token-subset (add_arguments)
serve the audit as well; requested() leaves any --audit* flag to the audit.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import html
import ipaddress
import json
import math
import os
import random
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import namedtuple
from contextlib import contextmanager
from http.cookies import CookieError, SimpleCookie
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable

from openmmla.commands.ses import code as C

CAMPAIGN_FILE = 'campaign.yml'
ASSIGNMENT_FILE = 'assignment.json'
DESIGN_FILE = 'design.json'
LOG_FILE = 'requests.jsonl'
RELEASE_FILE = 'release.json'
LINKS_DIR = 'links'
CLIPS_DIR = 'clips'
LOCKED_DIR = 'locked'
CAMPAIGN_NAME = re.compile(r'[A-Za-z0-9_-]{1,40}')
# the `prev` of a log's first line
GENESIS = '0' * 64
MAX_BODY = 64 * 1024
MAX_NOTE = 2000
MAX_SECONDS = 36000.0
# with min_view: full a class key counts once this share of the window's own clip has played, at no more
# than MAX_RATE times its speed (the page holds the rate there)
MIN_COVER = 0.95
MAX_RATE = 1.25
COOKIE_DAYS = 30
DEFAULT_PORT = 8766
TAILNET = (ipaddress.ip_network('100.64.0.0/10'), ipaddress.ip_network('fd7a:115c:a1e0::/48'))
LOOPBACK = (ipaddress.ip_network('127.0.0.0/8'), ipaddress.ip_network('::1/128'))
# the modules whose code a locked server runs, hashed into its start line
LOCKED_MODULES = ('openmmla.commands.ses.code', 'openmmla.commands.ses.code_locked', 'openmmla.commands.ses.code_text')
# the subsets an audit link may carry (audit_page.sequence)
SUBSETS = ('reliability',)
# how much of the sha256 of a device secret the log keeps: enough to tell devices apart, never the secret
DEVICE_DIGITS = 16

Identity = namedtuple('Identity', 'name token_id scope subset device', defaults=(None,))
Request = namedtuple('Request', 'path query body rest')


class CampaignError(Exception):
    """a refusal, said to the operator as it is"""


def sha256_hex(data: bytes | str) -> str:
    return hashlib.sha256(data.encode('utf-8') if isinstance(data, str) else data).hexdigest()


def file_sha256(path) -> str | None:
    """the sha256 of a file's bytes, None when there is no such file"""
    try:
        digest = hashlib.sha256()
        with open(path, 'rb') as file:
            for block in iter(lambda: file.read(1 << 20), b''):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


# the sha256 of each server module's file when it was imported: in a shared checkout a file can change between
# the import and the start line, and the start line must name the code that runs
LOADED: dict[str, str | None] = {}


def register_loaded(name: str, path: str | None) -> None:
    """keep the sha256 of a module's file as it is imported (each module a server runs calls this at its import)"""
    if path and name not in LOADED:
        LOADED[name] = file_sha256(path)


for _name in ('openmmla.commands.ses.code_text', 'openmmla.commands.ses.code', __name__):
    register_loaded(_name, getattr(sys.modules.get(_name), '__file__', None))


def _write_json(path: Path, data: Any) -> None:
    """a JSON file written whole, through a temp file and a rename"""
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


# ---- the campaign file ----

class Campaign:
    """a campaign folder and its campaign.yml"""

    def __init__(self, folder):
        self.folder = Path(folder).expanduser().resolve()
        self.path = self.folder / CAMPAIGN_FILE
        self._lock = threading.Lock()
        self._data: dict | None = None
        self._stamp = None
        self._digest: str | None = None
        # called with the file's new sha256 whenever data() finds campaign.yml changed (a server logs it)
        self.watch: Callable[[str], Any] | None = None

    @property
    def name(self) -> str:
        return self.folder.name

    def exists(self) -> bool:
        return self.path.exists()

    def _parse(self, raw: bytes) -> dict:
        import yaml
        data = yaml.safe_load(raw.decode('utf-8')) or {}
        if not isinstance(data, dict):
            raise CampaignError(f'{self.path} is not a campaign file')
        data.setdefault('coders', [])
        data.setdefault('sessions', [])
        return data

    def _read(self) -> dict:
        return self._parse(self.path.read_bytes())

    def data(self) -> dict:
        """the campaign as campaign.yml holds it now, read again whenever the file changed, so a link the
        operator revokes is refused at the next request; a change after the first read is passed to
        watch() with the new file's sha256. The dict is shared: never change it."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            raise CampaignError(f'no campaign at {self.folder} (--campaign-init)') from None
        stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        changed = None
        with self._lock:
            if stamp != self._stamp or self._data is None:
                raw = self.path.read_bytes()
                self._data, self._stamp = self._parse(raw), stamp
                digest = sha256_hex(raw)
                if self._digest is not None and digest != self._digest:
                    changed = digest
                self._digest = digest
            data = self._data
        # outside the lock: the watcher writes the request log
        if changed and self.watch is not None:
            self.watch(changed)
        return data

    @contextmanager
    def _file_lock(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        with open(self.folder / 'campaign.lock', 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def update(self, change: Callable[[dict], Any], create: bool = False) -> Any:
        """change(data) on the file as it is now, under the campaign's file lock (the server's claims and the
        operator's commands both write it), written back through a temp file and a rename when it changed
        anything; returns what change returned. `create`: start from nothing when there is no file."""
        import yaml
        with self._lock, self._file_lock():
            if self.path.exists():
                data = self._read()
            elif create:
                data = {}
            else:
                raise CampaignError(f'no campaign at {self.folder} (--campaign-init)')
            before = json.dumps(data, sort_keys=True, default=str)
            result = change(data)
            if json.dumps(data, sort_keys=True, default=str) != before:
                temp = self.folder / f'.{CAMPAIGN_FILE}.{os.getpid()}.tmp'
                temp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
                os.replace(temp, self.path)
                self._stamp = None
            return result

    # tokens ----

    @staticmethod
    def _find_token(data: dict, token: str):
        """(coder, token entry) whose token hash is `token`'s, else (None, None)"""
        if not isinstance(token, str) or not 8 <= len(token) <= 100:
            return None, None
        digest = sha256_hex(token)
        found = None, None
        for coder in data.get('coders', []):
            for entry in coder.get('tokens', []):
                if hmac.compare_digest(str(entry.get('token_sha256', '')), digest):
                    found = coder, entry
        return found

    def link_state(self, token: str, kind: str) -> tuple[str, str | None, str | None]:
        """('open' | 'used' | 'invalid', the coder's name, the token id) of a link, changing nothing: invalid
        is an unknown token, a revoked one or one of another scope"""
        coder, entry = self._find_token(self.data(), token)
        if entry is None:
            return 'invalid', None, None
        if entry.get('revoked_at') or entry.get('scope') != kind:
            return 'invalid', coder['name'], entry['token_id']
        return ('used' if entry.get('claimed_at') else 'open'), coder['name'], entry['token_id']

    def claim(self, token: str, kind: str) -> tuple[str, str | None, str | None, str | None]:
        """spend a link: ('claimed', name, token id, the device secret to set as the cookie), or ('used' |
        'invalid', name, token id, None). Two claims at once are one claim and one 'used'."""
        def change(data):
            coder, entry = self._find_token(data, token)
            if entry is None:
                return 'invalid', None, None, None
            if entry.get('revoked_at') or entry.get('scope') != kind:
                return 'invalid', coder['name'], entry['token_id'], None
            if entry.get('claimed_at'):
                return 'used', coder['name'], entry['token_id'], None
            secret = secrets.token_urlsafe(32)
            entry['claimed_at'] = C.now_utc()
            entry['device_sha256'] = sha256_hex(secret)
            return 'claimed', coder['name'], entry['token_id'], secret
        return self.update(change)

    def authorize(self, secret: str | None, kind: str) -> Identity | None:
        """the coder whose claimed, unrevoked link of scope `kind` set this device secret, else None"""
        if not isinstance(secret, str) or not 8 <= len(secret) <= 200:
            return None
        digest = sha256_hex(secret)
        found = None
        for coder in self.data().get('coders', []):
            for entry in coder.get('tokens', []):
                device = entry.get('device_sha256')
                if device and hmac.compare_digest(str(device), digest) and not entry.get('revoked_at') \
                        and entry.get('scope') == kind:
                    found = Identity(coder['name'], entry['token_id'], entry.get('scope'), entry.get('subset'),
                                     digest[:DEVICE_DIGITS])
        return found


def issue_token(campaign: Campaign, name: str, scope: str = 'code', subset: str | None = None) -> tuple[str, str, str]:
    """a new link for `name` (added as a coder when new), logged in the campaign's request log: (token,
    token id, 'issue' | 'reissue'). A name may hold one unrevoked link per scope; another coder's name that
    differs only in case is refused, and so is a subset other than SUBSETS or on a coding link."""
    if subset is not None and (scope != 'audit' or subset not in SUBSETS):
        raise CampaignError(f"--token-subset is {' or '.join(SUBSETS)}, and for an audit link only")

    def change(data):
        coders = data.setdefault('coders', [])
        other = next((c for c in coders if c['name'] != name and c['name'].lower() == name.lower()), None)
        if other:
            raise CampaignError(f"{other['name']!r} is a coder of this campaign already: names may not differ in case only")
        coder = next((c for c in coders if c['name'] == name), None)
        action = 'reissue' if coder and any(t.get('scope') == scope for t in coder['tokens']) else 'issue'
        if coder is None:
            coder = {'name': name, 'tokens': []}
            coders.append(coder)
        if any(t.get('scope') == scope and not t.get('revoked_at') for t in coder['tokens']):
            raise CampaignError(f'{name} holds a {scope} link not revoked: --revoke-token {name} first')
        taken = {t['token_id'] for c in coders for t in c.get('tokens', [])}
        while True:
            token = secrets.token_urlsafe(24)
            digest = sha256_hex(token)
            if digest[:8] not in taken:
                break
        coder['tokens'].append({'token_id': digest[:8], 'token_sha256': digest, 'scope': scope, 'subset': subset,
                                'issued_at': C.now_utc(), 'claimed_at': None, 'device_sha256': None, 'revoked_at': None})
        return token, digest[:8], action
    token, token_id, action = campaign.update(change)
    _event(campaign, 'token', action=action, name=name, token_id=token_id, scope=scope, subset=subset,
           campaign_sha256=file_sha256(campaign.path))
    return token, token_id, action


def revoke_token(campaign: Campaign, name: str, scope: str | None = None) -> list[str]:
    """revoke `name`'s unrevoked links (of `scope`, or all), each logged: the token ids revoked"""
    def change(data):
        coder = next((c for c in data.get('coders', []) if c['name'] == name), None)
        if coder is None:
            raise CampaignError(f'{name} is no coder of this campaign')
        revoked = []
        for entry in coder.get('tokens', []):
            if not entry.get('revoked_at') and (scope is None or entry.get('scope') == scope):
                entry['revoked_at'] = C.now_utc()
                revoked.append(entry['token_id'])
        return revoked
    revoked = campaign.update(change)
    for token_id in revoked:
        _event(campaign, 'token', action='revoke', name=name, token_id=token_id,
               campaign_sha256=file_sha256(campaign.path))
    return revoked


def coder_id(coder: dict) -> str:
    """a coder's id outside the server: their first token's id (no name)"""
    tokens = coder.get('tokens') or [{}]
    return tokens[0].get('token_id') or 'none'


# ---- the request log ----

def _tail(fd: int, size: int) -> tuple[str, int, bool]:
    """(the sha256 of a log's last line, the last seq a line of it holds, whether the file ends inside a
    line) from its last bytes; a last line that does not parse takes the seq of the line before it"""
    if size <= 0:
        return GENESIS, -1, False
    block, data, at = 1 << 16, b'', size
    while True:
        n = min(block, at)
        at -= n
        data = os.pread(fd, n, at) + data
        cut = not data.endswith(b'\n')
        lines = (data if cut else data[:-1]).split(b'\n')
        whole = lines if at == 0 else lines[1:]  # the first line read may begin before the block
        if whole:
            seq = None
            for line in reversed(whole):
                try:
                    seq = int(json.loads(line)['seq'])
                    break
                except (ValueError, KeyError, TypeError, IndexError):
                    continue
            if seq is not None or at == 0:
                return sha256_hex(whole[-1]), -1 if seq is None else seq, cut
        if at == 0:
            return GENESIS, -1, cut


class RequestLog:
    """the hash-chained, append-only request log of a campaign or an audit; every write holds an flock"""

    def __init__(self, path, fsync_every: float = 5.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd: int | None = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
        self._lock = threading.Lock()
        self._size = -1  # the file's size when this process last read or wrote it
        self._head, self._seq, self._cut = GENESIS, -1, False
        self._every = fsync_every
        self._synced = time.monotonic()
        self._unsynced = False

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                if self._unsynced:
                    os.fsync(self._fd)
                os.close(self._fd)
                self._fd = None

    @contextmanager
    def held(self):
        """the log's lock, within this process and across processes, with its last line read: inside it
        next_seq() is the seq the next write() takes, and nothing else writes in between"""
        with self._lock:
            if self._fd is None:
                raise CampaignError(f'the request log {self.path} is closed')
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                size = os.fstat(self._fd).st_size
                if size != self._size:  # another process wrote, or this one has not read yet
                    self._head, self._seq, self._cut = _tail(self._fd, size)
                    self._size = size
                yield self
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)

    def next_seq(self) -> int:
        return self._seq + 1

    def write(self, entry: dict, sync: bool = False) -> dict:
        """inside held(): append `entry` as the next line (seq and t first, prev last); the line as written,
        with its own sha256"""
        line = {'seq': self._seq + 1, 't': C.now_utc()}
        line.update((k, v) for k, v in entry.items() if k not in ('seq', 't', 'prev'))
        line['prev'] = self._head
        data = json.dumps(line, ensure_ascii=False, separators=(',', ':'), default=str).encode('utf-8')
        out = (b'\n' if self._cut else b'') + data + b'\n'  # a line cut by a crash is ended first
        view = memoryview(out)
        while view:
            view = view[os.write(self._fd, view):]
        self._head, self._seq, self._cut = sha256_hex(data), self._seq + 1, False
        self._size = os.fstat(self._fd).st_size
        self._unsynced = True
        if sync or time.monotonic() - self._synced >= self._every:
            self._fsync()
        return {**line, 'sha256': self._head}

    def _fsync(self) -> None:
        os.fsync(self._fd)
        self._synced, self._unsynced = time.monotonic(), False

    def append(self, entry: dict, sync: bool = False) -> dict:
        with self.held():
            return self.write(entry, sync)

    def event(self, event: str, /, **fields) -> dict:
        """an operator's or a lifecycle line, synced to disk at once"""
        return self.append({'event': event, **fields}, sync=True)

    def head(self) -> tuple[str, int]:
        """(the sha256 of the last line, its seq)"""
        with self.held():
            return self._head, self._seq

    def sync(self) -> None:
        with self._lock:
            if self._fd is not None and self._unsynced:
                self._fsync()


def _parsed(line: bytes | None) -> dict | None:
    try:
        record = json.loads(line) if line is not None else None
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _links(record: dict, number: int, links: dict) -> str | None:
    """what a log line breaks of the links' rules, or None: a link is claimed once, after its logged issue
    and before any logged revoke; a request with a device comes from the device its link's logged claim set
    up, before the link's revoke. `links` (token id -> {'issued', 'claimed', 'device', 'revoked'}) is
    carried from line to line."""
    token, device = record.get('token_id'), record.get('device')
    state = links.setdefault(str(token), {'issued': None, 'claimed': None, 'device': None, 'revoked': None})
    if record.get('event') == 'token':
        if record.get('action') in ('issue', 'reissue'):
            state['issued'] = state['issued'] if state['issued'] is not None else record.get('seq')
        elif record.get('action') == 'revoke':
            state['revoked'] = record.get('seq')
        return None
    if record.get('claim') == 'claimed':
        if state['issued'] is None:
            return f'line {number}: link {token} was claimed, but no line of the log issued it'
        if state['claimed'] is not None:
            return f"line {number}: link {token} was claimed a second time (first at seq {state['claimed']})"
        if state['revoked'] is not None:
            return f'line {number}: link {token} was claimed after it was revoked'
        if not device:
            return f'line {number}: the claim of link {token} names no device'
        state['claimed'], state['device'] = record.get('seq'), device
        return None
    if device:
        if state['claimed'] is None or device != state['device']:
            return (f'line {number}: a request of link {token} came from a device no logged claim of the link set up '
                    f'(campaign.yml changed by hand?)')
        if state['revoked'] is not None:
            return f"line {number}: a request of link {token} after its revoke at seq {state['revoked']}"
    return None


def verify_log(path, files: Iterable[tuple[str, Path]] | None = None) -> tuple[bool, str, dict]:
    """check a request log: every line is JSON, its seq follows the line before and its prev is the sha256
    of that line's bytes (GENESIS for the first), and the links' rules hold (_links). A line that is not
    JSON passes as a crash cut when the line after it chains to it (the next writer ends a cut line and
    chains over it) or it is the last line and does not end; its place is kept in info['cuts']. With
    `files` ((label_file as logged, path), ...), also both directions of the saved records: every line of
    those files was saved by a logged request (label_sha256) of the same link (a line of an open audit,
    `open`, by a request of the auditor it names), in the order of their seqs, and every logged record is
    in its file. (ok, what was found, {'lines', 'head', 'seq', 'labels', 'cuts'})."""
    info = {'lines': 0, 'head': GENESIS, 'seq': -1, 'labels': 0, 'cuts': []}
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        return False, f'the log cannot be read: {error.strerror or error}', info
    ended = data.endswith(b'\n')
    lines = data[:-1].split(b'\n') if ended else (data.split(b'\n') if data else [])
    prev, seq, logged, links = GENESIS, -1, {}, {}
    for number, line in enumerate(lines, 1):
        record = _parsed(line)
        last = number == len(lines)
        if record is None:
            following = _parsed(lines[number]) if not last else None
            if (last and not ended) or (following is not None and following.get('prev') == sha256_hex(line)):
                info['cuts'].append(number)
                prev = sha256_hex(line)
                info.update(lines=number, head=prev)
                continue
            return False, f'line {number} is not a JSON line, and no line chains over it as over a crash cut: edited', info
        if record.get('prev') != prev:
            return False, (f'line {number} (seq {record.get("seq")}) does not chain to the line before it: a line was '
                           f'edited, removed or inserted at or before it'), info
        if record.get('seq') != seq + 1:
            return False, f'line {number} has seq {record.get("seq")} where {seq + 1} was due', info
        seq, prev = record['seq'], sha256_hex(line)
        broken = _links(record, number, links)
        if broken:
            return False, broken, info
        if record.get('label_sha256'):
            logged.setdefault(str(record.get('label_file')), {})[record['label_sha256']] = \
                (seq, record.get('token_id'), record.get('coder'))
        info.update(lines=number, head=prev, seq=seq)
        if last and not ended:
            info['cuts'].append(number)
    if files is not None:
        for rel, file in files:
            expected = logged.pop(rel, {})
            try:
                content = Path(file).read_bytes()
            except OSError as error:
                return False, f'{rel} cannot be read: {error.strerror or error}', info
            last = -1
            for number, line in enumerate(content.split(b'\n'), 1):
                if not line:
                    continue
                digest = sha256_hex(line)
                if digest not in expected:
                    return False, f'{rel} line {number} was saved by no logged request: edited or added afterwards', info
                at, token, coder = expected.pop(digest)
                if at < last:
                    return False, f'{rel} line {number} is out of the order its requests were logged in', info
                saved = _parsed(line) or {}
                if token is not None and str(saved.get('token_id')) != str(token):
                    return False, f'{rel} line {number} names link {saved.get("token_id")}, its request came from link {token}', info
                # an open audit's line (no link) names the auditor its request typed
                if saved.get('open') is True and saved.get('auditor') != coder:
                    return False, f'{rel} line {number} names another auditor than the one its request typed', info
                last = at
                info['labels'] += 1
            if expected:
                return False, (f'{rel}: {len(expected)} saved line(s) the log names are not in the file '
                               f'(first at seq {min(at for at, *_ in expected.values())})'), info
        for rel, expected in logged.items():
            if expected:
                return False, f'{rel}: the log names {len(expected)} saved line(s) of a file that is gone', info
    checked = f", {info['labels']} saved lines matched both ways" if files is not None else ''
    if info['cuts']:
        where = ', '.join(str(n) for n in info['cuts'])
        return True, (f"intact apart from {len(info['cuts'])} line(s) a crash cut short (line {where}), {info['lines']} "
                      f"lines, head {info['head']}{checked}"), info
    return True, f"intact, {info['lines']} lines, head {info['head']}{checked}", info


# ---- who may connect ----

def parse_allow(text: str | None) -> tuple:
    """the networks of --allow-from (addresses or CIDRs, comma-separated)"""
    networks = []
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        try:
            network = ipaddress.ip_network(part, strict=False)
        except ValueError:
            raise CampaignError(f'--allow-from {part}: not an address or a network') from None
        if network.prefixlen == 0:
            raise CampaignError(f'--allow-from {part} names every address: give the coding machines\' addresses')
        networks.append(network)
    return tuple(networks)


def check_bind(bind: str, allow_from: str | None, allow_wide: bool = False) -> tuple:
    """the networks whose clients may connect to a locked or audit server on `bind`. The bind must be
    loopback (then --allow-from may narrow it) or the tailnet address (then --allow-from is required, and
    names hosts: a network, which would let in every tailnet device in it, a coder's own laptop or phone
    included, needs `allow_wide`); 0.0.0.0 and LAN or public addresses are refused, since plain HTTP
    would carry clips, frames and cookies across them."""
    try:
        address = ipaddress.ip_address('127.0.0.1' if bind == 'localhost' else bind)
    except ValueError:
        raise CampaignError(f"--bind {bind}: give an address, 127.0.0.1 or this machine's tailnet address") from None
    networks = parse_allow(allow_from)
    if address.is_unspecified:
        raise CampaignError(f'--bind {bind} listens on every network: bind 127.0.0.1 (behind an SSH forward) or the tailnet address')
    if address.is_loopback:
        return networks or LOOPBACK
    if any(address.version == n.version and address in n for n in TAILNET):
        if not networks:
            raise CampaignError('a tailnet address needs --allow-from: the addresses of the machines the coders use')
        wide = [n for n in networks if n.prefixlen < n.max_prefixlen]
        if wide and not allow_wide:
            raise CampaignError(f"--allow-from {wide[0]} is a network: give the coding machines' own addresses "
                                '(--allow-wide lets a network in, recorded in the start line)')
        return networks
    raise CampaignError(f'--bind {bind} is a LAN or public address, where plain HTTP would carry clips and cookies: '
                        'bind the tailnet address or 127.0.0.1 behind an SSH forward')


def _client_allowed(host: str, allow: tuple) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if getattr(address, 'ipv4_mapped', None):
        address = address.ipv4_mapped
    return any(address.version == n.version and address in n for n in allow)


def _is_ses_code(command: str) -> bool:
    return 'ses-code' in command or 'openmmla.commands.ses.code' in command


def _process_cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(f'/proc/{pid}/cwd'))
    except OSError:
        pass
    try:
        out = subprocess.run(['lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return next((Path(line[1:]) for line in out.splitlines() if line.startswith('n')), None)


def _artifacts_of(pid: int, command: str) -> Path | None:
    """the artifacts root a ses-code process serves: its -a/--artifacts, else <its cwd>/artifacts; None when
    it cannot be told"""
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    given = None
    for i, word in enumerate(words):
        if word in ('-a', '--artifacts') and i + 1 < len(words):
            given = words[i + 1]
        elif word.startswith('--artifacts='):
            given = word.split('=', 1)[1]
    cwd = _process_cwd(pid)
    if given is not None and os.path.isabs(given):
        return Path(given).resolve()
    if cwd is None:
        return None
    return (cwd / (given or 'artifacts')).resolve()


def other_instances(artifacts: Path, table: str | None = None) -> list[dict[str, Any]]:
    """the other running ses-code processes (neither this one nor its parents) that serve `artifacts` or
    whose artifacts cannot be told: [{'pid', 'command'}]. `table` is `ps -axo pid=,ppid=,command=` (read
    here when None). A convenience check, not a control."""
    if table is None:
        try:
            table = subprocess.run(['ps', '-axo', 'pid=,ppid=,command='], capture_output=True, text=True,
                                   timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return []
    rows: dict[int, tuple[int, str]] = {}
    for line in table.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows[int(parts[0])] = (int(parts[1]), parts[2])
    own, pid = set(), os.getpid()
    while pid and pid not in own:
        own.add(pid)
        pid = rows.get(pid, (0, ''))[0]
    target = Path(artifacts).resolve()
    found = []
    for pid, (_, command) in sorted(rows.items()):
        if pid in own or not _is_ses_code(command):
            continue
        where = _artifacts_of(pid, command)
        if where is None or where == target:
            found.append({'pid': pid, 'command': command[:300]})
    return found


# ---- what the server runs ----

def module_states(names: Iterable[str]) -> dict[str, Any]:
    """the sha256 of the loaded modules `names` as they were imported (LOADED; else as the file is now, and
    `hashed` says which), whether the file changed since, their git state ('clean', 'modified', 'untracked'
    or None), the checkout's commit and whether any tracked file of it is changed: the shared worktree may
    run code its commit does not hold"""
    modules, files = {}, {}
    for name in names:
        path = getattr(sys.modules.get(name), '__file__', None)
        now = file_sha256(path) if path else None
        loaded = LOADED.get(name)
        modules[name] = {'file': path, 'sha256': loaded or now, 'hashed': 'at import' if loaded else 'at start',
                         'changed_since_import': bool(loaded and now and loaded != now), 'git': None}
        if path:
            files[name] = os.path.realpath(path)
    out = {'modules': modules, 'commit': None, 'tree_dirty': None}
    if not files:
        return out

    def git(where, *args):
        done = subprocess.run(['git', '-C', where, *args], capture_output=True, text=True, timeout=10)
        return done.stdout if done.returncode == 0 else None
    try:
        root = (git(os.path.dirname(next(iter(files.values()))), 'rev-parse', '--show-toplevel') or '').strip()
        if not root:
            return out
        root = os.path.realpath(root)
        out['commit'] = (git(root, 'rev-parse', 'HEAD') or '').strip() or None
        tracked = git(root, 'status', '--porcelain=v1', '--untracked-files=no')
        out['tree_dirty'] = None if tracked is None else bool(tracked.strip())
        # porcelain paths, and the paths given here, are relative to the checkout's root
        relative = {name: os.path.relpath(path, root) for name, path in files.items()}
        status = git(root, 'status', '--porcelain=v1', '--untracked-files=all', '--', *relative.values()) or ''
        states = {}
        for line in status.splitlines():
            if len(line) > 3:
                states[line[3:].strip().strip('"')] = 'untracked' if line.startswith('??') else 'modified'
        for name, rel in relative.items():
            modules[name]['git'] = states.get(rel, 'clean')
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def software() -> dict[str, Any]:
    try:
        from openmmla.utils.session_provenance import software_info
        return software_info(Path(__file__).resolve().parent)
    except Exception as error:  # the log line says so rather than not starting
        return {'error': f'{type(error).__name__}: {error}'}


class _Stop(Exception):
    pass


def serve(handler_cls, bind: str, port: int, argv: list[str], modules: Iterable[str] = LOCKED_MODULES,
          extra: dict | None = None, ready: Callable[[Any], None] | None = None) -> int:
    """serve `handler_cls` (a GuardedHandler with its log) until Ctrl-C or SIGTERM, between a start line
    (the flags, the software, the modules' sha256 and git state, the bind and who may connect) and a stop
    line. `ready(server)` is called once it listens."""
    server_cls = ThreadingHTTPServer
    if ':' in bind:
        server_cls = type('ThreadingHTTPServer6', (ThreadingHTTPServer,), {'address_family': socket.AF_INET6})
    server = server_cls((bind, port), handler_cls)
    log: RequestLog = handler_cls.log
    log.event('start', argv=list(argv), pid=os.getpid(), kind=handler_cls.KIND, bind=bind,
              port=server.server_address[1], allow_from=[str(n) for n in handler_cls.allow],
              software=software(), **module_states(modules), **(extra or {}))
    previous = None
    if threading.current_thread() is threading.main_thread():
        def stop(signum, frame):
            raise _Stop()
        previous = signal.signal(signal.SIGTERM, stop)
    try:
        if ready:
            ready(server)
        server.serve_forever()
    except (KeyboardInterrupt, _Stop):
        pass
    finally:
        server.server_close()
        log.event('stop')
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    return 0


# ---- the guarded handler ----

CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "media-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'none'")
NOT_VALID = 'This link is not valid. Ask for a new one.'
USED = ('This link was already used. Each link works once, in one browser. If you did not open it before, '
        'tell the person who gave it to you.')
OPEN_LINK = 'Open your personal link to use this page.'


def message_page(text: str, title: str = 'Session coding') -> str:
    return ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>body{{margin:40px 16px;font:16px/1.5 -apple-system,Helvetica,Arial,'
            'sans-serif;background:#111;color:#eee}button{font:inherit;background:#2d6cdf;color:#fff;border:0;'
            f'border-radius:6px;padding:10px 18px;cursor:pointer}}</style></head><body><p>{html.escape(text)}</p></body></html>')


def claim_page(title: str) -> str:
    """the page of an unused link: a button, so a link preview or a prefetch (GET) never spends it"""
    return message_page('This is your personal link. It works once, in one browser: press Start in the browser '
                        'you will use.', title).replace(
        '</body>', '<form method="post"><button type="submit">Start</button></form></body>')


def handler_class(base: type, **attributes) -> type:
    """a subclass of `base` with these class attributes: one per server, so two servers (or tests) never
    share them"""
    return type(base.__name__, (base,), attributes)


def _first(query: dict, name: str) -> str | None:
    values = query.get(name)
    return values[0] if values else None


class GuardedHandler:
    """the guard of a locked or audit server, as a mixin before BaseHTTPRequestHandler (see the module's
    docstring)"""
    KIND = 'code'
    HOME = '/'
    TITLE = 'Session coding'
    campaign: Campaign | None = None
    log: RequestLog | None = None
    allow: tuple = LOOPBACK
    ROUTES: dict[tuple[str, str], str] = {}
    PREFIX_ROUTES: tuple[tuple[str, str, str], ...] = ()
    server_version = 'mmla'
    sys_version = ''

    def log_message(self, format, *args):  # the request log replaces it
        pass

    @property
    def cookie_name(self) -> str:
        # cookies are not kept apart by port, so the port is in the name
        return f'mmla_{self.KIND}_{self.server.server_address[1]}'

    def send_response(self, code, message=None):
        self._status = code
        super().send_response(code, message)

    def end_headers(self):
        # no answer is kept by the browser, sniffed or framed, and no other site gets a referrer; not
        # no-referrer, under which the browser sends the page's own POSTs with Origin: null
        self.send_header('cache-control', 'no-store')
        self.send_header('x-content-type-options', 'nosniff')
        self.send_header('x-frame-options', 'DENY')
        self.send_header('referrer-policy', 'same-origin')
        super().end_headers()

    def do_GET(self):
        self._serve('GET')

    def do_POST(self):
        self._serve('POST')

    def do_HEAD(self):
        self._serve('HEAD')

    def do_PUT(self):
        self._serve('PUT')

    def do_DELETE(self):
        self._serve('DELETE')

    def do_PATCH(self):
        self._serve('PATCH')

    def do_OPTIONS(self):
        self._serve('OPTIONS')

    # answers ----

    def send_body(self, status: int, content_type: str, body: bytes, headers=()) -> None:
        self.send_response(status)
        self.send_header('content-type', content_type)
        self.send_header('content-length', str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)
        self._bytes = len(body)

    def send_json(self, data: Any, status: int = 200) -> None:
        self.send_body(status, 'application/json', json.dumps(data).encode())

    def _json(self, data: Any, status: int = 200) -> None:  # what code.Handler's methods answer with
        self.send_json(data, status)

    def send_html(self, text: str, status: int = 200) -> None:
        self.send_body(status, 'text/html; charset=utf-8', text.encode(), (('content-security-policy', CSP),))

    def send_message(self, path: str, text: str, status: int) -> None:
        """a refusal: JSON for /api/ paths, a page of one line for the others"""
        if path.startswith('/api/'):
            self.send_json({'error': text}, status)
        else:
            self.send_html(message_page(text, self.TITLE), status)

    def send_file(self, path: Path, content_type: str) -> None:
        """a file, or the one byte range a Range header asks for (206; Safari plays video only so)"""
        data = Path(path).read_bytes()
        size = len(data)
        wanted = (self.headers.get('range') or '').strip()
        if not wanted:
            return self.send_body(200, content_type, data, (('accept-ranges', 'bytes'),))
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', wanted)
        first = last = None
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                first = int(match.group(1))
                last = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
            else:
                first, last = max(0, size - int(match.group(2))), size - 1
        if first is None or first > last or first >= size:
            return self.send_body(416, 'application/json', b'{"error": "range not satisfiable"}',
                                  (('content-range', f'bytes */{size}'),))
        self.send_body(206, content_type, data[first:last + 1],
                       (('accept-ranges', 'bytes'), ('content-range', f'bytes {first}-{last}/{size}')))

    def note(self, **fields) -> None:
        """fields for the request's log line"""
        self._fields.update(fields)

    def closed(self) -> bool:
        """whether the campaign (or audit) was closed: then no clip, picture or transcript is served"""
        return bool(self.campaign is not None and self.campaign.data().get('closed_at'))

    def note_query(self, query: dict) -> None:
        """what of a query the log keeps: the session, start and item as given (cut short) and the
        prefetch and context flags; a subclass resolves them"""
        for name in ('session', 'start', 'item'):
            value = _first(query, name)
            if value is not None:
                self._fields[name] = value[:64]
        for name in ('prefetch', 'context'):
            if _first(query, name) not in (None, '', '0'):
                self._fields[name] = True

    def _entry(self) -> dict:
        name, token_id = self._who
        agent = (self.headers.get('user-agent') or '') if self.headers else ''
        return {'coder': name, 'token_id': token_id, 'ip': self.client_address[0], 'ua': agent[:200],
                'method': self._method, 'path': self._shown, **self._fields, 'status': self._status,
                'bytes': self._bytes, 'ms': round((time.monotonic() - self._started) * 1000, 1)}

    def append_record(self, path: Path, label_file: str, record: dict, answer: Any = None,
                      check: Callable[[dict], tuple[int, str] | None] | None = None) -> None:
        """append `record` (its request_seq set here) as one line of `path` and log the request with that
        line's sha256 and `label_file` (the file as --verify-log finds it), both under the log's lock and
        both synced to disk, then answer `answer` (default {"ok": true}), or 500 when the line cannot be
        written. `check(record)`, called under the lock before anything is written, sees every record saved
        before (two requests at once are checked one after the other); it may complete `record`, or refuse
        it with (status, error), which is answered and logged, and nothing is appended."""
        with self.log.held():
            refused = check(record) if check is not None else None
            if refused:
                status, reply = refused[0], {'error': refused[1]}
            else:
                record['request_seq'] = self.log.next_seq()
                line = json.dumps(record, ensure_ascii=False).encode('utf-8')
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('a+b') as file:
                        end = file.seek(0, os.SEEK_END)
                        if end:  # a line cut short by a crash is ended first
                            file.seek(end - 1)
                            if file.read(1) != b'\n':
                                file.write(b'\n')
                        file.write(line + b'\n')
                        file.flush()
                        os.fsync(file.fileno())
                    status, reply = 200, ({'ok': True} if answer is None else answer)
                    self.note(label_file=label_file, label_sha256=sha256_hex(line))
                except OSError as error:
                    status, reply = 500, {'error': f'the record cannot be written: {error.strerror or "file error"}'}
            body = json.dumps(reply).encode()
            self._status, self._bytes = status, len(body)
            # synced at once: a saved line must not outlive its log line in a power cut
            self.log.write(self._entry(), sync=True)
            self._logged = True
        self.send_body(status, 'application/json', body)

    # the request ----

    def _route(self, method: str, path: str) -> tuple[str, str] | None:
        name = self.ROUTES.get((method, path))
        if name:
            return name, ''
        for verb, prefix, name in self.PREFIX_ROUTES:
            if verb == method and path.startswith(prefix) and len(path) > len(prefix):
                return name, path[len(prefix):]
        return None

    def _origin_ok(self) -> bool:
        origin = self.headers.get('origin')
        return origin is None or origin == f"http://{self.headers.get('host', '')}"

    def _identity(self) -> Identity | None:
        raw = self.headers.get('cookie')
        if not raw or self.campaign is None:
            return None
        jar = SimpleCookie()
        try:
            jar.load(raw)
        except CookieError:
            return None
        morsel = jar.get(self.cookie_name)
        return self.campaign.authorize(morsel.value, self.KIND) if morsel else None

    def _read_body(self, json_only: bool = True):
        """the POST's JSON body (a dict), or None after answering why not"""
        if not self._origin_ok():
            self.send_message(urllib.parse.urlsplit(self.path).path, 'Refused: the request came from another page.', 403)
            return None
        kind = (self.headers.get('content-type') or '').split(';')[0].strip().lower()
        try:
            length = int(self.headers.get('content-length') or 0)
        except ValueError:
            length = -1
        if length < 0:
            self.send_json({'error': 'give a content-length'}, 400)
            return None
        if length > MAX_BODY:
            self.close_connection = True
            left = min(length, 16 * MAX_BODY)  # read and dropped, so the client hears the answer
            while left > 0:
                chunk = self.rfile.read(min(left, MAX_BODY))
                if not chunk:
                    break
                left -= len(chunk)
            self.send_json({'error': f'the body is over {MAX_BODY // 1024} KiB'}, 413)
            return None
        raw = self.rfile.read(length) if length else b''
        if not json_only:
            return {}
        if kind != 'application/json':
            self.send_json({'error': 'send application/json'}, 415)
            return None
        try:
            body = json.loads(raw or b'{}')
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self.send_json({'error': 'the body is not a JSON object'}, 400)
            return None
        return body

    def _claim(self, method: str, token: str) -> None:
        """/c/<token>: GET shows the Start button of an unused link, POST spends it for this browser"""
        if self.campaign is None:
            return self.send_html(message_page(NOT_VALID, self.TITLE), 401)
        state, name, token_id = self.campaign.link_state(token, self.KIND)
        self._who = (name, token_id)
        if state == 'used':
            current = self._identity()
            if current is not None and current.token_id == token_id:  # this browser's own link, opened again
                self.note(claim='reopened', device=current.device)
                return self.send_body(303, 'text/plain', b'', (('location', self.HOME),))
        if method not in ('GET', 'POST'):
            return self.send_json({'error': 'not found'}, 404)
        if state != 'open':
            self.note(claim=state)
            return self.send_html(message_page(USED if state == 'used' else NOT_VALID, self.TITLE),
                                  410 if state == 'used' else 401)
        if method == 'GET':
            self.note(claim='shown')
            return self.send_html(claim_page(self.TITLE))
        if self._read_body(json_only=False) is None:
            self.note(claim='refused')
            return
        outcome, name, token_id, secret = self.campaign.claim(token, self.KIND)
        self._who = (name, token_id)
        self.note(claim=outcome)
        if secret:
            # the device the link now belongs to: every later request of the link names the same one
            self.note(device=sha256_hex(secret)[:DEVICE_DIGITS])
        if outcome != 'claimed':
            return self.send_html(message_page(USED if outcome == 'used' else NOT_VALID, self.TITLE),
                                  410 if outcome == 'used' else 401)
        days = (self.campaign.data().get('cookie_days') or COOKIE_DAYS)
        cookie = f'{self.cookie_name}={secret}; HttpOnly; SameSite=Strict; Path=/; Max-Age={int(days * 86400)}'
        self.send_body(303, 'text/plain', b'', (('location', self.HOME), ('set-cookie', cookie)))

    def _serve(self, method: str) -> None:
        self._started = time.monotonic()
        self._status, self._bytes, self._logged = None, 0, False
        self._fields: dict[str, Any] = {}
        self._who: tuple[str | None, str | None] = (None, None)
        self.identity: Identity | None = None
        self._method = method
        url = urllib.parse.urlsplit(self.path)
        # a token is never logged, also when its request is refused
        self._shown = '/c/*' if url.path.startswith('/c/') else url.path[:200]
        try:
            if not _client_allowed(self.client_address[0], self.allow):
                self.note(refused='address')
                return self.send_message(url.path, 'This address may not open this page.', 403)
            if url.path.startswith('/c/'):
                return self._claim(method, url.path[3:])
            # every other request needs a claimed link's cookie (a server without a campaign serves nothing)
            self.identity = self._identity()
            if self.identity is None:
                return self.send_message(url.path, OPEN_LINK, 401)
            self._who = (self.identity.name, self.identity.token_id)
            self.note(device=self.identity.device)
            query = urllib.parse.parse_qs(url.query)
            self.note_query(query)
            route = self._route(method, url.path)
            if route is None:
                return self.send_json({'error': 'not found'}, 404)
            body = None
            if method == 'POST':
                body = self._read_body()
                if body is None:
                    return
            getattr(self, route[0])(Request(url.path, query, body, route[1]))
        except (BrokenPipeError, ConnectionResetError):
            self.note(error='the client went away')
        except Exception as error:  # a request that fails is logged and answered without its details
            self.note(error=type(error).__name__)
            if self._status is None:
                try:
                    self.send_json({'error': 'the server failed'}, 500)
                except OSError:
                    pass
            else:
                self._status = 500 if self._status < 400 else self._status
        finally:
            if not self._logged and self.log is not None:
                try:
                    self.log.append(self._entry())
                except Exception as error:  # the answer has gone; the operator sees it
                    print(f'the request log cannot be written: {type(error).__name__}: {error}', file=sys.stderr)


# ---- the locked coding server ----

class Plan(namedtuple('Plan', 'starts offsets by_offset by_abs context')):
    """a coder's windows of one session: the absolute starts and their offsets from the session's start, in
    time order; offset key -> start, start key -> offset, and the preceding windows' offset key -> start"""

    def context_at(self, offset: float) -> float | None:
        # the page computes w.start - j * step itself: a key off by a millisecond is the same window
        for delta in (0.0, -0.001, 0.001):
            found = self.context.get(f'{offset + delta:.3f}')
            if found is not None:
                return found
        return None


def _key(value: float) -> str:
    return f'{float(value):.3f}'


def build_plans(sessions: list[dict], assignment: dict | None, window: float, step: float, prior: int,
                seed: int = 1) -> tuple[dict[str, dict[str, Plan]], dict[str, str]]:
    """({set name: {session id: Plan}}, {coder or '*': set name}) from the assignment, or every window of
    every session when there is none; an assigned window off the sessions' grid is refused"""
    grids = {s['id']: [w['start'] for w in C.windows_of(s, window, step, 1.0, 300.0, seed)] for s in sessions}
    if assignment is None:
        sets = {'all': {sid: list(grid) for sid, grid in grids.items()}}
        coders = {'*': 'all'}
    else:
        if float(assignment.get('window', window)) != float(window):
            raise CampaignError(f"the assignment was drawn for {assignment.get('window')} s windows, the campaign has {window}")
        sets = {}
        for name, entries in (assignment.get('sets') or {}).items():
            by_session: dict[str, set] = {}
            for entry in entries:
                sid = entry.get('session')
                if sid not in grids:
                    raise CampaignError(f'the assignment names a session that is not in the campaign: {sid}')
                by_session.setdefault(sid, set()).add(_key(entry['window_start']))
            sets[name] = {}
            for sid, keys in by_session.items():
                on_grid = {_key(s): s for s in grids[sid]}
                off = sorted(keys - set(on_grid))
                if off:
                    raise CampaignError(f'{len(off)} assigned window(s) of {sid} are not on its grid (first {off[0]})')
                sets[name][sid] = sorted(on_grid[k] for k in keys)
        coders = dict(assignment.get('coders') or {'*': next(iter(sets), 'main')})
        missing = sorted(set(coders.values()) - set(sets))
        if missing:
            raise CampaignError(f'the assignment gives coders a set it does not hold: {missing[0]}')
    plans = {}
    for name, chosen in sets.items():
        plans[name] = {}
        for session in sessions:
            starts = chosen.get(session['id'], [])
            if not starts:
                continue
            grid, t0 = grids[session['id']], session['start']
            at = {_key(s): i for i, s in enumerate(grid)}
            offsets = [round(s - t0, 3) for s in starts]
            context = {}
            for s in starts:
                i = at[_key(s)]
                for j in range(1, prior + 1):
                    if i - j >= 0:
                        context[_key(round(grid[i - j] - t0, 3))] = grid[i - j]
            plans[name][session['id']] = Plan(starts, offsets, {_key(o): s for o, s in zip(offsets, starts)},
                                              {_key(s): o for o, s in zip(offsets, starts)}, context)
    return plans, coders


def clip_file(folder: Path, alias: str, offset: float, window: float) -> Path:
    return folder / alias / f'{offset:.3f}_{window:g}s.mp4'


def campaign_clip(session: dict, start: float, window: float, out: Path, width: int = 640) -> Path:
    """the window's clip as code.make_clip cuts it (the cameras side by side, the microphone), into the
    campaign's own cache, without the recordings' container metadata (dates, devices), on two threads"""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-v', 'error', '-y']
    for video in session['videos']:
        command += ['-ss', f"{start - video['start_time']:.3f}", '-t', f'{window:.3f}', '-i', video['path']]
    audio = session.get('audio')
    if audio:
        command += ['-ss', f"{start - audio['start_time']:.3f}", '-t', f'{window:.3f}', '-i', audio['path']]
    n = len(session['videos'])
    scaled = ''.join(f'[{i}:v]scale={width}:-2,setsar=1[v{i}];' for i in range(n))
    stack = ''.join(f'[v{i}]' for i in range(n)) + (f'hstack=inputs={n}[out]' if n > 1 else 'copy[out]')
    command += ['-filter_complex', scaled + stack, '-map', '[out]']
    if audio:
        command += ['-map', f'{n}:a', '-c:a', 'aac', '-b:a', '96k']
    temp = out.with_name(f'{out.stem}.{os.getpid()}.tmp.mp4')
    command += ['-map_metadata', '-1', '-threads', '2', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temp)]
    try:
        subprocess.run(command, check=True, timeout=300, capture_output=True)
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)
    return out


# the fields of a label line the page may send; the server sets the others
LABEL_FIELDS = ('label', 'key', 'secondary', 'secondary_key', 'note', 'coded_at', 'edited_at', 'seconds_spent',
                'played_seconds', 'clip_ended', 'playback_rate', 'replays', 'context_played')
SERVER_FIELDS = ('session', 'window_start', 'window_end', 'coder', 'campaign', 'token_id', 'request_seq', 'saved_at',
                 'undone_at')
COUNTS = ('replays', 'context_played')
SAFE_NOTES = ('translation loading: the English follows once the model is ready',
              'from the cache: the transcripts cannot be read now', 'transcripts are not served by this page')


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def clean_record(body: dict) -> tuple[dict | None, list[str], str | None]:
    """(the fields of a label line kept from a page's body, the names of the fields dropped, why it is
    refused or None). The key and secondary_key are the codebook's for the codes given."""
    dropped = sorted(k for k in body if k not in LABEL_FIELDS and k not in SERVER_FIELDS)
    keys = {c['label']: c['key'] for c in C.CODEBOOK['classes']}
    label = body.get('label')
    if label not in C.CODES:
        return None, dropped, f'unknown label {str(label)[:40]!r}'
    fields: dict[str, Any] = {'label': label, 'key': keys[label]}
    secondary = body.get('secondary')
    if secondary is not None:
        error = C.secondary_error({'label': label, 'secondary': secondary})
        if error:
            return None, dropped, error
        fields.update(secondary=secondary, secondary_key=keys[secondary])
    note = body.get('note')
    if note is None:
        note = ''
    if not isinstance(note, str):
        return None, dropped, 'the note is not text'
    if len(note) > MAX_NOTE:
        return None, dropped, f'the note is longer than {MAX_NOTE} characters'
    fields['note'] = note
    for name in ('coded_at', 'edited_at'):
        value = body.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or len(value) > 40 or C.parse_time(value) is None:
            return None, dropped, f'{name} is not a time'
        fields[name] = value
    for name in ('seconds_spent', 'played_seconds') + COUNTS:
        if body.get(name) is None:
            continue
        number = _number(body[name]) if not isinstance(body[name], str) else None
        if number is None:
            return None, dropped, f'{name} is not a number'
        number = min(max(number, 0.0), MAX_SECONDS)
        fields[name] = int(number) if name in COUNTS else round(number, 3)
    if body.get('clip_ended') is not None:
        if not isinstance(body['clip_ended'], bool):
            return None, dropped, 'clip_ended is not true or false'
        fields['clip_ended'] = body['clip_ended']
    if body.get('playback_rate') is not None:
        rate = _number(body['playback_rate']) if not isinstance(body['playback_rate'], str) else None
        if rate is None or not 0.25 <= rate <= 4:
            return None, dropped, 'playback_rate is not a rate from 0.25 to 4'
        fields['playback_rate'] = round(rate, 3)
    return fields, dropped, None


def _safe_note(note: str | None) -> str | None:
    """a transcript note the page may show: the known ones as they are, any other (which may name a path or
    a session) as a plain reason"""
    if note is None or note in SAFE_NOTES:
        return note
    return 'translation unavailable' if note.startswith('translation') else 'transcript unavailable'


# what of a coder's own label line the page gets back
PAGE_FIELDS = LABEL_FIELDS


class LockedHandler(GuardedHandler, C.Handler):
    """the coding page of a campaign, for the coder its cookie names"""
    KIND = 'code'
    HOME = '/'
    ROUTES = {('GET', '/'): '_page', ('GET', '/api/boot'): '_boot', ('GET', '/api/windows'): '_windows_route',
              ('GET', '/api/progress'): '_progress_route', ('GET', '/api/text'): '_text_route',
              ('GET', '/clip'): '_clip_route', ('POST', '/api/label'): '_label_route',
              ('POST', '/api/unlabel'): '_unlabel_route'}
    plans: dict[str, dict[str, Plan]] = {}
    coder_sets: dict[str, str] = {}
    by_alias: dict[str, dict] = {}
    clips: Path | None = None
    artifacts: Path | None = None
    lock = threading.Lock()  # one clip cut at a time

    # what the inherited code reads ----

    def _labels_path(self, session: dict[str, Any], coder: str) -> Path:
        return Path(session['dir']) / 'labels' / LOCKED_DIR / self.campaign.name / f'{C.safe_name(coder)}.jsonl'

    def _plan(self) -> dict[str, Plan]:
        name = self.identity.name if self.identity else None
        chosen = self.coder_sets.get(name, self.coder_sets.get('*'))
        return self.plans.get(chosen, {})

    def _windows(self, session: dict[str, Any]) -> list[dict[str, float]]:
        plan = self._plan().get(session['id'])
        window = self.settings['window']
        return [{'start': s, 'end': round(s + window, 3)} for s in plan.starts] if plan else []

    def _ordered(self) -> list[dict]:
        """the coder's sessions, in an order of their own (seeded by the campaign and the coder)"""
        plan, seed, name = self._plan(), self.settings.get('seed', 1), self.identity.name
        listed = [s for s in self.sessions if s['id'] in plan]
        return sorted(listed, key=lambda s: sha256_hex(f"{seed}:{name}:{s['id']}"))

    def _session(self, query) -> dict[str, Any] | None:
        return self._aliased(_first(query, 'session'))

    def _aliased(self, alias) -> dict[str, Any] | None:
        session = self.by_alias.get(alias) if isinstance(alias, str) else None
        return session if session is not None and session['id'] in self._plan() else None

    def note_query(self, query: dict) -> None:
        super().note_query(query)
        alias = _first(query, 'session')
        session = self._aliased(alias) if self.identity else None
        if session is None:
            return
        self.note(session=session['id'], alias=alias)
        start = _number(_first(query, 'start'))
        if start is not None:
            plan = self._plan()[session['id']]
            found = plan.by_offset.get(_key(start))
            if found is None and _first(query, 'context') not in (None, '', '0'):
                found = plan.context_at(start)
            if found is not None:
                self._fields.pop('start', None)
                self.note(window_start=found)

    def _scrub(self, record: dict, session: dict, plan: Plan) -> dict:
        """a label line as the page gets it back: its own fields, the alias and offsets"""
        out = {k: record[k] for k in PAGE_FIELDS if k in record}
        offset = plan.by_abs[_key(record['window_start'])]
        out.update(session=session['alias'], window_start=offset, window_end=round(offset + self.settings['window'], 3),
                   coder=self.identity.name)
        return out

    # routes ----

    def _page(self, request: Request) -> None:
        self.send_html(LOCKED_PAGE)

    def _boot(self, request: Request) -> None:
        data, plan = self.campaign.data(), self._plan()
        sessions = []
        for session in self._ordered():
            audio = session.get('audio')
            sessions.append({'id': session['alias'], 'start': 0.0, 'end': round(session['end'] - session['start'], 3),
                             'n_windows': len(plan[session['id']].starts), 'cameras': len(session.get('videos') or []),
                             'audio': {'scope': audio.get('scope')} if audio else None})
        self.send_json({'codebook': C.CODEBOOK,
                        'locked': {'coder': self.identity.name, 'text': bool(data.get('text')),
                                   'min_view': data.get('min_view') or 'full',
                                   'prior_windows': int(data.get('prior_windows') or 0),
                                   'window': self.settings['window'], 'step': self.settings['step'],
                                   'closed': bool(data.get('closed_at'))},
                        'sessions': sessions, 'hidden': []})

    def _windows_route(self, request: Request) -> None:
        session = self._session(request.query)
        if session is None:
            return self.send_json({'error': 'no such session'}, 404)
        plan, window = self._plan()[session['id']], self.settings['window']
        labels = self._labels(session, self.identity.name)
        self.send_json({'windows': [{'start': o, 'end': round(o + window, 3)} for o in plan.offsets],
                        'labels': [self._scrub(r, session, plan) for k, r in labels.items() if k in plan.by_abs]})

    def _progress_route(self, request: Request) -> None:
        progress = {}
        for session in self._ordered():
            plan = self._plan()[session['id']]
            coded = set(self._labels(session, self.identity.name)) & set(plan.by_abs)
            progress[session['alias']] = {'coded': len(coded), 'windows': len(plan.starts)}
        self.send_json({'progress': progress})

    def _text_route(self, request: Request) -> None:
        if not self.campaign.data().get('text'):
            return self.send_json({'error': 'not found'}, 404)
        if self.closed():
            return self.send_json({'error': 'the campaign is closed'}, 410)
        session = self._session(request.query)
        if session is None:
            return self.send_json({'error': 'no such session'}, 404)
        start = _number(_first(request.query, 'start'))
        if start is None:
            return self.send_json({'error': 'give start=<window start>'}, 400)
        absolute = self._plan()[session['id']].by_offset.get(_key(start))
        if absolute is None:
            return self.send_json({'error': 'not found'}, 404)
        window, t0 = self.settings['window'], session['start']
        out = {'start': start, 'end': round(start + window, 3), 'context': None, 'lines': [], 'translated': False,
               'model': None, 'note': 'transcripts are not served by this page', 'pending': False}
        if self.text_source is not None:
            try:
                data = self.text_source.text(session, absolute, prefetch=_first(request.query, 'prefetch') not in (None, '', '0'))
            except Exception as error:  # the page says the transcript is unavailable, not why
                self.note(error=type(error).__name__)
                data = {'note': 'transcript unavailable'}
            lines = []
            for line in data.get('lines') or []:
                at = _number(line.get('start'))
                lines.append({'speaker': str(line.get('speaker') or ''), 'approximate': bool(line.get('approximate')),
                              'start': round(at - t0, 3) if at is not None else None,
                              **{part: str(line.get(part) or '') for part in ('before', 'inside', 'after')},
                              'en': line.get('en') if isinstance(line.get('en'), str) else None})
            out.update(context=data.get('context'), lines=lines, translated=bool(data.get('translated')),
                       model=data.get('model') if isinstance(data.get('model'), str) else None,
                       note=_safe_note(data.get('note')), pending=bool(data.get('pending')))
        self.send_json(out)

    def _clip_route(self, request: Request) -> None:
        # a closed campaign's clips were deleted: none is cut again
        if self.closed():
            return self.send_json({'error': 'the campaign is closed'}, 410)
        session = self._session(request.query)
        if session is None:
            return self.send_json({'error': 'no such session'}, 404)
        start = _number(_first(request.query, 'start'))
        if start is None:
            return self.send_json({'error': 'give start=<window start>'}, 400)
        plan = self._plan()[session['id']]
        absolute = plan.by_offset.get(_key(start))
        if absolute is None and _first(request.query, 'context') not in (None, '', '0'):
            absolute = plan.context_at(start)
        if absolute is None:
            return self.send_json({'error': 'not found'}, 404)
        offset = round(absolute - session['start'], 3)
        try:
            with self.lock:
                path = campaign_clip(session, absolute, self.settings['window'],
                                     clip_file(self.clips, session['alias'], offset, self.settings['window']))
        except (subprocess.SubprocessError, OSError) as error:
            self.note(error=type(error).__name__)
            return self.send_json({'error': 'the clip cannot be cut'}, 500)
        if _first(request.query, 'prefetch') not in (None, '', '0'):
            return self.send_body(204, 'text/plain', b'')
        self.send_file(path, 'video/mp4')

    def _label_route(self, request: Request) -> None:
        self._save(request.body, undo=False)

    def _unlabel_route(self, request: Request) -> None:
        self._save(request.body, undo=True)

    def _save(self, body: dict, undo: bool) -> None:
        """a class key, a note, an also-state or an undo of the bound coder, checked (A.7) and appended with
        its log line"""
        data, me = self.campaign.data(), self.identity.name
        if data.get('closed_at'):
            return self.send_json({'error': 'the campaign is closed'}, 409)
        session = self._aliased(body.get('session'))
        if session is None:
            return self.send_json({'error': 'no such session'}, 404)
        self.note(session=session['id'], alias=session['alias'])
        if body.get('coder') is not None and body.get('coder') != me:
            return self.send_json({'error': f'this page is for {me}: reload it'}, 409)
        plan, window = self._plan()[session['id']], self.settings['window']
        start = _number(body.get('window_start'))
        absolute = plan.by_offset.get(_key(start)) if start is not None else None
        if absolute is None:
            return self.send_json({'error': 'not a window of this campaign'}, 400)
        self.note(window_start=absolute)
        now = C.now_utc()
        ids = {'coder': me, 'campaign': self.campaign.name, 'token_id': self.identity.token_id}
        if undo:
            record = {'session': session['id'], 'window_start': absolute, 'label': None, 'undone_at': now, 'saved_at': now, **ids}
        else:
            if body.get('window_end') is not None:
                end = _number(body.get('window_end'))
                if end is None or abs(end - (start + window)) > 0.01:
                    return self.send_json({'error': 'window_end is not window_start plus the window'}, 400)
            fields, dropped, error = clean_record(body)
            if dropped:
                self.note(dropped_keys=dropped)
            if error:
                return self.send_json({'error': error}, 400)
            coded = _key(absolute) in self._labels(session, me)
            if (data.get('min_view') or 'full') == 'full' and not coded \
                    and fields.get('played_seconds', 0.0) < MIN_COVER * window - 0.05:
                return self.send_json({'error': 'watch the whole window first'}, 400)
            record = {'session': session['id'], 'window_start': absolute, 'window_end': round(absolute + window, 3),
                      **fields, **ids, 'saved_at': now}
        path = self._labels_path(session, me)
        self.append_record(path, path.relative_to(self.artifacts).as_posix(), record)


# ---- the locked page: the coding page with the coder bound, offsets for times and the viewing gate ----

LOCKED_JS = r"""// ---- a locked campaign (mmla ses-code --locked): the coder is the one the personal link bound, times are offsets
// into the recording, and with min_view full a class key counts once the window's own clip has played ----
let locked = null;
const MIN_COVER = __MIN_COVER__, MAX_RATE = __MAX_RATE__;
// per window key: the parts of its own clip played ([from, to] in clip seconds), whether it played to its end, the
// fastest rate it played at, the replays (space) and how many preceding windows played before it (p)
const seen = {};
// what the video plays: {k, main} (main: the window's own clip, else a preceding window), and the preceding windows
// still to play ({k, starts, done})
let playing = null, contextQueue = null;
function mmss(t) { const s = Math.max(0, Math.round(t)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; }
function seenOf(k) { return seen[k] || (seen[k] = {ranges: [], ended: false, rate: 1, replays: 0, context: 0}); }
function union(ranges) {
  const out = [];
  for (const [a, b] of ranges.filter(r => r[1] > r[0]).sort((x, y) => x[0] - y[0])) {
    const last = out[out.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b); else out.push([a, b]);
  }
  return out;
}
// the played parts of the window's own clip, kept before its source changes (which empties video.played)
function keepPlayed() {
  const v = $('video'), p = v.played;
  if (!playing || !playing.main || !p) return;
  const s = seenOf(playing.k), more = [];
  for (let i = 0; i < p.length; i++) more.push([p.start(i), p.end(i)]);
  s.ranges = union(s.ranges.concat(more));
}
function covered(k) { keepPlayed(); return seenOf(k).ranges.reduce((sum, [a, b]) => sum + b - a, 0); }
function playClip(src) { const v = $('video'); v.src = src; v.load(); v.play().catch(() => {}); }
optionText = function (s) { const p = progress[s.id]; return `${s.id} · ${s.n_windows} window${s.n_windows === 1 ? '' : 's'}${p ? ` · ${p.coded}/${p.windows} coded` : ''}`; };
clipSound = function () { const a = session.audio; return !a ? 'the clip has no sound' : a.scope === 'group' ? 'the clip plays the group mic' : 'the clip plays a worn mic'; };
const pageToggleText = toggleText;
toggleText = function () { if (locked && locked.text) pageToggleText(); };
const pageRender = render;
render = function (quiet) {
  keepPlayed(); contextQueue = null;
  const dropped = pageRender(quiet), w = windows[index];
  playing = w ? {k: key(w), main: true} : null;
  return dropped;
};
const pageLabel = label;
label = function (cls) {
  const w = windows[index];
  // a double press, or a key while a save is out, stays the page's own to drop or refuse
  if (w && locked && locked.min_view === 'full' && !saving && Date.now() - advancedAt >= 400 && !labels[key(w)]) {
    const left = locked.window * MIN_COVER - covered(key(w));
    if (left > 0.05) { said(`NOT saved: watch the whole window first (${Math.ceil(left)} s left)`, true); return; }
  }
  return pageLabel(cls);
};
const pagePost = post;
post = function (path, record) {
  // a class key's record says how its window was watched; a re-saved note or also-state keeps what its code said
  if (path === '/api/label' && !('played_seconds' in record)) {
    const k = Number(record.window_start).toFixed(3), s = seenOf(k), played = covered(k);
    Object.assign(record, {played_seconds: Math.round(played * 10) / 10, clip_ended: s.ended, playback_rate: s.rate,
                           replays: s.replays, context_played: s.context});
  }
  return pagePost(path, record);
};
// p: the preceding windows (as many as the campaign allows), then this window's clip again
function playContext() {
  const w = windows[index]; if (!w || !locked) return;
  const starts = [];
  for (let j = locked.prior_windows; j >= 1; j--) { const s = Math.round((w.start - j * locked.step) * 1000) / 1000; if (s >= 0) starts.push(s); }
  if (!starts.length) { said(locked.prior_windows ? 'nothing precedes the first window' : 'this campaign plays no preceding window', 'plain'); return; }
  keepPlayed();
  contextQueue = {k: key(w), starts, done: 0}; playing = {k: key(w), main: false};
  playClip(`/clip?session=${encodeURIComponent(session.id)}&start=${starts[0]}&context=1`);
  said(`playing the ${starts.length === 1 ? 'preceding window' : `${starts.length} preceding windows`}, then this window again`, 'plain');
}
function clipEnded() {
  if (!playing) return;
  if (playing.main) { keepPlayed(); seenOf(playing.k).ended = true; return; }
  const q = contextQueue; if (!q) return;
  q.done++; q.starts.shift(); seenOf(q.k).context = Math.max(seenOf(q.k).context, q.done);
  if (q.starts.length) { playClip(`/clip?session=${encodeURIComponent(session.id)}&start=${q.starts[0]}&context=1`); return; }
  contextQueue = null;
  const w = windows[index];
  if (w && key(w) === q.k) { playing = {k: q.k, main: true}; playClip(`/clip?session=${encodeURIComponent(session.id)}&start=${w.start}`); }
}
// faster than MAX_RATE does not count as watching: the rate is held there
function rateChanged() {
  const v = $('video');
  if (v.playbackRate > MAX_RATE) v.playbackRate = MAX_RATE;
  if (playing && playing.main) { const s = seenOf(playing.k); s.rate = Math.max(s.rate, v.playbackRate); }
}
if ($('video').addEventListener) { $('video').addEventListener('ended', clipEnded); $('video').addEventListener('ratechange', rateChanged); }
document.addEventListener('keydown', e => {
  if (e.target === $('note') || !codebook || !locked || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
  if (e.key === ' ' && playing && playing.main) seenOf(playing.k).replays++;
  if (e.key === 'p') { e.preventDefault(); playContext(); }
});
"""


def _patched(page: str, patches: list[tuple[str, str]]) -> str:
    for old, new in patches:
        if page.count(old) != 1:
            raise RuntimeError(f'the coding page changed: {old[:60]!r} is not in it once; update code_locked.LOCKED_PAGE')
        page = page.replace(old, new)
    return page


# the default page is left as it is: the locked one is derived from it, so the two share their code
LOCKED_PAGE = _patched(C.PAGE, [
    # the header says "Coding as NAME" with no field and no change name: the link binds the name
    ('<label id="naming" style="display:none">Coding as <input id="newname" size="14" autocomplete="off" spellcheck="false" '
     'placeholder="your name"></label><button id="rename" type="button" style="display:none">change name</button>'
     '<input id="coder" type="hidden">', '<input id="coder" type="hidden">'),
    ('· <b>t</b> transcript</div>', '· <b>t</b> transcript · <b>p</b> preceding window</div>'),
    ("  if (!visitCoder) remember('coder', coder);\n", ''),
    ("  remember('coder', name);\n", ''),
    ("${new Date(w.start * 1000).toISOString().replace('T', ' ').slice(0, 19)}Z · ${Math.round(w.start - session.start)} s into the session",
     '${mmss(w.start)} into the recording'),
    ("  visitCoder = (query.get('coder') || '').trim() || null; ownName = recall('coder') || '';\n"
     "  $('coder').value = visitCoder || ownName; showVisit();\n"
     "  $('rename').onclick = () => editName(); $('newname').onblur = () => closeName();\n"
     "  const given = visitCoder ? '' : (query.get('name') || '').trim();\n"
     "  if (query.has('name')) { query.delete('name'); const rest = query.toString(); history.replaceState(null, '', "
     "`${location.pathname}${rest ? `?${rest}` : ''}${location.hash}`); }\n"
     "  const refused = given && await nameRefused(given);\n"
     "  if (given && !refused) ownCoder(given);\n"
     "  // a refused name stays in the field, said why; with no name yet the field is open for one\n"
     "  if (refused) { editName(); $('newname').value = given; nameSaid = said(`name not changed: ${refused}`, true); }\n"
     "  else if (!$('coder').value) editName(); else nameField(false);\n",
     "  locked = boot.locked; visitCoder = null;\n"
     "  $('coder').value = locked.coder; $('codingas').textContent = `Coding as ${locked.coder}`; showVisit();\n"
     "  if (locked.closed) { $('hidden').textContent = 'This campaign is closed: codes are no longer saved.'; $('hidden').style.display = ''; }\n"
     "  if (!locked.text) $('texttoggle').style.display = 'none';\n"),
    ("  showText = recall('showText') === '1';", "  showText = !!locked.text && recall('showText') === '1';"),
    ("(async () => {\n  const boot = await api('/api/boot'); codebook = boot.codebook; sessions = boot.sessions;\n",
     LOCKED_JS.replace('__MIN_COVER__', repr(MIN_COVER)).replace('__MAX_RATE__', repr(MAX_RATE))
     + "(async () => {\n  const boot = await api('/api/boot'); codebook = boot.codebook; sessions = boot.sessions;\n"),
])


# ---- the operator's commands ----

def add_arguments(parser) -> None:
    """the campaign flags of mmla ses-code"""
    group = parser.add_argument_group('locked coding campaigns', 'blind coding over one-time personal links; see '
                                      'docs/analytics/coding_and_audit.md')
    group.add_argument('--campaign', default=None, metavar='DIR', help="the campaign folder (e.g. artifacts/runtime/coding/<name>); every flag below but --verify-log needs it")
    group.add_argument('--campaign-init', action='store_true', help="write the campaign's campaign.yml (with --campaign-sessions)")
    group.add_argument('--campaign-sessions', default='', help="the campaign's exact session ids, comma-separated")
    group.add_argument('--campaign-no-text', action='store_true', help="no Transcript button in the campaign")
    group.add_argument('--min-view', choices=('full', 'none'), default='full',
                       help="full (default): a class key counts once the window's clip has played")
    group.add_argument('--prior-windows', type=int, default=1, help="preceding windows the p key plays (default 1)")
    group.add_argument('--draw-assignment', default=None, metavar='N|all', help="draw the windows the coders code (code_assign)")
    group.add_argument('--strata-coders', default='', help="the default-mode coders whose labels define the strata, comma-separated")
    group.add_argument('--strata', default='lesson,majority', help="lesson, majority and/or unanimity (default lesson,majority)")
    group.add_argument('--assign-filter', choices=('all', 'non-unanimous'), default='all',
                       help="non-unanimous: only the windows the strata coders coded differently")
    group.add_argument('--min-per-stratum', type=int, default=3, help="the fewest windows drawn from a stratum (default 3)")
    group.add_argument('--assign-seed', type=int, default=1, help="the seed of the aliases and the draw (default 1)")
    group.add_argument('--issue-token', default=None, metavar='NAME', help="give NAME a one-time link, written to <campaign>/links/NAME.txt")
    group.add_argument('--token-scope', choices=('code', 'audit'), default='code', help="the server the link opens (default code)")
    group.add_argument('--token-subset', default=None, choices=SUBSETS, help="an audit link for the second auditor: the reliability subset only")
    group.add_argument('--link-base', default=None, help=f"the server's address in the link (default http://127.0.0.1:{DEFAULT_PORT})")
    group.add_argument('--revoke-token', default=None, metavar='NAME', help="revoke NAME's links")
    group.add_argument('--log-note', default=None, metavar='TEXT', help="append an operator's note to the request log (a port probe, an approval)")
    group.add_argument('--locked', action='store_true', help=f"serve the campaign (port {DEFAULT_PORT} unless -p)")
    group.add_argument('--allow-from', default='', help="the addresses that may connect (needed unless bound to 127.0.0.1)")
    group.add_argument('--allow-wide', action='store_true', help="let --allow-from name a network, not only hosts (logged)")
    group.add_argument('--i-know-another-instance-runs', action='store_true',
                       help="start although another ses-code serves the same artifacts (logged)")
    group.add_argument('--prepare-clips', action='store_true', help="cut every assigned window's clip into the campaign's cache, then exit")
    group.add_argument('--close-campaign', action='store_true', help="refuse every later save and delete the clip cache")
    group.add_argument('--release-campaign', action='store_true', help="after close: copy the coders' labels to labels/")
    group.add_argument('--despite-log-failure', action='store_true',
                       help="close, release or score although the request log does not verify (logged)")
    group.add_argument('--anchor', action='store_true', help="print only hashes (the log's head, each coder file) to send outside")
    group.add_argument('--verify-log', default=None, metavar='PATH', help="check a request log's chain (and its campaign's label lines); exit 3: crash cuts only")


ACTIONS = ('campaign_init', 'draw_assignment', 'issue_token', 'revoke_token', 'log_note', 'locked', 'prepare_clips',
           'close_campaign', 'release_campaign', 'anchor')


def requested(args) -> bool:
    """whether the flags ask for a campaign command (an --audit* flag is the audit's, even with --campaign)"""
    if any(value for name, value in vars(args).items() if name.startswith('audit')):
        return False
    return bool(args.campaign or args.verify_log or any(getattr(args, a, None) for a in ACTIONS))


def campaign_sessions(campaign: Campaign, artifacts: Path) -> list[dict]:
    """the campaign's sessions as the page serves them (code.load_sessions, S1 and --hold not applied),
    each with its alias, in the campaign's order"""
    wanted = campaign.data()['sessions']
    shown, _ = C.load_sessions(artifacts, None, show_all=True)
    by_id = {s['id']: s for s in shown}
    missing = [s['id'] for s in wanted if s['id'] not in by_id]
    if missing:
        raise CampaignError(f'{len(missing)} campaign session(s) have no listed video under {artifacts}, first {missing[0]}')
    return [{**by_id[s['id']], 'alias': s['alias']} for s in wanted]


def label_files(campaign: Campaign, artifacts: Path) -> list[tuple[str, Path]]:
    """every file in the campaign's label folders (a sensing audit's folder: its answers files), as
    label_file names it"""
    files = []
    data = campaign.data()
    for entry in data['sessions']:
        folder = artifacts / entry['id'] / 'labels' / LOCKED_DIR / campaign.name
        if data.get('audit_id'):
            folder = artifacts / entry['id'] / 'audit' / data['audit_id'] / 'answers'
        for path in sorted(folder.iterdir()) if folder.is_dir() else []:
            if path.is_file():
                files.append((path.relative_to(artifacts).as_posix(), path))
    return files


def _artifacts(args, campaign: Campaign | None) -> Path:
    if args.artifacts:
        return Path(args.artifacts).resolve()
    if campaign is not None and campaign.exists() and campaign.data().get('artifacts'):
        return Path(campaign.data()['artifacts'])
    return Path(os.getcwd(), 'artifacts').resolve()


def _log(campaign: Campaign) -> RequestLog:
    return RequestLog(campaign.folder / LOG_FILE)


def _event(campaign: Campaign, event: str, /, **fields) -> dict:
    log = _log(campaign)
    try:
        return log.event(event, pid=os.getpid(), **fields)
    finally:
        log.close()


def _coding(campaign: Campaign) -> dict:
    """the campaign, refused when the folder is a sensing audit's (its links and log only are shared)"""
    data = campaign.data()
    if data.get('audit_id'):
        raise CampaignError(f"{campaign.folder} is the sensing audit {data['audit_id']}: it is served and scored by the "
                            '--audit flags')
    return data


def _open_campaign(campaign: Campaign) -> dict:
    data = campaign.data()
    if data.get('closed_at'):
        raise CampaignError(f"the campaign was closed at {data['closed_at']}")
    return data


def checked_log(campaign: Campaign, artifacts: Path, override: bool, step: str) -> dict:
    """the request log verified with the campaign's label files before `step` (close, release, score): its
    head and what the check found, for the step's own line and outputs. A log that does not verify refuses
    the step unless `override` (--despite-log-failure), which the result records; crash cuts pass, named."""
    ok, message, info = verify_log(campaign.folder / LOG_FILE, label_files(campaign, artifacts))
    if not ok and not override:
        raise CampaignError(f'the request log does not verify: {message}. Nothing was done (--despite-log-failure '
                            f'{step}s anyway, logged)')
    return {'ok': ok, 'message': message, 'head': info['head'], 'seq': info['seq'], 'cuts': info['cuts'],
            'despite_failure': not ok}


def cmd_init(campaign: Campaign, args, artifacts: Path, argv) -> int:
    if not CAMPAIGN_NAME.fullmatch(campaign.name):
        raise CampaignError("the campaign folder's name names its labels folder: letters, digits, - and _, at most 40")
    ids = [i.strip() for i in args.campaign_sessions.split(',') if i.strip()]
    if not ids:
        raise CampaignError('give --campaign-sessions ID,ID,...')
    if len(set(ids)) != len(ids):
        raise CampaignError('a session is listed twice in --campaign-sessions')
    if args.prior_windows < 0:
        raise CampaignError('--prior-windows is a count of windows, 0 or more')
    shown, _ = C.load_sessions(artifacts, None, show_all=True)
    listed = {s['id'] for s in shown}
    missing = [i for i in ids if i not in listed]
    if missing:
        raise CampaignError(f'{len(missing)} session(s) have no listed video under {artifacts}, first {missing[0]}')
    order = list(ids)
    random.Random(f'{args.assign_seed}:{campaign.name}:aliases').shuffle(order)
    alias = {sid: f'R{i + 1:02d}' for i, sid in enumerate(order)}
    data = {'campaign': campaign.name, 'created_at': C.now_utc(), 'artifacts': str(artifacts), 'window': args.window,
            'step': args.step, 'seed': args.assign_seed, 'sessions': [{'id': sid, 'alias': alias[sid]} for sid in ids],
            'text': not args.campaign_no_text, 'min_view': args.min_view, 'prior_windows': args.prior_windows,
            'cookie_days': COOKIE_DAYS, 'assignment': None, 'closed_at': None, 'released_at': None, 'coders': []}

    def change(current):
        if current:
            raise CampaignError(f'{campaign.path} exists already')
        current.update(data)
    campaign.update(change, create=True)
    _event(campaign, 'init', campaign_sha256=file_sha256(campaign.path), sessions=len(ids), argv=list(argv))
    print(f"campaign {campaign.name}: {len(ids)} session{'s' if len(ids) > 1 else ''}, aliases R01 to R{len(ids):02d}; "
          f'written to {campaign.path}')
    return 0


def cmd_draw(campaign: Campaign, args, artifacts: Path, argv) -> int:
    from openmmla.commands.ses import code_assign
    _coding(campaign)
    data = _open_campaign(campaign)
    if any(t.get('claimed_at') for c in data['coders'] for t in c.get('tokens', [])):
        raise CampaignError('a link was used already: the assignment cannot be drawn again')
    coders = [c.strip() for c in args.strata_coders.split(',') if c.strip()]
    sessions = campaign_sessions(campaign, artifacts)
    try:
        assignment, design = code_assign.draw_assignment(
            campaign.name, sessions, float(data['window']), float(data['step']), coders,
            None if args.draw_assignment == 'all' else int(args.draw_assignment),
            strata=tuple(s.strip() for s in args.strata.split(',') if s.strip()), population_filter=args.assign_filter,
            min_per_stratum=args.min_per_stratum, seed=args.assign_seed, artifacts=artifacts)
    except ValueError as error:
        raise CampaignError(str(error)) from None
    build_plans(sessions, assignment, float(data['window']), float(data['step']), int(data.get('prior_windows') or 0))
    _write_json(campaign.folder / ASSIGNMENT_FILE, assignment)
    _write_json(campaign.folder / DESIGN_FILE, design)
    campaign.update(lambda d: d.update(assignment=ASSIGNMENT_FILE))
    _event(campaign, 'assignment', assignment_sha256=file_sha256(campaign.folder / ASSIGNMENT_FILE),
           design_sha256=file_sha256(campaign.folder / DESIGN_FILE), windows=len(assignment['sets']['main']),
           argv=list(argv))
    print(f"{len(assignment['sets']['main'])} windows drawn from {design['population']} in {len(design['strata'])} strata; "
          f"{ASSIGNMENT_FILE} and {DESIGN_FILE} written to {campaign.folder}")
    return 0


def name_refusal(name: str, sessions: list[dict]) -> str | None:
    """why `name` cannot be a campaign coder's name, or None: it must be its own safe name, not the
    adjudicated file's, and no labels file of the default page (a coder's or a model's) in a campaign
    session may hold it, in any case"""
    if not name or C.safe_name(name) != name:
        return 'a coder name is letters, digits, - and _ only'
    error = C.name_error(name)
    if error:
        return error
    if name.lower() == C.ADJUDICATED:
        return f"'{C.ADJUDICATED}' is the consensus file, not a coder"
    for session in sessions:
        folder = Path(session['dir']) / 'labels'
        for path in sorted(folder.glob('*.jsonl')) if folder.is_dir() else []:
            if path.stem.lower() == name.lower():
                kind = C.file_kind(path.read_text(encoding='utf-8', errors='replace'))
                return (f"{path.stem!r} holds a model's labels" if kind == 'model' else
                        f'{path.stem!r} is a coder of the default page in a campaign session: give another name')
    return None


def cmd_issue(campaign: Campaign, args, artifacts: Path, argv) -> int:
    _open_campaign(campaign)
    name = args.issue_token
    refusal = name_refusal(name, campaign_sessions(campaign, artifacts))
    if refusal:
        raise CampaignError(refusal)
    base = (args.link_base or f'http://127.0.0.1:{DEFAULT_PORT}').rstrip('/')
    if not re.fullmatch(r'https?://[^/\s]+', base):
        raise CampaignError('--link-base is http://HOST:PORT')
    token, token_id, action = issue_token(campaign, name, args.token_scope, args.token_subset)
    links = campaign.folder / LINKS_DIR
    links.mkdir(mode=0o700, exist_ok=True)
    os.chmod(links, 0o700)
    path = links / f'{name}.txt'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as file:
        file.write(f'{base}/c/{token}\n')
    os.chmod(path, 0o600)
    print(f'{action}d a {args.token_scope} link for {name} (token {token_id}): written to {path}, readable by you only. '
          'Hand it over yourself (paper or typed), then delete the file; it works once, in one browser.')
    return 0


def cmd_revoke(campaign: Campaign, args, artifacts: Path, argv) -> int:
    revoked = revoke_token(campaign, args.revoke_token)
    link = campaign.folder / LINKS_DIR / f'{args.revoke_token}.txt'
    link.unlink(missing_ok=True)
    print(f"revoked {len(revoked)} link(s) of {args.revoke_token}" + (f": {', '.join(revoked)}" if revoked else ''))
    return 0


def cmd_note(campaign: Campaign, args, artifacts: Path, argv) -> int:
    campaign.data()
    entry = _event(campaign, 'note', text=args.log_note[:2000])
    print(f"noted at seq {entry['seq']}")
    return 0


def _explicit_port(argv) -> bool:
    return any(a in ('-p', '--port') or a.startswith('--port=') or (a.startswith('-p') and a[2:].isdigit()) for a in argv)


def watch_campaign(campaign: Campaign, log: RequestLog) -> None:
    """log every change a server finds in campaign.yml, with the file's new sha256: the operator's links and
    revokes, the server's own claims, and any edit by hand"""
    campaign.watch = lambda digest: log.event('campaign-changed', campaign_sha256=digest)


def cmd_serve(campaign: Campaign, args, artifacts: Path, argv) -> int:
    data = _coding(campaign)
    if data.get('closed_at'):
        raise CampaignError(f"the campaign was closed at {data['closed_at']}: it is not served again")
    allow = check_bind(args.bind, args.allow_from, args.allow_wide)
    port = args.port if _explicit_port(argv) else DEFAULT_PORT
    sessions = campaign_sessions(campaign, artifacts)
    assignment = json.loads((campaign.folder / data['assignment']).read_text()) if data.get('assignment') else None
    window, step = float(data['window']), float(data['step'])
    plans, coder_sets = build_plans(sessions, assignment, window, step, int(data.get('prior_windows') or 0),
                                    int(data.get('seed') or 1))
    others = other_instances(artifacts)
    if others and not args.i_know_another_instance_runs:
        listed = '; '.join(f"pid {o['pid']}: {o['command'][:120]}" for o in others)
        raise CampaignError(f'another ses-code may serve {artifacts} ({listed}): stop it or rebind it to 127.0.0.1 and '
                            'limit the coders to this port, then give --i-know-another-instance-runs')
    text_source = None
    if data.get('text'):
        influx_config = args.influx_config or os.path.join(os.getcwd(), C.DEFAULT_INFLUX_CONFIG)
        text_source = C.TextSource(window, args.context, influx_config, C.Translator(threads=args.threads))
    log = _log(campaign)
    handler = handler_class(LockedHandler, campaign=campaign, log=log, allow=allow, sessions=sessions,
                            hidden=[], by_alias={s['alias']: s for s in sessions}, plans=plans, coder_sets=coder_sets,
                            settings={'window': window, 'step': step, 'sample': 1.0, 'block': 300.0,
                                      'seed': int(data.get('seed') or 1)},
                            text_source=text_source, clips=campaign.folder / CLIPS_DIR, artifacts=artifacts,
                            lock=threading.Lock())
    windows = sum(len(p.starts) for chosen in plans.values() for p in chosen.values())
    campaign.data()
    watch_campaign(campaign, log)
    extra = {'campaign': campaign.name, 'campaign_sha256': file_sha256(campaign.path),
             'assignment_sha256': file_sha256(campaign.folder / data['assignment']) if data.get('assignment') else None,
             'another_instances': others or None, 'allow_wide': bool(args.allow_wide)}
    print(f"campaign {campaign.name}: {len(sessions)} sessions, {windows} assigned windows; serving "
          f"http://{args.bind}:{port}/ to {', '.join(str(n) for n in allow)} (Ctrl-C stops)")
    try:
        return serve(handler, args.bind, port, argv, LOCKED_MODULES, extra)
    finally:
        campaign.watch = None
        log.close()


def cmd_prepare_clips(campaign: Campaign, args, artifacts: Path, argv) -> int:
    data = _coding(campaign)
    sessions = campaign_sessions(campaign, artifacts)
    assignment = json.loads((campaign.folder / data['assignment']).read_text()) if data.get('assignment') else None
    window, step = float(data['window']), float(data['step'])
    plans, _ = build_plans(sessions, assignment, window, step, int(data.get('prior_windows') or 0),
                           int(data.get('seed') or 1))
    failed = 0
    for session in sessions:
        starts = sorted({s for chosen in plans.values() if session['id'] in chosen
                         for s in list(chosen[session['id']].starts) + list(chosen[session['id']].context.values())})
        done = 0
        for start in starts:
            offset = round(start - session['start'], 3)
            try:
                campaign_clip(session, start, window, clip_file(campaign.folder / CLIPS_DIR, session['alias'], offset, window))
                done += 1
            except (subprocess.SubprocessError, OSError) as error:
                failed += 1
                print(f"{session['alias']}: the clip at {offset:.0f} s cannot be cut ({type(error).__name__})")
        print(f"{session['alias']}: {done} of {len(starts)} clips cut")
    return 1 if failed else 0


def _say_log(check: dict) -> None:
    if check['despite_failure']:
        print(f"the request log does not verify ({check['message']}): going on as --despite-log-failure asks, logged")
    elif check['cuts']:
        print(f"the request log: {check['message']}")


def cmd_close(campaign: Campaign, args, artifacts: Path, argv) -> int:
    _open_campaign(campaign)
    check = checked_log(campaign, artifacts, args.despite_log_failure, 'close')

    def change(data):
        if data.get('closed_at'):
            raise CampaignError(f"the campaign was closed at {data['closed_at']}")
        data['closed_at'] = C.now_utc()
    campaign.update(change)
    clips = campaign.folder / CLIPS_DIR
    removed = sum(1 for p in clips.rglob('*') if p.is_file()) if clips.is_dir() else 0
    shutil.rmtree(clips, ignore_errors=True)
    files = {rel: file_sha256(path) for rel, path in label_files(campaign, artifacts)}
    _event(campaign, 'close', files=files, clips_removed=removed, log_check=check,
           campaign_sha256=file_sha256(campaign.path))
    _say_log(check)
    print(f'campaign {campaign.name} closed: saves are refused from now on; {removed} cached clips deleted')
    return 0


def cmd_release(campaign: Campaign, args, artifacts: Path, argv) -> int:
    data = _coding(campaign)
    if not data.get('closed_at'):
        raise CampaignError('close the campaign first (--close-campaign)')
    if data.get('released_at'):
        raise CampaignError(f"the campaign was released at {data['released_at']}")
    check = checked_log(campaign, artifacts, args.despite_log_failure, 'release')
    copies = []
    for rel, path in label_files(campaign, artifacts):
        if path.suffix != '.jsonl':
            continue
        target = path.parent.parent.parent / path.name  # labels/locked/<campaign>/<name>.jsonl -> labels/<name>.jsonl
        clash = [p for p in target.parent.glob('*.jsonl') if p.stem.lower() == target.stem.lower()]
        if clash:
            raise CampaignError(f'{clash[0].relative_to(artifacts)} exists: nothing is released')
        copies.append((path, target))
    released = []
    for source, target in copies:
        shutil.copyfile(source, target)
        digest = file_sha256(source)
        if file_sha256(target) != digest:
            raise CampaignError(f'{target} is not a true copy')
        released.append({'from': source.relative_to(artifacts).as_posix(), 'to': target.relative_to(artifacts).as_posix(),
                         'sha256': digest})
    when = C.now_utc()
    log = {k: check[k] for k in ('ok', 'message', 'head', 'seq', 'cuts', 'despite_failure')}
    _write_json(campaign.folder / RELEASE_FILE, {'campaign': campaign.name, 'released_at': when, 'files': released,
                                                 'log': log})
    campaign.update(lambda d: d.update(released_at=when))
    _event(campaign, 'release', files={r['to']: r['sha256'] for r in released},
           release_sha256=file_sha256(campaign.folder / RELEASE_FILE), log_check=log)
    _say_log(check)
    print(f'{len(released)} label files released to labels/; {RELEASE_FILE} lists them with their sha256')
    return 0


def anchor_files(campaign: Campaign, artifacts: Path) -> dict[str, str | None]:
    """each label (or answers) file's sha256, keyed by its session's alias and its coder's id (their first
    token's id; for a name no coder of campaign.yml holds, 'n' and the first 8 hex digits of the name's
    sha256), never by a name or a session id; two files under one key are refused"""
    data = campaign.data()
    ids = {c['name']: coder_id(c) for c in data['coders']}
    aliases = {s['id']: s['alias'] for s in data['sessions']}
    files: dict[str, str | None] = {}
    for rel, path in label_files(campaign, artifacts):
        sid, name = rel.split('/', 1)[0], Path(rel).stem
        key = f"{aliases.get(sid, '?')}:{ids[name] if name in ids else 'n' + sha256_hex(name)[:8]}"
        if key in files:
            raise CampaignError(f'two files would be anchored as {key}: nothing was anchored')
        files[key] = file_sha256(path)
    return files


def cmd_anchor(campaign: Campaign, args, artifacts: Path, argv) -> int:
    data = campaign.data()
    files = anchor_files(campaign, artifacts)
    frozen: list[tuple[str, str | None]] = []
    if data.get('audit_id'):
        from openmmla.commands.ses import audit
        frozen = audit.anchor_hashes(artifacts, data['audit_id'])
    entry = _event(campaign, 'anchor', files=files, audit_files=dict(frozen) or None)
    print(f"log_head {entry['sha256']} {entry['seq']}")
    print(f'campaign {file_sha256(campaign.path)}')
    if data.get('assignment'):
        print(f"assignment {file_sha256(campaign.folder / data['assignment'])}")
    for name, digest in frozen:
        print(f'{name} {digest}')
    for name, digest in sorted(files.items()):
        print(f'file {name} {digest}')
    return 0


def cmd_verify(args, argv) -> int:
    path = Path(args.verify_log).resolve()
    campaign = Campaign(path.parent)
    files = None
    if campaign.exists():
        files = label_files(campaign, _artifacts(args, campaign))
    ok, message, info = verify_log(path, files)
    print(message)
    return (3 if info['cuts'] else 0) if ok else 1


COMMANDS = {'campaign_init': cmd_init, 'draw_assignment': cmd_draw, 'issue_token': cmd_issue,
            'revoke_token': cmd_revoke, 'log_note': cmd_note, 'locked': cmd_serve, 'prepare_clips': cmd_prepare_clips,
            'close_campaign': cmd_close, 'release_campaign': cmd_release, 'anchor': cmd_anchor}


def run(args, argv) -> int:
    """a campaign command of mmla ses-code; 0 when done, 1 refused, 2 a usage error"""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if args.verify_log:
            return cmd_verify(args, argv)
        actions = [a for a in ACTIONS if getattr(args, a, None)]
        if not args.campaign or len(actions) != 1:
            print('give --campaign DIR and one of ' + ', '.join('--' + a.replace('_', '-') for a in ACTIONS))
            return 2
        campaign = Campaign(args.campaign)
        return COMMANDS[actions[0]](campaign, args, _artifacts(args, campaign), argv)
    except CampaignError as error:
        print(f'refused: {error}')
        return 1
