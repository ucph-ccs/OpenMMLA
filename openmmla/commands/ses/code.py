"""mmla ses-code: a browser page to code a session's 10-second windows by hand.

Ground truth for the interaction classes comes from a person watching the recordings, ten seconds
at a time. This command serves the sessions under artifacts/: for every window it cuts a clip on
demand with ffmpeg (up to two cameras side by side with the group microphone, exact to the
window's start), and the page plays it and takes one key per window (1 individual or parallel
work, 2 social interaction, 3 collaborative interaction, 4 not at the table, 0 unclear, with an
optional note). Labels go to artifacts/<session>/labels/<coder>.jsonl, one line per save (a
re-saved note or an undo appends a line; the last line of a window counts), so a second coder
writes a second file and the two are compared for agreement. The page counts a label as saved only
when the server answers {"ok": true}, and /api/progress gives a coder's count per session. Clips
are cached under artifacts/<session>/labels/clips/ and can be deleted at any time.

A mixed window keeps its main state (the one that fills most of it) and may carry one also-state
(Shift and a class key), saved as `secondary` and `secondary_key` in the same record; an unclear
window carries none. Every line the server appends carries its own `saved_at`. The /agreement page,
reached by its URL only, compares two coders on the listed windows both coded (/api/agreement:
percent agreement, Cohen's kappa over the five codes, the three classes and two binary splits, a
lenient share that counts a match with the other's also-state, the confusion matrix and the
disagreeing windows), and all coders at once with Krippendorff's alpha (/api/agreement?all=1), on
the current labels or on those saved by a cutoff (asof=<ISO 8601 time>); /api/coders lists the
coders, and the consensus file `adjudicated` apart.

The Transcript button shows what was said in the window, in Danish and in a local English
translation (openmmla.commands.ses.code_text): the session's asr_transcription events from InfluxDB
(--influx-config), the words of the window with --context seconds around them, translated on the
CPU by a MarianMT model and cached under artifacts/<session>/analysis/transcripts/. --prepare-text
translates every listed window ahead of time and exits.

Sessions the interaction classifier leaves out are hidden, so nobody codes windows no model will
read: a session whose fused table (artifacts/<session>/analysis/features/<session>_window_features.csv)
fails the inclusion rule S1 (openmmla.analytics.interaction.layout.session_inclusion) is not listed.
A session without a fused table cannot be judged and is kept, and so is every session when the
analytics package (pandas) cannot be imported. --all lists every session. A voided session is one
moved out of artifacts/; the manifest has no marker for it.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openmmla.collection.recording import default_audio_scope
from openmmla.commands.ses.code_text import CONTEXT, DEFAULT_INFLUX_CONFIG, TextSource, Translator

# the coder's instruction to mark teacher talk: a note only a person can add, left out of Jev's question
TEACHER_NOTE = ', and add the note "teacher" when the teacher talks to the group or the class for most of the window'
CODEBOOK = {
    'classes': [
        {'key': '1', 'label': 'individual', 'title': 'Individual or parallel work',
         'definition': 'Nobody in the group interacts with another member during most of the window: each works alone, waits, or watches the teacher. A member looking elsewhere while another works is individual. Glances without exchange do not count.'},
        {'key': '2', 'label': 'social', 'title': 'Social interaction',
         'definition': 'Members interact (talk, gesture, look at each other) but not about the task: chat, jokes, phones, waiting together.'},
        {'key': '3', 'label': 'collaborative', 'title': 'Collaborative interaction',
         'definition': 'Members interact about the task: talking about it, joint attention on the shared artifact (micro:bit, microscope, tablet, sheet), pointing, handing over, working on one thing together, explaining or asking. One member following another\'s work on the shared artifact for most of the window counts, even in silence; a glance does not.'},
        {'key': '4', 'label': 'absent', 'title': 'Not at the table',
         'definition': "Fewer than two group members are at the group's place for most of the window: everyone is away, or one of a pair is. In a group of three with one member away two remain, so code the window normally. Code what the video shows, not what the sensors show."},
        {'key': '0', 'label': 'unclear', 'title': 'Unclear',
         'definition': "The group is at its place but you cannot tell its state: members are out of frame and inaudible. A window two states share is not unclear: code the state that fills more of it."},
    ],
    'rule': 'Label the group as a whole with the state that fills most of the ten seconds. Use the preceding windows as context. If two members collaborate while a third works alone, it is still collaborative interaction. The teacher\'s talk does not make a window social or collaborative: code what the members do with each other and with the shared artifact' + TEACHER_NOTE + '. Members looking at the shared artifact while the teacher talks, with no member working on it, is individual work: they follow the teacher, not each other, so it is not the joint attention of collaborative interaction. One member working on it while another follows is collaborative.',
}
# the codes in the page's order (keys 1, 2, 3, 4, 0), which is also the order of the agreement's confusion matrix
CODES = tuple(c['label'] for c in CODEBOOK['classes'])
CLASSES = ('individual', 'social', 'collaborative')
UNCLEAR = 'unclear'
# the file whose labels overrule the coders' (openmmla.analytics.interaction.labels.ADJUDICATED): no coder of its own
ADJUDICATED = 'adjudicated'
AUDIO_PREFERENCE = ('jabra-0', 'vimo-0-ch0', 'vimo-0', 'badge-0')
CAMERA_PREFERENCE = ('c920-01', 'c920-04', 'c920-05', 'c920-06', 'c920-02', 'c920-03')


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def fused_table(session_dir: Path) -> Path:
    return session_dir / 'analysis' / 'features' / f'{session_dir.name}_window_features.csv'


def _inclusion_rule():
    """the classifier's layout module, which holds S1; imported here so ses-code starts without
    pandas. Raises ImportError when the analytics package cannot be loaded."""
    from openmmla.analytics.interaction import layout
    return layout


def inclusion(session_dir: Path, rule=None) -> tuple[bool | None, str]:
    """S1 for one session, as the classifier applies it: (True, reason) or (False, reason) from its
    fused table and its roster (the pupils its manifest declares, else the rules), or (None, why
    it cannot be judged) when there is no table or it cannot be read.
    `rule` is the layout module (None: import it here)."""
    path = fused_table(session_dir)
    if not path.exists():
        return None, 'no fused table (mmla ses-fuse), so S1 is not checked'
    try:
        layout = rule or _inclusion_rule()
    except ImportError as error:
        return None, f'the analytics package cannot be imported ({error}), so S1 is not checked'
    try:
        table = layout.read_table(path)
        return layout.session_inclusion(table, layout.session_roster(table, session_dir))
    except Exception as error:  # a broken table must not stop the coding page
        return None, f'the fused table cannot be read ({type(error).__name__}: {error}), so S1 is not checked'


def load_sessions(artifacts: Path, pattern: str | None = None,
                  show_all: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(shown, hidden): every artifacts/exp_* session with video whose id contains `pattern`.
    A session that fails S1 goes to `hidden` ({'id', 'reason'}) unless `show_all`; every shown
    session carries 'included' (True, False, or None when it could not be judged) and its
    'inclusion' reason."""
    sessions, hidden = [], []
    rule = None
    for session_dir in sorted(artifacts.glob('exp_*')):
        if pattern and pattern not in session_dir.name:
            continue
        manifest = _read(session_dir / 'manifest.json')
        recordings = [r for r in manifest.get('recordings', []) if os.path.exists(r.get('path', ''))]
        videos = [r for r in recordings if r['modality'] == 'video']
        audios = [r for r in recordings if r['modality'] == 'audio']
        if not videos:
            continue
        if rule is None and fused_table(session_dir).exists():
            try:
                rule = _inclusion_rule()
            except ImportError:
                rule = False  # inclusion() says why, per session
        included, reason = inclusion(session_dir, rule or None)
        if included is False and not show_all:
            hidden.append({'id': session_dir.name, 'reason': reason})
            continue
        videos.sort(key=lambda r: (CAMERA_PREFERENCE.index(r['device']) if r['device'] in CAMERA_PREFERENCE else 99, r['device']))
        audios.sort(key=lambda r: (AUDIO_PREFERENCE.index(r['device']) if r['device'] in AUDIO_PREFERENCE else 99, r['device']))
        start = max(r['start_time'] for r in recordings)
        end = min(r['start_time'] + (r.get('duration') or 0) for r in recordings if r.get('duration'))
        audio = dict(audios[0]) if audios else None
        if audio and not audio.get('scope'):  # the page says whose microphone the clip plays
            audio['scope'] = default_audio_scope(audio.get('device'), audio.get('method'), audio.get('host'))
        sessions.append({'id': session_dir.name, 'dir': str(session_dir), 'start': start, 'end': end,
                         'videos': videos[:2], 'audio': audio,
                         'experiment': manifest.get('experiment_id'), 'group': manifest.get('group_id'),
                         'included': included, 'inclusion': reason})
    return sessions, hidden


def _test_sessions() -> set[str]:
    """the classifier's TEST sessions, which are coded in full whatever the sampling; empty when
    the analytics package cannot be imported."""
    try:
        from openmmla.analytics.interaction.splits import TEST_SESSIONS
    except Exception:
        return set()
    return set(TEST_SESSIONS)


def windows_of(session: dict[str, Any], window: float, step: float, sample: float, block: float, seed: int) -> list[dict[str, float]]:
    """the windows to code, in time order; `sample` < 1 keeps that share of `block`-second blocks,
    drawn with a fixed seed so two coders see the same windows. A TEST session is always coded in
    full, since it is scored on every window."""
    if session['id'] in _test_sessions():
        sample = 1.0
    starts = []
    t = session['start']
    while t + window <= session['end']:
        starts.append(t)
        t += step
    if sample < 1.0:
        blocks: dict[int, list[float]] = {}
        for s in starts:
            blocks.setdefault(int((s - session['start']) // block), []).append(s)
        keys = sorted(blocks)
        rng = random.Random(f"{seed}:{session['id']}")
        keep = set(rng.sample(keys, max(1, round(len(keys) * sample))))
        starts = [s for k in keys if k in keep for s in blocks[k]]
    return [{'start': round(s, 3), 'end': round(s + window, 3)} for s in starts]


def clip_path(session: dict[str, Any], start: float, window: float) -> Path:
    return Path(session['dir']) / 'labels' / 'clips' / f"{start:.3f}_{window:g}s.mp4"


def make_clip(session: dict[str, Any], start: float, window: float, width: int = 640) -> Path:
    """the window's clip: the cameras side by side, the microphone as its audio, exact to the start"""
    out = clip_path(session, start, window)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-v', 'error', '-y']
    for video in session['videos']:
        command += ['-ss', f"{start - video['start_time']:.3f}", '-t', f"{window:.3f}", '-i', video['path']]
    audio = session['audio']
    if audio:
        command += ['-ss', f"{start - audio['start_time']:.3f}", '-t', f"{window:.3f}", '-i', audio['path']]
    n = len(session['videos'])
    scaled = ''.join(f"[{i}:v]scale={width}:-2,setsar=1[v{i}];" for i in range(n))
    stack = ''.join(f"[v{i}]" for i in range(n)) + (f"hstack=inputs={n}[out]" if n > 1 else "copy[out]")
    command += ['-filter_complex', scaled + stack, '-map', '[out]']
    if audio:
        command += ['-map', f'{n}:a', '-c:a', 'aac', '-b:a', '96k']
    command += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)]
    temp = out.with_suffix('.tmp.mp4')
    subprocess.run(command[:-1] + [str(temp)], check=True, timeout=300)
    temp.replace(out)
    return out


def safe_name(coder: str) -> str:
    """the coder's name as the name of their labels file"""
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in coder) or 'anonymous'


# the longest safe name a labels file may have, in UTF-8 bytes: well under the 255 of the file systems
NAME_BYTES = 100


def name_error(coder: str) -> str | None:
    """why `coder` cannot name a labels file, or None; checked before the file system is touched"""
    if len(safe_name(coder).encode('utf-8')) > NAME_BYTES:
        return f'the coder name is too long for a file name (at most {NAME_BYTES} bytes)'
    return None


def secondary_error(record: dict[str, Any]) -> str | None:
    """why a label record's also-state (`secondary`) cannot be saved, or None: it is optional, one
    of the codes, never unclear, never the main label itself, and an unclear window takes none."""
    secondary = record.get('secondary')
    if secondary is None:
        return None
    if secondary not in CODES:
        return f'unknown secondary {secondary!r}'
    if secondary == UNCLEAR:
        return 'unclear is never a secondary'
    if record.get('label') == UNCLEAR:
        return "the label 'unclear' takes no secondary"
    if secondary == record.get('label'):
        return 'the secondary must differ from the label'
    return None


_TIME = re.compile(r'(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?\s*(Z|[+-]\d{2}:?\d{2})?', re.I)


def parse_time(text: Any) -> datetime | None:
    """an ISO 8601 date and time, as the page (UTC with Z) and the server (local time with an offset
    such as +0200) write them, on any Python the package supports; naive when it has no zone, None
    when it is not such a time."""
    match = _TIME.fullmatch(text.strip()) if isinstance(text, str) else None
    if not match:
        return None
    day, hour, minute, second, fraction, zone = match.groups()
    zone_info = None
    if zone and zone.upper() == 'Z':
        zone_info = timezone.utc
    elif zone:
        sign = -1 if zone[0] == '-' else 1
        zone_info = timezone(sign * timedelta(hours=int(zone[1:3]), minutes=int(zone[-2:])))
    try:
        return datetime.strptime(day, '%Y-%m-%d').replace(
            hour=int(hour), minute=int(minute), second=int(second or 0),
            microsecond=int((fraction or '0')[:6].ljust(6, '0')), tzinfo=zone_info)
    except ValueError:  # a day or an hour out of range
        return None


def saved_time(record: dict[str, Any]) -> datetime | None:
    """when a line of a labels file was saved: the server's `saved_at`, else (a line written before
    it) the page's `edited_at` or `coded_at`, or an undo line's `undone_at`; a time without a zone
    is taken as UTC, as the page writes it. None when the line has no time."""
    for field in ('saved_at', 'edited_at', 'coded_at', 'undone_at'):
        value = parse_time(record.get(field))
        if value is not None:
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def utc_text(moment: datetime | None) -> str | None:
    """an aware time as `saved_at` holds it: UTC, to the millisecond, with Z (None stays None)"""
    return moment.astimezone(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z') if moment else None


def now_utc() -> str:
    """the server's time now, as `saved_at` holds it"""
    return utc_text(datetime.now(timezone.utc))


def replay(text: str, asof: datetime | None = None) -> tuple[dict[str, dict[str, Any]], int]:
    """a coder's labels from their file, keyed by window start (to the millisecond): the last line of
    a window counts and a line without a label (an undo) removes it. With `asof` only the lines
    saved at or before it count, and a line without a time is left out; the second value is how
    many lines were left out that way. A line cut short by a crash is skipped, as
    labels._read_coder does."""
    labels: dict[str, dict[str, Any]] = {}
    undated = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = f"{float(record['window_start']):.3f}"
        except (ValueError, KeyError, TypeError):
            continue
        if asof is not None:
            saved = saved_time(record)
            if saved is None:
                undated += 1
                continue
            if saved > asof:
                continue
        if record.get('label') is None:
            labels.pop(key, None)  # an undo line
        else:
            labels[key] = record
    return labels, undated


# ---- agreement between coders, in plain Python so ses-code starts without numpy or pandas ----

def kappa(confusion: list[list[int]]) -> float | None:
    """Cohen's kappa of a square confusion matrix (rows one coder, columns the other), computed as
    openmmla.analytics.interaction.labels does; None when the matrix is empty or chance agreement
    is 1 (both coders gave one and the same code throughout)."""
    n = sum(map(sum, confusion))
    if n == 0:
        return None
    observed = sum(confusion[i][i] for i in range(len(confusion))) / n
    expected = sum(sum(row) * sum(column) for row, column in zip(confusion, zip(*confusion))) / n ** 2
    return (observed - expected) / (1 - expected) if expected < 1 else None


def pair_agreement(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """two coders' agreement over `pairs`, one (a's record, b's record) per window both coded, each
    label one of CODES: the windows (n), the share with the same code (agree), Cohen's kappa over
    the five codes, over the three classes on the windows both gave a class (n_classes), over
    collaborative against the other two classes and over interaction (social or collaborative)
    against individual on those same windows, the lenient share (a main label that equals the
    other coder's main label or also-state agrees) and the five-code confusion matrix (rows a,
    columns b, in CODES order)."""
    at = {label: i for i, label in enumerate(CODES)}
    codes = [[0] * len(CODES) for _ in CODES]
    classes = [[0] * len(CLASSES) for _ in CLASSES]
    collaborative = [[0, 0], [0, 0]]
    interaction = [[0, 0], [0, 0]]
    lenient = 0
    for x, y in pairs:
        p, q = x['label'], y['label']
        codes[at[p]][at[q]] += 1
        if p in CLASSES and q in CLASSES:
            classes[CLASSES.index(p)][CLASSES.index(q)] += 1
            collaborative[p == 'collaborative'][q == 'collaborative'] += 1
            interaction[p != 'individual'][q != 'individual'] += 1
        lenient += p == q or p == y.get('secondary') or q == x.get('secondary')
    n = len(pairs)
    return {'n': n, 'agree': sum(codes[i][i] for i in range(len(CODES))) / n if n else None,
            'kappa_codes': kappa(codes), 'n_classes': sum(map(sum, classes)), 'kappa_classes': kappa(classes),
            'kappa_collaborative': kappa(collaborative), 'kappa_interaction': kappa(interaction),
            'lenient': lenient / n if n else None, 'confusion': codes}


def krippendorff_alpha(units: list[list[str]]) -> float | None:
    """Krippendorff's alpha for nominal codes with missing values. `units` holds, per window, the
    codes the coders gave it (a coder who did not code it is left out); a window with fewer than
    two codes cannot be paired and adds nothing. None when no pair is left or every paired code is
    one and the same."""
    coincidence: dict[tuple[str, str], float] = {}
    for values in units:
        m = len(values)
        if m < 2:
            continue
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        for c, n_c in counts.items():
            for k, n_k in counts.items():
                coincidence[c, k] = coincidence.get((c, k), 0.0) + n_c * (n_k - (c == k)) / (m - 1)
    totals: dict[str, float] = {}
    for (c, _), value in coincidence.items():
        totals[c] = totals.get(c, 0.0) + value
    n = sum(totals.values())
    observed = sum(value for (c, k), value in coincidence.items() if c != k)
    expected = n * n - sum(t * t for t in totals.values())  # the sum of n_c * n_k over c != k
    return 1 - (n - 1) * observed / expected if expected > 0 else None


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Session coding</title>
<style>
 body{margin:0;font:15px/1.4 -apple-system,Helvetica,Arial,sans-serif;background:#111;color:#eee}
 header{display:flex;gap:16px;align-items:center;padding:10px 16px;background:#1b1b1b;flex-wrap:wrap}
 select,input,button{font:inherit;background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 10px}
 button{cursor:pointer} button.active{background:#2d6cdf;border-color:#2d6cdf}
 main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;padding:16px}
 video{width:100%;background:#000;border-radius:8px}
 .keys button{display:block;width:100%;text-align:left;margin:6px 0;padding:10px}
 .keys b{display:inline-block;width:26px;height:26px;line-height:26px;text-align:center;background:#333;border-radius:5px;margin-right:8px}
 .def{color:#aaa;font-size:13px;margin:2px 0 10px 34px}
 .meta{color:#aaa;font-size:13px}
 #coded{font-size:14px;margin:4px 0} #coded.none{color:#aaa}
 .strip{display:flex;flex-wrap:wrap;gap:2px;margin:8px 0}
 .cell{flex:0 0 9px;height:14px;background:#2e2e2e;border-radius:2px;cursor:pointer}
 .cell.here{outline:2px solid #fff;outline-offset:1px;position:relative;z-index:1}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px}
 .l-individual{background:#5b8def} .l-social{background:#e0a93b} .l-collaborative{background:#3fb66b} .l-absent{background:#a070d0} .l-unclear{background:#9a9a9a}
 /* a mixed window: the main state's colour above a band of the also-state's */
 .cell.also-individual{box-shadow:inset 0 -5px 0 #5b8def,inset 0 -6px 0 #111} .cell.also-social{box-shadow:inset 0 -5px 0 #e0a93b,inset 0 -6px 0 #111}
 .cell.also-collaborative{box-shadow:inset 0 -5px 0 #3fb66b,inset 0 -6px 0 #111} .cell.also-absent{box-shadow:inset 0 -5px 0 #a070d0,inset 0 -6px 0 #111}
 .keys button.also{outline:2px dashed #ddd;outline-offset:-5px}
 /* a class button with the chevron that shows or folds its definition; the rule and the also-state sentence fold the same way */
 .keys .row{display:flex;gap:4px;align-items:center} .keys .row button[data-label]{flex:1;min-width:0}
 .keys button.chev{flex:0 0 30px;width:30px;margin:6px 0;padding:6px 0;text-align:center;background:none;border:none;color:#aaa}
 .keys button.chev:hover{color:#fff}
 .fold{display:flex;align-items:center;justify-content:space-between;color:#aaa;font-size:13px}
 #visit{background:#7a4b00;color:#fff;border-radius:6px;padding:4px 10px;font-weight:600} #coder.visiting{border-color:#d9a441}
 textarea{width:100%;height:60px;background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px}
 #status{color:#8c8} #status.plain{color:#ccc} #status.fail{color:#f66;font-weight:600}
 #text{margin-top:12px;padding:10px 12px;background:#1b1b1b;border-radius:8px;font-size:14px}
 #text .line{margin:0 0 10px} #text .who{color:#8ab4f8;font-size:12px;margin-right:6px}
 #text .ctx{color:#777} #text .en{color:#bbb;font-style:italic;margin-top:2px}
 #text .approx{color:#d9a441;font-size:12px;margin-left:6px} #text .voice{color:#8ab4f8;font-size:12px}
</style></head><body>
<header>
 <div id="hidden" class="meta" style="flex-basis:100%;display:none"></div>
 <label>Coder <input id="coder" size="10"></label>
 <span id="visit" style="display:none"></span>
 <label>Session <select id="session"></select></label>
 <button id="texttoggle" onclick="toggleText()">Transcript</button>
 <button id="advancetoggle" onclick="toggleAdvance()">Auto-advance</button>
 <span id="progress" class="meta"></span><span id="status"></span>
</header>
<main>
 <div>
  <video id="video" controls autoplay playsinline></video>
  <div class="meta" id="when"></div>
  <div id="coded"></div>
  <div class="strip" id="strip"></div>
  <div class="meta">Keys: <b>1</b> <b>2</b> <b>3</b> <b>4</b> <b>0</b> label (and move on when Auto-advance is on) · <b>shift+1</b>–<b>4</b> also-state (<b>shift+0</b> clears) · <b>a</b> auto-advance · <b>space</b> replay · <b>←</b> <b>→</b> move · <b>n</b> note · <b>u</b> undo · <b>t</b> transcript</div>
  <div id="text" style="display:none"></div>
 </div>
 <div class="keys">
  <div id="classes"></div>
  <div class="fold"><span>Rule</span><button class="chev" id="chev-rule" aria-expanded="false" aria-label="the rule" onclick="this.blur(); toggleFold('rule')">▸</button></div>
  <div class="def" id="def-rule" hidden></div>
  <div class="fold"><span>Also-state</span><button class="chev" id="chev-also" aria-expanded="false" aria-label="the also-state" onclick="this.blur(); toggleFold('also')">▸</button></div>
  <div class="def" id="def-also" hidden>When a second state fills a clear part of the window, add it as the also-state with Shift and its key; the main state is the one that fills more of the window. A window coded 0 Unclear takes no also-state.</div>
  <textarea id="note" placeholder="note (optional), saved with the next class key; Enter re-saves the note of a coded window"></textarea>
 </div>
</main>
<script>
const $ = id => document.getElementById(id);
let codebook, sessions, session, windows = [], labels = {}, index = 0, shownAt = 0, showText = false, textAbort = null, textRetry = null;
let progress = {}, autoAdvance = true, saving = false, savingWindow = null, advancedAt = 0, undoStack = [], generation = 0;
// an also-state keyed on a window not coded yet, sent with its main state; a deep link's window to open
let pendingAlso = null, jumpTo = null;
// the window auto-advance last left, which a shift key within 2 s still reaches; a shift key pressed while
// the main state of the window on screen was being saved, keyed once that save is confirmed
let advanced = null, queuedAlso = null;
// a name from the link (?coder=), coded under for this visit only: the remembered name stays
let visitCoder = null;
// the number of the last status said, and the last refusal ({n, text}), which a later answer never hides
let told = 0, refusal = null;
async function api(path, options) {
  const r = await fetch(path, options);
  if (!r.ok) { const data = await r.json().catch(() => ({})); throw new Error(`${r.status} ${data.error || r.statusText}`.trim()); }
  return r.json();
}
// a save counts only when the server answers {ok: true}; one that hangs gives up after 30 s
async function post(path, record) {
  const data = await api(path, {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(record),
                                signal: AbortSignal.timeout ? AbortSignal.timeout(30000) : undefined});
  if (!data || data.ok !== true) throw new Error('the server did not confirm it');
}
function why(e) { return e.name === 'TimeoutError' ? 'no answer in 30 s' : e instanceof TypeError ? 'the server cannot be reached' : e.message; }
// tone: true for a refusal or a failure (red), 'plain' for what is neither done nor refused; returns the line's number
function said(text, tone) {
  $('status').textContent = text; $('status').classList.toggle('fail', tone === true); $('status').classList.toggle('plain', tone === 'plain');
  told++; if (tone === true) refusal = {n: told, text};
  return told;
}
// the answer to a save that was out while other keys were pressed: a refusal said since the save began stays, the answer after it
function answered(text, since, tone) { return refusal && refusal.n > since ? said(`${refusal.text} · ${text}`, true) : said(text, tone); }
function clock() { return new Date().toTimeString().slice(0, 8); }
function key(w) { return w.start.toFixed(3); }
function recall(name) { try { return localStorage.getItem(name); } catch (e) { return null; } }
function remember(name, value) { try { localStorage.setItem(name, value); } catch (e) {} }
function esc(text) { return String(text || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function attr(text) { return esc(text).replace(/"/g, '&quot;'); }
function part(text, cls) { return text ? `<span class="${cls}">${esc(text)}</span> ` : ''; }
function classOf(l) { return codebook.classes.find(c => c.label === l.label) || {key: l.key, label: l.label, title: l.label}; }
function alsoOf(l) { return l && l.secondary ? classOf({label: l.secondary, key: l.secondary_key}) : null; }
// what a shift key asks for, as the top row names it
function alsoName(k) { const c = codebook.classes.find(x => x.key === k); return k === '0' || !c ? 'shift+0 (clear the also-state)' : `also-state ${c.key} ${c.title}`; }
function pendingOn(w) { return !!(pendingAlso && w && session && pendingAlso.session === session.id && pendingAlso.k === key(w) && !labels[key(w)]); }
function coderName() { return $('coder').value.trim() || 'anonymous'; }
function showVisit() {
  $('visit').textContent = visitCoder ? `coding as ${visitCoder} for this visit` : ''; $('visit').style.display = visitCoder ? '' : 'none';
  $('coder').classList.toggle('visiting', !!visitCoder);
}
// a class's definition, the rule or the also-state sentence, shown or folded; the browser remembers each one's state
function fold(id, open) {
  $(`def-${id}`).hidden = !open;
  const chev = $(`chev-${id}`); chev.textContent = open ? '▾' : '▸'; chev.setAttribute('aria-expanded', open ? 'true' : 'false');
}
function toggleFold(id) { const open = !!$(`def-${id}`).hidden; fold(id, open); remember(`open:${id}`, open ? '1' : '0'); }
// a pupil's line is what that pupil's worn mic transcribed, the partner and the teacher included; it names no speaker
function speaker(s) { return /^pupil /.test(s) ? `${s}'s mic` : s; }
// one block per microphone, the group mic first, then the pupils' mics; a block keeps its lines in
// time order, and a group-mic line carries its diarized voice
function blocks(lines) {
  const order = [], by = {};
  for (const l of lines) {
    const m = /^group mic(?: · (.+))?$/.exec(l.speaker || '');
    const source = m ? 'group mic' : speaker(l.speaker);
    if (!by[source]) { by[source] = []; order.push(source); }
    by[source].push({...l, voice: m ? (m[1] || '') : ''});
  }
  order.sort((a, b) => (b === 'group mic') - (a === 'group mic') || a.localeCompare(b, undefined, {numeric: true}));
  return order.map(source => [source, by[source]]);
}
// the clip's sound is the session's preferred microphone, a worn one when it has no group microphone
function clipSound() { const a = session.audio; return !a ? 'the clip has no sound' : a.scope === 'group' ? 'the clip plays the group mic' : `the clip plays ${a.device}${a.scope === 'personal' ? ', a worn mic' : ''}`; }
function s1(s) { return s.included === false ? ', left out by S1' : s.included === null ? ', S1 not checked' : ''; }
function optionText(s) {
  const p = progress[s.id];
  return `${s.id} (${Math.round((s.end - s.start) / 60)} min, ${s.videos.length} cam${s.audio ? ', audio' : ', no audio'}${s1(s)})${p ? ` · ${p.coded}/${p.windows} coded` : ''}`;
}
function showProgress() { for (const o of $('session').options) { const s = sessions.find(x => x.id === o.value); if (s) o.textContent = optionText(s); } }
async function loadProgress() {
  const coder = coderName();
  try { const data = await api(`/api/progress?coder=${encodeURIComponent(coder)}`); if (coder !== coderName()) return; progress = data.progress; }
  catch (e) { progress = {}; }
  if (session && windows.length) renderMarks(); else showProgress();
}
function toggleText() {
  showText = !showText; remember('showText', showText ? '1' : '0');
  $('texttoggle').classList.toggle('active', showText); $('texttoggle').blur();
  renderText();
}
function toggleAdvance() {
  autoAdvance = !autoAdvance; remember('autoAdvance', autoAdvance ? '1' : '0');
  $('advancetoggle').classList.toggle('active', autoAdvance); $('advancetoggle').blur();
  said(`Auto-advance ${autoAdvance ? 'on' : 'off'}`);
}
async function renderText(again) {
  // a window left behind drops its requests, so they never hold the connections the clips need
  if (textAbort) textAbort.abort();
  clearTimeout(textRetry);
  const box = $('text'); box.style.display = showText ? '' : 'none';
  const w = windows[index]; if (!showText || !w) return;
  if (!again) box.innerHTML = '<div class="meta">…</div>';
  const at = w.start, id = session.id, abort = new AbortController(); textAbort = abort;
  let data;
  try { data = await api(`/api/text?session=${encodeURIComponent(id)}&start=${at}`, {signal: abort.signal}); }
  catch (e) { if (abort.signal.aborted) return; data = {lines: [], note: `transcript unavailable: ${why(e)}`}; }
  if (abort.signal.aborted || !showText || !windows[index] || windows[index].start !== at || session.id !== id) return;
  const lines = data.lines || [];
  let html = `<div class="meta">Transcript of the window, <span class="ctx">grey: ${data.context ?? ''} s before and after</span> · a pupil's mic picks up the others too; ${esc(clipSound())}</div>`;
  if (!lines.length && !data.note) html += '<div class="meta">nothing transcribed in the window</div>';
  for (const [source, group] of blocks(lines)) {
    html += `<div class="line"><div><span class="who">${esc(source)}</span>${group.some(l => l.approximate) ? '<span class="approx">approximate: a chunk has no word times</span>' : ''}</div>`;
    for (const l of group) {
      html += `<div>${l.voice ? `<span class="voice">${esc(l.voice)}</span> ` : ''}${part(l.before, 'ctx')}${part(l.inside, 'in')}${part(l.after, 'ctx')}</div>`;
      if (l.en) html += `<div class="en${l.inside ? '' : ' ctx'}">${esc(l.en)}</div>`;
    }
    html += '</div>';
  }
  if (data.note) html += `<div class="meta">${esc(data.note)}</div>`;
  box.innerHTML = html;
  // the model is still loading: ask again for this window, and prefetch nothing until it is ready
  if (data.pending) { textRetry = setTimeout(() => renderText(true), 3000); return; }
  for (const next of windows.slice(index + 1, index + 3)) fetch(`/api/text?session=${encodeURIComponent(id)}&start=${next.start}&prefetch=1`, {signal: abort.signal}).catch(() => {});
}
// an also-state waits for its own window's main state; opening another window drops it. The text that says so, or ''
function dropPending(w) {
  if (!pendingAlso || pendingOn(w)) return '';
  const text = `also-state ${pendingAlso.cls.key} ${pendingAlso.cls.title} dropped: its window was left without a main state`;
  pendingAlso = null;
  return text;
}
// opens the window at index; `quiet` leaves the saying of a dropped also-state to the caller, which gets its text back
function render(quiet) {
  const w = windows[index], dropped = dropPending(w);
  if (dropped && !quiet) said(dropped, 'plain');
  if (!w) { renderMarks(); return dropped; }
  const v = $('video'); v.src = `/clip?session=${session.id}&start=${w.start}`; v.load(); v.play().catch(() => {});
  shownAt = Date.now();
  const current = labels[key(w)];
  $('note').value = current ? (current.note || '') : '';
  for (const next of windows.slice(index + 1, index + 4)) fetch(`/clip?session=${session.id}&start=${next.start}&prefetch=1`);
  renderMarks();
  renderText();
  return dropped;
}
// what the codes show: the buttons, the coded line, the strip and the counts; the clip and the transcript stay
function renderMarks() {
  const w = windows[index], current = w && labels[key(w)];
  const done = windows.filter(x => labels[key(x)]).length;
  progress[session.id] = {coded: done, windows: windows.length}; showProgress();
  $('progress').textContent = `${done} of ${windows.length} coded`;
  $('when').textContent = w ? `window ${index + 1} of ${windows.length} · ${new Date(w.start * 1000).toISOString().replace('T', ' ').slice(0, 19)}Z · ${Math.round(w.start - session.start)} s into the session` : 'no windows to code in this session';
  const c = current && classOf(current), s = alsoOf(current), p = !current && pendingOn(w) ? pendingAlso.cls : null;
  $('coded').innerHTML = current ? `<i class="sw l-${attr(c.label)}"></i>${esc(`coded: ${c.key} ${c.title}${s ? ` · also ${s.key} ${s.title}` : ''}${current.note ? ` · note: ${current.note}` : ''}`)}`
    : w ? esc(`not coded yet${p ? ` · also ${p.key} ${p.title}, waiting for the main state` : ''}`) : '';
  $('coded').classList.toggle('none', !current);
  const alsoLabel = s ? s.label : p ? p.label : null;
  document.querySelectorAll('#classes button[data-label]').forEach(b => {
    b.classList.toggle('active', !!current && b.dataset.label === current.label);
    b.classList.toggle('also', b.dataset.label === alsoLabel);
  });
  $('strip').innerHTML = windows.map((x, i) => {
    const l = labels[key(x)], a = alsoOf(l);
    const title = l ? `window ${i + 1} · ${classOf(l).title}${a ? ` · also ${a.title}` : ''}${l.note ? ` · ${l.note}` : ''}` : `window ${i + 1} · not coded`;
    return `<span class="cell${l ? ` l-${attr(l.label)}` : ''}${a ? ` also-${attr(a.label)}` : ''}${i === index ? ' here' : ''}" data-i="${i}" title="${attr(title)}"></span>`;
  }).join('');
}
function remembered(record) {
  // the undo history holds each window once, at the place of its latest save
  undoStack = undoStack.filter(u => !(u.session === record.session && u.window_start === record.window_start));
  undoStack.push({session: record.session, window_start: record.window_start});
}
async function label(cls) {
  const w = windows[index]; if (!w) return;
  // a second press on the window being saved is dropped; on another window it is said, not dropped unseen
  if (saving) { if (savingWindow !== w) said('NOT saved: the previous save is still waiting for the server — try again', true); return; }
  // nor does a double press land on the window auto-advance has just opened
  if (Date.now() - advancedAt < 400) return;
  const coder = $('coder').value.trim(); if (!coder) { said('type your name as coder first', true); return; }
  if (!visitCoder) remember('coder', coder);
  const record = {session: session.id, window_start: w.start, window_end: w.end, label: cls.label, key: cls.key,
                  note: $('note').value.trim(), coder, coded_at: new Date().toISOString(), seconds_spent: Math.round((Date.now() - shownAt) / 100) / 10};
  // the also-state goes with the main state: the saved one's, else the one keyed on this window. It is dropped when it
  // equals the new main state, and on a window coded unclear, which takes none
  const current = labels[key(w)], pend = pendingOn(w) ? pendingAlso : null;
  const carried = current && current.secondary ? {label: current.secondary, key: current.secondary_key} : pend ? {label: pend.cls.label, key: pend.cls.key} : null;
  const drop = !carried ? '' : cls.label === 'unclear' ? 'an unclear window takes none' : carried.label === cls.label ? 'it is the main state now' : '';
  if (carried && !drop) { record.secondary = carried.label; record.secondary_key = carried.key; }
  if (pend) pendingAlso = null;
  const gen = generation, at = index;
  saving = true; savingWindow = w; const since = said('saving…');
  try { await post('/api/label', record); }
  catch (e) {
    // one waiting on this window waits again while the page is still on it; a shift key pressed during the save is lost
    const kept = !!pend && !pendingAlso && gen === generation && index === at;
    if (kept) pendingAlso = pend;
    const lost = (pend && !kept ? ` · also-state ${pend.cls.key} ${pend.cls.title} dropped` : '') + (queuedAlso ? ` · ${alsoName(queuedAlso.k)} not saved either` : '');
    queuedAlso = null;
    answered(`NOT saved: ${why(e)} — try again${lost}`, since, true); return;
  }
  finally { saving = false; savingWindow = null; }
  const queued = queuedAlso && queuedAlso.w === w ? queuedAlso.k : null; queuedAlso = null;
  const alsoText = record.secondary ? ` · also ${classOf({label: record.secondary, key: record.secondary_key}).title}` : drop ? ` · also-state dropped: ${drop}` : '';
  const text = `saved ✓ ${clock()} · ${index !== at ? `window ${at + 1} · ` : ''}${cls.title}${alsoText}`;
  if (coder === $('coder').value.trim()) remembered(record);
  // another session or coder was loaded while the save was out: a shift key held back for it is not saved, and said so
  if (gen !== generation || session.id !== record.session) { answered(text + (queued !== null ? ` · ${alsoName(queued)} not saved` : ''), since, queued !== null || undefined); return; }
  labels[key(w)] = record;
  let moved = false, dropped = '';
  if (autoAdvance && index === at) {
    // the next window after it not yet coded, else simply the next one
    const next = windows.findIndex((x, i) => i > at && !labels[key(x)]);
    index = next >= 0 ? next : Math.min(at + 1, windows.length - 1);
    if (index !== at) { moved = true; advancedAt = Date.now(); advanced = {session: session.id, gen, from: at, to: index, at: advancedAt}; dropped = render(true); }
  }
  if (!moved) renderMarks();
  answered(text + (dropped ? ` · ${dropped}` : ''), since);
  if (queued !== null) also(queued, at, since);
}
// shift + a class key: the also-state, a second state that fills a clear part of the window; the same key again or
// shift+0 clears it. It never moves the page on. A coded window saves it at once, as a new line of the same record
// and not a label of its own (u still takes back the last class key, the whole record with it); one not coded yet
// keeps it for its main state. `target` is the index of the window it is for, when a save held it back, and `after`
// the number of that save's 'saving…': a refusal said since then stays in what this one says
async function also(k, target, after) {
  let at = target === undefined ? index : target, w = windows[at]; if (!w) return;
  const tell = (text, tone) => after === undefined ? said(text, tone) : answered(text, after, tone);
  const cls = k === '0' ? null : codebook.classes.find(c => c.key === k);
  if (k !== '0' && !cls) return;
  if (cls && cls.label === 'unclear') { tell('NOT set: unclear is never an also-state', true); return; }
  if (saving) {
    // pressed while the main state of the window on screen is being saved: it is for that window, once the save is confirmed
    if (target === undefined && savingWindow === w) { queuedAlso = {w, k}; return; }
    said('NOT saved: the previous save is still waiting for the server — try again', true); return;
  }
  // within 2 s of auto-advance it is for the window just coded, whether or not the window it opened is coded: a shift key
  // so soon is not meant for a clip just begun
  const a = advanced;
  if (target === undefined && a && a.gen === generation && a.session === session.id && a.to === index && Date.now() - a.at < 2000
      && !pendingOn(w) && windows[a.from] && labels[key(windows[a.from])]) { at = a.from; w = windows[at]; }
  const there = at !== index ? ` of window ${at + 1}` : '';
  const current = labels[key(w)], had = current ? current.secondary : pendingOn(w) ? pendingAlso.cls.label : null;
  if (current && current.label === 'unclear') { tell(`NOT set: ${at !== index ? `window ${at + 1}` : 'this window'} is coded 0 Unclear, which takes no also-state`, true); return; }
  const next = cls && cls.label !== had ? cls : null;
  if (!next && !had) { tell(`no also-state to clear${at !== index ? ` on window ${at + 1}` : ''}`, 'plain'); return; }
  if (current && next && next.label === current.label) { tell(`NOT set: the also-state must differ from the main state${there}, ${classOf(current).title}`, true); return; }
  if (!current) {
    pendingAlso = next ? {session: session.id, k: key(w), cls: next} : null;
    tell(next ? `also-state ${next.key} ${next.title}: waits for this window's main state` : 'also-state cleared', 'plain');
    return renderMarks();
  }
  const coder = $('coder').value.trim(); if (!coder) { tell('type your name as coder first', true); return; }
  const record = {...current, coder, edited_at: new Date().toISOString()};
  if (next) { record.secondary = next.label; record.secondary_key = next.key; } else { delete record.secondary; delete record.secondary_key; }
  const gen = generation, since = after === undefined ? told : after;
  // a refusal said since a held-back key's save began stays on screen while this one is out
  saving = true; if (!refusal || refusal.n <= since) said('saving…');
  try { await post('/api/label', record); }
  catch (e) {
    // the page has moved on: the failure names the window and the key, which a shift key on screen now no longer reaches
    answered(at !== index ? `NOT saved: ${next ? `also-state ${next.key} ${next.title}` : 'clearing the also-state'}${there}: ${why(e)} — open window ${at + 1} and press shift+${k} again`
      : `NOT saved: ${why(e)} — try again`, since, true);
    return;
  }
  finally { saving = false; }
  answered(index !== at ? `${next ? `also ${next.title} saved` : 'also-state cleared'} on window ${at + 1} ✓ ${clock()}`
    : `saved ✓ ${clock()} · ${classOf(record).title}${next ? ` · also ${next.title}` : ' · also-state cleared'}`, since);
  if (gen === generation && session.id === record.session) { labels[key(w)] = record; renderMarks(); }
}
async function saveNote() {
  const w = windows[index], current = w && labels[key(w)], note = $('note').value.trim();
  if (!current || note === (current.note || '')) return;
  if (saving) { said('NOT saved: the previous save is still waiting for the server — try again', true); return; }
  const coder = $('coder').value.trim(); if (!coder) { said('type your name as coder first', true); return; }
  const record = {...current, note, coder, edited_at: new Date().toISOString()}, gen = generation;
  saving = true; const since = said('saving…');
  try { await post('/api/label', record); }
  catch (e) { answered(`NOT saved: ${why(e)} — try again`, since, true); return; }
  finally { saving = false; }
  answered(`saved ✓ ${clock()} · ${classOf(record).title} · note`, since);
  if (gen === generation && session.id === record.session) { labels[key(w)] = record; renderMarks(); }
}
async function undo() {
  if (saving || !session) return;
  const coder = $('coder').value.trim(); if (!coder) { said('type your name as coder first', true); return; }
  // a window of this session whose label is gone already has nothing left to undo
  undoStack = undoStack.filter(u => u.session !== session.id || labels[u.window_start.toFixed(3)]);
  const last = undoStack[undoStack.length - 1], w = windows[index];
  if (last && last.session !== session.id) { said(`the last label is in ${last.session}: open that session to undo it`); return; }
  const target = last || (w && labels[key(w)] ? {session: session.id, window_start: w.start} : null);
  if (!target) { said('nothing to undo'); return; }
  const k = target.window_start.toFixed(3), old = labels[k], gen = generation;
  saving = true; const since = said('undoing…');
  try { await post('/api/unlabel', {session: target.session, window_start: target.window_start, coder}); }
  catch (e) { answered(`NOT undone: ${why(e)} — try again`, since, true); return; }
  finally { saving = false; }
  undoStack = undoStack.filter(u => !(u.session === target.session && u.window_start === target.window_start));
  if (gen !== generation || session.id !== target.session) { answered(`undone ✓ ${clock()}`, since); return; }
  const i = windows.findIndex(x => key(x) === k);
  delete labels[k];
  // the jump to the undone window drops an also-state waiting on the one left, and the confirmation says both
  let dropped = '';
  if (i >= 0 && i !== index) { index = i; dropped = render(true); } else renderMarks();
  answered(`undone ✓ ${clock()} · window ${i + 1}${old ? `, was ${classOf(old).title}` : ''}${dropped ? ` · ${dropped}` : ''}`, since);
}
async function loadSession(id) {
  const gen = ++generation, next = sessions.find(s => s.id === id), coder = coderName();
  let data;
  try { data = await api(`/api/windows?session=${encodeURIComponent(id)}&coder=${encodeURIComponent(coder)}`); }
  // the list goes back to the session still on screen, so it names what is coded and a second pick retries
  catch (e) { if (gen === generation) { said(`session not loaded: ${why(e)}`, true); if (session) $('session').value = session.id; } return; }
  if (gen !== generation) return;
  session = next; windows = data.windows; labels = {}; for (const l of data.labels) labels[l.window_start.toFixed(3)] = l;
  // a deep link opens its window; otherwise the first window not coded yet
  const linked = jumpTo && jumpTo.session === id ? windows.findIndex(w => key(w) === jumpTo.start.toFixed(3)) : -1, missed = !!jumpTo && jumpTo.session === id && linked < 0;
  if (jumpTo && jumpTo.session === id) jumpTo = null;
  index = linked >= 0 ? linked : windows.findIndex(w => !labels[key(w)]); if (index < 0) index = 0;
  remember('session', id);
  render();
  if (missed) said('the linked window is not listed in this session');
}
document.addEventListener('keydown', e => {
  if (e.target === $('note')) { if (e.key === 'Enter') { e.preventDefault(); $('note').blur(); saveNote(); } return; }
  // Enter commits the name: the blur fires its change event, and the keys reach the page again
  if (e.target === $('coder')) { if (e.key === 'Enter') { e.preventDefault(); $('coder').blur(); } return; }
  if (!codebook || e.metaKey || e.ctrlKey || e.altKey) return;
  // shift + a digit is the also-state, read from the physical key (row or numpad), since what shift+1 types depends on the layout
  const digit = e.shiftKey && /^(?:Digit|Numpad)([0-9])$/.exec(e.code || '');
  if (digit) { e.preventDefault(); if (!e.repeat) also(digit[1]); return; }
  const cls = codebook.classes.find(c => c.key === e.key);
  // a held key repeats: only the arrows may, so holding 3 or u never codes or undoes a run of windows
  if (e.repeat && (cls || [' ', 'a', 'n', 't', 'u'].includes(e.key))) { e.preventDefault(); return; }
  if (cls) { e.preventDefault(); label(cls); return; }
  if (e.key === ' ') { e.preventDefault(); $('video').currentTime = 0; $('video').play(); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); if (index > 0) { index--; render(); } }
  if (e.key === 'ArrowRight') { e.preventDefault(); if (index < windows.length - 1) { index++; render(); } }
  if (e.key === 'a') { e.preventDefault(); toggleAdvance(); }
  if (e.key === 'n') { e.preventDefault(); $('note').focus(); }
  if (e.key === 't') { e.preventDefault(); toggleText(); }
  if (e.key === 'u') { e.preventDefault(); undo(); }
});
(async () => {
  const boot = await api('/api/boot'); codebook = boot.codebook; sessions = boot.sessions;
  // a class button labels the window; the chevron beside it shows or folds its definition
  $('classes').innerHTML = codebook.classes.map(c => `<div class="row"><button data-label="${c.label}" onclick="this.blur(); label(codebook.classes.find(x=>x.key==='${c.key}'))"><b>${c.key}</b><i class="sw l-${c.label}"></i>${c.title}</button>` +
    `<button class="chev" id="chev-${c.key}" aria-expanded="false" aria-label="the definition of ${attr(c.title)}" onclick="this.blur(); toggleFold('${c.key}')">▸</button></div><div class="def" id="def-${c.key}" hidden>${c.definition}</div>`).join('');
  $('def-rule').textContent = codebook.rule;
  for (const id of [...codebook.classes.map(c => c.key), 'rule', 'also']) fold(id, recall(`open:${id}`) === '1');
  // ?coder=<name> (the Agreement page's adjudicate links) codes under that name for this visit, and the header says so
  const query = new URLSearchParams(location.search);
  visitCoder = (query.get('coder') || '').trim() || null;
  $('coder').value = visitCoder || recall('coder') || ''; showVisit();
  showText = recall('showText') === '1'; $('texttoggle').classList.toggle('active', showText);
  autoAdvance = recall('autoAdvance') !== '0'; $('advancetoggle').classList.toggle('active', autoAdvance);
  $('session').innerHTML = sessions.map(s => `<option value="${attr(s.id)}" title="${attr(s.inclusion)}"></option>`).join(''); showProgress();
  if (boot.hidden.length) {
    $('hidden').textContent = `${boot.hidden.length} session${boot.hidden.length === 1 ? '' : 's'} hidden: left out of the analysis by the inclusion rule S1`;
    $('hidden').title = boot.hidden.map(h => `${h.id}: ${h.reason}`).join('\n');
    $('hidden').style.display = '';
  }
  $('session').onchange = e => { e.target.blur(); loadSession(e.target.value); };
  // a name typed in is the coder's own again, remembered; an also-state keyed under the previous name is dropped
  $('coder').onchange = () => {
    const coder = $('coder').value.trim(); visitCoder = null; showVisit();
    if (coder) remember('coder', coder);
    // a shift key held back by a save still out is said not saved when that save answers
    undoStack = []; advanced = null;
    if (pendingAlso) { said(`also-state ${pendingAlso.cls.key} ${pendingAlso.cls.title} dropped: the coder changed`, 'plain'); pendingAlso = null; }
    loadProgress(); loadSession($('session').value);
  };
  $('strip').onclick = e => { const i = e.target.dataset.i; if (i !== undefined && +i !== index) { index = +i; render(); } };
  // /?session=<id>&start=<window start> (the Agreement page's links) opens that window instead of the remembered session
  const linked = query.get('session'), start = Number(query.get('start')), last = recall('session');
  if (linked && sessions.some(s => s.id === linked)) { $('session').value = linked; if (query.get('start') && isFinite(start)) jumpTo = {session: linked, start}; }
  else if (last && sessions.some(s => s.id === last)) $('session').value = last;
  loadProgress(); loadSession($('session').value);
})();
</script></body></html>"""

AGREEMENT_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coder agreement</title>
<style>
 body{margin:0;font:15px/1.4 -apple-system,Helvetica,Arial,sans-serif;background:#111;color:#eee}
 header{display:flex;gap:10px 16px;align-items:center;padding:10px 16px;background:#1b1b1b;flex-wrap:wrap}
 header label{display:flex;gap:6px;align-items:center;min-width:0;max-width:100%}
 select,input{font:inherit;background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 10px;min-width:0;max-width:62vw}
 input{color-scheme:dark}
 a{color:#8ab4f8}
 main{padding:4px 16px 24px;max-width:1100px}
 h2{font-size:16px;margin:20px 0 8px}
 .meta{color:#aaa;font-size:13px}
 .scroll{overflow-x:auto;max-width:100%}
 table{border-collapse:collapse;font-size:14px}
 th,td{padding:5px 10px;border-bottom:1px solid #2a2a2a;text-align:left;white-space:nowrap;vertical-align:top}
 th{color:#aaa;font-weight:600} .num{text-align:right;font-variant-numeric:tabular-nums}
 tr.pooled td{font-weight:600;border-top:1px solid #555}
 td.diag{background:#1f3a26} td.zero{color:#555}
 .note{color:#aaa;font-size:13px;white-space:normal;max-width:320px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
 .l-individual{background:#5b8def} .l-social{background:#e0a93b} .l-collaborative{background:#3fb66b} .l-absent{background:#a070d0} .l-unclear{background:#9a9a9a}
 #status{color:#8c8} #status.fail{color:#f66;font-weight:600}
</style></head><body>
<header>
 <a href="/">Coding page</a>
 <label>Coder A <select id="a"></select></label>
 <label>Coder B <select id="b"></select></label>
 <label>Session <select id="session"></select></label>
 <label>As of <input id="asof" type="datetime-local" step="1"></label>
 <span id="status"></span>
</header>
<main>
 <p class="meta">For reliability the second coder codes the same windows under their own name before looking at this page, which shows every coder's codes and notes. The kappas to report are those with As of set to a time before any discussion, which count only the codes saved before it. The disagreements are then resolved as the coder adjudicated, through each one's adjudicate link, and the coders' own files stay as they were.</p>
 <div id="summary"></div>
 <div id="confusion"></div>
 <div id="alpha"></div>
 <div id="disagreements"></div>
</main>
<script>
const $ = id => document.getElementById(id);
let codebook, sessions = [], coders = [], adjudicated = null, generation = 0;
function recall(name) { try { return localStorage.getItem(name); } catch (e) { return null; } }
function remember(name, value) { try { localStorage.setItem(name, value); } catch (e) {} }
function esc(text) { return String(text ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function attr(text) { return esc(text).replace(/"/g, '&quot;'); }
function said(text, failed) { $('status').textContent = text; $('status').classList.toggle('fail', !!failed); }
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) { const data = await r.json().catch(() => ({})); throw new Error(`${r.status} ${data.error || r.statusText}`.trim()); }
  return r.json();
}
function pct(x) { return x == null ? '–' : `${(100 * x).toFixed(1)} %`; }
function num(x) { return x == null ? '–' : (Math.abs(x) < 0.005 ? 0 : x).toFixed(2); }
function classOf(label) { return codebook.classes.find(c => c.label === label) || {key: '?', label, title: label}; }
function code(r) {
  const c = classOf(r.label), s = r.secondary ? classOf(r.secondary) : null;
  return `<i class="sw l-${attr(c.label)}"></i>${esc(`${c.key} ${c.title}`)}${s ? esc(` · also ${s.key} ${s.title}`) : ''}${r.note ? `<div class="note">${esc(r.note)}</div>` : ''}`;
}
function into(id, t) { const s = sessions.find(x => x.id === id); if (!s) return ''; const d = Math.max(0, Math.round(t - s.start)); return `${Math.floor(d / 60)}:${String(d % 60).padStart(2, '0')}`; }
// the cutoff the numbers are computed on, as the server used it: in the viewer's time and in UTC
function cutoff(iso) {
  if (!iso) return 'as of now';
  const d = new Date(iso), p = n => String(n).padStart(2, '0');
  return `as of ${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())} local time (${iso.replace('T', ' ').replace(/\.\d+Z$/, 'Z')})`;
}
function undated(data) { return data.undated ? `<p class="meta">${data.undated} line${data.undated === 1 ? '' : 's'} without a time left out by the cutoff</p>` : ''; }
function row(name, r, pooled) {
  return `<tr${pooled ? ' class="pooled"' : ''}><td>${esc(name)}</td><td class="num">${r.n}</td><td class="num">${pct(r.agree)}</td><td class="num">${num(r.kappa_codes)}</td>` +
    `<td class="num">${num(r.kappa_classes)} <span class="meta">(${r.n_classes})</span></td><td class="num">${num(r.kappa_collaborative)}</td><td class="num">${num(r.kappa_interaction)}</td><td class="num">${pct(r.lenient)}</td></tr>`;
}
function summary(data, only) {
  const html = `<h2>${esc(data.a)} and ${esc(data.b)}: the listed windows both coded, ${esc(cutoff(data.asof))}</h2>${undated(data)}`;
  if (!data.pooled.n) return html + `<p class="meta">${esc(data.a)} and ${esc(data.b)} have no listed window both coded${only ? ' in this session' : ''}.</p>`;
  let table = '<div class="scroll"><table><tr><th>session</th><th class="num">windows</th><th class="num">agree</th>' +
    '<th class="num" title="Cohen\'s kappa over the five codes">κ 5 codes</th><th class="num" title="Cohen\'s kappa over the three classes, on the windows both gave a class (their number)">κ 3 classes (windows)</th>' +
    '<th class="num" title="collaborative against individual and social, on the windows both gave a class">κ collaborative</th>' +
    '<th class="num" title="social or collaborative against individual (the classifier\'s binary κ), on the windows both gave a class">κ interaction</th>' +
    '<th class="num" title="a main state that equals the other coder\'s main state or also-state agrees">lenient</th></tr>';
  table += data.sessions.map(r => row(r.session, r)).join('');
  if (!only) table += row('pooled', data.pooled, true);
  return html + table + '</table></div>';
}
function confusion(data, only) {
  const m = data.pooled.confusion, codes = data.codes.map(classOf);
  const cols = codes.map((_, j) => m.reduce((sum, r) => sum + r[j], 0));
  let html = `<h2>Confusion${only ? '' : ', pooled'}: rows ${esc(data.a)}, columns ${esc(data.b)}</h2><div class="scroll"><table><tr><th></th>` +
    codes.map(c => `<th class="num"><i class="sw l-${attr(c.label)}"></i>${esc(`${c.key} ${c.label}`)}</th>`).join('') + '<th class="num">total</th></tr>';
  html += m.map((r, i) => `<tr><th><i class="sw l-${attr(codes[i].label)}"></i>${esc(`${codes[i].key} ${codes[i].label}`)}</th>` +
    r.map((v, j) => `<td class="num${i === j ? ' diag' : v ? '' : ' zero'}">${v}</td>`).join('') + `<td class="num">${r.reduce((a, b) => a + b, 0)}</td></tr>`).join('');
  return html + `<tr><th>total</th>${cols.map(v => `<td class="num">${v}</td>`).join('')}<td class="num">${data.pooled.n}</td></tr></table></div>`;
}
// each disagreement opens the coding page at its window for the remembered coder, or as the coder adjudicated
function disagreements(data) {
  if (!data.disagreements.length) return '<h2>Disagreements</h2><p class="meta">none</p>';
  let html = `<h2>Disagreements (${data.disagreements.length})</h2><div class="scroll"><table><tr><th>window</th><th>${esc(data.a)}</th><th>${esc(data.b)}</th><th>consensus</th></tr>`;
  html += data.disagreements.map(d => {
    const link = `/?session=${encodeURIComponent(d.session)}&start=${d.window_start}`;
    return `<tr><td><a href="${link}">${esc(d.session)} · ${into(d.session, d.window_start)}</a></td><td>${code(d.a)}</td><td>${code(d.b)}</td><td><a href="${link}&coder=adjudicated">adjudicate</a></td></tr>`;
  }).join('');
  return html + '</table></div>';
}
async function showAlpha(only, cut, gen) {
  if (coders.length < 3) { $('alpha').innerHTML = ''; return; }
  let data;
  try { data = await api(`/api/agreement?all=1${only ? `&session=${encodeURIComponent(only)}` : ''}${cut}`); }
  catch (e) { if (gen === generation) $('alpha').innerHTML = `<p class="meta">alpha not loaded: ${esc(e.message)}</p>`; return; }
  if (gen !== generation) return;
  let html = `<h2>Krippendorff's alpha, all ${data.coders.length} coders (nominal, five codes), ${esc(cutoff(data.asof))}</h2>${undated(data)}<div class="scroll"><table><tr><th>session</th><th class="num" title="listed windows coded by two coders or more">windows</th><th class="num">alpha</th></tr>`;
  html += data.sessions.map(r => `<tr><td>${esc(r.session)}</td><td class="num">${r.windows}</td><td class="num">${num(r.alpha)}</td></tr>`).join('');
  if (!only) html += `<tr class="pooled"><td>pooled</td><td class="num">${data.pooled.windows}</td><td class="num">${num(data.pooled.alpha)}</td></tr>`;
  $('alpha').innerHTML = html + '</table></div>';
}
async function show() {
  const a = $('a').value, b = $('b').value, only = $('session').value, gen = ++generation;
  remember('agreementA', a); remember('agreementB', b); remember('agreementSession', only);
  // the As of field is the viewer's local time, sent to the server in UTC; empty: the current codes
  const typed = $('asof').value, moment = typed ? new Date(typed) : null;
  if (moment && isNaN(moment.getTime())) { said('As of is not a date and time', true); return; }
  const cut = moment ? `&asof=${encodeURIComponent(moment.toISOString())}` : '';
  showAlpha(only, cut, gen);
  if (!a || !b || a === b) {
    $('summary').innerHTML = `<p class="meta">${coders.length < 2 ? 'Agreement needs a second coder, who codes the same windows under their own name.' : 'Pick two different coders.'}</p>`;
    $('confusion').innerHTML = $('disagreements').innerHTML = ''; return;
  }
  let data;
  try { data = await api(`/api/agreement?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}${only ? `&session=${encodeURIComponent(only)}` : ''}${cut}`); }
  catch (e) { if (gen === generation) said(`not loaded: ${e.message}`, true); return; }
  if (gen !== generation) return;
  said('');
  $('summary').innerHTML = summary(data, only);
  // two coders with no window in common have no confusion and no disagreement to show
  $('confusion').innerHTML = data.pooled.n ? confusion(data, only) : '';
  $('disagreements').innerHTML = data.pooled.n ? disagreements(data) : '';
}
(async () => {
  try {
    const boot = await api('/api/boot'); codebook = boot.codebook; sessions = boot.sessions;
    const listed = await api('/api/coders'); coders = listed.coders; adjudicated = listed.adjudicated;
  }
  catch (e) { said(`not loaded: ${e.message}`, true); return; }
  const names = coders.map(c => c.coder).concat(adjudicated ? [adjudicated.coder] : []);
  const option = c => `<option value="${attr(c.coder)}">${esc(c.coder)} (${c.windows})</option>`;
  // the consensus is offered apart from the coders and never picked by default
  $('a').innerHTML = $('b').innerHTML = coders.map(option).join('') + (adjudicated ? `<optgroup label="consensus">${option(adjudicated)}</optgroup>` : '');
  $('session').innerHTML = '<option value="">all sessions</option>' + sessions.map(s => `<option value="${attr(s.id)}">${esc(s.id)}</option>`).join('');
  const pick = (name, fallback) => { const v = recall(name); return v && names.includes(v) ? v : fallback; };
  // the two coders with the most windows, unless the browser remembers a pick
  $('a').value = pick('agreementA', coders[0] ? coders[0].coder : ''); $('b').value = pick('agreementB', coders[1] ? coders[1].coder : '');
  const s = recall('agreementSession'); $('session').value = s && sessions.some(x => x.id === s) ? s : '';
  for (const id of ['a', 'b', 'session', 'asof']) $(id).onchange = show;
  show();
})();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    settings: dict[str, Any] = {}
    sessions: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    text_source: Any = None  # a code_text.TextSource; None: no transcripts on the page
    lock = threading.Lock()  # one clip cut at a time
    write_lock = threading.Lock()  # label writes, apart from the clips so a save never waits for ffmpeg

    def log_message(self, format, *args):  # quiet
        pass

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _session(self, query) -> dict[str, Any] | None:
        wanted = (query.get('session') or [''])[0]
        return next((s for s in self.sessions if s['id'] == wanted), None)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        if url.path in ('/', '/agreement'):
            body = (PAGE if url.path == '/' else AGREEMENT_PAGE).encode()
            self.send_response(200)
            self.send_header('content-type', 'text/html; charset=utf-8')
            self.send_header('content-length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == '/api/coders':
            coders = self._coders()
            self._json({'coders': [row for row in coders if row['coder'] != ADJUDICATED],
                        'adjudicated': next((row for row in coders if row['coder'] == ADJUDICATED), None)})
        elif url.path == '/api/agreement':
            only = (query.get('session') or [''])[0] or None
            if only is not None and not self._session(query):
                return self._json({'error': 'no such session'}, 404)
            # a '+' of the zone typed into the URL unencoded arrives as a space: it is put back
            asof_text = re.sub(r'(:\d{2}(?:[.,]\d+)?) (\d{2}:?\d{2})$', r'\1+\2', (query.get('asof') or [''])[0].strip())
            asof = parse_time(asof_text) if asof_text else None
            if asof_text and (asof is None or asof.tzinfo is None):
                return self._json({'error': 'give asof=<ISO 8601 time with its zone, e.g. 2026-09-23T12:00:00Z>'}, 400)
            if (query.get('all') or ['0'])[0] not in ('', '0'):
                return self._json(self._alpha(only, asof))
            a, b = ((query.get(name) or [''])[0].strip() for name in ('a', 'b'))
            if not a or not b or safe_name(a) == safe_name(b):
                return self._json({'error': 'give two different coders: a=<name>&b=<name>'}, 400)
            error = name_error(a) or name_error(b)
            if error:
                return self._json({'error': error}, 400)
            self._json(self._agreement(a, b, only, asof))
        elif url.path == '/api/boot':
            self._json({'codebook': CODEBOOK, 'sessions': [{k: v for k, v in s.items() if k != 'dir'} for s in self.sessions],
                        'hidden': self.hidden})
        elif url.path == '/api/windows':
            session = self._session(query)
            if not session:
                return self._json({'error': 'no such session'}, 404)
            coder = (query.get('coder') or ['anonymous'])[0]
            if name_error(coder):
                return self._json({'error': name_error(coder)}, 400)
            labels = self._labels(session, coder)
            self._json({'windows': self._windows(session), 'labels': list(labels.values())})
        elif url.path == '/api/progress':
            coder = (query.get('coder') or [''])[0].strip()
            if not coder:
                return self._json({'error': 'give coder=<name>'}, 400)
            if name_error(coder):
                return self._json({'error': name_error(coder)}, 400)
            self._json({'progress': self._progress(coder)})
        elif url.path == '/api/text':
            session = self._session(query)
            if not session:
                return self._json({'error': 'no such session'}, 404)
            try:
                start = float(query['start'][0])
            except (KeyError, IndexError, ValueError):
                return self._json({'error': 'give start=<window start>'}, 400)
            if self.text_source is None:
                return self._json({'start': start, 'lines': [], 'note': 'transcripts are not served by this page'})
            self._json(self.text_source.text(session, start, prefetch=bool(query.get('prefetch'))))
        elif url.path == '/clip':
            session = self._session(query)
            if not session:
                return self._json({'error': 'no such session'}, 404)
            start = float(query['start'][0])
            try:
                with self.lock:
                    path = make_clip(session, start, self.settings['window'])
            except subprocess.SubprocessError as error:
                return self._json({'error': f'ffmpeg failed: {error}'}, 500)
            if query.get('prefetch'):  # the clip is cut and cached; nothing to send
                self.send_response(204)
                self.send_header('content-length', '0')
                self.end_headers()
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('content-type', 'video/mp4')
            self.send_header('content-length', str(len(data)))
            self.send_header('cache-control', 'max-age=3600')
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({'error': 'not found'}, 404)

    def _labels_path(self, session: dict[str, Any], coder: str) -> Path:
        return Path(session['dir']) / 'labels' / f'{safe_name(coder)}.jsonl'

    def _labels(self, session: dict[str, Any], coder: str, asof: datetime | None = None,
                undated: list[int] | None = None) -> dict[str, dict[str, Any]]:
        """`coder`'s labels in `session` (replay), those saved by `asof` when it is given; the lines
        left out for having no time are added to undated[0]."""
        path = self._labels_path(session, coder)
        if not path.exists():
            return {}
        labels, left_out = replay(path.read_text(encoding='utf-8', errors='replace'), asof)
        if undated is not None:
            undated[0] += left_out
        return labels

    def _windows(self, session: dict[str, Any]) -> list[dict[str, float]]:
        s = self.settings
        return windows_of(session, s['window'], s['step'], s['sample'], s['block'], s['seed'])

    def _listed(self, session: dict[str, Any]) -> set[str]:
        """the keys (as _labels keys them) of the windows the page lists for `session`"""
        return {f"{w['start']:.3f}" for w in self._windows(session)}

    def _progress(self, coder: str) -> dict[str, dict[str, int]]:
        """per listed session, its listed windows and how many of them `coder` has a label for now
        (after the undo lines); a label of a window no longer listed is not counted."""
        progress = {}
        for session in self.sessions:
            listed, labels = self._listed(session), self._labels(session, coder)
            progress[session['id']] = {'coded': len(listed & set(labels)), 'windows': len(listed)}
        return progress

    def _coders(self) -> list[dict[str, Any]]:
        """every coder with a labels file in a listed session, by the file's name (a name
        _labels_path makes), with how many listed windows they have a current label for (as
        /api/progress counts them) and in how many sessions, most windows first; the adjudicated
        file is among them."""
        found: dict[str, dict[str, int]] = {}
        for session in self.sessions:
            folder = Path(session['dir']) / 'labels'
            files = [path for path in sorted(folder.glob('*.jsonl')) if safe_name(path.stem) == path.stem] \
                if folder.is_dir() else []  # a file of another name is not one this page writes
            listed = self._listed(session) if files else set()
            for path in files:
                n = len(listed & set(self._labels(session, path.stem)))
                row = found.setdefault(path.stem, {'windows': 0, 'sessions': 0})
                row['windows'] += n
                row['sessions'] += n > 0
        rows = [{'coder': name, **row} for name, row in found.items()]
        return sorted(rows, key=lambda row: (-row['windows'], row['coder']))

    def _agreement(self, a: str, b: str, only: str | None = None, asof: datetime | None = None) -> dict[str, Any]:
        """coders a and b over the listed windows both have a label for, now or as saved by `asof`
        (keyed as _labels keys them): pair_agreement per session (every session both coded, or the
        one asked for) and pooled, and every window they disagree on, in time order per session.
        `asof` is echoed in UTC, and `undated` counts the lines it left out for having no time."""
        rows, pooled, disagreements, undated = [], [], [], [0]
        for session in self.sessions:
            if only is not None and session['id'] != only:
                continue
            first, second = self._labels(session, a, asof, undated), self._labels(session, b, asof, undated)
            pairs = []
            for key in sorted(set(first) & set(second) & self._listed(session), key=float):
                x, y = first[key], second[key]
                if x.get('label') not in CODES or y.get('label') not in CODES:
                    continue
                pairs.append((x, y))
                if x['label'] != y['label']:
                    disagreements.append({'session': session['id'], 'window_start': float(key),
                                          'window_end': x.get('window_end', y.get('window_end')),
                                          'a': {k: x.get(k) for k in ('label', 'secondary', 'note')},
                                          'b': {k: y.get(k) for k in ('label', 'secondary', 'note')}})
            if pairs or only is not None:
                rows.append({'session': session['id'], **pair_agreement(pairs)})
            pooled += pairs
        return {'a': a, 'b': b, 'codes': list(CODES), 'asof': utc_text(asof), 'undated': undated[0],
                'sessions': rows, 'pooled': pair_agreement(pooled), 'disagreements': disagreements}

    def _alpha(self, only: str | None = None, asof: datetime | None = None) -> dict[str, Any]:
        """Krippendorff's alpha (nominal, the five codes, missing allowed) over every coder but the
        adjudicated file, on the listed windows, now or as saved by `asof`, per session (every
        session with a window two of them coded, or the one asked for) and pooled; `windows`
        counts the windows coded by two coders or more."""
        coders = [row['coder'] for row in self._coders() if row['coder'] != ADJUDICATED]
        rows, pooled, undated = [], [], [0]
        for session in self.sessions:
            if only is not None and session['id'] != only:
                continue
            by = [self._labels(session, coder, asof, undated) for coder in coders]
            keys = sorted(set().union(*by) & self._listed(session), key=float) if by else []
            units = [[labels[key]['label'] for labels in by if key in labels and labels[key].get('label') in CODES]
                     for key in keys]
            units = [unit for unit in units if len(unit) >= 2]
            if units or only is not None:
                rows.append({'session': session['id'], 'alpha': krippendorff_alpha(units), 'windows': len(units)})
            pooled += units
        return {'coders': coders, 'asof': utc_text(asof), 'undated': undated[0], 'sessions': rows,
                'pooled': {'alpha': krippendorff_alpha(pooled), 'windows': len(pooled)}}

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        length = int(self.headers.get('content-length') or 0)
        try:
            record = json.loads(self.rfile.read(length) or b'{}')
            float(record['window_start'])
        except (ValueError, KeyError, TypeError):
            return self._json({'error': 'the body is not a record with a window_start'}, 400)
        session = next((s for s in self.sessions if s['id'] == record.get('session')), None)
        if not session:
            return self._json({'error': 'no such session'}, 404)
        coder = record.get('coder', '')
        if not isinstance(coder, str):
            return self._json({'error': 'the coder is not a name'}, 400)
        if url.path in ('/api/label', '/api/unlabel') and name_error(coder):
            return self._json({'error': name_error(coder)}, 400)
        if url.path == '/api/label':
            # a record without a known label would read back as an undo line
            if record.get('label') not in [c['label'] for c in CODEBOOK['classes']]:
                return self._json({'error': f"unknown label {record.get('label')!r}"}, 400)
            error = secondary_error(record)
            if error:
                return self._json({'error': error}, 400)
            if record.get('secondary') is None:  # no also-state: the record carries neither field
                record.pop('secondary', None)
                record.pop('secondary_key', None)
            path = self._labels_path(session, coder)
            record['saved_at'] = now_utc()  # the server's own time, which an as-of cutoff reads
        elif url.path == '/api/unlabel':
            path = self._labels_path(session, coder)
            record = {'session': session['id'], 'window_start': record['window_start'], 'label': None,
                      'undone_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'saved_at': now_utc()}
        else:
            return self._json({'error': 'not found'}, 404)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.write_lock, path.open('a+b') as file:
                # a line cut short by a crash is ended first, so a confirmed record is always a line of its own
                end = file.seek(0, os.SEEK_END)
                if end:
                    file.seek(end - 1)
                    if file.read(1) != b'\n':
                        file.write(b'\n')
                file.write((json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8'))
        except OSError as error:  # the page says NOT saved and keeps the window as it was
            return self._json({'error': f'the labels cannot be written: {error.strerror or error}'}, 500)
        self._json({'ok': True})


def prepare_text(source: TextSource, sessions: list[dict[str, Any]], settings: dict[str, Any]) -> int:
    """translate every listed window of every session into the cache; 1 when the model or a
    session's transcripts cannot be had."""
    error = source.translator.load()
    if error:
        print(error)
        return 1
    failed = 0
    for session in sessions:
        starts = [w['start'] for w in windows_of(session, settings['window'], settings['step'], settings['sample'],
                                                 settings['block'], settings['seed'])]
        try:
            done, kept = source.prepare(session, starts)
        except Exception as error:  # one session's missing transcripts must not stop the others
            print(f"{session['id']}: not prepared ({type(error).__name__}: {error})")
            failed += 1
            continue
        print(f"{session['id']}: {done} windows translated, {kept} cached already or silent")
    return 1 if failed else 0


def get_parser():
    parser = argparse.ArgumentParser(prog='mmla ses-code', description="Code a session's windows by hand in the browser.")
    parser.add_argument('-a', '--artifacts', default=None, help="artifacts root (default <cwd>/artifacts)")
    parser.add_argument('-s', '--sessions', default=None, help="only sessions whose id contains this text")
    parser.add_argument('-w', '--window', type=float, default=10.0, help="window length in seconds (default 10)")
    parser.add_argument('-st', '--step', type=float, default=10.0, help="step between windows (default 10)")
    parser.add_argument('--sample', type=float, default=1.0, help="share of five-minute blocks to code, 0 to 1 (default 1: everything)")
    parser.add_argument('--block', type=float, default=300.0, help="block length the sampling draws from (default 300 s)")
    parser.add_argument('--seed', type=int, default=1, help="sampling seed, the same for every coder (default 1)")
    parser.add_argument('--all', dest='show_all', action='store_true',
                        help="list every session, also those the inclusion rule S1 leaves out of the analysis")
    parser.add_argument('--influx-config', default=None,
                        help=f"config whose InfluxDB section holds the transcripts (default <cwd>/{DEFAULT_INFLUX_CONFIG})")
    parser.add_argument('--context', type=float, default=CONTEXT,
                        help=f"seconds of speech shown before and after the window (default {CONTEXT:g})")
    parser.add_argument('--prepare-text', action='store_true',
                        help="translate the transcript of every listed window into the cache, then exit")
    parser.add_argument('--threads', type=int, default=4, help="CPU threads of the translation model (default 4)")
    parser.add_argument('-p', '--port', type=int, default=8765)
    parser.add_argument('--bind', default='127.0.0.1', help="address to listen on (default 127.0.0.1: this machine only; a Tailscale address or 0.0.0.0 lets others in, mind who can reach it)")
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    artifacts = Path(args.artifacts or os.path.join(os.getcwd(), 'artifacts')).resolve()
    Handler.sessions, Handler.hidden = load_sessions(artifacts, args.sessions, args.show_all)
    Handler.settings = {'window': args.window, 'step': args.step, 'sample': args.sample, 'block': args.block, 'seed': args.seed}
    influx_config = args.influx_config or os.path.join(os.getcwd(), DEFAULT_INFLUX_CONFIG)
    Handler.text_source = TextSource(args.window, args.context, influx_config, Translator(threads=args.threads))
    for s in Handler.hidden:
        print(f"hidden: {s['id']}: {s['reason']}")
    for s in Handler.sessions:
        if s['included'] is False:
            print(f"shown (--all): {s['id']}: {s['reason']}")
        elif s['included'] is None:
            print(f"kept, not judged: {s['id']}: {s['inclusion']}")
    if not Handler.sessions:
        if Handler.hidden:
            print(f"every session with video under {artifacts} is left out by S1; --all lists them")
        else:
            print(f"no sessions with video under {artifacts}")
        return 1
    if args.prepare_text:
        return prepare_text(Handler.text_source, Handler.sessions, Handler.settings)
    total = sum(len(windows_of(s, args.window, args.step, args.sample, args.block, args.seed)) for s in Handler.sessions)
    print(f"{len(Handler.sessions)} sessions ({len(Handler.hidden)} hidden by S1), {total} windows to code; open http://{args.bind if args.bind != '0.0.0.0' else '<this machine>'}:{args.port}/  (Ctrl-C stops)")
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
