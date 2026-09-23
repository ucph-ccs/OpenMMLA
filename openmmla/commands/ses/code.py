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
import subprocess
import sys
import threading
import time
import urllib.parse
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
         'definition': "The group is at its place but you cannot tell its state: members are out of frame and inaudible, or the window is a transition with no dominant state."},
    ],
    'rule': 'Label the group as a whole with the state that fills most of the ten seconds. Use the preceding windows as context. If two members collaborate while a third works alone, it is still collaborative interaction. The teacher\'s talk does not make a window social or collaborative: code what the members do with each other and with the shared artifact' + TEACHER_NOTE + '. Members looking at the shared artifact while the teacher talks, with no member working on it, is individual work: they follow the teacher, not each other, so it is not the joint attention of collaborative interaction. One member working on it while another follows is collaborative.',
}
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
 textarea{width:100%;height:60px;background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px}
 #status{color:#8c8} #status.fail{color:#f66;font-weight:600}
 #text{margin-top:12px;padding:10px 12px;background:#1b1b1b;border-radius:8px;font-size:14px}
 #text .line{margin:0 0 10px} #text .who{color:#8ab4f8;font-size:12px;margin-right:6px}
 #text .ctx{color:#777} #text .en{color:#bbb;font-style:italic;margin-top:2px}
 #text .approx{color:#d9a441;font-size:12px;margin-left:6px}
</style></head><body>
<header>
 <div id="hidden" class="meta" style="flex-basis:100%;display:none"></div>
 <label>Coder <input id="coder" size="10"></label>
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
  <div class="meta">Keys: <b>1</b> <b>2</b> <b>3</b> <b>4</b> <b>0</b> label (and move on when Auto-advance is on) · <b>a</b> auto-advance · <b>space</b> replay · <b>←</b> <b>→</b> move · <b>n</b> note · <b>u</b> undo · <b>t</b> transcript</div>
  <div id="text" style="display:none"></div>
 </div>
 <div class="keys">
  <div id="classes"></div>
  <div class="def" id="rule"></div>
  <textarea id="note" placeholder="note (optional), saved with the next class key; Enter re-saves the note of a coded window"></textarea>
 </div>
</main>
<script>
const $ = id => document.getElementById(id);
let codebook, sessions, session, windows = [], labels = {}, index = 0, shownAt = 0, showText = false, textAbort = null, textRetry = null;
let progress = {}, autoAdvance = true, saving = false, savingWindow = null, advancedAt = 0, undoStack = [], generation = 0;
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
function said(text, failed) { $('status').textContent = text; $('status').classList.toggle('fail', !!failed); }
function clock() { return new Date().toTimeString().slice(0, 8); }
function key(w) { return w.start.toFixed(3); }
function recall(name) { try { return localStorage.getItem(name); } catch (e) { return null; } }
function remember(name, value) { try { localStorage.setItem(name, value); } catch (e) {} }
function esc(text) { return String(text || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function attr(text) { return esc(text).replace(/"/g, '&quot;'); }
function part(text, cls) { return text ? `<span class="${cls}">${esc(text)}</span> ` : ''; }
function classOf(l) { return codebook.classes.find(c => c.label === l.label) || {key: l.key, label: l.label, title: l.label}; }
function coderName() { return $('coder').value.trim() || 'anonymous'; }
// a pupil's line is what that pupil's worn mic transcribed, the partner and the teacher included; it names no speaker
function speaker(s) { return /^pupil /.test(s) ? `${s}'s mic` : s; }
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
  for (const l of lines) {
    html += `<div class="line"><div><span class="who">${esc(speaker(l.speaker))}</span>${l.approximate ? '<span class="approx">approximate: the chunk has no word times</span>' : ''}</div>`;
    html += `<div>${part(l.before, 'ctx')}${part(l.inside, 'in')}${part(l.after, 'ctx')}</div>`;
    if (l.en) html += `<div class="en${l.inside ? '' : ' ctx'}">${esc(l.en)}</div>`;
    html += '</div>';
  }
  if (data.note) html += `<div class="meta">${esc(data.note)}</div>`;
  box.innerHTML = html;
  // the model is still loading: ask again for this window, and prefetch nothing until it is ready
  if (data.pending) { textRetry = setTimeout(() => renderText(true), 3000); return; }
  for (const next of windows.slice(index + 1, index + 3)) fetch(`/api/text?session=${encodeURIComponent(id)}&start=${next.start}&prefetch=1`, {signal: abort.signal}).catch(() => {});
}
function render() {
  const w = windows[index]; if (!w) return renderMarks();
  const v = $('video'); v.src = `/clip?session=${session.id}&start=${w.start}`; v.load(); v.play().catch(() => {});
  shownAt = Date.now();
  const current = labels[key(w)];
  $('note').value = current ? (current.note || '') : '';
  for (const next of windows.slice(index + 1, index + 4)) fetch(`/clip?session=${session.id}&start=${next.start}&prefetch=1`);
  renderMarks();
  renderText();
}
// what the codes show: the buttons, the coded line, the strip and the counts; the clip and the transcript stay
function renderMarks() {
  const w = windows[index], current = w && labels[key(w)];
  const done = windows.filter(x => labels[key(x)]).length;
  progress[session.id] = {coded: done, windows: windows.length}; showProgress();
  $('progress').textContent = `${done} of ${windows.length} coded`;
  $('when').textContent = w ? `window ${index + 1} of ${windows.length} · ${new Date(w.start * 1000).toISOString().replace('T', ' ').slice(0, 19)}Z · ${Math.round(w.start - session.start)} s into the session` : 'no windows to code in this session';
  const c = current && classOf(current);
  $('coded').innerHTML = current ? `<i class="sw l-${attr(c.label)}"></i>${esc(`coded: ${c.key} ${c.title}${current.note ? ` · note: ${current.note}` : ''}`)}` : w ? 'not coded yet' : '';
  $('coded').classList.toggle('none', !current);
  document.querySelectorAll('#classes button').forEach(b => b.classList.toggle('active', !!current && b.dataset.label === current.label));
  $('strip').innerHTML = windows.map((x, i) => {
    const l = labels[key(x)], title = l ? `window ${i + 1} · ${classOf(l).title}${l.note ? ` · ${l.note}` : ''}` : `window ${i + 1} · not coded`;
    return `<span class="cell${l ? ` l-${attr(l.label)}` : ''}${i === index ? ' here' : ''}" data-i="${i}" title="${attr(title)}"></span>`;
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
  remember('coder', coder);
  const record = {session: session.id, window_start: w.start, window_end: w.end, label: cls.label, key: cls.key,
                  note: $('note').value.trim(), coder, coded_at: new Date().toISOString(), seconds_spent: Math.round((Date.now() - shownAt) / 100) / 10};
  const gen = generation, at = index;
  saving = true; savingWindow = w; said('saving…');
  try { await post('/api/label', record); }
  catch (e) { said(`NOT saved: ${why(e)} — try again`, true); return; }
  finally { saving = false; savingWindow = null; }
  said(`saved ✓ ${clock()} · ${index !== at ? `window ${at + 1} · ` : ''}${cls.title}`);
  if (coder === $('coder').value.trim()) remembered(record);
  // another session or coder was loaded while the save was out
  if (gen !== generation || session.id !== record.session) return;
  labels[key(w)] = record;
  if (autoAdvance && index === at) {
    // the next window after it not yet coded, else simply the next one
    const next = windows.findIndex((x, i) => i > at && !labels[key(x)]);
    index = next >= 0 ? next : Math.min(at + 1, windows.length - 1);
    if (index !== at) { advancedAt = Date.now(); return render(); }
  }
  renderMarks();
}
async function saveNote() {
  const w = windows[index], current = w && labels[key(w)], note = $('note').value.trim();
  if (!current || note === (current.note || '')) return;
  if (saving) { said('NOT saved: the previous save is still waiting for the server — try again', true); return; }
  const coder = $('coder').value.trim(); if (!coder) { said('type your name as coder first', true); return; }
  const record = {...current, note, coder, edited_at: new Date().toISOString()}, gen = generation;
  saving = true; said('saving…');
  try { await post('/api/label', record); }
  catch (e) { said(`NOT saved: ${why(e)} — try again`, true); return; }
  finally { saving = false; }
  said(`saved ✓ ${clock()} · ${classOf(record).title} · note`);
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
  saving = true; said('undoing…');
  try { await post('/api/unlabel', {session: target.session, window_start: target.window_start, coder}); }
  catch (e) { said(`NOT undone: ${why(e)} — try again`, true); return; }
  finally { saving = false; }
  undoStack = undoStack.filter(u => !(u.session === target.session && u.window_start === target.window_start));
  if (gen !== generation || session.id !== target.session) { said(`undone ✓ ${clock()}`); return; }
  const i = windows.findIndex(x => key(x) === k);
  said(`undone ✓ ${clock()} · window ${i + 1}${old ? `, was ${classOf(old).title}` : ''}`);
  delete labels[k];
  if (i >= 0 && i !== index) { index = i; render(); } else renderMarks();
}
async function loadSession(id) {
  const gen = ++generation, next = sessions.find(s => s.id === id), coder = coderName();
  let data;
  try { data = await api(`/api/windows?session=${encodeURIComponent(id)}&coder=${encodeURIComponent(coder)}`); }
  // the list goes back to the session still on screen, so it names what is coded and a second pick retries
  catch (e) { if (gen === generation) { said(`session not loaded: ${why(e)}`, true); if (session) $('session').value = session.id; } return; }
  if (gen !== generation) return;
  session = next; windows = data.windows; labels = {}; for (const l of data.labels) labels[l.window_start.toFixed(3)] = l;
  index = windows.findIndex(w => !labels[key(w)]); if (index < 0) index = 0;
  remember('session', id);
  render();
}
document.addEventListener('keydown', e => {
  if (e.target === $('note')) { if (e.key === 'Enter') { e.preventDefault(); $('note').blur(); saveNote(); } return; }
  // Enter commits the name: the blur fires its change event, and the keys reach the page again
  if (e.target === $('coder')) { if (e.key === 'Enter') { e.preventDefault(); $('coder').blur(); } return; }
  if (!codebook || e.metaKey || e.ctrlKey || e.altKey) return;
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
  $('classes').innerHTML = codebook.classes.map(c => `<button data-label="${c.label}" onclick="this.blur(); label(codebook.classes.find(x=>x.key==='${c.key}'))"><b>${c.key}</b><i class="sw l-${c.label}"></i>${c.title}</button><div class="def">${c.definition}</div>`).join('');
  $('rule').textContent = codebook.rule;
  $('coder').value = recall('coder') || '';
  showText = recall('showText') === '1'; $('texttoggle').classList.toggle('active', showText);
  autoAdvance = recall('autoAdvance') !== '0'; $('advancetoggle').classList.toggle('active', autoAdvance);
  $('session').innerHTML = sessions.map(s => `<option value="${attr(s.id)}" title="${attr(s.inclusion)}"></option>`).join(''); showProgress();
  if (boot.hidden.length) {
    $('hidden').textContent = `${boot.hidden.length} session${boot.hidden.length === 1 ? '' : 's'} hidden: left out of the analysis by the inclusion rule S1`;
    $('hidden').title = boot.hidden.map(h => `${h.id}: ${h.reason}`).join('\n');
    $('hidden').style.display = '';
  }
  $('session').onchange = e => { e.target.blur(); loadSession(e.target.value); };
  $('coder').onchange = () => { const coder = $('coder').value.trim(); if (coder) remember('coder', coder); undoStack = []; loadProgress(); loadSession($('session').value); };
  $('strip').onclick = e => { const i = e.target.dataset.i; if (i !== undefined && +i !== index) { index = +i; render(); } };
  const last = recall('session');
  if (last && sessions.some(s => s.id === last)) $('session').value = last;
  loadProgress(); loadSession($('session').value);
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
        if url.path == '/':
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('content-type', 'text/html; charset=utf-8')
            self.send_header('content-length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == '/api/boot':
            self._json({'codebook': CODEBOOK, 'sessions': [{k: v for k, v in s.items() if k != 'dir'} for s in self.sessions],
                        'hidden': self.hidden})
        elif url.path == '/api/windows':
            session = self._session(query)
            if not session:
                return self._json({'error': 'no such session'}, 404)
            s = self.settings
            windows = windows_of(session, s['window'], s['step'], s['sample'], s['block'], s['seed'])
            coder = (query.get('coder') or ['anonymous'])[0]
            labels = self._labels(session, coder)
            self._json({'windows': windows, 'labels': list(labels.values())})
        elif url.path == '/api/progress':
            coder = (query.get('coder') or [''])[0].strip()
            if not coder:
                return self._json({'error': 'give coder=<name>'}, 400)
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
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in coder) or 'anonymous'
        return Path(session['dir']) / 'labels' / f'{safe}.jsonl'

    def _labels(self, session: dict[str, Any], coder: str) -> dict[str, dict[str, Any]]:
        path = self._labels_path(session, coder)
        labels: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = f"{float(record['window_start']):.3f}"
                except (ValueError, KeyError, TypeError):  # a line cut short by a crash, skipped as labels._read_coder does
                    continue
                if record.get('label') is None:
                    labels.pop(key, None)  # an undo line
                else:
                    labels[key] = record
        return labels

    def _progress(self, coder: str) -> dict[str, dict[str, int]]:
        """per listed session, its listed windows and how many of them `coder` has a label for now
        (after the undo lines); a label of a window no longer listed is not counted."""
        s = self.settings
        progress = {}
        for session in self.sessions:
            windows = windows_of(session, s['window'], s['step'], s['sample'], s['block'], s['seed'])
            labels = self._labels(session, coder)
            progress[session['id']] = {'coded': sum(f"{w['start']:.3f}" in labels for w in windows),
                                       'windows': len(windows)}
        return progress

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
        if url.path == '/api/label':
            # a record without a known label would read back as an undo line
            if record.get('label') not in [c['label'] for c in CODEBOOK['classes']]:
                return self._json({'error': f"unknown label {record.get('label')!r}"}, 400)
            path = self._labels_path(session, record.get('coder', ''))
        elif url.path == '/api/unlabel':
            path = self._labels_path(session, record.get('coder', ''))
            record = {'session': session['id'], 'window_start': record['window_start'], 'label': None, 'undone_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
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
