"""mmla ses-code: a browser page to code a session's 10-second windows by hand.

Ground truth for the interaction classes comes from a person watching the recordings, ten seconds
at a time. This command serves the sessions under artifacts/: for every window it cuts a clip on
demand with ffmpeg (up to two cameras side by side with the group microphone, exact to the
window's start), and the page plays it and takes one key per window (1 individual or parallel
work, 2 social interaction, 3 collaborative interaction, 0 unclear, with an optional note).
Labels go to artifacts/<session>/labels/<coder>.jsonl, one line per window, so a second coder
writes a second file and the two are compared for agreement. Clips are cached under
artifacts/<session>/labels/clips/ and can be deleted at any time.
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

CODEBOOK = {
    'classes': [
        {'key': '1', 'label': 'individual', 'title': 'Individual or parallel work',
         'definition': 'Nobody in the group interacts with another member during most of the window: each works alone, waits, or watches the teacher. Glances without exchange do not count.'},
        {'key': '2', 'label': 'social', 'title': 'Social interaction',
         'definition': 'Members interact (talk, gesture, look at each other) but not about the task: chat, jokes, phones, waiting together.'},
        {'key': '3', 'label': 'collaborative', 'title': 'Collaborative interaction',
         'definition': 'Members interact about the task: talking about it, joint attention on the shared artifact (micro:bit, microscope, tablet, sheet), pointing, handing over, working on one thing together, explaining or asking.'},
        {'key': '0', 'label': 'unclear', 'title': 'Unclear',
         'definition': 'Cannot tell: the group is out of view, the audio is missing, or the window is a transition with no dominant state.'},
    ],
    'rule': 'Label the group as a whole with the state that fills most of the ten seconds. Use the preceding windows as context. If two members collaborate while a third works alone, it is still collaborative interaction.',
}
AUDIO_PREFERENCE = ('jabra-0', 'vimo-0-ch0', 'vimo-0', 'badge-0')
CAMERA_PREFERENCE = ('c920-01', 'c920-04', 'c920-05', 'c920-06', 'c920-02', 'c920-03')


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def load_sessions(artifacts: Path, pattern: str | None = None) -> list[dict[str, Any]]:
    sessions = []
    for session_dir in sorted(artifacts.glob('exp_*')):
        if pattern and pattern not in session_dir.name:
            continue
        manifest = _read(session_dir / 'manifest.json')
        recordings = [r for r in manifest.get('recordings', []) if os.path.exists(r.get('path', ''))]
        videos = [r for r in recordings if r['modality'] == 'video']
        audios = [r for r in recordings if r['modality'] == 'audio']
        if not videos:
            continue
        videos.sort(key=lambda r: (CAMERA_PREFERENCE.index(r['device']) if r['device'] in CAMERA_PREFERENCE else 99, r['device']))
        audios.sort(key=lambda r: (AUDIO_PREFERENCE.index(r['device']) if r['device'] in AUDIO_PREFERENCE else 99, r['device']))
        start = max(r['start_time'] for r in recordings)
        end = min(r['start_time'] + (r.get('duration') or 0) for r in recordings if r.get('duration'))
        sessions.append({'id': session_dir.name, 'dir': str(session_dir), 'start': start, 'end': end,
                         'videos': videos[:2], 'audio': audios[0] if audios else None,
                         'experiment': manifest.get('experiment_id'), 'group': manifest.get('group_id')})
    return sessions


def windows_of(session: dict[str, Any], window: float, step: float, sample: float, block: float, seed: int) -> list[dict[str, float]]:
    """the windows to code, in time order; `sample` < 1 keeps that share of `block`-second blocks,
    drawn with a fixed seed so two coders see the same windows"""
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
 main{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:16px}
 video{width:100%;background:#000;border-radius:8px}
 .keys button{display:block;width:100%;text-align:left;margin:6px 0;padding:10px}
 .keys b{display:inline-block;width:26px;height:26px;line-height:26px;text-align:center;background:#333;border-radius:5px;margin-right:8px}
 .def{color:#aaa;font-size:13px;margin:2px 0 10px 34px}
 .meta{color:#aaa;font-size:13px} .bar{height:6px;background:#333;border-radius:3px;margin:8px 0} .bar div{height:6px;background:#2d6cdf;border-radius:3px}
 textarea{width:100%;height:60px;background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px}
 #status{color:#8c8}
</style></head><body>
<header>
 <label>Coder <input id="coder" size="10"></label>
 <label>Session <select id="session"></select></label>
 <span id="progress" class="meta"></span><span id="status"></span>
</header>
<main>
 <div>
  <video id="video" controls autoplay playsinline></video>
  <div class="meta" id="when"></div>
  <div class="bar"><div id="fill" style="width:0"></div></div>
  <div class="meta">Keys: <b>1</b> <b>2</b> <b>3</b> <b>0</b> label and go on · <b>space</b> replay · <b>←</b> <b>→</b> move · <b>n</b> note · <b>u</b> undo the last label</div>
 </div>
 <div class="keys">
  <div id="classes"></div>
  <div class="def" id="rule"></div>
  <textarea id="note" placeholder="note (optional), then press Enter or the class key"></textarea>
  <div class="meta" id="last"></div>
 </div>
</main>
<script>
const $ = id => document.getElementById(id);
let codebook, sessions, session, windows = [], labels = {}, index = 0, shownAt = 0;
async function api(path, options) { const r = await fetch(path, options); return r.json(); }
function key(w) { return w.start.toFixed(3); }
function render() {
  const w = windows[index]; if (!w) return;
  const v = $('video'); v.src = `/clip?session=${session.id}&start=${w.start}`; v.load(); v.play().catch(() => {});
  shownAt = Date.now();
  const at = new Date(w.start * 1000);
  const done = Object.keys(labels).length;
  $('when').textContent = `window ${index + 1} of ${windows.length} · ${at.toISOString().replace('T', ' ').slice(0, 19)}Z · ${Math.round(w.start - session.start)} s into the session`;
  $('progress').textContent = `${done} of ${windows.length} coded`;
  $('fill').style.width = `${100 * done / windows.length}%`;
  const current = labels[key(w)];
  document.querySelectorAll('#classes button').forEach(b => b.classList.toggle('active', !!current && b.dataset.label === current.label));
  $('note').value = current ? (current.note || '') : '';
  for (const next of windows.slice(index + 1, index + 4)) fetch(`/clip?session=${session.id}&start=${next.start}&prefetch=1`);
}
async function label(cls) {
  const w = windows[index]; if (!w) return;
  const coder = $('coder').value.trim(); if (!coder) { $('status').textContent = 'type your name as coder first'; return; }
  localStorage.setItem('coder', coder);
  const record = {session: session.id, window_start: w.start, window_end: w.end, label: cls.label, key: cls.key,
                  note: $('note').value.trim(), coder, coded_at: new Date().toISOString(), seconds_spent: Math.round((Date.now() - shownAt) / 100) / 10};
  labels[key(w)] = record;
  await api('/api/label', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(record)});
  $('last').textContent = `saved: ${cls.title}`;
  index = Math.min(index + 1, windows.length - 1);
  while (index < windows.length - 1 && labels[key(windows[index])]) index++;
  render();
}
async function loadSession(id) {
  session = sessions.find(s => s.id === id);
  const coder = $('coder').value.trim() || 'anonymous';
  const data = await api(`/api/windows?session=${id}&coder=${encodeURIComponent(coder)}`);
  windows = data.windows; labels = {}; for (const l of data.labels) labels[l.window_start.toFixed(3)] = l;
  index = windows.findIndex(w => !labels[key(w)]); if (index < 0) index = 0;
  localStorage.setItem('session', id);
  render();
}
document.addEventListener('keydown', e => {
  if (e.target === $('note') && e.key !== 'Enter') return;
  if (e.target === $('note') && e.key === 'Enter') { e.preventDefault(); $('note').blur(); return; }
  if (e.target === $('coder')) return;
  const cls = codebook.classes.find(c => c.key === e.key);
  if (cls) { e.preventDefault(); label(cls); return; }
  if (e.key === ' ') { e.preventDefault(); $('video').currentTime = 0; $('video').play(); }
  if (e.key === 'ArrowLeft') { index = Math.max(0, index - 1); render(); }
  if (e.key === 'ArrowRight') { index = Math.min(windows.length - 1, index + 1); render(); }
  if (e.key === 'n') { e.preventDefault(); $('note').focus(); }
  if (e.key === 'u') { const prev = index - 1; if (prev >= 0) { const w = windows[prev]; delete labels[key(w)]; api('/api/unlabel', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({session: session.id, window_start: w.start, coder: $('coder').value.trim()})}); index = prev; render(); } }
});
(async () => {
  const boot = await api('/api/boot'); codebook = boot.codebook; sessions = boot.sessions;
  $('classes').innerHTML = codebook.classes.map(c => `<button data-label="${c.label}" onclick="label(codebook.classes.find(x=>x.key==='${c.key}'))"><b>${c.key}</b>${c.title}</button><div class="def">${c.definition}</div>`).join('');
  $('rule').textContent = codebook.rule;
  $('coder').value = localStorage.getItem('coder') || '';
  $('session').innerHTML = sessions.map(s => `<option value="${s.id}">${s.id} (${Math.round((s.end - s.start) / 60)} min, ${s.videos.length} cam${s.audio ? ', audio' : ', no audio'})</option>`).join('');
  $('session').onchange = e => loadSession(e.target.value);
  $('coder').onchange = () => loadSession($('session').value);
  const remembered = localStorage.getItem('session');
  if (remembered && sessions.some(s => s.id === remembered)) $('session').value = remembered;
  loadSession($('session').value);
})();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    settings: dict[str, Any] = {}
    sessions: list[dict[str, Any]] = []
    lock = threading.Lock()

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
            self._json({'codebook': CODEBOOK, 'sessions': [{k: v for k, v in s.items() if k != 'dir'} for s in self.sessions]})
        elif url.path == '/api/windows':
            session = self._session(query)
            if not session:
                return self._json({'error': 'no such session'}, 404)
            s = self.settings
            windows = windows_of(session, s['window'], s['step'], s['sample'], s['block'], s['seed'])
            coder = (query.get('coder') or ['anonymous'])[0]
            labels = self._labels(session, coder)
            self._json({'windows': windows, 'labels': list(labels.values())})
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
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    record = json.loads(line)
                    key = f"{float(record['window_start']):.3f}"
                    if record.get('label') is None:
                        labels.pop(key, None)  # an undo line
                    else:
                        labels[key] = record
        return labels

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        length = int(self.headers.get('content-length') or 0)
        record = json.loads(self.rfile.read(length) or b'{}')
        session = next((s for s in self.sessions if s['id'] == record.get('session')), None)
        if not session:
            return self._json({'error': 'no such session'}, 404)
        if url.path == '/api/label':
            path = self._labels_path(session, record.get('coder', ''))
        elif url.path == '/api/unlabel':
            path = self._labels_path(session, record.get('coder', ''))
            record = {'session': session['id'], 'window_start': record['window_start'], 'label': None, 'undone_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        else:
            return self._json({'error': 'not found'}, 404)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._json({'ok': True})


def get_parser():
    parser = argparse.ArgumentParser(prog='mmla ses-code', description="Code a session's windows by hand in the browser.")
    parser.add_argument('-a', '--artifacts', default=None, help="artifacts root (default <cwd>/artifacts)")
    parser.add_argument('-s', '--sessions', default=None, help="only sessions whose id contains this text")
    parser.add_argument('-w', '--window', type=float, default=10.0, help="window length in seconds (default 10)")
    parser.add_argument('-st', '--step', type=float, default=10.0, help="step between windows (default 10)")
    parser.add_argument('--sample', type=float, default=1.0, help="share of five-minute blocks to code, 0 to 1 (default 1: everything)")
    parser.add_argument('--block', type=float, default=300.0, help="block length the sampling draws from (default 300 s)")
    parser.add_argument('--seed', type=int, default=1, help="sampling seed, the same for every coder (default 1)")
    parser.add_argument('-p', '--port', type=int, default=8765)
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    artifacts = Path(args.artifacts or os.path.join(os.getcwd(), 'artifacts')).resolve()
    Handler.sessions = load_sessions(artifacts, args.sessions)
    Handler.settings = {'window': args.window, 'step': args.step, 'sample': args.sample, 'block': args.block, 'seed': args.seed}
    if not Handler.sessions:
        print(f"no sessions with video under {artifacts}")
        return 1
    total = sum(len(windows_of(s, args.window, args.step, args.sample, args.block, args.seed)) for s in Handler.sessions)
    print(f"{len(Handler.sessions)} sessions, {total} windows to code; open http://localhost:{args.port}/  (Ctrl-C stops)")
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
