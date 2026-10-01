/**
 * Session explorer (index.html): find a session, see at a glance what it holds, and open its Live or
 * Analysis view. The list comes from /api/sessions (the server caches it 15 s) and refreshes every
 * 30 s; sessions that write data now get a band of their own whose clocks tick locally and whose
 * state is re-read every 5 s from the light /state route (the full meta, with the per-modality
 * details, only when a session joins the band, every minute after, and when a badge tooltip asks).
 * After the first load every poll waits while the tab is hidden, so a forgotten tab costs the server
 * nothing.
 */

import {
  h, clear, fmt, api, poll, topbar, tooltip, emptyState, statusPill, icon, sessionTitle, debounce,
} from './core.js';

const LIST_EVERY = 30000;
// while a report job runs the Analysis column follows it more closely
const LIST_BUSY_EVERY = 5000;
const LIVE_EVERY = 5000;
const HEALTH_EVERY = 30000;
// a live session whose newest record is older than this gets a warning dot
const LIVE_WARN_AGE = 10;
// the meta of a live session (per-modality counts and newest records) is re-read this often (s);
// the 5 s poll reads only GET /api/sessions/<sid>/state
const LIVE_META_EVERY = 60;
// a modality tooltip on meta older than this (s) re-reads it and redraws when the answer comes
const LIVE_META_FRESH = 10;

const MODALITIES = [
  { key: 'asr', label: 'ASR', name: 'speech', count: 'asr_recognition', bucket: 3, unit: 'speech windows' },
  { key: 'ips', label: 'IPS', name: 'badge positions', count: 'ips_translation', bucket: 1, unit: 'windows' },
  { key: 'vfa', label: 'VFA', name: 'video features', count: 'vfa_features', bucket: 1, unit: 'frame sets' },
];

const SPEECH_LABELS = {
  group: 'Group mic',
  wearer: 'Worn mics',
  'wearer+group': 'Worn + group',
  individual: 'Named speakers',
};

// a session with worn mics beside a group mic matches both filters
const SPEECH_FILTERS = {
  group: ['group', 'wearer+group'],
  worn: ['wearer', 'wearer+group'],
};

const SORTS = [
  { value: 'newest', label: 'Newest first' },
  { value: 'oldest', label: 'Oldest first' },
  { value: 'longest', label: 'Longest first' },
];

const JOB_WORDS = { ready: 'ready', stale: 'out of date', running: 'computing', error: 'failed', missing: 'not computed' };

const monthFmt = new Intl.DateTimeFormat('en-GB', { month: 'long', year: 'numeric' });

const $ = (id) => document.getElementById(id);
const localNow = () => Date.now() / 1000;
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

const state = {
  sessions: null, // entries of the last good /api/sessions answer
  generatedAt: null, // server epoch the index was built at
  loadedAt: null, // browser epoch of the last good answer
  warnings: [],
  error: null, // {status, message} of the last failed list request
  health: null,
  healthError: null,
  offset: 0, // server clock minus browser clock (s), from /api/health
  filters: { q: '', task: 'all', speech: 'all', sort: 'newest' },
  live: new Map(), // sid -> {entry, meta, metaAt, cur: {t0, last_event}, anchor: {server, local}, stateMissing}
  endedAt: new Map(), // sid -> server epoch at which the meta said the session stopped being live
};

// tooltips: elements carry a content function; one delegated listener shows it on hover and focus

const tips = new WeakMap();
let tipShown = null;

function withTip(el, content) {
  tips.set(el, content);
  el.dataset.tip = '';
  return el;
}

function tipTarget(node) {
  const el = node && node.closest ? node.closest('[data-tip]') : null;
  return el && tips.has(el) ? el : null;
}

function showTip(el) {
  const content = tips.get(el);
  const body = typeof content === 'function' ? content() : content;
  if (!body) return;
  tipShown = el;
  tooltip.show(el, body);
}

function hideTip() {
  tipShown = null;
  tooltip.hide();
}

document.addEventListener('pointerover', (e) => {
  const el = tipTarget(e.target);
  if (el && el !== tipShown) showTip(el);
  else if (!el && tipShown) hideTip();
});
document.addEventListener('pointerout', (e) => {
  const el = tipTarget(e.target);
  if (el && el === tipShown && !el.contains(e.relatedTarget)) hideTip();
});
document.addEventListener('focusin', (e) => {
  const el = tipTarget(e.target);
  if (el && e.target === el) showTip(el);
});
document.addEventListener('focusout', (e) => {
  if (tipShown && e.target === tipShown) hideTip();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && tipShown) hideTip();
});

// session helpers

function recordedEpoch(e) {
  if (isNum(e.recorded_start)) return e.recorded_start;
  if (isNum(e.t0)) return e.t0;
  return null;
}

function isoDateEpoch(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(iso || ''));
  return m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])).getTime() / 1000 : null;
}

function sortEpoch(e) {
  const t = recordedEpoch(e);
  if (t != null) return t;
  const d = isoDateEpoch(e.date);
  return d == null ? -Infinity : d;
}

function counts(e) {
  return (e && e.counts) || {};
}

function hasData(e) {
  return Object.values(counts(e)).some((n) => isNum(n) && n > 0);
}

function hasModality(e, m) {
  const n = counts(e)[m.count];
  return (Array.isArray(e.modalities) && e.modalities.includes(m.key)) || (isNum(n) && n > 0);
}

/** records over the records a full span would hold (one per bucket), capped at 1; null when unknown. */
function coverage(n, duration, bucket) {
  if (!isNum(n) || n <= 0) return 0;
  if (!isNum(duration) || duration <= 0) return null;
  return Math.min(1, (n * bucket) / duration);
}

function suppressedLive(e) {
  const ended = state.endedAt.get(e.id);
  return ended != null && (state.generatedAt == null || state.generatedAt <= ended);
}

/** the session's state with the band's fresher knowledge applied (the index is cached 15 s). */
function isLive(e) {
  return e.state === 'live' && !suppressedLive(e);
}

function liveHref(id) {
  return `/live?session=${encodeURIComponent(id)}`;
}

function analysisHref(id) {
  return `/analysis?session=${encodeURIComponent(id)}`;
}

function speechLabel(mode) {
  return SPEECH_LABELS[mode] || null;
}

function monthKey(e) {
  const t = recordedEpoch(e);
  const d = t != null ? new Date(t * 1000) : (isoDateEpoch(e.date) != null ? new Date(isoDateEpoch(e.date) * 1000) : null);
  if (!d) return { key: 'none', label: 'Date unknown' };
  return { key: `${d.getFullYear()}-${d.getMonth()}`, label: monthFmt.format(d) };
}

/** list durations to the minute ("36 min", "1 h 01 min"); seconds only under a minute. */
function listDuration(sec) {
  if (!isNum(sec)) return fmt.na;
  if (sec < 60 || Math.round(sec / 60) >= 60) return fmt.duration(sec);
  return fmt.duration(Math.round(sec / 60) * 60);
}

function plural(n, one, many) {
  return `${fmt.int(n)} ${n === 1 ? one : many}`;
}

// filters

const FILTER_DEFAULTS = { q: '', task: 'all', speech: 'all', sort: 'newest' };

function readFiltersFromUrl() {
  try {
    const p = new URLSearchParams(window.location.search);
    const q = (p.get('q') || '').slice(0, 200);
    const task = p.get('task') || 'all';
    const speech = p.get('speech');
    const sort = p.get('sort');
    state.filters = {
      q,
      task: /^[A-Za-z0-9_.:-]{1,64}$/.test(task) ? task : 'all',
      speech: speech === 'group' || speech === 'worn' ? speech : 'all',
      sort: SORTS.some((o) => o.value === sort) ? sort : 'newest',
    };
  } catch {
    state.filters = { ...FILTER_DEFAULTS };
  }
}

function writeFiltersToUrl() {
  try {
    const p = new URLSearchParams();
    for (const k of Object.keys(FILTER_DEFAULTS)) {
      const v = state.filters[k];
      if (v && v !== FILTER_DEFAULTS[k]) p.set(k, v);
    }
    const qs = p.toString();
    const url = `${window.location.pathname}${qs ? `?${qs}` : ''}${window.location.hash}`;
    window.history.replaceState(null, '', url);
  } catch {
    // the filters still apply; only the shareable URL is lost
  }
}

function filtersActive() {
  const f = state.filters;
  return !!(f.q.trim() || f.task !== 'all' || f.speech !== 'all');
}

const WORD_SPLIT = /[\s_\-.,:·/]+/;

/** the words a search can start: id parts, title, date forms, month, speech setup, live. */
function searchWords(e) {
  const t = sessionTitle(e);
  const epoch = recordedEpoch(e);
  const text = [
    e.id, e.experiment, e.task, e.group, t.title, t.date, e.date,
    epoch != null ? monthFmt.format(new Date(epoch * 1000)) : null,
    speechLabel(e.speech_mode),
    isLive(e) ? 'live' : null,
  ].filter(Boolean).join(' ').toLowerCase();
  return text.split(WORD_SPLIT).filter(Boolean);
}

/** every query word starts a word of the session ("group 02" does not match the 02 inside 20250520),
 * or the whole query is part of the id (a pasted fragment like "t0826"). */
function matchesQuery(e, query, tokens) {
  if (!tokens.length) return true;
  if (String(e.id).toLowerCase().includes(query)) return true;
  const words = searchWords(e);
  return tokens.every((tk) => words.some((w) => w.startsWith(tk)));
}

function applyFilters(list) {
  const f = state.filters;
  const query = f.q.trim().toLowerCase();
  const tokens = query.split(WORD_SPLIT).filter(Boolean);
  const modes = SPEECH_FILTERS[f.speech] || null;
  const out = list.filter((e) => {
    if (f.task !== 'all' && e.task !== f.task) return false;
    if (modes && !modes.includes(e.speech_mode)) return false;
    return matchesQuery(e, query, tokens);
  });
  const byId = (a, b) => (a.id < b.id ? 1 : a.id > b.id ? -1 : 0);
  if (f.sort === 'oldest') out.sort((a, b) => sortEpoch(a) - sortEpoch(b) || -byId(a, b));
  else if (f.sort === 'longest') out.sort((a, b) => (isNum(b.duration) ? b.duration : -1) - (isNum(a.duration) ? a.duration : -1) || byId(a, b));
  else out.sort((a, b) => sortEpoch(b) - sortEpoch(a) || byId(a, b));
  return out;
}

function taskOptions() {
  const tasks = Array.from(new Set((state.sessions || []).map((e) => e.task).filter((t) => typeof t === 'string' && t)));
  tasks.sort((a, b) => a.localeCompare(b));
  return tasks;
}

let filterUi = null;

function buildFilters() {
  const box = $('filters');
  clear(box);
  const input = h('input', {
    class: 'input',
    id: 'session-search',
    type: 'search',
    value: state.filters.q,
    attrs: { autocomplete: 'off', spellcheck: 'false', 'aria-describedby': 'result-count' },
  });
  const onSearch = debounce(() => {
    state.filters.q = input.value;
    writeFiltersToUrl();
    renderList();
  }, 120);
  input.addEventListener('input', onSearch);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && input.value) {
      e.stopPropagation();
      input.value = '';
      onSearch();
    }
  });

  const taskSeg = segmentedField('Task', 'task', [], (v) => {
    state.filters.task = v;
    writeFiltersToUrl();
    renderList();
  });
  const speechSeg = segmentedField('Speech', 'speech', [
    { value: 'all', label: 'All' },
    { value: 'group', label: 'Group mic' },
    { value: 'worn', label: 'Worn mics' },
  ], (v) => {
    state.filters.speech = v;
    writeFiltersToUrl();
    renderList();
  });

  const select = h('select', { class: 'select', id: 'session-sort' },
    SORTS.map((o) => h('option', { value: o.value, selected: o.value === state.filters.sort, text: o.label })));
  select.value = state.filters.sort;
  select.addEventListener('change', () => {
    state.filters.sort = select.value;
    writeFiltersToUrl();
    renderList();
  });

  const count = h('p', { class: 'count', id: 'result-count', attrs: { role: 'status' } });

  box.append(
    h('div', { class: 'field search-field' },
      h('label', { attrs: { for: 'session-search' }, text: 'Search' }),
      h('div', { class: 'search-box' }, icon('search', 14), input)),
    taskSeg.field,
    speechSeg.field,
    h('div', { class: 'field' }, h('label', { attrs: { for: 'session-sort' }, text: 'Sort' }), select),
    count,
  );
  filterUi = { input, task: taskSeg, speech: speechSeg, select, count };
  syncTaskOptions();
}

let segCounter = 0;

function segmentedField(label, key, options, onChange) {
  segCounter += 1;
  const labelId = `filter-label-${segCounter}`;
  const seg = segmentedControl(options, state.filters[key], onChange, labelId);
  const field = h('div', { class: 'field' }, h('span', { class: 'field-label', id: labelId, text: label }), seg.el);
  return { field, ...seg };
}

/** a radiogroup like core.segmented, labelled by the visible field label (aria-labelledby). */
function segmentedControl(options, value, onChange, labelId) {
  const el = h('div', { class: 'seg', attrs: { role: 'radiogroup', 'aria-labelledby': labelId } });
  let opts = [];
  let current = value;

  function sync() {
    const idx = opts.findIndex((o) => o.value === current);
    Array.from(el.children).forEach((b, i) => {
      b.setAttribute('aria-checked', i === idx ? 'true' : 'false');
      b.tabIndex = i === idx || (idx < 0 && i === 0) ? 0 : -1;
    });
  }

  function choose(v, fire) {
    if (v === current) return;
    current = v;
    sync();
    if (fire) onChange(v);
  }

  function setOptions(next, v = current) {
    opts = next.slice();
    current = v;
    clear(el);
    for (const o of opts) {
      el.appendChild(h('button', {
        attrs: { type: 'button', role: 'radio', 'aria-checked': 'false', title: o.title || null },
        on: { click: () => choose(o.value, true) },
        text: o.label,
      }));
    }
    sync();
  }

  el.addEventListener('keydown', (e) => {
    const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'];
    if (!keys.includes(e.key) || !opts.length) return;
    e.preventDefault();
    let i = opts.findIndex((o) => o.value === current);
    if (e.key === 'Home') i = 0;
    else if (e.key === 'End') i = opts.length - 1;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') i = i <= 0 ? opts.length - 1 : i - 1;
    else i = i >= opts.length - 1 ? 0 : i + 1;
    choose(opts[i].value, true);
    el.children[i].focus();
  });

  setOptions(options, value);
  return { el, setOptions, value: () => current, setValue: (v) => choose(v, false) };
}

function syncTaskOptions() {
  if (!filterUi) return;
  const tasks = taskOptions();
  if (state.filters.task !== 'all' && !tasks.includes(state.filters.task)) {
    state.filters.task = 'all';
    writeFiltersToUrl();
  }
  const key = tasks.join('\n');
  if (filterUi.taskKey === key) return;
  filterUi.taskKey = key;
  filterUi.task.setOptions([{ value: 'all', label: 'All' }, ...tasks.map((t) => ({ value: t, label: t }))], state.filters.task);
  filterUi.task.field.hidden = tasks.length < 2;
}

function clearFilters() {
  state.filters = { ...state.filters, q: '', task: 'all', speech: 'all' };
  if (filterUi) {
    filterUi.input.value = '';
    filterUi.task.setValue('all');
    filterUi.speech.setValue('all');
  }
  writeFiltersToUrl();
  renderList();
  if (filterUi) filterUi.input.focus();
}

// small pieces

function meter(ratio) {
  const w = ratio == null ? 0 : Math.max(0, Math.min(1, ratio));
  return h('span', { class: 'meter', attrs: { 'aria-hidden': 'true' } }, h('span', { style: { width: `${(w * 100).toFixed(1)}%` } }));
}

function modalityText(e, m) {
  const n = counts(e)[m.count];
  if (!hasModality(e, m)) return `${m.label}: no ${m.name} in this session`;
  const cov = coverage(n, e.duration, m.bucket);
  const parts = [`${m.label} ${fmt.int(n)} ${m.unit}`];
  if (m.key === 'asr' && isNum(counts(e).asr_transcription)) parts.push(plural(counts(e).asr_transcription, 'transcript', 'transcripts'));
  parts.push(cov == null ? 'coverage n/a' : `${fmt.pct(cov)} coverage`);
  return parts.join(', ');
}

function modalityBadges(e) {
  const box = h('div', { class: 'mods' });
  for (const m of MODALITIES) {
    const present = hasModality(e, m);
    const text = modalityText(e, m);
    const cov = present ? coverage(counts(e)[m.count], e.duration, m.bucket) : null;
    const badge = h('span', {
      class: ['badge', 'mod-badge', present ? null : 'is-missing'],
      attrs: { role: 'img', 'aria-label': text },
    }, h('span', { text: m.label }), present ? meter(cov) : null);
    withTip(badge, () => ({ title: text }));
    box.appendChild(badge);
  }
  return box;
}

function analysisInfo(e) {
  if (!hasData(e)) {
    return { kind: 'none', label: 'No data', title: 'Nothing of this session is in InfluxDB yet.', rows: [] };
  }
  const r = e.report || {};
  const jobs = [
    { name: 'Speech and space', st: r.light || 'missing' },
    { name: 'Video analysis', st: r.video || 'missing' },
  ];
  const rows = jobs.map((j) => ({ value: JOB_WORDS[j.st] || j.st, label: j.name, key: 'none' }));
  const done = (st) => st === 'ready' || st === 'stale';
  if (jobs.some((j) => j.st === 'running')) {
    return { kind: 'running', label: 'Computing', title: 'Analysis is being computed', rows };
  }
  if (jobs.some((j) => j.st === 'error')) {
    return { kind: 'error', label: 'Failed', title: 'An analysis job failed', rows, note: 'Open Analysis to see the error and run it again.' };
  }
  const stale = jobs.some((j) => j.st === 'stale');
  const staleNote = stale ? 'Newer data arrived after it was computed; opening Analysis recomputes it.' : null;
  if (done(jobs[0].st) && done(jobs[1].st)) return { kind: 'ready', label: 'Ready', title: 'Analysis is computed', rows, note: staleNote };
  if (done(jobs[0].st) || done(jobs[1].st)) {
    return { kind: 'partial', label: 'Partly ready', title: 'Part of the analysis is computed', rows, note: staleNote || 'Opening Analysis computes the rest.' };
  }
  return { kind: 'missing', label: 'Not computed', title: 'Analysis has not run yet', rows, note: 'Opening Analysis computes it.' };
}

function analysisCell(e) {
  const info = analysisInfo(e);
  let mark = null;
  if (info.kind === 'ready' || info.kind === 'partial') mark = icon('check', 14);
  else if (info.kind === 'error') mark = icon('alert', 14);
  else if (info.kind === 'running') mark = h('span', { class: 'spinner', attrs: { 'aria-hidden': 'true' } });
  const el = h('span', { class: 'ana', dataset: { kind: info.kind } }, mark, h('span', { text: info.label }));
  if (info.rows.length) {
    el.setAttribute('aria-label', `${info.label}: ${info.rows.map((r) => `${r.label} ${r.value}`).join(', ')}`);
    withTip(el, () => ({ title: info.title, rows: info.rows, note: info.note }));
  }
  return el;
}

function speechCell(e) {
  const label = speechLabel(e.speech_mode);
  return label ? h('span', { text: label }) : h('span', { class: 'muted', text: fmt.na });
}

function primaryAction(e, { sm = true } = {}) {
  const live = isLive(e);
  const cls = ['btn', sm ? 'sm' : null, live ? 'primary' : null];
  if (!live && !hasData(e)) {
    const activeInMongo = e.mongo && e.mongo.status === 'active';
    if (!activeInMongo) {
      return h('a', {
        class: cls,
        attrs: { 'aria-disabled': 'true', title: 'Nothing to replay: this session has no data in InfluxDB.' },
        dataset: { action: 'live' },
        text: 'Replay',
      });
    }
  }
  const label = live || (e.mongo && e.mongo.status === 'active' && !hasData(e)) ? 'Live' : 'Replay';
  return h('a', { class: cls, href: liveHref(e.id), dataset: { action: 'live' }, text: label });
}

// health chips

const HEALTH_ITEMS = [
  { key: 'influx', label: 'InfluxDB' },
  { key: 'mongo', label: 'MongoDB' },
  { key: 'worker', label: 'Report worker' },
  { key: 'media', label: 'Stream server' },
];

function healthState(key, v) {
  if (state.healthError && !state.health) return { tone: 'off', word: 'unknown', tip: state.healthError };
  if (!v) return { tone: 'off', word: 'checking', tip: 'Waiting for the dashboard server to answer.' };
  const err = typeof v.error === 'string' && v.error ? v.error : null;
  switch (key) {
    case 'influx':
      if (v.ok) return { tone: 'good', word: 'ok', tip: `${v.url || 'InfluxDB'}, bucket ${v.bucket || fmt.na}` };
      return { tone: 'bad', word: 'unreachable', tip: err || `InfluxDB at ${v.url || 'its configured address'} did not answer.` };
    case 'mongo':
      if (v.ok == null) return { tone: 'off', word: 'not configured', tip: 'No MongoDB in config.yml, so devices and provenance are left out.' };
      if (v.ok) return { tone: 'good', word: 'ok', tip: 'Session documents are read from MongoDB.' };
      return { tone: 'bad', word: 'unreachable', tip: err || 'MongoDB did not answer.' };
    case 'worker': {
      const queue = v.queue || 'mmla-dashboard';
      if (v.ok) return { tone: 'good', word: 'ok', tip: `${plural(v.workers || 0, 'Celery worker', 'Celery workers')} on the ${queue} queue.` };
      if (v.ok == null) return { tone: 'neutral', word: 'local', tip: 'Reports are computed by the dashboard server itself (DASHBOARD_JOBS=local).' };
      if (v.mode === 'celery') {
        return { tone: 'bad', word: 'unreachable', tip: `No Celery worker consumes the ${queue} queue, so report jobs wait.${err ? ` ${err}` : ''}` };
      }
      return { tone: 'neutral', word: 'local', tip: `No Celery worker consumes the ${queue} queue, so the dashboard server computes reports itself.${err ? ` ${err}` : ''}` };
    }
    case 'media':
      if (v.ok == null) return { tone: 'off', word: 'not configured', tip: 'No stream server is set in System Settings, so camera tiles draw skeletons only.' };
      if (v.ok) {
        const parts = [`MediaMTX at ${v.host || fmt.na}`];
        if (isNum(v.paths)) parts.push(plural(v.paths, 'stream ready', 'streams ready'));
        if (v.webrtc === false) parts.push('WebRTC is off, so camera tiles draw skeletons only');
        else if (v.webrtc === true) parts.push('WebRTC on');
        return { tone: 'good', word: 'ok', tip: `${parts.join(', ')}.` };
      }
      return { tone: 'bad', word: 'unreachable', tip: err || `MediaMTX at ${v.host || fmt.na} did not answer.` };
    default:
      return { tone: 'off', word: fmt.na, tip: '' };
  }
}

function renderHealth() {
  const ul = $('health');
  const focusedKey = document.activeElement && document.activeElement.closest && document.activeElement.closest('.health-chip')
    ? document.activeElement.closest('.health-chip').dataset.key : null;
  if (tipShown && ul.contains(tipShown)) hideTip();
  clear(ul);
  for (const item of HEALTH_ITEMS) {
    const st = healthState(item.key, state.health && state.health[item.key]);
    const chip = h('li', {
      class: 'chip health-chip',
      dataset: { tone: st.tone, key: item.key },
      attrs: { tabindex: '0', 'aria-label': `${item.label}: ${st.word}. ${st.tip}` },
    }, h('span', { class: 'dot', attrs: { 'aria-hidden': 'true' } }), h('span', { text: item.label }), h('span', { class: 'health-word', text: st.word }));
    withTip(chip, () => ({ title: `${item.label}: ${st.word}`, note: st.tip }));
    ul.appendChild(chip);
  }
  if (focusedKey) {
    const again = ul.querySelector(`[data-key="${focusedKey}"]`);
    if (again) again.focus({ preventScroll: true });
  }
}

let healthInflight = null;

async function loadHealth() {
  if (healthInflight) return healthInflight;
  healthInflight = (async () => {
    const before = localNow();
    const res = await api('/api/health');
    const after = localNow();
    if (res.ok && res.data && typeof res.data === 'object') {
      state.health = res.data;
      state.healthError = null;
      if (isNum(res.data.time)) state.offset = res.data.time - (before + after) / 2;
    } else {
      state.healthError = res.error || 'The dashboard server did not answer.';
      state.health = null;
    }
    renderHealth();
    renderSub();
    // the empty state names the bucket, which only health knows
    if (state.sessions && !state.sessions.length) renderList();
  })();
  try {
    await healthInflight;
  } finally {
    healthInflight = null;
  }
}

// header line and notes

function renderSub() {
  const el = $('page-sub');
  if (!state.sessions) {
    el.textContent = state.error ? 'The session list could not be read' : 'Reading the session list';
    return;
  }
  const n = state.sessions.length;
  const bucket = state.health && state.health.influx && state.health.influx.bucket;
  el.textContent = `${plural(n, 'session', 'sessions')} in InfluxDB${bucket ? ` bucket ${bucket}` : ''}`;
}

function renderNotes() {
  const box = $('notes');
  clear(box);
  const lines = [];
  if (state.sessions && state.error) {
    const when = state.loadedAt ? fmt.time(state.loadedAt) : fmt.na;
    lines.push(h('p', { class: 'note-line', dataset: { kind: 'warn' } },
      icon('alert', 14),
      h('span', { text: `Could not refresh the list (${state.error.message}). Showing the list from ${when}.` }),
      h('button', { class: 'btn sm', attrs: { type: 'button' }, on: { click: () => loadSessions() }, text: 'Retry' })));
  }
  for (const w of state.warnings) lines.push(h('p', { class: 'note-line' }, icon('info', 14), h('span', { text: w })));
  box.append(...lines);
  box.hidden = !lines.length;
}

// live now band

const liveCards = new Map(); // sid -> {el, refs}

function serverNow(rec) {
  const now = localNow();
  if (rec && rec.anchor) return rec.anchor.server + (now - rec.anchor.local);
  return now + state.offset;
}

function liveFields(rec) {
  const cur = rec.cur || {};
  const e = rec.entry;
  const t0 = isNum(cur.t0) ? cur.t0 : e.t0;
  const last = isNum(cur.last_event) ? cur.last_event : e.last_event;
  return { t0, last };
}

function liveModalityTip(sid, m, badge) {
  // looked up at show time: the band replaces its records on every list load
  const rec = state.live.get(sid);
  if (!rec) return null;
  // per-modality newest records come from the meta, which the 5 s poll does not read: refresh it
  // when it is older than a few seconds (at most once per LIVE_META_FRESH, even when reads fail)
  // and redraw the tooltip that asked
  const here = localNow();
  if ((!rec.metaAt || here - rec.metaAt > LIVE_META_FRESH) && here - (metaTriedAt.get(sid) || 0) > LIVE_META_FRESH) {
    readLiveMeta(sid).then((ended) => {
      if (ended) {
        renderLive();
        renderList();
      } else if (badge && tipShown === badge && badge.isConnected) showTip(badge);
    });
  }
  const meta = rec.meta;
  const mod = meta && meta.modalities ? meta.modalities[m.key] : undefined;
  const now = serverNow(rec);
  if (mod === null || (mod === undefined && !hasModality(rec.entry, m))) return { title: `${m.label}: no ${m.name} yet` };
  let last = null;
  let n = counts(rec.entry)[m.count];
  if (mod) {
    if (m.key === 'asr') {
      last = mod.recognition ? mod.recognition.last : null;
      n = mod.recognition ? mod.recognition.n : n;
    } else {
      last = mod.last;
      n = mod.n;
    }
  }
  const rows = [{ value: fmt.int(n), label: m.unit, key: 'none' }];
  if (isNum(last)) rows.push({ value: fmt.ago(now - last), label: 'newest record', key: 'none' });
  return { title: m.label, rows };
}

function livePresence(rec) {
  const meta = rec.meta;
  return MODALITIES.map((m) => (meta && meta.modalities ? !!meta.modalities[m.key] : hasModality(rec.entry, m)));
}

function liveBadges(sid, presence) {
  const box = h('div', { class: 'mods' });
  MODALITIES.forEach((m, i) => {
    const present = presence[i];
    const badge = h('span', {
      class: ['badge', 'mod-badge', present ? null : 'is-missing'],
      attrs: { role: 'img', 'aria-label': `${m.label}: ${present ? 'receiving data' : 'no data'}` },
    }, h('span', { text: m.label }));
    withTip(badge, () => liveModalityTip(sid, m, badge));
    box.appendChild(badge);
  });
  return box;
}

function buildLiveCard(sid) {
  const rec = state.live.get(sid);
  const e = rec.entry;
  const t = sessionTitle(e);
  const clock = h('span', { class: 'live-clock', attrs: { 'aria-hidden': 'true' } });
  const age = h('span', { class: 'live-age' }, h('span', { class: 'dot', attrs: { 'aria-hidden': 'true' } }), h('span'));
  const badgeSlot = h('div');
  const speech = speechLabel(e.speech_mode);
  const el = h('article', { class: 'card live-card', attrs: { 'aria-label': `${t.title}, live` } },
    h('div', { class: 'live-who' },
      h('div', { class: 'live-title-row' },
        h('h3', { class: 'live-title', text: t.title }),
        statusPill({ state: 'live' }),
        t.date ? h('span', { class: 'live-date', text: t.date }) : null),
      h('span', { class: 'ses-id', text: e.id, title: e.id })),
    h('div', { class: 'live-stat' }, h('span', { class: 'live-stat-label', text: 'Elapsed' }), clock),
    h('div', { class: 'live-stat' }, h('span', { class: 'live-stat-label', text: 'Data' }), badgeSlot),
    h('div', { class: 'live-stat' }, h('span', { class: 'live-stat-label', text: 'Speech' }), h('span', { class: ['live-speech', speech ? null : 'muted'], text: speech || fmt.na })),
    h('div', { class: 'live-stat' }, h('span', { class: 'live-stat-label', text: 'Last data' }), age),
    h('div', { class: 'live-actions' },
      h('a', { class: 'btn primary', href: liveHref(sid), text: 'Open live' }),
      h('a', { class: 'btn', href: analysisHref(sid), text: 'Analysis' })));
  const card = { el, clock, age, badgeSlot, sid };
  updateLiveCard(card);
  return card;
}

function updateLiveCard(card) {
  const rec = state.live.get(card.sid);
  if (!rec) return;
  // rebuild the badges only when a modality appears or disappears, so an open tooltip stays put
  const presence = livePresence(rec);
  const key = presence.join(',');
  if (card.presenceKey !== key) {
    card.presenceKey = key;
    if (tipShown && card.badgeSlot.contains(tipShown)) hideTip();
    clear(card.badgeSlot).appendChild(liveBadges(card.sid, presence));
  }
  tickLiveCard(card);
}

function tickLiveCard(card) {
  const rec = state.live.get(card.sid);
  if (!rec) return;
  const { t0, last } = liveFields(rec);
  const now = serverNow(rec);
  const elapsed = isNum(t0) ? Math.max(0, now - t0) : null;
  card.clock.textContent = elapsed == null ? fmt.na : fmt.clockSpan(elapsed)(elapsed);
  card.clock.parentElement.setAttribute('aria-label', `Elapsed ${elapsed == null ? fmt.na : fmt.duration(elapsed)}`);
  const ageSec = isNum(last) ? Math.max(0, now - last) : null;
  card.age.lastChild.textContent = ageSec == null ? fmt.na : fmt.ago(ageSec);
  card.age.dataset.tone = ageSec != null && ageSec > LIVE_WARN_AGE ? 'warn' : 'good';
}

function renderLive() {
  const band = $('live-band');
  const list = $('live-list');
  const ids = Array.from(state.live.keys());
  for (const [sid, card] of liveCards) {
    if (!state.live.has(sid)) {
      card.el.remove();
      liveCards.delete(sid);
    }
  }
  ids.forEach((sid, i) => {
    let card = liveCards.get(sid);
    if (!card) {
      card = buildLiveCard(sid);
      liveCards.set(sid, card);
    } else updateLiveCard(card);
    // move a card only when the order changed: moving a node drops keyboard focus inside it
    if (list.children[i] !== card.el) list.insertBefore(card.el, list.children[i] || null);
  });
  band.hidden = !ids.length;
  $('live-sub').textContent = ids.length ? `${plural(ids.length, 'session is', 'sessions are')} writing data now` : '';
}

function syncLiveFromIndex() {
  const liveEntries = (state.sessions || []).filter(isLive);
  const keep = new Set(liveEntries.map((e) => e.id));
  for (const sid of Array.from(state.live.keys())) if (!keep.has(sid)) state.live.delete(sid);
  liveEntries.sort((a, b) => sortEpoch(b) - sortEpoch(a));
  const next = new Map();
  let added = false;
  for (const e of liveEntries) {
    if (!state.live.has(e.id)) added = true;
    next.set(e.id, { ...(state.live.get(e.id) || {}), entry: e });
  }
  state.live.clear();
  for (const [k, v] of next) state.live.set(k, v);
  // a session new to the band gets its exact clock right away instead of at the next 5 s read
  if (added) setTimeout(() => refreshLive(), 0);
}

let liveInflight = null;
const metaInflight = new Map(); // sid -> promise of a meta read
const metaTriedAt = new Map(); // sid -> browser epoch of the last meta read started

/**
 * Applies a live state ({live, last_event, lag, t0}, from /state or the meta's state) to the band's
 * record; true when the session stopped being live. Records are looked up again after every await:
 * a list load replaces them.
 */
function applyLiveState(sid, st, arrived) {
  const rec = state.live.get(sid);
  if (!rec || !st || typeof st !== 'object') return false;
  const cur = rec.cur || {};
  rec.cur = {
    t0: isNum(st.t0) ? st.t0 : cur.t0,
    last_event: isNum(st.last_event) ? st.last_event : cur.last_event,
  };
  if (isNum(st.last_event) && isNum(st.lag)) rec.anchor = { server: st.last_event + st.lag, local: arrived };
  if (st.live === false) {
    state.endedAt.set(sid, rec.anchor ? rec.anchor.server : arrived + state.offset);
    state.live.delete(sid);
    return true;
  }
  return false;
}

/** reads the full meta of a live session (per-modality details); resolves true when it ended. */
function readLiveMeta(sid) {
  if (metaInflight.has(sid)) return metaInflight.get(sid);
  metaTriedAt.set(sid, localNow());
  const p = (async () => {
    const res = await api(`/api/sessions/${encodeURIComponent(sid)}`);
    const arrived = localNow();
    const rec = state.live.get(sid);
    if (!rec || !res.ok || !res.data || typeof res.data !== 'object') return false;
    const meta = res.data;
    rec.meta = meta;
    rec.metaAt = arrived;
    if (meta.report && typeof meta.report === 'object') rec.entry = { ...rec.entry, report: meta.report };
    return applyLiveState(sid, { ...(meta.state || {}), t0: meta.t0 }, arrived);
  })().finally(() => metaInflight.delete(sid));
  metaInflight.set(sid, p);
  return p;
}

/** one 5 s read of a live session: /state, or the meta when it is due or the server lacks /state. */
async function refreshLiveOne(sid) {
  const rec = state.live.get(sid);
  if (!rec) return false;
  const metaDue = !rec.meta || !rec.metaAt || localNow() - rec.metaAt >= LIVE_META_EVERY;
  if (!rec.stateMissing && !metaDue) {
    const res = await api(`/api/sessions/${encodeURIComponent(sid)}/state`);
    const arrived = localNow();
    if (res.ok && res.data && typeof res.data === 'object') return applyLiveState(sid, res.data, arrived);
    if (res.status !== 404) return false;
    // a server without the /state route: this session reads the meta from now on
    const again = state.live.get(sid);
    if (again) again.stateMissing = true;
  }
  return readLiveMeta(sid);
}

async function refreshLive() {
  if (!state.live.size) return;
  if (liveInflight) return;
  liveInflight = (async () => {
    const ended = await Promise.all(Array.from(state.live.keys()).map((sid) => refreshLiveOne(sid)));
    renderLive();
    if (ended.some(Boolean)) renderList();
  })();
  try {
    await liveInflight;
  } finally {
    liveInflight = null;
  }
}

function tick() {
  if (document.visibilityState === 'hidden') return;
  for (const card of liveCards.values()) tickLiveCard(card);
}

// session list

let listInflight = null;
let lastSignature = null;

function loadSessions() {
  if (!listInflight) {
    listInflight = doLoadSessions().finally(() => {
      listInflight = null;
    });
  }
  return listInflight;
}

async function doLoadSessions() {
  const res = await api('/api/sessions');
  if (res.ok && res.data && Array.isArray(res.data.sessions)) {
    state.sessions = res.data.sessions.filter((e) => e && typeof e.id === 'string' && e.id);
    state.generatedAt = isNum(res.data.generated_at) ? res.data.generated_at : null;
    state.warnings = Array.isArray(res.data.warnings) ? res.data.warnings.filter((w) => typeof w === 'string' && w) : [];
    state.loadedAt = localNow();
    state.error = null;
    // the band's knowledge of an ended session is only needed until the index catches up
    for (const [sid, at] of Array.from(state.endedAt)) {
      if (state.generatedAt != null && state.generatedAt > at) state.endedAt.delete(sid);
    }
    syncLiveFromIndex();
  } else {
    state.error = { status: res.status, message: res.error || 'The session list could not be read.' };
  }
  if (state.sessions && !filterUi) buildFilters();
  else syncTaskOptions();
  // filters only make sense once there is something to filter
  $('filters').hidden = !(state.sessions && state.sessions.length);
  renderSub();
  renderNotes();
  renderLive();
  renderList();
}

function anyRunning() {
  return (state.sessions || []).some((e) => e.report && (e.report.light === 'running' || e.report.video === 'running'));
}

function skeletonTable() {
  const head = tableHead();
  const body = h('tbody');
  for (let i = 0; i < 6; i += 1) {
    const bar = (w) => h('span', { class: 'skeleton', style: { width: w } });
    body.appendChild(h('tr', { class: 'list-skeleton' },
      h('td', {}, bar('84px'), bar('56px')),
      h('td', { class: 'c-ses' }, bar(`${40 + ((i * 17) % 30)}%`), bar('70%')),
      h('td', {}, bar('64px')),
      h('td', {}, bar('180px')),
      h('td', {}, bar('72px')),
      h('td', {}, bar('84px')),
      h('td', {}, bar('128px'))));
  }
  return h('table', { class: 'data sessions', attrs: { 'aria-busy': 'true', 'aria-label': 'Loading sessions' } }, head, body);
}

function tableHead() {
  const th = (text, cls, sr) => h('th', { class: cls || null, attrs: { scope: 'col' } }, sr ? h('span', { class: 'sr-only', text }) : text);
  return h('thead', {}, h('tr', {},
    th('Recorded'), th('Session', 'c-ses'), th('Duration', 'c-dur'), th('Data'), th('Speech'), th('Analysis'), th('Actions', 'c-act', true)));
}

function sessionRow(e) {
  const t = sessionTitle(e);
  const epoch = recordedEpoch(e);
  const live = isLive(e);
  const dateText = epoch != null ? fmt.date(epoch) : (t.date || fmt.na);
  const timeText = epoch != null ? `${fmt.hm(epoch)} ${fmt.zone(epoch)}`.trim() : null;
  const tr = h('tr', { class: 'session-row', dataset: { sid: e.id }, attrs: { tabindex: '0' } },
    h('td', { class: 'c-rec', dataset: { label: 'Recorded' } },
      h('div', { text: dateText }),
      timeText ? h('div', { class: 'line2 muted', text: timeText }) : null),
    h('td', { class: 'c-ses' },
      h('div', { class: 'ses-title' }, h('span', { text: t.title, title: t.title }), live ? statusPill({ state: 'live' }) : null),
      h('span', { class: 'ses-id line2', text: e.id, title: e.id })),
    h('td', { class: 'c-dur num', dataset: { label: 'Duration' }, text: listDuration(e.duration), title: isNum(e.duration) ? fmt.duration(e.duration) : null }),
    h('td', { class: 'c-data', dataset: { label: 'Data' } }, modalityBadges(e)),
    h('td', { class: 'c-speech', dataset: { label: 'Speech' } }, speechCell(e)),
    h('td', { class: 'c-ana', dataset: { label: 'Analysis' } }, analysisCell(e)),
    h('td', { class: 'c-act' }, h('div', { class: 'acts' },
      primaryAction(e),
      h('a', { class: 'btn sm', href: analysisHref(e.id), dataset: { action: 'analysis' }, text: 'Analysis' }))));
  return tr;
}

function groupRows(list) {
  if (state.filters.sort === 'longest') return [{ label: null, items: list }];
  const groups = [];
  let cur = null;
  for (const e of list) {
    const m = monthKey(e);
    if (!cur || cur.key !== m.key) {
      cur = { key: m.key, label: m.label, items: [] };
      groups.push(cur);
    }
    cur.items.push(e);
  }
  return groups;
}

function signatureOf(list) {
  return JSON.stringify([state.filters.sort, Array.from(state.endedAt.keys()), list.map((e) => [
    e.id, e.state, e.duration, e.counts, e.modalities, e.speech_mode, e.report, e.recorded_start, e.t0, e.mongo && e.mongo.status,
  ])]);
}

function renderList() {
  const box = $('list');
  const count = filterUi ? filterUi.count : null;
  if (!state.sessions) {
    lastSignature = null;
    if (state.error) {
      const unreachable = state.error.status === 0;
      const influx = state.error.status === 503;
      clear(box).appendChild(emptyState({
        kind: 'error',
        title: unreachable ? 'The dashboard server did not answer' : influx ? 'Could not read sessions from InfluxDB' : 'The session list could not be read',
        body: state.error.message,
        action: { label: 'Retry', onClick: () => retryFromPanel() },
      }));
    } else if (!box.querySelector('.list-skeleton')) {
      clear(box).appendChild(skeletonTable());
    }
    return;
  }
  const all = state.sessions;
  if (!all.length) {
    lastSignature = null;
    if (count) count.textContent = '';
    const bucket = state.health && state.health.influx && state.health.influx.bucket;
    clear(box).appendChild(emptyState({
      title: 'No sessions yet',
      body: `Sessions appear here once a pipeline writes to InfluxDB${bucket ? ` bucket ${bucket}` : ''}. Start one from the Launcher in mmla tui, or run a base with the mmla command.`,
    }));
    return;
  }
  const list = applyFilters(all);
  if (count) {
    count.textContent = filtersActive() || list.length !== all.length
      ? `${fmt.int(list.length)} of ${plural(all.length, 'session', 'sessions')}`
      : plural(all.length, 'session', 'sessions');
  }
  if (!list.length) {
    lastSignature = null;
    clear(box).appendChild(emptyState({
      title: 'No sessions match these filters',
      body: 'Try another search, or show every task and microphone setup.',
      action: { label: 'Clear filters', onClick: clearFilters },
    }));
    return;
  }
  const sig = signatureOf(list);
  if (sig === lastSignature && box.querySelector('table.sessions:not([aria-busy])')) return;
  lastSignature = sig;

  // keep keyboard focus on the same row or button across a refresh
  const active = document.activeElement;
  let focusSid = null;
  let focusAction = null;
  if (active && box.contains(active)) {
    const row = active.closest('tr.session-row');
    focusSid = row ? row.dataset.sid : null;
    focusAction = active.dataset && active.dataset.action ? active.dataset.action : null;
  }
  if (tipShown && box.contains(tipShown)) hideTip();

  const body = h('tbody');
  const groups = groupRows(list);
  for (const g of groups) {
    if (g.label) {
      body.appendChild(h('tr', { class: 'month-row' },
        h('th', { attrs: { colspan: '7', scope: 'colgroup' } },
          h('span', { text: g.label }),
          h('span', { class: 'month-count', text: plural(g.items.length, 'session', 'sessions') }))));
    }
    for (const e of g.items) body.appendChild(sessionRow(e));
  }
  const table = h('table', { class: 'data sessions' }, h('caption', { class: 'sr-only', text: 'Sessions; select a row to open its analysis' }), tableHead(), body);
  clear(box).appendChild(table);

  if (focusSid) {
    const row = Array.from(body.querySelectorAll('tr.session-row')).find((r) => r.dataset.sid === focusSid);
    const target = row && focusAction ? row.querySelector(`[data-action="${focusAction}"]`) : row;
    if (target) target.focus({ preventScroll: true });
  }
}

async function retryFromPanel() {
  state.error = null;
  renderList();
  renderSub();
  await Promise.all([loadSessions(), loadHealth()]);
}

function go(href, newTab) {
  if (newTab) window.open(href, '_blank', 'noopener');
  else window.location.href = href;
}

function wireListEvents() {
  const box = $('list');
  box.addEventListener('click', (e) => {
    const link = e.target.closest('a[aria-disabled="true"]');
    if (link) {
      e.preventDefault();
      return;
    }
    if (e.target.closest('a, button, input, select, label')) return;
    const row = e.target.closest('tr.session-row');
    if (!row) return;
    const sel = window.getSelection ? String(window.getSelection() || '') : '';
    if (sel.trim()) return;
    go(analysisHref(row.dataset.sid), e.metaKey || e.ctrlKey);
  });
  box.addEventListener('auxclick', (e) => {
    if (e.button !== 1 || e.target.closest('a, button')) return;
    const row = e.target.closest('tr.session-row');
    if (row) go(analysisHref(row.dataset.sid), true);
  });
  box.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || !e.target.matches('tr.session-row')) return;
    e.preventDefault();
    go(analysisHref(e.target.dataset.sid), e.metaKey || e.ctrlKey);
  });
}

// keyboard: "/" jumps to the search field

document.addEventListener('keydown', (e) => {
  if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
  const t = e.target;
  if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
  if (!filterUi) return;
  e.preventDefault();
  filterUi.input.focus();
  filterUi.input.select();
});

// start

/** run fn now, whatever the visibility (a tab opened in the background is ready when shown), then
 * every interval() ms while the tab is visible. */
function repeat(fn, interval) {
  Promise.resolve(fn()).finally(() => {
    setTimeout(() => poll(fn, { interval, visible: true }), interval());
  });
}

function start() {
  readFiltersFromUrl();
  const slot = $('topbar');
  slot.replaceWith(topbar({ crumbs: [{ label: 'Sessions' }] }));
  renderHealth();
  renderSub();
  renderList();
  wireListEvents();

  repeat(loadSessions, () => (anyRunning() ? LIST_BUSY_EVERY : LIST_EVERY));
  repeat(loadHealth, () => HEALTH_EVERY);
  // the band's first read follows the first list load (syncLiveFromIndex), so this starts one interval later
  setTimeout(() => poll(refreshLive, { interval: LIVE_EVERY, visible: true }), LIVE_EVERY);
  setInterval(tick, 1000);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') tick();
  });
}

start();
