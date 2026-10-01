/**
 * The analysis page of one session: the header, the section navigation with the range bar, and the
 * loading of the meta and the four report parts. The parts are computed by background jobs, so each
 * is polled while the server answers 202 and every section renders as soon as its part arrives.
 * One time range (session offsets) drives every section; it lives in the URL hash as #range=a-b.
 */

import {
  h, clear, fmt, api, poll, theme, topbar, sessionIdFromUrl, sessionParamInvalid, sessionTitle, sessionHeading, statusPill, emptyState,
  icon, identity, sortTags, isPupilTag, tooltip, announce, downloadLink,
} from './core.js';
import {
  buildSections, SECTION_NAV, speechModel, spaceModel, normalizeRange, parseRangeHash, rangeHash, exportLinks, PART_JOB,
} from './analysis-sections.js';

const PARTS = ['speech', 'space', 'attention', 'timeline'];
const SPEECH_MODE = { group: 'Group mic', wearer: 'Worn mics', 'wearer+group': 'Worn + group', individual: 'Named speakers' };

theme.init();

const sid = sessionIdFromUrl();
const enc = sid ? encodeURIComponent(sid) : '';
const app = document.getElementById('app');
const bar = topbar({ session: sid ? { id: sid } : null, view: sid ? 'analysis' : null, crumbs: sid ? null : [{ label: 'Sessions', href: '/' }, { label: 'Analysis' }] });
const main = h('main', { class: 'page', id: 'main' });
app.append(bar, main);

const ctx = {
  sid,
  meta: null,
  duration: 0,
  span: [0, 1],
  clock: fmt.clockSpan(1),
  parts: { speech: null, space: null, attention: null, timeline: null },
  status: { speech: { state: 'loading' }, space: { state: 'loading' }, attention: { state: 'loading' }, timeline: { state: 'loading' } },
  models: { speech: null, space: null },
  version: 0,
  range: null,
  cursor: null,
  ident: identity([]),
  roster: [],
  setRange,
  setCursor,
  runJob,
};

let sections = null;
const loaders = {};
const ui = {};

/**
 * fn at most once per frame with the latest call; a timer backs the frame up, since a page that is
 * not being painted (a hidden tab or pane) gets no animation frames.
 */
function frameThrottle(fn) {
  let pending = false;
  let timer = 0;
  const run = () => {
    if (!pending) return;
    pending = false;
    clearTimeout(timer);
    fn();
  };
  return () => {
    if (pending) return;
    pending = true;
    timer = setTimeout(run, document.visibilityState === 'hidden' ? 30 : 120);
    requestAnimationFrame(run);
  };
}

// the page renders at most once per frame, whatever arrives
const schedule = frameThrottle(() => {
  if (sections) sections.render();
  renderStatusChip();
});

if (!sid) {
  document.title = 'Session analysis';
  const invalid = sessionParamInvalid();
  main.appendChild(emptyState({
    title: invalid ? 'This is not a valid session id.' : 'No session chosen.',
    body: invalid ? 'Check the link, or open a session from the session list to see its analysis.' : 'Open a session from the session list to see its analysis.',
    action: { label: 'Sessions', href: '/' },
    kind: invalid ? 'error' : undefined,
  }));
} else {
  document.title = `Analysis: ${sid}`;
  loadMeta();
}

// meta and header

async function loadMeta() {
  clear(main);
  main.appendChild(h('div', { class: 'an-head-skeleton', attrs: { 'aria-hidden': 'true' } },
    h('span', { class: 'skeleton', style: { width: '280px', height: '22px' } }),
    h('span', { class: 'skeleton', style: { width: '420px', maxWidth: '100%', height: '14px' } }),
    h('span', { class: 'skeleton', style: { width: '360px', maxWidth: '100%', height: '14px' } })));
  main.appendChild(h('p', { class: 'sr-only', attrs: { role: 'status' }, text: 'Loading the session.' }));
  const res = await api(`/api/sessions/${enc}`);
  if (!res.ok || !res.data || !res.data.id) {
    clear(main);
    let title = 'Could not load this session.';
    let body = res.error || null;
    if (res.status === 404) {
      title = 'No session with this id.';
      body = 'Neither InfluxDB nor MongoDB knows it. Check the id or pick a session from the list.';
    } else if (res.status === 400) title = 'This is not a valid session id.';
    else if (res.status === 503) title = 'InfluxDB is not reachable.';
    const actions = h('div', { class: 'row' },
      res.status === 404 || res.status === 400 ? null : h('button', { class: 'btn sm', attrs: { type: 'button' }, on: { click: loadMeta }, text: 'Retry' }),
      h('a', { class: 'btn sm', href: '/', text: 'Sessions' }));
    main.appendChild(emptyState({ title, body, action: actions, kind: res.status === 404 ? 'empty' : 'error' }));
    return;
  }
  applyMeta(res.data);
  buildPage();
  const fromHash = normalizeRange(parseRangeHash(window.location.hash), ctx.span);
  if (fromHash) ctx.range = fromHash;
  renderRangeBar();
  for (const p of PARTS) startPart(p);
  schedule();
  if (ctx.meta.state && ctx.meta.state.live) followLiveMeta();
}

function applyMeta(meta) {
  ctx.meta = meta;
  ctx.duration = Number(meta.duration) || Math.max(0, (meta.t1 || 0) - (meta.t0 || 0)) || 1;
  ctx.span = [0, ctx.duration];
  ctx.clock = fmt.clockSpan(ctx.duration);
  const t = sessionTitle(meta);
  document.title = `${t.title}${t.date ? `, ${t.date}` : ''}: analysis`;
  bar.update({ session: meta, view: 'analysis' });
}

function buildPage() {
  clear(main);
  ui.head = h('div', { class: 'an-head' });
  ui.headMain = null;
  ui.actions = null;
  main.appendChild(ui.head);
  renderHeader();

  ui.nav = h('nav', { class: 'an-nav', attrs: { 'aria-label': 'Sections' } });
  ui.links = {};
  for (const s of SECTION_NAV) {
    const a = h('a', {
      href: `#${s.id}`, text: s.label,
      on: {
        click: (e) => {
          e.preventDefault();
          const el = document.getElementById(s.id);
          if (!el) return;
          const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
          el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
          setActiveLink(s.id);
        },
      },
    });
    ui.links[s.id] = a;
    ui.nav.appendChild(a);
  }
  ui.rangeText = h('span', { class: 'an-range-text', attrs: { role: 'status', 'aria-live': 'polite' } });
  ui.rangeClear = h('button', { class: 'btn sm', attrs: { type: 'button' }, on: { click: () => setRange(null) } }, icon('close', 14), h('span', { text: 'Clear' }));
  ui.range = h('div', { class: 'an-range' }, ui.rangeText, ui.rangeClear);
  ui.sticky = h('div', { class: 'an-sticky' }, ui.nav, ui.range);
  main.appendChild(ui.sticky);
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => {
      document.documentElement.style.setProperty('--an-sticky-h', `${Math.ceil(ui.sticky.getBoundingClientRect().height)}px`);
    }).observe(ui.sticky);
  }

  const body = h('div', { class: 'an-body' });
  main.appendChild(body);
  sections = buildSections(ctx, body);
  spy();
}

function setActiveLink(id) {
  for (const [k, a] of Object.entries(ui.links || {})) {
    if (k === id) a.setAttribute('aria-current', 'true');
    else a.removeAttribute('aria-current');
  }
}

// the section whose top passed under the sticky bar most recently is the current one
function spy() {
  const update = frameThrottle(() => {
    const offset = (ui.sticky ? ui.sticky.getBoundingClientRect().bottom : 120) + 24;
    let current = SECTION_NAV[0].id;
    for (const s of SECTION_NAV) {
      const el = document.getElementById(s.id);
      if (el && el.getBoundingClientRect().top <= offset) current = s.id;
    }
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) current = SECTION_NAV[SECTION_NAV.length - 1].id;
    setActiveLink(current);
  });
  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
}

function chip(label, { value, title, color, muted, kind = 'dot' } = {}) {
  const el = h('span', { class: ['chip', muted ? 'is-muted' : null], attrs: { title: title || null } });
  if (color) el.appendChild(h('span', { class: ['swatch', kind === 'dot' ? 'dot' : null], style: { '--swatch': color } }));
  el.appendChild(h('span', { text: label }));
  if (value != null) el.appendChild(h('span', { class: 'an-chip-value', text: value }));
  return el;
}

/**
 * The heading, meta line and chips follow the meta (a live session refreshes it every 30 s); the
 * action buttons and the export menu are built once, so a refresh keeps an open menu and the focus.
 */
function renderHeader() {
  const meta = ctx.meta;
  const left = h('div', { class: 'an-head-main' });
  left.appendChild(sessionHeading(meta));
  const live = !!(meta.state && meta.state.live);
  ui.statePill = statusPill({ state: live ? 'live' : 'ended' });
  const span = meta.t0 != null && meta.t1 != null
    ? `${fmt.hm(meta.t0)} to ${fmt.hm(meta.t1)} ${fmt.zone(meta.t0)} (${fmt.duration(meta.duration)})`.replace(/\s+\(/, ' (')
    : fmt.na;
  ui.metaLine = h('p', { class: 'an-meta-line' }, ui.statePill, h('span', { class: 'num', text: span }));
  ui.jobPill = null;
  left.appendChild(ui.metaLine);
  ui.chips = h('div', { class: 'an-chips', attrs: { role: 'list', 'aria-label': 'Session data' } });
  left.appendChild(ui.chips);
  renderChips();

  if (!ui.actions) {
    ui.actions = h('div', { class: 'an-actions' });
    ui.liveLink = h('a', { class: 'btn', href: `/live?session=${enc}` });
    ui.actions.appendChild(ui.liveLink);
    ui.refresh = h('button', { class: 'btn', attrs: { type: 'button', title: 'Compute every part of the analysis again from the raw events' }, on: { click: refreshAll } }, icon('refresh', 14), h('span', { text: 'Refresh analysis' }));
    ui.actions.appendChild(ui.refresh);
    ui.actions.appendChild(exportMenu());
  }
  if (ui.liveLink.dataset.live !== String(live)) {
    ui.liveLink.dataset.live = String(live);
    clear(ui.liveLink).append(icon(live ? 'dot' : 'play', 14), h('span', { text: live ? 'Open live' : 'Replay' }));
  }
  if (ui.headMain && ui.headMain.parentNode === ui.head) ui.head.replaceChild(left, ui.headMain);
  else ui.head.append(left, ui.actions);
  ui.headMain = left;
}

function renderChips() {
  if (!ui.chips) return;
  const meta = ctx.meta;
  const m = meta.modalities || {};
  clear(ui.chips);
  const add = (el) => {
    el.setAttribute('role', 'listitem');
    ui.chips.appendChild(el);
  };
  const asr = m.asr && m.asr.recognition;
  add(asr ? chip('ASR', { value: fmt.pct(asr.coverage), title: `${fmt.int(asr.n)} speech buckets, ${fmt.pct(asr.coverage)} coverage` }) : chip('ASR', { value: fmt.na, muted: true, title: 'No speech data' }));
  add(m.ips ? chip('IPS', { value: fmt.pct(m.ips.coverage), title: `${fmt.int(m.ips.n)} badge windows, ${fmt.pct(m.ips.coverage)} coverage` }) : chip('IPS', { value: fmt.na, muted: true, title: 'No badge positions' }));
  add(m.vfa ? chip('VFA', { value: fmt.pct(m.vfa.coverage), title: `${fmt.int(m.vfa.n)} video frame sets, ${fmt.pct(m.vfa.coverage)} coverage` }) : chip('VFA', { value: fmt.na, muted: true, title: 'No video features' }));
  const cams = (meta.devices && meta.devices.cameras) || (ctx.parts.attention && ctx.parts.attention.cameras.map((c) => c.id)) || (meta.ips_cameras || []).map((c) => c.id);
  if (cams && cams.length) add(chip(`${cams.length} ${cams.length === 1 ? 'camera' : 'cameras'}`, { title: cams.join(', ') }));
  const mode = m.asr && m.asr.mode;
  if (mode) add(chip(SPEECH_MODE[mode] || mode));
  for (const id of ctx.roster) {
    const t = ctx.ident.tag(id);
    add(chip(t.label, { color: t.color }));
  }
  const space = ctx.parts.space;
  for (const tg of (space && space.tags) || []) {
    if (!tg.rare || ctx.roster.includes(tg.id)) continue;
    const el = chip(`Tag ${tg.id}`, { color: 'var(--tag-other)', muted: true, title: `seen in ${fmt.pct(tg.present, 1)} of windows: a rare read, left out of the pairs` });
    el.tabIndex = 0;
    const tip = () => ({ title: `Tag ${tg.id}`, rows: [{ value: fmt.pct(tg.present, 1), label: 'of windows', color: 'var(--tag-other)', key: 'dot' }], note: 'A rare read, left out of the pairs and the heat map.' });
    el.addEventListener('pointerenter', (e) => tooltip.show(e, tip()));
    el.addEventListener('pointerleave', () => tooltip.hide());
    el.addEventListener('focus', () => tooltip.show(el, tip()));
    el.addEventListener('blur', () => tooltip.hide());
    add(el);
  }
}

// a small chip in the meta line while any job runs
function renderStatusChip() {
  if (!ui.metaLine) return;
  if (!ui.jobPill) {
    ui.jobPill = statusPill({ state: 'running', text: 'Computing' });
    ui.metaLine.appendChild(ui.jobPill);
  }
  const running = PARTS.map((p) => ctx.status[p]).filter((s) => s && (s.state === 'queued' || s.state === 'running'));
  const video = ['attention', 'timeline'].some((p) => running.includes(ctx.status[p]));
  const light = ['speech', 'space'].some((p) => running.includes(ctx.status[p]));
  ui.jobPill.hidden = !running.length;
  if (running.length) {
    const what = light && video ? 'Computing the analysis' : video ? 'Computing the video analysis' : 'Computing speech and space';
    ui.jobPill.update({ state: 'running', text: what });
  }
  if (ui.refresh) {
    const busy = !!ctx.refreshing;
    ui.refresh.disabled = busy;
    ui.refresh.setAttribute('aria-busy', busy ? 'true' : 'false');
  }
}

// one listener for the page: a click outside the open export menu closes it
document.addEventListener('click', (e) => {
  if (ui.menu && ui.menu.open && !ui.menu.contains(e.target)) ui.menu.open = false;
});

function exportMenu() {
  const details = h('details', { class: 'an-menu' });
  const summary = h('summary', { class: 'btn', attrs: { 'aria-label': 'Export' } }, icon('download', 14), h('span', { text: 'Export' }), icon('chevron-down', 14));
  const pop = h('div', { class: 'an-menu-pop' });
  details.append(summary, pop);
  const fill = () => {
    clear(pop);
    for (const g of exportLinks(ctx)) {
      pop.appendChild(h('p', { class: 'an-menu-group', text: g.group }));
      const ul = h('ul', { class: 'an-dl' });
      for (const it of g.items) {
        ul.appendChild(h('li', {}, it.disabled
          ? h('span', { class: 'muted an-dl-off' }, icon('download', 14), h('span', { text: `${it.label}${it.reason ? `, ${it.reason}` : ', no data'}` }))
          : downloadLink(it.href, it.label)));
      }
      pop.appendChild(ul);
    }
  };
  details.addEventListener('toggle', () => {
    if (details.open) fill();
  });
  ui.menu = details;
  details.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && details.open) {
      details.open = false;
      summary.focus();
    }
  });
  return details;
}

// range and cursor

function sameRange(a, b) {
  if (!a || !b) return a === b;
  return Math.abs(a[0] - b[0]) < 0.05 && Math.abs(a[1] - b[1]) < 0.05;
}

function renderRangeBar() {
  if (!ui.range) return;
  const r = ctx.range;
  if (r) {
    ui.rangeText.textContent = `Range ${ctx.clock(r[0])} to ${ctx.clock(r[1])} (${fmt.duration(r[1] - r[0])})`;
    ui.range.dataset.active = '';
    ui.rangeClear.hidden = false;
  } else {
    clear(ui.rangeText).append(`Whole session, ${fmt.duration(ctx.duration)}.`, h('span', { class: 'an-range-hint', text: ' Drag on the session timeline to choose a range.' }));
    delete ui.range.dataset.active;
    ui.rangeClear.hidden = true;
  }
}

function setRange(range) {
  const next = normalizeRange(range, ctx.span);
  if (sameRange(next, ctx.range)) return;
  ctx.range = next;
  const url = `${window.location.pathname}${window.location.search}${rangeHash(next)}`;
  try {
    window.history.replaceState(null, '', url);
  } catch {
    // a sandboxed page may refuse; the range still applies
  }
  renderRangeBar();
  announce(next ? `Range ${ctx.clock(next[0])} to ${ctx.clock(next[1])}` : 'Range cleared, whole session');
  schedule();
}

window.addEventListener('hashchange', () => {
  if (!ctx.meta) return;
  setRange(parseRangeHash(window.location.hash));
});

function setCursor(t, { reveal = false } = {}) {
  ctx.cursor = Number.isFinite(t) ? t : null;
  if (sections) sections.setCursor(ctx.cursor);
  if (reveal && sections && sections.timelineCard) {
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    sections.timelineCard.el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' });
  }
}

// report parts

function updateRoster() {
  const P = ctx.parts;
  const ids = new Set();
  if (ctx.models.space) for (const id of ctx.models.space.kept) ids.add(id);
  if (P.timeline && P.timeline.roster) for (const id of P.timeline.roster.kept || []) ids.add(String(id));
  if (P.attention) for (const id of P.attention.pupils || []) ids.add(String(id));
  if (P.speech) for (const e of P.speech.entities || []) if (e.kind === 'wearer' && String(e.key).startsWith('tag:')) ids.add(e.key.slice(4));
  const roster = sortTags(Array.from(ids).filter(isPupilTag));
  if (roster.join(',') !== ctx.roster.join(',')) {
    ctx.roster = roster;
    ctx.ident = identity(roster);
  }
  renderChips();
}

function setPartData(part, data) {
  ctx.parts[part] = data;
  try {
    if (part === 'speech') ctx.models.speech = speechModel(data);
    if (part === 'space') ctx.models.space = spaceModel(data);
  } catch {
    ctx.models[part] = null;
  }
  ctx.version += 1;
  updateRoster();
}

async function fetchPart(part, signal) {
  const res = await api(`/api/sessions/${enc}/report/${part}`, { signal });
  if (res.aborted || (signal && signal.aborted)) return { done: true };
  const body = res.data || {};
  const prev = ctx.status[part] || {};
  if (res.status === 200 && body.status === 'ready') {
    const changed = !ctx.parts[part] || body.computed_at !== prev.computed_at;
    ctx.status[part] = { state: 'ready', computed_at: body.computed_at, stale: !!body.stale, jobError: body.job_error || null };
    if (changed && body.data) setPartData(part, body.data);
    schedule();
    // a stale part is being recomputed: look again until the new one is in; when its recompute
    // failed, the server retries later, so look again less often
    if (!body.stale) return { done: true };
    return { done: false, next: body.job_error ? 60000 : 10000 };
  }
  if (res.status === 202) {
    ctx.status[part] = { ...prev, state: body.status || 'running', progress: body.progress || {}, runner: body.runner };
    schedule();
    return { done: false, next: 2000 };
  }
  if (res.status === 500 && body.status === 'error') {
    ctx.status[part] = { ...prev, state: 'error', error: body.error || 'The job failed.' };
    schedule();
    return { done: true };
  }
  if (res.status === 404 || res.status === 400) {
    ctx.status[part] = { ...prev, state: 'error', error: res.error };
    schedule();
    return { done: true };
  }
  // InfluxDB or the network is down: keep what is shown and try again
  ctx.status[part] = ctx.parts[part] ? { ...prev, state: 'ready' } : { ...prev, state: 'error', error: res.error };
  schedule();
  return { done: false, next: 10000 };
}

function startPart(part) {
  if (loaders[part]) loaders[part].abort();
  const ctrl = new AbortController();
  loaders[part] = ctrl;
  // no visibility gate: a job's progress is cheap to ask for and a part polls only until it is in
  poll(() => fetchPart(part, ctrl.signal), {
    interval: (r) => (r && r.next) || 2000,
    until: (r) => !r || r.done,
    signal: ctrl.signal,
  });
}

/** Run again after a failure: resubmit the job and follow its parts. */
async function runJob(job) {
  const parts = job === 'all' ? PARTS : PARTS.filter((p) => PART_JOB[p] === job);
  for (const p of parts) if (!ctx.parts[p]) ctx.status[p] = { state: 'queued', progress: { step: 'Starting' } };
  schedule();
  const res = await api(`/api/sessions/${enc}/report/refresh`, { method: 'POST', body: { job } });
  if (!res.ok) {
    for (const p of parts) if (!ctx.parts[p]) ctx.status[p] = { state: 'error', error: res.error };
    schedule();
    return;
  }
  for (const p of parts) startPart(p);
}

/**
 * Refresh: recompute every part. The old parts stay on screen (held at half opacity) while the
 * meta says the jobs run; each part is fetched again once its job is done.
 */
async function refreshAll() {
  if (ctx.refreshing) return;
  ctx.refreshing = true;
  for (const p of PARTS) ctx.status[p] = { ...ctx.status[p], state: 'queued', progress: { step: 'Starting' } };
  schedule();
  const res = await api(`/api/sessions/${enc}/report/refresh`, { method: 'POST', body: { job: 'all' } });
  if (!res.ok) {
    ctx.refreshing = false;
    for (const p of PARTS) ctx.status[p] = { ...ctx.status[p], state: ctx.parts[p] ? 'ready' : 'error', error: res.error };
    schedule();
    announce(`Refresh failed: ${res.error}`);
    return;
  }
  announce('Recomputing the analysis.');
  const pending = new Set(['light', 'video']);
  let rounds = 0;
  const ctrl = new AbortController();
  await poll(async () => {
    rounds += 1;
    const m = await api(`/api/sessions/${enc}`, { signal: ctrl.signal });
    if (!m.ok || !m.data) return rounds > 150;
    const rep = m.data.report || {};
    for (const job of Array.from(pending)) {
      const state = rep[job];
      if (state === 'running' && rounds > 0) {
        for (const p of PARTS.filter((x) => PART_JOB[x] === job)) {
          ctx.status[p] = { ...ctx.status[p], state: 'running', progress: { step: job === 'video' ? 'Recomputing the video analysis' : 'Recomputing speech and space' } };
        }
        continue;
      }
      if (rounds < 2) continue;
      pending.delete(job);
      for (const p of PARTS.filter((x) => PART_JOB[x] === job)) {
        ctx.status[p] = { ...ctx.status[p], state: ctx.parts[p] ? 'ready' : 'loading' };
        startPart(p);
      }
    }
    schedule();
    return pending.size === 0 || rounds > 300;
  }, { interval: 2000, until: (done) => !!done, signal: ctrl.signal });
  ctx.refreshing = false;
  schedule();
  announce('The analysis is up to date.');
}

// a live session grows: refresh its meta now and then so the span follows
function followLiveMeta() {
  poll(async () => {
    const res = await api(`/api/sessions/${enc}`);
    if (res.ok && res.data && res.data.id) {
      const wasLive = !!(ctx.meta.state && ctx.meta.state.live);
      applyMeta(res.data);
      ctx.version += 1;
      renderHeader();
      renderRangeBar();
      schedule();
      const live = !!(res.data.state && res.data.state.live);
      if (wasLive && !live) for (const p of PARTS) startPart(p);
      return live;
    }
    return true;
  }, { interval: 30000, until: (live) => !live, visible: true });
}
