/**
 * The live page: one session as it runs (follow mode), or an ended one replayed at 1 to 16 times
 * real time from any moment; an ended session opens paused at its start. The page opens one
 * Server-Sent Events stream at a time (/api/sessions/<sid>/stream), feeds every batch into a
 * LiveModel (live-model.js) and redraws from the model at most four times a second; the heavier
 * panels (KPIs, who looks at whom, speaking share, pairs, transcript) once a second. Replay keeps no
 * stream open while paused: Play reconnects at the paused moment and backfills only what the last
 * connection had not sent yet, a seek clears the model and reconnects with up to five minutes of
 * history (none before the session's start). A dropped stream reconnects after 1, 2, 5, then every
 * 10 seconds; a page back from the back/forward cache reconnects at once. The Cameras card plays live
 * video in follow mode and, in the replay of an ended session, each camera's recorded file at the
 * replay clock (cameras.js), from the list the recordings route gives; the Sound control plays one of
 * the session's microphone files at the same clock (sound.js).
 */

import {
  h, clear, fmt, api, theme, topbar, identity, emptyState, statusPill, segmented, sessionIdFromUrl, sessionParamInvalid,
  sessionHeading, sessionTitle, icon, sortTags, isPupilTag, announce, tooltip,
} from './core.js';
import { card, timeline, barList, networkGraph, statTile, bullet } from './charts.js';
import { floorMap } from './floormap.js';
import {
  LiveModel, VoiceColors, projectCameras, roomExtent, extentContains, extentUnion, pairList, CATEGORIES, CATEGORY_LABELS,
  GROUP_LABEL_RE, JA_PAIR_MIN_SHARE, TAG_MEMORY_SECONDS,
} from './live-model.js';
import { cameraWall, cameraItems, mediaOriginFor, probeMediaOrigin, awaitMediaAnswers } from './cameras.js';
import { ReplaySound, soundSources, defaultSource, MAX_SOUND_SPEED } from './sound.js';

const SPEEDS = [1, 2, 4, 8, 16];
const WINDOWS = [
  { value: 120, label: '2 min' },
  { value: 300, label: '5 min' },
  { value: 600, label: '10 min' },
];
const BACKOFF = [1, 2, 5, 10];
const BACKFILL = 300;
const FRAME_MS = 250;
const HEAVY_MS = 1000;
const STALE_AGE = 10;
const LOOKS_SECONDS = 60;
const MEDIA_REFRESH_MS = 60000;
// a recordings list that could not be read is asked for again after this long
const RECORDINGS_RETRY_MS = 60000;
const GAZE_FILL = {
  partner_face: 'var(--ord-4)',
  partner_hands: 'var(--ord-3)',
  other_people: 'var(--ord-2)',
  task: 'var(--ord-1)',
  elsewhere: 'var(--grid)',
  unreadable: 'hatch',
};
// pairs are relations between two tags, as on the analysis page: no hue of their own, drawn in ink
// as small multiples and named by the two tag dots and 'Tag 0 and Tag 1'
const PAIR_INK = 'var(--ink-2)';
// a pair's distance lane in the activity timeline
const PAIR_LANE_HEIGHT = 40;
// voice and tag colour slots follow the analysis once the speech and timeline parts answer; the page
// waits this long for those answers before it colours voices and tags by the order it sees them
const COLOUR_WAIT_MS = 3000;
// a live session (or one whose speech part was not ready) asks again this often, for voices not drawn yet
const SPEECH_RECHECK_MS = 120000;
// a session without measurements yet asks for its meta this often
const EMPTY_POLL_MS = 5000;

const finite = (v) => typeof v === 'number' && Number.isFinite(v);
const nowWall = () => Date.now() / 1000;

const S = {
  sid: null,
  meta: null,
  t0: null,
  t1: null,
  live: false,
  model: null,
  epoch: 0,
  ident: null,
  voices: new VoiceColors(),
  // the colour sources (speech and timeline parts) were asked for and have not answered
  coloursPending: false,
  coloursLoading: false,
  colourRoster: new Set(),
  tagsProvisional: false,
  speechReady: false,
  roster: [],
  mode: 'replay',
  intent: 'idle',
  playing: false,
  speed: 1,
  clock: null,
  clockPerf: 0,
  shown: null,
  connectAt: null,
  backfilling: false,
  backfillSecs: 0,
  // replay: the stream clock up to which the server has sent every record (the drawn clock runs
  // ahead of it between batches); null while the model lacks its history (five minutes, or back to t0)
  sent: null,
  preview: false,
  ended: false,
  es: null,
  connId: 0,
  connState: 'idle',
  attempt: 0,
  retryTimer: 0,
  retryAt: 0,
  lastError: null,
  lag: null,
  lastEvent: null,
  skew: 0,
  floor: null,
  cameras: [],
  extent: null,
  window: 300,
  scrubbing: false,
  scrubValue: null,
  trails: true,
  distances: true,
  media: null,
  mediaAt: 0,
  // the recordings route's answer (the replay of an ended session plays the camera files it lists)
  recordings: null,
  recordingsAt: 0,
  recordingsBusy: false,
  // where the recorded files load from: the dashboard's media origins that answered, in its order
  // (empty: the page's own origin, with fewer files at once), and the last probe of each origin
  mediaOrigins: [],
  mediaProbes: new Map(),
  // which mediaOrigins call the list belongs to, so a port that answers late joins only its own
  mediaOriginsGen: 0,
  // the Sound control: the microphone the viewer picked (its source key, 'off', or null for the
  // default), and the mute toggle
  soundPick: null,
  soundMuted: false,
  sound: null,
  camsOpen: false,
  raf: 0,
  lastFrame: 0,
  heavyAt: 0,
  heavyKey: '',
  laneKey: '',
  force: true,
  gotData: false,
};

S.ident = tagIdentity();

const UI = {};

// stream

function streamUrl(params) {
  return `/api/sessions/${encodeURIComponent(S.sid)}/stream?${new URLSearchParams(params)}`;
}

function stopStream() {
  S.connId += 1;
  if (S.es) {
    try {
      S.es.close();
    } catch {
      // already closed
    }
    S.es = null;
  }
  clearTimeout(S.retryTimer);
  S.retryTimer = 0;
}

/** open the stream; every handler ignores events of an older connection. */
function connect({ mode, at = null, backfill = 0, preview = false }) {
  stopStream();
  const id = S.connId;
  S.mode = mode;
  S.preview = preview;
  S.connState = 'connecting';
  S.connectAt = at;
  S.backfillSecs = Math.max(0, Math.round(backfill));
  S.backfilling = S.backfillSecs > 0;
  S.shown = null;
  const params = { mode, backfill: String(S.backfillSecs) };
  if (mode === 'replay') {
    params.at = Number(at).toFixed(3);
    params.speed = String(S.speed);
  }
  let es;
  try {
    es = new EventSource(streamUrl(params));
  } catch (err) {
    scheduleReconnect(`Could not open the live stream (${(err && err.message) || 'error'}).`);
    return;
  }
  S.es = es;
  const on = (name, fn) => es.addEventListener(name, (e) => {
    if (id !== S.connId) return;
    let data = null;
    try {
      data = JSON.parse(e.data);
    } catch {
      return;
    }
    fn(data);
    requestFrame();
  });
  on('hello', onHello);
  on('batch', onBatch);
  on('floor', onFloor);
  on('status', onStatus);
  on('tick', onTick);
  on('end', onEnd);
  es.onerror = () => {
    if (id !== S.connId) return;
    // EventSource would retry with the same URL (the original `at`), so the page reconnects itself
    stopStream();
    scheduleReconnect('The live stream was interrupted.');
  };
  requestFrame();
}

function onHello(d) {
  S.connState = 'open';
  S.attempt = 0;
  S.lastError = null;
  if (finite(d.t0)) S.t0 = d.t0;
  if (finite(d.t1)) S.t1 = Math.max(S.t1 || d.t1, d.t1);
  S.live = !!d.live;
  if (d.group_id) S.model.configure({ groupId: d.group_id });
  if (d.floor && !S.floor) setFloor(d.floor);
  if (finite(d.clock)) {
    // the backfill ends at the hello clock (replay: the requested moment; follow: the newest event)
    S.connectAt = d.clock;
    S.clock = d.clock;
    S.clockPerf = performance.now();
    // a replay without a backfill goes on from what the connection before it sent
    if (S.mode === 'replay' && !S.backfilling) S.sent = d.clock;
  }
  S.model.configure({ t0: S.t0, t1: S.t1 });
}

function onBatch(d) {
  if (S.preview && !d.backfill) {
    // the backfill is in; a paused view keeps no stream open
    finishPreview();
    return;
  }
  const added = S.model.addBatch(d);
  if (!S.gotData && (added.ips || added.vfa || added.asr || added.tr || d.backfill)) {
    // the first records after a (re)start leave the loading state at once
    S.gotData = true;
    S.force = true;
  }
  if (d.backfill) {
    if (finite(d.clock) && S.connectAt != null && d.clock >= S.connectAt - 0.01) {
      S.backfilling = false;
      S.clockPerf = performance.now();
      if (S.mode === 'replay') S.sent = S.connectAt;
      if (S.preview) finishPreview();
    }
  } else S.backfilling = false;
  if (S.mode === 'follow') {
    if (finite(d.clock)) S.clock = S.clock == null ? d.clock : Math.max(S.clock, d.clock);
  } else if (!d.backfill && finite(d.clock)) {
    S.clock = d.clock;
    S.clockPerf = performance.now();
    S.sent = d.clock;
  }
  if (finite(d.clock) && S.t1 != null && d.clock > S.t1) S.t1 = d.clock;
}

function onFloor(basis) {
  if (!S.floor) setFloor(basis);
}

function onStatus(d) {
  S.live = !!d.live;
  // status comes once the backfill is out (follow mode also sends it when no record is new)
  if (S.mode === 'follow') S.backfilling = false;
  S.lag = finite(d.lag) ? d.lag : null;
  if (finite(d.last_event)) {
    S.lastEvent = d.last_event;
    if (S.t1 == null || d.last_event > S.t1) S.t1 = d.last_event;
    if (finite(d.lag)) S.skew = d.last_event + d.lag - nowWall();
  }
  if (finite(d.clock)) {
    if (S.mode === 'follow') S.clock = S.clock == null ? d.clock : Math.max(S.clock, d.clock);
    else if (!S.preview && !S.backfilling) {
      // a stalled replay repeats its clock in every status: the drawn clock holds (clockRunning)
      if (S.clock == null || d.clock > S.clock) S.clockPerf = performance.now();
      S.clock = d.clock;
      S.sent = d.clock;
    }
  }
}

/**
 * a playing replay's clock moved with no record to send (the server says so at least once a second):
 * the clock goes on as after a batch, so the recorded video and the sound keep playing through
 * stretches with few records. A stalled replay sends none, and its clock holds (clockRunning).
 */
function onTick(d) {
  if (S.mode !== 'replay' || !finite(d.clock)) return;
  if (S.preview) {
    // the backfill is in; a paused view keeps no stream open
    finishPreview();
    return;
  }
  S.backfilling = false;
  S.clock = d.clock;
  S.clockPerf = performance.now();
  S.sent = d.clock;
  if (S.t1 != null && d.clock > S.t1) S.t1 = d.clock;
}

function onEnd(d) {
  stopStream();
  if (d && d.reason === 'replay_end') {
    S.connState = 'idle';
    S.playing = false;
    S.intent = 'idle';
    S.ended = true;
    if (S.t1 != null) S.clock = Math.max(S.clock ?? S.t1, S.t1);
    S.shown = null;
    announce('The replay reached the end of the session.');
    return;
  }
  scheduleReconnect((d && d.message) || 'The live stream stopped with an error.');
}

function finishPreview() {
  stopStream();
  S.preview = false;
  S.backfilling = false;
  S.connState = 'idle';
  S.intent = 'idle';
  S.force = true;
}

function scheduleReconnect(message) {
  clearTimeout(S.retryTimer);
  const wait = BACKOFF[Math.min(S.attempt, BACKOFF.length - 1)];
  S.attempt += 1;
  S.connState = 'reconnecting';
  S.lastError = message;
  S.retryAt = performance.now() + wait * 1000;
  S.retryTimer = setTimeout(reconnectNow, wait * 1000);
  // the meta request says why (an InfluxDB outage answers 503 with the reason)
  api(`/api/sessions/${encodeURIComponent(S.sid)}`).then((res) => {
    if (S.connState !== 'reconnecting') return;
    if (!res.ok && res.error) S.lastError = res.status === 503 ? res.error : `${message} ${res.error}`;
    requestFrame();
  });
  requestFrame();
}

function reconnectNow() {
  clearTimeout(S.retryTimer);
  S.retryTimer = 0;
  const empty = S.model.newestTime() == null;
  if (S.intent === 'follow') {
    // fetch what was written while the stream was away; the model drops what it already holds
    const gap = S.clock != null && S.live ? Math.ceil(serverNow() - S.clock) + 10 : BACKFILL;
    connect({ mode: 'follow', backfill: empty ? BACKFILL : Math.max(30, Math.min(900, gap)) });
  } else if (S.intent === 'play') {
    const at = S.clock ?? S.t0;
    connect({ mode: 'replay', at, backfill: replayBackfill(at) });
  } else if (S.intent === 'preview') {
    const at = S.connectAt ?? S.clock;
    connect({ mode: 'replay', at, backfill: historyBefore(at), preview: true });
  } else S.connState = 'idle';
  requestFrame();
}

/**
 * the backfill of a replay connection at `at`: its history while the model lacks it (a seek or the
 * opening preview still loading), else what the last connection had not sent yet (the drawn clock
 * runs ahead of the batches), plus a second of overlap the model drops.
 */
function replayBackfill(at) {
  if (S.sent == null) return historyBefore(at);
  return at > S.sent ? Math.ceil(at - S.sent) + 1 : 0;
}

/**
 * the history a replay loads before the moment `at`: five minutes, never reaching back before the
 * session's start (the second it adds holds the records stamped at the start itself).
 */
function historyBefore(at) {
  if (S.t0 == null || !finite(at)) return BACKFILL;
  return Math.max(1, Math.min(BACKFILL, Math.ceil(at - S.t0) + 1));
}

function serverNow() {
  return nowWall() + S.skew;
}

// actions

function resetModel() {
  S.model.reset();
  if (S.floor) S.model.setFloor(S.floor);
  S.epoch += 1;
  S.sent = null;
  S.gotData = false;
  S.heavyKey = '';
  S.laneKey = '';
  S.force = true;
}

function startFollow() {
  resetModel();
  S.mode = 'follow';
  S.intent = 'follow';
  S.playing = false;
  S.ended = false;
  S.attempt = 0;
  connect({ mode: 'follow', backfill: BACKFILL });
}

function startPreview(at) {
  resetModel();
  S.mode = 'replay';
  S.intent = 'preview';
  S.playing = false;
  S.ended = false;
  S.clock = at;
  S.attempt = 0;
  connect({ mode: 'replay', at, backfill: historyBefore(at), preview: true });
}

/**
 * where an ended session opens (and the banner's Replay this session goes): paused at its start,
 * with nothing before it loaded; Play plays it from there, Jump to end goes to its last moments.
 */
function openingMoment() {
  return S.t0;
}

function atEnd() {
  return S.mode === 'replay' && (S.ended || (S.t1 != null && S.clock != null && S.clock >= S.t1 - 0.5));
}

function play() {
  if (S.mode !== 'replay') return;
  if (atEnd()) {
    replayFromStart();
    return;
  }
  const at = displayNow() ?? S.clock ?? S.t0;
  S.clock = at;
  S.playing = true;
  S.ended = false;
  S.intent = 'play';
  S.attempt = 0;
  // a preview still loading asks for its whole history again; the model drops what it holds
  connect({ mode: 'replay', at, backfill: replayBackfill(at) });
}

function pause() {
  S.clock = displayNow() ?? S.clock;
  stopStream();
  S.playing = false;
  S.intent = 'idle';
  S.connState = 'idle';
  S.shown = null;
  requestFrame();
}

function setSpeed(v) {
  S.speed = v;
  if (S.mode === 'replay' && S.playing) {
    const at = displayNow() ?? S.clock;
    S.clock = at;
    connect({ mode: 'replay', at, backfill: replayBackfill(at) });
  }
  requestFrame();
}

function seek(target) {
  if (!finite(target) || S.t0 == null) return;
  const t = Math.max(S.t0, Math.min(S.t1 ?? target, target));
  resetModel();
  S.ended = false;
  if (S.mode === 'follow') {
    // looking back from live plays from there; Back to live returns
    S.mode = 'replay';
    S.playing = true;
  }
  S.clock = t;
  S.clockPerf = performance.now();
  S.attempt = 0;
  if (S.playing) {
    S.intent = 'play';
    connect({ mode: 'replay', at: t, backfill: historyBefore(t) });
  } else startPreview(t);
  announce(`Moved to ${fmt.clock(t - S.t0)}`);
}

function replayFromStart() {
  resetModel();
  S.mode = 'replay';
  S.playing = true;
  S.ended = false;
  S.intent = 'play';
  S.clock = S.t0;
  S.clockPerf = performance.now();
  S.attempt = 0;
  connect({ mode: 'replay', at: S.t0, backfill: historyBefore(S.t0) });
}

function setFloor(basis) {
  S.floor = basis;
  S.model.setFloor(basis);
  S.cameras = projectCameras((S.meta && S.meta.ips_cameras) || [], basis);
  if (UI.room) UI.room.map.setCameras(S.cameras);
  S.extent = null;
  S.force = true;
}

/** the stream clock to draw: interpolated between batches while a replay plays (never ahead by more than 1.5 s of wall time). */
function displayNow() {
  if (S.clock == null) return null;
  let c = S.clock;
  if (S.mode === 'replay' && S.playing && S.connState === 'open' && !S.backfilling && !S.preview) {
    const dt = Math.min((performance.now() - S.clockPerf) / 1000, 1.5);
    c = S.clock + dt * S.speed;
    if (S.t1 != null) c = Math.min(c, S.t1);
    if (S.shown != null && c < S.shown) c = S.shown;
    S.shown = c;
  }
  return c;
}

// layout

function chip(color, label, title) {
  return h('span', { class: 'chip', title: title || null }, h('span', { class: 'swatch dot', style: { '--swatch': color || 'var(--ink-2)' } }), h('span', { text: label }));
}

function toggleButton(label, pressed, onToggle, title) {
  const btn = h('button', { class: 'btn sm', attrs: { type: 'button', 'aria-pressed': pressed ? 'true' : 'false', title: title || null } }, h('span', { text: label }));
  btn.addEventListener('click', () => {
    const next = btn.getAttribute('aria-pressed') !== 'true';
    btn.setAttribute('aria-pressed', next ? 'true' : 'false');
    onToggle(next);
  });
  return btn;
}

function build(root) {
  clear(root);
  const meta = S.meta;

  // session bar
  UI.pill = statusPill({ state: 'paused' });
  UI.lag = h('span', { class: 'num' });
  UI.stateNote = h('span', { class: 'muted' });
  UI.clock = h('div', { class: 'live-clock-value', attrs: { role: 'timer', 'aria-live': 'off' }, text: fmt.na });
  UI.clockSub = h('div', { class: 'live-clock-sub' });
  root.appendChild(h('section', { class: 'live-head' },
    h('div', { style: { minWidth: '0' } }, sessionHeading(meta), h('div', { class: 'live-state' }, UI.pill, UI.lag, UI.stateNote)),
    h('div', { class: 'live-clock' }, UI.clock, UI.clockSub)));

  // controls
  UI.followBtn = h('button', { class: 'btn', attrs: { type: 'button', 'aria-pressed': 'true', title: 'Following the newest data' } }, h('span', { class: 'dot', style: { '--swatch': 'var(--good)' } }), h('span', { text: 'Follow' }));
  UI.followBtn.addEventListener('click', () => {
    if (S.mode !== 'follow') startFollow();
  });
  UI.playBtn = h('button', { class: 'btn icon-btn', attrs: { type: 'button', 'aria-label': 'Play' } }, icon('play', 14));
  UI.playBtn.addEventListener('click', () => {
    // the click lets the replay's sound play (autoplay rules want a gesture), Pause's too
    S.sound.arm();
    if (S.playing) pause();
    else play();
  });
  UI.speed = segmented({
    label: 'Replay speed',
    value: S.speed,
    options: SPEEDS.map((v) => ({ value: v, label: `${v}x` })),
    onChange: (v) => setSpeed(v),
  });
  buildSound();
  UI.scrub = h('input', {
    attrs: { type: 'range', min: 0, max: 1, step: 1, 'aria-label': 'Position in the session' },
    value: '0',
  });
  UI.scrubStart = h('span', { class: 'scrub-label', text: '00:00' });
  UI.scrubEnd = h('span', { class: 'scrub-label' });
  let scrubTimer = 0;
  UI.scrub.addEventListener('input', () => {
    S.scrubbing = true;
    S.scrubValue = Number(UI.scrub.value);
    requestFrame();
  });
  UI.scrub.addEventListener('change', () => {
    clearTimeout(scrubTimer);
    S.scrubbing = true;
    S.scrubValue = Number(UI.scrub.value);
    // arrow keys fire a change per step: seek once they pause
    scrubTimer = setTimeout(() => {
      scrubTimer = 0;
      S.scrubbing = false;
      seek(S.t0 + Number(UI.scrub.value));
    }, 250);
  });
  UI.jumpEnd = h('button', { class: 'btn', attrs: { type: 'button' }, text: 'Jump to end' });
  UI.jumpEnd.addEventListener('click', () => seek(S.t1));
  UI.backLive = h('button', { class: 'btn', attrs: { type: 'button' } }, h('span', { class: 'dot', style: { '--swatch': 'var(--good)' } }), h('span', { text: 'Back to live' }));
  UI.backLive.addEventListener('click', () => startFollow());
  UI.fromStart = h('button', { class: 'btn', attrs: { type: 'button' } }, icon('reconnect', 14), h('span', { text: 'Replay from start' }));
  UI.fromStart.addEventListener('click', () => {
    S.sound.arm();
    replayFromStart();
  });
  root.appendChild(h('div', { class: 'live-controls', attrs: { role: 'group', 'aria-label': 'Playback' } },
    UI.followBtn, UI.playBtn, UI.speed, UI.sound.el,
    h('div', { class: 'scrub' }, UI.scrubStart, UI.scrub, UI.scrubEnd),
    UI.fromStart, UI.jumpEnd, UI.backLive));

  // banner (errors, reconnecting, ended)
  UI.banner = h('div', { class: 'live-banner', hidden: true, attrs: { role: 'status' } });
  root.appendChild(UI.banner);

  // health strip
  UI.health = h('div', { class: 'health', attrs: { 'aria-label': 'Data health' } });
  root.appendChild(UI.health);

  // KPIs
  UI.kpiCaption = h('p', { class: 'kpi-caption' });
  root.appendChild(UI.kpiCaption);
  UI.kpi = {};
  const kpiRow = h('div', { class: 'kpis' });
  for (const [key, label] of [['speech', 'Speech activity'], ['switches', 'Turn switches'], ['balance', 'Speaking balance'], ['distance', 'Median distance'], ['ja', 'Joint attention above baseline'], ['social', 'Social gaze']]) {
    UI.kpi[key] = statTile({ label, pending: 'Waiting for data' });
    kpiRow.appendChild(UI.kpi[key]);
  }
  UI.kpi.ja.title = `The pairs' share of frames with the gaze on the same spot, minus each pair's own rate 20 to 40 s earlier, pooled over the pairs seen together in at least ${Math.round(JA_PAIR_MIN_SHARE * 60)} frames a minute and weighted by their frames. Live counts only the frames in which the camera read both badges, or kept them on their tracks for at most ${TAG_MEMORY_SECONDS} s after a read.`;
  UI.kpi.social.title = 'Share of the camera frames each pupil was seen in with the gaze on a partner\'s face or hands, unreadable frames included, all pupils pooled.';
  root.appendChild(kpiRow);

  const grid = h('div', { class: 'grid stretch' });
  root.appendChild(grid);

  // room
  const trailsBtn = toggleButton('Trails', S.trails, (v) => { S.trails = v; requestFrame(); }, 'Show the last 30 s of each badge');
  const distBtn = toggleButton('Distances', S.distances, (v) => { S.distances = v; requestFrame(); }, 'Show the pair lines; hover or select a badge for its distances');
  const roomCard = card({ title: 'Room', subtitle: 'Badge positions from above, 30 s trails, facing arrows', span: 7, actions: [trailsBtn, distBtn] });
  const mapEl = h('div');
  roomCard.body.appendChild(mapEl);
  const map = floorMap(mapEl, { cameras: [], height: 400, label: 'Room plan, top view in metres: badges, trails and facing arrows' });
  UI.room = { card: roomCard, map };
  grid.appendChild(roomCard.el);

  // transcript
  const trCard = card({ title: 'Transcript', subtitle: 'Newest at the bottom', span: 5 });
  const list = h('ol', { attrs: { 'aria-label': 'Transcript chunks' } });
  const emptyLine = h('p', { class: 'muted', text: 'No transcript yet.', hidden: true });
  const pending = h('div', { class: 'tr-pending', hidden: true }, h('span', { class: 'spinner' }), h('span', { text: 'Transcribing...' }));
  const jumpBtn = h('button', { class: 'btn sm', attrs: { type: 'button' } }, icon('chevron-down', 12), h('span', { text: 'Jump to latest' }));
  const jump = h('div', { class: 'tr-jump', hidden: true }, jumpBtn);
  const scroller = h('div', { class: 'transcript', attrs: { tabindex: 0, 'aria-label': 'Transcript, scrollable' } }, emptyLine, list, pending, jump);
  trCard.body.appendChild(scroller);
  UI.tr = { card: trCard, list, emptyLine, pending, jump, scroller, nodes: new Map(), epoch: -1, stick: true };
  scroller.addEventListener('scroll', () => {
    const atBottom = scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 24;
    UI.tr.stick = atBottom;
    if (atBottom) jump.hidden = true;
  }, { passive: true });
  jumpBtn.addEventListener('click', () => {
    UI.tr.stick = true;
    jump.hidden = true;
    scroller.scrollTop = scroller.scrollHeight;
  });
  grid.appendChild(trCard.el);

  // activity
  UI.windowSeg = segmented({
    label: 'Time window',
    size: 'sm',
    value: S.window,
    options: WINDOWS,
    onChange: (v) => {
      S.window = v;
      S.model.configure({ keep: v + 120 });
      S.heavyKey = '';
      S.laneKey = '';
      S.force = true;
      requestFrame();
    },
  });
  const actCard = card({
    title: 'Activity',
    subtitle: 'Speech, speakers, presence, gaze and pair distance over the trailing window',
    span: 12,
    actions: [UI.windowSeg],
    legend: [
      ...CATEGORIES.map((c) => ({ label: CATEGORY_LABELS[c], color: GAZE_FILL[c], kind: GAZE_FILL[c] === 'hatch' ? 'hatch' : 'rect' })),
    ],
  });
  const tlEl = h('div');
  actCard.body.appendChild(tlEl);
  UI.act = { card: actCard, el: tlEl, tl: null };
  grid.appendChild(actCard.el);

  // bottom row
  const looksCard = card({ title: 'Who looks at whom', subtitle: `Share of each pupil's frames on a partner's face or hands, last ${LOOKS_SECONDS} s`, span: 4, table: looksTable });
  const looksEl = h('div');
  looksCard.body.appendChild(looksEl);
  UI.looks = { card: looksCard, el: looksEl, chart: null, data: null };
  grid.appendChild(looksCard.el);

  const shareCard = card({ title: 'Speaking share', subtitle: 'Trailing window; the tick marks the share since this view started', span: 4, table: shareTable });
  const shareEl = h('div');
  shareCard.body.appendChild(shareEl);
  UI.share = { card: shareCard, el: shareEl, chart: null, data: null };
  grid.appendChild(shareCard.el);

  const pairsCard = card({ title: 'Pairs', subtitle: `Distance now, time within 1 m, joint attention over the last ${LOOKS_SECONDS} s`, span: 4 });
  const pairsTableEl = h('div', { class: 'pairs-table' });
  const jaTitle = h('p', { class: 'pairs-sub', text: `Joint attention, last ${LOOKS_SECONDS} s, against the pair's baseline` });
  const jaEl = h('div');
  pairsCard.body.append(pairsTableEl, jaTitle, jaEl);
  UI.pairs = { card: pairsCard, tableEl: pairsTableEl, jaTitle, jaEl, chart: null, sig: '' };
  grid.appendChild(pairsCard.el);

  // cameras
  const camToggle = h('button', { class: 'btn sm', attrs: { type: 'button', 'aria-expanded': 'false' } }, icon('camera', 14), h('span', { text: 'Show cameras' }));
  const overlayBtn = toggleButton('Overlay', true, (v) => UI.cams.wall.setOptions({ overlay: v }), 'Draw the skeletons over the video');
  const syncBtn = toggleButton('Sync overlay', true, (v) => UI.cams.wall.setOptions({ sync: v }), 'Hold the video back so the skeletons line up');
  const videoToggles = h('span', { class: 'cam-toggles', hidden: true }, overlayBtn, syncBtn);
  const camCard = card({ title: 'Cameras', subtitle: 'Skeletons, gaze rays and AprilTags of the newest frame set, pupils in their tag colour; live video when the session streams it, the recorded video in a replay', span: 12, actions: [videoToggles, camToggle] });
  // the kept line is dashed by the page's style (its data-key)
  const camLegend = [
    { key: 'cam-kept', label: `Dashed: badge not read now, kept from a read under ${TAG_MEMORY_SECONDS} s ago`, color: 'var(--ink-2)', kind: 'line' },
    { key: 'cam-untagged', label: `Grey: no badge read now or on its track in the last ${TAG_MEMORY_SECONDS} s`, color: 'var(--tag-other)', kind: 'line' },
  ];
  const wall = cameraWall({ tagColor: (t) => S.ident.tag(t).color, tagLabel: (t) => `Tag ${t}` });
  camCard.body.appendChild(wall.el);
  camCard.body.hidden = true;
  camToggle.addEventListener('click', () => {
    S.camsOpen = !S.camsOpen;
    camToggle.setAttribute('aria-expanded', S.camsOpen ? 'true' : 'false');
    camToggle.lastChild.textContent = S.camsOpen ? 'Hide cameras' : 'Show cameras';
    // the card's own state shows its body again, so a closed card goes back to ready first
    if (!S.camsOpen) camCard.setState('ready');
    camCard.body.hidden = !S.camsOpen;
    camCard.setLegend(S.camsOpen ? camLegend : null);
    wall.setExpanded(S.camsOpen);
    if (S.camsOpen) {
      loadMedia(true);
      // files downloaded since the card was last open play too
      if (endedReplay()) loadRecordings(true);
    }
    S.force = true;
    requestFrame();
  });
  UI.cams = { card: camCard, wall, toggle: camToggle, videoToggles, syncBtn };
  grid.appendChild(camCard.el);

  for (const c of [roomCard, trCard, actCard, looksCard, shareCard, pairsCard]) c.setState('loading', 'Connecting to the live stream');
}

async function loadMedia(force = false) {
  if (!force && performance.now() - S.mediaAt < MEDIA_REFRESH_MS) return;
  S.mediaAt = performance.now();
  const res = await api(`/api/sessions/${encodeURIComponent(S.sid)}/media`);
  if (res.ok && res.data) S.media = res.data;
  else S.media = { webrtc: null, streams: [], reason: res.error ? `The stream server could not be asked: ${res.error}` : null };
  S.force = true;
  requestFrame();
}

/** the replay of an ended session: its cameras play their recorded files */
function endedReplay() {
  return S.mode === 'replay' && !S.live;
}

/**
 * ask for the session's recordings (once, when the replay of an ended session opens, and again when
 * the Cameras card is opened; a failed answer again after a minute). The tiles play the video files,
 * the Sound control the audio files.
 */
async function loadRecordings(force = false) {
  if (S.recordingsBusy) return;
  if (!force && S.recordings && !(S.recordings.failed && performance.now() - S.recordingsAt >= RECORDINGS_RETRY_MS)) return;
  S.recordingsBusy = true;
  S.recordingsAt = performance.now();
  const res = await api(`/api/sessions/${encodeURIComponent(S.sid)}/recordings`);
  const d = res.ok && res.data && typeof res.data === 'object' ? res.data : null;
  // the files wait for the media ports' answers, so none loads from the page's origin first (a
  // dashboard of one media port names media_port alone)
  if (d && d.enabled) {
    const ports = Array.isArray(d.media_ports) ? d.media_ports : [d.media_port];
    S.mediaOrigins = await mediaOrigins(ports, d.media_instance);
  }
  S.recordingsBusy = false;
  if (d) {
    S.recordings = {
      enabled: !!d.enabled,
      files: (Array.isArray(d.files) ? d.files : []).filter((f) => f && (f.modality === 'video' || f.modality === 'audio')),
      reason: typeof d.reason === 'string' ? d.reason : null,
      failed: false,
    };
  } else {
    const why = res.error ? ` (${String(res.error).replace(/\.$/, '')})` : '';
    S.recordings = { enabled: false, files: [], reason: `The recordings could not be listed${why}.`, failed: true };
  }
  S.force = true;
  requestFrame();
}

/**
 * the origins the recorded files load from: the page's host on each of the dashboard's media ports
 * where this same dashboard answers, named by `instance`, in the dashboard's order; the first carries
 * the sound. The ports are asked together; the list is ready once one answers and the others had
 * cameras.js MEDIA_PROBE_GRACE_MS more (awaitMediaAnswers; all of MEDIA_PROBE_MS while none has), and
 * a port that answers after that joins S.mediaOrigins then. Empty: the page's own origin, where the files share
 * fewer connections (MAX_MEDIA). Each answer is kept for the page while the recordings name the same
 * instance; one that came back empty is asked again with the recordings, at most once a minute
 */
async function mediaOrigins(ports, instance) {
  const wanted = [];
  for (const port of ports || []) {
    const origin = mediaOriginFor(window.location, port);
    if (origin && !wanted.some((w) => w.origin === origin)) wanted.push({ origin, port });
  }
  const answers = wanted.map(({ origin, port }) => {
    const last = S.mediaProbes.get(origin);
    if (!last || last.instance !== instance || (last.ok === false && performance.now() - last.at >= RECORDINGS_RETRY_MS)) {
      const probe = { instance, at: performance.now(), ok: null, answer: null };
      probe.answer = probeMediaOrigin(origin, port, instance).then((ok) => {
        probe.ok = ok;
        return ok;
      });
      S.mediaProbes.set(origin, probe);
    }
    return S.mediaProbes.get(origin);
  });
  const gen = ++S.mediaOriginsGen;
  const answered = () => wanted.filter((_, i) => answers[i].ok === true).map((w) => w.origin);
  await awaitMediaAnswers(answers.map((a) => a.answer));
  // a port that answers after that joins the list of this same call
  for (const a of answers) {
    if (a.ok !== null) continue;
    a.answer.then((ok) => {
      if (!ok || gen !== S.mediaOriginsGen) return;
      S.mediaOrigins = answered();
      S.force = true;
      requestFrame();
    });
  }
  return answered();
}

/** whether the replay clock advances now (displayNow interpolates it), so recorded video plays */
function clockRunning(now) {
  if (!(S.mode === 'replay' && S.playing && S.connState === 'open' && !S.backfilling && !S.preview)) return false;
  // the drawn clock holds 1.5 s after the newest batch, status or tick (a stalled stream), and at
  // the session's end
  if ((performance.now() - S.clockPerf) / 1000 >= 1.5) return false;
  return !(S.t1 != null && now != null && now >= S.t1);
}

// sound

const SOUND_FOLLOW = 'No sound in follow mode: the microphones stream AAC, which WebRTC does not carry. The replay of an ended session plays its recorded microphones.';

/** the Sound control of the session bar: Off or one of the session's microphones, and a mute toggle */
function buildSound() {
  const audio = h('audio', { class: 'live-audio', attrs: { preload: 'none' }, hidden: true });
  S.sound = new ReplaySound({ audio });
  const select = h('select', { class: 'select', attrs: { id: 'live-sound-select' } }, h('option', { attrs: { value: 'off' }, text: 'Off' }));
  select.addEventListener('change', () => {
    S.soundPick = select.value;
    // picking a microphone is a gesture too: it may play from now on
    if (select.value !== 'off') S.sound.arm();
    S.force = true;
    requestFrame();
  });
  const mute = h('button', { class: 'btn icon-btn', attrs: { type: 'button', 'aria-pressed': 'false', 'aria-label': 'Mute the sound', title: 'Mute the sound' } }, icon('volume', 14));
  mute.addEventListener('click', () => {
    S.soundMuted = !S.soundMuted;
    if (!S.soundMuted) S.sound.arm();
    S.force = true;
    requestFrame();
  });
  const note = h('span', { class: 'live-sound-note' });
  // a sound the browser held back plays from a click on this note, in one click
  const unblock = h('button', { class: 'btn sm live-sound-unblock', attrs: { type: 'button', title: 'The browser held the sound back until a click on the page; this click lets it play.' }, hidden: true }, icon('volume', 12), h('span', { text: 'Play the sound' }));
  unblock.addEventListener('click', () => {
    S.sound.arm();
    S.force = true;
    requestFrame();
  });
  const label = h('label', { class: 'live-sound-label', attrs: { for: 'live-sound-select' }, text: 'Sound' });
  const el = h('div', { class: 'live-sound', attrs: { role: 'group', 'aria-label': 'Sound' } }, label, select, mute, note, unblock, audio);
  UI.sound = { el, select, mute, note, unblock, audio, sig: '', optionsSig: '' };
}

let soundCache = { rec: null, sources: [] };

/** whether this view has sound: {ok, sources} or {ok: false, reason} */
function soundState() {
  if (S.mode === 'follow') return { ok: false, reason: SOUND_FOLLOW };
  if (S.live) return { ok: false, reason: 'Sound plays in the replay of an ended session, from its microphone recordings.' };
  const rec = S.recordings;
  if (!rec) return { ok: false, reason: 'Looking for the session\'s microphone recordings.' };
  if (!rec.enabled) return { ok: false, reason: rec.reason || 'The recordings are not available on this dashboard.' };
  if (soundCache.rec !== rec) soundCache = { rec, sources: soundSources(rec.files) };
  if (!soundCache.sources.length) return { ok: false, reason: 'The dashboard\'s machine holds no microphone recording of this session that has a start time.' };
  return { ok: true, sources: soundCache.sources };
}

/** the microphone that plays: the viewer's pick while it is listed, else the default; null for Off */
function soundSource(st) {
  if (!st.ok || S.soundPick === 'off') return null;
  return (S.soundPick && st.sources.find((s) => s.key === S.soundPick)) || defaultSource(st.sources);
}

/** the small note beside the control: what the sound does now */
function soundNote(res) {
  switch (res.state) {
    case 'fast': return { text: `Muted above ${MAX_SOUND_SPEED}x`, title: `Speech played faster than ${MAX_SOUND_SPEED}x cannot be understood, so the sound pauses; it plays at 1x to ${MAX_SOUND_SPEED}x.` };
    case 'gap': return { text: 'No recording at this moment', title: 'The microphone\'s file does not cover this moment of the session.' };
    case 'loading': return { text: 'Loading', title: '' };
    case 'error': return { text: 'Could not play', title: res.message || 'The recording could not be played.' };
    // the note becomes a button (UI.sound.unblock)
    case 'blocked': return { text: '', title: '' };
    default: return { text: '', title: '' };
  }
}

/** keep the replay's sound on the clock and the control up to date (every redraw) */
function renderSound(now) {
  if (endedReplay()) loadRecordings(false);
  const st = soundState();
  const source = soundSource(st);
  const res = S.sound.update({ source, now, speed: S.speed, running: clockRunning(now), muted: S.soundMuted, origin: S.mediaOrigins[0] || null });
  const U = UI.sound;
  const optionsSig = st.ok ? st.sources.map((s) => `${s.key}=${s.label}`).join('\n') : '';
  if (optionsSig !== U.optionsSig) {
    U.optionsSig = optionsSig;
    clear(U.select).appendChild(h('option', { attrs: { value: 'off' }, text: 'Off' }));
    for (const s of st.ok ? st.sources : []) U.select.appendChild(h('option', { attrs: { value: s.key }, text: s.label }));
  }
  const value = source ? source.key : 'off';
  if (U.select.value !== value) U.select.value = value;
  const note = soundNote(res);
  const sig = [st.ok, st.reason, value, S.soundMuted, res.state, note.text, note.title].join('|');
  if (sig === U.sig) return;
  U.sig = sig;
  U.el.dataset.state = res.state;
  U.select.disabled = !st.ok;
  // a disabled control shows no tooltip of its own: the group says why
  U.el.title = st.ok ? '' : st.reason;
  U.select.title = st.ok ? 'The microphone the replay plays, at the replay clock' : st.reason;
  U.mute.disabled = !source;
  U.mute.setAttribute('aria-pressed', S.soundMuted ? 'true' : 'false');
  const label = S.soundMuted ? 'Unmute the sound' : 'Mute the sound';
  U.mute.setAttribute('aria-label', label);
  U.mute.title = label;
  clear(U.mute).appendChild(icon(S.soundMuted ? 'volume-off' : 'volume', 14));
  U.note.textContent = note.text;
  U.note.title = note.title;
  U.unblock.hidden = res.state !== 'blocked';
}

// colours

/**
 * Tag colours: the analysis's roster first (its sorted pupils, from the timeline part), then any tag the
 * stream shows later in the next free slot (never repainting one). Until the colour sources answer, a tag
 * without a slot is drawn neutral and gets no slot, so nothing is fixed before the answer.
 */
function tagIdentity() {
  const real = identity([], { grow: true });
  return {
    tag(id) {
      const tid = String(id);
      if (S.coloursPending && isPupilTag(tid) && !real.tags().includes(tid)) {
        S.tagsProvisional = true;
        return { key: `tag:${tid}`, id: tid, slot: null, color: 'var(--tag-other)', label: `Tag ${tid}`, provisional: true };
      }
      return real.tag(tid);
    },
    addTags: (ids) => (S.coloursPending ? real.tags() : real.addTags(ids)),
    tags: () => real.tags(),
    // the analysis's roster: slots in sorted order for the tags without one
    seed: (ids) => real.addTags(ids),
  };
}

function seedRoster(ids) {
  const tags = (ids || []).map(String).filter(isPupilTag);
  if (S.coloursPending) for (const t of tags) S.colourRoster.add(t);
  else S.ident.seed(tags);
}

/**
 * ask for the parts that fix the colours on the analysis page: the speech part's entities give the
 * voices their slots (whole-session speaking rank), the timeline part's roster the tags theirs. The
 * request only reads the cache (submit=0): watching never starts a report job, and a part not computed
 * yet (404) leaves the stream's order in place. Later calls (a live session's recheck) ask for the
 * speech part only and fill voices that have no colour yet.
 */
async function loadColourSources(first = false) {
  if (S.coloursLoading) return;
  S.coloursLoading = true;
  const ask = async (part) => {
    const res = await api(`/api/sessions/${encodeURIComponent(S.sid)}/report/${part}?submit=0`);
    const env = res.ok ? res.data : null;
    const data = env && env.status === 'ready' && env.data && typeof env.data === 'object' ? env.data : null;
    if (!data) return;
    if (part === 'speech') {
      S.speechReady = true;
      S.voices.seed(data);
      seedRoster((data.entities || []).filter((e) => e && e.kind === 'wearer' && String(e.key).startsWith('tag:')).map((e) => String(e.key).slice(4)));
    } else if (data.roster && Array.isArray(data.roster.kept)) seedRoster(data.roster.kept);
  };
  await Promise.all((first ? ['speech', 'timeline'] : ['speech']).map(ask));
  S.coloursLoading = false;
  settleColours();
}

/** stop waiting for the colour sources; what was drawn neutral meanwhile is drawn again. */
function settleColours() {
  if (S.coloursPending) {
    S.coloursPending = false;
    S.voices.pending = false;
    if (S.colourRoster.size) S.ident.seed(Array.from(S.colourRoster));
  }
  if (S.voices.provisional || S.tagsProvisional) {
    S.voices.provisional = false;
    S.tagsProvisional = false;
    // the transcript keeps its chunk nodes: build them again with the colours
    if (UI.tr) UI.tr.epoch = -1;
  }
  S.heavyKey = '';
  S.laneKey = '';
  for (const k of ['share', 'looks', 'pairs']) if (UI[k]) UI[k].sig = '';
  if (UI.room) UI.room.legendSig = '';
  roomSig = '';
  roomKey = '';
  healthSig = '';
  S.force = true;
  requestFrame();
}

// rendering

function requestFrame() {
  if (!S.raf && typeof requestAnimationFrame === 'function') S.raf = requestAnimationFrame(frame);
}

function heavyEvery() {
  if (S.mode === 'replay' && S.playing) return Math.max(FRAME_MS, HEAVY_MS / Math.min(S.speed, 4));
  return HEAVY_MS;
}

function frame() {
  S.raf = 0;
  if (!S.model || document.visibilityState === 'hidden') return;
  const t = performance.now();
  if (!S.force && t - S.lastFrame < FRAME_MS - 30) return;
  S.lastFrame = t;
  const now = displayNow();
  renderBar(now);
  renderHealth(now);
  if (now != null) {
    const key = `${S.model.version}|${S.window}|${S.roster.join(',')}|${S.trails}|${S.distances}|${S.voices.version}|${S.voices.pending}`;
    if (S.force || (t - S.heavyAt >= heavyEvery() && (key !== S.heavyKey || Math.abs(now - (S.heavyNow ?? now)) >= 1))) {
      S.heavyAt = t;
      S.heavyKey = key;
      S.heavyNow = now;
      heavy(now);
    }
    renderTimelineView(now);
    renderRoom(now);
    renderCameras(now);
  }
  renderSound(now);
  S.force = false;
}

function clockFormat() {
  const span = S.t0 != null && S.t1 != null ? S.t1 - S.t0 : 0;
  return fmt.clockSpan(span);
}

let barSig = '';
function renderBar(now) {
  const replay = S.mode === 'replay';
  const ended = atEnd();
  const shownNow = S.scrubbing && S.scrubValue != null && S.t0 != null ? S.t0 + S.scrubValue : now;
  const cf = clockFormat();
  UI.clock.textContent = shownNow != null && S.t0 != null ? cf(Math.max(0, shownNow - S.t0)) : fmt.na;
  const dur = S.t0 != null && S.t1 != null ? S.t1 - S.t0 : null;
  UI.clockSub.textContent = shownNow != null ? `Wall clock ${fmt.time(shownNow)} ${fmt.zone(shownNow)}` : '';
  if (dur != null) {
    const max = Math.max(1, Math.ceil(dur));
    if (Number(UI.scrub.max) !== max) UI.scrub.max = String(max);
    if (!S.scrubbing && now != null) UI.scrub.value = String(Math.round(Math.max(0, Math.min(max, now - S.t0))));
    UI.scrubEnd.textContent = cf(dur);
    UI.scrub.setAttribute('aria-valuetext', `${cf(Math.max(0, (shownNow ?? S.t0) - S.t0))} of ${cf(dur)}`);
  }
  UI.scrub.disabled = S.t0 == null;

  let pill;
  if (S.connState === 'reconnecting') pill = ['reconnecting', 'Reconnecting'];
  else if (S.mode === 'follow') pill = S.live ? ['live', 'Live'] : ['ended', 'Ended'];
  else if (S.playing) pill = ['replay', `Replay ${S.speed}x`];
  else if (ended) pill = ['ended', 'Ended'];
  else pill = ['paused', 'Paused'];
  const sig = `${pill.join()}|${replay}|${S.playing}|${ended}|${S.live}|${S.connState}|${S.preview}|${S.backfilling}`;
  if (sig !== barSig) {
    barSig = sig;
    UI.pill.update({ state: pill[0], text: pill[1] });
    UI.followBtn.hidden = !(S.live && S.mode === 'follow');
    UI.playBtn.hidden = !replay;
    clear(UI.playBtn).appendChild(icon(S.playing ? 'pause' : 'play', 14));
    UI.playBtn.setAttribute('aria-label', S.playing ? 'Pause' : ended ? 'Replay from start' : 'Play');
    UI.playBtn.title = S.playing ? 'Pause' : ended ? 'Replay from start' : 'Play';
    UI.speed.hidden = !replay;
    UI.jumpEnd.hidden = !(replay && !S.live) || ended;
    UI.backLive.hidden = !(replay && S.live);
    UI.fromStart.hidden = !(replay && ended);
    UI.stateNote.textContent = '';
    if (S.connState === 'connecting' || (S.backfilling && S.connState === 'open')) {
      // a replay that resumes backfills only a few seconds: no five minutes to announce
      UI.stateNote.textContent = S.connState === 'connecting' ? 'Connecting'
        : S.mode === 'follow' ? 'Loading the last 5 minutes'
          : S.backfillSecs >= BACKFILL ? 'Loading the 5 minutes before this moment' : 'Loading';
    }
  }
  if (S.mode === 'follow' && S.live && S.clock != null) UI.lag.textContent = `${fmt.num(Math.max(0, serverNow() - S.clock), 0)} s behind`;
  else UI.lag.textContent = '';
  renderBanner();
}

let bannerSig = '';
function renderBanner() {
  let kind = null;
  let text = '';
  let action = null;
  if (S.connState === 'reconnecting') {
    kind = 'warning';
    const wait = Math.max(0, Math.ceil((S.retryAt - performance.now()) / 1000));
    text = `${S.lastError || 'The live stream was interrupted.'} Retrying in ${wait} s.`;
    action = 'retry';
  } else if (S.mode === 'follow' && !S.live && S.connState === 'open') {
    kind = 'info';
    const last = S.lastEvent ?? S.t1;
    text = `This session has ended${last != null ? `: its last data arrived ${fmt.dateTime(last)}` : ''}. Follow shows its last 5 minutes.`;
    action = 'replay';
  }
  const sig = `${kind}|${text}|${action}`;
  if (sig === bannerSig) return;
  bannerSig = sig;
  clear(UI.banner);
  UI.banner.hidden = !kind;
  if (!kind) return;
  UI.banner.dataset.kind = kind;
  UI.banner.appendChild(icon(kind === 'info' ? 'info' : 'alert', 16));
  UI.banner.appendChild(h('span', { class: 'live-banner-text', text }));
  if (action === 'retry') {
    UI.banner.appendChild(h('button', { class: 'btn sm', attrs: { type: 'button' }, on: { click: () => reconnectNow() } }, h('span', { text: 'Retry now' })));
  } else if (action === 'replay') {
    UI.banner.appendChild(h('button', { class: 'btn sm primary', attrs: { type: 'button', title: 'Open the replay paused at the session\'s start' }, on: { click: () => startPreview(openingMoment()) } }, icon('play', 12), h('span', { text: 'Replay this session' })));
  }
}

let healthSig = '';
function renderHealth(now) {
  if (now == null || !S.gotData) {
    const sig = 'none';
    if (sig !== healthSig) {
      healthSig = sig;
      clear(UI.health).appendChild(h('span', { class: 'muted', text: S.connState === 'reconnecting' ? 'No data while the stream is away.' : 'Waiting for the first records.' }));
    }
    return;
  }
  const hl = S.model.health(now);
  const ref = S.mode === 'follow' ? serverNow() : now;
  const items = [];
  const metaMods = (S.meta && S.meta.modalities) || {};
  for (const [label, key, mod] of [['ASR', 'asr', 'asr'], ['IPS', 'ips', 'ips'], ['VFA', 'vfa', 'vfa']]) {
    const t = hl[key];
    if (t == null) {
      items.push({ label, text: metaMods[mod] ? 'no data in view' : 'not in this session', status: 'none' });
      continue;
    }
    const age = Math.max(0, ref - t);
    const stale = age > STALE_AGE;
    items.push({ label, text: age < 1.5 ? 'now' : fmt.ago(age), status: stale ? 'warning' : 'good' });
  }
  const cams = hl.cameras.total ? `${hl.cameras.seen} of ${hl.cameras.total}` : null;
  const trLag = hl.tr != null ? Math.max(0, now - hl.tr) : null;
  const tags = hl.tags.filter((t) => S.roster.includes(t));
  const sig = JSON.stringify([items, cams, trLag == null ? null : Math.round(trLag), tags]);
  if (sig === healthSig) return;
  healthSig = sig;
  clear(UI.health);
  items.forEach((it) => {
    const mark = it.status === 'good' ? h('span', { class: 'dot', style: { '--swatch': 'var(--good)' } })
      : it.status === 'warning' ? icon('alert', 12) : h('span', { class: 'dot', style: { '--swatch': 'var(--axis)' } });
    UI.health.appendChild(h('span', { class: 'health-item', dataset: { status: it.status }, title: it.status === 'warning' ? `Newest ${it.label} record is more than ${STALE_AGE} s old` : null }, mark, h('b', { text: it.label }), h('span', { text: it.text })));
  });
  UI.health.appendChild(h('span', { class: 'health-sep', attrs: { 'aria-hidden': 'true' } }));
  if (cams) UI.health.appendChild(h('span', { class: 'health-item' }, icon('camera', 12), h('span', { text: `Cameras ${cams}` })));
  if (trLag != null) UI.health.appendChild(h('span', { class: 'health-item', title: 'Stream clock minus the end of the newest transcript chunk' }, h('span', { text: `Transcript lag ${fmt.secs(trLag, 0)}` })));
  const tagWrap = h('span', { class: 'health-tags' });
  if (tags.length) for (const t of tags) tagWrap.appendChild(chip(S.ident.tag(t).color, `Tag ${t}`));
  else tagWrap.appendChild(h('span', { class: 'muted', text: 'none' }));
  UI.health.appendChild(h('span', { class: 'health-item' }, h('span', { text: 'Seen in the last 10 s' }), tagWrap));
}

function heavy(now) {
  const m = S.model;
  const roster = m.roster();
  S.ident.addTags(roster);
  S.roster = sortTags(roster);
  for (const k of m.voiceOrder) S.voices.get(k);
  const empty = m.newestTime() == null;
  const loading = !S.gotData && (S.connState === 'connecting' || S.backfilling || S.preview);
  updateExtent();
  renderKpis(now, empty, loading);
  renderLanes(now, empty, loading);
  renderLooks(now, loading);
  renderShare(now, loading);
  renderPairs(now, loading);
  renderTranscript(now, loading);
}

function windowLabel() {
  return (WINDOWS.find((w) => w.value === S.window) || WINDOWS[1]).label;
}

function signedPp(v) {
  if (!finite(v)) return null;
  const n = Math.round(v * 1000) / 10;
  return `${n > 0 ? '+' : ''}${fmt.num(n, 1)}`;
}

function renderKpis(now, empty, loading) {
  UI.kpiCaption.textContent = `Last ${windowLabel()}. Trend lines: one value per minute over the last 10 min.`;
  if (empty) {
    const pending = loading ? 'Loading' : S.connState === 'reconnecting' ? 'Reconnecting' : 'No data yet';
    for (const tile of Object.values(UI.kpi)) tile.update({ pending, note: '', spark: null });
    return;
  }
  const k = S.model.kpis(now, S.window, S.roster);
  const pct0 = (v) => (finite(v) ? fmt.num(v * 100, 0) : null);
  const speakers = k.speakers;
  UI.kpi.speech.update({ pending: null, value: pct0(k.speech), unit: '%', note: 'voiced share of 3 s buckets', spark: { values: k.spark.speech, max: 1, min: 0 } });
  UI.kpi.switches.update({ pending: null, value: finite(k.switches) ? fmt.num(k.switches, 1) : null, unit: '/min', note: 'between speakers', spark: { values: k.spark.switches, min: 0 } });
  const who = S.model.entityMode() === 'voices' ? 'voices' : 'speakers';
  const pupils = S.roster.length >= 2;
  UI.kpi.balance.update({ pending: null, value: finite(k.balance) ? fmt.num(k.balance, 2) : null, note: finite(k.balance) ? `${speakers} ${who}, 1 = even` : `needs 2 ${who}`, spark: { values: k.spark.balance, max: 1, min: 0 } });
  UI.kpi.distance.update({ pending: null, value: finite(k.distance) ? fmt.num(k.distance, 2) : null, unit: 'm', note: S.roster.length < 2 ? 'needs 2 badges' : 'all pairs pooled', spark: { values: k.spark.distance, min: 0 } });
  const jaNote = !pupils ? 'needs 2 pupils'
    : k.jaPairs ? `${k.jaPairs} ${k.jaPairs === 1 ? 'pair' : 'pairs'}, vs 20 to 40 s earlier`
      : k.jaFew ? 'too few frames' : 'no pair data';
  UI.kpi.ja.update({ pending: null, value: signedPp(k.ja), unit: 'pp', note: jaNote, spark: pupils ? { values: k.spark.ja } : null });
  UI.kpi.social.update({ pending: null, value: pupils ? pct0(k.social) : null, unit: '%', note: pupils ? 'of frames, on a partner\'s face or hands' : 'needs 2 pupils', spark: pupils ? { values: k.spark.social, max: 1, min: 0 } : null });
}

function voiceKeysSlotted() {
  // voices with a colour slot, in slot order; the rest fold into Other voices
  return S.voices.slotted(S.model.voiceOrder);
}

function pairLabel(a, b) {
  return `Tag ${a} and Tag ${b}`;
}

/** the two tag dots that name a pair (a pair has no colour of its own). */
function pairDots(a, b) {
  const ca = S.ident.tag(a).color;
  const cb = S.ident.tag(b).color;
  return h('span', { class: 'pair-dots', attrs: { 'aria-hidden': 'true' }, dataset: { sig: `${ca}|${cb}` } },
    h('span', { class: 'swatch dot', style: { '--swatch': ca } }),
    h('span', { class: 'swatch dot', style: { '--swatch': cb } }));
}

/** the timeline draws one swatch per lane label: put the two tag dots in front of each pair lane's label. */
function decoratePairLabels() {
  const labels = UI.act && UI.act.labelsEl;
  const named = UI.act && UI.act.pairLabels;
  if (!labels || !named) return;
  // a narrow label column (phones) keeps the dots and the two ids; the title keeps the full name
  const compact = (parseFloat(labels.style.width) || labels.offsetWidth) < 124;
  for (const el of labels.children) {
    const pair = named.get(el.title);
    if (!pair) continue;
    const text = compact ? `${pair[0]} and ${pair[1]}` : el.title;
    if (el.lastChild && el.lastChild.textContent !== text) el.lastChild.textContent = text;
    const dots = pairDots(pair[0], pair[1]);
    const old = el.querySelector('.pair-dots');
    if (old && old.dataset.sig === dots.dataset.sig) continue;
    if (old) old.replaceWith(dots);
    else el.insertBefore(dots, el.lastChild);
  }
}

function renderLanes(now, empty, loading) {
  const c = UI.act.card;
  if (empty) {
    if (loading) c.setState('loading', 'Loading the stream');
    else c.setState('empty', S.connState === 'reconnecting' ? 'Reconnecting to the live stream.' : 'No data in this window yet.');
    return;
  }
  const key = `${S.model.version}|${S.window}|${Math.floor(now)}|${S.roster.join(',')}|${S.voices.version}|${S.voices.pending}`;
  if (key === S.laneKey && UI.act.tl) return;
  S.laneKey = key;
  const m = S.model;
  const W = S.window;
  const a = now - W - 3;
  const b = now + 1;
  const rel = (t) => t - S.t0;
  const lanes = [];
  const act = m.speechActivity(a, b);
  lanes.push({ kind: 'area', key: 'speech', label: 'Speech', swatch: 'var(--ink-2)', color: 'var(--ink-2)', t: act.t.map(rel), v: act.v, step: 3, max: 1, format: (v) => (v == null ? 'unreadable' : fmt.pct(v)) });

  const mode = m.entityMode();
  const turns = m.turns(a, b);
  const speakerLanes = [];
  const segFmt = (seg) => (seg ? 'speaking' : 'silent');
  if (mode === 'voices') {
    const slotted = voiceKeysSlotted();
    const set = new Set(slotted);
    const by = new Map(slotted.map((k) => [k, []]));
    const other = [];
    for (const [s0, e0, k] of turns) {
      if (set.has(k)) by.get(k).push([rel(s0), rel(e0), k]);
      else other.push([rel(s0), rel(e0), k]);
    }
    for (const k of slotted) {
      const v = S.voices.get(k);
      speakerLanes.push({ kind: 'segments', key: k, label: v.label, swatch: v.color, color: v.color, segs: by.get(k), format: segFmt });
    }
    if (other.length || m.voiceOrder.length > slotted.length) {
      speakerLanes.push({ kind: 'segments', key: 'other', label: 'Other voices', swatch: 'var(--voice-other)', color: 'var(--voice-other)', segs: other, format: (seg) => (seg ? S.voices.get(seg[2]).label : 'silent') });
    }
  } else {
    const keys = mode === 'wearers' ? S.roster.map((t) => `tag:${t}`) : Array.from(new Set(turns.map((x) => x[2])));
    for (const k of keys) {
      const ent = mode === 'wearers' ? S.ident.tag(k.slice(4)) : S.voices.get(k);
      const label = mode === 'wearers' ? `${ent.label} (worn mic)` : ent.label;
      speakerLanes.push({ kind: 'segments', key: k, label, swatch: ent.color, color: ent.color, segs: turns.filter((x) => x[2] === k).map((x) => [rel(x[0]), rel(x[1]), k]), format: segFmt });
    }
  }
  if (speakerLanes.length) lanes.push({ kind: 'header', label: 'Speakers' }, ...speakerLanes);

  const tags = S.roster;
  if (tags.length) {
    const pr = m.presence(a, b, tags);
    lanes.push({ kind: 'header', label: 'Presence' });
    for (const tag of tags) {
      const id = S.ident.tag(tag);
      lanes.push({
        kind: 'cells', key: `p${tag}`, label: id.label, swatch: id.color, t: pr.t.map(rel), v: pr.byTag[tag], step: 1,
        fillOf: (v) => (v === 1 ? id.color : v === 0 ? null : 'hatch'),
        format: (v) => (v === 1 ? 'present' : v === 0 ? 'absent' : 'unknown'),
      });
    }
    const gz = m.gazeSeries(a, b, tags);
    lanes.push({ kind: 'header', label: 'Gaze' });
    for (const tag of tags) {
      const id = S.ident.tag(tag);
      lanes.push({
        kind: 'cells', key: `g${tag}`, label: id.label, swatch: id.color, t: gz.t.map(rel), v: gz.byTag[tag], step: 1,
        fillOf: (v) => (v == null ? null : GAZE_FILL[CATEGORIES[v]]),
        format: (v) => (v == null ? 'not seen' : CATEGORY_LABELS[CATEGORIES[v]]),
      });
    }
  }
  const pairs = pairList(tags);
  const pairLabels = new Map();
  if (pairs.length) {
    const pd = m.pairDistances(a, b, tags);
    // one shared scale; it follows the bulk of the distances, so one misread window does not flatten the lines
    const all = [];
    for (const { d } of pd.values()) all.push(...d);
    all.sort((x, y) => x - y);
    const p98 = all.length ? all[Math.min(all.length - 1, Math.floor(all.length * 0.98))] : 1;
    const top = Math.max(1.5, Math.min(4, Math.ceil(p98 * 2) / 2));
    // the 0.5 m line keeps its label only where it sits clear of the 1 m label
    const refGap = (0.5 / top) * (PAIR_LANE_HEIGHT - 6);
    const refs = [{ v: 0.5, label: refGap >= 11 ? '0.5 m' : '' }, { v: 1, label: '1 m' }];
    lanes.push({ kind: 'header', label: 'Distance' });
    // one lane per pair (small multiples): pairs carry no colour of their own
    for (const [x, y, k] of pairs) {
      const label = pairLabel(x, y);
      pairLabels.set(label, [x, y]);
      lanes.push({
        kind: 'lines', key: `dist-${k}`, label, height: PAIR_LANE_HEIGHT, domain: [0, top],
        series: [{ key: k, label: 'distance', color: PAIR_INK, t: pd.get(k).t.map(rel), v: pd.get(k).d }],
        refs, format: (v) => fmt.metres(v),
      });
    }
  }
  UI.act.lanes = lanes;
  UI.act.pairLabels = pairLabels;
  UI.act.legendSig = UI.act.legendSig || '';
  const legendItems = CATEGORIES.map((cat) => ({ label: CATEGORY_LABELS[cat], color: GAZE_FILL[cat], kind: GAZE_FILL[cat] === 'hatch' ? 'hatch' : 'rect' }));
  const lsig = JSON.stringify(legendItems);
  if (lsig !== UI.act.legendSig) {
    UI.act.legendSig = lsig;
    c.setLegend(legendItems);
  }
  c.setState('ready');
  const opts = timelineOpts(now);
  if (!UI.act.tl) {
    UI.act.tl = timeline(UI.act.el, { ...opts, lanes, label: 'Activity over the trailing window' });
    // the timeline redraws its labels when the lanes change: put the pair dots back each time
    UI.act.labelsEl = UI.act.el.querySelector('.tl-labels');
    if (UI.act.labelsEl && typeof MutationObserver === 'function') {
      UI.act.labelObserver = new MutationObserver(decoratePairLabels);
      UI.act.labelObserver.observe(UI.act.labelsEl, { childList: true });
    }
  } else UI.act.tl.update({ ...opts, lanes });
  decoratePairLabels();
}

function timelineOpts(now) {
  const end = now - S.t0;
  const dur = S.t1 != null ? S.t1 - S.t0 : end;
  return { span: [0, Math.max(dur, end, 1)], view: [end - S.window, end], now: end, axisFormat: clockFormat() };
}

function renderTimelineView(now) {
  if (!UI.act.tl || !UI.act.lanes) return;
  const opts = timelineOpts(now);
  const key = `${opts.view[0].toFixed(2)}|${opts.view[1].toFixed(2)}`;
  if (key === UI.act.viewKey) return;
  UI.act.viewKey = key;
  UI.act.tl.update(opts);
}

function updateExtent() {
  const m = S.model;
  const pts = [];
  for (const r of m.ips.items) for (const tag of S.roster) if (r.p[tag]) pts.push(r.p[tag]);
  if (!pts.length && !S.cameras.length) return;
  const want = roomExtent(pts, S.cameras);
  if (!want) return;
  if (S.extent && extentContains(S.extent, want)) return;
  S.extent = extentUnion(S.extent, want);
  UI.room.map.setExtent(S.extent);
}

let roomSig = '';
let roomKey = '';
function renderRoom(now) {
  const m = S.model;
  const c = UI.room.card;
  const key = `${m.version}|${now.toFixed(2)}|${S.trails}|${S.distances}|${S.roster.join(',')}|${S.connState}|${S.gotData}`;
  if (key === roomKey && !S.force) return;
  roomKey = key;
  const loading = !S.gotData && (S.connState === 'connecting' || S.backfilling || S.preview);
  const tags = S.roster;
  const pos = m.latestPositions(now, tags, 30);
  const sigState = pos.size ? 'ready' : loading ? 'loading' : 'empty';
  if (sigState !== roomSig) {
    roomSig = sigState;
    if (sigState === 'ready') c.setState('ready');
    else if (sigState === 'loading') c.setState('loading', 'Loading badge positions');
    else {
      const noIps = S.meta && S.meta.modalities && !S.meta.modalities.ips;
      c.setState('empty', noIps ? 'This session has no IPS data.' : 'No badge positions in this window.');
    }
    c.setLegend(tags.length ? tags.map((t) => ({ label: `Tag ${t}`, color: S.ident.tag(t).color, kind: 'dot' })) : null);
  }
  if (!pos.size) return;
  const badges = [];
  for (const [tag, p] of pos) {
    const id = S.ident.tag(tag);
    badges.push({ id: tag, color: id.color, label: id.label, short: tag, u: p.u, v: p.v, heading: p.heading, age: p.age });
  }
  const trails = S.trails ? m.trails(now, Array.from(pos.keys()), 30) : {};
  const edges = m.facingEdges(now, tags);
  const pairs = [];
  if (S.distances) {
    for (const [k, v] of m.distanceNow(now, tags)) {
      if (!v) continue;
      const [a, b] = k.split('|');
      pairs.push({ a, b, d: v.d });
    }
  }
  const legendSig = tags.join(',');
  if (legendSig !== UI.room.legendSig) {
    UI.room.legendSig = legendSig;
    c.setLegend(tags.map((t) => ({ label: `Tag ${t}`, color: S.ident.tag(t).color, kind: 'dot' })));
  }
  UI.room.map.setLive({ badges, trails, edges, pairs });
}

function looksTable() {
  const d = UI.looks.data;
  if (!d) return { columns: [], rows: [] };
  return {
    columns: [{ key: 'from', label: 'Looker' }, { key: 'to', label: 'Target' }, { key: 'share', label: 'Share of frames', align: 'right', format: (v) => fmt.pct(v) }, { key: 'n', label: 'Frame sets', align: 'right' }],
    rows: d.edges.map((e) => ({ from: { text: `Tag ${e.from}`, color: S.ident.tag(e.from).color }, to: e.to === 'other' ? 'Others' : { text: `Tag ${e.to}`, color: S.ident.tag(e.to).color }, share: e.weight, n: e.n })),
  };
}

function renderLooks(now, loading) {
  const c = UI.looks.card;
  const tags = S.roster;
  if (!tags.length) {
    if (loading) c.setState('loading');
    else c.setState('empty', 'No pupils seen yet.');
    return;
  }
  const lk = S.model.looks(now - LOOKS_SECONDS, now + 1e-6, tags);
  const pos = S.model.latestPositions(now, tags, 30);
  const edges = [];
  let anySeen = false;
  let other = false;
  for (const [l, o] of lk) {
    if (o.seen) anySeen = true;
    for (const [tg, n] of o.any) {
      if (!o.seen) continue;
      if (tg === 'other') other = true;
      edges.push({ from: l, to: tg, weight: n / o.seen, n, label: fmt.pct(n / o.seen) });
    }
  }
  if (!anySeen) {
    c.setState('empty', `No pupil was seen on camera in the last ${LOOKS_SECONDS} s.`);
    UI.looks.data = null;
    return;
  }
  const nodes = tags.map((t) => {
    const p = pos.get(t);
    return { key: t, label: `Tag ${t}`, color: S.ident.tag(t).color, x: p ? p.u : null, y: p ? p.v : null };
  });
  if (other) nodes.push({ key: 'other', label: 'Others', color: 'var(--tag-other)' });
  const given = nodes.filter((n) => finite(n.x)).length >= 2;
  const data = { nodes, edges: edges.map((e) => ({ ...e, mutual: false })), layout: given ? 'given' : 'circle', height: 260, format: (v) => fmt.pct(v), weightLabel: 'of the looker\'s frames', outLabel: 'Looks out', inLabel: 'Looked at', label: 'Who looks at whom' };
  UI.looks.data = data;
  const sig = JSON.stringify([nodes.map((n) => [n.key, n.x == null ? null : Math.round(n.x * 10), n.y == null ? null : Math.round(n.y * 10)]), edges.map((e) => [e.from, e.to, Math.round(e.weight * 100)])]);
  c.setState('ready');
  if (sig === UI.looks.sig) return;
  UI.looks.sig = sig;
  if (!UI.looks.chart) UI.looks.chart = networkGraph(UI.looks.el, data);
  else UI.looks.chart.update(data);
  if (!edges.length) c.setNote(`No looks at a partner's face or hands in the last ${LOOKS_SECONDS} s.`);
  else c.setNote('');
}

function shareTable() {
  const d = UI.share.data;
  if (!d) return { columns: [], rows: [] };
  return {
    columns: [{ key: 'who', label: 'Speaker' }, { key: 'secs', label: 'Seconds', align: 'right', format: (v) => fmt.num(v, 1) }, { key: 'share', label: 'Share', align: 'right', format: (v) => fmt.pct(v) }, { key: 'sofar', label: 'Since view start', align: 'right', format: (v) => fmt.pct(v) }],
    rows: d.map((it) => ({ who: { text: it.label, color: it.color }, secs: it.seconds, share: it.value, sofar: it.sub })),
  };
}

function renderShare(now, loading) {
  const c = UI.share.card;
  const m = S.model;
  const mode = m.entityMode();
  const secs = m.speakingSeconds(now - S.window, now + 1e-6);
  const prefix = mode === 'voices' ? 'v:' : mode === 'wearers' ? 'tag:' : 'spk:';
  const totals = new Map(Array.from(m.totals).filter(([k]) => k.startsWith(prefix)));
  let total = 0;
  for (const v of secs.values()) total += v;
  let totalSoFar = 0;
  for (const v of totals.values()) totalSoFar += v;
  if (!total && !totalSoFar) {
    if (loading) c.setState('loading');
    else c.setState('empty', `No speech in the last ${windowLabel()}.`);
    UI.share.data = null;
    return;
  }
  const items = [];
  if (mode === 'voices') {
    const slotted = voiceKeysSlotted();
    const set = new Set(slotted);
    for (const k of slotted) {
      const v = S.voices.get(k);
      const s0 = secs.get(k) || 0;
      const so = totals.get(k) || 0;
      if (!s0 && !so) continue;
      items.push({ key: k, label: v.label, color: v.color, seconds: s0, value: total ? s0 / total : 0, sub: totalSoFar ? so / totalSoFar : null });
    }
    let os = 0;
    let oso = 0;
    for (const [k, v] of secs) if (!set.has(k)) os += v;
    for (const [k, v] of totals) if (!set.has(k)) oso += v;
    if (os || oso) items.push({ key: 'other', label: 'Other voices', color: 'var(--voice-other)', seconds: os, value: total ? os / total : 0, sub: totalSoFar ? oso / totalSoFar : null });
  } else {
    const keys = mode === 'wearers' ? S.roster.map((t) => `tag:${t}`) : Array.from(new Set([...secs.keys(), ...totals.keys()]));
    for (const k of keys) {
      const ent = mode === 'wearers' ? S.ident.tag(k.slice(4)) : S.voices.get(k);
      const s0 = secs.get(k) || 0;
      const so = totals.get(k) || 0;
      items.push({ key: k, label: mode === 'wearers' ? `${ent.label} (worn mic)` : ent.label, color: ent.color, seconds: s0, value: total ? s0 / total : 0, sub: totalSoFar ? so / totalSoFar : null });
    }
  }
  UI.share.data = items;
  c.setState('ready');
  const opts = {
    items: items.map((it) => ({ key: it.key, label: it.label, value: it.value, color: it.color, note: fmt.duration(it.seconds), sub: it.sub })),
    format: (v) => fmt.pct(v), max: 1, subLabel: 'Since this view started', valueLabel: `Last ${windowLabel()}`, label: 'Speaking share',
  };
  const sig = JSON.stringify(opts.items.map((x) => [x.key, x.color, Math.round(x.value * 1000), Math.round((x.sub ?? -1) * 1000), x.note]));
  if (sig === UI.share.sig) return;
  UI.share.sig = sig;
  if (!UI.share.chart) UI.share.chart = barList(UI.share.el, opts);
  else UI.share.chart.update(opts);
}

function renderPairs(now, loading) {
  const c = UI.pairs.card;
  const tags = S.roster;
  const pairs = pairList(tags);
  if (pairs.length === 0) {
    if (loading) c.setState('loading');
    else c.setState('empty', tags.length ? 'Pairs need at least two badges.' : 'No pupils seen yet.');
    UI.pairs.sig = '';
    return;
  }
  const m = S.model;
  const dn = m.distanceNow(now, tags);
  const pd = m.pairDistances(now - S.window, now + 1e-6, tags);
  const ja = m.jointAttention(now - LOOKS_SECONDS, now + 1e-6, tags);
  const rows = pairs.map(([x, y, k]) => {
    const ds = pd.get(k).d;
    return {
      k, x, y,
      now: dn.get(k) ? dn.get(k).d : null,
      within: ds.length ? ds.filter((d) => d < 1).length / ds.length : null,
      ja: ja.get(k),
    };
  });
  const sig = JSON.stringify(rows.map((r) => [r.k, r.now == null ? null : Math.round(r.now * 100), r.within == null ? null : Math.round(r.within * 100), r.ja.ratio == null ? null : Math.round(r.ja.ratio * 1000), r.ja.baseline == null ? null : Math.round(r.ja.baseline * 1000)]));
  c.setState('ready', 'Baseline: the pair\'s own rate 20 to 40 s earlier; n/a under 10 comparisons.');
  if (sig === UI.pairs.sig) return;
  UI.pairs.sig = sig;
  const table = h('table', { class: 'data' },
    h('caption', { class: 'sr-only', text: 'Pairs: distance now and share of the window within 1 m' }),
    h('thead', {}, h('tr', {}, h('th', { attrs: { scope: 'col' }, text: 'Pair' }), h('th', { attrs: { scope: 'col', 'data-align': 'right' }, text: 'Now' }), h('th', { attrs: { scope: 'col', 'data-align': 'right' }, text: `Within 1 m, ${windowLabel()}` }))),
    h('tbody', {}, rows.map((r) => h('tr', {},
      h('td', {}, pairDots(r.x, r.y), document.createTextNode(pairLabel(r.x, r.y))),
      h('td', { attrs: { 'data-align': 'right' }, text: fmt.metres(r.now) }),
      h('td', { attrs: { 'data-align': 'right' }, text: fmt.pct(r.within) })))));
  clear(UI.pairs.tableEl).appendChild(h('div', { class: 'table-wrap' }, table));
  const opts = {
    // ink bars: the pair is named by its label, never by a hue
    items: rows.map((r) => ({ key: r.k, label: pairLabel(r.x, r.y), value: r.ja.ratio, baseline: r.ja.baseline, domain: [0, 1] })),
    format: (v) => fmt.pct(v), valueLabel: 'Joint attention', baselineLabel: 'Baseline', excessLabel: 'Above baseline', label: 'Joint attention against baseline',
  };
  if (!UI.pairs.chart) UI.pairs.chart = bullet(UI.pairs.jaEl, opts);
  else UI.pairs.chart.update(opts);
}

function trKey(r) {
  const k3 = (v) => (finite(v) ? v.toFixed(3) : '');
  return `${k3(r.t)}|${k3(r.e)}|${r.pt ?? ''}|${r.sp ?? ''}`;
}

function speakerChips(r) {
  if (r.pt != null) {
    const pupil = isPupilTag(r.pt);
    return [chip(pupil ? S.ident.tag(r.pt).color : 'var(--tag-other)', pupil ? `Tag ${r.pt} (worn mic)` : `${r.pt} (worn mic)`)];
  }
  const keys = [];
  for (const tn of r.turns || []) if (tn && tn[2] && !keys.includes(tn[2])) keys.push(tn[2]);
  for (const w of r.w || []) if (w && w[3] && !keys.includes(w[3])) keys.push(w[3]);
  if (keys.length) {
    const out = keys.slice(0, 3).map((k) => {
      const v = S.voices.get(k);
      return chip(v.color, v.label);
    });
    if (keys.length > 3) out.push(h('span', { class: 'muted', text: `+${keys.length - 3}`, title: keys.slice(3).map((k) => S.voices.get(k).label).join(', ') }));
    return out;
  }
  const sp = r.sp != null ? String(r.sp) : '';
  if (sp && !GROUP_LABEL_RE.test(sp) && S.model.entityMode() === 'speakers' && sp !== 'silent' && sp !== 'unknown') {
    const v = S.voices.get(`spk:${sp}`);
    return [chip(v.color, v.label)];
  }
  return [chip('var(--ink-2)', 'Group mic')];
}

function chunkNode(r) {
  const cf = clockFormat();
  const meta = h('div', { class: 'tr-meta' }, h('span', { class: 'tr-time', text: cf(Math.max(0, r.t - S.t0)), title: fmt.time(r.t) }), speakerChips(r));
  const p = h('p', { class: 'tr-text' });
  const words = Array.isArray(r.w) ? r.w : [];
  if (words.length) {
    words.forEach((w, i) => {
      if (i) p.appendChild(document.createTextNode(' '));
      const span = h('span', { text: String(w && w[0] != null ? w[0] : '') });
      const key = w && w[3];
      const cls = w && w[4];
      if (r.pt != null && (cls === 'crosstalk' || cls === 'other')) {
        span.className = 'w-dim';
        span.title = cls === 'crosstalk' ? 'Crosstalk: likely another wearer' : 'Likely not the wearer';
      } else if (key && r.pt == null) {
        const v = S.voices.get(key);
        span.className = 'w-v';
        span.style.setProperty('--u', v.color);
        span.title = v.label;
      }
      p.appendChild(span);
    });
  } else if (r.text) p.textContent = r.text;
  else {
    // the chunk heard speech but the transcriber found no words in it
    const length = finite(r.e) && finite(r.t) && r.e > r.t ? r.e - r.t : null;
    p.appendChild(h('span', { class: 'tr-empty-text', text: length != null ? `No words recognised in ${fmt.duration(Math.max(1, length))}` : 'No words recognised' }));
  }
  return h('li', { class: 'tr-chunk' }, meta, p);
}

function renderTranscript(now, loading) {
  const T = UI.tr;
  const c = T.card;
  if (T.epoch !== S.epoch) {
    clear(T.list);
    T.nodes.clear();
    T.epoch = S.epoch;
    T.stick = true;
    T.jump.hidden = true;
  }
  const chunks = S.model.chunks(now);
  if (!chunks.length) {
    if (loading) {
      c.setState('loading', 'Loading the transcript');
      return;
    }
    c.setState('ready');
    const asr = S.meta && S.meta.modalities && S.meta.modalities.asr;
    T.emptyLine.textContent = !asr ? 'This session has no speech data.' : 'No transcript in the buffered window yet.';
    T.emptyLine.hidden = false;
  } else {
    c.setState('ready');
    T.emptyLine.hidden = true;
  }
  const keep = new Set();
  let added = false;
  let prev = null;
  for (const r of chunks) {
    const k = trKey(r);
    keep.add(k);
    let node = T.nodes.get(k);
    if (!node) {
      node = chunkNode(r);
      T.nodes.set(k, node);
      if (prev) prev.after(node);
      else T.list.prepend(node);
      added = true;
    }
    prev = node;
  }
  for (const [k, node] of T.nodes) {
    if (!keep.has(k)) {
      node.remove();
      T.nodes.delete(k);
    }
  }
  // only while records still flow: a paused replay transcribes nothing
  T.pending.hidden = !((S.mode === 'follow' || S.playing) && S.model.transcribing(now));
  if (added || !T.pending.hidden) {
    if (T.stick) T.scroller.scrollTop = T.scroller.scrollHeight;
    else if (added) T.jump.hidden = false;
  }
}

function renderCameras(now) {
  const wall = UI.cams.wall;
  // one tile per camera the session's document names (VFA first, then IPS), and the VFA cameras of
  // the frame sets it does not; without a document, the cameras it would have listed
  const video = S.meta && Array.isArray(S.meta.video) ? S.meta.video : [];
  const vfaIds = sortTags(Array.from(S.model.cameraIds));
  const fallback = !video.length && !vfaIds.length ? sortTags((S.meta && S.meta.devices && S.meta.devices.cameras) || []) : [];
  const ids = cameraItems(video, vfaIds.length ? vfaIds : fallback);
  if (S.camsOpen) {
    if (!ids.length) {
      const noVfa = S.meta && S.meta.modalities && !S.meta.modalities.vfa;
      UI.cams.card.setState('empty', noVfa ? 'This session has no video features: VFA did not run.' : 'No camera frames in this window yet.');
    } else UI.cams.card.setState('ready');
    if (S.mode === 'follow') loadMedia(false);
    else if (endedReplay()) loadRecordings(false);
  }
  const newestVfa = S.model.vfa.last;
  // the sound's file takes one of the connections the tiles' recorded files share (on the page's
  // origin; from the media ports it has one of its own, on the first)
  const reserved = S.sound.takesMedia(soundSource(soundState()), S.speed) ? 1 : 0;
  wall.update({
    cameras: ids,
    sid: S.sid,
    frameAt: (t) => S.model.vfaAt(t),
    now,
    mode: S.mode,
    live: S.live,
    media: S.media,
    serverNow: serverNow(),
    vfaLag: newestVfa ? serverNow() - newestVfa.t : null,
    recordings: S.recordings,
    speed: S.speed,
    running: clockRunning(now),
    mediaOrigins: S.mediaOrigins,
    mediaReserved: reserved,
  });
  UI.cams.videoToggles.hidden = !wall.hasVideo();
  // holding the video back only concerns live video
  UI.cams.syncBtn.hidden = !wall.hasLiveVideo();
}

// page

function errorPage(root, res) {
  clear(root);
  let title = 'Could not load this session.';
  let body = res.error || null;
  let action = { label: 'Retry', onClick: () => window.location.reload() };
  if (res.status === 404) {
    title = 'Session not found.';
    body = `No session with the id ${S.sid} is in InfluxDB or MongoDB.`;
    action = { label: 'Go to sessions', href: '/' };
  } else if (res.status === 400) {
    title = 'This link has no valid session id.';
    action = { label: 'Go to sessions', href: '/' };
  } else if (res.status === 503) title = 'InfluxDB did not answer.';
  else if (res.status === 0) title = 'Could not reach the dashboard server.';
  root.appendChild(h('div', { class: 'live-empty' }, emptyState({ title, body, action, kind: res.status === 404 || res.status === 400 ? 'empty' : 'error' })));
}

function loadingPage(root) {
  clear(root);
  root.appendChild(h('div', { attrs: { 'aria-busy': 'true' } },
    h('div', { class: 'skeleton', style: { width: '280px', height: '24px', marginBottom: '10px' } }),
    h('div', { class: 'skeleton', style: { width: '180px', height: '14px', marginBottom: '20px' } }),
    h('div', { class: 'shimmer', style: { minHeight: '56px', marginBottom: '16px' } }),
    h('div', { class: 'shimmer', style: { minHeight: '320px' } })));
}

async function main() {
  theme.init();
  const top = document.getElementById('topbar');
  const root = document.getElementById('live-root');
  S.sid = sessionIdFromUrl();
  if (!S.sid) {
    top.appendChild(topbar({ crumbs: [{ label: 'Sessions', href: '/' }, { label: 'Live' }] }));
    root.appendChild(h('div', { class: 'live-empty' }, emptyState({ title: sessionParamInvalid() ? 'This is not a valid session id.' : 'No session chosen.', body: 'Open a session from the Sessions page to watch it live or replay it.', action: { label: 'Go to sessions', href: '/' } })));
    return;
  }
  UI.top = topbar({ session: { id: S.sid }, view: 'live' });
  top.appendChild(UI.top);
  loadingPage(root);
  // the colours come from the report parts: ask now, beside the meta, and give no voice or tag a colour
  // slot until they answer (or COLOUR_WAIT_MS passes)
  S.coloursPending = true;
  S.voices.pending = true;
  loadColourSources(true);
  setTimeout(() => {
    if (S.coloursPending) settleColours();
  }, COLOUR_WAIT_MS);
  const res = await api(`/api/sessions/${encodeURIComponent(S.sid)}`);
  if (!res.ok || !res.data) {
    errorPage(root, res);
    return;
  }
  applyMeta(res.data);
  if (S.t0 == null && !S.live) {
    clear(root);
    root.appendChild(h('div', { class: 'live-empty' }, sessionHeading(S.meta), h('div', { style: { marginTop: '16px' } }, emptyState({
      title: 'This session has no measurements yet.',
      body: 'The live view fills in once a pipeline writes to InfluxDB for this session.',
      action: { label: 'Retry', onClick: () => window.location.reload() },
    }))));
    waitForData(root);
    return;
  }
  start(root);
}

function applyMeta(meta) {
  S.meta = meta;
  S.t0 = finite(meta.t0) ? meta.t0 : null;
  S.t1 = finite(meta.t1) ? meta.t1 : null;
  S.live = !!(meta.state && meta.state.live);
  S.lastEvent = meta.state && finite(meta.state.last_event) ? meta.state.last_event : null;
  UI.top.update({ session: meta });
  document.title = `Live: ${sessionTitle(meta).title}`;
  S.model = new LiveModel({ t0: S.t0, t1: S.t1, groupId: meta.group || null, speechMode: (meta.modalities && meta.modalities.asr && meta.modalities.asr.mode) || null, keep: S.window + 120 });
}

/** a session without measurements yet: ask for its meta every few seconds and open the view once data arrives. */
function waitForData(root) {
  let busy = false;
  const timer = setInterval(async () => {
    if (busy || document.visibilityState === 'hidden') return;
    busy = true;
    const res = await api(`/api/sessions/${encodeURIComponent(S.sid)}`);
    busy = false;
    const meta = res.ok ? res.data : null;
    if (!meta || (!finite(meta.t0) && !(meta.state && meta.state.live))) return;
    clearInterval(timer);
    applyMeta(meta);
    start(root);
  }, EMPTY_POLL_MS);
}

function start(root) {
  const meta = S.meta;
  build(root);
  S.cameras = projectCameras(meta.ips_cameras || [], null);
  UI.room.map.setCameras(S.cameras);
  const forced = new URLSearchParams(window.location.search).get('mode');
  if (S.live || forced === 'follow') startFollow();
  else startPreview(openingMoment());
  setInterval(() => {
    if (document.visibilityState !== 'hidden') requestFrame();
  }, FRAME_MS);
  // a live session's speech part grows: ask again for voices that have no colour yet (never repainting one)
  setInterval(() => {
    if (document.visibilityState === 'hidden') return;
    if (S.live || !S.speechReady) loadColourSources(false);
  }, SPEECH_RECHECK_MS);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'hidden') {
      S.force = true;
      requestFrame();
    } else {
      tooltip.hide();
      // a hidden page lets go of the sound's file, as the tiles do of theirs; the next redraw loads it again
      S.sound.release();
    }
  });
  window.addEventListener('pagehide', (e) => {
    // a playing replay goes on from the moment drawn; the backfill covers what was not sent yet
    if (S.mode === 'replay' && S.playing) S.clock = displayNow() ?? S.clock;
    stopStream();
    S.connState = 'idle';
    S.sound.release();
    // a page kept in the back/forward cache keeps its tiles, with every video closed
    if (e.persisted) UI.cams.wall.suspend();
    else UI.cams.wall.destroy();
  });
  window.addEventListener('pageshow', (e) => {
    if (!e.persisted) return;
    // back from the back/forward cache: open the stream again as the page left it (follow, play or
    // preview; a paused replay stays closed)
    UI.cams.wall.resume();
    if (S.raf && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(S.raf);
    S.raf = 0;
    S.attempt = 0;
    S.force = true;
    reconnectNow();
  });
  // a hook for checks from the browser console
  window.__live = { S, UI, seek, play, pause, setSpeed, startFollow, startPreview, replayFromStart, sound: () => S.sound.inspect() };
}

main();
