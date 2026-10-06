/**
 * Camera tiles of the live page. A tile plays the camera's live video over WebRTC (WHEP against
 * MediaMTX) when the session streams it and the stream server has WebRTC on, with the VFA skeletons
 * drawn over the video. In the replay of an ended session it plays the camera's recorded file from
 * the dashboard (the recordings route), kept in step with the replay clock. Otherwise it draws the
 * skeletons of the newest frame set on a blank canvas.
 *
 * Video costs bandwidth on the camera's uplink and the stream server, so a tile connects only while
 * the Cameras card is open, the tile is on screen and the tab is visible, and at most MAX_PLAYING
 * tiles play at once; everything else closes its RTCPeerConnection or lets go of its file. A recorded
 * file holds one of the browser's connections to the dashboard, so the files load from the
 * dashboard's media ports: up to MAX_MEDIA_VIDEOS beside the replay's sound on the first that
 * answers, up to MAX_ORIGIN_FILES on each other one (mediaPools, assignFiles); without one they load
 * from the page's origin and share MAX_MEDIA with the sound. A tile scrolled off screen keeps its
 * file, paused, while that allows, so scrolling back shows the moment without loading the file
 * again, and a tile keeps the port its file loads from while it holds one. The WHEP exchange lives
 * in whepNegotiate and WhepPlayer, which take their RTCPeerConnection and fetch as arguments so the
 * offer/answer flow can be tested without a browser or a server; FilePlayer takes its video element
 * and clock the same way.
 */

import { h, clear, fmt, tooltip, theme, icon } from './core.js';
import { colorResolver } from './charts.js';
import { CATEGORY_LABELS, gazeCategory, TAG_MEMORY_SECONDS } from './live-model.js';

// live tiles that play at once (the IPS cameras' tiles play video too)
export const MAX_PLAYING = 6;
// a browser opens at most six connections per origin (scheme, host and port) over HTTP/1.1, and a
// recorded file (a camera's video or the replay's sound) holds one while it plays, and while it is
// paused in place. The page loads the files from the dashboard's media ports, origins of their own
// (mediaOriginFor): on each, up to MAX_ORIGIN_FILES files, which leaves one of the six free there for
// a seek. The page's own origin keeps its six for the API requests and the live stream
export const MAX_ORIGIN_FILES = 5;
// the first media origin that answers carries the replay's sound too: up to MAX_MEDIA_VIDEOS camera
// videos beside it (the place stays the sound's while it is off, so no video moves when it comes on)
export const MAX_MEDIA_VIDEOS = MAX_ORIGIN_FILES - 1;
// without a media port (or when it does not answer) the files load from the page's origin, where
// the live stream takes one connection and one stays free for the page's other requests: the
// videos and the sound share the other four
export const MAX_MEDIA = 4;
// how long the page waits for the media port's answer before it loads the files from its own origin
export const MEDIA_PROBE_MS = 3000;
// once one media port answered, the others are waited for this long only (awaitMediaAnswers)
export const MEDIA_PROBE_GRACE_MS = 300;
export const RECONNECT_STEPS = [1, 2, 5, 10];
export const ICE_TIMEOUT_MS = 2000;
// the browser holds at most this much video back to line it up with the overlay
export const MAX_VIDEO_DELAY = 4;
export const KEYPOINT_MIN_CONF = 0.3;
// a recorded video seeks once it is this far (media seconds) off the replay clock, at 1x and 2x
export const RECORDING_DRIFT = 0.5;
// a paused replay shows the frame of its moment, give or take this much
export const PAUSED_DRIFT = 0.05;
// a smaller gap closes by playing this much slower or faster
export const RATE_NUDGE = 0.1;
// the playback rates browsers accept
export const MIN_RATE = 0.0625;
export const MAX_RATE = 16;
// a speed the browser refuses: the paused video steps to the clock this often
export const STEP_MS = 1000;
// a recording that starts within this many seconds of wall time loads before the clock reaches it
export const RECORDING_LOOKAHEAD = 3;
// a player that cannot play faster (the sound at its 4x limit) and lags the clock seeks ahead by
// this much wall time (the time a seek takes), at most once per CATCH_UP_MS
export const SEEK_LEAD = 0.1;
export const CATCH_UP_MS = 1000;
// COCO-17 limbs: face, arms, torso, legs
export const COCO_LIMBS = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

function finite(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

// WHEP

export class WhepError extends Error {
  constructor(message, status = null) {
    super(message);
    this.name = 'WhepError';
    this.status = status;
  }
}

/** `${webrtc}/${path}/whep`, each path segment encoded. */
export function whepUrl(base, path) {
  const root = String(base || '').replace(/\/+$/, '');
  const p = String(path || '').split('/').filter(Boolean).map(encodeURIComponent).join('/');
  return `${root}/${p}/whep`;
}

/** resolves when ICE gathering is complete or after `timeoutMs`, whichever comes first. */
export function waitIceGathering(pc, timeoutMs = ICE_TIMEOUT_MS, { setTimer = setTimeout, clearTimer = clearTimeout } = {}) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve(true);
  return new Promise((resolve) => {
    let timer = 0;
    const done = (complete) => {
      clearTimer(timer);
      pc.removeEventListener('icegatheringstatechange', onChange);
      resolve(complete);
    };
    const onChange = () => {
      if (pc.iceGatheringState === 'complete') done(true);
    };
    pc.addEventListener('icegatheringstatechange', onChange);
    timer = setTimer(() => done(false), timeoutMs);
  });
}

/**
 * One WHEP exchange on `pc`: a receive-only transceiver of each of `kinds` (video and audio; a
 * microphone's sound takes audio alone), an offer with its ICE candidates (gathering waited for up
 * to `iceTimeout` ms), POST it as application/sdp, expect 201 with the answer, set it. `what` names
 * the media in the errors. Returns {location} (the session URL for DELETE, or null).
 */
export async function whepNegotiate(pc, url, {
  fetchImpl = globalThis.fetch, iceTimeout = ICE_TIMEOUT_MS, signal, timers, kinds = ['video', 'audio'], what = 'video',
} = {}) {
  for (const kind of kinds) pc.addTransceiver(kind, { direction: 'recvonly' });
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitIceGathering(pc, iceTimeout, timers);
  const sdp = (pc.localDescription && pc.localDescription.sdp) || offer.sdp;
  let res;
  try {
    res = await fetchImpl(url, { method: 'POST', headers: { 'Content-Type': 'application/sdp' }, body: sdp, signal });
  } catch (err) {
    if (err && err.name === 'AbortError') throw err;
    throw new WhepError(`The stream server did not answer (${(err && err.message) || 'network error'}).`);
  }
  if (res.status !== 201) {
    let detail = '';
    try {
      detail = (await res.text()).trim().slice(0, 160);
    } catch {
      detail = '';
    }
    const why = res.status === 404 ? 'The stream is not published' : `The stream server refused the ${what} (HTTP ${res.status})`;
    throw new WhepError(`${why}${detail ? `: ${detail}` : '.'}`, res.status);
  }
  const answer = await res.text();
  if (!answer || !/^v=0/m.test(answer)) throw new WhepError('The stream server sent no SDP answer.', res.status);
  await pc.setRemoteDescription({ type: 'answer', sdp: answer });
  let location = null;
  try {
    const loc = res.headers && res.headers.get ? res.headers.get('Location') : null;
    location = loc ? new URL(loc, url).href : null;
  } catch {
    location = null;
  }
  return { location };
}

/** true when the browser can hold received video back (RTCRtpReceiver.jitterBufferTarget). */
export function supportsVideoDelay() {
  return typeof RTCRtpReceiver !== 'undefined' && 'jitterBufferTarget' in RTCRtpReceiver.prototype;
}

/**
 * A live video over WHEP that reconnects with backoff (1, 2, 5, 10 s) until stopped.
 * onState({state: 'connecting'|'playing'|'retrying'|'stopped', message, retryIn}). With
 * `kinds: ['audio']` and an <audio> element as `video` it plays a microphone's sound (`what` names
 * the media in its messages).
 */
export class WhepPlayer {
  constructor({
    url, video = null, onState = () => {}, RTC = globalThis.RTCPeerConnection,
    fetchImpl = globalThis.fetch ? globalThis.fetch.bind(globalThis) : null, iceTimeout = ICE_TIMEOUT_MS,
    steps = RECONNECT_STEPS, setTimer = (fn, ms) => setTimeout(fn, ms), clearTimer = (id) => clearTimeout(id),
    disconnectGrace = 4000, kinds = ['video', 'audio'], what = 'video',
  } = {}) {
    this.url = url;
    this.video = video;
    this.kinds = kinds;
    this.what = what;
    this.onState = onState;
    this.RTC = RTC;
    this.fetchImpl = fetchImpl;
    this.iceTimeout = iceTimeout;
    this.steps = steps;
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
    this.disconnectGrace = disconnectGrace;
    this.pc = null;
    this.location = null;
    this.wanted = false;
    this.attempt = 0;
    this.retryTimer = 0;
    this.graceTimer = 0;
    this.delay = 0;
    this.abort = null;
    this.state = 'stopped';
  }

  emit(state, extra = {}) {
    this.state = state;
    try {
      this.onState({ state, ...extra });
    } catch {
      // a broken listener must not stop the player
    }
  }

  start() {
    if (this.wanted) return;
    this.wanted = true;
    this.attempt = 0;
    this.connect();
  }

  async connect() {
    if (!this.wanted) return;
    if (!this.RTC) {
      this.wanted = false;
      this.emit('stopped', { message: 'This browser has no WebRTC.' });
      return;
    }
    this.closePc();
    this.emit('connecting');
    let pc;
    try {
      pc = new this.RTC({ bundlePolicy: 'max-bundle' });
    } catch (err) {
      this.fail(new WhepError(`WebRTC is not available (${(err && err.message) || 'error'}).`));
      return;
    }
    this.pc = pc;
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    this.abort = controller;
    pc.ontrack = (e) => {
      if (pc !== this.pc || !this.video) return;
      const stream = e.streams && e.streams[0] ? e.streams[0] : null;
      if (stream) {
        if (this.video.srcObject !== stream) this.video.srcObject = stream;
      } else if (typeof MediaStream !== 'undefined') {
        const ms = this.video.srcObject instanceof MediaStream ? this.video.srcObject : new MediaStream();
        ms.addTrack(e.track);
        this.video.srcObject = ms;
      }
      this.applyDelay();
    };
    pc.onconnectionstatechange = () => this.onConnectionState(pc);
    try {
      const { location } = await whepNegotiate(pc, this.url, {
        fetchImpl: this.fetchImpl, iceTimeout: this.iceTimeout, signal: controller ? controller.signal : undefined,
        timers: { setTimer: this.setTimer, clearTimer: this.clearTimer }, kinds: this.kinds, what: this.what,
      });
      if (pc !== this.pc) return;
      this.location = location;
      this.applyDelay();
      if (pc.connectionState === 'connected') this.onConnectionState(pc);
    } catch (err) {
      if (pc !== this.pc || !this.wanted || (err && err.name === 'AbortError')) return;
      this.fail(err);
    }
  }

  onConnectionState(pc) {
    if (pc !== this.pc) return;
    const st = pc.connectionState;
    if (st === 'connected') {
      this.clearTimer(this.graceTimer);
      this.graceTimer = 0;
      this.attempt = 0;
      this.emit('playing');
    } else if (st === 'failed') {
      this.fail(new WhepError(`The ${this.what} connection failed.`));
    } else if (st === 'disconnected') {
      this.clearTimer(this.graceTimer);
      this.graceTimer = this.setTimer(() => {
        if (pc === this.pc && pc.connectionState === 'disconnected') this.fail(new WhepError(`The ${this.what} connection dropped.`));
      }, this.disconnectGrace);
    }
  }

  fail(err) {
    this.closePc();
    if (!this.wanted) return;
    const wait = this.steps[Math.min(this.attempt, this.steps.length - 1)];
    this.attempt += 1;
    this.emit('retrying', { message: (err && err.message) || `The ${this.what} stopped.`, retryIn: wait });
    this.clearTimer(this.retryTimer);
    this.retryTimer = this.setTimer(() => {
      this.retryTimer = 0;
      this.connect();
    }, wait * 1000);
  }

  closePc() {
    this.clearTimer(this.graceTimer);
    this.graceTimer = 0;
    if (this.abort) {
      try {
        this.abort.abort();
      } catch {
        // already aborted
      }
      this.abort = null;
    }
    const pc = this.pc;
    this.pc = null;
    if (pc) {
      pc.ontrack = null;
      pc.onconnectionstatechange = null;
      try {
        pc.close();
      } catch {
        // closing twice is fine
      }
    }
    const loc = this.location;
    this.location = null;
    if (loc && this.fetchImpl) {
      // free the session on the stream server; best effort
      Promise.resolve().then(() => this.fetchImpl(loc, { method: 'DELETE' })).catch(() => {});
    }
  }

  stop() {
    this.wanted = false;
    this.clearTimer(this.retryTimer);
    this.retryTimer = 0;
    this.closePc();
    if (this.video) {
      try {
        this.video.srcObject = null;
      } catch {
        // detached video
      }
    }
    this.emit('stopped');
  }

  /** hold the video back by `seconds` (capped at MAX_VIDEO_DELAY); true when the browser can. */
  setDelay(seconds) {
    this.delay = Math.max(0, Math.min(MAX_VIDEO_DELAY, Number(seconds) || 0));
    return this.applyDelay();
  }

  applyDelay() {
    const pc = this.pc;
    if (!pc || typeof pc.getReceivers !== 'function') return false;
    let ok = false;
    for (const r of pc.getReceivers()) {
      if (r && r.track && r.track.kind === 'video' && 'jitterBufferTarget' in r) {
        try {
          r.jitterBufferTarget = Math.round(this.delay * 1000);
          ok = true;
        } catch {
          ok = false;
        }
      }
    }
    return ok;
  }
}

// recorded video

/**
 * how far (media seconds) a recorded video may run off the replay clock before it seeks: 0.5 s, or a
 * quarter second of wall time from 4x on, where the replay clock itself steps that much between the
 * stream's batches (the overlay follows the video's own time, so the skeletons stay on the picture).
 */
export function driftTolerance(speed) {
  const s = finite(speed) && speed > 0 ? speed : 1;
  return Math.max(RECORDING_DRIFT, 0.25 * s);
}

/**
 * [start, end] of a recording in epoch seconds, null without a start. The end comes from the listed
 * duration, Infinity without one (the recordings route lists a length for every file the dashboard's
 * machine can read, from its manifest, ffprobe or the recorder's start and stop). Never from the length the browser reads: Firefox reads a
 * fragmented MP4 whose header names no length (a stream server's cut, a CMAF segment) one fragment
 * at a time, so its duration grows from the first fragment's as the file loads and stops where the
 * download pauses, and a span cut to it would leave the tile without its file, and without the
 * player that could read further, for the rest of the replay.
 */
export function recordingSpan(file) {
  const start = file && finite(file.start) ? file.start : null;
  if (start == null) return null;
  const d = file.duration;
  return [start, finite(d) && d > 0 ? start + d : Infinity];
}

/**
 * the recording of `files` that holds the moment t (epoch), the one that started last when several
 * do (its last frame stays up at its very end, where a replay that reaches the session's end stops);
 * null when none does.
 */
export function recordingAt(files, t) {
  if (!finite(t)) return null;
  let best = null;
  for (const f of files || []) {
    const span = recordingSpan(f);
    if (span && t >= span[0] && t <= span[1] && (!best || f.start > best.start)) best = f;
  }
  return best;
}

/** the first recording of `files` that starts after t and within `within` seconds of it; null when none does. */
export function nextRecording(files, t, within) {
  if (!finite(t) || !(within > 0)) return null;
  let best = null;
  for (const f of files || []) {
    if (f && finite(f.start) && f.start > t && f.start - t <= within && (!best || f.start < best.start)) best = f;
  }
  return best;
}

/** the box a vw x vh video fills inside a w x h element with object-fit: contain; null without sizes. */
export function containBox(vw, vh, w, hgt) {
  if (!(vw > 0) || !(vh > 0) || !(w > 0) || !(hgt > 0)) return null;
  const k = Math.min(w / vw, hgt / vh);
  return { x: (w - vw * k) / 2, y: (hgt - vh * k) / 2, w: vw * k, h: vh * k };
}

/**
 * a recording's url for the player: from `origin` (the media origin, see mediaOriginFor) when one is
 * given and the url is a path, and inline, so the dashboard answers the ranges a seek asks for.
 */
export function inlineUrl(url, origin = null) {
  let u = String(url || '');
  if (origin && u.startsWith('/') && !u.startsWith('//')) u = `${String(origin).replace(/\/+$/, '')}${u}`;
  return `${u}${u.includes('?') ? '&' : '?'}inline=1`;
}

/**
 * an origin the recorded files load from: the page's scheme and host on a media port of the
 * dashboard (`port`, one of the recordings route's media_ports), so the files hold connections of
 * their own; null without a port, for the page's own port, or for a page not served over http(s).
 */
export function mediaOriginFor(loc, port) {
  const p = Number(port);
  if (!loc || !Number.isInteger(p) || p < 1 || p > 65535) return null;
  const scheme = String(loc.protocol || '');
  const hostname = String(loc.hostname || '');
  if ((scheme !== 'http:' && scheme !== 'https:') || !hostname) return null;
  const own = Number(loc.port) || (scheme === 'https:' ? 443 : 80);
  if (p === own) return null;
  // an IPv6 hostname keeps its brackets in location.hostname
  return `${scheme}//${hostname}:${p}`;
}

/**
 * resolves once one of `answers` (promises of true or false, the media ports' probes) came back true
 * and the others had `graceMs` more, or once all came back, whichever is first; never rejects. A port
 * a firewall drops gives up only after MEDIA_PROBE_MS, and the files would otherwise wait for it on
 * every load while another port already answered.
 */
export function awaitMediaAnswers(answers, { graceMs = MEDIA_PROBE_GRACE_MS, setTimer = (fn, ms) => setTimeout(fn, ms) } = {}) {
  const list = (answers || []).map((a) => Promise.resolve(a).then((ok) => ok === true, () => false));
  if (!list.length) return Promise.resolve();
  return new Promise((resolve) => {
    let left = list.length;
    let timer = false;
    for (const p of list) {
      p.then((ok) => {
        left -= 1;
        if (!left) resolve();
        else if (ok && !timer) {
          timer = true;
          setTimer(resolve, graceMs);
        }
      });
    }
  });
}

/**
 * whether this same dashboard answers on `origin` within `timeoutMs`: its /api/media-origin names
 * `port` among its media ports and `instance` (the recordings route's media_ports and media_instance,
 * a mark of the dashboard's process), so another dashboard that holds that port number on the page's
 * host (a page opened through a tunnel on another local port) is not taken for it; false for no
 * answer, another server, or an error. Never throws.
 */
export async function probeMediaOrigin(origin, port, instance, {
  fetchImpl = globalThis.fetch ? globalThis.fetch.bind(globalThis) : null, timeoutMs = MEDIA_PROBE_MS,
  setTimer = (fn, ms) => setTimeout(fn, ms), clearTimer = (id) => clearTimeout(id),
} = {}) {
  if (!origin || !fetchImpl || typeof instance !== 'string' || !instance) return false;
  const ctl = typeof AbortController !== 'undefined' ? new AbortController() : null;
  let timer = 0;
  const late = new Promise((resolve) => {
    timer = setTimer(() => {
      if (ctl) ctl.abort();
      resolve(false);
    }, timeoutMs);
  });
  const ask = (async () => {
    try {
      const res = await fetchImpl(`${String(origin).replace(/\/+$/, '')}/api/media-origin`, {
        cache: 'no-store', credentials: 'omit', headers: { Accept: 'application/json' }, signal: ctl ? ctl.signal : undefined,
      });
      if (!res || !res.ok) return false;
      const data = await res.json();
      if (!data || data.media_instance !== instance) return false;
      // a dashboard of one media port answers media_port alone
      const ports = Array.isArray(data.media_ports) ? data.media_ports : [data.media_port];
      return ports.some((p) => p != null && p !== '' && Number(p) === Number(port));
    } catch {
      return false;
    }
  })();
  try {
    return await Promise.race([ask, late]);
  } finally {
    clearTimer(timer);
  }
}

/**
 * how many recorded camera videos the tiles may hold at once: MAX_MEDIA_VIDEOS from the media port
 * (`separate`), else what the sound (`reserved`, 0 or 1) leaves of MAX_MEDIA on the page's origin
 */
export function videoCap(separate, reserved = 0) {
  if (separate) return MAX_MEDIA_VIDEOS;
  const r = finite(reserved) ? Math.max(0, Math.round(reserved)) : 0;
  return Math.max(0, MAX_MEDIA - r);
}

/** the media origins as a list, each once, in order: from a list, one origin, or nothing */
export function mediaOriginList(origins) {
  const list = Array.isArray(origins) ? origins : [origins];
  const out = [];
  for (const o of list) if (typeof o === 'string' && o && !out.includes(o)) out.push(o);
  return out;
}

/**
 * where the tiles' recorded videos may load from, each {origin, cap}: the media origins that
 * answered (`origins`, in the dashboard's order), MAX_MEDIA_VIDEOS on the first (it carries the
 * replay's sound) and MAX_ORIGIN_FILES on each other one; without one, the page's own origin (origin
 * null) with what the sound (`reserved`) leaves of MAX_MEDIA (videoCap)
 */
export function mediaPools(origins, reserved = 0) {
  const list = mediaOriginList(origins);
  if (!list.length) return [{ origin: null, cap: videoCap(false, reserved) }];
  return list.map((origin, i) => ({ origin, cap: i === 0 ? MAX_MEDIA_VIDEOS : MAX_ORIGIN_FILES }));
}

/**
 * which tiles hold a recorded video, and the origin each loads it from. `playing`: the tiles on
 * screen that want one, in priority order; `parked`: the tiles scrolled away that hold one, the most
 * recently seen first; `held`: the origin each tile's file loads from now (Map id -> origin, null for
 * the page's origin), for the tiles that hold one; `pools`: mediaPools(). The tiles on screen come
 * first and the parked ones take what they leave, up to the pools' caps together; every pool holds
 * at most its cap. A tile keeps its origin while that pool has room for it, so another tile letting
 * go of its file reloads nothing; a parked tile keeps its file only on its own origin (a parked video
 * loads nothing). A tile that holds no file yet goes to its home origin (`homes`, homeOrigins) while
 * that has room, else where the most room is left, the first pool on a tie: the browser keeps a
 * file in its cache by origin, so the same camera loading from the same origin finds its file there
 * after a scroll back or a reopen. Returns {playing: Map id -> origin, parked: Map id -> origin,
 * cap: the pools' caps together}.
 */
export function assignFiles({ playing = [], parked = [], held = new Map(), pools = [], homes = new Map() } = {}) {
  const caps = new Map();
  for (const p of pools) if (p && !caps.has(p.origin)) caps.set(p.origin, Math.max(0, Math.round(Number(p.cap) || 0)));
  const used = new Map(Array.from(caps.keys(), (o) => [o, 0]));
  let cap = 0;
  for (const c of caps.values()) cap += c;
  let total = 0;
  const kept = new Map();
  const fresh = [];
  const room = (o) => caps.has(o) && used.get(o) < caps.get(o);
  const keep = (id, o) => {
    used.set(o, used.get(o) + 1);
    kept.set(id, o);
    total += 1;
  };
  const onScreen = [];
  for (const id of playing) {
    if (total >= cap) break;
    onScreen.push(id);
    if (held.has(id) && room(held.get(id))) keep(id, held.get(id));
    else {
      fresh.push(id);
      total += 1;
    }
  }
  // a camera on screen that holds no file goes home first, before a parked one keeps its place
  // there: a parked tile is the lesser loss, as it loads from its own home again later
  const left = [];
  for (const id of fresh) {
    if (homes.has(id) && room(homes.get(id))) {
      used.set(homes.get(id), used.get(homes.get(id)) + 1);
      kept.set(id, homes.get(id));
    } else left.push(id);
  }
  const away = [];
  for (const id of parked) {
    if (total >= cap) break;
    if (held.has(id) && room(held.get(id))) {
      keep(id, held.get(id));
      away.push(id);
    }
  }
  // the kept ones fit their pools and all of them fit the caps together: the others fit what is left
  for (const id of left) {
    let best;
    let free = 0;
    for (const [o, c] of caps) {
      if (c - used.get(o) > free) {
        free = c - used.get(o);
        best = o;
      }
    }
    if (best === undefined) break;
    used.set(best, used.get(best) + 1);
    kept.set(id, best);
  }
  return {
    playing: new Map(onScreen.filter((id) => kept.has(id)).map((id) => [id, kept.get(id)])),
    parked: new Map(away.map((id) => [id, kept.get(id)])),
    cap,
  };
}

/**
 * the origin each camera's recorded files load from first (Map key -> origin): `keys`, the cameras
 * that have recordings, in the order of their names, each to the pool whose homes fill the least of
 * its cap (the larger cap, then the earlier pool, on a tie), so any set of cameras that fits the
 * caps together fits at home. `kept`: homes dealt before over the same pools, which stay as they
 * are (a camera that joins later gets a home of its own and moves no other one).
 */
export function homeOrigins(keys, pools, kept = null) {
  const usable = (pools || []).filter((p) => p && p.cap > 0);
  const homes = new Map();
  if (!usable.length) return homes;
  const count = new Map(usable.map((p) => [p.origin, 0]));
  const sorted = Array.from(keys || []).map(String).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  for (const key of sorted) {
    if (kept && kept.has(key) && count.has(kept.get(key))) {
      homes.set(key, kept.get(key));
      count.set(kept.get(key), count.get(kept.get(key)) + 1);
    }
  }
  for (const key of sorted) {
    if (homes.has(key)) continue;
    let best = usable[0];
    for (const p of usable.slice(1)) {
      const a = count.get(p.origin) / p.cap;
      const b = count.get(best.origin) / best.cap;
      if (a < b || (a === b && p.cap > best.cap)) best = p;
    }
    homes.set(key, best.origin);
    count.set(best.origin, count.get(best.origin) + 1);
  }
  return homes;
}

function mediaErrorText(err) {
  const code = err && err.code;
  if (code === 4) return 'The browser could not open the recording: its format does not play here, or the dashboard did not serve it.';
  if (code === 3) return 'The browser could not decode the recording.';
  if (code === 2) return 'The recording stopped loading.';
  return 'The recording could not be played.';
}

/**
 * A recorded video kept in step with the replay clock. load(url) points the video at a file (nothing
 * loads before start()); sync() then runs on every redraw with the media time the clock asks for:
 * the video seeks when it is more than `tolerance` off, plays at the replay speed (a tenth slower or
 * faster to close a smaller gap), pauses with the replay and, paused, shows the frame of the moment.
 * A speed the browser refuses leaves the video paused, stepping to the clock once a second. A file
 * loads from the moment asked for (a media fragment, #t=), and the picture counts as shown
 * (`showing`) only once the video stands within the tolerance of the clock: after a load, and after
 * park() (a tile scrolled off screen keeps its file, paused) and unpark(), the tile draws the
 * skeletons alone until the seek lands, never the file's first frame or an old one.
 * onState({state: 'idle'|'loading'|'ready'|'error', message}). The element may be an <audio>: the
 * replay's sound (sound.js) plays a microphone file by the same rules, with `catchUp`: at its
 * fastest rate it cannot close a lag by playing faster, so it seeks a little ahead instead.
 */
export class FilePlayer {
  constructor({
    video = null, onState = () => {}, now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now()),
    muted = true, maxRate = MAX_RATE, catchUp = false,
  } = {}) {
    this.video = video;
    this.onState = onState;
    this.now = now;
    // a camera's video plays muted; the replay's sound (sound.js) sets muted itself
    this.muted = muted;
    // the fastest rate a nudge may ask for (browsers silence audio above 4x)
    this.maxRate = maxRate;
    this.catchUp = catchUp;
    this.wanted = false;
    this.url = null;
    this.source = null;
    this.state = 'idle';
    this.message = null;
    this.hadFrame = false;
    this.target = null;
    this.seekTo = null;
    this.lastSeek = -Infinity;
    // the seeks asked for, for checks from the browser console
    this.seeks = 0;
    this.stepping = false;
    this.blocked = false;
    // the video stands at the asked moment (see checkAligned); a parked one keeps its file, paused
    this.aligned = false;
    this.parked = false;
    this.tolerance = RECORDING_DRIFT;
    this.badRates = new Set();
    this.listeners = [];
  }

  emit(state, message = null) {
    this.state = state;
    this.message = message;
    try {
      this.onState({ state, message });
    } catch {
      // a broken listener must not stop the player
    }
  }

  start() {
    this.wanted = true;
  }

  /**
   * point the video at `url` (`source` names the file for the caller; the url by default), from the
   * media time `at` (seconds into the file): the browser fetches that part first, and the file's
   * first frame never shows on the way to it.
   */
  load(url, source = url, at = null) {
    if (!this.wanted || !this.video || !url || url === this.url) return;
    this.detach();
    const v = this.video;
    this.url = url;
    this.source = source;
    const on = (name, fn) => {
      v.addEventListener(name, fn);
      this.listeners.push([name, fn]);
    };
    on('loadedmetadata', () => {
      // the seek asked for before the file's length was known (the fragment may not have been taken)
      if (this.target != null && !this.parked && Math.abs(v.currentTime - this.target) > this.tolerance) this.seek(this.target);
    });
    on('loadeddata', () => {
      this.hadFrame = true;
      this.checkAligned();
      this.emit('ready');
    });
    on('seeked', () => {
      this.checkAligned();
      if (this.state !== 'error') this.emit(this.hadFrame ? 'ready' : this.state);
    });
    on('error', () => this.emit('error', mediaErrorText(v.error)));
    if (this.muted) v.muted = true;
    v.preload = 'auto';
    v.src = finite(at) && at > 0 ? `${url}#t=${at.toFixed(3)}` : url;
    this.emit('loading');
  }

  detach() {
    const v = this.video;
    for (const [name, fn] of this.listeners) v.removeEventListener(name, fn);
    this.listeners = [];
    if (this.url && v) {
      try {
        v.pause();
        v.removeAttribute('src');
        // drops the download and the decoder
        v.load();
      } catch {
        // a detached video
      }
    }
    this.url = null;
    this.source = null;
    this.hadFrame = false;
    this.aligned = false;
    this.parked = false;
    this.target = null;
    this.seekTo = null;
    this.lastSeek = -Infinity;
    this.stepping = false;
  }

  stop() {
    this.wanted = false;
    this.detach();
    this.emit('idle');
  }

  /**
   * true while the video shows a frame of the asked moment (the last one stays up during a seek the
   * clock's drift asks for, but not on the way from a load or from off screen)
   */
  get showing() {
    return !!(this.video && this.url && this.state !== 'error' && this.hadFrame && this.aligned && !this.parked && this.target != null);
  }

  /** the video stands within the tolerance of the asked moment, no seek under way: it may be shown */
  checkAligned() {
    const v = this.video;
    if (!this.aligned && v && this.url && this.hadFrame && this.target != null && !v.seeking
        && finite(v.currentTime) && Math.abs(v.currentTime - this.target) <= this.tolerance) this.aligned = true;
    return this.aligned;
  }

  /**
   * off screen: the video keeps its file and place, paused, follows no clock and is not shown, so
   * the tile comes back on screen with the skeletons of the moment, never the picture it left with
   */
  park() {
    if (!this.parked) {
      this.parked = true;
      this.aligned = false;
    }
    if (this.video && this.url) this.pauseVideo();
  }

  /** back on screen: shown again once it stands at the clock's moment */
  unpark() {
    this.parked = false;
  }

  mediaTime() {
    return this.video && finite(this.video.currentTime) ? this.video.currentTime : null;
  }

  seek(t) {
    try {
      this.video.currentTime = t;
    } catch {
      return;
    }
    this.seekTo = t;
    this.lastSeek = this.now();
    this.seeks += 1;
  }

  /** false when the browser refuses `rate` (a refused rate is not asked again). */
  trySetRate(rate) {
    const v = this.video;
    if (this.badRates.has(rate)) return false;
    if (v.playbackRate === rate) return true;
    try {
      v.playbackRate = rate;
    } catch {
      this.badRates.add(rate);
      return false;
    }
    if (Math.abs(v.playbackRate - rate) > 1e-9) {
      this.badRates.add(rate);
      return false;
    }
    return true;
  }

  pauseVideo() {
    if (!this.video.paused) this.video.pause();
  }

  playVideo() {
    const r = this.video.play();
    // a play cut short by a pause rejects; the next sync asks again. One the browser's autoplay
    // rules refused marks the player blocked until a play succeeds
    if (r && typeof r.then === 'function') {
      r.then(() => {
        this.blocked = false;
      }, (err) => {
        if (err && err.name === 'NotAllowedError') this.blocked = true;
      });
    }
  }

  /**
   * target: media seconds the replay clock asks for (null: no recording at this moment); speed: the
   * replay speed; playing: whether the replay clock runs.
   */
  sync({ target = null, speed = 1, playing = false, tolerance = driftTolerance(speed) } = {}) {
    const v = this.video;
    this.target = finite(target) ? Math.max(0, target) : null;
    this.tolerance = tolerance;
    this.stepping = false;
    if (!v || !this.url || this.state === 'error') return;
    if (this.parked) {
      this.pauseVideo();
      return;
    }
    if (this.target == null) {
      this.pauseVideo();
      return;
    }
    // the seek waits for the metadata (loadedmetadata seeks to the newest target)
    if (!(v.readyState >= 1)) return;
    const drift = v.currentTime - this.target;
    const inFlight = (tol) => v.seeking && this.seekTo != null && Math.abs(this.seekTo - this.target) <= tol;
    if (!playing) {
      this.pauseVideo();
      if (Math.abs(drift) > PAUSED_DRIFT && !inFlight(PAUSED_DRIFT)) this.seek(this.target);
      this.checkAligned();
      return;
    }
    const s = finite(speed) && speed > 0 ? speed : 1;
    const gap = Math.abs(drift);
    const nudge = gap > tolerance / 4 && gap <= tolerance && !v.seeking ? (drift > 0 ? 1 - RATE_NUDGE : 1 + RATE_NUDGE) : 1;
    const nudged = s * nudge;
    const want = nudged >= MIN_RATE && nudged <= Math.min(MAX_RATE, this.maxRate) ? nudged : s;
    let ok = this.trySetRate(want);
    if (!ok && want !== s) ok = this.trySetRate(s);
    if (!ok) {
      this.stepping = true;
      this.pauseVideo();
      if (this.now() - this.lastSeek >= STEP_MS && gap > PAUSED_DRIFT && !v.seeking) this.seek(this.target);
      this.checkAligned();
      return;
    }
    // behind with no faster rate left to close the gap (a nudge above the cap falls back to the
    // speed itself): a catch-up player seeks a little ahead, as the seek takes a moment
    const cannotHurry = drift < 0 && s * (1 + RATE_NUDGE) > Math.min(MAX_RATE, this.maxRate) + 1e-9;
    if (gap > tolerance && !inFlight(tolerance)) this.seek(this.target);
    else if (this.catchUp && cannotHurry && gap > tolerance / 2 && !v.seeking && this.now() - this.lastSeek >= CATCH_UP_MS) {
      this.seek(this.target + s * SEEK_LEAD);
    }
    if (v.paused && !v.ended) this.playVideo();
    this.checkAligned();
  }
}

// drawing

/** what an untagged body is called on its tile: `track 1836`, else `unknown 2` (its person_id) */
export function untaggedLabel(p) {
  if (p && finite(p.tr)) return `track ${p.tr}`;
  const id = p && p.id != null ? String(p.id).replace(/_/g, ' ').trim() : '';
  return id || 'no badge';
}

/**
 * what a pupil's body is called on its tile: `Tag 2`, or `Tag 2? 23 s` when the camera did not read
 * the badge in this frame set and its tracker kept the tag from a read of the track 23 s before
 * (`age`, set by the live model's TagMemory)
 */
export function taggedLabel(p, tagLabel = (t) => `Tag ${t}`) {
  const name = tagLabel(String(p.tag));
  return finite(p.age) ? `${name}? ${Math.round(p.age)} s` : name;
}

/** the hover text of a body without a counted badge: an untagged one, or one whose kept tag ran out */
export function untaggedNote(p) {
  const limit = `${TAG_MEMORY_SECONDS} s`;
  if (p && p.xt != null) {
    return `The camera's tracker still carries Tag ${p.xt} on this person, but their track has not read that badge in the last ${limit} (as far back as this page has loaded), so no tag's measures include them, as on the Analysis page.`;
  }
  // a session the VFA server did not track has no track to carry a read, here or in the Analysis
  if (!p || !finite(p.tr)) return 'The camera read no badge on this person in this frame set, so no tag\'s measures include them.';
  return `No badge was read on this person's track in the last ${limit}, so no tag's live measures include them. The Analysis can still count these frames for a pupil when the same track reads that pupil's badge within ${limit} before or after.`;
}

/** the hover text of a pupil whose tag the camera kept on the track without reading it */
export function keptNote(p, tagLabel = (t) => `Tag ${t}`) {
  const name = tagLabel(String(p.tag));
  return `Not read in this frame set: the camera's tracker kept ${name} on this person since their track last read the badge, ${Math.round(p.age)} s ago. The measures count it for up to ${TAG_MEMORY_SECONDS} s after a read, as the Analysis does; the dashed lines mark it.`;
}

/** the COCO-17 limbs and joints of `kp` scored at least KEYPOINT_MIN_CONF (null keypoints skipped); the number of limbs drawn */
function drawSkeleton(ctx, kp, X, Y, { color, lineWidth = 2, radius = 2, dash = null }) {
  const sure = (pt) => Array.isArray(pt) && finite(pt[0]) && finite(pt[1]) && pt[2] >= KEYPOINT_MIN_CONF;
  let limbs = 0;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (const [i, j] of COCO_LIMBS) {
    const a = kp[i];
    const b = kp[j];
    if (!sure(a) || !sure(b)) continue;
    ctx.moveTo(X(a[0]), Y(a[1]));
    ctx.lineTo(X(b[0]), Y(b[1]));
    limbs += 1;
  }
  if (dash) ctx.setLineDash(dash);
  ctx.stroke();
  if (dash) ctx.setLineDash([]);
  ctx.fillStyle = color;
  for (const pt of kp) {
    if (!sure(pt)) continue;
    ctx.beginPath();
    ctx.arc(X(pt[0]), Y(pt[1]), radius, 0, Math.PI * 2);
    ctx.fill();
  }
  return limbs;
}

/**
 * a filled label with its text's baseline at (x, y); with `placed` (the rectangles of the labels drawn
 * before it), moved down until it covers none of them (a few steps at most)
 */
function drawLabel(ctx, text, x, y, { color, size = 12, font, placed = null }) {
  ctx.font = font(size);
  const tw = ctx.measureText(text).width;
  const rect = () => [x, y - size - 1, x + tw + 10, y + 3];
  if (placed) {
    const hits = (r) => placed.some((q) => r[0] < q[2] && q[0] < r[2] && r[1] < q[3] && q[1] < r[3]);
    for (let i = 0; i < 6 && hits(rect()); i += 1) y += size + 5;
    placed.push(rect());
  }
  ctx.fillStyle = color;
  ctx.fillRect(x, y - size - 1, tw + 10, size + 4);
  ctx.fillStyle = textOn(color);
  ctx.fillText(text, x + 5, y);
}

/**
 * Draw one camera's frame set (slim VFA camera {id, w, h, tg, ps, pr}) into a canvas context whose
 * user space is css pixels `width` x `height`. Pupils (tagged persons) in their tag colours with
 * skeleton, label and gaze ray, a tag the camera only kept on the track (`age`) with dashed box and
 * skeleton and a label like `Tag 2? 23 s`; people without a counted badge (untagged bodies, and those
 * whose kept tag ran out) in grey, with box, skeleton and their track as the label; AprilTag centres
 * as small diamonds. Returns the hit regions of the persons for the hover readout (the untagged ones
 * with `untagged: true`, the kept ones with `kept`, the seconds since the read). `box` ({x, y, w, h}
 * in css pixels) is where a video shows the camera's picture: the frame set's w x h is stretched onto
 * it; without one the frame set is fitted into the canvas. `stats`, given, is filled with what was
 * drawn: {tagged, kept, untagged, untaggedSkeletons}. `turn` draws it turned 180° (x -> w - x,
 * y -> h - y), over a picture the tile shows turned: boxes, labels and hit regions keep their corners
 * in order on the canvas.
 */
export function drawCamera(ctx, cam, { width, height, resolve, tagColor, tagLabel = (t) => `Tag ${t}`, opacity = 1, background = null, box = null, stats = null, turn = false } = {}) {
  const regions = [];
  if (stats) Object.assign(stats, { tagged: 0, kept: 0, untagged: 0, untaggedSkeletons: 0 });
  ctx.save();
  ctx.clearRect(0, 0, width, height);
  if (background) {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);
  }
  if (!cam || !(cam.w > 0) || !(cam.h > 0)) {
    ctx.restore();
    return regions;
  }
  const fit = box && box.w > 0 && box.h > 0 ? box : containBox(cam.w, cam.h, width, height);
  const kx = fit.w / cam.w;
  const ky = fit.h / cam.h;
  const X = turn ? (x) => fit.x + (cam.w - x) * kx : (x) => fit.x + x * kx;
  const Y = turn ? (y) => fit.y + (cam.h - y) * ky : (y) => fit.y + y * ky;
  // a box [x0, y0, x1, y1] on the canvas, left top first however the picture is turned
  const rect = (b) => {
    const xa = X(b[0]);
    const xb = X(b[2]);
    const ya = Y(b[1]);
    const yb = Y(b[3]);
    return [Math.min(xa, xb), Math.min(ya, yb), Math.max(xa, xb), Math.max(ya, yb)];
  };
  ctx.globalAlpha = opacity;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const grey = resolve('var(--tag-other)');
  const surface = resolve('var(--surface)');
  const font = (size, weight = 600) => `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;

  // untagged bodies first, under the pupils: grey box, skeleton and track, so a camera that read no
  // badge still shows whom it detected
  const untagged = (cam.ps || []).filter((p) => p && p.tag == null && (p.b || Array.isArray(p.k)));
  for (const p of untagged) {
    if (p.b) {
      const r = rect(p.b);
      ctx.strokeStyle = grey;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = opacity * 0.85;
      ctx.strokeRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
      ctx.globalAlpha = opacity;
    }
    const limbs = Array.isArray(p.k) ? drawSkeleton(ctx, p.k, X, Y, { color: grey, lineWidth: 1.5, radius: 1.5 }) : 0;
    if (stats) {
      stats.untagged += 1;
      if (limbs) stats.untaggedSkeletons += 1;
    }
    regions.push({ untagged: true, tag: null, label: untaggedLabel(p), note: untaggedNote(p), box: p.b ? rect(p.b) : null, ray: null });
  }
  // their labels go over every grey skeleton, still under the pupils, and clear of each other
  const placed = [];
  for (const p of untagged) {
    if (!p.b) continue;
    const r = rect(p.b);
    drawLabel(ctx, untaggedLabel(p), r[0], Math.max(14, r[1] - 4), { color: grey, size: 11, font, placed });
  }
  ctx.globalAlpha = opacity;

  for (const p of cam.ps || []) {
    if (!p || p.tag == null) continue;
    const tag = String(p.tag);
    const color = resolve(tagColor(tag));
    // a tag the tracker kept without reading it: dashed, so it reads as a guess
    const kept = finite(p.age);
    const dash = kept ? [5, 4] : null;
    if (stats) {
      stats.tagged += 1;
      if (kept) stats.kept += 1;
    }
    const pr = p.b ? rect(p.b) : null;
    if (pr) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = opacity * (kept ? 0.8 : 0.55);
      if (dash) ctx.setLineDash(dash);
      ctx.strokeRect(pr[0], pr[1], pr[2] - pr[0], pr[3] - pr[1]);
      if (dash) ctx.setLineDash([]);
      ctx.globalAlpha = opacity;
    }
    const kp = Array.isArray(p.k) ? p.k : null;
    if (kp) drawSkeleton(ctx, kp, X, Y, { color, lineWidth: 2, radius: 2, dash });
    // gaze ray from the face box centre (else the nose, else the box top) to the gaze point
    let ray = null;
    const g = p.g;
    if (g && Array.isArray(g.p) && finite(g.p[0]) && finite(g.p[1])) {
      let from = null;
      if (Array.isArray(p.fb)) from = [(p.fb[0] + p.fb[2]) / 2, (p.fb[1] + p.fb[3]) / 2];
      else if (kp && kp[0] && kp[0][2] >= KEYPOINT_MIN_CONF) from = [kp[0][0], kp[0][1]];
      else if (p.b) from = [(p.b[0] + p.b[2]) / 2, p.b[1]];
      if (from) {
        ray = [X(from[0]), Y(from[1]), X(g.p[0]), Y(g.p[1])];
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(ray[0], ray[1]);
        ctx.lineTo(ray[2], ray[3]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(ray[2], ray[3], 4, 0, Math.PI * 2);
        ctx.fillStyle = surface;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    // label above the box
    const lx = pr ? pr[0] : ray ? ray[0] : 0;
    const ly = pr ? Math.max(16, pr[1] - 4) : ray ? ray[1] - 8 : 16;
    drawLabel(ctx, taggedLabel(p, tagLabel), lx, ly, { color, size: 12, font });
    regions.push({
      tag,
      kept: kept ? p.age : null,
      label: taggedLabel(p, tagLabel),
      note: kept ? keptNote(p, tagLabel) : null,
      box: pr,
      ray,
      cat: g ? g.cat : null,
      to: g ? g.to : null,
      yaw: p.yaw,
    });
  }

  // AprilTag centres
  for (const [tag, pt] of Object.entries(cam.tg || {})) {
    if (!Array.isArray(pt)) continue;
    const x = X(pt[0]);
    const y = Y(pt[1]);
    ctx.fillStyle = resolve(tagColor(tag));
    ctx.strokeStyle = surface;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x + 5, y);
    ctx.lineTo(x, y + 5);
    ctx.lineTo(x - 5, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
  return regions;
}

function textOn(hex) {
  const m = /^#([0-9a-f]{6})$/i.exec(String(hex || '').trim());
  if (!m) return '#0b0b0b';
  const n = parseInt(m[1], 16);
  const lin = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => {
    const x = c / 255;
    return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  });
  const L = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
  return 1.05 / (L + 0.05) >= (L + 0.05) / 0.0548 ? '#ffffff' : '#0b0b0b';
}

function segDistance(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/**
 * the person under (x, y): the nearest gaze ray within 8 px, else a tagged person's box holding the
 * point, else an untagged one's.
 */
export function hitRegion(regions, x, y) {
  let best = null;
  let bd = 8;
  for (const r of regions || []) {
    if (!r.ray) continue;
    const d = segDistance(x, y, ...r.ray);
    if (d <= bd) {
      bd = d;
      best = r;
    }
  }
  if (best) return best;
  const inside = (r) => r.box && x >= r.box[0] && x <= r.box[2] && y >= r.box[1] && y <= r.box[3];
  return (regions || []).find((r) => !r.untagged && inside(r)) || (regions || []).find((r) => r.untagged && inside(r)) || null;
}

// tiles

/** "c920-05 · VFA", "c920-01 · IPS", "c920-04 · VFA + IPS" */
export function itemLabel(item) {
  const what = item.vfa != null && item.ips != null ? 'VFA + IPS' : item.vfa != null ? 'VFA' : 'IPS';
  return `${item.key} · ${what}`;
}

/**
 * one tile's camera: {key, label, vfa, ips, paths, rotate} of a video item of the session meta
 * (`vfa` the VFA base id that names the camera in the frame sets, `ips` the IPS base's, `paths` its
 * stream paths, `rotate` how the capture turned the picture), or of a bare id (a VFA camera of the
 * frame sets that the session's document does not name); null for anything else.
 */
export function tileItem(x) {
  if (typeof x === 'string' || typeof x === 'number') {
    const id = String(x);
    return id ? { key: id, label: `${id} · VFA`, vfa: id, ips: null, paths: [], rotate: 0 } : null;
  }
  if (!x || x.key == null || String(x.key) === '') return null;
  const item = {
    key: String(x.key),
    label: null,
    vfa: x.vfa != null && x.vfa !== '' ? String(x.vfa) : null,
    ips: x.ips != null && x.ips !== '' ? String(x.ips) : null,
    paths: Array.isArray(x.paths) ? x.paths.filter((v) => typeof v === 'string' && v).map(String) : [],
    rotate: [90, 180, 270].includes(Number(x.rotate)) ? Number(x.rotate) : 0,
  };
  if (item.vfa == null && item.ips == null) item.vfa = item.key;
  item.label = typeof x.label === 'string' && x.label ? x.label : itemLabel(item);
  return item;
}

/**
 * the tiles' cameras: the session's video items (meta.video: VFA cameras first, then the IPS ones),
 * with the VFA cameras of the frame sets (`vfaIds`) that no item names placed after the named VFA
 * ones; an id that is an IPS-only item's key joins that item.
 */
export function cameraItems(video, vfaIds) {
  const items = [];
  const byKey = new Map();
  for (const raw of Array.isArray(video) ? video : []) {
    const item = tileItem(raw && typeof raw === 'object' ? raw : null);
    if (!item || byKey.has(item.key)) continue;
    byKey.set(item.key, item);
    items.push(item);
  }
  const named = new Set(items.filter((it) => it.vfa != null).map((it) => it.vfa));
  const extra = [];
  for (const raw of vfaIds || []) {
    const id = String(raw);
    if (!id || named.has(id)) continue;
    named.add(id);
    const same = byKey.get(id);
    if (same) {
      if (same.vfa == null) {
        same.vfa = id;
        same.label = itemLabel(same);
      }
      continue;
    }
    const item = tileItem(id);
    byKey.set(id, item);
    extra.push(item);
  }
  return [...items.filter((it) => it.vfa != null), ...extra, ...items.filter((it) => it.vfa == null)];
}

/** the ready live stream of a tile's camera: by its stream paths, else its stream name, else (a session whose document names no paths) by base id */
export function tileStream(streams, item) {
  const list = (streams || []).filter((x) => x && x.kind === 'video' && x.ready);
  const paths = item.paths || [];
  const byPath = list.find((x) => paths.includes(String(x.path)));
  if (byPath) return byPath;
  const byName = list.find((x) => x.stream != null && String(x.stream) === item.key);
  if (byName || paths.length) return byName || null;
  const ids = [item.vfa, item.ips, item.key].filter((v) => v != null);
  return list.find((x) => ids.includes(String(x.camera)) || ids.includes(String(x.base_id))) || null;
}

/**
 * whether a recordings route file is of a tile's camera: an archived stream cut by its stream path
 * (or, when the tile knows no paths, its stream name or camera), a capture file by its device, the
 * tile's key or one of its base ids
 */
export function fileOfTile(f, item) {
  if (!f) return false;
  const ids = [item.key, item.vfa, item.ips].filter((v) => v != null);
  const paths = item.paths || [];
  if (f.source === 'stream') {
    if (f.stream_path && paths.length) return paths.includes(String(f.stream_path));
    return String(f.device) === item.key || (f.camera != null && ids.includes(String(f.camera)));
  }
  return f.device != null && ids.includes(String(f.device));
}

/**
 * whether Sync overlay holds this tile's live video back: only a tile that draws VFA frame sets has an
 * overlay to wait for, so an IPS camera's video plays as it comes
 */
export function holdsBack(tile) {
  return !!tile && tile.vfa != null;
}

const TURN_KEY = 'openmmla.dashboard.turn180';
// a camera whose video the viewer hid on this machine (a slow one need not decode every camera)
const HIDE_KEY = 'openmmla.dashboard.hidevideo';

/** whether the viewer turned this camera's tile of this session 180° (kept in this browser only) */
export function readTurn(sid, key, storage) {
  try {
    // the page's storage is looked up inside the try: a browser that blocks site data throws on the lookup
    const s = storage === undefined ? globalThis.localStorage : storage;
    return !!s && s.getItem(`${TURN_KEY}.${sid}.${key}`) === '1';
  } catch {
    return false;
  }
}

export function writeTurn(sid, key, on, storage) {
  try {
    const s = storage === undefined ? globalThis.localStorage : storage;
    if (!s) return;
    if (on) s.setItem(`${TURN_KEY}.${sid}.${key}`, '1');
    else s.removeItem(`${TURN_KEY}.${sid}.${key}`);
  } catch {
    // private windows and blocked storage keep the turn for this page only
  }
}

/** whether the viewer hid this camera's video of this session (kept in this browser only) */
export function readHidden(sid, key, storage) {
  try {
    const s = storage === undefined ? globalThis.localStorage : storage;
    return !!s && s.getItem(`${HIDE_KEY}.${sid}.${key}`) === '1';
  } catch {
    return false;
  }
}

export function writeHidden(sid, key, on, storage) {
  try {
    const s = storage === undefined ? globalThis.localStorage : storage;
    if (!s) return;
    if (on) s.setItem(`${HIDE_KEY}.${sid}.${key}`, '1');
    else s.removeItem(`${HIDE_KEY}.${sid}.${key}`);
  } catch {
    // private windows and blocked storage keep it hidden for this page only
  }
}

/**
 * The tiles of the Cameras card. opts: {tagColor(tag), tagLabel(tag), onNote(text)}.
 * Returns {el, update(state), setExpanded(bool), setOptions({overlay, sync}), hasVideo(), hasLiveVideo(),
 * inspect(), suspend(), resume(), destroy()}. suspend() closes every video but keeps the tiles (a page
 * going into the back/forward cache); resume() lets them play again.
 * update(state): {cameras: [items (cameraItems) or VFA ids], sid (the session, whose tiles the viewer may
 *                 have turned 180°), frameAt(t) -> vfa record, now (stream clock), mode ('follow'|'replay'),
 *                 live (the session runs), media (the /media answer | null), serverNow (epoch),
 *                 vfaLag (s | null), recordings ({enabled, files, reason} of the recordings route | null
 *                 while it is asked), speed (replay speed), running (the replay clock advances),
 *                 mediaOrigins (where the files load from: the dashboard's media origins that answered, in
 *                 its order, the first carrying the sound; empty: the page's origin; a single mediaOrigin
 *                 does too), mediaReserved (how many of the MAX_MEDIA connections the page's sound takes,
 *                 0 or 1; it counts only without a media origin)}.
 * A tile is a camera (its key the stream's name, else the base id): a VFA camera's tile draws the
 * frame sets of its VFA id over the video, an IPS camera's shows the video alone (its badges are on
 * the Room card). The replay of an ended session plays each camera's recorded file (fileOfTile: an
 * archived stream cut of its stream path, a capture file of its device) at the clock; a tile whose
 * recording does not cover the moment draws the skeletons alone. A tile of a camera whose capture
 * did not turn its picture (rotate 0) offers Turn 180°, for a camera mounted upside down before the
 * capture turned pictures: the video and the overlay turn together, remembered per session and
 * camera in this browser. The
 * tiles on screen hold a file first, then those scrolled away (paused, the most recently seen first),
 * up to the caps of mediaPools() (MAX_MEDIA_VIDEOS on the first media origin, MAX_ORIGIN_FILES on each
 * other one; without one, MAX_MEDIA less the sound's share on the page's origin), each tile on the
 * origin it holds its file from (assignFiles); a tile past that lets go of its file.
 */
export function cameraWall({ tagColor, tagLabel = (t) => `Tag ${t}` } = {}) {
  const el = h('div', { class: 'cam-wall' });
  const tiles = new Map();
  const st = {
    expanded: false,
    docVisible: typeof document === 'undefined' || document.visibilityState !== 'hidden',
    overlay: true,
    sync: true,
    last: null,
    priority: [],
    alive: true,
    suspended: false,
    // the session whose turned tiles are remembered
    sid: null,
    // where the recorded files load from (the media origins that answered; none: the page's origin),
    // the connections the sound takes on the page's origin, and the recorded files the tiles may hold
    // besides, on all origins together
    origins: [],
    reserved: 0,
    fileCap: MAX_MEDIA,
  };
  const clockMs = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const io = typeof IntersectionObserver !== 'undefined'
    ? new IntersectionObserver((entries) => {
      for (const en of entries) {
        const tile = Array.from(tiles.values()).find((t) => t.el === en.target);
        if (!tile) continue;
        // when it was last on screen: a tile scrolled away longest lets go of its file first
        if (tile.onScreen || en.isIntersecting) tile.seenAt = clockMs();
        tile.onScreen = en.isIntersecting;
      }
      schedulePlayback();
    }, { threshold: 0.01 })
    : null;
  const onVisibility = () => {
    st.docVisible = document.visibilityState !== 'hidden';
    schedulePlayback();
  };
  if (typeof document !== 'undefined') document.addEventListener('visibilitychange', onVisibility);
  // a redraw that moves no video: the clock of the last update is a moment old
  const offTheme = theme.onChange(() => {
    for (const t of tiles.values()) t.drawnKey = null;
    paintAll();
  });
  const ro = typeof ResizeObserver !== 'undefined'
    ? new ResizeObserver(() => {
      for (const t of tiles.values()) t.drawnKey = null;
      paintAll();
    })
    : null;

  /** why a tile draws skeletons instead of video (an IPS camera's tile: why it shows nothing) */
  function skeletonNote(t, s) {
    const lead = t.vfa == null ? 'No video' : 'Skeleton view';
    const tail = t.vfa == null ? ' This IPS camera\'s badge positions are on the Room card.' : '';
    if (s.mode !== 'follow') {
      if (s.live) return `${lead}: recorded video plays once the session has ended.${tail}`;
      const rec = s.recordings;
      if (!rec) return `${lead}: looking for the camera recordings.${tail}`;
      if (!rec.enabled) return rec.reason ? `${lead}. ${rec.reason}${tail}` : `${lead}: the recordings are not available.${tail}`;
      const unplaced = (rec.files || []).some((f) => f && f.modality === 'video' && fileOfTile(f, t) && !finite(f.start));
      if (unplaced) return `${lead}: the recording of this camera has no start time, so it cannot follow the clock.${tail}`;
      return `${lead}: the dashboard's machine holds no recording of this camera.${tail}`;
    }
    const media = s.media;
    if (!media) return `${lead}: this session has no live video.${tail}`;
    const others = (media.streams || []).some((x) => x && x.kind === 'video' && x.ready);
    if (media.webrtc && others) return `${lead}: this camera has no live stream.${tail}`;
    return media.reason ? `${lead}. ${media.reason}${tail}` : `${lead}: this session has no live video.${tail}`;
  }

  function streamFor(t, s) {
    const media = s && s.media;
    if (!media || !media.webrtc || s.mode !== 'follow') return null;
    return tileStream(media.streams, t);
  }

  /** the camera's recorded video files, in the replay of an ended session */
  function recordingsFor(t, s) {
    const rec = s && s.recordings;
    if (!rec || !rec.enabled || s.mode !== 'replay' || s.live) return [];
    return (rec.files || []).filter((f) => f && f.modality === 'video' && fileOfTile(f, t) && finite(f.start) && f.url);
  }

  /** show a tile turned 180° (the video by CSS, the overlay by drawCamera) or upright */
  function applyTurn(t) {
    const turned = t.rotate === 0 && t.turned;
    t.stage.classList.toggle('is-turned', turned);
    t.turnBtn.hidden = t.rotate !== 0;
    t.turnBtn.setAttribute('aria-pressed', turned ? 'true' : 'false');
    t.drawnKey = null;
  }

  /**
   * show a tile's video or hide it: a hidden one plays nothing (its connection and its decoding go
   * to the others, and a camera waiting for a place plays), a VFA camera keeps its skeletons and an
   * IPS camera, which has none, folds to its heading
   */
  function applyHidden(t) {
    t.el.classList.toggle('is-video-hidden', t.hidden);
    t.el.classList.toggle('is-video-only', t.vfa == null);
    t.hideBtn.setAttribute('aria-pressed', t.hidden ? 'true' : 'false');
    t.hideLabel.textContent = t.hidden ? 'Show video' : 'Hide video';
    t.drawnKey = null;
  }

  /** a tile's camera changed what the session says of it: its name, base ids, paths and turn */
  function setItem(t, item) {
    const same = t.label === item.label && t.vfa === item.vfa && t.ips === item.ips && t.rotate === item.rotate
      && t.paths.join('\n') === item.paths.join('\n');
    if (same) return;
    Object.assign(t, { label: item.label, vfa: item.vfa, ips: item.ips, paths: item.paths.slice(), rotate: item.rotate });
    t.name.textContent = item.label;
    // a narrow tile cuts the name short: the whole of it in the tooltip
    t.name.title = item.label;
    t.canvas.setAttribute('aria-label', item.vfa == null
      ? `Camera ${item.label}: video only`
      : `Camera ${item.label}: skeletons of the newest frame set`);
    t.age.hidden = item.vfa == null;
    applyTurn(t);
    applyHidden(t);
  }

  function makeTile(item) {
    const id = item.key;
    const name = h('span', { class: 'cam-name truncate', text: item.label });
    const age = h('span', { class: 'badge cam-age num', text: fmt.na, title: 'Age of the frame set drawn' });
    const stateEl = h('span', { class: 'cam-state muted' });
    const playBtn = h('button', { class: 'btn sm', attrs: { type: 'button' }, hidden: true }, icon('play', 12), h('span', { text: 'Play' }));
    const turnBtn = h('button', {
      class: 'btn sm',
      attrs: { type: 'button', 'aria-pressed': 'false', title: 'Turn the video and the overlay 180° (a camera mounted upside down, recorded before the capture turned its pictures)' },
    }, h('span', { text: 'Turn 180°' }));
    const hideLabel = h('span', { text: 'Hide video' });
    const hideBtn = h('button', {
      class: 'btn sm',
      attrs: { type: 'button', 'aria-pressed': 'false', title: 'Hide this camera\'s video on this machine, or show it again: every camera plays until you hide one' },
    }, hideLabel);
    const head = h('div', { class: 'cam-head' }, name, age, stateEl, h('span', { class: 'spacer' }), playBtn, hideBtn, turnBtn);
    const video = h('video', { class: 'cam-video', attrs: { muted: true, playsinline: true, autoplay: true }, muted: true, hidden: true });
    // the recorded file of a replay: hidden (not display: none, so it keeps loading) until it has a frame
    const fvideo = h('video', { class: 'cam-video cam-file', attrs: { muted: true, playsinline: true, preload: 'auto', disablepictureinpicture: true }, muted: true, style: { visibility: 'hidden' } });
    const canvas = h('canvas', { class: 'cam-canvas', attrs: { role: 'img', 'aria-label': `Camera ${id}: skeletons of the newest frame set` } });
    const stage = h('div', { class: 'cam-stage' }, video, fvideo, canvas);
    const note = h('p', { class: 'cam-note' });
    const tileEl = h('div', { class: 'cam-tile', dataset: { camera: id } }, head, stage, note);
    const tile = {
      id, el: tileEl, name, age, stateEl, playBtn, hideBtn, hideLabel, turnBtn, video, fvideo, canvas, stage, note,
      label: null, vfa: null, ips: null, paths: [], rotate: 0, turned: !!st.sid && readTurn(st.sid, id), turnSid: st.sid,
      hidden: !!st.sid && readHidden(st.sid, id),
      onScreen: !io, seenAt: 0, player: null, playerState: null, stream: null, regions: [], drawnKey: null, aspect: 16 / 9,
      videoMode: false, kind: null, files: [], file: null, next: null, onVideo: false, drawn: null,
      // the origin its recorded file loads from while it holds one (null: the page's), else undefined
      fileOrigin: undefined,
    };
    playBtn.addEventListener('click', () => {
      st.priority = [id, ...st.priority.filter((x) => x !== id)];
      schedulePlayback();
    });
    hideBtn.addEventListener('click', () => {
      tile.hidden = !tile.hidden;
      if (st.sid) writeHidden(st.sid, id, tile.hidden);
      applyHidden(tile);
      schedulePlayback();
      renderTileState(tile);
      if (st.expanded && st.last) paint(tile, st.last);
    });
    turnBtn.addEventListener('click', () => {
      tile.turned = !tile.turned;
      if (st.sid) writeTurn(st.sid, id, tile.turned);
      applyTurn(tile);
      if (st.expanded && st.last) paint(tile, st.last);
    });
    stage.addEventListener('pointermove', (e) => {
      const r = canvas.getBoundingClientRect();
      const hit = hitRegion(tile.regions, e.clientX - r.left, e.clientY - r.top);
      if (!hit) {
        tooltip.hide();
        return;
      }
      if (hit.untagged) {
        tooltip.show(e, { title: hit.label, note: hit.note });
        return;
      }
      const cat = hit.cat ? gazeCategory(hit.cat, hit.to) : null;
      tooltip.show(e, {
        title: hit.kept != null ? `${tagLabel(hit.tag)}?` : tagLabel(hit.tag),
        note: hit.note || undefined,
        rows: [
          { value: cat ? CATEGORY_LABELS[cat] : fmt.na, label: 'gaze', color: tagColor(hit.tag), key: 'line' },
          hit.to != null ? { value: hit.to === 'other' ? 'someone else' : tagLabel(hit.to), label: 'target', key: 'none' } : null,
          hit.cat && cat && hit.cat.replace(/_/g, ' ') !== CATEGORY_LABELS[cat].toLowerCase()
            ? { value: hit.cat.replace(/_/g, ' '), label: 'as the server named it', key: 'none' } : null,
          finite(hit.yaw) ? { value: `${Math.round(hit.yaw)}°`, label: 'head yaw', key: 'none' } : null,
        ].filter(Boolean),
      });
    });
    stage.addEventListener('pointerleave', () => tooltip.hide());
    if (io) io.observe(tileEl);
    if (ro) ro.observe(stage);
    setItem(tile, item);
    applyTurn(tile);
    return tile;
  }

  function syncTiles(list) {
    const items = [];
    for (const raw of list) {
      const item = tileItem(raw);
      if (item && !items.some((it) => it.key === item.key)) items.push(item);
    }
    const want = items.map((it) => it.key);
    let changed = false;
    for (const [id, t] of tiles) {
      if (!want.includes(id)) {
        if (t.player) t.player.stop();
        if (io) io.unobserve(t.el);
        if (ro) ro.unobserve(t.stage);
        t.el.remove();
        tiles.delete(id);
        changed = true;
      }
    }
    for (const item of items) {
      const t = tiles.get(item.key);
      if (!t) {
        tiles.set(item.key, makeTile(item));
        changed = true;
        continue;
      }
      setItem(t, item);
      // the session the page shows came later than the tile
      if (t.turnSid !== st.sid) {
        t.turnSid = st.sid;
        t.turned = !!st.sid && readTurn(st.sid, t.id);
        t.hidden = !!st.sid && readHidden(st.sid, t.id);
        applyTurn(t);
        applyHidden(t);
      }
    }
    if (changed) {
      clear(el);
      for (const id of want) el.appendChild(tiles.get(id).el);
    }
  }

  function schedulePlayback() {
    if (!st.alive) return;
    const open = st.expanded && st.docVisible && !st.suspended;
    const canPlay = (t) => open && t.onScreen && !t.hidden;
    const now = clockMs();
    for (const t of tiles.values()) if (canPlay(t)) t.seenAt = now;
    const eligible = [];
    for (const t of tiles.values()) if (t.videoMode) eligible.push(t.id);
    const order = [...st.priority.filter((id) => eligible.includes(id)), ...eligible.filter((id) => !st.priority.includes(id))];
    // live video comes over WebRTC from the stream server: up to MAX_PLAYING tiles on screen
    const liveOn = order.filter((id) => tiles.get(id).kind === 'live' && canPlay(tiles.get(id)));
    const live = new Set(liveOn.slice(0, MAX_PLAYING));
    // a recorded file holds a connection to the dashboard: the tiles on screen first, then the ones
    // scrolled away that still hold theirs (paused, kept in place), the most recently seen first, on
    // the media origins each up to its cap (assignFiles)
    const fileOn = order.filter((id) => tiles.get(id).kind === 'file' && canPlay(tiles.get(id)));
    const away = open ? order
      .filter((id) => {
        const t = tiles.get(id);
        return t.kind === 'file' && !t.hidden && !canPlay(t) && t.player instanceof FilePlayer && !!t.player.url;
      })
      .sort((a, b) => tiles.get(b).seenAt - tiles.get(a).seenAt) : [];
    const held = new Map();
    for (const t of tiles.values()) if (t.player instanceof FilePlayer && t.fileOrigin !== undefined) held.set(t.id, t.fileOrigin);
    const pools = mediaPools(st.origins, st.reserved);
    const poolsKey = JSON.stringify(pools);
    if (st.homesKey !== poolsKey) {
      st.homesKey = poolsKey;
      st.homes = new Map();
    }
    const recorded = Array.from(tiles.values()).filter((t) => t.files && t.files.length).map((t) => t.id);
    st.homes = homeOrigins(recorded, pools, st.homes);
    const homes = st.homes;
    const { playing: files, parked, cap } = assignFiles({ playing: fileOn, parked: away, held, pools, homes });
    st.fileCap = cap;
    for (const t of tiles.values()) {
      const fileOrigin = t.kind === 'file' && t.videoMode ? (files.has(t.id) ? files.get(t.id) : parked.get(t.id)) : undefined;
      // a tile that holds no file has no origin; one that holds a file (or is about to) loads it from
      // its own (syncRecording), and a new origin loads the file again from there
      t.fileOrigin = fileOrigin;
      if (!t.videoMode) {
        if (t.player) {
          t.player.stop();
          t.player = null;
        }
        t.playBtn.hidden = true;
        renderTileState(t);
        continue;
      }
      if (fileOrigin !== undefined) {
        if (!(t.player instanceof FilePlayer)) {
          if (t.player) t.player.stop();
          t.playerState = null;
          // the file to load is picked from the clock on the next update
          t.player = new FilePlayer({
            video: t.fvideo,
            onState: (s) => {
              t.playerState = s;
              renderTileState(t);
              if (st.expanded && st.last) paint(t, st.last);
            },
          });
          t.player.start();
        }
        if (parked.has(t.id)) {
          if (!t.player.parked) {
            // off screen: the picture goes at once, so the tile comes back with the skeletons of
            // the moment until its video stands there again
            t.player.park();
            t.drawnKey = null;
            if (st.expanded && st.last) paint(t, st.last);
          }
        } else if (t.player.parked) {
          // back on screen: the seek to the clock starts now, not at the next redraw
          t.player.unpark();
          t.drawnKey = null;
          if (st.expanded && st.last) {
            syncRecording(t, st.last);
            paint(t, st.last);
          }
        }
        t.playBtn.hidden = true;
      } else if (t.kind === 'live' && live.has(t.id)) {
        if (!t.player || t.player instanceof FilePlayer || t.player.url !== t.url) {
          if (t.player) t.player.stop();
          t.player = new WhepPlayer({
            url: t.url,
            video: t.video,
            onState: (s) => {
              t.playerState = s;
              renderTileState(t);
            },
          });
          t.player.start();
        }
        t.playBtn.hidden = true;
      } else {
        if (t.player) {
          t.player.stop();
          t.player = null;
        }
        // a tile that could play but lost to the limit offers Play
        const over = t.kind === 'file' ? fileOn.length > cap : liveOn.length > MAX_PLAYING;
        t.playBtn.hidden = !(canPlay(t) && over);
      }
      renderTileState(t);
    }
  }

  /** what a tile that lost to the limit says */
  function limitText(t) {
    if (t.kind !== 'file') return `Up to ${MAX_PLAYING} cameras play at once`;
    const n = st.fileCap;
    const plays = `${n} ${n === 1 ? 'camera plays' : 'cameras play'}`;
    // from the media origins the sound has a connection of its own
    if (st.reserved && !st.origins.length) return n ? `Up to ${plays} beside the sound` : 'The sound takes the last connection';
    return `Up to ${plays} at once`;
  }

  function renderTileState(t) {
    const ps = t.playerState;
    if (t.hidden) {
      t.stateEl.textContent = 'Video hidden';
      return;
    }
    if (!t.videoMode) {
      t.stateEl.textContent = '';
      return;
    }
    if (!t.player) t.stateEl.textContent = t.playBtn.hidden ? 'Paused' : limitText(t);
    else if (t.player instanceof FilePlayer) {
      const loading = t.file && (!ps || ps.state === 'loading' || ps.state === 'idle');
      t.stateEl.textContent = ps && ps.state === 'error' ? 'Could not play' : loading ? 'Loading the recording' : '';
    } else if (!ps || ps.state === 'connecting') t.stateEl.textContent = 'Connecting';
    else if (ps.state === 'playing') t.stateEl.textContent = '';
    else if (ps.state === 'retrying') t.stateEl.textContent = `${ps.message} Retrying in ${ps.retryIn} s.`;
    else t.stateEl.textContent = ps.message || '';
  }

  function sizeCanvas(t, cam) {
    const aspect = cam && cam.w > 0 && cam.h > 0 ? cam.w / cam.h : t.aspect;
    if (Math.abs(aspect - t.aspect) > 1e-3) {
      t.aspect = aspect;
      t.drawnKey = null;
    }
    t.stage.style.aspectRatio = String(t.aspect);
    const w = Math.max(1, Math.floor(t.stage.clientWidth));
    const hgt = Math.max(1, Math.round(w / t.aspect));
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    const cw = Math.round(w * dpr);
    const ch = Math.round(hgt * dpr);
    if (t.canvas.width !== cw || t.canvas.height !== ch) {
      t.canvas.width = cw;
      t.canvas.height = ch;
      t.drawnKey = null;
    }
    return { w, h: hgt, dpr };
  }

  function render(s) {
    st.last = s;
    st.origins = mediaOriginList(s.mediaOrigins !== undefined ? s.mediaOrigins : s.mediaOrigin);
    st.reserved = finite(s.mediaReserved) ? Math.max(0, Math.min(MAX_MEDIA, Math.round(s.mediaReserved))) : 0;
    if (s.sid != null) st.sid = String(s.sid);
    syncTiles(s.cameras || []);
    for (const t of tiles.values()) {
      const stream = streamFor(t, s);
      const url = stream ? whepUrl(s.media.webrtc, stream.path) : null;
      t.files = stream ? [] : recordingsFor(t, s);
      const kind = stream ? 'live' : t.files.length ? 'file' : null;
      pickRecording(t, s, kind);
      // a recorded camera takes a player (and one of the mediaPools() places) only while a file holds the clock
      const videoMode = kind === 'live' || (kind === 'file' && !!(t.file || t.next));
      if (kind !== t.kind || url !== t.url || videoMode !== t.videoMode) {
        t.kind = kind;
        t.videoMode = videoMode;
        t.url = url;
        t.video.hidden = kind !== 'live';
        t.el.classList.toggle('is-video', videoMode);
        t.drawnKey = null;
      }
    }
    schedulePlayback();
    if (!st.expanded) return;
    for (const t of tiles.values()) {
      if (t.kind === 'file') syncRecording(t, s);
      paint(t, s);
    }
  }

  /** the recording that holds the clock (t.file) or, when none does, the one about to (t.next) */
  function pickRecording(t, s, kind) {
    const file = kind === 'file' ? recordingAt(t.files, s.now) : null;
    const speed = finite(s.speed) && s.speed > 0 ? s.speed : 1;
    t.next = kind === 'file' && !file ? nextRecording(t.files, s.now, RECORDING_LOOKAHEAD * speed) : null;
    if (file !== t.file) t.drawnKey = null;
    t.file = file;
  }

  /**
   * keep the video of the recording at the clock's moment (a coming one loads, paused, from its
   * start); a parked one (off screen) keeps its file and place and follows no clock
   */
  function syncRecording(t, s) {
    const p = t.player instanceof FilePlayer ? t.player : null;
    if (!p) return;
    const pick = t.file || t.next;
    if (pick && !p.parked) p.load(inlineUrl(pick.url, t.fileOrigin), pick.url, pick === t.file ? s.now - pick.start : 0);
    // a parked player only notes the moment
    const target = t.file && p.source === t.file.url ? s.now - t.file.start : null;
    p.sync({ target, speed: s.speed, playing: !!s.running, tolerance: driftTolerance(s.speed) });
  }

  function paintAll() {
    if (!st.expanded || !st.last) return;
    for (const t of tiles.values()) paint(t, st.last);
  }

  /** what a recorded video's tile says under it */
  function recordingNote(t, s) {
    const p = t.player instanceof FilePlayer ? t.player : null;
    if (p && p.state === 'error') return `Skeleton view. ${p.message || 'The recording could not be played.'}`;
    if (!t.file) return 'No recording at this moment.';
    if (!p) return '';
    if (p.stepping) return `This browser cannot play the video at ${s.speed}x: it steps once a second.`;
    if (t.file.source === 'stream') return 'Recorded video: the stream server\'s cut, archived.';
    return t.file.host ? `Recorded video from ${t.file.host}.` : 'Recorded video.';
  }

  /** what an IPS camera's tile adds under its video */
  const IPS_NOTE = 'IPS camera: video only, its badge positions are on the Room card.';

  /** draw one tile: the overlay of a video shows the frame set at the video's own time */
  function paint(t, s) {
    let at = s.now;
    let delayNote = '';
    let picture = null;
    if (t.kind === 'live' && !t.hidden) {
      picture = t.video;
      if (finite(s.serverNow)) {
        const lag = finite(s.vfaLag) ? Math.max(0, s.vfaLag) : 0;
        let applied = 0;
        if (st.sync && holdsBack(t) && t.player && supportsVideoDelay()) {
          applied = Math.min(lag, MAX_VIDEO_DELAY);
          if (t.player.setDelay(applied) && applied >= 0.2) delayNote = `Video delayed ${fmt.num(applied, 1)} s to match the overlay.`;
        } else if (t.player) t.player.setDelay(0);
        at = s.serverNow - applied;
      }
    } else if (t.kind === 'file') {
      const p = t.player instanceof FilePlayer ? t.player : null;
      if (p && t.file && p.source === t.file.url && p.showing && p.mediaTime() != null) {
        picture = t.fvideo;
        at = t.file.start + p.mediaTime();
      }
    }
    // a tile without a picture draws the skeletons alone, on its own background
    const onVideo = !!picture;
    if (onVideo !== t.onVideo) {
      t.onVideo = onVideo;
      t.drawnKey = null;
    }
    t.fvideo.style.visibility = onVideo && t.kind === 'file' ? 'visible' : 'hidden';
    // an IPS camera's tile has no frame sets: the video alone
    const rec = s.frameAt && t.vfa != null ? s.frameAt(at) : null;
    const cam = rec ? (rec.c || []).find((c) => String(c.id) === t.vfa) : null;
    const turned = t.rotate === 0 && t.turned;
    const { w, h: hgt, dpr } = sizeCanvas(t, cam);
    // object-fit: contain letterboxes a picture whose shape differs from the frame set's
    const box = picture ? containBox(picture.videoWidth, picture.videoHeight, w, hgt) : null;
    const showOverlay = !onVideo || st.overlay;
    const boxKey = box ? `${box.x.toFixed(1)},${box.y.toFixed(1)},${box.w.toFixed(1)},${box.h.toFixed(1)}` : '';
    const key = `${rec ? rec.t : 'none'}|${w}x${hgt}|${showOverlay}|${t.kind}|${onVideo}|${boxKey}|${turned}`;
    if (key !== t.drawnKey) {
      t.drawnKey = key;
      const resolve = colorResolver(el);
      const ctx = t.canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      t.drawn = { tagged: 0, untagged: 0, untaggedSkeletons: 0 };
      if (!showOverlay) {
        ctx.clearRect(0, 0, w, hgt);
        t.regions = [];
      } else {
        t.regions = drawCamera(ctx, cam, {
          width: w, height: hgt, resolve, tagColor, tagLabel, box, turn: turned,
          opacity: onVideo ? 0.8 : 1,
          background: onVideo ? null : resolve('var(--surface-2)'),
          stats: t.drawn,
        });
        if (!cam && !onVideo) {
          ctx.fillStyle = resolve('var(--muted)');
          ctx.font = '500 13px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
          ctx.textAlign = 'center';
          const empty = t.vfa == null ? 'No video of this IPS camera now.' : rec ? 'This camera is missing from the newest frame set.' : 'No frame set yet.';
          ctx.fillText(empty, w / 2, hgt / 2);
          ctx.textAlign = 'start';
        }
      }
    }
    const ageS = rec ? (finite(s.serverNow) && s.mode === 'follow' ? s.serverNow - rec.t : at - rec.t) : null;
    t.age.textContent = ageS == null ? 'no frame' : ageS < 1.5 ? 'now' : fmt.ago(ageS);
    t.age.dataset.stale = ageS != null && ageS > 10 ? 'true' : 'false';
    const ips = t.vfa == null && t.kind ? ` ${IPS_NOTE}` : '';
    if (t.hidden) t.note.textContent = t.vfa == null ? '' : 'Video hidden on this machine: the skeletons of the frame set.';
    else if (t.kind === 'live') t.note.textContent = `${delayNote}${ips}`.trim();
    else if (t.kind === 'file') t.note.textContent = `${recordingNote(t, s)}${ips}`.trim();
    else t.note.textContent = skeletonNote(t, s);
  }

  return {
    el,
    update(s) {
      if (!st.alive) return;
      render(s || {});
    },
    setExpanded(on) {
      st.expanded = !!on;
      for (const t of tiles.values()) t.drawnKey = null;
      schedulePlayback();
      if (st.last && st.expanded) render(st.last);
    },
    setOptions({ overlay, sync } = {}) {
      if (overlay != null) st.overlay = !!overlay;
      if (sync != null) st.sync = !!sync;
      for (const t of tiles.values()) t.drawnKey = null;
      paintAll();
    },
    hasVideo() {
      return Array.from(tiles.values()).some((t) => t.videoMode);
    },
    hasLiveVideo() {
      return Array.from(tiles.values()).some((t) => t.kind === 'live' && holdsBack(t));
    },
    /** each tile's video as it stands, for checks from the browser console */
    inspect() {
      return Array.from(tiles.values()).map((t) => {
        const p = t.player instanceof FilePlayer ? t.player : null;
        const v = t.kind === 'file' ? t.fvideo : t.video;
        return {
          id: t.id, kind: t.kind, playing: !!t.player, state: t.playerState ? t.playerState.state : null,
          file: t.file ? t.file.id : null, target: p ? p.target : null, time: v.currentTime, paused: v.paused,
          rate: v.playbackRate, onVideo: t.onVideo, stepping: p ? p.stepping : false,
          parked: p ? p.parked : false, aligned: p ? p.aligned : false, onScreen: t.onScreen,
          origin: t.fileOrigin, src: t.fvideo.getAttribute('src'), visibility: t.fvideo.style.visibility,
          picture: [v.videoWidth, v.videoHeight], stage: [t.stage.clientWidth, t.stage.clientHeight], note: t.note.textContent,
          drawn: t.drawn ? { ...t.drawn } : null,
        };
      });
    },
    playing() {
      return Array.from(tiles.values()).filter((t) => t.player).map((t) => t.id);
    },
    /** how many recorded files the tiles hold (each keeps a connection to the dashboard) */
    mediaHeld() {
      return Array.from(tiles.values()).filter((t) => t.fvideo.getAttribute('src')).length;
    },
    /** each origin the tiles' files may load from, with its cap and the files the tiles hold there */
    mediaUse() {
      return mediaPools(st.origins, st.reserved).map(({ origin, cap }) => ({
        origin, cap, held: Array.from(tiles.values()).filter((t) => t.fileOrigin === origin && t.fvideo.getAttribute('src')).length,
      }));
    },
    suspend() {
      st.suspended = true;
      schedulePlayback();
    },
    resume() {
      st.suspended = false;
      st.docVisible = typeof document === 'undefined' || document.visibilityState !== 'hidden';
      schedulePlayback();
    },
    destroy() {
      st.alive = false;
      for (const t of tiles.values()) if (t.player) t.player.stop();
      if (io) io.disconnect();
      if (ro) ro.disconnect();
      offTheme();
      if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', onVisibility);
      tiles.clear();
      clear(el);
    },
  };
}
