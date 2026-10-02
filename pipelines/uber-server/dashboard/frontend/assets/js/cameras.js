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
 * dashboard's media port, where up to MAX_MEDIA_VIDEOS play beside the replay's sound; without one
 * they load from the page's origin and share MAX_MEDIA with the sound. A tile scrolled off screen
 * keeps its file, paused, while that allows, so scrolling back shows the moment without loading the
 * file again. The WHEP exchange lives
 * in whepNegotiate and WhepPlayer, which take their RTCPeerConnection and fetch as arguments so the
 * offer/answer flow can be tested without a browser or a server; FilePlayer takes its video element
 * and clock the same way.
 */

import { h, clear, fmt, tooltip, theme, icon } from './core.js';
import { colorResolver } from './charts.js';
import { CATEGORY_LABELS, gazeCategory, TAG_MEMORY_SECONDS } from './live-model.js';

export const MAX_PLAYING = 4;
// a browser opens at most six connections per origin (scheme, host and port) over HTTP/1.1, and a
// recorded file (a camera's video or the replay's sound) holds one while it plays, and while it is
// paused in place. The page loads the files from the dashboard's media port, an origin of their
// own (mediaOriginFor): up to MAX_MEDIA_VIDEOS camera videos and the sound beside them, which leaves
// one of the six free there for a seek. The page's own origin keeps its six for the API requests and
// the live stream
export const MAX_MEDIA_VIDEOS = 4;
// without a media port (or when it does not answer) the files load from the page's origin, where
// the live stream takes one connection and one stays free for the page's other requests: the
// videos and the sound share the other four
export const MAX_MEDIA = 4;
// how long the page waits for the media port's answer before it loads the files from its own origin
export const MEDIA_PROBE_MS = 3000;
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
 * One WHEP exchange on `pc`: receive-only video and audio transceivers, an offer with its ICE
 * candidates (gathering waited for up to `iceTimeout` ms), POST it as application/sdp, expect 201
 * with the answer, set it. Returns {location} (the session URL for DELETE, or null).
 */
export async function whepNegotiate(pc, url, { fetchImpl = globalThis.fetch, iceTimeout = ICE_TIMEOUT_MS, signal, timers } = {}) {
  pc.addTransceiver('video', { direction: 'recvonly' });
  pc.addTransceiver('audio', { direction: 'recvonly' });
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
    const what = res.status === 404 ? 'The stream is not published' : `The stream server refused the video (HTTP ${res.status})`;
    throw new WhepError(`${what}${detail ? `: ${detail}` : '.'}`, res.status);
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
 * onState({state: 'connecting'|'playing'|'retrying'|'stopped', message, retryIn}).
 */
export class WhepPlayer {
  constructor({
    url, video = null, onState = () => {}, RTC = globalThis.RTCPeerConnection,
    fetchImpl = globalThis.fetch ? globalThis.fetch.bind(globalThis) : null, iceTimeout = ICE_TIMEOUT_MS,
    steps = RECONNECT_STEPS, setTimer = (fn, ms) => setTimeout(fn, ms), clearTimer = (id) => clearTimeout(id),
    disconnectGrace = 4000,
  } = {}) {
    this.url = url;
    this.video = video;
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
        timers: { setTimer: this.setTimer, clearTimer: this.clearTimer },
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
      this.fail(new WhepError('The video connection failed.'));
    } else if (st === 'disconnected') {
      this.clearTimer(this.graceTimer);
      this.graceTimer = this.setTimer(() => {
        if (pc === this.pc && pc.connectionState === 'disconnected') this.fail(new WhepError('The video connection dropped.'));
      }, this.disconnectGrace);
    }
  }

  fail(err) {
    this.closePc();
    if (!this.wanted) return;
    const wait = this.steps[Math.min(this.attempt, this.steps.length - 1)];
    this.attempt += 1;
    this.emit('retrying', { message: (err && err.message) || 'The video stopped.', retryIn: wait });
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
 * duration or the one the browser read from the file, the shorter of the two; Infinity while neither
 * is known.
 */
export function recordingSpan(file, mediaDuration = null) {
  const lengths = [file && file.duration, mediaDuration].filter((d) => finite(d) && d > 0);
  const start = file && finite(file.start) ? file.start : null;
  if (start == null) return null;
  return [start, lengths.length ? start + Math.min(...lengths) : Infinity];
}

/**
 * the recording of `files` that holds the moment t (epoch), the one that started last when several
 * do (its last frame stays up at its very end, where a replay that reaches the session's end stops);
 * null when none does. `durations` maps a file's url to the length the browser read from it.
 */
export function recordingAt(files, t, durations = null) {
  if (!finite(t)) return null;
  let best = null;
  for (const f of files || []) {
    const span = recordingSpan(f, durations && f ? durations.get(f.url) : null);
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
 * the origin the recorded files load from: the page's scheme and host on the dashboard's media port
 * (`port`, the recordings route's media_port), so the files hold connections of their own; null
 * without a media port, for one that is the page's own port, or for a page not served over http(s).
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
 * whether this same dashboard answers on `origin` within `timeoutMs`: its /api/media-origin names
 * `port` and `instance` (the recordings route's media_port and media_instance, a mark of the
 * dashboard's process), so another dashboard that holds that port number on the page's host (a page
 * opened through a tunnel on another local port) is not taken for it; false for no answer, another
 * server, or an error. Never throws.
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
      return !!data && Number(data.media_port) === Number(port) && data.media_instance === instance;
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

  /** the file's length as the browser read it, null before its metadata */
  get duration() {
    const d = this.video ? this.video.duration : null;
    return this.url && finite(d) && d > 0 ? d : null;
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
 * drawn: {tagged, kept, untagged, untaggedSkeletons}.
 */
export function drawCamera(ctx, cam, { width, height, resolve, tagColor, tagLabel = (t) => `Tag ${t}`, opacity = 1, background = null, box = null, stats = null } = {}) {
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
  const X = (x) => fit.x + x * kx;
  const Y = (y) => fit.y + y * ky;
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
      ctx.strokeStyle = grey;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = opacity * 0.85;
      ctx.strokeRect(X(p.b[0]), Y(p.b[1]), (p.b[2] - p.b[0]) * kx, (p.b[3] - p.b[1]) * ky);
      ctx.globalAlpha = opacity;
    }
    const limbs = Array.isArray(p.k) ? drawSkeleton(ctx, p.k, X, Y, { color: grey, lineWidth: 1.5, radius: 1.5 }) : 0;
    if (stats) {
      stats.untagged += 1;
      if (limbs) stats.untaggedSkeletons += 1;
    }
    regions.push({ untagged: true, tag: null, label: untaggedLabel(p), note: untaggedNote(p), box: p.b ? [X(p.b[0]), Y(p.b[1]), X(p.b[2]), Y(p.b[3])] : null, ray: null });
  }
  // their labels go over every grey skeleton, still under the pupils, and clear of each other
  const placed = [];
  for (const p of untagged) {
    if (!p.b) continue;
    drawLabel(ctx, untaggedLabel(p), X(p.b[0]), Math.max(14, Y(p.b[1]) - 4), { color: grey, size: 11, font, placed });
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
    if (p.b) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = opacity * (kept ? 0.8 : 0.55);
      if (dash) ctx.setLineDash(dash);
      ctx.strokeRect(X(p.b[0]), Y(p.b[1]), (p.b[2] - p.b[0]) * kx, (p.b[3] - p.b[1]) * ky);
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
    const lx = p.b ? X(p.b[0]) : ray ? ray[0] : 0;
    const ly = p.b ? Math.max(16, Y(p.b[1]) - 4) : ray ? ray[1] - 8 : 16;
    drawLabel(ctx, taggedLabel(p, tagLabel), lx, ly, { color, size: 12, font });
    regions.push({
      tag,
      kept: kept ? p.age : null,
      label: taggedLabel(p, tagLabel),
      note: kept ? keptNote(p, tagLabel) : null,
      box: p.b ? [X(p.b[0]), Y(p.b[1]), X(p.b[2]), Y(p.b[3])] : null,
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

/**
 * The tiles of the Cameras card. opts: {tagColor(tag), tagLabel(tag), onNote(text)}.
 * Returns {el, update(state), setExpanded(bool), setOptions({overlay, sync}), hasVideo(), hasLiveVideo(),
 * inspect(), suspend(), resume(), destroy()}. suspend() closes every video but keeps the tiles (a page
 * going into the back/forward cache); resume() lets them play again.
 * update(state): {cameras: [ids], frameAt(t) -> vfa record, now (stream clock), mode ('follow'|'replay'),
 *                 live (the session runs), media (the /media answer | null), serverNow (epoch),
 *                 vfaLag (s | null), recordings ({enabled, files, reason} of the recordings route | null
 *                 while it is asked), speed (replay speed), running (the replay clock advances),
 *                 mediaOrigin (where the files load from, the dashboard's media port; null: the page's origin),
 *                 mediaReserved (how many of the MAX_MEDIA connections the page's sound takes, 0 or 1;
 *                 it counts only without a mediaOrigin)}.
 * The replay of an ended session plays each camera's recorded file (files[].device is the VFA camera
 * id) at the clock; a tile whose recording does not cover the moment draws the skeletons alone. The
 * tiles on screen hold a file first, then those scrolled away (paused, the most recently seen first),
 * up to videoCap() (MAX_MEDIA_VIDEOS from the media origin, else MAX_MEDIA less the sound's share); a
 * tile past that lets go of its file.
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
    // where the recorded files load from (null: the page's origin), the connections the sound takes
    // there, and the recorded files the tiles may hold besides
    origin: null,
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

  /** why a tile draws skeletons instead of video */
  function skeletonNote(id, s) {
    if (s.mode !== 'follow') {
      if (s.live) return 'Skeleton view: recorded video plays once the session has ended.';
      const rec = s.recordings;
      if (!rec) return 'Skeleton view: looking for the camera recordings.';
      if (!rec.enabled) return rec.reason ? `Skeleton view. ${rec.reason}` : 'Skeleton view: the recordings are not available.';
      const unplaced = (rec.files || []).some((f) => f && f.modality === 'video' && String(f.device) === id && !finite(f.start));
      if (unplaced) return 'Skeleton view: the recording of this camera has no start time, so it cannot follow the clock.';
      return 'Skeleton view: the dashboard\'s machine holds no recording of this camera.';
    }
    const media = s.media;
    if (!media) return 'Skeleton view: this session has no live video.';
    const others = (media.streams || []).some((x) => x && x.kind === 'video' && x.ready);
    if (media.webrtc && others) return 'Skeleton view: this camera has no live stream.';
    return media.reason ? `Skeleton view. ${media.reason}` : 'Skeleton view: this session has no live video.';
  }

  function streamFor(id, s) {
    const media = s && s.media;
    if (!media || !media.webrtc || s.mode !== 'follow') return null;
    return (media.streams || []).find((x) => x && x.kind === 'video' && x.ready && (x.camera === id || x.base_id === id)) || null;
  }

  /** the camera's recorded video files, in the replay of an ended session */
  function recordingsFor(id, s) {
    const rec = s && s.recordings;
    if (!rec || !rec.enabled || s.mode !== 'replay' || s.live) return [];
    return (rec.files || []).filter((f) => f && f.modality === 'video' && String(f.device) === id && finite(f.start) && f.url);
  }

  function makeTile(id) {
    const name = h('span', { class: 'cam-name', text: id });
    const age = h('span', { class: 'badge cam-age num', text: fmt.na, title: 'Age of the frame set drawn' });
    const stateEl = h('span', { class: 'cam-state muted' });
    const playBtn = h('button', { class: 'btn sm', attrs: { type: 'button' }, hidden: true }, icon('play', 12), h('span', { text: 'Play' }));
    const head = h('div', { class: 'cam-head' }, name, age, stateEl, h('span', { class: 'spacer' }), playBtn);
    const video = h('video', { class: 'cam-video', attrs: { muted: true, playsinline: true, autoplay: true }, muted: true, hidden: true });
    // the recorded file of a replay: hidden (not display: none, so it keeps loading) until it has a frame
    const fvideo = h('video', { class: 'cam-video cam-file', attrs: { muted: true, playsinline: true, preload: 'auto', disablepictureinpicture: true }, muted: true, style: { visibility: 'hidden' } });
    const canvas = h('canvas', { class: 'cam-canvas', attrs: { role: 'img', 'aria-label': `Camera ${id}: skeletons of the newest frame set` } });
    const stage = h('div', { class: 'cam-stage' }, video, fvideo, canvas);
    const note = h('p', { class: 'cam-note' });
    const tileEl = h('div', { class: 'cam-tile', dataset: { camera: id } }, head, stage, note);
    const tile = {
      id, el: tileEl, name, age, stateEl, playBtn, video, fvideo, canvas, stage, note,
      onScreen: !io, seenAt: 0, player: null, playerState: null, stream: null, regions: [], drawnKey: null, aspect: 16 / 9,
      videoMode: false, kind: null, files: [], file: null, next: null, durations: new Map(), onVideo: false, drawn: null,
    };
    playBtn.addEventListener('click', () => {
      st.priority = [id, ...st.priority.filter((x) => x !== id)];
      schedulePlayback();
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
    return tile;
  }

  function syncTiles(ids) {
    const want = Array.from(new Set(ids.map(String)));
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
    for (const id of want) {
      if (!tiles.has(id)) {
        tiles.set(id, makeTile(id));
        changed = true;
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
    const canPlay = (t) => open && t.onScreen;
    const now = clockMs();
    for (const t of tiles.values()) if (canPlay(t)) t.seenAt = now;
    const eligible = [];
    for (const t of tiles.values()) if (t.videoMode) eligible.push(t.id);
    const order = [...st.priority.filter((id) => eligible.includes(id)), ...eligible.filter((id) => !st.priority.includes(id))];
    // live video comes over WebRTC from the stream server: up to MAX_PLAYING tiles on screen
    const liveOn = order.filter((id) => tiles.get(id).kind === 'live' && canPlay(tiles.get(id)));
    const live = new Set(liveOn.slice(0, MAX_PLAYING));
    // a recorded file holds a connection to the dashboard: the tiles on screen first, then the ones
    // scrolled away that still hold theirs (paused, kept in place), the most recently seen first
    const cap = videoCap(!!st.origin, st.reserved);
    st.fileCap = cap;
    const fileOn = order.filter((id) => tiles.get(id).kind === 'file' && canPlay(tiles.get(id)));
    const files = new Set(fileOn.slice(0, cap));
    const parked = new Set(open ? order
      .filter((id) => {
        const t = tiles.get(id);
        return t.kind === 'file' && !canPlay(t) && t.player instanceof FilePlayer && !!t.player.url;
      })
      .sort((a, b) => tiles.get(b).seenAt - tiles.get(a).seenAt)
      .slice(0, cap - files.size) : []);
    for (const t of tiles.values()) {
      if (!t.videoMode) {
        if (t.player) {
          t.player.stop();
          t.player = null;
        }
        t.playBtn.hidden = true;
        continue;
      }
      if (t.kind === 'file' && (files.has(t.id) || parked.has(t.id))) {
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
    // from the media origin the sound has a connection of its own
    if (st.reserved && !st.origin) return n ? `Up to ${plays} beside the sound` : 'The sound takes the last connection';
    return `Up to ${plays} at once`;
  }

  function renderTileState(t) {
    const ps = t.playerState;
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
    st.origin = typeof s.mediaOrigin === 'string' && s.mediaOrigin ? s.mediaOrigin : null;
    st.reserved = finite(s.mediaReserved) ? Math.max(0, Math.min(MAX_MEDIA, Math.round(s.mediaReserved))) : 0;
    syncTiles(s.cameras || []);
    for (const t of tiles.values()) {
      const stream = streamFor(t.id, s);
      const url = stream ? whepUrl(s.media.webrtc, stream.path) : null;
      t.files = stream ? [] : recordingsFor(t.id, s);
      const kind = stream ? 'live' : t.files.length ? 'file' : null;
      pickRecording(t, s, kind);
      // a recorded camera takes a player (and one of the videoCap() places) only while a file holds the clock
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
    const p = t.player instanceof FilePlayer ? t.player : null;
    // the length the browser read ends a file that is shorter than its listing says
    if (p && p.source && p.duration != null) t.durations.set(p.source, p.duration);
    const file = kind === 'file' ? recordingAt(t.files, s.now, t.durations) : null;
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
    if (pick && !p.parked) p.load(inlineUrl(pick.url, st.origin), pick.url, pick === t.file ? s.now - pick.start : 0);
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
    return t.file.host ? `Recorded video from ${t.file.host}.` : 'Recorded video.';
  }

  /** draw one tile: the overlay of a video shows the frame set at the video's own time */
  function paint(t, s) {
    let at = s.now;
    let delayNote = '';
    let picture = null;
    if (t.kind === 'live') {
      picture = t.video;
      if (finite(s.serverNow)) {
        const lag = finite(s.vfaLag) ? Math.max(0, s.vfaLag) : 0;
        let applied = 0;
        if (st.sync && t.player && supportsVideoDelay()) {
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
    const rec = s.frameAt ? s.frameAt(at) : null;
    const cam = rec ? (rec.c || []).find((c) => String(c.id) === t.id) : null;
    const { w, h: hgt, dpr } = sizeCanvas(t, cam);
    // object-fit: contain letterboxes a picture whose shape differs from the frame set's
    const box = picture ? containBox(picture.videoWidth, picture.videoHeight, w, hgt) : null;
    const showOverlay = !onVideo || st.overlay;
    const boxKey = box ? `${box.x.toFixed(1)},${box.y.toFixed(1)},${box.w.toFixed(1)},${box.h.toFixed(1)}` : '';
    const key = `${rec ? rec.t : 'none'}|${w}x${hgt}|${showOverlay}|${t.kind}|${onVideo}|${boxKey}`;
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
          width: w, height: hgt, resolve, tagColor, tagLabel, box,
          opacity: onVideo ? 0.8 : 1,
          background: onVideo ? null : resolve('var(--surface-2)'),
          stats: t.drawn,
        });
        if (!cam && !onVideo) {
          ctx.fillStyle = resolve('var(--muted)');
          ctx.font = '500 13px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(rec ? 'This camera is missing from the newest frame set.' : 'No frame set yet.', w / 2, hgt / 2);
          ctx.textAlign = 'start';
        }
      }
    }
    const ageS = rec ? (finite(s.serverNow) && s.mode === 'follow' ? s.serverNow - rec.t : at - rec.t) : null;
    t.age.textContent = ageS == null ? 'no frame' : ageS < 1.5 ? 'now' : fmt.ago(ageS);
    t.age.dataset.stale = ageS != null && ageS > 10 ? 'true' : 'false';
    if (t.kind === 'live') t.note.textContent = delayNote;
    else if (t.kind === 'file') t.note.textContent = recordingNote(t, s);
    else t.note.textContent = skeletonNote(t.id, s);
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
      return Array.from(tiles.values()).some((t) => t.kind === 'live');
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
          src: t.fvideo.getAttribute('src'), visibility: t.fvideo.style.visibility,
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
