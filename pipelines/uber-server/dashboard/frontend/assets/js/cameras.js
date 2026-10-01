/**
 * Camera tiles of the live page. A tile plays the camera's live video over WebRTC (WHEP against
 * MediaMTX) when the session streams it and the stream server has WebRTC on, with the VFA skeletons
 * drawn over the video; otherwise it draws the skeletons of the newest frame set on a blank canvas.
 *
 * Video costs bandwidth on the camera's uplink and the stream server, so a tile connects only while
 * the Cameras card is open, the tile is on screen and the tab is visible, and at most MAX_PLAYING
 * tiles play at once; everything else closes its RTCPeerConnection. The WHEP exchange lives in
 * whepNegotiate and WhepPlayer, which take their RTCPeerConnection and fetch as arguments so the
 * offer/answer flow can be tested without a browser or a server.
 */

import { h, clear, fmt, tooltip, theme, icon } from './core.js';
import { colorResolver } from './charts.js';
import { CATEGORY_LABELS, gazeCategory } from './live-model.js';

export const MAX_PLAYING = 4;
export const RECONNECT_STEPS = [1, 2, 5, 10];
export const ICE_TIMEOUT_MS = 2000;
// the browser holds at most this much video back to line it up with the overlay
export const MAX_VIDEO_DELAY = 4;
export const KEYPOINT_MIN_CONF = 0.3;
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

// drawing

/**
 * Draw one camera's frame set (slim VFA camera {id, w, h, tg, ps, pr}) into a canvas context whose
 * user space is css pixels `width` x `height`. Pupils (tagged persons) in their tag colours with
 * skeleton, label and gaze ray; untagged bodies as thin grey boxes; AprilTag centres as small
 * diamonds. Returns the hit regions of the tagged persons for the hover readout.
 */
export function drawCamera(ctx, cam, { width, height, resolve, tagColor, tagLabel = (t) => `Tag ${t}`, opacity = 1, background = null } = {}) {
  const regions = [];
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
  const k = Math.min(width / cam.w, height / cam.h);
  const ox = (width - cam.w * k) / 2;
  const oy = (height - cam.h * k) / 2;
  const X = (x) => ox + x * k;
  const Y = (y) => oy + y * k;
  ctx.globalAlpha = opacity;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const grey = resolve('var(--tag-other)');
  const surface = resolve('var(--surface)');
  const font = (size, weight = 600) => `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;

  // untagged bodies first, under the pupils
  for (const p of cam.ps || []) {
    if (!p || p.tag != null || !p.b) continue;
    ctx.strokeStyle = grey;
    ctx.globalAlpha = opacity * 0.7;
    ctx.lineWidth = 1;
    ctx.strokeRect(X(p.b[0]) + 0.5, Y(p.b[1]) + 0.5, (p.b[2] - p.b[0]) * k, (p.b[3] - p.b[1]) * k);
  }
  ctx.globalAlpha = opacity;

  for (const p of cam.ps || []) {
    if (!p || p.tag == null) continue;
    const tag = String(p.tag);
    const color = resolve(tagColor(tag));
    if (p.b) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = opacity * 0.55;
      ctx.strokeRect(X(p.b[0]), Y(p.b[1]), (p.b[2] - p.b[0]) * k, (p.b[3] - p.b[1]) * k);
      ctx.globalAlpha = opacity;
    }
    const kp = Array.isArray(p.k) ? p.k : null;
    if (kp) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const [i, j] of COCO_LIMBS) {
        const a = kp[i];
        const b = kp[j];
        if (!a || !b || !(a[2] >= KEYPOINT_MIN_CONF) || !(b[2] >= KEYPOINT_MIN_CONF)) continue;
        ctx.moveTo(X(a[0]), Y(a[1]));
        ctx.lineTo(X(b[0]), Y(b[1]));
      }
      ctx.stroke();
      ctx.fillStyle = color;
      for (const pt of kp) {
        if (!pt || !(pt[2] >= KEYPOINT_MIN_CONF)) continue;
        ctx.beginPath();
        ctx.arc(X(pt[0]), Y(pt[1]), 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
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
    const label = tagLabel(tag);
    ctx.font = font(12);
    const tw = ctx.measureText(label).width;
    const lx = p.b ? X(p.b[0]) : ray ? ray[0] : 0;
    const ly = p.b ? Math.max(16, Y(p.b[1]) - 4) : ray ? ray[1] - 8 : 16;
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 13, tw + 10, 16);
    ctx.fillStyle = textOn(color);
    ctx.fillText(label, lx + 5, ly);
    regions.push({
      tag,
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

/** the tagged person under (x, y): the nearest gaze ray within 8 px, else a box holding the point. */
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
  for (const r of regions || []) {
    if (r.box && x >= r.box[0] && x <= r.box[2] && y >= r.box[1] && y <= r.box[3]) return r;
  }
  return null;
}

// tiles

/**
 * The tiles of the Cameras card. opts: {tagColor(tag), tagLabel(tag), onNote(text)}.
 * Returns {el, update(state), setExpanded(bool), setOptions({overlay, sync}), hasVideo(), suspend(),
 * resume(), destroy()}. suspend() closes every video but keeps the tiles (a page going into the
 * back/forward cache); resume() lets them play again.
 * update(state): {cameras: [ids], frameAt(t) -> vfa record, now (stream clock), mode ('follow'|'replay'),
 *                 media (the /media answer | null), serverNow (epoch), vfaLag (s | null)}.
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
  };
  const io = typeof IntersectionObserver !== 'undefined'
    ? new IntersectionObserver((entries) => {
      for (const en of entries) {
        const tile = Array.from(tiles.values()).find((t) => t.el === en.target);
        if (tile) tile.onScreen = en.isIntersecting;
      }
      schedulePlayback();
    }, { threshold: 0.01 })
    : null;
  const onVisibility = () => {
    st.docVisible = document.visibilityState !== 'hidden';
    schedulePlayback();
  };
  if (typeof document !== 'undefined') document.addEventListener('visibilitychange', onVisibility);
  const offTheme = theme.onChange(() => {
    for (const t of tiles.values()) t.drawnKey = null;
    if (st.last) render(st.last);
  });
  const ro = typeof ResizeObserver !== 'undefined'
    ? new ResizeObserver(() => {
      for (const t of tiles.values()) t.drawnKey = null;
      if (st.last) render(st.last);
    })
    : null;

  /** why a tile draws skeletons instead of video */
  function skeletonNote(id, s) {
    if (s.mode !== 'follow') return 'Skeleton view: a replay shows no video.';
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

  function makeTile(id) {
    const name = h('span', { class: 'cam-name', text: id });
    const age = h('span', { class: 'badge cam-age num', text: fmt.na, title: 'Age of the frame set drawn' });
    const stateEl = h('span', { class: 'cam-state muted' });
    const playBtn = h('button', { class: 'btn sm', attrs: { type: 'button' }, hidden: true }, icon('play', 12), h('span', { text: 'Play' }));
    const head = h('div', { class: 'cam-head' }, name, age, stateEl, h('span', { class: 'spacer' }), playBtn);
    const video = h('video', { class: 'cam-video', attrs: { muted: true, playsinline: true, autoplay: true }, muted: true, hidden: true });
    const canvas = h('canvas', { class: 'cam-canvas', attrs: { role: 'img', 'aria-label': `Camera ${id}: skeletons of the newest frame set` } });
    const stage = h('div', { class: 'cam-stage' }, video, canvas);
    const note = h('p', { class: 'cam-note' });
    const tileEl = h('div', { class: 'cam-tile', dataset: { camera: id } }, head, stage, note);
    const tile = {
      id, el: tileEl, name, age, stateEl, playBtn, video, canvas, stage, note,
      onScreen: !io, player: null, playerState: null, stream: null, regions: [], drawnKey: null, aspect: 16 / 9,
      videoMode: false,
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
      const cat = hit.cat ? gazeCategory(hit.cat, hit.to) : null;
      tooltip.show(e, {
        title: tagLabel(hit.tag),
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
    const eligible = [];
    for (const t of tiles.values()) if (t.videoMode) eligible.push(t.id);
    const canPlay = (t) => st.expanded && st.docVisible && !st.suspended && t.onScreen;
    const order = [...st.priority.filter((id) => eligible.includes(id)), ...eligible.filter((id) => !st.priority.includes(id))];
    const active = new Set(order.filter((id) => canPlay(tiles.get(id))).slice(0, MAX_PLAYING));
    for (const t of tiles.values()) {
      if (!t.videoMode) {
        if (t.player) {
          t.player.stop();
          t.player = null;
        }
        t.playBtn.hidden = true;
        continue;
      }
      if (active.has(t.id)) {
        if (!t.player || t.player.url !== t.url) {
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
        t.playBtn.hidden = !(canPlay(t) && eligible.length > MAX_PLAYING);
      }
      renderTileState(t);
    }
  }

  function renderTileState(t) {
    const ps = t.playerState;
    if (!t.videoMode) {
      t.stateEl.textContent = '';
      return;
    }
    if (!t.player) t.stateEl.textContent = t.playBtn.hidden ? 'Paused' : `Up to ${MAX_PLAYING} cameras play at once`;
    else if (!ps || ps.state === 'connecting') t.stateEl.textContent = 'Connecting';
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
    syncTiles(s.cameras || []);
    for (const t of tiles.values()) {
      const stream = streamFor(t.id, s);
      const url = stream ? whepUrl(s.media.webrtc, stream.path) : null;
      const videoMode = !!stream;
      if (videoMode !== t.videoMode || url !== t.url) {
        t.videoMode = videoMode;
        t.url = url;
        t.video.hidden = !videoMode;
        t.el.classList.toggle('is-video', videoMode);
        t.drawnKey = null;
      }
    }
    schedulePlayback();
    if (!st.expanded) return;
    const resolve = colorResolver(el);
    for (const t of tiles.values()) {
      // the overlay of a video shows the frame set at the video's (delayed) time
      let at = s.now;
      let delayNote = '';
      if (t.videoMode && finite(s.serverNow)) {
        const lag = finite(s.vfaLag) ? Math.max(0, s.vfaLag) : 0;
        let applied = 0;
        if (st.sync && t.player && supportsVideoDelay()) {
          applied = Math.min(lag, MAX_VIDEO_DELAY);
          if (t.player.setDelay(applied) && applied >= 0.2) delayNote = `Video delayed ${fmt.num(applied, 1)} s to match the overlay.`;
        } else if (t.player) t.player.setDelay(0);
        at = s.serverNow - applied;
      }
      const rec = s.frameAt ? s.frameAt(at) : null;
      const cam = rec ? (rec.c || []).find((c) => String(c.id) === t.id) : null;
      const { w, h: hgt, dpr } = sizeCanvas(t, cam);
      const showOverlay = !t.videoMode || st.overlay;
      const key = `${rec ? rec.t : 'none'}|${w}x${hgt}|${showOverlay}|${t.videoMode}`;
      if (key !== t.drawnKey) {
        t.drawnKey = key;
        const ctx = t.canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (!showOverlay) {
          ctx.clearRect(0, 0, w, hgt);
          t.regions = [];
        } else {
          t.regions = drawCamera(ctx, cam, {
            width: w, height: hgt, resolve, tagColor, tagLabel,
            opacity: t.videoMode ? 0.8 : 1,
            background: t.videoMode ? null : resolve('var(--surface-2)'),
          });
          if (!cam && !t.videoMode) {
            ctx.fillStyle = resolve('var(--muted)');
            ctx.font = '500 13px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(rec ? 'This camera is missing from the newest frame set.' : 'No frame set yet.', w / 2, hgt / 2);
            ctx.textAlign = 'start';
          }
        }
      }
      const ageS = rec ? (finite(s.serverNow) && s.mode === 'follow' ? s.serverNow - rec.t : s.now - rec.t) : null;
      t.age.textContent = ageS == null ? 'no frame' : ageS < 1.5 ? 'now' : fmt.ago(ageS);
      t.age.dataset.stale = ageS != null && ageS > 10 ? 'true' : 'false';
      if (t.videoMode) t.note.textContent = delayNote;
      else t.note.textContent = skeletonNote(t.id, s);
    }
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
      if (st.last) render(st.last);
    },
    hasVideo() {
      return Array.from(tiles.values()).some((t) => t.videoMode);
    },
    playing() {
      return Array.from(tiles.values()).filter((t) => t.player).map((t) => t.id);
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
