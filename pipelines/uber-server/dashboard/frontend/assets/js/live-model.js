/**
 * Rolling state of the live page. It keeps the records of the live stream (recognition buckets,
 * transcript chunks, IPS windows and VFA frame sets, in the slim formats of
 * openmmla.analytics.report.live) by their absolute time, trims them to the last KEEP_SECONDS (or
 * the timeline window when that is longer), and derives the panels' numbers on demand for a window
 * [a, b) of absolute epoch seconds. Nothing here touches the DOM, so the derived metrics can be
 * checked in node against a recorded stream.
 *
 * The rules follow the analysis (report/speech.py, fusion/window_features.py) where the slim records
 * allow it. A tag the VFA server kept on a camera's track counts for TAG_MEMORY_SECONDS after the
 * track last read it (TagMemory), as in the analysis; the analysis also names the untagged frames of
 * a track after its nearest read, before or after them, which a live view cannot do. One
 * approximation: a slim recognition bucket drops the segment starts, so a speaker segment several
 * bases reported is recognised by its (name, duration) instead of (name, segment).
 */

import { sortTags, isPupilTag, voiceLabel } from './core.js';

export const KEEP_SECONDS = 900;
export const BUCKET = 3.0;
export const SILENT_LABELS = new Set(['silent', 'unknown', '']);
export const GROUP_LABEL_RE = /^group_\d+$/;
// a turn of the same voice this soon after the one before it continues it
export const MERGE_GAP = 0.5;
// a change of voice this soon after the last turn is a switch
export const SWITCH_GAP = 2.0;
// a change of wearer counts across this much silence
export const BRIDGE_SECONDS = 9.0;
// beside worn microphones, a bucket only wearers named is the group microphone's silence when the
// group microphone named speech this recently
export const GROUP_SILENCE_REACH = 30.0;
// two gazes within this share of the frame's width are on the same spot
export const JA_WIDTH = 0.05;
export const JA_LAGS = [20, 30, 40];
export const JA_SLACK = 0.5;
export const JA_MIN = 10;
// a pair counts in the joint attention headline once seen together in frames numbering this share of
// the span's seconds (VFA keeps about one frame set a second; the analysis asks for 10 windows of 10 s)
export const JA_PAIR_MIN_SHARE = 0.2;
// a tag counts as a participant (and gets a colour) once seen in this many distinct seconds
export const ROSTER_MIN_SECONDS = 5;
// how long after a camera's track last read a tag the page trusts that tag on the track: the VFA
// server keeps a read tag on its ByteTrack track with no time limit (tag_match 'track', sent as
// rm: 1), and the analysis trusts it this long (openmmla/analytics/fusion/window_features.py
// TAG_MEMORY_SECONDS, expire_track_tags); a body with an older kept tag counts for nobody
export const TAG_MEMORY_SECONDS = 60;
export const CATEGORIES = ['partner_face', 'partner_hands', 'other_people', 'task', 'elsewhere', 'unreadable'];
export const CATEGORY_LABELS = {
  partner_face: 'Partner face',
  partner_hands: 'Partner hands',
  other_people: 'Other people',
  task: 'Task',
  elsewhere: 'Elsewhere',
  unreadable: 'Unreadable',
};
// the raw VFA categories a gaze point is not drawn for (window_features._NO_POINT)
const NO_POINT = new Set(['unknown', 'out_of_frame']);

function finite(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function validRecord(r) {
  return r && typeof r === 'object' && finite(r.t);
}

/** the six display categories from a raw VFA gaze category and its target ("other" = not a pupil). */
export function gazeCategory(cat, to) {
  switch (cat) {
    case 'partner_face':
      return to != null && to !== 'other' ? 'partner_face' : 'other_people';
    case 'partner_hands':
      return to != null && to !== 'other' ? 'partner_hands' : 'other_people';
    case 'other_face':
    case 'other_hands':
      return 'other_people';
    case 'own_hands':
    case 'zone':
    case 'work_area':
      return 'task';
    case 'elsewhere':
      return 'elsewhere';
    default:
      return 'unreadable';
  }
}

export function pairKey(a, b) {
  const [x, y] = sortTags([a, b]);
  return `${x}|${y}`;
}

/** every unordered pair of the tags, sorted: [[a, b, "a|b"], ...] */
export function pairList(tags) {
  const t = sortTags(tags || []);
  const out = [];
  for (let i = 0; i < t.length; i += 1) for (let j = i + 1; j < t.length; j += 1) out.push([t[i], t[j], `${t[i]}|${t[j]}`]);
  return out;
}

export function median(values) {
  const v = values.filter(finite).sort((x, y) => x - y);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/** normalised Shannon entropy of positive amounts; null for fewer than two. */
export function entropyBalance(values) {
  const pos = Array.from(values || []).filter((v) => finite(v) && v > 0);
  if (pos.length < 2) return null;
  const total = pos.reduce((a, b) => a + b, 0);
  let h = 0;
  for (const v of pos) h -= (v / total) * Math.log(v / total);
  return h / Math.log(pos.length);
}

/** the union length of intervals [[s, e], ...]. */
export function unionLength(intervals) {
  const iv = intervals.filter((x) => x[1] > x[0]).sort((x, y) => x[0] - y[0]);
  let total = 0;
  let cs = null;
  let ce = null;
  for (const [s, e] of iv) {
    if (cs == null || s > ce) {
      if (cs != null) total += ce - cs;
      cs = s;
      ce = e;
    } else if (e > ce) ce = e;
  }
  if (cs != null) total += ce - cs;
  return total;
}

/** turns sorted by start, a turn of the same key at most MERGE_GAP after the one before joined to it. */
export function mergeTurns(turns) {
  const merged = [];
  for (const [s, e, k] of turns.slice().sort((x, y) => x[0] - y[0] || x[1] - y[1])) {
    const last = merged[merged.length - 1];
    if (last && last[2] === k && s - last[1] <= MERGE_GAP) last[1] = Math.max(last[1], e);
    else merged.push([s, e, k]);
  }
  return merged;
}

/** camera-frame metres (x, y, z) to floor (u, v, h); without a basis the camera's x-z plane. */
export function project(p, basis) {
  if (!p || p.length < 3) return null;
  const [x, y, z] = p.map(Number);
  if (![x, y, z].every(Number.isFinite)) return null;
  if (!basis || !basis.ex || !basis.ef || !basis.g) return [x, z, -y];
  const dot = (a) => a[0] * x + a[1] * y + a[2] * z;
  return [dot(basis.ex), dot(basis.ef), -dot(basis.g)];
}

/** the IPS cameras of the meta (main camera frame) on the floor: [{id, u, v, main, heading}], main first. */
export function projectCameras(cameras, basis) {
  const out = [];
  for (const c of cameras || []) {
    const p = project(c && c.position, basis);
    if (!p) continue;
    let heading = null;
    const ax = project(c.axis, basis);
    if (ax && Math.hypot(ax[0], ax[1]) > 1e-9) heading = Math.atan2(ax[0], ax[1]);
    out.push({ id: String(c.id), u: p[0], v: p[1], main: !!c.main, heading });
  }
  return out.sort((a, b) => (b.main ? 1 : 0) - (a.main ? 1 : 0));
}

function percentile(sorted, q) {
  if (!sorted.length) return null;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

/**
 * The plan to draw: the 1st to 99th percentile of the badge positions, the cameras within `reach`
 * metres of that box, padded and snapped outward to `snap`; at least 2 m on each axis. null without
 * positions or cameras.
 */
export function roomExtent(points, cameras = [], { pad = 0.3, reach = 6, snap = 0.5, min = 2 } = {}) {
  const us = points.map((p) => p[0]).filter(finite).sort((a, b) => a - b);
  const vs = points.map((p) => p[1]).filter(finite).sort((a, b) => a - b);
  let box = null;
  if (us.length && vs.length) box = { u: [percentile(us, 0.01), percentile(us, 0.99)], v: [percentile(vs, 0.01), percentile(vs, 0.99)] };
  for (const c of cameras || []) {
    if (!finite(c.u) || !finite(c.v)) continue;
    if (box) {
      const du = Math.max(box.u[0] - c.u, 0, c.u - box.u[1]);
      const dv = Math.max(box.v[0] - c.v, 0, c.v - box.v[1]);
      if (Math.hypot(du, dv) > reach) continue;
      box.u = [Math.min(box.u[0], c.u), Math.max(box.u[1], c.u)];
      box.v = [Math.min(box.v[0], c.v), Math.max(box.v[1], c.v)];
    } else box = { u: [c.u, c.u], v: [c.v, c.v] };
  }
  if (!box) return null;
  const fit = (r) => {
    let lo = Math.floor((r[0] - pad) / snap) * snap;
    let hi = Math.ceil((r[1] + pad) / snap) * snap;
    if (hi - lo < min) {
      const mid = (lo + hi) / 2;
      lo = Math.floor((mid - min / 2) / snap) * snap;
      hi = lo + Math.ceil(min / snap) * snap;
    }
    return [Number(lo.toFixed(3)), Number(hi.toFixed(3))];
  };
  return { u: fit(box.u), v: fit(box.v) };
}

/** true when extent `outer` already holds `inner`. */
export function extentContains(outer, inner) {
  if (!outer || !inner) return false;
  return outer.u[0] <= inner.u[0] + 1e-9 && outer.u[1] >= inner.u[1] - 1e-9 && outer.v[0] <= inner.v[0] + 1e-9 && outer.v[1] >= inner.v[1] - 1e-9;
}

/** the union of two extents. */
export function extentUnion(a, b) {
  if (!a) return b;
  if (!b) return a;
  return { u: [Math.min(a.u[0], b.u[0]), Math.max(a.u[1], b.u[1])], v: [Math.min(a.v[0], b.v[0]), Math.max(a.v[1], b.v[1])] };
}

/**
 * The badge reads of each camera's tracks, learnt from the frame sets in time order: per (camera id,
 * track id) the time and tag of the track's last read (a tagged body with rm 0, or an older record
 * without rm). apply(rec) settles the kept tags of a frame set (rm 1) by the analysis's rule
 * (window_features.expire_track_tags): a kept tag whose track last read the same tag at most
 * TAG_MEMORY_SECONDS before stays on the body, with `age`, the seconds since that read; any other
 * (an older read, another tag read last, or no read in the page's history) is taken off, and the
 * body is an untagged one (tag null, `id` track_<n>, `xt` the tag it was kept as), its camera's
 * pairs with that tag dropped and the gazes that landed on it aimed at 'other', unless another body
 * of the camera still carries the tag. The records are changed in place, so the tiles, the roster
 * and every measure see the same bodies. The page's history is what the model holds: a seek or a
 * new start clears it with the model and rebuilds it from the history it loads (up to five minutes).
 */
export class TagMemory {
  constructor(memory = TAG_MEMORY_SECONDS) {
    this.memory = memory;
    this.reads = new Map();
  }

  reset() {
    this.reads.clear();
  }

  /** settle the kept tags of a frame set (in time order); returns how many bodies lost their tag */
  apply(rec) {
    if (!rec || !finite(rec.t) || !Array.isArray(rec.c)) return 0;
    const t = rec.t;
    let taken = 0;
    let pairsChanged = false;
    for (const c of rec.c) {
      if (!c || !Array.isArray(c.ps)) continue;
      const cam = String(c.id);
      // this frame set's reads first, as the analysis learns them before it judges the kept tags
      for (const p of c.ps) {
        if (!p || p.tag == null || p.rm === 1 || !finite(p.tr)) continue;
        const key = `${cam}\u0000${p.tr}`;
        const prev = this.reads.get(key);
        if (!prev || prev.t <= t) this.reads.set(key, { t, tag: String(p.tag) });
      }
      const lost = new Set();
      for (const p of c.ps) {
        if (!p || p.tag == null || p.rm !== 1) continue;
        const tag = String(p.tag);
        const read = finite(p.tr) ? this.reads.get(`${cam}\u0000${p.tr}`) : null;
        if (read && read.tag === tag && t - read.t <= this.memory) {
          p.age = Math.max(0, t - read.t);
          continue;
        }
        p.xt = tag;
        p.tag = null;
        p.id = finite(p.tr) ? `track_${p.tr}` : (p.id ?? null);
        delete p.age;
        lost.add(tag);
        taken += 1;
      }
      if (!lost.size) continue;
      // a tag another body of the camera still carries keeps its pairs and the looks at it
      for (const p of c.ps) if (p && p.tag != null) lost.delete(String(p.tag));
      if (!lost.size) continue;
      if (c.pr && typeof c.pr === 'object') {
        for (const key of Object.keys(c.pr)) {
          if (key.split('|').some((x) => lost.has(x))) {
            delete c.pr[key];
            pairsChanged = true;
          }
        }
      }
      for (const p of c.ps) if (p && p.g && p.g.to != null && lost.has(String(p.g.to))) p.g.to = 'other';
    }
    if (pairsChanged) {
      // the frame set's pairs again, as slim_vfa picks them: from the first camera that measured
      // the pair's gaze distance, else the first that holds the pair
      const pairs = {};
      for (const c of rec.c) {
        for (const [key, v] of Object.entries((c && c.pr) || {})) {
          if (!(key in pairs) || (pairs[key][0] == null && Array.isArray(v) && v[0] != null)) pairs[key] = v;
        }
      }
      rec.pr = pairs;
    }
    return taken;
  }

  /** forget the reads made before `before` (no later frame set can trust them) */
  forget(before) {
    for (const [key, read] of this.reads) if (read.t < before) this.reads.delete(key);
  }
}

/** records sorted by t, unique by key; `range(a, b)` is [a, b). */
class TimeStore {
  constructor(keyOf) {
    this.items = [];
    this.keys = new Set();
    this.keyOf = keyOf;
  }

  get first() {
    return this.items.length ? this.items[0] : null;
  }

  get last() {
    return this.items.length ? this.items[this.items.length - 1] : null;
  }

  /** first index with t >= x */
  lower(x) {
    const a = this.items;
    let lo = 0;
    let hi = a.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (a[mid].t < x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /** first index with t > x */
  upper(x) {
    const a = this.items;
    let lo = 0;
    let hi = a.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (a[mid].t <= x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  add(rec) {
    const k = this.keyOf(rec);
    if (this.keys.has(k)) return false;
    this.keys.add(k);
    const n = this.items.length;
    if (!n || this.items[n - 1].t <= rec.t) this.items.push(rec);
    else this.items.splice(this.upper(rec.t), 0, rec);
    return true;
  }

  range(a, b) {
    return this.items.slice(this.lower(a), this.lower(b));
  }

  /** the newest record at or before t */
  at(t) {
    const i = this.upper(t) - 1;
    return i >= 0 ? this.items[i] : null;
  }

  trim(before) {
    const cut = this.lower(before);
    if (cut <= 0) return;
    for (const r of this.items.slice(0, cut)) this.keys.delete(this.keyOf(r));
    this.items = this.items.slice(cut);
  }
}

const k3 = (v) => (finite(v) ? v.toFixed(3) : '');

export class LiveModel {
  /** opts: {t0, t1, groupId, speechMode ('group'|'wearer'|'wearer+group'|'individual'|null), keep} */
  constructor(opts = {}) {
    this.t0 = null;
    this.t1 = null;
    this.groupId = null;
    this.speechMode = null;
    this.keep = KEEP_SECONDS;
    this.floor = null;
    this.version = 0;
    this.configure(opts);
    this.reset();
  }

  configure({ t0, t1, groupId, speechMode, keep } = {}) {
    if (finite(t0)) this.t0 = t0;
    if (finite(t1)) this.t1 = t1;
    if (groupId !== undefined && groupId !== null) this.groupId = String(groupId);
    if (speechMode !== undefined) this.speechMode = speechMode || null;
    if (finite(keep)) this.keep = Math.max(KEEP_SECONDS, keep);
  }

  /** forget every record (a seek); colours live in the page and stay. */
  reset() {
    this.asr = new TimeStore((r) => k3(r.t));
    this.tr = new TimeStore((r) => `${k3(r.t)}|${k3(r.e)}|${r.pt ?? ''}|${r.sp ?? ''}`);
    this.ips = new TimeStore((r) => k3(r.t));
    this.vfa = new TimeStore((r) => k3(r.t));
    this.tagMemory = new TagMemory();
    this.tagSeen = new Map();
    this.voiceOrder = [];
    this.voiceSet = new Set();
    this.names = { group: false, wearers: new Set(), speakers: new Set() };
    this.totals = new Map();
    this.maxCameras = 0;
    this.cameraIds = new Set();
    this.version += 1;
  }

  /** take in one SSE batch; returns how many new records of each kind it held. */
  addBatch(batch) {
    const added = { asr: 0, tr: 0, ips: 0, vfa: 0 };
    if (!batch) return added;
    for (const r of batch.asr || []) if (validRecord(r) && this.asr.add(r)) { added.asr += 1; this.onAsr(r); }
    for (const r of batch.tr || []) if (validRecord(r) && this.tr.add(r)) { added.tr += 1; this.onTr(r); }
    for (const r of batch.ips || []) if (validRecord(r) && this.ips.add(r)) { added.ips += 1; this.onIps(r); }
    for (const r of batch.vfa || []) if (validRecord(r) && this.vfa.add(r)) { added.vfa += 1; this.onVfa(r); }
    if (added.asr + added.tr + added.ips + added.vfa) {
      this.version += 1;
      this.trim();
    }
    return added;
  }

  onAsr(r) {
    if (!Array.isArray(r.sp)) r.sp = [];
    if (!Array.isArray(r.du)) r.du = [];
    for (const n of r.sp) {
      if (SILENT_LABELS.has(n)) continue;
      if (this.isGroupLabel(n)) this.names.group = true;
      else if (isPupilTag(n)) this.names.wearers.add(n);
      else this.names.speakers.add(n);
    }
    for (const kind of ['wearers', 'speakers']) {
      const prefix = kind === 'wearers' ? 'tag:' : 'spk:';
      for (const [n, s] of this.namedIn(r, kind)) this.totals.set(prefix + n, (this.totals.get(prefix + n) || 0) + s);
    }
  }

  onTr(r) {
    if (r.pt != null) return;
    const byKey = new Map();
    for (const tn of r.turns || []) {
      if (!tn || !tn[2]) continue;
      if (!byKey.has(tn[2])) byKey.set(tn[2], []);
      byKey.get(tn[2]).push([tn[0], tn[1]]);
    }
    for (const w of r.w || []) {
      const key = w && w[3];
      if (key && !byKey.has(key)) byKey.set(key, []);
    }
    for (const [key, iv] of byKey) {
      if (!this.voiceSet.has(key)) {
        this.voiceSet.add(key);
        this.voiceOrder.push(key);
      }
      this.totals.set(key, (this.totals.get(key) || 0) + unionLength(iv));
    }
  }

  onIps(r) {
    if (!r.p || typeof r.p !== 'object') r.p = {};
    if (!r.hd || typeof r.hd !== 'object') r.hd = {};
    if (!Array.isArray(r.f)) r.f = [];
    if (!this.floor) r.legacy = true;
    for (const tag of Object.keys(r.p)) this.seeTag(tag, r.t);
  }

  onVfa(r) {
    if (!Array.isArray(r.c)) r.c = [];
    // a tag kept on a track for longer than the analysis trusts it counts for nobody, here too
    this.tagMemory.apply(r);
    this.maxCameras = Math.max(this.maxCameras, r.c.length);
    for (const c of r.c) {
      if (c && c.id != null) this.cameraIds.add(String(c.id));
      for (const p of (c && c.ps) || []) if (p && p.tag != null) this.seeTag(p.tag, r.t);
      for (const tag of Object.keys((c && c.tg) || {})) this.seeTag(tag, r.t);
    }
  }

  seeTag(tag, t) {
    const id = String(tag);
    if (!isPupilTag(id)) return;
    const sec = Math.floor(t);
    let st = this.tagSeen.get(id);
    if (!st) {
      st = { last: null, n: 0, first: t };
      this.tagSeen.set(id, st);
    }
    if (st.last !== sec) {
      st.last = sec;
      st.n += 1;
    }
  }

  /** pupil tags seen in at least `min` distinct seconds, sorted. */
  roster(min = ROSTER_MIN_SECONDS) {
    return sortTags(Array.from(this.tagSeen).filter(([, st]) => st.n >= min).map(([id]) => id));
  }

  /** the floor basis arrived after some windows: lay those (sent in the camera's x-z plane) on it. */
  setFloor(basis) {
    if (!basis || !basis.ex || !basis.ef || !basis.g) return;
    this.floor = basis;
    for (const r of this.ips.items) {
      if (!r.legacy) continue;
      for (const tag of Object.keys(r.p)) {
        const q = r.p[tag];
        // legacy (u, v, h) = (x, z, -y)
        const p = Array.isArray(q) ? project([q[0], -q[2], q[1]], basis) : null;
        if (p) r.p[tag] = p;
        else delete r.p[tag];
      }
      // a legacy heading lost the vertical part of the facing, so it cannot be laid on the floor
      r.hd = {};
      r.legacy = false;
    }
    this.version += 1;
  }

  trim() {
    const newest = this.newestTime();
    if (newest == null) return;
    const before = newest - this.keep;
    // the reads are judged by frame sets only, which can lag the other records in follow mode
    const lastVfa = this.vfa.last;
    this.asr.trim(before);
    // a chunk starts up to 30 s before its end
    this.tr.trim(before - 60);
    this.ips.trim(before);
    this.vfa.trim(before);
    if (lastVfa) this.tagMemory.forget(lastVfa.t - this.tagMemory.memory - 10);
  }

  newestTime() {
    const ts = [this.asr.last, this.tr.last, this.ips.last, this.vfa.last].filter(Boolean).map((r) => r.t);
    return ts.length ? Math.max(...ts) : null;
  }

  oldestTime() {
    const ts = [this.asr.first, this.tr.first, this.ips.first, this.vfa.first].filter(Boolean).map((r) => r.t);
    return ts.length ? Math.min(...ts) : null;
  }

  /** newest transcript chunk end among the last two minutes of chunks. */
  newestChunkEnd() {
    const items = this.tr.items;
    if (!items.length) return null;
    const lastT = items[items.length - 1].t;
    let best = null;
    for (let i = items.length - 1; i >= 0 && items[i].t >= lastT - 120; i -= 1) {
      const e = finite(items[i].e) ? items[i].e : items[i].t;
      if (best == null || e > best) best = e;
    }
    return best;
  }

  isGroupLabel(name) {
    const n = String(name ?? '');
    return !!n && (n === this.groupId || GROUP_LABEL_RE.test(n));
  }

  /** 'voices' (group microphone, diarized), 'wearers' (worn microphones) or 'speakers' (named). */
  entityMode() {
    const m = this.speechMode;
    if (m === 'wearer' || m === 'wearer+group') return 'wearers';
    if (m === 'individual') return 'speakers';
    if (m === 'group') return 'voices';
    if (this.names.wearers.size) return 'wearers';
    if (this.names.speakers.size) return 'speakers';
    return 'voices';
  }

  /** a group microphone runs beside worn ones: speech and silence are read from its entries only. */
  groupApart() {
    if (this.speechMode) return this.speechMode === 'wearer+group';
    return this.names.group && this.names.wearers.size > 0;
  }

  entries(r) {
    const d = r.d > 0 ? r.d : BUCKET;
    return r.sp.map((n, j) => [String(n), finite(r.du[j]) ? Math.max(0, r.du[j]) : d]);
  }

  /**
   * voiced fraction of a bucket: min(sum over unique (name, duration) entries not silent of
   * min(duration, d), d) / d. Beside worn microphones only the group microphone's entries count; a
   * bucket only wearers named is its silence when it spoke within GROUP_SILENCE_REACH (`groupNear`),
   * else unreadable (null). null for a bucket without entries.
   */
  bucketVoiced(r, groupNear = true) {
    const d = r.d > 0 ? r.d : BUCKET;
    let entries = this.entries(r);
    if (this.groupApart()) {
      const kept = entries.filter(([n]) => !isPupilTag(n));
      if (!kept.length && entries.length) entries = groupNear ? [['silent', d]] : null;
      else entries = kept;
    }
    if (!entries || !entries.length) return null;
    const heard = new Map();
    for (const [n, s] of entries) heard.set(`${n}\u0000${s}`, [n, Math.min(s, d)]);
    let spoken = 0;
    for (const [n, s] of heard.values()) if (!SILENT_LABELS.has(n)) spoken += s;
    return Math.min(spoken, d) / d;
  }

  /** {t: [bucket starts], v: [voiced fraction | null]} for the buckets starting in [a, b). */
  speechActivity(a, b) {
    const out = { t: [], v: [], step: BUCKET };
    const apart = this.groupApart();
    let lastGroup = -Infinity;
    const items = this.asr.items;
    for (let i = this.asr.lower(apart ? a - GROUP_SILENCE_REACH : a); i < items.length && items[i].t < b; i += 1) {
      const r = items[i];
      if (apart && r.sp.some((n, j) => this.isGroupLabel(n) && !(finite(r.du[j]) && r.du[j] <= 0))) lastGroup = r.t;
      if (r.t < a) continue;
      out.t.push(r.t);
      out.v.push(this.bucketVoiced(r, !apart || r.t - lastGroup <= GROUP_SILENCE_REACH));
    }
    return out;
  }

  /** mean voiced fraction of the readable buckets in [a, b); null without one. */
  speechRatio(a, b) {
    const v = this.speechActivity(a, b).v.filter(finite);
    return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null;
  }

  /** a bucket's named speakers of one kind (wearers: badge tags; speakers: other names) with seconds. */
  namedIn(r, kind) {
    const d = r.d > 0 ? r.d : BUCKET;
    const out = new Map();
    (r.sp || []).forEach((name, j) => {
      const n = String(name);
      if (SILENT_LABELS.has(n) || this.isGroupLabel(n)) return;
      const pupil = isPupilTag(n);
      if (kind === 'wearers' ? !pupil : pupil) return;
      const s = finite(r.du && r.du[j]) ? Math.max(0, r.du[j]) : d;
      out.set(n, Math.max(out.get(n) || 0, Math.min(s, d)));
    });
    return out;
  }

  /**
   * speaker turns overlapping [a, b) as [start, end, key, seconds]: voices from the diarized turns
   * of the group chunks (absolute = chunk start + relative), wearers and named speakers from the
   * recognition buckets that name them (one bucket = one turn; seconds = its voiced time).
   */
  turns(a, b) {
    const mode = this.entityMode();
    const out = [];
    if (mode === 'voices') {
      const items = this.tr.items;
      for (let i = this.tr.lower(a - 120); i < items.length && items[i].t < b; i += 1) {
        const r = items[i];
        if (r.pt != null) continue;
        for (const tn of r.turns || []) {
          if (!tn || !tn[2] || !finite(tn[0]) || !finite(tn[1])) continue;
          const s = r.t + tn[0];
          const e = r.t + tn[1];
          if (e <= a || s >= b || e <= s) continue;
          out.push([s, e, String(tn[2]), e - s]);
        }
      }
    } else {
      const prefix = mode === 'wearers' ? 'tag:' : 'spk:';
      for (const r of this.asr.range(a, b)) {
        const d = r.d > 0 ? r.d : BUCKET;
        for (const [n, s] of this.namedIn(r, mode)) if (s > 0) out.push([r.t, r.t + d, prefix + n, s]);
      }
    }
    out.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    return out;
  }

  /** speaking seconds per entity key in [a, b): voices as the union of their turns, named speakers as bucket time. */
  speakingSeconds(a, b) {
    const mode = this.entityMode();
    const out = new Map();
    if (mode === 'voices') {
      const by = new Map();
      for (const [s, e, k] of this.turns(a, b)) {
        if (!by.has(k)) by.set(k, []);
        by.get(k).push([Math.max(s, a), Math.min(e, b)]);
      }
      for (const [k, iv] of by) out.set(k, unionLength(iv));
    } else {
      for (const [, , k, sec] of this.turns(a, b)) out.set(k, (out.get(k) || 0) + sec);
    }
    return out;
  }

  /** the minutes of [a, b) that hold speech data (the session or the buffer may start inside it). */
  speechMinutes(a, b) {
    const firsts = [this.asr.first, this.tr.first].filter(Boolean).map((r) => r.t);
    if (!firsts.length) return null;
    const start = Math.max(a, Math.min(...firsts));
    return b > start ? (b - start) / 60 : null;
  }

  /**
   * turn switches in [a, b): voices: merged turns starting in the range, a switch when consecutive
   * keys differ and the gap is at most SWITCH_GAP; wearers and named speakers: a name new in a
   * bucket after the last bucket that named anyone (within BRIDGE_SECONDS), per name that dropped out.
   */
  switches(a, b) {
    const mode = this.entityMode();
    let count = 0;
    if (mode === 'voices') {
      const merged = mergeTurns(this.turns(a, b).filter((x) => x[0] >= a).map((x) => [x[0], x[1], x[2]]));
      for (let i = 1; i < merged.length; i += 1) {
        if (merged[i][2] !== merged[i - 1][2] && merged[i][0] - merged[i - 1][1] <= SWITCH_GAP) count += 1;
      }
    } else {
      let last = null;
      let lastEnd = null;
      for (const r of this.asr.range(a, b)) {
        const current = new Set(this.namedIn(r, mode).keys());
        if (!current.size) continue;
        if (last && r.t - lastEnd <= BRIDGE_SECONDS) {
          const gone = [...last].filter((n) => !current.has(n)).length;
          const fresh = [...current].filter((n) => !last.has(n)).length;
          count += gone * fresh;
        }
        last = current;
        lastEnd = r.t + (r.d > 0 ? r.d : BUCKET);
      }
    }
    const minutes = this.speechMinutes(a, b);
    return { count, minutes, perMin: minutes ? count / minutes : null };
  }

  /** IPS windows starting in [a, b) */
  ipsRange(a, b) {
    return this.ips.range(a, b);
  }

  vfaRange(a, b) {
    return this.vfa.range(a, b);
  }

  /** the newest frame set at or before t */
  vfaAt(t) {
    return this.vfa.at(t);
  }

  base() {
    if (finite(this.t0)) return this.t0;
    const o = this.oldestTime();
    return o == null ? 0 : o;
  }

  /**
   * presence per tag per second in [a, b): 1 when an IPS window placed the tag or a camera saw it,
   * 0 when IPS or VFA ran that second without it, null for a second inside the range neither ran
   * (gaps between observed seconds only). {t: [absolute second], byTag: {tag: [1|0|null]}}
   */
  presence(a, b, tags) {
    const base = this.base();
    const slots = new Map();
    const touch = (t) => {
      const k = Math.round(t - base);
      let st = slots.get(k);
      if (!st) {
        st = new Set();
        slots.set(k, st);
      }
      return st;
    };
    for (const r of this.ips.range(a, b)) {
      const st = touch(r.t);
      for (const tag of Object.keys(r.p)) st.add(tag);
    }
    for (const r of this.vfa.range(a, b)) {
      const st = touch(r.t);
      for (const c of r.c) {
        for (const p of c.ps || []) if (p && p.tag != null) st.add(String(p.tag));
        for (const tag of Object.keys(c.tg || {})) st.add(tag);
      }
    }
    const keys = Array.from(slots.keys()).sort((x, y) => x - y);
    const t = [];
    const idx = [];
    for (let i = 0; i < keys.length; i += 1) {
      if (i > 0) for (let k = keys[i - 1] + 1; k < keys[i]; k += 1) {
        t.push(base + k);
        idx.push(null);
      }
      t.push(base + keys[i]);
      idx.push(keys[i]);
    }
    const byTag = {};
    for (const tag of tags || []) byTag[tag] = idx.map((k) => (k == null ? null : slots.get(k).has(tag) ? 1 : 0));
    return { t, byTag };
  }

  /** the category of a tag in a frame set: the first camera (by id) where it has a known gaze; 'unreadable' when seen without one; null when not seen. */
  gazeOf(rec, tag) {
    let seen = false;
    for (const c of rec.c) {
      for (const p of c.ps || []) {
        if (!p || String(p.tag) !== tag) continue;
        seen = true;
        const g = p.g;
        if (g && g.cat && g.cat !== 'unknown') return gazeCategory(g.cat, g.to);
      }
    }
    return seen ? 'unreadable' : null;
  }

  /** {t: [frame set times], byTag: {tag: [category index | null]}} for frame sets in [a, b). */
  gazeSeries(a, b, tags) {
    const recs = this.vfa.range(a, b);
    const out = { t: recs.map((r) => r.t), byTag: {} };
    for (const tag of tags || []) {
      out.byTag[tag] = recs.map((r) => {
        const c = this.gazeOf(r, tag);
        return c == null ? null : CATEGORIES.indexOf(c);
      });
    }
    return out;
  }

  /**
   * social gaze over [a, b), counted as the analysis counts it (report/video.py _category_frames):
   * camera frames with the gaze on a partner's face or hands over every camera frame a pupil was
   * seen in, unreadable ones included, all pupils pooled.
   */
  socialGaze(a, b, tags) {
    const set = new Set(tags || []);
    let social = 0;
    let frames = 0;
    for (const r of this.vfa.range(a, b)) {
      for (const c of r.c) {
        for (const p of c.ps || []) {
          const tag = p && p.tag != null ? String(p.tag) : null;
          if (!tag || !set.has(tag)) continue;
          frames += 1;
          const g = p.g;
          const cat = g && g.cat ? gazeCategory(g.cat, g.to) : 'unreadable';
          if (cat === 'partner_face' || cat === 'partner_hands') social += 1;
        }
      }
    }
    return { social, frames, ratio: frames ? social / frames : null };
  }

  /**
   * who looks at whom in [a, b): per looker {seen: frame sets it was seen in, face, hands, any:
   * Map target -> frame sets}, deduplicated per (frame set, looker, target, kind); targets are
   * pupils of `tags` or 'other'.
   */
  looks(a, b, tags) {
    const set = new Set(tags || []);
    const out = new Map();
    for (const tag of set) out.set(tag, { seen: 0, face: new Map(), hands: new Map(), any: new Map() });
    const inc = (m, k) => m.set(k, (m.get(k) || 0) + 1);
    for (const r of this.vfa.range(a, b)) {
      const seen = new Set();
      const hits = new Map();
      for (const c of r.c) {
        for (const p of c.ps || []) {
          const tag = p && p.tag != null ? String(p.tag) : null;
          if (!tag || !set.has(tag)) continue;
          seen.add(tag);
          const g = p.g;
          const kind = g && g.cat === 'partner_face' ? 'face' : g && g.cat === 'partner_hands' ? 'hands' : null;
          if (!kind) continue;
          const target = g.to != null && set.has(String(g.to)) && String(g.to) !== tag ? String(g.to) : 'other';
          hits.set(`${tag}\u0000${target}\u0000${kind}`, [tag, target, kind]);
        }
      }
      for (const tag of seen) out.get(tag).seen += 1;
      const any = new Map();
      for (const [l, tg, kind] of hits.values()) {
        inc(out.get(l)[kind], tg);
        any.set(`${l}\u0000${tg}`, [l, tg]);
      }
      for (const [l, tg] of any.values()) inc(out.get(l).any, tg);
    }
    return out;
  }

  /** 3D-equivalent floor distance per pair per IPS window in [a, b): Map "a|b" -> {t: [], d: []}. */
  pairDistances(a, b, tags) {
    const pairs = pairList(tags);
    const out = new Map(pairs.map(([, , k]) => [k, { t: [], d: [] }]));
    for (const r of this.ips.range(a, b)) {
      for (const [x, y, k] of pairs) {
        const p = r.p[x];
        const q = r.p[y];
        if (!p || !q) continue;
        const d = Math.hypot(p[0] - q[0], p[1] - q[1], (p[2] || 0) - (q[2] || 0));
        if (!finite(d)) continue;
        out.get(k).t.push(r.t);
        out.get(k).d.push(d);
      }
    }
    return out;
  }

  /** per pair the distance in the newest window (at most `maxAge` s before `now`) that holds both. */
  distanceNow(now, tags, maxAge = 10) {
    const pairs = pairList(tags);
    const out = new Map(pairs.map(([, , k]) => [k, null]));
    const items = this.ips.items;
    let left = pairs.length;
    for (let i = this.ips.upper(now) - 1; i >= 0 && left > 0; i -= 1) {
      const r = items[i];
      if (r.t < now - maxAge) break;
      for (const [x, y, k] of pairs) {
        if (out.get(k) != null || !r.p[x] || !r.p[y]) continue;
        const p = r.p[x];
        const q = r.p[y];
        out.set(k, { d: Math.hypot(p[0] - q[0], p[1] - q[1], (p[2] || 0) - (q[2] || 0)), age: now - r.t });
        left -= 1;
      }
    }
    return out;
  }

  /** the newest position of each tag within `maxAge` s before `now`: Map tag -> {u, v, h, heading, age, t}. */
  latestPositions(now, tags, maxAge = 30) {
    const out = new Map();
    const want = new Set(tags || []);
    const items = this.ips.items;
    for (let i = this.ips.upper(now) - 1; i >= 0 && out.size < want.size; i -= 1) {
      const r = items[i];
      if (r.t < now - maxAge) break;
      for (const tag of want) {
        if (out.has(tag) || !r.p[tag]) continue;
        const p = r.p[tag];
        out.set(tag, { u: p[0], v: p[1], h: p[2], heading: finite(r.hd[tag]) ? r.hd[tag] : null, age: Math.max(0, now - r.t), t: r.t });
      }
    }
    return out;
  }

  /**
   * {tag: [[u, v] | null, ...]} of the windows in [now - seconds, now], smoothed by a rolling median of 3
   * within runs (gap <= 1.5 s), as the space part's tracks are: one misread window would otherwise
   * draw a spike across the room.
   */
  trails(now, tags, seconds = 30) {
    const out = {};
    for (const tag of tags || []) {
      const t = [];
      const u = [];
      const v = [];
      for (const r of this.ips.range(now - seconds, now + 1e-6)) {
        const p = r.p[tag];
        if (!p) continue;
        t.push(r.t);
        u.push(p[0]);
        v.push(p[1]);
      }
      out[tag] = smoothRun(t, u, v);
    }
    return out;
  }

  /** facing edges of the newest IPS window within `maxAge` s: [{from, to, mutual}] (a mutual pair once). */
  facingEdges(now, tags, maxAge = 5) {
    const r = this.ips.at(now);
    if (!r || r.t < now - maxAge) return [];
    const set = new Set(tags || []);
    const dirs = new Set(r.f.filter((e) => Array.isArray(e) && e.length >= 2).map((e) => `${e[0]}|${e[1]}`));
    const out = [];
    const done = new Set();
    for (const e of r.f) {
      if (!Array.isArray(e) || e.length < 2) continue;
      const from = String(e[0]);
      const to = String(e[1]);
      if (!set.has(from) || !set.has(to) || from === to) continue;
      const mutual = dirs.has(`${to}|${from}`);
      const key = mutual ? pairKey(from, to) : `${from}>${to}`;
      if (done.has(key)) continue;
      done.add(key);
      out.push({ from, to, mutual });
    }
    return out;
  }

  /** per (camera, tag) the gaze points of frame sets in [a, b): Map "cam\0tag" -> {t, x, y, w}. */
  gazeIndex(a, b, tags) {
    const set = new Set(tags || []);
    const index = new Map();
    for (const r of this.vfa.range(a, b)) {
      for (const c of r.c) {
        const w = Number(c.w) || 0;
        if (!w) continue;
        for (const p of c.ps || []) {
          const tag = p && p.tag != null ? String(p.tag) : null;
          if (!tag || !set.has(tag) || !p.g || !p.g.p || NO_POINT.has(p.g.cat || 'unknown')) continue;
          const key = `${c.id}\u0000${tag}`;
          let e = index.get(key);
          if (!e) {
            e = { t: [], x: [], y: [], w: [] };
            index.set(key, e);
          }
          e.t.push(r.t);
          e.x.push(p.g.p[0]);
          e.y.push(p.g.p[1]);
          e.w.push(w);
        }
      }
    }
    return index;
  }

  /**
   * joint attention of each pair in [a, b): ratio = frames (per camera) whose gaze distance is at
   * most JA_WIDTH of the frame's width, over the frames that measured it; baseline = the same test
   * between one pupil's gaze at t and the other's nearest t - 20/30/40 s (within JA_SLACK, same
   * camera, both directions), null under JA_MIN comparisons; excess = ratio - baseline.
   * `index` is a gazeIndex covering at least [a - 41, b).
   */
  jointAttention(a, b, tags, index = null) {
    const pairs = pairList(tags);
    const out = new Map(pairs.map(([x, y, k]) => [k, { a: x, b: y, frames: 0, hits: 0, comparisons: 0, baseHits: 0 }]));
    for (const r of this.vfa.range(a, b)) {
      for (const c of r.c) {
        const pr = c.pr || {};
        for (const [, , k] of pairs) {
          const v = pr[k];
          if (!v || !finite(v[0])) continue;
          const o = out.get(k);
          o.frames += 1;
          if (v[0] <= JA_WIDTH) o.hits += 1;
        }
      }
    }
    const idx = index || this.gazeIndex(a - Math.max(...JA_LAGS) - 1, b, tags);
    const cams = new Set();
    for (const key of idx.keys()) cams.add(key.split('\u0000')[0]);
    for (const [x, y, k] of pairs) {
      const o = out.get(k);
      for (const cam of cams) {
        for (const [p, q] of [[x, y], [y, x]]) {
          const src = idx.get(`${cam}\u0000${p}`);
          const oth = idx.get(`${cam}\u0000${q}`);
          if (!src || !oth || !oth.t.length) continue;
          const i0 = lowerIdx(src.t, a);
          const i1 = lowerIdx(src.t, b);
          for (let i = i0; i < i1; i += 1) {
            for (const lag of JA_LAGS) {
              const m = src.t[i] - lag;
              const j = nearestIdx(oth.t, m);
              if (j < 0 || Math.abs(oth.t[j] - m) > JA_SLACK) continue;
              o.comparisons += 1;
              if (Math.hypot(src.x[i] - oth.x[j], src.y[i] - oth.y[j]) <= JA_WIDTH * src.w[i]) o.baseHits += 1;
            }
          }
        }
      }
      o.ratio = o.frames ? o.hits / o.frames : null;
      o.baseline = o.comparisons >= JA_MIN ? o.baseHits / o.comparisons : null;
      o.excess = o.ratio != null && o.baseline != null ? o.ratio - o.baseline : null;
    }
    return out;
  }

  /**
   * the six headline numbers over [a, b): speech activity, turn switches per minute, speaking
   * balance, median pair distance, joint attention excess (pairs seen together in at least
   * JA_PAIR_MIN_SHARE frames per second of the span, weighted by their frames; jaPairs counted, jaFew
   * left out), social gaze.
   */
  metrics(a, b, tags, index = null) {
    const secs = this.speakingSeconds(a, b);
    const individual = [];
    for (const [k, s] of secs) if (k !== 'v:?') individual.push(s);
    const dists = [];
    for (const { d } of this.pairDistances(a, b, tags).values()) dists.push(...d);
    let jaSum = 0;
    let jaFrames = 0;
    let jaPairs = 0;
    let jaFew = 0;
    const jaMinFrames = Math.max(JA_MIN, JA_PAIR_MIN_SHARE * (b - a));
    for (const o of this.jointAttention(a, b, tags, index).values()) {
      if (o.excess == null) continue;
      if (o.frames < jaMinFrames) {
        // a pair seen together in a few frames only would swing the headline as much as a well seen one
        jaFew += 1;
        continue;
      }
      jaSum += o.excess * o.frames;
      jaFrames += o.frames;
      jaPairs += 1;
    }
    return {
      speech: this.speechRatio(a, b),
      switches: this.switches(a, b).perMin,
      balance: entropyBalance(individual),
      speakers: individual.filter((s) => s > 0).length,
      distance: median(dists),
      ja: jaFrames ? jaSum / jaFrames : null,
      jaPairs,
      jaFew,
      social: this.socialGaze(a, b, tags).ratio,
    };
  }

  /** metrics over the trailing window plus one value per minute over the last `minutes` minutes (sparklines). */
  kpis(now, windowSeconds, tags, minutes = 10) {
    const span = Math.max(windowSeconds, minutes * 60);
    const index = this.gazeIndex(now - span - Math.max(...JA_LAGS) - 1, now + 1e-6, tags);
    const main = this.metrics(now - windowSeconds, now + 1e-6, tags, index);
    const spark = { speech: [], switches: [], balance: [], distance: [], ja: [], social: [] };
    const oldest = this.oldestTime();
    for (let i = minutes - 1; i >= 0; i -= 1) {
      const a = now - (i + 1) * 60;
      const b = now - i * 60 + (i === 0 ? 1e-6 : 0);
      const m = oldest != null && b > oldest ? this.metrics(a, b, tags, index) : null;
      for (const key of Object.keys(spark)) spark[key].push(m ? m[key] : null);
    }
    return { ...main, spark };
  }

  /**
   * one-line health per modality: newest record time (transcripts: newest chunk end), the cameras of
   * the newest frame set, and the tags seen (IPS or VFA) in the last `recent` seconds before `now`.
   */
  health(now, recent = 10) {
    const vfa = this.vfa.at(now);
    const seenTags = new Set();
    for (const r of this.ips.range(now - recent, now + 1e-6)) for (const tag of Object.keys(r.p)) seenTags.add(tag);
    for (const r of this.vfa.range(now - recent, now + 1e-6)) {
      for (const c of r.c) for (const p of c.ps || []) if (p && p.tag != null) seenTags.add(String(p.tag));
    }
    const asrLast = this.asr.at(now);
    const ipsLast = this.ips.at(now);
    return {
      asr: asrLast ? asrLast.t : null,
      tr: this.newestChunkEnd(),
      ips: ipsLast ? ipsLast.t : null,
      vfa: vfa ? vfa.t : null,
      cameras: { seen: vfa ? vfa.c.length : 0, total: Math.max(this.maxCameras, this.cameraIds.size) },
      tags: sortTags(Array.from(seenTags).filter(isPupilTag)),
    };
  }

  /** true when recognition heard speech after the newest transcript chunk ended (within `reach` s before now). */
  transcribing(now, reach = 60) {
    if (!this.asr.items.length) return false;
    const end = this.newestChunkEnd();
    const from = Math.max(now - reach, end == null ? -Infinity : end);
    const act = this.speechActivity(from, now + 1e-6);
    return act.v.some((v) => finite(v) && v > 0.15);
  }

  /** transcript chunks ending at or before `now`, by start. */
  chunks(now = Infinity) {
    return this.tr.items.filter((r) => (finite(r.e) ? r.e : r.t) <= now + 1e-6);
  }
}

export const VOICE_SLOTS = 5;

/**
 * Colour slots of the voices (and named speakers) of one page, kept for the life of the page.
 *
 * The analysis colours a voice by its whole-session speaking rank (the speech part's entities: slot 1
 * to 5, the rest in Other voices), so a ready speech part seeds the slots here and the same voice has
 * the same colour on both pages. Without one, voices take slots in the order the stream shows them.
 * A voice keeps the colour it was first drawn in: seeding later only fills voices not drawn yet, and a
 * voice not in the speech part takes the next free slot, or Other voices once all five are taken.
 * Beside worn microphones the analysis colours wearers, not voices, so a speech part of a wearer
 * mode folds every voice not drawn yet into Other voices.
 *
 * While `pending` (the speech part was asked for and has not answered), get() hands out a neutral
 * colour without assigning, so nothing is fixed before the answer.
 */
export class VoiceColors {
  constructor() {
    this.slots = new Map();
    this.pending = false;
    this.fold = false;
    this.seeded = false;
    // bumps on every new assignment, so renderers know to redraw
    this.version = 0;
    // a neutral colour was handed out while pending: what was drawn with it needs a redraw
    this.provisional = false;
  }

  usedSlots() {
    const used = new Set();
    for (const s of this.slots.values()) if (s) used.add(s);
    return used;
  }

  set(key, slot) {
    this.slots.set(key, slot);
    this.version += 1;
  }

  /**
   * take the slots of a ready speech part (`data` of /report/speech) for keys not assigned yet;
   * returns how many keys it assigned.
   */
  seed(speech) {
    if (!speech || typeof speech !== 'object') return 0;
    const before = this.slots.size;
    const mode = speech.mode || null;
    const ents = Array.isArray(speech.entities) ? speech.entities : [];
    const take = (key, slot) => {
      if (!key || this.slots.has(key)) return;
      if (!slot) {
        this.set(key, 0);
        return;
      }
      // a slot a voice drawn earlier already holds is not taken twice: this voice gets the next free one later
      if (!this.usedSlots().has(slot)) this.set(key, slot);
    };
    if (mode === 'wearer' || mode === 'wearer+group') this.fold = true;
    const voices = ents.filter((e) => e && typeof e.key === 'string' && e.key.startsWith('v:'));
    for (const e of voices) take(e.key, Number.isInteger(e.slot) && e.slot >= 1 && e.slot <= VOICE_SLOTS ? e.slot : 0);
    // named speakers carry no slot: the analysis numbers them in entity order
    const named = ents.filter((e) => e && typeof e.key === 'string' && e.key.startsWith('spk:'));
    named.forEach((e, i) => take(e.key, i + 1 <= VOICE_SLOTS ? i + 1 : 0));
    const members = speech.other && Array.isArray(speech.other.members) ? speech.other.members : [];
    for (const k of members) if (typeof k === 'string' && k !== 'v:?') take(k, 0);
    this.seeded = true;
    return this.slots.size - before;
  }

  has(key) {
    return this.slots.has(String(key));
  }

  /** the slot of a key (1 to 5, or 0 for Other voices), assigning one when the key is new. */
  assign(key) {
    const k = String(key);
    if (this.slots.has(k)) return this.slots.get(k);
    let slot = 0;
    if (!(this.fold && k.startsWith('v:'))) {
      const used = this.usedSlots();
      for (let i = 1; i <= VOICE_SLOTS; i += 1) {
        if (!used.has(i)) {
          slot = i;
          break;
        }
      }
    }
    this.set(k, slot);
    return slot;
  }

  /** {key, slot, color, label, other}; Other voices and Unlinked always take the neutral colour. */
  get(key) {
    const k = String(key ?? '');
    const other = (label) => ({ key: k, slot: null, color: 'var(--voice-other)', label, other: true });
    if (k === 'other') return other('Other voices');
    if (!k || k === 'v:?') return other(voiceLabel(k || 'v:?'));
    if (!this.slots.has(k) && this.pending) {
      this.provisional = true;
      return { ...other(voiceLabel(k)), provisional: true };
    }
    const slot = this.assign(k);
    if (!slot) return other(voiceLabel(k));
    return { key: k, slot, color: `var(--voice-${slot})`, label: voiceLabel(k), other: false };
  }

  /** the keys with a colour slot, in slot order. */
  slotted(keys) {
    const out = [];
    for (const k of keys) {
      const v = this.get(k);
      if (v.slot) out.push([v.slot, k]);
    }
    return out.sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  }
}

const med3 = (a, b, c) => Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));

/**
 * [[u, v] | null]: each sample replaced by the median of itself and its neighbours in the same run
 * (gap <= 1.5 s), a null between runs so a drawn trail breaks where the readings stop.
 */
export function smoothRun(t, u, v, gap = 1.5) {
  const out = [];
  for (let i = 0; i < t.length; i += 1) {
    const prev = i > 0 && t[i] - t[i - 1] <= gap;
    const next = i < t.length - 1 && t[i + 1] - t[i] <= gap;
    if (i > 0 && !prev) out.push(null);
    if (prev && next) out.push([med3(u[i - 1], u[i], u[i + 1]), med3(v[i - 1], v[i], v[i + 1])]);
    else out.push([u[i], v[i]]);
  }
  return out;
}

function lowerIdx(arr, x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** index of the value nearest x (the earlier on a tie); -1 when empty */
function nearestIdx(arr, x) {
  if (!arr.length) return -1;
  const i = lowerIdx(arr, x);
  if (i === 0) return 0;
  if (i === arr.length) return i - 1;
  return x - arr[i - 1] <= arr[i] - x ? i - 1 : i;
}
