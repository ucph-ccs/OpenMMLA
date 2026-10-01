/**
 * The sections of the analysis page and the range arithmetic behind them. The report parts carry
 * session totals plus per-window, per-minute and per-turn arrays; every panel answers to one time
 * range, so the aggregates are recomputed here from those arrays (the derive functions are pure and
 * have no DOM, so they can be checked against the server's totals). Without a range the server's
 * own totals are shown wherever its rule cannot be rebuilt exactly from the arrays.
 */

import { h, clear, fmt, icon, sortTags, voiceLabel, segmented, emptyState, downloadLink } from './core.js';
import {
  card, legend, timeline, barList, stackedBars, heatMatrix, lineChart, histogram, networkGraph, statTile,
  bullet, scaleLegend, seqColor,
} from './charts.js';
import { floorMap } from './floormap.js';

const NBSP = ' ';
const SWITCH_GAP = 2.0;
const BRIDGE_SECONDS = 9.0;
const GAP_TOLERANCE = 0.5;
const TRAIL_GAP = 1.5;
const FEW_WINDOWS = 10;

export const CATEGORIES = [
  { key: 'partner_face', label: 'Partner face', fill: 'var(--ord-4)' },
  { key: 'partner_hands', label: 'Partner hands', fill: 'var(--ord-3)' },
  { key: 'other_people', label: 'Other people', fill: 'var(--ord-2)' },
  { key: 'task', label: 'Task', fill: 'var(--ord-1)' },
  { key: 'elsewhere', label: 'Elsewhere', fill: 'var(--grid)' },
  { key: 'unreadable', label: 'Unreadable', fill: 'hatch' },
];

export const STATES = [
  { key: '0', label: 'Individual', fill: 'var(--ord-1)' },
  { key: '1', label: 'Social', fill: 'var(--ord-2)' },
  { key: '2', label: 'Collaborative', fill: 'var(--ord-4)' },
  { key: '-2', label: 'Absent', fill: 'hatch' },
];
const STATE_FILL = { 0: 'var(--ord-1)', 1: 'var(--ord-2)', 2: 'var(--ord-4)', '-2': 'hatch' };
const STATE_LABEL = { 0: 'Individual', 1: 'Social', 2: 'Collaborative', '-2': 'Absent' };

// pairs are relations between two tags: drawn as small multiples in ink, never in a hue of their own
const PAIR_INK = 'var(--ink-2)';

// numbers

function finite(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function sum(arr) {
  let s = 0;
  for (const v of arr) if (finite(v)) s += v;
  return s;
}

function median(values) {
  const v = values.filter(finite).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

function mean(values) {
  const v = values.filter(finite);
  return v.length ? sum(v) / v.length : null;
}

function entropyBalance(seconds) {
  const pos = seconds.filter((s) => finite(s) && s > 0);
  if (pos.length < 2) return null;
  const total = sum(pos);
  let e = 0;
  for (const s of pos) e -= (s / total) * Math.log(s / total);
  return e / Math.log(pos.length);
}

function histogramCounts(values, edges) {
  const counts = new Array(edges.length + 1).fill(0);
  for (const v of values) {
    let i = 0;
    while (i < edges.length && v + 1e-9 >= edges[i]) i += 1;
    counts[i] += 1;
  }
  return counts;
}

function unionLength(intervals) {
  const sorted = intervals.filter((iv) => iv[1] > iv[0]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let total = 0;
  let cs = null;
  let ce = null;
  for (const [a, b] of sorted) {
    if (cs != null && a <= ce) ce = Math.max(ce, b);
    else {
      if (cs != null) total += ce - cs;
      cs = a;
      ce = b;
    }
  }
  if (cs != null) total += ce - cs;
  return total;
}

function coveredTwice(intervals) {
  const edges = [];
  for (const [a, b] of intervals) {
    if (b > a) {
      edges.push([a, 1]);
      edges.push([b, -1]);
    }
  }
  edges.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  let total = 0;
  let depth = 0;
  let last = null;
  for (const [m, c] of edges) {
    if (last != null && depth >= 2) total += m - last;
    depth += c;
    last = m;
  }
  return total;
}

// ranges

/** [a, b] clipped to the span, at least 5 s long, or null. */
export function normalizeRange(range, span) {
  if (!range || !finite(range[0]) || !finite(range[1])) return null;
  let a = Math.min(range[0], range[1]);
  let b = Math.max(range[0], range[1]);
  if (span) {
    a = Math.max(span[0], a);
    b = Math.min(span[1], b);
  }
  if (!(b - a >= 1)) return null;
  return [Math.round(a * 10) / 10, Math.round(b * 10) / 10];
}

/** "#range=720-1470" -> [720, 1470] */
export function parseRangeHash(hash) {
  const m = /(?:^#|&)range=(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)/.exec(String(hash || ''));
  return m ? [Number(m[1]), Number(m[2])] : null;
}

export function rangeHash(range) {
  if (!range) return '';
  const f = (v) => String(Math.round(v * 10) / 10);
  return `#range=${f(range[0])}-${f(range[1])}`;
}

function rangeLength(range, span) {
  if (range) return range[1] - range[0];
  return span ? span[1] - span[0] : 0;
}

/**
 * indices of fixed-step windows (window starts t) whose midpoint lies in the range; a window cut by
 * the session end (end) has its midpoint inside the session, so a whole-session range keeps it.
 */
export function windowIndices(t, step, range, end = Infinity) {
  const out = [];
  if (!Array.isArray(t)) return out;
  for (let i = 0; i < t.length; i += 1) {
    if (!range) {
      out.push(i);
      continue;
    }
    const e = Math.max(t[i], Math.min(t[i] + step, finite(end) ? end : Infinity));
    const mid = (t[i] + e) / 2;
    if (mid >= range[0] && mid <= range[1]) out.push(i);
  }
  return out;
}

/** weight of each minute bin inside the range (1 for whole minutes, a fraction at the edges); the
 * last bin ends with the session, so its weight is relative to its own length. */
function minuteWeights(t, range, end = Infinity) {
  return t.map((m) => {
    if (!range) return 1;
    const e = Math.max(m + 1e-6, Math.min(m + 60, end));
    return Math.max(0, Math.min(e, range[1]) - Math.max(m, range[0])) / (e - m);
  });
}

function clippedUnion(intervals, a, b) {
  const out = [];
  for (const [s, e] of intervals) {
    const x = Math.max(s, a);
    const y = Math.min(e, b);
    if (y > x) out.push([x, y]);
  }
  return unionLength(out);
}

/**
 * The amount of a per-minute series inside the range: whole minutes as they are, a minute cut by the
 * range in proportion to the part of the key's turns (shape) that falls inside it, or to time when
 * the key has no turns there.
 */
function minuteSeriesInRange(series, pmT, range, end, shape) {
  if (!series) return 0;
  const w = minuteWeights(pmT, range, end);
  let total = 0;
  for (let i = 0; i < pmT.length; i += 1) {
    const v = series[i];
    if (!finite(v) || !(w[i] > 0)) continue;
    if (w[i] >= 1 - 1e-9) {
      total += v;
      continue;
    }
    const m = pmT[i];
    const me = Math.min(m + 60, end);
    const full = shape ? clippedUnion(shape, m, me) : 0;
    if (full > 0) total += v * (clippedUnion(shape, Math.max(m, range[0]), Math.min(me, range[1])) / full);
    else total += v * w[i];
  }
  return total;
}

// speech

/** session-constant helpers of a speech part: key folding, merged turns, named buckets. */
export function speechModel(speech) {
  const mode = speech.mode;
  const named = mode === 'wearer' || mode === 'wearer+group' || mode === 'individual';
  const entities = speech.entities || [];
  const entityKeys = entities.map((e) => e.key);
  const entitySet = new Set(entityKeys);
  const other = speech.other || null;
  const keys = entityKeys.concat(other ? ['other'] : []);
  const fold = (key) => (entitySet.has(key) ? key : 'other');
  const turns = (speech.turns || []).filter((tr) => Array.isArray(tr) && finite(tr[0]) && finite(tr[1]));
  const activity = speech.activity || { t: [], v: [], step: 3 };
  const step = activity.step || 3;
  let namedBuckets = null;
  if (named) {
    // a bucket names everyone whose run of contiguous buckets covers its start
    namedBuckets = activity.t.map(() => new Set());
    for (const [s, e, key] of turns) {
      let i = lowerBound(activity.t, s - 1e-6);
      for (; i < activity.t.length && activity.t[i] < e - 1e-6; i += 1) namedBuckets[i].add(key);
    }
  }
  const wordClasses = (speech.transcript || []).some((c) => (c.words || []).some((w) => w[4] != null));
  return { speech, mode, named, entities, keys, other, fold, turns, activity, step, namedBuckets, wordClasses };
}

function lowerBound(arr, x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** every speech aggregate over a range (null = whole session, then the server's totals where exact). */
export function speechStats(model, range) {
  const { speech, mode, named, keys, fold, turns, activity, step } = model;
  const kp = speech.kpis || {};
  const span = [0, speech.duration || 0];
  const inStart = (t) => !range || (t >= range[0] && t < range[1]);

  // activity buckets: voiced fraction, silence, pauses
  const idx = [];
  for (let i = 0; i < activity.t.length; i += 1) if (inStart(activity.t[i])) idx.push(i);
  let voiced = 0;
  let readable = 0;
  let silent = 0;
  const pauses = [];
  let run = 0;
  let prev = null;
  for (const i of idx) {
    const v = activity.v[i];
    const t = activity.t[i];
    if (finite(v)) {
      voiced += v;
      readable += 1;
      if (v === 0) silent += 1;
    }
    const isSilent = v === 0;
    const contiguous = prev != null && t - prev <= step + GAP_TOLERANCE;
    if (isSilent && (contiguous || run === 0)) run += 1;
    else {
      if (run) pauses.push(run * step);
      run = isSilent ? 1 : 0;
    }
    prev = t;
  }
  if (run) pauses.push(run * step);
  const speechRatio = readable ? voiced / readable : null;
  const silenceRatio = readable ? silent / readable : null;
  const voicedSeconds = voiced * step;

  // turns in the range (counted by their start)
  const rTurns = turns.filter((tr) => inStart(tr[0]));
  const turnLengths = rTurns.map((tr) => tr[1] - tr[0]);

  // speaking seconds per entity
  const seconds = {};
  const rawSeconds = {};
  for (const k of keys) seconds[k] = 0;
  if (!range) {
    for (const e of model.entities) seconds[e.key] = e.seconds || 0;
    if (model.other) seconds.other = model.other.seconds || 0;
  } else {
    const pm = speech.per_minute || { t: [], by_entity: {} };
    const shapes = new Map();
    for (const [s, e, key] of turns) {
      const f = fold(key);
      if (!shapes.has(f)) shapes.set(f, []);
      shapes.get(f).push([s, e]);
    }
    for (const k of keys) seconds[k] = minuteSeriesInRange((pm.by_entity || {})[k], pm.t, range, span[1], shapes.get(k));
    if (!named) {
      // balance counts every voice apart, folded or not: their turns clipped to the range; the
      // unlinked pool mixes many speakers, so it stays out (as on the server and the live page)
      const per = new Map();
      for (const [s, e, key] of turns) {
        if (key === 'v:?') continue;
        const a = Math.max(s, range[0]);
        const b = Math.min(e, range[1]);
        if (b <= a) continue;
        if (!per.has(key)) per.set(key, []);
        per.get(key).push([a, b]);
      }
      for (const [key, spans] of per) rawSeconds[key] = unionLength(spans);
    }
  }
  const total = sum(Object.values(seconds));

  // turns and words per entity
  const turnCount = {};
  const turnSeconds = {};
  for (const k of keys) {
    turnCount[k] = 0;
    turnSeconds[k] = 0;
  }
  for (const [s, e, key] of rTurns) {
    const f = fold(key);
    turnCount[f] = (turnCount[f] || 0) + 1;
    turnSeconds[f] = (turnSeconds[f] || 0) + (e - s);
  }
  const words = {};
  for (const k of keys) words[k] = 0;
  if (!range) {
    for (const e of model.entities) words[e.key] = e.words || 0;
    if (model.other) words.other = model.other.words || 0;
  } else if (mode === 'group') {
    for (const c of speech.transcript || []) {
      if (c.lane !== 'group' || c.t1 < range[0] || c.t0 > range[1]) continue;
      for (const w of c.words || []) {
        if (w[3] != null && inStart(w[1])) {
          const f = fold(w[3]);
          words[f] = (words[f] || 0) + 1;
        }
      }
    }
  } else if (mode === 'individual') {
    for (const c of speech.transcript || []) {
      if (c.lane !== 'group' || c.t1 < range[0] || c.t0 > range[1]) continue;
      const k = fold(`spk:${c.speaker}`);
      const n = (c.words || []).length ? c.words.filter((w) => inStart(w[1])).length : (inStart(c.t0) ? String(c.text || '').split(/\s+/).filter(Boolean).length : 0);
      words[k] = (words[k] || 0) + n;
    }
  } else {
    for (const e of model.entities) {
      if (model.wordClasses) {
        let n = 0;
        for (const c of speech.transcript || []) {
          if (c.lane !== e.key || c.t1 < range[0] || c.t0 > range[1]) continue;
          for (const w of c.words || []) if (w[4] === 'wearer' && inStart(w[1])) n += 1;
        }
        words[e.key] = n;
      } else {
        words[e.key] = e.seconds ? Math.round((e.words || 0) * (seconds[e.key] / e.seconds)) : 0;
      }
    }
  }

  // who follows whom
  const tkeys = (speech.transitions && speech.transitions.keys) || [];
  const tindex = new Map(tkeys.map((k, i) => [k, i]));
  let counts = tkeys.map(() => tkeys.map(() => 0));
  let switches = 0;
  let overlap = null;
  if (!named) {
    if (!tkeys.length) {
      switches = null;
      counts = [];
    } else {
      for (let i = 1; i < rTurns.length; i += 1) {
        const x = rTurns[i - 1];
        const y = rTurns[i];
        if (x[2] !== y[2] && y[0] - x[1] <= SWITCH_GAP) {
          switches += 1;
          const a = tindex.get(fold(x[2]));
          const b = tindex.get(fold(y[2]));
          if (a != null && b != null) counts[a][b] += 1;
        }
      }
    }
    const clipped = [];
    for (const [s, e] of turns) {
      const a = range ? Math.max(s, range[0]) : s;
      const b = range ? Math.min(e, range[1]) : e;
      if (b > a) clipped.push([a, b]);
    }
    const spoken = unionLength(clipped);
    overlap = range ? (spoken ? coveredTwice(clipped) / spoken : null) : kp.overlap_ratio ?? null;
  } else {
    const nb = model.namedBuckets;
    let last = null;
    let lastEnd = null;
    let withOne = 0;
    let withTwo = 0;
    for (const i of idx) {
      const cur = nb[i];
      if (!cur || !cur.size) continue;
      withOne += 1;
      if (cur.size >= 2) withTwo += 1;
      const start = activity.t[i];
      if (last && start - lastEnd <= BRIDGE_SECONDS) {
        for (const a of last) {
          if (cur.has(a)) continue;
          for (const b of cur) {
            if (last.has(b)) continue;
            const ia = tindex.get(a);
            const ib = tindex.get(b);
            if (ia != null && ib != null) counts[ia][ib] += 1;
            switches += 1;
          }
        }
      }
      last = cur;
      lastEnd = start + step;
    }
    overlap = withOne ? withTwo / withOne : null;
  }
  const minutes = rangeLength(range, span) / 60;
  const switchesPerMin = switches == null || !(minutes > 0) ? null : switches / minutes;

  // words in the range, and the speech rate
  let wordsTotal;
  if (!range) wordsTotal = kp.words ?? null;
  else {
    const pm = speech.per_minute || { t: [], words: [] };
    const w = minuteWeights(pm.t, range, span[1]);
    wordsTotal = 0;
    for (let i = 0; i < pm.t.length; i += 1) if (finite(pm.words[i])) wordsTotal += pm.words[i] * w[i];
    wordsTotal = Math.round(wordsTotal);
  }
  const wpm = mode === 'wearer' || !(voicedSeconds > 0) || wordsTotal == null ? null : wordsTotal / (voicedSeconds / 60);

  // balance over every individual voice or wearer (not folded)
  let balance;
  if (!range) balance = kp.balance ?? null;
  else if (!named) balance = entropyBalance(Object.values(rawSeconds));
  else balance = entropyBalance(model.entities.map((e) => seconds[e.key]));

  const entityRows = keys.map((k) => {
    const ent = k === 'other' ? model.other : model.entities.find((e) => e.key === k);
    const sec = seconds[k] || 0;
    return {
      key: k,
      label: ent ? ent.label : voiceLabel(k),
      kind: ent && ent.kind ? ent.kind : 'voice',
      seconds: sec,
      share: total > 0 ? sec / total : null,
      turns: turnCount[k] || 0,
      meanTurn: turnCount[k] ? turnSeconds[k] / turnCount[k] : null,
      words: words[k] || 0,
      wpm: sec > 0 && (ent ? ent.kind === 'voice' || ent.kind === 'speaker' : true) ? (words[k] || 0) / (sec / 60) : null,
    };
  });

  return {
    range,
    speechRatio: range ? speechRatio : kp.speech_ratio ?? speechRatio,
    silenceRatio: range ? silenceRatio : kp.silence_ratio ?? silenceRatio,
    overlapRatio: overlap,
    switches,
    switchesPerMin: range ? switchesPerMin : kp.switches_per_min ?? switchesPerMin,
    turns: rTurns.length,
    meanTurn: turnLengths.length ? sum(turnLengths) / turnLengths.length : null,
    meanPause: pauses.length ? sum(pauses) / pauses.length : null,
    // without a range the server's histograms and matrix: the turns here are rounded to 0.01 s,
    // which moves a few turns sitting on a bin edge or the switch gap
    pauses: !range && speech.pauses ? speech.pauses.counts : histogramCounts(pauses, (speech.pauses && speech.pauses.edges) || []),
    turnLengths: !range && speech.turn_lengths ? speech.turn_lengths.counts : histogramCounts(turnLengths, (speech.turn_lengths && speech.turn_lengths.edges) || []),
    words: wordsTotal,
    wpm: range ? wpm : kp.wpm ?? wpm,
    balance,
    entities: entityRows,
    totalSeconds: total,
    transitions: !range && speech.transitions ? speech.transitions : { keys: tkeys, counts },
    voicedSeconds,
    readable,
  };
}


/** turn switches per minute bin (for sparklines). */
export function switchesPerMinute(model) {
  const pm = model.speech.per_minute || { t: [] };
  const counts = pm.t.map(() => 0);
  const bin = (t) => Math.min(counts.length - 1, Math.max(0, Math.floor(t / 60)));
  if (!counts.length) return counts;
  if (model.named) {
    // the wearers' rule: a name new in a bucket follows the last named bucket's names
    const { activity, step, namedBuckets } = model;
    let last = null;
    let lastEnd = null;
    for (let i = 0; i < activity.t.length; i += 1) {
      const cur = namedBuckets[i];
      if (!cur || !cur.size) continue;
      const start = activity.t[i];
      if (last && start - lastEnd <= BRIDGE_SECONDS) {
        let n = 0;
        for (const a of last) if (!cur.has(a)) for (const b of cur) if (!last.has(b)) n += 1;
        counts[bin(start)] += n;
      }
      last = cur;
      lastEnd = start + step;
    }
    return counts;
  }
  const turns = model.turns;
  for (let i = 1; i < turns.length; i += 1) {
    const x = turns[i - 1];
    const y = turns[i];
    if (x[2] !== y[2] && y[0] - x[1] <= SWITCH_GAP) {
      counts[bin(y[0])] += 1;
    }
  }
  return counts;
}

// space

/** session-constant helpers of a space part: tag lists and time lookups for the tracks. */
export function spaceModel(space) {
  const tags = (space.tags || []).map((t) => t.id);
  const rare = new Set((space.tags || []).filter((t) => t.rare).map((t) => t.id));
  const kept = sortTags(tags.filter((t) => !rare.has(t)));
  const tracks = space.tracks || {};
  const byTime = {};
  for (const id of Object.keys(tracks)) {
    const m = new Map();
    const tr = tracks[id];
    for (let i = 0; i < (tr.t || []).length; i += 1) m.set(Math.round(tr.t[i] * 100), i);
    byTime[id] = m;
  }
  const pairs = [];
  for (let i = 0; i < kept.length; i += 1) for (let j = i + 1; j < kept.length; j += 1) pairs.push([kept[i], kept[j]]);
  return { space, tags: sortTags(tags), rare, kept, tracks, byTime, pairs };
}

/** movement, proximity and occupancy over a range, from the 1 s floor tracks. */
export function spaceStats(model, range, heatTags) {
  const { space, tracks, byTime } = model;
  const inR = (t) => !range || (t >= range[0] && t <= range[1]);
  const tagStats = {};
  for (const id of model.tags) {
    const tr = tracks[id] || { t: [], u: [], v: [], h: [] };
    let n = 0;
    let path = 0;
    let active = 0;
    const us = [];
    const vs = [];
    let pi = -1;
    for (let i = 0; i < tr.t.length; i += 1) {
      if (!inR(tr.t[i])) {
        pi = -1;
        continue;
      }
      n += 1;
      us.push(tr.u[i]);
      vs.push(tr.v[i]);
      if (pi >= 0 && tr.t[i] - tr.t[pi] <= TRAIL_GAP) {
        const d = Math.hypot(tr.u[i] - tr.u[pi], tr.v[i] - tr.v[pi]);
        if (d >= 0.05 && d <= 1.0) path += d;
        if (d > 0.1 && d <= 1.0) active += tr.t[i] - tr.t[pi];
      }
      pi = i;
    }
    const meta = (space.tags || []).find((t) => t.id === id) || {};
    tagStats[id] = {
      id,
      seconds: n,
      path,
      mPerMin: n > 0 ? path / (n / 60) : null,
      active,
      u: median(us),
      v: median(vs),
      present: range ? null : meta.present ?? null,
      rare: !!meta.rare,
    };
  }
  const pairs = [];
  const pooled = [];
  for (const [a, b] of model.pairs) {
    const A = tracks[a];
    const Bm = byTime[b];
    const B = tracks[b];
    const d = [];
    if (A && B) {
      for (let i = 0; i < A.t.length; i += 1) {
        if (!inR(A.t[i])) continue;
        const j = Bm.get(Math.round(A.t[i] * 100));
        if (j == null) continue;
        d.push(Math.hypot(A.u[i] - B.u[j], A.v[i] - B.v[j], (A.h[i] || 0) - (B.h[j] || 0)));
      }
    }
    for (const x of d) pooled.push(x);
    const server = (space.pairs || []).find((p) => p.a === a && p.b === b);
    if (!range && server) {
      pairs.push({ a, b, median: server.median, lt05: server.lt05, lt1: server.lt1, copresent: server.copresent, samples: d });
    } else {
      pairs.push({
        a,
        b,
        median: median(d),
        lt05: d.length ? d.filter((x) => x < 0.5).length / d.length : null,
        lt1: d.length ? d.filter((x) => x < 1).length / d.length : null,
        copresent: d.length,
        samples: d,
      });
    }
  }
  // distance from each tag to any other kept tag, pooled
  const toOthers = {};
  for (const id of model.kept) {
    const vals = [];
    for (const p of pairs) if (p.a === id || p.b === id) for (const x of p.samples) vals.push(x);
    toOthers[id] = median(vals);
  }
  let heat = null;
  const occ = space.occupancy;
  if (occ && occ.nu && occ.nv) {
    const counts = new Array(occ.nu * occ.nv).fill(0);
    const use = (heatTags && heatTags.length ? heatTags : model.kept).filter((t) => !model.rare.has(t));
    for (const id of use) {
      const tr = tracks[id];
      if (!tr) continue;
      for (let i = 0; i < tr.t.length; i += 1) {
        if (!inR(tr.t[i])) continue;
        const iu = Math.floor((tr.u[i] - occ.u0) / occ.cell + 1e-9);
        const iv = Math.floor((tr.v[i] - occ.v0) / occ.cell + 1e-9);
        if (iu < 0 || iu >= occ.nu || iv < 0 || iv >= occ.nv) continue;
        counts[iv * occ.nu + iu] += 1;
      }
    }
    heat = { u0: occ.u0, v0: occ.v0, cell: occ.cell, nu: occ.nu, nv: occ.nv, counts };
  }
  return { tags: tagStats, pairs, pooledMedian: median(pooled), toOthers, heat };
}

// attention and timeline (10 s windows of the video job)

/** gaze allocation, looks and social gaze over a range, from the per-window frame counts. */
export function attentionStats(att, range) {
  const idx = windowIndices(att.t || [], att.step || 10, range, att.duration);
  const pupils = att.pupils || [];
  const allocation = {};
  const socialGaze = {};
  const frames = {};
  const allocFrames = {};
  for (const tag of pupils) {
    const aw = (att.allocation_windows || {})[tag];
    if (!aw) continue;
    let fr = 0;
    const cats = {};
    for (const c of CATEGORIES) cats[c.key] = 0;
    for (const i of idx) {
      fr += aw.frames[i] || 0;
      for (const c of CATEGORIES) cats[c.key] += (aw[c.key] && aw[c.key][i]) || 0;
    }
    frames[tag] = fr;
    allocFrames[tag] = cats;
    allocation[tag] = {};
    for (const c of CATEGORIES) allocation[tag][c.key] = fr > 0 ? cats[c.key] / fr : null;
    socialGaze[tag] = fr > 0 ? (cats.partner_face + cats.partner_hands) / fr : null;
  }
  const looks = att.looks || {};
  const targets = looks.targets || [];
  const share = (kind) => pupils.map((_, li) => {
    let seen = 0;
    const hits = targets.map(() => 0);
    for (const i of idx) {
      seen += ((looks.seen && looks.seen.windows[i]) || [])[li] || 0;
      const row = ((looks[kind] && looks[kind].windows[i]) || [])[li] || [];
      for (let j = 0; j < targets.length; j += 1) hits[j] += row[j] || 0;
    }
    return { seen, hits, share: hits.map((x) => (seen > 0 ? x / seen : null)) };
  });
  return { idx, allocation, allocFrames, socialGaze, frames, looks: { targets, face: share('face'), hands: share('hands') } };
}

/** interaction state, joint attention, hands and presence over a range, from the timeline windows. */
export function timelineStats(tl, att, range) {
  const idx = windowIndices(tl.t || [], tl.step || 10, range, tl.duration);
  let state = null;
  if (tl.state && tl.state.values) {
    const counts = { 0: 0, 1: 0, 2: 0, '-2': 0 };
    let n = 0;
    for (const i of idx) {
      const v = tl.state.values[i];
      if (v == null) continue;
      counts[String(v)] = (counts[String(v)] || 0) + 1;
      n += 1;
    }
    state = { windows: n, shares: {} };
    for (const k of Object.keys(counts)) state.shares[k] = n ? counts[k] / n : null;
    if (!range && tl.state.shares) state.shares = { ...state.shares, ...tl.state.shares };
  }
  const pairs = {};
  for (const key of Object.keys(tl.pairs || {})) {
    const p = tl.pairs[key];
    const [a, b] = key.split('|');
    const server = att && (att.joint_attention || []).find((j) => j.a === a && j.b === b);
    // windows weigh by the frames both pupils were seen in (the server weighs by the pair's frames)
    const fa = att && att.allocation_windows && att.allocation_windows[a] ? att.allocation_windows[a].frames : null;
    const fb = att && att.allocation_windows && att.allocation_windows[b] ? att.allocation_windows[b].frames : null;
    let ja = [];
    let ex = [];
    const wts = [];
    for (const i of idx) {
      if (finite(p.ja[i]) && finite(p.ja_excess[i])) {
        ja.push(p.ja[i]);
        ex.push(p.ja_excess[i]);
        wts.push(fa && fb ? Math.min(fa[i] || 0, fb[i] || 0) : 1);
      }
    }
    // hands close weighs by the pair's frames (as on the server); a part written before the timeline
    // carried them falls back to the plain window mean
    const closeVals = [];
    const closeWts = [];
    for (const i of idx) {
      const v = p.hands_close && p.hands_close[i];
      if (!finite(v)) continue;
      closeVals.push(v);
      closeWts.push(Array.isArray(p.frames) ? p.frames[i] || 0 : 1);
    }
    const close = closeVals.length ? weighted(closeVals, closeWts) : null;
    const mutual = sum(idx.map((i) => p.face_mutual && p.face_mutual[i]));
    const dist = idx.map((i) => p.dist && p.dist[i]).filter(finite);
    if (!range && server) {
      pairs[key] = {
        a, b, ratio: server.ratio, baseline: server.baseline, excess: server.excess, windows: server.windows ?? ja.length,
        frames: server.frames ?? null, close, mutual, dist: median(dist),
      };
    } else {
      const r = weighted(ja, wts);
      const e = weighted(ex, wts);
      pairs[key] = {
        a, b, ratio: r, baseline: r != null && e != null ? r - e : null, excess: e, windows: ja.length, frames: null,
        close, mutual, dist: median(dist),
      };
    }
    ja = null;
    ex = null;
  }
  const tags = {};
  for (const tag of Object.keys(tl.tags || {})) {
    const s = tl.tags[tag];
    let present = 0;
    let observed = 0;
    let hw = 0;
    let hs = 0;
    const aw = att && att.allocation_windows ? att.allocation_windows[tag] : null;
    for (const i of idx) {
      const p = s.present[i];
      if (p != null) {
        observed += 1;
        if (p) present += 1;
      }
      const ha = s.hands_active && s.hands_active[i];
      if (finite(ha)) {
        // the ratio is a share of hand steps, so the windows weigh by them (as on the server); a part
        // written before the timeline carried them falls back to the pupil's camera frames
        const w = Array.isArray(s.hand_steps) ? s.hand_steps[i] || 0 : aw ? aw.frames[i] || 0 : 1;
        hw += w;
        hs += ha * w;
      }
    }
    const server = !range && att && att.hands ? att.hands[tag] : null;
    tags[tag] = {
      present: observed ? present / observed : null,
      presentSeconds: present * (tl.step || 10),
      handsActive: server ? server.active : hw > 0 ? hs / hw : null,
    };
  }
  return { idx, state, pairs, tags };
}

function weighted(values, weights) {
  let ws = 0;
  let s = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (!finite(values[i])) continue;
    const w = finite(weights[i]) ? weights[i] : 0;
    ws += w;
    s += w * values[i];
  }
  if (ws > 0) return s / ws;
  return mean(values);
}



// shared page helpers

export const PART_JOB = { speech: 'light', space: 'light', attention: 'video', timeline: 'video' };
const PART_LIST = ['speech', 'space', 'attention', 'timeline'];

/** "Reading video features: 12 min of 61 min", "Counting looks: 1,200 of 3,651 frame sets" */
export function progressText(st) {
  if (!st || st.state === 'loading') return 'Loading';
  // a failed job keeps its last progress step; never show it as if the job still ran
  if (st.state === 'error') return 'Failed';
  const p = st.progress || {};
  const step = p.step || (st.state === 'queued' ? 'Waiting to start' : 'Computing');
  if (finite(p.done) && finite(p.total) && p.total > 1) {
    if (/^Counting looks/.test(step)) return `${step}: ${fmt.int(p.done)} of ${fmt.int(p.total)} frame sets`;
    if (/^Reading/.test(step)) return `${step}: ${fmt.duration(p.done)} of ${fmt.duration(p.total)}`;
    return `${step}: ${fmt.int(p.done)} of ${fmt.int(p.total)}`;
  }
  return step;
}

function busy(st) {
  return !!st && (st.state === 'queued' || st.state === 'running');
}

export function pairKey(a, b) {
  const [x, y] = sortTags([a, b]);
  return `${x}|${y}`;
}

function pairLabel(key) {
  const [a, b] = String(key).split('|');
  return `Tag ${a} and Tag ${b}`;
}

/** every pair key the parts mention, in tag order, with a stable colour per pair. */
export function pairList(ctx) {
  const keys = new Set();
  const { space, timeline: tl, attention: att } = ctx.parts;
  if (ctx.models.space) for (const [a, b] of ctx.models.space.pairs) keys.add(pairKey(a, b));
  if (tl && tl.pairs) for (const k of Object.keys(tl.pairs)) keys.add(pairKey(...k.split('|')));
  if (att) for (const j of att.joint_attention || []) keys.add(pairKey(j.a, j.b));
  if (space && !ctx.models.space) for (const p of space.pairs || []) keys.add(pairKey(p.a, p.b));
  const list = Array.from(keys).sort((x, y) => {
    const [xa, xb] = x.split('|');
    const [ya, yb] = y.split('|');
    const ox = sortTags([xa, ya]);
    if (xa !== ya) return ox[0] === xa ? -1 : 1;
    const oy = sortTags([xb, yb]);
    return xb === yb ? 0 : oy[0] === xb ? -1 : 1;
  });
  return list.map((key) => ({ key, label: pairLabel(key), color: PAIR_INK, a: key.split('|')[0], b: key.split('|')[1] }));
}

/** the derived numbers for the current range, cached until the parts or the range change. */
export function derive(ctx) {
  const key = `${ctx.version}|${ctx.range ? ctx.range.join(',') : ''}`;
  if (ctx.derived && ctx.derivedKey === key) return ctx.derived;
  const P = ctx.parts;
  const d = { pairs: pairList(ctx) };
  const safe = (fn) => {
    try {
      return fn();
    } catch (err) {
      return null;
    }
  };
  d.sm = ctx.models.speech || null;
  d.sp = d.sm ? safe(() => speechStats(d.sm, ctx.range)) : null;
  d.pm = ctx.models.space || null;
  d.spc = d.pm ? safe(() => spaceStats(d.pm, ctx.range, null)) : null;
  d.at = P.attention ? safe(() => attentionStats(P.attention, ctx.range)) : null;
  d.tl = P.timeline ? safe(() => timelineStats(P.timeline, P.attention, ctx.range)) : null;
  ctx.derived = d;
  ctx.derivedKey = key;
  return d;
}

/**
 * Decide a card's state from the parts it needs: empty when the session lacks the modality, error
 * when a job failed, loading with the job's progress while a part is computed. Returns true when
 * every part is there.
 */
function gate(c, ctx, parts, { modality, emptyText } = {}) {
  const mods = ctx.meta && ctx.meta.modalities;
  if (modality && mods && !mods[modality]) {
    c.setState('empty', emptyText || 'This session has no data for this panel.');
    return false;
  }
  for (const p of parts) {
    const st = ctx.status[p] || {};
    if (!ctx.parts[p]) {
      if (st.state === 'error') {
        c.setState('error', {
          title: 'This part of the analysis failed.',
          body: st.error || null,
          action: { label: 'Run again', onClick: () => ctx.runJob(PART_JOB[p]) },
        });
      } else c.setState('loading', progressText(st));
      return false;
    }
  }
  // show the body before drawing: a chart measures its container when it is created
  c.setState('ready');
  return true;
}

/** after a ready render: hold it at half opacity while a refresh recomputes its parts. */
function settle(c, ctx, parts, note) {
  const running = parts.map((p) => ctx.status[p]).find(busy);
  if (running) {
    const p = progressText(running);
    c.setState('loading', /^Recomputing/.test(p) ? p : `Recomputing. ${p}`);
  }
  else {
    const stale = parts.map((p) => ctx.status[p]).filter((s) => s && s.stale);
    const failed = stale.find((s) => s.jobError);
    const staleNote = failed ? `Newer data exists, but recomputing this panel failed (${failed.jobError}); it shows the earlier result.`
      : stale.length ? 'Newer data exists: this panel is being recomputed.' : null;
    c.setState('ready', [note, staleNote].filter(Boolean).join(' '));
  }
}

const RANGE_NOTE = 'Whole session: this panel has no per-window data, so it ignores the range.';

function swatchEl(color, kind) {
  if (color === 'hatch') return h('span', { class: 'swatch hatch' });
  return h('span', { class: ['swatch', kind === 'dot' ? 'dot' : null, kind === 'line' ? 'line' : null], style: { '--swatch': color } });
}

function tagCell(ctx, id) {
  const t = ctx.ident.tag(id);
  return { text: t.label, color: t.color };
}

function entityColor(ctx, key) {
  const sp = ctx.parts.speech;
  return ctx.ident.entity(key, sp).color;
}

function entityLabel(ctx, key) {
  const sp = ctx.parts.speech;
  if (key === 'other') return (sp && sp.other && sp.other.label) || 'Other voices';
  const e = sp && (sp.entities || []).find((x) => x.key === key);
  return e ? e.label : voiceLabel(key);
}

function pctValue(v, digits = 0) {
  if (!finite(v)) return null;
  const n = Math.round(v * 100 * 10 ** digits) / 10 ** digits;
  return n.toFixed(digits);
}

function ppValue(v) {
  if (!finite(v)) return null;
  const n = Math.round(v * 1000) / 10;
  return `${n > 0 ? '+' : ''}${n.toFixed(1)}`;
}

function minuteIndices(ctx) {
  const sp = ctx.parts.speech;
  const n = Math.max(1, Math.ceil((ctx.duration || 0) / 60));
  const t = sp && sp.per_minute ? sp.per_minute.t : Array.from({ length: n }, (_, i) => i * 60);
  const out = [];
  for (let i = 0; i < t.length; i += 1) {
    if (ctx.range && (t[i] + 60 <= ctx.range[0] || t[i] >= ctx.range[1])) continue;
    out.push(i);
  }
  return { t, idx: out };
}

function dl(rows) {
  const el = h('dl', { class: 'an-kv' });
  for (const [k, v, title] of rows) {
    if (v == null) continue;
    el.append(h('dt', { text: k, title: title || null }), h('dd', {}, v instanceof Node ? v : String(v)));
  }
  return el;
}

function simpleTable(columns, rows, caption) {
  const table = h('table', { class: 'data' });
  if (caption) table.appendChild(h('caption', { class: 'sr-only', text: caption }));
  table.appendChild(h('thead', {}, h('tr', {}, columns.map((c) => h('th', { attrs: { scope: 'col', 'data-align': c.align || null, title: c.title || null }, text: c.label })))));
  const tbody = h('tbody');
  for (const r of rows) {
    const tr = h('tr', { class: r._class || null });
    for (const c of columns) {
      const v = r[c.key];
      const td = h('td', { attrs: { 'data-align': c.align || null } });
      if (v instanceof Node) td.appendChild(v);
      else if (v && typeof v === 'object' && 'text' in v) {
        if (v.color) td.appendChild(swatchEl(v.color));
        td.appendChild(document.createTextNode(String(v.text)));
      } else td.textContent = v == null ? fmt.na : String(v);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  return h('div', { class: 'table-wrap' }, table);
}

function banner(text, kind = 'info') {
  return h('div', { class: 'an-banner', dataset: { kind } }, icon(kind === 'warning' ? 'alert' : 'info', 16), h('p', { text }));
}

function sectionShell(id, title, sub) {
  const head = h('div', { class: 'section-head' }, h('h2', { text: title, id: `${id}-title` }), sub ? h('p', { text: sub }) : null);
  const notes = h('div', { class: 'an-notes' });
  const grid = h('div', { class: 'grid' });
  const el = h('section', { class: 'section an-section', id, attrs: { 'aria-labelledby': `${id}-title` } }, head, notes, grid);
  return { el, head, notes, grid };
}

// overview

const KPI_DEFS = [
  { key: 'speech', label: 'Speech activity', part: 'speech', modality: 'asr' },
  { key: 'switches', label: 'Turn switches', part: 'speech', modality: 'asr' },
  { key: 'balance', label: 'Speaking balance', part: 'speech', modality: 'asr' },
  { key: 'distance', label: 'Median distance', part: 'space', modality: 'ips' },
  { key: 'ja', label: 'Joint attention above baseline', part: 'timeline', also: 'attention', modality: 'vfa' },
  { key: 'collab', label: 'Collaborative windows', part: 'timeline', modality: null },
];

function pendingWord(st) {
  if (!st || st.state === 'loading') return 'Loading';
  if (st.state === 'queued') return 'Queued';
  if (st.state === 'running') return 'Computing';
  if (st.state === 'error') return 'Failed';
  return 'Loading';
}

function kpiValues(ctx, d, key) {
  const P = ctx.parts;
  const { t: mt, idx: mi } = minuteIndices(ctx);
  if (key === 'speech') {
    const pm = P.speech.per_minute || {};
    return { value: pctValue(d.sp && d.sp.speechRatio), unit: '%', note: P.speech.mode === 'wearer+group' ? 'group mic, voiced share' : 'voiced share of 3 s buckets', spark: mi.map((i) => (pm.speech || [])[i]) };
  }
  if (key === 'switches') {
    const per = switchesPerMinute(d.sm);
    const v = d.sp ? d.sp.switchesPerMin : null;
    return {
      value: finite(v) ? fmt.num(v, 1) : null,
      unit: '/min',
      note: v == null && d.sm && !d.sm.named ? 'no linked voices' : d.sm && d.sm.named ? 'between wearers' : 'between voices',
      spark: mi.map((i) => per[i]),
    };
  }
  if (key === 'balance') {
    const pm = P.speech.per_minute || {};
    const keys = Object.keys(pm.by_entity || {});
    const spark = mi.map((i) => entropyBalance(keys.map((k) => pm.by_entity[k][i])));
    const v = d.sp ? d.sp.balance : null;
    return { value: finite(v) ? fmt.num(v, 2) : null, unit: '', note: v == null ? 'needs 2 speakers' : '1 = even shares', spark };
  }
  if (key === 'distance') {
    const pairs = (P.space.pairs || []).filter((p) => p.series);
    const bins = mt.map(() => []);
    for (const p of pairs) {
      for (let i = 0; i < p.series.t.length; i += 1) {
        const b = Math.floor(p.series.t[i] / 60);
        if (b >= 0 && b < bins.length && finite(p.series.d[i])) bins[b].push(p.series.d[i]);
      }
    }
    const v = d.spc ? d.spc.pooledMedian : null;
    const n = d.spc ? d.spc.pairs.filter((p) => p.samples.length).length : 0;
    return { value: finite(v) ? fmt.num(v, 2) : null, unit: 'm', note: n ? `over ${n} ${n === 1 ? 'pair' : 'pairs'}` : 'no pair seen together', spark: mi.map((i) => median(bins[i] || [])) };
  }
  if (key === 'ja') {
    const tl = P.timeline;
    const vals = d.tl ? Object.values(d.tl.pairs).filter((p) => finite(p.excess) && p.windows >= FEW_WINDOWS).map((p) => p.excess) : [];
    const few = d.tl ? Object.values(d.tl.pairs).filter((p) => finite(p.excess) && p.windows < FEW_WINDOWS).length : 0;
    const spark = mi.map(() => []);
    const keys = Object.keys(tl.pairs || {});
    for (let i = 0; i < (tl.t || []).length; i += 1) {
      const b = Math.floor(tl.t[i] / 60);
      const pos = mi.indexOf(b);
      if (pos < 0) continue;
      for (const k of keys) if (finite(tl.pairs[k].ja_excess[i])) spark[pos].push(tl.pairs[k].ja_excess[i]);
    }
    return {
      value: vals.length ? ppValue(mean(vals)) : null,
      unit: 'pp',
      note: vals.length ? `mean of ${vals.length} ${vals.length === 1 ? 'pair' : 'pairs'}` : few ? 'too few windows' : 'no pair data',
      spark: spark.map((b) => mean(b)),
    };
  }
  if (key === 'collab') {
    const tl = P.timeline;
    if (!tl.state || !d.tl || !d.tl.state) return { value: null, note: tl.state_error ? 'not estimated' : 'no state', spark: null };
    const spark = mi.map(() => [0, 0]);
    for (let i = 0; i < tl.t.length; i += 1) {
      const v = tl.state.values[i];
      if (v == null) continue;
      const pos = mi.indexOf(Math.floor(tl.t[i] / 60));
      if (pos < 0) continue;
      spark[pos][1] += 1;
      if (v === 2) spark[pos][0] += 1;
    }
    return { value: pctValue(d.tl.state.shares['2']), unit: '%', note: 'estimated, not validated', spark: spark.map(([a, n]) => (n ? a / n : null)) };
  }
  return { value: null };
}

function overviewSection(ctx) {
  const sec = sectionShell('overview', 'Overview');
  const kpiEl = h('div', { class: 'kpis an-kpis', style: { '--span': '12' } });
  const tiles = KPI_DEFS.map((def) => {
    const tile = statTile({ label: def.label, pending: 'Loading' });
    kpiEl.appendChild(tile);
    return { def, tile };
  });
  sec.grid.appendChild(kpiEl);

  const tlCard = card({
    title: 'Session timeline',
    subtitle: 'Drag across the lanes to select a range. Interaction state: estimated from fixed thresholds, not validated.',
    span: 12,
    id: 'session-timeline',
    table: () => timelineTable(ctx),
  });
  sec.grid.appendChild(tlCard.el);
  let tlChart = null;
  let tlVersion = -1;

  const partCard = card({ title: 'Participants', subtitle: 'One row per pupil badge. Hover a column title for what it measures.', span: 12 });
  sec.grid.appendChild(partCard.el);

  function renderKpis(d) {
    for (const { def, tile } of tiles) {
      const mods = ctx.meta && ctx.meta.modalities;
      if (def.modality && mods && !mods[def.modality]) {
        tile.update({ value: null, unit: '', note: `No ${def.modality.toUpperCase()} data`, pending: null, spark: null });
        continue;
      }
      if (def.key === 'collab' && mods && !mods.ips && !mods.vfa) {
        tile.update({ value: null, unit: '', note: 'No IPS or VFA data', pending: null, spark: null });
        continue;
      }
      const parts = [def.part, def.also].filter(Boolean);
      const missing = parts.find((p) => !ctx.parts[p]);
      if (missing) {
        tile.update({ pending: pendingWord(ctx.status[missing]), note: def.key === 'collab' || def.key === 'ja' ? 'video analysis' : null, spark: null });
        continue;
      }
      let v;
      try {
        v = kpiValues(ctx, d, def.key);
      } catch (err) {
        v = { value: null, note: 'could not compute' };
      }
      const spark = v.spark && v.spark.some(finite) ? { values: v.spark, color: 'var(--ink-2)' } : null;
      tile.update({ value: v.value, unit: v.unit || '', note: v.note || '', pending: null, spark });
    }
  }

  function renderTimeline(d) {
    const P = ctx.parts;
    if (!P.speech && !P.space && !P.timeline) {
      const anyErr = ['speech', 'space', 'timeline'].map((p) => ctx.status[p]).find((st) => st && st.state === 'error');
      if (anyErr) tlCard.setState('error', { title: 'The analysis failed.', body: anyErr.error || null, action: { label: 'Run again', onClick: () => ctx.runJob('all') } });
      else tlCard.setState('loading', progressText(ctx.status.speech));
      return;
    }
    if (!tlChart) tlCard.setState('ready');
    if (tlVersion !== ctx.version || !tlChart) {
      const lanes = timelineLanes(ctx, d);
      const gaps = P.speech && P.speech.coverage ? P.speech.coverage.gaps || [] : [];
      if (!tlChart) {
        tlChart = timeline(tlCard.body, {
          span: ctx.span, lanes, gaps, brush: true, labelWidth: 196, label: 'Session timeline',
          axisFormat: ctx.clock,
          onBrush: (r) => ctx.setRange(r, 'timeline'),
          onClick: (t) => ctx.setCursor(t, { from: 'timeline' }),
        });
      } else tlChart.update({ lanes, gaps, span: ctx.span });
      tlVersion = ctx.version;
      tlCard.setLegend(timelineLegend(ctx));
    }
    tlChart.setBrush(ctx.range);
    tlChart.setCursor(ctx.cursor);
    const pending = [];
    if (!P.timeline && ctx.meta && ctx.meta.modalities && (ctx.meta.modalities.vfa || ctx.meta.modalities.ips)) {
      const st = ctx.status.timeline;
      pending.push(st && st.state === 'error'
        ? 'The video analysis failed, so the interaction state, social gaze and joint attention lanes are missing.'
        : `Interaction state, social gaze and joint attention lanes appear when the video analysis finishes. ${progressText(st)}.`);
    }
    if (!P.speech && ctx.meta && ctx.meta.modalities && ctx.meta.modalities.asr) pending.push(`Speech lanes: ${progressText(ctx.status.speech)}.`);
    if (!P.space && ctx.meta && ctx.meta.modalities && ctx.meta.modalities.ips) pending.push(`Presence and distance: ${progressText(ctx.status.space)}.`);
    settle(tlCard, ctx, ['speech', 'space', 'timeline'].filter((p) => P[p]), pending.join(' '));
  }

  function renderParticipants(d) {
    const P = ctx.parts;
    if (!P.space && !P.timeline && !P.speech) {
      const own = ['space', 'timeline', 'speech'];
      const failed = own.filter((p) => (ctx.status[p] || {}).state === 'error');
      const waiting = own.map((p) => ctx.status[p]).find((st) => !st || st.state !== 'error');
      if (waiting) partCard.setState('loading', progressText(waiting));
      else {
        // every part this card reads failed: say so and offer to run their jobs again
        const jobs = Array.from(new Set(failed.map((p) => PART_JOB[p])));
        const errors = Array.from(new Set(failed.map((p) => ctx.status[p].error).filter(Boolean)));
        partCard.setState('error', {
          title: 'This part of the analysis failed.',
          body: errors.join(' ') || null,
          action: { label: 'Run again', onClick: () => ctx.runJob(jobs.length > 1 ? 'all' : jobs[0]) },
        });
      }
      return;
    }
    const tags = ctx.roster;
    clear(partCard.body);
    if (!tags.length) {
      partCard.setState('empty', 'No pupil badge was seen in this session.');
      return;
    }
    const worn = d.sm && d.sm.named && d.sm.mode !== 'individual';
    const cols = [
      { key: 'tag', label: 'Tag' },
      { key: 'present', label: 'Present', align: 'right', title: P.timeline ? 'Share of the 10 s windows with badge or video data in which the pupil was seen' : 'Share of the badge position windows in which the badge was seen (video presence joins when the video analysis is ready)' },
      { key: 'moved', label: 'Moved', align: 'right', title: 'Floor distance walked per minute present, steps of 5 cm to 1 m (badge positions)' },
      { key: 'dist', label: 'Median distance to others', align: 'right', title: 'Median 3D distance to the other badges, over the seconds they were seen together' },
      { key: 'gaze', label: 'Social gaze', align: 'right', title: 'Share of frames with the gaze on a partner\'s face or hands (video)' },
      { key: 'hands', label: 'Hands active', align: 'right', title: 'Share of hand steps with the hands moving (video)' },
      { key: 'speak', label: 'Speaking', align: 'right', title: worn ? 'Share of the wearers\' speaking time and the words their own microphone gave them' : 'A group microphone records everyone: voices are anonymous and not linked to badges' },
    ];
    const rows = tags.map((id) => {
      const s = d.spc && d.spc.tags[id];
      let present = null;
      if (d.tl && d.tl.tags[id]) present = d.tl.tags[id].present;
      else if (s) present = ctx.range ? s.seconds / Math.max(1, ctx.range[1] - ctx.range[0]) : s.present;
      let speak;
      if (worn) {
        const e = d.sp && d.sp.entities.find((x) => x.key === `tag:${id}`);
        speak = e ? `${fmt.pct(e.share)}, ${fmt.int(e.words)} words` : fmt.na;
      } else speak = h('span', { class: 'muted', text: P.speech ? 'Group mic' : fmt.na });
      return {
        tag: tagCell(ctx, id),
        present: present == null ? fmt.na : fmt.pct(present),
        moved: s && finite(s.mPerMin) ? `${fmt.num(s.mPerMin, 2)}${NBSP}m/min` : fmt.na,
        dist: d.spc && finite(d.spc.toOthers[id]) ? fmt.metres(d.spc.toOthers[id]) : fmt.na,
        gaze: d.at && finite(d.at.socialGaze[id]) ? fmt.pct(d.at.socialGaze[id]) : fmt.na,
        hands: d.tl && d.tl.tags[id] && finite(d.tl.tags[id].handsActive) ? fmt.pct(d.tl.tags[id].handsActive) : fmt.na,
        speak,
      };
    });
    partCard.body.appendChild(simpleTable(cols, rows, 'Participants'));
    const waiting = [];
    const videoFailed = ['timeline', 'attention'].some((p) => !P[p] && ctx.status[p] && ctx.status[p].state === 'error');
    const mods = (ctx.meta && ctx.meta.modalities) || {};
    if (videoFailed) waiting.push('The video analysis failed, so social gaze and hands are missing.');
    else if ((!P.timeline || !P.attention) && mods.vfa) waiting.push('Social gaze and hands fill in when the video analysis finishes.');
    settle(partCard, ctx, ['space', 'timeline', 'attention', 'speech'].filter((p) => P[p]), waiting.join(' '));
  }

  return {
    ...sec,
    render() {
      const d = derive(ctx);
      renderKpis(d);
      renderTimeline(d);
      renderParticipants(d);
    },
    setCursor(t) {
      if (tlChart) tlChart.setCursor(t);
    },
    timelineCard: tlCard,
  };
}

function timelineLegend(ctx) {
  const P = ctx.parts;
  const wrap = h('div', { class: 'an-legend-row' });
  const items = [];
  if (P.timeline && P.timeline.state) for (const st of STATES) items.push({ label: st.label, color: st.fill, kind: st.fill === 'hatch' ? 'hatch' : 'rect' });
  if (P.speech) items.push({ label: 'Speech activity', color: 'var(--ink-2)', kind: 'line' });
  if (P.speech && P.speech.coverage && (P.speech.coverage.gaps || []).length) items.push({ label: 'No speech data', color: 'hatch', kind: 'hatch' });
  wrap.appendChild(legend(items));
  const pairs = pairList(ctx);
  if (P.timeline) wrap.appendChild(scaleLegend({ min: 0, max: socialGazeMax(P.timeline), format: (v) => fmt.pct(v), label: 'Social gaze' }));
  return wrap;
}

function quantile(values, q) {
  const v = values.filter(finite).sort((a, b) => a - b);
  if (!v.length) return null;
  return v[Math.min(v.length - 1, Math.max(0, Math.round(q * (v.length - 1))))];
}

/** a y domain that keeps a few outliers from flattening the lines: 0 to the 98th percentile. */
function distanceDomain(series) {
  const all = [];
  for (const se of series) for (const v of se.v || []) if (finite(v)) all.push(v);
  const hi = quantile(all, 0.98);
  return [0, Math.max(1.2, Math.ceil((hi == null ? 1 : hi) * 2) / 2)];
}

function excessDomain(series) {
  const all = [];
  for (const se of series) for (const v of se.v || []) if (finite(v)) all.push(v);
  const lo = quantile(all, 0.02);
  const hi = quantile(all, 0.98);
  return [Math.min(-0.1, Math.floor((lo == null ? 0 : lo) * 10) / 10), Math.max(0.1, Math.ceil((hi == null ? 0 : hi) * 10) / 10)];
}

function socialGazeMax(tl) {
  let max = 0;
  for (const tag of Object.keys(tl.tags || {})) for (const v of tl.tags[tag].social_gaze || []) if (finite(v) && v > max) max = v;
  return Math.max(0.1, max);
}

function timelineLanes(ctx, d) {
  const P = ctx.parts;
  const lanes = [];
  const tl = P.timeline;
  if (tl && tl.state && tl.state.values) {
    lanes.push({
      kind: 'cells', key: 'state', label: 'Interaction state', height: 14, t: tl.t, v: tl.state.values, step: tl.step,
      // under 1 px per window (phones) each bin shows its most frequent state
      binning: 'majority', binMinPx: 1,
      fillOf: (v) => (v == null ? null : STATE_FILL[String(v)] || null),
      format: (v) => (v == null ? 'not estimated' : STATE_LABEL[String(v)] || String(v)),
    });
  }
  if (P.speech && P.speech.activity) {
    lanes.push({
      kind: 'area', key: 'activity', label: 'Speech activity', t: P.speech.activity.t, v: P.speech.activity.v,
      // 3 s buckets narrower than 2 px are averaged into wider bins, so the lane reads as a profile
      step: P.speech.activity.step, max: 1, color: 'var(--ink-2)', format: (v) => fmt.pct(v), binning: 'mean',
    });
  }
  if (d.sm && d.sm.keys.length) {
    lanes.push({ kind: 'header', label: d.sm.named && d.sm.mode !== 'individual' ? 'Speakers (worn mics)' : 'Speakers' });
    const segs = new Map(d.sm.keys.map((k) => [k, []]));
    for (const tr of d.sm.turns) {
      const f = d.sm.fold(tr[2]);
      if (segs.has(f)) segs.get(f).push(tr);
    }
    for (const k of d.sm.keys) {
      const color = entityColor(ctx, k);
      lanes.push({
        kind: 'segments', key: `spk-${k}`, label: entityLabel(ctx, k), swatch: color, color, segs: segs.get(k),
        format: (seg) => (seg ? (k === 'other' ? voiceLabel(seg[2]) : 'speaking') : 'no'),
      });
    }
  }
  const tags = ctx.roster;
  if (tags.length && (tl || P.space)) {
    lanes.push({ kind: 'header', label: 'Presence' });
    for (const id of tags) {
      const t = ctx.ident.tag(id);
      if (tl && tl.tags && tl.tags[id]) {
        lanes.push({
          kind: 'cells', key: `pres-${id}`, label: t.label, swatch: t.color, t: tl.t, v: tl.tags[id].present, step: tl.step,
          binning: 'majority', binMinPx: 1,
          fillOf: (v) => (v === 1 ? t.color : v === 0 ? null : 'hatch'),
          format: (v) => (v === 1 ? 'present' : v === 0 ? 'absent' : 'not observed'),
        });
      } else if (P.space && P.space.presence) {
        lanes.push({
          kind: 'segments', key: `pres-${id}`, label: t.label, swatch: t.color, color: t.color,
          segs: (P.space.presence[id] || []).map(([a, b]) => [a, b, id]),
          format: (seg) => (seg ? 'badge seen' : 'not seen'),
        });
      }
    }
  }
  if (tl && tl.tags && tags.some((id) => tl.tags[id])) {
    const max = socialGazeMax(tl);
    lanes.push({ kind: 'header', label: 'Social gaze' });
    for (const id of tags) {
      if (!tl.tags[id]) continue;
      const t = ctx.ident.tag(id);
      lanes.push({
        kind: 'cells', key: `gaze-${id}`, label: t.label, swatch: t.color, t: tl.t, v: tl.tags[id].social_gaze, step: tl.step,
        binning: 'mean', binMinPx: 1,
        fillOf: (v) => (v == null ? null : seqColor(v, max)),
        format: (v) => (v == null ? fmt.na : fmt.pct(v)),
      });
    }
  }
  const pairs = d.pairs;
  if (P.space && (P.space.pairs || []).length) {
    const series = [];
    for (const p of pairs) {
      const sp = (P.space.pairs || []).find((q) => pairKey(q.a, q.b) === p.key);
      if (sp && sp.series && sp.series.d.some(finite)) series.push({ key: p.key, label: p.label, color: p.color, t: sp.series.t, v: sp.series.d });
    }
    // one lane per pair (small multiples) on a shared scale: pairs carry no colour of their own
    if (series.length) {
      const domain = distanceDomain(series);
      lanes.push({ kind: 'header', label: 'Distance' });
      for (const se of series) {
        lanes.push({
          kind: 'lines', key: `dist-${se.key}`, label: se.label, height: 34, domain,
          // both reference lines, one label: in a short lane the two labels would collide
          series: [{ ...se, label: 'distance' }], refs: [{ v: 0.5, label: '' }, { v: 1, label: '1 m' }], format: (v) => fmt.metres(v),
        });
      }
    }
  }
  if (tl && tl.pairs && Object.keys(tl.pairs).length) {
    const series = [];
    for (const p of pairs) {
      const tp = tl.pairs[p.key] || tl.pairs[`${p.b}|${p.a}`];
      if (tp && (tp.ja_excess || []).some(finite)) series.push({ key: p.key, label: p.label, color: p.color, t: tl.t, v: tp.ja_excess });
    }
    if (series.length) {
      const domain = excessDomain(series);
      lanes.push({ kind: 'header', label: 'Joint attention above baseline' });
      for (const se of series) {
        lanes.push({
          kind: 'lines', key: `ja-${se.key}`, label: se.label, height: 34, domain,
          series: [{ ...se, label: 'above baseline' }], refs: [{ v: 0, label: '0' }], format: (v) => fmt.pp(v),
        });
      }
    }
  }
  if (tl && tl.coverage) {
    lanes.push({ kind: 'header', label: 'Data' });
    for (const [k, label] of [['asr', 'ASR'], ['ips', 'IPS'], ['vfa', 'VFA']]) {
      if (!tl.coverage[k]) continue;
      lanes.push({
        kind: 'cells', key: `cov-${k}`, label, t: tl.t, v: tl.coverage[k], step: tl.step, binning: 'majority', binMinPx: 1,
        fillOf: (v) => (v ? 'var(--ord-1)' : 'hatch'), format: (v) => (v ? 'recorded' : 'missing'),
      });
    }
  } else if (P.speech && P.speech.activity) {
    lanes.push({ kind: 'header', label: 'Data' });
    lanes.push({
      kind: 'cells', key: 'cov-asr', label: 'ASR', t: P.speech.activity.t, v: P.speech.activity.v.map((v) => (v == null ? 0 : 1)),
      step: P.speech.activity.step, binning: 'majority', binMinPx: 1,
      fillOf: (v) => (v ? 'var(--ord-1)' : 'hatch'), format: (v) => (v ? 'recorded' : 'unreadable'),
    });
  }
  return lanes;
}

function timelineTable(ctx) {
  const P = ctx.parts;
  const { t, idx } = minuteIndices(ctx);
  const pm = P.speech ? P.speech.per_minute : null;
  const tl = P.timeline;
  const rows = idx.map((i) => {
    const m = t[i];
    const row = { minute: ctx.clock(m) };
    row.speech = pm ? fmt.pct(pm.speech[i]) : fmt.na;
    row.words = pm ? fmt.int(pm.words[i]) : fmt.na;
    if (tl && tl.state) {
      let n = 0;
      let c = 0;
      const ja = [];
      for (let w = 0; w < tl.t.length; w += 1) {
        if (Math.floor(tl.t[w] / 60) !== i) continue;
        const v = tl.state.values[w];
        if (v != null) {
          n += 1;
          if (v === 2) c += 1;
        }
        for (const k of Object.keys(tl.pairs || {})) if (finite(tl.pairs[k].ja_excess[w])) ja.push(tl.pairs[k].ja_excess[w]);
      }
      row.collab = n ? fmt.pct(c / n) : fmt.na;
      row.ja = ja.length ? fmt.pp(mean(ja)) : fmt.na;
    }
    if (P.space) {
      const vals = [];
      for (const p of P.space.pairs || []) {
        if (!p.series) continue;
        for (let j = 0; j < p.series.t.length; j += 1) if (Math.floor(p.series.t[j] / 60) === i && finite(p.series.d[j])) vals.push(p.series.d[j]);
      }
      row.dist = vals.length ? fmt.metres(median(vals)) : fmt.na;
    }
    return row;
  });
  const columns = [{ key: 'minute', label: 'Minute from' }, { key: 'speech', label: 'Speech activity', align: 'right' }, { key: 'words', label: 'Words', align: 'right' }];
  if (P.space) columns.push({ key: 'dist', label: 'Median pair distance', align: 'right' });
  if (tl && tl.state) columns.push({ key: 'collab', label: 'Collaborative windows', align: 'right' }, { key: 'ja', label: 'Joint attention above baseline', align: 'right' });
  return { columns, rows };
}

// speech

function speechSection(ctx) {
  const sec = sectionShell('speech', 'Speech');
  const timeCard = card({ title: 'Speaking time', subtitle: 'Seconds of speech per speaker, share of all speaking time', span: 5, table: () => speakingTable(ctx) });
  const transCard = card({ title: 'Who speaks after whom', subtitle: 'Turn switches, row speaks first', span: 4, table: () => transitionsTable(ctx) });
  const histCard = card({ title: 'Turns and pauses', subtitle: 'Counts of turn and pause lengths', span: 3, table: () => histTable(ctx) });
  const overCard = card({ title: 'Speech over time', subtitle: 'Speaking time per minute by speaker, and words per minute', span: 12, table: () => perMinuteTable(ctx) });
  const txCard = transcriptCard(ctx);
  sec.grid.append(timeCard.el, transCard.el, histCard.el, overCard.el, txCard.card.el);
  // inner layouts join their card on the first ready render, so a loading card shows a shimmer
  const histBody = { turns: h('div'), pauses: h('div') };
  histBody.el = h('div', {}, h('h3', { class: 'an-sub', text: 'Turn length' }), histBody.turns, h('h3', { class: 'an-sub', text: 'Pause length' }), histBody.pauses);
  const overBody = { cols: h('div'), words: h('div') };
  overBody.el = h('div', {}, overBody.cols, h('h3', { class: 'an-sub', text: 'Words per minute' }), overBody.words);
  const charts = {};
  const opts = { modality: 'asr', emptyText: 'No ASR data in this session.' };
  let notesVersion = -1;

  function renderNotes() {
    if (notesVersion === ctx.version) return;
    notesVersion = ctx.version;
    clear(sec.notes);
    const sp = ctx.parts.speech;
    if (!sp) return;
    if (sp.mode === 'group') sec.notes.appendChild(banner('One group microphone: speakers are anonymous voices from diarization and are not linked to badges.'));
    else if (sp.mode === 'wearer+group') sec.notes.appendChild(banner('Worn microphones and a group microphone: speaking time and turns come from the worn microphones (an energy vote between them), speech activity and words from the group microphone.'));
    else if (sp.mode === 'wearer') sec.notes.appendChild(banner('Worn microphones only: speaking time and turns come from an energy vote between the worn microphones. Each microphone also hears the teacher and neighbours, so there is no session speech rate.'));
    else if (sp.mode === 'individual') sec.notes.appendChild(banner('Named speakers: the transcriber matched each chunk to an enrolled speaker.'));
    for (const n of sp.notes || []) sec.notes.appendChild(banner(n, 'warning'));
  }

  function renderTime(d) {
    if (!gate(timeCard, ctx, ['speech'], opts)) return;
    const ents = d.sp ? d.sp.entities : [];
    if (!ents.length) {
      timeCard.setState('empty', 'No speaker was told apart in this session.');
      return;
    }
    const items = ents.map((e) => ({ key: e.key, label: e.label, value: e.seconds, color: entityColor(ctx, e.key), note: fmt.pct(e.share) }));
    const o = { items, format: (v) => fmt.duration(v), valueLabel: 'speaking time', noteLabel: 'share', label: 'Speaking time per speaker' };
    if (charts.time) charts.time.update(o);
    else charts.time = barList(timeCard.body, o);
    const note = d.sm.named && d.sm.mode !== 'individual' ? 'A wearer counts in the 3 s buckets their microphone won the energy vote; several can win the same bucket.' : null;
    settle(timeCard, ctx, ['speech'], note);
  }

  function renderTransitions(d) {
    if (!gate(transCard, ctx, ['speech'], opts)) return;
    const tr = d.sp ? d.sp.transitions : null;
    if (!tr || !tr.keys.length) {
      transCard.setState('empty', d.sm && !d.sm.named ? 'No turn is linked to a voice, so turns cannot be followed across chunks.' : 'No turn switches.');
      return;
    }
    const rows = tr.keys.map((k) => ({ key: k, label: entityLabel(ctx, k), swatch: entityColor(ctx, k) }));
    const cols = rows.map((r) => (r.key === 'other' ? { ...r, label: 'Others' } : r));
    const o = {
      // a voice never follows itself, so the diagonal stays blank; switches between two of the folded
      // Other voices are still listed in the table
      rows, cols, values: tr.counts, format: (v) => fmt.int(v), diagonal: 'blank', valueLabel: 'switches',
      title: (r, c) => `${r.label} then ${c.label}`, scaleLabel: 'Switches', label: 'Turn switches from row to column',
    };
    if (charts.trans) charts.trans.update(o);
    else charts.trans = heatMatrix(transCard.body, o);
    settle(transCard, ctx, ['speech'], d.sm.named ? 'A wearer who starts speaking follows the wearers of the last named bucket, across up to 9 s of silence.' : 'Next turn of another voice within 2 s.');
  }

  function renderHist(d) {
    if (!gate(histCard, ctx, ['speech'], opts)) return;
    if (!histBody.el.isConnected) histCard.body.appendChild(histBody.el);
    const sp = ctx.parts.speech;
    const t = { edges: sp.turn_lengths.edges, counts: d.sp.turnLengths, format: (v) => `${v}`, xLabel: 'seconds', countLabel: 'turns', color: 'var(--seq-4)', height: 120, label: 'Turn lengths' };
    const p = { edges: sp.pauses.edges, counts: d.sp.pauses, format: (v) => `${v}`, xLabel: 'seconds', countLabel: 'pauses', color: 'var(--seq-6)', height: 120, label: 'Pause lengths' };
    if (charts.turns) charts.turns.update(t);
    else charts.turns = histogram(histBody.turns, t);
    if (charts.pauses) charts.pauses.update(p);
    else charts.pauses = histogram(histBody.pauses, p);
    settle(histCard, ctx, ['speech'], `Mean turn ${fmt.secs(d.sp.meanTurn)}, mean pause ${fmt.secs(d.sp.meanPause)}.`);
  }

  function renderOverTime(d) {
    if (!gate(overCard, ctx, ['speech'], opts)) return;
    if (!overBody.el.isConnected) overCard.body.appendChild(overBody.el);
    const sp = ctx.parts.speech;
    const pm = sp.per_minute;
    const { idx } = minuteIndices(ctx);
    const keys = Object.keys(pm.by_entity || {});
    const cats = keys.map((k) => ({ key: k, label: entityLabel(ctx, k), fill: entityColor(ctx, k) }));
    const rows = idx.map((i) => ({ key: String(i), label: ctx.clock(pm.t[i]), values: Object.fromEntries(keys.map((k) => [k, pm.by_entity[k][i]])) }));
    const colOpts = { rows, categories: cats, orient: 'vertical', normalize: false, height: 180, marginLeft: 48, format: (v) => fmt.secs(v, 0), label: 'Speaking time per minute' };
    if (charts.cols) charts.cols.update(colOpts);
    else charts.cols = stackedBars(overBody.cols, colOpts);
    const t0 = idx.length ? pm.t[idx[0]] : 0;
    const t1 = idx.length ? pm.t[idx[idx.length - 1]] + 60 : 60;
    const wOpts = {
      span: [t0, t1], height: 120, marginLeft: 48, format: (v) => fmt.int(v), xFormat: ctx.clock, label: 'Words per minute',
      series: [{ key: 'words', label: 'Words', color: 'var(--ink-2)', t: idx.map((i) => pm.t[i] + 30), v: idx.map((i) => pm.words[i]) }],
    };
    if (charts.words) charts.words.update(wOpts);
    else charts.words = lineChart(overBody.words, wOpts);
    overCard.setLegend(cats.map((c) => ({ label: c.label, color: c.fill })));
    settle(overCard, ctx, ['speech'], ctx.range ? 'Whole minutes that the range touches.' : null);
  }

  return {
    ...sec,
    render() {
      const d = derive(ctx);
      renderNotes();
      renderTime(d);
      renderTransitions(d);
      renderHist(d);
      renderOverTime(d);
      txCard.render(d);
    },
    setCursor(t) {
      txCard.setCursor(t);
    },
  };
}

function speakingTable(ctx) {
  const d = derive(ctx);
  if (!d.sp) return null;
  const voices = d.sm && !d.sm.named;
  const columns = [
    { key: 'who', label: 'Speaker' }, { key: 'time', label: 'Speaking time', align: 'right' }, { key: 'share', label: 'Share', align: 'right' },
    { key: 'turns', label: 'Turns', align: 'right' }, { key: 'mean', label: 'Mean turn', align: 'right' }, { key: 'words', label: 'Words', align: 'right' },
  ];
  if (voices || (d.sm && d.sm.mode === 'individual')) columns.push({ key: 'wpm', label: 'Words per minute', align: 'right' });
  const rows = d.sp.entities.map((e) => ({
    who: { text: e.label, color: entityColor(ctx, e.key) }, time: fmt.duration(e.seconds), share: fmt.pct(e.share, 1), turns: fmt.int(e.turns),
    mean: fmt.secs(e.meanTurn), words: fmt.int(e.words), wpm: fmt.num(e.wpm, 0),
  }));
  return { columns, rows };
}

function transitionsTable(ctx) {
  const d = derive(ctx);
  if (!d.sp || !d.sp.transitions.keys.length) return null;
  const tr = d.sp.transitions;
  const rows = [];
  tr.keys.forEach((a, i) => tr.keys.forEach((b, j) => {
    if (tr.counts[i][j] > 0) rows.push({ from: { text: entityLabel(ctx, a), color: entityColor(ctx, a) }, to: { text: entityLabel(ctx, b), color: entityColor(ctx, b) }, n: tr.counts[i][j] });
  }));
  rows.sort((x, y) => y.n - x.n);
  return { columns: [{ key: 'from', label: 'From' }, { key: 'to', label: 'To' }, { key: 'n', label: 'Switches', align: 'right' }], rows };
}

function histTable(ctx) {
  const d = derive(ctx);
  const sp = ctx.parts.speech;
  if (!d.sp || !sp) return null;
  const binLabels = (edges, n) => Array.from({ length: n }, (_, i) => (i === 0 ? `under ${edges[0]} s` : i === edges.length ? `${edges[i - 1]} s or more` : `${edges[i - 1]} to ${edges[i]} s`));
  const rows = [];
  binLabels(sp.turn_lengths.edges, d.sp.turnLengths.length).forEach((l, i) => rows.push({ what: 'Turn', bin: l, n: d.sp.turnLengths[i] }));
  binLabels(sp.pauses.edges, d.sp.pauses.length).forEach((l, i) => rows.push({ what: 'Pause', bin: l, n: d.sp.pauses[i] }));
  return { columns: [{ key: 'what', label: 'Kind' }, { key: 'bin', label: 'Length' }, { key: 'n', label: 'Count', align: 'right' }], rows };
}

function perMinuteTable(ctx) {
  const sp = ctx.parts.speech;
  if (!sp) return null;
  const pm = sp.per_minute;
  const { idx } = minuteIndices(ctx);
  const keys = Object.keys(pm.by_entity || {});
  const columns = [{ key: 'm', label: 'Minute from' }].concat(keys.map((k) => ({ key: k, label: entityLabel(ctx, k), align: 'right' })), [{ key: 'words', label: 'Words', align: 'right' }]);
  const rows = idx.map((i) => {
    const r = { m: ctx.clock(pm.t[i]), words: fmt.int(pm.words[i]) };
    for (const k of keys) r[k] = fmt.secs(pm.by_entity[k][i], 1);
    return r;
  });
  return { columns, rows };
}

// transcript

const PAGE = 200;
const PAGINATE_OVER = 400;

function transcriptCard(ctx) {
  const c = card({ title: 'Transcript', subtitle: 'Chunks in the range. Select a time to show it on the session timeline.', span: 12, id: 'transcript' });
  const inputId = `tx-search-${Math.random().toString(36).slice(2, 7)}`;
  const search = h('input', { class: 'input', id: inputId, attrs: { type: 'search', autocomplete: 'off', spellcheck: 'false' } });
  const field = h('div', { class: 'field tx-search' }, h('label', { attrs: { for: inputId }, text: 'Search' }), search);
  const chips = h('div', { class: 'tx-chips', attrs: { role: 'group', 'aria-label': 'Speakers shown' } });
  const chipField = h('div', { class: 'field' }, h('span', { class: 'field-label', text: 'Speakers' }), chips);
  const count = h('p', { class: 'tx-count', attrs: { role: 'status', 'aria-live': 'polite' } });
  const list = h('ol', { class: 'tx-list', attrs: { 'aria-label': 'Transcript chunks' } });
  const more = h('button', { class: 'btn sm', attrs: { type: 'button' }, text: 'Show more', hidden: true });
  const legendEl = h('div', { class: 'tx-legend' });
  const inner = h('div', {}, h('div', { class: 'toolbar tx-toolbar' }, field, chipField), legendEl, count, list, h('div', { class: 'tx-more' }, more));
  const st = { query: '', off: new Set(), limit: PAGE, version: -1, rangeKey: '', filterKey: '', cursor: null, options: [] };

  const rerender = () => {
    st.filterKey = '';
    render(derive(ctx));
  };
  search.addEventListener('input', debounceInput(() => {
    st.query = search.value.trim().toLowerCase();
    st.limit = PAGE;
    rerender();
  }));
  more.addEventListener('click', () => {
    st.limit += PAGE;
    rerender();
  });

  function filterOptions() {
    const sp = ctx.parts.speech;
    const m = ctx.models.speech;
    if (!sp || !m) return [];
    if (m.mode === 'group') return m.keys.map((k) => ({ key: k, label: entityLabel(ctx, k), color: entityColor(ctx, k) }));
    if (m.mode === 'individual') return m.keys.map((k) => ({ key: k, label: entityLabel(ctx, k), color: entityColor(ctx, k) }));
    return (sp.lanes || []).map((ln) => ({ key: ln.key, label: ln.label, color: ln.key === 'group' ? 'var(--ink-2)' : ctx.ident.tag(ln.key.slice(4)).color }));
  }

  function chunkKeys(ch, m) {
    if (m.mode === 'group') {
      const ks = new Set();
      for (const w of ch.words || []) if (w[3] != null) ks.add(m.fold(w[3]));
      return ks;
    }
    if (m.mode === 'individual') return new Set([m.fold(`spk:${ch.speaker}`)]);
    return new Set([ch.lane]);
  }

  function dominant(ch, m) {
    if (ch.lane !== 'group') {
      const t = ctx.ident.tag(ch.lane.slice(4));
      return { label: ch.label || `${t.label} (worn mic)`, color: t.color };
    }
    if (m.mode === 'individual') {
      const k = m.fold(`spk:${ch.speaker}`);
      return { label: entityLabel(ctx, k), color: entityColor(ctx, k) };
    }
    const n = new Map();
    for (const w of ch.words || []) if (w[3] != null) n.set(w[3], (n.get(w[3]) || 0) + 1);
    if (!n.size) return { label: 'Group mic', color: 'var(--ink-2)' };
    const top = Array.from(n.entries()).sort((a, b) => b[1] - a[1])[0][0];
    const f = m.fold(top);
    return { label: f === 'other' ? voiceLabel(top) : entityLabel(ctx, f), color: entityColor(ctx, f), extra: n.size - 1 };
  }

  function wordsOf(ch) {
    if ((ch.words || []).length) return ch.words.map((w) => ({ text: String(w[0]), voice: w[3], cls: w[4], t: w[1] }));
    const text = String(ch.text || '').trim();
    return text ? text.split(/\s+/).map((x) => ({ text: x, voice: null, cls: null, t: null })) : [];
  }

  function chunkRow(ch, m, words) {
    const li = h('li', { class: 'tx-row', dataset: { t0: ch.t0 } });
    const time = h('button', {
      class: 'tx-time', attrs: { type: 'button', 'aria-label': `${ctx.clock(ch.t0)}, show on the session timeline` },
      text: ctx.clock(ch.t0), on: { click: () => ctx.setCursor(ch.t0, { from: 'transcript', reveal: true }) },
    });
    const dom = dominant(ch, m);
    const chip = h('span', { class: 'chip tx-chip' }, swatchEl(dom.color, 'dot'), h('span', { text: dom.label }), dom.extra ? h('span', { class: 'muted', text: `+${dom.extra}`, title: `${dom.extra} more ${dom.extra === 1 ? 'voice' : 'voices'} in this chunk` }) : null);
    const text = h('p', { class: 'tx-text' });
    const worn = ch.lane !== 'group';
    // match ranges over the words joined by single spaces
    let marks = null;
    if (st.query) {
      const joined = words.map((w) => w.text).join(' ').toLowerCase();
      marks = new Set();
      let from = 0;
      const starts = [];
      let pos = 0;
      for (const w of words) {
        starts.push(pos);
        pos += w.text.length + 1;
      }
      for (;;) {
        const at = joined.indexOf(st.query, from);
        if (at < 0) break;
        const end = at + st.query.length;
        for (let i = 0; i < words.length; i += 1) if (starts[i] < end && starts[i] + words[i].text.length > at) marks.add(i);
        from = at + 1;
      }
    }
    words.forEach((w, i) => {
      const cls = ['tx-w'];
      let style = null;
      if (!worn && w.voice != null) {
        cls.push('is-voiced');
        style = { '--u': entityColor(ctx, m.fold(w.voice)) };
      }
      if (worn && (w.cls === 'crosstalk' || w.cls === 'other')) cls.push('is-dim');
      const span = h('span', { class: cls, style, title: !worn && w.voice != null ? voiceLabel(w.voice) : worn && w.cls ? wordClassLabel(w.cls) : null });
      if (marks && marks.has(i)) span.appendChild(h('mark', { text: w.text }));
      else span.textContent = w.text;
      text.appendChild(span);
      text.appendChild(document.createTextNode(' '));
    });
    if (!words.length) text.appendChild(h('span', { class: 'muted', text: 'No text' }));
    li.append(time, chip, text);
    return li;
  }

  function render(d) {
    if (!gate(c, ctx, ['speech'], { modality: 'asr', emptyText: 'No ASR data in this session.' })) return;
    if (!inner.isConnected) c.body.appendChild(inner);
    const sp = ctx.parts.speech;
    const m = ctx.models.speech;
    const rangeKey = ctx.range ? ctx.range.join(',') : '';
    if (st.version !== ctx.version) {
      st.options = filterOptions();
      clear(chips);
      for (const o of st.options) {
        const b = h('button', {
          class: 'chip', attrs: { type: 'button', 'aria-pressed': st.off.has(o.key) ? 'false' : 'true' },
          on: {
            click: () => {
              if (st.off.has(o.key)) st.off.delete(o.key);
              else st.off.add(o.key);
              b.setAttribute('aria-pressed', st.off.has(o.key) ? 'false' : 'true');
              st.limit = PAGE;
              rerender();
            },
          },
        }, swatchEl(o.color, 'dot'), h('span', { text: o.label }));
        chips.appendChild(b);
      }
      clear(legendEl);
      const worn = (sp.lanes || []).some((ln) => ln.key !== 'group');
      const items = [];
      if (m.mode === 'group') items.push({ label: 'Underline: the diarized voice of a word', color: 'var(--ink-2)', kind: 'line' });
      if (worn && m.wordClasses) items.push({ label: 'Dimmed: worn-mic words given to someone else (crosstalk) or to no wearer', color: 'var(--muted)', kind: 'rect' });
      if (items.length) legendEl.appendChild(legend(items));
    }
    const filterKey = `${ctx.version}|${rangeKey}|${st.query}|${Array.from(st.off).sort().join(',')}|${st.limit}`;
    if (filterKey === st.filterKey) {
      settle(c, ctx, ['speech']);
      return;
    }
    st.filterKey = filterKey;
    st.version = ctx.version;
    const all = sp.transcript || [];
    const inRange = all.filter((ch) => !ctx.range || (ch.t1 >= ctx.range[0] && ch.t0 <= ctx.range[1]));
    const allOn = st.off.size === 0;
    const matched = [];
    for (const ch of inRange) {
      if (!allOn) {
        const ks = chunkKeys(ch, m);
        if (!ks.size || !Array.from(ks).some((k) => !st.off.has(k))) continue;
      }
      const words = wordsOf(ch);
      if (st.query) {
        const joined = words.map((w) => w.text).join(' ').toLowerCase();
        if (!joined.includes(st.query)) continue;
      }
      matched.push([ch, words]);
    }
    const paginate = matched.length > PAGINATE_OVER;
    const shown = paginate ? matched.slice(0, st.limit) : matched;
    clear(list);
    const frag = document.createDocumentFragment();
    for (const [ch, words] of shown) frag.appendChild(chunkRow(ch, m, words));
    list.appendChild(frag);
    more.hidden = !paginate || shown.length >= matched.length;
    const filtered = st.query || !allOn;
    let msg;
    if (!inRange.length) msg = ctx.range ? 'No transcript chunk in this range.' : 'This session has no transcripts.';
    else if (filtered) msg = `${fmt.int(matched.length)} of ${fmt.int(inRange.length)} chunks match`;
    else msg = `${fmt.int(inRange.length)} ${inRange.length === 1 ? 'chunk' : 'chunks'}${ctx.range ? ' in the range' : ''}`;
    if (paginate && shown.length < matched.length) msg += `, showing the first ${fmt.int(shown.length)}`;
    count.textContent = `${msg}.`;
    if (!matched.length && inRange.length) list.appendChild(h('li', { class: 'tx-none' }, emptyState({ title: 'No chunk matches the search and the speakers chosen.' })));
    markCursor(false);
    settle(c, ctx, ['speech']);
  }

  function markCursor(scroll) {
    const t = st.cursor;
    let best = null;
    for (const li of list.children) {
      li.classList.remove('is-current');
      const t0 = Number(li.dataset.t0);
      if (finite(t) && finite(t0) && t0 <= t + 0.01) best = li;
    }
    if (best && finite(t)) {
      best.classList.add('is-current');
      if (scroll) list.scrollTop = Math.max(0, best.offsetTop - list.offsetTop - 8);
    }
  }

  return {
    card: c,
    render,
    setCursor(t) {
      st.cursor = t;
      markCursor(true);
    },
  };
}

function wordClassLabel(cls) {
  if (cls === 'wearer') return 'said by the wearer';
  if (cls === 'crosstalk') return 'crosstalk: another wearer';
  if (cls === 'other') return 'not a wearer';
  return null;
}

function debounceInput(fn) {
  let id = 0;
  return () => {
    clearTimeout(id);
    id = setTimeout(fn, 160);
  };
}

// space

function spaceSection(ctx) {
  const sec = sectionShell('space', 'Space', 'Badge positions on the floor (IPS), top view in metres.');
  const roomCard = card({ title: 'Room', subtitle: 'Top view: v points away from the main camera', span: 7, table: () => roomTable(ctx) });
  const proxCard = card({ title: 'Proximity', subtitle: 'Median distance per pair while both badges were seen', span: 5, table: () => proximityTable(ctx) });
  const distCard = card({ title: 'Distance over time', subtitle: 'Median distance per 10 s, one chart per pair. Drag to narrow the range.', span: 8, table: () => distanceTable(ctx) });
  const moveCard = card({ title: 'Movement', subtitle: 'Floor distance walked per minute present', span: 4, table: () => roomTable(ctx) });
  sec.grid.append(roomCard.el, proxCard.el, distCard.el, moveCard.el);
  const opts = { modality: 'ips', emptyText: 'No IPS data in this session.' };
  const charts = {};
  const room = { mode: 'heat', off: new Set(), map: null, version: -1, controls: null, chips: null, mapEl: h('div', { class: 'an-room' }) };
  const prox = { matrix: h('div'), table: h('div'), graph: h('div'), graphNote: h('p', { class: 'muted an-small' }) };
  prox.el = h('div', {}, prox.matrix, prox.table, h('h3', { class: 'an-sub', text: 'Facing' }), prox.graphNote, prox.graph);

  function roomControls() {
    const seg = segmented({
      label: 'Room view', size: 'sm', value: room.mode,
      options: [{ value: 'heat', label: 'Heat' }, { value: 'trails', label: 'Trails' }],
      onChange: (v) => {
        room.mode = v;
        renderRoom();
      },
    });
    roomCard.setActions([seg]);
    room.chips = h('div', { class: 'tx-chips an-room-tags', attrs: { role: 'group', 'aria-label': 'Badges shown' } });
    clear(roomCard.body);
    roomCard.body.append(room.chips, room.mapEl);
  }

  function renderRoom() {
    if (!gate(roomCard, ctx, ['space'], opts)) return;
    const space = ctx.parts.space;
    const pm = ctx.models.space;
    if (!pm.tags.length) {
      roomCard.setState('empty', 'No badge positions in this session.');
      return;
    }
    if (room.version !== ctx.version) {
      if (!room.chips) roomControls();
      clear(room.chips);
      for (const id of pm.kept) {
        const t = ctx.ident.tag(id);
        const b = h('button', {
          class: 'chip', attrs: { type: 'button', 'aria-pressed': room.off.has(id) ? 'false' : 'true' },
          on: {
            click: () => {
              if (room.off.has(id)) room.off.delete(id);
              else room.off.add(id);
              b.setAttribute('aria-pressed', room.off.has(id) ? 'false' : 'true');
              renderRoom();
            },
          },
        }, swatchEl(t.color, 'dot'), h('span', { text: t.label }));
        room.chips.appendChild(b);
      }
      if (room.map) room.map.destroy();
      room.map = floorMap(room.mapEl, { extent: space.extent || undefined, cameras: space.cameras || [], height: 420, label: 'Room plan, top view in metres' });
      room.version = ctx.version;
    }
    const on = pm.kept.filter((id) => !room.off.has(id));
    if (room.mode === 'heat') {
      const st = spaceStats(pm, ctx.range, on.length ? on : ['\u0000none']);
      room.map.setTrails(null);
      if (st.heat && on.length) {
        room.map.setHeat({ grid: st.heat, label: on.length === 1 ? `Time of Tag ${on[0]} per cell` : 'Time per cell', format: (v) => fmt.duration(v) });
      } else room.map.setHeat(null);
    } else {
      room.map.setHeat(null);
      const tracks = {};
      for (const id of on) {
        const tr = space.tracks[id];
        if (!tr) continue;
        const t = ctx.ident.tag(id);
        tracks[id] = { color: t.color, label: t.label, t: tr.t, u: tr.u, v: tr.v };
      }
      room.map.setTrails({ tracks, range: ctx.range, timeFormat: ctx.clock });
    }
    const notes = [];
    if (space.floor && space.floor.method === 'camera-xz') notes.push('Too few badge rotations to find the floor: positions use the camera\'s x and z axes.');
    const rareIds = pm.tags.filter((id) => pm.rare.has(id));
    if (rareIds.length) notes.push(`Rare reads left out: ${rareIds.map((x) => `Tag ${x}`).join(', ')}.`);
    if (room.mode === 'heat') notes.push(on.length > 1 ? 'Heat pools the badges chosen.' : on.length ? '' : 'Choose a badge to show its heat.');
    settle(roomCard, ctx, ['space'], notes.filter(Boolean).join(' '));
  }

  function renderProximity(d) {
    if (!gate(proxCard, ctx, ['space'], opts)) return;
    const pm = ctx.models.space;
    if (pm.kept.length < 2) {
      proxCard.setState('empty', 'Fewer than two badges were seen, so there are no pairs.');
      return;
    }
    const space = ctx.parts.space;
    if (!d.spc || !d.spc.pairs.some((p) => p.copresent > 0)) {
      proxCard.setState('empty', ctx.range ? 'No two badges were seen together in this range.' : 'No two badges were seen together.');
      return;
    }
    if (!prox.el.isConnected) proxCard.body.appendChild(prox.el);
    const rows = pm.kept.map((id) => ({ key: id, label: ctx.ident.tag(id).label, swatch: ctx.ident.tag(id).color }));
    const byPair = new Map(d.spc.pairs.map((p) => [pairKey(p.a, p.b), p]));
    const values = pm.kept.map((a) => pm.kept.map((b) => (a === b ? null : (byPair.get(pairKey(a, b)) || {}).median ?? null)));
    const o = { rows, cols: rows, values, diagonal: 'blank', format: (v) => fmt.metres(v), valueLabel: 'median distance', title: (r, c) => `${r.label} and ${c.label}`, scaleLabel: 'Median distance', cellSize: 52, label: 'Median distance per pair' };
    if (charts.prox) charts.prox.update(o);
    else charts.prox = heatMatrix(prox.matrix, o);
    clear(prox.table);
    const mutual = new Map((space.mutual_facing || []).map((m) => [pairKey(m.a, m.b), m.count]));
    prox.table.appendChild(simpleTable(
      [{ key: 'pair', label: 'Pair' }, { key: 'median', label: 'Median', align: 'right' }, { key: 'lt1', label: 'Under 1 m', align: 'right' }, { key: 'co', label: 'Seen together', align: 'right' }],
      d.spc.pairs.map((p) => ({ pair: pairLabel(pairKey(p.a, p.b)), median: fmt.metres(p.median), lt1: fmt.pct(p.lt1), co: fmt.duration(p.copresent) })),
      'Pairs',
    ));
    const nodes = pm.kept.map((id) => {
      const s = d.spc.tags[id];
      return { key: id, label: ctx.ident.tag(id).label, color: ctx.ident.tag(id).color, x: s ? s.u : null, y: s ? s.v : null };
    });
    const edges = (space.facing || []).filter((f) => f.count > 0 && !pm.rare.has(f.src) && !pm.rare.has(f.tgt)).map((f) => ({
      from: f.src, to: f.tgt, weight: f.ratio,
      label: `${fmt.pct(f.ratio, 1)} of ${fmt.int(f.copresent)} windows together (${fmt.int(f.count)})`,
    }));
    const g = { nodes, edges, layout: nodes.some((n) => finite(n.x)) ? 'given' : 'circle', height: 220, format: (v) => fmt.pct(v, 1), weightLabel: 'facing', label: 'Who faces whom' };
    if (charts.facing) charts.facing.update(g);
    else charts.facing = networkGraph(prox.graph, g);
    const mtext = Array.from(mutual.entries()).filter(([, n]) => n > 0).map(([k, n]) => `${pairLabel(k)} ${fmt.int(n)}`).join(', ');
    prox.graphNote.textContent = `Arrows: share of the windows two badges were seen together in which one faced the other (whole session). Badges at their median positions${ctx.range ? ' in the range' : ''}.${mtext ? ` Windows facing each other: ${mtext}.` : ''}`;
    settle(proxCard, ctx, ['space']);
  }

  function renderDistance(d) {
    if (!gate(distCard, ctx, ['space'], opts)) return;
    const space = ctx.parts.space;
    const series = [];
    for (const p of d.pairs) {
      const sp = (space.pairs || []).find((q) => pairKey(q.a, q.b) === p.key);
      if (sp && sp.series) series.push({ key: p.key, label: p.label, a: p.a, b: p.b, color: PAIR_INK, t: sp.series.t, v: sp.series.d });
    }
    if (!series.some((se) => se.v.some(finite))) {
      distCard.setState('empty', 'No two badges were seen together.');
      return;
    }
    // small multiples, one per pair, on one scale
    const domain = distanceDomain(series);
    const sig = `${ctx.version}|${series.map((x) => x.key).join(',')}`;
    if (!charts.dist || charts.dist.sig !== sig) {
      if (charts.dist) for (const m of charts.dist.list) m.chart.destroy();
      clear(distCard.body);
      const list = series.map((se) => {
        const head = h('div', { class: 'an-pair-head' },
          swatchEl(ctx.ident.tag(se.a).color, 'dot'), swatchEl(ctx.ident.tag(se.b).color, 'dot'), h('span', { text: se.label }), h('span', { class: 'an-pair-stat' }));
        const el = h('div');
        distCard.body.append(head, el);
        return { key: se.key, head, el, chart: null };
      });
      charts.dist = { sig, list };
    }
    const byPair = new Map(d.spc ? d.spc.pairs.map((p) => [pairKey(p.a, p.b), p]) : []);
    series.forEach((se, i) => {
      const m = charts.dist.list[i];
      const o = {
        // the y ticks name the reference lines already
        span: ctx.range || ctx.span, series: [{ ...se, label: 'distance' }], domain, refs: [{ v: 0.5, label: '' }, { v: 1, label: '' }],
        format: (v) => fmt.metres(v, 1), height: 96, marginLeft: 44, brush: true, xFormat: ctx.clock, label: `Distance over time, ${se.label}`,
        onBrush: (r) => {
          if (r) ctx.setRange(r, 'distance');
        },
      };
      if (m.chart) m.chart.update(o);
      else m.chart = lineChart(m.el, o);
      m.chart.setBrush(null);
      const st = byPair.get(se.key);
      m.head.lastChild.textContent = st && finite(st.median) ? `median ${fmt.metres(st.median)}` : '';
    });
    distCard.setLegend(null);
    settle(distCard, ctx, ['space']);
  }

  function renderMovement(d) {
    if (!gate(moveCard, ctx, ['space'], opts)) return;
    const pm = ctx.models.space;
    if (!pm.kept.length) {
      moveCard.setState('empty', 'No badge positions in this session.');
      return;
    }
    const items = pm.kept.map((id) => {
      const s = d.spc.tags[id];
      const t = ctx.ident.tag(id);
      return { key: id, label: t.label, value: s ? s.mPerMin : null, color: t.color, note: s ? `${fmt.duration(s.seconds)} seen` : '' };
    });
    const o = { items, format: (v) => `${fmt.num(v, 2)}${NBSP}m/min`, valueLabel: 'walked per minute', noteLabel: 'badge seen', label: 'Movement per badge' };
    if (charts.move) charts.move.update(o);
    else charts.move = barList(moveCard.body, o);
    settle(moveCard, ctx, ['space']);
  }

  return {
    ...sec,
    render() {
      const d = derive(ctx);
      renderRoom();
      renderProximity(d);
      renderDistance(d);
      renderMovement(d);
    },
  };
}

function roomTable(ctx) {
  const d = derive(ctx);
  if (!d.spc) return null;
  const rows = d.pm.tags.map((id) => {
    const s = d.spc.tags[id];
    return {
      tag: tagCell(ctx, id), seen: fmt.duration(s.seconds), path: fmt.metres(s.path, 1), rate: fmt.num(s.mPerMin, 2),
      active: fmt.duration(s.active), u: fmt.num(s.u, 2), v: fmt.num(s.v, 2), rare: s.rare ? 'rare read' : '',
    };
  });
  return {
    columns: [
      { key: 'tag', label: 'Tag' }, { key: 'seen', label: 'Seen', align: 'right' }, { key: 'path', label: 'Walked', align: 'right' },
      { key: 'rate', label: 'm/min', align: 'right' }, { key: 'active', label: 'Moving', align: 'right' }, { key: 'u', label: 'Median u (m)', align: 'right' },
      { key: 'v', label: 'Median v (m)', align: 'right' }, { key: 'rare', label: 'Note' },
    ],
    rows,
  };
}

function proximityTable(ctx) {
  const d = derive(ctx);
  if (!d.spc) return null;
  const space = ctx.parts.space;
  const rows = d.spc.pairs.map((p) => {
    const k = pairKey(p.a, p.b);
    const f1 = (space.facing || []).find((f) => f.src === p.a && f.tgt === p.b);
    const f2 = (space.facing || []).find((f) => f.src === p.b && f.tgt === p.a);
    const m = (space.mutual_facing || []).find((x) => pairKey(x.a, x.b) === k);
    return {
      pair: pairLabel(k), median: fmt.metres(p.median), lt05: fmt.pct(p.lt05), lt1: fmt.pct(p.lt1), co: fmt.duration(p.copresent),
      ab: f1 ? fmt.pct(f1.ratio, 1) : fmt.na, ba: f2 ? fmt.pct(f2.ratio, 1) : fmt.na, mutual: m ? fmt.int(m.count) : fmt.na,
    };
  });
  return {
    columns: [
      { key: 'pair', label: 'Pair' }, { key: 'median', label: 'Median', align: 'right' }, { key: 'lt05', label: 'Under 0.5 m', align: 'right' },
      { key: 'lt1', label: 'Under 1 m', align: 'right' }, { key: 'co', label: 'Seen together', align: 'right' },
      { key: 'ab', label: 'First faces second', align: 'right' }, { key: 'ba', label: 'Second faces first', align: 'right' },
      { key: 'mutual', label: 'Facing each other (windows)', align: 'right' },
    ],
    rows,
  };
}

function distanceTable(ctx) {
  const space = ctx.parts.space;
  const d = derive(ctx);
  if (!space) return null;
  const pairs = d.pairs.map((p) => ({ p, s: (space.pairs || []).find((q) => pairKey(q.a, q.b) === p.key) })).filter((x) => x.s && x.s.series);
  if (!pairs.length) return null;
  const t = pairs[0].s.series.t;
  const rows = [];
  for (let i = 0; i < t.length; i += 1) {
    if (ctx.range && (t[i] + 10 < ctx.range[0] || t[i] > ctx.range[1])) continue;
    const r = { t: ctx.clock(t[i]) };
    for (const { p, s } of pairs) r[p.key] = fmt.metres(s.series.d[i]);
    rows.push(r);
  }
  return { columns: [{ key: 't', label: 'Window from' }].concat(pairs.map(({ p }) => ({ key: p.key, label: p.label, align: 'right' }))), rows };
}

// attention

function attentionSection(ctx) {
  const sec = sectionShell('attention', 'Attention', 'Gaze, joint attention and hands from the cameras (video analysis, 10 s windows).');
  const looksCard = card({ title: 'Who looks at whom', subtitle: 'Share of each pupil\'s frame sets with the gaze on a partner', span: 5, table: () => looksTable(ctx) });
  const whereCard = card({ title: 'Where each pupil looks', subtitle: 'Share of frames per gaze target', span: 7, table: () => whereTable(ctx) });
  const jaCard = card({ title: 'Joint attention', subtitle: 'Baseline: the pair\'s own rate 20 to 40 s earlier.', span: 6, table: () => jaTable(ctx) });
  const handsCard = card({ title: 'Hands', subtitle: 'Share of hand steps with the hands moving', span: 6, table: () => handsTable(ctx) });
  const camCard = card({ title: 'Camera coverage', subtitle: 'Share of each camera\'s frame sets in which it saw the pupil', span: 6, table: () => cameraTable(ctx) });
  const qualCard = card({ title: 'Video data quality', subtitle: 'How complete the video features are', span: 6 });
  sec.grid.append(looksCard.el, whereCard.el, jaCard.el, handsCard.el, camCard.el, qualCard.el);
  const opts = { modality: 'vfa', emptyText: 'No video features in this session.' };
  const charts = {};
  const looks = { kind: 'face', seg: null };
  const handsBody = { bars: h('div'), pairs: h('div') };
  handsBody.el = h('div', {}, handsBody.bars, h('h3', { class: 'an-sub', text: 'Pairs' }), handsBody.pairs);

  function renderLooks(d) {
    if (!gate(looksCard, ctx, ['attention'], opts)) return;
    const att = ctx.parts.attention;
    if (!looks.seg) {
      looks.seg = segmented({
        label: 'Gaze target', size: 'sm', value: looks.kind,
        options: [{ value: 'face', label: 'Face' }, { value: 'hands', label: 'Hands' }],
        onChange: (v) => {
          looks.kind = v;
          renderLooks(derive(ctx));
        },
      });
      looksCard.setActions([looks.seg]);
    }
    const pupils = att.pupils || [];
    if (!pupils.length || !d.at) {
      looksCard.setState('empty', 'No pupil was seen on the cameras.');
      return;
    }
    const targets = d.at.looks.targets;
    const rows = pupils.map((id) => ({ key: id, label: ctx.ident.tag(id).label, swatch: ctx.ident.tag(id).color }));
    const cols = targets.map((tg) => (tg === 'other' ? { key: 'other', label: 'Others' } : { key: tg, label: ctx.ident.tag(tg).label, swatch: ctx.ident.tag(tg).color }));
    const values = d.at.looks[looks.kind].map((r) => r.share);
    const o = {
      rows, cols, values, diagonal: 'blank', format: (v) => fmt.pct(v, 1), valueLabel: `of the looker's frame sets`, max: undefined,
      title: (r, c) => `${r.label} looks at ${c.key === 'other' ? 'others\'' : `${c.label}'s`} ${looks.kind}`, scaleLabel: 'Share', label: 'Who looks at whom',
    };
    if (charts.looks) charts.looks.update(o);
    else charts.looks = heatMatrix(looksCard.body, o);
    settle(looksCard, ctx, ['attention'], 'Others: any person who is not a pupil, untagged bodies included.');
  }

  function renderWhere(d) {
    if (!gate(whereCard, ctx, ['attention'], opts)) return;
    const att = ctx.parts.attention;
    const pupils = (att.pupils || []).filter((id) => d.at && d.at.allocFrames[id]);
    if (!pupils.length) {
      whereCard.setState('empty', 'No pupil was seen on the cameras.');
      return;
    }
    const rows = pupils.map((id) => ({ key: id, label: ctx.ident.tag(id).label, swatch: ctx.ident.tag(id).color, values: d.at.allocFrames[id] }));
    const o = { rows, categories: CATEGORIES, normalize: true, format: (v) => `${fmt.int(v)} frames`, valueLabel: 'frames', label: 'Gaze target shares per pupil' };
    if (charts.where) charts.where.update(o);
    else charts.where = stackedBars(whereCard.body, o);
    whereCard.setLegend(CATEGORIES.map((c) => ({ label: c.label, color: c.fill, kind: c.fill === 'hatch' ? 'hatch' : 'rect' })));
    settle(whereCard, ctx, ['attention'], 'Task: own hands, the work area or a work zone. Unreadable: gaze out of frame or not estimated.');
  }

  function renderJa(d) {
    if (!gate(jaCard, ctx, ['attention', 'timeline'], opts)) return;
    const pairs = d.pairs.filter((p) => d.tl && (d.tl.pairs[p.key] || d.tl.pairs[`${p.b}|${p.a}`]));
    if (!pairs.length) {
      jaCard.setState('empty', 'No two pupils were seen on the same camera.');
      return;
    }
    let max = 0;
    const items = pairs.map((p) => {
      const s = d.tl.pairs[p.key] || d.tl.pairs[`${p.b}|${p.a}`];
      const few = !(s.windows >= FEW_WINDOWS);
      if (finite(s.ratio)) max = Math.max(max, s.ratio);
      if (finite(s.baseline)) max = Math.max(max, s.baseline);
      return { key: p.key, label: few ? `${p.label} (${fmt.int(s.windows)} ${s.windows === 1 ? 'window' : 'windows'})` : p.label, swatch: few ? 'var(--tag-other)' : p.color, value: s.ratio, baseline: s.baseline };
    });
    const dom = [0, Math.min(1, Math.max(0.2, Math.ceil(max * 11) / 10))];
    const o = { items: items.map((it) => ({ ...it, domain: dom })), format: (v) => fmt.pct(v), valueLabel: 'gaze points together', baselineLabel: 'baseline', excessLabel: 'above baseline', label: 'Joint attention per pair' };
    if (charts.ja) charts.ja.update(o);
    else charts.ja = bullet(jaCard.body, o);
    const few = items.filter((it) => it.swatch === 'var(--tag-other)').length;
    settle(jaCard, ctx, ['attention', 'timeline'], [
      'Share of frames with both gaze points within 5 % of the frame width, averaged over 10 s windows weighted by frames; the tick marks the baseline.',
      few ? `Grey pairs rest on fewer than ${FEW_WINDOWS} windows.` : null,
      ctx.range ? 'In a range the windows weigh by the frames both pupils were seen in.' : null,
    ].filter(Boolean).join(' '));
  }

  function renderHands(d) {
    if (!gate(handsCard, ctx, ['attention', 'timeline'], opts)) return;
    if (!handsBody.el.isConnected) handsCard.body.appendChild(handsBody.el);
    const att = ctx.parts.attention;
    const pupils = att.pupils || [];
    if (!pupils.length) {
      handsCard.setState('empty', 'No pupil was seen on the cameras.');
      return;
    }
    const items = pupils.map((id) => {
      const t = ctx.ident.tag(id);
      const v = d.tl && d.tl.tags[id] ? d.tl.tags[id].handsActive : (att.hands[id] || {}).active;
      const still = !ctx.range && att.hands[id] ? att.hands[id].still : null;
      return { key: id, label: t.label, value: v, color: t.color, note: finite(still) ? `still ${fmt.pct(still)}` : '' };
    });
    const o = { items, format: (v) => fmt.pct(v), max: 1, valueLabel: 'hands moving', noteLabel: '', label: 'Hands active per pupil' };
    if (charts.hands) charts.hands.update(o);
    else charts.hands = barList(handsBody.bars, o);
    clear(handsBody.pairs);
    const rows = (att.pair_hands || []).map((p) => {
      const k = pairKey(p.a, p.b);
      const s = d.tl ? d.tl.pairs[k] || d.tl.pairs[`${p.b}|${p.a}`] : null;
      return { pair: pairLabel(k), close: ctx.range && s && finite(s.close) ? fmt.pct(s.close) : fmt.pct(p.close), follow: fmt.pct(p.follow), both: fmt.pct(p.both_active), one: fmt.pct(p.one_active) };
    });
    if (rows.length) {
      handsBody.pairs.appendChild(simpleTable([
        { key: 'pair', label: 'Pair' },
        { key: 'close', label: 'Hands close', align: 'right', title: 'Share of frames with both pupils\' hands within a hand length of each other' },
        { key: 'follow', label: 'Follow', align: 'right', title: 'Share of steps where one pupil\'s hands move right after the other\'s (whole session)' },
        { key: 'both', label: 'Both active', align: 'right', title: 'Whole session' },
        { key: 'one', label: 'One active', align: 'right', title: 'Whole session' },
      ], rows, 'Hands per pair'));
    } else handsBody.pairs.appendChild(h('p', { class: 'muted an-small', text: 'No pair data.' }));
    settle(handsCard, ctx, ['attention', 'timeline'], ctx.range ? 'In a range: hands active and hands close follow the range; follow, both active and one active cover the whole session.' : null);
  }

  function renderCameras() {
    if (!gate(camCard, ctx, ['attention'], opts)) return;
    const att = ctx.parts.attention;
    const cc = att.camera_coverage || {};
    if (!(cc.cameras || []).length || !(cc.tags || []).length) {
      camCard.setState('empty', 'No camera saw a pupil.');
      return;
    }
    const rows = cc.cameras.map((id) => ({ key: id, label: id }));
    const cols = cc.tags.map((id) => ({ key: id, label: ctx.ident.tag(id).label, swatch: ctx.ident.tag(id).color }));
    const o = { rows, cols, values: cc.share, format: (v) => fmt.pct(v), max: 1, valueLabel: 'of the camera\'s frame sets', title: (r, c) => `${r.label} sees ${c.label}`, scaleLabel: 'Share', label: 'Camera coverage' };
    if (charts.cams) charts.cams.update(o);
    else charts.cams = heatMatrix(camCard.body, o);
    settle(camCard, ctx, ['attention'], ctx.range ? RANGE_NOTE : null);
  }

  function renderQuality() {
    if (!gate(qualCard, ctx, ['attention'], opts)) return;
    const att = ctx.parts.attention;
    const q = att.quality || {};
    clear(qualCard.body);
    const unknown = Object.keys(q.unknown_share || {}).map((id) => `${ctx.ident.tag(id).label} ${fmt.pct(q.unknown_share[id])}`).join(', ');
    qualCard.body.appendChild(dl([
      ['Frame sets', fmt.int(q.frame_sets), 'One frame set per second: a frame from every camera'],
      ['Incomplete frame sets', fmt.int(q.incomplete), 'Frame sets missing a camera'],
      ['Camera frames', fmt.int(q.camera_frames)],
      ['Gaze errors', fmt.int(q.gaze_errors)],
      ['Gaze not estimated', unknown || fmt.na, 'Share of each pupil\'s frames whose gaze target is unknown'],
      ['Untagged bodies per frame', fmt.num(q.untagged_per_frame, 2), 'People the pose model found without a readable badge'],
      ['Frames with a badge read twice', fmt.int(q.duplicate_tag_frames)],
      ['Pose model', q.pose_model || fmt.na],
      ['Cameras', (att.cameras || []).map((c) => `${c.id} ${c.width}x${c.height}`).join(', ') || fmt.na],
    ]));
    settle(qualCard, ctx, ['attention'], ctx.range ? RANGE_NOTE : null);
  }

  return {
    ...sec,
    render() {
      const d = derive(ctx);
      renderLooks(d);
      renderWhere(d);
      renderJa(d);
      renderHands(d);
      renderCameras();
      renderQuality();
    },
  };
}

function looksTable(ctx) {
  const d = derive(ctx);
  const att = ctx.parts.attention;
  if (!d.at || !att) return null;
  const rows = [];
  (att.pupils || []).forEach((id, li) => {
    d.at.looks.targets.forEach((tg, j) => {
      if (tg === id) return;
      rows.push({
        looker: tagCell(ctx, id), target: tg === 'other' ? 'Others' : tagCell(ctx, tg),
        face: fmt.int(d.at.looks.face[li].hits[j]), faceShare: fmt.pct(d.at.looks.face[li].share[j], 1),
        hands: fmt.int(d.at.looks.hands[li].hits[j]), handsShare: fmt.pct(d.at.looks.hands[li].share[j], 1),
        seen: fmt.int(d.at.looks.face[li].seen),
      });
    });
  });
  return {
    columns: [
      { key: 'looker', label: 'Looker' }, { key: 'target', label: 'Target' }, { key: 'face', label: 'Face, frame sets', align: 'right' },
      { key: 'faceShare', label: 'Face share', align: 'right' }, { key: 'hands', label: 'Hands, frame sets', align: 'right' },
      { key: 'handsShare', label: 'Hands share', align: 'right' }, { key: 'seen', label: 'Looker seen', align: 'right' },
    ],
    rows,
  };
}

function whereTable(ctx) {
  const d = derive(ctx);
  const att = ctx.parts.attention;
  if (!d.at || !att) return null;
  const rows = (att.pupils || []).filter((id) => d.at.allocation[id]).map((id) => {
    const r = { tag: tagCell(ctx, id), frames: fmt.int(d.at.frames[id]) };
    for (const c of CATEGORIES) r[c.key] = fmt.pct(d.at.allocation[id][c.key], 1);
    return r;
  });
  return { columns: [{ key: 'tag', label: 'Pupil' }, { key: 'frames', label: 'Frames', align: 'right' }].concat(CATEGORIES.map((c) => ({ key: c.key, label: c.label, align: 'right' }))), rows };
}

function jaTable(ctx) {
  const d = derive(ctx);
  if (!d.tl) return null;
  const rows = d.pairs.map((p) => {
    const s = d.tl.pairs[p.key] || d.tl.pairs[`${p.b}|${p.a}`];
    if (!s) return null;
    return { pair: p.label, ratio: fmt.pct(s.ratio, 1), base: fmt.pct(s.baseline, 1), ex: fmt.pp(s.excess), windows: fmt.int(s.windows), mutual: fmt.int(s.mutual) };
  }).filter(Boolean);
  return {
    columns: [
      { key: 'pair', label: 'Pair' }, { key: 'ratio', label: 'Joint attention', align: 'right' }, { key: 'base', label: 'Baseline', align: 'right' },
      { key: 'ex', label: 'Above baseline', align: 'right' }, { key: 'windows', label: 'Windows', align: 'right' }, { key: 'mutual', label: 'Mutual gaze frames', align: 'right' },
    ],
    rows,
  };
}

function handsTable(ctx) {
  const d = derive(ctx);
  const att = ctx.parts.attention;
  if (!att) return null;
  const rows = (att.pupils || []).map((id) => ({
    tag: tagCell(ctx, id),
    active: fmt.pct(d.tl && d.tl.tags[id] ? d.tl.tags[id].handsActive : (att.hands[id] || {}).active, 1),
    still: fmt.pct((att.hands[id] || {}).still, 1),
    steps: fmt.int((att.hands[id] || {}).steps),
  }));
  return { columns: [{ key: 'tag', label: 'Pupil' }, { key: 'active', label: 'Active', align: 'right' }, { key: 'still', label: 'Still (session)', align: 'right' }, { key: 'steps', label: 'Steps (session)', align: 'right' }], rows };
}

function cameraTable(ctx) {
  const att = ctx.parts.attention;
  if (!att || !att.camera_coverage) return null;
  const cc = att.camera_coverage;
  const rows = [];
  cc.cameras.forEach((cam, i) => cc.tags.forEach((tag, j) => rows.push({ cam, tag: tagCell(ctx, tag), share: fmt.pct(cc.share[i][j], 1), frames: fmt.int(cc.frames ? cc.frames[i][j] : null) })));
  return { columns: [{ key: 'cam', label: 'Camera' }, { key: 'tag', label: 'Pupil' }, { key: 'share', label: 'Share', align: 'right' }, { key: 'frames', label: 'Frame sets', align: 'right' }], rows };
}

// data and exports

const EVENT_TYPES = [
  ['asr_recognition', 'Speech buckets (ASR recognition)'],
  ['asr_transcription', 'Transcripts (ASR transcription)'],
  ['ips_translation', 'Badge positions (IPS translation)'],
  ['ips_rotation', 'Badge rotations (IPS rotation)'],
  ['ips_relation', 'Facing relations (IPS relation)'],
  ['vfa_features', 'Video features (VFA)'],
];

export function exportLinks(ctx) {
  const base = `/api/sessions/${encodeURIComponent(ctx.sid)}/export/`;
  const mods = (ctx.meta && ctx.meta.modalities) || {};
  const has = { asr: !!mods.asr, ips: !!mods.ips, vfa: !!mods.vfa };
  const videoReady = !!ctx.parts.timeline;
  return [
    { group: 'Report', items: [{ name: 'report.json', label: 'Report (JSON, every computed part)' }, { name: 'window_features.csv', label: 'Window features (CSV, 10 s windows)', disabled: !videoReady, reason: 'after the video analysis' }] },
    { group: 'Transcript', items: [{ name: 'transcript.txt', label: 'Transcript (text)', disabled: !has.asr }, { name: 'transcript.srt', label: 'Transcript (subtitles, SRT)', disabled: !has.asr }] },
    { group: 'Raw events (JSON lines)', items: EVENT_TYPES.map(([t, label]) => ({ name: `${t}.jsonl`, label, disabled: !has[t.slice(0, 3)] })) },
  ].map((g) => ({ ...g, items: g.items.map((it) => ({ ...it, href: `${base}${it.name}` })) }));
}

function dataSection(ctx) {
  const sec = sectionShell('data', 'Data and exports');
  const covCard = card({ title: 'Coverage', subtitle: 'Records per modality against one per bucket over the session', span: 7 });
  const dlCard = card({ title: 'Downloads', subtitle: 'Files are named after the session', span: 5 });
  const provCard = card({ title: 'Devices and provenance', subtitle: 'What recorded the session and which code computed it', span: 12 });
  sec.grid.append(covCard.el, dlCard.el, provCard.el);
  let provOpen = false;

  function renderCoverage() {
    const meta = ctx.meta;
    if (!meta) return;
    const m = meta.modalities || {};
    const dur = meta.duration || 0;
    const sp = ctx.parts.speech;
    const space = ctx.parts.space;
    const att = ctx.parts.attention;
    const rows = [];
    if (m.asr) {
      const rec = m.asr.recognition || {};
      const gaps = sp && sp.coverage ? sp.coverage.gaps || [] : null;
      const gapSec = gaps ? sum(gaps.map((g) => g[1] - g[0])) : null;
      rows.push({ what: 'ASR speech buckets (3 s)', n: fmt.int(rec.n), expected: fmt.int(Math.round(dur / 3)), cov: fmt.pct(rec.coverage), note: gaps == null ? 'gaps after the speech analysis' : gaps.length ? `${fmt.int(gaps.length)} ${gaps.length === 1 ? 'gap' : 'gaps'}, ${fmt.duration(gapSec)}` : 'no gaps' });
      rows.push({ what: 'ASR transcripts', n: fmt.int((m.asr.transcription || {}).n), expected: fmt.na, cov: fmt.na, note: `${fmt.int(m.asr.worn_transcripts || 0)} worn mic, ${fmt.int(m.asr.diarized_transcripts || 0)} diarized` });
    } else rows.push({ what: 'ASR', n: '0', expected: fmt.na, cov: fmt.na, note: 'not recorded' });
    if (m.ips) {
      const c = space && space.coverage;
      rows.push({ what: 'IPS badge windows (1 s)', n: fmt.int(m.ips.n), expected: fmt.int(Math.round(dur)), cov: fmt.pct(m.ips.coverage), note: c ? `${fmt.int(c.windows - c.nonempty)} windows with no badge` : '' });
    } else rows.push({ what: 'IPS', n: '0', expected: fmt.na, cov: fmt.na, note: 'not recorded' });
    if (m.vfa) {
      const q = att && att.quality;
      rows.push({ what: 'VFA frame sets (1 s)', n: fmt.int(m.vfa.n), expected: fmt.int(Math.round(dur)), cov: fmt.pct(m.vfa.coverage), note: q ? `${fmt.int(q.incomplete)} incomplete` : '' });
    } else rows.push({ what: 'VFA', n: '0', expected: fmt.na, cov: fmt.na, note: 'not recorded' });
    clear(covCard.body);
    covCard.body.appendChild(simpleTable([
      { key: 'what', label: 'Modality' }, { key: 'n', label: 'Records', align: 'right' }, { key: 'expected', label: 'Expected', align: 'right' },
      { key: 'cov', label: 'Coverage', align: 'right' }, { key: 'note', label: 'Notes' },
    ], rows, 'Coverage per modality'));
  }

  function renderDownloads() {
    clear(dlCard.body);
    for (const g of exportLinks(ctx)) {
      const ul = h('ul', { class: 'an-dl' });
      for (const it of g.items) {
        ul.appendChild(h('li', {}, it.disabled
          ? h('span', { class: 'muted an-dl-off' }, icon('download', 14), h('span', { text: `${it.label}${it.reason ? `, ${it.reason}` : ', no data'}` }))
          : downloadLink(it.href, it.label)));
      }
      dlCard.body.append(h('h3', { class: 'an-sub', text: g.group }), ul);
    }
    dlCard.setNote('Raw events stream from InfluxDB; video features run to about 80 MB per hour.');
  }

  function renderProvenance() {
    const meta = ctx.meta;
    if (!meta) return;
    const p = meta.provenance || {};
    const dev = meta.devices || {};
    clear(provCard.body);
    const details = h('details', { class: 'an-details', open: provOpen || null });
    details.addEventListener('toggle', () => {
      provOpen = details.open;
    });
    details.appendChild(h('summary', { text: 'Show devices and provenance' }));
    const yesNo = (v) => (v === true ? 'yes' : v === false ? 'no' : null);
    details.appendChild(dl([
      ['Cameras', (dev.cameras || []).join(', ') || null],
      ['Microphones', (dev.microphones || []).join(', ') || null],
      ['IPS main camera', meta.ips_main || null],
      ['Transcriber', p.transcriber || null],
      ['Language', p.language || null],
      ['Diarization', yesNo(p.diarize)],
      ['Pose model', p.pose_model || null],
      ['Gaze', yesNo(p.gaze)],
      ['Git commit', p.git_commit ? h('span', { class: 'mono', text: p.git_commit }) : null],
      ['openmmla', p.openmmla || null],
      ['Session record', meta.mongo ? 'MongoDB' : 'none (InfluxDB only)'],
    ]));
    const comps = p.components || [];
    if (comps.length) {
      details.appendChild(simpleTable([
        { key: 'key', label: 'Component' }, { key: 'host', label: 'Host' }, { key: 'commit', label: 'Git commit' }, { key: 'started', label: 'Started' },
      ], comps.map((c) => ({ key: h('span', { class: 'mono', text: c.key || fmt.na }), host: c.host || fmt.na, commit: c.git_commit ? h('span', { class: 'mono', text: c.git_commit }) : fmt.na, started: c.started_at ? fmt.dateTime(c.started_at) : fmt.na })), 'Components'));
    }
    if (!meta.provenance && !meta.devices) details.appendChild(h('p', { class: 'muted', text: 'This session has no MongoDB record, so devices and provenance are unknown.' }));
    provCard.body.appendChild(details);
  }

  return {
    ...sec,
    render() {
      renderCoverage();
      renderDownloads();
      renderProvenance();
    },
  };
}

/** builds every section into `container`; render() refreshes them all from ctx. */
export function buildSections(ctx, container) {
  const list = [overviewSection(ctx), speechSection(ctx), spaceSection(ctx), attentionSection(ctx), dataSection(ctx)];
  for (const s of list) container.appendChild(s.el);
  const byId = Object.fromEntries(list.map((s) => [s.el.id, s]));
  // a section redraws only when its own inputs changed, so polling one job's progress does not
  // redraw (and drop the hover of) charts that do not depend on it
  const deps = { overview: PART_LIST, speech: ['speech'], space: ['space'], attention: ['attention', 'timeline'], data: [] };
  const last = {};
  return {
    sections: list,
    render() {
      for (const s of list) {
        const sig = [ctx.version, ctx.range ? ctx.range.join(',') : '', ctx.refreshing ? 1 : 0]
          .concat((deps[s.el.id] || []).map((p) => JSON.stringify(ctx.status[p] || null))).join('|');
        if (last[s.el.id] === sig) continue;
        last[s.el.id] = sig;
        try {
          s.render();
        } catch (err) {
          // one broken panel must not take the page down
          setTimeout(() => {
            throw err;
          });
        }
      }
    },
    setCursor(t) {
      byId.overview.setCursor(t);
      byId.speech.setCursor(t);
    },
    timelineCard: byId.overview.timelineCard,
  };
}

export const SECTION_NAV = [
  { id: 'overview', label: 'Overview' },
  { id: 'speech', label: 'Speech' },
  { id: 'space', label: 'Space' },
  { id: 'attention', label: 'Attention' },
  { id: 'data', label: 'Data and exports' },
];

