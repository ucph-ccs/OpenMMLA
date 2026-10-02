/**
 * Shared helpers for the dashboard pages: DOM building, formatting, the API client, theme handling,
 * participant and voice identity, and the small shell components every page uses (top bar, tooltip,
 * status pill, segmented control). Untrusted text always goes in through textContent: h() and s() have
 * no way to set innerHTML.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';
const NBSP = ' ';
const PROPERTY_KEYS = new Set(['value', 'checked', 'selected', 'indeterminate', 'muted', 'defaultMuted', 'srcObject', 'volume']);

export const SESSION_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;

// dom

function applyProps(el, props, svg) {
  if (!props) return;
  for (const key of Object.keys(props)) {
    const v = props[key];
    if (v == null || v === false) continue;
    switch (key) {
      case 'class':
      case 'className': {
        const cls = Array.isArray(v) ? v.filter(Boolean).join(' ') : String(v);
        if (cls) el.setAttribute('class', cls);
        break;
      }
      case 'style':
        if (typeof v === 'string') el.setAttribute('style', v);
        else {
          for (const sk of Object.keys(v)) {
            const sv = v[sk];
            if (sv == null || sv === false) continue;
            if (sk.includes('-')) el.style.setProperty(sk, String(sv));
            else el.style[sk] = typeof sv === 'number' && !svg ? `${sv}px` : String(sv);
          }
        }
        break;
      case 'dataset':
        for (const dk of Object.keys(v)) if (v[dk] != null) el.dataset[dk] = String(v[dk]);
        break;
      case 'attrs':
        for (const ak of Object.keys(v)) {
          const av = v[ak];
          if (av == null || av === false) continue;
          el.setAttribute(ak, av === true ? '' : String(av));
        }
        break;
      case 'on':
        for (const ev of Object.keys(v)) if (typeof v[ev] === 'function') el.addEventListener(ev, v[ev]);
        break;
      case 'text':
        el.textContent = String(v);
        break;
      default:
        if (!svg && PROPERTY_KEYS.has(key)) el[key] = v;
        else el.setAttribute(key, v === true ? '' : String(v));
    }
  }
}

function appendChildren(el, children) {
  for (const c of children) {
    if (c == null || c === false || c === true) continue;
    if (Array.isArray(c)) appendChildren(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}

/** Create an HTML element. Props: class, style (object), dataset, attrs, on, text; other keys become attributes. */
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  applyProps(el, props, false);
  appendChildren(el, children);
  return el;
}

/** Create an SVG element (same props as h; numbers in style stay unitless). */
export function s(tag, props = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyProps(el, props, true);
  appendChildren(el, children);
  return el;
}

export function clear(el) {
  if (!el) return el;
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

// formatting

const NA = 'n/a';
const numberFormats = new Map();

function nf(minDigits, maxDigits) {
  const k = `${minDigits}:${maxDigits}`;
  let f = numberFormats.get(k);
  if (!f) {
    f = new Intl.NumberFormat('en-GB', { minimumFractionDigits: minDigits, maximumFractionDigits: maxDigits });
    numberFormats.set(k, f);
  }
  return f;
}

function finite(v) {
  if (v == null || v === '' || typeof v === 'boolean') return null;
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function fixed(n, digits) {
  const d = Math.max(0, Math.min(6, digits | 0));
  const p = 10 ** d;
  let r = Math.round(n * p) / p;
  if (Object.is(r, -0) || r === 0) r = 0;
  return nf(d, d).format(r);
}

function pad2(n) {
  return n < 10 ? `0${n}` : String(n);
}

const dateFmt = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
const timeFmt = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' });
const hmFmt = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
let zoneFmt = null;

function zoneName(date) {
  try {
    if (!zoneFmt) zoneFmt = new Intl.DateTimeFormat('en-GB', { timeZoneName: 'short' });
    const part = zoneFmt.formatToParts(date).find((p) => p.type === 'timeZoneName');
    return part ? part.value : '';
  } catch {
    return '';
  }
}

function clockHMS(sec) {
  const neg = sec < 0;
  let t = Math.floor(Math.abs(sec) + 1e-6);
  const hh = Math.floor(t / 3600);
  t -= hh * 3600;
  const mm = Math.floor(t / 60);
  const ss = t - mm * 60;
  return `${neg ? '-' : ''}${pad2(hh)}:${pad2(mm)}:${pad2(ss)}`;
}

function clockMS(sec) {
  const neg = sec < 0;
  const t = Math.floor(Math.abs(sec) + 1e-6);
  const mm = Math.floor(t / 60);
  return `${neg ? '-' : ''}${pad2(mm)}:${pad2(t - mm * 60)}`;
}

export const fmt = {
  na: NA,
  /** ratio (0..1) as a percentage: 0.86 -> "86 %". */
  pct(v, digits = 0) {
    const n = finite(v);
    return n == null ? NA : `${fixed(n * 100, digits)}${NBSP}%`;
  },
  /** ratio difference as signed percentage points: 0.032 -> "+3.2 pp". */
  pp(v, digits = 1) {
    const n = finite(v);
    if (n == null) return NA;
    const txt = fixed(n * 100, digits);
    return `${n > 0 && txt !== fixed(0, digits) ? '+' : ''}${txt}${NBSP}pp`;
  },
  num(v, digits = 1) {
    const n = finite(v);
    return n == null ? NA : fixed(n, digits);
  },
  int(v) {
    const n = finite(v);
    return n == null ? NA : fixed(n, 0);
  },
  /** 1,284 / 12.9K / 4.2M: for stat tiles and tight labels. */
  compact(v) {
    const n = finite(v);
    if (n == null) return NA;
    const a = Math.abs(n);
    if (a < 10000) return Number.isInteger(n) ? fixed(n, 0) : fixed(n, a < 10 ? 2 : 1);
    if (a < 1e6) return `${fixed(n / 1e3, a < 1e5 ? 1 : 0)}K`;
    if (a < 1e9) return `${fixed(n / 1e6, a < 1e7 ? 1 : 0)}M`;
    return `${fixed(n / 1e9, 1)}B`;
  },
  metres(v, digits = 2) {
    const n = finite(v);
    return n == null ? NA : `${fixed(n, digits)}${NBSP}m`;
  },
  secs(v, digits = 1) {
    const n = finite(v);
    return n == null ? NA : `${fixed(n, digits)}${NBSP}s`;
  },
  /** "1 h 01 min", "12 min 30 s", "12 min", "45 s". */
  duration(sec) {
    const n = finite(sec);
    if (n == null) return NA;
    let t = Math.round(Math.abs(n));
    const sign = n < 0 ? '-' : '';
    if (t < 60) return `${sign}${t}${NBSP}s`;
    if (t < 3600) {
      const m = Math.floor(t / 60);
      const r = t - m * 60;
      return `${sign}${m}${NBSP}min${r ? ` ${r}${NBSP}s` : ''}`;
    }
    t = Math.round(t / 60);
    const hh = Math.floor(t / 60);
    return `${sign}${hh}${NBSP}h ${pad2(t - hh * 60)}${NBSP}min`;
  },
  /** session offset in seconds -> "01:02:03". */
  clock(sec) {
    const n = finite(sec);
    return n == null ? NA : clockHMS(n);
  },
  /** a clock formatter suited to a span: "MM:SS" under one hour, else "HH:MM:SS". */
  clockSpan(span) {
    const n = finite(span);
    if (n != null && n < 3600) return (v) => (finite(v) == null ? NA : clockMS(finite(v)));
    return (v) => (finite(v) == null ? NA : clockHMS(finite(v)));
  },
  /** epoch -> local "08:26:13". */
  time(epoch) {
    const n = finite(epoch);
    return n == null ? NA : timeFmt.format(new Date(n * 1000));
  },
  /** epoch -> local "08:26". */
  hm(epoch) {
    const n = finite(epoch);
    return n == null ? NA : hmFmt.format(new Date(n * 1000));
  },
  /** epoch -> "3 Jun 2026" (local). */
  date(epoch) {
    const n = finite(epoch);
    return n == null ? NA : dateFmt.format(new Date(n * 1000));
  },
  /** epoch -> "3 Jun 2026, 08:26 CEST" (local, with the zone abbreviation). */
  dateTime(epoch) {
    const n = finite(epoch);
    if (n == null) return NA;
    const d = new Date(n * 1000);
    const z = zoneName(d);
    return `${dateFmt.format(d)}, ${hmFmt.format(d)}${z ? ` ${z}` : ''}`;
  },
  /** local zone abbreviation at an epoch ("CEST"). */
  zone(epoch) {
    const n = finite(epoch);
    return zoneName(n == null ? new Date() : new Date(n * 1000));
  },
  /** age in seconds -> "3 s ago", "2 min ago", "1 h 05 min ago". */
  ago(sec) {
    const n = finite(sec);
    if (n == null) return NA;
    const t = Math.max(0, Math.round(n));
    if (t < 60) return `${t}${NBSP}s ago`;
    if (t < 3600) return `${Math.floor(t / 60)}${NBSP}min ago`;
    if (t < 86400) {
      const hh = Math.floor(t / 3600);
      const mm = Math.floor((t - hh * 3600) / 60);
      return `${hh}${NBSP}h${mm ? ` ${pad2(mm)}${NBSP}min` : ''} ago`;
    }
    const d = Math.floor(t / 86400);
    return `${d}${NBSP}${d === 1 ? 'day' : 'days'} ago`;
  },
};

// api

/** fetch JSON. Resolves to {ok, status, data, error}; never throws. A 202 is ok (status tells). */
export async function api(path, { method = 'GET', body, signal, headers } = {}) {
  const init = { method, signal, cache: 'no-store', headers: { Accept: 'application/json', ...(headers || {}) } };
  if (body !== undefined && body !== null) {
    if (typeof body === 'string' || (typeof FormData !== 'undefined' && body instanceof FormData)) init.body = body;
    else {
      init.body = JSON.stringify(body);
      init.headers['Content-Type'] = 'application/json';
    }
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (err) {
    const aborted = !!(err && err.name === 'AbortError');
    return {
      ok: false,
      status: 0,
      data: null,
      aborted,
      error: aborted ? 'Request cancelled.' : `Could not reach the dashboard server (${(err && err.message) || 'network error'}).`,
    };
  }
  let data = null;
  try {
    const text = await res.text();
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
  } catch (err) {
    if (err && err.name === 'AbortError') return { ok: false, status: res.status, data: null, aborted: true, error: 'Request cancelled.' };
  }
  let error = null;
  if (!res.ok) {
    error = (data && typeof data.error === 'string' && data.error)
      || `${res.status}${res.statusText ? ` ${res.statusText}` : ''}`;
  }
  return { ok: res.ok, status: res.status, data, error };
}

function sleep(ms, signal) {
  return new Promise((resolve) => {
    if (signal && signal.aborted) return resolve();
    const id = setTimeout(done, ms);
    function done() {
      clearTimeout(id);
      if (signal) signal.removeEventListener('abort', done);
      resolve();
    }
    if (signal) signal.addEventListener('abort', done, { once: true });
  });
}

function whenVisible(signal) {
  if (typeof document === 'undefined' || document.visibilityState !== 'hidden') return Promise.resolve();
  return new Promise((resolve) => {
    function done() {
      if (document.visibilityState === 'hidden' && !(signal && signal.aborted)) return;
      document.removeEventListener('visibilitychange', done);
      if (signal) signal.removeEventListener('abort', done);
      resolve();
    }
    document.addEventListener('visibilitychange', done);
    if (signal) signal.addEventListener('abort', done, { once: true });
  });
}

/**
 * Call fn repeatedly, `interval` ms apart, until until(result) is true or the signal aborts; resolves
 * to the last result. Without `until` it keeps going until aborted. `visible: true` waits while the
 * tab is hidden.
 */
export async function poll(fn, { interval = 2000, until, signal, visible = false } = {}) {
  let last;
  for (;;) {
    if (signal && signal.aborted) return last;
    if (visible) await whenVisible(signal);
    if (signal && signal.aborted) return last;
    last = await fn();
    if (signal && signal.aborted) return last;
    if (typeof until === 'function' && until(last)) return last;
    await sleep(typeof interval === 'function' ? interval(last) : interval, signal);
  }
}

export function debounce(fn, ms = 150) {
  let id = 0;
  const wrapped = (...args) => {
    clearTimeout(id);
    id = setTimeout(() => fn(...args), ms);
  };
  wrapped.cancel = () => clearTimeout(id);
  return wrapped;
}

/** run fn at most once per animation frame with the latest arguments. */
export function rafThrottle(fn) {
  let id = 0;
  let args = null;
  const wrapped = (...a) => {
    args = a;
    if (!id) {
      id = requestAnimationFrame(() => {
        id = 0;
        fn(...args);
      });
    }
  };
  wrapped.cancel = () => {
    if (id) cancelAnimationFrame(id);
    id = 0;
  };
  return wrapped;
}

// theme

const THEME_KEY = 'openmmla.dashboard.theme';
const THEME_MODES = ['system', 'light', 'dark'];
let themeMode = 'system';
let themeReady = false;
let darkQuery = null;

function readStoredTheme() {
  try {
    const v = window.localStorage.getItem(THEME_KEY);
    return THEME_MODES.includes(v) ? v : 'system';
  } catch {
    return 'system';
  }
}

function resolvedTheme() {
  if (themeMode === 'light' || themeMode === 'dark') return themeMode;
  return darkQuery && darkQuery.matches ? 'dark' : 'light';
}

function applyTheme() {
  const root = document.documentElement;
  if (themeMode === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', themeMode);
  root.dataset.themeResolved = resolvedTheme();
}

function emitTheme() {
  window.dispatchEvent(new CustomEvent('themechange', { detail: { mode: themeMode, resolved: resolvedTheme() } }));
}

export const theme = {
  /** read the stored choice, apply it, follow OS changes. Safe to call more than once. */
  init() {
    if (themeReady) return themeMode;
    themeReady = true;
    darkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
    themeMode = readStoredTheme();
    applyTheme();
    if (darkQuery) {
      const onOs = () => {
        if (themeMode !== 'system') return;
        applyTheme();
        emitTheme();
      };
      if (darkQuery.addEventListener) darkQuery.addEventListener('change', onOs);
      else if (darkQuery.addListener) darkQuery.addListener(onOs);
    }
    window.addEventListener('storage', (e) => {
      if (e.key !== THEME_KEY) return;
      const next = THEME_MODES.includes(e.newValue) ? e.newValue : 'system';
      if (next === themeMode) return;
      themeMode = next;
      applyTheme();
      emitTheme();
    });
    return themeMode;
  },
  /** 'system' | 'light' | 'dark' */
  mode() {
    return themeMode;
  },
  /** 'light' | 'dark' after resolving 'system'. */
  resolved() {
    return resolvedTheme();
  },
  set(mode) {
    if (!themeReady) theme.init();
    const next = THEME_MODES.includes(mode) ? mode : 'system';
    if (next === themeMode) return;
    themeMode = next;
    try {
      if (next === 'system') window.localStorage.removeItem(THEME_KEY);
      else window.localStorage.setItem(THEME_KEY, next);
    } catch {
      // storage can be blocked; the choice still applies to this page
    }
    applyTheme();
    emitTheme();
  },
  /** fn({mode, resolved}) on every change; returns an unsubscribe function. */
  onChange(fn) {
    const handler = (e) => fn(e.detail || { mode: themeMode, resolved: resolvedTheme() });
    window.addEventListener('themechange', handler);
    return () => window.removeEventListener('themechange', handler);
  },
};

/** computed value of a custom property; accepts '--tag-1', 'tag-1' or 'var(--tag-1)'. */
export function cssVar(name, el = document.documentElement) {
  let n = String(name || '').trim();
  let fallback = '';
  const m = /^var\(\s*(--[A-Za-z0-9_-]+)\s*(?:,\s*(.*))?\)$/.exec(n);
  if (m) {
    n = m[1];
    fallback = (m[2] || '').trim();
  }
  if (!n.startsWith('--')) n = `--${n}`;
  const v = getComputedStyle(el).getPropertyValue(n).trim();
  return v || fallback;
}

// icons (24 px grid, stroked like the rest of the chrome)

const ICONS = {
  copy: [['rect', { x: 9, y: 9, width: 12, height: 12, rx: 2 }], ['path', { d: 'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1' }]],
  check: [['path', { d: 'M20 6 9 17l-5-5' }]],
  sun: [['circle', { cx: 12, cy: 12, r: 4 }], ['path', { d: 'M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41' }]],
  moon: [['path', { d: 'M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z' }]],
  monitor: [['rect', { x: 2, y: 3, width: 20, height: 14, rx: 2 }], ['path', { d: 'M8 21h8M12 17v4' }]],
  play: [['path', { d: 'M7 4.5v15a.8.8 0 0 0 1.2.7l12.3-7.5a.8.8 0 0 0 0-1.4L8.2 3.8A.8.8 0 0 0 7 4.5Z', fill: 'currentColor', stroke: 'none' }]],
  pause: [['rect', { x: 6, y: 4, width: 4, height: 16, rx: 1, fill: 'currentColor', stroke: 'none' }], ['rect', { x: 14, y: 4, width: 4, height: 16, rx: 1, fill: 'currentColor', stroke: 'none' }]],
  stop: [['rect', { x: 6, y: 6, width: 12, height: 12, rx: 2, fill: 'currentColor', stroke: 'none' }]],
  alert: [['path', { d: 'M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z' }], ['path', { d: 'M12 9v4M12 17h.01' }]],
  info: [['circle', { cx: 12, cy: 12, r: 10 }], ['path', { d: 'M12 16v-4M12 8h.01' }]],
  download: [['path', { d: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3' }]],
  refresh: [['path', { d: 'M21 12a9 9 0 1 1-2.64-6.36L21 8' }], ['path', { d: 'M21 3v5h-5' }]],
  reconnect: [['path', { d: 'M3 12a9 9 0 1 0 2.64-6.36L3 8' }], ['path', { d: 'M3 3v5h5' }]],
  table: [['rect', { x: 3, y: 4, width: 18, height: 16, rx: 2 }], ['path', { d: 'M3 10h18M3 15h18M9 4v16' }]],
  close: [['path', { d: 'M18 6 6 18M6 6l12 12' }]],
  'chevron-down': [['path', { d: 'm6 9 6 6 6-6' }]],
  'chevron-right': [['path', { d: 'm9 18 6-6-6-6' }]],
  external: [['path', { d: 'M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6' }]],
  camera: [['path', { d: 'M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2Z' }], ['circle', { cx: 12, cy: 13, r: 4 }]],
  clock: [['circle', { cx: 12, cy: 12, r: 10 }], ['path', { d: 'M12 6v6l4 2' }]],
  search: [['circle', { cx: 11, cy: 11, r: 7 }], ['path', { d: 'm21 21-4.3-4.3' }]],
  dot: [['circle', { cx: 12, cy: 12, r: 5, fill: 'currentColor', stroke: 'none' }]],
  volume: [['path', { d: 'M11 5 6 9H2v6h4l5 4V5Z' }], ['path', { d: 'M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14' }]],
  'volume-off': [['path', { d: 'M11 5 6 9H2v6h4l5 4V5Z' }], ['path', { d: 'm22 9-6 6M16 9l6 6' }]],
};

/** small inline icon (aria-hidden); names: copy check sun moon monitor play pause stop alert info download refresh reconnect table close chevron-down chevron-right external camera clock search dot volume volume-off */
export function icon(name, size = 14) {
  const parts = ICONS[name] || ICONS.dot;
  return s('svg', {
    class: `icon icon-${name}`,
    attrs: {
      viewBox: '0 0 24 24', width: size, height: size, fill: 'none', stroke: 'currentColor',
      'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true', focusable: 'false',
    },
  }, parts.map(([tag, attrs]) => s(tag, { attrs })));
}

// announcements for screen readers

let liveRegion = null;

/** say something politely to assistive technology (copy confirmations, keyboard readouts). */
export function announce(text) {
  if (!liveRegion) {
    liveRegion = h('div', { class: 'sr-only', attrs: { 'aria-live': 'polite', 'aria-atomic': 'true' } });
    document.body.appendChild(liveRegion);
  }
  liveRegion.textContent = '';
  // a fresh text node after a tick makes repeated messages announce again
  setTimeout(() => {
    liveRegion.textContent = String(text || '');
  }, 30);
}

// tags and identity

/** numeric tags first in numeric order, then the rest lexicographically (same rule as the backend). */
export function tagSortKey(tag) {
  const t = String(tag);
  return /^\d+$/.test(t) ? [0, Number(t), ''] : [1, 0, t];
}

export function sortTags(tags) {
  return Array.from(new Set(Array.from(tags || [], (t) => String(t)))).sort((a, b) => {
    const ka = tagSortKey(a);
    const kb = tagSortKey(b);
    return ka[0] - kb[0] || ka[1] - kb[1] || (ka[2] < kb[2] ? -1 : ka[2] > kb[2] ? 1 : 0);
  });
}

/** a pupil badge: digits only, at most 12 (untagged bodies and higher tags are never participants). */
export function isPupilTag(tag) {
  const t = String(tag);
  return /^\d+$/.test(t) && Number(t) <= 12;
}

export function voiceLabel(key) {
  const k = String(key);
  if (k === 'other') return 'Other voices';
  if (k === 'v:?') return 'Unlinked';
  let m = /^v:(\d+)@(\d+)$/.exec(k);
  if (m) return `Voice ${m[1]} (mic ${m[2]})`;
  m = /^v:(\d+)$/.exec(k);
  if (m) return `Voice ${m[1]}`;
  if (k.startsWith('spk:')) return k.slice(4);
  if (k.startsWith('tag:')) return `Tag ${k.slice(4)}`;
  if (k === 'group' || k.startsWith('group:')) return 'Group mic';
  return k;
}

/**
 * Colour and label assignment for one page.
 * roster: tag ids; slots follow sorted order (tag-1 .. tag-5, then tag-other).
 * options.grow: tags first seen later get the next free slot (live pages: never repaint a tag).
 * voice(entity|key): entity.slot from the speech part wins; bare keys get slots by first-seen order.
 * entity(key, speech): any speech key (v:, spk:, tag:, group:, other) resolved against the speech part.
 */
export function identity(roster = [], { grow = false } = {}) {
  const tagSlots = new Map();
  for (const id of sortTags(roster || [])) tagSlots.set(id, tagSlots.size + 1);
  const voiceSlots = new Map();

  function tagSlotOf(id) {
    let slot = tagSlots.get(id);
    if (slot == null && grow && isPupilTag(id)) {
      slot = tagSlots.size + 1;
      tagSlots.set(id, slot);
    }
    return slot != null && slot <= 5 ? slot : null;
  }

  function tag(id) {
    const tid = String(id);
    const slot = tagSlotOf(tid);
    return { key: `tag:${tid}`, id: tid, slot, color: slot ? `var(--tag-${slot})` : 'var(--tag-other)', label: `Tag ${tid}` };
  }

  function addTags(ids) {
    for (const id of sortTags(ids || [])) {
      if (!tagSlots.has(id) && isPupilTag(id)) tagSlots.set(id, tagSlots.size + 1);
    }
    return Array.from(tagSlots.keys());
  }

  function voiceOther(key, label) {
    return { key, slot: null, color: 'var(--voice-other)', label: label || voiceLabel(key), other: true };
  }

  function voice(entity) {
    const ent = typeof entity === 'string' ? { key: entity } : entity || {};
    const key = String(ent.key ?? '');
    if (key === 'other' || key === 'v:?') return voiceOther(key, ent.label);
    let slot = Number.isInteger(ent.slot) ? ent.slot : null;
    if (slot == null) {
      if (voiceSlots.has(key)) slot = voiceSlots.get(key);
      else {
        const used = new Set(voiceSlots.values());
        for (let i = 1; i <= 5; i += 1) {
          if (!used.has(i)) {
            slot = i;
            break;
          }
        }
        voiceSlots.set(key, slot);
      }
    } else if (slot != null) voiceSlots.set(key, slot);
    if (!slot || slot > 5) return voiceOther(key, ent.label);
    return { key, slot, color: `var(--voice-${slot})`, label: ent.label || voiceLabel(key) };
  }

  function entity(key, speech) {
    const k = String(key);
    const ents = (speech && speech.entities) || [];
    const found = ents.find((e) => e.key === k);
    if (k.startsWith('tag:')) {
      const t = tag(k.slice(4));
      return { ...t, key: k, label: (found && found.label) || t.label };
    }
    if (k === 'group' || k.startsWith('group:')) return { key: k, slot: null, color: 'var(--ink-2)', label: 'Group mic' };
    if (k === 'other') return voiceOther(k, (speech && speech.other && speech.other.label) || 'Other voices');
    if (k.startsWith('spk:') && found && found.slot == null) {
      const named = ents.filter((e) => String(e.key).startsWith('spk:'));
      const idx = named.indexOf(found) + 1;
      return idx >= 1 && idx <= 5
        ? { key: k, slot: idx, color: `var(--voice-${idx})`, label: found.label || voiceLabel(k) }
        : voiceOther(k, found.label);
    }
    if (speech) {
      if (found) return voice(found);
      return voiceOther(k);
    }
    return voice(k);
  }

  return {
    tag,
    addTags,
    tags: () => Array.from(tagSlots.keys()),
    voice,
    entity,
  };
}

// tooltip (one per page)

let tipEl = null;
let tipHideBound = false;

function tipNode() {
  if (!tipEl) {
    tipEl = h('div', { class: 'tooltip', attrs: { role: 'tooltip', 'aria-hidden': 'true' }, hidden: true });
    document.body.appendChild(tipEl);
  }
  if (!tipHideBound) {
    tipHideBound = true;
    window.addEventListener('scroll', () => tooltip.hide(), { passive: true, capture: true });
    window.addEventListener('blur', () => tooltip.hide());
  }
  return tipEl;
}

function keyNode(row) {
  const kind = row.key || 'line';
  if (row.color === 'hatch' || kind === 'hatch') return h('span', { class: 'tooltip-key hatch' });
  if (!row.color || kind === 'none') return h('span', { class: 'tooltip-key none' });
  return h('span', { class: `tooltip-key ${kind}`, style: { background: row.color } });
}

function anchorPoint(at) {
  if (!at) return { x: 0, y: 0, rect: null };
  if (typeof Element !== 'undefined' && at instanceof Element) {
    const r = at.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top, rect: r };
  }
  if (at.clientX != null) return { x: at.clientX, y: at.clientY, rect: null };
  if (at.target && at.target instanceof Element && at.type && at.type.startsWith('focus')) return anchorPoint(at.target);
  return { x: at.x || 0, y: at.y || 0, rect: at.rect || null };
}

export const tooltip = {
  /**
   * at: a pointer event, {x, y} in client pixels, or an element (keyboard focus).
   * content: {title, rows: [{value, label, color, key: 'line'|'rect'|'dot'|'hatch'} | {section}], note}
   */
  show(at, { title, rows = [], note } = {}) {
    const el = tipNode();
    clear(el);
    if (title != null && title !== '') el.appendChild(h('div', { class: 'tooltip-title', text: title }));
    if (rows.length) {
      const grid = h('div', { class: 'tooltip-rows' });
      for (const row of rows) {
        if (!row) continue;
        if (row.section != null) {
          grid.appendChild(h('div', { class: 'tooltip-section', text: row.section }));
          continue;
        }
        grid.appendChild(keyNode(row));
        grid.appendChild(h('span', { class: 'tooltip-value', text: row.value == null ? NA : row.value }));
        grid.appendChild(h('span', { class: 'tooltip-label', text: row.label == null ? '' : row.label }));
      }
      el.appendChild(grid);
    }
    if (note) el.appendChild(h('div', { class: 'tooltip-note', text: note }));
    el.hidden = false;
    const { x, y, rect } = anchorPoint(at);
    const w = el.offsetWidth;
    const ht = el.offsetHeight;
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;
    let left;
    let top;
    if (rect) {
      left = rect.left + rect.width / 2 - w / 2;
      top = rect.top - ht - 8;
      if (top < 8) top = rect.bottom + 8;
    } else {
      left = x + 14;
      top = y + 14;
      if (left + w > vw - 8) left = x - w - 14;
      if (top + ht > vh - 8) top = y - ht - 14;
    }
    left = Math.max(8, Math.min(left, vw - w - 8));
    top = Math.max(8, Math.min(top, vh - ht - 8));
    el.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  },
  hide() {
    if (tipEl) tipEl.hidden = true;
  },
};

// small components

/** empty or error panel. action: a Node or {label, onClick, href}. kind: 'empty' | 'error' | 'info'. */
export function emptyState({ title, body, action, kind = 'empty' } = {}) {
  const el = h('div', { class: 'empty', dataset: { kind }, attrs: { role: kind === 'error' ? 'alert' : 'status' } });
  el.appendChild(h('p', { class: 'empty-title' }, kind === 'error' ? icon('alert', 16) : null, h('span', { text: title || (kind === 'error' ? 'Something went wrong.' : 'Nothing to show.') })));
  if (body) el.appendChild(h('p', { class: 'empty-body', text: body }));
  if (action) {
    let node = action;
    if (!(action instanceof Node)) {
      node = action.href
        ? h('a', { class: 'btn sm', href: action.href, text: action.label || 'Open' })
        : h('button', { class: 'btn sm', attrs: { type: 'button' }, on: { click: action.onClick }, text: action.label || 'Retry' });
    }
    el.appendChild(h('div', { class: 'empty-action' }, node));
  }
  return el;
}

const PILL_DEFAULTS = {
  live: 'Live', ended: 'Ended', replay: 'Replay', paused: 'Paused', error: 'Error',
  reconnecting: 'Reconnecting', warning: 'Warning', ready: 'Ready', running: 'Computing',
};

function pillMark(state) {
  switch (state) {
    case 'live': return h('span', { class: 'pill-mark' }, h('span', { class: 'pill-dot' }));
    case 'ended': return h('span', { class: 'pill-mark' }, icon('stop', 12));
    case 'replay': return h('span', { class: 'pill-mark' }, icon('play', 12));
    case 'paused': return h('span', { class: 'pill-mark' }, icon('pause', 12));
    case 'reconnecting': return h('span', { class: 'pill-mark' }, icon('reconnect', 12));
    case 'ready': return h('span', { class: 'pill-mark' }, icon('check', 12));
    case 'running': return h('span', { class: 'pill-mark' }, h('span', { class: 'spinner', style: { width: '10px', height: '10px' } }));
    case 'warning':
    case 'error': return h('span', { class: 'pill-mark' }, icon('alert', 12));
    default: return h('span', { class: 'pill-mark' }, h('span', { class: 'pill-dot' }));
  }
}

/** status pill with an icon and a label; the returned element has update({state, text}). */
export function statusPill({ state = 'ended', text } = {}) {
  const el = h('span', { class: 'pill', attrs: { role: 'status' } });
  el.update = ({ state: st = el.dataset.state, text: tx } = {}) => {
    el.dataset.state = st;
    clear(el);
    el.append(pillMark(st), h('span', { text: tx || PILL_DEFAULTS[st] || st }));
  };
  el.update({ state, text });
  return el;
}

async function writeClipboard(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the textarea copy
  }
  try {
    const ta = h('textarea', { attrs: { readonly: true, 'aria-hidden': 'true' }, style: { position: 'fixed', left: '-9999px', top: '0' } });
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

/** icon button that copies text; label names what is copied for screen readers. */
export function copyButton(text, label = 'Copy session id') {
  const btn = h('button', {
    class: 'btn ghost sm icon-btn copy-btn',
    attrs: { type: 'button', 'aria-label': label, title: label },
  }, icon('copy', 14));
  let timer = 0;
  btn.addEventListener('click', async () => {
    const ok = await writeClipboard(String(text));
    clearTimeout(timer);
    clear(btn).appendChild(icon(ok ? 'check' : 'alert', 14));
    if (ok) btn.dataset.copied = '';
    announce(ok ? 'Copied' : 'Copy failed');
    timer = setTimeout(() => {
      delete btn.dataset.copied;
      clear(btn).appendChild(icon('copy', 14));
    }, 1400);
  });
  return btn;
}

/**
 * Accessible single-choice control (radiogroup; arrow keys move and select).
 * options: [{value, label, title, icon, disabled}]. Returns the element with .value and
 * setValue(v) (no onChange), setOptions(options, value).
 */
export function segmented({ options = [], value, onChange, label, size, iconsOnly = false } = {}) {
  const el = h('div', {
    class: ['seg', size === 'sm' ? 'sm' : null, iconsOnly ? 'icons' : null],
    attrs: { role: 'radiogroup', 'aria-label': label || null },
  });
  let current = value;
  let opts = [];

  function sync() {
    const buttons = Array.from(el.children);
    const idx = opts.findIndex((o) => o.value === current);
    buttons.forEach((b, i) => {
      const on = i === idx;
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.tabIndex = on || (idx < 0 && i === 0) ? 0 : -1;
    });
  }

  function select(v, fire) {
    if (v === current) return;
    current = v;
    sync();
    if (fire && typeof onChange === 'function') onChange(v);
  }

  function build() {
    clear(el);
    opts.forEach((o) => {
      const b = h('button', {
        attrs: { type: 'button', role: 'radio', 'aria-checked': 'false', title: o.title || (iconsOnly ? o.label : null), 'aria-label': iconsOnly ? o.label : null, disabled: !!o.disabled },
        on: { click: () => select(o.value, true) },
      }, o.icon ? (o.icon instanceof Node ? o.icon : icon(o.icon, 14)) : null, iconsOnly ? null : h('span', { text: o.label }));
      el.appendChild(b);
    });
    sync();
  }

  el.addEventListener('keydown', (e) => {
    const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'];
    if (!keys.includes(e.key)) return;
    e.preventDefault();
    const enabled = opts.map((o, i) => (o.disabled ? -1 : i)).filter((i) => i >= 0);
    if (!enabled.length) return;
    let pos = enabled.indexOf(opts.findIndex((o) => o.value === current));
    if (e.key === 'Home') pos = 0;
    else if (e.key === 'End') pos = enabled.length - 1;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') pos = pos <= 0 ? enabled.length - 1 : pos - 1;
    else pos = pos >= enabled.length - 1 ? 0 : pos + 1;
    const i = enabled[pos];
    select(opts[i].value, true);
    el.children[i].focus();
  });

  Object.defineProperty(el, 'value', { get: () => current });
  el.setValue = (v) => select(v, false);
  el.setOptions = (next, v = current) => {
    opts = Array.from(next || []);
    current = v;
    build();
  };
  el.setOptions(options, value);
  return el;
}

/** ?session= from the URL when it is a valid session id, else null. */
export function sessionIdFromUrl() {
  try {
    const sid = new URLSearchParams(window.location.search).get('session');
    return sid && SESSION_ID_RE.test(sid) ? sid : null;
  } catch {
    return null;
  }
}

/** true when the URL carries a ?session= that is not a valid session id (a broken link, not a missing one). */
export function sessionParamInvalid() {
  try {
    const sid = new URLSearchParams(window.location.search).get('session');
    return sid != null && sid !== '' && !SESSION_ID_RE.test(sid);
  } catch {
    return false;
  }
}

export function downloadLink(href, label) {
  return h('a', { class: 'download-link', href, attrs: { download: '' } }, icon('download', 14), h('span', { text: label }));
}

function dateFromIso(d) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(d || ''));
  if (!m) return null;
  return dateFmt.format(new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
}

/** {title: "microbit · group 01", date: "3 Jun 2026", id} from a session meta or index entry. */
export function sessionTitle(meta) {
  const m = meta || {};
  const group = m.group ? String(m.group).replace(/_/g, ' ') : null;
  const title = [m.task, group].filter(Boolean).join(' · ') || m.experiment || m.id || 'Session';
  const date = dateFromIso(m.date) || (m.recorded_start != null ? fmt.date(m.recorded_start) : m.t0 != null ? fmt.date(m.t0) : null);
  return { title, date, id: m.id || null };
}

/** page heading for a session: title, date in ink-2, id in mono with one copy button. */
export function sessionHeading(meta, { level = 'h1' } = {}) {
  const t = sessionTitle(meta);
  const head = h('div', { class: 'session-head' });
  head.appendChild(h(level, { class: 'session-title' }, h('span', { text: t.title }), t.date ? h('span', { class: 'session-date', text: t.date }) : null));
  if (t.id) head.appendChild(h('div', { class: 'session-id' }, h('span', { text: t.id, title: t.id }), copyButton(t.id)));
  return head;
}

function themeSwitch() {
  const seg = segmented({
    label: 'Colour theme',
    size: 'sm',
    iconsOnly: true,
    value: theme.mode(),
    options: [
      { value: 'system', label: 'System theme', icon: 'monitor' },
      { value: 'light', label: 'Light theme', icon: 'sun' },
      { value: 'dark', label: 'Dark theme', icon: 'moon' },
    ],
    onChange: (v) => theme.set(v),
  });
  theme.onChange(({ mode }) => seg.setValue(mode));
  return seg;
}

function viewSwitch(sid, view) {
  const q = `?session=${encodeURIComponent(sid)}`;
  const link = (v, label, href) => h('a', { href: `${href}${q}`, attrs: { 'aria-current': view === v ? 'page' : null }, text: label });
  return h('nav', { class: 'seg sm', attrs: { 'aria-label': 'View' } }, link('live', 'Live', '/live'), link('analysis', 'Analysis', '/analysis'));
}

/**
 * The page header bar. crumbs: [{label, href}] (default: Sessions / session title);
 * session: meta or {id}; view: 'live' | 'analysis' | null. The element has update({crumbs, session, view}).
 */
export function topbar({ crumbs, session = null, view = null } = {}) {
  theme.init();
  const header = h('header', { class: 'topbar' });
  const inner = h('div', { class: 'topbar-inner' });
  header.appendChild(inner);
  const right = h('div', { class: 'topbar-right' });
  const themeSeg = themeSwitch();
  let state = { crumbs, session, view };

  function render() {
    const { session: ses, view: vw } = state;
    const sid = ses ? (typeof ses === 'string' ? ses : ses.id) : null;
    let cr = state.crumbs;
    if (!cr) {
      cr = ses && typeof ses === 'object'
        ? [{ label: 'Sessions', href: '/' }, { label: sessionTitle(ses).title }]
        : sid ? [{ label: 'Sessions', href: '/' }, { label: sid }] : [{ label: 'Sessions' }];
    }
    clear(inner);
    inner.appendChild(h('a', { class: 'wordmark', href: '/', text: 'OpenMMLA' }));
    const ol = h('ol');
    cr.forEach((c, i) => {
      const last = i === cr.length - 1;
      ol.appendChild(h('li', {}, c.href && !last
        ? h('a', { href: c.href, text: c.label })
        : h('span', { attrs: { 'aria-current': last ? 'page' : null }, text: c.label })));
    });
    inner.appendChild(h('nav', { class: 'crumbs', attrs: { 'aria-label': 'Breadcrumb' } }, ol));
    clear(right);
    if (sid && vw) right.appendChild(viewSwitch(sid, vw));
    right.appendChild(themeSeg);
    inner.appendChild(right);
  }

  header.update = (next = {}) => {
    state = { ...state, ...next };
    render();
  };
  render();
  return header;
}
