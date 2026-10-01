/**
 * Dependency-free charts for the dashboard. Every chart renders into a container element, takes its
 * width from that container (ResizeObserver), keeps a fixed height, re-renders on `themechange`, reads
 * colours from CSS custom properties ('var(--tag-1)' strings go straight into fills) and returns a
 * handle {update(opts), destroy()}. Marks follow one spec: bars at most 24 px thick with a 4 px rounded
 * data end, 2 px lines, 2 px surface gaps and rings, solid hairline grids. Values reach the reader
 * three ways: direct labels where they fit, a tooltip on hover and keyboard focus, and the card's
 * table view. The timeline draws its lanes on a canvas so an hour of 1 s cells stays fast.
 */

import { h, s, clear, fmt, tooltip, theme, emptyState, icon, announce } from './core.js';

const FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const HATCH_ID = 'mmla-hatch';
const BAR_MAX = 24;
const GAP = 2;
const RADIUS = 4;

// text measurement

let measureCtx = null;

function textWidth(text, size = 12, weight = 400) {
  if (!measureCtx) measureCtx = document.createElement('canvas').getContext('2d');
  measureCtx.font = `${weight} ${size}px ${FONT_STACK}`;
  return measureCtx.measureText(String(text)).width;
}

function fitText(text, maxW, size = 12, weight = 400) {
  const str = String(text ?? '');
  if (maxW <= 8) return '';
  if (textWidth(str, size, weight) <= maxW) return str;
  let lo = 0;
  let hi = str.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (textWidth(`${str.slice(0, mid)}…`, size, weight) <= maxW) lo = mid;
    else hi = mid - 1;
  }
  return lo > 0 ? `${str.slice(0, lo)}…` : '';
}

// colour helpers

/** resolves 'var(--x)' strings to computed colours (canvas drawing and contrast checks). */
export function colorResolver(el = document.documentElement) {
  const cs = getComputedStyle(el);
  const cache = new Map();
  return (c) => {
    if (c == null) return null;
    let out = cache.get(c);
    if (out !== undefined) return out;
    out = String(c);
    const m = /^var\(\s*(--[A-Za-z0-9_-]+)\s*(?:,\s*(.+))?\)$/.exec(out.trim());
    if (m) out = cs.getPropertyValue(m[1]).trim() || (m[2] || '').trim() || '#888888';
    cache.set(c, out);
    return out;
  };
}

function parseColor(str) {
  const v = String(str || '').trim();
  let m = /^#([0-9a-f]{3,8})$/i.exec(v);
  if (m) {
    let hex = m[1];
    if (hex.length === 3 || hex.length === 4) hex = hex.split('').map((c) => c + c).join('');
    return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
  }
  m = /^rgba?\(([^)]+)\)$/i.exec(v);
  if (m) {
    const p = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
    if (p.length >= 3) return p.slice(0, 3);
  }
  return null;
}

function luminance(rgb) {
  const lin = rgb.map((c) => {
    const x = c / 255;
    return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
}

/** ink or white, whichever reads better on a resolved fill colour. */
function textOn(fill) {
  const rgb = parseColor(fill);
  if (!rgb) return '#0b0b0b';
  const L = luminance(rgb);
  const cWhite = 1.05 / (L + 0.05);
  const cInk = (L + 0.05) / 0.0548;
  return cWhite >= cInk ? '#ffffff' : '#0b0b0b';
}

/** id of the shared 45 degree hatch <pattern> (lines in --axis on --surface), injected once. */
export function hatchPatternId() {
  if (!document.getElementById(HATCH_ID)) {
    const svg = s('svg', {
      attrs: { width: 0, height: 0, 'aria-hidden': 'true', focusable: 'false' },
      style: { position: 'absolute', width: '0', height: '0', overflow: 'hidden' },
    }, s('defs', {}, s('pattern', {
      attrs: { id: HATCH_ID, patternUnits: 'userSpaceOnUse', width: 5, height: 5, patternTransform: 'rotate(45)' },
    }, s('rect', { attrs: { width: 5, height: 5 }, style: { fill: 'var(--surface)' } }),
    s('line', { attrs: { x1: 1, y1: 0, x2: 1, y2: 5 }, style: { stroke: 'var(--axis)', strokeWidth: 1.6 } }))));
    document.body.appendChild(svg);
  }
  return HATCH_ID;
}

function paint(color) {
  if (color === 'hatch') return `url(#${hatchPatternId()})`;
  return color || 'none';
}

const SEQ_STEPS = 8;
/** sequential ramp tokens, low to high. */
export const SEQ = Array.from({ length: SEQ_STEPS }, (_, i) => `var(--seq-${i + 1})`);

/** sequential fill for v in [0, max]: null and 0 take --surface-2. scale 'linear' | 'sqrt'. */
export function seqColor(v, max, scale = 'linear') {
  if (v == null || !Number.isFinite(v) || v <= 0 || !(max > 0)) return 'var(--surface-2)';
  let r = Math.min(1, v / max);
  if (scale === 'sqrt') r = Math.sqrt(r);
  return SEQ[Math.min(SEQ_STEPS - 1, Math.max(0, Math.ceil(r * SEQ_STEPS) - 1))];
}

/** small scale legend: label, low value, the eight ramp steps, high value. */
export function scaleLegend({ min = 0, max, format = (v) => fmt.num(v, 1), label } = {}) {
  return h('div', { class: 'chart-scale', attrs: { 'aria-hidden': 'true' } },
    label ? h('span', { text: label }) : null,
    h('span', { text: format(min) }),
    h('span', { class: 'chart-scale-ramp' }, SEQ.map((c) => h('span', { style: { background: c } }))),
    h('span', { text: format(max) }));
}

// geometry and scales

function barPath(x, y, w, ht, r = RADIUS, end = 'right') {
  if (!(w > 0) || !(ht > 0)) return '';
  if (end === 'right') {
    const rr = Math.min(r, w, ht / 2);
    return `M${x},${y}H${x + w - rr}A${rr},${rr} 0 0 1 ${x + w},${y + rr}V${y + ht - rr}A${rr},${rr} 0 0 1 ${x + w - rr},${y + ht}H${x}Z`;
  }
  if (end === 'top') {
    const rr = Math.min(r, ht, w / 2);
    return `M${x},${y + ht}V${y + rr}A${rr},${rr} 0 0 1 ${x + rr},${y}H${x + w - rr}A${rr},${rr} 0 0 1 ${x + w},${y + rr}V${y + ht}Z`;
  }
  return `M${x},${y}h${w}v${ht}h${-w}Z`;
}

function niceStep(range, count) {
  const raw = Math.abs(range) / Math.max(1, count);
  if (!(raw > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(raw));
  const f = raw / p;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return nice * p;
}

/** [lo, hi, ticks] covering [min, max] on round numbers. */
function niceScale(min, max, count = 4) {
  let lo = Number.isFinite(min) ? min : 0;
  let hi = Number.isFinite(max) ? max : 1;
  if (hi === lo) {
    hi = lo === 0 ? 1 : lo + Math.abs(lo) * 0.5;
    lo = lo === 0 ? 0 : lo - Math.abs(lo) * 0.5;
  }
  const step = niceStep(hi - lo, count);
  lo = Math.floor(lo / step + 1e-9) * step;
  hi = Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) ticks.push(Math.round(v / step) * step);
  return [lo, hi, ticks];
}

const TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 21600, 43200];

function timeTicks(a, b, px) {
  const want = Math.max(2, Math.floor(px / 92));
  const span = b - a;
  const step = TIME_STEPS.find((st) => span / st <= want) || TIME_STEPS[TIME_STEPS.length - 1];
  const out = [];
  for (let t = Math.ceil(a / step - 1e-9) * step; t <= b + 1e-9; t += step) out.push(t);
  return out;
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

function medianStep(t) {
  if (!t || t.length < 2) return 1;
  const n = Math.min(t.length - 1, 64);
  const d = [];
  for (let i = 1; i <= n; i += 1) d.push(t[i] - t[i - 1]);
  d.sort((a, b) => a - b);
  return d[d.length >> 1] || 1;
}

/** index of the step cell [t_i, t_i + step) that holds x, or -1. */
function cellAt(t, step, x) {
  if (!t || !t.length) return -1;
  const i = lowerBound(t, x + 1e-9) - 1;
  if (i < 0) return -1;
  return x < t[i] + step ? i : -1;
}

/** index of the sample nearest x within tol, or -1. */
function nearestAt(t, x, tol) {
  if (!t || !t.length) return -1;
  const i = lowerBound(t, x);
  let best = -1;
  let bd = Infinity;
  for (const j of [i - 1, i]) {
    if (j < 0 || j >= t.length) continue;
    const d = Math.abs(t[j] - x);
    if (d < bd) {
      bd = d;
      best = j;
    }
  }
  return bd <= tol ? best : -1;
}

function finiteNum(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

/** 0.5 -> "0.5", 2 -> "2", 0.125 -> "0.13" */
function trimNum(v) {
  if (!finiteNum(v)) return fmt.na;
  return String(Math.round(v * 100) / 100);
}

function autoFormat(v) {
  if (!finiteNum(v)) return fmt.na;
  return Number.isInteger(v) ? fmt.int(v) : fmt.num(v, Math.abs(v) < 10 ? 2 : 1);
}

// mounting

function mountChart(el, cls, initial, render) {
  el.classList.add('chart');
  if (cls) el.classList.add(cls);
  const st = { opts: { ...initial }, width: 0, alive: true, raf: 0 };
  const draw = () => {
    if (!st.alive) return;
    const w = Math.floor(el.clientWidth);
    st.width = w;
    if (w <= 0) return;
    // a redraw replaces the marks, so a focused mark would drop focus to the page: give it back to
    // the chart's current tab stop (the same mark index), which shows its tooltip again
    const had = typeof document !== 'undefined' && el.contains(document.activeElement);
    render(w, st.opts);
    if (had && !el.contains(document.activeElement)) {
      const mark = el.querySelector('.c-mark[tabindex="0"]');
      if (mark) mark.focus({ preventScroll: true });
    }
  };
  const schedule = () => {
    if (!st.raf) {
      st.raf = requestAnimationFrame(() => {
        st.raf = 0;
        draw();
      });
    }
  };
  const ro = typeof ResizeObserver !== 'undefined'
    ? new ResizeObserver(() => {
      if (Math.floor(el.clientWidth) !== st.width) schedule();
    })
    : null;
  if (ro) ro.observe(el);
  const offTheme = theme.onChange(schedule);
  draw();
  return {
    st,
    draw,
    update(next = {}) {
      st.opts = { ...st.opts, ...next };
      draw();
    },
    destroy() {
      st.alive = false;
      if (ro) ro.disconnect();
      offTheme();
      if (st.raf) cancelAnimationFrame(st.raf);
      tooltip.hide();
      clear(el);
      el.classList.remove('chart');
      if (cls) el.classList.remove(cls);
    },
  };
}

/**
 * One tab stop per chart: arrow keys move focus between marks. With cols > 0 the list is a row-major
 * grid and may hold nulls (blank cells), which are skipped.
 */
function rovingNav(el) {
  let marks = [];
  let cols = 0;
  let idx = -1;
  const firstIdx = () => marks.findIndex(Boolean);
  const focusAt = (i) => {
    idx = i;
    marks.forEach((m, j) => m && m.setAttribute('tabindex', j === i ? '0' : '-1'));
    marks[i].focus();
  };
  // a mark focused by Tab or a click becomes the tab stop, so a redraw gives focus back to it
  el.addEventListener('focusin', (e) => {
    const i = marks.indexOf(e.target);
    if (i < 0 || i === idx) return;
    idx = i;
    marks.forEach((m, j) => m && m.setAttribute('tabindex', j === i ? '0' : '-1'));
  });
  el.addEventListener('keydown', (e) => {
    if (!marks.length) return;
    const cur = marks.indexOf(document.activeElement);
    if (cur < 0) return;
    let delta = 0;
    switch (e.key) {
      case 'ArrowRight': delta = 1; break;
      case 'ArrowLeft': delta = -1; break;
      case 'ArrowDown': delta = cols || 1; break;
      case 'ArrowUp': delta = -(cols || 1); break;
      case 'Home': e.preventDefault(); focusAt(firstIdx()); return;
      case 'End': {
        e.preventDefault();
        for (let j = marks.length - 1; j >= 0; j -= 1) if (marks[j]) { focusAt(j); break; }
        return;
      }
      case 'Escape': tooltip.hide(); return;
      default: return;
    }
    e.preventDefault();
    let next = cur + delta;
    while (next >= 0 && next < marks.length && !marks[next]) next += delta;
    if (next >= 0 && next < marks.length) focusAt(next);
  });
  return {
    set(list, c = 0) {
      marks = list.slice();
      cols = c;
      if (idx < 0 || idx >= marks.length || !marks[idx]) idx = firstIdx();
      marks.forEach((m, j) => m && m.setAttribute('tabindex', j === idx ? '0' : '-1'));
    },
  };
}

/** hover, touch and focus all show the same tooltip for a mark. */
function bindTip(node, content, { onEnter, onLeave } = {}) {
  node.addEventListener('pointermove', (e) => {
    tooltip.show(e, content());
    if (onEnter) onEnter(e);
  });
  node.addEventListener('pointerdown', (e) => {
    if (e.pointerType !== 'mouse') tooltip.show(e, content());
  });
  node.addEventListener('pointerleave', () => {
    tooltip.hide();
    if (onLeave) onLeave();
  });
  node.addEventListener('focus', () => {
    tooltip.show(node, content());
    if (onEnter) onEnter(null);
  });
  node.addEventListener('blur', () => {
    tooltip.hide();
    if (onLeave) onLeave();
  });
}

function markGroup(label, extraClass) {
  return s('g', { class: ['c-mark', extraClass], attrs: { role: 'img', tabindex: -1, 'aria-label': label } });
}

function svgRoot(width, height, label) {
  return s('svg', {
    attrs: { width, height, viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': label || null },
  });
}

/**
 * Shared x-axis interaction for time charts: crosshair hover, click, drag to brush (handles resize,
 * dragging the band moves it), click clears, Escape clears; keyboard arrows move a readout and
 * Shift+arrows select a range.
 */
function xInteraction(target, cb) {
  let drag = null;
  let pending = null;
  let keyT = null;
  let anchor = null;
  let kbDirty = false;

  const tAt = (px, g) => g.v0 + ((px - g.left) / g.width) * (g.v1 - g.v0);
  const xAt = (t, g) => g.left + ((t - g.v0) / (g.v1 - g.v0)) * g.width;
  const clampT = (t, g) => Math.max(g.v0, Math.min(g.v1, t));
  const localX = (e) => e.clientX - target.getBoundingClientRect().left;
  const minSpan = (r, g) => {
    if (!r) return null;
    let [a, b] = r;
    const need = cb.minSpan ? cb.minSpan() : 5;
    if (b - a < need) {
      b = a + need;
      if (b > g.v1) {
        b = g.v1;
        a = Math.max(g.v0, b - need);
      }
    }
    return [a, b];
  };
  const drawPending = (r) => {
    pending = r;
    cb.drawBrush(r);
  };

  target.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    const g = cb.geom();
    if (!g) return;
    const px = localX(e);
    if (px < g.left - 4 || px > g.left + g.width + 4) return;
    const t = clampT(tAt(px, g), g);
    if (!cb.brushOn()) {
      drag = { mode: 'click', startX: px, t, moved: false };
      return;
    }
    const b = cb.brush();
    let mode = 'new';
    if (b) {
      const xa = xAt(b[0], g);
      const xb = xAt(b[1], g);
      if (Math.abs(px - xa) <= 6) mode = 'a';
      else if (Math.abs(px - xb) <= 6) mode = 'b';
      else if (px > xa && px < xb) mode = 'move';
    }
    drag = { mode, startX: px, t, orig: b ? b.slice() : null, moved: false };
    pending = b ? b.slice() : null;
    try {
      target.setPointerCapture(e.pointerId);
    } catch {
      // capture is a nicety (drags that leave the plot keep working)
    }
  });

  target.addEventListener('pointermove', (e) => {
    const g = cb.geom();
    if (!g) return;
    const px = localX(e);
    if (drag) {
      if (!drag.moved && Math.abs(px - drag.startX) >= 3) drag.moved = true;
      if (drag.mode !== 'click' && drag.moved) {
        tooltip.hide();
        cb.hover(null);
        const t = clampT(tAt(px, g), g);
        let r;
        if (drag.mode === 'new') r = [Math.min(drag.t, t), Math.max(drag.t, t)];
        else if (drag.mode === 'a') r = [Math.min(t, drag.orig[1]), Math.max(t, drag.orig[1])];
        else if (drag.mode === 'b') r = [Math.min(drag.orig[0], t), Math.max(drag.orig[0], t)];
        else {
          const d = t - drag.t;
          let a = drag.orig[0] + d;
          let b2 = drag.orig[1] + d;
          if (a < g.v0) {
            b2 += g.v0 - a;
            a = g.v0;
          }
          if (b2 > g.v1) {
            a -= b2 - g.v1;
            b2 = g.v1;
          }
          r = [a, b2];
        }
        target.classList.add('is-brushing');
        drawPending(r);
        return;
      }
    }
    if (px < g.left || px > g.left + g.width) {
      cb.hover(null);
      return;
    }
    cb.hover(clampT(tAt(px, g), g), e);
  });

  const finish = (e) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    target.classList.remove('is-brushing');
    try {
      target.releasePointerCapture(e.pointerId);
    } catch {
      // nothing captured
    }
    if (d.mode === 'click') {
      if (!d.moved && cb.click) cb.click(d.t);
      return;
    }
    if (!d.moved) {
      if (cb.brush()) cb.commit(null);
      else if (cb.click) cb.click(d.t);
      return;
    }
    const g = cb.geom();
    cb.commit(minSpan(pending, g));
  };
  target.addEventListener('pointerup', finish);
  target.addEventListener('pointercancel', () => {
    if (!drag) return;
    drag = null;
    target.classList.remove('is-brushing');
    cb.drawBrush(cb.brush());
  });
  target.addEventListener('pointerleave', () => {
    if (!drag) cb.hover(null);
  });

  target.addEventListener('keydown', (e) => {
    const g = cb.geom();
    if (!g) return;
    if (e.key === 'Escape') {
      if (cb.brush()) {
        e.preventDefault();
        cb.commit(null);
      }
      keyT = null;
      anchor = null;
      cb.hover(null);
      return;
    }
    const step = Math.max(cb.step ? cb.step() : 1, (g.v1 - g.v0) / 400);
    let next;
    if (keyT == null) {
      const c = cb.cursor ? cb.cursor() : null;
      keyT = finiteNum(c) ? clampT(c, g) : (g.v0 + g.v1) / 2;
    }
    switch (e.key) {
      case 'ArrowLeft': next = keyT - step; break;
      case 'ArrowRight': next = keyT + step; break;
      case 'PageUp': next = keyT - step * 10; break;
      case 'PageDown': next = keyT + step * 10; break;
      case 'Home': next = g.v0; break;
      case 'End': next = g.v1; break;
      default: return;
    }
    e.preventDefault();
    next = clampT(next, g);
    if (e.shiftKey && cb.brushOn()) {
      if (anchor == null) anchor = keyT;
      kbDirty = true;
      drawPending([Math.min(anchor, next), Math.max(anchor, next)]);
    } else {
      anchor = null;
    }
    keyT = next;
    cb.hover(keyT, null);
  });
  const commitKeyboard = () => {
    if (!kbDirty) return;
    kbDirty = false;
    anchor = null;
    const g = cb.geom();
    if (pending && g && pending[1] > pending[0]) cb.commit(minSpan(pending, g));
  };
  // commit when the key is released (a held arrow repeats keydown and sends one keyup)
  target.addEventListener('keyup', (e) => {
    if (e.key === 'Shift' || ['ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End'].includes(e.key)) {
      const keepAnchor = e.key !== 'Shift' && e.shiftKey;
      const a = anchor;
      commitKeyboard();
      if (keepAnchor) anchor = a;
    }
  });
  target.addEventListener('focusin', () => {
    keyT = null;
  });
  target.addEventListener('focusout', () => {
    commitKeyboard();
    keyT = null;
    cb.hover(null);
  });
}

function brushNodes(g, range, height, xAt) {
  const out = [];
  if (!range) return out;
  const xa = Math.max(g.left, Math.min(g.left + g.width, xAt(range[0])));
  const xb = Math.max(g.left, Math.min(g.left + g.width, xAt(range[1])));
  out.push(s('rect', { class: 'c-dim', attrs: { x: g.left, y: 0, width: Math.max(0, xa - g.left), height } }));
  out.push(s('rect', { class: 'c-dim', attrs: { x: xb, y: 0, width: Math.max(0, g.left + g.width - xb), height } }));
  out.push(s('rect', { class: 'c-brush', attrs: { x: xa, y: 0, width: Math.max(0, xb - xa), height } }));
  for (const x of [xa, xb]) {
    const xx = Math.round(x) + 0.5;
    out.push(s('line', { class: 'c-brush-edge', attrs: { x1: xx, x2: xx, y1: 0, y2: height } }));
    out.push(s('rect', { class: 'c-brush-handle', attrs: { x: xx - 3, y: height / 2 - 9, width: 6, height: 18, rx: 2 } }));
  }
  return out;
}

// legend and card

/** legend items [{label, color, kind: 'rect'|'line'|'dot'|'hatch', key}] -> <ul>. */
export function legend(items = []) {
  const ul = h('ul', { class: 'legend' });
  for (const it of items || []) {
    if (!it) continue;
    const kind = it.kind || 'rect';
    const sw = kind === 'hatch' || it.color === 'hatch'
      ? h('span', { class: 'swatch hatch' })
      : h('span', { class: ['swatch', kind === 'line' ? 'line' : null, kind === 'dot' ? 'dot' : null], style: { '--swatch': it.color || 'var(--ink-2)' } });
    ul.appendChild(h('li', { class: 'legend-item', dataset: { key: it.key ?? null } }, sw, h('span', { text: it.label })));
  }
  return ul;
}

function cellText(v, col) {
  if (typeof col.format === 'function') return col.format(v);
  if (v == null || (typeof v === 'number' && !Number.isFinite(v))) return fmt.na;
  if (typeof v === 'number') return autoFormat(v);
  return String(v);
}

function buildTable(spec) {
  const { columns = [], rows = [], caption } = spec || {};
  const table = h('table', { class: 'data' });
  if (caption) table.appendChild(h('caption', { class: 'sr-only', text: caption }));
  const thead = h('thead', {}, h('tr', {}, columns.map((c) => h('th', {
    attrs: { scope: 'col', 'data-align': c.align || null },
    text: c.label ?? c.key,
  }))));
  const tbody = h('tbody');
  for (const r of rows) {
    const tr = h('tr');
    for (const c of columns) {
      const v = r[c.key];
      const td = h('td', { attrs: { 'data-align': c.align || (typeof v === 'number' ? 'right' : null) } });
      if (v instanceof Node) td.appendChild(v);
      else if (v && typeof v === 'object' && 'text' in v) {
        if (v.color) td.appendChild(v.color === 'hatch' ? h('span', { class: 'swatch hatch' }) : h('span', { class: 'swatch', style: { '--swatch': v.color } }));
        td.appendChild(document.createTextNode(String(v.text)));
      } else td.textContent = cellText(v, c);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.append(thead, tbody);
  return h('div', { class: 'table-wrap' }, table);
}

/**
 * A chart card. Returns {el, body, setLegend(items|Node|null), setState(state, message), setTitle(t),
 * setSubtitle(t), setActions(nodes), refreshTable()}. table: () => ({columns: [{key, label, align,
 * format}], rows: [{...}]}); a cell may be a string, number, Node or {text, color}.
 */
export function card({ title, subtitle, legend: legendItems, actions, table, body, id, span, label } = {}) {
  const titleId = `card-${Math.random().toString(36).slice(2, 9)}`;
  const sec = h('section', {
    class: 'card',
    id: id || null,
    dataset: { span: span || null },
    style: span ? { '--span': String(span) } : null,
    attrs: { 'aria-labelledby': titleId },
  });
  const titleEl = h('h2', { class: 'card-title', id: titleId, text: title || '' });
  const subEl = h('p', { class: 'card-sub', text: subtitle || '', title: subtitle || null, hidden: !subtitle });
  const actionsEl = h('div', { class: 'card-actions' });
  const head = h('div', { class: 'card-head' }, h('div', { class: 'card-titles' }, titleEl, subEl), actionsEl);
  const legendEl = h('div', { class: 'card-legend' });
  const bodyEl = body instanceof Node && body.classList ? body : h('div');
  bodyEl.classList.add('card-body');
  if (body instanceof Node && body !== bodyEl) bodyEl.appendChild(body);
  const stateEl = h('div', { class: 'card-state' });
  const tableEl = h('div', { class: 'card-table', hidden: true });
  const noteEl = h('p', { class: 'card-note', attrs: { role: 'status' } });
  sec.append(head, legendEl, bodyEl, stateEl, tableEl, noteEl);

  let tableOn = false;
  let tableBtn = null;
  let extraActions = [];
  let state = 'ready';

  function renderActions() {
    clear(actionsEl);
    for (const a of extraActions) if (a) actionsEl.appendChild(a);
    if (tableBtn) actionsEl.appendChild(tableBtn);
    actionsEl.hidden = !actionsEl.childNodes.length;
  }

  function refreshTable() {
    if (!tableOn || typeof table !== 'function') return;
    clear(tableEl);
    let spec = null;
    try {
      spec = table();
    } catch (err) {
      spec = null;
    }
    if (!spec || !spec.rows || !spec.rows.length) tableEl.appendChild(emptyState({ title: 'No rows to show.' }));
    else tableEl.appendChild(buildTable({ caption: title, ...spec }));
  }

  function applyVisibility() {
    const blocked = state === 'empty' || state === 'error' || (state === 'loading' && sec.classList.contains('is-shimmer'));
    bodyEl.hidden = tableOn || blocked;
    tableEl.hidden = !tableOn || blocked;
    stateEl.hidden = !blocked;
  }

  if (typeof table === 'function') {
    tableBtn = h('button', {
      class: 'btn ghost sm',
      attrs: { type: 'button', 'aria-pressed': 'false', title: 'Show the numbers as a table' },
      on: {
        click: () => {
          tableOn = !tableOn;
          tableBtn.setAttribute('aria-pressed', tableOn ? 'true' : 'false');
          tooltip.hide();
          if (tableOn) refreshTable();
          applyVisibility();
        },
      },
    }, icon('table', 14), h('span', { text: 'Table' }));
  }
  extraActions = Array.isArray(actions) ? actions : actions ? [actions] : [];
  renderActions();

  function setLegend(items) {
    clear(legendEl);
    if (!items) return;
    legendEl.appendChild(items instanceof Node ? items : legend(items));
  }
  setLegend(legendItems);

  function setState(next, message) {
    state = next || 'ready';
    sec.classList.remove('is-loading', 'is-shimmer');
    clear(stateEl);
    noteEl.textContent = '';
    if (state === 'loading') {
      // hold a previous render at half opacity; shimmer only when there is nothing to hold
      if (bodyEl.childElementCount > 0) sec.classList.add('is-loading');
      else {
        sec.classList.add('is-shimmer');
        const minH = Math.max(120, Math.round(bodyEl.getBoundingClientRect().height) || 160);
        stateEl.appendChild(h('div', { class: 'shimmer', style: { minHeight: `${minH}px` }, attrs: { 'aria-hidden': 'true' } }));
      }
      if (message) noteEl.textContent = typeof message === 'string' ? message : message.title || '';
      sec.setAttribute('aria-busy', 'true');
    } else {
      sec.removeAttribute('aria-busy');
      if (state === 'empty' || state === 'error') {
        const m = typeof message === 'string' || message == null
          ? (state === 'error'
            ? { title: 'Could not load this panel.', body: message || null }
            : { title: message || 'Nothing to show.' })
          : message;
        stateEl.appendChild(emptyState({ ...m, kind: state === 'error' ? 'error' : 'empty' }));
      } else if (message) noteEl.textContent = typeof message === 'string' ? message : '';
    }
    applyVisibility();
    if (tableOn && state === 'ready') refreshTable();
  }
  applyVisibility();

  return {
    el: sec,
    body: bodyEl,
    setLegend,
    setState,
    setTitle(t) {
      titleEl.textContent = t || '';
    },
    setSubtitle(t) {
      subEl.textContent = t || '';
      subEl.title = t || '';
      subEl.hidden = !t;
    },
    setActions(nodes) {
      extraActions = Array.isArray(nodes) ? nodes : nodes ? [nodes] : [];
      renderActions();
    },
    setNote(t) {
      noteEl.textContent = t || '';
    },
    refreshTable,
  };
}

// timeline

const LANE_HEIGHTS = { header: 18, area: 32, segments: 12, cells: 12, lines: 44 };
const LANE_GAP = 4;
const HEADER_TOP = 8;
const AXIS_BAND = 20;

function laneLayout(lanes) {
  let y = 0;
  const rows = [];
  lanes.forEach((ln, i) => {
    const kind = ln.kind || 'cells';
    if (kind === 'header') {
      if (i > 0) y += HEADER_TOP;
      const hh = ln.height || LANE_HEIGHTS.header;
      rows.push({ ln, kind, y, h: hh });
      y += hh;
      return;
    }
    const hh = ln.height || LANE_HEIGHTS[kind] || 12;
    rows.push({ ln, kind, y, h: hh });
    y += hh + LANE_GAP;
  });
  return { rows, height: Math.max(0, y - LANE_GAP) };
}

function makeHatchPattern(ctx, col, dpr) {
  const size = Math.max(1, Math.round(5 * dpr));
  const tile = document.createElement('canvas');
  tile.width = size;
  tile.height = size;
  const t = tile.getContext('2d');
  t.fillStyle = col('var(--surface)');
  t.fillRect(0, 0, size, size);
  t.strokeStyle = col('var(--axis)');
  t.lineWidth = 1.4 * dpr;
  t.beginPath();
  for (const o of [-size, 0, size]) {
    t.moveTo(o, size);
    t.lineTo(o + size, 0);
  }
  t.stroke();
  const p = ctx.createPattern(tile, 'repeat');
  if (p && p.setTransform && typeof DOMMatrix !== 'undefined') p.setTransform(new DOMMatrix().scale(1 / dpr));
  return p;
}

const BIN_MODES = new Set(['mean', 'max', 'majority']);

/**
 * The bins an area or cells lane with `binning` uses at k px per second: when one step would be
 * narrower than `binMinPx` (default 2 px), the smallest multiple of the step that is at least that
 * wide, on the lane's own grid. null when the lane does not bin or its cells are wide enough.
 */
function laneBin(ln, k) {
  if (!BIN_MODES.has(ln.binning)) return null;
  const t = ln.t || [];
  if (!t.length || !(k > 0)) return null;
  const step = ln.step || medianStep(t);
  if (!(step > 0)) return null;
  const minPx = finiteNum(ln.binMinPx) && ln.binMinPx > 0 ? ln.binMinPx : 2;
  if (step * k >= minPx) return null;
  const size = Math.ceil(minPx / (step * k) - 1e-9) * step;
  const origin = ((t[0] % size) + size) % size;
  return { size, origin, mode: ln.binning, end: t[t.length - 1] + step };
}

/** one value for a bin: mean or max of the finite values, or the most frequent value (ties: not null). */
function binValue(mode, vals) {
  if (mode === 'majority') {
    const counts = new Map();
    let best;
    let bestN = 0;
    for (const v of vals) {
      const key = v === undefined ? null : v;
      const n = (counts.get(key) || 0) + 1;
      counts.set(key, n);
      if (n > bestN || (n === bestN && best == null && key != null)) {
        best = key;
        bestN = n;
      }
    }
    return bestN ? best : null;
  }
  let acc = null;
  let n = 0;
  for (const v of vals) {
    if (!finiteNum(v)) continue;
    if (mode === 'max') acc = acc == null ? v : Math.max(acc, v);
    else acc = (acc || 0) + v;
    n += 1;
  }
  if (!n) return null;
  return mode === 'max' ? acc : acc / n;
}

/** the binned samples of a lane between a and b: {t: [bin starts], v: [bin values]}. */
function binLane(ln, bin, a, b) {
  const t = ln.t || [];
  const v = ln.v || [];
  const start = bin.origin + Math.floor((a - bin.origin) / bin.size) * bin.size;
  const i0 = lowerBound(t, start - 1e-9);
  const i1 = lowerBound(t, b + bin.size);
  const bt = [];
  const bv = [];
  let cur = null;
  let vals = [];
  for (let i = i0; i < i1; i += 1) {
    const idx = Math.floor((t[i] - bin.origin) / bin.size + 1e-9);
    if (idx !== cur) {
      if (cur != null) {
        bt.push(bin.origin + cur * bin.size);
        bv.push(binValue(bin.mode, vals));
      }
      cur = idx;
      vals = [];
    }
    vals.push(v[i]);
  }
  if (cur != null) {
    bt.push(bin.origin + cur * bin.size);
    bv.push(binValue(bin.mode, vals));
  }
  return { t: bt, v: bv };
}

/** [start, end] of the bin that holds t in a binned row, or null. */
function binSpanAt(row, t) {
  const bin = row.bin;
  if (!bin) return null;
  const a = bin.origin + Math.floor((t - bin.origin) / bin.size + 1e-9) * bin.size;
  return [Math.max(a, (row.ln.t || [a])[0]), Math.min(a + bin.size, bin.end)];
}

function laneValueAt(row, t) {
  const { ln, kind } = row;
  if ((kind === 'area' || kind === 'cells') && row.bin) {
    const span = binSpanAt(row, t);
    const lt = ln.t || [];
    const i0 = lowerBound(lt, span[0] - 1e-9);
    const i1 = lowerBound(lt, span[1] - 1e-9);
    if (i1 <= i0) return null;
    return binValue(row.bin.mode, (ln.v || []).slice(i0, i1));
  }
  if (kind === 'area' || kind === 'cells') {
    const step = ln.step || medianStep(ln.t);
    const i = cellAt(ln.t, step, t);
    return i < 0 ? null : ln.v[i];
  }
  if (kind === 'segments') {
    const segs = ln.segs || [];
    let hit = null;
    // segments are sorted by start; scan back from the insertion point (turns can overlap)
    let lo = 0;
    let hi = segs.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (segs[mid][0] <= t) lo = mid + 1;
      else hi = mid;
    }
    for (let i = lo - 1, n = 0; i >= 0 && n < 64; i -= 1, n += 1) {
      if (segs[i][0] <= t && t < segs[i][1]) {
        hit = segs[i];
        break;
      }
    }
    return hit;
  }
  return null;
}

/**
 * Stacked time lanes over a shared x axis (session offsets in seconds).
 * opts: {span: [0, duration], view, lanes, brush, onBrush(range|null), onHover(t|null), onClick(t),
 *        cursor, now, gaps: [[s, e]], axisFormat, labelWidth: 132, label}
 * lanes: {kind: 'header', label} | {kind: 'area', key, label, swatch, height, t, v, step, max, color, format}
 *      | {kind: 'segments', key, label, swatch, height, segs: [[s, e, key?]], color | colorOf(key), format}
 *      | {kind: 'cells', key, label, swatch, height, t, v, step, fillOf(v), format(v)}
 *   area and cells lanes may set binning: 'mean' | 'max' | 'majority' (and binMinPx, default 2): when a
 *   step would be narrower than binMinPx, the lane draws bins of the smallest multiple of the step that
 *   is at least that wide, and the tooltip names the bin's span.
 *      | {kind: 'lines', key, label, height, series: [{key, label, color, t, v}], domain, refs: [{v, label}], format}
 * Returns {update(opts), setCursor(t), setBrush(range|null), destroy()}.
 */
export function timeline(el, opts = {}) {
  clear(el);
  el.classList.add('chart', 'tl');
  const labelsEl = h('div', { class: 'tl-labels', attrs: { 'aria-hidden': 'true' } });
  const plot = h('div', { class: 'tl-plot', attrs: { tabindex: 0, role: 'group', 'aria-roledescription': 'timeline' } });
  const canvas = h('canvas', { attrs: { 'aria-hidden': 'true' } });
  const svg = s('svg', { attrs: { 'aria-hidden': 'true' } });
  const overlay = s('g');
  const brushG = s('g');
  const cursorG = s('g');
  const cross = s('line', { class: 'c-crosshair', attrs: { x1: 0, x2: 0, y1: 0, y2: 0, visibility: 'hidden' } });
  svg.append(overlay, brushG, cursorG, cross);
  plot.append(canvas, svg);
  el.append(labelsEl, plot);

  const st = {
    o: { labelWidth: 132, brush: false, ...opts },
    brush: null,
    cursor: opts.cursor ?? null,
    geom: null,
    layout: null,
    labelSig: '',
    hoverT: null,
    hoverEvt: null,
    width: 0,
    alive: true,
    raf: 0,
    lastAnnounce: 0,
  };

  function viewOf(o) {
    const span = o.span || [0, 1];
    let v = o.view || span;
    if (!(v[1] > v[0])) v = [v[0], v[0] + 1];
    return v;
  }

  function fmtTime(t) {
    const o = st.o;
    if (typeof o.axisFormat === 'function') return o.axisFormat(t);
    const span = o.span || viewOf(o);
    return fmt.clockSpan(Math.max(span[1], Math.abs(span[0])))(t);
  }

  function renderLabels(L, labelW) {
    const sig = `${labelW}|${L.rows.map((r) => `${r.kind}:${r.ln.label}:${r.y}:${r.h}:${typeof r.ln.swatch === 'object' ? JSON.stringify(r.ln.swatch) : r.ln.swatch || ''}`).join('|')}`;
    if (sig === st.labelSig) return;
    st.labelSig = sig;
    clear(labelsEl);
    labelsEl.style.width = `${labelW}px`;
    for (const r of L.rows) {
      const ln = r.ln;
      const sw = ln.swatch;
      let swatch = null;
      if (sw) {
        const color = typeof sw === 'object' ? sw.color : sw;
        const kind = typeof sw === 'object' ? sw.kind : null;
        swatch = color === 'hatch' || kind === 'hatch'
          ? h('span', { class: 'swatch hatch' })
          : h('span', { class: ['swatch', kind === 'line' ? 'line' : null, kind === 'dot' ? 'dot' : null], style: { '--swatch': color } });
      }
      const top = r.y + r.h / 2 - 8;
      labelsEl.appendChild(h('div', {
        class: ['tl-label', r.kind === 'header' ? 'is-header' : null],
        style: { top: `${top}px`, width: `${labelW}px` },
        title: ln.label || null,
      }, swatch, h('span', { text: ln.label || '' })));
    }
  }

  function drawCanvas(W, L, g, ticks) {
    const dpr = window.devicePixelRatio || 1;
    const H = L.height;
    const cw = Math.max(1, Math.round(W * dpr));
    const ch = Math.max(1, Math.round(H * dpr));
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const col = colorResolver(el);
    let hatch = null;
    const hatchFill = () => {
      if (!hatch) hatch = makeHatchPattern(ctx, col, dpr);
      return hatch;
    };
    const k = W / (g.v1 - g.v0);
    const X = (t) => (t - g.v0) * k;
    const snap = (x) => Math.round(x * dpr) / dpr;

    ctx.fillStyle = col('var(--grid)');
    for (const t of ticks) ctx.fillRect(snap(X(t)), 0, 1, H);

    for (const r of L.rows) {
      const { ln, kind, y } = r;
      const hh = r.h;
      if (kind === 'header') continue;
      if (kind === 'cells' || kind === 'area') {
        r.bin = laneBin(ln, k);
        // a binned lane draws one value per bin (lane.binning: 'mean' | 'max' | 'majority')
        const binned = r.bin ? binLane(ln, r.bin, g.v0, g.v1) : null;
        const t = binned ? binned.t : ln.t || [];
        const v = binned ? binned.v : ln.v || [];
        if (!t.length) continue;
        const step = binned ? r.bin.size : ln.step || medianStep(t);
        const i0 = Math.max(0, lowerBound(t, g.v0 - step));
        const i1 = Math.min(t.length, lowerBound(t, g.v1 + 1e-9));
        if (kind === 'cells') {
          const fillOf = typeof ln.fillOf === 'function' ? ln.fillOf : (val) => (val ? ln.color || 'var(--ink-2)' : null);
          let runFill = null;
          let ra = 0;
          let rb = 0;
          const flush = () => {
            if (runFill == null) return;
            const a = snap(Math.max(0, ra));
            const b = snap(Math.min(W, rb));
            if (b > a) {
              ctx.fillStyle = runFill;
              ctx.fillRect(a, y, b - a, hh);
            }
          };
          for (let i = i0; i < i1; i += 1) {
            const f = fillOf(v[i]);
            const fs = f == null ? null : f === 'hatch' ? hatchFill() : col(f);
            const xa = X(t[i]);
            const xb = X(t[i] + step);
            if (fs != null && fs === runFill && xa <= rb + 0.01) {
              rb = xb;
              continue;
            }
            flush();
            runFill = fs;
            ra = xa;
            rb = xb;
          }
          flush();
        } else {
          const max = ln.max || 1;
          const color = col(ln.color || 'var(--ink-2)');
          const yOf = (val) => y + hh - Math.max(0, Math.min(1, val / max)) * (hh - 1);
          // bin to pixel columns (mean) so long sessions draw one step per pixel
          const pts = [];
          let binPx = null;
          let sum = 0;
          let n = 0;
          let bx0 = 0;
          let bx1 = 0;
          const flushBin = () => {
            if (binPx == null) return;
            pts.push(n ? [bx0, bx1, sum / n] : [bx0, bx1, null]);
            binPx = null;
            sum = 0;
            n = 0;
          };
          for (let i = i0; i < i1; i += 1) {
            const xa = Math.max(0, X(t[i]));
            const xb = Math.min(W, X(t[i] + step));
            if (xb <= xa) continue;
            const val = v[i];
            const px = Math.floor(xa);
            const contiguous = binPx != null && xa <= bx1 + 0.01;
            if (binPx != null && (px !== binPx || !contiguous || (val == null) !== (n === 0))) flushBin();
            if (binPx == null) {
              binPx = px;
              bx0 = xa;
            }
            bx1 = xb;
            if (val != null && Number.isFinite(val)) {
              sum += val;
              n += 1;
            }
          }
          flushBin();
          ctx.save();
          ctx.beginPath();
          let open = false;
          let lastX = 0;
          for (const [a, b, m] of pts) {
            if (m == null) {
              if (open) {
                ctx.lineTo(lastX, y + hh);
                ctx.closePath();
                open = false;
              }
              continue;
            }
            const yy = yOf(m);
            if (!open || a > lastX + 0.5) {
              if (open) {
                ctx.lineTo(lastX, y + hh);
                ctx.closePath();
              }
              ctx.moveTo(a, y + hh);
              open = true;
            }
            ctx.lineTo(a, yy);
            ctx.lineTo(b, yy);
            lastX = b;
          }
          if (open) {
            ctx.lineTo(lastX, y + hh);
            ctx.closePath();
          }
          ctx.globalAlpha = 0.16;
          ctx.fillStyle = color;
          ctx.fill();
          ctx.globalAlpha = 1;
          ctx.beginPath();
          open = false;
          lastX = 0;
          for (const [a, b, m] of pts) {
            if (m == null) {
              open = false;
              continue;
            }
            const yy = yOf(m);
            if (!open || a > lastX + 0.5) {
              ctx.moveTo(a, yy);
              open = true;
            } else ctx.lineTo(a, yy);
            ctx.lineTo(b, yy);
            lastX = b;
          }
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.lineJoin = 'round';
          ctx.stroke();
          ctx.fillStyle = col('var(--axis)');
          ctx.fillRect(0, y + hh - 1 / dpr, W, 1 / dpr);
          ctx.restore();
        }
      } else if (kind === 'segments') {
        const segs = ln.segs || [];
        let runFill = null;
        let ra = 0;
        let rb = 0;
        const flush = () => {
          if (runFill == null) return;
          const a = snap(Math.max(0, ra));
          let b = snap(Math.min(W, rb));
          if (b - a < 1) b = a + 1;
          ctx.fillStyle = runFill;
          ctx.fillRect(a, y, b - a, hh);
        };
        for (const sg of segs) {
          if (!sg || sg[1] < g.v0 || sg[0] > g.v1) continue;
          const c = typeof ln.colorOf === 'function' ? ln.colorOf(sg[2]) : ln.color;
          const fs = c === 'hatch' ? hatchFill() : col(c || 'var(--ink-2)');
          const xa = X(sg[0]);
          const xb = X(sg[1]);
          if (fs === runFill && xa <= rb + 0.5) {
            rb = Math.max(rb, xb);
            continue;
          }
          flush();
          runFill = fs;
          ra = xa;
          rb = xb;
        }
        flush();
      } else if (kind === 'lines') {
        const series = ln.series || [];
        const dom = lineDomain(ln);
        const pad = 3;
        const yOf = (val) => y + pad + (1 - (val - dom[0]) / (dom[1] - dom[0])) * (hh - 2 * pad);
        ctx.fillStyle = col('var(--grid)');
        ctx.fillRect(0, snap(y + hh - 0.5), W, 1);
        for (const ref of ln.refs || []) {
          if (!finiteNum(ref.v) || ref.v < dom[0] || ref.v > dom[1]) continue;
          ctx.globalAlpha = 0.5;
          ctx.fillStyle = col('var(--ink-2)');
          ctx.fillRect(0, snap(yOf(ref.v)), W, 1 / dpr);
          ctx.globalAlpha = 1;
        }
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, y, W, hh);
        ctx.clip();
        for (const se of series) {
          drawLineSeries(ctx, se, g, X, yOf, W, col);
        }
        ctx.restore();
      }
    }

    for (const gap of st.o.gaps || []) {
      if (!gap || gap[1] < g.v0 || gap[0] > g.v1) continue;
      const a = snap(Math.max(0, X(gap[0])));
      const b = snap(Math.min(W, X(gap[1])));
      if (b - a < 0.5) continue;
      ctx.fillStyle = hatchFill();
      ctx.globalAlpha = 0.9;
      ctx.fillRect(a, 0, b - a, H);
      ctx.globalAlpha = 1;
    }
  }

  function drawOverlay(W, L, g, ticks) {
    const H = L.height + AXIS_BAND;
    svg.setAttribute('width', W);
    svg.setAttribute('height', H);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    clear(overlay);
    const k = W / (g.v1 - g.v0);
    const X = (t) => (t - g.v0) * k;
    overlay.appendChild(s('line', { class: 'c-axis', attrs: { x1: 0, x2: W, y1: L.height + 0.5, y2: L.height + 0.5 } }));
    const nowX = finiteNum(st.o.now) && st.o.now >= g.v0 && st.o.now <= g.v1 ? X(st.o.now) : null;
    for (const t of ticks) {
      const x = X(t);
      if (nowX != null && Math.abs(x - nowX) < 40) continue;
      const anchor = x < 24 ? 'start' : x > W - 24 ? 'end' : 'middle';
      overlay.appendChild(s('text', { class: 'c-tick', attrs: { x, y: L.height + 14, 'text-anchor': anchor }, text: fmtTime(t) }));
    }
    for (const r of L.rows) {
      if (r.kind !== 'lines') continue;
      const ln = r.ln;
      const dom = lineDomain(ln);
      const pad = 3;
      const yOf = (val) => r.y + pad + (1 - (val - dom[0]) / (dom[1] - dom[0])) * (r.h - 2 * pad);
      const f = typeof ln.format === 'function' ? ln.format : autoFormat;
      overlay.appendChild(s('text', { class: 'c-ref-label', attrs: { x: 2, y: r.y + 9 }, text: f(dom[1]) }));
      for (const ref of ln.refs || []) {
        if (!finiteNum(ref.v) || ref.v < dom[0] || ref.v > dom[1]) continue;
        overlay.appendChild(s('text', { class: 'c-ref-label', attrs: { x: W - 2, y: yOf(ref.v) - 2, 'text-anchor': 'end' }, text: ref.label ?? f(ref.v) }));
      }
    }
    if (nowX != null) {
      const x = Math.round(nowX) - 0.75;
      overlay.appendChild(s('line', { class: 'c-now', attrs: { x1: x, x2: x, y1: 0, y2: L.height + 4 } }));
      overlay.appendChild(s('text', { class: 'c-now-label', attrs: { x: Math.min(W, x + 2), y: L.height + 14, 'text-anchor': x > W - 30 ? 'end' : 'start' }, text: 'now' }));
    }
    cross.setAttribute('y2', L.height);
    drawBrush(st.brush);
    drawCursor();
  }

  function drawBrush(range) {
    clear(brushG);
    const g = st.geom;
    const L = st.layout;
    if (!g || !L || !range) return;
    const k = g.width / (g.v1 - g.v0);
    for (const n of brushNodes({ left: 0, width: g.width }, range, L.height, (t) => (t - g.v0) * k)) brushG.appendChild(n);
  }

  function drawCursor() {
    clear(cursorG);
    const g = st.geom;
    const L = st.layout;
    const c = st.cursor;
    if (!g || !L || !finiteNum(c) || c < g.v0 || c > g.v1) return;
    const x = Math.round(((c - g.v0) / (g.v1 - g.v0)) * g.width) + 0.25;
    cursorG.appendChild(s('line', { class: 'c-cursor', attrs: { x1: x, x2: x, y1: 0, y2: L.height } }));
    cursorG.appendChild(s('circle', { attrs: { cx: x, cy: 0, r: 3 }, style: { fill: 'var(--ink)' } }));
  }

  function tooltipRows(t) {
    const rows = [];
    // header lanes and lines lanes open a section; the first plain lane after a lines lane closes it
    let open = null;
    for (const r of st.layout.rows) {
      const ln = r.ln;
      if (r.kind === 'header') {
        rows.push({ section: ln.label || '' });
        open = 'header';
        continue;
      }
      if (r.kind !== 'lines' && open === 'lines') {
        rows.push({ section: '' });
        open = null;
      }
      if (r.kind === 'lines') {
        const series = ln.series || [];
        if (!series.length) continue;
        rows.push({ section: ln.label || '' });
        open = 'lines';
        const f = typeof ln.format === 'function' ? ln.format : autoFormat;
        for (const se of series) {
          const tol = (ln.step || medianStep(se.t)) * 0.75;
          const i = nearestAt(se.t, t, tol);
          const val = i < 0 ? null : se.v[i];
          rows.push({ value: val == null ? fmt.na : f(val), label: se.label, color: se.color, key: 'line' });
        }
        continue;
      }
      const val = laneValueAt(r, t);
      if (r.kind === 'segments') {
        const key = val ? val[2] : null;
        const f = typeof ln.format === 'function' ? ln.format : (seg) => (seg ? (seg[2] != null ? String(seg[2]) : 'yes') : 'no');
        const color = val ? (typeof ln.colorOf === 'function' ? ln.colorOf(key) : ln.color) : null;
        rows.push({ value: f(val), label: ln.label, color: color || 'var(--grid)', key: 'line' });
        continue;
      }
      // a binned lane reports the bin it shows: 'Speech activity, 00:12:00 to 00:12:06'
      const span = r.bin ? binSpanAt(r, t) : null;
      const label = span ? `${ln.label || ''}, ${fmtTime(span[0])} to ${fmtTime(span[1])}` : ln.label;
      if (r.kind === 'area') {
        const f = typeof ln.format === 'function' ? ln.format : (x) => fmt.pct(x);
        rows.push({ value: val == null ? fmt.na : f(val), label, color: ln.color, key: 'line' });
        continue;
      }
      const f = typeof ln.format === 'function' ? ln.format : (x) => (x == null ? fmt.na : String(x));
      const fill = typeof ln.fillOf === 'function' ? ln.fillOf(val) : (val ? ln.color : null);
      rows.push({ value: f(val), label, color: fill || 'var(--grid)', key: fill === 'hatch' ? 'hatch' : 'line' });
    }
    // drop sections that hold no rows
    return rows.filter((row, i) => row.section == null || (rows[i + 1] && rows[i + 1].section == null));
  }

  function hover(t, evt) {
    const g = st.geom;
    if (t == null || !g) {
      st.hoverT = null;
      st.hoverEvt = null;
      cross.setAttribute('visibility', 'hidden');
      tooltip.hide();
      if (typeof st.o.onHover === 'function') st.o.onHover(null);
      return;
    }
    st.hoverT = t;
    st.hoverEvt = evt ? { clientX: evt.clientX, clientY: evt.clientY } : null;
    const x = Math.round(((t - g.v0) / (g.v1 - g.v0)) * g.width) + 0.5;
    cross.setAttribute('x1', x);
    cross.setAttribute('x2', x);
    cross.setAttribute('visibility', 'visible');
    const content = { title: fmtTime(t), rows: tooltipRows(t) };
    if (evt) tooltip.show(evt, content);
    else {
      const rect = plot.getBoundingClientRect();
      tooltip.show({ x: rect.left + x, y: rect.top + 4 }, content);
      const now = performance.now();
      if (now - st.lastAnnounce > 400) {
        st.lastAnnounce = now;
        announce(`${content.title}. ${content.rows.filter((r) => r.section == null).map((r) => `${r.label} ${r.value}`).join(', ')}`);
      }
    }
    if (typeof st.o.onHover === 'function') st.o.onHover(t);
  }

  function render(W) {
    const o = st.o;
    const view = viewOf(o);
    const labelW = Math.min(o.labelWidth ?? 132, Math.max(76, Math.round(W * 0.3)));
    const plotW = Math.max(40, W - labelW);
    const L = laneLayout(o.lanes || []);
    const H = L.height + AXIS_BAND;
    el.style.height = `${H}px`;
    plot.style.left = `${labelW}px`;
    plot.style.width = `${plotW}px`;
    plot.style.height = `${H}px`;
    const g = { left: 0, width: plotW, v0: view[0], v1: view[1] };
    st.geom = g;
    st.layout = L;
    const help = o.brush
      ? 'Arrow keys move the time readout, Shift with arrow keys selects a range, Escape clears it.'
      : 'Arrow keys move the time readout.';
    plot.setAttribute('aria-label', `${o.label || 'Timeline'}. ${help}`);
    renderLabels(L, labelW);
    const ticks = timeTicks(view[0], view[1], plotW);
    drawCanvas(plotW, L, g, ticks);
    drawOverlay(plotW, L, g, ticks);
    if (st.hoverT != null) {
      if (st.hoverEvt) hover(Math.max(g.v0, Math.min(g.v1, st.hoverT)), st.hoverEvt);
      else if (document.activeElement === plot) hover(Math.max(g.v0, Math.min(g.v1, st.hoverT)), null);
    }
  }

  function draw() {
    if (!st.alive) return;
    const w = Math.floor(el.clientWidth);
    st.width = w;
    if (w <= 0) return;
    render(w);
  }
  const schedule = () => {
    if (!st.raf) {
      st.raf = requestAnimationFrame(() => {
        st.raf = 0;
        draw();
      });
    }
  };
  const ro = typeof ResizeObserver !== 'undefined'
    ? new ResizeObserver(() => {
      if (Math.floor(el.clientWidth) !== st.width) schedule();
    })
    : null;
  if (ro) ro.observe(el);
  const offTheme = theme.onChange(schedule);

  xInteraction(plot, {
    geom: () => st.geom,
    brushOn: () => !!st.o.brush,
    brush: () => st.brush,
    drawBrush: (r) => drawBrush(r),
    commit: (r) => {
      st.brush = r;
      drawBrush(r);
      if (typeof st.o.onBrush === 'function') st.o.onBrush(r ? [r[0], r[1]] : null);
    },
    hover,
    click: (t) => {
      if (typeof st.o.onClick === 'function') st.o.onClick(t);
    },
    cursor: () => st.cursor,
    step: () => {
      let m = Infinity;
      for (const r of (st.layout && st.layout.rows) || []) {
        if (r.kind === 'cells' || r.kind === 'area') m = Math.min(m, r.ln.step || medianStep(r.ln.t));
      }
      return Number.isFinite(m) ? m : 1;
    },
  });

  draw();

  return {
    update(next = {}) {
      if ('cursor' in next) st.cursor = next.cursor;
      st.o = { ...st.o, ...next };
      draw();
    },
    setCursor(t) {
      st.cursor = finiteNum(t) ? t : null;
      drawCursor();
    },
    setBrush(range) {
      st.brush = range && finiteNum(range[0]) && finiteNum(range[1]) ? [Math.min(range[0], range[1]), Math.max(range[0], range[1])] : null;
      drawBrush(st.brush);
    },
    destroy() {
      st.alive = false;
      if (ro) ro.disconnect();
      offTheme();
      if (st.raf) cancelAnimationFrame(st.raf);
      tooltip.hide();
      clear(el);
      el.classList.remove('chart', 'tl');
      el.style.height = '';
    },
  };
}

function lineDomain(ln) {
  if (ln.domain && finiteNum(ln.domain[0]) && finiteNum(ln.domain[1]) && ln.domain[1] > ln.domain[0]) return ln.domain;
  let lo = Infinity;
  let hi = -Infinity;
  for (const se of ln.series || []) {
    for (const v of se.v || []) {
      if (!finiteNum(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  for (const ref of ln.refs || []) {
    if (finiteNum(ref.v)) {
      lo = Math.min(lo, ref.v);
      hi = Math.max(hi, ref.v);
    }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  const [a, b] = niceScale(Math.min(0, lo), hi, 2);
  return [a, b];
}

/** draw one series on a canvas with null breaks and per-pixel min/max decimation. */
function drawLineSeries(ctx, se, g, X, yOf, W, col) {
  const t = se.t || [];
  const v = se.v || [];
  if (!t.length) return;
  const step = medianStep(t);
  const i0 = Math.max(0, lowerBound(t, g.v0) - 1);
  const i1 = Math.min(t.length, lowerBound(t, g.v1) + 1);
  const dense = i1 - i0 > W * 2;
  ctx.beginPath();
  let open = false;
  let lastT = null;
  let binPx = null;
  let bmin = 0;
  let bmax = 0;
  let bfirst = 0;
  let blast = 0;
  const flushBin = () => {
    if (binPx == null) return;
    const x = binPx + 0.5;
    if (!open) {
      ctx.moveTo(x, yOf(bfirst));
      open = true;
    } else ctx.lineTo(x, yOf(bfirst));
    if (bmin !== bfirst || bmax !== bfirst) {
      ctx.lineTo(x, yOf(bmin));
      ctx.lineTo(x, yOf(bmax));
    }
    ctx.lineTo(x, yOf(blast));
    binPx = null;
  };
  for (let i = i0; i < i1; i += 1) {
    const val = v[i];
    const tt = t[i];
    const broken = val == null || !Number.isFinite(val) || (lastT != null && tt - lastT > step * 3);
    if (broken) {
      flushBin();
      open = false;
      if (val == null || !Number.isFinite(val)) {
        lastT = null;
        continue;
      }
    }
    lastT = tt;
    const x = X(tt);
    if (dense) {
      const px = Math.floor(x);
      if (binPx != null && px !== binPx) flushBin();
      if (binPx == null) {
        binPx = px;
        bmin = val;
        bmax = val;
        bfirst = val;
      }
      bmin = Math.min(bmin, val);
      bmax = Math.max(bmax, val);
      blast = val;
    } else if (!open) {
      ctx.moveTo(x, yOf(val));
      open = true;
    } else ctx.lineTo(x, yOf(val));
  }
  flushBin();
  ctx.strokeStyle = col(se.color || 'var(--ink-2)');
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.stroke();
}

// bar list

/**
 * Horizontal bars with the value at the tip.
 * opts: {items: [{key, label, value, color, note, sub}], format, max, height: 22, subLabel, valueLabel, label}
 * `sub` (number) draws a reference tick on the bar (for example the session-so-far share); a string
 * `sub` is shown after the note.
 */
export function barList(el, opts = {}) {
  const nav = rovingNav(el);
  return publicHandle(mountChart(el, 'c-barlist', opts, (W, o) => {
    const items = (o.items || []).filter(Boolean);
    const format = o.format || autoFormat;
    const band = Math.max(14, o.height || 22);
    const pitch = band + 6;
    const thick = Math.min(BAR_MAX, Math.max(6, band - 8));
    let max = finiteNum(o.max) ? o.max : 0;
    for (const it of items) {
      if (finiteNum(it.value)) max = Math.max(max, it.value);
      if (!finiteNum(o.max) && finiteNum(it.sub)) max = Math.max(max, it.sub);
    }
    if (!(max > 0)) max = 1;
    const valueTexts = items.map((it) => {
      const val = finiteNum(it.value) ? format(it.value) : fmt.na;
      const note = [it.note, typeof it.sub === 'string' ? it.sub : null].filter(Boolean).join('  ');
      return { val, note };
    });
    const valueW = Math.max(30, ...valueTexts.map((t) => textWidth(t.val, 12, 600) + (t.note ? textWidth(` ${t.note}`, 12) + 6 : 0))) + 10;
    const maxLabel = Math.max(0, ...items.map((it) => textWidth(it.label ?? '', 13)));
    let labelW = Math.min(maxLabel + 22, Math.round(W * 0.42));
    if (W - labelW - valueW < 60) labelW = Math.max(60, W - valueW - 60);
    const barW = Math.max(20, W - labelW - valueW);
    const H = Math.max(pitch, items.length * pitch);
    clear(el);
    const svg = svgRoot(W, H, o.label);
    const marks = [];
    items.forEach((it, i) => {
      const y = i * pitch;
      const cy = y + pitch / 2;
      const color = it.color || 'var(--tag-1)';
      const val = finiteNum(it.value) ? it.value : null;
      const len = val == null ? 0 : Math.max(val > 0 ? 1 : 0, (Math.min(val, max) / max) * barW);
      const vt = valueTexts[i];
      const aria = `${it.label ?? ''}: ${vt.val}${vt.note ? `, ${vt.note}` : ''}`;
      const g = markGroup(aria);
      g.appendChild(s('rect', { class: 'c-hit', attrs: { x: 0, y, width: W, height: pitch } }));
      g.appendChild(s('rect', { attrs: { x: 0, y: cy - 5, width: 10, height: 10, rx: 3 }, style: { fill: paint(color) } }));
      g.appendChild(s('text', { class: 'c-label', attrs: { x: 16, y: cy + 4.5 }, text: fitText(it.label ?? '', labelW - 22, 13) }));
      if (len > 0) g.appendChild(s('path', { class: 'c-fill', attrs: { d: barPath(labelW, cy - thick / 2, len, thick) }, style: { fill: paint(color) } }));
      if (finiteNum(it.sub)) {
        const sx = labelW + (Math.min(it.sub, max) / max) * barW;
        g.appendChild(s('line', { attrs: { x1: sx, x2: sx, y1: cy - thick / 2 - 3, y2: cy + thick / 2 + 3 }, style: { stroke: 'var(--ink)', strokeWidth: 2, strokeLinecap: 'round' } }));
      }
      const tx = labelW + len + 6;
      const text = s('text', { class: 'c-value', attrs: { x: tx, y: cy + 4 } }, vt.val);
      if (vt.note) text.appendChild(s('tspan', { class: 'c-note', attrs: { dx: 6 }, text: vt.note }));
      g.appendChild(text);
      bindTip(g, () => ({
        title: it.label ?? '',
        rows: [
          { value: vt.val, label: o.valueLabel || '', color, key: 'rect' },
          it.note ? { value: it.note, label: o.noteLabel || '', key: 'none' } : null,
          finiteNum(it.sub) ? { value: format(it.sub), label: o.subLabel || 'Reference', color: 'var(--ink)', key: 'line' } : null,
        ].filter(Boolean),
      }));
      svg.appendChild(g);
      marks.push(g);
    });
    el.appendChild(svg);
    nav.set(marks);
  }));
}

function publicHandle(m) {
  return { update: m.update, destroy: m.destroy };
}

// stacked bars

/**
 * Part-to-whole bars. Horizontal by default (one bar per row); orient: 'vertical' draws columns along
 * x (for example one per minute) with a y axis.
 * opts: {rows: [{key, label, swatch, values: {cat: n}}], categories: [{key, label, fill}], normalize: true,
 *        format (raw amounts; shares always show as %), valueLabel, orient: 'horizontal'|'vertical',
 *        height (vertical: 200), marginLeft (vertical), label}. Vertical columns default to normalize: false.
 */
export function stackedBars(el, opts = {}) {
  const nav = rovingNav(el);
  return publicHandle(mountChart(el, 'c-stacked', opts, (W, o) => {
    if (o.orient === 'vertical' || o.orient === 'v') renderColumns(el, W, o, nav);
    else renderStackRows(el, W, o, nav);
  }));
}

function stackTotals(rows, cats) {
  return rows.map((r) => cats.reduce((acc, c) => acc + (finiteNum(r.values && r.values[c.key]) && r.values[c.key] > 0 ? r.values[c.key] : 0), 0));
}

function renderStackRows(el, W, o, nav) {
  const rows = (o.rows || []).filter(Boolean);
  const cats = o.categories || [];
  const normalize = o.normalize !== false;
  const format = o.format || autoFormat;
  const totals = stackTotals(rows, cats);
  const maxTotal = Math.max(1e-9, ...totals);
  const pitch = 32;
  const thick = 18;
  const maxLabel = Math.max(0, ...rows.map((r) => textWidth(r.label ?? '', 13)));
  const labelW = Math.min(maxLabel + (rows.some((r) => r.swatch) ? 22 : 8), Math.round(W * 0.38));
  const totalW = normalize ? 0 : Math.max(...totals.map((t) => textWidth(format(t), 12, 600))) + 10;
  const barW = Math.max(30, W - labelW - totalW);
  const H = Math.max(pitch, rows.length * pitch);
  const col = colorResolver(el);
  clear(el);
  const svg = svgRoot(W, H, o.label);
  const marks = [];
  rows.forEach((r, ri) => {
    const cy = ri * pitch + pitch / 2;
    const total = totals[ri];
    const lx = r.swatch ? 16 : 0;
    if (r.swatch) svg.appendChild(s('rect', { attrs: { x: 0, y: cy - 5, width: 10, height: 10, rx: 3 }, style: { fill: paint(r.swatch) } }));
    svg.appendChild(s('text', { class: 'c-label', attrs: { x: lx, y: cy + 4.5 }, text: fitText(r.label ?? '', labelW - lx - 6, 13) }));
    if (!(total > 0)) {
      svg.appendChild(s('text', { class: 'c-note', attrs: { x: labelW, y: cy + 4 }, text: fmt.na }));
      return;
    }
    const full = normalize ? barW : (total / maxTotal) * barW;
    const segs = cats
      .map((c) => ({ c, v: r.values ? r.values[c.key] : null }))
      .filter((x) => finiteNum(x.v) && x.v > 0);
    let x = labelW;
    segs.forEach((sg, si) => {
      const w = (sg.v / total) * full;
      const last = si === segs.length - 1;
      const drawW = last ? w : Math.max(0.5, w - GAP);
      const share = sg.v / total;
      const valText = normalize ? fmt.pct(share) : format(sg.v);
      const g = markGroup(`${r.label ?? ''}, ${sg.c.label}: ${valText}`);
      g.appendChild(s('rect', { class: 'c-hit', attrs: { x, y: cy - pitch / 2, width: Math.max(w, 1), height: pitch } }));
      g.appendChild(s('path', { class: 'c-fill', attrs: { d: barPath(x, cy - thick / 2, drawW, thick, RADIUS, last ? 'right' : 'none') }, style: { fill: paint(sg.c.fill) } }));
      const lbl = normalize ? fmt.pct(share) : format(sg.v);
      if (sg.c.fill !== 'hatch' && drawW >= textWidth(lbl, 11, 600) + 10) {
        g.appendChild(s('text', {
          class: 'c-inlabel',
          attrs: { x: x + drawW / 2, y: cy + 4, 'text-anchor': 'middle' },
          style: { fill: textOn(col(sg.c.fill)) },
          text: lbl,
        }));
      }
      bindTip(g, () => ({
        title: r.label ?? '',
        rows: [
          { value: normalize ? fmt.pct(share) : format(sg.v), label: sg.c.label, color: sg.c.fill, key: sg.c.fill === 'hatch' ? 'hatch' : 'rect' },
          normalize ? { value: format(sg.v), label: o.valueLabel || 'Amount', key: 'none' } : null,
        ].filter(Boolean),
        note: normalize ? null : `Total ${format(total)}`,
      }));
      svg.appendChild(g);
      marks.push(g);
      x += w;
    });
    if (!normalize) svg.appendChild(s('text', { class: 'c-value', attrs: { x: labelW + full + 6, y: cy + 4 }, text: format(total) }));
  });
  el.appendChild(svg);
  nav.set(marks);
}

function renderColumns(el, W, o, nav) {
  const rows = (o.rows || []).filter(Boolean);
  const cats = o.categories || [];
  const normalize = o.normalize === true;
  const format = o.format || autoFormat;
  const H = o.height || 200;
  const totals = stackTotals(rows, cats);
  const [lo, hi, ticks] = normalize ? [0, 1, [0, 0.25, 0.5, 0.75, 1]] : niceScale(0, Math.max(...totals, 0), 4);
  const tickFmt = normalize ? (v) => fmt.pct(v) : (o.tickFormat || format);
  const left = Math.max(o.marginLeft || 0, Math.ceil(Math.max(...ticks.map((t) => textWidth(tickFmt(t), 11)))) + 8);
  const top = 8;
  const bottom = 20;
  const plotW = Math.max(40, W - left - 4);
  const plotH = H - top - bottom;
  const n = Math.max(1, rows.length);
  const slot = plotW / n;
  const colW = Math.min(BAR_MAX, Math.max(1, slot - GAP));
  const Y = (v) => top + plotH - ((v - lo) / (hi - lo)) * plotH;
  clear(el);
  const svg = svgRoot(W, H, o.label);
  for (const t of ticks) {
    const y = Math.round(Y(t)) + 0.5;
    svg.appendChild(s('line', { class: t === 0 ? 'c-axis' : 'c-grid', attrs: { x1: left, x2: W, y1: y, y2: y } }));
    svg.appendChild(s('text', { class: 'c-tick', attrs: { x: left - 6, y: y + 4, 'text-anchor': 'end' }, text: tickFmt(t) }));
  }
  const maxLabel = Math.max(1, ...rows.map((r) => textWidth(r.label ?? '', 11)));
  const every = Math.max(1, Math.ceil((maxLabel + 10) / slot));
  const marks = [];
  rows.forEach((r, i) => {
    const x0 = left + i * slot;
    const cx = x0 + slot / 2;
    const total = totals[i];
    if (i % every === 0) {
      const lw = textWidth(r.label ?? '', 11);
      const lx = Math.max(left, Math.min(W - lw - 1, cx - lw / 2));
      svg.appendChild(s('text', { class: 'c-tick', attrs: { x: lx, y: H - 5 }, text: r.label ?? '' }));
    }
    const g = markGroup(`${r.label ?? ''}: ${normalize ? '' : format(total)}`);
    g.appendChild(s('rect', { class: 'c-hit', attrs: { x: x0, y: top, width: slot, height: plotH } }));
    const segs = cats.map((c) => ({ c, v: r.values ? r.values[c.key] : null })).filter((x) => finiteNum(x.v) && x.v > 0);
    let base = 0;
    segs.forEach((sg, si) => {
      const val = normalize ? sg.v / (total || 1) : sg.v;
      const y1 = Y(base);
      const y2 = Y(base + val);
      const lastSeg = si === segs.length - 1;
      const hgt = Math.max(0, y1 - y2 - (lastSeg ? 0 : GAP));
      if (hgt > 0) {
        g.appendChild(s('path', {
          class: 'c-fill',
          attrs: { d: barPath(cx - colW / 2, y1 - hgt, colW, hgt, Math.min(RADIUS, colW / 2), lastSeg ? 'top' : 'none') },
          style: { fill: paint(sg.c.fill) },
        }));
      }
      base += val;
    });
    bindTip(g, () => ({
      title: r.label ?? '',
      rows: segs.slice().reverse().map((sg) => ({
        value: normalize ? fmt.pct(sg.v / (total || 1)) : format(sg.v),
        label: sg.c.label,
        color: sg.c.fill,
        key: sg.c.fill === 'hatch' ? 'hatch' : 'rect',
      })),
      note: normalize ? null : `Total ${format(total)}`,
    }));
    svg.appendChild(g);
    marks.push(g);
  });
  el.appendChild(svg);
  nav.set(marks);
}

// heat matrix

/**
 * Matrix of values with the sequential ramp (rows: from / looker, cols: to / target).
 * opts: {rows: [{key, label, swatch}], cols: [{key, label, swatch}], values: [[n | null]], format, max,
 *        diagonal: 'blank'|'show', cellSize, scaleLabel, valueLabel, title(row, col), label}
 */
export function heatMatrix(el, opts = {}) {
  const nav = rovingNav(el);
  return publicHandle(mountChart(el, 'c-matrix', opts, (W, o) => {
    const rows = o.rows || [];
    const cols = o.cols || [];
    const values = o.values || [];
    const format = o.format || autoFormat;
    const blankDiag = o.diagonal === 'blank';
    let max = finiteNum(o.max) ? o.max : 0;
    if (!finiteNum(o.max)) {
      rows.forEach((r, i) => cols.forEach((c, j) => {
        if (blankDiag && r.key === c.key) return;
        const v = values[i] && values[i][j];
        if (finiteNum(v)) max = Math.max(max, v);
      }));
    }
    const hasSw = rows.some((r) => r.swatch);
    const maxRowLabel = Math.max(0, ...rows.map((r) => textWidth(r.label ?? '', 13)));
    const rowLabelW = Math.min(maxRowLabel + (hasSw ? 24 : 10), Math.round(W * 0.4));
    const avail = W - rowLabelW;
    const maxColLabel = Math.max(0, ...cols.map((c) => textWidth(c.label ?? '', 12) + (c.swatch ? 14 : 0)));
    const ncol = Math.max(1, cols.length);
    let cell = Math.floor(Math.max(18, Math.min(o.cellSize || 56, avail / ncol)));
    let rotate = maxColLabel > cell - 6;
    if (rotate && cols.length) {
      // a rotated head reaches past its column to the right: keep the last one inside the chart
      const last = cols[cols.length - 1];
      const lastW = Math.min(84, textWidth(last.label ?? '', 12)) + (last.swatch ? 12 : 0);
      const reach = (c) => 4 + Math.cos((40 * Math.PI) / 180) * lastW - c / 2;
      const over = rowLabelW + ncol * cell + reach(cell) - (W - 2);
      if (over > 0) {
        cell = Math.floor(Math.max(18, (W - 2 - rowLabelW - 4 - Math.cos((40 * Math.PI) / 180) * lastW) / (ncol - 0.5)));
        rotate = maxColLabel > cell - 6;
      }
    }
    const headH = rotate ? Math.min(90, Math.ceil(maxColLabel * 0.72) + 16) : 22;
    const H = headH + rows.length * cell;
    const col = colorResolver(el);
    clear(el);
    const svg = svgRoot(W, H, o.label);
    cols.forEach((c, j) => {
      const cx = rowLabelW + j * cell + cell / 2;
      if (rotate) {
        const tx = cx + 4;
        const ty = headH - 6;
        const g = s('g', { attrs: { transform: `translate(${tx},${ty}) rotate(-40)` } });
        if (c.swatch) g.appendChild(s('rect', { attrs: { x: 0, y: -8, width: 8, height: 8, rx: 2 }, style: { fill: paint(c.swatch) } }));
        g.appendChild(s('text', { class: 'c-label', attrs: { x: c.swatch ? 12 : 0, y: 0 }, style: { fontSize: '12px' }, text: fitText(c.label ?? '', 84, 12) }));
        svg.appendChild(g);
      } else {
        const lw = textWidth(c.label ?? '', 12);
        const total = lw + (c.swatch ? 12 : 0);
        let x = Math.max(rowLabelW, Math.min(W - total - 1, cx - total / 2));
        if (c.swatch) {
          svg.appendChild(s('rect', { attrs: { x, y: headH - 14, width: 8, height: 8, rx: 2 }, style: { fill: paint(c.swatch) } }));
          x += 12;
        }
        svg.appendChild(s('text', { class: 'c-label', attrs: { x, y: headH - 6 }, style: { fontSize: '12px' }, text: c.label ?? '' }));
      }
    });
    const marks = [];
    rows.forEach((r, i) => {
      const y = headH + i * cell;
      const lx = r.swatch ? 16 : 0;
      if (r.swatch) svg.appendChild(s('rect', { attrs: { x: 0, y: y + cell / 2 - 5, width: 10, height: 10, rx: 3 }, style: { fill: paint(r.swatch) } }));
      svg.appendChild(s('text', { class: 'c-label', attrs: { x: lx, y: y + cell / 2 + 4.5 }, text: fitText(r.label ?? '', rowLabelW - lx - 8, 13) }));
      cols.forEach((c, j) => {
        const x = rowLabelW + j * cell;
        const blank = blankDiag && r.key === c.key;
        const v = values[i] ? values[i][j] : null;
        if (blank) {
          marks.push(null);
          return;
        }
        const fill = seqColor(finiteNum(v) ? v : null, max);
        const txt = finiteNum(v) ? format(v) : fmt.na;
        const title = typeof o.title === 'function' ? o.title(r, c) : `${r.label} → ${c.label}`;
        const g = markGroup(`${title}: ${txt}`);
        g.appendChild(s('rect', { class: 'c-fill', attrs: { x: x + 1, y: y + 1, width: cell - GAP, height: cell - GAP, rx: 2 }, style: { fill: fill } }));
        if (cell >= 30 && finiteNum(v)) {
          const label = txt;
          if (textWidth(label, 11, 600) <= cell - 8) {
            const resolved = col(fill);
            g.appendChild(s('text', {
              class: 'c-inlabel',
              attrs: { x: x + cell / 2, y: y + cell / 2 + 4, 'text-anchor': 'middle' },
              style: { fill: v > 0 ? textOn(resolved) : 'var(--muted)', fontWeight: v > 0 ? '600' : '400' },
              text: label,
            }));
          }
        }
        bindTip(g, () => ({ title, rows: [{ value: txt, label: o.valueLabel || '', color: fill, key: 'rect' }] }));
        svg.appendChild(g);
        marks.push(g);
      });
    });
    el.appendChild(svg);
    if (o.scale !== false && rows.length && cols.length) el.appendChild(scaleLegend({ min: 0, max, format, label: o.scaleLabel }));
    nav.set(marks, cols.length);
  }));
}

// line chart

/**
 * Multi-series line chart over time (session offsets).
 * opts: {span, series: [{key, label, color, t, v}], domain, refs: [{v, label}], format, height: 180,
 *        brush, onBrush, onHover, xFormat, cursor, marginLeft (align stacked charts), label}
 */
export function lineChart(el, opts = {}) {
  const st = { brush: null, geom: null, hoverT: null };
  let svgRef = null;
  let crossRef = null;
  let dotsRef = null;
  let brushRef = null;
  let plotH = 0;
  let current = opts;
  let Yref = null;

  const fmtX = (t) => (typeof current.xFormat === 'function' ? current.xFormat(t) : fmt.clockSpan((current.span || [0, 0])[1])(t));

  const m = mountChart(el, 'c-linechart', opts, (W, o) => {
    current = o;
    const series = (o.series || []).filter(Boolean);
    const H = o.height || 180;
    const format = o.format || autoFormat;
    const span = o.span || autoSpan(series);
    let lo;
    let hi;
    let yt;
    if (o.domain && finiteNum(o.domain[0]) && finiteNum(o.domain[1])) {
      [lo, hi] = o.domain;
      yt = niceScale(lo, hi, 4)[2].filter((v) => v >= lo - 1e-9 && v <= hi + 1e-9);
    } else {
      let mn = Infinity;
      let mx = -Infinity;
      for (const se of series) for (const v of se.v || []) if (finiteNum(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
      for (const r of o.refs || []) if (finiteNum(r.v)) { mn = Math.min(mn, r.v); mx = Math.max(mx, r.v); }
      if (!Number.isFinite(mn)) { mn = 0; mx = 1; }
      [lo, hi, yt] = niceScale(Math.min(0, mn), mx, 4);
    }
    const left = Math.max(o.marginLeft || 0, Math.ceil(Math.max(...yt.map((v) => textWidth(format(v), 11)))) + 10);
    const top = 8;
    const bottom = 22;
    const right = 8;
    const plotW = Math.max(40, W - left - right);
    plotH = H - top - bottom;
    const X = (t) => left + ((t - span[0]) / (span[1] - span[0] || 1)) * plotW;
    const Y = (v) => top + plotH - ((v - lo) / (hi - lo || 1)) * plotH;
    Yref = Y;
    st.geom = { left, width: plotW, v0: span[0], v1: span[1], top, X, Y, series, format };
    clear(el);
    const svg = svgRoot(W, H, null);
    svg.setAttribute('tabindex', '0');
    svg.setAttribute('role', 'group');
    svg.setAttribute('aria-roledescription', 'line chart');
    svg.setAttribute('aria-label', `${o.label || 'Line chart'}. Arrow keys move the readout${o.brush ? ', Shift with arrow keys selects a range, Escape clears it' : ''}.`);
    for (const v of yt) {
      const y = Math.round(Y(v)) + 0.5;
      svg.appendChild(s('line', { class: v === 0 ? 'c-axis' : 'c-grid', attrs: { x1: left, x2: left + plotW, y1: y, y2: y } }));
      svg.appendChild(s('text', { class: 'c-tick', attrs: { x: left - 6, y: y + 4, 'text-anchor': 'end' }, text: format(v) }));
    }
    for (const t of timeTicks(span[0], span[1], plotW)) {
      const x = X(t);
      const anchor = x < left + 20 ? 'start' : x > left + plotW - 20 ? 'end' : 'middle';
      svg.appendChild(s('text', { class: 'c-tick', attrs: { x, y: H - 6, 'text-anchor': anchor }, text: fmtX(t) }));
    }
    for (const r of o.refs || []) {
      if (!finiteNum(r.v) || r.v < lo || r.v > hi) continue;
      const y = Math.round(Y(r.v)) + 0.5;
      svg.appendChild(s('line', { class: 'c-ref', attrs: { x1: left, x2: left + plotW, y1: y, y2: y } }));
      svg.appendChild(s('text', { class: 'c-ref-label', attrs: { x: left + plotW - 2, y: y - 4, 'text-anchor': 'end' }, text: r.label ?? format(r.v) }));
    }
    const clipId = `clip-${Math.random().toString(36).slice(2, 8)}`;
    svg.appendChild(s('defs', {}, s('clipPath', { attrs: { id: clipId } }, s('rect', { attrs: { x: left, y: top - 2, width: plotW, height: plotH + 4 } }))));
    const lines = s('g', { attrs: { 'clip-path': `url(#${clipId})` } });
    for (const se of series) {
      const d = linePathD(se, span, X, Y, plotW);
      if (d) lines.appendChild(s('path', { class: 'c-line', attrs: { d }, style: { stroke: se.color || 'var(--ink-2)' } }));
    }
    svg.appendChild(lines);
    brushRef = s('g');
    svg.appendChild(brushRef);
    if (finiteNum(o.cursor) && o.cursor >= span[0] && o.cursor <= span[1]) {
      const x = Math.round(X(o.cursor)) + 0.25;
      svg.appendChild(s('line', { attrs: { x1: x, x2: x, y1: top, y2: top + plotH }, style: { stroke: 'var(--ink)', strokeWidth: 1.5 } }));
    }
    crossRef = s('line', { class: 'c-crosshair', attrs: { x1: 0, x2: 0, y1: top, y2: top + plotH, visibility: 'hidden' } });
    dotsRef = s('g', { attrs: { 'pointer-events': 'none' } });
    svg.append(crossRef, dotsRef);
    el.appendChild(svg);
    svgRef = svg;
    drawLineBrush(st.brush);
  });

  function drawLineBrush(range) {
    if (!brushRef) return;
    clear(brushRef);
    const g = st.geom;
    if (!g || !range) return;
    const k = g.width / (g.v1 - g.v0 || 1);
    const nodes = brushNodes({ left: g.left, width: g.width }, range, plotH, (t) => g.left + (t - g.v0) * k);
    const wrap = s('g', { attrs: { transform: `translate(0,${g.top})` } }, nodes);
    brushRef.appendChild(wrap);
  }

  function hover(t, evt) {
    const g = st.geom;
    if (!crossRef || !dotsRef) return;
    clear(dotsRef);
    if (t == null || !g) {
      crossRef.setAttribute('visibility', 'hidden');
      tooltip.hide();
      if (typeof current.onHover === 'function') current.onHover(null);
      return;
    }
    const grid = (g.series[0] && g.series[0].t) || [];
    let tt = t;
    const gi = nearestAt(grid, t, Infinity);
    if (gi >= 0) tt = grid[gi];
    const x = Math.round(g.X(tt)) + 0.5;
    crossRef.setAttribute('x1', x);
    crossRef.setAttribute('x2', x);
    crossRef.setAttribute('visibility', 'visible');
    const rows = [];
    for (const se of g.series) {
      const tol = medianStep(se.t) * 0.75;
      const i = nearestAt(se.t, tt, tol);
      const v = i < 0 ? null : se.v[i];
      rows.push({ value: finiteNum(v) ? g.format(v) : fmt.na, label: se.label, color: se.color, key: 'line' });
      if (finiteNum(v)) dotsRef.appendChild(s('circle', { class: 'c-dot', attrs: { cx: g.X(se.t[i]), cy: Yref(v), r: 4 }, style: { fill: se.color || 'var(--ink-2)' } }));
    }
    const content = { title: fmtX(tt), rows };
    if (evt) tooltip.show(evt, content);
    else {
      const rect = svgRef.getBoundingClientRect();
      tooltip.show({ x: rect.left + x, y: rect.top + g.top }, content);
    }
    if (typeof current.onHover === 'function') current.onHover(tt);
  }

  // listeners live on the container, so they survive re-renders of the svg
  xInteraction(el, {
    geom: () => st.geom,
    brushOn: () => !!current.brush,
    brush: () => st.brush,
    drawBrush: drawLineBrush,
    commit: (r) => {
      st.brush = r;
      drawLineBrush(r);
      if (typeof current.onBrush === 'function') current.onBrush(r ? [r[0], r[1]] : null);
    },
    hover,
    cursor: () => current.cursor,
    step: () => medianStep(((st.geom && st.geom.series[0]) || {}).t),
  });

  return {
    update: m.update,
    setBrush(range) {
      st.brush = range ? [Math.min(range[0], range[1]), Math.max(range[0], range[1])] : null;
      drawLineBrush(st.brush);
    },
    setCursor(t) {
      m.update({ cursor: t });
    },
    destroy: m.destroy,
  };
}

function autoSpan(series) {
  let a = Infinity;
  let b = -Infinity;
  for (const se of series) {
    const t = se.t || [];
    if (t.length) {
      a = Math.min(a, t[0]);
      b = Math.max(b, t[t.length - 1]);
    }
  }
  return Number.isFinite(a) && b > a ? [a, b] : [0, 1];
}

function linePathD(se, span, X, Y, plotW) {
  const t = se.t || [];
  const v = se.v || [];
  if (!t.length) return '';
  const step = medianStep(t);
  const dense = t.length > plotW * 2;
  let d = '';
  let open = false;
  let lastT = null;
  let binPx = null;
  let bmin = 0;
  let bmax = 0;
  let bfirst = 0;
  let blast = 0;
  const p = (x, y) => `${x.toFixed(1)},${y.toFixed(1)}`;
  const flushBin = () => {
    if (binPx == null) return;
    const x = binPx + 0.5;
    d += `${open ? 'L' : 'M'}${p(x, Y(bfirst))}`;
    open = true;
    if (bmin !== bfirst || bmax !== bfirst) d += `L${p(x, Y(bmin))}L${p(x, Y(bmax))}`;
    d += `L${p(x, Y(blast))}`;
    binPx = null;
  };
  for (let i = 0; i < t.length; i += 1) {
    const val = v[i];
    const tt = t[i];
    if (tt < span[0] - step || tt > span[1] + step) continue;
    if (!finiteNum(val) || (lastT != null && tt - lastT > step * 3)) {
      flushBin();
      open = false;
      if (!finiteNum(val)) {
        lastT = null;
        continue;
      }
    }
    lastT = tt;
    const x = X(tt);
    if (dense) {
      const px = Math.floor(x);
      if (binPx != null && px !== binPx) flushBin();
      if (binPx == null) {
        binPx = px;
        bmin = val;
        bmax = val;
        bfirst = val;
      }
      bmin = Math.min(bmin, val);
      bmax = Math.max(bmax, val);
      blast = val;
    } else {
      d += `${open ? 'L' : 'M'}${p(x, Y(val))}`;
      open = true;
    }
  }
  flushBin();
  return d;
}

// histogram

/**
 * Columns for binned counts. counts has edges.length + 1 entries (below the first edge, between
 * edges, at or above the last edge) or edges.length - 1 (closed bins). Edge values label the
 * boundaries. opts: {edges, counts, format (edge values), countFormat, color, height: 140, xLabel, countLabel, label}
 */
export function histogram(el, opts = {}) {
  const nav = rovingNav(el);
  return publicHandle(mountChart(el, 'c-hist', opts, (W, o) => {
    const edges = o.edges || [];
    const counts = o.counts || [];
    const ef = o.format || trimNum;
    const cf = o.countFormat || ((v) => fmt.int(v));
    const color = o.color || 'var(--seq-4)';
    const open = counts.length === edges.length + 1;
    const bins = counts.map((c, i) => {
      let a;
      let b;
      if (open) {
        a = i === 0 ? null : edges[i - 1];
        b = i === edges.length ? null : edges[i];
      } else {
        a = edges[i];
        b = edges[i + 1];
      }
      const label = a == null ? `under ${ef(b)}` : b == null ? `${ef(a)} or more` : `${ef(a)} to ${ef(b)}`;
      return { c: finiteNum(c) ? c : 0, a, b, label };
    });
    const H = o.height || 140;
    const top = 16;
    const bottom = o.xLabel ? 34 : 20;
    const plotH = H - top - bottom;
    const n = Math.max(1, bins.length);
    const slot = W / n;
    const colW = Math.min(BAR_MAX, Math.max(2, slot - GAP));
    const max = Math.max(1, ...bins.map((b) => b.c));
    const Y = (v) => top + plotH - (v / max) * plotH;
    clear(el);
    const svg = svgRoot(W, H, o.label);
    svg.appendChild(s('line', { class: 'c-axis', attrs: { x1: 0, x2: W, y1: top + plotH + 0.5, y2: top + plotH + 0.5 } }));
    const fitsAll = bins.every((b) => textWidth(cf(b.c), 11, 600) <= slot - 2);
    const maxIdx = bins.reduce((bi, b, i) => (b.c > bins[bi].c ? i : bi), 0);
    const marks = [];
    bins.forEach((bn, i) => {
      const cx = i * slot + slot / 2;
      const hgt = Math.max(bn.c > 0 ? 1 : 0, (bn.c / max) * plotH);
      const g = markGroup(`${bn.label}: ${cf(bn.c)}`);
      g.appendChild(s('rect', { class: 'c-hit', attrs: { x: i * slot, y: top, width: slot, height: plotH } }));
      if (hgt > 0) g.appendChild(s('path', { class: 'c-fill', attrs: { d: barPath(cx - colW / 2, Y(bn.c), colW, hgt, Math.min(RADIUS, colW / 2), 'top') }, style: { fill: paint(color) } }));
      if (bn.c > 0 && (fitsAll || i === maxIdx)) {
        g.appendChild(s('text', { class: 'c-value', attrs: { x: cx, y: Y(bn.c) - 4, 'text-anchor': 'middle' }, style: { fontSize: '11px' }, text: cf(bn.c) }));
      }
      bindTip(g, () => ({ title: bn.label, rows: [{ value: cf(bn.c), label: o.countLabel || 'Count', color, key: 'rect' }] }));
      svg.appendChild(g);
      marks.push(g);
    });
    // boundary labels between columns, thinned when they would collide
    const bounds = open ? edges.map((e, i) => ({ x: (i + 1) * slot, v: e })) : edges.map((e, i) => ({ x: i * slot, v: e }));
    let lastEnd = -Infinity;
    for (const b of bounds) {
      const txt = ef(b.v);
      const w = textWidth(txt, 11);
      const x0 = Math.max(0, Math.min(W - w, b.x - w / 2));
      if (x0 < lastEnd + 4) continue;
      lastEnd = x0 + w;
      svg.appendChild(s('text', { class: 'c-tick', attrs: { x: x0, y: top + plotH + 14 }, text: txt }));
    }
    if (o.xLabel) svg.appendChild(s('text', { class: 'c-note', attrs: { x: W / 2, y: H - 3, 'text-anchor': 'middle' }, style: { fontSize: '11px' }, text: o.xLabel }));
    el.appendChild(svg);
    nav.set(marks);
  }));
}

// network graph

const NET_LABEL_SIZE = 13;
const NET_LABEL_H = 16;
const NET_BEND_MAX = 26;
const NET_CLOSE = 40;
const NET_DIRS = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => ({ x: Math.cos((i * Math.PI) / 4), y: Math.sin((i * Math.PI) / 4) }));

/** where a node label sits when it is put beside the node in direction dir (a unit vector, y down). */
function netLabelSpot(p, r, dir, w) {
  const gap = r + 5;
  let anchor = 'middle';
  let x = p.x;
  let left = p.x - w / 2;
  if (dir.x > 0.38) {
    anchor = 'start';
    x = p.x + dir.x * gap;
    left = x;
  } else if (dir.x < -0.38) {
    anchor = 'end';
    x = p.x + dir.x * gap;
    left = x - w;
  }
  let top = p.y - NET_LABEL_H / 2;
  if (dir.y > 0.38) top = p.y + dir.y * gap - 2;
  else if (dir.y < -0.38) top = p.y + dir.y * gap - NET_LABEL_H + 2;
  return { x, y: top + 12, anchor, box: { x0: left, y0: top, x1: left + w, y1: top + NET_LABEL_H } };
}

function boxOverlap(a, b) {
  const w = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0);
  const hh = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
  return w > 0 && hh > 0 ? w * hh : 0;
}

/**
 * Directed weighted graph (who looks at whom, facing). layout 'circle' or 'given' (node x, y in data
 * units, y up, equal aspect): given positions are fitted into the box with room for the labels, nodes
 * that would touch are nudged apart, and nodes without a position stand in a column on the right.
 * Mutual edges draw as one double-headed arrow; a pair with two separate directions draws two arrows
 * curved apart (the bend grows with the distance, capped, so arcs stay inside the chart). Edge width
 * runs from 2 to 6 px with the weight. Labels go on the side away from the other nodes.
 * opts: {nodes: [{key, label, color, x, y, weight}], edges: [{from, to, weight, label, mutual, color}],
 *        layout, height: 240, format (edge weight), weightLabel, outLabel, inLabel, nodeWeightLabel, label}
 */
export function networkGraph(el, opts = {}) {
  const nav = rovingNav(el);
  return publicHandle(mountChart(el, 'c-network', opts, (W, o) => {
    const nodes = (o.nodes || []).filter((nd) => nd && nd.key != null);
    const H = o.height || 240;
    const format = o.format || autoFormat;
    const labelOf = (nd) => String(nd.label ?? nd.key);
    const maxNodeW = Math.max(0, ...nodes.map((nd) => (finiteNum(nd.weight) ? nd.weight : 0)));
    const radius = new Map(nodes.map((nd) => [nd.key, maxNodeW > 0 && finiteNum(nd.weight) ? 7 + 8 * Math.sqrt(nd.weight / maxNodeW) : 10]));
    const rMax = Math.max(10, ...radius.values());
    const labelW = new Map(nodes.map((nd) => [nd.key, textWidth(labelOf(nd), NET_LABEL_SIZE)]));
    const maxLabelW = Math.max(0, ...labelW.values());
    // room for a label beside (x) or above / below (y) a node at the edge of the box
    const padX = Math.min(W / 3, rMax + 9 + maxLabelW);
    const padY = Math.min(H / 3, rMax + 7 + NET_LABEL_H);
    const pos = new Map();
    const given = o.layout === 'given' && nodes.some((nd) => finiteNum(nd.x) && finiteNum(nd.y));
    let box = { x0: padX, x1: Math.max(padX + 1, W - padX), y0: padY, y1: Math.max(padY + 1, H - padY) };
    if (given) {
      const pts = nodes.filter((nd) => finiteNum(nd.x) && finiteNum(nd.y));
      const missing = nodes.filter((nd) => !(finiteNum(nd.x) && finiteNum(nd.y)));
      if (missing.length) {
        // a column on the right for nodes without a position (e.g. 'Others')
        const colX = W - padX;
        box = { ...box, x1: Math.max(box.x0 + 1, colX - 2 * rMax - 28) };
        const stepY = 2 * rMax + 22;
        missing.forEach((nd, i) => {
          const y = H / 2 + (i - (missing.length - 1) / 2) * stepY;
          pos.set(nd.key, { x: colX, y: Math.max(padY, Math.min(H - padY, y)) });
        });
      }
      const x0 = Math.min(...pts.map((p) => p.x));
      const x1 = Math.max(...pts.map((p) => p.x));
      const y0 = Math.min(...pts.map((p) => p.y));
      const y1 = Math.max(...pts.map((p) => p.y));
      const bw = box.x1 - box.x0;
      const bh = box.y1 - box.y0;
      const kx = x1 - x0 > 1e-9 ? bw / (x1 - x0) : Infinity;
      const ky = y1 - y0 > 1e-9 ? bh / (y1 - y0) : Infinity;
      const k = Number.isFinite(Math.min(kx, ky)) ? Math.min(kx, ky) : 0;
      const ox = box.x0 + (bw - (x1 - x0) * k) / 2;
      const oy = box.y0 + (bh - (y1 - y0) * k) / 2;
      for (const nd of pts) pos.set(nd.key, { x: ox + (nd.x - x0) * k, y: oy + (y1 - nd.y) * k });
    } else {
      const R = Math.max(30, Math.min((box.x1 - box.x0) / 2, (box.y1 - box.y0) / 2));
      const cx = W / 2;
      const cy = H / 2;
      nodes.forEach((nd, i) => {
        const a = -Math.PI / 2 + (i / Math.max(1, nodes.length)) * Math.PI * 2;
        pos.set(nd.key, { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) });
      });
    }
    // nudge apart nodes that would touch (two badges at about the same median position)
    const clampPos = (p) => {
      p.x = Math.max(rMax + 2, Math.min(W - rMax - 2, p.x));
      p.y = Math.max(rMax + 2, Math.min(H - rMax - 2, p.y));
    };
    for (let it = 0; it < 40; it += 1) {
      let moved = false;
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const a = pos.get(nodes[i].key);
          const b = pos.get(nodes[j].key);
          const need = radius.get(nodes[i].key) + radius.get(nodes[j].key) + 20;
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let d = Math.hypot(dx, dy);
          if (d >= need - 0.01) continue;
          if (d < 1e-6) {
            const ang = (i + j) * 2.39996;
            dx = Math.cos(ang);
            dy = Math.sin(ang);
            d = 1;
          }
          const push = (need - Math.hypot(b.x - a.x, b.y - a.y)) / 2;
          a.x -= (dx / d) * push;
          a.y -= (dy / d) * push;
          b.x += (dx / d) * push;
          b.y += (dy / d) * push;
          clampPos(a);
          clampPos(b);
          moved = true;
        }
      }
      if (!moved) break;
    }
    const byKey = new Map(nodes.map((nd) => [nd.key, nd]));
    const cen = { x: 0, y: 0 };
    for (const nd of nodes) {
      cen.x += pos.get(nd.key).x / Math.max(1, nodes.length);
      cen.y += pos.get(nd.key).y / Math.max(1, nodes.length);
    }
    if (nodes.length < 2) {
      cen.x = W / 2;
      cen.y = H;
    }
    const edges = (o.edges || []).filter((e) => e && e.from !== e.to && pos.has(e.from) && pos.has(e.to) && finiteNum(e.weight) && e.weight > 0);
    const maxW = Math.max(1e-9, ...edges.map((e) => e.weight));
    const dirSet = new Set(edges.map((e) => `${e.from}\u0000${e.to}`));
    clear(el);
    const svg = svgRoot(W, H, o.label);
    const edgeLayer = s('g');
    const nodeLayer = s('g');
    svg.append(edgeLayer, nodeLayer);
    const marks = [];
    const edgePts = [];
    const sorted = edges.slice().sort((a, b) => a.weight - b.weight);
    for (const e of sorted) {
      const A = pos.get(e.from);
      const B = pos.get(e.to);
      const ra = radius.get(e.from) + 3;
      const rb = radius.get(e.to) + 3;
      const reverse = !e.mutual && dirSet.has(`${e.to}\u0000${e.from}`);
      const sw = Math.max(2, Math.min(6, 2 + 4 * (e.weight / maxW)));
      const color = e.color || (byKey.get(e.from) || {}).color || 'var(--ink-2)';
      const dx = B.x - A.x;
      const dy = B.y - A.y;
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;
      // two one-way arrows bend apart, in proportion to the chord and capped
      const bend = reverse ? Math.min(NET_BEND_MAX, len * 0.2) : 0;
      const C = {
        x: Math.max(0, Math.min(W, (A.x + B.x) / 2 + nx * bend)),
        y: Math.max(0, Math.min(H, (A.y + B.y) / 2 + ny * bend)),
      };
      const dirA = norm(C.x - A.x, C.y - A.y);
      const dirB = norm(B.x - C.x, B.y - C.y);
      // short edges between close nodes get smaller heads so the shaft stays visible
      const avail = Math.max(0, len - ra - rb);
      const heads = e.mutual ? 2 : 1;
      const head = Math.max(3, Math.min(4 + sw, (avail - 2) / (heads + 0.5)));
      const S = { x: A.x + dirA.x * (ra + (e.mutual ? head * 0.9 : 0)), y: A.y + dirA.y * (ra + (e.mutual ? head * 0.9 : 0)) };
      const tipB = { x: B.x - dirB.x * rb, y: B.y - dirB.y * rb };
      const E = { x: tipB.x - dirB.x * head * 0.9, y: tipB.y - dirB.y * head * 0.9 };
      const d = bend ? `M${S.x},${S.y}Q${C.x},${C.y} ${E.x},${E.y}` : `M${S.x},${S.y}L${E.x},${E.y}`;
      for (let i = 1; i < 8; i += 1) {
        const t = i / 8;
        const q = 1 - t;
        edgePts.push(bend
          ? { x: q * q * S.x + 2 * q * t * C.x + t * t * E.x, y: q * q * S.y + 2 * q * t * C.y + t * t * E.y }
          : { x: S.x + (E.x - S.x) * t, y: S.y + (E.y - S.y) * t });
      }
      const fromL = (byKey.get(e.from) || {}).label || e.from;
      const toL = (byKey.get(e.to) || {}).label || e.to;
      const txt = e.label != null ? String(e.label) : format(e.weight);
      const title = e.mutual ? `${fromL} ↔ ${toL}` : `${fromL} → ${toL}`;
      const g = markGroup(`${title}: ${txt}`);
      g.appendChild(s('path', { class: 'c-hit', attrs: { d }, style: { stroke: 'transparent', strokeWidth: Math.max(14, sw + 10), fill: 'none' } }));
      g.appendChild(s('path', { class: 'c-fill', attrs: { d }, style: { stroke: color, strokeWidth: sw, fill: 'none', strokeLinecap: 'round', strokeOpacity: 0.8 } }));
      g.appendChild(arrowHead(tipB, dirB, head, color));
      if (e.mutual) {
        const tipA = { x: A.x + dirA.x * ra, y: A.y + dirA.y * ra };
        g.appendChild(arrowHead(tipA, { x: -dirA.x, y: -dirA.y }, head, color));
      }
      bindTip(g, () => ({
        title,
        rows: [{ value: txt, label: o.weightLabel || '', color, key: 'line' }],
        note: e.mutual ? 'Both directions' : null,
      }));
      edgeLayer.appendChild(g);
      marks.push(g);
    }
    // labels: away from the graph's centre, and for nodes closer than NET_CLOSE px away from their
    // local cluster, avoiding other labels, nodes, arcs and the chart edge
    const nodeBoxes = nodes.map((nd) => {
      const p = pos.get(nd.key);
      const r = radius.get(nd.key);
      return { x0: p.x - r, y0: p.y - r, x1: p.x + r, y1: p.y + r };
    });
    const close = new Map(nodes.map((nd) => {
      const p = pos.get(nd.key);
      return [nd.key, nodes.filter((q) => q !== nd && Math.hypot(pos.get(q.key).x - p.x, pos.get(q.key).y - p.y) < NET_CLOSE)];
    }));
    const order = nodes.slice().sort((a, b) => close.get(b.key).length - close.get(a.key).length);
    const placed = [];
    const spots = new Map();
    for (const nd of order) {
      const p = pos.get(nd.key);
      const r = radius.get(nd.key);
      const w = labelW.get(nd.key);
      let pref = Math.hypot(p.x - cen.x, p.y - cen.y) > 1 ? norm(p.x - cen.x, p.y - cen.y) : { x: 0, y: -1 };
      const near = close.get(nd.key);
      if (near.length) {
        const lc = { x: p.x, y: p.y };
        for (const q of near) {
          lc.x += pos.get(q.key).x;
          lc.y += pos.get(q.key).y;
        }
        lc.x /= near.length + 1;
        lc.y /= near.length + 1;
        if (Math.hypot(p.x - lc.x, p.y - lc.y) > 0.5) {
          const away = norm(p.x - lc.x, p.y - lc.y);
          pref = norm(away.x * 1.5 + pref.x * 0.5, away.y * 1.5 + pref.y * 0.5);
        }
      }
      let best = null;
      for (const dir of NET_DIRS) {
        const spot = netLabelSpot(p, r, dir, w);
        const b = spot.box;
        let score = 1 - (dir.x * pref.x + dir.y * pref.y);
        for (const pb of placed) score += boxOverlap(b, pb) / 12;
        nodeBoxes.forEach((nb) => {
          score += boxOverlap(b, nb) / 12;
        });
        const inside = boxOverlap(b, { x0: 1, y0: 1, x1: W - 1, y1: H - 1 });
        score += ((b.x1 - b.x0) * (b.y1 - b.y0) - inside) / 4;
        for (const ep of edgePts) if (ep.x > b.x0 && ep.x < b.x1 && ep.y > b.y0 && ep.y < b.y1) score += 0.6;
        if (!best || score < best.score) best = { score, spot };
      }
      placed.push(best.spot.box);
      spots.set(nd.key, best.spot);
    }
    for (const nd of nodes) {
      const p = pos.get(nd.key);
      const r = radius.get(nd.key);
      const out = edges.filter((e) => e.from === nd.key).reduce((a, e) => a + e.weight, 0);
      const inc = edges.filter((e) => e.to === nd.key).reduce((a, e) => a + e.weight, 0);
      const g = markGroup(`${labelOf(nd)}`);
      g.appendChild(s('circle', { class: 'c-hit', attrs: { cx: p.x, cy: p.y, r: Math.max(12, r + 4) } }));
      g.appendChild(s('circle', { class: 'c-fill c-dot', attrs: { cx: p.x, cy: p.y, r }, style: { fill: paint(nd.color || 'var(--tag-other)') } }));
      const spot = spots.get(nd.key);
      g.appendChild(s('text', { class: 'c-label c-halo', attrs: { x: spot.x, y: spot.y, 'text-anchor': spot.anchor }, text: labelOf(nd) }));
      bindTip(g, () => ({
        title: labelOf(nd),
        rows: [
          { value: format(out), label: o.outLabel || 'Outgoing', color: nd.color, key: 'dot' },
          { value: format(inc), label: o.inLabel || 'Incoming', color: nd.color, key: 'dot' },
          finiteNum(nd.weight) ? { value: format(nd.weight), label: o.nodeWeightLabel || 'Weight', key: 'none' } : null,
        ].filter(Boolean),
      }));
      nodeLayer.appendChild(g);
      marks.push(g);
    }
    el.appendChild(svg);
    nav.set(marks);
  }));
}

function norm(x, y) {
  const l = Math.hypot(x, y) || 1;
  return { x: x / l, y: y / l };
}

function arrowHead(tip, dir, size, color) {
  const bx = tip.x - dir.x * size;
  const by = tip.y - dir.y * size;
  const px = -dir.y * size * 0.55;
  const py = dir.x * size * 0.55;
  return s('path', {
    class: 'c-fill',
    attrs: { d: `M${tip.x},${tip.y}L${bx + px},${by + py}L${bx - px},${by - py}Z` },
    style: { fill: color, fillOpacity: 0.9 },
  });
}

// sparkline, stat tile, bullet

/** tiny trend line; null values break it. opts: {values, color, height: 24, max, min} */
export function sparkline(el, opts = {}) {
  return publicHandle(mountChart(el, 'c-spark', opts, (W, o) => {
    const values = o.values || [];
    const H = o.height || 24;
    const nums = values.filter(finiteNum);
    clear(el);
    const svg = s('svg', { attrs: { width: W, height: H, viewBox: `0 0 ${W} ${H}`, 'aria-hidden': 'true' } });
    if (nums.length) {
      const max = finiteNum(o.max) ? o.max : Math.max(...nums);
      const min = finiteNum(o.min) ? o.min : Math.min(0, ...nums);
      const span = max - min || 1;
      const n = values.length;
      const X = (i) => (n <= 1 ? W / 2 : 2 + (i / (n - 1)) * (W - 4));
      const Y = (v) => 3 + (1 - (Math.min(max, Math.max(min, v)) - min) / span) * (H - 6);
      let d = '';
      let open = false;
      let last = -1;
      values.forEach((v, i) => {
        if (!finiteNum(v)) {
          open = false;
          return;
        }
        d += `${open ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`;
        open = true;
        last = i;
      });
      const color = o.color || 'var(--ink-2)';
      svg.appendChild(s('path', { attrs: { d }, style: { fill: 'none', stroke: color, strokeWidth: 1.5, strokeLinejoin: 'round', strokeLinecap: 'round' } }));
      if (last >= 0) svg.appendChild(s('circle', { attrs: { cx: X(last), cy: Y(values[last]), r: 2.5 }, style: { fill: color } }));
    }
    el.appendChild(svg);
  }));
}

/**
 * KPI figure: label, value (proportional figures), unit, note, optional sparkline and status.
 * spark: {values, color, max} or an array; status: {state: 'good'|'warning'|'serious'|'critical', text}.
 * The element has update(props) and destroy().
 */
export function statTile(props = {}) {
  const labelEl = h('div', { class: 'stat-label' });
  const valueEl = h('div', { class: 'stat-value' });
  const noteEl = h('div', { class: 'stat-note' });
  const sparkEl = h('div', { class: 'stat-spark', hidden: true });
  const el = h('div', { class: 'stat' }, labelEl, valueEl, noteEl, sparkEl);
  let spark = null;
  let state = {};
  function render(p) {
    state = { ...state, ...p };
    const { label, value, unit, note, status, pending } = state;
    labelEl.textContent = label || '';
    labelEl.title = label || '';
    const na = value == null || value === '' || (typeof value === 'number' && !Number.isFinite(value));
    const valText = na ? fmt.na : typeof value === 'number' ? fmt.compact(value) : String(value);
    clear(valueEl);
    valueEl.classList.toggle('is-na', na && !pending);
    valueEl.appendChild(h('span', { text: pending ? String(pending) : valText }));
    if (!na && !pending && unit) valueEl.appendChild(h('span', { class: 'stat-unit', text: unit }));
    clear(noteEl);
    if (status && status.state) {
      const ic = status.state === 'good' ? 'check' : 'alert';
      noteEl.appendChild(h('span', { class: 'stat-status', dataset: { status: status.state } }, icon(ic, 12), h('span', { text: status.text || '' })));
    }
    if (note) noteEl.appendChild(h('span', { text: note }));
    el.classList.toggle('is-pending', !!pending);
    const sp = pending ? null : state.spark;
    if (sp) {
      const sv = Array.isArray(sp) ? { values: sp } : sp;
      sparkEl.hidden = false;
      // a detached tile draws its sparkline once the ResizeObserver sees a width
      if (spark) spark.update(sv);
      else spark = sparkline(sparkEl, sv);
    } else {
      sparkEl.hidden = true;
      if (spark) {
        spark.destroy();
        spark = null;
      }
    }
  }
  render(props);
  el.update = (p) => render(p || {});
  el.destroy = () => {
    if (spark) spark.destroy();
    el.remove();
  };
  return el;
}

/**
 * Value bars against a baseline tick (joint attention vs its baseline).
 * opts: {items: [{key, label, swatch, value, baseline, domain}], format (default pct), excess: true, label}
 */
export function bullet(el, opts = {}) {
  const nav = rovingNav(el);
  return publicHandle(mountChart(el, 'c-bullet', opts, (W, o) => {
    const items = (o.items || []).filter(Boolean);
    const format = o.format || ((v) => fmt.pct(v));
    const showExcess = o.excess !== false;
    const pitch = 30;
    const thick = 10;
    const texts = items.map((it) => {
      const v = finiteNum(it.value) ? format(it.value) : fmt.na;
      const b = finiteNum(it.baseline) ? `vs ${format(it.baseline)}` : '';
      const ex = showExcess && finiteNum(it.value) && finiteNum(it.baseline) ? fmt.pp(it.value - it.baseline) : '';
      return { v, b, ex };
    });
    const textW = Math.max(40, ...texts.map((t) => textWidth(t.v, 12, 600) + (t.b ? textWidth(t.b, 12) + 8 : 0) + (t.ex ? textWidth(t.ex, 12) + 8 : 0))) + 18;
    const hasSw = items.some((it) => it.swatch);
    const maxLabel = Math.max(0, ...items.map((it) => textWidth(it.label ?? '', 13)));
    let labelW = Math.min(maxLabel + (hasSw ? 24 : 10), Math.round(W * 0.4));
    if (W - labelW - textW < 60) labelW = Math.max(50, W - textW - 60);
    const trackW = Math.max(30, W - labelW - textW);
    const H = Math.max(pitch, items.length * pitch);
    clear(el);
    const svg = svgRoot(W, H, o.label);
    const marks = [];
    items.forEach((it, i) => {
      const cy = i * pitch + pitch / 2;
      const dom = it.domain || o.domain || [0, 1];
      const X = (v) => labelW + ((Math.min(dom[1], Math.max(dom[0], v)) - dom[0]) / (dom[1] - dom[0] || 1)) * trackW;
      const color = it.swatch || 'var(--ink-2)';
      const t = texts[i];
      const g = markGroup(`${it.label ?? ''}: ${t.v} ${t.b} ${t.ex}`.trim());
      g.appendChild(s('rect', { class: 'c-hit', attrs: { x: 0, y: cy - pitch / 2, width: W, height: pitch } }));
      const lx = hasSw ? 16 : 0;
      if (it.swatch) g.appendChild(s('rect', { attrs: { x: 0, y: cy - 5, width: 10, height: 10, rx: 3 }, style: { fill: paint(it.swatch) } }));
      g.appendChild(s('text', { class: 'c-label', attrs: { x: lx, y: cy + 4.5 }, text: fitText(it.label ?? '', labelW - lx - 8, 13) }));
      g.appendChild(s('rect', { attrs: { x: labelW, y: cy - 3, width: trackW, height: 6, rx: 3 }, style: { fill: 'var(--surface-2)' } }));
      if (finiteNum(it.value)) {
        const w = Math.max(1, X(it.value) - labelW);
        g.appendChild(s('path', { class: 'c-fill', attrs: { d: barPath(labelW, cy - thick / 2, w, thick) }, style: { fill: paint(color) } }));
      }
      if (finiteNum(it.baseline)) {
        const bx = X(it.baseline);
        g.appendChild(s('line', { attrs: { x1: bx, x2: bx, y1: cy - 8, y2: cy + 8 }, style: { stroke: 'var(--ink)', strokeWidth: 2, strokeLinecap: 'round' } }));
      }
      const text = s('text', { class: 'c-value', attrs: { x: labelW + trackW + 10, y: cy + 4 } }, t.v);
      if (t.b) text.appendChild(s('tspan', { class: 'c-note', attrs: { dx: 6 }, text: t.b }));
      if (t.ex) text.appendChild(s('tspan', { class: 'c-note', attrs: { dx: 6 }, text: t.ex }));
      g.appendChild(text);
      bindTip(g, () => ({
        title: it.label ?? '',
        rows: [
          { value: t.v, label: o.valueLabel || 'Value', color, key: 'rect' },
          finiteNum(it.baseline) ? { value: format(it.baseline), label: o.baselineLabel || 'Baseline', color: 'var(--ink)', key: 'line' } : null,
          t.ex ? { value: t.ex, label: o.excessLabel || 'Above baseline', key: 'none' } : null,
        ].filter(Boolean),
      }));
      svg.appendChild(g);
      marks.push(g);
    });
    el.appendChild(svg);
    nav.set(marks);
  }));
}
