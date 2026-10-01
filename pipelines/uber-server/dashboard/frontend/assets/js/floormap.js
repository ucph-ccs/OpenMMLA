/**
 * Top-down room plan in floor metres for the badge positions (IPS). u runs to the right and v away
 * from the main camera (up on screen), drawn at equal aspect with a 1 m grid and a 1 m scale bar.
 * Three overlays share the plan: live badges (positions, heading arrows, short trails, dyad lines,
 * facing arrows), trails over a time range, and an occupancy heat map. Headings follow the backend
 * convention: 0 points along +v, pi/2 along +u.
 */

import { h, s, clear, fmt, tooltip, theme } from './core.js';
import { seqColor, scaleLegend } from './charts.js';

const BADGE_R = 6;
const ARROW_LEN = 18;
const TRAIL_GAP = 1.5;
const DEFAULT_EXTENT = { u: [-2, 2], v: [0, 4] };

function finite(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function validExtent(ex) {
  return ex && ex.u && ex.v && finite(ex.u[0]) && finite(ex.u[1]) && finite(ex.v[0]) && finite(ex.v[1])
    && ex.u[1] > ex.u[0] && ex.v[1] > ex.v[0];
}

function copyExtent(ex) {
  return { u: [ex.u[0], ex.u[1]], v: [ex.v[0], ex.v[1]] };
}

/** opacity for a badge whose newest position is `age` seconds old: 1 up to 2 s, .35 at 10 s, hidden after 30 s. */
function ageOpacity(age) {
  if (!finite(age) || age <= 2) return 1;
  if (age > 30) return 0;
  if (age >= 10) return 0.35;
  return 1 - ((age - 2) / 8) * 0.65;
}

function headingVec(hd) {
  return { x: Math.sin(hd), y: -Math.cos(hd) };
}

function arrowPath(x1, y1, x2, y2, size, both) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const px = -uy * size * 0.55;
  const py = ux * size * 0.55;
  const head = (tx, ty, dirx, diry) => {
    const bx = tx - dirx * size;
    const by = ty - diry * size;
    return `M${tx},${ty}L${bx + px},${by + py}L${bx - px},${by - py}Z`;
  };
  let d = head(x2, y2, ux, uy);
  if (both) d += head(x1, y1, -ux, -uy);
  return d;
}

/**
 * opts: {extent: {u: [a, b], v: [a, b]}, cameras: [{id, u, v, main, heading}], height: 420, label,
 *        onSelect(id|null)}. Without an extent the plan starts at 4 x 4 m and grows to fit the data.
 * Returns {setLive, setTrails, setHeat, setExtent, setCameras, clear, destroy}.
 */
export function floorMap(el, opts = {}) {
  clear(el);
  el.classList.add('floormap');
  const st = {
    extent: validExtent(opts.extent) ? copyExtent(opts.extent) : copyExtent(DEFAULT_EXTENT),
    auto: opts.autoExtent ?? !validExtent(opts.extent),
    cameras: opts.cameras || [],
    height: opts.height || 420,
    live: null,
    trails: null,
    heat: null,
    selected: null,
    hovered: null,
    g: null,
    width: 0,
    alive: true,
    raf: 0,
    trailPts: [],
  };
  const svg = s('svg', { attrs: { role: 'group', 'aria-label': opts.label || 'Room plan, top view in metres' } });
  const L = {
    heat: s('g', { attrs: { 'aria-hidden': 'true' } }),
    grid: s('g', { attrs: { 'aria-hidden': 'true' } }),
    cams: s('g'),
    trails: s('g', { attrs: { 'aria-hidden': 'true' } }),
    dyads: s('g', { attrs: { 'aria-hidden': 'true' } }),
    edges: s('g', { attrs: { 'aria-hidden': 'true' } }),
    badges: s('g'),
    hover: s('g', { attrs: { 'aria-hidden': 'true', 'pointer-events': 'none' } }),
  };
  svg.append(L.heat, L.grid, L.cams, L.trails, L.dyads, L.edges, L.badges, L.hover);
  const legendEl = h('div', { class: 'fm-legend', hidden: true });
  el.append(svg, legendEl);
  const badgeNodes = new Map();

  function geom(W) {
    const ex = st.extent;
    const pad = 26;
    const bottom = 30;
    const ur = ex.u[1] - ex.u[0];
    const vr = ex.v[1] - ex.v[0];
    // when the width limits the scale (phones), the map shrinks to the plan instead of keeping empty bands
    const H = Math.max(160, Math.min(st.height, Math.ceil(((W - 2 * pad) / ur) * vr + pad + bottom)));
    const k = Math.max(4, Math.min((W - 2 * pad) / ur, (H - pad - bottom) / vr));
    const ox = (W - ur * k) / 2;
    const oy = pad + (H - pad - bottom - vr * k) / 2;
    return {
      W, H, k, ox, oy,
      X: (u) => ox + (u - ex.u[0]) * k,
      Y: (v) => oy + (ex.v[1] - v) * k,
      U: (x) => ex.u[0] + (x - ox) / k,
      V: (y) => ex.v[1] - (y - oy) / k,
    };
  }

  function grow(points) {
    if (!st.auto) return false;
    const ex = st.extent;
    let changed = false;
    for (const [u, v] of points) {
      if (!finite(u) || !finite(v)) continue;
      if (u < ex.u[0] + 0.25) { ex.u[0] = Math.floor((u - 0.5) * 2) / 2; changed = true; }
      if (u > ex.u[1] - 0.25) { ex.u[1] = Math.ceil((u + 0.5) * 2) / 2; changed = true; }
      if (v < ex.v[0] + 0.25) { ex.v[0] = Math.floor((v - 0.5) * 2) / 2; changed = true; }
      if (v > ex.v[1] - 0.25) { ex.v[1] = Math.ceil((v + 0.5) * 2) / 2; changed = true; }
    }
    return changed;
  }

  function drawStatic(g) {
    clear(L.grid);
    clear(L.cams);
    const ex = st.extent;
    const top = g.Y(ex.v[1]);
    const bottom = g.Y(ex.v[0]);
    const left = g.X(ex.u[0]);
    const right = g.X(ex.u[1]);
    for (let u = Math.ceil(ex.u[0]); u <= ex.u[1] + 1e-9; u += 1) {
      const x = Math.round(g.X(u)) + 0.5;
      L.grid.appendChild(s('line', { class: 'fm-grid', attrs: { x1: x, x2: x, y1: top, y2: bottom } }));
    }
    for (let v = Math.ceil(ex.v[0]); v <= ex.v[1] + 1e-9; v += 1) {
      const y = Math.round(g.Y(v)) + 0.5;
      L.grid.appendChild(s('line', { class: 'fm-grid', attrs: { x1: left, x2: right, y1: y, y2: y } }));
    }
    L.grid.appendChild(s('rect', { class: 'fm-frame', attrs: { x: Math.round(left) + 0.5, y: Math.round(top) + 0.5, width: Math.round(right - left), height: Math.round(bottom - top) } }));
    // 1 m scale bar under the plan, left aligned
    const sx = Math.round(left) + 0.5;
    const sy = Math.round(Math.min(g.H - 12, bottom + 16)) + 0.5;
    const len = g.k;
    L.grid.appendChild(s('path', { class: 'fm-scale', attrs: { d: `M${sx},${sy - 4}V${sy}H${sx + len}V${sy - 4}` }, style: { fill: 'none' } }));
    L.grid.appendChild(s('text', { class: 'fm-scale-label', attrs: { x: sx + len + 6, y: sy + 3 }, text: '1 m' }));
    for (const cam of st.cameras || []) {
      if (!finite(cam.u) || !finite(cam.v)) continue;
      const x = g.X(cam.u);
      const y = g.Y(cam.v);
      const gg = s('g', { class: 'fm-camera', attrs: { role: 'img', 'aria-label': `Camera ${cam.id}${cam.main ? ', main' : ''}` } });
      if (finite(cam.heading)) {
        const d = headingVec(cam.heading);
        const tip = { x: x + d.x * 9, y: y + d.y * 9 };
        const back = { x: x - d.x * 5, y: y - d.y * 5 };
        const p = { x: -d.y * 6, y: d.x * 6 };
        gg.appendChild(s('path', { class: ['fm-cam', cam.main ? 'is-main' : null], attrs: { d: `M${tip.x},${tip.y}L${back.x + p.x},${back.y + p.y}L${back.x - p.x},${back.y - p.y}Z` } }));
      } else {
        gg.appendChild(s('rect', { class: ['fm-cam', cam.main ? 'is-main' : null], attrs: { x: x - 5, y: y - 5, width: 10, height: 10, rx: 2 } }));
      }
      const anchorRight = x > g.W - 70;
      gg.appendChild(s('text', { class: 'fm-cam-label', attrs: { x: anchorRight ? x - 10 : x + 10, y: y + 14, 'text-anchor': anchorRight ? 'end' : 'start' }, text: cam.id }));
      gg.addEventListener('pointermove', (e) => tooltip.show(e, {
        title: `Camera ${cam.id}`,
        rows: [
          { value: `${fmt.num(cam.u, 2)}, ${fmt.num(cam.v, 2)}`, label: 'u, v (m)', key: 'none' },
          cam.main ? { value: 'Main', label: 'reference frame', key: 'none' } : null,
        ].filter(Boolean),
      }));
      gg.addEventListener('pointerleave', () => tooltip.hide());
      L.cams.appendChild(gg);
    }
  }

  function drawHeat(g) {
    clear(L.heat);
    legendEl.hidden = true;
    clear(legendEl);
    const hm = st.heat;
    if (!hm || !hm.grid) return;
    const { u0, v0, cell, nu, nv, counts } = hm.grid;
    if (!finite(u0) || !finite(v0) || !(cell > 0) || !nu || !nv || !counts) return;
    let max = 0;
    for (const c of counts) if (finite(c) && c > max) max = c;
    st.heatMax = max;
    const x0 = g.X(u0);
    const yTop = g.Y(v0 + nv * cell);
    L.heat.appendChild(s('rect', { attrs: { x: x0, y: yTop, width: nu * cell * g.k, height: nv * cell * g.k }, style: { fill: 'var(--surface-2)' } }));
    const cw = cell * g.k;
    for (let iv = 0; iv < nv; iv += 1) {
      const y = g.Y(v0 + (iv + 1) * cell);
      let run = null;
      const flush = (end) => {
        if (!run) return;
        L.heat.appendChild(s('rect', { attrs: { x: g.X(u0 + run.start * cell), y, width: (end - run.start) * cw + 0.3, height: cw + 0.3 }, style: { fill: run.fill } }));
        run = null;
      };
      for (let iu = 0; iu < nu; iu += 1) {
        const c = counts[iv * nu + iu];
        const fill = finite(c) && c > 0 ? seqColor(c, max, 'sqrt') : null;
        if (run && fill === run.fill) continue;
        flush(iu);
        if (fill) run = { start: iu, fill };
      }
      flush(nu);
    }
    const format = hm.format || ((v) => fmt.duration(v));
    legendEl.appendChild(scaleLegend({ min: 0, max, format, label: hm.label || 'Time per cell' }));
    legendEl.hidden = false;
  }

  function trailSegments(tr, range) {
    const segs = [];
    const t = tr.t || [];
    const u = tr.u || [];
    const v = tr.v || [];
    let cur = null;
    let lastT = null;
    for (let i = 0; i < t.length; i += 1) {
      if (range && (t[i] < range[0] || t[i] > range[1])) {
        if (cur) segs.push(cur);
        cur = null;
        lastT = null;
        continue;
      }
      if (!finite(u[i]) || !finite(v[i])) continue;
      if (cur && lastT != null && t[i] - lastT > TRAIL_GAP) {
        segs.push(cur);
        cur = null;
      }
      if (!cur) cur = [];
      cur.push(i);
      lastT = t[i];
    }
    if (cur) segs.push(cur);
    return segs;
  }

  function drawTrails(g) {
    clear(L.trails);
    st.trailPts = [];
    const tr = st.trails;
    if (tr && tr.tracks) {
      for (const id of Object.keys(tr.tracks)) {
        const track = tr.tracks[id];
        const color = track.color || 'var(--tag-other)';
        const segs = trailSegments(track, tr.range);
        let first = null;
        let last = null;
        for (const seg of segs) {
          let d = '';
          let lx = null;
          let ly = null;
          for (let j = 0; j < seg.length; j += 1) {
            const i = seg[j];
            const x = g.X(track.u[i]);
            const y = g.Y(track.v[i]);
            const end = j === seg.length - 1;
            if (lx != null && !end && Math.abs(x - lx) < 0.75 && Math.abs(y - ly) < 0.75) continue;
            d += `${d ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
            lx = x;
            ly = y;
            st.trailPts.push({ x, y, id, i, track });
          }
          if (seg.length === 1) d += `l0.01,0`;
          if (!first) first = seg[0];
          last = seg[seg.length - 1];
          L.trails.appendChild(s('path', { class: 'fm-trail', attrs: { d }, style: { stroke: color, strokeOpacity: 0.75 } }));
        }
        if (first != null) {
          L.trails.appendChild(s('circle', { attrs: { cx: g.X(track.u[first]), cy: g.Y(track.v[first]), r: 3 }, style: { fill: color } }));
          L.trails.appendChild(s('circle', { attrs: { cx: g.X(track.u[last]), cy: g.Y(track.v[last]), r: 4.5 }, style: { fill: 'var(--surface)', stroke: color, strokeWidth: 2 } }));
        }
      }
    }
    const live = st.live;
    if (live && live.trails) {
      const colorOf = new Map((live.badges || []).map((b) => [String(b.id), b.color]));
      for (const id of Object.keys(live.trails)) {
        // a null entry breaks the line (a gap in the badge's readings)
        let d = '';
        let run = 0;
        for (const p of live.trails[id] || []) {
          if (!p || !finite(p[0]) || !finite(p[1])) {
            run = 0;
            continue;
          }
          d += `${run ? 'L' : 'M'}${g.X(p[0]).toFixed(1)},${g.Y(p[1]).toFixed(1)}`;
          run += 1;
        }
        if (!d.includes('L')) continue;
        L.trails.appendChild(s('path', { class: 'fm-trail', attrs: { d }, style: { stroke: colorOf.get(String(id)) || 'var(--tag-other)', strokeOpacity: 0.45 } }));
      }
    }
  }

  function visibleBadges() {
    const out = new Map();
    for (const b of (st.live && st.live.badges) || []) {
      if (!b || !finite(b.u) || !finite(b.v)) continue;
      if (ageOpacity(b.age) <= 0) continue;
      out.set(String(b.id), b);
    }
    return out;
  }

  function drawRelations(g) {
    clear(L.dyads);
    clear(L.edges);
    const live = st.live;
    if (!live) return;
    const vis = visibleBadges();
    const active = st.hovered || st.selected;
    for (const p of live.pairs || []) {
      const A = vis.get(String(p.a));
      const B = vis.get(String(p.b));
      if (!A || !B) continue;
      const on = active != null && (String(p.a) === active || String(p.b) === active);
      const x1 = g.X(A.u);
      const y1 = g.Y(A.v);
      const x2 = g.X(B.u);
      const y2 = g.Y(B.v);
      L.dyads.appendChild(s('line', { class: ['fm-dyad', on ? 'is-active' : null], attrs: { x1, y1, x2, y2 } }));
      if (on && finite(p.d)) {
        L.dyads.appendChild(s('text', { class: 'fm-dist', attrs: { x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 4, 'text-anchor': 'middle' }, text: fmt.metres(p.d) }));
      }
    }
    const dirs = new Set((live.edges || []).map((e) => `${e.from}|${e.to}`));
    for (const e of live.edges || []) {
      const A = vis.get(String(e.from));
      const B = vis.get(String(e.to));
      if (!A || !B) continue;
      const mutual = !!e.mutual;
      if (mutual && String(e.from) > String(e.to) && dirs.has(`${e.to}|${e.from}`)) continue;
      let x1 = g.X(A.u);
      let y1 = g.Y(A.v);
      let x2 = g.X(B.u);
      let y2 = g.Y(B.v);
      const len = Math.hypot(x2 - x1, y2 - y1);
      if (len < 2 * (BADGE_R + 6)) continue;
      const ux = (x2 - x1) / len;
      const uy = (y2 - y1) / len;
      // two one-way arrows between the same badges sit side by side
      const off = !mutual && dirs.has(`${e.to}|${e.from}`) ? 3 : 0;
      const ox = -uy * off;
      const oy = ux * off;
      x1 += ux * (BADGE_R + 5) + ox;
      y1 += uy * (BADGE_R + 5) + oy;
      x2 -= ux * (BADGE_R + 5) - ox;
      y2 -= uy * (BADGE_R + 5) - oy;
      const size = 7;
      const sx1 = mutual ? x1 + ux * size * 0.8 : x1;
      const sy1 = mutual ? y1 + uy * size * 0.8 : y1;
      L.edges.appendChild(s('line', { class: 'fm-edge', attrs: { x1: sx1, y1: sy1, x2: x2 - ux * size * 0.8, y2: y2 - uy * size * 0.8 } }));
      L.edges.appendChild(s('path', { class: 'fm-arrow', attrs: { d: arrowPath(x1, y1, x2, y2, size, mutual) } }));
    }
  }

  function badgeTip(b) {
    const rows = [
      { value: `${fmt.num(b.u, 2)}, ${fmt.num(b.v, 2)}`, label: 'u, v (m)', color: b.color, key: 'dot' },
    ];
    if (finite(b.heading)) rows.push({ value: `${Math.round(((b.heading * 180) / Math.PI + 360) % 360)}°`, label: 'heading', key: 'none' });
    for (const p of (st.live && st.live.pairs) || []) {
      const other = String(p.a) === String(b.id) ? p.b : String(p.b) === String(b.id) ? p.a : null;
      if (other != null && finite(p.d)) rows.push({ value: fmt.metres(p.d), label: `to Tag ${other}`, key: 'none' });
    }
    return {
      title: b.label || `Tag ${b.id}`,
      rows,
      note: finite(b.age) && b.age > 2 ? `seen ${fmt.ago(b.age)}` : null,
    };
  }

  function drawBadges(g) {
    const vis = visibleBadges();
    for (const [id, node] of badgeNodes) {
      if (!vis.has(id)) {
        if (document.activeElement === node.g) tooltip.hide();
        node.g.remove();
        badgeNodes.delete(id);
      }
    }
    for (const [id, b] of vis) {
      let node = badgeNodes.get(id);
      if (!node) {
        const gg = s('g', { class: 'fm-badge', attrs: { tabindex: 0, role: 'img' } });
        node = {
          g: gg,
          hit: s('circle', { attrs: { r: 14 }, style: { fill: 'transparent' } }),
          arrowLine: s('line', { style: { strokeWidth: 2, strokeLinecap: 'round' } }),
          arrowHead: s('path'),
          ring: s('circle', { class: 'fm-ring', attrs: { r: BADGE_R + 4 } }),
          dot: s('circle', { attrs: { r: BADGE_R }, style: { stroke: 'var(--surface)', strokeWidth: 2 } }),
          label: s('text', { class: 'fm-badge-label' }),
          data: b,
        };
        gg.append(node.hit, node.arrowLine, node.arrowHead, node.ring, node.dot, node.label);
        const n = node;
        gg.addEventListener('pointermove', (e) => {
          tooltip.show(e, badgeTip(n.data));
          if (st.hovered !== String(n.data.id)) {
            st.hovered = String(n.data.id);
            drawRelations(st.g);
          }
        });
        gg.addEventListener('pointerleave', () => {
          tooltip.hide();
          st.hovered = null;
          drawRelations(st.g);
        });
        gg.addEventListener('focus', () => {
          tooltip.show(gg, badgeTip(n.data));
          st.hovered = String(n.data.id);
          drawRelations(st.g);
        });
        gg.addEventListener('blur', () => {
          tooltip.hide();
          st.hovered = null;
          drawRelations(st.g);
        });
        const toggle = () => {
          const key = String(n.data.id);
          st.selected = st.selected === key ? null : key;
          drawRelations(st.g);
          if (typeof opts.onSelect === 'function') opts.onSelect(st.selected);
        };
        gg.addEventListener('click', toggle);
        gg.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggle();
          }
        });
        badgeNodes.set(id, node);
        L.badges.appendChild(gg);
      }
      node.data = b;
      const x = g.X(b.u);
      const y = g.Y(b.v);
      const color = b.color || 'var(--tag-other)';
      node.g.setAttribute('transform', `translate(${x.toFixed(1)},${y.toFixed(1)})`);
      node.g.style.opacity = String(ageOpacity(b.age));
      node.g.setAttribute('aria-label', `${b.label || `Tag ${b.id}`}${finite(b.age) && b.age > 2 ? `, seen ${fmt.ago(b.age)}` : ''}`);
      node.dot.style.fill = color;
      if (finite(b.heading)) {
        const d = headingVec(b.heading);
        const ex = d.x * ARROW_LEN;
        const ey = d.y * ARROW_LEN;
        node.arrowLine.setAttribute('x1', d.x * BADGE_R);
        node.arrowLine.setAttribute('y1', d.y * BADGE_R);
        node.arrowLine.setAttribute('x2', ex - d.x * 4);
        node.arrowLine.setAttribute('y2', ey - d.y * 4);
        node.arrowLine.style.stroke = color;
        node.arrowHead.setAttribute('d', arrowPath(0, 0, ex, ey, 6, false));
        node.arrowHead.style.fill = color;
        node.arrowLine.style.display = '';
        node.arrowHead.style.display = '';
      } else {
        node.arrowLine.style.display = 'none';
        node.arrowHead.style.display = 'none';
      }
      // label sits opposite the heading arrow so the two never overlap
      let lx = 12;
      let ly = -8;
      if (finite(b.heading)) {
        const d = headingVec(b.heading);
        lx = -d.x * 14;
        ly = -d.y * 14 + 4;
      }
      node.label.setAttribute('x', lx);
      node.label.setAttribute('y', ly);
      node.label.setAttribute('text-anchor', 'middle');
      node.label.textContent = b.short != null ? String(b.short) : String(b.id);
    }
  }

  function render() {
    if (!st.alive) return;
    const W = Math.floor(el.clientWidth);
    st.width = W;
    if (W <= 0) return;
    const g = geom(W);
    st.g = g;
    svg.setAttribute('width', W);
    svg.setAttribute('height', g.H);
    svg.setAttribute('viewBox', `0 0 ${W} ${g.H}`);
    drawHeat(g);
    drawStatic(g);
    drawTrails(g);
    drawRelations(g);
    drawBadges(g);
  }

  function renderDynamic() {
    if (!st.g || !st.alive) return render();
    drawTrails(st.g);
    drawRelations(st.g);
    drawBadges(st.g);
  }

  const schedule = () => {
    if (!st.raf) {
      st.raf = requestAnimationFrame(() => {
        st.raf = 0;
        render();
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

  // nearest trail sample or heat cell under the pointer
  svg.addEventListener('pointermove', (e) => {
    const g = st.g;
    if (!g || e.target.closest('.fm-badge, .fm-camera')) return;
    const r = svg.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    clear(L.hover);
    if (st.trailPts.length) {
      let best = null;
      let bd = 144;
      for (const p of st.trailPts) {
        const d = (p.x - x) ** 2 + (p.y - y) ** 2;
        if (d < bd) {
          bd = d;
          best = p;
        }
      }
      if (best) {
        const tr = best.track;
        L.hover.appendChild(s('circle', { class: 'c-dot', attrs: { cx: best.x, cy: best.y, r: 4 }, style: { fill: tr.color || 'var(--tag-other)', stroke: 'var(--surface)', strokeWidth: 2 } }));
        const clock = st.trails && typeof st.trails.timeFormat === 'function' ? st.trails.timeFormat : (t) => fmt.clock(t);
        tooltip.show(e, {
          title: tr.label || `Tag ${best.id}`,
          rows: [
            { value: clock(tr.t[best.i]), label: 'time', color: tr.color, key: 'line' },
            { value: `${fmt.num(tr.u[best.i], 2)}, ${fmt.num(tr.v[best.i], 2)}`, label: 'u, v (m)', key: 'none' },
          ],
        });
        return;
      }
    }
    const hm = st.heat && st.heat.grid;
    if (hm && hm.counts) {
      const u = g.U(x);
      const v = g.V(y);
      const iu = Math.floor((u - hm.u0) / hm.cell);
      const iv = Math.floor((v - hm.v0) / hm.cell);
      if (iu >= 0 && iu < hm.nu && iv >= 0 && iv < hm.nv) {
        const c = hm.counts[iv * hm.nu + iu];
        const format = st.heat.format || ((val) => fmt.duration(val));
        const ua = hm.u0 + iu * hm.cell;
        const va = hm.v0 + iv * hm.cell;
        tooltip.show(e, {
          title: `u ${fmt.num(ua, 2)} to ${fmt.num(ua + hm.cell, 2)} m, v ${fmt.num(va, 2)} to ${fmt.num(va + hm.cell, 2)} m`,
          rows: [{ value: finite(c) ? format(c) : fmt.na, label: st.heat.label || 'time in cell', color: seqColor(c, st.heatMax, 'sqrt'), key: 'rect' }],
        });
        return;
      }
    }
    tooltip.hide();
  });
  svg.addEventListener('pointerleave', () => {
    clear(L.hover);
    tooltip.hide();
  });

  render();

  return {
    /** live badges: [{id, color, label, short, u, v, heading, age}], trails {id: [[u, v] | null]} (null breaks the line), edges [{from, to, mutual}], pairs [{a, b, d}] */
    setLive(live) {
      st.live = live || null;
      const pts = [];
      if (live) {
        for (const b of live.badges || []) if (b && ageOpacity(b.age) > 0) pts.push([b.u, b.v]);
        for (const id of Object.keys(live.trails || {})) for (const p of live.trails[id] || []) if (p) pts.push(p);
      }
      if (grow(pts)) render();
      else renderDynamic();
    },
    /** tracks {id: {color, label, t, u, v}}, range [a, b] | null (session offsets), timeFormat(t) optional */
    setTrails(trails) {
      st.trails = trails && trails.tracks ? trails : null;
      if (st.trails) {
        const pts = [];
        for (const id of Object.keys(st.trails.tracks)) {
          const tr = st.trails.tracks[id];
          for (let i = 0; i < (tr.u || []).length; i += 1) pts.push([tr.u[i], tr.v[i]]);
        }
        if (grow(pts)) return render();
      }
      renderDynamic();
    },
    /** grid {u0, v0, cell, nu, nv, counts} (row-major, v rows), label, format(count) optional */
    setHeat(heat) {
      st.heat = heat && heat.grid ? heat : null;
      if (st.heat && st.auto) {
        const gd = st.heat.grid;
        grow([[gd.u0, gd.v0], [gd.u0 + gd.nu * gd.cell, gd.v0 + gd.nv * gd.cell]]);
      }
      render();
    },
    setExtent(extent) {
      if (validExtent(extent)) {
        st.extent = copyExtent(extent);
        st.auto = false;
      }
      render();
    },
    setCameras(cameras) {
      st.cameras = cameras || [];
      render();
    },
    clear() {
      st.live = null;
      st.trails = null;
      st.heat = null;
      st.selected = null;
      st.hovered = null;
      tooltip.hide();
      render();
    },
    destroy() {
      st.alive = false;
      if (ro) ro.disconnect();
      offTheme();
      if (st.raf) cancelAnimationFrame(st.raf);
      tooltip.hide();
      badgeNodes.clear();
      clear(el);
      el.classList.remove('floormap');
    },
  };
}
