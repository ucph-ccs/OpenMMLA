"""Where the badges were in the room, how they moved, how close the pupils stayed and who faced whom,
from a session's IPS events (ips_translation, ips_rotation, ips_relation).

Positions arrive in the main camera's OpenCV frame (x right, y down, z along the optical axis). That
camera looks down at 23-47 degrees, so its x-z plane is not the floor: depth shrinks by the cosine of
the pitch and badge height leaks into it. A hanging badge's tag y axis points down, so the mean of the
rotations' second column over a session is the room's gravity in camera coordinates, and the plane
normal to it is the floor. `floor_basis` builds that basis and `project` maps a camera-frame point to
floor coordinates (u to the right of the main camera, v away from it, h up), which keeps distances
(the basis is orthonormal) and turns the map into a true top-down plan.

The positions are in the camera's frame as the sensor gives it, whatever turned the picture on its
way (openmmla.utils.video.turn), so a main camera hung upside down has its x axis pointing to the
room's left and its y axis up, and one on its side has its x axis up or down. The basis therefore
takes u from the picture's right as it hangs: the gravity says which quarter turn stands the picture
upright (the one whose down axis lies nearest g), and the right axis of the picture so turned, laid
flat, is u, so the plan is never a mirror image ((u, v, h) right-handed) and holds for a camera on
its side. For a camera hung upright (rolled less than 45 degrees) that is the camera's x axis laid
flat, as it always was. Without a basis the plan falls back on the camera's own x-z plane, which
cannot tell which way up the camera hangs: `camera_floor` takes the main camera's turn
(`main_camera_turn`, from the session document, which the dashboard's report job passes) and lays
the plane of the picture as it was turned upright instead. The dashboard's live and replay stream
does not take this fallback (stream._usable_basis): it keeps the camera's own plane until it finds
the floor from the badges' rotations.

Badge reads are noisy (5-8 mm of jitter per second, jumps when the reading camera changes) and
flicker, so tracks are median-smoothed within runs of consecutive windows, movement ignores steps
below the jitter and above a plausible walk, a tag seen in under 2 % of the windows in which any
badge was read counts as a misread (a session whose cameras saw nobody most of the time keeps its
real pupils), and facing is normalised by the windows in which both badges were read.

Headings are radians on the floor, clockwise from v (away from the main camera) towards u.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from openmmla.utils.video.turn import TURNS, axis_turn, normalize_turn, total_turn

RARE_PRESENCE = 0.02
MAX_PUPIL_TAG = 12
RUN_GAP = 1.5
PRESENCE_GAP = 2.0
STEP_MIN = 0.05
STEP_MAX = 1.0
ACTIVE_STEP = 0.1
CELL = 0.1
CELL_GROWTH = 0.05
MAX_CELLS = 80
EXTENT_PAD = 0.3
CAMERA_REACH = 6.0
SERIES_STEP = 10.0
FULL_TRACKS_SECONDS = 4 * 3600.0


def _parsed(value):
    """a payload field as a Python object; Influx stores them as JSON strings."""
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _scalar(value) -> float:
    if isinstance(value, (list, tuple)) or getattr(value, 'ndim', 0):
        return float(value[0])
    return float(value)


def position(translation) -> tuple[float, float, float] | None:
    """a badge position (x, y, z) from the synchronizer's [[x], [y], [z]] or a flat [x, y, z];
    None unless it holds three finite numbers."""
    translation = _parsed(translation)
    try:
        values = [_scalar(v) for v in translation]
    except (TypeError, ValueError, IndexError):
        return None
    if len(values) < 3 or not all(math.isfinite(v) for v in values[:3]):
        return None
    return values[0], values[1], values[2]


def _matrix(rotation) -> tuple[tuple[float, ...], ...] | None:
    """a 3x3 rotation (row-major nested or flat 9) as float rows; None when it is not one."""
    rotation = _parsed(rotation)
    try:
        rows = [[float(v) for v in row] for row in rotation]
    except (TypeError, ValueError):
        try:
            flat = [float(v) for v in rotation]
        except (TypeError, ValueError):
            return None
        rows = [flat[0:3], flat[3:6], flat[6:9]] if len(flat) == 9 else []
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        return None
    if not all(math.isfinite(v) for row in rows for v in row):
        return None
    return tuple(tuple(row) for row in rows)


def _rotations_of(record) -> list:
    """the rotation matrices of an ips_rotation record or of a plain {tag: R} dict."""
    if not isinstance(record, dict):
        return []
    rotations = _parsed(record['rotations']) if 'rotations' in record else record
    return list(rotations.values()) if isinstance(rotations, dict) else []


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(v) -> tuple[float, float, float] | None:
    norm = math.sqrt(_dot(v, v))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    return v[0] / norm, v[1] / norm, v[2] / norm


def _gravity(rotation_records) -> tuple[tuple[float, float, float], int]:
    """the summed tag y axes (column 1) of every rotation observation, and how many there were."""
    sx = sy = sz = 0.0
    n = 0
    for record in rotation_records or ():
        for rotation in _rotations_of(record):
            m = _matrix(rotation)
            if m is None:
                continue
            column = (m[0][1], m[1][1], m[2][1])
            norm = math.sqrt(_dot(column, column))
            # a rotation's columns are unit vectors; anything far off is not a pose
            if not 0.5 < norm < 1.5:
                continue
            sx, sy, sz = sx + column[0], sy + column[1], sz + column[2]
            n += 1
    return (sx, sy, sz), n


def upright_turn(g) -> int:
    """the quarter turn (0, 90, 180, 270, as openmmla.utils.video.turn turns a picture) that stands a
    camera's picture upright, from the down direction g in its frame: the one whose turned picture's
    down axis (row 1 of axis_turn) lies nearest g. 0 for a camera rolled less than 45 degrees, 180 for
    one hung upside down."""
    return max(TURNS, key=lambda turn: float(np.dot(axis_turn(turn)[1], g)))


def floor_basis(rotation_records: list[dict], min_obs: int = 30) -> dict | None:
    """the floor plane of a session in its main camera's frame, from the badges' gravity.

    g is the mean down direction of the hanging badges (their tag y axis), ex the picture's right
    laid flat on the floor, ef the direction away from the camera on the floor. The picture's right
    is the camera's x axis for a camera hung upright, -x for one hung upside down (whose x axis points
    to the room's left) and -y or y for one on its side: the right axis of the quarter turn whose
    down axis lies nearest g (`upright_turn`), so that (ex, ef, -g) is right-handed however the
    camera hangs. None when there are fewer than `min_obs` rotation observations (or they do not
    define a plane, as for a camera looking straight down)."""
    total, n = _gravity(rotation_records)
    if n < max(1, min_obs):
        return None
    g = _unit(total)
    if g is None:
        return None
    right = tuple(float(v) for v in axis_turn(upright_turn(g))[0])
    rg = _dot(right, g)
    ex = _unit((right[0] - rg * g[0], right[1] - rg * g[1], right[2] - rg * g[2]))
    if ex is None:
        return None
    # the camera's z axis minus its parts along g and ex (Gram-Schmidt), so ef is orthogonal to both
    zg = g[2]
    z_minus_g = (-zg * g[0], -zg * g[1], 1.0 - zg * g[2])
    zx = _dot(z_minus_g, ex)
    ef = _unit((z_minus_g[0] - zx * ex[0], z_minus_g[1] - zx * ex[1], z_minus_g[2] - zx * ex[2]))
    if ef is None:
        return None
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, g[2]))))
    return {
        'g': [round(v, 6) for v in g],
        'ex': [round(v, 6) for v in ex],
        'ef': [round(v, 6) for v in ef],
        'pitch_deg': round(pitch, 2),
        'n': n,
        'method': 'badge-gravity',
    }


def camera_floor(turn=0, n: int = 0) -> dict:
    """the fallback floor of a session whose badge rotations are too few for `floor_basis`: the main
    camera's x-z plane as its picture was turned upright, given in the frame the positions are in
    (the sensor's). `turn` is how far the picture was turned to be upright (the capture's turn and
    Base.rotate, `main_camera_turn`); 0 is the camera's own x-z plane, (x, z, -y), right for a camera
    hung upright and a mirror image for one hung upside down when its turn is not known. With p_up =
    Q p (openmmla.utils.video.turn.axis_turn), ex, g and ef are the rows of Q."""
    turn = normalize_turn(turn)
    Q = axis_turn(turn)
    row = lambda i: [round(float(v), 6) + 0.0 for v in Q[i]]  # + 0.0: no -0.0 in the report
    return {'g': row(1), 'ex': row(0), 'ef': row(2), 'pitch_deg': None, 'n': n, 'method': 'camera-xz',
            'turn': turn}


def main_camera_turn(doc: dict | None) -> int:
    """how far the main IPS camera's picture was turned to be upright, from a session document: its
    base's parameters (`capture_turn` and `rotate`, of a base that reports its poses in the sensor's
    frame, `pose_frame: sensor`), else the turn its stream was captured with
    (`sources[].capture.rotate`), else 0. A base from before 2026-10-02 reported its poses on the
    turned picture, which is upright already, so its `rotate` does not count."""
    from openmmla.analytics.report.sessions import _components, _ips_main, _newest_first

    main = _ips_main(doc)
    if not main:
        return 0
    for entry in _newest_first(_components(doc)):
        parameters = entry.get('parameters') if isinstance(entry.get('parameters'), dict) else {}
        if entry.get('pipeline') != 'ips' or entry.get('role') != 'base' or str(parameters.get('base_id')) != main:
            continue
        if parameters.get('pose_frame') == 'sensor':
            return total_turn(parameters.get('capture_turn'), parameters.get('rotate'))
        break
    sources = (doc or {}).get('sources')
    for entry in sources if isinstance(sources, list) else ():
        if isinstance(entry, dict) and entry.get('pipeline') == 'ips' and str(entry.get('base_id')) == main:
            capture = entry.get('capture') if isinstance(entry.get('capture'), dict) else {}
            return normalize_turn(capture.get('rotate'))
    return 0


def project(p, basis: dict | None) -> tuple[float, float, float]:
    """a camera-frame point (x, y, z) in metres as floor coordinates (u, v, h): u to the right of
    the main camera, v away from it, h up. Without a basis, the camera's own x-z plane: (x, z, -y)
    (`camera_floor` is the same plane for a camera whose picture was turned upright)."""
    if p and isinstance(p[0], (list, tuple)):
        p = position(p)
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if not basis:
        return x, z, -y
    ex, ef, g = basis['ex'], basis['ef'], basis['g']
    return (x * ex[0] + y * ex[1] + z * ex[2],
            x * ef[0] + y * ef[1] + z * ef[2],
            -(x * g[0] + y * g[1] + z * g[2]))


def floor_heading(direction, basis: dict | None) -> float | None:
    """the heading (radians, clockwise from v towards u) of a camera-frame direction laid on the
    floor, e.g. a badge's facing (-column 2 of its rotation) or a camera's optical axis; None when
    the direction is missing or points straight up or down."""
    d = position(direction)
    if d is None:
        return None
    u, v, _ = project(d, basis)
    if math.hypot(u, v) < 1e-9:
        return None
    return math.atan2(u, v)


def _r(value, digits: int = 3):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _tag_key(tag: str) -> tuple:
    return (0, int(tag), tag) if tag.isdigit() else (1, 0, tag)


def _is_pupil_tag(tag: str) -> bool:
    return tag.isdigit() and int(tag) <= MAX_PUPIL_TAG


def _num(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _runs(times: list[float], gap: float) -> list[tuple[int, int]]:
    """index ranges [i, j) of consecutive samples no more than `gap` apart."""
    runs = []
    start = 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > gap:
            runs.append((start, i))
            start = i
    if times:
        runs.append((start, len(times)))
    return runs


def _median3(values: list[float], i: int, lo: int, hi: int) -> float:
    """the rolling median of 3 centred on i within [lo, hi); the mean of the two at a run's ends."""
    window = values[max(lo, i - 1):min(hi, i + 2)]
    if len(window) == 3:
        return sorted(window)[1]
    return sum(window) / len(window)


def _extent(us: np.ndarray, vs: np.ndarray, cameras: list[dict]) -> dict | None:
    """a robust plan extent: 1st-99th percentile of the samples padded, plus nearby cameras."""
    if len(us):
        u0, u1 = (float(x) for x in np.percentile(us, [1, 99]))
        v0, v1 = (float(x) for x in np.percentile(vs, [1, 99]))
        u0, u1, v0, v1 = u0 - EXTENT_PAD, u1 + EXTENT_PAD, v0 - EXTENT_PAD, v1 + EXTENT_PAD
        for camera in cameras:
            cu, cv = camera['u'], camera['v']
            # how far the camera lies outside the data box; a far one would leave the badges a corner
            outside = math.hypot(max(u0 - cu, 0.0, cu - u1), max(v0 - cv, 0.0, cv - v1))
            if outside <= CAMERA_REACH:
                u0, u1 = min(u0, cu - EXTENT_PAD), max(u1, cu + EXTENT_PAD)
                v0, v1 = min(v0, cv - EXTENT_PAD), max(v1, cv + EXTENT_PAD)
    elif cameras:
        u0 = min(c['u'] for c in cameras) - 1.0
        u1 = max(c['u'] for c in cameras) + 1.0
        v0 = min(c['v'] for c in cameras) - 1.0
        v1 = max(c['v'] for c in cameras) + 1.0
    else:
        return None
    return {'u': [math.floor(u0 * 10 + 1e-9) / 10, math.ceil(u1 * 10 - 1e-9) / 10],
            'v': [math.floor(v0 * 10 + 1e-9) / 10, math.ceil(v1 * 10 - 1e-9) / 10]}


def _grid(extent: dict) -> tuple[float, int, int]:
    """the occupancy cell size and grid shape: 0.1 m cells, grown until no axis exceeds MAX_CELLS."""
    width = max(extent['u'][1] - extent['u'][0], CELL)
    depth = max(extent['v'][1] - extent['v'][0], CELL)
    step = 0
    while True:
        cell = round(CELL + step * CELL_GROWTH, 4)
        nu = max(1, math.ceil(width / cell - 1e-9))
        nv = max(1, math.ceil(depth / cell - 1e-9))
        if nu <= MAX_CELLS and nv <= MAX_CELLS:
            return cell, nu, nv
        step += 1


def _occupancy(us, vs, u0, v0, cell, nu, nv) -> list[int]:
    if not len(us):
        return [0] * (nu * nv)
    iu = np.floor((np.asarray(us) - u0) / cell + 1e-9).astype(int)
    iv = np.floor((np.asarray(vs) - v0) / cell + 1e-9).astype(int)
    inside = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    counts = np.bincount(iv[inside] * nu + iu[inside], minlength=nu * nv)
    return [int(c) for c in counts]


def _pair_summary(offsets: list[float], distances: list[float], bucket: float, n_bins: int) -> dict:
    out = {'median': None, 'mean': None, 'p10': None, 'p90': None, 'lt05': None, 'lt1': None,
           'copresent': _r(len(distances) * bucket, 1)}
    bins: dict[int, list[float]] = defaultdict(list)
    if distances:
        d = np.asarray(distances, dtype=float)
        p10, median, p90 = np.percentile(d, [10, 50, 90])
        out.update(median=_r(median), mean=_r(d.mean()), p10=_r(p10), p90=_r(p90),
                   lt05=_r((d < 0.5).mean()), lt1=_r((d < 1.0).mean()))
        for offset, distance in zip(offsets, distances):
            if offset >= -1e-6:
                bins[max(0, int(offset // SERIES_STEP))].append(distance)
    n_bins = max([n_bins] + [b + 1 for b in bins])
    out['series'] = {
        't': [round(i * SERIES_STEP, 2) for i in range(n_bins)],
        'd': [_r(float(np.median(bins[i]))) if i in bins else None for i in range(n_bins)],
    }
    return out


def build_space(translations: list[dict], rotations: list[dict], relations: list[dict], t0: float,
                t1: float, cameras: list[dict] | None = None, main_turn=0) -> dict:
    """the space part of a session report from its parsed IPS records (see the module docstring).
    `main_turn` is how far the main camera's picture was turned to be upright (`main_camera_turn`),
    which only the fallback floor without badge rotations needs."""
    t0 = float(t0)
    t1 = float(t1) if t1 is not None else t0
    duration = max(0.0, t1 - t0)

    basis = floor_basis(rotations or [])
    if basis is None:
        _, n_obs = _gravity(rotations or [])
        floor = camera_floor(main_turn, n_obs)
        # an unturned camera keeps the plain (x, z, -y) of project
        basis = floor if floor['turn'] else None
    else:
        floor = basis

    # one entry per IPS window: {tag: camera-frame position}; a repeated window merges
    windows: dict[float, dict[str, tuple[float, float, float]]] = {}
    lengths = []
    for record in translations or ():
        ws = _num(record.get('window_start_time'))
        if ws is None:
            continue
        slot = windows.setdefault(ws, {})
        values = _parsed(record.get('translations'))
        if isinstance(values, dict):
            for tag, value in values.items():
                p = position(value)
                if p is not None:
                    slot[str(tag)] = p
        we = _num(record.get('window_end_time'))
        if we is not None and we > ws:
            lengths.append(we - ws)
    bucket = float(np.median(lengths)) if lengths else 1.0
    starts = sorted(windows)
    n_windows = len(starts)

    samples: dict[str, list[tuple[float, tuple[float, float, float]]]] = defaultdict(list)
    for ws in starts:
        for tag, p in windows[ws].items():
            samples[tag].append((ws, p))
    tags = sorted(samples, key=_tag_key)
    present = {tag: len(samples[tag]) / n_windows for tag in tags} if n_windows else {}
    n_nonempty = sum(1 for ws in starts if windows[ws])
    rare = {tag: len(samples[tag]) < RARE_PRESENCE * n_nonempty or not _is_pupil_tag(tag) for tag in tags}
    kept = [tag for tag in tags if not rare[tag]]

    projected_cameras = []
    for camera in cameras or []:
        p = position(camera.get('position'))
        if p is None:
            continue
        u, v, _ = project(p, basis)
        axis = camera.get('axis')
        projected_cameras.append({
            'id': str(camera.get('id')), 'u': _r(u), 'v': _r(v), 'main': bool(camera.get('main')),
            'heading': _r(floor_heading(axis, basis), 4) if axis is not None else None,
        })
    projected_cameras.sort(key=lambda c: (not c['main'], _tag_key(c['id'])))

    tag_rows = []
    tracks = {}
    presence = {}
    smooth: dict[str, tuple[list[float], list[float], list[float], list[float]]] = {}
    stride = max(1, math.ceil(duration / FULL_TRACKS_SECONDS)) if duration > FULL_TRACKS_SECONDS else 1
    for tag in tags:
        seen = samples[tag]
        times = [ws for ws, _ in seen]
        raw = [project(p, basis) for _, p in seen]
        ru, rv, rh = [p[0] for p in raw], [p[1] for p in raw], [p[2] for p in raw]
        su, sv, sh = [], [], []
        path = 0.0
        active = 0
        for lo, hi in _runs(times, RUN_GAP):
            for i in range(lo, hi):
                su.append(round(_median3(ru, i, lo, hi), 3))
                sv.append(round(_median3(rv, i, lo, hi), 3))
                sh.append(round(_median3(rh, i, lo, hi), 3))
            for i in range(lo + 1, hi):
                step = math.hypot(su[i] - su[i - 1], sv[i] - sv[i - 1])
                if STEP_MIN <= step <= STEP_MAX:
                    path += step
                if ACTIVE_STEP < step <= STEP_MAX:
                    active += 1
        offsets = [round(ws - t0, 2) for ws in times]
        smooth[tag] = (offsets, su, sv, sh)
        tracks[tag] = {'t': offsets[::stride], 'u': su[::stride], 'v': sv[::stride], 'h': sh[::stride]}

        spans = []
        for lo, hi in _runs(times, PRESENCE_GAP):
            spans.append([round(times[lo] - t0, 2), round(times[hi - 1] + bucket - t0, 2)])
        presence[tag] = spans

        seconds = len(seen) * bucket
        tag_rows.append({
            'id': tag,
            'present': _r(present[tag], 4),
            'seconds': _r(seconds, 1),
            'path_m': _r(path, 2),
            'm_per_min': _r(path / (seconds / 60.0), 3) if seconds > 0 else None,
            'active_s': _r(active * bucket, 1),
            'rare': rare[tag],
        })

    pooled_u = [u for tag in kept for u in smooth[tag][1]]
    pooled_v = [v for tag in kept for v in smooth[tag][2]]
    if not pooled_u:
        # nothing but misreads: still draw them on a plan
        pooled_u = [u for tag in tags for u in smooth[tag][1]]
        pooled_v = [v for tag in tags for v in smooth[tag][2]]
    extent = _extent(np.asarray(pooled_u, dtype=float), np.asarray(pooled_v, dtype=float), projected_cameras)
    occupancy = None
    if extent is not None:
        cell, nu, nv = _grid(extent)
        u0, v0 = extent['u'][0], extent['v'][0]
        # the plan spans the whole grid, so its last row and column are not cut off
        extent = {'u': [u0, round(u0 + nu * cell, 3)], 'v': [v0, round(v0 + nv * cell, 3)]}
        by_tag = {tag: _occupancy(smooth[tag][1], smooth[tag][2], u0, v0, cell, nu, nv) for tag in kept}
        pooled = [sum(column) for column in zip(*by_tag.values())] if by_tag else [0] * (nu * nv)
        occupancy = {'cell': cell, 'u0': u0, 'v0': v0, 'nu': nu, 'nv': nv, 'pooled': pooled, 'by_tag': by_tag}

    n_bins = max(1, math.ceil(duration / SERIES_STEP - 1e-9))
    pairs = []
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            offsets, distances = [], []
            for ws in starts:
                window = windows[ws]
                if a in window and b in window:
                    offsets.append(ws - t0)
                    distances.append(math.dist(window[a], window[b]))
            pairs.append({'a': a, 'b': b, **_pair_summary(offsets, distances, bucket, n_bins)})

    graphs: dict[float, dict[str, set[str]]] = {}
    for record in relations or ():
        ws = _num(record.get('window_start_time'))
        graph = _parsed(record.get('graph'))
        if ws is None or not isinstance(graph, dict):
            continue
        faces = graphs.setdefault(ws, {})
        for src, targets in graph.items():
            targets = _parsed(targets)
            faces.setdefault(str(src), set()).update(
                str(t) for t in (targets if isinstance(targets, (list, tuple, set)) else []))
    copresent: dict[tuple[str, str], int] = defaultdict(int)
    facing_counts: dict[tuple[str, str], int] = defaultdict(int)
    mutual: dict[tuple[str, str], int] = defaultdict(int)
    for faces in graphs.values():
        for i, a in enumerate(kept):
            if a not in faces:
                continue
            for b in kept[i + 1:]:
                if b not in faces:
                    continue
                copresent[(a, b)] += 1
                ab, ba = b in faces[a], a in faces[b]
                facing_counts[(a, b)] += ab
                facing_counts[(b, a)] += ba
                mutual[(a, b)] += ab and ba
    facing, mutual_facing = [], []
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            n = copresent.get((a, b), 0)
            if not n:
                continue
            for src, tgt in ((a, b), (b, a)):
                count = facing_counts[(src, tgt)]
                facing.append({'src': src, 'tgt': tgt, 'count': count, 'copresent': n, 'ratio': _r(count / n, 4)})
            mutual_facing.append({'a': a, 'b': b, 'count': mutual[(a, b)]})
    facing.sort(key=lambda e: (_tag_key(e['src']), _tag_key(e['tgt'])))

    return {
        't0': t0,
        'duration': _r(duration, 2),
        'floor': floor,
        'extent': extent,
        'cameras': projected_cameras,
        'tags': tag_rows,
        'tracks': tracks,
        'presence': presence,
        'occupancy': occupancy,
        'pairs': pairs,
        'facing': facing,
        'mutual_facing': mutual_facing,
        'coverage': {
            'windows': n_windows,
            'nonempty': sum(1 for ws in starts if windows[ws]),
            'expected': int(round(duration / bucket)) if bucket > 0 else None,
        },
    }
