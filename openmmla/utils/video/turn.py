"""a picture turned on its way to a base, and what the turn does to the camera's geometry.

A camera mounted upside down or on its side is turned upright once,
where it is captured, by the Streams entry's `rotate` (0, 90, 180 or 270, clockwise as cv2.rotate
turns): the console's ffmpeg turns the picture on the capture host (stream_panel._build_ffmpeg_cmd)
and the session notes it as the stream's `sources[].capture.rotate`. A base's own `Base.rotate`
turns what reaches it once more, for a source the console does not capture, and a Bases entry's
`capture_rotate` says a file (or a stream no Streams entry describes) was turned before it reached
the base, such as a recording ses-tidy flipped. The turn a base works with is the sum of the
capture's and its own. A stream must be restarted after its rotate changes: until then it still
sends the picture as it was started, while the bases already read the new turn.

The camera itself did not move, so what was measured on the picture as the sensor gives it holds
turned with the frame:

- intrinsics: a pixel (u, v) of a W x H sensor frame lands, turned 90 clockwise, at
  (H - 1 - v, u); 180 at (W - 1 - u, H - 1 - v); 270 at (v, W - 1 - u). The camera-frame point
  (X, Y, Z) is seen turned by the same quarter turns about the optical axis, p_turned = Q p_sensor
  with Q the rotation by the turn about z (z forward, y down, so clockwise on the picture is a
  positive angle): 90 gives (-Y, X, Z), 180 (-X, -Y, Z), 270 (Y, -X, Z). Projecting the turned
  point gives fx' = fy, fy' = fx, cx' = H - 1 - cy, cy' = cx for 90; fx, fy, W - 1 - cx,
  H - 1 - cy for 180; and fy, fx, cy, W - 1 - cx for 270 (`turned_intrinsics`).
- distortion: a fisheye camera's (Kannala-Brandt k1..k4) is radial and holds as it is; a pinhole
  camera's tangential p1, p2 would turn too (180 negates both, 90 gives (p2, -p1), 270 (-p2, p1)),
  but no base undistorts with them.
- poses: a pose (R, t) found on the turned frame is in the turned camera's frame; Q^T R and Q^T t
  give it in the sensor's (`pose_in_sensor_frame`), which is how IPS reports it whatever turned
  the picture, so the Camera Sync matrices, fitted on pictures as the sensor gives them, hold."""
from __future__ import annotations

import numpy as np

TURNS = (0, 90, 180, 270)

def normalize_turn(value) -> int:
    """0, 90, 180 or 270 of a turn in degrees (a number or its text, -90 is 270); anything else,
    an empty field included, is 0."""
    try:
        degrees = int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return 0
    degrees %= 360
    return degrees if degrees in TURNS else 0


def total_turn(*turns) -> int:
    """the turn of turns applied one after the other (the capture's, then the base's)."""
    return sum(normalize_turn(turn) for turn in turns) % 360


def turned_size(width: int, height: int, turn) -> tuple[int, int]:
    """(width, height) of a width x height frame turned by `turn`."""
    return (int(height), int(width)) if normalize_turn(turn) in (90, 270) else (int(width), int(height))


def sensor_size(width: int, height: int, turn) -> tuple[int, int]:
    """(width, height) the sensor gave a frame that is width x height after being turned by `turn`."""
    return turned_size(width, height, turn)  # a quarter turn swaps the sides either way


def turned_pixel(u, v, size, turn) -> tuple[float, float]:
    """where pixel (u, v) of a frame of `size` (width, height) lands when the frame is turned by
    `turn`, as cv2.rotate turns it."""
    width, height = size
    turn = normalize_turn(turn)
    if turn == 90:
        return height - 1 - v, u
    if turn == 180:
        return width - 1 - u, height - 1 - v
    if turn == 270:
        return v, width - 1 - u
    return u, v


def turned_intrinsics(params, size, turn) -> list[float]:
    """fx, fy, cx, cy for the frame turned by `turn`, of `params` measured on the frame as the
    sensor gives it, `size` (width, height) before the turn."""
    fx, fy, cx, cy = (float(v) for v in params)
    width, height = float(size[0]), float(size[1])
    turn = normalize_turn(turn)
    if turn == 90:
        return [fy, fx, height - 1 - cy, cx]
    if turn == 180:
        return [fx, fy, width - 1 - cx, height - 1 - cy]
    if turn == 270:
        return [fy, fx, cy, width - 1 - cx]
    return [fx, fy, cx, cy]


def turned_camera_matrix(K, size, turn) -> np.ndarray:
    """the 3x3 camera matrix K (no skew) for the frame turned by `turn`, `size` before the turn."""
    K = np.asarray(K, dtype=float).reshape(3, 3)
    fx, fy, cx, cy = turned_intrinsics((K[0, 0], K[1, 1], K[0, 2], K[1, 2]), size, turn)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def axis_turn(turn) -> np.ndarray:
    """Q: the rotation about the optical axis the picture's turn makes of the camera's frame,
    p_turned = Q p_sensor."""
    turn = normalize_turn(turn)
    c, s = {0: (1.0, 0.0), 90: (0.0, 1.0), 180: (-1.0, 0.0), 270: (0.0, -1.0)}[turn]
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose_in_sensor_frame(R, t, turn) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of a pose found on a frame turned by `turn`, in the camera's frame as the sensor
    gives it; t keeps its shape (3 or 3x1)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float)
    if not normalize_turn(turn):
        return R.copy(), t.copy()
    Qt = axis_turn(turn).T
    return Qt @ R, (Qt @ t.reshape(3)).reshape(t.shape)


def stream_capture_turn(config: dict | None, url: str | None = None, name: str | None = None, default=0) -> int:
    """the turn the capture applies to the stream a base pulls from `url` (the Streams entry
    `name`): the `rotate` of the Streams entry whose target or read_target is that URL, else of the
    entry `name`, else of the one served at the same Stream Server path; `default` when none is (a
    file, a local camera, a stream from elsewhere)."""
    from openmmla.utils.session_sources import stream_url_path

    streams = (config or {}).get('Streams') or {}
    if not isinstance(streams, dict):
        return normalize_turn(default)
    entries = {str(key): entry for key, entry in streams.items() if isinstance(entry, dict)}
    url = str(url or '').strip()
    name = str(name or '').strip()

    def urls(entry):
        return {str(entry.get(field) or '').strip() for field in ('target', 'read_target')} - {''}

    found = [key for key, entry in entries.items() if url and url in urls(entry)]
    if name in found or (not found and name in entries):
        return normalize_turn(entries[name].get('rotate'))
    if not found and url:
        path = stream_url_path(url)
        found = [key for key, entry in entries.items()
                 if path and any(stream_url_path(other) == path for other in urls(entry))]
    return normalize_turn(entries[found[0]].get('rotate') if found else default)


def recording_turns(records: dict) -> dict:
    """{device: turn} the recorder gave each file, from the manifest rows {device: row} a replay or
    ses-calibrate reads: a Collection recording notes its own `rotate` (the file is upright by it,
    0 included), which says more about the file than the session's sources, which describe the
    stream; a row without the field (one from before recorders turned, a stream cut) is left out,
    so the sources' turn holds for it."""
    return {device: normalize_turn(row.get('rotate'))
            for device, row in (records or {}).items() if isinstance(row, dict) and 'rotate' in row}


def base_capture_turn(config: dict | None, base_entry: dict | None, source: str, url=None, name=None) -> int:
    """the turn the capture gave the frames of a base: for a stream, the rotate of the Streams entry
    it pulls (stream_capture_turn); for anything else, and a stream no entry describes, the Bases
    entry's own `capture_rotate`, which says how a file was recorded (a session's recording replayed
    from its file), else 0."""
    declared = normalize_turn((base_entry or {}).get('capture_rotate'))
    if source == 'stream':
        return stream_capture_turn(config, url, name, default=declared)
    return declared
