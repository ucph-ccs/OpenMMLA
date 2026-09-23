"""mmla ses-align: the recordings of a session on one clock and one start.

A session's recordings come from different clocks: a per-person microphone (badge, vimo) is
stamped by the base that recorded it, a camera's file by the machine that wrote it, at best to
the second and after the delay of a stream. The camera's own audio track and the microphones
heard the same room, so cross-correlating them says how far the camera's nominal start is off,
and --apply moves that track and the videos that share its start (the camera's, or the crops of
one recording) by that much, and nothing else: not the reference, not a per-person microphone
that shares the start. --trim then cuts every recording to the session's common start, the
latest start among them: audio to the sample, video at the last keyframe before it (a stream
copy, nothing re-encoded), and names every file with its exact start. Nothing is kept: what
came before the common start is gone.

--end SECONDS cuts the other side: every recording ends SECONDS after the common start (the
latest start, what the manifest's initial_sync_time says after a rebuild, the reference --trim
uses). Audio is cut to the sample, video by stream copy, which needs no keyframe at an end: the
copy stops in decode order, so a video with B-frames keeps one or two frames past the cut (a
30 fps test cut at 4.30 s ended at 4.37 s) and its own audio track ends within one packet of it;
the manifest takes the length ffprobe reads. A recording that already ends by then (or within
END_SLACK after, so a second run cuts nothing again) is left as is. This one is reversible: the
full-length originals move under raw/<host>/<audio|video>/ (their paths under collection/, out
of the sources), the manifests are rebuilt with the new lengths and a note naming the cut.

--devices with --warp, --cut, --cut-zeros or --cut-head edits the timing inside audio files, for
what no shift of a start can fix: two channels that lost audio on their own (a piecewise time map
from a measured lag track), the silence an import put in for a wall-clock step of the recording
machine (the digital-zero runs at given times), a recording that runs a constant time late (its
head). Samples are cut or silence inserted where the timing changes, never resampled (that would
change the pitch of speech), the start stamp of the name stays, the original moves under raw/
(the first original, when a file is edited twice) with an .edits.json beside it, and the manifests
get the new lengths and a note with the numbers. --dry-run says what would change.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ENVELOPE_RATE = 100          # bins per second the audio is reduced to before correlating
MIC_DEVICES = ('vimo', 'badge')   # per-person microphones (device label prefix): the clock the others are aligned to
END_SLACK = 0.25             # seconds past --end a recording may run and count as ended (a stream-copied video does)
SAME_START = 0.0015          # seconds within which two file names give one start (the names carry milliseconds)


def envelope(path: str, offset: float, duration: float) -> np.ndarray:
    """the loudness of a stretch of a wav file, ENVELOPE_RATE bins a second, zero mean unit variance"""
    import soundfile as sf
    with sf.SoundFile(path) as f:
        rate = f.samplerate
        first = max(0, int(round(offset * rate)))
        f.seek(first)
        data = f.read(int(round(duration * rate)), dtype='float32', always_2d=True)[:, 0]
    bins = max(1, rate // ENVELOPE_RATE)
    usable = len(data) - len(data) % bins
    if usable <= 0:
        return np.zeros(0, dtype='float32')
    env = np.abs(data[:usable]).reshape(-1, bins).mean(axis=1)
    env = env - env.mean()
    scale = env.std()
    return env / scale if scale > 0 else env


def measure_lag(reference: dict[str, Any], other: dict[str, Any], window: float = 1200.0, max_lag: float = 120.0) -> dict[str, Any]:
    """how much later than its nominal start `other` really began, on the reference's clock:
    positive = its content is found later in the reference, so its start should move forward.
    Both are recordings {path, start_time, duration}."""
    from scipy.signal import correlate

    ref_start, ref_end = reference['start_time'], reference['start_time'] + reference['duration']
    oth_start, oth_end = other['start_time'], other['start_time'] + other['duration']
    begin = max(ref_start, oth_start) + max_lag
    length = min(window, min(ref_end, oth_end) - max_lag - begin)
    if length < 30:
        return {'lag': None, 'confidence': 0.0, 'reason': 'less than 30 s of overlap to compare'}
    ref_env = envelope(reference['path'], begin - max_lag - ref_start, length + 2 * max_lag)
    oth_env = envelope(other['path'], begin - oth_start, length)
    if len(oth_env) < ENVELOPE_RATE * 10 or len(ref_env) <= len(oth_env):
        return {'lag': None, 'confidence': 0.0, 'reason': 'too little audio to compare'}
    scores = correlate(ref_env, oth_env, mode='valid') / len(oth_env)
    best = int(np.argmax(scores))
    lag = (best - int(round(max_lag * ENVELOPE_RATE))) / ENVELOPE_RATE
    spread = scores.std() or 1e-9
    confidence = float((scores[best] - scores.mean()) / spread)
    # the runner-up outside a second of the peak: a real alignment stands well above it
    away = np.abs(np.arange(len(scores)) - best) > ENVELOPE_RATE
    runner_up = float(scores[away].max()) if away.any() else float('-inf')
    margin = float((scores[best] - runner_up) / spread)
    return {'lag': round(lag, 2), 'confidence': round(confidence, 1), 'margin': round(margin, 1),
            'compared_seconds': round(length, 1)}


def _recordings(session_dir: Path) -> list[dict[str, Any]]:
    data = json.loads((session_dir / 'manifest.json').read_text(encoding='utf-8'))
    return [r for r in data.get('recordings', []) if isinstance(r, dict) and os.path.exists(r.get('path', ''))]


def pick_reference(recordings: list[dict[str, Any]], host: str | None = None) -> dict[str, Any] | None:
    audio = [r for r in recordings if r['modality'] == 'audio' and r.get('duration')]
    if host:
        return next((r for r in audio if r['host'] == host), None)
    mics = [r for r in audio if str(r.get('device') or '').startswith(MIC_DEVICES)]
    return max(mics, key=lambda r: r['duration']) if mics else None


def measure_session(session_dir: Path, reference_host: str | None = None, window: float = 1200.0,
                    max_lag: float = 120.0) -> dict[str, Any]:
    from openmmla.commands.ses.tidy import audio_scope_of
    recordings = _recordings(session_dir)
    reference = pick_reference(recordings, reference_host)
    if reference is None:
        return {'reference': None, 'results': [], 'reason': 'no per-person microphone to align to'}
    results = []
    for other in recordings:
        if other['modality'] != 'audio' or other is reference or not other.get('duration'):
            continue
        result = measure_lag(reference, other, window, max_lag)
        results.append({'host': other['host'], 'device': other.get('device'), 'channel': other.get('channel'), 'start_time': other['start_time'],
                        'path': other['path'], 'scope': audio_scope_of(other),
                        'shares_start_with': sorted(r['host'] + '/' + r['modality'] for r in recordings
                                                    if r is not other and abs(r['start_time'] - other['start_time']) < SAME_START),
                        **result})
    return {'reference': reference['host'], 'reference_path': reference['path'], 'results': results}


def shift_plan(session_dir: Path, measured: dict[str, Any], reference: str | None = None
               ) -> tuple[list[Path], list[tuple[Path, str]]]:
    """(the files a lag measured for the recording `measured` {path, host, start_time} moves, the
    files sharing its start that stay and why). What moves is the recording itself and, when it is
    not a per-person microphone, every video that starts with it to the millisecond, on any host:
    the video of the camera it is the audio track of (a track filed under another machine than its
    video: jabra-0 under ericli, c920-01 under raspi4-01), or the crops of the one recording they
    were all cut from (2024-12-10 group_02: jabra-0, c920-01 and c920-04 out of one OBS mosaic), a
    track never moving without its picture. Nothing else moves along: not the reference, not a
    per-person microphone (a take split can give it the camera's start, as it did the 2025-05-13
    vimos, which a shift for the camera then moved too), and not another audio recording (each is
    measured on its own)."""
    from openmmla.collection.recording import default_audio_scope
    from openmmla.commands.ses.tidy import audio_scope_of, parse_recording_name
    own = Path(measured['path']).resolve()
    ref = Path(reference).resolve() if reference else None
    scopes = {Path(r['path']).resolve(): audio_scope_of(r) for r in _recordings(session_dir) if r.get('modality') == 'audio'}
    sharing = []
    for path in sorted((session_dir / 'collection').glob('*/*/*')):
        parsed = parse_recording_name(path.name) if path.is_file() else None
        if parsed and abs(parsed['start'] - measured['start_time']) < SAME_START and path.resolve() != own:
            sharing.append((path, parsed))
    personal = (measured.get('scope') or scopes.get(own)) == 'personal'
    videos = [] if personal else [path for path, parsed in sharing if parsed['modality'] == 'video']
    moves, stays = [Path(measured['path'])] + videos, []
    for path, parsed in sharing:
        if path in videos:
            continue
        if ref is not None and path.resolve() == ref:
            why = 'the reference'
        elif parsed['modality'] == 'video':
            why = 'the measured recording is a per-person microphone, which no video goes with'
        elif (scopes.get(path.resolve()) or default_audio_scope(parsed['device'], None, parsed['host'])) == 'personal':
            why = 'a per-person microphone'
        else:
            why = 'another audio recording, which goes by its own measurement'
        stays.append((path, why))
    return moves, stays


def apply_shift(session_dir: Path, measured: dict[str, Any], lag: float, reference: str | None = None,
                log=print) -> list[str]:
    """the recording `measured` {path, host, start_time} and the videos that go with it
    (shift_plan) renamed to start `lag` seconds later; what shares their start and stays is said"""
    from openmmla.commands.ses.tidy import parse_recording_name, rebuild_manifests
    moves, stays = shift_plan(session_dir, measured, reference)
    renamed = []
    for path in moves:
        parsed = parse_recording_name(path.name)
        target = path.with_name(f"{parsed['modality']}_{parsed['host']}_{parsed['device']}_{parsed['start'] + lag:.3f}.{parsed['ext']}")
        path.rename(target)
        renamed.append(target.name)
        log(f"  {path.name} -> {target.name}")
    for path, why in stays:
        log(f"  {path.name} starts with it and stays: {why}")
    rebuild_manifests(session_dir, notes=[f"start of {', '.join(renamed)} moved by {lag:+.2f} s, measured by cross-correlation"
                                          + (f"; left in place though starting with it: {', '.join(p.name for p, _ in stays)}" if stays else '')],
                      log=log)
    return renamed


def last_keyframe_before(video: str, seconds: float) -> float:
    """the time of the last keyframe at or before `seconds` into the video (0 when none is found)"""
    if seconds <= 0:
        return 0.0
    lower = max(0.0, seconds - 12.0)
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-read_intervals', f'{lower:.3f}%{seconds + 0.01:.3f}',
                          '-skip_frame', 'nokey', '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', video],
                         capture_output=True, text=True, timeout=600).stdout
    times = [float(line.strip().rstrip(',')) for line in out.splitlines() if line.strip().rstrip(',')]
    before = [t for t in times if t <= seconds + 1e-3]
    return max(before) if before else 0.0


def cut_video(source: str, destination: str, seconds: float) -> None:
    """the video from a keyframe on, streams copied"""
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{seconds:.3f}', '-i', source, '-map', '0', '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero', destination], check=True, timeout=3600)


def cut_audio(source: str, destination: str, seconds: float) -> float:
    import soundfile as sf
    with sf.SoundFile(source) as f:
        rate = f.samplerate
        f.seek(min(f.frames, int(round(seconds * rate))))
        data = f.read(dtype='int16', always_2d=True)[:, 0]
    sf.write(destination, data, rate, subtype='PCM_16')
    return len(data) / rate


def trim_session(session_dir: Path, log=print, cut_video_fn=cut_video, cut_audio_fn=cut_audio,
                 keyframe_fn=last_keyframe_before) -> dict[str, Any]:
    """every recording cut to the session's common start"""
    from openmmla.commands.ses.tidy import parse_recording_name, rebuild_manifests
    recordings = _recordings(session_dir)
    if not recordings:
        return {'common_start': None, 'cut': []}
    common = max(r['start_time'] for r in recordings)
    videos = [r for r in recordings if r['modality'] == 'video']
    # one video: everything starts at its keyframe, exactly together; several: each at its own
    cut_at = {}
    for r in videos:
        offset = common - r['start_time']
        cut_at[r['path']] = r['start_time'] + (keyframe_fn(r['path'], offset) if offset > 0.001 else 0.0)
    audio_start = min(cut_at.values()) if len(videos) == 1 else common
    report = []
    for r in recordings:
        path = Path(r['path'])
        parsed = parse_recording_name(path.name)
        new_start = cut_at[r['path']] if r['modality'] == 'video' else audio_start
        seconds = new_start - r['start_time']
        if seconds <= 0.001:
            continue
        target = path.with_name(f"{parsed['modality']}_{parsed['host']}_{parsed['device']}_{new_start:.3f}.{parsed['ext']}")
        temp = path.with_name(f".{target.stem}.cutting.{parsed['ext']}")
        log(f"  {path.name}: first {seconds:.2f} s dropped -> {target.name}")
        if r['modality'] == 'video':
            cut_video_fn(str(path), str(temp), seconds)
        else:
            cut_audio_fn(str(path), str(temp), seconds)
        path.unlink()
        temp.rename(target)
        report.append({'host': r['host'], 'modality': r['modality'], 'dropped_seconds': round(seconds, 3), 'file': target.name})
    rebuild_manifests(session_dir, notes=[f"recordings cut to the common start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(audio_start))}"
                                          + (f" (video at its keyframe, within {max(common - v for v in cut_at.values()):.2f} s)" if videos else "")], log=log)
    return {'common_start': audio_start, 'cut': report}


def cut_video_end(source: str, destination: str, seconds: float) -> float | None:
    """the video's first `seconds`, streams copied (it ends within a few frames after the cut, the
    copy stopping in decode order); returns the new length ffprobe reads"""
    from openmmla.commands.ses.imp import probe
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', source, '-map', '0', '-c', 'copy', '-t', f'{seconds:.3f}',
                    destination], check=True, timeout=3600)
    return probe(destination).get('duration')


AUDIO_DTYPES = {'PCM_16': 'int16', 'PCM_24': 'int32', 'PCM_32': 'int32', 'FLOAT': 'float32', 'DOUBLE': 'float64'}


def cut_audio_end(source: str, destination: str, seconds: float) -> float:
    """the audio's first `seconds`, to the sample, every channel and the sample format kept;
    returns the new length"""
    import soundfile as sf
    with sf.SoundFile(source) as f:
        rate, subtype, fmt = f.samplerate, f.subtype, f.format
        data = f.read(min(f.frames, int(round(seconds * rate))), dtype=AUDIO_DTYPES.get(subtype, 'float64'), always_2d=True)
    sf.write(destination, data, rate, subtype=subtype, format=fmt)
    return len(data) / rate


def _iso(epoch: float) -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch))


def _set_durations(session_dir: Path, durations: dict[tuple, float]) -> None:
    """the new length of each cut recording (by modality, start and device) in every manifest, so
    the rebuild keeps it instead of the full length"""
    from openmmla.commands.ses.tidy import _read, _recording_keys, _write
    for path in [session_dir / 'manifest.json'] + sorted((session_dir / 'collection').glob('*/manifest.json')):
        data = _read(path)
        changed = False
        for r in data.get('recordings', []):
            if not isinstance(r, dict):
                continue
            hit = next((durations[key] for key in _recording_keys(r) if key in durations), None)
            if hit is not None:
                r['duration'] = round(hit, 3)
                r['stopped_at'] = round(float(r.get('start_time') or 0) + hit, 3)
                changed = True
        if changed:
            _write(path, data)


def end_session(session_dir: Path, seconds: float, dry_run: bool = False, log=print, cut_video_fn=cut_video_end,
                cut_audio_fn=cut_audio_end) -> dict[str, Any]:
    """every recording cut to end `seconds` after the session's common start, the full-length
    originals kept under raw/; with dry_run only what would change is said"""
    from openmmla.commands.ses.imp import probe
    from openmmla.commands.ses.tidy import parse_recording_name, rebuild_manifests
    if seconds <= 0:
        raise ValueError(f"--end wants a positive number of seconds, not {seconds}")
    recordings = _recordings(session_dir)
    if not recordings:
        return {'common_start': None, 'end': None, 'cut': [], 'dry_run': dry_run}
    start = max(r['start_time'] for r in recordings)
    end = start + seconds
    plan = []
    for r in recordings:
        path = Path(r['path'])
        duration = r.get('duration') or probe(str(path)).get('duration')
        if not duration:
            log(f"  {path.name}: length unknown, left as is")
            continue
        if r['start_time'] + duration <= end + END_SLACK:
            continue
        # every recording starts by the common start, so something of each is kept
        keep = end - r['start_time']
        raw = session_dir / 'raw' / path.relative_to(session_dir / 'collection')
        if raw.exists():
            raise FileExistsError(f"{raw} exists: an earlier cut's original is there")
        plan.append((r, path, raw, keep, duration))
    verb = 'would be' if dry_run else 'is'
    log(f"  common start {_iso(start)}, end {_iso(end)} ({seconds:g} s later)")
    for r, path, raw, keep, duration in plan:
        log(f"  {path.name} {verb} cut from {duration:.2f} s to {keep:.2f} s; the original under {raw.relative_to(session_dir)}")
    if not plan:
        log("  every recording already ends by then, nothing cut")
    report = [{'host': r['host'], 'modality': r['modality'], 'device': r.get('device'), 'file': path.name,
               'kept_seconds': round(keep, 3), 'dropped_seconds': round(duration - keep, 3),
               'raw': str(raw.relative_to(session_dir))} for r, path, raw, keep, duration in plan]
    if dry_run or not plan:
        return {'common_start': start, 'end': end, 'cut': report, 'dry_run': dry_run}
    durations, done = {}, []
    try:
        for r, path, raw, keep, duration in plan:
            raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(raw))
            temp = path.with_name(f".{path.stem}.cutting{path.suffix}")
            try:
                cut = (cut_video_fn if r['modality'] == 'video' else cut_audio_fn)(str(raw), str(temp), keep)
            except BaseException:
                # the original goes back where it was, nothing half-cut is left
                temp.unlink(missing_ok=True)
                shutil.move(str(raw), str(path))
                raise
            temp.rename(path)
            parsed = parse_recording_name(path.name)
            durations[(r['modality'], round(parsed['start'], 3), parsed['device'])] = float(cut or keep)
            done.append(path.name)
    finally:
        # what was cut before a failure is in the manifests too, so a second run skips it
        if done:
            _set_durations(session_dir, durations)
            rebuild_manifests(session_dir, notes=[f"recordings cut to end {seconds:g} s after the common start {_iso(start)}, "
                                                  f"at {_iso(end)} (audio to the sample, video by stream copy, within a few "
                                                  f"frames after): {', '.join(done)}; the full-length originals under raw/"],
                              log=log)
    return {'common_start': start, 'end': end, 'cut': report, 'dry_run': False}


# the edits of one device's audio file (--warp, --cut, --cut-zeros, --cut-head): samples are cut
# or silence inserted where the timing changes, nothing is resampled (a stretch would change the
# pitch of speech), the start stamp of the name stays, and the original is kept under raw/
MIN_ZERO_RUN = 0.01          # seconds of digital zero a --cut-zeros run needs: 2025-06-16's clock steps left runs of 49 ms and 1.36 s in each vimo, whose ~550 other zero runs last 1-10 ms
ZERO_RUN_SEARCH = 0.5        # seconds from the time given within which a --cut-zeros run must start
NOTE_KNOTS = 12              # knots of a warp the manifest note lists; a measured lag track has hundreds (the 2025-10/11 vimo pairs 219-1052), which only the .edits.json keeps


def warp_pieces(frames: int, steps: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """(first input sample, end input sample, output position) of each stretch a file keeps, for
    steps (input sample, shift in samples): from each step's sample on, the input sits `shift`
    samples later than it is (the first step's shift holds from the start; of two steps at one
    sample the later given wins). A stretch that would land on output already written loses its
    head (samples cut), one that lands past it leaves silence (samples inserted)."""
    ordered = sorted(dict(steps).items())
    if not ordered:
        return [(0, frames, 0)] if frames else []
    starts = [0] + [max(0, min(frames, first)) for first, _ in ordered[1:]]
    ends = starts[1:] + [frames]
    pieces, written = [], 0
    for first, end, (_, shift) in zip(starts, ends, ordered):
        position = first + shift
        if position < written:
            first, position = first + written - position, written
        if first < end:
            pieces.append((first, end, position))
            written = position + end - first
    return pieces


def render_pieces(data: np.ndarray, pieces: list[tuple[int, int, int]]) -> np.ndarray:
    """the output the pieces make of `data` (frames x channels), silence between them"""
    length = max((position + end - first for first, end, position in pieces), default=0)
    out = np.zeros((length, data.shape[1]), dtype=data.dtype)
    for first, end, position in pieces:
        out[position:position + end - first] = data[first:end]
    return out


def cut_steps(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """the steps that cut these sample ranges [first, end) out of a file, overlapping ones merged"""
    merged: list[list[int]] = []
    for first, end in sorted(ranges):
        if end <= first:
            continue
        if merged and first <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([first, end])
    steps, cut = [(0, 0)], 0
    for first, end in merged:
        cut += end - first
        steps.append((first, -cut))
    return steps


def zero_runs(data: np.ndarray, rate: int) -> list[tuple[int, int]]:
    """the stretches [first, end) where every channel is digital zero for MIN_ZERO_RUN or longer"""
    silent = np.all(data == 0, axis=1).astype(np.int8)
    edges = np.diff(np.concatenate([[0], silent, [0]]))
    firsts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = ends - firsts >= int(round(MIN_ZERO_RUN * rate))
    return list(zip(firsts[keep].tolist(), ends[keep].tolist()))


def zero_run_at(data: np.ndarray, rate: int, at: int, length: int | None = None) -> tuple[int, int]:
    """the digital-zero run [first, end) that starts nearest sample `at`, within ZERO_RUN_SEARCH, cut
    to its first `length` samples when given; ValueError when there is none"""
    near = [(abs(first - at), first, end) for first, end in zero_runs(data, rate)
            if abs(first - at) <= ZERO_RUN_SEARCH * rate]
    if not near:
        raise ValueError(f"no digital-zero run of {MIN_ZERO_RUN * 1000:.0f} ms or more starts within "
                         f"{ZERO_RUN_SEARCH} s of {at / rate:.3f} s")
    _, first, end = min(near)
    if length is not None:
        if length > end - first:
            raise ValueError(f"the zero run at {first / rate:.3f} s lasts {(end - first) / rate:.3f} s, "
                             f"less than the {length / rate:.3f} s to cut")
        end = first + length
    return first, end


def read_time_map(value: str) -> tuple[list[tuple[float, float]], str | None]:
    """(time, lag) steps and the map's own note, from a JSON file or inline JSON: {"lags": [[time,
    lag], ...]} (from `time`, in seconds into the file as it is, on, its content sits `lag` seconds
    later), {"knots": [[t_in, t_out], ...]} (the stretch from t_in to the next knot starts at
    t_out), or a list of {"time", "lag"} or {"t_in", "t_out"} objects; an optional "note" says how
    the map was measured"""
    text = Path(value).read_text(encoding='utf-8') if os.path.isfile(value) else value
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"--warp wants a JSON file or inline JSON: {error}") from None
    note = data.get('note') if isinstance(data, dict) else None
    if isinstance(data, dict) and 'lags' in data:
        pairs = [(float(t), float(lag)) for t, lag in data['lags']]
    elif isinstance(data, dict) and 'knots' in data:
        pairs = [(float(t_in), float(t_out) - float(t_in)) for t_in, t_out in data['knots']]
    elif isinstance(data, list) and data and all(isinstance(k, dict) for k in data):
        pairs = [(float(k['time']), float(k['lag'])) if 'lag' in k else (float(k['t_in']), float(k['t_out']) - float(k['t_in']))
                 for k in data]
    else:
        raise ValueError('--warp wants {"lags": [[time, lag], ...]}, {"knots": [[t_in, t_out], ...]} or a list of such objects')
    times = [t for t, _ in pairs]
    if not pairs or any(not np.isfinite(v) for pair in pairs for v in pair) or times[0] < 0 \
            or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("--warp wants finite times from 0 on, each later than the one before")
    return pairs, note


def parse_cut(value: str, in_samples: bool = False) -> tuple[float, float]:
    """START:END or START+LENGTH, in seconds (in samples with --samples), as (start, end)"""
    for sign in (':', '+'):
        first, found, second = str(value).partition(sign)
        if found:
            try:
                start, other = (int(first), int(second)) if in_samples else (float(first), float(second))
            except ValueError:
                break
            end = start + other if sign == '+' else other
            if start < 0 or end <= start:
                break
            return start, end
    raise ValueError(f"--cut wants START:END or START+LENGTH ({'samples' if in_samples else 'seconds'}), not {value!r}")


def parse_zero_run(value: str, in_samples: bool = False) -> tuple[float, float | None]:
    """TIME or TIME:LENGTH, in seconds (in samples with --samples)"""
    at, _, length = str(value).partition(':')
    try:
        number = int if in_samples else float
        parsed = (number(at), number(length) if length else None)
    except ValueError:
        parsed = (-1, None)
    if parsed[0] < 0 or (parsed[1] is not None and parsed[1] <= 0):
        raise ValueError(f"--cut-zeros wants TIME or TIME:LENGTH ({'samples' if in_samples else 'seconds'}), not {value!r}")
    return parsed


def find_audio(recordings: list[dict[str, Any]], spec: str) -> dict[str, Any]:
    """the one audio recording [HOST/]DEVICE names; ValueError when none or several do"""
    host, _, device = str(spec).strip().rpartition('/')
    hits = [r for r in recordings if r.get('modality') == 'audio' and r.get('device') == device
            and (not host or r.get('host') == host)]
    if len(hits) != 1:
        found = ', '.join(f"{r.get('host')}/{r.get('device')}" for r in recordings if r.get('modality') == 'audio')
        raise ValueError(f"{spec} names {'no' if not hits else len(hits)} audio recording(s) of the session "
                         f"({'give HOST/DEVICE' if hits else 'it has ' + (found or 'none')})")
    return hits[0]


def _seconds(samples: int, rate: int) -> str:
    return f"{samples / rate:.3f}"


def edit_audio(session_dir: Path, devices: list[str], *, warp: tuple[list[tuple[float, float]], str | None] | None = None,
               cuts: list[tuple[float, float]] = (), zeros: list[tuple[float, float | None]] = (),
               head: float | None = None, in_samples: bool = False, note: str | None = None,
               dry_run: bool = False, log=print) -> dict[str, Any]:
    """the audio files of `devices` ([HOST/]DEVICE) re-timed in place, each by its own samples:
    `warp` (time, lag) steps, or cuts of sample ranges, of the digital-zero runs at given times and
    of the first `head` seconds. The start stamp of each name stays; the original moves under
    raw/<host>/audio/ (an earlier edit's original stays there, the first one), next to a
    <name>.edits.json of every edit made, and the manifests are rebuilt with the new lengths and a
    note of what was done. With dry_run only what would change is said."""
    import soundfile as sf
    from openmmla.commands.ses.tidy import rebuild_manifests
    if warp is not None and (cuts or zeros or head):
        raise ValueError("--warp goes alone, without --cut, --cut-zeros and --cut-head")
    if warp is None and not (cuts or zeros or head):
        raise ValueError("nothing to do: give --warp, --cut, --cut-zeros or --cut-head")
    if head is not None and head <= 0:
        raise ValueError(f"--cut-head wants a positive number of seconds, not {head}")
    recordings = _recordings(session_dir)
    targets = [find_audio(recordings, spec) for spec in devices]
    if not targets:
        raise ValueError("--devices names no recording to change")
    plans = []
    for r in targets:
        path = Path(r['path'])
        with sf.SoundFile(str(path)) as f:
            rate, subtype, fmt = f.samplerate, f.subtype, f.format
            data = f.read(dtype=AUDIO_DTYPES.get(subtype, 'float64'), always_2d=True)
        to_samples = (lambda v: int(v)) if in_samples else (lambda v: int(round(v * rate)))
        if warp is not None:
            steps = [(int(round(t * rate)), int(round(lag * rate))) for t, lag in warp[0]]
            ranges = []
        else:
            ranges = [(0, int(round(head * rate)))] if head else []
            ranges += [(to_samples(a), to_samples(b)) for a, b in cuts]
            ranges += [zero_run_at(data, rate, to_samples(at), None if length is None else to_samples(length))
                       for at, length in zeros]
            if any(end > len(data) for _, end in ranges):
                raise ValueError(f"{path.name}: a cut reaches past its end ({_seconds(len(data), rate)} s)")
            steps = cut_steps(ranges)
        pieces = warp_pieces(len(data), steps)
        kept = sum(end - first for first, end, _ in pieces)
        length = max((position + end - first for first, end, position in pieces), default=0)
        raw = session_dir / 'raw' / path.relative_to(session_dir / 'collection')
        plans.append({'record': r, 'path': path, 'raw': raw, 'rate': rate, 'subtype': subtype, 'format': fmt,
                      'data': data, 'steps': steps, 'ranges': sorted(ranges), 'pieces': pieces,
                      'cut': len(data) - kept, 'inserted': length - kept, 'frames': len(data), 'length': length})
    kind = 'warp' if warp is not None else 'cut'
    verb = 'would be' if dry_run else 'is'
    for plan in plans:
        rate = plan['rate']
        what = (f"warped at {len(plan['steps'])} steps" if kind == 'warp'
                else f"cut at {', '.join(f'[{a}, {b})' for a, b in plan['ranges'])} (samples at {rate} Hz)")
        log(f"  {plan['path'].name} {verb} {what}: {_seconds(plan['cut'], rate)} s cut, {_seconds(plan['inserted'], rate)} s "
            f"of silence inserted, {_seconds(plan['frames'], rate)} s -> {_seconds(plan['length'], rate)} s; the start "
            f"stamp kept, the original {'already ' if plan['raw'].exists() else ''}under {plan['raw'].relative_to(session_dir)}")
        if kind == 'warp':
            previous = 0
            for index, (sample, shift) in enumerate(sorted(plan['steps'])):
                change = shift - previous if index else shift
                previous = shift
                if change:
                    log(f"    at {_seconds(sample if index else 0, rate)} s: {'+' if change > 0 else '-'}{_seconds(abs(change), rate)} s "
                        f"({'silence inserted' if change > 0 else 'cut'}), from then on {shift / rate:+.3f} s")
    report = [{'host': p['record']['host'], 'device': p['record'].get('device'), 'file': p['path'].name,
               'raw': str(p['raw'].relative_to(session_dir)), 'cut_seconds': round(p['cut'] / p['rate'], 4),
               'inserted_seconds': round(p['inserted'] / p['rate'], 4), 'seconds': round(p['length'] / p['rate'], 4),
               'ranges': [list(r) for r in p['ranges']], 'steps': [list(s) for s in sorted(p['steps'])], 'rate': p['rate']}
              for p in plans]
    if dry_run:
        return {'edited': report, 'dry_run': True}

    from openmmla.commands.ses.tidy import parse_recording_name
    durations, done = {}, []
    when = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    try:
        for plan, entry in zip(plans, report):
            path, raw = plan['path'], plan['raw']
            kept_before = raw.exists()
            if not kept_before:
                raw.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(raw))
            temp = path.with_name(f".{path.stem}.editing{path.suffix}")
            try:
                sf.write(str(temp), render_pieces(plan['data'], plan['pieces']), plan['rate'], subtype=plan['subtype'], format=plan['format'])
            except BaseException:
                # the original goes back where it was, nothing half-written is left
                temp.unlink(missing_ok=True)
                if not kept_before:
                    shutil.move(str(raw), str(path))
                raise
            temp.replace(path)
            log_path = raw.with_name(raw.name + '.edits.json')
            edits = json.loads(log_path.read_text(encoding='utf-8')) if log_path.exists() else []
            edits.append({'at': when, 'edit': kind, 'from_seconds': round(plan['frames'] / plan['rate'], 4),
                          **{k: entry[k] for k in ('seconds', 'cut_seconds', 'inserted_seconds', 'ranges', 'steps', 'rate')},
                          **({'map': [list(pair) for pair in warp[0]], 'map_note': warp[1]} if warp is not None else {}),
                          **({'note': note} if note else {})})
            log_path.write_text(json.dumps(edits, indent=2) + "\n", encoding='utf-8')
            parsed = parse_recording_name(path.name)
            durations[('audio', round(parsed['start'], 3), parsed['device'])] = plan['length'] / plan['rate']
            done.append(entry)
    finally:
        if done:
            _set_durations(session_dir, durations)
            rebuild_manifests(session_dir, notes=[_edit_note(kind, done, warp, head, note)], log=log)
    return {'edited': report, 'dry_run': False}


def _edit_note(kind: str, done: list[dict[str, Any]], warp, head: float | None, note: str | None) -> str:
    names = ', '.join(e['device'] for e in done)
    tail = (f"; the start stamps kept, nothing resampled, the originals under raw/ with an .edits.json each"
            + (f"; {note}" if note else ''))
    if kind == 'warp':
        lags = [lag for _, lag in warp[0]]
        pairs = (', '.join(f"{t:.3f}:{lag:+.3f}" for t, lag in warp[0]) if len(warp[0]) <= NOTE_KNOTS
                 else f"{len(warp[0])} knots from {warp[0][0][0]:.3f}:{lags[0]:+.3f} to {warp[0][-1][0]:.3f}:{lags[-1]:+.3f}, "
                      f"the lag between {min(lags):+.3f} and {max(lags):+.3f} (every knot in the .edits.json)")
        totals = '; '.join(f"{e['device']} {e['cut_seconds']:.3f} s cut, {e['inserted_seconds']:.3f} s of silence inserted"
                           for e in done)
        return (f"{names} re-timed by a piecewise time map, samples cut or silence inserted "
                f"where the lag changes (s into the file as it was: s later): {pairs}"
                + (f" ({warp[1]})" if warp[1] else '') + f"; {totals}" + tail)
    parts = [f"{e['device']} {e['cut_seconds']:.4f} s at {', '.join(f'[{a}, {b})' for a, b in e['ranges'])} ({e['rate']} Hz)"
             for e in done]
    lead = f"the first {head:.3f} s cut, so what follows sits {head:.3f} s earlier; " if head else ''
    return f"{names}: {lead}samples cut, [first, end) of the file as it was: {'; '.join(parts)}" + tail


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mmla ses-align',
        description="Measure how far a session's recordings are off one clock (by cross-correlating the audio), "
                    "move them by that much, and cut them to one common start; or re-time the audio inside a file.")
    parser.add_argument('session', help="the session folder under artifacts/, or its id")
    parser.add_argument('--reference', default=None, help="the host whose clock the others are aligned to (default: the longest per-person mic)")
    parser.add_argument('--apply', action='store_true', help="move the recordings whose start is measured off by more than --tolerance")
    parser.add_argument('--tolerance', type=float, default=0.25, help="seconds of offset left alone (default 0.25)")
    parser.add_argument('--min-confidence', type=float, default=8.0, help="how far the correlation peak must stand above the rest (default 8)")
    parser.add_argument('--trim', action='store_true', help="cut every recording to the session's common start")
    parser.add_argument('--end', type=float, default=None, metavar='SECONDS',
                        help="cut every recording to end SECONDS after the common start, the originals kept under raw/")
    parser.add_argument('--dry-run', action='store_true', help="with --end or an edit: say what would change, change nothing")
    edits = parser.add_argument_group(
        'edits of audio files', "re-time the audio of --devices by cutting samples or inserting silence (nothing is "
        "resampled); the start stamp in the name stays, the original moves under raw/, the manifest notes what was done")
    edits.add_argument('--devices', action='append', default=[], metavar='[HOST/]DEVICE[,...]',
                       help="the audio recordings to edit (vimo-0-ch1, ericli/vimo-0; repeatable)")
    edits.add_argument('--warp', default=None, metavar='MAP',
                       help='a piecewise time map, a JSON file or inline JSON: {"lags": [[time, lag], ...]} (from time, '
                            'in s into the file as it is, on, its content sits lag s later; the first lag holds from the '
                            'start) or {"knots": [[t_in, t_out], ...]}, optionally with a "note"')
    edits.add_argument('--cut', action='append', default=[], metavar='START:END',
                       help="cut this stretch, START:END or START+LENGTH in s into the file as it is (repeatable)")
    edits.add_argument('--cut-zeros', action='append', default=[], metavar='TIME[:LENGTH]',
                       help=f"cut the digital-zero run that starts within {ZERO_RUN_SEARCH} s of TIME, all of it or its "
                            f"first LENGTH (repeatable); refused when there is none")
    edits.add_argument('--cut-head', type=float, default=None, metavar='SECONDS',
                       help="cut the first SECONDS and keep the start stamp, so the rest sits SECONDS earlier "
                            "(a recording measured SECONDS late; --trim is the cut that keeps the timing)")
    edits.add_argument('--samples', action='store_true', help="--cut and --cut-zeros positions are sample indices, not seconds")
    edits.add_argument('--note', default=None, help="said in the manifest note of the edit (why, how it was measured)")
    parser.add_argument('--window', type=float, default=1200.0, help="seconds of audio compared (default 1200)")
    parser.add_argument('--max-lag', type=float, default=120.0, help="largest offset looked for, in seconds (default 120)")
    parser.add_argument('-a', '--artifacts', default=None, help="artifacts root (default <cwd>/artifacts)")
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    session_dir = Path(args.session).expanduser()
    if not session_dir.is_dir():
        session_dir = Path(args.artifacts or os.path.join(os.getcwd(), 'artifacts')) / args.session
    if not session_dir.is_dir():
        print(f"no session at {session_dir}")
        return 1
    session_dir = session_dir.resolve()
    editing = bool(args.warp or args.cut or args.cut_zeros or args.cut_head is not None)
    if editing and (args.apply or args.trim or args.end is not None):
        print("--warp, --cut, --cut-zeros and --cut-head go without --apply, --trim and --end")
        return 1
    if args.dry_run and not editing and (args.end is None or args.apply or args.trim):
        print("--dry-run goes with --end alone, or with an edit")
        return 1
    if args.end is not None and args.end <= 0:
        print(f"--end wants a positive number of seconds, not {args.end}")
        return 1
    if editing or args.devices:
        try:
            devices = [d for value in args.devices for d in value.split(',') if d.strip()]
            if not devices:
                raise ValueError("an edit wants --devices")
            if not editing:
                raise ValueError("--devices goes with --warp, --cut, --cut-zeros or --cut-head")
            warp = read_time_map(args.warp) if args.warp else None
            cuts = [parse_cut(v, args.samples) for v in args.cut]
            zeros = [parse_zero_run(v, args.samples) for v in args.cut_zeros]
            print(session_dir.name)
            edit_audio(session_dir, devices, warp=warp, cuts=cuts, zeros=zeros, head=args.cut_head,
                       in_samples=args.samples, note=args.note, dry_run=args.dry_run)
        except (ValueError, FileNotFoundError) as error:
            print(error)
            return 1
        return 0
    print(session_dir.name)
    # --end alone needs no measuring
    measured = measure_session(session_dir, args.reference, args.window, args.max_lag) \
        if args.end is None or args.apply else {'reference': None, 'results': [], 'reason': None}
    if measured['reference'] is None and measured['reason']:
        print(f"  {measured['reason']}")
    elif measured['reference'] is not None:
        print(f"  reference: {measured['reference']}")
        for r in measured['results']:
            lag = 'no result' if r['lag'] is None else f"{r['lag']:+.2f} s"
            extra = r.get('reason') or f"confidence {r['confidence']}, margin {r.get('margin')}, over {r.get('compared_seconds')} s"
            print(f"  {r['host']}/{r.get('device') or '?':12} off by {lag:>10}  ({extra}); shares its start with {', '.join(r['shares_start_with']) or 'nothing'}")
    if args.apply and measured['reference'] is not None:
        moved = False
        for r in measured['results']:
            if r['lag'] is None or abs(r['lag']) <= args.tolerance or r['confidence'] < args.min_confidence:
                continue
            apply_shift(session_dir, r, r['lag'], reference=measured['reference_path'])
            moved = True
        if not moved:
            print("  nothing moved")
    if args.trim:
        result = trim_session(session_dir)
        if not result['cut']:
            print("  already on one start, nothing cut")
    if args.end is not None:
        end_session(session_dir, args.end, dry_run=args.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
