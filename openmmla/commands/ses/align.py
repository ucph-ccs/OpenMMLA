"""mmla ses-align: the recordings of a session on one clock and one start.

A session's recordings come from different clocks: a per-person microphone (badge, vimo) is
stamped by the base that recorded it, a camera's file by the machine that wrote it, at best to
the second and after the delay of a stream. The camera's own audio track and the microphones
heard the same room, so cross-correlating them says how far the camera's nominal start is off,
and --apply moves the camera's recordings (its video and its audio track share one start) by
that much. --trim then cuts every recording to the session's common start, the latest start
among them: audio to the sample, video at the last keyframe before it (a stream copy, nothing
re-encoded), and names every file with its exact start. Nothing is kept: what came before the
common start is gone.
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
                        'shares_start_with': sorted(r['host'] + '/' + r['modality'] for r in recordings
                                                    if r is not other and abs(r['start_time'] - other['start_time']) < 0.0015),
                        **result})
    return {'reference': reference['host'], 'results': results}


def apply_shift(session_dir: Path, start_time: float, lag: float, log=print) -> list[str]:
    """every recording that starts at `start_time` (a camera's video and its audio track) renamed
    to start `lag` seconds later"""
    from openmmla.commands.ses.tidy import parse_recording_name, rebuild_manifests
    renamed = []
    for path in sorted((session_dir / 'collection').glob('*/*/*')):
        parsed = parse_recording_name(path.name) if path.is_file() else None
        if parsed and abs(parsed['start'] - start_time) < 0.0015:
            target = path.with_name(f"{parsed['modality']}_{parsed['host']}_{parsed['device']}_{parsed['start'] + lag:.3f}.{parsed['ext']}")
            path.rename(target)
            renamed.append(target.name)
            log(f"  {path.name} -> {target.name}")
    rebuild_manifests(session_dir, notes=[f"start of {', '.join(renamed)} moved by {lag:+.2f} s, measured by cross-correlation"], log=log)
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


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mmla ses-align',
        description="Measure how far a session's recordings are off one clock (by cross-correlating the audio), "
                    "move them by that much, and cut them to one common start.")
    parser.add_argument('session', help="the session folder under artifacts/, or its id")
    parser.add_argument('--reference', default=None, help="the host whose clock the others are aligned to (default: the longest per-person mic)")
    parser.add_argument('--apply', action='store_true', help="move the recordings whose start is measured off by more than --tolerance")
    parser.add_argument('--tolerance', type=float, default=0.25, help="seconds of offset left alone (default 0.25)")
    parser.add_argument('--min-confidence', type=float, default=8.0, help="how far the correlation peak must stand above the rest (default 8)")
    parser.add_argument('--trim', action='store_true', help="cut every recording to the session's common start")
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
    print(session_dir.name)
    measured = measure_session(session_dir, args.reference, args.window, args.max_lag)
    if measured['reference'] is None:
        print(f"  {measured['reason']}")
    else:
        print(f"  reference: {measured['reference']}")
        for r in measured['results']:
            lag = 'no result' if r['lag'] is None else f"{r['lag']:+.2f} s"
            extra = r.get('reason') or f"confidence {r['confidence']}, margin {r.get('margin')}, over {r.get('compared_seconds')} s"
            print(f"  {r['host']}/{r.get('device') or '?':12} off by {lag:>10}  ({extra}); shares its start with {', '.join(r['shares_start_with']) or 'nothing'}")
    if args.apply and measured['reference'] is not None:
        done = set()
        for r in measured['results']:
            if r['lag'] is None or abs(r['lag']) <= args.tolerance or r['confidence'] < args.min_confidence or r['start_time'] in done:
                continue
            apply_shift(session_dir, r['start_time'], r['lag'])
            done.add(r['start_time'])
        if not done:
            print("  nothing moved")
    if args.trim:
        result = trim_session(session_dir)
        if not result['cut']:
            print("  already on one start, nothing cut")
    return 0


if __name__ == '__main__':
    sys.exit(main())
