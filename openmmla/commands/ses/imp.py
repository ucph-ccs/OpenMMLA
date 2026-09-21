"""mmla ses-import: a recorded session in whatever shape an earlier run left it, brought into the
collection layout the pipelines replay with source: file.

The collection layout is artifacts/<session>/collection/<host>/{audio,video}/<kind>_<host>_<device>_<unix start>.<ext>
with a manifest per host and one for the session (openmmla.collection.recording). What earlier
runs left instead: a video named by the local time it started (OBS, QuickTime: '2025-05-13 11-01-06.mov',
'record_20250616_124506.mp4'), a video or audio file already named with its unix start
('record_1759826194.755_raspi4-01.mp4', 'audio_1759832419.338_ch0.wav'), and an ASR base's
folder of three-second recordings ('badge_0/records/badge_0_record_<unix>.wav', or 'segments/'
when the raw records were not kept), sometimes downloaded twice ('... (2).wav'). This command
finds them, moves the continuous files under their new names, concatenates a base's segments into
one file that starts when the session's first segment did (silence where nothing was recorded),
extracts a video's own audio track as a group microphone, writes the manifests, keeps the rest of
the folder as legacy/<original name>/ and a report of what went where.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
AUDIO_EXTS = ('.wav', '.m4a', '.mp3', '.flac', '.aac')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
SEGMENT_FOLDERS = ('records', 'segments', 'chunks')  # in order of preference: raw before trimmed
LEFTOVER_FOLDERS = ('profiles', 'temp', 'logger', 'logs', 'visualizations')  # a base's other folders: never recordings
MIN_SEGMENTS = 5
SEGMENT_RATE = 16000
DEFAULT_TZ = 'Europe/Copenhagen'
DEFAULT_EXPERIMENT = 'exp_wegrow_life'
DEFAULT_GROUP = 'group_01'

UNIX_RE = re.compile(r'(?<!\d)(\d{10}(?:\.\d+)?)(?!\d)')
LOCAL_DASH_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})-(\d{2})-(\d{2})')
LOCAL_COMPACT_RE = re.compile(r'(?<!\d)(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?!\d)')
HOST_RE = re.compile(r'(?<![a-z0-9])((?:raspi|pi|mac|dell|desktop|nuc)[a-z0-9]*-\d+)', re.I)
CHANNEL_RE = re.compile(r'_ch(\d+)(?=\.)', re.I)
COPY_RE = re.compile(r' \(\d+\)(?=\.[A-Za-z0-9]+$)')
SESSION_DIR_RE = re.compile(r'session_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})Z')


def _safe(label: str) -> str:
    """a host or device label the file names and folder names can carry"""
    text = re.sub(r'[^A-Za-z0-9]+', '-', str(label)).strip('-').lower()
    return text or 'item'


def unix_in_name(name: str) -> float | None:
    """the unix start time a file name carries ('..._1759826194.755_raspi4-01.mp4'), else None"""
    match = UNIX_RE.search(name)
    return float(match.group(1)) if match else None


def local_time_in_name(name: str, tz: str) -> float | None:
    """the local start time a file name carries ('2025-05-13 11-01-06.mov', 'record_20250616_124506.mp4'),
    as unix time, else None"""
    match = LOCAL_DASH_RE.search(name) or LOCAL_COMPACT_RE.search(name)
    if not match:
        return None
    parts = [int(v) for v in match.groups()]
    return datetime(*parts, tzinfo=ZoneInfo(tz)).timestamp()


def session_start_in_path(path: str) -> tuple[str, float] | None:
    """(the session folder name, its unix start) when the path carries session_<ISO>Z, else None"""
    match = SESSION_DIR_RE.search(str(path))
    if not match:
        return None
    started = datetime.strptime(match.group(1), '%Y-%m-%dT%H-%M-%S').replace(tzinfo=ZoneInfo('UTC'))
    return match.group(0), started.timestamp()


def session_id_for(experiment_id: str, group_id: str, start: float) -> str:
    return f"{experiment_id}_{group_id}_{time.strftime('%y%m%dT%H%MZ', time.gmtime(start))}"


def probe(path: str) -> dict[str, Any]:
    """duration, streams and a creation time of a media file, from ffprobe; {} when it cannot say"""
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path],
                             capture_output=True, text=True, timeout=120).stdout
        info = json.loads(out or '{}')
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    fmt = info.get('format', {})
    streams = info.get('streams', [])
    audio = next((s for s in streams if s.get('codec_type') == 'audio'), None)
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    result: dict[str, Any] = {'duration': float(fmt.get('duration') or 0.0) or None}
    if video:
        result['video'] = {'codec': video.get('codec_name'), 'width': video.get('width'), 'height': video.get('height'),
                           'fps': video.get('r_frame_rate')}
    if audio:
        result['audio'] = {'codec': audio.get('codec_name'), 'sample_rate': int(audio.get('sample_rate') or 0),
                           'channels': int(audio.get('channels') or 0)}
    created = (fmt.get('tags') or {}).get('creation_time')
    if created:
        result['creation_time'] = created
    return result


@dataclass
class Item:
    """one recording of the new layout: what it is made from, and where it goes"""
    modality: str                    # 'video' | 'audio'
    method: str                      # 'move' | 'concatenate' | 'extract'
    host: str
    device: str                      # video device or audio channel label
    start: float
    sources: list[str]
    ext: str = ''
    duration: float | None = None
    pipeline_hint: str | None = None  # 'ips' | 'vfa' | 'asr' when the source folder said
    notes: list[str] = field(default_factory=list)
    destination: str = ''
    duplicates: list[str] = field(default_factory=list)  # second downloads of the same segments, dropped with the sources

    @property
    def file_name(self) -> str:
        return f"{self.modality}_{self.host}_{self.device}_{self.start:.3f}{self.ext}"

    def as_recording(self, path: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            'id': Path(path).stem, 'modality': self.modality, 'status': 'stopped', 'path': path,
            'start_time': round(self.start, 3), 'host': self.host, 'format': self.ext.lstrip('.'),
            'imported': {'method': self.method, 'sources': self.sources[:5], 'source_count': len(self.sources)},
        }
        if self.duration:
            record['duration'] = round(self.duration, 3)
            record['stopped_at'] = round(self.start + self.duration, 3)
        if self.modality == 'audio':
            record.update({'channel': self.device, 'channels': 1, 'sample_rate': SEGMENT_RATE})
        else:
            record['device'] = self.device
        if self.pipeline_hint:
            record['pipeline_hint'] = self.pipeline_hint
        if self.notes:
            record['notes'] = self.notes
        return record


@dataclass
class Plan:
    source: str
    session_dir_name: str | None
    session_start: float | None
    session_id: str
    experiment_id: str
    group_id: str
    target: str
    items: list[Item]
    warnings: list[str]
    skipped: list[str]

    def describe(self) -> str:
        lines = [f"{self.source}", f"  -> {self.target}  (session {self.session_id})"]
        if self.session_start:
            lines.append(f"  session start {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime(self.session_start))}")
        for item in sorted(self.items, key=lambda i: (i.modality, i.start)):
            when = time.strftime('%H:%M:%SZ', time.gmtime(item.start))
            length = f"{item.duration / 60:.1f} min" if item.duration else "?"
            src = item.sources[0] if len(item.sources) == 1 else f"{len(item.sources)} segments from {os.path.dirname(item.sources[0])}"
            lines.append(f"  {item.modality:5} {item.method:11} {item.file_name:60} {when} {length:>9}  <- {src}")
            for note in item.notes:
                lines.append(f"        note: {note}")
        for text in self.warnings:
            lines.append(f"  WARNING: {text}")
        for text in self.skipped:
            lines.append(f"  skipped: {text}")
        return "\n".join(lines)


def _pipeline_hint(path: str) -> str | None:
    lowered = path.lower()
    for token, hint in (('indoor_positioning', 'ips'), ('/ips/', 'ips'), ('/vfa/', 'vfa'), ('video_frame', 'vfa'), ('/asr', 'asr')):
        if token in lowered:
            return hint
    return None


def _segment_bases(root: Path, only: str | None) -> list[tuple[Path, list[Path]]]:
    """the ASR base folders under root and the segment files each one keeps, raw records first"""
    bases: dict[Path, dict[str, list[Path]]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != 'legacy' and not d.startswith('.')]
        folder = Path(dirpath)
        if folder.name not in SEGMENT_FOLDERS:
            continue
        if only and only not in str(folder):
            continue
        wavs = [folder / f for f in filenames if f.lower().endswith('.wav') and unix_in_name(f) is not None]
        if len(wavs) >= MIN_SEGMENTS:
            bases.setdefault(folder.parent, {})[folder.name] = wavs
    chosen = []
    for base, kinds in bases.items():
        for kind in SEGMENT_FOLDERS:
            if kind in kinds:
                chosen.append((base, kinds[kind]))
                break
    return sorted(chosen, key=lambda pair: str(pair[0]))


def _session_end(root: Path, session_start: float) -> float:
    """where the session's share of the folder ends: at the next session_<ISO>Z the folder names,
    else four hours on"""
    starts = set()
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != 'legacy' and not d.startswith('.')]
        found = session_start_in_path(dirpath)
        if found:
            starts.add(found[1])
    later = [s for s in starts if s > session_start + 60]
    return min(later) if later else session_start + 4 * 3600


def _is_leftover(path: Path, root: Path) -> bool:
    """a file inside a base's segment or bookkeeping folders, whichever kind was chosen"""
    return any(part in SEGMENT_FOLDERS or part in LEFTOVER_FOLDERS for part in path.relative_to(root).parts[:-1])


def _dedupe(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """one file per stamp: a download made twice keeps the plain name, drops ' (2)' and ' (3)'"""
    by_stamp: dict[str, Path] = {}
    dropped = []
    for path in sorted(files, key=lambda p: (COPY_RE.search(p.name) is not None, p.name)):
        key = COPY_RE.sub('', path.name)
        if key in by_stamp:
            dropped.append(path)
            continue
        by_stamp[key] = path
    return sorted(by_stamp.values(), key=lambda p: unix_in_name(p.name)), dropped


def _continuous_media(root: Path, only: str | None, segment_files: set[Path]) -> list[Path]:
    media = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != 'legacy' and not d.startswith('.')]
        folder = Path(dirpath)
        for name in filenames:
            path = folder / name
            if name.startswith('.') or path in segment_files or _is_leftover(path, root):
                continue
            if name.lower().endswith(VIDEO_EXTS + AUDIO_EXTS):
                media.append(path)
    return media


def _camera_labels(videos: list[tuple[float, float | None]]) -> list[str]:
    """labels for videos that name no host, in start order: two that overlap in time are two
    cameras, one that starts after the other ended is the same camera recording again"""
    labels, cameras = [], []  # cameras: [(last end)]
    for start, duration in videos:
        end = start + (duration or 0)
        for index, last_end in enumerate(cameras):
            if start >= last_end - 1.0:
                cameras[index] = end
                labels.append(f"cam-{index + 1}")
                break
        else:
            cameras.append(end)
            labels.append(f"cam-{len(cameras)}")
    return labels


def plan_import(source: str, artifacts_root: str, experiment_id: str = DEFAULT_EXPERIMENT, group_id: str = DEFAULT_GROUP,
                tz: str = DEFAULT_TZ, only: str | None = None, session_id: str | None = None,
                probe_media=probe) -> Plan:
    """what an import of `source` would do, without doing it"""
    root = Path(source).expanduser().resolve()
    warnings_, skipped, items = [], [], []
    session = session_start_in_path(str(root)) or (session_start_in_path(only) if only else None)
    session_dir_name, session_start = session if session else (None, None)

    # the ASR bases' segments: one continuous file per base, all starting with the session's first segment
    bases = _segment_bases(root, only)
    segment_files: set[Path] = set()
    base_items: list[Item] = []
    for base_dir, files in bases:
        files, dropped = _dedupe(files)
        segment_files.update(files)
        segment_files.update(dropped)
        stamps = [unix_in_name(f.name) for f in files]
        first, last = min(stamps), max(stamps)
        item = Item('audio', 'concatenate', host=_safe(base_dir.name), device='mic', start=first,
                    sources=[str(f) for f in files], ext='.wav', duration=last - first + 3.0,
                    pipeline_hint='asr')
        item.notes.append(f"{len(files)} segments from {base_dir.name}/{files[0].parent.name}")
        if files[0].parent.name != 'records':
            item.notes.append("trimmed speech segments, not raw records: the pauses between them are silence")
        if dropped:
            item.duplicates = [str(f) for f in dropped]
            item.notes.append(f"{len(dropped)} duplicate downloads dropped")
        gaps = sum(1 for a, b in zip(stamps, stamps[1:]) if b - a > 10)
        if gaps:
            item.notes.append(f"{gaps} gaps longer than 10 s, filled with silence")
        base_items.append(item)
    if base_items:
        t0 = min(item.start for item in base_items)
        for item in base_items:
            if item.start > t0 + 1.0:
                item.notes.append(f"joined {(item.start - t0) / 60:.1f} min after the first base: leading silence added")
                item.duration = (item.duration or 0) + (item.start - t0)
            item.start = t0
        items.extend(base_items)

    # continuous video and audio files
    unnamed: list[Item] = []
    for path in sorted(_continuous_media(root, only, segment_files)):
        rel = str(path.relative_to(root))
        in_a_session = session_start_in_path(rel)
        if only and in_a_session and only not in rel:
            continue  # another session's file
        name = path.name
        start = unix_in_name(name)
        origin = 'unix time in the name'
        if start is None:
            start = local_time_in_name(name, tz)
            origin = f'local time in the name ({tz})'
        if start is None:
            skipped.append(f"{rel}: no start time in the name")
            continue
        if only and not in_a_session and session_start is not None:
            # a shared file (video_recordings/ next to several sessions) belongs to the session in
            # whose share of the day it starts, or that it is still running into
            info = probe_media(str(path))
            end = start + (info.get('duration') or 0)
            session_end = _session_end(root, session_start)
            if not (session_start - 900 <= start < session_end) and not (start < session_start < end):
                continue
        else:
            info = probe_media(str(path))
        duration = info.get('duration')
        is_video = name.lower().endswith(VIDEO_EXTS)
        host_match = HOST_RE.search(name)
        if is_video:
            host = _safe(host_match.group(1)) if host_match else ''
            item = Item('video', 'move', host=host, device='video0', start=start, sources=[str(path)],
                        ext=path.suffix.lower(), duration=duration, pipeline_hint=_pipeline_hint(rel))
            item.notes.append(f"start from the {origin}")
            items.append(item)
            if info.get('audio'):
                mix = Item('audio', 'extract', host=host, device='mix', start=start, sources=[str(path)], ext='.wav',
                           duration=duration, pipeline_hint='asr')
                mix.notes.append("the video's own audio track, 16 kHz mono")
                items.append(mix)
            if not host:
                unnamed.append(item)
        else:
            if duration is not None and duration < 30:
                skipped.append(f"{rel}: {duration:.0f} s, not a continuous recording")
                continue
            channel = CHANNEL_RE.search(name)
            host = _safe(host_match.group(1)) if host_match else 'mic'
            item = Item('audio', 'move', host=host, device=f"ch{channel.group(1)}" if channel else 'mix', start=start,
                        sources=[str(path)], ext=path.suffix.lower(), duration=duration, pipeline_hint='asr')
            item.notes.append(f"start from the {origin}")
            if (info.get('audio') or {}).get('sample_rate') not in (None, 0, SEGMENT_RATE):
                item.notes.append(f"sample rate {info['audio']['sample_rate']} Hz")
            items.append(item)

    # videos that name no camera: cam-1, cam-2 ... by when they run
    unnamed.sort(key=lambda i: i.start)
    for item, label in zip(unnamed, _camera_labels([(i.start, i.duration) for i in unnamed])):
        for each in items:
            if each.sources == item.sources:
                each.host = label
    takes = {}
    for item in unnamed:
        takes.setdefault(item.host, []).append(item)
    for label, group in takes.items():
        if len(group) > 1:
            for item in group:
                item.notes.append(f"{label} recorded {len(group)} takes; a replay starts at the session's initial_sync_time")

    # sanity: every start against the session's start
    if session_start is not None:
        for item in items:
            if item.start < session_start - 900:
                warnings_.append(f"{item.file_name} starts {(session_start - item.start) / 60:.0f} min before the session: "
                                 f"check the time zone of its name")
            if item.start > session_start + 4 * 3600:
                warnings_.append(f"{item.file_name} starts {(item.start - session_start) / 3600:.1f} h after the session")
    if not items:
        warnings_.append("no recordings found")
    start_for_id = session_start if session_start is not None else (min(i.start for i in items) if items else time.time())
    sid = session_id or session_id_for(experiment_id, group_id, start_for_id)
    target = os.path.join(artifacts_root, sid)
    if os.path.exists(target):
        warnings_.append(f"{target} exists already")
    for item in items:
        item.destination = os.path.join(target, 'collection', item.host, item.modality, item.file_name)
    return Plan(str(root), session_dir_name, session_start, sid, experiment_id, group_id, target, items, warnings_, skipped)


def concatenate(files: list[str], start: float, destination: str) -> float:
    """one 16 kHz mono wav from timestamped segments, each at its own offset from `start`; silence
    where nothing was recorded, the earlier segment kept where two overlap. Returns the duration."""
    import numpy as np
    import soundfile as sf

    written = 0
    with sf.SoundFile(destination, 'w', samplerate=SEGMENT_RATE, channels=1, subtype='PCM_16') as out:
        for path in sorted(files, key=lambda p: unix_in_name(os.path.basename(p))):
            stamp = unix_in_name(os.path.basename(path))
            data, rate = sf.read(path, dtype='int16', always_2d=True)
            data = data[:, 0]
            if rate != SEGMENT_RATE:
                from openmmla.streams.resampling import resample_audio
                data = resample_audio(data.astype('float32') / 32768.0, rate, SEGMENT_RATE)
                data = np.clip(data * 32768.0, -32768, 32767).astype('int16')
            offset = int(round((stamp - start) * SEGMENT_RATE))
            if offset > written:
                out.write(np.zeros(offset - written, dtype='int16'))
                written = offset
            elif offset < written:
                data = data[written - offset:]
            if len(data):
                out.write(data)
                written += len(data)
    return written / SEGMENT_RATE


def extract_audio(video: str, destination: str) -> None:
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-vn', '-ac', '1', '-ar', str(SEGMENT_RATE),
                    '-c:a', 'pcm_s16le', destination], check=True, timeout=3600)


def execute(plan: Plan, copy: bool = False, log=print) -> dict[str, Any]:
    """the plan carried out: files moved (or copied), segments concatenated, audio tracks extracted,
    manifests written, the rest of the source kept as legacy/, and a report returned and saved"""
    from openmmla.collection.recording import update_manifest, _dump_yaml

    report: dict[str, Any] = {'source': plan.source, 'session_id': plan.session_id, 'target': plan.target,
                              'imported_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'copy': copy,
                              'recordings': [], 'warnings': list(plan.warnings), 'skipped': list(plan.skipped)}
    transfer = shutil.copy2 if copy else shutil.move
    meta = read_meta(plan)
    for item in sorted(plan.items, key=lambda i: (i.modality, i.start)):
        os.makedirs(os.path.dirname(item.destination), exist_ok=True)
        log(f"  {item.method:11} {item.destination}")
        if item.method == 'move':
            transfer(item.sources[0], item.destination)
        elif item.method == 'concatenate':
            item.duration = concatenate(item.sources, item.start, item.destination)
            if not copy:
                for path in item.sources + item.duplicates:
                    os.remove(path)
        elif item.method == 'extract':
            extract_audio(item.sources[0] if os.path.exists(item.sources[0]) else _moved_video(plan, item), item.destination)
        record = item.as_recording(item.destination)
        host_dir = Path(plan.target) / 'collection' / item.host
        update_manifest(host_dir, plan.session_id, item.start, record)
        report['recordings'].append(record)

    # the session manifest names the experiment and the group, and what a meta.txt said
    manifest = Path(plan.target) / 'manifest.json'
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        data['experiment_id'], data['group_id'] = plan.experiment_id, plan.group_id
        data['imported_from'] = plan.source
        if meta:
            data['legacy_meta'] = meta
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding='utf-8')
        (Path(plan.target) / 'manifest.yml').write_text("\n".join(_dump_yaml(data)) + "\n", encoding='utf-8')

    # what is left of the source becomes legacy/
    if not copy:
        legacy = Path(plan.target) / 'legacy'
        legacy.mkdir(parents=True, exist_ok=True)
        report['legacy'] = []
        for moved_root in _legacy_roots(plan):
            if not moved_root.exists():
                continue
            relative = moved_root.relative_to(Path(plan.source)) if moved_root != Path(plan.source) else Path(moved_root.name)
            destination = legacy / relative
            if destination.exists():
                destination = legacy / f"{relative}_{int(time.time())}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(moved_root), str(destination))
            report['legacy'].append(str(destination))
            _prune_empty(destination)
        _prune_empty(Path(plan.source))
    os.makedirs(plan.target, exist_ok=True)
    with open(os.path.join(plan.target, 'import_report.json'), 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    return report


def read_meta(plan: Plan) -> dict[str, Any]:
    """'settings' and 'participants' from a meta.txt in the session folder (an earlier run's note:
    'participants: valdemer=0, thorbjorn=1'), {} when there is none"""
    root = _legacy_root(plan) or Path(plan.source)
    for candidate in sorted(root.glob('meta*.txt')) if root.exists() else []:
        meta: dict[str, Any] = {}
        for line in candidate.read_text(encoding='utf-8', errors='replace').splitlines():
            key, _, value = line.partition(':')
            key, value = key.strip().lower(), value.strip()
            if key == 'participants':
                pairs = [p.strip() for p in value.split(',') if '=' in p]
                meta['participants'] = {n.strip(): t.strip() for n, t in (p.split('=', 1) for p in pairs)}
            elif key in ('settings', 'session') and value:
                meta[key] = value
        if meta:
            return meta
    return {}


def _moved_video(plan: Plan, item: Item) -> str:
    """the extracted audio's video after the video was moved: the same source, at its destination"""
    for other in plan.items:
        if other.modality == 'video' and other.sources == item.sources:
            return other.destination
    raise FileNotFoundError(item.sources[0])


def _legacy_roots(plan: Plan) -> list[Path]:
    """the folders whose leftovers become legacy/: the source itself when it is the session's
    folder, else every folder inside it that carries the session's name (one per pipeline in
    the oldest layout), kept under their relative paths"""
    root = Path(plan.source)
    if plan.session_dir_name and root.name != plan.session_dir_name:
        return sorted(p for p in root.rglob(plan.session_dir_name) if p.is_dir())
    return [root]


def _legacy_root(plan: Plan) -> Path | None:
    roots = _legacy_roots(plan)
    return roots[0] if len(roots) == 1 else (Path(plan.source) if roots and roots[0] == Path(plan.source) else None)


def _prune_empty(root: Path) -> None:
    """empty folders left behind by moved media (video_recordings/ once its files are gone)"""
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        folder = Path(dirpath)
        if folder == root:
            continue
        leftovers = [f for f in filenames if f not in ('.DS_Store', 'Thumbs.db')]
        if not leftovers and not any((folder / d).exists() for d in dirnames):
            shutil.rmtree(folder, ignore_errors=True)


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mmla ses-import',
        description="Bring a recorded session in an earlier layout into artifacts/<session>/collection/, "
                    "the layout the pipelines replay with source: file.")
    parser.add_argument('source', help="the session's folder (a session_<ISO>Z folder, or a folder holding one)")
    parser.add_argument('-e', '--experiment', default=DEFAULT_EXPERIMENT, help=f"experiment id (default {DEFAULT_EXPERIMENT})")
    parser.add_argument('-g', '--group', default=DEFAULT_GROUP, help=f"group id (default {DEFAULT_GROUP})")
    parser.add_argument('-sid', '--session-id', default=None, help="session id (default <experiment>_<group>_<start as YYMMDDTHHMMZ>)")
    parser.add_argument('--only', default=None, help="when the folder holds several sessions: the session_<ISO>Z name to import")
    parser.add_argument('--tz', default=DEFAULT_TZ, help=f"time zone of file names that carry a local time (default {DEFAULT_TZ})")
    parser.add_argument('-a', '--artifacts', default=None, help="artifacts root (default <project>/artifacts)")
    parser.add_argument('--copy', action='store_true', help="copy the files instead of moving them (needs the space)")
    parser.add_argument('-n', '--dry-run', action='store_true', help="show the plan and change nothing")
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    artifacts = args.artifacts or os.path.join(os.getcwd(), 'artifacts')
    plan = plan_import(args.source, artifacts, args.experiment, args.group, args.tz, args.only, args.session_id)
    print(plan.describe())
    if args.dry_run:
        return 0
    if not plan.items:
        print("nothing to import")
        return 1
    if os.path.exists(plan.target) and os.listdir(plan.target):
        print(f"{plan.target} exists and is not empty: pick another session id with -sid")
        return 1
    report = execute(plan, copy=args.copy)
    print(f"imported {len(report['recordings'])} recordings into {plan.target}; report in import_report.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
