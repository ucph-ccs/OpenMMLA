"""The sensing audit of mmla ses-code: a person checks what the VFA and ASR pipelines said about a
sample of camera frames and windows, without being shown what they said.

The questions, asked by a page of their own (audit_page, served by --audit):

  roster      per recording, which person wears each badge: crops of persons whose badge the camera
              read, each answered yes, no or cannot tell. The confirmed crops are that auditor's
              reference pictures of the recording's pupils, who are named A, B and C, never by badge.
  identity    per sampled camera frame, every person box the pose model drew, numbered left to right
              (the default blind mode shows no badge and no pipeline answer): which pupil it is,
              someone outside the group, not a person, or cannot tell; and how many group members the
              frame shows without a box. The answers are locked once saved, before any gaze question
              of the frame is shown.
  gaze        for every box the auditor called a pupil and every box a frozen pipeline version calls a
              pupil, with a close-up of the head cropped from the person's own keypoints: where the
              gaze lands, in the order the pipeline breaks ties (any face before any hands; among hands
              wherever it lands nearest; then the task material or work area; then elsewhere in the
              picture; then outside it), or between two of these (both named), or cannot tell.
  who speaks  per sampled 10 s window of a recording with a group microphone, its clip: no one, a
              group member, the teacher or another adult, another group, or cannot tell.

The answers do not depend on the pipeline version: the frames are drawn by their times alone (seeded,
per lesson, systematic within each camera), the windows likewise, and each version's outputs are
frozen apart (pipeline_<version>.json) and matched to the displayed boxes by their overlap, so one
pass of answers scores both the version a paper reports and a re-run (audit_score).

Which video frame a frame set is: every frame set's time must be, within TIME_TOLERANCE, the time a
base of the replay config acquired its keyframe (vfa_base: the sync time, the interval added k times
to the file's own offset, plus the file's start), else the session is left out: a sync time the TUI
rewrote, or another interval, would put every item on another frame. A second version must read the
same frames: its replay config's sync, interval and files equal the design's and its frame sets sit at
the items' times, else it is refused for the session.

Integrity: every step logs the sha256 of the files it wrote (design, view, each pipeline_<v>.json) in
the audit's request log; --anchor prints them, the scorer refuses a frozen version whose file is not
the one its freeze logged, and a version frozen once (its file deleted or not) is never frozen again.
Freezing and rendering refuse once a scored answer exists, unless --audit-after-answers, which is
logged and named in every score's header.

The steps, each a flag of mmla ses-code (docs/analytics/coding_and_audit.md):

  --audit-sample ID     the sessions, the items, the plan, and the display version's outputs frozen
  --audit-freeze ID     another version's outputs frozen for the same items
  --audit-render ID     the frames decoded as the bases decoded them and checked by their AprilTags,
                        the numbered boxes, head close-ups, roster crops and clips (vfa-base, cv2)
  --audit ID            the page (audit_page), on its own port and request log; with --audit-open no
                        links: each auditor types their name and chooses the full audit or the
                        reliability subset; with --audit-with ID2 a transcription audit served too,
                        from the same server and port under /t, each with its own request log
  --audit-estimate ID   the hours the answers take
  --audit-score ID      a version scored against the answers (audit_score)
  --audit-purge ID      the images and clips deleted

A transcription audit (--audit-sample ID --audit-task transcript; its plan's task is 'transcript') is drawn,
frozen and closed by audit_speech: listeners write down blind what is said in sampled windows, then rate the
text the content model read once an operator closes their blind pass (--audit-close-blind ID --audit-auditor
NAME; --audit-export-references ID writes their transcripts for the content model to read again).

Files:

  artifacts/runtime/audit/<id>/
    plan.json          the sessions with their aliases, lessons and strata, the sizes, the seed, the
                       declared analyses (served: no pipeline output in it)
    campaign.yml       the auditors' one-time links (code_locked; --campaign <this folder> --issue-token
                       NAME --token-scope audit [--token-subset reliability]) and whether the audit is
                       closed (an open audit has no links)
    requests.jsonl     the hash-chained request log of the server and of every step here
    media/<alias>/     the images and clips (--audit-purge deletes them)
    render_index.json  what the renderer found: the frame checks, crop brightness, errors
    scores/            audit_score's outputs
  artifacts/<session>/audit/<id>/
    view.json          what the server may serve: items, numbered boxes, image names, the boxes asked
                       about after identity
    pipeline.json      the sampling design and the display version's stored frames (never served)
    pipeline_<v>.json  version v's outputs for every item (never served)
    answers/<auditor>.jsonl   the auditors' answers, appended with their log lines
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from openmmla.commands.ses import code as C
from openmmla.commands.ses import code_locked as L

L.register_loaded(__name__, __file__)

AUDIT_ID = re.compile(r'[A-Za-z0-9_-]{1,40}')
VERSIONS = ('reported', 'rerun')
MODES = ('blind', 'verify')
# what --audit-task draws besides the sensing audit
TASKS = ('transcript',)
CODEBOOK_VERSION = 1
PLAN_FILE = 'plan.json'
VIEW_FILE = 'view.json'
DESIGN_FILE = 'pipeline.json'
RENDER_INDEX = 'render_index.json'
MEDIA_DIR = 'media'
ANSWERS_DIR = 'answers'
SCORES_DIR = 'scores'
LETTERS = 'ABC'
# the sizes a flag does not give, chosen so that the primary auditor's answers take a few hours
DEFAULTS = {'person_frames': 50, 'speech': 10, 'roster': 4, 'reliability': 0.2, 'practice': 12, 'seed': 1,
            'margin': 10.0, 'tag_px': 2.0, 'min_tag_match': 0.95, 'min_tag_frames': 3, 'check_frames': 20,
            'boot': 2000, 'boot_seed': 0}
# of the practice items, these many are who-speaks windows when the practice lesson has a group microphone
PRACTICE_SPEECH = 4
# a frame set belongs to a second when its time is this close to it (the synchronizer's match_tolerance)
MATCH_TOLERANCE = 0.5
# a frame set's time is a base's acquired time to float precision: farther than this from every base's keyframe
# time, the replay config is not the one the bases ran with
TIME_TOLERANCE = 1e-3
# a version's person is a displayed box when their boxes overlap this much (IoU): the same pose model
# drew both on the same frame
BOX_IOU = 0.5
# the face height and head brightness terciles of the gaze breakdowns, fixed before any answer is read
FACE_PX_CUTS = (24.0, 48.0)
LUMA_CUTS = (70.0, 130.0)
# the keypoint confidence the features endpoint scores with by default: a head box is made from keypoints this sure
KEYPOINT_CONFIDENCE = 0.3
# the assumed seconds per judgement of the estimate, until practice answers measure them
SECONDS = {'roster': 3.0, 'identity_frame': 5.0, 'identity_box': 3.0, 'gaze_box': 6.0, 'speech': 15.0, 'practice': 20.0,
           'transcribe': 240.0, 'reveal': 45.0}
# the analyses audit_score reports, fixed here before any answer is scored
DECLARED = [
    'identity: precision of the pipeline member boxes (correct over correct, swapped, outside the group and not a '
    'person), with its error mix, by tag source (read, server track memory, propagated) and for the untagged boxes '
    'at a missing pupil\'s seat',
    'identity: recall of the members the auditor saw, the misses split into undetected (no box or no person of the '
    'version at the box), untagged and mistagged; missed members per frame',
    'identity: the server\'s stored naming against the fused naming',
    'gaze: readability (pipeline unknown against auditor cannot tell) on the pipeline member boxes',
    'gaze: accuracy, Cohen\'s kappa and the confusion matrix over the eight classes and member-directed against not, '
    'on the boxes both could read; between answers apart; also on the boxes whose identity is correct',
    'gaze: by face source, face height tercile and head brightness tercile',
    'who speaks: speech present against speech_ratio > 0, > 0.1 and > 0.3',
    'who speaks: the share of windows with measured speech whose speaker is not the group\'s, unweighted and weighted '
    'by speech_ratio, and at speech_ratio >= 0.3',
    'who speaks: worn-microphone attribution (member words >= half the words) against the speaker, where worn '
    'microphones ran',
    'every table per lesson first, then by split, camera count and group size, then pooled with a lesson-cluster '
    'bootstrap; design-weighted beside unweighted',
    'inter-auditor agreement on the items two auditors answered',
]


class AuditError(Exception):
    """a refusal, said to the operator as it is"""


# ---- where things are ----

def audit_dir(artifacts: Path, audit_id: str) -> Path:
    return Path(artifacts) / 'runtime' / 'audit' / audit_id


def session_audit_dir(artifacts: Path, session_id: str, audit_id: str) -> Path:
    return Path(artifacts) / session_id / 'audit' / audit_id


def answers_dir(artifacts: Path, session_id: str, audit_id: str) -> Path:
    return session_audit_dir(artifacts, session_id, audit_id) / ANSWERS_DIR


def version_file(artifacts: Path, session_id: str, audit_id: str, version: str) -> Path:
    return session_audit_dir(artifacts, session_id, audit_id) / f'pipeline_{version}.json'


def record_files(artifacts: Path, audit_id: str, session_ids) -> list[tuple[str, Path]]:
    """every answers file of the audit, as its log lines name it (relative to the artifacts root)"""
    files = []
    for sid in session_ids:
        folder = answers_dir(artifacts, sid, audit_id)
        for path in sorted(folder.iterdir()) if folder.is_dir() else []:
            if path.is_file():
                files.append((path.relative_to(artifacts).as_posix(), path))
    return files


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: Path, data: Any) -> None:
    """a JSON file written whole, through a temp file and a rename"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=_plain) + '\n', encoding='utf-8')
    os.replace(temp, path)


def _plain(value):
    if hasattr(value, 'item'):  # a numpy number
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def load_plan(artifacts: Path, audit_id: str) -> dict:
    path = audit_dir(artifacts, audit_id) / PLAN_FILE
    if not path.exists():
        raise AuditError(f'no audit {audit_id} under {artifacts} (--audit-sample {audit_id})')
    return read_json(path)


def _event(artifacts: Path, audit_id: str, event: str, /, **fields) -> dict:
    """an operator's step in the audit's request log"""
    log = L.RequestLog(audit_dir(artifacts, audit_id) / L.LOG_FILE)
    try:
        return log.event(event, pid=os.getpid(), **fields)
    finally:
        log.close()


def _rel(artifacts: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(Path(artifacts).resolve()).as_posix()


def file_hashes(artifacts: Path, paths) -> dict[str, str | None]:
    """{path relative to the artifacts root: its sha256} of files a step wrote, for the step's log line"""
    return {_rel(artifacts, p): L.file_sha256(p) for p in paths}


def logged_files(artifacts: Path, audit_id: str, events=('audit-sample', 'audit-freeze')) -> dict[str, list]:
    """per file, the (seq, sha256, event) of every line of those steps in the audit's log that wrote it"""
    path = audit_dir(artifacts, audit_id) / L.LOG_FILE
    out: dict[str, list] = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_bytes().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get('event') in events:
            for rel, digest in (record.get('files') or {}).items():
                out[rel].append((record.get('seq'), digest, record['event']))
    return out


def scored_answers(artifacts: Path, plan: dict) -> int:
    """how many answers of the audit count for the scores (not practice) were saved by now"""
    count = 0
    for _, path in record_files(artifacts, plan['audit_id'], [s['id'] for s in plan['sessions']]):
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                count += not json.loads(line).get('practice')
            except (ValueError, AttributeError):
                continue
    return count


def anchor_hashes(artifacts: Path, audit_id: str) -> list[tuple[str, str | None]]:
    """what --anchor prints of an audit besides the answers: the plan, the render index and per recording
    (by alias) its design, view and frozen versions, each with its sha256"""
    folder = audit_dir(artifacts, audit_id)
    plan = load_plan(artifacts, audit_id)
    out = [('plan', L.file_sha256(folder / PLAN_FILE))]
    if (folder / RENDER_INDEX).exists():
        out.append(('render_index', L.file_sha256(folder / RENDER_INDEX)))
    for entry in sorted(plan['sessions'], key=lambda s: s['alias']):
        here = session_audit_dir(artifacts, entry['id'], audit_id)
        out.append((f"design {entry['alias']}", L.file_sha256(here / DESIGN_FILE)))
        out.append((f"view {entry['alias']}", L.file_sha256(here / VIEW_FILE)))
        for version in VERSIONS:
            path = version_file(artifacts, entry['id'], audit_id, version)
            if path.exists():
                out.append((f"frozen {entry['alias']} {version}", L.file_sha256(path)))
    return out


# ---- the sessions ----

def _expand(pattern: str, sid: str) -> Path:
    return Path(os.path.expanduser(pattern.format(sid=sid))).resolve()


def table_path(artifacts: Path, sid: str, pattern: str | None) -> Path:
    return _expand(pattern, sid) if pattern else C.fused_table(Path(artifacts) / sid)


def parameters_path(table: Path, sid: str, pattern: str | None) -> Path:
    return _expand(pattern, sid) if pattern else table.parent / 'fusion' / 'parameters.json'


def replay_path(artifacts: Path, sid: str, pattern: str | None) -> Path:
    if pattern:
        return _expand(pattern, sid)
    return Path(artifacts).resolve().parent / 'pipelines' / 'vfa-base' / f'config_replay_{sid}.yml'


def manifest_recordings(session_dir: Path) -> tuple[list[dict], list[dict]]:
    """(videos, audios) of a session's manifest whose files exist"""
    from openmmla.collection.recording import default_audio_scope
    manifest = C._read(session_dir / 'manifest.json')
    recordings = [r for r in manifest.get('recordings', []) if os.path.exists(r.get('path', ''))]
    videos = [dict(r) for r in recordings if r.get('modality') == 'video']
    audios = []
    for r in recordings:
        if r.get('modality') == 'audio':
            r = dict(r)
            r['scope'] = r.get('scope') or default_audio_scope(r.get('device'), r.get('method'), r.get('host'))
            audios.append(r)
    return videos, audios


def group_mic(audios: list[dict]) -> dict | None:
    """the session's group microphone (scope group), the coding page's preferred one first"""
    group = [a for a in audios if a.get('scope') == 'group']
    group.sort(key=lambda r: (C.AUDIO_PREFERENCE.index(r['device']) if r.get('device') in C.AUDIO_PREFERENCE else 99,
                              str(r.get('device'))))
    return group[0] if group else None


FILE_START = re.compile(r'_(\d+(?:\.\d+)?)\.')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')


def replay_bases(config_path: Path, project_dir: Path) -> dict:
    """what a VFA replay config says each base read: the sync time, the keyframe interval, the rotation,
    and per file base its id, camera, angle, video path and the file's start (from its name), as
    vfa_base resolves them (a bare file name in Base.file_dir or the project; the sync time configured,
    else the latest file start in the base's own folder; the base's camera, else the first of Cameras);
    fisheye cameras carry their K and D. The config's sha256 is kept, so the render can tell it changed."""
    import yaml
    try:
        raw = Path(config_path).read_bytes()
    except OSError:
        raise AuditError(f'no replay config at {config_path}') from None
    config = yaml.safe_load(raw.decode('utf-8')) or {}
    base = config.get('Base') or {}
    cameras = config.get('Cameras') or {}
    out = {'interval': float(base.get('keyframe_interval', 30.0)), 'rotate': int(base.get('rotate', 0) or 0),
           'resolution': list(base.get('resolution') or (1920, 1080)), 'bases': [],
           'config': {'path': str(Path(config_path).resolve()), 'sha256': L.sha256_hex(raw)}}
    configured = base.get('initial_sync_time')
    configured = None if configured is None or str(configured).strip() == '' else float(configured)
    for entry in config.get('Bases') or []:
        if not isinstance(entry, dict) or entry.get('id') is None or str(entry['id']).startswith('<'):
            continue
        if (entry.get('source') or base.get('source')) != 'file':
            continue
        named = str(entry.get('source_index') or '')
        if named and not os.path.isabs(named):
            if os.path.dirname(named):
                named = os.path.join(project_dir, named)
            else:  # a bare name: in Base.file_dir of an older config, else in the project
                folder = base.get('file_dir') or str(project_dir)
                named = os.path.join(folder if os.path.isabs(folder) else os.path.join(project_dir, folder), named)
        match = FILE_START.search(os.path.basename(named))
        if not match:
            raise AuditError(f'{config_path}: base {entry["id"]} names no <name>_<start>.<ext> file')
        name = entry.get('camera') if entry.get('camera') in cameras else (sorted(cameras)[0] if cameras else None)
        camera = cameras.get(name) or {}
        if configured is not None:
            sync = configured
        else:
            from openmmla.utils.config import compute_initial_sync_time
            sync = compute_initial_sync_time(os.path.dirname(named), None, exts=VIDEO_EXTS)
        out['bases'].append({'id': str(entry['id']), 'camera': name, 'angle': entry.get('camera_angle'),
                             'path': os.path.realpath(named), 'file_start': float(match.group(1)), 'sync': sync,
                             'fisheye': {'K': camera['K'], 'D': camera['D']} if camera.get('fisheye') else None})
    if not out['bases']:
        raise AuditError(f'{config_path} has no file base')
    syncs = sorted({b['sync'] for b in out['bases']})
    if len(syncs) > 1:
        raise AuditError(f'{config_path}: its bases started from different sync times ({syncs[0]} to {syncs[-1]}), '
                         'so no keyframe is one moment of every camera')
    out['sync'] = syncs[0]
    out['bases'].sort(key=lambda b: b['id'])
    return out


def keyframe_times(sync: float, file_start: float, interval: float, n: int) -> list[float]:
    """the acquired times of a base's keyframes 0 to n - 1, in vfa_base's own float arithmetic (the offset
    into the file, the interval added once per keyframe, then the file's start added)"""
    out, at = [], sync - file_start
    for _ in range(max(0, n)):
        out.append(at + file_start)
        at += interval
    return out


def same_frames(replay: dict, design: dict) -> str | None:
    """None when a replay config reads the frames the design was drawn from (its sync time, interval and
    files), else what differs"""
    if abs(float(replay['sync']) - float(design['sync'])) > TIME_TOLERANCE:
        return f"its sync time is {replay['sync']}, the design's {design['sync']}"
    if abs(float(replay['interval']) - float(design['interval'])) > 1e-9:
        return f"its keyframe interval is {replay['interval']}, the design's {design['interval']}"
    mine = {b['id']: (b['path'], b['file_start']) for b in replay['bases']}
    theirs = {b['id']: (b['path'], b['file_start']) for b in design['bases']}
    if mine != theirs:
        return 'its bases read other video files than the design names'
    return None


def frame_k(frame_time: float, sync: float, interval: float) -> int:
    """the keyframe a frame set's time belongs to: the base's loop counter at that time"""
    return int(round((frame_time - sync) / interval))


def video_time(sync: float, file_start: float, interval: float, k: int) -> float:
    """where in its file a base read keyframe k: vfa_base._process_keyframes' own float arithmetic, the
    interval added k times"""
    at = sync - file_start
    for _ in range(k):
        at += interval
    return at


class CameraOf:
    """the base a frame of a frame set came from: its echoed camera (the base id), else for an angle
    several bases share its place among them (the synchronizer sends them in base id order), else
    the one base of its angle"""

    def __init__(self, bases: list[dict]):
        self.ids = {b['id'] for b in bases}
        self.by_angle: dict[str, list[str]] = defaultdict(list)
        for b in sorted(bases, key=lambda b: b['id']):
            self.by_angle[str(b.get('angle') or 'frame')].append(b['id'])

    def __call__(self, key: str) -> str | None:
        if key in self.ids:
            return key
        angle, _, place = key.partition('#')
        ids = self.by_angle.get(angle) or []
        if place.isdigit():
            return ids[int(place)] if int(place) < len(ids) else None
        return ids[0] if len(ids) == 1 else None


# ---- events, tables and the fused view ----

def _parse_records(loaded) -> list[dict]:
    from openmmla.utils.constants import EVENT_TYPE_VFA_FEATURES
    from openmmla.utils.querys import deep_parse_json
    if isinstance(loaded, dict):
        loaded = loaded.get(EVENT_TYPE_VFA_FEATURES) or loaded.get('records') or []
    if not isinstance(loaded, list):
        raise AuditError('the events file holds no list of vfa_features records')
    records = [deep_parse_json(r) for r in loaded if isinstance(r, dict)]
    from openmmla.analytics.fusion.window_features import _time
    records.sort(key=_time)
    return records


def load_events(sid: str, source: str | None, influx_config: str | None) -> tuple[list[dict], dict]:
    """a session's vfa_features records and where they came from: InfluxDB (source None), or a folder
    holding <sid>.json.gz, <sid>.json or a measurements export <sid>/ (Sessions -> Export)"""
    from openmmla.utils.constants import EVENT_TYPE_VFA_FEATURES
    if source is None:
        if not influx_config or not os.path.exists(influx_config):
            raise AuditError(f'no InfluxDB config at {influx_config} (--influx-config), and no --audit-events')
        from openmmla.utils.client.influx_client import InfluxDBClientWrapper
        from openmmla.utils.querys import fetch_and_process_data
        records = fetch_and_process_data(sid, EVENT_TYPE_VFA_FEATURES, InfluxDBClientWrapper(influx_config))
        body = json.dumps(records, sort_keys=True, default=str).encode()
        return records, {'source': 'influxdb', 'sha256': L.sha256_hex(body), 'count': len(records)}
    folder = Path(os.path.expanduser(source.format(sid=sid))).resolve()
    for name in (f'{sid}.json.gz', f'{sid}.json'):
        path = folder / name
        if path.is_file():
            data = path.read_bytes()
            text = gzip.decompress(data) if name.endswith('.gz') else data
            records = _parse_records(json.loads(text))
            return records, {'source': str(path), 'sha256': L.sha256_hex(data), 'count': len(records)}
    if (folder / sid).is_dir() or folder.is_dir():
        from openmmla.analytics.fusion.window_features import export_files, load_events_from_export
        where = folder / sid if (folder / sid).is_dir() else folder
        paths = export_files(str(where), sid).get(EVENT_TYPE_VFA_FEATURES) or []
        if paths:
            records = load_events_from_export(str(where), sid).get(EVENT_TYPE_VFA_FEATURES, [])
            digest = L.sha256_hex(''.join(L.file_sha256(p) or '' for p in sorted(paths)))
            return records, {'source': str(where), 'sha256': digest, 'count': len(records)}
    raise AuditError(f'no vfa_features of {sid} under {folder} (<sid>.json.gz, <sid>.json or an export <sid>/)')


def fusion_parameters(path: Path) -> dict:
    """what a fused table was made with (ses-fuse's parameters.json): its parameters and outputs, with
    the defaults of a table fused before a parameter was recorded"""
    try:
        record = read_json(path)
    except (OSError, ValueError):
        raise AuditError(f'no fusion parameters at {path}') from None
    params = record.get('parameters') or {}
    hand = params.get('hand_circle') or {}
    # a table fused before a step was recorded did not take it: the face refusal is off then
    return {'tag_memory': params.get('tag_memory_seconds'), 'face_refusal': params.get('face_refusal_frames'),
            'hand_relabel': bool(hand.get('relabelled', True)),
            'pupils': [str(p) for p in params['pupils']] if params.get('pupils') else None,
            'participants': [str(p) for p in params['participants']] if params.get('participants') else None,
            'window': float(params.get('window', 10.0)), 'step': float(params.get('step', 10.0)),
            'events': params.get('events') or {}, 'speech': params.get('speech') or {},
            'outputs': {Path(o.get('path', '')).name: o.get('sha256') for o in record.get('outputs') or []}}


class FusedView:
    """a session's frames as its fused table saw them, aligned to the stored records:
    window_features' own steps in its own order (the tag memory, the face's refusals of remembered tags,
    the hand circle, the work areas, the tags carried along the tracks), and the pupils' seats and tracks"""

    def __init__(self, records: list[dict], params: dict):
        from openmmla.analytics.fusion import window_features as W
        from openmmla.services.vfa import features as F
        self.stored = list(records)
        self.layout = W.frame_set_layout(self.stored)
        memory = params.get('tag_memory')
        raw = self.stored
        if memory is not None and W._tracked(raw):
            raw = W.expire_track_tags(raw, self.layout, memory)
        refusal = params.get('face_refusal')
        if refusal and W._tracked(raw):
            if not hasattr(W, 'refuse_track_tags'):
                raise AuditError("the table's fusion took remembered tags off on the face's word, which this "
                                 "checkout's window_features cannot")
            raw = W.refuse_track_tags(raw, self.layout, refusal)
        self.participants = params.get('participants') or W.participants_of({W.EVENT_TYPE_VFA_FEATURES: self.stored})
        self.pupils = [str(p) for p in params['pupils']] if params.get('pupils') is not None \
            else W.default_pupils(self.participants)
        relabel = params.get('hand_relabel', True)
        self.nudge = F.HAND_NUDGES[W.table_hand_circle(raw, relabel)]
        circled = W.relabel_hand_circles(raw, self.nudge) if relabel and raw else raw
        labelled = W.label_work_areas(circled, self.pupils, self.layout, self.nudge) if circled else circled
        self.tracked = W._tracked(raw)
        self.fused = W.propagate_track_tags(labelled, self.layout, memory=memory) if self.tracked else labelled
        self.raw = raw
        # the pupils' seats as window_features names them (seats_of per tag, so the participants do not matter)
        self.pupil_seats = W.seats_of(raw, self.pupils, self.layout) if any(W._time(r) > 0 for r in raw) else None
        self.pupil_tracks = W.pupil_tracks_of(raw, self.pupils, self.layout)
        self.times = [W._time(r) for r in self.stored]
        self.order = sorted((i for i, t in enumerate(self.times) if t > 0), key=lambda i: self.times[i])

    def frames(self, i: int, which: str = 'fused') -> list[dict]:
        from openmmla.analytics.fusion.window_features import _frames_of
        return _frames_of({'stored': self.stored, 'raw': self.raw, 'fused': self.fused}[which][i])

    def keys(self, i: int) -> list[str]:
        from openmmla.analytics.fusion.window_features import camera_keys
        return camera_keys(self.frames(i, 'stored'), self.layout[1])

    def known_keys(self, i: int) -> list[str | None]:
        """keys(i), less the places among cameras of a shared angle (`<angle>#<k>`) in a frame set that lost
        or gained a frame: the place no longer names its camera there (None)"""
        keys = self.keys(i)
        if len(keys) == self.layout[0]:
            return keys
        return [None if '#' in key else key for key in keys]

    def nearest(self, moment: float) -> int | None:
        """the record whose time is nearest `moment`, within MATCH_TOLERANCE"""
        import bisect
        times = [self.times[i] for i in self.order]
        at = bisect.bisect_left(times, moment)
        best = None
        for n in (at - 1, at):
            if 0 <= n < len(times) and abs(times[n] - moment) <= MATCH_TOLERANCE:
                if best is None or abs(times[n] - moment) < abs(self.times[best] - moment):
                    best = self.order[n]
        return best

    def frame_of(self, i: int, base: str, camera_of: CameraOf) -> int | None:
        """the index of base's frame in record i"""
        for j, key in enumerate(self.known_keys(i)):
            if key is not None and camera_of(key) == base:
                return j
        return None


def parity(view: FusedView, table, kept: list[str]) -> str | None:
    """None when the fused view reproduces the table's per-window camera counts and gaze shares of the
    kept persons, else the first difference. A window boundary the rounding of the table's times can
    move a frame across is tried a millisecond either way."""
    from openmmla.analytics.fusion import window_features as W
    index = W.EventIndex(view.fused, instant=True)
    gaze_index = W.gaze_points(index.records, view.layout)
    columns = [(f'p{t}_frame_sets', None) for t in kept] + \
              [(f'p{t}_gaze_{c}_ratio', None) for t in kept for c in W.GAZE_CATEGORIES]
    present = [name for name, _ in columns if name in table.columns]
    if not present:
        return 'the table has no camera columns for the kept persons'

    def values(ws, we):
        return W.body_gaze_features(index, ws, we, [str(t) for t in kept], view.layout, pupils=view.pupils,
                                    gaze_index=gaze_index, work_area=True, seats=view.pupil_seats, nudge=view.nudge,
                                    pupil_tracks=view.pupil_tracks)

    def same(row, out):
        for name in present:
            want, got = row[name], out.get(name)
            want = None if want is None or (isinstance(want, float) and math.isnan(want)) else float(want)
            if want is None and got is None:
                continue
            if want is None or got is None or abs(float(got) - want) > 1e-3:
                return name, want, got
        return None

    for _, row in table.iterrows():
        ws, we = float(row['window_start']), float(row['window_end'])
        miss = same(row, values(ws, we))
        if miss and all(same(row, values(ws + d, we + d)) for d in (-0.001, 0.001)):
            return f'window at {ws - float(table["window_start"].iloc[0]):.0f} s: {miss[0]} is {miss[1]} in the table, {miss[2]} here'
    return None


def check_freeze(table_file: Path, params: dict, events: dict) -> list[str]:
    """why a table may not describe these events (an empty list when it does): its sha256 against the one
    its parameters recorded, and the events' count against the count the fusion read"""
    problems = []
    recorded = params['outputs'].get(table_file.name)
    if recorded and recorded != L.file_sha256(table_file):
        problems.append('the table changed since it was fused (its sha256 is not the one parameters.json recorded)')
    fused = params['events'].get('vfa_features')
    if fused is not None and int(fused) != int(events['count']):
        problems.append(f"the fusion read {fused} vfa_features events, the events here are {events['count']}")
    return problems


# ---- the boxes and the persons ----

def iou(a, b) -> float:
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
        bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    except (TypeError, ValueError, IndexError):
        return 0.0
    w, h = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if w <= 0 or h <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - w * h
    return w * h / union if union > 0 else 0.0


def numbered_boxes(persons: list[dict]) -> dict[str, dict]:
    """the displayed boxes of a stored frame: every person with a box, numbered from 1 by the box's
    centre left to right: {number: {'person': index, 'bbox': [x1, y1, x2, y2]}}"""
    boxed = []
    for index, person in enumerate(persons):
        try:
            box = [float(v) for v in (person.get('bbox') or [])[:4]]
        except (TypeError, ValueError):
            continue
        if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
            boxed.append(((box[0] + box[2]) / 2, index, box))
    boxed.sort()
    return {str(n + 1): {'person': index, 'bbox': [round(v, 1) for v in box]} for n, (_, index, box) in enumerate(boxed)}


def match_boxes(boxes: dict[str, dict], persons: list[dict]) -> dict[str, tuple[int, float]]:
    """displayed box number -> (index of the version's person, IoU): pairs of overlap BOX_IOU or more, the
    largest first, one to one"""
    pairs = sorted(((iou(box['bbox'], person.get('bbox') or []), n, j) for n, box in boxes.items()
                    for j, person in enumerate(persons)), key=lambda p: (-p[0], int(p[1]), p[2]))
    taken_boxes, taken_persons, out = set(), set(), {}
    for overlap, n, j in pairs:
        if overlap < BOX_IOU:
            break
        if n in taken_boxes or j in taken_persons:
            continue
        out[n] = (j, round(overlap, 3))
        taken_boxes.add(n)
        taken_persons.add(j)
    return out


def head_crop_box(person: dict) -> tuple[list[float], str] | None:
    """(the head box the close-up is cut around, where it came from): features.head_box from the person's
    own keypoints ('keypoints'), else the top 30 % of the person's box ('box'); None without either"""
    from openmmla.services.vfa import features as F
    try:
        box = F.head_box(person, KEYPOINT_CONFIDENCE)
    except (KeyError, TypeError, ValueError, IndexError):
        box = None
    if box is not None:
        return [float(v) for v in box], 'keypoints'
    try:
        x1, y1, x2, y2 = (float(v) for v in person['bbox'][:4])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return [x1, y1, x2, y1 + 0.3 * (y2 - y1)], 'box'


def _face(person: dict) -> tuple[str | None, float | None, list | None]:
    box = person.get('face_bbox')
    if not box or len(box) < 4:
        return None, None, None
    source = 'pose' if person.get('face_source') == 'pose' else 'detector'
    return source, round(float(box[3]) - float(box[1]), 1), [float(v) for v in box[:4]]


def person_values(view: FusedView, i: int, j: int, k: int, kept: set[str]) -> dict:
    """a version's outputs for the person at (record i, frame j, person k): the fused naming and the
    server's stored naming, the gaze label the table counts and what it rests on"""
    from openmmla.analytics.fusion import window_features as W
    fused_frame = view.frames(i, 'fused')[j]
    persons = fused_frame.get('persons') or []
    person = persons[k]
    stored = (view.frames(i, 'stored')[j].get('persons') or [])[k]
    pupils = set(view.pupils)
    seats_here = (view.pupil_seats or {}).get(view.keys(i)[j])
    seated = W._seated_untagged(persons, seats_here, pupils)
    tag = None if person.get('tag_id') is None else str(person['tag_id'])
    seat_of = None
    if tag is None and seats_here:
        tagged = {str(p['tag_id']) for p in persons if p.get('tag_id') is not None}
        centre = W._box_centre(person)
        if centre is not None:
            seat_of = next((t for t, seat in sorted(seats_here.items()) if t in pupils and t not in tagged
                            and W._at_seat(centre, seat)), None)
    gaze = person.get('gaze') or {}
    target = gaze.get('target') or {}
    source, face_h, face_box = _face(person)
    head = head_crop_box(person)
    inside = None
    if face_box is not None and head is not None:
        cx, cy = (face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2
        inside = head[0][0] <= cx <= head[0][2] and head[0][1] <= cy <= head[0][3]
    reid = person.get('reid') if isinstance(person.get('reid'), dict) else None
    return {'tag': tag, 'match': person.get('tag_match'), 'member': tag if tag in kept else None, 'seat_of': seat_of,
            'track': person.get('track_id'),
            'stored': {'tag': None if stored.get('tag_id') is None else str(stored['tag_id']),
                       'match': stored.get('tag_match'), 'person_id': stored.get('person_id')},
            'gaze': {'label': W._gaze_label(person, pupils, True, seated), 'category': target.get('category'),
                     'target': None if target.get('person_id') is None else str(target.get('person_id')),
                     'inout': gaze.get('inout'), 'face_source': source, 'face_h': face_h, 'face_inside_head': inside},
            'reid': {str(c): (r or {}).get('verdict') for c, r in reid.items()} if reid else None}


# ---- drawing items ----

def allocate(total: int, sizes: dict, floor: int = 0) -> dict:
    """`total` items over strata in proportion to their sizes, each at least min(floor, size) and at most
    its size, by largest remainder; keys in order break ties"""
    keys = [k for k in sizes if sizes[k] > 0]
    total = min(total, sum(sizes[k] for k in keys))
    out = {k: min(floor, sizes[k]) for k in keys}
    while sum(out.values()) < total:
        left = total - sum(out.values())
        open_keys = [k for k in keys if out[k] < sizes[k]]
        weight = sum(sizes[k] for k in open_keys)
        shares = {k: left * sizes[k] / weight for k in open_keys}
        whole = {k: min(int(shares[k]), sizes[k] - out[k]) for k in open_keys}
        if sum(whole.values()) == 0:
            ranked = sorted(open_keys, key=lambda k: (-(shares[k] - int(shares[k])), keys.index(k)))
            for k in ranked[:left]:
                out[k] += 1
        else:
            for k, n in whole.items():
                out[k] += n
    return {k: out.get(k, 0) for k in sizes}


def systematic(population: list, n: int, rng: random.Random) -> list:
    """n items of a time-ordered list, evenly spaced from a random start"""
    if n >= len(population):
        return list(population)
    if n <= 0:
        return []
    step = len(population) / n
    start = rng.random() * step
    return [population[int(start + i * step)] for i in range(n)]


def _rng(seed, audit_id, *parts) -> random.Random:
    return random.Random(':'.join(str(p) for p in (seed, audit_id) + parts))


def _flag_share(items: list[dict], share: float, rng: random.Random, flag: str) -> None:
    if not items or share <= 0:
        return
    n = min(len(items), max(1, int(round(share * len(items)))))
    for item in rng.sample(items, n):
        item[flag] = True


class Session:
    """a session as the audit draws from it: its recordings, replay, table, roster and display events"""

    def __init__(self, artifacts: Path, sid: str, args, version_sources: dict):
        from openmmla.analytics.interaction import layout as Y
        from openmmla.analytics.interaction import splits
        self.id, self.dir = sid, Path(artifacts) / sid
        # the audit's designs and scores keep the classifier's DEV/TEST split as a stratum, which the
        # coding page no longer shows
        self.lesson, self.task = splits.lesson_key(sid), splits.task_of(sid)
        self.split = 'TEST' if sid in splits.TEST_SESSIONS else 'DEV'
        self.videos, audios = manifest_recordings(self.dir)
        if not self.videos:
            raise AuditError('no video recording')
        self.mic = group_mic(audios)
        self.replay = replay_bases(replay_path(artifacts, sid, args.audit_replay_configs), Path(artifacts).resolve().parent)
        paths = {os.path.realpath(v['path']) for v in self.videos}
        stray = [b['id'] for b in self.replay['bases'] if b['path'] not in paths]
        if stray:
            raise AuditError(f'the replay read a video that is not a recording of the manifest (base {stray[0]})')
        self.table_file = table_path(artifacts, sid, version_sources['tables'])
        if not self.table_file.exists():
            raise AuditError(f'no fused table at {self.table_file}')
        self.params_file = parameters_path(self.table_file, sid, version_sources['parameters'])
        self.params = fusion_parameters(self.params_file)
        self.table = Y.read_table(self.table_file)
        roster = Y.session_roster(self.table, self.dir)
        self.kept = [str(t) for t in roster.kept]
        self.group_size = int(roster.group_size)
        self.included = Y.session_inclusion(self.table, roster)[0]
        self.records, self.events = load_events(sid, version_sources['events'], version_sources['influx'])
        self.problems = check_freeze(self.table_file, self.params, self.events)
        self.view = FusedView(self.records, self.params)
        self.keyframe = keyframes_of(self.view, self.replay)
        mismatch = parity(self.view, self.table, self.kept)
        if mismatch:
            self.problems.append(f'the fused view does not reproduce the table: {mismatch}')
        self.camera_of = CameraOf(self.replay['bases'])

    def sources(self) -> dict:
        return {'events': self.events, 'table': str(self.table_file), 'table_sha256': L.file_sha256(self.table_file),
                'parameters': str(self.params_file), 'parameters_sha256': L.file_sha256(self.params_file),
                'replay_config': self.replay['config'], 'problems': self.problems}

    def population(self, margin: float) -> dict[str, list[tuple[int, int, int, float]]]:
        """per base, its frames by time, (k, record, frame, time), within the session's frame sets less
        `margin` seconds at either end"""
        times = [self.view.times[i] for i in self.view.order]
        if not times:
            return {}
        first, last = times[0] + margin, times[-1] - margin
        out: dict[str, dict[int, tuple]] = defaultdict(dict)
        for i in self.view.order:
            t = self.view.times[i]
            if not first <= t <= last:
                continue
            k = self.keyframe[i]
            for j, key in enumerate(self.view.known_keys(i)):
                base = self.camera_of(key) if key is not None else None
                if base is not None:
                    out[base].setdefault(k, (k, i, j, t))
        return {base: sorted(found.values()) for base, found in out.items()}


def keyframes_of(view: FusedView, replay: dict) -> dict[int, int]:
    """record index -> the keyframe k its frame set is: its time must be a keyframe time of every base of the
    replay config (keyframe_times) within TIME_TOLERANCE, else the config's sync time or interval is not
    the one the bases ran with, and AuditError says so"""
    sync, interval = float(replay['sync']), float(replay['interval'])
    ks = {i: frame_k(view.times[i], sync, interval) for i in view.order}
    if not ks:
        return {}
    n = max(ks.values()) + 1
    tables = [keyframe_times(sync, b['file_start'], interval, n) for b in replay['bases']]
    off = []
    for i, k in ks.items():
        t = view.times[i]
        worst = max(abs(table[k] - t) for table in tables) if k >= 0 else math.inf
        if worst > TIME_TOLERANCE:
            off.append((i, worst))
    if off:
        i, worst = off[0]
        raise AuditError(f'{len(off)} of {len(ks)} frame sets are not at the keyframe times of the replay config '
                         f'(the first, {view.times[i] - view.times[view.order[0]]:.1f} s in, is '
                         f'{"before its sync time" if worst == math.inf else f"{worst:.3f} s off"}): its sync time or '
                         'keyframe interval is not the one the bases ran with')
    return ks


def _stored_frame(view: FusedView, i: int, j: int) -> dict:
    """a stored frame as the renderer reads it: its size, tags and persons (boxes, keypoints, faces)"""
    frame = view.frames(i, 'stored')[j]
    keep = ('person_id', 'tag_id', 'tag_match', 'track_id', 'bbox', 'keypoints', 'face_bbox', 'face_source', 'gaze')
    return {'width': frame.get('width'), 'height': frame.get('height'), 'tags': frame.get('tags') or {},
            'persons': [{k: p.get(k) for k in keep} for p in frame.get('persons') or []]}


def roster_candidates(session: Session, margin: float, used: set) -> dict[str, dict[str, list]]:
    """per kept tag, per base, the frames whose stored person read that tag (torso or box) with the tag's
    centre inside the person's box: (k, record, frame, person, time, centre)"""
    population = session.population(margin)
    found: dict[str, dict[str, list]] = {t: defaultdict(list) for t in session.kept}
    for base, frames in population.items():
        for k, i, j, t in frames:
            if (base, k) in used:
                continue
            frame = session.view.frames(i, 'stored')[j]
            tags = frame.get('tags') or {}
            for p, person in enumerate(frame.get('persons') or []):
                tag = person.get('tag_id')
                if tag is None or str(tag) not in found or person.get('tag_match') not in ('torso', 'box'):
                    continue
                centre = tags.get(str(tag))
                box = person.get('bbox') or []
                if not centre or len(box) < 4 or not (box[0] <= centre[0] <= box[2] and box[1] <= centre[1] <= box[3]):
                    continue
                found[str(tag)][base].append((k, i, j, p, t, [float(centre[0]), float(centre[1])]))
    return found


def sample(artifacts: Path, audit_id: str, sessions: list[Session], sizes: dict, mode: str, version: str,
           argv=None) -> tuple[dict, dict, dict]:
    """draw the items: (the plan, the view of each session, the design of each session)"""
    seed = sizes['seed']
    ids = [s.id for s in sessions]
    _rng(seed, audit_id, 'aliases').shuffle(ids)
    alias = {sid: f'R{n + 1:02d}' for n, sid in enumerate(ids)}
    by_id = {s.id: s for s in sessions}
    lessons: dict[str, list[Session]] = defaultdict(list)
    for sid in ids:
        lessons[by_id[sid].lesson].append(by_id[sid])
    margin = sizes['margin']
    populations = {s.id: s.population(margin) for s in sessions}
    vision: dict[str, list[dict]] = defaultdict(list)
    speech: dict[str, list[dict]] = defaultdict(list)
    weights: dict[str, dict] = defaultdict(dict)
    speech_pop: dict[str, list[tuple[float, float]]] = {}
    for lesson, members in lessons.items():
        group = max(s.group_size for s in members) or 1
        n_frames = math.ceil(sizes['person_frames'] / group)
        per_session = allocate(n_frames, {s.id: sum(len(f) for f in populations[s.id].values()) for s in members}, floor=1)
        for s in members:
            bases = sorted(populations[s.id])
            per_base = {b: 0 for b in bases}
            order = list(bases)
            _rng(seed, audit_id, 'cameras', s.id).shuffle(order)
            for n in range(per_session[s.id]):
                open_bases = [b for b in order if per_base[b] < len(populations[s.id][b])]
                if not open_bases:
                    break
                per_base[min(open_bases, key=lambda b: (per_base[b], order.index(b)))] += 1
            for base in bases:
                drawn = systematic(populations[s.id][base], per_base[base], _rng(seed, audit_id, 'vision', s.id, base))
                if drawn:
                    weights[s.id][base] = {'population': len(populations[s.id][base]), 'sampled': len(drawn),
                                           'weight': len(populations[s.id][base]) / len(drawn)}
                for k, i, j, t in drawn:
                    vision[s.id].append({'base': base, 'k': k, 'record': i, 'frame': j, 'time': t,
                                         'weight': len(populations[s.id][base]) / len(drawn)})
        # the who-speaks windows: the fused grid's windows inside the group microphone and the videos
        mic_sessions = [s for s in members if s.mic is not None]
        pops = {}
        for s in mic_sessions:
            starts = [float(v) for v in s.table['window_start']]
            ends = [float(v) for v in s.table['window_end']]
            lo = max([s.mic['start_time']] + [v['start_time'] for v in s.videos])
            hi = min([s.mic['start_time'] + (s.mic.get('duration') or 0)] +
                     [v['start_time'] + (v.get('duration') or 0) for v in s.videos])
            inside = [(a, b) for a, b in zip(starts, ends) if a >= lo + margin and b <= hi - margin]
            pops[s.id] = inside
            speech_pop[s.id] = inside
        per_mic = allocate(sizes['speech'], {sid: len(p) for sid, p in pops.items()})
        for sid, windows in pops.items():
            drawn = sorted(_rng(seed, audit_id, 'speech', sid).sample(windows, per_mic[sid])) if per_mic[sid] else []
            for a, b in drawn:
                speech[sid].append({'window_start': a, 'window_end': b, 'weight': len(windows) / len(drawn)})
    # practice: the first DEV lesson in the order, items apart from the drawn ones
    practice_lesson = next((by_id[sid].lesson for sid in ids if by_id[sid].split == 'DEV'), by_id[ids[0]].lesson if ids else None)
    practice: dict[str, list] = {'vision': [], 'speech': []}
    if practice_lesson and sizes['practice'] > 0:
        # up to PRACTICE_SPEECH who-speaks windows the draw left, and frames for the rest
        rng = _rng(seed, audit_id, 'practice')
        members = lessons[practice_lesson]
        heard = {(sid, it['window_start']) for sid in speech for it in speech[sid]}
        windows = [(s.id, a, b) for s in members for a, b in speech_pop.get(s.id, []) if (s.id, a) not in heard]
        for sid, a, b in rng.sample(windows, min(len(windows), PRACTICE_SPEECH, sizes['practice'])):
            practice['speech'].append((sid, {'window_start': a, 'window_end': b, 'weight': None}))
        taken = {(s.id, it['base'], it['k']) for s in members for it in vision[s.id]}
        frames = [(s.id, base, entry) for s in members for base, found in sorted(populations[s.id].items())
                  for entry in found if (s.id, base, entry[0]) not in taken]
        for sid, base, (k, i, j, t) in rng.sample(frames, min(len(frames), sizes['practice'] - len(practice['speech']))):
            practice['vision'].append((sid, {'base': base, 'k': k, 'record': i, 'frame': j, 'time': t, 'weight': None}))
    # the items, numbered in the order the page shows them
    plan_sessions, views, designs = [], {}, {}
    for sid in ids:
        s = by_id[sid]
        a = alias[sid]
        letters = {LETTERS[n]: t for n, t in enumerate(sorted(s.kept, key=lambda t: (len(t), t))[:len(LETTERS)])}
        cams = {b['id']: n + 1 for n, b in enumerate(s.replay['bases'])}
        items_v = [dict(it, practice=False, reliability=False) for it in vision[sid]]
        items_v += [dict(it, practice=True, reliability=False) for psid, it in practice['vision'] if psid == sid]
        items_s = [dict(it, practice=False, reliability=False) for it in speech[sid]]
        items_s += [dict(it, practice=True, reliability=False) for psid, it in practice['speech'] if psid == sid]
        for part, items in (('vision', items_v), ('speech', items_s)):
            # the page shows a part's items in a seeded order of their own, the practice items apart
            regular, practising = [it for it in items if not it['practice']], [it for it in items if it['practice']]
            _rng(seed, audit_id, 'order', sid, part).shuffle(regular)
            items[:] = regular + practising
        for n, it in enumerate([it for it in items_v if not it['practice']], 1):
            it['item'] = f'{audit_id}-{a}-v-{n:03d}'
        for n, it in enumerate([it for it in items_v if it['practice']], 1):
            it['item'] = f'{audit_id}-{a}-pv-{n:02d}'
        for n, it in enumerate([it for it in items_s if not it['practice']], 1):
            it['item'] = f'{audit_id}-{a}-s-{n:03d}'
        for n, it in enumerate([it for it in items_s if it['practice']], 1):
            it['item'] = f'{audit_id}-{a}-ps-{n:02d}'
        used = {(it['base'], it['k']) for it in items_v}
        candidates = roster_candidates(s, margin, used)
        roster_items = []
        for letter, tag in letters.items():
            by_base = candidates.get(tag) or {}
            bases = sorted(by_base)
            per = {b: 0 for b in bases}
            for n in range(sizes['roster']):
                open_bases = [b for b in bases if per[b] < len(by_base[b])]
                if not open_bases:
                    break
                per[min(open_bases, key=lambda b: (per[b], b))] += 1
            for base in bases:
                for k, i, j, p, t, centre in systematic(by_base[base], per[base], _rng(seed, audit_id, 'roster', sid, tag, base)):
                    roster_items.append({'pupil': letter, 'tag': tag, 'base': base, 'k': k, 'time': t, 'record': i,
                                         'frame': j, 'person': p, 'centre': centre})
                    used.add((base, k))
        roster_items.sort(key=lambda it: (it['pupil'], it['time']))
        for n, it in enumerate(roster_items, 1):
            it['item'] = f'{audit_id}-{a}-r-{n:03d}'
        # the frames the renderer checks besides the items': stored tags on each camera
        checks = []
        for base, frames in sorted(populations[sid].items()):
            tagged = [f for f in frames if (base, f[0]) not in used and (s.view.frames(f[1], 'stored')[f[2]].get('tags') or {})]
            for k, i, j, t in _rng(seed, audit_id, 'checks', sid, base).sample(tagged, min(len(tagged), sizes['check_frames'])):
                frame = s.view.frames(i, 'stored')[j]
                checks.append({'base': base, 'k': k, 'time': t, 'width': frame.get('width'), 'height': frame.get('height'),
                               'tags': frame.get('tags') or {}})
        designs[sid] = {'audit_id': audit_id, 'session': sid, 'alias': a, 'lesson': s.lesson, 'split': s.split,
                        'task': s.task, 'n_cameras': len(s.replay['bases']), 'group_size': s.group_size, 'kept': s.kept,
                        'letters': letters, 'pupils': s.view.pupils, 'group_mic': s.mic is not None,
                        'display_version': version, 'sync': s.replay['sync'], 'interval': s.replay['interval'],
                        'rotate': s.replay['rotate'], 'resolution': s.replay['resolution'],
                        'replay_config': s.replay['config'],
                        'bases': [{**b, 'index': cams[b['id']]} for b in s.replay['bases']],
                        'videos': [{'device': v.get('device'), 'path': v['path'], 'start_time': v['start_time'],
                                    'duration': v.get('duration')} for v in s.videos],
                        'mic': {'device': s.mic.get('device'), 'path': s.mic['path'], 'start_time': s.mic['start_time'],
                                'duration': s.mic.get('duration')} if s.mic else None,
                        'weights': weights.get(sid, {}),
                        'speech_population': len(speech_pop.get(sid, [])),
                        'vision': [{'item': it['item'], 'base': it['base'], 'k': it['k'], 'time': it['time'],
                                    'weight': it['weight'], 'practice': it['practice'], 'reliability': False,
                                    'frame': _stored_frame(s.view, it['record'], it['frame'])} for it in items_v],
                        'roster': [{'item': it['item'], 'pupil': it['pupil'], 'tag': it['tag'], 'base': it['base'],
                                    'k': it['k'], 'time': it['time'], 'centre': it['centre'],
                                    'frame': _stored_frame(s.view, it['record'], it['frame']), 'person': it['person']}
                                   for it in roster_items],
                        'speech': [{'item': it['item'], 'window_start': it['window_start'], 'window_end': it['window_end'],
                                    'weight': it['weight'], 'practice': it['practice'], 'reliability': False}
                                   for it in items_s],
                        'checks': checks, 'sources': {version: s.sources()}}
    # the reliability subset: a share of each lesson's vision and speech items
    for lesson, members in lessons.items():
        for part in ('vision', 'speech'):
            items = [it for s in members for it in designs[s.id][part] if not it['practice']]
            _flag_share(items, sizes['reliability'], _rng(seed, audit_id, 'reliability', lesson, part), 'reliability')
    for sid in ids:
        design = designs[sid]
        views[sid] = {'audit_id': audit_id, 'alias': design['alias'], 'mode': mode, 'group_size': design['group_size'],
                      'pupils': list(design['letters']), 'cameras': design['n_cameras'],
                      'items': [{'item': it['item'], 'part': 'roster', 'pupil': it['pupil'], 'render': None}
                                for it in design['roster']]
                      + [{'item': it['item'], 'part': 'vision', 'camera': next(b['index'] for b in design['bases']
                                                                             if b['id'] == it['base']),
                          'size': [it['frame']['width'], it['frame']['height']],
                          'boxes': {n: b['bbox'] for n, b in numbered_boxes(it['frame']['persons']).items()},
                          'practice': it['practice'], 'reliability': it['reliability'], 'render': None}
                         for it in design['vision']]
                      + [{'item': it['item'], 'part': 'speech', 'practice': it['practice'],
                          'reliability': it['reliability'], 'render': None} for it in design['speech']]}
        plan_sessions.append({'id': sid, 'alias': design['alias'], 'lesson': design['lesson'], 'split': design['split'],
                              'task': design['task'], 'n_cameras': design['n_cameras'], 'group_size': design['group_size'],
                              'group_mic': design['group_mic'], 's1_included': by_id[sid].included,
                              'vision': sum(1 for it in design['vision'] if not it['practice']),
                              'speech': sum(1 for it in design['speech'] if not it['practice']),
                              'roster': len(design['roster']),
                              'practice': sum(1 for p in ('vision', 'speech') for it in design[p] if it['practice'])})
    from openmmla.commands.ses.code_locked import software
    plan = {'audit_id': audit_id, 'created_at': C.now_utc(), 'seed': seed, 'mode': mode, 'codebook_version': CODEBOOK_VERSION,
            'display_version': version, 'sizes': sizes, 'sessions': plan_sessions, 'order': [alias[sid] for sid in ids],
            'practice_lesson': practice_lesson, 'tag_check': {'px': sizes['tag_px'], 'min_rate': sizes['min_tag_match'],
                                                              'min_frames': sizes['min_tag_frames']},
            'cuts': {'face_px': list(FACE_PX_CUTS), 'luma': list(LUMA_CUTS)}, 'declared': DECLARED,
            'software': software(), 'argv': [a for a in (argv or [])]}
    plan['estimate_hours'] = estimate(plan, designs.values())
    return plan, views, designs


def estimate(plan: dict, designs, measured: dict | None = None) -> dict:
    """the hours the answers take, by part, for the primary auditor and for the reliability subset, from
    the assumed seconds per judgement or from `measured` ones"""
    seconds = dict(SECONDS, **(measured or {}))
    if plan.get('task') == 'transcript':
        return _estimate_transcript(designs, seconds)
    hours = defaultdict(float)
    second = defaultdict(float)
    for design in designs:
        hours['roster'] += len(design['roster']) * seconds['roster']
        for it in design['vision']:
            boxes = len(numbered_boxes(it['frame']['persons']))
            members = sum(1 for p in it['frame']['persons'] if p.get('tag_id') is not None and str(p['tag_id']) in design['kept'])
            cost = seconds['identity_frame'] + boxes * seconds['identity_box'] + max(members, 1) * seconds['gaze_box']
            if it['practice']:
                hours['practice'] += cost
                second['practice'] += cost
            else:
                hours['vision'] += cost
                if it['reliability']:
                    second['vision'] += cost
        for it in design['speech']:
            part = 'practice' if it['practice'] else 'speech'
            hours[part] += seconds['speech']
            second[part] += seconds['speech'] if it['practice'] or it['reliability'] else 0.0
        if any(it['reliability'] for p in ('vision', 'speech') for it in design[p]):
            second['roster'] += len(design['roster']) * seconds['roster']
    out = {part: round(value / 3600, 2) for part, value in hours.items()}
    out['total'] = round(sum(hours.values()) / 3600, 2)
    out['reliability_auditor'] = round(sum(second.values()) / 3600, 2)
    return out


def _estimate_transcript(designs, seconds: dict) -> dict:
    """a transcription audit's hours: the practice block, then per sweep a blind transcription and a reveal rating
    of each scored item, for the primary auditor; the practice and the flagged items' transcriptions for the
    reliability subset"""
    hours = defaultdict(float)
    second = 0.0
    for design in designs:
        for it in design['speech']:
            if it['practice']:
                hours['practice'] += seconds['transcribe']
                second += seconds['transcribe']
                continue
            hours[f"transcribe_{it.get('sweep', 1)}"] += seconds['transcribe']
            hours[f"reveal_{it.get('sweep', 1)}"] += seconds['reveal']
            second += seconds['transcribe'] if it['reliability'] else 0.0
    out = {part: round(value / 3600, 2) for part, value in sorted(hours.items())}
    out['sweep_1'] = round((hours['practice'] + hours['transcribe_1'] + hours['reveal_1']) / 3600, 2)
    out['total'] = round(sum(hours.values()) / 3600, 2)
    out['reliability_auditor'] = round(second / 3600, 2)
    return out


# ---- a version's outputs, frozen ----

def freeze(artifacts: Path, audit_id: str, design: dict, version: str, session: Session) -> dict:
    """version `version`'s outputs for every item of a session: per vision item each displayed box's
    person (matched by overlap) with the fused and stored naming and the gaze label, the version's
    persons no box shows, and per who-speaks window the table's speech measures"""
    view, kept = session.view, set(session.kept)
    bases = {b['id'] for b in design['bases']}
    out_vision = {}
    for it in design['vision']:
        i = view.nearest(it['time'])
        entry: dict[str, Any] = {'record_time': None if i is None else view.times[i], 'frame': False, 'boxes': {}, 'extra': []}
        if i is not None and abs(view.times[i] - it['time']) > TIME_TOLERANCE:
            # a frame set a fraction of a second away is another keyframe's, of another frame
            entry['reason'] = f"the nearest frame set is {view.times[i] - it['time']:+.3f} s from the item's time"
            i = None
        elif i is None:
            entry['reason'] = 'no frame set at this time'
        j = view.frame_of(i, it['base'], session.camera_of) if i is not None and it['base'] in bases else None
        if i is not None and j is None:
            entry['reason'] = "the frame set holds no frame of this camera that can be told"
        if j is not None:
            entry['frame'] = True
            persons = view.frames(i, 'fused')[j].get('persons') or []
            boxes = numbered_boxes(it['frame']['persons'])
            matched = match_boxes(boxes, persons)
            for n, (k, overlap) in matched.items():
                entry['boxes'][n] = dict(person_values(view, i, j, k, kept), iou=overlap)
            taken = {k for k, _ in matched.values()}
            for k in range(len(persons)):
                if k not in taken:
                    values = person_values(view, i, j, k, kept)
                    entry['extra'].append({'tag': values['tag'], 'member': values['member'], 'match': values['match']})
        out_vision[it['item']] = entry
    out_speech = {}
    table = session.table
    words_columns = [f'p{t}_words' for t in session.kept if f'p{t}_words' in table.columns]
    worn = bool(words_columns) and bool((session.params.get('speech') or {}).get('wearers'))
    for it in design['speech']:
        found = None
        for _, row in table.iterrows():
            if abs(float(row['window_start']) - it['window_start']) <= MATCH_TOLERANCE:
                found = row
                break
        if found is None:
            out_speech[it['item']] = {'row': False}
            continue

        def value(name):
            v = found.get(name) if name in table.columns else None
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        member = [value(c) for c in words_columns]
        out_speech[it['item']] = {'row': True, 'window_start': float(found['window_start']),
                                  'speech_ratio': value('speech_ratio'), 'words': value('words'),
                                  'n_asr_recognition': value('n_asr_recognition'), 'worn': worn,
                                  'member_words': sum(v for v in member if v is not None) if worn else None}
    return {'audit_id': audit_id, 'session': session.id, 'version': version, 'frozen_at': C.now_utc(),
            'sources': session.sources(), 'kept': session.kept, 'pupils': view.pupils,
            'group_size': session.group_size, 'vision': out_vision, 'speech': out_speech}


# ---- the commands ----

def add_arguments(parser) -> None:
    """the audit's flags of mmla ses-code; every default is None, so that the campaign commands
    (code_locked.requested) tell an audit command by any flag set"""
    group = parser.add_argument_group('sensing audit', 'identity, gaze and who speaks checked blind by a person; see '
                                      'docs/analytics/coding_and_audit.md')
    group.add_argument('--audit-sample', default=None, metavar='ID', help="draw the audit's items and freeze the display version")
    group.add_argument('--audit-freeze', default=None, metavar='ID', help="freeze another version's outputs for the audit's items")
    group.add_argument('--audit-render', default=None, metavar='ID', help="decode, check and draw the frames, cut the clips (vfa-base)")
    group.add_argument('--audit', default=None, metavar='ID', help="serve the audit's page (port 8766 unless -p)")
    group.add_argument('--audit-open', action='store_true', default=None,
                       help="with --audit: no links, each auditor types their name and chooses the full audit or the "
                            "reliability subset (--allow-from optional; recorded in the start line)")
    group.add_argument('--audit-with', default=None, metavar='ID',
                       help="with --audit: serve this transcription audit too, from the same server and port, under /t "
                            "(the entry page at / then leads to both audits and to the coding page)")
    group.add_argument('--audit-code-port', type=int, default=None, metavar='PORT',
                       help="with --audit-with: the coding page's port; the entry page at / links to that port of the "
                            "host it was opened at (default 8765; 0 leaves the link out)")
    group.add_argument('--audit-estimate', default=None, metavar='ID', help="print the hours the answers take")
    group.add_argument('--audit-score', default=None, metavar='ID', help="score a version against the answers")
    group.add_argument('--audit-purge', default=None, metavar='ID', help="delete the audit's images and clips")
    group.add_argument('--audit-version', default=None, choices=VERSIONS, help="the pipeline version: reported or rerun")
    group.add_argument('--audit-events', default=None, metavar='DIR', help="the version's vfa_features: <DIR>/<sid>.json.gz, <sid>.json or an export <sid>/ (default InfluxDB)")
    group.add_argument('--audit-tables', default=None, metavar='PATTERN', help="the version's fused tables, {sid} for the session (default the session's own)")
    group.add_argument('--audit-parameters', default=None, metavar='PATTERN', help="the tables' parameters.json (default <table folder>/fusion/parameters.json)")
    group.add_argument('--audit-replay-configs', default=None, metavar='PATTERN', help="the VFA replay configs (default pipelines/vfa-base/config_replay_{sid}.yml)")
    group.add_argument('--audit-sessions', default=None, help="exact session ids, comma-separated (default every session with video, a table and a replay)")
    group.add_argument('--audit-include-held', action='store_true', default=None, help="also the sessions --hold names")
    group.add_argument('--audit-mode', default=None, choices=MODES, help="blind (default): no badge or pipeline answer drawn; verify draws them")
    group.add_argument('--audit-person-frames', type=int, default=None, help=f"member judgements per lesson (default {DEFAULTS['person_frames']})")
    group.add_argument('--audit-speech', type=int, default=None, help=f"who-speaks windows per lesson with a group microphone (default {DEFAULTS['speech']}); transcript: the fresh windows of a lesson the source drew none of")
    group.add_argument('--audit-roster', type=int, default=None, help=f"roster crops per pupil (default {DEFAULTS['roster']})")
    group.add_argument('--audit-reliability', type=float, default=None, help=f"the share of items in the second auditor's subset (default {DEFAULTS['reliability']}); transcript: of the social windows (default 0.25)")
    group.add_argument('--audit-practice', type=int, default=None, help=f"practice items (default {DEFAULTS['practice']})")
    group.add_argument('--audit-seed', type=int, default=None, help=f"the seed of every draw (default {DEFAULTS['seed']})")
    group.add_argument('--audit-margin', type=float, default=None, help=f"seconds left out at either end of a session (default {DEFAULTS['margin']:g})")
    group.add_argument('--audit-check-frames', type=int, default=None, help=f"extra frames per camera checked by their tags (default {DEFAULTS['check_frames']})")
    group.add_argument('--audit-tag-px', type=float, default=None, help=f"how far a re-detected tag may lie from the stored one (default {DEFAULTS['tag_px']:g} px)")
    group.add_argument('--audit-min-tag-match', type=float, default=None, help=f"the share of checked frames a camera must match (default {DEFAULTS['min_tag_match']})")
    group.add_argument('--audit-min-tag-frames', type=int, default=None, help=f"the fewest checked frames a camera needs (default {DEFAULTS['min_tag_frames']})")
    group.add_argument('--audit-allow-drift', action='store_true', default=None, help="keep a session whose table, events or fused view do not agree (recorded)")
    group.add_argument('--audit-after-answers', action='store_true', default=None,
                       help="freeze or render although scored answers exist (logged, named in every score)")
    group.add_argument('--audit-auditor', default=None, help="the primary auditor of the scores (default the one whose audit link has no subset; open, the one name that chose the full audit)")
    group.add_argument('--audit-boot', type=int, default=None, help=f"bootstrap resamples (default {DEFAULTS['boot']})")
    group.add_argument('--audit-boot-seed', type=int, default=None, help=f"the bootstrap's seed (default {DEFAULTS['boot_seed']})")
    group.add_argument('--audit-out', default=None, metavar='DIR', help="where the scores go (default <audit>/scores/<version>_<time>); with --audit-export-references the file")
    # the transcription audit (audit_speech); its sizes default to audit_speech.TRANSCRIPT_DEFAULTS
    group.add_argument('--audit-task', default=None, choices=TASKS, help="with --audit-sample: transcript draws a blind transcription audit (audit_speech); default the sensing audit")
    group.add_argument('--audit-windows-from', default=None, metavar='ID', help="transcript: take this sensing audit's who-speaks windows, aliases and order")
    group.add_argument('--audit-first-ranks', type=int, default=None, help="transcript: the ranks per lesson in sweep 1 (default 3)")
    group.add_argument('--audit-sweeps', type=int, default=None, help="with --audit: the sweeps of a transcription audit served (default 1)")
    group.add_argument('--audit-social', type=int, default=None, help="transcript: windows the named coders labelled social (default 28)")
    group.add_argument('--audit-social-coders', default=None, metavar='NAMES', help="transcript: whose social labels draw those windows, comma-separated")
    group.add_argument('--audit-social-both', type=float, default=None, help="transcript: the share of them every named coder labelled social (default 0.65)")
    group.add_argument('--audit-gap', type=float, default=None, help="transcript: the seconds a social or practice window keeps from the other windows (default 20)")
    group.add_argument('--audit-reliability-ranks', type=int, default=None, help="transcript: the ranks per lesson in the reliability subset (default 1)")
    group.add_argument('--audit-asr-events', default=None, metavar='PATTERN', help="transcript: the version's asr_transcription events, a file pattern with {sid} (JSON lines, gzipped or not, or a JSON list) or influx")
    group.add_argument('--audit-asr-reference', default=None, metavar='FILE', help="transcript: the texts a content model read (JSON lines); every frozen text must be its row's")
    group.add_argument('--audit-content-scores', default=None, metavar='ARM=FILE,...', help="transcript: content scores of the version's text, frozen (or, scoring, read) per arm")
    group.add_argument('--audit-content-scores-ref', default=None, metavar='ARM=FILE,...', help="transcript, scoring: the content scores of the primary auditor's references")
    group.add_argument('--audit-primary', default=None, metavar='NAME', help="transcript, with --audit-sample: the primary auditor, written into the plan (optional)")
    group.add_argument('--audit-close-blind', default=None, metavar='ID', help="close --audit-auditor's blind pass of a transcription audit (logged)")
    group.add_argument('--audit-export-references', default=None, metavar='ID', help="write --audit-auditor's blind transcripts to --audit-out (mode 0600, logged)")
    group.add_argument('--audit-compare-with', default=None, metavar='ID', help="transcript, scoring: the sensing audit whose who-speaks answers the scores join")
    group.add_argument('--audit-exclude-sessions', default=None, metavar='IDS', help="transcript, scoring: a sensitivity without these sessions, comma-separated (named in the header)")
    group.add_argument('--audit-interim', action='store_true', default=None, help="transcript: score before the primary's blind close (logged, named in every header)")


ACTIONS = ('audit_sample', 'audit_freeze', 'audit_render', 'audit', 'audit_estimate', 'audit_score', 'audit_purge',
           'audit_close_blind', 'audit_export_references')


def requested(args) -> bool:
    """whether any audit flag is given (run then asks for one step when none or several are)"""
    return any(value for name, value in vars(args).items() if name.startswith('audit'))


def _size(args, name):
    value = getattr(args, f'audit_{name}', None)
    return DEFAULTS[name] if value is None else value


def _artifacts(args) -> Path:
    return Path(args.artifacts or os.path.join(os.getcwd(), 'artifacts')).resolve()


def _sources(args) -> dict:
    return {'events': args.audit_events, 'tables': args.audit_tables, 'parameters': args.audit_parameters,
            'influx': args.influx_config or os.path.join(os.getcwd(), C.DEFAULT_INFLUX_CONFIG)}


def candidate_sessions(artifacts: Path, args) -> list[str]:
    if args.audit_sessions:
        return [s.strip() for s in args.audit_sessions.split(',') if s.strip()]
    hold = () if args.audit_include_held else tuple(h.strip() for h in args.hold.split(',') if h.strip())
    found = []
    for directory in sorted(artifacts.glob('exp_*')):
        if args.sessions and args.sessions not in directory.name:
            continue
        if any(h in directory.name for h in hold):
            print(f'held: {directory.name}')
            continue
        found.append(directory.name)
    return found


def load_session(artifacts: Path, sid: str, args, sources: dict) -> Session:
    """a Session, any failure of its files (a corrupt events file, a config that does not parse) said as an
    AuditError naming its kind, so one session's failure leaves it out and the others go on"""
    try:
        return Session(artifacts, sid, args, sources)
    except AuditError:
        raise
    except Exception as error:  # the session is left out with the reason, not the whole step
        raise AuditError(f'{type(error).__name__}: {str(error)[:300]}') from error


def cmd_sample(args, argv) -> int:
    if args.audit_task == 'transcript':
        from openmmla.commands.ses import audit_speech
        return audit_speech.cmd_sample(args, argv)
    artifacts, audit_id = _artifacts(args), args.audit_sample
    if not args.audit_version:
        raise AuditError('give --audit-version reported|rerun: the version whose stored frames give the boxes')
    folder = audit_dir(artifacts, audit_id)
    if (folder / PLAN_FILE).exists():
        raise AuditError(f'{folder / PLAN_FILE} exists already: an audit is drawn once')
    sizes = {name: _size(args, name) for name in DEFAULTS}
    sources = _sources(args)
    sessions = []
    for sid in candidate_sessions(artifacts, args):
        try:
            session = load_session(artifacts, sid, args, sources)
        except AuditError as error:
            print(f'left out: {sid}: {error}')
            continue
        if session.problems and not args.audit_allow_drift:
            print(f"left out: {sid}: {'; '.join(session.problems)} (--audit-allow-drift keeps it, recorded)")
            continue
        sessions.append(session)
    if not sessions:
        raise AuditError('no session can be audited')
    plan, views, designs = sample(artifacts, audit_id, sessions, sizes, args.audit_mode or 'blind', args.audit_version, argv)
    by_id = {s.id: s for s in sessions}
    written = []
    for sid, design in designs.items():
        here = session_audit_dir(artifacts, sid, audit_id)
        write_json(here / DESIGN_FILE, design)
        write_json(here / VIEW_FILE, views[sid])
        write_json(version_file(artifacts, sid, audit_id, args.audit_version),
                   freeze(artifacts, audit_id, design, args.audit_version, by_id[sid]))
        written += [here / DESIGN_FILE, here / VIEW_FILE, version_file(artifacts, sid, audit_id, args.audit_version)]
    write_json(folder / PLAN_FILE, plan)
    campaign = L.Campaign(folder)
    campaign.update(lambda d: d.update({
        'campaign': audit_id, 'kind': 'audit', 'audit_id': audit_id, 'created_at': plan['created_at'],
        'artifacts': str(artifacts), 'window': 10.0, 'step': 10.0, 'seed': sizes['seed'],
        'sessions': [{'id': s['id'], 'alias': s['alias']} for s in plan['sessions']], 'cookie_days': L.COOKIE_DAYS,
        'closed_at': None, 'released_at': None, 'coders': []}), create=True)
    _event(artifacts, audit_id, 'audit-sample', plan_sha256=L.file_sha256(folder / PLAN_FILE),
           version=args.audit_version, sessions=len(designs), files=file_hashes(artifacts, written), argv=list(argv))
    _print_plan(plan)
    return 0


def _print_plan(plan: dict) -> None:
    print(f"audit {plan['audit_id']} ({plan['mode']}, display version {plan['display_version']}): "
          f"{len(plan['sessions'])} sessions")
    for s in plan['sessions']:
        print(f"  {s['alias']} {s['split'] or '-'} {s['n_cameras']} camera(s), group of {s['group_size']}"
              f"{', group mic' if s['group_mic'] else ''}: {s['roster']} roster, {s['vision']} frames, "
              f"{s['speech']} windows{', ' + str(s['practice']) + ' practice' if s['practice'] else ''}")
    hours = plan['estimate_hours']
    print('estimate: ' + ', '.join(f'{k} {v} h' for k, v in hours.items()))


def after_answers(artifacts: Path, plan: dict, args, step: str) -> int:
    """the scored answers saved so far; a step that changes what is scored or shown refuses once there are
    some, unless --audit-after-answers (the count goes into the step's log line and the scores' header)"""
    count = scored_answers(artifacts, plan)
    if count and not args.audit_after_answers:
        raise AuditError(f'{count} scored answers are saved: {step} now and the scores rest on outputs or pictures '
                         'chosen after the answers (--audit-after-answers does it anyway, logged and named in the scores)')
    return count


def cmd_freeze(args, argv) -> int:
    artifacts, audit_id = _artifacts(args), args.audit_freeze
    plan = load_plan(artifacts, audit_id)
    if plan.get('task') == 'transcript':
        from openmmla.commands.ses import audit_speech
        return audit_speech.cmd_freeze(args, argv, plan)
    if args.audit_task:
        raise AuditError(f'{audit_id} is a sensing audit, not a {args.audit_task} audit')
    if not args.audit_version:
        raise AuditError('give --audit-version reported|rerun')
    answered = after_answers(artifacts, plan, args, 'freezing a version')
    rendered = (audit_dir(artifacts, audit_id) / RENDER_INDEX).exists()
    sources = _sources(args)
    before = logged_files(artifacts, audit_id)
    frozen, refused, skipped, written, refusals = 0, 0, 0, [], []
    for entry in plan['sessions']:
        sid, alias = entry['id'], entry['alias']
        design = read_json(session_audit_dir(artifacts, sid, audit_id) / DESIGN_FILE)
        target = version_file(artifacts, sid, audit_id, args.audit_version)
        earlier = [seq for seq, _, _ in before.get(_rel(artifacts, target), [])]
        if earlier:
            # frozen and logged once: a file deleted and frozen again would be a version chosen afterwards
            print(f'{alias}: {args.audit_version} was frozen at seq {earlier[0]} of the log: a version is frozen once')
            skipped += 1
            continue
        if target.exists() and not read_json(target).get('refused'):
            print(f'{alias}: {target.name} exists already: a version is frozen once')
            skipped += 1
            continue
        stub = {'audit_id': audit_id, 'session': sid, 'version': args.audit_version}
        try:
            session = load_session(artifacts, sid, args, sources)
            differs = same_frames(session.replay, design)
            if differs:
                raise AuditError(f"this version's replay config reads other frames than the design's: {differs}")
        except AuditError as error:
            write_json(target, {**stub, 'refused': str(error)})
            refusals.append(target)
            print(f'{alias}: refused: {error}')
            refused += 1
            continue
        if session.problems and not args.audit_allow_drift:
            write_json(target, {**stub, 'refused': '; '.join(session.problems)})
            refusals.append(target)
            print(f"{alias}: refused: {'; '.join(session.problems)}")
            refused += 1
            continue
        try:
            data = freeze(artifacts, audit_id, design, args.audit_version, session)
        except Exception as error:  # one session's failure leaves it out, not the others
            write_json(target, {**stub, 'refused': f'{type(error).__name__}: {str(error)[:300]}'})
            refusals.append(target)
            print(f'{alias}: refused: {type(error).__name__}: {error}')
            refused += 1
            continue
        data.update(after_answers=answered, after_render=rendered)
        write_json(target, data)
        written.append(target)
        frozen += 1
        print(f'{alias}: {args.audit_version} frozen')
    if rendered and frozen:
        print(f'note: the render ran before this freeze, so the gaze question asks about none of the boxes only '
              f'{args.audit_version} calls pupils; render again before any answer to ask them (the scores count them '
              f'as member boxes not asked)')
    _event(artifacts, audit_id, 'audit-freeze', version=args.audit_version, frozen=frozen, refused=refused,
           skipped=skipped, after_answers=answered, after_render=rendered, files=file_hashes(artifacts, written),
           refused_files=file_hashes(artifacts, refusals), argv=list(argv))
    return 0 if frozen else 1


def cmd_estimate(args, argv) -> int:
    artifacts, audit_id = _artifacts(args), args.audit_estimate
    plan = load_plan(artifacts, audit_id)
    designs = [read_json(session_audit_dir(artifacts, s['id'], audit_id) / DESIGN_FILE) for s in plan['sessions']]
    measured = measured_seconds(artifacts, plan)
    hours = estimate(plan, designs, measured)
    print('estimate' + (' (from the practice answers)' if measured else ' (assumed seconds per judgement)') + ': '
          + ', '.join(f'{k} {v} h' for k, v in hours.items()))
    return 0


def measured_seconds(artifacts: Path, plan: dict) -> dict:
    """the median seconds the practice answers took, per kind of judgement, when there are some"""
    spent = defaultdict(list)
    for entry in plan['sessions']:
        for _, path in record_files(artifacts, plan['audit_id'], [entry['id']]):
            for line in path.read_text(encoding='utf-8').splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not record.get('practice') or not isinstance(record.get('seconds_spent'), (int, float)):
                    continue
                answer = record.get('answer') or {}
                if record.get('phase') == 'identity':
                    spent['identity_box'].append(record['seconds_spent'] / max(1, len(answer.get('boxes') or {})))
                elif record.get('phase') == 'gaze':
                    spent['gaze_box'].append(record['seconds_spent'] / max(1, len(answer.get('boxes') or {})))
                elif record.get('phase') in ('speech', 'transcribe', 'reveal'):
                    spent[record['phase']].append(record['seconds_spent'])
    out = {}
    for name, values in spent.items():
        values.sort()
        out[name] = values[len(values) // 2]
    if 'identity_box' in out:
        out['identity_frame'] = 0.0
    return out


def cmd_purge(args, argv) -> int:
    artifacts, audit_id = _artifacts(args), args.audit_purge
    load_plan(artifacts, audit_id)
    media = audit_dir(artifacts, audit_id) / MEDIA_DIR
    files = sum(1 for p in media.rglob('*') if p.is_file()) if media.is_dir() else 0
    size = sum(p.stat().st_size for p in media.rglob('*') if p.is_file()) if media.is_dir() else 0
    shutil.rmtree(media, ignore_errors=True)
    _event(artifacts, audit_id, 'audit-purge', files=files, bytes=size)
    print(f'{files} images and clips ({size / 1e6:.1f} MB) deleted from {media}; the plan, views, frozen outputs and '
          'answers are kept')
    return 0


def run(args, argv) -> int:
    """an audit command of mmla ses-code; 0 done, 1 refused, 2 a usage error"""
    argv = list(sys.argv[1:] if argv is None else argv)
    actions = [a for a in ACTIONS if getattr(args, a, None)]
    if len(actions) != 1:
        print('give one of ' + ', '.join('--' + a.replace('_', '-') for a in ACTIONS))
        return 2
    if getattr(args, 'audit_open', None) and actions[0] != 'audit':
        print('--audit-open goes with --audit ID (the scorer finds an open audit in its request log)')
        return 2
    if getattr(args, 'audit_with', None) and actions[0] != 'audit':
        print('--audit-with goes with --audit ID (both audits served at one address)')
        return 2
    if getattr(args, 'audit_code_port', None) is not None and not getattr(args, 'audit_with', None):
        print('--audit-code-port goes with --audit ID --audit-with ID2 (the page at / that leads to both)')
        return 2
    # the id names folders every step reads, writes or deletes: never a path
    ids = [getattr(args, actions[0])] + ([args.audit_with] if getattr(args, 'audit_with', None) else [])
    if not all(AUDIT_ID.fullmatch(str(audit_id)) for audit_id in ids):
        print('refused: an audit id is letters, digits, - and _, at most 40')
        return 1
    try:
        if actions[0] == 'audit_sample':
            return cmd_sample(args, argv)
        if actions[0] == 'audit_freeze':
            return cmd_freeze(args, argv)
        if actions[0] == 'audit_render':
            from openmmla.commands.ses import audit_render
            return audit_render.cmd_render(args, argv)
        if actions[0] == 'audit':
            from openmmla.commands.ses import audit_page
            return audit_page.cmd_serve(args, argv)
        if actions[0] == 'audit_estimate':
            return cmd_estimate(args, argv)
        if actions[0] == 'audit_score':
            from openmmla.commands.ses import audit_score
            return audit_score.cmd_score(args, argv)
        if actions[0] in ('audit_close_blind', 'audit_export_references'):
            from openmmla.commands.ses import audit_speech
            return (audit_speech.cmd_close_blind if actions[0] == 'audit_close_blind'
                    else audit_speech.cmd_export_references)(args, argv)
        return cmd_purge(args, argv)
    except (AuditError, L.CampaignError) as error:
        print(f'refused: {error}')
        return 1
