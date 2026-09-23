"""Replay the collection recordings of sessions through the ASR, VFA and IPS pipelines offline,
one session after another, the three pipelines of a session at once, and fuse what they wrote
into the session's 10 s window table.

Run on the machine that holds the artifacts and the pipeline environments (server-01), inside
tmux, from the repository root in the uber-base environment:

    python scripts/replay_sessions.py --dry-run           # the plan, nothing launched
    python scripts/replay_sessions.py                     # every session under artifacts/
    python scripts/replay_sessions.py --sessions exp_...  # some of them
    python scripts/replay_sessions.py --pipelines vfa,ips # some pipelines

For every session the runner reads `artifacts/<session>/manifest.json`, picks the group
microphone (jabra-0, else vimo-0-ch0, vimo-0, badge-0) and every video, writes one config per
pipeline from that pipeline's template config (`--asr-template` ...: the pilot configs, whose
system sections and camera intrinsics are kept), puts the session's transformation matrices in
`<project_dir>/camera_sync/` (a multi-camera session calibrates itself first with `mmla
ses-calibrate`: a camera's own fit is taken when it rests on enough paired sightings, else the
given calibration's entry when the session's pairs judge it close, else, for a camera with too
few pairs to judge, the own fit of the same camera pair from the rig's session nearest in time
(checked on sightings shared with a third camera where there are any), else the given entry,
taken when such sightings judge it close or unchecked when nothing judges it; a camera stays out
of the IPS run only when the evidence puts the given entry too far off or the given calibration
has no entry for it; see `choose_matrices`),
launches every base and synchronizer in a
tmux session `replay-<session>` (each in its pipeline's conda environment, logging under
`artifacts/<session>/pipelines/<pipeline>-base/logs/replay_*.log`), sends START on each
pipeline's control channel once all of them wait for it, watches the logs until every video was
read to its end (and the ASR queue drained), sends STOP, ends the session in MongoDB and runs
`mmla ses-fuse`. A pipeline whose events the session already has is skipped unless `--force`.

A session with several microphones replays each personal one (vimo, badge) through its own Vimo
base, bound to its participant, beside the group microphone.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import yaml

from openmmla.collection.recording import default_audio_scope, natural_device_key

AUDIO_PREFERENCE = ('jabra-0', 'vimo-0-ch0', 'vimo-0', 'badge-0')
PERSONAL_BASE_TYPE = 'Vimo'
VIMO_BLOCK = {  # the Vimo block of the local pipelines/asr-base/config.yml, added when the template has none
    'asr_scope': 'participant', 'speaker_verification': 'auto', 'register_duration': 10, 'recognize_duration': 3,
    'recognize_sp_duration': 4, 'recognize_threshold': 0.2, 'recognize_sp_threshold': 0.2, 'keep_threshold': 0.2,
    'keep_sp_threshold': 0.1, 'rms_threshold': 1000, 'rms_peak_threshold': 5000, 'update_threshold': 0.8,
    'gain': 10, 'score_amplified': 'True',
    'stream_kwargs': {'rate': 16000, 'format': 'int16', 'chunk_size': 512, 'buffer_duration': 5.0,
                      'resample_method': 'audio_librosa'}}
REPLAY_EXPIRY_SECONDS = 86400.0  # session seconds: unpaced file bases drift apart, so no bucket expires before STOP flushes it
CAMERA_ANGLE = 'front-top-45'
ENVS = {'asr': 'asr-base', 'vfa': 'vfa-base', 'ips': 'ips-base'}
SYNC_PROGRESS = {'asr': 'Speaker Recognition', 'vfa': 'Features of time bucket', 'ips': 'Uploaded bucket'}  # a synchronizer's log line per bucket
EVENT_OF = {'asr': 'asr_transcription', 'vfa': 'vfa_features', 'ips': 'ips_translation'}
EVENTS_OF = {'asr': ['asr_recognition', 'asr_transcription'], 'vfa': ['vfa_features'],
             'ips': ['ips_translation', 'ips_rotation', 'ips_relation']}  # what --force clears before a pipeline runs again
CALIBRATIONS_DIR = os.path.join('pipelines', 'ips-base', 'camera_sync', 'calibrations')
MIN_INLIERS = 10  # paired sightings a camera's own fit must rest on
MAX_P90_M = 0.15  # and the residual its p90 must stay within
MAX_GIVEN_MEDIAN_M = 0.3  # a given entry scored worse than this on the session's pairs is not used
MAX_FAIR_P90_M = 0.3  # an own fit within this is still taken when the given entry is no better
MAX_BORROWED_MEDIAN_M = 0.3  # a borrowed own fit, or a given entry no pair judged, this far off on the pairs kept to check it is not used (with no such pairs it is taken unchecked)
IPS_CAMERA = 'logitechC920'  # the intrinsics every classroom camera (a C920) is read with, by the IPS bases and ses-calibrate alike
NEAR_WINDOW_S = 0.2  # sightings of a tag this close in time count as near-simultaneous (ses-calibrate -nw)


def log(message: str) -> None:
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


# ---- planning (pure) ----

def calibration_for(session_id: str, cameras: list[str]) -> tuple[str | None, str, list[str]]:
    """(calibration folder or None, main camera, the cameras the IPS run takes) of a session.
    A single camera is its own main without matrices; the wegrow (microscope) rig is the
    2025-06-16 calibration with raspi5-01 (c920-05) as main; the micro:bit rig from 2025-10-15 on
    the calibration of that day, with the flipped-main variant for 2025-10-07, when camera 1
    hung upside down and its video was flipped; an earlier micro:bit session (2024-12-10, two
    cameras) takes the June 2025 calibration of that rig (a, b, c = c920-02/03/04)."""
    date = session_id.split('_')[1]
    cameras = sorted(cameras)
    if len(cameras) == 1:
        return None, cameras[0], cameras
    if '_wegrow_' in session_id:
        main = 'c920-05' if 'c920-05' in cameras else cameras[0]
        return 'wegrow-2025-06-16-m-is-c920-05', main, cameras
    main = 'c920-01' if 'c920-01' in cameras else cameras[0]
    if date == '20251007':
        return 'microbit-2025-10-15-upsidedown-main-flipped', main, cameras
    if date >= '20251015':
        return 'microbit-2025-10-15', main, cameras
    return 'microbit-2025-06-12', main, cameras


def audio_scope(record: dict) -> str | None:
    """a manifest record's scope when it says personal or group, else its device's default."""
    if record.get('scope') in ('personal', 'group'):
        return record['scope']
    return default_audio_scope(record.get('device'), (record.get('imported') or {}).get('method'), record.get('host'))


def plan_session(manifest: dict) -> dict:
    """what a session's replay takes, from its manifest: the group microphone, the personal
    microphones of a session with several (a base each, bound to its participant), the videos,
    the tag size, the sync time and the IPS calibration."""
    recordings = manifest.get('recordings') or []
    audio = {r['device']: r['path'] for r in recordings if '/audio/' in str(r.get('path'))}
    videos = {r['device']: r['path'] for r in recordings if '/video/' in str(r.get('path'))}
    microphone = next((device for device in AUDIO_PREFERENCE if device in audio), None)
    audio_records = [r for r in recordings if '/audio/' in str(r.get('path'))]
    personal: list[dict] = []
    mine = [r for r in audio_records if audio_scope(r) == 'personal']
    # the microphones of known scope, not the files: one recorded in two takes, or beside a file
    # of unknown kind, is still one microphone and replays as before
    microphones = {(r.get('host'), r['device']) for r in audio_records if audio_scope(r) in ('personal', 'group')}
    if len(microphones) >= 2 and mine:  # several microphones and a worn one: a base each
        mine.sort(key=lambda r: (natural_device_key(r.get('device')), str(r.get('host'))))
        hosts_of = defaultdict(set)
        for r in mine:
            hosts_of[r['device']].add(r.get('host'))
        seen = set()
        for r in mine:
            # the host names the base only when the same device was worn on two machines
            pid = r['device'] if len(hosts_of[r['device']]) == 1 else f"{r.get('host')}-{r['device']}"
            if pid in seen:  # the same device and host twice: the first recording is replayed
                continue
            seen.add(pid)
            tag = r.get('participant')
            personal.append({'id': pid, 'device': r['device'], 'path': r['path'],
                             'participant': str(tag) if tag not in (None, '') else pid})
        groups = sorted({r['device'] for r in audio_records if audio_scope(r) == 'group'}, key=natural_device_key)
        microphone = next((d for d in AUDIO_PREFERENCE if d in groups), None) or next(iter(groups), None)
    calibration, main, ips_cameras = calibration_for(manifest['session_id'], list(videos)) if videos else (None, None, [])
    return {
        'session_id': manifest['session_id'],
        'experiment_id': manifest.get('experiment_id'),
        'group_id': manifest.get('group_id'),
        'sync_time': float(manifest['initial_sync_time']),
        'tag_size': float(manifest.get('tag_size') or 0.08),
        'microphone': microphone,
        'audio_path': audio.get(microphone),
        'personal': personal,
        'videos': videos,
        'calibration': calibration,
        'ips_main': main,
        'ips_cameras': ips_cameras,
        'minutes': max((float(r.get('duration') or 0) for r in recordings), default=0.0) / 60.0,
    }


def asr_config(template: dict, plan: dict) -> dict:
    if not plan.get('personal'):
        config = json.loads(json.dumps(template))
        base_type = next(iter(config['Base']))
        config['Base'][base_type]['initial_sync_time'] = plan['sync_time']
        config['Bases'] = [{'id': plan['microphone'], 'base_type': base_type, 'source': 'file', 'source_index': plan['audio_path']}]
        return config
    # the group microphone on the template's own block, every personal one on a Vimo block
    config = json.loads(json.dumps(template))
    blocks = config.setdefault('Base', {})
    group_type = next((k for k in blocks if k != PERSONAL_BASE_TYPE), None)
    vimo = blocks.setdefault(PERSONAL_BASE_TYPE, json.loads(json.dumps(VIMO_BLOCK)))
    vimo['initial_sync_time'] = plan['sync_time']
    bases = []
    if plan['microphone'] and group_type:
        blocks[group_type]['initial_sync_time'] = plan['sync_time']
        vimo['recognize_duration'] = blocks[group_type].get('recognize_duration', vimo.get('recognize_duration', 3))  # one segment grid
        bases.append({'id': plan['microphone'], 'base_type': group_type, 'source': 'file', 'source_index': plan['audio_path']})
    bases += [{'id': p['id'], 'base_type': PERSONAL_BASE_TYPE, 'source': 'file', 'source_index': p['path'],
               'participant': p['participant']} for p in plan['personal']]
    config['Bases'] = bases
    config.setdefault('Synchronizer', {})['result_expiry_time'] = REPLAY_EXPIRY_SECONDS
    return config


VFA_PACE = {1: 4.0, 2: 4.0}  # frame sets per second the bases pace themselves at, by camera count; more cameras keep the template's


def vfa_config(template: dict, plan: dict) -> dict:
    config = json.loads(json.dumps(template))
    config['Base']['initial_sync_time'] = plan['sync_time']
    config['Base']['tag_size'] = plan['tag_size']
    # the bases' pace bounds a replay, not the server (~105 ms a frame): one or two cameras can go faster
    config['Base']['processing_rate'] = VFA_PACE.get(len(plan['videos']), config['Base'].get('processing_rate', 2.0))
    config['Bases'] = [{'id': device, 'camera': IPS_CAMERA, 'source': 'file', 'source_index': path, 'camera_angle': CAMERA_ANGLE}
                       for device, path in sorted(plan['videos'].items())]
    return config


def ips_config(template: dict, plan: dict) -> dict:
    config = json.loads(json.dumps(template))
    config['Base']['initial_sync_time'] = plan['sync_time']
    config['Base']['tag_size'] = plan['tag_size']
    config['Bases'] = [{'id': device, 'camera': IPS_CAMERA, 'source': 'file', 'source_index': plan['videos'][device],
                        'main': device == plan['ips_main']} for device in plan['ips_cameras']]
    return config


def session_day(session_id: str) -> datetime.date | None:
    try:
        return datetime.datetime.strptime(session_id.split('_')[1], '%Y%m%d').date()
    except (IndexError, ValueError):
        return None


def borrowable_fits(project: str, session_id: str, calibration: str | None, main: str) -> dict:
    """{camera: candidate} of the own fits another session of the same rig (the same given
    calibration and main camera) took for the same camera pair, the one nearest in date per camera
    (the one on more pairs when two are as near): the cameras are rarely moved, so such a fit stands
    in for a camera that saw too few tags together with the main one. A candidate is {session,
    matrix, inliers, p90, days}; the fits read are the `own fit` decisions of each session's
    analysis/calibration/matrices_used.json."""
    if not calibration:
        return {}
    day = session_day(session_id)
    best: dict = {}
    for path in sorted(glob.glob(os.path.join(project, 'artifacts', '*', 'analysis', 'calibration', 'matrices_used.json'))):
        other = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        if other == session_id:
            continue
        try:
            used = json.load(open(path))
        except (OSError, ValueError):
            continue
        if used.get('given') != calibration or used.get('main') != main:
            continue
        try:
            report = json.load(open(os.path.join(os.path.dirname(path), 'calibration_report.json')))
        except (OSError, ValueError):
            report = {}
        other_day = session_day(other)
        days = abs((other_day - day).days) if day and other_day else 10 ** 6
        for camera, decision in (used.get('decisions') or {}).items():
            if not str(decision).startswith('own fit') or camera not in (used.get('matrices') or {}):
                continue
            entry = (report.get('cameras') or {}).get(camera) or {}
            candidate = {'session': other, 'matrix': used['matrices'][camera], 'inliers': entry.get('inliers'),
                         'p90': (entry.get('residual_m') or {}).get('p90'), 'days': days}
            key = (days, -(candidate['inliers'] or 0), other)
            if camera not in best or key < best[camera][0]:
                best[camera] = (key, candidate)
    return {camera: candidate for camera, (_, candidate) in best.items()}


def check_on_near_pairs(matrix: dict, near: list | None) -> dict | None:
    """{pairs, direct, via, median, p90} of a transform's residuals (m) on the pairs ses-calibrate
    kept to check a camera with few paired sightings (near_pairs.json): its near-simultaneous
    sightings with the main camera (`direct`) and those shared with a third camera already placed
    (`via`, {camera: pairs}); None when there are none."""
    if not near:
        return None
    p_main = np.array([p['p_main'] for p in near], dtype=float)
    p_alt = np.array([p['p_alt'] for p in near], dtype=float)
    res = np.linalg.norm(p_alt @ np.asarray(matrix['R'], float).T + np.asarray(matrix['T'], float).reshape(3) - p_main, axis=1)
    via: dict = {}
    for p in near:
        if p.get('via'):
            via[p['via']] = via.get(p['via'], 0) + 1
    return {'pairs': len(near), 'direct': len(near) - sum(via.values()), 'via': via,
            'median': round(float(np.median(res)), 3), 'p90': round(float(np.percentile(res, 90)), 3)}


def _on(check: dict) -> str:
    """what a check rested on, in words."""
    parts = [f"{check['direct']} near-simultaneous pairs"] if check['direct'] else []
    parts += [f"{n} pairs through {camera}" for camera, n in check['via'].items()]
    return ' and '.join(parts)


def choose_matrices(report: dict, own: dict, given: dict | None, main: str,
                    min_inliers: int = MIN_INLIERS, max_p90: float = MAX_P90_M,
                    max_given_median: float = MAX_GIVEN_MEDIAN_M, max_fair_p90: float = MAX_FAIR_P90_M,
                    borrowed: dict | None = None, near: dict | None = None,
                    max_borrowed_median: float = MAX_BORROWED_MEDIAN_M) -> tuple[dict, list[str], dict]:
    """which transform each camera of an IPS run takes: its own fit from the session (ses-calibrate's
    report and matrices) when it rests on at least `min_inliers` pairs with a p90 residual within
    `max_p90`; else that fit still, within `max_fair_p90`, when the given entry is no better on the
    same pairs; else the given calibration's entry when the session's pairs scored it within
    `max_given_median`. A camera with fewer than `min_inliers` pairs then takes the own fit of the
    same camera pair borrowed from the rig's session nearest in time (`borrowed`, from
    borrowable_fits); else a given entry no pair judged. Either is checked on the pairs
    ses-calibrate kept for such a camera (`near`, its near_pairs.json: near-simultaneous sightings,
    and sightings shared with a camera already placed) and refused when they put its median residual
    over `max_borrowed_median`; either is taken unchecked when there are no such pairs. The camera
    stays out only when the evidence puts the given entry too far off, or the given calibration has
    no entry for it. Returns (matrices for transformation_matrices_<main>.json, the cameras of the
    run, the decisions)."""
    borrowed, near = borrowed or {}, near or {}
    matrices, cameras, decisions = {}, [main], {}
    for camera, entry in (report.get('cameras') or {}).items():
        fit = entry.get('residual_m') or {}
        scored = (entry.get('given') or {}).get('residual_m') or {}
        own_ok = camera in own and entry.get('inliers', 0) >= min_inliers
        if own_ok and fit.get('p90', 1e9) <= max_p90:
            matrices[camera] = own[camera]
            decisions[camera] = f"own fit ({entry['inliers']} pairs, p90 {fit['p90']} m)"
        elif own_ok and fit.get('p90', 1e9) <= max_fair_p90 and fit.get('median', 1e9) <= scored.get('median', 1e9):
            matrices[camera] = own[camera]
            decisions[camera] = (f"own fit, fair ({entry['inliers']} pairs, p90 {fit['p90']} m"
                                 + (f", the given entry {scored['median']} m off)" if scored else ", no given entry)"))
        elif given and camera in given and scored and scored.get('median', 1e9) <= max_given_median:
            matrices[camera] = given[camera]
            decisions[camera] = f"given calibration ({entry.get('pairs', 0)} pairs to judge it, its residual median {scored['median']} m)"
        else:
            pairs = entry.get('pairs', 0)
            why = f"the given entry is {scored['median']} m off" if scored else 'no given entry'  # an unjudged one is taken below
            few = pairs < min_inliers
            candidate = borrowed.get(camera) if few else None
            refused = ''
            if candidate:
                source = f"borrowed own fit from {candidate['session']} ({candidate['inliers']} pairs, p90 {candidate['p90']} m)"
                check = check_on_near_pairs(candidate['matrix'], near.get(camera))
                if not check or check['median'] <= max_borrowed_median:
                    matrices[camera] = candidate['matrix']
                    decisions[camera] = source + (f", checked on {_on(check)}: median {check['median']} m"
                                                  if check else ', unchecked: no sightings here to check it on')
                    cameras.append(camera)
                    continue
                refused = f"; the {source} is {check['median']} m off on {_on(check)}"
            unjudged = not scored and given and camera in given
            check = check_on_near_pairs(given[camera], near.get(camera)) if unjudged else None
            if check and check['median'] <= max_borrowed_median:
                matrices[camera] = given[camera]
                decisions[camera] = f"given calibration ({pairs} pairs, checked on {_on(check)}: median {check['median']} m){refused}"
                cameras.append(camera)
                continue
            if unjudged and not check:
                # nothing judges the given entry and there is nothing to borrow (a borrowed fit no sighting checks is
                # taken above): the file is the best there is, e.g. a rig no other session shares
                matrices[camera] = given[camera]
                decisions[camera] = 'given calibration (unchecked: no shared sightings, nothing to borrow)'
                cameras.append(camera)
                continue
            if check:
                why = f"the given entry is {check['median']} m off on {_on(check)}"
            borrow = ', no own fit of the pair to borrow' if few and not candidate else ''
            decisions[camera] = (f"left out ({pairs} pairs, own fit p90 {fit.get('p90')} m on {entry.get('inliers', 0)}, "
                                 f"{why}{borrow}{refused})")
            continue
        cameras.append(camera)
    return matrices, sorted(cameras), decisions


def commands(plan: dict, pipelines: list[str], configs: dict[str, str]) -> dict[str, dict[str, str]]:
    """{pipeline: {window: command}} of a session: every base and the synchronizer."""
    out: dict[str, dict[str, str]] = {}
    sid = plan['session_id']
    if 'asr' in pipelines and (plan['microphone'] or plan.get('personal')):
        cfg = configs['asr']
        if not plan.get('personal'):
            out['asr'] = {'sync': f"mmla asr-sync -c {cfg} -sid {sid} -nb 1",
                          plan['microphone']: f"mmla asr-base -c {cfg} -sid {sid} -b {plan['microphone']} -m live -dia true -s false"}
        else:
            # -bt Vimo: the recognize_duration of both blocks is one, so the buckets are right
            group = [plan['microphone']] if plan['microphone'] else []
            n = len(group) + len(plan['personal'])
            out['asr'] = {'sync': f"mmla asr-sync -c {cfg} -sid {sid} -bt {PERSONAL_BASE_TYPE} -nb {n}"}
            for mic in group:
                out['asr'][mic] = f"mmla asr-base -c {cfg} -sid {sid} -b {mic} -m live -dia true -s false"
            for p in plan['personal']:
                out['asr'][p['id']] = f"mmla asr-base -c {cfg} -sid {sid} -b {p['id']} -m live -s false"  # no -dia: dia_* are the group's
    if 'vfa' in pipelines and plan['videos']:
        cfg = configs['vfa']
        out['vfa'] = {'sync': f"mmla vfa-sync -c {cfg} -sid {sid} -nb {len(plan['videos'])} -a false -pose true -gaze true"}
        for device in sorted(plan['videos']):
            out['vfa'][device] = f"mmla vfa-base -c {cfg} -sid {sid} -b {device} -g false -s false"
    if 'ips' in pipelines and plan['ips_cameras']:
        cfg = configs['ips']
        out['ips'] = {'sync': f"mmla ips-sync -c {cfg} -sid {sid} -mc {plan['ips_main']} -p {{project}}"}
        for device in plan['ips_cameras']:
            out['ips'][device] = f"mmla ips-base -c {cfg} -sid {sid} -b {device} -g false -s false -p {{project}}"
    return out


# ---- running ----

LAUNCHER = """#!/bin/bash
# started by scripts/replay_sessions.py: <env> <log> <command>
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$1"
cd "{project}"
export PYTHONUNBUFFERED=1
bash -c "$3" 2>&1 | tee "$2"
echo "== exited at $(date -u +%FT%TZ) ==" | tee -a "$2"
exec sleep 3600
"""


def calibration_written(out_dir: str, main: str, started: float) -> bool:
    """whether ses-calibrate wrote this run's report and matrices (both files, modified since
    `started`), whatever its exit code; a report left from an earlier run does not count."""
    paths = [os.path.join(out_dir, 'calibration_report.json'), os.path.join(out_dir, f'transformation_matrices_{main}.json')]
    return all(os.path.isfile(path) and os.path.getmtime(path) >= started - 1.0 for path in paths)


class Runner:
    def __init__(self, project: str, pipelines: list[str], templates: dict[str, str], force: bool, dry_run: bool):
        self.project, self.pipelines, self.force, self.dry_run = project, pipelines, force, dry_run
        self.templates = {name: yaml.safe_load(open(os.path.join(project, path))) for name, path in templates.items()}
        self.template_paths = templates
        self.clients = None

    # -- clients, made once, only when something is run
    def _clients(self):
        if self.clients is None:
            from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, RedisClientWrapper
            cfg = os.path.join(self.project, self.template_paths['ips'])
            self.clients = (InfluxDBClientWrapper(cfg), MongoDBClientWrapper(cfg), RedisClientWrapper(cfg))
        return self.clients

    def events(self, sid: str, event_type: str) -> int:
        return len(self._clients()[0].query_events(sid, event_type))

    def sh(self, command: str, check: bool = True) -> str:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if check and result.returncode:
            raise RuntimeError(f"{command!r} failed: {result.stderr.strip()[-300:]}")
        return result.stdout

    def count(self, path: str, needle: str) -> int:
        try:
            with open(path, errors='replace') as f:
                return sum(1 for line in f if needle in line)
        except OSError:
            return 0

    def replay(self, sid: str) -> dict:
        manifest = json.load(open(os.path.join(self.project, 'artifacts', sid, 'manifest.json')))
        plan = plan_session(manifest)
        wanted = [p for p in self.pipelines if self.force or self.events(sid, EVENT_OF[p]) == 0] if not self.dry_run else list(self.pipelines)
        personal = [p['id'] for p in plan.get('personal', [])]
        summary = {'session': sid, 'minutes': round(plan['minutes'], 1), 'videos': len(plan['videos']), 'microphone': plan['microphone'],
                   'personal': personal,
                   'calibration': plan['calibration'], 'ips_main': plan['ips_main'], 'pipelines': wanted, 'status': 'planned'}
        log(f"{sid}: {plan['minutes']:.0f} min, {len(plan['videos'])} videos, mic {plan['microphone']}"
            f"{f', {len(personal)} personal mics' if personal else ''}, IPS main {plan['ips_main']} "
            f"with {plan['calibration'] or 'no matrices'} over {plan['ips_cameras']}; pipelines {wanted}")
        if self.dry_run or not wanted:
            summary['status'] = 'dry-run' if self.dry_run else 'skipped (events exist)'
            if not self.dry_run:
                self.fuse(sid)
            return summary
        if self.force:
            for pipeline in wanted:
                self._clients()[0].delete_event_types(sid, EVENTS_OF[pipeline])
                log(f"{sid}: cleared the {pipeline} events of an earlier run")
        logs_of = {p: os.path.join(self.project, 'artifacts', sid, 'pipelines', f'{p}-base', 'logs') for p in wanted}
        for d in logs_of.values():
            os.makedirs(d, exist_ok=True)
            for old in glob.glob(os.path.join(d, 'replay_*.log')):
                os.remove(old)
        if 'ips' in wanted:
            try:
                self.calibrate(plan, self.template_paths['ips'])
            except Exception as e:
                log(f"{sid}: {type(e).__name__}: {e}; IPS runs with the main camera alone")
                plan['ips_cameras'], plan['ips_matrices'] = [plan['ips_main']], {}
            summary['ips_cameras'], summary['ips_decisions'] = plan['ips_cameras'], plan.get('ips_decisions')
        configs = self.write_configs(plan, wanted)
        if 'asr' in wanted and plan['microphone'] and plan.get('personal'):
            written = asr_config(self.templates['asr'], plan)
            if not any(b.get('id') == plan['microphone'] for b in written.get('Bases') or []):
                log(f"{sid}: no group Base block in the ASR template, {plan['microphone']} left out")
                plan['microphone'] = None
        launcher = os.path.join(self.project, 'artifacts', sid, 'pipelines', 'replay_run.sh')
        with open(launcher, 'w') as f:
            f.write(LAUNCHER.format(project=self.project))
        os.chmod(launcher, 0o755)
        session_name = 'replay-' + sid.split('_')[-1]  # the session's start stamp: unique per session
        self.sh(f"tmux kill-session -t {session_name} 2>/dev/null", check=False)
        ordered = commands(plan, wanted, configs)
        started_at = time.time()
        try:
            first = True
            for pipeline, windows in ordered.items():
                for window, command in windows.items():
                    command = command.replace('{project}', self.project)
                    logfile = os.path.join(logs_of[pipeline], f'replay_{window}.log')
                    quoted = command.replace('"', '\\"')
                    spec = f'bash {launcher} {ENVS[pipeline]} {logfile} "{quoted}"'
                    if first:
                        self.sh(f"tmux new-session -d -s {session_name} -n {pipeline}-{window} '{spec}'")
                        first = False
                    else:
                        self.sh(f"tmux new-window -t {session_name} -n {pipeline}-{window} '{spec}'")
            self.ensure_session(plan)
            self.wait_ready(ordered, logs_of)
            for pipeline in ordered:
                receivers = self._clients()[2].publish(f"{sid}/{pipeline}/control", 'START')
                log(f"{sid}: START {pipeline} ({receivers} receivers)")
            self.wait_done(plan, ordered, logs_of)
            summary['status'] = 'done'
        except Exception as e:
            log(f"{sid}: FAILED: {type(e).__name__}: {e}")
            summary['status'] = f'failed: {type(e).__name__}: {str(e)[:120]}'
        finally:
            for pipeline in ordered:
                try:
                    self._clients()[2].publish(f"{sid}/{pipeline}/control", 'STOP')
                except Exception as e:
                    log(f"{sid}: STOP {pipeline} not sent ({type(e).__name__})")
            time.sleep(20)
            try:
                self._clients()[1].end_session(sid)
            except Exception as e:
                log(f"{sid}: end_session failed ({type(e).__name__})")
            self.sh(f"tmux kill-session -t {session_name} 2>/dev/null", check=False)
        summary['minutes_wall'] = round((time.time() - started_at) / 60, 1)
        summary['events'] = {p: self.events(sid, EVENT_OF[p]) for p in self.pipelines}
        self.fuse(sid)
        return summary

    def write_configs(self, plan: dict, wanted: list[str]) -> dict[str, str]:
        sid = plan['session_id']
        makers = {'asr': asr_config, 'vfa': vfa_config, 'ips': ips_config}
        paths = {}
        for pipeline in wanted:
            path = os.path.join('pipelines', f'{pipeline}-base', f'config_replay_{sid}.yml')
            with open(os.path.join(self.project, path), 'w') as f:
                yaml.safe_dump(makers[pipeline](self.templates[pipeline], plan), f, sort_keys=False)
            paths[pipeline] = path
        if 'ips' in wanted:
            camera_sync = os.path.join(self.project, 'camera_sync')
            os.makedirs(camera_sync, exist_ok=True)
            target = os.path.join(camera_sync, f"transformation_matrices_{plan['ips_main']}.json")
            with open(target, 'w') as f:
                json.dump(plan.get('ips_matrices') or {}, f, indent=2)
        return paths

    def calibrate(self, plan: dict, ips_config: str) -> None:
        """a multi-camera session calibrates itself: mmla ses-calibrate over its videos (the given
        calibration scored on the way), then choose_matrices sets the IPS cameras and matrices."""
        sid, main = plan['session_id'], plan['ips_main']
        given_path = os.path.join(self.project, CALIBRATIONS_DIR, plan['calibration'], f'transformation_matrices_{main}.json') if plan['calibration'] else None
        given = json.load(open(given_path)) if given_path and os.path.isfile(given_path) else None
        if len(plan['ips_cameras']) < 2:
            plan['ips_matrices'] = {}
            return
        out_dir = os.path.join(self.project, 'artifacts', sid, 'analysis', 'calibration')
        command = (f"source ~/miniforge3/etc/profile.d/conda.sh && conda activate {ENVS['ips']} && cd {self.project} && "
                   f"mmla ses-calibrate -c {ips_config} -sid {sid} -mc {main} -cams {','.join(plan['ips_cameras'])} -cam {IPS_CAMERA} -st 2 "
                   f"-nw {NEAR_WINDOW_S} -nb {MIN_INLIERS}"
                   + (f" -v {given_path}" if given else ''))
        log(f"{sid}: calibrating from the recordings ({len(plan['ips_cameras'])} cameras)")
        started = time.time()
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
        with open(os.path.join(self.project, 'artifacts', sid, 'pipelines', 'ips-base', 'logs', 'replay_calibrate.log'), 'w') as f:
            f.write(result.stdout + result.stderr)
        report_path = os.path.join(out_dir, 'calibration_report.json')
        written = calibration_written(out_dir, main, started)
        if not written:
            raise RuntimeError(f"ses-calibrate failed (exit {result.returncode}): {result.stderr.strip()[-200:]}")
        if result.returncode:
            # the AprilTag detector can crash the interpreter at exit, after every result is written
            log(f"{sid}: ses-calibrate exited {result.returncode} after writing its results; they are used")
        report = json.load(open(report_path))
        own = json.load(open(os.path.join(out_dir, f'transformation_matrices_{main}.json')))
        near_path = os.path.join(out_dir, 'near_pairs.json')
        near = json.load(open(near_path)) if os.path.isfile(near_path) else {}
        borrowed = borrowable_fits(self.project, sid, plan['calibration'], main)
        matrices, cameras, decisions = choose_matrices(report, own, given, main, borrowed=borrowed, near=near)
        plan['ips_matrices'], plan['ips_cameras'], plan['ips_decisions'] = matrices, cameras, decisions
        # the borrowed fits, with the session they came from, in the report and in what the IPS run took
        lent = {}
        for camera, decision in decisions.items():
            if 'borrowed own fit from' in decision and camera in borrowed:
                candidate = borrowed[camera]
                lent[camera] = {'session': candidate['session'], 'inliers': candidate['inliers'], 'p90': candidate['p90'],
                                'days': candidate['days'], 'check': check_on_near_pairs(candidate['matrix'], near.get(camera)),
                                'taken': camera in matrices}
                report['cameras'][camera]['borrowed'] = lent[camera]
        if lent:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        # what the IPS run took, kept with the session for the archive
        with open(os.path.join(out_dir, 'matrices_used.json'), 'w') as f:
            json.dump({'main': main, 'cameras': cameras, 'matrices': matrices, 'decisions': decisions,
                       'given': plan['calibration'], 'borrowed': lent}, f, indent=2)
        for camera, decision in decisions.items():
            log(f"{sid}: {camera} -> {main}: {decision}")

    def ensure_session(self, plan: dict) -> None:
        mongo = self._clients()[1]
        if not mongo.get_session(plan['session_id']):
            mongo.create_session(plan['session_id'], plan['experiment_id'], plan['group_id'], participants=[],
                                 metadata={'source': 'file replay of the collection recordings (scripts/replay_sessions.py)',
                                           'initial_sync_time': plan['sync_time']})
            log(f"{plan['session_id']}: session document created")

    def wait_ready(self, ordered: dict, logs_of: dict, timeout: float = 600) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            missing = [f"{p}/{w}" for p, windows in ordered.items() for w in windows
                       if self.count(os.path.join(logs_of[p], f'replay_{w}.log'), 'Wait for START') == 0]
            if not missing:
                return
            exited = [f"{p}/{w}" for p, windows in ordered.items() for w in windows
                      if self.count(os.path.join(logs_of[p], f'replay_{w}.log'), 'exited at') > 0]
            if exited:
                raise RuntimeError(f"processes exited before START: {exited}")
            time.sleep(5)
        raise RuntimeError(f"not everyone waits for START after {timeout:.0f} s: {missing}")

    def wait_done(self, plan: dict, ordered: dict, logs_of: dict) -> None:
        """until every video base read its file to the end and the ASR queue drained; a
        pipeline that is done gets its STOP at once so its synchronizer wraps up."""
        expected = plan['minutes'] * 60
        deadline = time.time() + max(30 * 60, 2.5 * expected * 0.5 + 15 * 60)
        pending = dict(ordered)
        stable = {}
        while pending and time.time() < deadline:
            for pipeline in list(pending):
                windows = pending[pipeline]
                bases = [w for w in windows if w != 'sync']
                logs = {w: os.path.join(logs_of[pipeline], f'replay_{w}.log') for w in windows}
                if pipeline == 'asr':
                    heard = sum(self.count(logs[b], 'Speaker Transcription') for b in bases)
                    ended = all(self.count(logs[b], 'Reach the end of the file') > 0 or self.count(logs[b], 'exited at') > 0
                                for b in bases)
                else:
                    # the video bases read their files faster than the server answers, so the frame sets still
                    # queued at the end of the files are done only once the synchronizer's bucket count stops growing
                    heard = self.count(logs['sync'], SYNC_PROGRESS[pipeline])
                    ended = all(self.count(logs[b], 'Reached end of video file') > 0 or self.count(logs[b], 'exited at') > 0 for b in bases)
                stable[pipeline] = stable.get(pipeline, 0) + 1 if ended and heard == stable.get(f'{pipeline}_heard') else 0
                stable[f'{pipeline}_heard'] = heard
                done = ended and stable[pipeline] >= 2
                errors = self.count(logs['sync'], 'Traceback')
                if done:
                    time.sleep(30)  # the last buckets
                    receivers = self._clients()[2].publish(f"{plan['session_id']}/{pipeline}/control", 'STOP')
                    log(f"{plan['session_id']}: {pipeline} done, STOP ({receivers} receivers){f', {errors} tracebacks in its synchronizer log' if errors else ''}")
                    del pending[pipeline]
            if pending:
                progress = ', '.join(f"{p}: {self.count(os.path.join(logs_of[p], 'replay_sync.log'), SYNC_PROGRESS[p])}"
                                     for p in pending)
                log(f"{plan['session_id']}: waiting ({progress})")
                time.sleep(60)
        if pending:
            raise RuntimeError(f"{list(pending)} still running at the deadline")

    def fuse(self, sid: str) -> None:
        out = self.sh(f"cd {self.project} && mmla ses-fuse -c {self.template_paths['ips']} -sid {sid} 2>&1 | tail -1", check=False)
        log(f"{sid}: ses-fuse: {out.strip()[-160:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--project', default=os.getcwd(), help='the repository root that holds artifacts/ (default: the working directory)')
    parser.add_argument('--sessions', nargs='*', help='session ids to replay (default: every artifacts/exp_* with a manifest)')
    parser.add_argument('--pipelines', default='asr,vfa,ips', help='comma-separated: asr, vfa, ips')
    parser.add_argument('--asr-template', default='pipelines/asr-base/config_pilot_260603.yml')
    parser.add_argument('--vfa-template', default='pipelines/vfa-base/config_pilot_260603.yml')
    parser.add_argument('--ips-template', default='pipelines/ips-base/config_pilot_260603.yml')
    parser.add_argument('--force', action='store_true', help='replay a pipeline even when the session already has its events (they are cleared first)')
    parser.add_argument('--dry-run', action='store_true', help='print the plan of every session and launch nothing')
    args = parser.parse_args()
    project = os.path.abspath(args.project)
    pipelines = [p.strip() for p in args.pipelines.split(',') if p.strip()]
    sessions = args.sessions or sorted(os.path.basename(os.path.dirname(m)) for m in glob.glob(os.path.join(project, 'artifacts', 'exp_*', 'manifest.json')))
    runner = Runner(project, pipelines, {'asr': args.asr_template, 'vfa': args.vfa_template, 'ips': args.ips_template}, args.force, args.dry_run)
    summaries = []
    for sid in sessions:
        summaries.append(runner.replay(sid))
        with open(os.path.join(project, 'artifacts', 'replay_batch.jsonl'), 'a') as f:
            f.write(json.dumps(summaries[-1]) + '\n')
    log('summary:')
    for s in summaries:
        log(f"  {s['session']}: {s['status']} {s.get('events', '')} {s.get('minutes_wall', '')}")
    return 0 if all(s['status'] in ('done', 'dry-run') or s['status'].startswith('skipped') for s in summaries) else 1


if __name__ == '__main__':
    sys.exit(main())
