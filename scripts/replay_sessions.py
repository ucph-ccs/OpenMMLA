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
given calibration's entry, else the camera stays out of the IPS run; see `choose_matrices`),
launches every base and synchronizer in a
tmux session `replay-<session>` (each in its pipeline's conda environment, logging under
`artifacts/<session>/pipelines/<pipeline>-base/logs/replay_*.log`), sends START on each
pipeline's control channel once all of them wait for it, watches the logs until every video was
read to its end (and the ASR queue drained), sends STOP, ends the session in MongoDB and runs
`mmla ses-fuse`. A pipeline whose events the session already has is skipped unless `--force`.
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

import yaml

AUDIO_PREFERENCE = ('jabra-0', 'vimo-0-ch0', 'vimo-0', 'badge-0')
CAMERA_ANGLE = 'front-top-45'
ENVS = {'asr': 'asr-base', 'vfa': 'vfa-base', 'ips': 'ips-base'}
EVENT_OF = {'asr': 'asr_transcription', 'vfa': 'vfa_features', 'ips': 'ips_translation'}
EVENTS_OF = {'asr': ['asr_recognition', 'asr_transcription'], 'vfa': ['vfa_features'],
             'ips': ['ips_translation', 'ips_rotation', 'ips_relation']}  # what --force clears before a pipeline runs again
CALIBRATIONS_DIR = os.path.join('pipelines', 'ips-base', 'camera_sync', 'calibrations')
MIN_INLIERS = 30  # paired sightings a camera's own fit must rest on
MAX_P90_M = 0.15  # and the residual its p90 must stay within


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


def plan_session(manifest: dict) -> dict:
    """what a session's replay takes, from its manifest: the group microphone, the videos, the
    tag size, the sync time and the IPS calibration."""
    recordings = manifest.get('recordings') or []
    audio = {r['device']: r['path'] for r in recordings if '/audio/' in str(r.get('path'))}
    videos = {r['device']: r['path'] for r in recordings if '/video/' in str(r.get('path'))}
    microphone = next((device for device in AUDIO_PREFERENCE if device in audio), None)
    calibration, main, ips_cameras = calibration_for(manifest['session_id'], list(videos)) if videos else (None, None, [])
    return {
        'session_id': manifest['session_id'],
        'experiment_id': manifest.get('experiment_id'),
        'group_id': manifest.get('group_id'),
        'sync_time': float(manifest['initial_sync_time']),
        'tag_size': float(manifest.get('tag_size') or 0.08),
        'microphone': microphone,
        'audio_path': audio.get(microphone),
        'videos': videos,
        'calibration': calibration,
        'ips_main': main,
        'ips_cameras': ips_cameras,
        'minutes': max((float(r.get('duration') or 0) for r in recordings), default=0.0) / 60.0,
    }


def asr_config(template: dict, plan: dict) -> dict:
    config = json.loads(json.dumps(template))
    base_type = next(iter(config['Base']))
    config['Base'][base_type]['initial_sync_time'] = plan['sync_time']
    config['Bases'] = [{'id': plan['microphone'], 'base_type': base_type, 'source': 'file', 'source_index': plan['audio_path']}]
    return config


def vfa_config(template: dict, plan: dict) -> dict:
    config = json.loads(json.dumps(template))
    config['Base']['initial_sync_time'] = plan['sync_time']
    config['Base']['tag_size'] = plan['tag_size']
    config['Bases'] = [{'id': device, 'camera': 'logitechC920', 'source': 'file', 'source_index': path, 'camera_angle': CAMERA_ANGLE}
                       for device, path in sorted(plan['videos'].items())]
    return config


def ips_config(template: dict, plan: dict) -> dict:
    config = json.loads(json.dumps(template))
    config['Base']['initial_sync_time'] = plan['sync_time']
    config['Base']['tag_size'] = plan['tag_size']
    config['Bases'] = [{'id': device, 'camera': 'logitechC920', 'source': 'file', 'source_index': plan['videos'][device],
                        'main': device == plan['ips_main']} for device in plan['ips_cameras']]
    return config


def choose_matrices(report: dict, own: dict, given: dict | None, main: str,
                    min_inliers: int = MIN_INLIERS, max_p90: float = MAX_P90_M) -> tuple[dict, list[str], dict]:
    """which transform each camera of an IPS run takes: its own fit from the session (ses-calibrate's
    report and matrices) when it rests on at least `min_inliers` pairs with a p90 residual within
    `max_p90`, else the given calibration's entry, else none, and the camera stays out. Returns
    (matrices for transformation_matrices_<main>.json, the cameras of the run, the decisions)."""
    matrices, cameras, decisions = {}, [main], {}
    for camera, entry in (report.get('cameras') or {}).items():
        fit = entry.get('residual_m') or {}
        if camera in own and entry.get('inliers', 0) >= min_inliers and fit.get('p90', 1e9) <= max_p90:
            matrices[camera] = own[camera]
            decisions[camera] = f"own fit ({entry['inliers']} pairs, p90 {fit['p90']} m)"
        elif given and camera in given:
            matrices[camera] = given[camera]
            scored = (entry.get('given') or {}).get('residual_m') or {}
            decisions[camera] = f"given calibration ({entry.get('pairs', 0)} pairs to judge it" + \
                                (f", its residual median {scored['median']} m)" if scored else ")")
        else:
            decisions[camera] = f"left out ({entry.get('pairs', 0)} pairs, no given entry)"
            continue
        cameras.append(camera)
    return matrices, sorted(cameras), decisions


def commands(plan: dict, pipelines: list[str], configs: dict[str, str]) -> dict[str, dict[str, str]]:
    """{pipeline: {window: command}} of a session: every base and the synchronizer."""
    out: dict[str, dict[str, str]] = {}
    sid = plan['session_id']
    if 'asr' in pipelines and plan['microphone']:
        cfg = configs['asr']
        out['asr'] = {'sync': f"mmla asr-sync -c {cfg} -sid {sid} -nb 1",
                      plan['microphone']: f"mmla asr-base -c {cfg} -sid {sid} -b {plan['microphone']} -m live -dia true -s false"}
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
        summary = {'session': sid, 'minutes': round(plan['minutes'], 1), 'videos': len(plan['videos']), 'microphone': plan['microphone'],
                   'calibration': plan['calibration'], 'ips_main': plan['ips_main'], 'pipelines': wanted, 'status': 'planned'}
        log(f"{sid}: {plan['minutes']:.0f} min, {len(plan['videos'])} videos, mic {plan['microphone']}, IPS main {plan['ips_main']} "
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
                   f"mmla ses-calibrate -c {ips_config} -sid {sid} -mc {main} -cams {','.join(plan['ips_cameras'])} -st 2"
                   + (f" -v {given_path}" if given else ''))
        log(f"{sid}: calibrating from the recordings ({len(plan['ips_cameras'])} cameras)")
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
        with open(os.path.join(self.project, 'artifacts', sid, 'pipelines', 'ips-base', 'logs', 'replay_calibrate.log'), 'w') as f:
            f.write(result.stdout + result.stderr)
        report_path = os.path.join(out_dir, 'calibration_report.json')
        if result.returncode or not os.path.isfile(report_path):
            raise RuntimeError(f"ses-calibrate failed: {result.stderr.strip()[-200:]}")
        report = json.load(open(report_path))
        own = json.load(open(os.path.join(out_dir, f'transformation_matrices_{main}.json')))
        matrices, cameras, decisions = choose_matrices(report, own, given, main)
        plan['ips_matrices'], plan['ips_cameras'], plan['ips_decisions'] = matrices, cameras, decisions
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
                    base = logs[bases[0]]
                    heard = self.count(base, 'Speaker Transcription')
                    ended = self.count(base, 'Reach the end of the file') > 0 or self.count(base, 'exited at') > 0
                    stable[pipeline] = stable.get(pipeline, 0) + 1 if ended and heard == stable.get('asr_heard') else 0
                    stable['asr_heard'] = heard
                    done = ended and stable[pipeline] >= 2
                else:
                    done = all(self.count(logs[b], 'Reached end of video file') > 0 or self.count(logs[b], 'exited at') > 0 for b in bases)
                errors = self.count(logs['sync'], 'Traceback')
                if done:
                    time.sleep(30)  # the last buckets
                    receivers = self._clients()[2].publish(f"{plan['session_id']}/{pipeline}/control", 'STOP')
                    log(f"{plan['session_id']}: {pipeline} done, STOP ({receivers} receivers){f', {errors} tracebacks in its synchronizer log' if errors else ''}")
                    del pending[pipeline]
            if pending:
                progress = ', '.join(f"{p}: {self.count(os.path.join(logs_of[p], 'replay_sync.log'), needle)}"
                                     for p, needle in (('asr', 'Speaker Recognition'), ('vfa', 'Features of time bucket'), ('ips', 'Uploaded bucket')) if p in pending)
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
