import argparse
import functools
import json
import os


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-calibrate",
        description="Compute a session's transformation matrices between its cameras from its own recordings "
                    "(the tags two cameras saw at the same moments), and check given matrices against them.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory (holds artifacts/); if not set, the current working directory', shortname='-p')
    add_arg('config_path', str, None,
            'an IPS base config: its Cameras section gives the intrinsics, its Base section the tag family and rotation',
            shortname='-c', required=True)
    add_arg('session_id', str, None, 'the session (artifacts/<session>/manifest.json names its videos)', shortname='-sid',
            required=True)
    add_arg('main', str, None, 'the main camera (a device of the manifest, e.g. c920-01); if not set, c920-01, else c920-05, '
            'else the first', shortname='-mc')
    add_arg('cameras', str, None, 'comma-separated devices to take; if not set, every video of the manifest', shortname='-cams')
    add_arg('camera', str, None, "the Cameras entry (intrinsics) the videos were recorded with; if not set, the config's first",
            shortname='-cam')
    add_arg('step', float, 2.0, 'seconds between the sampled frames', shortname='-st')
    add_arg('tag_size', float, None, "the AprilTag size in metres; if not set, the manifest's, else the config's", shortname='-ts')
    add_arg('verify', str, None, 'a transformation_matrices_<main>.json to score on the same paired sightings '
            '(e.g. one of pipelines/ips-base/camera_sync/calibrations/<name>/)', shortname='-v')
    add_arg('near_window', float, 0.2, 'for a camera with fewer than -nb paired sightings, the most seconds apart two sightings '
            'of a tag may lie to count as near-simultaneous (extra frames are read around the sampled ones); 0 turns it off',
            shortname='-nw')
    add_arg('near_below', int, 10, 'the paired sightings under which a camera is searched for near-simultaneous ones and for '
            'sightings shared with a third camera whose own fit rests on at least this many pairs', shortname='-nb')
    add_arg('out', str, None, 'where to write transformation_matrices_<main>.json and calibration_report.json; '
            'if not set, artifacts/<session>/analysis/calibration/', shortname='-o')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    from openmmla.utils.args import print_arguments
    print_arguments(args)
    from openmmla.bases.ips.calibration import (calibrate, near_pairs, near_stamps, observe, relayed_pairs, tag_detector,
                                                verify)
    from openmmla.utils.config import load_yaml_config

    project_dir = os.path.abspath(args.project_dir or os.getcwd())
    manifest_path = os.path.join(project_dir, 'artifacts', args.session_id, 'manifest.json')
    if not os.path.isfile(manifest_path):
        parser.error(f"no manifest at {manifest_path}")
    manifest = json.load(open(manifest_path))
    config = load_yaml_config(args.config_path)
    cameras = config.get('Cameras') or {}
    if not cameras:
        parser.error(f"{args.config_path} has no Cameras section")
    camera = args.camera or sorted(cameras)[0]
    if camera not in cameras:
        parser.error(f"Cameras has no entry {camera!r}: {sorted(cameras)}")
    if cameras[camera].get('fisheye'):
        parser.error(f"camera {camera} is fisheye: ses-calibrate reads the frames as they are")
    base = config.get('Base') or {}
    tag_size = args.tag_size or float(manifest.get('tag_size') or base.get('tag_size') or 0.08)
    families = str(base.get('families') or 'tag36h11')
    rotate = int(base.get('rotate') or 0)

    videos = {r['device']: r for r in manifest.get('recordings', []) if '/video/' in str(r.get('path'))}
    wanted = [c.strip() for c in args.cameras.split(',')] if args.cameras else sorted(videos)
    missing = [c for c in wanted if c not in videos]
    if missing:
        parser.error(f"the manifest has no video of {missing}; it has {sorted(videos)}")
    main_camera = args.main or next((c for c in ('c920-01', 'c920-05') if c in wanted), wanted[0])
    if main_camera not in wanted:
        parser.error(f"main camera {main_camera} is not among {wanted}")
    if len(wanted) < 2:
        parser.error("at least two cameras are needed")

    sync = float(manifest['initial_sync_time'])
    end = min(float(videos[c]['start_time']) + float(videos[c].get('duration') or 0) for c in wanted)
    stamps = [sync + k * args.step for k in range(int((end - sync) / args.step) + 1)]
    print(f"{len(stamps)} frames per camera every {args.step} s from {sync:.3f} ({(end - sync) / 60:.1f} min); "
          f"tag size {tag_size} m, family {families}, camera {camera}, main {main_camera}")
    detect = tag_detector(cameras[camera]['params'], tag_size, families)
    observations = {}
    for device in wanted:
        obs = observe(videos[device]['path'], float(videos[device]['start_time']), stamps, detect, rotate=rotate,
                      progress=lambda i, n, d=device: print(f"  {d}: frame {i}, {n} with tags", flush=True))
        tags = sorted({tag for poses in obs.values() for tag in poses})
        print(f"{device}: {len(obs)} of {len(stamps)} frames hold tags {tags}", flush=True)
        observations[device] = obs

    given = json.load(open(args.verify)) if args.verify else None
    matrices, report = calibrate(observations, main_camera, given)
    # a camera with few paired sightings: the frames around the moments the two cameras saw a tag one
    # sampled step apart are read as well, and the sightings within near_window seconds are kept, with
    # those it shares with a third camera whose own fit is good (taken into the main frame by that fit);
    # they check a transform from another session or file (near_pairs.json), they do not fit one
    near_found = {}
    placed = [c for c, e in report['cameras'].items()
              if c in matrices and e.get('inliers', 0) >= args.near_below and (e.get('residual_m') or {}).get('p90', 1e9) <= 0.15]
    for alt, entry in report['cameras'].items():
        if args.near_window <= 0 or entry['pairs'] >= args.near_below:
            continue
        extra = near_stamps(observations[main_camera], observations[alt], args.step, args.near_window)
        fine = {device: observe(videos[device]['path'], float(videos[device]['start_time']), extra, detect, rotate=rotate) if extra else {}
                for device in (main_camera, alt)}
        found = [(p, None) for p in near_pairs({**observations[main_camera], **fine[main_camera]},
                                               {**observations[alt], **fine[alt]}, args.near_window)]
        via = {}
        for other in placed:
            if other != alt:
                relayed = relayed_pairs(observations[other], observations[alt], matrices[other]['R'], matrices[other]['T'])
                found += [(p, other) for p in relayed]
                if relayed:
                    via[other] = len(relayed)
        entry['near'] = {'pairs': len(found) - sum(via.values()), 'max_gap_s': args.near_window, 'frames_read': len(extra),
                         'via': via, 'tags': sorted({int(p[1]) for p, _ in found})}
        if given and alt in given and found:
            entry['near']['given'] = verify(given[alt]['R'], given[alt]['T'], [p for p, _ in found])
        near_found[alt] = [{'stamp': round(float(p[0]), 3), 'tag': int(p[1]), 'gap_s': p[6], 'via': other,
                            'p_main': [round(float(x), 4) for x in p[2]], 'p_alt': [round(float(x), 4) for x in p[3]]}
                           for p, other in found]
    out_dir = args.out or os.path.join(project_dir, 'artifacts', args.session_id, 'analysis', 'calibration')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'near_pairs.json'), 'w') as f:
        json.dump(near_found, f)
    matrices_path = os.path.join(out_dir, f'transformation_matrices_{main_camera}.json')
    with open(matrices_path, 'w') as f:
        json.dump(matrices, f, indent=2)
    report.update({'session_id': args.session_id, 'step': args.step, 'frames': len(stamps), 'tag_size': tag_size,
                   'camera': camera, 'verified': args.verify})
    with open(os.path.join(out_dir, 'calibration_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    for alt, entry in report['cameras'].items():
        line = f"{alt} -> {main_camera}: {entry['pairs']} paired sightings"
        if 'inliers' in entry:
            line += (f", fit on {entry['inliers']}: residual median {entry['residual_m']['median']} m, p90 {entry['residual_m']['p90']} m"
                     f"; pose-to-pose average differs by {entry['pose_average']['rotation_difference_deg']} deg / "
                     f"{entry['pose_average']['translation_difference_m']} m")
        if 'given' in entry:
            g = entry['given']
            line += (f"\n    given: residual median {g['residual_m']['median']} m, p90 {g['residual_m']['p90']} m, "
                     f"{g['within_0.15_m'] * 100:.0f} % within 0.15 m")
            if 'difference_to_fit' in g:
                line += f"; {g['difference_to_fit']['rotation_deg']} deg / {g['difference_to_fit']['translation_m']} m from the fit"
        if 'problem' in entry:
            line += f": {entry['problem']}"
        if 'near' in entry:
            line += (f"\n    near-simultaneous (within {entry['near']['max_gap_s']} s, {entry['near']['frames_read']} extra frames "
                     f"per camera): {entry['near']['pairs']} pairs"
                     + ''.join(f", {n} through {other}" for other, n in entry['near']['via'].items()))
            if 'given' in entry['near']:
                line += f"; the given entry's residual median on them {entry['near']['given']['residual_m']['median']} m"
        print(line)
    print(f"wrote {matrices_path}, calibration_report.json and near_pairs.json")


if __name__ == "__main__":
    main()
