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
    add_arg('out', str, None, 'where to write transformation_matrices_<main>.json and calibration_report.json; '
            'if not set, artifacts/<session>/analysis/calibration/', shortname='-o')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    from openmmla.utils.args import print_arguments
    print_arguments(args)
    from openmmla.bases.ips.calibration import calibrate, observe, tag_detector
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
    out_dir = args.out or os.path.join(project_dir, 'artifacts', args.session_id, 'analysis', 'calibration')
    os.makedirs(out_dir, exist_ok=True)
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
        print(line)
    print(f"wrote {matrices_path} and calibration_report.json")


if __name__ == "__main__":
    main()
