import argparse
import functools
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla vfa-sync",
        description="Run VFA base synchronizer for synchronizing video frames between multiple bases.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('session_id', str, None,
            'session id to use; if not set, choose/create one interactively. Given (as the console does), the '
            'synchronizer runs that session at once, with no menu, and exits when the session is stopped',
            shortname='-sid')
    add_arg('num_bases', int, None,
            "number of bases to synchronize; if not set, asked interactively, or with -sid, the number of entries "
            "in the config's 'Bases' list", shortname='-nb')
    add_arg('actions', bool, None,
            "whether every synchronized frame set is sent for its action labels (the VLM; the vfa_action event); "
            "if not set, what the config's Synchronizer.actions says (true by default)", shortname='-a')
    add_arg('pose', bool, None,
            "whether every synchronized frame set is sent to the frame analyzer's features endpoint for its "
            "skeletons, AprilTags and head yaws (no VLM; the vfa_features event, one per frame set, so set the "
            "bases' keyframe_interval to about 1 second); if not set, what the config's Synchronizer.pose says "
            "(false by default)", shortname='-pose')
    add_arg('gaze', bool, None,
            "whether those features come with the gaze model's gazes (where each person looks: a partner's face or "
            "hands, own hands, a zone); a gaze needs the pose, so this turns the pose on too; if not set, what the "
            "config's Synchronizer.gaze says (false by default)", shortname='-gaze')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.bases.vfa import VFASynchronizer
    from openmmla.utils.args import print_arguments

    print_arguments(args)

    try:
        vfa_synchronizer = VFASynchronizer(
            project_dir=args.project_dir,
            config_path=args.config_path,
            session_id=args.session_id,
            num_bases=args.num_bases,
            actions=args.actions,
            pose=args.pose,
            gaze=args.gaze,
        )
    except Exception as e:
        # it connects to MQTT and MongoDB as it is made, so there is no menu yet to fall back to
        print(f"\nThe VFA synchronizer could not start: {type(e).__name__}: {e}\n"
              f"It connects to MQTT and MongoDB as it starts: check that they are running and reachable "
              f"(System Services on the console) and that {args.config_path} is right, then start it again.")
        sys.exit(1)
    vfa_synchronizer.run()


if __name__ == '__main__':
    main()
