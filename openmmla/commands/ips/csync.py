import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ips-csync",
        description="Run camera sync manager for multi-cameras coordinate synchronization.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("project_dir", str, None,
            "path to the project directory; if not set, defaults to the current working directory", shortname="-p")
    add_arg("config_path", str, None, "path to the configuration file", shortname="-c", required=True)
    add_arg("base", str, None,
            "alternative base id from config 'Bases' to synchronize against the main base of its room "
            "(the one flagged main: true); if omitted, choose interactively", shortname="-b")
    add_arg("main", str, None,
            "main base id to synchronize to, when the Bases have a main per room; if omitted, the main of "
            "the room of --base, the only main there is, or choose interactively", shortname="-m")
    add_arg("time_threshold", float, 0.2,
            "seconds apart the main and the alternative detection of a tag may reach the manager "
            "and still be paired", shortname="-t")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.bases.ips import CameraSyncManager
    from openmmla.utils.args import print_arguments

    print_arguments(args)

    try:
        camera_sync_manager = CameraSyncManager(
            project_dir=args.project_dir,
            config_path=args.config_path,
            base=args.base,
            main=args.main,
            time_threshold_sync=args.time_threshold,
            time_threshold_unsync=args.time_threshold,
        )
    except ValueError as e:
        # the Bases of the config cannot be synced: say why, without a traceback
        print(f"\nCamera sync cannot start: {e}")
        raise SystemExit(2)
    camera_sync_manager.run()


if __name__ == "__main__":
    main()
