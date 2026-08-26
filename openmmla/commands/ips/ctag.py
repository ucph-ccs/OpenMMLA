import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ips-ctag",
        description="Run camera tag detector for multi-cameras coordinate synchronization.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("project_dir", str, None,
            "path to the project directory; if not set, defaults to the current working directory", shortname="-p")
    add_arg("config_path", str, None, "path to the configuration file", shortname="-c", required=True)
    add_arg("graphics", bool, True,
            "whether to display annotated video frames; set false for headless/remote runs", shortname="-g")
    add_arg("headless", bool, False,
            "no display (forces graphics off); for remote runs", shortname="-hl")
    add_arg("base", str, None,
            "base id from config 'Bases' (pulls camera/source/id from that entry); "
            "if omitted, choose interactively", shortname="-b")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.bases.ips import CameraTagDetector
    from openmmla.utils.args import print_arguments

    print_arguments(args)

    camera_tag_detector = CameraTagDetector(
        project_dir=args.project_dir,
        config_path=args.config_path,
        graphics=args.graphics,
        headless=args.headless,
        base=args.base,
    )
    camera_tag_detector.run()


if __name__ == "__main__":
    main()
