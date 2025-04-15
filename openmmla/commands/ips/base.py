import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ips-base",
        description="Run IPS base of real-time indoor positioning system.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('graphics', bool, True, 'whether to display video frames', shortname='-g')
    add_arg('store', bool, False, 'whether to store video frames', shortname='-s')
    add_arg('verbose', bool, False, 'whether to print debug information', shortname='-v')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.bases.ips import IPSBase
    from openmmla.utils.args import print_arguments

    print_arguments(args)

    video_base = IPSBase(
        project_dir=args.project_dir,
        config_path=args.config_path,
        graphics=args.graphics,
        store=args.store,
        verbose=args.verbose
    )
    video_base.run()


if __name__ == '__main__':
    main()
