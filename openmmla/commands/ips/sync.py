import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ips-sync",
        description="Run IPS synchronizer of real-time indoor positioning system.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('verbose', bool, False, 'whether to print debug information', shortname='-v')
    add_arg('session_id', str, None, 'session id to use; if not set, choose/create one interactively',
            shortname='-sid')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.bases.ips import IPSSynchronizer
    from openmmla.utils.args import print_arguments

    print_arguments(args)

    ips_synchronizer = IPSSynchronizer(
        project_dir=args.project_dir,
        config_path=args.config_path,
        verbose=args.verbose,
        session_id=args.session_id
    )
    ips_synchronizer.run()


if __name__ == "__main__":
    main()
