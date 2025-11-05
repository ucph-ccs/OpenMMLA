import argparse
import functools
import os


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-ctl",
        description="Control bucket session.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    return parser


def run_session_control(args):
    """Main session control function with restart capability."""
    print(f"\033]0;Session Control\007")
    
    from openmmla.utils.control import start_control

    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    start_control(config_path)


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.utils.args import print_arguments
    print_arguments(args)

    run_session_control(args)


if __name__ == "__main__":
    main()
