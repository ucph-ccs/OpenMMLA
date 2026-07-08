import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla asr-sync",
        description="Run ASR synchronizer of real-time audio analyzer.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)

    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('dominant', bool, False, 'whether to select the dominant speaker or not', shortname='-d')
    add_arg('sp', bool, False, 'whether the audio bases do speech separation or not', shortname='-sp')
    add_arg('session_id', str, None, 'session id to use; if not set, choose/create one interactively',
            shortname='-sid')
    return parser


def main():
    """Main function."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Only import when actually running the logic
    from openmmla.bases.asr import start_asr_synchronizer
    from openmmla.utils.args import print_arguments

    print_arguments(args)
    
    # Call the centralized start_asr_synchronizer function with restart capability
    start_asr_synchronizer(
        project_dir=args.project_dir,
        config_path=args.config_path,
        mode='live',  # Default mode for synchronizer
        dominant=args.dominant,
        sp=args.sp,
        session_id=args.session_id
    )


if __name__ == "__main__":
    main()
