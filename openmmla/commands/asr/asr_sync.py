import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run ASR audio synchronizer for synchronizing results from audio bases.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)

    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('base_type', str, None, 'audio base type', shortname='-b', required=True)
    add_arg('dominant', bool, False, 'whether to select the dominant speaker or not', shortname='-d')
    add_arg('sp', bool, False, 'whether the audio bases do speech separation or not', shortname='-sp')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.bases.asr import AudioSynchronizer
    from openmmla.utils.args import print_arguments

    print_arguments(args)
    synchronizer = AudioSynchronizer(
        project_dir=args.project_dir,
        config_path=args.config_path,
        base_type=args.base_type,
        dominant=args.dominant,
        sp=args.sp
    )
    synchronizer.run()


if __name__ == "__main__":
    main()
