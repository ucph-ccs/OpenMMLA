import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla asr-base",
        description="Run ASR base of real-time audio analyzer",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('mode', str, 'record', 'operating mode', choices=['record', 'recognize', 'full'], shortname='-m')
    add_arg('store', bool, True, 'whether to store audio', shortname='-s')
    add_arg('vad', bool, True, 'whether to use VAD', shortname='-vad')
    add_arg('nr', bool, True, 'whether to use noise reduction', shortname='-nr')
    add_arg('tr', bool, True, 'whether to transcribe speech to text', shortname='-tr')
    add_arg('sp', bool, False, 'whether to do speech separation', shortname='-sp')
    add_arg('hsr', bool, True, 'whether to apply Half-Scaled Recognition at speaker boundaries', shortname='-hsr')
    return parser


def main():
    """Main function."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Only import when actually running the logic
    from openmmla.bases.asr import start_asr_base
    from openmmla.utils.args import print_arguments

    print_arguments(args)
    
    # Call the centralized start_asr_base function with restart capability
    start_asr_base(
        project_dir=args.project_dir,
        config_path=args.config_path,
        mode=args.mode,
        store=args.store,
        vad=args.vad,
        nr=args.nr,
        tr=args.tr,
        sp=args.sp,
        hsr=args.hsr
    )


if __name__ == "__main__":
    main()
