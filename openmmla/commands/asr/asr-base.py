import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run ASR audio base for speaker recognition and transcription.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('base_type', str, None, 'audio base type', shortname='-b', required=True)
    add_arg('mode', str, 'full', 'operating mode', choices=['record', 'recognize', 'full'], shortname='-m')
    add_arg('store', bool, True, 'whether to store audio', shortname='-s')
    add_arg('vad', bool, True, 'whether to use VAD', shortname='-vad')
    add_arg('nr', bool, True, 'whether to use noise reduction', shortname='-nr')
    add_arg('tr', bool, True, 'whether to transcribe speech to text', shortname='-tr')
    add_arg('sp', bool, False, 'whether to do speech separation', shortname='-sp')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Only import when actually running the logic
    from openmmla.bases.asr import AudioBase
    from openmmla.utils.args import print_arguments

    print_arguments(args)
    badge_audio_base = AudioBase(
        project_dir=args.project_dir,
        config_path=args.config_path,
        base_type=args.base_type,
        mode=args.mode,
        vad=args.vad,
        nr=args.nr,
        tr=args.tr,
        sp=args.sp,
        store=args.store
    )
    badge_audio_base.run()


if __name__ == "__main__":
    main()
