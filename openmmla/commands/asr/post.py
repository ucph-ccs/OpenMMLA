import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla asr-post",
        description="Run ASR post-time audio analyser.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)

    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('vad', bool, True, 'whether to use the VAD or not', shortname='-vad')
    add_arg('nr', bool, True, 'whether to use the denoiser to enhance speech or not', shortname='-nr')
    add_arg('sp', bool, False, 'whether to use the separation model or not', shortname='-sp')
    add_arg('tr', bool, True, 'whether to transcribe the audio segments or not', shortname='-tr')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.bases.asr import ASRPostAnalyzer
    from openmmla.utils.args import print_arguments

    print("\033]0;Audio Post Analyzer\007")  # Set terminal title
    print_arguments(args)

    post_audio_analyzer = ASRPostAnalyzer(
        project_dir=args.project_dir,
        config_path=args.config_path,
        vad=args.vad,
        nr=args.nr,
        sp=args.sp,
        tr=args.tr
    )
    post_audio_analyzer.run()


if __name__ == "__main__":
    main()
