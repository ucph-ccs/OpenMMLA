"""This script runs the ASRPostAnalyzer of the post-time automatic speech recognition system."""
import argparse
import functools
import os

from openmmla.bases.asr import ASRPostAnalyzer
from openmmla.utils.args import add_arguments, print_arguments

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')


def run_asr_post_analyzer(args):
    print("\033]0;Audio Post Analyzer\007")
    post_audio_analyzer = ASRPostAnalyzer(project_dir=args.project_dir, config_path=args.config_path,
                                          vad=args.vad, nr=args.nr, sp=args.sp, tr=args.tr)
    post_audio_analyzer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, project_dir, 'path to the project directory', shortname='-p')
    add_arg('config_path', str, config_path, 'path to the configuration file', shortname='-c')
    add_arg('vad', bool, True, 'whether to use the VAD or not', shortname='-vad')
    add_arg('nr', bool, True, 'whether to use the denoiser to enhance speech or not', shortname='-nr')
    add_arg('sp', bool, False, 'whether to use the separation model or not', shortname='-sp')
    add_arg('tr', bool, True, 'whether to transcribe the audio segments or not', shortname='-tr')

    input_args = parser.parse_args()
    print_arguments(input_args)

    run_asr_post_analyzer(args=input_args)
