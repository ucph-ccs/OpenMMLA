"""This script runs the ASRBase of the real-time automatic speech recognition system."""
import argparse
import functools
import os

from openmmla.bases.asr import ASRBase
from openmmla.utils.args import add_arguments, print_arguments

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')


def run_asr_base(args):
    asr_base = ASRBase(project_dir=args.project_dir, config_path=args.config_path, base_type=args.base_type,
                       mode=args.mode, vad=args.vad, nr=args.nr, tr=args.tr, sp=args.sp, store=args.store, hsr=args.hsr)
    asr_base.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, project_dir, 'path to the project directory', shortname='-p')
    add_arg('config_path', str, config_path, 'path to the configuration file', shortname='-c')
    add_arg('base_type', str, None, 'audio base type', shortname='-b')
    add_arg('mode', str, 'full', 'operating mode', choices=['record', 'recognize', 'full'], shortname='-m')
    add_arg('store', bool, True, 'whether to store audio', shortname='-s')
    add_arg('vad', bool, True, 'whether to use the VAD', shortname='-vad')
    add_arg('nr', bool, True, 'whether to use the denoiser to enhance speech', shortname='-nr')
    add_arg('tr', bool, True, 'whether to transcribe speech to text', shortname='-tr')
    add_arg('sp', bool, False, 'whether to do speech separation for overlapped segment', shortname='-sp')
    add_arg('hsr', bool, True, 'whether to apply Half-Scaled Recognition at speaker boundaries', shortname='-hsr')

    input_args = parser.parse_args()
    print_arguments(input_args)
    run_asr_base(input_args)
