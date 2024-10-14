"""This script runs the Jabra audio base."""
import argparse
import functools
import os

from openmmla.bases.asr_with_diarization import JabraAudioBase
from openmmla.utils.args import add_arguments, print_arguments

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = 'config.yml'


def run_audio_base(args):
    jabra_audio_base = JabraAudioBase(project_dir=args.project_dir, config_path=args.config_path, mode=args.mode,
                                      local=args.local, vad=args.vad, nr=args.nr, tr=args.tr, sp=args.sp,
                                      store=args.store)
    jabra_audio_base.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, project_dir, 'path to the project directory', shortname='-p')
    add_arg('config_path', str, config_path, 'path to the configuration file', shortname='-c')
    add_arg('mode', str, 'full', 'operating mode', choices=['record', 'recognize', 'full'], shortname='-m')
    add_arg('local', bool, False, 'whether to run the audio base locally', shortname='-l')
    add_arg('vad', bool, True, 'whether to use the VAD', shortname='-v')
    add_arg('nr', bool, True, 'whether to use the denoiser to enhance speech', shortname='-n')
    add_arg('tr', bool, True, 'whether to transcribe speech to text', shortname='-t')
    add_arg('sp', bool, False, 'whether to do speech separation for overlapped segment', shortname='-s')
    add_arg('store', bool, True, 'whether to store audio', shortname='-st')

    input_args = parser.parse_args()
    print_arguments(input_args)
    run_audio_base(input_args)
