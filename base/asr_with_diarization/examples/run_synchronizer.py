"""This script runs the synchronizer."""
import argparse
import functools
import os

from openmmla.bases.asr_with_diarization import AudioSynchronizer
from openmmla.utils.args import add_arguments, print_arguments

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = 'config.yml'


def run_synchronizer(args):
    synchronizer = AudioSynchronizer(base_type=args.base_type, project_dir=project_dir, config_path=args.config_path,
                                     dominant=args.dominant, sp=args.sp)
    synchronizer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('base_type', str, None, 'audio base type.', shortname='-b')
    add_arg('project_dir', str, project_dir, 'path to the project directory', shortname='-p')
    add_arg('config_path', str, config_path, 'path to the configuration file', shortname='-c')
    add_arg('dominant', bool, False, 'whether to select the dominant speaker or not', shortname='-d')
    add_arg('sp', bool, False, 'whether the audio bases do speech separation or not', shortname='-s')

    input_args = parser.parse_args()
    print_arguments(input_args)
    run_synchronizer(input_args)
