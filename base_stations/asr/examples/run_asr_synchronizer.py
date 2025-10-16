"""This script runs ASRSynchronizer of the real-time automatic speech recognition system."""
import argparse
import functools
import os

from openmmla.bases.asr import start_asr_synchronizer
from openmmla.utils.args import add_arguments, print_arguments

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, project_dir, 'path to the project directory', shortname='-p')
    add_arg('config_path', str, config_path, 'path to the configuration file', shortname='-c')
    add_arg('mode', str, 'full', 'operating mode', choices=['recognize', 'full'], shortname='-m')
    add_arg('dominant', bool, False, 'whether to select the dominant speaker or not', shortname='-d')
    add_arg('sp', bool, False, 'whether the audio bases do speech separation or not', shortname='-sp')

    args = parser.parse_args()
    print_arguments(args)
    
    # Call the centralized start_asr_synchronizer function with restart capability
    start_asr_synchronizer(
        project_dir=args.project_dir,
        config_path=args.config_path,
        mode=args.mode,
        dominant=args.dominant,
        sp=args.sp
    )


if __name__ == "__main__":
    main()
