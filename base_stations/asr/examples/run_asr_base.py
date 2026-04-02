"""This script runs the ASRBase of the real-time automatic speech recognition system."""
import argparse
import functools
import os

from openmmla.bases.asr import start_asr_base
from openmmla.utils.args import add_arguments, print_arguments

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, project_dir, 'path to the project directory', shortname='-p')
    add_arg('config_path', str, config_path, 'path to the configuration file', shortname='-c')
    add_arg('mode', str, 'full', 'operating mode', choices=['record', 'recognize', 'full'], shortname='-m')
    add_arg('store', bool, True, 'whether to store audio', shortname='-s')
    add_arg('vad', bool, True, 'whether to use the VAD', shortname='-vad')
    add_arg('nr', bool, True, 'whether to use the denoiser to enhance speech', shortname='-nr')
    add_arg('tr', bool, True, 'whether to transcribe speech to text', shortname='-tr')
    add_arg('sp', bool, False, 'whether to do speech separation for overlapped segment', shortname='-sp')
    add_arg('hsr', bool, True, 'whether to apply Half-Scaled Recognition at speaker boundaries', shortname='-hsr')

    args = parser.parse_args()
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
