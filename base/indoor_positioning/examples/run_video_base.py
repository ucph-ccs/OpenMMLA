"""This script runs the video bases."""
import argparse
import functools
import os
import sys

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = 'conf/video_base.ini'
sys.path.append(project_dir)

from openmmla_vision.utils.args_utils import add_arguments, print_arguments
from openmmla_vision.bases.video_base import VideoBase


def run_video_base(args):
    video_base = VideoBase(project_dir=args.project_dir, config_path=args.config_path, graphics=args.graphics,
                           record=args.record)
    video_base.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, project_dir, 'path to the project directory', shortname='-p')
    add_arg('config_path', str, config_path, 'path to the configuration file', shortname='-c')
    add_arg('graphics', bool, True, 'whether shows graphics', shortname='-g')
    add_arg('record', bool, False, 'whether records video frames', shortname='-r')

    input_args = parser.parse_args()
    print_arguments(input_args)
    run_video_base(input_args)
