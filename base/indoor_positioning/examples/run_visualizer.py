"""This script runs the video visualizer."""
import argparse
import functools
import os
import sys

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = 'conf/video_base.ini'
sys.path.append(project_dir)

from openmmla_vision.utils.args_utils import add_arguments, print_arguments
from openmmla_vision.bases.visualizer import Visualizer


def run_visualizer(args):
    visualizer = Visualizer(project_dir=args.project_dir, config_path=args.config_path, store=args.store)
    visualizer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, project_dir, 'path to the project directory', shortname='-p')
    add_arg('config_path', str, config_path, 'path to the configuration file', shortname='-c')
    add_arg("store", bool, False, "whether to store the visualization images in the local directory.", shortname='-s')

    input_args = parser.parse_args()
    print_arguments(input_args)
    run_visualizer(input_args)
