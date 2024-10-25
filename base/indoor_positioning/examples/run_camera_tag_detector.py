"""This script runs the camera tag detector."""
import argparse
import functools
import os
import sys

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = 'conf/video_base.ini'
sys.path.append(project_dir)

from openmmla_vision.camera.camera_tag_detector import CameraTagDetector
from openmmla_vision.utils.args_utils import add_arguments, print_arguments


def run_camera_tag_detector(args):
    camera_tag_detector = CameraTagDetector(project_dir=args.project_dir, config_path=args.config_path)
    camera_tag_detector.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("project_dir", str, project_dir, "path to the project directory", shortname="-p")
    add_arg("config_path", str, config_path, "path to the configuration file", shortname="-c")

    input_args = parser.parse_args()
    print_arguments(input_args)
    run_camera_tag_detector(input_args)
