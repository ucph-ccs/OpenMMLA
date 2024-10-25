"""This script demonstrates how to detect apriltags in an image."""
import os
import sys

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(project_dir)

from openmmla_vision.utils.apriltag_utils import detect_apriltags

image_path = os.path.join(project_dir, 'data/llm_test/1.jpg')
detect_apriltags(image_path, render=True, show=True, save=False)
