"""This script demonstrates how to detect apriltags in an image."""
import os

from openmmla_vision.utils.apriltag_utils import detect_apriltags
from pupil_apriltags import Detector

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
image_path = os.path.join(project_dir, 'data/llm_test/1.jpg')
detector = Detector(families='tag36h11', nthreads=4)
detect_apriltags(image_path, detector, render=True, show=True, save=False)
