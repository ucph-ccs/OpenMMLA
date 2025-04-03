"""This script demonstrates how to detect apriltags in an image."""
import os

from pupil_apriltags import Detector

from openmmla.utils.video.apriltag import detect_apriltags

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
image_path = os.path.join(project_dir, 'data/img.png')
detector = Detector(families='tag36h11', nthreads=4)
detect_apriltags(image_path, detector, render=False, show=True, save=False)
