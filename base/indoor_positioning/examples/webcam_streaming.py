"""This script demonstrates how to stream video from a webcam."""
import logging
import os
import sys

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(project_dir)

import cv2

from openmmla_vision.utils.stream_utils import WebcamVideoStream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

src = int(input("Please input the camera source: "))
webcam = WebcamVideoStream(save_path='', format='MJPG', res=(1920, 1080), src=src).start()
try:
    while True:
        frame = webcam.read()
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Display", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    logger.info("You pressed Ctrl+C! Stopping all threads...")
finally:
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    webcam.stop()

# ffmpeg -framerate 30 -i ./output/frame_%07d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
