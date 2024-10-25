import json
import os
import time

import cv2
import numpy as np
from pupil_apriltags import Detector

from openmmla.bases import Base
from openmmla.utils.client import MQTTClientWrapper
from openmmla.utils.logger import get_logger
from .input import get_function_base
from .stream import WebcamVideoStream


class CameraTagDetector(Base):
    """Camera detector class for detecting AprilTags from camera feed"""
    logger = get_logger('camera-tag-detector')

    def __init__(self, project_dir: str, config_path: str, max_badge_id: int = 15):
        """Initialize the camera tag detector.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            max_badge_id: maximum badge ID to detect
        """
        super().__init__(project_dir, config_path)

        """Camera detector parameters."""
        self.max_badge_id = max_badge_id
        self.cameras_dir = os.path.join(self.project_dir, 'camera_calib/cameras')

        """Runtime attributes"""
        self.camera_info = {}
        self.camera_seed = None
        self.sender_id = None
        self.video_stream = None
        self.camera_configured = False

        """Client attributes."""
        self.detector = Detector(families=self.families, nthreads=4)
        self.mqtt_client = MQTTClientWrapper(self.config_path)

        self._setup_from_yaml()

    def _setup_from_yaml(self):
        """Set up attributes from YAML configuration."""
        tag_config = self.config.get('Tag', {})
        image_config = self.config.get('Image', {})

        self.tag_size = float(tag_config.get('tag_size', 0.08))
        self.families = tag_config.get('families', 'tag36h11')
        self.res_width = int(image_config.get('res_width', 1920))
        self.res_height = int(image_config.get('res_height', 1080))
        self.res = (self.res_width, self.res_height)

    def _setup_from_input(self):
        """Set up camera configuration from user input."""
        self.camera_info = self._configure_camera_params()
        if self.camera_info is None:
            self.logger.warning("Camera configuration failed, please calibrate your camera first.")
            return False

        available_seeds = self._detect_video_seeds()
        self.camera_seed = self.choose_camera_seed(available_seeds)
        if self.camera_seed is None:
            self.logger.warning("No available camera seed found.")
            return False

        self.sender_id = input("Input your sender id, 'm' for main camera, and 'a', 'b', 'c', 'd' for alternatives "
                               "camera: ")
        return True

    def run(self):
        """Run the camera tag detector."""
        print('\033]0;Camera Detector\007')
        func_map = {1: self._start, 2: self._set_camera}

        while True:
            try:
                select_fun = get_function_base()
                if select_fun == 0:
                    self.logger.info("Exiting video base...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning("%s, Come back to the main menu.", e, exc_info=True)

    def _start(self):
        """Start AprilTag detection"""
        if not self.camera_configured:
            self.logger.warning("Camera is not configured.")
            return self._set_camera()

        # MQTT client reinitialization
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        # Start video stream
        self._configure_video_capture(self.camera_seed)

        try:
            self._process_frames()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning("%s, capture interrupted.", e, exc_info=False)
        finally:
            self.video_stream.stop()
            self.mqtt_client.loop_stop()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    def _set_camera(self):
        """Configure camera settings."""
        self.camera_configured = False
        if self._setup_from_input():
            self.camera_configured = True
            print(f'\033]0;Camera Detector {self.sender_id}\007')

    def _configure_camera_params(self):
        """Configure camera parameters from the selected camera."""
        camera_choices = [d for d in os.listdir(self.cameras_dir) if os.path.isdir(os.path.join(self.cameras_dir, d))]
        for idx, choice in enumerate(camera_choices):
            print(f"{idx}: {choice}")

        if not camera_choices:
            return None

        while True:
            try:
                selection = int(input("Choose your camera name with number: "))
                if not 0 <= selection < len(camera_choices):
                    self.logger.warning("Invalid selection. Please choose a valid number.")
                else:
                    chosen_camera = camera_choices[selection]
                    break
            except ValueError:
                self.logger.warning("Please enter a valid number.")

        camera_config = self.config.get(chosen_camera, {})
        fisheye = camera_config.get('fisheye', False)
        params = camera_config.get('params', [])
        camera_info = {"fisheye": fisheye, "params": params, "res": self.res}

        if fisheye:
            K = np.array(camera_config.get('K', []))
            D = np.array(camera_config.get('D', []))
            map_1, map_2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, self.res, cv2.CV_16SC2)
            camera_info.update({"K": K, "D": D, "map_1": map_1, "map_2": map_2})

        return camera_info

    def _configure_video_capture(self, camera_seed):
        self.video_stream = WebcamVideoStream(format='MJPG', src=camera_seed, res=(1920, 1080))
        self.video_stream.start()

    def _process_frames(self):
        while True:
            frame = self.video_stream.read()
            if self.camera_info.get("fisheye", False):
                frame = cv2.remap(frame, self.camera_info["map_1"], self.camera_info["map_2"],
                                  interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_info["params"],
                                           tag_size=self.tag_size)

            tags = {}

            for tag in results:
                if int(tag.tag_id) > self.max_badge_id:
                    continue

                corners = np.int32(tag.corners)
                normal, tag = self.get_outward_normal(tag)
                tags[tag.tag_id] = [list(tag.pose_R.tolist()), list(tag.pose_t.tolist())]

                # Drawing annotations
                tag_center = np.mean(corners, axis=0)
                arrow_dir = normal[:2]
                scale_factor = 50
                end_point = tag_center + scale_factor * arrow_dir
                cv2.arrowedLine(frame, tuple(np.int32(tag_center)), tuple(np.int32(end_point)), (0, 0, 255), 2)
                cv2.polylines(frame, [corners], True, (0, 255, 0), thickness=2)
                cv2.putText(frame, str(tag.tag_id), org=(int(tag_center[0]) + 10, int(tag_center[1]) + 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"Rot: {tag.pose_R}",
                            (tag.corners[0][0].astype(int), tag.corners[0][1].astype(int) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Trans: {tag.pose_t}",
                            (tag.corners[0][0].astype(int), tag.corners[0][1].astype(int) - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow(f'AprilTags Detection from camera {self.sender_id}', display_frame)

            message = {
                "sender_id": self.sender_id,
                "tags": tags,
                "timestamp": time.time(),
            }
            message_str = json.dumps(message)
            self.mqtt_client.publish("camera/synchronize", message_str, qos=0, retain=False)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    @staticmethod
    def get_outward_normal(tag):
        """Get the normal vector of the tag which perpendicular to the surface"""
        normal = np.dot(tag.pose_R, np.array([0, 0, 1])).ravel()
        if np.dot(normal, tag.pose_t.ravel()) < 0:
            tag.pose_R[:, 2] = -tag.pose_R[:, 2]  # Flip the sign of the third column of the rotation matrix
        else:
            normal = -normal
        return normal, tag

    def _detect_video_seeds(self):
        available_video_seeds = []
        number_of_detected_seeds = 0
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"{number_of_detected_seeds} : Camera seed {i} is available.")
                number_of_detected_seeds += 1
                available_video_seeds.append(i)
            cap.release()

        if 'RTMP' in self.config and 'video_streams' in self.config['RTMP']:
            video_stream_list = self.config['RTMP']['video_streams'].split(',')
            for streams in video_stream_list:
                if streams:
                    print(f"{number_of_detected_seeds} : RTMP stream {streams} is available.")
                    available_video_seeds.append(streams)
                    number_of_detected_seeds += 1

        return available_video_seeds

    @staticmethod
    def choose_camera_seed(available_video_seeds):
        if not available_video_seeds:
            return None
        while True:
            try:
                seed_id = int(input("Choose your video seed id: "))
                if 0 <= seed_id < len(available_video_seeds):
                    print(f"Selected video seed: {available_video_seeds[seed_id]}")
                    return available_video_seeds[seed_id]
                else:
                    print("Invalid selection. Please choose a valid video seed.")
            except ValueError:
                print("Please enter a valid number.")
