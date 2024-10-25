import datetime
import json
import os

import cv2
import numpy as np
from pupil_apriltags import Detector

from openmmla.bases import Base
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper
from openmmla.utils.logger import get_logger
from .enums import ROTATIONS
from .input import get_bucket_name, get_function_base
from .stream import WebcamVideoStream
from .vector import is_tag_looking_at_another_2d, get_2d_outward_normal_vector


class VideoBase(Base):
    """Video base class for AprilTag detection from video stream."""
    logger = get_logger('video-base')

    def __init__(self, project_dir: str = None, config_path: str = None, graphics: bool = True, record: bool = False):
        """Initialize the video base.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            graphics: whether to show graphics
            record: whether to record video frames
        """
        super().__init__(project_dir, config_path)

        """Video-base specific parameters."""
        self.graphics = graphics
        self.record = record
        self.max_badge_id = 12

        """Runtime attributes."""
        self.camera_info = {}
        self.camera_seed = None
        self.sender_id = None
        self.bucket_name = None
        self.video_stream = None
        self.transformation_id = None
        self.transform_matrices_dict = None
        self.camera_configured = False

        """Client attributes."""
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.detector = Detector(families=self.families, nthreads=4)

        self._setup_from_yaml()
        self._setup_directories()

    def _setup_from_yaml(self):
        """Set up attributes from YAML configuration."""
        tag_config = self.config.get('Tag', {})
        self.tag_size = float(tag_config.get('tag_size', 0.0))
        self.families = tag_config.get('families', '')

        image_config = self.config.get('Image', {})
        self.res_width = int(image_config.get('res_width', 0))
        self.res_height = int(image_config.get('res_height', 0))
        self.res = (self.res_width, self.res_height)
        self.rotate = int(image_config.get('rotate', 0))

    def _setup_directories(self):
        """Set up directories."""
        self.recordings_dir = os.path.join(self.project_dir, 'recordings')
        self.camera_sync_dir = os.path.join(self.project_dir, 'camera_sync')
        self.camera_calib_dir = os.path.join(self.project_dir, 'camera_calib')
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.camera_sync_dir, exist_ok=True)
        os.makedirs(self.camera_calib_dir, exist_ok=True)

    def run(self):
        print('\033]0;Video Base\007')
        func_map = {1: self._start, 2: self._set_camera}

        while True:
            try:
                select_fun = get_function_base()
                if select_fun == 0:
                    self.logger.info("Exiting video base...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"During running the video base, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)

    def _start(self):
        """Start AprilTag detection"""
        if not self.camera_configured:
            self.logger.warning("Camera is not configured.")
            return self._set_camera()

        # Bucket selection
        self.bucket_name = get_bucket_name(self.influx_client)

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
            if self.graphics:
                cv2.destroyWindow(f'AprilTags Detection from camera {self.sender_id}')
                cv2.waitKey(1)
            # time.sleep(5) # Wait for the threads to terminate

    def _set_camera(self):
        self.camera_configured = False

        self.transform_matrices_dict = self._load_transform_matrices()
        if self.transform_matrices_dict is None:
            self.logger.warning("No transformation matrices found,  please do the camera sync first.")
            return

        self.camera_info = self._configure_camera_params()
        if self.camera_info is None:
            self.logger.warning("Camera configuration failed.")
            return

        available_seeds = self._detect_video_seeds()
        self.camera_seed = self._choose_camera_seed(available_seeds)
        if self.camera_seed is None:
            self.logger.warning("No available camera seed found.")
            return

        self.sender_id = self._choose_sender_id()
        self.camera_configured = True

    def _configure_video_capture(self, camera_seed):
        if self.record:
            save_path = os.path.join(self.recordings_dir, f'{self.bucket_name}/{self.sender_id}')
        else:
            save_path = ''
        self.video_stream = WebcamVideoStream(save_path=save_path, format='MJPG', src=camera_seed, res=(1920, 1080))
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
            tag_relations = {}

            for tag in results:
                if int(tag.tag_id) > self.max_badge_id:
                    continue

                corners = np.int32(tag.corners)
                normal, tag = get_2d_outward_normal_vector(tag)
                tags[tag.tag_id] = [list(tag.pose_R.tolist()), list(tag.pose_t.tolist())]

                tag_relations.setdefault(tag.tag_id, [])
                for other_tag in results:
                    if other_tag != tag and other_tag.tag_id not in tag_relations[tag.tag_id]:
                        if is_tag_looking_at_another_2d(tag, other_tag, cosine_threshold=-0.94, distance_threshold=1):
                            tag_relations[tag.tag_id].append(str(other_tag.tag_id))

                # Drawing annotations
                if self.graphics:
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

            if self.graphics:
                display_frame = cv2.resize(frame, (960, 540))
                if self.rotate in ROTATIONS:
                    display_frame = cv2.rotate(display_frame, ROTATIONS[self.rotate])
                # Display timestamp on frame
                now = datetime.datetime.now()
                current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(display_frame, current_time_str, (display_frame.shape[1] - 300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.imshow(f'AprilTags Detection from camera {self.sender_id}', display_frame)

            message = {
                "sender_id": self.sender_id,
                "tags": tags,
                "tag_relations": tag_relations,
            }
            message_str = json.dumps(message)
            self.mqtt_client.publish(f'{self.bucket_name}/video', message_str, qos=0, retain=False)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _load_transform_matrices(self):
        transformation_choices = [d for d in os.listdir(self.camera_sync_dir) if
                                  d.startswith('transformation_matrices_')]
        for idx, choice in enumerate(transformation_choices):
            print(f"{idx}: {choice}")

        if not transformation_choices:
            return None

        while True:
            try:
                selection = int(input("Choose your main transformation matrices with number: "))
                if not 0 <= selection < len(transformation_choices):
                    self.logger.warning("Invalid selection. Please choose a valid number.")
                else:
                    chosen_transformation = transformation_choices[selection]
                    self.transformation_id = chosen_transformation.split('_')[-1].split('.')[0]
                    break
            except ValueError:
                self.logger.warning("Please enter a valid number.")

        with open(os.path.join(self.camera_sync_dir, chosen_transformation), 'r') as file:
            return json.load(file)

    def _configure_camera_params(self):
        cameras_dir = os.path.join(self.camera_calib_dir, 'cameras')
        camera_choices = [d for d in os.listdir(cameras_dir) if os.path.isdir(os.path.join(cameras_dir, d))]
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

        fisheye = self.config.getboolean(chosen_camera, 'fisheye')
        params = json.loads(self.config.get(chosen_camera, 'params'))
        camera_info = {"fisheye": fisheye, "params": params, "res": self.res}

        if fisheye:
            K = np.array(json.loads(self.config.get(chosen_camera, 'K')))
            D = np.array(json.loads(self.config.get(chosen_camera, 'D')))
            map_1, map_2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, self.res, cv2.CV_16SC2)
            camera_info.update({"K": K, "D": D, "map_1": map_1, "map_2": map_2})

        return camera_info

    def _choose_camera_seed(self, available_video_seeds):
        if not available_video_seeds:
            return None
        while True:
            try:
                seed_id = int(input("Choose your video seed id: "))
                if 0 <= seed_id < len(available_video_seeds):
                    return available_video_seeds[seed_id]
                else:
                    self.logger.warning("Invalid selection. Please choose a valid video seed.")
            except ValueError:
                self.logger.warning("Please enter a valid number.")

    def _choose_sender_id(self):
        """Prompt user to choose a sender ID based on available keys in transform_matrices.json configuration."""
        print(f"Available sender ids:\n- {self.transformation_id}")
        for key in self.transform_matrices_dict.keys():
            print(f"- {key}")
        while True:
            sender_id = input("Enter your sender id: ")
            if sender_id in self.transform_matrices_dict or sender_id == self.transformation_id:
                return sender_id
            else:
                print("Invalid selection. Please enter a valid sender id.")

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

        rtmp_config = self.config.get('RTMP', {})
        if 'video_streams' in rtmp_config:
            video_stream_list = rtmp_config['video_streams'].split(',')
            for streams in video_stream_list:
                if streams:
                    print(f"{number_of_detected_seeds} : RTMP stream {streams} is available.")
                    available_video_seeds.append(streams)
                    number_of_detected_seeds += 1

        return available_video_seeds
