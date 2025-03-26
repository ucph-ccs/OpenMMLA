import datetime
import json
import os
import threading

import cv2
import numpy as np
from pupil_apriltags import Detector

from openmmla.bases.base import Base
from openmmla.streams.video_stream import VideoStream
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.logger import get_logger
from .enums import ROTATIONS
from .input import get_bucket_name, get_function_base
from .vector import is_tag_looking_at_another_2d, get_2d_outward_normal_vector


class VideoBase(Base):
    """Video base class for AprilTag detection from video stream."""
    logger = get_logger('video-base')

    def __init__(self, project_dir: str, config_path: str, graphics: bool = True, store: bool = False):
        """Initialize the video base.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            graphics: whether to show graphics, default to True
            store: whether to store video frames, default to False
        """
        super().__init__(project_dir, config_path)

        """Video-base specific parameters."""
        self.graphics = graphics
        self.store = store
        self.max_badge_id = 12

        """Runtime attributes."""
        self.chosen_camera = None
        self.camera_info = {}
        self.camera_seed = None
        self.base_id = None  # the base camera id
        self.main_id = None  # the main camera id
        self.transform_matrices_dict = None
        self.camera_configured = False
        self.bucket_name = None
        self.video_stream = None

        """Threading attributes."""
        self.stop_event = threading.Event()
        self.threads = []

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        """Set up attributes from YAML configuration."""
        tag_config = self.config.get('AprilTag', {})
        self.tag_size = float(tag_config.get('tag_size', 0.061))
        self.families = tag_config.get('families', 'tag36h11')

        image_config = self.config.get('Image', {})
        self.res_width = int(image_config.get('res_width', 1920))
        self.res_height = int(image_config.get('res_height', 1080))
        self.res = (self.res_width, self.res_height)
        self.rotate = int(image_config.get('rotate', 0))
        self.fps = int(image_config.get('fps', 30))

    def _setup_directories(self):
        """Set up directories."""
        self.runtime_dir = os.path.join(self.project_dir, 'real-time/runtime')
        self.camera_sync_dir = os.path.join(self.project_dir, 'camera_sync')
        self.camera_calib_dir = os.path.join(self.project_dir, 'camera_calib')
        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.camera_sync_dir, exist_ok=True)
        os.makedirs(self.camera_calib_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.detector = Detector(families=self.families, nthreads=4)

    def run(self):
        print('\033]0;Video Base\007')
        func_map = {1: self._set_camera, 2: self._start}

        while True:
            try:
                select_fun = get_function_base(self.chosen_camera, self.camera_seed, self.base_id, self.main_id)
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
        self._listen_for_start_signal()

        # MQTT client reinitialization
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        # Start video stream
        self._configure_video_stream(self.camera_seed)

        exception_occurred = None
        try:
            self._create_thread(self._listen_for_stop_signal)
            self._start_threads()
            self._process_frames()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning("%s, capture interrupted.", e, exc_info=False)
            exception_occurred = e
        finally:
            self._detection_handler(exception_occurred)

    def _detection_handler(self, e):
        """Handle exceptions and stop all threads.

        Args:
            e: the exception occurred during the detection process
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")
        self._clean_up()

    def _clean_up(self):
        """Clean up resources."""
        self.stop_event.set()
        self.video_stream.stop()
        self.mqtt_client.loop_stop()
        if self.graphics:
            cv2.destroyWindow(f'AprilTags Detection from camera {self.base_id}')
            cv2.waitKey(1)
        self.threads.clear()

    def _set_camera(self):
        """Set up camera seed and id."""
        self.camera_configured = False

        self.transform_matrices_dict = self._load_transform_matrices()
        if self.transform_matrices_dict is None:
            self.logger.warning("No transformation matrices found, please do the camera sync first.")
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

        self.base_id = self._choose_base_id()
        self.camera_configured = True
        print(f'\033]0;Video Base {self.base_id}\007')

    def _configure_camera_params(self):
        """Configure camera intrinsic parameters."""
        cameras = self.config.get('Cameras', {})
        camera_choices = sorted(list(cameras.keys()))
        if not camera_choices:
            return None
        for idx, choice in enumerate(camera_choices):
            print(f"{idx}: {choice}")

        default_selection = 0  # Default to the first camera
        while True:
            try:
                selection_input = input(f"Choose your camera name with number [{default_selection}]: ")
                if selection_input == '':
                    selection = default_selection
                else:
                    selection = int(selection_input)
                if not 0 <= selection < len(camera_choices):
                    self.logger.warning("Invalid selection. Please choose a valid number.")
                else:
                    self.chosen_camera = camera_choices[selection]
                    break
            except ValueError:
                self.logger.warning("Please enter a valid number or press Enter for default.")

        camera_config = cameras[self.chosen_camera]
        fisheye = camera_config['fisheye']
        params = camera_config['params']
        camera_info = {"fisheye": fisheye, "params": params, "res": self.res}

        if fisheye:
            K = np.array(camera_config['K'])
            D = np.array(camera_config['D'])
            map_1, map_2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, self.res, cv2.CV_16SC2)
            camera_info.update({"K": K, "D": D, "map_1": map_1, "map_2": map_2})

        print(camera_info)
        return camera_info

    def _detect_video_seeds(self):
        """Detect available video seeds."""
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

    def _choose_camera_seed(self, available_video_seeds):
        """Choose a camera seed."""
        if not available_video_seeds:
            return None
        default_seed_id = 0  # Default to the first available seed
        while True:
            try:
                seed_id_input = input(f"Choose your video seed id [{default_seed_id}]: ")
                if seed_id_input == '':
                    seed_id = default_seed_id
                else:
                    seed_id = int(seed_id_input)
                if 0 <= seed_id < len(available_video_seeds):
                    self.logger.info(f"Selected video seed: {available_video_seeds[seed_id]}")
                    return available_video_seeds[seed_id]
                else:
                    self.logger.warning("Invalid selection. Please choose a valid video seed.")
            except ValueError:
                self.logger.warning("Please enter a valid number or press Enter for default.")

    def _choose_base_id(self):
        """Prompt user to choose a sender ID based on available keys in transform_matrices.json configuration."""
        print(f"Available sender ids:\n- {self.main_id}")
        for key in self.transform_matrices_dict.keys():
            print(f"- {key}")
        default_base_id = self.main_id  # Default sender id
        while True:
            base_id_input = input(f"Enter your sender id [{default_base_id}]: ")
            base_id = base_id_input if base_id_input else default_base_id
            if base_id in self.transform_matrices_dict or base_id == self.main_id:
                return base_id
            else:
                print("Invalid selection. Please enter a valid sender id or press Enter for default.")

    def _configure_video_stream(self, camera_seed):
        """Configure video stream."""
        self.video_stream = VideoStream(source=camera_seed, buffer_duration=0.08, format='MJPG', resolution=self.res,
                                        fps=self.fps)
        self.video_stream.start()

    def _process_frames(self):
        """Process video frames and detect AprilTags."""
        print("Processing frames...")

        save_path = os.path.join(self.runtime_dir, f'{self.bucket_name}/{self.base_id}')
        frames_count = 0
        os.makedirs(save_path, exist_ok=True)

        while not self.stop_event.is_set():
            video_frame = self.video_stream.read()[-1]
            frame = video_frame.data
            frames_count += 1
            acquired_time = video_frame.timestamp  # Capture the frame timestamp

            # Frame preprocessing.
            if self.camera_info.get("fisheye", False):
                frame = cv2.remap(frame, self.camera_info["map_1"], self.camera_info["map_2"],
                                  interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if self.rotate in ROTATIONS:
                frame = cv2.rotate(frame, ROTATIONS[self.rotate])

            # AprilTag detection.
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

                # Detect tag relations.
                tag_relations.setdefault(tag.tag_id, [])

                other_tags = [t for t in results if t.tag_id != tag.tag_id]
                for other_tag in other_tags:
                    if other_tag.tag_id not in tag_relations[tag.tag_id]:
                        if is_tag_looking_at_another_2d(tag, other_tag, cosine_threshold=-0.94, distance_threshold=1):
                            tag_relations[tag.tag_id].append(str(other_tag.tag_id))

                if self.graphics:  # drawing annotations.
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

            if self.graphics:  # display frame.
                display_frame = cv2.resize(frame, (960, 540))
                now = datetime.datetime.now()
                current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(display_frame, current_time_str, (display_frame.shape[1] - 300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.imshow(f'AprilTags Detection from camera {self.base_id}', display_frame)

            if self.store and frames_count % self.fps == 0:
                frames_count = 0
                cv2.imwrite(os.path.join(save_path, f'{acquired_time}.jpg'), frame)

            # Publish message with the acquired timestamp.
            message = {
                "base_id": self.base_id,
                "tags": tags,
                "tag_relations": tag_relations,
                "acquired_time": acquired_time
            }
            message_str = json.dumps(message)
            self.mqtt_client.publish(f'{self.bucket_name}/video', message_str, qos=0, retain=False)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _load_transform_matrices(self):
        """Load transformation matrices."""
        transformation_choices = [d for d in os.listdir(self.camera_sync_dir) if
                                  d.startswith('transformation_matrices_')]
        for idx, choice in enumerate(transformation_choices):
            print(f"{idx}: {choice}")

        if not transformation_choices:
            return None

        default_selection = 0  # Default to the first transformation matrix
        while True:
            try:
                selection_input = input(f"Choose your main transformation matrices with number [{default_selection}]: ")
                if selection_input == '':
                    selection = default_selection
                else:
                    selection = int(selection_input)
                if not 0 <= selection < len(transformation_choices):
                    self.logger.warning("Invalid selection. Please choose a valid number.")
                else:
                    chosen_transformation = transformation_choices[selection]
                    self.main_id = chosen_transformation.split('_')[-1].split('.')[0]
                    break
            except ValueError:
                self.logger.warning("Please enter a valid number or press Enter for default.")

        with open(os.path.join(self.camera_sync_dir, chosen_transformation), 'r') as file:
            return json.load(file)
