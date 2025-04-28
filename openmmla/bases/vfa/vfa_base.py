import datetime
import json
import logging
import os
import threading
import time

import cv2
import numpy as np

from openmmla.bases.base import Base
from openmmla.streams.video_stream import VideoStream
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import get_bucket_name, get_id, flush_input
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import resolve_url
from .input import get_function_base, get_mode


class VFABase(Base):
    """VFABase class for video frame analysis."""
    logger = get_logger('vfa-base')

    def __init__(self, project_dir: str | None, config_path: str, mode: str = 'full', graphics: bool = True,
                 verbose: bool = False):
        """Initializes the VFABase class.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            mode: operating mode, 'record', 'analyze', or 'full'. (default: 'full')
            graphics: whether to display graphics (default: True)
            verbose: whether to enable verbose logging (default: False)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        # VFABase specific parameters
        self.mode = mode
        self.graphics = graphics
        self.verbose = verbose

        # Runtime attributes
        self.chosen_camera = None
        self.selected_source = None
        self.base_id = None
        self.camera_configured = False
        self.bucket_name = None
        self.video_stream = None

        # Threading attributes
        self.stop_event = threading.Event()
        self.threads = []

        self._setup_yaml()
        self._setup_directories()
        self._setup_clients()

    def _setup_yaml(self):
        """Load and assign configuration parameters from the YAML configuration file."""
        base_config = self.config['Base']
        vfa_server_config = self.config['Server']['vfa']

        # Load angle configurations
        self.angle_config = base_config.get('angle_config', {})
        self.camera_angle = None  # Will be set during camera configuration

        self.res = tuple(base_config.get('resolution', (1920, 1080)))
        self.rotate = int(base_config.get('rotate', 0))
        self.fps = int(base_config.get('fps', 30))
        self.interval = int(base_config.get('interval', 30))

        self.source = base_config['source']
        self.stream_kwargs = base_config['stream_kwargs']

        self.vllm_frame_analyzer_url = resolve_url(vfa_server_config['vllm_frame_analyzer'])

    def _setup_directories(self):
        """Create and set up the necessary directories for runtime operations."""
        self.logger_dir = os.path.join(self.project_dir, 'logger')
        self.runtime_dir = os.path.join(self.project_dir, 'real-time', 'runtime')
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.runtime_dir, exist_ok=True)

    def _setup_clients(self):
        """Initialize external service clients and internal processing objects."""
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)

    def run(self):
        """Run the VFA base."""
        print('\033]0;VFA Base\007')

        func_map = {1: self._start, 2: self._set_camera, 3: self._switch_mode}
        while True:
            try:
                select_fun = get_function_base(self.chosen_camera, self.selected_source, self.camera_angle,
                                               self.base_id, self.mode)
                if select_fun == 0:
                    self.logger.info("Exiting VFA base...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"During running the VFA base, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)

    def _start(self):
        if not self.camera_configured:
            self.logger.warning("Camera is not configured.")
            return self._set_camera()

        self.bucket_name = get_bucket_name(self.influx_client)
        self.logger = get_logger(f'vfa-{self.bucket_name}',
                                 os.path.join(self.project_dir, f'logger/{self.bucket_name}_vfa_{self.base_id}.log'),
                                 console_level=logging.DEBUG if self.verbose else logging.INFO, mode='a')

        self._listen_for_start_signal()

        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()
        self.video_stream = VideoStream(source=self.source, **self.stream_kwargs)
        self.video_stream.start()

        try:
            self._create_thread(self._listen_for_stop_signal)
            self._start_threads()
            self._process_frames()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning("Interrupted: %s", e)
        finally:
            self._clean_up()

    def _switch_mode(self):
        """Switch the operating mode between 'record', 'analyze' and 'full'."""
        self.mode = get_mode()
        self.logger.info(f"Switched to {self.mode} mode.")

    def _set_camera(self):
        """Set up video source, base id, and camera angle."""
        self.camera_configured = False

        self.camera_info = self._configure_camera_params()
        if self.camera_info is None:
            self.logger.warning("No camera parameters found, please calibrate camera first.")
            return

        self.camera_angle = self._set_camera_angle()
        available_sources = self._detect_video_sources()
        self.selected_source = self._choose_video_source(available_sources)

        if self.selected_source is None:
            self.logger.warning("No available video source found.")
            return

        # Configure stream based on source type
        if self.source == 'opencv':
            self.stream_kwargs['camera_index'] = self.selected_source
        elif self.source == 'rtmp':
            self.stream_kwargs['rtmp_url'] = self.selected_source

        self.base_id = get_id()
        self.camera_configured = True
        print(f'\033]0;VFA Base {self.base_id}, Camera {self.selected_source}\007')

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

    def _set_camera_angle(self):
        """Set the camera angle."""
        # Configure camera angle
        if self.angle_config:
            angles = list(self.angle_config.keys())
            print("Available camera angles:")
            for idx, angle in enumerate(angles):
                print(f"{idx}: {angle} - {self.angle_config[angle]}")
            print(f"{len(angles)}: None - No specific angle")

            while True:
                try:
                    flush_input()
                    angle_input = input("Choose camera angle (press Enter for None): ")
                    if angle_input == '':
                        return None

                    angle_idx = int(angle_input)
                    if 0 <= angle_idx < len(angles):
                        return angles[angle_idx]
                    elif angle_idx == len(angles):
                        return None
                    else:
                        print("Invalid selection. Please choose a valid option.")
                except ValueError:
                    print("Please enter a valid number or press Enter for None.")
        else:
            return None

    def _detect_video_sources(self):
        """Detect available video sources based on the source type."""
        available_sources = []
        number_of_detected_sources = 0

        if self.source == 'opencv':
            # Detect available camera indices
            for i in range(4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"{number_of_detected_sources} : Camera index {i} is available.")
                    number_of_detected_sources += 1
                    available_sources.append(i)
                cap.release()

        elif self.source == 'rtmp':
            # Get RTMP URLs from configuration
            rtmp_config = self.config.get('RTMP', {})
            if 'video_streams' in rtmp_config:
                video_stream_list = rtmp_config['video_streams'].split(',')
                for url in video_stream_list:
                    if url:
                        print(f"{number_of_detected_sources} : RTMP stream {url} is available.")
                        available_sources.append(url)
                        number_of_detected_sources += 1
            else:
                self.logger.warning("No RTMP video streams found in the configuration, please set it in "
                                    "yaml config file ['RTMP']['video_streams'].")

        return available_sources

    def _choose_video_source(self, available_sources):
        """Choose a video source (camera index or RTMP URL)."""
        if not available_sources:
            return None
        default_source_idx = 0  # Default to the first available source
        while True:
            try:
                flush_input()
                source_idx_input = input(f"Choose your video source index [{default_source_idx}]: ")
                if source_idx_input == '':
                    source_idx = default_source_idx
                else:
                    source_idx = int(source_idx_input)
                if 0 <= source_idx < len(available_sources):
                    selected_source = available_sources[source_idx]
                    self.logger.info(f"Selected video source: {selected_source}")
                    return selected_source
                else:
                    self.logger.warning("Invalid selection. Please choose a valid source index.")
            except ValueError:
                self.logger.warning("Please enter a valid number or press Enter for default.")

    def _clean_up(self):
        self.stop_event.set()
        self.video_stream.stop()
        self.mqtt_client.loop_stop()
        if self.graphics:
            cv2.destroyWindow(f'VFA Base {self.base_id}, Camera {self.selected_source}')
            cv2.waitKey(1)
        self.threads.clear()

    def _process_frames(self):
        print("Processing VFA frames...")
        save_path = os.path.join(self.runtime_dir, f'{self.bucket_name}/{self.chosen_camera}_{self.base_id}')
        os.makedirs(save_path, exist_ok=True)

        if self.mode == 'analyze':
            # Process existing images in analyze mode
            self._analyze_existing_frames(save_path)
            return

        # For record and full modes, process live frames
        last_saved_time = time.time() - self.interval + 1

        while not self.stop_event.is_set():
            video_frame = self.video_stream.read()[-1]
            frame = video_frame.data
            current_time = time.time()

            if self.graphics:
                display_frame = cv2.resize(frame, (960, 540))
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(display_frame, timestamp, (display_frame.shape[1] - 300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(f'VFA Base {self.base_id}, Camera {self.selected_source}', display_frame)

            if current_time - last_saved_time >= self.interval:
                acquired_time = video_frame.timestamp
                image_path = os.path.join(save_path, f'{acquired_time}.jpg')
                cv2.imwrite(image_path, frame)
                last_saved_time = current_time

                if self.mode == 'full':
                    try:
                        self._publish_frame(image_path, acquired_time)
                    except Exception as e:
                        self.logger.warning(f"VFA publish failed: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _analyze_existing_frames(self, save_path):
        """Analyze existing frames in the save path."""
        if not os.path.exists(save_path):
            self.logger.warning(f"No frames found in {save_path}")
            return

        # Get all image files and sort by timestamp
        frame_files = [f for f in os.listdir(save_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        frame_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort by timestamp

        for frame_file in frame_files:
            acquired_time = float(frame_file.split('.')[0])
            if self.stop_event.is_set():
                break

            image_path = os.path.join(save_path, frame_file)
            try:
                self._publish_frame(image_path, acquired_time)
            except Exception as e:
                self.logger.warning(f"VFA publish failed for {frame_file}: {e}")

    def _publish_frame(self, image_path, acquired_time):
        """Publish frame to MQTT for synchronization."""
        # For MQTT, we'll publish just the path and metadata
        # The synchronizer will load the actual image
        frame_data = {
            "base_id": str(self.base_id),
            "angle": self.camera_angle,
            "image_path": image_path,
            "acquired_time": acquired_time
        }

        self.mqtt_client.publish(f"{self.bucket_name}/vfa", json.dumps(frame_data))
        self.logger.info(f"Published frame with angle {self.camera_angle} at {acquired_time}")
