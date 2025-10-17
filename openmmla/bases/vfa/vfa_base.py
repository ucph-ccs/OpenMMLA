import datetime
import gc
import json
import logging
import os
import re
import threading
import time

import cv2
import numpy as np

from openmmla.bases.base import Base
from openmmla.streams.video_stream import VideoStream
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_bucket, get_id, flush_input
from openmmla.utils.logger import get_logger
from openmmla.utils.validation import validate_unix_timestamp
from .enums import ROTATIONS
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

        # Load angle configurations
        self.angle_config = base_config.get('angle_config', {})
        self.camera_angle = None  # Will be set during camera configuration

        self.res = tuple(base_config.get('resolution', (1920, 1080)))
        self.rotate = int(base_config.get('rotate', 0))
        self.fps = int(base_config.get('fps', 30))
        
        # frame processing configuration (unified for all sources)
        self.keyframe_interval = float(base_config.get('keyframe_interval', 30.0))
        self.processing_rate = float(base_config.get('processing_rate', 1.0))
        self.enable_timing_sync = base_config.get('enable_timing_sync', True)

        self.source = base_config['source']
        self.stream_kwargs = base_config['stream_kwargs']

        source_list = ['opencv', 'rtmp', 'lsl', 'file']
        if self.source not in source_list:
            raise ValueError(f'Unknown source {self.source}, must be one of {source_list}')

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

    def _clean_up(self):
        self.stop_event.set()
        self.mqtt_client.loop_stop()
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None
        if self.graphics:
            cv2.destroyWindow(f'VFA Base {self.base_id}, Camera {self.selected_source}')
            cv2.waitKey(1)
        self._clear_threads()
        gc.collect()

    def _reinit(self):
        """Reinitialize VFABase by calling __init__ again with stored parameters."""
        self.logger.info("Starting VFA base reinitialization...")
        
        # Store the original initialization parameters
        project_dir = getattr(self, 'project_dir', None)
        config_path = getattr(self, 'config_path', None)
        mode = getattr(self, 'mode', 'full')
        graphics = getattr(self, 'graphics', True)
        verbose = getattr(self, 'verbose', False)
        
        # Clean up current state
        self._clean_up()
        
        # Call __init__ again with the original parameters
        self.__init__(project_dir=project_dir, config_path=config_path, 
                     mode=mode, graphics=graphics, verbose=verbose)
        
        self.logger.info("VFA base reinitialization completed successfully")

    def run(self):
        """Run the VFA base."""
        print('\033]0;VFA Base\007')

        func_map = {1: self._start, 2: self._set_camera, 3: self._switch_mode, 4: self._reinit}
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
            finally:
                self._clean_up()

    def _start(self):
        """Start the video streaming and MQTT client for the VFA base."""
        if not self.camera_configured:
            self.logger.warning("Camera is not configured.")
            return self._set_camera()

        self.bucket_name = select_or_create_bucket(self.influx_client)
        self._create_bucket_logger()

        if self.mode != 'analyze':
            self._configure_video_stream()
            
        self._listen_for_start_signal()

        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        self._create_thread(self._listen_for_stop_signal)

        try:
            self._start_threads()
            self._process_frames()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning("Interrupted: %s", e)
        finally:
            self._clean_up()

    def _configure_video_stream(self):
        """Configure video stream."""
        self.video_stream = VideoStream(source=self.source, **self.stream_kwargs)
        self.video_stream.start()

    def _create_bucket_logger(self):
        """Create logger for the bucket."""
        self.bucket_logger_dir = os.path.join(self.logger_dir, f'{self.bucket_name}')
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        self.logger = get_logger(f'vfa-{self.bucket_name}',
                                 os.path.join(self.bucket_logger_dir, f'vfa_base_{self.base_id}.log'),
                                 console_level=logging.DEBUG if self.verbose else logging.INFO)

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
        elif self.source == 'file':
            self.stream_kwargs['file_path'] = self.selected_source

        self.base_id = get_id()
        self.camera_configured = True
        print(f'\033]0;VFA Base {self.base_id}, Camera {self.selected_source}\007')

    def _configure_camera_params(self) -> dict | None:
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

    def _set_camera_angle(self) -> str | None:
        """Set the camera angle."""
        # Configure camera angle
        if self.angle_config:
            angles = list(self.angle_config.keys())
            print("Available camera angles:")
            for idx, angle in enumerate(angles):
                print(f"{idx}: {angle} - {self.angle_config[angle]}")
            print(f"{len(angles)}: unspecified - No specific angle")

            while True:
                try:
                    flush_input()
                    angle_input = input("Choose camera angle (press Enter for unspecified): ")
                    if angle_input == '':
                        return 'unspecified'

                    angle_idx = int(angle_input)
                    if 0 <= angle_idx < len(angles):
                        return angles[angle_idx]
                    elif angle_idx == len(angles):
                        return 'unspecified'
                    else:
                        print("Invalid selection. Please choose a valid option.")
                except ValueError:
                    print("Please enter a valid number or press Enter for unspecified.")
        else:
            return 'unspecified'

    def _detect_video_sources(self) -> list[str | int]:
        """Detect available video sources based on the source type."""
        available_sources = []
        available_source_idx = 0

        if self.source == 'opencv':
            # Detect available camera indices
            for i in range(4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"{available_source_idx} : Camera index {i} is available.")
                    available_source_idx += 1
                    available_sources.append(i)
                cap.release()

        elif self.source == 'rtmp':
            # Get RTMP URLs from configuration
            if 'RTMP' not in self.config:
                raise ValueError("RTMP configuration is missing in the YAML file.")
            if 'video_streams' not in self.config['RTMP']:
                raise ValueError("RTMP: video_streams configuration is missing in the YAML file.")
            for url in self.config['RTMP']['video_streams']:
                print(f"{available_source_idx} : RTMP stream {url} is available.")
                available_sources.append(url)
                available_source_idx += 1

        elif self.source == 'file':
            base_config = self.config.get('Base', {})

            if 'initial_sync_time' not in base_config:
                raise ValueError("initial_sync_time configuration is missing in the YAML file.")
            self.initial_sync_time = float(base_config['initial_sync_time'])
            if not validate_unix_timestamp(self.initial_sync_time):
                raise ValueError(f"Invalid initial_sync_time ({self.initial_sync_time})")

            if 'file_dir' not in base_config:
                raise ValueError("File directory configuration is missing in the YAML file.")
            file_dir = base_config['file_dir']
            if not os.path.isabs(file_dir):
                file_dir = os.path.join(self.project_dir, file_dir)
            if not os.path.exists(file_dir):
                raise ValueError(f"File directory does not exist: {file_dir}")

            # find all video files in directory (opencv-supported formats)
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            for filename in sorted(os.listdir(file_dir)):
                match = re.search(r'_(\d+(?:\.\d+)?)\.', filename)
                if match:
                    file_start_time = float(match.group(1))
                    if not validate_unix_timestamp(file_start_time):
                        self.logger.warning(f"Skipping file {filename}: invalid file_start_time ({file_start_time})")
                        continue
                    if file_start_time > self.initial_sync_time:
                        self.logger.warning(
                            f"Skipping file {filename}: file_start_time ({file_start_time}) is greater than initial_sync_time ({self.initial_sync_time})")
                        continue
                    file_path = os.path.join(file_dir, filename)
                    if any(filename.lower().endswith(ext) for ext in video_extensions):
                        print(f"{available_source_idx} : File {file_path} is available.")
                        available_sources.append(file_path)
                        available_source_idx += 1

        if not available_sources:
            self.logger.warning(f"No video sources found for {self.source}.")

        return available_sources

    def _choose_video_source(self, available_sources: list[str | int]) -> str | int | None:
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

    def _process_frames(self):
        """Process video frames and handle frame analysis."""
        self.save_path = os.path.join(self.runtime_dir, f'{self.bucket_name}/{self.chosen_camera}_{self.base_id}')
        os.makedirs(self.save_path, exist_ok=True)

        if self.mode == 'analyze':  # processing existing frames
            time.sleep(2) # wait for the synchronizer to start
            self._analyze_existing_frames(self.save_path)
            return

        if self.source == 'file':
            self._process_keyframes()
        else:
            self._process_continuous_frames()

    def _process_keyframes(self):
        """Process video frames using keyframe synchronization for file mode."""
        print("Processing VFA keyframes with timing synchronization...")
        
        filename = os.path.basename(self.selected_source)
        match = re.search(r'_(\d+(?:\.\d+)?)\.', filename)
        file_start_time = float(match.group(1))
        timestamp_offset = file_start_time
        
        # start from initial sync time, advance by keyframe_interval
        current_video_time = self.initial_sync_time - file_start_time
        target_interval = self.keyframe_interval / self.processing_rate
        
        # accumulative timing to prevent drift
        expected_real_time = time.time()
        frame_count = 0
        
        self.logger.info(f"VFA keyframe processing: interval={self.keyframe_interval}s, rate={self.processing_rate}x, "
                        f"target_interval={target_interval:.3f}s, effective FPS: {1/self.keyframe_interval:.3f}")
        
        while not self.stop_event.is_set():
            # read frame at specific time (keyframe)
            video_frame = self.video_stream.read(start_time=current_video_time)
            if video_frame is None:
                self.logger.info("Reached end of video file")
                break
                
            frame = video_frame.data
            acquired_time = current_video_time + timestamp_offset
            
            # process the frame
            processed_frame = self._process_single_frame(frame, acquired_time)
            
            # save and publish frame
            image_path = os.path.join(self.save_path, f'{acquired_time}.jpg')
            cv2.imwrite(image_path, processed_frame)
            
            if self.mode == 'full':
                try:
                    self._publish_frame(image_path, acquired_time)
                except Exception as e:
                    self.logger.warning(f"VFA publish failed: {e}")
            
            # accumulative timing synchronization
            if self.enable_timing_sync:
                frame_count += 1
                next_expected_time = expected_real_time + (frame_count * target_interval)
                current_time = time.time()
                
                if current_time < next_expected_time:
                    sleep_time = next_expected_time - current_time
                    self.logger.debug(f"Frame {frame_count}: sleeping {sleep_time:.3f}s to maintain sync")
                    time.sleep(sleep_time)
                else:
                    drift = current_time - next_expected_time
                    self.logger.debug(f"Frame {frame_count}: running {drift:.3f}s behind schedule (catching up)")
            
            # advance to next keyframe
            current_video_time += self.keyframe_interval
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _process_continuous_frames(self):
        """Process real-time video streams continuously (opencv, rtmp, lsl) using keyframe_interval for frame saving."""
        print("Processing VFA real-time streams...")
        last_saved_time = 0

        while not self.stop_event.is_set():
            video_frame = self.video_stream.read()[-1]  # read latest frame
            if video_frame is None:
                continue

            frame = video_frame.data
            acquired_time = video_frame.timestamp

            # process the frame
            processed_frame = self._process_single_frame(frame, acquired_time)

            if acquired_time - last_saved_time >= self.keyframe_interval:
                image_path = os.path.join(self.save_path, f'{acquired_time}.jpg')
                cv2.imwrite(image_path, processed_frame)
                last_saved_time = acquired_time

                if self.mode == 'full':
                    try:
                        self._publish_frame(image_path, acquired_time)
                    except Exception as e:
                        self.logger.warning(f"VFA publish failed: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _process_single_frame(self, frame, acquired_time):
        """Process a single frame and return the processed frame."""
        if self.camera_info.get("fisheye", False):
            frame = cv2.remap(frame, self.camera_info["map_1"], self.camera_info["map_2"],
                              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if self.rotate in ROTATIONS:
            frame = cv2.rotate(frame, ROTATIONS[self.rotate])

        if self.graphics:
            display_frame = cv2.resize(frame, (960, 540))
            timestamp = datetime.datetime.fromtimestamp(acquired_time).strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, (display_frame.shape[1] - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f'VFA Base {self.base_id}, Camera {self.selected_source}', display_frame)

        return frame

    def _analyze_existing_frames(self, save_path: str):
        """Analyze existing frames in the save path with timing synchronization."""
        print("Analyzing VFA existing frames with timing synchronization...")

        # Get all image files and sort by timestamp
        frame_files = [f for f in os.listdir(save_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        frame_files.sort(key=lambda x: float(x.split('.')[0].split('_')[1]))  # Sort by timestamp (assuming filename format: _timestamp.jpg)

        if not frame_files:
            self.logger.warning("No image files found in save path for analysis")
            return

        # Calculate timing parameters based on keyframe_interval and processing_rate
        target_interval = self.keyframe_interval / self.processing_rate 
        expected_real_time = time.time()
        frame_count = 0
        
        self.logger.info(f"VFA analyze mode: interval={self.keyframe_interval}s, rate={self.processing_rate}x, "
                        f"target_interval={target_interval:.3f}s, total frames: {len(frame_files)}")

        for frame_file in frame_files:
            if self.stop_event.is_set():
                break

            try:
                # Extract timestamp from filename (assuming format: timestamp.jpg)
                acquired_time = float(frame_file.split('.')[0].split('_')[1])
                image_path = os.path.join(save_path, frame_file)
                
                # Publish frame
                self._publish_frame(image_path, acquired_time)
                
                # Apply timing synchronization to simulate real-time behavior
                if self.enable_timing_sync:
                    frame_count += 1
                    next_expected_time = expected_real_time + (frame_count * target_interval)
                    current_time = time.time()
                    
                    if current_time < next_expected_time:
                        sleep_time = next_expected_time - current_time
                        self.logger.debug(f"Frame {frame_count}: sleeping {sleep_time:.3f}s to maintain sync")
                        time.sleep(sleep_time)
                    else:
                        drift = current_time - next_expected_time
                        self.logger.debug(f"Frame {frame_count}: running {drift:.3f}s behind schedule (catching up)")
                        
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Failed to parse timestamp from filename {frame_file}: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"VFA publish failed for {frame_file}: {e}")
                continue

    def _publish_frame(self, image_path: str, acquired_time: float):
        """Publish frame to MQTT for synchronization."""
        frame_data = {
            "base_id": str(self.base_id),
            "angle": self.camera_angle,
            "image_path": image_path,
            "acquired_time": acquired_time
        }

        self.mqtt_client.publish(f"{self.bucket_name}/vfa", json.dumps(frame_data))
        self.logger.info(f"Published frame path {image_path} with angle {self.camera_angle} at {acquired_time}")

    @property
    def bucket_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current bucket_name."""
        if self.bucket_name:
            return f'{self.bucket_name}/vfa/control'
        return None

