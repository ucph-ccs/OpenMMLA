import datetime
import gc
import json
import logging
import os
import re
import shutil
import threading
import time

import cv2
import numpy as np

from openmmla.bases.base import Base
from openmmla.streams.video_stream import VideoStream
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, runtime_pipeline_artifact_dir
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.config import (
    get_bases, get_base_by_id, select_source_by_index_or_name, compute_initial_sync_time,
)
from openmmla.utils.input import select_or_create_session
from openmmla.utils.logger import get_logger
from openmmla.utils.validation import validate_unix_timestamp
from .enums import ROTATIONS
from .input import get_mode


class VFABase(Base):
    """VFABase class for video frame analysis."""
    logger = get_logger('vfa-base')

    def __init__(self, project_dir: str | None, config_path: str, mode: str = 'live', graphics: bool = True,
                 store: bool = True, verbose: bool = False, session_id: str | None = None,
                 base: str | None = None):
        """Initializes the VFABase class.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            mode: operating mode, 'capture', 'analyze', or 'live'. (default: 'live')
            graphics: whether to display graphics (default: True)
            store: whether to store frames locally (default: True)
            verbose: whether to enable verbose logging (default: False)
            base: which base id from the config 'Bases' list to run; if omitted,
                pick one interactively (the only interaction).
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        # VFABase specific parameters
        self.mode = mode
        self.graphics = graphics
        self.store = store
        self.verbose = verbose
        self.launch_session_id = session_id
        self.launch_base = base

        # profile-driven: every run loads a base from the config 'Bases' list
        # (single source of truth) — by id if given, else picked interactively.
        entry = get_base_by_id(self.config, base) if base else self._pick_base_interactively()
        if entry is None:
            raise ValueError(f"Base '{base}' not found in config 'Bases'.")
        self._camera_name = entry.get('camera')
        # source_index is overloaded by source type (index / file name / stream
        # name); keep it raw and interpret it once the source is known.
        self._source_index = entry.get('source_index')
        self._base_id_override = str(entry.get('id', base))
        self._base_source = entry.get('source')  # per-base override of Base.source
        self._camera_angle = entry.get('camera_angle')  # per-base viewing angle label

        # Runtime attributes
        self.chosen_camera = None
        self.selected_source = None
        self.base_id = None
        self.camera_configured = False
        self.session_id = None
        self.video_stream = None
        self.save_path = None
        self.temp_save_path = None
        self.frame_output_path = None

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

        # source comes from the per-base Bases entry (single source of truth);
        # Base.source has been removed, so a base must define its own source
        self.source = self._base_source or base_config.get('source')
        if not self.source:
            raise ValueError(
                f"Base '{self._base_id_override}' has no 'source'. Set 'source' in its Bases entry.")
        self.stream_kwargs = base_config['stream_kwargs']

        source_list = ['opencv', 'rtmp', 'lsl', 'file']
        if self.source not in source_list:
            raise ValueError(f'Unknown source {self.source}, must be one of {source_list}')

    def _setup_directories(self):
        """Create and set up the necessary directories for runtime operations."""
        self.logger_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'vfa-base', 'logger'))
        self.runtime_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'vfa-base', 'real-time', 'runtime'))
        self.temp_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'vfa-base', 'real-time', 'temp'))
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def _setup_clients(self):
        """Initialize external service clients and internal processing objects."""
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.mongo_client = MongoDBClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)

    def _clean_up(self):
        if self.threads:
            self._stop_threads()
        self._clear_threads()
        self.stop_event.set()
        self.mqtt_client.loop_stop()
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None
        if self.graphics:
            cv2.destroyWindow(f'VFA Base {self.base_id}, Camera {self.selected_source}')
            cv2.waitKey(1)
        gc.collect()

    def _clean_stale_temp_frames(self):
        """Clean stale temporary frames before starting a new non-persistent run."""
        if getattr(self, 'store', True):
            return

        temp_save_path = getattr(self, 'temp_save_path', None)
        if not temp_save_path:
            return

        temp_root = os.path.abspath(self.temp_dir)
        temp_save_path = os.path.abspath(temp_save_path)
        try:
            if os.path.commonpath([temp_root, temp_save_path]) != temp_root:
                self.logger.warning(f"Refusing to delete non-temp VFA frame path: {temp_save_path}")
                return
        except ValueError:
            self.logger.warning(f"Refusing to delete invalid VFA frame path: {temp_save_path}")
            return

        if os.path.isdir(temp_save_path):
            shutil.rmtree(temp_save_path, ignore_errors=True)

    def _reinit(self):
        """Reinitialize VFABase by calling __init__ again with stored parameters."""
        self.logger.info("Starting VFA base reinitialization...")
        
        # Store the original initialization parameters
        project_dir = getattr(self, 'project_dir', None)
        config_path = getattr(self, 'config_path', None)
        mode = getattr(self, 'mode', 'live')
        graphics = getattr(self, 'graphics', True)
        store = getattr(self, 'store', True)
        verbose = getattr(self, 'verbose', False)
        session_id = getattr(self, 'launch_session_id', None)
        base = getattr(self, 'launch_base', None)

        # Clean up current state
        self._clean_up()

        # Call __init__ again with the original parameters
        self.__init__(project_dir=project_dir, config_path=config_path,
                     mode=mode, graphics=graphics, store=store, verbose=verbose,
                     session_id=session_id, base=base)
        
        self.logger.info("VFA base reinitialization completed successfully")

    def _pick_base_interactively(self):
        """Pick a base from the config 'Bases' list (the only interaction)."""
        bases = get_bases(self.config)
        if not bases:
            raise ValueError(
                "No bases defined. Add entries under 'Bases' in config.yml "
                "(each with id, camera, source and source_index).")
        print("Select a base:")
        for idx, b in enumerate(bases):
            print(f"  {idx}: id={b.get('id')} (camera: {b.get('camera')}, source: {b.get('source')}, "
                  f"source_index: {b.get('source_index')}, angle: {b.get('camera_angle')})")
        while True:
            sel = input("Base number [0]: ").strip()
            try:
                index = int(sel) if sel else 0
            except ValueError:
                index = -1
            if 0 <= index < len(bases):
                return bases[index]
            print("Invalid selection. Please enter a valid base number.")

    def run(self):
        """Run the VFA base — fully profile-driven (no menus)."""
        print('\033]0;VFA Base\007')
        try:
            self._set_camera()
            if self.camera_configured:
                self._start()
            else:
                self.logger.error("Camera setup failed (no camera/source resolved from config).")
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning(
                f"VFA base stopped: "
                f"{'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}",
                exc_info=not isinstance(e, KeyboardInterrupt))
        finally:
            self._clean_up()

    def _start(self):
        """Start the video streaming and MQTT client for the VFA base."""
        if not self.camera_configured:
            self.logger.warning("Camera is not configured.")
            return self._set_camera()

        self.session_id = self.launch_session_id or select_or_create_session(self.mongo_client)
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
        self.stream_kwargs['project_dir'] = self.project_dir
        self.video_stream = VideoStream(source=self.source, **self.stream_kwargs)
        self.video_stream.start()

    def _create_bucket_logger(self):
        """Create logger for the bucket."""
        self.bucket_logger_dir = os.fspath(
            pipeline_section_dir(self.project_dir, self.session_id, 'vfa-base', 'logger')
        )
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        copy_config_snapshot(self.config_path, self.project_dir, self.session_id, 'vfa-base')
        self.logger = get_logger(f'vfa-{self.session_id}',
                                 os.path.join(self.bucket_logger_dir, f'vfa_base_{self.base_id}.log'),
                                 console_level=logging.DEBUG if self.verbose else logging.INFO)

    def _switch_mode(self):
        """Switch the operating mode between 'capture', 'analyze' and 'live'."""
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
        elif self.source == 'lsl':
            self.stream_kwargs['lsl_name'] = self.selected_source

        # base id comes from the selected base entry
        self.base_id = str(self._base_id_override) if self._base_id_override else '1'
        self.camera_configured = True
        print(f'\033]0;VFA Base {self.base_id}, Camera {self.selected_source}\007')

    def _configure_camera_params(self) -> dict | None:
        """Configure camera intrinsic parameters."""
        cameras = self.config.get('Cameras', {})
        camera_choices = sorted(list(cameras.keys()))
        if not camera_choices:
            return None

        # camera comes from the selected base entry (Bases[].camera); fall back
        # to the first calibrated camera if unspecified
        if self._camera_name and self._camera_name in cameras:
            self.chosen_camera = self._camera_name
        else:
            self.chosen_camera = camera_choices[0]
        self.logger.info(f"Using camera '{self.chosen_camera}'")

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
        """Resolve the camera angle from the selected base entry (Bases[].camera_angle)."""
        angle = self._camera_angle
        if angle and not str(angle).startswith('<'):
            if self.angle_config and angle not in self.angle_config:
                self.logger.warning(
                    f"camera_angle '{angle}' is not in Base.angle_config; using it as-is.")
            return str(angle)
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
            from openmmla.utils.constants import get_stream_urls
            rtmp_urls = get_stream_urls(self.config, "rtmp")
            if not rtmp_urls:
                raise ValueError("No RTMP streams found in Streams (or legacy RTMP) config section.")
            for url in rtmp_urls:
                print(f"{available_source_idx} : RTMP stream {url} is available.")
                available_sources.append(url)
                available_source_idx += 1

        elif self.source == 'file':
            base_config = self.config.get('Base', {})

            # file_dir is optional and defaults to the project directory
            file_dir = base_config.get('file_dir') or self.project_dir
            if not os.path.isabs(file_dir):
                file_dir = os.path.join(self.project_dir, file_dir)
            if not os.path.isdir(file_dir):
                raise ValueError(f"File directory does not exist: {file_dir}")

            # initial_sync_time is auto-computed as the latest file start time
            # (the common point where every file has begun), unless explicitly set
            video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
            self.initial_sync_time = compute_initial_sync_time(
                file_dir, base_config.get('initial_sync_time'), exts=video_exts)
            if not validate_unix_timestamp(self.initial_sync_time):
                raise ValueError(f"Invalid initial_sync_time ({self.initial_sync_time})")

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

        elif self.source == 'lsl':
            # the LSL stream is selected by name (source_index holds the name)
            if self._source_index:
                print(f"0 : LSL stream '{self._source_index}' is available.")
                available_sources.append(self._source_index)
            else:
                raise ValueError("LSL source requires a stream name in the base's source_index.")

        if not available_sources:
            self.logger.warning(f"No video sources found for {self.source}.")

        return available_sources

    def _choose_video_source(self, available_sources: list[str | int]) -> str | int | None:
        """Pick the video source for the selected base (source_index is an index
        for opencv/rtmp, or a file/stream name for file/lsl)."""
        if not available_sources:
            return None
        selected_source = select_source_by_index_or_name(self._source_index, available_sources)
        self.logger.info(f"Using video source {selected_source}")
        return selected_source

    def _process_frames(self):
        """Process video frames and handle frame analysis."""
        real_time_dir = pipeline_section_dir(self.project_dir, self.session_id, 'vfa-base', 'real-time')
        self.runtime_dir = os.fspath(real_time_dir / 'runtime')
        self.temp_dir = os.fspath(real_time_dir / 'temp')
        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        self.save_path = os.path.join(self.runtime_dir, f'{self.chosen_camera}_{self.base_id}')
        self.temp_save_path = os.path.join(self.temp_dir, f'{self.chosen_camera}_{self.base_id}')

        if self.mode == 'analyze':  # analyze this session's previously recorded frames
            frame_source_path = self.save_path
            if not frame_source_path:
                self.logger.warning("No frame source is configured for analysis.")
                return
            os.makedirs(frame_source_path, exist_ok=True)
            time.sleep(2)  # wait for the synchronizer to start
            self._analyze_existing_frames(str(frame_source_path))
            return

        self.frame_output_path = self.save_path if self.store else self.temp_save_path
        if self.store or self.mode == 'live':
            if not self.store:
                self._clean_stale_temp_frames()
            os.makedirs(self.frame_output_path, exist_ok=True)
        elif self.mode == 'capture':
            self.logger.warning("VFA capture mode is running with store frames disabled; frames will not be written.")

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
            
            if self.store or self.mode == 'live':
                image_path = os.path.join(self.frame_output_path, f'{acquired_time}.jpg')
                cv2.imwrite(image_path, processed_frame)

                if self.mode == 'live':
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
                if self.store or self.mode == 'live':
                    image_path = os.path.join(self.frame_output_path, f'{acquired_time}.jpg')
                    cv2.imwrite(image_path, processed_frame)
                last_saved_time = acquired_time

                if self.mode == 'live':
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

    @staticmethod
    def _frame_timestamp(filename: str) -> float:
        stem = os.path.splitext(os.path.basename(filename))[0]
        candidates = [stem]
        if '_' in stem:
            candidates.append(stem.rsplit('_', 1)[1])
        candidates.extend(re.findall(r'\d+(?:\.\d+)?', stem))

        for candidate in reversed(candidates):
            try:
                timestamp = float(candidate)
            except ValueError:
                continue
            if validate_unix_timestamp(timestamp):
                return timestamp
        raise ValueError(f"no unix timestamp found in frame filename: {filename}")

    @staticmethod
    def _frame_files(path: str) -> list[str]:
        if not os.path.isdir(path):
            return []
        return [
            f for f in os.listdir(path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def _analyze_existing_frames(self, save_path: str):
        """Analyze existing frames in the save path with timing synchronization."""
        print("Analyzing VFA existing frames with timing synchronization...")

        # Get all image files and sort by timestamp
        frame_files = self._frame_files(save_path)
        valid_frame_files = []
        for frame_file in frame_files:
            try:
                valid_frame_files.append((self._frame_timestamp(frame_file), frame_file))
            except ValueError as e:
                self.logger.warning(f"Skipping frame {frame_file}: {e}")
        valid_frame_files.sort(key=lambda item: item[0])

        if not valid_frame_files:
            self.logger.warning("No image files found in save path for analysis")
            return

        # Calculate timing parameters based on keyframe_interval and processing_rate
        target_interval = self.keyframe_interval / self.processing_rate 
        expected_real_time = time.time()
        frame_count = 0
        
        self.logger.info(f"VFA analyze mode: interval={self.keyframe_interval}s, rate={self.processing_rate}x, "
                        f"target_interval={target_interval:.3f}s, total frames: {len(valid_frame_files)}")

        for acquired_time, frame_file in valid_frame_files:
            if self.stop_event.is_set():
                break

            try:
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
        store_frames = bool(self.store) or self.mode == 'analyze'
        frame_data = {
            "base_id": str(self.base_id),
            "angle": self.camera_angle,
            "image_path": image_path,
            "acquired_time": acquired_time,
            "store_frames": store_frames
        }

        self.mqtt_client.publish(f"{self.session_id}/vfa", json.dumps(frame_data))
        self.logger.info(
            f"Published frame path {image_path} with angle {self.camera_angle} at {acquired_time} "
            f"(store_frames={store_frames})"
        )

    @property
    def session_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current session_id."""
        if self.session_id:
            return f'{self.session_id}/vfa/control'
        return None
