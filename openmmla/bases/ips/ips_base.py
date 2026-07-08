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
from pupil_apriltags import Detector

from openmmla.bases.base import Base
from openmmla.streams.video_stream import VideoStream
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, runtime_pipeline_artifact_dir
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_session, show_error_and_pause
from openmmla.utils.logger import get_logger
from openmmla.utils.validation import validate_unix_timestamp
from .enums import ROTATIONS
from .input import get_bases, get_base_by_id, select_source_by_index_or_name, compute_initial_sync_time
from .track_utils import PoseStabilizer, NormalVectorStabilizer, TagRelationTracker
from .vector import is_tag_looking_at_another_2d, get_2d_outward_normal_vector


class IPSBase(Base):
    """IPSBase class for real-time indoor positioning."""
    logger = get_logger('ips-base')

    def __init__(self, project_dir: str | None, config_path: str, graphics: bool = True,
                 store: bool = True, verbose: bool = False, session_id: str | None = None,
                 base: str | None = None):
        """Initialize the IPSBase class.

        Args:
            config_path: path to the configuration file
            project_dir: path to the project directory
            graphics: whether to show graphics (default: True)
            store: whether to store the video frames (default: True)
            verbose: whether to enable verbose logging (default: False)
            base: which base id from the config 'Bases' list to run; if omitted,
                pick one interactively.
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        # IPSBase specific parameters
        self.graphics = graphics
        self.store = store
        self.verbose = verbose
        self.launch_session_id = session_id
        self.max_badge_id = 12

        # profile-driven: load this base from the config 'Bases' list
        entry = get_base_by_id(self.config, base) if base else self._pick_base_interactively()
        if entry is None:
            raise ValueError(f"Base '{base}' not found in config 'Bases'.")
        self._camera_name = entry.get('camera')
        # source_index is overloaded by source type (index / file name / stream
        # name); keep it raw and interpret it once the source is known.
        self._source_index = entry.get('source_index')
        self._base_id_override = str(entry.get('id', base))
        self._base_source = entry.get('source')  # per-base override of Base.source

        # Runtime attributes
        self.chosen_camera = None
        self.camera_info = {}
        self.selected_source = None
        self.base_id = None  # the base camera id
        self.main_id = None  # the main camera id
        self.transform_matrices_dict = None
        self.camera_configured = False
        self.session_id = None
        self.video_stream = None

        # Threading attributes
        self.stop_event = threading.Event()
        self.threads = []

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        """Set up attributes from YAML configuration."""
        base_config = self.config.get('Base', {})
        self.tag_size = float(base_config.get('tag_size', 0.061))
        self.families = base_config.get('families', 'tag36h11')
        self.res = tuple(base_config.get('resolution', (1920, 1080)))
        self.rotate = int(base_config.get('rotate', 0))
        self.fps = int(base_config.get('fps', 30))

        # file processing configuration (file sources always use keyframe processing)
        self.keyframe_interval = float(base_config.get('keyframe_interval', 1.0))
        self.processing_rate = float(base_config.get('processing_rate', 1.0))
        self.enable_timing_sync = base_config.get('enable_timing_sync', True)

        # source comes from the per-base Bases entry (single source of truth);
        # Base.source has been removed, so a base must define its own source
        self.source = self._base_source or base_config.get('source')
        if not self.source:
            raise ValueError(
                f"Base '{self._base_id_override}' has no 'source'. Set 'source' in its Bases entry.")
        self.stream_kwargs = base_config['stream_kwargs']
        self.stream_kwargs['resolution'] = self.res
        self.stream_kwargs['fps'] = self.fps

        source_list = ['opencv', 'rtmp', 'lsl', 'file']
        if self.source not in source_list:
            raise ValueError(f'Unknown source {self.source}, must be one of {source_list}')

    def _setup_directories(self):
        """Set up directories."""
        self.runtime_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'ips-base', 'real-time', 'runtime'))
        self.logger_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'ips-base', 'logger'))
        self.camera_sync_dir = os.path.join(self.project_dir, 'camera_sync')
        self.camera_calib_dir = os.path.join(self.project_dir, 'camera_calib')
        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.camera_sync_dir, exist_ok=True)
        os.makedirs(self.camera_calib_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.mongo_client = MongoDBClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.detector = Detector(families=self.families, nthreads=4)
        self.pose_stabilizer = PoseStabilizer(smoothing=0.7)
        self.normal_stabilizer = NormalVectorStabilizer(smoothing=0.7)
        self.relation_tracker = TagRelationTracker(min_consistent_frames=2)

    def _clean_up(self):
        """Clean up runtime variables and free memory."""
        if self.threads:
            self._stop_threads()
        self._clear_threads()
        self.mqtt_client.loop_stop()
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None
        if self.graphics:
            cv2.destroyWindow(f'AprilTags Detection from camera {self.base_id}')
            cv2.waitKey(1)
        gc.collect()

    def run(self):
        """Run the IPS base — fully profile-driven (no menus)."""
        print('\033]0;IPS Base\007')
        try:
            self._set_camera()
            if self.camera_configured:
                self._start_detection()
            else:
                self.logger.error("Camera setup failed (no camera/source resolved from config).")
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning(
                f"IPS base stopped: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}",
                exc_info=not isinstance(e, KeyboardInterrupt))
        finally:
            self._clean_up()

    def _start_detection(self):
        """Start AprilTag detection"""
        if not self.camera_configured:
            self.logger.warning("Camera is not configured.")
            return self._set_camera()

        # bucket selection
        self.session_id = self.launch_session_id or select_or_create_session(self.mongo_client)
        self._create_bucket_logger()

        # configure video stream and start it
        self._configure_video_stream()
        self._listen_for_start_signal()

        # reinitialize mqtt client
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        # create threads
        self._create_thread(self._listen_for_stop_signal)

        exception_occurred = None
        try:
            self._start_threads()
            self._process_frames()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning("%s, capture interrupted.", e, exc_info=False)
            exception_occurred = e
        finally:
            self._detection_handler(exception_occurred)

    def _create_bucket_logger(self):
        self.bucket_logger_dir = os.fspath(
            pipeline_section_dir(self.project_dir, self.session_id, 'ips-base', 'logger')
        )
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        copy_config_snapshot(self.config_path, self.project_dir, self.session_id, 'ips-base')
        self.logger = get_logger(f'ips-base-{self.session_id}',
                                 os.path.join(self.bucket_logger_dir, f'ips_base_{self.base_id}.log'),
                                 console_level=logging.DEBUG if self.verbose else logging.INFO)

    def _detection_handler(self, e: Exception | None):
        """Handle exceptions and stop all threads.

        Args:
            e: the exception occurred during the detection process
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")
        self._clean_up()

    def _set_camera(self):
        """Set up video source and base id."""
        self.camera_configured = False

        self.transform_matrices_dict = self._load_transform_matrices()
        if self.transform_matrices_dict is None:
            self.logger.warning("No transformation matrices found, please do the camera sync first.")
            return

        self.camera_info = self._configure_camera_params()
        if self.camera_info is None:
            self.logger.warning("No camera parameters found, please calibrate camera first.")
            return

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
        self.logger.info(f"Using source: {self.selected_source}")

        self.base_id = str(self._base_id_override) if self._base_id_override else '1'
        self.camera_configured = True
        print(f'\033]0;IPS Base {self.base_id}\007')

    def _configure_camera_params(self):
        """Configure camera intrinsic parameters for the selected base's camera."""
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

    def _detect_video_sources(self):
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

    def _choose_video_source(self, available_sources: list[str] | None) -> str | None:
        """Pick the video source for the selected base (source_index is an index
        for opencv/rtmp, or a file/stream name for file/lsl)."""
        if not available_sources:
            return None
        selected_source = select_source_by_index_or_name(self._source_index, available_sources)
        self.logger.info(f"Using video source {selected_source}")
        return selected_source

    def _pick_base_interactively(self):
        """Pick a base from the config 'Bases' list (the only interaction)."""
        bases = get_bases(self.config)
        if not bases:
            raise ValueError(
                "No bases defined. Add entries under 'Bases' in config.yml (each with an 'id').")
        print("Select a base:")
        for idx, b in enumerate(bases):
            print(f"  {idx}: id={b.get('id')} (camera: {b.get('camera')}, source_index: {b.get('source_index')})")
        while True:
            sel = input("Base number [0]: ").strip()
            try:
                index = int(sel) if sel else 0
            except ValueError:
                index = -1
            if 0 <= index < len(bases):
                return bases[index]
            print("Invalid selection. Please enter a valid base number.")

    def _configure_video_stream(self):
        """Configure video stream."""
        self.stream_kwargs['project_dir'] = self.project_dir
        self.video_stream = VideoStream(source=self.source, **self.stream_kwargs)
        self.video_stream.start()

    def _process_frames(self):
        """Process video frames and detect AprilTags."""
        if self.source == 'file':
            self._process_keyframes()
        else:
            self._process_continuous_frames()

    def _process_keyframes(self):
        """Process video frames using keyframe synchronization for file mode."""
        print("Processing keyframes with timing synchronization...")
        runtime_root = pipeline_section_dir(self.project_dir, self.session_id, 'ips-base', 'real-time') / 'runtime'
        runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = os.fspath(runtime_root)
        save_path = os.path.join(self.runtime_dir, f'ips_{self.base_id}')
        os.makedirs(save_path, exist_ok=True)
        
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
        
        self.logger.info(f"Keyframe processing: interval={self.keyframe_interval}s, rate={self.processing_rate}x, "
                         f"target_interval={target_interval:.3f}s, effective FPS: {1/self.keyframe_interval:.1f}")
        
        while not self.stop_event.is_set():
            # read frame at specific time (keyframe)
            video_frame = self.video_stream.read(start_time=current_video_time)
            if video_frame is None:
                self.logger.info("Reached end of video file")
                break
                
            frame = video_frame.data
            acquired_time = current_video_time + timestamp_offset
            
            # process the frame
            tags, tag_relations = self._process_single_frame(frame, acquired_time)
            
            # save frame if store is enabled
            if self.store:
                cv2.imwrite(os.path.join(save_path, f'{acquired_time}.jpg'), frame)
            
            # publish result
            message = {
                "base_id": self.base_id,
                "tags": tags,
                "tag_relations": tag_relations,
                "acquired_time": acquired_time
            }
            self.logger.debug(message)
            message_str = json.dumps(message)
            self.mqtt_client.publish(f'{self.session_id}/ips', message_str, qos=0, retain=False)
            
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
        """Process real-time video streams continuously (opencv, rtmp, lsl)."""
        print("Processing real-time streams...")
        runtime_root = pipeline_section_dir(self.project_dir, self.session_id, 'ips-base', 'real-time') / 'runtime'
        runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = os.fspath(runtime_root)
        save_path = os.path.join(self.runtime_dir, f'ips_{self.base_id}')
        os.makedirs(save_path, exist_ok=True)
        frames_count = 0

        while not self.stop_event.is_set():
            video_frame = self.video_stream.read()[-1]  # read latest frame
            if video_frame is None:
                continue

            frame = video_frame.data
            frames_count += 1
            acquired_time = video_frame.timestamp

            # process the frame
            tags, tag_relations = self._process_single_frame(frame, acquired_time)

            if self.store and frames_count % self.fps == 0:
                frames_count = 0
                cv2.imwrite(os.path.join(save_path, f'{acquired_time}.jpg'), frame)

            message = {
                "base_id": self.base_id,
                "tags": tags,
                "tag_relations": tag_relations,
                "acquired_time": acquired_time
            }
            self.logger.debug(message)
            message_str = json.dumps(message)
            self.mqtt_client.publish(f'{self.session_id}/ips', message_str, qos=0, retain=False)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _process_single_frame(self, frame, acquired_time):
        """Process a single frame and return detected tags and relations."""
        if self.camera_info.get("fisheye", False):
            frame = cv2.remap(frame, self.camera_info["map_1"], self.camera_info["map_2"],
                              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if self.rotate in ROTATIONS:
            frame = cv2.rotate(frame, ROTATIONS[self.rotate])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_info["params"],
                                       tag_size=self.tag_size)

        self.relation_tracker.step()

        tags = {}
        tag_relations = {}

        for tag in results:
            if tag.decision_margin < 10 or int(tag.tag_id) > self.max_badge_id:
                continue

            # stabilize pose
            stabilized_R, stabilized_t = self.pose_stabilizer.update(tag.tag_id, tag.pose_R, tag.pose_t)
            tag.pose_R = stabilized_R
            tag.pose_t = stabilized_t

            # stabilize normal
            normal, tag = get_2d_outward_normal_vector(tag)
            smoothed_normal = self.normal_stabilizer.update(tag.tag_id, normal)

            tags[tag.tag_id] = [list(tag.pose_R.tolist()), list(tag.pose_t.tolist())]
            tag_relations.setdefault(tag.tag_id, [])

            # decide if the tag is looking at another tag
            other_tags = [t for t in results if t.tag_id != tag.tag_id]
            for other_tag in other_tags:
                is_seeing = is_tag_looking_at_another_2d(tag, other_tag, cosine_threshold=-0.94,
                                                         distance_threshold=1)
                self.relation_tracker.update(tag.tag_id, other_tag.tag_id, is_seeing)

                if self.relation_tracker.is_confirmed(tag.tag_id, other_tag.tag_id):
                    tag_relations[tag.tag_id].append(str(other_tag.tag_id))

            # draw visualizations
            if self.graphics:
                corners = np.int32(tag.corners)
                tag_center = np.mean(corners, axis=0)
                arrow_dir = smoothed_normal[:2]
                scale_factor = 50
                end_point = tag_center + scale_factor * arrow_dir
                cv2.arrowedLine(frame, tuple(np.int32(tag_center)), tuple(np.int32(end_point)), (0, 0, 255), 2)
                cv2.polylines(frame, [corners], True, (0, 255, 0), thickness=2)
                cv2.putText(frame, str(tag.tag_id), org=(int(tag_center[0]) + 10, int(tag_center[1]) + 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)

        if self.graphics:
            display_frame = cv2.resize(frame, (960, 540))
            timestamp = datetime.datetime.fromtimestamp(acquired_time).strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, (display_frame.shape[1] - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f'AprilTags Detection from camera {self.base_id}', display_frame)

        return tags, tag_relations

    def _load_transform_matrices(self):
        """Load transformation matrices."""
        transformation_choices = [d for d in os.listdir(self.camera_sync_dir) if
                                  d.startswith('transformation_matrices_')]
        for idx, choice in enumerate(transformation_choices):
            print(f"{idx}: {choice}")

        if not transformation_choices:
            return None

        # non-interactive: use the first transform matrix file; main_id is its
        # suffix (e.g. transformation_matrices_m.json -> main camera id 'm')
        chosen_transformation = sorted(transformation_choices)[0]
        self.main_id = chosen_transformation.split('_')[-1].split('.')[0]
        self.logger.info(f"Using transform matrices '{chosen_transformation}' (main: {self.main_id})")

        with open(os.path.join(self.camera_sync_dir, chosen_transformation), 'r') as file:
            return json.load(file)

    @property
    def session_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current session_id."""
        if self.session_id:
            return f'{self.session_id}/ips/control'
        return None
    
