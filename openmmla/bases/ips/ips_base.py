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
from openmmla.streams.stream_receiver import StreamUnavailable
from openmmla.streams.video_stream import VideoStream
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, runtime_pipeline_artifact_dir
from openmmla.utils.config import bases_by_room, main_of_base, main_without_matrices
from openmmla.utils import session_provenance
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_session, show_error_and_pause
from openmmla.utils.logger import get_logger
from openmmla.utils.session_sources import record_joined, record_left, source_entry
from openmmla.utils.validation import validate_unix_timestamp
from openmmla.utils.video.turn import (base_capture_turn, normalize_turn, pose_in_sensor_frame, sensor_size,
                                       total_turn, turned_camera_matrix, turned_intrinsics, turned_size)
from .enums import ROTATIONS
from .input import get_bases, get_base_by_id, select_source_by_index_or_name, compute_initial_sync_time
from .intrinsics import FrameIntrinsics, calibration_resolution
from .track_utils import PoseStabilizer, outward_normal_2d
from .vector import canonical_rotation, is_tag_looking_at_another_2d

# a weaker detection is not a badge (calibration.MIN_DECISION_MARGIN)
MIN_DECISION_MARGIN = 10


class IPSBase(Base):
    """IPSBase class for real-time indoor positioning."""
    logger = get_logger('ips-base')
    # the turn the capture gave the frames (the rotate of the Streams entry the source pulls, else
    # the Bases entry's capture_rotate): none until the source is known
    capture_turn = 0

    def __init__(self, project_dir: str | None, config_path: str, graphics: bool | None = None,
                 store: bool = True, verbose: bool = False, session_id: str | None = None,
                 base: str | None = None):
        """Initialize the IPSBase class.

        Args:
            config_path: path to the configuration file
            project_dir: path to the project directory
            graphics: whether to show the frames in a window; None (default) shows them unless the
                source is a stream, which a base pulls on a machine nobody watches, often over SSH
                with no display (the dashboard's camera tiles draw what the bases found)
            store: whether to store the video frames (default: True)
            verbose: whether to enable verbose logging (default: False)
            session_id: the session to join; if omitted, choose or create one interactively
            base: which base id from the config 'Bases' list to run; if omitted,
                pick one interactively (launched with a session id, the only entry
                there is is taken without asking).
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        # IPSBase specific parameters
        self.graphics = graphics
        self.store = store
        self.verbose = verbose
        self.launch_session_id = session_id
        self.max_badge_id = 12

        # profile-driven: load this base from the config 'Bases' list
        entry = self._resolve_base(base)
        self._base_entry = entry  # noted in the session it joins (session_sources)
        self._camera_name = entry.get('camera')
        # source_index is overloaded by source type (index / file name / stream
        # name); keep it raw and interpret it once the source is known.
        self._source_index = entry.get('source_index')
        self._base_id_override = str(entry.get('id', base))
        self._base_source = entry.get('source')  # per-base override of Base.source

        # Runtime attributes
        self.chosen_camera = None
        self.camera_info = {}
        self.intrinsics = None  # the detector's intrinsics per frame size (FrameIntrinsics)
        self.selected_source = None
        self.base_id = None  # the base camera id
        self.main_id = None  # the main camera id
        self.transform_matrices_dict = None
        self.transform_matrices_file = None  # the camera_sync file the matrices came from
        self.camera_configured = False
        self.session_id = None
        self.video_stream = None
        self.stream_name = None  # the Streams entry a 'stream' source pulls
        self._joined = None  # (session id, source key) once noted in the session, until it leaves

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
        # 0, 90, 180 or 270 (-90 is 270, anything else 0): the frame is turned by what the
        # intrinsics and the poses are turned by
        self.rotate = normalize_turn(base_config.get('rotate', 0))
        self.fps = int(base_config.get('fps', 30))

        # file processing configuration (file sources always use keyframe processing)
        self.keyframe_interval = float(base_config.get('keyframe_interval', 1.0))
        self.processing_rate = float(base_config.get('processing_rate', 1.0))
        self.enable_timing_sync = base_config.get('enable_timing_sync', True)

        # the window's drawing alone is smoothed; what the base publishes is the raw pose of every detection
        self.display_smoothing = float(base_config.get('display_smoothing', 0.7))
        self.display_reset_seconds = float(base_config.get('display_reset_seconds', 2.0))

        # source comes from the per-base Bases entry (single source of truth);
        # Base.source has been removed, so a base must define its own source
        self.source = self._base_source or base_config.get('source')
        if not self.source:
            raise ValueError(
                f"Base '{self._base_id_override}' has no 'source'. Set 'source' in its Bases entry.")
        self.stream_kwargs = base_config['stream_kwargs']
        self.stream_kwargs['resolution'] = self.res
        self.stream_kwargs['fps'] = self.fps

        from openmmla.utils.constants import normalize_source
        self.source = normalize_source(self.source)
        source_list = ['opencv', 'stream', 'lsl', 'file']
        if self.source not in source_list:
            raise ValueError(f'Unknown source {self.source}, must be one of {source_list}')
        if self.graphics is None:
            self.graphics = self.source != 'stream'

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
        # for the window only: never on the poses the base publishes
        self.display_stabilizer = PoseStabilizer(smoothing=self.display_smoothing,
                                                 reset_seconds=self.display_reset_seconds)

    def _clean_up(self):
        """Clean up runtime variables and free memory."""
        # noted first: stopping the threads and the stream can take seconds, and a process killed
        # meanwhile (a closed terminal window) would never note that it left
        self._note_left()
        if self.threads:
            self._stop_threads()
        self._clear_threads()
        self.mqtt_client.loop_stop()
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None
        if self.graphics:
            try:
                cv2.destroyWindow(f'AprilTags Detection from camera {self.base_id}')
                cv2.waitKey(1)
            except cv2.error as e:
                # no window was opened (STOP before START), or an earlier clean-up closed it
                self.logger.debug(f"No detection window to close: {e}")
        gc.collect()

    def _close_clients(self):
        """Close the connections before the process exits."""
        for close in (self.mqtt_client.disconnect, self.influx_client.close, self.mongo_client.close,
                      self.redis_client.close):
            try:
                close()
            except Exception as e:
                self.logger.debug(f"While closing a client on exit: {e}")

    def run(self):
        """Run the IPS base — fully profile-driven (no menus)."""
        print('\033]0;IPS Base\007')
        try:
            self._set_camera()
            if self.camera_configured:
                self._start_detection()
            else:
                self.logger.error("Camera setup failed (no camera/source resolved from config).")
        except StreamUnavailable as e:
            # in plain words, without a traceback: the stream is not there
            self.logger.error(f"{e} IPS base {self.base_id} exits.")
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning(
                f"IPS base stopped: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}",
                exc_info=not isinstance(e, KeyboardInterrupt))
        finally:
            self._clean_up()
            self._close_clients()

    def _start_detection(self):
        """Start AprilTag detection"""
        if not self.camera_configured:
            self.logger.warning("Camera is not configured.")
            return self._set_camera()

        # bucket selection
        self.session_id = self.launch_session_id or select_or_create_session(self.mongo_client)
        self._create_bucket_logger()
        self._note_joined()
        self._record_provenance()

        # subscribed first: a stream that is not up yet is waited for (connect_wait), and a START or STOP
        # sent meanwhile is heard once it is up
        self._subscribe_control()
        # configure video stream and start it
        self._configure_video_stream()
        if not self._listen_for_start_signal():
            # STOP came before START: the run ended before a frame was processed
            self.logger.info(f"The run of session {self.session_id} was stopped before it started; "
                             f"IPS base {self.base_id} leaves the session.")
            self._clean_up()
            return

        # reinitialize mqtt client
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        # create threads
        self._create_thread(self._listen_for_stop_signal)

        exception_occurred = None
        try:
            self._start_threads()
            self._process_frames()
        except StreamUnavailable as e:
            # the stream did not come back after a drop: said in plain words, and the run ends
            self.logger.error(f"{e} Capture interrupted: IPS base {self.base_id} exits.")
            exception_occurred = e
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
        if self.intrinsics is not None:
            self.intrinsics.logger = self.logger  # a scaling of the intrinsics is said in the session's log

    def _record_provenance(self):
        """Note in the session what this base runs with (openmmla.utils.session_provenance): its
        flags, the camera and its intrinsics, the tag and frame settings, the stream it takes, the
        transformation matrices (the file copied next to the config too) and the config with the
        secrets masked. A failure is a warning, never a stop."""
        if not self.session_id:
            return
        try:
            cameras = self.config.get('Cameras', {}) or {}
            camera = {'name': self.chosen_camera, **(cameras.get(self.chosen_camera) or {})} if self.chosen_camera else None
            matrices = None
            if self.main_id is not None:
                matrices = {'file': self.transform_matrices_file, 'main_id': self.main_id,
                            'matrices': self.transform_matrices_dict}
            if self.transform_matrices_file:
                copy_config_snapshot(os.path.join(self.camera_sync_dir, self.transform_matrices_file),
                                     self.project_dir, self.session_id, 'ips-base')
            entry = session_provenance.component_entry(
                'ips', 'base', self.base_id,
                arguments={'graphics': self.graphics, 'store': self.store, 'verbose': self.verbose,
                           'session_id': self.launch_session_id, 'base': self._base_entry.get('id')},
                parameters={
                    'base_id': self.base_id, 'main_id': self.main_id, 'camera': camera,
                    'tag_size': self.tag_size, 'families': self.families, 'max_badge_id': self.max_badge_id,
                    'resolution': self.res, 'rotate': self.rotate, 'capture_turn': self.capture_turn,
                    'fps': self.fps,
                    'calibration_resolution': (self.intrinsics.calibration_size if self.intrinsics else None),
                    'published_poses': 'raw', 'pose_frame': 'sensor', 'display_smoothing': self.display_smoothing,
                    'display_reset_seconds': self.display_reset_seconds,
                    'keyframe_interval': self.keyframe_interval, 'processing_rate': self.processing_rate,
                    'enable_timing_sync': self.enable_timing_sync,
                    'source': self.source, 'source_index': self._source_index,
                    'selected_source': self.selected_source, 'stream': self.stream_name,
                    'initial_sync_time': getattr(self, 'initial_sync_time', None),
                    'stream_kwargs': self.stream_kwargs,
                },
                files={'transformation_matrices': matrices},
                config=self.config, config_path=self.config_path, project_dir=self.project_dir)
            session_provenance.record_component(self.mongo_client, self.session_id, entry, self.project_dir,
                                                'ips-base', log=self.logger)
        except Exception as e:
            self.logger.warning(f"Could not note in session {self.session_id} what IPS base {self.base_id} runs with: {e}")

    def _note_joined(self):
        """Note in the session which Bases entry this base is and the stream it takes, so that the
        console finds the session's own recordings later. A failure is a warning, never a stop."""
        stream, url = (self.stream_name, self.selected_source) if self.source == 'stream' else (None, None)
        try:
            entry = source_entry('ips', self._base_entry, self.config, stream=stream, url=url)
        except Exception as e:
            self.logger.warning(f"Could not note in session {self.session_id} which stream base "
                                f"{self.base_id} takes: {e}")
            return
        if record_joined(self.mongo_client, self.session_id, entry, log=self.logger):
            self._joined = (self.session_id, entry['key'])

    def _note_left(self):
        """Note once, on the way out, that this base left the session it joined."""
        if not self._joined:
            return
        (session_id, key), self._joined = self._joined, None
        record_left(self.mongo_client, session_id, key, log=self.logger)

    def _detection_handler(self, e: Exception | None):
        """Handle exceptions and stop all threads.

        Args:
            e: the exception occurred during the detection process
        """
        self._note_left()  # before the thread joins below, which can take seconds
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
        elif self.source == 'stream':
            self.stream_kwargs['url'] = self.selected_source
        elif self.source == 'file':
            self.stream_kwargs['file_path'] = self.selected_source
        elif self.source == 'lsl':
            self.stream_kwargs['lsl_name'] = self.selected_source
        self.logger.info(f"Using source: {self.selected_source}")

        self.base_id = str(self._base_id_override) if self._base_id_override else '1'
        self._set_capture_turn()
        self.camera_configured = True
        print(f'\033]0;IPS Base {self.base_id}\007')

    @property
    def turn(self) -> int:
        """how far the frames the detector reads are turned from the sensor's picture: the
        capture's turn, then Base.rotate."""
        return total_turn(self.capture_turn, self.rotate)

    def _set_capture_turn(self):
        """the turn the capture applies to the stream this base pulls (openmmla.utils.video.turn):
        the intrinsics are turned with the frames, a fisheye camera's frames are remapped turned,
        and the poses are reported in the sensor's frame all the same."""
        self.capture_turn = base_capture_turn(self.config, self._base_entry, self.source, self.selected_source,
                                              self.stream_name)
        if self.rotate:
            # until 2026-10-02 a base turning its frames reported the poses on the turned frame
            self.logger.warning(f"Base.rotate {self.rotate}: the poses are reported in the camera's frame as the "
                                f"sensor gives it, not on the turned frame. Camera Sync matrices fitted with "
                                f"ips-ctag under this Base.rotate before 2026-10-02 are in the turned frame: fit "
                                f"them again (or with mmla ses-calibrate).")
        if self.capture_turn and self.rotate:
            self.logger.warning(f"The frames of {self.selected_source} are turned {self.capture_turn} degrees "
                                f"where they are captured and {self.rotate} more by Base.rotate: turned twice. "
                                f"Set Base.rotate to 0 when the capture already turned the picture upright.")
        elif self.capture_turn:
            self.logger.info(f"The frames of {self.selected_source} are turned {self.capture_turn} degrees where "
                             f"they are captured: the intrinsics are turned with them, and the poses are reported "
                             f"in the camera's frame as the sensor gives it.")
        if self.capture_turn and self.camera_info.get('fisheye'):
            # the remap takes the frame as it comes, turned: K turned with it (D, radial, holds)
            K = turned_camera_matrix(self.camera_info['K'], self.res, self.capture_turn)
            size = turned_size(*self.res, self.capture_turn)
            map_1, map_2 = cv2.fisheye.initUndistortRectifyMap(K, self.camera_info['D'], np.eye(3), K, size,
                                                               cv2.CV_16SC2)
            self.camera_info.update({"map_1": map_1, "map_2": map_2})

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
        # the frame size the intrinsics were calibrated at (the entry's, else the one its principal
        # point gives): the detector scales them to the frames it gets (a 960x540 recording read
        # with 1920x1080 intrinsics places every tag twice as far)
        self.intrinsics = FrameIntrinsics(params, calibration_resolution(camera_config), camera=self.chosen_camera,
                                          logger=self.logger)
        camera_info = {"fisheye": fisheye, "params": params, "res": self.res,
                       "calibration_resolution": self.intrinsics.calibration_size}

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

        elif self.source == 'stream':
            from openmmla.utils.constants import get_stream_sources
            stream_sources = get_stream_sources(self.config)
            if not stream_sources:
                raise ValueError("No pullable stream (rtmp/rtsp/srt URL) found in the Streams config section.")
            for name, url in stream_sources:
                print(f"{available_source_idx} : Stream '{name}' ({url}) is available.")
                available_sources.append(url)
                available_source_idx += 1

        elif self.source == 'file':
            base_config = self.config.get('Base', {})

            # the entry names its file by its full path (Browse… on the Config
            # tab), or by a path below the project directory: that file's folder is
            # the one listed and synchronized over. A bare name is looked up in the
            # Base.file_dir of an older config, else in the project directory
            named = str(self._source_index or "")
            if named and not os.path.isabs(named) and os.path.dirname(named):
                named = self._source_index = os.path.join(self.project_dir, named)
            if os.path.isabs(named):
                file_dir = os.path.dirname(named)
            else:
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
        for opencv, a Streams entry's name for stream, a file/stream name for
        file/lsl)."""
        if not available_sources:
            return None
        if self.source == 'stream':
            # by name, and an error rather than the first stream when it names none
            from openmmla.utils.constants import resolve_stream_source
            name, url = resolve_stream_source(self.config, self._source_index)
            self.logger.info(f"Using stream '{name}': {url}")
            self.stream_name = name  # noted in the session with the url
            return url
        selected_source = select_source_by_index_or_name(self._source_index, available_sources)
        self.logger.info(f"Using video source {selected_source}")
        return selected_source

    def _resolve_base(self, base: str | None) -> dict:
        """The 'Bases' entry this base runs as.

        -b names it. Launched from the console (a session id given) and without -b, the only entry
        there is; when -b names no entry, or there are several to choose from, why is printed and
        the entry is picked here, as a base run by hand does.
        """
        if base:
            entry = get_base_by_id(self.config, base)
            if entry is not None:
                return entry
            if not self.launch_session_id:
                raise ValueError(f"Base '{base}' not found in config 'Bases'.")
            ids = ', '.join(str(b.get('id')) for b in get_bases(self.config)) or 'none'
            print(f"\nBase '{base}' is not in the config's Bases (there are: {ids}): pick another base on "
                  f"the IPS Base card, or pick one below.\n")
        elif self.launch_session_id:
            bases = get_bases(self.config)
            if len(bases) == 1:
                return bases[0]
            if bases:
                print(f"\nNo base was given and there are {len(bases)} in the config's Bases: pick the base "
                      f"on the IPS Base card, or pick one below.\n")
        return self._pick_base_interactively()

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
        # the session's logger goes as its own argument, never into stream_kwargs, which the session's
        # provenance stores as JSON: the stream's waits and drops are then in the session's log
        self.video_stream = VideoStream(source=self.source, log=self.logger, **self.stream_kwargs)
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
            tags, tag_relations, quality = self._process_single_frame(frame, acquired_time)
            
            # save frame if store is enabled
            if self.store:
                cv2.imwrite(os.path.join(save_path, f'{acquired_time}.jpg'), frame)
            
            # publish result
            message = {
                "base_id": self.base_id,
                "tags": tags,
                "quality": quality,
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
        """Process real-time video streams continuously (opencv, stream, lsl)."""
        print("Processing real-time streams...")
        runtime_root = pipeline_section_dir(self.project_dir, self.session_id, 'ips-base', 'real-time') / 'runtime'
        runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = os.fspath(runtime_root)
        save_path = os.path.join(self.runtime_dir, f'ips_{self.base_id}')
        os.makedirs(save_path, exist_ok=True)
        frames_count = 0

        while not self.stop_event.is_set():
            # a short timeout: while the stream is opened again after a drop no frame comes, and STOP is
            # heard within a second; one that does not come back raises StreamUnavailable, which ends the run
            frames = self.video_stream.read(timeout=1.0)
            if not frames:
                if self.graphics:
                    cv2.waitKey(1)  # the window stays responsive through the gap
                continue
            video_frame = frames[-1]  # read latest frame

            frame = video_frame.data
            frames_count += 1
            acquired_time = video_frame.timestamp

            # process the frame
            tags, tag_relations, quality = self._process_single_frame(frame, acquired_time)

            if self.store and frames_count % self.fps == 0:
                frames_count = 0
                cv2.imwrite(os.path.join(save_path, f'{acquired_time}.jpg'), frame)

            message = {
                "base_id": self.base_id,
                "tags": tags,
                "quality": quality,
                "tag_relations": tag_relations,
                "acquired_time": acquired_time
            }
            self.logger.debug(message)
            message_str = json.dumps(message)
            self.mqtt_client.publish(f'{self.session_id}/ips', message_str, qos=0, retain=False)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _process_single_frame(self, frame, acquired_time):
        """Detect the tags of a frame: (tags, tag_relations, quality), as the base publishes them.

        `tags` holds the raw pose of every detection, {tag id: [R, t]} in this camera's frame as
        the sensor gives it, however the picture was turned on the way (the capture's turn and
        Base.rotate), so that the Camera Sync matrices hold; nothing is carried over from earlier
        frames (R turned, when the detector put its z axis towards the camera, so that -column 2 is
        the outward normal: vector.canonical_rotation). `quality` holds each detection's
        {margin, err}: the decoder's decision margin and the pose's object-space error.
        `tag_relations` says who faces whom on this frame's raw poses, as the turned frame shows
        them (upright, its x-z plane the floor's). The window, when there is one, draws each tag's
        pose smoothed (Base.display_smoothing) on the turned frame.
        """
        fisheye = self.camera_info.get("fisheye", False)
        if fisheye:
            frame = cv2.remap(frame, self.camera_info["map_1"], self.camera_info["map_2"],
                              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # the intrinsics hold for the sensor's picture: scaled to its size (the frame's turned back
        # by the capture's turn), then turned with the frame (a fisheye frame is remapped with the
        # calibration's own K, so its intrinsics are not scaled)
        height, width = frame.shape[:2]
        size = tuple(self.res) if fisheye else sensor_size(width, height, self.capture_turn)
        params = self.camera_info["params"] if fisheye or self.intrinsics is None \
            else self.intrinsics.for_frame(*size)
        params = turned_intrinsics(params, size, self.turn)
        if self.rotate in ROTATIONS:
            frame = cv2.rotate(frame, ROTATIONS[self.rotate])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray, estimate_tag_pose=True, camera_params=params,
                                       tag_size=self.tag_size)

        detections = []
        for tag in results:
            if tag.decision_margin < MIN_DECISION_MARGIN or int(tag.tag_id) > self.max_badge_id:
                continue
            t = np.asarray(tag.pose_t, dtype=float).reshape(3, 1)
            detections.append((tag, canonical_rotation(tag.pose_R, t), t))

        tags = {}
        quality = {}
        tag_relations = {}
        for tag, R, t in detections:
            R_sensor, t_sensor = pose_in_sensor_frame(R, t, self.turn)
            tags[tag.tag_id] = [R_sensor.tolist(), t_sensor.tolist()]
            err = float(getattr(tag, 'pose_err', float('nan')))
            quality[tag.tag_id] = {'margin': round(float(tag.decision_margin), 2),
                                   'err': err if np.isfinite(err) else None}
            # who this tag faces on this frame alone: the synchronizer counts it over its bucket
            facing = tag_relations.setdefault(tag.tag_id, [])
            for other, other_R, other_t in detections:
                if other.tag_id != tag.tag_id and is_tag_looking_at_another_2d(
                        [R, t], [other_R, other_t], cosine_threshold=-0.94, distance_threshold=1):
                    facing.append(str(other.tag_id))

            # draw visualizations, from the smoothed pose
            if self.graphics:
                shown_R, _ = self.display_stabilizer.update(tag.tag_id, R, t, acquired_time)
                corners = np.int32(tag.corners)
                tag_center = np.mean(corners, axis=0)
                arrow_dir = outward_normal_2d(shown_R)[:2]
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

        return tags, tag_relations, quality

    def _load_transform_matrices(self):
        """Load the transformation matrices of this base's room: the file camera sync exported for
        the main base of its room, transformation_matrices_<main>.json (a config without rooms has
        one main), else, without rooms, the first file there is: another room's file holds other
        coordinates, so a base of a room whose main has none has no matrices. A base that can be
        the main camera alone (the main of its room, or the only Bases entry) needs none: its poses
        are the session's coordinates as they are, {} with itself as the main."""
        transformation_choices = [d for d in os.listdir(self.camera_sync_dir) if
                                  d.startswith('transformation_matrices_')]
        for idx, choice in enumerate(transformation_choices):
            print(f"{idx}: {choice}")

        # non-interactive: the file of this base's room, else the first one; main_id is its
        # suffix (e.g. transformation_matrices_m.json -> main camera id 'm')
        room_main_id = main_of_base(self.config, self._base_id_override)
        wanted = f'transformation_matrices_{room_main_id}.json' if room_main_id else None
        other_rooms = bool(wanted and wanted not in transformation_choices
                           and list(bases_by_room(self.config)) != [''])
        if not transformation_choices or other_rooms:
            if main_without_matrices(self.config, self._base_id_override):
                self.transform_matrices_file = None
                self.main_id = self._base_id_override
                self.logger.info(f"No transformation_matrices_{self.main_id}.json: base {self.main_id} is the main "
                                 f"camera alone, its poses the session's coordinates as they are.")
                return {}
            if other_rooms:
                self.logger.warning(f"There is no {wanted} for the main base {room_main_id} of this base's room; "
                                    f"the other files are other rooms' coordinates.")
            return None
        chosen_transformation = wanted if wanted in transformation_choices else sorted(transformation_choices)[0]
        self.transform_matrices_file = chosen_transformation
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
    
