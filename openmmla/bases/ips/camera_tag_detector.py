import json

import cv2
import gc
import numpy as np
from pupil_apriltags import Detector

from openmmla.bases.base import Base
from openmmla.streams.video_stream import VideoStream
from openmmla.utils.client import MQTTClientWrapper
from openmmla.utils.logger import get_logger
from .enums import ROTATIONS
from .input import get_function_base
from .vector import get_outward_normal_vector


class CameraTagDetector(Base):
    """Class for detecting AprilTags from camera feed"""
    logger = get_logger('camera-tag-detector')

    def __init__(self, project_dir: str | None, config_path: str, max_badge_id: int = 15):
        """Initialize the camera tag detector.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            max_badge_id: maximum badge ID to detect (default: 15)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        """Camera detector parameters."""
        self.max_badge_id = max_badge_id
        # self.cameras_dir = os.path.join(self.project_dir, 'camera_calib/cameras')

        """Runtime attributes"""
        self.chosen_camera = None
        self.camera_info = {}
        self.selected_source = None
        self.camera_configured = False
        self.base_id = None
        self.video_stream = None

        self._setup_yaml()
        self._setup_objects()

    def _setup_yaml(self):
        """Set up attributes from YAML configuration."""
        base_config = self.config.get('Base', {})
        self.tag_size = float(base_config.get('tag_size', 0.061))
        self.families = base_config.get('families', 'tag36h11')
        self.res = tuple(base_config.get('resolution', (1920, 1080)))
        self.rotate = int(base_config.get('rotate', 0))
        self.fps = int(base_config.get('fps', 30))

        self.source = base_config['source']
        self.stream_kwargs = base_config['stream_kwargs']
        self.stream_kwargs['resolution'] = self.res
        self.stream_kwargs['fps'] = self.fps

    def _setup_objects(self):
        self.detector = Detector(families=self.families, nthreads=4)
        self.mqtt_client = MQTTClientWrapper(self.config_path)

    def _clean_up(self):
        """Clean up resources."""
        self.mqtt_client.loop_stop()
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None
        gc.collect()

    def run(self):
        """Run the camera tag detector."""
        print('\033]0;Camera Detector\007')
        func_map = {1: self._start_detection, 2: self._set_camera}

        while True:
            try:
                select_fun = get_function_base(self.chosen_camera, self.selected_source, self.base_id, main_id='None')
                if select_fun == 0:
                    self.logger.info("Exiting video base...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"During running the tag detector, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)
            finally:
                self._clean_up()

    def _start_detection(self):
        """Start AprilTag detection"""
        if not self.camera_configured:
            self.logger.warning("Camera is not configured.")
            return self._set_camera()

        # MQTT client reinitialization
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        # Start video stream
        self._configure_video_stream()

        try:
            self._process_frames()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning("%s, capture interrupted.", e, exc_info=False)
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    def _set_camera(self):
        """Set up camera source and id."""
        self.camera_configured = False

        self.camera_info = self._configure_camera_params()
        if self.camera_info is None:
            self.logger.warning("Camera configuration failed, please calibrate your camera first.")
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

        # Config camera base id
        self.base_id = input("Input your sender id, 'm' for main camera, and 'a', 'b', 'c', 'd' for alternatives "
                             "camera [m]: ")
        if not self.base_id:
            self.base_id = 'm'
        self.camera_configured = True
        print(f'\033]0;Camera Detector {self.base_id}\007')

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
            # Get RTMP URLs from configuration
            if 'RTMP' not in self.config:
                raise ValueError("RTMP configuration is missing in the YAML file.")
            if 'video_streams' not in self.config['RTMP']:
                raise ValueError("RTMP: video_streams configuration is missing in the YAML file.")
            for url in self.config['RTMP']['video_streams']:
                print(f"{available_source_idx} : RTMP stream {url} is available.")
                available_sources.append(url)
                available_source_idx += 1

        if not available_sources:
            self.logger.warning(f"No video sources found for {self.source}.")

        return available_sources

    def _choose_video_source(self, available_sources):
        """Choose a video source (camera index or RTMP URL)."""
        if not available_sources:
            return None
        default_source_idx = 0  # Default to the first available source
        while True:
            try:
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

    def _configure_video_stream(self):
        """Configure video stream."""
        self.video_stream = VideoStream(source=self.source, **self.stream_kwargs)
        self.video_stream.start()

    def _process_frames(self):
        print("Processing frames...")

        while True:
            video_frame = self.video_stream.read()[-1]
            frame = video_frame.data
            acquired_time = video_frame.timestamp

            if self.camera_info.get("fisheye", False):
                frame = cv2.remap(frame, self.camera_info["map_1"], self.camera_info["map_2"],
                                  interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if self.rotate in ROTATIONS:
                frame = cv2.rotate(frame, ROTATIONS[self.rotate])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_info["params"],
                                           tag_size=self.tag_size)

            tags = {}

            for tag in results:
                if tag.decision_margin < 10 or int(tag.tag_id) > self.max_badge_id:
                    continue

                corners = np.int32(tag.corners)
                normal, tag = get_outward_normal_vector(tag)
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
                print(f"Tag ID: {tag.tag_id}, Rotation: {tag.pose_R}, Translation: {tag.pose_t}")

            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow(f'AprilTags Detection from camera {self.base_id}', display_frame)

            message = {
                "base_id": self.base_id,
                "tags": tags,
                "acquired_time": acquired_time
            }
            message_str = json.dumps(message)
            self.mqtt_client.publish("camera/synchronize", message_str, qos=0, retain=False)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
