import json

import cv2
import gc
import numpy as np
from pupil_apriltags import Detector

from openmmla.bases.base import Base
from openmmla.streams.video_stream import VideoStream
from openmmla.utils.client import MQTTClientWrapper
from openmmla.utils.input import show_error_and_pause
from openmmla.utils.logger import get_logger
from .enums import ROTATIONS
from .input import get_base_by_id, get_bases, select_source_by_index_or_name
from .vector import get_outward_normal_vector


class CameraTagDetector(Base):
    """Class for detecting AprilTags from camera feed"""
    logger = get_logger('camera-tag-detector')

    def __init__(self, project_dir: str | None, config_path: str, max_badge_id: int = 15,
                 graphics: bool = True, headless: bool = False, base: str | None = None):
        """Initialize the camera tag detector.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            max_badge_id: maximum badge ID to detect (default: 15)
            graphics: whether to display annotated video frames in a window.
                Set False for headless/remote runs; detection/MQTT unaffected.
            headless: no display (forces graphics off).
            base: which base id from the config 'Bases' list to run; if omitted,
                pick one interactively.
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        """Camera detector parameters."""
        self.max_badge_id = max_badge_id
        # display is independent of selection; headless implies no display
        self.graphics = graphics and not headless
        # profile-driven: every run loads a base from the config 'Bases' list
        # (single source of truth) — by id if given, else picked interactively.
        entry = get_base_by_id(self.config, base) if base else self._pick_base_interactively()
        if entry is None:
            raise ValueError(f"Base '{base}' not found in config 'Bases'.")
        self._camera_name = entry.get('camera')
        # source_index is overloaded by source type: an index (opencv/rtmp), a
        # file name (file) or a stream name (lsl) — keep it raw and interpret it
        # when the source is known.
        self._source_index = entry.get('source_index')
        self._base_id_override = str(entry.get('id', base))
        self._base_source = entry.get('source')  # per-base override of Base.source

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

        # per-base source (from the Bases entry) overrides the global Base.source
        self.source = self._base_source or base_config['source']
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

    def _pick_base_interactively(self):
        """Pick a base from the config 'Bases' list (the only interaction)."""
        bases = get_bases(self.config)
        if not bases:
            raise ValueError(
                "No bases defined. Add entries under 'Bases' in config.yml "
                "(each with id, camera and source_index).")
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

    def run(self):
        """Run the camera tag detector — fully profile-driven (no menus)."""
        print('\033]0;Camera Detector\007')
        try:
            self._set_camera()
            if self.camera_configured:
                self._start_detection()
            else:
                self.logger.error("Camera setup failed (no camera/source resolved from config).")
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning(
                f"Tag detector stopped: "
                f"{'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}",
                exc_info=not isinstance(e, KeyboardInterrupt))
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
            if self.graphics:
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
        elif self.source == 'lsl':
            self.stream_kwargs['lsl_name'] = self.selected_source

        # base id comes from the selected base entry
        self.base_id = str(self._base_id_override) if self._base_id_override else '1'
        self.camera_configured = True
        print(f'\033]0;Camera Detector {self.base_id}\007')

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

    def _choose_video_source(self, available_sources):
        """Pick the video source for the selected base (source_index is an index
        for opencv/rtmp, or a file/stream name for file/lsl)."""
        if not available_sources:
            return None
        selected_source = select_source_by_index_or_name(self._source_index, available_sources)
        self.logger.info(f"Using video source {selected_source}")
        return selected_source

    def _configure_video_stream(self):
        """Configure video stream."""
        self.stream_kwargs['project_dir'] = self.project_dir
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

            if self.graphics:
                display_frame = cv2.resize(frame, (960, 540))
                cv2.imshow(f'AprilTags Detection from camera {self.base_id}', display_frame)

            message = {
                "base_id": self.base_id,
                "tags": tags,
                "acquired_time": acquired_time
            }
            message_str = json.dumps(message)
            self.mqtt_client.publish("camera/synchronize", message_str, qos=0, retain=False)

            if self.graphics:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
