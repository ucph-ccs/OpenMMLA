import gc
import inspect
import json
import logging
import os
import threading
import time

from openmmla.analytics.video.analyze import session_analysis_video
from openmmla.utils.client import InfluxDBClientWrapper
from openmmla.utils.client import MQTTClientWrapper
from openmmla.utils.client import RedisClientWrapper
from openmmla.utils.logger import get_logger
from openmmla.utils.threads import RaisingThread
from .input import get_function_synchronizer, get_bucket_name
from .vector import is_tag_looking_at_another_2d
from .transform import transform_point, transform_rotation


class Synchronizer:
    """Synchronizer class for synchronizing detection results from multiple cameras and uploading to InfluxDB"""
    logger = get_logger('synchronizer')

    def __init__(self, project_dir: str = None, config_path: str = None):
        """Initialize the synchronizer.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        # Set the project root directory
        if project_dir is None:
            caller_frame = inspect.stack()[1]
            caller_module = inspect.getmodule(caller_frame[0])
            if caller_module is None:
                self.project_dir = os.getcwd()
            else:
                self.project_dir = os.path.dirname(os.path.abspath(caller_module.__file__))
        else:
            self.project_dir = project_dir

        # Determine the configuration path
        if config_path:
            if os.path.isabs(config_path):
                self.config_path = config_path
            else:
                self.config_path = os.path.join(self.project_dir, config_path)
        else:
            self.config_path = os.path.join(self.project_dir, 'conf/video_base.ini')

        # Check if the configuration file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        self.logger_dir = os.path.join(self.project_dir, 'logger')
        self.camera_sync_dir = os.path.join(self.project_dir, 'camera_sync')
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.camera_sync_dir, exist_ok=True)

        self.main_camera_id = None
        self.transform_matrices_dict = None
        self.bucket_name = None
        self.graph_dict = None
        self.location_dict = None
        self.alive = False
        self.threads = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)

    def run(self):
        print('\033]0;Video Synchronizer\007')
        func_map = {1: self._start_synchronizing, 2: self._set_main_camera}

        while True:
            try:
                select_fun = get_function_synchronizer()
                if select_fun == 0:
                    self.logger.info("Exiting video synchronizer...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"During running the synchronizer, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)

    def _start_synchronizing(self):
        if self.transform_matrices_dict is None:
            self.logger.warning("Main camera id or transformation matrices not set, please set them first.")
            return self._set_main_camera()

        self.graph_dict = {}
        self.location_dict = {}
        self.bucket_name = get_bucket_name(self.influx_client)
        self.logger = get_logger(f'synchronizer-{self.bucket_name}',
                                 os.path.join(self.logger_dir, f'{self.bucket_name}_synchronizer.log'),
                                 level=logging.DEBUG, console_level=logging.INFO, file_level=logging.DEBUG)
        exception_occurred = None

        self._listen_for_start_signal()

        # Reinitialize MQTT client with new topic and on_message callback
        self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics=f'{self.bucket_name}/video')
        self.mqtt_client.loop_start()

        # Create threads
        self._create_thread(self._listen_for_stop_signal)
        self._create_thread(self._upload_data)

        # Start threads
        try:
            self._start_threads()
            self._join_threads()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning(
                f"During voice recognition, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}",
                exc_info=True)
            exception_occurred = e
        finally:
            self._synchronization_handler(exception_occurred)

    def _set_main_camera(self):
        self.transform_matrices_dict = self._load_transform_matrices()
        if self.transform_matrices_dict is None:
            self.logger.warning("No transformation matrices found, please check your main camera id or do the camera "
                                "sync first.")

    def _synchronization_handler(self, e):
        """Handle exceptions and stop all threads."""
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")

        self.mqtt_client.loop_stop()
        session_analysis_video(self.project_dir, self.bucket_name, self.influx_client)
        self._clean_up()

    def _create_thread(self, target, *args):
        """Create a new thread and add it to the thread list.

        :param target: the target function to run in the thread
        """
        t = RaisingThread(target=target, args=args)
        self.threads.append(t)

    def _start_threads(self):
        """Start all threads."""
        self.stop_event.clear()
        for t in self.threads:
            t.start()

    def _join_threads(self):
        """Wait for all threads to finish."""
        for t in self.threads:
            t.join()

    def _stop_threads(self):
        """Stop all threads and free memory."""
        self.stop_event.set()
        for t in self.threads:
            if threading.current_thread() != t:
                try:
                    t.join(timeout=5)
                except Exception as e:
                    self.logger.warning("During the thread stopping, catch exception %s", e, exc_info=True)
        self.threads.clear()

    def _clean_up(self):
        """Free memory by resetting dictionaries."""
        self.bucket_name = None
        self.graph_dict = None
        self.location_dict = None
        gc.collect()

    def _listen_for_start_signal(self):
        """Listen on the redis bucket control channel for the START signal."""
        p = self.redis_client.subscribe(f"{self.bucket_name}/control")
        self.logger.info("Wait for START signal...")
        while True:
            message = p.get_message()
            if message and message['data'] == b'START':
                self.logger.info("Received START signal, start synchronizing...")
                break

    def _listen_for_stop_signal(self):
        """Listen on the redis bucket control channel for the STOP signal."""
        p = self.redis_client.subscribe(f"{self.bucket_name}/control")
        self.logger.info("Listening for STOP command...")

        while not self.stop_event.is_set():
            message = p.get_message()
            if message and message['data'] == b'STOP':
                self.logger.info("Received STOP signal, stop synchronizing...")
                self.stop_event.set()

    def _handle_base_result(self, client, userdata, msg):
        with self.lock:
            if not self.stop_event.is_set():
                self.alive = True
                message = json.loads(msg.payload)
                sender_id = message["sender_id"]

                if sender_id.isnumeric():  # results from nicla vision's onboard apriltag detection
                    self.graph_dict.setdefault(sender_id, []).extend(message['detected_tags'])
                else:  # msg from base camera
                    tags = message["tags"]
                    tag_relations = message["tag_relations"]

                    if sender_id != self.main_camera_id:  # convert to main coordinates
                        R = self.transform_matrices_dict[sender_id]['R']
                        T = self.transform_matrices_dict[sender_id]['T']
                        for tag_id, tag_data in tags.items():
                            main_rotation = transform_rotation(R, tag_data[0])
                            main_translation = transform_point(tag_data[1], R, T)
                            self.location_dict[tag_id] = [main_rotation, main_translation]
                    else:
                        self.location_dict.update(tags)

                    # Store tag relations into graph
                    for tag_id, look_at_tags in tag_relations.items():
                        self.graph_dict.setdefault(tag_id, []).extend(look_at_tags)

                    # Detect tag relations again under main camera's coordinate system
                    for tag_id, tag_data in self.location_dict.items():
                        self.graph_dict.setdefault(tag_id, [])
                        for target_id, target_data in self.location_dict.items():
                            if target_id != tag_id and target_id not in self.graph_dict[tag_id]:
                                if is_tag_looking_at_another_2d(tag_data, target_data, cosine_threshold=-0.7, distance_threshold=1.2):
                                    self.graph_dict[tag_id].append(target_id)

    def _upload_data(self):
        """Upload data to InfluxDB for every 1 second"""
        while not self.stop_event.is_set():
            with self.lock:
                if self.alive:
                    segment_start_time = int(time.time()) - 1.0
                    rotations_dict = {}
                    translations_dict = {}
                    for tag_id, (rotation, translation) in self.location_dict.items():
                        rotations_dict[tag_id] = rotation
                        translations_dict[tag_id] = translation

                    # Prepare and upload the data for badge translations,rotations and relations
                    self._write_to_influx("badge translations", {
                        "segment_start_time": segment_start_time,
                        "translations": json.dumps(translations_dict),
                    })
                    self._write_to_influx("badge rotations", {
                        "segment_start_time": segment_start_time,
                        "rotations": json.dumps(rotations_dict),
                    })
                    self._write_to_influx("badge relations", {
                        "segment_start_time": segment_start_time,
                        "graph": json.dumps(self.graph_dict),
                    })

                # Reset for next cycle
                self.graph_dict.clear()
                self.location_dict.clear()
                self.alive = False

            # Schedule the next upload outside the lock to avoid potential deadlocks
            time.sleep(1)

    def _write_to_influx(self, measurement, fields):
        data = {
            "measurement": measurement,
            "fields": fields,
        }
        # self.logger.info(f"{data}")
        self.influx_client.write(self.bucket_name, record=data)

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
                    self.main_camera_id = chosen_transformation.removesuffix('.json').split('_')[-1]
                    self.logger.info(f"Main camera id: {self.main_camera_id} has been selected.")
                    break
            except ValueError:
                self.logger.warning("Please enter a valid number.")

        with open(os.path.join(self.camera_sync_dir, chosen_transformation), 'r') as file:
            return json.load(file)
