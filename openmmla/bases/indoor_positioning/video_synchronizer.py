import gc
import json
import logging
import os
import threading
import time

from openmmla.analytics.video.analyze import session_analysis_video
from openmmla.bases.synchronizer import Synchronizer
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.logger import get_logger
from .input import get_function_synchronizer, get_bucket_name
from .transform import transform_point, transform_rotation
from .vector import is_tag_looking_at_another_2d


class VideoSynchronizer(Synchronizer):
    """Synchronizer class for synchronizing detection results from multiple cameras and uploading to InfluxDB"""
    logger = get_logger('synchronizer')

    def __init__(self, project_dir: str, config_path: str):
        """Initialize the synchronizer.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir, config_path)

        """Runtime attributes."""
        self.threads = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.bucket_name = None
        self.transformation_id = None
        self.transform_matrices_dict = None
        self.graph_dict = None
        self.location_dict = None
        self.alive = False

        self._setup_directories()
        self._setup_objects()

    def _setup_directories(self):
        """Set up directories."""
        self.logger_dir = os.path.join(self.project_dir, 'logger')
        self.camera_sync_dir = os.path.join(self.project_dir, 'camera_sync')
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.camera_sync_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.redis_client = RedisClientWrapper(self.config_path)  # Redis wrapped client
        self.mqtt_client = MQTTClientWrapper(self.config_path)  # MQTT wrapped client
        self.influx_client = InfluxDBClientWrapper(self.config_path)  # InfluxDB wrapped client

    def _clean_up(self):
        """Free memory by resetting dictionaries."""
        self.bucket_name = None
        self.graph_dict = None
        self.location_dict = None
        gc.collect()

    def run(self):
        """Main menu for video synchronizer."""
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
        """Start the synchronization process."""
        if self.transform_matrices_dict is None:
            self.logger.warning("Main camera id or transformation matrices not set, please set them first.")
            return self._set_main_camera()

        self.graph_dict = {}
        self.location_dict = {}
        self.bucket_name = get_bucket_name(self.influx_client)
        self.logger = get_logger(f'synchronizer-{self.bucket_name}',
                                 os.path.join(self.logger_dir, f'{self.bucket_name}_synchronizer.log'),
                                 level=logging.DEBUG, console_level=logging.INFO, file_level=logging.DEBUG)

        self._listen_for_start_signal()

        # Reinitialize MQTT client with new topics and on_message callback
        self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics=f'{self.bucket_name}/video')
        self.mqtt_client.loop_start()

        # Create threads
        self._create_thread(self._listen_for_stop_signal)
        self._create_thread(self._upload_merged_result)

        # Start threads
        exception_occurred = None
        try:
            self._start_threads()
            self._join_threads()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning(
                f"During synchronization, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}",
                exc_info=True)
            exception_occurred = e
        finally:
            self._synchronization_handler(exception_occurred)

    def _set_main_camera(self):
        """Set the main camera id and load transformation matrices."""
        self.transform_matrices_dict = self._load_transform_matrices()
        if self.transform_matrices_dict is None:
            self.logger.warning("No transformation matrices found, please check your main camera id or do the camera "
                                "sync first.")

    def _synchronization_handler(self, e):
        """Handle exceptions and stop all threads.

        Args:
            e: the exception that occurred during the synchronization process
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")

        self.mqtt_client.loop_stop()
        session_analysis_video(self.project_dir, self.bucket_name, self.influx_client)
        self._clean_up()

    def _handle_base_result(self, client, userdata, msg):
        """Handle the received base result.

        Args:
            client: the client instance for this callback
            userdata: the private user data as a set in Client() or user_data_set()
            message: an instance of MQTTMessage
        """
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

                    if sender_id != self.transformation_id:  # convert to main coordinates
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
                                if is_tag_looking_at_another_2d(tag_data, target_data, cosine_threshold=-0.7,
                                                                distance_threshold=1.2):
                                    self.graph_dict[tag_id].append(target_id)

    def _upload_merged_result(self):
        """Log and upload merge segment result to InfluxDB"""
        while not self.stop_event.is_set():
            with self.lock:
                if self.alive:
                    segment_start_time = int(time.time()) - 1.0
                    rotations_dict = {}
                    translations_dict = {}
                    for tag_id, (rotation, translation) in self.location_dict.items():
                        rotations_dict[tag_id] = rotation
                        translations_dict[tag_id] = translation

                    # Prepare and upload the data for badge translations, rotations and relations
                    translation_data = {
                        "measurement": "badge translations",
                        "fields": {
                            "segment_start_time": segment_start_time,
                            "translations": json.dumps(translations_dict),
                        }
                    }

                    rotation_data = {
                        "measurement": "badge rotations",
                        "fields": {
                            "segment_start_time": segment_start_time,
                            "rotations": json.dumps(rotations_dict),
                        }
                    }

                    relation_data = {
                        "measurement": "badge relations",
                        "fields": {
                            "segment_start_time": segment_start_time,
                            "graph": json.dumps(self.graph_dict),
                        }
                    }

                    self.influx_client.write(self.bucket_name, translation_data)
                    self.influx_client.write(self.bucket_name, rotation_data)
                    self.influx_client.write(self.bucket_name, relation_data)

                # Reset for next cycle
                self.graph_dict.clear()
                self.location_dict.clear()
                self.alive = False

            # Schedule the next upload outside the lock to avoid potential deadlocks
            time.sleep(1)

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
                    self.transformation_id = chosen_transformation.split('_')[-1].split('.')[0]
                    break
            except ValueError:
                self.logger.warning("Please enter a valid number or press Enter for default.")

        with open(os.path.join(self.camera_sync_dir, chosen_transformation), 'r') as file:
            return json.load(file)
