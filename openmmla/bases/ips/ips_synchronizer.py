import gc
import json
import logging
import os
import threading

from openmmla.bases.synchronizer import Synchronizer
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, shared_pipeline_artifact_dir
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_session, show_error_and_pause
from openmmla.utils.logger import get_logger
from .input import get_function_synchronizer
from .transform import transform_point, transform_rotation
from .vector import is_tag_looking_at_another_2d


class IPSSynchronizer(Synchronizer):
    """IPSSynchronizer class for synchronizing detection results from multiple cameras under a unified spatial
    coordinate and uploading them to InfluxDB"""
    logger = get_logger('ips-synchronizer')

    def __init__(self, project_dir: str | None, config_path: str, verbose: bool = False,
                 session_id: str | None = None):
        """Initialize the IPSSynchronizer class.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            verbose: whether to enable verbose logging (default: False)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.verbose = verbose
        self.launch_session_id = session_id

        # Runtime attributes
        self.main_id = None
        self.transform_matrices_dict = None
        self.merged_tags = None
        self.merged_relations = None
        self.session_id = None
        self.allowed_tag_ids = None
        self.time_bucket_key = None  # start timestamp of time bucket
        self.time_bucket_end = None
        self.alive = False

        # Threading attributes
        self.threads = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        self.bucket_duration = float(self.config['Synchronizer']['bucket_duration'])

    def _setup_directories(self):
        """Set up directories."""
        self.logger_dir = os.fspath(shared_pipeline_artifact_dir(self.project_dir, 'ips-base', 'logger'))
        self.camera_sync_dir = os.path.join(self.project_dir, 'camera_sync')
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.camera_sync_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.mongo_client = MongoDBClientWrapper(self.config_path)

    def _clean_up(self):
        """Clean up runtime variables and free memory."""
        if self.threads:
            self._stop_threads()
        self._clear_threads()
        self.mqtt_client.loop_stop()
        self.session_id = None
        self.allowed_tag_ids = None
        self.merged_relations = None
        self.merged_tags = None
        gc.collect()

    def run(self):
        """Run the IPS synchronizer."""
        print('\033]0;IPS Synchronizer\007')
        func_map = {1: self._start_synchronization, 2: self._set_main_camera, }

        while True:
            try:
                select_fun = get_function_synchronizer(self.main_id)
                if select_fun == 0:
                    self.logger.info("Exiting IPS synchronizer...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"During running the synchronizer, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)
                if not isinstance(e, KeyboardInterrupt):
                    show_error_and_pause(e, "return to the IPS Synchronizer menu")
            finally:
                self._clean_up()

    def _start_synchronization(self):
        """Start the synchronization process."""
        if self.transform_matrices_dict is None:
            self.logger.warning("Main camera id or transformation matrices not set, please set them first.")
            return self._set_main_camera()

        self.merged_relations = {}
        self.merged_tags = {}

        # select or create bucket
        self.session_id = self.launch_session_id or select_or_create_session(self.mongo_client)
        self._create_bucket_logger()
        self._resolve_session_tag_filter()
        self._listen_for_start_signal()

        # reinitialize mqtt client with new topics and on_message callback
        self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics=f'{self.session_id}/ips')
        self.mqtt_client.loop_start()

        # create threads
        self._create_thread(self._listen_for_stop_signal)

        # start threads and wait for them to finish
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

    def _create_bucket_logger(self):
        """Create logger for the bucket."""
        self.bucket_logger_dir = os.fspath(
            pipeline_section_dir(self.project_dir, self.session_id, 'ips-base', 'logger')
        )
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        copy_config_snapshot(self.config_path, self.project_dir, self.session_id, 'ips-base')
        self.logger = get_logger(f'ips-synchronizer-{self.session_id}',
                                 os.path.join(self.bucket_logger_dir, f'ips_synchronizer.log'),
                                 console_level=logging.DEBUG if self.verbose else logging.INFO)

    def _resolve_session_tag_filter(self):
        """Limit IPS aggregation to tag ids assigned to the selected session group."""
        self.allowed_tag_ids = None
        try:
            session_doc = self.mongo_client.get_session(self.session_id)
        except Exception as e:
            self.logger.warning(f"Could not load session document for IPS tag filtering: {e}")
            return
        if not session_doc:
            self.logger.warning(f"No MongoDB session document found for {self.session_id}; IPS will not filter tags by group.")
            return

        tag_ids = set()
        for participant in session_doc.get("participants", []) or []:
            if not isinstance(participant, dict):
                continue
            tag_id = participant.get("tag_id")
            if tag_id is None:
                continue
            text = str(tag_id).strip()
            if text:
                tag_ids.add(text)

        if tag_ids:
            self.allowed_tag_ids = tag_ids
            self.logger.info(f"IPS tag filter for {self.session_id}: {sorted(tag_ids)}")
        else:
            self.logger.warning(f"No participant tag ids found for {self.session_id}; IPS will not filter tags by group.")

    def _tag_allowed(self, tag_id) -> bool:
        return self.allowed_tag_ids is None or str(tag_id) in self.allowed_tag_ids

    def _filter_tags(self, tags: dict) -> dict:
        if self.allowed_tag_ids is None:
            return tags
        return {
            tag_id: tag_data
            for tag_id, tag_data in tags.items()
            if self._tag_allowed(tag_id)
        }

    def _filter_relations(self, relations: dict) -> dict:
        if self.allowed_tag_ids is None:
            return relations
        filtered = {}
        for tag_id, look_at_tags in relations.items():
            if not self._tag_allowed(tag_id):
                continue
            filtered[tag_id] = [
                target_id for target_id in look_at_tags
                if self._tag_allowed(target_id)
            ]
        return filtered

    def _synchronization_handler(self, e: Exception | KeyboardInterrupt | None):
        """Handle exceptions and stop all threads.

        Args:
            e: the exception that occurred during the synchronization process
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")
        self._clean_up()

    def _set_main_camera(self):
        """Set the main camera id and load transformation matrices."""
        self.transform_matrices_dict = self._load_transform_matrices()
        if self.transform_matrices_dict is None:
            self.logger.warning("No transformation matrices found, please check your main camera id or do the camera "
                                "sync first.")

    def _handle_base_result(self, client, userdata, msg):
        """Handle the received base result.

        Args:
            client: the client instance for this callback
            userdata: the private user data as a set in Client() or user_data_set()
            message: an instance of MQTTMessage
        """
        if not self.stop_event.is_set():
            self.alive = True
            base_result = json.loads(msg.payload)
            base_id = base_result["base_id"]
            base_result_time = float(base_result["acquired_time"])

            # unified timing logic for both file and real-time modes
            if self.time_bucket_key is None:
                self.time_bucket_key = base_result_time
                self.time_bucket_end = self.time_bucket_key + self.bucket_duration
                self.logger.info(
                    f"Initialized time bucket: {self.time_bucket_key:.2f}s - {self.time_bucket_end:.2f}s")

            # check if we need to upload current bucket and advance to next
            elif base_result_time >= self.time_bucket_end:
                self._upload_current_bucket()
                self.time_bucket_key = base_result_time
                self.time_bucket_end = self.time_bucket_key + self.bucket_duration
                self.logger.info(
                    f"Advanced to time bucket: {self.time_bucket_key:.2f}s - {self.time_bucket_end:.2f}s)")

            # check if result falls within current bucket
            valid = self.time_bucket_key <= base_result_time < self.time_bucket_end
            if valid:
                if base_id.isnumeric() and int(base_id) > 50000:  # msg from nicla vision's onboard apriltag detection (if used)
                    if not self._tag_allowed(base_id):
                        return
                    detected_tags = [
                        tag_id for tag_id in base_result['detected_tags']
                        if self._tag_allowed(tag_id)
                    ]
                    if base_id not in self.merged_relations:
                        self.merged_relations[base_id] = set()
                    self.merged_relations[base_id].update(detected_tags)
                else:  # msg from environmental camera
                    tags = self._filter_tags(base_result["tags"])
                    tag_relations = self._filter_relations(base_result["tag_relations"])

                    if base_id != self.main_id:  # convert to main coordinates
                        R = self.transform_matrices_dict[base_id]['R']
                        T = self.transform_matrices_dict[base_id]['T']
                        for tag_id, tag_data in tags.items():
                            main_rotation = transform_rotation(R, tag_data[0])
                            main_translation = transform_point(tag_data[1], R, T)
                            self.merged_tags[tag_id] = [main_rotation, main_translation]
                    else:
                        self.merged_tags.update(tags)

                    # store tag relations into graph (avoid duplicates)
                    for tag_id, look_at_tags in tag_relations.items():
                        if tag_id not in self.merged_relations:
                            self.merged_relations[tag_id] = set()
                        self.merged_relations[tag_id].update(look_at_tags)

                    # detect tag relations again under main camera's coordinate system
                    for tag_id, tag_data in self.merged_tags.items():
                        if tag_id not in self.merged_relations:
                            self.merged_relations[tag_id] = set()
                        for target_id, target_data in self.merged_tags.items():
                            if target_id != tag_id and target_id not in self.merged_relations[tag_id]:
                                if is_tag_looking_at_another_2d(tag_data, target_data, cosine_threshold=-0.94,
                                                                distance_threshold=1.2):
                                    self.merged_relations[tag_id].add(target_id)

    def _upload_current_bucket(self):
        """Upload the current time bucket's aggregated results for both file and real-time modes."""
        from openmmla.utils.constants import (
            EVENT_TYPE_IPS_TRANSLATION, EVENT_TYPE_IPS_ROTATION, EVENT_TYPE_IPS_RELATION,
        )

        rotations_dict = {}
        translations_dict = {}
        for tag_id, (rotation, translation) in self.merged_tags.items():
            rotations_dict[tag_id] = rotation
            translations_dict[tag_id] = translation

        translation_fields = {
            "window_start_time": self.time_bucket_key,
            "window_end_time": self.time_bucket_end,
            "translations": json.dumps(translations_dict),
        }
        rotation_fields = {
            "window_start_time": self.time_bucket_key,
            "window_end_time": self.time_bucket_end,
            "rotations": json.dumps(rotations_dict),
        }

        relations_dict = {}
        for tag_id, relations in self.merged_relations.items():
            relations_dict[tag_id] = list(relations)

        relation_fields = {
            "window_start_time": self.time_bucket_key,
            "window_end_time": self.time_bucket_end,
            "graph": json.dumps(relations_dict),
        }

        self.logger.debug(translation_fields)
        self.logger.debug(rotation_fields)
        self.logger.debug(relation_fields)

        self.influx_client.write_event(self.session_id, EVENT_TYPE_IPS_TRANSLATION, translation_fields)
        self.influx_client.write_event(self.session_id, EVENT_TYPE_IPS_ROTATION, rotation_fields)
        self.influx_client.write_event(self.session_id, EVENT_TYPE_IPS_RELATION, relation_fields)

        self.logger.info(f"Uploaded bucket: {self.time_bucket_key:.2f}s - {self.time_bucket_end:.2f}s "
                         f"({len(self.merged_tags)} tags, {len(self.merged_relations)} relations)")

        # reset for next cycle
        self.merged_relations.clear()
        self.merged_tags.clear()

    def _load_transform_matrices(self):
        """Load transformation matrices."""
        transformation_choices = [d for d in os.listdir(self.camera_sync_dir) if
                                  d.startswith('transformation_matrices_')]
        for idx, choice in enumerate(transformation_choices):
            print(f"{idx}: {choice}")

        if not transformation_choices:
            return None

        default_selection = 0  # default to the first transformation matrix
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

    @property
    def session_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current session_id."""
        if self.session_id:
            return f'{self.session_id}/ips/control'
        return None
