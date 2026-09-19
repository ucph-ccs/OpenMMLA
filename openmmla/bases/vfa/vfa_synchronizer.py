import gc
import json
import os
import queue
import threading

from typing import Any

from openmmla.bases.synchronizer import Synchronizer
from openmmla.services.vfa.requests import request_multi_angle_frame_analyze
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, runtime_pipeline_artifact_dir
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.config import get_bases
from openmmla.utils.input import select_or_create_session, get_number_of_bases, pause_after_error, show_error_and_pause
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import build_service_url
from openmmla.utils.sync_strategy import TimeBucketSynchronizer, SyncStrategy
from .enums import BLUE, ENDC
from .input import get_function_synchronizer


class VFASynchronizer(Synchronizer):
    """VFASynchronizer class for synchronizing video frames from multiple angles."""
    logger = get_logger('vfa-synchronizer')

    def __init__(self, project_dir: str | None, config_path: str, session_id: str | None = None,
                 num_bases: int | None = None):
        """Initialize the VFASynchronizer class.
        
        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            session_id: the session to synchronize. Given, the synchronizer was
                launched from the console: it runs that session at once, with
                no menu, and returns when the run ends so the process exits.
            num_bases: how many bases to synchronize; if omitted, it is asked
                (from the menu) or, launched from the console, the number of
                entries in the config's 'Bases' list.
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.launch_session_id = session_id
        self.launch_num_bases = num_bases

        # Runtime attributes
        self.threads = []
        self.stop_event = threading.Event()
        self.session_id = None
        self.number_of_bases = None
        self.latest_time = None
        self.time_bucket_buffer = {}  # Buffer for {time_bucket_key: {base_id: {<angle>, <path>, <base_result_time>}}}
        self.selected_participant_descriptions = None  # Selected participant descriptions for current session

        # VLLM request queue and processing thread
        self.vllm_queue = queue.Queue()
        self.vllm_processing_thread = None

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        """Load configuration parameters."""
        sync_config = self.config['Synchronizer']
        vfa_server_config = self.config['Server']['vfa']
        base_config = self.config['Base']

        self.buffer_expiry_time = float(sync_config.get('result_expiry_time', 30))
        self.match_tolerance = float(sync_config.get('match_tolerance', 0.5))
        self.vllm_frame_analyzer_url = build_service_url(self.config, vfa_server_config['vllm_frame_analyzer'])
        self.angle_config = base_config.get('angle_config', {})
        
        self.logger.info(f"Loaded angle configurations: {list(self.angle_config.keys()) if self.angle_config else 'None'}")

    def _setup_directories(self):
        """Set up required directories."""
        self.logger_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'vfa-base', 'logger'))
        self.temp_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'vfa-base', 'real-time', 'temp'))
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

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
        self.number_of_bases = None
        self.latest_time = None
        self.time_bucket_buffer = {}
        self.selected_participant_descriptions = None
        self.vllm_queue = queue.Queue()
        gc.collect()

    def _reinit(self):
        """Reload the config, keeping the session and the number of bases the
        synchronizer was launched with (the base class would drop them)."""
        self.logger.info("Starting synchronizer reinitialization...")
        project_dir, config_path = self.project_dir, self.config_path
        session_id, num_bases = self.launch_session_id, self.launch_num_bases
        self._clean_up()
        self.__init__(project_dir=project_dir, config_path=config_path, session_id=session_id,
                      num_bases=num_bases)
        self.logger.info("Synchronizer reinitialization completed successfully")

    def _close_clients(self):
        """Close the broker and database connections on the way out."""
        for name, close in (('mqtt_client', 'disconnect'), ('redis_client', 'close'),
                            ('influx_client', 'close'), ('mongo_client', 'close')):
            client = getattr(self, name, None)
            if client is None:
                continue
            try:
                getattr(client, close)()
            except Exception as e:
                self.logger.debug(f"Closing the {name} failed: {e}")

    def run(self):
        """Run the VFA synchronizer.

        Launched from the console (with a session id) it runs that session at
        once and returns when the run ends, so the process exits. When that
        run cannot start, it says why and falls back to the menu below, where
        the cause can be fixed and the run started again."""
        print('\033]0;VFA Synchronizer\007')
        if self.launch_session_id and self._run_from_console():
            self._close_clients()
            return

        func_map = {1: self._start_synchronization, 2: self._reinit}

        while True:
            try:
                select_fun = get_function_synchronizer()
                if select_fun == 0:
                    print("------------------------------------------------")
                    clear_directory(os.path.join(self.temp_dir))
                    self.logger.info("Exiting VFA synchronizer...")
                    break
                ended = func_map.get(select_fun, lambda: print("Invalid option."))()
                if select_fun == 1 and ended is True and self.launch_session_id:
                    # launched from the console, it came to this menu to fix something: a run
                    # started from here ends the process on STOP all the same
                    self.logger.info(f"The run of session {self.launch_session_id} ended; exiting VFA synchronizer.")
                    break
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"\nDuring running synchronizer, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)
                if not isinstance(e, KeyboardInterrupt):
                    show_error_and_pause(e, "return to the VFA Synchronizer menu")
            finally:
                self._clean_up()
        self._close_clients()

    def _run_from_console(self) -> bool:
        """Run the launch session once, at once, with no menu and no question:
        the way the console starts the synchronizer.

        Returns:
            True when the run ended (STOP on the session's control channel,
            also before START, or Ctrl+C), so the process exits; False when it
            could not start or ended on an error, after saying why and what to
            do, so run() falls back to the menu.
        """
        session_id = self.launch_session_id
        try:
            number_of_bases = self._console_number_of_bases()
        except ValueError as e:
            self.logger.error(f"The VFA synchronizer cannot start session {session_id}: {e}")
            print("Falling back to the VFA Synchronizer menu: fix the above, then choose 1 to start.")
            return False

        self.logger.info(f"Synchronizing {number_of_bases} base(s) of session {session_id}; "
                         f"the synchronizer exits when the session is stopped.")
        try:
            ended = self._start_synchronization(number_of_bases=number_of_bases)
        except KeyboardInterrupt:
            self.logger.info(f"Interrupted; leaving session {session_id}.")
            ended = True
        except Exception as e:
            self.logger.warning(f"The VFA synchronizer could not run session {session_id}: {e}", exc_info=True)
            print(f"The VFA synchronizer could not run session {session_id}: {e}\n"
                  f"Check that Redis, MQTT and MongoDB are running (System Services on the console) and that "
                  f"the config is right, then choose 1 in the menu to start again.")
            pause_after_error("open the VFA Synchronizer menu")
            return False
        finally:
            self._clean_up()

        if not ended:
            print(f"The run of session {session_id} stopped on an error (see above), not on STOP.\n"
                  f"Check that Redis, MQTT and the VFA server are reachable (System Services on the console), "
                  f"then choose 1 in the menu below to start again.")
            return False

        print("------------------------------------------------")
        clear_directory(self.temp_dir)
        self.logger.info(f"The run of session {session_id} ended; exiting VFA synchronizer.")
        return True

    def _console_number_of_bases(self) -> int:
        """How many bases a run launched from the console synchronizes: -nb
        when it was given, else the number of entries in the config's 'Bases'
        list.

        Raises:
            ValueError: in plain words, when neither gives a number to use.
        """
        if self.launch_num_bases is not None:
            if self.launch_num_bases < 1:
                raise ValueError(
                    f"-nb/--num_bases is {self.launch_num_bases}, but at least 1 base is needed. "
                    f"Set Num Bases on the VFA Base card to 1 or more.")
            return self.launch_num_bases
        count = len(get_bases(self.config))
        if not count:
            raise ValueError(
                "-nb/--num_bases was not given and the config's 'Bases' list is empty, so the number of bases "
                "to synchronize is unknown. Add the bases under 'Bases' in config.yml (the Config tab of the "
                "VFA Base card), or pass -nb.")
        return count

    def _start_synchronization(self, number_of_bases: int | None = None) -> bool:
        """Start the synchronization process.

        Args:
            number_of_bases: how many bases to synchronize; if omitted, -nb
                when it was given, else asked.

        Returns:
            True when the run ended with STOP (also STOP before START, when
            nothing was synchronized) or Ctrl+C; False when it ended on an
            error, such as the connection to Redis lost.
        """
        # reset attributes
        self.latest_time = 0
        self.time_bucket_buffer = {}

        # bucket selection
        self.session_id = self.launch_session_id or select_or_create_session(self.mongo_client)
        if number_of_bases is None:
            has_launch_number = self.launch_num_bases is not None and self.launch_num_bases > 0
            number_of_bases = self.launch_num_bases if has_launch_number else get_number_of_bases()
        self.number_of_bases = number_of_bases
        real_time_dir = pipeline_section_dir(self.project_dir, self.session_id, 'vfa-base', 'real-time')
        self.temp_dir = os.fspath(real_time_dir / 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # resolve participant descriptions from experiment assignments
        from openmmla.utils.experiments import get_participant_descriptions
        session_doc = self.mongo_client.get_session(self.session_id)
        if session_doc:
            exp_id = session_doc.get("experiment_id", "")
            group_id = session_doc.get("group_id", "")
            session_participants = session_doc.get("participants", []) or []
            self.selected_participant_descriptions = {
                str(participant.get("tag_id")): participant.get("description")
                for participant in session_participants
                if isinstance(participant, dict) and participant.get("tag_id") is not None and participant.get("description")
            }
            if not self.selected_participant_descriptions:
                self.selected_participant_descriptions = get_participant_descriptions(exp_id, group_id)
            if self.selected_participant_descriptions:
                self.logger.info(f"Resolved participant descriptions for {exp_id}/{group_id}: {self.selected_participant_descriptions}")
            else:
                self.logger.warning(f"No participant descriptions found for {exp_id}/{group_id}")
        else:
            self.logger.warning(f"Could not retrieve session document for {self.session_id}")
            self.selected_participant_descriptions = None
        
        self._create_bucket_logger()

        # listen for start signal
        if not self._listen_for_start_signal():
            # STOP came before START: nothing was started
            self._clean_up()
            return True

        # reinitialize mqtt client with a new topic and on_message callback
        self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics=f'{self.session_id}/vfa')
        self.mqtt_client.loop_start()

        # create threads
        self._create_thread(self._listen_for_stop_signal)
        self._create_thread(self._process_vllm_requests)  # add vllm processing thread

        # start and join threads, handling exceptions if they occur
        exception_occurred = None
        try:
            self._start_threads()
            self._join_threads()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning(
                f"\nDuring synchronization, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}",
                exc_info=True)
            exception_occurred = e
        finally:
            self._synchronization_handler(exception_occurred)
        # Ctrl+C ends the run as STOP does; any other exception is an error
        return exception_occurred is None or isinstance(exception_occurred, KeyboardInterrupt)

    def _create_bucket_logger(self):
        self.bucket_logger_dir = os.fspath(
            pipeline_section_dir(self.project_dir, self.session_id, 'vfa-base', 'logger')
        )
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        copy_config_snapshot(self.config_path, self.project_dir, self.session_id, 'vfa-base')
        self.logger = get_logger(f'vfa-synchronizer-{self.session_id}',
                                 os.path.join(self.bucket_logger_dir, f'vfa_synchronizer.log'))

    def _handle_base_result(self, client, userdata, message):
        """Handle received frame from a VFA base."""
        try:
            base_result = json.loads(message.payload.decode('utf-8'))
            base_result_time = float(base_result['acquired_time'])
            base_id = base_result['base_id']
            angle = base_result['angle']
            image_path = base_result['image_path']
            store_frames = bool(base_result.get('store_frames', True))

            if not os.path.exists(image_path):
                self.logger.warning(f"Image path does not exist: {image_path}")
                return

            # Clean up expired frame sets
            expired_times = TimeBucketSynchronizer.get_expired_buckets(
                base_result_time,
                self.time_bucket_buffer,
                self.buffer_expiry_time
            )

            for t in expired_times:
                frame_set = self.time_bucket_buffer[t]
                if len(frame_set) >= 2:  # If there are at least two frames, try to process
                    self.logger.info(
                        f"Processing incomplete frame set at {t} with {len(frame_set)}/{self.number_of_bases} frames")
                    # Add to VLLM queue instead of processing directly
                    self.vllm_queue.put({
                        'time_bucket_key': t,
                        'frames': frame_set
                    })
                else:
                    self.logger.warning(f"Dropping expired frame set at {t} with only {len(frame_set)} frame(s)")
                    self._cleanup_unstored_frames(frame_set)
                del self.time_bucket_buffer[t]

            # Find the closest time bucket using the utility class
            closest_time = TimeBucketSynchronizer.find_closest_time_bucket(
                current_time=base_result_time,
                time_buckets=self.time_bucket_buffer,
                base_id=base_id,
                match_tolerance=self.match_tolerance,
                strategy=SyncStrategy.NEAREST  # Using nearest strategy for VFA
            )

            if not closest_time:
                closest_time = base_result_time  # Create a new time bucket starting at base_result_time
                self.time_bucket_buffer[closest_time] = {}

            # Store frame info
            if base_id in self.time_bucket_buffer[closest_time]:
                self.logger.debug(f"Overwriting frame for base {base_id} at time bucket {closest_time}")
                self._cleanup_unstored_frames({
                    base_id: self.time_bucket_buffer[closest_time][base_id]
                })

            self.time_bucket_buffer[closest_time][base_id] = {
                'angle': angle,
                'path': image_path,
                'base_result_time': base_result_time,
                'store_frames': store_frames
            }

            # Check if we've received frames from all cameras for this time bucket
            if len(self.time_bucket_buffer[closest_time]) == self.number_of_bases:
                # Add to VLLM queue instead of processing directly
                self.vllm_queue.put({
                    'time_bucket_key': closest_time,  # closest_time is the time_bucket_key (start time of the bucket)
                    'frames': self.time_bucket_buffer[closest_time]
                })
                del self.time_bucket_buffer[closest_time]

        except Exception as e:
            self.logger.error(f"Error handling frame: {e}", exc_info=True)

    def _synchronization_handler(self, e: Exception | KeyboardInterrupt | None):
        """Handle exceptions and stop all threads.

        Args:
            e: the exception that occurred during the synchronization process
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped.")
    
        if self.vllm_queue.empty():
            self.logger.info("VLLM queue processing completed")
        else:
            self.logger.warning(f"VLLM queue still has {self.vllm_queue.qsize()} items")
        
        clear_directory(self.temp_dir)
        self._clean_up()

    def _process_vllm_requests(self):
        """Process VLLM requests from the queue."""
        while not self.stop_event.is_set():
            try:
                # get a frame set from the queue with a timeout
                frame_set = self.vllm_queue.get(timeout=1.0)
                frames = {}
                
                try:
                    time_bucket_key = frame_set['time_bucket_key']  # the start time of this time bucket
                    frames = frame_set['frames']

                    # prepare data for multi-angle analysis
                    image_paths = []
                    angles = []
                    angle_descriptions = []

                    for base_id, frame_info in sorted(frames.items()):
                        if os.path.exists(frame_info['path']):
                            angle = frame_info['angle']
                            image_paths.append(frame_info['path'])
                            angles.append(angle)
                            
                            # get angle description from config, or create generic one if not found
                            angle_desc = self.angle_config.get(angle, f"Image from {angle} perspective")
                            angle_descriptions.append(angle_desc)
                        else:
                            self.logger.warning(f"Image path no longer exists: {frame_info['path']}")

                    if not image_paths:
                        self.logger.warning(f"No valid images found for time bucket {time_bucket_key}")
                    else:
                        try:
                            # request analysis from VLLM server
                            self.logger.info(f"Requesting multi-angle frame analysis for time bucket {time_bucket_key}: "
                                           f"{len(image_paths)} images from angles {angles} -> {self.vllm_frame_analyzer_url}")
                            
                            result = request_multi_angle_frame_analyze(
                                image_paths=image_paths,
                                angles=angles,
                                angle_descriptions=angle_descriptions,
                                session_id=self.session_id,
                                url=self.vllm_frame_analyzer_url,
                                participant_descriptions=self.selected_participant_descriptions,
                            )
                            
                            if result:
                                self.logger.info(f"Successfully received analysis result for time bucket {time_bucket_key}")
                                self._upload_result(time_bucket_key, result)
                            else:
                                self.logger.warning(f"Received null/empty analysis result for time bucket {time_bucket_key}")

                        except Exception as e:
                            self.logger.error(f"Error processing frame set: {e}", exc_info=True)
                
                finally:
                    if frames:
                        self._cleanup_unstored_frames(frames)
                    # if not stopped, mark the task as done since vllm_queue is still existing
                    if not self.stop_event.is_set():
                        self.vllm_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in VLLM processing thread: {e}", exc_info=True)

    def _cleanup_unstored_frames(self, frames: dict[str, dict[str, Any]]) -> None:
        """Remove temporary frames when bases did not request persistent frame storage."""
        temp_dir = os.path.abspath(self.temp_dir)
        for frame_info in frames.values():
            if frame_info.get('store_frames', True):
                continue

            image_path = frame_info.get('path')
            if not image_path:
                continue

            image_path = os.path.abspath(image_path)
            try:
                if os.path.commonpath([temp_dir, image_path]) != temp_dir:
                    self.logger.warning(f"Refusing to delete non-temp VFA frame path: {image_path}")
                    continue
            except ValueError:
                self.logger.warning(f"Refusing to delete invalid VFA frame path: {image_path}")
                continue

            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except OSError as e:
                self.logger.warning(f"Failed to delete temporary VFA frame {image_path}: {e}")

    def _upload_result(self, time_bucket_key: float, result: dict[str, Any] | None):
        """Upload analysis results to InfluxDB.
        
        Args:
            time_bucket_key: timestamp of the processed frame set (the start time of the time bucket)
            result: analysis results from the multi-angle frame analyzer, or None if analysis failed
        """
        if result is None:
            self.logger.warning(f"No analysis results for time bucket {time_bucket_key}")
            return

        if self.session_id is None:
            self.logger.warning(f"Cannot upload result for time bucket {time_bucket_key}: session_id is None (synchronizer may be shutting down)")
            return

        from openmmla.utils.constants import EVENT_TYPE_VFA_ACTION
        fields = {
            "window_start_time": time_bucket_key,
            "window_end_time": time_bucket_key,
            "action_recognition": json.dumps(result),
        }
        print(f"{BLUE}[Action Recognition]{ENDC} {time_bucket_key}: "
              f"{BLUE}Multi-angle analysis results: {result}{ENDC}")

        if not self.influx_client.write_event(self.session_id, EVENT_TYPE_VFA_ACTION, fields):
            self.logger.error(f"Failed to upload result for time bucket {time_bucket_key}")

    @property
    def session_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current session_id."""
        if self.session_id:
            return f'{self.session_id}/vfa/control'
        return None
