import gc
import json
import os
import queue
import threading

from typing import Any

from openmmla.analysis.vfa.analyze import vfa_session_analysis
from openmmla.bases.synchronizer import Synchronizer
from openmmla.services.vfa.requests import request_multi_angle_frame_analyze
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_bucket, get_number_of_bases
from openmmla.utils.logger import get_logger
from openmmla.utils.sync_strategy import TimeBucketSynchronizer, SyncStrategy
from .enums import BLUE, ENDC
from .input import get_function_synchronizer, select_participant_descriptions


class VFASynchronizer(Synchronizer):
    """VFASynchronizer class for synchronizing video frames from multiple angles."""
    logger = get_logger('vfa-synchronizer')

    def __init__(self, project_dir: str | None, config_path: str):
        """Initialize the VFASynchronizer class.
        
        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        # Runtime attributes
        self.threads = []
        self.stop_event = threading.Event()
        self.bucket_name = None
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
        self.vllm_frame_analyzer_url = vfa_server_config['vllm_frame_analyzer']
        
        self.participant_config = sync_config.get('participant_config', {})
        self.angle_config = base_config.get('angle_config', {})
        
        self.logger.info(f"Loaded participant configurations: {list(self.participant_config.keys()) if self.participant_config else 'None'}")
        self.logger.info(f"Loaded angle configurations: {list(self.angle_config.keys()) if self.angle_config else 'None'}")

    def _setup_directories(self):
        """Set up required directories."""
        self.logger_dir = os.path.join(self.project_dir, 'logger')
        self.temp_dir = os.path.join(self.project_dir, 'real-time', 'temp')
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.redis_client = RedisClientWrapper(self.config_path)  # Redis wrapped client
        self.mqtt_client = MQTTClientWrapper(self.config_path)  # MQTT wrapped client
        self.influx_client = InfluxDBClientWrapper(self.config_path)  # InfluxDB wrapped client

    def _clean_up(self):
        """Clean up runtime variables and free memory."""
        self.mqtt_client.loop_stop()
        self.bucket_name = None
        self.number_of_bases = None
        self.latest_time = None
        self.time_bucket_buffer = {}
        self.selected_participant_descriptions = None
        self.vllm_queue = queue.Queue()
        self._clear_threads()
        gc.collect()

    def run(self):
        """Run the VFA synchronizer."""
        print('\033]0;VFA Synchronizer\007')
        func_map = {1: self._start_synchronization, 2: self._reinit}

        while True:
            try:
                select_fun = get_function_synchronizer()
                if select_fun == 0:
                    print("------------------------------------------------")
                    clear_directory(os.path.join(self.temp_dir))
                    self.logger.info("Exiting VFA synchronizer...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"\nDuring running synchronizer, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)
            finally:
                self._clean_up()

    def _start_synchronization(self):
        """Start the synchronization process."""
        # reset attributes
        self.latest_time = 0
        self.time_bucket_buffer = {}

        # bucket selection
        self.bucket_name = select_or_create_bucket(self.influx_client)
        self.number_of_bases = get_number_of_bases()
        
        # select participant descriptions
        self.selected_participant_descriptions = select_participant_descriptions(self.participant_config)
        
        self._create_bucket_logger()

        # listen for start signal
        self._listen_for_start_signal()

        # reinitialize mqtt client with a new topic and on_message callback
        self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics=f'{self.bucket_name}/vfa')
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

    def _create_bucket_logger(self):
        self.bucket_logger_dir = os.path.join(self.logger_dir, f'{self.bucket_name}')
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        self.logger = get_logger(f'vfa-synchronizer-{self.bucket_name}',
                                 os.path.join(self.bucket_logger_dir, f'vfa_synchronizer.log'))

    def _handle_base_result(self, client, userdata, message):
        """Handle received frame from a VFA base."""
        try:
            base_result = json.loads(message.payload.decode('utf-8'))
            base_result_time = float(base_result['acquired_time'])
            base_id = base_result['base_id']
            angle = base_result['angle']
            image_path = base_result['image_path']

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

            self.time_bucket_buffer[closest_time][base_id] = {
                'angle': angle,
                'path': image_path,
                'base_result_time': base_result_time
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
        
        vfa_session_analysis(self.project_dir, self.bucket_name, self.influx_client)
        clear_directory(self.temp_dir)
        self._clean_up()

    def _process_vllm_requests(self):
        """Process VLLM requests from the queue."""
        while not self.stop_event.is_set():
            try:
                # get a frame set from the queue with a timeout
                frame_set = self.vllm_queue.get(timeout=1.0)
                
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
                                session_id=self.bucket_name,
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
                    # if not stopped, mark the task as done since vllm_queue is still existing
                    if not self.stop_event.is_set():
                        self.vllm_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in VLLM processing thread: {e}", exc_info=True)

    def _upload_result(self, time_bucket_key: float, result: dict[str, Any] | None):
        """Upload analysis results to InfluxDB.
        
        Args:
            time_bucket_key: timestamp of the processed frame set (the start time of the time bucket)
            result: analysis results from the multi-angle frame analyzer, or None if analysis failed
        """
        if result is None:
            self.logger.warning(f"No analysis results for time bucket {time_bucket_key}")
            return

        # Check if bucket_name is available (might be None if cleanup has already occurred)
        if self.bucket_name is None:
            self.logger.warning(f"Cannot upload result for time bucket {time_bucket_key}: bucket_name is None (synchronizer may be shutting down)")
            return

        analysis_data = {
            "measurement": "action_recognition",
            "fields": {
                "window_start_time": time_bucket_key,
                "window_end_time": time_bucket_key,
                "action_recognition": json.dumps(result)
            },
        }
        print(f"{BLUE}[Action Recognition]{ENDC} {analysis_data['fields']['window_start_time']}: "
              f"{BLUE}Multi-angle analysis results: {result}{ENDC}")
        
        try:
            self.influx_client.write(self.bucket_name, analysis_data)
        except Exception as e:
            self.logger.error(f"Failed to upload result for time bucket {time_bucket_key}: {e}")

    @property
    def bucket_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current bucket_name."""
        if self.bucket_name:
            return f'{self.bucket_name}/vfa/control'
        return None
