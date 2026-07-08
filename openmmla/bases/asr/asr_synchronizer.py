import gc
import json
import os
import threading
import time

from openmmla.bases.synchronizer import Synchronizer
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, runtime_pipeline_artifact_dir
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_session, get_number_of_bases, show_error_and_pause
from openmmla.utils.logger import get_logger
from openmmla.utils.sync_strategy import TimeBucketSynchronizer, SyncStrategy
from .enums import BLUE, ENDC
from .input import get_function_synchronizer, get_base_type, get_synchronizer_mode


def start_asr_synchronizer(
    project_dir: str,
    config_path: str,
    mode: str = 'live',
    dominant: bool = False,
    sp: bool = False,
    session_id: str | None = None,
):
    """Start ASR Synchronizer with restart capability.
    
    Args:
        project_dir: Path to the project directory
        config_path: Path to the configuration file
        mode: Operating mode ('analyze' or 'live')
        dominant: Whether to select the dominant speaker
        sp: Whether the audio bases do speech separation
    """
    # Restart loop - allows restarting the entire process
    while True:
        try:
            synchronizer = ASRSynchronizer(
                project_dir=project_dir,
                config_path=config_path,
                mode=mode,
                dominant=dominant,
                sp=sp,
                session_id=session_id,
            )
            synchronizer.run()
        except KeyboardInterrupt as e:
            if "Exit" in str(e):
                print("\n👋 Goodbye!")
                break  # Exit completely when 'q' is pressed
            else:
                print("\n🔄 Restarting ASR Synchronizer...")
                continue  # Restart on Ctrl+C during runtime
        except Exception as e:
            show_error_and_pause(e, "restart ASR Synchronizer")
            print("\n🔄 Restarting ASR Synchronizer...")
            continue


class ASRSynchronizer(Synchronizer):
    """ASRSynchronizer class for synchronizing speaker recognition results from ASRBases among the same session and
    uploading the segment result to InfluxDB."""
    logger = get_logger('asr-synchronizer')

    def __init__(
        self,
        project_dir: str | None,
        config_path: str,
        mode: str = 'live',
        dominant: bool = False,
        sp: bool = False,
        session_id: str | None = None,
    ):
        """Initialize the ASRSynchronizer class.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            dominant: whether to select the dominant speaker or not (default: False)
            sp: tag of whether the audio bases do speech separation (default: False)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.mode = mode
        self.dominant = dominant
        self.sp = sp
        self.launch_session_id = session_id

        # Runtime attributes
        self.threads = []
        self.stop_event = threading.Event()
        self.session_id = None  # Session bucket name
        self.number_of_bases = None  # Number of group members
        self.latest_time = None  # Record start time of the most recent received frame
        self.time_bucket_buffer = {}  # Buffer for {time_bucket_key: {base_id: {<speakers>, <similarities>, <durations>, <segment_start_times>}}}

        self.base_type = get_base_type(self.config)

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        """Set up attributes from YAML configuration."""
        sync_config = self.config['Synchronizer']
        self.buffer_expiry_time = float(sync_config['result_expiry_time'])  # Expiry time of retained results
        recognize_duration = float(self.config['Base'][self.base_type]['recognize_sp_duration']) if self.sp else int(
            self.config['Base'][self.base_type]['recognize_duration'])
        self.bucket_duration = float(sync_config.get('bucket_duration', recognize_duration))
        self.match_tolerance = float(sync_config.get('match_tolerance', recognize_duration))

    def _setup_directories(self):
        """Set up required directories."""
        self.logger_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'asr-base', 'logger'))
        self.temp_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'asr-base', 'temp'))
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
        gc.collect()

    def run(self):
        """Run the ASR synchronizer."""
        print(f'\033]0;ASR Synchronizer for {self.base_type}\007')
        func_map = {1: self._start_synchronization, 2: self._switch_mode, 3: self._reset}

        while True:
            try:
                select_fun = get_function_synchronizer(self.mode)
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except KeyboardInterrupt as e:
                if "Exit" in str(e):
                    # 'q' was pressed in top-level menu - re-raise to be caught by outer restart loop
                    raise
                else:
                    # Ctrl+C during runtime or 'q' in lower-level menu - log and continue
                    self.logger.warning("Ctrl+C pressed during runtime, returning to main menu.", exc_info=True)
            except Exception as e:
                self.logger.warning(f"During running synchronizer, catch: {e}, Come back to the main menu.", exc_info=True)
                show_error_and_pause(e, "return to the ASR Synchronizer menu")
            finally:
                self._clean_up()

    def _start_synchronization(self):
        """Start the synchronization process."""
        # bucket selection
        self.session_id = self.launch_session_id or select_or_create_session(self.mongo_client)
        self.number_of_bases = get_number_of_bases()
        self._create_bucket_logger()

        # reset attributes
        self.latest_time = 0
        self.time_bucket_buffer = {}

        # listen for start signal
        self._listen_for_start_signal()

        # reinitialize mqtt client with a new topic and on_message callback
        self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics=f'{self.session_id}/asr')
        self.mqtt_client.loop_start()

        # create threads
        self._create_thread(self._send_start_regularly)
        self._create_thread(self._listen_for_stop_signal)

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

    def _switch_mode(self):
        """Switch the operating mode between 'analyze' and 'live'."""
        self.mode = get_synchronizer_mode()
        if self.mode == 'analyze': # post-time analysis
            self.buffer_expiry_time = 1000000000
        else:
            self.buffer_expiry_time = int(self.config['Synchronizer']['result_expiry_time'])

        self.logger.info(f"Switched to {self.mode} mode.")

    def _reset(self):
        """Reset the ASR synchronizer.
        
        Reinitialize the ASR synchronizer by calling the constructor with the current configuration,
        logs the reset status, and performs garbage collection.
        """
        self.__init__(project_dir=self.project_dir, config_path=self.config_path, mode=self.mode,
                      dominant=self.dominant, sp=self.sp)
        self.logger.info(f"ASR Synchronizer reset successfully.")
        gc.collect()

    def _create_bucket_logger(self):
        self.bucket_logger_dir = os.fspath(
            pipeline_section_dir(self.project_dir, self.session_id, 'asr-base', 'logger')
        )
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        copy_config_snapshot(self.config_path, self.project_dir, self.session_id, 'asr-base')
        self.logger = get_logger(f'synchronizer-{self.session_id}',
                                 os.path.join(self.bucket_logger_dir,
                                              f'asr_synchronizer_{self.base_type}.log'))

    def _handle_base_result(self, client, userdata, message):
        """Handle the received base recognition result from MQTT message.

        Process incoming ASR base results, organize them into time buckets, and determine
        when to merge and upload results. This is the core synchronization logic that:
        1. Groups results from the same time period
        2. Manages expired time buckets 
        3. Merges results when all bases have reported

        Args:
            client: The MQTT client instance that received the message
            userdata: Private user data as set in Client() or user_data_set()
            message: MQTT message instance containing the ASR base result payload
        """
        base_result = json.loads(message.payload.decode('utf-8'))
        base_result_time = float(base_result['segment_start_time'])

        # Initialize for the first received message
        if not self.latest_time:
            self.latest_time = base_result_time
            self.time_bucket_buffer[self.latest_time] = {}
            self._update_time_bucket_buffer(self.latest_time, base_result)
            return

        # Check and clean up expired frames based on buffer expiry time
        expired_times = TimeBucketSynchronizer.get_expired_buckets(
            base_result_time,
            self.time_bucket_buffer,
            self.buffer_expiry_time
        )

        for time_bucket_key in expired_times:
            frame_set = self.time_bucket_buffer[time_bucket_key]
            merged_result = self._merge_base_results(frame_set)
            merged_result['window_start_time'] = time_bucket_key  # start timestamp of the time bucket
            self.logger.debug(
                f"Expired frame set {time_bucket_key} with result {self.time_bucket_buffer[time_bucket_key]}")
            self._upload_merged_result(merged_result)
            del self.time_bucket_buffer[time_bucket_key]

        # Find the closest time bucket for the current message
        closest_time = TimeBucketSynchronizer.find_closest_time_bucket(
            current_time=base_result_time,
            time_buckets=self.time_bucket_buffer,
            base_id=base_result['base_id'],
            match_tolerance=self.match_tolerance,
            strategy=SyncStrategy.EARLIEST  # Using the earliest strategy for ASR
        )

        # Handle the current message
        if closest_time is None:
            if base_result_time > self.latest_time:
                # Create new time bucket for new message
                self.latest_time = base_result_time
                self.time_bucket_buffer[self.latest_time] = {}
                self._update_time_bucket_buffer(self.latest_time, base_result)
            else:  # outdated message
                self.logger.debug(
                    f"{base_result['base_id']} couldn't find a time bucket and is outdated, the base result time is {base_result_time}")
                return
        else:
            # Add to existing time bucket
            self._update_time_bucket_buffer(closest_time, base_result)

        # Process complete time buckets (all bases have reported)
        time_bucket_key = closest_time if closest_time else self.latest_time
        self.logger.debug(
            f"Time bucket {time_bucket_key} is selected for {base_result['base_id']} base result time is {base_result_time}")
        if len(self.time_bucket_buffer[time_bucket_key]) == self.number_of_bases:
            merged_result = self._merge_base_results(self.time_bucket_buffer[time_bucket_key])
            merged_result['window_start_time'] = time_bucket_key
            self._upload_merged_result(merged_result)
            del self.time_bucket_buffer[time_bucket_key]

    def _synchronization_handler(self, e: Exception | KeyboardInterrupt | None):
        """Handle exceptions during synchronization and perform cleanup.

        Stops all threads, disconnects from MQTT, runs session analysis, and 
        cleans up resources when synchronization is complete or encounters an error.

        Args:
            e: The exception that occurred during synchronization, or None if 
               synchronization completed normally
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped.")

        clear_directory(self.temp_dir)
        self._clean_up()

    def _send_start_regularly(self):
        """Send periodic START signals to ASR bases.

        Continuously sends START control signals to all ASR bases at regular intervals
        defined by self.bucket_duration_sec. This ensures that bases keep recording and processing
        audio even if they miss an initial start signal.

        The loop continues until the stop_event is set during shutdown.
        """
        while not self.stop_event.is_set():
            self.redis_client.publish(f"{self.session_id}/asr/control", 'START')
            time.sleep(self.bucket_duration)

    def _update_time_bucket_buffer(self, time_bucket_key: float, latest_base_result: dict):
        """Update the time bucket buffer with the latest base recognition result.

        Stores or updates recognition results from a specific base in the appropriate time bucket.
        If a result from the same base already exists for this time bucket, it will be overwritten
        with the newer result.

        Args:
            time_bucket_key: timestamp representing the start time of the time bucket
            latest_base_result: Recognition result dictionary from a single ASR base, containing:
                               - base_id: identifier of the source base
                               - speakers: JSON string of recognized speaker names
                               - similarities: JSON string of similarity scores
                               - durations: JSON string of audio durations
                               - segment_start_time: start time of the recorded segment
        """
        base_id = latest_base_result['base_id']
        if base_id in self.time_bucket_buffer[time_bucket_key]:
            self.logger.debug(
                f"Overwrite results: {self.time_bucket_buffer[time_bucket_key][base_id]}")

        # Store parsed (decoded from JSON) values in the buffer
        self.time_bucket_buffer[time_bucket_key][base_id] = {
            'speakers': json.loads(latest_base_result['speakers']),
            'similarities': json.loads(latest_base_result['similarities']),
            'durations': json.loads(latest_base_result['durations']),
            'segment_start_time': latest_base_result['segment_start_time'],
        }

    def _merge_base_results(self, frame_results: dict) -> dict:
        """Merge ASR results from multiple bases for a single time bucket.

        Based on synchronization strategy (dominant speaker or all speakers),
        this method consolidates recognition results from multiple audio bases
        into a single result set for the time period.

        Args:
            frame_results: Dictionary of base results for one time bucket.
                           Format: {base_id: {'speakers': [...], 'similarities': [...], 
                                   'durations': [...], 'segment_start_time': float}}

        Returns:
            Dictionary with merged results containing:
            - speakers: List of recognized speaker names
            - similarities: List of corresponding similarity scores
            - durations: List of corresponding audio durations
            - segment_start_times: List of corresponding segment start times
        """
        speakers, similarities, durations, segment_start_times = [], [], [], []

        if self.dominant:
            # Dominant speaker mode: only select the single most confident recognition
            best_result, i = self.find_best_base_result(frame_results)
            segment_start_times.append(best_result['segment_start_time'])
            speakers.append(best_result['speakers'][i])
            similarities.append(best_result['similarities'][i])
            durations.append(best_result['durations'][i])
        else:
            # All speakers mode: include all valid speaker recognitions
            for res in frame_results.values():
                # Append real speakers (exclude unknown and silent segments)
                for i, speaker in enumerate(res['speakers']):
                    if speaker not in ['unknown', 'silent']:
                        segment_start_times.append(res['segment_start_time'])
                        speakers.append(res['speakers'][i])
                        similarities.append(res['similarities'][i])
                        durations.append(res['durations'][i])

            # Fallback to best result if no valid speakers were found
            if not speakers:
                best_result, i = self.find_best_base_result(frame_results)
                segment_start_times.append(best_result['segment_start_time'])
                speakers.append(best_result['speakers'][i])
                similarities.append(best_result['similarities'][i])
                durations.append(best_result['durations'][i])

        return {
            'speakers': speakers,
            'similarities': similarities,
            'durations': durations,
            'segment_start_times': segment_start_times,
        }

    def _upload_merged_result(self, merged_result: dict):
        """Log and upload the merged ASR segment result to InfluxDB.
        
        Creates a final record for the synchronized segment containing all
        speaker information and metadata, displays a console log, and 
        persists the data to the database.
        
        Args:
            merged_result: Dictionary containing consolidated speaker recognition data:
                          - window_start_time: Start time of the aggregated time bucket
                          - speakers: List of recognized speaker names
                          - similarities: List of corresponding similarity scores
                          - durations: List of corresponding audio durations
                          - segment_start_times: List of base recording start times
        """
        from openmmla.utils.constants import EVENT_TYPE_ASR_RECOGNITION
        window_start = float(merged_result['window_start_time'])
        fields = {
            "window_start_time": window_start,
            "window_end_time": window_start + float(self.bucket_duration),
            "speakers": json.dumps(merged_result['speakers']),
            "similarities": json.dumps(merged_result['similarities']),
            "durations": json.dumps(merged_result['durations']),
            "segment_start_times": json.dumps(merged_result['segment_start_times']),
        }
        print(f"{BLUE}[Speaker Recognition]{ENDC}{window_start}: "
              f"{BLUE}{fields['speakers']}{ENDC}, "
              f"similarity: {fields['similarities']}")
        self.influx_client.write_event(self.session_id, EVENT_TYPE_ASR_RECOGNITION, fields)

    @staticmethod
    def find_best_base_result(segment_results: dict) -> tuple:
        """Find the best (most confident) speaker recognition result among all base results.

        Identifies which base and which speaker has the highest confidence score in the given 
        time segment. This is useful for determining the dominant speaker or as a fallback 
        when no valid speakers are detected.

        Args:
            segment_results: Dictionary of base results for one time segment.
                            Format: {base_id: {'speakers': [...], 'similarities': [...], 
                                    'durations': [...], 'segment_start_time': float}}

        Returns:
            Tuple containing:
            - best_result: The base result dictionary with highest confidence score
            - max_index: Index of the best speaker within that result's arrays
        """
        best_result = None
        max_similarity = -2  # Start with a very low value
        max_index = -1

        for result in segment_results.values():
            # Find the speaker with highest similarity score in this base result
            current_index, current_similarity = max(enumerate(result["similarities"]), key=lambda x: x[1])

            # Update if this is better than our previous best
            if current_similarity > max_similarity:
                best_result = result
                max_similarity = current_similarity
                max_index = current_index

        return best_result, max_index

    @property
    def session_control(self) -> str | None:
        """Get the Redis control channel name for the current bucket.
        
        This property dynamically constructs the communication channel name that
        should be used for sending control signals (START/STOP) to ASR bases.
        
        Returns:
            A Redis channel string in format '{session_id}/asr/control' if session_id
            is set, or None if no bucket is currently active.
        """
        if self.session_id:
            return f'{self.session_id}/asr/control'
        return None
