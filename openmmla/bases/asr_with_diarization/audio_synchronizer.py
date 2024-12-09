import gc
import json
import logging
import os
import threading
import time

from openmmla.analytics.audio.analyze import session_analysis_audio
from openmmla.bases.synchronizer import Synchronizer
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.logger import get_logger
from .enums import BLUE, ENDC
from .input import get_bucket_name, get_number_of_group_members, get_function_synchronizer


class AudioSynchronizer(Synchronizer):
    """The synchronizer synchronizes speaker recognition results from audio bases among the same session."""
    logger = get_logger('synchronizer')

    def __init__(self, project_dir: str, config_path: str, base_type: str, dominant: bool = False, sp: bool = False):
        """
        Initialize the synchronizer object.

        Args:
            project_dir: root directory of the project
            config_path: path to the configuration file, either absolute or relative to the root directory
            base_type: the audio base type
            dominant: whether to select the dominant speaker or not
            sp: tag of whether the audio bases do speech separation
        """
        super().__init__(project_dir, config_path)
        self.base_type = base_type.capitalize()
        self.dominant = dominant
        self.sp = sp

        """Runtime attributes."""
        self.threads = []
        self.stop_event = threading.Event()
        self.bucket_name = None
        self.number_of_speaker = None  # Number of group members
        self.latest_time = None  # Record start time of the most recent received segment
        self.retained_segments_results = None  # Results to be handled {segment_time: {base_id: {'speaker':x,
        # 'similarity':x, 'duration':x}}}

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        """Set up attributes from YAML configuration."""
        self.result_expiry_time = int(
            self.config['Synchronizer']['result_expiry_time'])  # Expiry time of retained results
        self.time_range = int(self.config[self.base_type]['recognize_sp_duration']) if self.sp else int(
            self.config[self.base_type]['recognize_duration'])  # Time range for finding the closest segment

    def _setup_directories(self):
        """Set up required directories."""
        self.logger_dir = os.path.join(self.project_dir, 'logger')
        self.audio_temp_dir = os.path.join(self.project_dir, 'audio', 'temp')
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.audio_temp_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.redis_client = RedisClientWrapper(self.config_path)  # Redis wrapped client
        self.mqtt_client = MQTTClientWrapper(self.config_path)  # MQTT wrapped client
        self.influx_client = InfluxDBClientWrapper(self.config_path)  # InfluxDB wrapped client

    def _clean_up(self):
        """Free memory by resetting dictionaries."""
        self.bucket_name = None
        self.number_of_speaker = None
        self.latest_time = None
        self.retained_segments_results = None
        gc.collect()

    def run(self):
        """Main menu for audio synchronizer."""
        print('\033]0;Audio Synchronizer\007')
        func_map = {1: self._start_synchronizing}

        while True:
            try:
                select_fun = get_function_synchronizer()
                if select_fun == 0:
                    print("------------------------------------------------")
                    clear_directory(os.path.join(self.audio_temp_dir))
                    self.logger.info("Exiting audio synchronizer...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"\nDuring running synchronizer, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)

    def _start_synchronizing(self):
        """Start the synchronization process."""
        self.bucket_name = get_bucket_name(self.influx_client)
        self.number_of_speaker = get_number_of_group_members()
        self.latest_time = 0
        self.retained_segments_results = {}
        self.logger = get_logger(f'synchronizer-{self.bucket_name}',
                                 os.path.join(self.logger_dir, f'{self.bucket_name}_synchronizer.log'),
                                 level=logging.DEBUG, console_level=logging.INFO, file_level=logging.DEBUG)

        self._listen_for_start_signal()

        # Reinitialize MQTT client with a new topic and on_message callback
        self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics=f'{self.bucket_name}/audio')
        self.mqtt_client.loop_start()

        # Create threads
        self._create_thread(self._send_start_regularly)
        self._create_thread(self._listen_for_stop_signal)

        # Start threads
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

    def _synchronization_handler(self, e):
        """Handle exceptions and stop all threads.

        Args:
            e: the exception that occurred during the synchronization process
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped.")

        self.mqtt_client.loop_stop()
        self.redis_client.publish(f"{self.bucket_name}/control", 'STOP')  # send STOP command to all bases
        session_analysis_audio(self.project_dir, self.bucket_name, self.influx_client)
        clear_directory(self.audio_temp_dir)
        self._clean_up()

    def _send_start_regularly(self):
        """Send the START signal to all bases regularly."""
        while not self.stop_event.is_set():
            self.redis_client.publish(f"{self.bucket_name}/control", 'START')
            time.sleep(self.time_range)

    def _handle_base_result(self, client, userdata, message):
        """Handle the received base result.

        Args:
            client: the client instance for this callback
            userdata: the private user data as a set in Client() or user_data_set()
            message: an instance of MQTTMessage
        """
        new_base_result = json.loads(message.payload.decode('utf-8'))
        record_start_time = float(new_base_result['record_start_time'])

        if not self.latest_time:
            self.latest_time = int(record_start_time)
            self.retained_segments_results[self.latest_time] = {}
            self._update_retained_segments_results(self.latest_time, new_base_result)
            return

        expired_segments = [segment_time for segment_time in self.retained_segments_results.keys() if
                            record_start_time - segment_time > self.result_expiry_time]

        for segment_time in expired_segments:
            merged_segment_result = self._merge_segment_results(self.retained_segments_results[segment_time])
            merged_segment_result['segment_start_time'] = segment_time
            self.logger.debug(
                f"\033[91mExpired segment {segment_time} with result {self.retained_segments_results[segment_time]}\033[0m")
            self._upload_merged_result(merged_segment_result)
            del self.retained_segments_results[segment_time]

        closest_segment_time = self._find_closest_segment(record_start_time, new_base_result['base_id'])
        if closest_segment_time is None:
            if self.latest_time > record_start_time:  # outdated message
                self.logger.debug(
                    f"\033[91m{new_base_result['base_id']} results is outdated, record time is {record_start_time}\033[0m")
                return
            else:
                # self.latest_time = self._update_time(self.latest_time, record_start_time)
                # self.latest_time += self.time_range
                self.latest_time = record_start_time  # Update the latest time to the record start time
                self.retained_segments_results[self.latest_time] = {}
                self._update_retained_segments_results(self.latest_time, new_base_result)
        else:
            self._update_retained_segments_results(closest_segment_time, new_base_result)

        segment_time = closest_segment_time if closest_segment_time else self.latest_time
        self.logger.debug(
            f"Segment {segment_time} is selected for {new_base_result['base_id']} record time is {record_start_time}")
        if len(self.retained_segments_results[segment_time]) == self.number_of_speaker:
            merged_segment_result = self._merge_segment_results(self.retained_segments_results[segment_time])
            merged_segment_result['segment_start_time'] = segment_time
            self._upload_merged_result(merged_segment_result)
            del self.retained_segments_results[segment_time]

    def _update_retained_segments_results(self, segment_time, latest_base_result):
        """Helper function to update the base result of one segment.

        Args:
            segment_time: the segment time of the recognition result
            latest_base_result: the latest recognition result from one base
        """
        if latest_base_result['base_id'] in self.retained_segments_results[segment_time]:
            self.logger.debug(
                f"\033[91mOverwrite results: {self.retained_segments_results[segment_time][latest_base_result['base_id']]}\033[0m")
        self.retained_segments_results[segment_time][latest_base_result['base_id']] = {
            'record_start_time': latest_base_result['record_start_time'],
            'speakers': json.loads(latest_base_result['speakers']),
            'similarities': json.loads(latest_base_result['similarities']),
            'durations': json.loads(latest_base_result['durations']),
        }

    def _merge_segment_results(self, segment_results):
        """Merge the base results of one segment, keys are base id, values are base results.

        Args:
            segment_results: the base results dictionary of one segment

        Returns:
            The consolidated result dictionary of one segment
        """
        speakers, similarities, durations, record_start_times = [], [], [], []
        if self.dominant:
            best_result, i = self.find_best_base_result(segment_results)
            record_start_times.append(best_result['record_start_time'])
            speakers.append(best_result['speakers'][i])
            similarities.append(best_result['similarities'][i])
            durations.append(best_result['durations'][i])
        else:
            for res in segment_results.values():
                # Append real speakers
                for i, speaker in enumerate(res['speakers']):
                    if speaker not in ['unknown', 'silent']:
                        record_start_times.append(res['record_start_time'])
                        speakers.append(res['speakers'][i])
                        similarities.append(res['similarities'][i])
                        durations.append(res['durations'][i])
            if not speakers:
                best_result, i = self.find_best_base_result(segment_results)
                record_start_times.append(best_result['record_start_time'])
                speakers.append(best_result['speakers'][i])
                similarities.append(best_result['similarities'][i])
                durations.append(best_result['durations'][i])

        return {'speakers': speakers, 'similarities': similarities, 'record_start_times': record_start_times,
                'durations': durations}

    # def _find_closest_segment(self, current_time):
    #     time_differences = {key: abs(current_time - key) for key in self.retained_results.keys()}
    #     valid_segments = {key: diff for key, diff in time_differences.items() if diff <= self.time_range}
    #     closest_segment_time = min(valid_segments.keys(), default=None)
    #     return closest_segment_time

    def _find_closest_segment(self, current_time, base_id):
        """Find the closest segment of the new received segment among the retained segments results.

        Args:
            current_time: the record start time of the received segment
            base_id: the base id of the received segment

        Returns:
            The segment time of the closest segment, or None if no valid segments are found
        """
        # Calculate time differences for all segments in retained results
        time_differences = {key: abs(current_time - key) for key in self.retained_segments_results.keys()}

        # Filter valid segments based on time difference and exclude segments where base_id already exists
        valid_segments = {
            key: diff for key, diff in time_differences.items()
            if diff <= self.time_range and base_id not in self.retained_segments_results[key]
        }

        # If no valid segments are found, return None
        if not valid_segments:
            return None

        # Choose the segment with the earliest segment time
        closest_segment_time = min(valid_segments.keys())

        # Choose the segment with the nearest segment time
        # closest_segment_time = min(valid_segments, key=valid_segments.get)

        return closest_segment_time

    # def _update_time(self, latest_time, record_start_time):
    #     n = (record_start_time - latest_time) // self.time_range
    #     updated_latest_time = latest_time + self.time_range * n
    #     if record_start_time - updated_latest_time >= self.time_range / 2:
    #         updated_latest_time += self.time_range
    #     return updated_latest_time

    def _upload_merged_result(self, merged_result):
        """Log and upload the merged segment result to InfluxDB.

        Args:
            merged_result: the merged result dictionary of one segment
        """
        recognition_data = {
            "measurement": "speaker recognition",
            "fields": {
                "segment_start_time": float(merged_result['segment_start_time']),
                "speakers": json.dumps(merged_result['speakers']),
                "similarities": json.dumps(merged_result['similarities']),
                "record_start_times": json.dumps(merged_result['record_start_times']),
                "durations": json.dumps(merged_result['durations'])
            },
        }
        print(f"{BLUE}[Speaker Recognition]{ENDC}{recognition_data['fields']['segment_start_time']}: "
              f"{BLUE}{recognition_data['fields']['speakers']}{ENDC}, "
              f"similarity: {recognition_data['fields']['similarities']}")
        self.influx_client.write(self.bucket_name, recognition_data)

    @staticmethod
    def find_best_base_result(segment_results: dict):
        """Find the best result among the base results of one segment.

        Args:
            segment_results: the base results dictionary of one segment

        Returns:
            The best result and its index
        """
        best_result = None
        max_similarity = -2
        max_index = -1

        for result in segment_results.values():
            current_index, current_similarity = max(enumerate(result["similarities"]), key=lambda x: x[1])
            if current_similarity > max_similarity:
                best_result = result
                max_similarity = current_similarity
                max_index = current_index

        return best_result, max_index
