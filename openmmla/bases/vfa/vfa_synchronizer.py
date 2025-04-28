import gc
import json
import os
import threading
import time
from typing import Any

from openmmla.analysis.vfa.analyze import vfa_session_analysis
from openmmla.bases.synchronizer import Synchronizer
from openmmla.services.vfa.requests import request_multi_angle_frame_analyze
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import get_bucket_name
from openmmla.utils.logger import get_logger
from .enums import BLUE, ENDC
from .input import get_function_synchronizer, get_number_of_cameras


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
        self.number_of_cameras = None  # Number of camera bases
        self.latest_time = None
        self.frame_buffer = {}  # Buffer for frames {timestamp: {base_id: {angle, path, acquired_time}}}

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        """Load configuration parameters."""
        sync_config = self.config['Synchronizer']
        vfa_server_config = self.config['Server']['vfa']

        self.buffer_expiry_time = int(sync_config.get('result_expiry_time', 30))
        self.time_range = float(sync_config.get('time_range', 0.5))  # Time window for syncing frames (in seconds)
        self.vllm_frame_analyzer_url = vfa_server_config['vllm_frame_analyzer']

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
        """Free memory by resetting attributes."""
        self.bucket_name = None
        self.number_of_cameras = None
        self.latest_time = None
        self.frame_buffer = {}
        gc.collect()

    def run(self):
        """Run the VFA synchronizer."""
        print('\033]0;VFA Synchronizer\007')
        func_map = {1: self._start_synchronization}

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

    def _start_synchronization(self):
        """Start the synchronization process."""
        self.bucket_name = get_bucket_name(self.influx_client)
        self.number_of_cameras = get_number_of_cameras()
        self.latest_time = 0
        self.frame_buffer = {}
        self.logger = get_logger(f'synchronizer-{self.bucket_name}',
                                 os.path.join(self.logger_dir, f'{self.bucket_name}_vfa_synchronizer.log'))

        self._listen_for_start_signal()

        # Reinitialize MQTT client with a new topic and on_message callback
        self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics=f'{self.bucket_name}/vfa')
        self.mqtt_client.loop_start()

        # Create threads
        self._create_thread(self._send_start_regularly)
        self._create_thread(self._listen_for_stop_signal)
        self._create_thread(self._cleanup_expired_frames)

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
        vfa_session_analysis(self.project_dir, self.bucket_name, self.influx_client)
        clear_directory(self.temp_dir)
        self._clean_up()

    def _send_start_regularly(self):
        """Send the START signal to all bases regularly."""
        while not self.stop_event.is_set():
            self.redis_client.publish(f"{self.bucket_name}/control", 'START')
            time.sleep(5)  # Send START every 5 seconds

    def _handle_base_result(self, client, userdata, message):
        """Handle received frame from a VFA base.
        
        Args:
            client: the client instance for this callback
            userdata: the private user data
            message: an instance of MQTTMessage
        """
        try:
            frame_data = json.loads(message.payload.decode('utf-8'))
            acquired_time = float(frame_data['acquired_time'])
            base_id = frame_data['base_id']
            angle = frame_data['angle']
            image_path = frame_data['image_path']

            if not os.path.exists(image_path):
                self.logger.warning(f"Image path does not exist: {image_path}")
                return

            # Find or create closest time bucket
            closest_time = self._find_closest_time_bucket(acquired_time, base_id)
            if not closest_time:
                closest_time = acquired_time
                self.frame_buffer[closest_time] = {}

            # Store frame info
            if base_id in self.frame_buffer[closest_time]:
                self.logger.debug(f"Overwriting frame for base {base_id} at time {closest_time}")

            self.frame_buffer[closest_time][base_id] = {
                'angle': angle,
                'path': image_path,
                'acquired_time': acquired_time
            }

            # Check if we've received frames from all cameras for this time bucket
            if len(self.frame_buffer[closest_time]) == self.number_of_cameras:
                self._process_frame_set(closest_time)

        except Exception as e:
            self.logger.error(f"Error handling frame: {e}", exc_info=True)

    def _find_closest_time_bucket(self, time_value: float, base_id: str) -> float | None:
        """Find the closest time bucket for synchronization.
        
        Args:
            time_value: timestamp to find closest bucket for
            base_id: ID of the base sending the frame
            
        Returns:
            float or None: Timestamp of the closest bucket or None if no suitable bucket exists
        """
        if not self.frame_buffer:
            return None

        valid_times = [t for t in self.frame_buffer.keys()
                       if abs(t - time_value) <= self.time_range and base_id not in self.frame_buffer[t]]

        if not valid_times:
            return None

        return min(valid_times, key=lambda t: abs(t - time_value))

    def _process_frame_set(self, time_bucket: float):
        """Process a complete set of synchronized frames.
        
        Args:
            time_bucket: timestamp of the frame set to process
        """
        frame_set = self.frame_buffer[time_bucket]

        # Prepare data for multi-angle analysis
        images = []
        angles = []

        for base_id, frame_info in sorted(frame_set.items()):
            if os.path.exists(frame_info['path']):
                images.append(frame_info['path'])
                angles.append(frame_info['angle'])
            else:
                self.logger.warning(f"Image path no longer exists: {frame_info['path']}")

        if not images:
            self.logger.warning(f"No valid images found for time bucket {time_bucket}")
            del self.frame_buffer[time_bucket]
            return

        try:
            # Use request function for multiple images
            result = request_multi_angle_frame_analyze(
                image_paths=images,
                session_id=self.bucket_name,
                url=self.vllm_frame_analyzer_url,
                angles=angles
            )

            # Upload results
            self._upload_result(time_bucket, result)

        except Exception as e:
            self.logger.error(f"Error processing frame set: {e}", exc_info=True)

        # Clean up processed frame set
        del self.frame_buffer[time_bucket]

    def _cleanup_expired_frames(self):
        """Periodically clean up expired frames from the buffer."""
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                expired_times = [t for t in self.frame_buffer.keys()
                                 if current_time - t > self.buffer_expiry_time]

                for t in expired_times:
                    frame_set = self.frame_buffer[t]
                    # If we have at least 2 frames, try to process what we have
                    if len(frame_set) >= 2:
                        self.logger.info(
                            f"Processing incomplete frame set at {t} with {len(frame_set)}/{self.number_of_cameras} frames")
                        self._process_frame_set(t)
                    else:
                        self.logger.warning(f"Dropping expired frame set at {t} with only {len(frame_set)} frame(s)")
                        del self.frame_buffer[t]
            except Exception as e:
                self.logger.error(f"Error in cleanup thread: {e}", exc_info=True)

            time.sleep(5)  # Check every 5 seconds

    def _upload_result(self, time_bucket: float, result: dict[str, Any] | None):
        """Upload analysis results to InfluxDB.
        
        Args:
            time_bucket: timestamp of the processed frame set
            result: analysis results from the multi-angle frame analyzer, or None if analysis failed
        """
        if result is None:
            self.logger.warning(f"No analysis results for time bucket {time_bucket}")
            return

        analysis_data = {
            "measurement": "action recognition",
            "fields": {
                "acquired_time": time_bucket,
                "action_recognition": json.dumps(result)
            },
        }
        print(f"{BLUE}[Action Recognition]{ENDC} {analysis_data['fields']['acquired_time']}: "
              f"{BLUE}Multi-angle analysis results: {result}{ENDC}")
        self.influx_client.write(self.bucket_name, analysis_data)
