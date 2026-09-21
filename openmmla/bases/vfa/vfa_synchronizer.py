import gc
import json
import os
import queue
import threading
import time

from typing import Any

from openmmla.bases.synchronizer import Synchronizer
from openmmla.services.vfa.requests import request_frame_features, request_multi_angle_frame_analyze
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, runtime_pipeline_artifact_dir
from openmmla.utils import session_provenance
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.config import get_bases
from openmmla.utils.input import select_or_create_session, get_number_of_bases, pause_after_error, show_error_and_pause
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import build_service_url
from openmmla.utils.sync_strategy import TimeBucketSynchronizer, SyncStrategy
from .enums import BLUE, ENDC
from .input import get_function_synchronizer


def _config_flag(value, default: bool) -> bool:
    """a config flag: true/false, 1/0, yes/no, on/off; `default` for none or an unfilled placeholder."""
    text = str(value).strip().lower() if value is not None else ""
    if not text or (text.startswith("<") and text.endswith(">")):
        return default
    if isinstance(value, bool):
        return value
    return text in {"true", "1", "yes", "y", "on"}


class VFASynchronizer(Synchronizer):
    """VFASynchronizer class for synchronizing video frames from multiple angles."""
    logger = get_logger('vfa-synchronizer')

    def __init__(self, project_dir: str | None, config_path: str, session_id: str | None = None,
                 num_bases: int | None = None, actions: bool | None = None, pose: bool | None = None,
                 gaze: bool | None = None):
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
            actions: whether every synchronized frame set is sent to the frame
                analyzer for its action labels (the VLM; the vfa_action event);
                None leaves it to the config (Synchronizer.actions)
            pose: whether every synchronized frame set is sent to the frame
                analyzer's features endpoint for its skeletons, tags and head
                yaws (no VLM; the vfa_features event, one per frame set, so as
                often as the bases' keyframe_interval); None leaves it to the
                config (Synchronizer.pose)
            gaze: whether those features come with the gaze model's gazes, which
                need the pose (a gaze lands on someone's face or hands), so gaze
                on means pose on; None leaves it to the config (Synchronizer.gaze)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.launch_session_id = session_id
        self.launch_num_bases = num_bases
        self.launch_actions = actions
        self.launch_pose = pose
        self.launch_gaze = gaze

        # Runtime attributes
        self.threads = []
        self.stop_event = threading.Event()
        self.session_id = None
        self.number_of_bases = None
        self.latest_time = None
        self.time_bucket_buffer = {}  # Buffer for {time_bucket_key: {base_id: {<angle>, <path>, <base_result_time>}}}
        self.selected_participant_descriptions = None  # Selected participant descriptions for current session

        # one queue per request kind, each with its own worker: the action labels wait on the
        # VLM (seconds to a minute), the features on the pose model (a second), and neither
        # holds the other up. A frame set goes to both, and the last to finish cleans its frames
        self.vllm_queue = queue.Queue()
        self.features_queue = queue.Queue()
        self.vllm_processing_thread = None
        self._jobs_lock = threading.Lock()

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
        # the features endpoint: whether the skeletons ride along in every vfa_features event, and
        # the zones a gaze may land in, from a JSON file ({name: polygon} for every angle, or
        # {angle: {name: polygon}}) named by the config; an unfilled placeholder is no file
        # what each frame set is sent for: the config's say, unless the launch flags (-a, -pose,
        # -gaze) said otherwise. The gazes need the pose (a gaze lands on someone's face or
        # hands), so gaze on means pose on
        config_actions = _config_flag(sync_config.get('actions'), True)
        config_pose = _config_flag(sync_config.get('pose'), False)
        config_gaze = _config_flag(sync_config.get('gaze'), False)
        self.actions = config_actions if self.launch_actions is None else bool(self.launch_actions)
        self.gaze = config_gaze if self.launch_gaze is None else bool(self.launch_gaze)
        self.pose = (config_pose if self.launch_pose is None else bool(self.launch_pose)) or self.gaze
        self.features_keypoints = _config_flag(sync_config.get('pose_keypoints'), True)
        self.feature_zones = self._load_feature_zones(sync_config.get('feature_zones_file'))
        # the VLM at its own pace: with the bases sending a frame set every second for the
        # features, the action labels are asked for at most once per action_interval seconds
        # (0: every frame set)
        try:
            self.action_interval = max(0.0, float(sync_config.get('action_interval') or 0))
        except (TypeError, ValueError):
            self.action_interval = 0.0
        self._last_action_time = None
        
        self.logger.info(f"Loaded angle configurations: {list(self.angle_config.keys()) if self.angle_config else 'None'}")

    def _load_feature_zones(self, path) -> dict | None:
        """the zones of the features endpoint, from the JSON file the config names; None for
        none, or for a file that cannot be read (said in the log, never a stop)."""
        text = str(path or "").strip()
        if not text or (text.startswith("<") and text.endswith(">")):
            return None
        if not os.path.isabs(text):
            text = os.path.join(os.path.dirname(os.path.abspath(self.config_path)), text)
        try:
            with open(text, 'r', encoding='utf-8') as handle:
                zones = json.load(handle)
            if not isinstance(zones, dict):
                raise ValueError("not a JSON object")
            self.logger.info(f"Feature zones: {sorted(zones)} from {text}")
            return zones
        except Exception as e:
            self.logger.warning(f"Synchronizer.feature_zones_file {text} could not be read ({e}): no zones")
            return None

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
        self.features_queue = queue.Queue()
        gc.collect()

    def _reinit(self):
        """Reload the config, keeping the session and the number of bases the
        synchronizer was launched with (the base class would drop them)."""
        self.logger.info("Starting synchronizer reinitialization...")
        project_dir, config_path = self.project_dir, self.config_path
        session_id, num_bases = self.launch_session_id, self.launch_num_bases
        actions, pose, gaze = self.launch_actions, self.launch_pose, self.launch_gaze
        self._clean_up()
        self.__init__(project_dir=project_dir, config_path=config_path, session_id=session_id,
                      num_bases=num_bases, actions=actions, pose=pose, gaze=gaze)
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
        self._last_action_time = None

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
        self._record_provenance()

        # listen for start signal
        if not self._listen_for_start_signal():
            # STOP came before START: nothing was started
            self._clean_up()
            return True

        # reinitialize mqtt client with a new topic and on_message callback
        self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics=f'{self.session_id}/vfa')
        self.mqtt_client.loop_start()
        if not self.actions and not self.pose:
            self.logger.warning("Neither action labels (-a) nor pose (-pose) nor gaze (-gaze) are asked for: the "
                                "synchronizer merges the frames and sends nothing to the frame analyzer.")
        else:
            self.logger.info(f"Each frame set goes to the frame analyzer for: "
                             f"{', '.join(name for name, on in (('action labels', self.actions), ('pose', self.pose), ('gaze', self.gaze)) if on)}")

        # create threads: one worker per request kind
        self._create_thread(self._listen_for_stop_signal)
        self._create_thread(self._process_vllm_requests)
        self._create_thread(self._process_feature_requests)

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

    def _record_provenance(self):
        """Note in the session what this synchronizer runs with (openmmla.utils.session_provenance):
        how many bases it merges, its tolerances, the participants it describes to the frame
        analyzer, whether it asks for action labels and for features (and with which zones and
        whether the skeletons ride along); what that analyzer runs (its models, prompt profile,
        pose model and thresholds) is asked after, in a thread. A failure is a warning, never a
        stop."""
        if not self.session_id:
            return
        try:
            entry = session_provenance.component_entry(
                'vfa', 'synchronizer',
                arguments={'session_id': self.launch_session_id, 'num_bases': self.launch_num_bases,
                           'actions': self.launch_actions, 'pose': self.launch_pose, 'gaze': self.launch_gaze},
                parameters={'number_of_bases': self.number_of_bases, 'buffer_expiry_time': self.buffer_expiry_time,
                            'match_tolerance': self.match_tolerance, 'angle_config': self.angle_config,
                            'participant_descriptions': self.selected_participant_descriptions,
                            'actions': self.actions, 'pose': self.pose, 'gaze': self.gaze,
                            'pose_keypoints': self.features_keypoints, 'feature_zones': self.feature_zones,
                            'action_interval': self.action_interval,
                            'service_urls': {'vllm_frame_analyzer': self.vllm_frame_analyzer_url}},
                config=self.config, config_path=self.config_path, project_dir=self.project_dir)
            session_provenance.record_component(self.mongo_client, self.session_id, entry, self.project_dir,
                                                'vfa-base', log=self.logger)
            session_provenance.record_services_later(
                self.mongo_client, self.session_id, entry, {'vllm_frame_analyzer': self.vllm_frame_analyzer_url},
                self.project_dir, 'vfa-base', log=self.logger)
        except Exception as e:
            self.logger.warning(f"Could not note in session {self.session_id} what the VFA synchronizer runs with: {e}")

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
                    self._dispatch(t, frame_set)
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
                # closest_time is the time_bucket_key (start time of the bucket)
                self._dispatch(closest_time, self.time_bucket_buffer[closest_time])
                del self.time_bucket_buffer[closest_time]

        except Exception as e:
            self.logger.error(f"Error handling frame: {e}", exc_info=True)

    def _dispatch(self, time_bucket_key: float, frames: dict[str, dict[str, Any]]) -> None:
        """hand a synchronized frame set to the workers it is for: the action labels, the
        features, or both; a set no one is asked for is cleaned up at once."""
        job = {'time_bucket_key': time_bucket_key, 'frames': frames,
               'pending': int(bool(self.actions)) + int(bool(self.pose))}
        if not job['pending']:
            self._cleanup_unstored_frames(frames)
            return
        if self.actions:
            self.vllm_queue.put(job)
        if self.pose:
            self.features_queue.put(job)

    def _finish(self, job: dict) -> None:
        """one worker is done with a frame set; the last one cleans its unstored frames."""
        with self._jobs_lock:
            job['pending'] = job.get('pending', 1) - 1
            last = job['pending'] <= 0
        if last:
            self._cleanup_unstored_frames(job['frames'])

    def _frame_set_paths(self, frames: dict[str, dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
        """the image paths, angles and angle descriptions of a frame set, in base order, leaving
        out a frame whose file is gone."""
        image_paths, angles, angle_descriptions = [], [], []
        for base_id, frame_info in sorted(frames.items()):
            if os.path.exists(frame_info['path']):
                angle = frame_info['angle']
                image_paths.append(frame_info['path'])
                angles.append(angle)
                angle_descriptions.append(self.angle_config.get(angle, f"Image from {angle} perspective"))
            else:
                self.logger.warning(f"Image path no longer exists: {frame_info['path']}")
        return image_paths, angles, angle_descriptions

    def _ask(self, request, what: str, time_bucket_key: float, **kwargs) -> dict[str, Any] | None:
        """what the frame analyzer answers to `request(**kwargs)`, or None when STOP came first.

        A request that is out cannot be interrupted, and the analyzer can take minutes to answer
        (the VLM behind it is a remote API), so the request goes out on a daemon thread of its
        own while the worker watches the stop event: on STOP the worker gives the request up at
        once, the `what` of that time bucket are dropped, and the request runs out on its own,
        holding no one up. Without STOP this is the plain call: its answer is returned, its
        exception raised.
        """
        answer: dict[str, Any] = {}

        def call():
            try:
                answer['result'] = request(**kwargs)
            except Exception as e:
                answer['error'] = e

        thread = threading.Thread(target=call, daemon=True, name=f'{what}-request')
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0.5)
            if thread.is_alive() and self.stop_event.is_set():
                self.logger.warning(f"STOP came while the frame analyzer was still working on the {what} of time "
                                    f"bucket {time_bucket_key}: they are dropped, and the request is left to run "
                                    f"out on its own")
                return None
        if 'error' in answer:
            raise answer['error']
        return answer.get('result')

    def _process_feature_requests(self):
        """the features worker: every frame set to the features endpoint, for its skeletons, tags,
        head yaws and (when asked) gazes, as fast as the pose model answers."""
        while not self.stop_event.is_set():
            try:
                job = self.features_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                time_bucket_key = job['time_bucket_key']
                image_paths, angles, _ = self._frame_set_paths(job['frames'])
                if not image_paths:
                    self.logger.warning(f"No valid images found for the features of time bucket {time_bucket_key}")
                else:
                    result = self._ask(request_frame_features, 'features', time_bucket_key,
                                       image_paths=image_paths, angles=angles, session_id=self.session_id,
                                       url=self.vllm_frame_analyzer_url, zones=self.feature_zones,
                                       keypoints=self.features_keypoints, gaze=self.gaze)
                    if result:
                        self._note_gaze_errors(result)
                        self._upload_features(time_bucket_key, result)
                        self.logger.info(f"Features of time bucket {time_bucket_key}: "
                                         f"{sum(len(f.get('persons', [])) for f in result.get('frames', []))} persons "
                                         f"in {len(result.get('frames', []))} frames")
                    elif not self.stop_event.is_set():
                        self.logger.warning(f"Received no features for time bucket {time_bucket_key}")
            except Exception as e:
                self.logger.error(f"Error getting the features of a frame set: {e}", exc_info=True)
            finally:
                self._finish(job)
                if not self.stop_event.is_set():
                    self.features_queue.task_done()

    def _note_gaze_errors(self, result: dict):
        """a frame whose gazes the analyzer could not compute says why in gaze_error; the console
        hears each distinct reason once, not every second"""
        seen = getattr(self, '_gaze_errors_seen', None)
        if seen is None:
            seen = self._gaze_errors_seen = set()
        for frame in result.get('frames', []):
            error = frame.get('gaze_error')
            if error and error not in seen:
                seen.add(error)
                self.logger.warning(f"The frame analyzer could not compute the gazes of the {frame.get('angle')} frames "
                                    f"({error}): the features carry skeletons and tags only until it can")

    def _synchronization_handler(self, e: Exception | KeyboardInterrupt | None):
        """Handle exceptions and stop all threads.

        Args:
            e: the exception that occurred during the synchronization process
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped.")
    
        if self.vllm_queue.empty() and self.features_queue.empty():
            self.logger.info("VLLM and features queues processed")
        else:
            self.logger.warning(f"VLLM queue still has {self.vllm_queue.qsize()} items, "
                                f"the features queue {self.features_queue.qsize()}")
        
        clear_directory(self.temp_dir)
        self._clean_up()

    def _process_vllm_requests(self):
        """the action-labels worker: a frame set to the frame analyzer's VLM, at most once per
        action_interval; the answer is the vfa_action event."""
        backlog_said = 0.0
        while not self.stop_event.is_set():
            try:
                job = self.vllm_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                # this worker waits on the VLM: frame sets that come faster than it answers pile
                # up here, and a run at 1 frame set per second can; say so now and then
                backlog = self.vllm_queue.qsize()
                if backlog >= 10 and time.time() - backlog_said > 30:
                    backlog_said = time.time()
                    self.logger.warning(f"{backlog} frame sets wait for action labels: the VLM answers slower than "
                                        f"the bases send (raise Synchronizer.action_interval or Base.keyframe_interval)")
                time_bucket_key = job['time_bucket_key']
                # the labels are due when no interval is set, or the last ones are old enough
                due = (self.action_interval <= 0 or self._last_action_time is None
                       or time_bucket_key - self._last_action_time >= self.action_interval - 1e-6)
                if not due:
                    continue
                image_paths, angles, angle_descriptions = self._frame_set_paths(job['frames'])
                if not image_paths:
                    self.logger.warning(f"No valid images found for time bucket {time_bucket_key}")
                    continue
                self._last_action_time = time_bucket_key
                self.logger.info(f"Requesting multi-angle frame analysis for time bucket {time_bucket_key}: "
                                 f"{len(image_paths)} images from angles {angles} -> {self.vllm_frame_analyzer_url}")
                result = self._ask(request_multi_angle_frame_analyze, 'action labels', time_bucket_key,
                                   image_paths=image_paths, angles=angles, angle_descriptions=angle_descriptions,
                                   session_id=self.session_id, url=self.vllm_frame_analyzer_url,
                                   participant_descriptions=self.selected_participant_descriptions)
                if result:
                    self.logger.info(f"Successfully received analysis result for time bucket {time_bucket_key}")
                    self._upload_result(time_bucket_key, result)
                elif not self.stop_event.is_set():
                    self.logger.warning(f"Received null/empty analysis result for time bucket {time_bucket_key}")
            except Exception as e:
                self.logger.error(f"Error processing frame set: {e}", exc_info=True)
            finally:
                self._finish(job)
                if not self.stop_event.is_set():
                    self.vllm_queue.task_done()

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

    def _upload_features(self, time_bucket_key: float, result: dict[str, Any]):
        """Upload the features of a frame set to InfluxDB as one vfa_features event: the frames
        as the features endpoint answered them (one per angle, each with its persons, tags,
        zones and pairs; see openmmla.services.vfa.features), and the pose model they came from.

        Args:
            time_bucket_key: the start time of the frame set's time bucket
            result: the answer of the features endpoint, {frames, pose_model, gaze}
        """
        if self.session_id is None:
            self.logger.warning(f"Cannot upload features for time bucket {time_bucket_key}: session_id is None")
            return
        from openmmla.utils.constants import EVENT_TYPE_VFA_FEATURES
        frames = result.get('frames') or []
        fields = {
            "window_start_time": time_bucket_key,
            "window_end_time": time_bucket_key,
            "features": json.dumps(frames),
            "pose_model": str(result.get('pose_model') or ''),
            "gaze": 1.0 if result.get('gaze') else 0.0,
        }
        seen = ', '.join(f"{frame.get('angle')}: {', '.join(str(p.get('person_id')) for p in frame.get('persons', []))}"
                         for frame in frames) or 'no one'
        print(f"{BLUE}[Features]{ENDC} {time_bucket_key}: {BLUE}{seen}{ENDC}")
        if not self.influx_client.write_event(self.session_id, EVENT_TYPE_VFA_FEATURES, fields):
            self.logger.error(f"Failed to upload features for time bucket {time_bucket_key}")
        # and to the bases, which draw their own angle's on their live window
        try:
            self.mqtt_client.publish(f'{self.session_id}/vfa/features',
                                     json.dumps({'time': time_bucket_key, 'frames': frames}))
        except Exception as e:
            self.logger.debug(f"Could not publish the features of {time_bucket_key} to the bases: {e}")

    @property
    def session_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current session_id."""
        if self.session_id:
            return f'{self.session_id}/vfa/control'
        return None
