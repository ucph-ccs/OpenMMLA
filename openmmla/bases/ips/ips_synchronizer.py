import gc
import json
import logging
import math
import os
import threading

import numpy as np

from openmmla.bases.synchronizer import Synchronizer
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, runtime_pipeline_artifact_dir
from openmmla.utils import session_provenance
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.config import base_room, is_main_base
from openmmla.utils.input import select_or_create_session, show_error_and_pause
from openmmla.utils.logger import get_logger
from .fusion import DEFAULT_GATE, fuse_bucket, place
from .input import get_bases, get_function_synchronizer

MATRICES_PREFIX = 'transformation_matrices_'


def matrices_file_name(main_id) -> str:
    """The file IPS Camera Sync exports for a main camera: transformation_matrices_<id>.json."""
    return f'{MATRICES_PREFIX}{main_id}.json'


def main_id_of(file_name: str) -> str:
    """The main camera id of a transformation_matrices_<id>.json file (the id may hold underscores)."""
    stem = file_name[:-len('.json')] if file_name.endswith('.json') else file_name
    return stem[len(MATRICES_PREFIX):]


def new_bucket() -> dict:
    """an open time bucket: {'tags': {tag: {camera: [detection]}}, 'votes': {camera: {(a, b): [seen,
    both]}}, 'nicla': {badge: set of tags}} (see fusion.fuse_bucket)."""
    return {'tags': {}, 'votes': {}, 'nicla': {}}


class IPSSynchronizer(Synchronizer):
    """IPSSynchronizer class for synchronizing detection results from multiple cameras under a unified spatial
    coordinate and uploading them to InfluxDB.

    The bases publish the raw pose of every detection. The synchronizer files each into the time
    bucket of its frame (buckets of `bucket_duration` seconds from the first frame it got), keeps a
    bucket open until every base that is still sending has moved past it (a base more than
    `max_lateness` seconds behind the newest frame no longer holds it up), and then fuses the
    cameras' detections of each tag (openmmla.bases.ips.fusion) and writes the bucket. A frame of
    a bucket already written is left out, and counted.
    """
    logger = get_logger('ips-synchronizer')

    def __init__(self, project_dir: str | None, config_path: str, verbose: bool = False,
                 session_id: str | None = None, main_camera: str | None = None):
        """Initialize the IPSSynchronizer class.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            verbose: whether to enable verbose logging (default: False)
            session_id: the session to synchronize; given on the command line (as the console does),
                the synchronizer starts at once and exits when the run ends, instead of showing its menu
            main_camera: the base id whose camera_sync/transformation_matrices_<id>.json to load;
                if omitted, the Bases entry marked main: true, else the only exported file
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.verbose = verbose
        self.launch_session_id = session_id
        self.launch_main_camera = str(main_camera).strip() if main_camera is not None else ''

        # Runtime attributes
        self.main_id = None
        self.transform_matrices_dict = None
        self.session_id = None
        self.allowed_tag_ids = None
        self.unregistered_tag_ids = set()  # detected tags this session has no individual for, logged once each
        self.unplaced_base_ids = set()  # bases the main camera's matrices do not place, logged once each
        self._reset_buckets()
        self.alive = False

        # Threading attributes
        self.threads = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        sync_config = self.config['Synchronizer']
        self.bucket_duration = float(sync_config['bucket_duration'])
        # how far (seconds of frame time) a base may fall behind the newest frame before the buckets are
        # written without it; a file replay at processing_rate 4 runs four of them per second of its own
        self.max_lateness = float(sync_config.get('max_lateness', 5.0))
        # how far (metres) a camera's position of a tag may lie from the cameras' median and still be averaged
        self.fusion_gate = float(sync_config.get('fusion_gate', DEFAULT_GATE))

    def _reset_buckets(self):
        """Forget the open buckets and the bases' progress, for a new run."""
        self.buckets = {}  # bucket start -> new_bucket()
        self.bucket_origin = None  # the first frame's time: buckets start at it plus whole bucket durations
        self.newest_time = None  # the newest frame any base sent
        self.base_last = {}  # base id -> the newest frame it sent
        self.written_until = None  # the end of the newest bucket written
        self.late_frames = {}  # base id -> frames that came after their bucket was written

    def _setup_directories(self):
        """Set up directories."""
        self.logger_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'ips-base', 'logger'))
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
        self.unregistered_tag_ids = set()
        self.unplaced_base_ids = set()
        self._reset_buckets()
        gc.collect()

    def _close_clients(self):
        """Close the connections before the process exits."""
        for close in (self.mqtt_client.disconnect, self.influx_client.close, self.mongo_client.close,
                      self.redis_client.close):
            try:
                close()
            except Exception as e:
                self.logger.debug(f"While closing a client on exit: {e}")

    def run(self):
        """Run the IPS synchronizer.

        Launched from the console (a session id on the command line), it takes its main camera from
        -mc (or the default), synchronizes at once and exits when the run ends. Run by hand, or when
        that run cannot start, it shows its menu.
        """
        print('\033]0;IPS Synchronizer\007')
        if self.launch_session_id:
            if self._run_from_console():
                return
        elif self.launch_main_camera:
            problem = self._use_main_camera()
            if problem:
                print(f"\n{problem}\n")
        self._run_menu()

    def _run_from_console(self) -> bool:
        """Synchronize the session given on the command line at once, then end.

        Returns True once the run has ended (STOP, also before START, or Ctrl+C), and the process
        exits; False when it could not start or ended on an error: why is printed, and the menu
        follows so that it can be fixed in this window.
        """
        problem = self._use_main_camera()
        if problem:
            print(f"\n{problem}\n"
                  f"The IPS Synchronizer menu follows: choose 2 to pick the main camera, then 1 to start.\n")
            return False
        try:
            ended = self._start_synchronization()
        except KeyboardInterrupt:
            self.logger.info("IPS synchronizer stopped with Ctrl+C.")
            ended = True
        except Exception as e:
            self.logger.warning(f"IPS synchronizer could not start session {self.launch_session_id}: {e}",
                                exc_info=True)
            show_error_and_pause(e, "open the IPS Synchronizer menu and try again")
            return False
        finally:
            self._clean_up()
        if not ended:
            print(f"\nThe synchronization of session {self.launch_session_id} stopped on an error (see above), "
                  f"not on STOP. Check that Redis and MQTT are running (System Services on the console).\n"
                  f"The IPS Synchronizer menu follows: choose 1 to start it again.\n")
            return False
        self.logger.info(f"Synchronization of session {self.launch_session_id} ended, exiting IPS synchronizer.")
        self._close_clients()
        return True

    def _run_menu(self):
        """Run the IPS synchronizer's interactive menu until the user exits it."""
        func_map = {1: self._start_synchronization, 2: self._set_main_camera, }
        ended_from_console = False

        while True:
            try:
                select_fun = get_function_synchronizer(self.main_id)
                if select_fun == 0:
                    self.logger.info("Exiting IPS synchronizer...")
                    break
                ended = func_map.get(select_fun, lambda: print("Invalid option."))()
                if select_fun == 1 and ended is True and self.launch_session_id:
                    # launched from the console, it came to this menu to fix something: a run
                    # started from here ends the process on STOP all the same
                    self.logger.info(f"Synchronization of session {self.launch_session_id} ended, "
                                     f"exiting IPS synchronizer.")
                    ended_from_console = True
                    break
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"During running the synchronizer, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)
                if not isinstance(e, KeyboardInterrupt):
                    show_error_and_pause(e, "return to the IPS Synchronizer menu")
            finally:
                self._clean_up()
        if ended_from_console:
            self._close_clients()

    def _start_synchronization(self) -> bool | None:
        """Start the synchronization process.

        Returns:
            True when the run ended with STOP (also STOP before START, when nothing was
            synchronized) or Ctrl+C; False when it ended on an error, such as the connection to
            Redis lost; None when it did not start (no main camera: it is picked instead).
        """
        if self.transform_matrices_dict is None:
            self.logger.warning("Main camera id or transformation matrices not set, please set them first.")
            return self._set_main_camera()

        self._reset_buckets()

        # select or create bucket
        self.session_id = self.launch_session_id or select_or_create_session(self.mongo_client)
        self._create_bucket_logger()
        self._resolve_session_tag_filter()
        self._record_provenance()
        if not self._listen_for_start_signal():
            self._clean_up()  # STOP came before START: nothing was started
            return True

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
        # Ctrl+C ends the run as STOP does; any other exception is an error
        return exception_occurred is None or isinstance(exception_occurred, KeyboardInterrupt)

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

    def _record_provenance(self):
        """Note in the session what this synchronizer runs with (openmmla.utils.session_provenance):
        the main camera and its transformation matrices (the file copied next to the config too),
        the bucket duration, the tags it keeps. A failure is a warning, never a stop."""
        if not self.session_id:
            return
        try:
            matrices = None
            if self.main_id is not None:
                file_name = matrices_file_name(self.main_id)
                matrices = {'file': file_name, 'main_id': self.main_id, 'matrices': self.transform_matrices_dict}
                copy_config_snapshot(os.path.join(self.camera_sync_dir, file_name),
                                     self.project_dir, self.session_id, 'ips-base')
            entry = session_provenance.component_entry(
                'ips', 'synchronizer',
                arguments={'verbose': self.verbose, 'session_id': self.launch_session_id,
                           'main_camera': self.launch_main_camera or None},
                parameters={'main_id': self.main_id, 'bucket_duration': self.bucket_duration,
                            'max_lateness': self.max_lateness, 'fusion_gate': self.fusion_gate,
                            'stored_poses': 'raw detections, cameras fused per bucket',
                            'allowed_tag_ids': None if self.allowed_tag_ids is None else sorted(
                                self.allowed_tag_ids, key=str)},
                files={'transformation_matrices': matrices},
                config=self.config, config_path=self.config_path, project_dir=self.project_dir)
            session_provenance.record_component(self.mongo_client, self.session_id, entry, self.project_dir,
                                                'ips-base', log=self.logger)
        except Exception as e:
            self.logger.warning(f"Could not note in session {self.session_id} what the IPS synchronizer runs with: {e}")

    def _resolve_session_tag_filter(self):
        """Limit IPS aggregation to tag ids assigned to the selected session group."""
        self.allowed_tag_ids = None
        self.unregistered_tag_ids = set()
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
        if self.allowed_tag_ids is None:
            return True
        text = str(tag_id)
        if text in self.allowed_tag_ids:
            return True
        if text not in self.unregistered_tag_ids:
            # a tag the cameras see but no individual carries is dropped; say so once, or the
            # session looks like it detected nothing at all
            self.unregistered_tag_ids.add(text)
            self.logger.warning(
                "Tag %s is not carried by any individual of this session (tags %s), so it is left out "
                "of the aggregation; give an individual this tag id to include it.",
                text, sorted(self.allowed_tag_ids))
        return False

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
        self._write_remaining_buckets()
        self._clean_up()

    def _exported_matrices(self) -> list[str]:
        """The transformation_matrices_<id>.json files in camera_sync/, sorted by name."""
        try:
            names = os.listdir(self.camera_sync_dir)
        except OSError:
            return []
        return sorted(name for name in names
                      if name.startswith(MATRICES_PREFIX) and name.endswith('.json')
                      and os.path.isfile(os.path.join(self.camera_sync_dir, name)))

    def _resolve_main_camera(self) -> tuple[str | None, str]:
        """The main camera a run takes without asking: (id, "") or (None, why there is none).

        -mc names it. Without -mc: the Bases entry marked main: true when its transformation file is
        there, else the only transformation file in camera_sync/.
        """
        exported = self._exported_matrices()
        listed = ', '.join(exported)
        if self.launch_main_camera:
            main_id = self.launch_main_camera
            if matrices_file_name(main_id) in exported:
                return main_id, ''
            there = f"camera_sync holds {listed}" if exported else "camera_sync holds no transformation files"
            return None, (f"There is no camera_sync/{matrices_file_name(main_id)} for main camera {main_id} in "
                          f"{self.project_dir} ({there}): run IPS Camera Sync and export the transformations, "
                          f"or pick another main camera on the IPS Base card.")

        main_bases = [base for base in get_bases(self.config) if is_main_base(base)]
        mains = [str(base.get('id')) for base in main_bases]
        with_file = [main_id for main_id in mains if matrices_file_name(main_id) in exported]
        if len(mains) > 1:
            # one room's file being there does not make the session that room's
            rooms = ', '.join(f"{base.get('id')} (room {base_room(base) or '-'})" for base in main_bases)
            return None, (f"The IPS synchronizer has no main camera: the Bases have a main base per room ({rooms}), "
                          f"and a session is one room's. Give it the main camera of that room (-mc, the Main Camera "
                          f"of the IPS Base card).")
        if with_file:
            return with_file[0], ''
        if len(exported) == 1 and not any(base_room(base) for base in get_bases(self.config)):
            # without rooms the only file is the one camera sync made; with rooms it may be another room's
            return main_id_of(exported[0]), ''
        wanted = (f"the main base {mains[0]} has no camera_sync/{matrices_file_name(mains[0])}" if mains
                  else "no Bases entry is marked main: true")
        if exported:
            return None, (f"The IPS synchronizer has no main camera: {wanted}, and camera_sync holds "
                          f"{len(exported)} transformation files ({listed}). Pick the main camera on the "
                          f"IPS Base card.")
        return None, (f"The IPS synchronizer has no main camera: {wanted}, and camera_sync in {self.project_dir} "
                      f"holds no transformation files. Run IPS Camera Sync and export the transformations first.")

    def _use_main_camera(self) -> str:
        """Load the main camera the process was launched with (-mc, or the default).

        Returns "" once its transformation matrices are loaded, else why they are not, in words the
        user can act on.
        """
        main_id, problem = self._resolve_main_camera()
        if main_id is None:
            return problem
        file_name = matrices_file_name(main_id)
        try:
            self._set_main_camera(main_id)
        except (OSError, ValueError) as e:
            return (f"camera_sync/{file_name} could not be read ({e}): export the transformations again from "
                    f"IPS Camera Sync, or pick another main camera on the IPS Base card.")
        if self.transform_matrices_dict is None:
            return (f"camera_sync/{file_name} is gone: run IPS Camera Sync and export the transformations "
                    f"again, or pick another main camera on the IPS Base card.")
        self.logger.info(f"Main camera {main_id}: loaded camera_sync/{file_name}.")
        return ''

    def _set_main_camera(self, main_id: str | None = None):
        """Set the main camera id and load its transformation matrices.

        Args:
            main_id: the main camera id to load; if None, the user picks a file from camera_sync/
        """
        self.transform_matrices_dict = self._load_transform_matrices(main_id)
        if self.transform_matrices_dict is None:
            self.logger.warning("No transformation matrices found, please check your main camera id or do the camera "
                                "sync first.")

    def _handle_base_result(self, client, userdata, msg):
        """Handle the received base result: file its detections into the bucket of its frame, and
        write the buckets every base still sending has moved past.

        Args:
            client: the client instance for this callback
            userdata: the private user data as a set in Client() or user_data_set()
            message: an instance of MQTTMessage
        """
        if self.stop_event.is_set():
            return
        self.alive = True
        base_result = json.loads(msg.payload)
        base_id = str(base_result["base_id"])
        base_result_time = float(base_result["acquired_time"])
        nicla = base_id.isnumeric() and int(base_id) > 50000  # nicla vision's onboard apriltag detection (if used)

        if not nicla and base_id != self.main_id and base_id not in self.transform_matrices_dict:
            # another room's base (or one never synced to this main): its tags are in a frame
            # this session cannot place, and its relations are about another room's people
            if base_id not in self.unplaced_base_ids:
                self.unplaced_base_ids.add(base_id)
                self.logger.warning(
                    f"Base {base_id} has no matrix in camera_sync/{matrices_file_name(self.main_id)}: "
                    f"it is in another room, or was not synced to main camera {self.main_id}. Its "
                    f"detections are left out of this session.")
            return

        with self.lock:
            # a badge's own clock is not the cameras': its frames join the buckets, but never decide
            # how far the run has got
            bucket = self._bucket_of(base_id, base_result_time, track=not nicla)
            if bucket is not None:
                if nicla:
                    if self._tag_allowed(base_id):
                        detected = [tag_id for tag_id in base_result.get('detected_tags') or []
                                    if self._tag_allowed(tag_id)]
                        bucket['nicla'].setdefault(base_id, set()).update(detected)
                else:
                    self._add_detections(bucket, base_id, base_result_time, base_result)
            self._write_ready_buckets()

    def _bucket_of(self, base_id: str, frame_time: float, track: bool = True) -> dict | None:
        """The open bucket a base's frame belongs to, noting how far the base has got (`track`);
        None when that bucket was written already (the frame is counted, and the base named once)."""
        if self.bucket_origin is None:
            self.bucket_origin = frame_time
            self.logger.info(f"First frame at {frame_time:.2f}s: buckets of {self.bucket_duration:g}s from there.")
        previous = self.base_last.get(base_id)  # the base's newest frame before this one
        if track:
            self.base_last[base_id] = max(self.base_last.get(base_id, frame_time), frame_time)
            self.newest_time = frame_time if self.newest_time is None else max(self.newest_time, frame_time)
        index = math.floor((frame_time - self.bucket_origin) / self.bucket_duration + 1e-6)
        start = self.bucket_origin + index * self.bucket_duration
        if self.written_until is not None and start < self.written_until - 1e-6:
            self.late_frames[base_id] = self.late_frames.get(base_id, 0) + 1
            if self.late_frames[base_id] == 1:
                behind = max(0.0, self.newest_time - frame_time) if self.newest_time is not None else 0.0
                if not track:
                    why = "a badge's frames never hold the buckets up"
                elif previous is None:
                    why = "the base joined after that bucket was written"
                elif frame_time < previous:
                    why = f"the frame is older than one the base sent before ({previous:.2f}s)"
                else:
                    why = (f"the base was more than {self.max_lateness:g}s behind the newest frame when the "
                           f"bucket was written")
                self.logger.warning(
                    f"Base {base_id} sent a frame of {frame_time:.2f}s, {behind:.1f}s behind the newest frame, "
                    f"after its bucket was written: {why}. Its late frames are left out (counted, the count is "
                    f"logged at the end).")
            return None
        return self.buckets.setdefault(start, new_bucket())

    def _add_detections(self, bucket: dict, base_id: str, frame_time: float, base_result: dict):
        """File a camera frame's raw detections into a bucket, in the main camera's frame, and count
        on this camera's frames who faced whom."""
        tags = self._filter_tags(base_result.get("tags") or {})
        relations = self._filter_relations(base_result.get("tag_relations") or {})
        quality = base_result.get("quality") or {}
        matrix = None if base_id == self.main_id else self.transform_matrices_dict[base_id]
        for tag_id, (rotation, translation) in tags.items():
            tag_id = str(tag_id)
            # the tag's distance from the camera that saw it, which weighs the cameras against each other
            distance = float(np.linalg.norm(np.asarray(translation, dtype=float)))
            R, t = place(rotation, translation, matrix)
            margin = (quality.get(tag_id) or {}).get('margin')
            bucket['tags'].setdefault(tag_id, {}).setdefault(base_id, []).append(
                {'t': t, 'R': R, 'd': distance, 'm': margin, 'time': frame_time})
        seen = [str(tag_id) for tag_id in tags]
        faced = {str(tag_id): {str(target) for target in targets} for tag_id, targets in relations.items()}
        votes = bucket['votes'].setdefault(base_id, {})
        for a in seen:
            for b in seen:
                if a != b:
                    count = votes.setdefault((a, b), [0, 0])
                    count[0] += b in faced.get(a, ())
                    count[1] += 1

    def _bucket_done(self, start: float) -> bool:
        """Whether every base still sending has moved past the bucket: a base whose newest frame is
        more than max_lateness behind the newest of all no longer holds it up. For the first
        max_lateness seconds a base the matrices place that has sent nothing yet holds it up too,
        so that a base that starts a little later keeps its first frames."""
        if self.newest_time is None:
            return True
        if self.newest_time - self.bucket_origin <= self.max_lateness:
            expected = {str(self.main_id), *(str(base_id) for base_id in (self.transform_matrices_dict or {}))}
            if any(base_id not in self.base_last for base_id in expected):
                return False
        end = start + self.bucket_duration
        return all(last >= end - 1e-6 or last < self.newest_time - self.max_lateness
                   for last in self.base_last.values())

    def _write_ready_buckets(self):
        """Write the open buckets, oldest first, as long as the oldest is done."""
        while self.buckets:
            start = min(self.buckets)
            if not self._bucket_done(start):
                break
            self._upload_bucket(start, self.buckets.pop(start))

    def _write_remaining_buckets(self):
        """Write every bucket still open, at the end of the run, and say how many late frames were
        left out."""
        with self.lock:
            for start in sorted(self.buckets):
                try:
                    self._upload_bucket(start, self.buckets.pop(start))
                except Exception as e:
                    self.logger.warning(f"Could not write the bucket at {start:.2f}s at the end of the run: {e}")
            if self.late_frames:
                self.logger.warning("Frames left out for coming after their bucket was written: " + ', '.join(
                    f"base {base_id}: {count}" for base_id, count in sorted(self.late_frames.items())))

    def _upload_bucket(self, start: float, bucket: dict):
        """Upload a time bucket: the cameras' detections fused per tag (positions and rotations in
        the main camera's frame), who faced whom, and every camera's own detection."""
        from openmmla.utils.constants import (
            EVENT_TYPE_IPS_TRANSLATION, EVENT_TYPE_IPS_ROTATION, EVENT_TYPE_IPS_RELATION,
        )

        end = start + self.bucket_duration
        translations_dict, rotations_dict, relations_dict, detections = fuse_bucket(bucket, start, self.fusion_gate)

        translation_fields = {
            "window_start_time": start,
            "window_end_time": end,
            "translations": json.dumps(translations_dict),
            "detections": json.dumps(detections),
        }
        rotation_fields = {
            "window_start_time": start,
            "window_end_time": end,
            "rotations": json.dumps(rotations_dict),
        }
        relation_fields = {
            "window_start_time": start,
            "window_end_time": end,
            "graph": json.dumps(relations_dict),
        }

        self.logger.debug(translation_fields)
        self.logger.debug(rotation_fields)
        self.logger.debug(relation_fields)

        self.influx_client.write_event(self.session_id, EVENT_TYPE_IPS_TRANSLATION, translation_fields)
        self.influx_client.write_event(self.session_id, EVENT_TYPE_IPS_ROTATION, rotation_fields)
        self.influx_client.write_event(self.session_id, EVENT_TYPE_IPS_RELATION, relation_fields)
        self.written_until = end if self.written_until is None else max(self.written_until, end)

        cameras = {camera for by_camera in detections.values() for camera in by_camera}
        self.logger.info(f"Uploaded bucket: {start:.2f}s - {end:.2f}s ({len(translations_dict)} tags from "
                         f"{len(cameras)} cameras, {len(relations_dict)} relations)")

    def _load_transform_matrices(self, main_id: str | None = None):
        """Load transformation matrices.

        Args:
            main_id: the main camera id whose camera_sync/transformation_matrices_<id>.json to load
                (the result is None when that file is not there); if None, list the files there and
                ask for one
        """
        if main_id is not None:
            path = os.path.join(self.camera_sync_dir, matrices_file_name(main_id))
            if not os.path.isfile(path):
                return None
            with open(path, 'r') as file:
                matrices = json.load(file)
            self.main_id = str(main_id)
            return matrices

        transformation_choices = self._exported_matrices()
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
                    self.main_id = main_id_of(chosen_transformation)
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
