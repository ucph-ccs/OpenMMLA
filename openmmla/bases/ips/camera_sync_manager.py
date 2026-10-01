import gc
import json
import os
import queue
import time
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from openmmla.bases.synchronizer import Synchronizer
from openmmla.utils.client import MQTTClientWrapper
from openmmla.utils.input import show_error_and_pause
from openmmla.utils.logger import get_logger
from openmmla.utils.config import base_room, bases_by_room, camera_sync_problem, is_main_base

from .input import get_base_by_id, get_bases, get_function_sync_manager
from .transform import (
    direct_transform_matrices, average_transform_matrices,
    transform_point, distance_between_points, transform_rotation,
    distance_between_rotations, export_main_transformations_json
)

# the sync plots with Tk, chosen where it plots (_start_synchronization): set
# here, on import, it switched every importer's matplotlib to Tk too
PLOT_BACKEND = 'TkAgg'


class CameraSyncManager(Synchronizer):
    """Class for synchronizing the coordinate system between the main camera and the alternative camera."""
    logger = get_logger('camera-sync-manager')
    colors = cycle('bgrcmyk')

    def __init__(self, project_dir: str | None, config_path: str, sync: bool = True,
                 top_K: int = 5000, distance_threshold: float = 0.1, angle_threshold: float = 3.0,
                 time_threshold_sync: float = 0.2, time_threshold_unsync: float = 0.2,
                 base: str | None = None, main: str | None = None):
        """Initialize the camera sync manager.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            sync: whether to synchronize cameras (default: True)
            top_K: maximum number of matrices to keep (default: 5000)
            distance_threshold: threshold for point distance (default: 0.1)
            angle_threshold: threshold for rotation angle (default: 3.0)
            time_threshold_sync: seconds apart the main and the alternative detection of a tag
                may reach this manager and still be paired, in sync mode (default: 0.2)
            time_threshold_unsync: the same in unsync (validation) mode (default: 0.2)
            base: alternative base id (from config 'Bases') to synchronize against
                the main base of its room (the base of that room flagged ``main: true``).
                If omitted, the alternative is picked interactively.
            main: the main base to synchronize against when the config has a main per
                room; if omitted, the main of the room of `base`, the only main there
                is, or picked interactively.
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.launch_base = base
        self.launch_main = main

        """Synchronization parameters."""
        self.sync = sync
        self.top_K = top_K
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.time_threshold_sync = time_threshold_sync
        self.time_threshold_unsync = time_threshold_unsync
        self.time_threshold = self.time_threshold_sync if self.sync else self.time_threshold_unsync

        """Runtime attributes."""
        self.main_id = None
        self.alt_id = None
        self.main_rotations = {}
        self.main_translations = {}
        self.main_last_update_time = {}
        self.alt_rotations = {}
        self.alt_translations = {}
        self.alt_last_update_time = {}
        self.rotation_matrices = []
        self.translations_matrices = []
        self.retrieved_R = None
        self.retrieved_T = None
        self.fig = None
        self.ax = None
        self.anim = None
        self.plot_points_queue = queue.Queue()
        self.plot_points_list = []
        self.tag_color_map = {}

        self._setup_directories()
        self._setup_objects()

        # profile-driven: resolve main/alt base ids from the config 'Bases' list
        # (main = the entry of a room flagged main: true; alt = -b entry, else picked).
        self._resolve_bases(base, main)

    def _resolve_bases(self, base, main=None):
        """Resolve main_id and alt_id: the main base of a room (the room of -b, -m,
        the only main there is, or picked) and another base of that room (-b or picked)."""
        problem = camera_sync_problem(self.config)
        if problem:
            raise ValueError(f"{problem} (config: {self.config_path})")
        bases = get_bases(self.config)
        mains = [b for b in bases if is_main_base(b)]
        alt = None
        if base is not None:
            alt = get_base_by_id(self.config, base)
            if alt is None or is_main_base(alt):
                raise ValueError(f"Alternative base '{base}' not found (or is a main base) in config 'Bases'.")
            # camera_sync_problem saw to it that every room has one main
            main_entry = next(m for m in mains if base_room(m) == base_room(alt))
            if main is not None and str(main) != str(main_entry.get('id')):
                raise ValueError(f"Base '{base}' is in room {base_room(alt) or '-'}, whose main base is "
                                 f"'{main_entry.get('id')}', not '{main}'.")
        elif main is not None:
            main_entry = get_base_by_id(self.config, main)
            if main_entry is None or not is_main_base(main_entry):
                raise ValueError(f"Main base '{main}' not found (or not marked main: true) in config 'Bases'.")
        elif len(mains) == 1:
            main_entry = mains[0]
        else:
            main_entry = self._pick_interactively(
                mains, "The config has a main base per room. Select the main base to synchronize to:",
                "Main base number", lambda b: f"id={b.get('id')} (room: {base_room(b)}, camera: {b.get('camera')})")
        self.main_id = str(main_entry.get('id'))
        room = base_room(main_entry)

        if alt is None:
            alts = [b for b in bases if base_room(b) == room and not is_main_base(b)]
            if not alts:
                raise ValueError(f"Room {room} has its main base '{self.main_id}' alone: give another base "
                                 f"room: {room} to sync it to this one.")
            alt = alts[0] if len(alts) == 1 else self._pick_interactively(
                alts, f"Main base is '{self.main_id}'. Select the alternative base to synchronize:",
                "Alternative base number", lambda b: f"id={b.get('id')} (camera: {b.get('camera')})")
        self.alt_id = str(alt.get('id'))

        where = f" (room {room})" if room else ""
        self.logger.info(f"Camera sync: main={self.main_id}, alternative={self.alt_id}{where}")
        print(f"\033]0;Camera Sync Manager: main:{self.main_id} - alternative:{self.alt_id}\007")

    @staticmethod
    def _pick_interactively(entries, title, prompt, describe):
        """Pick one of `entries` by its number (the only interaction)."""
        print(title)
        for idx, b in enumerate(entries):
            print(f"  {idx}: {describe(b)}")
        while True:
            sel = input(f"{prompt} [0]: ").strip()
            try:
                index = int(sel) if sel else 0
            except ValueError:
                index = -1
            if 0 <= index < len(entries):
                return entries[index]
            print("Invalid selection. Please enter a valid base number.")

    def _setup_directories(self):
        """Set up required directories."""
        self.camera_sync_dir = os.path.join(self.project_dir, 'camera_sync')
        os.makedirs(self.camera_sync_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.mqtt_client = MQTTClientWrapper(self.config_path)

    def run(self):
        """Run the camera sync manager."""
        func_map = {1: self._start_synchronization, 2: self._set_camera_id, 3: self._switch,
                    4: self._export_transformations, 5: self._clear_transformations}
        while True:
            try:
                select_fun = get_function_sync_manager(self.main_id, self.alt_id, self.sync, self._clear_scope())
                if select_fun == 0:
                    self.logger.info("Exiting video synchronizer...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning("%s, Come back to the main menu.", e, exc_info=True)
                if not isinstance(e, KeyboardInterrupt):
                    show_error_and_pause(e, "return to the Camera Sync Manager menu")

    def _set_camera_id(self):
        """Re-resolve main/alt camera IDs from the config 'Bases' list (no input)."""
        self._resolve_bases(self.launch_base, self.launch_main)

    def _start_synchronization(self):
        try:
            if not self.main_id or not self.alt_id:
                return self._set_camera_id()

            self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics="camera/synchronize")
            self.mqtt_client.loop_start()
            matplotlib.use(PLOT_BACKEND)
            self.fig, self.ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
            plt.get_current_fig_manager().set_window_title(
                f"Camera Sync Manager: main:{self.main_id} - alternative:{self.alt_id}")
            self.anim = FuncAnimation(self.fig, self._animate_plot, interval=10, cache_frame_data=False)
            plt.show()
        except KeyboardInterrupt:
            self.logger.info("Ctrl+C, exiting video synchronizer...")
        finally:
            self.mqtt_client.loop_stop()
            self._clean_up()

    def _switch(self):
        self.sync = not self.sync
        self.time_threshold = self.time_threshold_sync if self.sync else self.time_threshold_unsync
        print(f"Switching the synchronization mode to {self.sync} with time threshold {self.time_threshold}.")

    def _clean_up(self):
        """Reinitialize the variables and plot based on sync mode."""
        self.main_translations, self.main_rotations, self.main_last_update_time = {}, {}, {}
        self.alt_translations, self.alt_rotations, self.alt_last_update_time = {}, {}, {}
        self.rotation_matrices, self.translations_matrices = [], []
        self.retrieved_R, self.retrieved_T = None, None
        self.plot_points_queue = queue.Queue()
        self.plot_points_list = []
        self.tag_color_map = {}

        if self.fig:
            plt.close(self.fig)

        gc.collect()

    def _animate_plot(self, i):
        """Update the plot based on points data queue."""
        self.ax.clear()
        try:
            plot_points_list = self.plot_points_queue.get_nowait()
            for main_point, transformed_point, color, tag in plot_points_list:
                self.ax.scatter(main_point[:, 0], main_point[:, 1], main_point[:, 2], c=color, marker='o',
                                label=f'Main Camera (Tag {tag})')
                self.ax.scatter(transformed_point[:, 0], transformed_point[:, 1], transformed_point[:, 2], c=color,
                                marker='x',
                                label=f'Alternative Camera (Transformed, Tag {tag})')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.legend()
        except queue.Empty:
            pass

    def _handle_base_result(self, client, userdata, msg):
        message = json.loads(msg.payload)

        base_id = message["base_id"]
        tags = message["tags"]
        # a detection counts as fresh by when it reached this manager, not by the
        # acquired_time it carries: that stamp is in the capture side's clock, and
        # the stream, the detection and the MQTT round trip through the broker put
        # it behind this clock by the whole pipeline's delay (over 50 ms already
        # when the broker is a relayed hop away), so compared with it nothing was
        # ever fresh. Both cameras take the same way here, so the time between
        # their arrivals is what tells whether their frames show the same moment.
        current_time = time.monotonic()

        # Remove outdated positions for main camera
        for tag in list(self.main_translations.keys()):
            if (current_time - self.main_last_update_time.get(tag, 0)) > self.time_threshold:
                del self.main_translations[tag]
                del self.main_last_update_time[tag]

        # Remove outdated positions for alternative camera
        for tag in list(self.alt_translations.keys()):
            if (current_time - self.alt_last_update_time.get(tag, 0)) > self.time_threshold:
                del self.alt_translations[tag]
                del self.alt_last_update_time[tag]

        for tag_id, tag_data in tags.items():
            rotation = tag_data[0]
            position = tag_data[1]

            if base_id == self.main_id:
                self.main_translations[tag_id] = position
                self.main_rotations[tag_id] = rotation
                self.main_last_update_time[tag_id] = current_time
            elif base_id == self.alt_id:
                self.alt_translations[tag_id] = position
                self.alt_rotations[tag_id] = rotation
                self.alt_last_update_time[tag_id] = current_time

            if tag_id in self.main_translations and tag_id in self.alt_translations:
                color = self.tag_color_map.setdefault(tag_id, next(self.colors))
                if self.sync:
                    self._synchronize(tag_id, color)
                else:
                    self._validate(tag_id, color)

        if self.plot_points_list:
            self.plot_points_queue.put(self.plot_points_list)
            self.plot_points_list = []

    def _synchronize(self, tag, color):
        R, T = direct_transform_matrices(self.main_rotations[tag], self.main_translations[tag],
                                         self.alt_rotations[tag], self.alt_translations[tag])
        self.rotation_matrices.append(R)
        self.translations_matrices.append(T)
        self.rotation_matrices = self.rotation_matrices[-self.top_K:]
        self.translations_matrices = self.translations_matrices[-self.top_K:]

        R_avg, T_avg = average_transform_matrices(self.rotation_matrices, self.translations_matrices)
        print(f"R matrix for tag {tag}: {R_avg}, T vector for tag {tag}: {T_avg}")

        transformed_point = transform_point(self.alt_translations[tag], R_avg, T_avg)
        transformed_rotation = transform_rotation(R_avg, self.alt_rotations[tag])
        point_distance = distance_between_points(transformed_point, self.main_translations[tag])
        rotation_distance = distance_between_rotations(transformed_rotation, self.main_rotations[tag])

        print(
            f"Main position for tag {tag}: {self.main_translations[tag]}, "
            f"alternative position for tag {tag}: {self.alt_translations[tag]}, "
            f"converted position for tag {tag}: {transformed_point}")
        print(f"Distances for tag {tag}: {point_distance} m, angles: {rotation_distance} °")

        if point_distance > self.distance_threshold or rotation_distance > self.angle_threshold:
            print(f"Outlier detected for tag {tag}. Rolling back to last valid transform matrices.")
            self.rotation_matrices.pop(-1)
            self.translations_matrices.pop(-1)
        else:
            self._update_json_file(R_avg, T_avg)

        self._add_points_to_list(tag, color, transformed_point)

    def _validate(self, tag, color):
        if self.retrieved_R is not None and self.retrieved_T is not None:
            transformed_point = transform_point(self.alt_translations[tag], self.retrieved_R, self.retrieved_T)
            transformed_rotation = transform_rotation(self.retrieved_R, self.alt_rotations[tag])
            point_distance = distance_between_points(transformed_point, self.main_translations[tag])
            rotation_distance = distance_between_rotations(transformed_rotation, self.main_rotations[tag])

            print(
                f"Main position for tag {tag}: {self.main_translations[tag]}, "
                f"alternative position for tag {tag}: {self.alt_translations[tag]},"
                f"converted position for tag {tag}: {transformed_point}")
            print(f"Distances for tag {tag}: {point_distance} m, angles: {rotation_distance} °")

            self._add_points_to_list(tag, color, transformed_point)
        else:
            with open(os.path.join(self.camera_sync_dir, 'transformation_matrices.json'), 'r') as file:
                data = json.load(file)
            key = f'{self.alt_id}-{self.main_id}'
            self.retrieved_R, self.retrieved_T = np.array(data[key]['R']), np.array(
                data[key]['T'])

    def _add_points_to_list(self, tag, color, transformed_point):
        main_point = np.array(self.main_translations[tag]).reshape(1, 3)
        transformed_point = np.array(transformed_point).reshape(1, 3)
        self.plot_points_list.append((main_point, transformed_point, color, tag))

    def _update_json_file(self, R, T):
        json_file_path = os.path.join(self.camera_sync_dir, 'transformation_matrices.json')
        data = None
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        if data is None:
            data = {}

        key = f'{self.alt_id}-{self.main_id}'
        data[key] = {"R": R, "T": T}
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _export_transformations(self):
        # the "main system" is the main base (flagged main: true) resolved at init
        main_system = self.main_id
        if not main_system:
            self.logger.warning("No main base resolved; cannot export transformations.")
            return
        input_path = os.path.join(self.camera_sync_dir, 'transformation_matrices.json')
        output_path = os.path.join(self.camera_sync_dir, f'transformation_matrices_{main_system}.json')
        export_main_transformations_json(input_path, output_path, main_system)

    def _room(self) -> str | None:
        """the room clear works on: None when the config names no room (all of it is
        one), else the main base's room, "" for the bases that name none beside rooms."""
        if list(bases_by_room(self.config)) == [""]:
            return None
        main = get_base_by_id(self.config, self.main_id) if self.main_id else None
        return base_room(main) if main else None

    def _clear_scope(self) -> str:
        """what clear removes, as the menu says it: "" for everything."""
        room = self._room()
        return "" if room is None else f"room {room}" if room else "the bases without a room"

    def _clear_transformations(self):
        """Remove the matrices of the main base's room. A config without rooms is one
        room, so every transformation_matrices*.json goes; with rooms, only the pairs
        that involve one of its bases and the transformation_matrices_<id>.json of its
        bases go, and the other rooms keep theirs (their cameras were not re-synced)."""
        room = self._room()
        if room is None:
            for file in os.listdir(self.camera_sync_dir):
                if file.startswith("transformation_matrices"):
                    os.remove(os.path.join(self.camera_sync_dir, file))
                    print(f"File {file} has been removed.")
            return

        scope = self._clear_scope()
        ids = {str(base.get('id')) for base in bases_by_room(self.config).get(room, [])}
        for base_id in sorted(ids):
            exported = f'transformation_matrices_{base_id}.json'
            if os.path.exists(os.path.join(self.camera_sync_dir, exported)):
                os.remove(os.path.join(self.camera_sync_dir, exported))
                print(f"File {exported} of {scope} has been removed.")

        pairs_path = os.path.join(self.camera_sync_dir, 'transformation_matrices.json')
        if not os.path.exists(pairs_path):
            return
        try:
            with open(pairs_path, 'r') as file:
                pairs = json.load(file)
        except (OSError, ValueError):
            pairs = None
        if not isinstance(pairs, dict):
            # cut short by a sync killed while writing it: no room's pairs can be read from it
            os.remove(pairs_path)
            print("transformation_matrices.json could not be read and has been removed (every room's pairs).")
            return
        # a pair is `<alt>-<main>`; ids may hold dashes, so match the ends against the room's ids
        kept = {key: value for key, value in pairs.items()
                if not any(key.startswith(f'{i}-') or key.endswith(f'-{i}') for i in ids)}
        removed = [key for key in pairs if key not in kept]
        if kept:
            with open(pairs_path, 'w') as file:
                json.dump(kept, file, indent=4)
        else:
            os.remove(pairs_path)
        if removed:
            print(f"Pairs {', '.join(removed)} of {scope} have been removed from transformation_matrices.json"
                  + (f"; {', '.join(kept)} of other rooms are kept." if kept else "."))
