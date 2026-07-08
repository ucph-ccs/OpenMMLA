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
from .input import get_base_by_id, get_bases, get_function_sync_manager
from .transform import (
    direct_transform_matrices, average_transform_matrices,
    transform_point, distance_between_points, transform_rotation,
    distance_between_rotations, export_main_transformations_json
)

matplotlib.use('TkAgg')


class CameraSyncManager(Synchronizer):
    """Class for synchronizing the coordinate system between the main camera and the alternative camera."""
    logger = get_logger('camera-sync-manager')
    colors = cycle('bgrcmyk')

    def __init__(self, project_dir: str | None, config_path: str, sync: bool = True,
                 top_K: int = 5000, distance_threshold: float = 0.1, angle_threshold: float = 3.0,
                 time_threshold_sync: float = 0.05, time_threshold_unsync: float = 0.1,
                 base: str | None = None):
        """Initialize the camera sync manager.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            sync: whether to synchronize cameras (default: True)
            top_K: maximum number of matrices to keep (default: 5000)
            distance_threshold: threshold for point distance (default: 0.1)
            angle_threshold: threshold for rotation angle (default: 3.0)
            time_threshold_sync: time threshold for sync mode (default: 0.05)
            time_threshold_unsync: time threshold for unsync mode (default: 0.1)
            base: alternative base id (from config 'Bases') to synchronize against
                the main base. The main base is the one flagged ``main: true``.
                If omitted, the alternative is picked interactively.
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.launch_base = base

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
        # (main = the entry flagged main: true; alt = -b entry, else picked).
        self._resolve_bases(base)

    def _resolve_bases(self, base):
        """Resolve main_id (the base flagged main) and alt_id (-b or picked)."""
        bases = get_bases(self.config)
        if not bases:
            raise ValueError(
                "No bases defined. Add entries under 'Bases' in config.yml "
                "(mark exactly one with main: true).")

        mains = [b for b in bases if str(b.get('main')).lower() in ('true', '1', 'yes')]
        if not mains:
            raise ValueError("No main base found. Mark exactly one base with main: true in config 'Bases'.")
        if len(mains) > 1:
            raise ValueError(
                f"Multiple main bases found ({[m.get('id') for m in mains]}); mark exactly one with main: true.")
        self.main_id = str(mains[0].get('id'))

        alts = [b for b in bases if str(b.get('id')) != self.main_id]
        if not alts:
            raise ValueError("No alternative base found; add at least one non-main base under 'Bases'.")

        if base is not None:
            alt = get_base_by_id(self.config, base)
            if alt is None or str(alt.get('id')) == self.main_id:
                raise ValueError(f"Alternative base '{base}' not found (or is the main base) in config 'Bases'.")
        elif len(alts) == 1:
            alt = alts[0]
        else:
            alt = self._pick_alt_interactively(alts)
        self.alt_id = str(alt.get('id'))

        self.logger.info(f"Camera sync: main={self.main_id}, alternative={self.alt_id}")
        print(f"\033]0;Camera Sync Manager: main:{self.main_id} - alternative:{self.alt_id}\007")

    def _pick_alt_interactively(self, alts):
        """Pick the alternative base from non-main entries (the only interaction)."""
        print(f"Main base is '{self.main_id}'. Select the alternative base to synchronize:")
        for idx, b in enumerate(alts):
            print(f"  {idx}: id={b.get('id')} (camera: {b.get('camera')})")
        while True:
            sel = input("Alternative base number [0]: ").strip()
            try:
                index = int(sel) if sel else 0
            except ValueError:
                index = -1
            if 0 <= index < len(alts):
                return alts[index]
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
                select_fun = get_function_sync_manager(self.main_id, self.alt_id, self.sync)
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
        self._resolve_bases(self.launch_base)

    def _start_synchronization(self):
        try:
            if not self.main_id or not self.alt_id:
                return self._set_camera_id()

            self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics="camera/synchronize")
            self.mqtt_client.loop_start()
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
        acquired_time = message["acquired_time"]
        current_time = time.time()

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
                self.main_last_update_time[tag_id] = acquired_time
            elif base_id == self.alt_id:
                self.alt_translations[tag_id] = position
                self.alt_rotations[tag_id] = rotation
                self.alt_last_update_time[tag_id] = acquired_time

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

    def _clear_transformations(self):
        for file in os.listdir(self.camera_sync_dir):
            if file.startswith("transformation_matrices"):
                os.remove(os.path.join(self.camera_sync_dir, file))
                print(f"File {file} has been removed.")
