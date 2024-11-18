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
from openmmla.utils.logger import get_logger
from .input import get_function_sync_manager
from .transform import (
    direct_transform_matrices, average_transform_matrices,
    transform_point, distance_between_points, transform_rotation,
    distance_between_rotations, export_main_transformations_json
)

matplotlib.use('TkAgg')


class CameraSyncManager(Synchronizer):
    """Class for synchronizing the coordinate system between the main camera and alternative camera."""
    logger = get_logger('camera-sync-manager')
    colors = cycle('bgrcmyk')

    def __init__(self, project_dir: str, config_path: str, sync: bool = True, top_K: int = 5000,
                 distance_threshold: float = 0.1, angle_threshold: float = 3.0, time_threshold_sync: float = 0.02,
                 time_threshold_unsync: float = 0.1):
        """Initialize the camera sync manager.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            sync: whether to synchronize cameras
            top_K: maximum number of matrices to keep
            distance_threshold: threshold for point distance
            angle_threshold: threshold for rotation angle
            time_threshold_sync: time threshold for sync mode
            time_threshold_unsync: time threshold for unsync mode
        """
        super().__init__(project_dir, config_path)

        """Synchronization parameters."""
        self.sync = sync
        self.top_K = top_K
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.time_threshold_sync = time_threshold_sync
        self.time_threshold_unsync = time_threshold_unsync
        self.time_threshold = self.time_threshold_sync if self.sync else self.time_threshold_unsync

        """Runtime attributes."""
        self.main_camera_id = ''
        self.alternative_camera_id = ''
        self.main_positions = {}
        self.main_rotations = {}
        self.main_last_update = {}
        self.alternative_positions = {}
        self.alternative_rotations = {}
        self.alternative_last_update = {}
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

    def _setup_directories(self):
        """Set up required directories."""
        self.camera_sync_dir = os.path.join(self.project_dir, 'camera_sync')
        os.makedirs(self.camera_sync_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.mqtt_client = MQTTClientWrapper(self.config_path)

    def run(self):
        """Run the camera sync manager."""
        func_map = {1: self._start, 2: self._switch, 3: self._set_camera_id,
                    4: self._export_transformations, 5: self._clear_transformations}
        while True:
            try:
                select_fun = get_function_sync_manager()
                if select_fun == 0:
                    self.logger.info("Exiting video synchronizer...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning("%s, Come back to the main menu.", e, exc_info=True)

    def _set_camera_id(self):
        """Set camera IDs through user input."""
        self.main_camera_id = input("Please enter the main camera ID: ")
        self.alternative_camera_id = input("Please enter the alternative camera ID: ")
        print(f"\033]0;Camera Sync Manager: main:{self.main_camera_id} - alternative:{self.alternative_camera_id}\007")

    def _start(self):
        try:
            if not self.main_camera_id or not self.alternative_camera_id:
                return self._set_camera_id()

            self.mqtt_client.reinitialise(on_message=self._handle_base_result, topics="camera/synchronize")
            self.mqtt_client.loop_start()
            self.fig, self.ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
            plt.get_current_fig_manager().set_window_title(
                f"Camera Sync Manager: main:{self.main_camera_id} - alternative:{self.alternative_camera_id}")
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
        self.main_positions, self.main_rotations, self.main_last_update = {}, {}, {}
        self.alternative_positions, self.alternative_rotations, self.alternative_last_update = {}, {}, {}
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

        sender_id = message["sender_id"]
        tags = message["tags"]
        current_time = time.time()

        for tag in list(self.main_positions.keys()):
            if (current_time - self.main_last_update.get(tag, 0)) > self.time_threshold:
                del self.main_positions[tag]
                del self.main_last_update[tag]

        for tag in list(self.alternative_positions.keys()):
            if (current_time - self.alternative_last_update.get(tag, 0)) > self.time_threshold:
                del self.alternative_positions[tag]
                del self.alternative_last_update[tag]

        for tag_id, tag_data in tags.items():
            position = tag_data[1]
            rotation = tag_data[0]

            if sender_id == self.main_camera_id:
                self.main_positions[tag_id] = position
                self.main_rotations[tag_id] = rotation
                self.main_last_update[tag_id] = current_time
            elif sender_id == self.alternative_camera_id:
                self.alternative_positions[tag_id] = position
                self.alternative_rotations[tag_id] = rotation
                self.alternative_last_update[tag_id] = current_time

            if tag_id in self.main_positions and tag_id in self.alternative_positions:
                color = self.tag_color_map.setdefault(tag_id, next(self.colors))
                if self.sync:
                    self._synchronize(tag_id, color)
                else:
                    self._validate(tag_id, color)

        if self.plot_points_list:
            self.plot_points_queue.put(self.plot_points_list)
            self.plot_points_list = []

    def _synchronize(self, tag, color):
        R, T = direct_transform_matrices(self.main_rotations[tag], self.main_positions[tag],
                                         self.alternative_rotations[tag], self.alternative_positions[tag])
        self.rotation_matrices.append(R)
        self.translations_matrices.append(T)
        self.rotation_matrices = self.rotation_matrices[-self.top_K:]
        self.translations_matrices = self.translations_matrices[-self.top_K:]

        R_avg, T_avg = average_transform_matrices(self.rotation_matrices, self.translations_matrices)
        print(f"R matrix for tag {tag}: {R_avg}, T vector for tag {tag}: {T_avg}")

        transformed_point = transform_point(self.alternative_positions[tag], R_avg, T_avg)
        transformed_rotation = transform_rotation(R_avg, self.alternative_rotations[tag])
        point_distance = distance_between_points(transformed_point, self.main_positions[tag])
        rotation_distance = distance_between_rotations(transformed_rotation, self.main_rotations[tag])

        print(
            f"Main position for tag {tag}: {self.main_positions[tag]}, "
            f"alternative position for tag {tag}: {self.alternative_positions[tag]}, "
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
            transformed_point = transform_point(self.alternative_positions[tag], self.retrieved_R, self.retrieved_T)
            transformed_rotation = transform_rotation(self.retrieved_R, self.alternative_rotations[tag])
            point_distance = distance_between_points(transformed_point, self.main_positions[tag])
            rotation_distance = distance_between_rotations(transformed_rotation, self.main_rotations[tag])

            print(
                f"Main position for tag {tag}: {self.main_positions[tag]}, "
                f"alternative position for tag {tag}: {self.alternative_positions[tag]},"
                f"converted position for tag {tag}: {transformed_point}")
            print(f"Distances for tag {tag}: {point_distance} m, angles: {rotation_distance} °")

            self._add_points_to_list(tag, color, transformed_point)
        else:
            with open(os.path.join(self.camera_sync_dir, 'transformation_matrices.json'), 'r') as file:
                data = json.load(file)
            key = f'{self.alternative_camera_id}-{self.main_camera_id}'
            self.retrieved_R, self.retrieved_T = np.array(data[key]['R']), np.array(
                data[key]['T'])

    def _add_points_to_list(self, tag, color, transformed_point):
        main_point = np.array(self.main_positions[tag]).reshape(1, 3)
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

        key = f'{self.alternative_camera_id}-{self.main_camera_id}'
        data[key] = {"R": R, "T": T}
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _export_transformations(self):
        main_system = input("Please enter the main system ID: ")
        input_path = os.path.join(self.camera_sync_dir, 'transformation_matrices.json')
        output_path = os.path.join(self.camera_sync_dir, f'transformation_matrices_{main_system}.json')
        export_main_transformations_json(input_path, output_path, main_system)

    def _clear_transformations(self):
        for file in os.listdir(self.camera_sync_dir):
            if file.startswith("transformation_matrices"):
                os.remove(os.path.join(self.camera_sync_dir, file))
                print(f"File {file} has been removed.")
