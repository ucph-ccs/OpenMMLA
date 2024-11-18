import json
import os
import platform
import time
from multiprocessing import Process

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm

from openmmla.bases.base import Base
from openmmla.utils.client import InfluxDBClientWrapper, RedisClientWrapper
from openmmla.utils.logger import get_logger
from .input import get_bucket_name, get_function_visualizer

# Setup for multiprocessing on macOS
# https://stackoverflow.com/questions/44144584/typeerror-cant-pickle-thread-lock-objects/78013322#78013322
# Add 'export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES' to ~/.zshrc for multiprocess fork
if platform.system() != "Linux":
    from multiprocessing import set_start_method

    set_start_method("fork")


class VideoVisualizer(Base):
    """Real Time Visualization class for visualizing the real-time badge relations and positions"""
    logger = get_logger('visualizer')

    def __init__(self, project_dir: str, config_path: str, store: bool = False):
        """Initialize the visualizer.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            store: whether to store the visualization images in the local directory
        """
        super().__init__(project_dir, config_path)

        """Visualizer specific parameters."""
        self.store = store

        """Runtime attributes."""
        self.bucket_name = None
        self.processes = []

        self._setup_directories()
        self._setup_objects()

    def _setup_directories(self):
        """Set up directories."""
        self.visualizations_dir = os.path.join(self.project_dir, 'visualizations')
        os.makedirs(self.visualizations_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.redis_client = RedisClientWrapper(self.config_path)
        self.influx_client_main = InfluxDBClientWrapper(self.config_path)

    def run(self):
        print('\033]0;Video Visualizer\007')
        func_map = {1: self._start_visualizing}

        while True:
            try:
                select_fun = get_function_visualizer()
                if select_fun == 0:
                    self.logger.info("Exiting video visualizer...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"During running the visualizer, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)

    def _start_visualizing(self):
        self.bucket_name = get_bucket_name(self.influx_client_main)

        if self.store:
            dir_path = os.path.join(self.visualizations_dir, f'{self.bucket_name}/real-time')
            os.makedirs(dir_path, exist_ok=True)

        self._listen_for_start_signal()
        self.create_process(self._start_2d_plot)
        self.create_process(self._start_3d_plot)

        try:
            self.start_processes()
            self.join_processes()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning("%s, Come back to the main menu.", e, exc_info=True)
        finally:
            self.stop_processes()
            self.bucket_name = None

    def create_process(self, target):
        p = Process(target=target, daemon=True)
        self.processes.append(p)

    def start_processes(self):
        for p in self.processes:
            p.start()

    def join_processes(self):
        for p in self.processes:
            p.join()

    def stop_processes(self):
        for p in self.processes:
            if p.is_alive():
                p.terminate()  # Send termination request
                p.join()  # Wait for the process to finish
        self.processes.clear()

    def _start_2d_plot(self):
        influx_client = InfluxDBClientWrapper(self.config_path)
        fig = plt.figure()
        ani = FuncAnimation(fig, self._animate, fargs=(influx_client,), interval=50, cache_frame_data=False)
        plt.show()

    def _start_3d_plot(self):
        influx_client = InfluxDBClientWrapper(self.config_path)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ani = FuncAnimation(fig, self._animate_3d, fargs=(fig, ax, influx_client,),
                            interval=50, cache_frame_data=False)
        plt.show()

    def _animate(self, i, influx_client):
        plt.cla()

        # Get node relations and pos
        graph_dict, segment_time = self._get_node_relations(influx_client)
        if graph_dict is None:
            return
        pos = self._get_node_positions(influx_client, segment_time=segment_time, dimension='2d')
        G = self._build_graph(graph_dict, pos)

        options = {
            "font_size": 15,
            "node_size": 1000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 3,
            "width": 3,
        }
        nx.draw_networkx(G, pos, **options)

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        if self.store:
            plt.savefig(
                os.path.join(self.visualizations_dir, f'{self.bucket_name}/real-time/image_{segment_time}_2d.png'))

    def _animate_3d(self, i, fig, ax, influx_client):
        plt.cla()

        # Get node relations and pos
        graph_dict, segment_time = self._get_node_relations(influx_client)
        if graph_dict is None:
            return
        pos_3d = self._get_node_positions(segment_time=segment_time, dimension='3d', influx_client=influx_client)
        G = self._build_graph(graph_dict, pos_3d)

        # Draw the 3D graph
        for node, coordinates in pos_3d.items():
            camera_x, camera_y, camera_z = coordinates
            ax.scatter(camera_x, camera_z, camera_y, s=200, c='white', edgecolors='green')
            ax.text(camera_x, camera_z, camera_y, node, fontsize=6, color='green', ha='center', va='center', zorder=40)

        # Draw directed edges as arrows
        for edge in G.edges():
            x1, y1, z1 = pos_3d[edge[0]]
            x2, y2, z2 = pos_3d[edge[1]]
            self.draw_arrow(ax, x1, z1, y1, x2, z2, y2)

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.view_init(elev=20., azim=30)  # adjust the viewing angle for better visualization
        if self.store:
            plt.savefig(
                os.path.join(self.visualizations_dir, f'{self.bucket_name}/real-time/image_{segment_time}_3d.png'))

    def _build_graph(self, graph_dict, pos):
        G = nx.DiGraph()
        G.add_node('B')
        # Add nodes and edges based on the graph_dict
        for badge_id, detected_tags in graph_dict.items():
            if badge_id not in G:
                G.add_node(badge_id)
            for tag_id in detected_tags:
                if str(tag_id) not in G:
                    G.add_node(str(tag_id))
                G.add_edge(badge_id, str(tag_id))

        # Remove missing nodes
        missing_nodes = [node for node in G.nodes() if node not in pos]
        if missing_nodes:
            self.logger.warning("Missing positions for nodes: %s", missing_nodes)
        for node in missing_nodes:
            G.remove_node(node)
        return G

    def _get_node_relations(self, influx_client):
        start_time = int(time.time()) - 20
        query = f"""from(bucket: "{self.bucket_name}")
                    |> range(start: {start_time})
                    |> last()
                    |> filter(fn: (r) => r._measurement == "badge relations")
                    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
                    """
        tables = influx_client.query(query)
        data = json.loads(tables.to_json(indent=5))
        if not data:
            print("No data found for the specified bucket.")
            return None, None
        graph_dict_str = data[0]["graph"]
        graph_dict = json.loads(graph_dict_str)
        segment_time = data[0]["segment_start_time"]
        return graph_dict, segment_time

    def _get_node_positions(self, influx_client, segment_time, dimension='2d'):
        start_time = int(segment_time) - 20
        query = f"""from(bucket: "{self.bucket_name}")
                   |> range(start: {start_time})
                   |> filter(fn: (r) => r._measurement == "badge translations")
                   |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
                   |> filter(fn: (r) => r.segment_start_time == {segment_time})
                  """
        tables = influx_client.query(query)
        data = json.loads(tables.to_json(indent=5))
        translate_dict = json.loads(data[0]["translations"])

        positions = {'B': (0, 0)} if dimension == '2d' else {'B': (0, 0, 0)}
        for badge_id, translation in translate_dict.items():
            if dimension == '2d':
                x = translation[0][0]
                z = translation[2][0]
                positions[badge_id] = (z, -x)
            else:
                x = translation[0][0]
                y = translation[1][0]
                z = translation[2][0]
                positions[badge_id] = (x, -y, z)
        return positions

    @staticmethod
    def draw_arrow(ax, x1, y1, z1, x2, y2, z2, node_radius=0.04):
        arrow_vector = np.array([x2 - x1, y2 - y1, z2 - z1])
        arrow_unit_vector = arrow_vector / norm(arrow_vector)
        start_point = np.array([x1, y1, z1]) + node_radius * arrow_unit_vector
        end_point = np.array([x2, y2, z2]) - node_radius * arrow_unit_vector
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]],
                c='green', linewidth=0.5, zorder=4)
        ax.scatter([end_point[0]], [end_point[1]], [end_point[2]], c='red', s=10, marker='.', zorder=10)
