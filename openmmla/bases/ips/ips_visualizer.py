import json
import os
import threading
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm

from openmmla.bases.base import Base
from openmmla.utils.client import InfluxDBClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_bucket
from openmmla.utils.logger import get_logger
from .input import get_function_visualizer


class IPSVisualizer(Base):
    """IPSVisualizer class for visualizing the tracing badges' real-time positions and relations"""
    logger = get_logger('ips-visualizer')

    def __init__(self, config_path: str, project_dir: str | None = None,
                 store: bool = True, use_3d: bool = False):
        """Initialize the IPSVisualizer class.

        Args:
            config_path: path to the configuration file
            project_dir: path to the project directory
            store: whether to store the visualization plots (default: True)
            use_3d: if True, run the 3D visualization; otherwise use 2D visualization
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.store = store
        self.use_3d = use_3d

        # Runtime attribute
        self.bucket_name = None

        # Threading attribute
        self.stop_event = threading.Event()
        self.threads = []

        self._setup_directories()
        self._setup_objects()

    def _setup_directories(self):
        """Set up directories."""
        self.logger_dir = os.path.join(self.project_dir, 'logger')
        self.visualizations_dir = os.path.join(self.project_dir, 'visualizations')
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.redis_client = RedisClientWrapper(self.config_path)
        self.influx_client_main = InfluxDBClientWrapper(self.config_path)

    def run(self):
        """Run the IPS visualizer."""
        print('\033]0;IPS Visualizer\007')
        func_map = {1: self._start_visualization, 2: self._switch_dimension}

        while True:
            try:
                dimension = '3d' if self.use_3d else '2d'
                select_fun = get_function_visualizer(dimension=dimension)
                if select_fun == 0:
                    self.logger.info("Exiting IPS visualizer...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(
                    f"During running the visualizer, caught: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, returning to main menu.",
                    exc_info=True)

    def _start_visualization(self):
        self.bucket_name = select_or_create_bucket(self.influx_client_main)
        self._create_bucket_logger()

        if self.store:
            dir_path = os.path.join(self.visualizations_dir, f'{self.bucket_name}/real-time')
            os.makedirs(dir_path, exist_ok=True)

        self._listen_for_start_signal()
        self._create_thread(self._listen_for_stop_signal)
        self._start_threads()

        try:
            if self.use_3d:
                self._start_3d_plot()
            else:
                self._start_2d_plot()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning("%s, returning to main menu.", e, exc_info=True)
        finally:
            self.bucket_name = None

    def _create_bucket_logger(self):
        self.bucket_logger_dir = os.path.join(self.logger_dir, f'{self.bucket_name}')
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        self.logger = get_logger(f'ips-visualizer-{self.bucket_name}',
                                 os.path.join(self.bucket_logger_dir, f'ips_visualizer.log'))

    def _start_2d_plot(self):
        influx_client = InfluxDBClientWrapper(self.config_path)
        fig = plt.figure()
        self.ani = FuncAnimation(fig, self._animate, fargs=(influx_client,), interval=50, cache_frame_data=False)
        plt.show()

    def _start_3d_plot(self):
        influx_client = InfluxDBClientWrapper(self.config_path)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.ani = FuncAnimation(fig, self._animate_3d, fargs=(fig, ax, influx_client,), interval=50,
                                 cache_frame_data=False)
        plt.show()

    def _animate(self, i, influx_client: InfluxDBClientWrapper):
        plt.cla()
        # Get node relations and positions
        graph_dict, timestamp = self._get_node_relations(influx_client)
        if graph_dict is None:
            return
        pos = self._get_node_positions(influx_client, timestamp=timestamp, dimension='2d')
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

        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")

        if self.store:
            plt.savefig(
                os.path.join(self.visualizations_dir, f'{self.bucket_name}/real-time/image_{timestamp}_2d.png'))

    def _switch_dimension(self):
        self.use_3d = not self.use_3d

    def _animate_3d(self, i, fig, ax, influx_client: InfluxDBClientWrapper):
        plt.cla()
        # Get node relations and positions
        graph_dict, timestamp = self._get_node_relations(influx_client)
        if graph_dict is None:
            return
        pos_3d = self._get_node_positions(timestamp=timestamp, dimension='3d', influx_client=influx_client)
        G = self._build_graph(graph_dict, pos_3d)

        # Draw the 3D graph
        for node, coordinates in pos_3d.items():
            camera_x, camera_y, camera_z = coordinates
            ax.scatter(camera_x, camera_z, camera_y, s=200, c='white', edgecolors='green')
            ax.text(camera_x, camera_z, camera_y, node, fontsize=6, color='green',
                    ha='center', va='center', zorder=40)

        # Draw directed edges as arrows
        for edge in G.edges():
            x1, y1, z1 = pos_3d[edge[0]]
            x2, y2, z2 = pos_3d[edge[1]]
            self.draw_arrow(ax, x1, z1, y1, x2, z2, y2)

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.view_init(elev=20., azim=30)
        if self.store:
            plt.savefig(
                os.path.join(self.visualizations_dir, f'{self.bucket_name}/real-time/image_{timestamp}_3d.png'))

    def _build_graph(self, graph_dict: dict, pos: dict) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_node('B')
        for badge_id, detected_tags in graph_dict.items():
            if badge_id not in G:
                G.add_node(badge_id)
            for tag_id in detected_tags:
                tag_str = str(tag_id)
                if tag_str not in G:
                    G.add_node(tag_str)
                G.add_edge(badge_id, tag_str)

        missing_nodes = [node for node in G.nodes() if node not in pos]
        if missing_nodes:
            self.logger.warning("Missing positions for nodes: %s", missing_nodes)
        for node in missing_nodes:
            G.remove_node(node)
        return G

    def _get_node_relations(self, influx_client: InfluxDBClientWrapper) -> tuple[dict | None, float | None]:
        start_time = int(time.time()) - 20
        query = f"""from(bucket: "{self.bucket_name}")
                    |> range(start: {start_time})
                    |> last()
                    |> filter(fn: (r) => r._measurement == "badge_relation")
                    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
                    """
        tables = influx_client.query(query)
        data = json.loads(tables.to_json(indent=5))
        if not data:
            print("No data found for the specified bucket.")
            return None, None
        graph_dict_str = data[0]["graph"]
        graph_dict = json.loads(graph_dict_str)
        timestamp = data[0]["window_start_time"]
        return graph_dict, timestamp

    def _get_node_positions(self, influx_client: InfluxDBClientWrapper, timestamp: float,
                            dimension: str = '2d') -> dict:
        start_time = int(timestamp) - 20
        query = f"""from(bucket: "{self.bucket_name}")
                   |> range(start: {start_time})
                   |> filter(fn: (r) => r._measurement == "badge_translation")
                   |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
                   |> filter(fn: (r) => r.window_start_time == {timestamp})
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

    def _stop_threads(self):
        super()._stop_threads()
        self.ani.pause()

    @staticmethod
    def draw_arrow(ax, x1, y1, z1, x2, y2, z2, node_radius=0.04):
        arrow_vector = np.array([x2 - x1, y2 - y1, z2 - z1])
        arrow_unit_vector = arrow_vector / norm(arrow_vector)
        start_point = np.array([x1, y1, z1]) + node_radius * arrow_unit_vector
        end_point = np.array([x2, y2, z2]) - node_radius * arrow_unit_vector
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                c='green', linewidth=0.5, zorder=4)
        ax.scatter([end_point[0]], [end_point[1]], [end_point[2]],
                   c='red', s=10, marker='.', zorder=10)

    @property
    def bucket_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current bucket_name."""
        if self.bucket_name:
            return f'{self.bucket_name}/ips/control'
        return None
