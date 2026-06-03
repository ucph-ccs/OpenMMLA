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
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, shared_pipeline_artifact_dir
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_session, show_error_and_pause
from openmmla.utils.logger import get_logger
from .input import get_function_visualizer


class IPSVisualizer(Base):
    """IPSVisualizer class for visualizing the tracing badges' real-time positions and relations"""
    logger = get_logger('ips-visualizer')

    def __init__(self, config_path: str, project_dir: str | None = None,
                 store: bool = True, use_3d: bool = False, session_id: str | None = None):
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
        self.launch_session_id = session_id

        # Runtime attribute
        self.session_id = None

        # Threading attribute
        self.stop_event = threading.Event()
        self.threads = []

        self._setup_directories()
        self._setup_objects()

    def _setup_directories(self):
        """Set up directories."""
        self.logger_dir = os.fspath(shared_pipeline_artifact_dir(self.project_dir, 'ips-base', 'logger'))
        self.visualizations_dir = os.fspath(
            shared_pipeline_artifact_dir(self.project_dir, 'ips-base', 'visualizations')
        )
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

    def _setup_objects(self):
        """Set up client objects."""
        self.redis_client = RedisClientWrapper(self.config_path)
        self.influx_client_main = InfluxDBClientWrapper(self.config_path)
        self.mongo_client = MongoDBClientWrapper(self.config_path)

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
                if not isinstance(e, KeyboardInterrupt):
                    show_error_and_pause(e, "return to the IPS Visualizer menu")

    def _start_visualization(self):
        self.session_id = self.launch_session_id or select_or_create_session(self.mongo_client)
        self._create_bucket_logger()

        if self.store:
            self.visualizations_dir = os.fspath(
                pipeline_section_dir(self.project_dir, self.session_id, 'ips-base', 'visualizations')
            )
            dir_path = os.path.join(self.visualizations_dir, 'real-time')
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
            self.session_id = None

    def _create_bucket_logger(self):
        self.bucket_logger_dir = os.fspath(
            pipeline_section_dir(self.project_dir, self.session_id, 'ips-base', 'logger')
        )
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        copy_config_snapshot(self.config_path, self.project_dir, self.session_id, 'ips-base')
        self.logger = get_logger(f'ips-visualizer-{self.session_id}',
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
                os.path.join(self.visualizations_dir, f'real-time/image_{timestamp}_2d.png'))

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
                os.path.join(self.visualizations_dir, f'real-time/image_{timestamp}_3d.png'))

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
        from openmmla.utils.constants import EVENT_TYPE_IPS_RELATION
        from openmmla.utils.querys import deep_parse_json
        event = influx_client.query_latest_event(self.session_id, EVENT_TYPE_IPS_RELATION)
        if not event:
            print("No data found for the specified session.")
            return None, None
        event = deep_parse_json(event)
        graph_dict = event["graph"]
        timestamp = event["window_start_time"]
        return graph_dict, timestamp

    def _get_node_positions(self, influx_client: InfluxDBClientWrapper, timestamp: float,
                            dimension: str = '2d') -> dict:
        from openmmla.utils.querys import get_node_positions
        return get_node_positions(self.session_id, influx_client, int(timestamp), dimension)

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
    def session_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current session_id."""
        if self.session_id:
            return f'{self.session_id}/ips/control'
        return None
