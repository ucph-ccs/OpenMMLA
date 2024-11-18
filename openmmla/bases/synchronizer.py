import os
import threading
from abc import ABC, abstractmethod
from typing import List

import yaml

from openmmla.utils.client import RedisClientWrapper
from openmmla.utils.logger import get_logger
from openmmla.utils.threads import RaisingThread


class Synchronizer(ABC):
    """Base class for synchronizer implementations."""
    logger = get_logger('synchronizer')

    def __init__(self, project_dir: str, config_path: str):
        """Initialize the synchronizer base class.
        
        Args:
            project_dir: The project directory
            config_path: Path to the configuration file
        """
        self.project_dir = project_dir
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_dir, config_path)

        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Project directory not found at {project_dir}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        self.config_path = config_path
        self.config = self._load_config()

        self.threads: List[RaisingThread] = None
        self.stop_event: threading.Event = None
        self.bucket_name: str = None
        self.redis_client: RedisClientWrapper = None

    def _load_config(self):
        """Load the configuration file."""
        with open(self.config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def _setup_yaml(self):
        """Set up attributes from YAML configuration."""
        pass

    def _setup_directories(self):
        """Set up required directories."""
        pass

    def _setup_objects(self):
        """Set up client objects."""
        pass

    def _clean_up(self):
        """Free memory by resetting attributes."""
        pass

    def _create_thread(self, target, *args):
        """Create a new thread and add it to the thread list."""
        t = RaisingThread(target=target, args=args)
        self.threads.append(t)

    def _start_threads(self):
        """Start all threads."""
        self.stop_event.clear()
        for t in self.threads:
            t.start()

    def _join_threads(self):
        """Wait for all threads to finish."""
        for t in self.threads:
            t.join()

    def _stop_threads(self):
        """Stop all threads and free memory."""
        self.stop_event.set()
        for t in self.threads:
            if threading.current_thread() != t:
                try:
                    t.join(timeout=5)
                except Exception as e:
                    self.logger.warning(f"During thread stopping, catch: {e}", exc_info=True)
        self.threads.clear()

    def _listen_for_start_signal(self):
        """Listen on the redis bucket control channel for the START signal."""
        p = self.redis_client.subscribe(f"{self.bucket_name}/control")
        self.logger.info("Wait for START signal...")
        while True:
            message = p.get_message()
            if message and message['data'] == b'START':
                self.logger.info("Received START signal, start synchronizing...")
                break

    def _listen_for_stop_signal(self):
        """Listen on the redis bucket control channel for the STOP signal."""
        p = self.redis_client.subscribe(f"{self.bucket_name}/control")
        self.logger.info("Listening for STOP signal...")

        while not self.stop_event.is_set():
            message = p.get_message()
            if message and message['data'] == b'STOP':
                self.logger.info("Received STOP signal, stop synchronizing...")
                self._stop_threads()

    @abstractmethod
    def run(self, *args, **kwargs):
        """Main entry point for the synchronizer."""
        pass

    @abstractmethod
    def _handle_base_result(self, *args, **kwargs):
        """Handle results received from bases."""
        pass
