import os
import threading
import time
from abc import ABC, abstractmethod

import yaml

from openmmla.utils.client import RedisClientWrapper
from openmmla.utils.logger import get_logger
from openmmla.utils.threads import RaisingThread


class Base(ABC):
    """Base class for data processing pipeline."""
    logger = get_logger('Base')

    def __init__(self, project_dir: str | None = None, config_path: str | None = None):
        """Initialize the data processing pipeline base class.
        
        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        if project_dir:
            if not os.path.isabs(project_dir):
                project_dir = os.path.join(os.getcwd(), project_dir)
            if not os.path.exists(project_dir):
                raise FileNotFoundError(f"Project directory not found at {project_dir}")
            self.project_dir = project_dir
        else:
            self.project_dir = os.getcwd()

        if config_path:
            if not os.path.isabs(config_path):
                config_path = os.path.join(os.getcwd(), config_path)
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found at {config_path}")

        self.config_path = config_path
        self.config = self._load_config() if self.config_path else None

        self.threads: list[RaisingThread] | None = None
        self.stop_event: threading.Event | None = None
        self.session_id: str | None = None
        self.redis_client: RedisClientWrapper | None = None
        # (channel, pubsub) subscribed ahead of the listening: Redis keeps no message for a later subscriber
        self._control_subscription = None
        self._stop_heard = False  # the STOP listener heard STOP in this run

    @property
    def session_control(self):
        """Dynamic property that returns the control channel name based on current session_id."""
        if self.session_id:
            return f"{self.session_id}/control"
        return None

    def _load_config(self):
        """Load the configuration file, decrypting sensitive values and encrypting any
        plaintext secrets back to the YAML file."""
        with open(self.config_path, 'r') as config_file:
            raw_data = yaml.safe_load(config_file)
        try:
            from openmmla.utils.crypto import process_config_dict
            from openmmla.utils.yaml_dump import dump_yaml_pretty
            runtime_data, needs_rewrite = process_config_dict(raw_data)
            if needs_rewrite:
                dump_yaml_pretty(raw_data, self.config_path)
            return runtime_data
        except (FileNotFoundError, ImportError):
            return raw_data

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

    def _reinit(self):
        """Reinitialize by calling __init__ again with stored parameters."""
        self.logger.info("Starting reinitialization...")
        
        # Store the original initialization parameters
        project_dir = getattr(self, 'project_dir', None)
        config_path = getattr(self, 'config_path', None)
        
        # Clean up current state
        self._clean_up()
        
        # Call __init__ again with the original parameters
        self.__init__(project_dir=project_dir, config_path=config_path)
        
        self.logger.info("Reinitialization completed successfully")

    def _create_thread(self, target, daemon=True, name=None, *args):
        """Create a new thread and add it to the thread list."""
        t = RaisingThread(target=target, args=args, daemon=daemon, name=name)
        self.threads.append(t)
        return t

    def _clear_threads(self):
        """Clear all threads."""
        self.threads.clear()

    def _start_threads(self):
        """Start all threads."""
        self.stop_event.clear()
        for t in self.threads:
            t.start()

    def _join_threads(self):
        """Wait for all threads to finish without timeout."""
        for t in self.threads:
            t.join()

    def _stop_threads(self):
        """Stop all threads other than the current thread with timeout."""
        self.stop_event.set()
        for t in self.threads:
            if threading.current_thread() != t:
                try:
                    t.join(timeout=5)
                    if t.is_alive():
                        self.logger.warning(f"Thread {t.name or 'unnamed'} did not stop within 5 second timeout")
                except Exception as e:
                    self.logger.warning(f"During thread stopping, catch: {e}", exc_info=True)

    def _subscribe_control(self):
        """Subscribe to the control channel now, ahead of the listening: a START or STOP sent while the
        base still opens its stream (which may wait for it) is heard when the listening begins. Redis
        pubsub keeps nothing for a subscriber that comes later, and Session Control counts no
        subscribers."""
        channel = f"{self.session_control}"
        kept = getattr(self, '_control_subscription', None)
        if not kept or kept[0] != channel:
            self._drop_control_subscription()
            self._control_subscription = (channel, self.redis_client.subscribe(channel))
        return self._control_subscription[1]

    def _take_control_subscription(self):
        """the subscription made ahead for this session's channel, once; one made for another session is
        closed, never read. Without one, a fresh subscription."""
        channel = f"{self.session_control}"
        kept = getattr(self, '_control_subscription', None)
        if kept and kept[0] == channel:
            self._control_subscription = None
            return kept[1]
        self._drop_control_subscription()
        return self.redis_client.subscribe(channel)

    def _drop_control_subscription(self):
        kept, self._control_subscription = getattr(self, '_control_subscription', None), None
        self._close_control_subscription(kept)

    def _close_control_subscription(self, kept):
        """close a (channel, pubsub) that nobody reads any more"""
        if kept and hasattr(kept[1], 'close'):
            try:
                kept[1].close()
            except Exception as e:
                self.logger.debug(f"Closing the control subscription of {kept[0]}: {e}")

    def _listen_for_start_signal(self) -> bool:
        """Listen on the redis bucket control channel for the START signal.

        Returns:
            True on START; False when STOP comes first: the run ended before it started, and the
            caller starts no work, cleans up and ends the run as it would on STOP.
        """
        p = self._take_control_subscription()
        self.logger.info(f"Wait for START signal on {self.session_control}...")

        while True:
            message = p.get_message(timeout=5)
            if message and message['data'] == b'START':
                self.logger.info("Received START signal, start...")
                # kept for the STOP listener, which reads on from here: a STOP queued behind this START is
                # heard, and nothing sent before the listener begins is lost
                self._control_subscription = (f"{self.session_control}", p)
                return True
            if message and message['data'] == b'STOP':
                self.logger.info(f"Received STOP on {self.session_control} before START: the run ends "
                                 f"without starting.")
                return False
            time.sleep(0.05)

    def _listen_for_stop_signal(self):
        """Listen on the redis bucket control channel for the STOP signal."""
        p = self._take_control_subscription()
        self.logger.info(f"Listening for STOP signal on {self.session_control}...")

        while not self.stop_event.is_set():
            message = p.get_message(timeout=5)
            if message and message['data'] == b'STOP':
                self.logger.info("Received STOP signal, stop...")
                # noted: a run that is ending for another reason (an ASR recording error) must not go on
                self._stop_heard = True
                self._stop_threads()
            time.sleep(0.05)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Main entry point for the base."""
        pass
