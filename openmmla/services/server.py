import os
import time
from abc import abstractmethod, ABC

import yaml

from openmmla.utils.logger import get_logger


class Server(ABC):
    """Base class for servers."""

    def __init__(self, project_dir: str | None = None, config_path: str | None = None):
        """Initialize the server base class.

        Args:
            project_dir: path to the project directory (default: current working directory)
            config_path: path to the configuration file (default: None)
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

        # setup server directories
        self.server_logger_dir = os.path.join(self.project_dir, 'logger')
        self.server_temp_folder = os.path.join(self.project_dir, 'temp')
        os.makedirs(self.server_logger_dir, exist_ok=True)
        os.makedirs(self.server_temp_folder, exist_ok=True)
        self.logger = get_logger(f'{self.__class__.__name__}_{time.time()}',
                                 os.path.join(self.server_logger_dir, f'{self.__class__.__name__.lower()}_server.log'),
                                 mode='a')

    def _load_config(self):
        """Load the configuration file."""
        with open(self.config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def _get_temp_file_path(self, prefix, base_id, extension):
        """Generate a temporary file path."""
        return os.path.join(self.server_temp_folder, f'{prefix}_{base_id}.{extension}')

    @abstractmethod
    def process_request(self):
        """Process the incoming request. To be implemented by subclasses."""
        pass
