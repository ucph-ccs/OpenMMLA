import configparser
import os
from abc import ABC


class Base(ABC):
    """
    Base class for data pipelines which process data streams into specific measurement features.
    """

    def __init__(self, project_dir: str = None, config_path: str = None):
        self.project_dir = project_dir
        if not os.path.exists(self.project_dir):
            raise FileNotFoundError(f"Project directory not found at {self.project_dir}")

        if os.path.isabs(config_path):
            self.config_path = config_path
        else:
            self.config_path = os.path.join(self.project_dir, config_path)

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
