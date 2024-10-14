import os
from abc import ABC

import yaml


class Base(ABC):
    """Base class for base components (e.g. data processing pipeline, synchronizer, etc.)."""

    def __init__(self, project_dir: str, config_path: str):
        """Initialize the base class.
        Args:
            project_dir (str): The project directory.
            config_path (str, optional): Path to the configuration file.
        """
        self.project_dir = project_dir
        if not os.path.exists(self.project_dir):
            raise FileNotFoundError(f"Project directory not found at {self.project_dir}")

        if os.path.isabs(config_path):
            self.config_path = config_path
        else:
            self.config_path = os.path.join(self.project_dir, config_path)

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        self.config = self._load_config()

    def _load_config(self):
        """Load the configuration file."""
        with open(self.config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def _setup_from_yaml(self):
        """Set up attributes from YAML configuration."""
        pass

    def _setup_from_input(self):
        """Set up attributes from user input."""
        pass

    def _setup_objects(self):
        """Set up object attributes."""
        pass