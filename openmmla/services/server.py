import os
import yaml
from openmmla.utils.logger import get_logger


class Server:
    """Base class for all server components in the OpenMMLA platform."""

    def __init__(self, project_dir, config_path=None, use_cuda=True, use_onnx=False):
        """Initialize the base server.

        Args:
            project_dir (str): The project directory.
            config_path (str, optional): Path to the configuration file.
            use_cuda (bool): Whether to use CUDA or not.
        """
        # Check if the project directory exists
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Project directory not found at {project_dir}")

        self.project_dir = project_dir
        self.use_cuda = use_cuda
        self.use_onnx = use_onnx

        # Set up config
        if config_path:
            if os.path.isabs(config_path):
                self.config_path = config_path
            else:
                self.config_path = os.path.join(project_dir, config_path)

            # Check if the configuration file exists
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

            # Load configuration
            self.config = self.load_config()
        else:
            self.config_path = None
            self.config = None

        # Set up directories
        self.server_logger_dir = os.path.join(project_dir, 'logger')
        self.server_file_folder = os.path.join(project_dir, 'temp')
        os.makedirs(self.server_logger_dir, exist_ok=True)
        os.makedirs(self.server_file_folder, exist_ok=True)

        # Set up logger
        self.logger = self.setup_logger()

    def load_config(self):
        """Load the configuration file."""
        with open(self.config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def setup_logger(self):
        """Set up the logger for the server."""
        return get_logger(self.__class__.__name__,
                          os.path.join(self.server_logger_dir, f'{self.__class__.__name__.lower()}_server.log'))

    def get_temp_file_path(self, prefix, base_id, extension):
        """Generate a temporary file path."""
        return os.path.join(self.server_file_folder, f'{prefix}_{base_id}.{extension}')

    def process_request(self):
        """Process the incoming request. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process_request method")
