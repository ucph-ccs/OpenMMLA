"""This script runs the control base."""
import os

from openmmla.utils.control import start_control


def main():
    """Main function for ASR control."""
    project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    config_path = os.path.join(project_dir, 'config.yml')
    
    start_control(config_path)


if __name__ == "__main__":
    main()
