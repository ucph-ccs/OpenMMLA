import logging
import os
from typing import Optional


def get_logger(name: str,
               log_file: Optional[str] = None,
               level: int = logging.INFO,
               console_level: int = logging.INFO,
               file_level: int = logging.INFO,
               mode: str = 'w') -> logging.Logger:
    """
    Function to create a logger with a given name and log to both console and file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): The file path to write the log output. Defaults to None.
        level (int): The threshold level for the logger. Defaults to logging.INFO.
        console_level (int): The threshold level for logging output to the console. Defaults to logging.INFO.
        file_level (int): The threshold level for logging output to the file. Defaults to logging.INFO.
        mode (str): The mode in which the file is opened. Defaults to 'w' - write, can be 'a' - append.

    Raises:
        FileNotFoundError: If the directory specified in 'log_file' does not exist.

    Returns:
        logging.Logger: A configured logger object that logs to both the file and console.

    Example:
        logger = get_logger('example_logger', 'logs/app.log', level=logging.DEBUG, console_level=logging.INFO, file_level=logging.ERROR)
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Check if log_file is provided and is a valid path
    if log_file:
        # Check if the directory for the log_file exists, if not raise an error
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            raise FileNotFoundError(f"The directory '{log_dir}' does not exist. Cannot create log file.")
        # Create a file handler and set level and formatter
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Always create and add a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger
