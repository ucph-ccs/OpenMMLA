import os

from openmmla.utils.clean import flush_input
from .enums import LIGHT_BLUE, ENDC


def get_base_mode():
    """Get the operating mode from user input."""
    while True:
        try:
            flush_input()
            selected_mode = int(input("Please select the mode:"
                                      "\n1. Record, store audio locally without recognizing"
                                      "\n2. Recognize, recognize locally stored audio without recording"
                                      "\n3. Full, record and recognize on-the-fly"
                                      "\nSelected mode:"))
            if selected_mode == 1:
                return 'record'
            elif selected_mode == 2:
                return 'recognize'
            elif selected_mode == 3:
                return 'full'
            else:
                print("Invalid mode, please select again.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def get_synchronizer_mode():
    """Get the operating mode from user input."""
    while True:
        try:
            flush_input()
            selected_mode = int(input("Please select the mode:"
                                      "\n1. Recognize, recognize locally stored audio"
                                      "\n2. Full, recognize on-the-fly"
                                      "\nSelected mode:"))
            if selected_mode == 1:
                return 'recognize'
            elif selected_mode == 2:
                return 'full'
            else:
                print("Invalid mode, please select again.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def get_name():
    """Get the name of the speaker from user input."""
    flush_input()
    return input("Please enter the name of the speaker: ")


def get_base_type(config: dict) -> str:
    """Get the base type from user input."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            print("Please select the base type: ")
            # base types is the keys of the config dictionary and start with 'Base_'    
            base_types = [key for key in config.keys() if key.startswith('Base_')]
            for i, base_type in enumerate(base_types):
                print(f"{i} : {base_type}")
            index = int(input("Please select the base type: "))
            if 0 <= index < len(base_types):
                return base_types[index]
            else:
                print(f"Invalid input. Please select a valid base type from 0 to {len(base_types)}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


def get_id():
    """Get the unique base id from user input."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            return int(input("Enter the your base id: "))
        except ValueError:
            print("Invalid input. Please enter an integer as your unique base id.")

def get_file(files: list[str]) -> str:
    """Get the file as audio stream source from user input."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            print("Please select an audio file:")
            for i, file_path in enumerate(files):
                filename = os.path.basename(file_path)
                print(f"{i}: {filename}")
            
            index = int(input("Please select the file index: "))
            if 0 <= index < len(files):
                return files[index]
            else:
                print(f"Invalid input. Please select a valid file index from 0 to {len(files) - 1}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def get_input_device_index(available_indexes: list[int]) -> int:
    """Get the input device index from user input."""
    while True:
        try:
            flush_input()
            input_device_index = int(input("Please select your input device index for PyAudio: "))
            if input_device_index in available_indexes:
                return input_device_index
            else:
                print(f"Invalid input. Please select a valid device index from {available_indexes}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


def get_rtmp_url(available_urls: list[str]) -> str:
    """Get the RTMP URL from user input."""
    while True:
        try:
            flush_input()
            index = int(input("Please select your RTMP URL by entering the index: "))
            if 0 <= index < len(available_urls):
                return available_urls[index]
            else:
                print(f"Invalid input. Please select a valid device index from 0 to {len(available_urls)}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


def get_function_base(id: int, mode: str):
    """Get the function to be performed from user input for base."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(
                f"Please select your function:\n"
                f"1 : register speaker profiles\n"
                f"2 : start voice recognition\n"
                f"3 : reset (id:{LIGHT_BLUE}{id}{ENDC})\n"
                f"4 : switch (mode:{LIGHT_BLUE}{mode}{ENDC})\n"
                f"0 : exit\n"
                "Selected function: ")

            if select_fun.strip():  # check if input is not empty after removing leading/trailing whitespace
                return int(select_fun)
            else:
                print('Please enter a value')

        except EOFError:
            print(
                "\nUnexpected input received. If you resized the terminal or pressed certain keys, please avoid doing "
                "so and try again.")

        except ValueError:
            print('Please enter a valid integer')


def get_function_synchronizer(mode: str):
    """Get the function to be performed from user input for synchronizer."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(
                f"Please select your function:\n"
                f"1 : start\n"
                f"2 : switch (mode:{LIGHT_BLUE}{mode}{ENDC})\n"
                f"0 : exit\n"
                f"Selected function: ")

            if select_fun.strip():  # check if input is not empty after removing leading/trailing whitespace
                return int(select_fun)
            else:
                print('Please enter a value')

        except EOFError:
            print(
                "\nUnexpected input received. If you resized the terminal or pressed certain keys, please avoid doing "
                "so and try again.")

        except ValueError:
            print('Please enter a valid integer')


def get_channel_selection(device_info: dict) -> int | None:
    """Get the channel selection for stereo devices.

    Args:
        device_info: PyAudio device info dictionary.
    Returns:
        Channel selection: channel index.
    """
    device_channels = device_info.get('maxInputChannels', 1)
    while True:
        try:
            flush_input()
            selection = input(f"\nDevice has {device_channels} channels. Select channel option:\n"
                              f"<channel_index> : Specify channel index (0-{device_channels - 1})\n"
                              f"e.g: 0 -> First channel only\n"
                              f"e.g: 1 -> Second channel only\n"
                              f"Please select your channel option: ")
            selection = int(selection.strip())
            if 0 <= selection < device_channels:
                return selection
            else:
                print(f"Invalid selection. Please enter a valid integer from 0 to {device_channels - 1}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
