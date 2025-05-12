from openmmla.utils.clean import flush_input
from .enums import LIGHT_BLUE, ENDC


def get_mode():
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


def get_input_device_index(available_indexes: list[int]) -> int:
    "Get the input device index from user input."""
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
    "Get the RTMP URL from user input."""
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


def get_function_synchronizer():
    """Get the function to be performed from user input for synchronizer."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(
                "Please select your function:\n"
                "1 : start\n"
                "0 : exit\n"
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


def get_channel_selection(device_info: dict) -> str | None:
    """Get the channel selection for stereo devices.

    Args:
        device_info: PyAudio device info dictionary.
    Returns:
        Channel selection: 'left', 'right', or None.
    """
    device_channels = device_info.get('maxInputChannels', 1)

    if device_channels == 2:
        while True:
            try:
                flush_input()
                selection = input(f"\nDevice has {device_channels} channels (stereo). Select channel option:\n"
                                  "1 : Left channel only\n"
                                  "2 : Right channel only\n"
                                  "3 : Mix (mono)\n"
                                  "None: Stereo (left and right)\n"
                                  "Selected option: ")

                if selection.strip() == "":
                    return None
                elif selection == "1":
                    return "left"
                elif selection == "2":
                    return "right"
                elif selection == "3":
                    return "mix"
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer or press enter for default.")

    return None
