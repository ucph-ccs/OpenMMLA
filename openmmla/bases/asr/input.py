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


def get_number_of_group_members():
    """Get the number of group members from user input."""
    while True:
        try:
            flush_input()
            number = int(input("Please specify how many group members: "))
            if number > 0:
                return number
            else:
                print("Please enter a number greater than 0.")
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
