from openmmla.utils.clean import flush_input
from .enums import LIGHT_BLUE, ENDC


def get_function_calibrator():
    """Get the function to be performed from user input for calibrator."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: capture chess pattern images from you camera\n"
                               f"2: calibrate your selected camera\n"
                               f"0: exit\n"
                               f"Selected function: ")
            if select_fun.strip():
                return int(select_fun)
            else:
                print('Please enter a value')
        except EOFError:
            print(
                "\nUnexpected input received. If you resized the terminal or pressed certain keys, please avoid doing "
                "so and try again.")
        except ValueError:
            print('Please enter a valid integer')


def get_function_sync_manager(main_id: str, alt_id: str, sync: bool):
    """Get the function to be performed from user input for sync manager."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
                               f"2: set camera id (main:{LIGHT_BLUE}{main_id}{ENDC}, alt:{LIGHT_BLUE}{alt_id}{ENDC})\n"
                               f"3: switch mode (sync:{LIGHT_BLUE}{sync}{ENDC})\n"
                               f"4: export transformations\n"
                               f"5: clear transformations\n"
                               f"0: exit\n"
                               f"Selected function: ")
            if select_fun.strip():
                return int(select_fun)
            else:
                print('Please enter a value')
        except EOFError:
            print(
                "\nUnexpected input received. If you resized the terminal or pressed certain keys, please avoid doing "
                "so and try again.")
        except ValueError:
            print('Please enter a valid integer')


def get_function_base(chosen_camera: str, stream_source: str, base_id: str, main_id: str):
    """Get the function to be performed from user input for base."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
                               f"2: set camera (camera:{LIGHT_BLUE}{chosen_camera}{ENDC}, camera seed:{LIGHT_BLUE}{stream_source}{ENDC}, base:{LIGHT_BLUE}{base_id}{ENDC}, main:{LIGHT_BLUE}{main_id}{ENDC})\n"
                               f"0: exit\n"
                               f"Selected function: ")

            if select_fun.strip():  # Check if input is not empty after removing leading/trailing whitespace
                return int(select_fun)
            else:
                print('Please enter a value')

        except EOFError:
            print(
                "\nUnexpected input received. If you resized the terminal or pressed certain keys, please avoid doing "
                "so and try again.")

        except ValueError:
            print('Please enter a valid integer')


def get_function_synchronizer(main_id: str):
    """Get the function to be performed from user input for synchronizer."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
                               f"2: set main camera (main:{LIGHT_BLUE}{main_id}{ENDC})\n"
                               f"0: exit\n"
                               f"Selected function: ")

            if select_fun.strip():  # Check if input is not empty after removing leading/trailing whitespace
                return int(select_fun)
            else:
                print('Please enter a value')

        except EOFError:
            print(
                "\nUnexpected input received. If you resized the terminal or pressed certain keys, please avoid doing "
                "so and try again.")

        except ValueError:
            print('Please enter a valid integer')


def get_function_visualizer(dimension: str):
    """Get the function to be performed from user input for visualizer."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
                               f"2: switch (dimension:{LIGHT_BLUE}{dimension}{ENDC})\n"
                               f"0: exit\n"
                               f"Selected function: ")

            if select_fun.strip():  # Check if input is not empty after removing leading/trailing whitespace
                return int(select_fun)
            else:
                print('Please enter a value')

        except EOFError:
            print(
                "\nUnexpected input received. If you resized the terminal or pressed certain keys, please avoid doing "
                "so and try again.")

        except ValueError:
            print('Please enter a valid integer')
