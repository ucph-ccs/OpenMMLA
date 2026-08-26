from openmmla.utils.clean import flush_input
from .enums import LIGHT_BLUE, ENDC


def get_function_base(chosen_camera: str | None, camera_seed: str | None, camera_angle: str | None, base_id: int | None,
                      mode: str | None):
    """Get the function to be performed from user input for base."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
                               f"2: set camera (camera:{LIGHT_BLUE}{chosen_camera}{ENDC}, "
                               f"camera seed:{LIGHT_BLUE}{camera_seed}{ENDC}, "
                               f"camera angle:{LIGHT_BLUE}{camera_angle}{ENDC}, "
                               f"base:{LIGHT_BLUE}{base_id}{ENDC}, "
                               f"mode:{LIGHT_BLUE}{mode}{ENDC})\n"
                               f"3: switch mode\n"
                               f"4: reinitialize (reload config)\n"
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


def get_function_synchronizer():
    """Get the function to be performed from user input for synchronizer."""
    while True:
        try:
            flush_input()
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
                               f"2: reinitialize (reload config)\n"
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


def get_mode():
    """Get the operating mode from user input."""
    while True:
        try:
            flush_input()
            selected_mode = int(input("Please select the mode:"
                                      "\n1. Capture, store video frame locally without analyzing"
                                      "\n2. Analyze, analyze locally stored video frame without recording"
                                      "\n3. Live, record and analyze on-the-fly"
                                      "\nSelected mode:"))
            if selected_mode == 1:
                return 'capture'
            elif selected_mode == 2:
                return 'analyze'
            elif selected_mode == 3:
                return 'live'
            else:
                print("Invalid mode, please select again.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


