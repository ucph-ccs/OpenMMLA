from datetime import datetime

from openmmla.utils.clean import flush_input


def get_bucket_name(influx_client):
    """Get the bucket name from user input, either select an existing bucket or create a new one."""
    bucket_name = None

    while True:
        flush_input()
        print("------------------------------------------------")
        bucket_list = influx_client.get_buckets()
        bucket_names = [bucket.name for bucket in bucket_list.buckets if bucket.name not in ['_tasks', '_monitoring']]

        try:
            bucket_names = sorted(bucket_names, key=lambda x: datetime.strptime(x.split('_')[1], '%Y-%m-%dT%H:%M:%SZ'))
            for i, name in enumerate(bucket_names, start=1):
                print(f"{i}. {name}")
            bucket_idx = input("Enter the number of the bucket you want to enroll in, or enter 'n' for a new bucket, "
                               "or 'r' to refresh the list: ")
        except Exception as e:
            print(f'No compatible bucket session to sort, {e}')
            bucket_idx = 'n'

        if bucket_idx.isdigit() and 1 <= int(bucket_idx) <= len(bucket_names):
            bucket_name = bucket_names[int(bucket_idx) - 1]
            print(f"Bucket: {bucket_name} has been selected.")
            break
        elif bucket_idx in ['n', 'new']:
            timestamp = datetime.utcnow().isoformat().split('.')[0] + 'Z'
            bucket_name = 'session_' + timestamp
            influx_client.create_bucket(bucket_name=bucket_name)
            print(f"Bucket: {bucket_name} has been created.")
            break
        elif bucket_idx in ['r', 'refresh', '']:
            continue
        else:
            print("Invalid input. Please enter a number from the list, or 'n' for a new bucket, or 'r' to refresh the "
                  "list")

    return bucket_name


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


def get_function_sync_manager():
    """Get the function to be performed from user input for sync manager."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
                               f"2: switch mode\n"
                               f"3: set camera id\n"
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


def get_function_base():
    """Get the function to be performed from user input for base."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
                               f"2: set camera\n"
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
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
                               f"2: set main camera\n"
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


def get_function_visualizer():
    """Get the function to be performed from user input for visualizer."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(f"Please input your operation:\n"
                               f"1: start\n"
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
