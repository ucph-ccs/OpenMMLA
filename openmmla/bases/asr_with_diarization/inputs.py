from datetime import datetime

from openmmla.utils.clean import flush_input


def get_mode():
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
    """Get the name of the speaker from user input"""
    flush_input()
    return input("Please enter the name of the speaker: ")


def get_id():
    """Get the unique base id from user input"""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            return int(input("Enter the your base id: "))
        except ValueError:
            print("Invalid input. Please enter an integer as your unique base id.")


def get_number_of_group_members():
    """Get the number of group members from user input"""
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


def get_bucket_name(influx_client) -> str:
    """Get the bucket name from user input or create a new bucket if needed"""
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


def get_function_base():
    """Get the function to be performed from user input for base"""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(
                "Please select your function:\n"
                "1 : register audio to the voice-print library\n"
                "2 : perform voice-print recognition\n"
                "3 : reset port\n"
                "4 : switch mode\n"
                "0 : exit\n"
                "Selected function: ")

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
    """Get the function to be performed from user input for synchronizer"""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            select_fun = input(
                "Please select your function:\n"
                "1 : start\n"
                "0 : exit\n"
                "Selected function: ")

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
