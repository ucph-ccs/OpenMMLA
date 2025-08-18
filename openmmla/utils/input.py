from datetime import datetime, timezone

from .clean import flush_input
from .client import InfluxDBClientWrapper


def select_bucket(influx_client: InfluxDBClientWrapper) -> str:
    """Get the bucket name from the user."""
    bucket_name = None

    while True:
        flush_input()
        print("------------------------------------------------")
        bucket_list = influx_client.get_buckets()
        bucket_names = [bucket.name for bucket in bucket_list.buckets if bucket.name not in ['_tasks', '_monitoring']]
        bucket_names = [name for name in bucket_names if 'session_' in name]

        try:
            bucket_names = sorted(bucket_names, key=lambda x: datetime.strptime(x.split('_')[1], '%Y-%m-%dT%H:%M:%SZ'))
            if len(bucket_names) == 0:
                print("No bucket sessions found.")
                break
            for i, name in enumerate(bucket_names, start=1):
                print(f"{i}. {name}")
            bucket_idx = input("Enter the number of the bucket you want to select:")
        except Exception as e:
            print(f'No compatible bucket session to sort, {e}')
            break

        if bucket_idx.isdigit() and 1 <= int(bucket_idx) <= len(bucket_names):
            bucket_name = bucket_names[int(bucket_idx) - 1]
            print(f"Bucket: {bucket_name} has been selected.")
            break
        else:
            print("Invalid input. Please enter a number from the list.")
            continue

    return bucket_name


def select_or_create_bucket(influx_client):
    """Get the bucket name from user input, either select an existing bucket or create a new one."""
    bucket_name = None

    while True:
        flush_input()
        print("------------------------------------------------")
        bucket_list = influx_client.get_buckets()
        bucket_names = [bucket.name for bucket in bucket_list.buckets if bucket.name not in ['_tasks', '_monitoring']]
        bucket_names = [name for name in bucket_names if 'session_' in name]

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
            timestamp = datetime.now(timezone.utc).isoformat().split('.')[0] + 'Z'
            bucket_name = 'session_' + timestamp
            influx_client.create_bucket(bucket_name)
            print(f"Bucket: {bucket_name} has been created.")
            break
        elif bucket_idx in ['r', 'refresh', '']:
            continue
        else:
            print("Invalid input. Please enter a number from the list, or 'n' for a new bucket, or 'r' to refresh the "
                  "list")

    return bucket_name


def get_id():
    """Get the unique base id from user input."""
    while True:
        try:
            flush_input()
            print("------------------------------------------------")
            return int(input("Enter the your base id: "))
        except ValueError:
            print("Invalid input. Please enter an integer as your unique base id.")


def get_number_of_bases() -> int:
    """Get the number of bases to synchronize.

    Returns:
        int: Number of bases
    """
    while True:
        try:
            flush_input()
            num = int(input("Enter the number of bases to synchronize: ") or "1")
            if num > 0:
                return num
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")


def select_participant_descriptions(participant_descriptions_config: dict) -> dict | None:
    """Select participant descriptions for the current session.
    
    Args:
        participant_descriptions_config: Dictionary of participant descriptions from config
                                       Format: {session_key: {tag_id: description}}
        
    Returns:
        dict: Selected participant descriptions as {tag_id: description} mapping, 
              or None if no selection made
    """
    if not participant_descriptions_config:
        print("No participant descriptions available in configuration")
        return None
        
    print("------------------------------------------------")
    print("Available participant description sets:")
    
    description_keys = list(participant_descriptions_config.keys())
    for idx, key in enumerate(description_keys):
        participant_count = len(participant_descriptions_config[key])
        print(f"{idx + 1}: {key} ({participant_count} participants)")
    
    print(f"{len(description_keys) + 1}: None - No participant descriptions")
    
    while True:
        try:
            flush_input()
            selection_input = input(f"Choose participant description set (1-{len(description_keys) + 1}) or press Enter for None: ")
            
            if selection_input == '':
                return None
                
            selection = int(selection_input)
            if 1 <= selection <= len(description_keys):
                selected_key = description_keys[selection - 1]
                selected_descriptions = participant_descriptions_config[selected_key]
                print(f"Selected participant descriptions: {selected_key}")
                print("Participants:")
                for tag_id, description in selected_descriptions.items():
                    print(f"  Tag ID {tag_id}: {description}")
                return selected_descriptions
            elif selection == len(description_keys) + 1:
                return None
            else:
                print("Invalid selection. Please choose a valid option.")
        except ValueError:
            print("Please enter a valid number or press Enter for None.")
