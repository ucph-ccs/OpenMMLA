from datetime import datetime, timezone
from .clean import flush_input


def get_bucket_name(influx_client):
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
            influx_client.create_bucket(bucket_name=bucket_name)
            print(f"Bucket: {bucket_name} has been created.")
            break
        elif bucket_idx in ['r', 'refresh', '']:
            continue
        else:
            print("Invalid input. Please enter a number from the list, or 'n' for a new bucket, or 'r' to refresh the "
                  "list")

    return bucket_name
