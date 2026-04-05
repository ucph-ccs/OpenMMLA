import os

from openmmla.utils.querys import fetch_and_process_data, save_to_json_file


def vfa_session_analysis(project_dir, bucket_name, influx_client):
    """Retrieves action recognition data from InfluxDB and saves it to a JSON file.
    
    Args:
        project_dir: Path to the project directory
        bucket_name: Name of the bucket in InfluxDB
        influx_client: InfluxDB client wrapper instance
    """
    if not project_dir or not os.path.exists(project_dir):
        print("Warning: project_dir is not set or does not exist. Setting to current working directory.")
        project_dir = os.getcwd()

    logs_dir = os.path.join(project_dir, 'logs')
    log_dir = os.path.join(logs_dir, f'{bucket_name}')
    os.makedirs(log_dir, exist_ok=True)

    # Fetch and log action recognition data
    action_data = fetch_and_process_data(bucket_name, "action_recognition", influx_client)
    action_json_file_path = save_to_json_file(bucket_name, action_data, "action_recognition", log_dir)

    print(f"Action Recognition session analysis completed for session: {bucket_name}")
    print(f"Data saved to: {action_json_file_path}")

    return action_json_file_path
