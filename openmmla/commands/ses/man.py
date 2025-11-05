import argparse
import functools
import os

import shutil
from datetime import datetime, timezone


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-man",
        description="Manage bucket data and local data.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    return parser


def cleanup_bucket_data(influx_client, bucket_name) -> None:
    """Clean up selected measurement data in the specified bucket."""
    from openmmla.utils.input import multi_interactive_menu
    
    try:
        # Define measurement groups by pipeline
        measurement_groups = {
            'asr': ['speaker_recognition', 'speaker_transcription'],
            'ips': ['badge_relation', 'badge_translation', 'badge_rotation'],
            'vfa': ['action_recognition']
        }
        
        # Show interactive menu for measurement selection
        pipeline_options = ['asr', 'ips', 'vfa']
        descriptions = [
            'Automatic Speech Recognition (speaker_recognition, speaker_transcription)',
            'Indoor Positioning System (badge_relation, badge_translation, badge_rotation)',
            'Video Frame Analysis (action_recognition)'
        ]
        
        selected_indices = multi_interactive_menu("Select Pipelines to Clean Up", pipeline_options, descriptions, exit_on_q=False, prompt_enter=False)
        
        if not selected_indices:
            print("No pipelines selected. Cleanup cancelled.")
            return
        
        # Convert indices to pipeline names
        selected_pipelines = [pipeline_options[i] for i in selected_indices]
        
        # Build list of measurements to delete
        measurements_to_delete = []
        for pipeline in selected_pipelines:
            measurements_to_delete.extend(measurement_groups[pipeline])
        
        # Confirm deletion
        print(f"\nThe following measurements will be deleted from bucket '{bucket_name}':")
        for measurement in measurements_to_delete:
            print(f"  - {measurement}")
        
        confirm = input("\nAre you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return
        
        # Delete selected measurements
        influx_client.delete_measurements(bucket_name, measurements_to_delete)
        print(f"✅ Successfully cleaned up measurements in bucket: {bucket_name}")
    except Exception as e:
        print(f"❌ Failed to clean up bucket: {e}")


def delete_bucket(influx_client, bucket_name) -> None:
    """Delete the specified bucket."""
    try:
        confirm = input(f"Bucket: {bucket_name} will be deleted. Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Deletion cancelled.")
            return
        influx_client.delete_bucket(bucket_name)
        print(f"✅ Successfully deleted bucket: {bucket_name}")
    except Exception as e:
        print(f"❌ Failed to delete bucket: {e}")


def create_new_bucket(influx_client) -> None:
    """Create a new bucket."""
    try:
        timestamp = datetime.now(timezone.utc).isoformat().split('.')[0] + 'Z'
        bucket_name = 'session_' + timestamp
        influx_client.create_bucket(bucket_name)
        print(f"✅ Successfully created new bucket: {bucket_name}")
    except Exception as e:
        print(f"❌ Failed to create new bucket: {e}")


def cleanup_local_data(project_dir, bucket_name) -> None:
    """Clean up local data associated with the bucket."""
    try:
        confirm = input(f"Local data for bucket: {bucket_name} will be cleaned up. Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return
        directories = [
            os.path.join(project_dir, 'logger', f'*{bucket_name}*'),
            os.path.join(project_dir, 'logs', f'*{bucket_name}*'),
            os.path.join(project_dir, 'visualizations', f'*{bucket_name}*'),
            os.path.join(project_dir, 'real-time', 'runtime', bucket_name)
        ]
        for dir_pattern in directories:
            import glob
            for dir_path in glob.glob(dir_pattern):
                if os.path.exists(dir_path):
                    if os.path.isdir(dir_path):
                        shutil.rmtree(dir_path)
                    else:
                        os.remove(dir_path)
                    print(f"✅ Cleaned up: {dir_path}")
        print(f"✅ Successfully cleaned up local data for bucket: {bucket_name}")
    except Exception as e:
        print(f"❌ Failed to clean up local data: {e}")


def run_bucket_management(args):
    print(f"\033]0;Bucket Management\007")

    from openmmla.utils.logger import get_logger
    from openmmla.utils.input import select_bucket, interactive_menu
    from openmmla.utils.client import InfluxDBClientWrapper

    logger = get_logger('bucket_management')

    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    while True:
        try:
            influx_client = InfluxDBClientWrapper(config_path)
            options = [
                "➕ Create New Bucket",
                "🗑️  Delete Bucket", 
                "🧹 Clean Up Bucket Data",
                "📁 Clean Up Local Data",
            ]
            descriptions = [
                "Create a new session bucket in the database",
                "Delete an existing bucket from the database",
                "Clean up selected measurements in an existing bucket",
                "Clean up local files (logs, visualizations, runtime, etc.)",
            ]
            
            operation = interactive_menu("Bucket Management", options, descriptions, exit_on_q=True, prompt_enter=True)
            
            if operation == 0:
                create_new_bucket(influx_client)
            elif operation == 1:
                bucket_name = select_bucket(influx_client)
                if bucket_name:
                    delete_bucket(influx_client, bucket_name)
            elif operation == 2:
                bucket_name = select_bucket(influx_client)
                if bucket_name:
                    cleanup_bucket_data(influx_client, bucket_name)
            elif operation == 3:
                bucket_name = select_bucket(influx_client)
                if bucket_name:
                    cleanup_local_data(os.path.dirname(config_path), bucket_name)
        except KeyboardInterrupt as e:
            if "Exit" in str(e):
                print("\n👋 Goodbye!")
                break
            else:
                logger.warning("During running bucket management, catch: KeyboardInterrupt, Come back to the main menu.", exc_info=True)
        except Exception as e:
            logger.warning(f"During running bucket management, catch: {e}, Come back to the main menu.", exc_info=True)


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.utils.args import print_arguments
    print_arguments(args)

    run_bucket_management(args)


if __name__ == "__main__":
    main()
