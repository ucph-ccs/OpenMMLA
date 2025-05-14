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
    """Clean up all data in the specified bucket."""
    try:
        influx_client.delete_bucket(bucket_name)
        influx_client.create_bucket(bucket_name)
        print(f"✅ Successfully cleaned up bucket: {bucket_name}")
    except Exception as e:
        print(f"❌ Failed to clean up bucket: {e}")


def delete_bucket(influx_client, bucket_name) -> None:
    """Delete the specified bucket."""
    try:
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
    print(f"\033]0;Bucket Cleanup\007")

    from openmmla.utils.clean import flush_input
    from openmmla.utils.input import select_bucket
    from openmmla.utils.logger import get_logger
    from openmmla.utils.client import InfluxDBClientWrapper

    logger = get_logger('cleanup')

    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    while True:
        try:
            influx_client = InfluxDBClientWrapper(config_path)

            flush_input()
            operation = input(
                "Please select an operation:\n"
                "1: Create new bucket (database)\n"
                "2: Delete bucket (database)\n"
                "3: Clean up bucket data (database)\n"
                "4: Clean up local data (logs, visualizations, runtime, etc.)\n"
                "0: Exit\n"
                "Selected function: "
            ).strip()

            if operation in ['2', '3', '4']:
                bucket_name = select_bucket(influx_client)
                if bucket_name is None:
                    continue
                if operation == '2':
                    delete_bucket(influx_client, bucket_name)
                elif operation == '3':
                    cleanup_bucket_data(influx_client, bucket_name)
                elif operation == '4':
                    cleanup_local_data(os.path.dirname(config_path), bucket_name)
            elif operation == '1':
                create_new_bucket(influx_client)
            elif operation == '0':
                break
            else:
                print("Invalid operation. Please input 1, 2, 3, 4, or 0.")
        except (Exception, KeyboardInterrupt) as e:
            logger.warning(
                f"Interrupted: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}. Returning to main menu.",
                exc_info=True
            )


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.utils.args import print_arguments
    print_arguments(args)

    run_bucket_management(args)


if __name__ == "__main__":
    main()
