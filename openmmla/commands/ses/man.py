import argparse
import functools
import os

import shutil
from datetime import datetime, timezone


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-man",
        description="Manage session data and local data.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    return parser


def cleanup_session_events(influx_client, session_id) -> None:
    """Clean up selected event types for a session."""
    from openmmla.utils.input import multi_interactive_menu
    from openmmla.utils.constants import EVENT_TYPES_ASR, EVENT_TYPES_IPS, EVENT_TYPES_VFA

    try:
        event_type_groups = {
            'asr': EVENT_TYPES_ASR,
            'ips': EVENT_TYPES_IPS,
            'vfa': EVENT_TYPES_VFA,
        }

        pipeline_options = ['asr', 'ips', 'vfa']
        descriptions = [
            f'Automatic Speech Recognition ({", ".join(EVENT_TYPES_ASR)})',
            f'Indoor Positioning System ({", ".join(EVENT_TYPES_IPS)})',
            f'Video Frame Analysis ({", ".join(EVENT_TYPES_VFA)})',
        ]

        selected_indices = multi_interactive_menu("Select Pipelines to Clean Up", pipeline_options, descriptions, exit_on_q=False, prompt_enter=False)

        if not selected_indices:
            print("No pipelines selected. Cleanup cancelled.")
            return

        selected_pipelines = [pipeline_options[i] for i in selected_indices]

        event_types_to_delete = []
        for pipeline in selected_pipelines:
            event_types_to_delete.extend(event_type_groups[pipeline])

        print(f"\nThe following event types will be deleted for session '{session_id}':")
        for et in event_types_to_delete:
            print(f"  - {et}")

        confirm = input("\nAre you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return

        influx_client.delete_event_types(session_id, event_types_to_delete)
        print(f"✅ Successfully cleaned up event types for session: {session_id}")
    except Exception as e:
        print(f"❌ Failed to clean up session events: {e}")


def delete_session(influx_client, mongo_client, session_id) -> None:
    """Delete all data for a session from both InfluxDB and MongoDB."""
    try:
        confirm = input(f"Session: {session_id} will be deleted from InfluxDB and MongoDB. Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Deletion cancelled.")
            return
        influx_client.delete_session_data(session_id)
        mongo_client.delete_session(session_id)
        print(f"✅ Successfully deleted session: {session_id}")
    except Exception as e:
        print(f"❌ Failed to delete session: {e}")


def create_new_session(mongo_client) -> None:
    """Create a new session with experiment/group naming."""
    try:
        from openmmla.utils.experiments import select_experiment_and_group
        from openmmla.utils.input import _make_session_id
        exp_id, group_id = select_experiment_and_group()
        session_id = _make_session_id(exp_id, group_id, datetime.now(timezone.utc))
        mongo_client.create_session(session_id, exp_id, group_id)
        print(f"✅ Successfully created new session: {session_id}")
    except Exception as e:
        print(f"❌ Failed to create new session: {e}")


def cleanup_local_data(project_dir, session_id) -> None:
    """Clean up local data associated with the session."""
    try:
        confirm = input(f"Local data for session: {session_id} will be cleaned up. Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return
        directories = [
            os.path.join(project_dir, 'logger', f'*{session_id}*'),
            os.path.join(project_dir, 'logs', f'*{session_id}*'),
            os.path.join(project_dir, 'visualizations', f'*{session_id}*'),
            os.path.join(project_dir, 'real-time', 'runtime', session_id)
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
        print(f"✅ Successfully cleaned up local data for session: {session_id}")
    except Exception as e:
        print(f"❌ Failed to clean up local data: {e}")


def run_session_management(args):
    print(f"\033]0;Session Management\007")

    from openmmla.utils.logger import get_logger
    from openmmla.utils.input import select_session, interactive_menu
    from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper

    logger = get_logger('session_management')

    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    while True:
        try:
            influx_client = InfluxDBClientWrapper(config_path)
            mongo_client = MongoDBClientWrapper(config_path)
            options = [
                "➕ Create New Session",
                "🗑️  Delete Session",
                "🧹 Clean Up Session Events",
                "📁 Clean Up Local Data",
            ]
            descriptions = [
                "Create a new session in the database",
                "Delete a session from InfluxDB and MongoDB",
                "Clean up selected event types for a session",
                "Clean up local files (logs, visualizations, runtime, etc.)",
            ]

            operation = interactive_menu("Session Management", options, descriptions, exit_on_q=True, prompt_enter=True)

            if operation == 0:
                create_new_session(mongo_client)
            elif operation == 1:
                session_id = select_session(mongo_client)
                if session_id:
                    delete_session(influx_client, mongo_client, session_id)
            elif operation == 2:
                session_id = select_session(mongo_client)
                if session_id:
                    cleanup_session_events(influx_client, session_id)
            elif operation == 3:
                session_id = select_session(mongo_client)
                if session_id:
                    cleanup_local_data(os.path.dirname(config_path), session_id)
        except KeyboardInterrupt as e:
            if "Exit" in str(e):
                print("\n👋 Goodbye!")
                break
            else:
                logger.warning("During running session management, catch: KeyboardInterrupt, Come back to the main menu.", exc_info=True)
        except Exception as e:
            logger.warning(f"During running session management, catch: {e}, Come back to the main menu.", exc_info=True)


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.utils.args import print_arguments
    print_arguments(args)

    run_session_management(args)


if __name__ == "__main__":
    main()
