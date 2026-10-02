import argparse
import functools
import os

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
    from openmmla.utils.constants import EVENT_TYPES_ANALYTICS, EVENT_TYPES_ASR, EVENT_TYPES_IPS, EVENT_TYPES_VFA

    try:
        event_type_groups = {
            'asr': EVENT_TYPES_ASR,
            'ips': EVENT_TYPES_IPS,
            'vfa': EVENT_TYPES_VFA,
            'analytics': EVENT_TYPES_ANALYTICS,
        }

        pipeline_options = ['asr', 'ips', 'vfa', 'analytics']
        descriptions = [
            f'Automatic Speech Recognition ({", ".join(EVENT_TYPES_ASR)})',
            f'Indoor Positioning System ({", ".join(EVENT_TYPES_IPS)})',
            f'Video Frame Analysis ({", ".join(EVENT_TYPES_VFA)})',
            f'Online analytics ({", ".join(EVENT_TYPES_ANALYTICS)})',
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


# the folders below the one mmla ses-ana runs in (here the config's) where it, and an older
# layout of the bases, kept a session's own <folder>/<session>/: (folder, what the log names)
LOCAL_ANALYSIS_FOLDERS = (
    ("logs", "analysis logs"),
    ("visualizations", "analysis plots"),
    ("real-time/runtime", "runtime files (old layout)"),
)


def _ask(ask, question: str) -> str:
    """the answer to a y/N question; an input that has ended is a no."""
    try:
        return ask(question)
    except EOFError:
        return ""


def delete_session(influx_client, mongo_client, session_id, *, settings_root=None, log=None, ask=input) -> bool:
    """Delete Session, the one the Sessions tab and `mmla ses-delete` run (openmmla.commands.ses.delete):
    what goes and what stays is read and printed first, then a y/N question names the archive's path
    and what stays, and only y deletes, in the same order (the archive and the dashboard's cache, then
    InfluxDB, then MongoDB), stopping at the first step that fails. True once all of it is gone."""
    from rich.markup import escape

    from openmmla.commands.ses import archive, delete

    log = log or archive._ConsoleCallbacks().log
    settings_root = settings_root or archive._settings_root()
    try:
        plan = delete.plan_session(session_id, influx=influx_client, mongo=mongo_client, project_root=settings_root,
                                   settings_root=settings_root)
    except Exception as error:  # a failure is this question's, not the menu's
        log(f"[red]✗ What Delete Session would remove could not be read: {escape(str(error))}. Nothing is "
            f"deleted.[/red]")
        return False
    for line in delete.describe_session_plan(plan, "Answer y below to delete"):
        log(line)
    if not plan.ready:
        return False
    if not delete.confirmed(_ask(ask, delete.session_question(plan))):
        log("[yellow]Nothing was deleted.[/yellow]")
        return False
    return delete.delete_session(plan, influx=influx_client, mongo=mongo_client, log=log, header=False)


def create_new_session(mongo_client) -> None:
    """Create a new session with experiment/group naming."""
    try:
        from openmmla.utils.experiments import get_participant_aliases, select_experiment_and_group
        from openmmla.utils.input import _make_session_id
        exp_id, group_id = select_experiment_and_group()
        session_id = _make_session_id(exp_id, group_id, datetime.now(timezone.utc))
        participants = list(get_participant_aliases(exp_id, group_id).values())
        mongo_client.create_session(session_id, exp_id, group_id, participants=participants)
        print(f"✅ Successfully created new session: {session_id}")
    except Exception as e:
        print(f"❌ Failed to create new session: {e}")


def delete_local_files(session_id, mongo_client=None, *, analysis_dir=None, settings_root=None, log=None,
                       ask=input) -> bool:
    """Delete Files on this machine, the one the Sessions tab (Local) and `mmla ses-delete <session>
    --files-on local` run: this checkout's artifacts/<session>/ and collection/<session>/, and
    <folder>/<session>/ of `analysis_dir` for each of LOCAL_ANALYSIS_FOLDERS, with the same checks
    (a folder named exactly the session id directly inside its root, no link, the root itself never).
    The folders and their sizes are printed first, and only y deletes them. True once they are gone."""
    from rich.markup import escape

    from openmmla.commands.ses import archive, delete

    log = log or archive._ConsoleCallbacks().log
    settings_root = settings_root or archive._settings_root()
    extra = []
    if analysis_dir:
        base = os.path.abspath(str(analysis_dir))
        extra = [delete.FilesRoot(os.path.join(base, folder), what) for folder, what in LOCAL_ANALYSIS_FOLDERS]
    record = None
    if mongo_client is not None:
        record = archive._find_record(mongo_client, session_id)[0]
    try:
        plan = delete.plan_files("local", session_id, project_root=settings_root, record=record,
                                 settings_root=settings_root, extra_roots=extra)
    except Exception as error:  # a failure is this question's, not the menu's
        log(f"[red]✗ The files on this machine could not be read: {escape(str(error))}. Nothing is deleted.[/red]")
        return False
    for line in delete.describe_files_plan(plan, "Answer y below to delete"):
        log(line)
    if not plan.ready:
        return False
    if not delete.confirmed(_ask(ask, delete.files_question(plan))):
        log("[yellow]Nothing was deleted.[/yellow]")
        return False
    return delete.delete_files(plan, log, header=False)


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
                "📁 Delete Local Files",
            ]
            descriptions = [
                "Create a new session in the database",
                "Delete a session everywhere central, as the Sessions tab does: its archive, the dashboard's "
                "cache, its InfluxDB events and its MongoDB document (asks first)",
                "Clean up selected event types for a session",
                "Delete a session's folders on this machine, as Delete Files does: artifacts/, collection/, and "
                "ses-ana's logs and plots (asks first)",
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
                    delete_local_files(session_id, mongo_client, analysis_dir=os.path.dirname(config_path))
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
