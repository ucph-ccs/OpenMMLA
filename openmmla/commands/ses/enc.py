import argparse
import functools
import json
import os


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-enc",
        description="Encode online/offline hybrid indicators from session measurements.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150),
    )
    from openmmla.utils.args import add_arguments

    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg("config_path", str, None, "path to the configuration file", shortname="-c", required=True)
    add_arg("session_id", str, None, "session id to encode; if omitted, select interactively")
    add_arg("project_dir", str, None, "project directory used to resolve relative config paths")
    add_arg("output_path", str, None, "optional JSON file for one-shot snapshot export")
    add_arg("task_config", str, None, "task config under root config/tasks, e.g. config/tasks/programming.yaml")
    add_arg("status_profile", str, None, "status profile name defined in config/analytics/status_profiles.yml")
    add_arg("asr_scope", str, None, "ASR attribution scope: participant or group")
    add_arg("window_size", int, None, "window size in seconds override")
    add_arg("step_size", int, None, "step size / polling interval in seconds override")
    add_arg("iterations", int, None, "number of polling iterations in watch mode; omit for continuous polling")
    add_arg("watch", bool, False, "continuously poll measurements instead of running once")
    add_arg("writeback", bool, True, "write participant/group indicators back to InfluxDB")
    return parser


def _select_session_id(config_path: str, explicit_session_id: str | None) -> str:
    if explicit_session_id:
        return explicit_session_id

    from openmmla.utils.client import MongoDBClientWrapper
    from openmmla.utils.input import select_session

    mongo_client = MongoDBClientWrapper(config_path)
    session_id = select_session(mongo_client)
    if not session_id:
        raise ValueError("No session selected.")
    return session_id


def run_session_encoding(args):
    print(f"\033]0; Session Encoding \007")

    from openmmla.analytics.realtime import RealtimeIndicatorEncoder

    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    project_dir = args.project_dir
    if project_dir and not os.path.isabs(project_dir):
        project_dir = os.path.join(os.getcwd(), project_dir)

    session_id = _select_session_id(config_path, args.session_id)
    encoder = RealtimeIndicatorEncoder(
        config_path=config_path,
        session_id=session_id,
        project_dir=project_dir,
        window_size=args.window_size,
        step_size=args.step_size,
        task_config=args.task_config,
        status_profile=args.status_profile,
        asr_scope=args.asr_scope,
        writeback=bool(args.writeback),
    )

    if args.watch:
        encoder.run_forever(
            poll_interval=args.step_size or encoder.pipeline.feature_config.step_size,
            iterations=args.iterations,
        )
        return None

    snapshot = encoder.encode_once()
    if args.output_path:
        output_path = args.output_path
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.getcwd(), output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(snapshot, output_file, indent=2, ensure_ascii=False)
        print(f"Snapshot saved to {output_path}")
    else:
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    return snapshot


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.utils.args import print_arguments

    print_arguments(args)
    run_session_encoding(args)


if __name__ == "__main__":
    main()
