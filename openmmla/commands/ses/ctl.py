import argparse
import functools
import os

from openmmla.utils.clean import flush_input


def get_parser():
    parser = argparse.ArgumentParser(
        prog="openmmla ses-ctl",
        description="Start/stop bucket session.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    return parser


def run_session_control(args):
    print(f"\033]0; Control Base \007")

    from openmmla.utils.logger import get_logger
    from openmmla.utils.client import InfluxDBClientWrapper, RedisClientWrapper
    from openmmla.utils.input import get_bucket_name

    logger = get_logger('control')

    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    while True:
        try:
            influx_client = InfluxDBClientWrapper(config_path)
            redis_client = RedisClientWrapper(config_path)

            flush_input()
            operation = input(
                "Please input your operation:\n"
                "1: Reconnect nodes\n"
                "2: Disconnect nodes\n"
                "0: Exit\n"
                "Selected function: "
            ).strip()

            if operation == '1':
                bucket_name = get_bucket_name(influx_client)
                redis_client.publish(f"{bucket_name}/control", 'START')
                print("✅ Start signal sent.")
            elif operation == '2':
                bucket_name = get_bucket_name(influx_client)
                redis_client.publish(f"{bucket_name}/control", 'STOP')
                print("🛑 Stop signal sent.")
            elif operation == '0':
                break
            else:
                print("Invalid operation. Please input 1, 2, or 0.")
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

    run_session_control(args)


if __name__ == "__main__":
    main()
