"""This script demonstrates how to control the nodes."""
import os

from openmmla.bases.ips.input import get_bucket_name
from openmmla.utils.clean import flush_input
from openmmla.utils.client import InfluxDBClientWrapper, RedisClientWrapper
from openmmla.utils.logger import get_logger

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')
logger = get_logger('control')

while True:
    try:
        influx_client = InfluxDBClientWrapper(config_path)
        redis_client = RedisClientWrapper(config_path)

        flush_input()
        operation = input(f"Please input your operation:\n"
                          f"1: reconnect nodes\n"
                          f"2: disconnect nodes\n"
                          f"0: exit\n"
                          f"Selected function:")
        if operation == '1':
            bucket_name = get_bucket_name(influx_client)
            redis_client.publish(f"{bucket_name}/control", 'START')
            print("Start signal sent.")
        elif operation == '2':
            bucket_name = get_bucket_name(influx_client)
            redis_client.publish(f"{bucket_name}/control", 'STOP')
            print("Stop signal sent.")
        elif operation == '0':
            break
        else:
            print("Invalid operation. Please input 1, or 0.")
    except (Exception, KeyboardInterrupt) as e:
        logger.warning(
            f"During running the control base, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
            exc_info=True)
