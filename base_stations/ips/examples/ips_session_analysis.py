"""This script demonstrates how to perform session analysis based on IPS measurement data"""
import os

from openmmla.analysis.ips.analyze import ips_session_analysis
from openmmla.utils.input import get_bucket_name
from openmmla.utils.client import InfluxDBClientWrapper

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')

influx_client = InfluxDBClientWrapper(config_path)
bucket_name = get_bucket_name(influx_client)
ips_session_analysis(project_dir, bucket_name, influx_client)
