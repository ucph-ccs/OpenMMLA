"""This script demonstrates how to perform session analysis based on video measurement data"""
import os
import sys

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = 'conf/video_base.ini'
sys.path.append(project_dir)

from openmmla_vision.utils.analyze_utils import session_analysis_video
from openmmla_vision.utils.influx_client import InfluxDBClientWrapper
from openmmla_vision.utils.input_utils import get_bucket_name

influx_client = InfluxDBClientWrapper(config_path)
bucket_name = get_bucket_name(influx_client)
session_analysis_video(project_dir, bucket_name, influx_client)
