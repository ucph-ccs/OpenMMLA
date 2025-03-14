"""This script demonstrates how to perform session analysis based on audio measurement data."""
import os

from openmmla.analytics.asr_diarization.analyze import session_analysis_audio
from openmmla.bases.asr_diarization.input import get_bucket_name
from openmmla.utils.client.influx_client import InfluxDBClientWrapper

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')
influx_client = InfluxDBClientWrapper(config_path)
bucket_name = get_bucket_name(influx_client)
session_analysis_audio(project_dir, bucket_name, influx_client)
