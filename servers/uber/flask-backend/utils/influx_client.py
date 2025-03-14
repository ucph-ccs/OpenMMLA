import configparser

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS


class InfluxDBClientWrapper(influxdb_client.InfluxDBClient):
    """Extended InfluxDB client that loads configuration from a file and adds custom functionalities."""

    def __init__(self, config_path):
        """Initialize InfluxDB client with configurations"""
        config = configparser.ConfigParser()
        config.read(config_path)

        super().__init__(url=config['InfluxDB']['url'], token=config['InfluxDB']['token'],
                         org=config['InfluxDB']['org'])

        # Initialize APIs
        self.query_api = self.query_api()
        self.write_api = self.write_api(write_options=SYNCHRONOUS)
        self.bucket_api = self.buckets_api()

    def query(self, query):
        """Query data from InfluxDB"""
        return self.query_api.query(query)

    def write(self, bucket, record):
        """Write data to InfluxDB"""
        return self.write_api.write(bucket=bucket, record=record)

    def get_buckets(self):
        """Get all buckets from InfluxDB"""
        return self.bucket_api.find_buckets()

    def create_bucket(self, bucket_name):
        """Create a new bucket in InfluxDB"""
        return self.bucket_api.create_bucket(bucket_name=bucket_name)
