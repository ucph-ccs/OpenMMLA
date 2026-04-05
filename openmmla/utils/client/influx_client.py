import influxdb_client
import yaml

from datetime import datetime, timedelta
from influxdb_client.client.write_api import SYNCHRONOUS


class InfluxDBClientWrapper(influxdb_client.InfluxDBClient):
    """Extended InfluxDB client that loads configuration from a file and adds custom functionalities."""

    def __init__(self, config_path):
        """Initialize InfluxDB client with configurations"""
        config = yaml.safe_load(open(config_path, 'r'))
        super().__init__(url=config['InfluxDB']['url'], token=config['InfluxDB']['token'],
                         org=config['InfluxDB']['org'])

        # Initialize APIs
        self.query_api = self.query_api()
        self.write_api = self.write_api(write_options=SYNCHRONOUS)
        self.bucket_api = self.buckets_api()
        self.delete_api = self.delete_api()

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

    def delete_bucket(self, bucket_name):
        """Delete a bucket from InfluxDB"""
        bucket = self.bucket_api.find_bucket_by_name(bucket_name)
        if not bucket:
            raise ValueError(f"Bucket {bucket_name} not found.")
        return self.bucket_api.delete_bucket(bucket)
    
    def delete_measurements(self, bucket_name, measurements):
        """Delete specific measurements from a bucket.
        
        Args:
            bucket_name: Name of the bucket
            measurements: List of measurement names to delete
        """
        bucket = self.bucket_api.find_bucket_by_name(bucket_name)
        if not bucket:
            raise ValueError(f"Bucket {bucket_name} not found.")
        
        # extract the bucket start time from the bucket name
        # new format: <exp>_<group>_YYMMDDTHHMMZ  |  legacy: session_YYYY-MM-DDTHH:MM:SSZ
        start_time = None
        last_seg = bucket_name.rsplit('_', 1)[-1]
        try:
            start_time = datetime.strptime(last_seg, '%y%m%dT%H%MZ')
        except ValueError:
            pass
        if start_time is None:
            try:
                timestamp_str = bucket_name.split('_', 1)[1]
                start_time = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
            except (IndexError, ValueError):
                start_time = datetime.now() - timedelta(days=365)
        
        end_time = datetime.now()
        
        # Build the delete predicate for each measurement
        for measurement in measurements:
            predicate = f'_measurement="{measurement}"'
            self.delete_api.delete(start_time, end_time, predicate, bucket.name, self.org)