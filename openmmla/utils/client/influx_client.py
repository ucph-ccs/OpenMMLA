import logging
import yaml
from datetime import datetime, timedelta, timezone
from typing import Any

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from openmmla.utils.constants import INFLUXDB_MEASUREMENT, INFLUXDB_DEFAULT_BUCKET

logger = logging.getLogger(__name__)


def _to_flux_time(dt: datetime) -> str:
    """convert datetime to Flux range-compatible time(v: ...) expression."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return f'time(v: "{dt.isoformat()}")'


class InfluxDBClientWrapper:
    """InfluxDB client for single-bucket, single-measurement schema with event_type tags."""

    def __init__(self, config_path: str):
        config = yaml.safe_load(open(config_path, 'r'))
        influx_config = config['InfluxDB']

        self.url = influx_config['url']
        self.token = influx_config['token']
        self.org = influx_config['org']
        self.bucket = influx_config.get('bucket', INFLUXDB_DEFAULT_BUCKET)

        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.delete_api = self.client.delete_api()

        logger.info("InfluxDB connected: %s, bucket: %s", self.url, self.bucket)

    # ---- write ----

    def write_event(self, session_id: str, event_type: str, fields: dict[str, Any],
                    timestamp: datetime | None = None) -> bool:
        try:
            point = Point(INFLUXDB_MEASUREMENT)
            point = point.tag("session_id", session_id)
            point = point.tag("event_type", event_type)

            for key, value in fields.items():
                if value is not None:
                    if isinstance(value, (int, float)):
                        point = point.field(key, float(value))
                    else:
                        point = point.field(key, str(value))

            if timestamp:
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                point = point.time(timestamp, WritePrecision.NS)
            elif "window_end_time" in fields and fields["window_end_time"]:
                end_time = fields["window_end_time"]
                if isinstance(end_time, (int, float)) and end_time > 0:
                    point = point.time(
                        datetime.fromtimestamp(end_time, tz=timezone.utc),
                        WritePrecision.NS,
                    )

            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True
        except Exception as e:
            logger.warning("write_event failed: %s", e)
            return False

    # ---- query ----

    def query_events(self, session_id: str, event_type: str,
                     start_time: datetime | None = None,
                     end_time: datetime | None = None) -> list[dict]:
        if start_time is None:
            start_time = datetime.now(timezone.utc) - timedelta(days=365)
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: {_to_flux_time(start_time)}, stop: {_to_flux_time(end_time)})
            |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}")
            |> filter(fn: (r) => r.session_id == "{session_id}")
            |> filter(fn: (r) => r.event_type == "{event_type}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"])
        '''
        return self._execute_query(query)

    def query_latest_event(self, session_id: str, event_type: str,
                           lookback_seconds: int = 20) -> dict | None:
        start_time = datetime.now(timezone.utc) - timedelta(seconds=lookback_seconds)
        query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: {_to_flux_time(start_time)})
            |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}")
            |> filter(fn: (r) => r.session_id == "{session_id}")
            |> filter(fn: (r) => r.event_type == "{event_type}")
            |> last()
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        events = self._execute_query(query)
        return events[0] if events else None

    def get_all_session_ids(self) -> list[str]:
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: -365d)
                |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}")
                |> keep(columns: ["session_id"])
                |> distinct(column: "session_id")
            '''
            result = self.query_api.query(org=self.org, query=query)
            session_ids = set()
            for table in result:
                for record in table.records:
                    sid = record.values.get("session_id")
                    if sid:
                        session_ids.add(sid)
            return sorted(list(session_ids))
        except Exception as e:
            logger.warning("get_all_session_ids failed: %s", e)
            return []

    def count_session_events(self, session_id: str) -> int:
        try:
            query = f'''
                from(bucket: "{self.bucket}")
                |> range(start: 0)
                |> filter(fn: (r) => r.session_id == "{session_id}")
                |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}")
                |> count()
            '''
            result = self.query_api.query(org=self.org, query=query)
            for table in result:
                for record in table.records:
                    return int(record.get_value() or 0)
            return 0
        except Exception as e:
            logger.warning("count_session_events failed: %s", e)
            return 0

    def query(self, query_str: str):
        """execute a raw Flux query (backward compatibility)."""
        return self.query_api.query(query_str)

    # ---- delete ----

    def delete_session_data(self, session_id: str) -> bool:
        try:
            predicate = f'session_id="{session_id}"'
            self.delete_api.delete(
                datetime(1970, 1, 1, tzinfo=timezone.utc),
                datetime.now(timezone.utc),
                predicate,
                bucket=self.bucket,
                org=self.org,
            )
            logger.info("session data deleted: %s", session_id)
            return True
        except Exception as e:
            logger.warning("delete_session_data failed: %s", e)
            return False

    def delete_event_types(self, session_id: str, event_types: list[str]) -> bool:
        try:
            for event_type in event_types:
                predicate = f'session_id="{session_id}" AND event_type="{event_type}"'
                self.delete_api.delete(
                    datetime(1970, 1, 1, tzinfo=timezone.utc),
                    datetime.now(timezone.utc),
                    predicate,
                    bucket=self.bucket,
                    org=self.org,
                )
            logger.info("event types %s deleted for session: %s", event_types, session_id)
            return True
        except Exception as e:
            logger.warning("delete_event_types failed: %s", e)
            return False

    # ---- internal ----

    def _execute_query(self, query: str) -> list[dict]:
        try:
            result = self.query_api.query(org=self.org, query=query)
            events = []
            for table in result:
                for record in table.records:
                    event = {
                        "time": record.get_time(),
                        "session_id": record.values.get("session_id"),
                        "event_type": record.values.get("event_type"),
                    }
                    for key, value in record.values.items():
                        if key not in (
                            "_start", "_stop", "_time", "_measurement",
                            "session_id", "event_type", "result", "table",
                        ):
                            event[key] = value
                    events.append(event)
            return events
        except Exception as e:
            logger.warning("query execution failed: %s", e)
            return []

    def close(self):
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("InfluxDB connection closed")
