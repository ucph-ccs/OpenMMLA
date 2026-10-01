import json
import os
from typing import Any, TYPE_CHECKING

from openmmla.utils.constants import INFLUXDB_MEASUREMENT, EVENT_TYPE_IPS_TRANSLATION

if TYPE_CHECKING:
    from openmmla.utils.client import InfluxDBClientWrapper


def fetch_and_process_data(session_id: str, event_type: str, influx_client: "InfluxDBClientWrapper") -> list[dict]:
    """Query InfluxDB for specified data and sort by window_start_time."""
    events = influx_client.query_events(session_id, event_type)
    data = [deep_parse_json(e) for e in events]
    data.sort(key=lambda x: x.get('window_start_time', 0))
    return data


def fetch_latest_entry(session_id: str, event_type: str, influx_client: "InfluxDBClientWrapper") -> dict | None:
    """Retrieve the most recent entry from InfluxDB."""
    event = influx_client.query_latest_event(session_id, event_type)
    return deep_parse_json(event) if event else None


def get_node_positions(session_id: str, influx_client: "InfluxDBClientWrapper", timestamp: float, dimension: str = '2d') -> dict:
    """Retrieve badge positions from InfluxDB filtered by window_start_time.

    The window key is the synchronizer's bucket start, a float epoch with sub-second precision, and
    it is stored as a float field; the literal is written out in full so the equality holds. Returns
    an empty dict when no translation was recorded for that window.
    """
    from openmmla.utils.client.influx_client import _to_flux_time
    from datetime import datetime, timezone

    window_start = float(timestamp)
    start_dt = datetime.fromtimestamp(window_start - 20, tz=timezone.utc)
    query = f'''
        from(bucket: "{influx_client.bucket}")
        |> range(start: {_to_flux_time(start_dt)})
        |> filter(fn: (r) => r._measurement == "{INFLUXDB_MEASUREMENT}")
        |> filter(fn: (r) => r.session_id == "{session_id}")
        |> filter(fn: (r) => r.event_type == "{EVENT_TYPE_IPS_TRANSLATION}")
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> filter(fn: (r) => r.window_start_time == {window_start!r})
    '''
    events = influx_client._execute_query(query)
    data = [deep_parse_json(e) for e in events]
    if not data:
        return {}
    translate_dict = data[0]["translations"]

    positions = {'B': (0, 0)} if dimension == '2d' else {'B': (0, 0, 0)}
    for badge_id, translation in translate_dict.items():
        if dimension == '2d':
            x = translation[0][0]
            z = translation[2][0]
            positions[badge_id] = (z, -x)
        else:
            x = translation[0][0]
            y = translation[1][0]
            z = translation[2][0]
            positions[badge_id] = (x, -y, z)
    return positions


def read_json_file(file_path: str) -> Any:
    """Reads and returns the contents of a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return None


def _json_default(value: Any) -> Any:
    """what json cannot write by itself: the `time` the InfluxDB client adds to every row is a
    datetime (Influx _time), written as ISO 8601 with its offset; numpy scalars as numbers."""
    from datetime import date, datetime

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    item = getattr(value, 'item', None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_to_json_file(session_id: str, data: Any, suffix: str, log_dir: str, compact: bool = False) -> str:
    """Saves data to a JSON file named by session_id and suffix; `compact` writes it without
    indentation (a large export, such as the features of a session, is a fraction of the size).
    The file is written beside its name and moved into place once complete, so a failed export
    leaves no cut-off file behind."""
    json_path = os.path.join(log_dir, f"{session_id}_{suffix}.json")
    partial = json_path + '.partial'
    try:
        with open(partial, 'w') as f:
            if compact:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'), default=_json_default)
            else:
                json.dump(data, f, ensure_ascii=False, indent=5, default=_json_default)
        os.replace(partial, json_path)
    except BaseException:
        try:
            os.unlink(partial)
        except OSError:
            pass
        raise
    print(f"{suffix} saved to {session_id}_{suffix}.json")
    return json_path


def convert_json_to_dataframe(json_data: Any, json_columns: list):
    """Converts JSON data into a pandas DataFrame and transforms JSON-formatted string columns into Python
    dictionaries."""
    import pandas as pd

    df = pd.DataFrame(json_data)[json_columns]

    def try_json_loads(x: Any) -> Any:
        if isinstance(x, str):
            try:
                return deep_parse_json(x)
            except json.JSONDecodeError:
                return x
        return x

    for column in json_columns:
        df[column] = df[column].apply(try_json_loads)

    return df

def deep_parse_json(obj: Any, max_depth: int = 5) -> Any:
    """Recursively parse JSON strings at any depth within a nested data structure.
    
    This function traverses through strings, dictionaries, and lists, attempting to parse
    any JSON-formatted strings it encounters. It handles nested scenarios where JSON strings
    may contain other JSON strings (e.g., InfluxDB data with multiple levels of JSON encoding).
    
    Args:
        obj (Any): The object to parse - can be a string, dict, list, or any other type
        max_depth (int, optional): Maximum recursion depth to prevent infinite loops. Defaults to 5.
    
    Returns:
        Any: The parsed object with all JSON strings converted to their corresponding Python objects.
             Non-JSON strings and other data types are returned unchanged.
    """
    if max_depth <= 0:
        return obj

    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            try:
                parsed = json.loads(stripped)
                return deep_parse_json(parsed, max_depth - 1)
            except json.JSONDecodeError:
                return obj
        return obj
    
    elif isinstance(obj, dict):
        return {k: deep_parse_json(v, max_depth - 1) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [deep_parse_json(item, max_depth - 1) for item in obj]
    
    return obj
