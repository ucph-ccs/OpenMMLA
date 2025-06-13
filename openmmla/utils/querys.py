import json
import os
import time
from typing import Any

import pandas as pd
from openmmla.utils.client import InfluxDBClientWrapper

def generate_query(bucket_name: str, measurement: str) -> str:
    """Constructs an InfluxDB query string for a specific bucket, measurement, and fields."""
    bucket_start_time = bucket_name.split('_')[1]
    return f"""from(bucket: "{bucket_name}")
                |> range(start: {bucket_start_time})
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            """


def fetch_and_process_data(bucket_name: str, measurement: str, influx_client: InfluxDBClientWrapper) -> list[dict]:
    """Queries InfluxDB for specified data, converts it to JSON, and sorts it based on 'window_start_time'."""
    query = generate_query(bucket_name, measurement)
    tables = influx_client.query(query)
    json_str = tables.to_json(indent=5)
    data = deep_parse_json(json_str)
    data.sort(key=lambda x: x['window_start_time'])
    return data


def fetch_latest_entry(bucket_name: str, measurement: str, influx_client: InfluxDBClientWrapper) -> dict | None:
    """Retrieve the most recent entry of a measurement from InfluxDB and return as a dictionary."""

    start_time = int(time.time()) - 20  # 20 seconds ago up to now
    query = f"""from(bucket: "{bucket_name}")
                |> range(start: {start_time})
                |> last()
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                """

    tables = influx_client.query(query)
    json_str = tables.to_json(indent=5)
    data = deep_parse_json(json_str)
    return data[0] if data else None


def get_node_positions(bucket_name: str, influx_client: InfluxDBClientWrapper, timestamp: int, dimension: str = '2d') -> dict:
    """Retrieve segments' badge positions from InfluxDB and return as a dictionary of badge_id, position tuples."""
    start_time = int(timestamp) - 20
    query = f"""from(bucket: "{bucket_name}")
               |> range(start: {start_time})
               |> filter(fn: (r) => r._measurement == "badge_translation")
               |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
               |> filter(fn: (r) => r.window_start_time == {timestamp})
              """
    tables = influx_client.query(query)
    json_str = tables.to_json(indent=5)
    data = deep_parse_json(json_str)
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


def save_to_json_file(bucket_name: str, data: Any, suffix: str, log_dir: str) -> str:
    """Saves the given data to a JSON file in a specified directory, naming it based on the bucket name and a suffix."""
    json_path = os.path.join(log_dir, f"{bucket_name}_{suffix}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=5)
    print(f"{suffix} saved to {bucket_name}_{suffix}.json")
    return json_path


def convert_json_to_dataframe(json_data: Any, json_columns: list) -> pd.DataFrame:
    """Converts JSON data into a pandas DataFrame and transforms JSON-formatted string columns into Python
    dictionaries."""
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
    """
    Recursively parse JSON strings at any depth within a nested data structure.
    
    This function traverses through strings, dictionaries, and lists, attempting to parse
    any JSON-formatted strings it encounters. It handles nested scenarios where JSON strings
    may contain other JSON strings (e.g., InfluxDB data with multiple levels of JSON encoding).
    
    Args:
        obj (Any): The object to parse - can be a string, dict, list, or any other type
        max_depth (int, optional): Maximum recursion depth to prevent infinite loops. Defaults to 5.
    
    Returns:
        Any: The parsed object with all JSON strings converted to their corresponding Python objects.
             Non-JSON strings and other data types are returned unchanged.
    
    Examples:
        # simple JSON string parsing
        deep_parse_json('{"key": "value"}') -> {'key': 'value'}
        
        # nested JSON string parsing
        deep_parse_json('{"words": "[{\\"word\\": \\"hello\\"}]"}') 
        -> {'words': [{'word': 'hello'}]}
        
        # handles mixed data structures
        deep_parse_json({'text': 'hello', 'data': '{"nested": "json"}'})
        -> {'text': 'hello', 'data': {'nested': 'json'}}
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
                return obj  # not a JSON object/array string
        return obj  # skip strings like "11"
    
    elif isinstance(obj, dict):
        return {k: deep_parse_json(v, max_depth - 1) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [deep_parse_json(item, max_depth - 1) for item in obj]
    
    return obj

