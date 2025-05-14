import json
import os
import time

import pandas as pd


def generate_query(bucket_name, measurement):
    """Constructs an InfluxDB query string for a specific bucket, measurement, and fields."""
    bucket_start_time = bucket_name.split('_')[1]
    return f"""from(bucket: "{bucket_name}")
                |> range(start: {bucket_start_time})
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            """


def fetch_and_process_data(bucket_name, measurement, influx_client):
    """Queries InfluxDB for specified data, converts it to JSON, and sorts it based on 'segment_start_time'."""
    query = generate_query(bucket_name, measurement)
    tables = influx_client.query(query)
    json_str = tables.to_json(indent=5)
    data = json.loads(json_str)
    data.sort(key=lambda x: x['time_bucket'])
    return json.dumps(data, ensure_ascii=False, indent=5)


def fetch_latest_entry(bucket_name, measurement, influx_client):
    """Retrieve the most recent entry of a measurement from InfluxDB and return as a dictionary.
    If the measurement is 'badge relations', it returns the graph dictionary and segment time."""

    start_time = int(time.time()) - 20  # 20 seconds ago up to now
    query = f"""from(bucket: "{bucket_name}")
                |> range(start: {start_time})
                |> last()
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                """

    tables = influx_client.query(query)
    data = json.loads(tables.to_json(indent=5))
    if not data:
        return None

    if measurement == "badge relations":
        graph_dict_str = data[0]["graph"]
        graph_dict = json.loads(graph_dict_str)
        time_bucket = data[0]["time_bucket"]
        return graph_dict, time_bucket
    else:
        return data[0]


def get_node_positions(bucket_name, influx_client, time_bucket, dimension='2d'):
    """Retrieve segments' badge positions from InfluxDB and return as a dictionary of badge_id, position tuples."""
    start_time = int(time_bucket) - 20
    query = f"""from(bucket: "{bucket_name}")
               |> range(start: {start_time})
               |> filter(fn: (r) => r._measurement == "badge translations")
               |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
               |> filter(fn: (r) => r.time_bucket == {time_bucket})
              """
    tables = influx_client.query(query)
    data = json.loads(tables.to_json(indent=5))
    translate_dict = json.loads(data[0]["translations"])

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


def read_json_file(file_path):
    """Reads and returns the contents of a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return None


def save_to_json_file(bucket_name, data, suffix, log_dir):
    """Saves the given data to a JSON file in a specified directory, naming it based on the bucket name and a suffix."""
    json_path = os.path.join(log_dir, f"{bucket_name}_{suffix}.json")
    with open(json_path, 'w') as f:
        f.write(data)
    print(f"{suffix} saved to {bucket_name}_{suffix}.json")
    return json_path


def convert_json_to_dataframe(json_data, json_columns):
    """Converts JSON data into a pandas DataFrame and transforms JSON-formatted string columns into Python
    dictionaries."""
    df = pd.DataFrame(json_data)[json_columns]

    def try_json_loads(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return x
        return x

    for column in json_columns:
        df[column] = df[column].apply(try_json_loads)

    return df
