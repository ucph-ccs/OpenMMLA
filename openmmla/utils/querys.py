import json
import os


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
    if measurement == 'speaker transcription':
        data.sort(key=lambda x: x['chunk_start_time'])
    else:
        data.sort(key=lambda x: x['segment_start_time'])
    return json.dumps(data, ensure_ascii=False, indent=5)


def save_to_json_file(bucket_name, data, suffix, log_dir):
    """Saves the given data to a JSON file in a specified directory, naming it based on the bucket name and a suffix."""
    json_path = os.path.join(log_dir, f"{bucket_name}_{suffix}.json")
    with open(json_path, 'w') as f:
        f.write(data)
    print(f"{suffix} saved to {bucket_name}_{suffix}.json")
    return json_path
