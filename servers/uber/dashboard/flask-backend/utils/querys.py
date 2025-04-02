import json
import time


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
        segment_time = data[0]["segment_start_time"]
        return graph_dict, segment_time
    else:
        return data[0]


def get_node_positions(bucket_name, influx_client, segment_time, dimension='2d'):
    """Retrieve segments' badge positions from InfluxDB and return as a dictionary of badge_id, position tuples."""
    start_time = int(segment_time) - 20
    query = f"""from(bucket: "{bucket_name}")
               |> range(start: {start_time})
               |> filter(fn: (r) => r._measurement == "badge translations")
               |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
               |> filter(fn: (r) => r.segment_start_time == {segment_time})
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
