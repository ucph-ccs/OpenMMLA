import json
import os
from collections import Counter
from collections import defaultdict
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.charts import Page
from pyecharts.charts import Pie, Bar
from pyecharts.commons.utils import JsCode
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

matplotlib.use('Agg')

BADGE_COLOR_MAP = {
    '0': 'red',
    '1': 'blue',
    '2': 'green',
    '3': 'purple',
    '4': 'orange',
    '5': 'pink',
    '6': 'brown',
    '7': 'cyan',
    '8': 'magenta',
    '9': 'yellow',
    '10': 'black',
    '11': 'lime',
    '12': 'maroon',
    '13': 'olive',
    '14': 'navy',
    '15': 'teal',
    '16': 'gray',
    '17': 'sienna',
    '18': 'violet',
    '19': 'turquoise'
}
CANDIDATES = [str(i) for i in range(20)]


def session_analysis(project_dir, bucket_name, influx_client):
    """Retrieves data from InfluxDB for speaker recognition, transcription, badge relations, translations,
    and rotations, and then visualizes them."""
    logs_dir = os.path.join(project_dir, 'flask-backend/static/logs')
    log_dir = os.path.join(logs_dir, f'{bucket_name}')
    visualizations_dir = os.path.join(project_dir, 'flask-backend/static/visualizations')
    visualization_dir = os.path.join(visualizations_dir, f'{bucket_name}', 'post-time')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Log
    recognition_data = fetch_and_process_data(bucket_name, "speaker recognition", influx_client)
    transcription_data = fetch_and_process_data(bucket_name, "speaker transcription", influx_client)
    relations_data = fetch_and_process_data(bucket_name, "badge relations", influx_client)
    translations_data = fetch_and_process_data(bucket_name, "badge translations", influx_client)
    rotations_data = fetch_and_process_data(bucket_name, "badge rotations", influx_client)
    recognition_json_file_path = save_to_json_file(bucket_name, recognition_data, "speaker_recognition", log_dir)
    relations_json_file = save_to_json_file(bucket_name, relations_data, "badge_relations", log_dir)
    translation_json_file = save_to_json_file(bucket_name, translations_data, "badge_translations", log_dir)
    save_to_json_file(bucket_name, transcription_data, "speaker_transcription", log_dir)
    save_to_json_file(bucket_name, rotations_data, "badge_rotations", log_dir)

    # Visualize
    plot_speaker_diarization_interactive(recognition_json_file_path, visualization_dir)
    plot_speaking_interaction_network(recognition_json_file_path, visualization_dir)
    plot_badge_locations_and_trajectories(translation_json_file, visualization_dir)
    plot_2d_heatmap(translation_json_file, visualization_dir)
    plot_physical_interaction_network(relations_json_file, visualization_dir)

    # Analyze across sessions
    across_sessions_analysis_audio(logs_dir, visualizations_dir)
    across_sessions_analysis_video(logs_dir, visualizations_dir)


def format_time(hours, minutes, seconds):
    """Formats a time duration into a string in the format of hours, minutes, and seconds."""
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


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
    return json_path


def read_json_file(file_path):
    """Reads and returns the contents of a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return None


def convert_to_dataframe(json_data, json_columns):
    """Converts JSON data into a pandas DataFrame and transforms specific JSON-formatted string columns into Python
    dictionaries."""
    df = pd.DataFrame(json_data)[json_columns]
    for column in json_columns:
        if column != 'segment_start_time':
            df[column] = df[column].apply(json.loads)
    return df


def weight_to_width(wt, min_wt=0.0, max_wt=0.5, min_width=1, max_width=20):
    """Converts a numerical weight value to a corresponding width value for visual representation in a graph."""
    wt = max(min_wt, min(max_wt, wt))
    width = min_width + (wt - min_wt) / (max_wt - min_wt) * (max_width - min_width)
    return width


# Credit for this method goes to Stackoverflow user kcoskun
# https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
def my_draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color="k",
                                 font_family="sans-serif", font_weight="normal", alpha=None, bbox=None,
                                 horizontalalignment="center", verticalalignment="center", ax=None, rotate=True,
                                 clip_on=True, rad=0):
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        # if bbox is None:
        #     bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def plot_badge_locations_and_trajectories(json_file_path, visualization_dir, plot_type='both'):
    """Visualizes the 2D and 3D locations and trajectories of badges from JSON data."""
    bucket_name = f"session_{os.path.basename(json_file_path).split('_')[1]}"

    # Load the JSON file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Initialize a dictionary to store the badge locations
    badge_translations = defaultdict(list)
    badge_time_stamps = defaultdict(list)

    # Extract and store badge locations from the JSON data
    for entry in json_data:
        coords_dict = json.loads(entry['translations'])
        segment_time = entry['segment_start_time']
        for badge, coords in coords_dict.items():
            if badge in CANDIDATES:
                x, y, z = coords
                badge_translations[badge].append({
                    'x': x[0],
                    'y': y[0],
                    'z': z[0]
                })
                badge_time_stamps[badge].append(segment_time)

    if plot_type in ['both', 'position']:
        # 2D scatter plot of positions
        plt.figure(figsize=(12, 8))

        # Plot the positions in the order of the color map
        for badge in BADGE_COLOR_MAP.keys():
            if badge in badge_translations:
                z_coords = [entry['z'] for entry in badge_translations[badge]]
                x_coords = [entry['x'] for entry in badge_translations[badge]]
                plt.scatter(z_coords, x_coords, s=1, color=BADGE_COLOR_MAP[badge], label=f"Badge {badge}")

        plt.xlabel('Z Coordinate')
        plt.ylabel('X Coordinate')
        plt.title('2D Localisation of Badges (Z, X)')
        plt.legend()
        plt.savefig(os.path.join(visualization_dir, f"{bucket_name}_2d_positions.png"), dpi=300)

    if plot_type in ['both', 'trajectory']:
        # 2D scatter plot with interpolated trajectory lines
        plt.figure(figsize=(12, 8))

        # Plot dummy scatter points for legend
        for badge in BADGE_COLOR_MAP.keys():
            if badge in badge_translations:
                plt.scatter([], [], color=BADGE_COLOR_MAP[badge], label=f"Badge {badge} Trajectory")

        # Plot the actual trajectories
        for badge, coords_list in badge_translations.items():
            z_coords = np.array([entry['z'] for entry in coords_list])
            x_coords = np.array([entry['x'] for entry in coords_list])
            time_stamps = badge_time_stamps[badge]
            # Initialize start point for the new segment
            start_idx = 0

            for idx in range(1, len(time_stamps)):
                # Check if the time difference exceeds 1.5 seconds
                if time_stamps[idx] - time_stamps[idx - 1] > 1.5:
                    # Interpolate and plot the current segment
                    segment_z = z_coords[start_idx:idx]
                    segment_x = x_coords[start_idx:idx]
                    if len(set(time_stamps[start_idx:idx])) < 2:
                        plt.plot(z_coords[start_idx], x_coords[start_idx], linestyle=':', marker=None,
                                 color=BADGE_COLOR_MAP.get(badge, 'black'))
                        continue
                    # Determine interpolation type
                    interpolation_type = 'cubic' if len(segment_z) >= 4 else 'linear'
                    t = np.linspace(0, 1, len(segment_z))
                    interpolator = interp1d(t, (segment_z, segment_x), kind=interpolation_type)
                    new_t = np.linspace(0, 1, max(500, len(segment_z)))
                    new_z_coords, new_x_coords = interpolator(new_t)
                    plt.plot(new_z_coords, new_x_coords, linestyle=':', marker=None,
                             color=BADGE_COLOR_MAP.get(badge, 'black'))
                    # Update start point for the new segment
                    start_idx = idx

            # Interpolate and plot the last segment
            if len(set(time_stamps[start_idx:])) >= 2:
                segment_z = z_coords[start_idx:]
                segment_x = x_coords[start_idx:]
                interpolation_type = 'cubic' if len(segment_z) >= 4 else 'linear'
                t = np.linspace(0, 1, len(segment_z))
                interpolator = interp1d(t, (segment_z, segment_x), kind=interpolation_type)
                new_t = np.linspace(0, 1, max(500, len(segment_z)))
                new_z_coords, new_x_coords = interpolator(new_t)
                plt.plot(new_z_coords, new_x_coords, linestyle=':', marker=None,
                         color=BADGE_COLOR_MAP.get(badge, 'black'))

            # Mark start and end points
            plt.scatter(z_coords[0], x_coords[0], color=BADGE_COLOR_MAP.get(badge, 'black'), marker='.',
                        zorder=5)
            plt.scatter(z_coords[-1], x_coords[-1], color=BADGE_COLOR_MAP.get(badge, 'black'), marker='X', s=20,
                        zorder=5)

        plt.xlabel('Z Coordinate')
        plt.ylabel('X Coordinate')
        plt.title('2D Interpolated Trajectories of Badges (Z, X)')
        plt.legend()
        plt.savefig(os.path.join(visualization_dir, f"{bucket_name}_2d_interpolated_trajectories.png"), dpi=300)

    if plot_type in ['both', 'position']:
        # 3D scatter plot of positions
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the positions in the order of the color map
        for badge in BADGE_COLOR_MAP.keys():
            if badge in badge_translations:
                z_coords = [entry['z'] for entry in badge_translations[badge]]
                x_coords = [entry['x'] for entry in badge_translations[badge]]
                y_coords = [entry['y'] for entry in badge_translations[badge]]
                ax.scatter(z_coords, x_coords, y_coords, s=1, color=BADGE_COLOR_MAP[badge], label=f"Badge {badge}")

        ax.set_xlabel('Z Coordinate')
        ax.set_ylabel('X Coordinate')
        ax.set_zlabel('Y Coordinate')
        ax.set_title('3D Localisation of Badges')
        ax.legend()
        plt.savefig(os.path.join(visualization_dir, f"{bucket_name}_3d_positions.png"), dpi=300)

    if plot_type in ['both', 'trajectory']:
        # 3D scatter plot with interpolated trajectory lines
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot dummy scatter points for legend
        for badge in BADGE_COLOR_MAP.keys():
            if badge in badge_translations:
                ax.scatter([], [], [], color=BADGE_COLOR_MAP[badge], label=f'Badge {badge} Trajectory')

        # Plot the actual trajectories
        for badge, coords_list in badge_translations.items():
            z_coords = np.array([entry['z'] for entry in coords_list])
            x_coords = np.array([entry['x'] for entry in coords_list])
            y_coords = np.array([entry['y'] for entry in coords_list])
            time_stamps = badge_time_stamps[badge]
            # Initialize start point for the new segment
            start_idx = 0

            for idx in range(1, len(time_stamps)):
                # Check if the time difference exceeds 1.5 seconds
                if time_stamps[idx] - time_stamps[idx - 1] > 1.5:
                    # Interpolate and plot the current segment
                    if len(set(time_stamps[start_idx:idx])) < 2:
                        ax.plot(z_coords[start_idx], x_coords[start_idx], y_coords[start_idx], linestyle=':',
                                color=BADGE_COLOR_MAP.get(badge, 'black'))
                        continue
                    # Determine interpolation type
                    interpolation_type = 'cubic' if idx - start_idx >= 4 else 'linear'
                    segment_z = z_coords[start_idx:idx]
                    segment_x = x_coords[start_idx:idx]
                    segment_y = y_coords[start_idx:idx]
                    t = np.linspace(0, 1, len(segment_z))
                    interpolator = interp1d(t, (segment_z, segment_x, segment_y), kind=interpolation_type)
                    new_t = np.linspace(0, 1, max(500, len(segment_z)))
                    new_z_coords, new_x_coords, new_y_coords = interpolator(new_t)
                    ax.plot(new_z_coords, new_x_coords, new_y_coords, linestyle=':',
                            color=BADGE_COLOR_MAP.get(badge, 'black'))
                    # Update start point for the new segment
                    start_idx = idx

            # Interpolate and plot the last segment
            if len(set(time_stamps[start_idx:])) >= 2:
                segment_z = z_coords[start_idx:]
                segment_x = x_coords[start_idx:]
                segment_y = y_coords[start_idx:]
                interpolation_type = 'cubic' if len(segment_z) >= 4 else 'linear'
                t = np.linspace(0, 1, len(segment_z))
                interpolator = interp1d(t, (segment_z, segment_x, segment_y), kind=interpolation_type)
                new_t = np.linspace(0, 1, max(500, len(segment_z)))
                new_z_coords, new_x_coords, new_y_coords = interpolator(new_t)
                ax.plot(new_z_coords, new_x_coords, new_y_coords, linestyle=':',
                        color=BADGE_COLOR_MAP.get(badge, 'black'))

            # Mark start and end points
            ax.scatter(z_coords[0], x_coords[0], y_coords[0], color=BADGE_COLOR_MAP.get(badge, 'black'),
                       marker='.', s=50, zorder=5)
            ax.scatter(z_coords[-1], x_coords[-1], y_coords[-1], color=BADGE_COLOR_MAP.get(badge, 'black'),
                       marker='X', s=20, zorder=5)

        ax.set_xlabel('Z Coordinate')
        ax.set_ylabel('X Coordinate')
        ax.set_zlabel('Y Coordinate')
        ax.set_title('3D Interpolated Trajectories of Badges')
        ax.legend()
        plt.savefig(os.path.join(visualization_dir, f'{bucket_name}_3d_interpolated_trajectories.png'), dpi=300)


def plot_2d_heatmap(json_file_path, visualization_dir):
    """Creates a 2D heatmap visualization of badge positions based on their Z and X coordinates."""
    bucket_name = f"session_{os.path.basename(json_file_path).split('_')[1]}"

    # Load the JSON file to inspect its structure
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Initialize a dictionary to store the badge translations
    badge_translations = defaultdict(list)

    # Iterate over the JSON data to extract and store badge translations
    for entry in json_data:
        coords_dict = json.loads(entry['translations'])
        for badge, coords in coords_dict.items():
            if badge in CANDIDATES:
                x, y, z = coords  # Extracting x, y, and z coordinates
                badge_translations[badge].append({
                    'x': x[0],
                    'y': y[0],
                    'z': z[0]
                })

    # Initialize the Matplotlib figure for 2D heatmap using Z and X coordinates
    plt.figure(figsize=(12, 8))
    # Prepare data for heatmap
    all_z_coords = []
    all_x_coords = []
    for badge, coords_list in badge_translations.items():
        all_z_coords.extend([entry['z'] for entry in coords_list])
        all_x_coords.extend([entry['x'] for entry in coords_list])
    # Create the 2D histogram (heatmap)
    plt.hist2d(all_z_coords, all_x_coords, bins=(50, 50), cmap='inferno')
    # Add colorbar, labels, and title
    plt.colorbar(label='Frequency')
    plt.xlabel('Z Coordinate')
    plt.ylabel('X Coordinate')
    plt.title('2D Heatmap of Badge Positions (Z, X)')
    plt.savefig(os.path.join(visualization_dir, f"{bucket_name}_hm.png"), dpi=300)


def plot_physical_interaction_network(json_file_path, visualization_dir):
    """Plots a network graph representing physical interactions between badges."""
    json_data = read_json_file(json_file_path)
    df = convert_to_dataframe(json_data, ['graph', 'segment_start_time'])
    normalize_factor = len(df)  # Normalized by number of entries
    physical_interactions = calculate_physical_interactions(df, normalize_factor)
    session_name = os.path.basename(json_file_path).split('_')[1]

    # Create a directed graph
    G = nx.DiGraph()
    for src, targets in physical_interactions.items():
        for tgt, freq in targets.items():
            G.add_edge(src, tgt, weight=freq)

    # fixed_positions = {'0': (0, 0), '1': (0, 1), '2': (1, 1), '3': (1, 0)}  # specify positions here
    fixed_positions = None
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42) if fixed_positions is None else fixed_positions

    # Calculate accumulative weights for each node
    accumulative_weights = {node: sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes}

    # Node size parameters
    max_node_size = 1000  # Maximum node size
    min_node_size = 400  # Minimum node size

    # Normalize node sizes using a logarithmic scale
    node_sizes = {}
    for node, weight in accumulative_weights.items():
        node_sizes[node] = min_node_size + (weight - 0) / (2 - 0) * (
                max_node_size - min_node_size) if weight > 0 else min_node_size

    # Draw nodes with size based on accumulative weight
    nx.draw_networkx_nodes(G, pos, node_size=[node_sizes[node] for node in G.nodes], alpha=0.2)

    # Draw edges
    # Identify curved edges
    curved_edges = [edge for edge in G.edges() if G.has_edge(edge[1], edge[0])]
    straight_edges = list(set(G.edges()) - set(curved_edges))

    # Draw straight edges
    for (u, v) in straight_edges:
        wt = G[u][v]['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=weight_to_width(wt, max_wt=1, max_width=10),
                               connectionstyle=f'arc3, rad = 0.2',
                               edge_color='moccasin', style='solid')

    # Draw curved edges with arc style
    for (u, v) in curved_edges:
        wt = G[u][v]['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=weight_to_width(wt, max_wt=1, max_width=10),
                               connectionstyle=f'arc3, rad = 0.2',
                               edge_color='orange', style='dashed')

    # Draw edge labels using the custom function
    ax = plt.gca()
    arc_rad = 0.25
    edge_weights = nx.get_edge_attributes(G, 'weight')
    curved_edge_labels = {edge: round(edge_weights[edge], 4) for edge in curved_edges}
    straight_edge_labels = {edge: round(edge_weights[edge], 4) for edge in straight_edges}
    my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=True, rad=arc_rad)
    my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=True, rad=arc_rad)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title(f'Normalized Physical Interaction Network ({session_name})')
    bucket_name = f"session_{os.path.basename(json_file_path).split('_')[1]}"
    plt.savefig(os.path.join(visualization_dir, f"{bucket_name}_npin.png"), dpi=300)


def calculate_physical_interactions(df, normalize_factor):
    """Calculates the frequencies of physical interactions between badges based on DataFrame data."""
    interaction_counts = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        graph_dict = row['graph']
        for src, targets in graph_dict.items():
            if src in CANDIDATES:
                for tgt in targets:
                    if tgt in CANDIDATES and tgt != src:
                        interaction_counts[src][str(tgt)] += 1 / normalize_factor

    return interaction_counts


def across_sessions_analysis_video(logs_dir, visualizations_dir):
    """Analyzes and visualizes various movement metrics across multiple sessions using saved log data."""
    session_data = []

    for bucket_dir_name in sorted(os.listdir(logs_dir)):
        bucket_dir_path = os.path.join(logs_dir, bucket_dir_name)
        if os.path.isdir(bucket_dir_path):
            session_name = bucket_dir_name.split('_')[1]
            # Initialize metrics with null values
            stm, nstm, srm, nsrm = None, None, None, None
            for filename in os.listdir(bucket_dir_path):
                if filename.endswith("_translations.json"):
                    translation_file = os.path.join(bucket_dir_path, filename)
                    _, stm, nstm = analyze_translations_log(translation_file)
                if filename.endswith("_rotations.json"):
                    rotation_file = os.path.join(bucket_dir_path, filename)
                    _, srm, nsrm = analyze_rotations_log(rotation_file)
            session_data.append((session_name, stm, nstm, srm, nsrm))

    # Unpack the session data for plotting
    session_names, stm_lst, nstm_lst, srm_lst, nsrm_lst = zip(*session_data)

    # Call plotting functions
    plot_across_sessions_analysis_video(session_names, stm_lst, nstm_lst, srm_lst, nsrm_lst, visualizations_dir)
    plot_across_sessions_analysis_video_interactive(session_names, stm_lst, nstm_lst, srm_lst, nsrm_lst,
                                                    visualizations_dir)


def plot_across_sessions_analysis_video(session_names, stm_lst, nstm_lst, srm_lst, nsrm_lst, save_dir):
    """Creates and saves matplotlib plots of various video analysis metrics across different sessions."""
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plotting the first graph
    ax1.set_title("STM and NSTM")
    ax1.set_xlabel("Session")
    ax1.set_ylabel("Session Translate Movement (m)", color='tab:blue')
    ax1.plot(session_names, stm_lst, 'o-', color='tab:blue', label='STM')
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Normalized Session Translate Movement (m/s)", color='tab:red')
    ax1_twin.plot(session_names, nstm_lst, 'x-', color='tab:red', label='NSTM')
    ax1_twin.tick_params(axis="y", labelcolor="tab:red")

    # Plotting the second graph
    ax2.set_title("SRM and NSRM")
    ax2.set_xlabel("Session")
    ax2.set_ylabel("Session Rotate Movement (°)", color='tab:green')
    ax2.plot(session_names, srm_lst, 's-', color='tab:green', label='SRM')
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Normalized Session Rotate Movement (°/s)", color='tab:purple')
    ax2_twin.plot(session_names, nsrm_lst, 'd-', color='tab:purple', label='NSRM')
    ax2_twin.tick_params(axis="y", labelcolor="tab:purple")

    # Show legends
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    # Show plot
    fig.suptitle("Video Analysis Metrics Across Sessions")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'video_analysis_across_sessions.png'), dpi=300)


def plot_across_sessions_analysis_video_interactive(session_names, stm_lst, nstm_lst, srm_lst, nsrm_lst, save_dir):
    """Visualize audio analysis metrics across sessions with pyecharts."""
    # Generate visualization page
    page = Page(page_title="video metrics analysis across sessions", layout=Page.SimplePageLayout)

    def format_list(data):
        return [f"{x:.4f}" if x is not None else '0' for x in data]

    stm_lst = format_list(stm_lst)
    nstm_lst = format_list(nstm_lst)
    srm_lst = format_list(srm_lst)
    nsrm_lst = format_list(nsrm_lst)
    yaxis_min_js = JsCode("function(value){return Math.round(value.min * 0.8 * 1000) / 1000;}")
    yaxis_max_js = JsCode("function(value){return Math.round(value.max * 1.2 * 1000) / 1000;}")

    # Create Line chart for STM and NSTM
    line1 = (
        Line(init_opts=opts.InitOpts(width="1500px", height="400px"))
        .add_xaxis(session_names)
        .add_yaxis("STM", stm_lst, is_smooth=False, yaxis_index=0)
        .add_yaxis("NSTM", nstm_lst, is_smooth=False, yaxis_index=1)
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="Normalized Session Translate Movement(m/s)",
                min_=yaxis_min_js,
                max_=yaxis_max_js,
                type_="value",
                position="right",
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="STM and NSTM", pos_top="0%", pos_left="center"),
            legend_opts=opts.LegendOpts(
                pos_top="7%",
                item_width=20,
                item_height=10,
                textstyle_opts=opts.TextStyleOpts(font_size=10)
            ),
            yaxis_opts=opts.AxisOpts(
                name="Session Translate Movement(m)",
                min_=yaxis_min_js,
                max_=yaxis_max_js,
            ),
            xaxis_opts=opts.AxisOpts(name="Session", name_location="middle", name_gap=30)
        )
    )

    # Create Line chart for SRM and NSRM
    line2 = (
        Line(init_opts=opts.InitOpts(width="1500px", height="400px"))
        .add_xaxis(session_names)
        .add_yaxis("SRM", srm_lst, is_smooth=False, yaxis_index=0)
        .add_yaxis("NSRM", nsrm_lst, is_smooth=False, yaxis_index=1)
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="Normalized Session Rotate Movement(°/s)",
                min_=yaxis_min_js,
                max_=yaxis_max_js,
                type_="value",
                position="right",
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="SRM and NSRM", pos_top="0%", pos_left="center"),
            legend_opts=opts.LegendOpts(
                pos_top="7%",
                item_width=20,
                item_height=10,
                textstyle_opts=opts.TextStyleOpts(font_size=10)
            ),
            yaxis_opts=opts.AxisOpts(
                name="Session Rotate Movement(°)",
                min_=yaxis_min_js,
                max_=yaxis_max_js,
            ),
            xaxis_opts=opts.AxisOpts(name="Session", name_location="middle", name_gap=30)
        )
    )

    page.add(line1)
    page.add(line2)
    page.render(os.path.join(save_dir, f'video_analysis_across_sessions.html'))


def analyze_translations_log(json_path):
    """Calculates translation-based movement levels using Euclidean distances between translation vectors."""
    json_data = read_json_file(json_path)
    df = convert_to_dataframe(json_data, ['segment_start_time', 'translations'])
    session_duration = len(df) * 1
    badges_movement = {}

    for index, row in df.iterrows():
        translations = row['translations']
        for badge_id, coordinates in translations.items():
            coordinates = np.array(coordinates).flatten()
            if badge_id not in badges_movement:
                badges_movement[badge_id] = {'previous_coordinates': None, 'total_movement': 0}
            if badges_movement[badge_id]['previous_coordinates'] is not None:
                distance = euclidean(badges_movement[badge_id]['previous_coordinates'], coordinates)
                badges_movement[badge_id]['total_movement'] += distance
            badges_movement[badge_id]['previous_coordinates'] = coordinates

    individual_badge_movement = {badge_id: data['total_movement'] for badge_id, data in badges_movement.items()}
    total_session_movement = sum(individual_badge_movement.values())
    normalized_session_movement = total_session_movement / session_duration
    return individual_badge_movement, total_session_movement, normalized_session_movement


def analyze_rotations_log(json_path):
    """Computes rotation-based movement levels using degree changes between rotation matrices."""
    json_data = read_json_file(json_path)
    df = convert_to_dataframe(json_data, ['segment_start_time', 'rotations'])
    session_duration = len(df) * 1  # Assuming each row represents 1 unit of time
    badges_movement = {}

    for index, row in df.iterrows():
        rotations = row['rotations']
        for badge_id, rotation_matrix in rotations.items():
            if badge_id not in badges_movement:
                badges_movement[badge_id] = {'previous_rotation': None, 'total_movement': 0}
            if badges_movement[badge_id]['previous_rotation'] is not None:
                # Calculate the rotation matrix that represents the rotation from previous to current
                R_diff = np.dot(np.linalg.inv(np.array(badges_movement[badge_id]['previous_rotation'])),
                                np.array(rotation_matrix))
                # Calculate the degree of rotation
                degree_of_rotation = calculate_degree_of_rotation(R_diff)
                badges_movement[badge_id]['total_movement'] += degree_of_rotation
            badges_movement[badge_id]['previous_rotation'] = np.array(rotation_matrix)

    individual_badge_movement = {badge_id: data['total_movement'] for badge_id, data in badges_movement.items()}
    total_session_movement = sum(individual_badge_movement.values())
    normalized_session_movement = total_session_movement / session_duration
    return individual_badge_movement, total_session_movement, normalized_session_movement


def calculate_degree_of_rotation(R):
    """Calculate the degree of rotation from a rotation matrix."""
    trace_value = np.trace(R)
    angle_rad = np.arccos((trace_value - 1) / 2)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def plot_speaker_diarization_interactive(json_file_path, visualization_dir):
    """Visualizes speaker diarization data using a stacked bar chart and pie charts with pyecharts."""
    colors = ['#c23531', '#2f4554', '#61a0a8', '#d48265', '#749f83', '#ca8622', '#bda29a', '#6e7074', '#546570',
              '#c4ccd3', '#f05b72', '#ef5b9c', '#f47920', '#905a3d', '#fab27b', '#2a5caa', '#444693', '#726930',
              '#b2d235', '#6d8346', '#ac6767', '#1d953f', '#6950a1', '#918597']
    speaker_color_map = {}
    bars = {'all': []}
    speaker_time_dict = {}  # To hold total duration for each speaker
    speaker_counts_dict = {}  # To hold total number of segments for each speaker
    session_name = os.path.basename(json_file_path).split('_')[1]

    with open(json_file_path, 'r') as f:
        json_dicts = json.load(f)
    l = len(json_dicts)
    segment_start_time = datetime.fromtimestamp(int(json_dicts[0]['segment_start_time']))

    x_bar = []
    count = 0
    for i in range(l):
        speaker_list = json.loads(json_dicts[i]['speakers'])
        probability_list = json.loads(json_dicts[i]['similarities'])
        duration_list = json.loads(json_dicts[i]['durations'])
        time = datetime.fromtimestamp(json_dicts[i]['segment_start_time'])

        # Increase the timeline axis values
        count += 1
        period = time - segment_start_time
        # Convert timedelta to hours, minutes, and seconds.
        hours, remainder = divmod(period.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = format_time(hours, minutes, seconds)
        x_bar.append(time_str)
        # Add one more time segment bar
        for key in bars.keys():
            bars[key].append(0.0)

        for j in range(len(speaker_list)):
            speaker = speaker_list[j]
            probability = probability_list[j]
            duration = duration_list[j]

            if speaker not in speaker_time_dict:
                speaker_time_dict[speaker] = 0
                speaker_counts_dict[speaker] = 0
                bars[speaker] = [0.0 for _ in range(count)]

            bars[speaker][-1] += probability
            bars['all'][-1] += probability
            speaker_time_dict[speaker] += duration
            speaker_counts_dict[speaker] += 1

    labels = list(speaker_time_dict.keys())
    seconds_distribution = [round(speaker_time_dict[label], 1) for label in labels]
    counts_distribution = [round(speaker_counts_dict[label], 1) for label in labels]

    # Generate visualization page
    page = Page(page_title="mbox speaker visualization", layout=Page.SimplePageLayout)
    stacked_bar = Bar(init_opts=opts.InitOpts(width="3000px", height="250px"))
    pie1 = Pie(init_opts=opts.InitOpts(width="500px", height="500px"))
    pie2 = Pie(init_opts=opts.InitOpts(width="500px", height="500px"))

    # Now, generate stacked bar graph for 'all'
    color_index = -1
    stacked_bar.add_xaxis(x_bar)
    sorted_bar_keys = sorted(bars.keys())
    for key in sorted_bar_keys:
        if key == 'all':
            continue
        color_index += 1
        color = colors[color_index]
        speaker_color_map[key] = color
        bar_data = [
            {"value": value, "percent": 0 if bars['all'][i] == 0 else value / bars['all'][i]}
            for i, value in enumerate(bars[key])
        ]
        name = key

        stacked_bar.add_yaxis(
            series_name=name,
            y_axis=bar_data,
            stack="stack1",
            category_gap="0%",
            label_opts=opts.LabelOpts(
                is_show=False,
                position="top",
                formatter=JsCode(
                    """
                    function(x) {
                        var val = x.data.value;
                        if (val > 0) {
                            return val.toFixed(1);
                        } else {
                            return '';
                        }
                    }
                    """
                )
            ),
            itemstyle_opts=opts.ItemStyleOpts(color=color),
        )

    stacked_bar.set_global_opts(
        title_opts=opts.TitleOpts(title=f"Speaker Alignment ({session_name})", pos_top='0%', pos_left='center'),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        xaxis_opts=opts.AxisOpts(name="time", boundary_gap=False),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter='{value}'), interval=1),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        legend_opts=opts.LegendOpts(pos_top="13%"),
    )

    # add data
    pie1.add("",
             [list(z) for z in zip(labels, seconds_distribution)],
             label_opts=opts.LabelOpts(
                 position="inside",
                 formatter="{b|{b}: }{c}  {per|{d}%}",
                 border_width=0,
                 border_radius=0,
                 rich={
                     "b": {"fontSize": 15, "lineHeight": 20},
                     "per": {
                         "color": "#eee",
                         "padding": [1, 1],
                         "borderRadius": 0,
                     },
                 },
             ),
             )
    pie1_color = [speaker_color_map[label] for label in labels]
    pie1.set_colors(pie1_color)
    pie1.set_global_opts(
        title_opts=opts.TitleOpts(
            title="Time Distribution (seconds)",
            pos_top="0%",
            pos_left='center',
            title_textstyle_opts=opts.TextStyleOpts(font_size=15, ),
        ),
        legend_opts=opts.LegendOpts(
            pos_top=25,
            pos_left='center',
            textstyle_opts=opts.TextStyleOpts(
                font_size=10,  # Change the font size of the legend
            ),
        )
    )

    # add data
    pie2.add("",
             [list(z) for z in zip(labels, counts_distribution)],
             label_opts=opts.LabelOpts(
                 position="inside",
                 formatter="{b|{b}: }{c}  {per|{d}%}",
                 border_width=0,
                 border_radius=0,
                 rich={
                     "b": {"fontSize": 15, "lineHeight": 20},
                     "per": {
                         "color": "#eee",
                         "padding": [1, 1],
                         "borderRadius": 0,
                     },
                 },
             ),
             )
    pie2_color = [speaker_color_map[label] for label in labels]
    pie2.set_colors(pie2_color)
    pie2.set_global_opts(
        title_opts=opts.TitleOpts(
            title="Segment Distribution",
            pos_top="0%",
            pos_left='center',
            title_textstyle_opts=opts.TextStyleOpts(font_size=15, ),
        ),
        legend_opts=opts.LegendOpts(
            pos_top=25,
            pos_left='center',
            textstyle_opts=opts.TextStyleOpts(
                font_size=10,  # Change the font size of the legend
            ),
        )
    )

    page.add(stacked_bar)
    page.add(pie1)
    page.add(pie2)
    bucket_name = f"session_{os.path.basename(json_file_path).split('_')[1]}"
    page.render(os.path.join(visualization_dir, f'{bucket_name}_speaker_diarization.html'))


def plot_speaking_interaction_network(json_file_path, visualization_dir):
    """Creates and visualizes a network graph of speaking interactions using NetworkX and matplotlib."""
    speaker_interactions = calculate_speaker_interactions(json_file_path)
    session_name = os.path.basename(json_file_path).split('_')[1]

    # Create a directed graph
    G = nx.DiGraph()
    for src, targets in speaker_interactions.items():
        for tgt, freq in targets.items():
            G.add_edge(src, tgt, weight=freq)

    fixed_positions = None
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42) if fixed_positions is None else fixed_positions

    # Calculate accumulative weights for each node
    accumulative_weights = {node: sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes}

    # Node size parameters
    max_node_size = 2000
    min_node_size = 500

    # Normalize node sizes
    node_sizes = {}
    for node, weight in accumulative_weights.items():
        node_sizes[node] = min_node_size + (weight - 0) / (0.5 - 0) * (
                max_node_size - min_node_size) if weight > 0 else min_node_size

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=[node_sizes[node] for node in G.nodes], alpha=0.2)

    # Draw edges
    # Identify curved edges
    curved_edges = [edge for edge in G.edges() if G.has_edge(edge[1], edge[0])]
    straight_edges = list(set(G.edges()) - set(curved_edges))

    # Draw straight edges
    for (u, v) in straight_edges:
        wt = G[u][v]['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=weight_to_width(wt),
                               connectionstyle=f'arc3, rad = 0.2',
                               edge_color='moccasin', style='solid')

    # Draw curved edges with arc style
    for (u, v) in curved_edges:
        wt = G[u][v]['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=weight_to_width(wt),
                               connectionstyle=f'arc3, rad = 0.2',
                               edge_color='orange', style='dashed')

    # Draw edge labels using the custom function
    ax = plt.gca()
    arc_rad = 0.25
    edge_weights = nx.get_edge_attributes(G, 'weight')
    curved_edge_labels = {edge: round(edge_weights[edge], 4) for edge in curved_edges}
    straight_edge_labels = {edge: round(edge_weights[edge], 4) for edge in straight_edges}
    my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=True, rad=arc_rad)
    my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=True, rad=arc_rad)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title(f'Normalized Speaking Interaction Network ({session_name})')
    bucket_name = f"session_{os.path.basename(json_file_path).split('_')[1]}"
    plt.savefig(os.path.join(visualization_dir, f"{bucket_name}_nsin.png"), dpi=300)


def calculate_speaker_interactions(json_file_path):
    """Calculates the frequency of interactions between speakers in a conversation."""
    # Load JSON data from the provided file path
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    normalize_factor = len(data)  # Normalized by number of segments

    # Initialize a dictionary to hold the interaction counts
    interaction_counts = {}

    # Function to update the interaction counts
    def update_interactions(cur_speakers, nex_speakers):
        for next_speaker in nex_speakers:
            if next_speaker not in interaction_counts:
                interaction_counts[next_speaker] = {}
            for current in cur_speakers:
                if current != next_speaker:
                    interaction_counts[next_speaker][current] = (interaction_counts[next_speaker].get(current, 0) + 1 /
                                                                 normalize_factor)

    # Iterate through the JSON data
    for i in range(len(data) - 1):
        current_segment = data[i]
        next_segment = data[i + 1]

        # Parse speakers from the current and next segments
        current_speakers = json.loads(current_segment['speakers'])
        next_speakers = json.loads(next_segment['speakers'])

        # Filter out 'silent' and 'unknown'
        current_speakers = [s for s in current_speakers if s not in ['silent', 'unknown']]
        next_speakers = [s for s in next_speakers if s not in ['silent', 'unknown']]

        # Update interaction counts
        update_interactions(current_speakers, next_speakers)

    return interaction_counts


def across_sessions_analysis_audio(logs_dir, visualizations_dir):
    """Analyzes and visualizes various conversation metrics across multiple sessions using saved log data."""
    session_data = []
    for bucket_dir_name in sorted(os.listdir(logs_dir)):
        bucket_dir_path = os.path.join(logs_dir, bucket_dir_name)
        session_name = bucket_dir_name.split('_')[1]
        if os.path.isdir(bucket_dir_path):
            apd, peq, nttc, sr = None, None, None, None
            for filename in os.listdir(bucket_dir_path):
                if filename.endswith("_recognition.json"):
                    json_path = os.path.join(bucket_dir_path, filename)
                    apd, peq, nttc, sr = analyze_recognition_log(json_path)
            session_data.append((session_name, apd, peq, nttc, sr))

    # Sorting the session data by session names
    session_data.sort(key=lambda x: x[0])

    # Unpacking the sorted data into separate lists
    session_names, apd_lst, peq_lst, nttc_lst, sr_lst = zip(*session_data)

    plot_across_sessions_analysis_audio(session_names, apd_lst, nttc_lst, sr_lst, peq_lst, visualizations_dir)
    plot_across_sessions_analysis_audio_interactive(session_names, apd_lst, nttc_lst, sr_lst, peq_lst,
                                                    visualizations_dir)


def analyze_recognition_log(json_file_path, unknown=False):
    """An updated version of analyze_conversation_metrics, offering refined analysis of conversation metrics."""
    # Read JSON file
    with open(json_file_path, 'r') as f:
        logs = json.load(f)

    if not unknown:
        # Filter out logs with 'unknown' speaker
        logs = [log for log in logs if 'unknown' not in log['speakers']]

    # Initialize variables
    turn_taking_count = 0
    current_silence_duration = 0
    silence_durations = []
    speaker_counter = Counter()
    speaker_durations = Counter()
    normalize_factor = len(logs)  # Normalized by number of segments

    # Loop through each log entry to calculate metrics
    for j in range(len(logs) - 1):
        log = logs[j]
        next_log = logs[j + 1]
        speakers = json.loads(log['speakers'])
        durations = json.loads(log['durations'])
        next_speakers = json.loads(next_log['speakers'])

        for i, speaker in enumerate(speakers):
            duration = durations[i]
            speaker_counter[speaker] += 1
            speaker_durations[speaker] += duration

            if speaker == "silent":
                current_silence_duration += duration
                # Reset and store silence duration if the next speaker is not silent
                if speaker not in next_speakers:
                    silence_durations.append(current_silence_duration)
                    current_silence_duration = 0
            # Turn-taking count, speaker A -> speaker B counts once, speaker A -> silent doesn't count
            if speaker not in next_speakers and 'silent' not in next_speakers:
                turn_taking_count += 1

        if j == len(logs) - 2:
            next_durations = json.loads(next_log['durations'])
            for i, speaker in enumerate(next_speakers):
                duration = next_durations[i]
                speaker_counter[speaker] += 1
                speaker_durations[speaker] += duration

                if speaker == 'silent':
                    current_silence_duration += duration
                    silence_durations.append(current_silence_duration)
                    current_silence_duration = 0

    # Calculate Average Pause Duration (APD)
    apd = sum(silence_durations) / len(silence_durations) if silence_durations else 0

    # Calculate Participation Equality using Gini Coefficient (PEQ)
    filtered_speaker_counter = {k: v for k, v in speaker_counter.items() if k not in ["silent", "unknown"]}
    n = len(filtered_speaker_counter)
    total_utterances = sum(filtered_speaker_counter.values())
    mean_utterances = total_utterances / n if n else 0

    if n == 0 or total_utterances == 0:  # Handle case where no known speakers are present
        peq = None
    else:
        gini_coefficient = sum(
            sum(abs(x - y) for y in filtered_speaker_counter.values())
            for x in filtered_speaker_counter.values()) / (2 * n * n * mean_utterances)
        peq = 1 - gini_coefficient

    # Normalized Turn Taking Count (NTTC)
    nttc = turn_taking_count / normalize_factor

    # Silence Ratio (SR)
    sr = speaker_durations["silent"] / sum(speaker_durations.values()) if "silent" in speaker_durations else 0

    # Return calculated metrics
    return apd, peq, nttc, sr


def plot_across_sessions_analysis_audio(session_names, apd_lst, nttc_lst, sr_lst, peq_lst, save_dir):
    """Creates and saves matplotlib plots of various audio analysis metrics across different sessions."""
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plotting the first graph
    ax1.set_title("APD and NTTC")
    ax1.set_xlabel("Session")
    ax1.set_ylabel("Average Pause Duration (seconds/silence pause)", color='tab:blue')
    ax1.plot(session_names, apd_lst, 'o-', color='tab:blue', label='APD')
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Normalized Turn Taking Count (turns/segment)", color='tab:red')
    ax1_twin.plot(session_names, nttc_lst, 'x-', color='tab:red', label='NTTC')
    ax1_twin.tick_params(axis="y", labelcolor="tab:red")

    # Plotting the second graph
    ax2.set_title("SR and PEQ")
    ax2.set_xlabel("Session")
    ax2.set_ylabel("Silence Ratio (%)", color='tab:green')
    ax2.plot(session_names, sr_lst, 's-', color='tab:green', label='SR')
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Participation Equality (%)", color='tab:purple')
    ax2_twin.plot(session_names, peq_lst, 'd-', color='tab:purple', label='PEQ')
    ax2_twin.tick_params(axis="y", labelcolor="tab:purple")

    # Show legends
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    # Show plot
    fig.suptitle("Audio Analysis Metrics Across Sessions")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'audio_analysis_across_sessions.png'), dpi=300)


def plot_across_sessions_analysis_audio_interactive(session_names, apd_lst, nttc_lst, sr_lst, peq_lst, save_dir):
    """Visualize audio analysis metrics across sessions with pyecharts."""
    # Generate visualization page
    page = Page(page_title="audio metrics analysis across sessions", layout=Page.SimplePageLayout)

    def format_list(data):
        return [f"{x:.4f}" if x is not None else '0' for x in data]

    apd_lst = format_list(apd_lst)
    nttc_lst = format_list(nttc_lst)
    sr_lst = format_list(sr_lst)
    peq_lst = format_list(peq_lst)
    yaxis_min_js = JsCode("function(value){return Math.round(value.min * 0.8 * 1000) / 1000;}")
    yaxis_max_js = JsCode("function(value){return Math.round(value.max * 1.2 * 1000) / 1000;}")

    # Create Line chart for APD and NTTC
    line1 = (
        Line(init_opts=opts.InitOpts(width="1500px", height="400px"))
        .add_xaxis(session_names)
        .add_yaxis("APD", apd_lst, is_smooth=False, yaxis_index=0)
        .add_yaxis("NTTC", nttc_lst, is_smooth=False, yaxis_index=1)
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="Normalized Turn Taking Count (turns/segment)",
                min_=yaxis_min_js,
                max_=yaxis_max_js,
                type_="value",
                position="right",
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="APD and NTTC", pos_top="0%", pos_left="center"),
            legend_opts=opts.LegendOpts(
                pos_top="7%",
                item_width=20,
                item_height=10,
                textstyle_opts=opts.TextStyleOpts(font_size=10)
            ),
            yaxis_opts=opts.AxisOpts(
                name="Average Pause Duration (seconds/silent pause)",
                min_=yaxis_min_js,
                max_=yaxis_max_js,
            ),
            xaxis_opts=opts.AxisOpts(name="Session", name_location="middle", name_gap=30)
        )
    )

    # Create Line chart for PEQ and SR
    line2 = (
        Line(init_opts=opts.InitOpts(width="1500px", height="400px"))
        .add_xaxis(session_names)
        .add_yaxis("SR", sr_lst, is_smooth=False, yaxis_index=0)
        .add_yaxis("PEQ", peq_lst, is_smooth=False, yaxis_index=1)
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="Participation Equality (%)",
                min_=yaxis_min_js,
                max_=yaxis_max_js,
                type_="value",
                position="right",
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="SR and PEQ", pos_top="0%", pos_left="center"),
            legend_opts=opts.LegendOpts(
                pos_top="7%",
                item_width=20,
                item_height=10,
                textstyle_opts=opts.TextStyleOpts(font_size=10)
            ),
            yaxis_opts=opts.AxisOpts(
                name="Silence Ratio (%)",
                min_=yaxis_min_js,
                max_=yaxis_max_js,
            ),
            xaxis_opts=opts.AxisOpts(name="Session", name_location="middle", name_gap=30)
        )
    )

    page.add(line1)
    page.add(line2)
    page.render(os.path.join(save_dir, f'audio_analysis_across_sessions.html'))
