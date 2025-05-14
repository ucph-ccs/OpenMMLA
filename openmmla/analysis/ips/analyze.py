import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.charts import Page
from pyecharts.commons.utils import JsCode
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

from openmmla.utils.querys import fetch_and_process_data, save_to_json_file, read_json_file, convert_json_to_dataframe
from openmmla.utils.visualization import weight_to_width, draw_networkx_edge_labels, format_list_for_pyecharts, \
    get_pyecharts_js_functions

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


def ips_session_analysis(project_dir, bucket_name, influx_client):
    """Retrieves data from InfluxDB for badge relations, translations, and rotations, and then visualizes them."""
    if not project_dir or not os.path.exists(project_dir):
        print("Warning: project_dir is not set or does not exist. Please set it to a valid directory, set it to current"
              "working directory.")
        project_dir = os.getcwd()

    logs_dir = os.path.join(project_dir, 'logs')
    log_dir = os.path.join(logs_dir, f'{bucket_name}')
    visualizations_dir = os.path.join(project_dir, 'visualizations')
    visualization_dir = os.path.join(visualizations_dir, bucket_name, "post-time")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Log
    relations_data = fetch_and_process_data(bucket_name, "badge relations", influx_client)
    translations_data = fetch_and_process_data(bucket_name, "badge translations", influx_client)
    rotations_data = fetch_and_process_data(bucket_name, "badge rotations", influx_client)
    translation_json_file = save_to_json_file(bucket_name, translations_data, "badge_translations", log_dir)
    relations_json_file = save_to_json_file(bucket_name, relations_data, "badge_relations", log_dir)
    save_to_json_file(bucket_name, rotations_data, "badge_rotations", log_dir)

    # Visualize
    plot_badge_locations_and_trajectories(translation_json_file, visualization_dir)
    plot_2d_heatmap(translation_json_file, visualization_dir)
    plot_physical_interaction_network(relations_json_file, visualization_dir)

    # Analyze across sessions
    ips_across_sessions_analysis(logs_dir, visualizations_dir)


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
        time_bucket = entry['time_bucket']
        for badge, coords in coords_dict.items():
            if badge in CANDIDATES:
                x, y, z = coords
                badge_translations[badge].append({
                    'x': x[0],
                    'y': y[0],
                    'z': z[0]
                })
                badge_time_stamps[badge].append(time_bucket)

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
        print(f"Saved 2D positions to {bucket_name}_2d_positions.png")

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
        print(f"Saved 2D interpolated trajectories to {bucket_name}_2d_interpolated_trajectories.png")

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
        print(f"Saved 3D positions to {bucket_name}_3d_positions.png")

    if plot_type in ['both', 'trajectory']:
        # 3D scatter plot with interpolated trajectory lines
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot dummy scatter points for legend
        for badge in BADGE_COLOR_MAP.keys():
            if badge in badge_translations:
                ax.scatter([], [], [], color=BADGE_COLOR_MAP[badge], label=f"Badge {badge} Trajectory")

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
        plt.savefig(os.path.join(visualization_dir, f"{bucket_name}_3d_interpolated_trajectories.png"), dpi=300)
        print(f"Saved 3D interpolated trajectories to {bucket_name}_3d_interpolated_trajectories.png")


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
    print(f"Saved 2D heatmap to {bucket_name}_hm.png")


def plot_physical_interaction_network(json_file_path, visualization_dir):
    """Plots a network graph representing physical interactions between badges."""
    physical_interactions = calculate_physical_interactions(json_file_path)
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
    draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=True, rad=arc_rad)
    draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=True, rad=arc_rad)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title(f'Normalized Physical Interaction Network ({session_name})')
    bucket_name = f"session_{os.path.basename(json_file_path).split('_')[1]}"
    plt.savefig(os.path.join(visualization_dir, f"{bucket_name}_npin.png"), dpi=300)


def calculate_physical_interactions(json_file_path):
    """Calculates the frequencies of physical interactions between badges based on DataFrame data."""
    json_data = read_json_file(json_file_path)
    df = convert_json_to_dataframe(json_data, ['graph', 'time_bucket'])
    normalize_factor = len(df)  # Normalized by number of entries
    interaction_counts = defaultdict(lambda: defaultdict(float))

    for _, row in df.iterrows():
        graph_dict = row['graph']
        for src, targets in graph_dict.items():
            if src in CANDIDATES:
                for tgt in targets:
                    if tgt in CANDIDATES and tgt != src:
                        interaction_counts[src][str(tgt)] += 1 / normalize_factor

    return interaction_counts


def ips_across_sessions_analysis(logs_dir, visualizations_dir):
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
    plot_ips_across_sessions_analysis(session_names, stm_lst, nstm_lst, srm_lst, nsrm_lst, visualizations_dir)
    plot_interactive_ips_across_sessions_analysis(session_names, stm_lst, nstm_lst, srm_lst, nsrm_lst,
                                                  visualizations_dir)


def plot_ips_across_sessions_analysis(session_names, stm_lst, nstm_lst, srm_lst, nsrm_lst, save_dir):
    """Visualize IPS across sessions analysis with matplotlib.

    Args:
        session_names (list): List of session names.
        stm_lst (list): List of Session Translate Movement values.
        nstm_lst (list): List of Normalized Session Translate Movement values.
        srm_lst (list): List of Session Rotate Movement values.
        nsrm_lst (list): List of Normalized Session Rotate Movement values.
        save_dir (str): Directory to save the plots.
    """
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
    fig.suptitle("IPS across sessions analysis")
    fig.tight_layout()

    plt.savefig(os.path.join(save_dir, 'ips_across_sessions_analysis.png'), dpi=300)
    print("IPS across sessions analysis saved to 'ips_across_sessions_analysis.png'")


def plot_interactive_ips_across_sessions_analysis(session_names, stm_lst, nstm_lst, srm_lst, nsrm_lst, save_dir):
    """Visualize IPS across sessions analysis with pyecharts."""
    # Generate visualization page
    page = Page(page_title="IPS across sessions analysis", layout=Page.SimplePageLayout)

    # Get JS functions for pyecharts
    js_funcs = get_pyecharts_js_functions()
    yaxis_min_js = js_funcs['yaxis_min_js']
    yaxis_max_js = js_funcs['yaxis_max_js']

    # Format data for pyecharts
    stm_lst = format_list_for_pyecharts(stm_lst)
    nstm_lst = format_list_for_pyecharts(nstm_lst)
    srm_lst = format_list_for_pyecharts(srm_lst)
    nsrm_lst = format_list_for_pyecharts(nsrm_lst)

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
    page.render(os.path.join(save_dir, f'ips_across_sessions_analysis.html'))
    print("IPS across sessions analysis saved to 'ips_across_sessions_analysis.html'")


def analyze_translations_log(json_path):
    """Calculates translation-based movement levels using Euclidean distances between translation vectors."""
    json_data = read_json_file(json_path)
    df = convert_json_to_dataframe(json_data, ['time_bucket', 'translations'])
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
    df = convert_json_to_dataframe(json_data, ['time_bucket', 'rotations'])
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
