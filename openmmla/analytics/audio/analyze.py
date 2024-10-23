import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.charts import Page
from pyecharts.charts import Pie, Bar
from pyecharts.commons.utils import JsCode

from openmmla.utils.querys import fetch_and_process_data, save_to_json_file, read_json_file, convert_json_to_dataframe
from .text_processing import convert_transcription_json_to_txt


def session_analysis_audio(project_dir, bucket_name, influx_client):
    """Retrieves data from InfluxDB for speaker recognition, transcription, and then visualizes them."""
    logs_dir = os.path.join(project_dir, 'logs')
    log_dir = os.path.join(logs_dir, f'{bucket_name}')
    visualizations_dir = os.path.join(project_dir, 'visualizations')
    visualization_dir = os.path.join(visualizations_dir, f'{bucket_name}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Log
    recognition_data = fetch_and_process_data(bucket_name, "speaker recognition", influx_client)
    transcription_data = fetch_and_process_data(bucket_name, "speaker transcription", influx_client)
    recognition_json_file_path = save_to_json_file(bucket_name, recognition_data, "speaker_recognition", log_dir)
    transcription_json_file_path = save_to_json_file(bucket_name, transcription_data, "speaker_transcription", log_dir)
    convert_transcription_json_to_txt(transcription_json_file_path)

    # Visualize
    plot_speaker_diarization_interactive(recognition_json_file_path, visualization_dir)
    plot_speaking_interaction_network(recognition_json_file_path, visualization_dir)

    # Analyze across sessions
    across_sessions_analysis_audio(logs_dir, visualizations_dir)


def format_time(hours, minutes, seconds):
    """Formats a time duration into a string in the format of hours, minutes, and seconds."""
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


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


def plot_speaker_diarization_interactive(json_file_path, save_dir):
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
    page.render(os.path.join(save_dir, f'{bucket_name}_speaker_diarization.html'))
    print(f"Speaker diarization visualization saved to '{bucket_name}_speaker_diarization.html'")


def plot_speaking_interaction_network(json_file_path, save_dir):
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
    plt.savefig(os.path.join(save_dir, f"{bucket_name}_nsin.png"), dpi=300)
    print(f"Normalized speaking interaction network saved to '{bucket_name}_nsin.png'")


def calculate_speaker_interactions(json_file_path):
    """Calculates the frequency of interactions between speakers in a conversation."""
    json_data = read_json_file(json_file_path)
    df = convert_json_to_dataframe(json_data, ['speakers', 'segment_start_time'])
    normalize_factor = len(df)  # Normalized by number of segments
    interaction_counts = defaultdict(lambda: defaultdict(float))

    for i in range(len(df) - 1):
        current_speakers = df.iloc[i]['speakers']
        next_speakers = df.iloc[i + 1]['speakers']

        # Filter out 'silent' and 'unknown'
        current_speakers = [s for s in current_speakers if s not in ['silent', 'unknown']]
        next_speakers = [s for s in next_speakers if s not in ['silent', 'unknown']]

        # Update interaction counts
        for next_speaker in next_speakers:
            for current_speaker in current_speakers:
                if current_speaker != next_speaker:
                    interaction_counts[next_speaker][current_speaker] += 1 / normalize_factor

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
    """Creates and saves matplotlib plots of various audio analysis metrics across different sessions.

    Args:
        session_names (list): List of session names.
        apd_lst (list): List of Average Pause Durations (APD).
        nttc_lst (list): List of Normalized Turn Taking Counts (NTTC).
        sr_lst (list): List of Silence Ratios (SR).
        peq_lst (list): List of Participation Equalities (PEQ).
        save_dir (str): Directory to save the plots.
    """
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
    print("Audio analysis metrics across sessions saved to 'audio_analysis_across_sessions.png'")


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
    print("Audio analysis metrics across sessions saved to 'audio_analysis_across_sessions.html'")
