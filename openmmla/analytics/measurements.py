import ast
import json
from typing import List

import pandas as pd


def read_logs(log_path: str, time_column: str | None = None) -> pd.DataFrame:
    """Read the log file and return a pandas DataFrame.
    
    Args:
        log_path: Path to the JSON log file
        time_column: Name of the column to be used as time reference
        
    Returns:
        DataFrame with an additional 'time' column that mirrors the specified time_column
    """
    with open(log_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if time_column:
        df['time'] = df[time_column]
    return df


def aggregate_measurements(dfs: List[pd.DataFrame], window_size: int, save_path: str | None = None) -> pd.DataFrame:
    """Aggregate measurements from multiple dataframes into a single dataframe by time window.
    
    Args:
        dfs: List of pandas DataFrames, each containing measurement data
        window_size: Size of the time window in seconds
        save_path: Optional path to save the aggregated data
    
    Returns:
        DataFrame with aggregated measurements by time window
    """
    start_time = min([df['time'].min() for df in dfs])
    end_time = max([df['time'].max() for df in dfs])

    num_windows = int((end_time - start_time) // window_size) + 1
    windows = []

    measurement_types = [df['_measurement'].iloc[0] for df in dfs]

    # Create windows and aggregate measurements
    for i in range(num_windows):
        window_start = start_time + (i * window_size)
        window_end = window_start + window_size

        window_data = {
            'window_start': window_start,
            'window_end': window_end
        }

        # For each measurement type, collect entries within the window
        for df, measurement_type in zip(dfs, measurement_types):
            window_entries = df[
                (df['time'] >= window_start) &
                (df['time'] < window_end)
                ]

            if not window_entries.empty:
                # Convert DataFrame entries to dictionary format
                entries_list = window_entries.to_dict('records')
                window_data[measurement_type] = entries_list
            else:
                window_data[measurement_type] = None

        windows.append(window_data)

    result_df = pd.DataFrame(windows)

    if save_path:
        result_df.to_json(save_path, orient='records', indent=4)

    return result_df

#
#
# def location_status(badge_translation: tuple, zone: tuple) -> bool:
#     """Determine if a badge location is inside a specified zone.
#
#     Args:
#         badge_translation: Tuple of (x, y) coordinates of the badge
#         zone: Tuple of (x_min, y_min, x_max, y_max) defining the zone boundaries
#
#     Returns:
#         bool: True if badge is inside the zone, False otherwise
#     """
#     # Extract coordinates
#     badge_x, badge_y = badge_translation
#     zone_x_min, zone_y_min, zone_x_max, zone_y_max = zone
#
#     # Check if badge coordinates are within zone boundaries
#     is_inside = (
#             zone_x_min <= badge_x <= zone_x_max and
#             zone_y_min <= badge_y <= zone_y_max
#     )
#
#     return is_inside
#
#
# def action_status(action_type: str, action_map: dict | None = None) -> int:
#     """Determine if an action is task-relevant based on the action map.
#
#     Args:
#         action_type: String describing the type of action (e.g., 'working with software', 'distracted')
#         action_map: Dictionary mapping action types to their task relevance
#
#     Returns:
#         int: 1 if action is task-relevant, 0 if not relevant or unclear
#     """
#     # Default action map
#     default_map = {
#         'working with software': True,
#         'reading documentation': True,
#         'team discussion': True,
#         'writing code': True,
#         'debugging': True,
#         'distracted': False,
#         'on phone': False,
#         'idle': False,
#         'not clear': False
#     }
#
#     action_map = action_map or default_map
#     action_type = action_type.lower().strip()
#
#     return 1 if action_map.get(action_type, False) else 0
#
#
# LATENT_PATTERNS = {
#     "constructing_shared_knowledge": {
#         "description": "Expresses one's own ideas and attempts to understand others' ideas",
#         "patterns": [
#             # speaking_status, speaking_content, location_status, action_status
#             (2, 1, 1, 1),  # Ideal pattern: speaking while facing others, relevant content, in zone, relevant action
#             (1, 1, 1, 1),  # Speaking but not facing, still constructing knowledge
#             (2, 1, 1, -1),  # Don't need to consider action status for knowledge sharing
#             (1, 1, 1, -1)  # Alternative pattern
#         ]
#     },
#
#     "negotiation_coordination": {
#         "description": "Achieves an agreed solution plan ready to execute",
#         "patterns": [
#             # speaking_status, speaking_content, location_status, action_status
#             (2, 1, 1, 1),  # Ideal: speaking while facing, relevant content, in zone, relevant action
#             (2, 1, 1, -1),  # Coordinating through discussion, action status less important
#             (1, 1, 1, 1),  # Speaking without facing but still coordinating
#             (2, -1, 1, 1)  # Content might be varied during coordination
#         ]
#     },
#
#     "maintaining_team_function": {
#         "description": "Sustains the team dynamics",
#         "patterns": [
#             # speaking_status, speaking_content, location_status, action_status
#             (2, -1, 1, -1),  # Engaging with team (facing others, in zone)
#             (1, -1, 1, -1),  # Present and speaking in zone
#             (-1, -1, 1, 2),  # In zone doing relevant work
#             (2, 1, 1, -1)  # Active discussion in zone
#         ]
#     }
# }
#
#
# def match_pattern(current_state: tuple, pattern: tuple) -> bool:
#     """
#     Check if current state matches a pattern, considering 'x' wildcards
#     """
#     return all(
#         p == 'x' or c == p
#         for c, p in zip(current_state, pattern)
#     )
#
#
# def detect_facet(speaking_status: int, speaking_content: int,
#                  location_status: int, action_status: int) -> List[str]:
#     """
#     Detect which facets are present based on current indicators
#     """
#     current_state = (speaking_status, speaking_content, location_status, action_status)
#     active_facets = []
#
#     for facet, facet_info in LATENT_PATTERNS.items():
#         for pattern in facet_info["patterns"]:
#             if match_pattern(current_state, pattern):
#                 active_facets.append(facet)
#                 break
#
#     return active_facets
