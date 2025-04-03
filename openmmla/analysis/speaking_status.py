import ast
from typing import List


def speaking_status(speaker_recognition: List[dict], badge_relations: List[dict], time_threshold: float = 2.0) -> tuple[
    int, dict]:
    """Determine speaking status based on speaker recognition and badge relations.
    
    Args:
        speaker_recognition: List of speaker recognition records in the time window
        badge_relations: List of badge relation records in the time window
        time_threshold: Maximum time difference (in seconds) to consider events related
    
    Returns:
        status: The highest status of all speakers
            
        0: Nobody speaking (Silent)
        1: Someone speaking but not facing others
        2: Someone speaking and facing others
        
        speaker_statuses: The status and time of each speaker
        {speaker_id: [(status, time), ...]}
    """
    # If no speaker recognition data, return 0 (silent)
    if not speaker_recognition:
        return 0, {}

    # Initialize dictionary to track speaker statuses and times
    # {speaker_id: [(status, time), ...]}
    speaker_records = {}

    # Process speaker recognition data
    for record in speaker_recognition:
        speakers_str = record.get('speakers', '[]')
        speak_time = record.get('time', 0.0)

        try:
            speakers = ast.literal_eval(speakers_str)
            # Skip if speakers is empty or only contains "silent"
            if not speakers or all(s == "silent" for s in speakers):
                continue

            # Add status-time pair for each speaker
            for speaker in speakers:
                if speaker != "silent":
                    if speaker not in speaker_records:
                        speaker_records[speaker] = []
                    speaker_records[speaker].append((1, speak_time))

        except (ValueError, SyntaxError):
            continue

    # If no valid speakers found, return 0
    if not speaker_records:
        return 0, {}

    # Process badge relations data
    for relation in badge_relations:
        graph_str = relation.get('graph', '{}')
        relation_time = relation.get('time', 0.0)

        try:
            graph = ast.literal_eval(graph_str)

            # Check each speaker
            for speaker_id in speaker_records:
                speaker_id_str = str(speaker_id)

                # If the speaker exists in the graph and is facing someone
                if speaker_id_str in graph and graph[speaker_id_str]:
                    # Check each speaking time for this speaker
                    for idx, (status, speak_time) in enumerate(speaker_records[speaker_id]):
                        time_diff = abs(relation_time - speak_time)

                        if time_diff <= time_threshold:
                            # Update status to 2 for this specific time point
                            speaker_records[speaker_id][idx] = (2, speak_time)

        except (ValueError, SyntaxError):
            continue

    # Find the highest status across all speakers and their time points
    max_status = 0
    for records in speaker_records.values():
        for status, _ in records:
            max_status = max(max_status, status)

    return max_status, speaker_records
