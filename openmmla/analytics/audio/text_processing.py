import json
import os
import re
from datetime import datetime, timedelta

import jiwer


def convert_transcription_json_to_txt(json_file_path):
    """Converts a JSON file with speech data to a formatted text file."""

    def unix_to_hms(timestamp, first_timestamp):
        """Converts Unix timestamp to HH:MM:SS format based on the first timestamp."""
        offset = timedelta(seconds=(timestamp - first_timestamp))
        return str(datetime(1970, 1, 1) + offset).split()[1]

    def format_text(data):
        """Formats the text as per requirement."""
        if not data:
            return ""

        first_timestamp = data[0]["chunk_start_time"]
        formatted_text = ""
        for entry in data:
            speaker = entry["speaker"].capitalize()
            start_time = unix_to_hms(entry["chunk_start_time"], first_timestamp)
            end_time = unix_to_hms(entry["chunk_end_time"], first_timestamp)
            text = entry.get("text")
            text = text.strip() if text is not None else ""
            formatted_text += f"{speaker}  {start_time}  {end_time}\n{text}\n\n"
        return formatted_text

    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    formatted_data = format_text(json_data)
    output_file_path = json_file_path.rsplit('.', 1)[0] + '.txt'
    with open(output_file_path, "w") as file:
        file.write(formatted_data)


def generate_rttm_from_txt(file_path, file_id, file_duration=None):
    """ Generate RTTM format from a file and save it in the same directory. """

    def parse_line(line):
        """ Extract speaker name, start time, and optionally end time from a line of text. """
        match = re.match(r"(\w+) +(\d{1,2}:\d{2})(:\d{2})?(\.\d+)?( +(\d{1,2}:\d{2})(:\d{2})?(\.\d+)?)?", line)
        if match:
            speaker = match.group(1)
            start_time_str = match.group(2) + (match.group(3) if match.group(3) else '') + (
                match.group(4) if match.group(4) else '')
            end_time_str = None
            if match.group(6):
                end_time_str = match.group(6) + (match.group(7) if match.group(7) else '') + (
                    match.group(8) if match.group(8) else '')
            return speaker, start_time_str, end_time_str
        return None, None, None

    def time_to_seconds(time_str):
        """ Convert time string to seconds, including fractional seconds. """
        parts = time_str.split(':')
        if len(parts) == 2:  # Format 0:00 or 0:00.0
            m, s = parts[0], parts[1]
            return float(m) * 60 + float(s)
        elif len(parts) == 3:  # Format 00:00:00 or 00:00:00.0
            h, m, s = parts[0], parts[1], parts[2]
            return float(h) * 3600 + float(m) * 60 + float(s)
        return 0

    directory = os.path.dirname(file_path)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    segments = []
    for i in range(len(lines)):
        speaker, start_time, end_time = parse_line(lines[i])
        if speaker and speaker not in ['Silent', 'Unknown']:
            start_time_seconds = time_to_seconds(start_time)
            if end_time:
                end_time_seconds = time_to_seconds(end_time)
                duration = end_time_seconds - start_time_seconds
            else:
                # Determine duration based on next speaker's start time or file duration
                duration = 10  # Default value
                for j in range(i + 1, len(lines)):
                    next_speaker, next_start_time, _ = parse_line(lines[j])
                    if next_speaker:
                        next_start_time_seconds = time_to_seconds(next_start_time)
                        duration = next_start_time_seconds - start_time_seconds
                        break
                    if j == len(lines) - 1 and file_duration:
                        duration = file_duration - start_time_seconds

            segments.append(
                f"SPEAKER {file_id} 1 {start_time_seconds:.6f} {duration:.6f} <NA> <NA> {speaker} <NA> <NA>")

    rttm_content = '\n'.join(segments)
    rttm_file_path = os.path.join(directory, f"{file_id}.rttm")
    with open(rttm_file_path, 'w') as file:
        file.write(rttm_content)

    return rttm_content, rttm_file_path


def extract_transcription_from_txt(file_path):
    """ Extract transcription text from txt file. """
    transcription = ""

    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"(\w+) +(\d{1,2}:\d{2})(:\d{2})?", line)
            if match:
                continue
            if len(line) > 0:
                transcription += line.strip() + " "

    return transcription


def extract_transcription_without_unknown(file_path):
    """ Extract transcription text from a given file, skipping speaker lines, and all lines for 'Unknown' speaker. """
    transcription = ""
    skip_lines = False  # Flag to indicate whether to skip lines

    with open(file_path, 'r') as file:
        for line in file:
            speaker_match = re.match(r"(\w+) +(\d{1,2}:\d{2})(:\d{2})?", line)
            if speaker_match:
                if speaker_match.group(1) == "Unknown":
                    skip_lines = True  # Skip all lines until next speaker if 'Unknown'
                else:
                    skip_lines = False
                continue  # Skip all speaker lines
            if skip_lines:
                continue
            if len(line.strip()) > 0:
                transcription += line.strip() + " "
    return transcription


def preprocess_text_with_jiwer(text):
    """ Preprocess the text using jiwer transformations. """
    transformation = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    transformed_text = transformation(text)
    words = [word for sublist in transformed_text for word in sublist]

    return words


def word_overlap_percentage(text1, text2):
    """ Calculate the single word overlapping percentage between generated text and referenced text. """
    words_text1 = set(preprocess_text_with_jiwer(text1))
    words_text2 = set(preprocess_text_with_jiwer(text2))
    common_words = words_text1.intersection(words_text2)  # Find common words
    total_unique_words = len(words_text2)

    if total_unique_words == 0:
        return 0.0

    overlap_percentage = (len(common_words) / total_unique_words) * 100

    return overlap_percentage
