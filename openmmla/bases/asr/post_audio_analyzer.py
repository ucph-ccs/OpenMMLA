import base64
import json
import os
import shutil
import tempfile
import wave
from datetime import datetime, timedelta

import numpy as np
import soundfile as sf
from tqdm import tqdm

from openmmla.analysis.asr.analyze import plot_speaking_interaction_network, \
    plot_speaker_diarization_interactive
from openmmla.analysis.asr.transcription import convert_transcription_json_to_txt
from openmmla.bases.base import Base
from openmmla.services.asr.requests import request_voice_activity_detection, request_speech_enhancement, \
    request_speech_separation, request_speech_transcription
from openmmla.utils.audio.auga import normalize_decibel
from openmmla.utils.audio.augf import resample_audio
from openmmla.utils.audio.files import format_wav, segment_wav, crop_and_concatenate_wav
from openmmla.utils.audio.properties import get_audio_properties
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import resolve_url
from .audio_recognizer import AudioRecognizer


class PostAudioAnalyzer(Base):
    logger = get_logger('audio-post-analyzer')

    def __init__(self, project_dir: str | None, config_path: str, filenames: str = None, vad: bool = True,
                 nr: bool = True, sp: bool = False, tr: bool = True):
        """Initialize the PostAudioAnalyzer object.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            filenames: specified filenames in /post-time/origin/ to process (default: all files in the directory)
            vad: whether to use the VAD or not (default: True)
            nr: whether to use the denoiser to enhance speech or not (default: True)
            sp: whether to use the separation model or not (default: False)
            tr: whether to transcribe the audio segments or not (default: True)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.vad = vad
        self.nr = nr
        self.sp = sp
        self.tr = tr

        self.filename: str = ''
        self.session_name: str = ''  # filename without extension
        self.session_logs_dir: str = ''
        self.session_runtime_dir: str = ''
        self.session_segments_dir: str = ''
        self.session_chunks_dir: str = ''

        self._setup_yaml()
        self._setup_directories()

        origin_files = [f for f in os.listdir(self.origin_dir) if
                        not f.startswith('.') and not f.endswith('.DS_Store') and os.path.isfile(
                            os.path.join(self.origin_dir, f))]

        if filenames:
            self.process_files = filenames.split(',')
        else:
            self.process_files = origin_files

        if not self.process_files:
            raise ValueError(
                "You must specify an audio file to process or place it under the /post-time/origin folder.")

        self._setup_objects()

    @property
    def base_type(self):
        return 'PostAnalyzer'

    def _setup_yaml(self):
        self.segment_duration = int(self.config['PostAnalyzer']['segment_duration'])
        self.threshold = float(self.config['PostAnalyzer']['threshold'])
        self.keep_threshold = float(self.config['PostAnalyzer']['keep_threshold'])
        self.speech_transcriber_url = resolve_url(self.config['Server']['asr']['speech_transcription'])
        self.speech_separator_url = resolve_url(self.config['Server']['asr']['speech_separation'])
        self.speech_enhancer_url = resolve_url(self.config['Server']['asr']['speech_enhancement'])
        self.vad_url = resolve_url(self.config['Server']['asr']['voice_activity_detection'])

    def _setup_directories(self):
        self.runtime_dir = os.path.join(self.project_dir, 'post-time', 'runtime')
        self.origin_dir = os.path.join(self.project_dir, 'post-time', 'origin')
        self.profiles_dir = os.path.join(self.project_dir, 'post-time', 'profiles')
        self.temp_dir = os.path.join(self.project_dir, 'post-time', 'temp')
        self.logs_dir = os.path.join(self.project_dir, 'logs')
        self.visualizations_dir = os.path.join(self.project_dir, 'visualizations')

        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.origin_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

    def _setup_objects(self):
        self.recognizer = AudioRecognizer(config_path=self.config_path,
                                          audio_db=os.path.join(self.profiles_dir,
                                                                os.path.splitext(self.process_files[0])[0]))

    def run(self):
        """Process all specified files."""
        self._process_audio_files()

    def _process_audio_files(self):
        for audio_filename in tqdm(self.process_files, desc='Processing audio files', unit='session'):
            self._process_single_audio_file(audio_filename)
            self.logger.info(f"Processing file: {audio_filename}")

    def _process_single_audio_file(self, filename):
        """Process a single audio file.

        Args:
            filename: the name of the audio file to process
        """
        self.filename = filename
        self.session_name = os.path.splitext(filename)[0]
        speakers_corpus_dir = os.path.join(self.origin_dir, self.session_name)

        if not os.path.exists(speakers_corpus_dir):
            os.makedirs(speakers_corpus_dir)
            raise ValueError(
                f"{speakers_corpus_dir} not exist, please add your raw speaker corpus for {filename}")

        if not os.listdir(speakers_corpus_dir):
            raise ValueError(
                f"{speakers_corpus_dir} is empty, please add your raw speaker corpus for {filename}")

        self.session_logs_dir = os.path.join(self.logs_dir, f'session_{self.session_name}')
        self.session_runtime_dir = os.path.join(self.runtime_dir, f'session_{self.session_name}')
        self.session_segments_dir = os.path.join(self.runtime_dir, f'session_{self.session_name}', 'segments')
        self.session_chunks_dir = os.path.join(self.runtime_dir, f'session_{self.session_name}', 'chunks')

        # Clean directories if they exist, otherwise create them
        for directory in [self.session_logs_dir, self.session_runtime_dir, self.session_segments_dir,
                          self.session_chunks_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

        # Reset audio recognizer's audio database
        audio_db = os.path.join(self.profiles_dir, self.session_name)
        self.recognizer.reset_db(audio_db)

        #  Register speakers' raw audio files, set enhance to True to apply NR and VAD
        self._register_speakers(speakers_corpus_dir, enhance=True)

        # Format the origin audio file, segment it, and process the segments
        self._format_origin_file()
        self._segment_formatted_file()
        if self.sp:
            self._process_segments_sp()
        else:
            self._process_segments()

    def _register_speakers(self, speakers_corpus_dir, enhance=True):
        """Register speakers' raw audio files to the recognizer.

        Args:
            speakers_corpus_dir: the directory containing the raw audio files of the speakers
            enhance: whether to apply NR and VAD to the audio files or not
        """
        speaker_corpus = [f for f in os.listdir(speakers_corpus_dir) if
                          not f.startswith('.') and not f.endswith('.DS_Store')]

        for speaker_raw_filename in speaker_corpus:
            speaker_name = speaker_raw_filename.split('.')[0]
            speaker_raw_filepath = os.path.join(speakers_corpus_dir, speaker_raw_filename)
            format_wav(speaker_raw_filepath)

            speaker_audio_db = os.path.join(self.recognizer.audio_db, speaker_name)
            if os.path.exists(speaker_audio_db):
                shutil.rmtree(speaker_audio_db)

            if enhance:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_audio_path = temp_file.name
                shutil.copy2(speaker_raw_filepath, temp_audio_path)
                self._audio_preprocessing(temp_audio_path, inplace=1)
                self.recognizer.register(temp_audio_path, speaker_name)
                os.unlink(temp_audio_path)
            else:
                self.recognizer.register(speaker_raw_filepath, speaker_name)

    def _format_origin_file(self):
        """Format the origin audio file to 16kHz, 16-bit PCM WAV format."""
        origin_path = os.path.join(self.origin_dir, self.filename)
        formatted_audio_path = os.path.join(self.session_runtime_dir, f'{self.session_name}.wav')
        format_wav(origin_path, formatted_audio_path)
        properties = get_audio_properties(formatted_audio_path)
        for key, value in properties.items():
            print(f'{key}: {value}')

    def _segment_formatted_file(self):
        """Segment the formatted audio file into fixed-duration segments."""
        formatted_audio_path = os.path.join(self.session_runtime_dir, f'{self.session_name}.wav')
        segment_wav(formatted_audio_path, self.session_segments_dir, window_length_ms=int(self.segment_duration * 1000))

    def _process_segments(self):
        """Process segments of the audio file.

        Outputs:
            JSON files containing the speaker recognition and transcription results for the session
        """
        speaker_recognition_log_path = os.path.join(self.logs_dir, f'session_{self.session_name}',
                                                    f'session_{self.session_name}_speaker_recognition.json')
        speaker_transcription_log_path = os.path.join(self.logs_dir, f'session_{self.session_name}',
                                                      f'session_{self.session_name}_speaker_transcription.json')
        visualization_dir = os.path.join(self.visualizations_dir, f'session_{self.session_name}')
        os.makedirs(visualization_dir, exist_ok=True)

        segments_path_list = sorted(
            [os.path.join(self.session_segments_dir, f) for f in os.listdir(self.session_segments_dir) if
             not f.endswith('.DS_Store')],
            key=lambda x: int(os.path.basename(x).split('_')[-1][:-4])
        )

        start_time = datetime.now()  # starting datetime object of the conversation
        time = timedelta(seconds=0)
        segment_no = 0
        speaker_recognition_log_entries = []

        with tqdm(total=len(segments_path_list), desc=f"Processing segments for {self.session_name}",
                  unit="segment", position=0, leave=True) as pbar:
            for segment_path in segments_path_list:
                assert isinstance(segment_path, str), "segment_path must be a string"
                processed_segment_path = self._audio_preprocessing(segment_path, inplace=1)

                # Create speaker recognition log entry and update the speaker_recognition_log_entries list
                speaker = 'silent' if processed_segment_path is None else 'unknown'
                similarity = 0
                duration = self.segment_duration

                if processed_segment_path:
                    duration = self.calculate_audio_duration(segment_path)
                    normalize_decibel(segment_path, rms_level=-20)
                    name, similarity = self.recognizer.recognize(segment_path, update_threshold=0.5)
                    if similarity > self.threshold:
                        speaker = name

                time += timedelta(seconds=self.segment_duration)
                segment_start_time = int((start_time + time).timestamp())
                recognition_entry = {
                    "segment_no": segment_no,
                    "segment_start_time": segment_start_time,
                    "speakers": json.dumps([speaker]),
                    "similarities": json.dumps([np.round(np.float64(similarity), 4)]),
                    "durations": json.dumps([duration]),
                }
                speaker_recognition_log_entries.append(recognition_entry)
                segment_no += 1
                pbar.update()

        # Write all speaker_recognition_log_entries to the JSON file at once
        with open(speaker_recognition_log_path, 'w') as f:
            json.dump(speaker_recognition_log_entries, f, indent=5)

        # Visualize the speaker recognition results
        plot_speaking_interaction_network(speaker_recognition_log_path, visualization_dir)
        plot_speaker_diarization_interactive(speaker_recognition_log_path, visualization_dir)

        # Process half-scaled recognition at speaker change borders
        formatted_audio_path = os.path.join(self.session_runtime_dir, f'{self.session_name}.wav')
        chunk_list = self._aggregate_segments_by_speaker(speaker_recognition_log_path)
        borders = self.identify_speaker_change_borders(chunk_list)  # A list of tuple, (border_time, [candidates])
        added_chunks_no = 0
        for index, (border_time, candidates) in enumerate(borders):
            adjusted_index = index + added_chunks_no
            left_temp_path = os.path.join(self.temp_dir, 'left_temp_post_analyzer.wav')
            right_temp_path = os.path.join(self.temp_dir, 'right_temp_post_analyzer.wav')
            left_start_time = max(0, 1000 * (border_time - self.segment_duration / 2))  # Segment before the border
            right_end_time = 1000 * (border_time + self.segment_duration / 2)  # Segment after the border
            crop_and_concatenate_wav(formatted_audio_path, [(left_start_time, 1000 * border_time)], left_temp_path)
            crop_and_concatenate_wav(formatted_audio_path, [(1000 * border_time, right_end_time)], right_temp_path)

            if self._apply_vad(left_temp_path, inplace=0):
                left_speaker, _ = self.recognizer.recognize_among_candidates(left_temp_path, candidates,
                                                                             candidates[0], self.keep_threshold)
            else:
                left_speaker = 'silent'

            if self._apply_vad(right_temp_path, inplace=0):
                right_speaker, _ = self.recognizer.recognize_among_candidates(right_temp_path, candidates,
                                                                              candidates[1], self.keep_threshold)
            else:
                right_speaker = 'silent'

            result = self.update_chunk_list(chunk_list, adjusted_index, left_speaker, right_speaker,
                                            self.segment_duration / 2)
            if result:
                added_chunks_no += result

        # Aggregate the chunks and transcribe them
        chunk_list = self.aggregate_chunks(chunk_list)
        speaker_transcription_log_entries = self._transcribe_by_chunks(chunk_list)

        with open(speaker_transcription_log_path, 'w') as f:
            json.dump(speaker_transcription_log_entries, f, indent=5)

        convert_transcription_json_to_txt(speaker_transcription_log_path)

    def _process_segments_sp(self):
        """Process segments of the audio file with using the separation model.

        Outputs:
            JSON files containing the speaker recognition and transcription results for the session
        """
        speaker_recognition_log_path = os.path.join(self.logs_dir, f'session_{self.session_name}',
                                                    f'session_{self.session_name}_speaker_recognition.json')
        speaker_transcription_log_path = os.path.join(self.logs_dir, f'session_{self.session_name}',
                                                      f'session_{self.session_name}_speaker_transcription.json')
        visualization_dir = os.path.join(self.visualizations_dir, f'session_{self.session_name}')
        os.makedirs(visualization_dir, exist_ok=True)

        segments_path_list = sorted(
            [os.path.join(self.session_segments_dir, f) for f in os.listdir(self.session_segments_dir) if
             not f.endswith('.DS_Store')],
            key=lambda x: int(os.path.basename(x).split('_')[-1][:-4])
        )

        start_time = datetime.now()  # starting datetime object of the conversation
        time = timedelta(seconds=0)
        segment_no = 0
        speaker_recognition_log_entries = []

        with tqdm(total=len(segments_path_list), desc=f"Processing segments for {self.session_name}",
                  unit="segment", position=0, leave=True) as pbar:
            for segment_path in segments_path_list:
                assert isinstance(segment_path, str), "segment_path must be a string"

                time += timedelta(seconds=self.segment_duration)
                segment_start_time = int((start_time + time).timestamp())
                speakers, similarities, durations = [], [], []
                processed_segment_path = self._audio_preprocessing(segment_path, inplace=1)

                if processed_segment_path:
                    resample_audio(segment_path, 8000)
                    result = self._separate_speech(segment_path)

                    for i, signal in enumerate(result):
                        save_file = f'{segment_path[:-4]}_spk{i}.wav'
                        sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)

                        # speaker recognition on separated signals
                        processed_save_file = self._apply_vad(save_file, inplace=0)
                        speaker = 'unknown' if processed_save_file else 'silent'
                        duration = self.segment_duration
                        similarity = 0

                        if processed_save_file:
                            duration = self.calculate_audio_duration(save_file)
                            normalize_decibel(save_file, rms_level=-20)
                            name, similarity = self.recognizer.recognize(save_file)

                            if similarity > self.threshold:
                                speaker = name

                        speakers.append(speaker)
                        similarities.append(np.round(np.float64(similarity), 4))
                        durations.append(duration)
                else:
                    speakers.append('silent')
                    similarities.append(0)
                    durations.append(self.segment_duration)

                final_speakers, final_similarities, final_durations = self._finalize_separated_speaker_recognition(
                    speakers, similarities, durations)
                speaker_recognition_log_entries.append(
                    self.speaker_recognition_results(segment_no, segment_start_time, final_speakers, final_similarities,
                                                     final_durations))
                segment_no += 1
                pbar.update()

        # Write all speaker_recognition_log_entries to the JSON file at once
        with open(speaker_recognition_log_path, 'w') as f:
            json.dump(speaker_recognition_log_entries, f, indent=5)

        # Visualize the speaker recognition results
        plot_speaking_interaction_network(speaker_recognition_log_path, visualization_dir)
        plot_speaker_diarization_interactive(speaker_recognition_log_path, visualization_dir)

        # Aggregate segments into chunks and transcribe them
        chunk_list = self._aggregate_segments_by_speaker(speaker_recognition_log_path)
        speaker_transcription_log_entries = self._transcribe_by_chunks(chunk_list)

        with open(speaker_transcription_log_path, 'w') as f:
            json.dump(speaker_transcription_log_entries, f, indent=5)

        convert_transcription_json_to_txt(speaker_transcription_log_path)

    def _aggregate_segments_by_speaker(self, speaker_recognition_log_path):
        """Aggregate segments by speaker.

        Args:
            speaker_recognition_log_path: the path to the speaker recognition log file

        Returns:
            A list of tuples, each containing the speaker and the start and end times of the segment
        """
        with open(speaker_recognition_log_path, 'r') as file:
            speaker_recognition_log = json.load(file)

        speakers_segments = {}  # Dictionary to hold the current speaker segments
        chunk_list = []
        for entry in speaker_recognition_log:
            segment_start_time = entry['segment_start_time']
            segment_end_time = segment_start_time + self.segment_duration
            speakers = json.loads(entry['speakers'])

            # Update existing speakers' end time and add new speakers
            for speaker in speakers:
                if speaker in speakers_segments:
                    speakers_segments[speaker][1] = segment_end_time
                else:
                    speakers_segments[speaker] = [segment_start_time, segment_end_time]

            # Check for speakers who are not in the current segment
            for speaker in list(speakers_segments.keys()):
                if speaker not in speakers:
                    # Update the end time for this speaker and then remove from the dictionary
                    speakers_segments[speaker][1] = segment_start_time
                    chunk_list.append((speaker, [speakers_segments[speaker][0], speakers_segments[speaker][1]]))
                    del speakers_segments[speaker]

        # Handle the last segment for remaining speakers
        if speaker_recognition_log:
            last_segment_start_time = speaker_recognition_log[-1]['segment_start_time']
            last_segment_end_time = last_segment_start_time + self.segment_duration
            for speaker, times in speakers_segments.items():
                times[1] = last_segment_end_time
                chunk_list.append((speaker, times))

        return chunk_list

    def _transcribe_by_chunks(self, chunk_list):
        """Transcribe the audio by chunk.

        Args:
            chunk_list: chunk list containing speaker and start and end times

        Returns:
            A list of transcription entries
        """
        formatted_audio_path = os.path.join(self.session_runtime_dir, f'{self.session_name}.wav')
        entries = []
        if self.tr:
            audio_start_time = chunk_list[0][1][0]
            with tqdm(total=len(chunk_list), desc=f"Processing aggregated segments chunk",
                      unit="chunk", position=0, leave=True) as pbar:
                for index, chunk in enumerate(chunk_list):
                    speaker = chunk[0]
                    chunk_start_time = chunk[1][0]
                    chunk_end_time = chunk[1][1]
                    start_offset_ms = 1000 * (chunk_start_time - audio_start_time)
                    end_offset_ms = 1000 * (chunk_end_time - audio_start_time)
                    chunk_path = os.path.join(self.session_chunks_dir, f'chunk_{index}_{speaker}.wav')
                    crop_and_concatenate_wav(formatted_audio_path, [(start_offset_ms, end_offset_ms)], chunk_path)
                    self._apply_nr(chunk_path)
                    normalize_decibel(chunk_path, rms_level=-20)
                    text = self._transcribe(chunk_path) if speaker != 'silent' else ''
                    transcription_entry = {
                        "chunk_no": index,
                        "chunk_start_time": chunk_start_time,
                        "chunk_end_time": chunk_end_time,
                        "speaker": speaker,
                        "text": text,
                    }
                    print(transcription_entry)
                    entries.append(transcription_entry)
                    pbar.update()

        return entries

    def _transcribe(self, input_path):
        """Transcribe audio file to text.

        Args:
            input_path: input audio file path

        Returns:
            transcribed text
        """
        with wave.open(input_path, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            frame_rate = wav_file.getframerate()

        text = request_speech_transcription(frames, frame_rate, f'{self.base_type.lower()}',
                                            self.speech_transcriber_url)
        return text

    def _separate_speech(self, segment_audio_path) -> list:
        """Separate speech from the overlapped segment audio.

        Args:
            segment_audio_path: input audio file path

        Returns:
            separated speech signals
        """
        separated_result = request_speech_separation(segment_audio_path, f'{self.base_type.lower()}',
                                                     self.speech_separator_url)
        result = [base64.b64decode(encoded_bytes_stream) for encoded_bytes_stream in separated_result]
        return result

    def _audio_preprocessing(self, input_path: str, inplace: int) -> str | None:
        """Apply both NR and VAD processing to audio file.


        Args:
            input_path: input audio file path
            inplace: whether to overwrite the input file when applying vad

        Returns:
            processed audio file path
        """
        self._apply_nr(input_path)
        return self._apply_vad(input_path, inplace)

    def _apply_vad(self, input_path: str, inplace: int) -> str | None:
        """Apply voice activity detection to audio file.

        Args:
            input_path: input audio file path
            inplace: whether to overwrite the input file

        Returns:
            processed audio file path
        """
        if self.vad:
            return request_voice_activity_detection(input_path, f'{self.base_type.lower()}', inplace, self.vad_url)
        return input_path

    def _apply_nr(self, input_path: str) -> str:
        """Apply noise reduction to audio file.

        Args:
            input_path: input audio file path

        Returns:
            processed audio file path
        """
        if self.nr:
            request_speech_enhancement(input_path, f'{self.base_type.lower()}', self.speech_enhancer_url)
        return input_path

    def _finalize_separated_speaker_recognition(self, speakers, similarities, durations):
        """Finalize the speaker recognition results of separated signals to handle edge cases.

        Args:
            speakers: list of recognized speakers
            similarities: list of speaker recognition similarities
            durations: list of speaking durations

        Returns:
            A tuple containing the final speakers, similarities, and durations
        """
        # If there's only one speaker, keep it as is
        if len(speakers) == 1:
            return speakers, similarities, durations

        # If both speakers are the same, keep the one with the highest similarity
        if speakers[0] == speakers[1]:
            max_similarity_index = 0 if similarities[0] > similarities[1] else 1
            return [speakers[max_similarity_index]], [similarities[max_similarity_index]], [
                durations[max_similarity_index]]

        # If both are real speakers, keep them as is
        if all(speaker not in ['silent', 'unknown'] for speaker in speakers):
            return speakers, similarities, durations

        # If there's at least one real speaker, keep only the real ones
        real_speakers, real_similarities, real_durations = self.filter_real_speakers(speakers, similarities, durations)
        if real_speakers:
            return real_speakers, real_similarities, real_durations

        # Special cases for 'silent' and 'unknown'
        if 'unknown' in speakers and 'silent' in speakers:
            return ['unknown'], [similarities[speakers.index('unknown')]], [durations[speakers.index('unknown')]]
        if speakers.count('silent') == 2:
            return ['silent'], [similarities[0]], [durations[0]]
        if speakers.count('unknown') == 2:
            return ['unknown'], [max(similarities)], [max(durations)]

        return [], [], []

    @staticmethod
    def filter_real_speakers(speakers, similarities, durations):
        """Filter out 'silent' and 'unknown' from speakers and their associated similarities and durations.

        Args:
            speakers: list of recognized speakers
            similarities: list of speaker recognition similarities
            durations: list of speaking durations

        Returns:
            A tuple containing the filtered real speakers, similarities, and durations
        """
        real_speakers = []
        real_similarities = []
        real_durations = []
        for i, speaker in enumerate(speakers):
            if speaker not in ['silent', 'unknown']:
                real_speakers.append(speaker)
                real_similarities.append(similarities[i])
                real_durations.append(durations[i])

        return real_speakers, real_similarities, real_durations

    @staticmethod
    def speaker_recognition_results(segment_no, segment_start_time, final_speakers=None,
                                    final_similarities=None, final_durations=None, ):
        """Create a speaker recognition log entry.

        Args:
            segment_no: segment number
            segment_start_time: segment start time
            final_speakers: final recognized speakers
            final_similarities: final speaker recognition similarities
            final_durations: final speaking durations

        Returns:
            A dictionary containing the speaker recognition results
        """
        recognition_entry = {
            "segment_no": segment_no,
            "segment_start_time": segment_start_time,
            "speakers": json.dumps(final_speakers),
            "similarities": json.dumps(final_similarities),
            "durations": json.dumps(final_durations),
        }
        return recognition_entry

    @staticmethod
    def identify_speaker_change_borders(chunk_list):
        """Identify speaker change borders from the chunk list.

        Note: this function only applicable to recognition without speech separation

        Args:
            chunk_list: A list of tuples, each containing the speaker and the start and end times of the segment

        Returns:
            A list of tuples, each containing the time of the speaker change and the candidates
        """

        borders = []
        start_time = chunk_list[0][1][0]
        for i in range(1, len(chunk_list)):
            if chunk_list[i][0] != chunk_list[i - 1][0]:  # Speaker change detected
                border_time = chunk_list[i][1][0] - start_time  # Time offset from the start of the conversation
                candidates = [chunk_list[i - 1][0], chunk_list[i][0]]
                borders.append((border_time, candidates))

        return borders

    @staticmethod
    def update_chunk_list(chunk_list, border_index, left_speaker, right_speaker, segment_half_duration):
        """Update the chunk list by adding two new half-scaled recognized chunks at the speaker change border.

        Note: this function only applicable to recognition without speech separation

        Args:
            chunk_list: A list of tuples, each containing the speaker and the start and end times of the segment
            border_index: the index of the speaker change border
            left_speaker: the speaker on the left side of the border
            right_speaker: the speaker on the right side of the border
            segment_half_duration: half of the segment duration

        Returns:
            The number of added chunks
        """
        left_chunk_speaker = chunk_list[border_index][0]
        right_chunk_speaker = chunk_list[border_index + 1][0]

        first_start_time = chunk_list[border_index][1][0]
        first_end_time = chunk_list[border_index][1][1] - segment_half_duration
        second_start_time = first_end_time
        second_end_time = chunk_list[border_index][1][1]
        third_start_time = second_end_time
        third_end_time = chunk_list[border_index + 1][1][0] + segment_half_duration
        fourth_start_time = third_end_time
        fourth_end_time = chunk_list[border_index + 1][1][1]

        chunk_list[border_index] = (left_chunk_speaker, (first_start_time, first_end_time))
        new_chunk_left = (left_speaker, (second_start_time, second_end_time))
        chunk_list.insert(border_index + 1, new_chunk_left)
        new_chunk_right = (right_speaker, (third_start_time, third_end_time))
        chunk_list.insert(border_index + 2, new_chunk_right)
        chunk_list[border_index + 3] = (right_chunk_speaker, (fourth_start_time, fourth_end_time))

        return 2

    @staticmethod
    def aggregate_chunks(chunk_list):
        """Aggregate the chunks by combining consecutive chunks with the same speaker.

        Args:
            chunk_list: a list of tuples, each containing the speaker and the start and end times of the segment

        Returns:
            A list of aggregated chunks
        """
        aggregated_chunks = []
        current_speaker = None
        current_start_time = None
        current_end_time = None

        for speaker, (start_time, end_time) in chunk_list:
            # Skip chunks with identical start and end times
            if start_time == end_time:
                continue

            # If the current speaker is the same as the last, extend the current chunk
            if speaker == current_speaker:
                current_end_time = end_time
            else:
                # If there's a current chunk, add it to the aggregated list
                if current_speaker is not None:
                    aggregated_chunks.append((current_speaker, (current_start_time, current_end_time)))

                # Start a new chunk
                current_speaker = speaker
                current_start_time = start_time
                current_end_time = end_time

        # Add the last chunk if it exists
        if current_speaker is not None:
            aggregated_chunks.append((current_speaker, (current_start_time, current_end_time)))

        return aggregated_chunks

    @staticmethod
    def calculate_audio_duration(audio_path):
        """Calculate the duration of an audio file in seconds.

        Args:
            audio_path: audio file path

        Returns:
            The duration of the audio file in seconds
        """
        with sf.SoundFile(audio_path) as f:
            return len(f) / f.samplerate
