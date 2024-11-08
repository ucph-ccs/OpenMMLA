import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import soundfile as sf
from tqdm import tqdm

from openmmla.analytics.audio.analyze import plot_speaking_interaction_network, plot_speaker_diarization_interactive
from openmmla.analytics.audio.text_processing import convert_transcription_json_to_txt
from openmmla.bases import Base
from openmmla.utils.audio.auga import normalize_decibel
from openmmla.utils.audio.processing import format_wav, get_audio_properties, segment_wav, crop_and_concatenate_wav, \
    resample_audio_file
from openmmla.utils.audio.transcriber import get_transcriber
from openmmla.utils.logger import get_logger
from .audio_recognizer import AudioRecognizer
from .audio_recorder import AudioRecorder


class AudioPostAnalyzer(Base):
    logger = get_logger('audio-post-analyzer')

    def __init__(self, project_dir: str = None, config_path: str = None, filename: str = None, vad: bool = True,
                 nr: bool = True, sp: bool = True, tr: bool = True):
        """Initialize the AudioPostAnalyzer object.

        Args:
            project_dir: root directory of the project, default to the directory of the caller module when not specified
            config_path: path to the configuration file, either absolute or relative to the root directory, default to
                            'conf/audio_base.ini' when not specified
            filename: specified filename in /audio/post-time/origin/ to process, default to all files in the directory
                            when not specified
            vad: whether to use the VAD or not, default to True
            nr: whether to use the denoiser to enhance speech or not, default to True
            sp: whether to use the separation model or not, default to True
            tr: whether to transcribe the audio segments or not, default to True
        """
        super().__init__(project_dir, config_path)
        self.filename = filename
        self.vad = vad
        self.nr = nr
        self.sp = sp
        self.tr = tr

        self._setup_from_yaml()
        self._setup_directories()
        self._setup_objects()

        filenames_in_dir = [f for f in os.listdir(self.audio_origin_dir) if not f.startswith('.')]
        self.filenames_to_process = [filename] if filename else filenames_in_dir
        if not self.filenames_to_process:
            raise ValueError("You must specify an audio file to process or place it under the audio/post-time/origin "
                             "folder.")

    def _setup_from_yaml(self):
        self.frame_rate = int(self.config['PostAnalyzer']['frame_rate'])
        self.channels = int(self.config['PostAnalyzer']['channels'])
        self.sample_width = int(self.config['PostAnalyzer']['sample_width'])
        self.segment_duration = int(self.config['PostAnalyzer']['segment_duration'])
        self.threshold = float(self.config['PostAnalyzer']['threshold'])
        self.keep_threshold = float(self.config['PostAnalyzer']['keep_threshold'])
        self.sr_model = self.config['PostAnalyzer']['sr_model']
        self.language = self.config['PostAnalyzer']['language']

    def _setup_directories(self):
        self.speakers_corpus_dir = os.path.join(self.project_dir, 'audio_db', 'post-time')
        self.audio_origin_dir = os.path.join(self.project_dir, 'audio', 'post-time', 'origin')
        self.audio_formatted_dir = os.path.join(self.project_dir, 'audio', 'post-time', 'formatted')
        self.audio_segments_dir = os.path.join(self.project_dir, 'audio', 'post-time', 'segments')
        self.audio_chunks_dir = os.path.join(self.project_dir, 'audio', 'post-time', 'chunks')
        self.audio_temp_dir = os.path.join(self.project_dir, 'audio', 'temp')
        self.audio_db_dir = os.path.join(self.project_dir, 'audio_db')
        self.logs_dir = os.path.join(self.project_dir, 'logs')
        self.visualizations_dir = os.path.join(self.project_dir, 'visualizations')
        os.makedirs(self.speakers_corpus_dir, exist_ok=True)
        os.makedirs(self.audio_origin_dir, exist_ok=True)
        os.makedirs(self.audio_formatted_dir, exist_ok=True)
        os.makedirs(self.audio_segments_dir, exist_ok=True)
        os.makedirs(self.audio_chunks_dir, exist_ok=True)
        os.makedirs(self.audio_temp_dir, exist_ok=True)
        os.makedirs(self.audio_db_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

    def _setup_objects(self):
        if self.tr:
            self.transcriber = get_transcriber(self.config['PostAnalyzer']['tr_model'], language=self.language,
                                               use_cuda=False)
        if self.sp:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            logging.getLogger('modelscope').setLevel(logging.WARNING)
            try:
                self.separator = pipeline(Tasks.speech_separation, model=self.config['PostAnalyzer']['sp_model'])
            except ValueError:
                self.separator = pipeline(Tasks.speech_separation, model=self.config['PostAnalyzer']['sp_model_local'])

        self.recorder = AudioRecorder(config_path=self.config_path, vad_enable=self.vad, nr_enable=self.nr,
                                      use_onnx=False, use_cuda=True, vad_local=True, nr_local=True)
        self.recognizer = AudioRecognizer(
            config_path=self.config_path,
            audio_db=os.path.join(self.audio_db_dir, os.path.splitext(self.filenames_to_process[0])[0]),
            model_path=self.sr_model, use_onnx=False, use_cuda=True, local=True)

    def run(self):
        """Process all specified files under the audio/post-time/origin folder."""
        for audio_file_name in tqdm(self.filenames_to_process, desc='Processing audio files', unit='session'):
            if audio_file_name.endswith('.DS_Store'):
                continue
            self._process_single_audio_file(audio_file_name)
            self.logger.info(f"Processing file: {audio_file_name}")

    def _process_single_audio_file(self, filename):
        """Process a single audio file.

        Args:
            filename: the name of the audio file to process
        """
        session_name = os.path.splitext(filename)[0]  # Get the session name from the filename without extension
        speakers_corpus_dir = os.path.join(self.speakers_corpus_dir, session_name)

        if not os.path.exists(speakers_corpus_dir):
            os.makedirs(speakers_corpus_dir)
            raise ValueError(
                f"{speakers_corpus_dir} not exist, please add your raw speaker corpus for {filename}")

        if not os.listdir(speakers_corpus_dir):
            raise ValueError(
                f"{speakers_corpus_dir} is empty, please add your raw speaker corpus for {filename}")

        logs_dir = os.path.join(self.logs_dir, f'session_{session_name}')
        os.makedirs(logs_dir, exist_ok=True)

        # Reset audio recognizer's audio database
        speakers_db = os.path.join(self.audio_db_dir, session_name)
        self.recognizer.reset_db(speakers_db)

        #  Register speakers' raw audio files, set enhance to True to apply NR and VAD
        self._register_audio_file_speakers(speakers_corpus_dir, enhance=True)

        # Format the origin audio file, segment it, and process the segments
        formatted_audio_file_path = self._format_origin_audio_file(filename)
        self._segment_audio_file(session_name, formatted_audio_file_path)
        if self.sp:
            self._process_segments_sp(session_name)
        else:
            self._process_segments(session_name)

    def _register_audio_file_speakers(self, speakers_corpus_dir, enhance=True):
        """Register speakers' raw audio files to the recognizer.

        Args:
            speakers_corpus_dir: the directory containing the raw audio files of the speakers
            enhance: whether to apply NR and VAD to the audio files or not
        """
        for speaker_raw_audio in os.listdir(speakers_corpus_dir):
            if speaker_raw_audio.endswith('.DS_Store'):
                continue
            speaker_raw_audio_path = os.path.join(speakers_corpus_dir, speaker_raw_audio)
            speaker_raw_audio_path = format_wav(speaker_raw_audio_path)
            self.logger.info(f"Speaker raw audio path: {speaker_raw_audio_path}")

            if enhance:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_audio_path = temp_file.name

                shutil.copy2(speaker_raw_audio_path, temp_audio_path)
                self.recorder.apply_nr(input_path=temp_audio_path)
                self.recorder.apply_vad(input_path=temp_audio_path, sampling_rate=16000, inplace=True)
                self.recognizer.register(temp_audio_path, speaker_raw_audio.split('.')[0])
                os.unlink(temp_audio_path)
            else:
                self.recognizer.register(speaker_raw_audio_path, speaker_raw_audio.split('.')[0])

    def _format_origin_audio_file(self, filename):
        """Format the origin audio file to 16kHz, 16-bit PCM WAV format.

        Args:
            filename: the name of the audio file to format

        Returns:
            The path to the formatted audio file
        """
        input_file_path = os.path.join(self.audio_origin_dir, filename)
        output_file_path = os.path.join(self.audio_formatted_dir, f'{os.path.splitext(filename)[0]}_formatted.wav')
        format_wav(input_file_path, output_file_path)
        properties = get_audio_properties(output_file_path)
        for key, value in properties.items():
            print(f'{key}: {value}')

        return output_file_path

    def _segment_audio_file(self, session_name, formatted_audio_file_path):
        """Segment the formatted audio file into fixed-duration segments.

        Args:
            session_name: the name of the session
            formatted_audio_file_path: the path to the formatted audio file
        """
        output_dir = os.path.join(self.audio_segments_dir, session_name)
        segment_wav(formatted_audio_file_path, output_dir, window_length_ms=int(self.segment_duration * 1000))

    def _process_segments(self, session_name):
        """Process segments of the audio file.

        Args:
            session_name: the name of the session

        Outputs:
            JSON files containing the speaker recognition and transcription results for the session
        """
        segments_dir = os.path.join(self.audio_segments_dir, session_name)
        speaker_recognition_log_path = os.path.join(self.logs_dir, f'session_{session_name}',
                                                    f'session_{session_name}_speaker_recognition.json')
        speaker_transcription_log_path = os.path.join(self.logs_dir, f'session_{session_name}',
                                                      f'session_{session_name}_speaker_transcription.json')
        visualization_dir = os.path.join(self.visualizations_dir, f'session_{session_name}')
        os.makedirs(visualization_dir, exist_ok=True)

        segments_path_list = sorted(
            [os.path.join(segments_dir, file) for file in os.listdir(segments_dir)],
            key=lambda x: int(os.path.basename(x).split('_')[-1][:-4])
        )

        start_time = datetime.now()  # starting datetime object of the conversation
        time = timedelta(seconds=0)
        segment_no = 0
        speaker_recognition_log_entries = []

        with tqdm(total=len(segments_path_list), desc=f"Processing segments for {session_name}",
                  unit="segment", position=0, leave=True) as pbar:
            for segment_path in segments_path_list:
                assert isinstance(segment_path, str), "segment_path must be a string"
                if segment_path.endswith('.DS_Store'):
                    continue

                processed_segment_path = self.recorder.post_processing(segment_path, sampling_rate=16000, inplace=True)

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
        formatted_audio_file_path = os.path.join(self.audio_formatted_dir, f'{session_name}_formatted.wav')
        chunk_list = self._aggregate_segments_by_speaker(speaker_recognition_log_path)
        borders = self.identify_speaker_change_borders(chunk_list)  # A list of tuple, (border_time, [candidates])
        added_chunks_no = 0
        for index, (border_time, candidates) in enumerate(borders):
            adjusted_index = index + added_chunks_no
            left_temp_path = os.path.join(self.audio_temp_dir, 'left_temp_post_analyzer.wav')
            right_temp_path = os.path.join(self.audio_temp_dir, 'right_temp_post_analyzer.wav')
            left_start_time = max(0, 1000 * (border_time - self.segment_duration / 2))  # Segment before the border
            right_end_time = 1000 * (border_time + self.segment_duration / 2)  # Segment after the border
            crop_and_concatenate_wav(formatted_audio_file_path, [(left_start_time, 1000 * border_time)], left_temp_path)
            crop_and_concatenate_wav(formatted_audio_file_path, [(1000 * border_time, right_end_time)], right_temp_path)

            if self.recorder.apply_vad(left_temp_path, sampling_rate=16000, inplace=False):
                left_speaker, _ = self.recognizer.recognize_among_candidates(left_temp_path, candidates,
                                                                             candidates[0], self.keep_threshold)
            else:
                left_speaker = 'silent'

            if self.recorder.apply_vad(right_temp_path, sampling_rate=16000, inplace=False):
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
        speaker_transcription_log_entries = self._transcribe_by_chunks(formatted_audio_file_path, chunk_list,
                                                                       session_name)

        with open(speaker_transcription_log_path, 'w') as f:
            json.dump(speaker_transcription_log_entries, f, indent=5)

        convert_transcription_json_to_txt(speaker_transcription_log_path)

    def _process_segments_sp(self, session_name):
        """Process segments of the audio file with using the separation model.

        Args:
            session_name: the name of the session

        Outputs:
            JSON files containing the speaker recognition and transcription results for the session
        """
        speaker_recognition_log_path = os.path.join(self.logs_dir, f'{session_name}',
                                                    f'session_{session_name}_speaker_recognition.json')
        speaker_transcription_log_path = os.path.join(self.logs_dir, f'{session_name}',
                                                      f'session_{session_name}_speaker_transcription.json')
        segments_dir = os.path.join(self.audio_segments_dir, session_name)
        visualization_dir = os.path.join(self.visualizations_dir, f'session_{session_name}')
        os.makedirs(visualization_dir, exist_ok=True)

        segments_path_list = sorted(
            [os.path.join(segments_dir, file) for file in os.listdir(segments_dir)],
            key=lambda x: int(os.path.basename(x).split('_')[-1][:-4])
        )

        start_time = datetime.now()  # starting datetime object of the conversation
        time = timedelta(seconds=0)
        segment_no = 0
        speaker_recognition_log_entries = []

        with tqdm(total=len(segments_path_list), desc=f"Processing segments for {session_name}",
                  unit="segment", position=0, leave=True) as pbar:
            for segment_path in segments_path_list:
                assert isinstance(segment_path, str), "segment_path must be a string"
                if segment_path.endswith('.DS_Store'):
                    continue
                time += timedelta(seconds=self.segment_duration)
                segment_start_time = int((start_time + time).timestamp())
                speakers, similarities, durations = [], [], []
                processed_segment_path = self.recorder.post_processing(segment_path, sampling_rate=16000, inplace=True)

                if processed_segment_path:
                    resample_audio_file(segment_path, 8000)
                    result = self.separator(segment_path)

                    for i, signal in enumerate(result['output_pcm_list']):
                        save_file = f'{segment_path[:-4]}_spk{i}.wav'
                        sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)

                        # speaker recognition on separated signals
                        processed_save_file = self.recorder.apply_vad(save_file, sampling_rate=8000, inplace=False)
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

                final_speakers, final_similarities, final_durations = self._post_process_sp_results(speakers,
                                                                                                    similarities,
                                                                                                    durations)
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
        formatted_audio_file_path = os.path.join(self.audio_formatted_dir, f'{session_name}_formatted.wav')
        speaker_transcription_log_entries = self._transcribe_by_chunks(formatted_audio_file_path, chunk_list,
                                                                       session_name)

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

    def _transcribe_by_chunks(self, audio_path, chunk_list, session_name):
        """Transcribe the audio by chunk.

        Args:
            audio_path: formatted audio file path
            chunk_list: chunk list containing speaker and start and end times
            session_name: the name of the session

        Returns:
            A list of transcription entries
        """
        chunk_dir = os.path.join(self.audio_chunks_dir, session_name)
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir)

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
                    chunk_path = os.path.join(chunk_dir, f'chunk_{index}_{speaker}.wav')
                    crop_and_concatenate_wav(audio_path, [(start_offset_ms, end_offset_ms)], chunk_path)
                    self.recorder.apply_nr(chunk_path)
                    normalize_decibel(chunk_path, rms_level=-20)
                    text = self.transcriber.transcribe(chunk_path) if speaker != 'silent' else ''
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

    def _post_process_sp_results(self, speakers, similarities, durations):
        """Post-process the recognition results of separated signals to handle edge cases.

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

        Args:
            chunk_list: A list of tuples, each containing the speaker and the start and end times of the segment

        Returns:
            A list of tuples, each containing the time of the speaker change and the candidates
        """
        # Note: this function only applicable to normal processing without speech separation
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
