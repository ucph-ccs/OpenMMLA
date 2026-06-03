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

from openmmla.analytics.asr.analyze import plot_speaking_interaction_network, \
    plot_speaker_diarization_interactive
from openmmla.analytics.asr.transcription import convert_transcription_json_to_txt
from openmmla.bases.base import Base
from openmmla.services.asr.requests import request_voice_activity_detection, request_speech_enhancement, \
    request_speech_separation, request_speech_transcription
from openmmla.utils.audio.auga import normalize_decibel
from openmmla.utils.audio.augf import resample_audio
from openmmla.utils.audio.files import format_wav, segment_wav, crop_and_concatenate_wav
from openmmla.utils.audio.properties import get_audio_properties
from openmmla.utils.artifact_paths import shared_pipeline_artifact_dir
from openmmla.utils.input import get_interactive_files, PURPLE, GREEN, GREY, ENDC, show_error_and_pause
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import resolve_url
from .audio_recognizer import AudioRecognizer
from .input import get_function_post


class ASRPostAnalyzer(Base):
    """ASRPostAnalyzer class for analyzing audio files with automatic speech recognition and speaker diarization."""
    logger = get_logger('asr-post-analyzer')

    def __init__(self, project_dir: str | None = None, config_path: str | None = None,
                 vad: bool = True, nr: bool = True, sp: bool = False, tr: bool = True):
        """Initialize the ASRPostAnalyzer class.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            vad: whether to use the VAD or not (default: True)
            nr: whether to use the denoiser to enhance speech or not (default: True)
            sp: whether to use the separation model or not (default: False)
            tr: whether to transcribe the audio segments or not (default: True)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)
        self.base_type = 'PostAnalyzer'

        self.vad = vad
        self.nr = nr
        self.sp = sp
        self.tr = tr
        self.selected_files: list[str] = []
        self.selected_speaker_files: list[str] = []

        self.segment_duration: int = 0
        self.threshold: float = 0.0
        self.keep_threshold: float = 0.0
        self.speech_transcriber_url: str = ''
        self.speech_separator_url: str = ''
        self.speech_enhancer_url: str = ''
        self.vad_url: str = ''

        self.origin_dir: str = ''
        self.runtime_dir: str = ''
        self.temp_dir: str = ''
        self.logs_dir: str = ''
        self.logger_dir: str = ''
        self.visualizations_dir: str = ''
        self.recognizer = None

        self.filename: str = ''
        self.session_name: str = ''  # session_name: filename without extension
        self.session_logs_dir: str = ''
        self.session_runtime_dir: str = ''
        self.session_segments_dir: str = ''
        self.session_chunks_dir: str = ''

        self._setup_yaml()
        self._setup_directories()

    def _setup_yaml(self):
        """Load and assign configuration parameters from the YAML configuration file."""
        post_analyzer_config = self.config['PostAnalyzer']
        asr_server_config = self.config['Server']['asr']
        
        # Load processing parameters
        self.segment_duration = int(post_analyzer_config['segment_duration'])
        self.threshold = float(post_analyzer_config['threshold'])
        self.keep_threshold = float(post_analyzer_config['keep_threshold'])
        
        # Load server URLs
        self.speech_transcriber_url = resolve_url(asr_server_config['speech_transcriber'])
        self.speech_separator_url = resolve_url(asr_server_config['speech_separator'])
        self.speech_enhancer_url = resolve_url(asr_server_config['speech_enhancer'])
        self.vad_url = resolve_url(asr_server_config['voice_activity_detector'])

    def _setup_directories(self):
        """Set up the directory structure for the ASRPostAnalyzer."""
        self.runtime_dir = os.path.join(self.project_dir, 'post-time', 'runtime')
        self.origin_dir = os.path.join(self.project_dir, 'post-time', 'origin')
        self.temp_dir = os.path.join(self.project_dir, 'post-time', 'temp')
        self.logs_dir = os.path.join(self.project_dir, 'logs')
        self.logger_dir = os.fspath(shared_pipeline_artifact_dir(self.project_dir, 'asr-post', 'logger'))
        self.visualizations_dir = os.path.join(self.project_dir, 'visualizations')

        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.origin_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

    def _setup_objects(self):
        """Initialize the AudioRecognizer object."""
        if self.selected_files:
            first_file_name = os.path.splitext(os.path.basename(self.selected_files[0]))[0]
            profiles_dir = os.path.join(self.runtime_dir, f'session_{first_file_name}', 'profiles')
            os.makedirs(profiles_dir, exist_ok=True)
            self.recognizer = AudioRecognizer(config_path=self.config_path, profiles_dir=profiles_dir)

    def run(self):
        """Run the ASR Post Analyzer with interactive menu."""
        print('\033]0;ASR Post Analyzer\007')
        func_map = {1: self._select_speaker_profiles, 2: self._select_files, 3: self._start_processing}

        while True:
            try:
                select_fun = get_function_post(self.selected_speaker_files, self.selected_files)
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except KeyboardInterrupt as e:
                if "Exit" in str(e):
                    # 'q' was pressed in top-level menu - re-raise to be caught by outer restart loop
                    raise
                else:
                    # ctrl+c during runtime or 'q' in lower-level menu - log and continue
                    self.logger.warning(f"During running the ASR post analyzer, catch: {e}, Come back to main menu.", exc_info=True)
            except Exception as e:
                self.logger.warning(f"During running the ASR post analyzer, catch: {e}, Come back to the main menu.", exc_info=True)
                show_error_and_pause(e, "return to the ASR Post Analyzer menu")
            finally:
                self._clean_up()

    def _clean_up(self):
        """Clean up runtime variables and free memory."""
        pass

    def _select_speaker_profiles(self):
        """Select speaker profile files."""
        print(f"\n{PURPLE}🎤 Select Speaker Profile Files{ENDC}")
        print(f"{GREY}Choose speaker audio files for recognition (e.g., speaker1.wav, speaker2.wav){ENDC}")
        
        try:
            audio_extensions = ('.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.wma')
            selected_speaker_files = get_interactive_files(self.project_dir, file_extensions=audio_extensions)
            if not selected_speaker_files:
                self.logger.warning("No speaker files selected.")
                return
            
            self.selected_speaker_files = selected_speaker_files

            # validate speaker files are in proper audio format
            supported_formats = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
            for speaker_file in self.selected_speaker_files:
                if not speaker_file.lower().endswith(supported_formats):
                    self.selected_speaker_files.remove(speaker_file)
            self.logger.info(f"Selected {len(selected_speaker_files)} speaker files: {selected_speaker_files}")

        except Exception as e:
            self.logger.error(f"Error selecting speaker profiles: {e}")

    def _select_files(self):
        """Select audio files to process."""
        print(f"\n{PURPLE}📁 Select Audio Files{ENDC}")
        print(f"{GREY}Choose audio files to analyze (multiple selection supported){ENDC}")
        
        try:
            audio_extensions = ('.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.wma')
            selected_files = get_interactive_files(self.project_dir, file_extensions=audio_extensions)
            if not selected_files:
                self.logger.warning("No files selected.")
                return
            
            self.selected_files = selected_files
            self.logger.info(f"Selected {len(selected_files)} files: {selected_files}")
            # Initialize recognizer
            self._setup_objects()
                
        except Exception as e:
            self.logger.error(f"Error selecting files: {e}")

    def _start_processing(self):
        """Start processing the selected files."""
        if not self.selected_speaker_files:
            self.logger.warning("Please select speaker profile files first.")
            return
        
        if not self.selected_files:
            self.logger.warning("Please select audio files first.")
            return
        
        self.logger.info("Starting ASR Post Analysis...")
        self.logger.info(f"Speaker profile files: {self.selected_speaker_files}")
        self.logger.info(f"Files to process: {self.selected_files}")
        
        # Process each selected file
        for audio_file in tqdm(self.selected_files, desc='Processing audio files', unit='session'):
            filename = os.path.basename(audio_file)
            self._create_bucket_logger(filename)
            self.logger.info(f"Processing file: {audio_file}")
            self._process_single_audio_file(audio_file)


    def _create_bucket_logger(self, filename: str):
        """Create a logger for a single audio file.
        
        Args:
            filename: the name of the audio file to process
        """
        self.file_logger_dir = os.path.join(self.logger_dir, f'session_{filename}')
        os.makedirs(self.file_logger_dir, exist_ok=True)
        self.logger = get_logger(f'asr-post-{filename}',
                                 os.path.join(self.file_logger_dir,
                                              f'asr_post_{filename}.log'))

    def _process_single_audio_file(self, audio_file_path: str):
        """Process a single audio file.

        Args:
            audio_file_path: the full path to the audio file to process
        """
        self.filename = os.path.basename(audio_file_path)
        self.session_name = os.path.splitext(self.filename)[0]
        self.session_logs_dir = os.path.join(self.logs_dir, f'session_{self.session_name}')
        self.session_runtime_dir = os.path.join(self.runtime_dir, f'session_{self.session_name}')
        self.session_segments_dir = os.path.join(self.session_runtime_dir, 'segments')
        self.session_chunks_dir = os.path.join(self.session_runtime_dir, 'chunks')
        self.session_profiles_dir = os.path.join(self.session_runtime_dir, 'profiles')

        for directory in [self.session_logs_dir, self.session_runtime_dir, self.session_segments_dir,
                          self.session_chunks_dir, self.session_profiles_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        self.recognizer.reset_profiles(self.session_profiles_dir)

        # register speakers with NR and VAD enhanced
        self._register_speakers(self.selected_speaker_files, enhance=True)

        # format the audio file, segment it, and process the segments
        self._format_runtime_audio(audio_file_path)
        self._segment_formatted_file()
        if self.sp:
            self._process_segments_sp()
        else:
            self._process_segments()

    def _register_speakers(self, speaker_files: list[str], enhance: bool = True):
        """Register speakers' raw audio files to the recognizer.

        Args:
            speaker_files: list of paths to the raw audio files of the speakers
            enhance: whether to apply NR and VAD to the audio files or not
        """
        for speaker_raw_filepath in speaker_files:
            speaker_raw_filename = os.path.basename(speaker_raw_filepath)
            speaker_name = os.path.splitext(speaker_raw_filename)[0]
            # the foramtted path should be in the same directory as the raw file with the same name but different extension
            formatted_speaker_filepath = os.path.join(os.path.dirname(speaker_raw_filepath), f'{speaker_name}.wav')
            format_wav(speaker_raw_filepath, formatted_speaker_filepath )

            speaker_profiles_dir = os.path.join(self.recognizer.profiles_dir, speaker_name)
            if os.path.exists(speaker_profiles_dir):
                shutil.rmtree(speaker_profiles_dir)

            if enhance:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_audio_path = temp_file.name
                shutil.copy2(formatted_speaker_filepath, temp_audio_path)
                self._audio_preprocessing(temp_audio_path, inplace=1)
                self.recognizer.register(temp_audio_path, speaker_name)
                os.unlink(temp_audio_path)
            else:
                self.recognizer.register(formatted_speaker_filepath, speaker_name)

    def _format_runtime_audio(self, audio_file_path: str):
        """Format the audio file to 16kHz, 16-bit PCM WAV format."""
        # Copy the original audio file to the origin directory
        origin_audio_path = os.path.join(self.origin_dir, f'{self.session_name}{os.path.splitext(audio_file_path)[1]}')
        shutil.copy2(audio_file_path, origin_audio_path)
        
        # Format the audio file to 16kHz, 16-bit PCM WAV format
        formatted_audio_path = os.path.join(self.session_runtime_dir, f'{self.session_name}.wav')
        format_wav(audio_file_path, formatted_audio_path)
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
                segment_start_time = float((start_time + time).timestamp())
                recognition_entry = {
                    "segment_no": segment_no,
                    "window_start_time": segment_start_time,
                    "window_end_time": segment_start_time + self.segment_duration,
                    "speakers": [speaker],
                    "similarities": [np.round(np.float64(similarity), 4)],
                    "durations": [duration],
                }
                speaker_recognition_log_entries.append(recognition_entry)
                segment_no += 1
                pbar.update()

        # Write all speaker_recognition_log_entries to the JSON file at once
        with open(speaker_recognition_log_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_recognition_log_entries, f, indent=5, ensure_ascii=False)

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

        with open(speaker_transcription_log_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_transcription_log_entries, f, indent=5, ensure_ascii=False)

        convert_transcription_json_to_txt(speaker_transcription_log_path)

    def _process_segments_sp(self):
        """Process segments of the audio file with speech separation.

        Outputs:
            JSON files containing the speaker recognition and transcription results for the audio file
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
                segment_start_time = float((start_time + time).timestamp())
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
        with open(speaker_recognition_log_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_recognition_log_entries, f, indent=5, ensure_ascii=False)

        # Visualize the speaker recognition results
        plot_speaking_interaction_network(speaker_recognition_log_path, visualization_dir)
        plot_speaker_diarization_interactive(speaker_recognition_log_path, visualization_dir)

        # Aggregate segments into chunks and transcribe them
        chunk_list = self._aggregate_segments_by_speaker(speaker_recognition_log_path)
        speaker_transcription_log_entries = self._transcribe_by_chunks(chunk_list)

        with open(speaker_transcription_log_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_transcription_log_entries, f, indent=5, ensure_ascii=False)

        convert_transcription_json_to_txt(speaker_transcription_log_path)

    def _aggregate_segments_by_speaker(self, speaker_recognition_log_path: str) -> list[tuple[str, tuple[float, float]]]:
        """Aggregate segments by speaker.

        Args:
            speaker_recognition_log_path: path to the speaker recognition log file

        Returns:
            A list of tuples, each containing the speaker and the start and end times of the chunk
        """
        with open(speaker_recognition_log_path, 'r') as file:
            speaker_recognition_log = json.load(file)

        speakers_segments = {}  # Dictionary to hold the current speaker segments
        chunk_list = []
        for entry in speaker_recognition_log:
            segment_start_time = entry['window_start_time']
            segment_end_time = segment_start_time + self.segment_duration
            speakers = entry['speakers']

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
            last_segment_start_time = speaker_recognition_log[-1]['window_start_time']
            last_segment_end_time = last_segment_start_time + self.segment_duration
            for speaker, times in speakers_segments.items():
                times[1] = last_segment_end_time
                chunk_list.append((speaker, times))

        return chunk_list

    def _transcribe_by_chunks(self, chunk_list: list[tuple[str, tuple[float, float]]]) -> list[dict]:
        """Transcribe the audio by chunk.

        Args:
            chunk_list: a list of tuples, each containing the speaker and the start and end times of the chunk

        Returns:
            A list of dictionaries, each containing the transcription results for a chunk
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

                    transcribe_result = self._transcribe(chunk_path) if speaker != 'silent' else None
                    transcribe_result = {} if transcribe_result is None else transcribe_result

                    transcription_entry = {
                        "chunk_no": index,
                        "window_start_time": chunk_start_time,
                        "window_end_time": chunk_end_time,
                        "speaker": speaker,
                        "text": transcribe_result.get("text", ""),
                        "words": transcribe_result.get("words", []),
                    }
                    print(transcription_entry)
                    entries.append(transcription_entry)
                    pbar.update()

        return entries

    def _transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: path to the audio file to transcribe

        Returns:
            transcribed text
        """
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            frame_rate = wav_file.getframerate()

        text = request_speech_transcription(frames, frame_rate, f'{self.base_type.lower()}', self.speech_transcriber_url)
        return text

    def _separate_speech(self, audio_path: str) -> list:
        """Separate speech from the overlapped segment audio.

        Args:
            audio_path: path to the audio file to separate speech from

        Returns:
            separated speech signals
        """
        separated_result = request_speech_separation(audio_path, f'{self.base_type.lower()}',
                                                     self.speech_separator_url)
        result = [base64.b64decode(encoded_bytes_stream) for encoded_bytes_stream in separated_result]
        return result

    def _audio_preprocessing(self, audio_path: str, inplace: int) -> str | None:
        """Apply both NR and VAD processing to audio file.


        Args:
            audio_path: path to the audio file to process
            inplace: whether to overwrite the input file when applying vad

        Returns:
            processed audio file path
        """
        self._apply_nr(audio_path)
        return self._apply_vad(audio_path, inplace)

    def _apply_vad(self, audio_path: str, inplace: int) -> str | None:
        """Apply voice activity detection to audio file.

        Args:
            audio_path: path to the audio file to process
            inplace: whether to overwrite the input file

        Returns:
            processed audio file path
        """
        if self.vad:
            return request_voice_activity_detection(audio_path, f'{self.base_type.lower()}', inplace, self.vad_url)
        return audio_path

    def _apply_nr(self, audio_path: str) -> str:
        """Apply noise reduction to audio file.

        Args:
            audio_path: path to the audio file to process

        Returns:
            processed audio file path
        """
        if self.nr:
            request_speech_enhancement(audio_path, f'{self.base_type.lower()}', self.speech_enhancer_url)
        return audio_path

    def _finalize_separated_speaker_recognition(self, speakers: list[str], similarities: list[float], durations: list[float]) -> tuple[list[str], list[float], list[float]]:
        """Finalize the speaker recognition results of separated signals to handle edge cases.

        Args:
            speakers: list of recognized speakers
            similarities: list of speaker recognition similarities
            durations: list of speaking durations in seconds

        Returns:
            A tuple containing the finalized speakers, similarities, and durations
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
    def filter_real_speakers(speakers: list[str], similarities: list[float], durations: list[float]) -> tuple[list[str], list[float], list[float]]:
        """Filter out 'silent' and 'unknown' from speakers and their associated similarities and durations.

        Args:
            speakers: list of recognized speakers
            similarities: list of speaker recognition similarities
            durations: list of speaking durations in seconds

        Returns:
            A tuple containing the filtered speakers, similarities, and durations
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

    def speaker_recognition_results(self, segment_no: int, segment_start_time: float, final_speakers: list[str],
                                    final_similarities: list[float], final_durations: list[float]) -> dict:
        """Create a speaker recognition log entry.

        Args:
            segment_no: segment number
            segment_start_time: segment start time
            final_speakers: finalized recognized speakers list
            final_similarities: finalized speaker recognition similarities list
            final_durations: finalized speaking durations list

        Returns:
            A dictionary containing the speaker recognition results
        """
        recognition_entry = {
            "segment_no": segment_no,
            "window_start_time": segment_start_time,
            "window_end_time": segment_start_time + self.segment_duration,
            "speakers": json.dumps(final_speakers),
            "similarities": json.dumps(final_similarities),
            "durations": json.dumps(final_durations),
        }
        return recognition_entry

    @staticmethod
    def identify_speaker_change_borders(chunk_list: list[tuple[str, tuple[float, float]]]) -> list[tuple[float, list[str]]]:
        """Identify speaker change borders from the chunk list.

        Note: this function only applicable to recognition without speech separation

        Args:
            chunk_list: a list of tuples, each containing the speaker and the start and end times of the chunk

        Returns:
            A list of tuples, each containing the time of the speaker change and the candidates for the speaker change
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
    def update_chunk_list(chunk_list: list[tuple[str, tuple[float, float]]], border_index: int, left_speaker: str, right_speaker: str, segment_half_duration: float) -> int:
        """Update the chunk list by adding two new half-scaled recognized chunks at the speaker change border.

        Note: this function only applicable to recognition without speech separation

        Args:
            chunk_list: a list of tuples, each containing the speaker and the start and end times of the chunk
            border_index: the index of the speaker change border
            left_speaker: the speaker on the left side of the border
            right_speaker: the speaker on the right side of the border
            segment_half_duration: half of the segment duration in seconds

        Returns:
            The number of added chunks (2)
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
    def aggregate_chunks(chunk_list: list[tuple[str, tuple[float, float]]]) -> list[tuple[str, tuple[float, float]]]:
        """Aggregate the chunks by combining consecutive chunks with the same speaker.

        Args:
            chunk_list: a list of tuples, each containing the speaker and the start and end times of the chunk

        Returns:
            A list of tuples, each containing the speaker and the start and end times of the aggregated chunk
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
    def calculate_audio_duration(audio_path: str) -> float:
        """Calculate the duration of an audio file in seconds.

        Args:
            audio_path: path to the audio file

        Returns:
            The duration of the audio file in seconds
        """
        with sf.SoundFile(audio_path) as f:
            return len(f) / f.samplerate


def start_asr_post_analyzer(project_dir: str, config_path: str, vad: bool = True, nr: bool = True, 
                           sp: bool = False, tr: bool = True):
    """Start ASR Post Analyzer with restart capability."""
    while True:
        try:
            post_analyzer = ASRPostAnalyzer(project_dir=project_dir, config_path=config_path,
                                           vad=vad, nr=nr, sp=sp, tr=tr)
            post_analyzer.run()
        except KeyboardInterrupt as e:
            if "Exit" in str(e):
                print("\n👋 Goodbye!")
                break  # Exit completely when 'q' is pressed
            else:
                print("\n🔄 Restarting ASR Post Analyzer...")
                continue  # Restart on Ctrl+C during runtime
        except Exception as e:
            show_error_and_pause(e, "restart ASR Post Analyzer")
            print("\n🔄 Restarting ASR Post Analyzer...")
            continue
