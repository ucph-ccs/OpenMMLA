import base64
import gc
import json
import os
import queue
import shutil
import threading
import time
from abc import abstractmethod, ABC

import librosa
import numpy as np

from openmmla.bases.asr_diarization.errors import RecordingError, TranscribingError
from openmmla.bases.base import Base
from openmmla.services.asr_diarization.requests import request_speech_transcription, request_speech_separation, \
    request_speech_enhancement, request_voice_activity_detection
from openmmla.utils.audio.io import read_bytes_from_wav
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import resolve_url
from .audio_recognizer import AudioRecognizer
from .enums import BLUE, ENDC, GREEN
from .input import get_function_base, get_id

try:
    from openmmla.utils.audio.transcriber import get_transcriber
except ImportError:
    get_transcriber = None

try:
    import torch
except ImportError:
    torch = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    from denoiser import pretrained
except ImportError:
    pretrained = None

try:
    from denoiser.dsp import convert_audio
except ImportError:
    convert_audio = None

try:
    from silero_vad import load_silero_vad, read_audio, save_audio, get_speech_timestamps, collect_chunks
except ImportError:
    load_silero_vad = None
    read_audio = None
    get_speech_timestamps = None
    save_audio = None
    collect_chunks = None

try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
except ImportError:
    pipeline, Tasks = None, None


class AudioBase(Base, ABC):
    """An abstract base class for audio processing pipelines that handles speaker recognition, transcription, and audio management.

    This class provides a comprehensive framework for real-time audio processing which includes:
      - Speaker profile registration and management.
      - Speech transcription and enhancement.
      - Voice activity detection (VAD) and noise reduction (NR).
      - Multi-speaker separation for overlapped speech.

    Key Features:
      - Multiple operating modes: 'record', 'recognize', and 'full'.
      - Configurable pipeline components (VAD, NR, transcription, separation).
      - Persistent storage management for audio files and speaker profiles.
      - Concurrent audio processing using threads.
      - Integration with external services via MQTT, Redis, and InfluxDB.
      - Support for both real-time and pre-recorded audio processing.

    Concrete implementations must provide the following abstract methods:
      - _start_register: Handle speaker profile registration.
      - _start_recognize: Handle real-time voice recognition.
      - _reset: Reset or reinitialize the audio base.
      - _switch_mode: Switch the operating mode of the audio base.
    """

    logger = get_logger(f'audio-base')

    def __init__(self, project_dir: str, config_path: str, mode: str = 'full', store: bool = True, vad: bool = True,
                 nr: bool = True, tr: bool = True, sp: bool = False):
        """Initialize the audio processing pipeline base class.

        Sets up the configuration, directories, and required objects for audio processing.

        Args:
            project_dir (str): path to the project directory.
            config_path (str): path to the configuration file (absolute or relative to project_dir).
            mode (str): operating mode ('record', 'recognize', or 'full'). Defaults to 'full'.
            store (bool): whether to store audio files. Defaults to True.
            vad (bool): whether to apply Voice Activity Detection. Defaults to True.
            nr (bool): whether to apply noise reduction. Defaults to True.
            tr (bool): whether to transcribe speech to text. Defaults to True.
            sp (bool): whether to perform speech separation. Defaults to False.
        """
        super().__init__(project_dir, config_path)

        # Audio base specific parameters.
        self.mode = mode
        self.store = store
        self.vad = vad
        self.nr = nr
        self.tr = tr
        self.sp = sp

        # Runtime attributes.
        self.bucket_name = None
        self.last_speaker = None
        self.audio_dir = None
        self.audio_queue = None
        self.transcription_queue = None
        self.speaker_frames_dict = None
        self.threads = []
        self.stop_event = threading.Event()
        self.cuda_enable = torch is not None and torch.cuda.is_available()

        self._setup_yaml()
        self._setup_input()
        self._setup_directories()
        self._setup_objects()

    @property
    @abstractmethod
    def base_type(self):
        """Abstract property that returns the type of the audio base.

        Returns:
            str: The audio base type.
        """
        pass

    def _setup_input(self):
        """Setup the input identifier for the audio base.

        Retrieves and assigns a unique identifier for this instance.
        """
        self.id = get_id()

    def _setup_yaml(self):
        """Load and assign configuration parameters from the YAML configuration file.

        Reads various settings such as durations, thresholds, and service URLs required for the audio processing pipeline.
        """
        self.register_duration = int(self.config[self.base_type]['register_duration'])
        self.recognize_duration = int(self.config[self.base_type]['recognize_sp_duration']) if self.sp else int(
            self.config[self.base_type]['recognize_duration'])
        self.rms_threshold = int(self.config[self.base_type]['rms_threshold'])
        self.rms_peak_threshold = int(self.config[self.base_type]['rms_peak_threshold'])
        self.threshold = float(self.config[self.base_type]['recognize_sp_threshold']) if self.sp else float(
            self.config[self.base_type]['recognize_threshold'])
        self.keep_threshold = float(self.config[self.base_type]['keep_sp_threshold']) if self.sp else float(
            self.config[self.base_type]['keep_threshold'])
        self.speech_transcriber_url = resolve_url(self.config['Server']['asr']['speech_transcription'])
        self.speech_separator_url = resolve_url(self.config['Server']['asr']['speech_separation'])
        self.speech_enhancer_url = resolve_url(self.config['Server']['asr']['speech_enhancement'])
        self.vad_url = resolve_url(self.config['Server']['asr']['voice_activity_detection'])

    def _setup_directories(self):
        """Create and set up the necessary directories for runtime operations.

        Creates directories for runtime files, temporary files, speaker profiles, and audio databases.
        Ensures that the required folder structure exists.
        """
        self.runtime_dir = os.path.join(self.project_dir, 'real-time', 'runtime')
        self.temp_dir = os.path.join(self.project_dir, 'real-time', 'temp')
        self.profiles_dir = os.path.join(self.project_dir, 'real-time', 'profiles')
        self.audio_db = os.path.join(self.profiles_dir, f'{self.base_type}_{self.id}')
        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.audio_db, exist_ok=True)

    def _setup_objects(self):
        """Initialize external service clients and internal processing objects.

        Sets up clients for InfluxDB, Redis, and MQTT, warms up the audio resampler, and initializes the AudioRecognizer.
        """
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.warm_up_resampler()
        self.audio_recognizer = AudioRecognizer(config_path=self.config_path, audio_db=self.audio_db, store=self.store)

    def run(self):
        """Start the audio processing loop.

        Provides an interactive interface to:
          1. Register speaker profiles.
          2. Start real-time voice recognition.
          3. Reset or switch the audio base mode.

        Continuously prompts the user for input until termination.
        """
        func_map = {1: self._start_register, 2: self._start_recognize, 3: self._reset, 4: self._switch_mode}
        while True:
            try:
                print(f"\033]0;Audio Base {self.base_type} {self.id} \007")
                select_fun = get_function_base()
                if select_fun == 0:
                    print("------------------------------------------------")
                    clear_directory(self.temp_dir)
                    self.logger.info("Exiting the program...")
                    break
                func_map.get(select_fun, lambda: self.logger.warning("Invalid option"))()
            except (Exception, KeyboardInterrupt) as e:
                self._clean_up()
                self.logger.warning(
                    f"During running the audio base, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)

    @abstractmethod
    def _start_register(self):
        """Abstract method to start the speaker profile registration process.

        Implementations should handle recording, processing, and saving speaker profiles.
        """
        pass

    @abstractmethod
    def _start_recognize(self):
        """Abstract method to start real-time voice recognition.

        Implementations should handle audio recording, speaker recognition, speech transcription,
        and listening for a stop signal.
        """
        pass

    @abstractmethod
    def _reset(self):
        """Abstract method to reset the audio base.

        Implementations should reinitialize the audio base, for example by setting a new port.
        """
        pass

    @abstractmethod
    def _switch_mode(self):
        """Abstract method to switch the operating mode of the audio base.

        Implementations should handle switching between modes (e.g., from 'record' to 'recognize').
        """
        pass

    def _clean_up(self):
        """Clean up runtime variables and free memory.

        Resets runtime attributes and calls garbage collection to free memory.
        """
        self.bucket_name = None
        self.last_speaker = None
        self.audio_dir = None
        self.audio_queue = None
        self.transcription_queue = None
        self.speaker_frames_dict = None
        gc.collect()

    def _prepare_directories(self):
        """Prepare subdirectories for audio processing based on the operating mode.

        Creates subdirectories for segments, chunks, separations, temporary files, and records.
        Clears specific directories based on the current operating mode.
        """
        sub_dirs = ['segments', 'chunks', 'separations', 'temp', 'records']
        for subdir in sub_dirs:
            directory_path = os.path.join(self.audio_dir, subdir)
            os.makedirs(directory_path, exist_ok=True)

        if self.mode == 'recognize':
            for subdir in ['segments', 'chunks', 'separations']:
                clear_directory(os.path.join(self.audio_dir, subdir))

        # Note: The following commented code may clear the records folder when in record mode.
        # if self.mode == 'record':
        #     clear_directory(os.path.join(self.audio_dir, 'records'))

        clear_directory(os.path.join(self.audio_dir, 'temp'))

    def _load_and_queue_recorded_files(self):
        """Load pre-recorded audio files and queue them for recognition.

        In 'recognize' mode, this method loads .wav files from the records directory,
        reads their content, and queues them for further processing.

        Raises:
            RecordingError: If an error occurs while loading or queuing the audio files.
        """
        while not self.stop_event.is_set():
            try:
                print(f"{GREEN}[Pre-recorded Audio]{ENDC}Loading pre-recorded audio files...")
                audio_files = [os.path.join(self.audio_dir, 'records', f) for f in
                               os.listdir(os.path.join(self.audio_dir, 'records'))
                               if f.endswith('.wav')]
                audio_files_sorted = sorted(audio_files, key=lambda x: os.path.getmtime(x))

                for file_path in audio_files_sorted:
                    frames = read_bytes_from_wav(file_path)
                    temp_file_path = os.path.join(self.audio_dir, 'temp', os.path.basename(file_path))
                    self.audio_queue.put((temp_file_path, frames))
                print(f"{GREEN}[Pre-recorded Audio]{ENDC}Pre-recorded audio files loaded and queued.")
                return
            except Exception as e:
                raise RecordingError(
                    f'RecordingError occurred when loading pre-recorded audio files and queue: {e}') from e

    def _update_chunk_list(self, left_speaker, right_speaker, last_speaker, current_speaker, chunk_frames, frames,
                           record_start_time) -> tuple:
        """Update the chunk list based on recognition results at a speaker turn boundary.

        Adjusts the audio chunks based on recognition outcomes from different segments,
        handling various cases such as extending the chunk to include additional audio.

        Args:
            left_speaker: Recognized speaker from the left half of the segment.
            right_speaker: Recognized speaker from the right half of the segment.
            last_speaker: Recognized speaker from the previous segment.
            current_speaker: Recognized speaker from the current segment.
            chunk_frames: Audio frames of the current chunk associated with the last speaker.
            frames: Audio frames of the current segment.
            record_start_time: Start time of the current audio segment.

        Returns:
            tuple: Updated chunk_frames, remaining frames, new record_start_time, and chunk_end_time.
        """
        num_frames = int(len(frames) / 2)

        if left_speaker == right_speaker and left_speaker == last_speaker:  # Case AAAB: Extend the left chunk
            chunk_frames = chunk_frames + frames[:num_frames]
            frames = frames[num_frames:]
            record_start_time = str(float(record_start_time) + self.recognize_duration / 2)
            chunk_end_time = record_start_time
        elif left_speaker == right_speaker and right_speaker == current_speaker:  # Case ABBB: Extend the right chunk
            frames = chunk_frames[-num_frames:] + frames
            chunk_frames = chunk_frames[:-num_frames]
            record_start_time = str(float(record_start_time) - self.recognize_duration / 2)
            chunk_end_time = record_start_time
        else:  # Other cases: AABB, A_BB, AA_B, A__B (excluding ABAB, AB_B, A_AB)
            chunk_end_time = record_start_time
            if left_speaker != last_speaker:
                chunk_frames = chunk_frames[:-num_frames]
                chunk_end_time = str(float(record_start_time) - self.recognize_duration / 2)
            if right_speaker != current_speaker:
                frames = frames[num_frames:]
                record_start_time = str(float(record_start_time) + self.recognize_duration / 2)

        return chunk_frames, frames, record_start_time, chunk_end_time

    def _add_to_transcription_queue(self, frames, speaker, chunk_start_time, chunk_end_time):
        """Add an audio chunk to the transcription queue.

        If transcription is enabled (self.tr), enqueues the audio frames along with speaker and timing details.

        Args:
            frames: Audio frames to be transcribed.
            speaker: Recognized speaker for the audio chunk.
            chunk_start_time: Start time of the audio chunk.
            chunk_end_time: End time of the audio chunk.
        """
        if self.tr:
            self.transcription_queue.put((frames, speaker, chunk_start_time, chunk_end_time))

    def _continuous_transcribing(self):
        """Continuously process audio chunks for transcription.

        Runs in a loop until a stop event is set. Retrieves audio frames from the transcription queue,
        transcribes them into text, and uploads the transcription.

        Raises:
            TranscribingError: If an error occurs during the transcription process.
        """
        frame_rate = 8000 if self.sp else 16000
        while not self.stop_event.is_set():
            try:
                frames, speaker, chunk_start_time, chunk_end_time = self.transcription_queue.get(timeout=2)
                text = self._transcribe(frames, frame_rate)
                self._upload_speech_transcription(speaker, text, chunk_start_time, chunk_end_time)
            except queue.Empty:
                continue
            except Exception as e:
                raise TranscribingError(f'TranscribingError occurred when transcribing: {e}') from e

    def _upload_speech_transcription(self, speaker, text, chunk_start_time, chunk_end_time):
        """Upload the transcribed speech chunk to the database.

        Constructs a transcription record and writes it to InfluxDB.
        Also prints the transcription for logging purposes.

        Args:
            speaker: Recognized speaker for the transcribed chunk.
            text: Transcribed text.
            chunk_start_time: Start time of the audio chunk.
            chunk_end_time: End time of the audio chunk.
        """
        transcription_record = {
            "measurement": "speaker transcription",
            "fields": {
                "chunk_start_time": float(chunk_start_time),
                "chunk_end_time": float(chunk_end_time),
                "text": text,
                "speaker": speaker,
            },
        }
        print(f"{GREEN}[Speaker Transcription]{ENDC}{transcription_record['fields']['chunk_start_time']}: "
              f"{GREEN}{speaker} : {text}{ENDC}")
        self.influx_client.write(self.bucket_name, record=transcription_record)

    def _consolidate_separated_recognition_results(self, speakers, similarities, durations, signals) -> tuple:
        """Consolidate speaker recognition results for separated audio signals to handle edge cases.

        Processes recognition results and returns a consolidated set of speakers along with
        their similarity scores, durations, and corresponding signals.

        Args:
            speakers (list): List of recognized speakers.
            similarities (list): List of similarity scores.
            durations (list): List of durations for each speaker.
            signals (list): List of separated audio signals.

        Returns:
            tuple: lists of speakers, similarities, durations, and signals.
        """
        # If there's only one speaker, keep it as is.
        if len(speakers) == 1:
            return speakers, similarities, durations, signals

        # If both speakers are the same, keep the one with the highest similarity.
        if speakers[0] == speakers[1]:
            max_similarity_index = 0 if similarities[0] > similarities[1] else 1
            return [speakers[max_similarity_index]], [similarities[max_similarity_index]], [
                durations[max_similarity_index]], [signals[max_similarity_index]]

        # If both are real speakers, keep them as is.
        if all(speaker not in ['silent', 'unknown'] for speaker in speakers):
            return speakers, similarities, durations, signals

        # If there's at least one real speaker, keep only the real ones.
        real_speakers, real_similarities, real_durations, real_texts = self.filter_real_speakers(speakers, similarities,
                                                                                                 durations, signals)
        if real_speakers:
            return real_speakers, real_similarities, real_durations, real_texts

        # Special cases for 'silent' and 'unknown'
        if 'unknown' in speakers and 'silent' in speakers:
            return (['unknown'], [similarities[speakers.index('unknown')]], [durations[speakers.index('unknown')]],
                    [signals[speakers.index('unknown')]])
        if speakers.count('silent') == 2:
            return ['silent'], [similarities[0]], [durations[0]], [signals[0]]
        if speakers.count('unknown') == 2:
            max_index = durations.index(max(durations))
            return ['unknown'], [similarities[max_index]], [durations[max_index]], [signals[max_index]]

        return [], [], [], []

    def _publish_recognition_results(self, record_start_time, recognize_start_time, speakers, similarities, durations):
        """Log and publish speaker recognition results via MQTT.

        Constructs a JSON record with recognition details and publishes it on the designated channel.

        Args:
            record_start_time: Start time of the recorded segment.
            recognize_start_time: Start time of the recognition process.
            speakers (list): List of recognized speakers.
            similarities (list): List of similarity scores.
            durations (list): List of audio durations.
        """
        base_recognition_result = {
            'base_id': f'{self.base_type.lower()}_{self.id}',
            'record_start_time': record_start_time,
            'speakers': json.dumps(speakers),
            'similarities': json.dumps(similarities),
            'durations': json.dumps(durations)
        }
        print(f"{BLUE}[Speaker Recognition]{ENDC}{base_recognition_result['record_start_time']}: "
              f"{BLUE}{base_recognition_result['speakers']}{ENDC}, similarity: {base_recognition_result['similarities']},"
              f"processed time: {time.time() - recognize_start_time} seconds")
        result_str = json.dumps(base_recognition_result)
        self.mqtt_client.publish(f'{self.bucket_name}/audio', result_str)

    def _transcribe(self, frames, frame_rate) -> str:
        """Transcribe audio frames to text using an external service.

        Args:
            frames: Audio frames to be transcribed.
            frame_rate: Sample rate of the audio frames.

        Returns:
            str: The transcribed text.
        """
        text = request_speech_transcription(frames, frame_rate, f'{self.base_type.lower()}_{self.id}',
                                            self.speech_transcriber_url)
        return text

    def _separate_speech(self, segment_audio_path) -> list:
        """Separate overlapping speech from an audio segment.

        Uses an external speech separation service to process the audio file and decodes the separated signals.

        Args:
            segment_audio_path (str): Path to the audio segment file.

        Returns:
            list: A list of separated speech signals (decoded from base64).
        """
        separated_result = request_speech_separation(segment_audio_path, f'{self.base_type.lower()}_{self.id}',
                                                     self.speech_separator_url)
        result = [base64.b64decode(encoded_bytes_stream) for encoded_bytes_stream in separated_result]
        return result

    def _audio_preprocessing(self, input_path: str, inplace: int) -> str | None:
        """Preprocess an audio file by applying noise reduction and voice activity detection.

        Args:
            input_path (str): Path to the input audio file.
            inplace (int): Flag indicating whether to overwrite the input file with the processed version.

        Returns:
            str or None: The path to the processed audio file, or None if processing fails.
        """
        self._apply_nr(input_path)
        return self._apply_vad(input_path, inplace)

    def _apply_vad(self, input_path: str, inplace: int) -> str | None:
        """Apply Voice Activity Detection (VAD) to an audio file.

        Args:
            input_path (str): Path to the input audio file.
            inplace (int): Flag indicating whether to overwrite the input file.

        Returns:
            str or None: The path to the audio file after VAD processing, or the original path if VAD is disabled.
        """
        if self.vad:
            return request_voice_activity_detection(input_path, f'{self.base_type.lower()}_{self.id}', inplace,
                                                    self.vad_url)
        return input_path

    def _apply_nr(self, input_path: str) -> str:
        """Apply Noise Reduction (NR) to an audio file.

        Args:
            input_path (str): Path to the input audio file.

        Returns:
            str: The path to the audio file after noise reduction processing.
        """
        if self.nr:
            request_speech_enhancement(input_path, f'{self.base_type.lower()}_{self.id}', self.speech_enhancer_url)
        return input_path

    def _store_audio(self, source_path, dest_path):
        """Manage the storage of audio files based on the storage flag.

        If storage is enabled, moves the audio file from source_path to dest_path.
        Otherwise, deletes the source file.

        Args:
            source_path (str): Original audio file path.
            dest_path (str): Destination path for the audio file.
        """
        if self.store:
            shutil.move(source_path, dest_path)
        else:
            os.remove(source_path)

    @staticmethod
    def filter_real_speakers(speakers, similarities, durations, texts) -> tuple:
        """Filter out non-valid speakers (e.g., 'silent', 'unknown') from recognition results.

        Args:
            speakers (list): List of recognized speakers.
            similarities (list): List of similarity scores.
            durations (list): List of audio durations.
            texts (list): List of associated texts or signals.

        Returns:
            tuple: A tuple containing lists of valid speakers, their similarity scores, durations, and texts.
        """
        real_speakers = []
        real_similarities = []
        real_durations = []
        real_texts = []
        for i, speaker in enumerate(speakers):
            if speaker not in ['silent', 'unknown']:
                real_speakers.append(speaker)
                real_similarities.append(similarities[i])
                real_durations.append(durations[i])
                real_texts.append(texts[i])

        return real_speakers, real_similarities, real_durations, real_texts

    @staticmethod
    def warm_up_resampler(sample_rate_original=44100, sample_rate_target=16000):
        """Warm up the audio resampler to avoid delays during the first resampling operation.

        Generates a short segment of silence and performs resampling using librosa.

        Args:
            sample_rate_original (int): Original sample rate (default is 44100).
            sample_rate_target (int): Target sample rate (default is 16000).
        """
        dummy_audio = np.zeros(sample_rate_original)
        _ = librosa.resample(dummy_audio, orig_sr=sample_rate_original, target_sr=sample_rate_target)
        print("Resampler has been warmed up.")

    @staticmethod
    def recording_prompt(seconds: float):
        """Display a recording prompt to the user for speaker registration.

        Prompts the user to press Enter and read a series of sentences within the given time frame.

        Args:
            seconds (float): Duration (in seconds) allowed for reading the prompt.
        """
        input(f"Press the Enter key to start recording, and read the following sentence in {seconds} seconds:\n"
              "1. The boy was there when the sun rose.\n"
              "2. A rod is used to catch pink salmon.\n"
              "3. The source of the huge river is the clear spring.\n"
              "4. Kick the ball straight and follow through.\n"
              "5. Help the woman get back to her feet.\n"
              "6. A pot of tea helps to pass the evening.\n"
              "7. Smoky fires lack flame and heat.\n"
              "8. The soft cushion broke the man's fall.\n"
              "9. The salt breeze came across from the sea.\n"
              "10. The girl at the booth sold fifty bonds."
              )
        print("------------------------------------------------")
