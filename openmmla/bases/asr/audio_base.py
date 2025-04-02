import base64
import gc
import json
import os
import queue
import shutil
import threading
import time

import librosa
import numpy as np
import soundfile as sf

from openmmla.bases.asr.errors import RecordingError, RecognizingError, TranscribingError
from openmmla.bases.base import Base
from openmmla.services.asr.requests import request_speech_transcription, request_speech_separation, \
    request_speech_enhancement, request_voice_activity_detection
from openmmla.streams.audio_stream import AudioStream, write_frame_to_wav
from openmmla.utils.audio.auga import normalize_decibel, apply_gain
from openmmla.utils.audio.augf import resample_audio
from openmmla.utils.audio.io import read_bytes_from_wav, write_bytes_to_wav
from openmmla.utils.audio.properties import get_energy_level, calculate_audio_duration
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.logger import get_logger
from openmmla.utils.ports import free_port
from openmmla.utils.requests import resolve_url
from .audio_recognizer import AudioRecognizer
from .enums import BLUE, ENDC, GREEN
from .input import get_function_base, get_id, get_input_device_index, get_rtmp_url, get_mode, get_bucket_name, get_name


class AudioBase(Base):
    """AudioBase for audio processing pipelines that handles speaker recognition, transcription, and audio management.

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

    def __init__(self, project_dir: str | None, config_path: str, base_type: str, mode: str = 'full',
                 store: bool = True, vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False):
        """Initialize the audio processing pipeline base class.

        Sets up the configuration, directories, and required objects for audio processing.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            base_type: type of the audio base
            mode: operating mode, 'record', 'recognize', or 'full' (default: 'full')
            store: whether to store audio files (default: True)
            vad: whether to apply Voice Activity Detection (default: True)
            nr: whether to apply noise reduction (default: True)
            tr: whether to transcribe speech to text (default: True)
            sp: whether to perform speech separation (default: False)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        # Audio base specific parameters.
        if not base_type:
            raise ValueError("base_type must be specified.")

        self.base_type = base_type
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

        self._setup_yaml()
        self._setup_input()
        self._setup_directories()
        self._setup_objects()

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
        self.gain = float(self.config[self.base_type]['gain'])

        self.speech_transcriber_url = resolve_url(self.config['Server']['asr']['speech_transcription'])
        self.speech_separator_url = resolve_url(self.config['Server']['asr']['speech_separation'])
        self.speech_enhancer_url = resolve_url(self.config['Server']['asr']['speech_enhancement'])
        self.vad_url = resolve_url(self.config['Server']['asr']['voice_activity_detection'])

        self.source = self.config[self.base_type]['source']
        self.stream_kwargs = self.config[self.base_type]['stream_kwargs']

    def _setup_input(self):
        """Setup the input identifier for the audio base.

        Retrieves and assigns a unique identifier for this instance, calculates the port number by
        adding the port offset to the unique identifier (id), and ensures the port is free.
        """
        self.id = get_id()

        # set port number for 'udp/tcp'
        if self.source in ['udp', 'tcp']:
            self.port_offset = int(self.config[self.base_type].get('port_offset', 0))
            self.port = self.id + self.port_offset
            free_port(self.port)
            self.stream_kwargs['port'] = self.port

        # set input_device_index for 'pyaudio'
        if self.source == 'pyaudio':
            try:
                import pyaudio
            except ImportError:
                raise ImportError("pyaudio is not installed. Please install it using 'pip install pyaudio'.")

            p = pyaudio.PyAudio()
            info = p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            available_indexes = []

            for i in range(0, num_devices):
                if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print(i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
                    available_indexes.append(i)
            self.input_device_index = get_input_device_index(available_indexes)

        # set url for 'rtmp'
        if self.source == 'rtmp':
            # check self.config['RTMP']['audio_streams'] exist, if not, raise error
            if 'RTMP' not in self.config:
                raise ValueError("RTMP configuration is missing in the YAML file.")
            if 'audio_streams' not in self.config['RTMP']:
                raise ValueError("RTMP: audio_streams configuration is missing in the YAML file.")

            audio_stream_list = self.config['RTMP']['audio_streams'].split(',')
            for i, stream_rul in enumerate(audio_stream_list):
                print(f'{i} : {stream_rul}')
            self.url = get_rtmp_url(audio_stream_list)
            self.stream_kwargs['url'] = self.url

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

        Sets up clients for InfluxDB, Redis, and MQTT, warms up the audio resampler, and initializes the AudioRecognizer
        and AudioStream.
        """
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.warm_up_resampler()
        self.audio_recognizer = AudioRecognizer(config_path=self.config_path, audio_db=self.audio_db, store=self.store)
        self.audio_stream = AudioStream(source=self.source, **self.stream_kwargs)

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

    def _start_register(self):
        """Start the speaker profile registration process.

        Prompts the user to record a segment sample, applies gain and preprocessing, and then registers the speaker
        profile using the audio recognizer. If the recorded audio is too short or no name is provided, the registration
        is skipped.
        """
        print("------------------------------------------------")
        output_path = os.path.join(self.temp_dir, f'{self.base_type}_{self.id}_register.wav')
        self.recording_prompt(self.register_duration)

        self.audio_stream.start()
        audio_frame = self.audio_stream.read(duration=self.register_duration, latest=True)
        self.audio_stream.stop()
        write_frame_to_wav(output_path, audio_frame)

        apply_gain(output_path, self.gain)
        audio_path = self._audio_preprocessing(output_path, 1)

        if audio_path is None:
            self.logger.info("The recorded audio file is not long enough, please record again.")
            return

        name = get_name()
        if name == '':
            self.logger.info('Empty name, skip the registering process.')
            return

        self.audio_recognizer.register(audio_path, name)

    def _start_recognize(self, bucket_name=None):
        """Start the real-time voice recognition process.

        Sets up directories, queues, and MQTT communication before creating threads for:
          - Continuous recording.
          - Loading and queuing pre-recorded files (if in 'recognize' mode).
          - Continuous recognition (with or without speech separation).
          - Continuous transcription (if enabled).
          - Listening for stop signals.

        Args:
            bucket_name (str, optional): The bucket name for storing recognition results. If not provided,
                                         it is obtained interactively.
        """
        if self.mode in ['full', 'recognize'] and len(self.audio_recognizer.speaker_names) == 0:
            print("------------------------------------------------")
            self.logger.info("Audio database is empty.")
            return

        self.bucket_name = get_bucket_name(self.influx_client) if not bucket_name else bucket_name
        self.last_speaker = None
        self.audio_dir = os.path.join(self.runtime_dir, f'{self.bucket_name}', f'{self.base_type}_{self.id}')
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.speaker_frames_dict = {}
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        self._prepare_directories()
        self._listen_for_start_signal()

        # Create threads based on the operating mode
        if self.mode in ['record', 'full']:
            self._create_thread(self._continuous_recording)
        if self.mode == 'recognize':
            self._create_thread(self._enqueue_recorded_files)
        if self.mode in ['recognize', 'full']:
            recognition_task = self._continuous_recognizing_sp if self.sp else self._continuous_recognizing
            self._create_thread(recognition_task)
            if self.tr:
                self._create_thread(self._continuous_transcribing)
        self._create_thread(self._listen_for_stop_signal)

        # Start and join threads, handling exceptions if they occur
        exception_occurred = None
        try:
            self._start_threads()
            self._join_threads()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning(
                f"During voice recognition, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}",
                exc_info=True)
            exception_occurred = e
        finally:
            self._recognition_handler(exception_occurred)

    def _reset(self):
        """Reset the audio base.

        Reinitializes the audio base by calling the constructor with the current configuration,
        logs the reset status, and performs garbage collection.
        """
        self.__init__(project_dir=self.project_dir, config_path=self.config_path, base_type=self.base_type,
                      mode=self.mode, vad=self.vad, nr=self.nr, tr=self.tr, sp=self.sp, store=self.store)
        self.logger.info(f"Audio DB reset to {self.audio_db}")
        gc.collect()

    def _switch_mode(self):
        """Switch the operating mode between 'record', 'recognize' and 'full'."""
        self.mode = get_mode()
        self.logger.info(f"Switched to {self.mode} mode.")

    def _continuous_recording(self):
        """Continuously record audio from the audio stream and enqueue it for processing.

        Depending on the operating mode:
          - In 'record' mode, writes recorded frames to a file.
          - In 'full' mode, puts the audio frame bytes into the audio queue.

        Raises:
            RecordingError: If an error occurs during the recording process.
        """
        first_time = True
        sub_dir = 'records' if self.mode == 'record' else 'temp'
        self.audio_stream.start()

        while not self.stop_event.is_set():
            try:
                audio_frame = self.audio_stream.read(duration=self.recognize_duration, latest=first_time)
                first_time = False
                frames = audio_frame.to_bytes()
                acquire_time = audio_frame.timestamp
                output_path = os.path.join(self.audio_dir, sub_dir,
                                           f'{self.base_type}_{self.id}_record_{acquire_time:.4f}.wav')

                if self.mode == 'record':
                    write_frame_to_wav(output_path, audio_frame)
                    print(f"{BLUE}[Recording]{ENDC} {os.path.basename(output_path)} {len(frames)} frames")
                else:
                    self.audio_queue.put((output_path, frames))
            except Exception as e:
                raise RecordingError(f'RecordingError occurred when continuous recording: {e}') from e

    def _continuous_recognizing(self):
        """Continuously process and recognize audio segments from the audio queue.

        For each segment, the method:
          - Writes the received bytes to a WAV file.
          - Preprocesses the audio (e.g., applying gain, NR, VAD, etc.).
          - Checks the energy level to determine if the segment is 'silent' or contains speech.
          - If speech is detected, recognizes the speaker using the audio recognizer.
          - Assembles the audio chunk with hsr and stores/publishes the recognition result.

        Raises:
            RecognizingError: If an error occurs during the recognition process.
        """
        while not self.stop_event.is_set():
            try:
                segment_audio_path, frames = self.audio_queue.get(timeout=1)
                record_start_time = os.path.basename(segment_audio_path).split('_')[-1][:-4]
                recognize_start_time = time.time()
                write_bytes_to_wav(segment_audio_path, frames)

                # Audio pre-processing
                apply_gain(segment_audio_path, self.gain)
                processed_audio_path = self._audio_preprocessing(segment_audio_path, inplace=1)

                # Evaluate energy levels for quality check
                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                if processed_audio_path and rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold:
                    speaker = 'unknown'
                else:
                    speaker = 'silent'

                duration = self.recognize_duration
                similarity = 0

                if speaker == 'unknown':  # voice detected
                    normalize_decibel(segment_audio_path, rms_level=-20)
                    name, similarity = self.audio_recognizer.recognize(segment_audio_path)
                    duration = calculate_audio_duration(segment_audio_path)

                    if similarity > self.threshold:
                        speaker = name
                        if self.base_type == 'Badge':
                            energy_level_factor = np.log(rms_value) / np.log(self.rms_threshold)
                            similarity = min(similarity * energy_level_factor, 1)
                    else:
                        c1 = rms_value + peak_value
                        c2 = self.rms_threshold + self.rms_peak_threshold
                        ratio = c1 / (c1 + c2)
                        similarity = self.threshold * ratio

                self._assemble_chunk_with_hsr(speaker, record_start_time, frames)
                self._publish_recognition(record_start_time, recognize_start_time, [speaker],
                                          [np.round(np.float64(similarity), 4)], [duration])

                if self.store:
                    shutil.move(segment_audio_path,
                                os.path.join(self.audio_dir, 'segments', f'{speaker}_{float(record_start_time)}.wav'))
                else:
                    os.remove(segment_audio_path)
            except queue.Empty:
                continue
            except Exception as e:
                raise RecognizingError(f'RecognizingError occurred when continuous recognizing: {e}') from e
            finally:
                gc.collect()

    def _continuous_recognizing_sp(self):
        """Continuously process and recognize audio segments with speech separation.

        For each segment, the method:
          - Write the received bytes to a WAV file.
          - Preprocess the audio (e.g., applying gain, NR, VAD).
          - Checks the energy level to determine if the segment is 'silent' or contains speech.
          - If speech is detected, applied speech separation and recognizes the speaker on each separated signal.
          - Assembles the audio chunk with hsr and stores/publishes the recognition result.

        Raises:
            RecognizingError: If an error occurs during the recognition process.
        """
        while not self.stop_event.is_set():
            try:
                segment_audio_path, frames = self.audio_queue.get(timeout=1)
                record_start_time = os.path.basename(segment_audio_path).split('_')[-1][:-4]
                recognize_start_time = time.time()
                write_bytes_to_wav(segment_audio_path, frames)

                # Audio pre-processing
                apply_gain(segment_audio_path, self.gain)
                processed_audio_path = self._audio_preprocessing(segment_audio_path, inplace=0)

                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                # speaker = 'silent' if processed_audio_path is None else 'unknown'
                speaker = 'unknown' if processed_audio_path else 'silent'
                duration = self.recognize_duration
                similarity = 0
                best_separate_path = ''
                best_separate_frames = None

                resample_audio(segment_audio_path, 8000)
                if processed_audio_path and rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold:
                    sp_result = self._separate_speech(segment_audio_path)

                    # Recognize separated audio streams
                    for i, signal in enumerate(sp_result):
                        save_file = f'{segment_audio_path[:-4]}_spk{i}.wav'
                        sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)
                        processed_save_file = self._apply_vad(save_file, inplace=True)

                        # Skip file if VAD fails, removing it immediately.
                        if not processed_save_file:
                            os.remove(save_file)
                            continue

                        normalize_decibel(save_file, rms_level=-20)
                        temp_name, temp_similarity = self.audio_recognizer.recognize(save_file)

                        # If a better result is found, update best info and remove any old file.
                        if temp_similarity > similarity:
                            similarity = temp_similarity
                            duration = calculate_audio_duration(save_file)
                            best_separate_frames = signal

                            if similarity > self.threshold:
                                speaker = temp_name

                            if best_separate_path:
                                os.remove(best_separate_path)
                            best_separate_path = save_file
                        else:
                            os.remove(save_file)

                    if similarity > self.threshold and self.base_type == 'Badge':
                        energy_level_factor = np.log(rms_value) / np.log(self.rms_threshold)
                        similarity = min(similarity * energy_level_factor, 1)

                if speaker == 'unknown' and similarity == 0:
                    speaker = 'silent'

                resampled_segment_bytes = read_bytes_from_wav(segment_audio_path)
                self._assemble_chunk_with_hsr(speaker, record_start_time, resampled_segment_bytes, best_separate_frames)
                self._publish_recognition(record_start_time, recognize_start_time, [speaker],
                                          [np.round(np.float64(similarity), 4)], [duration])

                if self.store:
                    if best_separate_path:
                        shutil.move(best_separate_path, os.path.join(self.audio_dir, 'separations',
                                                                     f'{speaker}_{round(float(record_start_time))}_spk.wav'))
                    shutil.move(segment_audio_path, os.path.join(self.audio_dir, 'segments',
                                                                 f'{speaker}_{round(float(record_start_time))}.wav'))
                else:
                    if best_separate_path:
                        os.remove(best_separate_path)
                    os.remove(segment_audio_path)
            except queue.Empty:
                continue
            except Exception as e:
                raise RecognizingError(f'RecognizingError occurred when continuous recognizing: {e}') from e
            finally:
                gc.collect()

    def _enqueue_recorded_files(self):
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
                self._upload_transcription(speaker, text, chunk_start_time, chunk_end_time)
            except queue.Empty:
                continue
            except Exception as e:
                raise TranscribingError(f'TranscribingError occurred when transcribing: {e}') from e

    def _assemble_chunk_with_hsr(self, speaker, record_start_time, origin_frames, separate_frames=None):
        """Assemble and process audio chunks with half-scaled recognition (HSR) at speaker boundaries.

        If the current recognized speaker matches the previous speaker, appends the audio frames.
        Otherwise, performs HSR by processing half-segments before and after the speaker change,
        updating the internal speaker frames dictionary accordingly. Also, adds transcribed chunks
        to the transcription queue if applicable.

        Args:
            speaker: The recognized speaker of the current segment.
            record_start_time: The start time of the current segment.
            origin_frames: The original audio frames of the current segment.
            separate_frames: Optional; speech separated frames from the current segment.
        """
        frames = separate_frames if separate_frames else origin_frames
        fr = 8000 if self.sp else 16000

        if not self.last_speaker:
            self.speaker_frames_dict[speaker] = (record_start_time, frames)
        else:
            if self.last_speaker == speaker:
                chunk_start_time, last_speaker_frames = self.speaker_frames_dict[speaker]
                last_speaker_frames += frames
                self.speaker_frames_dict[speaker] = (chunk_start_time, last_speaker_frames)
            else:
                chunk_start_time, chunk_frames = self.speaker_frames_dict.pop(self.last_speaker)
                chunk_end_time = record_start_time

                # Perform half-scaled recognition on speaker turn border
                if chunk_frames:
                    left_temp_path = os.path.join(self.audio_dir, 'temp', f'{self.base_type}_{self.id}_left_temp.wav')
                    right_temp_path = os.path.join(self.audio_dir, 'temp',
                                                   f'{self.base_type}_{self.id}_right_temp.wav')
                    number_frames = int(len(frames) / 2)
                    candidates = [self.last_speaker, speaker]
                    write_bytes_to_wav(left_temp_path, chunk_frames[-number_frames:], framerate=fr)
                    write_bytes_to_wav(right_temp_path, frames[:number_frames], framerate=fr)

                    if not self.sp and self.base_type == 'Badge':
                        apply_gain(left_temp_path)
                        apply_gain(right_temp_path)

                    if self._apply_vad(left_temp_path, inplace=0):
                        left_speaker, _ = self.audio_recognizer.recognize_among_candidates(
                            left_temp_path, candidates, self.last_speaker, self.keep_threshold)
                    else:
                        left_speaker = 'silent'

                    if self._apply_vad(right_temp_path, inplace=0):
                        right_speaker, _ = self.audio_recognizer.recognize_among_candidates(
                            right_temp_path, candidates, speaker, self.keep_threshold)
                    else:
                        right_speaker = 'silent'

                    chunk_frames, frames, record_start_time, chunk_end_time = self._update_chunk_list(
                        left_speaker, right_speaker, self.last_speaker, speaker, chunk_frames, frames,
                        record_start_time)

                    os.remove(left_temp_path)
                    os.remove(right_temp_path)

                if chunk_frames:
                    if self.last_speaker not in ['silent', 'unknown']:
                        self._enqueue_transcription(chunk_frames, self.last_speaker, chunk_start_time,
                                                    chunk_end_time)

                    # Optionally store the audio chunk locally
                    if self.store:
                        chunk_audio_path = os.path.join(self.audio_dir, 'chunks',
                                                        f'{self.last_speaker}_chunk_{round(float(chunk_start_time))}.wav')
                        write_bytes_to_wav(chunk_audio_path, chunk_frames, framerate=fr)
                        if self.last_speaker != 'silent':
                            normalize_decibel(chunk_audio_path, rms_level=-20)

                self.speaker_frames_dict[speaker] = (record_start_time, frames)

        self.last_speaker = speaker

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

    def _enqueue_transcription(self, frames, speaker, chunk_start_time, chunk_end_time):
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

    def _upload_transcription(self, speaker, text, chunk_start_time, chunk_end_time):
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

    def _publish_recognition(self, record_start_time, recognize_start_time, speakers, similarities, durations):
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

    def _recognition_handler(self, e):
        """Handle exceptions during the recognition process and perform cleanup.

        Stops all threads and external clients, cleans up runtime variables, and if a RecordingError
        occurred, restarts the recognition service with the current bucket.

        Args:
            e (Exception): The exception that occurred during recognition, if any.
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")

        current_bucket = self.bucket_name  # assign bucket name before cleaning up
        self.mqtt_client.loop_stop()
        self.audio_stream.stop()
        self._clean_up()
        if isinstance(e, RecordingError):
            self.logger.info("Restarting recognizing service.")
            self._start_recognize(current_bucket)

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
