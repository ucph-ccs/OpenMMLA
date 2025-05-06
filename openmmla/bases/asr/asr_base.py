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
from openmmla.utils.input import select_or_create_bucket
from openmmla.utils.logger import get_logger
from openmmla.utils.ports import free_port
from openmmla.utils.requests import resolve_url
from .audio_recognizer import AudioRecognizer
from .enums import BLUE, ENDC, GREEN
from .input import get_function_base, get_id, get_input_device_index, get_rtmp_url, get_mode, get_name


class ASRBase(Base):
    """ASRBase class for automatic speech recognition with speaker diarization."""

    logger = get_logger(f'asr-base')

    def __init__(self, project_dir: str | None, config_path: str, base_type: str, mode: str = 'full',
                 store: bool = True, vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False, hsr: bool = True):
        """Initialize the ASRBase class.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            base_type: type of the ASR base
            mode: operating mode, 'record', 'recognize', or 'full' (default: 'full')
            store: whether to store audio files (default: True)
            vad: whether to apply Voice Activity Detection (default: True)
            nr: whether to apply noise reduction (default: True)
            tr: whether to transcribe speech to text (default: True)
            sp: whether to perform speech separation (default: False)
            hsr: whether to apply Half-Scaled Recognition at speaker boundaries (default: True)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        # ASRBase specific parameters
        if not base_type:
            raise ValueError("base_type must be specified.")

        self.base_type = base_type
        self.mode = mode
        self.store = store
        self.vad = vad
        self.nr = nr
        self.tr = tr
        self.sp = sp
        self.hsr = hsr

        # Runtime attributes
        self.bucket_name = None
        self.last_speaker = None
        self.audio_dir = None
        self.audio_queue = None
        self.transcription_queue = None
        self.speaker_frames_dict = None
        self.stop_event = threading.Event()
        self.threads = []

        self._setup_yaml()
        self._setup_input()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        """Load and assign configuration parameters from the YAML configuration file.

        Read various settings such as durations, thresholds, and service URLs required for the audio processing pipeline.
        """
        base_config = self.config[self.base_type]
        asr_server_config = self.config['Server']['asr']

        self.register_duration = int(base_config['register_duration'])
        self.recognize_duration = int(base_config['recognize_sp_duration']) if self.sp else int(
            base_config['recognize_duration'])
        self.rms_threshold = int(base_config['rms_threshold'])
        self.rms_peak_threshold = int(base_config['rms_peak_threshold'])
        self.threshold = float(base_config['recognize_sp_threshold']) if self.sp else float(
            base_config['recognize_threshold'])
        self.keep_threshold = float(base_config['keep_sp_threshold']) if self.sp else float(
            base_config['keep_threshold'])
        self.gain = float(base_config['gain'])

        self.source = base_config['source']
        self.stream_kwargs = base_config['stream_kwargs']

        self.speech_transcriber_url = resolve_url(asr_server_config['speech_transcriber'])
        self.speech_separator_url = resolve_url(asr_server_config['speech_separator'])
        self.speech_enhancer_url = resolve_url(asr_server_config['speech_enhancer'])
        self.vad_url = resolve_url(asr_server_config['voice_activity_detector'])

    def _setup_input(self):
        """Set up the identifier for the base from user input.

        Retrieve and assigns a unique identifier for this instance, calculates the port number by
        adding the port offset to the unique identifier (id), and ensures the port is free.
        """
        self.id = get_id()
        print(f"\033]0;ASR Base {self.base_type} {self.id} \007")

        # set port number for 'udp/tcp'
        if self.source in ['udp', 'tcp']:
            self.port_offset = int(self.config[self.base_type].get('port_offset', 0))
            self.port = self.id + self.port_offset
            free_port(self.port)
            self.stream_kwargs['port'] = self.port

        # set input_device_index for 'pyaudio'
        elif self.source == 'pyaudio':
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
            p.terminate()

            self.input_device_index = get_input_device_index(available_indexes)
            self.stream_kwargs['input_device_index'] = self.input_device_index

        # set url for 'rtmp'
        elif self.source == 'rtmp':
            if 'RTMP' not in self.config:
                raise ValueError("RTMP configuration is missing in the YAML file.")
            if 'audio_streams' not in self.config['RTMP']:
                raise ValueError("RTMP: audio_streams configuration is missing in the YAML file.")

            audio_stream_list = [url.strip() for url in self.config['RTMP']['audio_streams'].split(',') if url.strip()]
            for i, stream_rul in enumerate(audio_stream_list):
                print(f'{i} : {stream_rul}')
            self.url = get_rtmp_url(audio_stream_list)
            self.stream_kwargs['url'] = self.url

    def _setup_directories(self):
        """Create and set up the necessary directories for runtime operations.

        Create directories for runtime files, temporary files, speaker profiles, and audio databases.
        Ensures that the required folder structure exists.
        """
        self.logger_dir = os.path.join(self.project_dir, 'logger')
        self.runtime_dir = os.path.join(self.project_dir, 'real-time', 'runtime')
        self.temp_dir = os.path.join(self.project_dir, 'real-time', 'temp')
        self.profiles_dir = os.path.join(self.project_dir, 'real-time', 'profiles')
        self.audio_db = os.path.join(self.profiles_dir, f'{self.base_type}_{self.id}')

        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.audio_db, exist_ok=True)

    def _setup_objects(self):
        """Initialize external service clients and internal processing objects.

        Set up the clients for InfluxDB, Redis, and MQTT, warms up the audio resampler, and initializes the
        AudioRecognizer and AudioStream.
        """
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.warm_up_resampler()
        self.audio_recognizer = AudioRecognizer(config_path=self.config_path, audio_db=self.audio_db, store=self.store)
        self.audio_stream = AudioStream(source=self.source, **self.stream_kwargs)

    def run(self):
        """Run the ASR base.

        Provides an interactive interface to:
          1. Register speaker profiles.
          2. Start real-time voice recognition.
          3. Reset or switch the ASR base mode.

        Continuously prompts the user for input until termination.
        """
        func_map = {1: self._start_registration, 2: self._start_recognition, 3: self._reset, 4: self._switch_mode}
        while True:
            try:
                select_fun = get_function_base(self.id, self.mode)
                if select_fun == 0:
                    print("------------------------------------------------")
                    clear_directory(self.temp_dir)
                    self.logger.info("Exiting the program...")
                    break
                func_map.get(select_fun, lambda: self.logger.warning("Invalid option"))()
            except (Exception, KeyboardInterrupt) as e:
                self._clean_up()
                self.logger.warning(
                    f"During running the ASR base, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)

    def _start_registration(self):
        """Start the speaker profile registration process.

        Prompt the user to record a segment sample, applies gain and preprocessing, and then registers the speaker
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

    def _start_recognition(self, bucket_name: str | None = None):
        """Start the real-time voice recognition process.

        Set up directories, queues, and MQTT communication before creating threads for:
          - Continuous recording.
          - Loading and queuing pre-recorded files (if in 'recognize' mode).
          - Continuous recognition (with or without speech separation).
          - Continuous transcription (if enabled).
          - Listening for stop signals.

        Args:
            bucket_name: The bucket name for storing recognition results. If not provided, it is obtained interactively.
        """
        if self.mode in ['full', 'recognize'] and len(self.audio_recognizer.speaker_names) == 0:
            print("------------------------------------------------")
            self.logger.info("Audio database is empty.")
            return

        self.bucket_name = select_or_create_bucket(self.influx_client) if not bucket_name else bucket_name
        self.logger = get_logger(f'asr-base-{self.bucket_name}',
                                 os.path.join(self.logger_dir,
                                              f'{self.bucket_name}_asr_{self.base_type}_{self.id}.log'))

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

    def _recognition_handler(self, e: Exception | KeyboardInterrupt | None):
        """Handle exceptions during the recognition process and perform cleanup.

        Stop all threads and external clients, cleans up runtime variables, and if a RecordingError
        occurred, restarts the recognition service with the current bucket.

        Args:
            e: The exception that occurred during recognition, if any.
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
            self._start_recognition(current_bucket)

    def _reset(self):
        """Reset the ASR base.

        Reinitialize the ASR base by calling the constructor with the current configuration,
        logs the reset status, and performs garbage collection.
        """
        self.__init__(project_dir=self.project_dir, config_path=self.config_path, base_type=self.base_type,
                      mode=self.mode, vad=self.vad, nr=self.nr, tr=self.tr, sp=self.sp, store=self.store, hsr=self.hsr)
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
                output_path = os.path.join(self.audio_dir, sub_dir, f'{self.base_type}_{self.id}_record_{acquire_time:.4f}.wav')

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
                record_start_time = float(os.path.basename(segment_audio_path).split('_')[-1][:-4])
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
                                os.path.join(self.audio_dir, 'segments', f'{speaker}_{record_start_time}.wav'))
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
                record_start_time = float(os.path.basename(segment_audio_path).split('_')[-1][:-4])
                recognize_start_time = time.time()
                write_bytes_to_wav(segment_audio_path, frames)

                # Audio pre-processing
                apply_gain(segment_audio_path, self.gain)
                processed_audio_path = self._audio_preprocessing(segment_audio_path, inplace=0)

                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
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
                        processed_save_file = self._apply_vad(save_file, inplace=1)

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
                                                                     f'{speaker}_{record_start_time}_spk.wav'))
                    shutil.move(segment_audio_path,
                                os.path.join(self.audio_dir, 'segments', f'{speaker}_{record_start_time}.wav'))
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

    def _assemble_chunk_with_hsr(self, speaker: str, record_start_time: float, origin_frames: bytes,
                                 separate_frames: bytes | None = None):
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
            if speaker == self.last_speaker:
                chunk_start_time, last_speaker_frames = self.speaker_frames_dict[speaker]
                last_speaker_frames += frames
                self.speaker_frames_dict[speaker] = (chunk_start_time, last_speaker_frames)
            else:
                chunk_start_time, chunk_frames = self.speaker_frames_dict.pop(self.last_speaker)
                chunk_end_time = record_start_time

                # Perform half-scaled recognition on speaker turn border if hsr is enabled
                if chunk_frames and self.hsr:
                    self.logger.info(f"Performing half-scaled recognition on speaker turn border for {self.last_speaker} and {speaker}")
                    left_temp_path = os.path.join(self.audio_dir, 'temp', f'{self.base_type}_{self.id}_left_temp.wav')
                    right_temp_path = os.path.join(self.audio_dir, 'temp', f'{self.base_type}_{self.id}_right_temp.wav')
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
                        self._enqueue_transcription(chunk_frames, self.last_speaker, chunk_start_time, chunk_end_time)

                    # Optionally store the audio chunk locally
                    if self.store:
                        chunk_audio_path = os.path.join(self.audio_dir, 'chunks',
                                                        f'{self.last_speaker}_chunk_{chunk_start_time}.wav')
                        write_bytes_to_wav(chunk_audio_path, chunk_frames, framerate=fr)
                        if self.last_speaker != 'silent':
                            normalize_decibel(chunk_audio_path, rms_level=-20)

                self.speaker_frames_dict[speaker] = (record_start_time, frames)

        self.last_speaker = speaker

    def _update_chunk_list(self, left_speaker: str, right_speaker: str, last_speaker: str, current_speaker: str,
                           chunk_frames: bytes, frames: bytes, record_start_time: float) -> tuple:
        """Update the chunk list based on recognition results at a speaker turn boundary.

        Adjust the audio chunks based on recognition outcomes from different segments,
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
            record_start_time = record_start_time + self.recognize_duration / 2
            chunk_end_time = record_start_time
        elif left_speaker == right_speaker and right_speaker == current_speaker:  # Case ABBB: Extend the right chunk
            frames = chunk_frames[-num_frames:] + frames
            chunk_frames = chunk_frames[:-num_frames]
            record_start_time = record_start_time - self.recognize_duration / 2
            chunk_end_time = record_start_time
        else:  # Other cases: AABB, A_BB, AA_B, A__B (excluding ABAB, AB_B, A_AB)
            chunk_end_time = record_start_time
            if left_speaker != last_speaker:
                chunk_frames = chunk_frames[:-num_frames]
                chunk_end_time = record_start_time - self.recognize_duration / 2
            if right_speaker != current_speaker:
                frames = frames[num_frames:]
                record_start_time = record_start_time + self.recognize_duration / 2

        return chunk_frames, frames, record_start_time, chunk_end_time

    def _enqueue_transcription(self, frames: bytes, speaker: str, chunk_start_time: float, chunk_end_time: float):
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

    def _transcribe(self, frames: bytes, frame_rate: int) -> str:
        """Transcribe audio frames to text using an external speech-to-text service.

        Sends the audio frames to a remote transcription service and retrieves the 
        resulting text. The transcription is performed with the base's identifier
        to maintain context in multi-device setups.

        Args:
            frames: Raw audio byte data to be transcribed.
            frame_rate: Sample rate of the audio frames in Hz (typically 8000 or 16000).

        Returns:
            The transcribed text as a string. Empty string if transcription failed.
        """
        text = request_speech_transcription(frames, frame_rate, f'{self.base_type.lower()}_{self.id}',
                                            self.speech_transcriber_url)
        return text if text is not None else ""

    def _upload_transcription(self, speaker: str, text: str, chunk_start_time: float, chunk_end_time: float):
        """Upload the transcribed speech chunk to the database.

        Constructs a transcription record with speaker, text content, and timing information,
        then writes it to InfluxDB for persistent storage. Also displays the transcription
        in the console for real-time monitoring.

        Args:
            speaker: Identified speaker for the transcribed chunk.
            text: Transcribed text content from the audio chunk.
            chunk_start_time: Start timestamp of the audio chunk.
            chunk_end_time: End timestamp of the audio chunk.
        """
        transcription_record = {
            "measurement": "speaker transcription",
            "fields": {
                "time_bucket": chunk_start_time,
                "text": text,
                "speaker": speaker,
                "chunk_start_time": chunk_start_time,
                "chunk_end_time": chunk_end_time,
            },
        }
        print(f"{GREEN}[Speaker Transcription]{ENDC}{transcription_record['fields']['time_bucket']}: "
              f"{GREEN}{speaker} : {text}{ENDC}")
        self.influx_client.write(self.bucket_name, record=transcription_record)

    def _publish_recognition(self, record_start_time: float, recognize_start_time: float, speakers: list[str],
                             similarities: list[float], durations: list[float]):
        """Log and publish speaker recognition results via MQTT.

        Constructs a JSON record with recognition details and publishes it on the designated MQTT channel.
        The record includes speaker identities, similarity scores, and timing information.

        Args:
            record_start_time: Start time of the recorded segment (timestamp).
            recognize_start_time: Start time of the recognition process (timestamp).
            speakers: List of recognized speakers.
            similarities: List of similarity scores (0.0-1.0) corresponding to speakers.
            durations: List of audio durations in seconds for each speaker segment.
        """
        base_recognition_result = {
            'base_id': f'{self.base_type.lower()}_{self.id}',
            'speakers': json.dumps(speakers),
            'similarities': json.dumps(similarities),
            'durations': json.dumps(durations),
            'record_start_time': record_start_time
        }
        print(f"{BLUE}[Speaker Recognition]{ENDC}{base_recognition_result['record_start_time']}: "
              f"{BLUE}{base_recognition_result['speakers']}{ENDC}, similarity: {base_recognition_result['similarities']},"
              f"processed time: {time.time() - recognize_start_time} seconds")
        result_str = json.dumps(base_recognition_result)
        self.mqtt_client.publish(f'{self.bucket_name}/asr', result_str)

    def _separate_speech(self, segment_audio_path: str) -> list[bytes]:
        """Separate overlapping speech from an audio segment using source separation.

        Uses an external speech separation service to process the audio file, identifying and
        isolating different speakers in the recording. The separated signals are returned as
        raw audio bytes for further processing.

        Args:
            segment_audio_path: Path to the audio segment file containing potentially overlapped speech.

        Returns:
            A list of separated speech signals as raw bytes (decoded from base64). Each element 
            represents an isolated speaker's audio. Returns an empty list if separation fails.
        """
        separated_result = request_speech_separation(segment_audio_path, f'{self.base_type.lower()}_{self.id}',
                                                     self.speech_separator_url)
        if separated_result is None:
            return []

        result = [base64.b64decode(encoded_bytes_stream) for encoded_bytes_stream in separated_result]
        return result

    def _audio_preprocessing(self, input_path: str, inplace: int) -> str | None:
        """Preprocess an audio file by applying noise reduction and voice activity detection.
        
        Performs a sequence of audio enhancement steps to improve recognition quality:
        1. Noise reduction (if enabled) - Enhances speech by reducing background noise
        2. Voice activity detection (if enabled) - Identifies and isolates speech segments
        
        Args:
            input_path: Path to the input audio file to be processed.
            inplace: Flag indicating whether to modify the input file (1) when VAD is enabled.
        
        Returns:
            Path to the processed audio file if successful, or None if processing failed
            (e.g., if no voice activity was detected in the file).
        """
        self._apply_nr(input_path)
        return self._apply_vad(input_path, inplace)

    def _apply_vad(self, input_path: str, inplace: int) -> str | None:
        """Apply Voice Activity Detection (VAD) to an audio file.

        Uses an external VAD service to identify speech segments in the audio file. 
        This helps filter out silence and non-speech portions to improve recognition quality.

        Args:
            input_path: Path to the input audio file to process.
            inplace: Flag indicating whether to modify the input file (1).

        Returns:
            Path to the processed audio file if VAD is enabled and successful, the original path if VAD 
            is disabled, or None if no speech was detected.
        """
        if self.vad:
            return request_voice_activity_detection(input_path, f'{self.base_type.lower()}_{self.id}', inplace,
                                                    self.vad_url)
        return input_path

    def _apply_nr(self, input_path: str) -> str:
        """Apply Noise Reduction (NR) to an audio file.

        Uses an external speech enhancement service to reduce background noise in the audio file.
        This improves speech clarity for better recognition results.

        Args:
            input_path: Path to the input audio file to process.

        Returns:
            Path to the audio file after noise reduction (the same as input_path, as the file is
            modified in place by the external service).
        """
        if self.nr:
            request_speech_enhancement(input_path, f'{self.base_type.lower()}_{self.id}', self.speech_enhancer_url)
        return input_path

    def _prepare_directories(self):
        """Prepare subdirectories for audio processing based on the operating mode.

        Create subdirectories for segments, chunks, separations, temporary files, and records.
        Clear specific directories based on the current operating mode.
        """
        sub_dirs = ['segments', 'chunks', 'separations', 'temp', 'records']
        for subdir in sub_dirs:
            directory_path = os.path.join(self.audio_dir, subdir)
            os.makedirs(directory_path, exist_ok=True)

        if self.mode == 'recognize':
            for subdir in ['segments', 'chunks', 'separations']:
                clear_directory(os.path.join(self.audio_dir, subdir))

        clear_directory(os.path.join(self.audio_dir, 'temp'))

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
    def warm_up_resampler(sample_rate_original: int = 44100, sample_rate_target: int = 16000):
        """Warm up the audio resampler to avoid delays during the first resampling operation.

        Performs a no-op resampling operation on a silent audio segment to ensure that when the
        actual resampling is needed, there won't be initialization delays. This is particularly
        important for real-time processing scenarios.

        Args:
            sample_rate_original: Original sample rate in Hz (default is 44100 Hz).
            sample_rate_target: Target sample rate in Hz (default is 16000 Hz).
        """
        dummy_audio = np.zeros(sample_rate_original)
        _ = librosa.resample(dummy_audio, orig_sr=sample_rate_original, target_sr=sample_rate_target)
        print("Resampler has been warmed up.")

    @staticmethod
    def recording_prompt(seconds: float):
        """Display a recording prompt to the user for speaker registration.

        Guides the user through the speaker registration process by presenting a standardized
        set of phonetically balanced sentences to read aloud. These sentences are designed
        to capture various speech characteristics for more accurate speaker identification.

        Args:
            seconds: Duration (in seconds) allowed for recording the prompt sentences.
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

    @property
    def bucket_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current bucket_name.
        
        This property provides the MQTT topic name used for start/stop control signals.
        The channel name is constructed using the current bucket_name, which allows
        for controlling specific sessions without affecting others.
        
        Returns:
            The control channel string in format '{bucket_name}/asr/control' if bucket_name 
            is set, otherwise None.
        """
        if self.bucket_name:
            return f'{self.bucket_name}/asr/control'
        return None
