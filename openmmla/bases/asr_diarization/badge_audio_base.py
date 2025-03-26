import gc
import os
import queue
import shutil
import time

import numpy as np
import soundfile as sf

from openmmla.bases.asr_diarization.errors import RecordingError, RecognizingError
from openmmla.streams.audio_stream import AudioStream, write_frame_to_wav
from openmmla.utils.audio.auga import normalize_decibel, apply_gain
from openmmla.utils.audio.augf import resample_audio
from openmmla.utils.audio.io import read_bytes_from_wav, write_bytes_to_wav
from openmmla.utils.audio.properties import get_energy_level, calculate_audio_duration
from openmmla.utils.logger import get_logger
from openmmla.utils.ports import free_port
from .audio_base import AudioBase
from .enums import BLUE, ENDC
from .input import get_mode, get_bucket_name, get_name


class BadgeAudioBase(AudioBase):
    """BadgeAudioBase processes audio streams recorded from wireless wearable badges (TCP/UDP).

    This class extends the AudioBase abstract class with functionality specific to wearable badge devices.
    It supports both speaker profile registration and real-time voice recognition. The class handles configuration,
    audio stream setup, recording, and recognition processes. It also manages threading for continuous processing,
    and interacts with external services for MQTT, InfluxDB, and Redis.

    Key Features:
      - Configures connection parameters (listening IP, protocol, port offset) for the badge.
      - Sets up an AudioStream to read audio data from the badge device.
      - Implements methods for speaker registration and real-time recognition.
      - Supports both continuous recording and processing with or without speech separation.
      - Manages audio chunk assembly with half-scaled recognition.
    """

    logger = get_logger(f'badge-audio-base')

    def __init__(self, project_dir: str, config_path: str, mode: str = 'full', store: bool = True,
                 vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False):
        """Initialize the BadgeAudioBase.

        Initializes the BadgeAudioBase with the given configuration and operating parameters.
        Inherits and extends the AudioBase setup, and initializes the badge-specific audio stream.

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
        super().__init__(project_dir=project_dir, config_path=config_path, mode=mode, store=store, vad=vad, nr=nr,
                         tr=tr, sp=sp)

    @property
    def base_type(self):
        """Return the type of the audio base.

        Returns:
            str: The string 'Badge', indicating this is a badge audio base.
        """
        return 'Badge'

    def _setup_yaml(self):
        """Extend YAML configuration setup for badge-specific parameters.

        Calls the parent YAML setup and then loads additional settings:
          - host: The IP address to listen for audio data.
          - source: The communication protocol used by the badge.
          - port_offset: The offset used to compute the port number.
        """
        super()._setup_yaml()
        self.source = self.config[self.base_type]['source']
        self.port_offset = int(self.config[self.base_type]['port_offset'])
        self.stream_kwargs = self.config[self.base_type]['stream_kwargs']

    def _setup_input(self):
        """Setup input parameters for the badge audio stream.

        Calls the parent object setup and then calculates the port number by
        adding the port offset to the unique identifier (id), and ensures the port is free.
        """
        super()._setup_input()
        self.port = self.id + self.port_offset
        free_port(self.port)

    def _setup_objects(self):
        """Initialize badge-specific objects.

        Calls the parent object setup and then creates an AudioStream object configured with the
        protocol, listening IP, and port number for badge devices.
        """
        super()._setup_objects()
        self.audio_stream = AudioStream(source=self.source, port=self.port, **self.stream_kwargs)

    def _start_register(self):
        """Start the speaker profile registration process for the badge.

        Prompts the user to record a segment sample, applies gain and preprocessing, and then
        registers the speaker profile using the audio recognizer. If the recorded audio is too
        short or no name is provided, the registration is skipped.
        """
        print("------------------------------------------------")
        output_path = os.path.join(self.temp_dir, f'badge_{self.id}_register.wav')
        self.recording_prompt(self.register_duration)

        self.audio_stream.start()
        audio_frame = self.audio_stream.read(duration=self.register_duration, latest=True)
        self.audio_stream.stop()
        write_frame_to_wav(output_path, audio_frame)

        apply_gain(output_path)
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
        self.audio_dir = os.path.join(self.runtime_dir, f'{self.bucket_name}', f'badge_{self.id}')
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
            self._create_thread(self._load_and_queue_recorded_files)
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

        current_bucket = self.bucket_name
        self.mqtt_client.loop_stop()
        self.audio_stream.stop()
        self._clean_up()
        if isinstance(e, RecordingError):
            self.logger.info("Restarting recognizing service.")
            self._start_recognize(current_bucket)

    def _reset(self):
        """Reset the Jabra audio base.

        Reinitializes the audio base by calling the constructor with the current configuration,
        logs the reset status, and performs garbage collection.
        """
        self.__init__(project_dir=self.project_dir, config_path=self.config_path, mode=self.mode, vad=self.vad,
                      nr=self.nr, tr=self.tr, sp=self.sp, store=self.store)
        self.logger.info(f"Audio DB reset to {self.audio_db}")
        gc.collect()

    def _switch_mode(self):
        """Switch the operating mode between 'record', 'recognize' and 'full'."""
        self.mode = get_mode()
        self.logger.info(f"Switched to {self.mode} mode.")

    def _continuous_recording(self):
        """Continuously record audio from the badge stream and enqueue it for processing.

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
                frame_bytes = audio_frame.to_bytes()
                acquire_time = audio_frame.timestamp
                output_path = os.path.join(self.audio_dir, sub_dir, f'badge_{self.id}_record_{acquire_time:.4f}.wav')

                if self.mode == 'record':
                    write_frame_to_wav(output_path, audio_frame)
                    print(f"{BLUE}[Recording]{ENDC} {os.path.basename(output_path)} {len(frame_bytes)} frames")
                else:
                    self.audio_queue.put((output_path, frame_bytes))
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
            RecognizingError: If an error occurs during processing.
        """
        while not self.stop_event.is_set():
            try:
                segment_audio_path, frame_bytes = self.audio_queue.get(timeout=1)
                record_start_time = os.path.basename(segment_audio_path).split('_')[-1][:-4]
                recognize_start_time = time.time()
                write_bytes_to_wav(segment_audio_path, frame_bytes)

                # Audio pre-processing
                apply_gain(segment_audio_path)
                processed_audio_path = self._audio_preprocessing(segment_audio_path, inplace=1)

                # Evaluate energy levels for quality check
                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                if processed_audio_path and rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold:
                    speaker = 'unknown'
                else:
                    speaker = 'silent'

                duration = self.recognize_duration
                similarity = 0

                if speaker != 'silent':
                    normalize_decibel(segment_audio_path, rms_level=-20)
                    name, similarity = self.audio_recognizer.recognize(segment_audio_path)
                    duration = calculate_audio_duration(segment_audio_path)

                    if similarity > self.threshold:
                        speaker = name
                        energy_level_factor = np.log(rms_value) / np.log(self.rms_threshold)
                        similarity = min(similarity * energy_level_factor, 1)

                if speaker == 'unknown' and similarity == 0:
                    ratio = rms_value / self.rms_threshold if rms_value <= self.rms_threshold else peak_value / self.rms_peak_threshold
                    similarity = self.threshold * ratio

                self._assemble_chunk_with_hsr(speaker, record_start_time, frame_bytes)
                self._publish_recognition_results(record_start_time, recognize_start_time, [speaker],
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
                apply_gain(segment_audio_path)
                processed_audio_path = self._audio_preprocessing(segment_audio_path, inplace=0)

                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                speaker = 'silent' if processed_audio_path is None else 'unknown'
                duration = self.recognize_duration
                similarity = 0
                best_separate_path = ''
                best_separate_frames = None

                resample_audio(segment_audio_path, 8000)
                if processed_audio_path:
                    if rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold:
                        sp_result = self._separate_speech(segment_audio_path)

                        # Recognize separated audio streams
                        for i, signal in enumerate(sp_result):
                            save_file = f'{segment_audio_path[:-4]}_spk{i}.wav'
                            sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)
                            processed_save_file = self._apply_vad(save_file, inplace=1)

                            if processed_save_file:
                                normalize_decibel(save_file, rms_level=-20)
                                temp_name, temp_similarity = self.audio_recognizer.recognize(save_file)

                                # Select the best recognition result
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
                            else:
                                os.remove(save_file)

                        if similarity > self.threshold:
                            energy_level_factor = np.log(rms_value) / np.log(self.rms_threshold)
                            similarity = min(similarity * energy_level_factor, 1)

                if speaker == 'unknown' and similarity == 0:
                    speaker = 'silent'

                resampled_segment_bytes = read_bytes_from_wav(segment_audio_path)
                self._assemble_chunk_with_hsr(speaker, record_start_time, resampled_segment_bytes,
                                              best_separate_frames)
                self._publish_recognition_results(record_start_time, recognize_start_time, [speaker],
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
                    left_temp_path = os.path.join(self.audio_dir, 'temp', f'badge_{self.id}_left_temp.wav')
                    right_temp_path = os.path.join(self.audio_dir, 'temp', f'badge_{self.id}_right_temp.wav')
                    number_frames = int(len(frames) / 2)
                    candidates = [self.last_speaker, speaker]
                    write_bytes_to_wav(left_temp_path, chunk_frames[-number_frames:], framerate=fr)
                    write_bytes_to_wav(right_temp_path, frames[:number_frames], framerate=fr)

                    if not self.sp:
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
                        self._add_to_transcription_queue(chunk_frames, self.last_speaker, chunk_start_time,
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
