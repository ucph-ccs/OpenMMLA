import gc
import os
import queue
import time

import numpy as np
import soundfile as sf

from openmmla.bases.asr_diarization.errors import RecordingError, RecognizingError
from openmmla.streams.audio_stream import AudioStream, write_frame_to_wav
from openmmla.utils.audio.auga import normalize_decibel
from openmmla.utils.audio.augf import resample_audio
from openmmla.utils.audio.io import write_bytes_to_wav
from openmmla.utils.audio.properties import get_energy_level, calculate_audio_duration
from openmmla.utils.logger import get_logger
from .audio_base import AudioBase
from .enums import BLUE, ENDC
from .input import get_bucket_name, get_name, get_mode


class JabraAudioBase(AudioBase):
    """JabraAudioBase processes audio streams from built-in sound devices (PyAudio).

    This class extends the AudioBase abstract class with functionality specific to built-in sound devices.
    It supports both speaker profile registration and real-time voice recognition. The class handles configuration,
    audio stream setup, recording, and recognition processes. It also manages threading for continuous processing,
    and interacts with external services for MQTT, InfluxDB, and Redis.

    Key Features:
      - Configures protocol settings for Jabra devices.
      - Sets up an AudioStream to read audio data from built-in sound devices.
      - Implements methods for speaker registration and real-time recognition.
      - Supports both continuous recording and processing with or without speech separation.
      - Manages audio chunk assembly with half-scaled recognition when speech separation is not applied.
      - Manages audio chunk assembly without half-scaled recognition when speech separation is applied.
    """

    logger = get_logger(f'jabra-audio-base')

    def __init__(self, project_dir: str, config_path: str, mode: str = 'full', store: bool = True,
                 vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False):
        """Initialize the JabraAudioBase.

        Initializes the JabraAudioBase with the given configuration and operating parameters.
        Inherits and extends the AudioBase setup, and initializes the jabra-specific audio stream.

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
        super().__init__(project_dir=project_dir, config_path=config_path, mode=mode,
                         store=store, vad=vad, nr=nr, tr=tr, sp=sp)

    @property
    def base_type(self):
        """Return the type of the audio base.

        Returns:
            str: The string 'Jabra', indicating this is a Jabra audio base.
        """
        return 'Jabra'

    def _setup_yaml(self):
        """Extend YAML configuration setup for Jabra-specific parameters.

        Calls the parent setup method and sets the protocol for Jabra devices.
        """
        super()._setup_yaml()
        self.source = self.config[self.base_type]['source']
        self.stream_kwargs = self.config[self.base_type]['stream_kwargs']

    def _setup_objects(self):
        """Initialize Jabra-specific objects.

        Calls the parent object setup and then creates an AudioStream configured with the
        protocol setting for Jabra devices.
        """
        super()._setup_objects()
        self.audio_stream = AudioStream(source=self.source, **self.stream_kwargs)

    def _start_register(self):
        """Start the speaker profile registration process for Jabra devices.

        Prompts the user to record a registration sample, preprocesses the recorded audio, and then
        registers the speaker profile using the audio recognizer. If the recorded audio is too short
        or no name is provided, the registration is skipped.
        """
        print("------------------------------------------------")
        output_path = os.path.join(self.temp_dir, f'jabra_{self.id}_register.wav')
        self.recording_prompt(self.register_duration)

        self.audio_stream.start()
        audio_frame = self.audio_stream.read(self.register_duration, latest=True)
        self.audio_stream.stop()
        write_frame_to_wav(output_path, audio_frame)

        audio_path = self._audio_preprocessing(output_path, 1)

        if audio_path is None:
            self.logger.info("The recorded audio file is not long enough, please record again.")
            return

        name = get_name()
        if name == '':
            self.logger.info('Empty name, skip the registering process.')
            return

        self.audio_recognizer.register(audio_path, name)

    def _start_recognize(self):
        """Start the real-time voice recognition process.

        Sets up directories, queues, and MQTT communication before creating threads for:
          - Continuous recording.
          - Loading and queuing pre-recorded files (if in 'recognize' mode).
          - Continuous recognition (with or without speech separation).
          - Continuous transcription (if enabled).
          - Listening for stop signals.
        """
        if self.mode in ['full', 'recognize'] and len(self.audio_recognizer.speaker_names) == 0:
            print("------------------------------------------------")
            self.logger.info("Audio database is empty.")
            return

        self.bucket_name = get_bucket_name(self.influx_client)
        self.last_speaker = None
        self.audio_dir = os.path.join(self.runtime_dir, f'{self.bucket_name}', f'jabra_{self.id}')
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.speaker_frames_dict = {}
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        self._prepare_directories()
        self._listen_for_start_signal()

        # Create threads based on the current mode and options.
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

        Stops all threads and external clients, cleans up runtime variables.

        Args:
            e (Exception): The exception that occurred during recognition, if any.
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")

        self.mqtt_client.loop_stop()
        self.audio_stream.stop()
        self._clean_up()

    def _reset(self):
        """Reset the Jabra audio base.

        Reinitializes the audio base by calling the constructor with the current configuration,
        logs the reset status, and performs garbage collection.
        """
        self.__init__(project_dir=self.project_dir, config_path=self.config_path, mode=self.mode,
                      vad=self.vad, nr=self.nr, tr=self.tr, sp=self.sp, store=self.store)
        self.logger.info(f"Audio DB reset to {self.audio_db}")
        gc.collect()

    def _switch_mode(self):
        """Switch the operating mode between 'record', 'recognize' and 'full'."""
        self.mode = get_mode()
        self.logger.info(f"Switched to {self.mode} mode.")

    def _continuous_recording(self):
        """Continuously record audio from the Jabra stream and enqueue it for processing.

        Depending on the current mode:
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
                output_path = os.path.join(self.audio_dir, sub_dir, f'jabra_{self.id}_record_{acquire_time:.4f}.wav')

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
          - Preprocesses the audio (e.g., NR, VAD, etc.).
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
                processed_audio_path = self._audio_preprocessing(segment_audio_path, inplace=1)

                # Check voice quality via energy levels.
                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                if processed_audio_path and rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold:
                    speaker = 'unknown'
                else:
                    speaker = 'silent'

                duration = self.recognize_duration
                similarity = 0

                if speaker != 'silent':
                    normalize_decibel(segment_audio_path, rms_level=-18)
                    name, similarity = self.audio_recognizer.recognize(segment_audio_path)
                    duration = calculate_audio_duration(segment_audio_path)

                    if similarity > self.threshold:
                        speaker = name

                self._assemble_chunk_with_hsr(speaker, record_start_time, frames)
                self._publish_recognition_results(record_start_time, recognize_start_time, [speaker],
                                                  [np.round(np.float64(similarity), 4)], [duration])
                self._store_audio(segment_audio_path, os.path.join(self.audio_dir, 'segments',
                                                                   f'{speaker}_{round(float(record_start_time))}.wav'))
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
          - Preprocess the audio (e.g., NR, VAD).
          - Checks the energy level to determine if the segment is 'silent' or contains speech.
          - If speech is detected, applied speech separation and recognizes the speaker on each separated signal.
          - Assembles the audio chunk without hsr and stores/publishes the recognition result.

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
                processed_audio_path = self._audio_preprocessing(segment_audio_path, inplace=0)

                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                if processed_audio_path and rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold:
                    speaker = 'unknown'
                else:
                    speaker = 'silent'

                speakers, similarities, durations, signals = [], [], [], []
                if speaker != 'silent':
                    resample_audio(segment_audio_path, 8000)
                    sp_result = self._separate_speech(segment_audio_path)

                    # Process each separated audio stream.
                    for i, signal in enumerate(sp_result):
                        save_file = f'{segment_audio_path[:-4]}_spk{i}.wav'
                        sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)
                        processed_save_file = self._apply_vad(save_file, inplace=1)

                        speaker = 'unknown' if processed_save_file else 'silent'
                        duration = self.recognize_duration
                        similarity = 0

                        if processed_save_file:
                            duration = calculate_audio_duration(save_file)
                            normalize_decibel(save_file, rms_level=-18)
                            name, similarity = self.audio_recognizer.recognize(save_file)

                            if similarity > self.threshold:
                                speaker = name

                            self._store_audio(save_file, os.path.join(self.audio_dir, 'separations',
                                                                      f"{name if similarity > self.threshold else 'unknown'}_{round(float(record_start_time))}_spk{i}.wav"))
                        else:
                            self._store_audio(save_file, os.path.join(self.audio_dir, 'separations',
                                                                      f'silent_{round(float(record_start_time))}_spk{i}.wav'))

                        speakers.append(speaker)
                        similarities.append(np.round(np.float64(similarity), 4))
                        durations.append(duration)
                        signals.append(signal)
                else:
                    speakers.append('silent')
                    similarities.append(0)
                    durations.append(self.recognize_duration)
                    signals.append(bytes())

                self._store_audio(segment_audio_path,
                                  os.path.join(self.audio_dir, 'separations', f'{round(float(record_start_time))}.wav'))
                speakers, similarities, durations, signals = (
                    self._consolidate_separated_recognition_results(speakers, similarities, durations, signals))

                # Assemble audio chunk without half-scaled recognition.
                self._assemble_chunk_without_hsr(speakers, record_start_time, signals)
                self._publish_recognition_results(record_start_time, recognize_start_time, speakers, similarities,
                                                  durations)
            except queue.Empty:
                continue
            except Exception as e:
                raise RecognizingError(f'RecognizingError occurred when continuous recognizing: {e}') from e
            finally:
                gc.collect()

    def _assemble_chunk_with_hsr(self, speaker, record_start_time, frames):
        """Assemble and processs audio chunks with half-scaled recognition (HSR) at speaker boundaries.

        If the current recognized speaker matches the previous speaker, appends the audio frames.
        Otherwise, performs HSR by processing half-segments before and after the speaker change,
        updating the internal speaker frames dictionary accordingly. Also, adds transcribed chunks
        to the transcription queue if applicable.

        Note: This function is applicable only for recognition without speech separation.

        Args:
            speaker: The recognized speaker for the current segment.
            record_start_time: The start time of the current segment.
            frames: The audio frames of the current segment.
        """
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
                    left_temp_path = os.path.join(self.audio_dir, 'temp', f'jabra_{self.id}_left_temp.wav')
                    right_temp_path = os.path.join(self.audio_dir, 'temp', f'jabra_{self.id}_right_temp.wav')
                    number_frames = int(len(frames) / 2)
                    candidates = [self.last_speaker, speaker]
                    write_bytes_to_wav(left_temp_path, chunk_frames[-number_frames:], framerate=fr)
                    write_bytes_to_wav(right_temp_path, frames[:number_frames], framerate=fr)

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

                    # Optionally store the audio chunk locally.
                    if self.store:
                        chunk_audio_path = os.path.join(self.audio_dir, 'chunks',
                                                        f'{self.last_speaker}_chunk_{round(float(chunk_start_time))}.wav')
                        write_bytes_to_wav(chunk_audio_path, chunk_frames, framerate=fr)
                        if self.last_speaker != 'silent':
                            normalize_decibel(chunk_audio_path, rms_level=-20)

                self.speaker_frames_dict[speaker] = (record_start_time, frames)

        self.last_speaker = speaker

    def _assemble_chunk_without_hsr(self, speakers, record_start_time, signals):
        """Assemble audio chunks without half-scaled recognition for speech separation cases.

        If the current recognized speakers match the previous speakers, the audio signals are appended; otherwise,
        new entries are created. For previous speakers not present in the current chunk, the chunk is finalized and
        popped out.

        Args:
            speakers (list): List of finalized recognized speakers.
            record_start_time (str): Start time of the current audio segment.
            signals (list): List of finalized audio signals corresponding to the recognized speakers.
        """
        fr = 8000 if self.sp else 16000

        if not self.last_speaker:
            self.last_speaker = []
            for i, speaker in enumerate(speakers):
                self.speaker_frames_dict[speaker] = (record_start_time, signals[i])
                self.last_speaker.append(speaker)
        else:
            for speaker in self.last_speaker:
                if speaker in speakers:
                    chunk_start_time, chunk_frames = self.speaker_frames_dict[speaker]
                    chunk_frames += signals[speakers.index(speaker)]
                    self.speaker_frames_dict[speaker] = (chunk_start_time, chunk_frames)
                else:
                    chunk_start_time, chunk_frames = self.speaker_frames_dict.pop(speaker)
                    chunk_end_time = record_start_time
                    self.last_speaker.remove(speaker)

                    if speaker not in ['silent', 'unknown']:
                        self._add_to_transcription_queue(chunk_frames, speaker, chunk_start_time, chunk_end_time)

                    # Optionally store the audio chunk locally.
                    if self.store:
                        chunk_audio_path = os.path.join(self.audio_dir, 'chunks',
                                                        f'{speaker}_chunk_{round(float(chunk_start_time))}.wav')
                        write_bytes_to_wav(chunk_audio_path, chunk_frames, framerate=fr)
                        if speaker != 'silent':
                            normalize_decibel(chunk_audio_path, rms_level=-20)

            for i, speaker in enumerate(speakers):
                if speaker not in self.last_speaker:
                    self.last_speaker.append(speaker)
                    self.speaker_frames_dict[speaker] = (record_start_time, signals[i])
