import gc
import os
import queue
import shutil
import socket
import time

import numpy as np
import soundfile as sf

from openmmla.utils.audio.auga import normalize_decibel, apply_gain
from openmmla.utils.audio.processing import get_energy_level, read_frames_from_wav, write_frames_to_wav, \
    calculate_audio_duration, resample_audio_file
from openmmla.utils.errors import RecordingError, RecognizingError
from openmmla.utils.logger import get_logger
from openmmla.utils.sockets import read_frames_tcp, clear_socket_udp, read_frames_udp
from .audio_base import AudioBase
from .enums import BLUE, ENDC
from .input import get_mode, get_bucket_name, get_name


class BadgeAudioBase(AudioBase):
    """The badge audio base process the audio streams recorded from wireless wearable badges."""
    logger = get_logger(f'badge-audio-base')

    def __init__(self, project_dir: str, config_path: str, mode: str = 'full', local: bool = False,
                 vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False, store: bool = True):
        """Initialize the badge audio base.

        Args:
            project_dir: root directory of the project
            config_path: path to the configuration file, either absolute or relative to the root directory
            mode: operating mode, either 'record', 'recognize', or 'full', default to 'full'
            local: whether to run the audio base locally, default to False
            vad: whether to use the VAD, default to True
            nr: whether to use the denoiser to enhance speech, default to True
            tr: whether to transcribe speech to text, default to True
            sp: whether to do speech separation for overlapped segment, default to False
            store: whether to store audio files, default to True
        """
        super().__init__(project_dir=project_dir, config_path=config_path, mode=mode, local=local, vad=vad, nr=nr,
                         tr=tr, sp=sp, store=store)

    @property
    def base_type(self):
        return 'Badge'

    def _setup_yaml(self):
        super()._setup_yaml()
        self.listening_ip = self.config[self.base_type]['listening_ip']
        self.protocol = self.config[self.base_type]['protocol']
        self.port_offset = int(self.config[self.base_type]['port_offset'])

    def _setup_input(self):
        super()._setup_input()
        self.port = self.id + self.port_offset

    def _register_profile(self):
        """Register participant's voice to audio database."""
        print("------------------------------------------------")
        output_path = os.path.join(self.audio_temp_dir, f'badge_{self.id}_register.wav')
        sock_param = (socket.AF_INET, socket.SOCK_STREAM) if self.protocol == 'TCP' else (
            socket.AF_INET, socket.SOCK_DGRAM)
        record_func = self.audio_recorder.record_registry_tcp if self.protocol == 'TCP' else (
            self.audio_recorder.record_registry_udp)

        with socket.socket(*sock_param) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.listening_ip, self.port))
            audio_path = record_func(self.register_duration, output_path, f'badge_{self.id}', sock)

        if audio_path is None:
            self.logger.info("The recorded audio file is not long enough, please record again.")
            return

        name = get_name()
        if name == '':
            self.logger.info('Empty name, skip the registering process.')
            return

        self.audio_recognizer.register(audio_path, name)

    def _recognize_voice(self, bucket_name=None):
        """Start the real-time voice recognition, including audio recording, speaker recognition, speech transcription,
        and listening on stop signal.

        Args:
            bucket_name: the bucket name for storing the results, default to None
        """
        if len([file for file in os.listdir(self.audio_db) if file != '.DS_Store']) == 0:
            print("------------------------------------------------")
            self.logger.info("Audio database is empty.")
            return

        self.bucket_name = get_bucket_name(self.influx_client) if not bucket_name else bucket_name
        self.last_speaker = None
        self.audio_dir = os.path.join(self.audio_realtime_dir, f'{self.bucket_name}', f'badge_{self.id}')
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.speaker_frames_dict = {}
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        sock_param = (socket.AF_INET, socket.SOCK_STREAM) if self.protocol == 'TCP' else (
            socket.AF_INET, socket.SOCK_DGRAM)

        self._prepare_directories()

        with socket.socket(*sock_param) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.listening_ip, self.port))

            self._listen_for_start_signal()
            if self.protocol == 'UDP':
                clear_socket_udp(sock)
            self.logger.info("Buffer cleared, start recording...")

            # Create threads
            if self.mode in ['record', 'full']:
                self._create_thread(self._continuous_recording, sock)
            if self.mode == 'recognize':
                self._create_thread(self._load_and_queue_recorded_files)
            if self.mode in ['recognize', 'full']:
                recognition_task = self._continuous_recognizing_sp if self.sp else self._continuous_recognizing
                self._create_thread(recognition_task)
                if self.tr:
                    self._create_thread(self._continuous_transcribing)
            self._create_thread(self._listen_for_stop_signal)

            # Start threads
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
        """Handle exceptions and stop all threads.

        Args:
            e: the exception occurred during the recognition process
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")

        current_bucket = self.bucket_name
        self.mqtt_client.loop_stop()
        self._clean_up()
        if isinstance(e, RecordingError):
            self.logger.info("Restarting recognizing service.")
            self._recognize_voice(current_bucket)

    def _reset(self):
        """Set the new port for the audio base and reinitialize audio base."""
        self.__init__(project_dir=self.project_dir, config_path=self.config_path, mode=self.mode, vad=self.vad,
                      nr=self.nr, tr=self.tr, sp=self.sp, store=self.store)
        self.logger.info(f"Audio DB reset to {self.audio_db}")
        gc.collect()

    def _switch_mode(self):
        """Switch the mode between record and recognize."""
        self.mode = get_mode()
        self.logger.info(f"Switched to {self.mode} mode.")

    def _continuous_recording(self, sock):
        """Continuously record audio from port and put it into the audio queue.

        Args:
            sock: the socket object for receiving audio stream
        """
        if self.protocol == 'TCP':
            sock.listen(1)
            conn, addr = sock.accept()

            while not self.stop_event.is_set():
                try:
                    sub_dir = 'records' if self.mode == 'record' else 'temp'
                    output_path = os.path.join(self.audio_dir, sub_dir,
                                               f'badge_{self.id}_record_{time.time():.4f}.wav')

                    frames = read_frames_tcp(sock, conn, self.recognize_duration)

                    # write frames to file if in record mode and put into audio queue if in full mode
                    if self.mode == 'record':
                        write_frames_to_wav(output_path, frames)
                        print(f"{BLUE}[Recording]{ENDC} {os.path.basename(output_path)} {len(frames)} frames")
                    else:
                        self.audio_queue.put((output_path, frames))
                except Exception as e:
                    raise RecordingError(f'RecordingError occurred when continuous recording: {e}') from e
        else:
            while not self.stop_event.is_set():
                try:
                    sub_dir = 'records' if self.mode == 'record' else 'temp'
                    output_path = os.path.join(self.audio_dir, sub_dir,
                                               f'badge_{self.id}_record_{time.time():.4f}.wav')
                    frames = read_frames_udp(sock, self.recognize_duration)

                    # write frames to file if in record mode and put into audio queue if in full mode
                    if self.mode == 'record':
                        write_frames_to_wav(output_path, frames)
                        print(f"{BLUE}[Recording]{ENDC} {os.path.basename(output_path)} {len(frames)} frames")
                    else:
                        self.audio_queue.put((output_path, frames))
                except Exception as e:
                    raise RecordingError(f'RecordingError occurred when continuous recording: {e}') from e

    def _continuous_recognizing(self):
        """Continuously recognize audio from the audio queue and put the results into the Redis channel."""
        while not self.stop_event.is_set():
            try:
                segment_audio_path, frames = self.audio_queue.get(timeout=1)
                record_start_time = os.path.basename(segment_audio_path).split('_')[-1][:-4]
                recognize_start_time = time.time()
                write_frames_to_wav(segment_audio_path, frames)

                # Audio pre-processing
                apply_gain(segment_audio_path)
                processed_audio_path = self.audio_recorder.post_processing(segment_audio_path, sampling_rate=16000,
                                                                           inplace=1, base_id=f'badge_{self.id}')

                # Voice quality check
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
                        energy_level_factor = np.log(rms_value) / np.log(self.rms_threshold)  # Weight factor
                        similarity = min(similarity * energy_level_factor, 1)

                if speaker == 'unknown' and similarity == 0:
                    ratio = rms_value / self.rms_threshold if rms_value <= self.rms_threshold \
                        else peak_value / self.rms_peak_threshold
                    similarity = self.threshold * ratio

                self._assemble_chunk_with_hsr(speaker, record_start_time, frames)
                self._log_and_publish_results(record_start_time, recognize_start_time, [speaker],
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
        """Continuously recognize audio with speech separation from the audio queue and publish the results into
        a Redis/MQTT channel."""
        while not self.stop_event.is_set():
            try:
                segment_audio_path, frames = self.audio_queue.get(timeout=1)
                record_start_time = os.path.basename(segment_audio_path).split('_')[-1][:-4]
                recognize_start_time = time.time()
                write_frames_to_wav(segment_audio_path, frames)

                # Audio pre-processing
                apply_gain(segment_audio_path)
                processed_audio_path = self.audio_recorder.post_processing(segment_audio_path, sampling_rate=16000,
                                                                           inplace=0, base_id=f'badge_{self.id}')

                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                speaker = 'silent' if processed_audio_path is None else 'unknown'
                duration = self.recognize_duration
                similarity = 0
                best_separate_path = ''
                best_separate_frames = None

                resample_audio_file(segment_audio_path, 8000)
                if processed_audio_path:
                    if rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold:
                        sp_result = self._separate_speech(segment_audio_path)

                        # Recognize separated audio streams
                        for i, signal in enumerate(sp_result):
                            save_file = f'{segment_audio_path[:-4]}_spk{i}.wav'
                            sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)
                            processed_save_file = self.audio_recorder.apply_vad(save_file, sampling_rate=8000,
                                                                                inplace=1, base_id=f'badge_{self.id}')

                            if processed_save_file:
                                normalize_decibel(save_file, rms_level=-20)
                                temp_name, temp_similarity = self.audio_recognizer.recognize(save_file)

                                # Compare and select best result
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

                resampled_segment_bytes = read_frames_from_wav(segment_audio_path)
                self._assemble_chunk_with_hsr(speaker, record_start_time, resampled_segment_bytes,
                                              best_separate_frames)
                self._log_and_publish_results(record_start_time, recognize_start_time, [speaker],
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
        """Assemble the chunk of audio frames if the current recognized speaker is the same as the last recognized
        speaker, if not, do speaker recognition on half-segment before and after the border of the speaker turn. Update
        the speaker audio dictionary and last speaker.

        Args:
            speaker: recognized speaker of the current segment
            record_start_time: record start time of the current segment
            origin_frames: audio frames of the current segment
            separate_frames: speech separated frames of the current segment, default to None
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

                # Half-scaled recognition
                if chunk_frames:
                    left_temp_path = os.path.join(self.audio_dir, 'temp', f'badge_{self.id}_left_temp.wav')
                    right_temp_path = os.path.join(self.audio_dir, 'temp', f'badge_{self.id}_right_temp.wav')
                    number_frames = int(len(frames) / 2)
                    candidates = [self.last_speaker, speaker]
                    write_frames_to_wav(left_temp_path, chunk_frames[-number_frames:], framerate=fr)
                    write_frames_to_wav(right_temp_path, frames[:number_frames], framerate=fr)

                    if not self.sp:
                        apply_gain(left_temp_path)
                        apply_gain(right_temp_path)

                    if self.audio_recorder.apply_vad(left_temp_path, sampling_rate=fr, inplace=0,
                                                     base_id=f'badge_{self.id}'):
                        left_speaker, _ = self.audio_recognizer.recognize_among_candidates(left_temp_path, candidates,
                                                                                           self.last_speaker,
                                                                                           self.keep_threshold)
                    else:
                        left_speaker = 'silent'

                    if self.audio_recorder.apply_vad(right_temp_path, sampling_rate=fr, inplace=0,
                                                     base_id=f'badge_{self.id}'):
                        right_speaker, _ = self.audio_recognizer.recognize_among_candidates(right_temp_path, candidates,
                                                                                            speaker,
                                                                                            self.keep_threshold)
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

                    # Store transcribed chunk locally
                    if self.store:
                        chunk_audio_path = os.path.join(self.audio_dir, 'chunks',
                                                        f'{self.last_speaker}_chunk_{round(float(chunk_start_time))}.wav')
                        write_frames_to_wav(chunk_audio_path, chunk_frames, framerate=fr)
                        if self.last_speaker != 'silent':
                            normalize_decibel(chunk_audio_path, rms_level=-20)

                self.speaker_frames_dict[speaker] = (record_start_time, frames)

        self.last_speaker = speaker
