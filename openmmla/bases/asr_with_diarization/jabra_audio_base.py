import gc
import os
import queue
import time

import numpy as np
import soundfile as sf

from openmmla.utils.audio.auga import normalize_decibel
from openmmla.utils.audio.processing import resample_audio_file, get_energy_level, write_frames_to_wav, \
    calculate_audio_duration
from openmmla.utils.errors import RecordingError, RecognizingError
from openmmla.utils.logger import get_logger
from .audio_base import AudioBase
from .enums import BLUE, ENDC
from .input import get_bucket_name, get_name, get_mode


class JabraAudioBase(AudioBase):
    """The jabra audio base processes the audio streams recorded from built-in or USB-wired speakers."""
    logger = get_logger(f'jabra-audio-base')

    def __init__(self, project_dir: str, config_path: str, mode: str = 'full', local: bool = True,
                 vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False, store: bool = True):
        """Initialize the Jabra Audio Base.

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
        return 'Jabra'

    def _register_profile(self):
        """Register participant's voice to audio database."""
        print("------------------------------------------------")
        output_path = os.path.join(self.audio_temp_dir, f'jabra_{self.id}_register.wav')
        audio_path = self.audio_recorder.record_registry_stream(duration=self.register_duration,
                                                                output_path=output_path, base_id=f'jabra_{self.id}')

        if audio_path is None:
            self.logger.info("The recorded audio file is not long enough, please record again.")
            return

        name = get_name()
        if name == '':
            return

        self.audio_recognizer.register(audio_path, name)

    def _recognize_voice(self):
        """Start the real-time voice recognition, including audio recording, speaker recognition, speech transcription,
        and listening on stop signal."""
        if len([file for file in os.listdir(self.audio_db) if file != '.DS_Store']) == 0:
            print("------------------------------------------------")
            self.logger.info("Audio database is empty.")
            return

        self.bucket_name = get_bucket_name(self.influx_client)
        self.last_speaker = None
        self.audio_dir = os.path.join(self.audio_realtime_dir, f'{self.bucket_name}', f'jabra_{self.id}')
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.speaker_frames_dict = {}
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        self._prepare_directories()
        self._listen_for_start_signal()

        # Create threads
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
            e: exception occurred during the recognition process
        """
        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")

        self.mqtt_client.loop_stop()
        self._clean_up()

    def _reset(self):
        """Set the new port for the audio base and reset it."""
        self.__init__(project_dir=self.project_dir, config_path=self.config_path, mode=self.mode, vad=self.vad,
                      nr=self.nr, tr=self.tr, sp=self.sp, store=self.store)
        self.logger.info(f"Audio DB reset to {self.audio_db}")
        gc.collect()

    def _switch_mode(self):
        """Switch the mode between record and recognize."""
        self.mode = get_mode()
        self.logger.info(f"Switched to {self.mode} mode.")

    def _continuous_recording(self):
        """Continuously record audio from stream and put it into the audio queue."""
        self.audio_recorder.start_recording_stream()

        try:
            while not self.stop_event.is_set():
                sub_dir = 'records' if self.mode == 'record' else 'temp'
                output_path = os.path.join(self.audio_dir, sub_dir,
                                           f'jabra_{self.id}_record_{time.time():.4f}.wav')

                frames = self.audio_recorder.read_frames_stream(self.recognize_duration)

                # write frames to file if in record mode and put into audio queue if in full mode
                if self.mode == 'record':
                    write_frames_to_wav(output_path, frames)
                    print(f"{BLUE}[Recording]{ENDC} {os.path.basename(output_path)} {len(frames)} frames")
                else:
                    self.audio_queue.put((output_path, frames))
        except Exception as e:
            raise RecordingError(f'RecordingError occurred when continuous recording: {e}') from e
        finally:
            self.audio_recorder.stop_recording_stream()

    def _continuous_recognizing(self):
        """Continuously recognize audio from the audio queue and publish the results into a Redis/MQTT channel."""
        while not self.stop_event.is_set():
            try:
                segment_audio_path, frames = self.audio_queue.get(timeout=1)
                record_start_time = os.path.basename(segment_audio_path).split('_')[-1][:-4]
                recognize_start_time = time.time()
                write_frames_to_wav(segment_audio_path, frames)

                # Audio pre-processing
                processed_audio_path = self.audio_recorder.post_processing(segment_audio_path, sampling_rate=16000,
                                                                           inplace=1, base_id=f'jabra_{self.id}')

                # Voice quality check
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
                self._store_audio(segment_audio_path, os.path.join(self.audio_dir, 'segments',
                                                                   f'{speaker}_{round(float(record_start_time))}.wav'))
                self._log_and_publish_results(record_start_time, recognize_start_time, [speaker],
                                              [np.round(np.float64(similarity), 4)], [duration])
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
                processed_audio_path = self.audio_recorder.post_processing(segment_audio_path, sampling_rate=16000,
                                                                           inplace=0, base_id=f'jabra_{self.id}')

                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                if processed_audio_path and rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold:
                    speaker = 'unknown'
                else:
                    speaker = 'silent'

                speakers, similarities, durations, signals = [], [], [], []
                if speaker != 'silent':
                    resample_audio_file(segment_audio_path, 8000)
                    sp_result = self._separate_speech(segment_audio_path)

                    # Recognize separated audio streams
                    for i, signal in enumerate(sp_result):
                        save_file = f'{segment_audio_path[:-4]}_spk{i}.wav'
                        sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)
                        processed_save_file = self.audio_recorder.apply_vad(save_file, sampling_rate=8000,
                                                                            inplace=1, base_id=f'jabra_{self.id}')

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
                final_speakers, final_similarities, final_durations, final_signals = self._finalize_sp_results(
                    speakers, similarities, durations, signals)

                # Assemble chunk without half-scaled recognition
                self._assemble_chunk_without_hsr(final_speakers, record_start_time, final_signals)
                self._log_and_publish_results(record_start_time, recognize_start_time, final_speakers,
                                              final_similarities, final_durations)
            except queue.Empty:
                continue
            except Exception as e:
                raise RecognizingError(f'RecognizingError occurred when continuous recognizing: {e}') from e
            finally:
                gc.collect()

    def _assemble_chunk_without_hsr(self, final_speakers, record_start_time, final_signals):
        """Assemble the chunk of audio frames without half-scaled recognition. Update the speaker audio dictionary and
        last speaker.

        Args:
            final_speakers: a list of finalized recognized speakers
            record_start_time: segment start time
            final_signals: a list of finalized audio signals
        """
        if not self.last_speaker:
            self.last_speaker = []
            for i, speaker in enumerate(final_speakers):
                self.speaker_frames_dict[speaker] = (record_start_time, final_signals[i])
                self.last_speaker.append(speaker)
        else:
            for speaker in self.last_speaker:
                if speaker in final_speakers:
                    chunk_start_time, chunk_frames = self.speaker_frames_dict[speaker]
                    chunk_frames += final_signals[final_speakers.index(speaker)]
                    self.speaker_frames_dict[speaker] = (chunk_start_time, chunk_frames)
                else:  # end of the speaker turn, pop out the speaker and transcribe the chunk
                    chunk_start_time, chunk_frames = self.speaker_frames_dict.pop(speaker)
                    chunk_end_time = record_start_time
                    self.last_speaker.remove(speaker)

                    if speaker not in ['silent', 'unknown']:
                        self._add_to_transcription_queue(chunk_frames, speaker, chunk_start_time,
                                                         chunk_end_time)

                    # Store transcribed chunk locally
                    if self.store:
                        chunk_audio_path = os.path.join(self.audio_dir, 'chunks',
                                                        f'{speaker}_chunk_{round(float(chunk_start_time))}.wav')
                        write_frames_to_wav(chunk_audio_path, chunk_frames, framerate=8000)
                        if speaker != 'silent':
                            normalize_decibel(chunk_audio_path, rms_level=-20)

            for i, speaker in enumerate(final_speakers):
                if speaker not in self.last_speaker:  # add new speaker and its corresponding frames
                    self.last_speaker.append(speaker)
                    self.speaker_frames_dict[speaker] = (record_start_time, final_signals[i])

    def _assemble_chunk_with_hsr(self, speaker, record_start_time, frames):
        """Assemble the chunk of audio frames if the current recognized speaker is the same as the last recognized
        speaker, if not, do speaker recognition on half-segment before and after the border of the speaker turn.
        Update the speaker audio dictionary and last speaker.

        Args:
            speaker: recognized speaker of the current segment
            record_start_time: record start time of the current segment
            frames: audio frames of the current segment
        """
        fr = 16000

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
                    left_temp_path = os.path.join(self.audio_dir, 'temp', f'jabra_{self.id}_left_temp.wav')
                    right_temp_path = os.path.join(self.audio_dir, 'temp', f'jabra_{self.id}_right_temp.wav')
                    number_frames = int(len(frames) / 2)
                    candidates = [self.last_speaker, speaker]
                    write_frames_to_wav(left_temp_path, chunk_frames[-number_frames:], framerate=fr)
                    write_frames_to_wav(right_temp_path, frames[:number_frames], framerate=fr)

                    if self.audio_recorder.apply_vad(left_temp_path, sampling_rate=fr, inplace=0,
                                                     base_id=f'jabra_{self.id}'):
                        left_speaker, _ = self.audio_recognizer.recognize_among_candidates(left_temp_path, candidates,
                                                                                           self.last_speaker,
                                                                                           self.keep_threshold)
                    else:
                        left_speaker = 'silent'

                    if self.audio_recorder.apply_vad(right_temp_path, sampling_rate=fr, inplace=0,
                                                     base_id=f'jabra_{self.id}'):
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
