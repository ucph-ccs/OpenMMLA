import base64
import gc
import json
import os
import queue
import shutil
import threading
import time
from abc import abstractmethod, ABC
from typing import Union

import librosa
import numpy as np

from openmmla.bases.asr_with_diarization.errors import RecordingError, TranscribingError
from openmmla.bases.base import Base
from openmmla.services.audio.requests import request_speech_transcription, request_speech_separation, \
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
    logger = get_logger(f'audio-base')

    def __init__(self, project_dir: str, config_path: str, mode: str = 'full', vad: bool = True, nr: bool = True,
                 tr: bool = True, sp: bool = False, store: bool = True):
        """Initialize the Audio Base.

        Args:
            project_dir: root directory of the project
            config_path: path to the configuration file, either absolute or relative to the root directory
            mode: operating mode, either 'record', 'recognize', or 'full', default to 'full'
            vad: whether to use the VAD, default to True
            nr: whether to use the denoiser to enhance speech, default to True
            tr: whether to transcribe speech to text, default to True
            sp: whether to do speech separation for overlapped segment, default to False
            store: whether to store audio files, default to True
        """
        super().__init__(project_dir, config_path)

        """Audio base specific parameters."""
        self.mode = mode
        self.vad = vad
        self.nr = nr
        self.tr = tr
        self.sp = sp
        self.store = store

        """Runtime attributes."""
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
        """Return the type of the audio base."""
        pass

    def _setup_input(self):
        self.id = get_id()

    def _setup_yaml(self):
        self.register_duration = int(self.config[self.base_type]['register_duration'])
        self.rms_threshold = int(self.config[self.base_type]['rms_threshold'])
        self.rms_peak_threshold = int(self.config[self.base_type]['rms_peak_threshold'])
        self.threshold = float(self.config[self.base_type]['recognize_sp_threshold']) if self.sp else float(
            self.config[self.base_type]['recognize_threshold'])
        self.keep_threshold = float(self.config[self.base_type]['keep_sp_threshold']) if self.sp else float(
            self.config[self.base_type]['keep_threshold'])
        self.recognize_duration = int(self.config[self.base_type]['recognize_sp_duration']) if self.sp else int(
            self.config[self.base_type]['recognize_duration'])
        self.speech_transcriber_url = resolve_url(self.config['Server']['asr']['speech_transcription'])
        self.speech_separator_url = resolve_url(self.config['Server']['asr']['speech_separation'])
        self.speech_enhancer_url = resolve_url(self.config['Server']['asr']['speech_enhancement'])
        self.vad_url = resolve_url(self.config['Server']['asr']['voice_activity_detection'])

    def _setup_directories(self):
        self.audio_realtime_dir = os.path.join(self.project_dir, 'audio', 'real-time')
        self.audio_temp_dir = os.path.join(self.project_dir, 'audio', 'temp')
        self.audio_db_dir = os.path.join(self.project_dir, 'audio_db')
        self.audio_db = os.path.join(self.audio_db_dir, f'{self.base_type}_{self.id}')
        os.makedirs(self.audio_realtime_dir, exist_ok=True)
        os.makedirs(self.audio_temp_dir, exist_ok=True)
        os.makedirs(self.audio_db_dir, exist_ok=True)
        os.makedirs(self.audio_db, exist_ok=True)

    def _setup_objects(self):
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.warm_up_resampler()
        # self.audio_recorder = AudioRecorder(config_path=self.config_path, vad_enable=self.vad, nr_enable=self.nr)
        self.audio_recognizer = AudioRecognizer(config_path=self.config_path, audio_db=self.audio_db)

    def run(self):
        """Interface for running the Audio Base."""
        func_map = {1: self._register_profile, 2: self._recognize_voice, 3: self._reset, 4: self._switch_mode}
        while True:
            try:
                print(f"\033]0;Audio Base {self.base_type} {self.id} \007")
                select_fun = get_function_base()
                if select_fun == 0:
                    print("------------------------------------------------")
                    clear_directory(self.audio_temp_dir)
                    self.logger.info("Exiting the program...")
                    break
                func_map.get(select_fun, lambda: self.logger.warning("Invalid option"))()
            except (Exception, KeyboardInterrupt) as e:
                self._clean_up()
                self.logger.warning(
                    f"During running the audio base, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}, Come back to the main menu.",
                    exc_info=True)

    @abstractmethod
    def _register_profile(self):
        """Register participant's voice to audio database."""
        pass

    @abstractmethod
    def _recognize_voice(self):
        """Start the real-time voice recognition, including audio recording, speaker recognition, speech transcription,
        and listening on stop signal."""
        pass

    @abstractmethod
    def _reset(self):
        """Set the new port for the audio base and reinitialize audio base."""
        pass

    @abstractmethod
    def _switch_mode(self):
        """Switch the mode of the audio base."""
        pass

    def _clean_up(self):
        """Free memory by resetting the runtime variables."""
        self.bucket_name = None
        self.last_speaker = None
        self.audio_dir = None
        self.audio_queue = None
        self.transcription_queue = None
        self.speaker_frames_dict = None
        gc.collect()

    def _prepare_directories(self):
        """Prepare and manage directories based on the operation mode."""
        sub_dirs = ['segments', 'chunks', 'separations', 'temp', 'records']
        for subdir in sub_dirs:
            directory_path = os.path.join(self.audio_dir, subdir)
            os.makedirs(directory_path, exist_ok=True)

        if self.mode == 'recognize':
            for subdir in ['segments', 'chunks', 'separations']:
                clear_directory(os.path.join(self.audio_dir, subdir))

        # Buggy code: when the badge is disconnected, the records folder will be cleared
        # if self.mode == 'record':
        #     clear_directory(os.path.join(self.audio_dir, 'records'))

        clear_directory(os.path.join(self.audio_dir, 'temp'))

    def _load_and_queue_recorded_files(self):
        """Load pre-recorded audio files and queue them for recognition when in recognize mode."""
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
                           record_start_time):
        """Update the chunk list based on the half-scaled recognition results at the speaker turn border.

        Args:
            left_speaker: recognized speaker of the left half-segment
            right_speaker: recognized speaker of the right half-segment
            last_speaker: recognized speaker of the last segment
            current_speaker: recognized speaker of the current segment
            chunk_frames: audio frames of the current chunk belonging to the last speaker
            frames: audio frames of the current segment
            record_start_time: record start time of the current segment
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
        else:  # Other cases, AABB, A_BB, AA_B, A__B (excluding ABAB, AB_B, A_AB)
            chunk_end_time = record_start_time
            if left_speaker != last_speaker:
                chunk_frames = chunk_frames[:-num_frames]
                chunk_end_time = str(float(record_start_time) - self.recognize_duration / 2)
            if right_speaker != current_speaker:
                frames = frames[num_frames:]
                record_start_time = str(float(record_start_time) + self.recognize_duration / 2)

        return chunk_frames, frames, record_start_time, chunk_end_time

    def _add_to_transcription_queue(self, frames, speaker, chunk_start_time, chunk_end_time):
        """Add audio chunk to the transcription queue.

        Args:
            frames: audio frames to be transcribed
            speaker: recognized speaker of the audio frames
            chunk_start_time: start time of the audio frames
            chunk_end_time: end time of the audio frames
        """
        if self.tr:
            self.transcription_queue.put((frames, speaker, chunk_start_time, chunk_end_time))

    def _continuous_transcribing(self):
        """Continuously transcribe audio from the transcription queue."""
        while not self.stop_event.is_set():
            try:
                frames, speaker, chunk_start_time, chunk_end_time = self.transcription_queue.get(timeout=2)
                text = self._transcribe(frames)
                self._upload_speech_transcription(speaker, text, chunk_start_time, chunk_end_time)
            except queue.Empty:
                continue
            except Exception as e:
                raise TranscribingError(f'TranscribingError occurred when transcribing: {e}') from e

    def _upload_speech_transcription(self, speaker, text, chunk_start_time, chunk_end_time):
        """Upload speech transcription to InfluxDB.

        Args:
            speaker: recognized speaker name of the transcribed chunk
            text: transcribed text of the chunk
            chunk_start_time: start time of the chunk
            chunk_end_time:  end time of the chunk
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

    def _finalize_speech_recognition_with_sp(self, speakers, similarities, durations, signals) -> tuple:
        """Finalize the speaker recognition results of separated signals to handle edge cases.

        Args:
            speakers: list of recognized speakers
            similarities: list of similarity values
            durations: list of audio durations
            signals: list of separated signals

        Returns:
            post-processed speakers, similarities, durations, signals
        """
        # If there's only one speaker, keep it as is
        if len(speakers) == 1:
            return speakers, similarities, durations, signals

        # If both speakers are the same, keep the one with the highest similarity
        if speakers[0] == speakers[1]:
            max_similarity_index = 0 if similarities[0] > similarities[1] else 1
            return [speakers[max_similarity_index]], [similarities[max_similarity_index]], [
                durations[max_similarity_index]], [signals[max_similarity_index]]

        # If both are real speakers, keep them as is
        if all(speaker not in ['silent', 'unknown'] for speaker in speakers):
            return speakers, similarities, durations, signals

        # If there's at least one real speaker, keep only the real ones
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

    def _publish_speaker_recognition(self, record_start_time, recognize_start_time, speakers, similarities, durations):
        """Log and publish base speaker recognition results in JSON string to Redis on bucket channel.

        Args:
            record_start_time: record start time of the segment
            recognize_start_time: start time of the recognition process
            speakers: recognized speakers
            similarities: similarity values
            durations: audio durations
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

    def _transcribe(self, frames):
        """Transcribe audio frames to text.

        Args:
            frames: audio frames to be transcribed

        Returns:
            transcribed text
        """
        text = request_speech_transcription(frames, f'{self.base_type.lower()}_{self.id}', self.sp,
                                            self.speech_transcriber_url)
        return text

    def _separate_speech(self, segment_audio_path) -> list:
        """Separate speech from the overlapped segment audio.

        Args:
            segment_audio_path:

        Returns:
            separated speech signals
        """
        separated_result = request_speech_separation(segment_audio_path, f'{self.base_type.lower()}_{self.id}',
                                                     self.speech_separator_url)
        result = [base64.b64decode(encoded_bytes_stream) for encoded_bytes_stream in separated_result]
        return result

    def post_process_audio(self, input_path: str, inplace: int) -> Union[str, None]:
        """Apply both NR and VAD processing to audio file.


        Args:
            input_path: input audio file path
            inplace: whether to overwrite the input file when applying vad

        Returns:
            processed audio file path
        """
        self.apply_nr(input_path)
        return self.apply_vad(input_path, inplace)

    def apply_vad(self, input_path: str, inplace: int) -> Union[str, None]:
        """Apply voice activity detection to audio file.

        Args:
            input_path: input audio file path
            inplace: whether to overwrite the input file

        Returns:
            processed audio file path
        """
        if not self.vad:
            return input_path
        return request_voice_activity_detection(input_path, f'{self.base_type.lower()}_{self.id}', inplace,
                                                self.vad_url)

    def apply_nr(self, input_path: str) -> str:
        """Apply noise reduction to audio file.

        Args:
            input_path: input audio file path

        Returns:
            processed audio file path
        """
        if not self.nr:
            return input_path
        request_speech_enhancement(input_path, f'{self.base_type.lower()}_{self.id}', self.speech_enhancer_url)

    def _store_audio(self, source_path, dest_path):
        """Store or remove audio files based on the store flag.

        Args:
            source_path: original audio file path
            dest_path: destination audio file path
        """
        if self.store:
            shutil.move(source_path, dest_path)
        else:
            os.remove(source_path)

    @staticmethod
    def filter_real_speakers(speakers, similarities, durations, texts) -> tuple:
        """Filter out 'silent' and 'unknown' from speakers and their associated similarities and durations.

        Args:
            speakers: list of speakers
            similarities: list of similarities
            durations: list of durations
            texts: list of texts

        Returns:
            list of real_speakers, real_similarities, real_durations, real_texts
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
        """Warm up the resampler by generating a short segment of silence and performing the resampling operation, to
        avoid delay in the first resampling operation.

        Args:
            sample_rate_original: original sample rate, default to 44100
            sample_rate_target: target sample rate, default to 16000
        """
        dummy_audio = np.zeros(sample_rate_original)
        _ = librosa.resample(dummy_audio, orig_sr=sample_rate_original, target_sr=sample_rate_target)
        print("Resampler has been warmed up.")

    @staticmethod
    def recording_prompt(seconds: float):
        """Reading prompt for registering."""
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
