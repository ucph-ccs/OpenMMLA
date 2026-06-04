import base64
import gc
import json
import os
import queue
import re
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
from openmmla.utils.audio.files import format_wav
from openmmla.utils.audio.auga import normalize_decibel, apply_gain
from openmmla.utils.audio.augf import resample_audio
from openmmla.utils.audio.io import read_bytes_from_wav, write_bytes_to_wav
from openmmla.utils.audio.properties import get_energy_level, calculate_audio_duration
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, runtime_pipeline_artifact_dir
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_session, get_id, get_interactive_files, get_rtmp_url, show_error_and_pause
from openmmla.utils.logger import get_logger
from openmmla.utils.ports import free_port
from openmmla.utils.requests import resolve_url
from openmmla.analytics.realtime.status_engine import normalize_asr_scope
from .audio_recognizer import AudioRecognizer
from .enums import BLUE, ENDC, GREEN, PURPLE, GREY
from .input import get_base_type, get_function_base, get_name, get_base_mode, get_input_device_index, get_channel_selection, get_edit_speaker_options, get_speaker_selection, get_speaker_deletion


def _resolve_speaker_verification(value, asr_scope: str) -> bool:
    """resolve the speaker verifier setting from config.

    auto follows the ASR attribution scope: participant-level ASR verifies speakers,
    group-level ASR skips speaker profile verification by default.
    """
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return asr_scope == "participant"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def start_asr_base(project_dir: str, config_path: str, mode: str = 'full', store: bool = True,
                   vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False,
                   hsr: bool = True, session_id: str | None = None):
    """Start ASR Base with restart capability.
    
    Args:
        project_dir: Path to the project directory
        config_path: Path to the configuration file
        mode: Operating mode ('record', 'recognize', or 'full')
        store: Whether to store audio files
        vad: Whether to apply Voice Activity Detection
        nr: Whether to use denoiser to enhance speech
        tr: Whether to transcribe speech to text
        sp: Whether to do speech separation for overlapped segments
        hsr: Whether to apply Half-Scaled Recognition at speaker boundaries
    """
    # restart loop - allows restarting the entire process
    while True:
        try:
            asr_base = ASRBase(project_dir=project_dir, config_path=config_path, mode=mode, 
                              vad=vad, nr=nr, tr=tr, sp=sp, store=store, hsr=hsr,
                              session_id=session_id)
            asr_base.run()
        except KeyboardInterrupt as e:
            if "Exit" in str(e):
                print("\n👋 Goodbye!")
                break  # Exit completely when 'q' is pressed
            else:
                print("\n🔄 Restarting ASR Base...")
                continue  # Restart on Ctrl+C during runtime
        except Exception as e:
            show_error_and_pause(e, "restart ASR Base")
            print("\n🔄 Restarting ASR Base...")
            continue


class ASRBase(Base):
    """ASRBase class for automatic speech recognition with speaker diarization."""

    logger = get_logger(f'asr-base')

    def __init__(self, project_dir: str | None, config_path: str, mode: str = 'record', store: bool = True,
                 vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False,
                 hsr: bool = True, session_id: str | None = None):
        """Initialize the ASRBase class.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            mode: operating mode, 'record', 'recognize', or 'full' (default: 'record')
            store: whether to store audio files (default: True)
            vad: whether to apply Voice Activity Detection (default: True)
            nr: whether to apply noise reduction (default: True)
            tr: whether to transcribe speech to text (default: True)
            sp: whether to perform speech separation (default: False)
            hsr: whether to apply Half-Scaled Recognition at speaker boundaries (default: True)
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        # base specific parameters
        self.mode = mode
        self.store = store
        self.vad = vad
        self.nr = nr
        self.tr = tr
        self.sp = sp
        self.hsr = hsr
        self.launch_session_id = session_id

        # runtime attributes
        self.session_id = None
        self.last_speaker = None
        self.audio_dir = None
        self.audio_queue = None
        self.transcription_queue = None
        self.speaker_frames_dict = None
        self.stop_event = threading.Event()
        self.threads = []
        self.selected_speakers = None
        self.asr_scope = "participant"
        self.speaker_verification = True
        self.group_speaker_id = "group"

        self.base_type = get_base_type(self.config)
        self.id = get_id()
        print(f"\033]0;ASR Base {self.base_type} {self.id} \007")

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _setup_yaml(self):
        """Load and assign configuration parameters from the YAML configuration file.

        Read various settings such as durations, thresholds, and service URLs required for the audio processing pipeline.
        """
        base_config = self.config['Base'][self.base_type]
        asr_server_config = self.config['Server']['asr']
        analytics_config = self.config.get('Analytics', {})
        self.asr_scope = normalize_asr_scope(analytics_config.get('asr_scope'))
        self.speaker_verification = _resolve_speaker_verification(
            analytics_config.get('speaker_verification', 'auto'),
            self.asr_scope,
        )
        self.group_speaker_id = str(
            analytics_config.get('group_id')
            or analytics_config.get('group_speaker_id')
            or 'group'
        )

        self.register_duration = int(base_config['register_duration'])
        self.recognize_duration = int(base_config['recognize_sp_duration']) if self.sp else int(
            base_config['recognize_duration'])
        self.rms_threshold = int(base_config['rms_threshold'])
        self.rms_peak_threshold = int(base_config['rms_peak_threshold'])
        self.threshold = float(base_config['recognize_sp_threshold']) if self.sp else float(
            base_config['recognize_threshold'])
        self.keep_threshold = float(base_config['keep_sp_threshold']) if self.sp else float(
            base_config['keep_threshold'])
        self.update_threshold = float(base_config.get('update_threshold', 0.6))
        self.gain = float(base_config['gain'])
        self.score_amplified = bool(base_config.get('score_amplified', False))

        self.source = base_config['source']
        self.stream_kwargs = base_config['stream_kwargs']

        self.speech_transcriber_url = resolve_url(asr_server_config['speech_transcriber'])
        self.speech_separator_url = resolve_url(asr_server_config['speech_separator'])
        self.speech_enhancer_url = resolve_url(asr_server_config['speech_enhancer'])
        self.vad_url = resolve_url(asr_server_config['voice_activity_detector'])

        source_list = ['udp', 'tcp', 'pyaudio', 'rtmp', 'lsl', 'file']
        if self.source not in source_list:
            raise ValueError(f'Unknown source {self.source}, must be one of {source_list}')

        # set port number for 'udp/tcp'
        if self.source in ['udp', 'tcp']:
            self.port_offset = int(base_config.get('port_offset', 0))
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
            device_info_list = []

            # show all available devices regardless of channel count
            for i in range(0, num_devices):
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                max_input_channels = device_info.get('maxInputChannels')
                if max_input_channels > 0:  # Only show devices that can input audio
                    available_indexes.append(i)
                    device_info_list.append(device_info)

            self.input_device_index = get_input_device_index(available_indexes, device_info_list)
            self.stream_kwargs['input_device_index'] = self.input_device_index
            device_info = p.get_device_info_by_host_api_device_index(0, self.input_device_index)
            
            # update channels based on selected device capabilities
            device_channels = device_info.get('maxInputChannels', 1)
            self.stream_kwargs['channels'] = device_channels
            
            # allow channel selection if device has multiple channels
            self.stream_kwargs['channel_select'] = get_channel_selection(device_info) if device_channels > 1 else None
            self.logger.info(f"Selected device: {device_info.get('name')} with {device_channels} channels")
            self.logger.info(f"Selected channel option: {self.stream_kwargs['channel_select']}")
            p.terminate()

        # set url for 'rtmp'
        elif self.source == 'rtmp':
            from openmmla.utils.constants import get_stream_urls
            rtmp_urls = get_stream_urls(self.config, "rtmp")
            if not rtmp_urls:
                raise ValueError("No RTMP streams found in Streams (or legacy RTMP) config section.")
            self.url = get_rtmp_url(rtmp_urls)
            self.stream_kwargs['url'] = self.url
            self.logger.info(f"Using RTMP URL: {self.url}")

        # set file_path for 'file'
        elif self.source == 'file':
            if 'file_dir' not in base_config:
                # default to project directory if not specified
                file_dir = self.project_dir
                self.logger.info(f"No file_dir specified in config, using project directory: {file_dir}")
            else:
                file_dir = base_config['file_dir']
                if not os.path.isabs(file_dir):
                    file_dir = os.path.join(self.project_dir, file_dir)
                
                if not os.path.exists(file_dir):
                    # fallback to project directory if specified directory doesn't exist
                    self.logger.warning(f"Specified file directory does not exist: {file_dir}")
                    file_dir = self.project_dir
                    self.logger.info(f"Using project directory instead: {file_dir}")
            
            # use interactive file browser to select file and get initial_sync_time
            audio_extensions = ('.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.wma')
            print(f"\n{PURPLE}📁 Select Audio File{ENDC}")
            print(f"{GREY}Choose audio file for ASR Base{ENDC}")
            file_path, initial_sync_time = get_interactive_files(file_dir, file_extensions=audio_extensions, multiple=False, sync_input=True)
            self.initial_sync_time = initial_sync_time
            
            self.stream_kwargs['file_path'] = file_path
            self.logger.info(f"Using audio file: {file_path}")
            self.logger.info(f"Using initial_sync_time: {self.initial_sync_time}")
        
    def _setup_directories(self):
        """Create and set up the necessary directories for runtime operations.

        Create directories for runtime files, temporary files, speaker profiles, and audio databases.
        Ensures that the required folder structure exists.
        """
        self.logger_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'asr-base', 'logger'))
        self.runtime_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'asr-base', 'real-time', 'runtime'))
        self.temp_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'asr-base', 'temp'))
        self.profiles_dir = os.fspath(runtime_pipeline_artifact_dir(self.project_dir, 'asr-base', 'profiles'))

        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)

    def _setup_objects(self):
        """Initialize external service clients and internal processing objects.

        Set up the clients for InfluxDB, Redis, and MQTT, warms up the audio resampler, and initializes the
        AudioRecognizer and AudioStream.
        """
        self.influx_client = InfluxDBClientWrapper(self.config_path)
        self.mongo_client = MongoDBClientWrapper(self.config_path)
        self.redis_client = RedisClientWrapper(self.config_path)
        self.mqtt_client = MQTTClientWrapper(self.config_path)
        self.warm_up_resampler()
        self.audio_recognizer = AudioRecognizer(config_path=self.config_path, profiles_dir=self.profiles_dir, store=self.store, selected_speakers=self.selected_speakers)
        self.audio_stream = AudioStream(source=self.source, **self.stream_kwargs)

    def _clean_up(self):
        """Clean up runtime variables and free memory."""
        if self.threads:
            self._stop_threads()
        self._clear_threads()
        self.mqtt_client.loop_stop()
        if self.audio_stream:
            self.audio_stream.stop()
        self.session_id = None
        self.last_speaker = None
        self.audio_dir = None
        self.audio_queue = None
        self.transcription_queue = None
        self.speaker_frames_dict = None
        gc.collect()

    def run(self):
        """Run the ASR base.

        Continuously prompts the user for input until termination.
        """
        func_map = {1: self._edit_speakers, 2: self._start_recognition, 3: self._switch_mode, 4: self._reset}
        while True:
            try:
                select_fun = get_function_base(self.id, self.mode)
                func_map.get(select_fun, lambda: self.logger.warning("Invalid option"))()
            except KeyboardInterrupt as e:
                if "Exit" in str(e):
                    # 'q' was pressed in top-level menu - re-raise to be caught by outer restart loop
                    raise
                else:
                    # ctrl+c during runtime or 'q' in lower-level menu - log and continue
                    self.logger.warning(f"During running the ASR base, catch: {e}, Come back to main menu.", exc_info=True)
            except Exception as e:
                self.logger.warning(f"During running the ASR base, catch: {e}, Come back to the main menu.", exc_info=True)
                show_error_and_pause(e, "return to the ASR Base menu")
            finally:
                self._clean_up()

    def _edit_speakers(self):
        """Edit speaker profiles - register, select/deselect, or delete speakers."""
        print("------------------------------------------------")
        
        # get available speakers from profiles directory
        available_speakers = []
        if os.path.exists(self.profiles_dir):
            available_speakers = [d for d in os.listdir(self.profiles_dir) 
                                if os.path.isdir(os.path.join(self.profiles_dir, d)) and not d.startswith('.')]
        
        # get currently selected speakers (default to all if None)
        if self.selected_speakers is None:
            self.selected_speakers = available_speakers.copy()
        
        while True:
            try:
                option = get_edit_speaker_options(available_speakers, self.selected_speakers)
                
                if option == 0:  # Register from stream
                    self._register_speaker_from_stream()
                    # refresh available speakers
                    available_speakers = []
                    if os.path.exists(self.profiles_dir):
                        available_speakers = [d for d in os.listdir(self.profiles_dir) 
                                            if os.path.isdir(os.path.join(self.profiles_dir, d)) and not d.startswith('.')]
                    # add new speaker to selected list if not already there
                    if available_speakers and available_speakers[-1] not in self.selected_speakers:
                        self.selected_speakers.append(available_speakers[-1])
                
                elif option == 1:  # Register from files
                    self._register_speaker_from_files()
                    # refresh available speakers
                    available_speakers = []
                    if os.path.exists(self.profiles_dir):
                        available_speakers = [d for d in os.listdir(self.profiles_dir) 
                                            if os.path.isdir(os.path.join(self.profiles_dir, d)) and not d.startswith('.')]
                    # add new speaker to selected list if not already there
                    if available_speakers and available_speakers[-1] not in self.selected_speakers:
                        self.selected_speakers.append(available_speakers[-1])
                
                elif option == 2:  # Select/deselect speakers
                    if not available_speakers:
                        print(f"{GREY}No speakers available for selection.{ENDC}")
                        continue
                    self.selected_speakers = get_speaker_selection(available_speakers, self.selected_speakers)
                    # update recognizer with new selection
                    self.audio_recognizer.reset_profiles(self.profiles_dir, self.selected_speakers)
                
                elif option == 3:  # Delete speaker
                    if not available_speakers:
                        print(f"{GREY}No speakers available for deletion.{ENDC}")
                        continue
                    speakers_to_delete = get_speaker_deletion(available_speakers)
                    if speakers_to_delete:
                        # confirm deletion
                        if len(speakers_to_delete) == 1:
                            confirm_msg = f"Are you sure you want to delete speaker '{speakers_to_delete[0]}'? (y/N): "
                        else:
                            speakers_list = "', '".join(speakers_to_delete)
                            confirm_msg = f"Are you sure you want to delete speakers '{speakers_list}'? (y/N): "
                        
                        confirm = input(confirm_msg).strip().lower()
                        if confirm == 'y':
                            for speaker_to_delete in speakers_to_delete:
                                self.audio_recognizer.delete_speaker_profile(speaker_to_delete)
                                # remove from available and selected lists
                                if speaker_to_delete in available_speakers:
                                    available_speakers.remove(speaker_to_delete)
                                if speaker_to_delete in self.selected_speakers:
                                    self.selected_speakers.remove(speaker_to_delete)
                                print(f"{GREEN}Speaker '{speaker_to_delete}' deleted successfully.{ENDC}")
                        else:
                            print("Deletion cancelled.")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.warning(f"Error in speaker editing: {e}")

    def _register_speaker_from_stream(self):
        """Register a new speaker profile from audio stream."""
        if self.source == 'file':
            self.logger.info("Cannot register from stream when source is 'file'. Please use 'Register from Files' option.")
            return

        output_path = os.path.join(self.temp_dir, f'{self.base_type}_{self.id}_register.wav')
        self.audio_stream = AudioStream(source=self.source, **self.stream_kwargs)
        self.recording_prompt(self.register_duration)
        
        self.audio_stream.start()
        audio_frame = self.audio_stream.read(duration=self.register_duration, latest=True)
        self.audio_stream.stop()
        write_frame_to_wav(output_path, audio_frame)

        apply_gain(output_path, self.gain)
        audio_path = self._audio_preprocessing(output_path, 1)

        if audio_path is None:
            self.logger.info(
                "The recorded audio file is not long enough or audio pre-processing failed, please record again.")
            return

        name = get_name()
        if name == '':
            self.logger.info('Empty name, skip the registering process.')
            return

        self.audio_recognizer.register(audio_path, name)
        self.logger.info(f"Speaker '{name}' has been successfully registered from stream!")

    def _register_speaker_from_files(self):
        """Register a new speaker profile from reference audio files."""
        print(f"\n{PURPLE}📁 Select Reference Audio Files{ENDC}")
        print(f"{GREY}Choose reference audio files for speaker registration (multiple selection supported){ENDC}")
        
        try:
            # use interactive file browser to select reference files
            audio_extensions = ('.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.wma')
            reference_files = get_interactive_files(self.project_dir, file_extensions=audio_extensions, multiple=True, sync_input=False)
            
            if not reference_files:
                self.logger.info("No files selected, registration cancelled.")
                return
            
            # get speaker name
            name = get_name()
            if name == '':
                self.logger.info('Empty name, skip the registering process.')
                return
            
            # process each reference file
            processed_files = []
            for i, file_path in enumerate(reference_files):
                try:
                    temp_file = os.path.join(self.temp_dir, f'{self.base_type}_{self.id}_ref_{i}.wav')
                    formatted_file = os.path.join(self.temp_dir, f'{self.base_type}_{self.id}_formatted_{i}.wav')
                    format_wav(file_path, formatted_file)
                    
                    # apply gain and preprocessing
                    processed_path = self._audio_preprocessing(formatted_file, 1)
                    
                    if processed_path is None:
                        self.logger.warning(f"Preprocessing failed for {os.path.basename(file_path)}, skipping.")
                        continue
                    
                    # copy processed file to temp location
                    shutil.copy2(processed_path, temp_file)
                    processed_files.append(temp_file)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {os.path.basename(file_path)}: {e}")
                    continue
            
            if not processed_files:
                self.logger.error("No files could be processed successfully.")
                return
            
            # register speaker using the first processed file (AudioRecognizer will handle multiple files internally)
            # for now, we'll register each file separately - this could be enhanced to batch process
            for i, processed_file in enumerate(processed_files):
                try:
                    if i == 0:
                        # first file - register as new speaker
                        self.audio_recognizer.register(processed_file, name)
                    else:
                        # additional files - add to existing speaker profile
                        self.audio_recognizer.register(processed_file, name)
                except Exception as e:
                    self.logger.warning(f"Error registering file {i+1}: {e}")
            
            self.logger.info(f"Speaker '{name}' has been successfully registered from {len(processed_files)} reference files!")
            
        except Exception as e:
            self.logger.error(f"Error in file-based registration: {e}")

    def _start_recognition(self, session_id: str | None = None):
        """Start the real-time voice recognition process.

        Set up directories, queues, and MQTT communication before creating threads for:
          - Continuous recording.
          - Loading and queuing pre-recorded files (if in 'recognize' mode).
          - Continuous recognition (with or without speech separation).
          - Continuous transcription (if enabled).
          - Listening for stop signals.

        Args:
            session_id: The bucket name for storing recognition results. If not provided, it is obtained interactively.
        """
        # check if any speakers are selected for participant-level recognition
        if self.speaker_verification and (not self.selected_speakers or len(self.selected_speakers) == 0):
            print("------------------------------------------------")
            if self.mode in ['full', 'recognize']:
                self.logger.info("No speakers selected for recognition. Please register and select speaker profiles or switch to 'record' mode.")
                return
            elif self.mode == 'record':
                self.logger.warning("No speakers selected. Recording will continue without speaker recognition.")
        elif self.speaker_verification and self.mode in ['full', 'recognize'] and len(self.audio_recognizer.speaker_names) == 0:
            print("------------------------------------------------")
            self.logger.info("Audio database is empty, please register speaker profiles or either switch the mode to 'record'.")
            return
        elif self.speaker_verification and self.mode == 'record' and len(self.audio_recognizer.speaker_names) == 0:
            print("------------------------------------------------")
            self.logger.warning("Audio database is empty. Recording will continue without speaker recognition.")
        
        # show selected speakers and ask for confirmation
        print("------------------------------------------------")
        if self.speaker_verification:
            print(f"{PURPLE}Selected speakers for recognition:{ENDC}")
            if self.selected_speakers and len(self.selected_speakers) > 0:
                for i, speaker in enumerate(self.selected_speakers, 1):
                    print(f"  {i}. {speaker}")
            else:
                print(f"  {GREY}No speakers selected{ENDC}")
            print(f"\n{GREEN}Total speakers: {len(self.selected_speakers) if self.selected_speakers else 0}{ENDC}")
        else:
            print(f"{PURPLE}Speaker verification disabled.{ENDC}")
            print(f"{GREEN}ASR chunks will be attributed at group scope.{ENDC}")
        
        # select or create bucket
        launch_session_id = session_id or self.launch_session_id
        self.session_id = select_or_create_session(self.mongo_client) if not launch_session_id else launch_session_id
        self._resolve_group_speaker_id()
        self._create_bucket_logger()
        self._create_speaker_profile_snapshot()

        # reset attributes
        self.last_speaker = None
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.speaker_frames_dict = {}

        self._prepare_directories()
        self._listen_for_start_signal()

        # reinitialize mqtt client
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        # create threads based on the operating mode
        if self.mode in ['record', 'full']:
            self._create_thread(self._continuous_recording)
        if self.mode == 'recognize':
            self._create_thread(self._enqueue_recorded_files)
        if self.mode in ['recognize', 'full']:
            if self.speaker_verification:
                recognition_task = self._continuous_recognizing_sp if self.sp else self._continuous_recognizing
            else:
                recognition_task = self._continuous_recognizing
            self._create_thread(recognition_task)
            if self.tr:
                self._create_thread(self._continuous_transcribing)
        self._create_thread(self._listen_for_stop_signal)

        # start and join threads, handling exceptions if they occur
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

    def _resolve_group_speaker_id(self):
        """Prefer the selected session group id for group-level ASR attribution."""
        if self.asr_scope != "group" or not self.session_id:
            return
        try:
            session = self.mongo_client.get_session(self.session_id) or {}
            group_id = str(session.get("group_id") or "").strip()
            if group_id:
                self.group_speaker_id = group_id
        except Exception as e:
            self.logger.warning(f"Could not resolve group id for ASR attribution: {e}")
        self.logger.info(f"Group-level ASR speaker id: {self.group_speaker_id}")

    def _create_bucket_logger(self):
        """Create a logger for a bucket."""
        self.bucket_logger_dir = os.fspath(
            pipeline_section_dir(self.project_dir, self.session_id, 'asr-base', 'logger')
        )
        os.makedirs(self.bucket_logger_dir, exist_ok=True)
        copy_config_snapshot(self.config_path, self.project_dir, self.session_id, 'asr-base')
        self.logger = get_logger(f'asr-base-{self.session_id}',
                                 os.path.join(self.bucket_logger_dir,
                                              f'asr_{self.base_type}_{self.id}.log'))

    def _create_speaker_profile_snapshot(self):
        """Create a snapshot of the speaker profiles used in the current session.
        
        This copies the profile files from the base's audio_db to a bucket-specific
        folder to record exactly which speaker profiles were used in this session.
        The snapshot is stored in the runtime directory under the bucket folder.
        """
        if not self.session_id:
            return
        if not self.speaker_verification or not self.selected_speakers:
            self.logger.info("Skipping speaker profile snapshot because speaker verification is disabled.")
            return

        runtime_root = pipeline_section_dir(self.project_dir, self.session_id, 'asr-base', 'real-time') / 'runtime'
        runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = os.fspath(runtime_root)
        snapshot_dir = os.path.join(self.runtime_dir, f'{self.base_type}_{self.id}', 'profiles')
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)  # Clear any existing snapshot
        else:
            os.makedirs(snapshot_dir)

        self.logger.info(f"Creating snapshot of speaker profiles for bucket '{self.session_id}'")
        try:
            # copy only selected speaker profiles
            for speaker_name in self.selected_speakers:
                speaker_source_dir = os.path.join(self.profiles_dir, speaker_name)
                speaker_dest_dir = os.path.join(snapshot_dir, speaker_name)
                if os.path.exists(speaker_source_dir):
                    shutil.copytree(speaker_source_dir, speaker_dest_dir)
        except Exception as e:
            self.logger.warning(f"Error creating speaker profile snapshot: {e}")

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

        # process any remaining audio chunks before cleanup
        self._process_final_chunks()
        
        current_bucket = self.session_id  # assign bucket name before cleaning up
        self._clean_up()
        if isinstance(e, RecordingError):
            self.logger.info("Restarting recognizing service.")
            self._start_recognition(current_bucket)

    def _reset(self):
        """Reset the ASR base.

        Reinitialize the ASR base by calling the constructor with the current configuration,
        logs the reset status, and performs garbage collection.
        """
        self.__init__(project_dir=self.project_dir, config_path=self.config_path, mode=self.mode,
                      vad=self.vad, nr=self.nr, tr=self.tr, sp=self.sp, store=self.store, hsr=self.hsr)
        self.logger.info(f"Profiles directory reset to {self.profiles_dir}")
        gc.collect()

    def _switch_mode(self):
        """Switch the operating mode between 'record', 'recognize' and 'full'."""
        self.mode = get_base_mode()
        self.logger.info(f"Switched to {self.mode} mode.")

    def _continuous_recording(self):
        """Continuously record audio from the audio stream and enqueue it for processing.

        Depending on the operating mode:
          - In 'record' mode, writes recorded frames to a file.
          - In 'full' mode, puts the audio frame bytes into the audio queue.

        Raises:
            RecordingError: If an error occurs during the recording process.
        """
        # handle file source differently - files need sequential time-based reading
        if self.source == 'file':
            self._continuous_file_reading()
            return
            
        first_time = True
        sub_dir = 'records' if self.mode == 'record' else 'temp'
        self.audio_stream = AudioStream(source=self.source, **self.stream_kwargs)
        self.audio_stream.start()

        while not self.stop_event.is_set():
            try:
                audio_frame = self.audio_stream.read(duration=self.recognize_duration, latest=first_time)
                first_time = False
                frames = audio_frame.to_bytes()
                acquired_time = audio_frame.timestamp
                output_path = os.path.join(self.audio_dir, sub_dir,
                                           f'{self.base_type}_{self.id}_record_{acquired_time:.4f}.wav')
                if self.mode == 'record':
                    write_frame_to_wav(output_path, audio_frame)
                    print(f"{BLUE}[Recording]{ENDC} {os.path.basename(output_path)} {len(frames)} frames")
                else:
                    self.audio_queue.put((output_path, frames))
            except Exception as e:
                raise RecordingError(f'RecordingError occurred when continuous recording: {e}') from e

    def _continuous_file_reading(self):
        """Read audio segments from a file sequentially to simulate continuous recording.
        
        This method reads segments from the audio file in sequence, mimicking the behavior
        of continuous recording but from a pre-recorded file.
        
        Raises:
            RecordingError: If an error occurs during the file reading process.
        """
        try:
            sub_dir = 'records' if self.mode == 'record' else 'temp'
            self.audio_stream = AudioStream(source=self.source, **self.stream_kwargs)
            self.audio_stream.start()

            # extract timestamp from audio filename (format: <prefix>_<timestamp>.<affix>)
            filename = os.path.basename(self.stream_kwargs['file_path'])
            match = re.search(r'_(\d+(?:\.\d+)?)\.', filename)
            file_start_time = float(match.group(1))
            timestamp_offset = file_start_time
            
            file_duration = len(self.audio_stream.file_data) / self.audio_stream.file_sample_rate
            start_time = (self.initial_sync_time - file_start_time)
            self.logger.info(f"file_duration: {file_duration}, start_time: {start_time}")

            while not self.stop_event.is_set():
                try:
                    if start_time < file_duration:
                        # read segment from file at current time
                        audio_frame = self.audio_stream.read(
                            start_time=start_time,
                            duration=self.recognize_duration
                        )

                        if audio_frame is None:
                            self.logger.warning(f"Audio file {os.path.basename(output_path)} at time {start_time} not readable.")
                            continue

                        frames = audio_frame.to_bytes()
                        acquired_time = start_time + timestamp_offset
                        output_path = os.path.join(self.audio_dir, sub_dir, f'{self.base_type}_{self.id}_record_{acquired_time:.4f}.wav')

                        if self.mode == 'record':
                            write_frame_to_wav(output_path, audio_frame)
                            print(f"{BLUE}[Recording]{ENDC} {os.path.basename(output_path)} {len(frames)} frames")
                        else:
                            self.audio_queue.put((output_path, frames))

                        # advance to next segment
                        start_time += self.recognize_duration
                    else:
                        self.logger.info(f"Reach the end of the file, start time {start_time}.")
                        break
                    
                except Exception as e:
                    raise RecordingError(f'RecordingError occurred when reading file segment at {start_time}s: {e}') from e
                
        except Exception as e:
            raise RecordingError(f'RecordingError occurred when continuous file reading: {e}') from e

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
                segment_start_time = float(os.path.basename(segment_audio_path).split('_')[-1][:-4])
                recognize_start_time = time.time()
                write_bytes_to_wav(segment_audio_path, frames)  # default: 16000 Hz, 16-bit, mono

                # audio pre-processing
                apply_gain(segment_audio_path, self.gain)
                processed_audio_path = self._audio_preprocessing(segment_audio_path, inplace=1)

                # evaluate energy levels for quality check
                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                if processed_audio_path and rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold:
                    speaker = 'unknown'
                else:
                    speaker = 'silent'

                duration = self.recognize_duration
                similarity = 0

                if speaker == 'unknown' and not self.speaker_verification:
                    speaker = self.group_speaker_id
                    similarity = 1.0
                    duration = calculate_audio_duration(segment_audio_path)
                elif speaker == 'unknown':  # voice detected
                    normalize_decibel(segment_audio_path, rms_level=-20)
                    name, similarity = self.audio_recognizer.recognize(segment_audio_path,
                                                                       update_threshold=self.update_threshold)
                    duration = calculate_audio_duration(segment_audio_path)

                    if similarity > self.threshold:
                        speaker = name
                        if self.score_amplified:
                            energy_level_factor = np.log(rms_value) / np.log(self.rms_threshold)
                            similarity = min(similarity * energy_level_factor, 1)
                    else:
                        c1 = rms_value + peak_value
                        c2 = self.rms_threshold + self.rms_peak_threshold
                        ratio = c1 / (c1 + c2)
                        similarity = self.threshold * ratio

                self._assemble_chunk_with_hsr(speaker, segment_start_time, frames)
                self._publish_recognition(segment_start_time, recognize_start_time, [speaker],
                                          [np.round(np.float64(similarity), 4)], [duration])

                if self.store:
                    shutil.move(segment_audio_path,
                                os.path.join(self.audio_dir, 'segments', f'{speaker}_{segment_start_time}.wav'))
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
                segment_start_time = float(os.path.basename(segment_audio_path).split('_')[-1][:-4])
                recognize_start_time = time.time()
                write_bytes_to_wav(segment_audio_path, frames)

                # audio pre-processing
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

                    # recognize separated audio streams
                    for i, signal in enumerate(sp_result):
                        save_file = f'{segment_audio_path[:-4]}_spk{i}.wav'
                        sf.write(save_file, np.frombuffer(signal, dtype=np.int16), 8000)
                        processed_save_file = self._apply_vad(save_file, inplace=1)

                        # skip file if vad fails, removing it immediately.
                        if not processed_save_file:
                            os.remove(save_file)
                            continue

                        normalize_decibel(save_file, rms_level=-20)
                        temp_name, temp_similarity = self.audio_recognizer.recognize(save_file,
                                                                                     update_threshold=self.update_threshold)

                        # if a better result is found, update best info and remove any old file.
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

                    if similarity > self.threshold and self.score_amplified:
                        energy_level_factor = np.log(rms_value) / np.log(self.rms_threshold)
                        similarity = min(similarity * energy_level_factor, 1)

                if speaker == 'unknown' and similarity == 0:
                    speaker = 'silent'

                resampled_segment_bytes = read_bytes_from_wav(segment_audio_path)
                self._assemble_chunk_with_hsr(speaker, segment_start_time, resampled_segment_bytes, best_separate_frames)
                self._publish_recognition(segment_start_time, recognize_start_time, [speaker],
                                          [np.round(np.float64(similarity), 4)], [duration])

                if self.store:
                    if best_separate_path:
                        shutil.move(best_separate_path, os.path.join(self.audio_dir, 'separations',
                                                                     f'{speaker}_{segment_start_time}_spk.wav'))
                    shutil.move(segment_audio_path,
                                os.path.join(self.audio_dir, 'segments', f'{speaker}_{segment_start_time}.wav'))
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
        frame_rate = 8000 if self.sp and self.speaker_verification else 16000
        while not self.stop_event.is_set():
            try:
                frames, speaker, chunk_start_time, chunk_end_time = self.transcription_queue.get(timeout=2)
                transcribe_result = self._transcribe(frames, frame_rate)
                self._upload_transcription(speaker, transcribe_result, chunk_start_time, chunk_end_time)
            except queue.Empty:
                continue
            except Exception as e:
                raise TranscribingError(f'TranscribingError occurred when transcribing: {e}') from e

    def _assemble_chunk_with_hsr(self, speaker: str, segment_start_time: float, origin_frames: bytes,
                                 separate_frames: bytes | None = None):
        """Assemble and process audio chunks with half-scaled recognition (HSR) at speaker boundaries.

        If the current recognized speaker matches the previous speaker, appends the audio frames.
        Otherwise, performs HSR by processing half-segments before and after the speaker change,
        updating the internal speaker frames dictionary accordingly. Also, adds transcribed chunks
        to the transcription queue if applicable.

        Args:
            speaker: Recognized speaker of the current segment.
            segment_start_time: Start time of the current segment.
            origin_frames: Original audio frames of the current segment.
            separate_frames (Optional): Speech separated frames from the current segment.
        """
        frames = separate_frames if separate_frames else origin_frames
        fr = 8000 if self.sp and self.speaker_verification else 16000

        if not self.last_speaker:
            self.speaker_frames_dict[speaker] = (segment_start_time, frames)
        else:
            if speaker == self.last_speaker:
                chunk_start_time, last_speaker_frames = self.speaker_frames_dict[speaker]
                last_speaker_frames += frames
                self.speaker_frames_dict[speaker] = (chunk_start_time, last_speaker_frames)
            else:
                chunk_start_time, chunk_frames = self.speaker_frames_dict.pop(self.last_speaker)
                chunk_end_time = segment_start_time

                # perform half-scaled recognition on speaker turn border if hsr is enabled
                if chunk_frames and self.hsr and self.speaker_verification:
                    self.logger.info(
                        f"Performing half-scaled recognition on speaker turn border for {self.last_speaker} and {speaker}")
                    left_temp_path = os.path.join(self.audio_dir, 'temp', f'{self.base_type}_{self.id}_left_temp.wav')
                    right_temp_path = os.path.join(self.audio_dir, 'temp', f'{self.base_type}_{self.id}_right_temp.wav')
                    number_frames = int(len(frames) / 2)
                    candidates = [self.last_speaker, speaker]
                    write_bytes_to_wav(left_temp_path, chunk_frames[-number_frames:], framerate=fr)
                    write_bytes_to_wav(right_temp_path, frames[:number_frames], framerate=fr)

                    if not self.sp:
                        apply_gain(left_temp_path, self.gain)
                        apply_gain(right_temp_path, self.gain)

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

                    chunk_frames, frames, segment_start_time, chunk_end_time = self._update_chunk_list(
                        left_speaker, right_speaker, self.last_speaker, speaker, chunk_frames, frames,
                        segment_start_time)

                    os.remove(left_temp_path)
                    os.remove(right_temp_path)

                if chunk_frames:
                    if self.last_speaker not in ['silent', 'unknown']:
                        self._enqueue_transcription(chunk_frames, self.last_speaker, chunk_start_time, chunk_end_time)

                    # optionally store the audio chunk locally
                    if self.store:
                        chunk_audio_path = os.path.join(self.audio_dir, 'chunks',
                                                        f'{self.last_speaker}_chunk_{chunk_start_time}.wav')
                        write_bytes_to_wav(chunk_audio_path, chunk_frames, framerate=fr)
                        if self.last_speaker != 'silent':
                            normalize_decibel(chunk_audio_path, rms_level=-20)

                self.speaker_frames_dict[speaker] = (segment_start_time, frames)

        self.last_speaker = speaker

    def _update_chunk_list(self, left_speaker: str, right_speaker: str, last_speaker: str, current_speaker: str,
                           chunk_frames: bytes, frames: bytes, segment_start_time: float) -> tuple:
        """Update the chunk list based on recognition results at a speaker turn boundary.

        Adjust the audio chunks based on recognition outcomes from different segments,
        handling various cases such as extending the chunk to include additional audio.

        Args:
            left_speaker: recognized speaker from the left half of the segment.
            right_speaker: recognized speaker from the right half of the segment.
            last_speaker: recognized speaker from the previous segment.
            current_speaker: recognized speaker from the current segment.
            chunk_frames: audio frames (bytes) of the current chunk associated with the last speaker.
            frames: audio frames (bytes) of the current segment.
            segment_start_time: start time of the current segment.

        Returns:
            A tuple containing the updated chunk_frames, remaining frames, updated segment_start_time, and chunk_end_time.
        """
        num_frames = int(len(frames) / 2)

        if left_speaker == right_speaker and left_speaker == last_speaker:  # Case AAAB: Extend the left chunk
            chunk_frames = chunk_frames + frames[:num_frames]
            frames = frames[num_frames:]
            segment_start_time = segment_start_time + self.recognize_duration / 2
            chunk_end_time = segment_start_time
        elif left_speaker == right_speaker and right_speaker == current_speaker:  # Case ABBB: Extend the right chunk
            frames = chunk_frames[-num_frames:] + frames
            chunk_frames = chunk_frames[:-num_frames]
            segment_start_time = segment_start_time - self.recognize_duration / 2
            chunk_end_time = segment_start_time
        else:  # Other cases: AABB, A_BB, AA_B, A__B (excluding ABAB, AB_B, A_AB)
            chunk_end_time = segment_start_time
            if left_speaker != last_speaker:
                chunk_frames = chunk_frames[:-num_frames]
                chunk_end_time = segment_start_time - self.recognize_duration / 2
            if right_speaker != current_speaker:
                frames = frames[num_frames:]
                segment_start_time = segment_start_time + self.recognize_duration / 2

        return chunk_frames, frames, segment_start_time, chunk_end_time

    def _enqueue_transcription(self, frames: bytes, speaker: str, chunk_start_time: float, chunk_end_time: float):
        """Add an audio chunk to the transcription queue.

        If transcription is enabled (self.tr), enqueues the audio frames along with speaker and timing details.

        Args:
            frames: audio frames (bytes) to be transcribed.
            speaker: recognized speaker for the audio chunk.
            chunk_start_time: start time of the audio chunk.
            chunk_end_time: end time of the audio chunk.
        """
        if self.tr:
            self.transcription_queue.put((frames, speaker, chunk_start_time, chunk_end_time))

    def _process_final_chunks(self):
        """Process any remaining audio chunks when recording ends.
        
        This method handles the edge case where the recording ends while someone is still speaking.
        It processes any remaining audio in speaker_frames_dict that hasn't been transcribed yet.
        """
        if not self.speaker_frames_dict or not self.last_speaker:
            return
            
        fr = 8000 if self.sp and self.speaker_verification else 16000
        
        # process each remaining speaker's audio
        for speaker, (chunk_start_time, chunk_frames) in self.speaker_frames_dict.items():
            if not chunk_frames:
                continue
                
            # calculate end time based on audio duration
            chunk_duration = len(chunk_frames) / (fr * 2)  # Assuming 16-bit audio (2 bytes per sample)
            chunk_end_time = chunk_start_time + chunk_duration
            
            self.logger.info(f"Processing final chunk for speaker {speaker} from {chunk_start_time:.2f}s to {chunk_end_time:.2f}s")
            
            # process transcription directly since transcription thread has stopped
            if self.tr and speaker not in ['silent', 'unknown']:
                try:
                    transcribe_result = self._transcribe(chunk_frames, fr)
                    self._upload_transcription(speaker, transcribe_result, chunk_start_time, chunk_end_time)
                    self.logger.info(f"Final chunk transcription completed for speaker {speaker}")
                except Exception as e:
                    self.logger.warning(f"Failed to transcribe final chunk for speaker {speaker}: {e}")
            
            # store the audio chunk locally if enabled
            if self.store:
                chunk_audio_path = os.path.join(self.audio_dir, 'chunks',
                                                f'{speaker}_chunk_{chunk_start_time}.wav')
                write_bytes_to_wav(chunk_audio_path, chunk_frames, framerate=fr)
                if speaker != 'silent':
                    normalize_decibel(chunk_audio_path, rms_level=-20)

    def _transcribe(self, frames: bytes, frame_rate: int) -> dict:
        """Transcribe audio frames to text using an external speech-to-text service.

        Sends the audio frames to a remote transcription service and retrieves the 
        resulting text. The transcription is performed with the base's identifier
        to maintain context in multi-device setups.

        Args:
            frames: audio frames (bytes) to be transcribed.
            frame_rate: sample rate of the audio frames in Hz (typically 8000 or 16000).

        Returns:
            The transcription result as a dictionary. Empty dictionary if transcription failed.
        """
        response = request_speech_transcription(frames, frame_rate, f'{self.base_type.lower()}_{self.id}',
                                            self.speech_transcriber_url)
        return response if response is not None else {}

    def _upload_transcription(self, speaker: str, transcribe_result: dict, chunk_start_time: float, chunk_end_time: float):
        """Upload the transcribed speech chunk to the database.

        Constructs a transcription record with speaker, text content, and timing information,
        then writes it to InfluxDB for persistent storage. Also displays the transcription
        in the console for real-time monitoring.

        Args:
            speaker: identified speaker for the transcribed chunk.
            transcribe_result: the result of the transcription, including text and words.
            chunk_start_time: start timestamp of the audio chunk.
            chunk_end_time: end timestamp of the audio chunk.
        """
        from openmmla.utils.constants import EVENT_TYPE_ASR_TRANSCRIPTION
        fields = {
            "window_start_time": chunk_start_time,
            "window_end_time": chunk_end_time,
            "text": transcribe_result.get("text", ""),
            "words": json.dumps(transcribe_result.get("words", [])),
            "speaker": speaker,
        }
        print(f"{GREEN}[Speaker Transcription]{ENDC}{chunk_start_time}: "
              f"{GREEN}{speaker} : {transcribe_result.get('text', 'N/A')}{ENDC}")
        self.influx_client.write_event(self.session_id, EVENT_TYPE_ASR_TRANSCRIPTION, fields)

    def _publish_recognition(self, segment_start_time: float, recognize_start_time: float, speakers: list[str],
                             similarities: list[float], durations: list[float]):
        """Log and publish speaker recognition results via MQTT.

        Constructs a JSON record with recognition details and publishes it on the designated MQTT channel.
        The record includes speaker identities, similarity scores, and timing information.

        Args:
            segment_start_time: start time of the recorded segment (timestamp).
            recognize_start_time: start time of the recognition process (timestamp).
            speakers: list of recognized speakers.
            similarities: list of similarity scores (0.0-1.0) corresponding to speakers.
            durations: list of audio durations in seconds for each speaker segment.
        """
        base_recognition_result = {
            'base_id': f'{self.base_type.lower()}_{self.id}',
            'speakers': json.dumps(speakers),
            'similarities': json.dumps(similarities),
            'durations': json.dumps(durations),
            'segment_start_time': segment_start_time
        }
        print(f"{BLUE}[Speaker Recognition]{ENDC}{base_recognition_result['segment_start_time']}: "
              f"{BLUE}{base_recognition_result['speakers']}{ENDC}, similarity: {base_recognition_result['similarities']},"
              f"processed time: {time.time() - recognize_start_time} seconds")
        result_str = json.dumps(base_recognition_result)
        self.mqtt_client.publish(f'{self.session_id}/asr', result_str)

    def _separate_speech(self, audio_path: str) -> list[bytes]:
        """Separate overlapping speech from an audio file using source separation.

        Uses an external speech separation service to process the audio file, identifying and
        isolating different speakers in the recording. The separated signals are returned as
        raw audio bytes for further processing.

        Args:
            audio_path: path to the audio file to process.

        Returns:
            A list of separated speech signals as raw bytes (decoded from base64). Each element 
            represents an isolated speaker's audio. Returns an empty list if separation fails.
        """
        separated_result = request_speech_separation(audio_path, f'{self.base_type.lower()}_{self.id}',
                                                     self.speech_separator_url)
        if separated_result is None:
            return []

        result = [base64.b64decode(encoded_bytes_stream) for encoded_bytes_stream in separated_result]
        return result

    def _audio_preprocessing(self, audio_path: str, inplace: int) -> str | None:
        """Preprocess an audio file by applying noise reduction and voice activity detection.
        
        Performs a sequence of audio enhancement steps to improve recognition quality:
        1. Noise reduction (if enabled) - Enhances speech by reducing background noise
        2. Voice activity detection (if enabled) - Identifies and isolates speech segments
        
        Args:
            audio_path: path to the audio file to process.
            inplace: flag indicating whether to modify the input file (1) when VAD is enabled.
        
        Returns:
            Path to the processed audio file if successful, or None if processing failed
            (e.g., if no voice activity was detected in the file).
        """
        self._apply_nr(audio_path)
        return self._apply_vad(audio_path, inplace)

    def _apply_vad(self, audio_path: str, inplace: int) -> str | None:
        """Apply Voice Activity Detection (VAD) to an audio file.

        Uses an external VAD service to identify speech segments in the audio file. 
        This helps filter out silence and non-speech portions to improve recognition quality.

        Args:
            audio_path: path to the audio file to process.
            inplace: flag indicating whether to modify the input file (1).

        Returns:
            Path to the processed audio file if VAD is enabled and successful, the original path if VAD 
            is disabled, or None if no speech was detected.
        """
        if self.vad:
            return request_voice_activity_detection(audio_path, f'{self.base_type.lower()}_{self.id}', inplace,
                                                    self.vad_url)
        return audio_path

    def _apply_nr(self, audio_path: str) -> str:
        """Apply Noise Reduction (NR) to an audio file.

        Uses an external speech enhancement service to reduce background noise in the audio file.
        This improves speech clarity for better recognition results.

        Args:
            audio_path: path to the audio file to process.

        Returns:
            Path to the audio file after noise reduction (the same as audio_path, as the file is
            modified in place by the external service).
        """
        if self.nr:
            request_speech_enhancement(audio_path, f'{self.base_type.lower()}_{self.id}', self.speech_enhancer_url)
        return audio_path

    def _prepare_directories(self):
        """Prepare subdirectories for audio processing based on the operating mode.

        Create subdirectories for segments, chunks, separations, temporary files, and records.
        Clear specific directories based on the current operating mode.
        """
        runtime_root = pipeline_section_dir(self.project_dir, self.session_id, 'asr-base', 'real-time') / 'runtime'
        runtime_root.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = os.fspath(runtime_root)
        self.audio_dir = os.path.join(self.runtime_dir, f'{self.base_type}_{self.id}')
        sub_dirs = ['segments', 'chunks', 'separations', 'temp', 'records']
        for subdir in sub_dirs:
            directory_path = os.path.join(self.audio_dir, subdir)
            os.makedirs(directory_path, exist_ok=True)

        if self.mode == 'recognize':
            for subdir in ['segments', 'chunks', 'separations']:
                clear_directory(os.path.join(self.audio_dir, subdir))

        clear_directory(os.path.join(self.audio_dir, 'temp'))

    @staticmethod
    def warm_up_resampler(sample_rate_original: int = 44100, sample_rate_target: int = 16000):
        """Warm up the audio resampler to avoid delays during the first resampling operation.

        Performs a no-op resampling operation on a silent audio segment to ensure that when the
        actual resampling is needed, there won't be initialization delays. This is particularly
        important for real-time processing scenarios.

        Args:
            sample_rate_original: original sample rate in Hz (default is 44100 Hz).
            sample_rate_target: target sample rate in Hz (default is 16000 Hz).
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
            seconds: duration allowed for recording the prompt sentences.
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
    def session_control(self) -> str | None:
        """Dynamic property that returns the control channel name based on current session_id.
        
        This property provides the MQTT topic name used for start/stop control signals.
        The channel name is constructed using the current session_id, which allows
        for controlling specific sessions without affecting others.
        
        Returns:
            The control channel string in format '{session_id}/asr/control' if session_id 
            is set, otherwise None.
        """
        if self.session_id:
            return f'{self.session_id}/asr/control'
        return None
    
