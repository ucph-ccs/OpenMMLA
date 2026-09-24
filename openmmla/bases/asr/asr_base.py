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
from openmmla.utils.artifact_paths import copy_config_snapshot, pipeline_section_dir, runtime_pipeline_artifact_dir, session_artifact_dir
from openmmla.utils import session_provenance
from openmmla.utils.asr_scope import LAUNCH_GROUP, LAUNCH_SPEAKERS, chunk_cap, launch_attribution, normalize_asr_scope, participant_of, resolve_speaker_verification as _resolve_speaker_verification
from openmmla.bases.asr.attribution import LEVEL_HOP_SECONDS, SPEECH_GATE_SNR_DB, NoiseFloor, as_decibels, as_number, energy_record, level_trace, levels_record, relative_speech, segment_energy, snr_db, speech_gate_of, transcript_time
from openmmla.bases.asr.chunking import QUIET_CUT_SECONDS, quiet_cut
from openmmla.bases.asr.voices import VOICE_LINK_THRESHOLD, VOICE_MIN_SECONDS, VoiceRegistry, speaker_seconds, with_voices
from openmmla.utils.clean import clear_directory
from openmmla.utils.client import InfluxDBClientWrapper, MongoDBClientWrapper, MQTTClientWrapper, RedisClientWrapper
from openmmla.utils.input import select_or_create_session, get_id, get_interactive_files, get_stream_url, show_error_and_pause, pause_after_error
from openmmla.utils.logger import get_logger
from openmmla.utils.ports import free_port
from openmmla.utils.requests import resolve_url, build_service_url, service_problems
from openmmla.utils.session_sources import record_joined, record_left, source_entry
from .audio_recognizer import AudioRecognizer
from .speaker_profiles import REGISTRATION_SENTENCES, name_problem, parse_speakers, profiles_dir as speaker_profiles_dir
from .enums import BLUE, ENDC, GREEN, PURPLE, GREY, RED, YELLOW
from .input import get_base_type, get_function_base, get_name, get_base_mode, get_input_device_index, get_channel_selection, get_edit_speaker_options, get_speaker_selection, get_speaker_deletion, explain_cannot_start
from openmmla.utils.config import get_bases, get_base_by_id


def start_asr_base(project_dir: str, config_path: str, mode: str = 'live', store: bool = True,
                   vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False,
                   hsr: bool = True, session_id: str | None = None, base: str | None = None,
                   speakers: str | None = None, language: str | None = None, diarize: bool = False,
                   participant: str | None = None):
    """Start ASR Base with restart capability.
    
    Args:
        project_dir: Path to the project directory
        config_path: Path to the configuration file
        mode: Operating mode ('capture', 'analyze', or 'live')
        store: Whether to store audio files
        vad: Whether to apply Voice Activity Detection
        nr: Whether to use denoiser to enhance speech
        tr: Whether to transcribe speech to text
        sp: Whether to do speech separation for overlapped segments
        hsr: Whether to apply Half-Scaled Recognition at speaker boundaries
        session_id: Session to join; given, the base starts at once and exits when the run ends with STOP
        base: Id of the config 'Bases' entry this base is
        speakers: Comma-separated speaker profiles to recognize; if omitted, every registered one
        language: Language to transcribe in, whatever the speech transcriber is configured for
        participant: Whom the base's speech is attributed to: a participant's tag, 'group' or
            'speakers'; if omitted, the config decides
        diarize: Whether to ask the speech transcriber for anonymous speaker turns with every chunk
    """
    # restart loop - allows restarting the entire process
    while True:
        try:
            asr_base = ASRBase(project_dir=project_dir, config_path=config_path, mode=mode,
                              vad=vad, nr=nr, tr=tr, sp=sp, store=store, hsr=hsr,
                              session_id=session_id, base=base, speakers=speakers, language=language,
                              diarize=diarize, participant=participant)
            asr_base.run()
            break  # run() returns only once a run launched from the console has ended with STOP
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
    participant: str | None = None  # the tag of whoever wears this base's microphone (Bases.participant)
    _noise_floor = None  # the base's running noise floor (NoiseFloor), fresh each run
    # the voices this base heard in its session (VoiceRegistry): kept while the session stays, so a run
    # restarted after a recording error goes on numbering them; the session it is of, and its key
    # (voice_registry in the transcripts), which a base launched again into the session changes
    _voice_registry = None
    _voice_session = None
    _voice_key = None
    _voice_lock = threading.Lock()  # a transcription thread that outlived its stop may link beside the final chunks
    _embeddings_told = False  # whether a transcriber that diarized without speaker embeddings was named
    speech_gate = 'absolute'  # absolute: rms/peak thresholds after gain; relative: raw snr over the base's floor
    speech_gate_snr_db = SPEECH_GATE_SNR_DB  # the relative gate's dB over the floor

    def __init__(self, project_dir: str | None, config_path: str, mode: str = 'capture', store: bool = True,
                 vad: bool = True, nr: bool = True, tr: bool = True, sp: bool = False,
                 hsr: bool = True, session_id: str | None = None, base: str | None = None,
                 speakers: str | list[str] | None = None, registration: str | None = None,
                 language: str | None = None, diarize: bool = False, participant: str | None = None):
        """Initialize the ASRBase class.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
            mode: operating mode, 'capture', 'analyze', or 'live' (default: 'capture')
            store: whether to store audio files (default: True)
            vad: whether to apply Voice Activity Detection (default: True)
            nr: whether to apply noise reduction (default: True)
            tr: whether to transcribe speech to text (default: True)
            sp: whether to perform speech separation (default: False)
            hsr: whether to apply Half-Scaled Recognition at speaker boundaries (default: True)
            session_id: the session to join; given, the base was launched from the console: it
                asks nothing, starts at once and exits when the run ends with STOP (default: None)
            base: id of the config 'Bases' entry this base is; if omitted, the only entry when
                launched from the console, else picked interactively (default: None)
            speakers: the speaker profiles a base launched from the console recognizes, comma-separated
                or a list; if omitted, every registered one (default: None)
            registration: 'stream' or 'files' builds a base that only registers speaker profiles
                (mmla asr-speakers): it needs `base`, asks nothing, connects to no database or broker,
                frees no port, and for 'files' leaves its source alone (default: None)
            language: the language this base's speech is transcribed in ('en', 'da', 'zh-CN'), sent
                with every request and taken for it alone, whatever the speech transcriber is
                configured for; if omitted, that configured language (default: None)
            diarize: whether every chunk is sent for its anonymous speaker turns (SPEAKER_00,
                SPEAKER_01 ... within the chunk, no names), which the transcript record then
                carries as `diarization`; a group-level base (asr_scope: group) gets who-of-how-many
                spoke when without any speaker profile. Only a local WhisperX transcriber can
                (default: False)
            participant: whom this base's speech is attributed to, as the Launch tab picked it: a
                participant's tag (a microphone they wear: wearer mode with that tag), 'group' (the
                session's group) or 'speakers' (speaker verification); it wins over asr_scope, the
                Bases entry's participant and the session's Collection pick. If omitted, those
                decide (default: None)
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
        self.launch_speakers = parse_speakers(speakers)
        self.launch_participant = launch_attribution(participant)
        self.registration = registration
        self.language = str(language).strip() if language and str(language).strip() else None
        self.diarize = bool(diarize)

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
        self.asr_scope = "individual"
        self.speaker_verification = True
        self.max_chunk_duration = None
        self.group_speaker_id = "group"
        self.stream_name = None  # the Streams entry a 'stream' source pulls
        self._language_told = False  # whether a transcriber that did not take our language was named
        self._diarize_told = False  # whether a transcriber that did not diarize for us was named
        self._embeddings_told = False  # whether a transcriber that diarized without speaker embeddings was named
        self.url = None
        self._joined_session = None  # (session id, source key) this base noted itself in (session sources)

        # base identity/type/device come from the config 'Bases' list (single
        # source of truth) instead of typing the id / picking the type at startup
        if base:
            base_entry = get_base_by_id(self.config, base)
            if base_entry is None:
                if not self.launch_session_id or self.registration:
                    raise ValueError(f"Base '{base}' not found in config 'Bases'.")
                explain_cannot_start(
                    "ASR Base", f"-b {base} is not an id in the config's Bases list.",
                    "Pick this base's entry below; the ids are the entries under Bases on the ASR Base card's "
                    "Config tab.")
                base_entry = self._choose_base_from_config()
        elif self.registration:
            raise ValueError("Registering a speaker needs the base whose settings it takes (-b).")
        elif self.launch_session_id:
            base_entry = self._default_base()
        else:
            base_entry = self._choose_base_from_config()
        self._base_entry = base_entry
        self.base_type = str(base_entry['base_type'])
        self.id = base_entry['id']  # identity (string or number); port is separate
        if not self.registration:
            print(f"\033]0;ASR Base {self.base_type} {self.id} \007")

        self._setup_yaml()
        self._setup_directories()
        self._setup_objects()

    def _choose_base_from_config(self):
        """Interactively pick a base from the config 'Bases' list."""
        bases = get_bases(self.config)
        if not bases:
            raise ValueError(
                "No bases defined. Add entries under 'Bases' in config.yml "
                "(each with id and base_type).")
        print("Select a base:")
        for idx, b in enumerate(bases):
            print(f"  {idx}: id={b.get('id')} (type: {b.get('base_type')})")
        while True:
            sel = input("Base number [0]: ").strip()
            try:
                index = int(sel) if sel else 0
            except ValueError:
                index = -1
            if 0 <= index < len(bases):
                return bases[index]
            print("Invalid selection. Please enter a valid base number.")

    def _default_base(self):
        """The Bases entry a base launched from the console without -b takes: the
        only one there is. With several it says so and asks, as by hand."""
        bases = get_bases(self.config)
        if len(bases) == 1:
            return bases[0]
        if bases:
            explain_cannot_start(
                "ASR Base", f"the config's Bases list has {len(bases)} entries and no -b names this base's.",
                "Pick this base's entry below, or start it with -b <base id>.")
        return self._choose_base_from_config()

    @staticmethod
    def _pick_stream(stream_sources: list[tuple[str, str]]) -> tuple[str, str]:
        """(name, url) of the stream picked in the stream menu."""
        url = get_stream_url([stream_url for _, stream_url in stream_sources])
        name = next((name for name, stream_url in stream_sources if stream_url == url), None)
        return name, url

    def _setup_yaml(self):
        """Load and assign configuration parameters from the YAML configuration file.

        Read various settings such as durations, thresholds, and service URLs required for the audio processing pipeline.
        """
        base_config = self.config['Base'][self.base_type]
        asr_server_config = self.config['Server']['asr']

        # recognition scope is a per-device setting: a room microphone profile
        # is typically group-scope (transcription only), a microphone worn by
        # one person wearer-scope (labelled with its wearer, no verification),
        # a badge that tells speakers apart individual-scope (speaker
        # verification + transcription). the actual group id is resolved from
        # the session at runtime.
        self.asr_scope = normalize_asr_scope(base_config.get('asr_scope'))
        self.speaker_verification = _resolve_speaker_verification(
            base_config.get('speaker_verification', 'auto'),
            self.asr_scope,
        )
        # a microphone worn by one participant (Bases.participant): its speech is theirs, told apart
        # from the neighbours' by the synchronizer's energy vote, never by speaker verification; it
        # chunks like a group microphone
        self.participant = participant_of(self._base_entry.get('participant'))
        if self.participant is not None:
            self.speaker_verification = False
        self._wearer_source = 'config' if self.participant is not None else None
        # what the Launch tab picked for this base (--participant) wins over all of that, for every run
        launched = getattr(self, 'launch_participant', None)
        if launched == LAUNCH_GROUP:
            self.asr_scope, self.participant, self.speaker_verification = 'group', None, False
        elif launched == LAUNCH_SPEAKERS:
            self.asr_scope, self.participant, self.speaker_verification = 'individual', None, True
        elif launched is not None:
            self.asr_scope, self.participant, self.speaker_verification = 'wearer', launched, False
        if launched is not None:
            self._wearer_source = 'launch' if self.participant is not None else None
        # the longest a chunk of one speaker may grow (30 s for a group microphone unless set)
        self.max_chunk_duration = chunk_cap(base_config.get('max_chunk_duration'),
                                            'group' if self.participant is not None else self.asr_scope)
        # what the Bases entry and the Launch tab say, which a session's noted wearer overrides for
        # one run only (never a Launch pick)
        self._configured_wearer = (self.participant, self.speaker_verification, self.max_chunk_duration)
        self.wearer_source = self._wearer_source

        self.register_duration = int(base_config['register_duration'])
        self.recognize_duration = int(base_config['recognize_sp_duration']) if self.sp else int(
            base_config['recognize_duration'])
        # speech_gate: absolute (default) compares the level after gain, NR and VAD with rms_threshold and
        # rms_peak_threshold; relative compares the raw level with this base's own noise floor
        self.speech_gate = speech_gate_of(base_config.get('speech_gate'))
        self.speech_gate_snr_db = as_decibels(base_config.get('speech_gate_snr_db'), SPEECH_GATE_SNR_DB)
        if self.speech_gate == 'relative':
            # unused by the gate, so they may be left blank
            self.rms_threshold = int(as_number(base_config.get('rms_threshold'), 0))
            self.rms_peak_threshold = int(as_number(base_config.get('rms_peak_threshold'), 0))
        else:
            self.rms_threshold = int(base_config['rms_threshold'])
            self.rms_peak_threshold = int(base_config['rms_peak_threshold'])
        self.threshold = float(base_config['recognize_sp_threshold']) if self.sp else float(
            base_config['recognize_threshold'])
        self.keep_threshold = float(base_config['keep_sp_threshold']) if self.sp else float(
            base_config['keep_threshold'])
        self.update_threshold = float(base_config.get('update_threshold', 0.6))
        self.gain = float(base_config['gain'])
        self.score_amplified = bool(base_config.get('score_amplified', False))

        self.speech_transcriber_url = build_service_url(self.config, asr_server_config['speech_transcriber'])
        self.speech_separator_url = build_service_url(self.config, asr_server_config['speech_separator'])
        self.speech_enhancer_url = build_service_url(self.config, asr_server_config['speech_enhancer'])
        self.vad_url = build_service_url(self.config, asr_server_config['voice_activity_detector'])

        self.source = None
        self.stream_kwargs = {}
        if self.registration != 'files':  # a registration from files records nothing
            self._setup_source(base_config)

    def _setup_source(self, base_config: dict):
        """Resolve where the base takes its audio from: its Bases entry's source and
        source_index, into self.source and self.stream_kwargs."""
        # source comes from the per-base Bases entry (single source of truth);
        # Base.<device>.source has been removed, so the base must define its source
        self.source = self._base_entry.get('source') or base_config.get('source')
        if not self.source:
            raise ValueError(
                f"Base '{self.id}' has no 'source'. Set 'source' in its Bases entry.")
        # a copy: what the source adds below is this base's, not its base type's
        self.stream_kwargs = dict(base_config.get('stream_kwargs') or {})

        from openmmla.utils.constants import normalize_source
        self.source = normalize_source(self.source)
        source_list = ['udp', 'tcp', 'pyaudio', 'stream', 'lsl', 'file']
        if self.source not in source_list:
            raise ValueError(f'Unknown source {self.source}, must be one of {source_list}')

        # set port number for 'udp/tcp' (explicit per-base 'port', decoupled from id)
        if self.source in ['udp', 'tcp']:
            if self._base_entry.get('port') is None:
                raise ValueError(
                    f"Base '{self.id}' uses source '{self.source}' but has no 'port' "
                    "in its Bases entry. Add a 'port' to that base.")
            self.port = int(self._base_entry['port'])
            if not self.registration:  # a registration leaves a base listening there alone
                free_port(self.port)
            self.stream_kwargs['port'] = self.port
            # where it listens and how its packets are laid out are the entry's too;
            # an older config has them in Base.<type>.stream_kwargs
            self.stream_kwargs['host'] = (self._base_entry.get('host') or self.stream_kwargs.get('host')
                                          or '0.0.0.0')
            self.stream_kwargs['packet_format'] = (self._base_entry.get('packet_format')
                                                   or self.stream_kwargs.get('packet_format') or 'auto')

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

            # device index comes from the base entry's source_index (profile-driven)
            if self._base_entry.get('source_index') is None:
                raise ValueError(
                    f"Base '{self.id}' uses source 'pyaudio' but has no 'source_index' "
                    "(input device index) in its Bases entry.")
            self.input_device_index = int(self._base_entry['source_index'])
            self.stream_kwargs['input_device_index'] = self.input_device_index
            device_info = p.get_device_info_by_host_api_device_index(0, self.input_device_index)

            # update channels based on selected device capabilities
            device_channels = device_info.get('maxInputChannels', 1)
            self.stream_kwargs['channels'] = device_channels

            # the one channel of a multi-channel device this base keeps
            # (Bases.channel_select; 'channel' is its old name), else all of them
            select = self._base_entry.get('channel_select', self._base_entry.get('channel'))
            self.stream_kwargs['channel_select'] = None if select in (None, "") else int(select)
            self.logger.info(f"Selected device: {device_info.get('name')} with {device_channels} channels")
            self.logger.info(f"Selected channel option: {self.stream_kwargs['channel_select']}")
            p.terminate()

        # set url for 'stream': the Streams entry the base's source_index names,
        # as for the IPS and VFA bases. It used to be asked in a menu whatever
        # the entry said, which a base started from the console waited on
        elif self.source == 'stream':
            from openmmla.utils.constants import get_stream_sources, resolve_stream_source
            source_index = self._base_entry.get('source_index')
            stream_sources = get_stream_sources(self.config)
            if source_index in (None, "") and len(stream_sources) > 1:
                if self.registration:
                    raise ValueError(
                        f"base {self.id} names no stream in its Bases entry (source_index), and there are "
                        f"{len(stream_sources)} to pull: {', '.join(name for name, _ in stream_sources)}. "
                        "Set its source_index on the ASR Base card's Config tab.")
                # the entry names none of several: ask, as a base started by hand
                if self.launch_session_id:
                    explain_cannot_start(
                        "ASR Base",
                        f"base {self.id} names no stream in its Bases entry (source_index), and there are "
                        f"{len(stream_sources)} to pull: {', '.join(name for name, _ in stream_sources)}.",
                        "Pick its stream below; set it on the ASR Base card's Config tab to skip this next time.",
                        wait=True)  # the stream menu clears the screen
                self.stream_name, self.url = self._pick_stream(stream_sources)
                self.logger.info(f"Using stream URL: {self.url}")
            else:
                try:
                    self.stream_name, self.url = resolve_stream_source(self.config, source_index)
                except ValueError as e:
                    # launched from the console, a stream it cannot find is picked in the menu instead
                    if not self.launch_session_id or not stream_sources:
                        raise
                    explain_cannot_start(
                        "ASR Base", f"base {self.id}: {e}",
                        "Pick its stream below; fix its source_index on the ASR Base card's Config tab to skip "
                        "this next time.", wait=True)  # the stream menu clears the screen
                    self.stream_name, self.url = self._pick_stream(stream_sources)
                self.logger.info(f"Using stream '{self.stream_name}': {self.url}")
            self.stream_kwargs['url'] = self.url

        # set lsl_name for 'lsl' (the stream is selected by name; the base
        # entry carries it in source_index, matching the unified config form)
        elif self.source == 'lsl':
            lsl_name = self._base_entry.get('source_index')
            if not lsl_name:
                raise ValueError(
                    f"Base '{self.id}' uses source 'lsl' but has no stream name in "
                    f"'source_index'. Set source_index to the LSL stream name.")
            self.stream_kwargs['lsl_name'] = lsl_name
            self.logger.info(f"Using LSL stream name: {lsl_name}")

        # set file_path for 'file'
        elif self.source == 'file':
            if self.registration:
                raise ValueError(f"base {self.id} reads a file (source: file), so it has no stream to record a "
                                 "speaker from: register from files instead.")
            # a bare name in source_index is looked up in the Base.<type>.file_dir of an
            # older config, else in the project directory; a `file_dir:` left empty is none
            file_dir = base_config.get('file_dir')
            if not file_dir:
                file_dir = self.project_dir
                self.logger.info(f"No file_dir specified in config, using project directory: {file_dir}")
            else:
                if not os.path.isabs(file_dir):
                    file_dir = os.path.join(self.project_dir, file_dir)

                if not os.path.exists(file_dir):
                    # fallback to project directory if specified directory doesn't exist
                    self.logger.warning(f"Specified file directory does not exist: {file_dir}")
                    file_dir = self.project_dir
                    self.logger.info(f"Using project directory instead: {file_dir}")

            audio_extensions = ('.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.wma')
            sel = self._base_entry.get('source_index')
            cand = None
            if sel:
                cand = str(sel) if os.path.isabs(str(sel)) else os.path.join(file_dir, str(sel))
                if not os.path.exists(cand):
                    if not self.launch_session_id:
                        raise ValueError(
                            f"Base '{self.id}' source 'file' references '{sel}' but it was "
                            f"not found in {file_dir}.")
                    # launched from the console: say so and browse for the file below
                    explain_cannot_start(
                        "ASR Base", f"base {self.id} reads the file '{sel}', which is not in {file_dir}.",
                        "Pick the audio file below; fix its source_index on the ASR Base card's Config tab to "
                        "skip this next time.")
                    cand = None
            elif self.launch_session_id:
                explain_cannot_start(
                    "ASR Base", f"base {self.id} reads a file (source: file) but names none in source_index.",
                    "Pick the audio file below; set its source_index on the ASR Base card's Config tab to skip "
                    "this next time.")
            if cand:
                # profile-driven: the base entry names the file (source_index) —
                # resolve it and derive initial_sync_time without prompting
                file_path = cand
                ist = base_config.get('initial_sync_time')
                if ist is None:
                    m = re.search(r'_(\d+(?:\.\d+)?)\.', os.path.basename(file_path))
                    ist = float(m.group(1)) if m else None
                self.initial_sync_time = float(ist) if ist is not None else None
            else:
                # interactive fallback: browse for a file and read initial_sync_time
                print(f"\n{PURPLE}📁 Select Audio File{ENDC}")
                print(f"{GREY}Choose audio file for ASR Base{ENDC}")
                file_path, initial_sync_time = get_interactive_files(
                    file_dir, file_extensions=audio_extensions, multiple=False, sync_input=True)
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
        self.profiles_dir = speaker_profiles_dir(self.project_dir)

        os.makedirs(self.logger_dir, exist_ok=True)
        os.makedirs(self.runtime_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)

    def _setup_objects(self):
        """Initialize external service clients and internal processing objects.

        Set up the clients for InfluxDB, Redis, and MQTT, warms up the audio resampler, and initializes the
        AudioRecognizer and AudioStream.
        """
        if self.registration:
            self.influx_client = self.mongo_client = self.redis_client = self.mqtt_client = None
            self.audio_stream = None  # a registration from the stream opens its own
            self.audio_recognizer = AudioRecognizer(config_path=self.config_path, profiles_dir=self.profiles_dir,
                                                    store=self.store, selected_speakers=self.selected_speakers)
            return
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
        self._restore_configured_wearer()

    def _restore_configured_wearer(self):
        """Put back the wearer, speaker verification and chunk cap the Bases entry gives."""
        configured = getattr(self, '_configured_wearer', None)
        if configured is None:
            return
        self.participant, self.speaker_verification, self.max_chunk_duration = configured
        self.wearer_source = getattr(self, '_wearer_source', 'config' if self.participant is not None else None)

    def _apply_session_wearer(self, session_id):
        """A base whose Bases entry names no participant and that pulls a stream takes the wearer the
        session's Collection Start noted for that stream (the session document's wearers), for this
        run only; the Bases config is left as it is."""
        self._restore_configured_wearer()
        if getattr(self, 'launch_participant', None) is not None:
            return  # picked on the Launch tab: that is who it is
        if self.participant is not None or getattr(self, 'source', None) != 'stream':
            return
        if not getattr(self, 'stream_name', None) or not session_id:
            return
        try:
            session = self.mongo_client.get_session(session_id) or {}
        except Exception as e:
            self.logger.warning(f"Could not read the wearers of session {session_id}: {e}")
            return
        wearers = session.get('wearers') if isinstance(session, dict) else None
        tag = participant_of(wearers.get(self.stream_name)) if isinstance(wearers, dict) else None
        if tag is None:
            return
        base_blocks = (getattr(self, 'config', None) or {}).get('Base')
        base_config = base_blocks.get(getattr(self, 'base_type', None)) if isinstance(base_blocks, dict) else None
        base_config = base_config if isinstance(base_config, dict) else {}
        self.participant = tag
        self.speaker_verification = False
        self.max_chunk_duration = chunk_cap(base_config.get('max_chunk_duration'), 'group')
        self.wearer_source = 'session'
        self.logger.info(f"Stream '{self.stream_name}' is worn by participant {tag} in session {session_id} "
                         f"(its Collection Start): wearer mode for this run.")

    def _close_clients(self):
        """Close the connections to MQTT, Redis, InfluxDB and MongoDB before the process exits."""
        for close in (getattr(self.mqtt_client, 'disconnect', None), getattr(self.redis_client, 'close', None),
                      getattr(self.influx_client, 'close', None), getattr(self.mongo_client, 'close', None)):
            try:
                if close:
                    close()
            except Exception as e:
                self.logger.debug(f"Closing a client on exit: {e}")

    def _available_speakers(self) -> list[str]:
        """The speakers registered in the profiles directory."""
        if not os.path.exists(self.profiles_dir):
            return []
        return [d for d in os.listdir(self.profiles_dir)
                if os.path.isdir(os.path.join(self.profiles_dir, d)) and not d.startswith('.')]

    def run(self):
        """Run the ASR base.

        Launched from the console (with a session id) it starts recognizing at
        once, without the menu, with every registered speaker profile selected,
        and returns when that run ends with STOP, so the process exits. When it
        cannot start (individual scope without speaker profiles) it says why and
        shows its menu, where Edit Speaker Profiles fixes it; a run that ends with
        an error also comes back to the menu. Started by hand, it prompts with the
        menu until termination.
        """
        func_map = {1: self._edit_speakers, 2: self._start_recognition, 3: self._switch_mode, 4: self._reset}
        start_at_once = bool(self.launch_session_id)
        ended = False
        while True:
            try:
                if start_at_once:
                    start_at_once = False
                    select_fun = 2
                    if self.speaker_verification and self.selected_speakers is None:
                        # the selection Edit Speaker Profiles starts from
                        self.selected_speakers = self._launch_selection()
                else:
                    select_fun = get_function_base(self.id, self.mode)
                outcome = func_map.get(select_fun, lambda: self.logger.warning("Invalid option"))()
                if select_fun == 2 and self.launch_session_id:
                    if outcome is True:
                        ended = True
                        break
                    if outcome is None:
                        explain_cannot_start(
                            "ASR Base",
                            f"base {self.id} verifies speakers (asr_scope: {self.asr_scope}), and {self.mode} mode "
                            "needs at least one registered speaker profile selected.",
                            "Register and select speakers with Edit Speaker Profiles below, then choose Start "
                            "(next time, register and pick them before Start with Speakers on the ASR Base card). "
                            f"Or switch to capture mode, or set asr_scope: group for base type {self.base_type} "
                            "on the ASR Base card's Config tab.")
                    else:
                        print(f"The run of base {self.id} did not end with STOP (see above). "
                              "Choose Start below to run it again.")
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
                self._leave_session()
                self._clean_up()

        if ended:
            self._close_clients()
            print(f"The run of session {self.launch_session_id} ended with STOP: ASR Base {self.id} exits.")

    def _launch_selection(self) -> list[str]:
        """The speakers a base launched from the console starts with: the -spk names
        registered on this host, in that order, else every registered profile. The
        recognizer is left with those only."""
        available = self._available_speakers()
        if self.launch_speakers is None:
            return available
        missing = [name for name in self.launch_speakers if name not in available]
        if missing:
            print(f"{RED}Not registered on this host, so not recognized: {', '.join(missing)}{ENDC}")
        chosen = [name for name in self.launch_speakers if name in available]
        self.audio_recognizer.reset_profiles(self.profiles_dir, chosen)
        return chosen

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
                self.logger.warning(f"Error in speaker editing: {e}", exc_info=True)
                show_error_and_pause(e, "return to the menu")

    def _register_speaker_from_stream(self):
        """Register a new speaker profile from audio stream."""
        if self.source == 'file':
            self.logger.info("Cannot register from stream when source is 'file'. Please use 'Register from Files' option.")
            return

        self.recording_prompt(self.register_duration)
        audio_path = self._record_for_registration()

        if audio_path is None:
            msg = self._no_speech_message()
            self.logger.info(msg)
            print(f"\n{RED}❌ {msg}{ENDC}")
            pause_after_error("return to the menu")
            return

        name = get_name()
        if name == '':
            self.logger.info('Empty name, skip the registering process.')
            return
        problem = name_problem(name)
        if problem:
            print(f"{RED}'{name}': {problem}. The recording is not registered.{ENDC}")
            return

        self.audio_recognizer.register(audio_path, name)
        self.logger.info(f"Speaker '{name}' has been successfully registered from stream!")

    def _record_for_registration(self, duration: float | None = None) -> str | None:
        """Record `duration` seconds (register_duration if not given) from the base's source and
        prepare them for registration (gain, noise reduction, VAD): the wav to register, or None
        when no speech was found in it."""
        output_path = os.path.join(self.temp_dir, f'{self.base_type}_{self.id}_register.wav')
        self.audio_stream = AudioStream(source=self.source, **self.stream_kwargs)
        self.audio_stream.start()
        try:
            audio_frame = self.audio_stream.read(duration=duration or self.register_duration, latest=True)
        finally:
            self.audio_stream.stop()
        write_frame_to_wav(output_path, audio_frame)

        apply_gain(output_path, self.gain)
        return self._audio_preprocessing(output_path, 1)

    def _no_speech_message(self) -> str:
        """why a recording left nothing to register. The pre-processing services are asked
        again, so that a failure names its cause; when they answer, the recording held no speech."""
        problems = self._service_problems(*self._preprocessing_services())
        if problems:
            return (f"Audio pre-processing failed: {problems}. Register again once that is fixed, or skip "
                    f"pre-processing (-vad False -nr False; on the console, the card's VAD and noise reduction).")
        return (f"No speech was found in the recording. Record again, closer to the microphone, or check that "
                f"base {self.id} records the right microphone.")

    def _preprocessing_services(self) -> list[str]:
        """the keys of Server.asr that pre-processing calls: noise reduction and VAD, when on."""
        return [key for on, key in ((self.nr, 'speech_enhancer'), (self.vad, 'voice_activity_detector')) if on]

    def _service_problems(self, *keys: str) -> str:
        """why the ASR server's services under these keys of Server.asr cannot serve, asked
        again now (service_problems); empty when every one answers."""
        endpoints = self.config['Server']['asr']
        return "; ".join(service_problems(
            [build_service_url(self.config, endpoints[key], resolve=False) for key in keys if endpoints.get(key)]))

    def register_speaker_from_stream(self, name: str, duration: float | None = None) -> int:
        """Register speaker `name` (or add to that profile) from `duration` seconds of the base's
        source, asking nothing (mmla asr-speakers). Returns how many embeddings the profile holds."""
        problem = name_problem(name)
        if problem:
            raise ValueError(f"'{name}': {problem}.")
        seconds = duration or self.register_duration
        print(f"Recording {seconds:g} s from base {self.id} ({self.source}): read the sentences aloud now.", flush=True)
        audio_path = self._record_for_registration(seconds)
        if audio_path is None:
            raise RuntimeError(self._no_speech_message())
        print("Recorded; making the voice features...", flush=True)
        return self._register_checked(audio_path, name)

    def register_speaker_from_files(self, name: str, files: list[str]) -> tuple[int, int]:
        """Register speaker `name` (or add to that profile) from reference audio files on this host,
        asking nothing (mmla asr-speakers). Returns (the files used, the embeddings the profile holds)."""
        problem = name_problem(name)
        if problem:
            raise ValueError(f"'{name}': {problem}.")
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"Not on this host: {', '.join(missing)}")
        before = self._embedding_count(name)
        used = self._register_files(name, files)
        after = self._embedding_count(name)
        if not used or after <= before:
            self._drop_if_empty(name)
            problems = self._service_problems(*self._preprocessing_services(), 'audio_inferer')
            raise RuntimeError(f"No voice features could be made from these files: "
                               f"{problems or 'no speech was found in them'}.")
        return used, after

    def _register_checked(self, audio_path: str, name: str) -> int:
        """Register `audio_path` as `name` and make sure it added embeddings; a profile folder
        the attempt left empty is removed."""
        before = self._embedding_count(name)
        self.audio_recognizer.register(audio_path, name)
        after = self._embedding_count(name)
        if after <= before:
            self._drop_if_empty(name)
            problems = self._service_problems('audio_inferer')
            raise RuntimeError(f"No voice features could be made from the recording: "
                               f"{problems or 'the audio inferer answers, but made none (see its Logs)'}.")
        return after

    def _embedding_count(self, name: str) -> int:
        folder = os.path.join(self.profiles_dir, name)
        try:
            return sum(1 for file_name in os.listdir(folder) if file_name.endswith('.pkl'))
        except OSError:
            return 0

    def _drop_if_empty(self, name: str):
        """Remove the profile folder of `name` when it holds nothing (a first registration
        that made no embedding), so it is not listed as a speaker."""
        folder = os.path.join(self.profiles_dir, name)
        try:
            if os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)
        except OSError:
            pass

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
            problem = name_problem(name)
            if problem:
                print(f"{RED}'{name}': {problem}. Nothing is registered.{ENDC}")
                return

            used = self._register_files(name, reference_files)
            if not used:
                self.logger.error("No files could be processed successfully.")
                return
            self.logger.info(f"Speaker '{name}' has been successfully registered from {used} reference files!")

        except Exception as e:
            self.logger.error(f"Error in file-based registration: {e}")

    def _register_files(self, name: str, reference_files: list[str]) -> int:
        """Prepare each reference file (format, noise reduction, VAD) and register it as `name`;
        returns how many were registered. A file that fails is logged and skipped."""
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

        # each file adds its embeddings to the profile; the first makes it
        registered = 0
        for i, processed_file in enumerate(processed_files):
            try:
                self.audio_recognizer.register(processed_file, name)
                registered += 1
            except Exception as e:
                self.logger.warning(f"Error registering file {i+1}: {e}")
        return registered

    def _start_recognition(self, session_id: str | None = None):
        """Start the real-time voice recognition process.

        Set up directories, queues, and MQTT communication before creating threads for:
          - Continuous recording.
          - Loading and queuing pre-recorded files (if in 'analyze' mode).
          - Continuous recognition (with or without speech separation).
          - Continuous transcription (if enabled).
          - Listening for stop signals.

        Args:
            session_id: The bucket name for storing recognition results. If not provided, it is obtained interactively.

        Returns:
            True when the run ended with STOP, False when it ended with an error, and None when it
            could not start (speaker verification without a selected speaker profile).
        """
        # a stream the session's Collection Start noted a wearer for is that person's, for this run
        self._apply_session_wearer(session_id or self.launch_session_id)
        # check if any speakers are selected for individual-level recognition
        if self.speaker_verification and (not self.selected_speakers or len(self.selected_speakers) == 0):
            print("------------------------------------------------")
            if self.mode in ['live', 'analyze']:
                self.logger.info("No speakers selected for recognition. Please register and select speaker profiles or switch to 'capture' mode.")
                return None
            elif self.mode == 'capture':
                self.logger.warning("No speakers selected. Recording will continue without speaker recognition.")
        elif self.speaker_verification and self.mode in ['live', 'analyze'] and len(self.audio_recognizer.speaker_names) == 0:
            print("------------------------------------------------")
            self.logger.info("Audio database is empty, please register speaker profiles or either switch the mode to 'capture'.")
            return None
        elif self.speaker_verification and self.mode == 'capture' and len(self.audio_recognizer.speaker_names) == 0:
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
            if self.participant is not None:
                print(f"{GREEN}ASR chunks are attributed to participant {self.participant} when their channel is "
                      f"the loudest (energy attribution).{ENDC}")
            elif self.asr_scope == "wearer":
                # a worn microphone nobody is noted as wearing: its speech has no one to go to
                print(f"{YELLOW}Base type {self.base_type} is asr_scope wearer, but no one is noted as wearing "
                      f"this microphone (the participant of its Bases entry, or for a stream the wearer the "
                      f"session's Collection Start picked): its ASR chunks are attributed at group scope for "
                      f"this run.{ENDC}")
                self.logger.warning(f"Base {self.id} is asr_scope wearer with no wearer: group scope for this run.")
            else:
                print(f"{GREEN}ASR chunks will be attributed at group scope.{ENDC}")
        print(f"{GREEN}{self._speech_gate_text()}{ENDC}")
        
        # select or create bucket
        launch_session_id = session_id or self.launch_session_id
        self.session_id = select_or_create_session(self.mongo_client) if not launch_session_id else launch_session_id
        if self.session_id != (session_id or self.launch_session_id):
            # picked in the menu just now: its wearer is known only from here on
            self._apply_session_wearer(self.session_id)
        self._join_session()
        self._resolve_group_speaker_id()
        self._create_bucket_logger()
        self._create_speaker_profile_snapshot()
        self._record_provenance()

        # reset attributes
        self.last_speaker = None
        self._noise_floor = NoiseFloor()
        self._voices_for_session()
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.speaker_frames_dict = {}

        self._prepare_directories()
        if not self._listen_for_start_signal():
            # STOP came before START: the run ended before anything was recorded
            self.logger.info(f"The run of session {self.session_id} was stopped before it started; "
                             f"ASR base {self.id} leaves the session.")
            self._leave_session()
            self._clean_up()
            return True

        # reinitialize mqtt client
        self.mqtt_client.reinitialise()
        self.mqtt_client.loop_start()

        # create threads based on the operating mode
        if self.mode in ['capture', 'live']:
            self._create_thread(self._continuous_recording)
        if self.mode == 'analyze':
            self._create_thread(self._enqueue_recorded_files)
        if self.mode in ['analyze', 'live']:
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
        ended_with_stop = False
        try:
            self._start_threads()
            self._join_threads()
        except (Exception, KeyboardInterrupt) as e:
            self.logger.warning(
                f"During voice recognition, catch: {'KeyboardInterrupt' if isinstance(e, KeyboardInterrupt) else e}",
                exc_info=True)
            exception_occurred = e
        finally:
            ended_with_stop = self._recognition_handler(exception_occurred)
        return ended_with_stop

    def _join_session(self):
        """Note in the session which Bases entry this base is and which stream it
        takes (openmmla.utils.session_sources), once per run: a run restarted after
        a recording error is still in."""
        if not self.session_id or (self._joined_session and self._joined_session[0] == self.session_id):
            return
        # a 'stream' source resolved its stream itself, possibly in the stream menu
        stream, url = (self.stream_name, self.url) if self.source == 'stream' else (None, None)
        try:
            entry = source_entry('asr', self._base_entry, self.config, stream=stream, url=url)
        except Exception as e:
            self.logger.warning(f"Could not note in session {self.session_id} which stream base {self.id} "
                                f"takes: {e}")
            return
        if record_joined(self.mongo_client, self.session_id, entry, log=self.logger):
            # the key it joined with (it names the stream), which its leaving must name too
            self._joined_session = (self.session_id, entry['key'])

    def _leave_session(self):
        """Note that this base left the session it joined; once."""
        if not self._joined_session:
            return
        (session_id, key), self._joined_session = self._joined_session, None
        record_left(self.mongo_client, session_id, key, log=self.logger)

    def _resolve_group_speaker_id(self):
        """Prefer the selected session group id for group-level ASR attribution."""
        if self.participant is not None:
            return  # a worn microphone is labelled with its wearer, never the group
        # a wearer-scope base with no wearer noted falls back on the group too
        if self.asr_scope not in ("group", "wearer") or not self.session_id:
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

    def _record_provenance(self):
        """Note in the session what this base runs with (openmmla.utils.session_provenance):
        its flags, the thresholds and durations of its base type, the stream it takes, the
        speaker profiles it recognizes, its config with the secrets masked; what its servers
        run (the transcriber's model and language ...) is asked after, in a thread. A failure
        is a warning, never a stop."""
        if not self.session_id:
            return
        try:
            profiles = None
            if self.speaker_verification and self.selected_speakers:
                snapshot_dir = os.path.join(self.runtime_dir, f'{self.base_type}_{self.id}', 'profiles')
                profiles = {'names': list(self.selected_speakers),
                            'snapshot': os.path.relpath(snapshot_dir, session_artifact_dir(self.project_dir, self.session_id))}
            entry = session_provenance.component_entry(
                'asr', 'base', self.id,
                arguments={'mode': self.mode, 'store': self.store, 'vad': self.vad, 'nr': self.nr, 'tr': self.tr,
                           'sp': self.sp, 'hsr': self.hsr, 'session_id': self.launch_session_id,
                           'base': self._base_entry.get('id'), 'speakers': self.launch_speakers,
                           'language': self.language, 'diarize': self.diarize},
                parameters={
                    'base_type': self.base_type, 'id': self.id, 'asr_scope': self.asr_scope,
                    'speaker_verification': self.speaker_verification, 'max_chunk_duration': self.max_chunk_duration,
                    # how far before its cap a chunk may be cut at its quietest moment (None: at the cap)
                    'quiet_cut_seconds': QUIET_CUT_SECONDS if self._quiet_cuts() else None,
                    # how the speakers of a diarized chunk are linked into the voices of the session (a
                    # base that asks for turns; one whose transcriber diarizes every file links too, and
                    # its transcripts say so by their voices)
                    'voice_link_threshold': VOICE_LINK_THRESHOLD if self.diarize else None,
                    'voice_min_seconds': VOICE_MIN_SECONDS if self.diarize else None,
                    'selected_speakers': list(self.selected_speakers or []),
                    'group_speaker_id': self.group_speaker_id, 'language': self.language,
                    'participant': self.participant,
                    'wearer_source': getattr(self, 'wearer_source', None),
                    'attribution': 'energy' if self.participant is not None else None,
                    # the step of the level trace a worn microphone's transcripts and recognitions carry
                    'level_hop_seconds': LEVEL_HOP_SECONDS if self._keeps_levels() else None,
                    'register_duration': self.register_duration, 'recognize_duration': self.recognize_duration,
                    'rms_threshold': self.rms_threshold, 'rms_peak_threshold': self.rms_peak_threshold,
                    'speech_gate': self.speech_gate, 'speech_gate_snr_db': self.speech_gate_snr_db,
                    'recognize_threshold': self.threshold, 'keep_threshold': self.keep_threshold,
                    'update_threshold': self.update_threshold, 'gain': self.gain,
                    'score_amplified': self.score_amplified,
                    'source': self.source, 'source_index': self._base_entry.get('source_index'),
                    'stream': self.stream_name, 'url': self.url, 'port': getattr(self, 'port', None),
                    'input_device_index': getattr(self, 'input_device_index', None),
                    'initial_sync_time': getattr(self, 'initial_sync_time', None),
                    'stream_kwargs': self.stream_kwargs,
                    'service_urls': {'speech_transcriber': self.speech_transcriber_url,
                                     'speech_separator': self.speech_separator_url,
                                     'speech_enhancer': self.speech_enhancer_url,
                                     'voice_activity_detector': self.vad_url,
                                     'audio_inferer': getattr(self.audio_recognizer, 'audio_inferer_url', None)},
                },
                files={'speaker_profiles': profiles},
                config=self.config, config_path=self.config_path, project_dir=self.project_dir)
            session_provenance.record_component(self.mongo_client, self.session_id, entry, self.project_dir,
                                                'asr-base', log=self.logger)
            urls = {}
            if self.tr:
                urls['speech_transcriber'] = self.speech_transcriber_url
            if self.vad:
                urls['voice_activity_detector'] = self.vad_url
            if self.nr:
                urls['speech_enhancer'] = self.speech_enhancer_url
            if self.sp:
                urls['speech_separator'] = self.speech_separator_url
            if self.speaker_verification:
                urls['audio_inferer'] = getattr(self.audio_recognizer, 'audio_inferer_url', None)
            session_provenance.record_services_later(self.mongo_client, self.session_id, entry, urls,
                                                     self.project_dir, 'asr-base', log=self.logger)
        except Exception as e:
            self.logger.warning(f"Could not note in session {self.session_id} what ASR base {self.id} runs with: {e}")

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

        Returns:
            True when the run ended with STOP, False when it ended with an error.
        """
        restart = isinstance(e, RecordingError)
        if not restart:
            # noted first: the thread joins and the last chunks' HTTP calls below can take long, and
            # a process killed meanwhile (a closed terminal window) would never note that it left
            self._leave_session()

        if e:
            self._stop_threads()
        else:
            self.logger.info("All threads stopped properly.")

        # process any remaining audio chunks before cleanup
        self._process_final_chunks()

        current_bucket = self.session_id  # assign bucket name before cleaning up
        self._clean_up()
        if restart:
            self.logger.info("Restarting recognizing service.")
            return self._start_recognition(current_bucket) is True
        return e is None

    def _reset(self):
        """Reset the ASR base.

        Reinitialize the ASR base by calling the constructor with the current configuration,
        logs the reset status, and performs garbage collection. One launched from the console
        keeps its session and Bases entry.
        """
        self.__init__(project_dir=self.project_dir, config_path=self.config_path, mode=self.mode,
                      vad=self.vad, nr=self.nr, tr=self.tr, sp=self.sp, store=self.store, hsr=self.hsr,
                      session_id=self.launch_session_id,
                      base=str(self.id) if self.launch_session_id else None,
                      speakers=self.launch_speakers)
        self.logger.info(f"Profiles directory reset to {self.profiles_dir}")
        gc.collect()

    def _switch_mode(self):
        """Switch the operating mode between 'capture', 'analyze' and 'live'."""
        self.mode = get_base_mode()
        self.logger.info(f"Switched to {self.mode} mode.")

    def _continuous_recording(self):
        """Continuously record audio from the audio stream and enqueue it for processing.

        Depending on the operating mode:
          - In 'capture' mode, writes recorded frames to a file.
          - In 'live' mode, puts the audio frame bytes into the audio queue.

        Raises:
            RecordingError: If an error occurs during the recording process.
        """
        # handle file source differently - files need sequential time-based reading
        if self.source == 'file':
            self._continuous_file_reading()
            return
            
        first_time = True
        sub_dir = 'records' if self.mode == 'capture' else 'temp'
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
                if self.mode == 'capture':
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
            sub_dir = 'records' if self.mode == 'capture' else 'temp'
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

                        if self.mode == 'capture':
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

    def _speech_label(self) -> str:
        """the name a segment with speech takes without speaker verification: the wearer's
        participant id, else the group's id."""
        return self.participant if self.participant is not None else self.group_speaker_id

    def _segment_energy(self, segment_start_time: float, frames: bytes) -> dict:
        """the level of a segment's raw frames and the base's noise floor at that moment, as a
        recognition carries it (no floor for the first segment of a run: NoiseFloor)."""
        if self._noise_floor is None:
            self._noise_floor = NoiseFloor()
        rms, peak = segment_energy(frames)
        floor = self._noise_floor.update(segment_start_time, rms)
        return energy_record(rms, peak, floor)

    def _segment_levels(self, frames: bytes, energy: dict | None) -> dict | None:
        """a worn microphone's segment level every 100 ms and its floor (levels_record), which its
        recognition carries so that the synchronizer's bucket holds every worn microphone's levels,
        speech or silence; None for a base that keeps none."""
        if not self._keeps_levels() or energy is None:
            return None
        return levels_record(level_trace(frames, 16000), energy.get('floor_db'))

    def _keeps_levels(self) -> bool:
        """whether this base's transcripts and recognitions carry their level trace: a worn
        microphone's (its words are decided from the levels of every worn microphone), whose chunks
        are its raw 16 kHz frames."""
        return self.participant is not None and not getattr(self, 'sp', False)

    def _chunk_levels(self, frames: bytes) -> dict | None:
        """a finished chunk's level every 100 ms from its start and this base's noise floor as it
        stood when the chunk ended (levels_record), taken then, in the order the segments arrived,
        so a live run and a replay give the same; None for a base that keeps none."""
        if not self._keeps_levels():
            return None
        floor = self._noise_floor.last if self._noise_floor is not None else None
        return levels_record(level_trace(frames, 16000), floor)

    def _is_speech(self, processed_audio_path, rms_value, peak_value, energy) -> bool:
        """whether a segment is speech: VAD kept speech in it and, with the absolute gate, its level
        after gain, noise reduction and VAD is over rms_threshold and rms_peak_threshold; with the
        relative gate, its raw level stands speech_gate_snr_db over this base's noise floor (its
        energy)."""
        if not processed_audio_path:
            return False
        if self.speech_gate == 'relative':
            return relative_speech(energy, self.speech_gate_snr_db)
        return bool(rms_value > self.rms_threshold and peak_value > self.rms_peak_threshold)

    def _amplification(self, rms_value, energy) -> float:
        """the factor score_amplified multiplies a recognized speaker's similarity by: log(rms) /
        log(rms_threshold) with the absolute gate, the segment's snr over speech_gate_snr_db with
        the relative one (1.0 when that has no meaning)."""
        if self.speech_gate == 'relative':
            snr = snr_db(energy)
            if snr is None or self.speech_gate_snr_db <= 0:
                return 1.0
            return snr / self.speech_gate_snr_db
        return np.log(rms_value) / np.log(self.rms_threshold)

    def _energy_ratio(self, rms_value, peak_value, energy) -> float:
        """how far over the gate an unrecognized segment stands, 0-1, which scales its similarity
        below the threshold: (rms + peak) / (rms + peak + rms_threshold + rms_peak_threshold) with
        the absolute gate, snr / (snr + speech_gate_snr_db) with the relative one (0.5 when that has
        no meaning)."""
        if self.speech_gate == 'relative':
            snr = snr_db(energy)
            if snr is None or snr <= 0 or self.speech_gate_snr_db <= 0:
                return 0.5
            return snr / (snr + self.speech_gate_snr_db)
        c1 = rms_value + peak_value
        c2 = self.rms_threshold + self.rms_peak_threshold
        return c1 / (c1 + c2)

    def _speech_gate_text(self) -> str:
        """how this base tells speech from silence, in one line."""
        if self.speech_gate == 'relative':
            return (f"Speech gate: relative, {self.speech_gate_snr_db:g} dB or more over this base's own noise "
                    f"floor, before gain (rms_threshold and rms_peak_threshold unused).")
        return (f"Speech gate: absolute, rms over {self.rms_threshold} and peak over {self.rms_peak_threshold} "
                f"after {self.gain:g} dB gain.")

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
                # the level of the raw segment, before any gain, against this base's noise floor
                energy = self._segment_energy(segment_start_time, frames)
                write_bytes_to_wav(segment_audio_path, frames)  # default: 16000 Hz, 16-bit, mono

                # audio pre-processing
                apply_gain(segment_audio_path, self.gain)
                processed_audio_path = self._audio_preprocessing(segment_audio_path, inplace=1)

                # evaluate energy levels for quality check
                rms_value, peak_value = get_energy_level(segment_audio_path, verbose=True)
                if self._is_speech(processed_audio_path, rms_value, peak_value, energy):
                    speaker = 'unknown'
                else:
                    speaker = 'silent'

                duration = self.recognize_duration
                similarity = 0

                if speaker == 'unknown' and not self.speaker_verification:
                    speaker = self._speech_label()
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
                            similarity = min(similarity * self._amplification(rms_value, energy), 1)
                    else:
                        similarity = self.threshold * self._energy_ratio(rms_value, peak_value, energy)

                self._assemble_chunk_with_hsr(speaker, segment_start_time, frames)
                self._publish_recognition(segment_start_time, recognize_start_time, [speaker],
                                          [np.round(np.float64(similarity), 4)], [duration], energy=energy,
                                          levels=self._segment_levels(frames, energy))

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
                # the relative gate needs the raw segment's level against this base's floor
                energy = self._segment_energy(segment_start_time, frames) if self.speech_gate == 'relative' else None
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
                if self._is_speech(processed_audio_path, rms_value, peak_value, energy):
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
                        similarity = min(similarity * self._amplification(rms_value, energy), 1)

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

        In 'analyze' mode, this method loads .wav files from the records directory,
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
                frames, speaker, chunk_start_time, chunk_end_time, levels = self.transcription_queue.get(timeout=2)
                transcribe_result = self._transcribe(frames, frame_rate)
                self._upload_transcription(speaker, transcribe_result, chunk_start_time, chunk_end_time,
                                           levels=levels)
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
                chunk_end_time = segment_start_time + self.recognize_duration
                if self.max_chunk_duration and chunk_end_time - chunk_start_time >= self.max_chunk_duration:
                    # the chunk has grown to its cap: it goes on its own. without speaker verification
                    # (group, wearer) it is cut at its quietest moment before the cap and the rest
                    # starts the next chunk (_quiet_cuts); else, with too little audio for that, or when
                    # that moment ends the chunk, it goes whole, and the next segment of this speaker
                    # starts a new one
                    cut = quiet_cut(last_speaker_frames, fr, self.max_chunk_duration) if self._quiet_cuts() else None
                    if cut is None or 2 * cut >= len(last_speaker_frames):
                        self._finish_chunk(speaker, chunk_start_time, chunk_end_time, last_speaker_frames, fr)
                        self.speaker_frames_dict[speaker] = (chunk_end_time, b'')
                    else:
                        # the rest ends with this segment's audio, so its start is counted back from this
                        # segment's stamp rather than on from the chunk's start: audio a live stream lost
                        # in an earlier segment shifts no later chunk
                        rest = last_speaker_frames[2 * cut:]
                        cut_time = segment_start_time + (len(frames) - len(rest)) / (2 * fr)
                        cut_time = min(max(cut_time, chunk_start_time), chunk_end_time)
                        self._finish_chunk(speaker, chunk_start_time, cut_time, last_speaker_frames[:2 * cut], fr)
                        self.speaker_frames_dict[speaker] = (cut_time, rest)
                else:
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
                    self._finish_chunk(self.last_speaker, chunk_start_time, chunk_end_time, chunk_frames, fr)

                self.speaker_frames_dict[speaker] = (segment_start_time, frames)

        self.last_speaker = speaker

    def _quiet_cuts(self) -> bool:
        """whether a chunk that reaches its cap is cut at its quietest moment before it
        (chunking.quiet_cut): a base without speaker verification (group, wearer) that separates no
        speech, with a cap of at least two segments. The cut never falls before half the cap, so the
        rest it leaves is shorter than half the cap plus a segment, which such a cap keeps below the
        cap: rests cannot pile up from one chunk to the next."""
        cap = self.max_chunk_duration
        return (bool(cap) and not self.speaker_verification and not self.sp
                and float(cap) >= 2 * float(self.recognize_duration or 0))

    def _finish_chunk(self, speaker: str, chunk_start_time: float, chunk_end_time: float, chunk_frames: bytes,
                      framerate: int):
        """a chunk of one speaker is complete (the speaker changed, or the chunk reached its cap):
        it goes for transcription unless no one spoke, and to disk when audio is stored."""
        if speaker not in ['silent', 'unknown']:
            self._enqueue_transcription(chunk_frames, speaker, chunk_start_time, chunk_end_time)
        if self.store:
            chunk_audio_path = os.path.join(self.audio_dir, 'chunks', f'{speaker}_chunk_{chunk_start_time}.wav')
            write_bytes_to_wav(chunk_audio_path, chunk_frames, framerate=framerate)
            if speaker != 'silent':
                normalize_decibel(chunk_audio_path, rms_level=-20)

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

        If transcription is enabled (self.tr), enqueues the audio frames along with speaker and timing
        details, and, for a worn microphone, the chunk's level trace and floor as they are now.

        Args:
            frames: audio frames (bytes) to be transcribed.
            speaker: recognized speaker for the audio chunk.
            chunk_start_time: start time of the audio chunk.
            chunk_end_time: end time of the audio chunk.
        """
        if self.tr:
            self.transcription_queue.put((frames, speaker, chunk_start_time, chunk_end_time,
                                          self._chunk_levels(frames)))

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
                    self._upload_transcription(speaker, transcribe_result, chunk_start_time, chunk_end_time,
                                               levels=self._chunk_levels(chunk_frames))
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
                                            self.speech_transcriber_url, language=self.language,
                                            diarize=self.diarize)
        if response is None:
            return {}
        self._check_language(response)
        self._check_diarize(response)
        return response

    def _check_diarize(self, response: dict) -> None:
        """say once, when this base asked for speaker turns, that the speech transcriber gave none:
        its backend cannot (only a local WhisperX model diarizes), its pyannote pipeline could not
        be made (no Hugging Face token, terms not accepted), or it runs code from before a request
        could ask."""
        if response.get("diarized") and response.get("diarization") and not self._embeddings_told \
                and not isinstance(response.get("speaker_embeddings"), dict):
            # turns without embeddings: a service built before it returned them, whose chunks go unlinked
            self._embeddings_told = True
            self.logger.warning(
                "The speech transcriber diarized this chunk without speaker embeddings, so its speakers are not "
                "linked into the voices of the session: its service runs code from before it returned them (or a "
                "WhisperX whose pipeline cannot). Transcripts keep each chunk's own SPEAKER_NN until the speech "
                "transcriber image is built anew and started again.")
        if not self.diarize or self._diarize_told:
            return
        if response.get("diarized"):
            self._diarize_told = True
            return
        self._diarize_told = True
        self.logger.warning(
            "This base asked the speech transcriber to diarize, and it did not: its backend cannot (only a "
            "local WhisperX model diarizes), its pyannote pipeline could not be made (see its log: the model "
            "is gated on huggingface.co and needs hf_token or HF_TOKEN), or its service runs code from before "
            "a request could ask. Transcripts come without speaker turns until the ASR Server card is started "
            "again with that put right.")

    def _check_language(self, response: dict) -> None:
        """say once, when this base asked for a language, that the speech transcriber did not
        transcribe in it: its service answers without one when it runs code from before a request
        could name a language."""
        if not self.language or self._language_told:
            return
        said = str(response.get("language") or "")
        if said.split("-")[0].lower() == self.language.split("-")[0].lower():
            self._language_told = True
            return
        self._language_told = True
        self.logger.warning(
            f"This base asked the speech transcriber for '{self.language}', and it "
            f"{f'transcribed in {said}' if said else 'did not say which language it used'}: its service "
            f"may run code from before a request could name a language. Start the ASR Server card again, "
            f"which builds the speech transcriber anew.")

    def _upload_transcription(self, speaker: str, transcribe_result: dict, chunk_start_time: float, chunk_end_time: float,
                              levels: dict | None = None):
        """Upload the transcribed speech chunk to the database.

        Constructs a transcription record with speaker, text content, and timing information,
        then writes it to InfluxDB for persistent storage. Also displays the transcription
        in the console for real-time monitoring.

        Args:
            speaker: identified speaker for the transcribed chunk.
            transcribe_result: the result of the transcription, including text and words.
            chunk_start_time: start timestamp of the audio chunk.
            chunk_end_time: end timestamp of the audio chunk.
            levels: a worn microphone's level trace of the chunk and its floor (levels_record), which
                decides its words (default: None, stored without).
        """
        from openmmla.utils.constants import EVENT_TYPE_ASR_TRANSCRIPTION
        words = transcribe_result.get("words", [])
        turns = transcribe_result.get("diarization")
        voices = self._link_voices(transcribe_result)
        if voices is not None:
            # each diarized word and turn carries the session voice of its speaker beside its SPEAKER_NN
            words, turns = with_voices(words, voices), with_voices(turns, voices)
        fields = {
            "window_start_time": chunk_start_time,
            "window_end_time": chunk_end_time,
            "text": transcribe_result.get("text", ""),
            "words": json.dumps(words),
            "speaker": speaker,
        }
        if turns is not None:
            # the anonymous speaker turns of the chunk, in seconds from its start, as the words are
            fields["diarization"] = json.dumps(turns)
        if voices is not None:
            # {SPEAKER_NN: {voice, similarity}}: which voice of the session each speaker of the chunk is,
            # and the registry that numbered them (a base launched again into the session numbers anew)
            fields["voices"] = json.dumps(voices)
            fields["voice_registry"] = self._voice_key
        heard = f" ({len({turn.get('speaker') for turn in turns})} speakers, {len(turns)} turns)" if turns else ""
        if voices:
            heard += f" voices {', '.join(str(voice) for voice in sorted(link['voice'] for link in voices.values()))}"
        print(f"{GREEN}[Speaker Transcription]{ENDC}{chunk_start_time}: "
              f"{GREEN}{speaker} : {transcribe_result.get('text', 'N/A')}{heard}{ENDC}")
        if self.participant is not None:
            # a worn microphone's transcript: whose it is, and a point of its own (several bases' chunks end together)
            fields["participant"] = self.participant
            fields["attribution"] = "energy"
            if levels is not None:
                # the chunk's level every 100 ms from its start and its floor: a JSON string, as write_event
                # would str() a dict
                fields["levels"] = json.dumps(levels)
            self.influx_client.write_event(self.session_id, EVENT_TYPE_ASR_TRANSCRIPTION, fields,
                                           timestamp=transcript_time(chunk_end_time,
                                                                     f'{self.base_type.lower()}_{self.id}'))
        else:
            self.influx_client.write_event(self.session_id, EVENT_TYPE_ASR_TRANSCRIPTION, fields)

    def _voices_for_session(self):
        """keeps the voice registry while the session is the one it is of (a run restarted after a
        recording error goes on numbering the same voices) and starts a new one otherwise."""
        if self._voice_registry is None or self._voice_session != self.session_id:
            self._new_voice_registry()

    def _new_voice_registry(self):
        """a registry for the voices of this base's session, with the key its transcripts name it by
        (voice_registry: the base and the moment the registry began)."""
        self._voice_registry = VoiceRegistry()
        self._voice_session = self.session_id
        self._voice_key = f"{str(getattr(self, 'base_type', 'base')).lower()}_{getattr(self, 'id', None)}@{time.time():.3f}"

    def _link_voices(self, transcribe_result: dict) -> dict | None:
        """the session voices of a diarized chunk's speakers, {SPEAKER_NN: {voice, similarity}}
        (openmmla.bases.asr.voices): the chunks reach here one at a time and in order, live and in
        replay alike, so each is linked only to the voices of the chunks before it (under a lock: a
        transcription thread that outlived its stop may still link while the final chunks are). None
        when the chunk has no turns or the speech transcriber returned no speaker embeddings (a
        backend or a WhisperX that cannot, or a service from before it could)."""
        embeddings = transcribe_result.get("speaker_embeddings")
        turns = transcribe_result.get("diarization")
        if not turns or not isinstance(embeddings, dict):
            return None
        with self._voice_lock:
            if self._voice_registry is None:
                self._new_voice_registry()
            return self._voice_registry.link(embeddings, speaker_seconds(turns))

    def _publish_recognition(self, segment_start_time: float, recognize_start_time: float, speakers: list[str],
                             similarities: list[float], durations: list[float], energy: dict | None = None,
                             levels: dict | None = None):
        """Log and publish speaker recognition results via MQTT.

        Constructs a JSON record with recognition details and publishes it on the designated MQTT channel.
        The record includes speaker identities, similarity scores, and timing information.

        Args:
            segment_start_time: start time of the recorded segment (timestamp).
            recognize_start_time: start time of the recognition process (timestamp).
            speakers: list of recognized speakers.
            similarities: list of similarity scores (0.0-1.0) corresponding to speakers.
            durations: list of audio durations in seconds for each speaker segment.
            energy: the segment's level and this base's noise floor (rms_db, peak_db, floor_db), which
                the synchronizer's energy vote reads (default: None, sent without).
            levels: a worn microphone's segment level every 100 ms and its floor (levels_record), which
                the synchronizer keeps in the bucket (default: None, sent without).
        """
        base_recognition_result = {
            'base_id': f'{self.base_type.lower()}_{self.id}',
            'speakers': json.dumps(speakers),
            'similarities': json.dumps(similarities),
            'durations': json.dumps(durations),
            'segment_start_time': segment_start_time
        }
        if energy is not None:
            base_recognition_result['energy'] = energy  # a JSON object inside the payload
        if levels is not None:
            base_recognition_result['levels'] = levels  # a JSON object inside the payload
        if self.participant is not None:
            base_recognition_result['participant'] = self.participant
        worn = ""
        if self.participant is not None and energy is not None:
            floor = energy.get('floor_db')
            worn = (f" [{self.participant}, {energy['rms_db']:.1f} dB, "
                    f"{'no floor yet' if floor is None else f'floor {floor:.1f}'}]")
        print(f"{BLUE}[Speaker Recognition]{ENDC}{base_recognition_result['segment_start_time']}: "
              f"{BLUE}{base_recognition_result['speakers']}{ENDC}, similarity: {base_recognition_result['similarities']},"
              f"processed time: {time.time() - recognize_start_time} seconds{worn}")
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

        if self.mode == 'analyze':
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
        sentences = "\n".join(f"{i}. {sentence}" for i, sentence in enumerate(REGISTRATION_SENTENCES, 1))
        input(f"Press the Enter key to start recording, and read the following sentence in {seconds} seconds:\n"
              f"{sentences}")
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
    
