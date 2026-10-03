import datetime
import os
import select
import socket
import struct
import subprocess
import threading
import time
from collections import deque

import numpy as np
import pyaudio
import soundfile as sf

try:
    from pylsl import local_clock, resolve_byprop, StreamInlet
except ImportError:
    local_clock = None
    resolve_byprop = None
    StreamInlet = None

from openmmla.streams.resampling import resample_audio, ResampleMethod
from openmmla.utils.constants import normalize_source
from openmmla.utils.logger import get_logger
from openmmla.utils.sockets import clear_socket_udp
from openmmla.utils.threads import RaisingThread
from .frame import AudioFrame
from .stream_buffer import RingBuffer
from .stream_receiver import StreamReceiver, StreamUnavailable, as_sentence
from . import stream_receiver as receiver

logger = get_logger(__name__)

# how long one try at a network stream waits for ffmpeg's first chunk: an unreachable Stream Server costs
# this, and an RTMP pull probes ~6 s before it decodes
OPEN_TIMEOUT = 15.0
# no byte from ffmpeg for this long: the pull stalled without ending (a relay stall, a server gone without
# closing the connection), which ffmpeg itself never notices. As VideoStream's STALL_SECONDS
STALL_SECONDS = 10.0

# Define supported formats for our module.
SUPPORTED_FORMATS = {
    'int16': {'sample_width': 2, 'dtype': np.int16},
    'int32': {'sample_width': 4, 'dtype': np.int32},
    'float32': {'sample_width': 4, 'dtype': np.float32},
}

# Mapping for PyAudio formats.
PA_FORMATS = {
    'int16': pyaudio.paInt16,
    'int32': pyaudio.paInt32,
    'float32': pyaudio.paFloat32,
}

# Mapping for FFmpeg's raw audio format and codec names.
RTMP_FORMATS = {
    'int16': 's16le',
    'int32': 's32le',
    'float32': 'f32le',
}

RTMP_CODEC = {
    'int16': 'pcm_s16le',
    'int32': 'pcm_s32le',
    'float32': 'pcm_f32le',
}

# packet layouts accepted on udp/tcp sources. 'timestamped' is the wearable badge
# protocol (audio_streaming_udp_ms.py): an 18-byte header with a packet counter and
# the sender's UTC time, followed by one chunk of PCM. 'raw' is a header-less PCM
# byte stream, as sent by the badge firmware without '_ms' and by FFmpeg
# (`-c:a pcm_s16le -f s16le udp://...`). 'auto' sniffs the first packet.
PACKET_FORMATS = ('auto', 'timestamped', 'raw')
SOCKET_HEADER_FORMAT = '>I7H'
SOCKET_HEADER_SIZE = struct.calcsize(SOCKET_HEADER_FORMAT)  # 18 bytes
RAW_TIMESTAMP_MAX_DRIFT = 0.5  # seconds before raw stream timestamps are re-anchored to the receiver clock
SOCKET_RECV_SIZE = 65536


def _now() -> float:
    """current unix time, wrapped so tests can control the receiver clock."""
    return time.time()


def parse_socket_header(data: bytes) -> float | None:
    """Return the sender timestamp encoded in a wearable packet header, or None.

    The header is '>I7H': a packet counter, then year, month, day, hour, minute,
    second and milliseconds in UTC. Only structurally valid dates are accepted,
    which is enough to tell a header apart from PCM samples: six consecutive
    samples would all have to fall inside the calendar ranges.
    """
    if len(data) < SOCKET_HEADER_SIZE:
        return None
    _counter, year, month, day, hour, minute, second, millis = struct.unpack(
        SOCKET_HEADER_FORMAT, data[:SOCKET_HEADER_SIZE])
    if not (1 <= month <= 12 and 1 <= day <= 31 and hour < 24 and minute < 60
            and second < 60 and millis < 1000):
        return None
    try:
        timestamp = datetime.datetime(
            year, month, day, hour, minute, second, millis * 1000,  # milliseconds to microseconds
            tzinfo=datetime.timezone.utc
        ).timestamp()
    except ValueError:
        return None
    return timestamp


def _connection_of(frame) -> int | None:
    """the connection a network stream's chunk came on; None for every other source, which is never cut"""
    metadata = getattr(frame, 'metadata', None)
    return metadata.get('connection') if isinstance(metadata, dict) else None


def write_frame_to_wav(output_path: str, audio_frames: AudioFrame):
    """Write AudioFrame to a wave file. The default format is PCM_16.

    Args:
        output_path: The file path where the wave file will be saved.
        audio_frames: The audio frames to write.
    """
    if audio_frames.format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {audio_frames.format}")
    sf.write(output_path, audio_frames.data, audio_frames.sample_rate)


class AudioStream(StreamReceiver):
    """Audio stream implementation for continuous audio data capture."""

    default_log = logger

    def __init__(self, source: str, **kwargs):
        """Initialize audio stream.

        Args:
            source (str): Stream source type ('pyaudio', 'udp', 'tcp', 'stream', 'lsl', 'file';
                'rtmp' is an alias of 'stream').

        Keyword Args:
            buffer_duration (float, optional): Duration of the ring buffer in seconds (default: 5.0)
            format (str, optional): Audio format (default: 'int16')
            channels (int, optional): Number of channels of the audio stream (default: 1)
            channel_select (int, optional): selected channel index of the audio stream (pyaudio only) (default: None)
            rate (int, optional): Sample rate in Hz (default: 16000)
            chunk_size (int, optional): Size of audio chunk to read in frames (default: 512)
            resample_method (ResampleMethod, optional): Method for resampling (default: AUDIO_LIBROSA)
            host (str, optional): Socket host (required for 'udp' or 'tcp' sources)
            port (int, optional): Socket port (required for 'udp' or 'tcp' sources)
            packet_format (str, optional): udp/tcp packet layout: 'timestamped' (18-byte wearable header +
                one PCM chunk), 'raw' (header-less PCM byte stream, e.g. from FFmpeg) or 'auto' to sniff
                the first packet (default: 'auto')
            url (str, optional): stream URL, rtmp:// rtsp:// or srt:// (required for the 'stream' source)
            file_path (str, optional): Path to the audio file (required for 'file' source)
            timestamp_offset (float, optional): seconds added to every live chunk's stamp, the measured
                delay of this stream as a negative number (default: 0)
            connect_wait (float, optional): seconds a 'stream' or 'lsl' source that is not up yet is waited
                for at start (default: 30); a microphone, a socket or a file is tried at once
            reconnect_wait (float, optional): seconds a 'stream' or 'lsl' source that dropped is opened
                again for before read() raises StreamUnavailable (default: 3600)
            log (logging.Logger, optional): the caller's logger, for the waits and drops of the stream
            abort_event (threading.Event, optional): the caller's stop, which ends a wait for the stream
        """
        super().__init__(**kwargs)
        self.source = normalize_source(source)

        # Stream configuration
        self.buffer_duration = kwargs.get('buffer_duration', 5.0)
        self.format = kwargs.get('format', 'int16')
        if self.format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {self.format}")
        self.sample_width = SUPPORTED_FORMATS[self.format]['sample_width']
        self.dtype = SUPPORTED_FORMATS[self.format]['dtype']

        self.channels = int(kwargs.get('channels') or 1)  # a 'channels:' left empty in a config is 1 too
        self.channel_select = kwargs.get('channel_select', None)
        self.rate = kwargs.get('rate', 16000)
        self.chunk_size = kwargs.get('chunk_size', 512)
        self.resample_method = kwargs.get('resample_method', ResampleMethod.AUDIO_LIBROSA)

        # PyAudio objects
        if self.source == 'pyaudio':
            value = kwargs.get('input_device_index')
            self.input_device_index = int(value) if value is not None else None
            self.p = None
            self.stream = None

        # Socket objects for UDP/TCP sources
        if self.source in ['udp', 'tcp']:
            self.host = self.require_kwarg(kwargs, 'host', "UDP/TCP source requires a 'host' parameter")
            self.port = self.require_kwarg(kwargs, 'port', "UDP/TCP source requires a 'port' parameter")
            self.packet_format = str(kwargs.get('packet_format') or 'auto').strip().lower()
            if self.packet_format not in PACKET_FORMATS:
                raise ValueError(f"Unsupported packet_format: {self.packet_format}, must be one of {PACKET_FORMATS}")
            self.sock = None
            self.conn = None  # For TCP connection
            self._frame_bytes = self.channels * self.sample_width
            self._chunk_bytes = self.chunk_size * self._frame_bytes
            self._reset_socket_framing()

        # network stream objects: ffmpeg decodes the URL to raw PCM
        if self.source == 'stream':
            self.url = self.require_kwarg(kwargs, 'url', "Stream source requires a 'url' parameter")
            self.rtmp_url = self.url  # name from before MediaMTX, kept for callers
            self.ffmpeg_proc: subprocess.Popen | None = None
            self._first_chunk = None          # what the probe of a try read, handed to the buffer first
            self._stream_ended = False        # ffmpeg's output ended, or stalled: the stream dropped
            self._stalled_for = None          # seconds without a byte, when it stalled rather than ended
            self._ffmpeg_tail = deque(maxlen=5)  # ffmpeg's last lines, the reason a try failed
            self._proc_lock = threading.Lock()

        # LSL objects
        if self.source == 'lsl':
            self.lsl_name = self.require_kwarg(kwargs, 'lsl_name', "LSL source requires a 'lsl_name' parameter")
            self.lsl_inlet = None
            self.lsl_offset = None

        # File objects
        if self.source == 'file':
            self.file_path = self.require_kwarg(kwargs, 'file_path', "File source requires a 'file_path' parameter")
            self.file_data = None
            self.file_sample_rate = None
            self._converted_file_path = None  # Track if we created a temporary converted file

        # only a network stream is waited for: a microphone, a socket or a file is there or it is not
        self._set_waits(kwargs)

        # Frame metadata
        self._frame_metadata = {
            'sample_rate': self.rate,
            'channels': self.channels if not self.channel_select else 1,
            'format': self.format
        }
        # the network stream's chunks carry the number of their connection: read() never joins audio from
        # both sides of a reconnect into one segment
        self._connection = 0
        self._connection_metadata = self._frame_metadata

        # Calculate buffer size in frames
        # never fewer than two slots: the tail of a one-slot ring never moves,
        # and read() waits for it to
        buffer_frames = max(2, int(self.rate * self.buffer_duration / self.chunk_size))
        self.buffer = RingBuffer(buffer_frames)

        # Last read position tracking
        self._last_read_pos = -1

        # Threading control
        self._stop_event = threading.Event()
        self._receive_thread = None

    def start(self) -> None:
        """Start the audio stream and initialize data capturing source. A network stream that is not up
        yet is waited for, up to connect_wait seconds, and start() returns once ffmpeg delivered its first
        chunk; one that does not come up raises StreamUnavailable. A stop() from another thread or the
        abort_event (the ASR base's STOP) ends the wait, and start() then starts nothing."""
        self.stop()
        # cleared before the opening, which may wait: a stop() from another thread then ends that wait
        self._stop_event.clear()
        self.state, self.failure = 'connecting', None
        try:
            if self.source == 'pyaudio':
                self._initialize_pyaudio()
            elif self.source == 'udp':
                self._initialize_udp()
            elif self.source == 'tcp':
                self._initialize_tcp()
            elif self.source == 'stream':
                self._initialize_rtmp()
            elif self.source == 'lsl':
                self._initialize_lsl()
            elif self.source == 'file':
                self._initialize_file()
                # File source doesn't need a receive thread - data is read on demand
                self.state = 'live'
                self.log.info(f"Audio stream started with source: {self.source}")
                return
            else:
                raise ValueError(f"Unsupported source type: {self.source}")
        except Exception as e:
            self.state, self.failure = 'failed', as_sentence(e)
            raise
        except BaseException:
            self.state = 'stopped'
            raise
        if self._stopping():
            # the base's STOP came during the wait: no error, nothing started
            self._cleanup_source()
            self.state = 'stopped'
            return

        self.state = 'live'
        self._receive_thread = RaisingThread(target=self._receive_loop, name=f'audio-{self.source}')
        self._receive_thread.daemon = True
        self._receive_thread.start()
        self.log.info(f"Audio stream started with source: {self.source}")

    def stop(self) -> None:
        """Stop the audio stream and clean up resources."""
        self._stop_event.set()  # Signal the thread to stop
        if self.source == 'stream':
            # first: a receive thread blocked on ffmpeg's stdout (a stall MediaMTX has not ended) returns
            # once ffmpeg is gone
            self._cleanup_rtmp()

        thread, self._receive_thread = self._receive_thread, None
        if thread is not None and threading.current_thread() is not thread:
            try:
                thread.join(timeout=5)
            except Exception as e:
                self.log.warning(f"During thread stopping, caught: {e}", exc_info=True)
            if thread.is_alive():
                # still inside a read of its source, which must not be closed under it: the thread closes
                # it itself when that read returns
                self.log.warning(f"{self._what()} is still in a read; it is closed when that read returns.")
                self._gap_summary(self._what())
                self._last_read_pos = -1
                if self.state != 'failed':
                    self.state = 'stopped'
                return

        self._cleanup_source()
        self._gap_summary(self._what())
        self._last_read_pos = -1
        if self.state != 'failed':
            self.state = 'stopped'

    def _cleanup_source(self) -> None:
        """release what the source holds; safe to call again"""
        if self.source == 'pyaudio':
            self._cleanup_pyaudio()
        elif self.source in ['udp', 'tcp']:
            self._cleanup_socket()
        elif self.source == 'stream':
            self._cleanup_rtmp()
        elif self.source == 'lsl':
            self._cleanup_lsl()
        elif self.source == 'file':
            self._cleanup_file()

    def _what(self) -> str:
        """the stream, as the log lines name it"""
        if self.source == 'stream':
            return f"Audio stream {self.url}"
        if self.source == 'lsl':
            return f"LSL stream '{self.lsl_name}'"
        if self.source in ('udp', 'tcp'):
            return f"Audio {self.source} source {self.host}:{self.port}"
        if self.source == 'pyaudio':
            return f"Microphone {self.input_device_index}"
        return f"Audio file {getattr(self, 'file_path', '')}"

    def read(self, duration: float, target_rate: int | None = None, timeout: float = 5.0,
             latest: bool = False, start_time: float = 0.0) -> AudioFrame | None:
        """Read audio data with optional resampling.

        Args:
            duration (float): Duration to read in seconds.
            target_rate (int, optional): Optional target sample rate for resampling.
            timeout (float): Maximum time to wait for data in seconds.
            latest (bool): Whether to read from the most recent data or continue from last position.
            start_time (float): Start time in seconds for file source (only used with file source).

        Returns:
            AudioFrame: Containing the requested duration of audio data, or None if timeout is reached.
        """
        # File source - direct read without buffering
        if self.source == 'file':
            return self._read_from_file(start_time, duration, target_rate)

        # a stream that did not come back, or a source that stopped for good: the base ends its run on it
        if self.state == 'failed':
            raise StreamUnavailable(self.failure)

        # Other sources
        frames_needed = int(duration * self.rate / self.chunk_size)
        total_frames = []
        start_time_actual = time.time()

        if latest:
            self._last_read_pos = -1

        while len(total_frames) < frames_needed:
            current_tail = self.buffer.get_tail()

            if self._last_read_pos == -1:
                self._last_read_pos = current_tail
                continue

            if current_tail == self._last_read_pos:
                if self.state == 'failed':
                    raise StreamUnavailable(self.failure)
                if time.time() - start_time_actual > timeout:
                    # while the stream is opened again it says so itself; a live one that falls silent is
                    # worth a line
                    self._say_quietly_unless_live("Timeout reached while waiting for frames.")
                    break
                time.sleep(1)
                continue

            remaining_frames = frames_needed - len(total_frames)
            available_frames = self.buffer.frames_available(self._last_read_pos)
            end_pos = (self._last_read_pos + remaining_frames) % self.buffer.size \
                if available_frames >= remaining_frames else current_tail

            new_frames = self.buffer.get(start_pos=self._last_read_pos, end_pos=end_pos)
            first = (total_frames or new_frames)[0] if (total_frames or new_frames) else None
            connection = _connection_of(first)
            cut = next((i for i, frame in enumerate(new_frames) if _connection_of(frame) != connection), None)
            if cut is not None:
                # audio from before a reconnect and after it is never one segment: the stamp of its first
                # chunk would be put on audio that came a gap later. The rest is the next read's
                total_frames.extend(new_frames[:cut])
                self._last_read_pos = (self._last_read_pos + cut) % self.buffer.size
                break
            total_frames.extend(new_frames)
            self._last_read_pos = end_pos
            start_time_actual = time.time()

        return self._process_frames(total_frames, target_rate)

    def _say_quietly_unless_live(self, message: str) -> None:
        """a warning while the stream is live; a debug line while it is opened again (or not started),
        which its own lines already say, so that a long outage does not print it every few seconds"""
        (self.log.warning if self.state == 'live' else self.log.debug)(message)

    def _process_frames(self, frames: list[AudioFrame], target_rate: int | None) -> AudioFrame | None:
        """Process collected frames and apply resampling if needed.

        Args:
            frames: List of AudioFrames to process.
            target_rate: Optional target sample rate for resampling.

        Returns:
            AudioFrame: Processed AudioFrame or None if no frames available.
        """
        if not frames:
            self._say_quietly_unless_live("No frames collected within timeout period.")
            return None

        audio_data = np.concatenate([frame.data for frame in frames])
        start_timestamp = frames[0].timestamp

        if target_rate and target_rate != frames[0].sample_rate:
            audio_data = resample_audio(
                audio_data,
                frames[0].sample_rate,
                target_rate,
                method=self.resample_method
            )

        metadata = {
            'sample_rate': target_rate or frames[0].sample_rate,
            'channels': frames[0].channels,
            'format': frames[0].format,
        }
        connection = _connection_of(frames[0])
        if connection is not None:
            # the segment's connection goes with it (read() never joins two): the base then sees a reconnect
            # between two segments, and closes its open chunk there
            metadata['connection'] = connection

        return AudioFrame(timestamp=start_timestamp, data=audio_data, metadata=metadata)

    def _read_from_file(self, start_time: float, duration: float, target_rate: int | None = None) -> AudioFrame | None:
        """Read audio data directly from file based on start time and duration.
        
        Args:
            start_time (float): Start time in seconds.
            duration (float): Duration to read in seconds.
            target_rate (int, optional): Target sample rate for resampling.
            
        Returns:
            AudioFrame: Audio data for the specified time range, or None if invalid range.
        """
        if self.file_data is None:
            logger.error("File data not loaded")
            return None
            
        # Calculate start and end positions in samples
        start_sample = int(start_time * self.file_sample_rate)
        duration_samples = int(duration * self.file_sample_rate)
        end_sample = start_sample + duration_samples

        # Check bounds
        if start_sample >= len(self.file_data) or start_sample < 0:
            logger.warning(f"Start time {start_time} is out of bounds")
            return None
            
        # Adjust end sample if it exceeds file length
        if end_sample > len(self.file_data):
            end_sample = len(self.file_data)
            logger.info(f"Adjusting duration to fit file length: {(end_sample - start_sample) / self.file_sample_rate:.3f}s")
        
        # Extract audio data
        if self.file_data.ndim == 1:
            audio_data = self.file_data[start_sample:end_sample]
        else:
            audio_data = self.file_data[start_sample:end_sample, :]
            
        # Apply channel selection if needed
        if self.file_data.ndim > 1 and self.channel_select is not None:
            if 0 <= self.channel_select < self.file_data.shape[1]:
                audio_data = audio_data[:, self.channel_select]
            else:
                logger.warning(f"Invalid channel index: {self.channel_select}")
                audio_data = audio_data.mean(axis=1)
        elif self.file_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
            
        # Convert to the requested format
        if audio_data.dtype != self.dtype:
            audio_data = audio_data.astype(self.dtype)
            
        # Resample if necessary
        current_rate = self.file_sample_rate
        if target_rate and target_rate != current_rate:
            audio_data = resample_audio(audio_data, current_rate, target_rate, method=self.resample_method)
            current_rate = target_rate
        elif self.rate != self.file_sample_rate:
            audio_data = resample_audio(audio_data, self.file_sample_rate, self.rate, method=self.resample_method)
            current_rate = self.rate
            
        metadata = {
            'sample_rate': current_rate,
            'channels': 1 if audio_data.ndim == 1 else audio_data.shape[1],
            'format': self.format,
        }
        
        return AudioFrame(timestamp=start_time, data=audio_data, metadata=metadata)

    def _initialize_pyaudio(self, max_retries: int = 3) -> None:
        """Initialize PyAudio stream with configured parameters."""
        self.p = pyaudio.PyAudio()
        if self.format not in PA_FORMATS:
            raise ValueError(f"Unsupported audio format: {self.format} for {self.source}")
        stream_format = PA_FORMATS[self.format]

        self.stream = self.p.open(
            format=stream_format,
            rate=self.rate,
            channels=self.channels,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size
        )

        if not self.stream.is_active():
            if max_retries > 0:
                logger.warning("Failed to initialize PyAudio stream, retrying...")
                self._cleanup_pyaudio()
                self._initialize_pyaudio(max_retries=max_retries - 1)
            else:
                raise RuntimeError(
                    f"Failed to initialize PyAudio audio stream with input device index: {self.input_device_index}")
        else:
            logger.info(
                f"Successfully initialized PyAudio audio stream with input device index: {self.input_device_index}")

    def _initialize_udp(self, max_retries: int = 3) -> None:
        """Initialize UDP socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        clear_socket_udp(self.sock)
        self.sock.settimeout(3)
        try:
            self.sock.getsockname()
            logger.info(f"Successfully initialized UDP audio stream on {self.host}:{self.port}")
        except socket.error:
            if max_retries > 0:
                logger.warning(f"Failed to bind UDP socket on {self.host}:{self.port}, retrying...")
                self._cleanup_socket()
                self._initialize_udp(max_retries=max_retries - 1)
            else:
                raise RuntimeError(f"Failed to initialize UDP audio stream on {self.host}:{self.port}")

    def _initialize_tcp(self, max_retries: int = 3) -> None:
        """Initialize TCP socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        self.conn, addr = self.sock.accept()
        self.conn.settimeout(3)
        if not self.conn:
            if max_retries > 0:
                logger.warning(f"Failed to accept TCP connection from {addr}, retrying...")
                self._cleanup_socket()
                self._initialize_tcp(max_retries=max_retries - 1)
            else:
                raise RuntimeError(f"Failed to initialize TCP audio stream on {self.host}:{self.port}")
        else:
            logger.info(f"Successfully initialized TCP audio stream on {self.host}:{self.port}")

    def _initialize_rtmp(self, wait: float | None = None, after_drop: bool = False) -> None:
        """Open a network stream (rtmp/rtsp/srt): an ffmpeg pulls the URL and writes raw PCM on stdout,
        with read-ahead buffering off to keep latency low. It counts as open once ffmpeg wrote a first
        chunk (kept for the buffer); one that ends at once (404 while nobody publishes on the path) is
        tried again every RETRY_INTERVAL s for up to `wait` seconds (connect_wait at start, reconnect_wait
        after a drop), never spawned in a loop. Returns without ffmpeg when a stop came meanwhile; raises
        StreamUnavailable, with ffmpeg's reason, when the stream did not come up."""
        wait = self.connect_wait if wait is None else wait
        self._ffmpeg_tail.clear()
        if self._wait_for(self._open_rtmp_once, self._what(), wait):
            self._connection += 1
            # every chunk of this connection carries its number: read() never joins audio from both sides
            # of a gap
            self._connection_metadata = dict(self._frame_metadata, connection=self._connection)
            self._stream_ended = False
            self.log.info(f"Successfully initialized audio stream from {self.url}")
            return
        if self._stopping():
            return  # a stop came meanwhile: the caller starts nothing
        said = f" ffmpeg said: {self._ffmpeg_tail[-1]}" if self._ffmpeg_tail else ""
        came = 'come back' if after_drop else 'come up'
        raise StreamUnavailable(f"Audio stream {self.url} did not {came} within {wait:g} s.{said} Is its stream "
                                f"running (Streams tab, Status and Stream Server columns)?")

    def _ffmpeg_command(self) -> list[str]:
        fmt = RTMP_FORMATS[self.format]
        codec = RTMP_CODEC[self.format]
        # errors only on stderr: it is read for the reason a try failed, never shown line by line
        command = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error',
                   '-fflags', 'nobuffer', '-flags', 'low_delay']
        if self.url.startswith('rtsp://'):
            command += ['-rtsp_transport', 'tcp']  # no packet loss on Wi-Fi
        command += [
            '-i', self.url,
            '-f', fmt,
            '-acodec', codec,
            '-ar', str(self.rate),
            '-ac', str(self.channels),
            '-'  # Output to stdout
        ]
        return command

    def _open_rtmp_once(self) -> bool:
        """one try: ffmpeg is started and given OPEN_TIMEOUT seconds for its first chunk. False leaves no
        ffmpeg behind (it is ended and waited for); a missing ffmpeg raises at once, there is nothing to
        wait for"""
        self._cleanup_rtmp()
        try:
            # unbuffered: no byte waits in a reader's buffer where select() cannot see it, so a wait on the
            # pipe is a wait for ffmpeg (_pipe_read)
            proc = subprocess.Popen(self._ffmpeg_command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    bufsize=0)
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg is not installed on this machine: the base decodes network audio "
                               "streams with it.") from e
        with self._proc_lock:
            self.ffmpeg_proc = proc
        self._drain_stderr(proc)
        expected = self.chunk_size * self.channels * self.sample_width
        first = b''
        try:
            if not self._stopping():  # else stopped while it started: ended below
                # a select() bounds a try at an unreachable server: ffmpeg's own -timeout means 'listen' for
                # RTSP in ffmpeg 4. The base's STOP ends this wait too
                first = self._pipe_read(proc, expected, OPEN_TIMEOUT, self._stopping) or b''
        except (OSError, ValueError) as e:
            # a stop() ended ffmpeg and closed its pipe under this read
            self.log.debug(f"Reading ffmpeg's first chunk ended: {e}")
        if len(first) == expected and not self._stopping():
            self._first_chunk = first
            return True
        self._cleanup_rtmp()
        return False

    def _pipe_read(self, proc, size: int, timeout: float, stopping=None) -> bytes | None:
        """`size` bytes of ffmpeg's stdout, which is unbuffered (a read returns what the pipe holds, so the
        pieces are joined here): fewer at its end (ffmpeg exited), None when no byte came for `timeout`
        seconds or `stopping()` came meanwhile (default: stop()). Each wait is a select() in slices of
        0.5 s; on Windows, where select() takes no pipes, the reads block as before and nothing stalls"""
        stopping = stopping or self._stop_event.is_set
        data = b''
        while len(data) < size:
            if os.name != 'nt' and not self._readable(proc.stdout, timeout, stopping):
                return None
            piece = proc.stdout.read(size - len(data))
            if not piece:
                break  # its end: ffmpeg exited
            data += piece
        return data

    @staticmethod
    def _readable(pipe, timeout: float, stopping) -> bool:
        """wait up to `timeout` seconds for the pipe to be readable (data, or its end), in slices so that
        a stop ends the wait"""
        deadline = time.monotonic() + timeout
        while not stopping():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            ready, _, _ = select.select([pipe], [], [], min(0.5, remaining))
            if ready:
                return True
        return False

    def _drain_stderr(self, proc) -> None:
        """keep ffmpeg's last lines (why it could not open), and never let it block on a full pipe over
        hours of decode errors"""
        tail = self._ffmpeg_tail

        def drain():
            try:
                for line in iter(proc.stderr.readline, b''):
                    text = line.decode(errors='replace').strip()
                    if text:
                        tail.append(text)
            except (OSError, ValueError):
                pass  # the pipe was closed: ffmpeg is gone

        proc.stderr_drain = threading.Thread(target=drain, name='ffmpeg-stderr', daemon=True)
        proc.stderr_drain.start()

    def _initialize_lsl(self, wait: float | None = None, after_drop: bool = False):
        """Connect to the LSL stream, waiting for its outlet up to `wait` seconds (connect_wait at start,
        reconnect_wait after a drop). Returns without an inlet when a stop came meanwhile; raises
        StreamUnavailable when no outlet of that name showed up."""
        if resolve_byprop is None or StreamInlet is None or local_clock is None:
            raise ImportError(
                "pylsl package is not installed. Please install it with 'pip install pylsl' to use LSL features."
            )

        def open_once() -> bool:
            self._cleanup_lsl()
            # the resolve itself waits, a retry interval long
            streams = resolve_byprop('name', self.lsl_name, timeout=receiver.RETRY_INTERVAL)
            if not streams:
                return False
            self.lsl_inlet = StreamInlet(streams[0])
            self.lsl_offset = time.time() - local_clock()
            return True

        wait = self.connect_wait if wait is None else wait
        if self._wait_for(open_once, self._what(), wait, pause=False):
            self.log.info(f"Successfully initialized LSL audio stream with inlet: {self.lsl_inlet}")
            return
        if self._stopping():
            self._cleanup_lsl()
            return
        came = 'come back' if after_drop else 'come up'
        raise StreamUnavailable(f"LSL stream '{self.lsl_name}' did not {came} within {wait:g} s: no outlet of "
                                f"that name was found on the network.")

    def _initialize_file(self):
        """Initialize file source."""
        try:
            # Check if file needs format conversion
            actual_file_path = self._prepare_audio_file()
            
            self.file_data, self.file_sample_rate = sf.read(actual_file_path, dtype=self.dtype)
            logger.info(f"Successfully loaded audio file: {actual_file_path}")
            logger.info(f"File duration: {len(self.file_data) / self.file_sample_rate:.2f} seconds, "
                       f"Sample rate: {self.file_sample_rate} Hz, "
                       f"Channels: {1 if self.file_data.ndim == 1 else self.file_data.shape[1]}")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {self.file_path}: {e}") from e
            
    def _prepare_audio_file(self):
        """Prepare audio file for processing, converting format if necessary.
        
        Returns:
            str: Path to the processed audio file (may be the original or a converted version).
        """
        import os
        import tempfile
        from openmmla.utils.audio.files import format_wav
        
        # If already a WAV file with correct parameters, use as-is
        if self.file_path.lower().endswith('.wav'):
            try:
                import wave
                with wave.open(self.file_path, 'rb') as wav_file:
                    if (wav_file.getframerate() == self.rate and 
                        wav_file.getnchannels() == self.channels):
                        logger.info(f"Audio file already in correct format: {self.file_path}")
                        return self.file_path
            except (wave.Error, Exception):
                pass  # Fall through to conversion
        
        # Need to convert format - create temporary file
        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        temp_wav_path = os.path.join(temp_dir, f"{base_name}_converted.wav")
        
        logger.info(f"Converting audio file to standard format: {self.file_path} -> {temp_wav_path}")
        
        # Convert using format_wav
        converted_path = format_wav(
            input_file=self.file_path,
            output_file=temp_wav_path,
            codec="pcm_s16le",
            sample_rate=self.rate,
            channels=self.channels
        )
        
        # Track the converted file for cleanup
        self._converted_file_path = converted_path
        logger.info(f"Audio file converted successfully: {converted_path}")
        return converted_path

    def _cleanup_pyaudio(self) -> None:
        """Clean up PyAudio resources; safe to call again."""
        stream, self.stream = self.stream, None
        p, self.p = self.p, None
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()

    def _cleanup_socket(self) -> None:
        """Clean up socket resources."""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.sock:
            self.sock.close()
            self.sock = None
        self._reset_socket_framing()

    def _cleanup_rtmp(self) -> None:
        """End the stream's ffmpeg and wait for it (no zombie per try), then close its pipes; safe to call
        again and from another thread."""
        with self._proc_lock:
            proc, self.ffmpeg_proc = self.ffmpeg_proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        # ffmpeg is gone, so its stderr ends: its last lines (the reason a try failed) are read in full
        drain = getattr(proc, 'stderr_drain', None)
        if drain is not None and drain is not threading.current_thread():
            drain.join(timeout=1)
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
        self.log.debug("The ffmpeg of the audio stream ended.")

    def _cleanup_lsl(self):
        """Clean up lab streaming layer resources."""
        if self.lsl_inlet:
            self.lsl_inlet.close_stream()
        self.lsl_inlet = None
        self.lsl_offset = None

    def _cleanup_file(self):
        """Clean up file resources."""
        self.file_data = None
        self.file_sample_rate = None
        
        # Clean up converted file if we created one
        if self._converted_file_path and self._converted_file_path != self.file_path:
            try:
                import os
                if os.path.exists(self._converted_file_path):
                    os.remove(self._converted_file_path)
                    logger.info(f"Cleaned up converted file: {self._converted_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up converted file {self._converted_file_path}: {e}")
            finally:
                self._converted_file_path = None

    def _receive_loop(self) -> None:
        """Continuously receive data and store in buffer. A network stream that drops is opened again
        (_reconnect); this loop never raises: a fault reaches the base through read()."""
        failure_count = 0
        try:
            while not self._stop_event.is_set():
                try:
                    # pushed even when a stop came during the read
                    frame = self._stamped(self._read_chunk())
                except RuntimeError:
                    if self._stop_event.is_set():
                        break  # stop() ended the source under the read
                    if self.source != 'stream':
                        raise
                    frame, self._stream_ended = None, True
                if frame:
                    self.buffer.push(frame)
                    failure_count = 0
                    continue
                if self._stop_event.is_set():
                    break  # stop() ended the source under the read: no drop
                if self.source == 'stream' and self._stream_ended:
                    # ffmpeg's output ended, or stalled: the stream dropped. No ten tries, a pipe at its end
                    # answers at once
                    stalled, self._stalled_for = self._stalled_for, None
                    if not self._reconnect(stalled):
                        break
                    continue
                failure_count += 1
                if failure_count < 10:
                    continue
                failure_count = 0
                if self.source == 'lsl':
                    if not self._reconnect():
                        break
                    continue
                self.log.error("10 consecutive frame read failures. Reinitializing stream.")
                if self.source == 'pyaudio':
                    self._cleanup_pyaudio()
                    self._initialize_pyaudio()
                elif self.source == 'udp':
                    self._cleanup_socket()
                    self._initialize_udp()
                elif self.source == 'tcp':
                    self._cleanup_socket()
                    self._initialize_tcp()
        except Exception as e:
            # was: logged and stopped, after which read() returned None and the ASR base restarted its run
            # into a wait for a START that had been sent already
            self.state, self.failure = 'failed', as_sentence(f"{self._what()} stopped: {e}")
            self.log.error(self.failure)
        finally:
            if self._stop_event.is_set() or self.state == 'failed':
                # this thread is the only one using the source now (stop() leaves it alone while it runs)
                self._cleanup_source()

    def _reconnect(self, stalled: float | None = None) -> bool:
        """The stream dropped (ffmpeg's output ended: its publisher went, and MediaMTX ended the readers of
        its path; or no byte came for `stalled` seconds), or an LSL outlet fell silent. It is opened again
        while the base gets no audio: the audio of the gap is missing and none is made up. False when it did
        not come back (read() raises from then on) or a stop came."""
        # a stall began when the last audio came
        gap = [time.time() - (stalled or 0.0), None]
        self.gaps.append(gap)
        self.state = 'reconnecting'
        why = f"stalled: no audio came for {stalled:g} s" if stalled else "ended"
        self.log.warning(f"{self._what()} {why}. Opening it again for up to {self.reconnect_wait:g} s; its "
                         f"audio until then is missing.")
        try:
            if self.source == 'stream':
                self._initialize_rtmp(wait=self.reconnect_wait, after_drop=True)
            else:
                self._initialize_lsl(wait=self.reconnect_wait, after_drop=True)
        except Exception as e:
            self.state, self.failure = 'failed', as_sentence(e)
            self.log.error(f"{self.failure} The base ends its run on this.")
            return False
        if self._stopping():
            return False
        gap[1] = time.time()
        self.state = 'live'
        # the stamps may come from elsewhere now: said again
        self._stamp_source_logged = False
        self.log.info(f"{self._what()} is back after {gap[1] - gap[0]:.1f} s; the audio of that gap is missing.")
        return True

    def _reset_socket_framing(self) -> None:
        """Forget the sniffed packet format and any partially received chunk."""
        self._detected_packet_format = None if self.packet_format == 'auto' else self.packet_format
        self._pending_bytes = bytearray()
        self._pending_frames: deque[AudioFrame] = deque()
        self._raw_anchor_time: float | None = None
        self._raw_frames_emitted = 0

    def _read_socket_chunk(self) -> AudioFrame | None:
        """Read the next chunk from a udp/tcp source.

        Two packet layouts are supported (see PACKET_FORMATS): the wearable header
        followed by one chunk of PCM, whose timestamp is the sender's clock, and a
        header-less PCM byte stream that is re-framed here to chunk_size frames.
        A socket timeout counts as a missed read rather than an error.
        """
        if self._pending_frames:
            return self._pending_frames.popleft()
        try:
            if self.source == 'udp':
                return self._read_udp_chunk()
            return self._read_tcp_chunk()
        except socket.timeout:
            return None

    def _read_udp_chunk(self) -> AudioFrame | None:
        while not self._pending_frames:
            data, _ = self.sock.recvfrom(SOCKET_RECV_SIZE)
            received = _now()
            if not data:
                return None
            if self._resolve_packet_format(data, received) == 'timestamped':
                return self._frame_from_timestamped_packet(data, received)
            self._push_raw_bytes(data, received)
        return self._pending_frames.popleft()

    def _read_tcp_chunk(self) -> AudioFrame | None:
        if self._detected_packet_format is None:
            # sniff the first header-sized piece of the stream
            head = self._recv_exact(SOCKET_HEADER_SIZE)
            received = _now()
            if head is None:
                return None
            if self._resolve_packet_format(head, received) == 'timestamped':
                payload = self._recv_exact(self._chunk_bytes)
                if payload is None:
                    return None
                return self._frame_from_timestamped_packet(head + payload, received)
            self._push_raw_bytes(head, received)
        if self._detected_packet_format == 'timestamped':
            packet = self._recv_exact(SOCKET_HEADER_SIZE + self._chunk_bytes)
            if packet is None:
                return None
            return self._frame_from_timestamped_packet(packet, _now())
        while not self._pending_frames:
            data = self.conn.recv(SOCKET_RECV_SIZE)
            if not data:
                return None  # the peer closed the connection
            self._push_raw_bytes(data, _now())
        return self._pending_frames.popleft()

    def _recv_exact(self, size: int) -> bytes | None:
        """Read exactly size bytes from the tcp connection, or None if the peer closed it."""
        buf = bytearray()
        while len(buf) < size:
            data = self.conn.recv(size - len(buf))
            if not data:
                return None
            buf += data
        return bytes(buf)

    def _resolve_packet_format(self, data: bytes, received: float) -> str:
        """Return the packet format, sniffing the first packet when set to 'auto'."""
        if self._detected_packet_format is None:
            header_time = parse_socket_header(data)
            self._detected_packet_format = 'timestamped' if header_time is not None else 'raw'
            logger.info(f"Detected '{self._detected_packet_format}' packets on {self.source} stream "
                        f"{self.host}:{self.port}")
        return self._detected_packet_format

    def _frame_from_timestamped_packet(self, packet: bytes, received: float) -> AudioFrame | None:
        """Build a frame from a header + PCM packet, stamped with the sender's clock."""
        timestamp = parse_socket_header(packet)
        if timestamp is None:
            logger.warning(f"Invalid packet header on {self.source} stream, using the receiver clock")
            timestamp = received
        payload = packet[SOCKET_HEADER_SIZE:]
        usable = len(payload) - len(payload) % self._frame_bytes
        if usable <= 0:
            return None
        return AudioFrame(data=np.frombuffer(payload[:usable], dtype=self.dtype), timestamp=timestamp,
                          metadata=dict(self._frame_metadata, timestamp_source='sender_packet_header'))

    def _push_raw_bytes(self, data: bytes, received: float) -> None:
        """Append header-less PCM bytes and cut them into chunk_size frames.

        Timestamps count samples from the arrival of the first packet, which keeps
        them evenly spaced; when packet loss or a sender restart pulls them more than
        RAW_TIMESTAMP_MAX_DRIFT away from the receiver clock they are re-anchored.
        """
        self._pending_bytes += data
        pending_frames = len(self._pending_bytes) // self._frame_bytes
        if self._raw_anchor_time is None:
            # the newest sample was captured just before this packet arrived
            self._raw_anchor_time = received - pending_frames / self.rate
        else:
            expected = self._raw_anchor_time + (self._raw_frames_emitted + pending_frames) / self.rate
            drift = received - expected
            if abs(drift) > RAW_TIMESTAMP_MAX_DRIFT:
                logger.warning(f"Raw {self.source} stream drifted {drift:+.3f}s from the receiver clock, "
                               "re-anchoring timestamps")
                self._raw_anchor_time += drift
        while len(self._pending_bytes) >= self._chunk_bytes:
            chunk = bytes(self._pending_bytes[:self._chunk_bytes])
            del self._pending_bytes[:self._chunk_bytes]
            timestamp = self._raw_anchor_time + self._raw_frames_emitted / self.rate
            self._raw_frames_emitted += self.chunk_size
            self._pending_frames.append(AudioFrame(data=np.frombuffer(chunk, dtype=self.dtype),
                                                   timestamp=timestamp, metadata=self._frame_metadata))

    def _read_chunk(self) -> AudioFrame | None:
        """Read a chunk of audio data continuously.

        UDP/TCP sources are handled by _read_socket_chunk, which accepts both the
        18-byte wearable header (packet counter + sender UTC time) in front of one
        PCM chunk and a plain PCM byte stream that is re-framed to chunk_size.

        For network streams (ffmpeg), we assume FFmpeg outputs raw PCM data with no extra metadata.
        """
        try:
            if self.source == 'pyaudio':
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=self.dtype)
                audio_data = audio_data.reshape(-1, self.channels)
                if self.channels > 1 and self.channel_select is not None:
                    if 0 <= self.channel_select < self.channels:
                        audio_data = audio_data[:, self.channel_select]
                    else:
                        logger.warning(f"Invalid channel index: {self.channel_select}, but only {self.channels} channels available. Using all channels.")
                timestamp = time.time()
            elif self.source in ['udp', 'tcp']:
                return self._read_socket_chunk()
            elif self.source == 'stream':
                # for ffmpeg-decoded streams, expected bytes is the raw audio data only; the chunk the
                # opening read comes first
                expected_bytes = self.chunk_size * self.channels * self.sample_width
                data, self._first_chunk = self._first_chunk, None
                if data is None:
                    data = self._pipe_read(self.ffmpeg_proc, expected_bytes, STALL_SECONDS)
                if data is None:
                    if self._stop_event.is_set():
                        return None  # stop() came while it waited
                    # no byte for STALL_SECONDS and no end either: ffmpeg would wait for ever. It is ended, and
                    # the stream opened again as after a drop
                    self._stalled_for = STALL_SECONDS
                    self._stream_ended = True
                    self._cleanup_rtmp()
                    return None
                if len(data) < expected_bytes:
                    self._stream_ended = True  # a pipe read comes back short only at its end: ffmpeg exited
                    return None
                audio_data = np.frombuffer(data, dtype=self.dtype)
                return AudioFrame(data=audio_data, timestamp=time.time(), metadata=self._connection_metadata)
            elif self.source == 'lsl':
                chunk, timestamps = self.lsl_inlet.pull_chunk(timeout=1.0, max_samples=self.chunk_size)
                if not chunk:
                    return None
                audio_data = np.array(chunk, dtype=self.dtype)
                timestamp = (timestamps[0] + self.lsl_offset) if timestamps else time.time()
                return AudioFrame(data=audio_data, timestamp=timestamp,
                                  metadata=dict(self._frame_metadata, timestamp_source='lsl'))
            else:
                return None

            if len(audio_data) == 0:
                return None

            return AudioFrame(
                data=audio_data,
                timestamp=timestamp,
                metadata=self._frame_metadata
            )
        except Exception as e:
            raise RuntimeError(f"Error reading chunk from {self.source}: {e}") from e

    @staticmethod
    def require_kwarg(kwargs, key, message):
        value = kwargs.get(key)
        if value is None:
            raise ValueError(message)
        return value
