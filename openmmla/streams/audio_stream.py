import datetime
import socket
import struct
import subprocess
import threading
import time

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
from openmmla.utils.logger import get_logger
from openmmla.utils.sockets import clear_socket_udp
from openmmla.utils.threads import RaisingThread
from .frame import AudioFrame
from .stream_buffer import RingBuffer
from .stream_receiver import StreamReceiver

logger = get_logger(__name__)

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

    def __init__(self, source: str, **kwargs):
        """Initialize audio stream.

        Args:
            source (str): Stream source type ('pyaudio', 'udp', 'tcp', 'rtmp', 'lsl').

        Keyword Args:
            buffer_duration (float, optional): Duration of the ring buffer in seconds (default: 5.0)
            format (str, optional): Audio format (default: 'int16')
            channels (int, optional): Number of audio channels (default: 1)
            rate (int, optional): Sample rate in Hz (default: 16000)
            chunk_size (int, optional): Size of audio chunk to read in frames (default: 512)
            resample_method (ResampleMethod, optional): Method for resampling (default: AUDIO_LIBROSA)
            host (str, optional): Socket host (required for 'udp' or 'tcp' sources)
            port (int, optional): Socket port (required for 'udp' or 'tcp' sources)
            url (str, optional): RTMP URL (required for 'rtmp' source)
        """
        super().__init__(**kwargs)
        self.source = source

        # Stream configuration
        self.buffer_duration = kwargs.get('buffer_duration', 5.0)
        self.format = kwargs.get('format', 'int16')
        if self.format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {self.format}")
        self.sample_width = SUPPORTED_FORMATS[self.format]['sample_width']
        self.dtype = SUPPORTED_FORMATS[self.format]['dtype']

        self.channels = kwargs.get('channels', 1)
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
            self.sock: socket.socket | None = None
            self.conn: socket.socket | None = None  # For TCP connection

        # RTMP objects
        if self.source == 'rtmp':
            self.rtmp_url = self.require_kwarg(kwargs, 'url', "RTMP source requires a 'url' parameter")
            self.ffmpeg_proc: subprocess.Popen | None = None

        # LSL objects
        if self.source == 'lsl':
            self.lsl_name = self.require_kwarg(kwargs, 'lsl_name', "LSL source requires a 'lsl_name' parameter")
            self.lsl_inlet = None
            self.lsl_offset = None

        # Frame metadata
        self._frame_metadata = {
            'sample_rate': self.rate,
            'channels': self.channels,
            'format': self.format
        }

        # Calculate buffer size in frames
        buffer_frames = int(self.rate * self.buffer_duration / self.chunk_size)
        self.buffer = RingBuffer(buffer_frames)

        # Last read position tracking
        self._last_read_pos = -1

        # Threading control
        self._stop_event = threading.Event()
        self._receive_thread = None

    def start(self) -> None:
        """Start the audio stream and begin capturing data."""
        self.stop()

        if self.source == 'pyaudio':
            self._initialize_pyaudio()
        elif self.source == 'udp':
            self._initialize_udp()
        elif self.source == 'tcp':
            self._initialize_tcp()
        elif self.source == 'rtmp':
            self._initialize_rtmp()
        elif self.source == 'lsl':
            self._initialize_lsl()
        else:
            raise ValueError(f"Unsupported source type: {self.source}")

        self._stop_event.clear()
        self._receive_thread = RaisingThread(target=self._receive_loop)
        self._receive_thread.daemon = True
        self._receive_thread.start()

        if self.source == 'rtmp':
            time.sleep(3)

        logger.info(f"Audio stream started with source: {self.source}")

    def stop(self) -> None:
        """Stop the audio stream and clean up resources."""
        self._stop_event.set()  # Signal the thread to stop

        if self._receive_thread:
            try:
                if threading.current_thread() != self._receive_thread:
                    self._receive_thread.join(timeout=5)
            except Exception as e:
                logger.warning(f"During thread stopping, caught: {e}", exc_info=True)
            finally:
                self._receive_thread = None

        if self.source == 'pyaudio':
            self._cleanup_pyaudio()
        elif self.source in ['udp', 'tcp']:
            self._cleanup_socket()
        elif self.source == 'rtmp':
            self._cleanup_rtmp()
        elif self.source == 'lsl':
            self._cleanup_lsl()

        self._last_read_pos = -1

    def read(self, duration: float, target_rate: int | None = None, timeout: float = 5.0,
             latest: bool = False) -> AudioFrame | None:
        """Read audio data with optional resampling.

        Args:
            duration (float): Duration to read in seconds.
            target_rate (int, optional): Optional target sample rate for resampling.
            timeout (float): Maximum time to wait for data in seconds.
            latest (bool): Whether to read from the most recent data or continue from last position.

        Returns:
            AudioFrame: Containing the requested duration of audio data, or None if timeout is reached.
        """
        frames_needed = int(duration * self.rate / self.chunk_size)
        total_frames = []
        start_time = time.time()

        if latest:
            self._last_read_pos = -1

        while len(total_frames) < frames_needed:
            current_tail = self.buffer.get_tail()

            if self._last_read_pos == -1:
                self._last_read_pos = current_tail
                continue

            if current_tail == self._last_read_pos:
                if time.time() - start_time > timeout:
                    logger.warning("Timeout reached while waiting for frames.")
                    break
                time.sleep(1)
                continue

            remaining_frames = frames_needed - len(total_frames)
            available_frames = self.buffer.frames_available(self._last_read_pos)
            end_pos = (self._last_read_pos + remaining_frames) % self.buffer.size \
                if available_frames >= remaining_frames else current_tail

            new_frames = self.buffer.get(start_pos=self._last_read_pos, end_pos=end_pos)
            total_frames.extend(new_frames)
            self._last_read_pos = end_pos
            start_time = time.time()

        return self._process_frames(total_frames, target_rate)

    def _process_frames(self, frames: list[AudioFrame], target_rate: int | None) -> AudioFrame | None:
        """Process collected frames and apply resampling if needed.

        Args:
            frames: List of AudioFrames to process.
            target_rate: Optional target sample rate for resampling.

        Returns:
            AudioFrame: Processed AudioFrame or None if no frames available.
        """
        if not frames:
            logger.warning("No frames collected within timeout period.")
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

        return AudioFrame(timestamp=start_timestamp, data=audio_data, metadata=metadata)

    def _initialize_pyaudio(self) -> None:
        """Initialize PyAudio stream with configured parameters."""
        self.p = pyaudio.PyAudio()
        if self.format not in PA_FORMATS:
            raise ValueError(f"Unsupported audio format: {self.format} for {self.source}")
        stream_format = PA_FORMATS[self.format]
        self.stream = self.p.open(
            format=stream_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size
        )

    def _initialize_udp(self) -> None:
        """Initialize UDP socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        clear_socket_udp(self.sock)
        self.sock.settimeout(3)
        logger.info(f"UDP socket initialized on {self.host}:{self.port}")

    def _initialize_tcp(self) -> None:
        """Initialize TCP socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        self.conn, addr = self.sock.accept()
        self.conn.settimeout(3)
        logger.info(f"TCP connection accepted from {addr}")

    def _initialize_rtmp(self) -> None:
        """Initialize RTMP stream using FFmpeg.

        Spawns an FFmpeg process that connects to the provided RTMP URL and outputs
        raw PCM audio on stdout.
        """
        # Determine the correct format and codec based on self.format.
        fmt = RTMP_FORMATS[self.format]
        codec = RTMP_CODEC[self.format]

        command = [
            'ffmpeg',
            '-i', self.rtmp_url,
            '-f', fmt,
            '-acodec', codec,
            '-ar', str(self.rate),
            '-ac', str(self.channels),
            '-'  # Output to stdout
        ]
        try:
            self.ffmpeg_proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            if self.ffmpeg_proc.stdout is None:
                raise RuntimeError("Failed to capture stdout from ffmpeg process")
            logger.info(f"RTMP stream initialized from {self.rtmp_url}")
        except Exception as e:
            raise RuntimeError(f"Error initializing RTMP stream: {e}") from e

    def _initialize_lsl(self):
        """Initialize lab streaming layer stream."""
        if resolve_byprop is None or StreamInlet is None or local_clock is None:
            raise ImportError(
                "pylsl package is not installed. Please install it with 'pip install pylsl' to use LSL features."
            )
        
        streams = resolve_byprop('name', self.lsl_name)
        if not streams:
            raise RuntimeError(f"No LSL stream found with name: {self.lsl_name}")
            
        self.lsl_inlet = StreamInlet(streams[0])
        self.lsl_offset = time.time() - local_clock()
        logger.info(f"Subscribe LSL stream: {self.lsl_name}")

    def _cleanup_pyaudio(self) -> None:
        """Clean up PyAudio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def _cleanup_socket(self) -> None:
        """Clean up socket resources."""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.sock:
            self.sock.close()
            self.sock = None

    def _cleanup_rtmp(self) -> None:
        """Clean up RTMP (FFmpeg) process."""
        if self.ffmpeg_proc:
            self.ffmpeg_proc.terminate()
            self.ffmpeg_proc = None
            logger.info("RTMP stream process terminated.")

    def _cleanup_lsl(self):
        """Clean up lab streaming layer resources."""
        if self.lsl_inlet:
            self.lsl_inlet.close_stream()
        self.lsl_inlet = None
        self.lsl_offset = None

    def _receive_loop(self) -> None:
        """Continuously receive data and store in buffer."""
        while not self._stop_event.is_set():
            try:
                frame = self._read_chunk()
                if frame:
                    self.buffer.push(frame)
            except Exception as e:
                logger.error(f"Fatal error in receive loop: {e}")
                self.stop()

    def _read_chunk(self) -> AudioFrame | None:
        """Read a chunk of audio data continuously.

        For UDP/TCP sources, the packet format is:
          - Metadata (18 bytes):
              - 4 bytes (uint32): packet counter
              - 14 bytes (7 x uint16): timestamp (year, month, day, hour, minute, second, milliseconds)
          - Audio data: (chunk_size frames × channels × sample_width bytes)

        For RTMP, we assume FFmpeg outputs raw PCM data with no extra metadata.
        """
        try:
            if self.source == 'pyaudio':
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=self.dtype)
                timestamp = time.time()
            elif self.source in ['udp', 'tcp']:
                # For UDP/TCP, include 18 bytes of metadata.
                expected_bytes = self.chunk_size * self.channels * self.sample_width + 18
                if self.source == 'udp':
                    data, _ = self.sock.recvfrom(expected_bytes)
                else:
                    data = self.conn.recv(expected_bytes)

                if not data or len(data) < 18:
                    return None

                metadata_format = '>I7H'
                packet_counter, year, month, day, hour, minute, second, milliseconds = \
                    struct.unpack(metadata_format, data[:18])

                timestamp = datetime.datetime(
                    year, month, day, hour, minute, second,
                    milliseconds * 1000
                ).timestamp()

                audio_data = np.frombuffer(data[18:], dtype=self.dtype)
            elif self.source == 'rtmp':
                # For RTMP, expected bytes is the raw audio data only.
                expected_bytes = self.chunk_size * self.channels * self.sample_width
                data = self.ffmpeg_proc.stdout.read(expected_bytes)
                if not data or len(data) < expected_bytes:
                    return None
                audio_data = np.frombuffer(data, dtype=self.dtype)
                timestamp = time.time()
            elif self.source == 'lsl':
                chunk, timestamps = self.lsl_inlet.pull_chunk(timeout=1.0, max_samples=self.chunk_size)
                if not chunk:
                    return None
                audio_data = np.array(chunk, dtype=self.dtype)
                timestamp = (timestamps[0] + self.lsl_offset) if timestamps else time.time()
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
