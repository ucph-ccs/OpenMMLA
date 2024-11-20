import datetime
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np
import pyaudio

from openmmla.utils.logger import get_logger
from openmmla.utils.resampling import resample_audio, ResampleMethod
from openmmla.utils.sockets import clear_socket_udp
from openmmla.utils.threads import RaisingThread
from .frame import AudioFrame
from .stream_buffer import RingBuffer
from .stream_receiver import StreamReceiver

logger = get_logger(__name__)


class AudioStream(StreamReceiver):
    """Audio stream implementation for continuous audio data capture."""

    def __init__(self, source: str, buffer_duration: float = 5.0, **kwargs):
        """Initialize audio stream.
        
        Args:
            source: Stream source type ('pyaudio', 'udp', or 'tcp')
            buffer_duration: Duration of the ring buffer in seconds
            **kwargs: Additional configuration parameters
                format: Audio format (default: pyaudio.paInt16)
                channels: Number of channels (default: 1)
                rate: Sample rate in Hz (default: 16000)
                chunk_size: Frames per buffer (default: 1024)
                resample_method: Method for resampling (default: AUDIO_LIBROSA)
                host: Socket host (for 'udp' or 'tcp' source)
                port: Socket port (for 'udp' or 'tcp' source)
        """
        super().__init__(**kwargs)
        self.source = source

        # Socket configuration
        self.host = kwargs.get('host', 'localhost')
        self.port = kwargs.get('port', 8000)
        self.sock: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None  # For TCP connection

        # Stream configuration
        self.format = kwargs.get('format', pyaudio.paInt16)
        self.channels = kwargs.get('channels', 1)
        self.rate = kwargs.get('rate', 16000)
        self.chunk_size = kwargs.get('chunk_size', 512)

        # Frame metadata
        self._frame_metadata = {
            'sample_rate': self.rate,
            'channels': self.channels,
            'format': 'int16'
        }

        # Calculate buffer size in frames
        buffer_frames = int(self.rate * buffer_duration / self.chunk_size)
        self.buffer = RingBuffer(buffer_frames)

        # Last read position tracking
        self._last_read_pos = -1

        # Threading control
        self._stop_event = threading.Event()
        self._receive_thread = None

        # PyAudio objects
        self.p = None
        self.stream = None

        self.resample_method = kwargs.get('resample_method', ResampleMethod.AUDIO_LIBROSA)

    def start(self) -> None:
        """Start the audio stream and begin capturing data."""
        self.stop()

        if self.source == 'pyaudio':
            self._initialize_pyaudio()
        elif self.source == 'udp':
            self._initialize_udp()
        elif self.source == 'tcp':
            self._initialize_tcp()
        else:
            raise ValueError(f"Unsupported source type: {self.source}")

        self._stop_event.clear()
        self._receive_thread = RaisingThread(target=self._receive_loop)
        self._receive_thread.daemon = True
        self._receive_thread.start()
        logger.info(f"Audio stream started with source: {self.source}")

    def _initialize_pyaudio(self) -> None:
        """Initialize PyAudio stream with configured parameters."""
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

    def _initialize_udp(self) -> None:
        """Initialize UDP socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        clear_socket_udp(self.sock)
        logger.info(f"UDP socket initialized on {self.host}:{self.port}")

    def _initialize_tcp(self) -> None:
        """Initialize TCP socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        self.conn, addr = self.sock.accept()
        logger.info(f"TCP connection accepted from {addr}")

    def stop(self) -> None:
        """Stop the audio stream and clean up resources."""
        self._stop_event.set()
        if self._receive_thread:
            self._receive_thread.join()

        if self.source == 'pyaudio':
            self._cleanup_pyaudio()
        elif self.source in ['udp', 'tcp']:
            self._cleanup_socket()

    def _cleanup_pyaudio(self) -> None:
        """Clean up PyAudio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        self._last_read_pos = 0

    def _cleanup_socket(self) -> None:
        """Clean up socket resources."""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.sock:
            self.sock.close()
            self.sock = None
        self._last_read_pos = 0

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
                raise

    def _read_chunk(self) -> Optional[AudioFrame]:
        """Read a chunk of audio data continuously.
        
        For UDP/TCP sources, the packet format is:
        Metadata (18 bytes):
        - 4 bytes (uint32): packet counter
        - 14 bytes (7 x uint16): timestamp (year, month, day, hour, minute, second, milliseconds)
        Audio data (1024 bytes):
        - 512 samples in int16 format
        """
        try:
            if self.source == 'pyaudio':
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                timestamp = time.time()
            elif self.source in ['udp', 'tcp']:
                # Read the entire packet (metadata + audio data)
                if self.source == 'udp':
                    data, _ = self.sock.recvfrom(self.chunk_size * 2 + 18)  # 1024 bytes audio + 18 bytes metadata
                else:  # TCP
                    data = self.conn.recv(self.chunk_size * 2 + 18)

                if not data or len(data) < 18:  # Check if we have at least the metadata
                    return None

                metadata_format = '>I7H'  # 4 bytes + (7 * 2) bytes = 18 bytes
                packet_counter, year, month, day, hour, minute, second, milliseconds = \
                    struct.unpack(metadata_format, data[:18])

                timestamp = datetime.datetime(
                    year, month, day, hour, minute, second,
                    milliseconds * 1000  # Convert to microseconds
                ).timestamp()

                audio_data = np.frombuffer(data[18:], dtype=np.int16)
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
            logger.error(f"Error reading chunk from {self.source}: {e}")
            return None

    def read(self, duration: float, target_rate: Optional[int] = None,
             timeout: float = 5.0, latest: bool = False) -> Optional[AudioFrame]:
        """Read audio data with optional resampling.
        
        Args:
            duration: Duration to read in seconds
            target_rate: Optional target sample rate for resampling
            timeout: Maximum time to wait for data in seconds
            latest: Whether to read from the most recent data or continue from last position
            
        Returns:
            AudioFrame containing the requested duration of audio data,
            or None if timeout is reached
        """
        frames_needed = int(duration * self.rate / self.chunk_size)
        total_frames = []
        start_time = time.time()

        if latest:
            self._last_read_pos = -1  # Reset to sentinel value

        while len(total_frames) < frames_needed:
            current_tail = self.buffer.get_tail()

            # For first/latest read, start from most recent data
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

    def _process_frames(self, frames: list, target_rate: Optional[int]) -> Optional[AudioFrame]:
        """Process collected frames and apply resampling if needed.
        
        Args:
            frames: List of AudioFrames to process
            target_rate: Optional target sample rate for resampling
            
        Returns:
            Processed AudioFrame or None if no frames available
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
