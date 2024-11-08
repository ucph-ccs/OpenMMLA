import threading
import time
from typing import Optional, Dict, Any

import numpy as np
import pyaudio

from openmmla.utils.logger import get_logger
from openmmla.utils.resampling import resample_audio, ResampleMethod
from .frame import AudioFrame
from .stream_buffer import RingBuffer
from .stream_receiver import StreamReceiver

logger = get_logger(__name__)


class AudioStream(StreamReceiver):
    """Audio stream implementation for continuous audio data capture."""

    def __init__(self, source: str, buffer_duration: float = 5.0, **kwargs):
        """Initialize audio stream.
        
        Args:
            source: Stream source type (e.g., 'pyaudio')
            buffer_duration: Duration of the ring buffer in seconds
            **kwargs: Additional audio configuration parameters
                format: Audio format (default: pyaudio.paInt16)
                channels: Number of channels (default: 1)
                rate: Sample rate in Hz (default: 16000)
                chunk_size: Frames per buffer (default: 1024)
                resample_method: Method for resampling (default: AUDIO_LIBROSA)
        """
        super().__init__(**kwargs)
        self.source = source

        # Audio configuration
        self.format = kwargs.get('format', pyaudio.paInt16)
        self.channels = kwargs.get('channels', 1)
        self.rate = kwargs.get('rate', 16000)
        self.chunk_size = kwargs.get('chunk_size', 1024)

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
        self._last_read_pos = 0

        # Threading control
        self.running = False
        self._receive_thread = None

        # PyAudio objects
        self.p = None
        self.stream = None

        self.resample_method = kwargs.get('resample_method', ResampleMethod.AUDIO_LIBROSA)

    def start(self) -> None:
        """Start the audio stream and begin capturing data."""
        if self.running:
            return

        if self.source == 'pyaudio':
            self._initialize_pyaudio()
        else:
            raise ValueError(f"Unsupported source type: {self.source}")

        self.running = True
        self._receive_thread = threading.Thread(target=self._receive_loop)
        self._receive_thread.daemon = True
        self._receive_thread.start()

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

    def stop(self) -> None:
        """Stop the audio stream and clean up resources."""
        self.running = False
        if self._receive_thread:
            self._receive_thread.join()

        if self.source == 'pyaudio':
            self._cleanup_pyaudio()

    def _cleanup_pyaudio(self) -> None:
        """Clean up PyAudio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def _receive_loop(self) -> None:
        """Continuously receive data and store in buffer."""
        while self.running:
            try:
                frame = self._read_chunk()
                if frame:
                    self.buffer.push(frame)
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break

    def _read_chunk(self) -> Optional[AudioFrame]:
        """Read a chunk of audio data.
        
        Returns:
            AudioFrame containing the chunk data and metadata, or None if read fails
        """
        if self.source == 'pyaudio':
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            return AudioFrame(
                data=np.frombuffer(data, dtype=np.int16),
                metadata=self._frame_metadata
            )
        return None

    def read(self, duration: float, target_rate: Optional[int] = None, 
            timeout: float = 5.0) -> Optional[AudioFrame]:
        """Read audio data with optional resampling.
        
        Args:
            duration: Duration to read in seconds
            target_rate: Optional target sample rate for resampling
            timeout: Maximum time to wait for data in seconds
            
        Returns:
            AudioFrame containing the requested duration of audio data,
            or None if timeout is reached
        """
        frames_needed = int(duration * self.rate / self.chunk_size)
        total_frames = []

        start_time = time.time()
        while len(total_frames) < frames_needed:
            current_tail = self.buffer.get_tail()

            if current_tail == self._last_read_pos:
                if time.time() - start_time > timeout:
                    logger.warning("Timeout reached while waiting for frames.")
                    break
                time.sleep(0.1)
                continue

            remaining_frames = frames_needed - len(total_frames)
            available_frames = self.buffer.frames_available(self._last_read_pos)

            if available_frames >= remaining_frames:
                end_pos = (self._last_read_pos + remaining_frames) % self.buffer.size
                new_frames = self.buffer.get(start_pos=self._last_read_pos, end_pos=end_pos)
                total_frames.extend(new_frames)
                self._last_read_pos = end_pos
                break
            else:
                new_frames = self.buffer.get(start_pos=self._last_read_pos, end_pos=current_tail)
                total_frames.extend(new_frames)
                self._last_read_pos = current_tail
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
            'format': frames[0].format
        }

        return AudioFrame(data=audio_data, metadata=metadata)
