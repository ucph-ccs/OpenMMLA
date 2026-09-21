"""
StreamReceiver is a base class for receiving data streams, with specified sampling rate, channels, etc.
Data is received into a buffer, and the buffer can be read by the consumer.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# stamps that already carry the capture side's clock: a measured receive-side delay does not apply
CAPTURE_SIDE_STAMPS = ('rtmp_stream_start_pts', 'rtmp_absolute_pts', 'lsl', 'sender_packet_header')


class StreamReceiver(ABC):
    """Base class for receiving data streams.
    
    Handles continuous data streams with specified parameters (sampling rate, channels, etc.).
    Data is received into a buffer and can be read by the consumer.
    """

    def __init__(self, **kwargs):
        """Initialize stream receiver.
        
        Args:
            **kwargs: Configuration parameters for the stream
        """
        self.source = None  # stream type: 'pyaudio', 'socket', 'cv2', 'rtmp', 'lsl', etc.
        self.stream = None  # stream object from pyaudio, cv2, etc.
        self.buffer = None  # ring buffer for data storage
        self.config = kwargs
        # a measured constant delay of this stream, in seconds, added to the stamp of every live
        # frame: negative moves the stamps back from when a frame arrived to when it was captured
        raw_offset = kwargs.get('timestamp_offset')
        try:
            self.timestamp_offset = float(raw_offset or 0.0)
        except (TypeError, ValueError):
            logger.warning(f"timestamp_offset {raw_offset!r} is not a number of seconds: no offset applied")
            self.timestamp_offset = 0.0
        if self.timestamp_offset:
            logger.info(f"Stream stamps are moved by timestamp_offset {self.timestamp_offset:+.3f} s")
        self._stamp_source_logged = False

    def _stamped(self, frame):
        """the frame with this stream's timestamp_offset applied (None stays None). A frame whose
        stamp already comes from the capture side (timestamp_source in CAPTURE_SIDE_STAMPS) is
        left alone: the delay that was measured is not in it."""
        if frame is None:
            return None
        source = (frame.metadata or {}).get('timestamp_source') if isinstance(frame.metadata, dict) else None
        if not self._stamp_source_logged and source:
            logger.info(f"Stream stamps come from {source}")
            self._stamp_source_logged = True
        if self.timestamp_offset and source not in CAPTURE_SIDE_STAMPS:
            frame.timestamp = frame.timestamp + self.timestamp_offset
            # a fresh dict: audio frames share one metadata dict across frames
            frame.metadata = dict(frame.metadata or {}, timestamp_offset=self.timestamp_offset)
        return frame

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    @abstractmethod
    def start(self) -> None:
        """Start receiving data stream.
        
        Instantiates the stream object and begins receiving data
        at specified sampling rate, channels, etc.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop receiving data stream and clean up resources."""
        pass

    @abstractmethod
    def read(self, *args, **kwargs) -> Any:
        """Read data from the stream buffer.
        
        Blocks if buffer is empty until data is available.
        For read durations greater than buffer size, performs multiple reads.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            Data read from the stream buffer
        """
        pass
