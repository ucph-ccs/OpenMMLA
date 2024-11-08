# StreamReceiver is a base class for receiving data streams, with specified sampling rate, channels, etc.
# Data is received into a buffer, and the buffer can be read by the consumer.

from abc import ABC, abstractmethod
from typing import Any, Optional


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
        self.source = None  # stream type: 'pyaudio', 'socket', 'cv2'
        self.stream = None  # stream object from pyaudio, socket, etc.
        self.buffer = None  # ring buffer for data storage
        self.config = kwargs

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
