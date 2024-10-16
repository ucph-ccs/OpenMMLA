# StreamReceiver is a base class for receiving data streams, with specified sampling rate, channels, etc.
# Data is received into a buffer, and the buffer can be read by the consumer.

from abc import ABC, abstractmethod
class StreamReceiver(ABC):
    def __init__(self, source, **kwargs):
        self.source = source    # e.g. 'pyaudio', 'socket', 'cv2'
        self.config = kwargs
        self.stream = None
        self.buffer = None

    @abstractmethod
    def start(self):
        """Instantiate the stream object and start to receive data stream at specified sampling rate, channels, etc."""
        pass 

    @abstractmethod
    def stop(self):
        """Stop to receive data stream."""
        pass

    @abstractmethod
    def read(self):
        """Read data from the stream buffer.
        
        If the buffer is empty, the method should block until data is available.
        If the read duration is greater than the buffer size, then we need to read buffer size for several times.
        """
        pass

    @abstractmethod
    def write(self, data):
        """Write data to the stream buffer ."""
        pass
