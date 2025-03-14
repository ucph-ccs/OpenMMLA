from dataclasses import dataclass, field
from time import time
from typing import Any, Optional, Dict, Tuple, Union, List

import numpy as np


@dataclass
class StreamFrame:
    """Base class for stream data frames."""
    data: Any
    timestamp: float = field(default_factory=time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AudioFrame(StreamFrame):
    """Audio frame with specific audio properties."""
    data: Union[np.ndarray, bytes, List[bytes], Tuple[bytes]]

    def __post_init__(self):
        """Ensure required metadata fields are present."""
        required_fields = {'sample_rate', 'channels', 'format'}
        missing = required_fields - set(self.metadata.keys())
        if missing:
            raise ValueError(f"Missing required metadata: {missing}")

    def to_bytes(self) -> bytes:
        """Convert audio data to bytes."""
        if isinstance(self.data, np.ndarray):
            return self.data.tobytes()
        return self.data

    @classmethod
    def from_bytes(cls, data: bytes, sample_rate: int, channels: int,
                   format: str = 'int16', timestamp: Optional[float] = None) -> 'AudioFrame':
        """Create AudioFrame from bytes."""
        np_data = np.frombuffer(data, dtype=format)
        metadata = {
            'sample_rate': sample_rate,
            'channels': channels,
            'format': format
        }
        return cls(
            data=np_data,
            timestamp=timestamp or time(),
            metadata=metadata
        )

    @property
    def sample_rate(self) -> int:
        return self.metadata['sample_rate']

    @property
    def channels(self) -> int:
        return self.metadata['channels']

    @property
    def format(self) -> str:
        return self.metadata['format']


@dataclass
class VideoFrame(StreamFrame):
    """Video frame with specific video properties."""

    def __post_init__(self):
        """Ensure required metadata fields are present."""
        required_fields = {'resolution', 'format', 'fps'}
        missing = required_fields - set(self.metadata.keys())
        if missing:
            raise ValueError(f"Missing required metadata: {missing}")

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.metadata['resolution']

    @property
    def format(self) -> str:
        return self.metadata['format']

    @property
    def fps(self) -> float:
        return self.metadata['fps']
