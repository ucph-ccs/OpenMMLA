import threading
from typing import List, TypeVar, Optional

from .frame import StreamFrame

T = TypeVar('T', bound=StreamFrame)


class RingBuffer:
    """Thread-safe circular buffer for storing stream frames."""

    def __init__(self, size: int):
        self.size = size
        self.buffer: List[T] = [None] * size
        self.head = 0
        self.tail = 0
        self.full = False
        self._lock = threading.Lock()

    def push(self, frame: T) -> None:
        """Thread-safe push operation."""
        with self._lock:
            if self.full:
                self.head = (self.head + 1) % self.size
            self.buffer[self.tail] = frame
            self.tail = (self.tail + 1) % self.size
            self.full = self.tail == self.head

    def get(self, n: Optional[int] = None, start_pos: Optional[int] = None,
            end_pos: Optional[int] = None) -> List[T]:
        """Thread-safe get operation.
        
        Args:
            n: Number of frames to get. If None, get all available frames.
            start_pos: Starting position to get frames from. If None, start from head.
            end_pos: Ending position to get frames to. If None, read until tail.
        """
        with self._lock:
            if not self.full and self.head == self.tail:
                return []

            # Use provided positions or defaults
            current_head = start_pos if start_pos is not None else self.head
            current_tail = end_pos if end_pos is not None else self.tail

            if current_head < current_tail:
                frames = self.buffer[current_head:current_tail]
            else:
                frames = self.buffer[current_head:] + self.buffer[:current_tail]

            if n is not None:
                frames = frames[-n:]
            return frames

    def get_tail(self) -> int:
        """Get current tail position."""
        with self._lock:
            return self.tail

    def frames_available(self, start_pos: int) -> int:
        """Get the number of available frames from start_pos to tail."""
        with self._lock:
            if start_pos <= self.tail:
                return self.tail - start_pos
            return self.size - start_pos + self.tail

    def __len__(self) -> int:
        with self._lock:
            if self.full:
                return self.size
            if self.tail >= self.head:
                return self.tail - self.head
            return self.size - (self.head - self.tail)
