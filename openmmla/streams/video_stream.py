import threading
import time
from typing import Optional, Union, List

import cv2

from openmmla.streams.resampling import resample_video, ResampleMethod
from openmmla.utils.logger import get_logger
from openmmla.utils.threads import RaisingThread
from .frame import VideoFrame
from .stream_buffer import RingBuffer
from .stream_receiver import StreamReceiver

logger = get_logger(__name__)


class VideoStream(StreamReceiver):
    """Video stream implementation for continuous video capture."""

    def __init__(self, source: Union[int, str], buffer_duration: float = 1.0, **kwargs):
        """Initialize video stream.
        
        Args:
            source: Stream source (camera index or video file path)
            buffer_duration: Duration of the ring buffer in seconds
            **kwargs: Additional configuration parameters
                format: Video format (default: 'MJPG')
                resolution: Tuple of (width, height) (default: (640, 480))
                fps: Frames per second (default: 30)
        """
        super().__init__(**kwargs)
        self.source = source

        # Stream configuration
        self.format = kwargs.get('format', 'MJPG')
        self.resolution = kwargs.get('resolution', (640, 480))
        self.fps = kwargs.get('fps', 30)

        # Frame metadata
        self._frame_metadata = {
            'resolution': self.resolution,
            'fps': self.fps,
            'format': self.format
        }

        # Calculate buffer size in frames
        buffer_frames = int(self.fps * buffer_duration)
        self.buffer = RingBuffer(buffer_frames)

        # Last read position tracking
        self._last_read_pos = -1

        # Threading control
        self._stop_event = threading.Event()
        self._receive_thread = None

        # OpenCV objects
        self.stream = None

        self.resample_method = kwargs.get('resample_method', ResampleMethod.VIDEO_AVERAGE)

    def start(self) -> None:
        """Start the video stream and begin capturing frames."""
        self.stop()
        self._initialize_stream()

        self._stop_event.clear()
        self._receive_thread = RaisingThread(target=self._receive_loop)
        self._receive_thread.daemon = True
        self._receive_thread.start()
        logger.info(f"Video stream started with source: {self.source}")

    def _initialize_stream(self) -> None:
        """Initialize video capture stream with configured parameters."""
        self.stream = cv2.VideoCapture(self.source)

        # Set video properties
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.format))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, self.fps)
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    def stop(self) -> None:
        """Stop the video stream and clean up resources."""
        self._stop_event.set()
        if self._receive_thread:
            self._receive_thread.join()
        self._cleanup_stream()

    def _cleanup_stream(self) -> None:
        """Clean up video capture resources."""
        if self.stream:
            self.stream.release()
            self.stream = None
        self._last_read_pos = -1

    def _receive_loop(self) -> None:
        """Continuously receive frames and store in buffer."""
        frame_count = 0
        start_time = time.time()

        while not self._stop_event.is_set():
            try:
                frame = self._read_frame()
                if frame:
                    self.buffer.push(frame)

                    # Calculate and log FPS every 30 frames
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        logger.debug(f"Video capture FPS: {fps:.2f}")
                        start_time = time.time()
                        frame_count = 0

            except Exception as e:
                logger.error(f"Fatal error in receive loop: {e}")
                self.stop()
                raise

    def _read_frame(self) -> Optional[VideoFrame]:
        """Read a single frame from the video stream."""
        try:
            grabbed, frame = self.stream.read()
            if not grabbed:
                logger.warning("Failed to grab frame")
                return None

            timestamp = time.time()

            return VideoFrame(
                data=frame,
                timestamp=timestamp,
                metadata=self._frame_metadata
            )

        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None

    def read(self, duration: Optional[float] = None, target_fps: Optional[float] = None,
             timeout: float = 5.0, latest: bool = False) -> Optional[Union[VideoFrame, List[VideoFrame]]]:
        """Read video frames with optional fps conversion.
        
        Args:
            duration: Duration to read in seconds. If None or 0, returns most recent frame
            target_fps: Optional target frame rate for resampling
            timeout: Maximum time to wait for frames in seconds
            latest: Whether to read from the most recent frame or continue from last position
            
        Returns:
            Single VideoFrame if duration is None/0, or list of VideoFrames if duration > 0.
        
        Notes:
            If buffer duration is set too short, the read operation may timeout since the
            last_read_pos might be always equal to the current tail.
        """
        if duration is None or duration == 0:
            frames_needed = 1
        else:
            frames_needed = int(duration * self.fps)

        total_frames = []
        start_time = time.time()

        if latest:
            self._last_read_pos = -1

        while len(total_frames) < frames_needed:
            current_tail = self.buffer.get_tail()

            # For first/latest read, start from most recent frame
            if self._last_read_pos == -1:
                self._last_read_pos = current_tail
                continue

            if current_tail == self._last_read_pos:
                if time.time() - start_time > timeout:
                    logger.warning("Timeout reached while waiting for frames.")
                    break
                time.sleep(0.01)
                continue

            remaining_frames = frames_needed - len(total_frames)
            available_frames = self.buffer.frames_available(self._last_read_pos)
            end_pos = (self._last_read_pos + remaining_frames) % self.buffer.size \
                if available_frames >= remaining_frames else current_tail

            new_frames = self.buffer.get(start_pos=self._last_read_pos, end_pos=end_pos)
            total_frames.extend(new_frames)
            self._last_read_pos = end_pos
            start_time = time.time()

        return self._process_frames(total_frames, target_fps)

    def _process_frames(self, frames: list, target_fps: Optional[float]) -> Optional[
                        Union[VideoFrame, List[VideoFrame]]]:
        """Process collected frames and apply fps conversion if needed.
        
        Args:
            frames: List of VideoFrames to process
            target_fps: Optional target frame rate
            
        Returns:
            Processed VideoFrame(s) or None if no frames available
        """
        if not frames:
            logger.warning("No frames collected within timeout period.")
            return None

        # If no target_fps specified, return original frames
        if not target_fps or target_fps == self.fps:
            return frames

        # Convert frames to numpy arrays for resampling
        frame_arrays = [frame.data for frame in frames]

        resampled_arrays = resample_video(
            frame_arrays,
            source_fps=self.fps,
            target_fps=target_fps,
            method=self.resample_method
        )

        # Convert back to VideoFrames
        resampled_frames = []
        time_step = 1.0 / target_fps
        base_timestamp = frames[0].timestamp

        for i, frame_data in enumerate(resampled_arrays):
            metadata = {
                'resolution': self.resolution,
                'fps': target_fps,
                'format': self.format
            }
            resampled_frames.append(VideoFrame(
                data=frame_data,
                timestamp=base_timestamp + (i * time_step),
                metadata=metadata
            ))

        return resampled_frames
