import threading
import time

import cv2
import numpy as np

try:
    from pylsl import local_clock, StreamInlet, resolve_byprop
except ImportError:
    local_clock = None
    StreamInlet = None
    resolve_byprop = None

from openmmla.streams.resampling import resample_video, ResampleMethod
from openmmla.utils.logger import get_logger
from openmmla.utils.threads import RaisingThread
from .frame import VideoFrame
from .stream_buffer import RingBuffer
from .stream_receiver import StreamReceiver

logger = get_logger(__name__)


class VideoStream(StreamReceiver):
    """Video stream implementation for continuous video capture."""

    def __init__(self, source: str, **kwargs):
        """Initialize video stream.

        Args:
            source (str): Stream source type ('opencv' or 'rtmp')

        Keyword Args:
            buffer_duration (float, optional): Duration of the ring buffer in seconds (default: 0.08)
            format(str, optional): Video format, 'MJPG', 'JPEG', 'raw' (default: 'MJPG')
            resolution(tuple, optional): Tuple of (width, height) (default: (1920, 1080))
            fps(int, optional): Frames per second (default: 30)
            resample_method(ResampleMethod, optional): Resampling method for fps conversion (default: ResampleMethod.VIDEO_AVERAGE)
            camera_index(int, optional): Camera index for 'opencv' source (default: 0)
            rtmp_url(str, optional): RTMP URL for 'rtmp' source
            max_reconnect_attempts (int, optional): Maximum number of reconnection attempts for RTMP (default: 3)
            reconnect_delay (float, optional): Delay between reconnection attempts in seconds (default: 2.0)
        """
        super().__init__(**kwargs)
        self.source = source

        # Stream configuration
        self.buffer_duration = kwargs.get('buffer_duration', 0.08)
        self.resolution = kwargs.get('resolution', (1920, 1080))
        self.fps = kwargs.get('fps', 30)
        self.resample_method = kwargs.get('resample_method', ResampleMethod.VIDEO_AVERAGE)

        # Source-specific configuration
        if self.source == 'opencv':
            self.format = kwargs.get('format', 'MJPG')
            self.camera_index = kwargs.get('camera_index', 0)
            self.stream = None
        elif self.source == 'rtmp':
            self.format = kwargs.get('format', 'MJPG')
            self.rtmp_url = self.require_kwarg(kwargs, 'rtmp_url', "RTMP source requires a 'rtmp_url' parameter")
            self.stream = None
            self.max_reconnect_attempts = kwargs.get('max_reconnect_attempts', 3)
            self.reconnect_delay = kwargs.get('reconnect_delay', 2.0)
            self.consecutive_failures = 0
            self.reconnect_threshold = 10  # Number of consecutive failures before attempting reconnection
        elif self.source == 'lsl':
            self.format = kwargs.get('format', 'raw')
            self.lsl_name = self.require_kwarg(kwargs, 'lsl_name', "LSL source requires a 'lsl_name'")
            self.lsl_offset = None
            self.lsl_inlet: StreamInlet | None = None
        else:
            raise ValueError(f"Unsupported source type: {self.source}")

        # Frame metadata
        self._frame_metadata = {
            'resolution': self.resolution,
            'fps': self.fps,
            'format': self.format
        }

        # Buffer
        buffer_frames = int(self.fps * self.buffer_duration)
        self.buffer = RingBuffer(buffer_frames)
        self._last_read_pos = -1

        # Threading control
        self._stop_event = threading.Event()
        self._receive_thread = None

    def start(self) -> None:
        """Start the video stream and begin capturing frames."""
        self.stop()
        if self.source in ['opencv', 'rtmp']:
            self._initialize_opencv()
        elif self.source == 'lsl':
            self._initialize_lsl()
        else:
            raise ValueError(f"Unsupported source type: {self.source}")

        self._stop_event.clear()
        self._receive_thread = RaisingThread(target=self._receive_loop)
        self._receive_thread.daemon = True
        self._receive_thread.start()
        logger.info(f"Video stream started with source: {self.source}")

    def stop(self) -> None:
        """Stop the video stream and clean up resources."""
        self._stop_event.set()
        if self._receive_thread:
            try:
                if threading.current_thread() != self._receive_thread:
                    self._receive_thread.join(timeout=5)
            except Exception as e:
                logger.warning(f"During thread stopping, caught: {e}", exc_info=True)
            finally:
                self._receive_thread = None

        if self.source in ['opencv', 'rtmp']:
            self._cleanup_opencv()
        elif self.source == 'lsl':
            self._cleanup_lsl()

        self._last_read_pos = -1

    def _initialize_opencv(self) -> None:
        """Initialize OpenCV video capture with configured parameters."""
        self.stream = cv2.VideoCapture(self.camera_index if self.source == 'opencv' else self.rtmp_url)

        # Set video properties
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.format))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, self.fps)
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # For RTMP, verify the connection was established
        if self.source == 'rtmp':
            if not self.stream.isOpened():
                logger.warning(f"Failed to open RTMP stream at {self.rtmp_url}")
            else:
                logger.info(f"Successfully connected to RTMP stream at {self.rtmp_url}")
                self.consecutive_failures = 0

    def _initialize_lsl(self) -> None:
        """Initialize LSL stream connection."""
        if resolve_byprop is None or StreamInlet is None or local_clock is None:
            raise ImportError(
                "pylsl package is not installed. Please install it with 'pip install pylsl' to use LSL features."
            )
            
        streams = resolve_byprop('name', self.lsl_name)
        if not streams:
            raise RuntimeError(f"LSL stream '{self.lsl_name}' not found")
        
        self.lsl_inlet = StreamInlet(streams[0])
        self.lsl_offset = time.time() - local_clock()
        logger.info(f"Subscribed to LSL video stream: {self.lsl_name}")

    def _cleanup_opencv(self) -> None:
        """Clean up OpenCV stream resources."""
        if self.stream:
            self.stream.release()
            self.stream = None

    def _cleanup_lsl(self) -> None:
        """Clean up LSL stream resources."""
        if self.lsl_inlet:
            self.lsl_inlet.close_stream()
        self.lsl_inlet = None
        self.lsl_offset = None

    def _reconnect_rtmp(self) -> bool:
        """Attempt to reconnect to the RTMP stream.
        
        Returns:
            bool: True if reconnection was successful, False otherwise
        """
        if self.source != 'rtmp':
            return False
            
        logger.info(f"Attempting to reconnect to RTMP stream at {self.rtmp_url}")
        
        # Clean up existing connection
        self._cleanup_opencv()
        
        # Wait before reconnecting
        time.sleep(self.reconnect_delay)
        
        # Try to reconnect
        try:
            self._initialize_opencv()
            # Verify connection
            if self.stream and self.stream.isOpened():
                logger.info(f"Successfully reconnected to RTMP stream at {self.rtmp_url}")
                return True
            else:
                logger.warning(f"Failed to reconnect to RTMP stream at {self.rtmp_url}")
                return False
        except Exception as e:
            logger.error(f"Error during RTMP reconnection: {e}")
            return False

    def _receive_loop(self) -> None:
        """Continuously receive frames and store in buffer."""
        frame_count = 0
        start_time = time.time()
        logger.info("Starting video stream receive loop.")
        
        reconnect_attempts = 0

        while not self._stop_event.is_set():
            try:
                frame = self._read_frame()
                if frame:
                    self.buffer.push(frame)
                    frame_count += 1
                    self.consecutive_failures = 0  # Reset failure counter on success
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        logger.debug(f"Video capture FPS: {fps:.2f}")
                        start_time = time.time()
                        frame_count = 0
                elif self.source == 'rtmp':
                    self.consecutive_failures += 1
                    
                    if self.consecutive_failures >= self.reconnect_threshold:
                        logger.warning(f"Detected {self.consecutive_failures} consecutive frame grab failures, attempting to reconnect")
                        if reconnect_attempts < self.max_reconnect_attempts:
                            if self._reconnect_rtmp():
                                reconnect_attempts = 0  # Reset on successful reconnection
                                self.consecutive_failures = 0
                            else:
                                reconnect_attempts += 1
                        else:
                            logger.error(f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached. Giving up.")
                            self.stop()
                            break
            except Exception as e:
                logger.error(f"Fatal error in receive loop: {e}")
                self.stop()
                raise

    def _read_frame(self) -> VideoFrame | None:
        """Read a single frame from the video stream."""
        try:
            if self.source in ['opencv', 'rtmp']:
                grabbed, frame = self.stream.read()
                timestamp = time.time()
                if not grabbed:
                    logger.warning("Failed to grab frame")
                    return None
            elif self.source == 'lsl':
                sample, timestamp = self.lsl_inlet.pull_sample(timeout=1.0)
                if not sample:
                    return None

                if self.format == 'raw':    # raw RGB data
                    data_array = np.array(sample, dtype=np.float32)
                    expected_size = self.resolution[0] * self.resolution[1] * 3  # width * height * RGB
                    if len(data_array) != expected_size:
                        logger.warning(
                            f"Received LSL frame with unexpected size: {len(data_array)} (expected {expected_size})")
                        return None
                    frame = data_array.reshape((self.resolution[1], self.resolution[0], 3)).astype(np.uint8)
                elif self.format == 'JPEG':  # JPEG compressed data
                    data_array = np.array(sample, dtype=np.float32).astype(np.uint8)
                    try:
                        frame = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
                        if frame is None:
                            logger.warning("Failed to decode JPEG data")
                            return None
                    except Exception as e:
                        logger.error(f"Error decoding JPEG data: {e}", exc_info=True)
                        return None
                else:
                    raise ValueError(f"Unsupported format for LSL source: {self.format}")

                timestamp = (timestamp + self.lsl_offset) if timestamp else time.time()
            else:
                raise ValueError(f"Unsupported source type: {self.source}")

            return VideoFrame(
                data=frame,
                timestamp=timestamp,
                metadata=self._frame_metadata
            )

        except Exception as e:
            logger.error(f"Error reading frame: {e}", exc_info=True)
            return None

    def read(self, duration: float = None, target_fps: float = None, timeout: float = 10.0,
             latest: bool = False) -> VideoFrame | list[VideoFrame] | None:
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

    def _process_frames(self, frames: list, target_fps: float | None) -> VideoFrame | list[VideoFrame] | None:
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

    @staticmethod
    def require_kwarg(kwargs, key, message):
        value = kwargs.get(key)
        if value is None:
            raise ValueError(message)
        return value
