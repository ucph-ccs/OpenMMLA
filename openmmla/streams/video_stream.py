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

# logger = get_logger(__name__, console_level=logging.DEBUG)
logger = get_logger(__name__)


class VideoStream(StreamReceiver):
    """Video stream implementation for continuous video capture."""

    def __init__(self, source: str, **kwargs):
        """Initialize video stream.

        Args:
            source (str): Stream source type ('opencv', 'rtmp', 'lsl', or 'file')

        Keyword Args:
            buffer_duration (float, optional): Duration of the ring buffer in seconds (default: 0.08)
            format(str, optional): Video format, 'MJPG', 'JPEG', 'H264', 'raw' (default: 'MJPG')
            resolution(tuple, optional): Tuple of (width, height) (default: (1920, 1080))
            fps(int, optional): Frames per second (default: 30)
            resample_method(ResampleMethod, optional): Resampling method for fps conversion (default: ResampleMethod.VIDEO_AVERAGE)
            camera_index(int, optional): Camera index for 'opencv' source (default: 0)
            rtmp_url(str, optional): RTMP URL for 'rtmp' source
            file_path(str, optional): Path to the video file (required for 'file' source)
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
            self.format = kwargs.get('format', 'H264')
            self.rtmp_url = self.require_kwarg(kwargs, 'rtmp_url', "RTMP source requires a 'rtmp_url' parameter")
            self.stream = None
        elif self.source == 'lsl':
            self.format = kwargs.get('format', 'raw')
            self.lsl_name = self.require_kwarg(kwargs, 'lsl_name', "LSL source requires a 'lsl_name'")
            self.lsl_offset = None
            self.lsl_inlet: StreamInlet | None = None
        elif self.source == 'file':
            self.format = kwargs.get('format', 'H264')
            self.file_path = self.require_kwarg(kwargs, 'file_path', "File source requires a 'file_path' parameter")
            self.video_capture = None
            self.total_frames = None
            self.file_fps = None
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

        # PTS-to-wallclock calibration for RTMP sources
        self._pts_offset: float | None = None

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
        elif self.source == 'file':
            self._initialize_file()
            # 'file' source doesn't need a receive thread - data is read on demand
            logger.info(f"Video stream started with source: {self.source}")
            return
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
        elif self.source == 'file':
            self._cleanup_file()

        self._last_read_pos = -1

    def _initialize_opencv(self, max_retries: int = 3) -> None:
        """Initialize OpenCV video capture with configured parameters."""
        self._pts_offset = None
        video_seed = self.camera_index if self.source == 'opencv' else self.rtmp_url
        self.stream = cv2.VideoCapture(video_seed)

        if self.source == 'opencv':
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.format))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.stream.set(cv2.CAP_PROP_FPS, self.fps)
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        if not self.stream.isOpened():
            if max_retries > 0:
                logger.warning("Failed to initialize OpenCV video capture, retrying...")
                self._cleanup_opencv()
                self._initialize_opencv(max_retries=max_retries - 1)
            else:
                raise RuntimeError("Failed to initialize OpenCV video capture")
        else:
            logger.info(f"Successfully initialized {self.source} video stream with seed: {video_seed}")

    def _initialize_lsl(self, max_retries: int = 3) -> None:
        """Initialize LSL stream connection."""
        if resolve_byprop is None or StreamInlet is None or local_clock is None:
            raise ImportError(
                "pylsl package is not installed. Please install it with 'pip install pylsl' to use LSL features."
            )

        streams = resolve_byprop('name', self.lsl_name)
        if not streams:
            if max_retries > 0:
                logger.warning("Failed to initialize LSL video stream, retrying...")
                self._cleanup_lsl()
                self._initialize_lsl(max_retries=max_retries - 1)
            else:
                raise RuntimeError(f"LSL stream '{self.lsl_name}' not found")
        else:
            self.lsl_inlet = StreamInlet(streams[0])
            self.lsl_offset = time.time() - local_clock()
            logger.info(f"Successfully initialized LSL video stream with inlet: {self.lsl_inlet}")

    def _initialize_file(self) -> None:
        """Initialize file source for video."""
        try:
            self.video_capture = cv2.VideoCapture(self.file_path)
            if not self.video_capture.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.file_path}")
            
            # Get actual video properties from file
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.file_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            file_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            file_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            file_resolution = (file_width, file_height)
            
            # Check if kwargs resolution matches file resolution
            if self.resolution != file_resolution:
                logger.warning(f"Requested resolution {self.resolution} doesn't match file resolution {file_resolution}. "
                              f"Using file resolution: {file_resolution}")
                self.resolution = file_resolution
            
            # Check if kwargs fps matches file fps
            if abs(self.fps - self.file_fps) > 0.1:  # Allow small floating point differences
                logger.warning(f"Requested FPS {self.fps} doesn't match file FPS {self.file_fps}. "
                              f"Using file FPS: {self.file_fps}")
                self.fps = self.file_fps
            
            # Update frame metadata with actual file properties
            self._frame_metadata = {
                'resolution': self.resolution,
                'fps': self.fps,
                'format': self.format
            }
            
            logger.info(f"Successfully loaded video file: {self.file_path}")
            logger.info(f"File duration: {self.total_frames / self.file_fps:.2f} seconds, "
                       f"FPS: {self.file_fps}, "
                       f"Resolution: {file_width}x{file_height}, "
                       f"Total frames: {self.total_frames}")
        except Exception as e:
            raise RuntimeError(f"Failed to load video file {self.file_path}: {e}") from e

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

    def _cleanup_file(self) -> None:
        """Clean up file resources."""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.total_frames = None
        self.file_fps = None

    def _receive_loop(self) -> None:
        """Continuously receive frames and store in buffer."""
        frame_count = 0
        failure_count = 0
        start_time = time.time()
        logger.info("Starting video stream receive loop.")

        while not self._stop_event.is_set():
            try:
                frame = self._read_frame()
                if frame:
                    self.buffer.push(frame)
                    frame_count += 1
                    failure_count = 0
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        logger.debug(f"Video capture FPS: {fps:.2f}")
                        start_time = time.time()
                        frame_count = 0
                else:
                    failure_count += 1
                    if failure_count >= 10:
                        logger.error("10 consecutive frame read failures. Reinitializing stream.")
                        if self.source in ['opencv', 'rtmp']:
                            self._cleanup_opencv()
                            self._initialize_opencv()
                        elif self.source == 'lsl':
                            self._cleanup_lsl()
                            self._initialize_lsl()
                        failure_count = 0
            except Exception as e:
                logger.error(f"Fatal error in receive loop: {e}")
                self.stop()
                raise

    def _read_frame(self) -> VideoFrame | None:
        """Read a single frame from the video stream."""
        try:
            if self.source in ['opencv', 'rtmp']:
                grabbed, frame = self.stream.read()
                if not grabbed:
                    logger.warning("Failed to grab frame")
                    return None
                if self.source == 'rtmp':
                    pts_ms = self.stream.get(cv2.CAP_PROP_POS_MSEC)
                    if pts_ms > 0:
                        if self._pts_offset is None:
                            self._pts_offset = time.time() - pts_ms / 1000.0
                            logger.info(f"RTMP PTS calibrated: offset={self._pts_offset:.3f}s")
                        timestamp = pts_ms / 1000.0 + self._pts_offset
                    else:
                        timestamp = time.time()
                else:
                    timestamp = time.time()
            elif self.source == 'lsl':
                sample, timestamp = self.lsl_inlet.pull_sample(timeout=1.0)
                if not sample:
                    return None

                if self.format == 'raw':  # raw RGB data
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
             latest: bool = False, start_time: float = 0.0) -> VideoFrame | list[VideoFrame] | None:
        """Read video frames with optional fps conversion.

        Args:
            duration: Duration to read in seconds. If None or 0, returns most recent frame
            target_fps: Optional target frame rate for resampling
            timeout: Maximum time to wait for frames in seconds
            latest: Whether to read from the most recent frame or continue from last position
            start_time: Start time in seconds for file source (only used with file source)

        Returns:
            Single VideoFrame if duration is None/0, or list of VideoFrames if duration > 0.

        Notes:
            If buffer duration is set too short, the read operation may timeout since the
            last_read_pos might be always equal to the current tail.
        """
        # File source - direct read without buffering
        if self.source == 'file':
            return self._read_from_file(start_time, duration, target_fps)
        
        if duration is None or duration == 0:
            frames_needed = 1
        else:
            frames_needed = int(duration * self.fps)

        total_frames = []
        start_time_actual = time.time()

        if latest:
            self._last_read_pos = -1

        while len(total_frames) < frames_needed:
            current_tail = self.buffer.get_tail()

            # For first/latest read, start from most recent frame
            if self._last_read_pos == -1:
                self._last_read_pos = current_tail
                continue

            if current_tail == self._last_read_pos:
                if time.time() - start_time_actual > timeout:
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
            start_time_actual = time.time()

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

    def _read_from_file(self, start_time: float, duration: float = None, target_fps: float = None) -> VideoFrame | list[VideoFrame] | None:
        """Read video frames directly from file based on start time and duration.
        
        Args:
            start_time (float): Start time in seconds.
            duration (float, optional): Duration to read in seconds. If None, returns single frame at start_time.
            target_fps (float, optional): Target frame rate for resampling.
            
        Returns:
            VideoFrame or list[VideoFrame]: Video frame(s) for the specified time range, or None if invalid range.
        """
        if self.video_capture is None:
            logger.error("Video capture not initialized")
            return None
            
        # Calculate start frame position
        start_frame = int(start_time * self.file_fps)
        
        # Check bounds
        if start_frame >= self.total_frames or start_frame < 0:
            logger.warning(f"Start time {start_time} is out of bounds")
            return None
            
        # Set video position to start frame
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        if duration is None or duration == 0:
            # Read single frame
            ret, frame = self.video_capture.read()
            if not ret:
                logger.warning(f"Failed to read frame at time {start_time}")
                return None
                
            return VideoFrame(
                data=frame,
                timestamp=start_time,
                metadata={
                    'resolution': (frame.shape[1], frame.shape[0]),
                    'fps': self.file_fps,
                    'format': self.format
                }
            )
        else:
            # Read multiple frames for duration
            frames_to_read = int(duration * self.file_fps)
            end_frame = start_frame + frames_to_read
            
            # Adjust end frame if it exceeds file length
            if end_frame > self.total_frames:
                end_frame = self.total_frames
                actual_duration = (end_frame - start_frame) / self.file_fps
                logger.info(f"Adjusting duration to fit file length: {actual_duration:.3f}s")
                
            frames = []
            current_frame = start_frame
            
            while current_frame < end_frame:
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.warning(f"Failed to read frame {current_frame}")
                    break
                    
                timestamp = current_frame / self.file_fps
                frames.append(VideoFrame(
                    data=frame,
                    timestamp=timestamp,
                    metadata={
                        'resolution': (frame.shape[1], frame.shape[0]),
                        'fps': self.file_fps,
                        'format': self.format
                    }
                ))
                current_frame += 1
                
            if not frames:
                logger.warning("No frames read from file")
                return None
                
            # Apply fps conversion if needed
            if target_fps and target_fps != self.file_fps:
                return self._process_frames(frames, target_fps)
                
            return frames

    @staticmethod
    def require_kwarg(kwargs, key, message):
        value = kwargs.get(key)
        if value is None:
            raise ValueError(message)
        return value
