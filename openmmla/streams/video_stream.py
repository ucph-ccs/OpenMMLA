import os
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
from openmmla.utils.constants import normalize_source
from openmmla.utils.logger import get_logger
from openmmla.utils.stream_registry import resolve_stream_by_target, resolve_rtmp_timestamp
from openmmla.utils.threads import RaisingThread
from .frame import VideoFrame
from .stream_buffer import RingBuffer
from .stream_receiver import StreamReceiver, StreamUnavailable, as_sentence
from . import stream_receiver as receiver

# ffmpeg demuxer options OpenCV applies to network streams unless the operator
# exported OPENCV_FFMPEG_CAPTURE_OPTIONS: RTSP over TCP (no packet loss on
# Wi-Fi) and no read-ahead buffering, which is where most pull-side latency hides
DEFAULT_STREAM_CAPTURE_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

OPEN_TIMEOUT_MS = 15000   # a try at an unreachable Stream Server costs this, not OpenCV's 30 s; an RTMP pull probes ~6 s
READ_TIMEOUT_MS = 10000   # a stalled pull ends its read here (OpenCV's default is 30 s)
MAX_READ_FAILURES = 10    # reads failing one after another: the stream ended (MediaMTX ended this reader)
STALL_SECONDS = 10.0      # or no frame for this long: it stalls

# logger = get_logger(__name__, console_level=logging.DEBUG)
logger = get_logger(__name__)


def open_network_capture(url: str):
    """a capture of a network stream whose open and reads are bounded (OpenCV >= 4.5.2; an older one gets
    its own 30 s defaults)"""
    open_prop = getattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC', None)
    read_prop = getattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC', None)
    if open_prop is None or read_prop is None:
        return cv2.VideoCapture(url)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [open_prop, OPEN_TIMEOUT_MS, read_prop, READ_TIMEOUT_MS])


class VideoStream(StreamReceiver):
    """Video stream implementation for continuous video capture."""

    default_log = logger

    def __init__(self, source: str, **kwargs):
        """Initialize video stream.

        Args:
            source (str): Stream source type ('opencv', 'stream', 'lsl', or 'file'; 'rtmp' is an alias of 'stream')

        Keyword Args:
            buffer_duration (float, optional): Duration of the ring buffer in seconds (default: 0.08)
            format(str, optional): Video format, 'MJPG', 'JPEG', 'H264', 'raw' (default: 'MJPG')
            resolution(tuple, optional): Tuple of (width, height) (default: (1920, 1080))
            fps(int, optional): Frames per second (default: 30)
            resample_method(ResampleMethod, optional): Resampling method for fps conversion (default: ResampleMethod.VIDEO_AVERAGE)
            camera_index(int, optional): Camera index for 'opencv' source (default: 0)
            url(str, optional): stream URL (rtmp://, rtsp:// or srt://) for the 'stream' source; 'rtmp_url' is still accepted
            capture_options(str, optional): OPENCV_FFMPEG_CAPTURE_OPTIONS applied to the 'stream' source when the
                variable is unset (default: DEFAULT_STREAM_CAPTURE_OPTIONS)
            file_path(str, optional): Path to the video file (required for 'file' source)
            timestamp_offset(float, optional): seconds added to every live frame's stamp, the measured
                delay of this stream as a negative number (default: 0)
            connect_wait(float, optional): seconds a 'stream' or 'lsl' source that is not up yet is waited
                for at start (default: 30); a camera index or a file is tried at once
            reconnect_wait(float, optional): seconds a 'stream' or 'lsl' source that dropped is opened
                again for before read() raises StreamUnavailable (default: 3600)
            log(logging.Logger, optional): the caller's logger, for the waits and drops of the stream
        """
        super().__init__(**kwargs)
        self.source = normalize_source(source)

        # Stream configuration
        self.buffer_duration = kwargs.get('buffer_duration', 0.08)
        self.resolution = kwargs.get('resolution', (1920, 1080))
        self.fps = kwargs.get('fps', 30)
        self.resample_method = kwargs.get('resample_method', ResampleMethod.VIDEO_AVERAGE)
        self.project_dir = kwargs.get('project_dir')
        self.stream_registry_path = kwargs.get('stream_registry_path')

        # Source-specific configuration
        if self.source == 'opencv':
            self.format = kwargs.get('format', 'MJPG')
            self.camera_index = kwargs.get('camera_index', 0)
            self.stream = None
        elif self.source == 'stream':
            self.format = kwargs.get('format', 'H264')
            url = kwargs.get('url') or kwargs.get('rtmp_url')
            if not url:
                raise ValueError("Stream source requires a 'url' parameter (rtmp://, rtsp:// or srt://)")
            self.url = str(url)
            self.rtmp_url = self.url  # name from before MediaMTX, kept for callers
            self.capture_options = kwargs.get('capture_options', DEFAULT_STREAM_CAPTURE_OPTIONS)
            self.stream = None
            self._load_stream_registry_entry()
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

        # only a network stream is waited for: a camera index or a file is there or it is not
        self._set_waits(kwargs)
        self._last_frame_at = time.monotonic()
        self._lingering_thread = None
        # guards the swap of self.stream: a stop() from another thread and an opening never release one
        # capture twice
        self._capture_lock = threading.Lock()

        # Frame metadata
        self._frame_metadata = {
            'resolution': self.resolution,
            'fps': self.fps,
            'format': self.format
        }

        # Buffer
        # never fewer than two slots: read() waits for the tail to move, and the
        # tail of a one-slot ring never does (fps 15 with the template's 0.08 s
        # gave int(1.2) = 1, and a base that timed out on every read)
        buffer_frames = max(2, int(self.fps * self.buffer_duration))
        self.buffer = RingBuffer(buffer_frames)
        self._last_read_pos = -1

        # PTS-to-wallclock calibration for RTMP sources
        self._pts_offset: float | None = None

        # Threading control
        self._stop_event = threading.Event()
        self._receive_thread = None

    def start(self) -> None:
        """Start the video stream and begin capturing frames. A network stream that is not up yet is
        waited for, up to connect_wait seconds; one that does not come up raises StreamUnavailable. A
        stop() from another thread (or the abort_event) ends the wait, and start() then starts nothing."""
        self.stop()
        lingering = self._lingering_thread
        if lingering is not None and lingering.is_alive():
            # the receive thread of the last start is still inside an OpenCV call: it releases its capture
            # when that returns, and must be gone before a new capture takes its place and before the stop
            # event it shares is cleared, which would have it read on beside the new receive thread
            lingering.join(timeout=(OPEN_TIMEOUT_MS + READ_TIMEOUT_MS) / 1000)
            if lingering.is_alive():
                raise RuntimeError(f"{self._what()} cannot start again yet: the receive thread of its last start "
                                   f"is still in a read.")
        self._lingering_thread = None
        # cleared before the opening, which may wait: a stop() from another thread then ends that wait
        self._stop_event.clear()
        self.state, self.failure = 'connecting', None
        try:
            if self.source in ['opencv', 'stream']:
                self._initialize_opencv()
            elif self.source == 'lsl':
                self._initialize_lsl()
            elif self.source == 'file':
                self._initialize_file()
                # 'file' source doesn't need a receive thread - data is read on demand
                self.state = 'live'
                self.log.info(f"Video stream started with source: {self.source}")
                return
            else:
                raise ValueError(f"Unsupported source type: {self.source}")
        except Exception as e:
            self.state, self.failure = 'failed', as_sentence(e)
            raise
        except BaseException:
            self.state = 'stopped'
            raise
        if self._stopping():
            # stopped while it opened: nothing is started
            self._cleanup_source()
            self.state = 'stopped'
            return

        self.state = 'live'
        self._receive_thread = RaisingThread(target=self._receive_loop, name=f'video-{self.source}')
        self._receive_thread.daemon = True
        self._receive_thread.start()
        self.log.info(f"Video stream started with source: {self.source}")

    def stop(self) -> None:
        """Stop the video stream and clean up resources."""
        self._stop_event.set()
        thread, self._receive_thread = self._receive_thread, None
        if thread is not None and threading.current_thread() is not thread:
            try:
                thread.join(timeout=5)
            except Exception as e:
                self.log.warning(f"During thread stopping, caught: {e}", exc_info=True)
            if thread.is_alive():
                # still inside an OpenCV read or open, and a capture must not be released under it: the
                # thread releases it itself when that call returns (READ_TIMEOUT_MS / OPEN_TIMEOUT_MS at most)
                self.log.warning(f"{self._what()} is still in a read; it is released when that read returns.")
                self._lingering_thread = thread
                self._gap_summary(self._what())
                self._last_read_pos = -1
                if self.state != 'failed':
                    self.state = 'stopped'
                return

        lingering = self._lingering_thread
        if lingering is not None and lingering.is_alive() and threading.current_thread() is not lingering:
            # a second stop while the receive thread of the first is still in its read: that thread releases
            # the capture when the read returns, which must not be released under it here
            self._gap_summary(self._what())
            self._last_read_pos = -1
            if self.state != 'failed':
                self.state = 'stopped'
            return
        self._cleanup_source()
        self._gap_summary(self._what())
        self._last_read_pos = -1
        if self.state != 'failed':
            self.state = 'stopped'

    def _cleanup_source(self) -> None:
        """release what the source holds; safe to call again"""
        if self.source in ['opencv', 'stream']:
            self._cleanup_opencv()
        elif self.source == 'lsl':
            self._cleanup_lsl()
        elif self.source == 'file':
            self._cleanup_file()

    def _what(self) -> str:
        """the stream, as the log lines name it"""
        if self.source == 'stream':
            return f"Stream {self.url}"
        if self.source == 'opencv':
            return f"Camera index {self.camera_index}"
        if self.source == 'lsl':
            return f"LSL stream '{self.lsl_name}'"
        return f"Video file {getattr(self, 'file_path', '')}"

    def _load_stream_registry_entry(self) -> None:
        """Look the stream up in the runtime registry for its capture-side start time."""
        self._stream_registry_entry = resolve_stream_by_target(
            self.url,
            project_dir=self.project_dir,
            registry_path=self.stream_registry_path,
        )
        self._stream_start_time = (
            float(self._stream_registry_entry["stream_start_time"])
            if self._stream_registry_entry
            else None
        )

    def _open_capture(self) -> bool:
        """one try at the camera or the stream; False leaves nothing open"""
        self._cleanup_opencv()
        if self.source == 'stream':
            # read each try: a stream started again meanwhile may have a new start time in the registry
            self._load_stream_registry_entry()
            # an explicit value exported by the operator wins over the defaults
            if self.capture_options and not os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS'):
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = self.capture_options
            capture = open_network_capture(self.url)
        else:
            capture = cv2.VideoCapture(self.camera_index)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.format))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        with self._capture_lock:
            self.stream = capture
        if capture.isOpened() and not self._stopping():
            return True
        # also a capture that opened just as a stop came: nobody else releases it
        self._cleanup_opencv()
        return False

    def _initialize_opencv(self, wait: float | None = None, after_drop: bool = False) -> None:
        """Open the camera or the stream. A camera of this machine gets four quick tries, as before. A
        network stream that is not up yet (the Stream Server answers 404 while nobody publishes on the
        path, or cannot be reached) is tried again every RETRY_INTERVAL s for up to `wait` seconds:
        connect_wait at start, reconnect_wait after a drop. Returns without a capture when a stop came
        meanwhile; raises StreamUnavailable when the stream did not come up."""
        self._pts_offset = None
        if self.source == 'opencv':
            for attempt in range(4):
                if self._open_capture():
                    self.log.info(f"Successfully initialized opencv video stream with seed: {self.camera_index}")
                    return
                if attempt < 3:
                    self.log.warning(f"Camera index {self.camera_index} did not open, trying again.")
            raise RuntimeError(f"Camera index {self.camera_index} could not be opened.")
        wait = self.connect_wait if wait is None else wait
        if self._wait_for(self._open_capture, self._what(), wait):
            self.log.info(f"Successfully initialized stream video stream with seed: {self.url}")
            return
        if self._stopping():
            return  # a stop came meanwhile: the caller starts nothing
        came = 'come back' if after_drop else 'come up'
        raise StreamUnavailable(
            f"Stream {self.url} did not {came} within {wait:g} s: nobody publishes on that path (is its stream "
            f"running? Streams tab, Status and Stream Server columns), or the Stream Server cannot be reached.")

    def _initialize_lsl(self, wait: float | None = None, after_drop: bool = False) -> None:
        """Connect to the LSL stream, waiting for its outlet up to `wait` seconds (connect_wait at start,
        reconnect_wait after a drop). Returns without an inlet when a stop came meanwhile; raises
        StreamUnavailable when no outlet of that name showed up."""
        if resolve_byprop is None or StreamInlet is None or local_clock is None:
            raise ImportError(
                "pylsl package is not installed. Please install it with 'pip install pylsl' to use LSL features."
            )

        def open_once() -> bool:
            self._cleanup_lsl()
            # the resolve itself waits, a retry interval long
            streams = resolve_byprop('name', self.lsl_name, timeout=receiver.RETRY_INTERVAL)
            if not streams:
                return False
            self.lsl_inlet = StreamInlet(streams[0])
            self.lsl_offset = time.time() - local_clock()
            return True

        wait = self.connect_wait if wait is None else wait
        if self._wait_for(open_once, self._what(), wait, pause=False):
            self.log.info(f"Successfully initialized LSL video stream with inlet: {self.lsl_inlet}")
            return
        if self._stopping():
            self._cleanup_lsl()
            return
        came = 'come back' if after_drop else 'come up'
        raise StreamUnavailable(f"LSL stream '{self.lsl_name}' did not {came} within {wait:g} s: no outlet of "
                                f"that name was found on the network.")

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
        """Clean up OpenCV stream resources; safe to call again and from another thread."""
        lock = getattr(self, '_capture_lock', None)
        if lock is None:
            capture, self.stream = self.stream, None
        else:
            with lock:
                capture, self.stream = self.stream, None
        if capture:
            capture.release()

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
        """Continuously receive frames and store in buffer. A stream that drops is opened again
        (_reconnect); this loop never raises: a fault reaches the base through read()."""
        frame_count = 0
        failure_count = 0
        start_time = time.time()
        self._last_frame_at = time.monotonic()
        self.log.info("Starting video stream receive loop.")
        try:
            while not self._stop_event.is_set():
                # pushed even when a stop came during the read
                frame = self._stamped(self._read_frame())
                if frame:
                    self.buffer.push(frame)
                    self._last_frame_at = time.monotonic()
                    frame_count += 1
                    failure_count = 0
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        self.log.debug(f"Video capture FPS: {fps:.2f}")
                        start_time = time.time()
                        frame_count = 0
                    continue
                if self._stop_event.is_set():
                    break  # a read that failed as the stream was stopped: no drop
                failure_count += 1
                silent = time.monotonic() - self._last_frame_at
                # reads failing one after another are the end of the stream; a read that blocked until its
                # timeout, or no frame for STALL_SECONDS, is a stall
                if failure_count < MAX_READ_FAILURES and silent < STALL_SECONDS:
                    continue
                failure_count = 0
                if not self._reconnect(silent):
                    break
        except Exception as e:
            # a fault here must reach the base, not leave it waiting for frames that never come
            self.state, self.failure = 'failed', as_sentence(f"{self._what()} stopped receiving: {e}")
            self.log.error(self.failure, exc_info=True)
        finally:
            if self._stop_event.is_set() or self.state == 'failed':
                # this thread is the only one using the capture now (stop() leaves it alone while it runs)
                self._cleanup_source()

    def _reconnect(self, silent: float) -> bool:
        """The stream dropped: its publisher went (MediaMTX then ends the readers of its path), or nothing
        came for STALL_SECONDS. It is opened again while the base gets no frame: the frames of the gap are
        missing and none is made up, and the first frame after it is stamped afresh (the PTS calibration
        is reset and the registry read again). False when it did not come back (read() raises from then
        on) or a stop came."""
        gap = [time.time() - silent, None]
        self.gaps.append(gap)
        self.state = 'reconnecting'
        # reads that fail at once are an ended stream (MediaMTX ended this reader); else it stalled
        why = f"no frame for {silent:.0f} s" if silent >= STALL_SECONDS else "its reads failed"
        network = self.source in receiver.NETWORK_SOURCES
        again = f"Opening it again for up to {self.reconnect_wait:g} s" if network else "Opening it again"
        self.log.warning(f"{self._what()} dropped: {why}. {again}; its frames until then are missing.")
        try:
            if self.source in ['opencv', 'stream']:
                self._initialize_opencv(wait=self.reconnect_wait, after_drop=True)
            else:
                self._initialize_lsl(wait=self.reconnect_wait, after_drop=True)
        except Exception as e:
            self.state, self.failure = 'failed', as_sentence(e)
            self.log.error(f"{self.failure} The base ends its run on this.")
            return False
        if self._stopping():
            return False
        if self.source == 'stream' and self._stream_start_time is not None and self._stream_start_time < gap[0]:
            # the new connection's PTS counts from its own first packet, never from a start before the gap:
            # it is calibrated against this machine's clock (a stream started anew has a later start time)
            self._stream_start_time = None
        gap[1] = time.time()
        self.state = 'live'
        self._last_frame_at = time.monotonic()
        # the stamps may come from elsewhere now (the stream's start vs the calibrated PTS): said again
        self._stamp_source_logged = False
        self.log.info(f"{self._what()} is back after {gap[1] - gap[0]:.1f} s; the frames of that gap are missing.")
        return True

    def _read_frame(self) -> VideoFrame | None:
        """Read a single frame from the video stream."""
        try:
            if self.source in ['opencv', 'stream']:
                grabbed, frame = self.stream.read()
                if not grabbed:
                    # a drop is said once, by _reconnect, not by each failed read
                    self.log.debug("Failed to grab frame")
                    return None
                if self.source == 'stream':
                    pts_ms = self.stream.get(cv2.CAP_PROP_POS_MSEC)
                    received_time = time.time()
                    timestamp, timestamp_metadata = self._resolve_rtmp_timestamp(pts_ms, received_time)
                else:
                    timestamp = time.time()
                    timestamp_metadata = {"timestamp_source": "receiver_wallclock"}
            elif self.source == 'lsl':
                sample, timestamp = self.lsl_inlet.pull_sample(timeout=1.0)
                if not sample:
                    return None

                if self.format == 'raw':  # raw RGB data
                    data_array = np.array(sample, dtype=np.float32)
                    expected_size = self.resolution[0] * self.resolution[1] * 3  # width * height * RGB
                    if len(data_array) != expected_size:
                        self.log.warning(
                            f"Received LSL frame with unexpected size: {len(data_array)} (expected {expected_size})")
                        return None
                    frame = data_array.reshape((self.resolution[1], self.resolution[0], 3)).astype(np.uint8)
                elif self.format == 'JPEG':  # JPEG compressed data
                    data_array = np.array(sample, dtype=np.float32).astype(np.uint8)
                    try:
                        frame = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
                        if frame is None:
                            self.log.warning("Failed to decode JPEG data")
                            return None
                    except Exception as e:
                        self.log.error(f"Error decoding JPEG data: {e}", exc_info=True)
                        return None
                else:
                    raise ValueError(f"Unsupported format for LSL source: {self.format}")

                timestamp = (timestamp + self.lsl_offset) if timestamp else time.time()
                timestamp_metadata = {"timestamp_source": "lsl"}
            else:
                raise ValueError(f"Unsupported source type: {self.source}")

            metadata = dict(self._frame_metadata)
            metadata.update(timestamp_metadata)
            return VideoFrame(
                data=frame,
                timestamp=timestamp,
                metadata=metadata
            )

        except Exception as e:
            self.log.error(f"Error reading frame: {e}", exc_info=True)
            return None

    def _resolve_rtmp_timestamp(self, pts_ms: float, received_time: float) -> tuple[float, dict]:
        """resolve RTMP frame timestamp from stream-start metadata and media PTS."""
        previous_offset = self._pts_offset
        timestamp, metadata, self._pts_offset = resolve_rtmp_timestamp(
            pts_ms,
            received_time,
            stream_start_time=self._stream_start_time,
            receiver_pts_offset=self._pts_offset,
            stream_name=(self._stream_registry_entry or {}).get("name", ""),
        )
        if previous_offset is None and self._pts_offset is not None and metadata.get("timestamp_source") == "rtmp_receiver_calibrated_pts":
            self.log.info(f"RTMP PTS calibrated: offset={self._pts_offset:.3f}s")
        return timestamp, metadata

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

        # a stream that did not come back, or a source that stopped for good: the base ends its run on it
        if self.state == 'failed':
            raise StreamUnavailable(self.failure)

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
                if self.state == 'failed':
                    raise StreamUnavailable(self.failure)
                if time.time() - start_time_actual > timeout:
                    # while the stream is opened again it says so itself; a live one that falls silent is
                    # worth a line
                    self._say_quietly_unless_live("Timeout reached while waiting for frames.")
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

    def _say_quietly_unless_live(self, message: str) -> None:
        """a warning while the stream is live and no frame came for STALL_SECONDS; a debug line while it is
        opened again (or not started), which its own lines already say, and during the first STALL_SECONDS
        of a silence, after which the receive thread calls it a stall: a base reads every second, and
        warned twice a second until then"""
        silent = time.monotonic() - self._last_frame_at
        (self.log.warning if self.state == 'live' and silent >= STALL_SECONDS else self.log.debug)(message)

    def _process_frames(self, frames: list, target_fps: float | None) -> VideoFrame | list[VideoFrame] | None:
        """Process collected frames and apply fps conversion if needed.

        Args:
            frames: List of VideoFrames to process
            target_fps: Optional target frame rate

        Returns:
            Processed VideoFrame(s) or None if no frames available
        """
        if not frames:
            self._say_quietly_unless_live("No frames collected within timeout period.")
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
            metadata = dict(frames[0].metadata)
            metadata.update({
                'resolution': self.resolution,
                'fps': target_fps,
                'format': self.format
            })
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
