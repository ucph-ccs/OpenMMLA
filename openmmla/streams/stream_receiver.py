"""
StreamReceiver is a base class for receiving data streams, with specified sampling rate, channels, etc.
Data is received into a buffer, and the buffer can be read by the consumer.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# stamps that already carry the capture side's clock: a measured receive-side delay does not apply
CAPTURE_SIDE_STAMPS = ('rtmp_stream_start_pts', 'rtmp_absolute_pts', 'lsl', 'sender_packet_header')

# how long a base waits for a network stream (rtmp/rtsp/srt, an LSL outlet): one that is not up yet when the
# base starts (its publisher may still be starting), and one that dropped while it runs (a relay hiccup ended
# the publisher, and the capture side publishes again). The reconnect wait is about a session long, so that a
# stream that comes back within the session is taken up again. A camera index, a file or a microphone of this
# machine is there or not, and fails at once as before
NETWORK_SOURCES = ('stream', 'lsl')
DEFAULT_CONNECT_WAIT = 30.0
DEFAULT_RECONNECT_WAIT = 3600.0
RETRY_INTERVAL = 2.0        # between tries for the first SLOW_AFTER seconds of a wait
SLOW_RETRY_INTERVAL = 5.0   # then: a long outage is tried, and logged by the libraries, less often
SLOW_AFTER = 30.0
WAIT_LOG_EVERY = 15.0


class StreamUnavailable(RuntimeError):
    """a network stream that did not come up within its wait, at start or after a drop, or a source that
    stopped delivering for good: the base ends its run on it (an ASR base does not restart, which would wait
    for a START that was already sent)"""


def wait_seconds(value, default, name, log) -> float:
    """a wait from stream_kwargs, read leniently as timestamp_offset is: empty is the default, a negative
    number is 0 (one try, no waiting), and a typo is said and the default used"""
    if value is None or str(value).strip() == '':
        return float(default)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        log.warning(f"stream_kwargs.{name} {value!r} is not a number of seconds: {float(default):g} s is used.")
        return float(default)


def as_sentence(text) -> str:
    """the text ending with a full stop, so that a base can append its own sentence to it"""
    text = str(text).strip()
    return text if text.endswith(('.', '?', '!')) else f"{text}."


class StreamReceiver(ABC):
    """Base class for receiving data streams.

    Handles continuous data streams with specified parameters (sampling rate, channels, etc.).
    Data is received into a buffer and can be read by the consumer.
    """

    # where a stream says what it does when its caller passes no log of its own
    default_log = logger

    def __init__(self, **kwargs):
        """Initialize stream receiver.

        Args:
            **kwargs: Configuration parameters for the stream

        Keyword Args:
            log (logging.Logger, optional): the caller's logger. A base passes its session logger, so that
                the waits and drops of its stream are in the session's log file
            abort_event (threading.Event, optional): the caller's own stop (the ASR base's stop_event),
                which ends a wait for the stream as stop() does
            connect_wait (float, optional): seconds a network stream that is not up yet is waited for at
                start (default DEFAULT_CONNECT_WAIT); set by the subclasses, which know their source
            reconnect_wait (float, optional): seconds a network stream that dropped is opened again for
                (default DEFAULT_RECONNECT_WAIT)
        """
        self.source = None  # stream type: 'pyaudio', 'socket', 'cv2', 'rtmp', 'lsl', etc.
        self.stream = None  # stream object from pyaudio, cv2, etc.
        self.buffer = None  # ring buffer for data storage
        self.config = kwargs
        # the base's own logger: the waits and gaps go in the session's log file, which the module loggers
        # (console only) never reach
        caller_log = kwargs.get('log')
        self.log = caller_log or self.default_log
        # a measured constant delay of this stream, in seconds, added to the stamp of every live
        # frame: negative moves the stamps back from when a frame arrived to when it was captured
        raw_offset = kwargs.get('timestamp_offset')
        try:
            self.timestamp_offset = float(raw_offset or 0.0)
        except (TypeError, ValueError):
            (caller_log or logger).warning(f"timestamp_offset {raw_offset!r} is not a number of seconds: no offset applied")
            self.timestamp_offset = 0.0
        if self.timestamp_offset:
            (caller_log or logger).info(f"Stream stamps are moved by timestamp_offset {self.timestamp_offset:+.3f} s")
        self._stamp_source_logged = False

        # the caller's own stop: it ends a wait that runs in the caller's thread
        self.abort_event = kwargs.get('abort_event')
        self._stop_event = threading.Event()
        self.state = 'idle'      # 'connecting', 'live', 'reconnecting', 'failed' or 'stopped'
        self.failure = None      # what read() raises once the state is 'failed'
        self.gaps = []           # [lost_at, back_at or None] of each drop, wall clock
        self._gaps_said = 0
        self.connect_wait = 0.0
        self.reconnect_wait = 0.0

    def _set_waits(self, kwargs) -> None:
        """connect_wait and reconnect_wait of this source: a network stream is waited for, a camera index,
        a file or a microphone of this machine is not (0 s: one try, as before)"""
        network = self.source in NETWORK_SOURCES
        self.connect_wait = wait_seconds(kwargs.get('connect_wait'), DEFAULT_CONNECT_WAIT if network else 0.0,
                                         'connect_wait', self.log)
        self.reconnect_wait = wait_seconds(kwargs.get('reconnect_wait'),
                                           DEFAULT_RECONNECT_WAIT if network else 0.0, 'reconnect_wait', self.log)

    def _stamped(self, frame):
        """the frame with this stream's timestamp_offset applied (None stays None). A frame whose
        stamp already comes from the capture side (timestamp_source in CAPTURE_SIDE_STAMPS) is
        left alone: the delay that was measured is not in it."""
        if frame is None:
            return None
        source = (frame.metadata or {}).get('timestamp_source') if isinstance(frame.metadata, dict) else None
        if not self._stamp_source_logged and source:
            self.log.info(f"Stream stamps come from {source}")
            self._stamp_source_logged = True
        if self.timestamp_offset and source not in CAPTURE_SIDE_STAMPS:
            frame.timestamp = frame.timestamp + self.timestamp_offset
            # a fresh dict: audio frames share one metadata dict across frames
            frame.metadata = dict(frame.metadata or {}, timestamp_offset=self.timestamp_offset)
        return frame

    def _stopping(self) -> bool:
        """stop() was called, or the caller's own stop (abort_event) came"""
        return self._stop_event.is_set() or bool(self.abort_event is not None and self.abort_event.is_set())

    def _wait_for(self, open_once, what: str, wait: float, pause: bool = True) -> bool:
        """try open_once() until the stream is up, for up to `wait` seconds: True once it is up, False when
        the wait ran out or a stop came (_stopping() says which). Never a busy loop: after a failed try it
        waits RETRY_INTERVAL (SLOW_RETRY_INTERVAL after SLOW_AFTER seconds), in slices of 0.5 s, so that
        stop() or the abort_event end the wait at once. pause=False for a try that waits by itself (an LSL
        resolve). A try that takes long (an open at an unreachable server) may overrun the wait by one try."""
        started = said = time.monotonic()
        deadline = started + wait
        tries = 0
        while not self._stopping():
            tries += 1
            if open_once():
                if tries > 1:
                    self.log.info(f"{what} is up after {time.monotonic() - started:.0f} s ({tries} tries).")
                return True
            now = time.monotonic()
            if now >= deadline or self._stopping():
                return False
            if tries == 1:
                self.log.warning(f"{what} is not up: trying again every {RETRY_INTERVAL:g} s for up to {wait:g} s.")
            elif now - said >= WAIT_LOG_EVERY:
                self.log.info(f"Still waiting for {what}: {now - started:.0f} s of {wait:g} s.")
                said = now
            if pause:
                interval = RETRY_INTERVAL if now - started < SLOW_AFTER else SLOW_RETRY_INTERVAL
                end = min(deadline, now + interval)
                while time.monotonic() < end:
                    if self._stop_event.wait(max(0.0, min(0.5, end - time.monotonic()))) or self._stopping():
                        return False
        return False

    def _gap_summary(self, what: str) -> None:
        """one line on the drops of this stream, said when it stops (once: a base may stop it twice)"""
        if not self.gaps or self._gaps_said == len(self.gaps):
            return
        self._gaps_said = len(self.gaps)
        lost = sum((back or time.time()) - at for at, back in self.gaps)
        self.log.info(f"{what}: {len(self.gaps)} drop(s), {lost:.0f} s without data in all.")

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
