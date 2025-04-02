class RecordingError(Exception):
    """Exception raised when an error occurs in the recording process."""
    pass


class TranscribingError(Exception):
    """Exception raised when an error occurs in the transcribing process."""
    pass


class RecognizingError(Exception):
    """Exception raised when an error occurs in the recognizing process."""
    pass
