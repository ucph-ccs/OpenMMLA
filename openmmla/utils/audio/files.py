import numpy as np
from scipy.io.wavfile import read, write

# 16-bit integer range:
MIN_INT16 = -32768  # -2¹⁵
MAX_INT16 = 32767  # 2¹⁵ - 1


def int16_to_float32(data: np.ndarray) -> np.ndarray:
    """Convert int16 to float32 [-1.0, 1.0]."""
    return data.astype(np.float32) / 32768.0  # Use 2¹⁵


def float32_to_int16(data: np.ndarray) -> np.ndarray:
    """Convert float32 [-1.0, 1.0] to int16."""
    return np.clip(data * 32768.0, MIN_INT16, MAX_INT16).astype(np.int16)


def read_signal_from_wav(filename: str):
    """Read a wave file as normalized mono signal.

    Args:
        filename (str): string path or open file handle.

    Returns:
        tuple of sampling rate and audio data normalized to [-1.0, 1.0].
    """
    fr, sig = read(filename=filename)
    if sig.ndim == 1:
        samples = sig
    else:
        samples = sig[:, 0]  # Convert to mono if stereo

    if sig.dtype == np.int16:
        samples = int16_to_float32(samples)
    return fr, samples


def write_signal_to_wav(sig: np.ndarray, fs: int, filename: str):
    """Write processed signal to a wave file as mono.

    Args:
        sig (array): signal/audio array in [-1.0, 1.0] range.
        fs (int): sampling rate.
        filename (str): output file path.

    Outputs:
        Save a wave file to output_file_path.
    """
    if sig.dtype in [np.float32, np.float64]:
        sig = float32_to_int16(sig)
    write(filename=filename, rate=fs, data=sig)
