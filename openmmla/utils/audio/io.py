import wave
from typing import List, Tuple, Union

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
        sig (np.array): signal/audio array in [-1.0, 1.0] range.
        fs (int): sampling rate.
        filename (str): output file path.

    Outputs:
        Save a wave file to output_file_path.
    """
    if sig.dtype in [np.float32, np.float64]:
        sig = float32_to_int16(sig)
    write(filename=filename, rate=fs, data=sig)


def read_frames_from_wav(audio_path: str):
    """Read the audio file as a byte string.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        frames: The audio frames as a byte string.
    """
    with wave.open(audio_path, 'rb') as wav_file:
        byte_string = wav_file.readframes(wav_file.getnframes())
        return byte_string


def write_frames_to_wav(output_path: str, frames: Union[bytes, List[bytes], Tuple[bytes]], channels: int = 1,
                        sampwidth: int = 2, framerate: int = 16000):
    """Write audio frames to a wave file. This function supports writing a single bytes object or a collection (list or
     tuple) of bytes objects to a wave file.

    Args:
        output_path (str): The file path where the wave file will be saved.
        frames (bytes | List[bytes] | Tuple[bytes]): The audio frames to write. This can either be a single bytes object
         containing all frames, or a list/tuple of bytes objects, each representing a frame.
        channels (int, optional): The number of audio channels. Default is 1.
        sampwidth (int, optional): The sample width in bytes. Default is 2.
        framerate (int, optional): The frame rate in Hz. Default is 16000.

    Raises:
        ValueError: If 'frames' is not a bytes object or a list/tuple of bytes objects.
    """
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.writeframes(frames)
