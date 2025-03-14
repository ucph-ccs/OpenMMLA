"""This module contains utility functions for reading and writing audio files in np.array and byte string.

- int16_to_float32: Convert int16 to float32 [-1.0, 1.0].
- float32_to_int16: Convert float32 [-1.0, 1.0] to int16.
- read_signal_from_wav: Read a wave file as normalized mono signal.
- write_signal_to_wav: Write processed signal to a wave file as mono.
- read_frames_from_wav: Read the audio file as a byte string.
- write_frames_to_wav: Write audio frames to a wave file.
"""

import wave
from typing import List, Tuple, Union

import numpy as np
import soundfile as sf


# 16-bit integer range:
# MIN_INT16 = -32768  # -2¹⁵
# MAX_INT16 = 32767  # 2¹⁵ - 1


# def int16_to_float32(data: np.ndarray) -> np.ndarray:
#     """Convert int16 to float32 [-1.0, 1.0]."""
#     return data.astype(np.float32) / 32768.0  # Use 2¹⁵
#
#
# def float32_to_int16(data: np.ndarray) -> np.ndarray:
#     """Convert float32 [-1.0, 1.0] to int16."""
#     return np.clip(data * 32768.0, MIN_INT16, MAX_INT16).astype(np.int16)
#
#
# def read_signal_from_wav(filename: str):
#     """Read a wave file as normalized mono signal.
#
#     Args:
#         filename (str): string path or open file handle.
#
#     Returns:
#         tuple of sampling rate and audio data normalized to [-1.0, 1.0].
#     """
#     fr, sig = read(filename=filename)
#     if sig.ndim == 1:
#         samples = sig
#     else:
#         samples = sig[:, 0]  # Convert to mono if stereo
#
#     if sig.dtype == np.int16:
#         samples = int16_to_float32(samples)
#     return fr, samples
#
#
# def write_signal_to_wav(sig: np.ndarray, fs: int, filename: str):
#     """Write processed signal to a wave file as mono.
#
#     Args:
#         sig (np.array): signal/audio array in [-1.0, 1.0] range.
#         fs (int): sampling rate.
#         filename (str): output file path.
#
#     Outputs:
#         Save a wave file to output_file_path.
#     """
#     if sig.dtype in [np.float32, np.float64]:
#         sig = float32_to_int16(sig)
#     write(filename=filename, rate=fs, data=sig)


def read_signal_from_wav(audio_path: str):
    """Read a wave file as a normalized mono signal.

    Args:
        audio_path (str): Path to the wave file.
    Returns:
        tuple: (sampling_rate, mono_signal) with signal values in [-1.0, 1.0]
    """
    data, samplerate = sf.read(audio_path)  # returns float64 in [-1.0, 1.0]
    if data.ndim > 1:
        data = data[:, 0]
    return samplerate, data


def write_signal_to_wav(sig: np.ndarray, fs: int, output_path: str):
    """Write a normalized signal to a wave file.

    Args:
        output_path (str): Output file path.
        sig (np.array): Signal array in [-1.0, 1.0].
        fs (int): Sampling rate.
    """
    # Specify PCM_16 subtype to have float-to-int16 conversion
    sf.write(output_path, sig, fs, subtype='PCM_16')


def read_bytes_from_wav(audio_path: str):
    """Read the audio file as a byte string.

    Args:
        audio_path (str): Path to the wave file.

    Returns:
        frames: The audio frames as a byte string.
    """
    with wave.open(audio_path, 'rb') as wav_file:
        byte_string = wav_file.readframes(wav_file.getnframes())
        return byte_string


def write_bytes_to_wav(output_path: str, frames: Union[bytes, List[bytes], Tuple[bytes]], channels: int = 1,
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
    """
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.writeframes(frames)
