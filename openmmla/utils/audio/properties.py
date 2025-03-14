"""This module contains utility functions to calculate the properties of an audio file.

- get_energy_level: Calculate the volume level of an audio file.
- get_audio_properties: Get properties of an audio file.
- calculate_rms_db: Calculate the root-mean-square (RMS) energy of the audio data in decibels.
- calculate_audio_duration: Calculate the duration of the audio file.
"""
import wave
from typing import Dict

import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf


def get_energy_level(audio_file_path: str, verbose: bool = True) -> (float, float):
    """Calculate the volume level of audio file

    Args:
        verbose: Print out message or not
        audio_file_path(str): Path of the audio file

    Returns:
        tuple:
            - RMS Value (float): The Root Mean Square value representing the energy level of the audio file.
            - Peak Value (float): The maximum amplitude in the audio file.
    """
    rate, audio_samples = wav.read(audio_file_path)
    audio_samples = audio_samples.astype(float)
    rms_value = np.sqrt(np.mean(np.square(audio_samples)))
    peak_value = np.max(np.abs(audio_samples))
    if verbose:
        print(f"RMS Value: {rms_value}, Peak Value: {peak_value}")
    return rms_value, peak_value


def get_audio_properties(wav_file: str) -> Dict:
    """Get properties of an audio file.

    Args:
        wav_file (str): Path to the .wav audio file.

    Raises:
        Exception: If the file cannot be opened.

    Returns:
        dict: A dictionary containing properties of the audio file.
    """
    try:
        with wave.open(wav_file, 'rb') as wf:
            return {
                "framerate": wf.getframerate(),
                "channels": wf.getnchannels(),
                "sampwidth": wf.getsampwidth(),
                "nframes": wf.getnframes(),
                "duration": wf.getnframes() / float(wf.getframerate())
            }
    except wave.Error as e:
        raise Exception(f"Could not open file {wav_file} due to {str(e)}")


def calculate_rms_db(audio_samples: np.ndarray) -> float:
    """Calculate the root-mean-square (RMS) energy of the audio data in decibels.

    Args:
        audio_samples (np.ndarray): Numpy array of audio samples.

    Returns:
        float: Root mean square (RMS) energy of the audio data in decibels.
    """
    mean_square = np.mean(audio_samples ** 2)
    return 10 * np.log10(mean_square)


def calculate_audio_duration(audio_path: str):
    """Calculate the duration of the audio file

    Args:
        audio_path (str): the path of the audio

    Returns:
        float: the duration of the audio file in seconds
    """
    with sf.SoundFile(audio_path) as f:
        return len(f) / f.samplerate
