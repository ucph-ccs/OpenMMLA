"""This module contains utility functions for frequency-based audio data augmentation. Imported and modified based on pydiogment.

- convolve: Apply convolution to an audio file using the given impulse response file.
- change_tone: Change the tone of an audio file.
- apply_filter: Apply Butterworth filter to audio.
"""
import os
import subprocess

import librosa
import numpy as np
import soundfile as sf

from .filters import butter_filter
from .io import read_signal_from_wav, write_signal_to_wav


def resample_audio(infile: str, target_sr: int = 8000):
    """Resamples the audio file to target sample rate.

    Args:
        infile (str): The path to the audio file to be resampled.
        target_sr (int): The target sample rate for the audio data.
    """
    audio_data, original_sr = librosa.load(infile, sr=None)  # sr=None ensures original SR is used
    resampled_audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    sf.write(infile, resampled_audio_data, target_sr)


def convolve(infile: str, ir_fname: str, level: float = 0.5) -> None:
    """Apply convolution to infile using the given impulse response file.

    Args:
        infile: Input audio file path
        ir_fname: Impulse response file path
        level: Mixing level between 0 and 1 (default: 0.5)
    """
    fs1, x = read_signal_from_wav(audio_path=infile)
    _, ir = read_signal_from_wav(audio_path=ir_fname)

    # Apply convolution
    y = np.convolve(x, ir, 'full')[:x.shape[0]] * level + x * (1 - level)
    y /= np.mean(np.abs(y))  # Normalize

    ir_name = os.path.splitext(os.path.basename(ir_fname))[0]
    outfile = os.path.splitext(infile)[0] + f"_augmented_{ir_name}_convolved_with_level_{level}.wav"
    write_signal_to_wav(sig=y, fs=fs1, output_path=outfile)


def change_tone(infile: str, tone: int) -> None:
    """Change the tone of an audio file.

    Args:
        infile: Input audio file path
        tone: Tone change factor
    """
    fs, _ = read_signal_from_wav(audio_path=infile)
    outfile = os.path.splitext(infile)[0] + f"_augmented_{tone}_toned.wav"

    tone_change_command = [
        "ffmpeg", "-i", infile, "-af",
        f"asetrate={fs}*{tone},aresample={fs}",
        outfile
    ]
    subprocess.Popen(tone_change_command,
                     stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)


def apply_filter(infile: str, filter_type: str, low_cutoff_freq: float,
                 high_cutoff_freq: float = None, order: int = 5) -> None:
    """Apply Butterworth filter to audio.

    Args:
        infile: Input audio file path
        filter_type: Type of filter to apply
        low_cutoff_freq: Low cut-off frequency
        high_cutoff_freq: High cut-off frequency (optional)
        order: Filter order (default: 5)
    """
    fs, sig = read_signal_from_wav(audio_path=infile)

    # Apply filter
    y = butter_filter(sig=sig, fs=fs, ftype=filter_type,
                      low_cut=low_cutoff_freq,
                      high_cut=high_cutoff_freq,
                      order=order)

    outfile = os.path.splitext(infile)[0] + f"_augmented_{filter_type}_pass_filtered.wav"
    write_signal_to_wav(sig=y, fs=fs, output_path=outfile)
