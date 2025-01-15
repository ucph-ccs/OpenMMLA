# Frequency-based augmentation techniques/manipulations for audio data.
# Imported and modified based on pydiogment
import os
import subprocess

import numpy as np

from .filters import butter_filter
from .io import read_signal_from_wav, write_signal_to_wav


def convolve(infile: str, ir_fname: str, level: float = 0.5) -> None:
    """Apply convolution to infile using the given impulse response file.

    Args:
        infile: Input audio file path
        ir_fname: Impulse response file path
        level: Mixing level between 0 and 1 (default: 0.5)
    """
    fs1, x = read_signal_from_wav(filename=infile)
    _, ir = read_signal_from_wav(filename=ir_fname)

    # Apply convolution
    y = np.convolve(x, ir, 'full')[:x.shape[0]] * level + x * (1 - level)
    y /= np.mean(np.abs(y))  # Normalize

    ir_name = os.path.splitext(os.path.basename(ir_fname))[0]
    outfile = os.path.splitext(infile)[0] + f"_augmented_{ir_name}_convolved_with_level_{level}.wav"
    write_signal_to_wav(sig=y, fs=fs1, filename=outfile)


def change_tone(infile: str, tone: int) -> None:
    """Change the tone of an audio file.

    Args:
        infile: Input audio file path
        tone: Tone change factor
    """
    fs, _ = read_signal_from_wav(filename=infile)
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
    fs, sig = read_signal_from_wav(filename=infile)

    # Apply filter
    y = butter_filter(sig=sig, fs=fs, ftype=filter_type,
                      low_cut=low_cutoff_freq,
                      high_cut=high_cutoff_freq,
                      order=order)

    outfile = os.path.splitext(infile)[0] + f"_augmented_{filter_type}_pass_filtered.wav"
    write_signal_to_wav(sig=y, fs=fs, filename=outfile)
