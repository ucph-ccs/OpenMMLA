"""This module contains utility functions for amplitude-based audio data augmentation. Imported and modified based on pydiogment.

- apply_gain: Apply gain to the audio file.
- resample_audio: Resamples the audio file to target sample rate.
- add_noise: Augment data using noise injection.
- fade_in_and_out: Add a fade in and out effect to the audio file.
- normalize_decibel: Normalize the signal using peak or RMS normalization.
"""
import os

import numpy as np

from .io import read_signal_from_wav, write_signal_to_wav


def apply_gain(infile: str, gain: float = 10, inplace: bool = True) -> None:
    """Apply gain to the audio file.

    Args:
        infile: Input audio file path
        gain: Gain in dB (positive or negative)
        inplace: Whether to overwrite the input file
    """
    # Read and process audio
    fs, x = read_signal_from_wav(audio_path=infile)
    x = x * (10 ** (gain / 20.0))
    x = np.clip(x, -1.0, 1.0)  # Using clip instead of minimum/maximum

    suffix = ".wav" if inplace else f"_augmented_with_{gain}_gain.wav"
    outfile = os.path.splitext(infile)[0] + suffix
    write_signal_to_wav(sig=x, fs=fs, output_path=outfile)


def add_noise(infile: str, snr: float) -> None:
    """Augment data using noise injection.

    Note:
        Adds random values to the input file data based on the snr.

    Args:
        infile: Input audio file path
        snr: Signal-to-noise ratio in dB
    """
    fs, sig = read_signal_from_wav(audio_path=infile)
    noise = np.random.randn(len(sig))

    # Compute signal and noise power
    noise_power = np.mean(np.power(noise, 2))
    sig_power = np.mean(np.power(sig, 2))

    # Compute the scaling factor
    snr_linear = 10 ** (snr / 10.0)
    noise_factor = (sig_power / noise_power) * (1 / snr_linear)

    # Add noise
    y = sig + np.sqrt(noise_factor) * noise

    outfile = os.path.splitext(infile)[0] + f"_augmented_{snr}_noisy.wav"
    write_signal_to_wav(sig=y, fs=fs, output_path=outfile)


def fade_in_and_out(infile: str) -> None:
    """Add a fade in and out effect to the audio file.

    Args:
        infile: Input audio file path
    """
    fs, sig = read_signal_from_wav(audio_path=infile)

    # Apply fade effect
    window = np.hamming(len(sig))
    augmented_sig = window * sig
    augmented_sig /= np.mean(np.abs(augmented_sig))

    outfile = os.path.splitext(infile)[0] + "_augmented_fade_in_out.wav"
    write_signal_to_wav(sig=augmented_sig, fs=fs, output_path=outfile)


def normalize_decibel(infile: str, normalization_technique: str = "rms",
                      rms_level: float = -20, inplace: bool = True) -> None:
    """Normalize the signal using peak or RMS normalization.

    Args:
        infile: Input audio file path
        normalization_technique: Type of normalization ("peak" or "rms")
        rms_level: RMS level in dB
        inplace: Whether to overwrite the input file
    
    Raises:
        ValueError: If unknown normalization_technique is specified
    """
    fs, sig = read_signal_from_wav(audio_path=infile)

    # Normalize signal
    if normalization_technique == "peak":
        y = sig / np.max(sig)
    elif normalization_technique == "rms":
        r = 10 ** (rms_level / 20.0)
        a = np.sqrt((len(sig) * r ** 2) / np.sum(sig ** 2))
        y = sig * a
    else:
        raise ValueError(f"Unknown normalization_technique: {normalization_technique}")

    suffix = ".wav" if inplace else f"_augmented_{normalization_technique}_normalized.wav"
    outfile = os.path.splitext(infile)[0] + suffix
    write_signal_to_wav(sig=y, fs=fs, output_path=outfile)
