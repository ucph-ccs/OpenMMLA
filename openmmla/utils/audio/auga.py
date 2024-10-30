"""
- Description: amplitude based augmentation techniques/manipulations for audio data.
Imported and modified based on pydiogment
"""
import os

import numpy as np

from .files import read_file, write_file


def apply_gain(infile, gain=40, inplace=True):
    """
    Apply gain to infile.

    Args:
        inplace (bool): whether change in place or not.
        infile (str): input filename/path.
        gain (float): gain in dB (both positive and negative).
    """
    # Read an input file
    fs, x = read_file(filename=infile)
    # Apply gain
    x = np.copy(x)
    x = x * (10 ** (gain / 20.0))
    x = np.minimum(np.maximum(-1.0, x), 1.0)
    # x /= np.mean(np.abs(x)) # it will amplified the signal too much if the mean value is too low

    # Export data to file
    output_dir = os.path.dirname(infile)
    name_attribute = ".wav"
    if not inplace:
        name_attribute = "_augmented_with_%s_gain.wav" % str(gain)

    write_file(output_dir=output_dir, input_filename=infile, name_attribute=name_attribute, sig=x, fs=fs)


def add_noise(infile, snr):
    """
    Augment data using noise injection.

    Note:
        It simply adds some random values to the input file data based on the snr.

    Args:
        infile (str): input filename/path.
        snr (int): signal-to-noise ratio in dB.
    """
    # Read an input file
    fs, sig = read_file(filename=infile)

    # Compute and apply noise
    noise = np.random.randn(len(sig))

    # Compute powers
    noise_power = np.mean(np.power(noise, 2))
    sig_power = np.mean(np.power(sig, 2))

    # Compute snr and scaling factor
    snr_linear = 10 ** (snr / 10.0)
    noise_factor = (sig_power / noise_power) * (1 / snr_linear)

    # Add noise
    y = sig + np.sqrt(noise_factor) * noise

    # Construct file names
    output_dir = os.path.dirname(infile)
    name_attribute = "_augmented_%s_noisy.wav" % snr

    # Export data to file
    write_file(output_dir=output_dir, input_filename=infile, name_attribute=name_attribute, sig=y, fs=fs)


def fade_in_and_out(infile):
    """
    Add a fade in and out effect to the audio file.

    Args:
        infile (str): input filename/path.
    """
    # Read an input file
    fs, sig = read_file(filename=infile)

    # Construct file names
    output_dir = os.path.dirname(infile)
    name_attribute = "_augmented_fade_in_out.wav"

    # Fade in and out
    window = np.hamming(len(sig))
    augmented_sig = window * sig
    augmented_sig /= np.mean(np.abs(augmented_sig))

    # Export data to file
    write_file(output_dir=output_dir, input_filename=infile, name_attribute=name_attribute, sig=augmented_sig, fs=fs)


def normalize_rms(infile, normalization_technique="rms", rms_level=-20, inplace=True):
    """
    Normalize the signal given a certain technique (peak or rms).

    Args:
        inplace (bool): whether change inplace or not
        infile (str): input filename/path.
        normalization_technique (str): type of normalization technique to use. (default is peak)
        rms_level (int): rms level in dB.
    """
    # Read an input file
    fs, sig = read_file(filename=infile)

    # Normalize signal
    if normalization_technique == "peak":
        y = sig / np.max(sig)
    elif normalization_technique == "rms":
        # Linear rms level and scaling factor
        r = 10 ** (rms_level / 20.0)
        a = np.sqrt((len(sig) * r ** 2) / np.sum(sig ** 2))
        # Normalize
        y = sig * a
    else:
        raise Exception("ParameterError: Unknown normalization_technique variable.")

    # Construct file names
    output_dir = os.path.dirname(infile)
    name_attribute = ".wav"
    if not inplace:
        name_attribute = "_augmented_{}_normalized.wav".format(normalization_technique)

    # Export data to file
    write_file(output_dir=output_dir, input_filename=infile, name_attribute=name_attribute, sig=y, fs=fs)
