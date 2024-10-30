import os

import numpy as np
from scipy.io.wavfile import read, write


def read_file(filename):
    """Read a wave file as mono.

    Args:
        filename (str): string path or open file handle.

    Returns:
        tuple of sampling rate and audio data.
    """
    fr, sig = read(filename=filename)
    if sig.ndim == 1:
        samples = sig
    else:
        samples = sig[:, 0]
    if sig.dtype == np.int16:
        # print("This is 16-bit PCM audio. Normalize by dividing by 32767.0.")
        samples = samples / 32767.0
    return fr, samples


def write_file(output_dir, input_filename, name_attribute, sig, fs):
    """Write processed signal to a wave file as mono.

    Args:
        output_dir (str): directory to save the resulting wave file.
        input_filename (str): original path of the input signal.
        name_attribute (str): attribute to add to output file name.
        sig (array): signal/audio array.
        fs (int): sampling rate.

    Outputs:
        Save a wave file to output_file_path.
    """
    # Convert back to int16 if necessary (will decrease accuracy)
    if sig.dtype == np.float64:
        sig = np.int16(sig * 32767)

    # Set up the output file name
    output_basename = os.path.basename(input_filename).split(".wav")[0] + name_attribute
    output_filename = os.path.join(output_dir, output_basename)
    write(filename=output_filename, rate=fs, data=sig)
    # print("Writing data to " + output_filename + ".")
