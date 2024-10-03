import os

import numpy as np
from scipy.io.wavfile import read, write


def read_file(filename):
    """Read wave file as mono.

    Args:
        filename (str) : wave file / path.

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


def write_file(output_file_path, input_file_name, name_attribute, sig, fs):
    """Write wave file as mono.

    Args:
        output_file_path (str) : path to save resulting wave file to.
        input_file_name  (str) : name of processed wave file,
        name_attribute   (str) : attribute to add to output file name.
        sig            (array) : signal/audio array.
        fs               (int) : sampling rate.

    Outputs:
        Save wave file to output_file_path.
    """
    # Convert back to int16 if necessary (will decrease accuracy)
    if sig.dtype == np.float64:
        sig = np.int16(sig * 32767)

    # Set up the output file name
    file_name = os.path.basename(input_file_name).split(".wav")[0] + name_attribute
    file_path = os.path.join(output_file_path, file_name)
    write(filename=file_path, rate=fs, data=sig)
    # print("Writing data to " + file_path + ".")
