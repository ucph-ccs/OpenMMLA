"""This module contains utility functions for time-based audio data augmentation. Imported and modified based on pydiogment.

- eliminate_silence: Eliminate silence from the voice file using ffmpeg library.
- random_cropping: Crop the audio file randomly with minimum duration.
- slow_down: Slow or stretch a wave.
- speed: Speed or shrink a wave.
- shift_time: Shift audio in time.
- reverse: Reverse the audio signal.
- resample_audio: Resample the signal according a new input sampling rate with respect to the Nyquist-Shannon theorem.
"""
import math
import os
import random
import subprocess
import warnings

import numpy as np

from .io import read_signal_from_wav, write_signal_to_wav


def eliminate_silence(infile):
    """Eliminate silence from the voice file using ffmpeg library.

    Args:
        infile (str): Path to get the original voice file from.

    Returns:
        list including True for successful authentication, False otherwise and
        a percentage value representing the certainty of the decision.
    """
    outfile = infile.split(".wav")[0] + "_augmented_without_silence.wav"

    # Filter silence in wav
    remove_silence_command = ["ffmpeg", "-i", infile,
                              "-af",
                              "silenceremove=stop_periods=-1:stop_duration=0.25:stop_threshold=-36dB",
                              "-acodec", "pcm_s16le",
                              "-ac", "1", outfile]
    out = subprocess.Popen(remove_silence_command,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    out.wait()

    with_silence_duration = os.popen(
        "ffprobe -i '" + infile +
        "' -show_format -v quiet | sed -n 's/duration=//p'").read()
    no_silence_duration = os.popen(
        "ffprobe -i '" + outfile +
        "' -show_format -v quiet | sed -n 's/duration=//p'").read()
    return with_silence_duration, no_silence_duration


def random_cropping(infile: str, min_len: float = 1) -> None:
    """Crop the audio file randomly with minimum duration.

    Args:
        infile: Input audio file path
        min_len: Minimum duration in seconds
    """
    fs, x = read_signal_from_wav(audio_path=infile)
    t_end = x.size / fs

    if t_end > min_len:
        start = random.uniform(0.0, t_end - min_len)
        end = random.uniform(start + min_len, t_end)
        y = x[int(math.floor(start * fs)):int(math.ceil(end * fs))]

        outfile = os.path.splitext(infile)[0] + f"_augmented_randomly_cropped_{min_len}.wav"
        write_signal_to_wav(sig=y, fs=fs, output_path=outfile)
    else:
        warnings.warn("min_len provided is greater than the duration of the song.")


def slow_down(infile, coefficient=0.8):
    """Slow or stretch a wave.

    Args:
        infile (str): Input filename.
        coefficient (float): coefficient caracterising the slowing degree.
    """
    # Set up variables for paths and file names
    outfile = infile.split(".wav")[0] + "_augmented_slowed.wav"

    # Apply slowing command
    slowing_command = ["ffmpeg", "-i", infile, "-filter:a",
                       "atempo={0}".format(str(coefficient)),
                       outfile]
    print(" ".join(slowing_command))
    p = subprocess.Popen(slowing_command,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, error = p.communicate()
    print(output, error.decode("utf-8"))

    # For i in error.decode("utf-8") : print(i)
    print("Writing data to " + outfile + ".")


def speed(infile, coefficient=1.25):
    """Speed or shrink a wave.

    Args:
        infile (str): Input filename.
        coefficient (float): coefficient caracterising the speeding degree.
    """
    outfile = infile.split(".wav")[0] + "_augmented_speeded.wav"

    # Apply slowing command
    speeding_command = ["ffmpeg", "-i", infile, "-filter:a",
                        "atempo={0}".format(str(coefficient)),
                        outfile]
    _ = subprocess.Popen(speeding_command,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    print("Writing data to " + outfile + ".")


def shift_time(infile: str, tshift: int, direction: str) -> None:
    """Shift audio in time.

    Args:
        infile: Input audio file path
        tshift: Time shift in seconds
        direction: Shift direction ("left" or "right")
    """
    fs, sig = read_signal_from_wav(audio_path=infile)
    shift = int(tshift * fs) * int(direction == "left") - \
            int(tshift * fs) * int(direction == "right")

    augmented_sig = np.roll(sig, shift)
    outfile = os.path.splitext(infile)[0] + f"_augmented_{direction}_{tshift}_shifted.wav"
    write_signal_to_wav(sig=augmented_sig, fs=fs, output_path=outfile)


def reverse(infile: str) -> None:
    """Reverse the audio signal.

    Args:
        infile: Input audio file path
    """
    fs, sig = read_signal_from_wav(audio_path=infile)
    augmented_sig = sig[::-1]

    outfile = os.path.splitext(infile)[0] + "_augmented_reversed.wav"
    write_signal_to_wav(sig=augmented_sig, fs=fs, output_path=outfile)


def resample_audio(infile, sr):
    """Resample the signal according a new input sampling rate with respect to the
    Nyquist-Shannon theorem.

    Args:
        infile (str): input filename/path.
        sr (int): new sampling rate.
    """
    outfile = "{0}_augmented_resampled_to_{1}.wav".format(infile.split(".wav")[0],
                                                          sr)
    sampling_command = ["ffmpeg", "-i", infile, "-ar", str(sr), outfile]
    print(" ".join(sampling_command))
    _ = subprocess.Popen(sampling_command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
