import os
import shutil
import subprocess
import tempfile
import wave
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from pydub import AudioSegment

from .auga import normalize_decibel


def resample_audio_file(file_path: str, target_sr: int = 8000) -> None:
    """
    Resamples the audio data from the original sample rate to the target sample rate, and overwrites the original file
    with the resampled audio data.

    Args:
        file_path (str): The path to the audio file to be resampled.
        target_sr (int): The target sample rate for the audio data.
    """
    audio_data, original_sr = librosa.load(file_path, sr=None)  # sr=None ensures original SR is used
    resampled_audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
    sf.write(file_path, resampled_audio_data, target_sr)


def format_wav(input_file: str, output_file: str = None, codec: str = "pcm_s16le",
               sample_rate: int = 16000, channels: int = 1) -> str:
    """
    Formats an audio file to .wav using ffmpeg with specified parameters.
    Checks if the input file already matches the desired format before processing.

    Args:
        input_file (str): Input audio file path.
        output_file (str, optional): Output audio file path. If None, formats in-place.
        codec (str, optional): Audio codec to use. Defaults to "pcm_s16le".
        sample_rate (int, optional): Sample rate of the output audio. Defaults to 16000.
        channels (int, optional): Number of audio channels. Defaults to 1 (mono).

    Returns:
        str: Path of the output WAV file.

    Raises:
        Exception: If ffmpeg is not installed or not added to PATH.
    """
    if shutil.which("ffmpeg") is None:
        raise Exception("ffmpeg is not found. Please install it and add it to PATH.")

    # Determine the expected sample width based on the codec
    codec_to_sample_width = {
        "pcm_s16le": 2,
        "pcm_s24le": 3,
        "pcm_s32le": 4,
        # Add more mappings as needed
    }
    expected_sample_width = codec_to_sample_width.get(codec, 2)  # Default to 2 if codec is not recognized

    # Check if the input file is already in the desired format
    if input_file.lower().endswith('.wav'):
        try:
            with wave.open(input_file, 'rb') as wav_file:
                if (wav_file.getframerate() == sample_rate and
                        wav_file.getnchannels() == channels and
                        wav_file.getsampwidth() == expected_sample_width):
                    print(f"File {input_file} is already in the desired format. Skipping conversion.")
                    if output_file is not None:
                        shutil.copy(input_file, output_file)
                        return output_file
                    return input_file
        except wave.Error:
            # If there's an error reading the WAV file, we'll proceed with conversion
            pass

    # Create a temporary file for the conversion
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_output = temp_file.name

    # Run the ffmpeg command
    command = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-acodec", codec,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        temp_output
    ]
    subprocess.run(command, check=True)

    if output_file is None:
        # In-place formatting
        output_file = os.path.splitext(input_file)[0] + ".wav"
        os.remove(input_file)
        shutil.move(temp_output, output_file)
    else:
        # Output to a different file
        shutil.move(temp_output, output_file)

    return output_file


def crop_and_concatenate_wav(input_file: str, clip_ranges: List[Tuple[int, int]], output_file: str) -> None:
    """
    Crops segments from an audio file and concatenates them into a new .wav file.

    Args:
        input_file (str): Input .wav audio file path.
        clip_ranges (List[Tuple[int, int]]): List of tuples, where each tuple contains the start and end times (in ms) of a segment to crop.
        output_file (str): Output .wav audio file path.

    Raises:
        ValueError: If a clip range is invalid.
    """
    audio = AudioSegment.from_wav(input_file)
    concatenated_audio = AudioSegment.empty()

    for start, end in clip_ranges:
        if start < 0 or end < start or end > len(audio):
            raise ValueError(f"Invalid clip range: {start}-{end}")

        clip = audio[start:end]
        concatenated_audio += clip

    concatenated_audio.export(output_file, format="wav")


def segment_wav(input_file: str, output_dir: str, step_length_ms=None, window_length_ms=None) -> None:
    """
    Segments an audio file and exports each segment as a new .wav file.

    Args:
        input_file (str): Input .wav audio file path.
        output_dir (str): The output directory path.
        step_length_ms (int): The step length (overlap) between segments in milliseconds.
        window_length_ms (int): The duration of each segment in milliseconds.
    """
    audio = AudioSegment.from_wav(input_file)
    if window_length_ms is None:
        window_length_ms = len(audio)
    if step_length_ms is None:
        step_length_ms = window_length_ms

    num_segments = (len(audio) - window_length_ms) // step_length_ms + 1

    if os.path.exists(output_dir):
        response = input(f"Output directory {output_dir} already exists. Do you want to delete it? (Y/N): ")
        if response.upper() == "Y":
            shutil.rmtree(output_dir)
            print("Existing directory deleted.")
            os.makedirs(output_dir)
        else:
            response = input("Do you want to append to the existing directory? (Y/N), No will append nothing.: ")
            if response.upper() == "N":
                return
    else:
        os.makedirs(output_dir)

    existing_segments = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    for i in range(num_segments):
        start_time = i * step_length_ms
        end_time = start_time + window_length_ms
        if end_time > len(audio):
            break
        segment = audio[start_time:end_time]
        output_file = os.path.join(output_dir, f"segment_{existing_segments + i}.wav")
        segment.export(output_file, format="wav")
        normalize_decibel(output_file)


def get_energy_level(audio_file_path: str, verbose: bool = True) -> (float, float):
    """
    Calculate the volume level of audio file

    Args:
        verbose: Print out message or not
        audio_file_path(str): Path of audio file

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
    """
    Get properties of an audio file.

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


def read_frames_from_wav(audio_path):
    """
    Read the audio file as a byte string.

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
    """
    Write audio frames to a wave file.
    This function supports writing a single bytes object or a collection (list or tuple) of bytes objects to a wave file.

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


def calculate_rms_db(audio_samples: np.ndarray) -> float:
    """
    Calculate the root mean square (RMS) energy of the audio data in decibels.

    Args:
        audio_samples (np.ndarray): Numpy array of audio samples.

    Returns:
        float: Root mean square (RMS) energy of the audio data in decibels.
    """
    mean_square = np.mean(audio_samples ** 2)
    return 10 * np.log10(mean_square)


def calculate_audio_duration(audio_path: str):
    """
    Calculate the duration of the audio file

    Args:
        audio_path (str): the path of the audio

    Returns:
        float: the duration of the audio file in seconds
    """
    with sf.SoundFile(audio_path) as f:
        return len(f) / f.samplerate
