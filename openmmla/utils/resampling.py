from enum import Enum
from typing import List

import librosa
import numpy as np
from scipy import signal

from openmmla.utils.audio.files import int16_to_float32, float32_to_int16


class ResampleMethod(Enum):
    """Resampling methods for different data types."""
    # Audio methods
    AUDIO_LINEAR = 'audio_linear'
    AUDIO_POLYPHASE = 'audio_polyphase'
    AUDIO_FFT = 'audio_fft'
    AUDIO_LANCZOS = 'audio_lanczos'
    AUDIO_LIBROSA = 'audio_librosa'  # Using librosa's implementation

    # Video methods
    VIDEO_SKIP = 'video_skip'  # Skip frames for downsampling
    VIDEO_INTERPOLATE = 'video_interpolate'  # Interpolate frames for upsampling
    VIDEO_AVERAGE = 'video_average'  # Average neighboring frames

    # General signal methods
    SIGNAL_LINEAR = 'signal_linear'
    SIGNAL_CUBIC = 'signal_cubic'
    SIGNAL_NEAREST = 'signal_nearest'


def resample_audio(data: np.ndarray,
                   source_rate: int,
                   target_rate: int,
                   method: ResampleMethod = ResampleMethod.AUDIO_POLYPHASE,
                   **kwargs) -> np.ndarray:
    """Resample audio data."""
    if source_rate == target_rate:
        return data

    if isinstance(method, str):
        method = ResampleMethod(method)

    if method == ResampleMethod.AUDIO_LINEAR:
        return _audio_linear_resample(data, source_rate, target_rate)
    elif method == ResampleMethod.AUDIO_POLYPHASE:
        return _audio_polyphase_resample(data, source_rate, target_rate)
    elif method == ResampleMethod.AUDIO_FFT:
        return _audio_fft_resample(data, source_rate, target_rate)
    elif method == ResampleMethod.AUDIO_LANCZOS:
        return _audio_lanczos_resample(data, source_rate, target_rate, **kwargs)
    elif method == ResampleMethod.AUDIO_LIBROSA:
        return _audio_librosa_resample(data, source_rate, target_rate)
    else:
        raise ValueError(f"Unsupported resampling method: {method}")


def resample_video(frames: List[np.ndarray],
                   source_fps: float,
                   target_fps: float,
                   method: ResampleMethod = ResampleMethod.VIDEO_SKIP,
                   **kwargs) -> List[np.ndarray]:
    """Main video resampling function."""
    if source_fps == target_fps:
        return frames

    if isinstance(method, str):
        method = ResampleMethod(method)

    if method == ResampleMethod.VIDEO_SKIP:
        return _video_skip_resample(frames, source_fps, target_fps)
    elif method == ResampleMethod.VIDEO_AVERAGE:
        return _video_average_resample(frames, source_fps, target_fps)
    elif method == ResampleMethod.VIDEO_INTERPOLATE:
        return _video_interpolate_resample(frames, source_fps, target_fps)
    else:
        raise ValueError(f"Unsupported video resampling method: {method}")


def resample_signal(data: np.ndarray,
                    source_rate: float,
                    target_rate: float,
                    method: ResampleMethod = ResampleMethod.SIGNAL_LINEAR,
                    **kwargs) -> np.ndarray:
    """Main signal resampling function."""
    if source_rate == target_rate:
        return data

    if isinstance(method, str):
        method = ResampleMethod(method)

    if method == ResampleMethod.SIGNAL_LINEAR:
        return _signal_linear_resample(data, source_rate, target_rate)
    elif method == ResampleMethod.SIGNAL_CUBIC:
        return _signal_cubic_resample(data, source_rate, target_rate)
    elif method == ResampleMethod.SIGNAL_NEAREST:
        return _signal_nearest_resample(data, source_rate, target_rate)
    else:
        raise ValueError(f"Unsupported signal resampling method: {method}")


def _audio_linear_resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Simple linear interpolation resampling for audio.
    
    Uses straight-line interpolation between samples.
    Best for: Quick prototyping or resource-limited systems.
    Pros: Fast, computationally efficient, minimal memory usage
    Cons: Can introduce aliasing, poor quality for high frequencies
    
    Args:
        data: Input audio data (int16 or float32 in [-1.0, 1.0] range)
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        
    Returns:
        np.ndarray: Resampled audio data as int16
    """
    duration = len(data) / source_rate
    target_length = int(duration * target_rate)
    indices = np.linspace(0, len(data) - 1, target_length)
    return np.interp(indices, np.arange(len(data)), data).astype(np.int16)


def _audio_polyphase_resample(data: np.ndarray,
                              source_rate: int,
                              target_rate: int) -> np.ndarray:
    """Polyphase filter bank resampling for audio.
    
    Uses polyphase decomposition for efficient sample rate conversion.
    Best for: General-purpose audio resampling and production use.
    Pros: Good quality-to-performance ratio, memory efficient
    Cons: Slight edge effects possible
    
    Args:
        data: Input audio data (int16 or float32 in [-1.0, 1.0] range)
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        
    Returns:
        np.ndarray: Resampled audio data in same format as input (int16 or float32)
    """
    # Convert to float32
    if data.dtype == np.int16:
        float_data = int16_to_float32(data)
    else:
        float_data = data

    # Calculate resampling parameters
    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd

    # Apply polyphase resampling
    resampled = signal.resample_poly(float_data, up, down, window=('kaiser', 5.0))

    if data.dtype == np.int16:
        resampled = float32_to_int16(resampled)

    return resampled


def _audio_fft_resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """FFT-based resampling for audio.
    
    Performs resampling in frequency domain using FFT.
    Best for: High-quality offline processing.
    Pros: Very high quality, perfect for bandlimited signals
    Cons: Computationally intensive, higher memory usage
    
    Args:
        data: Input audio data (int16 or float32 in [-1.0, 1.0] range)
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        
    Returns:
        np.ndarray: Resampled audio data as int16
    """
    target_length = int(len(data) * target_rate / source_rate)
    resampled = signal.resample(data, target_length)
    return resampled.astype(np.int16)


def _audio_lanczos_resample(data: np.ndarray,
                            source_rate: int,
                            target_rate: int,
                            **kwargs) -> np.ndarray:
    """Lanczos resampling for audio.
    
    Uses Lanczos (windowed sinc) interpolation.
    Best for: Highest quality requirements and professional audio.
    Pros: Excellent quality, good high frequency preservation
    Cons: Most computationally intensive, complex implementation
    
    Args:
        data: Input audio data (int16 or float32 in [-1.0, 1.0] range)
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        a: Size of the Lanczos window (larger = higher quality but slower)
        
    Returns:
        np.ndarray: Resampled audio data as int16
    """
    duration = len(data) / source_rate
    target_length = int(duration * target_rate)
    a = kwargs.get('a', 3)

    def lanczos_kernel(x: float, a: int) -> float:
        if x == 0:
            return 1
        if abs(x) >= a:
            return 0
        x_pi = np.pi * x
        return a * np.sin(x_pi) * np.sin(x_pi / a) / (x_pi * x_pi)

    indices = np.linspace(0, len(data) - 1, target_length)
    resampled = np.zeros(target_length)

    for i, idx in enumerate(indices):
        left = max(0, int(idx - a))
        right = min(len(data), int(idx + a + 1))
        x = np.arange(left, right) - idx
        weights = np.array([lanczos_kernel(xi, a) for xi in x])
        resampled[i] = np.sum(data[left:right] * weights) / np.sum(weights)

    return resampled.astype(np.int16)


def _audio_librosa_resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Librosa-based resampling for audio.
    
    Uses librosa's high-quality resampling implementation based on polyphase filters.
    Best for: Professional audio processing where quality is critical.
    Pros: High quality, well-tested in production environments
    Cons: Slower than simpler methods, requires librosa dependency
    
    Args:
        data: Input audio data (int16 or float32 in [-1.0, 1.0] range)
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        
    Returns:
        np.ndarray: Resampled audio data in same format as input (int16 or float32)
    """
    if data.dtype == np.int16:
        float_data = int16_to_float32(data)
    else:
        float_data = data

    resampled = librosa.resample(float_data,
                                 orig_sr=source_rate,
                                 target_sr=target_rate)

    if data.dtype == np.int16:
        resampled = float32_to_int16(resampled)

    return resampled


def _video_skip_resample(frames: List[np.ndarray], source_fps: float, target_fps: float) -> List[np.ndarray]:
    """Frame skipping method for video downsampling.
    
    Simply picks frames at regular intervals based on the fps ratio.
    Best for: Quick downsampling where some frame loss is acceptable.
    Pros: Fast, memory-efficient
    Cons: Can cause jerky motion, loses information
    
    Args:
        frames: List of video frames as numpy arrays
        source_fps: Original frames per second
        target_fps: Target frames per second
        
    Returns:
        List[np.ndarray]: Resampled frames at target fps
    """
    ratio = source_fps / target_fps
    indices = [int(i * ratio) for i in range(int(len(frames) / ratio))]
    return [frames[i] for i in indices if i < len(frames)]


def _video_average_resample(frames: List[np.ndarray], source_fps: float, target_fps: float) -> List[np.ndarray]:
    """Frame averaging method for video downsampling.
    
    Averages multiple consecutive frames to create each output frame.
    Best for: Smooth downsampling where motion blur is acceptable.
    Pros: Smoother motion, preserves more information
    Cons: Can cause motion blur, more computationally intensive
    
    Args:
        frames: List of video frames as numpy arrays
        source_fps: Original frames per second
        target_fps: Target frames per second
        
    Returns:
        List[np.ndarray]: Resampled frames at target fps
    """
    ratio = source_fps / target_fps
    new_frames = []
    for i in range(0, len(frames), int(ratio)):
        end_idx = min(i + int(ratio), len(frames))
        avg_frame = np.mean(frames[i:end_idx], axis=0)
        new_frames.append(avg_frame.astype(frames[0].dtype))
    return new_frames


def _video_interpolate_resample(frames: List[np.ndarray], source_fps: float, target_fps: float) -> List[np.ndarray]:
    """Linear interpolation method for video upsampling.
    
    Creates new frames by linearly interpolating between existing frames.
    Best for: Simple upsampling where perfect quality isn't required.
    Pros: Simple to understand, reasonable results for small fps increases
    Cons: Can look artificial, doesn't handle complex motion well
    
    Args:
        frames: List of video frames as numpy arrays
        source_fps: Original frames per second
        target_fps: Target frames per second (must be higher than source_fps)
        
    Returns:
        List[np.ndarray]: Resampled frames at target fps
        
    Raises:
        ValueError: If source_fps > target_fps (downsampling not supported)
    """
    if source_fps > target_fps:
        raise ValueError("Interpolation method is only for upsampling")

    ratio = target_fps / source_fps
    new_frames = []
    for i in range(len(frames) - 1):
        new_frames.append(frames[i])
        for j in range(1, int(ratio)):
            alpha = j / ratio
            interp_frame = (1 - alpha) * frames[i] + alpha * frames[i + 1]
            new_frames.append(interp_frame.astype(frames[0].dtype))
    new_frames.append(frames[-1])
    return new_frames


def _signal_linear_resample(data: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """Linear interpolation for signal resampling.
    
    Uses linear interpolation between points.
    Best for: Simple signals without sharp transitions.
    Pros: Fast, good for smooth signals
    Cons: Can miss high-frequency details
    
    Args:
        data: Input signal data (any numeric type)
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        
    Returns:
        np.ndarray: Resampled signal data in same type as input
    """
    duration = len(data) / source_rate
    target_length = int(duration * target_rate)
    x = np.linspace(0, len(data) - 1, len(data))
    x_new = np.linspace(0, len(data) - 1, target_length)
    return np.interp(x_new, x, data)


def _signal_cubic_resample(data: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """Cubic spline interpolation for signal resampling.
    
    Uses cubic splines to create smooth curves between points.
    Best for: Complex signals where smoothness is important.
    Pros: Smooth results, better preservation of curves
    Cons: More computationally intensive, can overshoot
    
    Args:
        data: Input signal data (any numeric type)
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        
    Returns:
        np.ndarray: Resampled signal data in same type as input
    """
    from scipy.interpolate import CubicSpline
    duration = len(data) / source_rate
    target_length = int(duration * target_rate)
    x = np.linspace(0, duration, len(data))
    cs = CubicSpline(x, data)
    x_new = np.linspace(0, duration, target_length)
    return cs(x_new)


def _signal_nearest_resample(data: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """Nearest neighbor interpolation for signal resampling.
    
    Uses the value of the nearest sample point.
    Best for: Step-like signals or when preserving exact values is important.
    Pros: Preserves original values, no interpolation artifacts
    Cons: Can create jagged results
    
    Args:
        data: Input signal data (any numeric type)
        source_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        
    Returns:
        np.ndarray: Resampled signal data in same type as input
    """
    duration = len(data) / source_rate
    target_length = int(duration * target_rate)
    x = np.linspace(0, len(data) - 1, len(data))
    x_new = np.linspace(0, len(data) - 1, target_length)
    indices = np.searchsorted(x, x_new)
    indices = np.clip(indices, 0, len(data) - 1)
    return data[indices]
