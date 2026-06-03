"""time-window utilities for OpenMMLA-GD exports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeWindow:
    index: int
    start: float
    end: float


def build_windows(
    start_time: float,
    end_time: float,
    window_size: float = 30.0,
    step_size: float = 15.0,
) -> list[TimeWindow]:
    """build fixed-size sliding windows over a session interval."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time.")

    windows: list[TimeWindow] = []
    current = float(start_time)
    session_end = float(end_time)
    index = 0

    while current < session_end:
        windows.append(TimeWindow(index=index, start=current, end=current + window_size))
        current += step_size
        index += 1

    return windows
