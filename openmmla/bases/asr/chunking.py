"""Where a chunk that reached its cap is cut.

A group- or wearer-scope base hears no change of speaker, so its chunks end at a silent segment or
at the cap (max_chunk_duration, 30 s by default). Cut right at the cap, a chunk splits whatever is
said at that moment between two transcriptions. Instead it is cut at the quietest moment of its
last QUIET_CUT_SECONDS before the cap: the middle of the QUIET_FRAME_SECONDS of lowest level. The
part before goes for transcription, the part after starts the next chunk. Only audio the base
already has is read, so the cut waits for nothing and falls in the same place live and in replay;
the part after it is transcribed with the next chunk, so its words arrive up to one cap later than
a cut at the cap would deliver them. The base cuts so only with a cap of at least two segments
(ASRBase._quiet_cuts), which keeps each rest shorter than the cap.
"""
from __future__ import annotations

import numpy as np

QUIET_CUT_SECONDS = 10.0    # how far before the cap a capped chunk may be cut
QUIET_FRAME_SECONDS = 0.1   # the stretch whose level is compared
QUIET_HOP_SECONDS = 0.01    # how far apart the compared stretches start


def quiet_cut(frames: bytes, framerate: int, cap: float, window: float = QUIET_CUT_SECONDS,
              frame: float = QUIET_FRAME_SECONDS, hop: float = QUIET_HOP_SECONDS) -> int | None:
    """the sample at which a chunk of 16-bit mono audio that reached its cap is cut, counted from
    the chunk's first sample: the middle of its quietest `frame` seconds (lowest mean square) that
    lie wholly between `window` seconds before the cap (never earlier than half the cap) and the
    cap, the latest of equally quiet ones. When that stretch ends the chunk, the chunk is cut at its
    end (its number of samples), so the next one does not start with a sliver of it. None when the
    chunk holds too little audio for one such stretch there, and the caller cuts at the cap."""
    if not framerate or framerate <= 0 or not cap or cap <= 0:
        return None
    samples = np.frombuffer(frames[:len(frames) // 2 * 2], dtype=np.int16)
    width = int(round(frame * framerate))
    step = max(1, int(round(hop * framerate)))
    hi = min(len(samples), int(round(cap * framerate)))
    lo = int(round(max(cap - window, cap / 2) * framerate))
    if width <= 0 or hi - lo < width:
        return None
    x = samples[lo:hi].astype(np.float64)
    power = np.concatenate(([0.0], np.cumsum(x * x)))
    # every `step` samples, and the stretch that ends at the cap
    starts = np.unique(np.append(np.arange(0, hi - lo - width + 1, step), hi - lo - width))
    energy = power[starts + width] - power[starts]
    latest = len(energy) - 1 - int(np.argmin(energy[::-1]))
    start = lo + int(starts[latest])
    if start + width == len(samples):
        return len(samples)
    return start + width // 2
