from __future__ import annotations

import numpy as np

from .histogram import BINS, hz_to_cents
from .pitch import PitchTrack


def last_note_tonic_seed(
    track: PitchTrack,
    tail_seconds: float = 3.0,
    min_confidence: float = 0.5,
) -> tuple[float, float] | None:
    """Median pitch of the final voiced segment. Returns (cents_mod_1200, hz)."""
    mask = track.voiced & (track.confidence >= min_confidence) & np.isfinite(track.f0_hz)
    if not mask.any():
        return None
    last_time = track.times[mask][-1]
    tail_mask = mask & (track.times >= last_time - tail_seconds)
    f0 = track.f0_hz[tail_mask]
    if f0.size == 0:
        return None
    median_hz = float(np.median(f0))
    cents_mod = float(hz_to_cents(np.array([median_hz]))[0]) % BINS
    return cents_mod, median_hz
