from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .pitch import PitchTrack

BINS = 1200
REFERENCE_HZ = 440.0


def hz_to_cents(hz: np.ndarray, ref: float = REFERENCE_HZ) -> np.ndarray:
    return 1200.0 * np.log2(hz / ref)


def pitch_class_distribution(
    track: PitchTrack,
    sigma_cents: float = 10.0,
    min_confidence: float = 0.5,
) -> np.ndarray:
    mask = track.voiced & (track.confidence >= min_confidence) & np.isfinite(track.f0_hz)
    if not mask.any():
        return np.zeros(BINS, dtype=float)
    cents = hz_to_cents(track.f0_hz[mask]) % BINS
    bins = np.floor(cents).astype(int) % BINS
    hist = np.bincount(bins, weights=track.confidence[mask], minlength=BINS).astype(float)
    smoothed = gaussian_filter1d(hist, sigma=sigma_cents, mode="wrap")
    total = smoothed.sum()
    return smoothed / total if total > 0 else smoothed
