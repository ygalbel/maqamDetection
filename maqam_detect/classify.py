from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .histogram import BINS
from .templates import all_templates


@dataclass
class Match:
    maqam: str
    score: float
    tonic_shift_cents: int  # bin in input where the tonic sits (0..1199)


def classify(
    pcd: np.ndarray,
    tonic_seed_cents: float | None = None,
    seed_tolerance_cents: int = 60,
) -> list[Match]:
    """Bhattacharyya-coefficient classification at every circular shift.

    BC(P, Q_shift) = Σᵢ √(P[i] · Q[(i - shift) mod N]), bounded in [0, 1].
    Identical in form to cross-correlation of √P and √Q, so FFT-friendly.
    Peak-aware: a flat template can't earn a high score against a peaked
    PCD because √(small mass at peak position) doesn't accumulate enough.
    """
    sqrt_pcd_fft = np.fft.fft(np.sqrt(np.maximum(pcd, 0)))
    if tonic_seed_cents is not None:
        seed_bin = int(round(tonic_seed_cents)) % BINS
        shifts = np.arange(BINS)
        dist = np.minimum(np.abs(shifts - seed_bin), BINS - np.abs(shifts - seed_bin))
        seed_mask = dist <= seed_tolerance_cents
    else:
        seed_mask = None

    results: list[Match] = []
    for name, tpl in all_templates().items():
        sqrt_tpl_fft = np.fft.fft(np.sqrt(np.maximum(tpl, 0)))
        bc = np.real(np.fft.ifft(sqrt_pcd_fft * np.conj(sqrt_tpl_fft)))
        if seed_mask is not None:
            constrained = np.where(seed_mask, bc, -np.inf)
            best_shift = int(np.argmax(constrained))
        else:
            best_shift = int(np.argmax(bc))
        results.append(Match(name, float(bc[best_shift]), best_shift))
    results.sort(key=lambda m: -m.score)
    return results
