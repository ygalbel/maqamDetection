"""Peak detection on a PCD + alignment of detected peaks to template degrees,
used to explain *why* the classifier picked a particular maqam."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .histogram import BINS
from .templates import MAQAMAT

# Common Arabic/Western interval names by cents. Tolerance used when matching.
INTERVAL_NAMES: list[tuple[int, str]] = [
    (0, "unison"),
    (100, "minor 2nd"),
    (150, "half-flat 2 (sikah)"),
    (200, "major 2nd"),
    (300, "minor 3rd"),
    (350, "half-flat 3 (rast)"),
    (400, "major 3rd"),
    (498, "perfect 4th"),
    (600, "tritone"),
    (702, "perfect 5th"),
    (800, "minor 6th"),
    (850, "half-flat 6"),
    (900, "major 6th"),
    (1000, "minor 7th"),
    (1050, "half-flat 7"),
    (1100, "major 7th"),
]


def interval_label(cents: int, tol: int = 30) -> str:
    cents = cents % BINS
    best_cents, best_name = min(INTERVAL_NAMES, key=lambda iv: abs(iv[0] - cents))
    if abs(best_cents - cents) <= tol:
        return best_name
    return f"unknown ({cents}c)"


def detected_peaks(pcd: np.ndarray, max_peaks: int = 12,
                   min_height_ratio: float = 0.15, min_distance_cents: int = 25
                   ) -> list[tuple[int, float]]:
    """Top peaks in the PCD. Returns [(cents_bin, height), ...] sorted by height desc."""
    if pcd.max() <= 0:
        return []
    threshold = pcd.max() * min_height_ratio
    peaks_idx, _ = find_peaks(pcd, height=threshold, distance=min_distance_cents)
    if peaks_idx.size == 0:
        return []
    heights = pcd[peaks_idx]
    order = np.argsort(-heights)
    return [(int(peaks_idx[i]), float(heights[i])) for i in order[:max_peaks]]


def align_to_template(peaks: list[tuple[int, float]], maqam: str,
                      tonic_shift_cents: int, tol_cents: int = 35
                      ) -> tuple[list[tuple[int, float, int, float, int]],
                                 list[tuple[int, float]]]:
    """For each template degree, find the closest detected peak (if any).

    Returns:
      aligned: list of (cents_from_tonic, template_weight, matched_peak_bin,
                        matched_peak_height, distance_cents)
      missing: list of (cents_from_tonic, template_weight) for unmatched degrees
    """
    degrees = MAQAMAT[maqam]
    aligned = []
    missing = []
    for cents_rel, weight in degrees:
        target_bin = (int(round(cents_rel)) + tonic_shift_cents) % BINS
        best = None
        for peak_bin, height in peaks:
            dist = abs(peak_bin - target_bin)
            dist = min(dist, BINS - dist)
            if best is None or dist < best[2]:
                best = (peak_bin, height, dist)
        if best and best[2] <= tol_cents:
            aligned.append((int(cents_rel), float(weight), best[0], best[1], best[2]))
        else:
            missing.append((int(cents_rel), float(weight)))
    return aligned, missing
