from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

BINS = 1200

# Each maqam: list of (cents_from_tonic, weight). Tonic + 5th heaviest;
# characteristic quartertone degrees emphasized to discriminate from neighbors.
# Values approximate common-practice Arabic tuning; tune empirically.
MAQAMAT: dict[str, list[tuple[float, float]]] = {
    "Rast":     [(0, 3.0), (204, 1.0), (350, 2.0), (498, 1.5), (702, 2.5), (906, 1.0), (1052, 1.5)],
    "Bayati":   [(0, 3.0), (150, 2.0), (294, 1.0), (498, 1.5), (702, 2.5), (800, 1.0), (996, 1.0)],
    "Hijaz":    [(0, 3.0), (100, 1.5), (400, 2.0), (498, 1.5), (702, 2.5), (800, 1.0), (1000, 1.0)],
    "Saba":     [(0, 3.0), (150, 2.0), (294, 1.5), (400, 2.0), (702, 1.5), (800, 1.0), (996, 1.0)],
    "Kurd":     [(0, 3.0), (100, 2.0), (294, 1.0), (498, 1.5), (702, 2.5), (800, 1.0), (996, 1.0)],
    "Nahawand": [(0, 3.0), (200, 1.0), (300, 2.0), (500, 1.5), (700, 2.5), (800, 1.5), (1000, 1.0)],
    "Ajam":     [(0, 3.0), (200, 1.0), (400, 2.0), (500, 1.5), (700, 2.5), (900, 1.0), (1100, 1.5)],
    "Sikah":    [(0, 3.0), (150, 1.5), (348, 1.5), (498, 1.0), (702, 2.0), (852, 1.0), (1052, 2.5)],
}


def build_template(degrees: list[tuple[float, float]], sigma_cents: float = 10.0) -> np.ndarray:
    hist = np.zeros(BINS, dtype=float)
    for cents, weight in degrees:
        hist[int(round(cents)) % BINS] += weight
    smoothed = gaussian_filter1d(hist, sigma=sigma_cents, mode="wrap")
    total = smoothed.sum()
    return smoothed / total if total > 0 else smoothed


def all_templates(sigma_cents: float = 10.0) -> dict[str, np.ndarray]:
    return {name: build_template(degrees, sigma_cents) for name, degrees in MAQAMAT.items()}
