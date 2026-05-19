from __future__ import annotations

import concurrent.futures
import urllib.request
from pathlib import Path

from maqam_detect import (
    classify,
    extract_pitch,
    last_note_tonic_seed,
    pitch_class_distribution,
)

DATASET_URL = "https://raw.githubusercontent.com/ygalbel/MakamTest/master/music"

# Dataset folder name → our maqam template name
NAME_MAP: dict[str, str] = {
    "Ajam": "Ajam",
    "Bayat": "Bayati",
    "Hjaz": "Hijaz",
    "Saba": "Saba",
    "Siga": "Sikah",
    "nahwand": "Nahawand",
    "rast": "Rast",
}

INVENTORY: dict[str, int] = {
    "Ajam": 27,
    "Bayat": 33,
    "Hjaz": 21,
    "Saba": 20,
    "Siga": 23,
    "nahwand": 33,
    "rast": 27,
}


def _filename(folder: str, i: int) -> str:
    if folder == "rast":
        return f"Rast_{i}.mp3"
    if folder == "nahwand":
        return f"Nahwand_{i}.mp3"
    return f"{folder}_{i}.mp3"


def clip_list() -> list[tuple[str, str, str]]:
    """Returns [(url, relative_path, true_maqam), ...]."""
    out = []
    for folder, n in INVENTORY.items():
        label = NAME_MAP[folder]
        for i in range(1, n + 1):
            fname = _filename(folder, i)
            out.append((f"{DATASET_URL}/{folder}/{fname}", f"{folder}/{fname}", label))
    return out


def _download_one(args: tuple[str, Path]) -> str:
    url, dest = args
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return "downloaded"


def ensure_dataset(cache_dir: Path, workers: int = 8) -> dict[str, str]:
    """Download (if needed) all dataset clips. Returns {abs_path: true_maqam}."""
    truth: dict[str, str] = {}
    download_args: list[tuple[str, Path]] = []
    for url, rel, label in clip_list():
        dest = cache_dir / rel
        download_args.append((url, dest))
        truth[str(dest)] = label
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_download_one, download_args))
    return truth


def detect_one(path: str) -> tuple[str, str, float, float]:
    """Pickle-safe worker: returns (path, predicted_maqam, top_score, gap)."""
    try:
        track = extract_pitch(path)
        pcd = pitch_class_distribution(track)
        seed = last_note_tonic_seed(track)
        seed_cents = seed[0] if seed else None
        matches = classify(pcd, tonic_seed_cents=seed_cents)
        top = matches[0]
        gap = top.score - matches[1].score
        return path, top.maqam, top.score, gap
    except Exception as e:
        return path, f"ERROR:{type(e).__name__}", 0.0, 0.0


def detect_all(paths: list[str], workers: int = 4) -> dict[str, tuple[str, float, float]]:
    """Run detector in parallel across all clips. Returns {path: (predicted, score, gap)}."""
    results: dict[str, tuple[str, float, float]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for path, predicted, score, gap in ex.map(detect_one, paths):
            results[path] = (predicted, score, gap)
    return results
