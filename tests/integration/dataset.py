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
from maqam_detect.templates import MAQAMAT

DATASET_URL = "https://raw.githubusercontent.com/ygalbel/MakamTest/master/music"

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

# Stable ordering of templates → matches the feature vector order used by the ML calibrator
TEMPLATE_ORDER: list[str] = sorted(MAQAMAT.keys())


def _filename(folder: str, i: int) -> str:
    if folder == "rast":
        return f"Rast_{i}.mp3"
    if folder == "nahwand":
        return f"Nahwand_{i}.mp3"
    return f"{folder}_{i}.mp3"


def clip_list() -> list[tuple[str, str, str]]:
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
    truth: dict[str, str] = {}
    download_args: list[tuple[str, Path]] = []
    for url, rel, label in clip_list():
        dest = cache_dir / rel
        download_args.append((url, dest))
        truth[str(dest)] = label
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_download_one, download_args))
    return truth


def extract_features_one(path: str) -> dict:
    """Pickle-safe worker. Returns full feature dict per clip."""
    try:
        track = extract_pitch(path)
        pcd = pitch_class_distribution(track)
        seed = last_note_tonic_seed(track)
        seed_cents = seed[0] if seed else None
        matches = classify(pcd, tonic_seed_cents=seed_cents)
        score_by_maqam = {m.maqam: m.score for m in matches}
        ranked = sorted(matches, key=lambda m: -m.score)
        return {
            "path": path,
            "scores": score_by_maqam,
            "top": ranked[0].maqam,
            "top_score": ranked[0].score,
            "gap": ranked[0].score - ranked[1].score,
            "voiced_ratio": float(track.voiced.mean()) if track.voiced.size else 0.0,
            "tonic_shift_cents": int(ranked[0].tonic_shift_cents),
        }
    except Exception as e:
        return {"path": path, "error": f"ERROR:{type(e).__name__}"}


def extract_features_all(paths: list[str], workers: int = 4) -> dict[str, dict]:
    results: dict[str, dict] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for feat in ex.map(extract_features_one, paths):
            results[feat["path"]] = feat
    return results
