from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from .dataset import detect_all, ensure_dataset


@pytest.fixture(scope="session")
def dataset_cache(pytestconfig) -> Path:
    cache = Path(os.environ.get("MAQAM_DATASET_DIR", pytestconfig.rootpath / "tests" / "data" / "MakamTest"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture(scope="session")
def truth(dataset_cache: Path) -> dict[str, str]:
    t0 = time.time()
    out = ensure_dataset(dataset_cache)
    print(f"\n[dataset] {len(out)} clips ready in {time.time() - t0:.1f}s ({dataset_cache})")
    return out


@pytest.fixture(scope="session")
def predictions(truth: dict[str, str]) -> dict[str, tuple[str, float, float]]:
    workers = int(os.environ.get("MAQAM_WORKERS", "4"))
    paths = sorted(truth)
    t0 = time.time()
    print(f"[detect] running pYIN + classify on {len(paths)} clips ({workers} workers)...")
    out = detect_all(paths, workers=workers)
    print(f"[detect] done in {time.time() - t0:.0f}s")
    return out
