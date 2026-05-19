from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from maqam_detect.templates import MAQAMAT

from .dataset import ensure_dataset, extract_features_all


def _feature_cache_key() -> str:
    """Hash of template definitions; cached features are invalidated on template change."""
    blob = json.dumps(MAQAMAT, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


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
def features(truth: dict[str, str], dataset_cache: Path) -> dict[str, dict]:
    """Per-clip feature dict (scores, gap, voiced_ratio, ...). Cached to disk by template hash."""
    cache_file = dataset_cache / f"features_{_feature_cache_key()}.json"
    if cache_file.exists():
        print(f"[features] loaded from cache {cache_file.name}")
        return json.loads(cache_file.read_text())

    workers = int(os.environ.get("MAQAM_WORKERS", "4"))
    paths = sorted(truth)
    t0 = time.time()
    print(f"[features] extracting from {len(paths)} clips ({workers} workers)...")
    out = extract_features_all(paths, workers=workers)
    print(f"[features] done in {time.time() - t0:.0f}s; caching to {cache_file.name}")
    cache_file.write_text(json.dumps(out, indent=2))
    return out


@pytest.fixture(scope="session")
def predictions(features: dict[str, dict]) -> dict[str, tuple[str, float, float]]:
    """Back-compat: rule-based top prediction derived from the feature dict."""
    return {
        p: (
            f.get("top", f.get("error", "ERROR:unknown")),
            f.get("top_score", 0.0),
            f.get("gap", 0.0),
        )
        for p, f in features.items()
    }
