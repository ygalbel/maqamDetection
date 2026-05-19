from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .dataset import TEMPLATE_ORDER

pytestmark = pytest.mark.integration

# Families present in the user's labels (Kurd template exists but is never a true label)
FAMILIES = ["Ajam", "Bayati", "Hijaz", "Nahawand", "Rast", "Saba", "Sikah"]


def _featurize(features: dict, truth: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, paths). Drops clips that errored during extraction."""
    rows, labels, paths = [], [], []
    for path, feat in features.items():
        if "scores" not in feat:
            continue
        scores = [feat["scores"][m] for m in TEMPLATE_ORDER]
        gap = feat["gap"]
        voiced = feat["voiced_ratio"]
        rows.append(scores + [gap, voiced])
        labels.append(truth[path])
        paths.append(path)
    return np.array(rows), np.array(labels), paths


def _rule_proba(feat: dict, temperature: float = 50.0) -> np.ndarray:
    """Rule-based score → family probability via softmax over non-Kurd templates."""
    family_scores = np.array([feat["scores"][f] for f in FAMILIES])
    family_scores = family_scores - family_scores.max()
    e = np.exp(family_scores * temperature)
    return e / e.sum()


def _rule_top_family(feat: dict) -> str:
    return max(FAMILIES, key=lambda f: feat["scores"][f])


def test_calibration_cv(features, truth, capsys):
    X, y, paths = _featurize(features, truth)
    print(f"\n[calibration] {len(X)} clips, {X.shape[1]} features, classes={sorted(set(y))}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rule_acc, ml_acc, ens_acc = [], [], []
    ens_confusion: Counter = Counter()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            max_iter=2000,
            C=0.5,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X_tr, y_tr)

        ml_pred = clf.predict(X_te)
        ml_acc.append((ml_pred == y_te).mean())

        ml_proba = clf.predict_proba(X_te)
        # Re-index ML proba into FAMILIES order
        ml_proba_aligned = np.zeros((len(test_idx), len(FAMILIES)))
        class_to_col = {c: k for k, c in enumerate(clf.classes_)}
        for j, fam in enumerate(FAMILIES):
            if fam in class_to_col:
                ml_proba_aligned[:, j] = ml_proba[:, class_to_col[fam]]

        rule_preds = [_rule_top_family(features[paths[i]]) for i in test_idx]
        rule_probs = np.array([_rule_proba(features[paths[i]]) for i in test_idx])
        rule_acc.append(np.mean([rp == yt for rp, yt in zip(rule_preds, y_te)]))

        ens_proba = (ml_proba_aligned + rule_probs) / 2.0
        ens_pred = np.array([FAMILIES[k] for k in ens_proba.argmax(axis=1)])
        ens_acc.append((ens_pred == y_te).mean())
        for yt, ep in zip(y_te, ens_pred):
            ens_confusion[(yt, ep)] += 1

    print(f"\n=== 5-fold stratified CV ===")
    print(f"Rule-based: {np.mean(rule_acc):.1%} ± {np.std(rule_acc):.1%}  (per-fold: {[f'{a:.0%}' for a in rule_acc]})")
    print(f"ML only:    {np.mean(ml_acc):.1%} ± {np.std(ml_acc):.1%}  (per-fold: {[f'{a:.0%}' for a in ml_acc]})")
    print(f"Ensemble:   {np.mean(ens_acc):.1%} ± {np.std(ens_acc):.1%}  (per-fold: {[f'{a:.0%}' for a in ens_acc]})")

    print(f"\nEnsemble confusion (rows=truth, cols=predicted):")
    print(" " * 12 + "  ".join(f"{f[:6]:>6}" for f in FAMILIES))
    for yt in FAMILIES:
        row = "  ".join(f"{ens_confusion.get((yt, ep), 0):>6}" for ep in FAMILIES)
        print(f"  {yt:<10}  {row}")

    with capsys.disabled():
        pass

    assert np.mean(ens_acc) >= np.mean(rule_acc) - 0.02, (
        f"Ensemble ({np.mean(ens_acc):.1%}) is meaningfully worse than rule-only ({np.mean(rule_acc):.1%})"
    )
    assert np.mean(ml_acc) > 1 / len(FAMILIES), (
        f"ML accuracy {np.mean(ml_acc):.1%} is at-or-below chance ({1/len(FAMILIES):.1%})"
    )
