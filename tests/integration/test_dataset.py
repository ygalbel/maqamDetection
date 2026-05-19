from __future__ import annotations

import statistics
from collections import Counter, defaultdict

import pytest

pytestmark = pytest.mark.integration


def test_dataset_accuracy_report(truth, predictions, capsys):
    """End-to-end accuracy on the full MakamTest dataset.

    Asserts overall accuracy is meaningfully above chance (1/7 ≈ 14%).
    Threshold is intentionally conservative — first job is to establish a baseline
    and confirm the pipeline runs; we ratchet it up as templates/scoring improve.
    """
    per_correct: dict[str, int] = defaultdict(int)
    per_total: dict[str, int] = defaultdict(int)
    confusion: dict[str, Counter] = defaultdict(Counter)
    gaps_correct: list[float] = []
    gaps_wrong: list[float] = []

    for path, label in truth.items():
        predicted, _score, gap = predictions[path]
        per_total[label] += 1
        confusion[label][predicted] += 1
        if predicted == label:
            per_correct[label] += 1
            gaps_correct.append(gap)
        else:
            gaps_wrong.append(gap)

    total = sum(per_total.values())
    correct = sum(per_correct.values())
    accuracy = correct / total

    lines = ["", "=" * 60, f"MakamTest dataset: {correct}/{total} = {accuracy:.1%}", "=" * 60, ""]
    lines.append(f"{'Maqam':<10} {'Acc':>6}  {'N':>4}")
    for label in sorted(per_total):
        acc = per_correct[label] / per_total[label]
        lines.append(f"{label:<10} {acc:>6.0%}  {per_total[label]:>4}")

    preds_seen = sorted({p for c in confusion.values() for p in c})
    lines.append("")
    lines.append("Confusion matrix (rows=truth, cols=predicted):")
    lines.append(" " * 12 + "  ".join(f"{p[:6]:>6}" for p in preds_seen))
    for label in sorted(confusion):
        row = "  ".join(f"{confusion[label].get(p, 0):>6}" for p in preds_seen)
        lines.append(f"  {label:<10}  {row}")

    if gaps_correct:
        lines.append("")
        lines.append("Mean score-gap (top - 2nd):")
        lines.append(f"  correct: {statistics.mean(gaps_correct):.4f}")
        if gaps_wrong:
            lines.append(f"  wrong:   {statistics.mean(gaps_wrong):.4f}")

    report = "\n".join(lines)
    print(report)

    with capsys.disabled():
        pass

    assert accuracy >= 0.25, (
        f"Overall accuracy {accuracy:.1%} below baseline 25%. Report:\n{report}"
    )


@pytest.mark.parametrize("maqam", ["Ajam", "Bayati", "Hijaz", "Saba", "Sikah", "Nahawand", "Rast"])
def test_per_maqam_above_chance(truth, predictions, maqam):
    """Each maqam should be detected better than random chance (1/7 ≈ 14%)."""
    n_correct = sum(
        1 for path, label in truth.items()
        if label == maqam and predictions[path][0] == maqam
    )
    n_total = sum(1 for label in truth.values() if label == maqam)
    assert n_total > 0, f"no clips for {maqam}"
    acc = n_correct / n_total
    assert acc > 1 / 7, f"{maqam}: {n_correct}/{n_total} = {acc:.1%} is at-or-below chance"
