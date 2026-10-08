"""Classification metrics, calibration and latency percentiles. Pure Python so they are easy to test."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any


def accuracy(truth: Sequence[str], pred: Sequence[str]) -> float:
    if not truth:
        return float("nan")
    return sum(t == p for t, p in zip(truth, pred, strict=True)) / len(truth)


def confusion(truth: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """Rows are true labels, columns are predictions. Every label gets a row and column, zeros included."""
    table = {t: dict.fromkeys(labels, 0) for t in labels}
    for t, p in zip(truth, pred, strict=True):
        table.setdefault(t, dict.fromkeys(labels, 0))
        table[t][p] = table[t].get(p, 0) + 1
    return table


def f1_scores(truth: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    scores = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, pred, strict=True))
        denominator = 2 * tp + fp + fn
        scores[label] = 2 * tp / denominator if denominator else 0.0
    return scores


def macro_f1(truth: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Unweighted mean F1 over the classes that appear in the ground truth.

    A class that only ever shows up as a wrong prediction (for example not_visible on a two-class set)
    still costs recall elsewhere, so it is left out of the mean rather than counted as a zero class.
    """
    present = [label for label in (labels or sorted(set(truth))) if label in set(truth)]
    if not present:
        return float("nan")
    scores = f1_scores(truth, pred, present)
    return sum(scores.values()) / len(present)


def reliability_bins(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width bins on [0, 1]. Only non-empty bins are returned."""
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, ok in zip(confidences, correct, strict=True):
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, ok))
    result = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        result.append(
            {
                "lo": index / bins,
                "hi": (index + 1) / bins,
                "count": len(bucket),
                "mean_confidence": sum(c for c, _ in bucket) / len(bucket),
                "accuracy": sum(ok for _, ok in bucket) / len(bucket),
            }
        )
    return result


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    total = len(confidences)
    if not total:
        return float("nan")
    return sum(
        b["count"] / total * abs(b["accuracy"] - b["mean_confidence"])
        for b in reliability_bins(confidences, correct, bins)
    )


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100), the same convention as numpy's default."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def percentiles(values: Sequence[float | None]) -> dict[str, float | None]:
    clean = [v for v in values if v is not None]
    return {name: percentile(clean, q) for name, q in (("p50", 50), ("p90", 90), ("p99", 99))}


def task_summary(
    truth: Sequence[str],
    pred: Sequence[str],
    confidences: Sequence[float],
    labels: Sequence[str],
    latency_ms: Sequence[float | None],
    model_ms: Sequence[float | None],
) -> dict[str, Any]:
    correct = [t == p for t, p in zip(truth, pred, strict=True)]
    return {
        "n": len(truth),
        "labels": list(labels),
        "support": dict(Counter(truth)),
        "accuracy": accuracy(truth, pred),
        "macro_f1": macro_f1(truth, pred, labels),
        "confusion": confusion(truth, pred, labels),
        "ece": expected_calibration_error(confidences, correct),
        "reliability": reliability_bins(confidences, correct),
        "latency_ms": percentiles(latency_ms),
        "model_ms": percentiles(model_ms),
    }
