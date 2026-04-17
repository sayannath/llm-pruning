from __future__ import annotations

from collections import defaultdict
from typing import Any


def accuracy(golds: list[str], preds: list[str]) -> float:
    if len(golds) != len(preds):
        raise ValueError("golds and preds must have same length")
    if not golds:
        return 0.0
    return sum(g == p for g, p in zip(golds, preds, strict=True)) / len(golds)


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(int(row["correct"]) for row in predictions)
    by_subject: dict[str, list[int]] = defaultdict(list)
    for row in predictions:
        by_subject[row["subject"]].append(int(row["correct"]))
    return {
        "metric": "accuracy",
        "accuracy": correct / total if total else 0.0,
        "num_samples": total,
        "num_correct": correct,
        "per_subject_accuracy": {
            subject: sum(values) / len(values) for subject, values in sorted(by_subject.items())
        },
    }
