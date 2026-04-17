from __future__ import annotations

from llm_pruning_mmlu.evaluation.metrics import accuracy, compute_metrics


def test_accuracy():
    assert accuracy(["A", "B", "C"], ["A", "C", "C"]) == 2 / 3


def test_compute_metrics_by_subject():
    metrics = compute_metrics(
        [
            {"subject": "x", "correct": 1},
            {"subject": "x", "correct": 0},
            {"subject": "y", "correct": 1},
        ]
    )
    assert metrics["accuracy"] == 2 / 3
    assert metrics["per_subject_accuracy"]["x"] == 0.5
