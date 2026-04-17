from __future__ import annotations

from llm_pruning_mmlu.utils.io import write_jsonl


def save_predictions(path, predictions: list[dict]) -> None:
    write_jsonl(path, predictions)
