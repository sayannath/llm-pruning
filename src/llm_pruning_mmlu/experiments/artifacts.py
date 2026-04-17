from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_pruning_mmlu.config import dump_resolved_config
from llm_pruning_mmlu.evaluation.predictions import save_predictions
from llm_pruning_mmlu.utils.io import ensure_dir, write_json


def save_run_artifacts(
    run_path: str | Path,
    resolved_config: dict[str, Any],
    metrics: dict[str, Any],
    pruning_stats: dict[str, Any],
    predictions: list[dict[str, Any]],
    save_predictions_flag: bool = True,
    emissions: dict[str, Any] | None = None,
) -> None:
    run_path = ensure_dir(run_path)
    dump_resolved_config(resolved_config, run_path / "config_resolved.yaml")
    write_json(run_path / "metrics.json", metrics)
    write_json(run_path / "pruning_stats.json", pruning_stats)
    if emissions is not None:
        write_json(run_path / "emissions.json", emissions)
    if save_predictions_flag:
        save_predictions(run_path / "predictions.jsonl", predictions)
