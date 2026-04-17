from __future__ import annotations

from pathlib import Path

from llm_pruning_mmlu.utils.io import read_json


def sparsity_dir(run_dir: str | Path, model_name: str, sparsity: float) -> Path:
    return Path(run_dir) / model_name / f"sparsity_{int(sparsity):03d}"


def metrics_complete(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except Exception:
        return False
    required = {"accuracy", "num_samples", "sparsity_requested"}
    return required.issubset(data)


def should_skip(run_dir: str | Path, model_name: str, sparsity: float, resume: bool) -> bool:
    if not resume:
        return False
    return metrics_complete(sparsity_dir(run_dir, model_name, sparsity) / "metrics.json")
