from __future__ import annotations

from pathlib import Path

from llm_pruning_mmlu.utils.io import read_json


def sparsity_dir(
    run_dir: str | Path,
    model_name: str,
    sparsity: float,
    pruning_tag: str | None = None,
) -> Path:
    """Return the output directory for one (model, sparsity) run.

    pruning_tag is None for unstructured runs (preserves existing paths) and
    a non-empty string like "global_magnitude_structured__mlp_channel" for
    structured runs, inserted as an intermediate directory segment so
    structured and unstructured artifacts never share a path.
    """
    base = Path(run_dir) / model_name
    if pruning_tag:
        base = base / pruning_tag
    return base / f"sparsity_{int(sparsity):03d}"


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


def should_skip(
    run_dir: str | Path,
    model_name: str,
    sparsity: float,
    resume: bool,
    pruning_tag: str | None = None,
) -> bool:
    if not resume:
        return False
    return metrics_complete(
        sparsity_dir(run_dir, model_name, sparsity, pruning_tag) / "metrics.json"
    )
