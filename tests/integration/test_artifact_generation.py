from __future__ import annotations

from llm_pruning_mmlu.experiments.artifacts import save_run_artifacts


def test_artifact_files_created(tmp_path):
    save_run_artifacts(
        tmp_path,
        {"seed": 1},
        {"accuracy": 1.0, "num_samples": 1, "sparsity_requested": 0},
        {"total": 1, "nonzero": 1},
        [{"gold": "A", "pred": "A", "correct": 1}],
    )
    assert (tmp_path / "config_resolved.yaml").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "pruning_stats.json").exists()
    assert (tmp_path / "predictions.jsonl").exists()
