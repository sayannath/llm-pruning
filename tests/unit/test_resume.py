from __future__ import annotations

from llm_pruning_mmlu.experiments.resume import metrics_complete, should_skip
from llm_pruning_mmlu.utils.io import write_json


def test_resume_requires_metrics_schema(tmp_path):
    path = tmp_path / "m" / "sparsity_050" / "metrics.json"
    write_json(path, {"accuracy": 0.5})
    assert not metrics_complete(path)
    write_json(path, {"accuracy": 0.5, "num_samples": 2, "sparsity_requested": 50})
    assert metrics_complete(path)
    assert should_skip(tmp_path, "m", 50, True)
    assert not should_skip(tmp_path, "m", 50, False)
