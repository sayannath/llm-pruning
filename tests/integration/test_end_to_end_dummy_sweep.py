from __future__ import annotations

from llm_pruning_mmlu.config import load_config_dict, parse_config
from llm_pruning_mmlu.experiments.sweep import run_sweep


def test_end_to_end_dummy_sweep(tmp_path):
    data = load_config_dict("tests/fixtures/dummy_config.yaml")
    data["output_root"] = str(tmp_path)
    run_dir = run_sweep(parse_config(data), data, fail_fast=True)
    assert (run_dir / "dummy" / "sparsity_000" / "metrics.json").exists()
    assert (run_dir / "dummy" / "sparsity_050" / "metrics.json").exists()
    assert (run_dir / "combined_results.csv").exists()
    assert (run_dir / "combined_results.jsonl").exists()
