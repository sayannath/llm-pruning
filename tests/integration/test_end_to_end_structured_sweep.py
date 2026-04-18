from __future__ import annotations

from llm_pruning_mmlu.config import load_config_dict, parse_config
from llm_pruning_mmlu.experiments.sweep import run_sweep

_TAG = "global_magnitude_structured__mlp_channel"


def test_end_to_end_structured_sweep(tmp_path):
    data = load_config_dict("tests/fixtures/dummy_structured_config.yaml")
    data["output_root"] = str(tmp_path)
    run_dir = run_sweep(parse_config(data), data, fail_fast=True)

    # Artifacts land under the pruning tag subdirectory.
    sparsity_000 = run_dir / "dummy_mlp" / _TAG / "sparsity_000"
    sparsity_050 = run_dir / "dummy_mlp" / _TAG / "sparsity_050"
    assert sparsity_000.exists(), f"Missing sparsity_000 dir: {sparsity_000}"
    assert sparsity_050.exists(), f"Missing sparsity_050 dir: {sparsity_050}"

    assert (sparsity_000 / "metrics.json").exists()
    assert (sparsity_050 / "metrics.json").exists()
    assert (sparsity_000 / "pruning_stats.json").exists()
    assert (sparsity_050 / "pruning_stats.json").exists()

    # Combined CSV must exist and contain both rows.
    combined = run_dir / "combined_results.csv"
    assert combined.exists()
    lines = combined.read_text().strip().splitlines()
    assert len(lines) == 3, f"Expected header + 2 data rows, got {len(lines)} lines"

    # Structured-specific fields must be present in metrics.
    import json
    m0 = json.loads((sparsity_000 / "metrics.json").read_text())
    m50 = json.loads((sparsity_050 / "metrics.json").read_text())
    assert m0.get("pruning_method") == "global_magnitude_structured"
    assert m0.get("pruning_structure") == "mlp_channel"
    assert m50.get("num_groups_total") is not None
    assert m50.get("num_groups_pruned", 0) > 0

    # Pruning stats must carry group fields for the sparsity=50 run.
    s50 = json.loads((sparsity_050 / "pruning_stats.json").read_text())
    assert "num_groups_total" in s50
    assert "num_groups_pruned" in s50
    assert s50["group_sparsity"] > 0.0

    # Manifest must record the run as completed with no failures.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert len(manifest["failed"]) == 0
    assert len(manifest["completed"]) == 2


def test_structured_sweep_does_not_overwrite_unstructured_artifacts(tmp_path):
    """Structured and unstructured runs to the same output_root land in separate dirs."""
    from llm_pruning_mmlu.config import load_config_dict, parse_config
    from llm_pruning_mmlu.experiments.sweep import run_sweep

    unstructured_data = load_config_dict("tests/fixtures/dummy_config.yaml")
    unstructured_data["output_root"] = str(tmp_path)
    unstructured_run_dir = run_sweep(parse_config(unstructured_data), unstructured_data, fail_fast=True)

    structured_data = load_config_dict("tests/fixtures/dummy_structured_config.yaml")
    structured_data["output_root"] = str(tmp_path)
    structured_run_dir = run_sweep(parse_config(structured_data), structured_data, fail_fast=True)

    # Each run gets its own directory (different config hash).
    assert unstructured_run_dir != structured_run_dir

    # Unstructured uses flat layout, structured uses tagged layout.
    assert (unstructured_run_dir / "dummy" / "sparsity_000" / "metrics.json").exists()
    assert (structured_run_dir / "dummy_mlp" / _TAG / "sparsity_000" / "metrics.json").exists()

    # No cross-contamination: structured tag dir doesn't exist in unstructured run.
    assert not (unstructured_run_dir / "dummy" / _TAG).exists()
