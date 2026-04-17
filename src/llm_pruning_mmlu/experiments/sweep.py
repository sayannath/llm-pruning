from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from llm_pruning_mmlu.config import ExperimentConfig, load_config_dict, load_model_config
from llm_pruning_mmlu.data.mmlu import load_mmlu
from llm_pruning_mmlu.evaluation.runner import evaluate_examples
from llm_pruning_mmlu.experiments.artifacts import save_run_artifacts
from llm_pruning_mmlu.experiments.resume import should_skip, sparsity_dir
from llm_pruning_mmlu.models.loader import load_model_and_tokenizer
from llm_pruning_mmlu.pruning.apply import apply_masks
from llm_pruning_mmlu.pruning.magnitude import compute_global_magnitude_masks
from llm_pruning_mmlu.pruning.stats import pruning_stats
from llm_pruning_mmlu.pruning.targets import find_pruning_parameters
from llm_pruning_mmlu.reporting.plots import plot_sparsity_vs_accuracy
from llm_pruning_mmlu.reporting.tables import write_combined_results
from llm_pruning_mmlu.utils.emissions import EmissionsTracker
from llm_pruning_mmlu.utils.hashing import stable_hash
from llm_pruning_mmlu.utils.io import ensure_dir, write_json
from llm_pruning_mmlu.utils.logging_utils import configure_logging
from llm_pruning_mmlu.utils.seed import set_seed
from llm_pruning_mmlu.utils.versioning import runtime_metadata

_log = logging.getLogger("llm_pruning_mmlu")


def run_id_for_config(config_dict: dict[str, Any]) -> str:
    return f"mmlu_pruning_{stable_hash(config_dict)}"


def _annotate_per_prompt_emissions(
    predictions: list[dict[str, Any]],
    emissions: dict[str, Any] | None,
) -> None:
    """Attach proportional emissions to each prediction based on elapsed_s."""
    if not emissions or not emissions.get("emissions_kg_co2") or not predictions:
        return
    total_time = sum(p.get("elapsed_s", 0.0) for p in predictions)
    total_kg = emissions["emissions_kg_co2"]
    for pred in predictions:
        frac = pred.get("elapsed_s", 0.0) / total_time if total_time > 0 else 1.0 / len(predictions)
        pred["emissions_kg_co2"] = total_kg * frac


def _model_configs(config: ExperimentConfig):
    if config.model is not None:
        return [config.model]
    return [load_model_config(path) for path in config.models]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_sweep(
    config: ExperimentConfig, config_dict: dict[str, Any], fail_fast: bool = False
) -> Path:
    set_seed(config.seed)
    config_hash = stable_hash(config_dict)
    run_dir = ensure_dir(Path(config.output_root) / f"mmlu_pruning_{config_hash}")
    configure_logging(level=config.log_level)
    _log.info("Run directory: %s", run_dir)

    metadata = runtime_metadata()
    examples = load_mmlu(
        hf_id=config.dataset.hf_id,
        split=config.evaluation.split or config.dataset.split,
        subjects=config.evaluation.subjects,
        max_samples=config.evaluation.max_samples,
        fixture_path=config.dataset.fixture_path,
    )
    _log.info("Loaded %d evaluation examples", len(examples))

    manifest: dict[str, Any] = {
        "run_name": run_dir.name,
        "config_hash": config_hash,
        "start_time": _utcnow(),
        "end_time": None,
        "metadata": metadata,
        "completed": [],
        "failed": [],
        "skipped": [],
        "run_dir": str(run_dir),
    }

    results = []
    for model_cfg in _model_configs(config):
        for sparsity in config.pruning.sparsities:
            out_dir = sparsity_dir(run_dir, model_cfg.name, sparsity)
            if should_skip(run_dir, model_cfg.name, sparsity, config.resume):
                _log.info("Skipping %s sparsity=%s (already complete)", model_cfg.name, sparsity)
                manifest["skipped"].append({"model": model_cfg.name, "sparsity": sparsity})
                continue

            run_logger = configure_logging(
                level=config.log_level,
                log_file=ensure_dir(out_dir) / "run.log",
            )
            run_logger.info("Starting %s sparsity=%s", model_cfg.name, sparsity)
            model = None
            try:
                with EmissionsTracker() as tracker:
                    model, tokenizer = load_model_and_tokenizer(model_cfg, config.device)
                    targets = find_pruning_parameters(
                        model,
                        target_module_types=config.pruning.target_module_types,
                        target_parameter_names=config.pruning.target_parameter_names,
                        exclude_module_name_patterns=config.pruning.exclude_module_name_patterns,
                        prune_bias=config.pruning.prune_bias,
                    )
                    if float(sparsity) > 0:
                        masks = compute_global_magnitude_masks(targets, float(sparsity))
                        apply_masks(targets, masks)
                    stats = pruning_stats(targets)
                    run_logger.info(
                        "Pruning done: achieved_sparsity=%.4f%% total_params=%d nonzero=%d",
                        stats["sparsity"],
                        stats["total"],
                        stats["nonzero"],
                    )
                    metrics, predictions = evaluate_examples(
                        model, tokenizer, examples, config.dataset.answer_choices
                    )

                emissions = tracker.result
                _annotate_per_prompt_emissions(predictions, emissions)

                run_logger.info(
                    "Evaluation done: accuracy=%.4f num_samples=%d emissions_kg_co2=%s",
                    metrics["accuracy"],
                    metrics["num_samples"],
                    emissions["emissions_kg_co2"] if emissions else "n/a",
                )
                metrics.update(
                    {
                        "model_name": model_cfg.name,
                        "model_hf_id": model_cfg.hf_id,
                        "dataset": config.dataset.hf_id,
                        "split": config.evaluation.split,
                        "sparsity_requested": sparsity,
                        "sparsity_achieved": stats["sparsity"],
                        "num_total_target_params": stats["total"],
                        "num_nonzero_target_params": stats["nonzero"],
                        "seed": config.seed,
                        **metadata,
                    }
                )
                if emissions:
                    metrics["emissions_kg_co2"] = emissions["emissions_kg_co2"]
                save_run_artifacts(
                    out_dir, config_dict, metrics, stats, predictions,
                    config.save_predictions, emissions,
                )
                results.append(metrics)
                manifest["completed"].append({"model": model_cfg.name, "sparsity": sparsity})
            except Exception as exc:
                run_logger.exception("Failed %s sparsity=%s: %s", model_cfg.name, sparsity, exc)
                manifest["failed"].append(
                    {"model": model_cfg.name, "sparsity": sparsity, "error": repr(exc)}
                )
                if fail_fast:
                    raise
            finally:
                if model is not None:
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    manifest["end_time"] = _utcnow()
    write_json(run_dir / "manifest.json", manifest)
    rows = write_combined_results(run_dir)
    if rows and config.reporting.save_plot:
        plot_sparsity_vs_accuracy(
            run_dir / "combined_results.csv",
            run_dir / "plots" / "sparsity_vs_accuracy.png",
        )
    _log.info(
        "Sweep complete: %d completed, %d skipped, %d failed",
        len(manifest["completed"]),
        len(manifest["skipped"]),
        len(manifest["failed"]),
    )
    return run_dir


def run_sweep_from_config_path(path: str | Path, fail_fast: bool = False) -> Path:
    from llm_pruning_mmlu.config import load_config

    data = load_config_dict(path)
    return run_sweep(load_config(path), data, fail_fast=fail_fast)
