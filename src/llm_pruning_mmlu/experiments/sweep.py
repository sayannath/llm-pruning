from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from llm_pruning_mmlu.config import ExperimentConfig, PruningConfig, load_config_dict, load_model_config
from llm_pruning_mmlu.data.mmlu import load_mmlu
from llm_pruning_mmlu.evaluation.runner import evaluate_examples
from llm_pruning_mmlu.experiments.artifacts import save_run_artifacts
from llm_pruning_mmlu.experiments.resume import should_skip, sparsity_dir
from llm_pruning_mmlu.models.loader import load_model_and_tokenizer
from llm_pruning_mmlu.pruning.dispatch import prune_model
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


def _pruning_tag(pruning_config: PruningConfig) -> str | None:
    """Return a filesystem-safe tag that distinguishes run types.

    Unstructured     → None (preserves existing directory layout exactly).
    Semi-structured  → "{method}__{n}_{m}" e.g. global_magnitude_semi_structured__2_4
    Structured       → "{method}__{structure}"
    """
    if pruning_config.method == "global_magnitude_unstructured":
        return None
    if pruning_config.method == "global_magnitude_semi_structured":
        return f"{pruning_config.method}__{pruning_config.nm_n}_{pruning_config.nm_m}"
    structure = pruning_config.structure or "unknown"
    return f"{pruning_config.method}__{structure}"


def _annotate_per_prompt_emissions(
    predictions: list[dict[str, Any]],
    emissions: dict[str, Any] | None,
) -> None:
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

    tag = _pruning_tag(config.pruning)
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
        "pruning_method": config.pruning.method,
        "pruning_structure": config.pruning.structure,
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
            out_dir = sparsity_dir(run_dir, model_cfg.name, sparsity, tag)
            if should_skip(run_dir, model_cfg.name, sparsity, config.resume, tag):
                _log.info("Skipping %s sparsity=%s (already complete)", model_cfg.name, sparsity)
                manifest["skipped"].append({"model": model_cfg.name, "sparsity": sparsity})
                continue

            run_logger = configure_logging(
                level=config.log_level,
                log_file=ensure_dir(out_dir) / "run.log",
            )
            run_logger.info("Starting %s sparsity=%s", model_cfg.name, sparsity)
            model = None
            tokenizer = None
            targets = None
            try:
                with EmissionsTracker() as tracker:
                    model, tokenizer = load_model_and_tokenizer(model_cfg, config.device)
                    targets, stats, _ = prune_model(model, config.pruning, float(sparsity))
                    run_logger.info(
                        "Pruning done: method=%s structure=%s achieved_sparsity=%.4f%% "
                        "total_params=%d nonzero=%d groups_total=%s groups_pruned=%s "
                        "nm_n=%s nm_m=%s nm_sparsity=%.2f%%",
                        stats.get("method"),
                        stats.get("structure"),
                        stats["sparsity"],
                        stats["total"],
                        stats["nonzero"],
                        stats.get("num_groups_total", "n/a"),
                        stats.get("num_groups_pruned", "n/a"),
                        stats.get("nm_n", "n/a"),
                        stats.get("nm_m", "n/a"),
                        stats.get("nm_sparsity", 0.0),
                    )
                    metrics, predictions = evaluate_examples(
                        model, tokenizer, examples, config.dataset.answer_choices,
                        scoring_mode=config.evaluation.scoring_mode,
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
                        "pruning_method": stats.get("method", config.pruning.method),
                        "pruning_structure": stats.get("structure", config.pruning.structure),
                        "seed": config.seed,
                        **metadata,
                    }
                )
                # Structured-only fields — present only when groups exist.
                if stats.get("num_groups_total") is not None:
                    metrics.update(
                        {
                            "group_sparsity_requested": sparsity,
                            "group_sparsity_achieved": stats.get("group_sparsity"),
                            "num_groups_total": stats.get("num_groups_total"),
                            "num_groups_pruned": stats.get("num_groups_pruned"),
                        }
                    )
                # Semi-structured N:M fields — present only for semi-structured runs.
                if stats.get("nm_n") is not None:
                    metrics.update(
                        {
                            "nm_n": stats["nm_n"],
                            "nm_m": stats["nm_m"],
                            "nm_pattern": f"{stats['nm_n']}:{stats['nm_m']}",
                            "nm_block_dim": stats["block_dim"],
                            "nm_sparsity_requested": 50.0,
                            "nm_sparsity_achieved": stats["nm_sparsity"],
                            "num_nm_blocks_total": stats["num_blocks_total"],
                            "num_nm_complete_blocks": stats["num_complete_blocks"],
                            "num_nm_remainder_weights": stats["num_remainder_weights"],
                            "parameter_sparsity_achieved": stats["sparsity"],
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
                targets = None
                tokenizer = None
                if model is not None:
                    del model
                    model = None
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
