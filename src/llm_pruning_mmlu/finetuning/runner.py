from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from llm_pruning_mmlu.config import ExperimentConfig, load_config_dict, load_model_config
from llm_pruning_mmlu.evaluation.runner import evaluate_examples
from llm_pruning_mmlu.experiments.artifacts import save_run_artifacts
from llm_pruning_mmlu.experiments.resume import sparsity_dir
from llm_pruning_mmlu.finetuning.config import FinetuningConfig
from llm_pruning_mmlu.finetuning.datasets import load_sft_dataset
from llm_pruning_mmlu.finetuning.lora import attach_lora
from llm_pruning_mmlu.finetuning.mask_policy import MaskEnforcer
from llm_pruning_mmlu.finetuning.masked_trainer import train_with_masks
from llm_pruning_mmlu.finetuning.wandb_utils import init_wandb, wandb_run_name
from llm_pruning_mmlu.models.loader import load_model_and_tokenizer
from llm_pruning_mmlu.pruning.dispatch import prune_model
from llm_pruning_mmlu.utils.emissions import EmissionsTracker
from llm_pruning_mmlu.utils.hashing import stable_hash
from llm_pruning_mmlu.utils.io import ensure_dir, write_json
from llm_pruning_mmlu.utils.logging_utils import configure_logging
from llm_pruning_mmlu.utils.seed import set_seed
from llm_pruning_mmlu.utils.versioning import runtime_metadata

_log = logging.getLogger("llm_pruning_mmlu")


def sft_run_id(config_dict: dict[str, Any], ft_config_dict: dict[str, Any]) -> str:
    combined = {**config_dict, "_ft": ft_config_dict}
    return f"structured_sft_{stable_hash(combined)}"


def _pruning_tag(method: str, structure: str | None) -> str | None:
    if method == "global_magnitude_unstructured":
        return None
    if structure:
        return f"{method}__{structure}"
    return method


def _model_configs(config: ExperimentConfig):
    if config.model is not None:
        return [config.model]
    return [load_model_config(path) for path in config.models]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_sft_sweep(
    experiment_config: ExperimentConfig,
    experiment_config_dict: dict[str, Any],
    ft_config: FinetuningConfig,
    ft_config_dict: dict[str, Any],
    fail_fast: bool = False,
) -> Path:
    """Run structured sparse fine-tuning for all (model, sparsity) pairs.

    The pruning kind (structured vs semi-structured) is read from
    experiment_config.pruning.method at runtime — switching to semi-structured
    SFT requires only a different experiment config YAML, no code changes here.
    """
    set_seed(experiment_config.seed)
    run_dir = ensure_dir(
        Path(experiment_config.output_root) / sft_run_id(experiment_config_dict, ft_config_dict)
    )
    configure_logging(level=experiment_config.log_level)
    _log.info("SFT run directory: %s", run_dir)

    tag = _pruning_tag(experiment_config.pruning.method, experiment_config.pruning.structure)
    metadata = runtime_metadata()

    eval_examples = _load_eval_examples(experiment_config)
    _log.info("Loaded %d evaluation examples", len(eval_examples))

    manifest: dict[str, Any] = {
        "run_name": run_dir.name,
        "pruning_method": experiment_config.pruning.method,
        "pruning_structure": experiment_config.pruning.structure,
        "finetuning_method": ft_config.method,
        "start_time": _utcnow(),
        "end_time": None,
        "metadata": metadata,
        "completed": [],
        "failed": [],
        # "skipped": [],  # resume/skip logic removed
    }

    for model_cfg in _model_configs(experiment_config):
        for sparsity in experiment_config.pruning.sparsities:
            out_dir = sparsity_dir(run_dir, model_cfg.name, sparsity, tag)
            run_logger = configure_logging(
                level=experiment_config.log_level,
                log_file=ensure_dir(out_dir) / "run.log",
            )
            run_logger.info("Starting SFT %s sparsity=%s", model_cfg.name, sparsity)

            rname = wandb_run_name(model_cfg.name, float(sparsity), ft_config.method)
            wandb_run = init_wandb(
                ft_config.wandb,
                run_name=rname,
                config_dict={
                    "experiment": experiment_config_dict,
                    "finetuning": ft_config_dict,
                    "model": model_cfg.name,
                    "sparsity": sparsity,
                },
            )

            model = None
            parameters = None
            try:
                with EmissionsTracker() as tracker:
                    model, tokenizer = load_model_and_tokenizer(
                        model_cfg, experiment_config.device
                    )

                    parameters, pruning_stats_dict, masks = prune_model(
                        model, experiment_config.pruning, float(sparsity),
                        tokenizer=tokenizer,
                    )
                    run_logger.info(
                        "Pruned: sparsity=%.4f%% groups_pruned=%s",
                        pruning_stats_dict["sparsity"],
                        pruning_stats_dict.get("num_groups_pruned", "n/a"),
                    )

                    if masks is not None:
                        torch.save(masks, out_dir / "masks.pt")

                    enforcer = MaskEnforcer.from_prune_result(
                        parameters=parameters,
                        masks=masks,
                        pruning_method=experiment_config.pruning.method,
                        mask_policy_cfg=ft_config.mask_policy,
                    )

                    model = attach_lora(model, ft_config.lora)

                    train_dataset = load_sft_dataset(
                        ft_config.train_dataset, tokenizer, ft_config.max_seq_length
                    )
                    val_dataset = load_sft_dataset(
                        ft_config.validation_dataset, tokenizer, ft_config.max_seq_length
                    )

                    training_stats = train_with_masks(
                        model=model,
                        tokenizer=tokenizer,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        ft_cfg=ft_config,
                        enforcer=enforcer,
                        output_dir=out_dir,
                        run_name=rname,
                        wandb_run=wandb_run,
                    )
                    run_logger.info("Training done: %s", training_stats)

                    # Save LoRA adapter before evaluation
                    adapter_dir = out_dir / "adapter"
                    model.save_pretrained(str(adapter_dir))
                    tokenizer.save_pretrained(str(adapter_dir))

                    # Re-enforce masks on the merged weights so the evaluation
                    # reflects true structured sparsity
                    enforcer.enforce()

                    metrics, predictions = evaluate_examples(
                        model, tokenizer, eval_examples,
                        experiment_config.dataset.answer_choices,
                        scoring_mode=experiment_config.evaluation.scoring_mode,
                    )

                emissions = tracker.result

                run_logger.info(
                    "Evaluation: accuracy=%.4f samples=%d",
                    metrics["accuracy"],
                    metrics["num_samples"],
                )

                metrics.update(
                    {
                        "model_name": model_cfg.name,
                        "model_hf_id": model_cfg.hf_id,
                        "pruning_method": experiment_config.pruning.method,
                        "pruning_structure": experiment_config.pruning.structure,
                        "sparsity_requested": sparsity,
                        "sparsity_achieved": pruning_stats_dict["sparsity"],
                        "group_sparsity_achieved": pruning_stats_dict.get("group_sparsity"),
                        "finetuning_method": ft_config.method,
                        "train_dataset": ft_config.train_dataset.hf_id,
                        "train_split": ft_config.train_dataset.split,
                        "train_samples": len(train_dataset),
                        "eval_dataset": experiment_config.dataset.hf_id,
                        "eval_split": experiment_config.evaluation.split,
                        "epochs": ft_config.epochs,
                        "learning_rate": ft_config.learning_rate,
                        "lora_r": ft_config.lora.r,
                        "lora_alpha": ft_config.lora.alpha,
                        "mask_policy_preserve_base": ft_config.mask_policy.preserve_base_masks,
                        "mask_policy_lora_channels": ft_config.mask_policy.mask_lora_pruned_channels,
                        "pruning_kind": enforcer.pruning_kind,
                        **training_stats,
                        **metadata,
                    }
                )
                if emissions:
                    metrics["emissions_kg_co2"] = emissions["emissions_kg_co2"]

                save_run_artifacts(
                    out_dir, experiment_config_dict, metrics, pruning_stats_dict,
                    predictions, experiment_config.save_predictions, emissions,
                )
                write_json(out_dir / "training_stats.json", training_stats)

                wandb_run.log_summary({"mmlu_accuracy": metrics["accuracy"]})
                manifest["completed"].append({"model": model_cfg.name, "sparsity": sparsity})

            except Exception as exc:
                run_logger.exception("Failed %s sparsity=%s: %s", model_cfg.name, sparsity, exc)
                manifest["failed"].append(
                    {"model": model_cfg.name, "sparsity": sparsity, "error": repr(exc)}
                )
                if fail_fast:
                    raise
            finally:
                wandb_run.finish()
                parameters = None
                if model is not None:
                    del model
                    model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    manifest["end_time"] = _utcnow()
    write_json(run_dir / "manifest.json", manifest)
    _log.info(
        "SFT sweep complete: %d completed, %d failed",
        len(manifest["completed"]),
        len(manifest["failed"]),
    )
    return run_dir


def _load_eval_examples(config: ExperimentConfig):
    from llm_pruning_mmlu.data.mmlu import load_mmlu

    return load_mmlu(
        hf_id=config.dataset.hf_id,
        split=config.evaluation.split or config.dataset.split,
        subjects=config.evaluation.subjects,
        max_samples=config.evaluation.max_samples,
        fixture_path=config.dataset.fixture_path,
    )
