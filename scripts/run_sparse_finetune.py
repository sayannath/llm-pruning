#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_pruning_mmlu.config import load_config_dict, parse_config
from llm_pruning_mmlu.finetuning.config import load_finetuning_config, parse_finetuning_config
from llm_pruning_mmlu.finetuning.runner import run_sft_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structured sparse fine-tuning sweep (extends to semi-structured via config)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Experiment config (model, pruning, eval). Same format as run_sweep.py.",
    )
    parser.add_argument(
        "--finetune-config",
        required=True,
        help="Fine-tuning config (LoRA, dataset splits, wandb, mask policy).",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int,
                        help="Cap both SFT validation samples and final eval samples.")
    parser.add_argument(
        "--sparsities",
        type=int,
        nargs="+",
        help="Override sparsity points from the experiment config.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    # parser.add_argument("--no-resume", action="store_true")  # resume logic removed
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    exp_dict = load_config_dict(args.config)
    ft_dict = load_config_dict(args.finetune_config) if Path(args.finetune_config).exists() else {}

    if args.max_train_samples is not None:
        ft_dict.setdefault("finetuning", {}).setdefault(
            "train_dataset", {}
        )["max_samples"] = args.max_train_samples

    if args.max_eval_samples is not None:
        exp_dict.setdefault("evaluation", {})["max_samples"] = args.max_eval_samples
        ft_dict.setdefault("finetuning", {}).setdefault(
            "validation_dataset", {}
        )["max_samples"] = args.max_eval_samples

    if args.sparsities:
        exp_dict.setdefault("pruning", {})["sparsities"] = args.sparsities

    if args.no_wandb:
        ft_dict.setdefault("finetuning", {}).setdefault("wandb", {})["enabled"] = False

    # if args.no_resume:
    #     exp_dict["resume"] = False

    exp_cfg = parse_config(exp_dict)
    ft_cfg = parse_finetuning_config(ft_dict)

    run_dir = run_sft_sweep(
        experiment_config=exp_cfg,
        experiment_config_dict=exp_dict,
        ft_config=ft_cfg,
        ft_config_dict=ft_dict,
        fail_fast=args.fail_fast,
    )
    print(f"Wrote SFT artifacts to {run_dir}")


if __name__ == "__main__":
    main()
