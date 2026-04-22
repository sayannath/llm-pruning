#!/usr/bin/env python
"""Verify that the 2:4 sparse mask survived fine-tuning.

Loads masks.pt and the saved LoRA adapter from a completed SFT run directory,
then checks that every pruned base-weight position is still zero and that every
complete 2:4 block has exactly 2 zeros.

Usage:
    python scripts/validate_sparse_sft_masks.py \
        --run-dir outputs/sft_runs/<run_id>/<model>/global_magnitude_semi_structured__nm_2_4/sparsity_050 \
        --config configs/experiments/qwen3_semi_structured_2_4_sft.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from llm_pruning_mmlu.config import load_config_dict, parse_config
from llm_pruning_mmlu.pruning.semi_structured import validate_nm_mask


def _load_base_weights(run_dir: Path, config_path: str) -> dict[str, torch.Tensor]:
    cfg = parse_config(load_config_dict(config_path))
    model_cfgs = cfg.models if cfg.model is None else [cfg.model]
    # Infer model from run_dir name (first segment after run_id)
    hf_id = None
    for mc_path in model_cfgs:
        from llm_pruning_mmlu.config import load_model_config
        mc = load_model_config(mc_path)
        if mc.name.lower().replace("-", "_") in str(run_dir).lower().replace("-", "_"):
            hf_id = mc.hf_id
            break
    if hf_id is None:
        # Fall back to first model
        from llm_pruning_mmlu.config import load_model_config
        hf_id = load_model_config(model_cfgs[0]).hf_id

    adapter_dir = run_dir / "adapter"
    print(f"Loading base model {hf_id} ...")
    base = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )
    if adapter_dir.exists():
        print(f"Merging LoRA adapter from {adapter_dir} ...")
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model = model.merge_and_unload()
    else:
        print("No adapter/ directory found — validating base weights only.")
        model = base

    return {name: param.data.clone() for name, param in model.named_parameters()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 2:4 N:M masks after sparse SFT.")
    parser.add_argument("--run-dir", required=True, help="Path to a sparsity_050 run directory.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument(
        "--nm-n", type=int, default=2, help="N in N:M pattern (default: 2)."
    )
    parser.add_argument(
        "--nm-m", type=int, default=4, help="M in N:M pattern (default: 4)."
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    masks_path = run_dir / "masks.pt"

    if not masks_path.exists():
        print(f"ERROR: masks.pt not found at {masks_path}")
        sys.exit(1)

    print(f"Loading masks from {masks_path} ...")
    masks: dict[str, torch.Tensor] = torch.load(masks_path, map_location="cpu", weights_only=True)
    print(f"  {len(masks)} masked parameters")

    weights = _load_base_weights(run_dir, args.config)

    results: dict[str, dict] = {}
    total_violations = 0
    regrowth_count = 0

    for param_name, mask in masks.items():
        weight = weights.get(param_name)
        if weight is None:
            print(f"  WARNING: {param_name} not found in loaded model — skipping")
            continue

        # Check for weight regrowth: any position that was pruned (mask=False) should be zero
        pruned_positions = ~mask
        if pruned_positions.any():
            regrown = (weight[pruned_positions].abs() > 1e-8).sum().item()
            regrowth_count += int(regrown)
        else:
            regrown = 0

        nm_result = validate_nm_mask(weight, mask, n=args.nm_n, m=args.nm_m, block_dim=1)
        nm_result["regrown_weights"] = int(regrown)

        results[param_name] = nm_result
        total_violations += nm_result["violations"]

    passed = total_violations == 0 and regrowth_count == 0
    summary = {
        "passed": passed,
        "total_nm_violations": total_violations,
        "total_regrown_weights": regrowth_count,
        "parameters_checked": len(results),
        "nm_n": args.nm_n,
        "nm_m": args.nm_m,
        "per_parameter": results,
    }

    out_path = run_dir / "mask_validation.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'PASSED' if passed else 'FAILED'}")
    print(f"  Parameters checked : {len(results)}")
    print(f"  N:M block violations: {total_violations}")
    print(f"  Regrown weights    : {regrowth_count}")
    print(f"  Written            : {out_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
