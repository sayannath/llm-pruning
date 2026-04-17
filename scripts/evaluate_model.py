#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_pruning_mmlu.config import load_config_dict, load_model_config, parse_config
from llm_pruning_mmlu.experiments.sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config")
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    data = load_config_dict(args.config)
    if args.model_config:
        data["model"] = load_model_config(args.model_config).__dict__
        data.pop("models", None)
    data.setdefault("pruning", {})["sparsities"] = [args.sparsity]
    if args.max_samples is not None:
        data.setdefault("evaluation", {})["max_samples"] = args.max_samples
    if args.subjects:
        data.setdefault("evaluation", {})["subjects"] = args.subjects
    run_dir = run_sweep(parse_config(data), data, fail_fast=args.fail_fast)
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
