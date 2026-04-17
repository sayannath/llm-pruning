#!/usr/bin/env python
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_pruning_mmlu.config import load_config_dict, parse_config
from llm_pruning_mmlu.experiments.sweep import run_sweep


def main() -> None:
    data = load_config_dict("tests/fixtures/dummy_config.yaml")
    with tempfile.TemporaryDirectory() as tmp:
        data["output_root"] = tmp
        run_dir = run_sweep(parse_config(data), data, fail_fast=True)
        print(f"Smoke run completed: {Path(run_dir)}")


if __name__ == "__main__":
    main()
