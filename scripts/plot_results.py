#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_pruning_mmlu.reporting.plots import plot_sparsity_vs_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = args.output or str(
        Path(args.summary_csv).parent / "plots" / "sparsity_vs_accuracy.png"
    )
    plot_sparsity_vs_accuracy(args.summary_csv, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
