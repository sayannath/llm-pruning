#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_pruning_mmlu.reporting.tables import write_combined_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    rows = write_combined_results(args.run_dir)
    print(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
