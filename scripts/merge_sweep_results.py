#!/usr/bin/env python
"""Merge combined_results.csv files from multiple sweep run directories.

Usage:
    python scripts/merge_sweep_results.py \\
        --runs outputs/mmlu_pruning_<hash_unstructured> \\
               outputs/mmlu_pruning_<hash_structured> \\
               outputs/mmlu_pruning_<hash_2_4> \\
               outputs/mmlu_pruning_<hash_4_8> \\
        --out outputs/merged_comparison/

Reads metrics.json files from each run directory via collect_metrics, merges
all rows into a single CSV, and generates the cross-method comparison plots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from llm_pruning_mmlu.reporting.tables import collect_metrics
from llm_pruning_mmlu.reporting.plots import plot_sparsity_vs_accuracy
from llm_pruning_mmlu.utils.io import ensure_dir, write_csv, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="RUN_DIR",
        help="One or more sweep run directories to merge.",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="OUT_DIR",
        help="Output directory for merged CSV and plots.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out))
    all_rows: list[dict] = []
    for run_dir in args.runs:
        run_path = Path(run_dir)
        if not run_path.exists():
            print(f"WARNING: run directory not found, skipping: {run_path}", flush=True)
            continue
        rows = collect_metrics(run_path)
        print(f"  {run_path.name}: {len(rows)} rows", flush=True)
        all_rows.extend(rows)

    if not all_rows:
        print("No rows collected — nothing to write.", flush=True)
        return

    csv_path = out_dir / "merged_combined_results.csv"
    jsonl_path = out_dir / "merged_combined_results.jsonl"
    write_csv(csv_path, all_rows)
    write_jsonl(jsonl_path, all_rows)
    print(f"Wrote {len(all_rows)} rows → {csv_path}", flush=True)

    # Summary by model × method
    df = pd.DataFrame(all_rows)
    group_cols = [c for c in ("model_name", "pruning_method", "pruning_structure") if c in df.columns]
    if group_cols:
        summary = (
            df.groupby(group_cols, as_index=False)
            .agg(
                best_accuracy=("accuracy", "max"),
                worst_accuracy=("accuracy", "min"),
                num_points=("accuracy", "count"),
            )
            .to_dict(orient="records")
        )
        summary_path = out_dir / "merged_summary_by_model.csv"
        write_csv(summary_path, summary)
        print(f"Wrote summary → {summary_path}", flush=True)

    # Cross-method accuracy plot
    plot_path = out_dir / "plots" / "sparsity_vs_accuracy.png"
    try:
        plot_sparsity_vs_accuracy(csv_path, plot_path)
        print(f"Wrote plot → {plot_path}", flush=True)
    except Exception as exc:
        print(f"WARNING: plot failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
