from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_pruning_mmlu.utils.io import read_json, write_csv, write_jsonl


def collect_metrics(run_dir: str | Path) -> list[dict]:
    run_dir = Path(run_dir)
    rows = []
    for path in sorted(run_dir.glob("*/sparsity_*/metrics.json")):
        rows.append(read_json(path))
    return rows


def write_combined_results(run_dir: str | Path) -> list[dict]:
    run_dir = Path(run_dir)
    rows = collect_metrics(run_dir)
    write_csv(run_dir / "combined_results.csv", rows)
    write_jsonl(run_dir / "combined_results.jsonl", rows)
    if rows:
        df = pd.DataFrame(rows)
        summary = (
            df.groupby("model_name", as_index=False)
            .agg(
                best_accuracy=("accuracy", "max"),
                worst_accuracy=("accuracy", "min"),
                num_points=("accuracy", "count"),
            )
            .to_dict(orient="records")
        )
        write_csv(run_dir / "summary_by_model.csv", summary)
    return rows
