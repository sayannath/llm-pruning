#!/usr/bin/env python
"""Build the Phase 6 results CSV for semi-structured 2:4 sparse SFT.

Walks sft_runs to find completed SFT metrics.json files.
Optionally matches against prune-only and dense baselines from a separate
runs directory.

Usage:
    python scripts/summarize_sparse_sft.py \
        --sft-runs-dir outputs/sft_runs \
        --baseline-runs-dir outputs/runs \
        --output outputs/summaries/semi_structured_2_4_sparse_sft_summary.csv

    # With a specific pruning method filter:
    python scripts/summarize_sparse_sft.py \
        --sft-runs-dir outputs/sft_runs \
        --baseline-runs-dir outputs/runs \
        --method global_magnitude_semi_structured \
        --output outputs/summaries/magnitude_2_4_sft_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _collect_metrics(root: Path, method_filter: str | None) -> list[dict]:
    """Walk root recursively and return all metrics.json contents."""
    records = []
    for p in sorted(root.rglob("metrics.json")):
        try:
            m = _load_json(p)
        except Exception:
            continue
        m["_path"] = str(p)
        if method_filter and m.get("pruning_method") != method_filter:
            continue
        records.append(m)
    return records


def _build_baseline_index(
    baseline_dir: Path | None, method_filter: str | None
) -> dict[tuple[str, int], dict]:
    """Returns {(model_name, sparsity_requested): metrics} for baseline (prune-only) runs."""
    if baseline_dir is None or not baseline_dir.exists():
        return {}
    index: dict[tuple[str, int], dict] = {}
    for m in _collect_metrics(baseline_dir, method_filter):
        key = (m.get("model_name", ""), int(m.get("sparsity_requested", -1)))
        # Prefer the first entry found (sorted by path so deterministic)
        if key not in index:
            index[key] = m
    return index


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _recovery_fraction(
    sft_acc: float | None,
    prune_acc: float | None,
    dense_acc: float | None,
) -> float | None:
    if sft_acc is None or prune_acc is None or dense_acc is None:
        return None
    gap = dense_acc - prune_acc
    if abs(gap) < 1e-9:
        return None
    return (sft_acc - prune_acc) / gap


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise sparse SFT results into a CSV.")
    parser.add_argument("--sft-runs-dir", required=True)
    parser.add_argument(
        "--baseline-runs-dir",
        default=None,
        help="Directory with prune-only runs. If omitted, recovery columns will be empty.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Filter to a specific pruning_method (e.g. global_magnitude_semi_structured).",
    )
    parser.add_argument(
        "--output",
        default="outputs/summaries/semi_structured_2_4_sparse_sft_summary.csv",
    )
    args = parser.parse_args()

    sft_dir = Path(args.sft_runs_dir)
    if not sft_dir.exists():
        print(f"ERROR: --sft-runs-dir does not exist: {sft_dir}")
        sys.exit(1)

    baseline_index = _build_baseline_index(
        Path(args.baseline_runs_dir) if args.baseline_runs_dir else None,
        args.method,
    )
    dense_index: dict[str, float] = {
        model: m["accuracy"]
        for (model, sp), m in baseline_index.items()
        if sp == 0
    }

    sft_records = _collect_metrics(sft_dir, args.method)
    if not sft_records:
        print(f"No metrics.json files found under {sft_dir} (method filter: {args.method})")
        sys.exit(0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "pruning_method",
        "sparsity_requested",
        "sparsity_achieved",
        "finetuning_method",
        "dense_accuracy",
        "prune_only_accuracy",
        "sft_accuracy",
        "accuracy_recovered",
        "recovery_fraction",
        "train_samples",
        "eval_samples",
        "lora_r",
        "lora_alpha",
        "epochs",
        "learning_rate",
        "mask_policy_preserve_base",
        "mask_validation_passed",
        "runtime_s",
        "emissions_kg_co2",
        "sft_metrics_path",
    ]

    rows = []
    for m in sft_records:
        model_name = m.get("model_name", "")
        sparsity_req = int(m.get("sparsity_requested", -1))
        sft_acc = m.get("accuracy")

        prune_key = (model_name, sparsity_req)
        baseline_m = baseline_index.get(prune_key)
        prune_acc = baseline_m["accuracy"] if baseline_m else None
        dense_acc = dense_index.get(model_name)

        # Check mask_validation.json if present
        mask_valid: bool | None = None
        metrics_path = Path(m.get("_path", ""))
        mask_val_path = metrics_path.parent / "mask_validation.json"
        if mask_val_path.exists():
            try:
                mv = _load_json(mask_val_path)
                mask_valid = bool(mv.get("passed"))
            except Exception:
                pass

        acc_recovered = (
            round(sft_acc - prune_acc, 6)
            if sft_acc is not None and prune_acc is not None
            else None
        )

        rows.append({
            "model_name": model_name,
            "pruning_method": m.get("pruning_method", ""),
            "sparsity_requested": sparsity_req,
            "sparsity_achieved": m.get("sparsity_achieved"),
            "finetuning_method": m.get("finetuning_method", ""),
            "dense_accuracy": round(dense_acc, 6) if dense_acc is not None else "",
            "prune_only_accuracy": round(prune_acc, 6) if prune_acc is not None else "",
            "sft_accuracy": round(sft_acc, 6) if sft_acc is not None else "",
            "accuracy_recovered": round(acc_recovered, 6) if acc_recovered is not None else "",
            "recovery_fraction": round(
                _recovery_fraction(sft_acc, prune_acc, dense_acc), 4
            ) if _recovery_fraction(sft_acc, prune_acc, dense_acc) is not None else "",
            "train_samples": m.get("train_samples", ""),
            "eval_samples": m.get("num_samples", ""),
            "lora_r": m.get("lora_r", ""),
            "lora_alpha": m.get("lora_alpha", ""),
            "epochs": m.get("epochs", ""),
            "learning_rate": m.get("learning_rate", ""),
            "mask_policy_preserve_base": m.get("mask_policy_preserve_base", ""),
            "mask_validation_passed": "" if mask_valid is None else mask_valid,
            "runtime_s": m.get("runtime_s", ""),
            "emissions_kg_co2": m.get("emissions_kg_co2", ""),
            "sft_metrics_path": m.get("_path", ""),
        })

    rows.sort(key=lambda r: (r["model_name"], r["pruning_method"], r["sparsity_requested"]))

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
