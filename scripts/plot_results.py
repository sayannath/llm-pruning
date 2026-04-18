#!/usr/bin/env python
"""Generate all result visualisations for the pruning sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

COLORS = {"llama31_8b_instruct": "#4C72B0", "qwen3_8b": "#DD8452"}
LABELS = {"llama31_8b_instruct": "Llama-3.1-8B-Instruct", "qwen3_8b": "Qwen3-8B"}
CHANCE = 0.25  # 4-way MCQA random baseline


def load_data(run_dir: Path) -> pd.DataFrame:
    rows = []
    for mpath in sorted(run_dir.glob("*/sparsity_*/metrics.json")):
        m = json.load(open(mpath))
        epath = mpath.parent / "emissions.json"
        e = json.load(open(epath)) if epath.exists() else {}
        rows.append(
            {
                "model": m["model_name"],
                "sparsity": m["sparsity_requested"],
                "accuracy": m["accuracy"],
                "num_correct": m.get("num_correct", 0),
                "num_total": m.get("num_total_target_params", 0),
                "num_nonzero": m.get("num_nonzero_target_params", 0),
                "co2_kg": e.get("emissions_kg_co2", 0.0),
                "energy_kwh": e.get("energy_consumed_kwh", 0.0),
                "duration_s": e.get("duration_s", 0.0),
                "gpu_power_w": e.get("gpu_power_w", 0.0),
                "cpu_power_w": e.get("cpu_power_w", 0.0),
                "ram_power_w": e.get("ram_power_w", 0.0),
            }
        )
    df = pd.DataFrame(rows).sort_values(["model", "sparsity"]).reset_index(drop=True)
    for model in df["model"].unique():
        mask = df["model"] == model
        baseline = df.loc[mask & (df["sparsity"] == 0), "accuracy"].iloc[0]
        df.loc[mask, "accuracy_drop"] = baseline - df.loc[mask, "accuracy"]
        df.loc[mask, "accuracy_retained"] = df.loc[mask, "accuracy"] / baseline
        df.loc[mask, "co2_g"] = df.loc[mask, "co2_kg"] * 1000
        df.loc[mask, "energy_wh"] = df.loc[mask, "energy_kwh"] * 1000
    return df


def plot_accuracy_vs_sparsity(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, grp in df.groupby("model"):
        ax.plot(grp["sparsity"], grp["accuracy"], marker="o", linewidth=2,
                color=COLORS[model], label=LABELS[model])
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random (0.25)")
    ax.fill_betweenx([0, 1], 40, 50, alpha=0.07, color="red")
    ax.annotate("Cliff\n(40→50%)", xy=(45, 0.20), fontsize=9, color="red",
                ha="center", style="italic")
    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("MMLU Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Sparsity — Global Unstructured Magnitude Pruning", fontsize=13)
    ax.set_xlim(-2, 101); ax.set_ylim(0.15, 0.80)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_accuracy_retained(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, grp in df.groupby("model"):
        ax.plot(grp["sparsity"], grp["accuracy_retained"] * 100, marker="o",
                linewidth=2, color=COLORS[model], label=LABELS[model])
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense baseline")
    ax.axhline(50, color="red", linestyle=":", linewidth=1, label="50% retained")
    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Accuracy Retained (%)", fontsize=12)
    ax.set_title("Accuracy Retained Relative to Dense Baseline", fontsize=13)
    ax.set_xlim(-2, 101); ax.set_ylim(0, 115)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_co2_vs_sparsity(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, grp in df.groupby("model"):
        offset = -2 if model == "llama31_8b_instruct" else 2
        ax.bar(grp["sparsity"] + offset, grp["co2_g"], width=4,
               color=COLORS[model], label=LABELS[model], alpha=0.85)
    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("CO₂ Emissions (g CO₂eq)", fontsize=12)
    ax.set_title("Carbon Emissions per Sparsity Run", fontsize=13)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_accuracy_vs_co2(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, grp in df.groupby("model"):
        sc = ax.scatter(grp["co2_g"], grp["accuracy"], c=grp["sparsity"],
                        cmap="plasma", s=80, edgecolors=COLORS[model],
                        linewidths=1.5, label=LABELS[model], zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(f"{int(row['sparsity'])}%", (row["co2_g"], row["accuracy"]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7.5, color=COLORS[model])
    fig.colorbar(sc, ax=ax, label="Sparsity (%)")
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("CO₂ Emissions (g CO₂eq)", fontsize=12)
    ax.set_ylabel("MMLU Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Carbon Cost per Run", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_efficiency_frontier(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    df["acc_per_co2"] = df["accuracy"] / df["co2_g"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, grp in df.groupby("model"):
        ax.plot(grp["sparsity"], grp["acc_per_co2"], marker="o", linewidth=2,
                color=COLORS[model], label=LABELS[model])
    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Accuracy / g CO₂eq", fontsize=12)
    ax.set_title("Carbon Efficiency: Accuracy per Gram of CO₂", fontsize=13)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_power_breakdown(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, (model, grp) in zip(axes, df.groupby("model")):
        sp = grp["sparsity"].values
        x = np.arange(len(sp))
        w = 0.28
        ax.bar(x - w, grp["gpu_power_w"], width=w, label="GPU", color="#4C72B0", alpha=0.85)
        ax.bar(x,     grp["cpu_power_w"], width=w, label="CPU", color="#55A868", alpha=0.85)
        ax.bar(x + w, grp["ram_power_w"], width=w, label="RAM", color="#C44E52", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}%" for s in sp], rotation=45, ha="right")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Sparsity (%)"); ax.set_ylabel("Average Power (W)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("GPU / CPU / RAM Power Draw per Sparsity Run", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_cumulative_co2(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sp_labels = None
    for model, grp in df.groupby("model"):
        cumco2 = grp["co2_g"].cumsum().values
        sp_labels = grp["sparsity"].values
        ax.step(range(len(cumco2)), cumco2, where="post", linewidth=2,
                color=COLORS[model], label=LABELS[model])
        ax.fill_between(range(len(cumco2)), cumco2, step="post",
                        alpha=0.15, color=COLORS[model])
    ax.set_xticks(range(len(sp_labels)))
    ax.set_xticklabels([f"{s}%" for s in sp_labels], rotation=45, ha="right")
    ax.set_xlabel("Sparsity checkpoint"); ax.set_ylabel("Cumulative CO₂ (g CO₂eq)")
    ax.set_title("Cumulative Carbon Footprint Across Sweep", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_dashboard(df: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # A — accuracy vs sparsity
    ax1 = fig.add_subplot(gs[0, 0])
    for model, grp in df.groupby("model"):
        ax1.plot(grp["sparsity"], grp["accuracy"], marker="o", linewidth=2,
                 color=COLORS[model], label=LABELS[model])
    ax1.axhline(CHANCE, color="gray", linestyle="--", linewidth=1)
    ax1.fill_betweenx([0, 1], 40, 50, alpha=0.07, color="red")
    ax1.set_xlabel("Sparsity (%)"); ax1.set_ylabel("MMLU Accuracy")
    ax1.set_title("A — Accuracy vs Sparsity", fontweight="bold")
    ax1.set_xlim(-2, 101); ax1.set_ylim(0.15, 0.80)
    ax1.set_xticks([0, 20, 40, 60, 80, 99])
    ax1.legend(fontsize=8.5); ax1.grid(True, alpha=0.3)

    # B — CO2 per run
    ax2 = fig.add_subplot(gs[0, 1])
    for model, grp in df.groupby("model"):
        offset = -2 if model == "llama31_8b_instruct" else 2
        ax2.bar(grp["sparsity"] + offset, grp["co2_g"], width=4,
                color=COLORS[model], label=LABELS[model], alpha=0.85)
    ax2.set_xlabel("Sparsity (%)"); ax2.set_ylabel("CO₂ (g CO₂eq)")
    ax2.set_title("B — Carbon Emissions per Run", fontweight="bold")
    ax2.set_xticks([0, 20, 40, 60, 80, 99])
    ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3, axis="y")

    # C — accuracy vs CO2 scatter
    ax3 = fig.add_subplot(gs[1, 0])
    for model, grp in df.groupby("model"):
        sc = ax3.scatter(grp["co2_g"], grp["accuracy"], c=grp["sparsity"],
                         cmap="plasma", s=70, edgecolors=COLORS[model],
                         linewidths=1.2, label=LABELS[model], zorder=3)
    fig.colorbar(sc, ax=ax3, label="Sparsity (%)", pad=0.02)
    ax3.axhline(CHANCE, color="gray", linestyle="--", linewidth=1)
    ax3.set_xlabel("CO₂ (g CO₂eq)"); ax3.set_ylabel("MMLU Accuracy")
    ax3.set_title("C — Accuracy vs Carbon Cost", fontweight="bold")
    ax3.legend(fontsize=8.5); ax3.grid(True, alpha=0.3)

    # D — cumulative CO2
    ax4 = fig.add_subplot(gs[1, 1])
    sp_labels = None
    for model, grp in df.groupby("model"):
        cumco2 = grp["co2_g"].cumsum().values
        sp_labels = grp["sparsity"].values
        ax4.step(range(len(cumco2)), cumco2, where="post", linewidth=2,
                 color=COLORS[model], label=LABELS[model])
        ax4.fill_between(range(len(cumco2)), cumco2, step="post",
                         alpha=0.12, color=COLORS[model])
    ax4.set_xticks(range(len(sp_labels)))
    ax4.set_xticklabels([f"{s}%" for s in sp_labels], rotation=45, ha="right", fontsize=8)
    ax4.set_xlabel("Sparsity checkpoint"); ax4.set_ylabel("Cumulative CO₂ (g CO₂eq)")
    ax4.set_title("D — Cumulative Carbon Footprint", fontweight="bold")
    ax4.legend(fontsize=8.5); ax4.grid(True, alpha=0.3)

    fig.suptitle("LLM Pruning Sweep — Accuracy & Carbon Emissions Dashboard",
                 fontsize=14, fontweight="bold", y=0.998)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pruning sweep visualisations.")
    parser.add_argument("--run-dir", default="outputs/runs/mmlu_pruning_ade7d5ffbbb4",
                        help="Path to the sweep run directory.")
    parser.add_argument("--out-dir", default="outputs/plots",
                        help="Directory to write plots into.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {run_dir}")
    df = load_data(run_dir)
    print(f"Loaded {len(df)} runs across {df['model'].nunique()} models\n")

    print("Generating plots...")
    plot_accuracy_vs_sparsity(df,   out_dir / "accuracy_vs_sparsity.png")
    plot_accuracy_retained(df,      out_dir / "accuracy_retained.png")
    plot_co2_vs_sparsity(df,        out_dir / "co2_vs_sparsity.png")
    plot_accuracy_vs_co2(df,        out_dir / "accuracy_vs_co2.png")
    plot_efficiency_frontier(df,    out_dir / "carbon_efficiency.png")
    plot_power_breakdown(df,        out_dir / "power_breakdown.png")
    plot_cumulative_co2(df,         out_dir / "cumulative_co2.png")
    plot_dashboard(df,              out_dir / "dashboard.png")
    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
