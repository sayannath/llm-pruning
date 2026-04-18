#!/usr/bin/env python
"""Generate result visualisations for structured and semi-structured pruning sweeps."""
from __future__ import annotations

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
CHANCE = 0.25

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_structured(run_dir: Path) -> pd.DataFrame:
    rows = []
    for mpath in sorted(run_dir.glob("**/sparsity_*/metrics.json")):
        m = json.load(open(mpath))
        if m.get("pruning_method") != "global_magnitude_structured":
            continue
        epath = mpath.parent / "emissions.json"
        e = json.load(open(epath)) if epath.exists() else {}
        rows.append({
            "model": m["model_name"],
            "sparsity_requested": m["sparsity_requested"],
            "sparsity_achieved": m["sparsity_achieved"],
            "accuracy": m["accuracy"],
            "co2_kg": m.get("emissions_kg_co2", e.get("emissions_kg_co2", 0.0)),
            "num_groups_total": m.get("num_groups_total"),
            "num_groups_pruned": m.get("num_groups_pruned", 0),
            "group_sparsity": m.get("group_sparsity_achieved", 0.0),
        })
    df = pd.DataFrame(rows).sort_values(["model", "sparsity_requested"]).reset_index(drop=True)
    for model in df["model"].unique():
        mask = df["model"] == model
        baseline = df.loc[mask & (df["sparsity_requested"] == 0), "accuracy"].iloc[0]
        df.loc[mask, "accuracy_retained"] = df.loc[mask, "accuracy"] / baseline * 100
        df.loc[mask, "co2_g"] = df.loc[mask, "co2_kg"] * 1000
    return df


def load_semi_structured(run_dirs: dict[str, Path]) -> pd.DataFrame:
    """Load semi-structured runs; run_dirs maps pattern label → run dir."""
    rows = []
    for pattern, run_dir in run_dirs.items():
        for mpath in sorted(run_dir.glob("**/sparsity_*/metrics.json")):
            m = json.load(open(mpath))
            epath = mpath.parent / "emissions.json"
            e = json.load(open(epath)) if epath.exists() else {}
            rows.append({
                "pattern": pattern,
                "model": m["model_name"],
                "sparsity_requested": m["sparsity_requested"],
                "sparsity_achieved": m["sparsity_achieved"],
                "accuracy": m["accuracy"],
                "nm_n": m.get("nm_n"),
                "nm_m": m.get("nm_m"),
                "nm_sparsity": m.get("nm_sparsity_achieved", 0.0),
                "co2_kg": m.get("emissions_kg_co2", e.get("emissions_kg_co2", 0.0)),
            })
    df = pd.DataFrame(rows).sort_values(["pattern", "model", "sparsity_requested"]).reset_index(drop=True)
    for (pattern, model), grp in df.groupby(["pattern", "model"]):
        mask = (df["pattern"] == pattern) & (df["model"] == model)
        baseline = grp.loc[grp["sparsity_requested"] == 0, "accuracy"].iloc[0]
        df.loc[mask, "accuracy_retained"] = df.loc[mask, "accuracy"] / baseline * 100
        df.loc[mask, "co2_g"] = df.loc[mask, "co2_kg"] * 1000
    return df


# ---------------------------------------------------------------------------
# Structured plots
# ---------------------------------------------------------------------------

def plot_struct_accuracy(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, grp in df.groupby("model"):
        ax.plot(grp["sparsity_requested"], grp["accuracy"], marker="o", linewidth=2,
                color=COLORS[model], label=LABELS[model])
        # mark cliff
        cliff = grp[grp["accuracy"] < CHANCE + 0.05].head(1)
        if not cliff.empty:
            ax.axvline(cliff["sparsity_requested"].iloc[0], color=COLORS[model],
                       linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random (0.25)")
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("MMLU Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Sparsity — Structured MLP-Channel Pruning", fontsize=13)
    ax.set_xlim(-2, 75); ax.set_ylim(0.15, 0.80)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_struct_accuracy_retained(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, grp in df.groupby("model"):
        ax.plot(grp["sparsity_requested"], grp["accuracy_retained"], marker="o",
                linewidth=2, color=COLORS[model], label=LABELS[model])
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense baseline")
    ax.axhline(50, color="red", linestyle=":", linewidth=1, label="50% retained")
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("Accuracy Retained (%)", fontsize=12)
    ax.set_title("Accuracy Retained — Structured MLP-Channel Pruning", fontsize=13)
    ax.set_xlim(-2, 75); ax.set_ylim(0, 115)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_struct_weight_vs_group_sparsity(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, grp in df.groupby("model"):
        ax.plot(grp["sparsity_requested"], grp["sparsity_achieved"] * 100, marker="s",
                linewidth=2, color=COLORS[model], label=LABELS[model])
    # ideal 1:1 line
    xs = [0, 70]
    ax.plot(xs, xs, color="gray", linestyle="--", linewidth=1, label="1:1 ideal")
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("Weight Sparsity Achieved (%)", fontsize=12)
    ax.set_title("Group Sparsity vs Achieved Weight Sparsity", fontsize=13)
    ax.set_xlim(-2, 75); ax.set_ylim(-2, 75)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_struct_co2(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    models = df["model"].unique()
    x = np.arange(len(df["sparsity_requested"].unique()))
    sp_vals = sorted(df["sparsity_requested"].unique())
    width = 0.35
    for i, model in enumerate(sorted(models)):
        grp = df[df["model"] == model].sort_values("sparsity_requested")
        offset = (i - 0.5) * width
        ax.bar(x + offset, grp["co2_g"], width=width,
               color=COLORS[model], label=LABELS[model], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}%" for s in sp_vals])
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("CO₂ Emissions (g CO₂eq)", fontsize=12)
    ax.set_title("Carbon Emissions per Sparsity Run — Structured MLP", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_struct_groups_pruned(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, grp in df.groupby("model"):
        frac = grp["num_groups_pruned"] / grp["num_groups_total"] * 100
        ax.plot(grp["sparsity_requested"], frac, marker="o", linewidth=2,
                color=COLORS[model], label=LABELS[model])
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("% MLP Channels Pruned", fontsize=12)
    ax.set_title("MLP Channels Pruned vs Requested Sparsity", fontsize=13)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_struct_dashboard(df: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    for model, grp in df.groupby("model"):
        ax1.plot(grp["sparsity_requested"], grp["accuracy"], marker="o",
                 linewidth=2, color=COLORS[model], label=LABELS[model])
    ax1.axhline(CHANCE, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Group Sparsity (%)"); ax1.set_ylabel("MMLU Accuracy")
    ax1.set_title("A — Accuracy vs Group Sparsity", fontweight="bold")
    ax1.set_xlim(-2, 75); ax1.set_ylim(0.15, 0.80)
    ax1.legend(fontsize=8.5); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for model, grp in df.groupby("model"):
        ax2.plot(grp["sparsity_requested"], grp["accuracy_retained"], marker="o",
                 linewidth=2, color=COLORS[model], label=LABELS[model])
    ax2.axhline(100, color="gray", linestyle="--", linewidth=1)
    ax2.axhline(50, color="red", linestyle=":", linewidth=1)
    ax2.set_xlabel("Group Sparsity (%)"); ax2.set_ylabel("Accuracy Retained (%)")
    ax2.set_title("B — Accuracy Retained", fontweight="bold")
    ax2.set_xlim(-2, 75); ax2.set_ylim(0, 115)
    ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    for model, grp in df.groupby("model"):
        ax3.plot(grp["sparsity_requested"], grp["sparsity_achieved"] * 100, marker="s",
                 linewidth=2, color=COLORS[model], label=LABELS[model])
    ax3.plot([0, 70], [0, 70], color="gray", linestyle="--", linewidth=1, label="1:1 ideal")
    ax3.set_xlabel("Group Sparsity (%)"); ax3.set_ylabel("Weight Sparsity Achieved (%)")
    ax3.set_title("C — Group vs Weight Sparsity", fontweight="bold")
    ax3.set_xlim(-2, 75); ax3.set_ylim(-2, 75)
    ax3.legend(fontsize=8.5); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    for model, grp in df.groupby("model"):
        sc = ax4.scatter(grp["co2_g"], grp["accuracy"], c=grp["sparsity_requested"],
                         cmap="plasma", s=80, edgecolors=COLORS[model],
                         linewidths=1.5, label=LABELS[model], zorder=3)
        for _, row in grp.iterrows():
            ax4.annotate(f"{int(row['sparsity_requested'])}%",
                         (row["co2_g"], row["accuracy"]),
                         textcoords="offset points", xytext=(5, 4),
                         fontsize=7.5, color=COLORS[model])
    fig.colorbar(sc, ax=ax4, label="Group Sparsity (%)", pad=0.02)
    ax4.axhline(CHANCE, color="gray", linestyle="--", linewidth=1)
    ax4.set_xlabel("CO₂ (g CO₂eq)"); ax4.set_ylabel("MMLU Accuracy")
    ax4.set_title("D — Accuracy vs Carbon Cost", fontweight="bold")
    ax4.legend(fontsize=8.5); ax4.grid(True, alpha=0.3)

    fig.suptitle("Structured MLP-Channel Pruning — Accuracy & Carbon Dashboard",
                 fontsize=14, fontweight="bold", y=0.998)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Semi-structured plots
# ---------------------------------------------------------------------------

PATTERN_COLORS = {"2:4": "#2ca02c", "4:8": "#9467bd"}
PATTERN_MARKERS = {"2:4": "o", "4:8": "s"}


def _pattern_label(pattern: str, model: str) -> str:
    return f"{LABELS[model]} ({pattern})"


def plot_semi_accuracy_bar(df: pd.DataFrame, out: Path) -> None:
    """Bar chart comparing baseline vs pruned accuracy for both patterns and models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    models = sorted(df["model"].unique())
    patterns = sorted(df["pattern"].unique())
    x = np.arange(len(patterns))
    width = 0.35

    for ax, model in zip(axes, models):
        baseline_vals = []
        pruned_vals = []
        for pat in patterns:
            sub = df[(df["model"] == model) & (df["pattern"] == pat)]
            baseline_vals.append(sub.loc[sub["sparsity_requested"] == 0, "accuracy"].iloc[0])
            pruned_vals.append(sub.loc[sub["sparsity_requested"] == 50, "accuracy"].iloc[0])

        bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Dense (0%)",
                       color="#7fcdbb", alpha=0.9)
        bars2 = ax.bar(x + width / 2, pruned_vals, width, label="Pruned (50%)",
                       color=[PATTERN_COLORS[p] for p in patterns], alpha=0.9)
        ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random")
        ax.set_xticks(x); ax.set_xticklabels(patterns, fontsize=12)
        ax.set_xlabel("N:M Pattern", fontsize=12)
        ax.set_ylabel("MMLU Accuracy", fontsize=12)
        ax.set_title(LABELS[model], fontsize=12)
        ax.set_ylim(0.15, 0.80); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars1, baseline_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)
        for bar, val in zip(bars2, pruned_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Semi-Structured N:M Pruning — Accuracy at 50% Sparsity", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_semi_accuracy_drop(df: pd.DataFrame, out: Path) -> None:
    """Accuracy drop (dense → 50% pruned) grouped by model and pattern."""
    pruned = df[df["sparsity_requested"] == 50].copy()
    dense = df[df["sparsity_requested"] == 0][["pattern", "model", "accuracy"]].rename(
        columns={"accuracy": "baseline"})
    merged = pruned.merge(dense, on=["pattern", "model"])
    merged["drop_pp"] = (merged["baseline"] - merged["accuracy"]) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    patterns = sorted(merged["pattern"].unique())
    models = sorted(merged["model"].unique())
    x = np.arange(len(models))
    width = 0.35
    for i, pat in enumerate(patterns):
        sub = merged[merged["pattern"] == pat].sort_values("model")
        offset = (i - 0.5) * width
        ax.bar(x + offset, sub["drop_pp"], width=width, label=f"{pat} pattern",
               color=PATTERN_COLORS[pat], alpha=0.85)
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.text(j + offset, row["drop_pp"] + 0.5, f"{row['drop_pp']:.1f}pp",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=10)
    ax.set_ylabel("Accuracy Drop (percentage points)", fontsize=12)
    ax.set_title("Accuracy Drop at 50% N:M Sparsity vs Dense Baseline", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_semi_co2_comparison(df: pd.DataFrame, out: Path) -> None:
    """CO2 comparison: dense vs pruned, both patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    models = sorted(df["model"].unique())
    patterns = sorted(df["pattern"].unique())
    x = np.arange(len(patterns))
    width = 0.35
    for ax, model in zip(axes, models):
        dense_co2, pruned_co2 = [], []
        for pat in patterns:
            sub = df[(df["model"] == model) & (df["pattern"] == pat)]
            dense_co2.append(sub.loc[sub["sparsity_requested"] == 0, "co2_g"].iloc[0])
            pruned_co2.append(sub.loc[sub["sparsity_requested"] == 50, "co2_g"].iloc[0])
        ax.bar(x - width / 2, dense_co2, width, label="Dense (0%)", color="#7fcdbb", alpha=0.9)
        ax.bar(x + width / 2, pruned_co2, width, label="Pruned (50%)",
               color=[PATTERN_COLORS[p] for p in patterns], alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(patterns, fontsize=12)
        ax.set_xlabel("N:M Pattern"); ax.set_ylabel("CO₂ (g CO₂eq)")
        ax.set_title(LABELS[model], fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Carbon Emissions — Dense vs N:M Pruned", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_semi_dashboard(df: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # A — accuracy bar
    ax1 = fig.add_subplot(gs[0, 0])
    pruned50 = df[df["sparsity_requested"] == 50]
    for i, (model, grp) in enumerate(pruned50.groupby("model")):
        for j, (_, row) in enumerate(grp.sort_values("pattern").iterrows()):
            pat = row["pattern"]
            x_pos = i * 3 + j
            ax1.bar(x_pos, row["accuracy"], color=PATTERN_COLORS[pat], alpha=0.85, width=0.7)
            ax1.text(x_pos, row["accuracy"] + 0.005, f"{row['accuracy']:.3f}",
                     ha="center", va="bottom", fontsize=8)
    ax1.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random")
    ax1.set_xticks([0, 1, 3, 4])
    ax1.set_xticklabels(["L 2:4", "L 4:8", "Q 2:4", "Q 4:8"], fontsize=9)
    ax1.set_ylabel("MMLU Accuracy"); ax1.set_title("A — Accuracy at 50% Sparsity", fontweight="bold")
    ax1.set_ylim(0.15, 0.45); ax1.grid(True, alpha=0.3, axis="y")

    # Add legend patches
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color=PATTERN_COLORS["2:4"], label="2:4 pattern"),
        Patch(color=PATTERN_COLORS["4:8"], label="4:8 pattern"),
    ], fontsize=8)

    # B — accuracy retained
    ax2 = fig.add_subplot(gs[0, 1])
    pruned_r = df[df["sparsity_requested"] == 50]
    for i, (model, grp) in enumerate(pruned_r.groupby("model")):
        for j, (_, row) in enumerate(grp.sort_values("pattern").iterrows()):
            x_pos = i * 3 + j
            ax2.bar(x_pos, row["accuracy_retained"], color=PATTERN_COLORS[row["pattern"]],
                    alpha=0.85, width=0.7)
            ax2.text(x_pos, row["accuracy_retained"] + 0.5, f"{row['accuracy_retained']:.1f}%",
                     ha="center", va="bottom", fontsize=8)
    ax2.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense baseline")
    ax2.axhline(50, color="red", linestyle=":", linewidth=1, label="50% retained")
    ax2.set_xticks([0, 1, 3, 4])
    ax2.set_xticklabels(["L 2:4", "L 4:8", "Q 2:4", "Q 4:8"], fontsize=9)
    ax2.set_ylabel("Accuracy Retained (%)"); ax2.set_title("B — Accuracy Retained at 50%", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y"); ax2.legend(fontsize=8)

    # C — CO2 comparison
    ax3 = fig.add_subplot(gs[1, 0])
    dense_df = df[df["sparsity_requested"] == 0]
    pruned_df = df[df["sparsity_requested"] == 50]
    combos = [(m, p) for m in sorted(df["model"].unique()) for p in sorted(df["pattern"].unique())]
    xs = np.arange(len(combos))
    dense_co2 = [dense_df[(dense_df["model"] == m) & (dense_df["pattern"] == p)]["co2_g"].iloc[0]
                 for m, p in combos]
    pruned_co2 = [pruned_df[(pruned_df["model"] == m) & (pruned_df["pattern"] == p)]["co2_g"].iloc[0]
                  for m, p in combos]
    w = 0.35
    ax3.bar(xs - w / 2, dense_co2, w, label="Dense (0%)", color="#7fcdbb", alpha=0.9)
    ax3.bar(xs + w / 2, pruned_co2, w, label="Pruned (50%)", color="#fd8d3c", alpha=0.9)
    ax3.set_xticks(xs)
    ax3.set_xticklabels(["L 2:4", "L 4:8", "Q 2:4", "Q 4:8"], fontsize=9)
    ax3.set_ylabel("CO₂ (g CO₂eq)"); ax3.set_title("C — Carbon Emissions", fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")

    # D — accuracy drop heatmap-style
    ax4 = fig.add_subplot(gs[1, 1])
    merged = pruned_df.merge(
        dense_df[["pattern", "model", "accuracy"]].rename(columns={"accuracy": "baseline"}),
        on=["pattern", "model"]
    )
    merged["drop_pp"] = (merged["baseline"] - merged["accuracy"]) * 100
    for i, (model, grp) in enumerate(merged.groupby("model")):
        for j, (_, row) in enumerate(grp.sort_values("pattern").iterrows()):
            x_pos = i * 3 + j
            ax4.bar(x_pos, row["drop_pp"], color=PATTERN_COLORS[row["pattern"]], alpha=0.85, width=0.7)
            ax4.text(x_pos, row["drop_pp"] + 0.3, f"{row['drop_pp']:.1f}pp",
                     ha="center", va="bottom", fontsize=8)
    ax4.set_xticks([0, 1, 3, 4])
    ax4.set_xticklabels(["L 2:4", "L 4:8", "Q 2:4", "Q 4:8"], fontsize=9)
    ax4.set_ylabel("Accuracy Drop (pp)"); ax4.set_title("D — Accuracy Drop at 50%", fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.legend(handles=[
        Patch(color=PATTERN_COLORS["2:4"], label="2:4 pattern"),
        Patch(color=PATTERN_COLORS["4:8"], label="4:8 pattern"),
    ], fontsize=8)

    fig.suptitle("Semi-Structured N:M Pruning — Accuracy & Carbon Dashboard",
                 fontsize=14, fontweight="bold", y=0.998)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base = Path("outputs/runs")
    structured_run = base / "mmlu_pruning_bd584ae1ae1c"
    semi_runs = {
        "2:4": base / "mmlu_pruning_dd7c17966f53",
        "4:8": base / "mmlu_pruning_b8eefdc94e41",
    }
    struct_out = Path("outputs/plots/structured")
    semi_out = Path("outputs/plots/semi_structured")
    struct_out.mkdir(parents=True, exist_ok=True)
    semi_out.mkdir(parents=True, exist_ok=True)

    print("=== Structured MLP plots ===")
    df_struct = load_structured(structured_run)
    plot_struct_accuracy(df_struct,                    struct_out / "accuracy_vs_sparsity.png")
    plot_struct_accuracy_retained(df_struct,           struct_out / "accuracy_retained.png")
    plot_struct_weight_vs_group_sparsity(df_struct,    struct_out / "weight_vs_group_sparsity.png")
    plot_struct_co2(df_struct,                         struct_out / "co2_vs_sparsity.png")
    plot_struct_groups_pruned(df_struct,               struct_out / "groups_pruned.png")
    plot_struct_dashboard(df_struct,                   struct_out / "dashboard.png")

    print("\n=== Semi-structured N:M plots ===")
    df_semi = load_semi_structured(semi_runs)
    plot_semi_accuracy_bar(df_semi,        semi_out / "accuracy_bar.png")
    plot_semi_accuracy_drop(df_semi,       semi_out / "accuracy_drop.png")
    plot_semi_co2_comparison(df_semi,      semi_out / "co2_comparison.png")
    plot_semi_dashboard(df_semi,           semi_out / "dashboard.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
