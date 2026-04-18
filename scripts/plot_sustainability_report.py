#!/usr/bin/env python
"""Generate cross-method sustainability and GPU acceleration plots."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    "Unstructured": "#4C72B0",
    "Structured MLP": "#DD8452",
    "Semi 2:4": "#2ca02c",
    "Semi 4:8": "#9467bd",
}
MODEL_MARKERS = {"llama31_8b_instruct": "o", "qwen3_8b": "s"}
MODEL_LABELS  = {"llama31_8b_instruct": "Llama-3.1-8B", "qwen3_8b": "Qwen3-8B"}
CHANCE = 0.25

BASE = Path("outputs/runs")
RUNS = {
    "Unstructured":   BASE / "mmlu_pruning_ade7d5ffbbb4",
    "Structured MLP": BASE / "mmlu_pruning_bd584ae1ae1c",
    "Semi 2:4":       BASE / "mmlu_pruning_dd7c17966f53",
    "Semi 4:8":       BASE / "mmlu_pruning_b8eefdc94e41",
}
MODELS = ["llama31_8b_instruct", "qwen3_8b"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all() -> pd.DataFrame:
    rows = []
    for method, run_dir in RUNS.items():
        for mpath in sorted(run_dir.rglob("metrics.json")):
            m = json.load(open(mpath))
            epath = mpath.parent / "emissions.json"
            e = json.load(open(epath)) if epath.exists() else {}
            rows.append({
                "method": method,
                "model": m["model_name"],
                "sparsity_requested": float(m["sparsity_requested"]),
                "sparsity_achieved": float(m.get("sparsity_achieved", 0)),
                "accuracy": float(m["accuracy"]),
                "co2_kg": float(m.get("emissions_kg_co2") or e.get("emissions_kg_co2", 0)),
                "energy_kwh": float(e.get("energy_consumed_kwh", 0)),
                "duration_s": float(e.get("duration_s", 0)),
                "gpu_power_w": float(e.get("gpu_power_w", 0)),
                "cpu_power_w": float(e.get("cpu_power_w", 0)),
                "ram_power_w": float(e.get("ram_power_w", 0)),
            })
    df = pd.DataFrame(rows)
    df["co2_g"] = df["co2_kg"] * 1000
    df["energy_wh"] = df["energy_kwh"] * 1000
    df["throughput"] = 14042 / df["duration_s"].replace(0, np.nan)

    # compute accuracy_retained per (method, model)
    baselines = (
        df[df["sparsity_requested"] == 0]
        .groupby(["method", "model"])["accuracy"]
        .mean()
        .rename("baseline")
    )
    df = df.merge(baselines.reset_index(), on=["method", "model"], how="left")
    df["accuracy_retained_pct"] = df["accuracy"] / df["baseline"] * 100
    df["carbon_efficiency"] = df["accuracy"] / df["co2_g"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Plot 1 — Cross-method CO2 at 50% sparsity
# ---------------------------------------------------------------------------

def plot_co2_at_50(df: pd.DataFrame, out: Path) -> None:
    sp50 = df[df["sparsity_requested"] == 50].copy()
    methods = list(METHOD_COLORS.keys())
    x = np.arange(len(MODELS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, method in enumerate(methods):
        sub = sp50[sp50["method"] == method].sort_values("model")
        co2_vals = [sub[sub["model"] == m]["co2_g"].values[0] if not sub[sub["model"] == m].empty else 0
                    for m in sorted(MODELS)]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, co2_vals, width, label=method,
                      color=METHOD_COLORS[method], alpha=0.87)
        for bar, val in zip(bars, co2_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                    f"{val:.1f}g", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in sorted(MODELS)], fontsize=12)
    ax.set_ylabel("CO₂ Emissions (g CO₂eq)", fontsize=12)
    ax.set_title("Carbon Cost per Pruning Method at 50% Sparsity", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Accuracy vs CO2 sustainability frontier (all methods, all sparsities)
# ---------------------------------------------------------------------------

def plot_sustainability_frontier(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, model in zip(axes, sorted(MODELS)):
        sub = df[df["model"] == model]
        for method in METHOD_COLORS:
            msub = sub[sub["method"] == method].sort_values("co2_g")
            if msub.empty:
                continue
            ax.scatter(msub["co2_g"], msub["accuracy"],
                       color=METHOD_COLORS[method], marker="o",
                       s=70, label=method, zorder=3, alpha=0.9)
            ax.plot(msub["co2_g"], msub["accuracy"],
                    color=METHOD_COLORS[method], linewidth=1, alpha=0.4)
            # annotate sparsity
            for _, row in msub.iterrows():
                ax.annotate(f"{int(row['sparsity_requested'])}%",
                            (row["co2_g"], row["accuracy"]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6.5, color=METHOD_COLORS[method])
        ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random (0.25)")
        ax.set_xlabel("CO₂ per Run (g CO₂eq)", fontsize=11)
        ax.set_ylabel("MMLU Accuracy", fontsize=11)
        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Sustainability Frontier: Accuracy vs Carbon Cost (all methods)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Carbon efficiency (accuracy / g CO2) at 50% sparsity
# ---------------------------------------------------------------------------

def plot_carbon_efficiency(df: pd.DataFrame, out: Path) -> None:
    sp50 = df[df["sparsity_requested"] == 50].copy()
    methods = list(METHOD_COLORS.keys())
    x = np.arange(len(MODELS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, method in enumerate(methods):
        sub = sp50[sp50["method"] == method].sort_values("model")
        eff = [sub[sub["model"] == m]["carbon_efficiency"].values[0]
               if not sub[sub["model"] == m].empty else 0
               for m in sorted(MODELS)]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, eff, width, label=method,
                      color=METHOD_COLORS[method], alpha=0.87)
        for bar, val in zip(bars, eff):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in sorted(MODELS)], fontsize=12)
    ax.set_ylabel("Accuracy per g CO₂eq", fontsize=12)
    ax.set_title("Carbon Efficiency at 50% Sparsity (higher = more sustainable)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Total sweep CO2 (entire experiment cost per method)
# ---------------------------------------------------------------------------

def plot_total_sweep_co2(df: pd.DataFrame, out: Path) -> None:
    totals = df.groupby("method")["co2_g"].sum().reset_index()
    totals["energy_wh"] = df.groupby("method")["energy_wh"].sum().values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # CO2
    ax = axes[0]
    bars = ax.bar(totals["method"], totals["co2_g"],
                  color=[METHOD_COLORS[m] for m in totals["method"]], alpha=0.87)
    for bar, val in zip(bars, totals["co2_g"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 10,
                f"{val:.0f} g", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total CO₂ (g CO₂eq)", fontsize=12)
    ax.set_title("Total Carbon Footprint of Full Sweep", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Energy
    ax2 = axes[1]
    bars2 = ax2.bar(totals["method"], totals["energy_wh"],
                    color=[METHOD_COLORS[m] for m in totals["method"]], alpha=0.87)
    for bar, val in zip(bars2, totals["energy_wh"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 10,
                 f"{val:.0f} Wh", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Total Energy Consumed (Wh)", fontsize=12)
    ax2.set_title("Total Energy Consumed by Full Sweep", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Total Experiment Cost — CO₂ and Energy per Method", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 5 — GPU power and throughput observed during evaluation
# ---------------------------------------------------------------------------

def plot_gpu_and_throughput(df: pd.DataFrame, out: Path) -> None:
    sp50 = df[df["sparsity_requested"] == 50].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = list(METHOD_COLORS.keys())
    x = np.arange(len(MODELS))
    width = 0.18

    # GPU power
    ax = axes[0]
    for i, method in enumerate(methods):
        sub = sp50[sp50["method"] == method].sort_values("model")
        gpu_w = [sub[sub["model"] == m]["gpu_power_w"].values[0]
                 if not sub[sub["model"] == m].empty else 0
                 for m in sorted(MODELS)]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, gpu_w, width, label=method,
                      color=METHOD_COLORS[method], alpha=0.87)
        for bar, val in zip(bars, gpu_w):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f"{val:.0f}W", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in sorted(MODELS)], fontsize=11)
    ax.set_ylabel("Average GPU Power Draw (W)", fontsize=11)
    ax.set_title("GPU Power Draw at 50% Sparsity", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # Throughput (samples/sec)
    ax2 = axes[1]
    for i, method in enumerate(methods):
        sub = sp50[sp50["method"] == method].sort_values("model")
        tput = [sub[sub["model"] == m]["throughput"].values[0]
                if not sub[sub["model"] == m].empty else 0
                for m in sorted(MODELS)]
        offset = (i - 1.5) * width
        bars = ax2.bar(x + offset, tput, width, label=method,
                       color=METHOD_COLORS[method], alpha=0.87)
        for bar, val in zip(bars, tput):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS[m] for m in sorted(MODELS)], fontsize=11)
    ax2.set_ylabel("Observed Throughput (samples / sec)", fontsize=11)
    ax2.set_title("Evaluation Throughput at 50% Sparsity", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("GPU Utilisation & Throughput During Evaluation (50% sparsity)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 6 — Theoretical GPU speedup (hardware acceleration potential)
# ---------------------------------------------------------------------------

def plot_theoretical_speedup(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    methods = [
        "Unstructured\n(no sparse kernel)",
        "Structured MLP\n(materialized,\nchannel reduction)",
        "Semi 2:4\n(cuSPARSELt,\nA100/H100)",
        "Semi 4:8\n(custom kernel\nrequired)",
    ]
    # Lower / upper bounds for each method
    speedup_low  = [1.0, 1.5, 1.8, 1.2]
    speedup_high = [1.0, 3.0, 2.0, 1.6]
    speedup_mid  = [1.0, 2.0, 2.0, 1.4]   # midpoint / nominal

    colors = list(METHOD_COLORS.values())
    x = np.arange(len(methods))

    bars = ax.bar(x, speedup_mid, color=colors, alpha=0.85, width=0.55, zorder=2)
    # error bars for range
    err_lo = [m - l for m, l in zip(speedup_mid, speedup_low)]
    err_hi = [h - m for h, m in zip(speedup_high, speedup_mid)]
    ax.errorbar(x, speedup_mid, yerr=[err_lo, err_hi], fmt="none",
                color="black", capsize=6, capthick=1.5, elinewidth=1.5, zorder=3)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Dense baseline (1×)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10.5)
    ax.set_ylabel("Theoretical Inference Speedup (×)", fontsize=12)
    ax.set_title("Theoretical GPU Acceleration per Pruning Method\n"
                 "(error bars = expected range; requires hardware-specific sparse kernels)", fontsize=12)
    ax.set_ylim(0, 3.8)
    ax.grid(True, alpha=0.3, axis="y", zorder=1)

    # annotate bars
    notes = [
        "Dense matmul\nunchanged",
        "Reduced hidden dim\n(arch surgery needed)",
        "2× Tensor Core\nthroughput (NVIDIA)",
        "Non-standard;\ncustom kernel",
    ]
    for i, (bar, note) in enumerate(zip(bars, notes)):
        ax.text(bar.get_x() + bar.get_width() / 2, speedup_mid[i] + err_hi[i] + 0.12,
                note, ha="center", va="bottom", fontsize=8, style="italic", color="#333333")

    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 7 — Accuracy retained vs CO2 saved bubble chart (composite view)
# ---------------------------------------------------------------------------

def plot_accuracy_vs_co2_saved(df: pd.DataFrame, out: Path) -> None:
    """
    X = CO2 saved vs unstructured at same sparsity (g)
    Y = accuracy retained (%)
    Bubble size = throughput (samples/sec)
    """
    sp50 = df[df["sparsity_requested"] == 50].copy()
    unstructured_co2 = {
        model: sp50[(sp50["method"] == "Unstructured") & (sp50["model"] == model)]["co2_g"].values[0]
        for model in MODELS
    }
    unstructured_acc = {
        model: sp50[(sp50["method"] == "Unstructured") & (sp50["model"] == model)]["accuracy"].values[0]
        for model in MODELS
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model in zip(axes, sorted(MODELS)):
        for method in METHOD_COLORS:
            sub = sp50[(sp50["method"] == method) & (sp50["model"] == model)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            co2_saved = unstructured_co2[model] - row["co2_g"]
            acc_ret = row["accuracy_retained_pct"]
            tput = row["throughput"] if not np.isnan(row["throughput"]) else 5.0
            sc = ax.scatter(co2_saved, acc_ret, s=tput * 25,
                            color=METHOD_COLORS[method], alpha=0.85,
                            edgecolors="black", linewidths=0.8, zorder=3, label=method)
            ax.annotate(method, (co2_saved, acc_ret),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8.5, color=METHOD_COLORS[method])

        ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense baseline")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("CO₂ Saved vs Unstructured (g CO₂eq, higher = greener)", fontsize=10)
        ax.set_ylabel("Accuracy Retained vs Dense Baseline (%)", fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Sustainability Quadrant: Accuracy Retained vs CO₂ Saved\n"
                 "(bubble size = observed throughput in samples/sec)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 8 — Master sustainability dashboard
# ---------------------------------------------------------------------------

def plot_master_dashboard(df: pd.DataFrame, out: Path) -> None:
    sp50 = df[df["sparsity_requested"] == 50].copy()
    methods = list(METHOD_COLORS.keys())

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    # ---- A: Accuracy at 50% ----
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(sorted(MODELS)))
    width = 0.18
    for i, method in enumerate(methods):
        sub = sp50[sp50["method"] == method].sort_values("model")
        vals = [sub[sub["model"] == m]["accuracy"].values[0] if not sub[sub["model"] == m].empty else 0
                for m in sorted(MODELS)]
        ax_a.bar(x + (i - 1.5) * width, vals, width, label=method, color=METHOD_COLORS[method], alpha=0.87)
    ax_a.axhline(CHANCE, color="gray", linestyle="--", linewidth=1)
    ax_a.set_xticks(x); ax_a.set_xticklabels(["Llama", "Qwen3"], fontsize=10)
    ax_a.set_ylabel("MMLU Accuracy"); ax_a.set_title("A — Accuracy @ 50%", fontweight="bold")
    ax_a.set_ylim(0.15, 0.45); ax_a.legend(fontsize=7); ax_a.grid(True, alpha=0.3, axis="y")

    # ---- B: CO2 at 50% ----
    ax_b = fig.add_subplot(gs[0, 1])
    for i, method in enumerate(methods):
        sub = sp50[sp50["method"] == method].sort_values("model")
        vals = [sub[sub["model"] == m]["co2_g"].values[0] if not sub[sub["model"] == m].empty else 0
                for m in sorted(MODELS)]
        ax_b.bar(x + (i - 1.5) * width, vals, width, color=METHOD_COLORS[method], alpha=0.87)
    ax_b.set_xticks(x); ax_b.set_xticklabels(["Llama", "Qwen3"], fontsize=10)
    ax_b.set_ylabel("CO₂ (g CO₂eq)"); ax_b.set_title("B — Carbon Cost @ 50%", fontweight="bold")
    ax_b.grid(True, alpha=0.3, axis="y")
    # manual legend
    patches = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in methods]
    ax_b.legend(handles=patches, fontsize=7)

    # ---- C: Carbon efficiency at 50% ----
    ax_c = fig.add_subplot(gs[0, 2])
    for i, method in enumerate(methods):
        sub = sp50[sp50["method"] == method].sort_values("model")
        vals = [sub[sub["model"] == m]["carbon_efficiency"].values[0] if not sub[sub["model"] == m].empty else 0
                for m in sorted(MODELS)]
        ax_c.bar(x + (i - 1.5) * width, vals, width, color=METHOD_COLORS[method], alpha=0.87)
    ax_c.set_xticks(x); ax_c.set_xticklabels(["Llama", "Qwen3"], fontsize=10)
    ax_c.set_ylabel("Accuracy / g CO₂"); ax_c.set_title("C — Carbon Efficiency @ 50%", fontweight="bold")
    ax_c.grid(True, alpha=0.3, axis="y")

    # ---- D: Sustainability frontier (Llama) ----
    ax_d = fig.add_subplot(gs[1, 0:2])
    for method in methods:
        sub = df[(df["model"] == "llama31_8b_instruct") & (df["method"] == method)].sort_values("co2_g")
        if sub.empty: continue
        ax_d.plot(sub["co2_g"], sub["accuracy"], marker="o", linewidth=1.5,
                  color=METHOD_COLORS[method], label=method, alpha=0.9)
        for _, row in sub.iterrows():
            ax_d.annotate(f"{int(row['sparsity_requested'])}%", (row["co2_g"], row["accuracy"]),
                          textcoords="offset points", xytext=(3, 3), fontsize=6, color=METHOD_COLORS[method])
    ax_d.axhline(CHANCE, color="gray", linestyle="--", linewidth=1)
    ax_d.set_xlabel("CO₂ (g CO₂eq)"); ax_d.set_ylabel("MMLU Accuracy")
    ax_d.set_title("D — Sustainability Frontier: Llama-3.1-8B", fontweight="bold")
    ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.3)

    # ---- E: Sustainability frontier (Qwen3) ----
    ax_e = fig.add_subplot(gs[1, 2])
    for method in methods:
        sub = df[(df["model"] == "qwen3_8b") & (df["method"] == method)].sort_values("co2_g")
        if sub.empty: continue
        ax_e.plot(sub["co2_g"], sub["accuracy"], marker="s", linewidth=1.5,
                  color=METHOD_COLORS[method], label=method, alpha=0.9)
    ax_e.axhline(CHANCE, color="gray", linestyle="--", linewidth=1)
    ax_e.set_xlabel("CO₂ (g CO₂eq)"); ax_e.set_ylabel("MMLU Accuracy")
    ax_e.set_title("E — Sustainability Frontier: Qwen3-8B", fontweight="bold")
    ax_e.legend(fontsize=7); ax_e.grid(True, alpha=0.3)

    # ---- F: Total sweep CO2 ----
    ax_f = fig.add_subplot(gs[2, 0])
    total_co2 = df.groupby("method")["co2_g"].sum()
    method_order = list(METHOD_COLORS.keys())
    vals = [total_co2.get(m, 0) for m in method_order]
    bars = ax_f.bar(method_order, vals, color=[METHOD_COLORS[m] for m in method_order], alpha=0.87)
    for bar, val in zip(bars, vals):
        ax_f.text(bar.get_x() + bar.get_width() / 2, val + 5, f"{val:.0f}g",
                  ha="center", va="bottom", fontsize=8.5)
    ax_f.set_ylabel("CO₂ (g)"); ax_f.set_title("F — Total Sweep CO₂", fontweight="bold")
    ax_f.tick_params(axis="x", labelsize=8); ax_f.grid(True, alpha=0.3, axis="y")

    # ---- G: GPU power ----
    ax_g = fig.add_subplot(gs[2, 1])
    for i, method in enumerate(methods):
        sub = sp50[sp50["method"] == method].sort_values("model")
        vals = [sub[sub["model"] == m]["gpu_power_w"].values[0] if not sub[sub["model"] == m].empty else 0
                for m in sorted(MODELS)]
        ax_g.bar(x + (i - 1.5) * width, vals, width, color=METHOD_COLORS[method], alpha=0.87)
    ax_g.set_xticks(x); ax_g.set_xticklabels(["Llama", "Qwen3"], fontsize=10)
    ax_g.set_ylabel("Avg GPU Power (W)"); ax_g.set_title("G — GPU Power @ 50%", fontweight="bold")
    ax_g.grid(True, alpha=0.3, axis="y")

    # ---- H: Throughput ----
    ax_h = fig.add_subplot(gs[2, 2])
    for i, method in enumerate(methods):
        sub = sp50[sp50["method"] == method].sort_values("model")
        vals = [sub[sub["model"] == m]["throughput"].values[0] if not sub[sub["model"] == m].empty else 0
                for m in sorted(MODELS)]
        ax_h.bar(x + (i - 1.5) * width, vals, width, color=METHOD_COLORS[method], alpha=0.87)
    ax_h.set_xticks(x); ax_h.set_xticklabels(["Llama", "Qwen3"], fontsize=10)
    ax_h.set_ylabel("Samples / sec"); ax_h.set_title("H — Throughput @ 50%", fontweight="bold")
    ax_h.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Sustainability & GPU Acceleration — Master Dashboard\n(All Pruning Methods · MMLU · H100)",
                 fontsize=15, fontweight="bold", y=1.005)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out = Path("outputs/plots/sustainability")
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_all()
    print(f"Loaded {len(df)} rows\n")

    print("Generating plots...")
    plot_co2_at_50(df,                    out / "co2_at_50pct.png")
    plot_sustainability_frontier(df,      out / "sustainability_frontier.png")
    plot_carbon_efficiency(df,            out / "carbon_efficiency.png")
    plot_total_sweep_co2(df,              out / "total_sweep_cost.png")
    plot_gpu_and_throughput(df,           out / "gpu_and_throughput.png")
    plot_theoretical_speedup(             out / "theoretical_speedup.png")
    plot_accuracy_vs_co2_saved(df,        out / "accuracy_vs_co2_saved.png")
    plot_master_dashboard(df,             out / "master_dashboard.png")
    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    main()
