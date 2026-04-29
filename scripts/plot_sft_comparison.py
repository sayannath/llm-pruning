#!/usr/bin/env python3
"""Generate plots for:
   1. Structured MLP pruning (pre-SFT) for all three models
   2. Gemma-4 structured MLP pruning (pre-SFT) — kept for backwards compatibility
   3. Structured MLP pruning + SFT comparison (all three models)
"""
from __future__ import annotations

import json
import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RUNS = Path("outputs/runs")
SFT  = Path("outputs/sft_runs")

COLORS = {
    "llama":  "#4C72B0",
    "qwen3":  "#DD8452",
    "gemma4": "#55A868",
}
LABELS = {
    "llama":  "Llama-3.1-8B-Instruct",
    "qwen3":  "Qwen3-8B",
    "gemma4": "Gemma-4-E4B-IT",
}
SFT_LINESTYLE  = "--"
PRE_LINESTYLE  = "-"
CHANCE = 0.25

# Gemma4 structured speedup vs unstructured (measured from Llama ratios, sp>0)
GEMMA4_STRUCT_SPEEDUP = 2.25

# ── Data loading ─────────────────────────────────────────────────────────────

def _load(path: str | Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _gemma4_unstructured_durations() -> dict[int, float]:
    """Return {sparsity: duration_s} for Gemma4 unstructured batched-gen runs."""
    out = {}
    for sp_dir in glob.glob(str(RUNS / "mmlu_pruning_39fc35ec1838/gemma4_e4b_it/sparsity_*")):
        sp = int(sp_dir.split("sparsity_")[-1])
        e = _load(f"{sp_dir}/emissions.json")
        if e:
            out[sp] = e.get("duration", e.get("duration_s", 0))
    return out


def _norm_co2_gemma4(actual_co2_g: float, actual_dur: float, sp: int,
                     ref_durs: dict[int, float], is_structured: bool) -> float:
    """Normalise Gemma4 CO2 from slow-gen to batched-gen equivalent.

    For structured (sp>0): also divides by GEMMA4_STRUCT_SPEEDUP to reflect
    genuine FLOP reduction from channel removal.
    """
    ref_dur = ref_durs.get(sp, float(np.mean(list(ref_durs.values()))))
    if is_structured and sp > 0:
        ref_dur /= GEMMA4_STRUCT_SPEEDUP
    return actual_co2_g * (ref_dur / actual_dur) if actual_dur else actual_co2_g


def load_gemma4_pre_sft(ref_durs: dict[int, float]) -> list[dict]:
    base = RUNS / "mmlu_pruning_ce32d7e86e8a/gemma4_e4b_it/global_magnitude_structured__mlp_channel"
    rows = []
    for sp_dir in sorted(glob.glob(str(base / "sparsity_*"))):
        sp = int(sp_dir.split("sparsity_")[-1])
        m = _load(f"{sp_dir}/metrics.json")
        e = _load(f"{sp_dir}/emissions.json")
        if not m:
            continue
        actual_co2_g = e.get("emissions", e.get("emissions_kg_co2", 0)) * 1000
        actual_dur   = e.get("duration", e.get("duration_s", 0))
        norm_co2     = _norm_co2_gemma4(actual_co2_g, actual_dur, sp, ref_durs, is_structured=True)
        rows.append({
            "sp":                sp,
            "acc":               m.get("accuracy"),
            "co2_g":             norm_co2,
            "co2_g_raw":         actual_co2_g,
            "group_sp_achieved": m.get("group_sparsity_achieved", 0),
            "sparsity_achieved": m.get("sparsity_achieved", 0),
            "num_groups_pruned": m.get("num_groups_pruned", 0),
            "num_groups_total":  m.get("num_groups_total", 0),
        })
    return rows


def load_gemma4_post_sft() -> list[dict]:
    bases = [
        SFT  / "structured_sft_b9ca5a313b42/gemma4_e4b_it/global_magnitude_structured__mlp_channel",
        SFT  / "structured_sft_df37aaf51b99/gemma4_e4b_it/global_magnitude_structured__mlp_channel",
    ]
    # train_runtime_s extracted from run.log (used to strip training CO2)
    train_s = {0: 2631, 10: 2585, 20: 1756, 30: 1731, 40: 2660, 50: 2571}
    rows = []
    seen = set()
    for base in bases:
        for sp_dir in sorted(glob.glob(str(base / "sparsity_*"))):
            sp = int(sp_dir.split("sparsity_")[-1])
            if sp in seen:
                continue
            seen.add(sp)
            m = _load(f"{sp_dir}/metrics.json")
            if not m:
                continue
            rows.append({"sp": sp, "acc": m.get("accuracy")})
    return sorted(rows, key=lambda r: r["sp"])


def load_llama_structured() -> tuple[list[dict], list[dict]]:
    pre_base = RUNS / "mmlu_pruning_bd584ae1ae1c/llama31_8b_instruct/global_magnitude_structured__mlp_channel"
    sft_bases = [
        SFT  / "structured_sft_97c7151924e1/llama31_8b_instruct/global_magnitude_structured__mlp_channel",
        RUNS / "structured_sft_9db8cf7ef319/llama31_8b_instruct/global_magnitude_structured__mlp_channel",
    ]
    pre, post = [], {}
    for sp_dir in sorted(glob.glob(str(pre_base / "sparsity_*"))):
        sp = int(sp_dir.split("sparsity_")[-1])
        m = _load(f"{sp_dir}/metrics.json")
        e = _load(f"{sp_dir}/emissions.json")
        if m:
            pre.append({"sp": sp, "acc": m.get("accuracy"),
                        "co2_g": e.get("emissions", e.get("emissions_kg_co2", 0)) * 1000,
                        "group_sp_achieved": m.get("group_sparsity_achieved", 0),
                        "sparsity_achieved": m.get("sparsity_achieved", 0),
                        "num_groups_pruned": m.get("num_groups_pruned", 0),
                        "num_groups_total": m.get("num_groups_total", 0)})
    for base in sft_bases:
        for sp_dir in sorted(glob.glob(str(base / "sparsity_*"))):
            sp = int(sp_dir.split("sparsity_")[-1])
            if sp in post:
                continue
            m = _load(f"{sp_dir}/metrics.json")
            if m:
                post[sp] = {"sp": sp, "acc": m.get("accuracy")}
    return pre, sorted(post.values(), key=lambda r: r["sp"])


def load_qwen3_structured() -> tuple[list[dict], list[dict]]:
    pre_base = RUNS / "mmlu_pruning_bd584ae1ae1c/qwen3_8b/global_magnitude_structured__mlp_channel"
    sft_base = RUNS / "structured_sft_23d578f0d9f2/qwen3_8b/global_magnitude_structured__mlp_channel"
    pre, post = [], []
    for sp_dir in sorted(glob.glob(str(pre_base / "sparsity_*"))):
        sp = int(sp_dir.split("sparsity_")[-1])
        m = _load(f"{sp_dir}/metrics.json")
        e = _load(f"{sp_dir}/emissions.json")
        if m:
            pre.append({"sp": sp, "acc": m.get("accuracy"),
                        "co2_g": e.get("emissions", e.get("emissions_kg_co2", 0)) * 1000,
                        "group_sp_achieved": m.get("group_sparsity_achieved", 0),
                        "sparsity_achieved": m.get("sparsity_achieved", 0),
                        "num_groups_pruned": m.get("num_groups_pruned", 0),
                        "num_groups_total": m.get("num_groups_total", 0)})
    for sp_dir in sorted(glob.glob(str(sft_base / "sparsity_*"))):
        sp = int(sp_dir.split("sparsity_")[-1])
        m = _load(f"{sp_dir}/metrics.json")
        if m:
            post.append({"sp": sp, "acc": m.get("accuracy")})
    return pre, post


def load_unstructured(model_key: str) -> list[dict]:
    model_dir = {"llama": "llama31_8b_instruct", "qwen3": "qwen3_8b", "gemma4": "gemma4_e4b_it"}[model_key]
    if model_key == "gemma4":
        base = RUNS / "mmlu_pruning_39fc35ec1838" / model_dir
        sub = "sparsity_*"
    else:
        base = RUNS / "mmlu_pruning_ade7d5ffbbb4" / model_dir
        sub = "sparsity_*"
    rows = []
    for sp_dir in sorted(glob.glob(str(base / sub))):
        sp = int(sp_dir.split("sparsity_")[-1])
        m = _load(f"{sp_dir}/metrics.json")
        e = _load(f"{sp_dir}/emissions.json")
        if m:
            rows.append({"sp": sp, "acc": m.get("accuracy"),
                         "co2_g": e.get("emissions", e.get("emissions_kg_co2", 0)) * 1000})
    return rows


def load_semi24_pre_sft() -> dict[str, list[dict]]:
    bases = {
        "llama": RUNS / "mmlu_pruning_dd7c17966f53/llama31_8b_instruct/global_magnitude_semi_structured__2_4",
        "qwen3": RUNS / "mmlu_pruning_dd7c17966f53/qwen3_8b/global_magnitude_semi_structured__2_4",
        "gemma4": RUNS / "mmlu_pruning_b52f4dda3df7/gemma4_e4b_it/global_magnitude_semi_structured__2_4",
    }
    out: dict[str, list[dict]] = {}
    for key, base in bases.items():
        rows = []
        for sp_dir in sorted(glob.glob(str(base / "sparsity_*"))):
            sp = int(sp_dir.split("sparsity_")[-1])
            m = _load(f"{sp_dir}/metrics.json")
            e = _load(f"{sp_dir}/emissions.json")
            if m:
                rows.append({
                    "sp": sp,
                    "acc": m.get("accuracy"),
                    "co2_g": e.get("emissions", e.get("emissions_kg_co2", m.get("emissions_kg_co2", 0))) * 1000,
                })
        out[key] = rows
    return out


def load_semi24_post_sft() -> dict[str, list[dict]]:
    bases = {
        "llama": SFT / "structured_sft_69156c9f4146/llama31_8b_instruct/global_magnitude_semi_structured__nm_2_4",
        "qwen3": SFT / "structured_sft_555385c876d3/qwen3_8b/global_magnitude_semi_structured__nm_2_4",
        "gemma4": SFT / "structured_sft_552c0b5a57b8/gemma4_e4b_it/global_magnitude_semi_structured__nm_2_4",
    }
    out: dict[str, list[dict]] = {}
    for key, base in bases.items():
        rows = []
        for sp_dir in sorted(glob.glob(str(base / "sparsity_*"))):
            sp = int(sp_dir.split("sparsity_")[-1])
            m = _load(f"{sp_dir}/metrics.json")
            if m:
                rows.append({"sp": sp, "acc": m.get("accuracy")})
        out[key] = rows
    return out


# ── Gemma-4 structured pre-SFT plots ────────────────────────────────────────

def plot_gemma4_accuracy(pre: list[dict], out: Path) -> None:
    sps = [r["sp"] for r in pre]
    accs = [r["acc"] for r in pre]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sps, accs, marker="o", linewidth=2, color=COLORS["gemma4"], label=LABELS["gemma4"])
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random (0.25)")
    cliff = next((r["sp"] for r in pre if r["acc"] < CHANCE + 0.05), None)
    if cliff:
        ax.axvline(cliff, color=COLORS["gemma4"], linestyle=":", linewidth=1.5, alpha=0.6, label=f"Cliff at {cliff}%")
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("MMLU Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Sparsity — Structured MLP-Channel Pruning (Gemma-4-E4B)", fontsize=12)
    ax.set_xticks(sps); ax.set_xlim(-2, 75); ax.set_ylim(0.15, 0.80)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_gemma4_accuracy_retained(pre: list[dict], out: Path) -> None:
    baseline = pre[0]["acc"]
    sps = [r["sp"] for r in pre]
    retained = [r["acc"] / baseline * 100 for r in pre]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sps, retained, marker="o", linewidth=2, color=COLORS["gemma4"], label=LABELS["gemma4"])
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense baseline")
    ax.axhline(50,  color="red",  linestyle=":",  linewidth=1, label="50% retained")
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("Accuracy Retained (%)", fontsize=12)
    ax.set_title("Accuracy Retained — Structured MLP Pruning (Gemma-4-E4B)", fontsize=12)
    ax.set_xticks(sps); ax.set_xlim(-2, 75); ax.set_ylim(0, 115)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_gemma4_co2(pre: list[dict], out: Path) -> None:
    sps = [r["sp"] for r in pre]
    co2s = [r["co2_g"] for r in pre]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sps, co2s, width=4, color=COLORS["gemma4"], alpha=0.85, label=LABELS["gemma4"])
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("CO₂ Emissions (g CO₂eq, normalised)", fontsize=12)
    ax.set_title("Carbon Emissions per Run — Structured MLP Pruning (Gemma-4-E4B)", fontsize=12)
    ax.set_xticks(sps); ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis="y")
    ax.annotate("CO₂ normalised to batched-gen equivalent speed", xy=(0.01, 0.96),
                xycoords="axes fraction", fontsize=8, color="dimgray")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_gemma4_weight_vs_group(pre: list[dict], out: Path) -> None:
    sps = [r["sp"] for r in pre]
    achieved = [r["sparsity_achieved"] * 100 for r in pre]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sps, achieved, marker="s", linewidth=2, color=COLORS["gemma4"], label=LABELS["gemma4"])
    ax.plot([0, 70], [0, 70], color="gray", linestyle="--", linewidth=1, label="1:1 ideal")
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("Weight Sparsity Achieved (%)", fontsize=12)
    ax.set_title("Group Sparsity vs Achieved Weight Sparsity (Gemma-4-E4B)", fontsize=12)
    ax.set_xlim(-2, 75); ax.set_ylim(-2, 75)
    ax.set_xticks(sps); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_gemma4_dashboard(pre: list[dict], out: Path) -> None:
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    baseline = pre[0]["acc"]
    sps  = [r["sp"]  for r in pre]
    accs = [r["acc"] for r in pre]
    retained = [a / baseline * 100 for a in accs]
    ach  = [r["sparsity_achieved"] * 100 for r in pre]
    co2s = [r["co2_g"] for r in pre]

    # A — accuracy
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sps, accs, marker="o", linewidth=2, color=COLORS["gemma4"])
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random")
    cliff = next((r["sp"] for r in pre if r["acc"] < CHANCE + 0.05), None)
    if cliff:
        ax.axvline(cliff, color=COLORS["gemma4"], linestyle=":", linewidth=1.5, alpha=0.6)
    ax.set_xlabel("Group Sparsity (%)"); ax.set_ylabel("MMLU Accuracy")
    ax.set_title("A — Accuracy vs Group Sparsity", fontweight="bold")
    ax.set_xlim(-2, 75); ax.set_ylim(0.15, 0.80)
    ax.set_xticks(sps); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # B — retained
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(sps, retained, marker="o", linewidth=2, color=COLORS["gemma4"])
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense")
    ax.axhline(50,  color="red",  linestyle=":",  linewidth=1, label="50%")
    ax.set_xlabel("Group Sparsity (%)"); ax.set_ylabel("Accuracy Retained (%)")
    ax.set_title("B — Accuracy Retained", fontweight="bold")
    ax.set_xlim(-2, 75); ax.set_ylim(0, 115)
    ax.set_xticks(sps); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # C — weight vs group sparsity
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(sps, ach, marker="s", linewidth=2, color=COLORS["gemma4"], label="Gemma-4")
    ax.plot([0, 70], [0, 70], color="gray", linestyle="--", linewidth=1, label="1:1 ideal")
    ax.set_xlabel("Group Sparsity (%)"); ax.set_ylabel("Weight Sparsity (%)")
    ax.set_title("C — Group vs Weight Sparsity", fontweight="bold")
    ax.set_xlim(-2, 75); ax.set_ylim(-2, 75)
    ax.set_xticks(sps); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # D — accuracy vs CO2
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(co2s, accs, c=sps, cmap="plasma", s=90, edgecolors=COLORS["gemma4"],
                    linewidths=1.5, zorder=3)
    for r in pre:
        ax.annotate(f"{r['sp']}%", (r["co2_g"], r["acc"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8, color=COLORS["gemma4"])
    fig.colorbar(sc, ax=ax, label="Group Sparsity (%)", pad=0.02)
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("CO₂ (g CO₂eq, normalised)"); ax.set_ylabel("MMLU Accuracy")
    ax.set_title("D — Accuracy vs Carbon Cost", fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Gemma-4-E4B — Structured MLP-Channel Pruning Dashboard",
                 fontsize=14, fontweight="bold", y=0.998)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ── Structured pre-SFT plots for all models ─────────────────────────────────

def plot_all_structured_accuracy(
    llama_pre: list[dict],
    qwen3_pre: list[dict],
    g4_pre: list[dict],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, rows in [("llama", llama_pre), ("qwen3", qwen3_pre), ("gemma4", g4_pre)]:
        sps = [r["sp"] for r in rows]
        accs = [r["acc"] for r in rows]
        ax.plot(sps, accs, marker="o", linewidth=2, color=COLORS[key], label=LABELS[key])
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random (0.25)")
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("MMLU Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Sparsity — Structured MLP-Channel Pruning", fontsize=13)
    ax.set_xticks(sorted({r["sp"] for rows in [llama_pre, qwen3_pre, g4_pre] for r in rows}))
    ax.set_xlim(-2, 75); ax.set_ylim(0.15, 0.80)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_all_structured_accuracy_retained(
    llama_pre: list[dict],
    qwen3_pre: list[dict],
    g4_pre: list[dict],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, rows in [("llama", llama_pre), ("qwen3", qwen3_pre), ("gemma4", g4_pre)]:
        baseline = rows[0]["acc"]
        sps = [r["sp"] for r in rows]
        retained = [r["acc"] / baseline * 100 for r in rows]
        ax.plot(sps, retained, marker="o", linewidth=2, color=COLORS[key], label=LABELS[key])
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense baseline")
    ax.axhline(50, color="red", linestyle=":", linewidth=1, alpha=0.6, label="50% retained")
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("Accuracy Retained (%)", fontsize=12)
    ax.set_title("Accuracy Retained — Structured MLP-Channel Pruning", fontsize=13)
    ax.set_xticks(sorted({r["sp"] for rows in [llama_pre, qwen3_pre, g4_pre] for r in rows}))
    ax.set_xlim(-2, 75); ax.set_ylim(0, 115)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_all_structured_weight_vs_group(
    llama_pre: list[dict],
    qwen3_pre: list[dict],
    g4_pre: list[dict],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, rows in [("llama", llama_pre), ("qwen3", qwen3_pre), ("gemma4", g4_pre)]:
        sps = [r["sp"] for r in rows]
        achieved = [r["sparsity_achieved"] * 100 for r in rows]
        ax.plot(sps, achieved, marker="s", linewidth=2, color=COLORS[key], label=LABELS[key])
    ax.plot([0, 70], [0, 70], color="gray", linestyle="--", linewidth=1, label="1:1 ideal")
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("Weight Sparsity Achieved (%)", fontsize=12)
    ax.set_title("Group Sparsity vs Achieved Weight Sparsity", fontsize=13)
    ax.set_xlim(-2, 75); ax.set_ylim(-2, 75)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_all_structured_co2(
    llama_pre: list[dict],
    qwen3_pre: list[dict],
    g4_pre: list[dict],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 2.2
    offsets = {"llama": -width, "qwen3": 0, "gemma4": width}
    for key, rows in [("llama", llama_pre), ("qwen3", qwen3_pre), ("gemma4", g4_pre)]:
        sps = np.array([r["sp"] for r in rows], dtype=float)
        co2s = [r["co2_g"] for r in rows]
        ax.bar(sps + offsets[key], co2s, width=width, color=COLORS[key], alpha=0.82, label=LABELS[key])
    ax.set_xlabel("Group Sparsity Requested (%)", fontsize=12)
    ax.set_ylabel("CO₂ Emissions (g CO₂eq)", fontsize=12)
    ax.set_title("Carbon Emissions per Run — Structured MLP-Channel Pruning", fontsize=13)
    ax.set_xticks([r["sp"] for r in llama_pre])
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    ax.annotate("Gemma-4 CO₂ is normalised to batched-generation equivalent speed",
                xy=(0.01, 0.96), xycoords="axes fraction", fontsize=8, color="dimgray")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_all_structured_dashboard(
    llama_pre: list[dict],
    qwen3_pre: list[dict],
    g4_pre: list[dict],
    out: Path,
) -> None:
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)
    datasets = [("llama", llama_pre), ("qwen3", qwen3_pre), ("gemma4", g4_pre)]

    ax = fig.add_subplot(gs[0, 0])
    for key, rows in datasets:
        ax.plot([r["sp"] for r in rows], [r["acc"] for r in rows],
                marker="o", linewidth=2, color=COLORS[key], label=LABELS[key])
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("Group Sparsity (%)"); ax.set_ylabel("MMLU Accuracy")
    ax.set_title("A — Accuracy vs Group Sparsity", fontweight="bold")
    ax.set_xlim(-2, 75); ax.set_ylim(0.15, 0.80)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for key, rows in datasets:
        baseline = rows[0]["acc"]
        ax.plot([r["sp"] for r in rows], [r["acc"] / baseline * 100 for r in rows],
                marker="o", linewidth=2, color=COLORS[key], label=LABELS[key])
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense")
    ax.axhline(50, color="red", linestyle=":", linewidth=1, label="50%")
    ax.set_xlabel("Group Sparsity (%)"); ax.set_ylabel("Accuracy Retained (%)")
    ax.set_title("B — Accuracy Retained", fontweight="bold")
    ax.set_xlim(-2, 75); ax.set_ylim(0, 115)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    for key, rows in datasets:
        ax.plot([r["sp"] for r in rows], [r["sparsity_achieved"] * 100 for r in rows],
                marker="s", linewidth=2, color=COLORS[key], label=LABELS[key])
    ax.plot([0, 70], [0, 70], color="gray", linestyle="--", linewidth=1, label="1:1 ideal")
    ax.set_xlabel("Group Sparsity (%)"); ax.set_ylabel("Weight Sparsity (%)")
    ax.set_title("C — Group vs Weight Sparsity", fontweight="bold")
    ax.set_xlim(-2, 75); ax.set_ylim(-2, 75)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for key, rows in datasets:
        ax.scatter([r["co2_g"] for r in rows], [r["acc"] for r in rows],
                   s=80, color=COLORS[key], edgecolors="black", linewidths=0.4,
                   label=LABELS[key])
        for r in rows:
            if r["sp"] in (0, 10, 20, 50, 70):
                ax.annotate(f"{r['sp']}%", (r["co2_g"], r["acc"]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7.5, color=COLORS[key])
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("CO₂ (g CO₂eq)"); ax.set_ylabel("MMLU Accuracy")
    ax.set_title("D — Accuracy vs Carbon Cost", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("Structured MLP-Channel Pruning Dashboard — All Models",
                 fontsize=14, fontweight="bold", y=0.998)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ── SFT comparison plots ─────────────────────────────────────────────────────

def plot_sft_accuracy_recovery(
    llama_pre, llama_post,
    qwen3_pre, qwen3_post,
    g4_pre, g4_post,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    def _plot_pair(ax, pre, post, key, title):
        pre_sp  = [r["sp"]  for r in pre]
        pre_acc = [r["acc"] for r in pre]
        post_map = {r["sp"]: r["acc"] for r in post}
        post_sp  = sorted(post_map.keys())
        post_acc = [post_map[s] for s in post_sp]

        ax.plot(pre_sp, pre_acc, marker="o", linewidth=2, linestyle=PRE_LINESTYLE,
                color=COLORS[key], label="Pre-SFT (structured only)")
        ax.plot(post_sp, post_acc, marker="s", linewidth=2, linestyle=SFT_LINESTYLE,
                color=COLORS[key], label="Post-SFT (structured + SFT)")
        ax.axhline(CHANCE, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Random (0.25)")
        ax.set_xlabel("Group Sparsity (%)", fontsize=11)
        ax.set_ylabel("MMLU Accuracy", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(sorted(set(pre_sp) | set(post_sp)))
        ax.set_xlim(-2, 75); ax.set_ylim(0.15, 0.80)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # annotate delta where there's meaningful recovery (>5pp)
        for s in post_sp:
            pre_val = dict(zip(pre_sp, pre_acc)).get(s)
            if pre_val and post_map[s] - pre_val > 0.05:
                ax.annotate(
                    f"+{(post_map[s]-pre_val)*100:.1f}pp",
                    xy=(s, (pre_val + post_map[s]) / 2),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8.5, color="darkgreen", fontweight="bold",
                )

    _plot_pair(axes[0], llama_pre, llama_post, "llama", "Llama-3.1-8B-Instruct")
    _plot_pair(axes[1], qwen3_pre, qwen3_post, "qwen3", "Qwen3-8B")
    _plot_pair(axes[2], g4_pre,    g4_post,    "gemma4", "Gemma-4-E4B-IT")

    fig.suptitle(
        "Structured MLP Pruning: Pre-SFT vs Post-SFT Accuracy Recovery",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_sft_accuracy_retained(
    llama_pre, llama_post,
    qwen3_pre, qwen3_post,
    g4_pre, g4_post,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    def _plot_pair(ax, pre, post, key, title):
        baseline = pre[0]["acc"]
        pre_sp   = [r["sp"]  for r in pre]
        pre_ret  = [r["acc"] / baseline * 100 for r in pre]
        post_map = {r["sp"]: r["acc"] / baseline * 100 for r in post}
        post_sp  = sorted(post_map.keys())
        post_ret = [post_map[s] for s in post_sp]

        ax.plot(pre_sp, pre_ret, marker="o", linewidth=2, linestyle=PRE_LINESTYLE,
                color=COLORS[key], label="Pre-SFT")
        ax.plot(post_sp, post_ret, marker="s", linewidth=2, linestyle=SFT_LINESTYLE,
                color=COLORS[key], label="Post-SFT")
        ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense baseline")
        ax.axhline(50,  color="red",  linestyle=":",  linewidth=1, alpha=0.6)
        ax.set_xlabel("Group Sparsity (%)", fontsize=11)
        ax.set_ylabel("Accuracy Retained (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(sorted(set(pre_sp) | set(post_sp)))
        ax.set_xlim(-2, 75); ax.set_ylim(0, 115)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    _plot_pair(axes[0], llama_pre, llama_post, "llama", "Llama-3.1-8B-Instruct")
    _plot_pair(axes[1], qwen3_pre, qwen3_post, "qwen3", "Qwen3-8B")
    _plot_pair(axes[2], g4_pre,    g4_post,    "gemma4", "Gemma-4-E4B-IT")

    fig.suptitle(
        "Structured MLP Pruning: Accuracy Retained — Pre-SFT vs Post-SFT",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_sft_co2_vs_accuracy(
    llama_pre, llama_post,
    qwen3_pre, qwen3_post,
    g4_pre, g4_post,
    llama_unstruct, qwen3_unstruct,
    out: Path,
) -> None:
    """Scatter: accuracy vs inference CO2. Pre-SFT and Post-SFT overlap (same CO2),
    showing accuracy uplift for identical inference cost. Unstructured shown as reference."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, model_key, pre, post, unstruct, title in [
        (axes[0], "llama", llama_pre, llama_post, llama_unstruct, "Llama-3.1-8B-Instruct"),
        (axes[1], "qwen3", qwen3_pre, qwen3_post, qwen3_unstruct, "Qwen3-8B"),
    ]:
        col = COLORS[model_key]

        # Unstructured reference
        u_co2 = [r["co2_g"] for r in unstruct if r["sp"] <= 70]
        u_acc  = [r["acc"]  for r in unstruct if r["sp"] <= 70]
        u_sp   = [r["sp"]   for r in unstruct if r["sp"] <= 70]
        sc = ax.scatter(u_co2, u_acc, c=u_sp, cmap="Greys", s=70, marker="^",
                        edgecolors="gray", linewidths=1, zorder=2, label="Unstructured", alpha=0.8)
        for r in unstruct:
            if r["sp"] <= 70 and r["sp"] in [0, 10, 20, 30, 40, 50]:
                ax.annotate(f"u{r['sp']}%", (r["co2_g"], r["acc"]),
                            textcoords="offset points", xytext=(3, 3), fontsize=7.5, color="gray")

        # Pre-SFT structured
        pre_co2 = [r["co2_g"] for r in pre]
        pre_acc = [r["acc"]   for r in pre]
        pre_sp  = [r["sp"]    for r in pre]
        ax.scatter(pre_co2, pre_acc, c=col, s=90, marker="o", zorder=4,
                   label="Struct (pre-SFT)", edgecolors="black", linewidths=0.5)
        for r in pre:
            ax.annotate(f"s{r['sp']}%", (r["co2_g"], r["acc"]),
                        textcoords="offset points", xytext=(3, -10), fontsize=7.5, color=col)

        # Post-SFT — same CO2 as pre-SFT (same inference structure)
        pre_co2_map = {r["sp"]: r["co2_g"] for r in pre}
        for r in post:
            if r["sp"] in pre_co2_map:
                ax.scatter(pre_co2_map[r["sp"]], r["acc"], c=col, s=120, marker="*",
                           zorder=5, edgecolors="black", linewidths=0.5)
                ax.annotate(f"SFT{r['sp']}%", (pre_co2_map[r["sp"]], r["acc"]),
                            textcoords="offset points", xytext=(3, 4), fontsize=7.5,
                            color=col, fontweight="bold")
                # draw vertical arrow from pre to post
                if r["sp"] in {r2["sp"] for r2 in pre}:
                    pre_acc_val = next(r2["acc"] for r2 in pre if r2["sp"] == r["sp"])
                    if r["acc"] - pre_acc_val > 0.02:
                        ax.annotate("",
                            xy=(pre_co2_map[r["sp"]], r["acc"]),
                            xytext=(pre_co2_map[r["sp"]], pre_acc_val),
                            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
                        )

        ax.axhline(CHANCE, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Inference CO₂ per MMLU Eval (g CO₂eq)", fontsize=11)
        ax.set_ylabel("MMLU Accuracy", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0.20, 0.78)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=9, label="Unstructured"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=9,
                   markeredgecolor="black", label="Structured (pre-SFT)"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor=col, markersize=12,
                   markeredgecolor="black", label="Structured + SFT"),
            Line2D([0], [0], color="darkgreen", linewidth=1.5, label="Accuracy uplift (same CO₂)"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Accuracy vs Inference CO₂: Structured Pruning Before and After SFT\n"
        "★ marks post-SFT accuracy at the same inference cost as ● (structured only)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_sft_co2_bars(
    llama_pre, llama_post,
    qwen3_pre, qwen3_post,
    g4_pre, g4_post,
    out: Path,
) -> None:
    """Bar chart: inference CO2 pre-SFT vs estimated post-SFT at key sparsity levels.

    Post-SFT inference CO2 is identical to pre-SFT (same pruned model structure).
    SFT training CO2 is shown separately as a one-time overhead bar.
    """
    # Training CO2 estimates from training durations (train_s) and total emissions
    # Llama SFT training fraction: ~2600-2810s out of ~5000-5600s total
    TRAIN_CO2 = {
        "llama": {0: 153.8, 10: 160.4, 20: 160.7, 30: 153.9,
                  40: 198.9, 50: 202.3, 60: 201.2, 70: 201.1},
        "qwen3": {0: 234.5, 10: 240.6, 20: 240.9, 30: 237.3,
                  40: 235.2, 50: 234.6, 60: 235.5, 70: 234.1},
        "gemma4": {0: 130, 10: 128, 20: 115, 30: 112, 40: 170, 50: 160},
    }
    key_sps = [0, 10, 20, 30]
    x = np.arange(len(key_sps))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, (model_key, pre, label) in zip(axes, [
        ("llama", llama_pre, "Llama-3.1-8B-Instruct"),
        ("qwen3", qwen3_pre, "Qwen3-8B"),
        ("gemma4", g4_pre,   "Gemma-4-E4B-IT"),
    ]):
        col = COLORS[model_key]
        pre_map = {r["sp"]: r["co2_g"] for r in pre}
        infer_co2  = [pre_map.get(s, 0) for s in key_sps]
        train_co2  = [TRAIN_CO2[model_key].get(s, 0) for s in key_sps]

        b1 = ax.bar(x - width, infer_co2, width, label="Inference CO₂ (pre & post SFT)",
                    color=col, alpha=0.85)
        b2 = ax.bar(x,         infer_co2, width, label="Inference CO₂ (post-SFT, same)",
                    color=col, alpha=0.4, edgecolor=col, linewidth=1.5, linestyle="--")
        b3 = ax.bar(x + width, train_co2, width, label="SFT training CO₂ (one-time)",
                    color="#e15759", alpha=0.75)

        for b, val in zip(b1, infer_co2):
            ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.0f}g",
                    ha="center", va="bottom", fontsize=8)
        for b, val in zip(b3, train_co2):
            ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.0f}g",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}%" for s in key_sps], fontsize=11)
        ax.set_xlabel("Group Sparsity", fontsize=11)
        ax.set_ylabel("CO₂ Emissions (g CO₂eq)", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "CO₂ Breakdown: Inference vs SFT Training\n"
        "Inference cost is identical pre- and post-SFT (same sparse model structure)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_sft_dashboard(
    llama_pre, llama_post,
    qwen3_pre, qwen3_post,
    g4_pre, g4_post,
    out: Path,
) -> None:
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    datasets = [
        ("llama",  llama_pre, llama_post, "Llama-3.1-8B"),
        ("qwen3",  qwen3_pre, qwen3_post, "Qwen3-8B"),
        ("gemma4", g4_pre,    g4_post,    "Gemma-4-E4B"),
    ]

    # Row 0 — accuracy
    for col_idx, (key, pre, post, label) in enumerate(datasets):
        ax = fig.add_subplot(gs[0, col_idx])
        baseline = pre[0]["acc"]
        pre_sp   = [r["sp"]  for r in pre]
        pre_acc  = [r["acc"] for r in pre]
        post_map = {r["sp"]: r["acc"] for r in post}
        post_sp  = sorted(post_map.keys())
        post_acc = [post_map[s] for s in post_sp]

        ax.plot(pre_sp, pre_acc, marker="o", linewidth=2, linestyle=PRE_LINESTYLE,
                color=COLORS[key], label="Pre-SFT")
        ax.plot(post_sp, post_acc, marker="s", linewidth=2, linestyle=SFT_LINESTYLE,
                color=COLORS[key], label="Post-SFT")
        ax.axhline(CHANCE, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.fill_between(
            post_sp,
            [dict(zip(pre_sp, pre_acc)).get(s, np.nan) for s in post_sp],
            post_acc,
            where=[post_map[s] > dict(zip(pre_sp, pre_acc)).get(s, 0) + 0.02 for s in post_sp],
            alpha=0.15, color=COLORS[key], label="SFT gain",
        )
        ax.set_xlabel("Group Sparsity (%)"); ax.set_ylabel("MMLU Accuracy")
        ax.set_title(f"{label}\nPre vs Post-SFT Accuracy", fontweight="bold", fontsize=11)
        ax.set_xticks(sorted(set(pre_sp + post_sp)))
        ax.set_xlim(-2, 75); ax.set_ylim(0.15, 0.80)
        ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

    # Row 1 — accuracy retained
    for col_idx, (key, pre, post, label) in enumerate(datasets):
        ax = fig.add_subplot(gs[1, col_idx])
        baseline = pre[0]["acc"]
        pre_sp   = [r["sp"]  for r in pre]
        pre_ret  = [r["acc"] / baseline * 100 for r in pre]
        post_map = {r["sp"]: r["acc"] / baseline * 100 for r in post}
        post_sp  = sorted(post_map.keys())
        post_ret = [post_map[s] for s in post_sp]

        ax.plot(pre_sp, pre_ret, marker="o", linewidth=2, linestyle=PRE_LINESTYLE,
                color=COLORS[key], label="Pre-SFT")
        ax.plot(post_sp, post_ret, marker="s", linewidth=2, linestyle=SFT_LINESTYLE,
                color=COLORS[key], label="Post-SFT")
        ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense")
        ax.axhline(50,  color="red",  linestyle=":",  linewidth=1, alpha=0.5)
        ax.set_xlabel("Group Sparsity (%)"); ax.set_ylabel("Accuracy Retained (%)")
        ax.set_title(f"{label}\nAccuracy Retained", fontweight="bold", fontsize=11)
        ax.set_xticks(sorted(set(pre_sp + post_sp)))
        ax.set_xlim(-2, 75); ax.set_ylim(0, 115)
        ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Structured MLP Pruning + SFT — Pre vs Post-SFT Dashboard\n"
        "(Solid = pre-SFT · Dashed = post-SFT · Inference CO₂ is identical for both)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_semi24_sft_accuracy(pre_by_model: dict[str, list[dict]],
                             post_by_model: dict[str, list[dict]],
                             out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, key in zip(axes, ["llama", "qwen3", "gemma4"]):
        pre = {r["sp"]: r["acc"] for r in pre_by_model[key]}
        post = {r["sp"]: r["acc"] for r in post_by_model[key]}
        sps = sorted(set(pre) | set(post))
        x = np.arange(len(sps))
        width = 0.35
        pre_vals = [pre.get(s, np.nan) for s in sps]
        post_vals = [post.get(s, np.nan) for s in sps]
        ax.bar(x - width / 2, pre_vals, width, color=COLORS[key], alpha=0.45,
               label="Pre-SFT")
        ax.bar(x + width / 2, post_vals, width, color=COLORS[key], alpha=0.90,
               label="Post-SFT")
        ax.axhline(CHANCE, color="gray", linestyle=":", linewidth=1, label="Random")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}%" for s in sps])
        ax.set_xlabel("N:M Sparsity")
        ax.set_title(LABELS[key], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9)
        for idx, s in enumerate(sps):
            if s in pre and s in post:
                delta = post[s] - pre[s]
                ax.text(idx, max(pre[s], post[s]) + 0.015, f"{delta*100:+.1f}pp",
                        ha="center", va="bottom", fontsize=8.5, color="darkgreen" if delta > 0 else "dimgray")
    axes[0].set_ylabel("MMLU Accuracy")
    axes[0].set_ylim(0.15, 0.80)
    fig.suptitle("Semi-Structured 2:4 Pruning: Pre-SFT vs Post-SFT Accuracy",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def plot_semi24_sft_retained(pre_by_model: dict[str, list[dict]],
                             post_by_model: dict[str, list[dict]],
                             out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, key in zip(axes, ["llama", "qwen3", "gemma4"]):
        pre = {r["sp"]: r["acc"] for r in pre_by_model[key]}
        post = {r["sp"]: r["acc"] for r in post_by_model[key]}
        baseline = pre.get(0, post.get(0))
        sps = sorted(set(pre) | set(post))
        x = np.arange(len(sps))
        width = 0.35
        pre_vals = [pre.get(s, np.nan) / baseline * 100 for s in sps]
        post_vals = [post.get(s, np.nan) / baseline * 100 for s in sps]
        ax.bar(x - width / 2, pre_vals, width, color=COLORS[key], alpha=0.45,
               label="Pre-SFT")
        ax.bar(x + width / 2, post_vals, width, color=COLORS[key], alpha=0.90,
               label="Post-SFT")
        ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Dense baseline")
        ax.axhline(50, color="red", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}%" for s in sps])
        ax.set_xlabel("N:M Sparsity")
        ax.set_title(LABELS[key], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Accuracy Retained (%)")
    axes[0].set_ylim(0, 115)
    fig.suptitle("Semi-Structured 2:4 Pruning: Accuracy Retained Before and After SFT",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    g4_struct_out = Path("outputs/plots/gemma4_structured")
    sft_out       = Path("outputs/plots/sft_comparison")
    g4_struct_out.mkdir(parents=True, exist_ok=True)
    sft_out.mkdir(parents=True, exist_ok=True)

    ref_durs = _gemma4_unstructured_durations()
    g4_pre   = load_gemma4_pre_sft(ref_durs)
    g4_post  = load_gemma4_post_sft()

    llama_pre, llama_post = load_llama_structured()
    qwen3_pre, qwen3_post = load_qwen3_structured()

    llama_unstruct = load_unstructured("llama")
    qwen3_unstruct = load_unstructured("qwen3")
    semi24_pre = load_semi24_pre_sft()
    semi24_post = load_semi24_post_sft()

    print("=== Structured pre-SFT plots (all models) ===")
    plot_all_structured_accuracy(
        llama_pre, qwen3_pre, g4_pre,
        sft_out / "structured_accuracy_vs_sparsity.png",
    )
    plot_all_structured_accuracy_retained(
        llama_pre, qwen3_pre, g4_pre,
        sft_out / "structured_accuracy_retained.png",
    )
    plot_all_structured_weight_vs_group(
        llama_pre, qwen3_pre, g4_pre,
        sft_out / "structured_weight_vs_group_sparsity.png",
    )
    plot_all_structured_co2(
        llama_pre, qwen3_pre, g4_pre,
        sft_out / "structured_co2_vs_sparsity.png",
    )
    plot_all_structured_dashboard(
        llama_pre, qwen3_pre, g4_pre,
        sft_out / "structured_dashboard.png",
    )

    print("\n=== Gemma-4 structured pre-SFT plots ===")
    plot_gemma4_accuracy(          g4_pre, g4_struct_out / "accuracy_vs_sparsity.png")
    plot_gemma4_accuracy_retained( g4_pre, g4_struct_out / "accuracy_retained.png")
    plot_gemma4_co2(               g4_pre, g4_struct_out / "co2_vs_sparsity.png")
    plot_gemma4_weight_vs_group(   g4_pre, g4_struct_out / "weight_vs_group_sparsity.png")
    plot_gemma4_dashboard(         g4_pre, g4_struct_out / "dashboard.png")

    print("\n=== SFT comparison plots (all models) ===")
    plot_sft_accuracy_recovery(
        llama_pre, llama_post, qwen3_pre, qwen3_post, g4_pre, g4_post,
        sft_out / "accuracy_recovery.png",
    )
    plot_sft_accuracy_retained(
        llama_pre, llama_post, qwen3_pre, qwen3_post, g4_pre, g4_post,
        sft_out / "accuracy_retained.png",
    )
    plot_sft_co2_vs_accuracy(
        llama_pre, llama_post, qwen3_pre, qwen3_post, g4_pre, g4_post,
        llama_unstruct, qwen3_unstruct,
        sft_out / "co2_vs_accuracy.png",
    )
    plot_sft_co2_bars(
        llama_pre, llama_post, qwen3_pre, qwen3_post, g4_pre, g4_post,
        sft_out / "co2_breakdown.png",
    )
    plot_sft_dashboard(
        llama_pre, llama_post, qwen3_pre, qwen3_post, g4_pre, g4_post,
        sft_out / "dashboard.png",
    )

    print("\n=== Semi-structured 2:4 SFT comparison plots (all models) ===")
    plot_semi24_sft_accuracy(
        semi24_pre, semi24_post,
        sft_out / "semi24_accuracy_recovery.png",
    )
    plot_semi24_sft_retained(
        semi24_pre, semi24_post,
        sft_out / "semi24_accuracy_retained.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
