from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_sparsity_vs_accuracy(summary_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for model_name, group in df.groupby("model_name"):
        group = group.sort_values("sparsity_requested")
        plt.plot(group["sparsity_requested"], group["accuracy"], marker="o", label=model_name)
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
