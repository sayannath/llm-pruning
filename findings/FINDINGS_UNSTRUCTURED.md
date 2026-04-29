# Findings: Global Unstructured Magnitude Pruning on 8B LLMs

**Models evaluated:** Llama-3.1-8B-Instruct · Qwen3-8B · Gemma-4-E4B-IT  
**Benchmark:** MMLU (14,042 test examples, zero-shot scoring)  
**Pruning method:** Global unstructured magnitude pruning of all `nn.Linear` weights  
**Sparsity sweep:** 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99%  
**Cluster:** H100 GPU (80 GB VRAM) · Canada · tracked via CodeCarbon  
**Scoring:** Llama/Qwen3 use choice log-probability; Gemma-4 uses batched generation (30-token greedy decode + answer parsing) due to near-zero first-token probability on bare choice letters.

---

## 1. Accuracy vs Sparsity

![Accuracy vs Sparsity](../outputs/plots/accuracy_vs_sparsity.png)

The headline result is stark: **all three models tolerate moderate sparsity surprisingly well, then catastrophically collapse — Llama and Qwen3 at the 40→50% boundary, Gemma-4 already at 30→40%.**

| Sparsity | Llama-3.1-8B | Retained | Qwen3-8B | Retained | Gemma-4-E4B | Retained |
|---|---|---|---|---|---|---|
| 0% | 0.6533 | 100.0% | 0.7159 | 100.0% | 0.6652 | 100.0% |
| 10% | 0.6479 | 99.2% | 0.7161 | **100.0%** | 0.6652 | **100.0%** |
| 20% | 0.6314 | 96.6% | 0.6963 | 97.3% | 0.6518 | 98.0% |
| 30% | 0.5961 | 91.2% | 0.6619 | 92.5% | 0.6142 | 92.3% |
| **40%** | 0.5348 | 81.9% | 0.5632 | 78.7% | **0.4612** | **69.3%** |
| **50%** | **0.3423** | **52.4%** | **0.2723** | **38.0%** | **0.2295** | **34.5%** |
| 60% | ~0.25 | ~38% | ~0.25 | ~35% | 0.2295 | 34.5% |
| 70% | ~0.24 | ~37% | ~0.24 | ~34% | 0.2330 | 35.0% |
| 80% | ~0.23 | ~36% | ~0.23 | ~32% | 0.2332 | 35.1% |
| 90% | ~0.23 | ~36% | ~0.23 | ~32% | 0.2300 | 34.6% |
| 95% | ~0.23 | ~36% | ~0.23 | ~32% | 0.2290 | 34.4% |
| 99% | ~0.23 | ~36% | ~0.23 | ~32% | 0.2302 | 34.6% |

The degradation in the 0–30% range is gradual and smooth for all models. At 40%, Gemma-4 drops sharply to 69.3% retained — a warning sign that does not appear in Llama or Qwen3 until 50%. By 50%, all three models cross into near-random territory (chance = 0.25 on 4-way MCQA).

---

## 2. Accuracy Retained

![Accuracy Retained](../outputs/plots/accuracy_retained.png)

This view normalises performance against each model's own dense baseline, making the graceful-then-catastrophic pattern clearer. Both curves track each other closely until 40%, then diverge sharply at 50%: **Qwen3 falls further and faster** (38% retained vs 52% for Llama).

---

## 3. Key Insight — The Cliff Arrives at 50% for Llama/Qwen3, but 40% for Gemma-4

Most prior work on LLM pruning places the tolerable sparsity limit at 60–70%. These experiments show the cliff arrives earlier under global unstructured magnitude pruning:

| Model | Safe zone | Caution zone | Collapse |
|---|---|---|---|
| Llama-3.1-8B | ≤ 30% (> 91% retained) | 30–40% | ≥ 50% |
| Qwen3-8B | ≤ 30% (> 92% retained) | 30–40% | ≥ 50% |
| **Gemma-4-E4B** | **≤ 30% (> 92% retained)** | **30–40%** | **≥ 40%** |

Gemma-4 is the most brittle of the three: it retains only **69.3%** accuracy at 40% sparsity (vs 81.9% for Llama and 78.7% for Qwen3 at the same level). Once past 50%, all three models converge to the same near-random band (~0.23–0.25), with no meaningful difference between 50%, 95%, and 99% sparsity.

---

## 4. Model Comparison — Baseline Strength vs Pruning Robustness

![Accuracy vs Carbon Cost](../outputs/plots/accuracy_vs_co2.png)

Qwen3-8B outperforms the other two at baseline (0.7159), while Llama and Gemma-4 are closely matched (0.6533 vs 0.6652). Key observations:

- **All three models are immune to 10% pruning.** Qwen3 gains 0.0002 (noise), Gemma-4 gains 0.0000, Llama loses only 0.0054. Low-magnitude weights at 10% threshold contribute essentially nothing to output.
- **At 40% sparsity, Gemma-4 drops hardest** — 69.3% retained vs 81.9% (Llama) and 78.7% (Qwen3). Gemma-4's capacity appears more concentrated in fewer high-magnitude weights, making it more sensitive to pruning at moderate levels.
- **At the 50% cliff, Qwen3 collapses hardest** (38% retained), then Gemma-4 (34.5%), then Llama (52.4%). Despite its superior baseline, Qwen3 concentrates capacity more densely and is the most brittle under aggressive pruning.
- **Post-cliff (50%+), all three models converge** to ~0.23–0.27, indistinguishable from one another or from random chance. Model architecture and capability no longer matter once the weight structure is sufficiently destroyed.

---

## 5. Carbon Emissions per Run

![CO₂ per Run](../outputs/plots/co2_vs_sparsity.png)

**Carbon cost does not meaningfully scale with sparsity.** Each run costs roughly 90–115 g CO₂eq regardless of whether 0% or 99% of weights are zeroed. This is because:

1. Unstructured pruning does not reduce FLOPs — zeroed weights still participate in dense matrix multiplications.
2. The dominant cost is the **evaluation pass** over 14,042 examples, which is identical across sparsities.
3. The pruning operation itself (computing a global threshold and zeroing weights) takes < 1 second for 7B parameters and contributes negligible energy.

> **Implication:** The common intuition that "sparser models are greener at inference" does not hold for global unstructured pruning without sparse kernel support. Real emission savings require hardware-aware sparse computation (e.g., structured/N:M sparsity with cuSPARSELt).

**Notable outlier — Llama sparsity=0 (74.6 g, GPU=339 W):** This run executed on a different SLURM node (mgh5) where the GPU was less loaded, delivering higher utilisation and shorter wall time (~17 min vs ~37 min on mgh4). All subsequent runs ran on mgh4 at ~170 W GPU draw.

---

## 6. Carbon Efficiency

![Carbon Efficiency](../outputs/plots/carbon_efficiency.png)

Carbon efficiency (accuracy per gram of CO₂) is highest at 0% sparsity and monotonically degrades. The sharp drop at 50% mirrors the accuracy cliff. From a **carbon-per-unit-accuracy perspective, pruning is never beneficial** under this setup — you spend the same carbon to get less accuracy.

However, this metric ignores deployment carbon (inference at scale). If a pruned model could achieve real speedups through sparse kernels, the lifetime deployment savings could outweigh the evaluation carbon. That remains a hardware engineering problem, not a model problem.

---

## 7. Power Breakdown

![Power Breakdown](../outputs/plots/power_breakdown.png)

GPU dominates power draw at every sparsity level (~170 W average on mgh4). CPU and RAM contributions are roughly constant at ~50 W and ~45 W respectively. Two observations:

- **Sparsity has no effect on GPU power draw** — the GPU is fully saturated by dense matrix multiplications at all sparsity levels, confirming that unstructured zeros give no compute relief.
- **Qwen3 runs are slightly shorter than Llama runs** at the same sparsity (both models have ~7B target parameters, but Qwen3's inference appears marginally faster on this hardware configuration).

---

## 8. Cumulative Carbon Footprint

![Cumulative CO₂](../outputs/plots/cumulative_co2.png)

The full two-model sweep consumed **2,335 g CO₂eq (≈ 2.3 kg)** and **3,720 Wh (3.7 kWh)** of electricity — roughly equivalent to charging a smartphone 300 times or driving an electric car ~15 km.

Steps are even-sized because each sparsity point costs roughly the same to evaluate. The two lines (Llama and Qwen3) track each other closely throughout, confirming that per-run cost is driven by the benchmark, not by the model or pruning level.

---

## 9. Summary Dashboard

![Dashboard](../outputs/plots/dashboard.png)

---

## 10. Consolidated Findings

### What works
| Finding | Evidence |
|---|---|
| All three models tolerate up to 30% sparsity with < 9% accuracy drop | Table §1 |
| Qwen3-8B has the strongest baseline (+6.3 pts over Llama, +5.1 pts over Gemma-4) | §4 |
| All three models are effectively immune to 10% pruning | §4 |
| Gemma-4-E4B at 0% sparsity (0.6652) is competitive with Llama-3.1-8B (0.6533) | Table §1 |

### What breaks down
| Finding | Evidence |
|---|---|
| Gemma-4 collapses earlier: 40% sparsity retains only 69.3% vs 82% for Llama | Table §1, §3 |
| 50% sparsity triggers catastrophic collapse in all three models | Table §1 |
| Qwen3 collapses hardest at 50% (38% retained) despite strongest baseline | §4 |
| Post-cliff (50%+) all models converge to ~0.23–0.25, indistinguishable from random | Table §1 |
| Carbon cost is flat across sparsities — no green inference benefit | §5, §6 |

### Practical recommendations
1. **If pruning for deployment**, stop at ≤ 30% — all three models remain strongly usable (> 91% retained).
2. **For Gemma-4, treat 30% as the hard limit** — the extra brittleness at 40% (69% retained) makes it riskier than Llama or Qwen3 at the same level.
3. **Do not prune past 40%** without fine-tuning / distillation recovery; 50%+ is effectively model destruction under this method.
4. **Prefer Qwen3-8B at low sparsity** (highest absolute accuracy); **prefer Llama-3.1-8B at 40%** (most graceful degradation); **avoid Gemma-4 at 40%+** under unstructured pruning.
5. **For real inference carbon savings**, switch to structured or N:M sparsity patterns that sparse kernels can exploit.

---

## Reproducibility

Artifacts are saved under the following run directories:
- **Llama-3.1-8B + Qwen3-8B:** `outputs/runs/mmlu_pruning_ade7d5ffbbb4/`
- **Gemma-4-E4B-IT:** `outputs/runs/mmlu_pruning_39fc35ec1838/`


```
<model>/sparsity_<XYZ>/
├── metrics.json          # accuracy, per-subject breakdown, emissions_kg_co2
├── emissions.json        # GPU/CPU/RAM power, duration, energy, country
├── predictions.jsonl     # per-example gold, pred, scores, elapsed_s, emissions_kg_co2
├── pruning_stats.json    # per-layer sparsity breakdown
├── config_resolved.yaml  # exact config used
└── run.log               # timestamped pruning + evaluation log
```

To regenerate all plots:

```bash
python scripts/plot_results.py --run-dir outputs/runs/mmlu_pruning_ade7d5ffbbb4 --out-dir outputs/plots
```
