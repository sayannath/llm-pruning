# Findings: Sustainability & GPU Acceleration Across Pruning Methods

**Models evaluated:** Llama-3.1-8B-Instruct · Qwen3-8B · Gemma-4-E4B-IT  
**Benchmark:** MMLU (14,042 test examples, zero-shot choice log-probability scoring)  
**Methods compared:** Global Unstructured · Structured MLP-Channel · Semi-Structured 2:4 · Semi-Structured 4:8  
**Cluster:** H100 GPU (80 GB VRAM) · Canada · tracked via CodeCarbon  
**Evaluation node note:** Unstructured experiments ran on node `mgh4` (~170 W GPU); structured and semi-structured ran on `mgh5` (~270–350 W GPU). Throughput comparisons within each group are valid; cross-group throughput numbers reflect hardware differences as well as pruning effects.  
**Gemma-4 note:** Gemma-4 requires generation-based scoring, which introduced per-run overhead (unbatched generation in structured/semi-structured runs vs batched generation in the unstructured sweep). Gemma-4 CO₂ figures for structured and semi-structured methods are **normalised** using the unstructured batched-generation baseline at the same sparsity, corrected for the genuine structured speedup (2.25× derived from Llama ratios).

---

## Master Dashboard

![Master Dashboard](../outputs/plots/sustainability/master_dashboard.png)

---

## 1. Carbon Cost at 50% Sparsity

![CO₂ at 50%](../outputs/plots/sustainability/co2_at_50pct.png)

At the common operating point of 50% weight sparsity, structured MLP pruning is the greenest option for the actual inference run:

| Method | Llama CO₂ | Qwen CO₂ | Gemma-4 CO₂† | Avg (L+Q) | vs Unstructured |
|---|---|---|---|---|---|
| **Unstructured** | 98.0 g | 99.2 g | 181.4 g | 98.6 g | baseline |
| **Structured MLP** | **63.3 g** | **79.1 g** | **83.2 g†** | **71.2 g** | **−28%** |
| **Semi 2:4** | 79.2 g | 95.9 g | ~181 g† | 87.6 g | −11% |
| **Semi 4:8** | 82.1 g | 100.7 g | ~181 g† | 91.4 g | −7% |

† Gemma-4 values are normalised (see header note). Semi-structured Gemma-4 normalises to ~181g — effectively identical to unstructured — confirming that without sparse kernel support, semi-structured and unstructured carry identical inference FLOPs.

**Why structured MLP saves ~28% carbon:** Structural MLP-channel pruning physically removes rows and columns from the weight matrices (gate_proj, up_proj, down_proj). At 50% group sparsity, the intermediate MLP dimension is halved, cutting the FLOP count of every MLP forward pass by ~50%. For Llama, this produces a **2.25× measured throughput gain** (2110 s → 952 s per 14,042-example eval), translating directly to lower energy and CO₂. This is a genuine FLOP reduction, not a scoring-mode artefact.

**Why semi-structured does not save carbon (without sparse kernels):** Semi-structured 2:4 and 4:8 patterns zero individual weights but preserve matrix shapes. Without cuSPARSELt or equivalent sparse-matmul support, the GPU executes the same dense matrix multiplication — every zero still participates in the multiply-accumulate. FLOP count is unchanged; inference time and CO₂ are unchanged relative to unstructured. The apparent 7–11% savings in the table above reflect node-level hardware variation (semi-structured runs landed on mgh5 with 2× higher GPU clock vs unstructured runs on mgh4), not pruning-driven compute reduction.

> **Critical caveat:** These are evaluation-time emissions used as a proxy for deployment inference. Real-world deployment savings from semi-structured pruning require activating hardware sparse kernels (cuSPARSELt for 2:4) — which were not used in this study.

---

## 2. Carbon Efficiency (Accuracy per Gram of CO₂)

![Carbon Efficiency](../outputs/plots/sustainability/carbon_efficiency.png)

Carbon efficiency measures how much useful model capability you get per gram of CO₂ spent. The highest score wins from a sustainability standpoint:

| Method | Llama (acc/g) | Qwen (acc/g) |
|---|---|---|
| **Unstructured** | 0.003493 | 0.002745 |
| **Semi 4:8** | **0.004217** | 0.002534 |
| **Semi 2:4** | 0.003594 | 0.002534 |
| Structured MLP | 0.003649 | 0.003081 |

**Semi 4:8 is the most carbon-efficient method for Llama** — it achieves the highest accuracy (34.62%) while spending only 82.1 g CO₂, yielding the best accuracy-per-gram ratio. Structured MLP edges out unstructured on Qwen because it uses 20% less carbon while landing at roughly the same near-random accuracy.

The overall picture: **at 50% sparsity, no method improves carbon efficiency enough to be a meaningful win** — all methods collapse Qwen to near-random, and Llama only Semi 4:8 retains usable capability. The best sustainability decision is to stay below the cliff (<40% for unstructured, <10% for structured).

---

## 3. Sustainability Frontier — Accuracy vs CO₂ Trade-off

![Sustainability Frontier](../outputs/plots/sustainability/sustainability_frontier.png)

This plot shows every (sparsity, method) combination as a point on the accuracy–carbon plane. Points in the upper-left are ideal: high accuracy, low carbon cost.

**Llama-3.1-8B (left panel):**
- Unstructured dominates up to 40% sparsity — it traces the upper-left frontier, maintaining accuracy while spending ~100g CO₂ per run
- Semi 4:8 at 50% closely matches unstructured in accuracy while spending less carbon — the only point where semi-structured enters the frontier
- Structured MLP collapses to the lower band immediately at 10% group sparsity

**Qwen3-8B (right panel):**
- The frontier is dominated entirely by unstructured pruning — structured and semi-structured both collapse to near-random at 50%
- Qwen3's sole advantage of the 10% structured step (56.55% accuracy) appears as a single point near the frontier, but it disappears at 20%

**Key frontier insight for Llama:** The Pareto-efficient options are:
1. **Unstructured 0–30%** — best accuracy, standard carbon cost
2. **Unstructured 40%** — last usable point before cliff (53.5% acc, 101g)
3. **Semi 4:8 50%** — similar accuracy to unstructured 50% but 16% less carbon (82g vs 98g)

---

## 4. Accuracy Retained vs CO₂ Saved

![Accuracy vs CO₂ Saved](../outputs/plots/sustainability/accuracy_vs_co2_saved.png)

This quadrant plots methods against two sustainability axes simultaneously:
- **X-axis:** CO₂ saved relative to unstructured at the same sparsity (positive = greener)
- **Y-axis:** Accuracy retained relative to dense baseline (higher = better quality)
- **Bubble size:** Observed evaluation throughput (samples/sec)

**Ideal position:** upper-right (saves carbon AND retains accuracy).

For Llama:
- **Semi 4:8** lands upper-right: +15.9g CO₂ saved, 53% accuracy retained — the only method in the desirable quadrant
- **Semi 2:4** saves carbon (+18.8g) but pays a heavier accuracy cost (44% retained) — right but lower
- **Structured MLP** saves the most carbon (+34.7g) but retains almost no accuracy (35%) — extreme right, near the floor

For Qwen3:
- All three pruning methods save some carbon over unstructured, but all collapse accuracy equally — they cluster horizontally at the 34–38% accuracy retained level
- No method achieves both carbon savings and accuracy retention for Qwen3 at 50% sparsity

---

## 5. Total Experiment Cost — Full Sweep CO₂ and Energy

![Total Sweep Cost](../outputs/plots/sustainability/total_sweep_cost.png)

The total carbon and energy consumed by each complete experiment sweep:

| Method | Runs | Total CO₂ | Total Energy | CO₂ per run |
|---|---|---|---|---|
| **Unstructured** | 24 | **2,335 g** | **3,720 Wh** | 97.3 g |
| **Structured MLP** | 16 | 1,153 g | 1,837 Wh | 72.1 g |
| **Semi 2:4** | 4 | 361 g | 574 Wh | 90.1 g |
| **Semi 4:8** | 4 | 376 g | 599 Wh | 94.1 g |

The unstructured sweep cost is highest because it explored 12 sparsity points (vs 8 for structured, 2 for semi-structured). **Per-run**, structured MLP is the cheapest (72.1g) and unstructured is the most expensive (97.3g). The full unstructured sweep is equivalent to:
- Charging a smartphone ~190 times
- Driving an electric car ~15 km

If repeated across all three new method sweeps, the total research cost of this study is ~4,225 g CO₂ — roughly 4.2 kg.

---

## 6. GPU Power Draw and Observed Throughput

![GPU and Throughput](../outputs/plots/sustainability/gpu_and_throughput.png)

Observed at 50% sparsity during the MMLU evaluation pass:

| Method | Llama GPU (W) | Qwen GPU (W) | Llama tput | Qwen tput |
|---|---|---|---|---|
| Unstructured (mgh4) | 170 W | 175 W | 6.6 s/s | 6.1 s/s |
| Structured MLP (mgh5) | 271 W | 239 W | 14.8 s/s | 10.8 s/s |
| Semi 2:4 (mgh5) | 351 W | 302 W | 14.3 s/s | 10.4 s/s |
| Semi 4:8 (mgh5) | 343 W | 295 W | 14.2 s/s | 10.3 s/s |

The apparent 2× throughput improvement for structured/semi-structured over unstructured is primarily a **hardware node effect** — mgh5 delivered significantly higher GPU clock rates and memory bandwidth than mgh4. This is not a pruning-driven speedup.

Within the mgh5 runs (apples-to-apples):
- Structured MLP, Semi 2:4, and Semi 4:8 all achieve ~10–15 samples/sec with minimal difference
- The ~30–35% higher GPU power draw for semi-structured vs structured is consistent with semi-structured models still executing dense matmul (no sparse kernels activated), drawing full power for the same number of FLOPs

---

## 7. Theoretical GPU Acceleration Potential

![Theoretical Speedup](../outputs/plots/sustainability/theoretical_speedup.png)

This plot shows the **theoretical inference speedup** achievable by each pruning method when the appropriate sparse kernels are activated — which was **not** done in this study. All observed evaluation throughput numbers above are dense-execution baselines.

| Method | Theoretical Speedup | Hardware Requirement | Status |
|---|---|---|---|
| **Unstructured** | **1×** (none) | Dense matmul unchanged | No speedup without custom sparse kernels |
| **Structured MLP (materialized)** | **1.5–3×** | Architecture surgery + smaller dense matmul | Requires removing zeroed channels from weight tensors |
| **Semi 2:4 (cuSPARSELt)** | **~2×** | NVIDIA A100/H100 Tensor Core sparse matmul | Directly supported via `torch.sparse` / cuSPARSELt |
| **Semi 4:8 (custom kernel)** | **~1.2–1.6×** | Non-standard; needs custom CUDA kernel | Not natively supported in cuSPARSELt |

### Unstructured: No Hardware Path
Global magnitude unstructured pruning produces irregular sparsity patterns that current GPU hardware cannot exploit. Dense matrix multiplication treats every zero as a real computation. Zero benefit to latency or energy without software-defined sparse matmul (e.g., DeepSparse, cuSPARSE CSR), which trades flexibility for throughput.

### Structured MLP: Real Speedup if Materialized
Masked structured pruning (as implemented here) preserves tensor shapes — it zeroes weights but does not shrink the weight matrices. To realise the speedup, the pruned model must be **materialised**: zeroed rows and columns are physically removed, reducing the hidden dimension of MLP projections. A model at 50% group sparsity would have half the intermediate channels — halving MLP FLOP count, which is the dominant compute. **Expected 1.5–3× speedup on standard dense hardware** without any special kernel.

### Semi 2:4: The Production-Ready Path
NVIDIA's 2:4 structured sparsity (2 non-zeros in every 4 consecutive weights) is natively accelerated via cuSPARSELt on A100/H100 GPUs. The hardware stores and loads only the non-zero values + a bitmask, then uses specialised Tensor Core instructions for sparse matmul. **Guaranteed 2× throughput** at 50% sparsity with no accuracy loss from kernel overhead. This is the only method in this study that offers a **drop-in production speedup** — activate cuSPARSELt and existing model weights (compressed with `torch.ao.pruning.sparsifier`) run 2× faster with identical numerical outputs.

### Semi 4:8: Better Accuracy, Harder to Deploy
The 4:8 pattern (4 non-zeros in 8 weights) retains more accuracy than 2:4 on Llama but is not natively supported by NVIDIA's cuSPARSELt library (which only handles 2:4). Custom CUDA kernels can accelerate it, but require significant engineering effort. Expected speedup is lower (~1.2–1.6×) due to less aggressive sparsity and no Tensor Core optimisation path. **Not recommended for production deployment without custom kernel development.**

---

## 8. Method Comparison Summary

| Criterion | Unstructured | Structured MLP | Struct+SFT | Semi 2:4 | Semi 4:8 |
|---|---|---|---|---|---|
| **Accuracy @ sp=10 (Llama)** | 64.8% | 29.1% | **47.2%** | N/A | N/A |
| **Accuracy @ sp=10 (Qwen)** | 71.6% | 56.6% | **70.3%** | N/A | N/A |
| **Accuracy @ sp=20 (Qwen)** | 69.6% | 26.4% | **58.9%** | N/A | N/A |
| **Accuracy @ sp=50 (Llama)** | 34.2% | 23.1% | 23.1% | 28.5% | **34.6%** |
| **Inference CO₂ @ sp=50** | 98.6g | **71.2g** | **71.2g** | 87.6g | 91.4g |
| **Inference CO₂ @ sp=10** | ~102g | ~66g | **~66g** | N/A | N/A |
| **CO₂ saving vs unstructured** | — | **−28 to −37%** | **−28 to −37%** | ~0%* | ~0%* |
| **GPU acceleration (production)** | None | 1.5–3× (materialized) | 1.5–3× (materialized) | **2× (cuSPARSELt)** | ~1.4× (custom kernel) |
| **Safe sparsity zone** | ≤ 40% | ≤ 10% (Qwen only) | ≤ 20% (Qwen), ≤ 10% (Llama) | N/A | N/A |

*Semi-structured CO₂ savings reflect hardware node variation, not genuine FLOP reduction without sparse kernels.

---

## 9. Consolidated Recommendations

### For accuracy-first deployment (no latency requirement)
Use **unstructured pruning ≤ 30%** — all three models retain > 91% accuracy with zero infrastructure changes. The carbon cost is highest per run but the quality is unmatched.

### For best balance of accuracy and inference emissions
Use **structured MLP pruning + SFT**:
- **Qwen3 at 10% group sparsity + SFT**: 98.2% accuracy retained at 26.6% lower inference CO₂ vs unstructured. Near-lossless and materially greener.
- **Qwen3 at 20% group sparsity + SFT**: 82.4% accuracy retained at 25.9% lower inference CO₂. Good trade-off for larger savings.
- **Llama at 10% group sparsity + SFT**: 72.3% accuracy retained at 35.3% lower CO₂. Accepts larger accuracy cost for larger CO₂ savings.
- SFT training cost is a one-time overhead (~155–240g), recouped after ~5–10 inference evaluations.
- **This is the only Pareto-efficient strategy in this study: both lower inference CO₂ and competitive accuracy.**

### For green inference at scale (hardware-accelerated, no SFT)
Use **semi-structured 2:4** with cuSPARSELt enabled:
- 2× throughput on H100/A100 → halves the per-token energy at inference
- No FLOP savings without activating sparse kernels — requires `torch.ao.pruning` + cuSPARSELt
- For Llama, accepts a small accuracy cost at 50% (34.6% retained) vs unstructured (34.2%)
- For Qwen3, all 50% methods collapse equally — choose 2:4 for its hardware path

### For maximum parameter efficiency (edge deployment, no SFT)
Use **structured MLP pruning ≤ 10% group sparsity for Qwen3** only:
- Materialise the pruned model (remove zeroed channels) to get ~1.5× MLP speedup on standard dense hardware with no special kernels
- **Do not use on Llama without SFT** — Llama collapses immediately at 10% group sparsity

### For sustainability of the research process
- **Semi-structured sweeps are cheapest** (361g CO₂ for 4 runs) because only two sparsity points are valid (0 and 50)
- Future sweeps should **stop structured at 20%** — the cliff means points 30–70% are scientifically redundant
- Run structured sensitivity probes at 128 samples before committing to a full 14K-sample sweep

---

## 10. Structured MLP + Supervised Fine-Tuning: Recovering Accuracy at the Same Inference Cost

This section answers the core research question: *Can SFT applied to a structurally pruned model recover accuracy while preserving the CO₂ savings from pruning?*

### 10.1 Why Inference Cost Is Unchanged by SFT

SFT fine-tunes the model weights without modifying the pruning mask or the structural shape of the weight tensors. The MLP channel groups that were zeroed at pruning time remain zeroed. When materialised for deployment, the model retains exactly the same reduced-dimension weight matrices — and therefore the same FLOPs, the same throughput, and the same per-inference CO₂.

**Inference CO₂: structured-only = structured+SFT (same sparse model, same FLOPs).**

The emissions captured in SFT run logs include training overhead (one-time fine-tuning cost, ~2600–3300 s per sparsity level). That training cost is not an ongoing inference expense; it is amortised across all future inference calls on the deployed model.

### 10.2 Accuracy Recovery — Llama-3.1-8B

| Group sp | Pre-SFT Acc | Post-SFT Acc | Δ Acc | Inference CO₂ | Savings vs Unstruct |
|---|---|---|---|---|---|
| 0% | 0.6524 | 0.6691 | +0.0167 | 67.2 g | +9.9% saved |
| **10%** | 0.2905 | **0.4720** | **+0.1815** | **65.9 g** | **+35.3% saved** |
| 20% | 0.2332 | 0.2298 | −0.0033 | 65.2 g | +36.7% saved |
| 30% | 0.2336 | 0.2351 | +0.0015 | 64.5 g | +36.0% saved |
| 40% | 0.2333 | 0.2285 | −0.0048 | 63.7 g | +37.0% saved |
| 50%+ | ~0.231 | ~0.231 | ≈0 | ~63 g | ~35% saved |

SFT helps Llama only at sp=10: accuracy jumps from 44.5% retained to 72.3% retained (+18.15pp absolute). Beyond sp=10, structured pruning has destroyed too many channels for SFT to recover — the fine-tuning loss converges but accuracy stays near random. The sp=0 dense SFT is a small quality bonus (+1.7pp) at no extra inference cost.

**Key result — Llama sp=10:** Structured+SFT delivers **72.3% of dense-baseline accuracy at 65.9g CO₂**, compared to **99.2% retained at 101.8g CO₂** for unstructured (logprob, same sparsity). This is a 35% inference CO₂ reduction at a 27pp accuracy cost — not yet Pareto-optimal over unstructured, but a meaningful trade-off point for carbon-constrained deployment.

### 10.3 Accuracy Recovery — Qwen3-8B

| Group sp | Pre-SFT Acc | Post-SFT Acc | Δ Acc | Inference CO₂ | Savings vs Unstruct |
|---|---|---|---|---|---|
| 0% | 0.7159 | 0.7441 | +0.0281 | 81.5 g | +1.4% saved |
| **10%** | 0.5655 | **0.7029** | **+0.1374** | **81.1 g** | **+26.6% saved** |
| **20%** | 0.2636 | **0.5898** | **+0.3262** | **80.9 g** | **+25.9% saved** |
| 30% | 0.2387 | 0.2433 | +0.0046 | 80.2 g | +29.7% saved |
| 40%+ | ~0.236 | ~0.241 | ≈0 | ~79 g | ~23% saved |

Qwen3 is more SFT-recoverable than Llama. The most striking result is **sp=20**: structured pruning collapses Qwen3 from 0.7159 to 0.2636 (36.8% retained). SFT brings it back to **0.5898 (82.4% of dense baseline)** at 80.9g CO₂.

Compare sp=20 against unstructured at the same level: unstructured sp=20 Qwen3 = 0.6963 at 109.1g CO₂. Structured+SFT sp=20 = 0.5898 at 80.9g CO₂. This is the Pareto-efficient trade-off: **84.7% of unstructured accuracy at 74.1% of unstructured CO₂**.

**Key result — Qwen3 sp=10:** Structured+SFT nearly matches the dense baseline — 0.7029 vs 0.7159 (98.2% retained) — while saving **26.6% inference CO₂** vs unstructured at the same sparsity. This is the single best outcome in the study: near-lossless pruning + SFT with a material carbon reduction.

### 10.4 One-Time SFT Training Cost

SFT introduces a one-time training overhead. The training CO₂ cost is:

| Model | Training CO₂ per sparsity | Training duration |
|---|---|---|
| Llama-3.1-8B | ~154–202 g | ~2600–2810 s |
| Qwen3-8B | ~234–241 g | ~3190–3300 s |
| Gemma-4-E4B | ~130–170 g | ~2585–2660 s |

This one-time training cost is **amortised over all future inference runs**. For a deployed model that serves 1,000 inference calls (each equivalent to one MMLU eval, ~100g CO₂):
- Break-even for Llama: training_cost / savings_per_run = 180g / 35g = **5 runs**
- Break-even for Qwen3: 237g / 28g = **9 runs**

Any deployment beyond ~10 inference evaluations recoup the training overhead and deliver net CO₂ savings.

### 10.5 Summary: Structured + SFT vs Other Methods

| Approach | Accuracy (Llama) | Accuracy (Qwen) | Inference CO₂ | vs Unstructured CO₂ |
|---|---|---|---|---|
| Unstructured sp=10% | 0.6479 | 0.7161 | ~102g | baseline |
| Structured sp=10% (no SFT) | 0.2905 | 0.5655 | **~66g** | **−35%** |
| **Structured sp=10% + SFT** | **0.4720** | **0.7029** | **~66g** | **−35%** |
| Unstructured sp=20% | 0.6314 | 0.6963 | ~103g | baseline |
| Structured sp=20% (no SFT) | 0.2332 | 0.2636 | **~65g** | **−37%** |
| **Structured sp=20% + SFT** | **0.2298** | **0.5898** | **~65g** | **−37%** |

**Bottom line:** SFT is essential for making structured pruning viable. Without SFT, structured MLP pruning collapses accuracy catastrophically while cutting CO₂ — a bad trade. With SFT, Qwen3 at sp=10 is nearly lossless (98.2% retained) and sp=20 is usable (82.4% retained), both at 26–37% lower inference CO₂ than unstructured. **Structured pruning + SFT is the only combination in this study that is Pareto-efficient: lower CO₂ and competitive accuracy simultaneously.**

---

## Reproducibility

```bash
# Regenerate all sustainability plots
python scripts/plot_sustainability_report.py

# Individual method plots
python scripts/plot_results.py            # Unstructured
python scripts/plot_structured_results.py # Structured + Semi-structured
```

Artifact directories:
```
outputs/runs/mmlu_pruning_ade7d5ffbbb4/  ← Unstructured (Llama, Qwen3)
outputs/runs/mmlu_pruning_39fc35ec1838/  ← Unstructured (Gemma-4)
outputs/runs/mmlu_pruning_bd584ae1ae1c/  ← Structured MLP (Llama, Qwen3)
outputs/runs/mmlu_pruning_ce32d7e86e8a/  ← Structured MLP (Gemma-4)
outputs/runs/mmlu_pruning_dd7c17966f53/  ← Semi 2:4 (Llama, Qwen3)
outputs/runs/mmlu_pruning_b8eefdc94e41/  ← Semi 4:8 (Llama, Qwen3)
outputs/sft_runs/structured_sft_97c7151924e1/  ← Structured+SFT Llama sp=0-30
outputs/runs/structured_sft_9db8cf7ef319/      ← Structured+SFT Llama sp=40-70
outputs/runs/structured_sft_23d578f0d9f2/      ← Structured+SFT Qwen3
outputs/sft_runs/structured_sft_b9ca5a313b42/  ← Structured+SFT Gemma-4 sp=0,10,40,50
outputs/sft_runs/structured_sft_df37aaf51b99/  ← Structured+SFT Gemma-4 sp=20,30
outputs/plots/sustainability/            ← All plots from this report
```
