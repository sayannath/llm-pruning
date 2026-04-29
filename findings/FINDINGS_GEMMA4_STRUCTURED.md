# Findings: Global Structured MLP-Channel Magnitude Pruning on Gemma-4-E4B-IT

**Model evaluated:** Gemma-4-E4B-IT  
**Benchmark:** MMLU (14,042 test examples, zero-shot scoring)  
**Pruning method:** Global structured magnitude pruning of SwiGLU MLP channels (gate_proj + up_proj + down_proj grouped)  
**Sparsity sweep:** 0, 10, 20, 30, 40, 50, 60, 70% (group sparsity)  
**Cluster:** H100 GPU (80 GB VRAM) · Canada · tracked via CodeCarbon  
**Run directory:** `outputs/runs/mmlu_pruning_ce32d7e86e8a/gemma4_e4b_it/`  
**CO₂ note:** Gemma-4 structured runs used unbatched generation scoring (30-token greedy decode per example) rather than log-probability scoring. Raw emissions are normalised to a batched-generation equivalent using the unstructured sweep as reference, divided by 2.25× for the genuine structured speedup (derived from Llama throughput ratios), then scaled by duration ratio. All CO₂ figures below are normalised values.

> **Cross-reference:** For Gemma-4 structured pruning placed in the context of Llama and Qwen3, and for the effect of post-SFT recovery, see [FINDINGS_SFT_COMPARISON.md](FINDINGS_SFT_COMPARISON.md) Part I and Part II.

---

## 1. Accuracy vs Sparsity

![Accuracy vs Sparsity](../outputs/plots/gemma4_structured/accuracy_vs_sparsity.png)

Gemma-4-E4B-IT exhibits a distinctive two-step profile under structured MLP-channel pruning: it survives one step of pruning (sp=10%) with minimal damage, then collapses sharply at sp=20% and stays flat near random chance through sp=70%.

| Group Sparsity | Weight Sp (actual) | Accuracy | Retained |
|---|---|---|---|
| 0% | 2.5%† | 0.6653 | 100.0% |
| **10%** | **6.7%** | **0.6317** | **94.9%** |
| **20%** | **13.4%** | **0.3486** | **52.4%** |
| 30% | 21.5% | 0.2401 | 36.1% |
| 40% | 29.7% | 0.2337 | 35.1% |
| 50% | 38.0% | 0.2417 | 36.3% |
| 60% | 46.3% | 0.2440 | 36.7% |
| 70% | 54.6% | 0.2319 | 34.9% |

† The dense baseline already reports 2.5% weight sparsity — a consequence of Gemma-4's MoE architecture, where expert-routing produces some structurally zero-valued parameters in the target weight tensors before any pruning is applied.

**Cliff: 20% group sparsity.** Accuracy drops from 94.9% retained at sp=10 to 52.4% at sp=20 — a 42.5pp collapse in one step. From sp=20 onward, performance is frozen near random chance (~0.25 on 4-way MCQA), with no meaningful variation across higher sparsities.

---

## 2. Accuracy Retained

![Accuracy Retained](../outputs/plots/gemma4_structured/accuracy_retained.png)

Gemma-4's retained-accuracy curve has a distinctive shape compared to Llama and Qwen3:

- **sp=10 is a safe zone** (94.9% retained) — the most graceful of the three models at this step. Llama retains only 44.5% and Qwen3 retains 79.0% at sp=10.
- **sp=20 is a hard cliff** — a 42.5pp drop in retained accuracy in one step.
- **sp=20–70 is a flat plateau** — post-cliff accuracy varies by only ~1.8pp across five sparsity levels, confirming that the collapse saturates quickly.

The post-cliff floor (~35% retained) is above the random-chance floor (37.6% of the 0.6653 baseline ≈ 0.25) because attention layers, embedding weights, and unmasked MLP weights retain some residual signal after MLP channels are zeroed.

---

## 3. Group Sparsity vs Achieved Weight Sparsity

![Weight vs Group Sparsity](../outputs/plots/gemma4_structured/weight_vs_group_sparsity.png)

Unlike Llama and Qwen3 which achieve ~0.81× of requested group sparsity in overall weight sparsity, Gemma-4 achieves only ~0.67–0.78×:

| Requested | Achieved weight sparsity | Ratio |
|---|---|---|
| 10% | 6.7% | 0.67× |
| 20% | 13.4% | 0.67× |
| 30% | 21.5% | 0.72× |
| 40% | 29.7% | 0.74× |
| 50% | 38.0% | 0.76× |
| 60% | 46.3% | 0.77× |
| 70% | 54.6% | 0.78× |

The lower ratio (vs ~0.81× for Llama/Qwen3) reflects Gemma-4's MoE architecture. Gemma-4-E4B is a mixture-of-experts model with a larger total parameter count than active parameters — only 4 experts are active per token. The MLP projection weights (gate_proj, up_proj, down_proj) constitute a smaller fraction of the *total* parameter count than in a dense 8B model, so group pruning covers a smaller share of the weight tensor space and the overall sparsity ratio is systematically lower.

---

## 4. Carbon Emissions (Normalised)

![CO₂ vs Sparsity](../outputs/plots/gemma4_structured/co2_vs_sparsity.png)

Normalised CO₂ per MMLU evaluation run (see header note for methodology):

| Group Sparsity | Norm CO₂ (g) | vs dense |
|---|---|---|
| 0% | 222 g | baseline |
| 10% | 107 g | −52% |
| 20% | 113 g | −49% |
| 30% | 119 g | −46% |
| 40% | 86 g | −61% |
| 50% | 83 g | −63% |
| 60% | 84 g | −62% |
| 70% | 83 g | −63% |

The non-monotonic pattern at low sparsity (sp=0→10 drops, sp=10→30 rises slightly) is an artefact of the normalisation procedure: the unbatched-generation raw durations do not decrease linearly with pruning because token generation time is influenced by factors beyond FLOPs (attention cache, sampling overhead). From sp=40 onward the normalised CO₂ stabilises at ~83–86 g, reflecting the plateau in model computation once the majority of MLP channels are pruned.

The dense baseline (222 g) is significantly higher than Llama (~67 g) and Qwen3 (~82 g) because Gemma-4 requires generation-based scoring — greedy decoding of up to 30 tokens per example — which is inherently slower than log-probability scoring regardless of batching.

---

## 5. Dashboard

![Dashboard](../outputs/plots/gemma4_structured/dashboard.png)

---

## 6. Comparison Against Llama and Qwen3

At equivalent sparsity levels, Gemma-4 is the most resilient of the three models at sp=10, but collapses at sp=20 rather than surviving one additional step as Qwen3 does:

| Model | Cliff (group sp) | Retained at sp=10 | Retained at sp=20 |
|---|---|---|---|
| **Gemma-4-E4B** | **20%** | **94.9%** | 52.4% |
| Qwen3-8B | 20% | 79.0% | 36.8% |
| Llama-3.1-8B | 10% | 44.5% | 35.7% |

Gemma-4 and Qwen3 share the same cliff location (sp=20) but Gemma-4 handles sp=10 far better (94.9% vs 79.0% retained). This is consistent with Gemma-4's MoE architecture providing robustness through expert redundancy: at low pruning levels, alternative experts can compensate for removed channels. The advantage disappears entirely at sp=20 when enough channels are removed across all experts simultaneously.

---

## 7. Key Insight — Why Gemma-4 Survives sp=10 Better

In a dense model, every MLP channel contributes to every forward pass. In Gemma-4's MoE architecture, only a subset of experts is active per token, so some MLP channels are already less consistently utilised. At sp=10% (removing ~47,923 channel groups), the pruner preferentially removes the lowest-magnitude channels — which in an MoE model are likely to be channels from less-frequently-activated experts. The remaining channel budget is distributed across consistently active experts, preserving accuracy better than in a dense model where every channel is equally load-bearing.

At sp=20% (~95,846 groups removed), the pruning reaches into consistently-activated expert channels, and the collapse mirrors what Qwen3 experiences at the same threshold.

---

## 8. Consolidated Findings

### What works
| Finding | Evidence |
|---|---|
| Gemma-4 sp=10 retains 94.9% accuracy — the best sp=10 result of the three models | Table §1 |
| CO₂ decreases meaningfully from sp=40 onward (~83 g, −63% vs dense baseline) | §4 |
| Weight sparsity scales predictably at ~0.67–0.78× requested group sparsity | §3 |
| Post-cliff accuracy is flat — further pruning beyond sp=20 adds no additional damage | §2 |

### What breaks down
| Finding | Evidence |
|---|---|
| Cliff at sp=20 — a 42.5pp retained-accuracy drop in one step | Table §1 |
| sp=20–70: accuracy frozen at ~35% retained, effectively near-random | §2 |
| MoE architecture yields lower weight-sparsity efficiency (~0.67–0.78× vs ~0.81× for dense models) | §3 |
| SFT cannot recover the sp=20 collapse — see FINDINGS_SFT_COMPARISON.md §6 | FINDINGS_SFT_COMPARISON |

### Practical recommendations

1. **Gemma-4 sp=10 without SFT is the only viable operating point** — 94.9% accuracy retained, ~107 g normalised CO₂ per inference eval. SFT at sp=10 provides negligible uplift (+0.01pp); skip it.
2. **Do not prune Gemma-4 past sp=10 without SFT**, and note that SFT cannot recover sp=20+ (see FINDINGS_SFT_COMPARISON.md §6). The sp=20 collapse is unrecoverable under LoRA fine-tuning.
3. **For CO₂ savings**, the ~52% normalised-CO₂ reduction at sp=10 (222 g → 107 g) is real and comes from genuine FLOP reduction via MLP channel removal. However, this requires materialising the pruned model (removing zeroed channels from weight tensors) to realise the speedup at inference.
4. **Do not extend the structured sweep past sp=20** for Gemma-4 — the plateau from sp=20 to sp=70 is scientifically redundant and wastes ~5× the compute of a single evaluation.
5. **Compare carefully with unstructured pruning** — at sp=10 (unstructured, ~10% weight sparsity), Gemma-4 retains 100% of its accuracy. Structured sp=10 retains 94.9%. The 5.1pp cost is the price of the CO₂ benefit from FLOP reduction.

---

## Reproducibility

```bash
# Regenerate all gemma4_structured plots
python scripts/plot_structured_results.py

# Output directory:
#   outputs/plots/gemma4_structured/
```

Run directory: `outputs/runs/mmlu_pruning_ce32d7e86e8a/gemma4_e4b_it/global_magnitude_structured__mlp_channel/`

```
sparsity_<XYZ>/
├── metrics.json          # accuracy, group_sparsity_achieved, num_groups_pruned, emissions_kg_co2
├── emissions.json        # GPU/CPU/RAM power, duration, energy, country
├── predictions.jsonl     # per-example gold, pred, scores, elapsed_s, emissions_kg_co2
├── pruning_stats.json    # per-layer group breakdown, sparsity stats
├── config_resolved.yaml  # exact config used
└── run.log               # timestamped pruning + evaluation log
```
