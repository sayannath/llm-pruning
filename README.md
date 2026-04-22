# LLM Pruning MMLU

Reproducible pruning sweeps for causal LMs evaluated with zero-shot MMLU multiple-choice accuracy. Supports three pruning strategies — global unstructured magnitude, structured MLP-channel, and semi-structured N:M — with carbon emissions tracking via CodeCarbon.

**Models:** Llama-3.1-8B-Instruct · Qwen3-8B · Gemma-4-E4B-IT  
**Benchmark:** MMLU (14,042 test examples, zero-shot, choice log-probability scoring)  
**Cluster:** H100 80 GB · Canada

---

## Quick Results

| Method | Llama acc @ 50% sp | Qwen acc @ 50% sp | CO₂/run | GPU accel |
|---|---|---|---|---|
| Unstructured | 34.2% | 27.2% | ~98 g | None |
| Structured MLP | 23.1% | 24.4% | ~71 g | 1.5–3× (materialised) |
| Semi 2:4 | 28.5% | 24.3% | ~88 g | **2× (cuSPARSELt)** |
| **Semi 4:8** | **34.6%** | 25.5% | ~91 g | ~1.4× (custom kernel) |

Dense baselines: Llama 65.3% · Qwen 71.6%. See [FINDINGS_UNSTRUCTURED.md](findings/FINDINGS_UNSTRUCTURED.md), [FINDINGS_STRUCTURED.md](findings/FINDINGS_STRUCTURED.md), [FINDINGS_SEMI_STRUCTURED.md](findings/FINDINGS_SEMI_STRUCTURED.md), [FINDINGS_SUSTAINABILITY.md](findings/FINDINGS_SUSTAINABILITY.md).

---

## Setup

```bash
pip install -r requirements-dev.txt
```

Llama-3.1 requires accepting the Hugging Face licence and setting `HF_TOKEN`:

```bash
export HF_TOKEN="hf_..."
```

## Validate

```bash
python scripts/validate_env.py
pytest -q
```

---

## Pruning Methods

### 1. Global Unstructured Magnitude Pruning

Zeroes the globally lowest-magnitude individual weights across all `nn.Linear` layers. Preserves tensor shapes. No hardware speedup without custom sparse kernels.

```bash
# Single sparsity point
python scripts/evaluate_model.py \
  --config configs/experiments/llama31_mmlu_sweep.yaml \
  --sparsity 50 --max-samples 128

# Full sweep (0–99%)
python scripts/run_sweep.py \
  --config configs/experiments/all_models_mmlu_sweep.yaml
```

SLURM:
```bash
sbatch slurm/run_mmlu_pruning_sweep.slurm
```

### 2. Structured MLP-Channel Pruning

Zeroes entire SwiGLU intermediate channels globally — each pruned channel zeros `gate_proj[j,:]`, `up_proj[j,:]`, and `down_proj[:,j]` together. Preserves tensor shapes (masked, not materialised). Materialising pruned channels enables real 1.5–3× inference speedup on standard dense hardware.

```bash
# Full sweep (0–70% group sparsity)
python scripts/run_sweep.py \
  --config configs/experiments/all_models_mmlu_structured_mlp_sweep.yaml

# Single model
python scripts/run_sweep.py \
  --config configs/experiments/llama31_mmlu_structured_mlp_sweep.yaml
```

SLURM:
```bash
sbatch slurm/run_mmlu_structured_sweep.slurm

# Override config or cap samples
sbatch --export=ALL,CONFIG=configs/experiments/qwen3_mmlu_structured_mlp_sweep.yaml \
  slurm/run_mmlu_structured_sweep.slurm
sbatch --export=ALL,MAX_SAMPLES=128,FAIL_FAST=1 \
  slurm/run_mmlu_structured_sweep.slurm
```

### 3. Semi-Structured N:M Magnitude Pruning

Enforces exactly N zeros in every block of M consecutive weights per row. Two patterns supported:

| Pattern | Config key | Sparsity | Hardware |
|---|---|---|---|
| 2:4 | `nm_2_4` | 50% | NVIDIA A100/H100 via cuSPARSELt (2× throughput) |
| 4:8 | `nm_4_8` | 50% | Custom CUDA kernel required (~1.4×) |

```bash
# 2:4 sweep (both models)
python scripts/run_sweep.py \
  --config configs/experiments/all_models_mmlu_semi_structured_2_4_sweep.yaml

# 4:8 sweep (both models)
python scripts/run_sweep.py \
  --config configs/experiments/all_models_mmlu_semi_structured_4_8_sweep.yaml
```

SLURM:
```bash
sbatch slurm/run_mmlu_semi_structured_sweep.slurm          # 2:4 by default
sbatch slurm/run_mmlu_semi_structured_full_sweep.slurm     # both 2:4 and 4:8
```

---

## Output Structure

Each sweep writes under `outputs/runs/<run_id>/`. The directory layout differs by pruning method:

```
# Unstructured
outputs/runs/<run_id>/
└── <model_name>/
    └── sparsity_<XYZ>/
        ├── metrics.json
        ├── emissions.json
        ├── predictions.jsonl
        ├── pruning_stats.json
        ├── config_resolved.yaml
        └── run.log

# Structured MLP
outputs/runs/<run_id>/
└── <model_name>/
    └── global_magnitude_structured__mlp_channel/
        └── sparsity_<XYZ>/
            └── ...

# Semi-structured 2:4 / 4:8
outputs/runs/<run_id>/
└── <model_name>/
    └── global_magnitude_semi_structured__2_4/   (or __4_8)
        └── sparsity_<XYZ>/
            └── ...
```

Each `sparsity_*/` directory contains:

| File | Contents |
|---|---|
| `metrics.json` | accuracy, sparsity achieved, pruning stats, emissions |
| `emissions.json` | GPU/CPU/RAM power (W), duration (s), energy (kWh), CO₂ (kg) |
| `predictions.jsonl` | per-example gold label, prediction, logprobs, elapsed, CO₂ |
| `pruning_stats.json` | per-layer sparsity breakdown; groups/blocks for structured methods |
| `config_resolved.yaml` | fully merged config (after inheritance) |
| `run.log` | timestamped pruning and evaluation log |

A combined `combined_results.csv` and `combined_results.jsonl` are written at run root after the sweep completes, aggregating all models and sparsities.

---

## Configuration

Configs use YAML inheritance (`inherits:` key). The hierarchy is:

```
configs/base.yaml
  └── configs/datasets/mmlu.yaml
        └── configs/pruning/global_<method>.yaml
              └── configs/models/<model>.yaml
                    └── configs/experiments/<experiment>.yaml  ← entry point
```

### Pruning config fields (`PruningConfig`)

| Field | Type | Description |
|---|---|---|
| `method` | str | `global_magnitude_unstructured` · `global_magnitude_structured` · `global_magnitude_semi_structured` |
| `structure` | str \| None | `mlp_channel` for structured; `nm_2_4` / `nm_4_8` for semi-structured |
| `score` | str | Scoring function — `l2_norm` (default) |
| `scope` | str | `global` (default) · `local_block` (N:M only) |
| `nm_n` | int \| None | N in N:M pattern (required for semi-structured) |
| `nm_m` | int \| None | M in N:M pattern (required for semi-structured) |
| `block_dim` | int | Row dimension for N:M blocks (default: 1) |
| `sparsities` | list[float] | Sparsity sweep points in %; semi-structured accepts only `[0, 50]` |
| `exclude_module_name_patterns` | list[str] | Module name substrings to skip (e.g. `lm_head`) |

Example pruning config for structured MLP (`configs/pruning/global_structured_mlp.yaml`):

```yaml
pruning:
  method: global_magnitude_structured
  structure: mlp_channel
  score: l2_norm
  scope: global
  sparsities: [0, 10, 20, 30, 40, 50, 60, 70]
  exclude_module_name_patterns: [lm_head]
```

Example for semi-structured 2:4 (`configs/pruning/global_semi_structured_2_4.yaml`):

```yaml
pruning:
  method: global_magnitude_semi_structured
  structure: nm_2_4
  nm_n: 2
  nm_m: 4
  block_dim: 1
  sparsities: [0, 50]
  exclude_module_name_patterns: [lm_head]
```

---

## Plotting

```bash
# Unstructured results
python scripts/plot_results.py \
  --run-dir outputs/runs/mmlu_pruning_ade7d5ffbbb4 \
  --out-dir outputs/plots

# Structured + semi-structured results
python scripts/plot_structured_results.py

# Cross-method sustainability and GPU acceleration report
python scripts/plot_sustainability_report.py
```

Plots are saved to `outputs/plots/`, `outputs/plots/structured/`, `outputs/plots/semi_structured/`, and `outputs/plots/sustainability/`.

---

## Codebase Layout

```
src/llm_pruning_mmlu/
├── config.py                   # PruningConfig, ExperimentConfig, config loading + validation
├── pruning/
│   ├── dispatch.py             # prune_model() — routes to the correct pruner by method
│   ├── magnitude.py            # global unstructured magnitude pruning
│   ├── structured.py           # structured MLP-channel scoring and mask building
│   ├── structured_targets.py   # StructuredGroup / TensorSlice, discover_mlp_channel_groups()
│   ├── semi_structured.py      # N:M block pruning
│   ├── targets.py              # PruningParameter, find_pruning_parameters()
│   ├── masks.py                # apply_masks()
│   ├── apply.py                # weight zeroing
│   └── stats.py                # per-layer sparsity statistics
├── experiments/
│   ├── sweep.py                # run_sweep() — main experiment loop
│   ├── resume.py               # sparsity_dir(), should_skip() with pruning_tag support
│   └── artifacts.py            # save_run_artifacts()
├── evaluation/
│   ├── runner.py               # evaluate_examples()
│   ├── scorer.py               # choice log-probability scoring
│   ├── metrics.py              # accuracy, per-subject breakdown
│   └── predictions.py          # prediction serialisation
├── models/
│   ├── loader.py               # load_model_and_tokenizer(), DummyMlpCausalLM (tests)
│   └── tokenizer.py
├── data/
│   ├── mmlu.py                 # MMLU loading (HuggingFace or local fixture)
│   └── prompting.py
├── reporting/
│   ├── tables.py               # write_combined_results() — recursive glob across all methods
│   └── plots.py                # per-run sparsity-vs-accuracy plot
└── utils/
    ├── emissions.py            # EmissionsTracker (CodeCarbon wrapper)
    ├── seed.py
    ├── hashing.py
    ├── io.py
    └── logging_utils.py

scripts/
├── run_sweep.py                # CLI entry point for any sweep
├── evaluate_model.py           # single sparsity point
├── plot_results.py             # unstructured plots
├── plot_structured_results.py  # structured + semi-structured plots
├── plot_sustainability_report.py  # cross-method sustainability + GPU acceleration
├── merge_sweep_results.py      # merge multiple run CSVs
├── export_summary.py
├── smoke_test.py
└── validate_env.py

configs/
├── base.yaml
├── datasets/mmlu.yaml
├── models/llama31_8b_instruct.yaml · qwen3_8b.yaml · gemma4_e4b_it.yaml
├── pruning/
│   ├── global_unstructured.yaml
│   ├── global_structured_mlp.yaml
│   ├── global_semi_structured_2_4.yaml
│   └── global_semi_structured_4_8.yaml
└── experiments/
    ├── all_models_mmlu_sweep.yaml
    ├── all_models_mmlu_structured_mlp_sweep.yaml
    ├── all_models_mmlu_semi_structured_2_4_sweep.yaml
    ├── all_models_mmlu_semi_structured_4_8_sweep.yaml
    └── llama31_*/qwen3_*/gemma4_e4b_* variants for each method

slurm/
├── run_gemma4_unstructured_sweep.slurm         # unstructured, Gemma-4-E4B-IT (1-day wall-time)
├── run_gemma4_structured_sweep.slurm           # structured MLP, Gemma-4-E4B-IT (12-h wall-time)
├── run_gemma4_semi_structured_sweep.slurm      # semi-structured 2:4+4:8 job array, Gemma-4-E4B-IT
├── run_mmlu_pruning_sweep.slurm                # unstructured, Llama+Qwen
├── run_mmlu_structured_sweep.slurm             # structured MLP, Llama+Qwen
├── run_mmlu_semi_structured_sweep.slurm        # semi-structured 2:4, Llama+Qwen
└── run_mmlu_semi_structured_full_sweep.slurm   # semi-structured 2:4+4:8, Llama+Qwen

tests/
├── unit/
│   ├── test_magnitude_pruning.py
│   ├── test_structured_mlp_pruning.py      # group discovery, mask correctness
│   ├── test_structured_stats.py
│   ├── test_semi_structured_pruning.py     # N:M block masking
│   ├── test_semi_structured_stats.py
│   ├── test_pruning_dispatch.py            # all three methods via prune_model()
│   ├── test_pruning_targets.py
│   ├── test_pruning_stats.py
│   ├── test_config.py
│   ├── test_resume.py
│   ├── test_metrics.py
│   ├── test_prompt_formatting.py
│   └── test_io.py
└── integration/
    ├── test_end_to_end_dummy_sweep.py       # unstructured end-to-end
    ├── test_end_to_end_structured_sweep.py  # structured end-to-end + filesystem isolation
    ├── test_artifact_generation.py
    └── test_smoke_eval_tiny_model.py
```

---

## Findings

| Report | Description |
|---|---|
| [FINDINGS_UNSTRUCTURED.md](findings/FINDINGS_UNSTRUCTURED.md) | Global unstructured magnitude pruning — accuracy cliff at 50%, carbon analysis |
| [FINDINGS_STRUCTURED.md](findings/FINDINGS_STRUCTURED.md) | Structured MLP-channel pruning — cliff at 10% (Llama) / 20% (Qwen), group vs weight sparsity gap |
| [FINDINGS_SEMI_STRUCTURED.md](findings/FINDINGS_SEMI_STRUCTURED.md) | Semi-structured 2:4 and 4:8 pruning — 4:8 matches unstructured on Llama, production deployment paths |
| [FINDINGS_SUSTAINABILITY.md](findings/FINDINGS_SUSTAINABILITY.md) | Cross-method sustainability and GPU acceleration — carbon frontier, theoretical speedups, deployment recommendations |

---

## Reproducibility

All experiments used seed 42. The run ID (directory name suffix) is a hash of the fully resolved config, so re-running the same config always writes to the same directory. Existing completed sparsity points are skipped by default (`resume: true`).

To force a clean re-run:

```bash
python scripts/run_sweep.py \
  --config configs/experiments/all_models_mmlu_structured_mlp_sweep.yaml \
  --no-resume
```

## TODO experiments after semi-stuctured (magnitude) completes

```bash
sbatch slurm/sparse_sft/semi_structured/run_llama31_wanda_2_4_sft.slurm
sbatch slurm/sparse_sft/semi_structured/run_qwen3_wanda_2_4_sft.slurm
sbatch slurm/sparse_sft/semi_structured/run_gemma4_wanda_2_4_sft.slurm
```